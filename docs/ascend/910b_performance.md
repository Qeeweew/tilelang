# TileLang 910B 性能：GEMM 与 Attention

整理日期：2026-10-10；实现检查点：`6a3e9183`。本次为已有有效采集的汇总和指标
重算，没有重新测速。910B3 上，已调优的 dense MHA 完整手排接近或略快于对应
Torch NPU 控制；同算法自动排步仍有差距。GEMM 既有方阵结果约落后 3%；
causal GQA 和 decode 的最近保留采集更早，尚不能宣称已获得 dense MHA 的收益。

## 环境、版本与计时边界

设备为 Ascend 910B3，20 AIC / 40 AIV，采集频率 1800 MHz；CANN 9.1、
Torch 2.12.0+cpu、torch_npu 2.12.0.post2，目标 `dav-2201`。
每行 TileLang/Torch 来自同卡同组，采用 `msprof op` 默认 warm-up/replay 的
`OpBasicInfo.Task Duration(us)`。这是 kernel elapsed，不包含编译、Python 调用、
workspace 分配或主机检查。默认 replay 不是独立统计样本；无置信区间，几个百分点
的差异不视作统计显著。不同表的 Torch 时间可能不同，不跨采集相减计算优化收益。

| 证据组 | 源码/有效性 | 本文用途 |
| --- | --- | --- |
| G：GEMM checkpoint | 历史采集及 `e3f62c7e` 检查点的产物一致性核对 | GEMM 最近保留基线，未重验至 `6a3e9183` |
| A：FA current | `e3f62c7e`；25 个程序通过离线/设备正确性 | causal GQA、decode 最近保留基线 |
| B：balanced matrix | `182d8150`；六形状，同算法和普通流水候选均设备通过 | D128 推荐手排、自动排步差距 |
| C：D64 retune final | 同一 `282d2947…` 安装库，D64 专用策略 | D64 推荐手排；三形状与控制输出逐位一致 |
| R：sync reuse refactor | `6a3e9183`，安装库 `97cad0c5…` | 515 项 CPU 回归；8K D64/D128 手排 ASC 和机器码与 B/C 同策略基线完全一致 |

R 只把两份 8K 完整手排的设备证据关联到当前重构；没有证明所有历史 GEMM、GQA、
decode 或自动排步产物均与当前提交相同。表中历史行是覆盖缺口的明确记录。

Torch 基线按任务区分：GEMM 使用 Torch NPU matmul；prefill 使用
`torch_npu.npu_fusion_attention`（FAS / FlashAttentionScore）；decode 使用
`npu_fused_infer_attention_score`（FIA，关闭 LSE）。不要把所有 attention 基线都称作 FIA。
FAS 还产生 softmax 辅助输出，TileLang 路径只交付 O，输出工作量并非完全一致。
所有 attention 输入为 BNSD，scale=`D**-0.5`，无 dropout；同组匹配 mask、dtype
和可见 KV。FP16/BF16 路径有 FP32 累加，但不保证与 Torch 内部舍入顺序一致。

## MFU 与带宽的口径

本文的 MFU 是**算子有效数学 FLOPs / kernel 时间 / 全芯片理论 Cube 算力**，
不是整模型训练 MFU，也不是 PMU 的 Cube active ratio。一个 FMA 计两个 FLOPs。
FP16/BF16→FP32 采用 dense Cube 参考吞吐 8192 FLOPs/AIC/cycle：

```text
P_ref = 20 × 8192 × 1.8e9 = 294.912e12 FLOP/s
TFLOP/s = effective_flops / (elapsed_us × 1e6)
MFU = TFLOP/s / 294.912
GEMM FLOPs = 2 M N K
Dense attention FLOPs = 4 B Hq Sq Sk D
Causal self-attention FLOPs = 4 B Hq D × S(S+1)/2
```

峰值口径来源：CANN 9.1 `data/platform_config/Ascend910B3.ini` 的
`cube_core_cnt=20`、`cube_freq=1800`；msopprof 固定版本
[`56632dc4` 的 pmu_calculate.h](https://github.com/Ascend/msopprof/blob/56632dc4a944410c791538451dc2156a151bd105/csrc/op_profiling/profiling/device/data_parse/pmu_calculate.h)
定义 `CUBE_FOPS_FP16_NUM=8192`。这是按采集频率归一的理论参考，不是带宽实测，
也不使用稀疏算力或其他 910B SKU 的标称值。实际频率变化时应重新计算。

Attention 只计 QK/PV 的有效 GEMM 工作，softmax、exp、归约、rescale、mask、
DMA、同步均消耗时间但不加到分子；padding、causal 对角 tile 的无效计算也不计。
GQA 的计算量乘 Hq，KV 存储量乘 Hkv。不同 kernel 的额外工作不同，MFU 不能独自
证明瓶颈；D64 较低的 MFU 不等于调度器浪费了相同比例的时间。

## GEMM：最近保留基线 G

三个配置均为 tile M/N/K=128/256/256，自动 K 流水。前两个为 persistent20；
矩形使用默认 grid，不能把方阵结果当作默认 grid 的表现。dtype 以实际输入记录为准，
不从 Torch kernel 名中的 FP16 字样反推 BF16 输入类型。

| M/N/K | dtype | 遍历 | TL µs | Torch µs | TL/Torch ↓ | TL TFLOP/s | MFU TL / Torch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2048/2048/2048 | FP16 | persistent20 | 83.02 | 80.68 | 1.029 | 206.94 | 70.2% / 72.2% |
| 4096/4096/4096 | BF16 | persistent20 | 543.14 | 528.16 | 1.028 | 253.05 | 85.8% / 88.2% |
| 512/4096/2048 | BF16 | 默认 grid | 54.68 | 56.04 | 0.976 | 157.09 | 53.3% / 52.0% |

方阵接近 Torch，较小 M 的矩形有效 MFU 更低。此表不足以覆盖 Split-K、尾块、
转置组合或小矩阵，也不代表这些配置已在最新库上重新验收。

## Dense MHA：推荐的显式完整手排策略

Sq=Sk=S，Hq=Hkv=H，groups20、BQ128/BK1024；嵌套 PV 仍使用自动流水。
D128 与 D64 分开选参数，不能复制一套缓存策略覆盖所有 D。

| B/H/S/D | dtype | TL µs | Torch FAS µs | TL/Torch ↓ | TL TFLOP/s | MFU TL / Torch | 证据 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1/32/2048/128 | BF16 | 480.14 | 497.06 | 0.966 | 143.12 | 48.5% / 46.9% | B |
| 1/32/4096/128 | BF16 | 1811.36 | 1863.48 | 0.972 | 151.75 | 51.5% / 50.0% | B |
| 1/32/8192/128 | BF16 | 7208.82 | 7378.02 | 0.977 | 152.52 | 51.7% / 50.5% | B；R 同策略产物核对 |
| 1/4/4096/64 | FP16 | 248.24 | 274.64 | 0.904 | 69.21 | 23.5% / 21.2% | C |
| 4/16/2048/64 | FP16 | 794.14 | 819.86 | 0.969 | 86.53 | 29.3% / 28.4% | C |
| 1/32/8192/64 | BF16 | 6096.26 | 6645.64 | 0.917 | 90.18 | 30.6% / 28.1% | C；R 产物核对 |

| 策略 | D128 | D64 |
| --- | --- | --- |
| QK/PV chunk | 128 / 128 | 256 / 256 |
| PV rows | 128 | 64 |
| Vec1 / Vec2 rows | 8 / 64 | 8 / 128 |
| `cache_pv`, `separate_score_input`, `inplace_final_division` | true | true |
| `reuse_qk_b`, `stream_qk_b`, `stream_pv_p` | true | false |
| workspace payload，20 groups | 70 MiB | 65 MiB |

参数与限制的权威入口是 [FA 使用契约](../../examples/ascend/flash_attention/910b.md)。
推荐策略是显式选择，factory 默认仍是 Stage-only，不能把默认调用等同于表中完整手排。
`Sq` 整除256、`Sk` 整除1024；本表不覆盖 GQA、causal、tails、dropout 或 backward。

收益来自多个已分开验证的因素：均衡20组让8K的1024 query pairs分配为4×52+16×51，
避免16组每组64 pairs；D64保留完整K/P operand panels并调整native tile；更大的
逻辑BK减少状态更新与GM交接。改变BK也改变P的舍入边界，属于算法策略比较。
降低MTE2请求、增加MTE1请求和更短kernel可以同时发生，不能单看请求数判定关键路径。

## 自动流水：同组证据 B

“同算法自动”保留 paired-query 算法、BK1024、缓存和P舍入点，交给编译器选择
outer排步（移除显式Stage和固定顺序，自动版本上限4）；它与简单地把完整手排开关关闭得到的 Stage-only 不同。
“普通流水”使用普通FA表达和 `T.Pipelined`，只比较BK64/128/256三个有限候选，
表中最优均BK256，不能称作全局最优。两类自动化都不等于已经调优过的D64策略C。

| B/H/S/D | 同算法自动 µs | 普通流水最佳 µs | 同组 Torch µs | 自动/Torch | 普通/Torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1/32/2048/128 | 621.80 | 1497.58 | 497.06 | 1.251 | 3.013 |
| 1/32/4096/128 | 2440.42 | 5726.30 | 1863.48 | 1.310 | 3.073 |
| 1/32/8192/128 | 10282.26 | 22856.80 | 7378.02 | 1.394 | 3.098 |
| 1/32/8192/64 | 9549.44 | 23469.70 | 6642.58 | 1.438 | 3.533 |
| 4/16/2048/64 | 1195.86 | 3085.46 | 819.50 | 1.459 | 3.765 |
| 1/4/4096/64 | 366.32 | 756.78 | 274.88 | 1.333 | 2.753 |

dtype同上表。与B组同参数完整手排相比，同算法自动慢29.5%–42.6%；B组D64手排
仍使用D128派生策略，不能拿C组结果跨组计算这个差距。优先调查公共调度的phase
顺序、等待和重叠选择。[service weights](2201_service_weights.md) 是无量纲排序
启发式，GM/FIX/Vector等路径并没有完整的实测延迟模型，合法计划不保证性能最优。

## Causal GQA：最近保留基线 A

B1/Hq32/Hkv8/D128/BF16，Sq=Sk=S，query_start=0；Torch FAS sparse_mode=2。
普通自动策略BQ64/BK256、stage上限4，未做本轮调优，persistent手排不支持此契约。

| S | generic µs | 普通自动 µs | Torch µs | 自动/Torch | 自动 TFLOP/s | 有效 MFU 自动 / Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 4669.72 | 4538.68 | 1162.66 | 3.904 | 30.29 | 10.3% / 40.1% |
| 8192 | 16697.02 | 16222.08 | 4158.80 | 3.901 | 33.89 | 11.5% / 44.8% |

分子按有效三角形工作计算，不按完整S²计算；未声称实际kernel恰好跳过所有无效
位置。约3.9倍时延差距仍是待重新验证与优化的证据，dense MHA收益不能外推到这里。

## GQA decode：有效带宽与 MBU 的边界

Hq32/Hkv8/D128/BF16，Sq=1，访问全部KV，普通策略BQ16/BK256、stage上限4。
有效数据量按 Q/K/V 各读一次、O写一次计算，GQA共享KV不乘查询头数：

```text
bytes_min = element_bytes × B × (2 Hq Sq D + 2 Hkv Sk D)
effective_GB/s = bytes_min / (elapsed_us × 1000)
MBU_HBM = measured_HBM_bytes / elapsed_seconds / sustained_HBM_bytes_per_second
```

| B/Sk | 有效字节数 | generic µs | 自动 µs | Torch FIA µs | 自动/Torch | 有效 GB/s generic / 自动 / Torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/8192 | 33,570,816 | 96.90 | 83.76 | 65.26 | 1.283 | 346.45 / 400.80 / 514.42 |
| 4/32768 | 536,936,448 | 740.98 | 772.14 | 421.02 | 1.834 | 724.63 / 695.39 / 1275.32 |

本表为A组历史采集；32K自动还慢于generic，保留该结果。decode算术强度低，
带宽和固定开销通常比MFU更有解释力，但这些记录仅含BasicInfo，不能据此确诊
HBM已经饱和。910B3配置有192MiB L2，8K例的约32MiB有效数据可容纳于L2；
默认replay不是cold-HBM测试。KV重复读取、workspace和实际cache miss也未计入
`bytes_min`，所以本表**不提供伪精确的HBM MBU百分比**。

补齐MBU需要同一采集中分离L2与HBM实际字节、确定cache条件，并在同设备/频率
下标定可持续HBM带宽；用理论峰值归一时应明确另标为theoretical MBU。
不能用INI的per-core DMA rate或MTE/FIX active bandwidth代替全芯片HBM分母。

## 复核与下一步

重算示例（仅CPU算术，不启动设备）：

```python
peak = 20 * 8192 * 1.8e9
flops = 4 * 1 * 32 * 8192 * 8192 * 128
t = 7208.82e-6
print(flops / t / 1e12, 100 * flops / t / peak)  # 152.52 TFLOP/s, 51.7%
```

近期优先级是同算法自动调度，以及在当前库上补验GEMM、causal GQA和decode；
MBU另需带HBM/cache证据的采集。没有910B backward或其他attention变体的有效
性能表，不将[950 backward结果](../../examples/ascend/flash_attention/README.md)
混入本报告。尾块、极端数值和其他dtype也需要独立正确性验证。

原始证据保存在产品Git之外；以下是归档标识，不是仓库相对链接，阅读本文不依赖
这些目录。数值表和计算口径已完整保留在本页：

| 证据 | 外层归档标识 |
| --- | --- |
| G | `runs/compiler-gemm-checkpoint-20261009/README.md`、`region-context-RESULTS.md`、各应用JSON和BasicInfo CSV |
| A | `runs/fa-current-20261009/METRICS.json`、`RESULTS.md`；源`e3f62c7e`、库`fd52202b…` |
| B | `runs/fa-balanced-matrix-20261010/METRICS-*.json`、`RESULTS.md`；库`282d2947…` |
| C | `runs/fa-d64-retune-20261010/final/METRICS-*.json`、`SELECTED.json` |
| R | `runs/sync-reuse-refactor-20261010/VALIDATION.json`、`compiled.json`、`identity-after.json` |

所有比较均先有数值验证；不同BK不能要求逐位相等。记录全量输出误差、输入不变、
guard、源码/库/ASC/binary身份。后续更新沿用同卡同组协议，既有单次采集不包装成
多样本平均，也不将逐核pipe active相加当作kernel elapsed或用它推算同步耗时。
