# 910B：GEMM 与 Dense MHA

以下数据于2026-10-10使用当前实现重新测量。代码和参数如下。
设备为 Ascend 910B3（20 AIC / 40 AIV，1800 MHz），CANN 9.1，目标 `dav-2201`。
表中时间单位为 µs，越小越好；MFU 只列 TileLang：

```text
MFU = 有效 FLOPs / 时间 / 294.912 TFLOP/s
GEMM FLOPs = 2 M N K
MHA FLOPs = 4 B H S² D
```

MHA 的 FLOPs 计 QK 和 PV，不计 softmax。各形状均先验证正确性，再在同卡同进程
对比 Torch；使用 msprof 默认 replay 的 kernel 时间，不包含编译和主机分配。

## GEMM

计算 `C = A @ B.T`，FP16/BF16 输入、FP32 累加，输出与输入同 dtype。
统一使用persistent20，Tile M/N/K固定为128/256/256。Torch基线是 `torch.matmul(A, B.T)`。

| M/N/K | dtype | TileLang | Torch | MFU |
| --- | --- | ---: | ---: | ---: |
| 2048/2048/2048 | FP16 | 83.08 | 82.42 | 70.1% |
| 4096/4096/4096 | BF16 | 543.78 | 526.48 | 85.7% |
| 512/4096/2048 | BF16 | 50.94 | 55.32 | 57.2% |

### TileLang 代码

以下展示persistent20、无额外A缓存的完整逻辑；[实际工厂](../../examples/ascend/gemm_910b.py)
还支持尾块和显式 A 缓存。`T.Pipelined` 自动选择 K 循环的排步和 buffer 版本。

```python
from tilelang.ascend import language as T


def make_gemm(M, N, K, dtype="bfloat16"):
    BM, BN, BK = 128, 256, 256
    n_tiles = (N + BN - 1) // BN
    tiles = ((M + BM - 1) // BM) * n_tiles
    groups = min(tiles, 20)

    @T.macro
    def output_tile(A, B, C, tile_id):
        mi = tile_id // n_tiles
        ni = tile_id % n_tiles
        a_l1 = T.alloc_l1((BM, BK), dtype)
        b_l1 = T.alloc_l1((BN, BK), dtype)
        accumulator = T.alloc_l0c((BM, BN), "float32")
        for ki in T.Pipelined(T.ceildiv(K, BK)):
            T.copy(A[mi * BM:(mi + 1) * BM, ki * BK:(ki + 1) * BK], a_l1)
            T.copy(B[ni * BN:(ni + 1) * BN, ki * BK:(ki + 1) * BK], b_l1)
            T.gemm(a_l1, b_l1, accumulator, transpose_B=True, clear_accum=(ki == 0))
        T.copy(accumulator, C[mi * BM:(mi + 1) * BM, ni * BN:(ni + 1) * BN])

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype),
             B: T.Tensor((N, K), dtype),
             C: T.Tensor((M, N), dtype)):
        with T.Kernel(groups) as block:
            for iteration in T.serial(T.ceildiv(tiles - block, groups)):
                output_tile(A, B, C, block + iteration * groups)

    return main
```

20个block轮转处理输出tile；外层循环表达输出分配，内层 `T.Pipelined` 的
K流水由编译器自动排步。下面调用与表中使用同一策略。

编译与调用使用实际工厂：

```python
import tilelang
import torch
import torch_npu
from examples.ascend.gemm_910b import gemm

M, N, K = 4096, 4096, 4096
program = gemm(M, N, K, dtype="bfloat16", num_cores=20)
kernel = tilelang.compile(program, target={"kind": "ascend", "arch": "dav-2201"},
                          execution_backend="tvm_ffi", out_idx=None)
torch.npu.set_device("npu:0")
A = torch.randn((M, K), device="npu:0", dtype=torch.bfloat16)
B = torch.randn((N, K), device="npu:0", dtype=torch.bfloat16)
C = torch.empty((M, N), device="npu:0", dtype=torch.bfloat16)
kernel(A, B, C)
torch.npu.synchronize()
```

## Dense MHA

计算 `O = softmax(Q @ K.T / sqrt(D)) @ V`，输入为连续 `[B,H,S,D]`，
无mask、dropout或bias。Torch基线为 `torch_npu.npu_fusion_attention`（FAS），
它还返回softmax辅助信息；TileLang只输出O。

| B/H/S/D | dtype | 完整手排 | 同算法自动 | 普通流水 | Torch FAS | MFU（完整手排） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1/32/2048/128 | BF16 | 479.42 | 638.06 | 1494.00 | 495.92 | 48.6% |
| 1/32/4096/128 | BF16 | 1837.96 | 2472.74 | 5707.02 | 1871.38 | 50.7% |
| 1/32/8192/128 | BF16 | 7151.56 | 9882.92 | 22789.50 | 7390.40 | 52.1% |
| 1/4/4096/64 | FP16 | 253.94 | 261.64 | 760.56 | 274.12 | 22.9% |
| 4/16/2048/64 | FP16 | 793.94 | 797.18 | 3088.12 | 819.20 | 29.3% |
| 1/32/8192/64 | BF16 | 6201.86 | 6381.08 | 23502.74 | 6642.82 | 30.1% |

完整手排的outer阶段为QK→softmax→PV→更新O，偏移为0/1/1/2；内层PV自动流水。
同算法自动保留相同分块、缓存和P舍入点，移除outer Stage与固定顺序后自动排步。
普通流水使用独立的[普通FA实现](../../examples/ascend/flash_attention/910b.py)，
BQ64/BK256、stage上限4。本次为固定配置重测，没有重新搜索分块。

### TileLang 代码：逻辑流水

[实际实现](../../examples/ascend/flash_attention/persistent_910b.py)使用一个AIC和
两个AIV组成混合组。AIC计算两个query block，两个AIV各自计算一个block的softmax
和输出更新。`Score/P/PV`经GM workspace交接，UB按AIV私有；workspace由编译器
分配，不需要调用者传入。

主要中间量如下。`Score/P/PV`声明一个逻辑task的空间，由buffer版本保护不同
流水迭代；`OState`的两槽显式属于两个交错query，生命周期不同。

```python
with T.MixedKernel(groups, sids=2) as (group, sid):
    Score = T.alloc_workspace((groups, 2, block_q, block_kv), "float32")
    P = T.alloc_workspace((groups, 2, block_q, block_kv), dtype)
    PV = T.alloc_workspace((groups, 2, block_q, head_dim), "float32")
    OState = T.alloc_workspace((groups, 2, 2, block_q, head_dim), "float32")
    T.annotate_buffer_versions({Score: 2, P: 2, PV: 2, OState: 1})
    # L1/L0分配由AIC使用，softmax和O更新的UB放在T.AIVPrivate()作用域。
```

下面是实际源码的outer循环与关键运算摘录，省略了buffer声明和部分缓存分支；
可直接运行的工厂调用在下一节。`kv`是当前1024-key块，`query_slot`属于query状态，
同步与task版本由编译器生成，Python不编写延迟迭代或set/wait。

```python
# group 的 query pairs 轮转分配；一个流水跨越所有本地 query pairs。
tasks = ((total_pairs + groups - 1 - group) // groups) * kv_tiles
for t in T.Pipelined(tasks, num_stages=3, annotations={
        "enable_offset": True,
        "tl.ascend_preserve_issue_order": preserve_issue_order}):
    local_pair = t // kv_tiles
    flat_pair = group + local_pair * groups
    batch_index = flat_pair // (heads * query_pairs)
    head_index = (flat_pair // query_pairs) % heads
    query_pair = flat_pair % query_pairs
    kv = t % kv_tiles
    query_slot = local_pair % 2

    with T.Stage(0):
        with T.Cube():
            # D128 分支：Q 留驻 L0A，同一个 K chunk 给两个 owner 使用。
            if kv == 0:
                for owner in T.serial(2):
                    T.copy(Q[batch_index, head_index, query_pair, owner, :, :],
                           q_l1[owner, :, :])
            for owner in T.serial(2):
                T.copy(q_l1[owner, :, :], qk_a_l0[owner, :, :])
            for n in T.serial(block_kv // qk_extent):
                T.copy(K[batch_index, head_index,
                         kv * block_kv + n * qk_extent:
                         kv * block_kv + (n + 1) * qk_extent, :], k_l1)
                T.copy(k_l1, qk_b_l0)
                for owner in T.serial(2):
                    T.gemm(qk_a_l0[owner, :, :], qk_b_l0, score_acc,
                           transpose_B=True, clear_accum=True)
                    T.copy(score_acc, Score[group, owner, :,
                                            n * qk_extent:(n + 1) * qk_extent])

    with T.Stage(1):
        if kv == 0:
            T.fill(maximum[query_slot, :], -T.infinity("float32"))
            T.fill(denominator[query_slot, :], 0)
        for sub in T.serial(block_q // row_block):
            row = sub * row_block
            T.copy(Score[group, sid, row:row + row_block, :], score_input)
            for i, j in T.Parallel(row_block, block_kv):
                scores[i, j] = score_input[i, j] * T.float32(scale)
            T.reduce_max(scores, block_maximum, dim=1, clear=True)
            for i in T.Parallel(row_block):
                next_maximum[i] = T.max(maximum[query_slot, row + i], block_maximum[i])
                alpha[row + i] = T.exp(maximum[query_slot, row + i] - next_maximum[i])
            for i, j in T.Parallel(row_block, block_kv):
                scores[i, j] = scores[i, j] - next_maximum[i]
            for i, j in T.Parallel(row_block, block_kv):
                scores[i, j] = T.exp(scores[i, j])
                probabilities[i, j] = T.cast(scores[i, j], dtype)
            T.reduce_sum(scores, row_sum, dim=1, clear=True)
            for i in T.Parallel(row_block):
                denominator[query_slot, row + i] = (
                    denominator[query_slot, row + i] * alpha[row + i] + row_sum[i])
            T.copy(next_maximum, maximum[query_slot, row:row + row_block])
            T.copy(probabilities, P[group, sid, row:row + row_block, :])

    with T.Stage(1):
        with T.Cube():
            T.copy(V[batch_index, head_index,
                     kv * block_kv:(kv + 1) * block_kv, :], v_l1)
            for owner in T.serial(2):
                for part in T.serial(block_q // pv_rows):
                    for chunk in T.Pipelined(pv_chunks):
                        T.copy(P[group, owner, part * pv_rows:(part + 1) * pv_rows,
                                 chunk * pv_extent:(chunk + 1) * pv_extent], p_l1)
                        T.gemm(p_l1, v_l1[chunk * pv_extent:(chunk + 1) * pv_extent, :],
                               pv_acc, transpose_B=False, clear_accum=(chunk == 0))
                    T.copy(pv_acc, PV[group, owner,
                                       part * pv_rows:(part + 1) * pv_rows, :])

    with T.Stage(2):
        for sub in T.serial(block_q // row_block_vec2):
            row = sub * row_block_vec2
            T.copy(PV[group, sid, row:row + row_block_vec2, :], pv)
            if kv == 0:
                T.copy(pv, output)
            else:
                T.copy(OState[group, sid, query_slot, row:row + row_block_vec2, :], output)
                for i, j in T.Parallel(row_block_vec2, head_dim):
                    output[i, j] = output[i, j] * alpha[row + i]
                for i, j in T.Parallel(row_block_vec2, head_dim):
                    output[i, j] = output[i, j] + pv[i, j]
            if kv + 1 < kv_tiles:
                T.copy(output, OState[group, sid, query_slot, row:row + row_block_vec2, :])
            else:
                for i, j in T.Parallel(row_block_vec2, head_dim):
                    output[i, j] = output[i, j] / denominator[query_slot, row + i]
                for i, j in T.Parallel(row_block_vec2, head_dim):
                    output_cast[i, j] = T.cast(output[i, j], dtype)
                T.copy(output_cast, O[batch_index, head_index, query_pair,
                                     sid, row:row + row_block_vec2, :])
```

### 编译和运行完整 MHA

在仓库根目录运行。以下参数就是表中完整手排的配置：D128共享K并分块staging；
D64保留完整K/P panels，QK按owner遍历。两者逻辑BK均1024，P转换到输入dtype
后参与PV；FP32统计量与O状态贯穿全部KV块。

```python
import tilelang
import torch
import torch_npu
from examples.ascend.flash_attention.persistent_910b import persistent_attention_fwd

B, H, S, D = 1, 32, 8192, 128
# 可换成表中的其他形状；S 要整除256和1024。
dtype = "bfloat16"
wide = D == 128
program = persistent_attention_fwd(
    batch=B, heads=H, query_pairs=S // 256, kv_tiles=S // 1024,
    groups=20, block_q=128, block_kv=1024, head_dim=D, dtype=dtype,
    row_block=8, row_block_vec2=64 if wide else 128,
    qk_chunk=128 if wide else 256, pv_chunk=128 if wide else 256,
    pv_rows=128 if wide else 64,
    cache_pv=True, separate_score_input=True, inplace_final_division=True,
    reuse_qk_b=wide, stream_qk_b=wide, stream_pv_p=wide,
    preserve_issue_order=True,
)
kernel = tilelang.compile(program, target={"kind": "ascend", "arch": "dav-2201"},
                          execution_backend="tvm_ffi", out_idx=None)

torch.npu.set_device("npu:0")
q, k, v = [(torch.randn((B, H, S, D), dtype=torch.float32) * 0.3)
           .to(dtype=getattr(torch, dtype), device="npu:0") for _ in range(3)]
o = torch.empty_like(q)
paired = (B, H, S // 256, 2, 128, D)
kernel(q.view(paired), k, v, o.view(paired))
torch.npu.synchronize()
```

Q/O只是reshape，不做转置或复制。完整接口、buffer ownership与合法性约束见
[FA使用文档](../../examples/ascend/flash_attention/910b.md)。
