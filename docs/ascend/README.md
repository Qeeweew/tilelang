# Ascend 910B 开发文档

本目录索引 `dav-2201` 的当前实现、使用契约和性能证据。950 的 `dav-3510`
示例及性能保留在各自文档中，不能套用到 910B。

## 阅读顺序

| 目的 | 文档 |
| --- | --- |
| 查看 GEMM、dense MHA 代码与性能 | [910B 性能报告](910b_performance.md) |
| 选择 FA 接口、分块、缓存和手排策略 | [FA 使用契约](../../examples/ascend/flash_attention/910b.md) |
| 了解硬件边界、lowering、依赖与同步 | [编译器契约](../compiler_internals/ascend_910b.md) |
| 看懂 InsertSync 的整体算法和复杂度来源 | [同步算法阅读说明](../compiler_internals/ascend_910b_insert_sync.md) |
| 理解自动、Stage-only 和完整手排 | [调度契约](../compiler_internals/ascend_910b_scheduling.md) |
| 理解自动调度目标的单位和局限 | [2201 service weights](2201_service_weights.md) |
| 构建 Ascend 后端 | [后端指南](../../tilelang/ascend/README.md) |

## 按问题定位

| 问题 | 专题与主要代码 |
| --- | --- |
| native GEMM 分块、L0 容量、unit stream | [physical tiles](../compiler_internals/ascend_2201_physical_tiles.md)；`src/ascend/transform/` |
| 双 AIV 私有 UB 与参与者域 | [AIV ownership](../compiler_internals/ascend_2201_aiv_ownership.md) |
| owner 的完整结果与输入完成 | [owner completion](../compiler_internals/ascend_2201_owner_results.md) |
| GM 读释放、写许可 | [GM handoffs](../compiler_internals/ascend_2201_gm_handoffs.md) |
| L0C 字节复用与 L0 operand 槽 | [storage reuse](../compiler_internals/ascend_2201_storage_reuse.md)、[operand slots](../compiler_internals/ascend_2201_operand_slots.md) |
| 固定 Q 平面与跨 phase 的 operand credits | [fixed planes](../compiler_internals/ascend_2201_fixed_plane_credits.md) |
| 二值事件、有限 epoch 与填排 | [local events](../compiler_internals/ascend_local_events.md)；[InsertSync](../../src/ascend/transform/insert_sync.cc) |
| 普通归约的精度、scratch、effects | [reductions](../compiler_internals/ascend_2201_reduce.md) |

专题文档保留其独立证明边界，不把 readiness、last reader、可覆写、整条 MAD 完成
合并成一个模糊的“已同步”。`WaitFlag(V_S)` 可以证明匹配 notification 覆盖的
Vector 操作完成；flag ID 可复用的证明本身不增加数据完成边，也不授权 buffer alias。

## 实现入口与职责

- [gemm_910b.py](../../examples/ascend/gemm_910b.py)：普通自动 K 流水，默认 grid 或有界 persistent 输出遍历。
- [910b.py](../../examples/ascend/flash_attention/910b.py)：普通 generic / 自动流水 attention。
- [persistent_910b.py](../../examples/ascend/flash_attention/persistent_910b.py)：dense MHA paired-query 算法，连续跨 query 流水和有界 workspace。
- [Ascend transforms](../../src/ascend/transform)：依赖、调度、版本、同步与物理 storage 验证。
- [回归入口](../../testing/ascend/README.md)：CPU IR/codegen 与设备测试分层。

Python 负责算法、tile、遍历和缓存策略；编译器负责实际依赖、stage 位移、版本、
容量与同步。手排沿用 `T.Pipelined(..., num_stages=3, annotations=...)` 与
`T.Stage`，没有独立 ManualPipeline API。逻辑 task、query 状态、内层 chunk
使用各自的生命周期，不能用每 query drain 或 barrier 绕过完成性证明。

## 修改与验证

每次只验证一个明确假设。依次检查 CPU IR、生成 ASC、CANN 离线编译、少量设备
正确性，再做同卡 Torch 性能对比。目标显式指定 `dav-2201`；确认实际 Python 包、
加载库及生成产物身份，避免把旧缓存或别的安装当作本次构建。

设备前检查全卡健康、进程、显存及利用率，每进程显式映射物理卡到逻辑 0；
直接在当前容器运行，不使用 `/tmp` 锁。设备错误先停止定位，不 reset 或换卡绕过。
`msprof op --kernel-name` 每组一次默认 replay，不额外套应用计时循环。

保持私有 UB、双 AIV ownership、GM collective、Fixpipe `ND_PARA=1` 和有界
workspace。若 ASC 与机器码逐字节相同，可以复用对应的有效设备/profile 证据；
仅源码看似等价不够。编译成功也不能替代正确性或性能验收。

## 文档维护边界

使用契约只描述 API、合法性与推荐参数；编译器契约描述 pass 边界和证明；性能报告
集中记录带版本的观测值。实验过程、失败候选、ASC、profiler 和原始报告留在仓库外。
不在每份文档重复滚动追加“最新”历史，更新数字时同步更新其证据版本和测量口径。
