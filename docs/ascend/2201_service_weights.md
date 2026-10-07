# dav-2201 scheduling service weights

The dav-2201 automatic scheduler uses dimensionless service/order weights. Its
`tl.ascend.cost_model` is `dav-2201.service_weights.estimated.v1`. This model
helps compare relative resource pressure and buffer decisions; it predicts
neither instruction latency nor kernel time. The dav-3510 timing model is
separate.

The coarse FP16/BF16 reference ratio is MMAD : A Load3Dv2 : merged B LoadData2D =
**4 : 1 : 2**, for a native 128×128×128 FP16/BF16→FP32 MAD and 128×128 half-precision
operand copies. It is a rounded scheduling heuristic supported by current
2201 native measurements, not a cycle/tick conversion or an exact fit.

The reference evidence uses CANN 9.1 native calls on 20 AICs. Completed
32/64/128-call windows record raw `SYS_CNT` ticks, with an independent two-slot
ring and separately recorded controls. The counter period is not inferred from
the core frequency, and controls are not subtracted. Only their coarse relative
ordering informs these dimensionless weights; raw measurements stay outside
the product repository.

Both precision paths passed separate numerical gates. BF16 A Load3Dv2 uses
the installed CAPI's half storage view for its 16-bit transfer; this preserves
BF16 operand bits and does not convert their values. The matching MMAD uses
BF16 operands and FP32 accumulation. Measurements support the same coarse
ordering for these paths, rather than inferring BF16 performance from FP16.

For matching FP16 or BF16 L0 MMAD, the weight is the ceiling of four times its physical
16-element-fractal M×N×K work divided by 128³. A selected native A3D copy uses
one times its actual destination payload divided by 32KiB; a canonical,
non-transposed, full-physical-N merged B2D copy uses twice that payload ratio.
Every resulting weight is at least one. Leading storage-version axes select
one tile and do not multiply the payload. Source NZ bounding regions and
allocation-wide K caches describe dependencies/pitch, not transferred bytes,
and do not contribute to copy service work.

Scaling another geometry by work/payload is explicitly an estimate. Partial
N or segmented B2D, unknown layout, dynamic work and unmeasured
precision paths, GM DMA, FIX, vector and scalar leaves retain **uncalibrated
ordinal one**. One is an abstract fallback, never one tick or one cycle.
Serial composition, mutually exclusive branches and per-core phases retain
the existing task-cost aggregation rules. Explicit task hints retain their
existing meaning and take precedence.

The existing task fields named `latency` and `ii` carry the same dimensionless
weight on dav-2201. They are scheduling distances in this model, not proven
asynchronous completion bounds. Actual readiness/reuse completion remains
established by typed regions, dependency distances, buffer versions and
native synchronization. A favorable objective or additional buffer alone is
not a correctness or performance acceptance result.
