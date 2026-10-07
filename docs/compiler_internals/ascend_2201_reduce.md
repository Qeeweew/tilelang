# dav-2201 ordinary trailing-axis reductions

`AscendLowerVectorReduce` consumes ordinary TileLang sum/max nodes after
mixed-kernel row ownership and layout inference. It exposes native vector
operations, scratch allocations and memory effects before task materialization,
automatic buffer planning and synchronization. Users do not allocate reduction
scratch or call a composite softmax algorithm.

The architecture boundary remains dav-2201. The count-mode paths support
arbitrary positive static reduced extents through partial reduction levels.
Empty and dynamic row domains retain their ordinary row loops. Low-precision
input promotion and `clear=False` keep their established typed-node boundaries;
the multi-VL plan described below does not replace those paths.

## Static FP32 multi-VL row plan

The selected domain is `clear=True`, FP32 source/output, a real two-dimensional
compact source, static positive owned rows at most 255, and a complete trailing
row wider than 64 elements whose width is a multiple of 64. Columns and row bases
must be 32-byte aligned. Source physical row pitch equals its logical width;
output is compact with a 32-byte aligned region base. Retained output axes have
extent/capacity 1. Partial views, padded row pitches, unsupported layouts,
dynamic/empty owners, and `clear=False` keep the existing reduction plan.

Each 64-column chunk lowers to one normal-mask native WholeReduce across all
owned rows. For width C and R rows, the native arguments are:

| Native argument | Value | Unit |
| --- | --- | --- |
| repeat | R, in 1..255 | rows, `uint8_t` |
| destination repeat stride | 1 | output elements, `uint16_t` |
| source block stride | 1 | 32-byte blocks, `uint16_t` |
| source repeat stride | C/8 | 32-byte blocks, `uint16_t` |

The source repeat stride must fit `uint16_t`. Every native call explicitly
establishes normal/full mask; a preceding count-mode wrapper restores normal
mode without restoring a full mask. These are the installed CANN 9.1
`asc_repeat_reduce_sum` / `asc_repeat_reduce_max_only_value` six-argument
interfaces. Their dav-c220 implementations issue `vcadd(..., last_acc_mode=0)`
and `vcmax(..., ONLY_VALUE)` respectively, without a hidden barrier.

To bound IR expansion, the candidate is selected only when its native operation
count, `2*chunks-1` including ordinary binary combines, does not exceed the
existing per-row tree's `R*levels` native reductions. This is a conservative
structural heuristic, not measured latency. A single wide row therefore keeps
the shorter existing tree. Width 64 retains its existing single count-mode call.

Selection also requires conservative UB headroom. The common physical
allocation collector charges each root-owned storage once at its maximum
owner/alias/layout footprint, including physical padding and 32-byte alignment.
The selection adds already generated reduction scratch, reserves the future
combine scratch of previous selections, and charges this candidate's partials
plus one aligned R-element FP32 temporary per binary combine node. This includes
the final combine even when it can write directly to the output. Unknown
allocation sizes, absent allocation owners and explicit future UB version
requests keep the existing plan. These are conservative selection bounds;
ordinary resource planning and final allocation checks still apply.

The source's original strides and base alignment must also satisfy the late
native validator. A compact inferred layout cannot silently override a
noncompact original row view. The existing row/column linearity proof and the
candidate's row-pitch proof jointly establish the physical rectangle before
selection; unsupported layouts retain the existing plan or its established
legality diagnostic.

## Storage, effects and completion

Compiler-owned `shared.dyn` partials have shape
`[C/64, round_up(R,8)]`; each chunk starts at a 32-byte aligned FP32 address.
Only R valid results are written/read. The extra padded capacity is not treated
as initialized data. Ordinary Parallel Add/Max nodes form a balanced combine
tree over corresponding row partials and write the compact output. There is no
per-row scalar readback or store in the selected path.

`repeat_reduce_{sum,max}_rows_f32` has four typed IR operands: destination
region, source region, row repeats, source row stride. Chunk k reads exactly
`source[row_origin:row_origin+R,64*k:64*(k+1)]` and writes
`partials[k:k+1,0:R]`. A flat `R*64` access pointer would describe the wrong
logical memory effect when C exceeds 64; it is deliberately absent here.

The common region analysis and BufferVersion machinery consume these exact
effects. Version axes added later have singleton extents and independent
storage bases. `AscendLowerTileOp` validates this native boundary and only then
converts each exact source rectangle to its bounded pointer hull
`(R-1)*C+64`, preserving the chunk and version origins. Code generation emits
one native reduction per call. Ordinary task analysis/InsertSync provide the
completion edge from every partial producer to its combine. No helper hides
scratch, pipeline events or the complete row algorithm.

All rows belong to the existing private UB owner. The pass does not introduce
an SID axis or repartition rows. A dynamic owner count that can be zero uses the
existing fallback, preserving the enclosing task and mixed-core completion.
An explicit atomic Task/PerCoreTask cannot enclose this multi-instruction
expansion; the diagnostic requests ordinary expressions or Stage boundaries.

## Numerical contract and validation boundary

Sum uses FP32 native 64-lane partial reductions followed by explicit balanced
FP32 Add nodes. It neither widens to FP64 nor fuses rounding nodes. Its tree
can differ from the prior per-row partial tree, so numerical validation must
use an appropriate FP32 error tolerance rather than demand bit equality to
that earlier ordering. Cancellation-heavy, differing-chunk and finite-range
cases belong in device acceptance before a performance claim.

Max retains the existing `nan_propagate=False` domain. The lowering does not
establish an IEEE NaN selection or signed-zero bit-preservation guarantee for
the native Max family; `nan_propagate=True` remains diagnosed. No identity
padding is read by a native instruction, and valid infinities retain the
existing native contract.

Focused CPU regressions cover exact disjoint chunk regions, compact output,
ordinary combines and task boundaries, source/version offsets with wide index
types, mask setup, completion in generated source, fallback domains and ordinary
FA with Bkv128. Capacity regressions cover near-full UB, pending scratch and
combine reservations, alias/layout footprints and unknown future allocation
sizes. Successful CPU lowering alone does not establish native stride
legality or numerical/device performance: CANN offline compilation and locked
finite numerical comparisons are subsequent gates.
