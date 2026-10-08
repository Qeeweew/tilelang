# dav-2201 ordinary trailing-axis reductions

`AscendLowerVectorReduce` consumes ordinary TileLang sum/max nodes after
mixed-kernel row ownership and layout inference. It exposes native vector
operations, scratch allocations and memory effects before task materialization,
automatic buffer planning and synchronization. Users do not allocate reduction
scratch or call a composite softmax algorithm.

The architecture boundary remains dav-2201. The count-mode paths support
arbitrary positive static reduced extents through partial reduction levels.
Unknown or incompatible dynamic row domains retain their ordinary row loops. Low-precision
input promotion and `clear=False` keep their established typed-node boundaries;
the multi-VL plan described below does not replace those paths.

## FP32 multi-VL row plans

The selected domain is `clear=True`, FP32 source/output, a real two-dimensional
compact source, nonnegative owned rows with a proved positive maximum at most
255, and a complete trailing
row wider than 64 elements whose width is a multiple of 64. Columns and row bases
must be 32-byte aligned. Source physical row pitch equals its logical width;
output is compact with a 32-byte aligned region base. Retained output axes have
extent/capacity 1. Partial views, padded row pitches, unsupported layouts,
unknown row bounds, and `clear=False` keep the existing reduction plan.
Bounded dynamic owners use their actual row count in effects and native repeats;
an enclosing `rows > 0` guard prevents zero-repeat instructions. Scratch and
structural candidate selection use the proved maximum, not a sampled row count.

For static positive rows and `512 <= C <= 4096` with `C % 512 == 0`, the preferred candidate uses
three native tasks. Two count-mode BlockReduce operations compact each group
of eight FP32 elements, producing `[R,C/8]` and then `[R,C/64]` in independent
compiler-owned UB buffers. Both input counts are multiples of 64, so the plan
requires no partial count repeat or unknown tail-write behavior. These are the
installed CANN 9.1 `asc_datablock_reduce_sum` / `asc_datablock_reduce_max`
interfaces, which issue `vcgadd` / `vcgmax`. Their default destination repeat
stride is **one 32-byte block**, unlike WholeReduce's one-element stride.

The final normal-mask WholeReduce consumes `[R,P]`, where `P=C/64`, and writes
one result per row. Its active lane count is P, repeat count is R, source
repeat stride is `P/8` 32-byte blocks, and destination stride is one element.
P is in 8..64 and divisible by eight; every scratch row begins at a 32-byte
boundary and no padding is read. For `16x1024`, the two disjoint scratch
allocations total 9216 bytes. Operation counts describe the plan; they are
not a latency model.

When the larger allocation does not fit the common root budget, the same
compact domain retains the two-level candidate; bounded dynamic rows also use
this candidate instead of the static BlockReduce plan. A count-mode WholeReduce
consumes `R*C` elements and produces `[R,C/64]`, followed by the same masked
row reduction. This uses 1024 bytes for `16x1024`. WholeReduce's default source
repeat stride is eight 32-byte blocks and its destination stride is one
element; each group of 64 inputs produces one partial without crossing rows.

Both candidates charge their complete aligned scratch against the same root
allocation and manual-version facts. Already selected scratch remains charged
when selecting subsequent reductions. They retain the same physical
compactness, private owner, region-base alignment and fallback requirements.
If neither allocation or the width contract is admissible, the existing chunk
candidate remains available.

In that candidate, each 64-column chunk lowers to one normal-mask native WholeReduce across all
owned rows. For width C and R rows, the native arguments are:

| Native argument | Value | Unit |
| --- | --- | --- |
| repeat | R, in 1..255 | rows, `uint8_t` |
| destination repeat stride | 1 | output elements, `uint16_t` |
| source block stride | 1 | 32-byte blocks, `uint16_t` |
| source repeat stride | C/8 | 32-byte blocks, `uint16_t` |

The source repeat stride must fit `uint16_t`. Code generation establishes
normal mask and the exact active lanes before a row reduction; a preceding
count-mode wrapper restores normal mode without restoring the active mask.
These are the installed CANN 9.1
`asc_repeat_reduce_sum` / `asc_repeat_reduce_max_only_value` six-argument
interfaces. Their dav-c220 implementations issue `vcadd(..., last_acc_mode=0)`
and `vcmax(..., ONLY_VALUE)` respectively, without a hidden barrier.

To bound IR expansion, the candidate is selected only when its native operation
count, `2*chunks-1` including ordinary binary combines, does not exceed the
existing per-row tree's `R*levels` native reductions. This is a conservative
structural heuristic, not measured latency. When the preferred compact
candidates are unavailable, a single wide row therefore keeps the shorter
existing tree. Width 64 retains its existing single count-mode call.

Selection also requires conservative UB headroom. The common physical
allocation collector charges each root-owned storage once at its maximum
owner/alias/layout footprint, including physical padding and 32-byte alignment.
The selection adds already generated reduction scratch, reserves the future
combine scratch of previous selections, and charges this candidate's partials
plus one aligned R-element FP32 temporary per binary combine node. This includes
the final combine even when it can write directly to the output. Unknown
allocation sizes, absent allocation owners and automatic future UB version
requests keep the existing plan. These are conservative selection bounds;
ordinary resource planning and final allocation checks still apply.

An unrelated manual buffer ring already reserves physical storage and does
not make the root's headroom unknown. The same collector charges each leading
version slot with its own 32-byte alignment, while count-only annotations on
flat allocations preserve their existing physical size. Aliases still share
one storage owner and contribute their maximum physical footprint. If either
native reduction operand itself has multiple manual versions, selection keeps
the existing plan: the later version-layout rewrite can change its raw stride.

The source's original strides and base alignment must also satisfy the late
native validator. A compact inferred layout cannot silently override a
noncompact original row view. The existing row/column linearity proof and the
candidate's row-pitch proof jointly establish the physical rectangle before
selection; unsupported layouts retain the existing plan or its established
legality diagnostic.
Both compact candidates write the user's output directly through a native
row operation, so it also proves the original destination's compact row pitch
and aligned base independently of an inferred layout.

## Storage, effects and completion

For the chunk candidate, compiler-owned `shared.dyn` partials have shape
`[C/64, round_up(R,8)]`; each chunk starts at a 32-byte aligned FP32 address.
Only R valid results are written/read. The extra padded capacity is not treated
as initialized data. Ordinary Parallel Add/Max nodes form a balanced combine
tree over corresponding row partials and write the compact output. There is no
per-row scalar readback or store in the selected path.

Each `datablock_reduce_{sum,max}` call has three typed IR operands: exact
write and read `tl.access_ptr` effects, and a positive uint32 input count in
complete 64-lane repeats. The input span is exactly count elements and the
output is count/8 elements. The late native boundary checks compact FP32 UB
storage, aligned bases, independent storage, access masks and backing bounds.
Each level is a separate schedulable task; ordinary dependency and sync
analysis must complete its producer before the next level starts and complete
scratch readers before storage is reused.

`repeat_reduce_{sum,max}_rows_f32` has five typed IR operands: destination
region, source region, uint8 row repeats, uint16 source row stride in 32-byte
blocks, and uint8 valid lanes. The source region has exact shape
`[rows,valid_lanes]`; the output is a compact row vector, optionally retaining
a trailing unit axis. Full-VL chunks use valid lanes 64. Partial-lane calls
require compact scratch rows with pitch equal to valid lanes and column
origin zero. Chunk k reads exactly
`source[row_origin:row_origin+R,64*k:64*(k+1)]` and writes
`partials[k:k+1,0:R]`. A flat `R*64` access pointer would describe the wrong
logical memory effect when C exceeds 64; it is deliberately absent here.

The common region analysis and BufferVersion machinery consume these exact
effects. Version axes added later have singleton extents and independent
storage bases. `AscendLowerTileOp` validates this native boundary and only then
converts each exact source rectangle to its bounded pointer hull
`(R-1)*row_stride*8+valid_lanes`, preserving the chunk and version origins. Code generation emits
one native reduction per call. For dynamic rows, the validator proves the original
region extent is in `[1,255]` under the nonempty guard and matches the uint8
repeat exactly before narrowing. Ordinary task analysis/InsertSync provide the
completion edge from every partial producer to its combine. No helper hides
scratch, pipeline events or the complete row algorithm.

All rows belong to the existing private UB owner. The pass does not introduce
an SID axis or repartition rows. A bounded empty owner issues no reduction or
combine; its enclosing mixed-core collective completion remains independent
of the local nonempty guard.
An explicit atomic Task/PerCoreTask cannot enclose this multi-instruction
expansion; the diagnostic requests ordinary expressions or Stage boundaries.

## Numerical contract and validation boundary

Sum uses either two levels of native eight-element block sums followed by a
masked partial-row sum, native 64-lane partial sums followed by that masked
sum, or native chunks followed by explicit balanced FP32 Add nodes. It neither
widens to FP64 nor fuses rounding nodes. Its tree
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
combine reservations, manual slot padding, alias/layout footprints and unknown
future allocation sizes. Successful CPU lowering alone does not establish
native stride legality or numerical/device performance: CANN offline compilation
and locked finite numerical comparisons are subsequent gates.

## Bounded normal-mask reuse

The 2201 emitter can share normal/full mask setup across adjacent direct
`brcb_f32`, FP32 row sub/mul/div, and full-lane row reductions, including
intervening `PIPE_V` barriers. These exact CANN overloads preserve mask state.
Every sequence starts with unknown state; scope/control-flow boundaries,
other calls, count-mode operations and partial masks end a run. No state
escapes an inner sequence, branch, loop or function. This is an emitter-local
peephole over implicit setup, and does not remove mathematical instructions,
memory effects or completion boundaries from the scheduled IR.

A nested unknown argument call is evaluated exactly once into a C++ temporary
before this operation's mask setup. Leaving it inline after the setter could
clobber the current operation's mask even if both neighboring runs were
invalidated. Such arguments conservatively end the reusable run.
