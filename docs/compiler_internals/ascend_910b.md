# Ascend 910B compiler contracts

Use the Ascend dialect (`tilelang.ascend.language`) and an explicit target:

```python
target = {"kind": "ascend", "arch": "dav-2201"}
```

`ASCEND_NPU_ARCH` supplies a default when creating a target. The frontend fixes
that default into the target before lowering and binary caching. Explicit
`arch` or `mcpu` takes precedence; when both are given they must agree. Without
an explicit architecture or environment default, the target uses `dav-3510`.

## Hardware capabilities and storage

The target descriptor owns legality and storage capacities. It contains no
latency or bandwidth estimates. The CANN 9.1 2201 storage contract is:

| Storage | Capacity | Owner |
| --- | ---: | --- |
| UB | 192 KiB | Each AIV, privately |
| L1 | 512 KiB | AIC |
| L0A, L0B | 64 KiB each | AIC |
| L0C | 128 KiB | AIC |

A mixed group contains one AIC and two AIVs. Their UB capacities cannot be
added to permit a larger allocation. VF/RegBase execution, FP8/FP4 and MX GEMM
are unavailable on 2201. The 3510 capability and resource description is
separate. Public CANN declarations alone do not prove that a route exists on
2201: direct L0C-to-UB and L1-to-UB transfers are unsupported. CANN's UB-to-L1
Matmul helper uses a software GM handoff rather than a direct DMA instruction.

## Lowering and stage contracts

The frontend expresses tile sizes, traversal, cache reuse and the algorithm.
BufferRegion and Layout preserve logical shape, valid regions and physical
ordering. Native 2201 L0A uses ZZ order; it is distinct from L1's NZ layout.
The input capability boundary diagnoses unsupported operations before layout
inference or scheduling.

L1 GEMM is expanded before layout inference into explicit L0 allocations,
L1-to-L0 copies and ordinary L0 GEMM tasks. A bounded sub-K loop keeps IR size
independent of K; a final partial chunk has its own compact allocation. The
allocator and scheduler see these buffers and operations, including their
read/write regions. No private template mutex or hidden allocation participates
in this path. The L1 precision path uses matching FP16/BF16 NN or NT operands
and FP32 accumulation; transposed A remains unsupported.

When one logical accumulator exceeds L0C capacity, the same boundary can pair
its full-clear GEMM with its adjacent complete GM export. It replaces that
pair with an ordinary physical-tile loop, a compact L0C accumulator with two
explicit slots when capacity permits, and subtile exports into the original
GM region. K lowering then operates on each
physical tile. The whole GM reader remains outside these loops. See
[physical GEMM tiles](ascend_2201_physical_tiles.md) for admission and completion
contracts; this does not imply that the logical L1 or UB allocations fit.

Native LoadData2D consumes complete source fractals. Its lowering boundary
checks row16/C0-aligned origins, matching physical source/destination tiles,
allocation bounds and leading version indices. A larger L0 destination cannot
authorize extra L1 reads. GM tail padding must first materialize those source
fractals in a sufficient L1 allocation; unknown bounds have a diagnostic.
Nonempty GM-to-L1 copies require an in-bounds source origin on each matrix
axis. Fully out-of-bounds source tiles have a diagnostic; an already-empty
transfer remains empty and does not initialize later consumers' storage.
After OOB padding, native copy normalization exposes these complete physical
read/write regions before dependency analysis. Padding producers and later
storage reuse therefore participate in RAW/WAR dependencies. The GEMM keeps
its effective logical M/N/K. The late emitter consumes the same footprint
helper; it does not infer a separate set of allowed reads.

After this normalization, native Cube instruction selection may replace
2201 FP16/BF16 copies with Load3Dv2. A requires canonical L1 NZ and L0A ZA,
full padded M and a compact aligned K interval. NN B requires canonical L1 NZ,
its native B layout, full padded N and an aligned K-row interval. Proven
allocation bounds and actual regions remain shared scheduling/versioning
facts; only late lowering emits pointers and native element descriptors.
Annotated or explicitly atomic copies, incompatible layouts and unsupported
geometry retain the established copy path.

Selected loads share invariant but independent FMATRIX_A/B widths in each
kernel. One root AIC/PIPE_S task initializes both widths and the shared repeat,
padding and boundary state. Shared dependency analysis supplies storage-free
RAW edges; InsertSync emits one real root S-to-MTE1 acquire before all data
controls. Conditional and empty readers therefore inherit initialization
without repeatedly consuming a one-shot event. This readiness protocol adds
no buffer version, hidden event number or whole-MTE1 data completion. Different
widths on the same side retain ordinary loads; live reconfiguration is
unsupported. External and unmodeled opaque calls prevent selection, and an
already selected kernel diagnoses them at synchronization lowering because
they could overwrite the invariant state.

The selected path requires synchronization lowering and is disabled when
automatic scheduling is disabled. A hand-created PerCore reader must have one
flat Task load leaf; internally compound, repeated or conditional groups
retain their existing diagnostic. Descriptor count reduction alone is not
evidence of lower execution time. See the native operand-load contract in
[2201 physical tiles](ascend_2201_physical_tiles.md).

Logical L0C-to-UB and UB-to-L1 transitions become ordinary copies through
compiler-owned GM workspace before layout or ownership is assigned. Each
kernel group has a separate workspace slice, and each transfer retains its
actual valid region. The padded allocation does not enlarge the valid input.
An L1 consumer receives explicit initialization for any unwritten tail.
Handoff lowering first records a legal logical zero-fill region. Storage
normalization then extends its complete matrix axes to the established
physical allocation and consumes the padding intent. Partial slices and
empty domains retain their original bounds throughout these phases.

The mixed ownership phase connects ordinary Parallel expressions, copy
regions, row broadcasts and trailing-axis reductions through their shared UB
buffers. It gives each AIV a private row capacity and a bounded valid domain,
then rewrites logical row indices and copy regions together. An empty owner
has zero work while still participating in collective completion. Work
division is established in TIR before physical layout and code generation.
Connected GM boundaries establish the actual valid row domain before internal
UB copies are checked. A row-state copy may use that domain or the complete
logical capacity, independently of its statement position. It cannot invent
a smaller partial domain or shift the owned row origin.

The backend reuses upstream's task metadata, ScheduledTIR representation,
region dependency analysis, loop distances, buffer versions and multibuffer
plan. The pass responsibilities are:

| Stage | Established contract |
| --- | --- |
| Logical GM handoff | Explicit physical copies, group-isolated workspace and valid domains |
| L1 GEMM and mixed ownership | Native Cube tasks and the two private AIV row domains |
| Layout, OOB and native copy normalization | Physical layouts, valid logical domains and actual instruction regions |
| Vector and reduce lowering | Native task effects and explicit private UB scratch |
| MaterializeScheduleUnits | Dependency-bearing task boundaries and explicit core/pipe facts |
| AutoSchedule | Selected task order, stages and storage version counts |
| PrepareMultiBuffer and ResolveCore | Concrete version epochs and core streams |
| InsertSync | Completion and reuse dependencies, including finite fill/drain boundaries |
| MaterializeMultiBuffer | Physical addresses from the selected version plan |
| Codegen | Native instructions for the established layout and schedule |

Codegen does not partition logical work or choose buffer versions. Resource
validation and synchronization consume the same selected plan.

Ordinary CUBE/VECTOR blocks describe core ownership. Task materialization
retains that ownership in the existing core-mask metadata and derives each
task's hardware pipe from its native instruction effects. A VECTOR DMA task
therefore keeps its MTE2/MTE3 pipe. The 3510 SIMD/SIMT VF blocks retain their
atomic instruction boundary. On 2201 an explicit `T.Task` currently accepts
one native MemBase operation; a compound or repeated native expression has
an explicit diagnostic. Ordinary Parallel expressions expose their individual
native operations to automatic scheduling and completion analysis.
The original `cthread` variable, extent and `vector_count` remain lexical launch
facts. They do not permit an empty AIV owner to skip collective completion.

For explicitly indexed independent queries, `InsertSync` can prove a narrower
[owner result completion](ascend_2201_owner_results.md) from typed FIX/GM/UB
regions. It publishes a complete Cube result per owner and GM slot. For
ordinary private-UB-to-GM inputs consumed by an AIC owner traversal in the same
activation, it can use owner/slot IDs with staggered mode-2 participation:
the other AIV arrives after old-slot release, and the owner arrives after its
complete MTE3 export. AIC still receives both physical AIV contributions.
Ordinary paired reader release is retained and independently proves reuse.
These owner edges do not enter the all-participant closure or buffer-alias
lifetime proof. Unsupported geometry or unavailable IDs keeps collective
input readiness.

Common buffer-owner inference can recognize a full GEMM definition on the
first iteration of a guaranteed static descendant loop. Only positive,
unguarded, unit-step descendants contribute their first coordinate; the
proposed owner's coordinate remains symbolic. The clear predicate must prove
true for the complete, unique allocation, without an earlier scalar, opaque
or alias read. Conditional or unknown initialization and partial views cannot
supply this proof. The analysis preserves the GEMM's conservative read-write
region and existing L0A/B loop structure; scheduling, storage reuse and
synchronization consume the resulting common owner independently.

2201 NZ-to-ND Fixpipe copies configure `ND_PARA` explicitly with
`asc_set_l0c_copy_params(1, 0, 0)` before issuing the transfer. The installed
`asc_copy_l0c2gm` implementation issues the copy without configuring this
register; a native instruction in generated code alone is insufficient.
On 2201 this setter completely defines `ND_PARA=1`: one ND matrix with zero
source/destination inter-matrix strides. Row pitch, conversion and valid M/N
remain per-transfer arguments. Before task materialization,
`AscendSelectFixpipeConfig` can establish this invariant once in an explicit
kernel-root AIC/S task. Selected copies only read the shared register;
AutoSchedule and InsertSync share a storage-free root-to-consumer RAW edge
with finite completion and an empty-loop drain. Existing proven pipe paths
can cover that completion; a nonempty loop may wait only on its first epoch.
L0C readiness/reuse and GM handoff completion still come from buffer regions.
Unknown external calls, existing raw FIX operations, conditional copies and
explicit atomic copy tasks retain the per-copy setter and `PIPE_FIX` barrier
sequence from CANN's `dav_c220` Basic Fixpipe implementation. The internal
selected-copy marker is rejected without a preceding unconditional root
initializer. Disabling automatic scheduling retains this established local
configuration path; 3510 keeps its own setter contract.
L0C reads and GM writes retain the original explicit buffer regions. See the
[2201 configuration API](https://asc.gitcode.com/api/SIMD-API/c_api/cube_datamove/asc_set_l0c_copy_params.html).
The source stride uses the canonical allocation's padded M extent even when
the logical read region contains only a few valid rows. Tail validity changes
the transfer size and memory effects, never the NZ column-group pitch. Leading
version dimensions select a complete matrix allocation; nonzero origins in
the trailing L0C matrix dimensions currently have an explicit diagnostic.

Vector kernels and partitioned mixed kernels lower dense `T.Parallel`
expressions to native MemBase count operations after layout and OOB
normalization. Add, subtract, multiply, division, min/max, exp, fill and
scalar/row broadcast expose their read/write regions and compiler-owned UB
temporaries before scheduling. FP16/BF16/FP32 casts expose each widening or
narrowing operation; BF16 arithmetic widens and rounds at each expression node.
Uniform integer fill values use scalar conversion followed by a typed native
duplicate; they do not enable integer lane arithmetic. `T.infinity` retains
its ordinary typed constant semantics.
The count may have a static tail or a bounded dynamic extent, including zero;
each physical UB operand origin must satisfy 32-byte instruction alignment.
Multi-row DMA uses target-specific byte gaps on GM and 32-byte gaps on UB.

For 2201, layout inference preserves the logical Parallel loop axes and
validates their ordinary nest layout. Vector lowering chooses the contiguous
instruction domain after consuming row broadcasts and column predicates.
The 3510 fusion and VF layout contracts are unchanged. Independent dense
assignments over a partitioned AIV's static private row domain can use one
whole-tile count instruction, including consecutive assignments with their
typed rounding boundaries. Logical and physical continuity, alignment and
same-storage alias checks establish lane independence; unowned serial loops,
cross-row dependencies, predicates and dynamic domains retain their controls.

Static FP32 row broadcasts in ordinary `T.Parallel(R, C)` assignments can
use one native Brcb followed by one strided subtract, multiply or division
operation for each 64-column chunk. Brcb fetches eight compact FP32 row values
per repeat and copies each payload into one 32-byte block. Compiler-owned
private UB scratch has `R * 8` FP32 elements (`32 * R` bytes); it is allocated
before scheduling and has an ordinary buffer lifetime. Both row-factor operand
orders retain their mathematical order, including `row / matrix`. Division
uses the same native FP32 `vdiv` primitive as the existing count API and
inherits its accuracy and special-value limitations; it does not insert a
reciprocal algorithm.

This path requires a proved static positive row count divisible by eight and
at most 255, a real rank-2 dense matrix with `C` a positive multiple of 64,
compact row state, and 32-byte-aligned starts. The matrix row repeat stride is
`C / 8` 32-byte blocks and must fit UInt8, so the largest selected width is
1984. A row-factor block advances by one 32-byte block for each repeat. Every
chunk uses normal/full mask; its matrix and destination start at column
`0, 64, ...` and process exactly 64 columns of each owned row. Temporary matrix
results have a real `[R, C]` allocation, rather than a flattened alias.
Unconditional consecutive assignments keep Exp and terminal casts at the
whole-matrix count while preserving each expression node and assignment order.
Row state must not alias matrix or output storage. Same-storage matrix operands
require the identical typed physical start. Flattened 1D representations,
shifted views, lane predicates, dynamic or empty row domains, and unsupported
widths retain the existing lowering. Partitioned mixed kernels apply the same
checks to each AIV's actual private row domain.

Brcb retains typed FP32 `tl.access_ptr` effects for its contiguous read of `R`
values and write of `R * 8` blocks. Each strided binary instead retains exact
FP32 `tl.region` operands for its `[R, 64]` matrix/output chunk and `R * 8`
broadcast elements, plus its explicit row stride. The common memory detector
and multi-buffer rewriter consume these regions; independent source and
destination version axes remain singleton region axes. After scheduling and
synchronization, `AscendLowerTileOp` validates this contract for the three row
binary operations and converts their regions to pointers using the existing
buffer/layout remapping. The resulting late pointer extent is a bounded
linear hull `(R - 1) * C + 64`; it does not replace the earlier exact chunk
effects. Only native Brcb codegen reinterprets FP32 pointers as UInt32, matching
the CANN 9.1 b32 interface. These operations use PIPE_V and the shared
RAW/WAR/WAW, version and completion analyses; no hidden event or composite
softmax algorithm is introduced.

Private UB copies and direct dense Parallel assignments preserve source bits.
The 2201 count primitive uses UInt16 OR of the source with itself, including
two UInt16 lanes per FP32 element. It keeps the typed regions and exact element
count in TIR. This avoids the installed UB DMA's whole-32-byte-block transfer
contract. Source and destination origins require 32-byte alignment; shifted
overlap has a diagnostic, while an identical region is a no-op.

2201 Vector issue order does not establish completion of dependent UB accesses.
InsertSync emits `PIPE_V` barriers for the same-pipe RAW, WAR and WAW dependencies
found by the shared region and storage-version analysis. Independent regions
can remain consecutive. This rule is separate from 3510's VF execution path.

Trailing-axis FP32 sum/max and FP16 max reductions use the installed native
repeat interfaces with explicit partial-result scratch and completion before
compact scalar output stores. Sum with FP16 or BF16 input and FP32 output
exposes an ordinary widening cast, then uses the same FP32 partial plan. Its
`clear=False` combine is FP32. BF16 max widens, uses that plan, and narrows the
selected result; `clear=False` remains a separate BF16 max node after narrowing.
These composed paths use per-row aligned UB scratch allocated by the compiler;
users need no auxiliary buffer or cast.
FP16 max with `clear=False` also exposes its combine as an ordinary typed
vector max over aligned scratch. Compact old-output reads and final stores
remain scalar accesses; this avoids the unsupported 2201 scalar half max.

For matching native dtypes, clear-true rank-two rows exactly one 256-byte VL
wide can use one batched count instruction. Proven tight source rows and
contiguous destination results retain exact typed effects. The destination
base keeps the native alignment requirement; internally adjacent scalar
results use stride one. This path needs no reduction scratch or per-row scalar
readback. Wider compact FP32 rows whose width is a multiple of 64 can use
normal-mask native chunks across all static owned rows, followed by an ordinary
balanced Add/Max tree. Compact widths 512..4096 in multiples of 512 prefer
two BlockReduce compactions and a masked row reduction; a smaller two-level
WholeReduce plan remains available when the larger scratch does not fit.
Exact regions, compiler partials and combine scratch remain visible to scheduling; a conservative UB headroom check retains the
original plan when this expansion cannot be admitted. Dynamic/empty owners,
partial or strided layouts and clear-false reductions retain the ordinary row
plan. See [the 2201 reduction contract](ascend_2201_reduce.md) for stride units,
selection bounds, memory effects and numerical limits.

Low-precision sum output remains diagnosed: FP32 accumulation followed by one
narrowing does not preserve low-precision intermediate rounding. The promoted
paths inherit the native FP32 reduction's ordering and special-value limits;
`nan_propagate=True` remains unsupported. This does not establish NaN payload
or propagation behavior, signed-zero tie selection, or subnormal preservation.
Unsupported gathers, predicates, layouts and conversion modes have diagnostics
at the relevant lowering boundary.

## Automatic pipelines and cost evidence

Ascend `T.Pipelined(...)` retains automatic pipeline intent by default. One
joint solver selects issue order, stages and buffer versions under dependency
and storage constraints. `max_stages=4` is the default bounded optimization
horizon; active pipes and the loop trip count can tighten it. It is not a
request to allocate four versions. Explicit `num_stages=0` retains serial
behavior. Explicit positive values and `T.Stage` remain diagnostic controls.
The common TileLang dialect is unchanged.
Plain serial loops and explicit `num_stages=0` without an offset request keep
all tasks at stage zero. They do not search arbitrary cross-pipe epoch offsets.
Explicit positive version bounds retain their existing scheduling policy.

Straight-line sibling lists minimize makespan using the same task effects and
resource exclusions. Weighted data and pipe paths remove implied resource
choices while retaining any stronger issue-interval gap. Tasks with identical
costs and complete weighted data, pipe, resource and owner neighborhoods can
use source order to select one representative of an exact graph symmetry;
distinct independent operations retain their ordering choices. Proven resource
work, release and completion bounds and a feasible source-oriented assignment
bound the objective without fixing the optimized issue order. These bounds use
the supplied model's units, including dimensionless 2201 weights. A solver
`unknown` result reports its reason separately from `unsat` and retains the
existing fallback policy.

Modulo search bounds II by the total issue work of verified resource cliques.
Pipe masks nominate groups; only actual pairwise exclusions establish a
clique, and independent cores are bounded separately. With automatic stage
horizon one, nonnegative distance-zero paths can prove a resource order.
Removing its impossible reverse choice still retains the forward and wrap
gaps. Free stages and manual schedules retain their original choices.
For horizon one, a straight-line relaxation also bounds the residue window
from distance-zero releases and remaining issue paths. Unit completion tails
represent the last integer issue residue fitting below II; they do not change
native completion costs or require instructions to finish within the period.
`unknown` is distinct from `unsat`: trying a larger II after resource
exhaustion is a bounded search policy, not a proof that the smaller II is
infeasible. Such a result is reported as a selected feasible II with unproven
minimality. Manual search exhaustion has a separate diagnostic.

Ordinary automatic 2201 loops with offsets also consider prefetch at the
already selected II, before fixing stages or recalculating versions.
A typed GM-to-L1/UB copy and its forward RAW reader qualify only on the same
concrete core and distinct pipes, with one automatic storage owner and a
version variable. Read-modify-write roles, ambiguous or mixed-core units,
manual schedules and 3510 are excluded. The bounded joint refinement retains
every dependency, capacity and completion constraint while leaving stages and
versions free. It first prefers at least one epoch of consumer lag together
with producer-first physical issue, then fewer allocated bytes and a smaller
stage span. Physical issue follows residue, descending stage at equal residue,
then source index; a stage lead alone is insufficient. Only a strictly better,
independently validated feasible model is accepted. Unknown or unsat retains
the selected feasible incumbent. Start normalization and emitted counts are recomputed from
the selected plan. These are structural preferences in the existing cost
model, not measured load latency or proof of a performance improvement.

Before the solver-based prefetch refinement, one bounded scan can try moving
each qualifying producer to the stage immediately before its consumer while
keeping all issue residues and resource-order choices. Buffer-distance edges
determine any additional versions using exact integer arithmetic, shared with
the final emitted-version calculation. A candidate must strictly improve the
same preference objective and evaluate every original hard assertion to true
without model completion. An incomplete or unsupported model is retained.
This scan adds no solver query and does not change the optimization budgets.

For ordinary automatic 2201 mixed loops, shared memory effects identify forward
cross-core writer-to-reader handoffs separately from reverse slot-release
dependencies. The issue-quality heuristic prefers an independent handoff
producer and its same-core ancestors before an unrelated blocking reader.
Hard same-iteration reachability excludes that reader's own descendants even
through an ambiguous-core intermediate. Native task candidates use the same
instruction and memory capability function as AssignCore, intersected with
lexical ownership; ambiguous placement remains conservative.

After selecting a feasible II and recalculating versions, one bounded refinement
can select different issue residues with stages, emitted version counts and
all original constraints fixed. An unavailable or worse result retains the
feasible plan; verbose diagnostics report the solver status and unknown reason.
Useful prefetch predicates selected by the earlier joint refinement are kept
through this later issue refinement as frozen quality decisions, not new data
dependencies.
These structural preferences do not estimate wait duration and do not establish
optimal overlap. Manual schedules and 3510 retain their existing policy.

Native sub-K loops created by L1 GEMM lowering inherit the enclosing automatic
pipeline intent and optimization bound. Their L0 copies and MADs participate in
the same dependency, resource and version solver. An explicit pipeline setting
at a nested loop forms its own boundary; an explicit zero disables inheritance
within that loop. Clear-accumulator and final-completion expressions keep the
logical loop epoch when tasks receive a stage offset.

A closed native result stream can retain positive static sub-K stage shifts
and deeper scheduled controls when the typed MAD chain, complete C
definition/export and unit boundaries remain valid. The result and its direct
child controls remain unshifted. Each deeper control's incoming stage is
validated against its direct parent's consumed stage bound and offset policy;
its own bound only governs its children. Typed MTE1/MTE2 inputs may also
receive a positive stage inside that validated static nest, provided their
own condition cannot skip a producer occurrence. Zero-stage guarded inputs
retain their existing contract. Any shifted task or control, including a
sibling input subtree, requires one logical MAD task chain. Direct result
children and the complete FIX export remain at stage zero.
Existing recursive schedule lowering supplies fill, drain and logical
clear/final predicates, including an outer cache
prefetch and an inner native-K shift together. A local ready event in a
single-iteration child still needs a lifetime proof across repeated parent
activations: synchronization
keeps its actual leaf endpoints and flattened clock, then checks the parent's
real reverse completion path after parent dependencies are established. That
path determines the event ring width. Unconditional, unshifted ancestors with
positive static extents are required; unsupported activation or an unresolved
parent proof is rejected.
Neither unit completion nor a single child iteration implies whole-MAD
completion or permits unproved event reuse.

For automatic 2201 loops, L0A/B/C version admission normally reserves every
other root storage allocation before solving the current owner. The physical
slot footprint includes layout and alignment; aliases with the same storage
Var are charged once. Previously selected shared rings retain their maximum
width as a hard minimum in later owners, including refinement, version
recalculation and sequential fallback. Fixed version requests with an owner
are reserved before that owner is visited. A manually versioned allocation
already includes its physical version axis and is charged once.

L0A/B can share physical addresses across two closed AIC phases while keeping
their different canonical Buffers, layouts and version clocks. Early admission
requires exactly two root allocations in the same physical scope, FP16/BF16
canonical operands, typed L1 writers and native MAD readers, and separate
direct phase children of one unconditional positive static outer loop. Escaped
or aliased views, opaque or raw Cube accesses, MX scales, mutable activation,
manual physical rings, and a common eligible owner for both allocations retain
the conservative reservation. Disabling shared-memory reuse also disables this
selection.

Two independent canonical FP32 L0C allocations can use the same phase
contract when both request one fixed version. Each result must start with a
full-clear native MAD chain and end with a full C-to-GM FIX export. Admission
requires exactly two root C allocations, no views, opaque accesses, manual
rings or additional C users, and distinct closed phases in one static epoch.
Their individual physical footprints must fit L0C. The scheduler reserves
their maximum footprint; synchronization must then prove both alias
directions before the late byte allocator can share their base address.
Every query consumes this selected physical reservation, including a manual
ancestor of an automatically pipelined child; established ring counts remain
hard lower bounds.

A finite K accumulation is one result lifetime. Its first logical iteration
must clear the full physical C region. The lifetime includes every subsequent
accumulation and the final FIX read, and retains both MAD and FIX completion
endpoints. A dynamic clear remains a possible C read in physical effect
analysis; proving the first-iteration definition does not erase later reads.
Native unit readiness does not establish whole-MAD or whole-FIX completion.
A direct full-clear native MAD followed by its unique full GM export has the
same closed lifetime without a descendant K loop. Its clear predicate must
prove true without substituting result or owner coordinates. Both actual
completion endpoints remain required when checking later physical reuse.
Guarded, partial, escaping or multiversioned accumulations retain conservative
lifetime analysis.

Each Buffer keeps its own canonical shape and M pitch through native MAD and
FIX lowering. Different M pitches cannot be implemented by reinterpreting a
prefix of the larger Buffer. The older same-M early storage coalescing path
retains its separate eligibility rules.

The selected pair adds the existing cross-storage conflict contract at its
common epoch. Ordinary dependency and completion analysis therefore retires
every prior MAD reader before the other phase's MTE1 overwrite, and every
prior FIX reader before the other phase's C write, including the next-epoch
boundary. A dependency retains whether it came from an explicit conflict
between different storages. On 2201 that scope-wide conflict uses the
unconditional lexical epoch domain, so the emitted event covers the same
physical phase envelope as the lifetime proof. Guarded buffer dependencies
and invariant SPR dependencies keep their own domain selection. The enclosing
lexical synchronization remains balanced when a phase has no active data work.
Internal pipeline loads and MADs still use their own slot clocks and dependencies.

A retained child ring can cover a direct parent's WAR for that same storage.
The prepared version clock and physical materialization use the same complete
ancestor iteration count, so an odd child trip count does not reset the slot.
This requires a unique iteration owner, static unguarded positive loops,
full typed GM-to-L1/L1-to-L0 copies or L1-to-L0/native-MAD operand accesses,
and the actual retained RAW and full-ring release protocols. A MAD operand
must be the same complete canonical Buffer as the load destination; its
release is the real whole-MAD `M` completion, independent of partial C-ready
unit flags. Other accesses, counters and unsupported domains
retain the parent obligation. This coverage removes only the redundant storage
WAR; it adds no whole-loop completion edge and cannot prove C phase reuse.
This distinction matters when two operand versions cross a row-part boundary:
the next row part may overwrite a different slot before the previous part's
last MAD completes, while every same-slot overwrite still consumes its own
retained release. Parent owner and physical phase obligations remain separate.

A persistent 2201 local binary handshake in a static positive child loop can
also cover its parent's exact last-to-first boundary. The synchronization
pass requires the same real source and destination tasks, one slot at distance
one, exact first/last activation predicates, and an independently established
reverse completion path. The child protocol keeps its single kernel-root seed
and drain; the entire redundant parent protocol is removed. Its parent-level
completion relation is derived only from this finite protocol certificate.
Dynamic or empty children, shifted stages, mutable counters, extra endpoint
guards and different active sequences retain separate synchronization. The
2201 limit of eight local hard-event slots remains enforced.

A closed local lifetime may be summarized by an unconditional physical entry
and exit outside its data guard. The lifetime analyzer checks both endpoint
directions, zero local offsets, the unconditional parent lexical domain, and
absence of incoming or outgoing values before using this conservative
envelope. The real data predicate remains unchanged. This does not make a
changing predicate invariant or add completion edges: periodic separation
still requires both actual completion directions across lexical epochs.
For an operand loaded outside a repeated read-only child, the final child read
extends that incoming generation instead of creating another generation. This
summary requires a positive nonbreaking child with a one-iteration storage
period and no physical writes to the operand. Every read still needs a covering
must-write, guard implication and actual completion ordering. Counter clocks,
repeated writers and values escaping the child retain their normal analysis.
The summary retains the enclosing read-only controls and rejects reuse with
any allocation accessed inside them. Parent-level leaf ordering cannot stand
in for completion between different iterations within one such control.
A one-iteration control may carry an already validated repeated-child summary
into its parent only when every actual access to that storage is inside that
same control. Direct or sibling accesses prevent this fallback. Promotion
retains the existing guard, incoming/outgoing values and completion endpoints;
the control's extent alone supplies no alias proof.

The resource query charges the maximum of both complete versioned physical
sizes. A query with one variable member constrains that member normally and
checks its fixed mate independently; two variable members are outside this
selection. The internal root marker `tl.ascend_l0_phase_reuse_pairs` carries
the chosen pair to InsertSync, which requires a symmetric alias proof from its
final completion graph before consuming the marker. Missing proof is a
selected-contract error: it cannot silently deploy the independently allocated
sum after solving with the smaller budget. The existing late clique packer
then co-locates the two allocations after layout flattening. No shared logical
Buffer, changed canonical pitch or common version clock is introduced.

Other different storage Vars remain reserved even if a later lifetime proof
could reuse them. L0C coalescing runs before admission, so proven common storage
identity is already reflected in this accounting. The remaining conservative
reservation can reject a reuse plan and does not claim optimal allocation.
This boundary covers automatic L0A/B/C selection; it is not a complete bank
validator for manual kernels or kernels without an automatic loop. UB, L1 and
3510 retain their existing policies. Final physical allocations and native
completion still need their own validation.

Compiler-created GM handoff storage also uses this buffer-version and epoch
protocol. Its root provenance annotation,
`tl.ascend.compiler_workspace_storages`, registers allocation data Vars;
external tensors and unregistered user global allocations are not automatic
candidates. A version slot includes the complete grid capacity, so a workspace
`[groups, padded_rows, padded_columns]` becomes
`[versions, groups, padded_rows, padded_columns]` with the same logical transfer
regions. Ordinary dependencies determine readiness and when a slot may be
reused, including collective completion for an empty AIV partition.

GM versions do not consume UB/L1/L0 capacity. Each workspace has an independent
int64 byte-representation constraint in the solver, matching the runtime's
owning byte Tensor. This constraint does not estimate device memory capacity
or sum separate allocations. Fixed version counts are checked before byte
multiplication, and whole-byte slot strides are computed without an
intermediate conversion to bits that could overflow.

The 2201 cost model uses **dimensionless service/order weights**. Its function
attribute is
`tl.ascend.cost_model = "dav-2201.service_weights.estimated.v1"`. The objective
compares relative resource pressure and buffer choices in an ordinary automatic
pipeline; it predicts neither asynchronous completion latency nor kernel time.
3510 cycle estimates and INI rate fields do not supply measured 2201 costs.

The native FP16/BF16 reference is a 128×128×128 FP16/BF16-to-FP32 MMAD, a full-M
128×128 A Load3Dv2 and a full-physical-N merged 128×128 B LoadData2D, with NZ
L1 input and independent two-slot destinations. Current CANN 9.1/dav-2201
measurements cover one AIC and a 20-AIC grid. Counter windows use raw SYS_CNT
ticks and explicit target-pipe completion events. Loaded-chip profiler frequency
is reported separately; no SYS_CNT period or AIC-cycle conversion is inferred.
Raw measurements, binary identities, frequency records and residuals remain in
external experiment artifacts.

The coarse reference weights are MMAD:A3D:B2D = **4:1:2**. This rounded,
measurement-informed ratio is an estimated scheduling heuristic, not the raw
timer coefficients or an exact measured latency. MMAD scales by physical
16-element-fractal M×N×K work; these native loads scale by their actual padded
destination payload. Source dependency bounding regions and allocation-wide
K-cache pitches do not represent transferred bytes. The estimator and native
copy lowering share the merged-B2D descriptor predicate. Other geometries are
extrapolated estimates. Both calibrated precision paths passed independent
numerical gates; BF16 A3D preserves operand bits through the CAPI's half storage
view, while MMAD consumes BF16. Other precision paths, segmented/unidentified
copies, dynamic work and all other unknown leaves retain uncalibrated ordinal
one; one never means one cycle/tick. See
[the service-weight definition](../ascend/2201_service_weights.md).

Task cost overrides, debug output and scheduling plots use the same model
units. The existing fields named latency and II carry abstract scheduling
distances on 2201; a measured cycle/tick override cannot be mixed into these
weights. Typed dependencies, versions and native synchronization still establish
actual completion and safe reuse independently of this objective. Automatic
buffer selection and competitive kernel performance require separate acceptance.

## Build and runtime

A non-CUDA build initializes TVM and its pinned FFI dependencies without
fetching optional CUDA submodules. An external scikit-build `build-dir` can
produce a wheel installed into one environment. Run installed-package checks
outside the source checkout, and confirm the package and native library paths;
an old editable installation or `PYTHONPATH` override is not a new build.

The ACL stub distinguishes an absent runtime from the valid null-valued
`RTLD_DEFAULT` lookup scope. It excludes its own DSO during resolution,
preserves interposed launch callbacks and retains their providers until process
exit. Missing required binary/launch symbols are diagnosed before use. Torch
NPU owns allocation and the current stream/device context.
The adapter passes NPU tensors through a private zero-copy argument type with
a stable DLPack Exchange API. Its conversion callbacks delegate to Torch and
query Torch's current NPU stream on each call. This also covers scalar-only
kernels' zero-element allocator anchors. Torch's global Tensor API remains
unchanged, so TVM-FFI's per-type callback cache cannot make behavior depend on
whether a CPU Torch tensor reached FFI before the first Ascend invocation.

Ascend host lowering converts compiler workspace allocations to runtime
allocation bindings before host storage legalization. This preserves each
allocation's byte count and the same pointer in kernel arguments and release
calls, including BF16 handoffs. Device functions retain their mathematical
dtypes. Other backends retain their existing host pass order.

Each native device entry calls `asc_init()` and drains issued instructions with
`asc_sync_pipe(PIPE_ALL)` before returning. This kernel lifetime boundary does
not replace completion between scheduled tasks or mixed cores. VF and
`call_extern` helpers do not establish a separate kernel lifetime. The
[CANN matrix programming example](https://asc.gitcode.com/guide/operator_practice/simd_operator_impl/matrix_basic_api/decoupled_mode.html)
requires initialization at entry and the all-pipe wait at exit.

On 2201, the entry additionally sets CTRL bit 48 to one while preserving the
other bits. The installed CANN 9.1 initialization does not establish this
state; its `SetSaturationFlag<FLOAT>(false)` implementation sets bit 48.
The register contract specifies saturation of FP16/BF16 arithmetic and
narrowing with bit zero, and preserving special values with bit one. This
initialization is needed for ordinary vector arithmetic and casts. Fixpipe
Inf/NaN behavior has not been verified on a device, and public Fixpipe API
descriptions differ from the general register description; the backend does
not claim that this initialization alone establishes its special-value
behavior. See the CANN 9.1
[CTRL register contract](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/special_register_access/SetCtrlSpr_ISASI.md)
and
[saturation-state API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/special_register_access/SetSaturationFlag_ISASI.md).
