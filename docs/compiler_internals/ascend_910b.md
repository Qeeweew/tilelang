# Ascend 910B compiler contracts

For the reading order and source map, start with the [910B developer guide](../ascend/README.md).

Use the Ascend dialect (`tilelang.ascend.language`) and an explicit target:

```python
target = {"kind": "ascend", "arch": "dav-2201"}
```

`ASCEND_NPU_ARCH` supplies a default when creating a target. The frontend fixes
that default into the target before lowering and binary caching. Explicit
`arch` or `mcpu` takes precedence; when both are given they must agree. Without
an explicit architecture or environment default, the target uses `dav-3510`.

For the synchronization algorithm, proof categories and source reading order,
see the [InsertSync reading guide](ascend_910b_insert_sync.md).

Event lifetime planning returns an updated synchronization point or a retry
reason; it does not mutate analyzer retry state. Pending and proven
parent lifetimes are mutually exclusive states. Hardware allocation owns its
point-to-allocation mapping and reads synchronization points without modifying
them. Selected owner RAW coverage is indexed once per analysis attempt, with
storage, endpoints, scope, distance and core direction in the key; coverage does
not introduce an all-participant completion edge.

Site collection resolves parent IDs after canonicalization; per-scope analysis
reads the sorted site interval instead of enumerating task/domain keys again.
Input and result owner transfers share release validation and event construction.
Ordinary and owner flags share explicit reservation masks and contiguous-range
search, preserving low-ID versus high-ID preference and result-before-input order.


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

The launch grid counts groups in a mixed kernel and individual AIVs in a pure
Vector kernel. `T.Vector(vector=1|2)` declares the active width inside a mixed
group, with one consistent width per kernel. A pure Vector kernel requires
`vector=1`, whose hardware SID is zero; use `T.Kernel(N)` and its block index
for N independent AIVs. A two-participant SID domain cannot create another
AIV in a pure launch. Final code generation resolves the actual core regions,
active width and SID domains together. Automatic scheduling retains explicitly
declared widths when materializing those regions, including pure Vector output.

## Lowering and stage contracts

The frontend expresses tile sizes, traversal, cache reuse and the algorithm.
BufferRegion and Layout preserve logical shape, valid regions and physical
ordering. Native 2201 L0A uses ZZ order; it is distinct from L1's NZ layout.
The input capability boundary diagnoses unsupported operations before layout
inference or scheduling.

Before layout, mixed partitioning collects explicit CUBE regions, real
instruction/storage requirements, and unscoped cross-core endpoints whose pipe
pins execution to AIC. It does not interpret a synchronization endpoint mask as
an instruction requirement: `PIPE_ALL` fences the enclosing core and cannot
create another core. Scoped synchronization keeps its declared execution
region. This early intent is not final core placement; AssignCore retains input
availability and explicit-region checks. Both use the same exclusive-pipe core
facts, while S and MTE2 require context or storage to resolve their owner.

The [ordinary GEMM example](../../examples/ascend/gemm_910b.py) chooses its
output traversal in Python. Its default launches one logical block per output
tile. Optional `num_cores` bounds the logical launch and distributes output
tiles round robin through an ordinary serial loop with a per-core finite
extent; it does not select a physical NPU. Both paths use the same automatic
`T.Pipelined` K loop. `cache_a` keeps that output tile's full K in L1;
alternatively, `a_cache_k` chooses a bounded A cache width that is a positive
multiple of `tile_k`. Ordinary nested pipelines load A once per cache block
and stream B panels, skipping absent final panels. Both caches remain private
to each output tile and reload for the next output. Buffer versions and stages
are selected by the compiler; expressing nested pipeline intent alone does
not guarantee that a parent prefetch overlaps its child loop.
The compiler retains operand-slot completion and
last-MAD-to-FIX/FIX-to-next-clear completion when physical storage is reused.
Static leading operand planes can also share a bank with an independent native
ring after their last whole-MAD readers retire. A complete result loop at a
consumed incoming parent stage retains its full unit-stream transaction through
fill/drain. See [fixed operand planes and shifted streams](ascend_2201_fixed_plane_credits.md)
for admission, physical credits and completion obligations.

### Loop normalization and physical GEMM

Before scheduling, a serial unit-step loop with extent one can be replaced by
its body. Ordinary automatic pipeline intent alone does not prevent this:
the annotation set must be exactly `tl.ascend_auto_pipeline=True` and a positive
integer `num_stages`. Its minimum retains evaluate-on-entry semantics. Manual
annotations, explicit stages, loop-control targets and descendants with other
annotations keep their control boundary. A nested loop with only automatic
pipeline intent retains its own domain after substituting the one-trip parent;
this happens before version ownership and scheduling. The final child MAD
therefore remains visible to the following accumulator export without an
extra parent clock. Larger loops are not expanded.

L1 GEMM is expanded before layout inference into explicit L0 allocations,
L1-to-L0 copies and ordinary L0 GEMM tasks. A bounded sub-K loop keeps IR size
independent of K. After the capacity bound, lowering prefers equal 16-aligned
chunks when they cover K exactly with the same number of native MADs. This
avoids a separate tail allocation without increasing the native call count;
it does not establish a measured latency advantage. Other splits retain their
final partial chunk and its compact allocation. This selection does not grant
cross-phase aliasing: scheduling and synchronization retain their existing
epoch, version and completion contracts. The
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

### Dependency footprints and epochs

Dependency analysis retains each conflicting task pair's read/write modes and
the region pairs that formed its RAW, WAR or WAW hazards. A pair may carry
several hazards without adding duplicate scheduling edges or synchronization
points. Reversing owner order reverses both endpoints and their access records.
These regions remain conservative footprints, not exact coverage or completion
proofs. SPR ordering records name the register and have no memory regions.
The records belong to one analysis snapshot and are rebuilt after IR rewrites.
The same snapshot records complete read/write roles on the dependency storage
for both leaf endpoints and their enclosing nodes, including nonconflicting
regions. Whole-node roles include control-header accesses. Handoff and prefetch
selection consume these roles without confusing them with individual hazards;
reversing an owner dependency swaps both its roles and conflict records.

Within one ordered node-pair analysis, region comparisons share the prepared
task contexts: iteration renames and merged constraints. The key
includes both endpoints, distance, a forward-distance bound and whether distinct
physical owners allow equal coordinates. Ordinary carried edges keep ancestors
fixed and require the consumer coordinate to follow the producer; independent
owner epochs rename the full nest and may have equal coordinates.
While the exact leaf task pair remains current, its context can also retain
the populated analyzer for each distinct distance/owner relation. Region
queries only simplify and prove; they add no assumptions. Each query still
creates fresh per-side mutable-read snapshots. Changing either leaf releases
all active analyzers, bounding solver retention even in a large control pair.
Conflict hints still decide before storage/alias checks. Prepared contexts and
active analyzers do not survive another node pair, child list, IR rewrite or
sync attempt; no region result or cross-pass proof is cached.

Unowned persistent state retains its ordinary region and guard semantics. For
a backward stage edge, shared analysis proves the shortest potentially
conflicting forward prefix needed by that stage separation, using bounded
queries rather than enumerating all iterations. If that prefix is disjoint,
the edge retains a conservative lower bound for later conflicts. This is a
schedule constraint, not instruction completion or an inferred state ring.
Task-generation rings retain their selected version distance. A grouped query
state may therefore use `local_pair % slots` without equating its lifetime
with every task write. Scheduling and synchronization consume the same
dependency record; owner, guard or region rewrites invalidate the analysis.

ScheduledTIR decoding and physical alias analysis share one task-effect scope:
a leaf owns its statement and direct guards; a control header owns only its
minimum, extent, optional step and direct guards. Child bodies and descendant
guards retain their own owners. Guard conditions and expression-valued attribute
nodes contribute memory effects; attribute values have a separate core-input
availability contract. Each fragment is analyzed independently. Logical regions
and scalar accesses are stored on the task, while physical footprints, definite
writes and opaque accesses are freshly derived for lifetime analysis. Moving a
guard between active owners refreshes both logical access snapshots. A terminal
rewrite followed immediately by encoding relies on the next decoder instead of
maintaining a second cache or reanalyzing the whole tree.

### Native operand selection

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
widths on the same side can select one invariant group, leaving the other
copies unchanged. Selection ranks the static upper bound on saved Load2D
descriptors, using the ordinary lowering recipe and finite loop trip counts.
This is an instruction-count objective, not measured latency or bandwidth;
unknown bounds, overflow and ties retain the original conflicting-bank path.

Load3D configuration and MMAD direction have separate register-state contracts.
Known native Ops explicitly declare which state they preserve; an issue pipe
or a name prefix grants no such permission. Each setter writes its own state
and preserves the other. External calls, opaque library operations and unknown
Ops remain unmodeled. Load3D synchronization separately validates and permits
its unique root initializer; closed MAD/FIX streams consume the direction
contract while retaining their complete-stream and explicit-state checks.
Live reconfiguration is unsupported. External and unmodeled opaque calls
prevent selection, and an
already selected kernel diagnoses them at synchronization lowering because
they could overwrite the invariant state.

The selected path requires synchronization lowering and is disabled when
automatic scheduling is disabled. A hand-created PerCore reader must have one
flat Task load leaf; internally compound, repeated or conditional groups
retain their existing diagnostic. Descriptor count reduction alone is not
evidence of lower execution time. See the native operand-load contract in
[2201 physical tiles](ascend_2201_physical_tiles.md).

### GM handoffs and mixed ownership

Logical L0C-to-UB and UB-to-L1 transitions become ordinary copies through
compiler-owned GM workspace before layout or ownership is assigned. Each
kernel group has a separate workspace slice, and each transfer retains its
actual valid region. The padded allocation does not enlarge the valid input.
An L1 consumer receives explicit initialization for any unwritten tail.
Handoff lowering first records a legal logical zero-fill region. Storage
normalization then extends its complete matrix axes to the established
physical allocation and consumes the padding intent. Partial slices and
empty domains retain their original bounds throughout these phases.

On 2201, host lowering aligns compiler-allocated global workspace starts to
512 bytes. It reserves up to 511 extra bytes through the existing producer
DLPack allocator, passes an aligned interior pointer to the device kernel,
and frees the original allocation handle. Stream ownership and asynchronous
release use the existing runtime path. The allocation size check includes
this reservation. This is a performance policy supported by FIX-output
address experiments, not a minimum legal alignment for arbitrary GM tensors.
The 3510 allocation path and external tensor arguments retain their existing
behavior.

Generated GM handoffs round both matrix capacities to complete 16-element
axes. Their FP32 planes are therefore multiples of 1024 bytes, and FP16/BF16
planes are multiples of 512 bytes. Group slices and automatic version pitches
cover whole physical planes, preserving 512-byte-aligned group/version bases.
Subregion offsets and arbitrary internal strides retain their original
alignment. Valid tails and AIV row partitions remain ordinary regions; they
are not expanded to align every reader address. For caller-owned FIX output/workspace tensors,
the caller controls the actual base and physical strides. The compiler does
not silently shift an external pointer or copy the entire output merely to
meet the preferred alignment.

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
| Version-owner analysis | Eligible task generations and their loop/counter clocks, separate from persistent state regions |
| AutoSchedule | Selected task order, stages and storage version counts; common loop/list validation |
| PrepareMultiBuffer and ResolveCore | Concrete version epochs and core streams |
| InsertSync | Completion and reuse dependencies, including finite fill/drain boundaries |
| MaterializeMultiBuffer | Physical addresses from the selected version plan |
| LowerScheduledTIR | Stage-shifted indices, accesses, guards and sync; physical fill/drain interval |
| Codegen | Native instructions for the established layout and schedule |

Codegen does not partition logical work or choose buffer versions. Resource
validation and synchronization consume the same selected plan.

When automatic scheduling is disabled, `InferBufferAliases` proves pairwise
storage reuse from the explicit native instruction stream. This is separate
from `T.Stage`, which uses the common scheduled plan and synchronization path.
The analysis binds variables by their actual For/immutable Bind/cthread
identity and instantiates finite execution contexts; names do not identify
threads or events. Program order belongs to one physical core, AIV participant
and pipe. Only a real flag handshake or barrier supplies completion across
pipes or cores. Private UB lifetimes are checked per AIV; shared L1 needs all
participants. A 2201 mode-2 acquire requires both AIV contributions for the same
flag generation. Unknown event identities and unbounded or over-budget loops
retain may-access lifetimes and cannot manufacture a completion edge. Explicit
unpolluted synchronization outside a summarized loop can still prove whole-loop
retirement.

Within `PrepareMultiBuffer`, owner analysis retains the active epoch guard and
its definition position until guard canonicalization finishes. Equivalent
guards use one dominating snapshot, and both counter emission and serialized
epoch metadata consume the finalized plan. This avoids analyzing an owner
again or recovering a just-computed guard from annotations. These facts are
local to the preparation pass; later passes reconstruct the existing plan
from annotations after counter insertion or an IR round-trip.

### Synchronization planning and proof ownership

`InsertSync` keeps optional protocol selection separate from each analysis
attempt. An attempt builds fresh completion proofs and flag allocations and
reports its failure; one selection policy applies the corresponding fallback.
Proof failure, local-flag pressure and owner-event pressure keep their distinct
priorities. A retry can retain proven owner transfers while removing only those
without a release witness. Undoing a unit stream restarts from the tree captured
before unit rewrites and root markers, rebuilding candidates and epoch domains.
Only the successful attempt's proofs and analysis sites reach materialization.
An attempt returns either value-only failure observations or one complete plan.
Failed sites and partial allocations are destroyed before the selection policy
changes a protocol. The successful plan owns the analyzer and the allocator
that borrows its sites, together with owner events; the allocator is destroyed
first. It permits moving construction, not reassignment. Mandatory alias and
event-capacity gates complete before publication. Outer epoch domains, selected
protocols and the root remain valid until materialization consumes the plan.

The kernel driver separates three lifetimes. `KernelSyncPreparation` owns the
scheduled tree, pre-unit retry snapshot, storage/epoch facts and candidate
protocols; it is neither copyable nor movable because epoch analysis borrows
its multi-buffer plan. `PlanKernelSync` retries optional selections against
that preparation and returns either a complete `KernelSyncPlan` or value-only
root options. A root retry destroys the old preparation before rebuilding the
saved input in the driver's next iteration; it does not recursively retain
old trees and proof contexts. `MaterializeKernelSync` accepts only a complete
plan, publishes accepted aliases/diagnostics, then emits synchronization and
finite fill/drain. It performs no candidate selection or retry. The dependency
proof rules, fallback priorities, event allocation order and 2201/3510 protocol
choices remain independent of this orchestration boundary.

Within an attempt, `SyncAnalyzer` drives recursive dependency/completion
analysis. Once retained sync endpoints are fixed, `ScalarAdmissionProof`
certifies their closed Scalar-wait phases once and stores the result on each
`SyncPoint`. Flag-width reduction and event sharing consume the same
width-independent certificate: width reduction additionally requires the
lexical admission clock, while sharing requires one-bit, stage-zero phases.
There is no separate sharing-mode proof or certificate cache.
A matching V-to-S event still proves completion of the Vector operations
covered by its notification. The extra inference that Scalar consumes an old
flag before dispatching a later notification only bounds event lifetime; it
adds no data-completion edges and does not authorize storage aliases.
`FlagReuseAnalysis` reads finalized sync points, insertion sites and each
scope's dependency closure, records reusable event pairs, and promotes visible
sites across scope boundaries. Same-iteration and carried-channel pair checks
use the same interval-order test with opposite endpoint orientation; mixed
channels retain their separate handshake contract. It cannot change
dependencies, choose fallback protocols or allocate physical flags. Proofs and
results belong to one attempt and are discarded on failure. The site registry
and sync-point vector outlive these readers; appended points keep stable IDs.


Parent storage obligations consume one `ParentMemoryCycle` certificate per
analysis scope and storage plan. Its unique full writer/reader and retained
ready/release ring are proved once, then matched against individual dependency
pairs. Lexical cycles cover WAR and operand WAW; counter cycles additionally
cover their killed incoming RAW. Both use the same retained-ring matcher,
while rectangular lexical traversal and actual counter advancement keep their
separate admission proofs. These certificates never add parent completion
edges and are discarded with the analysis attempt, including cached failures.

Lexical event-clock construction shares immutable-bound and positive-step
validation. Each consumer still checks its own scope, signedness and active
interval. Binary-proof failure has one retry/error exit; private GM cuts may
trigger a fresh ordinary-release attempt, never weaken a failed condition.

The early L0 max-budget choice is a required physical-alias contract, not
evidence that storage can already overlap. A typed storage-pair snapshot owns
its annotation decoding and exact conflict-list identity. It keeps pair order
and marker presence, including an explicitly empty selection, without retaining
early phase-node pointers across rewrites. Each synchronization attempt checks
all selected pairs against that attempt's bidirectional alias facts once.
C-byte retirement, private GM handoff fallback and final publication consume
that same result. Retries rebuild it; only successful publication consumes the
marker. Operand credits still require the complete, unchanged scope conflict
list, and no selected pair supplies its own completion or alias proof.

### Core ownership and Fixpipe

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

### Vector lowering

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

Each normalized vector assignment first builds a local instruction plan with
selected typed destinations, actual scratch buffers, private ownership and
frozen operand/count expressions. CSE and row-broadcast caches end at that
assignment. Emission consumes the completed plan without traversing the source
expression, choosing destinations or allocating scratch. Prefix selection keeps
its mask writes and native operation together in the real chunk epoch; a
single chunk remains inline. The records stay within this lowering and use
existing TIR for scalar statements and regions; downstream passes consume TIR.

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
normal-mask native chunks across bounded owned rows, followed by an ordinary
balanced Add/Max tree. Compact widths 512..4096 in multiples of 512 prefer
two BlockReduce compactions and a masked row reduction; a smaller two-level
WholeReduce plan remains available when the larger scratch does not fit.
The reduction pass owns the actual partials and non-final combine buffers;
VectorTiles consumes explicit binary assignments. Candidate headroom uses
their shared physical footprints, without predicting later scratch. An
independent buffer with an explicitly fixed compiler version count reserves
that many aligned logical slots, rather than disabling native row reduction.
Manual physical rings retain their own slot-padding rules. A versioned native
reduction operand still needs its late layout and keeps the ordinary fallback;
unknown footprints and insufficient headroom also retain that fallback. Bounded
dynamic rows use static maximum capacity and guarded active effects; unknown
row bounds, partial or strided layouts and clear-false reductions retain the
ordinary row plan. Resource planning and final allocation still validate all
subsequent scratch. See [the 2201 reduction contract](ascend_2201_reduce.md) for stride units,
selection bounds, memory effects and numerical limits.

Low-precision sum output remains diagnosed: FP32 accumulation followed by one
narrowing does not preserve low-precision intermediate rounding. The promoted
paths inherit the native FP32 reduction's ordering and special-value limits;
`nan_propagate=True` remains unsupported. This does not establish NaN payload
or propagation behavior, signed-zero tie selection, or subnormal preservation.
Unsupported gathers, predicates, layouts and conversion modes have diagnostics
at the relevant lowering boundary.

## Scheduling

Automatic, Stage-only and complete manual plans share dependency, version,
capacity and completion contracts. See [scheduling contracts](ascend_910b_scheduling.md)
for loop annotations, the common validator, solver boundaries and the cost
objective. The [service-weight definition](../ascend/2201_service_weights.md)
is the authoritative description of the dimensionless 2201 cost model.
Measured performance is documented separately in the
[910B performance report](../ascend/910b_performance.md).

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
