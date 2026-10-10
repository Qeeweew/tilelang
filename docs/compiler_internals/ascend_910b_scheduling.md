# Ascend 910B scheduling contracts

This is the scheduling part of the [910B compiler contracts](ascend_910b.md).
For measured kernel performance, see the [performance report](../ascend/910b_performance.md).

## Loop intent and common validation

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

Manual plans use the same `T.Pipelined` entry: pass a positive buffer-version
bound (for example `num_stages=3`) and
`annotations={"enable_offset": True, "tl.ascend_preserve_issue_order": True}`.
The explicit bound retains the manual `T.Stage` offsets and limits buffer
version selection, not the largest stage index. A false issue-order annotation retains Stage-only
cross-pipe search. No separate manual-loop API is needed.
`with T.AIVPrivate():` collects new root UB allocations and emits the existing
`tl.ascend_aiv_private` membership list. Use one scope for the complete private
component declarations, outside control flow; non-UB and empty declarations
are rejected. Neither frontend declaration adds a compiler pass or changes the common
TileLang dialect. Existing raw annotations remain supported.

Automatic, Stage-only and complete source-order loop plans share a native
numeric contract before stages, timestamps or buffer versions are written to
IR. It checks the selected counts against data/completion edges, resource and
owner wrap gaps, pipe/Let order and residual storage capacity. Complete manual
order still uses deterministic completion without a solver search. Partial
manual completion may translate all stages by one common whole-period origin;
materialization keeps the requested stages. Automatic origin selection checks
its proposed cut with the same contract. An unchecked or malformed fallback is
rejected rather than repaired during materialization.

Each loop query first builds one immutable storage budget from actual allocation
footprints, eligible owners, fixed and previously selected rings, root
reservations and admitted L0 pairs. It owns the version-variable IDs, slot
bytes, minimum counts and residual capacity groups used by dependency distances,
manual completion, solver serialization and native validation. Compiler
workspaces have independent byte-representation bounds rather than a shared GM
capacity. The budget is rebuilt for each query; only a validated plan publishes
selected versions for later parent queries.

## Issue resources and solver search

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

Before a straight-line result changes timestamps or sibling order, native
validation consumes those same original weighted data, resource, owner and
manual pipe-order constraints. It checks nonnegative starts and representable
completion coordinates; repeated owner pairs retain one common orientation.
It does not rerun dependency analysis or impose the solver's search reductions.
An invalid automatic result keeps source order and zero straight-line metadata;
an invalid manual result reports the failed contract. Source-order fallback is
not a graph topological sort.

Issue resources are `(core, pipe)` pairs. AIC and AIV MTE2 engines and Scalar
units have separate occupancy; using the same pipe name does not serialize
their work. AutoSchedule derives these pairs from each leaf's capability and
lexical ownership before final core assignment. Compound nodes union their
leaf pairs, without inventing every combination of their aggregate cores and
pipes. The same resource masks govern exclusions, manual pipe order,
bounds on resource work and disabled-offset windows. The automatic stage horizon
still counts pipe kinds as a conservative search limit. HBM contention remains
a separate scheduling policy, and all data, storage release and collective
completion dependencies remain hard constraints. Removing a false pipe
exclusion alone does not prove useful C/V overlap or a latency improvement.

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
If automatic free search finds no feasible model, a source-oriented DAG can
provide one single-stage candidate. It sets physical minimum ring widths and
checks every original hard assertion without model completion, including both
resource arcs and carried buffer dependencies. This witness does not constrain
free search, replace an existing SAT result, or prove minimum II. Manual
schedules do not use it.

## Prefetch and mixed-pipeline refinement

Ordinary automatic 2201 loops with offsets also consider prefetch at the
already selected II, before fixing stages or recalculating versions.
A typed GM-to-L1/UB copy, or L1-to-L0A/B native load and its MAD leaf reader,
qualifies only on the same concrete core and distinct pipes, with one automatic storage owner and a
version variable. Native loads include the selected load3D operation and
stop at an asynchronous Cube leaf, without promoting an opaque compound.
Read-modify-write roles, ambiguous or mixed-core units,
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

A static, unguarded 2201 AIC automatic child can expose its first-to-last
issue interval per core/pipe and HBM resource to its parent. Each interval
reserves internal gaps; the parent still emits the complete child as one
source block. Forward, wrap and self conflicts follow that physical order.
Completion latency and data dependencies remain unchanged, and the child's
selected buffer versions remain charged to the parent's storage capacity.
Scalar/device effects, guarded or dynamic children, manual scheduling and
3510 retain the conservative envelope. The emitted plan is checked against
the original hard constraints; an unverified fallback is rejected. These
issue summaries permit parent prefetch decisions but do not establish
runtime overlap or predict elapsed kernel time.

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

## Native result streams

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

## Physical storage admission

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
direct phase children of one unconditional bounded outer loop. Its zero-origin,
unit-step extent must be nonnegative, finite and immutable on loop entry;
it may depend on launch coordinates and pure lexical bindings. Empty groups
execute no data epochs. Calls, mutable reads and unresolved parameters do not
establish this contract. Collective control still requires the same extent
on the AIC and both AIVs in each group. An
explicit offset loop may place those phases at different logical stages.
Before scheduling, ordinary automatic pipeline intent may name that same
bounded epoch. Prepared resource proofs require the intent to have been
consumed into the validated plan; an unconsumed search request is not a proof.
The shared offset policy reads automatic intent and an optional explicit
override. AutoSchedule retains the selected `enable_offset` value after
consuming intent, so later ownership and reuse analysis uses the same clock.
Selection records the storage pairs on that epoch as well as the root;
shared dependency analysis treats their physical lifetimes as exclusions,
rather than inventing a logical value flow between unrelated allocations.
The final order and stages determine the directed completion distances. Escaped
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
Shifted phases retain independent C allocations when their sum already fits;
pressure-driven selection still requires the complete alias proof. A real
completing pipe's direct order may have a negative logical iteration distance
between different stages. Lifetime analysis retains that exact physical
order alongside its nonnegative composed graph. Issue-only MTE/FIX order
does not provide this completion fact. In the selected C-byte protocol the
matching full FIX first completes into the real M drain, which then precedes
the next phase's MAD; unit readiness alone cannot shorten the lifetime.
The same full-result contract also applies when the two unaliased C
allocations fit independently and have no additional user conflict/alias
declarations. Keeping separate C bases does not require an extra whole-M
entry barrier. Optional late aliasing still needs its ordinary completion
proof; A/B credits still require each whole MAD to complete.
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

For a selected canonical L0A/B pair with unique full loads and MAD readers,
physical byte credits can replace the whole-phase envelope. Equal pitches
retain one credit per slot. Different version pitches are partitioned at
common byte boundaries: a larger load acquires every interval it overwrites,
and its last whole MAD releases each interval. Both logical rings keep their
original layout and generation clocks. Fill/drain and inactive epochs use the
same finite seed/drain protocol. Credits consume actual local `M_MTE1` event
capacity; they do not invent completion edges for disjoint slots or FIX.
The final alias and capacity gates remain mandatory. Selection facts are
consumed after successful synchronization, before later IR rewrites.

Prepared rings are not limited to two versions. A three-version operand ring
uses the same per-byte-interval credit protocol, with its original generation
index and stage shift. Admission checks the complete versioned byte footprint;
flag allocation checks the combined credits of both L0 banks and other
`M_MTE1` protocols against that eight-ID bank. A larger legal ring does not relax
the full typed load, finite reader, ownership or whole-MAD completion proof.

An iteration-versioned operand generation can serve one MAD reader site
inside a finite serial reader nest. Every intervening loop must be positive,
static, unguarded and at stage zero; the unique full load and all reader
occurrences use that generation's selected slot. Its credit returns after the
last occurrence's whole-MAD completion, rather than after the first reader or
unit result readiness. This permits an ordinary N-chunk traversal to load K
once for several query owners while keeping each owner's A operand separate.
Counter-owned generations retain the direct load/reader/commit contract.
Additional reader sites, independently guarded readers and unmodeled accesses
do not acquire this proof.

## Parent obligations and event lifetimes

A retained child ring can cover a direct parent's WAR for that same storage.
The prepared version clock and physical materialization use the same complete
ancestor iteration count, so an odd child trip count does not reset the slot.
This requires a unique iteration owner, static unguarded positive loops,
full typed GM-to-L1/L1-to-L0 copies or L1-to-L0/native-MAD operand accesses,
and the actual retained RAW and full-ring release protocols. A MAD operand
must be the same complete canonical Buffer as the load destination; its
release is the real whole-MAD `M` completion, independent of partial C-ready
unit flags. Other accesses, counters and unsupported domains
retain the parent obligation. A uniform outer stage on the native operand's
child translates this same slot clock and its events together. The closed
ready/MAD/release chain also covers that writer's parent WAW self-dependency.
This coverage removes only redundant storage obligations; it adds no whole-loop
completion edge and cannot prove C phase reuse. See
[operand slots](ascend_2201_operand_slots.md#parent-obligations-for-one-operand-ring).
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

The local capacity is eight IDs per physical core and directed HardEvent;
`V_S` and `S_V` use separate banks. Allocated ranges are not a proof of the
minimum simultaneously pending notifications. A real `V_S` wait blocks Scalar
dispatch, so a closed notify/wait pair in one physical activation needs one ID
even when the reverse data-completion distance is larger. The same lifetime
argument covers a distance-one wait-before-notify boundary around one control,
with its one startup seed and final drain. InsertSync checks the actual sites,
activation, stage and lexical clock before reducing that channel's width.
Multiple startup seeds retain distinct IDs. These flag-lifetime certificates
do not add data-completion edges or change storage versions; cross-channel
sharing still needs its separate nonoverlap proof.

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

## Compiler workspace versions

Compiler-created GM handoff storage and explicit `T.alloc_workspace(shape,
dtype)` allocations use this buffer-version and epoch protocol. The public
allocator describes one logical generation, including grid and owner axes;
it does not register an external parameter or change a caller-owned ABI.
Multiple allocator calls merge storage identities in the frontend, and
LowerGmHandoff canonicalizes that declaration before owner analysis. Its root provenance annotation,
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

## Cost-model units

The [2201 service-weight definition](../ascend/2201_service_weights.md) owns
reference geometries, the MMAD:A3D:B2D ratio, scaling and uncalibrated cases.
`tl.ascend.cost_model = "dav-2201.service_weights.estimated.v1"` identifies this
model. It predicts neither asynchronous completion latency nor kernel time;
3510 cycle estimates and INI rates are not measured 2201 costs.

Task overrides, debug output and scheduling plots use the same dimensionless
units. Fields named latency and II are scheduling distances; measured
cycle/tick overrides cannot be mixed into them. Typed dependencies, versions
and native synchronization establish completion independently of the objective.
A favorable objective is not a performance acceptance result; see the
[measured scheduling comparisons](../ascend/910b_performance.md#自动流水同组证据-b).
