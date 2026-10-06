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

After this normalization, native Cube instruction selection may replace a
2201 nontransposed A copy with Load3Dv2. The selection requires canonical
FP16/BF16 L1 NZ and L0A ZA layouts, one complete physical M tile, a compact
aligned sub-K tile and proven allocation bounds. Its typed operation retains
the established physical read/write regions through scheduling and versioning.
Only late lowering turns them into pointers and native element descriptors.
Annotated, conditional or explicitly atomic copies, incompatible layouts and
unsupported geometry retain the 2D path.

Selected loads in one kernel must share one static FMATRIX configuration.
An explicit root AIC/PIPE_S task initializes FMATRIX, repeat, padding and
boundary registers once. The loads read that register group. Shared dependency
analysis supplies storage-free RAW edges to the concrete MTE1 consumers;
InsertSync owns the S-to-MTE1 completion events and lexical first-use sites.
The configuration receives no buffer version or hidden event number. Different
FMATRIX tuples retain 2D for the kernel; live reconfiguration is unsupported.
The selected path requires synchronization lowering and is disabled when
automatic scheduling is disabled. A hand-created PerCore reader must have one
flat Task load leaf; compound, repeated or conditional readers have a diagnostic
because their completion cannot consume a one-shot root event repeatedly.
Descriptor count reduction alone is not evidence of lower execution time.

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

2201 NZ-to-ND Fixpipe copies configure `ND_PARA` explicitly with
`asc_set_l0c_copy_params(1, 0, 0)` before issuing the transfer. The installed
`asc_copy_l0c2gm` implementation issues the copy without configuring this
register; a native instruction in generated code alone is insufficient.
The single-matrix configuration and `PIPE_FIX` barrier follow CANN's
`dav_c220` Basic Fixpipe implementation and remain inside the atomic FIX task.
The scheduler preserves program order between these register writers and FIX
consumers through the shared Fixpipe NZ-to-ND special-register dependency.
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

Static FP32 row broadcasts in ordinary `T.Parallel(R, 64)` assignments can
use one native Brcb and one strided subtract, multiply or division operation.
Brcb fetches eight compact FP32 row values per repeat and copies each payload
into one 32-byte block. Compiler-owned private UB scratch has `R * 8` FP32
elements (`32 * R` bytes); it is allocated before scheduling and has an ordinary
buffer lifetime. Both row-factor operand orders retain their mathematical
order, including `row / matrix`. Division uses the same native FP32 `vdiv`
primitive as the existing count API and inherits its accuracy and special-value
limitations; it does not insert a reciprocal algorithm.

This path requires a proved static positive row count divisible by eight and
at most 255, contiguous physical matrix rows of exactly 64 FP32 elements,
compact row state, and 32-byte-aligned starts. Unconditional consecutive
assignments can keep Exp and terminal casts at the whole-matrix count while
preserving each expression node and assignment order. Row state must not alias
matrix or output storage. Same-storage matrix operands require the identical
typed physical start; shifted views, lane predicates, dynamic or empty row
domains, and other widths retain the existing lowering. Partitioned mixed
kernels use the same test on each AIV's actual private row domain.

The IR retains FP32 `tl.access_ptr` read/write regions: Brcb reads `R` values
and writes `R * 8`, and the strided binary reads a matrix of `R * 64` and the
`R * 8` broadcast blocks while writing `R * 64`. Only native Brcb codegen
reinterprets FP32 pointers as UInt32, matching the CANN 9.1 b32 interface.
Every normal-mode instruction explicitly defines normal/full mask. These
operations use PIPE_V and the shared RAW/WAR/WAW, version and completion
analyses; no hidden event or composite softmax algorithm is introduced.

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
readback. Dynamic rows, partial or strided layouts, other widths and clear-false
reductions retain the ordinary row plan.

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

Native sub-K loops created by L1 GEMM lowering inherit the enclosing automatic
pipeline intent and optimization bound. Their L0 copies and MADs participate in
the same dependency, resource and version solver. An explicit pipeline setting
at a nested loop forms its own boundary; an explicit zero disables inheritance
within that loop. Clear-accumulator and final-completion expressions keep the
logical loop epoch when tasks receive a stage offset.

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

The initial 2201 cost model uses one **dimensionless ordering weight** per
leaf operation. Its function attribute is
`tl.ascend.cost_model = "dav-2201.ordinal.uncalibrated"`. These weights do not
represent cycles, measured bandwidth, latency or competitive performance.
3510 cycle estimates and INI rate fields are never used as measured 2201 costs.
Calibration requires separately identified target, units, workloads and
measurement evidence before replacing these weights.
Task cost overrides, debug output and scheduling plots use the same model
units. A measured cycle override cannot be mixed with the uncalibrated 2201
weights; calibration must first establish a consistent model for that target.

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
