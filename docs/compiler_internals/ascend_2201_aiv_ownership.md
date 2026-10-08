# AIV ownership on dav-2201

A mixed dav-2201 kernel contains one Cube and two Vector cores. Each AIV has
private UB. The logical buffer domain and the division of work must be
established before physical layout, vector lowering and scheduling.

The default ordinary mixed-kernel path partitions a common logical row
domain between the two AIVs. `AscendLowerMixedPartition` establishes the
private row capacities, valid regions and GM offsets. This remains the
default for ordinary GEMM-to-vector kernels and attention tiles.

Connected GM boundaries must agree on one valid row domain. Their geometric
intersection provides the component domain, and each boundary must prove
equality in its own lexical context. This retains facts used by earlier
conditional simplification without treating different guards as interchangeable.
The shared domain must be in scope at every component use; loop-local and
branch-local values cannot escape their defining scope. Repeated mutable reads
cannot establish a stable domain. Empty AIV payloads retain collective
completion.

## Complete local domains

Some algorithms assign a different complete query tile to each AIV. These
use an existing two-SID launch and explicitly declare their local UB storage:

```python
with T.MixedKernel(groups, sids=2) as (group, sid):
    scores = T.alloc_shared((64, 128), "float32")
    result = T.alloc_shared((64, 128), dtype)
    state = T.alloc_shared((2, 128), "float32")
    T.annotate_aiv_private(scores, result, state)
    # GM regions select this AIV's query. The UB domain is already local.
    T.copy(Score[group, sid, task_slot, row:row + 64, :], scores)
    for i, j in T.Parallel(64, 128):
        result[i, j] = T.cast(scores[i, j] - state[query_slot, row + i], dtype)
    T.copy(result, P[group, sid, task_slot, row:row + 64, :])
```

The declaration retains the complete shapes, regions and Parallel domains
on **each** AIV. It does not divide the 64 rows again. It is not a launch,
buffer-version declaration, scheduling hint or synchronization operation.
The GM owner axis and query/task slot axes are independent coordinates.

Declare every UB allocation connected to a private component through
Parallel expressions, copies and reductions. A private component cannot
mix declared and automatically partitioned UB storage. Independent components
may use different ownership modes in the same kernel.

## Stage contract

`AscendLowerMixedPartition` validates the declaration at the pre-layout
boundary. Members use allocation data variables, so identical views preserve
storage identity. The current implementation admits compact, static,
kernel-root UB allocations and coordinate-preserving aliases. Rank-changing
aliases, custom private layouts, opaque pointer escapes and nested SID scopes
require separate contracts and are diagnosed.

GM writes must have an explicit SID-separated coordinate axis. The boundary
compares writes from the two SIDs over independent loop iterations and
requires their regions to be disjoint. An owner dimension is not inferred
from a buffer name, shape or slot count. Crossed writes and shared GM output
regions are rejected. Cube/vector handovers use explicit typed GM regions.
When private-written GM returns to UB, its reader must retain the same SID
plane. Row/private GM bridges and cross-SID reads cannot create implicit
AIV-to-AIV communication.

After validation, the root records the established private storage members.
Layout, copy and reduce planning retain their ordinary buffer regions.
Vector lowering consumes membership for the actual operation rather than
using a kernel-wide exemption. Reduce and vector lowerers append the scratch
they create in the same local execution domain; they do not reconstruct
ownership from a padded scratch shape.

Scheduling, storage planning and synchronization consume the resulting
ordinary core, region and loop-dependence facts. Code generation emits those
facts without adding owner offsets or changing the algorithm.

## Completion and current scope

The current mixed-kernel completion protocol remains collective across both
AIVs. Private data ownership does not make completion independent: both AIVs
must participate at collective task boundaries, including when an algorithm
has an inactive payload. Moving a SID predicate around a collective wait is
not a valid way to create an independent query stream.

This API does not implement a KFC request queue, per-request service progress,
or typed score/probability aliasing. A complete manual schedule also needs a
separate direct-plan interface; `T.Stage` retains its documented per-pipe
ordering semantics. Ordinary `T.Pipelined` still selects stages and versions
automatically.

The complete private-domain declaration currently targets dav-2201 and is
diagnosed on other architectures. Finite fill/drain, slot reuse, tails and
empty work still need to satisfy the common dependency and completion
contracts; the declaration does not waive resource or synchronization checks.
