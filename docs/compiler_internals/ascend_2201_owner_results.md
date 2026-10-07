# Owner result completion on dav-2201

Mixed kernels can describe independent AIV queries with ordinary buffer
coordinates: Cube writes `result[group, owner, slot, ...]` in a serial
two-owner loop, and each AIV reads `result[group, sid, previous_slot, ...]`
into validated private UB. `InsertSync` can publish each complete owner
result separately instead of waiting for both owners' exports.

This is a result-ready optimization, not a new task scheduler or a composite
attention primitive. Python still controls tiling, traversal, GEMM input reuse,
and the ordering of QK, softmax, PV, and output updates. In particular, a
chunk-outer PV algorithm that exports only after both accumulators finish
cannot acquire the same Cube/Vector overlap as an owner-outer PV algorithm
solely by changing its completion notification.

## Facts and pass boundary

`auto_schedule/owner_result.h` reads the already scheduled physical IR. The
proof uses canonical typed copy regions, the kernel SID, validated private UB,
manual GM buffer versions, lexical loop coordinates, and immutable activation
conditions. It does not use operation names or a FlashAttention shape table.

A candidate has one typed L0C-to-GM FIX writer and one typed GM-to-private-UB
MTE2 reader for the canonical storage. Their complete logical regions may be
different rectangular partitions, such as physical M/N exports versus vector
row blocks. The proof checks:

- an explicit serial owner coordinate in `[0, 2)` matches the reader's SID
  axis, with disjoint physical GM planes and compact or padded injective
  strides;
- source exports cover a rectangular logical result without holes or overlap,
  and the reader's regions cover that same result; the logical result may be
  smaller than its allocation;
- group/key coordinates and the GM slot agree at the dependence distance;
- two active writes fewer than the declared slot count apart cannot select
  the same physical slot; the version count alone is not a reuse proof;
- source and consumer have exact finite contiguous activation intervals,
  shifted by that distance, including conditions inside an atomic task;
- aliases, pointer escapes, scalar accesses, extra storage users, unsupported
  copy forms, loop breaks, or ambiguous coordinates do not enter the plan.

Proof enumeration is bounded to 256 rectangles/control instances. It does
not unroll the emitted IR. Unknown, dynamic, or larger cases retain ordinary
collective synchronization. The first implementation accepts static serial
epoch loops and positive static descendant traversals.

The resulting `OwnerResultPlan` is orthogonal to `EpochDomain`: the latter is
a storage lifetime/clock, whereas owner is a participant selector. The plan
retains both endpoint slots and activations. `InsertSync` removes only the
matching storage/task-pair/distance RAW from generic synchronization analysis.
The owner relation is **not** added to the all-participant dependency closure,
parent projections, flag reuse, or buffer-alias lifetime proofs. An unrelated
GM hazard therefore cannot inherit completion of the other owner's export.

## Completion, release, and finite execution

Cube publishes after an owner's final FIX export, retaining last-iteration
guards for physical M/N loops. Each AIV waits once before its first GM read,
retaining first-iteration guards for reader subtiles. The plan emits tokens
only for actual active results, with no synthetic root seed or drain tokens.

Each `(owner, GM slot)` has an exclusive result ID. The compiler independently
requires the ordinary collective reader-release closure to prove:

```text
old own result wait
  -> old GM reads complete
  -> collective reader release
  -> next write to that GM slot starts
  -> next FIX result notification on the same ID
```

Thus the consumed owner stream never needs more than one outstanding token per
ID. GM slots and result IDs remain separate concepts; their correspondence is
proved from region coordinates, not inferred from a shared loop phase.
P-input readiness, other GM hazards, and collective release retain their
existing contracts and validation.

Result IDs are allocated from the remaining 16 cross-core base slots, excluding
explicit reservations and generic allocations. No result ID is shared with
another protocol. Insufficient IDs, an unproved release, or loss of a generic
local binary reverse-completion proof causes a retry with ordinary collective
completion. The original binary and collective diagnostics remain in force.

## Hardware scope

This specialization is for dav-2201 with two physical AIVs. It follows the
CANN 9.1 B-shared/A-private Matmul service's mode-2 FIX notification and each
AIV client's distinct-ID wait. ID parity is not a hardware target-SID field;
the consumer chooses its result ID. It does not substitute dav-3510 mode 4,
introduce independent Vector-to-Cube counters, or model cross-core flags as
local binary hard events.

The public 2201 wrapper does not specify unused-SID notification-copy capacity
or task-boundary cleanup. The implementation follows the official ordinary
Matmul usage for those copies; its own-consumer finite token/reuse proof does
not establish an unlimited counter capacity. Device validation remains
necessary alongside generated-code and native completion review.
