# 2201 operand storage shared between phases

AutoSchedule can select two canonical logical L0A or L0B allocations to share
one bank. Each allocation retains its typed regions, layout, prepared versions
and iteration or counter clock. The early max-budget choice is provisional;
InsertSync must establish a physical reuse contract before the allocator may
place both at the same base address.

The conservative contract orders complete phases. On eligible 2201 programs,
InsertSync can instead establish a pool of physical operand-slot credits. A
load consumes its slot's credit before overwriting the bytes. The generation's
last whole-MAD reader returns that credit on M_MTE1. Each physical slot has one
initial credit and one terminal drain, shared across the logical allocations.
There is no phase-boundary seed or reset. An inactive generation performs no
load or return and leaves the pool state intact.

The selected members must have canonical compact rank-two FP16/BF16 storage,
one complete typed L1-to-L0 writer and one complete typed MAD reader site,
closed finite controls and no escaped or additional accesses. A fixed tile may
be read by a nested serial result loop; its credit is returned only after the
last result's MAD. Prepared rings retain their actual version expression at
each endpoint, including the endpoint's pipeline stage. Current selection
requires exactly two root allocations in the bank, under one unconditional,
static positive serial epoch and in distinct stage-zero phases. A prepared
ring has one generation owner and one or two versions. A counter ring also
requires stage-zero endpoints, one root initialization and one commit after
the reader. The analysis does not infer a new affine clock from a mutable
counter.

Two versioned members must have the same physical pitch. A smaller fixed tile
may occupy a prefix of the other member's slot zero. The pool must fit the
target bank, and its exclusive local event IDs count against the 2201 limit of
eight IDs per hard-event pair; it cannot use the 3510 flag-spill mechanism.
The pool implements the selected pair's intersecting-byte RAW, WAW
and WAR obligations through load, whole-MAD retirement and slot-credit reuse.
Each allocation's actual load-to-MAD readiness must still be completion-proven.
The unique full overwrite kills an older enclosing epoch's incoming value;
its old-write-to-new-reader relation does not require a phase-wide ready event.
Unsupported accesses, clocks, layouts, readiness or event capacity retain the
conservative phase protocol, including its existing checks and diagnostics.

This certificate establishes byte safety for the selected alias pair. It does
not establish whole-phase completion. In particular, an old disjoint slot or
FIX may remain active when the next phase loads slot zero. The certificate
does not relax C ownership, unit-flag retirement, GM handoffs or configuration
dependencies. Its exact generation recurrence can protect a local readiness
flag's lifetime; it must not become a parent phase-END to phase-START edge.

MaterializeMultiBuffer and late allocation keep the original logical pitches
and version expressions while assigning the proven shared base. Codegen emits
the established loads, MADs and flags without repairing the storage plan.
Dav-3510 does not select this pool.

## Parent obligations for one operand ring

A single allocation's retained load-to-MAD ready event and whole-MAD-to-load
slot release can also cover its direct parent's operand WAR and repeated-load
WAW. This requires one complete typed writer and reader, a closed positive
serial owner, multiple versions and actual completion paths for that storage.
A common outer stage on the child translates its physical clock and both event
endpoints together. With child extent `T`, width `W` and stage offset `s`, the
clock remains `(parent - s) * T + child`, with physical slot `clock % W`.
An odd `T` must rotate slots across invocations rather than reset them.

This covers the overwritten slot, allowing preparation of that slot while a
MAD still reads a different one. It establishes neither whole-panel completion
nor phase retirement. Guarded activation, mutable counters, extra or partial
accesses and cross-storage conflicts retain their existing protocols or target
diagnostics. The two-allocation pool's selection rules above are unchanged.
