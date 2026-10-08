# Ascend local event epochs

`InsertSync` consumes scheduled task effects, insertion sites, active epoch
domains, and the established buffer plan. Local event IDs are synchronization
resources; their count is not the physical buffer version count or the number
of compiler stages.

A carried dependency belongs to the loop whose children are being analyzed.
Its ordinary region comparison holds ancestor coordinates and definitions
fixed, while instantiating this loop and its descendants independently.
For example, an outer even-epoch spill and odd-epoch reload cannot create an
inner-loop carried edge. The parent analysis must retain their real boundary
hazard. A matching leaf task pair in a child dependency does not by itself
prove completion across the parent's iteration boundary. Independent storage
owners retain their separate epoch comparison.

## dav-2201 hardware contract

Local SetFlag writes a binary flag to one after the source pipe's preceding
work. It does not wait for the destination to consume that flag. WaitFlag
blocks its destination pipe until the flag is one, then clears it to zero.
Two sets cannot be treated as two accumulated notifications. See the CANN 9.1
[intra-core synchronization contract](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/基础API/同步控制/核内同步/核内同步能力概述.md).

For every reused local event ID, the previous wait must complete before the
following set completes. Source-code issue order between different pipes is
insufficient. A reverse completion path must follow established dependencies
and synchronization; weak DMA issue edges alone cannot discharge it.

Cross-core counters have a different contract. Their collective participants,
ID allocation and finite completion protocol are handled separately. This
local event rule does not change dav-3510 synchronization.

Each local hard-event namespace has eight IDs, independently for each physical
core: for example, `V_S` and `S_V` each have eight, and the two AIVs execute
their local event IDs in private banks. This is not a shared eight-ID budget
for all pipe directions or both AIVs. The late flag normalization
pass can compact sparse numbering into that namespace. CANN 9.1 exposes
`asc_lock`/`asc_unlock` only for dav-3510; dav-2201 cannot spill excess local
flags into that mutex pool. A 2201 layout requiring more than eight actual
slots is diagnosed with its hard event and ID ranges before native code is
emitted. Changing stage or buffer counts does not change this hardware limit.
The diagnostic describes the current allocation, not a proven minimum.
Compaction removes sparse numbering; sharing IDs between distinct channels
requires a completion/lifetime proof in `InsertSync`, including seed and drain
tokens. Summing separately allocated rings cannot prove they all overlap.
On dav-2201, local allocation uses both the concrete execution core and the
hard-event direction. The two AIV lanes run the same allocation in their
independent private banks. Final normalization preserves this distinction
inside the established `CUBE` and `VECTOR` blocks; it does not require a
lifetime-sharing proof between physically independent banks. If any call of
one hard-event direction lacks an unambiguous core block, normalization keeps
the conservative whole-function union for that direction, so common code and
core-specific code cannot receive inconsistent renamings. The dav-3510 mutex
spill policy retains its existing namespace and reservation rules.

## A carried edge

Let `src(i) -> dst(i+d)` be a dependency in one epoch domain. If the completion
closure establishes `dst(i) -> src(i+r)`, a sufficient local event ring width
is `R = d+r`. The event sequence is:

* source set: `base + i % R`;
* destination wait: `base + (i-d) % R`;
* initial sets: producer epochs `[-d, -1]`;
* final waits: producer epochs `[N-d, N-1]`, for `N` executed epochs.

Modulo denotes nonnegative index modulo. Only the `d` missing producer epochs
are seeded, even when the event ring has more IDs. When `N < d`, the final
waits include initial seeds that were not consumed by loop iterations. For
`N = 0`, the seeds and final waits cancel. Physical buffer slots and their
read/write regions retain their own indexing.

The reverse completion path ensures that the wait consuming source epoch `i`
precedes the next set using that event ID, at source epoch `i+R`. For a
same-iteration edge `d=0`, the existing event ring uses the reverse completion
distance. For the classic carried release with `r=0`, `R=d` and every event
ID is an initial seed.

If there is no reverse completion path, a bounded outermost lexical clock can
instead avoid event reuse. Let `U` bound its total trip count. When `U+d` fits
the local eight-ID limit, `R=max(1,U+d)` gives every real set and initial seed a
different ID. The same wait phase and runtime `N` drain remain valid. For
`d=0`, this is a finite ring of `U` IDs without seeds. Longer or unbounded
streams require a proven reverse completion path; they cannot reuse one flag
merely because their data regions are disjoint.

## Current lowering boundary

Shifted carried local events use an unconditional outermost lexical loop with
immutable bounds, a positive step, signed iteration/trip-count types, and no
break. Their flag clock must equal the loop's lexical iteration ordinal.
Source and destination must use the same active epoch sequence: either the
whole loop or one exactly proven contiguous interval with loop-invariant
integer cut points. Cuts may use immutable outer parameters or thread
coordinates, but cannot capture mutable reads or values changed in the loop.
Generated cut arithmetic is widened to signed Int64 before subtracting the
origin; ordinal ceiling uses quotient plus remainder. Its intermediate range
must be representable before clamping and narrowing back to the event clock.
Unknown or unsafe Int64 cut ranges are conservatively diagnosed.
The proof expands established condition definitions and checks equivalence
on actual loop iterations, including nonzero origins and positive steps.
Periodic, gapped, or unknown active sequences are diagnosed.

Before rebasing active spans, a carried local protocol can be removed when a
wider protocol already covers it at exactly the same physical source and
destination insertion positions. Both must use the same loop's raw lexical
ordinal and positive distance, with lexical domains owned by that loop.
The narrower source and destination guards must separately imply the wider
guards using only the loop context. Counter clocks, cross-core completion and
different physical endpoints are excluded. The compiler retains the wider
protocol's complete seed/set/wait/drain lifecycle; it does not assign the two
protocols one shared binary ID. Carried dependency projection into different
active sequences remains subject to its existing stricter checks.

For an active ordinal interval `[L,U)`, the event clock is rebased to `i-L`
and its trip count is `max(U-L,0)`. Physical buffer indexing and task guards
are retained. The same distance, seed and drain formulas then apply to these
active epochs, including an empty interval. Span expressions used by kernel
seeds and drains are simplified with the loop's outer context; loop-entry
facts cannot remove empty-interval protection from these hoisted expressions.
Shifted and finite no-reuse channels have exclusive event allocations because
the older sharing proof assumes equal phases and complete seed rings.

Classic nested and counter-clock protocols remain supported when their
established completion relation proves the existing persistent ring. A
shifted nested or counter clock requires an explicit common epoch contract;
it cannot be approximated by re-seeding each lexical loop entry. A local
drain blocks its destination pipe and does not prevent the source pipe from
executing a later seed.

A counter increment is Scalar work, not evidence that Vector readers have
finished. For a stage-zero counter owner with unconditional direct tasks,
`InsertSync` can complete a retained same-epoch `S -> V` dependency by emitting
a real `V -> S` handshake before its unique canonical `counter[0] += 1`.
Every reader of that counter group must be a preceding Vector task on the
same concrete core. The existing completion and binary event reuse proofs
must then accept both directions. Conditional tasks, mismatched counters,
other pipes or stages, and noncanonical advances remain outside this repair.
An inner owner with zero or one iteration may be entered repeatedly by an
ancestor; its local trip bound alone never permits unchecked event reuse.

Unsupported clock or completion relations are diagnosed before emitting an
unproved binary protocol. CPU regressions check actual finite task epochs,
storage reuse, and wait-consumption-before-next-set paths independently of
the selected event numbering.
