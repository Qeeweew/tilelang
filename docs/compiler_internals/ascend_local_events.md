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
whole loop or one exactly proven contiguous interval with static cut points.
The proof expands established condition definitions and checks equivalence
on actual loop iterations, including nonzero origins and positive steps.
Periodic, gapped, or unknown active sequences are diagnosed.

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

Unsupported clock or completion relations are diagnosed before emitting an
unproved binary protocol. CPU regressions check actual finite task epochs,
storage reuse, and wait-consumption-before-next-set paths independently of
the selected event numbering.
