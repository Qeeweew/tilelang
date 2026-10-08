# 2201 paired GM slot handoffs

## Output write acquire

For an established owner-result relation, InsertSync can delay the paired GM
overwrite acquire from phase entry to the first typed L0C-to-GM FIX. The
two-AIV read-release is selected independently as described below. An active
epoch consumes the acquire only at owner zero's first slice; an inactive epoch
consumes it at the original phase entry. This preserves the original lexical
clock, reverse WAR distance, event width, seeds and drains, while allowing
independent input preparation and MAD issue before the GM permission arrives.

Selection requires the current owner-result proof, canonical GM identity, two
disjoint owner planes and two storage slots, one exact reverse WAR, stage-zero
positive static serial paths and a finite exact producer activation interval.
It preserves the ordinary reverse distance `slots - result distance`; it does
not force a two-event clock for a distance-one result. Automatic/counter GM
versions and unsupported access geometry retain the original acquire.

The private certificate proves only old GM reads before the next same-slot
write. It does not prove entry to the whole Cube phase waits for those reads.
It stays outside completion closure, parent projection, alias retirement and
flag sharing. C, SPR, local binary lifetime and owner-ready checks remain
independent. Analysis restores the original acquire before discarding another
optional input release if any required proof or event allocation fails.

## Output read release

The same owner-result proof can move the paired result-slot release from the
whole Vector phase END to its final typed GM-to-private-UB MTE2 copy. Later
Vector math and output stores no longer delay permission to overwrite the old
GM result. Each physical AIV contributes after its own final reader slice;
the Cube acquire still requires both contributions. An inactive epoch retains
one complementary notification at the original wide END on each AIV.

This cut requires two manually declared GM slots, a distance-one owner result,
its exact inverse WAR, and unshifted positive static serial paths. Canonical
storage, private UB, complete coverage, aliases and activation are established
by the current owner-result analysis. It preserves the original distance-one
flag clock, root seed and terminal drain. It may compose with output write
acquire on the same permission, or retain the ordinary destination acquire.

Only the old result GM reads are certified complete. The early source does
not certify Vector phase END, output update completion, UB retirement, local
binary lifetime or any unrelated dependency. Both placement certificates stay
outside generic closure, parent projection, alias proofs and flag sharing.
Unsupported cases keep the wide release; a failed independent proof retries
without optional cuts before materializing events. Finite regressions must
check both AIVs, exact active/inactive contributions, all old read-to-new write
paths, bounded counters and terminal drain, while showing unrelated Vector
tail work is absent from the GM overwrite prerequisite.

## Input read release

InsertSync can release a paired GM input after its last typed GM-to-L1 copy,
while later operand loads, MADs and result exports continue. This changes the
placement of the existing Cube-to-vector slot-release notification. It does
not change the algorithm, storage versions, owner-input readiness or the
two-AIV collective protocol. Dav-3510 keeps its existing path.

Selection requires an already proven owner-input relation: private AIV writes
and Cube reads use the same canonical GM Buffer, two disjoint owner planes and
two physical slots. Typed rectangular accesses must cover the complete input
plane, with no aliases or additional accesses. Both sides have the same exact
finite activation interval in a positive static serial epoch. The inverse WAR
must be one exact reader/writer pair with distance and event width two. The
selected paths are unshifted, immutable, positive serial loops; automatic or
counter storage clocks retain the ordinary release.

The active notification follows the final GM-to-L1 copy across both owners
and all descendant slices. It is not emitted once per owner. For inactive
epochs, a complementary notification stays at the original phase boundary.
Together these sites emit exactly one contribution per original lexical
epoch. The destination acquire, event clock and width retain the ordinary
protocol, with two root notifications and two terminal waits per AIV. Event
IDs remain allocator-owned and need not have the same numeric base as the
wide protocol. Both physical AIVs consume each contribution, including epochs
with no payload. An inactive contribution may supply a later same-slot
overwrite permission; it neither declares data ready nor creates a payload
generation. A later active write still consumes its corresponding slot release.

This placement certificate proves only the selected input's GM read-before-
overwrite obligation and the paired acquire needed before owner-input
ready contributions. It does not prove the old phase END complete. Its point
does not enter generic dependency closure, parent projection, alias retirement
or flag sharing, and cannot remove an unrelated synchronization point. L1/L0
operand lifetimes, whole-MAD/FIX retirement and configuration dependencies
remain independent obligations.

Unsupported geometry, activation, sites or clocks retain the original wide
release. If removing its wide completion witness prevents another local
binary protocol, required alias proof, owner release or event allocation from
being established, analysis retries with the early release disabled before
event materialization. The ordinary checks and diagnostics remain in that
retry; no relaxed completion edge is introduced.

Finite validation must check the active last-owner/last-slice notification,
the complementary inactive contribution, both AIV consumers, original
seed/drain generations, and every old GM read before the next same-slot write.
It must also retain local binary flag, operand, alias and terminal checks.
Source placement alone does not prove a speedup or independent AIV execution.
