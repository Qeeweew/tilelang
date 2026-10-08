# 2201 paired GM input read release

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
