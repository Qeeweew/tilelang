# Fixed operand planes and shifted unit streams on dav-2201

An ordinary operand cache can have a static leading axis, for example
`A[2, 128, 128]` in L0A. This axis names two resident logical matrices; it is
not a compiler-generated buffer-version axis. The full allocation occupies
64 KiB for FP16/BF16. A later, independent native operand ring can reuse the
same bytes only after the readers of the cache have completed.

## Early storage choice

L0A/B phase selection admits canonical, contiguous FP16/BF16 allocations with
static leading axes and complete matrix planes. Every leading access has
extent one and a proven in-bounds origin. Both matrix axes have zero origin
and cover the complete allocation plane. Raw pointers, scalar accesses,
different logical views, opaque effects and noncanonical layouts retain the
existing exclusions.

The cache's writer and reader may occupy adjacent finite controls under one
resource epoch. Selection compares the complete lexical intervals of both
allocations, including their last access. Interleaved intervals cannot use a
maximum-byte budget. When a resident interval spans multiple controls, the
loop scheduler retains its complete physical issue order against its mate;
the cache's unversioned writer and final-reader controls share one incoming
stage. Independently orienting each endpoint against the mate would permit
the mate to acquire credits between the cache's acquire and release, creating
a cycle. Single-control native rings keep their ordinary owner-order choices.
The selected pair remains a mandatory downstream alias
obligation; its marker is never evidence that completion has been established.

## Whole-MAD credits for a resident plane set

The fixed-plane operand protocol currently admits one leading axis. One
unconditional finite loop writes each complete plane exactly once; a later
finite nest reads each plane once per repetition. The endpoint's leading
index must be its immediate plane loop variable, with the same static extent
as the allocation. Writer and reader loops keep distinct lexical indices.
The cache has no prepared automatic version owner or counter.
Synchronization checks the final scheduled interval again. A mate inserted
between the writer and final readers, or independently shifted cache
endpoints, cannot establish the selected physical alias.

The reader returns its plane's physical byte credits after the last enclosing
repetition that reads that plane. It does not wait for the other plane's
reader to issue, nor return another plane's credit. Missing plane readers,
independent endpoint guards, repeated writers in a different generation and
storage-dependent control do not establish this protocol.

The other member of the selected pair retains its ordinary automatic ring,
including its prepared counter or iteration clock. The pool partitions both
sets of physical slots at their common byte boundary when their pitches
differ. A larger native operand acquires and returns all intersecting credits.
Actual pool bytes must fit the target bank, and all credits must fit the
ordinary `M_MTE1` event bank.

Synchronization also proves the retained load-to-MAD readiness in the common
generation. Each credit is returned by an actual whole-MAD event. C unit
readiness, FIX completion, a chosen alias or a declared ring size cannot
replace operand completion. Late allocation consumes only the final,
completion-proven alias.

## A complete result at an incoming parent stage

A serial result loop can enter at a nonzero, already scheduled parent stage
without changing its own native-K chain. Selection requires a static finite
unit-step parent, known nonnegative child stages and enabled stage offsets.
An automatic parent's consumed integer stage bound must contain every child.
A complete manual parent instead supplies explicit offsets and its children's
actual finite stages. An unconsumed automatic request is not a schedule proof.

The existing complete C-domain, clear/accumulate, unique export, register
preservation, native-K ordering and operand-completion checks still apply.
Every entry and terminal boundary uses the result's incoming stage. When an
immutable activation guard needs a lexical shell, the shell owns that stage
and its children use local stage zero. This prevents a second shift during
parent fill/drain.

If every MAD and FIX belongs to a complete selected stream and every operation
preserves the direction register, the kernel can set direction N at entry and
restore M after terminal completion. Real stream boundaries remain. In
particular, separate N-chunk result streams still have their own terminal
drains; this contract does not merge them into one longer result stream.

The ordinary CPU regressions check physical overlap and actual binary flags
across three epochs, both operand dtypes, automatic rings, guarded incoming
stages, exact output transactions and invalid capacity/lifetime cases.
Device correctness and a same-card profiler comparison are separate gates for
using this compiler capability as a performance policy.
