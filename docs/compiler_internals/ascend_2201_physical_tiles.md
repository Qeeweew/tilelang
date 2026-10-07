# 2201 physical GEMM tiles

`AscendLowerL1Gemm` runs after logical GM handoff and before layout inference,
mixed ownership, dependency analysis and scheduling. It exposes ordinary
copies, GEMMs, buffers and loops; it adds no opaque compute or synchronization
primitive.

## Logical output and physical accumulator

For a single logical L0C accumulator larger than the target bank, the boundary
requires a unique full-clear L1 GEMM followed by its complete L0C-to-GM export.
It rewrites both operations together:

```text
for tile in Serial((M / tile_m) * (N / tile_n)):
    m, n = tile // (N / tile_n), tile % (N / tile_n)
    slot = tile % 2
    gemm(A[m slice, full K], B[n slice, full K], scratch[slot], clear=True)
    copy(scratch[slot], original_GM_output[m slice, n slice])
```

When capacity permits, the scratch has shape `[2, tile_m, tile_n]` and compact
canonical L0C layout. Both operations address the same explicit slot region.
The leading axis is registered as an existing manual period of two; automatic
versioning must not add another axis. Its canonical layout is registered with
the frontend storage-Var layout map when
the allocation is created; layout inference consumes that anchor. The
original oversized allocation and its layout/version entries disappear. The original GM
allocation, storage identity, leading coordinates and row pitch remain. Any
whole-output GM reader follows the physical loop. Each output element is
written exactly once after its complete K accumulation; the rewrite introduces
no GM partial sums or intermediate narrow accumulator.

The existing K lowerer expands each physical GEMM into bounded sub-K loads
and MADs. The first sub-K clears that physical tile, later sub-Ks accumulate,
and intermediate sub-Ks cannot publish whole-GEMM completion. One ordinary
static serial loop supplies the physical tile ordinal, including across M/N
boundaries; enclosing automatic pipeline intent remains visible to the existing
scheduling path.

When a physical GEMM needs only one native K chunk, native lowering loads A
on the first N tile of each M tile and reuses that L0A data for the remaining
N tiles. The generating pass carries the M/N geometry through fresh loop
identities within the pass; user annotations do not authorize this reuse.
The closed producer/export loop has no intervening L1 writer. A keeps one
fixed physical version so later N readers address the initialized slot.
Its typed regions retain first-load RAW and last-reader WAR across M tiles,
owners and enclosing epochs. Multiple K chunks retain their ordinary loads
because one L0A tile cannot cache all of K. This reduces repeated movement
without adding L0 capacity or changing GEMM accumulation or export order.

An adjacent producer/export pair may occur inside a shared conditional branch,
an Ascend Stage, or a transparent CUBE core block. The replacement stays inside
that same scope: its guard, stage and owner loop still dominate every physical
GEMM and export. A guarded producer followed by an export outside the guard,
or two separately guarded statements, does not satisfy this local contract.

## Capacity and native effects

Tile selection uses the target L0C capacity, canonical padded footprints and
alignment. Other root L0C allocations are reserved by physical storage identity,
including fixed versions and existing manual physical axes. Two slots are
reserved from the remaining capacity before choosing aligned tile dimensions.
If two minimum 16×16 FP32 slots cannot fit, one minimum slot may still use the
single-slot path with a fixed count of one. Manual shape includes its slots and
is charged once, rather than being multiplied by its period again. Subsequent
automatic L0A/B/C admission reserves the whole root and preserves counts chosen
for earlier owners, so another owner cannot later exceed the same bank.

The copy/GEMM regions retain actual source origins and strides. FIX reads the
compact scratch with its padded-M source pitch and writes with the original
GM row pitch; a subtile width does not replace that pitch. L1-to-L0 lowering
continues to validate complete source fractals and bounds. M slicing may use
the existing 2D load fallback when the full-M 3D-load contract no longer holds.

Dependency and synchronization lowering consume these ordinary effects:

- Each tile's MAD completes before FIX reads it.
- FIX releases a slot before the next physical tile using that same slot
  overwrites it. Adjacent different slots have no C-storage reuse dependency.
- Each enclosing owner/epoch boundary retains its own release obligation. The
  physical ordinal restarts there; a flattened flag clock alone does not prove
  slot reuse safe, especially after an odd number of physical tiles.
- All subtile exports complete before a whole-output reader or mixed publish.
- Both AIV readers release a mixed GM slot before the next outer epoch's first
  FIX overwrites it. Empty private partitions retain collective completion.

Loop completion projection belongs to `InsertSync`; code generation does not
invent event numbers or shift pointers to repair the rewrite.

## Explicit shared C column regions

Regular 2201 L0 MAD also accepts an aligned N-column region of a compact,
canonical NZ L0C allocation. For backing shape `[M, N]`, the destination
`C[0:M, n0:n0+n]` occupies the consecutive native elements
`[n0*M, (n0+n)*M)`. Backing M/N, n0 and n must be multiples of 16. The region
must cover every M row; a partial M region would require a different native
C pitch. Leading version axes select single planes. A/B regions retain their
existing zero-origin contract.

The loop-aware tile lowering boundary validates the same compact canonical
encoding for MAD and L0C-to-GM export, and proves every region axis is positive
and inside its backing allocation before flattening the pointer. Native
lowering consumes that contract. The common layout remapper computes the NZ address, while native
MAD M/N/K come from the operation regions rather than the backing allocation.
Custom layouts, noncompact storage and 3510 column regions are diagnosed.
The destination and layout key must identify the actual backing Buffer;
matching names or aliased Buffer views do not establish its native encoding.

This allows ordinary DSL kernels to share one explicitly allocated C bank
between a wide QK tile and narrower PV column regions. Their ordinary typed
read/write effects must still establish each FIX completion before an
overlapping MAD overwrite. Admitting a region neither inserts a separate
synchronization mechanism nor chooses this storage policy automatically.

## Explicit unit controls

Unselected native operations use unit control zero and ordinary completion events.
For an explicitly annotated L1 GEMM, K lowering retains the CHECK bit on every
nonterminal native MAD and preserves the original control on the final MAD,
including a separate K tail. Thus control 3 becomes `2, ..., 2, 3`; control 2
remains 2 throughout. An absent annotation stays absent. This does not publish
an intermediate K result or change which sub-K clears the accumulator.

On 2201, numeric controls alone do not remove completion events. They do not
prove matching C regions, safe initial reuse, finite drain or direction state.
The 3510 explicit compatibility path is separate from the 2201 selector.

## Closed 2201 unit streams

After scheduling and core resolution, `InsertSync` can select a serial result
loop containing ordinary native-L0 GEMMs followed by one complete result
export from C to GM. C must be the actual rank-two FP32 allocation with
canonical compact NZ layout, aligned M/N and one fixed physical slot. Each
result starts at zero in both dimensions and retains the complete backing M;
its aligned N may be smaller than the allocation N. Every MAD and the export
must use exactly that same effective domain. Keeping the full M preserves the
canonical NZ pitch; the full allocation still counts against the 128 KiB
capacity. The native MAD/FIX dimensions come from the effective operand
regions, rather than treating allocation N as the actual result width.
A/B are FP16 or BF16. The finite native-K chain clears C exactly once at its
start and accumulates thereafter. The selector assigns MAD controls `2, ..., 2, 3` and
FIX control 3 atomically after validating the whole transaction.

Only matching C RAW edges within that result and C WAR edges carried by that
exact result loop are discharged. Unit protection is not added to the generic
completion graph. M accumulation order, outer owner/epoch reuse, A/B readiness,
GM publication, register state and other storage dependencies retain ordinary
completion proofs.

The stream has explicit ordinary event boundaries allocated by the existing
flag allocator: M-to-S before selecting N direction, S-to-M before its first
MAD, S-to-FIX before its first export, full FIX-to-M after its last export,
and M-to-S before restoring M direction. The S-to-FIX gate carries the entry's
real MAD completion to the next activation's FIX, preserving enclosing
owner/epoch dependencies. A final S-to-M gate makes the restored state visible
to subsequent MADs, including unselected ones. The terminal drain runs once
per stream activation, including when an enclosing owner or epoch repeats.
Its dedicated binary flag is cleared before the following blocking Scalar
wait permits another activation to issue a notification; this flag-lifetime
proof does not add a whole-MAD/FIX completion edge. No unit seed is assumed.

If selection removes an unrelated event's reverse-completion witness or
exceeds the eight local hard-event IDs after safe reuse,
`InsertSync` retries the untouched ordinary scheduled program. Unit data
protection cannot establish GM or operand completion for flag reuse.

Mismatched MAD/export domains, partial M or nonzero C origins, M/N tails,
interleaved C writers, guarded terminal MAD/FIX, compound tasks, versioned C
and nonzero stages retain their original path.
An enclosing guard may enable or disable the stream. A condition directly on
the result loop is also supported when it depends only on immutable outer
loop values or their pure scalar Bind snapshots. In that case one lexical
activation scope contains the complete entry/setup/result/drain/restore/exit
transaction, so an inactive epoch issues none of its boundary events. Mutable
loads, opaque calls, result/K loop variables and attribute guards do not qualify
for this normalization. It preserves the original result and storage clocks;
an unrelated unsupported binary-event pattern still takes the original fallback
or diagnostic. Explicit direction or atomic-state control and opaque external
calls also prevent automatic selection. These conservative
boundaries do not require new user DSL and do not choose Python tiling policy
or emulate 3510 RegBase/VF execution.

## Supported boundary

The rewrite targets 2201 FP16/BF16 L1 inputs, FP32 accumulation and FP32 GM
export, with nontransposed A and NN or NT B. M/N are static positive multiples of 16 and
physical tiles divide them. Singleton leading operand/output regions and
fractal-aligned nonzero L1 origins are retained. Only one oversized root
accumulator is admitted. Accumulators already fitting the bank and 3510 keep
their existing path.

The logical accumulator may have an explicit fixed count of one. Removing it
also removes its keys from known layout, fixed/manual count and version-mode
maps; metadata for all other storage remains. A count greater than one or an
explicit version mode requires a different lifetime lowering and is diagnosed.

An oversized accumulator has a diagnostic when it has partial export, a
producer/export split across guards, multiple readers, alias or pointer escape,
custom layout, special completion control, an opaque block, a While loop, or
an explicit Task/schedule-unit or pipeline-order boundary. Cross-GEMM
accumulation (`clear=False`) and dynamic/unaligned M/N
tails require a different lifetime or remainder lowering. The logical L1
inputs, GM output and private UB consumers must still fit their own contracts.
This capacity mechanism makes no performance claim about a chosen Python tile.
Reserving a second slot can produce smaller tiles and more MAD/FIX commands.
It also leaves mixed C/V publication collective on 2201: independent completion
for each AIV requires a separate owner publication contract.
