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
