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
for m in Serial(M / tile_m):
    for n in Serial(N / tile_n):
        gemm(A[m slice, full K], B[n slice, full K], scratch, clear=True)
        copy(scratch, original_GM_output[m slice, n slice])
```

The scratch has compact canonical L0C layout and one physical version. Its
canonical layout is registered with the frontend storage-Var layout map when
the allocation is created; layout inference consumes that anchor. The
original oversized allocation and its layout entry disappear. The original GM
allocation, storage identity, leading coordinates and row pitch remain. Any
whole-output GM reader follows both physical loops. Each output element is
written exactly once after its complete K accumulation; the rewrite introduces
no GM partial sums or intermediate narrow accumulator.

The existing K lowerer expands each physical GEMM into bounded sub-K loads
and MADs. The first sub-K clears that physical tile, later sub-Ks accumulate,
and intermediate sub-Ks cannot publish whole-GEMM completion. New M/N loops
are ordinary static serial loops; enclosing automatic pipeline intent remains
visible to the existing scheduling path.

## Capacity and native effects

Tile selection uses the target L0C capacity, canonical padded footprints and
alignment. Other root L0C allocations are reserved by physical storage identity,
including fixed versions and existing manual physical axes. The scratch's
single-version override is recorded in the common root version map. Subsequent
automatic L0A/B/C admission reserves the whole root and preserves counts chosen
for earlier owners, so another owner cannot later exceed the same bank.

The copy/GEMM regions retain actual source origins and strides. FIX reads the
compact scratch with its padded-M source pitch and writes with the original
GM row pitch; a subtile width does not replace that pitch. L1-to-L0 lowering
continues to validate complete source fractals and bounds. M slicing may use
the existing 2D load fallback when the full-M 3D-load contract no longer holds.

Dependency and synchronization lowering consume these ordinary effects:

- Each tile's MAD completes before FIX reads it.
- FIX releases scratch before the next N/M tile or outer epoch overwrites it.
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

An oversized accumulator has a diagnostic when it has partial or guarded
export, multiple readers, alias or pointer escape, custom layout, manual
version metadata, special completion control, or an explicit Task/Stage
boundary. Cross-GEMM accumulation (`clear=False`) and dynamic/unaligned M/N
tails require a different lifetime or remainder lowering. The logical L1
inputs, GM output and private UB consumers must still fit their own contracts.
This capacity mechanism makes no performance claim about a chosen Python tile.
