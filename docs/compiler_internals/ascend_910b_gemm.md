# Continuous persistent GEMM on dav-2201

The [GEMM factory](../../examples/ascend/gemm_910b.py) supports FP16/BF16 NN
and NT operands, FP32 accumulation, dtype output and logical M/N/K tails.
Compile with `target={"kind": "ascend", "arch": "dav-2201"}`. Tile sizes
and traversal are Python policies; the compiler selects native operand
movement, stages, versions and synchronization.

`num_cores` distributes output tiles round robin across a bounded launch.
With `pipeline_outputs=False`, each output runs its own automatic K pipeline.
The default remains false. Explicit `pipeline_outputs=True` requires a
positive `num_cores`, `cache_a=False` and `a_cache_k=None`.

For `P = ceildiv(K, tile_k)`, the continuous path uses one automatic pipeline
of `ceildiv(output_tiles - block, groups) * P` logical tasks per block.
Task `t` owns output `block + (t // P) * groups` and K panel `t % P`.
Every output has exactly one owner, in increasing K order, one first-panel
clear and one last-panel export. A block's own extent excludes empty output
tasks. The compiler shifts task coordinates, valid regions and completion
together through fill/drain; Python encodes no stages, physical slot parity
or events.

The selected M128/N256/K256 policy uses the same capacities as the separate
output pipelines: 384 KiB L1, 32 KiB L0A, 64 KiB L0B and 128 KiB L0C.
Its native MAD is M128/N256/K64. These are measured-policy capacities, not
admission for arbitrary tile sizes. Logical tails retain padding and valid
GM output regions; resource and native geometry checks remain authoritative.

The guarded export uses ordinary C synchronization. Each L1/L0A/L0B slot
overwrite waits for its actual reader completion, and the final MAD completes
before export, followed by FIX completion before the next clear. Flattening
does not qualify the loop as a unit stream. The current compiler also places
MAD/FIX handshakes around panels without an actual export. Conditional FIX
keeps the established local `ND_PARA=1` configuration and `PIPE_FIX` barrier.
These costs remain candidates for a separately proved compiler optimization.

The measured policy changes output/K traversal without changing mathematics,
tile sizes, cache capacity or rounding. NN/NT, uneven owners and tails are
checked separately from performance. See the [performance report](../ascend/910b_performance.md)
for the three measured shapes, Torch routes and measurement limits.
