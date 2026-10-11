"""FP16/BF16 GEMM with ordinary automatic pipelining on dav-2201.

The Python kernel chooses logical tiles. Native sub-K movement, L0 scratch,
buffer versions and synchronization belong to the compiler. Compile the
returned PrimFunc with target={"kind": "ascend", "arch": "dav-2201"}.
"""


def gemm(m, n, k, *, dtype="bfloat16", transpose_b=True, tile_m=128, tile_n=256, tile_k=256, cache_a=False,
         a_cache_k=None, num_cores=None, pipeline_outputs=False):
    """Build C=A@B.T or C=A@B, with FP32 accumulation and dtype output.

    A is [m,k]; B is [n,k] for transpose_b=True or [k,n] otherwise.
    Logical dimensions can have tails. Tile sizes are Python algorithm
    choices; the target's resource planner diagnoses a tile that cannot fit.
    The default 128x256x256 tile is a throughput starting point for large
    matrices. For N <= 128, select tile_n=128 or smaller to avoid excessive
    padding; small M likewise benefits from an explicit smaller tile_m.
    Increasing tile_m to 256 requires tile_n <= 128 for the FP32 accumulator
    to fit dav-2201 L0C; operand versions must also fit the target resources.
    With cache_a=True, this output tile caches all of A's K dimension in
    L1 before the K loop. Its capacity is rounded to tile_k; select tiles that
    fit the target resource planner. The cache is private to each output tile.
    Alternatively, a_cache_k sets a bounded A cache width, a positive multiple
    of tile_k. An ordinary outer pipeline loads each A cache block once; an
    inner ordinary pipeline consumes tile_k slices with streamed B panels. It is
    mutually exclusive with cache_a=True. Both choices preserve increasing
    K accumulation order and use the compiler's automatic buffer versions.
    For example, M128/N256, a_cache_k=512 and tile_k=256 occupy 512 KiB of L1
    if both operands use two versions. Each loop expresses pipeline intent;
    actual versions and capacity remain the target planner's responsibility.
    With num_cores set to a positive integer, at most that many cores
    distribute output tiles round robin. Each core uses an ordinary serial
    output loop and the same automatic K pipeline; the default None launches
    one logical block per output tile. This is an algorithm traversal choice,
    independent of physical-device selection.
    With pipeline_outputs=True, a single automatic pipeline spans this core's
    output tiles and K panels. Each output still clears at its first panel
    and exports after its last panel. This requires streamed A and a positive
    num_cores; it is incompatible with either A cache policy.
    No manual stages or buffer versions are required.
    """
    from tilelang.ascend import language as T

    for name, value in (("m", m), ("n", n), ("k", k)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("tile_m", tile_m), ("tile_n", tile_n), ("tile_k", tile_k)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 16:
            raise ValueError(f"{name} must be a positive multiple of 16")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dav-2201 GEMM requires FP16 or BF16 operands")
    if not isinstance(transpose_b, bool):
        raise ValueError("transpose_b must be a bool")
    if not isinstance(cache_a, bool):
        raise ValueError("cache_a must be a bool")
    if a_cache_k is not None:
        if not isinstance(a_cache_k, int) or isinstance(a_cache_k, bool) or a_cache_k <= 0 or a_cache_k % tile_k:
            raise ValueError("a_cache_k must be None or a positive multiple of tile_k")
        if cache_a:
            raise ValueError("a_cache_k and cache_a=True are mutually exclusive")
    if num_cores is not None and (not isinstance(num_cores, int) or isinstance(num_cores, bool) or num_cores <= 0):
        raise ValueError("num_cores must be None or a positive integer")
    if not isinstance(pipeline_outputs, bool):
        raise ValueError("pipeline_outputs must be a bool")
    if pipeline_outputs and (num_cores is None or cache_a or a_cache_k is not None):
        raise ValueError("pipeline_outputs requires num_cores and streamed A")
    a_k_extent = ((k + tile_k - 1) // tile_k) * tile_k if cache_a else (a_cache_k if a_cache_k is not None else tile_k)
    b_shape = (n, k) if transpose_b else (k, n)
    n_tiles = (n + tile_n - 1) // tile_n
    grid = ((m + tile_m - 1) // tile_m) * n_tiles

    num_groups = grid if num_cores is None else min(grid, num_cores)
    k_panels = (k + tile_k - 1) // tile_k

    @T.macro
    def output_tile(A, B, C, tile_id):
        mi = tile_id // n_tiles
        ni = tile_id % n_tiles
        a_l1 = T.alloc_l1((tile_m, a_k_extent), dtype)
        b_l1 = T.alloc_l1((tile_n, tile_k) if transpose_b else (tile_k, tile_n), dtype)
        accumulator = T.alloc_l0c((tile_m, tile_n), "float32")
        if a_cache_k is not None:
            for cache_epoch in T.Pipelined(T.ceildiv(k, a_cache_k)):
                T.copy(A[mi * tile_m : (mi + 1) * tile_m,
                         cache_epoch * a_cache_k : (cache_epoch + 1) * a_cache_k], a_l1)
                # Intersect the cache domain with the complete input domain;
                # panels beyond all K cannot become active in any epoch.
                for panel in T.Pipelined(min(a_cache_k // tile_k, (k + tile_k - 1) // tile_k)):
                    panel_k = cache_epoch * a_cache_k + panel * tile_k
                    if panel_k < k:
                        if transpose_b:
                            T.copy(B[ni * tile_n : (ni + 1) * tile_n, panel_k : panel_k + tile_k], b_l1)
                        else:
                            T.copy(B[panel_k : panel_k + tile_k, ni * tile_n : (ni + 1) * tile_n], b_l1)
                        T.gemm(
                            a_l1[:, panel * tile_k : (panel + 1) * tile_k],
                            b_l1,
                            accumulator,
                            transpose_B=transpose_b,
                            clear_accum=panel_k == 0,
                        )
        else:
            if cache_a:
                T.copy(A[mi * tile_m : (mi + 1) * tile_m, 0:a_k_extent], a_l1)
            for ki in T.Pipelined(T.ceildiv(k, tile_k)):
                if not cache_a:
                    T.copy(A[mi * tile_m : (mi + 1) * tile_m, ki * tile_k : (ki + 1) * tile_k], a_l1)
                if transpose_b:
                    T.copy(B[ni * tile_n : (ni + 1) * tile_n, ki * tile_k : (ki + 1) * tile_k], b_l1)
                else:
                    T.copy(B[ki * tile_k : (ki + 1) * tile_k, ni * tile_n : (ni + 1) * tile_n], b_l1)
                if cache_a:
                    T.gemm(
                        a_l1[:, ki * tile_k : (ki + 1) * tile_k],
                        b_l1,
                        accumulator,
                        transpose_B=transpose_b,
                        clear_accum=ki == 0,
                    )
                else:
                    T.gemm(a_l1, b_l1, accumulator, transpose_B=transpose_b, clear_accum=ki == 0)
        T.copy(accumulator, C[mi * tile_m : (mi + 1) * tile_m, ni * tile_n : (ni + 1) * tile_n])

    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), dtype)):
        with T.Kernel(num_groups) as block:
            if pipeline_outputs:
                a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
                b_l1 = T.alloc_l1((tile_n, tile_k) if transpose_b else (tile_k, tile_n), dtype)
                accumulator = T.alloc_l0c((tile_m, tile_n), "float32")
                for task in T.Pipelined(T.ceildiv(grid - block, num_groups) * k_panels):
                    tile_id = block + (task // k_panels) * num_groups
                    mi = tile_id // n_tiles
                    ni = tile_id % n_tiles
                    ki = task % k_panels
                    T.copy(A[mi * tile_m : (mi + 1) * tile_m, ki * tile_k : (ki + 1) * tile_k], a_l1)
                    if transpose_b:
                        T.copy(B[ni * tile_n : (ni + 1) * tile_n, ki * tile_k : (ki + 1) * tile_k], b_l1)
                    else:
                        T.copy(B[ki * tile_k : (ki + 1) * tile_k, ni * tile_n : (ni + 1) * tile_n], b_l1)
                    T.gemm(a_l1, b_l1, accumulator, transpose_B=transpose_b, clear_accum=ki == 0)
                    if ki == k_panels - 1:
                        T.copy(accumulator, C[mi * tile_m : (mi + 1) * tile_m, ni * tile_n : (ni + 1) * tile_n])
            elif num_cores is None:
                output_tile(A, B, C, block)
            else:
                # A per-core finite extent avoids empty output iterations and
                # a dynamic guard around the complete automatic K pipeline.
                for iteration in T.serial(T.ceildiv(grid - block, num_groups)):
                    output_tile(A, B, C, block + iteration * num_groups)

    suffix = "_cache_a" if cache_a else ""
    if a_cache_k is not None:
        suffix += f"_a_cache_k_{a_cache_k}"
    if num_cores is not None:
        suffix += f"_cores_{num_groups}"
    if pipeline_outputs:
        suffix += "_pipeline_outputs"
    return main.with_attr("global_symbol", f"gemm_{m}_{n}_{k}_{dtype}{suffix}")
