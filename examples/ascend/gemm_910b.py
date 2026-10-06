"""FP16/BF16 GEMM with ordinary automatic pipelining on dav-2201.

The Python kernel chooses logical tiles. Native sub-K movement, L0 scratch,
buffer versions and synchronization belong to the compiler. Compile the
returned PrimFunc with target={"kind": "ascend", "arch": "dav-2201"}.
"""


def gemm(m, n, k, *, dtype="bfloat16", transpose_b=True, tile_m=128, tile_n=128, tile_k=256):
    """Build C=A@B.T or C=A@B, with FP32 accumulation and dtype output.

    A is [m,k]; B is [n,k] for transpose_b=True or [k,n] otherwise.
    Logical dimensions can have tails. Tile sizes are Python algorithm
    choices; the target's resource planner diagnoses a tile that cannot fit.
    No manual stages, buffer versions or physical-device choice are required.
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
    b_shape = (n, k) if transpose_b else (k, n)
    n_tiles = (n + tile_n - 1) // tile_n
    grid = ((m + tile_m - 1) // tile_m) * n_tiles

    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), dtype)):
        with T.Kernel(grid) as block:
            mi = block // n_tiles
            ni = block % n_tiles
            a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_n, tile_k) if transpose_b else (tile_k, tile_n), dtype)
            accumulator = T.alloc_l0c((tile_m, tile_n), "float32")
            for ki in T.Pipelined(T.ceildiv(k, tile_k)):
                T.copy(A[mi * tile_m : (mi + 1) * tile_m, ki * tile_k : (ki + 1) * tile_k], a_l1)
                if transpose_b:
                    T.copy(B[ni * tile_n : (ni + 1) * tile_n, ki * tile_k : (ki + 1) * tile_k], b_l1)
                else:
                    T.copy(B[ki * tile_k : (ki + 1) * tile_k, ni * tile_n : (ni + 1) * tile_n], b_l1)
                T.gemm(a_l1, b_l1, accumulator, transpose_B=transpose_b, clear_accum=ki == 0)
            T.copy(accumulator, C[mi * tile_m : (mi + 1) * tile_m, ni * tile_n : (ni + 1) * tile_n])

    return main.with_attr("global_symbol", f"gemm_{m}_{n}_{k}_{dtype}")
