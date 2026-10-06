"""Ordinary TileLang forward attention for dav-2201.

The two variants share an online-softmax algorithm and input contract. Their
KV loop is serial or automatically pipelined; neither owns AIV partitions,
instruction scratch, buffer versions, synchronization, or a fused softmax.
See 910b.md for the numerical contract and validation requirements.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FwdTiling:
    block_q: int = 64
    block_kv: int = 64
    max_stages: int = 4


def _contract(batch, query_heads, kv_heads, query_len, kv_len, head_dim, dtype, causal, query_start, softmax_scale, tiling):
    dimensions = (batch, query_heads, kv_heads, query_len, kv_len, head_dim)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
        raise ValueError("Attention dimensions must be positive integers")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dav-2201 attention supports float16 or bfloat16 operands")
    if not isinstance(query_start, int) or isinstance(query_start, bool) or query_start < 0:
        raise ValueError("query_start must be a nonnegative integer")
    if causal and query_start + query_len > kv_len:
        raise ValueError("Causal query positions must lie within the KV sequence")
    for name, value in (("block_q", tiling.block_q), ("block_kv", tiling.block_kv), ("head_dim", head_dim)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value % 16:
            raise ValueError(f"{name} must be a positive multiple of 16")
    if not isinstance(tiling.max_stages, int) or isinstance(tiling.max_stages, bool) or tiling.max_stages < 1:
        raise ValueError("max_stages must be a positive integer")
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    return scale


def flash_attention_fwd(
    batch,
    query_heads,
    kv_heads,
    query_len,
    kv_len,
    head_dim,
    *,
    dtype="bfloat16",
    variant="generic",
    causal=False,
    query_start=0,
    softmax_scale=None,
    tiling=None,
):
    """Build Q[B,Hq,Sq,D], K/V[B,Hkv,Sk,D] -> O[B,Hq,Sq,D].

    Consecutive groups of Hq/Hkv query heads share one KV head. A causal key
    is visible when its index <= query_start + query index. Accumulators and
    statistics are FP32; probabilities entering the second GEMM and output
    are rounded to dtype. Both variants retain the same typed math while
    exposing different issue boundaries.
    """
    tiling = tiling or FwdTiling()
    scale = _contract(batch, query_heads, kv_heads, query_len, kv_len, head_dim, dtype, causal, query_start, softmax_scale, tiling)
    if variant not in ("generic", "pipeline"):
        raise ValueError("variant must be 'generic' or 'pipeline'")

    from tilelang.ascend import language as T

    block_q, block_kv = tiling.block_q, tiling.block_kv
    max_stages = tiling.max_stages
    groups = query_heads // kv_heads
    q_tiles = (query_len + block_q - 1) // block_q
    kv_tiles = (kv_len + block_kv - 1) // block_kv

    @T.macro
    def kv_step(kv_tile, batch_id, query_head, kv_head, query_base, qrows, Q, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha):
        key_base = kv_tile * block_kv
        T.copy(K[batch_id, kv_head, key_base : key_base + block_kv, :], k_l1)
        T.copy(V[batch_id, kv_head, key_base : key_base + block_kv, :], v_l1)
        T.gemm(q_l1, k_l1, score_acc, transpose_B=True, clear_accum=True)
        # The logical producer retains its complete allocation. Its actual
        # read region supplies the one valid row domain to the GM bridge and
        # both AIV owners, including a tail with an empty second owner.
        T.copy(score_acc[0:qrows, :], scores)
        if not causal and kv_len % block_kv == 0:
            # Every key in each Python-selected tile belongs to the input.
            # Keep this dense case free of a redundant per-lane predicate.
            for i, j in T.Parallel(block_q, block_kv):
                scores[i, j] = scores[i, j] * T.float32(scale)
        else:
            for i, j in T.Parallel(block_q, block_kv):
                scores[i, j] = T.if_then_else(
                    (key_base + j < kv_len) & ((not causal) | (key_base + j <= query_start + query_base + i)),
                    scores[i, j] * T.float32(scale),
                    -T.infinity("float32"),
                )
        T.reduce_max(scores, next_maximum, dim=1, clear=True)
        for i in T.Parallel(block_q):
            next_maximum[i] = T.max(maximum[i], next_maximum[i])
            alpha[i] = T.exp(maximum[i] - next_maximum[i])
        for i, j in T.Parallel(block_q, block_kv):
            scores[i, j] = T.exp(scores[i, j] - next_maximum[i])
            p[i, j] = T.cast(scores[i, j], dtype)
        T.reduce_sum(scores, row_sum, dim=1, clear=True)
        for i in T.Parallel(block_q):
            denominator[i] = denominator[i] * alpha[i] + row_sum[i]
        # This is a state copy with exact valid-row effects, not arithmetic.
        # Keep the same epoch-carried maximum buffer for the next KV block.
        T.copy(next_maximum[0:qrows], maximum[0:qrows])
        if variant == "pipeline":
            # Independent FP32 rescale: this task does not read the future PV.
            # The same output epoch is completed by the later FP32 add.
            for i, j in T.Parallel(block_q, head_dim):
                output[i, j] = alpha[i] * output[i, j]
        # Unwritten Q-tail rows of P's L1 operand are initialized by the
        # compiler's physical bridge. V remains the ordinary [K,N] operand.
        T.copy(p[0:qrows, :], p_l1)
        T.gemm(p_l1, v_l1, pv_acc, transpose_B=False, clear_accum=True)
        T.copy(pv_acc[0:qrows, :], pv)
        if variant == "generic":
            for i, j in T.Parallel(block_q, head_dim):
                output[i, j] = alpha[i] * output[i, j] + pv[i, j]
        else:
            for i, j in T.Parallel(block_q, head_dim):
                output[i, j] = output[i, j] + pv[i, j]

    @T.prim_func
    def main(
        Q: T.Tensor((batch, query_heads, query_len, head_dim), dtype),
        K: T.Tensor((batch, kv_heads, kv_len, head_dim), dtype),
        V: T.Tensor((batch, kv_heads, kv_len, head_dim), dtype),
        O: T.Tensor((batch, query_heads, query_len, head_dim), dtype),
    ):
        with T.Kernel(batch * query_heads * q_tiles) as core:
            batch_id = core // (query_heads * q_tiles)
            query_head = core // q_tiles % query_heads
            kv_head = query_head // groups
            query_base = core % q_tiles * block_q
            qrows = T.min(block_q, query_len - query_base)

            # Q is reused across KV iterations. Tile shape and this cache
            # policy are algorithm choices expressed here in Python.
            q_l1 = T.alloc_l1((block_q, head_dim), dtype)
            k_l1 = T.alloc_l1((block_kv, head_dim), dtype)
            v_l1 = T.alloc_l1((block_kv, head_dim), dtype)
            p_l1 = T.alloc_l1((block_q, block_kv), dtype)
            score_acc = T.alloc_l0c((block_q, block_kv), "float32")
            pv_acc = T.alloc_l0c((block_q, head_dim), "float32")
            scores = T.alloc_shared((block_q, block_kv), "float32")
            p = T.alloc_shared((block_q, block_kv), dtype)
            pv = T.alloc_shared((block_q, head_dim), "float32")
            output = T.alloc_shared((block_q, head_dim), "float32")
            output_cast = T.alloc_shared((block_q, head_dim), dtype)
            maximum = T.alloc_shared((block_q,), "float32")
            next_maximum = T.alloc_shared((block_q,), "float32")
            denominator = T.alloc_shared((block_q,), "float32")
            row_sum = T.alloc_shared((block_q,), "float32")
            alpha = T.alloc_shared((block_q,), "float32")

            T.copy(Q[batch_id, query_head, query_base : query_base + block_q, :], q_l1)
            T.fill(output, 0)
            T.fill(denominator, 0)
            T.fill(maximum, -T.infinity("float32"))
            if variant == "generic":
                for kv_tile in T.serial(kv_tiles):
                    kv_step(kv_tile, batch_id, query_head, kv_head, query_base, qrows, Q, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)
            else:
                for kv_tile in T.Pipelined(kv_tiles, max_stages=max_stages):
                    kv_step(kv_tile, batch_id, query_head, kv_head, query_base, qrows, Q, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)

            for i, j in T.Parallel(block_q, head_dim):
                output_cast[i, j] = T.cast(output[i, j] / denominator[i], dtype)
            T.copy(output_cast[0:qrows, :], O[batch_id, query_head, query_base : query_base + qrows, :])

    return main


def _round_dtype(values, dtype):
    """CPU operand/output rounding, including BF16 round-to-nearest-even."""
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def attention_reference(q, k, v, *, dtype="bfloat16", block_kv=64, causal=False, query_start=0, softmax_scale=None):
    """CPU oracle for the shared FP32/rounded-block-probability contract.

    This models numerical operations, rather than instruction accuracy or
    device execution. Inputs and final output are rounded to dtype. Choosing
    another block_kv changes the probability-rounding points of the algorithm.
    Operand-rounded inputs and FP32 scores must be finite. Masked scores and
    the initial running maximum still use their defined negative infinities.
    """
    import numpy as np

    q, k, v = (np.asarray(value, dtype=np.float32) for value in (q, k, v))
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("Q/K/V must be rank four, with matching K/V shapes")
    batch, query_heads, query_len, head_dim = q.shape
    if k.shape[0] != batch or k.shape[3] != head_dim:
        raise ValueError("Q/K/V batch and head_dim must agree")
    kv_heads, kv_len = k.shape[1:3]
    scale = _contract(batch, query_heads, kv_heads, query_len, kv_len, head_dim, dtype, causal, query_start, softmax_scale, FwdTiling(block_kv=block_kv))
    with np.errstate(over="ignore", invalid="ignore"):
        q, k, v = (_round_dtype(value, dtype) for value in (q, k, v))
    if not all(np.isfinite(value).all() for value in (q, k, v)):
        raise ValueError("Q/K/V must be finite after operand dtype rounding")
    head_map = np.arange(query_heads) // (query_heads // kv_heads)
    k, v = k[:, head_map], v[:, head_map]
    maximum = np.full((batch, query_heads, query_len, 1), -np.inf, dtype=np.float32)
    denominator = np.zeros_like(maximum)
    output = np.zeros(q.shape, dtype=np.float32)
    query_positions = query_start + np.arange(query_len)[:, None]
    for start in range(0, kv_len, block_kv):
        stop = min(start + block_kv, kv_len)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = (q @ k[:, :, start:stop].swapaxes(-1, -2)) * np.float32(scale)
        if not np.isfinite(scores).all():
            raise ValueError("The attention contract requires finite FP32 scores before masking")
        if causal:
            scores = np.where(np.arange(start, stop)[None, :] <= query_positions, scores, -np.inf)
        next_maximum = np.maximum(maximum, scores.max(axis=-1, keepdims=True))
        alpha = np.exp(maximum - next_maximum)
        probabilities = np.exp(scores - next_maximum)
        denominator = denominator * alpha + probabilities.sum(axis=-1, keepdims=True, dtype=np.float32)
        output = output * alpha + _round_dtype(probabilities, dtype) @ v[:, :, start:stop]
        maximum = next_maximum
    return _round_dtype(output / denominator, dtype)
