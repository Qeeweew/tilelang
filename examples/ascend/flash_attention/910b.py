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


def _default_block_kv(kv_len):
    # Cover short sequences without paying for a wider single KV tile. Long
    # sequences amortize online-state updates and cross-core handoffs over 256
    # keys. This is an algorithm starting point, not a resource/performance model.
    return 64 if kv_len <= 64 else 128 if kv_len <= 128 else 256


def _default_tiling(query_heads, kv_heads, query_len, kv_len, head_dim):
    block_q = min(64, ((query_heads // kv_heads + 15) // 16) * 16) if query_len == 1 else 64
    # Bound the logical FP32 output/state tile independently of the KV width.
    # This targets 8192 elements, subject to the minimum 16-row Cube tile;
    # compiler versions, scratch and physical resource legality remain separate.
    block_q = min(block_q, max(16, (8192 // head_dim // 16) * 16))
    return FwdTiling(block_q, _default_block_kv(kv_len), 4)


def _as_tiling(tiling):
    if tiling is None:
        return FwdTiling()
    if isinstance(tiling, FwdTiling):
        return tiling
    if isinstance(tiling, dict):
        fields = {"block_q", "block_kv", "max_stages"}
        unknown, missing = set(tiling) - fields, fields - set(tiling)
        if unknown or missing:
            raise ValueError(
                "tiling requires exactly block_q, block_kv, max_stages; "
                f"unknown fields={sorted(map(str, unknown))}, missing fields={sorted(missing)}"
            )
        return FwdTiling(**tiling)
    raise ValueError("tiling must be a FwdTiling or a dictionary of its three integer fields")


def _check_query_grouping(value):
    if not isinstance(value, int) or isinstance(value, bool) or value not in (1, 2):
        raise ValueError("query_grouping must be the integer 1 or 2")


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


def get_tuning_configs(
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
    query_groupings=(1,),
):
    """Return complete JSON kwargs for existing AutoTuner.from_kernel.

    Try at most the three KV widths 64/128/256, keeping the default query tile
    and stage bound. Widths beyond the smallest one covering a short sequence
    add padding without reducing its trip count and are omitted. Compilation
    still decides physical resource legality; this is not a planner or a claim
    that any candidate is optimal. Explicit FwdTiling configurations remain
    available for other search spaces. query_groupings is a nonempty tuple or
    list of unique integers from (1, 2); the default searches only grouping 1.
    """
    scale = _contract(batch, query_heads, kv_heads, query_len, kv_len, head_dim, dtype, causal, query_start, softmax_scale, FwdTiling())
    if variant not in ("generic", "pipeline"):
        raise ValueError("variant must be 'generic' or 'pipeline'")
    if not isinstance(query_groupings, (tuple, list)) or not query_groupings:
        raise ValueError("query_groupings must be a nonempty tuple or list of unique 1/2 values")
    for grouping in query_groupings:
        _check_query_grouping(grouping)
    if len(set(query_groupings)) != len(query_groupings):
        raise ValueError("query_groupings must contain unique values")
    default = _default_tiling(query_heads, kv_heads, query_len, kv_len, head_dim)
    problem = dict(
        batch=batch, query_heads=query_heads, kv_heads=kv_heads,
        query_len=query_len, kv_len=kv_len, head_dim=head_dim,
        dtype=dtype, variant=variant, causal=bool(causal), query_start=query_start,
        softmax_scale=None if softmax_scale is None else scale,
    )
    return [
        dict(problem, query_grouping=grouping, tiling=dict(block_q=default.block_q, block_kv=width, max_stages=default.max_stages))
        for grouping in query_groupings
        for width in (64, 128, 256) if width <= default.block_kv
    ]


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
    query_grouping=1,
):
    """Build Q[B,Hq,Sq,D], K/V[B,Hkv,Sk,D] -> O[B,Hq,Sq,D].

    Consecutive groups of Hq/Hkv query heads share one KV head. A causal key
    is visible when its index <= query_start + query index. Accumulators and
    statistics are FP32; probabilities entering the second GEMM and output
    are rounded to dtype. Both variants retain the same typed math while
    exposing different issue boundaries. With tiling=None, KV width is the
    smallest of 64/128/256 covering Sk, or 256 for longer sequences. The query
    tile targets 8192 output elements with a minimum of 16 aligned rows.
    Explicit FwdTiling or a dictionary of all three fields overrides defaults.
    query_grouping=2 lets two adjacent query tiles in one head/KV-head group
    share each L1 K/V import. Each query retains its own UB component and
    online state; resource feasibility remains a compiler contract.
    """
    default_tiling = tiling is None
    tiling = _as_tiling(tiling)
    scale = _contract(batch, query_heads, kv_heads, query_len, kv_len, head_dim, dtype, causal, query_start, softmax_scale, tiling)
    if default_tiling:
        tiling = _default_tiling(query_heads, kv_heads, query_len, kv_len, head_dim)
    if variant not in ("generic", "pipeline"):
        raise ValueError("variant must be 'generic' or 'pipeline'")
    _check_query_grouping(query_grouping)

    from tilelang.ascend import language as T

    groups = query_heads // kv_heads
    packed_decode = query_len == 1
    # With one query position, adjacent GQA heads are contiguous rows in
    # BNSD storage. One head tile shares its K/V scan and useful Cube rows.
    block_q = tiling.block_q
    block_kv, max_stages = tiling.block_kv, tiling.max_stages
    logical_rows = groups if packed_decode else query_len
    group_rows = block_q * query_grouping
    q_tiles = (logical_rows + group_rows - 1) // group_rows
    kv_tiles = (kv_len + block_kv - 1) // block_kv

    @T.macro
    def query_step(key_base, first_query, qrows, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha):
        T.gemm(q_l1, k_l1, score_acc, transpose_B=True, clear_accum=True)
        # The logical producer retains its complete allocation. Its actual
        # read region supplies the one valid row domain to the GM bridge and
        # both AIV owners, including a tail with an empty second owner.
        # DMA input is read only by scale/mask. Destructive softmax work then
        # uses scores, so its reductions do not extend this input's lifetime.
        T.copy(score_acc[0:qrows, :], score_input)
        if not causal and kv_len % block_kv == 0:
            # Every key in each Python-selected tile belongs to the input.
            # Keep this dense case free of a redundant per-lane predicate.
            for i, j in T.Parallel(block_q, block_kv):
                scores[i, j] = score_input[i, j] * T.float32(scale)
        else:
            # Every row in this logical query group can see the complete KV
            # tile. Use the group's first query, independent of AIV ownership.
            # Subtraction is equivalent to key_base+B <= first_query+1 but
            # avoids an overflowing endpoint in the final Int64 KV tile.
            if causal and (T.cast(key_base, "int64") <= first_query + T.int64(1) - T.int64(block_kv)):
                for i, j in T.Parallel(block_q, block_kv):
                    scores[i, j] = score_input[i, j] * T.float32(scale)
            else:
                for i, j in T.Parallel(block_q, block_kv):
                    scores[i, j] = T.if_then_else(
                        (key_base + j < kv_len) & ((not causal) | (key_base + j <= first_query + (0 if packed_decode else i))),
                        score_input[i, j] * T.float32(scale),
                        -T.infinity("float32"),
                    )
        T.reduce_max(scores, next_maximum, dim=1, clear=True)
        for i in T.Parallel(block_q):
            next_maximum[i] = T.max(maximum[i], next_maximum[i])
            alpha[i] = T.exp(maximum[i] - next_maximum[i])
        for i, j in T.Parallel(block_q, block_kv):
            # The unnormalized score is dead after row max. Expose its
            # replacement before exp so this ordinary expression does not
            # need another full score tile for the subtraction result.
            scores[i, j] = scores[i, j] - next_maximum[i]
        for i, j in T.Parallel(block_q, block_kv):
            scores[i, j] = T.exp(scores[i, j])
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

    @T.macro
    def kv_step(kv_tile, batch_id, kv_head, query_base, qrows, active_kv_tiles, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha):
        key_base = kv_tile * block_kv
        first_query = T.int64(query_start) if packed_decode else T.int64(query_start) + T.cast(query_base, "int64")
        # One input epoch serves all present query readers in this group.
        T.copy(K[batch_id, kv_head, key_base : key_base + block_kv, :], k_l1)
        T.copy(V[batch_id, kv_head, key_base : key_base + block_kv, :], v_l1)
        if query_grouping == 2 and causal and not packed_decode:
            # The second query may visit more causal KV tiles. Do not append
            # completely invisible steps to the first query's recurrence.
            if kv_tile < active_kv_tiles:
                query_step(key_base, first_query, qrows, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)
        else:
            query_step(key_base, first_query, qrows, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)

    @T.macro
    def finish_query(batch_id, query_head, query_base, qrows, O_target, output, output_cast, denominator):
        for i, j in T.Parallel(block_q, head_dim):
            output_cast[i, j] = T.cast(output[i, j] / denominator[i], dtype)
        if packed_decode:
            T.copy(output_cast[0:qrows, :], O_target[batch_id, query_head + query_base : query_head + query_base + qrows, :])
        else:
            T.copy(output_cast[0:qrows, :], O_target[batch_id, query_head, query_base : query_base + qrows, :])

    @T.prim_func
    def main(
        Q: T.Tensor((batch, query_heads, query_len, head_dim), dtype),
        K: T.Tensor((batch, kv_heads, kv_len, head_dim), dtype),
        V: T.Tensor((batch, kv_heads, kv_len, head_dim), dtype),
        O: T.Tensor((batch, query_heads, query_len, head_dim), dtype),
    ):
        with T.Kernel(batch * (kv_heads if packed_decode else query_heads) * q_tiles) as core:
            if packed_decode:
                batch_id = core // (kv_heads * q_tiles)
                kv_head = core // q_tiles % kv_heads
                query_head = kv_head * groups
            else:
                batch_id = core // (query_heads * q_tiles)
                query_head = core // q_tiles % query_heads
                kv_head = query_head // groups
            query_base = core % q_tiles * group_rows
            qrows = T.min(block_q, logical_rows - query_base)
            if causal:
                # Exclusive largest visible key across this group's valid
                # queries. Keep this group-uniform, including an empty AIV
                # partition; it is unrelated to each SID's private row count.
                visible_end = T.int64(query_start + 1) if packed_decode else T.int64(query_start) + T.cast(query_base, "int64") + T.cast(qrows, "int64")
                kv_width = T.int64(block_kv)
                # Quotient plus remainder avoids visible_end + B - 1 overflow.
                active_kv_tiles = T.min(T.int64(kv_tiles), visible_end // kv_width + T.cast(visible_end % kv_width != 0, "int64"))
            else:
                active_kv_tiles = kv_tiles

            if query_grouping == 2:
                query_base_second = query_base + block_q
                qrows_second = T.min(block_q, T.max(0, logical_rows - query_base_second))
                if causal and not packed_decode:
                    # The last existing query has the longest visible prefix.
                    # This stays within the actual sequence on an odd pair.
                    group_visible_end = T.int64(query_start) + T.cast(query_base, "int64") + T.cast(T.min(group_rows, logical_rows - query_base), "int64")
                    group_kv_tiles = T.min(T.int64(kv_tiles), group_visible_end // kv_width + T.cast(group_visible_end % kv_width != 0, "int64"))
                else:
                    group_kv_tiles = active_kv_tiles

            # Q is reused across KV iterations. Tile shape and this cache
            # policy are algorithm choices expressed here in Python.
            q_l1 = T.alloc_l1((block_q, head_dim), dtype)
            k_l1 = T.alloc_l1((block_kv, head_dim), dtype)
            v_l1 = T.alloc_l1((block_kv, head_dim), dtype)
            p_l1 = T.alloc_l1((block_q, block_kv), dtype)
            score_acc = T.alloc_l0c((block_q, block_kv), "float32")
            pv_acc = T.alloc_l0c((block_q, head_dim), "float32")
            score_input = T.alloc_shared((block_q, block_kv), "float32")
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

            if query_grouping == 2:
                # Distinct leading-row components preserve different tail
                # domains. Any physical reuse requires compiler completion
                # proof; a query axis is not an AIV owner axis.
                q_l1_second = T.alloc_l1((block_q, head_dim), dtype)
                score_input_second = T.alloc_shared((block_q, block_kv), "float32")
                scores_second = T.alloc_shared((block_q, block_kv), "float32")
                p_second = T.alloc_shared((block_q, block_kv), dtype)
                pv_second = T.alloc_shared((block_q, head_dim), "float32")
                output_second = T.alloc_shared((block_q, head_dim), "float32")
                output_cast_second = T.alloc_shared((block_q, head_dim), dtype)
                maximum_second = T.alloc_shared((block_q,), "float32")
                next_maximum_second = T.alloc_shared((block_q,), "float32")
                denominator_second = T.alloc_shared((block_q,), "float32")
                row_sum_second = T.alloc_shared((block_q,), "float32")
                alpha_second = T.alloc_shared((block_q,), "float32")

            if packed_decode:
                # These views share the original parameter data; callers keep
                # their BNSD tensors. Initialize unused physical Cube rows.
                Q_heads = T.reshape(Q, (batch, query_heads, head_dim))
                O_heads = T.reshape(O, (batch, query_heads, head_dim))
                T.fill(q_l1, 0)
                T.copy(Q_heads[batch_id, query_head + query_base : query_head + query_base + qrows, :], q_l1[0:qrows, :])
            else:
                T.copy(Q[batch_id, query_head, query_base : query_base + block_q, :], q_l1)
            T.fill(output, 0)
            T.fill(denominator, 0)
            T.fill(maximum, -T.infinity("float32"))
            if query_grouping == 2:
                if qrows_second > 0:
                    if packed_decode:
                        T.fill(q_l1_second, 0)
                        T.copy(Q_heads[batch_id, query_head + query_base_second : query_head + query_base_second + qrows_second, :], q_l1_second[0:qrows_second, :])
                    else:
                        T.copy(Q[batch_id, query_head, query_base_second : query_base_second + block_q, :], q_l1_second)
                    T.fill(output_second, 0)
                    T.fill(denominator_second, 0)
                    T.fill(maximum_second, -T.infinity("float32"))
            if variant == "generic":
                for kv_tile in T.serial(group_kv_tiles if query_grouping == 2 else active_kv_tiles):
                    kv_step(kv_tile, batch_id, kv_head, query_base, qrows, active_kv_tiles, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)
                    if query_grouping == 2:
                        if qrows_second > 0:
                            query_step(kv_tile * block_kv, T.int64(query_start) if packed_decode else T.int64(query_start) + T.cast(query_base_second, "int64"), qrows_second, q_l1_second, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input_second, scores_second, p_second, pv_second, output_second, maximum_second, next_maximum_second, denominator_second, row_sum_second, alpha_second)
            else:
                for kv_tile in T.Pipelined(group_kv_tiles if query_grouping == 2 else active_kv_tiles, max_stages=max_stages):
                    kv_step(kv_tile, batch_id, kv_head, query_base, qrows, active_kv_tiles, K, V, q_l1, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input, scores, p, pv, output, maximum, next_maximum, denominator, row_sum, alpha)
                    if query_grouping == 2:
                        if qrows_second > 0:
                            query_step(kv_tile * block_kv, T.int64(query_start) if packed_decode else T.int64(query_start) + T.cast(query_base_second, "int64"), qrows_second, q_l1_second, k_l1, v_l1, p_l1, score_acc, pv_acc, score_input_second, scores_second, p_second, pv_second, output_second, maximum_second, next_maximum_second, denominator_second, row_sum_second, alpha_second)

            if packed_decode:
                finish_query(batch_id, query_head, query_base, qrows, O_heads, output, output_cast, denominator)
            else:
                finish_query(batch_id, query_head, query_base, qrows, O, output, output_cast, denominator)
            if query_grouping == 2:
                if qrows_second > 0:
                    if packed_decode:
                        finish_query(batch_id, query_head, query_base_second, qrows_second, O_heads, output_second, output_cast_second, denominator_second)
                    else:
                        finish_query(batch_id, query_head, query_base_second, qrows_second, O, output_second, output_cast_second, denominator_second)

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


def attention_reference(q, k, v, *, dtype="bfloat16", block_kv=None, causal=False, query_start=0, softmax_scale=None):
    """CPU oracle for the shared FP32/rounded-block-probability contract.

    This models numerical operations, rather than instruction accuracy or
    device execution. Inputs and final output are rounded to dtype. Choosing
    another block_kv changes the probability-rounding points of the algorithm.
    None uses the same sequence-width default as flash_attention_fwd.
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
    if block_kv is None:
        block_kv = _default_block_kv(kv_len)
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
