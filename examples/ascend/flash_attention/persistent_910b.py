"""Persistent paired-query FlashAttention forward for dav-2201.

Each mixed-core group traverses a divisible share of the query-pair grid.
Its AIVs own separate complete query tiles, with group-private GM handoffs
and FP32 output state. Explicit phase and ring hints still use the ordinary
Ascend lowering, AutoSchedule, capacity checks, and synchronization passes.
See 910b.md for the input, workspace, and completion contract.
"""


def persistent_attention_fwd(
    *,
    batch=1,
    heads=4,
    query_pairs=16,
    kv_tiles=32,
    groups=16,
    block_q=128,
    block_kv=128,
    head_dim=64,
    row_block=64,
    row_block_vec2=None,
    separate_score_input=False,
    inplace_final_division=True,
    qk_chunk=None,
    pv_chunk=None,
    pv_rows=None,
    cache_pv=False,
    preserve_issue_order=False,
    dtype="float16",
):
    """Build dense MHA with explicit group-private workspace arguments.

    Q/O are contiguous [B,H,query_pairs,2,block_q,D] views of BNSD tensors;
    K/V are [B,H,kv_tiles*block_kv,D]. The returned PrimFunc takes arguments
    (Q, K, V, Score, P, PV, OState, O). Score/P use
    [groups,2,2,block_q,block_kv]; PV/OState use
    [groups,2,2,block_q,D]. Score, PV, and OState are FP32, while P, Q/K/V,
    and O use dtype. The two middle workspace axes are AIV owner and slot.

    Dimensions are positive integers, tile dimensions are multiples of 16,
    row_block and row_block_vec2 divide block_q, and groups divides
    B*H*query_pairs. This
    factory has no causal mask, GQA, dropout, or sequence tails. Resource
    feasibility is checked by the target compiler, not by these shape rules.

    By default V is staged with QK and PV consumes a complete logical KV tile.
    An explicit positive, 16-aligned pv_chunk dividing block_kv instead stages
    V during PV, once per chunk for both query owners. Each owner accumulates
    all chunks in its own FP32 L0C region before one final PV export. Softmax
    and P rounding still cover the full logical block_kv; pv_chunk controls
    L1 staging rather than native Cube K tiling.

    Vec1 row_block and Vec2 row_block_vec2 are independent 8-aligned UB
    grains; row_block_vec2 defaults to row_block. separate_score_input keeps
    the DMA input separate from the destructive softmax work tile. Final
    division defaults to exact in-place FP32 output before the dtype cast.
    These choices preserve the same online recurrence and P rounding.

    qk_chunk optionally retains Q in L0A while computing complete N chunks.
    cache_pv=True instead retains the complete V tile in L1 and loads one
    full-KV P cache per pv_rows part. An ordinary automatic Pipelined loop
    consumes pv_chunk slices and exports each complete FP32 result once.
    pv_rows defaults to block_q and cache_pv requires pv_chunk. Tile/cache
    choices are Python policy; the compiler checks physical feasibility.
    preserve_issue_order fixes the complete outer task issue order instead
    of optimizing it. The compiler deterministically completes unspecified
    versions while preserving fixed versions, stage and issue order; illegal
    dependencies or insufficient capacity/version bounds are diagnosed.
    Nested automatic PV loops retain their own scheduling.
    """
    row_block_vec2 = row_block if row_block_vec2 is None else row_block_vec2
    pv_rows = block_q if pv_rows is None else pv_rows
    values = (batch, heads, query_pairs, kv_tiles, groups,
              block_q, block_kv, head_dim, row_block, row_block_vec2, pv_rows)
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
        raise ValueError("Persistent attention dimensions and groups must be positive integers")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("Persistent attention supports FP16/BF16 operands")
    if (any(x % 16 for x in (block_q, block_kv, head_dim, pv_rows))
            or block_q % pv_rows
            or any(x % 8 or block_q % x for x in (row_block, row_block_vec2))):
        raise ValueError("Aligned dense tiles and complete UB row subtiles are required")
    if any(not isinstance(x, bool) for x in
           (separate_score_input, inplace_final_division, cache_pv, preserve_issue_order)):
        raise ValueError("Score staging, final division, and PV cache options must be bools")
    if qk_chunk is not None and (not isinstance(qk_chunk, int) or isinstance(qk_chunk, bool)
                                 or qk_chunk <= 0 or qk_chunk % 16 or block_kv % qk_chunk):
        raise ValueError("qk_chunk must be positive, 16-aligned, and divide block_kv")
    if pv_chunk is not None and (not isinstance(pv_chunk, int) or isinstance(pv_chunk, bool) or pv_chunk <= 0):
        raise ValueError("pv_chunk must be a positive integer or None")
    if pv_chunk is not None and (pv_chunk % 16 or block_kv % pv_chunk):
        raise ValueError("pv_chunk must be 16-aligned and divide block_kv into complete PV chunks")
    if cache_pv and pv_chunk is None:
        raise ValueError("cache_pv requires an explicit pv_chunk")
    if not cache_pv and pv_rows != block_q:
        raise ValueError("pv_rows requires cache_pv=True")
    total_pairs = batch * heads * query_pairs
    if total_pairs % groups:
        raise ValueError(
            f"total_pairs={total_pairs} (batch={batch}, heads={heads}, "
            f"query_pairs={query_pairs}) must divide evenly across groups={groups}"
        )

    from tilelang.ascend import language as T

    pairs_per_group = total_pairs // groups
    tasks = pairs_per_group * kv_tiles
    sequence_k = kv_tiles * block_kv
    chunked_pv = pv_chunk is not None
    pv_extent = pv_chunk if chunked_pv else block_kv
    pv_chunks = block_kv // pv_extent
    pv_acc_shape = ((pv_rows, head_dim) if cache_pv else
                    (2, block_q, head_dim) if chunked_pv else (block_q, head_dim))
    p_shape = (pv_rows, block_kv) if cache_pv else (block_q, pv_extent)
    v_shape = (block_kv, head_dim) if cache_pv else (2, pv_extent, head_dim)
    qk_extent = block_kv if qk_chunk is None else qk_chunk
    scale = head_dim ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, query_pairs, 2, block_q, head_dim), dtype),
        K: T.Tensor((batch, heads, sequence_k, head_dim), dtype),
        V: T.Tensor((batch, heads, sequence_k, head_dim), dtype),
        Score: T.Tensor((groups, 2, 2, block_q, block_kv), "float32"),
        P: T.Tensor((groups, 2, 2, block_q, block_kv), dtype),
        PV: T.Tensor((groups, 2, 2, block_q, head_dim), "float32"),
        OState: T.Tensor((groups, 2, 2, block_q, head_dim), "float32"),
        O: T.Tensor((batch, heads, query_pairs, 2, block_q, head_dim), dtype),
    ):
        with T.MixedKernel(groups, sids=2) as (group, sid):
            q_l1 = T.alloc_l1((2, block_q, head_dim), dtype)
            k_l1 = T.alloc_l1((block_kv, head_dim), dtype)
            v_l1 = T.alloc_l1(v_shape, dtype)
            p_l1 = T.alloc_l1(p_shape, dtype)
            score_acc = T.alloc_l0c((block_q, qk_extent), "float32")
            pv_acc = T.alloc_l0c(pv_acc_shape, "float32")
            if qk_chunk is not None:
                qk_a_l0 = T.alloc_l0a((block_q, head_dim), dtype)
                qk_b_l0 = T.alloc_l0b((qk_extent, head_dim), dtype)
                T.annotate_buffer_versions({qk_a_l0: 1, qk_b_l0: 2})

            scores = T.alloc_shared((row_block, block_kv), "float32")
            probabilities = T.alloc_shared((row_block, block_kv), dtype)
            pv = T.alloc_shared((row_block_vec2, head_dim), "float32")
            output = T.alloc_shared((row_block_vec2, head_dim), "float32")
            output_cast = T.alloc_shared((row_block_vec2, head_dim), dtype)
            block_maximum = T.alloc_shared((row_block,), "float32")
            next_maximum = T.alloc_shared((row_block,), "float32")
            row_sum = T.alloc_shared((row_block,), "float32")
            maximum = T.alloc_shared((2, block_q), "float32")
            denominator = T.alloc_shared((2, block_q), "float32")
            alpha = T.alloc_shared((2, block_q), "float32")
            if separate_score_input:
                score_input = T.alloc_shared((row_block, block_kv), "float32")
                T.annotate_aiv_private(
                    score_input, scores, probabilities, pv, output, output_cast,
                    block_maximum, next_maximum, row_sum,
                    maximum, denominator, alpha,
                )
                T.annotate_buffer_versions({score_input: 1})
            else:
                T.annotate_aiv_private(
                    scores, probabilities, pv, output, output_cast,
                    block_maximum, next_maximum, row_sum,
                    maximum, denominator, alpha,
                )
            T.annotate_buffer_versions({
                q_l1: 1, k_l1: 1, p_l1: 1, score_acc: 1, pv_acc: 1,
                scores: 1, probabilities: 1, pv: 1, output: 1, output_cast: 1,
                block_maximum: 1, next_maximum: 1, row_sum: 1,
                maximum: 1, denominator: 1,
            })
            if cache_pv:
                T.annotate_buffer_versions({v_l1: 1})
            else:
                T.annotate_manual_multi_buffer({v_l1: 2})
            T.annotate_manual_multi_buffer({alpha: 2, Score: 2, P: 2, PV: 2})

            # Each group has one immutable clock through all of its pairs.
            # Local-pair parity owns the query state; flattened-pair parity
            # does not alternate when the group stride is even.
            for e in T.serial(tasks + 2, annotations=(
                    {"tl.ascend_preserve_issue_order": True} if preserve_issue_order else None)):
                with T.Stage(0):
                    if e < tasks:
                        with T.Cube():
                            local_pair = e // kv_tiles
                            flat_pair = group + local_pair * groups
                            batch_index = flat_pair // (heads * query_pairs)
                            head_index = (flat_pair // query_pairs) % heads
                            query_pair = flat_pair % query_pairs
                            kv = e % kv_tiles
                            task_slot = e % 2
                            if kv == 0:
                                for owner in T.serial(2):
                                    T.copy(Q[batch_index, head_index, query_pair, owner, :, :],
                                           q_l1[owner, :, :])
                            T.copy(K[batch_index, head_index,
                                     kv * block_kv:(kv + 1) * block_kv, :], k_l1)
                            if not chunked_pv:
                                T.copy(V[batch_index, head_index,
                                         kv * block_kv:(kv + 1) * block_kv, :],
                                       v_l1[task_slot, :, :])
                            for owner in T.serial(2):
                                if qk_chunk is not None:
                                    T.copy(q_l1[owner, :, :], qk_a_l0)
                                    for n in T.serial(block_kv // qk_extent):
                                        T.copy(k_l1[n * qk_extent:(n + 1) * qk_extent, :], qk_b_l0)
                                        T.gemm(qk_a_l0, qk_b_l0, score_acc,
                                               transpose_B=True, clear_accum=True)
                                        T.copy(score_acc, Score[group, owner, task_slot, :,
                                                                n * qk_extent:(n + 1) * qk_extent])
                                else:
                                    T.gemm(q_l1[owner, :, :], k_l1, score_acc,
                                           transpose_B=True, clear_accum=True)
                                    T.copy(score_acc, Score[group, owner, task_slot, :, :])

                with T.Stage(0):
                    if 1 <= e and e < tasks + 1:
                        t_vec1 = e - 1
                        kv_vec1 = t_vec1 % kv_tiles
                        query_slot_vec1 = (t_vec1 // kv_tiles) % 2
                        task_slot_vec1 = t_vec1 % 2
                        if kv_vec1 == 0:
                            T.fill(maximum[query_slot_vec1, :], -T.infinity("float32"))
                            T.fill(denominator[query_slot_vec1, :], 0)
                        for sub in T.serial(block_q // row_block):
                            row = sub * row_block
                            if separate_score_input:
                                T.copy(Score[group, sid, task_slot_vec1, row:row + row_block, :], score_input)
                                for i, j in T.Parallel(row_block, block_kv):
                                    scores[i, j] = score_input[i, j] * T.float32(scale)
                            else:
                                T.copy(Score[group, sid, task_slot_vec1, row:row + row_block, :], scores)
                                for i, j in T.Parallel(row_block, block_kv):
                                    scores[i, j] = scores[i, j] * T.float32(scale)
                            T.reduce_max(scores, block_maximum, dim=1, clear=True)
                            for i in T.Parallel(row_block):
                                next_maximum[i] = T.max(maximum[query_slot_vec1, row + i], block_maximum[i])
                                alpha[task_slot_vec1, row + i] = T.exp(maximum[query_slot_vec1, row + i] - next_maximum[i])
                            for i, j in T.Parallel(row_block, block_kv):
                                scores[i, j] = scores[i, j] - next_maximum[i]
                            for i, j in T.Parallel(row_block, block_kv):
                                scores[i, j] = T.exp(scores[i, j])
                                probabilities[i, j] = T.cast(scores[i, j], dtype)
                            T.reduce_sum(scores, row_sum, dim=1, clear=True)
                            for i in T.Parallel(row_block):
                                denominator[query_slot_vec1, row + i] = denominator[query_slot_vec1, row + i] * alpha[task_slot_vec1, row + i] + row_sum[i]
                            T.copy(next_maximum, maximum[query_slot_vec1, row:row + row_block])
                            T.copy(probabilities, P[group, sid, task_slot_vec1, row:row + row_block, :])

                with T.Stage(0):
                    if 1 <= e and e < tasks + 1:
                        with T.Cube():
                            if cache_pv:
                                t_pv = e - 1
                                local_pair_pv = t_pv // kv_tiles
                                flat_pair_pv = group + local_pair_pv * groups
                                batch_index_pv = flat_pair_pv // (heads * query_pairs)
                                head_index_pv = (flat_pair_pv // query_pairs) % heads
                                kv_pv = t_pv % kv_tiles
                                task_slot_pv = t_pv % 2
                                T.copy(V[batch_index_pv, head_index_pv,
                                         kv_pv * block_kv:(kv_pv + 1) * block_kv, :], v_l1)
                                for owner in T.serial(2):
                                    for part in T.serial(block_q // pv_rows):
                                        T.copy(P[group, owner, task_slot_pv,
                                                 part * pv_rows:(part + 1) * pv_rows, :], p_l1)
                                        for chunk in T.Pipelined(pv_chunks):
                                            T.gemm(p_l1[:, chunk * pv_extent:(chunk + 1) * pv_extent],
                                                   v_l1[chunk * pv_extent:(chunk + 1) * pv_extent, :],
                                                   pv_acc, transpose_B=False, clear_accum=(chunk == 0))
                                        T.copy(pv_acc, PV[group, owner, task_slot_pv,
                                                           part * pv_rows:(part + 1) * pv_rows, :])
                            elif chunked_pv:
                                t_pv = e - 1
                                local_pair_pv = t_pv // kv_tiles
                                flat_pair_pv = group + local_pair_pv * groups
                                batch_index_pv = flat_pair_pv // (heads * query_pairs)
                                head_index_pv = (flat_pair_pv // query_pairs) % heads
                                kv_pv = t_pv % kv_tiles
                                task_slot_pv = t_pv % 2
                                # A V chunk is loaded once and read by both owners.
                                # PV accumulators have an actual owner axis; their
                                # compiler version count stays one across chunks.
                                for chunk in T.serial(pv_chunks):
                                    chunk_begin = chunk * pv_chunk
                                    chunk_end = (chunk + 1) * pv_chunk
                                    v_slot = chunk % 2
                                    T.copy(V[batch_index_pv, head_index_pv,
                                             kv_pv * block_kv + chunk_begin:
                                             kv_pv * block_kv + chunk_end, :],
                                           v_l1[v_slot, :, :])
                                    for owner in T.serial(2):
                                        T.copy(P[group, owner, task_slot_pv, :,
                                                 chunk_begin:chunk_end], p_l1)
                                        T.gemm(p_l1, v_l1[v_slot, :, :],
                                               pv_acc[owner, :, :],
                                               transpose_B=False,
                                               clear_accum=(chunk == 0))
                                for owner in T.serial(2):
                                    T.copy(pv_acc[owner, :, :],
                                           PV[group, owner, task_slot_pv, :, :])
                            else:
                                task_slot_pv = (e - 1) % 2
                                for owner in T.serial(2):
                                    T.copy(P[group, owner, task_slot_pv, :, :], p_l1)
                                    T.gemm(p_l1, v_l1[task_slot_pv, :, :], pv_acc,
                                           transpose_B=False, clear_accum=True)
                                    T.copy(pv_acc, PV[group, owner, task_slot_pv, :, :])

                with T.Stage(0):
                    if 2 <= e:
                        t_vec2 = e - 2
                        local_pair_vec2 = t_vec2 // kv_tiles
                        flat_pair_vec2 = group + local_pair_vec2 * groups
                        batch_index_vec2 = flat_pair_vec2 // (heads * query_pairs)
                        head_index_vec2 = (flat_pair_vec2 // query_pairs) % heads
                        query_pair_vec2 = flat_pair_vec2 % query_pairs
                        query_slot_vec2 = local_pair_vec2 % 2
                        task_slot_vec2 = t_vec2 % 2
                        kv_vec2 = t_vec2 % kv_tiles
                        for sub in T.serial(block_q // row_block_vec2):
                            row = sub * row_block_vec2
                            T.copy(PV[group, sid, task_slot_vec2, row:row + row_block_vec2, :], pv)
                            if kv_vec2 == 0:
                                T.copy(pv, output)
                            else:
                                T.copy(OState[group, sid, query_slot_vec2, row:row + row_block_vec2, :], output)
                                for i, j in T.Parallel(row_block_vec2, head_dim):
                                    output[i, j] = output[i, j] * alpha[task_slot_vec2, row + i]
                                for i, j in T.Parallel(row_block_vec2, head_dim):
                                    output[i, j] = output[i, j] + pv[i, j]
                            if kv_vec2 + 1 < kv_tiles:
                                T.copy(output, OState[group, sid, query_slot_vec2, row:row + row_block_vec2, :])
                            else:
                                if inplace_final_division:
                                    for i, j in T.Parallel(row_block_vec2, head_dim):
                                        output[i, j] = output[i, j] / denominator[query_slot_vec2, row + i]
                                    for i, j in T.Parallel(row_block_vec2, head_dim):
                                        output_cast[i, j] = T.cast(output[i, j], dtype)
                                else:
                                    for i, j in T.Parallel(row_block_vec2, head_dim):
                                        output_cast[i, j] = T.cast(output[i, j] / denominator[query_slot_vec2, row + i], dtype)
                                T.copy(output_cast, O[batch_index_vec2, head_index_vec2, query_pair_vec2,
                                                      sid, row:row + row_block_vec2, :])

    return main.with_attr("global_symbol", ("chunked_fa_" if chunked_pv else "parallel_fa_") + dtype)
