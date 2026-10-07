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
    row_block divides block_q, and groups divides B*H*query_pairs. This
    factory has no causal mask, GQA, dropout, or sequence tails. Resource
    feasibility is checked by the target compiler, not by these shape rules.
    """
    values = (batch, heads, query_pairs, kv_tiles, groups,
              block_q, block_kv, head_dim, row_block)
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
        raise ValueError("Persistent attention dimensions and groups must be positive integers")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("Persistent attention supports FP16/BF16 operands")
    if any(x % 16 for x in (block_q, block_kv, head_dim, row_block)) or block_q % row_block:
        raise ValueError("Aligned dense tiles and complete UB row subtiles are required")
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
            v_l1 = T.alloc_l1((2, block_kv, head_dim), dtype)
            p_l1 = T.alloc_l1((block_q, block_kv), dtype)
            score_acc = T.alloc_l0c((block_q, block_kv), "float32")
            pv_acc = T.alloc_l0c((block_q, head_dim), "float32")

            scores = T.alloc_shared((row_block, block_kv), "float32")
            probabilities = T.alloc_shared((row_block, block_kv), dtype)
            pv = T.alloc_shared((row_block, head_dim), "float32")
            output = T.alloc_shared((row_block, head_dim), "float32")
            output_cast = T.alloc_shared((row_block, head_dim), dtype)
            block_maximum = T.alloc_shared((row_block,), "float32")
            next_maximum = T.alloc_shared((row_block,), "float32")
            row_sum = T.alloc_shared((row_block,), "float32")
            maximum = T.alloc_shared((2, block_q), "float32")
            denominator = T.alloc_shared((2, block_q), "float32")
            alpha = T.alloc_shared((2, block_q), "float32")

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
            T.annotate_manual_multi_buffer({v_l1: 2, alpha: 2,
                                            Score: 2, P: 2, PV: 2})

            # Each group has one immutable clock through all of its pairs.
            # Local-pair parity owns the query state; flattened-pair parity
            # does not alternate when the group stride is even.
            for e in T.serial(tasks + 2):
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
                            T.copy(V[batch_index, head_index,
                                     kv * block_kv:(kv + 1) * block_kv, :],
                                   v_l1[task_slot, :, :])
                            for owner in T.serial(2):
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
                        for sub in T.serial(block_q // row_block):
                            row = sub * row_block
                            T.copy(PV[group, sid, task_slot_vec2, row:row + row_block, :], pv)
                            if kv_vec2 == 0:
                                T.copy(pv, output)
                            else:
                                T.copy(OState[group, sid, query_slot_vec2, row:row + row_block, :], output)
                                for i, j in T.Parallel(row_block, head_dim):
                                    output[i, j] = output[i, j] * alpha[task_slot_vec2, row + i]
                                for i, j in T.Parallel(row_block, head_dim):
                                    output[i, j] = output[i, j] + pv[i, j]
                            if kv_vec2 + 1 < kv_tiles:
                                T.copy(output, OState[group, sid, query_slot_vec2, row:row + row_block, :])
                            else:
                                for i, j in T.Parallel(row_block, head_dim):
                                    output_cast[i, j] = T.cast(output[i, j] / denominator[query_slot_vec2, row + i], dtype)
                                T.copy(output_cast, O[batch_index_vec2, head_index_vec2, query_pair_vec2,
                                                      sid, row:row + row_block, :])

    return main.with_attr("global_symbol", "parallel_fa_" + dtype)
