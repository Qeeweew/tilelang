"""Independent prepared AIV clocks must not change an AIC byte-credit pool.

The ordinary frontend fixture combines a fixed Q cache, a counter-based B
ring, shifted PV operands and guarded AIV temporaries. Physical completion
checks consume real flags; no unit readiness becomes whole-M completion.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import allocated_buffer, nodes
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower, _root
from testing.ascend.auto_schedule.test_l0_operand_slot_pool import _check_physical_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace, _value


_COUNTERS = "tl.multi_buffer_counter_map"


def _program(*, active=3, inner_vector_loop=False):
    @T.prim_func
    def main(Q: T.Tensor((3, 2, 128, 64), "float16"),
             K: T.Tensor((3, 1024, 64), "float16"),
             P: T.Tensor((3, 2, 128, 1024), "float16"),
             V: T.Tensor((3, 1024, 64), "float16"),
             Score: T.Tensor((3, 2, 128, 1024), "float32"),
             Result: T.Tensor((3, 2, 128, 64), "float32"),
             VectorInput: T.Tensor((active + 2, 2, 128), "float32"),
             VectorOutput: T.Tensor((active + 2, 2, 128), "float32")):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((128, 64), "float16")
            k_l1 = T.alloc_l1((1024, 64), "float16")
            p_l1 = T.alloc_l1((64, 1024), "float16")
            v_l1 = T.alloc_l1((1024, 64), "float16")
            aq = T.alloc_l0a((128, 64), "float16")
            bq = T.alloc_l0b((256, 64), "float16")
            ap = T.alloc_l0a((64, 256), "float16")
            bp = T.alloc_l0b((256, 64), "float16")
            cq = T.alloc_l0c((128, 256), "float32")
            cp = T.alloc_l0c((64, 64), "float32")
            vector_in = T.alloc_shared((2, 128), "float32")
            vector_out = T.alloc_shared((2, 128), "float32")
            T.annotate_buffer_versions({q_l1: 1, k_l1: 1, p_l1: 1, v_l1: 1,
                                        aq: 1, bq: 2, cq: 1, cp: 1})
            for epoch in T.serial(active + 2):
                with T.Stage(0):
                    if epoch < active:
                        T.copy(K[epoch, :, :], k_l1)
                        for qk_owner in T.serial(2):
                            T.copy(Q[epoch, qk_owner, :, :], q_l1)
                            T.copy(q_l1, aq)
                            for qk_n in T.serial(4):
                                T.copy(k_l1[qk_n * 256:(qk_n + 1) * 256, :], bq)
                                T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
                                T.copy(cq, Score[epoch, qk_owner, :, qk_n * 256:(qk_n + 1) * 256])
                with T.Stage(0):
                    if 1 <= epoch and epoch < active + 1:
                        T.copy(V[epoch - 1, :, :], v_l1)
                        for pv_owner in T.serial(2):
                            for m_part in T.serial(2):
                                T.copy(P[epoch - 1, pv_owner, m_part * 64:(m_part + 1) * 64, :], p_l1)
                                for pv_k in T.Pipelined(4, max_stages=2):
                                    with T.Task(latency=2, ii=1):
                                        T.copy(p_l1[:, pv_k * 256:(pv_k + 1) * 256], ap)
                                    with T.Task(latency=2, ii=1):
                                        T.copy(v_l1[pv_k * 256:(pv_k + 1) * 256, :], bp)
                                    with T.Task(latency=3, ii=1):
                                        T.gemm(ap, bp, cp, transpose_B=False, clear_accum=pv_k == 0)
                                T.copy(cp, Result[epoch - 1, pv_owner, m_part * 64:(m_part + 1) * 64, :])
                with T.Stage(0):
                    if epoch >= 1:
                        if inner_vector_loop:
                            # Two real iterations give this independent
                            # counter-owned loop a local reuse proof. The
                            # existing singleton limitation is not this test.
                            for vector_sub in T.serial(2):
                                T.copy(VectorInput[epoch, :, :], vector_in)
                                for row, col in T.Parallel(2, 128):
                                    vector_out[row, col] = vector_in[row, col] + 1.0
                                T.copy(vector_out, VectorOutput[epoch, :, :])
                        else:
                            T.copy(VectorInput[epoch, :, :], vector_in)
                            for row, col in T.Parallel(2, 128):
                                vector_out[row, col] = vector_in[row, col] + 1.0
                            T.copy(vector_out, VectorOutput[epoch, :, :])
    return main


def _epoch(mod):
    matches = [loop for loop in nodes(mod, tirx.For) if loop.loop_var.name == "epoch"]
    assert len(matches) == 1
    return matches[0]


def _aic_events(mod):
    result = []
    for call, env, loops in _trace(_cube_only(mod), core=2, lane=0):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            coords = tuple((var.name, value) for var, value, _ in loops)
            result.append((name, call.args[0].value, _value(call.args[1], env), coords))
    return result


def _cube_only(mod):
    # LowerScheduledTIR has stripped task markers and split concrete cores
    # into CUBE/VECTOR SBlocks. Preserve the complete actual AIC stream, root
    # bindings and thread context; never feed the AIV block to a Cube oracle.
    blocks = [block for block in nodes(mod, tirx.SBlock)
              if block.name_hint in ("CUBE", "VECTOR")]
    assert sorted(block.name_hint for block in blocks) == ["CUBE", "VECTOR"]

    def mutate(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "VECTOR":
            return tirx.Evaluate(0)

    function = mod["main"]
    body = tirx.stmt_functor.ir_transform(function.body, None, mutate)
    return tvm.IRModule({"main": function.with_body(body)})


def test_aiv_owner_counter_metadata_preserves_aic_physical_completions():
    _, baseline = _lower(_program(inner_vector_loop=True))
    _, direct = _lower(_program())
    prepared = direct["tl.PrepareMultiBuffer"]
    counter_map = _epoch(prepared).annotations[_COUNTERS]
    vector_buffers = [allocated_buffer(prepared, name) for name in ("vector_in", "vector_out")]
    assert all(buffer.data in counter_map for buffer in vector_buffers)
    assert all(buffer.scope() in ("shared", "shared.dyn") for buffer in vector_buffers)
    assert not _epoch(baseline["tl.PrepareMultiBuffer"]).annotations.get(_COUNTERS)
    for snapshots in (baseline, direct):
        _check_physical_completions(_cube_only(snapshots["tl.LowerScheduledTIR"]), 3)
        aliases = _root(snapshots["tl.InsertSync"]).annotations["tl.buffer_alias_map"]
        for first, second in (("aq", "ap"), ("bq", "bp")):
            a, b = (allocated_buffer(snapshots["tl.PrepareMultiBuffer"], name).data
                    for name in (first, second))
            assert b in aliases[a] and a in aliases[b]
    assert _aic_events(baseline["tl.LowerScheduledTIR"]) == _aic_events(direct["tl.LowerScheduledTIR"])


def _insert_modified(prepared, key):
    """An unverified outer annotation must retain ordinary completion flags."""
    owner = _epoch(prepared)
    counters = dict(owner.annotations[_COUNTERS])
    counter = next(iter(counters.values()))
    counters[key] = counter

    def mutate(node):
        if isinstance(node, tirx.For) and node.same_as(owner):
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                            annotations={**dict(node.annotations), _COUNTERS: counters},
                            step=node.step)

    function = prepared["main"]
    body = tirx.stmt_functor.ir_transform(function.body, None, mutate)
    changed = tvm.IRModule({"main": function.with_body(body)})
    with function.attrs["target"]:
        synced = transform.InsertSync()(changed)
        materialized = transform.MaterializeMultiBuffer()(synced)
        return transform.LowerScheduledTIR()(materialized)


@pytest.mark.parametrize("metadata_key", ["unallocated", "l0-bank"])
def test_unverified_or_relevant_epoch_metadata_keeps_ordinary_operand_handoff(metadata_key):
    _, snapshots = _lower(_program(active=1))
    prepared = snapshots["tl.PrepareMultiBuffer"]
    key = (tirx.decl_buffer((1,), "float32", name="unallocated", scope="shared.dyn").data
           if metadata_key == "unallocated" else allocated_buffer(prepared, "bq").data)
    after = _insert_modified(prepared, key)
    _check_physical_completions(_cube_only(after), 1)
    original_ids = {item[2] for item in _aic_events(snapshots["tl.LowerScheduledTIR"])
                    if item[1] == "M_MTE1"}
    fallback_ids = {item[2] for item in _aic_events(after) if item[1] == "M_MTE1"}
    assert len(fallback_ids) > len(original_ids), "Unverified metadata must not receive physical credits"
