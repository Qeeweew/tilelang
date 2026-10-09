"""Logical task generations and grouped persistent state have different clocks.

This finite copy recurrence checks payloads and completion of every covered UB
read/write. It is not an arithmetic or whole-device backend oracle.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import schedule
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _Completion, _prove_binary_protocol


def _program(pairs, kvs, *, wrong_owner=False, mode="complete"):
    @T.prim_func
    def main(A: T.Tensor((2 * pairs * kvs, 64), "float32"),
             History: T.Tensor((2 * pairs * kvs, 64), "float32"),
             O: T.Tensor((2 * pairs, 64), "float32")):
        with T.Kernel(2) as group:
            data = T.alloc_shared((64,), "float32")
            state = T.alloc_shared((2, 64), "float32")
            T.annotate_buffer_versions({state: 1})
            for t in T.serial(pairs * kvs, annotations={
                    "enable_offset": True, "tl.ascend_preserve_issue_order": True}):
                with T.Stage(0):
                    T.copy(A[(group + (t // kvs) * 2) * kvs + t % kvs, :], data)
                with T.Stage(1):
                    query = group + (t // kvs) * 2
                    slot = query % 2 if wrong_owner else (t // kvs) % 2
                    if t % kvs != 0:
                        T.copy(state[slot, :], History[query * kvs + t % kvs, :])
                    T.copy(data, state[slot, :])
                with T.Stage(2):
                    query_out = group + (t // kvs) * 2
                    slot_out = query_out % 2 if wrong_owner else (t // kvs) % 2
                    if t % kvs == kvs - 1:
                        T.copy(state[slot_out, :], O[query_out, :])
    if mode != "complete":
        def post(node):
            if isinstance(node, tirx.For) and node.loop_var.name == "t":
                annotations = dict(node.annotations)
                annotations.pop("tl.ascend_preserve_issue_order")
                if mode == "auto":
                    annotations.update({"tl.ascend_auto_pipeline": True, "num_stages": 4})
                return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                                annotations=annotations, step=node.step)
            if mode == "auto" and isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_stage":
                return node.body
        main = main.with_body(tirx.stmt_functor.ir_transform(main.body, None, post))
    return main.with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))


def _check_payload_and_completion(after, pairs, kvs):
    trace, accesses, memory, outputs = [], [], {}, {}
    for call, env, _ in _trace(after, core=1, lane=0):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            trace.append(_Completion(source if setting else destination,
                                     "set" if setting else "wait",
                                     (event, _value(call.args[1], env))))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Completion(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        else:
            assert name == "tl.tileop.ascend_copy", name
            src, si, _ = _region(call.args[0], env)
            dst, di, _ = _region(call.args[1], env)
            pipe = "MTE2" if src.scope() == "global" else (
                "MTE3" if dst.scope() == "global" else "V")
            index = len(trace)
            trace.append(_Completion(pipe, "copy"))
            source_cell, destination_cell = (src.data, *si), (dst.data, *di)
            if src.name == "A":
                value = si[0]  # Distinct generated payload for every KV task.
            else:
                assert source_cell in memory, "Read of uninitialized persistent state"
                value = memory[source_cell]
                accesses.append((index, source_cell, False))
            if dst.scope() == "global":
                expected = di[0] - 1 if dst.name == "History" else di[0] * kvs + kvs - 1
                assert value == expected, (dst.name, di, value, expected)
                assert (dst.name, di[0]) not in outputs
                outputs[dst.name, di[0]] = value
            else:
                memory[destination_cell] = value
                accesses.append((index, destination_cell, True))
    expected = {("O", pair * 2) for pair in range(pairs)} | {
        ("History", pair * 2 * kvs + kv) for pair in range(pairs) for kv in range(1, kvs)}
    assert set(outputs) == expected
    precedes = _prove_binary_protocol(trace)
    writers, readers = {}, {}
    for index, cell, writing in accesses:
        if cell in writers:
            assert precedes(writers[cell], index), "Missing data readiness or write completion"
        if writing:
            for reader in readers.pop(cell, []):
                assert precedes(reader, index), "Slot reused before all old readers complete"
            writers[cell] = index
        else:
            assert cell in writers
            readers.setdefault(cell, []).append(index)


@pytest.mark.parametrize("pairs,kvs", [(1, 1), (2, 2), (3, 3), (5, 1)])
@pytest.mark.parametrize("mode", ["complete", "stage-only", "auto"])
def test_delayed_grouped_state_payload_and_completion(monkeypatch, pairs, kvs, mode):
    if mode == "complete":
        monkeypatch.setattr(z3_scheduler, "Z3_AVAILABLE", False)
    after = schedule(_program(pairs, kvs, mode=mode), through="LowerScheduledTIR")
    _check_payload_and_completion(after, pairs, kvs)


def test_global_group_stride_is_not_a_query_state_clock():
    # An even grid stride leaves flattened-query parity constant. A new query
    # would initialize the same state before the delayed old-query read.
    with pytest.raises(Exception, match="source-order-data-dependency"):
        schedule(_program(3, 1, wrong_owner=True), through="AutoSchedule")
