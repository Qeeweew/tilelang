"""Prefix selection keeps a reusable binary protocol under dynamic row tails."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tvm import tirx

from testing.ascend._ir import nodes
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _Completion, _prove_binary_protocol


@pytest.fixture(scope="module", params=[64, 130], ids=["one-vl", "multiple-vl"])
def selected_tail(request):
    columns = request.param
    pitch = (columns + 7) // 8 * 8

    @T.prim_func
    def main(A: T.Tensor((32, pitch), "float32"), C: T.Tensor((32, pitch), "float32"), active_rows: T.int32, tail: T.int32):
        with T.Kernel(1):
            source = T.alloc_shared((32, pitch), "float32")
            output = T.alloc_shared((32, pitch), "float32")
            T.copy(A, source)
            T.fill(output, 0.0)
            for epoch in T.serial(3):
                for i, j in T.Parallel(T.max(0, T.min(32, active_rows)), columns):
                    output[i, j] = T.if_then_else(j < tail - epoch * columns, source[i, j] * 0.5, -T.infinity("float32"))
            T.copy(output, C)

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendLowerVectorTiles", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(
            main.with_attr("global_symbol", "main"),
            target={"kind": "ascend", "arch": "dav-2201"},
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    assert result.kernel_source and set(snapshots) == {"tl.AscendLowerVectorTiles", "tl.LowerScheduledTIR"}
    return columns, snapshots


@pytest.mark.parametrize("rows,tail", [(0, 0), (1, 65), (32, 130)])
def test_dynamic_row_prefix_has_complete_binary_flag_lifetimes(selected_tail, rows, tail):
    _, snapshots = selected_tail
    scheduled = snapshots["tl.LowerScheduledTIR"]
    function = scheduled["main"]
    replacements = {
        var: tirx.const(rows if var.name == "active_rows" else tail, var.dtype)
        for var in function.params
        if var.name in ("active_rows", "tail")
    }
    body = tirx.stmt_functor.substitute(function.body, replacements)

    # Predicate values are not part of the flag-lifetime oracle. Preserve all
    # actual flag calls and guards; omit only the packed UInt64 mask writes.
    def omit_predicate_store(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.dtype == "uint64":
            return tirx.Evaluate(0)

    body = tirx.stmt_functor.ir_transform(body, None, omit_predicate_store)
    trace = []
    for call, env, _ in _trace(function.with_body(body), core=1, lane=0):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            trace.append(_Completion(source if setting else destination, "set" if setting else "wait", (event, _value(call.args[1], env))))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Completion(call.args[0].value.removeprefix("PIPE_"), "barrier"))
    if rows:
        assert any(event.flag and event.flag[0] == "S_V" for event in trace)
    _prove_binary_protocol(trace)


def test_one_vl_selection_has_no_artificial_single_iteration_owner(selected_tail):
    columns, snapshots = selected_tail
    chunks = [
        loop for loop in nodes(snapshots["tl.AscendLowerVectorTiles"], tirx.For) if loop.loop_var.name.startswith("membase_select_chunk_")
    ]
    assert len(chunks) == (columns > 64)
    assert all(int(loop.extent) > 1 for loop in chunks)
