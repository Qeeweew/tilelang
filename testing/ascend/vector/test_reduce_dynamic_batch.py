"""Bounded dynamic full-VL batches preserve tails, typed results and flags."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import tilelang
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import kernel, nodes, region
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _Completion, _prove_binary_protocol
from testing.ascend.vector.test_reduce_batched_rows import _lower, _program, _reduce_calls


def _integer(expression, variable, value):
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expression, {variable: tirx.const(value, variable.dtype)})))


@pytest.mark.parametrize("dtype,kind", [("float32", "max"), ("float16", "max"), ("float32", "sum")])
@pytest.mark.parametrize("active", [0, 1, 5])
def test_native_counter_batch_matches_exact_rows_and_leaves_output_guards(dtype, kind, active):
    rows = tirx.Var("active_rows", "int32")
    before, source, destination = _program(dtype=dtype, kind=kind, rows=5, row_extent=rows, source_row=2, output_row=2)
    after = _lower(before)
    (call,) = _reduce_calls(after)
    (guard,) = nodes(after, tirx.IfThenElse)
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.BufferStore)
    assert len(after["main"].body.block.alloc_buffers) == 2
    vl = 128 if dtype == "float16" else 64
    # Binary fractions keep this contract check independent of reduction-tree
    # rounding. FP16 max selects an existing representable input value.
    values = ((np.arange(7 * vl) * 37 % 83 - 41) / 8).astype(dtype).reshape(7, vl)
    output = np.full(7, 519, dtype=dtype)
    expected = output.copy()
    if active:
        selected = values[2 : 2 + active]
        expected[2 : 2 + active] = selected.max(axis=1) if kind == "max" else selected.sum(axis=1, dtype=np.float32)
    # Model the CANN count API's VL-sized partial outputs using the emitted
    # base and exact access extent, rather than an assumed logical row count.
    if _integer(guard.condition, rows, active):
        dst, src = call.args[:2]
        assert dst.args[0].buffer.same_as(destination) and src.args[0].buffer.same_as(source)
        count = _integer(call.args[-1], rows, active)
        read = _integer(src.args[1], rows, active)
        written = _integer(dst.args[1], rows, active)
        assert count == read == active * vl and written == active
        assert [int(index) for index in src.args[0].indices] == [2, 0]
        assert [int(index) for index in dst.args[0].indices] == [2]
        assert [int(src.args[2]), int(dst.args[2])] == [1, 2]
        lane_groups = values.reshape(-1)[2 * vl : 2 * vl + read].reshape(-1, vl)
        output[2 : 2 + written] = lane_groups.max(axis=1) if kind == "max" else lane_groups.sum(axis=1, dtype=np.float32)
    else:
        assert active == 0
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("dtype,kind", [("float32", "max"), ("float16", "max"), ("float32", "sum")])
def test_dynamic_clear_false_retains_the_typed_previous_output_combine(dtype, kind):
    rows = tirx.Var("active_rows", "int32")
    before, _, destination = _program(dtype=dtype, kind=kind, rows=5, row_extent=rows, clear=False)
    after = _lower(before)
    assert all(int(call.args[-1]) == (128 if dtype == "float16" else 64) for call in _reduce_calls(after))
    previous = [
        bind for bind in nodes(after, tirx.Bind) if isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.same_as(destination)
    ]
    assert previous and all(bind.var.dtype == dtype for bind in previous)
    assert any(loop.extent.same_as(rows) for loop in nodes(after, tirx.For))


@pytest.mark.parametrize(
    "options", [{"source_pitch": 72}, {"output_row": 1}, {"width": 17, "source_pitch": 24}, {"retain_axis": True, "output_pitch": 4}]
)
def test_dynamic_partial_or_noncompact_rows_keep_the_exact_fallback(options):
    rows = tirx.Var("active_rows", "int32")
    before, _, _ = _program(rows=5, row_extent=rows, **options)
    after = _lower(before)
    assert [int(call.args[-1]) for call in _reduce_calls(after)] == [options.get("width", 64)]
    assert any(loop.extent.same_as(rows) for loop in nodes(after, tirx.For))


def test_bounded_dynamic_batch_cannot_hide_guarded_completion_in_an_opaque_task():
    rows = tirx.Var("active_rows", "int32")
    before, _, _ = _program(rows=5, row_extent=rows)

    def wrap(statement):
        if (
            isinstance(statement, tirx.Evaluate)
            and isinstance(statement.value, tirx.Call)
            and statement.value.op.name == "tl.tileop.reduce"
        ):
            return tirx.AttrStmt({}, "tl.ascend_task", 1, statement)

    before["main"] = before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, wrap))
    with pytest.raises(ValueError, match="possibly empty batched reduction needs guarded completion"):
        _lower(before)


def test_dynamic_batch_region_bounds_are_checked_before_selection():
    rows = tirx.Var("active_rows", "int32")
    before, _, _ = _program(rows=5, row_extent=rows)
    outer = before["main"].body.block.body
    oversized = tirx.stmt_functor.substitute(outer.body, {rows: rows + 1})
    root = before["main"].body.block
    root = tirx.SBlock([], [], [], root.name_hint, tirx.For(rows, 0, 6, tirx.ForKind.SERIAL, oversized), alloc_buffers=root.alloc_buffers)
    before["main"] = before["main"].with_body(tirx.SBlockRealize([], True, root))
    with pytest.raises(ValueError, match="regions must lie within"):
        _lower(before)


def test_unbounded_row_capacity_does_not_select_an_unproved_native_counter():
    variable = tirx.Var("capacity", "int32")
    extent = tirx.max(variable, 0)
    source = tirx.decl_buffer((extent, 64), "float32", "source", scope="shared.dyn")
    destination = tirx.decl_buffer((extent,), "float32", "destination", scope="shared.dyn")
    reduction = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.reduce"),
        [region(source, 1), region(destination, 2), tirx.StringImm("max"), 1, tirx.const(True, "bool")],
    )
    before = kernel(tirx.Evaluate(reduction), buffers=[source, destination], params=[variable])
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    after = _lower(before)
    assert [int(call.args[-1]) for call in _reduce_calls(after)] == [64]
    assert any(tvm.arith.Analyzer().can_prove_equal(loop.extent, extent) for loop in nodes(after, tirx.For))


@pytest.fixture(scope="module")
def pipelined_tail():
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    spec = importlib.util.spec_from_file_location("_dynamic_reduce_attention", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    program = example.flash_attention_fwd(2, 4, 2, 65, 130, 64, dtype="float16", variant="pipeline", tiling=example.FwdTiling(64, 64, 4))
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.LowerScheduledTIR":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(
            program, target={"kind": "ascend", "arch": "dav-2201"}, enable_host_codegen=False, enable_device_compile=False
        )
    assert result.kernel_source and "membase_reduce_result" not in result.kernel_source and "membase_reduce_tmp" not in result.kernel_source
    return snapshots["tl.LowerScheduledTIR"]


@pytest.mark.parametrize("block,lane,active_rows", [(0, 0, 32), (0, 1, 32), (1, 0, 1), (1, 1, 0)])
def test_dynamic_native_reduce_passes_full_pipeline_and_binary_flag_lifetimes(pipelined_tail, block, lane, active_rows):
    (vector,) = [node for node in nodes(pipelined_tail, tirx.SBlock) if node.name_hint == "VECTOR"]
    coordinates = {var: tirx.const(block, var.dtype) for var in nodes(vector.body, tirx.Var) if var.name == "bx"}
    body = tirx.stmt_functor.substitute(vector.body, coordinates)
    assert all(not var.dtype.startswith("float") for branch in nodes(body, tirx.IfThenElse) for var in nodes(branch.condition, tirx.Var))

    def omit_scalar_data(node):
        # A flag-only oracle retains real counters, flags and guards. Packed
        # masks and floating broadcast values cannot control this fixture.
        if isinstance(node, tirx.BufferStore) and node.buffer.scope() != "local.var":
            return tirx.Evaluate(0)
        if isinstance(node, tirx.Bind) and node.var.dtype.startswith("float"):
            return tirx.Evaluate(0)

    body = tirx.stmt_functor.ir_transform(body, None, omit_scalar_data)
    trace, reductions = [], []
    function = pipelined_tail["main"].with_body(body)
    for call, env, _ in _trace(function, core=1, lane=lane):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            index = _value(call.args[1], env)
            assert 0 <= index < 8
            trace.append(_Completion(source if setting else destination, "set" if setting else "wait", (event, index)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Completion(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name == "tl.ascend_membase.repeat_reduce_max_only_value":
            reductions.append(_value(call.args[-1], env))
    assert reductions == ([active_rows * 64] * 3 if active_rows else [])
    _prove_binary_protocol(trace)
