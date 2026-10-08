"""Bounded private multi-VL rows retain exact effects and V-only results."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _Completion, _prove_binary_protocol
from testing.ascend.vector.test_ascend_multi_vl_reduce import TARGET, PREFIX, _lower, _program, _root_program, _strided


def _integer(expression, environment):
    substituted = tirx.stmt_functor.substitute(
        expression, {var: tirx.const(value, var.dtype) for var, value in environment.items()})
    return int(tvm.arith.Analyzer().simplify(substituted))


def _execute_rows(statement, environment, memory, issued):
    """Interpret native reduction contracts and ordinary typed combine IR."""
    def scalar(expression):
        if isinstance(expression, tirx.BufferLoad):
            return memory[expression.buffer.data][tuple(_integer(index, environment) for index in expression.indices)]
        if isinstance(expression, tirx.Add):
            return np.float32(scalar(expression.a) + scalar(expression.b))
        if isinstance(expression, tirx.Max):
            return np.maximum(scalar(expression.a), scalar(expression.b))
        raise AssertionError(expression)

    def access(argument):
        base = argument.args[0]
        array = memory[base.buffer.data]
        indices = tuple(_integer(index, environment) for index in base.indices)
        if argument.op.name == "tl.region":
            extents = tuple(_integer(extent, environment) for extent in argument.args[2:])
            assert all(0 <= index and index + extent <= capacity
                       for index, extent, capacity in zip(indices, extents, array.shape))
            return array[tuple(slice(index, index + extent) for index, extent in zip(indices, extents))]
        assert argument.op.name == "tl.access_ptr"
        offset = np.ravel_multi_index(indices, array.shape)
        extent = _integer(argument.args[1], environment)
        assert 0 <= offset and offset + extent <= array.size
        return array.reshape(-1)[offset : offset + extent]

    if isinstance(statement, tirx.SeqStmt):
        for child in statement.seq:
            _execute_rows(child, environment, memory, issued)
    elif isinstance(statement, tirx.IfThenElse):
        assert statement.else_case is None
        if _integer(statement.condition, environment):
            _execute_rows(statement.then_case, environment, memory, issued)
    elif isinstance(statement, tirx.For):
        start = _integer(statement.min, environment)
        for value in range(start, start + _integer(statement.extent, environment)):
            _execute_rows(statement.body, {**environment, statement.loop_var: value}, memory, issued)
    elif isinstance(statement, tirx.BufferStore):
        indices = tuple(_integer(index, environment) for index in statement.indices)
        memory[statement.buffer.data][indices] = scalar(statement.value)
    elif isinstance(statement, tirx.Evaluate):
        call = statement.value
        name = call.op.name
        output, source = (access(argument) for argument in call.args[:2])
        reduce = np.max if "max" in name else np.sum
        if name.endswith("_rows_f32"):
            repeats, _, lanes = (_integer(argument, environment) for argument in call.args[2:])
            assert 1 <= repeats <= 255 and source.shape == (repeats, lanes)
            result = reduce(source, axis=1)
        else:
            count = _integer(call.args[2], environment)
            lanes = 8 if "datablock_reduce" in name else 64
            assert count > 0 and count == source.size and count % lanes == 0
            result = reduce(source.reshape(-1, lanes), axis=1)
        assert output.size == result.size
        output[...] = result.reshape(output.shape)
        issued.append(name)
    else:
        raise AssertionError(statement)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("columns", [128, 256, 512, 1024])
@pytest.mark.parametrize("active", [0, 1, 4, 8])
def test_dynamic_native_math_exact_accesses_and_static_scratch(kind, columns, active):
    rows = tirx.Var("owned_rows", "int32")
    before, source, destination = _program(
        kind=kind, rows=8, columns=columns, row_extent=rows,
        source_row=2, output_row=8)
    after = _lower(before)
    assert _strided(after, kind) and not nodes(after, tirx.Bind)
    assert not calls(after, "tl.ascend_membase.datablock_reduce_" + kind)
    root = after["main"].body.block
    assert all(isinstance(extent, tirx.IntImm) for buffer in root.alloc_buffers for extent in buffer.shape)
    memory = {buffer.data: np.full(tuple(int(extent) for extent in buffer.shape), np.nan, np.float32)
              for buffer in root.alloc_buffers}
    values = ((np.arange(source.shape[0].value * columns) * 37 % 83 - 41) / 8).astype(np.float32).reshape(-1, columns)
    memory[source.data][...] = values
    memory[destination.data][...] = 519
    issued = []
    _execute_rows(root.body.body, {rows: active}, memory, issued)
    expected = np.full(int(destination.shape[0]), 519, np.float32)
    if active:
        selected = values[2 : 2 + active]
        expected[8 : 8 + active] = selected.max(axis=1) if kind == "max" else selected.sum(axis=1)
    np.testing.assert_array_equal(memory[destination.data], expected)
    np.testing.assert_array_equal(memory[source.data], values)
    assert bool(issued) == bool(active)
    # Exercise the separate native region boundary with the real dynamic
    # extent: all pointer spans remain exact, including guarded empty owners.
    with tvm.target.Target(TARGET):
        pointers = tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(_lower(before, vector=True)))
    for call in _strided(pointers, kind):
        assert call.args[2].dtype == "uint8"
        assert all(argument.op.name == "tirx.tvm_access_ptr" for argument in call.args[:2])


@pytest.mark.parametrize("options", [
    {"rows": 256}, {"columns": 130, "pitch": 136}, {"pitch": 264},
    {"output_row": 1}, {"clear": False},
])
def test_unproved_or_incompatible_dynamic_rows_keep_existing_fallback(options):
    rows = tirx.Var("owned_rows", "int32")
    before, _, _ = _program(row_extent=rows, **{"columns": 256, **options})
    after = _lower(before)
    assert not _strided(after)
    assert calls(after, PREFIX + "sum")


def test_unbounded_dynamic_row_capacity_keeps_existing_plan():
    capacity = tirx.Var("capacity", "int32")
    extent = tirx.max(capacity, 0)
    source = tirx.decl_buffer((extent, 256), "float32", "source", scope="shared.dyn")
    destination = tirx.decl_buffer((extent,), "float32", "destination", scope="shared.dyn")
    reduction = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(source, 1), region(destination, 2), tirx.StringImm("max"), 1, tirx.const(True, "bool")])
    before = kernel(tirx.Evaluate(reduction), buffers=[source, destination], params=[capacity])
    before["main"] = before["main"].with_attr("target", tvm.target.Target(TARGET))
    after = _lower(before)
    assert not _strided(after, "max")
    assert calls(after, PREFIX + "max_only_value")


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_dynamic_uint8_upper_endpoint_uses_static_maximum_capacity(index_dtype):
    rows = tirx.Var("owned_rows", index_dtype)
    before, _, _ = _program(rows=255, columns=128, row_extent=rows, index_dtype=index_dtype)
    after = _lower(before)
    native = _strided(after)
    assert len(native) == 2
    partials = native[0].args[0].args[0].buffer
    assert [int(extent) for extent in partials.shape] == [2, 256]
    assert all(_integer(call.args[2], {rows: 255}) == 255 for call in native)
    with tvm.target.Target(TARGET):
        transform.AscendLowerTileOp()(_lower(before, vector=True))


def test_dynamic_candidate_reserves_maximum_scratch_and_rejects_opaque_task():
    rows = tirx.Var("owned_rows", "int32")
    before, _, _ = _program(rows=8, columns=256, row_extent=rows, task=True)
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)
    before, _, _ = _program(rows=8, columns=256, row_extent=rows)
    # One 32B-aligned row vector per chunk and combine is charged using Rmax,
    # even when the current dynamic invocation happens to own just one row.
    owner_bytes = 8 * 256 * 4 + 32
    filler = tirx.decl_buffer(((196608 - owner_bytes - 192) // 4,),
                             "float32", "filler", scope="shared.dyn")
    assert not _strided(_lower(_root_program(before, extra_buffers=[filler])))


@pytest.mark.parametrize("bad", ["unguarded-zero", "overflow", "mismatched-repeat"])
def test_late_boundary_rejects_dynamic_narrowing_without_an_exact_positive_extent(bad):
    rows = tirx.Var("owned_rows", "int32")
    before, source, _ = _program(rows=8, columns=256, row_extent=rows)
    native = _strided(_lower(before))[0]
    args = list(native.args)
    maximum = 8
    guard = rows > 0
    if bad == "unguarded-zero":
        guard = tirx.const(True, "bool")
    elif bad == "overflow":
        maximum = 256
        source = tirx.decl_buffer((256, 256), "float32", "large_source", scope="shared.dyn")
        partials = tirx.decl_buffer((4, 256), "float32", "large_partials", scope="shared.dyn")
        args[0] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), [partials[0, 0], 2, 1, rows])
        args[1] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), [source[0, 0], 1, rows, 64])
    else:
        args[2] = tirx.Cast("uint8", rows + 1)
    partials = args[0].args[0].buffer
    body = tirx.For(rows, 0, maximum + 1, tirx.ForKind.SERIAL,
                    tirx.IfThenElse(guard, tirx.Evaluate(tirx.Call("void", native.op, args)), None))
    mod = kernel(body, buffers=[source, partials])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    with tvm.target.Target(TARGET), pytest.raises(tvm.error.InternalError, match="proved active rows"):
        transform.AscendLowerTileOp()(mod)


@pytest.fixture(scope="module", params=["generic", "pipeline"])
def packed_decode(request):
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    spec = importlib.util.spec_from_file_location("_dynamic_multi_vl_attention", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    program = example.flash_attention_fwd(
        1, 4, 1, 1, 4096, 64, dtype="bfloat16", variant=request.param,
        tiling=example.FwdTiling(16, 256, 4))
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.LowerScheduledTIR":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(program, target=TARGET, enable_host_codegen=False, enable_device_compile=False)
    assert "membase_reduce_result" not in artifact.kernel_source
    assert "asc_repeat_reduce_max_only_value(" in artifact.kernel_source
    return snapshots["tl.LowerScheduledTIR"]


@pytest.mark.parametrize("lane,active", [(0, 4), (1, 0)])
def test_packed_decode_dynamic_rows_have_finite_scalar_vector_flags_and_no_empty_native_issue(packed_decode, lane, active):
    (vector,) = [block for block in nodes(packed_decode, tirx.SBlock) if block.name_hint == "VECTOR"]
    coordinates = {var: tirx.const(0, var.dtype) for var in nodes(vector.body, tirx.Var) if var.name == "bx"}
    body = tirx.stmt_functor.substitute(vector.body, coordinates)

    def omit_scalar_data(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.scope() != "local.var":
            return tirx.Evaluate(0)
        if isinstance(node, tirx.Bind) and node.var.dtype.startswith("float"):
            return tirx.Evaluate(0)

    body = tirx.stmt_functor.ir_transform(body, None, omit_scalar_data)
    trace, native_repeats = [], []
    for call, env, _ in _trace(packed_decode["main"].with_body(body), core=1, lane=lane):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            index = _value(call.args[1], env)
            # Capacity is checked for every actual local event direction.
            assert 0 <= index < 8
            if event in ("V_S", "S_V"):
                trace.append(_Completion(source if setting else destination, "set" if setting else "wait", (event, index)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Completion(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name == PREFIX + "max_rows_f32":
            native_repeats.append(_value(call.args[2], env))
    assert native_repeats == ([active] * (16 * 4) if active else [])
    # This projection proves the scalar/vector channels affected by replacing
    # row-result scalar extraction. It is not a full mixed-kernel completion
    # graph: P's V_MTE3 reuse can close through two AIV arrivals, Cube P/PV,
    # then an AIV cross-core wait. On2201 that wait blocks Scalar admission;
    # omitting those cross-core nodes falsely removes the reverse path.
    # The separate native owner/collective tests retain their full controls.
    assert {node.flag[0] for node in trace if node.flag} == {"V_S", "S_V"}
    _prove_binary_protocol(trace)
    # Keep a balanced-set/wait negative control. Without the real S->V
    # reverse bridge, repeated V->S notifications must fail even though
    # every individual notification still has a matching wait.
    without_reverse = [node for node in trace if not node.flag or node.flag[0] != "S_V"]
    with pytest.raises(AssertionError, match="may be set again before its prior wait completes"):
        _prove_binary_protocol(without_reverse)
