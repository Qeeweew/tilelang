"""Native prefix/scalar selection keeps predicate storage and VL loops explicit."""

from unittest.mock import patch

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _program(dtype="float32", width=130, *, uniform=False, statement=False, dynamic=False, predicate=None):
    threshold = tirx.Var("threshold", "int32")
    active = tirx.Var("active", "int32")
    # Physical row capacity is rounded for a later row's 32-byte start.
    block = 32 // (2 if dtype == "float16" else 4)
    capacity = (width + block - 1) // block * block
    a, b, c = [tirx.decl_buffer((capacity,), dtype, name=name, scope="shared.dyn") for name in "abc"]
    j = tirx.Var("column", "int32")
    condition = predicate(j, threshold) if predicate else threshold > 0 if uniform else j < threshold
    value = a[j] * tirx.const(2, dtype)
    if statement:
        body = tirx.IfThenElse(condition, tirx.BufferStore(c, value, [j]), None)
    else:
        body = tirx.BufferStore(c, tirx.Select(condition, value, tirx.const(float("-inf"), dtype)), [j])
    extent = tirx.max(tirx.min(active, width), 0) if dynamic else width
    body = tirx.For(j, 0, extent, tirx.ForKind.PARALLEL, body)
    mod = kernel(body, buffers=[a, b, c], params=[threshold, *([active] if dynamic else [])])
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _native(mod):
    return [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _eval(expr, mod, *, threshold=0, active=0, chunk=None, row=None):
    substitutions = {param: tirx.IntImm(param.dtype, threshold if param.name == "threshold" else active) for param in mod["main"].params}
    for loop in nodes(mod, tirx.For):
        if loop.loop_var.name.startswith("membase_select_chunk_") and chunk is not None:
            substitutions[loop.loop_var] = tirx.IntImm(loop.loop_var.dtype, chunk)
        if loop.loop_var.name == "row" and row is not None:
            substitutions[loop.loop_var] = tirx.IntImm(loop.loop_var.dtype, row)
    return _constant_integer(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, substitutions)))


def _constant_integer(expr):
    """Evaluate constant TIR, including UInt64 shifts the analyzer leaves intact.

    The expected packed words are independently derived from prefix lengths in
    each test. This evaluator follows generic integer operators in emitted IR.
    """
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    if isinstance(expr, tirx.Cast):
        return _constant_integer(expr.value)
    if isinstance(expr, tirx.Select):
        return _constant_integer(expr.true_value if _constant_integer(expr.condition) else expr.false_value)
    binary = {
        tirx.Add: lambda a, b: a + b, tirx.Sub: lambda a, b: a - b,
        tirx.Mul: lambda a, b: a * b, tirx.Min: min, tirx.Max: max,
        tirx.LT: lambda a, b: a < b, tirx.LE: lambda a, b: a <= b,
        tirx.GT: lambda a, b: a > b, tirx.GE: lambda a, b: a >= b,
        tirx.EQ: lambda a, b: a == b, tirx.NE: lambda a, b: a != b,
    }
    for kind, operation in binary.items():
        if isinstance(expr, kind):
            return operation(_constant_integer(expr.a), _constant_integer(expr.b))
    if isinstance(expr, tirx.Call):
        arguments = [_constant_integer(argument) for argument in expr.args]
        if expr.op.name == "tirx.shift_left":
            assert 0 <= arguments[1] < 64, "generated predicate has an undefined 64-bit shift"
            return arguments[0] << arguments[1]
        if expr.op.name == "tirx.bitwise_not":
            return ~arguments[0]
    raise AssertionError(f"Expected constant integer TIR, got {expr}")


@pytest.mark.parametrize("dtype,vl", [("float32", 64), ("float16", 128)])
def test_prefix_select_exposes_packed_predicate_and_bounded_vl_chunks(dtype, vl):
    after = transform.AscendLowerVectorTiles()(_program(dtype))
    select = calls(after, "tl.ascend_membase.select_bits")[0]
    assert select.op.get_attr("TGlobalSymbol") == "tl_ascend_membase_select_bits"
    mask_pointer = select.args[3]
    assert mask_pointer.op.name == "tl.access_ptr"
    assert int(mask_pointer.args[1]) == 2 and int(mask_pointer.args[2]) == 1
    predicate = mask_pointer.args[0].buffer
    assert predicate.dtype == "uint64" and tuple(int(size) for size in predicate.shape) == (4,)
    assert predicate.scope() == "shared.dyn"
    (loop,) = nodes(after, tirx.For)
    assert loop.kind == tirx.ForKind.SERIAL
    assert int(loop.extent) == (130 + vl - 1) // vl
    assert [_eval(select.args[-1], after, chunk=chunk) for chunk in range(int(loop.extent))] == [
        min(vl, 130 - chunk * vl) for chunk in range(int(loop.extent))
    ]
    for chunk in range(int(loop.extent)):
        assert all(_eval(pointer.args[1], after, chunk=chunk) == min(vl, 130 - chunk * vl) for pointer in select.args[:3])
    # Materialization must accept the atomic V helper's CMPMASK clobber without
    # pretending its register setup is an independent Scalar issue task.
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len(calls(materialized, "tl.ascend_membase.select_bits")) == 1
    assert not nodes(after, tirx.IfThenElse)


@pytest.mark.parametrize("dtype,vl", [("float32", 64), ("float16", 128)])
def test_prefix_words_match_every_boundary_without_shift_by_64(dtype, vl):
    after = transform.AscendLowerVectorTiles()(_program(dtype))
    stores = [store for store in nodes(after, tirx.BufferStore) if store.buffer.name.startswith("membase_predicate_")]
    assert [int(store.indices[0]) for store in stores] == [0, 1]
    for threshold in (-1, 0, 1, 63, 64, 65, 127, 128, 129, 130):
        for chunk in range((130 + vl - 1) // vl):
            prefix = min(max(threshold - chunk * vl, 0), vl)
            expected = [((1 << min(prefix, 64)) - 1), ((1 << max(prefix - 64, 0)) - 1) if dtype == "float16" else 0]
            assert [(_eval(store.value, after, threshold=threshold, chunk=chunk) & ((1 << 64) - 1)) for store in stores] == expected


def test_scalar_condition_reuses_an_all_zero_or_all_one_pattern_for_the_whole_count():
    after = transform.AscendLowerVectorTiles()(_program(uniform=True))
    assert not nodes(after, tirx.For)
    (select,) = calls(after, "tl.ascend_membase.select_bits")
    assert int(select.args[-1]) == 130
    stores = nodes(after, tirx.BufferStore)
    for threshold, expected in ((-1, 0), (0, 0), (1, (1 << 64) - 1)):
        assert [(_eval(store.value, after, threshold=threshold) & ((1 << 64) - 1)) for store in stores] == [expected, expected]


def test_lane_predicated_store_explicitly_reads_the_previous_destination():
    after = transform.AscendLowerVectorTiles()(_program(statement=True))
    (select,) = calls(after, "tl.ascend_membase.select_bits")
    destination = select.args[0].args[0].buffer
    previous = select.args[2].args[0].buffer
    assert destination.data.same_as(previous.data)
    assert int(select.args[0].args[2]) == 2 and int(select.args[2].args[2]) == 1
    assert not nodes(after, tirx.IfThenElse)


def test_dynamic_zero_count_preserves_initialized_predicate_and_uniform_tasks():
    after = transform.AscendLowerVectorTiles()(_program(dynamic=True))
    (select,) = calls(after, "tl.ascend_membase.select_bits")
    for chunk in range(3):
        assert _eval(select.args[-1], after, active=0, chunk=chunk) == 0
        assert all(_eval(pointer.args[1], after, active=0, chunk=chunk) == 0 for pointer in select.args[:3])
    assert len([store for store in nodes(after, tirx.BufferStore) if store.buffer.dtype == "uint64"]) == 2
    materialized = transform.MaterializeScheduleUnits()(after)
    assert not nodes(materialized, tirx.IfThenElse)


@pytest.mark.parametrize("causal", [False, True])
def test_fa_tail_and_causal_prefix_intersection_is_one_scalar_cutoff(causal):
    width = 130
    a, c = [tirx.decl_buffer((2, 136), "float32", name=name, scope="shared.dyn") for name in "ac"]
    i, j = tirx.Var("row", "int32"), tirx.Var("column", "int32")
    condition = ((64 + j) < 100) & (tirx.const(not causal, "bool") | ((64 + j) <= (90 + i)))
    expression = tirx.if_then_else(condition, a[i, j] * 0.5, tirx.const(float("-inf"), "float32"))
    body = tirx.For(i, 0, 2, tirx.ForKind.PARALLEL,
        tirx.For(j, 0, width, tirx.ForKind.PARALLEL, tirx.BufferStore(c, expression, [i, j])))
    mod = kernel(body, buffers=[a, c])
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    after = transform.AscendLowerVectorTiles()(mod)
    assert all(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert len(calls(after, "tl.ascend_membase.select_bits")) == 1
    store = next(store for store in nodes(after, tirx.BufferStore) if store.buffer.dtype == "uint64" and int(store.indices[0]) == 0)
    for row in (0, 1):
        prefix = min(36, 27 + row) if causal else 36
        assert (_eval(store.value, after, row=row, chunk=0) & ((1 << 64) - 1)) == (1 << prefix) - 1


def test_non_prefix_predicate_is_a_clear_boundary_diagnostic():
    with pytest.raises(ValueError, match="affine column prefixes"):
        transform.AscendLowerVectorTiles()(_program(predicate=lambda j, threshold: j % 2 == 0))


@pytest.mark.parametrize("reverse", [False, True])
def test_equality_at_proven_lower_boundary_preserves_one_lane_prefix(reverse):
    def predicate(j, _):
        column = tirx.Cast("int64", j) + tirx.IntImm("int64", 64)
        boundary = tirx.IntImm("int64", 64)
        return tirx.EQ(boundary, column) if reverse else tirx.EQ(column, boundary)

    after = transform.AscendLowerVectorTiles()(_program(predicate=predicate))
    assert len(calls(after, "tl.ascend_membase.select_bits")) == 1
    stores = [store for store in nodes(after, tirx.BufferStore) if store.buffer.dtype == "uint64"]
    for chunk in range(3):
        assert [(_eval(store.value, after, chunk=chunk) & ((1 << 64) - 1)) for store in stores] == [int(chunk == 0), 0]


@pytest.mark.parametrize("boundary", [17, 129])
def test_equality_to_an_interior_or_last_lane_is_not_a_prefix(boundary):
    with pytest.raises(ValueError, match="affine column prefixes"):
        transform.AscendLowerVectorTiles()(_program(predicate=lambda j, _: tirx.EQ(j, boundary)))


def test_boundary_equality_keeps_safe_operands_at_signed_extremes():
    a, c = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in "ac"]
    column = tirx.Var("column", "int32")
    offset, boundary = (tirx.Var(name, "int64") for name in ("offset", "boundary"))
    maximum = (1 << 63) - 1
    condition = tirx.EQ(offset + tirx.Cast("int64", column), boundary)
    value = tirx.Select(condition, a[column], tirx.const(float("-inf"), "float32"))
    body = tirx.For(column, 0, 64, tirx.ForKind.PARALLEL, tirx.BufferStore(c, value, [column]))
    # Both original equality operands are representable throughout the domain.
    # Their difference need not fit Int64; lowering cannot introduce it.
    guard = tirx.And(tirx.And(offset >= 0, offset <= maximum - 63), boundary <= 0)
    mod = kernel(tirx.IfThenElse(guard, body, None), buffers=[a, c], params=[offset, boundary])
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    after = transform.AscendLowerVectorTiles()(mod)
    stores = [store for store in nodes(after, tirx.BufferStore) if store.buffer.dtype == "uint64"]
    assert len(calls(after, "tl.ascend_membase.select_bits")) == 1
    for start, bound, expected in ((0, 0, 1), (2, -(1 << 63), 0), (maximum - 63, -1, 0)):
        substitutions = {offset: tirx.IntImm("int64", start), boundary: tirx.IntImm("int64", bound)}
        for loop in nodes(after, tirx.For):
            if loop.loop_var.name.startswith("membase_select_chunk_"):
                substitutions[loop.loop_var] = tirx.IntImm(loop.loop_var.dtype, 0)
        words = [tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(store.value, substitutions)) for store in stores]
        assert [(_constant_integer(word) & ((1 << 64) - 1)) for word in words] == [expected, 0]


def test_bfloat_predicate_selection_requests_an_explicit_supported_precision():
    with pytest.raises(ValueError, match="promote BF16 values explicitly"):
        transform.AscendLowerVectorTiles()(_program("bfloat16"))


@pytest.mark.parametrize("per_core", [False, True], ids=["atomic", "per-core-candidates"])
@pytest.mark.parametrize("ordinary", ["exp", "add"])
def test_cmpmask_orders_selection_only_and_survives_per_core_decode(per_core, ordinary):
    buffers = [tirx.decl_buffer((8,), "float32", name=f"data{i}", scope="shared.dyn") for i in range(9)]
    masks = [tirx.decl_buffer((4,), "uint64", name=f"mask{i}", scope="shared.dyn") for i in range(2)]

    def pointer(buffer, count, effect):
        return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], count, effect])

    operations = [
        tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.select_bits"), [
            pointer(buffers[0], 8, 2), pointer(buffers[1], 8, 1), pointer(buffers[2], 8, 1),
            pointer(masks[0], 2, 1), tirx.const(8, "uint32"),
        ])),
        tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase." + ordinary), [
            pointer(buffers[3], 8, 2), pointer(buffers[4], 8, 1),
            *([pointer(buffers[8], 8, 1)] if ordinary == "add" else []), tirx.const(8, "uint32"),
        ])),
        tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.select_bits"), [
            pointer(buffers[5], 8, 2), pointer(buffers[6], 8, 1), pointer(buffers[7], 8, 1),
            pointer(masks[1], 2, 1), tirx.const(8, "uint32"),
        ])),
    ]
    tasks = []
    for operation in operations:
        if per_core:
            metadata = {"core_mask": tirx.IntImm("int64", 1), "latency": tirx.IntImm("int64", 1), "ii": tirx.IntImm("int64", 1)}
            grouped = tirx.AttrStmt(metadata, "tl.ascend_per_core_task", 1, unit(operation, core=1).body)
            tasks.append(unit(grouped, core=None, stage=-1))
        else:
            tasks.append(unit(operation, core=1, stage=-1, cost=(1, 1)))
    loop = tirx.For(tirx.Var("epoch", "int32"), 0, 3, tirx.ForKind.SERIAL, seq(*tasks), annotations={"num_stages": 2})
    before = kernel(unit(loop, core=None, stage=-1), buffers=[*buffers, *masks])
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    captured = []

    def schedule(*args, **kwargs):
        dependencies = kwargs.get("data_deps", args[4] if len(args) > 4 else None)
        captured.extend(tuple(int(value) for value in dependency) for dependency in dependencies)
        # Three V issues need three distinct phases. This also leaves the
        # CMPMASK wrap edge from the final selection to the first one legal.
        return [0, 1, 2], [], 3

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        transform.AutoSchedule()(before)
    edges = {dependency[:3] for dependency in captured}
    # All memory regions are disjoint. The native helpers still share their
    # internally configured CMPMASK, including after PerCoreTask decoding.
    assert {(0, 2, 0), (2, 0, 1)} <= edges
    # Ordinary exp/add neither reads nor writes CMPMASK; hardware pipe exclusion
    # is a separate solver constraint and must not become a false data edge.
    assert (1, 1, 1) in edges  # the ordinary task's one-version output is reused
    assert not any(source != destination and (source == 1 or destination == 1) for source, destination, _ in edges)
