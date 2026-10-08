"""Exact-overlap expression destinations, including recurring control flow.

These host regressions check actual native read/write regions and evaluate the
small count-mode programs. Device instruction acceptance is a separate gate.
"""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq


def _target(body, buffers, params=(), annotations=None):
    before = kernel(body, buffers=buffers, params=params, annotations=annotations)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    return tvm.IRModule({"main": before["main"].with_attr("target", target)})


def _buffers(count=64, dtype="float32", output_dtype="float16"):
    a = tirx.decl_buffer((count,), dtype, name="input", scope="shared.dyn")
    b = tirx.decl_buffer((count,), dtype, name="other", scope="shared.dyn")
    out = tirx.decl_buffer((count,), output_dtype, name="result", scope="shared.dyn")
    return a, b, out


def _parallel(out, expression, count=64):
    i = tirx.Var("i", "int32")
    return tirx.For(i, 0, count, tirx.ForKind.PARALLEL, tirx.BufferStore(out, expression(i), [i]))


def _native(after):
    return [call for call in nodes(after, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _destination(call):
    return call.args[0].args[0]


def _constant(expr):
    return _constant(expr.value) if isinstance(expr, tirx.Cast) else float(expr)


def _evaluate_count_ops(after, initial):
    """Evaluate exact read/write spans, reading both inputs before a write."""
    values = {buffer.data: np.array(value, copy=True) for buffer, value in initial}

    def read(pointer, count):
        load = pointer.args[0]
        start = int(load.indices[0])
        return values[load.buffer.data][start : start + count].copy()

    for op in _native(after):
        name = op.op.name.rsplit(".", 1)[1]
        count = int(op.args[-1])
        dst = _destination(op)
        if name.startswith("float2half_"):
            result = read(op.args[1], count).astype(np.float16)
        else:
            lhs = read(op.args[1], count)
            rhs = _constant(op.args[2]) if name.endswith("_scalar") else read(op.args[2], count)
            operation = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[name.removesuffix("_scalar")]
            result = operation(lhs, rhs)
        if dst.buffer.data not in values:
            values[dst.buffer.data] = np.empty(int(np.prod([int(dim) for dim in dst.buffer.shape])), dtype=str(dst.buffer.dtype))
        start = int(dst.indices[0])
        values[dst.buffer.data][start : start + count] = result
    return values


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_terminal_cast_uses_dead_exact_input_and_preserves_values(operation):
    count = 17  # Native count tail, not a rounded vector-sized write.
    a, b, out = _buffers(count)
    operators = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "div": lambda x, y: x / y}
    body = _parallel(out, lambda i: T.cast(operators[operation](a[i], b[i]), "float16"), count)
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out]))
    arithmetic, narrow = _native(after)
    assert _destination(arithmetic).buffer.same_as(a)
    assert arithmetic.args[1].args[0].buffer.same_as(a)
    assert narrow.args[1].args[0].buffer.same_as(a)
    assert _destination(narrow).buffer.same_as(out)
    assert [int(arithmetic.args[index].args[1]) for index in (0, 1, 2)] == [count] * 3
    assert not any(buffer.name.startswith("membase_tmp_") for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers)
    av = np.linspace(1, 2, count, dtype=np.float32)
    bv = np.linspace(0.25, 0.75, count, dtype=np.float32)
    actual = _evaluate_count_ops(after, [(a, av), (b, bv)])[out.data]
    np.testing.assert_array_equal(actual, operators[operation](av, bv).astype(np.float16))


@pytest.mark.parametrize("rounding,native", [("rm", "float2half_rd"), ("rp", "float2half_ru"), ("rz", "float2half_rz")])
def test_reuse_preserves_terminal_rounding(rounding, native):
    a, b, out = _buffers()
    after = transform.AscendLowerVectorTiles()(
        _target(_parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16", round=rounding)), [a, b, out])
    )
    assert _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.same_as(a)
    assert len(calls(after, "tl.ascend_membase." + native)) == 1


def test_later_old_value_reader_prevents_destructive_destination():
    a, b, out = _buffers()
    later = _parallel(b, lambda i: a[i] + T.float32(1))
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    after = transform.AscendLowerVectorTiles()(_target(seq(expression, later), [a, b, out]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)
    av = np.linspace(1, 2, 64, dtype=np.float32)
    values = _evaluate_count_ops(after, [(a, av)])
    np.testing.assert_array_equal(values[a.data], av)
    np.testing.assert_array_equal(values[b.data], av + np.float32(1))


def test_old_value_in_sibling_expression_prevents_reuse():
    a, b, out = _buffers()
    after = transform.AscendLowerVectorTiles()(
        _target(_parallel(out, lambda i: T.cast((a[i] + T.float32(1)) * a[i], "float16")), [a, b, out])
    )
    assert all(not _destination(op).buffer.data.same_as(a.data) for op in _native(after))
    av = np.linspace(1, 2, 64, dtype=np.float32)
    np.testing.assert_array_equal(_evaluate_count_ops(after, [(a, av)])[out.data], ((av + np.float32(1)) * av).astype(np.float16))


@pytest.mark.parametrize("initialize", ["none", "partial", "full", "both_branches", "one_branch"])
def test_loop_backedge_requires_complete_next_iteration_definition(initialize):
    a, b, out = _buffers()
    source = tirx.decl_buffer((64,), "float32", name="source", scope="global")
    epoch = tirx.Var("epoch", "int32")
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    if initialize == "none":
        body = expression
    elif initialize == "partial":
        body = seq(copy(source, a, src_shape=[32], dst_shape=[32]), expression)
    elif initialize == "full":
        body = seq(copy(source, a), expression)
    else:
        otherwise = copy(b, a) if initialize == "both_branches" else tirx.Evaluate(0)
        body = seq(tirx.IfThenElse(epoch % 2 == 0, copy(source, a), otherwise), expression)
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), [a, b, out], [source]))
    reused = _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)
    assert reused == (initialize in ("full", "both_branches"))


def test_full_write_of_smaller_same_storage_view_is_not_a_definition():
    a, b, out = _buffers()
    view = tirx.decl_buffer((32,), "float32", name="view", data=a.data)
    source = tirx.decl_buffer((32,), "float32", name="source", scope="global")
    epoch = tirx.Var("epoch", "int32")
    body = seq(copy(source, view), _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16")))
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), [a, b, out], [source]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_alternate_view_keeps_copy_definition_conservative():
    a, b, out = _buffers()
    alias = tirx.decl_buffer((8, 8), "float32", name="alternate", data=a.data)
    source = tirx.decl_buffer((64,), "float32", name="source", scope="global")
    epoch = tirx.Var("epoch", "int32")
    use_view = _parallel(b, lambda i: alias[i // 8, i % 8])
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    # Until the physical view span is proved, a canonical logical copy cannot
    # kill all same-data aliases (which may include padding or another pitch).
    body = seq(copy(source, a), use_view, expression)
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), [a, b, out], [source]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


@pytest.mark.parametrize("trips", [1, 2])
def test_singleton_serial_loop_has_no_backedge(trips):
    a, b, out = _buffers()
    epoch = tirx.Var("epoch", "int32")
    body = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, trips, tirx.ForKind.SERIAL, body), [a, b, out]))
    assert _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data) == (trips == 1)


@pytest.mark.parametrize("field", ["min", "extent", "annotation"])
def test_later_fill_coordinates_and_annotations_keep_old_values_live(field):
    a, b, out = _buffers()
    destination = tirx.decl_buffer((64,), "float32", name="filled", scope="global")
    coordinate = tirx.Cast("int32", a[0]) % 4
    start = coordinate * 8 if field == "min" else 0
    extent = (coordinate + 1) * 8 if field == "extent" else 8
    annotations = {"runtime_bound": coordinate} if field == "annotation" else None
    fill = tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.fill"),
            [region(destination, 2, [start], [extent]), tirx.const(0, "float32")],
            annotations=annotations,
        )
    )
    body = seq(_parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16")), fill)
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out], [destination]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


@pytest.mark.parametrize("parallel", [False, True])
def test_loop_step_is_a_live_use_even_without_a_backedge(parallel):
    a, b, out = _buffers()
    epoch = tirx.Var("epoch", "int32")
    if parallel:
        # A semantically unit step whose unsimplified expression still reads A.
        step = tirx.Add(tirx.Mul(tirx.Cast("int32", a[0]), tirx.const(0, "int32")), tirx.const(1, "int32"))
        i = tirx.Var("i", "int32")
        store = tirx.BufferStore(out, T.cast(a[i] * T.float32(0.5), "float16"), [i])
        body = tirx.For(i, 0, 64, tirx.ForKind.PARALLEL, store, step=step)
    else:
        step = tirx.Cast("int32", a[0]) % 2 + 1
        expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
        body = tirx.For(epoch, 0, 1, tirx.ForKind.SERIAL, expression, step=step)
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_predicated_block_definition_keeps_the_skipped_path_live():
    a, b, out = _buffers()
    source = tirx.decl_buffer((64,), "float32", name="source", scope="global")
    epoch = tirx.Var("epoch", "int32")
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    block = tirx.SBlock([], [], [], "conditional_definition", copy(source, a))
    body = seq(expression, tirx.SBlockRealize([], epoch % 2 == 0, block))
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), [a, b, out], [source]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_reduction_init_is_not_a_definition_on_every_realization():
    a, b, out = _buffers()
    source = tirx.decl_buffer((64,), "float32", name="source", scope="global")
    epoch, axis = tirx.Var("epoch", "int32"), tirx.Var("axis", "int32")
    reduction_axis = tirx.IterVar(tvm.ir.Range(0, 4), axis, tirx.IterVar.CommReduce)
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    block = tirx.SBlock([reduction_axis], [], [], "reduction", expression, init=copy(source, a))
    body = tirx.SBlockRealize([epoch], True, block)
    after = transform.AscendLowerVectorTiles()(_target(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), [a, b, out], [source]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_unknown_pointer_consumer_prevents_reuse_even_before_last_read():
    a, b, out = _buffers()
    pointer = tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [a[0], 64, 1])
    escape = tirx.Evaluate(tirx.call_extern("int32", "retain_pointer", pointer))
    body = seq(escape, _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16")))
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_pointer_bind_prevents_reuse_through_a_different_var():
    a, b, out = _buffers()
    pointer = tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [a[0], 64, 1])
    saved = tirx.Var("saved", "handle")
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    consumer = tirx.Evaluate(tirx.call_extern("int32", "read_saved", saved))
    after = transform.AscendLowerVectorTiles()(_target(seq(tirx.Bind(saved, pointer), expression, consumer), [a, b, out]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_match_buffer_alias_prevents_false_last_use():
    a, b, out = _buffers()
    alias = tirx.decl_buffer((64,), "float32", name="matched", scope="shared.dyn")
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    body = seq(expression, _parallel(b, lambda i: alias[i] + T.float32(1)))
    match = tirx.MatchBufferRegion(alias, tirx.BufferRegion(a, [tvm.ir.Range(0, 64)]))
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=[a, b, out], match_buffers=[match])
    function = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr(
        {"global_symbol": "main", "target": tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})}
    )
    after = transform.AscendLowerVectorTiles()(tvm.IRModule({"main": function}))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_later_reduce_keeps_its_original_input():
    a, b, out = _buffers()
    reduced = tirx.decl_buffer((1,), "float32", name="reduced", scope="shared.dyn")
    reduction = tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.reduce"),
            [region(a, 1), region(reduced, 2), tirx.StringImm("sum"), 0, tirx.const(True, "bool")],
        )
    )
    body = seq(_parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16")), reduction)
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out, reduced]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_source_must_be_a_local_allocation():
    a, b, out = _buffers()
    after = transform.AscendLowerVectorTiles()(_target(_parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16")), [out], [a]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_shared_statement_node_requires_last_use_at_every_occurrence():
    a, b, out = _buffers()
    expression = _parallel(out, lambda i: T.cast(a[i] * T.float32(0.5), "float16"))
    body = seq(expression, _parallel(b, lambda i: a[i] + T.float32(1)), expression)
    after = transform.AscendLowerVectorTiles()(_target(body, [a, b, out]))
    assert all(not _destination(op).buffer.data.same_as(a.data) for op in calls(after, "tl.ascend_membase.mul_scalar"))


def test_existing_owner_and_version_coordinates_are_preserved():
    source = tirx.decl_buffer((2, 2, 64), "float32", name="versioned", scope="shared.dyn")
    out = tirx.decl_buffer((64,), "bfloat16", name="result", scope="shared.dyn")
    owner, epoch = tirx.Var("owner", "int32"), tirx.Var("epoch", "int32")
    body = _parallel(out, lambda i: T.cast(source[owner % 2, epoch % 2, i] * T.float32(0.5), "bfloat16"))
    private = {"tl.ascend_aiv_private_validated": [source.data, out.data]}
    before = _target(body, [source, out], [owner, epoch], annotations=private)
    after = transform.AscendLowerVectorTiles()(before)
    arithmetic = calls(after, "tl.ascend_membase.mul_scalar")[0]
    destination = _destination(arithmetic)
    original = arithmetic.args[1].args[0]
    assert destination.buffer.same_as(source)
    tvm.ir.assert_structural_equal(destination.indices, original.indices)
    members = nodes(after, tirx.SBlock)[0].annotations["tl.ascend_aiv_private_validated"]
    assert len(members) == 2 and members[0].same_as(source.data) and members[1].same_as(out.data)
    assert len(calls(after, "tl.ascend_membase.float2bfloat16_rn")) == 1


def test_broadcast_across_serialized_parallel_rows_is_not_overwritten():
    a = tirx.decl_buffer((64,), "float32", name="input", scope="shared.dyn")
    out = tirx.decl_buffer((2, 64), "float16", name="result", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.For(
        i,
        0,
        2,
        tirx.ForKind.PARALLEL,
        tirx.For(j, 0, 64, tirx.ForKind.PARALLEL, tirx.BufferStore(out, T.cast(a[j] * T.float32(0.5), "float16"), [i, j])),
    )
    after = transform.AscendLowerVectorTiles()(_target(body, [a, out]))
    assert not _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.data.same_as(a.data)


def test_shifted_same_storage_operand_stays_separate():
    a = tirx.decl_buffer((72,), "float32", name="input", scope="shared.dyn")
    alias = tirx.decl_buffer((72,), "float32", name="alias", data=a.data)
    out = tirx.decl_buffer((64,), "float16", name="result", scope="shared.dyn")
    after = transform.AscendLowerVectorTiles()(_target(_parallel(out, lambda i: T.cast(a[i] + alias[i + 8], "float16")), [a, out]))
    assert not _destination(calls(after, "tl.ascend_membase.add")[0]).buffer.data.same_as(a.data)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_arithmetic_keeps_its_rounding_storage(dtype):
    a, b, out = _buffers(dtype=dtype, output_dtype="float32")
    after = transform.AscendLowerVectorTiles()(_target(_parallel(out, lambda i: T.cast(a[i] * b[i], "float32")), [a, b, out]))
    assert all(
        not _destination(op).buffer.data.same_as(a.data) and not _destination(op).buffer.data.same_as(b.data) for op in _native(after)
    )
    if dtype == "bfloat16":
        assert len(calls(after, "tl.ascend_membase.float2bfloat16_rn")) == 1


def test_declared_input_and_work_tiles_remain_distinct():
    a, work, out = _buffers()
    body = seq(_parallel(work, lambda i: a[i] * T.float32(0.5)), _parallel(out, lambda i: T.cast(work[i], "float16")))
    after = transform.AscendLowerVectorTiles()(_target(body, [a, work, out]))
    assert _destination(calls(after, "tl.ascend_membase.mul_scalar")[0]).buffer.same_as(work)
    assert calls(after, "tl.ascend_membase.float2half_rn")[0].args[1].args[0].buffer.same_as(work)


def test_row_normalization_reuses_matrix_and_keeps_compact_row_scratch():
    rows, columns = 128, 64
    matrix = tirx.decl_buffer((rows, columns), "float32", name="matrix", scope="shared.dyn")
    row = tirx.decl_buffer((rows,), "float32", name="row", scope="shared.dyn")
    out = tirx.decl_buffer((rows, columns), "float16", name="result", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.For(
        i,
        0,
        rows,
        tirx.ForKind.PARALLEL,
        tirx.For(j, 0, columns, tirx.ForKind.PARALLEL, tirx.BufferStore(out, T.cast(matrix[i, j] / row[i], "float16"), [i, j])),
    )
    after = transform.AscendLowerVectorTiles()(_target(body, [matrix, row, out]))
    assert all(_destination(op).buffer.same_as(matrix) for op in calls(after, "tl.ascend_membase.div_rows_f32"))
    scratch = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("membase_tmp_")]
    assert len(scratch) == 1 and int(scratch[0].shape[0]) == rows * 8
    assert calls(after, "tl.ascend_membase.float2half_rn")[0].args[1].args[0].buffer.same_as(matrix)
