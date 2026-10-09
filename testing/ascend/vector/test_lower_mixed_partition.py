"""Two AIVs share logical rows while keeping independent compact UB storage."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq


def _lower(mod):
    return tvm.ffi.get_global_func("tl.transform.AscendLowerMixedPartition")()(mod)


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _cube():
    a, b = [tirx.decl_buffer((16, 16), "float16", name=name, scope="shared.l1") for name in ("cube_a", "cube_b")]
    c = tirx.decl_buffer((16, 16), "float32", name="cube_c", scope="shared.l0c")
    return tirx.Evaluate(T.gemm(a, b, c, transpose_B=True)), [a, b, c]


def _parallel(body, *loops):
    for var, extent in reversed(loops):
        body = tirx.For(var, 0, extent, tirx.ForKind.PARALLEL, body)
    return body


def _program(rows=5, width=32, *, start=0, total=None, valid=None, row_math=False, extra=None):
    total = rows + start if total is None else total
    a, c = [tirx.decl_buffer((total, width), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((rows, width), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    expression = x[i, j] + (tirx.Cast("float32", i) if row_math else tirx.const(1, "float32"))
    math = _parallel(tirx.BufferStore(y, expression, [i, j]), (i, rows), (j, width))
    cube, buffers = _cube()
    shape = [rows if valid is None else valid, width]
    body = [cube, copy(a, x, src_indices=[start, 0], src_shape=shape, dst_shape=shape), math]
    if extra is not None:
        additions, private = extra(x, y, i, j)
        body += additions
        buffers += private
    body.append(copy(y, c, src_shape=shape, dst_indices=[start, 0], dst_shape=shape))
    params = [a, c] + ([valid] if isinstance(valid, tirx.Var) else [])
    return _target(kernel(seq(*body), buffers=[*buffers, x, y], params=params))


def _root(mod):
    return next(block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root")


def _sid(mod):
    return next(attr.node.var for attr in nodes(mod, tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == "cthread")


def _constant(expr, replacements):
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, replacements)))


def _regions(call):
    return [(argument.args[0].buffer, list(argument.args[0].indices), list(argument.args[2:])) for argument in call.args[:2]]


@pytest.mark.parametrize("rows, counts", [(5, [3, 2]), (4, [2, 2]), (1, [1, 0])])
def test_odd_even_and_empty_owner_regions_are_disjoint_and_complete(rows, counts):
    before = _program(rows)
    after = _lower(before)
    sid = _sid(after)
    root = _root(after)
    assert int(root.annotations["tl.ascend_row_partitioned"]) == 2
    assert int(root.annotations["vector_count"]) == 2
    private = [buffer for buffer in root.alloc_buffers if buffer.scope() == "shared.dyn"]
    assert [tuple(int(size) for size in buffer.shape) for buffer in private] == [(max(counts), 32)] * 2
    load, store = calls(after, "tl.tileop.ascend_copy")
    intervals = []
    for owner, count in enumerate(counts):
        replacements = {sid: tirx.IntImm(sid.dtype, owner)}
        gm, ub = _regions(load)
        origin = _constant(gm[1][0], replacements)
        assert _constant(gm[2][0], replacements) == count
        assert _constant(ub[2][0], replacements) == count
        assert int(ub[1][0]) == 0
        out_ub, out_gm = _regions(store)
        assert _constant(out_gm[1][0], replacements) == origin
        assert _constant(out_gm[2][0], replacements) == count
        assert _constant(out_ub[2][0], replacements) == count
        intervals.extend(range(origin, origin + count))
    assert intervals == list(range(rows))
    # Data-empty owners keep both whole-copy statements. The later native copy
    # pass supplies the internal count guard after collective synchronization.
    assert not nodes(after, tirx.IfThenElse)
    assert len(calls(after, "tl.tileop.gemm")) == 1


def test_gm_tail_clamps_the_actual_domain_before_owner_split():
    after = _lower(_program(rows=5, start=7, total=9))
    sid = _sid(after)
    load = calls(after, "tl.tileop.ascend_copy")[0]
    gm, ub = _regions(load)
    assert [_constant(ub[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [2, 0]
    assert [_constant(gm[1][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [7, 9]
    assert tuple(int(size) for size in ub[0].shape) == (3, 32)


def test_workspace_capacity_does_not_replace_a_dynamic_valid_region():
    valid = tirx.Var("valid", "int32")
    after = _lower(_program(rows=5, total=64, valid=valid))
    sid = _sid(after)
    gm, ub = _regions(calls(after, "tl.tileop.ascend_copy")[0])
    for actual, expected in ((0, [0, 0]), (2, [2, 0]), (4, [3, 1]), (5, [3, 2])):
        assert [
            _constant(ub[2][0], {sid: tirx.IntImm("int32", owner), valid: tirx.IntImm("int32", actual)})
            for owner in (0, 1)
        ] == expected
    assert int(gm[0].shape[0]) == 64


def test_local_ub_indices_and_logical_row_math_use_one_coordinate():
    after = _lower(_program(rows=5, row_math=True))
    sid = _sid(after)
    outer, inner = nodes(after, tirx.For)[::-1]
    assert outer.kind == tirx.ForKind.SERIAL
    assert inner.kind == tirx.ForKind.PARALLEL and int(inner.extent) == 32
    store = next(store for store in nodes(after, tirx.BufferStore) if store.buffer.name == "y")
    assert store.indices[0].same_as(outer.loop_var)
    source = next(load for load in nodes(store.value, tirx.BufferLoad) if load.buffer.name == "x")
    assert source.indices[0].same_as(outer.loop_var)
    cast = next(cast for cast in nodes(store.value, tirx.Cast) if cast.dtype == "float32")
    assert _constant(cast.value, {sid: tirx.IntImm("int32", 1), outer.loop_var: tirx.IntImm("int32", 1)}) == 4
    assert [_constant(outer.extent, {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]


def _reduce(src, dst, axis=1):
    return tirx.Evaluate(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [region(src, 1), region(dst, 2), tirx.StringImm("sum"), axis, tirx.const(True, "bool")])
    )


def test_trailing_reduce_and_row_broadcast_propagate_the_same_owner():
    def extra(x, y, i, j):
        sums = tirx.decl_buffer((5,), "float32", name="sums", scope="shared.dyn")
        result = _parallel(tirx.BufferStore(y, x[i, j] / sums[i], [i, j]), (i, 5), (j, 32))
        return [_reduce(x, sums), result], [sums]

    after = _lower(_program(extra=extra))
    sid = _sid(after)
    reduction = calls(after, "tl.tileop.reduce")[0]
    src, dst = _regions(reduction)[:2]
    assert [int(src[0].shape[0]), int(dst[0].shape[0])] == [3, 3]
    assert [_constant(dst[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]
    broadcast = next(load for load in nodes(after, tirx.BufferLoad) if load.buffer.name == "sums" and isinstance(load.indices[0], tirx.Var))
    assert broadcast.indices[0].name == "row"


def test_private_fill_initializes_capacity_but_export_uses_only_valid_rows():
    def extra(x, y, i, j):
        return [tirx.Evaluate(T.fill(y, 0))], []

    after = _lower(_program(rows=5, total=2, extra=extra))
    fill = calls(after, "tl.tileop.fill")[0]
    assert int(fill.args[0].args[2]) == 3
    sid = _sid(after)
    _, gm = _regions(calls(after, "tl.tileop.ascend_copy")[-1])
    assert [_constant(gm[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [2, 0]


def test_ub_copy_relation_propagates_row_ownership():
    def extra(x, y, i, j):
        temporary = tirx.decl_buffer((5, 32), "float32", name="temporary", scope="shared.dyn")
        return [copy(x, temporary), copy(temporary, y)], [temporary]

    after = _lower(_program(extra=extra))
    temporary = next(buffer for buffer in _root(after).alloc_buffers if buffer.name == "temporary")
    assert int(temporary.shape[0]) == 3
    sid = _sid(after)
    for call in calls(after, "tl.tileop.ascend_copy")[1:3]:
        src, dst = _regions(call)
        assert int(src[1][0]) == int(dst[1][0]) == 0
        assert [_constant(src[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]


def _tail_cache_program(*, rank=1, cache_first=False, copy_extent=None, source_origin=0, destination_origin=0, destination_extent=None):
    rows, width = 32, 16
    group = tirx.Var("query_tile", "int32")
    valid = tirx.min(rows, 33 - group * rows)
    a, c = [tirx.decl_buffer((33, width), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((rows, width), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    shape = (rows,) if rank == 1 else (rows, width)
    state, cached = [tirx.decl_buffer(shape, "float32", name=name, scope="shared.dyn") for name in ("state", "cached")]
    extent = valid if copy_extent is None else copy_extent
    dst_extent = extent if destination_extent is None else destination_extent
    cache = copy(state, cached, src_indices=[source_origin] + [0] * (rank - 1),
                 dst_indices=[destination_origin] + [0] * (rank - 1),
                 src_shape=[extent] + list(shape[1:]), dst_shape=[dst_extent] + list(shape[1:]))
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    if rank == 1:
        state_update = _reduce(x, state)
        value = cached[i]
    else:
        state_update = _parallel(tirx.BufferStore(state, x[i, j] + 1, [i, j]), (i, rows), (j, width))
        value = cached[i, j]
    math = _parallel(tirx.BufferStore(y, x[i, j] + value, [i, j]), (i, rows), (j, width))
    cube, buffers = _cube()
    body = [cube, tirx.Evaluate(T.fill(state, 0))]
    if cache_first:
        body.append(cache)
    body += [copy(a, x, src_indices=[group * rows, 0], src_shape=[valid, width], dst_shape=[valid, width]), state_update]
    if not cache_first:
        body.append(cache)
    body += [math, copy(y, c, src_shape=[valid, width], dst_indices=[group * rows, 0], dst_shape=[valid, width])]
    mod = _target(kernel(seq(*body), buffers=[*buffers, x, y, state, cached], params=[a, c]))
    grid = tirx.IterVar(tvm.ir.Range(0, 2), group, tirx.IterVar.ThreadIndex, "blockIdx.x")
    return tvm.IRModule({"main": mod["main"].with_body(tirx.AttrStmt(grid, "thread_extent", 2, mod["main"].body))}), group


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("cache_first", [False, True])
@pytest.mark.parametrize("full_copy", [False, True])
def test_ub_cache_copy_consumes_the_connected_valid_domain_independent_of_statement_order(rank, cache_first, full_copy):
    before, group = _tail_cache_program(rank=rank, cache_first=cache_first, copy_extent=32 if full_copy else None)
    after = _lower(before)
    sid = _sid(after)
    copies = calls(after, "tl.tileop.ascend_copy")
    cache = next(call for call in copies if _regions(call)[0][0].name == "state")
    src, dst = _regions(cache)
    assert int(src[0].shape[0]) == int(dst[0].shape[0]) == 16
    assert int(src[1][0]) == int(dst[1][0]) == 0
    for tile, expected in ((0, [16, 16]), (1, [1, 0])):
        for operand in (src, dst):
            assert [
                _constant(operand[2][0], {group: tirx.IntImm("int32", tile), sid: tirx.IntImm("int32", owner)})
                for owner in (0, 1)
            ] == expected
    # The owner with no tail rows retains the same whole-copy operation. No
    # owner-dependent task guard is introduced around collective completion.
    assert len(copies) == 3
    assert not nodes(after, tirx.IfThenElse)
    gm_store = next(call for call in copies if _regions(call)[1][0].name == "C")
    _, gm = _regions(gm_store)
    assert [
        _constant(gm[1][0], {group: tirx.IntImm("int32", 1), sid: tirx.IntImm("int32", owner)})
        for owner in (0, 1)
    ] == [32, 33]


@pytest.mark.parametrize("change,message", [
    ({"copy_extent": 2}, "full logical or established valid row domain"),
    ({"source_origin": 1}, "start at logical row zero"),
    ({"destination_origin": 1}, "destinations must start at row zero"),
    ({"destination_extent": 2}, "UB row copy extents must match"),
])
def test_partial_ub_copy_cannot_invent_or_shift_the_connected_valid_domain(change, message):
    before, _ = _tail_cache_program(**change)
    with pytest.raises(ValueError, match=message):
        _lower(before)


def test_partial_ub_copy_without_a_gm_valid_domain_remains_a_diagnostic():
    before, _ = _tail_cache_program(copy_extent=1)
    root = _root(before)
    body = [stmt for stmt in root.body.seq if not (
        isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)
        and stmt.value.op.name == "tl.tileop.ascend_copy"
        and any(operand[0].scope() == "global" for operand in _regions(stmt.value))
    )]
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=root.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    with pytest.raises(ValueError, match="full logical or established valid row domain"):
        _lower(before)


def test_scalar_row_output_remains_parallel_intent_with_bounded_dynamic_count():
    def extra(x, y, i, j):
        sums = tirx.decl_buffer((5,), "float32", name="sums", scope="shared.dyn")
        adjusted = tirx.decl_buffer((5,), "float32", name="adjusted", scope="shared.dyn")
        k = tirx.Var("row_output", "int32")
        math = _parallel(tirx.BufferStore(adjusted, sums[k] + 1, [k]), (k, 5))
        return [_reduce(x, sums), math], [sums, adjusted]

    after = _lower(_program(extra=extra))
    scalar_rows = next(loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "row_output")
    assert scalar_rows.kind == tirx.ForKind.PARALLEL
    sid = _sid(after)
    assert [_constant(scalar_rows.extent, {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]


def test_leading_group_axis_is_preserved_in_gm_regions():
    rows, width = 5, 32
    a, c = [tirx.decl_buffer((2, rows, width), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    cube, buffers = _cube()
    body = seq(
        cube,
        copy(a, x, src_indices=[1, 0, 0], src_shape=[1, rows, width]),
        _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width)),
        copy(x, c, dst_indices=[1, 0, 0], dst_shape=[1, rows, width]),
    )
    after = _lower(_target(kernel(body, buffers=[*buffers, x], params=[a, c])))
    gm, _ = _regions(calls(after, "tl.tileop.ascend_copy")[0])
    assert int(gm[1][0]) == int(gm[2][0]) == 1
    sid = _sid(after)
    assert [_constant(gm[1][1], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [0, 3]


def test_uniform_leading_zero_or_one_validity_composes_with_owned_rows():
    rows, width = 5, 32
    a, c = [tirx.decl_buffer((2, rows, width), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    active = tirx.Var("group_active", "int32")
    valid = tirx.max(tirx.min(active, 1), 0)
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    cube, buffers = _cube()
    body = seq(cube, copy(a, x, src_shape=[valid, rows, width]),
        _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width)),
        copy(x, c, dst_shape=[valid, rows, width]))
    after = _lower(_target(kernel(body, buffers=[*buffers, x], params=[a, c, active])))
    sid = _sid(after)
    _, ub = _regions(calls(after, "tl.tileop.ascend_copy")[0])
    for activation, expected in ((0, [0, 0]), (1, [3, 2])):
        assert [_constant(ub[2][0], {sid: tirx.IntImm("int32", owner), active: tirx.IntImm("int32", activation)}) for owner in (0, 1)] == expected
    assert not nodes(after, tirx.IfThenElse)


def test_outer_grid_context_proves_a_handover_singleton_is_live():
    rows, width = 5, 32
    a, c = [tirx.decl_buffer((2, rows, width), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    group = tirx.Var("group", "int32")
    live = tirx.Select(group < 2, 1, 0)
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    cube, buffers = _cube()
    body = seq(cube, copy(a, x, src_indices=[group, 0, 0], src_shape=[live, rows, width]),
        _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width)),
        copy(x, c, dst_indices=[group, 0, 0], dst_shape=[1, rows, width]))
    before = _target(kernel(body, buffers=[*buffers, x], params=[a, c]))
    grid = tirx.IterVar(tvm.ir.Range(0, 2), group, tirx.IterVar.ThreadIndex, "blockIdx.x")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(grid, "thread_extent", 2, before["main"].body))})
    after = _lower(before)
    sid = _sid(after)
    _, ub = _regions(calls(after, "tl.tileop.ascend_copy")[0])
    assert [_constant(ub[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]


def test_two_kernels_get_independent_owner_bindings_and_no_host_allocation():
    first, second = _program(5), _program(1)
    host = tirx.SBlock([], [], [], "root", seq(first["main"].body, second["main"].body))
    before = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, host)).with_attr("target", first["main"].attrs["target"])})
    after = _lower(before)
    roots = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root"]
    assert len(roots) == 2
    assert [int(next(buffer for buffer in root.alloc_buffers if buffer.name == "x").shape[0]) for root in roots] == [3, 1]
    bindings = [attr.node.var for attr in nodes(after, tirx.AttrStmt) if attr.attr_key == "thread_extent"]
    assert len(bindings) == 2 and not bindings[0].same_as(bindings[1])
    assert not next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "root").alloc_buffers


@pytest.mark.parametrize("arch,cube", [("dav-3510", True), ("dav-2201", False)])
def test_other_architecture_and_pure_vector_preserve_existing_ir(arch, cube):
    before = _program()
    if not cube:
        root = _root(before)
        body = tirx.SeqStmt(list(root.body.seq)[1:])
        root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=root.alloc_buffers)
        before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    before = _target(before, arch)
    tvm.ir.assert_structural_equal(_lower(before), before)


@pytest.mark.parametrize("pipe", ["PIPE_V", "PIPE_ALL"])
@pytest.mark.parametrize("auto_schedule", [False, True], ids=["no-auto", "auto"])
def test_pure_vector_barrier_does_not_require_mixed_row_ownership(pipe, auto_schedule):
    @T.prim_func
    def main(A: T.Tensor((5, 32), "float32"), C: T.Tensor((5, 32), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((5, 32), "float32")
            with T.Vector(vector=1):
                T.copy(A, x)
                T.ascend_pipe_barrier(pipe)
                T.copy(x, C)

    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": auto_schedule}):
        artifact = tilelang.lower(
            main, target={"kind": "ascend", "arch": "dav-2201"},
            enable_host_codegen=False, enable_device_compile=False,
        )
    source = artifact.kernel_source
    assert "__vector__" in source and "__mix__" not in source
    if pipe == "PIPE_ALL":
        assert source.count("asc_sync();") == 1
    else:
        assert "asc_sync_pipe(" + pipe + ");" in source


def test_repeated_pass_is_idempotent():
    once = _lower(_program())
    tvm.ir.assert_structural_equal(_lower(once), once)


def test_cross_row_shift_is_a_boundary_diagnostic():
    before = _program()
    root = _root(before)
    x, y = [next(buffer for buffer in root.alloc_buffers if buffer.name == name) for name in ("x", "y")]
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    shifted = _parallel(tirx.BufferStore(y, x[(i + 1) % 5, j], [i, j]), (i, 5), (j, 32))
    body = list(root.body.seq)
    body[2] = shifted
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=root.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    with pytest.raises(ValueError, match="gathers, shifts, or cross-row"):
        _lower(before)


def test_reduce_across_owner_axis_requires_a_collective_plan():
    def extra(x, y, i, j):
        columns = tirx.decl_buffer((32,), "float32", name="columns", scope="shared.dyn")
        return [_reduce(x, columns, axis=0)], [columns]

    with pytest.raises(ValueError, match="cross-AIV reductions require a collective plan"):
        _lower(_program(extra=extra))


def test_unexpanded_direct_cube_handover_is_rejected():
    def extra(x, y, i, j):
        l0c = tirx.decl_buffer((5, 32), "float32", name="unexpanded", scope="shared.l0c")
        return [copy(l0c, x)], [l0c]

    with pytest.raises(ValueError, match="expanded into GM copies before row partitioning"):
        _lower(_program(extra=extra))


def test_handover_only_cube_intent_is_partitioned_without_a_gemm_call():
    before = _program()
    root = _root(before)
    x = next(buffer for buffer in root.alloc_buffers if buffer.name == "x")
    l0c = tirx.decl_buffer((5, 32), "float32", name="handover_l0c", scope="shared.l0c")
    workspace = tirx.decl_buffer((5, 32), "float32", name="handover")
    body = list(root.body.seq)
    # The real AIC operation is FIX L0C→GM, followed by an ordinary AIV load.
    body[0] = copy(l0c, workspace)
    body[1] = copy(workspace, x)
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=[*root.alloc_buffers, l0c, workspace])
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    after = _lower(before)
    assert not calls(after, "tl.tileop.gemm")
    assert int(_root(after).annotations["tl.ascend_row_partitioned"]) == 2
    assert int(next(buffer for buffer in _root(after).alloc_buffers if buffer.name == "x").shape[0]) == 3


@pytest.mark.parametrize("intent", ["gemm", "handover"])
def test_vector_barrier_does_not_hide_independent_cube_data_intent(intent):
    before = _program()
    root = _root(before)
    body = list(root.body.seq)
    buffers = list(root.alloc_buffers)
    if intent == "handover":
        x = next(buffer for buffer in buffers if buffer.name == "x")
        l0c = tirx.decl_buffer((5, 32), "float32", name="handover_l0c", scope="shared.l0c")
        workspace = tirx.decl_buffer((5, 32), "float32", name="handover")
        body[0], body[1] = copy(l0c, workspace), copy(workspace, x)
        buffers.extend([l0c, workspace])
    # Scope state must end with this Vector region: the following real Cube
    # data operation still establishes mixed ownership for the row algorithm.
    body.insert(0, tirx.SBlock([], [], [], "VECTOR", tirx.Evaluate(T.ascend_pipe_barrier("PIPE_ALL"))))
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    after = _lower(before)
    assert int(_root(after).annotations["tl.ascend_row_partitioned"]) == 2
    assert int(next(buffer for buffer in _root(after).alloc_buffers if buffer.name == "x").shape[0]) == 3
    assert len(calls(after, "tl.tileop.gemm")) == int(intent == "gemm")
    assert len(calls(after, "tl.ascend_pipe_barrier")) == 1


def test_unscoped_cube_cross_endpoint_still_establishes_mixed_row_ownership():
    before = _program()
    root = _root(before)
    body = list(root.body.seq)
    # The explicit FIX endpoint belongs to Cube even without a GEMM/data
    # call. Its MTE3 consumer then runs on the two partitioned AIV owners.
    body[:1] = [
        tirx.Evaluate(T.ascend_cross_core_set_flag(2, "PIPE_FIX", 0)),
        tirx.Evaluate(T.ascend_cross_core_wait_flag(2, "PIPE_MTE3", 0)),
    ]
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=root.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    after = _lower(before)
    assert int(_root(after).annotations["tl.ascend_row_partitioned"]) == 2
    sid = _sid(after)
    load = calls(after, "tl.tileop.ascend_copy")[0]
    _, ub = _regions(load)
    assert [_constant(ub[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]
    assert len(calls(after, "tl.ascend_cross_core_set_flag")) == 1
    assert len(calls(after, "tl.ascend_cross_core_wait_flag")) == 1


@pytest.mark.parametrize("operation", ["typed-fill", "native-fill", "native-gm-to-l1"])
def test_l1_writes_establish_cube_intent_on_the_shared_mte2_pipe(operation):
    before = _program()
    root = _root(before)
    body = list(root.body.seq)
    params = list(before["main"].buffer_map.values())
    l1 = next(buffer for buffer in root.alloc_buffers if buffer.name == "cube_a")
    if operation == "typed-fill":
        body[0] = tirx.Evaluate(T.fill(l1, 0))
    elif operation == "native-fill":
        # One NZ repeat writes sixteen 32-byte blocks: the real 16x16 FP16
        # destination, with no byte offset or gap.
        body[0] = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_fill_l1"), [
            l1.access_ptr("w"), 0, tirx.IntImm("uint32", 0), 1, 16, 0, 16,
        ]))
    else:
        gm = tirx.decl_buffer((16, 16), "float16", name="l1_input")
        params.append(gm)
        # The native ABI carries source row pitch in bytes and n/d in
        # elements. This full ND2NZ copy writes the same 512-byte L1 tile.
        body[0] = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_copy_gm_to_cbuf"), [
            l1.access_ptr("w"), gm.access_ptr("r"), 0, 32, 0,
            16, 16, 0, 0, 0, 16, tirx.StringImm("half"),
        ]))
    after = _lower(_target(kernel(seq(*body), buffers=root.alloc_buffers, params=params)))
    assert not calls(after, "tl.tileop.gemm")
    assert int(_root(after).annotations["tl.ascend_row_partitioned"]) == 2
    sid = _sid(after)
    _, ub = _regions(calls(after, "tl.tileop.ascend_copy")[0])
    assert [_constant(ub[2][0], {sid: tirx.IntImm("int32", owner)}) for owner in (0, 1)] == [3, 2]


def test_frontend_opaque_blocks_keep_affine_row_facts_and_private_allocations():
    @T.prim_func
    def main(A: T.Tensor((5, 32), "float32"), C: T.Tensor((5, 32), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), "float16")
            b_l1 = T.alloc_l1((16, 16), "float16")
            c_l0 = T.alloc_l0c((16, 16), "float32")
            x = T.alloc_shared((5, 32), "float32")
            y = T.alloc_shared((5, 32), "float32")
            T.gemm(a_l1, b_l1, c_l0, transpose_B=True)
            T.copy(A, x)
            for i, j in T.Parallel(5, 32):
                y[i, j] = x[i, j] + 1.0
            T.copy(y, C)

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = tirx.transform.BindTarget(target)(tvm.IRModule.from_expr(main))
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True, lower_thread_binding=False, default_threads=None, unsupported_annotations=["cluster_dims"], launch_dim_tags=["cthread"]
    )(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    original_layouts = dict(_root(mod).annotations["layout_map"])
    assert original_layouts  # frontend alloc_l0c establishes its native layout
    with target:
        mod = transform.AscendLowerL1Gemm()(mod)
    after = _lower(mod)
    actual_layouts = _root(after).annotations["layout_map"]
    assert all(actual_layouts[data].same_as(layout) for data, layout in original_layouts.items())
    private = [buffer for buffer in _root(after).alloc_buffers if buffer.scope() == "shared.dyn"]
    assert len(private) == 2 and all(tuple(int(size) for size in buffer.shape) == (3, 32) for buffer in private)
    # Ownership consumes Var-keyed frontend layout intent. Vector instructions
    # consume the physical Buffer-keyed layouts established by inference.
    with target:
        prepared = transform.AscendLayoutInference()(after)
        prepared = transform.NormalizeAscendFractalStorage()(prepared)
    lowered = transform.AscendLowerVectorTiles()(prepared)
    assert len(calls(lowered, "tl.ascend_membase.add_scalar")) == 1
    host = next(block for block in nodes(lowered, tirx.SBlock) if block.name_hint == "root")
    assert not host.alloc_buffers


def test_owned_ub_layout_is_rejected_while_native_cube_layouts_are_preserved():
    from tilelang.layout import Layout

    before = _program()
    root = _root(before)
    x = next(buffer for buffer in root.alloc_buffers if buffer.name == "x")
    root = tirx.SBlock(
        [], [], [], "tilelang_root", root.body, alloc_buffers=root.alloc_buffers,
        annotations={"layout_map": {x.data: Layout((5, 32), lambda i, j: (i, j))}},
    )
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    with pytest.raises(ValueError, match="before owned UB layout annotations"):
        _lower(before)


def test_conflicting_valid_copy_domains_are_rejected():
    before = _program()
    root = _root(before)
    body = list(root.body.seq)
    x = next(buffer for buffer in root.alloc_buffers if buffer.name == "x")
    a = next(iter(before["main"].buffer_map.values()))
    body[1] = copy(a, x, src_shape=[2, 32], dst_shape=[2, 32])
    root = tirx.SBlock([], [], [], "tilelang_root", seq(*body), alloc_buffers=root.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    with pytest.raises(ValueError, match="disagree on the actual valid row domain"):
        _lower(before)


def _guarded_workspace_program(kind="single"):
    """A simplified workspace is full only while the second query is present."""
    rows, width = 16, 32
    workspace = tirx.decl_buffer((rows, width), "float32", name="workspace")
    output = tirx.decl_buffer((65, width), "float32", name="output")
    x, y = [tirx.decl_buffer((rows, width), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    group = tirx.Var("query_pair", "int32")
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    math = _parallel(tirx.BufferStore(y, x[i, j] + 1, [i, j]), (i, rows), (j, width))
    load = copy(workspace, x)
    export = copy(y, output, dst_indices=[group * 32 + 16, 0], dst_shape=[rows, width])
    cube, buffers = _cube()
    params = [workspace, output]
    if kind == "single":
        body = tirx.IfThenElse(group < 2, seq(load, math, export), None)
    elif kind == "equivalent":
        body = seq(
            tirx.IfThenElse(group < 2, load, None),
            tirx.IfThenElse(2 > group, math, None),
            tirx.IfThenElse(group <= 1, export, None),
        )
    elif kind == "shared-call":
        # Deliberately reuse the SAME Call node. The first occurrence must not
        # borrow the second occurrence's guard and disappear from validation.
        body = seq(
            tirx.IfThenElse(group >= 2, load, None),
            tirx.IfThenElse(group < 2, seq(load, math, export), None),
        )
    elif kind == "mutable":
        state = tirx.decl_buffer((1,), "int32", name="state")
        params.append(state)
        guard = tirx.And(state[0] == group, state[0] < 2)
        body = tirx.IfThenElse(guard, seq(load, math, export), None)
    elif kind in ("local-bind", "escaped-bind"):
        valid = tirx.Var("local_valid", "int32")
        load = copy(workspace, x, src_shape=[valid, width], dst_shape=[valid, width])
        inside = seq(tirx.Bind(valid, rows), load, math)
        body = (
            tirx.IfThenElse(group < 2, seq(tirx.Bind(valid, rows), load, math, export), None)
            if kind == "local-bind"
            else seq(tirx.IfThenElse(group < 2, inside, None), export)
        )
    else:
        raise AssertionError(kind)
    before = _target(kernel(seq(cube, body), buffers=[*buffers, x, y], params=params))
    grid = tirx.IterVar(tvm.ir.Range(0, 3), group, tirx.IterVar.ThreadIndex, "blockIdx.x")
    before = tvm.IRModule({"main": before["main"].with_body(tirx.AttrStmt(grid, "thread_extent", 3, before["main"].body))})
    return before, group


@pytest.mark.parametrize("kind", ["single", "equivalent", "local-bind"])
def test_guard_simplified_workspace_uses_a_common_clipped_domain(kind):
    before, group = _guarded_workspace_program(kind)
    after = _lower(before)
    sid = _sid(after)
    # The original presence guards survive; partitioning never hides an empty
    # AIV's collective completion inside a new owner-specific branch.
    assert len(nodes(after, tirx.IfThenElse)) == len(nodes(before, tirx.IfThenElse))
    transfers = calls(after, "tl.tileop.ascend_copy")
    assert len(transfers) == 2
    for tile in range(3):
        for owner in range(2):
            replacements = {group: tirx.IntImm("int32", tile), sid: tirx.IntImm("int32", owner)}
            if kind == "local-bind":
                valid = next(bind.var for bind in nodes(after, tirx.Bind) if bind.var.name == "local_valid")
                replacements[valid] = tirx.IntImm("int32", 16)
            expected = 8 if tile < 2 else 0
            for transfer in transfers:
                for buffer, _, extents in _regions(transfer):
                    assert _constant(extents[0], replacements) == expected
                    if buffer.scope() == "shared.dyn":
                        assert tuple(int(size) for size in buffer.shape) == (8, 32)


@pytest.mark.parametrize("kind", ["shared-call", "mutable"])
def test_boundary_proofs_do_not_borrow_another_occurrences_guard(kind):
    before, _ = _guarded_workspace_program(kind)
    with pytest.raises(ValueError, match="disagree on the actual valid row domain"):
        _lower(before)


def test_common_boundary_domain_cannot_escape_its_definition_scope():
    before, _ = _guarded_workspace_program("escaped-bind")
    with pytest.raises(ValueError, match="must be in scope at every component use"):
        _lower(before)


@pytest.mark.parametrize("overlap", [False, True])
def test_different_guarded_boundary_domains_are_rejected_in_each_own_context(overlap):
    rows, width = 16, 32
    a, c = [tirx.decl_buffer((rows, width), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    active = tirx.Var("active", "int32")
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    math = _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width))
    load = copy(a, x)
    store = copy(x, c, src_shape=[8, width], dst_shape=[8, width])
    # In the disjoint case combining then/else guards would prove anything.
    guarded = (
        tirx.IfThenElse(active > 0, seq(load, store), None)
        if overlap
        else tirx.IfThenElse(active > 0, load, store)
    )
    cube, buffers = _cube()
    before = _target(kernel(seq(cube, guarded, math), buffers=[*buffers, x], params=[a, c, active]))
    with pytest.raises(ValueError, match="disagree on the actual valid row domain"):
        _lower(before)


def test_mutable_outer_kernel_guards_do_not_prove_boundary_domains_equal():
    rows, width = 16, 32
    a, c = [tirx.decl_buffer((rows, width), "float32", name=name) for name in ("A", "C")]
    state = tirx.decl_buffer((1,), "int32", name="state")
    valid = tirx.Var("valid", "int32")
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    cube, buffers = _cube()
    body = seq(
        cube,
        copy(a, x, src_shape=[valid, width], dst_shape=[valid, width]),
        _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width)),
        copy(x, c, src_shape=[8, width], dst_shape=[8, width]),
    )
    before = _target(kernel(body, buffers=[*buffers, x], params=[a, c, state, valid]))
    # These reads are separated by a write. Treating both as one premise would
    # incorrectly derive valid == 8, even though valid may be 16 at runtime.
    guarded = tirx.IfThenElse(
        state[0] == valid,
        seq(tirx.BufferStore(state, 8, [0]), tirx.IfThenElse(state[0] == 8, before["main"].body, None)),
        None,
    )
    before = tvm.IRModule({"main": before["main"].with_body(guarded)})
    with pytest.raises(ValueError, match="disagree on the actual valid row domain"):
        _lower(before)


def test_outer_mutable_snapshot_remains_a_defined_ir_variable():
    rows, width = 5, 32
    a, c = [tirx.decl_buffer((64, width), "float32", name=name) for name in ("A", "C")]
    state = tirx.decl_buffer((1,), "int32", name="state")
    valid = tirx.Var("captured_valid", "int32")
    x = tirx.decl_buffer((rows, width), "float32", name="x", scope="shared.dyn")
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    cube, buffers = _cube()
    shape = [valid, width]
    body = seq(
        cube,
        copy(a, x, src_shape=shape, dst_shape=shape),
        _parallel(tirx.BufferStore(x, x[i, j] + 1, [i, j]), (i, rows), (j, width)),
        copy(x, c, src_shape=shape, dst_shape=shape),
    )
    before = _target(kernel(body, buffers=[*buffers, x], params=[a, c, state]))
    before = tvm.IRModule({"main": before["main"].with_body(seq(tirx.Bind(valid, state[0]), before["main"].body))})
    after = _lower(before)
    assert not tirx.analysis.undefined_vars(after["main"].body, after["main"].params)
    sid = _sid(after)
    for transfer in calls(after, "tl.tileop.ascend_copy"):
        for _, _, extents in _regions(transfer):
            # A fresh analyzer placeholder has no runtime definition. Every
            # extent must use the original snapshot and the new owner only.
            assert all(var.same_as(valid) or var.same_as(sid) for var in tirx.analysis.undefined_vars(extents[0]))
            for actual, expected in ((0, (0, 0)), (2, (2, 0)), (5, (3, 2))):
                for owner in (0, 1):
                    replacements = {valid: tirx.IntImm("int32", actual), sid: tirx.IntImm("int32", owner)}
                    assert _constant(extents[0], replacements) == expected[owner]
