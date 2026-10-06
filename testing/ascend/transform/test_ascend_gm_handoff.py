"""Mixed GM bridges expose bounded physical transfers before scheduling."""

import itertools

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import make_ascend_nz_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, nodes, seq, statements


def _buffer(shape=(32, 48), dtype="float32", scope="shared.l0c", name="src"):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope)


def _program(body, buffers, *, grid=(1,), arch="dav-2201", params=(), layouts=None):
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=buffers, annotations={"layout_map": layouts} if layouts else {})
    body = tirx.SBlockRealize([], True, root)
    axes = []
    for axis, extent in enumerate(grid):
        index = tirx.Var("block_" + "xyz"[axis], "int32")
        axes.append(index)
        iv = tirx.IterVar(tvm.ir.Range.from_min_extent(0, extent), index, tirx.IterVar.ThreadIndex, "blockIdx." + "xyz"[axis])
        body = tirx.AttrStmt(iv, "thread_extent", extent, body)
    func = tirx.PrimFunc(list(params), body).with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return tvm.IRModule({"main": func}), axes


def _lower(src, dst, *, grid=(1,), arch="dav-2201", params=(), layouts=None, **kwargs):
    before, axes = _program(copy(src, dst, **kwargs), [src, dst], grid=grid, arch=arch, params=params, layouts=layouts)
    return transform.AscendLowerGMHandoff()(before), axes


def _region(expression):
    assert expression.op.name == "tl.region"
    return expression.args[0].buffer, expression.args[0].indices, expression.args[2:]


def _constant(expression, mapping=None):
    if mapping:
        expression = tirx.stmt_functor.substitute(expression, mapping)
    return int(tvm.arith.Analyzer().simplify(expression))


def _workspace(module):
    return [buffer for block in nodes(module, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("gm_handoff_")]


def _physical_cores(module):
    result = []
    for statement, parents in statements(module):
        if isinstance(statement, tirx.Evaluate) and isinstance(statement.value, tirx.Call) and statement.value.op.name == "tl.tileop.ascend_copy":
            result.append(next(parent.name_hint for parent in reversed(parents) if isinstance(parent, tirx.SBlock) and parent.name_hint in ("CUBE", "VECTOR")))
    return result


def test_core_groups_have_distinct_canonical_workspace_matrices():
    src = _buffer((17, 33))
    dst = _buffer((17, 33), scope="shared", name="dst")
    after, axes = _lower(src, dst, grid=(3, 2, 4))
    (workspace,) = _workspace(after)
    assert workspace.scope() == "global"
    assert workspace.dtype == "float32"
    assert tuple(int(extent) for extent in workspace.shape) == (24, 32, 48)
    export, load = calls(after, "tl.tileop.ascend_copy")
    export_buffer, export_origin, export_shape = _region(export.args[1])
    load_buffer, load_origin, load_shape = _region(load.args[0])
    assert export_buffer.same_as(workspace) and load_buffer.same_as(workspace)
    tvm.ir.assert_structural_equal(export_origin, load_origin)
    tvm.ir.assert_structural_equal(export_shape, load_shape)
    assert tuple(int(value) for value in export_shape) == (1, 17, 33)
    assert _physical_cores(after) == ["CUBE", "VECTOR"]
    groups = set()
    for x, y, z in itertools.product(range(3), range(2), range(4)):
        actual = _constant(export_origin[0], dict(zip(axes, [tirx.IntImm("int32", index) for index in (x, y, z)])))
        assert actual == (z * 2 + y) * 3 + x
        groups.add(actual)
    assert groups == set(range(24))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ub_l1_preserves_dtype_and_ordinary_copy_contract(dtype):
    src = _buffer(dtype=dtype, scope="shared")
    dst = _buffer(dtype=dtype, scope="shared.l1", name="dst")
    after, _ = _lower(src, dst, annotations={"nd2nz": tirx.IntImm("int32", 1)})
    (workspace,) = _workspace(after)
    assert workspace.dtype == dtype
    assert _physical_cores(after) == ["VECTOR", "CUBE"]
    assert not calls(after, "tl.tileop.fill")
    for transfer in calls(after, "tl.tileop.ascend_copy"):
        assert "nd2nz" not in transfer.annotations


def test_original_source_bounds_define_tail_despite_padded_gm_capacity():
    src = _buffer((13, 15), "float16", "shared")
    dst = _buffer((17, 19), "float16", "shared.l1", "dst")
    after, _ = _lower(src, dst, src_shape=(17, 19), dst_shape=(17, 19), layouts={dst: make_ascend_nz_layout(dst)})
    (workspace,) = _workspace(after)
    assert tuple(int(extent) for extent in workspace.shape) == (1, 32, 32)
    export, load = calls(after, "tl.tileop.ascend_copy")
    for expression in (export.args[0], export.args[1], load.args[0], load.args[1]):
        _, _, extents = _region(expression)
        assert tuple(int(value) for value in extents[-2:]) == (13, 15)
    (fill,) = calls(after, "tl.tileop.fill")
    fill_buffer, origin, extents = _region(fill.args[0])
    assert fill_buffer.same_as(dst)
    assert tuple(int(value) for value in origin) == (0, 0)
    assert tuple(int(value) for value in extents) == (17, 19)
    assert int(fill.annotations["tl.ascend.initialize_l1_padding"]) == 1
    assert float(fill.args[1]) == 0
    # The mixed-intent classifier constructs Fill before storage normalization.
    # No intermediate semantic region may exceed the current logical buffer.
    fill_only, _ = _program(tirx.Evaluate(fill), [dst], layouts={dst: make_ascend_nz_layout(dst)})
    transform.AscendLowerMixedPartition()(fill_only)
    normalized = transform.NormalizeAscendFractalStorage()(after)
    (fill,) = calls(normalized, "tl.tileop.fill")
    buffer, _, extents = _region(fill.args[0])
    assert tuple(int(value) for value in buffer.shape) == (32, 32)
    assert tuple(int(value) for value in extents) == (32, 32)
    assert "tl.ascend.initialize_l1_padding" not in fill.annotations
    for transfer in calls(normalized, "tl.tileop.ascend_copy"):
        assert tuple(int(value) for value in _region(transfer.args[0])[2][-2:]) == (13, 15)
    tvm.ir.assert_structural_equal(normalized, transform.NormalizeAscendFractalStorage()(normalized))


def test_physical_padding_retains_empty_leading_domain():
    index = tirx.Var("matrix", "int32")
    src = _buffer((2, 13, 15), "float16", "shared")
    dst = _buffer((17, 19), "float16", "shared.l1", "dst")
    logical, _ = _lower(src, dst, src_indices=(index, 0, 0), src_shape=(1, 17, 19), params=[index], layouts={dst: make_ascend_nz_layout(dst)})
    physical = transform.NormalizeAscendFractalStorage()(logical)
    (logical_fill,) = calls(logical, "tl.tileop.fill")
    (physical_fill,) = calls(physical, "tl.tileop.fill")
    for position in (-1, 0, 1, 2):
        substitution = {index: tirx.IntImm("int32", position)}
        active = int(0 <= position < 2)
        assert _constant(_region(logical_fill.args[0])[2][0], substitution) == 17 * active
        assert _constant(_region(physical_fill.args[0])[2][0], substitution) == 32 * active
    assert not nodes(physical, tirx.IfThenElse)


def test_partial_handoff_initialization_does_not_expand_to_the_allocation():
    src = _buffer((13, 15), "float16", "shared")
    dst = _buffer((65, 80), "float16", "shared.l1", "dst")
    logical, _ = _lower(src, dst, src_shape=(32, 32), dst_indices=(16, 32), dst_shape=(32, 32), layouts={dst: make_ascend_nz_layout(dst)})
    physical = transform.NormalizeAscendFractalStorage()(logical)
    (fill,) = calls(physical, "tl.tileop.fill")
    _, origin, extents = _region(fill.args[0])
    assert tuple(int(value) for value in origin) == (16, 32)
    assert tuple(int(value) for value in extents) == (32, 32)


@pytest.mark.parametrize("direction", ["l0c_to_ub", "ub_to_l1"])
def test_explicit_valid_source_region_connects_to_full_destination_capacity(direction):
    dtype = "float32" if direction == "l0c_to_ub" else "float16"
    src_scope, dst_scope = ("shared.l0c", "shared") if direction == "l0c_to_ub" else ("shared", "shared.l1")
    src = _buffer((32, 64), dtype, src_scope)
    dst = _buffer((32, 64), dtype, dst_scope, "dst")
    after, _ = _lower(src, dst, src_shape=(3, 64))
    for transfer in calls(after, "tl.tileop.ascend_copy"):
        for expression in transfer.args:
            assert tuple(int(value) for value in _region(expression)[2][-2:]) == (3, 64)
    if direction == "ub_to_l1":
        (fill,) = calls(after, "tl.tileop.fill")
        assert tuple(int(value) for value in _region(fill.args[0])[2]) == (32, 64)
    else:
        assert not calls(after, "tl.tileop.fill")


def test_nonzero_origins_and_partial_destination_padding_are_exact():
    src = _buffer((20, 32), "float16", "shared")
    dst = _buffer((64, 80), "float16", "shared.l1", "dst")
    after, _ = _lower(src, dst, src_indices=(3, 2), dst_indices=(16, 32), src_shape=(32, 32), dst_shape=(32, 32))
    export, load = calls(after, "tl.tileop.ascend_copy")
    _, origin, extents = _region(export.args[0])
    assert tuple(int(value) for value in origin) == (3, 2)
    assert tuple(int(value) for value in extents) == (17, 30)
    _, origin, extents = _region(load.args[1])
    assert tuple(int(value) for value in origin) == (16, 32)
    assert tuple(int(value) for value in extents) == (17, 30)
    (fill,) = calls(after, "tl.tileop.fill")
    _, origin, extents = _region(fill.args[0])
    assert tuple(int(value) for value in origin) == (16, 32)
    assert tuple(int(value) for value in extents) == (32, 32)


def test_destination_only_tail_does_not_initialize_unrelated_l1_rows():
    src = _buffer((32, 32), "float16", "shared")
    dst = _buffer((32, 32), "float16", "shared.l1", "dst")
    after, _ = _lower(src, dst, dst_indices=(16, 16))
    assert not calls(after, "tl.tileop.fill")
    for transfer in calls(after, "tl.tileop.ascend_copy"):
        for expression in transfer.args:
            assert tuple(int(value) for value in _region(expression)[2][-2:]) == (16, 16)


@pytest.mark.parametrize("direction", ["l0c_to_ub", "ub_to_l1"])
def test_leading_singleton_validity_is_shared_by_both_physical_tasks(direction):
    index = tirx.Var("matrix_index", "int32")
    dtype = "float32" if direction == "l0c_to_ub" else "float16"
    src_scope, dst_scope = ("shared.l0c", "shared") if direction == "l0c_to_ub" else ("shared", "shared.l1")
    src = _buffer((2, 16, 32), dtype, src_scope)
    dst = _buffer((16, 32), dtype, dst_scope, "dst")
    after, _ = _lower(src, dst, src_indices=(index, 0, 0), src_shape=(1, 16, 32), params=[index])
    assert not calls(after, "tl.tileop.fill")
    export, load = calls(after, "tl.tileop.ascend_copy")
    export_valid = _region(export.args[1])[2][0]
    load_valid = _region(load.args[0])[2][0]
    tvm.ir.assert_structural_equal(export_valid, load_valid)
    for value in (-1, 0, 1, 2):
        assert _constant(export_valid, {index: tirx.IntImm("int32", value)}) == int(0 <= value < 2)
    assert not nodes(after, tirx.IfThenElse)
    assert len(calls(after, "tl.tileop.ascend_copy")) == 2


def test_invalid_destination_singleton_also_skips_export():
    src = _buffer((16, 32), "float16", "shared")
    dst = _buffer((1, 16, 32), "float16", "shared.l1", "dst")
    after, _ = _lower(src, dst, dst_indices=(1, 0, 0), dst_shape=(1, 16, 32))
    assert not calls(after, "tl.tileop.fill")
    export, load = calls(after, "tl.tileop.ascend_copy")
    assert int(_region(export.args[1])[2][0]) == int(_region(load.args[0])[2][0]) == 0


@pytest.mark.parametrize("shape", [(1, 32), (16, 1), (1, 1)])
def test_one_row_or_column_remains_a_matrix_axis(shape):
    src = _buffer(shape)
    dst = _buffer(shape, scope="shared", name="dst")
    after, _ = _lower(src, dst)
    for transfer in calls(after, "tl.tileop.ascend_copy"):
        assert tuple(int(value) for value in _region(transfer.args[1])[2][-2:]) == shape


def test_distributing_vector_scope_preserves_one_epoch_loop_and_realize_predicate():
    src = _buffer((16, 32))
    dst = _buffer((16, 32), scope="shared", name="dst")
    row, column, epoch = (tirx.Var(name, "int32") for name in ("row", "column", "epoch"))
    expression = tirx.For(row, 0, 16, tirx.ForKind.PARALLEL, tirx.For(column, 0, 32, tirx.ForKind.PARALLEL, tirx.BufferStore(dst, dst[row, column] + 1, [row, column])))
    loop = tirx.For(epoch, 0, 37, tirx.ForKind.SERIAL, seq(copy(src, dst), expression), annotations={"num_stages": 2})
    predicate = tirx.Var("active", "bool")
    core = tirx.SBlock([], [], [], "VECTOR", loop, annotations={"vector_count": 2})
    before, _ = _program(tirx.SBlockRealize([], predicate, core), [src, dst], params=[predicate])
    after = transform.AscendLowerGMHandoff()(before)
    assert len(nodes(after, tirx.For)) == 3
    (epoch_loop,) = [loop for loop in nodes(after, tirx.For) if loop.kind == tirx.ForKind.SERIAL]
    assert int(epoch_loop.extent) == 37 and int(epoch_loop.annotations["num_stages"]) == 2
    (realize,) = [item for item in nodes(after, tirx.SBlockRealize) if item.block.name_hint == "ascend_core_region"]
    assert realize.predicate.same_as(predicate)
    assert len(calls(after, "tl.tileop.ascend_copy")) == 2
    assert _physical_cores(after) == ["CUBE", "VECTOR"]
    (vector_expression,) = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "VECTOR" and isinstance(block.body, tirx.For)]
    assert int(vector_expression.annotations["vector_count"]) == 2
    assert len(nodes(vector_expression.body, tirx.For)) == 2


def test_two_channels_are_distinct_and_pass_is_idempotent():
    src = _buffer((16, 32))
    ub = _buffer((16, 32), scope="shared", name="ub")
    half = _buffer((16, 32), "float16", "shared", "half")
    l1 = _buffer((16, 32), "float16", "shared.l1", "l1")
    before, _ = _program(seq(copy(src, ub), copy(half, l1)), [src, ub, half, l1])
    after = transform.AscendLowerGMHandoff()(before)
    a, b = _workspace(after)
    assert not a.data.same_as(b.data)
    assert a.dtype == "float32" and b.dtype == "float16"
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    storages = root.annotations["tl.ascend.compiler_workspace_storages"]
    assert len(storages) == 2
    assert all(any(storage.same_as(buffer.data) for storage in storages) for buffer in (a, b))
    assert _physical_cores(after) == ["CUBE", "VECTOR", "VECTOR", "CUBE"]
    tvm.ir.assert_structural_equal(after, transform.AscendLowerGMHandoff()(after))


def test_workspace_provenance_uses_created_data_vars_not_global_scope_or_names():
    src = _buffer((16, 32))
    dst = _buffer((16, 32), scope="shared", name="dst")
    # A user allocation can even share the generated workspace spelling.
    user = _buffer((1, 16, 32), scope="global", name="gm_handoff_0")
    before, _ = _program(copy(src, dst), [src, dst, user])
    after = transform.AscendLowerGMHandoff()(before)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    storages = root.annotations["tl.ascend.compiler_workspace_storages"]
    workspace = _region(calls(after, "tl.tileop.ascend_copy")[0].args[1])[0]
    assert len(storages) == 1 and storages[0].same_as(workspace.data)
    assert not any(storage.same_as(user.data) for storage in storages)
    tvm.ir.assert_structural_equal(after, transform.AscendLowerGMHandoff()(after))


@pytest.mark.parametrize("grid, message", [((0,), "static positive block grid"), ((65536, 65536), "uint32 launch capacity")])
def test_invalid_grid_is_rejected_before_workspace_product(grid, message):
    src, dst = _buffer(), _buffer(scope="shared", name="dst")
    with pytest.raises(ValueError, match=message):
        _lower(src, dst, grid=grid)


def test_aligned_capacity_and_byte_product_overflow_are_diagnosed():
    src = _buffer((2**31 - 1, 16))
    dst = _buffer((2**31 - 1, 16), scope="shared", name="dst")
    with pytest.raises(ValueError, match="int32 aligned geometry"):
        _lower(src, dst)
    src = _buffer((2**30, 2**30))
    dst = _buffer((2**30, 2**30), scope="shared", name="dst")
    with pytest.raises(ValueError, match="byte size overflows int64"):
        _lower(src, dst, grid=(3,))


def test_negative_matrix_origin_and_partial_c0_tail_have_explicit_diagnostics():
    src = _buffer((16, 32), "float16", "shared")
    dst = _buffer((64, 64), "float16", "shared.l1", "dst")
    with pytest.raises(ValueError, match="nonnegative matrix origins"):
        _lower(src, dst, src_indices=(-1, 0), src_shape=(16, 32), dst_shape=(16, 32))
    with pytest.raises(ValueError, match="C0-aligned column footprint"):
        _lower(src, dst, src_shape=(32, 33), dst_shape=(32, 33))


def test_conversion_requires_an_ordinary_ub_expression():
    src = _buffer()
    dst = _buffer(dtype="float16", scope="shared", name="dst")
    with pytest.raises(ValueError, match="preserves dtype"):
        _lower(src, dst)


def test_dav3510_logical_route_is_unchanged():
    src, dst = _buffer(), _buffer(scope="shared", name="dst")
    before, _ = _program(copy(src, dst), [src, dst], arch="dav-3510")
    tvm.ir.assert_structural_equal(before, transform.AscendLowerGMHandoff()(before))
