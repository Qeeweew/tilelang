"""2201 private-UB bit copies preserve typed effects, tails, and owners."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _program(dtype="float32", *, width=17, source_origin=0, destination_origin=0, style="copy", dynamic=False, alias=False):
    source = tirx.decl_buffer((64,), dtype, name="source", scope="shared.dyn")
    destination = source if alias else tirx.decl_buffer((64,), dtype, name="destination", scope="shared.dyn")
    active = tirx.Var("active", "int32")
    count = tirx.max(tirx.min(active, width), 0) if dynamic else width
    if style == "copy":
        body = copy(source, destination, src_indices=[source_origin], dst_indices=[destination_origin], src_shape=[count], dst_shape=[count])
    else:
        i = tirx.Var("i", "int32")
        body = tirx.For(i, 0, count, tirx.ForKind.PARALLEL, tirx.BufferStore(destination, source[i + source_origin], [i + destination_origin]))
    return _target(kernel(body, buffers=[source] if alias else [source, destination], params=[active] if dynamic else []))


def _native(mod):
    return [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _value(expr, mod, active):
    parameter = mod["main"].params[0]
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, {parameter: tirx.IntImm(parameter.dtype, active)})))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("style", ["copy", "parallel"])
@pytest.mark.parametrize("count", [1, 17])
def test_copy_and_plain_assignment_use_one_bit_preserving_native_task(dtype, style, count):
    after = transform.AscendLowerVectorTiles()(_program(dtype, style=style, width=count))
    (op,) = _native(after)
    assert op.op.name == "tl.ascend_membase.copy_bits"
    assert op.op.get_attr("TGlobalSymbol") == "tl_ascend_membase_copy_bits"
    destination, source = op.args[:2]
    assert destination.op.name == source.op.name == "tl.access_ptr"
    assert destination.args[0].buffer.dtype == source.args[0].buffer.dtype == dtype
    assert destination.args[0].buffer.name == "destination" and source.args[0].buffer.name == "source"
    assert int(destination.args[1]) == int(source.args[1]) == int(op.args[-1]) == count
    assert int(destination.args[2]) == 2 and int(source.args[2]) == 1
    assert not nodes(after, tirx.For)
    assert not any(call.op.name in ("tl.ascend_membase.add", "tl.ascend_membase.mul") for call in _native(after))
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len([attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]) == 1


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_dynamic_tail_and_zero_keep_exact_typed_effects_and_completion(dtype):
    after = transform.AscendLowerVectorTiles()(_program(dtype, dynamic=True))
    (op,) = calls(after, "tl.ascend_membase.copy_bits")
    for active in (-1, 0, 1, 16, 17, 19):
        expected = min(max(active, 0), 17)
        assert _value(op.args[-1], after, active) == expected
        assert [_value(pointer.args[1], after, active) for pointer in op.args[:2]] == [expected, expected]
    assert not nodes(after, tirx.IfThenElse)
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len(calls(materialized, "tl.ascend_membase.copy_bits")) == 1
    assert len([attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]) == 1


@pytest.mark.parametrize("style", ["copy", "parallel"])
def test_exact_same_region_is_eliminated_but_shifted_overlap_is_rejected(style):
    same = transform.AscendLowerVectorTiles()(_program(alias=True, style=style))
    assert not _native(same) and not nodes(same, tirx.For)
    # Both origins are aligned; this is an alias contract failure, not an
    # alignment failure. A partial copy cannot depend on vor's issue order.
    with pytest.raises(ValueError, match="shifted overlapping"):
        transform.AscendLowerVectorTiles()(_program(alias=True, style=style, destination_origin=8))
    separate = transform.AscendLowerVectorTiles()(_program(alias=True, style=style, width=8, destination_origin=16))
    (op,) = calls(separate, "tl.ascend_membase.copy_bits")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(op.args[0].args[0].indices[0], 16)
    assert analyzer.can_prove_equal(op.args[1].args[0].indices[0], 0)


@pytest.mark.parametrize("style", ["copy", "parallel"])
def test_unaligned_origin_does_not_fall_back_to_scalar_elements(style):
    with pytest.raises(ValueError, match="32-byte aligned"):
        transform.AscendLowerVectorTiles()(_program(style=style, source_origin=1))


def test_copy_regions_must_match_and_fit_their_private_capacity():
    source = tirx.decl_buffer((64,), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((64,), "float32", name="destination", scope="shared.dyn")
    mismatch = copy(source, destination, src_shape=[17], dst_shape=[16])
    with pytest.raises(ValueError, match="same region domain"):
        transform.AscendLowerVectorTiles()(_target(kernel(mismatch, buffers=[source, destination])))
    with pytest.raises(ValueError, match="within their private allocations"):
        transform.AscendLowerVectorTiles()(_program(width=65))


def test_copy_does_not_hide_dtype_conversion_integer_lanes_or_dma_hints():
    source = tirx.decl_buffer((32,), "float16", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((32,), "float32", name="destination", scope="shared.dyn")
    with pytest.raises(ValueError, match="matching element types"):
        transform.AscendLowerVectorTiles()(_target(kernel(copy(source, destination), buffers=[source, destination])))
    with pytest.raises(ValueError, match="MemBase storage supports float16/bfloat16/float32"):
        transform.AscendLowerVectorTiles()(_program("int32"))
    with pytest.raises(ValueError, match="does not accept DMA or parallel-loop layout annotations"):
        transform.AscendLowerVectorTiles()(_target(kernel(copy(destination, destination, annotations={"transpose": tirx.IntImm("int32", 1)}), buffers=[destination])))


def test_noncontiguous_physical_layout_is_rejected_once_at_the_vector_boundary():
    source = tirx.decl_buffer((2, 32), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((2, 32), "float32", name="destination", scope="shared.dyn")
    mod = _target(kernel(copy(source, destination), buffers=[source, destination], annotations={"layout_map": {source: Layout((2, 32), lambda i, j: (j, i))}}))
    with pytest.raises(ValueError, match="strided/gather"):
        transform.AscendLowerVectorTiles()(mod)


def test_plain_ub_broadcast_is_still_one_scalar_read_and_duplicate():
    source = tirx.decl_buffer((1,), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((32,), "float32", name="destination", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 17, tirx.ForKind.PARALLEL, tirx.BufferStore(destination, source[0], [i]))
    after = transform.AscendLowerVectorTiles()(_target(kernel(body, buffers=[source, destination])))
    (bind,) = nodes(after, tirx.Bind)
    assert isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.name == "source"
    (duplicate,) = _native(after)
    assert duplicate.op.name == "tl.ascend_membase.duplicate" and int(duplicate.args[-1]) == 17


def test_common_copy_spelling_and_singleton_axis_use_the_same_region_contract():
    source = tirx.decl_buffer((1, 17), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((17,), "float32", name="destination", scope="shared.dyn")
    original = copy(source, destination).value
    common = tirx.Evaluate(tirx.Call(original.dtype, tvm.ir.Op.get("tl.tileop.copy"), original.args))
    after = transform.AscendLowerVectorTiles()(_target(kernel(common, buffers=[source, destination])))
    (op,) = calls(after, "tl.ascend_membase.copy_bits")
    assert int(op.args[-1]) == 17
    assert len(op.args[1].args[0].indices) == 2 and len(op.args[0].args[0].indices) == 1
    assert not nodes(after, tirx.For)


def test_two_aiv_private_copy_uses_only_the_established_valid_owner_rows():
    source = tirx.decl_buffer((3, 32), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((3, 32), "float32", name="destination", scope="shared.dyn")
    cube_gm = tirx.decl_buffer((16, 16), "float16", name="cube_gm")
    cube_l1 = tirx.decl_buffer((16, 16), "float16", name="cube_l1", scope="shared.l1")
    sid = tirx.Var("sid", "int32")
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    valid_rows = tirx.max(tirx.min(3, 2 - sid * 3), 0)
    owned_copy = copy(source, destination, src_shape=[valid_rows, 32], dst_shape=[valid_rows, 32])
    body = seq(copy(cube_gm, cube_l1), tirx.AttrStmt(thread, "thread_extent", 2, owned_copy))
    mod = _target(kernel(body, buffers=[source, destination, cube_l1], params=[cube_gm], annotations={"tl.ascend_row_partitioned": 2}))
    after = transform.AscendLowerVectorTiles()(mod)
    (op,) = calls(after, "tl.ascend_membase.copy_bits")
    actual_sid = next(attr.node.var for attr in nodes(after, tirx.AttrStmt) if attr.attr_key == "thread_extent")
    for owner, expected in [(0, 64), (1, 0)]:
        substitutions = {actual_sid: tirx.IntImm("int32", owner)}
        analyzer = tvm.arith.Analyzer()
        assert int(analyzer.simplify(tirx.stmt_functor.substitute(op.args[-1], substitutions))) == expected
        assert [int(analyzer.simplify(tirx.stmt_functor.substitute(pointer.args[1], substitutions))) for pointer in op.args[:2]] == [expected, expected]
    assert not nodes(after, tirx.For) and not nodes(after, tirx.IfThenElse)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("style", ["copy", "parallel"])
def test_frontend_bit_copy_emits_native_helper_without_floating_arithmetic(dtype, style):
    @T.prim_func
    def main(A: T.Tensor((17,), dtype), C: T.Tensor((17,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((17,), dtype)
            c = T.alloc_shared((17,), dtype)
            T.copy(A, a)
            if style == "copy":
                T.copy(a, c)
            else:
                for i in T.Parallel(17):
                    c[i] = a[i]
            T.copy(c, C)

    source = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}).kernel_source
    assert "tl_ascend_membase_copy_bits(" in source
    assert "asc_add(" not in source and "asc_mul(" not in source


def test_3510_copy_is_preserved_for_its_existing_lowering():
    before = _target(_program(), "dav-3510")
    after = transform.AscendLowerVectorTiles()(before)
    tvm.ir.assert_structural_equal(after, before)
