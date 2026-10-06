"""Native MemBase conversion effects and rounding; no device calls."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx
from testing.ascend._ir import calls, kernel, nodes


def _program(source, destination, expression, count=17, extra=()):
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, count, tirx.ForKind.PARALLEL, tirx.BufferStore(destination, expression(i), [i]))
    buffers = [source] + ([] if destination.same_as(source) else [destination]) + list(extra)
    mod = kernel(body, buffers=buffers)
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _buffers(src_dtype, dst_dtype):
    return (
        tirx.decl_buffer((17,), src_dtype, name="src", scope="shared.dyn"),
        tirx.decl_buffer((17,), dst_dtype, name="dst", scope="shared.dyn"),
    )


def _native(mod):
    return [op for op in nodes(mod, tirx.Call) if isinstance(op.op, tvm.ir.Op) and op.op.name.startswith("tl.ascend_membase.")]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,rounding,name",
    [
        ("float16", "float32", "", "half2float"),
        ("bfloat16", "float32", "", "bfloat162float"),
        ("float32", "float16", "", "float2half_rn"),
        ("float32", "bfloat16", "rn", "float2bfloat16_rn"),
        ("float32", "float16", "rm", "float2half_rd"),
        ("float32", "bfloat16", "rp", "float2bfloat16_ru"),
        ("float32", "float16", "rz", "float2half_rz"),
    ],
)
def test_dtype_changing_cast_has_exact_element_effects(src_dtype, dst_dtype, rounding, name):
    src, dst = _buffers(src_dtype, dst_dtype)
    after = transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(src[i], dst_dtype, round=rounding)))
    (op,) = _native(after)
    assert op.op.name == "tl.ascend_membase." + name
    assert op.op.get_attr("TGlobalSymbol") == "asc_" + name
    assert int(op.args[-1]) == 17
    assert op.args[0].args[0].buffer.dtype == dst_dtype
    assert op.args[1].args[0].buffer.dtype == src_dtype
    assert int(op.args[0].args[1]) == int(op.args[1].args[1]) == 17
    assert int(op.args[0].args[2]) == 2 and int(op.args[1].args[2]) == 1


def test_nested_widening_feeds_f32_expression_and_exposes_scratch():
    src, dst = _buffers("float16", "float32")
    extra = tirx.decl_buffer((17,), "float32", name="bias", scope="shared.dyn")
    after = transform.AscendLowerVectorTiles()(
        _program(src, dst, lambda i: tirx.exp(T.cast(src[i], "float32") + extra[i]), extra=[extra])
    )
    assert [op.op.name.rsplit(".", 1)[1] for op in _native(after)] == ["half2float", "add", "exp"]
    scratch = [buf for block in nodes(after, tirx.SBlock) for buf in block.alloc_buffers if buf.name.startswith("membase_tmp_")]
    assert len(scratch) == 2
    assert all(buf.scope() == "shared.dyn" and buf.dtype == "float32" and int(buf.shape[0]) == 24 for buf in scratch)
    materialized = transform.MaterializeScheduleUnits()(after)
    tasks = [node for node in nodes(materialized, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(tasks) == 3


@pytest.mark.parametrize("src_dtype,dst_dtype", [("float16", "bfloat16"), ("bfloat16", "float16")])
def test_half_bfloat_cross_conversion_uses_explicit_f32_stage(src_dtype, dst_dtype):
    src, dst = _buffers(src_dtype, dst_dtype)
    after = transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(src[i], dst_dtype)))
    ops = _native(after)
    assert len(ops) == 2
    assert ops[0].args[0].args[0].buffer.dtype == "float32"
    assert ops[1].args[1].args[0].buffer.same_as(ops[0].args[0].args[0].buffer)


def test_bfloat_arithmetic_is_visible_promote_compute_round():
    src, dst = _buffers("bfloat16", "bfloat16")
    rhs = tirx.decl_buffer((17,), "bfloat16", name="rhs", scope="shared.dyn")
    after = transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: src[i] + rhs[i], extra=[rhs]))
    assert [op.op.name.rsplit(".", 1)[1] for op in _native(after)] == [
        "bfloat162float", "bfloat162float", "add", "float2bfloat16_rn"
    ]
    first, second, add, narrow = _native(after)
    assert first.args[1].args[0].buffer.data.same_as(src.data)
    assert second.args[1].args[0].buffer.data.same_as(rhs.data)
    assert not first.args[0].args[0].buffer.data.same_as(second.args[0].args[0].buffer.data)
    assert add.args[1].args[0].buffer.data.same_as(first.args[0].args[0].buffer.data)
    assert add.args[2].args[0].buffer.data.same_as(second.args[0].args[0].buffer.data)
    assert narrow.args[1].args[0].buffer.data.same_as(add.args[0].args[0].buffer.data)


def test_repeated_bfloat_source_shares_widening_without_merging_distinct_storage():
    src, dst = _buffers("bfloat16", "bfloat16")
    rhs = tirx.decl_buffer((17,), "bfloat16", name="rhs", scope="shared.dyn")
    expression = lambda i: tirx.max(src[i] + rhs[i], src[i])
    after = transform.AscendLowerVectorTiles()(_program(src, dst, expression, extra=[rhs]))
    widen = [op for op in _native(after) if op.op.name == "tl.ascend_membase.bfloat162float"]
    # x and y may carry different values despite equal shape/dtype. The second
    # x use shares its conversion; the rounded BF16 Add result is widened anew.
    assert len(widen) == 3
    assert sum(op.args[1].args[0].buffer.data.same_as(src.data) for op in widen) == 1
    assert sum(op.args[1].args[0].buffer.data.same_as(rhs.data) for op in widen) == 1
    maximum = calls(after, "tl.ascend_membase.max")[0]
    assert maximum.args[2].args[0].buffer.data.same_as(widen[0].args[0].args[0].buffer.data)


def test_same_dtype_cast_copies_distinct_same_shaped_storage_without_arithmetic():
    src, dst = _buffers("float32", "float32")
    after = transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: tirx.Cast("float32", src[i])))
    (copy,) = _native(after)
    assert copy.op.name == "tl.ascend_membase.copy_bits"
    assert copy.args[0].args[0].buffer.data.same_as(dst.data)
    assert copy.args[1].args[0].buffer.data.same_as(src.data)
    assert int(copy.args[-1]) == int(copy.args[0].args[1]) == int(copy.args[1].args[1]) == 17


def test_same_dtype_cast_of_the_exact_same_storage_region_is_a_no_op():
    src, _ = _buffers("float32", "float32")
    after = transform.AscendLowerVectorTiles()(_program(src, src, lambda i: tirx.Cast("float32", src[i])))
    assert not _native(after) and not nodes(after, tirx.For)


def test_same_dtype_uniform_float_cast_uses_one_scalar_read_and_duplicate():
    src, dst = _buffers("float32", "float32")
    after = transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: tirx.Cast("float32", src[0])))
    (bind,) = nodes(after, tirx.Bind)
    assert isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.data.same_as(src.data)
    (duplicate,) = _native(after)
    assert duplicate.op.name == "tl.ascend_membase.duplicate"
    assert duplicate.args[1].same_as(bind.var) and int(duplicate.args[-1]) == 17


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_uniform_integer_control_cast_is_one_typed_duplicate(dtype):
    integer = tirx.Var("integer", "int32")
    dst = tirx.decl_buffer((17,), dtype, name="dst", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 17, tirx.ForKind.PARALLEL, tirx.BufferStore(dst, T.cast(integer, dtype), [i]))
    before = kernel(body, buffers=[dst], params=[integer])
    before = tvm.IRModule({"main": before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    after = transform.AscendLowerVectorTiles()(before)
    (duplicate,) = _native(after)
    assert duplicate.op.name == "tl.ascend_membase.duplicate"
    assert duplicate.args[1].dtype == duplicate.args[0].args[0].buffer.dtype == dtype
    assert isinstance(duplicate.args[1], tirx.Cast) and duplicate.args[1].value.same_as(after["main"].params[0])
    assert not nodes(after, tirx.For)


def test_uniform_integer_cast_does_not_drop_explicit_rounding():
    src, dst = _buffers("float32", "float32")
    with pytest.raises(ValueError, match="uniform integer-to-float scalar conversion supports default rounding only"):
        transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(tirx.const(16777217, "int32"), "float32", round="rm")))


def test_integer_lane_cast_remains_unsupported():
    src, dst = _buffers("int32", "float32")
    with pytest.raises(ValueError, match="MemBase storage supports float16/bfloat16/float32"):
        transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(src[i], "float32")))


def test_stochastic_rounding_is_a_precise_diagnostic():
    src, dst = _buffers("float32", "float16")
    with pytest.raises(ValueError, match="stochastic rounding is unsupported"):
        transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(src[i], "float16", round="rs", rbits=1)))


def test_dtype_changing_same_storage_overlap_is_not_an_in_place_cast():
    src = tirx.decl_buffer((32,), "float32", name="storage", scope="shared.dyn")
    dst = tirx.decl_buffer((64,), "float16", name="half_view", data=src.data)
    with pytest.raises(ValueError, match="overlapping"):
        transform.AscendLowerVectorTiles()(_program(src, dst, lambda i: T.cast(src[i], "float16")))
