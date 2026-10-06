"""Bounded runtime vector counts retain precise effects and static UB capacity."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq


def _native(mod):
    return [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _program(*, src_dtype="float32", dst_dtype="float32", expression=None, extent=None, scalar=False, mixed=False, marker=False):
    n = tirx.Var("active", "int32")
    extent = tirx.max(tirx.min(n, 17), 0) if extent is None else extent(n)
    src = tirx.decl_buffer((32,), src_dtype, name="src", scope="shared.dyn")
    dst = tirx.decl_buffer((32,), dst_dtype, name="dst", scope="shared.dyn")
    bias = tirx.decl_buffer((1,), src_dtype, name="bias", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    value = expression(src, i) if expression else src[i] + (bias[0] if scalar else tirx.const(1, src_dtype))
    body = tirx.For(i, 0, extent, tirx.ForKind.PARALLEL, tirx.BufferStore(dst, value, [i]))
    buffers = [src, dst] + ([bias] if scalar else [])
    params = [n]
    if mixed:
        gm = tirx.decl_buffer((16, 16), "float16", name="cube_gm")
        l1 = tirx.decl_buffer((16, 16), "float16", name="cube_l1", scope="shared.l1")
        body = seq(copy(gm, l1), body)
        params += [gm]
        buffers += [l1]
    mod = kernel(body, buffers=buffers, params=params, annotations={"tl.ascend_row_partitioned": 2} if marker else {})
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _value(expr, mod, active):
    var = next(param for param in mod["main"].params if param.name == "active")
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, {var: tirx.IntImm(var.dtype, active)})))


def test_active_tail_count_is_shared_by_instruction_and_all_memory_effects():
    after = transform.AscendLowerVectorTiles()(_program(expression=lambda src, i: tirx.exp(src[i] + 1)))
    assert [call.op.name.rsplit(".", 1)[1] for call in _native(after)] == ["add_scalar", "exp"]
    for active in (-2, 0, 1, 16, 17, 20):
        expected = min(max(active, 0), 17)
        for call in _native(after):
            assert _value(call.args[-1], after, active) == expected
            for pointer in call.args[:-1]:
                if isinstance(pointer, tirx.Call) and pointer.op.name == "tl.access_ptr":
                    assert _value(pointer.args[1], after, active) == expected
    (scratch,) = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("membase_tmp_")]
    assert int(scratch.shape[0]) == 24 and scratch.scope() == "shared.dyn"
    assert not nodes(after, tirx.For)
    assert not nodes(after, tirx.IfThenElse)


def test_zero_count_keeps_native_task_completion_sites():
    after = transform.AscendLowerVectorTiles()(_program(expression=lambda src, i: tirx.exp(src[i] + 1)))
    materialized = transform.MaterializeScheduleUnits()(after)
    tasks = [attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]
    assert len(tasks) == 2
    assert len(_native(materialized)) == 2
    assert all(_value(call.args[-1], materialized, 0) == 0 for call in _native(materialized))
    assert not nodes(materialized, tirx.IfThenElse)


def test_inactive_scalar_broadcast_has_a_short_circuit_scalar_value():
    after = transform.AscendLowerVectorTiles()(_program(scalar=True))
    (bind,) = nodes(after, tirx.Bind)
    assert isinstance(bind.value, tirx.Select)
    assert isinstance(bind.value.true_value, tirx.BufferLoad)
    assert bind.value.true_value.buffer.name == "bias"
    assert _value(bind.value.condition, after, 0) == 0
    assert isinstance(bind.value.false_value, tirx.FloatImm)
    assert float(bind.value.false_value) == 0.0
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len([attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]) == 2
    assert not nodes(materialized, tirx.IfThenElse)


def test_positive_scalar_broadcast_preserves_a_direct_scalar_load():
    after = transform.AscendLowerVectorTiles()(_program(scalar=True, extent=lambda n: tirx.max(tirx.min(n, 17), 1)))
    (bind,) = nodes(after, tirx.Bind)
    assert isinstance(bind.value, tirx.BufferLoad)


@pytest.mark.parametrize("source,destination,name", [("float16", "float32", "half2float"), ("float32", "bfloat16", "float2bfloat16_rn")])
def test_dtype_changing_dynamic_cast_retains_logical_element_count(source, destination, name):
    after = transform.AscendLowerVectorTiles()(
        _program(src_dtype=source, dst_dtype=destination, expression=lambda src, i: T.cast(src[i], destination))
    )
    (call,) = _native(after)
    assert call.op.name == "tl.ascend_membase." + name
    assert call.args[0].args[0].buffer.dtype == destination
    assert call.args[1].args[0].buffer.dtype == source
    for pointer in call.args[:2]:
        assert _value(pointer.args[1], after, 17) == 17
        assert _value(pointer.args[1], after, 0) == 0


def test_bfloat_math_scratch_uses_the_bound_while_each_task_uses_the_actual_count():
    after = transform.AscendLowerVectorTiles()(_program(src_dtype="bfloat16", dst_dtype="bfloat16", expression=lambda src, i: src[i] + src[i]))
    assert [call.op.name.rsplit(".", 1)[1] for call in _native(after)] == ["bfloat162float", "add", "float2bfloat16_rn"]
    assert all(_value(call.args[-1], after, 0) == 0 and _value(call.args[-1], after, 3) == 3 for call in _native(after))
    scratch = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("membase_tmp_")]
    assert scratch and all(buffer.dtype == "float32" and int(buffer.shape[0]) == 24 for buffer in scratch)


def test_statically_empty_parallel_does_not_allocate_or_issue_math():
    after = transform.AscendLowerVectorTiles()(_program(extent=lambda n: tirx.const(0, "int32")))
    assert not _native(after) and not nodes(after, tirx.For)
    assert not any(buffer.name.startswith("membase_tmp_") for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers)


@pytest.mark.parametrize("extent", [lambda n: n, lambda n: tirx.min(n, 17)])
def test_unbounded_or_negative_domain_is_diagnosed_once(extent):
    with pytest.raises(ValueError, match="nonnegative bounded private UB domain"):
        transform.AscendLowerVectorTiles()(_program(extent=extent))


def test_native_cube_transfer_requires_partition_marker_even_without_gemm():
    with pytest.raises(ValueError, match="two-AIV partition"):
        transform.AscendLowerVectorTiles()(_program(mixed=True))
    after = transform.AscendLowerVectorTiles()(_program(mixed=True, marker=True))
    assert len(_native(after)) == 1
    assert len(calls(after, "tl.tileop.ascend_copy")) == 1
