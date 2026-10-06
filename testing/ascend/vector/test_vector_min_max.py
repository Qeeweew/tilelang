"""Ordinary Min/Max use native V instructions with per-node BF16 conversion."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import kernel, nodes


def _program(kind, dtype, *, scalar=False, reverse=False, count=17):
    a, b, c = [tirx.decl_buffer((32,), dtype, name=name, scope="shared.dyn") for name in "abc"]
    i = tirx.Var("i", "int32")
    lhs, rhs = a[i], tirx.const(float("-inf") if kind == "max" else float("inf"), dtype) if scalar else b[i]
    if reverse:
        lhs, rhs = rhs, lhs
    expression = tirx.max(lhs, rhs) if kind == "max" else tirx.min(lhs, rhs)
    body = tirx.For(i, 0, count, tirx.ForKind.PARALLEL, tirx.BufferStore(c, expression, [i]))
    mod = kernel(body, buffers=[a, b, c])
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _native(mod):
    return [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_binary_min_max_has_exact_count_and_effects(kind, dtype):
    after = transform.AscendLowerVectorTiles()(_program(kind, dtype))
    (call,) = _native(after)
    assert call.op.name == "tl.ascend_membase." + kind
    assert call.op.get_attr("TGlobalSymbol") == "asc_" + kind
    assert int(call.args[-1]) == 17
    pointers = call.args[:-1]
    assert [int(pointer.args[1]) for pointer in pointers] == [17] * 3
    assert [int(pointer.args[2]) for pointer in pointers] == [2, 1, 1]


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_infinity_broadcast_uses_native_scalar_min_max(kind, reverse):
    after = transform.AscendLowerVectorTiles()(_program(kind, "float32", scalar=True, reverse=reverse))
    (call,) = _native(after)
    assert call.op.name == "tl.ascend_membase." + kind + "_scalar"
    assert float(call.args[2]) == (float("-inf") if kind == "max" else float("inf"))
    assert int(call.args[-1]) == 17
    assert not nodes(after, tirx.Bind)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_bfloat_min_max_widens_inputs_and_rounds_the_node_result(kind):
    after = transform.AscendLowerVectorTiles()(_program(kind, "bfloat16"))
    assert [call.op.name.rsplit(".", 1)[1] for call in _native(after)] == [
        "bfloat162float", "bfloat162float", kind, "float2bfloat16_rn"
    ]
    assert all(int(call.args[-1]) == 17 for call in _native(after))
    scratch = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.name.startswith("membase_tmp_")]
    assert all(buffer.dtype == "float32" and int(buffer.shape[0]) == 24 for buffer in scratch)


def test_min_max_expression_dag_stays_instruction_level_and_reuses_common_widening():
    before = _program("max", "bfloat16")
    root = before["main"].body.block
    a, b, c = root.alloc_buffers
    i = tirx.Var("i", "int32")
    expression = tirx.min(tirx.max(a[i], b[i]), a[i])
    body = tirx.For(i, 0, 17, tirx.ForKind.PARALLEL, tirx.BufferStore(c, expression, [i]))
    root = tirx.SBlock([], [], [], "tilelang_root", body, alloc_buffers=[a, b, c])
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    after = transform.AscendLowerVectorTiles()(before)
    names = [call.op.name.rsplit(".", 1)[1] for call in _native(after)]
    assert names.count("max") == names.count("min") == 1
    assert names.count("float2bfloat16_rn") == 2
    assert names.count("bfloat162float") == 3  # a, b, and the rounded Max result
