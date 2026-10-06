"""Empty L1 padding preserves its task and issues no native fill command."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _lower(rows, columns, arch):
    dst = tirx.decl_buffer((32, 64), "float16", name="dst", scope="shared.l1")
    region = tirx.BufferRegion(dst, [tvm.ir.Range.from_min_extent(0, rows), tvm.ir.Range.from_min_extent(0, columns)])
    body = unit(tirx.Evaluate(T.fill(region, 0)), core=2)
    params = [expr for expr in [rows, columns] if isinstance(expr, tirx.Var)]
    mod = kernel(body, buffers=[dst], params=params, annotations={"layout_map": {dst: make_ascend_major_k_layout(dst)}})
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", target)})
    with target:
        return transform.AscendLowerTileOp()(mod)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("rows, columns", [(0, 64), (32, 0)])
def test_empty_padding_keeps_task_without_native_command(rows, columns, arch):
    after = _lower(rows, columns, arch)
    assert not calls(after, "tl.ascend_fill_l1")
    assert len([task for task in nodes(after, tirx.AttrStmt) if task.attr_key == "tl.ascend_task"]) == 1


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_nonempty_constant_padding_has_no_redundant_guard(arch):
    after = _lower(13, 64, arch)
    (fill,) = calls(after, "tl.ascend_fill_l1")
    assert [int(arg) for arg in fill.args[3:6]] == [4, 13, 19]
    assert not nodes(after, tirx.IfThenElse)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_dynamic_row_padding_guard_is_inside_completion_task(arch):
    rows = tirx.Var("rows", "int32")
    after = _lower(rows, 64, arch)
    (fill,) = calls(after, "tl.ascend_fill_l1")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(fill.args[4], rows)
    (guard,) = nodes(after, tirx.IfThenElse)
    assert analyzer.can_prove_equal(guard.condition, rows > 0)
    for stmt, parents in statements(after):
        if stmt.same_as(guard):
            assert any(isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.ascend_task" for parent in parents)
        if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(fill):
            assert any(parent.same_as(guard) for parent in parents)
