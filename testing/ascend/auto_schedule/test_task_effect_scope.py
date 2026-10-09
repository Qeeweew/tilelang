"""Owned task effects preserve control inputs without absorbing child bodies.

The observations use public core/sync passes and canonical scheduled TIR.
These tests do not expose a task-analysis FFI or run a device.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _COPY, _GEMM, _fixture, _insert,
)


def _roundtrip(mod):
    return tvm.ir.load_json(tvm.ir.save_json(mod))


def _bind_core(mod, name):
    task, = [task for task in nodes(mod, tirx.AttrStmt)
             if task.attr_key == "tl.ascend_task" and isinstance(task.body, tirx.Bind)
             and task.body.var.name == name]
    return int(task.node["core_mask"])


@pytest.mark.parametrize("owner,component", [
    ("control", "min"), ("control", "extent"), ("control", "step"),
    ("control", "condition"), ("control", "assumption"),
    ("leaf", "condition"), ("leaf", "assumption"), ("leaf", "body-loop"),
])
def test_owned_core_local_reads_restrict_the_actual_execution_scope(owner, component):
    pred = tirx.decl_buffer((1,), "int32", name="pred", scope="shared.dyn")
    source = tirx.decl_buffer((16, 16), "float32", name="source")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    i = tirx.Var("i", "int32")
    body = unit(copy(source, l1), core=2)
    output = None
    if owner == "leaf":
        if component == "condition":
            body = unit(copy(source, l1), core=2, guard=pred[0] > 0)
        elif component == "assumption":
            body = unit(tirx.AttrStmt(pred[0] > 0, "tl.assume", 1, body.body), core=None)
        else:
            # This For belongs to one atomic semantic Task, rather than to a
            # ControlNode header. Its body read must remain a leaf effect.
            output = tirx.decl_buffer((2,), "int32", name="output")
            inner = tirx.Var("inner", "int32")
            body = unit(tirx.For(inner, 0, 2, tirx.ForKind.SERIAL,
                                 tirx.BufferStore(output, pred[0], [inner])), core=2)
    loop = tirx.For(
        i, pred[0] if component == "min" else 0,
        pred[0] if component == "extent" else 2, tirx.ForKind.SERIAL, body,
        step=pred[0] if component == "step" else None,
    )
    if owner == "control" and component == "assumption":
        loop = tirx.AttrStmt(pred[0] > 0, "tl.assume", 1, loop)
    before = kernel(unit(loop, core=None,
                         guard=pred[0] > 0 if owner == "control" and component == "condition" else None),
                    buffers=[pred, l1], params=[source] + ([output] if output is not None else []))
    diagnostic = ("Cannot evaluate a loop bound or control guard" if owner == "control"
                  else "unavailable on every legal execution core")
    for candidate in (before, _roundtrip(before)):
        original = tvm.ir.save_json(candidate)
        with pytest.raises(tvm.error.InternalError, match=diagnostic):
            transform.ResolveCore()(candidate)
        assert tvm.ir.save_json(candidate) == original


def test_header_and_direct_guards_demand_broadcast_without_absorbing_child_memory():
    source = tirx.decl_buffer((16, 16), "float32", name="source")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    minimum, extent, stride = [tirx.Var(name, "int32") for name in ("minimum", "extent", "stride")]
    active, assumed = [tirx.Var(name, "bool") for name in ("active", "assumed")]
    ignored = tirx.Var("annotation_value", "int32")
    definitions = [(minimum, 0), (extent, 2), (stride, 1), (active, True), (assumed, True), (ignored, 7)]
    i = tirx.Var("i", "int32")
    body = seq(unit(copy(source, l1), core=2), unit(copy(source, ub), core=1))
    loop = tirx.For(i, minimum, extent, tirx.ForKind.SERIAL, body, step=stride)
    # An attribute's PrimExpr node is its guard effect. Its value is retained
    # metadata, and must not create an extra broadcast requirement.
    loop = tirx.AttrStmt(assumed, "tl.assume", ignored, loop)
    before = kernel(seq(*(unit(tirx.Bind(var, tirx.const(value, var.dtype)), core=3)
                          for var, value in definitions),
                        unit(loop, core=None, guard=active)),
                    buffers=[l1, ub], params=[source])
    results = []
    for candidate in (before, _roundtrip(before)):
        original = tvm.ir.save_json(candidate)
        after = transform.ResolveCore()(candidate)
        assert tvm.ir.save_json(candidate) == original
        assert all(_bind_core(after, var.name) == 3 for var in (minimum, extent, stride, active, assumed))
        assert _bind_core(after, ignored.name) == 2  # Existing Cube-preferred fallback for an unused scalar.
        copies = [task for task in nodes(after, tirx.AttrStmt)
                  if task.attr_key == "tl.ascend_task" and calls(task.body, _COPY)]
        assert len(copies) == 2 and {int(task.node["core_mask"]) for task in copies} == {1, 2}
        results.append(after)
    tvm.ir.assert_structural_equal(*results, map_free_vars=True)


@pytest.mark.parametrize("native_k", [1, 3], ids=["single-MAD", "native-K-chain"])
def test_migrated_activation_roundtrips_through_fresh_core_resolution(native_k):
    before, facts = _fixture(native_k=native_k, c_backing_n=64,
                             activation_guard="snapshot", persistent_inputs=True)
    snapshot, = nodes(before, tirx.Bind)
    selected = _insert(before)
    assert len(calls(selected, _COPY)) == len(calls(before, _COPY))
    results = []
    for candidate in (selected, _roundtrip(selected)):
        original = tvm.ir.save_json(candidate)
        # InsertSync moved the result's activation condition to one enclosing
        # scope. A later public pass must reconstruct the same owned effects
        # from that canonical IR, including after serialization.
        after = transform.ResolveCore()(candidate)
        assert tvm.ir.save_json(candidate) == original
        definition, = nodes(after, tirx.Bind)
        assert definition.var.name == snapshot.var.name
        activation, = [branch for branch in nodes(after, tirx.IfThenElse)
                       if isinstance(branch.condition, tirx.Var)
                       and branch.condition.same_as(definition.var)]
        result_loop, = [loop for loop in nodes(activation, tirx.For)
                        if loop.loop_var.name == facts["result"].name]
        _, parents = next((stmt, parents) for stmt, parents in statements(after)
                          if stmt.same_as(result_loop))
        guard_position = next(index for index, parent in enumerate(parents) if parent.same_as(activation))
        assert any(isinstance(parent, tirx.For) for parent in parents[guard_position + 1:])
        assert all("unit_flag_ctrl" in call.annotations for call in calls(result_loop, _GEMM))
        assert _bind_core(after, definition.var.name) == 2
        tvm.ir.assert_structural_equal(candidate, after, map_free_vars=True)
        results.append(after)
    # Compare task code across JSON; target layout objects in the root's
    # allocation metadata have a separate serialization/equality contract.
    tvm.ir.assert_structural_equal(*(mod["main"].body.block.body for mod in results), map_free_vars=True)
