"""Mutable or binding guard scopes cannot form an automatic unit stream.

These are prepared InsertSync inputs. A rejected candidate must retain the
ordinary finite-event contract, including its existing unsupported-clock
diagnostic when that contract cannot be proved.
"""

from collections import Counter
import re

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import calls, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _COPY,
    _DIRECTION,
    _GEMM,
    _SET,
    _WAIT,
    _fixture,
    _insert,
    _is_export,
    _name,
    _trace,
    _value,
)


def _replace_root(mod, body, *, allocations=None, annotations=None):
    func = mod["main"]
    root = func.body.block
    replacement = tirx.SBlock(
        root.iter_vars, root.reads, root.writes, root.name_hint, body,
        init=root.init,
        alloc_buffers=root.alloc_buffers if allocations is None else allocations,
        match_buffers=root.match_buffers,
        annotations=root.annotations if annotations is None else annotations,
    )
    return tvm.IRModule({"main": func.with_body(tirx.SBlockRealize([], True, replacement))})


def _guarded_fixture(kind, native_k):
    mode = "snapshot" if kind == "captured-load" else "direct"
    before, facts = _fixture(
        native_k=native_k, c_backing_n=64,
        activation_guard=mode, persistent_inputs=True,
    )
    state = tirx.decl_buffer((1,), "int32", name="guard_state", scope="local.var")
    captured = None
    rewritten = 0

    def mutate(node):
        nonlocal captured, rewritten
        if (kind == "captured-load" and isinstance(node, tirx.Bind)
                and node.var.name == "phase_active"):
            rewritten += 1
            captured = node.var
            return tirx.Bind(node.var, tirx.And(node.value, state[0] == 1))
        if (isinstance(node, tirx.IfThenElse)
                and isinstance(node.then_case, tirx.For)
                and node.then_case.loop_var.same_as(facts["result"])):
            if kind == "direct-load":
                rewritten += 1
                return tirx.IfThenElse(
                    tirx.And(node.condition, state[0] == 1), node.then_case, None,
                )
            if kind == "attribute":
                rewritten += 1
                # This is a decoder-supported assumption scope, but moving
                # its condition together with that scope is not admitted by
                # the automatic stream selector.
                scoped = tirx.AttrStmt(node.condition, "tl.assume", 1, node.then_case)
                return tirx.IfThenElse(node.condition, scoped, None)

    root = before["main"].body.block
    body = tirx.stmt_functor.ir_transform(root.body, None, mutate)
    assert rewritten == 1
    if kind != "attribute":
        # A deterministic scalar value makes the successful ordinary path's
        # finite trace executable. It is still a BufferLoad in the selector's
        # input; this test must not rely on propagating its initializer.
        body = seq(unit(tirx.BufferStore(state, 1, [0]), core=2), body)
        before = _replace_root(before, body, allocations=[*root.alloc_buffers, state])
    else:
        before = _replace_root(before, body)
    return before, facts, state, captured


def _ordinary_baseline(before, facts):
    root = before["main"].body.block
    layouts = {buffer: layout for buffer, layout in root.annotations["layout_map"].items()
               if not buffer.same_as(facts["c"])}
    return _replace_root(
        before, root.body, annotations={**dict(root.annotations), "layout_map": layouts},
    )


def _shifted_diagnostic(error):
    message = str(error)
    assert "dav-2201 local binary flag:" in message
    reasons = (
        "shifted carried events require an unconditional outermost lexical loop",
        "shifted carried events require one exact contiguous active-epoch interval with static cut points",
    )
    matched = [reason for reason in reasons if reason in message]
    assert len(matched) == 1, "Expected the existing ordinary shifted-clock diagnostic"
    pipes = re.findall(r"pipe=([^,\s)]+)", message)
    assert pipes, "The ordinary shifted-clock diagnostic must identify its pipe"
    return matched[0], pipes[-1]


def _check_guard_scope(after, facts, kind, state, captured):
    results = [(stmt, parents) for stmt, parents in statements(after)
               if isinstance(stmt, tirx.For) and stmt.loop_var.same_as(facts["result"])]
    assert len(results) == 1
    _, parents = results[0]
    guards = [parent for parent in parents if isinstance(parent, tirx.IfThenElse)]
    assert guards, "The original result activation guard was lost"
    if kind == "direct-load":
        assert any(load.buffer.data.same_as(state.data)
                   for guard in guards for load in nodes(guard.condition, tirx.BufferLoad))
    elif kind == "captured-load":
        assert captured is not None
        assert any(guard.condition.same_as(captured) for guard in guards)
        definitions = [bind for bind in nodes(after, tirx.Bind) if bind.var.same_as(captured)]
        assert len(definitions) == 1
        assert any(load.buffer.data.same_as(state.data)
                   for load in nodes(definitions[0].value, tirx.BufferLoad))
    else:
        attributes = [parent for parent in parents
                      if isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.assume"]
        assert len(attributes) == 1
        # The assumption remains between the original condition and result
        # loop, rather than being migrated to a synthetic activation shell.
        assumption = attributes[0]
        assert any(tvm.ir.structural_equal(assumption.node, guard.condition)
                   and parents.index(guard) < parents.index(assumption) for guard in guards)


@pytest.mark.parametrize("native_k", [1, 3], ids=["single-MAD", "native-K-chain"])
@pytest.mark.parametrize("kind", ["direct-load", "captured-load", "attribute"])
def test_nonstable_result_activation_keeps_ordinary_contract(kind, native_k):
    before, facts, state, captured = _guarded_fixture(kind, native_k)
    ordinary = _ordinary_baseline(before, facts)
    original = tvm.ir.save_json(before)
    try:
        after = _insert(before)
    except tvm.error.InternalError as rejected:
        # Some guarded source/consumer clocks remain unsupported even without
        # unit selection. Preserve that precise failure, rather than accepting
        # a selected protocol which happens to bypass it.
        diagnostic = _shifted_diagnostic(rejected)
        with pytest.raises(tvm.error.InternalError) as baseline_error:
            _insert(ordinary)
        assert _shifted_diagnostic(baseline_error.value) == diagnostic
        assert tvm.ir.save_json(before) == original
        return

    baseline = _insert(ordinary)
    assert tvm.ir.save_json(before) == original
    assert not calls(after, _DIRECTION)
    for call in calls(after, _GEMM) + calls(after, _COPY):
        control = call.annotations.get("unit_flag_ctrl")
        assert control is None or _value(control, {}) == 0
    tvm.ir.assert_structural_equal(
        after["main"].body.block.body, baseline["main"].body.block.body,
        map_free_vars=True,
    )
    _check_guard_scope(after, facts, kind, state, captured)
    records = _trace(after)
    assert sum(_name(call) == _GEMM for call, _, _ in records) == 8 * native_k
    assert sum(_is_export(call, facts) for call, _, _ in records) == 8
    for event in ("M_FIX", "FIX_M"):
        for operation in (_SET, _WAIT):
            assert any(_name(call) == operation and call.args[0].value == event
                       for call, _, _ in records), (event, operation)
    totals = [Counter((call.args[0].value, _value(call.args[1], env))
                      for call, env, _ in records if _name(call) == operation)
              for operation in (_SET, _WAIT)]
    assert totals[0] == totals[1]
