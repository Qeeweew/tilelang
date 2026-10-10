"""A consumed incoming stage moves a whole unit transaction and its gates."""
from collections import defaultdict

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import calls
from testing.ascend._ir import seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_unit_flag_activation_guards import _replace_root
from testing.ascend.auto_schedule.test_unit_flag_native_pipeline import _lower, _physical_loop_facts, _check_brackets
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _COPY, _DIRECTION, _GEMM, _fixture, _is_export, _name, _trace, _value,
)


def _entry(guarded, rejection=None, manual=False):
    before, facts = _fixture(native_k=3, c_backing_n=64, persistent_inputs=True,
        activation_guard="direct" if guarded else None)
    parent_name = "epoch" if guarded else "owner"

    def mutate(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            body = node.body.then_case if isinstance(node.body, tirx.IfThenElse) else node.body
            if isinstance(body, tirx.For) and body.loop_var.same_as(facts["result"]):
                return unit(node.body, core=None, stage=1)
        # A guard is outside the schedule marker, so leave that activation
        # intact while replacing only its result control's incoming stage.
        if isinstance(node, tirx.For) and node.loop_var.name == parent_name:
            annotations = dict(node.annotations)
            if rejection != "missing-bound" and not manual:
                annotations["num_stages"] = 1 if rejection == "small-bound" else 2
            if manual:
                annotations["enable_offset"] = True
            if rejection == "offset-disabled":
                annotations["enable_offset"] = False
            if rejection == "unconsumed-auto":
                annotations["tl.ascend_auto_pipeline"] = True
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            seq(unit(tirx.Evaluate(0), core=2), node.body),
                            annotations=annotations)

    root = before["main"].body.block
    body = tirx.stmt_functor.ir_transform(root.body, None, mutate)
    return _replace_root(before, body), facts


@pytest.mark.parametrize("guarded", [False, True], ids=["owner-fill-drain", "guarded-epoch-fill-drain"])
@pytest.mark.parametrize("manual", [False, True], ids=["consumed-auto-bound", "complete-manual-stages"])
def test_complete_stream_at_incoming_stage_keeps_exact_transactions_and_gates(guarded, manual):
    before, facts = _entry(guarded, manual=manual)
    original = tvm.ir.save_json(before)
    _, synced, _, after = _lower(before)
    assert tvm.ir.save_json(before) == original
    assert calls(synced, _DIRECTION), "The complete shifted result was not selected"
    keys = ("epoch", "result", "k") if guarded else ("epoch", "owner", "result", "k")
    facts = _physical_loop_facts(after, facts, keys)
    records = _trace(after, max_extent=5)
    transactions = defaultdict(list)
    activations = defaultdict(list)
    pending = []
    for index, (call, env, _) in enumerate(records):
        if _name(call) == _GEMM:
            pending.append((index, call, env))
        if _is_export(call, facts):
            load = call.args[1].args[0]
            key = tuple(_value(axis, env) for axis in load.indices[:3])
            assert key not in transactions
            transactions[key] = [*pending, (index, call, env)]
            pending = []
        if facts["epoch"] in env:
            activation = (_value(facts["epoch"], env),
                          _value(facts["owner"], env) if facts["owner"] in env else 0)
            activations[activation].append(index)
    assert not pending
    expected_epochs = (1, 2) if guarded else range(3)
    expected_owners = (0,) if guarded else range(2)
    assert set(transactions) == {(e, owner, result) for e in expected_epochs
                                for owner in expected_owners for result in range(4)}
    for chain in transactions.values():
        mads, export = chain[:-1], chain[-1]
        assert len(mads) == 3
        assert [_value(call.args[9], env) for _, call, env in mads] == [1, 0, 0]
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env in mads] == [2, 2, 3]
        assert _value(export[1].annotations["unit_flag_ctrl"], export[2]) == 3
    active = {key: positions for key, positions in activations.items()
              if any(_name(records[i][0]) == _GEMM for i in positions)}
    _check_brackets(records, facts, active)


@pytest.mark.parametrize("rejection", ["missing-bound", "small-bound", "offset-disabled", "unconsumed-auto"])
def test_unproved_incoming_stage_cannot_select_a_unit_stream(rejection):
    before, _ = _entry(False, rejection)
    try:
        _, synced, _, _ = _lower(before)
    except tvm.error.InternalError as error:
        # Keep the ordinary scheduled-clock diagnostic when its source IR
        # has no legal finite flag clock. Selection cannot make it disappear.
        assert "dav-2201 local binary flag" in str(error)
    else:
        assert not calls(synced, _DIRECTION)
