"""Validate actual straight-line FFI starts before applying a native plan.

Automatic invalid results retain source order; explicit per-pipe schedules
must report the violated contract. Costs are diagnostic ordering weights.
"""

from inspect import signature

import pytest

pytest.importorskip("z3")
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


_INVALID = "Invalid Ascend list schedule: "


def _target(body, buffers=(), params=()):
    before = kernel(body, buffers=buffers, params=params)
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return before


def _exp(source, destination):
    def pointer(buffer, effect):
        return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], 32, effect])
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.exp"),
                                  [pointer(destination, 2), pointer(source, 1),
                                   tirx.IntImm("uint32", 32)]))


def _data_pair(manual):
    source, output = [tirx.decl_buffer((32,), "float32", name=name) for name in ("source", "output")]
    temp = tirx.decl_buffer((32,), "float32", name="temp", scope="shared.dyn")
    stage = 0 if manual else -1
    return _target(seq(
        unit(copy(source, temp), stage=stage, cost=(2, 1)),
        unit(copy(temp, output), stage=stage, cost=(1, 1)),
    ), [temp], [source, output])


def _independent_pair(manual=False, *, distinct_pipes=False):
    source = tirx.decl_buffer((32,), "float32", name="source")
    first, second, other = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn")
                            for name in ("first", "second", "other")]
    stage = 0 if manual else -1
    return _target(seq(
        unit(_exp(other, first) if distinct_pipes else copy(source, first), stage=stage, cost=(2, 2)),
        unit(copy(source, second), stage=stage, cost=(2, 2)),
    ), [first, second, other], [source])


def _owner_pair(manual):
    source, output = [tirx.decl_buffer((2, 32), "float32", name=name) for name in ("source", "output")]
    shared, other, result = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn")
                             for name in ("shared", "other", "result")]
    owners = []
    for index in range(2):
        epoch = tirx.Var(f"owner{index}", "int32")
        body = seq(
            unit(copy(source, shared, src_indices=[epoch, 0], src_shape=[1, 32]), stage=-1, cost=(2, 1)),
            unit(copy(shared, output, dst_indices=[epoch, 0], dst_shape=[1, 32]), stage=-1, cost=(1, 1)),
        ) if index == 0 else seq(
            unit(_exp(other, shared), stage=-1, cost=(2, 1)),
            unit(_exp(shared, result), stage=-1, cost=(1, 1)),
        )
        owners.append(unit(tirx.For(
            epoch, 0, 2, tirx.ForKind.SERIAL, body,
            annotations={"num_stages": 1, "multi_buffer_eligible": [shared.data]},
        ), core=None, stage=0 if manual else -1))
    return _target(seq(*owners), [shared, other, result], [source, output])


def _inject(monkeypatch, starts, inspect):
    original = z3_scheduler.z3_schedule_python
    observed = []

    def witness(*args, **kwargs):
        inputs = dict(signature(original).bind(*args, **kwargs).arguments)
        inspect(inputs)
        observed.append(inputs)
        # Native consumes starts and applies its own stable sort. The second
        # Python result is intentionally source order even for legal reversal.
        return starts, list(range(len(inputs["latencies"])))

    monkeypatch.setattr(z3_scheduler, "z3_schedule_python", witness)
    return observed


def _destinations(after):
    result = []
    for call in nodes(after, tirx.Call):
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name == "tl.tileop.ascend_copy":
            result.append(call.args[1].args[0].buffer.name)
        elif call.op.name == "tl.ascend_membase.exp":
            result.append(call.args[0].args[0].buffer.name)
    return result


def _stages_are_zero(after):
    stages = [int(node.node["stage"]) for node in nodes(after, tirx.AttrStmt)
              if node.attr_key == "tl.schedule_unit"]
    assert stages and all(stage == 0 for stage in stages)


def _check_invalid(before, manual, reason, capfd, labels):
    capfd.readouterr()
    if manual:
        with pytest.raises(tvm.error.InternalError, match=_INVALID + reason):
            transform.AutoSchedule()(before)
    else:
        after = transform.AutoSchedule()(before)
        assert _INVALID + reason in capfd.readouterr().err
        assert _destinations(after) == labels
        _stages_are_zero(after)
    return None if manual else after


@pytest.mark.parametrize("manual", [False, True], ids=["automatic-fallback", "manual-error"])
def test_native_list_result_must_preserve_real_data_completion(monkeypatch, capfd, manual):
    def inspect(inputs):
        assert (0, 1, 2) in map(tuple, inputs["data_deps"])
        assert not inputs["owner_exclusion_deps"]
    observed = _inject(monkeypatch, [4, 0], inspect)
    _check_invalid(_data_pair(manual), manual, "data-dependency", capfd, ["temp", "output"])
    assert len(observed) == 1


@pytest.mark.parametrize("manual", [False, True], ids=["automatic-fallback", "manual-error"])
def test_native_list_result_must_keep_supplied_resource_issue_gap(monkeypatch, capfd, manual):
    def inspect(inputs):
        assert inputs["iis"] == [2, 2]
        assert inputs["resource_deps"] == [(0, 1)]
        assert not inputs["data_deps"] and not inputs["owner_exclusion_deps"]
    observed = _inject(monkeypatch, [1, 0], inspect)
    _check_invalid(_independent_pair(manual), manual, "resource-overlap", capfd, ["first", "second"])
    assert len(observed) == 1


def test_native_list_manual_pipe_order_is_separate_from_resource_legality(monkeypatch, capfd):
    def inspect(inputs):
        assert inputs["iis"] == [2, 2]
        assert inputs["pipe_order_deps"] == [(0, 1)]
        assert not inputs["data_deps"] and not inputs["owner_exclusion_deps"]
    observed = _inject(monkeypatch, [2, 0], inspect)
    _check_invalid(_independent_pair(True), True, "pipe-order", capfd, [])
    assert len(observed) == 1


@pytest.mark.parametrize("manual", [False, True], ids=["automatic-fallback", "manual-error"])
@pytest.mark.parametrize("starts,reason", [
    pytest.param([0], "invalid-task-count", id="missing-start"),
    pytest.param([0, -1], "negative-start", id="negative-start"),
    pytest.param([(1 << 63) - 2, 0], "completion-overflow", id="int64-completion-overflow"),
])
def test_native_list_malformed_starts_do_not_mutate_the_selected_order(monkeypatch, capfd, manual, starts, reason):
    observed = _inject(monkeypatch, starts, lambda inputs: None)
    _check_invalid(_independent_pair(manual), manual, reason, capfd, ["first", "second"])
    assert len(observed) == 1


@pytest.mark.parametrize("distinct_pipes,starts,expected", [
    pytest.param(False, [2, 0], ["second", "first"], id="legal-resource-reversal"),
    pytest.param(True, [3, 3], ["first", "second"], id="independent-tie-source-order"),
])
def test_native_list_legal_results_keep_reversal_and_stable_ties(monkeypatch, capfd, distinct_pipes, starts, expected):
    def inspect(inputs):
        assert not inputs["data_deps"] and not inputs["owner_exclusion_deps"]
        assert not inputs["pipe_order_deps"]
        assert inputs["resource_deps"] == ([] if distinct_pipes else [(0, 1)])
    observed = _inject(monkeypatch, starts, inspect)
    capfd.readouterr()
    after = transform.AutoSchedule()(_independent_pair(distinct_pipes=distinct_pipes))
    assert _INVALID not in capfd.readouterr().err
    assert _destinations(after) == expected
    _stages_are_zero(after)
    assert len(observed) == 1


@pytest.mark.parametrize("manual", [False, True], ids=["automatic-fallback", "manual-error"])
def test_native_list_distinct_pipe_owners_still_exclude_shared_storage(monkeypatch, capfd, manual):
    def inspect(inputs):
        assert len(inputs["latencies"]) == 2
        assert not inputs["data_deps"] and not inputs["resource_deps"]
        assert not inputs["pipe_order_deps"]
        owners = inputs["owner_exclusion_deps"]
        assert owners and all(tuple(edge[:2]) == (0, 1) for edge in owners)
        assert all(forward > 1 and reverse > 1 for _, _, forward, reverse in owners)
    observed = _inject(monkeypatch, [1, 0], inspect)
    before = _owner_pair(manual)
    capfd.readouterr()
    if manual:
        with pytest.raises(tvm.error.InternalError, match=_INVALID + "owner-exclusion"):
            transform.AutoSchedule()(before)
    else:
        after = transform.AutoSchedule()(before)
        assert _INVALID + "owner-exclusion" in capfd.readouterr().err
        assert [loop.loop_var.name for loop in nodes(after, tirx.For)] == ["owner0", "owner1"]
        _stages_are_zero(after)
    assert len(observed) == 1
