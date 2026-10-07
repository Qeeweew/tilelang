"""Straight-line order reduction preserves resource gaps and optimal schedules.

All costs are diagnostic weights; no target timing model is assumed.
"""

import itertools

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _reduce(*, latencies, iis, flags, data=(), resources=(), owners=(), pipes=()):
    return z3_scheduler._reduce_straight_line_orders(
        latencies, iis, flags, data, resources, owners, pipes,
    )


def test_weighted_paths_remove_only_implied_resource_constraints():
    ordered, unordered, symmetric = _reduce(
        latencies=[1, 2, 3, 4], iis=[2, 5, 1, 1], flags=[1, 2, 4, 8],
        data=[(0, 1, 1), (1, 2, 1)],
        resources=[(0, 2), (1, 2), (2, 3)], pipes=[(2, 3)],
    )
    assert ordered == [(1, 2, 5), (2, 3, 1)]
    assert unordered == symmetric == []


def test_zero_intervals_do_not_make_a_reverse_disjunct_impossible():
    ordered, unordered, _ = _reduce(
        latencies=[1, 2], iis=[3, 0], flags=[1, 1],
        data=[(0, 1, 0)], resources=[(0, 1)],
    )
    assert ordered == []
    assert unordered == [(0, 1)]
    starts, _ = z3_scheduler.z3_schedule_python(
        [1, 2], [3, 0], [1, 1], [(0, 1, 0)], [(0, 1)],
    )
    assert starts == [0, 0]


def _diamond(**changes):
    values = dict(
        latencies=[1, 1, 1, 1], iis=[1, 1, 1, 1], flags=[2, 16, 16, 4],
        data=[(0, 1, 2), (0, 2, 2), (1, 3, 3), (2, 3, 3)],
        resources=[(1, 2)], owners=[], pipes=[],
    )
    values.update(changes)
    return values


def test_exact_twins_choose_one_source_order_representative():
    ordered, unordered, symmetric = _reduce(**_diamond())
    assert ordered == [(1, 2, 1)]
    assert unordered == []
    assert symmetric == [(1, 2)]


@pytest.mark.parametrize("difference", ["cost", "weighted_input", "resource", "owner", "pipe"])
def test_independent_tasks_with_different_constraints_are_not_twins(difference):
    values = _diamond()
    if difference == "cost":
        values["latencies"][2] = 2
    elif difference == "weighted_input":
        values["data"][1] = (0, 2, 3)
    elif difference == "resource":
        values["resources"].append((0, 1))
    elif difference == "owner":
        values["owners"] = [(0, 1, 2, 3), (0, 2, 2, 4)]
    else:
        values["pipes"] = [(0, 1)]
    assert _reduce(**values)[2] == []


def test_equal_external_owner_constraints_permit_symmetry():
    assert _reduce(**_diamond(owners=[(0, 1, 2, 3), (0, 2, 2, 3)]))[2] == [(1, 2)]


def test_internal_owner_constraint_conservatively_keeps_both_orders():
    assert _reduce(**_diamond(owners=[(1, 2, 2, 3)]))[2] == []


def test_resource_work_bound_includes_release_and_completion_tail():
    latencies, iis, flags = [1] * 5, [1] * 5, [2, 16, 16, 16, 4]
    data = [(0, i, 1) for i in (1, 2, 3)] + [(i, 4, 1) for i in (1, 2, 3)]
    resources = [(1, 2), (1, 3), (2, 3)]
    assert z3_scheduler._straight_line_makespan_bounds(latencies, iis, flags, data, resources, [], []) == (5, 5)
    # A mask without its actual pair exclusions cannot justify summing work.
    assert z3_scheduler._straight_line_makespan_bounds(latencies, iis, flags, data, [(1, 2)], [], [])[0] == 3


def _legal(starts, values):
    return (
        all(starts[j] >= starts[i] + weight for i, j, weight in values["data"])
        and all(starts[j] >= starts[i] + values["iis"][i]
                or starts[i] >= starts[j] + values["iis"][j]
                for i, j in values["resources"])
        and all(starts[j] - starts[i] >= forward or starts[i] - starts[j] >= reverse
                for i, j, forward, reverse in values["owners"])
        and all(starts[i] <= starts[j] for i, j in values["pipes"])
    )


@pytest.mark.parametrize("case", ["twins", "different_inputs", "reverse_data", "reverse_owner"])
def test_selected_plan_matches_exhaustive_optimum_and_all_original_constraints(case):
    if case == "twins":
        values = _diamond()
    elif case == "different_inputs":
        values = _diamond(data=[(0, 1, 4), (0, 2, 1), (1, 3, 1), (2, 3, 1)])
    elif case == "reverse_data":
        values = dict(latencies=[1, 2], iis=[1, 1], flags=[16, 16],
                      data=[(1, 0, 2)], resources=[(0, 1)], owners=[], pipes=[])
    else:
        values = dict(latencies=[1, 1, 1], iis=[1, 1, 1], flags=[16, 16, 4],
                      data=[(0, 2, 4)], resources=[(0, 1)], owners=[(0, 1, 8, 1)], pipes=[])
    # This finite oracle includes a feasible schedule with completion <= 9.
    candidates = (starts for starts in itertools.product(range(9), repeat=len(values["latencies"]))
                  if _legal(starts, values))
    optimum = min(max(start + latency for start, latency in zip(starts, values["latencies"]))
                  for starts in candidates)
    lower, upper = z3_scheduler._straight_line_makespan_bounds(
        values["latencies"], values["iis"], values["flags"], values["data"],
        values["resources"], values["owners"], values["pipes"],
    )
    assert lower <= optimum
    assert upper is None or optimum <= upper
    starts, indices = z3_scheduler.z3_schedule_python(
        values["latencies"], values["iis"], values["flags"], values["data"],
        values["resources"], values["owners"], values["pipes"],
    )
    assert _legal(starts, values)
    assert indices == sorted(range(len(starts)), key=lambda i: (starts[i], i))
    assert max(start + latency for start, latency in zip(starts, values["latencies"])) == optimum
    if case == "different_inputs":
        assert starts[2] < starts[1]
    elif case in ("reverse_data", "reverse_owner"):
        assert starts[1] < starts[0]


def test_inconsistent_graph_reports_unsat():
    with pytest.raises(RuntimeError, match="solver returned unsat"):
        z3_scheduler.z3_schedule_python([1, 1], [1, 1], [16, 16], [(0, 1, 1), (1, 0, 1)], [(0, 1)])


def test_solver_exhaustion_reports_unknown_and_reason(monkeypatch):
    original = z3.Optimize

    class ExhaustedOptimizer:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return "test resource exhaustion"

    monkeypatch.setattr(z3, "Optimize", ExhaustedOptimizer)
    with pytest.raises(RuntimeError, match=r"solver returned unknown \(test resource exhaustion\)"):
        z3_scheduler.z3_schedule_python([1], [1], [16], [], [])
