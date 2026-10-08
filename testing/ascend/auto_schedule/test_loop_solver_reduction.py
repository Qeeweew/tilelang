"""Modulo reductions retain cyclic resource gaps and stage freedom.

All costs are diagnostic integer weights; no target timing model is assumed.
"""

import itertools

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _bound(iis, flags, pairs):
    return z3_scheduler._loop_resource_ii_lower_bound(iis, flags, pairs)


def _reduce(iis, data, pairs):
    return z3_scheduler._reduce_single_stage_loop_resources(len(iis), iis, data, pairs)


def _window(iis, flags, data=(), pairs=(), pipes=()):
    return z3_scheduler._single_stage_loop_ii_lower_bound(iis, flags, data, pairs, pipes)


def test_complete_resource_clique_proves_total_issue_work():
    assert _bound([2, 3, 4], [16, 16, 16], [(0, 1), (0, 2), (1, 2)]) == 9


def test_pipe_masks_do_not_add_missing_resource_constraints():
    # A connected path is not a clique. In particular, its two end tasks may
    # overlap; summing all three issue intervals would remove feasible plans.
    assert _bound([2, 3, 4], [16, 16, 16], [(0, 1), (1, 2)]) == 4
    assert _bound([2, 3, 4], [16, 16, 16], []) == 4


def test_independent_core_cliques_are_bounded_separately():
    assert _bound([2, 3, 4, 5], [2] * 4, [(0, 1), (2, 3)]) == 9


@pytest.mark.parametrize("high_bit", [2 << 7, 1 << 35])
def test_core_pipe_namespaces_retain_each_explicit_resource_clique(high_bit):
    # Cube MTE2 and another core's MTE2 have independent cyclic resources.
    # Summing all four would be unsound; ignoring the high-bit clique would
    # miss its stronger bound. No conflict is inferred across namespaces.
    intervals = [2, 3, 4, 5]
    flags = [2, 2, high_bit, high_bit]
    resources = [(0, 1), (2, 3)]
    assert _bound(intervals, flags, resources) == 9
    lower, upper = z3_scheduler._straight_line_makespan_bounds(
        intervals, intervals, flags, [], resources, [], [],
    )
    assert lower == upper == 9


@pytest.mark.parametrize("bit", [2, 2 << 7, 1 << 35])
def test_resource_window_bound_uses_supplied_low_or_high_namespace(bit):
    assert _window([1, 2, 3, 1], [0, bit, bit, 0],
                   [(0, 1, 0, 2), (0, 2, 0, 2), (1, 3, 0, 4), (2, 3, 0, 4)],
                   [(1, 2)]) == 9


def test_resource_window_bound_accounts_for_earliest_release():
    # Three mutually exclusive vector issues all wait for one producer.
    assert _window([1] * 4, [2, 16, 16, 16],
                   [(0, i, 0, 2) for i in (1, 2, 3)], [(1, 2), (1, 3), (2, 3)]) == 5


def test_resource_window_bound_retains_downstream_distance_zero_tail():
    assert _window([1, 2, 3, 1], [2, 16, 16, 4],
                   [(0, 1, 0, 2), (0, 2, 0, 2), (1, 3, 0, 4), (2, 3, 0, 4)], [(1, 2)]) == 9


def test_resource_window_bound_omits_the_last_issue_interval_from_window():
    # The final long-issue task may finish after II. Its residue must fit, but
    # charging its entire interval would incorrectly strengthen 8 to 12.
    assert _window([1, 2, 5], [2, 16, 16], [(0, 1, 0, 5), (0, 2, 0, 5)], [(1, 2)]) == 8


@pytest.mark.parametrize("distance", [1, 2, -1, -2])
def test_resource_window_bound_ignores_recurrence_and_buffer_distance(distance):
    assert _window([1, 1], [2, 16], [(0, 1, distance, 100)]) == 1


def test_resource_window_bound_conservatively_handles_cycles_and_negative_weights():
    assert _window([1, 1], [16, 16], [(0, 1, 0, 0), (1, 0, 0, 0)], [(0, 1)]) == 0
    assert _window([1, 1], [2, 16], [(0, 1, 0, -100)]) == 1


def test_distance_zero_paths_force_only_the_proven_direction():
    assert _reduce([1, 1, 1], [(0, 2, 0, 0), (2, 1, 0, 2)], [(0, 1), (1, 2)]) == (
        [(0, 1), (2, 1)], [],
    )
    assert _reduce([1, 1], [(1, 0, 0, 0)], [(0, 1)]) == ([(1, 0)], [])


@pytest.mark.parametrize("distance", [1, 2, -1, -2])
def test_recurrence_or_buffer_distance_does_not_fix_residue_order(distance):
    assert _reduce([1, 1], [(0, 1, distance, 4)], [(0, 1)]) == ([], [(0, 1)])


def test_cyclic_distance_zero_graph_retains_resource_disjunctions():
    assert _reduce([1, 1], [(0, 1, 0, 0), (1, 0, 0, 0)], [(0, 1)]) == ([], [(0, 1)])


def test_zero_reverse_issue_gap_does_not_exclude_tied_residues():
    assert _reduce([2, 0], [(0, 1, 0, 0)], [(0, 1)]) == ([], [(0, 1)])


def _pair_legal(residues, ii, intervals, pair):
    i, j = pair
    delta = residues[j] - residues[i]
    return (delta >= intervals[i] and ii - delta >= intervals[j]
            or -delta >= intervals[j] and ii + delta >= intervals[i])


def test_small_graphs_preserve_every_feasible_residue_assignment():
    # Exhaust all forward zero-distance DAGs on three tasks (edge absent,
    # zero, or one), all resource graphs and three issue-cost assignments.
    # Compare original cyclic disjunctions with the reduced query, including
    # assignments below the proposed resource bound. No solver is the oracle.
    pairs = [(0, 1), (0, 2), (1, 2)]
    for intervals in ([1, 1, 1], [1, 2, 1], [2, 1, 2]):
        for edge_weights in itertools.product((None, 0, 1), repeat=3):
            data = [(i, j, 0, weight) for (i, j), weight in zip(pairs, edge_weights)
                    if weight is not None]
            for mask in range(8):
                resources = [pair for bit, pair in enumerate(pairs) if mask & (1 << bit)]
                ordered, unordered = _reduce(intervals, data, resources)
                lower = max(_bound(intervals, [16] * 3, resources),
                            _window(intervals, [16] * 3, data, resources))
                for ii in range(1, 7):
                    for residues in itertools.product(range(ii), repeat=3):
                        if not all(residues[j] - residues[i] >= weight for i, j, _, weight in data):
                            continue
                        original = (ii >= max(intervals)
                                    and all(_pair_legal(residues, ii, intervals, pair) for pair in resources))
                        reduced = (ii >= lower
                                   and all(residues[j] - residues[i] >= intervals[i]
                                           and residues[i] - residues[j] + ii >= intervals[j]
                                           for i, j in ordered)
                                   and all(_pair_legal(residues, ii, intervals, pair) for pair in unordered))
                        assert original == reduced, (intervals, data, resources, ii, residues)


def _inputs(**changes):
    values = dict(
        num_stages=1, latencies=[1, 2], iis=[1, 2], resource_flags=[16, 16],
        data_deps=[(0, 1, 0, 0)], resource_deps=[(0, 1)],
        owner_exclusion_deps=[], pipe_order_deps=[], buffer_sizes=[], memory_groups=[],
        stage_horizon=1,
    )
    values.update(changes)
    return values


def test_selected_plan_retains_forward_and_wrap_gaps():
    starts, versions, ii = z3_scheduler.z3_schedule_loop_python(**_inputs())
    assert ii == 3
    assert versions == []
    assert starts[1] - starts[0] >= 1
    assert starts[0] - starts[1] + ii >= 2


@pytest.mark.parametrize("high_bit", [2 << 7, 1 << 35])
def test_no_offset_windows_are_independent_across_core_pipe_namespaces(high_bit):
    intervals, resources = [1] * 4, [(0, 1), (2, 3)]
    starts, versions, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        num_stages=3, stage_horizon=3, enable_offset=False,
        latencies=[1, 1, 4, 1], iis=intervals,
        resource_flags=[2, 2, high_bit, high_bit], resource_deps=resources,
        data_deps=[(0, 2, 0, 4)],
    ))
    # [0, 1, 4, 5] is a finite SAT witness at II=2. Each core fits its own
    # window, while an artificial shared MTE2 window would exclude this II.
    witness = [0, 1, 4, 5]
    assert ii == 2 and versions == []
    for selected in (witness, starts):
        assert selected[2] - selected[0] >= 4
        assert max(selected) - min(selected) >= ii
        for i, j in resources:
            assert abs(selected[j] - selected[i]) < ii
            assert _pair_legal([start % ii for start in selected], ii, intervals, (i, j))


@pytest.mark.parametrize("bit", [2, 2 << 7, 1 << 35])
def test_no_offset_window_still_constrains_each_core_pipe(bit):
    starts, _, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        num_stages=3, stage_horizon=3, enable_offset=False,
        latencies=[4, 1], iis=[1, 1], resource_flags=[bit, bit],
        data_deps=[(0, 1, 0, 4)], resource_deps=[(0, 1)],
    ))
    assert ii == 5  # The same resource's required span is four, hence II > 4.
    assert 4 <= starts[1] - starts[0] < ii
    assert _pair_legal([start % ii for start in starts], ii, [1, 1], (0, 1))


@pytest.mark.parametrize("bit", [2, 2 << 7, 1 << 35])
def test_resource_masks_do_not_create_unsupplied_core_pipe_exclusions(bit):
    starts, _, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        iis=[2, 3], latencies=[2, 3], resource_flags=[bit, bit],
        data_deps=[], resource_deps=[],
    ))
    assert _bound([2, 3], [bit, bit], []) == ii == 3
    # A shared mask is insufficient proof of exclusion. II=3 cannot fit
    # both intervals without overlap, so inferred exclusion would fail here.
    assert not _pair_legal(starts, ii, [2, 3], (0, 1))


def test_single_stage_window_bounds_issue_residues_without_changing_native_completion():
    latencies = [5, 2, 5]
    starts, _, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        latencies=latencies, iis=[1, 2, 5], resource_flags=[2, 16, 16],
        data_deps=[(0, 1, 0, 5), (0, 2, 0, 5)], resource_deps=[(1, 2)],
    ))
    assert ii == 8
    assert all(0 <= start < ii for start in starts)
    assert starts[1] - starts[0] >= 5 and starts[2] - starts[0] >= 5
    assert _pair_legal(starts, ii, [1, 2, 5], (1, 2))
    assert max(start + latency for start, latency in zip(starts, latencies)) > ii


@pytest.mark.parametrize("controls", [
    dict(stage_horizon=None),
    dict(stage_horizon=2, enable_offset=True),
    dict(stage_horizon=1, manual_schedule=True, manual_stages=[0, 0]),
])
def test_single_stage_reduction_does_not_constrain_free_or_manual_stages(monkeypatch, controls):
    def forbidden(*args):
        raise AssertionError("distance-zero residue reduction requires automatic k=0")

    monkeypatch.setattr(z3_scheduler, "_reduce_single_stage_loop_resources", forbidden)
    monkeypatch.setattr(z3_scheduler, "_single_stage_loop_ii_lower_bound", forbidden)
    starts, _, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(**controls))
    assert starts[1] - starts[0] >= 0
    assert ii == 3


def test_automatic_pipeline_retains_cross_stage_recurrence_and_two_versions():
    starts, versions, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        num_stages=2, latencies=[3, 3], iis=[1, 1], resource_flags=[2, 1],
        data_deps=[(0, 1, 0, 3), (1, 0, -1, 3)], resource_deps=[],
        buffer_sizes=[32], memory_groups=[[64, 0]], stage_horizon=2, enable_offset=True,
    ))
    assert ii == 3
    assert versions == [2]
    assert starts[1] - starts[0] >= 3
    assert starts[0] - starts[1] + ii * versions[0] >= 3
    assert {start // ii for start in starts} == {0, 1}


def _exhaust_first_query(monkeypatch):
    original = z3.Solver
    calls = []

    class LimitedSolver:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def check(self):
            calls.append(1)
            return z3.unknown if len(calls) == 1 else self.inner.check()

        def reason_unknown(self):
            return "test resource exhaustion"

    monkeypatch.setattr(z3, "Solver", LimitedSolver)


def test_unknown_is_reported_as_unproven_not_infeasible(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    _exhaust_first_query(monkeypatch)
    _, _, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        latencies=[2, 2], iis=[1, 1], resource_flags=[2, 1],
        data_deps=[], resource_deps=[], verbose=True,
    ))
    output = capsys.readouterr().out
    assert ii == 4
    assert "II = 3 returned unknown (test resource exhaustion)" in output
    assert "II = 3 is infeasible" not in output
    assert "Selected feasible II = 4; minimality is unproven" in output
    assert "Minimal" not in output


def test_manual_unknown_does_not_claim_infeasibility_or_return_fallback(monkeypatch):
    _exhaust_first_query(monkeypatch)
    with pytest.raises(RuntimeError, match="Manual schedule search returned unknown.*feasibility is unproven"):
        z3_scheduler.z3_schedule_loop_python(**_inputs(
            latencies=[1], iis=[1], resource_flags=[16], data_deps=[], resource_deps=[],
            manual_schedule=True, manual_stages=[0],
        ))
