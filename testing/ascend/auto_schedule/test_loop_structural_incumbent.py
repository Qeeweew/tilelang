"""A checked structure backup rescues an exhausted free modulo search."""

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _inputs(**changes):
    values = dict(
        num_stages=2, latencies=[3, 3], iis=[1, 1], resource_flags=[16, 16],
        data_deps=[(0, 1, 0, 3), (1, 0, -1, 3)], resource_deps=[(0, 1)],
        owner_exclusion_deps=[], pipe_order_deps=[], buffer_sizes=[32],
        memory_groups=[[64, 0]], buffer_min_versions=[1],
        stage_horizon=2, enable_offset=True,
    )
    values.update(changes)
    return values


def _scripted_solver(monkeypatch, statuses, values=()):
    """Exercise result handling without asking a native solver to search."""
    checks, instances = [], []

    class ScriptedSolver:
        def __init__(self, *, ctx):
            self.ctx, self.hard, self.limits = ctx, [], []
            instances.append(self)

        def set(self, *args):
            self.limits.append(args)

        def add(self, *args):
            self.hard.extend(args)

        def check(self):
            self.ii = next(
                hard.arg(1).as_long() for hard in self.hard
                if z3.is_lt(hard) and str(hard.arg(0)) == "r_0"
            )
            status = statuses[min(len(checks), len(statuses) - 1)]
            checks.append(status)
            self.status = status
            return status

        def reason_unknown(self):
            return "test bounded search"

        def model(self):
            assert self.status == z3.sat, "unsat/unknown must not supply a model"
            model = z3.Model(ctx=self.ctx)
            for name, value in values:
                variable = z3.Bool(name, self.ctx) if isinstance(value, bool) else z3.Int(name, self.ctx)
                constant = z3.BoolVal(value, self.ctx) if isinstance(value, bool) else z3.IntVal(value, self.ctx)
                model.update_value(variable, constant)
            assert all(z3.is_true(model.eval(hard, model_completion=False)) for hard in self.hard)
            return model

    monkeypatch.setattr(z3, "Solver", ScriptedSolver)
    return checks, instances


def test_unknown_uses_checked_backup_without_changing_the_query_sequence(monkeypatch):
    problem = _inputs(
        latencies=[4] * 4, iis=[1] * 4, resource_flags=[2, 1, 16, 32],
        data_deps=[(0, 1, 0, 3), (1, 2, 0, 1), (2, 3, 0, 2),
                   (3, 0, -1, 5), (2, 1, 1, 1)],
        resource_deps=[(1, 2)], owner_exclusion_deps=[(3, 1, 2, 3)],
        pipe_order_deps=[(0, 2)], stage_order_deps=[(3, 0)],
        buffer_min_versions=[2], issue_priority_pairs=[(0, 3)],
    )
    checks, instances = _scripted_solver(monkeypatch, [z3.unknown])
    refined = []

    def retain(ctx, assertions, model, *args, **kwargs):
        assert all(z3.is_true(model.eval(hard, model_completion=False)) for hard in assertions)
        assert z3.is_true(model.eval(z3.Bool("O_1_2", ctx), model_completion=False))
        assert z3.is_false(model.eval(z3.Bool("Owner_3_1", ctx), model_completion=False))
        refined.append(1)
        return model

    monkeypatch.setattr(z3_scheduler, "_refine_loop_issue_order", retain)
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert starts == [0, 3, 4, 6] and counts == [2] and ii == 7
    assert ii < sum(problem["latencies"])
    assert checks == [z3.unknown] * 4
    assert [instance.ii for instance in instances] == [9, 13, 15, 16]
    # The witness builds assertions and evaluates a Model; it creates no
    # Solver and consumes none of the main query's unchanged resource limit.
    assert len(instances) == len(checks)
    assert all(instance.limits == [("random_seed", 42), ("rlimit", 1000000)] for instance in instances)
    assert refined == [1]


def test_free_sat_plan_keeps_its_selected_stages_and_resource_order(monkeypatch):
    # A reverse residue order with two physical slots improves the source
    # candidate's II6. The candidate's O_0_1=True must not constrain this query.
    checks, _ = _scripted_solver(monkeypatch, [z3.sat, z3.unsat], [
        ("k_0", 0), ("k_1", 1), ("r_0", 1), ("r_1", 0),
        ("buf_0", 2), ("O_0_1", False),
    ])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**_inputs())
    assert starts == [1, 4] and counts == [2] and ii == 4
    assert checks == [z3.sat, z3.unsat]


def test_backup_at_lower_bound_does_not_skip_or_replace_free_multistage_sat(monkeypatch):
    problem = _inputs(data_deps=[(0, 1, 0, 0), (1, 0, -1, 1)])
    _, candidate_ii = z3_scheduler._loop_source_order_candidate(
        problem["iis"], problem["data_deps"], problem["resource_deps"],
        [], [], problem["buffer_min_versions"],
    )
    lower = z3_scheduler._loop_resource_ii_lower_bound(
        problem["iis"], problem["resource_flags"], problem["resource_deps"],
    )
    assert candidate_ii == lower == 2

    def forbidden(*args):
        raise AssertionError("any free SAT keeps its plan without building a backup")

    monkeypatch.setattr(z3_scheduler, "_loop_source_order_candidate", forbidden)
    checks, instances = _scripted_solver(monkeypatch, [z3.sat, z3.unsat], [
        ("k_0", 0), ("k_1", 1), ("r_0", 1), ("r_1", 0),
        ("buf_0", 1), ("O_0_1", False),
    ])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert [instance.ii for instance in instances] == [4, 3]
    assert checks == [z3.sat, z3.unsat]
    assert starts == [1, 4] and counts == [1] and ii == 4
    assert {start // ii for start in starts} == {0, 1}


def test_candidate_promotes_ready_publication_and_keeps_common_handoff_order(monkeypatch):
    # Tasks2/3 update unrelated vector state after task1 has produced P.
    # Task4 may publish P before task3, so source-index serial fallback hides
    # useful issue freedom. Its common handoff must still precede acquire6.
    problem = _inputs(
        latencies=[3, 1, 1, 1, 1, 3, 1], iis=[1] * 7,
        resource_flags=[2, 16, 16, 16, 4, 2, 2],
        data_deps=[(0, 1, 0, 3), (1, 2, 0, 1), (2, 3, 0, 1),
                   (1, 4, 0, 1), (4, 5, 0, 1), (5, 6, 0, 3), (6, 0, 1, 1)],
        resource_deps=[(1, 2), (1, 3), (2, 3), (0, 6)],
        buffer_sizes=[], memory_groups=[], buffer_min_versions=[],
        issue_priority_pairs=[(4, 6), (1, 5)],
    )
    _scripted_solver(monkeypatch, [z3.unknown])

    def retain(ctx, assertions, model, k_vars, r_vars, buffers, pairs, **kwargs):
        assert all(z3.is_true(model.eval(hard, model_completion=False)) for hard in assertions)
        for producer, consumer in pairs:
            before = z3_scheduler._loop_issue_before(ctx, k_vars, r_vars, producer, consumer)
            assert z3.is_true(model.eval(before, model_completion=False))
        return model

    monkeypatch.setattr(z3_scheduler, "_refine_loop_issue_order", retain)
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert counts == [] and all(start < ii for start in starts)
    assert starts[4] < starts[3] and starts[4] < starts[6]


@pytest.mark.parametrize("status", [z3.unknown, z3.unsat])
@pytest.mark.parametrize("candidate", [([0, 1], 2), ([0], 2), ([-1, 2], 4)])
def test_invalid_or_incomplete_candidate_is_not_treated_as_feasible(monkeypatch, status, candidate):
    monkeypatch.setattr(z3_scheduler, "_loop_source_order_candidate", lambda *args: candidate)
    _scripted_solver(monkeypatch, [status])

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected candidates must not reach feasible-plan refinement")

    monkeypatch.setattr(z3_scheduler, "_refine_loop_prefetch", forbidden)
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(prefetch_priority_pairs=[(0, 1)]))
    # The pre-existing sequential fallback remains distinct from a validated
    # backup and cannot acquire a feasible model from unsat or unknown.
    assert starts == [0, 3] and counts == [1] and ii == 6


def test_cyclic_candidate_graph_leaves_reverse_order_free(monkeypatch):
    problem = _inputs(
        data_deps=[(1, 0, 0, 1)], buffer_sizes=[], memory_groups=[],
        buffer_min_versions=[],
    )
    assert z3_scheduler._loop_source_order_candidate(
        problem["iis"], problem["data_deps"], problem["resource_deps"], [], [], [],
    ) is None
    _scripted_solver(monkeypatch, [z3.sat], [
        ("k_0", 0), ("k_1", 0), ("r_0", 1), ("r_1", 0), ("O_0_1", False),
    ])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert starts == [1, 0] and counts == [] and ii == 2


def test_explicit_manual_stages_do_not_use_a_structural_candidate(monkeypatch):
    def forbidden(*args):
        raise AssertionError("manual scheduling must preserve its explicit contract")

    monkeypatch.setattr(z3_scheduler, "_loop_source_order_candidate", forbidden)
    _scripted_solver(monkeypatch, [z3.sat, z3.unsat], [
        ("k_0", 0), ("k_1", 1), ("r_0", 1), ("r_1", 0),
        ("buf_0", 2), ("O_0_1", False),
    ])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**_inputs(
        manual_schedule=True, manual_stages=[0, 1],
    ))
    assert starts == [1, 4] and counts == [2] and ii == 4


def test_unknown_single_stage_backup_checks_reduced_resource_wrap(monkeypatch):
    problem = _inputs(
        num_stages=1, latencies=[4, 4], iis=[1, 3],
        data_deps=[(0, 1, 0, 2)], buffer_sizes=[], memory_groups=[],
        buffer_min_versions=[], stage_horizon=1,
        issue_priority_pairs=[(0, 1)],
    )
    checks, instances = _scripted_solver(monkeypatch, [z3.unknown])
    checked = []

    def retain(ctx, assertions, model, k_vars, r_vars, buffers, pairs, **kwargs):
        assert all(z3.is_true(model.eval(hard, model_completion=False)) for hard in assertions)
        assert all(model.eval(stage).as_long() == 0 for stage in k_vars)
        forward = r_vars[1] - r_vars[0] >= 1
        wrap = r_vars[0] - r_vars[1] + 5 >= 3
        assert any(hard.eq(forward) for hard in assertions)
        assert any(hard.eq(wrap) for hard in assertions)
        # The distance-zero path removes the resource disjunction, but its
        # reverse gap must survive as an unconditional wrap constraint.
        assert not any("O_0_1" in str(hard) for hard in assertions)
        bad = z3.Model(ctx=ctx)
        for stage in k_vars:
            bad.update_value(stage, z3.IntVal(0, ctx))
        for residue, value in zip(r_vars, [0, 3]):
            bad.update_value(residue, z3.IntVal(value, ctx))
        assert z3.is_false(bad.eval(wrap, model_completion=False))
        assert all(z3.is_true(bad.eval(hard, model_completion=False))
                   for hard in assertions if not hard.eq(wrap))
        checked.append(1)
        return model

    monkeypatch.setattr(z3_scheduler, "_refine_loop_issue_order", retain)
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert starts == [0, 2] and counts == [] and ii == 5
    assert starts[1] - starts[0] >= problem["iis"][0]
    assert starts[0] - starts[1] + ii >= problem["iis"][1]
    assert checks and all(status == z3.unknown for status in checks)
    assert len(instances) == len(checks) and checked == [1]


@pytest.mark.parametrize("recalculate", [False, True])
def test_unknown_backup_keeps_distinct_negative_buffer_minima(monkeypatch, recalculate):
    negative_edges = [(2, 0, -1, 9), (1, 0, -2, 16)]
    problem = _inputs(
        num_stages=2, latencies=[6, 6, 6], iis=[1, 1, 1],
        resource_flags=[2, 16, 4],
        data_deps=[(0, 1, 0, 3), (1, 2, 0, 2), *negative_edges],
        resource_deps=[], buffer_sizes=[8, 32], memory_groups=[[112, 0, 1]],
        buffer_min_versions=[2, 3], recalculate_buffer_versions=recalculate,
    )
    checks, _ = _scripted_solver(monkeypatch, [z3.unknown])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert starts == [0, 3, 5] and counts == [2, 3] and ii == 7
    assert counts[1] > problem["num_stages"]
    assert starts[1] - starts[0] >= 3 and starts[2] - starts[1] >= 2
    assert sum(count * size for count, size in zip(counts, problem["buffer_sizes"])) == 112
    # Enumerate integer periods directly. Each listed edge owns its named
    # buffer; this oracle does not reuse the scheduler's negative-index or
    # ceiling-division helpers. Both rings need every supplied physical slot.
    legal_periods = [
        period for period in range(1, ii + 1)
        if all(0 <= start < period for start in starts)
        and all(starts[consumer] - starts[producer] + period * counts[buffer] >= latency
                for buffer, (producer, consumer, _, latency) in enumerate(negative_edges))
    ]
    assert legal_periods == [ii]
    for buffer, (producer, consumer, _, latency) in enumerate(negative_edges):
        assert starts[consumer] - starts[producer] + ii * counts[buffer] >= latency
        assert starts[consumer] - starts[producer] + ii * (counts[buffer] - 1) < latency
    assert checks and all(status == z3.unknown for status in checks)
