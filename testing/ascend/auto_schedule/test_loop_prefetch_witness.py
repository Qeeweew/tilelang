"""A local prefetch witness preserves the modulo plan's hard contract."""

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _problem(*, stages=(1, 1), residues=(1, 3), counts=(1,), capacity=64,
             num_stages=3, data=None, preferences=((0, 1),), ii=4):
    ctx = z3.Context()
    k = [z3.Int(f"k_{i}", ctx) for i in range(len(stages))]
    r = [z3.Int(f"r_{i}", ctx) for i in range(len(stages))]
    buffers = [z3.Int(f"buf_{i}", ctx) for i in range(len(counts))]
    starts = [stage * ii + residue for stage, residue in zip(k, r)]
    data = [(0, 1, 0, 1), (1, 0, -1, 1)] if data is None else data
    assertions = [value for stage, residue in zip(k, r)
                  for value in (stage >= 0, stage < 3, residue >= 0, residue < ii)]
    assertions += [value for buffer in buffers for value in (buffer >= 1, buffer <= num_stages)]
    assertions += [z3.Sum([buffer * 32 for buffer in buffers]) <= capacity]
    for u, v, distance, latency in data:
        versions = buffers[-distance - 1] if distance < 0 else distance
        assertions.append(starts[v] - starts[u] >= latency - ii * versions)
    model = z3.Model(ctx=ctx)
    for variables, values in ((k, stages), (r, residues), (buffers, counts)):
        for variable, value in zip(variables, values):
            model.update_value(variable, z3.IntVal(value, ctx))
    assert all(z3.is_true(model.eval(assertion)) for assertion in assertions)
    return dict(ctx=ctx, assertions=assertions, model=model, k_vars=k, r_vars=r,
                buffer_vars=buffers, prefetch_priority_pairs=preferences,
                buffer_sizes=[32] * len(counts), data_deps=data, ii=ii)


def _improve(problem):
    return z3_scheduler._improve_loop_prefetch_witness(**problem)


def _values(model, variables):
    return [model.eval(variable, model_completion=False).as_long() for variable in variables]


def test_readonly_load_lead_grows_two_slots_and_preserves_baseline(monkeypatch):
    problem = _problem()
    baseline = problem["model"].sexpr()

    def forbidden(*args, **kwargs):
        raise AssertionError("local witnesses must not query a solver")

    monkeypatch.setattr(z3.Solver, "check", forbidden)
    monkeypatch.setattr(z3.Optimize, "check", forbidden)
    candidate = _improve(problem)
    assert candidate is not problem["model"]
    assert _values(candidate, problem["k_vars"]) == [0, 1]
    assert _values(candidate, problem["r_vars"]) == [1, 3]
    assert _values(candidate, problem["buffer_vars"]) == [2]
    assert all(z3.is_true(candidate.eval(assertion)) for assertion in problem["assertions"])
    assert problem["model"].sexpr() == baseline


def test_predecessor_completion_blocks_earlier_load():
    # The desired load lead is useful, but moving task1 before its producer
    # completes violates a hard dependency, even if its destination has slots.
    problem = _problem(
        stages=(0, 1, 1), residues=(1, 0, 2),
        data=[(0, 1, 0, 3), (1, 2, 0, 1), (2, 1, -1, 1)], preferences=((1, 2),),
    )
    assert _improve(problem) is problem["model"]


def test_capacity_cannot_be_relaxed_for_useful_prefetch():
    problem = _problem(capacity=32)
    assert _improve(problem) is problem["model"]


def test_num_stages_cannot_be_relaxed_for_useful_prefetch():
    problem = _problem(num_stages=1)
    assert _improve(problem) is problem["model"]


def test_fixed_distance_wrap_constraint_remains_hard():
    problem = _problem(data=[(0, 1, 0, 1), (1, 0, -1, 1), (1, 0, 1, 1)])
    assert _improve(problem) is problem["model"]


def test_stage_order_constraint_remains_hard():
    problem = _problem()
    problem["assertions"].append(problem["k_vars"][1] <= problem["k_vars"][0])
    assert _improve(problem) is problem["model"]


def test_all_buffer_distance_edges_grow_their_required_versions():
    problem = _problem(
        stages=(1, 1, 2), residues=(1, 3, 0), counts=(1, 1), capacity=128,
        data=[(0, 1, 0, 1), (1, 0, -1, 1), (0, 2, 0, 1), (2, 0, -2, 1)],
    )
    candidate = _improve(problem)
    assert _values(candidate, problem["buffer_vars"]) == [2, 2]
    assert all(z3.is_true(candidate.eval(assertion)) for assertion in problem["assertions"])


def test_original_allocated_versions_are_not_shrunk():
    problem = _problem(counts=(3,), capacity=96)
    candidate = _improve(problem)
    assert _values(candidate, problem["k_vars"]) == [0, 1]
    assert _values(candidate, problem["buffer_vars"]) == [3]


def test_required_versions_use_exact_integer_ceil():
    ii = 2**54 + 1
    problem = _problem(ii=ii, residues=(0, ii - 1), counts=(2,), capacity=96,
                       data=[(0, 1, 0, 1), (1, 0, -1, 2)])
    candidate = _improve(problem)
    # After moving the producer, the reverse edge requires ceil((2*II+1)/II).
    # Float division can round that numerator to 2*II and incorrectly return 2.
    assert _values(candidate, problem["buffer_vars"]) == [3]
    stages = _values(candidate, problem["k_vars"])
    residues = _values(candidate, problem["r_vars"])
    starts = [stage * ii + residue for stage, residue in zip(stages, residues)]
    assert z3_scheduler._loop_buffer_versions(starts, problem["data_deps"], ii, [1]) == [3]


def test_version_recalculation_keeps_positive_slots_for_negative_numerators():
    # The first buffer edge needs no additional epoch distance. Physical
    # storage still has one slot, and the second incumbent keeps its 3 slots.
    assert z3_scheduler._loop_buffer_versions(
        [0, 100], [(0, 1, -1, 1), (0, 1, -2, 1)], 4, [1, 3],
    ) == [1, 3]
    assert z3_scheduler._loop_buffer_versions([0], [], 4, [0, -2]) == [1, 1]


def test_owner_forward_and_wrap_choices_are_cloned_without_mutation():
    problem = _problem()
    ctx, r = problem["ctx"], problem["r_vars"]
    owner = z3.Bool("Owner_0_1", ctx)
    marker = z3.Int("existing_auxiliary", ctx)
    problem["model"].update_value(owner, z3.BoolVal(True, ctx))
    problem["model"].update_value(marker, z3.IntVal(7, ctx))
    problem["assertions"] += [
        z3.Implies(owner, r[1] - r[0] >= 2),
        z3.Implies(owner, r[0] - r[1] + problem["ii"] >= 2),
        z3.Implies(z3.Not(owner), r[0] - r[1] >= 2),
        z3.Implies(z3.Not(owner), r[1] - r[0] + problem["ii"] >= 2), marker == 7,
    ]
    baseline = problem["model"].sexpr()
    candidate = _improve(problem)
    assert candidate is not problem["model"]
    assert z3.is_true(candidate.eval(owner, model_completion=False))
    assert candidate.eval(marker, model_completion=False).as_long() == 7
    assert all(z3.is_true(candidate.eval(assertion)) for assertion in problem["assertions"])
    assert problem["model"].sexpr() == baseline


def test_partial_model_is_retained_without_completion():
    problem = _problem()
    partial = z3.Model(ctx=problem["ctx"])
    for variable in problem["k_vars"] + problem["r_vars"]:
        partial.update_value(variable, problem["model"].eval(variable))
    problem["model"] = partial
    baseline = partial.sexpr()
    assert _improve(problem) is partial
    assert partial.sexpr() == baseline
    assert not z3.is_int_value(partial.eval(problem["buffer_vars"][0], model_completion=False))


def test_missing_owner_interpretation_cannot_be_completed_to_accept():
    problem = _problem()
    owner = z3.Bool("Owner_0_1", problem["ctx"])
    problem["assertions"].append(owner)
    assert _improve(problem) is problem["model"]
    assert not z3.is_true(problem["model"].eval(owner, model_completion=False))


def test_exchanging_useful_choices_without_strict_cost_improvement_is_rejected():
    # Moving task0 would gain (0,1), lose (2,0), and increase allocation.
    problem = _problem(
        stages=(1, 1, 0), residues=(1, 3, 0), counts=(1, 2), capacity=128,
        data=[(0, 1, 0, 1), (1, 0, -1, 1), (2, 0, 0, 1), (0, 2, -2, 1)],
        preferences=((0, 1), (2, 0)),
    )
    assert _improve(problem) is problem["model"]


def test_later_physical_issue_cannot_be_repaired_by_moving_only_stage():
    problem = _problem(stages=(0, 1), residues=(3, 1))
    assert _improve(problem) is problem["model"]


def test_one_scan_does_not_retry_an_already_useful_duplicate(monkeypatch):
    problem = _problem(preferences=((0, 1),) * 3)
    original = z3.Model
    copies = []

    def counted_model(*args, **kwargs):
        copies.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(z3, "Model", counted_model)
    candidate = _improve(problem)
    assert _values(candidate, problem["k_vars"]) == [0, 1]
    assert len(copies) == 1


def test_bounded_optimize_unknown_retains_the_improved_witness(monkeypatch):
    problem = _problem()
    checks = []
    limits = []

    class UnknownOptimize:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            limits.append(args)

        def add(self, *args):
            pass

        def minimize(self, *args):
            pass

        def check(self):
            checks.append(1)
            return z3.unknown

        def model(self):
            raise AssertionError("unknown models are not consumed by this change")

    monkeypatch.setattr(z3, "Optimize", UnknownOptimize)
    candidate = z3_scheduler._refine_loop_prefetch(**problem)
    assert checks == [1]
    assert limits == [("rlimit", 100000), ("timeout", 100)]
    assert _values(candidate, problem["k_vars"]) == [0, 1]
    assert _values(candidate, problem["buffer_vars"]) == [2]
    assert _values(problem["model"], problem["k_vars"]) == [1, 1]


def test_main_recalculates_witness_versions_before_handoff_freezes_them(monkeypatch):
    # The main search's first II=4 has a known model. Later queries return
    # unknown; neither search nor either refinement calls a native solver.
    checks, optimizers = [], []

    class SavedSolver:
        def __init__(self, *, ctx):
            self.ctx, self.hard = ctx, []

        def set(self, *args):
            pass

        def add(self, *args):
            self.hard.extend(args)

        def assertions(self):
            return self.hard

        def check(self):
            checks.append(1)
            return z3.sat if len(checks) == 1 else z3.unknown

        def reason_unknown(self):
            return "test bounded search"

        def model(self):
            model = z3.Model(ctx=self.ctx)
            for name, value in (("k_0", 1), ("k_1", 1), ("r_0", 1), ("r_1", 3), ("buf_0", 1)):
                model.update_value(z3.Int(name, self.ctx), z3.IntVal(value, self.ctx))
            assert all(z3.is_true(model.eval(assertion)) for assertion in self.hard)
            return model

    class UnknownOptimize:
        def __init__(self, *, ctx):
            self.ctx, self.hard, self.limits, self.checks = ctx, [], [], 0
            optimizers.append(self)

        def set(self, *args):
            self.limits.append(args)

        def add(self, *args):
            self.hard.extend(args)

        def minimize(self, *args):
            pass

        def check(self):
            self.checks += 1
            return z3.unknown

        def model(self):
            raise AssertionError("unknown candidates must not be read")

    monkeypatch.setattr(z3, "Solver", SavedSolver)
    monkeypatch.setattr(z3, "Optimize", UnknownOptimize)
    starts, versions, ii = z3_scheduler.z3_schedule_loop_python(
        num_stages=3, latencies=[3, 3], iis=[1, 1], resource_flags=[2, 1],
        data_deps=[(0, 1, 0, 1), (1, 0, -1, 1)], resource_deps=[],
        owner_exclusion_deps=[], pipe_order_deps=[], buffer_sizes=[32], memory_groups=[[64, 0]],
        enable_offset=True, stage_horizon=3,
        prefetch_priority_pairs=[(0, 1)], issue_priority_pairs=[(0, 1)],
    )
    assert ii == 4 and starts == [1, 7] and versions == [2]
    assert len(optimizers) == 2
    assert all(opt.checks == 1 and opt.limits == [("rlimit", 100000), ("timeout", 100)]
               for opt in optimizers)
    handoff = optimizers[1]
    selected = z3.Model(ctx=handoff.ctx)
    for name, value in (("k_0", 0), ("k_1", 1), ("r_0", 1), ("r_1", 3), ("buf_0", 2)):
        selected.update_value(z3.Int(name, handoff.ctx), z3.IntVal(value, handoff.ctx))
    assert all(z3.is_true(selected.eval(assertion)) for assertion in handoff.hard)
    selected.update_value(z3.Int("buf_0", handoff.ctx), z3.IntVal(1, handoff.ctx))
    assert any(z3.is_false(selected.eval(assertion)) for assertion in handoff.hard)
