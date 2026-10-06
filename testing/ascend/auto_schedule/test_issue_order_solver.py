"""Issue refinement preserves the existing feasible scheduling plan.

These durations are diagnostic ordering weights, not calibrated cycles.
"""

import pytest

z3 = pytest.importorskip('z3')
from tilelang.ascend.transform.z3_scheduler import _refine_loop_issue_order


def _plan(*, true_dependency=False, pipe_or_spr_order=False):
    # A remote publisher R supplies blocking consumer B; independent local
    # publisher P supplies remote reader C. P is already locally computable.
    # Original issue order: R, B, P, C. The preference is P before B.
    ctx = z3.Context()
    ii = 4
    k = [z3.Int(f'k{i}', ctx) for i in range(4)]
    r = [z3.Int(f'r{i}', ctx) for i in range(4)]
    versions = [z3.Int('versions', ctx)]
    hard = z3.Solver(ctx=ctx)
    for ki, ri in zip(k, r):
        hard.add(ki >= 0, ki < 2, ri >= 0, ri < ii)
    start = [ki * ii + ri for ki, ri in zip(k, r)]
    hard.add(versions[0] >= 1, versions[0] <= 2)
    hard.add(start[1] >= start[0] + 1)  # forward R -> B
    hard.add(start[3] >= start[2] + 1)  # forward P -> C
    # Single-slot reverse completion remains required before reusing P's slot.
    hard.add(start[2] - start[3] >= 1 - ii * versions[0])
    if true_dependency:
        hard.add(start[2] >= start[1] + 1)
    if pipe_or_spr_order:
        # This models an original same-pipe or special-register hard order.
        hard.add(r[2] >= r[1] + 1)
    chosen = z3.Solver(ctx=ctx)
    chosen.add(*hard.assertions())
    chosen.add(*[ki == 0 for ki in k], versions[0] == 2)
    chosen.add(*[ri == value for ri, value in zip(r, [0, 1, 2, 3])])
    assert chosen.check() == z3.sat
    baseline = chosen.model()
    refined = _refine_loop_issue_order(
        ctx, hard.assertions(), baseline, k, r, versions, [(2, 1)], buffer_counts=[1]
    )
    return ctx, ii, hard, baseline, refined, k, r, versions


def _value(model, var):
    return model.eval(var, model_completion=True).as_long()


def test_independent_handoff_publisher_precedes_blocking_consumer():
    _, _, _, _, model, _, r, _ = _plan()
    assert _value(model, r[2]) < _value(model, r[1])


def test_refinement_preserves_stage_slot_bounds_and_reverse_completion():
    _, ii, hard, before, after, k, r, versions = _plan()
    assert all(_value(before, var) == _value(after, var) for var in k)
    # The original solver may carry slack; the emitted plan already selected
    # one slot. Refinement must honor that actual allocation, not grow into two.
    assert _value(before, versions[0]) == 2
    assert _value(after, versions[0]) == 1
    assert all(z3.is_true(after.eval(assertion, model_completion=True))
               for assertion in hard.assertions())
    start = [_value(after, ki) * ii + _value(after, ri) for ki, ri in zip(k, r)]
    assert start[2] - start[3] >= 1 - ii * _value(after, versions[0])


@pytest.mark.parametrize('constraint', ['true_dependency', 'pipe_or_spr_order'])
def test_soft_preference_never_bypasses_existing_order(constraint):
    _, _, _, _, model, _, r, _ = _plan(**{constraint: True})
    assert _value(model, r[1]) < _value(model, r[2])


def test_no_preferences_keeps_original_model():
    ctx, _, hard, before, _, k, r, versions = _plan()
    after = _refine_loop_issue_order(ctx, hard.assertions(), before, k, r, versions, [])
    assert after is before
