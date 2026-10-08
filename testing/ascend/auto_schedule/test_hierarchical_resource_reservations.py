"""Atomic child issue summaries preserve data completion and physical FIFO.

All costs here are diagnostic weights, never measured hardware timings.
Finite traces enumerate real leaf issue intervals independently of the native
first/last summary. Native tests stop at AutoSchedule on the CPU.
"""

from inspect import signature
from unittest.mock import patch

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler as scheduler


# A child issues two B loads and interleaves MTE1/MMAD work. Its completion
# remains 30; only MTE2's resource span is [0,10), rather than [0,30).
_LEAVES = {
    0: {(0, 2): [(0, 2)], (1, 2): [(0, 2)]},
    1: {(0, 2): [(0, 2), (8, 10)], (1, 2): [(0, 2), (8, 10)],
        (0, 1): [(5, 6), (19, 20)], (0, 8): [(10, 14), (21, 25)]},
}


def _rows():
    return [(node, kind, bit, min(begin for begin, _ in intervals),
             max(end for _, end in intervals) - min(begin for begin, _ in intervals))
            for node, resources in _LEAVES.items()
            for (kind, bit), intervals in resources.items()]


def _inputs(**changes):
    values = dict(
        num_stages=3, latencies=[2, 30], iis=[2, 30], resource_flags=[2, 11],
        data_deps=[(0, 1, 0, 2), (1, 0, -1, 20)], resource_deps=[(0, 1)],
        owner_exclusion_deps=[], pipe_order_deps=[], buffer_sizes=[128 * 1024],
        # Child B already owns two 128KiB slots; they are not a new variable.
        memory_groups=[[256 * 1024, 0]], enable_offset=True, stage_horizon=3,
        resource_reservations=_rows(),
        resource_hbm_masks=[2, 2],
    )
    values.update(changes)
    return values


def _profile_plan(ii, stages, residues, *, rows=None, flags=None, iis=None, pairs=None):
    ctx = z3.Context()
    rows = _rows() if rows is None else rows
    count = len(flags or [2, 11])
    hbm_masks = [sum(bit for index, kind, bit, _, _ in rows if index == node and kind == 1)
                 for node in range(count)]
    profiles = scheduler._loop_resource_profiles(rows, flags or [2, 11], iis or [2, 30], hbm_masks)
    k = [z3.Int(f"k_{i}", ctx) for i in range(len(profiles))]
    r = [z3.Int(f"r_{i}", ctx) for i in range(len(profiles))]
    hard, _ = scheduler._loop_profile_constraints(ctx, ii, k, r, profiles,
                                                [(0, 1)] if pairs is None else pairs)
    solve = z3.Solver(ctx=ctx)
    solve.add(*hard, *[var == value for var, value in zip(k, stages)],
              *[var == value for var, value in zip(r, residues)])
    return solve, k, r


def _finite_issues(stages, residues, ii, epochs=4):
    order = sorted(range(2), key=lambda node: (residues[node], -stages[node], node))
    streams = {}
    for physical in range(epochs):
        for node in order:
            for key, intervals in _LEAVES[node].items():
                for begin, end in intervals:
                    streams.setdefault(key, []).append(
                        (physical * ii + residues[node] + begin,
                         physical * ii + residues[node] + end))
    # Source-order commands on each physical resource must remain FIFO and
    # nonoverlapping, including every wrap and the same child's next epoch.
    for stream in streams.values():
        assert all(end <= next_begin for (_, end), (next_begin, _) in zip(stream, stream[1:]))


def test_finite_child_reservations_keep_completion_and_two_slot_release():
    solve, _, _ = _profile_plan(24, [0, 1], [0, 2])
    assert solve.check() == z3.sat
    _finite_issues([0, 1], [0, 2], 24)
    # The last A reader is still at child+20. A single slot would overwrite
    # it at 24; two slots wait until 48. Child completion is still child+30.
    child = 24 + 2
    assert child + 20 <= 2 * 24 and child + 20 > 24
    assert child + 30 > 2 * 24


def test_main_solver_consumes_profiles_without_replacing_completion_edges():
    values = _inputs(prefetch_priority_pairs=[(0, 1)])
    starts, widths, ii = scheduler.z3_schedule_loop_python(**values)
    assert ii < sum(values["iis"]) and widths == [2]
    assert starts[1] - starts[0] >= 2
    assert starts[0] + widths[0] * ii >= starts[1] + 20
    _finite_issues([start // ii for start in starts], [start % ii for start in starts], ii)


@pytest.mark.parametrize("ii,stages,residues", [
    (24, [0, 1], [6, 2]),  # A fits a paper hole, but cannot splice into child.
    (11, [0, 0], [0, 2]),  # Child's final DMA collides with the next A.
    (14, [0, 0], [0, 2]),  # The child's own Cube span crosses its next epoch.
])
def test_forward_wrap_and_self_collision_are_hard(ii, stages, residues):
    solve, _, _ = _profile_plan(ii, stages, residues)
    assert solve.check() == z3.unsat


def test_touching_intervals_and_common_stage_translation_are_legal():
    for stages in ([0, 0], [4, 4]):
        solve, _, _ = _profile_plan(15, stages, [0, 2])
        assert solve.check() == z3.sat
        _finite_issues(stages, [0, 2], 15)


def test_self_collision_is_checked_without_another_parent_node():
    solve, _, _ = _profile_plan(14, [0], [0], rows=[(0, 0, 8, 10, 15)],
                                flags=[8], iis=[30], pairs=[])
    assert solve.check() == z3.unsat


@pytest.mark.parametrize("stages,expected", [([1, 0], z3.sat), ([0, 1], z3.unsat)])
def test_equal_residue_uses_descending_stage_atomic_source_order(stages, expected):
    rows = [(0, 0, 2, 0, 2), (1, 0, 2, 5, 5)]
    solve, _, _ = _profile_plan(20, stages, [0, 0], rows=rows, flags=[2, 2], iis=[2, 10])
    assert solve.check() == expected


def test_hbm_contention_is_independent_of_pipe_namespace():
    rows = [(0, 0, 2, 0, 2), (0, 1, 2, 0, 2),
            (1, 0, 32, 5, 1), (1, 1, 2, 0, 6)]
    solve, _, _ = _profile_plan(20, [0, 0], [0, 1], rows=rows, flags=[2, 32], iis=[2, 6])
    assert solve.check() == z3.unsat  # Different pipes still overlap AIC HBM.


def test_opaque_resource_pair_retains_its_original_envelope():
    starts, widths, ii = scheduler.z3_schedule_loop_python(**_inputs(
        resource_flags=[2, 8],
        resource_reservations=[(0, 0, 2, 0, 2), (1, 0, 8, 5, 15)],
        resource_hbm_masks=[0, 0],
    ))
    assert ii >= 32  # The supplied conflict has no known shared resource key.
    assert starts[0] + widths[0] * ii >= starts[1] + 20


def test_leaf_issue_end_does_not_authorize_early_read_or_slot_overwrite():
    starts, widths, ii = scheduler.z3_schedule_loop_python(**_inputs(
        num_stages=2, latencies=[9, 1], iis=[2, 1], resource_flags=[2, 8],
        data_deps=[(0, 1, 0, 9), (1, 0, -1, 1)], resource_deps=[],
        stage_horizon=2, buffer_sizes=[32], memory_groups=[[64, 0]],
        resource_reservations=[(0, 0, 2, 0, 2), (1, 0, 8, 0, 1)],
        resource_hbm_masks=[0, 0],
    ))
    assert starts[1] >= starts[0] + 9
    assert starts[0] + widths[0] * ii >= starts[1] + 1


@pytest.mark.parametrize("changes", [dict(manual_schedule=True), dict(enable_offset=False)])
def test_optional_profiles_do_not_reinterpret_explicit_schedule_controls(changes):
    with pytest.raises(ValueError, match="automatic offset pipeline"):
        scheduler.z3_schedule_loop_python(**_inputs(**changes))


@pytest.mark.parametrize("masks", [None, [2], [2, 8]])
def test_summary_requires_exact_hbm_masks(masks):
    with pytest.raises(ValueError, match="complete native HBM masks"):
        scheduler.z3_schedule_loop_python(**_inputs(resource_hbm_masks=masks))


def test_missing_hbm_span_cannot_erase_a_shared_port_conflict():
    values = _inputs(
        latencies=[2, 20], iis=[2, 20], resource_flags=[2, 34], data_deps=[],
        resource_reservations=[(0, 0, 2, 0, 2), (0, 1, 2, 0, 2),
                               (1, 0, 2, 0, 2), (1, 0, 32, 15, 2)],
        resource_hbm_masks=[2, 6],
    )
    with pytest.raises(ValueError, match="hierarchical HBM"):
        scheduler.z3_schedule_loop_python(**values)
    values["resource_reservations"] += [(1, 1, 2, 0, 17), (1, 1, 4, 15, 2)]
    _, _, ii = scheduler.z3_schedule_loop_python(**values)
    assert ii >= 19


@pytest.mark.parametrize("change", [
    lambda rows: rows[:-1],  # Incomplete resource masks cannot erase a pipe.
    lambda rows: rows + [rows[0]],
    lambda rows: [(0, 0, 3, 0, 2)] + rows[1:],
    lambda rows: [(0, 0, 2, -1, 2)] + rows[1:],
    lambda rows: [(0, 0, 2, 0, 0)] + rows[1:],
    lambda rows: [(0, 2, 2, 0, 2)] + rows[1:],
    lambda rows: [(0, 1, 8, 0, 2)] + rows[1:],
    lambda rows: [(0, 0, 2, 0, 3)] + rows[1:],
])
def test_malformed_or_incomplete_summary_is_rejected(change):
    with pytest.raises(ValueError, match="hierarchical"):
        scheduler.z3_schedule_loop_python(**_inputs(resource_reservations=change(_rows())))


def test_empty_profiles_retain_the_legacy_solver_path():
    values = _inputs(resource_reservations=None, prefetch_priority_pairs=[(0, 1)])
    baseline = scheduler.z3_schedule_loop_python(**values)
    values["resource_reservations"] = []
    assert scheduler.z3_schedule_loop_python(**values) == baseline


def test_unknown_search_uses_a_checked_envelope_witness(monkeypatch):
    class Exhausted:
        def __init__(self, **kwargs):
            pass

        def set(self, *args):
            pass

        def add(self, *args):
            pass

        def check(self):
            return z3.unknown

        def reason_unknown(self):
            return "focused resource exhaustion"

        def model(self):
            raise AssertionError("unknown must never supply a model")

    monkeypatch.setattr(z3, "Solver", Exhausted)
    starts, widths, ii = scheduler.z3_schedule_loop_python(**_inputs())
    assert starts[1] - starts[0] >= 2
    assert starts[0] + widths[0] * ii >= starts[1] + 20
    _finite_issues([start // ii for start in starts], [start % ii for start in starts], ii)


def test_actual_child_ring_is_a_capacity_reservation():
    # With child B2 fixed, A3 exceeds L1 even if its one-slot declaration fits.
    with pytest.raises(ValueError, match="minimum buffer versions"):
        scheduler.z3_schedule_loop_python(**_inputs(buffer_min_versions=[3]))


def test_final_validation_rejects_a_refiner_that_breaks_atomic_fifo(monkeypatch):
    def malicious(ctx, assertions, model, k, r, buffers, pairs, **kwargs):
        bad = z3.Model(ctx=ctx)
        for declaration in model.decls():
            bad.update_value(declaration, model[declaration])
        bad.update_value(r[1], model.eval(r[0], model_completion=False))
        return bad

    monkeypatch.setattr(scheduler, "_refine_loop_issue_order", malicious)
    with pytest.raises(RuntimeError, match="hard scheduling constraint"):
        scheduler.z3_schedule_loop_python(**_inputs(issue_priority_pairs=[(0, 1)]))


def _native_fixture(kind):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import copy, kernel, seq
    from testing.ascend.auto_schedule._scheduled_ir import unit

    elements = 32768  # 128KiB per L1 slot.
    a = tirx.decl_buffer((4, elements), "float32", name="A")
    b = tirx.decl_buffer((4, 2, elements), "float32", name="B")
    al1 = tirx.decl_buffer((elements,), "float32", name="a_l1", scope="shared.l1")
    bl1 = tirx.decl_buffer((elements,), "float32", name="b_l1", scope="shared.l1")
    al0 = tirx.decl_buffer((32,), "float32", name="a_l0", scope="shared.l0a")
    bl0 = tirx.decl_buffer((32,), "float32", name="b_l0", scope="shared.l0b")
    epoch, panel = tirx.Var("epoch", "int32"), tirx.Var("panel", "int32")
    params = [a, b]
    minimum = epoch * 2 if kind == "symbolic-min" else 0
    if kind == "memory-min":
        cut = tirx.decl_buffer((1,), "int32", name="cut")
        params.append(cut)
        minimum = cut[0]
    ordinal = panel - minimum
    stage = 0 if kind == "manual" else -1
    guard = panel % 2 == 0 if kind == "guarded" else None
    child_tasks = [
        unit(copy(b, bl1, src_indices=[epoch, ordinal, 0], src_shape=[1, 1, elements]),
             core=2, stage=stage, cost=(2, 1), guard=guard),
        unit(copy(al1, al0, src_indices=[ordinal * 32], src_shape=[32]),
             core=2, stage=stage, cost=(6, 1), guard=guard),
        unit(copy(bl1, bl0, src_shape=[32]), core=2, stage=stage, cost=(6, 1), guard=guard),
    ]
    child_body = seq(*child_tasks)
    extent = 2
    if kind == "dynamic":
        trips = tirx.Var("trips", "int32")
        params.append(trips)
        extent = tirx.max(0, tirx.min(trips, 2))
    child = unit(tirx.For(panel, minimum, extent, tirx.ForKind.SERIAL, child_body,
                         annotations={"tl.ascend_auto_pipeline": True,
                                      "num_stages": 4, "multi_buffer_eligible": [bl1.data]}),
                 core=None, stage=-1)
    outer = unit(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, seq(
        unit(copy(a, al1, src_indices=[epoch, 0], src_shape=[1, elements]),
             core=2, stage=-1, cost=(2, 1)), child),
        annotations={"tl.ascend_auto_pipeline": True, "num_stages": 4,
                     "multi_buffer_eligible": [al1.data]}), core=None, stage=-1)
    mod = kernel(outer, buffers=[al1, bl1, al0, bl0], params=params)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({
        "kind": "ascend", "arch": "dav-3510" if kind == "3510" else "dav-2201"}))
    return mod, al1, bl1


@pytest.mark.parametrize("kind", ["eligible", "symbolic-min", "memory-min", "guarded", "dynamic", "manual", "3510"])
def test_native_static_child_summary_and_conservative_fallback(kind):
    from tilelang.ascend import transform
    from tvm import tirx
    from testing.ascend._ir import nodes

    before, al1, bl1 = _native_fixture(kind)
    seen = []
    original = scheduler.z3_schedule_loop_python

    def observe(*args, **kwargs):
        inputs = dict(signature(original).bind(*args, **kwargs).arguments)
        result = original(*args, **kwargs)
        seen.append((inputs, result))
        return result

    with patch.object(scheduler, "z3_schedule_loop_python", side_effect=observe):
        after = transform.AutoSchedule()(before)
    outer, result = seen[-1]
    rows = outer["resource_reservations"]
    if kind not in ("eligible", "symbolic-min"):
        assert not rows
        if kind == "guarded":
            from tilelang import tvm

            loop, = [loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "panel"]
            guards = nodes(loop.body, tirx.IfThenElse)
            assert len(guards) == 3
            # These are real activation guards, not a condition proved true
            # while visiting the malformed outside-all-markers fixture.
            for guard in guards:
                active = [tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(
                    guard.condition, {loop.loop_var: tirx.const(value, loop.loop_var.dtype)}))
                          for value in (0, 1)]
                assert [int(value) for value in active] == [1, 0]
        return
    assert rows and outer["latencies"][1] == outer["iis"][1]
    mte2 = [row for row in rows if row[:3] == (1, 0, 2)]
    assert len(mte2) == 1 and mte2[0][4] < outer["iis"][1]
    assert any(row[:3] == (1, 1, 2) for row in rows)  # AIC HBM is retained.
    assert outer["resource_hbm_masks"] == [2, 2]
    assert outer["prefetch_priority_pairs"] == [(0, 1)]
    versions = after["main"].body.block.annotations["tl.buffer_versions_map"]
    a_slots, b_slots = int(versions[al1.data]), int(versions[bl1.data])
    assert b_slots == 2 and a_slots * 128 * 1024 + b_slots * 128 * 1024 <= 512 * 1024
    group, = [group for group in outer["memory_groups"] if 0 in group[1:]]
    assert group[0] == 512 * 1024 - b_slots * 128 * 1024
    assert result[1] == [a_slots]
    loop, = [loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "epoch"]
    emitted = list(loop.body.seq)
    roles = [1 if nodes(unit.body, tirx.For) else 0 for unit in emitted]
    starts, _, ii = result
    expected = sorted(range(2), key=lambda node: (starts[node] % ii, -(starts[node] // ii), node))
    assert roles == expected
    assert [int(unit.node["stage"]) for unit in emitted] == [starts[node] // ii for node in expected]
