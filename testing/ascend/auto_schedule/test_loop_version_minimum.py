"""A shared physical ring cannot shrink when another owner is scheduled."""

import pytest

z3 = pytest.importorskip("z3")
from tilelang.ascend.transform import z3_scheduler


def _inputs(**changes):
    result = dict(
        num_stages=4, latencies=[2, 1], iis=[1, 1], resource_flags=[2, 1],
        data_deps=[(0, 1, 0, 2), (1, 0, -1, 1)], resource_deps=[],
        owner_exclusion_deps=[], pipe_order_deps=[], buffer_sizes=[32],
        memory_groups=[[128, 0]], enable_offset=True, stage_horizon=2,
        buffer_min_versions=[3],
    )
    result.update(changes)
    return result


def _check_dependencies(problem, starts, counts, ii):
    for producer, consumer, distance, latency in problem["data_deps"]:
        slots = counts[-distance - 1] if distance < 0 else distance
        assert starts[consumer] - starts[producer] >= latency - ii * slots


@pytest.mark.parametrize("recalculate", [True, False])
@pytest.mark.parametrize("prefetch", [[], [(0, 1)]])
@pytest.mark.parametrize("issue", [[], [(0, 1)]])
def test_existing_ring_survives_solving_refinement_and_recalculation(recalculate, prefetch, issue):
    problem = _inputs(recalculate_buffer_versions=recalculate, prefetch_priority_pairs=prefetch,
                      issue_priority_pairs=issue)
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert 3 <= counts[0] <= 4
    if recalculate:
        assert counts == [3]
    _check_dependencies(problem, starts, counts, ii)


def test_existing_ring_can_exceed_a_later_owners_new_version_bound():
    problem = _inputs(num_stages=1, stage_horizon=1, memory_groups=[[96, 0]])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert counts == [3]
    assert all(0 <= start < ii for start in starts)
    _check_dependencies(problem, starts, counts, ii)


def test_sequential_fallback_retains_the_physical_ring(monkeypatch):
    checks = []

    class UnknownSolver:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args):
            pass

        def add(self, *args):
            pass

        def check(self):
            checks.append(1)
            return z3.unknown

        def reason_unknown(self):
            return "test bounded resource exhaustion"

    monkeypatch.setattr(z3, "Solver", UnknownSolver)
    problem = _inputs(memory_groups=[[96, 0]])
    starts, counts, ii = z3_scheduler.z3_schedule_loop_python(**problem)
    assert checks
    assert counts == [3]
    _check_dependencies(problem, starts, counts, ii)


def test_minimum_capacity_failure_cannot_escape_through_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("capacity failure must precede solver queries")

    monkeypatch.setattr(z3, "Solver", forbidden)
    with pytest.raises(ValueError, match="minimum buffer versions"):
        z3_scheduler.z3_schedule_loop_python(**_inputs(memory_groups=[[64, 0]]))


@pytest.mark.parametrize("minimum", [[], [0], [-1], [True], [1.0], [1, 2]])
def test_invalid_minimum_versions_are_rejected(minimum):
    with pytest.raises(ValueError, match="one positive integer per buffer"):
        z3_scheduler.z3_schedule_loop_python(**_inputs(buffer_min_versions=minimum))


def test_ffi_preserves_the_common_ring_minimum(monkeypatch):
    observed = []

    def schedule(*args, **kwargs):
        observed.append(kwargs["buffer_min_versions"])
        return [0, 2], kwargs["buffer_min_versions"], 3

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", schedule)
    problem = _inputs()
    leading = ("num_stages", "latencies", "iis", "resource_flags", "data_deps",
               "resource_deps", "owner_exclusion_deps", "pipe_order_deps",
               "buffer_sizes", "memory_groups")
    # The registered PackedFunc accepts positional arguments, like C++.
    starts, counts, ii = z3_scheduler.z3_schedule_loop_ffi(
        *(problem[name] for name in leading), [], True, [], False, 2, [], [], (3,),
    )
    assert list(starts) == [0, 2] and list(counts) == [3] and int(ii) == 3
    assert observed == [[3]]
