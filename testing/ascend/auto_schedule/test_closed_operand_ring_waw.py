"""A closed native ready/MAD/release ring includes the matching slot's WAW.

Parent same-pipe issue order is never treated as completion. Each test expands
physical addresses and real events, checking every operand RAW/WAR/WAW before
asserting that the unrelated whole-pipe fence is absent.
"""

import pytest

from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower
from testing.ascend.auto_schedule.test_nested_local_protocols import _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace
from testing.ascend.auto_schedule.test_parent_versioned_operand_war import (
    _check_physical_lifetime, _fixture, _lower as _lower_assigned,
)
from testing.ascend.auto_schedule.test_staged_parent_operand_war import (
    _native_slot_trace, _ordinary_gemm,
)


def _mte1_barriers(mod):
    return [(call, loops) for call, _, loops in _trace(mod, core=2, lane=0)
            if _name(call) == "tl.ascend_pipe_barrier" and call.args[0].value == "PIPE_MTE1"]


@pytest.mark.parametrize("tile_k", [192, 256], ids=["odd-ring-phase", "even-ring-phase"])
def test_ordinary_gemm_slot_completion_covers_parent_load_waw(tile_k):
    _, snapshots = _lower(_ordinary_gemm(tile_k))
    physical = snapshots["tl.LowerScheduledTIR"]
    trace, _, _ = _native_slot_trace(physical)
    _prove_completions(trace)
    assert not _mte1_barriers(physical)


@pytest.mark.parametrize("stage,trips,side,load3d", [(0, 3, "both", False),
                                                  (1, 3, "a", True),
                                                  (2, 4, "b", True)])
def test_closed_operands_keep_all_physical_hazards_without_parent_barriers(stage, trips, side, load3d):
    before, facts = _fixture(side, trips, outer_stage=stage, load3d=load3d)
    physical = _lower_assigned(before)
    _check_physical_lifetime(physical, facts)
    # Wider owners remain unqualified; only the direct parent's middle
    # invocation is used for this no-barrier performance contract.
    middle = [record for record in _mte1_barriers(physical)
              if any(var.name == "row_part" and value == stage + 1
                     for var, value, _ in record[1])]
    assert not middle


@pytest.mark.parametrize("kind,versions", [("ordinary", 1), ("extra-reader", 2),
                                          ("partial-reader", 2), ("counter", 2)])
def test_missing_closed_operand_certificate_retains_parent_waw_fence(kind, versions):
    before, facts = _fixture("b", 3, kind=kind, outer_stage=1, versions=versions)
    physical = _lower_assigned(before)
    counts = _check_physical_lifetime(physical, facts)
    # The original full-MAD parent protocol can independently cover this WAW.
    # Its presence, rather than a newly omitted barrier, remains mandatory.
    for epoch in range(3):
        for owner in range(2):
            assert counts[epoch, owner, 2, "wait"] > 3
