"""A uniform outer stage preserves a native operand ring's completion proof.

These checks interpret materialized physical slots and emitted event edges;
no parent completion or cross-pipe source-order completion is assumed.
"""

import importlib.util
from pathlib import Path

import pytest
from tilelang import tvm

from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower
from testing.ascend.auto_schedule.test_nested_local_protocols import (
    _Command, _blocks, _prove_completions,
)
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _name, _region, _trace, _value,
)
from testing.ascend.auto_schedule.test_parent_versioned_operand_war import (
    _check_physical_lifetime, _fixture, _lower as _lower_assigned,
)


@pytest.mark.parametrize("stage,trips,side,load3d", [(1, 3, "both", False), (1, 4, "a", True),
                                                  (2, 3, "b", True), (2, 4, "both", False)])
def test_uniform_child_stage_preserves_slots_payload_and_finite_completion(stage, trips, side, load3d):
    before, facts = _fixture(side, trips, outer_stage=stage, load3d=load3d)
    counts = _check_physical_lifetime(_lower_assigned(before), facts)
    for epoch in range(3):
        for owner in range(2):
            # This invocation has no wider-owner boundary. Only actual W2
            # child-slot acquisitions/releases remain, even after fill/drain.
            assert counts[epoch, owner, stage + 1, "wait"] == trips
            assert counts[epoch, owner, stage + 1, "set"] == trips


@pytest.mark.parametrize("kind,versions", [("ordinary", 1), ("extra-reader", 2),
                                          ("partial-reader", 2), ("counter", 2)])
def test_staging_does_not_erase_unqualified_parent_obligations(kind, versions):
    before, facts = _fixture("b", 3, kind=kind, outer_stage=1, versions=versions)
    counts = _check_physical_lifetime(_lower_assigned(before), facts)
    for epoch in range(3):
        for owner in range(2):
            assert counts[epoch, owner, 2, "wait"] > 3
            assert counts[epoch, owner, 2, "set"] > 3


def test_guarded_shifted_child_keeps_the_existing_target_diagnostic():
    before, _ = _fixture("b", 3, kind="guarded-child", outer_stage=1)
    with pytest.raises(tvm.error.InternalError,
                       match="shifted carried events require an unconditional outermost lexical loop"):
        _lower_assigned(before)


def _ordinary_gemm(tile_k):
    path = Path(__file__).resolve().parents[3] / "examples/ascend/gemm_910b.py"
    spec = importlib.util.spec_from_file_location("_staged_gemm_factory", path)
    factory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(factory)
    return factory.gemm(128, 256, 3 * tile_k, tile_n=256, tile_k=tile_k)


def _native_slot_trace(mod):
    trace, mads, loads = [], [], []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        coords = {var.name: value for var, value, _ in loops}
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            src, dst = call.args[0].value.split("_")
            setting = name == "tl.ascend_set_flag"
            trace.append(_Command(src if setting else dst, "set" if setting else "wait",
                                  (call.args[0].value, _value(call.args[1], env))))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            src, _, _ = _region(call.args[0], env)
            dst, _, _ = _region(call.args[1], env)
            pipe = ("MTE2" if src.scope() == "global" else
                    "MTE1" if dst.scope() in ("shared.l0a", "shared.l0b") else "FIX")
            if pipe == "MTE1":
                loads.append((len(trace), coords["ki"], _blocks(call.args[1], env)))
            # C accumulation order is native MAD semantics; this oracle proves
            # operand/L1 completion and the final MAD->FIX fence separately.
            trace.append(_Command(pipe, "copy", reads=() if pipe == "FIX" else _blocks(call.args[0], env),
                                  writes=_blocks(call.args[1], env)))
        elif name == "tl.tileop.gemm":
            assert "unit_flag_ctrl" not in call.annotations
            mads.append((len(trace), coords["ki"], _value(call.args[9], env)))
            trace.append(_Command("M", "mad", reads=(*_blocks(call.args[0], env), *_blocks(call.args[1], env))))
        else:
            assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config"), name
            trace.append(_Command("S", "scalar"))
    return trace, mads, loads


@pytest.mark.parametrize("tile_k", [192, 256], ids=["odd-three-subtiles", "even-four-subtiles"])
def test_ordinary_gemm_keeps_slot_hazards_without_whole_panel_retirement(tile_k):
    _, snapshots = _lower(_ordinary_gemm(tile_k))
    trace, mads, loads = _native_slot_trace(snapshots["tl.LowerScheduledTIR"])
    require = _prove_completions(trace)  # Every actual L1/L0 RAW, WAR and WAW.
    assert len(mads) == 3 * (tile_k // 64)
    assert [clear for _, _, clear in mads].count(True) == 1 and mads[0][2]
    for epoch in (2, 3):
        previous = next(index for index, ki, _ in reversed(mads) if ki == epoch - 1)
        following, _, slots = next(record for record in loads if record[1] == epoch)
        assert not set(trace[previous].reads).intersection(slots), "Different slots may overlap"
        with pytest.raises(AssertionError, match="Missing completion path"):
            require(3 * previous + 2, 3 * following + 1)
    fix = next(index for index, command in enumerate(trace) if command.pipe == "FIX" and command.kind == "copy")
    require(3 * mads[-1][0] + 2, 3 * fix + 1)
