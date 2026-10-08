"""CPU contracts for round-robin GEMM output ownership and automatic K loops."""

from collections import Counter
from itertools import product

import pytest

import tilelang
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import nodes, statements
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _PURE_DEFINITIONS, _SCALAR_VALUES, _region, _value,
)
from testing.ascend.language.test_tilelang_ascend_910b_gemm_cache_a import example


@pytest.mark.parametrize("num_cores", [0, -1, True, False, 1.5, "20"])
def test_persistent_core_count_requires_a_positive_integer(example, num_cores):
    with pytest.raises(ValueError, match="num_cores must be None or a positive integer"):
        example.gemm(35, 53, 70, num_cores=num_cores)


def _environments(loops, definitions, env=None):
    env = {_PURE_DEFINITIONS: definitions, _SCALAR_VALUES: {}} if env is None else env
    if not loops:
        yield env
        return
    loop, *remaining = loops
    start, extent = _value(loop.min, env), _value(loop.extent, env)
    assert extent > 0, "The launch must not contain an empty output owner"
    assert loop.step is None or _value(loop.step, env) == 1
    values = range(start, start + extent)
    if loop.thread_binding is not None and loop.thread_binding.thread_tag != "blockIdx.x":
        # Frontend launch lanes do not create another logical Cube output.
        values = (start,)
    for value in values:
        yield from _environments(remaining, definitions,
                                 {**env, loop.loop_var: tirx.const(value, loop.loop_var.dtype)})


@pytest.mark.parametrize("m,n,num_cores", [(35, 53, 1), (35, 53, 5),
                                          (35, 53, 12), (35, 53, 13),
                                          (16, 16, 8)],
                         ids=["one-owner", "uneven", "exact-grid", "overprovisioned", "single-tile"])
@pytest.mark.parametrize("cache_a,transpose_b", [(False, True), (True, False)],
                         ids=["streamed-nt", "cached-nn"])
def test_each_actual_output_region_has_exactly_one_nonempty_owner(example, m, n, num_cores,
                                                                  cache_a, transpose_b):
    program = example.gemm(m, n, 70, dtype="float16", tile_m=16, tile_n=16, tile_k=32,
                           num_cores=num_cores, cache_a=cache_a, transpose_b=transpose_b)
    launch, = [loop for loop in nodes(program, tirx.For)
               if loop.thread_binding is not None and loop.thread_binding.thread_tag == "blockIdx.x"]
    expected_tiles = set(product(range(0, m, 16), range(0, n, 16)))
    assert int(launch.extent) == min(len(expected_tiles), num_cores)
    output = program.buffer_map[program.params[2]]
    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    actual, cells, owners = Counter(), Counter(), Counter()
    for stmt, parents in statements(program):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        call = stmt.value
        if not isinstance(call.op, tvm.ir.Op) or call.op.name not in ("tl.tileop.copy", "tl.tileop.ascend_copy"):
            continue
        destination = call.args[1].args[0].buffer
        if not destination.data.same_as(output.data):
            continue
        # No dynamic tile-id tail branch may guard the complete K pipeline.
        assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)
        loops = [parent for parent in parents if isinstance(parent, tirx.For)]
        for env in _environments(loops, definitions):
            _, origin, shape = _region(call.args[1], env)
            assert shape == (16, 16) and origin in expected_tiles
            actual[origin] += 1
            owners[_value(launch.loop_var, env)] += 1
            for row in range(origin[0], min(origin[0] + shape[0], m)):
                for column in range(origin[1], min(origin[1] + shape[1], n)):
                    cells[row, column] += 1
    assert actual == Counter({tile: 1 for tile in expected_tiles})
    assert cells == Counter({cell: 1 for cell in product(range(m), range(n))})
    assert set(owners) == set(range(int(launch.extent)))
    assert max(owners.values()) - min(owners.values()) <= 1
    pipeline, = [loop for loop in nodes(program, tirx.For)
                 if bool(loop.annotations.get("tl.ascend_auto_pipeline", False))]
    assert int(pipeline.annotations["num_stages"]) == 4
    assert len([loop for loop in nodes(program, tirx.For)
                if loop.kind == tirx.ForKind.SERIAL and not loop.annotations]) == 1
    assert not nodes(program, tirx.IfThenElse)


def test_none_keeps_the_original_nonpersistent_launch_and_lowered_source(example):
    options = dict(m=128, n=256, k=512, dtype="bfloat16")
    default, explicit = example.gemm(**options), example.gemm(**options, num_cores=None)
    assert tvm.ir.structural_equal(default, explicit)
    assert len([loop for loop in nodes(default, tirx.For) if loop.kind == tirx.ForKind.SERIAL]) == 1
    target = {"kind": "ascend", "arch": "dav-2201"}
    sources = [tilelang.lower(program, target=target, enable_host_codegen=False,
                              enable_device_compile=False).kernel_source
               for program in (default, explicit)]
    assert sources[0] == sources[1]


@pytest.mark.parametrize("cache_a", [False, True], ids=["streamed-a", "reload-cached-a"])
def test_persistent_outputs_retire_fix_before_c_is_cleared_again(example, cache_a):
    from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower
    from testing.ascend.auto_schedule.test_nested_local_protocols import _prove_completions
    from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace
    from testing.ascend.auto_schedule.test_staged_parent_operand_war import _native_slot_trace

    # Three outputs per physical core, each with 3 panels x 3 native K64
    # MMADs. The odd operand count also changes the ring's slot phase between
    # outputs; the accumulator is reused only after its actual FIX completes.
    program = example.gemm(128, 768, 576, tile_n=256, tile_k=192, num_cores=1, cache_a=cache_a)
    _, snapshots = _lower(program)
    physical = snapshots["tl.LowerScheduledTIR"]
    trace, _, _ = _native_slot_trace(physical)
    require = _prove_completions(trace)  # All physical L1/L0A/L0B hazards and binary seed/drain.
    actual = list(_trace(physical, core=2, lane=0))
    assert len(actual) == len(trace)
    mads, exports, root_ring = {}, {}, Counter()
    for index, (call, env, loops) in enumerate(actual):
        coordinates = {var.name: value for var, value, _ in loops}
        output = coordinates.get("iteration")
        if _name(call) == "tl.tileop.gemm":
            assert output in range(3)
            mads.setdefault(output, []).append((index, bool(_value(call.args[9], env))))
        elif trace[index].pipe == "FIX" and trace[index].kind == "copy":
            assert output in range(3) and output not in exports
            exports[output] = index
        elif _name(call) in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and output is None:
            if call.args[0].value == "M_MTE1":
                root_ring[_name(call), _value(call.args[1], env)] += 1
    assert set(mads) == set(exports) == {0, 1, 2}
    for output in range(3):
        assert len(mads[output]) == 9
        assert [clear for _, clear in mads[output]] == [True] + [False] * 8
        # Native M accumulation uses its established same-pipe ordering.
        # The risk introduced here is across output epochs and across pipes:
        # last MAD must finish before FIX, and FIX before the next C clear.
        require(3 * mads[output][-1][0] + 2, 3 * exports[output] + 1)
        if output < 2:
            require(3 * exports[output] + 2, 3 * mads[output + 1][0][0] + 1)
    for slot in (0, 1):
        assert root_ring["tl.ascend_set_flag", slot] == 1
        assert root_ring["tl.ascend_wait_flag", slot] == 1
