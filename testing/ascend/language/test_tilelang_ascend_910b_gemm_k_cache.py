"""Bounded A/K reuse uses ordinary pipeline, region, and completion contracts."""

from collections import Counter, defaultdict
from itertools import product

import pytest

import tilelang
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes
from testing.ascend.auto_schedule.test_owner_result_completion import _region, _value
from testing.ascend.language.test_tilelang_ascend_910b_gemm_cache_a import (
    _copies_from, _parents_of, example,
)
from testing.ascend.language.test_tilelang_ascend_910b_gemm_persistent import _environments


@pytest.mark.parametrize("a_cache_k", [0, -32, True, False, 32.0, "64", 48, 17])
def test_bounded_cache_requires_an_integral_number_of_compute_panels(example, a_cache_k):
    with pytest.raises(ValueError, match="a_cache_k must be None or a positive multiple of tile_k"):
        example.gemm(17, 33, 70, tile_k=32, a_cache_k=a_cache_k)


def test_full_k_and_bounded_a_cache_are_distinct_choices(example):
    with pytest.raises(ValueError, match="mutually exclusive"):
        example.gemm(17, 33, 70, tile_k=32, cache_a=True, a_cache_k=64)


@pytest.mark.parametrize("cache_a,num_cores", [(False, None), (True, None), (False, 3)])
def test_none_retains_the_existing_frontend_program(example, cache_a, num_cores):
    options = dict(m=17, n=33, k=70, tile_m=16, tile_n=16, tile_k=32,
                   cache_a=cache_a, num_cores=num_cores)
    assert tvm.ir.structural_equal(example.gemm(**options), example.gemm(**options, a_cache_k=None))


def _active_environments(program, call):
    parents = _parents_of(program, call)
    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    for env in _environments([parent for parent in parents if isinstance(parent, tirx.For)], definitions):
        # These frontend operations lie in the then arm of the global-K tail
        # guard. Evaluate its actual activation, not its unclipped raw region.
        if all(_value(parent.condition, env) for parent in parents if isinstance(parent, tirx.IfThenElse)):
            yield env


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["nn", "nt"])
@pytest.mark.parametrize("k,a_cache_k,num_cores", [
    (70, 64, None),       # Two cache epochs and an absent final B panel.
    (64, 64, 1),         # Exact cache boundary; persistent output traversal.
    (1, 128, 20),        # Cache wider than all K; only one compute panel.
    (97, 96, 5),         # Three B panels/cache; uneven persistent ownership.
    (70, 32, None),       # Legal one-panel bounded cache.
])
def test_actual_regions_cover_each_k_panel_once_and_reuse_a(example, dtype, transpose_b,
                                                           k, a_cache_k, num_cores):
    m, n, tile_m, tile_n, tile_k = 17, 33, 16, 16, 32
    program = example.gemm(m, n, k, dtype=dtype, transpose_b=transpose_b,
                           tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                           a_cache_k=a_cache_k, num_cores=num_cores)
    mod = tvm.IRModule({"main": program})
    a, b, c = [program.buffer_map[param] for param in program.params]
    a_load, = _copies_from(mod, a)
    b_load, = _copies_from(mod, b)
    copies = calls(program, "tl.tileop.ascend_copy") + calls(program, "tl.tileop.copy")
    export, = [call for call in copies if call.args[1].args[0].buffer.data.same_as(c.data)]
    gemm, = calls(program, "tl.tileop.gemm")
    pipelines = [loop for loop in nodes(program, tirx.For)
                 if bool(loop.annotations.get("tl.ascend_auto_pipeline", False))]
    assert len(pipelines) == 2
    pipeline, = [loop for loop in pipelines
                 if any(loop.same_as(parent) for parent in _parents_of(program, a_load))]
    assert int(pipeline.extent) == (k + a_cache_k - 1) // a_cache_k
    assert int(pipeline.annotations["num_stages"]) == 4
    assert any(parent.same_as(pipeline) for parent in _parents_of(program, a_load))
    panels_pipeline, = [loop for loop in pipelines if not loop.same_as(pipeline)]
    assert int(panels_pipeline.extent) == min(a_cache_k // tile_k, (k + tile_k - 1) // tile_k)
    assert int(panels_pipeline.annotations["num_stages"]) == 4
    assert not any(parent.same_as(panels_pipeline) for parent in _parents_of(program, a_load))
    assert any(parent.same_as(panels_pipeline) for parent in _parents_of(program, b_load))
    assert any(parent.same_as(panels_pipeline) for parent in _parents_of(program, gemm))
    for loop in nodes(program, tirx.For):
        assert not {"tl.manual_multi_buffer", "tl.manual_buffer_versions", "enable_offset"}.intersection(loop.annotations)

    outputs = set(product(range(0, m, tile_m), range(0, n, tile_n)))
    actual_exports = Counter(_region(export.args[1], env)[1]
                             for env in _active_environments(program, export))
    assert actual_exports == Counter({output: 1 for output in outputs})

    a_loads, b_loads, panels, clears, k_cells = Counter(), Counter(), Counter(), Counter(), Counter()
    for load, counts in ((a_load, a_loads), (b_load, b_loads)):
        for env in _active_environments(program, load):
            _, output, _ = _region(export.args[1], env)
            counts[output] += 1
    for env in _active_environments(program, gemm):
        a_cache, a_slice, a_extent = _region(gemm.args[0], env)
        b_panel, b_slice, b_extent = _region(gemm.args[1], env)
        accumulator, c_slice, c_extent = _region(gemm.args[2], env)
        _, a_origin, a_copy_extent = _region(a_load.args[0], env)
        _, b_origin, b_copy_extent = _region(b_load.args[0], env)
        _, output, _ = _region(export.args[1], env)
        assert tuple(int(v) for v in a_cache.shape) == (tile_m, a_cache_k)
        assert a_cache.scope() == b_panel.scope() == "shared.l1"
        assert accumulator.scope() == "shared.l0c" and accumulator.dtype == "float32"
        assert a_extent == (tile_m, tile_k) and a_copy_extent == (tile_m, a_cache_k)
        assert c_slice == (0, 0) and c_extent == (tile_m, tile_n)
        assert b_slice == (0, 0)
        assert b_extent == b_copy_extent == ((tile_n, tile_k) if transpose_b else (tile_k, tile_n))
        assert bool(gemm.args[4]) == transpose_b
        assert tuple(int(v) for v in gemm.args[5:8]) == (tile_m, tile_n, tile_k)
        global_k = a_origin[1] + a_slice[1]
        assert a_origin[0] == output[0] and a_slice[0] == 0
        assert b_origin == ((output[1], global_k) if transpose_b else (global_k, output[1]))
        assert 0 <= a_slice[1] and a_slice[1] + tile_k <= a_cache_k
        assert 0 <= global_k < k
        panels[output, global_k] += 1
        clears[output] += _value(gemm.args[9], env)
        assert _value(gemm.args[9], env) == int(global_k == 0)
        for coordinate in range(global_k, min(global_k + tile_k, k)):
            k_cells[output, coordinate] += 1
    assert a_loads == Counter({output: (k + a_cache_k - 1) // a_cache_k for output in outputs})
    assert b_loads == Counter({output: (k + tile_k - 1) // tile_k for output in outputs})
    assert panels == Counter({(output, panel): 1 for output in outputs for panel in range(0, k, tile_k)})
    assert clears == Counter({output: 1 for output in outputs})
    assert k_cells == Counter({(output, coordinate): 1 for output in outputs for coordinate in range(k)})


def test_bounded_cache_still_uses_the_target_capacity_contract(example):
    program = example.gemm(16, 16, 17, tile_m=32, tile_n=16, tile_k=256, a_cache_k=16384)
    # A singleton cache loop may inline before scheduling; final allocation
    # still enforces the same target capacity when no loop owns its versions.
    with pytest.raises(tvm.error.TVMError, match=r'shared\.l1.*capacity'):
        tilelang.lower(program, target={"kind": "ascend", "arch": "dav-2201"},
                       enable_host_codegen=False, enable_device_compile=False)


def _physical_cache_commands(actual):
    """Use exact typed regions, including partial DMA and L1 padding writes.

    Common region boundaries partition each physical buffer into disjoint
    cells. This keeps K=577's one-element tail exact without rounding a DMA
    write to a fictitious 16x16 block or depending on a logical loop name.
    """
    from testing.ascend.auto_schedule.test_nested_local_protocols import _Command
    from testing.ascend.auto_schedule.test_owner_result_completion import _name

    operations, cuts = [], defaultdict(lambda: defaultdict(set))
    for call, env, _ in actual:
        name = _name(call)
        reads, writes, flag = (), (), None
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            source, destination = call.args[0].value.split("_")
            kind = "set" if name == "tl.ascend_set_flag" else "wait"
            pipe = source if kind == "set" else destination
            flag = (call.args[0].value, _value(call.args[1], env))
            assert 0 <= flag[1] < 8
        elif name == "tl.ascend_pipe_barrier":
            pipe, kind = call.args[0].value.removeprefix("PIPE_"), "barrier"
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            source, destination = (_region(arg, env) for arg in call.args[:2])
            source_scope, destination_scope = source[0].scope(), destination[0].scope()
            if source_scope == "global":
                assert destination_scope == "shared.l1"
                pipe = "MTE2"
            elif destination_scope in ("shared.l0a", "shared.l0b"):
                assert source_scope == "shared.l1"
                pipe = "MTE1"
            else:
                assert source_scope == "shared.l0c" and destination_scope == "global"
                pipe = "FIX"
            kind, writes = "copy", (destination,)
            reads = () if pipe == "FIX" else (source,)
        elif name == "tl.tileop.fill":
            destination = _region(call.args[0], env)
            assert destination[0].scope() == "shared.l1"
            pipe, kind, writes = "MTE2", "fill", (destination,)
        elif name == "tl.tileop.gemm":
            assert "unit_flag_ctrl" not in call.annotations
            pipe, kind = "M", "mad"
            reads = tuple(_region(arg, env) for arg in call.args[:2])
            # Native C accumulation is checked separately by the actual
            # last-MAD -> FIX -> next-output first-MAD completion paths.
        else:
            assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_mmad_direction"), name
            pipe, kind = "S", "scalar"
        for buffer, origin, shape in (*reads, *writes):
            assert len(origin) == len(shape) == len(buffer.shape)
            for axis, (start, extent, bound) in enumerate(zip(origin, shape, buffer.shape)):
                assert 0 <= start <= start + extent <= _value(bound, env)
                cuts[buffer.data][axis].update((start, start + extent))
        operations.append((pipe, kind, flag, reads, writes))

    def cells(regions):
        result = []
        for buffer, origin, shape in regions:
            axes = [[cut for cut in sorted(cuts[buffer.data][axis]) if start <= cut < start + extent]
                    for axis, (start, extent) in enumerate(zip(origin, shape))]
            result.extend((buffer.data, *coordinate) for coordinate in product(*axes))
        return tuple(result)

    return [_Command(pipe, kind, flag, reads=cells(reads), writes=cells(writes))
            for pipe, kind, flag, reads, writes in operations]


@pytest.mark.parametrize("k,tile_m,tile_n,tile_k,a_cache_k,transpose_b,native_mads", [
    (576, 128, 256, 192, 384, True, 9),
    (577, 128, 256, 192, 384, True, 12),
    (1, 16, 16, 32, 128, False, 1),
    (70, 16, 16, 32, 64, False, 3),
], ids=["absent-final-panel", "physical-k-tail", "nn-cache-wider-than-k", "nn-panel-tail"])
def test_nested_cache_and_persistent_outputs_keep_physical_operand_completions(
        example, k, tile_m, tile_n, tile_k, a_cache_k, transpose_b, native_mads):
    from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower
    from testing.ascend.auto_schedule.test_nested_local_protocols import _prove_completions
    from testing.ascend.auto_schedule.test_owner_result_completion import _name, _trace

    # Three outputs, three native K64 fragments/B panel. The last cache epoch
    # either has an absent B panel (odd ring phase) or one valid K tail element.
    program = example.gemm(tile_m, 3 * tile_n, k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                           a_cache_k=a_cache_k, transpose_b=transpose_b, num_cores=1)
    _, snapshots = _lower(program)
    physical = snapshots["tl.LowerScheduledTIR"]
    blocks = [node for node in nodes(physical, tirx.AttrStmt)
              if node.attr_key == "thread_extent" and isinstance(node.node, tirx.IterVar)
              and node.node.thread_tag == "blockIdx.x"]
    assert len(blocks) == 1 and int(blocks[0].value) == 1
    actual = list(_trace(physical, core=2, lane=0))
    trace = _physical_cache_commands(actual)
    require = _prove_completions(trace)  # Actual operand/L1 RAW/WAR/WAW and binary reuse.
    mads, exports = {}, {}
    for index, (call, env, loops) in enumerate(actual):
        output = {var.name: value for var, value, _ in loops}.get("iteration")
        if _name(call) == "tl.tileop.gemm":
            assert output in range(3)
            mads.setdefault(output, []).append((index, bool(_value(call.args[9], env))))
        elif trace[index].pipe == "FIX" and trace[index].kind == "copy":
            assert output in range(3) and output not in exports
            exports[output] = index
    assert set(mads) == set(exports) == {0, 1, 2}
    for output in range(3):
        assert len(mads[output]) == native_mads
        assert [clear for _, clear in mads[output]] == [True] + [False] * (native_mads - 1)
        require(3 * mads[output][-1][0] + 2, 3 * exports[output] + 1)
        if output < 2:
            require(3 * exports[output] + 2, 3 * mads[output + 1][0][0] + 1)
