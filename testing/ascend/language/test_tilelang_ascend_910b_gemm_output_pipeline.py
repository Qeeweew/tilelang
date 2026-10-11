"""Continuous output/K pipelining preserves coverage and full C completion."""
from collections import Counter
from itertools import product

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import calls, nodes
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.language.test_tilelang_ascend_910b_gemm_cache_a import example
from testing.ascend.language.test_tilelang_ascend_910b_gemm_k_cache import (
    _active_environments, _physical_cache_commands,
)


@pytest.mark.parametrize("options", [dict(pipeline_outputs=1), dict(pipeline_outputs=True),
                                    dict(pipeline_outputs=True, num_cores=2, cache_a=True),
                                    dict(pipeline_outputs=True, num_cores=2, a_cache_k=64)])
def test_continuous_pipeline_requires_explicit_streamed_traversal(example, options):
    with pytest.raises(ValueError, match="pipeline_outputs"):
        example.gemm(35, 53, 70, tile_m=16, tile_n=16, tile_k=32, **options)


@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("num_cores", [1, 5, 20])
def test_continuous_clock_covers_outputs_and_each_k_panel_once(example, transpose_b, num_cores):
    m, n, k = 35, 53, 70
    p = example.gemm(m, n, k, tile_m=16, tile_n=16, tile_k=32, num_cores=num_cores,
                     transpose_b=transpose_b, pipeline_outputs=True)
    a, b, c = [p.buffer_map[x] for x in p.params]
    copies = calls(p, "tl.tileop.ascend_copy") + calls(p, "tl.tileop.copy")
    load_a, = [x for x in copies if x.args[0].args[0].buffer.data.same_as(a.data)]
    load_b, = [x for x in copies if x.args[0].args[0].buffer.data.same_as(b.data)]
    export, = [x for x in copies if x.args[1].args[0].buffer.data.same_as(c.data)]
    mad, = calls(p, "tl.tileop.gemm")
    output_tiles = set(product(range(0, m, 16), range(0, n, 16)))
    exports = Counter(_region(export.args[1], env)[1] for env in _active_environments(p, export))
    assert exports == Counter({x: 1 for x in output_tiles})
    panels, clears = Counter(), Counter()
    for env in _active_environments(p,mad):
        _, ar, _ = _region(load_a.args[0], env)
        _, br, _ = _region(load_b.args[0], env)
        row, global_k = ar
        column = br[0] if transpose_b else br[1]
        assert br == ((column, global_k) if transpose_b else (global_k, column))
        panels[row, column, global_k] += 1
        clears[row, column] += _value(mad.args[9], env)
    assert panels == Counter({(*x, kk): 1 for x in output_tiles for kk in range(0, k, 32)})
    assert clears == Counter({x: 1 for x in output_tiles})
    automatic = [x for x in nodes(p, tirx.For) if bool(x.annotations.get("tl.ascend_auto_pipeline", False))]
    assert len(automatic) == 1


@pytest.mark.parametrize("transpose_b", [False, True], ids=["nn", "nt"])
def test_continuous_pipeline_retains_physical_operand_and_output_completion(example, transpose_b):
    from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower
    from testing.ascend.auto_schedule.test_nested_local_protocols import _prove_completions
    p = example.gemm(128, 5 * 256, 3 * 256, num_cores=2, transpose_b=transpose_b,
                     pipeline_outputs=True)
    _, snapshots = _lower(p)
    for block in range(2):
        def select(node):
            if (isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent"
                    and isinstance(node.node, tirx.IterVar) and node.node.thread_tag == "blockIdx.x"):
                return tirx.stmt_functor.substitute(
                    node.body, {node.node.var: tirx.const(block, node.node.var.dtype)})
        f = snapshots["tl.LowerScheduledTIR"]["main"]
        physical = tvm.IRModule({"main": f.with_body(tirx.stmt_functor.ir_transform(f.body, None, select))})
        actual = list(_trace(physical, core=2, lane=0))
        trace = _physical_cache_commands(actual)
        require = _prove_completions(trace)
        chains, pending = [], []
        for i, (call, env, _) in enumerate(actual):
            if _name(call) == "tl.tileop.gemm":
                pending.append((i, bool(_value(call.args[9], env))))
            if trace[i].pipe == "FIX" and trace[i].kind == "copy":
                assert len(pending) == 12 and [x[1] for x in pending] == [True] + [False] * 11
                require(3 * pending[-1][0] + 2, 3 * i + 1)
                if chains:
                    require(3 * chains[-1][1] + 2, 3 * pending[0][0] + 1)
                chains.append((pending, i))
                pending = []
        assert not pending and len(chains) == (5 - block + 1) // 2
        # Corrupt the first real operand release. Unit controls and source
        # issue order must never authorize this storage overwrite.
        first = chains[0][0][0][0]
        release = next(i for i in range(first + 1, len(trace))
                       if trace[i].kind == "set" and trace[i].flag[0] == "M_MTE1")
        corrupted = list(trace)
        corrupted.insert(first, corrupted.pop(release))
        with pytest.raises(AssertionError, match="Missing completion path"):
            _prove_completions(corrupted)
