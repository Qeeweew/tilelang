"""CPU contracts for Python-level A/K caching with automatic GEMM pipelining."""

import importlib.util
from pathlib import Path
import sys

import pytest

import tilelang
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes, statements


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/ascend/gemm_910b.py"
    spec = importlib.util.spec_from_file_location("_tilelang_gemm_910b_cache_a", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frontend(example, **changes):
    arguments = dict(m=17, n=33, k=70, dtype="float16", tile_m=16, tile_n=16, tile_k=32)
    arguments.update(changes)
    program = example.gemm(**arguments)
    mod = tvm.IRModule({"main": program})
    return tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)


def _region(operand):
    assert operand.op.name == "tl.region"
    return operand.args[0].buffer, operand.args[0].indices, operand.args[2:]


def _copies_from(mod, buffer):
    copies = calls(mod, "tl.tileop.ascend_copy") + calls(mod, "tl.tileop.copy")
    return [call for call in copies if _region(call.args[0])[0].data.same_as(buffer.data)]


def _parents_of(mod, call):
    return next(parents for stmt, parents in statements(mod) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call))


def _pipeline(mod):
    (loop,) = [loop for loop in nodes(mod, tirx.For) if bool(loop.annotations.get("tl.ascend_auto_pipeline", False))]
    assert int(loop.annotations["num_stages"]) == 4
    assert int(loop.extent) == 3
    assert not {"tl.manual_multi_buffer", "tl.manual_buffer_versions", "enable_offset"}.intersection(loop.annotations)
    return loop


@pytest.mark.parametrize("cache_a", [None, 0, 1, "yes"])
def test_cache_a_requires_boolean_algorithm_choice(example, cache_a):
    with pytest.raises(ValueError, match="cache_a must be a bool"):
        example.gemm(17, 33, 70, cache_a=cache_a)


@pytest.mark.parametrize("arguments,message", [
    ({"m": 0}, "m must be a positive integer"),
    ({"k": True}, "k must be a positive integer"),
    ({"tile_k": 24}, "tile_k must be a positive multiple of 16"),
    ({"dtype": "float32"}, "requires FP16 or BF16"),
    ({"transpose_b": 1}, "transpose_b must be a bool"),
])
def test_cache_preserves_existing_input_boundaries(example, arguments, message):
    options = dict(m=17, n=33, k=70, cache_a=True)
    options.update(arguments)
    with pytest.raises(ValueError, match=message):
        example.gemm(**options)


def test_default_is_explicit_non_cached_algorithm(example):
    default = example.gemm(17, 33, 70, tile_m=16, tile_n=16, tile_k=32)
    explicit = example.gemm(17, 33, 70, tile_m=16, tile_n=16, tile_k=32, cache_a=False)
    assert tvm.ir.structural_equal(default, explicit)
    mod = _frontend(example, cache_a=False)
    loop = _pipeline(mod)
    a = mod["main"].buffer_map[mod["main"].params[0]]
    (load,) = _copies_from(mod, a)
    assert any(parent.same_as(loop) for parent in _parents_of(mod, load))
    assert tuple(int(value) for value in _region(load.args[1])[0].shape) == (16, 32)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["nn", "nt"])
def test_cache_is_one_prologue_load_and_k_slices_in_ordinary_auto_loop(example, dtype, transpose_b):
    mod = _frontend(example, dtype=dtype, transpose_b=transpose_b, cache_a=True)
    loop = _pipeline(mod)
    a, b = [mod["main"].buffer_map[param] for param in mod["main"].params[:2]]
    (a_load,) = _copies_from(mod, a)
    (b_load,) = _copies_from(mod, b)
    assert not any(parent.same_as(loop) for parent in _parents_of(mod, a_load))
    assert any(parent.same_as(loop) for parent in _parents_of(mod, b_load))
    cache, cache_origin, cache_extent = _region(a_load.args[1])
    assert cache.scope() == "shared.l1" and cache.dtype == dtype
    assert tuple(int(value) for value in cache.shape) == (16, 96)
    assert tuple(int(value) for value in cache_origin) == (0, 0)
    assert tuple(int(value) for value in cache_extent) == (16, 96)
    # The GM tail is explicit ordinary copy intent: source logical K=70,
    # copy/cache capacity=96. OOB lowering owns zero padding before consumers.
    assert int(a.shape[1]) == 70
    assert tuple(int(value) for value in _region(a_load.args[0])[2]) == (16, 96)
    (gemm,) = calls(loop.body, "tl.tileop.gemm")
    operand, origin, extents = _region(gemm.args[0])
    assert operand.data.same_as(cache.data)
    assert tuple(int(value) for value in extents) == (16, 32)
    assert tuple(int(value) for value in gemm.args[5:8]) == (16, 16, 32)
    assert bool(gemm.args[4]) == transpose_b
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(origin[0], 0)
    for iteration, offset in ((0, 0), (1, 32), (2, 64)):
        substitution = {loop.loop_var: tirx.IntImm(loop.loop_var.dtype, iteration)}
        actual_offset = tirx.stmt_functor.substitute(origin[1], substitution)
        actual_clear = tirx.stmt_functor.substitute(gemm.args[9], substitution)
        assert analyzer.can_prove_equal(actual_offset, offset)
        assert int(analyzer.simplify(actual_clear)) == int(iteration == 0)


def test_cache_capacity_is_checked_by_existing_target_planner(example):
    # A alone needs 1 MiB of L1. No example-specific resource table or fallback
    # may hide this; the native target planner must reject it before codegen.
    program = example.gemm(16, 16, 16384, dtype="float16", tile_m=32, tile_n=16, tile_k=256, cache_a=True)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    with pytest.raises(tvm.error.TVMError, match='scope "shared.l1" capacity'):
        tilelang.lower(program, target=target)
