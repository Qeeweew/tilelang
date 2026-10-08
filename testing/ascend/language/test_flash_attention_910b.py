"""CPU mathematical and frontend contracts for the ordinary dav-2201 FA."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    if not path.exists():
        pytest.skip("The focused example regression needs examples/ascend/flash_attention/910b.py beside testing/")
    spec = importlib.util.spec_from_file_location("_tilelang_flash_attention_910b", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _full_attention(example, q, k, v, *, dtype, causal, query_start, scale):
    q, k, v = (example._round_dtype(array, dtype) for array in (q, k, v))
    head_map = np.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    scores = q @ k[:, head_map].swapaxes(-1, -2) * np.float32(scale)
    if causal:
        visible = np.arange(k.shape[2])[None, :] <= query_start + np.arange(q.shape[2])[:, None]
        scores = np.where(visible, scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return example._round_dtype(probabilities @ v[:, head_map], dtype)


@pytest.mark.parametrize("dtype,tolerance", [("float16", 0.0015), ("bfloat16", 0.015)])
@pytest.mark.parametrize("causal,query_start", [(False, 0), (True, 0), (True, 16)])
def test_online_block_rounding_tracks_full_attention_for_mha_gqa_and_tails(example, dtype, tolerance, causal, query_start):
    random = np.random.default_rng(43)
    q = random.normal(0, 0.5, (2, 4, 17, 16)).astype(np.float32)
    k = random.normal(0, 0.5, (2, 2, 33, 16)).astype(np.float32)
    v = random.normal(0, 0.5, k.shape).astype(np.float32)
    scale = 0.25
    actual = example.attention_reference(q, k, v, dtype=dtype, block_kv=16, causal=causal, query_start=query_start, softmax_scale=scale)
    expected = _full_attention(example, q, k, v, dtype=dtype, causal=causal, query_start=query_start, scale=scale)
    assert actual.shape == q.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_causal_blocks_with_no_visible_keys_do_not_change_prior_rows(example, dtype):
    q = np.zeros((1, 1, 3, 16), dtype=np.float32)
    k = np.zeros((1, 1, 33, 16), dtype=np.float32)
    v = np.broadcast_to(np.arange(33, dtype=np.float32)[None, None, :, None], k.shape).copy()
    output = example.attention_reference(q, k, v, dtype=dtype, block_kv=16, causal=True)
    np.testing.assert_array_equal(output[0, 0, :, 0], [0, 0.5, 1])
    np.testing.assert_array_equal(output, np.broadcast_to(output[..., :1], output.shape))


def test_bfloat16_rounding_is_ties_to_even(example):
    bits = np.array([0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32)
    rounded = example._round_dtype(bits.view(np.float32), "bfloat16").view(np.uint32)
    np.testing.assert_array_equal(rounded, [0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gqa_maps_consecutive_heads_and_keeps_batches_independent(example, dtype):
    q = np.zeros((2, 4, 17, 16), dtype=np.float32)
    k = np.zeros((2, 2, 33, 16), dtype=np.float32)
    v = np.empty_like(k)
    for batch, values in enumerate(((3, 11), (23, 37))):
        for head, value in enumerate(values):
            v[batch, head] = value
    actual = example.attention_reference(q, k, v, dtype=dtype, block_kv=16)
    # Independent expected head labels avoid reusing the oracle's head map.
    labels = np.array([[3, 3, 11, 11], [23, 23, 37, 37]], dtype=np.float32)
    np.testing.assert_array_equal(actual, np.broadcast_to(labels[:, :, None, None], actual.shape))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("query_start", [0, 16])
def test_rectangular_causal_offset_matches_exact_visible_prefix(example, dtype, query_start):
    q = np.zeros((1, 1, 17, 16), dtype=np.float32)
    k = np.zeros((1, 1, 33, 16), dtype=np.float32)
    v = np.broadcast_to(np.arange(33, dtype=np.float32)[None, None, :, None], k.shape).copy()
    actual = example.attention_reference(q, k, v, dtype=dtype, block_kv=16, causal=True, query_start=query_start)
    # Uniform scores average keys0..(query_start+i), including diagonal keys.
    expected = (query_start + np.arange(17, dtype=np.float32)) / 2
    np.testing.assert_array_equal(actual[0, 0, :, 0], expected)
    np.testing.assert_array_equal(actual, np.broadcast_to(expected[None, None, :, None], actual.shape))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_new_block_maximum_rescales_prior_numerator_and_denominator(example, dtype):
    q = np.zeros((1, 1, 1, 16), dtype=np.float32)
    q[..., 0] = 1
    k = np.zeros((1, 1, 33, 16), dtype=np.float32)
    k[..., :16, 0] = -20
    k[..., 16:32, 0] = 20
    v = np.full_like(k, -7)
    v[..., :16, :] = 5
    actual = example.attention_reference(q, k, v, dtype=dtype, block_kv=16, softmax_scale=1)
    np.testing.assert_array_equal(actual, np.full_like(q, -7))


def test_zero_scale_has_uniform_finite_attention(example):
    q = np.full((1, 1, 17, 16), 0.25, dtype=np.float32)
    k = np.full((1, 1, 33, 16), 0.75, dtype=np.float32)
    v = np.broadcast_to(np.arange(33, dtype=np.float32)[None, None, :, None], k.shape).copy()
    actual = example.attention_reference(q, k, v, dtype="float16", block_kv=16, softmax_scale=0)
    np.testing.assert_array_equal(actual, np.full_like(q, 16))


def test_oracle_rejects_nonfinite_operands_and_score_overflow(example):
    q = np.zeros((1, 1, 1, 16), dtype=np.float32)
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    q[..., 0] = np.inf
    with pytest.raises(ValueError, match="finite after operand dtype rounding"):
        example.attention_reference(q, k, v, dtype="float16", block_kv=16)
    q.fill(1e20)
    k.fill(1e20)
    with pytest.raises(ValueError, match="finite FP32 scores"):
        example.attention_reference(q, k, v, dtype="bfloat16", block_kv=16)


@pytest.mark.parametrize("builder", ["flash_attention_fwd", "get_tuning_configs"])
@pytest.mark.parametrize("changes,message", [({"query_heads": 3}, "divisible"), ({"head_dim": 17}, "multiple of 16"), ({"dtype": "float32"}, "float16 or bfloat16"), ({"query_len": 0}, "positive integers"), ({"causal": True, "query_start": 17}, "within the KV sequence"), ({"softmax_scale": float("inf")}, "finite"), ({"variant": "unknown"}, "variant")])
def test_invalid_input_contract_is_rejected_before_frontend_construction(example, builder, changes, message):
    arguments = dict(batch=1, query_heads=4, kv_heads=2, query_len=17, kv_len=33, head_dim=16)
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        getattr(example, builder)(**arguments)


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("kv_len,width", [(35, 64), (64, 64), (65, 128), (128, 128), (129, 256), (256, 256), (257, 256), (1024, 256)])
def test_default_frontend_kv_regions_and_trips_cover_the_sequence(example, variant, kv_len, width):
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    program = example.flash_attention_fwd(1, 4, 2, 33, kv_len, 16, variant=variant)
    (loop,) = [node for node in nodes(program, tirx.For) if node.kind == tirx.ForKind.SERIAL]
    assert _constant(loop.extent, {}) == (kv_len + width - 1) // width
    gemms = calls(program, "tl.tileop.gemm")
    assert [tuple(map(int, call.args[5:8])) for call in gemms] == [(64, width, 16), (64, 16, width)]
    for parameter in program.params[1:3]:
        data = program.buffer_map[parameter].data
        (load,) = [call for call in calls(program, "tl.tileop.ascend_copy")
                   if call.args[0].args[0].buffer.data.same_as(data)]
        assert int(load.args[0].args[4]) == width
        assert tuple(map(int, load.args[1].args[0].buffer.shape)) == (width, 16)
    assert bool(loop.annotations.get("tl.ascend_auto_pipeline", False)) == (variant == "pipeline")
    if variant == "pipeline":
        assert int(loop.annotations["num_stages"]) == 4


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("query_len,heads,head_dim,rows", [
    (65, 4, 128, 64),
    (65, 4, 144, 48),
    (65, 4, 256, 32),
    (33, 4, 512, 16),
    (1, 130, 128, 64),
    (1, 130, 256, 32),
], ids=["prefill-d128", "prefill-aligned-nonpower", "prefill-d256",
        "prefill-minimum", "packed-d128", "packed-d256"])
def test_default_query_tiles_preserve_regions_and_launch_coverage(example, variant, query_len, heads, head_dim, rows):
    from collections import Counter
    from itertools import product
    from tvm import tirx
    from testing.ascend._ir import calls, nodes
    from testing.ascend.auto_schedule.test_owner_result_completion import (
        _PURE_DEFINITIONS, _SCALAR_VALUES, _region,
    )

    program = example.flash_attention_fwd(2, heads, 2, query_len, 1024, head_dim, variant=variant)
    buffers = [program.buffer_map[parameter] for parameter in program.params]
    launch, = [loop for loop in nodes(program, tirx.For)
               if loop.thread_binding is not None and loop.thread_binding.thread_tag == "blockIdx.x"]
    transfers = calls(program, "tl.tileop.ascend_copy")
    q_load, = [call for call in transfers if call.args[0].args[0].buffer.data.same_as(buffers[0].data)]
    export, = [call for call in transfers if call.args[1].args[0].buffer.data.same_as(buffers[3].data)]
    assert tuple(map(int, q_load.args[1].args[0].buffer.shape)) == (rows, head_dim)
    assert tuple(map(int, export.args[0].args[0].buffer.shape)) == (rows, head_dim)
    assert [tuple(map(int, call.args[5:8])) for call in calls(program, "tl.tileop.gemm")] == [
        (rows, 256, head_dim), (rows, head_dim, 256),
    ]
    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    packed = query_len == 1
    groups = heads // 2
    # Enumerate independent expected output ownership, including each tail.
    expected = [(b, h, start, min(rows, (groups if packed else query_len) - start))
                for b, h in product(range(2), range(2 if packed else heads))
                for start in range(0, groups if packed else query_len, rows)]
    assert int(launch.extent) == len(expected)
    written = Counter()
    for block, (batch, head, start, count) in enumerate(expected):
        env = {_PURE_DEFINITIONS: definitions, _SCALAR_VALUES: {},
               launch.loop_var: tirx.const(block, launch.loop_var.dtype)}
        q_buffer, q_origin, q_shape = _region(q_load.args[0], env)
        o_buffer, o_origin, o_shape = _region(export.args[1], env)
        assert q_buffer.data.same_as(buffers[0].data) and o_buffer.data.same_as(buffers[3].data)
        if packed:
            assert q_origin == o_origin == (batch, head * groups + start, 0)
            assert q_shape == o_shape == (1, count, head_dim)
            written.update((batch, head * groups + row, 0) for row in range(start, start + count))
        else:
            assert q_origin == o_origin == (batch, head, start, 0)
            assert q_shape == (1, 1, rows, head_dim)
            assert o_shape == (1, 1, count, head_dim)
            written.update((batch, head, row) for row in range(start, start + count))
    assert written == Counter({position: 1 for position in product(range(2), range(heads), range(query_len))})


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("head_dim,rows", [(16, 32), (256, 64)])
def test_explicit_tiling_dictionary_preserves_dataclass_frontend_and_overrides_defaults(example, variant, head_dim, rows):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    arguments = dict(batch=1, query_heads=4, kv_heads=2, query_len=33, kv_len=513, head_dim=head_dim, variant=variant)
    explicit = dict(block_q=rows, block_kv=16, max_stages=2)
    dictionary = example.flash_attention_fwd(**arguments, tiling=explicit)
    dataclass = example.flash_attention_fwd(**arguments, tiling=example.FwdTiling(**explicit))
    tvm.ir.assert_structural_equal(dictionary, dataclass)
    assert [tuple(map(int, call.args[5:8])) for call in calls(dictionary, "tl.tileop.gemm")] == [(rows, 16, head_dim), (rows, head_dim, 16)]
    (loop,) = [node for node in nodes(dictionary, tirx.For) if node.kind == tirx.ForKind.SERIAL]
    assert _constant(loop.extent, {}) == 33
    if variant == "pipeline":
        assert int(loop.annotations["num_stages"]) == 2


@pytest.mark.parametrize("tiling,message", [
    ({"block_q": 64, "block_kv": 64, "max_stages": 4, "extra": 1}, "unknown fields"),
    ({"block_q": 64, "block_kv": 64}, "missing fields"),
    ({"block_q": 64, "block_kv": 64.0, "max_stages": 4}, "block_kv"),
    ({"block_q": True, "block_kv": 64, "max_stages": 4}, "block_q"),
    ({"block_q": 64, "block_kv": 64, "max_stages": True}, "max_stages"),
    ({"block_q": 64, "block_kv": 64, "max_stages": 0}, "max_stages"),
    ([], "FwdTiling or a dictionary"),
])
def test_tiling_dictionary_is_strict_before_frontend_construction(example, tiling, message):
    with pytest.raises(ValueError, match=message):
        example.flash_attention_fwd(1, 4, 2, 33, 513, 16, tiling=tiling)


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("query_len,kv_len,heads,kv_heads,head_dim,causal,start,rows,widths", [
    (33, 35, 4, 2, 16, False, 0, 64, [64]),
    (33, 65, 4, 2, 16, True, 32, 64, [64, 128]),
    (65, 130, 4, 2, 16, False, 0, 64, [64, 128, 256]),
    (1, 4096, 17, 1, 16, True, 4095, 32, [64, 128, 256]),
    (65, 1024, 4, 2, 128, True, 0, 64, [64, 128, 256]),
    (65, 1024, 4, 2, 256, True, 0, 32, [64, 128, 256]),
    (1, 4096, 130, 2, 256, True, 4095, 32, [64, 128, 256]),
])
def test_tuning_configs_roundtrip_to_direct_factory_calls_with_complete_problem(
        example, variant, query_len, kv_len, heads, kv_heads, head_dim, causal, start, rows, widths):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    problem = dict(batch=2, query_heads=heads, kv_heads=kv_heads, query_len=query_len,
                   kv_len=kv_len, head_dim=head_dim, dtype="float16", variant=variant,
                   causal=causal, query_start=start, softmax_scale=0.125)
    configs = example.get_tuning_configs(**problem)
    assert json.loads(json.dumps(configs, allow_nan=False)) == configs
    assert len({json.dumps(config, sort_keys=True) for config in configs}) == len(configs)
    assert [config["tiling"]["block_kv"] for config in configs] == widths
    for config in configs:
        assert {key: value for key, value in config.items() if key != "tiling"} == problem
        assert config["tiling"] == dict(block_q=rows, block_kv=config["tiling"]["block_kv"], max_stages=4)
        program = example.flash_attention_fwd(**config)
        assert all(int(call.args[5]) == rows for call in calls(program, "tl.tileop.gemm"))
        assert int(calls(program, "tl.tileop.gemm")[0].args[6]) == config["tiling"]["block_kv"]
        (loop,) = [node for node in nodes(program, tirx.For) if node.kind == tirx.ForKind.SERIAL]
        assert bool(loop.annotations.get("tl.ascend_auto_pipeline", False)) == (variant == "pipeline")
    # The last candidate is the same complete algorithm expression as the
    # default, including packed head ownership, tail bounds and typed math.
    tvm.ir.assert_structural_equal(example.flash_attention_fwd(**problem), example.flash_attention_fwd(**configs[-1]))


@pytest.mark.parametrize("dtype,tolerance", [("float16", 0.0015), ("bfloat16", 0.015)])
@pytest.mark.parametrize("kv_len,width", [(65, 128), (130, 256), (513, 256)])
@pytest.mark.parametrize("causal", [False, True])
def test_default_oracle_uses_matching_block_rounding_and_tracks_attention_math(example, dtype, tolerance, kv_len, width, causal):
    random = np.random.default_rng(941)
    q = random.normal(0, 0.5, (2, 4, 5, 16)).astype(np.float32)
    k = random.normal(0, 0.5, (2, 2, kv_len, 16)).astype(np.float32)
    v = random.normal(0, 0.5, k.shape).astype(np.float32)
    kwargs = dict(dtype=dtype, causal=causal, query_start=kv_len - 5, softmax_scale=0.25)
    actual = example.attention_reference(q, k, v, **kwargs)
    explicit = example.attention_reference(q, k, v, **kwargs, block_kv=width)
    np.testing.assert_array_equal(actual, explicit)
    expected = _full_attention(example, q, k, v, dtype=dtype, causal=causal, query_start=kv_len - 5, scale=0.25)
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("kv_len,causal", [(65, True), (64, False), (65, False), (64, True)])
def test_variants_share_typed_math_and_pipeline_exposes_independent_rescale(example, kv_len, causal):
    from tilelang import tvm
    import tilelang
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    arguments = dict(batch=1, query_heads=2, kv_heads=1, query_len=33, kv_len=kv_len, head_dim=16, dtype="float16", causal=causal, tiling=example.FwdTiling(32, 16, 3))
    programs = {}
    for variant in ("generic", "pipeline"):
        program = example.flash_attention_fwd(**arguments, variant=variant)
        mod = tvm.IRModule({"main": program})
        mod = tilelang.transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False, default_threads=None, unsupported_annotations=["cluster_dims"], launch_dim_tags=["cthread"])(mod)
        programs[variant] = mod
        assert len(calls(mod, "tl.tileop.gemm")) == 2
        assert not [block for block in nodes(mod, tirx.SBlock) if block.name_hint in ("SIMD_VF", "SIMT_VF")]
        assert not [attr for attr in nodes(mod, tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == "cthread"]
        assert not [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and "softmax" in call.op.name.lower()]
        assert bool(calls(mod, "tirx.if_then_else")) == (causal or kv_len % 16 != 0)
        transfers = calls(mod, "tl.tileop.ascend_copy")
        logical_bridges = [call for call in transfers if call.args[0].args[0].buffer.scope() == "shared.l0c"]
        assert len(logical_bridges) == 2
        (score_bridge,) = [call for call in logical_bridges
                           if call.args[1].args[0].buffer.name == "score_input"]
        score_input = score_bridge.args[1].args[0].buffer
        # The DMA destination is consumed only by scale/mask, not by the
        # destructive softmax recurrence. There is no intervening UB copy.
        assert not [store for store in nodes(mod, tirx.BufferStore)
                    if store.buffer.data.same_as(score_input.data)]
        input_consumers = [store for store in nodes(mod, tirx.BufferStore)
                           if any(load.buffer.data.same_as(score_input.data)
                                  for load in nodes(store.value, tirx.BufferLoad))]
        assert input_consumers and all(store.buffer.name == "scores" for store in input_consumers)
        assert not [call for call in transfers
                    if call.args[0].args[0].buffer.data.same_as(score_input.data)]
        group = next(attr.node.var for attr in nodes(mod, tirx.AttrStmt) if attr.attr_key == "thread_extent" and attr.node.thread_tag == "blockIdx.x")
        valid_rows = logical_bridges[0].args[0].args[2]
        analyzer = tvm.arith.Analyzer()
        definitions = {binding.var: binding.value for binding in nodes(mod, tirx.Bind)}
        expanded_rows = valid_rows
        for _ in definitions:
            expanded_rows = tirx.stmt_functor.substitute(expanded_rows, definitions)
        counts = [int(analyzer.simplify(tirx.stmt_functor.substitute(expanded_rows, {group: tirx.IntImm("int32", index)}))) for index in range(4)]
        assert counts == [32, 1, 32, 1]
        for call in logical_bridges:
            src_rows, dst_rows = call.args[0].args[2], call.args[1].args[2]
            assert int(dst_rows) == 32
            assert not isinstance(src_rows, tirx.IntImm)
            tvm.ir.assert_structural_equal(src_rows, valid_rows)
        p_to_l1 = [call for call in transfers if call.args[0].args[0].buffer.name == "p" and call.args[1].args[0].buffer.scope() == "shared.l1"]
        assert len(p_to_l1) == 1
        assert not isinstance(p_to_l1[0].args[0].args[2], tirx.IntImm)
        assert int(p_to_l1[0].args[1].args[2]) == 32
        tvm.ir.assert_structural_equal(p_to_l1[0].args[0].args[2], valid_rows)
        (export,) = [call for call in transfers if call.args[1].args[0].buffer.name == "O"]
        tvm.ir.assert_structural_equal(export.args[0].args[2], valid_rows)
        tvm.ir.assert_structural_equal(export.args[1].args[4], valid_rows)
        (state_copy,) = [call for call in transfers if call.args[0].args[0].buffer.name == "next_maximum" and call.args[1].args[0].buffer.name == "maximum"]
        assert state_copy.args[0].args[0].buffer.dtype == "float32"
        assert state_copy.args[1].args[0].buffer.dtype == "float32"
        tvm.ir.assert_structural_equal(state_copy.args[0].args[2], valid_rows)
        tvm.ir.assert_structural_equal(state_copy.args[1].args[2], valid_rows)
        assert not [store for store in nodes(mod, tirx.BufferStore) if store.buffer.name == "maximum" and isinstance(store.value, tirx.BufferLoad)]

    generic_loops = [loop for loop in nodes(programs["generic"], tirx.For) if loop.kind == tirx.ForKind.SERIAL]
    pipeline_loops = [loop for loop in nodes(programs["pipeline"], tirx.For) if loop.kind == tirx.ForKind.SERIAL]
    assert len(generic_loops) == len(pipeline_loops) == 1
    assert "tl.ascend_auto_pipeline" not in generic_loops[0].annotations
    assert bool(pipeline_loops[0].annotations["tl.ascend_auto_pipeline"])
    assert int(pipeline_loops[0].annotations["num_stages"]) == 3
    def output_stores(stmt):
        return [store for store in nodes(stmt, tirx.BufferStore) if store.buffer.name == "output"]

    generic_body = list(generic_loops[0].body.seq)
    pipeline_body = list(pipeline_loops[0].body.seq)
    # The recurrence retains every shared operation and dtype. Only the
    # output's FP32 multiply/add acquire separate dependency-bearing tasks.
    shared = lambda body: tirx.SeqStmt([stmt for stmt in body if not output_stores(stmt)])
    tvm.ir.assert_structural_equal(shared(generic_body), shared(pipeline_body), map_free_vars=True)
    (combined,) = output_stores(generic_loops[0].body)
    rescale, update = output_stores(pipeline_loops[0].body)
    assert isinstance(combined.value, tirx.Add)
    assert isinstance(combined.value.a, tirx.Mul)
    assert isinstance(rescale.value, tirx.Mul)
    assert isinstance(update.value, tirx.Add)
    assert all(store.value.dtype == "float32" for store in (combined, rescale, update))
    assert [load.buffer.name for load in nodes(rescale.value, tirx.BufferLoad)] == ["alpha", "output"]
    assert [load.buffer.name for load in nodes(update.value, tirx.BufferLoad)] == ["output", "pv"]
    rescale_index, update_index = [i for i, stmt in enumerate(pipeline_body) if output_stores(stmt)]
    (p_export_index,) = [i for i, stmt in enumerate(pipeline_body)
                        if any(call.args[0].args[0].buffer.name == "p" for call in calls(stmt, "tl.tileop.ascend_copy"))]
    (pv_import_index,) = [i for i, stmt in enumerate(pipeline_body)
                         if any(call.args[1].args[0].buffer.name == "pv" for call in calls(stmt, "tl.tileop.ascend_copy"))]
    assert rescale_index < p_export_index < pv_import_index < update_index


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("causal,query_len,kv_len,block_kv", [
    (False, 32, 128, 64),
    (True, 33, 65, 16),
], ids=["dense", "causal-gqa-tail-empty-owner"])
def test_lowered_score_dma_input_has_only_scale_read_effects(example, variant, causal, query_len, kv_len, block_kv):
    import tilelang
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import allocated_buffer, calls, nodes

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLowerVectorTiles":
                snapshots[info.name] = mod

    program = example.flash_attention_fwd(
        1, 2, 1, query_len, kv_len, 16, variant=variant, causal=causal,
        dtype="float16", tiling=example.FwdTiling(32, block_kv, 3),
    )
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(
            program, target={"kind": "ascend", "arch": "dav-2201"},
            enable_host_codegen=False, enable_device_compile=False,
        )
    assert artifact.kernel_source and "asc_mmad(" in artifact.kernel_source
    native = snapshots["tl.AscendLowerVectorTiles"]
    score_input = allocated_buffer(native, "score_input")
    scores = allocated_buffer(native, "scores")
    assert not score_input.data.same_as(scores.data)
    # This is the actual native-effect IR after ownership and vector lowering,
    # before allocation/scheduling. A later reduction cannot keep the DMA
    # input live or silently turn it back into a destructive work buffer.
    consumers = []
    for call in nodes(native, tirx.Call):
        if not isinstance(call.op, tvm.ir.Op) or not call.op.name.startswith("tl.ascend_membase."):
            continue
        pointers = [pointer for pointer in nodes(call, tirx.Call)
                    if isinstance(pointer.op, tvm.ir.Op)
                    and pointer.op.name in ("tl.access_ptr", "tl.region")
                    and pointer.args[0].buffer.data.same_as(score_input.data)]
        if pointers:
            consumers.append(call)
            assert call.op.name == "tl.ascend_membase.mul_scalar"
            assert all(int(pointer.args[-1] if pointer.op.name == "tl.access_ptr" else pointer.args[1]) == 1
                       for pointer in pointers)
    assert consumers
    (dma,) = [call for call in calls(native, "tl.tileop.ascend_copy")
              if call.args[1].args[0].buffer.data.same_as(score_input.data)]
    assert dma.args[0].args[0].buffer.scope() == "global"
    assert not [call for call in calls(native, "tl.tileop.ascend_copy")
                if call.args[0].args[0].buffer.data.same_as(score_input.data)]


def _causal_frontend(example, *, variant, query_start=0, kv_len=65):
    import tilelang
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import nodes

    program = example.flash_attention_fwd(
        2, 2, 1, 33, kv_len, 16, dtype="float16", variant=variant,
        causal=True, query_start=query_start, tiling=example.FwdTiling(32, 16, 3),
    )
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True, lower_thread_binding=False, default_threads=None,
        unsupported_annotations=["cluster_dims"], launch_dim_tags=["cthread"],
    )(tvm.IRModule({"main": program}))
    (loop,) = [node for node in nodes(mod, tirx.For) if node.kind == tirx.ForKind.SERIAL]
    group = next(attr.node.var for attr in nodes(mod, tirx.AttrStmt)
                 if attr.attr_key == "thread_extent" and attr.node.thread_tag == "blockIdx.x")
    definitions = {binding.var: binding.value for binding in nodes(mod, tirx.Bind)}
    extent = loop.extent
    for _ in definitions:
        extent = tirx.stmt_functor.substitute(extent, definitions)
    return mod, loop, group, extent


def _constant(expr, bindings):
    from tilelang import tvm
    from tvm import tirx
    values = {var: tirx.IntImm(var.dtype, value) for var, value in bindings.items()}
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, values)))


@pytest.mark.parametrize("query_start,expected", [(0, [2, 3]), (16, [3, 4]), (32, [4, 5])])
def test_causal_variants_share_group_uniform_tail_and_offset_trip_bounds(example, query_start, expected):
    from tvm import tirx
    from testing.ascend._ir import nodes

    extents = []
    for variant in ("generic", "pipeline"):
        mod, loop, group, extent = _causal_frontend(example, variant=variant, query_start=query_start)
        assert str(loop.loop_var.dtype) == str(extent.dtype) == "int64"
        assert not any("sid" in var.name for var in nodes(extent, tirx.Var))
        counts = [_constant(extent, {group: index}) for index in range(8)]
        assert counts == expected * 4  # Both batches and both query heads.
        # Capacity32 partitions into two private capacities16. The final group
        # has one valid row and an empty SID1, but both SIDs keep the group's
        # same KV loop; private ownership must never determine trip count.
        for index, count in enumerate(counts):
            base = (index % 2) * 32
            valid_rows = min(32, 33 - base)
            owner_rows = [min(16, max(valid_rows - sid * 16, 0)) for sid in (0, 1)]
            assert owner_rows == ([16, 16] if index % 2 == 0 else [1, 0])
            end = query_start + base + valid_rows
            assert count >= 1 and (count - 1) * 16 < end <= count * 16
            assert count <= (65 + 15) // 16
        extents.append(counts)
    assert extents[0] == extents[1]


def test_causal_trip_ceil_keeps_large_key_positions_in_int64(example):
    # An actual frontend tensor must have representable strides. This length
    # crosses Int32 when rounded to a KV tile, while all tensor strides fit
    # Int64; an INT64_MAX tensor would overflow before kernel construction.
    largest = (1 << 31) - 1
    offset = largest - 33
    expected = [(largest - 1) // 16 + ((largest - 1) % 16 != 0),
                largest // 16 + (largest % 16 != 0)]
    for variant in ("generic", "pipeline"):
        _, loop, group, extent = _causal_frontend(example, variant=variant, query_start=offset, kv_len=largest)
        assert str(loop.loop_var.dtype) == "int64"
        assert [_constant(extent, {group: index}) for index in range(2)] == expected


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("query_start", [0, 16])
def test_causal_group_bound_skips_only_identity_online_softmax_updates(example, dtype, query_start):
    random = np.random.default_rng(79)
    q = random.normal(0, 0.5, (1, 2, 33, 16)).astype(np.float32)
    k = random.normal(0, 0.5, (1, 1, 65, 16)).astype(np.float32)
    v = random.normal(0, 0.5, k.shape).astype(np.float32)
    for base in (0, 32):
        rows = min(32, 33 - base)
        end = query_start + base + rows
        limit = min(5, end // 16 + (end % 16 != 0))
        stop = min(65, limit * 16)
        query = q[:, :, base:base + rows]
        kwargs = dict(dtype=dtype, block_kv=16, causal=True, query_start=query_start + base)
        full = example.attention_reference(query, k, v, **kwargs)
        bounded = example.attention_reference(query, k[:, :, :stop], v[:, :, :stop], **kwargs)
        np.testing.assert_array_equal(bounded, full)


def test_dynamic_causal_extent_keeps_both_core_stage_guards_and_empty_sid_tasks(example):
    # Exercise the existing stage rewriter with the actual frontend expression.
    # Stages0/2 are a diagnostic pass input, not an automatic optimizer claim.
    from tilelang import tvm
    from tilelang.ascend import transform
    from tvm import tirx
    from testing.ascend._ir import copy, kernel, nodes, seq, statements
    from testing.ascend.auto_schedule._scheduled_ir import unit

    _, _, group, extent = _causal_frontend(example, variant="pipeline", query_start=16)
    epoch = tirx.Var("kv_epoch", "int64")
    sid = tirx.Var("sid", "int32")
    qrows = tirx.min(32, 33 - (group % 2) * 32)
    owned = tirx.min(16, tirx.max(qrows - sid * 16, 0))
    source = tirx.decl_buffer((80, 16), "float16", name="cube_source")
    l1 = tirx.decl_buffer((16, 16), "float16", name="cube_operand", scope="shared.l1")
    l1_late = tirx.decl_buffer((16, 16), "float16", name="cube_late", scope="shared.l1")
    workspace = tirx.decl_buffer((5, 32, 16), "float32", name="vector_source")
    ub = tirx.decl_buffer((16, 16), "float32", name="vector_private", scope="shared.dyn")
    ub_late = tirx.decl_buffer((16, 16), "float32", name="vector_late", scope="shared.dyn")
    cube = copy(source, l1, src_indices=[epoch * 16, 0], src_shape=[16, 16])
    vector = copy(workspace, ub, src_indices=[epoch, sid * 16, 0], src_shape=[1, owned, 16], dst_shape=[owned, 16])
    cube_late = copy(source, l1_late, src_indices=[epoch * 16, 0], src_shape=[16, 16])
    vector_late = copy(workspace, ub_late, src_indices=[epoch, sid * 16, 0], src_shape=[1, owned, 16], dst_shape=[owned, 16])
    loop = tirx.For(epoch, tirx.IntImm("int64", 0), extent, tirx.ForKind.SERIAL,
        seq(unit(cube, core=2, stage=0), unit(vector, core=1, stage=0),
            unit(cube_late, core=2, stage=2), unit(vector_late, core=1, stage=2)))
    before = kernel(unit(loop, core=None), buffers=[l1, l1_late, ub, ub_late], params=[group, source, workspace],
                    annotations={"vector_count": 2})
    lane = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    before["main"] = before["main"].with_body(tirx.AttrStmt(lane, "thread_extent", 2, before["main"].body))
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    before["main"] = before["main"].with_attr("target", target)
    with target:
        after = transform.LowerScheduledTIR()(before)
    vectors = [block for block in nodes(after, tirx.SBlock) if block.name_hint == "VECTOR"]
    assert vectors and all(int(block.annotations["vector_count"]) == 2 for block in vectors)
    sites = [(stmt.value, parents) for stmt, parents in statements(after)
             if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)
             and isinstance(stmt.value.op, tvm.ir.Op) and stmt.value.op.name == "tl.tileop.ascend_copy"]
    assert len(sites) == 4
    for group_index in (0, 1):
        trip_count = _constant(extent, {group: group_index})
        active = {"cube_source": [], "vector_source": []}
        for call, parents in sites:
            control = next(node for node in reversed(parents) if isinstance(node, tirx.For) and node.loop_var.name == "kv_epoch")
            guards = [node.condition for node in parents if isinstance(node, tirx.IfThenElse)]
            assert guards
            assert _constant(control.extent, {group: group_index}) == trip_count + 2
            source_name = call.args[0].args[0].buffer.name
            for owner in (0, 1):
                iterations = []
                for physical_epoch in range(_constant(control.extent, {group: group_index})):
                    bindings = {group: group_index, sid: owner, control.loop_var: physical_epoch}
                    if all(_constant(guard, bindings) for guard in guards):
                        logical = _constant(call.args[0].args[0].indices[0], bindings)
                        iterations.append(logical // 16 if source_name == "cube_source" else logical)
                assert iterations == list(range(trip_count))
                active[source_name].append(iterations)
            if source_name == "vector_source" and group_index == 1:
                assert _constant(call.args[0].args[3], {group: 1, sid: 1}) == 0
        assert active["cube_source"] == active["vector_source"]


def _whole_valid_score_branch(mod, loop):
    from tvm import tirx
    from testing.ascend._ir import nodes

    # The shared recurrence stays outside this control node. It branches only
    # between score scaling and the original per-lane prefix mask.
    candidates = []
    for branch in nodes(loop.body, tirx.IfThenElse):
        if branch.else_case is None:
            continue
        then_stores = nodes(branch.then_case, tirx.BufferStore)
        else_stores = nodes(branch.else_case, tirx.BufferStore)
        if (len(then_stores) == len(else_stores) == 1
                and then_stores[0].buffer.name == else_stores[0].buffer.name == "scores"
                and isinstance(then_stores[0].value, tirx.Mul)):
            candidates.append(branch)
    (branch,) = candidates
    definitions = {binding.var: binding.value for binding in nodes(mod, tirx.Bind)}
    condition = branch.condition
    for _ in definitions:
        condition = tirx.stmt_functor.substitute(condition, definitions)
    return branch, condition


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("query_start", [14, 15, 16])
def test_causal_whole_valid_scale_uses_first_query_and_keeps_original_mask(example, variant, query_start):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    mod, loop, group, extent = _causal_frontend(example, variant=variant, query_start=query_start)
    branch, condition = _whole_valid_score_branch(mod, loop)
    assert str(condition.dtype) == "bool"
    assert all(var.same_as(group) or var.same_as(loop.loop_var)
               for var in nodes(condition, tirx.Var))
    assert not calls(branch, "tl.tileop.ascend_copy")
    assert not calls(branch, "tl.tileop.gemm")
    assert not calls(branch, "tl.tileop.reduce")
    (dense,) = nodes(branch.then_case, tirx.BufferStore)
    (masked,) = nodes(branch.else_case, tirx.BufferStore)
    assert dense.value.dtype == masked.value.dtype == "float32"
    assert isinstance(dense.value, tirx.Mul)
    assert not calls(branch.then_case, "tirx.if_then_else")
    (select,) = calls(branch.else_case, "tirx.if_then_else")
    tvm.ir.assert_structural_equal(select.args[1], dense.value, map_free_vars=True)
    # Both original inequalities remain in the fallback: KV-tail visibility
    # and the inclusive causal diagonal. The score arithmetic stays identical.
    (tail_check,) = nodes(select.args[0], tirx.LT)
    (causal_check,) = nodes(select.args[0], tirx.LE)
    definitions = {binding.var: binding.value for binding in nodes(mod, tirx.Bind)}
    predicates = [tail_check, causal_check]
    for _ in definitions:
        predicates = [tirx.stmt_functor.substitute(expr, definitions) for expr in predicates]
    (row_loop, col_loop) = nodes(branch.else_case, tirx.For)[::-1]
    assert int(row_loop.extent) == 32 and int(col_loop.extent) == 16
    scale = np.float32(0.25)
    scores = np.arange(32 * 16, dtype=np.float32).reshape(32, 16) / 8 - 30
    seen = []
    for group_index in range(8):
        base = (group_index % 2) * 32
        rows = min(32, 33 - base)
        active = _constant(extent, {group: group_index})
        selected = []
        for epoch in range(active):
            key_base = epoch * 16
            bindings = {group: group_index, loop.loop_var: epoch}
            actual = bool(_constant(condition, bindings))
            expected = key_base + 16 <= query_start + base + 1
            assert actual == expected
            selected.append(actual)
            visible = ((key_base + np.arange(16)[None, :] < 65)
                       & (key_base + np.arange(16)[None, :]
                          <= query_start + base + np.arange(rows)[:, None]))
            if actual:
                assert visible.all()
                np.testing.assert_array_equal(
                    scores[:rows] * scale,
                    np.where(visible, scores[:rows] * scale, -np.inf),
                )
            # Check an actual first-row and last-valid-row lane against both
            # unchanged fallback inequalities, not a second mask algorithm.
            for row in (0, rows - 1):
                for col in (0, 15):
                    lane = {**bindings, row_loop.loop_var: row, col_loop.loop_var: col}
                    assert bool(_constant(predicates[0], lane)) == (key_base + col < 65)
                    assert bool(_constant(predicates[1], lane)) == (key_base + col <= query_start + base + row)
        seen.append(selected)
    assert seen[0] == seen[2] == seen[4] == seen[6]
    assert seen[1] == seen[3] == seen[5] == seen[7]
    assert seen[0][0] == (query_start >= 15)  # Exact whole-tile boundary.
    # A one-row qtail's empty SID1 uses this same group/epoch predicate:
    # the allowed-variable assertion excludes every private row/SID value.


def test_causal_whole_valid_tail_avoids_int32_endpoint_overflow(example):
    largest = (1 << 31) - 1
    for variant in ("generic", "pipeline"):
        mod, loop, group, extent = _causal_frontend(
            example, variant=variant, query_start=largest - 33, kv_len=largest,
        )
        _, condition = _whole_valid_score_branch(mod, loop)
        for group_index in (0, 1):
            active = _constant(extent, {group: group_index})
            bindings = {group: group_index, loop.loop_var: active - 1}
            assert (active - 1) * 16 + 16 > largest
            assert not _constant(condition, bindings)
            assert _constant(condition, {group: group_index, loop.loop_var: 0})
