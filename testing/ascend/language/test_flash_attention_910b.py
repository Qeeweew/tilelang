"""CPU mathematical and frontend contracts for the ordinary dav-2201 FA."""

import importlib.util
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


@pytest.mark.parametrize("changes,message", [({"query_heads": 3}, "divisible"), ({"head_dim": 17}, "multiple of 16"), ({"dtype": "float32"}, "float16 or bfloat16"), ({"query_len": 0}, "positive integers"), ({"causal": True, "query_start": 17}, "within the KV sequence"), ({"softmax_scale": float("inf")}, "finite")])
def test_invalid_input_contract_is_rejected_before_frontend_construction(example, changes, message):
    arguments = dict(batch=1, query_heads=4, kv_heads=2, query_len=17, kv_len=33, head_dim=16)
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        example.flash_attention_fwd(**arguments)


def test_variants_share_typed_math_and_pipeline_exposes_independent_rescale(example):
    from tilelang import tvm
    import tilelang
    from tvm import tirx
    from testing.ascend._ir import calls, nodes

    arguments = dict(batch=1, query_heads=2, kv_heads=1, query_len=33, kv_len=65, head_dim=16, dtype="float16", causal=True, tiling=example.FwdTiling(32, 16, 3))
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
        transfers = calls(mod, "tl.tileop.ascend_copy")
        logical_bridges = [call for call in transfers if call.args[0].args[0].buffer.scope() == "shared.l0c"]
        assert len(logical_bridges) == 2
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
