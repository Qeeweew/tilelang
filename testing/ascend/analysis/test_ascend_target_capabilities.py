"""The selected architecture rejects unsupported routes before scheduling."""

import pytest

from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, region, seq


def _target(mod, arch="dav-2201"):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    return tvm.IRModule({name: func.with_attr("target", target) for name, func in mod.functions.items()})


def _validate(mod, arch="dav-2201"):
    return transform.AscendValidateTarget()(_target(mod, arch))


def _gemm(*, dtype="float16", b_dtype=None, accum="float32", l1=False, transpose_b=True):
    a = tirx.decl_buffer((16, 32), dtype, name="a", scope="shared.l1" if l1 else "shared.l0a")
    b = tirx.decl_buffer((16, 32) if transpose_b else (32, 16), b_dtype or dtype, name="b", scope="shared.l1" if l1 else "shared.l0b")
    c = tirx.decl_buffer((16, 16), accum, name="c", scope="shared.l0c")
    return kernel(tirx.Evaluate(T.gemm(a, b, c, transpose_B=transpose_b, clear_accum=True)), buffers=[a, b, c])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("l1", [False, True], ids=["explicit-l0", "l1-expansion-candidate"])
def test_native_half_gemm_preserves_ir(dtype, l1):
    before = _target(_gemm(dtype=dtype, l1=l1))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_l1_nn_gemm_is_a_supported_native_expansion_candidate():
    before = _target(_gemm(l1=True, transpose_b=False))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_native_cube_copy_chain_preserves_ir_and_logical_tails():
    gm = tirx.decl_buffer((17, 29), "bfloat16", name="gm")
    l1 = tirx.decl_buffer((32, 32), "bfloat16", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((32, 32), "bfloat16", name="l0", scope="shared.l0a")
    c = tirx.decl_buffer((32, 32), "float32", name="c", scope="shared.l0c")
    out = tirx.decl_buffer((17, 29), "bfloat16", name="out")
    before = _target(
        kernel(
            seq(
                copy(gm, l1, src_shape=(17, 29), dst_shape=(17, 29)),
                copy(l1, l0),
                copy(c, out, src_shape=(17, 29), dst_shape=(17, 29)),
            ),
            buffers=[l1, l0, c],
            params=[gm, out],
        )
    )
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


@pytest.mark.parametrize(
    "kwargs",
    [{"dtype": "float32"}, {"accum": "float16"}, {"b_dtype": "bfloat16"}],
    ids=["unsupported-fp32-mode", "fp16-accumulation", "mixed-operands"],
)
def test_unsupported_gemm_precision_fails_at_input_boundary(kwargs):
    with pytest.raises(ValueError, match="dav-2201: GEMM precision.*FP32 accumulation"):
        _validate(_gemm(**kwargs))


@pytest.mark.parametrize(
    "src_scope,dst_scope,diagnostic",
    [
        ("shared.l1", "shared", "direct L1 -> UB"),
        ("shared", "shared.l1", "logical GM handoff requires"),
    ],
)
def test_unsupported_direct_cube_vector_routes(src_scope, dst_scope, diagnostic):
    src = tirx.decl_buffer((16, 16), "float32", name="src", scope=src_scope)
    dst = tirx.decl_buffer((16, 16), "float32", name="dst", scope=dst_scope)
    with pytest.raises(ValueError, match=diagnostic):
        _validate(kernel(copy(src, dst), buffers=[src, dst]))


@pytest.mark.parametrize("src_scope,dst_scope,dtype", [("shared.l0c", "shared.dyn", "float32"), ("shared.dyn", "shared.l1", "float16")])
def test_logical_mixed_copy_is_a_candidate_for_explicit_gm_handoff(src_scope, dst_scope, dtype):
    src = tirx.decl_buffer((16, 32), dtype, scope=src_scope)
    dst = tirx.decl_buffer((16, 32), dtype, scope=dst_scope)
    before = _target(kernel(copy(src, dst), buffers=[src, dst]))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_gm_to_l1_transpose_is_not_silently_emitted_as_3510_dn2nz():
    gm = tirx.decl_buffer((16, 16), "float16", name="gm")
    l1 = tirx.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    with pytest.raises(ValueError, match="DN-to-NZ transpose is unsupported"):
        _validate(kernel(copy(gm, l1, annotations={"transpose": tirx.IntImm("int32", 1)}), buffers=[l1], params=[gm]))


@pytest.mark.parametrize("region", ["SIMT_VF", "SIMD_VF"])
def test_vf_regions_require_3510(region):
    body = tirx.SBlockRealize([], True, tirx.SBlock([], [], [], region, tirx.Evaluate(0)))
    before = kernel(body)
    with pytest.raises(ValueError, match="VF/RegBase execution is unsupported"):
        _validate(before)
    expected = _target(before, "dav-3510")
    tvm.ir.assert_structural_equal(_validate(before, "dav-3510"), expected)


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float4_e2m1fn"])
def test_low_precision_storage_rejected_even_without_gemm(dtype):
    buf = tirx.decl_buffer((16, 32), dtype, name="low_precision", scope="shared.l1")
    with pytest.raises(ValueError, match="FP8/FP4 storage and arithmetic are unsupported"):
        _validate(kernel(tirx.Evaluate(0), buffers=[buf]))


def test_blockscaled_operation_rejected_without_parsing_3510_operands():
    call = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.gemm_blockscaled"), [])
    with pytest.raises(ValueError, match="block-scaled MX GEMM is unsupported"):
        _validate(kernel(tirx.Evaluate(call)))


def test_regbase_storage_is_rejected_outside_a_vf_region():
    buf = tirx.decl_buffer((1,), "float32x64", name="reg", scope="local.var")
    with pytest.raises(ValueError, match="VF/RegBase values are unsupported"):
        _validate(kernel(tirx.Evaluate(0), buffers=[buf]))


def test_pure_vector_parallel_is_a_candidate_for_membase_lowering():
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 16, tirx.ForKind.PARALLEL, tirx.Evaluate(i))
    before = _target(kernel(body))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_mixed_parallel_reaches_the_private_aiv_partition_boundary():
    mod = _gemm()
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 16, tirx.ForKind.PARALLEL, tirx.Evaluate(i))
    root = mod["main"].body.block
    mixed = kernel(seq(root.body, body), buffers=list(root.alloc_buffers))
    before = _target(mixed)
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_private_ub_copy_and_store_are_supported_membase_candidates():
    gm = tirx.decl_buffer((16,), "float32", name="gm")
    ub = tirx.decl_buffer((16,), "float32", name="ub", scope="shared")
    i = tirx.Var("i", "int32")
    body = seq(copy(gm, ub), tirx.For(i, 0, 16, tirx.ForKind.PARALLEL, tirx.BufferStore(ub, ub[i] + 1, [i])), copy(ub, gm))
    before = _target(kernel(body, buffers=[ub], params=[gm]))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_private_ub_bits_copy_reaches_the_physical_layout_boundary(dtype):
    src = tirx.decl_buffer((17,), dtype, scope="shared")
    dst = tirx.decl_buffer((17,), dtype, scope="shared.dyn")
    before = _target(kernel(copy(src, dst), buffers=[src, dst]))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


@pytest.mark.parametrize("annotation", ["transpose", "dual_dst_ctl"])
def test_private_ub_copy_rejects_nonordinary_routes(annotation):
    src = tirx.decl_buffer((32,), "float32", scope="shared")
    dst = tirx.decl_buffer((32,), "float32", scope="shared")
    with pytest.raises(ValueError, match="private UB copy.*ordinary axes"):
        _validate(kernel(copy(src, dst, annotations={annotation: tirx.IntImm("int32", 1)}), buffers=[src, dst]))


def test_native_l1_load_reports_unsupported_precision_at_input_boundary():
    src = tirx.decl_buffer((16, 16), "float32", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((16, 16), "float32", name="dst", scope="shared.l0a")
    with pytest.raises(ValueError, match="native LoadData2D path requires FP16 or BF16"):
        _validate(kernel(copy(src, dst), buffers=[src, dst]))


def test_ordinary_row_reduce_is_checked_at_native_layout_boundary():
    src = tirx.decl_buffer((3, 32), "float32", scope="shared.dyn")
    dst = tirx.decl_buffer((3,), "float32", scope="shared.dyn")
    call = tirx.Call(
        "handle", tvm.ir.Op.get("tl.tileop.reduce"),
        [region(src, 1), region(dst, 2), tirx.StringImm("sum"), 1, tirx.const(True, "bool")],
    )
    before = _target(kernel(tirx.Evaluate(call), buffers=[src, dst]))
    tvm.ir.assert_structural_equal(transform.AscendValidateTarget()(before), before)


def test_missing_target_fails_at_contract_boundary():
    mod = _gemm()
    mod = tvm.IRModule({"main": mod["main"].without_attr("target")})
    with pytest.raises(ValueError, match="requires a bound target"):
        transform.AscendValidateTarget()(mod)
