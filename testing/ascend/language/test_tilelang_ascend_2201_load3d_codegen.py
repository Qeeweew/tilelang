"""Selected A loads emit actual 2201 3Dv2 setup and the 16-bit intrinsic."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T

from testing.ascend.language.test_tilelang_ascend_2201_cube_codegen import _call_arguments, _kernel


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m", [127, 128])
def test_native_full_m_subk_codegen_fully_defines_shared_configuration(dtype, m):
    @T.prim_func
    def main(a: T.Tensor((m, 256), dtype),
             b: T.Tensor((32, 256), dtype),
             c: T.Tensor((m, 32), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, 256), dtype)
            b_l1 = T.alloc_l1((32, 256), dtype)
            a_l0 = T.alloc_l0a((m, 128), dtype)
            b_l0 = T.alloc_l0b((32, 128), dtype)
            c_l0 = T.alloc_l0c((m, 32), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.copy(a_l1[:, 128:256], a_l0)
            T.copy(b_l1[:, 128:256], b_l0)
            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
            T.copy(c_l0, c)

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    source = tilelang.lower(main, target=target).kernel_source
    args = _call_arguments(source, "asc_copy_l12l0a")
    assert len(args) == 17, source
    assert "(__ca__ half*)" in args[0] and "(__cbuf__ half*)" in args[1], source
    assert args[2:] == ["128", "128", "128", "0", "1", "1", "1", "1", "1", "1", "0", "0", "0", "0", "256"], source
    assert "uint64_t(128) | (uint64_t(1) << 16)" in source, source
    assert "load3d_fmatrix_b = { .config = uint64_t(1) | (uint64_t(1) << 16) }" in source, source
    assert ".config = uint64_t(1) << 16" in source, source
    ordered = ["asc_set_l13d_fmatrix(", "asc_set_l13d_fmatrix_b(",
               "asc_set_l13d_rpt(", "asc_set_l13d_padding(uint64_t(0))",
               "asc_set_l13d_size(uint64_t(0))",
               "asc_sync_notify(PIPE_S, PIPE_MTE1,",
               "asc_sync_wait(PIPE_S, PIPE_MTE1,", "asc_copy_l12l0a("]
    positions = [source.index(item) for item in ordered]
    assert positions == sorted(positions), source
    # Both banks and the shared SPRs have one root setup. The unused B bank
    # receives width1; this does not promote its transposed 2D operand load.
    for setter in ordered[:5]:
        assert source.count(setter) == 1, source
    offset = 0
    while (offset := source.find("for (", offset)) >= 0:
        body, offset = _braced_body(source, offset)
        assert all(setter not in body for setter in ordered[:5]), source
    b_args = _call_arguments(source, "asc_copy_l12l0b")
    assert len(b_args) == 6 and b_args[2:] == ["16", "16", "1", "0"], source
    # BF16 uses an exact storage reinterpretation, not a numerical conversion.
    assert "vconv_bf162f32" not in source and "vconv_f322bf16" not in source, source


def _braced_body(source, start):
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin + 1:end - 1], end


def test_mixed_two_aiv_configuration_is_once_on_the_aic_and_outside_load_loops():
    @T.prim_func
    def main(a: T.Tensor((32, 256), "float16"),
             b: T.Tensor((16, 256), "float16"),
             c: T.Tensor((32, 16), "float32"),
             v: T.Tensor((32, 16), "float32"),
             out: T.Tensor((32, 16), "float32")):
        with T.MixedKernel(1, sids=2):
            a_l1 = T.alloc_l1((32, 256), "float16")
            b_l1 = T.alloc_l1((16, 256), "float16")
            a_l0 = T.alloc_l0a((32, 128), "float16")
            b_l0 = T.alloc_l0b((16, 128), "float16")
            c_l0 = T.alloc_l0c((32, 16), "float32")
            ub = T.alloc_shared((32, 16), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            for i in T.serial(2):
                for j in T.serial(2):
                    T.copy(a_l1[:, j * 128:(j + 1) * 128], a_l0)
                    T.copy(b_l1[:, j * 128:(j + 1) * 128], b_l0)
                    T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=(j == 0))
            T.copy(c_l0, c)
            T.copy(v, ub)
            for i, j in T.Parallel(32, 16):
                ub[i, j] = ub[i, j] * T.float32(2)
            T.copy(ub, out)

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    source = tilelang.lower(main, target=target).kernel_source
    assert source.count("asc_set_l13d_fmatrix(") == 1, source
    assert source.count("asc_set_l13d_rpt(") == 1, source
    assert "asc_sync_notify(PIPE_S, PIPE_MTE1," in source, source
    assert "asc_sync_wait(PIPE_S, PIPE_MTE1," in source, source
    aic_bodies = []
    aiv_bodies = []
    for marker, bodies in [("if ASC_IS_AIC", aic_bodies), ("if ASC_IS_AIV", aiv_bodies)]:
        offset = 0
        while (offset := source.find(marker, offset)) >= 0:
            body, offset = _braced_body(source, offset)
            bodies.append(body)
    assert any("asc_set_l13d_fmatrix(" in body for body in aic_bodies), source
    assert aiv_bodies and all("asc_set_l13d_" not in body for body in aiv_bodies), source
    # Configuration is not repeated by either ordinary Serial loop.
    offset = 0
    while (offset := source.find("for (", offset)) >= 0:
        body, offset = _braced_body(source, offset)
        assert "asc_set_l13d_fmatrix(" not in body, source


def test_disabled_scheduler_keeps_2d_instead_of_uncompleted_root_configuration():
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        source = tilelang.lower(_kernel("float16", True), target=target).kernel_source
    assert "asc_set_l13d_fmatrix(" not in source, source
    assert len(_call_arguments(source, "asc_copy_l12l0a")) == 6, source
