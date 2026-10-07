"""Native 2201 Cube kernels use the installed 2D DMA and Fixpipe signatures."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T


def _kernel(dtype, transpose_b):
    m, n, block_k, total_k = 32, 48, 64, 128
    b_shape = (n, total_k) if transpose_b else (total_k, n)
    b_tile = (n, block_k) if transpose_b else (block_k, n)

    @T.prim_func
    def main(A: T.Tensor((m, total_k), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((m, n), dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, block_k), dtype)
            b_l1 = T.alloc_l1(b_tile, dtype)
            a_l0 = T.alloc_l0a((m, block_k), dtype)
            b_l0 = T.alloc_l0b(b_tile, dtype)
            c_l0 = T.alloc_l0c((m, n), "float32")
            for k in T.Pipelined(total_k // block_k):
                T.copy(A[:, k * block_k : (k + 1) * block_k], a_l1)
                if transpose_b:
                    T.copy(B[:, k * block_k : (k + 1) * block_k], b_l1)
                else:
                    T.copy(B[k * block_k : (k + 1) * block_k, :], b_l1)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.gemm(a_l0, b_l0, c_l0, transpose_B=transpose_b, clear_accum=k == 0)
            T.copy(c_l0, C)

    return main


def _call_arguments(source, name):
    start = source.index(name + "(") + len(name) + 1
    level, begin, arguments = 0, start, []
    for index in range(start, len(source)):
        char = source[index]
        if char in "({[":
            level += 1
        elif char in ")}]":
            if char == ")" and level == 0:
                arguments.append(source[begin:index].strip())
                return arguments
            level -= 1
        elif char == "," and level == 0:
            arguments.append(source[begin:index].strip())
            begin = index + 1
    raise AssertionError("Unterminated generated call: " + name)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["nn", "nt"])
def test_automatic_pipeline_emits_native_cube_dma_signatures(dtype, transpose_b):
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    source = tilelang.lower(_kernel(dtype, transpose_b), target=target).kernel_source
    gm_args = _call_arguments(source, "asc_copy_gm2l1_nd2nz")
    assert len(gm_args) == 10, source
    assert gm_args[2] == "1" and gm_args[5] == "0" and gm_args[8:] == ["1", "0"]
    assert "sizeof(*(" in gm_args[6]
    assert len(_call_arguments(source, "asc_copy_l12l0a")) == 17, source
    assert "asc_set_l13d_fmatrix(" in source and "asc_set_l13d_rpt(" in source, source
    if transpose_b:
        assert len(_call_arguments(source, "asc_copy_l12l0b")) == 6, source
    else:
        transpose_args = _call_arguments(source, "load_cbuf_to_cb")
        assert len(transpose_args) == 9 and transpose_args[7] == "1", source
    assert len(_call_arguments(source, "asc_copy_l0c2gm")) == 11, source
    assert _call_arguments(source, "asc_set_l0c_copy_params") == ["1", "0", "0"], source
    ordered = ["asc_set_l0c_copy_params(", "asc_sync_notify(PIPE_S, PIPE_FIX,",
               "asc_sync_wait(PIPE_S, PIPE_FIX,", "asc_copy_l0c2gm("]
    assert [source.index(item) for item in ordered] == sorted(source.index(item) for item in ordered), source
    assert "asc_mmad(" in source
    assert "asc_set_gm2l1_nz_para(" not in source
    assert "asc_set_l0c_copy_nz_para(" not in source
    assert "asc_copy_gm2l1_dn2nz(" not in source
    assert "asc_copy_l12l0b_transpose(" not in source
    assert "tl::ascend_gemm_l1" not in source
