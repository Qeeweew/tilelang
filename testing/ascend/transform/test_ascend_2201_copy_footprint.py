"""LoadData2D cannot infer extra source storage from a larger L0 tile."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, kernel


TARGET = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})


def _region(buffer, origins, extents):
    return tirx.BufferRegion(
        buffer, [tvm.ir.Range.from_min_extent(origin, extent) for origin, extent in zip(origins, extents)]
    )


def _lower_copy(source, destination, source_region, *, transpose=False, destination_region=None, params=()):
    destination_layout = (
        make_ascend_l0a_2201_layout(destination, transpose)
        if destination.scope() == "shared.l0a"
        else (make_ascend_major_mn_layout if transpose else make_ascend_major_k_layout)(destination)
    )
    mod = kernel(
        tirx.Evaluate(T.copy(source_region, destination if destination_region is None else destination_region)),
        buffers=[source, destination], params=params,
        annotations={"layout_map": {source: make_ascend_major_k_layout(source), destination: destination_layout}},
    )
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", TARGET)})
    with TARGET:
        return transform.AscendLowerTileOp()(mod)


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("axis", [0, 1], ids=["row16", "c0"])
def test_native_load_does_not_round_down_inner_source_origin(role, transpose, axis):
    source = tirx.decl_buffer((32, 32), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((16, 16), "float16", name="destination", scope="shared.l0" + role)
    origins = [0, 0]
    origins[axis] = 1
    with pytest.raises(tvm.error.InternalError, match="source.*origin"):
        _lower_copy(source, destination, _region(source, origins, [16, 16]), transpose=transpose)


@pytest.mark.parametrize("role", ["a", "b"])
@pytest.mark.parametrize("transpose", [False, True])
def test_source_region_cannot_authorize_reads_beyond_l1_allocation(role, transpose):
    source = tirx.decl_buffer((16, 16), "bfloat16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((16, 16), "bfloat16", name="destination", scope="shared.l0" + role)
    with pytest.raises(tvm.error.InternalError, match="source footprint.*physical L1 allocation"):
        _lower_copy(source, destination, _region(source, [16, 0], [16, 16]), transpose=transpose)


@pytest.mark.parametrize("operand", ["source", "destination"])
def test_native_load_checks_each_leading_version_index(operand):
    source = tirx.decl_buffer((2, 16, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((2, 16, 16), "float16", name="destination", scope="shared.l0b")
    source_index = 2 if operand == "source" else 1
    destination_index = 2 if operand == "destination" else 0
    with pytest.raises(tvm.error.InternalError, match="leading/version axis.*allocation " + operand):
        _lower_copy(
            source, destination, _region(source, [source_index, 0, 0], [1, 16, 16]),
            destination_region=_region(destination, [destination_index, 0, 0], [1, 16, 16]),
        )


def test_unknown_aligned_source_bounds_are_diagnosed():
    source = tirx.decl_buffer((32, 32), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((16, 16), "float16", name="destination", scope="shared.l0b")
    index = tirx.Var("index", "int32")
    with pytest.raises(tvm.error.InternalError, match="source footprint.*cannot be proven"):
        _lower_copy(source, destination, _region(source, [index * 16, 0], [16, 16]), params=[index])


def test_loop_bounded_subk_keeps_independent_source_destination_versions():
    source = tirx.decl_buffer((2, 16, 32), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((2, 16, 16), "float16", name="destination", scope="shared.l0b")
    epoch = tirx.Var("epoch", "int32")
    source_version, destination_version = epoch % 2, (epoch + 1) % 2
    operation = T.copy(
        _region(source, [source_version, 0, (epoch % 2) * 16], [1, 16, 16]),
        _region(destination, [destination_version, 0, 0], [1, 16, 16]),
    )
    mod = kernel(
        tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, tirx.Evaluate(operation)),
        buffers=[source, destination],
        annotations={"layout_map": {
            source: make_ascend_major_k_layout(source), destination: make_ascend_major_k_layout(destination),
        }},
    )
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", TARGET)})
    with TARGET:
        after = transform.AscendLowerTileOp()(mod)
    (load,) = calls(after, "tl.ascend_load_cbuf_to_cb_2d")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(load.args[0].args[2], destination_version * 256)
    assert analyzer.can_prove_equal(load.args[1].args[2], source_version * 512)
    assert analyzer.can_prove_equal(load.args[2], epoch % 2)


def _gemm(source_capacity, requested, dtype):
    @T.prim_func
    def main(A: T.Tensor((32, 32), dtype), B: T.Tensor((16, 16), dtype), C: T.Tensor((32, 32), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((32, 32), dtype)
            b_l1 = T.alloc_l1((source_capacity, source_capacity), dtype)
            a_l0 = T.alloc_l0a((32, 32), dtype)
            b_l0 = T.alloc_l0b((32, 32), dtype)
            c_l0 = T.alloc_l0c((32, 32), "float32")
            T.copy(A, a_l1)
            T.copy(B[0:source_capacity, 0:source_capacity], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1[0:requested, 0:requested], b_l0[0:32, 0:32])
            T.gemm(a_l0, b_l0, c_l0, transpose_B=True, clear_accum=True)
            T.copy(c_l0, C)
    return main


def _lower(program):
    return tilelang.lower(
        program, target=TARGET, target_host="c",
        enable_host_codegen=False, enable_device_compile=False,
    )


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("requested,diagnostic", [
    (16, "matching physical source/destination fractal footprints"),
    (32, "source footprint.*physical L1 allocation"),
], ids=["smaller-inbounds-source-region", "out-of-bounds-l1-source-region"])
def test_full_pipeline_rejects_missing_l1_storage_after_oob_padding(dtype, requested, diagnostic):
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _lower(_gemm(16, requested, dtype))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_pipeline_keeps_gm_tail_with_real_padded_l1_storage(dtype):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendInsertOOBPadding":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = _lower(_gemm(32, 32, dtype))
    # The GM tail is represented by real L1 zero writers before scheduling.
    assert len(calls(snapshots["tl.AscendInsertOOBPadding"], "tl.tileop.fill")) == 2
    assert artifact.kernel_source.count("asc_fill_l1(") == 2
    assert "asc_copy_l12l0b(" in artifact.kernel_source
    assert ", 0, 4, 1, 0);" in artifact.kernel_source
