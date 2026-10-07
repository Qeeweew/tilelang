"""Copy geometry keeps int64 addresses and typed alignment arithmetic."""

import importlib.util
from pathlib import Path
import sys

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import (
    Layout,
    make_ascend_l0a_2201_layout,
    make_ascend_major_k_layout,
    make_ascend_sf_layout,
    try_extract_fractal_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes


def _i(value, dtype="int64"):
    return tirx.IntImm(dtype, value)


def _lower(body, *, buffers=(), params=(), layouts=None, bindings=None, arch="dav-2201"):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    annotations = {"layout_map": layouts or {}, "tl.l0_sf_bindings": bindings or {}}
    mod = kernel(body, buffers=buffers, params=params, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(mod)


def _equal(actual, expected, *, loops=()):
    analyzer = tvm.arith.Analyzer()
    for loop in loops:
        analyzer.bind(loop.loop_var, tvm.ir.Range.from_min_extent(loop.min, loop.extent))
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert analyzer.can_prove_equal(left, right), (left, right)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("versioned", [False, True])
def test_layout_buffer_load_and_store_preserve_mixed_index_widths(arch, index_dtype, versioned):
    shape = (2, 32, 32) if versioned else (32, 32)
    source = tirx.decl_buffer(shape, "float32", name="source")
    destination = tirx.decl_buffer(shape, "float32", name="destination")
    row, col = _i(17, index_dtype), _i(31, "int32")
    indices = [_i(1), row, col] if versioned else [row, col]
    mapping = (lambda v, i, j: [v, i // 16, j // 8, i % 16, j % 8]) if versioned else (
        lambda i, j: [i // 16, j // 8, i % 16, j % 8])
    layout = Layout(shape, mapping)
    before = tirx.BufferStore(destination, tirx.BufferLoad(source, indices), indices)
    after = _lower(before, params=[source, destination],
                   layouts={source: layout, destination: layout}, arch=arch)
    (load,), (store,) = nodes(after, tirx.BufferLoad), nodes(after, tirx.BufferStore)
    expected = [row // 16, col // 8, row % 16, col % 8]
    if versioned:
        expected.insert(0, _i(1))
    for actual in (load.indices, store.indices):
        _equal(actual, expected)
        assert [str(index.dtype) for index in actual] == [str(index.dtype) for index in expected]


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_layout_forwarding_does_not_truncate_a_large_global_coordinate(arch):
    # A virtual global tensor exercises address mapping only; no allocation,
    # device code generation or hardware run is needed for this regression.
    source = tirx.decl_buffer((_i(2**32), _i(32), _i(32)), "float32", name="large_source")
    result = tirx.decl_buffer((1,), "float32", name="result")
    index = _i(2**31 + 7)
    # The canonical layout retains a wide leading batch coordinate; all
    # trailing tile coordinates and the final mapped address remain valid.
    layout = make_ascend_major_k_layout(source)
    before = tirx.BufferStore(result, tirx.BufferLoad(source, [index, _i(17), _i(31)]), [0])
    after = _lower(before, params=[source, result], layouts={source: layout}, arch=arch)
    (load,) = nodes(after, tirx.BufferLoad)
    _equal(load.indices, [index, _i(3), _i(1), _i(1), _i(7)])
    assert all(str(value.dtype) == "int64" for value in load.indices)


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("store", [False, True], ids=["gm-to-ub", "ub-to-gm"])
def test_versioned_dma_retains_int64_offsets_and_independent_byte_pitches(index_dtype, store):
    # The frontend's version materializer gives UB explicit int64 strides even
    # when the epoch Var is int32. A wide GM leading pitch also detects an
    # accidental address narrowing while keeping every allocation shape small.
    gm_pitch = 2**32 + 128
    gm = tirx.decl_buffer((2, 4, 32), "float16", name="gm",
                         strides=[_i(gm_pitch), _i(32), _i(1)])
    ub = tirx.decl_buffer((3, 4, 64), "float16", name="ub", scope="shared.dyn",
                         strides=[_i(256), _i(64), _i(1)])
    epoch = tirx.Var("epoch", index_dtype)
    rows = epoch % 3
    gm_indices, ub_indices = [epoch % 2, 1, 0], [epoch % 3, 0, 16]
    operation = (
        copy(ub, gm, src_indices=ub_indices, dst_indices=gm_indices,
             src_shape=[1, rows, 17], dst_shape=[1, rows, 17])
        if store else
        copy(gm, ub, src_indices=gm_indices, dst_indices=ub_indices,
             src_shape=[1, rows, 17], dst_shape=[1, rows, 17])
    )
    before = tirx.For(epoch, _i(0, index_dtype), _i(6, index_dtype), tirx.ForKind.SERIAL, operation)
    after = _lower(before, buffers=[ub], params=[gm])
    (dma,) = calls(after, "tl.ascend_copy_ubuf_to_gm" if store else "tl.ascend_copy_gm_to_ubuf")
    (actual_loop,) = nodes(after, tirx.For)
    actual_epoch = actual_loop.loop_var
    expected_gm = (actual_epoch % 2) * _i(gm_pitch) + _i(32)
    expected_ub = (actual_epoch % 3) * _i(256) + _i(16)
    expected_offsets = [expected_gm, expected_ub] if store else [expected_ub, expected_gm]
    _equal([dma.args[0].args[2], dma.args[1].args[2]], expected_offsets)
    assert str(dma.args[0].args[2].dtype) == str(dma.args[1].args[2].dtype) == "int64"
    # Two 17-element half rows require independent 64B GM and 128B UB pitches;
    # the 34-byte payload must not become a collapsed contiguous copy.
    pitch_positions = (6, 7) if store else (9, 10)
    _equal([dma.args[3], dma.args[4], *(dma.args[i] for i in pitch_positions)],
           [actual_epoch % 3, 34, 64, 128])
    (guard,) = nodes(after, tirx.IfThenElse)
    analyzer = tvm.arith.Analyzer()
    for iteration in range(3):
        predicate = tirx.stmt_functor.substitute(guard.condition, {actual_epoch: _i(iteration, index_dtype)})
        assert bool(analyzer.simplify(predicate)) == (iteration != 0)


@pytest.mark.parametrize("bad_origin", [False, True], ids=["unaligned-row-pitch", "unaligned-origin"])
def test_int64_geometry_still_rejects_illegal_ub_alignment(bad_origin):
    gm = tirx.decl_buffer((2, 32), "float16", name="gm", strides=[_i(32), _i(1)])
    pitch = 64 if bad_origin else 17
    ub = tirx.decl_buffer((2, pitch), "float16", name="ub", scope="shared.dyn",
                         strides=[_i(pitch), _i(1)])
    operation = copy(gm, ub, src_indices=[_i(0), _i(0)],
                     dst_indices=[_i(0), _i(1 if bad_origin else 0)],
                     src_shape=[_i(2), _i(16)], dst_shape=[_i(2), _i(16)])
    diagnostic = "32-byte-aligned UB origin" if bad_origin else "32-byte-aligned UB row pitch"
    with pytest.raises(ValueError, match=diagnostic):
        _lower(operation, buffers=[ub], params=[gm])


@pytest.mark.parametrize("role", ["a", "b"])
def test_int64_native_fractal_regions_preserve_independent_versions(role):
    source = tirx.decl_buffer((_i(2), _i(16), _i(32)), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((_i(3), _i(16), _i(16)), "float16", name="destination", scope="shared.l0" + role)
    epoch = tirx.Var("epoch", "int64")
    operation = copy(source, destination, src_indices=[epoch % 2, _i(0), _i(16)],
                     dst_indices=[epoch % 3, _i(0), _i(0)],
                     src_shape=[_i(1), _i(16), _i(16)], dst_shape=[_i(1), _i(16), _i(16)])
    destination_layout = make_ascend_l0a_2201_layout(destination) if role == "a" else make_ascend_major_k_layout(destination)
    after = _lower(tirx.For(epoch, _i(0), _i(6), tirx.ForKind.SERIAL, operation),
                   buffers=[source, destination], layouts={
                       source: make_ascend_major_k_layout(source), destination: destination_layout,
                   })
    (load,) = calls(after, "tl.ascend_load_cbuf_to_c" + role + "_2d")
    actual_loops = nodes(after, tirx.For)
    actual_epoch = next(loop.loop_var for loop in actual_loops if str(loop.loop_var.name) == "epoch")
    _equal([load.args[0].args[2], load.args[1].args[2]],
           [(actual_epoch % 3) * 256, (actual_epoch % 2) * 512], loops=actual_loops)
    assert str(load.args[0].args[2].dtype) == str(load.args[1].args[2].dtype) == "int64"


def test_3510_transpose_divisors_follow_int64_fractal_extents():
    source = tirx.decl_buffer((_i(32), _i(16)), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((_i(16), _i(32)), "float16", name="destination", scope="shared.l0a")
    operation = copy(source, destination, src_indices=[_i(0), _i(0)], dst_indices=[_i(0), _i(0)],
                     src_shape=[_i(32), _i(16)], dst_shape=[_i(16), _i(32)],
                     annotations={"transpose": _i(1, "int32")})
    after = _lower(operation, buffers=[source, destination], arch="dav-3510", layouts={
        source: make_ascend_major_k_layout(source), destination: make_ascend_major_k_layout(destination),
    })
    (load,) = calls(after, "tl.ascend_load_cbuf_to_ca")
    assert int(load.args[8]) == 1
    _equal(load.args[4:6], [2, 1])


def test_3510_raw_ub_to_l1_payload_modulus_keeps_int64_count():
    source = tirx.decl_buffer((_i(64),), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((_i(64),), "float32", name="destination", scope="shared.l1")
    after = _lower(copy(source, destination), buffers=[source, destination], arch="dav-3510")
    (dma,) = calls(after, "tl.ascend_copy_ubuf_to_cbuf")
    _equal([dma.args[3]], [8])
    assert str(dma.args[3].dtype) == "int64"


def test_3510_scale_factor_moduli_and_compact_stride_keep_int64_regions():
    data = tirx.decl_buffer((_i(2), _i(64), _i(128)), "float8_e4m3fn", name="data", scope="shared.l0a")
    scales = tirx.decl_buffer((_i(2), _i(64), _i(4)), "uint8", name="scales", scope="shared.l0a.sf")
    source = tirx.decl_buffer((_i(64), _i(8)), "uint8", name="source", scope="shared.l1")
    operation = copy(source, scales, src_indices=[_i(16), _i(2)], dst_indices=[_i(1), _i(0), _i(0)],
                     src_shape=[_i(32), _i(2)], dst_shape=[_i(1), _i(32), _i(2)])
    after = _lower(operation, buffers=[source, scales, data], arch="dav-3510", layouts={
        source: make_ascend_sf_layout(source), data: make_ascend_major_k_layout(data),
    }, bindings={scales.data: data.data})
    (load,) = calls(after, "tl.ascend_load_ca_sf")
    _equal([load.args[0].args[2]], [8192])
    _equal(load.args[2:], [1, 1, 2, 1, 4, 1])


@pytest.mark.parametrize("index_dtype,bits", [("int32", 32), ("int64", 64)])
def test_fractal_ceildiv_preserves_shape_dtype_without_endpoint_overflow(index_dtype, bits):
    # This tests metadata arithmetic without allocating an impossible L1 tile.
    # Runtime resource checks remain responsible for actual storage capacity.
    largest = 2 ** (bits - 1) - 1
    source = tirx.decl_buffer((_i(largest, index_dtype), _i(16, index_dtype)),
                              "float16", name="shape_only", scope="shared.l1")
    info = try_extract_fractal_layout(make_ascend_major_k_layout(source), source)
    assert info is not None
    _equal([info.outer0, info.outer1], [1, largest // 16 + 1])
    assert str(info.outer0.dtype) == str(info.outer1.dtype) == index_dtype


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_batched_gqa_pipeline_lowers_versioned_ub_copies_without_device_compile(dtype):
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    if not path.exists():
        pytest.skip("The full frontend regression needs the product FA example beside testing/")
    spec = importlib.util.spec_from_file_location("_copy_index_dtypes_flash_attention_910b", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    program = example.flash_attention_fwd(
        2, 4, 2, 17, 33, 16, dtype=dtype, variant="pipeline",
        tiling=example.FwdTiling(16, 16, 4),
    )
    artifact = tilelang.lower(program, target={"kind": "ascend", "arch": "dav-2201"},
                              target_host="c", enable_host_codegen=False, enable_device_compile=False)
    assert "asc_copy_gm2ub_align" in artifact.kernel_source
    assert "asc_copy_l12l0a" in artifact.kernel_source
