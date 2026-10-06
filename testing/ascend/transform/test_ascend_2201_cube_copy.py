"""2201 native LoadData2D preserves NZ source and role-specific MAD layout."""

import pytest

from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    try_extract_fractal_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes


def _region(buffer, mins, shape):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, shape)])


def _lower(src, dst, source_region, layouts, destination_region=None):
    body = tirx.Evaluate(T.copy(source_region, dst if destination_region is None else destination_region))
    mod = kernel(body, buffers=[src, dst], annotations={"layout_map": layouts})
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", target)})
    with target:
        return transform.AscendLowerTileOp()(mod)


def _equal(actual, expected):
    assert len(actual) == len(expected)
    analyzer = tvm.arith.Analyzer()
    for a, e in zip(actual, expected):
        assert analyzer.can_prove_equal(a, e), (a, e)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("role,transpose", [("a", False), ("a", True), ("b", False), ("b", True)])
def test_native_2d_load_follows_mad_role_and_source_nz_pitch(dtype, role, transpose):
    source_shape = (96, 64) if transpose else (64, 96)
    tile_shape = (48, 32) if transpose else (32, 48)
    mins = (32, 16) if transpose else (16, 32)
    src = tirx.decl_buffer(source_shape, dtype, name="src", scope="shared.l1")
    dst = tirx.decl_buffer(tile_shape, dtype, name="dst", scope="shared.l0" + role)
    if role == "a":
        dst_layout = make_ascend_l0a_2201_layout(dst, transpose)
    else:
        dst_layout = (make_ascend_major_mn_layout if transpose else make_ascend_major_k_layout)(dst)
    after = _lower(src, dst, _region(src, mins, tile_shape), {src: make_ascend_major_k_layout(src), dst: dst_layout})
    (load,) = calls(after, "tl.ascend_load_cbuf_to_c" + role + "_2d")
    (loop,) = nodes(after, tirx.For)
    frac = loop.loop_var
    expected = {
        ("a", False): (2, 9 + frac, 3, 4, frac * 3 * 256),
        ("a", True): (2, 8 + frac * 6, 3, 1, frac * 3 * 256),
        ("b", False): (3, 9 + frac * 4, 2, 1, frac * 2 * 256),
        ("b", True): (3, 8 + frac, 2, 6, frac * 2 * 256),
    }[role, transpose]
    count, start, repeat, stride, dst_offset = expected
    _equal([loop.extent, *load.args[2:]], [count, start, repeat, stride, 0, int(transpose)])
    assert load.args[0].op.name == "tirx.tvm_access_ptr"
    _equal([load.args[0].args[2]], [dst_offset])
    assert not calls(after, "tl.ascend_load_cbuf_to_c" + role)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("n", [127, 128])
def test_full_physical_n_native_b_load_is_one_descriptor(dtype, n):
    # NZ pads logical N=127 to eight row fractals, just as N=128. The source
    # K origin selects the second half of a larger L1 cache tile.
    src = tirx.decl_buffer((n, 256), dtype, name="src", scope="shared.l1")
    dst = tirx.decl_buffer((n, 128), dtype, name="dst", scope="shared.l0b")
    after = _lower(src, dst, _region(src, [0, 128], [n, 128]),
                   {src: make_ascend_major_k_layout(src), dst: make_ascend_major_k_layout(dst)})
    (load,) = calls(after, "tl.ascend_load_cbuf_to_cb_2d")
    assert not nodes(after, tirx.For)
    _equal(load.args[2:], [64, 64, 1, 0, 0])
    _equal([load.args[0].args[2], load.args[1].args[2]], [0, 0])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_merged_native_b_load_keeps_leading_version_in_each_pointer(dtype):
    src = tirx.decl_buffer((2, 128, 256), dtype, name="src", scope="shared.l1")
    dst = tirx.decl_buffer((2, 128, 128), dtype, name="dst", scope="shared.l0b")
    after = _lower(src, dst, _region(src, [1, 0, 128], [1, 128, 128]),
                   {src: make_ascend_major_k_layout(src), dst: make_ascend_major_k_layout(dst)},
                   _region(dst, [1, 0, 0], [1, 128, 128]))
    (load,) = calls(after, "tl.ascend_load_cbuf_to_cb_2d")
    assert not nodes(after, tirx.For)
    _equal(load.args[2:], [64, 64, 1, 0, 0])
    _equal([load.args[0].args[2], load.args[1].args[2]], [128 * 128, 128 * 256])


def test_full_n_load_keeps_segmented_descriptors_when_total_repeat_exceeds_uint8():
    # Isolate the instruction field boundary: resource validation of this
    # oversized L0B allocation belongs to the earlier allocation boundary.
    src = tirx.decl_buffer((256, 256), "float16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((256, 256), "float16", name="dst", scope="shared.l0b")
    after = _lower(src, dst, _region(src, [0, 0], src.shape),
                   {src: make_ascend_major_k_layout(src), dst: make_ascend_major_k_layout(dst)})
    (load,) = calls(after, "tl.ascend_load_cbuf_to_cb_2d")
    (loop,) = nodes(after, tirx.For)
    _equal([loop.extent, *load.args[2:]], [16, loop.loop_var * 16, 16, 1, 0, 0])


@pytest.mark.parametrize("transpose", [False, True])
def test_native_l0a_tag_preserves_actual_za_order(transpose):
    shape = (48, 32) if transpose else (32, 48)
    buf = tirx.decl_buffer(shape, "bfloat16", scope="shared.l0a")
    layout = make_ascend_l0a_2201_layout(buf, transpose)
    info = try_extract_fractal_layout(layout, buf)
    assert info.kind == (5 if transpose else 4)
    indices = [34, 18] if transpose else [18, 34]
    _equal(layout.map_forward_index(indices), [1, 2, 2, 2])
    _equal([info.outer0, info.outer1], [3, 2])


def test_native_l0a_rejects_mislabeled_zb_layout():
    src = tirx.decl_buffer((32, 48), "float16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((32, 48), "float16", name="dst", scope="shared.l0a")
    with pytest.raises(tvm.error.InternalError, match="native ZA layout"):
        _lower(src, dst, _region(src, [0, 0], src.shape), {src: make_ascend_major_k_layout(src), dst: make_ascend_major_k_layout(dst)})


def test_native_load_rejects_hidden_destination_pitch():
    src = tirx.decl_buffer((32, 48), "float16", name="src", scope="shared.l1")
    dst = tirx.decl_buffer((64, 48), "float16", name="dst", scope="shared.l0a")
    mod = kernel(
        tirx.Evaluate(T.copy(src, _region(dst, [0, 0], [32, 48]))),
        buffers=[src, dst],
        annotations={"layout_map": {src: make_ascend_major_k_layout(src), dst: make_ascend_l0a_2201_layout(dst)}},
    )
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", target)})
    with target, pytest.raises(tvm.error.InternalError, match="compact padded L0 tile"):
        transform.AscendLowerTileOp()(mod)
