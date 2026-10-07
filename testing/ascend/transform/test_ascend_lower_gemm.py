"""L0 MAD uses logical operand regions while retaining physical allocation sizes."""

import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_l0a_2201_layout,
    make_ascend_l0c_layout,
)
from tvm import tirx
from testing.ascend._ir import calls, allocated_buffer, kernel


def _lower(m, n, k, *, blockscaled=False):
    dtype = "float8_e4m3fn" if blockscaled else "float32"
    a = tirx.decl_buffer((64, 128), dtype, name="a", scope="shared.l0a")
    b = tirx.decl_buffer((64, 128), dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="c", scope="shared.l0c")

    def region(buffer, rows, cols):
        return tirx.BufferRegion(buffer, [tvm.ir.Range(0, rows), tvm.ir.Range(0, cols)])

    bindings = {}
    alloc_buffers = [a, b, c]
    if blockscaled:
        sfa = tirx.decl_buffer((64, 2), "int16", name="sfa", scope="shared.l0a.sf")
        sfb = tirx.decl_buffer((64, 2), "int16", name="sfb", scope="shared.l0b.sf")
        alloc_buffers += [sfa, sfb]
        bindings = {sfa.data: a.data, sfb.data: b.data}
        call = T.gemm_blockscaled(
            region(a, m, k), region(b, n, k), region(c, m, n), region(sfa, m, 2), region(sfb, n, 2), transpose_B=True, clear_accum=True
        )
    else:
        call = T.gemm(region(a, m, k), region(b, n, k), region(c, m, n), transpose_B=True, clear_accum=True)
    body = tirx.Evaluate(call)
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        body,
        alloc_buffers=alloc_buffers,
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )
    target = tvm.target.Target("ascend")
    body = tirx.SBlockRealize([], True, root)
    before = tirx.PrimFunc(tirx.analysis.undefined_vars(body), body).with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(tvm.IRModule({"main": before}))


@pytest.mark.parametrize("blockscaled", [False, True], ids=["mad", "blockscaled-mad"])
@pytest.mark.parametrize("symbolic", [False, True], ids=["static-region", "dynamic-region"])
def test_mad_uses_region_geometry_without_shrinking_storage(blockscaled, symbolic):
    m, n = (tirx.Var("m", "int32"), tirx.Var("n", "int32")) if symbolic else (17, 19)
    k = 128 if blockscaled else 24
    after = _lower(m, n, k, blockscaled=blockscaled)
    (mad,) = calls(after, "tl.ascend_mad_mx" if blockscaled else "tl.ascend_mad")
    analyzer = tvm.arith.Analyzer()
    for actual, expected in zip(mad.args[3:6], [m, k, n]):
        assert analyzer.can_prove_equal(actual, expected)
    for name in ("a", "b"):
        assert math.prod(int(x) for x in allocated_buffer(after, name).shape) == 64 * 128


def _sf_gemm(case):
    a = tirx.decl_buffer((2, 3, 32, 128), "float8_e4m3fn", name="a", scope="shared.l0a")
    a2 = tirx.decl_buffer(a.shape, a.dtype, name="a2", scope="shared.l0a")
    b = tirx.decl_buffer((32, 128), a.dtype, name="b", scope="shared.l0b")
    c = tirx.decl_buffer((32, 32), "float32", name="c", scope="shared.l0c")
    sf = tirx.decl_buffer((2, 3, 32, 2), "uint16", name="sf", scope="shared.l0b.sf" if case == "scope" else "shared.l0a.sf")
    sb = tirx.decl_buffer((32, 2), "uint16", name="sb", scope="shared.l0b.sf")
    bindings = {sf.data: a2.data if case == "wrong-data" else a.data, sb.data: b.data}
    if case == "missing":
        del bindings[sf.data]
    if case == "duplicate":
        bindings[tirx.decl_buffer(sf.shape, sf.dtype, scope="shared.l0a.sf").data] = a.data

    def region(buf, mins, extents):
        return tirx.BufferRegion(buf, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])

    ar = region(a, [1, 2, 0, 0], [1, 1, 17, 64 if case == "compact" else 128])
    sr = region(sf, [0 if case == "leading" else 1, 2, 1 if case == "origin" else 0, 0], [1, 1, 17, 1 if case in ("k", "compact") else 2])
    br = region(b, [0, 0], [32, 64 if case == "compact" else 128])
    cr = region(c, [0, 0], [17, 32])
    sbr = region(sb, [0, 0], [32, 1 if case == "compact" else 2])
    call = T.gemm_blockscaled(ar, br, cr, sr, sbr, transpose_B=True, clear_accum=True)
    return kernel(
        tirx.Evaluate(call),
        buffers=[a, a2, b, c, sf, sb],
        annotations={
            "tl.l0_sf_bindings": bindings,
            "layout_map": {a: make_ascend_major_k_layout(a), b: make_ascend_major_k_layout(b), c: make_ascend_l0c_layout(c)},
        },
    )


@pytest.mark.parametrize(
    "case,diagnostic",
    [
        ("wrong-data", "is bound to"),
        ("missing", "no allocation binding"),
        ("scope", "matching L0A/L0B SF scope"),
        ("leading", "same leading index"),
        ("origin", "zero-origin"),
        ("k", "compact data K extent"),
        ("duplicate", "multiple SF handles"),
    ],
)
def test_invalid_binding_and_slice_are_rejected(case, diagnostic):
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        transform.AscendLowerTileOp()(_sf_gemm(case))


@pytest.mark.parametrize("case", ["valid", "compact"])
def test_matching_leading_indices_and_compact_regions(case):
    after = transform.AscendLowerTileOp()(_sf_gemm(case))
    (mad,) = calls(after, "tl.ascend_mad_mx")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(mad.args[1].args[2], 5 * 32 * 128)
    assert analyzer.can_prove_equal(mad.args[4], 64 if case == "compact" else 128)


def test_mismatched_sf_binding_fails_before_layout_inference():
    with pytest.raises(tvm.error.InternalError, match="is bound to"):
        transform.AscendLayoutInference()(_sf_gemm("wrong-data"))


def _column_mad(*, origin=64, n=64, m=128, rows=128, cols=256, row_origin=0,
                a_origin=0, b_origin=0, owner_extent=None, leading=False,
                leading_extent=1, transpose_b=True, clear=True,
                layout="canonical", strides=False, arch="dav-2201"):
    """One ordinary L0 MAD; the owner domain belongs to the real TIR loop."""
    owner = tirx.Var("owner", "int32")
    slot = tirx.Var("slot", "int32")
    if owner_extent is not None:
        origin = owner * n
    a = tirx.decl_buffer((m + (16 if a_origin else 0), 128), "float16", name="a", scope="shared.l0a")
    b_extent = n + (16 if b_origin else 0)
    b = tirx.decl_buffer((b_extent, 128) if transpose_b else (128, b_extent),
                        "float16", name="b", scope="shared.l0b")
    shape = (2, rows, cols) if leading else (rows, cols)
    c = tirx.decl_buffer(shape, "float32", name="c", scope="shared.l0c",
                        strides=([rows * cols, cols * 2, 1] if leading else [cols * 2, 1]) if strides else None)
    c_view = tirx.decl_buffer(shape, c.dtype, name="c_view", data=c.data, scope=c.scope())
    operand_c = c_view if layout == "operand-view" else c

    def region(buffer, mins, extents):
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(x, y) for x, y in zip(mins, extents)])

    ar = region(a, [a_origin, 0], [m, 128])
    br = region(b, [b_origin, 0] if transpose_b else [0, b_origin], [n, 128] if transpose_b else [128, n])
    cr = region(operand_c, [slot, row_origin, origin] if leading else [row_origin, origin],
                [leading_extent, m, n] if leading else [m, n])
    call = T.gemm(ar, br, cr, transpose_B=transpose_b, clear_accum=clear)
    body = tirx.Evaluate(call)
    if owner_extent is not None:
        body = tirx.For(owner, 0, owner_extent, tirx.ForKind.SERIAL, body)
    if leading:
        body = tirx.For(slot, 0, 2, tirx.ForKind.SERIAL, body)
    layouts = {a: make_ascend_l0a_2201_layout(a),
               b: make_ascend_major_k_layout(b) if transpose_b else make_ascend_major_mn_layout(b)}
    allocations = [a, b, c]
    if layout == "same-name-other-data":
        other = tirx.decl_buffer(shape, c.dtype, name=c.name, scope="shared.l0c")
        assert other.name == c.name and not other.data.same_as(c.data)
        layouts[other] = make_ascend_l0c_layout(other)
        allocations.append(other)
    elif layout in ("same-data-layout-key", "operand-view"):
        assert not c_view.same_as(c) and c_view.data.same_as(c.data)
        layouts[c_view] = make_ascend_l0c_layout(c_view)
    elif layout != "missing":
        layouts[c] = make_ascend_l0c_layout(c) if layout == "canonical" else make_ascend_major_k_layout(c)
    mod = kernel(body, buffers=allocations, annotations={"layout_map": layouts})
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return mod


def _eval_offset(expression, **values):
    replacements = {var: tirx.IntImm(var.dtype, values[var.name]) for var in tirx.analysis.undefined_vars(expression)}
    value = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expression, replacements))
    assert isinstance(value, tirx.IntImm), value
    return int(value)


@pytest.mark.parametrize("transpose_b", [False, True], ids=["NN", "NT"])
@pytest.mark.parametrize("clear", [False, True], ids=["accumulate", "clear"])
def test_2201_c_column_regions_use_native_nz_pointer_and_operation_extent(transpose_b, clear):
    after = transform.AscendLowerTileOp()(_column_mad(owner_extent=2, transpose_b=transpose_b, clear=clear))
    (mad,) = calls(after, "tl.ascend_mad")
    assert [int(value) for value in mad.args[3:6]] == [128, 128, 64]
    assert bool(int(mad.args[9])) == clear
    c_ptr, a_ptr, b_ptr = mad.args[:3]
    assert int(c_ptr.args[3]) == 128 * 64
    assert int(c_ptr.args[4]) == 2
    assert math.prod(int(x) for x in allocated_buffer(after, "c").shape) == 128 * 256
    for owner in range(2):
        # Logical [0, owner*64] maps to a complete, contiguous NZ N region.
        assert _eval_offset(c_ptr.args[2], owner=owner) == owner * 64 * 128
    assert int(a_ptr.args[2]) == int(b_ptr.args[2]) == 0
    assert int(a_ptr.args[3]) == 128 * 128
    assert int(b_ptr.args[3]) == 64 * 128


def test_2201_c_column_region_preserves_leading_plane_and_disjoint_owner_ranges():
    after = transform.AscendLowerTileOp()(
        _column_mad(rows=32, m=32, cols=128, n=32, owner_extent=2, leading=True)
    )
    (mad,) = calls(after, "tl.ascend_mad")
    pointer = mad.args[0]
    assert int(pointer.args[3]) == 32 * 32
    assert math.prod(int(x) for x in allocated_buffer(after, "c").shape) == 2 * 32 * 128
    touched = set()
    for slot in range(2):
        for owner in range(2):
            offset = _eval_offset(pointer.args[2], slot=slot, owner=owner)
            assert offset == slot * 32 * 128 + owner * 32 * 32
            addresses = set(range(offset, offset + int(pointer.args[3])))
            assert not (touched & addresses)
            touched.update(addresses)


@pytest.mark.parametrize("origin", [16, 128, 192])
def test_2201_static_aligned_c_column_origin(origin):
    after = transform.AscendLowerTileOp()(_column_mad(origin=origin))
    (mad,) = calls(after, "tl.ascend_mad")
    assert int(mad.args[0].args[2]) == origin * 128
    assert int(mad.args[0].args[3]) == 128 * 64


@pytest.mark.parametrize(
    "kwargs,diagnostic",
    [
        ({"origin": 1}, "16-fractal-aligned"),
        ({"n": 17}, "16-fractal-aligned"),
        ({"rows": 127, "m": 127}, "16-fractal-aligned"),
        ({"row_origin": 16, "m": 96}, "zero-origin full M"),
        ({"m": 64}, "zero-origin full M"),
        ({"a_origin": 1}, "region A requires zero origin"),
        ({"b_origin": 1}, "region B requires zero origin"),
        ({"layout": "wrong-kind"}, "canonical L0C NZ layout"),
        ({"layout": "missing"}, "canonical L0C NZ layout"),
        ({"layout": "same-name-other-data"}, "canonical L0C NZ layout"),
        ({"layout": "same-data-layout-key"}, "canonical L0C NZ layout"),
        ({"strides": True}, "compact backing storage"),
        ({"arch": "dav-3510"}, "regular dav-2201 MAD"),
    ],
)
def test_c_column_region_encoding_rejects_unsupported_geometry(kwargs, diagnostic):
    with pytest.raises((tvm.error.InternalError, AssertionError, ValueError), match=diagnostic):
        transform.AscendLowerTileOp()(_column_mad(**kwargs))


@pytest.mark.parametrize("kwargs", [
    {"origin": -64}, {"origin": 256}, {"owner_extent": 5},
    {"origin": tirx.Var("unbounded_column", "int32") * 64},
])
def test_c_column_region_requires_loop_aware_backing_bounds(kwargs):
    # The native remap delinearizes a logical flat pointer using modulo.
    # Reject out-of-domain values first, rather than wrapping into another
    # row/plane or pretending the Python callback knows the owner bounds.
    with pytest.raises((tvm.error.InternalError, ValueError), match="column.*(inside|bounds|backing)"):
        transform.AscendLowerTileOp()(_column_mad(**kwargs))


def test_c_column_region_operand_view_must_match_actual_allocation_identity():
    # A layout keyed by the exact operand view is still insufficient if the
    # pointer remapper resolves its data to a different allocation Buffer.
    # Reject before allowing a logical n0 offset to bypass the NZ remap.
    with pytest.raises((tvm.error.InternalError, ValueError), match="column.*allocation"):
        transform.AscendLowerTileOp()(_column_mad(layout="operand-view"))
