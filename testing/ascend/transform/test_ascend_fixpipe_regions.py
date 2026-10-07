"""Fixpipe valid regions never change their canonical L0C storage pitch."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout, make_ascend_major_k_layout
from tvm import tirx
from testing.ascend._ir import calls, kernel


def _region(buffer, mins, extents):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(m, e) for m, e in zip(mins, extents)])


def _lower(arch, source, destination, source_region, destination_region):
    body = tirx.Evaluate(T.copy(source_region, destination_region))
    mod = kernel(body, buffers=[source], params=[destination], annotations={"layout_map": {source: make_ascend_l0c_layout(source)}})
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(mod)


def _equal(actual, expected):
    analyzer = tvm.arith.Analyzer()
    assert len(actual) == len(expected)
    assert all(analyzer.can_prove_equal(a, b) for a, b in zip(actual, expected)), (actual, expected)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("source_m, valid_m, valid_n", [(32, 3, 35), (35, 3, 19), (32, 17, 19)])
def test_l0c_valid_tail_preserves_allocation_pitch(arch, source_m, valid_m, valid_n):
    source = tirx.decl_buffer((source_m, 48), "float32", name="acc", scope="shared.l0c")
    destination = tirx.decl_buffer((source_m + 4, 37), "float32", name="out")
    after = _lower(arch, source, destination, _region(source, [0, 0], [valid_m, valid_n]), _region(destination, [2, 1], [valid_m, valid_n]))
    (copy,) = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    _equal(copy.args[3:7], [valid_n, valid_m, 37, ((source_m + 15) // 16) * 16])
    # Access effects preserve the logical valid tile; neither the read extent
    # nor the destination origin becomes a padded-allocation access.
    _equal(copy.args[1].args[2:4], [0, valid_m * valid_n])
    _equal(copy.args[0].args[2:4], [75, valid_m * valid_n])


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("source_m", [32, 35])
def test_l0c_leading_view_keeps_trailing_pitch(arch, source_m):
    source = tirx.decl_buffer((2, source_m, 48), "float32", name="acc", scope="shared.l0c")
    destination = tirx.decl_buffer((4, 8, 37), "float32", name="out")
    after = _lower(arch, source, destination, _region(source, [1, 0, 0], [1, 3, 35]), _region(destination, [2, 2, 1], [1, 3, 35]))
    (copy,) = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    physical_m = ((source_m + 15) // 16) * 16
    _equal(copy.args[3:7], [35, 3, 37, physical_m])
    _equal(copy.args[1].args[2:4], [physical_m * 48, 3 * 35])
    _equal(copy.args[0].args[2:4], [2 * 8 * 37 + 75, 3 * 35])


def test_l0c_partial_nonzero_matrix_origin_has_explicit_diagnostic():
    source = tirx.decl_buffer((32, 48), "float32", name="acc", scope="shared.l0c")
    destination = tirx.decl_buffer((3, 35), "float32", name="out")
    with pytest.raises(tvm.error.InternalError, match="zero-origin trailing matrix dimensions"):
        _lower("dav-2201", source, destination, _region(source, [16, 0], [3, 35]), _region(destination, [0, 0], [3, 35]))


def _column_export(*, origin=64, rows=128, cols=256, m=None, n=64,
                   row_origin=0, owner_extent=None, leading=False,
                   leading_extent=1, layout="canonical", strides=False,
                   arch="dav-2201", dtype="float32"):
    owner = tirx.Var("owner", "int32")
    slot = tirx.Var("slot", "int32")
    m = rows if m is None else m
    if owner_extent is not None:
        origin = owner * n
    source = tirx.decl_buffer((2, rows, cols) if leading else (rows, cols),
                             "float32", name="acc", scope="shared.l0c",
                             strides=([rows * cols, cols * 2, 1] if leading else [cols * 2, 1]) if strides else None)
    source_view = tirx.decl_buffer(source.shape, source.dtype, name="acc_view",
                                  data=source.data, scope=source.scope())
    operand_source = source_view if layout == "operand-view" else source
    # Every owner and leading plane has an independent GM result, with a
    # nonzero matrix origin and physical row pitch different from native N.
    destination = tirx.decl_buffer((2, 2, rows + 2, n + 9) if leading else (2, rows + 2, n + 9),
                                  dtype, name="out")
    src = _region(operand_source, [slot, row_origin, origin] if leading else [row_origin, origin],
                  [leading_extent, m, n] if leading else [m, n])
    owner_index = owner if owner_extent is not None else 0
    dst = _region(destination, [slot, owner_index, 1, 3] if leading else [owner_index, 1, 3],
                  [leading_extent, 1, m, n] if leading else [1, m, n])
    body = tirx.Evaluate(T.copy(src, dst))
    if owner_extent is not None:
        body = tirx.For(owner, 0, owner_extent, tirx.ForKind.SERIAL, body)
    if leading:
        body = tirx.For(slot, 0, 2, tirx.ForKind.SERIAL, body)
    buffers, layouts = [source], {}
    if layout == "same-name-other-data":
        other = tirx.decl_buffer(source.shape, source.dtype, name=source.name, scope=source.scope())
        assert other.name == source.name and not other.data.same_as(source.data)
        buffers.append(other)
        layouts[other] = make_ascend_l0c_layout(other)
    elif layout in ("same-data-layout-key", "operand-view"):
        assert not source_view.same_as(source) and source_view.data.same_as(source.data)
        layouts[source_view] = make_ascend_l0c_layout(source_view)
    elif layout != "missing":
        layouts[source] = make_ascend_l0c_layout(source) if layout == "canonical" else make_ascend_major_k_layout(source)
    mod = kernel(body, buffers=buffers, params=[destination], annotations={"layout_map": layouts})
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        return transform.AscendLowerTileOp()(mod)


def _eval_column_offset(expression, **values):
    substitutions = {var: tirx.IntImm(var.dtype, values[var.name]) for var in tirx.analysis.undefined_vars(expression)}
    result = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expression, substitutions))
    assert isinstance(result, tirx.IntImm), result
    return int(result)


@pytest.mark.parametrize("origin", [16, 128, 192])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_2201_fix_column_export_maps_nz_base_and_preserves_physical_pitch(origin, dtype):
    (copy,) = calls(_column_export(origin=origin, dtype=dtype), "tl.ascend_copy_matrix_cc_to_gm")
    _equal(copy.args[3:7], [64, 128, 73, 128])
    _equal(copy.args[1].args[2:4], [origin * 128, 128 * 64])
    _equal(copy.args[0].args[2:4], [73 + 3, 128 * 64])
    assert int(copy.args[10]) == (1 if dtype == "float16" else 0)
    assert int(copy.args[13]) == 1  # NZ -> ND remains enabled.


def test_2201_fix_column_export_preserves_owner_and_leading_version_planes():
    (copy,) = calls(_column_export(rows=32, cols=128, n=32, owner_extent=2, leading=True),
                    "tl.ascend_copy_matrix_cc_to_gm")
    _equal(copy.args[3:7], [32, 32, 41, 32])
    assert int(copy.args[1].args[3]) == 32 * 32
    touched = set()
    for slot in range(2):
        for owner in range(2):
            source_offset = _eval_column_offset(copy.args[1].args[2], slot=slot, owner=owner)
            assert source_offset == slot * 32 * 128 + owner * 32 * 32
            region = set(range(source_offset, source_offset + 32 * 32))
            assert not touched.intersection(region)
            touched.update(region)
            destination_offset = _eval_column_offset(copy.args[0].args[2], slot=slot, owner=owner)
            assert destination_offset == (slot * 2 + owner) * 34 * 41 + 41 + 3


@pytest.mark.parametrize("kwargs,diagnostic", [
    ({"origin": 1}, "16-fractal"),
    ({"n": 17}, "16-fractal"),
    ({"rows": 35}, "16-fractal"),
    ({"row_origin": 16, "m": 96}, "full M"),
    ({"m": 64}, "full M"),
    ({"strides": True}, "compact"),
    ({"layout": "wrong-kind"}, "canonical"),
    ({"layout": "missing"}, "canonical"),
    ({"layout": "same-name-other-data"}, "canonical"),
    ({"layout": "same-data-layout-key"}, "canonical"),
    ({"layout": "operand-view"}, "allocation"),
])
def test_fix_column_export_requires_shared_c_region_encoding_contract(kwargs, diagnostic):
    with pytest.raises((tvm.error.InternalError, ValueError), match="L0C column region.*" + diagnostic):
        _column_export(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"origin": -64}, {"origin": 256}, {"owner_extent": 5},
    {"origin": tirx.Var("unbounded_column", "int32") * 64},
    {"leading": True, "leading_extent": 2},
])
def test_fix_column_export_requires_loop_aware_backing_bounds(kwargs):
    with pytest.raises((tvm.error.InternalError, ValueError), match="L0C column region.*(bounds|singleton)"):
        _column_export(**kwargs)


def test_3510_fix_column_region_remains_explicitly_unsupported():
    with pytest.raises(tvm.error.InternalError, match="zero-origin trailing matrix dimensions"):
        _column_export(arch="dav-3510")
