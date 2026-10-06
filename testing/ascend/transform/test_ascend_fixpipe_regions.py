"""Fixpipe valid regions never change their canonical L0C storage pitch."""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout
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
