"""DMA tail proofs and clamping must not overflow the original index dtype."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq


def _region(buffer, origins, extents):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(origin, extent) for origin, extent in zip(origins, extents)])


def test_clamping_does_not_narrow_an_int64_backing_shape_before_subtraction():
    source = tirx.decl_buffer((tirx.IntImm("int64", 2147483648), 16), "float16", name="source")
    destination = tirx.decl_buffer((32, 16), "float16", name="destination", scope="shared.dyn")
    before = kernel(copy(source, destination, src_indices=[0, 15], src_shape=[32, 16]), buffers=[destination], params=[source])
    (operation,) = calls(transform.AscendInsertOOBPadding()(before), "tl.tileop.ascend_copy")
    # The column tail forces clamping. Its unrelated row axis remains 32,
    # although the backing shape cannot be represented by an int32 value.
    assert [int(value) for value in operation.args[0].args[2:]] == [32, 1]
    assert [int(value) for value in operation.args[1].args[2:]] == [32, 1]


def _row_copy(origin, extent=16, *, assumptions=(), scope="shared.dyn"):
    source = tirx.decl_buffer((32, 16), "float16", name="source")
    destination = tirx.decl_buffer((16, 16), "float16", name="destination", scope=scope)
    body = seq(
        *(tirx.Evaluate(tirx.assume(condition)) for condition in assumptions),
        copy(source, destination, src_indices=[origin, 0], src_shape=[extent, 16], dst_shape=[extent, 16]),
    )
    params = [source, *tirx.analysis.undefined_vars(body, [source.data, destination.data])]
    return kernel(body, buffers=[destination], params=params)


@pytest.mark.parametrize("origin", [-40, tirx.Var("unknown_origin", "int32")], ids=["remaining-negative", "unknown"])
def test_active_matrix_copy_requires_nonnegative_origin_after_legalization(origin):
    before = tilelang.transform.LegalizeNegativeIndex()(_row_copy(origin))
    with pytest.raises(tvm.error.InternalError, match="proven nonnegative matrix origin.*T.assume"):
        transform.AscendInsertOOBPadding()(before)


def test_assumed_nonnegative_matrix_origin_preserves_legal_copy():
    origin = tirx.Var("origin", "int32")
    before = _row_copy(origin, assumptions=[origin >= 0, origin <= 16])
    before = tilelang.transform.InjectAssumes()(before)
    before = tilelang.transform.LegalizeNegativeIndex()(before)
    after = transform.AscendInsertOOBPadding()(before)
    (operation,) = calls(after, "tl.tileop.ascend_copy")
    assert int(operation.args[0].args[2]) == 16
    assert int(operation.args[1].args[2]) == 16


def test_known_empty_copy_does_not_require_an_origin_bound():
    before = _row_copy(tirx.Var("irrelevant_origin", "int64"), extent=0)
    tvm.ir.assert_structural_equal(transform.AscendInsertOOBPadding()(before), before)


def test_dynamic_empty_copy_checks_origin_only_on_its_active_path():
    rows = tirx.Var("rows", "int32")
    # The inactive path deliberately carries an invalid origin. Its data task
    # remains present so later completion planning can still see the copy.
    origin = tirx.Select(rows > 0, 0, -1)
    before = _row_copy(origin, extent=rows, assumptions=[rows >= 0, rows <= 16])
    before = tilelang.transform.InjectAssumes()(before)
    after = transform.AscendInsertOOBPadding()(before)
    (operation,) = calls(after, "tl.tileop.ascend_copy")
    analyzer = tvm.arith.Analyzer()
    for row_count in [0, 5, 16]:
        extents = [
            int(analyzer.simplify(tirx.stmt_functor.substitute(operand.args[2], {rows: row_count})))
            for operand in operation.args[:2]
        ]
        assert extents == [row_count, row_count]


def test_full_pipeline_rejects_fully_outside_gm_source_before_l1_native_read():
    @T.prim_func
    def main(A: T.Tensor((32, 16), "float16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 16), "float16")
            l0 = T.alloc_l0a((16, 16), "float16")
            T.copy(A[32:48, :], l1)
            T.copy(l1, l0)

    with pytest.raises(tvm.error.InternalError, match="GM->L1 tail copy requires.*in-bounds source matrix origin"):
        tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}, target_host="c", enable_host_codegen=False, enable_device_compile=False)


@pytest.mark.parametrize("max_tile", [1, 2], ids=["in-bounds", "may-be-fully-outside"])
def test_nonnegative_gm_to_l1_origin_still_requires_an_in_bounds_range(max_tile):
    index = tirx.Var("tile_index", "int32")
    before = _row_copy(index * 16, assumptions=[index >= 0, index <= max_tile], scope="shared.l1")
    before = tilelang.transform.InjectAssumes()(before)
    if max_tile == 2:
        with pytest.raises(tvm.error.InternalError, match="GM->L1 tail copy requires.*in-bounds source matrix origin"):
            transform.AscendInsertOOBPadding()(before)
    else:
        tvm.ir.assert_structural_equal(transform.AscendInsertOOBPadding()(before), before)


@pytest.mark.parametrize("active", [False, True], ids=["known-empty", "dynamic-empty"])
def test_empty_gm_to_l1_copy_checks_matrix_bounds_only_when_active(active):
    source = tirx.decl_buffer((32, 16), "float16", name="source")
    destination = tirx.decl_buffer((16, 16), "float16", name="destination", scope="shared.l1")
    rows = tirx.Var("rows", "int32")
    extent = rows if active else 0
    origin = tirx.Select(rows > 0, 16, 32) if active else 32
    body = seq(
        tirx.Evaluate(tirx.assume(rows >= 0)),
        tirx.Evaluate(tirx.assume(rows <= 16)),
        copy(source, destination, src_indices=[origin, 0], src_shape=[extent, 16], dst_shape=[extent, 16]),
    )
    before = tilelang.transform.InjectAssumes()(kernel(body, buffers=[destination], params=[source, rows]))
    after = transform.AscendInsertOOBPadding()(before)
    (operation,) = calls(after, "tl.tileop.ascend_copy")
    for row_count in [0, 5, 16]:
        valid_rows = int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(operation.args[0].args[2], {rows: row_count})))
        assert valid_rows == (row_count if active else 0)


@pytest.mark.parametrize("direction", ["gm-to-ub", "ub-to-gm"])
@pytest.mark.parametrize("axis,index_dtype,origin", [
    (0, "int32", 2147483632),
    (1, "int32", 2147483640),
    (0, "int64", 9223372036854775791),
], ids=["int32-row-end", "int32-column-end", "int64-row-end"])
def test_full_pipeline_does_not_issue_dma_after_end_overflow(direction, axis, index_dtype, origin):
    origins = [0, 0]
    origins[axis] = tirx.IntImm(index_dtype, origin)

    @T.prim_func
    def main(A: T.Tensor((32, 16), "float16"), B: T.Tensor((32, 16), "float16")):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 16), "float16")
            if direction == "gm-to-ub":
                T.copy(_region(A, origins, [32, 16]), ub)
                T.copy(ub, B)
            else:
                T.copy(A, ub)
                T.copy(ub, _region(B, origins, [32, 16]))

    snapshots = {}
    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendInsertOOBPadding":
                snapshots[info.name] = mod
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}, target_host="c", enable_host_codegen=False, enable_device_compile=False)
    operations = calls(snapshots["tl.AscendInsertOOBPadding"], "tl.tileop.ascend_copy")
    operation = operations[0 if direction == "gm-to-ub" else -1]
    assert int(operation.args[0].args[2 + axis]) == 0
    assert int(operation.args[1].args[2 + axis]) == 0
    skipped = "asc_copy_gm2ub_align(" if direction == "gm-to-ub" else "asc_copy_ub2gm_align("
    retained = "asc_copy_ub2gm_align(" if direction == "gm-to-ub" else "asc_copy_gm2ub_align("
    assert skipped not in artifact.kernel_source
    assert retained in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
