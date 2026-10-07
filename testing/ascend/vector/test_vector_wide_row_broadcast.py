"""Wide private row broadcasts retain exact chunk regions before scheduling."""

import re

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.vector.test_vector_row_broadcast import (
    TARGET, _lower, _native, _program, _ptrs, _regions, _source,
)


def _equal(actual, expected):
    assert len(actual) == len(expected)
    analyzer = tvm.arith.Analyzer()
    assert all(analyzer.can_prove_equal(a, b) for a, b in zip(actual, expected)), (actual, expected)


@pytest.mark.parametrize("columns", [128, 192])
@pytest.mark.parametrize("operation", ["sub", "rsub", "mul", "rmul", "div", "rdiv"])
def test_wide_rows_have_exact_disjoint_chunk_effects(columns, operation):
    rows = 16
    after = _lower(_program(rows=rows, columns=columns, operation=operation))
    (brcb,) = calls(after, "tl.ascend_membase.brcb_f32")
    binaries = calls(after, "tl.ascend_membase." + operation.removeprefix("r") + "_rows_f32")
    assert len(binaries) == columns // 64
    row_lhs = operation.startswith("r")
    for chunk, binary in enumerate(binaries):
        assert int(binary.args[3]) == rows
        assert bool(int(binary.args[4])) == row_lhs
        assert int(binary.args[5]) == columns // 8
        assert not _ptrs(binary)
        regions = _regions(binary)
        assert len(regions) == 3
        assert [int(region.args[1]) for region in regions] == [2, 1, 1]
        for index, region in enumerate(regions):
            assert region.args[0].buffer.dtype == "float32"
            matrix = index == 0 or (index == 1 and not row_lhs) or (index == 2 and row_lhs)
            if matrix:
                _equal(region.args[0].indices, [0, chunk * 64])
                _equal(region.args[2:], [rows, 64])
                _equal(region.args[0].buffer.shape, [rows, columns])
                # The final chunk ends at C, rather than describing a dense
                # R*C span from a shifted column pointer past the allocation.
                assert int(region.args[0].indices[-1]) + int(region.args[-1]) <= columns
            else:
                _equal(region.args[0].indices, [0])
                _equal(region.args[2:], [rows * 8])
                assert region.args[0].buffer.data.same_as(brcb.args[0].args[0].buffer.data)
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)
    units = transform.MaterializeScheduleUnits()(after)
    assert len(calls(units, binaries[0].op.name)) == columns // 64


def test_largest_multiple_of_64_with_a_uint8_row_stride_is_selected():
    after = _lower(_program(rows=8, columns=1984, operation="mul"))
    binaries = calls(after, "tl.ascend_membase.mul_rows_f32")
    assert len(binaries) == 31
    assert all(int(binary.args[5]) == 248 for binary in binaries)
    _equal(_regions(binaries[-1])[0].args[0].indices, [0, 1920])
    _equal(_regions(binaries[-1])[0].args[2:], [8, 64])


@pytest.mark.parametrize("columns", [96, 2048])
def test_nonmultiple_or_unrepresentable_stride_keeps_existing_row_lowering(columns):
    after = _lower(_program(columns=columns))
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert nodes(after, tirx.Bind)


@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("flat", [False, True])
def test_dense_linear_index_is_not_misrepresented_as_a_row_rectangle(columns, flat):
    rows = 16
    shape = (rows * columns,) if flat else (1, rows * columns)

    @T.prim_func
    def main(A: T.Tensor(shape, "float32"), F: T.Tensor((rows,), "float32"), C: T.Tensor((rows, columns), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared(shape, "float32")
            row = T.alloc_shared((rows,), "float32")
            result = T.alloc_shared((rows, columns), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(rows, columns):
                if flat:
                    result[i, j] = matrix[i * columns + j] - row[i]
                else:
                    result[i, j] = matrix[0, i * columns + j] - row[i]
            T.copy(result, C)

    after = _lower(main)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert calls(after, "tl.ascend_membase.sub_scalar")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_wide_fa_store_boundaries_keep_whole_tile_exp_and_terminal_cast(dtype):
    @T.prim_func
    def main(A: T.Tensor((16, 128), "float32"), F: T.Tensor((16,), "float32"), P: T.Tensor((16, 128), dtype)):
        with T.Kernel(1):
            scores = T.alloc_shared((16, 128), "float32")
            row = T.alloc_shared((16,), "float32")
            p = T.alloc_shared((16, 128), dtype)
            T.copy(A, scores)
            T.copy(F, row)
            for i, j in T.Parallel(16, 128):
                scores[i, j] = T.exp(scores[i, j] - row[i])
                p[i, j] = T.cast(scores[i, j], dtype)
            T.copy(p, P)

    after = _lower(main)
    cast = "float2half_rn" if dtype == "float16" else "float2bfloat16_rn"
    assert [op.op.name for op in _native(after)] == [
        "tl.ascend_membase.brcb_f32", "tl.ascend_membase.sub_rows_f32",
        "tl.ascend_membase.sub_rows_f32", "tl.ascend_membase.exp",
        "tl.ascend_membase." + cast,
    ]
    exp = calls(after, "tl.ascend_membase.exp")[0]
    narrow = calls(after, "tl.ascend_membase." + cast)[0]
    assert int(exp.args[-1]) == int(narrow.args[-1]) == 2048
    _equal(exp.args[1].args[0].buffer.shape, [16, 128])
    assert exp.args[0].args[0].buffer.data.same_as(narrow.args[1].args[0].buffer.data)
    source = _source(main)
    assert source.count("asc_brcb(") == source.count("asc_exp(") == 1
    assert source.count("asc_sub(") == 2
    assert "membase_scalar_" not in source
    lines = [line for line in source.splitlines() if "asc_sub(" in line]
    assert all("uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(16), uint8_t(16), uint8_t(1)" in line for line in lines)
    # Each normal-mode chunk establishes a full mask. The common V completion
    # protects Brcb -> both chunks -> whole-tile Exp -> narrowing.
    for match in re.finditer(r"asc_sub\(", source):
        before = source[:match.start()]
        assert before.rfind("asc_set_vector_mask(~uint64_t(0), ~uint64_t(0));") > before.rfind("asc_brcb(")
    brcb_end = source.index(";", source.index("asc_brcb("))
    assert "asc_sync_pipe(PIPE_V)" in source[brcb_end:source.index("asc_sub(")]
    sub_end = source.index(";", source.rindex("asc_sub("))
    assert "asc_sync_pipe(PIPE_V)" in source[sub_end:source.index("asc_exp(")]
    exp_end = source.index(";", source.index("asc_exp("))
    assert "asc_sync_pipe(PIPE_V)" in source[exp_end:source.index("asc_" + cast + "(")]


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_chunk_regions_keep_independent_versions_until_late_pointer_lowering(index_dtype):
    after = _lower(_program(columns=128, operation="mul"))
    binaries = calls(after, "tl.ascend_membase.mul_rows_f32")
    first = _regions(binaries[0])
    dst, src, blocks = [region.args[0].buffer for region in first]
    epoch = tirx.Var("epoch", index_dtype)
    loop = tirx.For(epoch, tirx.IntImm(index_dtype, 0), tirx.IntImm(index_dtype, 7), tirx.ForKind.SERIAL,
        seq(*(unit(tirx.Evaluate(binary), core=1) for binary in binaries)),
        annotations={"multi_buffer_eligible": [src.data, dst.data]})
    mod = kernel(unit(loop, core=None), buffers=[src, dst, blocks], annotations={
        "tl.buffer_versions_map": {src.data: 2, dst.data: 3},
    })
    target = tvm.target.Target(TARGET)
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        versioned = transform.MaterializeMultiBuffer()(mod)
        actual_epoch = next(loop.loop_var for loop in nodes(versioned, tirx.For) if loop.loop_var.name == "epoch")
        selected = calls(versioned, "tl.ascend_membase.mul_rows_f32")
        assert len(selected) == 2
        for chunk, binary in enumerate(selected):
            d, s, row = _regions(binary)
            _equal(d.args[0].buffer.shape, [3, 16, 128])
            _equal(s.args[0].buffer.shape, [2, 16, 128])
            _equal(d.args[0].indices, [tirx.Cast("int32", actual_epoch % 3), 0, chunk * 64])
            _equal(s.args[0].indices, [tirx.Cast("int32", actual_epoch % 2), 0, chunk * 64])
            _equal(d.args[2:], [1, 16, 64])
            _equal(s.args[2:], [1, 16, 64])
            _equal(row.args[2:], [128])
        lowered = transform.AscendLowerTileOp()(versioned)
        lowered = tilelang.transform.LowerAccessPtr()(lowered)
    actual_epoch = next(loop.loop_var for loop in nodes(lowered, tirx.For) if loop.loop_var.name == "epoch")
    for chunk, binary in enumerate(calls(lowered, "tl.ascend_membase.mul_rows_f32")):
        assert not _regions(binary)
        d, s, row = binary.args[:3]
        assert all(pointer.op.name == "tirx.tvm_access_ptr" for pointer in [d, s, row])
        _equal([d.args[2], s.args[2]], [
            tirx.Cast("int64", tirx.Cast("int32", actual_epoch % 3)) * tirx.IntImm("int64", 2048) + chunk * 64,
            tirx.Cast("int64", tirx.Cast("int32", actual_epoch % 2)) * tirx.IntImm("int64", 2048) + chunk * 64,
        ])
        # This late span is the bounded pointer hull of the already consumed
        # exact rectangle, not a new scheduled contiguous effect.
        _equal([d.args[3], s.args[3], row.args[3]], [15 * 128 + 64, 15 * 128 + 64, 128])


@pytest.mark.parametrize("bad", ["stride", "shape", "mask", "leading", "pitch"])
def test_late_boundary_rejects_inconsistent_native_row_contract(bad):
    after = _lower(_program(columns=128))
    binary = calls(after, "tl.ascend_membase.sub_rows_f32")[0]
    args = list(binary.args)
    regions = _regions(binary)
    dst, src, row = [region.args[0].buffer for region in regions]
    if bad == "stride":
        args[5] = tirx.IntImm("uint8", 8)
    else:
        operand = list(args[0].args)
        if bad == "shape":
            operand[-1] = tirx.IntImm("int32", 65)
        elif bad == "mask":
            operand[1] = tirx.IntImm("int32", 1)
        elif bad == "pitch":
            dst = tirx.decl_buffer((16, 128), "float32", name="bad_pitch", scope="shared.dyn", strides=[136, 1])
            operand[0] = dst[0, 0]
        else:
            extra = tirx.decl_buffer((2, 16, 128), "float32", name="bad_leading", scope="shared.dyn")
            operand = [extra[0, 0, 0], tirx.IntImm("int32", 2), tirx.IntImm("int32", 2), tirx.IntImm("int32", 16), tirx.IntImm("int32", 64)]
            dst = extra
        args[0] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), operand)
    call = tirx.Call("void", binary.op, args)
    mod = kernel(tirx.Evaluate(call), buffers=[dst, src, row])
    target = tvm.target.Target(TARGET)
    mod["main"] = mod["main"].with_attr("target", target)
    with target, pytest.raises(tvm.error.InternalError):
        transform.AscendLowerTileOp()(mod)


def test_real_2d_row_view_with_a_wider_physical_pitch_uses_existing_row_lowering():
    matrix = tirx.decl_buffer((16, 128), "float32", name="matrix", scope="shared.dyn", strides=[136, 1])
    row = tirx.decl_buffer((16,), "float32", name="row", scope="shared.dyn")
    result = tirx.decl_buffer((16, 128), "float32", name="result", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = tirx.For(i, 0, 16, tirx.ForKind.PARALLEL,
        tirx.For(j, 0, 128, tirx.ForKind.PARALLEL,
            tirx.BufferStore(result, matrix[i, j] - row[i], [i, j])))
    before = kernel(body, buffers=[matrix, row, result])
    before["main"] = before["main"].with_attr("target", tvm.target.Target(TARGET))
    after = transform.AscendLowerVectorTiles()(before)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert calls(after, "tl.ascend_membase.sub_scalar")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))


def test_aligned_nonzero_row_origin_is_retained_in_each_chunk_region():
    @T.prim_func
    def main(A: T.Tensor((17, 128), "float32"), F: T.Tensor((16,), "float32"), C: T.Tensor((16, 128), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared((17, 128), "float32")
            row = T.alloc_shared((16,), "float32")
            result = T.alloc_shared((16, 128), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(16, 128):
                result[i, j] = matrix[i + 1, j] - row[i]
            T.copy(result, C)

    after = _lower(main)
    binaries = calls(after, "tl.ascend_membase.sub_rows_f32")
    assert len(binaries) == 2
    for chunk, binary in enumerate(binaries):
        dst, src, row = _regions(binary)
        _equal(dst.args[0].indices, [0, chunk * 64])
        _equal(src.args[0].indices, [1, chunk * 64])
        _equal(src.args[0].buffer.shape, [17, 128])
        _equal(src.args[2:], [16, 64])


@pytest.mark.parametrize("columns", [64, 128])
def test_3510_does_not_enter_the_membase_wide_row_contract(columns):
    from testing.ascend.vector.test_vector_row_broadcast import _frontend
    before = _frontend(_program(columns=columns))
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-3510"}))
    after = transform.AscendLowerVectorTiles()(before)
    tvm.ir.assert_structural_equal(before, after)
    assert not _native(after)
