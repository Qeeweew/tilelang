"""Ordinary DSL row broadcasts become explicit Brcb and one native row binary.

Host/compiler regressions only. These tests neither build kernels with CANN nor
launch a device; CANN and device acceptance are separate integration gates.
"""

import re

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform

from testing.ascend._ir import calls, nodes


TARGET = {"kind": "ascend", "arch": "dav-2201"}


def _program(rows=16, columns=64, operation="sub", dtype="float32", output_dtype=None, exp=False, row_offset=0):
    output_dtype = output_dtype or dtype
    operators = {
        "sub": lambda matrix, row: matrix - row,
        "rsub": lambda matrix, row: row - matrix,
        "mul": lambda matrix, row: matrix * row,
        "rmul": lambda matrix, row: row * matrix,
        "div": lambda matrix, row: matrix / row,
        "rdiv": lambda matrix, row: row / matrix,
    }
    binary = operators[operation]

    def expression(matrix, row):
        value = binary(matrix, row)
        if exp:
            value = T.exp(value)
        return T.cast(value, output_dtype)

    @T.prim_func
    def main(
        A: T.Tensor((rows, columns), dtype),
        F: T.Tensor((rows + row_offset,), dtype),
        C: T.Tensor((rows, columns), output_dtype),
    ):
        with T.Kernel(1):
            matrix = T.alloc_shared((rows, columns), dtype)
            row = T.alloc_shared((rows + row_offset,), dtype)
            result = T.alloc_shared((rows, columns), output_dtype)
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(rows, columns):
                result[i, j] = expression(matrix[i, j], row[i + row_offset])
            T.copy(result, C)

    return main


def _frontend(program):
    mod = tirx.transform.BindTarget(tvm.target.Target(TARGET))(tvm.IRModule.from_expr(program))
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    return tirx.transform.Simplify()(mod)


def _lower(program):
    return transform.AscendLowerVectorTiles()(_frontend(program))


def _native(mod):
    return [call for call in nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _ptrs(call):
    return [arg for arg in call.args if isinstance(arg, tirx.Call) and arg.op.name == "tl.access_ptr"]


def _regions(call):
    return [arg for arg in call.args if isinstance(arg, tirx.Call) and arg.op.name == "tl.region"]


def _source(program):
    return tilelang.lower(program, target=TARGET, enable_host_codegen=False, enable_device_compile=False).kernel_source


@pytest.mark.parametrize("rows", [8, 16, 248])
@pytest.mark.parametrize("operation", ["sub", "rsub", "mul", "rmul", "div", "rdiv"])
def test_static_full_vl_broadcast_exposes_exact_f32_regions(rows, operation):
    after = _lower(_program(rows=rows, operation=operation))
    (brcb,) = calls(after, "tl.ascend_membase.brcb_f32")
    (binary,) = calls(after, "tl.ascend_membase." + operation.removeprefix("r") + "_rows_f32")
    assert int(brcb.args[2]) == rows // 8
    assert int(binary.args[3]) == rows and int(binary.args[4]) == int(operation.startswith("r"))
    brcb_ptrs = _ptrs(brcb)
    assert [int(ptr.args[1]) for ptr in brcb_ptrs] == [rows * 8, rows]
    assert [int(ptr.args[2]) for ptr in brcb_ptrs] == [2, 1]
    assert all(ptr.args[0].buffer.dtype == "float32" for ptr in brcb_ptrs)
    binary_regions = _regions(binary)
    expected = [[rows, 64], [rows * 8], [rows, 64]] if operation.startswith("r") else [[rows, 64], [rows, 64], [rows * 8]]
    assert [[int(extent) for extent in region.args[2:]] for region in binary_regions] == expected
    assert [int(region.args[1]) for region in binary_regions] == [2, 1, 1]
    assert all(region.args[0].buffer.dtype == "float32" for region in binary_regions)
    assert int(binary.args[5]) == 8
    assert not _ptrs(binary)
    scratch_data = brcb_ptrs[0].args[0].buffer.data
    assert any(region.args[0].buffer.data.same_as(scratch_data) for region in binary_regions[1:])
    scratch = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if buffer.data.same_as(scratch_data)]
    assert len(scratch) == 1 and scratch[0].scope() == "shared.dyn"
    assert int(scratch[0].shape[0]) == rows * 8
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.For)
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len(calls(materialized, "tl.ascend_membase.brcb_f32")) == 1
    assert len(calls(materialized, binary.op.name)) == 1


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_ordinary_softmax_expression_has_one_sub_exp_and_terminal_cast(dtype):
    after = _lower(_program(output_dtype=dtype, exp=True))
    ops = _native(after)
    names = [op.op.name for op in ops]
    assert names[:3] == ["tl.ascend_membase.brcb_f32", "tl.ascend_membase.sub_rows_f32", "tl.ascend_membase.exp"]
    exp = calls(after, "tl.ascend_membase.exp")[0]
    assert int(exp.args[-1]) == 16 * 64
    assert [int(ptr.args[1]) for ptr in _ptrs(exp)] == [16 * 64, 16 * 64]
    if dtype != "float32":
        name = "float2half_rn" if dtype == "float16" else "float2bfloat16_rn"
        narrow = calls(after, "tl.ascend_membase." + name)[0]
        assert int(narrow.args[-1]) == 16 * 64
        assert [int(ptr.args[1]) for ptr in _ptrs(narrow)] == [16 * 64, 16 * 64]
        assert _ptrs(narrow)[0].args[0].buffer.dtype == dtype
        assert _ptrs(narrow)[1].args[0].buffer.dtype == "float32"
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.For)


@pytest.mark.parametrize("rows,columns,dtype,row_offset", [(7, 64, "float32", 0), (9, 64, "float32", 0),
    (256, 64, "float32", 0), (16, 32, "float32", 0), (16, 2048, "float32", 0),
    (16, 64, "float16", 0), (16, 64, "bfloat16", 0), (16, 64, "float32", 1)])
@pytest.mark.parametrize("operation", ["sub", "div", "rdiv"])
def test_unsupported_geometry_keeps_the_existing_row_route(rows, columns, dtype, row_offset, operation):
    after = _lower(_program(rows=rows, columns=columns, dtype=dtype, row_offset=row_offset, operation=operation))
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert not calls(after, "tl.ascend_membase.sub_rows_f32")
    assert not calls(after, "tl.ascend_membase.div_rows_f32")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert nodes(after, tirx.Bind)


@pytest.mark.parametrize("columns", [64, 128])
def test_private_mixed_owner_serial_row_shell_is_selected_without_pointer_offset_hack(columns):
    before = _frontend(_program(columns=columns))

    def private_owner(node):
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL and isinstance(node.body, tirx.For):
            return tirx.For(node.loop_var, node.min, node.extent, tirx.ForKind.SERIAL, node.body, annotations=node.annotations)
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, node.body, init=node.init,
                alloc_buffers=node.alloc_buffers, match_buffers=node.match_buffers,
                annotations={**dict(node.annotations), "tl.ascend_row_partitioned": 2})

    before = tvm.IRModule({gvar: func.with_body(ir_transform(func.body, None, private_owner)) for gvar, func in before.functions.items()})
    after = transform.AscendLowerVectorTiles()(before)
    assert len(calls(after, "tl.ascend_membase.brcb_f32")) == 1
    assert len(calls(after, "tl.ascend_membase.sub_rows_f32")) == columns // 64
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)


@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("operation", ["sub", "div", "rdiv"])
def test_bounded_dynamic_rows_keep_empty_owner_loop_semantics(columns, operation):
    binary = {"sub": lambda matrix, row: matrix - row,
              "div": lambda matrix, row: matrix / row,
              "rdiv": lambda matrix, row: row / matrix}[operation]

    @T.prim_func
    def main(A: T.Tensor((16, columns), "float32"), F: T.Tensor((16,), "float32"), C: T.Tensor((16, columns), "float32"), active: T.int32):
        with T.Kernel(1):
            matrix = T.alloc_shared((16, columns), "float32")
            row = T.alloc_shared((16,), "float32")
            result = T.alloc_shared((16, columns), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(T.max(T.min(active, 16), 0), columns):
                result[i, j] = binary(matrix[i, j], row[i])
            T.copy(result, C)

    after = _lower(main)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    loops = [loop for loop in nodes(after, tirx.For) if loop.kind == tirx.ForKind.SERIAL]
    assert len(loops) == 1
    active = next(param for param in after["main"].params if param.name == "active")
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(tirx.stmt_functor.substitute(loops[0].extent, {active: tirx.IntImm("int32", 0)}))) == 0


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_frontend_codegen_has_native_brcb_and_whole_tile_exp_cast(dtype):
    source = _source(_program(exp=True, output_dtype=dtype))
    assert source.count("asc_brcb(") == 1
    assert "reinterpret_cast<__ubuf__ uint32_t*>" in source
    assert "uint16_t(1), uint16_t(8), (uint8_t)2" in source
    sub_line = next(line for line in source.splitlines() if "asc_sub(" in line)
    assert "uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(8), uint8_t(8), uint8_t(1)" in sub_line
    assert "(uint8_t)16" in sub_line
    assert "membase_scalar_" not in source
    assert re.search(r"asc_exp\([^;]+\(uint32_t\)1024\)", source)
    cast = "asc_float2half_rn" if dtype == "float16" else "asc_float2bfloat16_rn"
    assert re.search(cast + r"\([^;]+\(uint32_t\)1024\)", source)
    # The shared native V dependency must survive the complete compiler:
    # Brcb -> Sub -> Exp -> narrowing. No per-row S snapshot is introduced.
    brcb_end = source.index(";", source.index("asc_brcb("))
    sub_begin = source.index("asc_sub(")
    assert "asc_sync_pipe(PIPE_V)" in source[brcb_end:sub_begin]


def test_mul_and_add_remain_separate_rounding_nodes():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32"), F: T.Tensor((16,), "float32"), B: T.Tensor((16, 64), "float32"), C: T.Tensor((16, 64), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared((16, 64), "float32")
            row = T.alloc_shared((16,), "float32")
            addend = T.alloc_shared((16, 64), "float32")
            result = T.alloc_shared((16, 64), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            T.copy(B, addend)
            for i, j in T.Parallel(16, 64):
                result[i, j] = matrix[i, j] * row[i] + addend[i, j]
            T.copy(result, C)

    after = _lower(main)
    assert [op.op.name for op in _native(after)] == ["tl.ascend_membase.brcb_f32", "tl.ascend_membase.mul_rows_f32", "tl.ascend_membase.add"]
    assert int(calls(after, "tl.ascend_membase.add")[0].args[-1]) == 1024
    source = _source(main)
    assert "asc_mul(" in source and "asc_add(" in source
    assert "vmadd" not in source and "membase_scalar_" not in source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ordinary_fa_two_assignments_preserve_whole_tile_node_order(dtype):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32"), F: T.Tensor((16,), "float32"), P: T.Tensor((16, 64), dtype)):
        with T.Kernel(1):
            scores = T.alloc_shared((16, 64), "float32")
            maximum = T.alloc_shared((16,), "float32")
            p = T.alloc_shared((16, 64), dtype)
            T.copy(A, scores)
            T.copy(F, maximum)
            for i, j in T.Parallel(16, 64):
                scores[i, j] = T.exp(scores[i, j] - maximum[i])
                p[i, j] = T.cast(scores[i, j], dtype)
            T.copy(p, P)

    after = _lower(main)
    cast = "float2half_rn" if dtype == "float16" else "float2bfloat16_rn"
    assert [op.op.name for op in _native(after)] == [
        "tl.ascend_membase.brcb_f32", "tl.ascend_membase.sub_rows_f32",
        "tl.ascend_membase.exp", "tl.ascend_membase." + cast,
    ]
    assert int(calls(after, "tl.ascend_membase.exp")[0].args[-1]) == 1024
    assert int(calls(after, "tl.ascend_membase." + cast)[0].args[-1]) == 1024
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)
    source = _source(main)
    assert source.count("asc_brcb(") == source.count("asc_sub(") == source.count("asc_exp(") == 1
    assert "membase_scalar_" not in source
    positions = [source.index(name) for name in ("asc_brcb(", "asc_sub(", "asc_exp(", "asc_" + cast + "(")]
    assert positions == sorted(positions)
    for before, after in zip(positions, positions[1:]):
        assert "asc_sync_pipe(PIPE_V)" in source[source.index(";", before):after]


@pytest.mark.parametrize("columns", [64, 128])
def test_shifted_matrix_rows_keep_the_existing_per_row_route(columns):
    @T.prim_func
    def main(A: T.Tensor((17, columns), "float32"), F: T.Tensor((16,), "float32"), C: T.Tensor((16, columns), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared((17, columns), "float32")
            row = T.alloc_shared((16,), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(16, columns):
                matrix[i + 1, j] = matrix[i, j] - row[i]
            T.copy(matrix[1:17, :], C)

    after = _lower(main)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert nodes(after, tirx.Bind)


def test_lane_predicates_keep_the_existing_masked_row_route():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32"), F: T.Tensor((16,), "float32"), C: T.Tensor((16, 64), "float32")):
        with T.Kernel(1):
            matrix = T.alloc_shared((16, 64), "float32")
            row = T.alloc_shared((16,), "float32")
            result = T.alloc_shared((16, 64), "float32")
            T.copy(A, matrix)
            T.copy(F, row)
            for i, j in T.Parallel(16, 64):
                result[i, j] = T.if_then_else(j < 32, matrix[i, j] - row[i], matrix[i, j])
            T.copy(result, C)

    after = _lower(main)
    assert not calls(after, "tl.ascend_membase.brcb_f32")
    assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))
    assert calls(after, "tl.ascend_membase.select_bits")


@pytest.mark.parametrize("operation", ["div", "rdiv"])
@pytest.mark.parametrize("output_dtype", ["float32", "float16", "bfloat16"])
def test_row_division_codegen_uses_native_normal_repeat_and_exact_cast(operation, output_dtype):
    program = _program(operation=operation, output_dtype=output_dtype)
    after = _lower(program)
    (division,) = calls(after, "tl.ascend_membase.div_rows_f32")
    assert int(division.args[3]) == 16
    assert bool(int(division.args[4])) == (operation == "rdiv")
    expected = [[16, 64], [128], [16, 64]] if operation == "rdiv" else [[16, 64], [16, 64], [128]]
    regions = _regions(division)
    assert [[int(extent) for extent in region.args[2:]] for region in regions] == expected
    assert [int(region.args[1]) for region in regions] == [2, 1, 1]
    assert all(region.args[0].buffer.dtype == "float32" for region in regions)
    assert int(division.args[5]) == 8 and not _ptrs(division)
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.For)
    source = _source(program)
    assert source.count("asc_brcb(") == source.count("asc_div(") == 1
    division_start = source.index("asc_div(")
    line = next(line for line in source.splitlines() if "asc_div(" in line)
    strides = "uint8_t(1), uint8_t(0), uint8_t(1), uint8_t(8), uint8_t(1), uint8_t(8)" if operation == "rdiv" else "uint8_t(1), uint8_t(1), uint8_t(0), uint8_t(8), uint8_t(8), uint8_t(1)"
    assert strides in line and "(uint8_t)16" in line
    # The count wrapper ends in normal mode but leaves count mask bits.
    # Each new normal binary explicitly establishes all 64 FP32 lanes.
    prefix = source[:division_start]
    assert prefix.rfind("asc_set_mask_norm();") > prefix.rfind("asc_brcb(")
    assert prefix.rfind("asc_set_vector_mask(~uint64_t(0), ~uint64_t(0));") > prefix.rfind("asc_brcb(")
    assert "asc_sync_pipe(PIPE_V)" in source[source.index(";", source.index("asc_brcb(")):division_start]
    assert "membase_scalar_" not in source and "asc_reciprocal" not in source
    if output_dtype != "float32":
        cast = "float2half_rn" if output_dtype == "float16" else "float2bfloat16_rn"
        assert int(calls(after, "tl.ascend_membase." + cast)[0].args[-1]) == 1024
        assert re.search("asc_" + cast + r"\([^;]+\(uint32_t\)1024\)", source)
        assert "asc_sync_pipe(PIPE_V)" in source[source.index(";", division_start):source.index("asc_" + cast + "(")]
