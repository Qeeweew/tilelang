"""A full 256B row is one native repeat and one compact typed result."""

import importlib.util
from pathlib import Path
import sys

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import kernel, nodes, region
from testing.ascend.auto_schedule._scheduled_ir import unit


def _program(*, dtype="float32", kind="max", rows=3, width=None,
             clear=True, retain_axis=False, source_pitch=None,
             source_row=0, output_row=0, source_column=0,
             output_pitch=1, row_extent=None, dst_layout=None):
    vl = 128 if dtype == "float16" else 64
    width = vl if width is None else width
    source_pitch = source_pitch or width + source_column
    capacity = max(rows, 1)
    src = tirx.decl_buffer((capacity + source_row, source_pitch), dtype,
                          "source", scope="shared.dyn")
    shape = (capacity + output_row, output_pitch) if retain_axis else (capacity + output_row,)
    dst = tirx.decl_buffer(shape, dtype, "output", scope="shared.dyn")
    extent = rows if row_extent is None else row_extent
    call = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(src, 1, [source_row, source_column], [extent, width]),
        region(dst, 2 if clear else 3, [output_row, 0] if retain_axis else [output_row],
               [extent, 1] if retain_axis else [extent]),
        tirx.StringImm(kind), 1, tirx.const(clear, "bool"),
    ])
    statement = tirx.Evaluate(call)
    if row_extent is not None:
        # Bound dynamic R in [0, capacity], including a real empty iteration.
        statement = tirx.For(row_extent, 0, capacity + 1, tirx.ForKind.SERIAL, statement)
    annotations = {"layout_map": {dst: dst_layout}} if dst_layout else None
    mod = kernel(statement, buffers=[src, dst], annotations=annotations)
    mod = tvm.IRModule({"main": mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    return mod, src, dst


def _lower(mod):
    return transform.AscendLowerVectorReduce()(mod)


def _reduce_calls(mod):
    return [call for call in nodes(mod, tirx.Call)
            if isinstance(call.op, tvm.ir.Op)
            and call.op.name.startswith("tl.ascend_membase.repeat_reduce_")]


@pytest.mark.parametrize("dtype,kind", [("float32", "max"), ("float16", "max"), ("float32", "sum")])
@pytest.mark.parametrize("rows", [1, 3, 255, 300])
@pytest.mark.parametrize("retain_axis", [False, True])
def test_full_vl_rows_are_one_exact_native_task_without_scalar_scratch(dtype, kind, rows, retain_axis):
    before, src, dst = _program(dtype=dtype, kind=kind, rows=rows, retain_axis=retain_axis)
    after = _lower(before)
    (call,) = _reduce_calls(after)
    vl = 128 if dtype == "float16" else 64
    assert int(call.args[-1]) == rows * vl
    assert call.args[-1].dtype == "uint32"
    assert call.args[0].args[0].buffer.same_as(dst)
    assert call.args[1].args[0].buffer.same_as(src)
    assert int(call.args[0].args[1]) == rows
    assert int(call.args[1].args[1]) == rows * vl
    assert int(call.args[0].args[2]) == 2 and int(call.args[1].args[2]) == 1
    assert not nodes(after, tirx.Bind)
    assert not nodes(after, tirx.BufferStore)
    assert not nodes(after, tirx.For)
    assert len(after["main"].body.block.alloc_buffers) == 2
    materialized = transform.MaterializeScheduleUnits()(after)
    tasks = [node for node in nodes(materialized, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(tasks) == 1


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_nonzero_aligned_region_origins_keep_one_batch_and_true_pointer_bases(dtype):
    before, _, _ = _program(dtype=dtype, source_row=2, output_row=2)
    (call,) = _reduce_calls(_lower(before))
    assert [int(i) for i in call.args[1].args[0].indices] == [2, 0]
    assert [int(i) for i in call.args[0].args[0].indices] == [2]
    assert int(call.args[0].args[1]) == 3


@pytest.mark.parametrize("options", [
    {"clear": False},
    {"width": 17, "source_pitch": 24},
    {"source_pitch": 72},
    {"output_row": 1},
    {"source_column": 16, "source_pitch": 80},
    {"retain_axis": True, "output_pitch": 4},
])
def test_partial_or_noncompact_contract_keeps_existing_row_plan(options):
    before, _, _ = _program(**options)
    after = _lower(before)
    native = _reduce_calls(after)
    assert [int(call.args[-1]) for call in native] == [options.get("width", 64)]
    assert any(loop.kind == tirx.ForKind.SERIAL and int(loop.extent) == 3
               for loop in nodes(after, tirx.For))


def test_proved_strided_destination_layout_falls_back():
    before, _, _ = _program(dst_layout=Layout((3,), lambda row: (2 * row,)))
    after = _lower(before)
    assert [int(call.args[-1]) for call in _reduce_calls(after)] == [64]
    assert nodes(after, tirx.For)


def test_bounded_dynamic_rows_use_one_guarded_exact_native_batch():
    row_extent = tirx.Var("dynamic_rows", "int32")
    before, _, _ = _program(rows=5, row_extent=row_extent)
    after = _lower(before)
    (call,) = _reduce_calls(after)
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(call.args[0].args[1], row_extent)
    assert analyzer.can_prove_equal(call.args[1].args[1], row_extent * 64)
    assert analyzer.can_prove_equal(call.args[-1], tirx.Cast("uint32", row_extent * 64))
    (loop,) = nodes(after, tirx.For)
    assert loop.loop_var.same_as(row_extent)
    (guard,) = nodes(after, tirx.IfThenElse)
    assert analyzer.can_prove_equal(guard.condition, row_extent > 0)
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.BufferStore)


def test_static_empty_rows_do_not_create_a_batched_zero_count_instruction():
    before, _, _ = _program(rows=0)
    after = _lower(before)
    assert [int(call.args[-1]) for call in _reduce_calls(after)] == [64]
    assert any(int(loop.extent) == 0 for loop in nodes(after, tirx.For))


def test_real_materialization_and_tile_lowering_preserve_independent_wide_slot_offsets():
    def index(value):
        return tirx.IntImm("int64", value)

    source = tirx.decl_buffer((index(5), index(64)), "float32", name="versioned_source",
                             scope="shared.dyn", strides=[index(64), index(1)])
    output = tirx.decl_buffer((index(8),), "float32", name="versioned_output",
                             scope="shared.dyn", strides=[index(1)])
    reduction = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(source, 1, [index(2), index(0)], [index(3), index(64)]),
        region(output, 2, [index(2)], [index(3)]),
        tirx.StringImm("max"), 1, tirx.const(True, "bool"),
    ])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    logical = kernel(tirx.Evaluate(reduction), buffers=[source, output])
    logical["main"] = logical["main"].with_attr("target", target)
    (native,) = _reduce_calls(_lower(logical))

    # Subregions consume 3 rows, while physical slots cover the complete
    # allocations: 5*64 input elements and 8 output elements respectively.
    epoch = tirx.Var("wide_epoch", "int64")
    start = 1 << 40
    loop = tirx.For(epoch, index(start), index(7), tirx.ForKind.SERIAL,
                   unit(tirx.Evaluate(native), core=1),
                   annotations={"multi_buffer_eligible": [source.data, output.data]})
    prepared = kernel(unit(loop, core=None), buffers=[source, output], annotations={
        "tl.buffer_versions_map": {source.data: 2, output.data: 3},
    })
    prepared["main"] = prepared["main"].with_attr("target", target)
    with target:
        materialized = transform.MaterializeMultiBuffer()(prepared)
        (versioned,) = _reduce_calls(materialized)
        destination, origin = [argument.args[0] for argument in versioned.args[:2]]
        assert tuple(int(value) for value in origin.buffer.shape) == (2, 5, 64)
        assert tuple(int(value) for value in destination.buffer.shape) == (3, 8)
        assert tuple(int(value) for value in origin.buffer.strides) == (320, 64, 1)
        assert tuple(int(value) for value in destination.buffer.strides) == (8, 1)
        assert all(value.dtype == "int64" for load in (origin, destination)
                   for value in load.buffer.strides)
        assert int(versioned.args[-1]) == 192
        assert [int(argument.args[1]) for argument in versioned.args[:2]] == [3, 192]

        lowered = transform.AscendLowerTileOp()(materialized)
        # LowerTileOp preserves native typed pointers; the existing mechanical
        # access_ptr pass exposes the final C pointer offsets for inspection.
        pointers = tilelang.transform.LowerAccessPtr()(lowered)
    (call,) = _reduce_calls(pointers)
    destination, origin = call.args[:2]
    assert all(argument.op.name.endswith(".tvm_access_ptr") for argument in (destination, origin))
    assert destination.args[1].same_as(output.data)
    assert origin.args[1].same_as(source.data)
    assert [int(argument.args[3]) for argument in (destination, origin)] == [3, 192]
    assert [int(argument.args[4]) for argument in (destination, origin)] == [2, 1]
    assert int(call.args[-1]) == 192 and call.args[-1].dtype == "uint32"
    assert destination.args[2].dtype == origin.args[2].dtype == "int64"
    analyzer = tvm.arith.Analyzer()
    for iteration in (0, 1, 2, 5, 6):
        replacement = {epoch: index(start + iteration)}
        src_offset = tirx.stmt_functor.substitute(origin.args[2], replacement)
        dst_offset = tirx.stmt_functor.substitute(destination.args[2], replacement)
        assert int(analyzer.simplify(src_offset)) == (iteration % 2) * 320 + 128
        assert int(analyzer.simplify(dst_offset)) == (iteration % 3) * 8 + 2


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_ordinary_full_vl_dsl_codegen_has_direct_compact_output(kind):
    @T.prim_func
    def main(A: T.Tensor((3, 64), "float32"), O: T.Tensor((3,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((3, 64), "float32")
            output = T.alloc_shared((3,), "float32")
            T.copy(A, source)
            if kind == "sum":
                T.reduce_sum(source, output, dim=1, clear=True)
            else:
                T.reduce_max(source, output, dim=1, clear=True)
            T.copy(output, O)

    artifact = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"},
                              target_host="c", enable_host_codegen=True,
                              enable_device_compile=False)
    symbol = "asc_repeat_reduce_sum(" if kind == "sum" else "asc_repeat_reduce_max_only_value("
    assert artifact.kernel_source.count(symbol) == 1
    assert "membase_reduce_result" not in artifact.kernel_source
    assert "membase_reduce_tmp" not in artifact.kernel_source
    assert artifact.host_mod


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
def test_ordinary_full_vl_attention_codegen_has_no_per_row_reduce_readback(variant):
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    if not path.exists():
        pytest.skip("The ordinary FA example must accompany this integration test")
    spec = importlib.util.spec_from_file_location("_batched_reduce_attention", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    program = example.flash_attention_fwd(
        1, 1, 1, 32, 64, 16, dtype="float16", variant=variant,
        tiling=example.FwdTiling(block_q=32, block_kv=64),
    )
    artifact = tilelang.lower(program, target={"kind": "ascend", "arch": "dav-2201"},
                              target_host="c", enable_host_codegen=True,
                              enable_device_compile=False)
    assert "asc_repeat_reduce_max_only_value(" in artifact.kernel_source
    assert "asc_repeat_reduce_sum(" in artifact.kernel_source
    assert "membase_reduce_result" not in artifact.kernel_source
    assert "membase_reduce_tmp" not in artifact.kernel_source
    assert artifact.host_mod
