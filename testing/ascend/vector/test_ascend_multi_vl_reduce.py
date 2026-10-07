"""Static FP32 row plans expose native levels, exact effects and UB scratch."""

import importlib.util
from pathlib import Path
import re
import sys

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


TARGET = {"kind": "ascend", "arch": "dav-2201"}
PREFIX = "tl.ascend_membase.repeat_reduce_"


def _program(*, kind="sum", rows=3, columns=128, pitch=None,
             source_row=0, source_column=0, output_row=0, retain_axis=False,
             clear=True, index_dtype="int32", row_extent=None, task=False):
    def index(value):
        return tirx.IntImm(index_dtype, value)

    pitch = pitch or columns + source_column
    capacity = max(rows, 1)
    source = tirx.decl_buffer(
        (index(capacity + source_row), index(pitch)), "float32", "source",
        scope="shared.dyn", strides=[index(pitch), index(1)])
    destination = tirx.decl_buffer(
        (index(capacity + output_row), index(1)) if retain_axis
        else (index(capacity + output_row),), "float32", "destination",
        scope="shared.dyn")
    count = index(rows) if row_extent is None else row_extent
    reduction = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(source, 1, [index(source_row), index(source_column)],
               [count, index(columns)]),
        region(destination, 2 if clear else 3,
               [index(output_row), index(0)] if retain_axis else [index(output_row)],
               [count, index(1)] if retain_axis else [count]),
        tirx.StringImm(kind), index(1), tirx.const(clear, "bool"),
    ])
    statement = tirx.Evaluate(reduction)
    if task:
        statement = tirx.AttrStmt({}, "tl.ascend_task", 1, statement)
    if row_extent is not None:
        statement = tirx.For(row_extent, index(0), index(capacity + 1),
                             tirx.ForKind.SERIAL, statement)
    mod = kernel(statement, buffers=[source, destination])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod, source, destination


def _lower(mod, *, vector=False):
    after = transform.AscendLowerVectorReduce()(mod)
    return transform.AscendLowerVectorTiles()(after) if vector else after


def _strided(ir, kind="sum"):
    name = "sum" if kind == "sum" else "max"
    return calls(ir, PREFIX + name + "_rows_f32")


def _root_program(before, *, extra_buffers=(), annotations=None, body=None, params=()):
    root = before["main"].body.block
    mod = kernel(root.body if body is None else body,
                 buffers=[*root.alloc_buffers, *extra_buffers],
                 annotations=annotations, params=params)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod


def _equal(actual, expected):
    analyzer = tvm.arith.Analyzer()
    assert len(actual) == len(expected)
    assert all(analyzer.can_prove_equal(a, b) for a, b in zip(actual, expected)), (actual, expected)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("rows,columns", [(2, 128), (3, 192), (64, 128), (128, 192), (255, 128)])
@pytest.mark.parametrize("retain_axis", [False, True])
def test_each_chunk_has_exact_regions_aligned_partials_and_no_row_readback(kind, rows, columns, retain_axis):
    before, source, destination = _program(
        kind=kind, rows=rows, columns=columns, retain_axis=retain_axis)
    after = _lower(before)
    native = _strided(after, kind)
    assert len(native) == columns // 64
    partials = native[0].args[0].args[0].buffer
    row_capacity = (rows + 7) // 8 * 8
    _equal(partials.shape, [columns // 64, row_capacity])
    assert partials.dtype == "float32" and partials.scope() == "shared.dyn"
    assert int(partials.shape[-1]) * 4 % 32 == 0
    assert any(partials.same_as(buffer) for buffer in after["main"].body.block.alloc_buffers)
    for chunk, call in enumerate(native):
        assert [str(arg.dtype) for arg in call.args[2:]] == ["uint8", "uint16", "uint8"]
        assert [int(arg) for arg in call.args[2:]] == [rows, columns // 8, 64]
        target, origin = call.args[:2]
        assert target.op.name == origin.op.name == "tl.region"
        assert [int(target.args[1]), int(origin.args[1])] == [2, 1]
        assert origin.args[0].buffer.same_as(source)
        assert target.args[0].buffer.same_as(partials)
        _equal(origin.args[0].indices, [0, chunk * 64])
        _equal(origin.args[2:], [rows, 64])
        _equal(target.args[0].indices, [chunk, 0])
        _equal(target.args[2:], [1, rows])
        assert int(origin.args[0].indices[-1]) + int(origin.args[-1]) <= columns
    assert not nodes(after, tirx.Bind)
    (loop,) = nodes(after, tirx.For)
    assert loop.kind == tirx.ForKind.PARALLEL and int(loop.extent) == rows
    (store,) = nodes(after, tirx.BufferStore)
    assert store.buffer.same_as(destination)
    values = nodes(store.value, tirx.BufferLoad)
    assert len(values) == columns // 64
    assert sorted(int(load.indices[0]) for load in values) == list(range(columns // 64))
    assert all(load.buffer.same_as(partials) and load.indices[1].same_as(loop.loop_var) for load in values)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_distinct_chunk_values_survive_vector_lowering_and_task_materialization(kind):
    before, source, destination = _program(kind=kind, columns=192)
    after = _lower(before, vector=True)
    reductions = _strided(after, kind)
    partials = reductions[0].args[0].args[0].buffer
    combines = calls(after, "tl.ascend_membase." + ("add" if kind == "sum" else "max"))
    assert len(reductions) == 3 and len(combines) == 2
    # Three different columns feed three different scratch rows. A same-buffer
    # CSE may not merge the independent chunk results into one repeated value.
    _equal([op.args[1].args[0].indices[-1] for op in reductions], [0, 64, 128])
    reads = [arg.args[0] for call in combines for arg in call.args[1:3]
             if arg.args[0].buffer.data.same_as(partials.data)]
    assert sorted(int(load.indices[0]) for load in reads) == [0, 1, 2]
    assert all(int(call.args[-1]) == 3 for call in combines)
    assert combines[-1].args[0].args[0].buffer.same_as(destination)
    units = transform.MaterializeScheduleUnits()(after)
    tasks = [task for task in nodes(units, tirx.AttrStmt) if task.attr_key == "tl.ascend_task"]
    assert len(tasks) == 5
    assert all(len(_strided(task.body, kind)) <= 1 for task in tasks)


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_nonzero_regions_and_independent_versions_retain_chunk_and_wide_base_offsets(index_dtype):
    before, source, destination = _program(
        source_row=2, output_row=8, index_dtype=index_dtype, columns=192)
    after = _lower(before)
    native = _strided(after)
    partials = native[0].args[0].args[0].buffer
    for chunk, call in enumerate(native):
        _equal(call.args[1].args[0].indices, [2, chunk * 64])
    epoch = tirx.Var("epoch", index_dtype)
    start = 1 << 40 if index_dtype == "int64" else 0
    loop = tirx.For(epoch, tirx.IntImm(index_dtype, start), tirx.IntImm(index_dtype, 7),
                   tirx.ForKind.SERIAL,
                   seq(*(unit(tirx.Evaluate(call)) for call in native)),
                   annotations={"multi_buffer_eligible": [source.data, partials.data]})
    mod = kernel(unit(loop, core=None), buffers=[source, partials], annotations={
        "tl.buffer_versions_map": {source.data: 2, partials.data: 3},
    })
    target = tvm.target.Target(TARGET)
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        versioned = transform.MaterializeMultiBuffer()(mod)
        for call in _strided(versioned):
            _equal(call.args[1].args[2:], [1, 3, 64])
            _equal(call.args[0].args[2:], [1, 1, 3])
            _equal(call.args[1].args[0].buffer.shape, [2, 5, 192])
            _equal(call.args[0].args[0].buffer.shape, [3, 3, 8])
        pointers = tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(versioned))
    actual_epoch = next(loop.loop_var for loop in nodes(pointers, tirx.For) if loop.loop_var.name == "epoch")
    analyzer = tvm.arith.Analyzer()
    for chunk, call in enumerate(_strided(pointers)):
        target_ptr, source_ptr = call.args[:2]
        assert all(ptr.op.name == "tirx.tvm_access_ptr" for ptr in [target_ptr, source_ptr])
        _equal([target_ptr.args[3], source_ptr.args[3]], [3, 2 * 192 + 64])
        for iteration in [0, 1, 2, 5, 6]:
            replacement = {actual_epoch: tirx.IntImm(index_dtype, start + iteration)}
            offsets = [analyzer.simplify(tirx.stmt_functor.substitute(ptr.args[2], replacement))
                       for ptr in [target_ptr, source_ptr]]
            _equal(offsets, [(iteration % 3) * 24 + chunk * 8,
                             (iteration % 2) * 960 + 2 * 192 + chunk * 64])


@pytest.mark.parametrize("options", [
    {"columns": 64}, {"columns": 130, "pitch": 136}, {"rows": 0},
    {"rows": 256}, {"clear": False}, {"pitch": 136}, {"output_row": 1},
    {"source_column": 64, "pitch": 192}, {"rows": 1}, {"rows": 2, "columns": 192},
])
def test_noneligible_cases_keep_existing_count_mode_plan(options):
    before, _, _ = _program(**options)
    after = _lower(before)
    assert not _strided(after)
    assert calls(after, PREFIX + "sum")
    if options.get("columns") != 64:
        assert any(loop.kind == tirx.ForKind.SERIAL for loop in nodes(after, tirx.For))


def test_dynamic_private_row_extent_including_empty_owner_keeps_existing_plan():
    extent = tirx.Var("owned_rows", "int32")
    before, _, _ = _program(row_extent=extent)
    after = _lower(before)
    assert not _strided(after)
    assert any(loop.extent.same_as(extent) for loop in nodes(after, tirx.For))
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [128, 2]


def test_explicit_atomic_task_cannot_hide_multi_instruction_reduce():
    before, _, _ = _program(task=True)
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)


def test_source_output_alias_is_still_diagnosed_before_native_selection():
    before, source, _ = _program()
    destination = tirx.decl_buffer((3,), "float32", "aliased_destination",
                                  scope="shared.dyn", data=source.data)
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(destination, 2)
    mod = kernel(tirx.Evaluate(tirx.Call("handle", original.op, args)),
                 buffers=[source])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    with pytest.raises(ValueError, match="storage aliasing"):
        _lower(mod)


def test_dav3510_reduction_is_not_rewritten_by_the_2201_plan():
    before, _, _ = _program()
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-3510"}))
    after = _lower(before)
    assert len(calls(after, "tl.tileop.reduce")) == 1
    assert not _strided(after)


def test_larger_expansion_keeps_the_short_existing_row_tree():
    # Outside the bounded two-level domain, 8192/64 chunks would create 255
    # operations while two rows need only six count-mode reductions.
    before, _, _ = _program(rows=2, columns=8192)
    after = _lower(before)
    assert not _strided(after)
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [8192, 128, 2]


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_near_ub_capacity_retains_the_small_existing_row_scratch(kind):
    before, source, destination = _program(kind=kind, rows=127, columns=384)
    # 195072B input + 512B output + one/two 32B old scratch allocations
    # fit in 192KiB. The candidate's partials alone would need 3072B.
    assert int(source.shape[0]) * int(source.shape[1]) * 4 == 195072
    assert (int(destination.shape[0]) * 4 + 31) // 32 * 32 == 512
    after = _lower(before)
    assert not _strided(after, kind)
    name = "sum" if kind == "sum" else "max_only_value"
    assert [int(call.args[-1]) for call in calls(after, PREFIX + name)] == [384, 6]
    scratch = [buffer for buffer in after["main"].body.block.alloc_buffers
               if buffer.name.startswith("membase_reduce_tmp_")]
    assert scratch and all(int(buffer.shape[0]) * 4 == 32 for buffer in scratch)


def test_headroom_includes_future_ordinary_combine_temporaries():
    before, _, _ = _program(rows=6, columns=384)
    assert len(_strided(_lower(before))) == 6
    # The original owners consume 9248B. Partials need 192B, which fits
    # exactly with this filler; five additional row temporaries do not fit.
    filler_bytes = 196608 - 9248 - 192
    filler = tirx.decl_buffer((filler_bytes // 4,), "float32", "filler",
                              scope="shared.dyn")
    after = _lower(_root_program(before, extra_buffers=[filler]))
    assert not _strided(after)
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [384, 6]


def test_pending_scratch_and_reserved_combines_are_charged_across_reductions():
    before, source, destination = _program()
    second = tirx.decl_buffer((3,), "float32", "second", scope="shared.dyn")
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(second, 2)
    body = seq(tirx.Evaluate(original),
               tirx.Evaluate(tirx.Call("handle", original.op, args)))
    # 160B is enough for the first 64B partial allocation and its reserved
    # 32B combine; adding the second candidate and reserve would require 192B.
    original_bytes = 3 * 128 * 4 + 2 * 32
    filler = tirx.decl_buffer(((196608 - original_bytes - 160) // 4,),
                              "float32", "filler", scope="shared.dyn")
    after = _lower(_root_program(before, extra_buffers=[second, filler], body=body))
    assert len(_strided(after)) == 2
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [128, 2]


@pytest.mark.parametrize("padded_layout", [False, True])
def test_owner_aliases_use_one_maximum_layout_footprint(padded_layout):
    before, _, _ = _program(rows=6, columns=384)
    # Exactly enough headroom for partials and every conservative binary
    # temporary when the storage is charged once. A padded alias expands the
    # same physical storage and must force fallback despite its compact owner.
    elements = (196608 - 9248 - (6 + 5) * 32) // 4
    filler = tirx.decl_buffer((elements,), "float32", "filler", scope="shared.dyn")
    alias = tirx.decl_buffer((elements,), "float32", "alias", scope="shared.dyn",
                             data=filler.data)
    stride = 2 if padded_layout else 1
    layouts = {filler: Layout([elements], lambda i: [i]),
               alias: Layout([elements], lambda i: [i * stride])}
    after = _lower(_root_program(before, extra_buffers=[filler],
                                 annotations={"layout_map": layouts}))
    assert len(_strided(after)) == (0 if padded_layout else 6)


def test_unknown_ub_allocation_keeps_existing_plan():
    before, _, _ = _program()
    capacity = tirx.Var("unknown_capacity", "int32")
    filler = tirx.decl_buffer((capacity,),
                              "float32", "filler", scope="shared.dyn")
    after = _lower(_root_program(before, extra_buffers=[filler], params=[capacity]))
    assert not _strided(after)
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [128, 2]


@pytest.mark.parametrize("key", ["tl.buffer_versions_map", "tl.manual_multi_buffer"])
def test_requested_future_ub_versions_keep_existing_plan(key):
    before, source, _ = _program()
    after = _lower(_root_program(before, annotations={
        key: {source.data: 2},
    }))
    assert not _strided(after)


def test_layout_compactness_cannot_override_the_native_raw_row_stride_boundary():
    before, _, destination = _program()
    source = tirx.decl_buffer((3, 128), "float32", "raw_strided_source",
                              scope="shared.dyn", strides=[136, 1])
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[0] = region(source, 1)
    # FlatOffset sees this compact layout, but late native validation still
    # sees the original Buffer stride. Do not select and then fail internally.
    mod = kernel(tirx.Evaluate(tirx.Call("handle", original.op, args)),
                 buffers=[source, destination], annotations={
                     "layout_map": {source: Layout([3, 128], lambda i, j: [i, j])},
                 })
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = _lower(mod)
    assert not _strided(after)
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [128, 2]


def test_two_level_layout_cannot_override_the_native_raw_destination_stride():
    before, source, _ = _program(columns=1024)
    destination = tirx.decl_buffer((3,), "float32", "raw_strided_destination",
                                   scope="shared.dyn", strides=[2])
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(destination, 2)
    mod = kernel(tirx.Evaluate(tirx.Call("handle", original.op, args)),
                 buffers=[source, destination], annotations={
                     "layout_map": {destination: Layout([3], lambda i: [i])},
                 })
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = _lower(mod)
    assert not _strided(after)
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [1024, 16]


@pytest.mark.parametrize("bad", ["rows", "stride", "lanes", "width", "pitch", "alignment", "leading"])
def test_late_native_boundary_rejects_inconsistent_encoded_regions(bad):
    before, source, _ = _program()
    (native, _) = _strided(_lower(before))
    args = list(native.args)
    partials = args[0].args[0].buffer
    buffers = [source, partials]
    if bad == "rows":
        args[2] = tirx.IntImm("uint8", 0)
    elif bad == "stride":
        args[3] = tirx.IntImm("uint16", 0)
    elif bad == "lanes":
        args[4] = tirx.IntImm("uint8", 65)
    elif bad in ["width", "pitch"]:
        operand = list(args[1].args)
        if bad == "width":
            operand[-1] = tirx.IntImm("int32", 65)
        else:
            source = tirx.decl_buffer((3, 128), "float32", "bad_pitch",
                                     scope="shared.dyn", strides=[136, 1])
            operand[0] = source[0, 0]
            buffers[0] = source
        args[1] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), operand)
    else:
        extra = tirx.decl_buffer((2, 3, 8), "float32", "bad_output", scope="shared.dyn")
        args[0] = region(extra, 2, [0, 0, 1] if bad == "alignment" else [0, 0, 0],
                         [1, 1, 3] if bad == "alignment" else [2, 1, 3])
        buffers[1] = extra
    mod = kernel(tirx.Evaluate(tirx.Call("void", native.op, args)), buffers=buffers)
    target = tvm.target.Target(TARGET)
    mod["main"] = mod["main"].with_attr("target", target)
    with target, pytest.raises(tvm.error.InternalError):
        transform.AscendLowerTileOp()(mod)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_ordinary_dsl_source_establishes_masks_and_vector_completion(kind):
    @T.prim_func
    def main(A: T.Tensor((3, 128), "float32"), O: T.Tensor((3,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((3, 128), "float32")
            output = T.alloc_shared((3,), "float32")
            T.copy(A, source)
            if kind == "sum":
                T.reduce_sum(source, output, dim=1, clear=True)
            else:
                T.reduce_max(source, output, dim=1, clear=True)
            T.copy(output, O)

    artifact = tilelang.lower(main, target=TARGET, target_host="c",
                              enable_host_codegen=True, enable_device_compile=False)
    source = artifact.kernel_source
    symbol = "asc_repeat_reduce_sum" if kind == "sum" else "asc_repeat_reduce_max_only_value"
    positions = [match.start() for match in re.finditer(symbol + r"\(", source)]
    assert len(positions) == 2
    # Establishment may be shared by an adjacent native run. Verify the
    # effective mode/mask at each call instead of requiring duplicate setters.
    normal, full = False, False
    checked = 0
    for line in source.splitlines():
        if "asc_set_mask_norm();" in line:
            normal = True
        elif "asc_set_mask_count" in line:
            normal = False
        elif "asc_set_vector_mask(" in line:
            full = "~uint64_t(0), ~uint64_t(0)" in line
        elif symbol + "(" in line:
            assert normal and full
            checked += 1
    assert checked == len(positions)
    native_lines = [line for line in source.splitlines() if symbol + "(" in line]
    assert all("(uint8_t)3, uint16_t(1), uint16_t(1), (uint16_t)16" in line for line in native_lines)
    combine = source.index("asc_add(" if kind == "sum" else "asc_max(")
    assert source.rfind("asc_sync_pipe(PIPE_V);", 0, combine) > max(positions)
    assert "membase_reduce_result" not in source
    assert "asc_repeat_reduce_sum_sync" not in source and artifact.host_mod


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("rows,columns", [(1, 512), (16, 1024), (3, 1536), (2, 4096)])
def test_compact_wide_rows_have_three_native_levels_and_no_combine_tree(kind, rows, columns):
    before, source, destination = _program(kind=kind, rows=rows, columns=columns)
    after = _lower(before, vector=True)
    count_name = "tl.ascend_membase.datablock_reduce_" + kind
    first, second = calls(after, count_name)
    (last,) = _strided(after, kind)
    blocks, partials = (call.args[0].args[0].buffer for call in (first, second))
    _equal(blocks.shape, [rows, columns // 8])
    _equal(partials.shape, [rows, columns // 64])
    assert not blocks.data.same_as(partials.data)
    for call, src, dst, read, written in (
        (first, source, blocks, rows * columns, rows * columns // 8),
        (second, blocks, partials, rows * columns // 8, rows * columns // 64),
    ):
        target, origin = call.args[:2]
        assert target.op.name == origin.op.name == "tl.access_ptr"
        assert target.args[0].buffer.same_as(dst)
        assert origin.args[0].buffer.same_as(src)
        _equal(target.args[0].indices, [0, 0])
        _equal(origin.args[0].indices, [0, 0])
        _equal([target.args[1], origin.args[1], call.args[2]], [written, read, read])
        assert [int(target.args[2]), int(origin.args[2])] == [2, 1]
        assert str(call.args[2].dtype) == "uint32"
    output, inputs = last.args[:2]
    assert output.op.name == inputs.op.name == "tl.region"
    assert output.args[0].buffer.same_as(destination)
    assert inputs.args[0].buffer.same_as(partials)
    _equal(output.args[2:], [rows])
    _equal(inputs.args[2:], [rows, columns // 64])
    assert [str(arg.dtype) for arg in last.args[2:]] == ["uint8", "uint16", "uint8"]
    assert [int(arg) for arg in last.args[2:]] == [rows, columns // 512, columns // 64]
    assert len(after["main"].body.block.alloc_buffers) == 4
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.For)
    assert not calls(after, "tl.ascend_membase.add")
    assert not calls(after, "tl.ascend_membase.max")
    units = transform.MaterializeScheduleUnits()(after)
    tasks = [task for task in nodes(units, tirx.AttrStmt) if task.attr_key == "tl.ascend_task"]
    assert len(tasks) == 3
    assert [len(calls(task, count_name)) for task in tasks] == [1, 1, 0]
    assert len(_strided(tasks[2], kind)) == 1


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("retain_axis", [False, True])
def test_three_level_plan_keeps_nonzero_origins_and_retained_output_axis(kind, retain_axis):
    before, source, destination = _program(
        kind=kind, rows=3, columns=1024, source_row=2, output_row=8,
        retain_axis=retain_axis, index_dtype="int64")
    after = _lower(before)
    first, intermediate = calls(after, "tl.ascend_membase.datablock_reduce_" + kind)
    (second,) = _strided(after, kind)
    _equal(first.args[1].args[0].indices, [2, 0])
    _equal(second.args[0].args[0].indices, [8, 0] if retain_axis else [8])
    _equal(second.args[0].args[2:], [3, 1] if retain_axis else [3])
    target = tvm.target.Target(TARGET)
    with target:
        pointers = tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(after))
    lowered_first, lowered_intermediate = calls(pointers, first.op.name)
    (lowered_second,) = _strided(pointers, kind)
    _equal([lowered_first.args[1].args[2], lowered_first.args[1].args[3]], [2048, 3072])
    _equal([lowered_first.args[0].args[2], lowered_first.args[0].args[3]], [0, 384])
    _equal([lowered_intermediate.args[0].args[2], lowered_intermediate.args[0].args[3]], [0, 48])
    _equal([lowered_intermediate.args[1].args[2], lowered_intermediate.args[1].args[3]], [0, 384])
    _equal([lowered_second.args[0].args[2], lowered_second.args[0].args[3]], [8, 3])
    _equal([lowered_second.args[1].args[2], lowered_second.args[1].args[3]], [0, 48])
    assert lowered_second.args[0].args[1].same_as(destination.data)


@pytest.mark.parametrize("options", [
    {"clear": False}, {"pitch": 1032}, {"output_row": 1},
    {"source_column": 64, "pitch": 1088}, {"rows": 0}, {"rows": 256},
])
def test_two_level_plan_preserves_count_fallback_domains(options):
    before, _, _ = _program(columns=1024, **options)
    after = _lower(before)
    assert not _strided(after)
    assert calls(after, PREFIX + "sum")


def test_non_block_aligned_partials_keep_the_existing_chunk_tree():
    before, _, _ = _program(rows=16, columns=768)
    after = _lower(before)
    assert len(_strided(after)) == 12
    assert all(int(call.args[4]) == 64 for call in _strided(after))
    assert not calls(after, PREFIX + "sum")


def test_two_level_plan_cannot_be_hidden_in_an_atomic_task():
    before, _, _ = _program(columns=1024, task=True)
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)


def test_two_level_scratch_is_charged_across_consecutive_reductions():
    before, source, destination = _program(columns=1024)
    second_output = tirx.decl_buffer((3,), "float32", "second_output", scope="shared.dyn")
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(second_output, 2)
    body = seq(tirx.Evaluate(original), tirx.Evaluate(tirx.Call("handle", original.op, args)))
    # Each two-level plan needs 192B. Leave room for one and 160B more,
    # which admits the established <=64B row fallback for the second reduce.
    owner_bytes = 3 * 1024 * 4 + 2 * 32
    filler = tirx.decl_buffer(((196608 - owner_bytes - 352) // 4,),
                              "float32", "filler", scope="shared.dyn")
    after = _lower(_root_program(before, extra_buffers=[second_output, filler], body=body))
    assert not calls(after, "tl.ascend_membase.datablock_reduce_sum")
    assert len(_strided(after)) == 1
    assert [int(call.args[-1]) for call in calls(after, PREFIX + "sum")] == [3072, 1024, 16]


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
def test_ordinary_attention_bkv128_uses_multi_vl_rows_without_scalar_reduce_stores(variant):
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/910b.py"
    if not path.exists():
        pytest.skip("The ordinary FA example must accompany this integration test")
    spec = importlib.util.spec_from_file_location("_multi_vl_attention", path)
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    program = example.flash_attention_fwd(
        1, 1, 1, 128, 256, 64, dtype="float16", variant=variant,
        tiling=example.FwdTiling(block_q=128, block_kv=128))
    artifact = tilelang.lower(program, target=TARGET, target_host="c",
                              enable_host_codegen=True, enable_device_compile=False)
    source = artifact.kernel_source
    assert source.count("asc_repeat_reduce_max_only_value(") == 2
    assert source.count("asc_repeat_reduce_sum(") == 2
    assert "membase_reduce_result" not in source and artifact.host_mod
