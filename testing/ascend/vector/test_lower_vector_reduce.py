"""Native2201 reductions expose every partial stage and compact scalar effect."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region, seq


def _lower(mod):
    return tvm.ffi.get_global_func("tl.transform.AscendLowerVectorReduce")()(mod)


def _reduce(kind="sum", *, count=130, rows=3, dtype="float32", clear=True, axis=1, arch="dav-2201", source_shape=None):
    # An aligned row pitch accommodates count tails without including padding
    # in the reduction. The source region retains exactly `count` values.
    block = 32 // (2 if dtype == "float16" else 4)
    width = (count + block - 1) // block * block
    src = tirx.decl_buffer(source_shape or (rows, width), dtype, name="src", scope="shared")
    dst = tirx.decl_buffer((rows,), dtype, name="dst", scope="shared")
    call = tirx.Call(
        "handle",
        tvm.ir.Op.get("tl.tileop.reduce"),
        [region(src, 1, shape=[rows, count]), region(dst, 2 if clear else 3), tirx.StringImm(kind), axis, tirx.const(clear, "bool")],
    )
    mod = kernel(tirx.Evaluate(call), buffers=[src, dst])
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})
    return mod, src, dst


def _native(ir):
    return [call for call in nodes(ir, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.repeat_reduce_")]


def _footprint(call):
    return [(arg.args[0].buffer.name, int(arg.args[1]), int(arg.args[2])) for arg in call.args[:2]]


def test_count_tail_produces_partials_then_a_distinct_merge_task():
    before, src, dst = _reduce(count=130)
    after = _lower(before)
    first, merge = _native(after)
    assert [int(call.args[-1]) for call in (first, merge)] == [130, 3]
    assert _footprint(first)[0][1:] == (3, 2)
    assert _footprint(first)[1] == (src.name, 130, 1)
    assert _footprint(merge)[0] == (dst.name, 1, 2)
    assert _footprint(merge)[1][1:] == (3, 1)
    (scratch,) = [buf for buf in after["main"].body.block.alloc_buffers if buf.name.startswith("membase_reduce_tmp_")]
    assert scratch.scope() == "shared.dyn" and int(scratch.shape[0]) == 8
    assert not nodes(after, tirx.Bind)
    materialized = transform.MaterializeScheduleUnits()(after)
    units = [node for node in nodes(materialized, tirx.AttrStmt) if node.attr_key == "tl.schedule_unit"]
    assert len(units) == 3  # one row control plus two separate vector tasks


def test_three_level_reduction_stays_compact_ir():
    before, _, _ = _reduce(count=5001, rows=1)
    after = _lower(before)
    assert [int(call.args[-1]) for call in _native(after)] == [5001, 79, 2]
    scratch = [buf for buf in after["main"].body.block.alloc_buffers if buf.name.startswith("membase_reduce_tmp_")]
    assert [int(buf.shape[0]) for buf in scratch] == [80, 8]
    assert len(nodes(after, tirx.For)) == 1


@pytest.mark.parametrize("dtype, count, partials", [("float32", 130, 3), ("float16", 257, 3)])
def test_max_only_value_exposes_aligned_scratch_readback_and_compact_store(dtype, count, partials):
    before, _, dst = _reduce("max", dtype=dtype, count=count)
    after = _lower(before)
    native = _native(after)
    assert all(call.op.get_attr("TGlobalSymbol") == "asc_repeat_reduce_max_only_value" for call in native)
    assert [int(call.args[-1]) for call in native] == [count, partials]
    assert all(call.args[0].args[0].buffer.name.startswith("membase_reduce_tmp_") for call in native)
    (bind,) = nodes(after, tirx.Bind)
    assert isinstance(bind.value, tirx.BufferLoad)
    (store,) = nodes(after, tirx.BufferStore)
    assert store.buffer.same_as(dst) and store.value.same_as(bind.var)
    assert len(nodes(after, tirx.For)) == 1


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_clear_false_reads_and_combines_the_previous_compact_output(kind):
    before, _, dst = _reduce(kind, clear=False)
    after = _lower(before)
    result, previous = nodes(after, tirx.Bind)
    assert previous.value.buffer.same_as(dst)
    (store,) = nodes(after, tirx.BufferStore)
    assert isinstance(store.value, tirx.Add if kind == "sum" else tirx.Max)
    assert store.value.a.same_as(result.var) and store.value.b.same_as(previous.var)


def test_one_repeat_sum_writes_directly_without_scratch():
    before, _, dst = _reduce(count=17)
    after = _lower(before)
    (call,) = _native(after)
    assert _footprint(call)[0] == (dst.name, 1, 2)
    assert int(call.args[-1]) == 17
    assert not nodes(after, tirx.Bind)
    assert len(after["main"].body.block.alloc_buffers) == 2


def test_aligned_single_row_max_can_write_directly_after_partial_merge():
    before, _, dst = _reduce("max", count=130, rows=1)
    after = _lower(before)
    assert _footprint(_native(after)[-1])[0] == (dst.name, 1, 2)
    assert not nodes(after, tirx.Bind)
    assert not nodes(after, tirx.BufferStore)


def test_misaligned_source_rows_are_diagnosed_without_overreading_padding():
    before, _, _ = _reduce(count=17, source_shape=(3, 17))
    with pytest.raises(ValueError, match="32-byte aligned"):
        _lower(before)


@pytest.mark.parametrize(
    "kwargs, diagnostic",
    [
        ({"axis": 0}, "trailing axis"),
        ({"kind": "sum", "dtype": "float16"}, "Low-precision sum output needs an explicit accumulator-precision contract"),
        ({"kind": "min"}, "only reduce_sum and reduce_max"),
        ({"count": 0}, "count zero must not issue"),
    ],
)
def test_unsupported_reduction_contracts_have_specific_diagnostics(kwargs, diagnostic):
    before, _, _ = _reduce(**kwargs)
    with pytest.raises(ValueError, match=diagnostic):
        _lower(before)


def test_compound_reduction_cannot_hide_in_one_opaque_task():
    before, _, _ = _reduce("max", count=130)
    root = before["main"].body.block
    root = tirx.SBlock([], [], [], root.name_hint, tirx.AttrStmt({}, "tl.ascend_task", 1, root.body), alloc_buffers=root.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, root))})
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)


def test_3510_reduction_retains_upstream_vf_path():
    before, _, _ = _reduce(arch="dav-3510")
    tvm.ir.assert_structural_equal(_lower(before), before)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("rows,columns", [(1, 512), (16, 1024), (3, 1536), (2, 4096), (64, 512)])
def test_dense_datablock_stages_have_exact_compact_effects_and_separate_tasks(kind, rows, columns):
    before, source, destination = _reduce(kind, count=columns, rows=rows)
    after = _lower(before)
    name = "tl.ascend_membase.datablock_reduce_" + kind
    first, second = calls(after, name)
    suffix = "sum" if kind == "sum" else "max"
    (last,) = calls(after, "tl.ascend_membase.repeat_reduce_" + suffix + "_rows_f32")
    blocks, partials = (call.args[0].args[0].buffer for call in (first, second))
    assert tuple(int(value) for value in blocks.shape) == (rows, columns // 8)
    assert tuple(int(value) for value in partials.shape) == (rows, columns // 64)
    assert not blocks.data.same_as(partials.data)
    for call, src, dst, count, produced in (
        (first, source, blocks, rows * columns, rows * columns // 8),
        (second, blocks, partials, rows * columns // 8, rows * columns // 64),
    ):
        assert call.op.get_attr("TGlobalSymbol") == "asc_datablock_reduce_" + kind
        assert len(call.args) == 3 and str(call.args[2].dtype) == "uint32"
        assert int(call.args[2]) == count and count % 64 == 0
        for access, buffer, extent, mask in ((call.args[0], dst, produced, 2), (call.args[1], src, count, 1)):
            assert access.op.name == "tl.access_ptr"
            assert access.args[0].buffer.same_as(buffer)
            assert [int(value) for value in access.args[0].indices] == [0, 0]
            assert [int(value) for value in access.args[1:]] == [extent, mask]
            assert buffer.dtype == "float32" and buffer.scope() in ("shared", "shared.dyn")
    assert last.args[0].args[0].buffer.same_as(destination)
    assert last.args[1].args[0].buffer.same_as(partials)
    assert [int(value) for value in last.args[1].args[2:]] == [rows, columns // 64]
    assert [int(value) for value in last.args[2:]] == [rows, columns // 512, columns // 64]
    assert not nodes(after, tirx.Bind) and not nodes(after, tirx.For)
    materialized = transform.MaterializeScheduleUnits()(transform.AscendLowerVectorTiles()(after))
    tasks = [node for node in nodes(materialized, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(tasks) == 3
    assert [len(calls(task, name)) for task in tasks] == [1, 1, 0]


def _capacity_program(before, *, extra_buffers=(), body=None):
    root = before["main"].body.block
    mod = kernel(root.body if body is None else body, buffers=[*root.alloc_buffers, *extra_buffers])
    mod["main"] = mod["main"].with_attr("target", before["main"].attrs["target"])
    return mod


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_datablock_headroom_falls_back_to_the_smaller_two_level_native_plan(kind):
    before, _, _ = _reduce(kind, count=1024, rows=16)
    # Root source/output=65536+64B. Leave exactly the old 1024B partials,
    # rather than the candidate's disjoint 8192+1024B scratch allocation.
    filler = tirx.decl_buffer(((196608 - 65536 - 64 - 1024) // 4,), "float32", "filler", scope="shared.dyn")
    after = _lower(_capacity_program(before, extra_buffers=[filler]))
    assert not calls(after, "tl.ascend_membase.datablock_reduce_" + kind)
    whole = "tl.ascend_membase.repeat_reduce_" + ("sum" if kind == "sum" else "max_only_value")
    (first,) = calls(after, whole)
    assert int(first.args[2]) == 16384
    (last,) = calls(after, "tl.ascend_membase.repeat_reduce_" + kind + "_rows_f32")
    assert tuple(int(value) for value in last.args[1].args[0].buffer.shape) == (16, 16)


def test_datablock_scratch_is_charged_before_the_next_reduce_selects_its_plan():
    before, _, _ = _reduce(count=1024, rows=16)
    second_output = tirx.decl_buffer((16,), "float32", "second_output", scope="shared.dyn")
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(second_output, 2)
    body = seq(tirx.Evaluate(original), tirx.Evaluate(tirx.Call("handle", original.op, args)))
    # Reserve exactly one 9216B candidate and one 1024B fallback. Two
    # candidates must not both consume the same remaining-root budget.
    filler_bytes = 196608 - 65536 - 2 * 64 - 9216 - 1024
    filler = tirx.decl_buffer((filler_bytes // 4,), "float32", "filler", scope="shared.dyn")
    after = _lower(_capacity_program(before, extra_buffers=[second_output, filler], body=body))
    assert len(calls(after, "tl.ascend_membase.datablock_reduce_sum")) == 2
    assert len(calls(after, "tl.ascend_membase.repeat_reduce_sum")) == 1
    assert len(calls(after, "tl.ascend_membase.repeat_reduce_sum_rows_f32")) == 2
    scratch = after["main"].body.block.alloc_buffers[-3:]
    assert sum(int(buffer.shape[0]) * int(buffer.shape[1]) * 4 for buffer in scratch) == 10240
