"""Native2201 reductions expose every partial stage and compact scalar effect."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region


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
        ({"kind": "sum", "dtype": "float16"}, "native float16 sum uses float16 intermediates"),
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
