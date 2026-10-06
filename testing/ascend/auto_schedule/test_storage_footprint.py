"""Scheduling, version origins and allocation share physical storage facts."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _target_2201(mod):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _scheduled_ub(ub, *, fixed=(), alias=None, versions=None, layout=None):
    """Explicit costs and eligibility isolate resource accounting from inference."""
    extent = ub.shape[0]
    a = tirx.decl_buffer((4, extent), ub.dtype, name="A")
    c = tirx.decl_buffer((4, extent), ub.dtype, name="C")
    i = tirx.Var("i", "int32")
    tasks = [unit(tirx.BufferStore(buffer, tirx.const(0, buffer.dtype), [0]), stage=-1, cost=(1, 1)) for buffer in fixed]
    if alias is not None:
        tasks.append(unit(tirx.BufferStore(alias, tirx.const(0, alias.dtype), [0]), stage=-1, cost=(1, 1)))
    tasks += [
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, extent]), stage=-1, cost=(1, 1)),
        unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, extent]), stage=-1, cost=(1, 1)),
    ]
    loop = unit(
        tirx.For(i, 0, 4, tirx.ForKind.SERIAL, seq(*tasks), annotations={"num_stages": 2, "multi_buffer_eligible": [ub.data]}),
        core=None,
        stage=-1,
    )
    annotations = {}
    if versions is not None:
        annotations["tl.buffer_versions_map"] = {ub.data: versions}
    if layout is not None:
        annotations["layout_map"] = {ub: layout}
    params = [a, c, extent] if isinstance(extent, tirx.Var) else [a, c]
    return _target_2201(kernel(loop, buffers=[ub, *fixed], params=params, annotations=annotations))


def test_capacity_includes_each_small_buffer_alignment():
    # Raw usage: 196432 + 2*68 = 196568 B, below 192 KiB.
    # Physical usage: 196448 + 2*96 = 196640 B, above 192 KiB.
    ub = tirx.decl_buffer((17,), "float32", name="ub", scope="shared.dyn")
    fixed = [
        tirx.decl_buffer((49108,), "float32", name="large", scope="shared.dyn"),
        tirx.decl_buffer((17,), "float32", name="small", scope="shared.dyn"),
    ]
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        transform.AutoSchedule()(_scheduled_ub(ub, fixed=fixed))


def test_fixed_versions_are_charged_aligned_slots_even_without_solver_variables():
    ub = tirx.decl_buffer((17,), "float32", name="ub", scope="shared.dyn")
    fixed = tirx.decl_buffer((49112,), "float32", name="fixed", scope="shared.dyn")
    # 160 B remain: one 96 B slot fits; two compact 68 B slots only appear to.
    transform.AutoSchedule()(_scheduled_ub(ub, fixed=[fixed], versions=1))
    with pytest.raises(tvm.error.InternalError, match="fixed-version buffers already exceed"):
        transform.AutoSchedule()(_scheduled_ub(ub, fixed=[fixed], versions=2))



def test_manual_leading_version_axis_includes_per_slot_padding():
    ub = tirx.decl_buffer((2, 17), "float32", name="manual_ub", scope="shared.dyn")
    fixed = tirx.decl_buffer((49112,), "float32", name="fixed", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(tirx.BufferStore(fixed, tirx.const(0, "float32"), [0]), stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(ub, tirx.const(1, "float32"), [i % 2, 0]), stage=-1, cost=(1, 1)),
    )
    loop = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None, stage=-1)
    before = _target_2201(kernel(loop, buffers=[ub, fixed], annotations={"tl.manual_multi_buffer": {ub.data: 2}}))
    # The compact whole ring rounds to 160 B, but the two native 96 B slots
    # require 192 B. Only 160 B remain.
    with pytest.raises(tvm.error.InternalError, match="Non-eligible and fixed-version buffers already exceed"):
        transform.AutoSchedule()(before)


def test_first_small_alias_cannot_hide_the_allocation_owner():
    ub = tirx.decl_buffer((200 * 1024 // 4,), "float32", name="ub", scope="shared.dyn")
    alias = tirx.decl_buffer((1,), "uint8", name="tiny_view", data=ub.data)
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        transform.AutoSchedule()(_scheduled_ub(ub, alias=alias))


def test_layout_output_shape_is_part_of_the_physical_footprint():
    ub = tirx.decl_buffer((17,), "float32", name="ub", scope="shared.dyn")
    fixed = tirx.decl_buffer((49120,), "float32", name="fixed", scope="shared.dyn")
    # Logical UB is 68 B; layout occupies 33 f32 values -> a 160 B slot.
    # Only 128 B remain after fixed storage.
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        transform.AutoSchedule()(_scheduled_ub(ub, fixed=[fixed], layout=Layout((17,), lambda i: i * 2)))


def test_dynamic_storage_is_diagnosed_instead_of_charged_zero():
    n = tirx.Var("n", "int32")
    ub = tirx.decl_buffer((n,), "float32", name="dynamic_ub", scope="shared.dyn")
    with pytest.raises(tvm.error.InternalError, match="requires a static physical storage footprint for dynamic_ub"):
        transform.AutoSchedule()(_scheduled_ub(ub))


@pytest.mark.parametrize("dtype, pitch, bytes_per_element", [("float16", 32, 2), ("float32", 24, 4)])
def test_aligned_version_origin_is_shared_by_reinterpreting_views(dtype, pitch, bytes_per_element):
    base = tirx.decl_buffer((17,), dtype, name="base", scope="shared.dyn")
    alias = tirx.decl_buffer((17 * bytes_per_element,), "uint8", name="bytes", data=base.data)
    i = tirx.Var("i", "int32")
    body = seq(
        unit(tirx.BufferStore(base, tirx.const(1, dtype), [0])),
        unit(tirx.BufferStore(alias, tirx.const(2, "uint8"), [0])),
    )
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [base.data]})
    before = kernel(unit(loop, core=None), buffers=[base], annotations={"tl.buffer_versions_map": {base.data: 2}})
    after = transform.MaterializeMultiBuffer()(before)
    stores = nodes(after, tirx.BufferStore)
    assert int(stores[0].buffer.strides[0]) == pitch
    assert int(stores[1].buffer.strides[0]) == pitch * bytes_per_element
    assert pitch * bytes_per_element % 32 == 0

    # The late manual-version layout pass must preserve this already aligned
    # automatic pitch, not divide physical offsets by a compact 17-value slot.
    after = transform.LowerScheduledTIR()(after)
    after = transform.NormalizeBufferVersion()(after)
    after = tilelang.transform.LowerOpaqueBlock()(after)
    after = transform.RewriteAscendBufferVersionLayout()(after)
    after = tilelang.transform.FlattenBuffer()(after)
    stores = nodes(after, tirx.BufferStore)
    (physical_loop,) = nodes(after, tirx.For)
    epoch = physical_loop.loop_var
    analyzer = tvm.arith.Analyzer()
    for store, byte_width in zip(stores, (bytes_per_element, 1)):
        offset = analyzer.simplify(tirx.stmt_functor.substitute(store.indices[0], {epoch: tirx.IntImm(epoch.dtype, 1)}))
        assert int(offset) * byte_width == pitch * bytes_per_element
    allocation = next(node.buffer for node in nodes(after, tirx.AllocBuffer) if node.buffer.name == "base")
    assert int(allocation.shape[0]) == 2 * pitch


def test_allocator_uses_the_same_ub_start_alignment():
    a = tirx.decl_buffer((17,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((17,), "float32", name="b", scope="shared.dyn")
    body = seq(tirx.AllocBuffer(a), tirx.AllocBuffer(b), tirx.BufferStore(a, tirx.const(1, "float32"), [0]), tirx.BufferStore(b, tirx.const(2, "float32"), [0]))
    before = tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt({a.data: [], b.data: []}, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(align_bytes=16, disable_reuse=True)(before)
    offsets = sorted(int(tvm.arith.Analyzer().simplify(store.indices[0])) * 4 for store in nodes(after, tirx.BufferStore))
    assert offsets[0] == 0
    assert offsets[1] == 96
