"""Scheduling, version origins and allocation share physical storage facts."""

from unittest.mock import patch

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
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


def _ub_storage(name, size, scope="shared.dyn"):
    return tirx.decl_buffer((size // 4,), "float32", name=name, scope=scope)


def _ub_owner(storage, name, out, *, child=None, auto_pipeline=True):
    epoch = tirx.Var(name, "int32")
    tasks = [
        unit(tirx.BufferStore(storage, tirx.const(1, "float32"), [0]), stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(out, storage[0], [0]), stage=-1, cost=(1, 1)),
    ]
    if child is not None:
        tasks.append(child)
    annotations = {"multi_buffer_eligible": [storage.data], "num_stages": 3}
    if auto_pipeline:
        annotations["tl.ascend_auto_pipeline"] = True
    return unit(tirx.For(
        epoch, 0, 3, tirx.ForKind.SERIAL, seq(*tasks),
        annotations=annotations,
    ), core=None, stage=-1)


def _ub_module(body, storages, out, *, annotations=None, arch="dav-2201"):
    module = kernel(body, buffers=storages, params=[out], annotations=annotations)
    module["main"] = module["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return module


def _capture_ub_budget(module, *, first_versions=None):
    """Observe the real FFI contract, with a deterministic legal serial witness.

    Unit costs isolate byte accounting from latency estimates. Choosing two
    versions for an explicitly bounded first owner is legal even though this
    serial witness only needs one, and exposes the physical ring shared by
    subsequent owners.
    """
    queries = []

    def schedule(*args, **kwargs):
        sizes = list(map(int, args[8]))
        minimum = list(map(int, kwargs["buffer_min_versions"]))
        counts = list(minimum)
        if not queries and first_versions is not None:
            assert len(counts) == 1
            counts[0] = first_versions
        stage_bound = int(args[0])
        assert all(lower <= count <= max(stage_bound, lower)
                   for lower, count in zip(minimum, counts))
        groups = [list(map(int, group)) for group in args[9]]
        assert all(sum(sizes[index] * counts[index] for index in group[1:]) <= group[0]
                   for group in groups)
        queries.append(dict(sizes=sizes, stage_bound=stage_bound, minimum=minimum,
                            groups=groups, selected=counts))
        starts, end = [], 0
        for latency, ii in zip(map(int, args[1]), map(int, args[2])):
            starts.append(end)
            end += max(latency, ii, 1)
        return starts, counts, max(end, 1)

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        after = transform.AutoSchedule()(module)
    return after, queries


def _ub_variable_capacity(query):
    # These fixtures have exactly one variable buffer, always in the UB group.
    group, = [group for group in query["groups"] if len(group) > 1]
    assert query["sizes"] and group[1:] == list(range(len(query["sizes"])))
    return group[0]


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
def test_root_only_epilogue_is_reserved_before_choosing_a_ub_ring(scope):
    ring = _ub_storage("ring", 64 * 1024, scope)
    epilogue = _ub_storage("epilogue", 96 * 1024, scope)
    out = tirx.decl_buffer((1,), "float32", name="out")
    finish = unit(tirx.BufferStore(epilogue, out[0], [0]), stage=-1, cost=(1, 1))
    _, queries = _capture_ub_budget(_ub_module(
        seq(_ub_owner(ring, "epoch", out), finish), [ring, epilogue], out))
    assert len(queries) == 1 and queries[0]["sizes"] == [64 * 1024]
    # Two ring slots fit if only this loop is budgeted; the actual epilogue
    # allocation leaves room for just one. Its tiny write is not its footprint.
    assert _ub_variable_capacity(queries[0]) == 96 * 1024
    assert queries[0]["selected"] == [1]


def test_root_only_ub_reservation_keeps_layout_padding_and_alignment():
    ring = _ub_storage("ring", 64 * 1024)
    epilogue = _ub_storage("epilogue", 68)
    out = tirx.decl_buffer((1,), "float32", name="out")
    module = _ub_module(_ub_owner(ring, "epoch", out), [ring, epilogue], out,
                        annotations={"layout_map": {epilogue: Layout((17,), lambda i: i * 2)}})
    _, queries = _capture_ub_budget(module)
    assert _ub_variable_capacity(queries[0]) == 192 * 1024 - 160


@pytest.mark.parametrize("accessed", [False, True])
def test_manual_ub_physical_ring_is_reserved_exactly_once(accessed):
    ring = _ub_storage("ring", 64 * 1024)
    manual = tirx.decl_buffer((2, 6144), "float32", name="manual", scope="shared.dyn")
    out = tirx.decl_buffer((1,), "float32", name="out")
    touch = unit(tirx.BufferStore(manual, tirx.const(2, "float32"), [0, 0]), stage=-1, cost=(1, 1)) if accessed else None
    module = _ub_module(_ub_owner(ring, "epoch", out, child=touch), [ring, manual], out,
                        annotations={"tl.manual_multi_buffer": {manual.data: 2}})
    _, queries = _capture_ub_budget(module)
    assert queries[0]["sizes"] == [64 * 1024]
    assert _ub_variable_capacity(queries[0]) == 144 * 1024


def test_previously_selected_ub_width_is_a_lower_bound_and_foreign_reservation():
    ring = _ub_storage("ring", 32 * 1024)
    other = _ub_storage("other", 96 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    # Only the first owner needs an explicit bound for the injected V2.
    # Later scalar-only auto owners keep bound1 and inherit the selected V2
    # as a shared minimum instead of silently allocating a smaller ring.
    module = _ub_module(seq(_ub_owner(ring, "first", out, auto_pipeline=False),
                            _ub_owner(ring, "second", out),
                            _ub_owner(other, "other_epoch", out)), [ring, other], out)
    after, queries = _capture_ub_budget(module, first_versions=2)
    assert [query["stage_bound"] for query in queries] == [3, 1, 1]
    assert [query["minimum"] for query in queries] == [[1], [2], [1]]
    assert [_ub_variable_capacity(query) for query in queries] == [96 * 1024, 96 * 1024, 128 * 1024]
    assert int(after["main"].body.block.annotations["tl.buffer_versions_map"][ring.data]) == 2


def test_parent_ub_budget_counts_the_selected_child_ring_without_double_charging():
    parent = _ub_storage("parent", 32 * 1024)
    child = _ub_storage("child", 32 * 1024)
    epilogue = _ub_storage("epilogue", 16 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    # Child scheduling occurs first. Give only that scalar-only child an
    # explicit bound for V2; the parent remains an automatic one-slot owner.
    body = _ub_owner(parent, "parent_epoch", out,
                     child=_ub_owner(child, "child_epoch", out, auto_pipeline=False))
    after, queries = _capture_ub_budget(_ub_module(body, [parent, child, epilogue], out), first_versions=2)
    assert [query["stage_bound"] for query in queries] == [3, 1]
    assert [query["minimum"] for query in queries] == [[1], [1]]
    assert [query["selected"] for query in queries] == [[2], [1]]
    assert [query["sizes"] for query in queries] == [[32 * 1024], [32 * 1024]]
    assert [_ub_variable_capacity(query) for query in queries] == [144 * 1024, 112 * 1024]
    versions = after["main"].body.block.annotations["tl.buffer_versions_map"]
    assert int(versions[child.data]) == 2 and int(versions[parent.data]) == 1


def test_future_fixed_ub_owner_reserves_its_requested_physical_width():
    current = _ub_storage("current", 64 * 1024)
    future = _ub_storage("future", 32 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    module = _ub_module(seq(_ub_owner(current, "first", out), _ub_owner(future, "future_epoch", out)),
                        [current, future], out, annotations={"tl.buffer_versions_map": {future.data: 2}})
    after, queries = _capture_ub_budget(module)
    assert _ub_variable_capacity(queries[0]) == 128 * 1024
    assert int(after["main"].body.block.annotations["tl.buffer_versions_map"][future.data]) == 2


def test_root_only_ub_storage_cannot_be_spent_on_the_loop_minimum():
    ring = _ub_storage("ring", 128 * 1024)
    epilogue = _ub_storage("epilogue", 80 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    module = _ub_module(_ub_owner(ring, "epoch", out), [ring, epilogue], out)
    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=AssertionError("capacity must fail before solve")):
        with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
            transform.AutoSchedule()(module)


def test_3510_keeps_its_existing_loop_local_ub_budget():
    ring = _ub_storage("ring", 64 * 1024)
    epilogue = _ub_storage("epilogue", 96 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    _, queries = _capture_ub_budget(_ub_module(_ub_owner(ring, "epoch", out), [ring, epilogue], out, arch="dav-3510"))
    assert _ub_variable_capacity(queries[0]) == 248 * 1024


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
