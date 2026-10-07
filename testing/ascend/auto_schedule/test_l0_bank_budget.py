"""Automatic L0A/B/C rings fit the complete kernel's physical bank budget.

The ordinary DSL cases run complete CPU lowering without CANN compilation or
device launches. Small explicitly scheduled inputs isolate reservation across
owners; their scalar task costs are unitless and their solver is replaced by a
known serial witness.
"""

import math
from unittest.mock import patch

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


TARGET = {"kind": "ascend", "arch": "dav-2201"}
BANK_BYTES = 64 * 1024


def _owners(module, storage):
    return [loop.loop_var.name for loop in nodes(module, tirx.For)
            if storage in loop.annotations.get("multi_buffer_eligible", [])]


def _root(module):
    return next(block for block in nodes(module, tirx.SBlock)
                if block.name_hint == "tilelang_root")


def _pv(module):
    gemm, = [call for call in calls(module, "tl.tileop.gemm")
             if int(call.args[5]) == 128 and int(call.args[6]) == 64]
    loop, = [loop for loop in nodes(module, tirx.For)
             if loop.loop_var.name.startswith("gemm_sub_k_")
             and calls(loop.body, "tl.tileop.gemm")]
    return gemm, loop


def _bytes(buffer):
    dtype = tvm.DataType(buffer.dtype)
    return math.prod(map(int, buffer.shape)) * dtype.bits * dtype.lanes // 8


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nested_pv_and_outer_qk_fit_the_final_physical_l0_banks(dtype):
    @T.prim_func
    def program(q: T.Tensor((3, 128, 64), dtype), k: T.Tensor((3, 256, 64), dtype),
                p: T.Tensor((3, 128, 256), dtype), v: T.Tensor((3, 256, 64), dtype),
                score_out: T.Tensor((3, 128, 256), "float32"),
                pv_out: T.Tensor((3, 128, 64), "float32")):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((128, 64), dtype)
            k_l1 = T.alloc_l1((256, 64), dtype)
            p_l1 = T.alloc_l1((128, 256), dtype)
            v_l1 = T.alloc_l1((256, 64), dtype)
            score = T.alloc_l0c((128, 256), "float32")
            pv = T.alloc_l0c((128, 64), "float32")
            for epoch in T.Pipelined(3):
                T.copy(q[epoch, :, :], q_l1)
                T.copy(k[epoch, :, :], k_l1)
                T.gemm(q_l1, k_l1, score, transpose_B=True, clear_accum=True)
                T.copy(score, score_out[epoch, :, :])
                T.copy(p[epoch, :, :], p_l1)
                T.copy(v[epoch, :, :], v_l1)
                T.gemm(p_l1, v_l1, pv, clear_accum=True)
                T.copy(pv, pv_out[epoch, :, :])

    snapshots = {}
    boundaries = {"tl.AnnotateMultiBufferEligible", "tl.AutoSchedule",
                  "tl.MaterializeMultiBuffer", "tl.AscendLowerTileOp"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, module, info):
            if info.name in boundaries:
                snapshots[info.name] = module

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(
            program.with_attr("global_symbol", "main"), target=TARGET,
            enable_host_codegen=False, enable_device_compile=False,
        )
    assert snapshots.keys() == boundaries
    before = snapshots["tl.AnnotateMultiBufferEligible"]
    scheduled = snapshots["tl.AutoSchedule"]
    original, original_loop = _pv(before)
    selected, selected_loop = _pv(scheduled)
    assert int(original_loop.extent) == int(selected_loop.extent) == 2
    assert original_loop.kind == selected_loop.kind == tirx.ForKind.SERIAL
    assert selected_loop.loop_var.same_as(original_loop.loop_var)
    tvm.ir.assert_structural_equal(selected.args[9], original.args[9])
    # Scheduling must keep native sub-K accumulation and the common owner,
    # rather than peeling a clear or moving the L0A allocation to the parent.
    for operand in original.args[:2]:
        storage = operand.args[0].buffer.data
        assert _owners(before, storage) == _owners(scheduled, storage) == [original_loop.loop_var.name]
    pv_a = selected.args[0].args[0].buffer
    versions = _root(scheduled).annotations["tl.buffer_versions_map"]
    assert _bytes(pv_a) == 32 * 1024
    assert int(versions[pv_a.data]) == 1
    for scope in ("shared.l0a", "shared.l0b"):
        allocated = [buffer for buffer in _root(scheduled).alloc_buffers if buffer.scope() == scope]
        assert len(allocated) == 2
        assert all(1 <= int(versions[buffer.data]) <= 2 for buffer in allocated)
        assert sum(_bytes(buffer) * int(versions[buffer.data]) for buffer in allocated) <= BANK_BYTES

    physical_pv_a, = [buffer for buffer in _root(snapshots["tl.MaterializeMultiBuffer"]).alloc_buffers
                      if buffer.data.same_as(pv_a.data)]
    assert _bytes(physical_pv_a) == 32 * 1024
    for scope in ("shared.l0a.dyn", "shared.l0b.dyn"):
        physical = [allocation.buffer for function in artifact.device_mod.functions.values()
                    for allocation in nodes(function, tirx.AllocBuffer)
                    if allocation.buffer.scope() == scope]
        assert len(physical) == 1
        assert 0 < _bytes(physical[0]) <= BANK_BYTES
    assert artifact.kernel_source


def _storage(name, size_bytes, scope):
    return tirx.decl_buffer((size_bytes // 2,), "float16", name=name, scope=scope)


def _owner(storage, name, out, *, child=None):
    epoch = tirx.Var(name, "int32")
    tasks = [unit(tirx.BufferStore(storage, tirx.const(1, storage.dtype), [0]),
                  core=2, stage=-1, cost=(1, 1)),
             unit(tirx.BufferStore(out, storage[0], [0]), core=2, stage=-1, cost=(1, 1))]
    if child is not None:
        tasks.append(child)
    return unit(tirx.For(
        epoch, 0, 3, tirx.ForKind.SERIAL, seq(*tasks),
        annotations={"multi_buffer_eligible": [storage.data],
                     "tl.ascend_auto_pipeline": True, "num_stages": 3},
    ), core=None, stage=-1)


def _module(body, storages, out, *, fixed=None, annotations=None):
    annotations = dict(annotations or {})
    if fixed:
        annotations["tl.buffer_versions_map"] = fixed
    module = kernel(body, buffers=storages, params=[out], annotations=annotations)
    module["main"] = module["main"].with_attr("target", tvm.target.Target(TARGET))
    return module


def _capture_reservations(module, *, first_versions=None):
    queries = []

    def schedule(*args, **kwargs):
        sizes = list(map(int, args[8]))
        counts = list(map(int, kwargs["buffer_min_versions"]))
        if not queries and first_versions is not None:
            assert len(counts) == 1
            counts[0] = first_versions
        groups = [list(map(int, group)) for group in args[9]]
        assert all(sum(sizes[index] * counts[index] for index in group[1:]) <= group[0]
                   for group in groups)
        queries.append({"sizes": sizes, "minimum": list(map(int, kwargs["buffer_min_versions"])),
                        "groups": groups, "selected": counts})
        starts, end = [], 0
        for latency, ii in zip(map(int, args[1]), map(int, args[2])):
            starts.append(end)
            end += max(latency, ii, 1)
        return starts, counts, max(end, 1)

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        after = transform.AutoSchedule()(module)
    return after, queries


def _variable_capacity(query):
    group, = [group for group in query["groups"] if len(group) > 1]
    return group[0]


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_fixed_root_allocation_is_reserved_outside_the_current_owner(scope):
    current = _storage("current", 32 * 1024, scope)
    fixed = _storage("fixed", 16 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    module = _module(_owner(current, "epoch", out), [current, fixed], out)
    _, queries = _capture_reservations(module)
    assert len(queries) == 1
    assert queries[0]["sizes"] == [32 * 1024]
    assert _variable_capacity(queries[0]) == 48 * 1024


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_manual_physical_ring_is_reserved_once_even_when_accessed(scope):
    current = _storage("current", 16 * 1024, scope)
    # Two real 24 KiB slots are already part of this 48 KiB allocation.
    # Charging the manual count again would incorrectly reject the kernel.
    manual = tirx.decl_buffer((2, 12 * 1024), "float16", name="manual", scope=scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    access = unit(tirx.BufferStore(manual, tirx.const(2, "float16"), [0, 0]),
                  core=2, stage=-1, cost=(1, 1))
    body = _owner(current, "epoch", out, child=access)
    module = _module(body, [current, manual], out,
                     annotations={"tl.manual_multi_buffer": {manual.data: 2}})
    after, queries = _capture_reservations(module)
    assert len(queries) == 1 and queries[0]["sizes"] == [16 * 1024]
    assert _variable_capacity(queries[0]) == 16 * 1024
    allocation, = [buffer for buffer in _root(after).alloc_buffers if buffer.data.same_as(manual.data)]
    assert tuple(map(int, allocation.shape)) == (2, 12 * 1024)
    assert _bytes(allocation) == 48 * 1024
    assert int(_root(after).annotations["tl.manual_multi_buffer"][manual.data]) == 2
    assert manual.data not in _root(after).annotations["tl.buffer_versions_map"]


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_reserved_root_layout_and_alignment_determine_the_physical_slot(scope):
    current = _storage("current", 32 * 1024, scope)
    padded = _storage("padded", 16 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    # The compact owner is 16 KiB. Its established layout reserves 16,383
    # f16 elements, whose 512-byte L0 alignment makes a 32 KiB physical slot.
    layout = Layout(padded.shape, lambda index: index * 2)
    module = _module(_owner(current, "epoch", out), [current, padded], out,
                     annotations={"layout_map": {padded: layout}})
    _, queries = _capture_reservations(module)
    assert len(queries) == 1 and queries[0]["sizes"] == [32 * 1024]
    assert _variable_capacity(queries[0]) == 32 * 1024


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_future_fixed_owner_reserves_all_its_versions(scope):
    current = _storage("current", 32 * 1024, scope)
    future = _storage("future", 16 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    module = _module(seq(_owner(current, "first", out), _owner(future, "future_epoch", out)),
                     [current, future], out, fixed={future.data: 2})
    after, queries = _capture_reservations(module)
    assert _variable_capacity(queries[0]) == 32 * 1024
    assert int(_root(after).annotations["tl.buffer_versions_map"][future.data]) == 2


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_previous_owner_count_is_a_lower_bound_and_other_owner_reservation(scope):
    shared_ring = _storage("shared_ring", 16 * 1024, scope)
    other = _storage("other", 32 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    body = seq(_owner(shared_ring, "first", out), _owner(shared_ring, "second", out),
               _owner(other, "other_epoch", out))
    after, queries = _capture_reservations(_module(body, [shared_ring, other], out), first_versions=2)
    assert len(queries) == 3
    assert queries[0]["minimum"] == [1]
    assert queries[1]["minimum"] == [2]
    assert queries[1]["selected"] == [2]
    assert queries[2]["sizes"] == [32 * 1024]
    assert _variable_capacity(queries[2]) == 32 * 1024
    assert int(_root(after).annotations["tl.buffer_versions_map"][shared_ring.data]) == 2


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_nested_owner_reserves_parent_and_parent_reserves_selected_child(scope):
    parent = _storage("parent", 16 * 1024, scope)
    child = _storage("child", 32 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    body = _owner(parent, "parent_epoch", out, child=_owner(child, "child_epoch", out))
    after, queries = _capture_reservations(_module(body, [parent, child], out))
    assert len(queries) == 2
    assert queries[0]["sizes"] == [32 * 1024]
    assert _variable_capacity(queries[0]) == 48 * 1024
    assert queries[1]["sizes"] == [16 * 1024]
    assert _variable_capacity(queries[1]) == 32 * 1024
    assert _owners(after, parent.data) == ["parent_epoch"]
    assert _owners(after, child.data) == ["child_epoch"]


@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_root_reservation_rejects_a_minimum_that_only_fits_the_child_alone(scope):
    current = _storage("current", 32 * 1024, scope)
    fixed = _storage("fixed", 48 * 1024, scope)
    out = tirx.decl_buffer((1,), "float16", name="out")
    module = _module(_owner(current, "epoch", out), [current, fixed], out)
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        _capture_reservations(module)


def _l0c_storage(name, size_bytes):
    return tirx.decl_buffer((size_bytes // 4,), "float32", name=name, scope="shared.l0c")


def test_l0c_fixed_root_allocation_is_reserved_before_selecting_another_ring():
    current = _l0c_storage("current", 64 * 1024)
    fixed = _l0c_storage("fixed", 64 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    _, queries = _capture_reservations(_module(_owner(current, "epoch", out), [current, fixed], out))
    assert len(queries) == 1 and queries[0]["sizes"] == [64 * 1024]
    assert _variable_capacity(queries[0]) == 64 * 1024


def test_l0c_previous_owner_width_is_a_minimum_and_a_foreign_reservation():
    shared_ring = _l0c_storage("shared_ring", 32 * 1024)
    other = _l0c_storage("other", 64 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    body = seq(_owner(shared_ring, "first", out), _owner(shared_ring, "second", out),
               _owner(other, "other_epoch", out))
    after, queries = _capture_reservations(_module(body, [shared_ring, other], out), first_versions=2)
    assert [query["minimum"] for query in queries] == [[1], [2], [1]]
    assert _variable_capacity(queries[2]) == 64 * 1024
    assert int(_root(after).annotations["tl.buffer_versions_map"][shared_ring.data]) == 2


def test_l0c_minimum_cannot_use_another_roots_reserved_capacity():
    current = _l0c_storage("current", 64 * 1024)
    fixed = _l0c_storage("fixed", 96 * 1024)
    out = tirx.decl_buffer((1,), "float32", name="out")
    module = _module(_owner(current, "epoch", out), [current, fixed], out)
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        _capture_reservations(module)
