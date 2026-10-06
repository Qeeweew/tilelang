"""Compiler-owned GM versions share the existing storage clock and byte pitch.

Costs are scheduling weights supplied by the fixture, not device latencies.
"""

from unittest.mock import patch

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


WORKSPACES = "tl.ascend.compiler_workspace_storages"


def _target(mod):
    return tvm.IRModule({"main": mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _ring(*, registered=True, external=False, steps=4, stages=4,
          groups=1024, explicit_versions=None, scheduled=False):
    scratch = tirx.decl_buffer((tirx.IntImm("int64", groups), 32, 48),
                              "float32", name="arbitrary_storage")
    accum = tirx.decl_buffer((32, 48), "float32", name="accum",
                            scope="shared.l0c")
    ub = tirx.decl_buffer((32, 48), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("epoch", "int32")
    stage = 0 if scheduled else -1
    body = seq(
        unit(copy(accum, scratch, dst_indices=[5, 0, 0],
                  dst_shape=[1, 32, 48]), core=2, stage=stage, cost=(3, 1)),
        unit(copy(scratch, ub, src_indices=[5, 0, 0],
                  src_shape=[1, 32, 48]), core=1, stage=stage, cost=(3, 1)),
    )
    annotations = {"tl.ascend_auto_pipeline": True, "num_stages": stages}
    loop = unit(tirx.For(i, 0, steps, tirx.ForKind.SERIAL, body,
                        annotations=annotations), core=None, stage=stage)
    root = {"vector_count": 2}
    if registered:
        root[WORKSPACES] = [scratch.data]
    if explicit_versions is not None:
        root["tl.buffer_versions_map"] = {scratch.data: explicit_versions}
    mod = kernel(loop, buffers=[accum, ub] + ([] if external else [scratch]),
                 params=[scratch] if external else [], annotations=root)
    return _target(mod), scratch


def _versions(mod, storage):
    return int(mod["main"].body.block.annotations["tl.buffer_versions_map"][storage])


def _physical_owner(mod, storage):
    return next(buffer for block in nodes(mod, tirx.SBlock)
                for buffer in block.alloc_buffers if buffer.data.same_as(storage))


def test_compiler_provenance_reaches_the_existing_owner_frontier():
    before, scratch = _ring()
    after = transform.AnnotateMultiBufferEligible()(before)
    (owner,) = nodes(after, tirx.For)
    assert scratch.data in owner.annotations["multi_buffer_eligible"]
    assert scratch.data in after["main"].body.block.annotations[WORKSPACES]


@pytest.mark.parametrize("external", [False, True], ids=["user-global", "parameter"])
def test_unregistered_global_storage_is_not_an_automatic_candidate(external):
    before, scratch = _ring(registered=False, external=external)
    after = transform.AnnotateMultiBufferEligible()(before)
    (owner,) = nodes(after, tirx.For)
    assert scratch.data not in owner.annotations["multi_buffer_eligible"]
    after = transform.AutoSchedule()(after)
    assert scratch.data not in after["main"].body.block.annotations.get("tl.buffer_versions_map", {})


def test_registered_workspace_must_have_a_root_allocation_owner():
    before, _ = _ring(external=True)
    with pytest.raises(tvm.error.InternalError, match="compiler workspace.*root global allocation"):
        transform.AnnotateMultiBufferEligible()(before)


def test_gm_versions_do_not_consume_ub_l1_or_l0_capacity():
    before, scratch = _ring()
    after = transform.AutoSchedule()(transform.AnnotateMultiBufferEligible()(before))
    # One GM slot is 6 MiB, above every individual 2201 on-chip pool.
    assert _versions(after, scratch.data) == 2


def test_solver_receives_independent_byte_bound_and_version_distance():
    before, scratch = _ring()
    before = transform.AnnotateMultiBufferEligible()(before)
    captured = {}

    def schedule(*args, **kwargs):
        captured["sizes"] = list(args[8])
        captured["groups"] = [list(group) for group in args[9]]
        captured["deps"] = [tuple(dependency) for dependency in args[4]]
        return [0, 3], [2] * len(captured["sizes"]), 3

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        after = transform.AutoSchedule()(before)
    index = captured["sizes"].index(1024 * 32 * 48 * 4)
    containing = [group for group in captured["groups"] if index in group[1:]]
    assert containing == [[(1 << 63) - 1, index]]
    assert any(int(distance) == -index - 1 for _, _, distance, _ in captured["deps"])
    assert _versions(after, scratch.data) == 2


@pytest.mark.parametrize("steps, stages", [(1, 4), (4, 1)])
def test_gm_versions_obey_the_existing_trip_and_stage_bounds(steps, stages):
    before, scratch = _ring(steps=steps, stages=stages)
    after = transform.AutoSchedule()(transform.AnnotateMultiBufferEligible()(before))
    assert _versions(after, scratch.data) == 1


def test_materialized_slot_pitch_covers_the_complete_grid_and_preserves_regions():
    before, scratch = _ring()
    after = transform.AutoSchedule()(transform.AnnotateMultiBufferEligible()(before))
    after = transform.MaterializeMultiBuffer()(after)
    owner = _physical_owner(after, scratch.data)
    assert tuple(int(value) for value in owner.shape) == (2, 1024, 32, 48)
    assert int(owner.strides[0]) == 1024 * 32 * 48
    accesses = [call for call in calls(after, "tl.region")
                if call.args[0].buffer.data.same_as(scratch.data)]
    assert len(accesses) == 2
    for access in accesses:
        # New version extent, original single group, valid rows, valid columns.
        assert tuple(int(value) for value in access.args[2:]) == (1, 1, 32, 48)
        assert int(access.args[0].indices[1]) == 5
    producer, consumer = [call.args[0].indices[0] for call in accesses]
    assert tvm.arith.Analyzer().can_prove_equal(producer, consumer)


def _large_byte_workspace(slot_bytes, versions):
    scratch = tirx.decl_buffer((tirx.IntImm("int64", slot_bytes),), "uint8",
                              name="large_workspace")
    out = tirx.decl_buffer((4,), "uint8", name="out")
    i = tirx.Var("i", "int32")
    body = seq(unit(tirx.BufferStore(scratch, tirx.const(1, "uint8"), [0])),
               unit(tirx.BufferStore(out, scratch[0], [i])))
    loop = tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body,
                   annotations={"multi_buffer_eligible": [scratch.data]})
    before = kernel(unit(loop, core=None), buffers=[scratch], params=[out],
                    annotations={WORKSPACES: [scratch.data],
                                 "tl.buffer_versions_map": {scratch.data: versions}})
    return _target(before), scratch


def test_byte_size_overflow_is_rejected_before_version_materialization():
    before, _ = _large_byte_workspace(1 << 62, 2)
    with pytest.raises(tvm.error.InternalError, match="workspace.*byte.*overflow"):
        transform.MaterializeMultiBuffer()(before)


def test_representable_byte_pitch_does_not_overflow_an_intermediate_bit_product():
    before, scratch = _large_byte_workspace((1 << 60) + 32, 2)
    after = transform.MaterializeMultiBuffer()(before)
    owner = _physical_owner(after, scratch.data)
    assert int(owner.strides[0]) == (1 << 60) + 32
    assert int(owner.shape[0]) == 2


def test_generated_matrix_pitch_does_not_overflow_int32_before_the_grid_axis():
    # The handoff boundary validates this 32 GiB slot without allocating it.
    # Its two int32 matrix axes have a legal int64 product of 2**32 elements.
    shape = (65536, 65536)
    source = tirx.decl_buffer(shape, "float32", name="source", scope="shared.l0c")
    destination = tirx.decl_buffer(shape, "float32", name="destination", scope="shared.dyn")
    before = _target(kernel(copy(source, destination), buffers=[source, destination]))
    group = tirx.Var("group", "int32")
    axis = tirx.IterVar(tvm.ir.Range.from_min_extent(0, 2), group,
                       tirx.IterVar.ThreadIndex, "blockIdx.x")
    before = tvm.IRModule({"main": before["main"].with_body(
        tirx.AttrStmt(axis, "thread_extent", 2, before["main"].body))})
    generated = transform.AscendLowerGMHandoff()(before)
    root = next(block for block in nodes(generated, tirx.SBlock)
                if block.name_hint == "tilelang_root")
    (storage,) = root.annotations[WORKSPACES]
    scratch = next(buffer for buffer in root.alloc_buffers
                   if buffer.data.same_as(storage))
    assert tuple(str(value.dtype) for value in scratch.shape) == ("int64", "int32", "int32")

    out = tirx.decl_buffer((4,), "float32", name="out")
    epoch = tirx.Var("epoch", "int32")
    loop = tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, seq(
        unit(tirx.BufferStore(scratch, tirx.const(1, "float32"), [1, 0, 0])),
        unit(tirx.BufferStore(out, scratch[1, 0, 0], [epoch])),
    ), annotations={"multi_buffer_eligible": [storage]})
    prepared = _target(kernel(unit(loop, core=None), buffers=[scratch], params=[out],
                             annotations={WORKSPACES: [storage],
                                          "tl.buffer_versions_map": {storage: 2}}))
    after = transform.MaterializeMultiBuffer()(prepared)
    owner = _physical_owner(after, storage)
    assert tuple(int(value) for value in owner.shape) == (2, 2, 65536, 65536)
    assert tuple(int(value) for value in owner.strides) == (1 << 33, 1 << 32, 65536, 1)


def test_gm_ring_uses_collective_readiness_reuse_and_balanced_fill_drain():
    before, scratch = _ring(explicit_versions=2, scheduled=True)
    before = transform.AnnotateMultiBufferEligible()(before)
    # This scheduled fixture already carries concrete Cube/Vector task masks;
    # AssignCore is the earlier boundary for unresolved task ownership.
    for factory in (transform.PrepareMultiBuffer, transform.ResolveCore,
                    transform.InsertSync):
        before = factory()(before)
    sets = calls(before, "tl.ascend_cross_core_set_flag")
    waits = calls(before, "tl.ascend_cross_core_wait_flag")
    assert sets and waits
    assert {int(call.args[0]) for call in sets + waits} == {2}
    initial = sorted(int(call.args[2]) for call in sets if isinstance(call.args[2], tirx.IntImm))
    drained = sorted(int(call.args[2]) for call in waits if isinstance(call.args[2], tirx.IntImm))
    assert initial == drained and len(initial) == 2
    after = transform.MaterializeMultiBuffer()(before)
    assert int(_physical_owner(after, scratch.data).shape[0]) == 2
