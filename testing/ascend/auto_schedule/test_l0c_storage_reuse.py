"""Capacity admission reuses physical L0C through ordinary hard dependencies.

The fixture costs are unitless scheduling weights. These CPU checks do not
compile AscendC or launch a device kernel.
"""

import math
from unittest.mock import patch

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tilelang.layout import (
    Layout,
    make_ascend_l0a_2201_layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _fixture(*, kind="ordinary", arch="dav-2201", nested=False, width=256):
    rows = 128
    second_rows = 64 if kind == "different-pitch" else rows
    first = tirx.decl_buffer((rows, width), "float32", name="first", scope="shared.l0c")
    second = tirx.decl_buffer(
        (second_rows, 64), "float32", name="second", scope="shared.l0c",
        strides=(80, 1) if kind == "strides" else None,
    )
    a = tirx.decl_buffer((rows, 16), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((width, 16), "float16", name="b", scope="shared.l0b")
    out_first = tirx.decl_buffer((rows, width), "float32", name="out_first")
    out_second = tirx.decl_buffer((second_rows, 64), "float32", name="out_second")
    epoch = tirx.Var("epoch", "int32")
    inner = tirx.Var("inner", "int32")

    def gemm(buffer, *, clear=True, active_rows=None):
        m, n = int(buffer.shape[0]), int(buffer.shape[1])
        if active_rows is not None:
            m = active_rows
        region = lambda value, r, c: tirx.BufferRegion(
            value, [tvm.ir.Range.from_min_extent(0, r), tvm.ir.Range.from_min_extent(0, c)])
        return tirx.Evaluate(T.gemm(region(a, m, 16), region(b, n, 16), region(buffer, m, n),
                                   transpose_B=True, clear_accum=clear,
                                   unit_flag_ctrl=1 if kind == "unit-flag" else None))

    clear = False if kind == "read-first" else epoch == 0 if kind == "first-epoch-only" else True
    requested_stage = 0 if kind == "manual-stage" else -1
    first_write = unit(gemm(first, clear=clear, active_rows=127 if kind == "tail-write" else None),
                       core=2, stage=requested_stage, cost=(1, 1),
                       guard=epoch % 2 == 0 if kind == "conditional" else None)
    second_write = unit(gemm(second, clear=inner == 0 if nested else True),
                        core=2, stage=requested_stage, cost=(1, 1))
    if nested:
        second_write = unit(tirx.For(inner, 0, 2, tirx.ForKind.SERIAL, second_write),
                            core=None, stage=requested_stage)
    first_read = copy(first, out_first)
    if kind in ("scalar-read", "opaque-address", "opaque-region"):
        # Keep the outer operation a supported typed FIX copy, so rejecting
        # an unrelated standalone scalar/foreign task cannot mask this gate.
        # The same storage also occurs as a precise typed source operand.
        if kind == "scalar-read":
            destination_row = tirx.Cast("int32", first[0, 0])
        elif kind == "opaque-address":
            destination_row = tirx.call_extern("int32", "foreign_index", T.address_of(first[0, 0]))
        else:
            destination_row = tirx.call_extern("int32", "foreign_index", region(first, 1))
        first_read = copy(first, out_first, dst_indices=[destination_row, 0])
    # A manual schedule must request stages for every sibling. Mixing one
    # requested stage with automatic siblings fails before capacity admission.
    tasks = [first_write, unit(first_read, core=2, stage=requested_stage, cost=(1, 1)),
             second_write, unit(copy(second, out_second), core=2, stage=requested_stage, cost=(1, 1))]
    if kind == "scalar-write":
        tasks.insert(1, unit(tirx.BufferStore(first, tirx.const(0, "float32"), [0, 0]),
                             core=2, stage=-1, cost=(1, 1)))
    if kind == "overlapping-lifetime":
        tasks.append(unit(copy(first, out_first), core=2, stage=-1, cost=(1, 1)))
    if kind == "escaped-pointer":
        tasks.append(unit(tirx.Evaluate(tirx.call_extern("int32", "foreign", second.data)),
                          core=2, stage=-1, cost=(1, 1)))
    steps = tirx.Var("steps", "int32") if kind == "dynamic-owner" else 3
    annotations = {"multi_buffer_eligible": [first.data, second.data],
                   "tl.ascend_auto_pipeline": True, "num_stages": 4}
    if kind == "conflict-hint":
        annotations["conflict_hint"] = [[first.data, second.data, tirx.IntImm("int32", -1),
                                        tirx.const(False, "bool")]]
    elif kind == "prepared-versions":
        annotations["tl.storage_epoch_guard_map"] = {second.data: tirx.const(True, "bool")}
    loop = unit(tirx.For(epoch, 0, steps, tirx.ForKind.SERIAL, seq(*tasks),
                        annotations=annotations), core=None, stage=-1)
    body = seq(loop, unit(copy(first, out_first), core=2, stage=-1, cost=(1, 1))) if kind == "outside-owner" else loop
    if kind == "different-owners":
        second_epoch = tirx.Var("second_epoch", "int32")
        body = seq(*[
            unit(tirx.For(coordinate, 0, 3, tirx.ForKind.SERIAL, seq(*phase),
                          annotations={**annotations, "multi_buffer_eligible": [buffer.data]}),
                 core=None, stage=-1)
            for coordinate, phase, buffer in ((epoch, tasks[:2], first),
                                               (second_epoch, tasks[2:], second))
        ])
    layouts = {first: make_ascend_l0c_layout(first), second: make_ascend_l0c_layout(second),
               a: make_ascend_l0a_2201_layout(a, False), b: make_ascend_major_k_layout(b)}
    if kind == "custom-layout":
        layouts[second] = Layout(second.shape, lambda i, j: [j // 16, i // 16, i % 16, (j + 1) % 16])
    root = {"layout_map": layouts}
    if kind == "explicit-versions":
        root["tl.buffer_versions_map"] = {second.data: 2}
    elif kind == "manual-versions":
        root["tl.manual_multi_buffer"] = {second.data: 2}
    elif kind == "version-mode":
        root["tl.buffer_version_mode"] = {second.data: "iteration"}
    elif kind == "alias-contract":
        root["tl.buffer_alias_map"] = {first.data: [second.data], second.data: [first.data]}
    elif kind == "padding-value":
        root["safe_value_map"] = {second.data: tirx.const(0, "float32")}
    params = [out_first, out_second] + ([steps] if isinstance(steps, tirx.Var) else [])
    before = kernel(body, buffers=[a, b, first, second], params=params, annotations=root)
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    before["main"] = before["main"].with_attr("target", target)
    return before, first, second


def _three_accumulator_fixture(*, overlapping_middle=False):
    before, first, second = _fixture()
    root = before["main"].body.block
    owner = root.body.body
    a, b = root.alloc_buffers[:2]
    third = tirx.decl_buffer((128, 128), "float32", name="third", scope="shared.l0c")
    out_third = tirx.decl_buffer((128, 128), "float32", name="out_third")

    def region(buffer, rows, columns):
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, rows),
                                          tvm.ir.Range.from_min_extent(0, columns)])

    third_write = unit(tirx.Evaluate(T.gemm(region(a, 128, 16), region(b, 128, 16),
                                           region(third, 128, 128), transpose_B=True,
                                           clear_accum=True)), core=2, stage=-1, cost=(1, 1))
    third_read = unit(copy(third, out_third), core=2, stage=-1, cost=(1, 1))
    first_write, first_read, second_write, second_read = owner.body.seq
    if overlapping_middle:
        # first [0,1] is disjoint from both other intervals, but third [2,5]
        # overlaps second [3,4]. Comparing only with the largest member would
        # therefore incorrectly merge all three and satisfy capacity.
        tasks = [first_write, first_read, third_write, second_write, second_read, third_read]
    else:
        tasks = [first_write, first_read, third_write, third_read, second_write, second_read]
    loop = unit(tirx.For(owner.loop_var, owner.min, owner.extent, owner.kind, seq(*tasks),
                        annotations={**dict(owner.annotations),
                                     "multi_buffer_eligible": [first.data, second.data, third.data]}),
                core=None, stage=-1)
    layouts = {**dict(root.annotations["layout_map"]), third: make_ascend_l0c_layout(third)}
    mod = kernel(loop, buffers=[*root.alloc_buffers, third],
                 params=[*before["main"].buffer_map.values(), out_third],
                 annotations={**dict(root.annotations), "layout_map": layouts})
    mod["main"] = mod["main"].with_attr("target", before["main"].attrs["target"])
    return mod, (first, third, second)


def _l0c_allocations(mod):
    return [buffer for block in nodes(mod, tirx.SBlock) for buffer in block.alloc_buffers
            if buffer.scope() == "shared.l0c"]


def _scheduled(before):
    """A legal source-order witness isolates admission/edges from optimization."""
    captured = []

    def schedule(*args, **kwargs):
        latencies = list(map(int, args[1]))
        captured.append({"deps": [tuple(map(int, item)) for item in args[4]],
                         "sizes": list(map(int, args[8])),
                         "groups": [list(map(int, group)) for group in args[9]]})
        # An entirely serial witness also covers a nested accumulation's
        # aggregate latency. It satisfies the single-version wraparound edge.
        starts, end = [], 0
        for latency, ii in zip(latencies, map(int, args[2])):
            starts.append(end)
            end += max(latency, ii, 1)
        return starts, [1] * len(args[8]), max(end, 1)

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        after = transform.AutoSchedule()(before)
    return after, captured


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested-accumulation"])
def test_admission_preserves_logical_layouts_regions_and_one_physical_owner(nested):
    before, first, second = _fixture(nested=nested)
    after, _ = _scheduled(before)
    (owner,) = _l0c_allocations(after)
    assert owner.data.same_as(first.data)
    assert tuple(map(int, owner.shape)) == (128, 256)
    gemms = calls(after, "tl.tileop.gemm")
    c_regions = [gemm.args[2] for gemm in gemms]
    logical = {tuple(map(int, region.args[2:])): region.args[0].buffer for region in c_regions}
    assert set(logical) == {(128, 256), (128, 64)}
    assert logical[(128, 256)].data.same_as(logical[(128, 64)].data)
    assert not logical[(128, 256)].same_as(logical[(128, 64)])
    root = after["main"].body.block
    assert int(root.annotations["tl.buffer_versions_map"][owner.data]) == 1
    assert second.data not in root.annotations["tl.buffer_versions_map"]
    for buffer in logical.values():
        tvm.ir.assert_structural_equal(root.annotations["layout_map"][buffer], make_ascend_l0c_layout(buffer))
    for loop in nodes(after, tirx.For):
        eligible = loop.annotations.get("multi_buffer_eligible", [])
        assert all(not storage.same_as(second.data) for storage in eligible)
        assert sum(storage.same_as(owner.data) for storage in eligible) <= 1


def test_common_storage_produces_same_epoch_and_loop_carried_completion_edges():
    before, _, _ = _fixture()
    _, plans = _scheduled(before)
    (plan,) = plans
    edges = {dependency[:3] for dependency in plan["deps"]}
    # The original buffers were distinct. These precise cross-buffer hazards
    # must arise from the common physical storage, not Cube/FIX issue order.
    assert (1, 2, 0) in edges  # first FIX read -> second MAD write
    assert (3, 0, 1) in edges  # second FIX read -> next epoch's first MAD
    assert (2, 0, 1) in edges  # cross-epoch WAW is retained as well
    # The 128 KiB fixed storage consumes the entire L0C bank; it is not a
    # second solver variable with a fictitious maximum-size capacity charge.
    assert 128 * 256 * 4 not in plan["sizes"]
    assert any(group == [0] for group in plan["groups"])


def test_three_disjoint_intervals_share_one_slot_and_keep_each_logical_view():
    before, buffers = _three_accumulator_fixture()
    after, (plan,) = _scheduled(before)
    (owner,) = _l0c_allocations(after)
    assert owner.data.same_as(buffers[0].data)
    assert tuple(map(int, owner.shape)) == (128, 256)
    logical = [call.args[2].args[0].buffer for call in calls(after, "tl.tileop.gemm")]
    assert [tuple(map(int, buffer.shape)) for buffer in logical] == [(128, 256), (128, 128), (128, 64)]
    assert all(buffer.data.same_as(owner.data) for buffer in logical)
    assert len(set(logical)) == 3
    root = after["main"].body.block
    versions = root.annotations["tl.buffer_versions_map"]
    assert int(versions[owner.data]) == 1
    assert all(buffer.data not in versions for buffer in buffers[1:])
    for buffer in logical:
        tvm.ir.assert_structural_equal(root.annotations["layout_map"][buffer], make_ascend_l0c_layout(buffer))
    edges = {dependency[:3] for dependency in plan["deps"]}
    assert {(1, 2, 0), (3, 4, 0), (5, 0, 1)} <= edges


def test_three_member_group_checks_overlap_against_every_accepted_member():
    before, _ = _three_accumulator_fixture(overlapping_middle=True)
    with pytest.raises(tvm.error.InternalError, match="shared.l0c"):
        transform.AutoSchedule()(before)


def test_sequential_distinct_unique_loop_owners_keep_independent_storage():
    before, first, second = _fixture(kind="different-owners", width=128)
    after, plans = _scheduled(before)
    assert len(plans) == 2
    assert {buffer.data for buffer in _l0c_allocations(after)} == {first.data, second.data}
    assert [set(loop.annotations["multi_buffer_eligible"]) for loop in nodes(after, tirx.For)] == [
        {first.data}, {second.data}
    ]


def test_distinct_loop_owners_cannot_borrow_each_others_l0c_capacity():
    before, _, _ = _fixture(kind="different-owners")
    snapshot = tvm.ir.save_json(before)
    # Independent 128KiB and 32KiB allocations coexist at the root. No
    # cross-owner reuse proof licenses admitting either owner in isolation.
    with pytest.raises(tvm.error.InternalError, match="shared.l0c.*insufficient for minimum buffer usage"):
        _scheduled(before)
    assert tvm.ir.save_json(before) == snapshot


@pytest.mark.parametrize("nested", [False, True])
def test_sync_and_late_lowering_consume_the_shared_storage_pitch(nested):
    before, _, _ = _fixture(nested=nested)
    after, _ = _scheduled(before)
    with after["main"].attrs["target"]:
        for factory in (transform.AssignCore, transform.PrepareMultiBuffer,
                        transform.ResolveCore, transform.InsertSync):
            after = factory()(after)
        # This direction is the capacity tradeoff: FIX completion protects
        # both same-epoch and loop-carried accumulator reuse.
        sets = [call for call in calls(after, "tl.ascend_set_flag") if str(call.args[0].value) == "FIX_M"]
        waits = [call for call in calls(after, "tl.ascend_wait_flag") if str(call.args[0].value) == "FIX_M"]
        assert sets and len(sets) == len(waits)
        assert {str(call.args[1]) for call in sets} == {str(call.args[1]) for call in waits}
        # At least one completion wait precedes the first MAD in the owner
        # body, and at least one protects the later logical accumulator.
        events = []
        for statement, parents in statements(after):
            if not isinstance(statement, tirx.Evaluate) or not isinstance(statement.value, tirx.Call):
                continue
            call = statement.value
            if not isinstance(call.op, tvm.ir.Op):
                continue
            if call.op.name == "tl.ascend_wait_flag" and str(call.args[0].value) == "FIX_M":
                if any(isinstance(parent, tirx.For) and parent.loop_var.name == "epoch" for parent in parents):
                    events.append("wait")
            elif call.op.name == "tl.tileop.gemm":
                events.append("mad")
        assert events and events[0] == "wait"
        assert events.count("wait") >= 2
        after = transform.MaterializeMultiBuffer()(after)
        (owner,) = _l0c_allocations(after)
        assert tuple(map(int, owner.shape)) == (128, 256)
        after = transform.LowerScheduledTIR()(after)
        after = transform.AscendLowerTileOp()(after)
    (owner,) = _l0c_allocations(after)
    assert tuple(map(int, owner.shape)) == (16, 8, 16, 16)
    mad = calls(after, "tl.ascend_mad")
    assert [tuple(map(int, call.args[3:6])) for call in mad] == [(128, 16, 256), (128, 16, 64)]
    assert mad[0].args[0].args[1].same_as(mad[1].args[0].args[1])
    fix = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    analyzer = tvm.arith.Analyzer()
    assert [tuple(int(analyzer.simplify(value)) for value in call.args[3:7]) for call in fix] == [
        (256, 128, 256, 128), (64, 128, 64, 128)]
    assert fix[0].args[1].args[1].same_as(fix[1].args[1].args[1])


@pytest.mark.parametrize("kind", ["read-first", "first-epoch-only", "tail-write", "conditional",
                                  "overlapping-lifetime", "outside-owner", "different-pitch",
                                  "strides", "custom-layout", "escaped-pointer", "dynamic-owner",
                                  "scalar-read", "scalar-write", "opaque-address", "opaque-region",
                                  "explicit-versions", "manual-versions", "version-mode",
                                  "alias-contract", "padding-value", "conflict-hint",
                                  "manual-stage", "unit-flag", "prepared-versions"])
def test_unproved_lifetimes_keep_the_original_capacity_diagnostic(kind):
    before, _, _ = _fixture(kind=kind)
    with pytest.raises(tvm.error.InternalError, match="shared.l0c"):
        transform.AutoSchedule()(before)


def test_no_memory_pressure_keeps_independent_storage_and_versions():
    before, first, second = _fixture(width=128)
    after, _ = _scheduled(before)
    assert {buffer.data for buffer in _l0c_allocations(after)} == {first.data, second.data}


def test_3510_never_uses_the_2201_capacity_repair():
    before, first, second = _fixture(arch="dav-3510")
    try:
        after, _ = _scheduled(before)
    except tvm.error.InternalError as error:
        assert "shared.l0c" in str(error)
    else:
        assert {buffer.data for buffer in _l0c_allocations(after)} == {first.data, second.data}


def test_ordinary_dsl_complete_lowering_allocates_one_real_l0c_slot():
    @T.prim_func
    def program(a: T.Tensor((128, 16), "float16"), b: T.Tensor((256, 16), "float16"),
                first_out: T.Tensor((128, 256), "float32"), second_out: T.Tensor((128, 64), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((128, 16), "float16")
            b_l1 = T.alloc_l1((256, 16), "float16")
            first = T.alloc_l0c((128, 256), "float32")
            second = T.alloc_l0c((128, 64), "float32")
            for epoch in T.Pipelined(3):
                T.copy(a, a_l1)
                T.copy(b, b_l1)
                T.gemm(a_l1, b_l1, first, transpose_B=True, clear_accum=True)
                T.copy(first, first_out)
                T.gemm(a_l1, b_l1[0:64, :], second, transpose_B=True, clear_accum=True)
                T.copy(second, second_out)

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AutoSchedule", "tl.InsertSync", "tl.AscendLowerTileOp"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(program.with_attr("global_symbol", "main"),
                                target={"kind": "ascend", "arch": "dav-2201"},
                                enable_host_codegen=False, enable_device_compile=False)
    for name in snapshots:
        (owner,) = _l0c_allocations(snapshots[name])
        assert math.prod(map(int, owner.shape)) * 4 == 128 * 1024
    assert len(snapshots) == 3
    assert "asc_sync_wait(PIPE_FIX, PIPE_M," in result.kernel_source
