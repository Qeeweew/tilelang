"""Native loop-plan validation checks concrete FFI results before mutation.

These host-only pass inputs inject plans at the existing Python adapter. Costs
are diagnostic weights; no solver timing or device behavior is being tested.
"""

from inspect import signature

import pytest

pytest.importorskip("z3")
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


_INVALID = "Invalid Ascend loop schedule: "


def _target(mod):
    mod["main"] = mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod


def _owner(epoch, body, *, eligible=(), num_stages=2, enable_offset=True,
           auto_pipeline=True):
    annotations = {"num_stages": num_stages, "enable_offset": enable_offset,
                   "multi_buffer_eligible": list(eligible)}
    if auto_pipeline:
        annotations["tl.ascend_auto_pipeline"] = True
    return unit(tirx.For(
        epoch, 0, 4, tirx.ForKind.SERIAL, body,
        annotations=annotations,
    ), core=None, stage=-1)


def _ring(*, tile=32):
    source = tirx.decl_buffer((4, tile), "float32", name="source")
    output = tirx.decl_buffer((4, tile), "float32", name="output")
    temp = tirx.decl_buffer((tile,), "float32", name="temp", scope="shared.dyn")
    epoch = tirx.Var("epoch", "int32")
    body = seq(
        unit(copy(source, temp, src_indices=[epoch, 0], src_shape=[1, tile]),
             core=1, stage=-1, cost=(2, 1)),
        unit(copy(temp, output, dst_indices=[epoch, 0], dst_shape=[1, tile]),
             core=1, stage=-1, cost=(1, 1)),
    )
    return _target(kernel(_owner(epoch, body, eligible=[temp.data]),
                          buffers=[temp], params=[source, output]))


def _two_loads(*, stages=(-1, -1)):
    source = tirx.decl_buffer((4, 32), "float32", name="source")
    temps = [tirx.decl_buffer((32,), "float32", name=f"temp{i}", scope="shared.dyn")
             for i in range(2)]
    epoch = tirx.Var("epoch", "int32")
    body = seq(*(unit(copy(source, temp, src_indices=[epoch, 0], src_shape=[1, 32]),
                      core=1, stage=stage, cost=(2, 2))
                 for temp, stage in zip(temps, stages)))
    return _target(kernel(_owner(epoch, body, num_stages=4),
                          buffers=temps, params=[source]))


def _inject(monkeypatch, plan, *, inspect=None):
    observed = []

    def witness(*args, **kwargs):
        observed.append((args, kwargs))
        if inspect is not None:
            inspect(args, kwargs)
        return plan

    # The registered adapter resolves this function dynamically even after
    # native AutoSchedule has cached the FFI callback in an earlier test.
    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", witness)
    return observed


def _assert_ring_inputs(args, kwargs):
    assert len(args[1]) == 2 and args[8] and len(args[8]) == 1
    assert kwargs["buffer_min_versions"] == [1]
    assert any(tuple(edge[:3]) == (0, 1, 0) for edge in args[4])
    assert any(tuple(edge[:3]) == (1, 0, -1) for edge in args[4])


@pytest.mark.parametrize("versions,valid", [(1, False), (2, True)])
def test_native_data_recurrence_uses_actual_returned_versions(monkeypatch, versions, valid):
    observed = _inject(monkeypatch, ([0, 5], [versions], 3), inspect=_assert_ring_inputs)
    # The reverse edge needs two actual slots; a slack internal solver count
    # must not legitimize a returned single-slot allocation.
    if valid:
        after = transform.AutoSchedule()(_ring())
        root = after["main"].body.block
        temp = next(buffer for buffer in root.alloc_buffers if buffer.name == "temp")
        assert int(root.annotations["tl.buffer_versions_map"][temp.data]) == 2
    else:
        with pytest.raises(tvm.error.InternalError, match=_INVALID + "data-dependency"):
            transform.AutoSchedule()(_ring())
    assert len(observed) == 1


def test_native_returned_versions_must_fit_residual_capacity(monkeypatch):
    observed = _inject(monkeypatch, ([0, 2], [2], 3), inspect=_assert_ring_inputs)
    # The minimum one-slot 128 KiB UB allocation fits 2201. The malicious
    # two-slot result passes data constraints but exceeds its physical bank.
    with pytest.raises(tvm.error.InternalError, match=_INVALID + "memory-capacity"):
        transform.AutoSchedule()(_ring(tile=32768))
    assert len(observed) == 1


@pytest.mark.parametrize("starts,valid", [([0, 3], False), ([0, 2], True)])
def test_native_resource_pair_checks_forward_and_wrap(monkeypatch, starts, valid):
    def inspect(args, kwargs):
        assert args[2] == [2, 2]
        assert (0, 1) in map(tuple, args[5])
        assert args[8] == []

    observed = _inject(monkeypatch, (starts, [], 4), inspect=inspect)
    # At phase 3 the forward gap is sufficient, but only one unit remains
    # before phase 0 of the next iteration. Phase 2 makes both arcs touch.
    if valid:
        transform.AutoSchedule()(_two_loads())
    else:
        with pytest.raises(tvm.error.InternalError, match=_INVALID + "resource-wrap"):
            transform.AutoSchedule()(_two_loads())
    assert len(observed) == 1


def test_native_manual_pipe_order_cannot_be_reversed_by_ffi(monkeypatch):
    def inspect(args, kwargs):
        assert kwargs["manual_schedule"]
        assert (0, 1) in map(tuple, args[7])

    observed = _inject(monkeypatch, ([2, 0], [], 4), inspect=inspect)
    # Both resource arcs are legal. The separately supplied manual pipe
    # order still requires the first source task to precede the second.
    with pytest.raises(tvm.error.InternalError, match=_INVALID + "pipe-order"):
        transform.AutoSchedule()(_two_loads(stages=(0, 0)))
    assert len(observed) == 1


@pytest.mark.parametrize("requested,starts,valid", [
    pytest.param((2, 2), [0, 2], True, id="all-stage-two-common-origin"),
    pytest.param((2, 3), [0, 6], True, id="different-stages-common-origin"),
    pytest.param((2, 3), [0, 2], False, id="inconsistent-stage-origin"),
])
def test_native_partial_manual_stages_allow_only_one_common_translation(
    monkeypatch, requested, starts, valid
):
    def inspect(args, kwargs):
        assert kwargs["manual_schedule"]
        assert kwargs["manual_stages"] == list(requested)

    observed = _inject(monkeypatch, (starts, [], 4), inspect=inspect)
    if valid:
        after = transform.AutoSchedule()(_two_loads(stages=requested))
        epoch, = [loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "epoch"]
        units = [node for node in nodes(epoch.body, tirx.AttrStmt)
                 if node.attr_key == "tl.schedule_unit"]
        assert [int(node.node["stage"]) for node in units] == list(requested)
    else:
        with pytest.raises(tvm.error.InternalError, match=_INVALID + "manual-stage"):
            transform.AutoSchedule()(_two_loads(stages=requested))
    assert len(observed) == 1


@pytest.mark.parametrize("plan,reason", [
    pytest.param(([0, 2], [1], 0), "invalid-ii", id="nonpositive-II"),
    pytest.param(([0], [1], 3), "invalid-task-count", id="missing-task-start"),
    pytest.param(([0, 2], [], 3), "invalid-storage-count", id="missing-storage-count"),
    pytest.param(([-1, 2], [1], 3), "negative-start", id="negative-start"),
    pytest.param(([0, 2], [3], 3), "version-bound", id="version-above-owner-bound"),
])
def test_native_malformed_concrete_loop_results_are_rejected(monkeypatch, plan, reason):
    observed = _inject(monkeypatch, plan, inspect=_assert_ring_inputs)
    with pytest.raises(tvm.error.InternalError, match=_INVALID + reason):
        transform.AutoSchedule()(_ring())
    assert len(observed) == 1


def test_native_let_clone_stage_order_is_checked_before_reordering(monkeypatch):
    output = tirx.decl_buffer((4,), "float32", name="output")
    state = tirx.decl_buffer((1,), "float32", name="state", scope="shared.dyn")
    value, epoch = tirx.Var("value", "float32"), tirx.Var("epoch", "int32")
    body = seq(
        unit(tirx.Bind(value, state[0]), core=3, stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(output, value, [epoch]), core=3, stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(state, tirx.FloatImm("float32", 1), [0]),
             core=3, stage=-1, cost=(1, 1)),
    )
    before = _target(kernel(_owner(epoch, body, auto_pipeline=False),
                            buffers=[state], params=[output],
                            annotations={"tl.buffer_versions_map": {state.data: 2}}))

    def inspect(args, kwargs):
        assert kwargs["stage_order_deps"] == [(1, 2)]
        assert args[8] == []

    observed = _inject(monkeypatch, ([0, 6, 2], [], 4), inspect=inspect)
    with pytest.raises(tvm.error.InternalError, match=_INVALID + "let-stage-order"):
        transform.AutoSchedule()(before)
    assert len(observed) == 1


def test_native_no_offset_window_is_per_resource_bank(monkeypatch):
    source, spare, temp, extra = [
        tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn")
        for name in ("source", "spare", "temp", "extra")]
    outputs = [tirx.decl_buffer((4, 32), "float32", name=f"output{i}") for i in range(2)]
    epoch = tirx.Var("epoch", "int32")

    def vector(dst):
        def ptr(buffer, effect):
            return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"),
                             [buffer[0], 32, effect])
        return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.exp"),
                                      [ptr(dst, 2), ptr(source, 1), tirx.IntImm("uint32", 32)]))

    body = seq(
        unit(vector(temp), core=1, stage=-1, cost=(4, 1)),
        unit(vector(extra), core=1, stage=-1, cost=(1, 1)),
        unit(copy(temp, outputs[0], dst_indices=[epoch, 0], dst_shape=[1, 32]),
             core=1, stage=-1, cost=(1, 1)),
        unit(copy(spare, outputs[1], dst_indices=[epoch, 0], dst_shape=[1, 32]),
             core=1, stage=-1, cost=(1, 1)),
    )
    before = _target(kernel(_owner(epoch, body, eligible=[temp.data], num_stages=3,
                                   enable_offset=False, auto_pipeline=False),
                            buffers=[source, spare, temp, extra], params=outputs))

    def inspect(args, kwargs):
        assert not kwargs["enable_offset"]
        assert set(map(tuple, args[5])) == {(0, 1), (2, 3)}
        assert (0, 2, 0, 4) in map(tuple, args[4])
        assert kwargs["buffer_min_versions"] == [1]

    observed = _inject(monkeypatch, ([0, 1, 4, 5], [3], 2), inspect=inspect)
    after = transform.AutoSchedule()(before)
    epoch, = [loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "epoch"]
    units = [node for node in nodes(epoch.body, tirx.AttrStmt)
             if node.attr_key == "tl.schedule_unit"]
    assert len(units) == 4 and all(int(node.node["stage"]) == 0 for node in units)
    assert len(observed) == 1


def test_native_existing_shared_minimum_may_exceed_later_owner_stage_bound(monkeypatch):
    source = tirx.decl_buffer((8, 32), "float32", name="source")
    output = tirx.decl_buffer((8, 32), "float32", name="output")
    temp = tirx.decl_buffer((32,), "float32", name="temp", scope="shared.dyn")
    owners = []
    for index, bound in enumerate((3, 1)):
        epoch = tirx.Var(f"owner{index}", "int32")
        row = epoch + index * 4
        body = seq(
            unit(copy(source, temp, src_indices=[row, 0], src_shape=[1, 32]),
                 core=1, stage=-1, cost=(2, 1)),
            unit(copy(temp, output, dst_indices=[row, 0], dst_shape=[1, 32]),
                 core=1, stage=-1, cost=(1, 1)),
        )
        owners.append(_owner(epoch, body, eligible=[temp.data], num_stages=bound,
                             auto_pipeline=False))
    before = _target(kernel(seq(*owners), buffers=[temp], params=[source, output]))
    minima = []

    def witness(*args, **kwargs):
        minima.append((args[0], kwargs["buffer_min_versions"]))
        return [0, 2], [3], 3

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", witness)
    after = transform.AutoSchedule()(before)
    assert minima == [(3, [1]), (1, [3])]
    assert int(after["main"].body.block.annotations["tl.buffer_versions_map"][temp.data]) == 3


def test_native_child_profile_checks_physical_order_after_legal_data_translation(monkeypatch):
    from testing.ascend.auto_schedule.test_hierarchical_resource_reservations import _native_fixture

    before, _, _ = _native_fixture("eligible")
    original = z3_scheduler.z3_schedule_loop_python
    observed = []

    def witness(*args, **kwargs):
        inputs = dict(signature(original).bind(*args, **kwargs).arguments)
        rows = inputs.get("resource_reservations")
        if not rows:
            # The real child solver must establish its versions and resource
            # summary before the outer malicious result is validated.
            return original(*args, **kwargs)
        observed.append(inputs)
        assert len(inputs["latencies"]) == 2
        assert inputs["buffer_min_versions"] == [1]
        assert inputs["enable_offset"] and inputs["stage_horizon"] >= 2
        assert not inputs["owner_exclusion_deps"]
        assert not inputs["pipe_order_deps"] and not inputs["stage_order_deps"]
        assert (0, 1) in map(tuple, inputs["resource_deps"])
        a_mte2, = [row for row in rows if tuple(row[:3]) == (0, 0, 2)]
        child_mte2, = [row for row in rows if tuple(row[:3]) == (1, 0, 2)]
        assert child_mte2[3] + child_mte2[4] > a_mte2[3]

        # A sufficiently long period and two actual A slots satisfy both
        # A-ready -> child and child-last-read -> next A-write. The two task
        # residues nevertheless coincide: emitted descending-stage order
        # puts the whole child block before A, whose MTE2 issue starts before
        # the child's last MTE2 issue. A completion/data check cannot certify
        # that atomic physical FIFO order.
        ii = 1 + max([*inputs["latencies"], *inputs["iis"],
                      *(edge[3] for edge in inputs["data_deps"]),
                      *(row[3] + row[4] for row in rows)])
        starts, versions = [0, ii], [2]
        for source, destination, distance, latency in inputs["data_deps"]:
            distance = distance if distance >= 0 else versions[-distance - 1]
            assert starts[destination] - starts[source] + ii * distance >= latency
        for capacity, *members in inputs["memory_groups"]:
            assert sum(inputs["buffer_sizes"][member] * versions[member]
                       for member in members) <= capacity
        return starts, versions, ii

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", witness)
    with pytest.raises(tvm.error.InternalError, match=_INVALID + "hierarchical-resource-order"):
        transform.AutoSchedule()(before)
    assert len(observed) == 1
