"""Paired GM-input release at the last typed read, not the phase END.

This assigned schedule needs neither AutoSchedule nor a performance model.
The shared DMA completion oracle retains both physical AIV recipients, actual
local flags, finite seed/drain generations and consume-before-reuse checks.
The unrelated AIC tail is real DMA work; it must not become a prerequisite
for overwriting P, or be certified complete by the early P notification.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_owner_input_completion import (
    _completion_graph, _epoch, _input_flags,
)
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _name, _region, _trace, _value,
)


_SET = "tl.ascend_cross_core_set_flag"
_WAIT = "tl.ascend_cross_core_wait_flag"
_COPY = "tl.tileop.ascend_copy"


def _program(*, epochs=3, first=0, private=True, alias=False,
             mismatched_activation=False, row_guard=False):
    source = tirx.decl_buffer((epochs, 2, 128, 32), "float16", name="Input")
    tail = tirx.decl_buffer((epochs, 2, 32, 32), "float16", name="Tail")
    probability = tirx.decl_buffer((1, 2, 2, 128, 32), "float16", name="P")
    read_p = (tirx.decl_buffer(probability.shape, probability.dtype,
                              name="PAlias", data=probability.data)
              if alias else probability)
    value = tirx.decl_buffer((16, 32), "float16", name="value", scope="shared.dyn")
    l1 = tirx.decl_buffer((32, 32), "float16", name="p_l1", scope="shared.l1")
    tail_l1 = tirx.decl_buffer((32, 32), "float16", name="tail_l1", scope="shared.l1")
    e, sub, owner, chunk, sid = [tirx.Var(name, "int32") for name in
                                ("epoch", "sub", "owner", "chunk", "sid")]
    active = tirx.And(first <= e, e < first + epochs)
    fill = unit(copy(source, value, src_indices=[e - first, sid, sub * 16, 0],
                     src_shape=[1, 1, 16, 32]), core=1)
    publish = copy(value, probability, dst_indices=[0, sid, e % 2, sub * 16, 0],
                   dst_shape=[1, 1, 1, 16, 32])
    if row_guard:
        publish = tirx.IfThenElse(sub < 7, publish, None)
    producer = unit(tirx.For(sub, 0, 8, tirx.ForKind.SERIAL,
                             seq(fill, unit(publish, core=1))),
                    core=None, guard=active)
    read = unit(copy(read_p, l1,
                     src_indices=[0, owner, e % 2, chunk * 32, 0],
                     src_shape=[1, 1, 1, 32, 32]), core=2)
    read_loop = unit(tirx.For(chunk, 0, 4, tirx.ForKind.SERIAL, read), core=None)
    # Same MTE2 queue, but another storage and no relation to the P slot.
    # The last P read of owner 1 is the exact release cut; this copy follows it.
    unrelated = unit(copy(tail, tail_l1,
                          src_indices=[e - first, owner, 0, 0],
                          src_shape=[1, 1, 32, 32]), core=2)
    consumer_active = (tirx.And(first <= e, e < first + epochs - 1)
                       if mismatched_activation else active)
    consumer = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL,
                            seq(read_loop, unrelated)),
                    core=None, guard=consumer_active)
    body = unit(tirx.For(e, 0, first + epochs + 1, tirx.ForKind.SERIAL,
                         seq(producer, consumer)), core=None)
    annotations = {"vector_count": 2,
                   "tl.manual_multi_buffer": {probability.data: 2}}
    if private:
        annotations["tl.ascend_aiv_private_validated"] = [value.data]
    mod = kernel(body, buffers=[value, l1, tail_l1],
                 params=[source, tail, probability], annotations=annotations)
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    func = mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))
    return tvm.IRModule({"main": func.with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _release_stream(mod):
    """Identify the paired release by its actual direction and copy storage."""
    stream = list(_trace(mod, core=2, lane=0))
    releases = defaultdict(list)
    reads = defaultdict(list)
    tails = defaultdict(list)
    for index, (call, env, loops) in enumerate(stream):
        epoch = _epoch(loops)
        if (_name(call) == _SET and call.args[1].value == "PIPE_MTE2"
                and epoch is not None):
            releases[epoch].append((index, _value(call.args[2], env), loops))
        elif _name(call) == _COPY:
            source, origin, _ = _region(call.args[0], env)
            if source.name in ("P", "PAlias"):
                reads[epoch].append((index, origin[1]))
            elif source.name == "Tail":
                tails[epoch].append((index, origin[1]))
    return stream, releases, reads, tails


@pytest.mark.parametrize("epochs,first", [(3, 0), (5, 0), (3, 1)])
def test_paired_release_moves_only_last_gm_read_and_keeps_finite_generations(epochs, first):
    before = _program(epochs=epochs, first=first)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json
    stream, releases, reads, tails = _release_stream(after)
    ticks = range(first + epochs + 1)
    assert set(releases) == set(ticks)
    assert all(len(events) == 1 for events in releases.values())
    assert len({event[1] for events in releases.values() for event in events}) == 2
    assert set(reads) == set(tails) == set(range(first, first + epochs))
    for epoch in ticks:
        release_index, _, loops = releases[epoch][0]
        positions = {var.name: value for var, value, _ in loops}
        if epoch in reads:
            assert len(reads[epoch]) == 8 and len(tails[epoch]) == 2
            assert release_index > max(index for index, _ in reads[epoch])
            assert release_index < next(index for index, owner in tails[epoch] if owner == 1)
            assert positions["owner"] == 1 and positions["chunk"] == 3
        else:
            assert "owner" not in positions and "chunk" not in positions

    owner_flags = _input_flags(after)
    assert len(owner_flags) == 4, "Read-release certificate must retain owner-input readiness"
    trace, precedes = _completion_graph(after)
    all_reads, writes, notifications, tail_nodes = defaultdict(list), defaultdict(list), {}, defaultdict(list)
    for index, node in enumerate(trace):
        if node.kind == "read":
            all_reads[node.epoch].append(index)
        elif node.kind == "write":
            writes[node.epoch].append(index)
        elif (node.kind == "cross-set" and node.core == 2 and node.pipe == "MTE2"
              and node.epoch is not None):
            assert node.epoch not in notifications
            notifications[node.epoch] = index
        elif node.kind == "copy" and node.core == 2:
            tail_nodes[node.epoch].append(index)
    for epoch in range(first, first + epochs):
        notify = notifications[epoch]
        assert all(precedes(read, notify) for read in all_reads[epoch])
        # The early release does not certify the unrelated phase tail END.
        assert len(tail_nodes[epoch]) == 2
        assert not precedes(tail_nodes[epoch][-1], notify)
        if epoch + 2 < first + epochs:
            assert all(precedes(read, write) for read in all_reads[epoch]
                       for write in writes[epoch + 2])
    # Both AIVs consume the unchanged paired clock, including fill and drain.
    release_ids = {event[1] for events in releases.values() for event in events}
    for lane in range(2):
        waits = [(call, env, loops) for call, env, loops in _trace(after, core=1, lane=lane)
                 if _name(call) == _WAIT and call.args[1].value == "PIPE_MTE3"
                 and _value(call.args[2], env) in release_ids]
        assert len(waits) == len(ticks) + 2
        assert sum(_epoch(loops) is None for _, _, loops in waits) == 2


@pytest.mark.parametrize("kwargs", [
    {"private": False}, {"alias": True}, {"mismatched_activation": True}, {"row_guard": True},
], ids=["no-private-proof", "distinct-buffer-view", "different-activation", "incomplete-owner"])
def test_unproven_input_retains_wide_paired_release(kwargs):
    before = _program(**kwargs)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json
    assert not _input_flags(after)
    _, releases, _, tails = _release_stream(after)
    assert tails and releases
    for epoch, copies in tails.items():
        assert len(releases[epoch]) == 1
        assert releases[epoch][0][0] > max(index for index, _ in copies)
