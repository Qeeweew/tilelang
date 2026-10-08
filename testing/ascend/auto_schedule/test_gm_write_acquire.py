"""Acquire a paired GM result at its first FIX, rather than phase BEGIN.

The assigned schedule has real unrelated Cube DMA preparation before two
owners' sliced exports. Its finite graph uses only pipe completion and actual
flags; the selected acquire never certifies the whole producer phase START.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_owner_input_completion import _epoch
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _name, _owner_sets, _region, _trace, _value,
)


_SET = "tl.ascend_cross_core_set_flag"
_WAIT = "tl.ascend_cross_core_wait_flag"
_COPY = "tl.tileop.ascend_copy"


def _program(*, epochs=3, first=0, distance=1, private=True, alias=False,
             complete=True, mismatched_activation=False):
    result = tirx.decl_buffer((1, 2, 2, 32, 32), "float32", name="Result")
    read_result = (tirx.decl_buffer(result.shape, result.dtype,
                                   name="ResultAlias", data=result.data)
                   if alias else result)
    source = tirx.decl_buffer((epochs, 32, 32), "float16", name="Prepare")
    output = tirx.decl_buffer((epochs, 2, 32, 32), "float32", name="Output")
    c = tirx.decl_buffer((16, 32), "float32", name="accum", scope="shared.l0c")
    l1 = tirx.decl_buffer((32, 32), "float16", name="prepare_l1", scope="shared.l1")
    value = tirx.decl_buffer((16, 32), "float32", name="value", scope="shared.dyn")
    e, phase, owner, part, sub, sid = [tirx.Var(name, "int32") for name in
                                     ("epoch", "cube_phase", "owner", "part", "sub", "sid")]
    active = tirx.And(first <= e, e < first + epochs)
    prepare = unit(copy(source, l1, src_indices=[e - first, 0, 0],
                        src_shape=[1, 32, 32]), core=2)
    publish = unit(copy(c, result,
                        dst_indices=[0, owner, e % 2, part * 16, 0],
                        dst_shape=[1, 1, 1, 16, 32]), core=2)
    pieces = unit(tirx.For(part, 0, 2 if complete else 1,
                          tirx.ForKind.SERIAL, publish), core=None)
    owners = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, pieces), core=None)
    # The one-trip phase is nonempty. Its old BEGIN precedes the preparation,
    # whereas the selected first actual GM write is owner0/part0 below it.
    producer = unit(tirx.For(phase, 0, 1, tirx.ForKind.SERIAL,
                             seq(prepare, owners)), core=None, guard=active)
    read = unit(copy(read_result, value,
                     src_indices=[0, sid, (e - distance) % 2, sub * 16, 0],
                     src_shape=[1, 1, 1, 16, 32]), core=1)
    export = unit(copy(value, output,
                       dst_indices=[e - first - distance, sid, sub * 16, 0],
                       dst_shape=[1, 1, 16, 32]), core=1)
    consumer_end = first + distance + epochs - int(mismatched_activation)
    consumer = unit(tirx.For(sub, 0, 2, tirx.ForKind.SERIAL, seq(read, export)),
                    core=None, guard=tirx.And(first + distance <= e, e < consumer_end))
    body = unit(tirx.For(e, 0, first + epochs + distance + 1,
                         tirx.ForKind.SERIAL, seq(producer, consumer)), core=None)
    annotations = {"vector_count": 2, "tl.manual_multi_buffer": {result.data: 2}}
    if private:
        annotations["tl.ascend_aiv_private_validated"] = [value.data]
    mod = kernel(body, buffers=[c, l1, value], params=[source, result, output],
                 annotations=annotations)
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    function = mod["main"].with_body(tirx.AttrStmt(
        thread, "thread_extent", 2, mod["main"].body))
    return tvm.IRModule({"main": function.with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})


def _acquire_stream(mod):
    acquisitions, prepares, writes = defaultdict(list), {}, defaultdict(list)
    for index, (call, env, loops) in enumerate(_trace(mod, core=2, lane=0)):
        epoch = _epoch(loops)
        if (_name(call) == _WAIT and call.args[1].value == "PIPE_FIX"
                and epoch is not None):
            acquisitions[epoch].append((index, _value(call.args[2], env), loops))
        elif _name(call) == _COPY:
            source, _, _ = _region(call.args[0], env)
            destination, origin, _ = _region(call.args[1], env)
            if source.name == "Prepare":
                assert epoch not in prepares
                prepares[epoch] = index
            elif destination.name == "Result":
                writes[epoch].append((index, origin[1], origin[-2]))
    return acquisitions, prepares, writes


@dataclass(frozen=True)
class _Event:
    core: int
    lane: int
    pipe: str
    kind: str
    epoch: int | None
    flag: tuple | None = None
    region: tuple | None = None


def _completion_graph(mod, *, first):
    """Real typed DMA completion, local binary flags and mode2 recipients."""
    trace = []
    owner_flags = {_value(call.args[2], env) for call, env, _ in _owner_sets(mod)}
    assert len(owner_flags) == 4
    owner_base = min(owner_flags)
    assert owner_flags == set(range(owner_base, owner_base + 4))
    for core, lane in ((2, 0), (1, 0), (1, 1)):
        for call, env, loops in _trace(mod, core=core, lane=lane):
            name, epoch = _name(call), _epoch(loops)
            if name in (_SET, _WAIT):
                assert int(call.args[0]) == 2
                flag = _value(call.args[2], env)
                assert 0 <= flag < 16
                if flag in owner_flags:
                    assert epoch is not None
                    if name == _SET:
                        owner = next(index for var, index, _ in loops if var.name == "owner")
                        assert core == 2 and flag == owner_base + owner * 2 + epoch % 2
                    else:
                        assert core == 1 and flag == owner_base + lane * 2 + (epoch - 1) % 2
                trace.append(_Event(core, lane, call.args[1].value.removeprefix("PIPE_"),
                                    "cross-set" if name == _SET else "cross-wait",
                                    epoch, (flag,)))
            elif name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                event = call.args[0].value
                source, destination = event.split("_")
                is_set = name == "tl.ascend_set_flag"
                assert 0 <= _value(call.args[1], env) < 8
                trace.append(_Event(core, lane, source if is_set else destination,
                                    "local-set" if is_set else "local-wait", epoch,
                                    (core, lane, event, _value(call.args[1], env))))
            elif name == "tl.ascend_pipe_barrier":
                trace.append(_Event(core, lane, call.args[0].value.removeprefix("PIPE_"),
                                    "barrier", epoch))
            elif name == _COPY:
                source, source_min, source_shape = _region(call.args[0], env)
                destination, destination_min, destination_shape = _region(call.args[1], env)
                kind, region = "copy", None
                if source.scope() == "shared.l0c":
                    assert core == 2 and destination.name == "Result"
                    pipe, kind = "FIX", "write"
                    assert destination_shape == (1, 1, 1, 16, 32)
                    region = (destination_min[1], destination_min[2], destination_min[-2], 16)
                elif source.scope() == "global":
                    pipe = "MTE2"
                    if source.name == "Prepare":
                        assert core == 2
                        kind = "prepare"
                    else:
                        assert source.name == "Result" and core == 1
                        assert source_shape == (1, 1, 1, 16, 32)
                        kind = "read"
                        region = (source_min[1], source_min[2], source_min[-2], 16)
                else:
                    assert source.scope() == "shared.dyn" and core == 1
                    assert destination.name == "Output"
                    pipe = "MTE3"
                trace.append(_Event(core, lane, pipe, kind, epoch, region=region))
            else:
                pytest.fail(f"Unexpected GM-write-acquire instruction: {name}")

    successors = [set() for _ in trace]
    last_pipe, sets, waits = {}, defaultdict(list), defaultdict(list)
    for index, node in enumerate(trace):
        key = node.core, node.lane, node.pipe
        if node.pipe == "ALL":
            for previous_key, previous in tuple(last_pipe.items()):
                if previous_key[:2] == key[:2]:
                    successors[previous].add(index)
                    last_pipe[previous_key] = index
        elif key in last_pipe:
            successors[last_pipe[key]].add(index)
        elif (node.core, node.lane, "ALL") in last_pipe:
            successors[last_pipe[node.core, node.lane, "ALL"]].add(index)
        last_pipe[key] = index
        if node.kind.startswith("local-"):
            (sets if node.kind == "local-set" else waits)[("local", *node.flag)].append(index)
        elif node.kind.startswith("cross-"):
            (sets if node.kind == "cross-set" else waits)[
                ("cross", node.core, node.lane, *node.flag)].append(index)

    strict_reuse, collective = [], []
    for key in sets.keys() | waits.keys():
        if key[0] != "local":
            continue
        assert len(sets[key]) == len(waits[key]), f"Unbalanced local flag: {key}"
        for notify, acquire in zip(sets[key], waits[key]):
            successors[notify].add(acquire)
        strict_reuse.extend(zip(waits[key], sets[key][1:]))
    cross_ids = {key[-1] for key in sets.keys() | waits.keys() if key[0] == "cross"}
    for flag in cross_ids:
        cube_sets, cube_waits = sets["cross", 2, 0, flag], waits["cross", 2, 0, flag]
        vector_sets = [sets["cross", 1, lane, flag] for lane in range(2)]
        vector_waits = [waits["cross", 1, lane, flag] for lane in range(2)]
        if cube_sets:
            assert flag in owner_flags and not cube_waits
            # Owner readiness consumes exactly the owning AIV's counter.
            # The unused recipient is not modeled as a FIFO or a completion
            # edge. This finite fixture sends at most three copies per ID.
            assert sum(map(len, vector_waits)) == len(cube_sets) <= 3
            assert sum(bool(acquires) for acquires in vector_waits) == 1
            for acquires in vector_waits:
                if acquires:
                    assert len(acquires) == len(cube_sets)
                    for notify, acquire in zip(cube_sets, acquires):
                        successors[notify].add(acquire)
                    strict_reuse.extend(zip(acquires, cube_sets[1:]))
        else:
            assert not any(vector_waits)
        if cube_waits:
            assert flag not in owner_flags
            for lane, notifications in enumerate(vector_sets):
                assert len(notifications) == len(cube_waits)
                assert trace[notifications[0]].epoch is None
                assert trace[cube_waits[-1]].epoch is None
                for notify, acquire in zip(notifications, cube_waits):
                    successors[notify].add(acquire)
                collective.append((flag, lane, notifications, cube_waits))
        else:
            assert not any(vector_sets)

    indegrees = [0] * len(trace)
    for outgoing in successors:
        for destination in outgoing:
            indegrees[destination] += 1
    ready = deque(index for index, count in enumerate(indegrees) if count == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for destination in successors[source]:
            indegrees[destination] -= 1
            if indegrees[destination] == 0:
                ready.append(destination)
    assert visited == len(trace), "GM-write-acquire completion graph deadlocks"

    def precedes(source, destination):
        frontier, seen = [source], set()
        while frontier:
            current = frontier.pop()
            if current == destination:
                return True
            if current not in seen:
                seen.add(current)
                frontier.extend(successors[current] - seen)
        return False

    assert all(precedes(acquire, notify) for acquire, notify in strict_reuse), \
        "Local/owner data flags require independent consume-before-reuse"
    assert collective
    body_epochs = sorted({node.epoch for node in trace if node.epoch is not None})
    writer_epochs = sorted({node.epoch for node in trace if node.kind == "write"})
    reader_epochs = sorted({node.epoch for node in trace if node.kind == "read"})
    assert body_epochs == list(range(body_epochs[-1] + 1))
    assert writer_epochs[0] == first and reader_epochs[0] == first + 1
    assert reader_epochs[-1] == writer_epochs[-1] + 1
    assert body_epochs[-1] == reader_epochs[-1] + 1
    empty_reader_prefix = [epoch for epoch in body_epochs if epoch < reader_epochs[0]]
    inactive_writer_suffix = [epoch for epoch in body_epochs if epoch > writer_epochs[-1]]
    for flag, lane, notifications, acquisitions in collective:
        # The existing counting protocol may accumulate only this fixture's
        # empty prefix/terminal contributions, then must completely drain.
        # CANN mode2 counters are four-bit; no binary assumption is imposed
        # on padding tokens, nor is arbitrary unconsumed publication allowed.
        assert len({trace[index].pipe for index in notifications}) == 1
        assert [trace[index].epoch for index in notifications] == [None, *body_epochs]
        assert [trace[index].epoch for index in acquisitions] == [*body_epochs, None]
        bound = max(ordinal + 1 - sum(precedes(acquire, notify) for acquire in acquisitions)
                    for ordinal, notify in enumerate(notifications))
        # Prefix: one root seed plus the epochs before the first Vector read.
        # Suffix: the last producer-active epoch's release plus two epochs
        # with no Cube writer. The first of those still reads the last result;
        # neither inactive Cube acquire has owner-data feedback forcing it
        # before the final empty Vector release. For T3/first0 the pending
        # contributions are e2/e3/e4, all consumed by the actual root drain.
        prefix_bound = 1 + len(empty_reader_prefix)
        suffix_bound = 1 + len(inactive_writer_suffix)
        assert (prefix_bound, suffix_bound) == (2 + first, 3)
        assert bound == max(prefix_bound, suffix_bound) < 16, (flag, lane, bound)
        assert all(precedes(notify, acquisitions[-1]) for notify in notifications)
    return trace, precedes, owner_flags


@pytest.mark.parametrize("epochs,first", [(3, 0), (5, 0), (3, 1)])
def test_first_fix_acquires_result_without_waiting_before_unrelated_preparation(epochs, first):
    before = _program(epochs=epochs, first=first)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json
    acquisitions, prepares, writes = _acquire_stream(after)
    ticks = range(first + epochs + 2)
    assert set(acquisitions) == set(ticks)
    assert all(len(events) == 1 for events in acquisitions.values())
    release_ids = {event[1] for events in acquisitions.values() for event in events}
    assert len(release_ids) == 1, "Keep the original distance1 paired WAR clock"
    active = range(first, first + epochs)
    assert set(prepares) == set(writes) == set(active)
    for epoch in ticks:
        acquire, _, loops = acquisitions[epoch][0]
        positions = {var.name: value for var, value, _ in loops}
        if epoch in writes:
            assert prepares[epoch] < acquire < min(index for index, _, _ in writes[epoch])
            assert positions["owner"] == positions["part"] == 0
            assert [(owner, row) for _, owner, row in writes[epoch]] == \
                [(owner, row) for owner in range(2) for row in (0, 16)]
        else:
            assert "owner" not in positions and "part" not in positions

    trace, precedes, owner_flags = _completion_graph(after, first=first)
    reads, exports, preparation, releases = defaultdict(list), defaultdict(list), {}, defaultdict(list)
    for index, node in enumerate(trace):
        if node.kind == "read":
            assert node.region[0] == node.lane
            assert node.region[1] == (node.epoch - 1) % 2
            reads[node.epoch].append(index)
        elif node.kind == "write":
            assert node.region[1] == node.epoch % 2
            exports[node.epoch].append(index)
        elif node.kind == "prepare":
            assert node.epoch not in preparation
            preparation[node.epoch] = index
        elif node.kind == "cross-set" and node.core == 1 and node.flag[0] in release_ids:
            releases[node.epoch].append(index)
    assert set(reads) == set(range(first + 1, first + epochs + 1))
    for epoch in active:
        assert len(exports[epoch]) == len(reads[epoch + 1]) == 4
        for owner in range(2):
            owner_writes = [index for index in exports[epoch] if trace[index].region[0] == owner]
            owner_reads = [index for index in reads[epoch + 1] if trace[index].region[0] == owner]
            assert all(precedes(write, read) for write in owner_writes for read in owner_reads)
        if epoch >= first + 2:
            old_reads = reads[epoch - 1]
            assert all(precedes(read, write) for read in old_reads for write in exports[epoch])
            assert len(releases[epoch - 1]) == 2
            assert {trace[index].lane for index in releases[epoch - 1]} == {0, 1}
            assert all(not precedes(release, preparation[epoch])
                       for release in releases[epoch - 1]), \
                "The late acquire cannot certify or block unrelated phase preparation"
    # No initial/final owner data-ready token; the collective WAR alone keeps
    # its root seed/drain and inactive complementary contributions.
    assert sum(node.kind == "cross-set" and node.core == 2 and node.flag[0] in owner_flags
               for node in trace) == 2 * epochs
    for lane in range(2):
        selected = [node for node in trace if node.kind == "cross-set" and node.core == 1
                    and node.lane == lane and node.flag[0] in release_ids]
        assert len(selected) == len(ticks) + 1
        assert sum(node.epoch is None for node in selected) == 1


@pytest.mark.parametrize("kwargs", [
    {"alias": True}, {"complete": False}, {"mismatched_activation": True},
    {"private": False}, {"distance": 0},
], ids=["distinct-buffer-alias", "incomplete-owner", "different-activation",
        "no-private-proof", "same-epoch-owner-result-not-supported"])
def test_unproven_output_retains_original_wide_acquire(kwargs):
    before = _program(**kwargs)
    before_json = tvm.ir.save_json(before)
    after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == before_json
    assert not _owner_sets(after)
    acquisitions, prepares, _ = _acquire_stream(after)
    assert acquisitions and prepares
    for epoch, prepare in prepares.items():
        assert len(acquisitions[epoch]) == 1
        assert acquisitions[epoch][0][0] < prepare
