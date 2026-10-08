"""Finite owner/slot V-to-C readiness from ordinary private-UB copies.

Each dav-2201 mode2 input event still has two physical AIV participants.
The non-owner contributes after the ordinary old-input release, while the
owner contributes after all of its GM stores. This CPU oracle models DMA
completion in each pipe and verifies consumption before counter reuse; it
does not turn an unconsumed notification into a counting FIFO.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _name, _region, _trace, _uses_var, _value,
)


_SET = "tl.ascend_cross_core_set_flag"
_WAIT = "tl.ascend_cross_core_wait_flag"
_COPY = "tl.tileop.ascend_copy"


def _scheduled_input(*, epochs=3, private=True, arch="dav-2201",
                     write_subs=8, read_subs=4, alias=False,
                     consumer_epochs=None, reserved_flags=0,
                     unstable_activation=False):
    """A complete 128-row owner is distributed over eight private UB stores.

    This is an assigned, prepared schedule input: stages are fixed, GM has an
    explicit two-slot axis, and no solver or layout inference builds the
    protocol that the test then checks.
    """
    consumer_epochs = epochs if consumer_epochs is None else consumer_epochs
    source = tirx.decl_buffer((epochs, 2, 128, 32), "float16", name="Input")
    probability = tirx.decl_buffer((1, 2, 2, 128, 32), "float16", name="P")
    read_probability = (tirx.decl_buffer(probability.shape, probability.dtype,
                                        name="PAlias", data=probability.data)
                        if alias else probability)
    value = tirx.decl_buffer((16, 32), "float16", name="value", scope="shared.dyn")
    l1 = tirx.decl_buffer((32, 32), "float16", name="p_l1", scope="shared.l1")
    e, sub, owner, chunk, sid = [tirx.Var(name, "int32") for name in
                                ("epoch", "sub", "owner", "chunk", "sid")]
    fill = unit(copy(source, value, src_indices=[e, sid, sub * 16, 0],
                     src_shape=[1, 1, 16, 32]), core=1)
    publish = copy(value, probability, dst_indices=[0, sid, e % 2, sub * 16, 0],
                   dst_shape=[1, 1, 1, 16, 32])
    if unstable_activation:
        # An activation depending on the distributed row cannot be moved to
        # a single epoch/owner publication cut.
        publish = tirx.IfThenElse(sub < 7, publish, None)
    producer = unit(tirx.For(sub, 0, write_subs, tirx.ForKind.SERIAL,
                             seq(fill, unit(publish, core=1))),
                    core=None, guard=e < epochs)
    read = unit(copy(read_probability, l1,
                     src_indices=[0, owner, e % 2, chunk * 32, 0],
                     src_shape=[1, 1, 1, 32, 32]), core=2)
    consumer = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL,
                            unit(tirx.For(chunk, 0, read_subs, tirx.ForKind.SERIAL, read),
                                 core=None)), core=None, guard=e < consumer_epochs)
    body = unit(tirx.For(e, 0, epochs + 1, tirx.ForKind.SERIAL,
                         seq(producer, consumer)), core=None)
    if reserved_flags:
        reservations = []
        for flag in range(reserved_flags):
            reservations.extend([
                unit(tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get(_SET),
                                             [2, "PIPE_FIX", flag])), core=2),
                unit(tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get(_WAIT),
                                             [2, "PIPE_MTE2", flag])), core=1),
            ])
        body = seq(*reservations, body)
    annotations = {"vector_count": 2,
                   "tl.manual_multi_buffer": {probability.data: 2}}
    if private:
        annotations["tl.ascend_aiv_private_validated"] = [value.data]
    mod = kernel(body, buffers=[value, l1], params=[source, probability], annotations=annotations)
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    func = mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))
    return tvm.IRModule({"main": func.with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _input_flags(mod):
    flags = set()
    for lane in range(2):
        for call, env, _ in _trace(mod, core=1, lane=lane):
            if (_name(call) == _SET and call.args[1].value == "PIPE_MTE3"
                    and any(var.name == "sid" and _uses_var(call.args[2], var)
                            for var in env if isinstance(var, tirx.Var))):
                flags.add(_value(call.args[2], env))
    return flags


def _epoch(loops):
    return next((index for var, index, _ in loops if var.name == "epoch"), None)


@dataclass(frozen=True)
class _Completion:
    core: int
    lane: int
    pipe: str
    kind: str
    flag: tuple | None = None
    epoch: int | None = None
    region: tuple | None = None


def _completion_graph(mod):
    """Use actual pipe completion, never scalar issue order across pipes."""
    trace = []
    for core, lane in ((2, 0), (1, 0), (1, 1)):
        for call, env, loops in _trace(mod, core=core, lane=lane):
            name, epoch = _name(call), _epoch(loops)
            if name in (_SET, _WAIT):
                assert int(call.args[0]) == 2
                trace.append(_Completion(core, lane, call.args[1].value.removeprefix("PIPE_"),
                                         "cross-set" if name == _SET else "cross-wait",
                                         (_value(call.args[2], env),), epoch))
            elif name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                event = call.args[0].value
                source, destination = event.split("_")
                is_set = name == "tl.ascend_set_flag"
                trace.append(_Completion(core, lane, source if is_set else destination,
                                         "local-set" if is_set else "local-wait",
                                         (core, lane, event, _value(call.args[1], env)), epoch))
            elif name == "tl.ascend_pipe_barrier":
                trace.append(_Completion(core, lane, call.args[0].value.removeprefix("PIPE_"),
                                         "barrier", epoch=epoch))
            elif name == _COPY:
                source, source_min, shape = _region(call.args[0], env)
                dest, dest_min, _ = _region(call.args[1], env)
                pipe = "MTE2" if source.scope() == "global" else "MTE3"
                kind, region = "copy", None
                if dest.name == "P":
                    kind, region = "write", (*dest_min[1:3], dest_min[-2], shape[-2])
                elif source.name == "P":
                    kind, region = "read", (*source_min[1:3], source_min[-2], shape[-2])
                trace.append(_Completion(core, lane, pipe, kind, epoch=epoch, region=region))
            else:
                pytest.fail(f"Unexpected owner input instruction: {name}")

    successors = [set() for _ in trace]
    last_pipe, sets, waits = {}, defaultdict(list), defaultdict(list)
    for index, node in enumerate(trace):
        key = node.core, node.lane, node.pipe
        if node.pipe == "ALL":
            for pipe, previous in tuple(last_pipe.items()):
                if pipe[:2] == key[:2]:
                    successors[previous].add(index)
                    last_pipe[pipe] = index
        elif key in last_pipe:
            successors[last_pipe[key]].add(index)
        elif (node.core, node.lane, "ALL") in last_pipe:
            successors[last_pipe[node.core, node.lane, "ALL"]].add(index)
        last_pipe[key] = index
        if node.kind in ("local-set", "local-wait"):
            (sets if node.kind == "local-set" else waits)[("local", *node.flag)].append(index)
        elif node.kind in ("cross-set", "cross-wait"):
            (sets if node.kind == "cross-set" else waits)[("cross", node.core, node.lane, *node.flag)].append(index)

    local_flags = {key for key in sets.keys() | waits.keys() if key[0] == "local"}
    reuse = []
    for key in local_flags:
        assert len(sets[key]) == len(waits[key]), f"Unbalanced local flag {key}"
        for notify, acquire in zip(sets[key], waits[key]):
            successors[notify].add(acquire)
        reuse.extend(zip(waits[key], sets[key][1:]))
    cross_ids = {key[-1] for key in sets.keys() | waits.keys() if key[0] == "cross"}
    broadcasts = {}
    for flag in cross_ids:
        cube_sets, cube_waits = sets[("cross", 2, 0, flag)], waits[("cross", 2, 0, flag)]
        vector_sets = [sets[("cross", 1, lane, flag)] for lane in range(2)]
        vector_waits = [waits[("cross", 1, lane, flag)] for lane in range(2)]
        # Counter recipients differ by direction. An ordinary allocation may
        # use the same numeric ID at AIC and AIV without making them one
        # shared state cell.
        if cube_sets:
            assert all(len(acquires) == len(cube_sets) for acquires in vector_waits)
            for acquires in vector_waits:
                for notify, acquire in zip(cube_sets, acquires):
                    successors[notify].add(acquire)
                reuse.extend(zip(acquires, cube_sets[1:]))
            for lane, acquires in enumerate(vector_waits):
                broadcasts[flag, lane] = cube_sets, acquires
        else:
            assert not any(vector_waits)
        if cube_waits:
            assert all(len(notifications) == len(cube_waits) for notifications in vector_sets)
            for notifications in vector_sets:
                for notify, acquire in zip(notifications, cube_waits):
                    successors[notify].add(acquire)
                reuse.extend(zip(cube_waits, notifications[1:]))
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
    assert visited == len(trace), "Owner input completion graph deadlocks"

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

    # Matching finite generations above contributes only notify->wait edges.
    # Local binary flags and every active owner/slot counter need independent
    # consume-before-reuse. The ordinary paired release can additionally
    # publish a final counting notification in an inactive iteration, then
    # drain it at the root. One initial empty epoch can likewise publish an
    # ordinary release before the first seed has been consumed. These are
    # narrowly bounded mode2 padding counts, not another data generation.
    owner_flags = _input_flags(mod)
    first_epoch = min(node.epoch for node in trace if node.epoch is not None)
    last_epoch = max(node.epoch for node in trace if node.epoch is not None)
    memory_epochs = {node.epoch for node in trace if node.kind in ("copy", "read", "write")}
    bounded_counters = set()
    for consumed, reset in reuse:
        if precedes(consumed, reset):
            continue
        old, new = trace[consumed], trace[reset]
        if (old.kind == "cross-wait" and old.core == 1 and
                new.kind == "cross-set" and new.core == 2 and
                old.pipe == "MTE3" and new.pipe == "MTE2" and
                old.flag == new.flag and new.flag[0] not in owner_flags and
                old.epoch == new.epoch == first_epoch and
                first_epoch not in memory_epochs and memory_epochs and
                min(memory_epochs) == first_epoch + 1):
            notifications, acquisitions = broadcasts[new.flag[0], old.lane]
            assert reset == notifications[1] and consumed == acquisitions[0]
            assert trace[notifications[0]].epoch is None, "Prefix must follow a root seed"
            assert trace[acquisitions[-1]].epoch is None, "Prefix count must reach the root drain"
            assert not any(node.kind == "cross-set" and node.flag[0] in owner_flags
                           and node.epoch == first_epoch for node in trace)
            bounded_counters.add((new.flag[0], old.lane))
            continue
        assert (old.kind == "cross-wait" and old.core == 1 and
                new.kind == "cross-set" and new.core == 2 and
                old.flag == new.flag and new.flag[0] not in owner_flags and
                old.epoch == new.epoch == last_epoch and
                last_epoch not in memory_epochs), \
            f"A data/local flag may be reused before consumption: {old} -> {new}"
        notifications, acquisitions = broadcasts[new.flag[0], old.lane]
        assert reset == notifications[-1] and consumed == acquisitions[-2]
        assert trace[acquisitions[-1]].epoch is None and precedes(reset, acquisitions[-1])
        assert not any(node.kind == "cross-set" and node.flag[0] in owner_flags
                       and node.epoch == last_epoch for node in trace)
        bounded_counters.add((new.flag[0], old.lane))
    for counter, (notifications, acquisitions) in broadcasts.items():
        # All notifications of this fixture's recipient counter use one DMA
        # pipe. At its jth completion, at least every predecessor wait has
        # consumed a notification. This bounds *all* topological interleavings
        # of the completion graph, rather than sampling one issue ordering.
        assert len({trace[index].pipe for index in notifications}) == 1
        bound = max(ordinal + 1 - sum(precedes(acquire, notify) for acquire in acquisitions)
                    for ordinal, notify in enumerate(notifications))
        assert bound <= (2 if counter in bounded_counters else 1), \
            f"Unbounded ordinary counter accumulation: {counter}, outstanding={bound}"
        assert len(notifications) == len(acquisitions)
        assert all(precedes(notify, acquisitions[-1]) for notify in notifications)
    return trace, precedes


def _assert_owner_input(mod, epochs):
    flags = _input_flags(mod)
    assert len(flags) == 4
    base = min(flags)
    assert flags == set(range(base, base + 4)) and 0 <= base <= 12
    for lane in range(2):
        published, writes = defaultdict(list), defaultdict(list)
        release_waits = defaultdict(list)
        for index, (call, env, loops) in enumerate(_trace(mod, core=1, lane=lane)):
            epoch, name = _epoch(loops), _name(call)
            if name == _SET and _value(call.args[2], env) in flags:
                assert call.args[1].value == "PIPE_MTE3" and epoch is not None
                published[epoch].append((index, _value(call.args[2], env)))
            elif name == _WAIT and call.args[1].value == "PIPE_MTE3":
                release_waits[epoch].append(index)
            elif name == _COPY:
                dest, origin, shape = _region(call.args[1], env)
                if dest.name == "P":
                    assert origin[1:3] == (lane, epoch % 2)
                    writes[epoch].append((index, origin[-2], shape[-2]))
        assert set(published) == set(writes) == set(range(epochs))
        for epoch in range(epochs):
            dummy, real = published[epoch]
            assert dummy[1] == base + (1 - lane) * 2 + epoch % 2
            assert real[1] == base + lane * 2 + epoch % 2
            assert dummy[0] < min(index for index, _, _ in writes[epoch])
            assert real[0] > max(index for index, _, _ in writes[epoch])
            assert sorted((row, row + height) for _, row, height in writes[epoch]) == \
                [(row, row + 16) for row in range(0, 128, 16)]
            assert release_waits[epoch] and max(release_waits[epoch]) < dummy[0], \
                "Empty contribution must follow the ordinary old-P release wait"

    acquired, reads, read_rows = Counter(), Counter(), defaultdict(list)
    for call, env, loops in _trace(mod, core=2, lane=0):
        epoch, name = _epoch(loops), _name(call)
        if name == _WAIT and _value(call.args[2], env) in flags:
            owner = next(index for var, index, _ in loops if var.name == "owner")
            assert call.args[1].value == "PIPE_MTE2" and not reads[(epoch, owner)]
            assert _value(call.args[2], env) == base + owner * 2 + epoch % 2
            acquired[(epoch, owner)] += 1
        elif name == _COPY:
            source, origin, shape = _region(call.args[0], env)
            if source.name == "P":
                owner = origin[1]
                assert acquired[(epoch, owner)] == 1
                assert origin[2] == epoch % 2
                reads[(epoch, owner)] += 1
                read_rows[(epoch, owner)].append((origin[-2], origin[-2] + shape[-2]))
    assert acquired == Counter({(epoch, owner): 1 for epoch in range(epochs) for owner in range(2)})
    assert reads == Counter({key: 4 for key in acquired})
    assert all(sorted(rectangles) == [(row, row + 32) for row in range(0, 128, 32)]
               for rectangles in read_rows.values())

    trace, precedes = _completion_graph(mod)
    writes = defaultdict(list)
    reads = defaultdict(list)
    arrivals = defaultdict(list)
    acquisitions = {}
    for index, node in enumerate(trace):
        if node.kind in ("write", "read"):
            owner, slot, _, _ = node.region
            assert slot == node.epoch % 2
            (writes if node.kind == "write" else reads)[node.epoch, owner].append(index)
        elif node.kind == "cross-set" and node.flag[0] in flags:
            owner = (node.flag[0] - base) // 2
            arrivals[node.epoch, owner].append(index)
        elif node.kind == "cross-wait" and node.flag[0] in flags:
            owner = (node.flag[0] - base) // 2
            acquisitions[node.epoch, owner] = index
    for epoch in range(epochs):
        for owner in range(2):
            assert {trace[index].lane for index in arrivals[epoch, owner]} == {0, 1}
            assert len(arrivals[epoch, owner]) == 2
            assert all(precedes(write, read) for write in writes[epoch, owner]
                       for read in reads[epoch, owner])
            if epoch >= 2:
                # Both participants' new generation must follow the old
                # aggregate wait, and all owners' old input reads must finish
                # before either physical AIV overwrites this shared slot.
                assert all(precedes(acquisitions[epoch - 2, owner], arrival)
                           for arrival in arrivals[epoch, owner])
                assert all(precedes(read, write)
                           for previous_owner in range(2) for read in reads[epoch - 2, previous_owner]
                           for write in writes[epoch, owner])
    return flags


@pytest.mark.parametrize("epochs", [3, 5])
def test_owner_input_staggers_two_participants_and_preserves_finite_release(epochs):
    after = transform.InsertSync()(_scheduled_input(epochs=epochs))
    _assert_owner_input(after, epochs)


@pytest.mark.parametrize("kwargs", [
    {"private": False}, {"write_subs": 7}, {"alias": True},
    {"consumer_epochs": 2}, {"unstable_activation": True},
    {"reserved_flags": 12}, {"arch": "dav-3510"},
], ids=["no-private-proof", "read-beyond-produced-owner", "distinct-buffer-alias",
        "mismatched-active-interval", "row-dependent-activation", "owner-id-pressure", "3510"])
def test_unproven_owner_input_keeps_collective_readiness_and_release(kwargs):
    after = transform.InsertSync()(_scheduled_input(**kwargs))
    assert not _input_flags(after)
    events = calls(after, _SET) + calls(after, _WAIT)
    assert events
    mode = 4 if kwargs.get("arch") == "dav-3510" else 2
    assert {int(event.args[0]) for event in events} == {mode}
    assert any(event.args[1].value == "PIPE_MTE3" for event in calls(after, _SET))
    assert any(event.args[1].value == "PIPE_MTE2" for event in calls(after, _WAIT))
