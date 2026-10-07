"""A wider protocol may cover the same physical lexical boundary.

These are prepared, assigned pass inputs. The finite proof uses source-fence
completion and destination wait gates, without ordinary DMA completion FIFO
or implicit backpressure on a second SetFlag.
"""

from collections import Counter, defaultdict, deque

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_sync_binary_flags import _Completion, _completion_trace


def _program(*, variant="covered"):
    source = tirx.decl_buffer((2, 4, 4, 16, 16), "float16", name="Input")
    consumed = tirx.decl_buffer(source.shape, "float16", name="Consumed", scope="shared.l0a")
    narrow = tirx.decl_buffer((16, 16), "float16", name="narrow", scope="shared.l1")
    counter_mode = variant == "counter"
    wide = tirx.decl_buffer((16, 16) if counter_mode else (2, 16, 16),
                           "float16", name="wide", scope="shared.l1")
    counter = tirx.decl_buffer((1,), "int32", name="counter", scope="local.var")
    e, chunk, count = (tirx.Var(name, "int32") for name in ("epoch", "chunk", "count"))
    active = tirx.And(e >= 1, e < 4)
    # A condition snapshot need not be structurally identical to the prepared
    # active guard. Both domain variants then stop at this same guarded For.
    enclosing = tirx.And(e > 0, e <= 3)
    wide_guard = e % 2 == 0 if variant == "holey" else tirx.const(True, "bool")
    if variant == "holey":
        enclosing = tirx.Or(active, wide_guard)

    def protocol(index, storage):
        is_wide = index == 1
        slot = e % 2 if variant == "distance" else chunk % 2
        origin = None if not is_wide or counter_mode else [slot, 0, 0]
        shape = None if origin is None else [1, 16, 16]
        guard = (wide_guard if is_wide else active) if variant == "holey" else None
        return [
            unit(copy(source, storage, src_indices=[index, e - 1 if variant != "holey" else e, chunk, 0, 0],
                      src_shape=[1, 1, 1, 16, 16], dst_indices=origin, dst_shape=shape), core=2, guard=guard),
            unit(copy(storage, consumed, src_indices=origin, src_shape=shape,
                      dst_indices=[index, e - 1 if variant != "holey" else e, chunk, 0, 0],
                      dst_shape=[1, 1, 1, 16, 16]), core=2, guard=guard),
        ]

    pairs = [protocol(0, narrow), protocol(1, wide)]
    # Four chunks exercise real reuse of the two-slot manual ring. A repeated
    # outer owner cannot justify binary reuse from a two-trip inner bound.
    if variant == "endpoints":
        bodies = [unit(tirx.For(chunk, 0, 4, tirx.ForKind.SERIAL, seq(*pair)),
                       core=None, guard=enclosing) for pair in pairs]
    else:
        bodies = [unit(tirx.For(chunk, 0, 4, tirx.ForKind.SERIAL, seq(*pairs[0], *pairs[1])),
                       core=None, guard=enclosing)]
    loop_annotations = {"tl.storage_epoch_guard_map": {narrow.data: active}}
    annotations = {"tl.manual_multi_buffer": {wide.data: 2}} if not counter_mode else {
        "tl.buffer_versions_map": {wide.data: 2}}
    buffers = [narrow, wide, consumed]
    if variant == "holey":
        loop_annotations["tl.storage_epoch_guard_map"][wide.data] = wide_guard
    if counter_mode:
        loop_annotations.update({"multi_buffer_eligible": [wide.data],
                                 "tl.multi_buffer_counter_map": {wide.data: counter}})
        loop_annotations["tl.storage_epoch_guard_map"][wide.data] = active
        bodies.append(unit(tirx.BufferStore(counter, counter[0] + 1, [0]), core=2, guard=enclosing))
        buffers.append(counter)
    owner = unit(tirx.For(e, 0, count, tirx.ForKind.SERIAL, seq(*bodies),
                          annotations=loop_annotations), core=None)
    body = seq(unit(tirx.BufferStore(counter, 0, [0]), core=2), owner) if counter_mode else owner
    mod = kernel(body, buffers=buffers, params=[source, count], annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, e, chunk, count


def _boundary_calls(mod, owner, child, operation):
    result = []
    for statement, parents in statements(mod):
        if not isinstance(statement, tirx.Evaluate) or not isinstance(statement.value, tirx.Call):
            continue
        call = statement.value
        if not isinstance(call.op, tvm.ir.Op) or call.op.name != "tl.ascend_" + operation + "_flag":
            continue
        if call.args[0].value != "MTE1_MTE2":
            continue
        loops = [parent.loop_var for parent in parents if isinstance(parent, tirx.For)]
        if any(var.same_as(owner) for var in loops) and not any(var.same_as(child) for var in loops):
            result.append(call)
    return result


def _strong_proof(trace, expected):
    dependencies, pending, gates = {}, defaultdict(set), {}
    notifications, acquisitions = defaultdict(list), defaultdict(list)
    scalar = None
    for index, command in enumerate(trace):
        issue, start, done = 3 * index, 3 * index + 1, 3 * index + 2
        dependencies[issue] = {scalar} if scalar is not None else set()
        dependencies[start], dependencies[done] = {issue}, {start}
        if command.pipe in gates:
            dependencies[start].add(gates[command.pipe])
        if command.kind in ("set", "barrier"):
            dependencies[done].update(pending[command.pipe])
            pending[command.pipe].clear()
        pending[command.pipe].add(done)
        if command.kind in ("wait", "barrier"):
            gates[command.pipe] = done
        if command.kind in ("set", "wait"):
            (notifications if command.kind == "set" else acquisitions)[command.flag].append(done)
        scalar = issue
    assert notifications.keys() == acquisitions.keys()
    for flag, sent in notifications.items():
        assert len(sent) == len(acquisitions[flag]), "Unbalanced seed/set/wait/drain"
        for notify, acquire in zip(sent, acquisitions[flag]):
            dependencies[acquire].add(notify)
    children, remaining = defaultdict(list), {node: len(parents) for node, parents in dependencies.items()}
    for node, parents in dependencies.items():
        for parent in parents:
            children[parent].append(node)
    ready, rank = deque(node for node, degree in remaining.items() if not degree), {}
    while ready:
        node = ready.popleft()
        rank[node] = len(rank)
        for child in children[node]:
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)
    assert len(rank) == len(dependencies), "Deadlocked completion graph"

    def require(before, after):
        frontier, seen = [after], set()
        while frontier:
            node = frontier.pop()
            if node == before:
                return
            if node not in seen and rank[node] > rank[before]:
                seen.add(node)
                frontier.extend(dependencies[node])
        raise AssertionError("Missing completion path")

    for flag, sent in notifications.items():
        for cleared, reset in zip(acquisitions[flag], sent[1:]):
            require(cleared, reset)
    producers, consumers = {}, {}
    for index, command in enumerate(trace):
        if command.kind in ("produce", "consume"):
            accesses = producers if command.kind == "produce" else consumers
            assert command.epoch not in accesses
            accesses[command.epoch] = index
    assert Counter(producers.keys()) == Counter(consumers.keys()) == Counter(expected)
    last_slot = {}
    for epoch, producer in producers.items():
        consumer = consumers[epoch]
        slot = trace[producer].slot
        assert slot == trace[consumer].slot
        require(3 * producer + 2, 3 * consumer + 1)
        if slot in last_slot:
            require(3 * consumers[last_slot[slot]] + 2, 3 * producer + 1)
        last_slot[slot] = epoch


@pytest.mark.parametrize("count", [0, 1, 2, 5])
def test_wide_same_boundary_protocol_covers_shifted_active_epochs(count):
    before, owner, child, parameter = _program()
    snapshot = tvm.ir.save_json(before)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        after = transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot
    for operation in ("set", "wait"):
        assert len(_boundary_calls(after, owner, child, operation)) == 1
    after = transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(after))
    trace = _completion_trace(after, {parameter: count})
    _strong_proof(trace, [(kind, epoch - 1, chunk) for epoch in range(count)
                         if 1 <= epoch < 4 for chunk in range(4) for kind in range(2)])


@pytest.mark.parametrize("variant", ["endpoints", "distance"])
def test_distinct_physical_endpoints_or_epoch_distance_keep_both_protocols(variant):
    before, owner, child, _ = _program(variant=variant)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        after = transform.InsertSync()(before)
    for operation in ("set", "wait"):
        assert len(_boundary_calls(after, owner, child, operation)) == 2


def test_counter_clock_cannot_cover_lexical_protocol():
    before, owner, child, parameter = _program(variant="counter")
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        after = transform.InsertSync()(before)
    for operation in ("set", "wait"):
        # The lexical protocol remains at the root boundary. The counter
        # protocol stays inside its owner, at the repeated child accesses.
        assert len(_boundary_calls(after, owner, child, operation)) == 1
        counter_events = [call for call in calls(after, "tl.ascend_" + operation + "_flag")
                          if call.args[0].value == "MTE1_MTE2" and nodes(call.args[1], tirx.BufferLoad)]
        assert len(counter_events) == 1
    after = transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(after))
    _strong_proof(_completion_trace(after, {parameter: 5}),
                  [(kind, epoch - 1, chunk) for epoch in range(1, 4)
                   for chunk in range(4) for kind in range(2)])


def test_incomparable_holey_guard_cannot_cover_lexical_protocol():
    before, _, _, _ = _program(variant="holey")
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="dav-2201 local binary flag:"):
        transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def test_completion_oracle_rejects_balanced_flags_without_clear_before_reset():
    flag = ("MTE2_MTE1", 0)
    trace = [_Completion("MTE2", "set", flag), _Completion("MTE1", "wait", flag),
             _Completion("MTE2", "set", flag), _Completion("MTE1", "wait", flag)]
    with pytest.raises(AssertionError, match="Missing completion path"):
        _strong_proof(trace, [])
