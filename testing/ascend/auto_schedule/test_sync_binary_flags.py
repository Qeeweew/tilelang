"""Local DMA flags are binary state, including pipeline fill and drain.

The finite oracle models DMA/flag completion in each pipe, not source issue
order between pipes. Set->wait edges discharge one notification. A completed
wait must independently precede the next set of that ID; the oracle never
adds that reuse edge on behalf of InsertSync. Typed DMA epochs must also have
RAW and overwrite-completion paths. This is a CPU protocol proof for these
finite fixtures, not a device or general instruction-latency model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import copy_ring, unit


@dataclass(frozen=True)
class _Completion:
    pipe: str
    kind: str
    flag: tuple | None = None
    epoch: tuple | None = None
    slot: tuple | None = None


def _completion_trace(mod, parameters=None):
    """Execute finite controls and retain asynchronous DMA completion nodes."""
    result, values, scalar_memory = [], dict(parameters or {}), {}
    analyzer = tvm.arith.Analyzer()

    def integer(expression):
        expression = tirx.stmt_functor.substitute(
            expression, {var: tirx.const(value, var.dtype) for var, value in values.items()})

        def scalar_load(node):
            if isinstance(node, tirx.BufferLoad):
                assert node.buffer.scope() == "local.var", "Only explicit counter state is interpreted"
                indices = tuple(int(analyzer.simplify(index)) for index in node.indices)
                return tirx.const(scalar_memory[(node.buffer.data, indices)], node.dtype)

        expression = tirx.stmt_functor.ir_transform(tirx.Evaluate(expression), None, scalar_load).value
        expression = analyzer.simplify(expression)
        assert isinstance(expression, tirx.IntImm), f"Fixture control is not constant: {expression}"
        return int(expression)

    def slot(load):
        indices = [integer(index) for index in load.indices]
        strides = [integer(stride) for stride in load.buffer.strides]
        if not strides:
            strides = [1] * len(indices)
            for axis in range(len(indices) - 2, -1, -1):
                strides[axis] = strides[axis + 1] * integer(load.buffer.shape[axis + 1])
        return load.buffer.data, sum(index * stride for index, stride in zip(indices, strides))

    def epoch(load):
        # Fixtures copy one complete trailing tile from each GM/consumer epoch.
        trailing = 1 if len(load.buffer.shape) == 2 else 2
        return tuple(integer(index) for index in load.indices[:-trailing])

    def visit(statement):
        if isinstance(statement, tirx.SeqStmt):
            for child in statement.seq:
                visit(child)
        elif isinstance(statement, tirx.SBlockRealize):
            assert integer(statement.predicate)
            visit(statement.block)
        elif isinstance(statement, (tirx.SBlock, tirx.AttrStmt)):
            visit(statement.body)
        elif isinstance(statement, tirx.For):
            assert statement.loop_var not in values
            start, count = integer(statement.min), integer(statement.extent)
            for index in range(start, start + count):
                values[statement.loop_var] = index
                visit(statement.body)
            values.pop(statement.loop_var, None)
        elif isinstance(statement, tirx.IfThenElse):
            branch = statement.then_case if integer(statement.condition) else statement.else_case
            if branch is not None:
                visit(branch)
        elif isinstance(statement, tirx.BufferStore):
            assert statement.buffer.scope() == "local.var"
            key = statement.buffer.data, tuple(integer(index) for index in statement.indices)
            scalar_memory[key] = integer(statement.value)
        elif isinstance(statement, tirx.Evaluate):
            call = statement.value
            if isinstance(call, tirx.IntImm):
                return
            assert isinstance(call, tirx.Call) and isinstance(call.op, tvm.ir.Op)
            name = call.op.name
            if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                event = str(call.args[0].value)
                source, destination = event.split("_")
                operation = "set" if name == "tl.ascend_set_flag" else "wait"
                event_id = integer(call.args[1])
                assert 0 <= event_id < 8, "Local binary event ID exceeds the dav-2201 bank"
                result.append(_Completion(source if operation == "set" else destination,
                                          operation, (event, event_id)))
            elif name == "tl.ascend_pipe_barrier":
                result.append(_Completion(str(call.args[0].value).removeprefix("PIPE_"), "barrier"))
            else:
                assert name == "tl.tileop.ascend_copy", f"Unexpected instruction in DMA fixture: {name}"
                source, destination = (argument.args[0] for argument in call.args)
                if source.buffer.scope() == "global":
                    result.append(_Completion("MTE2", "produce", epoch=epoch(source), slot=slot(destination)))
                elif destination.buffer.scope() == "global":
                    result.append(_Completion("MTE3", "consume", epoch=epoch(destination), slot=slot(source)))
                else:
                    assert source.buffer.scope() == "shared.l1" and destination.buffer.scope() == "shared.l0a"
                    result.append(_Completion("MTE1", "consume", epoch=epoch(destination), slot=slot(source)))
        else:
            raise AssertionError(f"Unexpected fixture statement: {type(statement).__name__}")

    visit(mod["main"].body)
    return result


def _prove_binary_protocol(trace):
    """Check every completion ordering, without inserting missing reuse edges."""
    successors = [set() for _ in trace]
    last_pipe, sets, waits = {}, defaultdict(list), defaultdict(list)
    for index, node in enumerate(trace):
        if node.pipe in last_pipe:
            # DMA pipe FIFO completion is the existing MTE contract. This
            # oracle deliberately contains no V arithmetic issue-order rule.
            successors[last_pipe[node.pipe]].add(index)
        last_pipe[node.pipe] = index
        if node.kind in ("set", "wait"):
            (sets if node.kind == "set" else waits)[node.flag].append(index)
    assert set(sets) == set(waits), "Unmatched local event IDs"
    for flag in sets:
        assert len(sets[flag]) == len(waits[flag]), f"Unbalanced fill/drain for {flag}"
        for notify, acquire in zip(sets[flag], waits[flag]):
            successors[notify].add(acquire)

    indegrees = [0] * len(trace)
    for outgoing in successors:
        for destination in outgoing:
            indegrees[destination] += 1
    ready, visited = deque(index for index, count in enumerate(indegrees) if count == 0), 0
    while ready:
        source = ready.popleft()
        visited += 1
        for destination in successors[source]:
            indegrees[destination] -= 1
            if indegrees[destination] == 0:
                ready.append(destination)
    assert visited == len(trace), "Local waits deadlock the finite DMA completion graph"

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

    for flag in sets:
        for consumed, reused in zip(waits[flag], sets[flag][1:]):
            assert precedes(consumed, reused), f"Binary flag {flag} may be set again before its prior wait completes"
    return precedes


def _prove_data_epochs(trace, precedes, expected):
    producers = {node.epoch: index for index, node in enumerate(trace) if node.kind == "produce"}
    consumers = {node.epoch: index for index, node in enumerate(trace) if node.kind == "consume"}
    assert len(producers) == sum(node.kind == "produce" for node in trace)
    assert len(consumers) == sum(node.kind == "consume" for node in trace)
    assert set(producers) == set(consumers) == set(expected), "Fill/drain must execute each logical DMA epoch once"
    last_slot = {}
    for epoch, producer in producers.items():
        consumer = consumers[epoch]
        storage_slot = trace[producer].slot
        assert storage_slot == trace[consumer].slot, "Delayed consumer selected another epoch's physical slot"
        assert precedes(producer, consumer), f"Missing producer completion before reading {epoch}"
        if storage_slot in last_slot:
            assert precedes(consumers[last_slot[storage_slot]], producer), "Storage overwritten before its old consumer completes"
        last_slot[storage_slot] = epoch


def _dma_program(trips, *, stages=1, versions=2, manual=False, nested=False, trip_count=None, clock_dtype="int32"):
    shape = (2, trips, 16, 16) if nested else (trips, 16, 16)
    source = tirx.decl_buffer(shape, "float16", name="Input")
    consumed = tirx.decl_buffer(shape, "float16", name="Consumed", scope="shared.l0a")
    storage = tirx.decl_buffer((versions, 16, 16) if manual else (16, 16),
                              "float16", name="storage", scope="shared.l1")
    i, outer = tirx.Var("epoch", clock_dtype), tirx.Var("outer", "int32")
    count = tirx.const(trips, clock_dtype) if trip_count is None else trip_count
    prefix = [outer] if nested else []
    # Manual ring fixtures spell out paired-style physical fill/drain. The
    # automatic ring fixture instead lets LowerScheduledTIR realize stages.
    read_epoch = i - stages if manual else i
    producer_slot = [i % versions, 0, 0] if manual else None
    consumer_slot = [read_epoch % versions, 0, 0] if manual else None
    tile = [1, 16, 16] if manual else None
    write = copy(source, storage, src_indices=[*prefix, i, 0, 0],
                 src_shape=[*([1] * (len(shape) - 2)), 16, 16],
                 dst_indices=producer_slot, dst_shape=tile)
    read = copy(storage, consumed, src_indices=consumer_slot, src_shape=tile,
                dst_indices=[*prefix, read_epoch, 0, 0],
                dst_shape=[*([1] * (len(shape) - 2)), 16, 16])
    body = seq(unit(write, core=2, guard=i < count if manual else None),
               unit(read, core=2, stage=0 if manual else stages,
                    guard=i >= stages if manual else None))
    owner = tirx.For(i, tirx.const(0, clock_dtype), count + stages if manual else count, tirx.ForKind.SERIAL, body,
                     annotations={} if manual else {"multi_buffer_eligible": [storage.data]})
    body = unit(owner, core=None)
    if nested:
        body = unit(tirx.For(outer, 0, 2, tirx.ForKind.SERIAL, body), core=None)
    annotations = {"tl.manual_multi_buffer": {storage.data: versions}} if manual else {
        "tl.buffer_versions_map": {storage.data: versions},
        "tl.buffer_version_mode": {storage.data: "iteration"},
    }
    params = [source] if trip_count is None else [source, trip_count]
    mod = kernel(body, buffers=[storage, consumed], params=params, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod


def _lower_protocol(mod, *, prepared=False):
    if not prepared:
        mod = transform.PrepareMultiBuffer()(mod)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        mod = transform.InsertSync()(mod)
    mod = transform.MaterializeMultiBuffer()(mod)
    return transform.LowerScheduledTIR()(mod)


@pytest.mark.parametrize("trips", [1, 2, 3, 5])
@pytest.mark.parametrize("manual,versions", [(False, 1), (False, 2), (True, 2)],
                         ids=["single-slot-delayed", "automatic-ring", "manual-phases"])
def test_delayed_dma_fill_drain_uses_binary_flags_and_preserves_data_epochs(trips, manual, versions):
    trace = _completion_trace(_lower_protocol(_dma_program(trips, manual=manual, versions=versions)))
    precedes = _prove_binary_protocol(trace)
    _prove_data_epochs(trace, precedes, [(epoch,) for epoch in range(trips)])


@pytest.mark.parametrize("trips", [1, 2, 3, 5, 7])
def test_distance_two_manual_ring_has_partial_seeding_and_correct_wrap(trips):
    trace = _completion_trace(_lower_protocol(_dma_program(trips, stages=2, versions=4, manual=True)))
    _prove_data_epochs(trace, _prove_binary_protocol(trace), [(epoch,) for epoch in range(trips)])
    first_producer = next(index for index, node in enumerate(trace) if node.kind == "produce")
    seeded = [node.flag for node in trace[:first_producer]
              if node.kind == "set" and node.flag[0] == "MTE2_MTE1"]
    # Four data slots do not mean four initial ready notifications. Filling
    # every event ID would overwrite notifications before their consumers.
    assert 0 < len(seeded) < 4
    assert len(set(seeded)) == len(seeded)
    if trips > 4:
        produced_slots = [node.slot for node in trace if node.kind == "produce"]
        assert len(set(produced_slots)) == 4 and produced_slots[0] == produced_slots[4]


@pytest.mark.parametrize("trips", [0, 1])
def test_symbolic_trip_can_be_empty_or_shorter_than_distance_two(trips):
    count = tirx.Var("trip_count", "int32")
    # Only the iteration count is symbolic; all backing storage has a fixed
    # five-epoch capacity. Runtime values here are within that capacity.
    lowered = _lower_protocol(_dma_program(5, stages=2, versions=4, manual=True, trip_count=count))
    trace = _completion_trace(lowered, {count: trips})
    _prove_data_epochs(trace, _prove_binary_protocol(trace), [(epoch,) for epoch in range(trips)])


def _disjoint_rows_program(trips):
    source, output = [tirx.decl_buffer((trips, 64), "float32", name=name)
                      for name in ("Input", "Output")]
    storage = tirx.decl_buffer((trips, 64), "float32", name="independent_rows", scope="shared.dyn")
    i = tirx.Var("row_epoch", "int32")
    body = seq(
        unit(copy(source, storage, src_indices=[i, 0], src_shape=[1, 64],
                  dst_indices=[i, 0], dst_shape=[1, 64])),
        unit(copy(storage, output, src_indices=[i, 0], src_shape=[1, 64],
                  dst_indices=[i, 0], dst_shape=[1, 64])),
    )
    # Every row is distinct. There is a RAW in each iteration but no memory
    # WAR providing an implicit reverse completion path between iterations.
    mod = kernel(unit(tirx.For(i, 0, trips, tirx.ForKind.SERIAL, body), core=None),
                 buffers=[storage], params=[source, output])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod


def test_distance_zero_cannot_reuse_a_flag_without_reverse_completion():
    before = transform.PrepareMultiBuffer()(_disjoint_rows_program(9))
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="dav-2201 local binary flag: "
                       "no reverse completion path for MTE2->MTE3"):
        transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


@pytest.mark.parametrize("trips", [1, 3, 5, 8])
def test_finite_disjoint_rows_use_each_flag_once_without_reverse_completion(trips):
    trace = _completion_trace(_lower_protocol(_disjoint_rows_program(trips)))
    _prove_data_epochs(trace, _prove_binary_protocol(trace), [(epoch,) for epoch in range(trips)])
    notifications = [node.flag for node in trace if node.kind == "set"]
    assert len(notifications) == len(set(notifications)), "A finite no-reverse protocol cannot reuse an event ID"


@pytest.mark.parametrize("nested,versions", [(False, 1), (False, 2), (True, 1)])
def test_distance_zero_dma_protocol_keeps_completion_and_finite_reuse(nested, versions):
    trace = _completion_trace(_lower_protocol(_dma_program(3, stages=0, versions=versions, nested=nested)))
    precedes = _prove_binary_protocol(trace)
    expected = [(outer, epoch) for outer in range(2) for epoch in range(3)] if nested else [(epoch,) for epoch in range(3)]
    _prove_data_epochs(trace, precedes, expected)


def test_counter_clock_distance_zero_protocol_remains_binary():
    before = copy_ring(mode="counter", versions=2, guarded=True, prepared=True)
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    trace = _completion_trace(_lower_protocol(before, prepared=True))
    _prove_data_epochs(trace, _prove_binary_protocol(trace), [(0,), (2,)])


def test_shifted_nested_owner_is_rejected_before_unsafe_local_reseeding():
    before = transform.PrepareMultiBuffer()(_dma_program(3, stages=1, versions=2, manual=True, nested=True))
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="(?s)dav-2201 local binary flag:.*"
                       "shifted carried events require an unconditional outermost lexical loop"):
        transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def test_shifted_unsigned_clock_is_rejected_before_negative_fill_epochs():
    before = transform.PrepareMultiBuffer()(_dma_program(3, manual=True, clock_dtype="uint32"))
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="dav-2201 local binary flag: "
                       "shifted epoch clocks require signed integer iteration and trip-count types"):
        transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def test_binary_oracle_rejects_balanced_counting_tokens():
    flag = ("MTE2_MTE1", 0)
    trace = [_Completion("MTE2", "set", flag), _Completion("MTE2", "set", flag),
             _Completion("MTE1", "wait", flag), _Completion("MTE1", "wait", flag)]
    with pytest.raises(AssertionError, match="set again before its prior wait completes"):
        _prove_binary_protocol(trace)


def test_binary_oracle_does_not_treat_lexical_alternation_as_completion():
    flag = ("MTE2_MTE1", 0)
    trace = [_Completion("MTE2", "set", flag), _Completion("MTE1", "wait", flag),
             _Completion("MTE2", "set", flag), _Completion("MTE1", "wait", flag)]
    with pytest.raises(AssertionError, match="set again before its prior wait completes"):
        _prove_binary_protocol(trace)
