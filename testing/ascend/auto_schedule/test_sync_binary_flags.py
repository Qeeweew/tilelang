"""Local flags are binary state, including pipeline fill and drain.

The finite oracle models DMA/flag completion in each pipe, not source issue
order between pipes. Set->wait edges discharge one notification. A completed
wait must independently precede the next set of that ID; the oracle never
adds that reuse edge on behalf of InsertSync. Typed DMA epochs must also have
RAW and overwrite-completion paths. This is a CPU protocol proof for these
finite fixtures, not a device or general instruction-latency model. The mask
fixtures also model blocking Scalar admission and Vector flag completion;
independent Vector arithmetic has no implicit completion-order edge.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace

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


def _completion_trace(mod, parameters=None, *, scalar_vector=False):
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

        def unsigned_residue(node):
            # Analyzer retains unsigned Mod even with literal operands. Fold
            # only these finite nonnegative residues and lossless event casts.
            if (isinstance(node, (tirx.Mod, tirx.FloorMod)) and node.dtype == "uint32"
                    and isinstance(node.a, tirx.IntImm) and isinstance(node.b, tirx.IntImm)):
                assert int(node.a) >= 0 and int(node.b) > 0
                return tirx.const(int(node.a) % int(node.b), node.dtype)
            if isinstance(node, tirx.Cast) and node.dtype == "int32" and isinstance(node.value, tirx.IntImm):
                assert -(1 << 31) <= int(node.value) < 1 << 31
                return tirx.const(int(node.value), "int32")

        expression = tirx.stmt_functor.ir_transform(tirx.Evaluate(expression), None, unsigned_residue).value
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
            if statement.buffer.scope() == "local.var":
                key = statement.buffer.data, tuple(integer(index) for index in statement.indices)
                if scalar_vector and isinstance(statement.value, tirx.Add):
                    result.append(_Completion("S", "tick", slot=(statement.buffer.data, 0)))
                scalar_memory[key] = integer(statement.value)
            else:
                assert scalar_vector and statement.buffer.scope() == "shared.dyn"
                kind = "mask_write" if statement.buffer.dtype == "uint64" else "scalar_write"
                result.append(_Completion("S", kind, slot=slot(statement.buffer[tuple(statement.indices)])))
        elif scalar_vector and isinstance(statement, tirx.Bind):
            def scalar_read(node):
                if isinstance(node, tirx.BufferLoad):
                    assert node.buffer.scope() == "shared.dyn"
                    result.append(_Completion("S", "scalar_read", slot=slot(node)))

            tirx.stmt_functor.post_order_visit(statement.value, scalar_read)
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
            elif scalar_vector and name == "tl.ascend_membase.select_bits":
                result.append(_Completion("V", "mask_read", slot=slot(call.args[3].args[0])))
            elif scalar_vector and name == "tl.ascend_membase.duplicate":
                result.append(_Completion("V", "vector_write", slot=slot(call.args[0].args[0])))
            elif scalar_vector and name == "tl.ascend_membase.copy_bits":
                result.append(_Completion("V", "vector_read", slot=slot(call.args[1].args[0])))
                result.append(_Completion("V", "vector_write", slot=slot(call.args[0].args[0])))
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
    vector_commands, vector_admission = [], []
    for index, node in enumerate(trace):
        if node.pipe == "V":
            if node.kind in ("set", "barrier"):
                # PIPE_V notification/barrier completes preceding commands.
                for previous in vector_commands:
                    successors[previous].add(index)
            else:
                # A wait/barrier gates later commands. Two independent
                # arithmetic reads need not complete in their issue order.
                for previous in vector_admission:
                    successors[previous].add(index)
            vector_commands.append(index)
            if node.kind in ("wait", "barrier"):
                vector_admission.append(index)
        elif node.pipe in last_pipe:
            # DMA completion FIFO and Scalar instruction order. Cross-pipe
            # Scalar wait admission is a separate flag-lifetime graph below.
            successors[last_pipe[node.pipe]].add(index)
        last_pipe[node.pipe] = index
        if node.kind in ("set", "wait"):
            (sets if node.kind == "set" else waits)[node.flag].append(index)
    assert set(sets) == set(waits), "Unmatched local event IDs"
    for flag in sets:
        assert len(sets[flag]) == len(waits[flag]), f"Unbalanced fill/drain for {flag}"
        for notify, acquire in zip(sets[flag], waits[flag]):
            successors[notify].add(acquire)

    def reachable(graph, source, destination):
        frontier, seen = [source], set()
        while frontier:
            current = frontier.pop()
            if current == destination:
                return True
            if current not in seen:
                seen.add(current)
                frontier.extend(graph[current] - seen)
        return False

    # A real destination-S wait consumes its bit before Scalar resumes
    # dispatching subsequent asynchronous notifications. This proves flag
    # lifetime only: it does not order two ordinary Vector data completions.
    # In particular a V/MTE destination wait does not stall Scalar dispatch.
    lifetime = [set(outgoing) for outgoing in successors]
    scalar_wait = None
    for index, node in enumerate(trace):
        if scalar_wait is not None and node.kind == "set":
            lifetime[scalar_wait].add(index)
        if node.kind == "wait" and node.pipe == "S":
            scalar_wait = index

    for graph in (successors, lifetime):
        indegrees = [0] * len(trace)
        for outgoing in graph:
            for destination in outgoing:
                indegrees[destination] += 1
        ready, visited = deque(index for index, count in enumerate(indegrees) if count == 0), 0
        while ready:
            source = ready.popleft()
            visited += 1
            for destination in graph[source]:
                indegrees[destination] -= 1
                if indegrees[destination] == 0:
                    ready.append(destination)
        assert visited == len(trace), "Local waits deadlock the finite DMA completion graph"

    for flag in sets:
        for consumed, reused in zip(waits[flag], sets[flag][1:]):
            assert reachable(lifetime, consumed, reused), f"Binary flag {flag} may be set again before its prior wait completes"
    return lambda source, destination: reachable(successors, source, destination)


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


def _counter_mask_program(*, tick="canonical", readers=1, versions=2):
    """Prepared pass input with a 0/1-row owner repeatedly entered by outer.

    The mask has one counter clock shared by its two scalar words. There are
    seventeen outer iterations, including holes in owner activation, so a finite
    eight-ID root allocation cannot substitute for a real reusable protocol.
    """
    mask = tirx.decl_buffer((2,), "uint64", name="mask", scope="shared.dyn")
    counter = tirx.decl_buffer((1,), "int32", name="mask_epoch", scope="local.var")
    other = tirx.decl_buffer((1,), "int32", name="other_epoch", scope="local.var")
    left, right = [tirx.decl_buffer((16,), "float32", name=name, scope="shared.dyn")
                   for name in ("left", "right")]
    outputs = [tirx.decl_buffer((17, 16), "float32", name=f"output{reader}", scope="shared.dyn")
               for reader in range(readers)]
    outer, row, sid = [tirx.Var(name, "int32") for name in ("outer", "row", "sid")]
    pointer = lambda load, count, effect: tirx.Call(
        "handle", tvm.ir.Op.get("tl.access_ptr"), [load, count, effect])
    consumers = [unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.select_bits"), [
        pointer(output[outer, 0], 16, 2), pointer(left[0], 16, 1),
        pointer(right[0], 16, 1), pointer(mask[0], 2, 1), tirx.IntImm("uint32", 16)])))
        for output in outputs]
    selected_counter = other if tick == "other-group" else counter
    advance_value = selected_counter[0] + (2 if tick == "add-two" else 1)
    if tick == "reset":
        advance_value = tirx.const(0, "int32")
    advance = unit(tirx.BufferStore(selected_counter, advance_value, [0]),
                   core=2 if tick == "wrong-core" else 1,
                   stage=1 if tick == "shifted" else 0,
                   guard=outer % 2 == 0 if tick == "guard-mismatch" else None)
    tasks = [unit(tirx.BufferStore(mask, tirx.Cast("uint64", outer + 1), [0])),
             unit(tirx.BufferStore(mask, tirx.const(0, "uint64"), [1]))]
    tasks += [advance, *consumers] if tick == "early" else [*consumers, advance]
    owner = tirx.For(row, 0, tirx.max(0, 1 - sid * 8), tirx.ForKind.SERIAL, seq(*tasks), annotations={
        "multi_buffer_eligible": [mask.data],
        "tl.multi_buffer_counter_map": {mask.data: counter},
        "tl.storage_epoch_guard_map": {mask.data: tirx.const(True, "bool")},
    })
    body = seq(unit(tirx.BufferStore(counter, 0, [0])), unit(tirx.BufferStore(other, 0, [0])),
               unit(tirx.For(outer, 0, 17, tirx.ForKind.SERIAL,
                             unit(owner, core=None, guard=outer % 3 != 1)), core=None))
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    mod = kernel(body, buffers=[mask, counter, other, left, right, *outputs],
                 annotations={"tl.buffer_versions_map": {mask.data: versions}})
    mod["main"] = mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, sid


def _prove_mask_epochs(trace, precedes, activations, readers, *, expected_ticks):
    latest_writes, pending_reads = {}, defaultdict(list)
    reads = ticks = writes = 0
    current_reads = []
    retirement_waits = [index for index, node in enumerate(trace)
                        if node.kind == "wait" and node.flag[0] == "V_S"]
    for index, node in enumerate(trace):
        if node.kind == "mask_write":
            writes += 1
            for prior in pending_reads.pop(node.slot, []):
                assert precedes(prior, index), "Mask overwritten before its Vector reader completes"
            latest_writes[node.slot] = index
        elif node.kind == "mask_read":
            reads += 1
            current_reads.append(index)
            assert any(precedes(index, wait) for wait in retirement_waits), "Vector reader has no real Scalar retirement wait"
            for word in range(2):
                cell = node.slot[0], node.slot[1] + word
                assert cell in latest_writes and precedes(latest_writes[cell], index), "Mask read before Scalar write completion"
                pending_reads[cell].append(index)
        elif node.kind == "tick":
            ticks += 1
            assert len(current_reads) == readers
            assert all(precedes(reader, index) for reader in current_reads), "Counter advanced before every Vector reader completes"
            current_reads.clear()
    assert (writes, reads, ticks) == (2 * activations, readers * activations, expected_ticks)


@pytest.mark.parametrize("sid_value", [0, 1], ids=["one-row", "empty-row"])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("versions", [1, 2])
def test_counter_mask_completion_gates_repeated_owner_and_every_reader(sid_value, readers, versions):
    before, sid = _counter_mask_program(readers=readers, versions=versions)
    snapshot = tvm.ir.save_json(before)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
        inserted = transform.InsertSync()(before)
        logical = transform.LowerScheduledTIR()(inserted)
        lowered = transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(inserted))
    assert tvm.ir.save_json(before) == snapshot
    activations = sum(outer % 3 != 1 for outer in range(17)) if sid_value == 0 else 0
    logical_trace = _completion_trace(logical, {sid: sid_value}, scalar_vector=True)
    _prove_mask_epochs(logical_trace, _prove_binary_protocol(logical_trace),
                       activations, readers, expected_ticks=activations)
    trace = _completion_trace(lowered, {sid: sid_value}, scalar_vector=True)
    precedes = _prove_binary_protocol(trace)
    # A unit-version counter is legitimately elided; its real Scalar wait
    # must remain and close every active notification and mask read.
    _prove_mask_epochs(trace, precedes, activations, readers,
                       expected_ticks=activations if versions > 1 else 0)
    if activations:
        for event in ("S_V", "V_S"):
            assert any(node.kind == "set" and node.flag[0] == event for node in trace)
        without_completion = [node for node in trace if node.flag is None or node.flag[0] != "V_S"]
        with pytest.raises(AssertionError, match="set again before its prior wait completes"):
            _prove_binary_protocol(without_completion)
        without_ready = [node for node in trace if node.flag is None or node.flag[0] != "S_V"]
        with pytest.raises(AssertionError, match="Mask read before Scalar write completion"):
            _prove_mask_epochs(without_ready, _prove_binary_protocol(without_ready),
                               activations, readers, expected_ticks=activations if versions > 1 else 0)


@pytest.mark.parametrize("tick", ["add-two", "reset", "other-group", "wrong-core", "guard-mismatch", "shifted", "early"])
def test_counter_mask_completion_rejects_noncanonical_or_mismatched_tick(tick):
    before, _ = _counter_mask_program(tick=tick)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="dav-2201 local binary flag: no reverse completion path for S->V"):
        with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
            transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def _scalar_wait_phases_program(*, rows=2, guarded=False, nested=False, read_placement="flat", root_pairs=3):
    """Closed phases with distinct storage epochs and conservative root IDs.

    Each Scalar read consumes three independent Vector writers through a
    real notification. The root pairs run once and have no overwrite WAR;
    their IDs stay outside the loop-phase certificate's admission set.
    """
    outer, sid = tirx.Var("phase_epoch", "int32"), tirx.Var("sid", "int32")
    storage = [[tirx.decl_buffer((16,), "float32", name=f"phase{phase}_lane{lane}", scope="shared.dyn")
                for lane in range(3)] for phase in range(2)]
    phases = []
    for phase, buffers in enumerate(storage):
        row = tirx.Var(f"phase_row{phase}", "int32")
        writers = [unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.duplicate"), [
            tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], 16, 2]),
            tirx.const(phase + lane + 1, "float32"), tirx.IntImm("uint32", 16)])))
            for lane, buffer in enumerate(buffers)]
        read = unit(tirx.Bind(tirx.Var(f"scalar{phase}", "float32"),
                              buffers[0][0] + buffers[1][0] + buffers[2][0]))
        if read_placement != "flat":
            scalar_row = tirx.Var(f"scalar_row{phase}", "int32")
            count = 2 if read_placement == "first-leaf" else 1 - sid
            read = unit(tirx.For(scalar_row, 0, count, tirx.ForKind.SERIAL, read), core=None)
        owner_rows = rows if read_placement == "lifted-entry" else rows * (1 - sid)
        owner = tirx.For(row, 0, owner_rows, tirx.ForKind.SERIAL, seq(*writers, read), annotations={
            "multi_buffer_eligible": [buffer.data for buffer in buffers],
            "tl.storage_epoch_guard_map": {buffer.data: tirx.const(True, "bool") for buffer in buffers},
        })
        phases.append(unit(owner, core=None, guard=outer % 3 != 1 if guarded else None))
    body = unit(tirx.For(outer, 0, 17, tirx.ForKind.SERIAL, seq(*phases)), core=None)
    if nested:
        repeat = tirx.Var("phase_repeat", "int32")
        body = unit(tirx.For(repeat, 0, 2, tirx.ForKind.SERIAL, body), core=None)
    roots = [tirx.decl_buffer((16,), "float32", name=f"root{index}", scope="shared.dyn")
             for index in range(root_pairs)]
    prefix = []
    for index, buffer in enumerate(roots):
        prefix.extend([
            unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.duplicate"), [
                tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], 16, 2]),
                tirx.const(index + 1, "float32"), tirx.IntImm("uint32", 16)]))),
            unit(tirx.Bind(tirx.Var(f"root_scalar{index}", "float32"), buffer[0])),
        ])
    body = seq(*prefix, body)
    buffers = [buffer for phase in storage for buffer in phase]
    mod = kernel(body, buffers=[*buffers, *roots], annotations={
        "tl.buffer_versions_map": {buffer.data: 1 for buffer in buffers},
    })
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    mod["main"] = mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, sid, storage


def _prove_scalar_read_epochs(trace, precedes, expected_reads, expected_writes):
    latest_writes, previous_reads = {}, {}
    reads = writes = 0
    for index, node in enumerate(trace):
        if node.kind == "vector_write":
            writes += 1
            if node.slot in previous_reads:
                assert precedes(previous_reads[node.slot], index), "Vector overwrite before its Scalar reader completes"
            latest_writes[node.slot] = index
        elif node.kind == "scalar_read":
            reads += 1
            assert node.slot in latest_writes and precedes(latest_writes[node.slot], index), "Scalar read before Vector completion"
            previous_reads[node.slot] = index
    assert (reads, writes) == (expected_reads, expected_writes)


@pytest.mark.parametrize("sid_value", [0, 1], ids=["active", "empty"])
@pytest.mark.parametrize("guarded", [False, True], ids=["every-epoch", "activation-holes"])
@pytest.mark.parametrize("read_placement", ["flat", "first-leaf", "lifted-entry"])
def test_closed_scalar_wait_phases_share_across_storage_epochs_without_overlap(sid_value, guarded, read_placement):
    # An unconditional lifted entry needs no outer V_S carry channels. Its
    # two local channels therefore need seven one-shot root IDs. Other layouts
    # also have two single-bit outer carry channels, requiring five root IDs
    # to retain nine-slot pressure after Scalar-admission width reduction.
    root_pairs = 7 if read_placement == "lifted-entry" and not guarded else 5
    before, sid, storage = _scalar_wait_phases_program(
        guarded=guarded, read_placement=read_placement, root_pairs=root_pairs)
    snapshot = tvm.ir.save_json(before)
    trace = _completion_trace(_lower_protocol(before, prepared=True), {sid: sid_value}, scalar_vector=True)
    assert tvm.ir.save_json(before) == snapshot
    precedes = _prove_binary_protocol(trace)
    activations = sum(not guarded or outer % 3 != 1 for outer in range(17))
    read_rows = 2 if read_placement == "first-leaf" else 1
    _prove_scalar_read_epochs(trace, precedes,
                              root_pairs + (6 * 2 * read_rows * activations if sid_value == 0 else 0),
                              root_pairs + (6 * 2 * activations if sid_value == 0 or read_placement == "lifted-entry" else 0))
    notifications = [node.flag for node in trace if node.kind == "set" and node.flag[0] == "V_S"]
    assert len(set(notifications)) <= 8
    if sid_value == 0:
        local_flags = []
        for buffers in storage:
            first_read = next(index for index, node in enumerate(trace)
                              if node.kind == "scalar_read" and node.slot[0].same_as(buffers[0].data))
            local_flags.append(next(node.flag for node in reversed(trace[:first_read])
                                    if node.kind == "wait" and node.flag[0] == "V_S"))
        assert len(set(local_flags)) == 1, "Disjoint closed phases must reuse under V_S capacity pressure"
    elif read_placement == "lifted-entry":
        first_write = next(index for index, node in enumerate(trace)
                           if node.kind == "vector_write" and
                           any(node.slot[0].same_as(buffer.data) for buffers in storage for buffer in buffers))
        for operation in ("set", "wait"):
            assert any(node.kind == operation and node.flag[0] == "V_S"
                       for node in trace[first_write:]), "Empty reader loop lost its real entry notification/wait"


def test_closed_scalar_wait_phases_preserve_shifted_nested_carry_rejection():
    before, _, _ = _scalar_wait_phases_program(guarded=True, nested=True)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="(?s)dav-2201 local binary flag:.*"
                       "shifted carried events require an unconditional outermost lexical loop"):
        with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
            transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def _scalar_admission_width_program(*, distance=0, read_stage=1, clock_dtype="int32"):
    """Prepared typed inputs with independently chosen stages and storage rings."""
    epoch = tirx.Var("admission_epoch", clock_dtype)
    sid, row = tirx.Var("admission_sid", "int32"), tirx.Var("admission_row", "int32")
    storage = tirx.decl_buffer((16,), "float32", name="admission_storage", scope="shared.dyn")
    count = tirx.IntImm("uint32", 16)

    def access(buffer, mask):
        return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], 16, mask])

    write = unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.duplicate"), [
        access(storage, 2), tirx.const(1, "float32"), count])), stage=1 if distance == 0 else 0)
    read = unit(tirx.Bind(tirx.Var("admission_scalar", "float32"), storage[0]))
    buffers = [storage]
    if distance == 0:
        # Both endpoints use the same root epoch. A possibly empty reader
        # forces the actual wait to its Control entry, outside the row loop.
        body = seq(write, unit(tirx.For(row, 0, 2 * (1 - sid), tirx.ForKind.SERIAL, read),
                               core=None, stage=read_stage))
        versions, eligible = 2, [storage.data]
    elif distance == 1:
        # The mixed inner owner has a local RAW and a projected outer carried
        # channel: its physical S wait is BEFORE the Control, V set AFTER it.
        owner = tirx.For(row, 0, 2 * (1 - sid), tirx.ForKind.SERIAL, seq(write, read), annotations={
            "multi_buffer_eligible": [storage.data],
            "tl.storage_epoch_guard_map": {storage.data: tirx.const(True, "bool")},
        })
        body, versions, eligible = unit(owner, core=None), 1, []
    else:
        assert distance == 2
        # Scalar writes slot e, then a stage-one Vector command reads it. Its
        # WAR permission for e+2 starts with two distinct initial tokens.
        temporary = tirx.decl_buffer((16,), "float32", name="admission_copy", scope="shared.dyn")
        buffers.append(temporary)
        body = seq(unit(tirx.BufferStore(storage, tirx.const(1, "float32"), [0])),
                   unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.copy_bits"), [
                       access(temporary, 2), access(storage, 1), tirx.IntImm("uint32", 1)])), stage=1))
        versions, eligible = 2, [storage.data]
    loop = tirx.For(epoch, tirx.const(0, clock_dtype), tirx.const(17, clock_dtype),
                   tirx.ForKind.SERIAL, body, annotations={
                       "multi_buffer_eligible": eligible,
                       "num_stages": max(2, read_stage + 1) if distance == 0 else 2,
                   })
    mod = kernel(unit(loop, core=None), buffers=buffers, annotations={
        "tl.buffer_versions_map": {storage.data: versions},
    })
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    mod["main"] = mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, sid


@pytest.mark.parametrize("sid_value", [0, 1], ids=["active", "empty-reader"])
@pytest.mark.parametrize("read_stage,clock_dtype,expected_width", [
    (1, "int32", 1), (2, "int32", 2), (1, "uint32", 2),
], ids=["same-stage-one-bit", "two-notifies-before-wait", "unsigned-clock-conservative"])
def test_scalar_admission_bounds_staged_distance_zero_width(sid_value, read_stage, clock_dtype, expected_width):
    before, sid = _scalar_admission_width_program(read_stage=read_stage, clock_dtype=clock_dtype)
    snapshot = tvm.ir.save_json(before)
    trace = _completion_trace(_lower_protocol(before, prepared=True), {sid: sid_value}, scalar_vector=True)
    assert tvm.ir.save_json(before) == snapshot
    precedes = _prove_binary_protocol(trace)
    _prove_scalar_read_epochs(trace, precedes, 34 if sid_value == 0 else 0, 17)
    flags = {node.flag for node in trace if node.kind == "set" and node.flag[0] == "V_S"}
    assert len(flags) == expected_width
    if read_stage == 2:
        # Preserve all counts while maliciously identifying the two live bits.
        collapsed = [replace(node, flag=("V_S", 0)) if node.flag and node.flag[0] == "V_S" else node
                     for node in trace]
        with pytest.raises(AssertionError, match="may be set again"):
            _prove_binary_protocol(collapsed)


@pytest.mark.parametrize("sid_value", [0, 1], ids=["active", "empty-owner"])
def test_scalar_admission_distance_one_control_has_one_seed_and_drain(sid_value):
    before, sid = _scalar_admission_width_program(distance=1)
    trace = _completion_trace(_lower_protocol(before, prepared=True), {sid: sid_value}, scalar_vector=True)
    _prove_scalar_read_epochs(trace, _prove_binary_protocol(trace), 34 if sid_value == 0 else 0,
                              34 if sid_value == 0 else 0)
    # Empty owners still execute all outer Control boundaries. Distinguish the
    # initial token from those notifications by using the first matching wait.
    first_wait = next(index for index, node in enumerate(trace) if node.kind == "wait" and node.flag[0] == "V_S")
    seeds = [node.flag for node in trace[:first_wait] if node.kind == "set" and node.flag[0] == "V_S"]
    assert len(seeds) == 1
    carried = seeds[0]
    assert sum(node.kind == "set" and node.flag == carried for node in trace) == 18
    assert sum(node.kind == "wait" and node.flag == carried for node in trace) == 18


def test_scalar_admission_keeps_distance_two_initial_tokens_independent():
    before, sid = _scalar_admission_width_program(distance=2)
    trace = _completion_trace(_lower_protocol(before, prepared=True), {sid: 0}, scalar_vector=True)
    precedes = _prove_binary_protocol(trace)
    writes, reads, last_write, last_read = 0, 0, {}, {}
    for index, node in enumerate(trace):
        if node.kind == "scalar_write":
            writes += 1
            if node.slot in last_read:
                assert precedes(last_read[node.slot], index), "Scalar overwrite before Vector read completes"
            last_write[node.slot] = index
        elif node.kind == "vector_read":
            reads += 1
            assert node.slot in last_write and precedes(last_write[node.slot], index), "Vector read before Scalar write completes"
            last_read[node.slot] = index
    assert (writes, reads) == (17, 17)
    first_write = next(index for index, node in enumerate(trace) if node.kind == "scalar_write")
    seeds = [node.flag for node in trace[:first_write] if node.kind == "set" and node.flag[0] == "V_S"]
    assert len(seeds) == len(set(seeds)) == 2
    collapsed = [replace(node, flag=("V_S", 0)) if node.flag and node.flag[0] == "V_S" else node
                 for node in trace]
    with pytest.raises(AssertionError, match="may be set again"):
        _prove_binary_protocol(collapsed)


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


def test_binary_oracle_accepts_real_scalar_wait_admission_only_for_flag_lifetime():
    flag = ("V_S", 0)
    trace = [_Completion("V", "set", flag), _Completion("S", "wait", flag),
             _Completion("V", "set", flag), _Completion("S", "wait", flag)]
    data_precedes = _prove_binary_protocol(trace)
    assert not data_precedes(1, 2), "Flag admission must not become a general data-completion edge"


@pytest.mark.parametrize("event", ["S_V", "V_MTE3"])
def test_binary_oracle_does_not_promote_non_scalar_wait_to_scalar_admission(event):
    source, destination = event.split("_")
    flag = (event, 0)
    trace = [_Completion(source, "set", flag), _Completion(destination, "wait", flag),
             _Completion(source, "set", flag), _Completion(destination, "wait", flag)]
    with pytest.raises(AssertionError, match="set again before its prior wait completes"):
        _prove_binary_protocol(trace)


def test_scalar_admission_does_not_complete_an_unretired_vector_read():
    ready, retired = ("S_V", 0), ("V_S", 0)
    mask = ("mask", 0)
    trace = [_Completion("S", "mask_write", slot=mask),
             _Completion("S", "mask_write", slot=("mask", 1)),
             _Completion("S", "set", ready), _Completion("V", "wait", ready),
             _Completion("V", "mask_read", slot=mask),
             _Completion("S", "mask_write", slot=mask),
             _Completion("V", "set", retired), _Completion("S", "wait", retired)]
    with pytest.raises(AssertionError, match="Mask overwritten before its Vector reader completes"):
        _prove_mask_epochs(trace, _prove_binary_protocol(trace), 1, 1, expected_ticks=0)


def test_scalar_wait_lifetime_oracle_rejects_two_overlapping_phases_on_one_id():
    flag = ("V_S", 0)
    trace = [_Completion("V", "set", flag), _Completion("V", "set", flag),
             _Completion("S", "wait", flag), _Completion("S", "wait", flag)]
    with pytest.raises(AssertionError, match="set again before its prior wait completes"):
        _prove_binary_protocol(trace)
