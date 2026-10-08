"""Guarded epilogues still need real binary flag reuse completion.

The ordinary DSL path is the small form of a final-output epilogue: periodic
active outer epochs load F32, narrow it and store F16.
An ordinary singleton is normalized away before scheduling; an explicitly
annotated one retains its control domain. Both need actual completion proof.
"""

from itertools import product

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_TARGET = {"kind": "ascend", "arch": "dav-2201"}
_COPY = "tl.tileop.ascend_copy"
_CAST = "tl.ascend_membase.float2half_rn"
_SHAPE = (16, 16)
_COUNT = 256


def _periodic_cast(epochs, period, preserve_child):
    @T.prim_func
    def main(Input: T.Tensor((epochs, 16, 16), "float32"),
             Output: T.Tensor((epochs, 16, 16), "float16")):
        with T.Kernel(1):
            value = T.alloc_shared((16, 16), "float32")
            narrowed = T.alloc_shared((16, 16), "float16")
            T.annotate_buffer_versions({value: 1, narrowed: 1})
            for epoch in T.serial(epochs):
                with T.Stage(0):
                    if epoch >= 2:
                        for sub in T.serial(1, annotations={"num_stages": 1} if preserve_child else {}):
                            if (epoch + 2) % period == period - 1:
                                T.copy(Input[epoch, :, :], value)
                                for row, column in T.Parallel(16, 16):
                                    narrowed[row, column] = T.cast(value[row, column], "float16")
                                T.copy(narrowed, Output[epoch, :, :])
    return main


def _lower(program):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AutoSchedule", "tl.InsertSync", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()],
                                  config={"tl.disable_shared_memory_reuse": True}):
        artifact = tilelang.lower(program, target=_TARGET, target_host="c",
                                  enable_host_codegen=False, enable_device_compile=False)
    assert set(snapshots) == {"tl.AutoSchedule", "tl.InsertSync", "tl.LowerScheduledTIR"}
    assert artifact.kernel_source
    return artifact.kernel_source, snapshots


def _region_cells(argument, env):
    buffer, origin, shape = _region(argument, env)
    return tuple((buffer.data, *indices) for indices in product(
        *(range(start, start + extent) for start, extent in zip(origin, shape))))


def _pointer_cells(argument, env):
    assert _name(argument) == "tl.access_ptr"
    load, count, _ = argument.args
    buffer = load.buffer
    shape = tuple(_value(extent, env) for extent in buffer.shape)
    origin = tuple(_value(index, env) for index in load.indices)
    assert not buffer.strides, "This fixture uses dense UB allocations"
    start = 0
    for index, extent in zip(origin, shape):
        start = start * extent + index
    result = []
    for offset in range(start, start + _value(count, env)):
        indices, remainder = [], offset
        for extent in reversed(shape):
            indices.append(remainder % extent)
            remainder //= extent
        assert remainder == 0
        result.append((buffer.data, *reversed(indices)))
    return tuple(result)


def _commands(mod):
    trace = []
    for call, env, loops in _trace(mod, core=1, lane=0):
        name = _name(call)
        coordinates = {var.name: index for var, index, _ in loops}
        assert len(coordinates) == len(loops)
        logical = (coordinates.get("epoch"),)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            kind = "set" if name == "tl.ascend_set_flag" else "wait"
            event_id = _value(call.args[1], env)
            assert 0 <= event_id < 8
            trace.append(_Command(source if kind == "set" else destination,
                                  kind, (event, event_id), logical=logical))
        elif name == "tl.ascend_pipe_barrier":
            pipe = call.args[0].value.removeprefix("PIPE_")
            # The shared oracle models exact source fences; these fixtures
            # need no kernel-wide all-pipe assumption to close a local bit.
            assert pipe != "ALL"
            trace.append(_Command(pipe, "barrier", logical=logical))
        elif name == _COPY:
            source, _, _ = _region(call.args[0], env)
            destination, _, _ = _region(call.args[1], env)
            assert (source.scope() == "global") != (destination.scope() == "global")
            pipe = "MTE2" if source.scope() == "global" else "MTE3"
            trace.append(_Command(pipe, "load" if pipe == "MTE2" else "store",
                                  reads=_region_cells(call.args[0], env),
                                  writes=_region_cells(call.args[1], env), logical=logical))
        else:
            assert name == _CAST, f"Unexpected epilogue operation: {name}"
            assert _value(call.args[2], env) == _COUNT
            trace.append(_Command("V", "cast", reads=_pointer_cells(call.args[1], env),
                                  writes=_pointer_cells(call.args[0], env), logical=logical))
    return trace


@pytest.mark.parametrize("epochs,period", [(12, 4), (15, 4), (11, 2)])
@pytest.mark.parametrize("preserve_child", [False, True], ids=["normalized", "annotated"])
def test_periodic_one_trip_epilogue_keeps_completion_and_binary_reuse(epochs, period, preserve_child):
    source, snapshots = _lower(_periodic_cast(epochs, period, preserve_child))
    scheduled = snapshots["tl.AutoSchedule"]
    assert any(loop.loop_var.name == "sub" and int(loop.extent) == 1
               for loop in nodes(scheduled, tirx.For)) == preserve_child
    assert calls(scheduled, _CAST), "Use the real native cast lowering"
    trace = _commands(snapshots["tl.LowerScheduledTIR"])
    expected = {(epoch,) for epoch in range(2, epochs)
                if (epoch + 2) % period == period - 1}
    for kind in ("cast", "store"):
        actual = [command.logical for command in trace if command.kind == kind]
        assert len(actual) == len(expected) and set(actual) == expected
    assert {command.logical for command in trace if command.kind == "load"} == expected
    assert any(command.flag and command.flag[0] == "V_MTE3" for command in trace)
    # Real source-fence completion / destination gates prove both the data
    # hazards and wait-before-next-set. No V FIFO or reuse edge is inserted.
    _prove_completions(trace)
    assert "asc_float2half_rn(" in source
    assert "asc_sync_notify(" in source and "asc_sync_wait(" in source


def _scheduled_cast(*, mismatch=False, disjoint=False):
    epochs = 10
    source = tirx.decl_buffer(_SHAPE, "float32", name="Input")
    destination = tirx.decl_buffer((epochs, *_SHAPE), "float16", name="Output")
    value = tirx.decl_buffer(_SHAPE, "float32", name="value", scope="shared.dyn")
    narrowed = tirx.decl_buffer((epochs, *_SHAPE) if disjoint else _SHAPE,
                                 "float16", name="narrowed", scope="shared.dyn")
    epoch, sub = (tirx.Var(name, "int32") for name in ("epoch", "sub"))
    indices = [epoch, 0, 0] if disjoint else [0, 0]
    pointer = lambda buffer, origin, effect: tirx.Call(
        "handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[tuple(origin)], _COUNT, effect])
    cast = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_CAST), [
        pointer(narrowed, indices, 2), pointer(value, [0, 0], 1), tirx.IntImm("uint32", _COUNT)]))
    writer_guard = epoch % 2 == 0 if mismatch else None
    reader_guard = epoch % 2 == 1 if mismatch else None
    child = unit(tirx.For(sub, 0, 1, tirx.ForKind.SERIAL, seq(
        unit(cast, guard=writer_guard),
        unit(copy(narrowed, destination, src_indices=indices,
                  src_shape=[1, *_SHAPE] if disjoint else list(_SHAPE),
                  dst_indices=[epoch, 0, 0], dst_shape=[1, *_SHAPE]), guard=reader_guard),
    )), core=None)
    mod = kernel(seq(unit(copy(source, value)),
                     unit(tirx.For(epoch, 0, epochs, tirx.ForKind.SERIAL, child), core=None)),
                 buffers=[value, narrowed], params=[source, destination])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(_TARGET))
    return mod


def test_one_trip_cannot_invent_reverse_completion_for_disjoint_outer_generations():
    before = _scheduled_cast(disjoint=True)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="no parent reverse completion path for single-iteration V->MTE3"):
        with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": True}):
            transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot


def test_distinct_periodic_activations_keep_parent_carried_protocol():
    before = _scheduled_cast(mismatch=True)
    snapshot = tvm.ir.save_json(before)
    with before["main"].attrs["target"], tvm.transform.PassContext(
            config={"tl.disable_shared_memory_reuse": True}):
        inserted = transform.InsertSync()(before)
        lowered = transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(inserted))
    assert tvm.ir.save_json(before) == snapshot
    trace = _commands(lowered)
    _prove_completions(trace)
    assert [command.logical for command in trace if command.kind == "cast"] == [
        (epoch,) for epoch in range(0, 10, 2)]
    assert [command.logical for command in trace if command.kind == "store"] == [
        (epoch,) for epoch in range(1, 10, 2)]
    # The inner invocation has mutually exclusive tasks, so it has no d=0
    # ready edge to defer. The ordinary parent ring must send and consume a
    # real tick in every epoch, including those without that pipe's data task.
    for event in ("V_MTE3", "MTE3_V"):
        for kind in ("set", "wait"):
            active = [command.logical for command in trace if command.kind == kind
                      and command.flag[0] == event and command.logical != (None,)]
            assert active == [(epoch,) for epoch in range(10)]
    latest = {}
    for index, command in enumerate(trace):
        if command.kind == "store":
            writers = {latest[cell] for cell in command.reads}
            assert len(writers) == 1
            writer = trace[writers.pop()]
            assert writer.kind == "cast"
            assert writer.logical == (command.logical[0] - 1,)
            assert len(command.reads) == _COUNT
        for cell in command.writes:
            latest[cell] = index
