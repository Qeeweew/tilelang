"""One retained full-operand cycle protects guarded ancestor activations.

The prepared clock is advanced by real native owners, including re-entry after
an absent panel. Proofs below replay physical slots, payload and whole-pipeline
events; they never treat task issue order or equal token counts as completion.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tvm import tirx
from tilelang.layout import make_ascend_l0c_layout

from testing.ascend._ir import calls, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_parent_versioned_operand_war import (
    _blocks, _check_physical_lifetime, _fixture, _lower,
)


_EVENT = "M_MTE1"
_DIAGNOSTIC = "dav-2201 local binary flag:"


def _rewrite(mod, post):
    function = mod["main"]
    return tvm.IRModule({"main": function.with_body(
        tirx.stmt_functor.ir_transform(function.body, None, post))})


def _separate_c_planes(mod):
    # This is an operand-cycle regression, not a nested L0C phase protocol.
    # Each of the two owners and three row invocations gets its own 4KiB
    # accumulator plane (24KiB total, below the physical 128KiB L0C bank).
    # Native chunks still reuse that plane and retain real MAD/FIX completions;
    # the outer epoch also reuses all six planes through the ordinary protocol.
    old = calls(mod, "tl.tileop.gemm")[0].args[2].args[0].buffer
    accumulator = tirx.decl_buffer((2, 3, 32, 32), "float32", name="accum_planes", scope="shared.l0c")
    loops = {loop.loop_var.name: loop.loop_var for loop in nodes(mod, tirx.For)}
    indices = [loops["owner"], loops["row_part"], 0, 0]

    def post(node):
        if isinstance(node, tirx.Call) and _name(node) in ("tl.tileop.gemm", "tl.tileop.ascend_copy"):
            arguments = list(node.args)
            index = 2 if _name(node) == "tl.tileop.gemm" else 0
            if not arguments[index].args[0].buffer.same_as(old):
                return None
            arguments[index] = region(accumulator, 3 if index == 2 else 1, indices, [1, 1, 32, 32])
            return tirx.Call(node.dtype, node.op, arguments, annotations=node.annotations)
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            allocations = [accumulator if buffer.same_as(old) else buffer for buffer in node.alloc_buffers]
            annotations = dict(node.annotations)
            layouts = {buffer: layout for buffer, layout in annotations["layout_map"].items()
                       if not buffer.same_as(old)}
            layouts[accumulator] = make_ascend_l0c_layout(accumulator)
            annotations["layout_map"] = layouts
            return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint,
                              node.body, node.init, allocations, node.match_buffers, annotations)

    return _rewrite(mod, post)


def _counter_fixture(*args, **kwargs):
    before, facts = _fixture(*args, **kwargs)
    return _separate_c_planes(before), facts


def _guard_complete_owner(mod):
    row = next(loop.loop_var for loop in nodes(mod, tirx.For) if loop.loop_var.name == "row_part")

    def post(node):
        if isinstance(node, tirx.For) and node.loop_var.name == "chunk":
            # Guard the entire native owner, so writer/MAD/tick have exactly
            # the same activation. Its innermost trip count remains constant.
            return tirx.IfThenElse(row < 2, node, None)

    return _rewrite(mod, post)


def _bind_epochs(mod, facts, epochs):
    function = mod["main"]
    return tvm.IRModule({"main": function.with_body(tirx.stmt_functor.substitute(
        function.body, {facts["epoch_count"]: tirx.const(epochs, "int32")}))})


def _root_credits(mod):
    return Counter(_name(call) for call, _, loops in _trace(mod, core=2, lane=0)
                   if _name(call) in ("tl.ascend_set_flag", "tl.ascend_wait_flag")
                   and call.args[0].value == _EVENT and not loops)


@pytest.mark.parametrize("stage", [0, 1], ids=["ordinary-owner", "translated-whole-owner"])
@pytest.mark.parametrize("trips", [3, 4], ids=["odd-native", "even-native"])
@pytest.mark.parametrize("epochs", [0, 1, 3], ids=["empty-root", "single-root", "reentered-root"])
def test_whole_owner_guard_keeps_the_actual_counter_operand_cycle(stage, trips, epochs):
    before, facts = _counter_fixture("b", trips, kind="counter", outer_stage=stage, runtime_epochs=True)
    physical = _bind_epochs(_lower(_guard_complete_owner(before)), facts, epochs)
    counts = _check_physical_lifetime(physical, {**facts, "epochs": epochs, "rows": 2})
    for epoch in range(epochs):
        for owner in range(2):
            for row in range(stage, stage + 2):
                # Only the actual per-slot credits remain, not an enclosing
                # cache/output completion signal at first/last native chunks.
                assert counts[epoch, owner, row, "set"] == trips
                assert counts[epoch, owner, row, "wait"] == trips
    assert _root_credits(physical) == {"tl.ascend_set_flag": 2, "tl.ascend_wait_flag": 2}


@pytest.mark.parametrize("stage", [0, 2])
def test_canonical_counter_is_a_supported_cycle_without_an_owner_guard(stage):
    before, facts = _counter_fixture("b", 3, kind="counter", outer_stage=stage)
    physical = _lower(before)
    counts = _check_physical_lifetime(physical, facts)
    for epoch in range(3):
        for owner in range(2):
            for row in range(stage, stage + 3):
                assert counts[epoch, owner, row, "set"] == 3
                assert counts[epoch, owner, row, "wait"] == 3
    assert _root_credits(physical) == {"tl.ascend_set_flag": 2, "tl.ascend_wait_flag": 2}


def _common_stage_cycle(versions, *, fault=None):
    before, facts = _counter_fixture("b", 3, kind="counter", versions=versions, runtime_epochs=True)
    loops = {loop.loop_var.name: loop.loop_var for loop in nodes(before, tirx.For)}
    k, epoch = loops["chunk"], loops["epoch"]
    activation = tirx.all(k % 2 == 0, epoch % 2 == 0)

    def post(node):
        if not isinstance(node, tirx.For) or node.loop_var.name != "chunk":
            return None
        body = []
        # A real stage-zero sibling makes stage1 a fill/drain translation,
        # rather than allowing an all-equal-stage loop to normalize to zero.
        body.append(unit(tirx.Evaluate(0), core=2))
        for statement in node.body.seq:
            assert isinstance(statement, tirx.AttrStmt) and statement.attr_key == "tl.schedule_unit"
            task = statement.body
            assert isinstance(task, tirx.AttrStmt) and task.attr_key == "tl.ascend_task"
            tick = bool(nodes(task, tirx.BufferStore))
            loads = [call for call in calls(task, "tl.tileop.ascend_copy")
                     if call.args[1].args[0].buffer.scope() == "shared.l0b"]
            mad = bool(calls(task, "tl.tileop.gemm"))
            export = any(call.args[0].args[0].buffer.scope() == "shared.l0c"
                         for call in calls(task, "tl.tileop.ascend_copy"))
            stage = 0 if fault == "writer-stage" and loads else (
                2 if fault == "reader-stage" and (mad or export) else (
                    0 if fault == "tick-stage" and tick else 1))
            guard = tirx.all(activation, k == 0) if (
                fault == "writer-guard" and loads or fault == "tick-guard" and tick) else activation
            if tick:
                # Match PrepareMultiBuffer: its activation is inside the
                # atomic Scalar task, not in TaskNode's outer guard list.
                task = tirx.AttrStmt(task.node, task.attr_key, task.value,
                                    tirx.IfThenElse(guard, task.body, None))
            else:
                task = tirx.IfThenElse(guard, task, None)
            body.append(tirx.AttrStmt({"stage": stage}, "tl.schedule_unit", 1, task))
        annotations = dict(node.annotations)
        annotations["tl.storage_epoch_guard_map"] = {facts["operands"]["b"].data: activation}
        return tirx.For(node.loop_var, node.min, node.extent, node.kind, seq(*body),
                        thread_binding=node.thread_binding, annotations=annotations, step=node.step)

    return _rewrite(before, post), {**facts, "read_stage": 1}


def _check_common_stage_payload(physical, facts, epochs):
    # Replay actual physical addresses and the prepared counter, including a
    # whole empty epoch and holes within each owner. Common source order is
    # never introduced as an inter-pipe completion edge.
    _prove_completions([command for command, _ in _physical_commands(physical)])
    payload, fills, reads, exports = {}, {}, Counter(), Counter()
    for call, env, loops in _trace(physical, core=2, lane=0):
        name = _name(call)
        coordinates = {var.name: value for var, value, _ in loops}
        if name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            source, _, _ = _region(call.args[0], env)
            destination, origin, _ = _region(call.args[1], env)
            source_blocks, destination_blocks = _blocks(call.args[0], env), _blocks(call.args[1], env)
            if source.scope() == "global":
                for block in source_blocks:
                    payload.setdefault(block, (source.name, *block[1:]))
            elif destination.data.same_as(facts["operands"]["b"].data):
                logical = tuple(coordinates[key] for key in ("epoch", "owner", "row_part")) + (
                    coordinates["chunk"] - 1,)
                assert logical not in fills
                fills[logical] = origin[0] if facts["versions"] > 1 else 0
            elif source.scope() == "shared.l0c":
                logical = tuple(origin[:4])
                for block in source_blocks:
                    assert payload[block] == ("result", *logical, *block[-2:])
                exports[logical] += 1
            for read, write in zip(source_blocks, destination_blocks, strict=True):
                assert read in payload
                payload[write] = payload[read]
        elif name == "tl.tileop.gemm":
            logical = tuple(coordinates[key] for key in ("epoch", "owner", "row_part")) + (
                coordinates["chunk"] - 1,)
            assert _value(call.args[9], env)
            for side, argument in zip(("a", "b"), call.args[:2], strict=True):
                for block in _blocks(argument, env):
                    expected = (("InputB", *logical, *block[-2:]) if side == "b" else
                                ("FixedA", *block[-2:]))
                    assert payload[block] == expected
            for block in _blocks(call.args[2], env):
                payload[block] = ("result", *logical, *block[-2:])
            reads[logical] += 1
    expected = [(e, owner, row, k) for e in range(epochs) if e % 2 == 0
                for owner in range(2) for row in range(3) for k in (0, 2)]
    assert reads == exports == Counter(expected)
    assert set(fills) == set(expected)
    assert [fills[key] for key in expected] == [index % facts["versions"] for index in range(len(expected))]


@pytest.mark.parametrize("versions", [1, 2], ids=["single-slot", "two-slots"])
@pytest.mark.parametrize("epochs", [0, 1, 3], ids=["empty-root", "single-root", "empty-middle-epoch"])
def test_common_stage_and_gapped_activation_keep_the_actual_counter_cycle(versions, epochs):
    before, facts = _common_stage_cycle(versions)
    physical = _bind_epochs(_lower(before), facts, epochs)
    _check_common_stage_payload(physical, facts, epochs)
    assert _root_credits(physical) == {"tl.ascend_set_flag": versions, "tl.ascend_wait_flag": versions}


@pytest.mark.parametrize("fault", ["writer-stage", "reader-stage", "tick-stage", "writer-guard", "tick-guard"])
def test_independent_counter_stage_or_activation_keeps_the_l0_diagnostic(fault):
    before, _ = _common_stage_cycle(1, fault=fault)
    with pytest.raises(tvm.error.InternalError,
                       match=_DIAGNOSTIC + r".*pipe=(?:MTE1->M|M->MTE1),"):
        _lower(before)


def _counter_and_owner(mod):
    (owner,) = [loop for loop in nodes(mod, tirx.For) if loop.loop_var.name == "chunk"]
    clock = next(iter(owner.annotations["tl.multi_buffer_counter_map"].values()))
    return clock, owner


def _malformed(mod, kind):
    clock, owner = _counter_and_owner(mod)
    selected = next(call.args[1].args[0].buffer for call in calls(mod, "tl.tileop.ascend_copy")
                    if call.args[1].args[0].buffer.scope() == "shared.l0b")
    view = tirx.decl_buffer(selected.shape, selected.dtype, name="operand_view",
                           data=selected.data, scope=selected.scope())
    row = next(loop.loop_var for loop in nodes(mod, tirx.For) if loop.loop_var.name == "row_part")
    (mad,) = calls(mod, "tl.tileop.gemm")

    def post(node):
        if isinstance(node, tirx.BufferStore) and node.buffer.same_as(clock):
            if kind == "advance-two" and not isinstance(node.value, tirx.IntImm):
                return tirx.BufferStore(clock, clock[0] + 2, [0])
        if isinstance(node, tirx.Call) and _name(node) == "tl.tileop.ascend_copy":
            if not node.args[1].args[0].buffer.same_as(selected):
                return None
            arguments = list(node.args)
            if kind == "partial-writer":
                source = arguments[0].args[0]
                arguments[0] = region(source.buffer, 1, list(source.indices), [1, 16, 16])
                arguments[1] = region(selected, 2, shape=[16, 16])
            elif kind == "view-writer":
                arguments[1] = region(view, 2)
            else:
                return None
            return tirx.Call(node.dtype, node.op, arguments, annotations=node.annotations)
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_task":
            if kind == "conditional-writer" and isinstance(node.body, tirx.Evaluate):
                call = node.body.value
                if isinstance(call, tirx.Call) and _name(call) == "tl.tileop.ascend_copy" and call.args[1].args[0].buffer.same_as(selected):
                    return tirx.IfThenElse(row == 0, node, None)
        if isinstance(node, tirx.For) and node.loop_var.same_as(owner.loop_var):
            body = list(node.body.seq) if isinstance(node.body, tirx.SeqStmt) else [node.body]
            if kind == "early-advance":
                ticks = [index for index, statement in enumerate(body)
                         if nodes(statement, tirx.BufferStore)]
                assert len(ticks) == 1
                body.insert(0, body.pop(ticks[0]))
            elif kind == "extra-reader":
                export = next(index for index, statement in enumerate(body)
                              if any(call.args[0].args[0].buffer.scope() == "shared.l0c"
                                     for call in calls(statement, "tl.tileop.ascend_copy")))
                # Both readers precede the genuine final FIX. Do not let an
                # unrelated carried M->FIX boundary mask the L0 rejection.
                body.insert(export, unit(tirx.Evaluate(mad), core=2))
            else:
                return None
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, seq(*body),
                            thread_binding=node.thread_binding, annotations=node.annotations,
                            step=node.step)
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            if kind == "late-reset":
                extra = unit(tirx.BufferStore(clock, 0, [0]), core=2)
            elif kind == "outside-reader":
                # A separately active root read is not part of the native
                # generation, even though its typed footprint is the same.
                outside = tirx.stmt_functor.substitute(tirx.Evaluate(mad), {
                    loop.loop_var: tirx.const(0, loop.loop_var.dtype)
                    for loop in nodes(mod, tirx.For)})
                extra = unit(outside, core=2)
            elif kind == "counter-escape":
                extra = unit(tirx.Evaluate(clock[0]), core=2)
            else:
                return None
            return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint,
                              seq(node.body, extra), node.init, node.alloc_buffers,
                              node.match_buffers, node.annotations)

    return _rewrite(mod, post)


@pytest.mark.parametrize("kind", ["partial-writer", "view-writer", "extra-reader", "conditional-writer",
                                  "early-advance", "advance-two", "late-reset", "counter-escape",
                                  "outside-reader"])
def test_incomplete_or_noncanonical_counter_cycle_keeps_the_target_diagnostic(kind):
    before, _ = _counter_fixture("b", 3, kind="counter")
    before = _guard_complete_owner(_malformed(before, kind))
    diagnostic = ("Multi-buffer task is outside every owner" if kind == "outside-reader"
                  else _DIAGNOSTIC + r".*pipe=(?:MTE1->M|M->MTE1),")
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _lower(before)


def _physical_commands(mod):
    commands = []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            source, destination = call.args[0].value.split("_")
            setting = name == "tl.ascend_set_flag"
            command = _Command(source if setting else destination, "set" if setting else "wait",
                               (call.args[0].value, _value(call.args[1], env)))
        elif name == "tl.ascend_pipe_barrier":
            command = _Command(call.args[0].value.removeprefix("PIPE_"), "barrier")
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            source = call.args[0].args[0].buffer
            destination = call.args[1].args[0].buffer
            pipe = ("MTE2" if source.scope() == "global" else
                    "MTE1" if destination.scope() in ("shared.l0a", "shared.l0b") else "FIX")
            command = _Command(pipe, "copy", reads=_blocks(call.args[0], env), writes=_blocks(call.args[1], env))
        elif name == "tl.tileop.gemm":
            command = _Command("M", "mad", reads=(*_blocks(call.args[0], env), *_blocks(call.args[1], env)),
                               writes=_blocks(call.args[2], env))
        else:
            assert name in ("tl.ascend_set_fixpipe_config", "tl.ascend_set_mmad_direction")
            command = _Command("S", "scalar")
        commands.append((command, bool(loops)))
    return commands


def test_balanced_early_operand_release_does_not_prove_completion():
    before, facts = _counter_fixture("b", 3, kind="counter")
    physical = _lower(_guard_complete_owner(before))
    _check_physical_lifetime(physical, {**facts, "rows": 2})
    actual = _physical_commands(physical)
    commands = [command for command, _ in actual]
    _prove_completions(commands)
    first_mad = next(index for index, command in enumerate(commands) if command.kind == "mad")
    release = next(index for index, (command, inside) in enumerate(actual)
                   if index > first_mad and inside and command.kind == "set"
                   and command.flag[0] == _EVENT)
    broken = list(commands)
    broken.insert(first_mad, broken.pop(release))
    # Preserve every event ID and token count. The missing real MAD completion
    # must still invalidate the physical lifetime, not pass as source order.
    assert Counter(command.flag for command in broken if command.kind == "set") == Counter(
        command.flag for command in commands if command.kind == "set")
    with pytest.raises(AssertionError, match="Missing completion path.*kind='mad'.*kind='copy'"):
        _prove_completions(broken)
