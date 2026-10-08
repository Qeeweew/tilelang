"""A closed kernel keeps direction N without per-owner Scalar acquires.

The finite oracle joins real source fences and destination waits. It checks
ordinary operand/GM effects and binary reuse, without adding completion FIFO
or treating the selected unit C-data protocol as whole MAD/FIX completion.
"""

from itertools import product

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend._ir import calls, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_unit_flag_activation_guards import _replace_root
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _CONFIG, _COPY, _DIRECTION, _GEMM, _SET, _WAIT,
    _check_direction_boundaries, _fixture, _insert, _is_export, _name, _trace, _value,
)


def _blocks(region, env):
    assert _name(region) == "tl.region"
    load = region.args[0]
    origin = tuple(_value(index, env) for index in load.indices)
    shape = tuple(_value(extent, env) for extent in region.args[2:])
    assert len(origin) == len(shape) >= 2 and all(extent == 1 for extent in shape[:-2])
    assert all(value % 16 == 0 for value in (*origin[-2:], *shape[-2:]))
    return tuple((load.buffer.data, *origin[:-2], row, column)
                 for row, column in product(
                     range(origin[-2] // 16, (origin[-2] + shape[-2]) // 16),
                     range(origin[-1] // 16, (origin[-1] + shape[-1]) // 16)))


def _commands(records):
    commands = []
    for call, env, _ in records:
        name = _name(call)
        if name in (_SET, _WAIT):
            event = call.args[0].value
            source, destination = event.split("_")
            event_id = _value(call.args[1], env)
            assert 0 <= event_id < 8
            kind = "set" if name == _SET else "wait"
            command = _Command(source if kind == "set" else destination, kind, (event, event_id))
        elif name == "tl.ascend_pipe_barrier":
            command = _Command(call.args[0].value.removeprefix("PIPE_"), "barrier")
        elif name == _GEMM:
            # Unit-selected C relations remain outside this completion graph.
            # A/B reads are ordinary whole-completion obligations throughout.
            selected = "unit_flag_ctrl" in call.annotations and _value(call.annotations["unit_flag_ctrl"], env) != 0
            c_blocks = () if selected else _blocks(call.args[2], env)
            c_reads = c_blocks if not _value(call.args[9], env) else ()
            command = _Command("M", "mad", reads=(*_blocks(call.args[0], env),
                                                   *_blocks(call.args[1], env), *c_reads),
                               writes=c_blocks)
        elif name == _COPY:
            src, dst = [arg.args[0].buffer for arg in call.args[:2]]
            if src.scope() == "global":
                pipe = "MTE2"
            elif dst.scope() in ("shared.l0a", "shared.l0b"):
                assert src.scope() == "shared.l1"
                pipe = "MTE1"
            else:
                assert src.scope() == "shared.l0c" and dst.scope() == "global"
                pipe = "FIX"
            selected = "unit_flag_ctrl" in call.annotations and _value(call.annotations["unit_flag_ctrl"], env) != 0
            reads = () if src.scope() == "shared.l0c" and selected else _blocks(call.args[0], env)
            command = _Command(pipe, "copy", reads=reads, writes=_blocks(call.args[1], env))
        else:
            assert name in (_CONFIG, _DIRECTION), f"Unexpected finite fixture operation: {name}"
            command = _Command("S", "scalar")
        commands.append(command)
    return commands


def _activations(records, facts):
    result = {}
    for index, (call, env, _) in enumerate(records):
        if facts["epoch"] not in env:
            continue
        owner = _value(facts["owner"], env) if facts["owner"] in env else 0
        key = (_value(facts["epoch"], env), owner)
        result.setdefault(key, []).append(index)
    # Some ancestor event shells execute in inactive epochs. They are not
    # native stream activations and cannot substitute for active C work.
    return {key: positions for key, positions in result.items()
            if any(_name(records[i][0]) == _GEMM for i in positions)}


def _prove_kernel(records, facts, *, invariant=True):
    commands = _commands(records)
    require = _prove_completions(commands)
    activations = _activations(records, facts)
    if activations:
        assert _check_direction_boundaries(records, facts, activations) == invariant
    directions = [(i, call.args[0].value, loops) for i, (call, _, loops) in enumerate(records)
                  if _name(call) == _DIRECTION]
    if invariant:
        assert [(name, loops) for _, name, loops in directions] == [("n", ()), ("m", ())]
        setup, restore = directions[0][0], directions[1][0]
        for index, (call, _, _) in enumerate(records):
            if _name(call) == _GEMM or _is_export(call, facts):
                require(3 * setup + 2, 3 * index + 1)
                require(3 * index + 2, 3 * restore + 1)
        assert not any(command.kind == "wait" and command.pipe == "S" and records[i][2]
                       for i, command in enumerate(commands) if setup < i < restore)
        terminal = [i for i, command in enumerate(commands)
                    if command.kind == "wait" and command.flag[0] == "M_S" and not records[i][2]]
        assert len(terminal) == 1 and setup < terminal[0] < restore
    previous = None
    for positions in activations.values():
        mads = [i for i in positions if _name(records[i][0]) == _GEMM]
        fixes = [i for i in positions if _is_export(records[i][0], facts)]
        assert mads and fixes
        if invariant:
            barriers = [i for i in positions
                        if _name(records[i][0]) == "tl.ascend_pipe_barrier"
                        and records[i][0].args[0].value.removeprefix("PIPE_") == "M"]
            assert len(barriers) == 1 and barriers[0] < mads[0]
            inner_names = {facts[key].name for key in ("result", "k")}
            inner_names |= {"later_" + name for name in inner_names}
            assert not any(loop.name in inner_names for loop in records[barriers[0]][2])
        if previous is not None:
            old_mads, old_fixes = previous
            # Both endpoints retire. C's unit-level data protection does not
            # prove either of these whole-pipe ancestor/lifetime relations.
            for index in old_mads:
                require(3 * index + 2, 3 * fixes[0] + 1)
                require(3 * index + 2, 3 * mads[0] + 1)
            for index in old_fixes:
                require(3 * index + 2, 3 * mads[0] + 1)
        previous = mads, fixes
    return require


@pytest.mark.parametrize("native_k", [1, 3])
@pytest.mark.parametrize("whole_guard", [False, True])
def test_closed_kernel_direction_preserves_binary_and_dual_completion(native_k, whole_guard):
    before, facts = _fixture(native_k=native_k, whole_guard=whole_guard)
    original = tvm.ir.save_json(before)
    after = _insert(before)
    assert tvm.ir.save_json(before) == original
    records = _trace(after)
    _prove_kernel(records, facts)
    transactions = {}
    for call, env, _ in records:
        if _name(call) == _GEMM or _is_export(call, facts):
            key = tuple(_value(facts[name], env) for name in ("epoch", "owner", "result"))
            transactions.setdefault(key, []).append((call, env))
    assert set(transactions) == set(product(range(3), range(2), range(4)))
    for transaction in transactions.values():
        assert [_value(call.annotations["unit_flag_ctrl"], env) for call, env in transaction] == (
            [2] * (native_k - 1) + [3, 3])
        assert [_value(call.args[9], env) for call, env in transaction[:-1]] == [1] + [0] * (native_k - 1)


def _inactive_streams(*, native_k, later_active):
    before, facts = _fixture(native_k=native_k, activation_guard="direct", persistent_inputs=True)
    changed = []

    def mutate(node):
        if (isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
                and isinstance(node.body, tirx.IfThenElse) and isinstance(node.body.then_case, tirx.For)
                and node.body.then_case.loop_var.same_as(facts["result"])):
            guarded = node.body
            changed.append(guarded)
            inactive = unit(guarded.then_case, core=None, guard=facts["epoch"] < 0)
            if not later_active:
                return inactive
            mapping = {facts["result"]: tirx.Var("later_result", "int32"),
                       facts["k"]: tirx.Var("later_native_k", "int32")}

            def freshen(loop):
                if isinstance(loop, tirx.For) and loop.loop_var in mapping:
                    return tirx.For(mapping[loop.loop_var], loop.min, loop.extent, loop.kind,
                                    loop.body, annotations=loop.annotations)

            later = tirx.stmt_functor.ir_transform(guarded.then_case, None, freshen)
            later = tirx.stmt_functor.substitute(later, mapping)
            return seq(inactive, unit(later, core=None, guard=guarded.condition))
        if (not later_active and isinstance(node, tirx.IfThenElse)
                and isinstance(node.then_case, tirx.For)
                and node.then_case.loop_var.name == "reader"):
            return tirx.IfThenElse(tirx.const(False, "bool"), node.then_case, None)

    body = tirx.stmt_functor.ir_transform(before["main"].body.block.body, None, mutate)
    assert len(changed) == 1
    return _replace_root(before, body), facts


@pytest.mark.parametrize("native_k", [1, 3])
@pytest.mark.parametrize("later_active", [False, True], ids=["empty-active-domain", "first-stream-inactive"])
def test_kernel_setup_is_consumed_without_first_stream_activation(native_k, later_active):
    before, facts = _inactive_streams(native_k=native_k, later_active=later_active)
    after = _insert(before)
    records = _trace(after)
    _prove_kernel(records, facts)
    mads = [(call, env, loops) for call, env, loops in records if _name(call) == _GEMM]
    assert len(mads) == (8 * native_k if later_active else 0)
    assert len([call for call, _, _ in records if _is_export(call, facts)]) == (8 if later_active else 0)
    if later_active:
        assert all(any(loop.name == "later_result" for loop in loops) for _, _, loops in mads)
        assert {_value(facts["epoch"], env) for _, env, _ in mads} == {1, 2}
    # The setup's real destination acquires must execute even when every
    # guarded result activation is empty. Root seed/drain balance alone is
    # insufficient if the setup waits were placed inside that first guard.
    for event in ("S_M", "S_FIX"):
        waits = [(call, loops) for call, _, loops in records
                 if _name(call) == _WAIT and call.args[0].value == event]
        assert any(not loops for _, loops in waits), event


@pytest.mark.parametrize("native_k", [1, 3])
def test_unselected_root_mad_keeps_scoped_direction_and_real_completion(native_k):
    before, facts = _fixture(native_k=native_k)
    root = before["main"].body.block
    selected = calls(before, _GEMM)[0]
    args = list(selected.args)
    args[9] = tirx.const(True, "bool")
    unselected = unit(tirx.Evaluate(tirx.Call(selected.dtype, selected.op, args,
                                            annotations=selected.annotations)), core=2)
    before = _replace_root(before, seq(root.body, unselected))
    after = _insert(before)
    records = _trace(after)
    _prove_kernel(records, facts, invariant=False)
    assert len(calls(after, _DIRECTION)) == 2
    directions = [(i, call.args[0].value) for i, (call, _, _) in enumerate(records)
                  if _name(call) == _DIRECTION]
    assert [name for _, name in directions] == ["n", "m"] * 6
    root_mads = [i for i, (call, _, loops) in enumerate(records) if _name(call) == _GEMM and not loops]
    assert len(root_mads) == 1
    last = root_mads[0]
    assert "unit_flag_ctrl" not in records[last][0].annotations
    require = _prove_completions(_commands(records))
    require(3 * directions[-1][0] + 2, 3 * last + 1)
    last_fix = max(i for i, (call, _, _) in enumerate(records) if _is_export(call, facts))
    require(3 * last_fix + 2, 3 * last + 1)


def test_explicit_direction_keeps_the_ordinary_protocol():
    before, facts = _fixture(native_k=3)
    explicit = unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_DIRECTION),
                                           [tirx.StringImm("m")])), core=2)
    before = _replace_root(before, seq(explicit, before["main"].body.block.body))
    after = _insert(before)
    directions = calls(after, _DIRECTION)
    assert len(directions) == 1 and directions[0].args[0].value == "m"
    assert all("unit_flag_ctrl" not in call.annotations
               for call in calls(after, _GEMM) + calls(after, _COPY))
    records = _trace(after)
    _prove_completions(_commands(records))
    assert sum(_name(call) == _GEMM for call, _, _ in records) == 72
    assert sum(_is_export(call, facts) for call, _, _ in records) == 24


def test_registered_opaque_scalar_keeps_scoped_direction():
    # This is a registered TGlobalSymbol, not call_extern. A name/registration
    # alone cannot establish that the operation preserves CTRL51.
    name = "testing.ascend.unit_direction_opaque_state"
    tvm.ir.register_op_attr(name, "TGlobalSymbol", "test_unit_direction_opaque_state")
    tvm.ir.register_op_attr(name, "TCallEffectKind", tirx.IntImm("int32", 3))
    opaque_op = tvm.ir.Op.get(name)
    opaque_op.set_num_inputs(0)
    before, facts = _fixture(native_k=3)
    opaque = unit(tirx.Evaluate(tirx.Call("void", opaque_op, [])), core=2)
    before = _replace_root(before, seq(opaque, before["main"].body.block.body))
    after = _insert(before)
    assert len(calls(after, name)) == 1
    records = _trace(after)
    activations = _activations(records, facts)
    assert len(activations) == 6
    assert not _check_direction_boundaries(records, facts, activations)
    directions = [(i, call.args[0].value, loops) for i, (call, _, loops) in enumerate(records)
                  if _name(call) == _DIRECTION]
    assert [direction for _, direction, _ in directions] == ["n", "m"] * 6
    assert all(loops for _, _, loops in directions)
    assert next(i for i, (call, _, _) in enumerate(records) if _name(call) == name) < directions[0][0]
    mads = [call for call, _, _ in records if _name(call) == _GEMM]
    assert len(mads) == 72 and all("unit_flag_ctrl" in call.annotations for call in mads)
