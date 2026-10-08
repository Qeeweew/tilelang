"""A staged enclosing rectangle preserves a complete native operand ring.

The prepared iteration clock is materialized before stage translation. The
oracle follows real physical blocks and events through two enclosing shifts,
including odd native trips, rather than adding a parent completion edge.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_parent_counter_operand_cycle import _separate_c_planes
from testing.ascend.auto_schedule.test_parent_versioned_operand_war import _blocks, _fixture


def _staged_ancestor(stage, trips, *, side="both", kind="ordinary", load3d=False):
    before, facts = _fixture(side, trips, kind=kind, outer_stage=1, load3d=load3d)
    if kind == "guarded-child":
        # Isolate the rejected operand activation from unrelated cross-row C
        # phase reuse. This helper changes C planes, never the operand clock.
        before = _separate_c_planes(before)

    def post(node):
        if (isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
                and isinstance(node.body, tirx.For) and node.body.loop_var.name == "row_part"):
            return tirx.AttrStmt({**dict(node.node), "stage": stage}, node.attr_key, node.value, node.body)
        if isinstance(node, tirx.For) and node.loop_var.name == "owner":
            # The stage-zero sibling makes the parent shift real. Its entire
            # row rectangle, including fills, native events and payload
            # indices, runs at logical owner = physical owner - stage.
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            seq(unit(tirx.Evaluate(0), core=2), node.body),
                            node.thread_binding, node.annotations, node.step)

    function = before["main"]
    before = tvm.IRModule({"main": function.with_body(tirx.stmt_functor.ir_transform(
        function.body, None, post))})
    assert not nodes(before, tirx.BufferStore), "This certificate must use the prepared iteration clock"
    return before, {**facts, "parent_stage": stage}


def _lower(before, *, shift_ready=False):
    with before["main"].attrs["target"], tvm.transform.PassContext(
            config={"tl.disable_shared_memory_reuse": True}):
        inserted = transform.InsertSync()(before)
        if shift_ready:
            def post(node):
                if (isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
                        and not nodes(node.body, tirx.For)):
                    setters = calls(node.body, "tl.ascend_set_flag")
                    if len(setters) == 1 and setters[0].args[0].value == "MTE1_M":
                        # A deliberately independent endpoint shift is not a
                        # common invocation translation. Delay readiness past
                        # the W2 overwrite, without moving its load or MAD.
                        return tirx.AttrStmt({**dict(node.node), "stage": 3},
                                            node.attr_key, node.value, node.body)
            function = inserted["main"]
            inserted = tvm.IRModule({"main": function.with_body(tirx.stmt_functor.ir_transform(
                function.body, None, post))})
        return transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(inserted))


def _check_physical(physical, facts, *, check_overlap=False):
    trace, payload, slots, releases, root_flags = [], {}, {}, Counter(), Counter()
    mads, exports, logical_mads, first_loads = [], [], {}, {}
    for call, env, loops in _trace(physical, core=2, lane=0):
        name = _name(call)
        coordinates = {var.name: value for var, value, _ in loops}
        assert len(coordinates) == len(loops)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            event_id = _value(call.args[1], env)
            assert 0 <= event_id < 8
            trace.append(_Command(source if setting else destination,
                                  "set" if setting else "wait", (event, event_id)))
            if event == "M_MTE1":
                if not loops:
                    root_flags["set" if setting else "wait", event_id] += 1
                if "chunk" in coordinates:
                    releases[coordinates["epoch"], coordinates["owner"] - facts["parent_stage"],
                             coordinates["row_part"] - facts["outer_stage"],
                             "set" if setting else "wait"] += 1
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            source, source_min, _ = _region(call.args[0], env)
            destination, destination_min, _ = _region(call.args[1], env)
            reads, writes = _blocks(call.args[0], env), _blocks(call.args[1], env)
            assert len(reads) == len(writes)
            if source.scope() == "global":
                pipe = "MTE2"
                for block in reads:
                    payload.setdefault(block, (source.name, *block[1:]))
            elif destination.scope() in ("shared.l0a", "shared.l0b"):
                pipe = "MTE1"
                assert source.scope() == "shared.l1"
                for side in facts["selected"]:
                    if destination.data.same_as(facts["operands"][side].data):
                        logical = (coordinates["epoch"], coordinates["owner"] - facts["parent_stage"],
                                   coordinates["row_part"] - facts["outer_stage"], coordinates["chunk"])
                        slot = destination_min[0]
                        clock = ((logical[0] * 2 + logical[1]) * 3 + logical[2]) * facts["trips"] + logical[3]
                        assert slot == clock % 2, (logical, slot, clock)
                        assert destination_min[1:] == (0, 0)
                        assert (side, *logical) not in slots
                        slots[(side, *logical)] = slot
                        first_loads.setdefault(logical, len(trace))
            else:
                pipe = "FIX"
                assert source.scope() == "shared.l0c" and destination.scope() == "global"
                for block in reads:
                    assert payload[block] == ("result", *destination_min[:4], *block[-2:])
                exports.append(len(trace))
            for read, write in zip(reads, writes):
                assert read in payload, (name, source_min, read)
                payload[write] = payload[read]
            trace.append(_Command(pipe, "copy", reads=reads, writes=writes))
        elif name == "tl.tileop.gemm":
            assert _value(call.args[9], env)
            assert "unit_flag_ctrl" not in call.annotations
            logical = (coordinates["epoch"], coordinates["owner"] - facts["parent_stage"],
                       coordinates["row_part"] - facts["outer_stage"],
                       coordinates["chunk"] - facts["read_stage"])
            reads = []
            for side, argument in zip(("a", "b"), call.args[:2]):
                for block in _blocks(argument, env):
                    expected = (("Input" + side.upper(), *logical, *block[-2:])
                                if side in facts["selected"] else ("Fixed" + side.upper(), *block[-2:]))
                    assert payload[block] == expected, (logical, payload[block], expected)
                    reads.append(block)
            writes = _blocks(call.args[2], env)
            for block in writes:
                payload[block] = ("result", *logical, *block[-2:])
            mads.append(len(trace))
            logical_mads[logical] = len(trace)
            trace.append(_Command("M", "mad", reads=tuple(reads), writes=writes))
        else:
            assert name in ("tl.ascend_set_fixpipe_config", "tl.ascend_set_mmad_direction",
                            "tl.ascend_set_load3d_config"), name
            trace.append(_Command("S", "scalar"))
    require = _prove_completions(trace)
    assert len(mads) == len(exports) == 3 * 2 * 3 * facts["trips"] * facts["readers"]
    for index, export in enumerate(exports):
        require(3 * mads[index] + 2, 3 * export + 1)
        if index + 1 < len(mads):
            require(3 * export + 2, 3 * mads[index + 1] + 1)
    assert len(slots) == len(facts["selected"]) * 3 * 2 * 3 * facts["trips"]
    if facts["trips"] % 2:
        for side in facts["selected"]:
            assert slots[side, 0, 0, 0, 0] != slots[side, 0, 0, 1, 0]
            assert slots[side, 0, 0, 0, 0] != slots[side, 0, 1, 0, 0]
    if check_overlap:
        for epoch in range(3):
            for owner in range(2):
                for row in range(2):
                    previous = logical_mads[epoch, owner, row, facts["trips"] - 1]
                    following = first_loads[epoch, owner, row + 1, 0]
                    assert not set(trace[previous].reads).intersection(trace[following].writes)
                    with pytest.raises(AssertionError, match="Missing completion path"):
                        require(3 * previous + 2, 3 * following + 1)
    return releases, root_flags


@pytest.mark.parametrize("stage,trips,side,load3d", [
    (1, 3, "both", False), (2, 3, "a", True), (1, 4, "b", False), (2, 4, "both", False),
])
def test_staged_ancestor_keeps_physical_clock_payload_and_complete_binary_ring(stage, trips, side, load3d):
    before, facts = _staged_ancestor(stage, trips, side=side, load3d=load3d)
    original = tvm.ir.save_json(before)
    physical = _lower(before)
    assert tvm.ir.save_json(before) == original
    releases, root_flags = _check_physical(physical, facts, check_overlap=True)
    for epoch in range(3):
        for owner in range(2):
            for row in range(3):
                assert releases[epoch, owner, row, "wait"] == trips
                assert releases[epoch, owner, row, "set"] == trips
    assert root_flags == {("set", 0): 1, ("set", 1): 1, ("wait", 0): 1, ("wait", 1): 1}
    assert not [(call, loops) for call, _, loops in _trace(physical, core=2, lane=0)
                if _name(call) == "tl.ascend_pipe_barrier" and call.args[0].value == "PIPE_MTE1"]


@pytest.mark.parametrize("kind", ["partial-reader", "extra-reader", "view"])
def test_staged_ancestor_does_not_cover_incomplete_or_additional_operand_accesses(kind):
    before, facts = _staged_ancestor(1, 3, side="b", kind=kind)
    releases, _ = _check_physical(_lower(before), facts)
    assert any(releases[0, 0, row, "wait"] > facts["trips"] for row in range(3))


def test_guarded_staged_operand_keeps_the_specific_ordinary_protocol_diagnostic():
    before, _ = _staged_ancestor(1, 3, side="b", kind="guarded-child")
    with pytest.raises(tvm.error.InternalError, match=(
            "shifted carried events require an unconditional outermost lexical loop")) as error:
        _lower(before)
    assert ("pipe=MTE1->M, distance=1, reverse_completion=1, width=1, "
            "analysis_loop=row_part, flag_loop=row_part") in str(error.value)


def test_independent_ready_endpoint_stage_is_not_a_common_translation():
    before, facts = _staged_ancestor(1, 3)
    # The real unequal leaf stages load=0/MAD=1 retain their original support.
    # Moving only the ready notification beyond W2 breaks that retained cycle.
    with pytest.raises(AssertionError, match="Finite nested local protocol deadlocks"):
        _check_physical(_lower(before, shift_ready=True), facts)
