"""Target legality of local-event compaction and the dav-3510 mutex spill."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, seq


EVENT = "MTE1_MTE2"


def _pair(event, event_id):
    if isinstance(event_id, int):
        event_id = tirx.IntImm("int32", event_id)
    return seq(*(
        tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.ascend_" + operation + "_flag"),
                               [tirx.StringImm(event), event_id]))
        for operation in ("set", "wait")
    ))


def _points(ids, event=EVENT):
    return seq(*(_pair(event, event_id) for event_id in ids))


def _range(base, count, event=EVENT):
    iteration = tirx.Var("iteration", "int32")
    event_id = tirx.Var("event_id", "int32")
    return tirx.For(iteration, 0, count, tirx.ForKind.SERIAL,
                    seq(tirx.Bind(event_id, base + iteration), _pair(event, event_id)))


def _module(body, arch="dav-2201", target_key="arch"):
    func = tirx.PrimFunc([], body).with_attr("global_symbol", "main")
    if arch is not None:
        target = tvm.target.Target("ascend" if arch == "default" else {"kind": "ascend", target_key: arch})
        func = func.with_attr("target", target)
    return tvm.IRModule({"main": func})


def _trace(mod):
    """Enumerate only these finite scalar loops and their actual event calls."""
    trace = []
    analyzer = tvm.arith.Analyzer()

    def integer(expr, values):
        result = analyzer.simplify(tirx.stmt_functor.substitute(expr, values))
        assert isinstance(result, tirx.IntImm), result
        return int(result)

    def visit(stmt, values):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, values)
        elif isinstance(stmt, tirx.For):
            start = integer(stmt.min, values)
            for iteration in range(start, start + integer(stmt.extent, values)):
                visit(stmt.body, {**values, stmt.loop_var: tirx.IntImm("int32", iteration)})
        elif isinstance(stmt, tirx.Bind):
            values[stmt.var] = tirx.IntImm(stmt.var.dtype, integer(stmt.value, values))
        elif isinstance(stmt, tirx.SBlock):
            visit(stmt.body, dict(values))
        elif isinstance(stmt, tirx.Evaluate):
            call = stmt.value
            assert isinstance(call, tirx.Call)
            trace.append((call.op.name, str(call.args[0].value), integer(call.args[1], values),
                          integer(call.args[2], values) if len(call.args) == 3 else None))
        else:
            raise AssertionError(type(stmt))

    visit(mod["main"].body, {})
    return trace


def _assert_compacted(before, after, slots):
    assert not calls(after, "tl.ascend_get_buf")
    assert not calls(after, "tl.ascend_rls_buf")
    original = _trace(before)
    compacted = _trace(after)
    assert len(original) == len(compacted)
    renaming = {}
    for (old_op, old_event, old_id, _), (new_op, new_event, new_id, _) in zip(original, compacted):
        assert (old_op, old_event) == (new_op, new_event)
        assert 0 <= new_id < 8
        key = (old_event, old_id)
        assert renaming.setdefault(key, new_id) == new_id
    assert len(set(renaming.values())) == len(renaming) == slots
    assert set(renaming.values()) == set(range(slots))


def _assert_overflow(mod, *, slots, ranges):
    with pytest.raises(ValueError) as error:
        transform.RewriteFlagToBuf()(mod)
    message = str(error.value)
    assert "dav-2201" in message
    assert "hard_event '" + EVENT + "'" in message
    assert f"requires {slots} local event slots from ranges " in message
    assert ranges in message
    assert "limit is 8" in message
    assert "Mutex spill requires dav-3510" in message


@pytest.mark.parametrize("ids", [tuple(range(8)), (0, 2, 7)], ids=["eight-slots", "legal-sparse"])
def test_2201_legal_flags_remain_unchanged(ids):
    before = _module(_points(ids))
    after = transform.RewriteFlagToBuf()(before)
    tvm.ir.assert_structural_equal(after, before)
    assert not calls(after, "tl.ascend_get_buf")
    assert not calls(after, "tl.ascend_rls_buf")


def test_2201_sparse_points_compact_without_spilling():
    before = _module(_points((0, 2, 4, 6, 8, 10, 12, 14)))
    _assert_compacted(before, transform.RewriteFlagToBuf()(before), slots=8)


def test_2201_bound_dynamic_sparse_ranges_preserve_each_pair():
    before = _module(seq(_range(10, 4), _range(20, 4)))
    _assert_compacted(before, transform.RewriteFlagToBuf()(before), slots=8)


def test_2201_overlapping_dynamic_ranges_count_the_union_once():
    before = _module(seq(_range(10, 5), _range(13, 5)))
    _assert_compacted(before, transform.RewriteFlagToBuf()(before), slots=8)


def test_2201_nine_constant_slots_have_target_capacity_diagnostic():
    _assert_overflow(_module(_points(range(9))), slots=9, ranges="[8..8]")


def test_2201_nested_bound_dynamic_range_has_target_capacity_diagnostic():
    outer = tirx.Var("outer", "int32")
    inner = tirx.Var("inner", "int32")
    event_id = tirx.Var("event_id", "int32")
    body = tirx.For(outer, 0, 3, tirx.ForKind.SERIAL,
                    tirx.For(inner, 0, 3, tirx.ForKind.SERIAL,
                             seq(tirx.Bind(event_id, outer * 3 + inner + 4), _pair(EVENT, event_id))))
    _assert_overflow(_module(body, target_key="mcpu"), slots=9, ranges="[4..12]")


def test_2201_hard_events_have_independent_eight_slot_namespaces():
    before = _module(seq(_points(range(8)), _points(range(8), event="MTE2_MTE1")))
    after = transform.RewriteFlagToBuf()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_2201_existing_conservative_core_union_is_not_silently_reused():
    cube = tirx.SBlock([], [], [], "CUBE", _points(range(5)))
    vector = tirx.SBlock([], [], [], "VECTOR", _points(range(5, 9)))
    # This pass has no lifetime/core-disjointness proof. Preserve its existing
    # per-hard-event union instead of granting new reuse while adding the gate.
    _assert_overflow(_module(seq(cube, vector)), slots=9, ranges="[8..8]")


@pytest.mark.parametrize("arch", ["dav-3510", "default", None], ids=["explicit-3510", "default-target", "unbound"])
def test_3510_and_legacy_unbound_inputs_keep_dynamic_mutex_spill(arch):
    # One indivisible nine-slot range must spill in full; splitting it across
    # the flag/mutex boundary would change dynamic event identity.
    after = transform.RewriteFlagToBuf()(_module(_range(4, 9), arch=arch))
    assert not calls(after, "tl.ascend_set_flag")
    assert not calls(after, "tl.ascend_wait_flag")
    assert len(calls(after, "tl.ascend_get_buf")) == 2
    assert len(calls(after, "tl.ascend_rls_buf")) == 2
    trace = _trace(after)
    expected = []
    for event_id in range(9):
        for pipe, mode in (("PIPE_MTE1", 1), ("PIPE_MTE2", 0)):
            expected.extend((operation, pipe, event_id, mode)
                            for operation in ("tl.ascend_get_buf", "tl.ascend_rls_buf"))
    assert trace == expected
