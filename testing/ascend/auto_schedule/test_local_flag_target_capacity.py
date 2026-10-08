"""Target legality of local-event compaction and the dav-3510 mutex spill."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


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


def _trace(mod, *, include_core=False):
    """Enumerate only these finite scalar loops and their actual event calls."""
    trace = []
    analyzer = tvm.arith.Analyzer()

    def integer(expr, values):
        result = analyzer.simplify(tirx.stmt_functor.substitute(expr, values))
        assert isinstance(result, tirx.IntImm), result
        return int(result)

    def visit(stmt, values, core=None):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, values, core)
        elif isinstance(stmt, tirx.For):
            start = integer(stmt.min, values)
            for iteration in range(start, start + integer(stmt.extent, values)):
                visit(stmt.body, {**values, stmt.loop_var: tirx.IntImm("int32", iteration)}, core)
        elif isinstance(stmt, tirx.Bind):
            values[stmt.var] = tirx.IntImm(stmt.var.dtype, integer(stmt.value, values))
        elif isinstance(stmt, tirx.SBlock):
            if stmt.name_hint in ("CUBE", "VECTOR"):
                core = str(stmt.name_hint)
            visit(stmt.body, dict(values), core)
        elif isinstance(stmt, tirx.AttrStmt):
            visit(stmt.body, dict(values), core)
        elif isinstance(stmt, tirx.Evaluate):
            call = stmt.value
            assert isinstance(call, tirx.Call)
            record = (call.op.name, str(call.args[0].value), integer(call.args[1], values),
                      integer(call.args[2], values) if len(call.args) == 3 else None)
            trace.append((core, *record) if include_core else record)
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


def _assert_overflow(mod, *, slots, ranges, event=EVENT, bank=None):
    with pytest.raises(ValueError) as error:
        transform.RewriteFlagToBuf()(mod)
    message = str(error.value)
    assert "dav-2201" in message
    assert "hard_event '" + event + "'" in message
    assert f"requires {slots} local event slots from ranges " in message
    assert ranges in message
    assert "limit is 8" in message
    assert "allocated layout, not a minimum slot requirement" in message
    if bank is not None:
        assert f"in {bank} bank" in message
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


def _core(name, body):
    return tirx.SBlock([], [], [], name, body)


@pytest.mark.parametrize("event", ["S_MTE2", "MTE2_S"])
@pytest.mark.parametrize("vector_base", [0, 8], ids=["already-legal", "independent-eight-slots"])
def test_2201_concrete_core_banks_each_have_eight_slots(event, vector_base):
    # Both directions are supported on AIC and AIV, unlike MTE1_MTE2.
    before = _module(seq(_core("CUBE", _points(range(8), event)),
                         _core("VECTOR", _points(range(vector_base, vector_base + 8), event))))
    after = transform.RewriteFlagToBuf()(before)
    assert not calls(after, "tl.ascend_get_buf")
    assert not calls(after, "tl.ascend_rls_buf")
    expected = _module(seq(_core("CUBE", _points(range(8), event)),
                           _core("VECTOR", _points(range(8), event))))
    assert _trace(after, include_core=True) == _trace(expected, include_core=True)
    if vector_base == 0:
        tvm.ir.assert_structural_equal(after, before)


def test_2201_sparse_banks_rename_shared_call_occurrences_independently():
    event = "S_MTE2"
    shared_pair = _pair(event, 20)
    before = _module(seq(
        _core("CUBE", _core("ordinary-inner-block", seq(_pair(event, 10), shared_pair))),
        _core("VECTOR", _core("ordinary-inner-block", seq(shared_pair, _pair(event, 30)))),
    ))
    after = transform.RewriteFlagToBuf()(before)
    renaming = {}
    for old, new in zip(_trace(before, include_core=True), _trace(after, include_core=True)):
        old_core, old_op, old_event, old_id, old_mode = old
        new_core, new_op, new_event, new_id, new_mode = new
        assert (old_core, old_op, old_event, old_mode) == (new_core, new_op, new_event, new_mode)
        assert renaming.setdefault((old_core, old_id), new_id) == new_id
    assert renaming == {("CUBE", 10): 0, ("CUBE", 20): 1, ("VECTOR", 20): 0, ("VECTOR", 30): 1}
    assert len(_trace(before)) == len(_trace(after)) == 8
    assert not calls(after, "tl.ascend_get_buf")


def test_2201_shared_call_keeps_occurrence_bounds_after_an_unchanged_bank():
    iteration = tirx.Var("iteration", "int32")
    shared_pair = _pair("S_MTE2", iteration)
    before = _module(seq(
        _core("CUBE", tirx.For(iteration, 0, 3, tirx.ForKind.SERIAL, shared_pair)),
        _core("VECTOR", tirx.For(iteration, 8, 3, tirx.ForKind.SERIAL, shared_pair)),
    ))
    after = transform.RewriteFlagToBuf()(before)
    trace = _trace(after, include_core=True)
    for core in ("CUBE", "VECTOR"):
        for operation in ("set", "wait"):
            assert [row[3] for row in trace if row[0] == core and row[1] == "tl.ascend_" + operation + "_flag"] == [0, 1, 2]
    assert len(trace) == 12
    assert not calls(after, "tl.ascend_get_buf")


@pytest.mark.parametrize("bank", ["CUBE", "VECTOR"])
def test_2201_one_bank_overflow_still_counts_all_its_regions(bank):
    other = "VECTOR" if bank == "CUBE" else "CUBE"
    body = seq(_core(bank, _points(range(5), "S_MTE2")),
               _core(other, _pair("S_MTE2", 0)),
               _core(bank, _points(range(5, 9), "S_MTE2")))
    _assert_overflow(_module(body), slots=9, ranges="[8..8]", event="S_MTE2", bank=bank)


@pytest.mark.parametrize("shared_wrapper", [False, True], ids=["unscoped", "broadcast-task"])
def test_2201_unknown_owner_keeps_same_event_in_one_conservative_union(shared_wrapper):
    shared = _pair("S_MTE2", 0)
    if shared_wrapper:
        shared = tirx.AttrStmt({"core_mask": tirx.IntImm("int64", 3)}, "tl.ascend_task", 1, shared)
    body = seq(_core("CUBE", _points(range(5), "S_MTE2")),
               _core("VECTOR", _points(range(5, 9), "S_MTE2")), shared)
    _assert_overflow(_module(body), slots=9, ranges="[8..8]", event="S_MTE2")


def test_2201_unknown_owner_uses_the_same_renaming_as_known_core_calls():
    before = _module(seq(_core("CUBE", _pair("S_MTE2", 10)),
                         _core("VECTOR", _pair("S_MTE2", 20)), _pair("S_MTE2", 20)))
    after = transform.RewriteFlagToBuf()(before)
    expected = _module(seq(_core("CUBE", _pair("S_MTE2", 0)),
                           _core("VECTOR", _pair("S_MTE2", 1)), _pair("S_MTE2", 1)))
    assert _trace(after, include_core=True) == _trace(expected, include_core=True)
    assert not calls(after, "tl.ascend_get_buf")


def test_2201_unknown_owner_only_disables_splitting_for_its_hard_event():
    before = _module(seq(_core("CUBE", _points(range(8), "S_MTE2")),
                         _core("VECTOR", _points(range(8, 16), "S_MTE2")), _pair("MTE2_S", 16)))
    after = transform.RewriteFlagToBuf()(before)
    assert not calls(after, "tl.ascend_get_buf")
    assert all(0 <= row[2] < 8 for row in _trace(after))


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_insert_sync_allocates_real_scalar_dma_protocols_in_target_core_banks(arch):
    source = tirx.decl_buffer((16, 16), "float32", name="source")
    output = tirx.decl_buffer((4,), "float32", name="output")
    buffers, tasks = [], []
    for core, scope in ((2, "shared.l1"), (1, "shared.dyn")):
        for index in range(2):
            local = tirx.decl_buffer((16, 16), "float32", name=f"local_{core}_{index}", scope=scope)
            buffers.append(local)
            # A real scalar read must complete before MTE2 overwrites that
            # storage. Distinct storages retain two protocols in each bank.
            tasks.append(unit(tirx.BufferStore(output, local[0, 0], [len(buffers) - 1]), core=core))
            tasks.append(unit(copy(source, local), core=core))
    before = kernel(seq(*tasks), buffers=buffers, params=[source, output])
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    resolved = transform.ResolveCore()(before)
    after = transform.InsertSync()(resolved)
    by_core = {}
    for statement, parents in statements(after):
        if not isinstance(statement, tirx.Evaluate) or not isinstance(statement.value, tirx.Call):
            continue
        call = statement.value
        if call.op.name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag") or call.args[0].value != "S_MTE2":
            continue
        owner = next(parent for parent in reversed(parents) if isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.ascend_task")
        core = int(owner.node["core_mask"])
        assert core in (1, 2) and isinstance(call.args[1], tirx.IntImm)
        by_core.setdefault((core, call.op.name), []).append(int(call.args[1]))
    for core in (1, 2):
        sets, waits = (by_core[core, "tl.ascend_" + operation + "_flag"] for operation in ("set", "wait"))
        assert len(sets) == len(waits) == len(set(sets)) == 2
        assert set(sets) == set(waits) <= set(range(8))
    cube = set(by_core[2, "tl.ascend_set_flag"])
    vector = set(by_core[1, "tl.ascend_set_flag"])
    assert (cube == vector) if arch == "dav-2201" else cube.isdisjoint(vector)


@pytest.mark.parametrize("arch", ["dav-3510", "default", None], ids=["explicit-3510", "default-target", "unbound"])
def test_3510_core_wrappers_preserve_function_wide_mutex_spill(arch):
    body = seq(_core("CUBE", _points(range(5), "S_MTE2")),
               _core("VECTOR", _points(range(5, 9), "S_MTE2")))
    after = transform.RewriteFlagToBuf()(_module(body, arch=arch))
    assert len(calls(after, "tl.ascend_get_buf")) == 2
    assert len(calls(after, "tl.ascend_rls_buf")) == 2
    assert {row[2] for row in _trace(after) if row[0] == "tl.ascend_set_flag"} == set(range(8))


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
