"""Target completion contracts on scheduled native GM handoff tasks.

Inputs are already scheduled. These host regressions check completion, epoch
and flag planning independently of layouts, solver decisions, and devices.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _target(mod, arch):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _with_lane(mod, sid):
    thread = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    return tvm.IRModule({"main": mod["main"].with_body(tirx.AttrStmt(thread, "thread_extent", 2, mod["main"].body))})


def _root_body(mod, body):
    root = mod["main"].body.block
    root = tirx.SBlock(root.iter_vars, root.reads, root.writes, root.name_hint, body,
                      init=root.init, alloc_buffers=root.alloc_buffers,
                      match_buffers=root.match_buffers, annotations=root.annotations)
    return tvm.IRModule({"main": mod["main"].with_body(tirx.SBlockRealize([], True, root))})


def _handoff(direction, *, arch="dav-2201", lane_guard=None, internal_guard=False, vector_count=2, ring=False, reservation=None):
    shape = (2, 16, 16) if ring else (16, 16)
    exchange = tirx.decl_buffer(shape, "float32", name="exchange")
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((16, 16), "float32", name="ub", scope="shared.dyn")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    i = tirx.Var("epoch", "int32")
    indices = [i % 2, 0, 0] if ring else None
    region = [1, 16, 16] if ring else None
    if direction == "cube_to_vector":
        tasks = [(copy(accum, exchange, dst_indices=indices, dst_shape=region), 2),
                 (copy(exchange, ub, src_indices=indices, src_shape=region), 1)]
    else:
        tasks = [(copy(ub, exchange, dst_indices=indices, dst_shape=region), 1),
                 (copy(exchange, l1, src_indices=indices, src_shape=region), 2)]
    tasks = [unit(tirx.IfThenElse(lane_guard, body, None)
                  if internal_guard and lane_guard is not None and core == 1 else body,
                  core=core, guard=lane_guard if not internal_guard and core == 1 else None)
             for body, core in tasks]
    body = seq(*tasks)
    annotations = {"vector_count": vector_count}
    if ring:
        body = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body), core=None)
        annotations["tl.manual_multi_buffer"] = {exchange.data: 2}
    if reservation is not None:
        mode, flag = reservation
        explicit = [unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(f"tl.ascend_cross_core_{operation}_flag"),
                                                [mode, "PIPE_MTE3", flag])), core=1)
                    for operation in ("set", "wait")]
        body = seq(*explicit, body)
    return _target(kernel(body, buffers=[accum, ub, l1], params=[exchange], annotations=annotations), arch)


def _events(mod, operation):
    return calls(mod, f"tl.ascend_cross_core_{operation}_flag")


def _event_core(stmt, parents):
    if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
        return None
    if not isinstance(stmt.value.op, tvm.ir.Op) or not stmt.value.op.name.startswith("tl.ascend_cross_core_"):
        return None
    return next(int(parent.node["core_mask"]) for parent in reversed(parents)
                if isinstance(parent, tirx.AttrStmt) and parent.attr_key == "tl.ascend_task")


@pytest.mark.parametrize("direction", ["cube_to_vector", "vector_to_cube"])
@pytest.mark.parametrize("arch, mode, cube_channels", [("dav-2201", 2, 1), ("dav-3510", 4, 2)])
@pytest.mark.parametrize("representation", ["ffi-int", "tir-int64"])
def test_native_gm_handoff_uses_target_completion(direction, arch, mode, cube_channels, representation):
    count = 2 if representation == "ffi-int" else tirx.IntImm("int64", 2)
    after = transform.InsertSync()(_handoff(direction, arch=arch, vector_count=count))
    assert {int(event.args[0]) for operation in ("set", "wait") for event in _events(after, operation)} == {mode}
    by_core = {1: [], 2: []}
    for stmt, parents in statements(after):
        core = _event_core(stmt, parents)
        if core is not None:
            by_core[core].append(stmt.value)
    assert len(by_core[1]) == 1
    assert len(by_core[2]) == cube_channels
    vector_flag = int(by_core[1][0].args[2])
    assert {int(event.args[2]) for event in by_core[2]} == {vector_flag + 16 * lane for lane in range(cube_channels)}


@pytest.mark.parametrize("direction", ["cube_to_vector", "vector_to_cube"])
def test_empty_lane_data_predicate_keeps_collective_completion(direction):
    sid = tirx.Var("sid", "int32")
    before = _with_lane(_handoff(direction, lane_guard=sid == 0, internal_guard=True), sid)
    after = transform.InsertSync()(before)
    events = [stmt for stmt, parents in statements(after) if _event_core(stmt, parents) is not None]
    assert len(events) == 2
    for stmt, parents in statements(after):
        if _event_core(stmt, parents) is not None:
            assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)


@pytest.mark.parametrize("direction", ["cube_to_vector", "vector_to_cube"])
def test_lane_guard_outside_task_is_rejected(direction):
    sid = tirx.Var("sid", "int32")
    before = _with_lane(_handoff(direction, lane_guard=sid == 0), sid)
    with pytest.raises(tvm.error.InternalError, match="collective.*(uniform|enclosed)"):
        transform.InsertSync()(before)


def test_lane_dependent_bind_is_not_a_uniform_epoch():
    sid = tirx.Var("sid", "int32")
    enabled = tirx.Var("enabled", "bool")
    before = _handoff("vector_to_cube", lane_guard=enabled)
    before = _root_body(before, seq(unit(tirx.Bind(enabled, sid == 0), core=3), before["main"].body.block.body))
    before = _with_lane(before, sid)
    with pytest.raises(tvm.error.InternalError, match="collective.*(uniform|enclosed)"):
        transform.InsertSync()(before)


@pytest.mark.parametrize("direction", ["cube_to_vector", "vector_to_cube"])
def test_owner_row_loop_cannot_skip_collective_epochs(direction):
    sid, row = tirx.Var("sid", "int32"), tirx.Var("row", "int32")
    before = _handoff(direction)
    body = unit(tirx.For(row, 0, 8 * (1 - sid), tirx.ForKind.SERIAL, before["main"].body.block.body), core=None)
    before = _with_lane(_root_body(before, body), sid)
    with pytest.raises(tvm.error.InternalError, match="collective.*(uniform|owner loop)"):
        transform.InsertSync()(before)


def test_single_aiv_mixed_completion_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="requires both physical AIV lanes"):
        transform.InsertSync()(_handoff("vector_to_cube", vector_count=1))


@pytest.mark.parametrize("count", [0, 3, 2.0, True, tirx.IntImm("bool", True), tirx.Var("count", "int32")])
def test_vector_count_requires_a_constant_integer_topology(count):
    with pytest.raises(tvm.error.InternalError, match="vector_count must be the constant integer 1 or 2"):
        transform.InsertSync()(_handoff("cube_to_vector", vector_count=count))


@pytest.mark.parametrize("direction", ["cube_to_vector", "vector_to_cube"])
def test_collective_ring_preserves_versions_and_balanced_fill_drain(direction):
    after = transform.InsertSync()(_handoff(direction, ring=True))
    sets, waits = _events(after, "set"), _events(after, "wait")
    assert sets and waits
    assert {int(event.args[0]) for event in sets + waits} == {2}
    analyzer = tvm.arith.Analyzer()
    for operation, events in [("set", sets), ("wait", waits)]:
        dynamic = [event.args[2] for event in events if not isinstance(event.args[2], tirx.IntImm)]
        assert len(dynamic) == 2, f"RAW and reuse each need one collective {operation}"
        assert all(0 <= int(event.args[2]) < 16 for event in events if isinstance(event.args[2], tirx.IntImm))
    initial = sorted(int(event.args[2]) for event in sets if isinstance(event.args[2], tirx.IntImm))
    drained = sorted(int(event.args[2]) for event in waits if isinstance(event.args[2], tirx.IntImm))
    assert initial == drained and len(initial) == 2
    released = sorted((analyzer.simplify(event.args[2]) for event in sets if not isinstance(event.args[2], tirx.IntImm)), key=str)
    acquired = sorted((analyzer.simplify(event.args[2]) for event in waits if not isinstance(event.args[2], tirx.IntImm)), key=str)
    assert all(analyzer.can_prove_equal(left, right) for left, right in zip(released, acquired))


def test_explicit_flag_reservation_is_shared_with_collective_allocator():
    after = transform.InsertSync()(_handoff("vector_to_cube", reservation=(0, 0)))
    generated = [event for operation in ("set", "wait") for event in _events(after, operation) if int(event.args[0]) == 2]
    assert generated and all(int(event.args[2]) != 0 for event in generated)


@pytest.mark.parametrize("mode, flag, diagnostic", [(4, 0, "supports modes"), (2, 16, "provably in")])
def test_3510_counter_protocol_is_not_aliased_on_2201(mode, flag, diagnostic):
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        transform.InsertSync()(_handoff("vector_to_cube", reservation=(mode, flag)))


def test_insert_sync_requires_target_contract():
    before = _handoff("vector_to_cube")
    before = tvm.IRModule({"main": before["main"].without_attr("target")})
    with pytest.raises(tvm.error.InternalError, match="requires a bound Ascend target"):
        transform.InsertSync()(before)
