"""A persistent child binary handshake can cover the same parent boundary.

These assigned native tasks keep the ordinary whole-completion C
protocol. The oracle expands real finite controls and joins completion only
at source fences and destination acquires. It adds neither cross-pipe issue
order nor a wait-before-reuse edge on behalf of InsertSync.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import product

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout, make_ascend_nz_layout,
)
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_COPY = "tl.tileop.ascend_copy"
_GEMM = "tl.tileop.gemm"
_SET = "tl.ascend_set_flag"
_WAIT = "tl.ascend_wait_flag"
_CONFIG = "tl.ascend_set_fixpipe_config"
_DIRECTION = "tl.ascend_set_mmad_direction"
_WAR = "M_MTE1"


def _native_cache(*, epochs=3, guarded=False, dynamic_owners=False, counter=False, runtime_epochs=False):
    """A is filled once per owner and read by three independent N results."""
    e, owner, n = [tirx.Var(name, "int32") for name in ("epoch", "owner", "n_tile")]
    owners = tirx.Var("owners", "int32") if dynamic_owners else tirx.IntImm("int32", 2)
    count = tirx.Var("count", "int32") if runtime_epochs else tirx.IntImm("int32", epochs)
    source = tirx.decl_buffer((3, 2, 32, 16), "float16", name="Input")
    rhs = tirx.decl_buffer((32, 16), "float16", name="RHS")
    output = tirx.decl_buffer((3, 2, 3, 32, 32), "float32", name="Output")
    a_l1 = tirx.decl_buffer((32, 16), "float16", name="a_l1", scope="shared.l1")
    b_l1 = tirx.decl_buffer((32, 16), "float16", name="b_l1", scope="shared.l1")
    a = tirx.decl_buffer((32, 16), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((32, 16), "float16", name="b", scope="shared.l0b")
    c = tirx.decl_buffer((32, 32), "float32", name="accum", scope="shared.l0c")
    clock = tirx.decl_buffer((1,), "int32", name="a_epoch", scope="local.var") if counter else None
    export = copy(c, output, dst_indices=[e, owner, n, 0, 0], dst_shape=[1, 1, 1, 32, 32],
                  annotations={"tl.ascend.root_fixpipe_config": tirx.IntImm("int32", 1)})
    results = unit(tirx.For(n, 0, 3, tirx.ForKind.SERIAL, seq(
        unit(tirx.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)), core=2),
        unit(export, core=2),
    )), core=None)
    owner_tasks = [
        unit(copy(source, a_l1, src_indices=[e, owner, 0, 0], src_shape=[1, 1, 32, 16]), core=2),
        unit(copy(a_l1, a), core=2), results,
    ]
    owner_annotations = {}
    if counter:
        owner_annotations = {
            "multi_buffer_eligible": [a.data],
            "tl.multi_buffer_counter_map": {a.data: clock},
            "tl.storage_epoch_guard_map": {a.data: tirx.const(True, "bool")},
        }
        owner_tasks.append(unit(tirx.BufferStore(clock, clock[0] + 1, [0]), core=2))
    child = unit(tirx.For(owner, 0, owners, tirx.ForKind.SERIAL, seq(*owner_tasks),
                          annotations=owner_annotations), core=None, guard=e < 2 if guarded else None)
    body = [unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_CONFIG), [])), core=2),
            # Explicit direction isolates ordinary operand/C completion from
            # the separately tested automatic native unit-stream selector.
            unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_DIRECTION), [tirx.StringImm("m")])), core=2),
            unit(copy(rhs, b_l1), core=2), unit(copy(b_l1, b), core=2)]
    if counter:
        body.append(unit(tirx.BufferStore(clock, 0, [0]), core=2))
    body.append(unit(tirx.For(e, 0, count, tirx.ForKind.SERIAL, child), core=None))
    layouts = {a_l1: make_ascend_nz_layout(a_l1), b_l1: make_ascend_nz_layout(b_l1),
               a: make_ascend_l0a_2201_layout(a), b: make_ascend_major_k_layout(b),
               c: make_ascend_l0c_layout(c)}
    mod = kernel(seq(*body), buffers=[a_l1, b_l1, a, b, c] + ([clock] if counter else []),
                 params=[source, rhs, output] + ([owners] if dynamic_owners else []) + ([count] if runtime_epochs else []),
                 annotations={"layout_map": layouts, "tl.buffer_versions_map": {a.data: 2} if counter else {}})
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    if not counter:
        # The common preparation pass records the enclosing active storage
        # epochs. A guarded owner must not be tested as an unconditional
        # prepared parent domain merely because this fixture has one C slot.
        with mod["main"].attrs["target"]:
            mod = transform.PrepareMultiBuffer()(mod)
    return mod, {"e": e, "owner": owner, "n": n, "owners": owners, "a": a, "counter": clock,
                 "epochs": epochs, "count": count, "active_epochs": min(epochs, 2) if guarded else epochs}


def _insert(before):
    with before["main"].attrs["target"], tvm.transform.PassContext(
            config={"tl.disable_shared_memory_reuse": True}):
        return transform.InsertSync()(before)


def _lower(after):
    with after["main"].attrs["target"]:
        return transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(after))


def _war_sites(mod, pipe=_WAR):
    result = {"set": [], "wait": []}
    for statement, parents in statements(mod):
        if not isinstance(statement, tirx.Evaluate) or not isinstance(statement.value, tirx.Call):
            continue
        call = statement.value
        if _name(call) not in (_SET, _WAIT) or call.args[0].value != pipe:
            continue
        operation = "set" if _name(call) == _SET else "wait"
        result[operation].append((call, parents))
    return result


def _inside(parents, var):
    return any(isinstance(parent, tirx.For) and parent.loop_var.same_as(var) for parent in parents)


@dataclass(frozen=True)
class _Command:
    pipe: str
    kind: str
    flag: tuple | None = None
    reads: tuple = ()
    writes: tuple = ()
    logical: tuple | None = None


def _blocks(arg, env):
    buffer, origin, shape = _region(arg, env)
    assert len(shape) >= 2 and all(extent == 1 for extent in shape[:-2])
    assert all(value % 16 == 0 for value in (*origin[-2:], *shape[-2:])), (origin, shape)
    return tuple((buffer.data, *origin[:-2], row, column)
                 for row, column in product(range(origin[-2] // 16, (origin[-2] + shape[-2]) // 16),
                                            range(origin[-1] // 16, (origin[-1] + shape[-1]) // 16)))


def _commands(mod, facts, parameters=None):
    parameters = parameters or {}
    # The shared trace interprets prepared mutable counters at their actual
    # stores. Substitute only externally supplied finite fixture parameters.
    if parameters:
        func = mod["main"]
        mod = tvm.IRModule({"main": func.with_body(tirx.stmt_functor.substitute(
            func.body, {var: tirx.const(value, var.dtype) for var, value in parameters.items()}))})
    trace = []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        # LowerScheduledTIR freshens loop variables. These deliberately unique
        # fixture loop names identify actual traced bindings, rather than
        # looking up the pre-lowering Var object or inferring a counter clock.
        coordinates = {var.name: value for var, value, _ in loops}
        assert len(coordinates) == len(loops), "Fixture loop names must be unique"
        logical = tuple(coordinates.get(facts[key].name) for key in ("e", "owner", "n"))
        if name in (_SET, _WAIT):
            event = call.args[0].value
            source, destination = event.split("_")
            kind = "set" if name == _SET else "wait"
            event_id = _value(call.args[1], env)
            assert 0 <= event_id < 8
            trace.append(_Command(source if kind == "set" else destination, kind, (event, event_id), logical=logical))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier", logical=logical))
        elif name == _COPY:
            src, _, _ = _region(call.args[0], env)
            dst, _, _ = _region(call.args[1], env)
            if src.scope() == "global":
                pipe = "MTE2"
            elif dst.scope() in ("shared.l0a", "shared.l0b"):
                assert src.scope() == "shared.l1"
                pipe = "MTE1"
            else:
                assert src.scope() == "shared.l0c" and dst.scope() == "global"
                pipe = "FIX"
            kind = "a_fill" if dst.data.same_as(facts["a"].data) else "copy"
            trace.append(_Command(pipe, kind, reads=_blocks(call.args[0], env),
                                  writes=_blocks(call.args[1], env), logical=logical))
        elif name == _GEMM:
            assert _value(call.args[9], env), "Each independent N result starts a new C epoch"
            assert "unit_flag_ctrl" not in call.annotations
            trace.append(_Command("M", "mad", reads=(*_blocks(call.args[0], env), *_blocks(call.args[1], env)),
                                  writes=_blocks(call.args[2], env), logical=logical))
        else:
            assert name in (_CONFIG, _DIRECTION), f"Unexpected native fixture operation: {name}"
            trace.append(_Command("S", "scalar", logical=logical))
    return trace


def _prove_completions(trace):
    """Return reachability of actual issue/start/completion event nodes."""
    parents, pending, gates = {}, defaultdict(set), {}
    notifications, acquisitions = defaultdict(list), defaultdict(list)
    scalar_issue = None
    pipes = ("S", "M", "FIX", "MTE1", "MTE2", "MTE3")
    for index, command in enumerate(trace):
        issue, start, done = 3 * index, 3 * index + 1, 3 * index + 2
        parents[issue] = {scalar_issue} if scalar_issue is not None else set()
        parents[start], parents[done] = {issue}, {start}
        selected_pipes = pipes if command.pipe == "ALL" else (command.pipe,)
        for pipe in selected_pipes:
            if pipe in gates:
                parents[start].add(gates[pipe])
            if command.kind in ("set", "barrier"):
                parents[done].update(pending[pipe])
                pending[pipe].clear()
            pending[pipe].add(done)
            if command.kind in ("wait", "barrier"):
                gates[pipe] = done
        if command.kind in ("set", "wait"):
            (notifications if command.kind == "set" else acquisitions)[command.flag].append(done)
        scalar_issue = issue
    assert notifications.keys() == acquisitions.keys()
    for flag, sent in notifications.items():
        assert len(sent) == len(acquisitions[flag]), (flag, "Unbalanced seed/set/wait/drain")
        for notification, acquisition in zip(sent, acquisitions[flag]):
            parents[acquisition].add(notification)
    children, remaining = defaultdict(list), {node: len(incoming) for node, incoming in parents.items()}
    for node, incoming in parents.items():
        for predecessor in incoming:
            children[predecessor].append(node)
    ready, rank = deque(node for node, degree in remaining.items() if not degree), {}
    while ready:
        node = ready.popleft()
        rank[node] = len(rank)
        for child in children[node]:
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)
    assert len(rank) == len(parents), "Finite nested local protocol deadlocks"
    proven = set()

    def require(before, after):
        if (before, after) in proven:
            return
        frontier, seen = [after], set()
        while frontier:
            node = frontier.pop()
            if node == before:
                proven.add((before, after))
                return
            if node not in seen and rank[node] > rank[before]:
                seen.add(node)
                frontier.extend(parents[node])
        raise AssertionError(f"Missing completion path {before // 3}:{trace[before // 3]} -> "
                             f"{after // 3}:{trace[after // 3]}")

    # Token pairing does not itself establish that a binary ID is safe to
    # reset. This must follow independently from the actual destination gates.
    for flag, sent in notifications.items():
        for consumed, reset in zip(acquisitions[flag], sent[1:]):
            require(consumed, reset)
    writers, readers = {}, defaultdict(set)
    for index, command in enumerate(trace):
        for block in command.reads:
            if block in writers:
                require(3 * writers[block] + 2, 3 * index + 1)
            readers[block].add(index)
        for block in command.writes:
            if block in writers:
                require(3 * writers[block] + 2, 3 * index + 1)
            for reader in readers[block]:
                if reader != index:
                    require(3 * reader + 2, 3 * index + 1)
            writers[block] = index
            readers[block].clear()
    return require


def _check_cache_epochs(trace, facts, owner_count=2):
    active = facts["active_epochs"]
    expected = {(e, owner, n) for e in range(active) for owner in range(owner_count) for n in range(3)}
    mads = {command.logical: i for i, command in enumerate(trace) if command.kind == "mad"}
    fills = {command.logical[:2]: i for i, command in enumerate(trace) if command.kind == "a_fill"}
    assert len(mads) == sum(command.kind == "mad" for command in trace) and set(mads) == expected
    assert len(fills) == sum(command.kind == "a_fill" for command in trace)
    assert set(fills) == {(e, owner) for e in range(active) for owner in range(owner_count)}
    require = _prove_completions(trace)
    previous_slots = {}
    for activation, fill in fills.items():
        for n in range(3):
            require(3 * fill + 2, 3 * mads[(*activation, n)] + 1)
        slot = trace[fill].writes
        if slot in previous_slots:
            previous = previous_slots[slot]
            for n in range(3):
                require(3 * mads[(*previous, n)] + 2, 3 * fill + 1)
        previous_slots[slot] = activation


@pytest.mark.parametrize("epochs,guarded", [(0, False), (1, False), (3, False), (3, True)],
                         ids=["empty-parent", "single-parent", "reentered-owner", "bounded-parent-activation"])
def test_persistent_owner_handshake_replaces_parent_cache_WAR(epochs, guarded):
    before, facts = _native_cache(epochs=epochs, guarded=guarded, runtime_epochs=True)
    original = tvm.ir.save_json(before)
    synced = _insert(before)
    assert tvm.ir.save_json(before) == original
    sites = _war_sites(synced)
    for operation in ("set", "wait"):
        inside = [(call, parents) for call, parents in sites[operation] if _inside(parents, facts["owner"])]
        outside = [(call, parents) for call, parents in sites[operation] if not _inside(parents, facts["owner"])]
        assert len(inside) == 1, "One persistent owner handshake must remain"
        assert len(outside) == 1, "Only its root seed/drain may remain, without a second parent protocol"
        assert not _inside(outside[0][1], facts["e"]), "The remaining seed/drain must be rooted once"
    trace = _commands(_lower(synced), facts, {facts["count"]: epochs})
    _check_cache_epochs(trace, facts)
    flags = {command.flag for command in trace if command.flag and command.flag[0] == _WAR}
    assert len(flags) == 1
    for kind in ("set", "wait"):
        assert sum(command.kind == kind and command.flag in flags for command in trace) == facts["active_epochs"] * 2 + 1


@pytest.mark.parametrize("owner_count", [0, 2])
def test_possibly_empty_owner_cannot_supply_a_persistent_parent_certificate(owner_count):
    before, facts = _native_cache(dynamic_owners=True)
    synced = _insert(before)
    for operation in ("set", "wait"):
        assert any(_inside(parents, facts["e"]) and not _inside(parents, facts["owner"])
                   for _, parents in _war_sites(synced)[operation]), "Dynamic child cardinality cannot erase parent completion"
    trace = _commands(_lower(synced), facts, {facts["owners"]: owner_count})
    _check_cache_epochs(trace, facts, owner_count)


def test_counter_owner_cannot_replace_lexical_parent_completion():
    before, facts = _native_cache(counter=True)
    synced = _insert(before)
    sites = _war_sites(synced)
    lexical_ids = []
    for operation in ("set", "wait"):
        # Parent endpoints may be projected into the actual first/last owner
        # tasks. The parent is identified by its separate lexical flag, not
        # by requiring its calls to sit outside the owner loop in the AST.
        lexical = [(call, parents) for call, parents in sites[operation]
                   if _inside(parents, facts["e"]) and not nodes(call.args[1], tirx.BufferLoad)]
        assert len(lexical) == 1, "A mutable child clock cannot erase the separate lexical parent protocol"
        assert any(isinstance(parent, tirx.IfThenElse) and
                   any(var.same_as(facts["owner"]) for var in nodes(parent.condition, tirx.Var))
                   for parent in lexical[0][1]), "The parent endpoint must select the first/last owner"
        lexical_ids.append(lexical[0][0].args[1])
        assert any(nodes(call.args[1], tirx.BufferLoad) for call, _ in sites[operation]), "The real counter protocol must remain"
    tvm.ir.assert_structural_equal(*lexical_ids)
    _check_cache_epochs(_commands(_lower(synced), facts), facts)


def _guarded_halves(*, child_active=None, parent_active=None, intervals=False):
    """The child and parent touch disjoint halves under independent guards."""
    source = tirx.decl_buffer((3, 32, 16), "float16", name="Input")
    storage = tirx.decl_buffer((32, 16), "float16", name="storage", scope="shared.l1")
    consumed = tirx.decl_buffer((3, 32, 16), "float16", name="Consumed", scope="shared.l0a")
    e, owner, n = [tirx.Var(name, "int32") for name in ("epoch", "owner", "unused_n")]
    p, q = ((e < 2, e >= 1) if intervals else
            (tirx.const(child_active, "bool"), tirx.const(parent_active, "bool")))

    def load(row, guard):
        return unit(copy(source, storage, src_indices=[e, row, 0], src_shape=[1, 16, 16],
                         dst_indices=[row, 0], dst_shape=[16, 16]), core=2, guard=guard)

    def read(row, guard):
        return unit(copy(storage, consumed, src_indices=[row, 0], src_shape=[16, 16],
                         dst_indices=[e, row, 0], dst_shape=[1, 16, 16]), core=2, guard=guard)

    child = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, seq(load(16, p), read(16, p)),
                          annotations={"tl.storage_epoch_guard_map": {storage.data: p}}), core=None)
    parent = unit(tirx.For(e, 0, 3, tirx.ForKind.SERIAL, seq(load(0, q), child, read(0, q)),
                           annotations={"tl.storage_epoch_guard_map": {storage.data: tirx.Or(p, q)}}), core=None)
    mod = kernel(parent, buffers=[storage, consumed], params=[source])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    with mod["main"].attrs["target"]:
        mod = transform.PrepareMultiBuffer()(mod)
    return mod, {"e": e, "owner": owner, "n": n, "a": storage}


def test_narrow_child_guard_keeps_the_wider_parent_memory_protocol():
    # Specialize the independent input predicates before preparation. An
    # unknown p|q has no provable finite active interval in the ordinary
    # baseline; it cannot serve as a valid fallback fixture for this feature.
    for p, q in ((False, True), (True, False), (True, True)):
        before, facts = _guarded_halves(child_active=p, parent_active=q)
        synced = _insert(before)
        if q:
            for operation in ("set", "wait"):
                assert any(_inside(parents, facts["e"]) and not _inside(parents, facts["owner"])
                           for _, parents in _war_sites(synced, "MTE1_MTE2")[operation])
        # A false child must never erase the live parent half. Every active
        # half still has real RAW/WAR and binary-reuse completion paths.
        trace = _commands(_lower(synced), facts)
        assert sum(command.kind == "a_fill" for command in trace) == 3 * (int(q) + 2 * int(p))
        _prove_completions(trace)


def test_independent_parent_guard_remains_live_after_child_activation_ends():
    # These independent prefix/suffix guards have a provable full union and
    # individually contiguous finite intervals. At e=2 the child is inactive
    # while the parent still overwrites and reads its own half of storage.
    before, facts = _guarded_halves(intervals=True)
    synced = _insert(before)
    for operation in ("set", "wait"):
        assert any(_inside(parents, facts["e"]) and not _inside(parents, facts["owner"])
                   for _, parents in _war_sites(synced, "MTE1_MTE2")[operation])
    trace = _commands(_lower(synced), facts)
    assert Counter(command.logical[0] for command in trace if command.kind == "a_fill") == {0: 2, 1: 3, 2: 1}
    _prove_completions(trace)


def test_oracle_rejects_balanced_notifications_without_binary_reuse_completion():
    flag = ("M_MTE1", 0)
    trace = [_Command("M", "set", flag), _Command("MTE1", "wait", flag),
             _Command("M", "set", flag), _Command("MTE1", "wait", flag)]
    with pytest.raises(AssertionError, match="Missing completion path"):
        _prove_completions(trace)
