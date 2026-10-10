"""Owner-specific dav-2201 result readiness from ordinary typed GM regions.

These are CPU lowering/placement regressions. They do not model an unconsumed
AIV flag copy as a FIFO or claim that an issued FIX has completed in hardware.
Native completion and finite flag reuse need the separate AscendC/binary gate.
"""

from collections import Counter, defaultdict

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


_CROSS_SET = "tl.ascend_cross_core_set_flag"
_CROSS_WAIT = "tl.ascend_cross_core_wait_flag"
_COPY = "tl.tileop.ascend_copy"
_SCALAR_VALUES = object()
_PURE_DEFINITIONS = object()


def _full_pv_program():
    """Two retained owner accumulators, then one complete FIX per owner."""
    @T.prim_func
    def main(a: T.Tensor((3, 2, 32, 32), "float16"),
             b: T.Tensor((3, 32, 32), "float16"),
             Result: T.Tensor((1, 2, 2, 32, 32), "float32"),
             Unowned: T.Tensor((1, 2, 16, 16), "float32"),
             out: T.Tensor((3, 2, 32, 32), "float32"),
             aux_out: T.Tensor((3, 2, 16, 16), "float32")):
        with T.MixedKernel(1, sids=2) as (group, sid):
            a_l1 = T.alloc_l1((32, 16), "float16")
            b_l1 = T.alloc_l1((16, 32), "float16")
            c = T.alloc_l0c((2, 32, 32), "float32")
            aux_c = T.alloc_l0c((16, 16), "float32")
            value = T.alloc_shared((16, 32), "float32")
            aux = T.alloc_shared((16, 16), "float32")
            T.annotate_aiv_private(value, aux)
            T.annotate_buffer_versions({a_l1: 1, b_l1: 1, c: 1,
                                        aux_c: 1, value: 1, aux: 1})
            T.annotate_manual_multi_buffer({Result: 2, Unowned: 2})
            for e in T.serial(4):
                with T.Stage(0):
                    if e < 3:
                        with T.Cube():
                            for chunk in T.serial(2):
                                T.copy(b[e, chunk * 16:(chunk + 1) * 16, :], b_l1)
                                for owner in T.serial(2):
                                    T.copy(a[e, owner, :, chunk * 16:(chunk + 1) * 16], a_l1)
                                    T.gemm(a_l1, b_l1, c[owner, :, :],
                                           transpose_B=False, clear_accum=(chunk == 0))
                            for owner in T.serial(2):
                                T.copy(c[owner, :, :], Result[group, owner, e % 2, :, :])
                            T.gemm(a_l1[0:16, :], b_l1[:, 0:16], aux_c,
                                   transpose_B=False, clear_accum=True)
                            T.copy(aux_c, Unowned[group, e % 2, :, :])
                with T.Stage(0):
                    if 1 <= e and e < 4:
                        for sub in T.serial(2):
                            row = sub * 16
                            T.copy(Result[group, sid, (e - 1) % 2, row:row + 16, :], value)
                            T.copy(value, out[e - 1, sid, row:row + 16, :])
                        T.copy(Unowned[group, (e - 1) % 2, :, :], aux)
                        T.copy(aux, aux_out[e - 1, sid, :, :])
    return main


def _mn_score_program():
    """Three physical C tiles must all finish before an owner's publication."""
    @T.prim_func
    def main(a: T.Tensor((3, 2, 1536, 64), "float16"),
             b: T.Tensor((3, 32, 64), "float16"),
             Result: T.Tensor((1, 2, 2, 1536, 32), "float32"),
             out: T.Tensor((3, 2, 1536, 32), "float32")):
        with T.MixedKernel(1, sids=2) as (group, sid):
            a_l1 = T.alloc_l1((1536, 64), "float16")
            b_l1 = T.alloc_l1((32, 64), "float16")
            c = T.alloc_l0c((1536, 32), "float32")
            value = T.alloc_shared((16, 32), "float32")
            T.annotate_aiv_private(value)
            T.annotate_buffer_versions({a_l1: 1, b_l1: 1, value: 1})
            T.annotate_manual_multi_buffer({Result: 2})
            for e in T.serial(4):
                with T.Stage(0):
                    if e < 3:
                        with T.Cube():
                            T.copy(b[e, :, :], b_l1)
                            for owner in T.serial(2):
                                T.copy(a[e, owner, :, :], a_l1)
                                T.gemm(a_l1, b_l1, c, transpose_B=True, clear_accum=True)
                                T.copy(c, Result[group, owner, e % 2, :, :])
                with T.Stage(0):
                    if 1 <= e and e < 4:
                        for sub in T.serial(96):
                            row = sub * 16
                            T.copy(Result[group, sid, (e - 1) % 2, row:row + 16, :], value)
                            T.copy(value, out[e - 1, sid, row:row + 16, :])
    return main


def _lower(program):
    snapshots = {}
    wanted = {"tl.AscendLowerL1Gemm", "tl.AscendLowerMixedPartition",
              "tl.InsertSync", "tl.AscendLowerTileOp"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(program.with_attr("global_symbol", "main"),
                                target={"kind": "ascend", "arch": "dav-2201"},
                                target_host="c", enable_host_codegen=True,
                                enable_device_compile=False)
    assert result.host_mod and set(snapshots) == wanted
    return result.kernel_source, snapshots


def _name(call):
    return call.op.name if isinstance(call.op, tvm.ir.Op) else ""


def _scalar_cell(buffer, indices, env):
    assert buffer.scope() == "local.var", f"Unexpected scalar storage: {buffer}"
    assert all(_value(dimension, env) == 1 for dimension in buffer.shape)
    assert tuple(_value(index, env) for index in indices) == (0,) * len(buffer.shape)
    return buffer.data


def _value(expr, env):
    # PrepareMultiBuffer can use a mutable local scalar epoch. Substitute its
    # actual value at this point in the trace, rather than treating a counter
    # load as an immutable Bind or guessing its value from the lexical epoch.
    def load_scalar(node):
        if isinstance(node, tirx.BufferLoad):
            cell = _scalar_cell(node.buffer, node.indices, env)
            assert cell in env[_SCALAR_VALUES], f"Uninitialized fixture scalar: {node}"
            return tirx.const(env[_SCALAR_VALUES][cell], node.dtype)

    scalar_resolved = tirx.stmt_functor.ir_transform(tirx.Evaluate(expr), None, load_scalar).value

    def resolve(var, expanding=()):
        if var in env:
            return env[var]
        definitions = env[_PURE_DEFINITIONS]
        if var not in definitions:
            return var
        assert all(not var.same_as(active) for active in expanding), f"Cyclic fixture SSA definition: {var}"
        rhs = definitions[var]
        substitutions = {dependency: resolve(dependency, (*expanding, var))
                         for dependency in nodes(rhs, tirx.Var)}
        return tirx.stmt_functor.substitute(rhs, substitutions)

    # A pure condition Bind can be assigned to the other core's scalar task.
    # Inline its SSA definition in the current lexical environment, without
    # caching a value computed for a different owner, row or epoch.
    bindings = {var: resolve(var) for var in nodes(scalar_resolved, tirx.Var)}
    value = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(scalar_resolved, bindings))
    assert isinstance(value, tirx.IntImm), f"Unbound fixture expression: {value}"
    return int(value)


def _trace(stmt, *, core, lane, group=0, env=None, loops=()):
    """Expand this finite fixture's control, retaining lexical loop facts."""
    env = {_SCALAR_VALUES: {}, _PURE_DEFINITIONS: {}} if env is None else env
    if isinstance(stmt, (tvm.IRModule, tirx.PrimFunc)):
        body = stmt["main"].body if isinstance(stmt, tvm.IRModule) else stmt.body
        # Only immutable arithmetic definitions may cross the trace's core
        # filter. Loads and calls retain their actual execution/state contract.
        for binding in nodes(body, tirx.Bind):
            if not nodes(binding.value, (tirx.BufferLoad, tirx.Call)):
                previous = env[_PURE_DEFINITIONS].get(binding.var)
                assert previous is None or tvm.ir.structural_equal(previous, binding.value)
                env[_PURE_DEFINITIONS][binding.var] = binding.value
        yield from _trace(body, core=core, lane=lane, group=group, env=env, loops=loops)
    elif isinstance(stmt, tirx.SeqStmt):
        for child in stmt.seq:
            yield from _trace(child, core=core, lane=lane, group=group, env=env, loops=loops)
    elif isinstance(stmt, tirx.SBlockRealize):
        assert _value(stmt.predicate, env)
        yield from _trace(stmt.block, core=core, lane=lane, group=group, env=env, loops=loops)
    elif isinstance(stmt, tirx.SBlock):
        yield from _trace(stmt.body, core=core, lane=lane, group=group, env=env, loops=loops)
    elif isinstance(stmt, tirx.AttrStmt):
        if stmt.attr_key == "tl.ascend_task" and int(stmt.node["core_mask"]) not in (core, 3):
            return
        local = env
        if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tirx.IterVar):
            coordinate = (lane if stmt.node.thread_tag == "cthread" else
                          group if stmt.node.thread_tag == "blockIdx.x" else 0)
            local = {**env, stmt.node.var: tirx.const(coordinate, stmt.node.var.dtype)}
        yield from _trace(stmt.body, core=core, lane=lane, group=group, env=local, loops=loops)
    elif isinstance(stmt, tirx.For):
        start, extent = _value(stmt.min, env), _value(stmt.extent, env)
        assert stmt.kind == tirx.ForKind.SERIAL and 0 <= extent <= 1536
        for index in range(start, start + extent):
            local = {**env, stmt.loop_var: tirx.const(index, stmt.loop_var.dtype)}
            yield from _trace(stmt.body, core=core, lane=lane, group=group, env=local,
                              loops=(*loops, (stmt.loop_var, index, extent)))
    elif isinstance(stmt, tirx.IfThenElse):
        branch = stmt.then_case if _value(stmt.condition, env) else stmt.else_case
        if branch is not None:
            yield from _trace(branch, core=core, lane=lane, group=group, env=env, loops=loops)
    elif isinstance(stmt, tirx.Bind):
        env[stmt.var] = tirx.const(_value(stmt.value, env), stmt.var.dtype)
    elif isinstance(stmt, tirx.AllocBuffer):
        # A guarded Bind can become a mutable local scalar after its enclosing
        # singleton is inlined. Execute its declared initialization; treating
        # the allocation as an immutable definition would erase inactive
        # epochs and conceal read-before-write errors in event conditions.
        buffer = stmt.buffer
        cell = _scalar_cell(buffer, [0] * len(buffer.shape), env)
        assert str(buffer.dtype).startswith(("int", "uint", "bool"))
        initial = stmt.annotations.get("tl.local_var_init")
        assert initial is not None, f"Uninitialized fixture allocation: {stmt}"
        env[_SCALAR_VALUES][cell] = int(tirx.const(_value(initial, env), buffer.dtype))
    elif isinstance(stmt, tirx.BufferStore):
        cell = _scalar_cell(stmt.buffer, stmt.indices, env)
        assert str(stmt.buffer.dtype).startswith(("int", "uint", "bool"))
        env[_SCALAR_VALUES][cell] = int(tirx.const(_value(stmt.value, env), stmt.buffer.dtype))
    elif isinstance(stmt, tirx.Evaluate):
        if isinstance(stmt.value, tirx.Call):
            # Later stores must not retroactively change an earlier event's
            # region/flag expression when the caller inspects its snapshot.
            yield stmt.value, {**env, _SCALAR_VALUES: dict(env[_SCALAR_VALUES])}, loops
        else:
            assert _value(stmt.value, env) == 0
    else:
        pytest.fail(f"Unexpected completion fixture control: {stmt}")


def _region(arg, env):
    return (arg.args[0].buffer, tuple(_value(i, env) for i in arg.args[0].indices),
            tuple(_value(i, env) for i in arg.args[2:]))


def _uses_var(expr, var):
    return any(node.same_as(var) for node in nodes(expr, tirx.Var))


def _owner_sets(mod):
    # A pair-wide ready/release event lives outside the serial owner loop.
    # Owner publication is identified by its owner-dependent flag expression.
    return [(call, env, loops) for call, env, loops in _trace(mod, core=2, lane=0)
            if _name(call) == _CROSS_SET and call.args[1].value == "PIPE_FIX"
            and any(extent == 2 and _uses_var(call.args[2], var)
                    for var, _, extent in loops)]


def _assert_owner_result(mod, *, rows, columns=32):
    published = _owner_sets(mod)
    assert len(published) == 6, "Exactly three complete results per owner"
    flags = {_value(call.args[2], env) for call, env, _ in published}
    assert len(flags) == 4 and flags == set(range(min(flags), min(flags) + 4))
    base = min(flags)
    assert 0 <= base and base + 3 < 16
    ready_records = {1: [], 2: []}
    coverage = defaultdict(list)
    for call, env, loops in _trace(mod, core=2, lane=0):
        if _name(call) == _COPY:
            source, _, _ = _region(call.args[0], env)
            dest, origin, shape = _region(call.args[1], env)
            if source.scope() == "shared.l0c" and dest.name == "Result":
                group, owner, slot, row, column = origin
                assert group == 0 and shape[:3] == (1, 1, 1)
                coverage[(owner, slot)].append((row, column, *shape[-2:]))
        elif _name(call) == _CROSS_SET and _value(call.args[2], env) in flags:
            assert int(call.args[0]) == 2 and call.args[1].value == "PIPE_FIX"
            epoch = next(index for _, index, extent in loops if extent == 4)
            owner = next(index for var, index, extent in loops
                         if extent == 2 and _uses_var(call.args[2], var))
            assert _value(call.args[2], env) == base + owner * 2 + epoch % 2
            tiles = coverage.pop((owner, epoch % 2))
            assert sum(height * width for _, _, height, width in tiles) == rows * columns
            for row in range(rows):
                intervals = sorted((col, col + width) for start, col, height, width in tiles
                                   if start <= row < start + height)
                assert intervals[0][0] == 0 and intervals[-1][1] == columns
                assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
            ready_records[2].append((epoch, owner, epoch % 2))
    assert not coverage
    assert ready_records[2] == [(e, owner, e % 2) for e in range(3) for owner in range(2)]

    # Each physical AIV waits once before the first row-block GM read, even
    # though the complete logical owner is consumed by several copy tasks.
    for lane in range(2):
        acquired, loads = Counter(), Counter()
        for call, env, loops in _trace(mod, core=1, lane=lane):
            if _name(call) not in (_CROSS_SET, _CROSS_WAIT, _COPY):
                continue
            if _name(call) == _CROSS_WAIT and _value(call.args[2], env) in flags:
                assert int(call.args[0]) == 2 and call.args[1].value == "PIPE_MTE2"
                epoch = next(index for _, index, extent in loops if extent == 4)
                assert 1 <= epoch < 4 and not loads[epoch]
                assert _value(call.args[2], env) == base + lane * 2 + (epoch - 1) % 2
                acquired[epoch] += 1
            elif _name(call) == _COPY:
                source, origin, _ = _region(call.args[0], env)
                if source.name == "Result":
                    epoch = next(index for _, index, extent in loops if extent == 4)
                    assert origin[1:3] == (lane, (epoch - 1) % 2)
                    assert acquired[epoch] == 1
                    loads[epoch] += 1
        assert acquired == Counter({1: 1, 2: 1, 3: 1})
        assert loads == Counter({1: rows // 16, 2: rows // 16, 3: rows // 16})
        ready_records[1].extend((e - 1, lane, (e - 1) % 2) for e in acquired)
    assert sorted(ready_records[1]) == sorted(ready_records[2])

    # The finite source epochs consume each reused owner/slot before its next
    # publication. This is a placement check, not a native completion proof.
    for owner in range(2):
        for slot in range(2):
            generations = [e for e, o, s in ready_records[2] if (o, s) == (owner, slot)]
            assert all(previous + 1 < following
                       for previous, following in zip(generations, generations[1:]))

    # Releasing a result slot remains collective: all AIV readers must finish
    # before a Cube writer reuses it. Owner readiness cannot replace this WAR.
    release_sets = {_value(call.args[2], env)
                    for call, env, _ in _trace(mod, core=1, lane=0)
                    if _name(call) == _CROSS_SET and call.args[1].value == "PIPE_MTE2"}
    release_waits = {_value(call.args[2], env)
                     for call, env, _ in _trace(mod, core=2, lane=0)
                     if _name(call) == _CROSS_WAIT and call.args[1].value == "PIPE_FIX"}
    assert release_sets & release_waits
    assert not flags & (release_sets | release_waits)

    # No initial data-ready token or final drain may be added for this plan;
    # reverse reader-release events use their separate collective allocation.
    for core, lanes, operation, count in ((2, (0,), _CROSS_SET, 6),
                                          (1, (0, 1), _CROSS_WAIT, 6)):
        selected = [(call, env) for lane in lanes for call, env, _ in _trace(mod, core=core, lane=lane)
                    if _name(call) == operation and _value(call.args[2], env) in flags]
        assert len(selected) == count
    return flags


@pytest.mark.parametrize("kind", ["full-pv", "mn-score"])
def test_ordinary_dsl_publishes_complete_owner_results_and_waits_once(kind):
    source, snapshots = _lower(_full_pv_program() if kind == "full-pv" else _mn_score_program())
    partitioned = snapshots["tl.AscendLowerMixedPartition"]
    root = next(block for block in nodes(partitioned, tirx.SBlock) if block.name_hint == "tilelang_root")
    assert root.annotations["tl.ascend_aiv_private_validated"]
    value = next(buffer for block in nodes(partitioned, tirx.SBlock)
                 for buffer in block.alloc_buffers if buffer.name == "value")
    assert tuple(map(int, value.shape)) == (16, 32)
    synced = snapshots["tl.InsertSync"]
    flags = _assert_owner_result(synced, rows=32 if kind == "full-pv" else 1536)
    # Code generation must preserve the selected ordinary block mode and both
    # native data paths. The strict native gate checks hardware completion.
    assert "asc_sync_block_arrive(PIPE_FIX" in source
    assert "asc_sync_block_wait(PIPE_MTE2" in source
    assert "asc_copy_l0c2gm(" in source and "asc_copy_gm2ub_align(" in source
    assert calls(snapshots["tl.AscendLowerTileOp"], "tl.ascend_copy_matrix_cc_to_gm")
    if kind == "full-pv":
        # The unowned plane still needs a pair-wide C->V readiness channel.
        pair_sets = [call for call, env, _ in _trace(synced, core=2, lane=0)
                     if _name(call) == _CROSS_SET and call.args[1].value == "PIPE_FIX"
                     and _value(call.args[2], env) not in flags]
        pair_waits = [call for call, env, _ in _trace(synced, core=1, lane=0)
                      if _name(call) == _CROSS_WAIT and call.args[1].value == "PIPE_MTE2"
                      and _value(call.args[2], env) not in flags]
        assert pair_sets and pair_waits
    else:
        # This exact fixture has an odd number of physical C subtiles. Each
        # owner and epoch must restart the physical ring independently.
        scratch = [buffer for block in nodes(snapshots["tl.AscendLowerL1Gemm"], tirx.SBlock)
                   for buffer in block.alloc_buffers if buffer.scope() == "shared.l0c"]
        assert len(scratch) == 1 and tuple(map(int, scratch[0].shape)) == (2, 512, 32)


def _scheduled_handoff(*, arch="dav-2201", private=True, complete=True,
                       read_complete=None, owner_extent=2, matching_owner=True,
                       consumer_epochs=3, alias=False, reserved_flags=0,
                       internal_activation=False, consumer_lower_guard=True,
                       repeated_slot=False):
    """Already scheduled typed IR isolates conservative inference boundaries."""
    result = tirx.decl_buffer((1, 2, 2, 32, 32), "float32", name="Result")
    read_result = (tirx.decl_buffer(result.shape, result.dtype, name="ResultAlias", data=result.data)
                   if alias else result)
    accum = tirx.decl_buffer((32, 32), "float32", name="accum", scope="shared.l0c")
    value = tirx.decl_buffer((16, 32), "float32", name="value", scope="shared.dyn")
    e, owner, sub, sid = [tirx.Var(name, "int32") for name in ("e", "owner", "sub", "sid")]
    producer_slot = (e // 2) % 2 if repeated_slot else e % 2
    consumer_slot = ((e - 1) // 2) % 2 if repeated_slot else (e - 1) % 2
    height = 32 if complete else 16
    publish_body = copy(accum, result, src_shape=[height, 32],
                        dst_indices=[0, owner, producer_slot, 0, 0],
                        dst_shape=[1, 1, 1, height, 32])
    if internal_activation:
        publish_body = tirx.IfThenElse(e < 3, publish_body, None)
    publish = unit(publish_body, core=2)
    publish = unit(tirx.For(owner, 0, owner_extent, tirx.ForKind.SERIAL, publish), core=None,
                   guard=None if internal_activation else e < 3)
    read_owner = sid if matching_owner else 0
    read_body = copy(read_result, value,
                     src_indices=[0, read_owner, consumer_slot, sub * 16, 0],
                     src_shape=[1, 1, 1, 16, 32])
    if owner_extent == 1:
        # The second lane has an empty data domain; it still participates in
        # the conservative pair-wide completion outside this task predicate.
        read_body = tirx.IfThenElse(sid == 0, read_body, None)
    consumer_guard = (tirx.And(e >= 1, e < consumer_epochs + 1)
                      if consumer_lower_guard else e < consumer_epochs + 1)
    if internal_activation:
        read_body = tirx.IfThenElse(consumer_guard, read_body, None)
    read = unit(read_body, core=1)
    read_complete = complete if read_complete is None else read_complete
    read = unit(tirx.For(sub, 0, 2 if read_complete else 1, tirx.ForKind.SERIAL, read), core=None,
                guard=None if internal_activation else consumer_guard)
    # Already-scheduled control predicates belong inside their schedule-unit
    # marker, where DecodeGuardedNode records the guard delta for re-encoding.
    body = seq(publish, read)
    body = unit(tirx.For(e, 0, 4, tirx.ForKind.SERIAL, body), core=None)
    if reserved_flags:
        explicit = []
        for flag in range(reserved_flags):
            for operation, core, pipe in ((_CROSS_SET, 2, "PIPE_FIX"), (_CROSS_WAIT, 1, "PIPE_MTE2")):
                explicit.append(unit(tirx.Evaluate(tirx.Call(
                    "handle", tvm.ir.Op.get(operation), [2, pipe, flag])), core=core))
        body = seq(*explicit, body)
    annotations = {"vector_count": 2, "tl.manual_multi_buffer": {result.data: 2}}
    if private:
        annotations["tl.ascend_aiv_private_validated"] = [value.data]
    mod = kernel(body, buffers=[accum, value], params=[result], annotations=annotations)
    lane = tirx.IterVar(tvm.ir.Range(0, 2), sid, tirx.IterVar.ThreadIndex, "cthread")
    function = mod["main"].with_body(tirx.AttrStmt(lane, "thread_extent", 2, mod["main"].body))
    return tvm.IRModule({"main": function.with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


@pytest.mark.parametrize("complete,rows,internal_activation",
                         [(True, 32, False), (False, 16, False), (True, 32, True)],
                         ids=["full-allocation-plane", "complete-logical-subregion",
                              "activation-inside-atomic-task"])
def test_scheduled_typed_regions_infer_owner_and_physical_slot(complete, rows, internal_activation):
    _assert_owner_result(transform.InsertSync()(_scheduled_handoff(
        complete=complete, internal_activation=internal_activation)), rows=rows)


@pytest.mark.parametrize("kwargs", [
    {"private": False}, {"complete": False, "read_complete": True}, {"owner_extent": 1},
    {"matching_owner": False}, {"arch": "dav-3510"}, {"consumer_epochs": 2},
    {"alias": True}, {"reserved_flags": 12}, {"consumer_lower_guard": False},
    {"repeated_slot": True},
], ids=["no-private-proof", "read-beyond-produced-region", "missing-owner", "wrong-owner", "3510",
        "mismatched-active-interval", "distinct-buffer-alias", "owner-id-pressure",
        "consumer-has-extra-leading-epoch", "slot-reused-before-declared-ring-period"])
def test_unproven_owner_result_keeps_target_collective_contract(kwargs):
    after = transform.InsertSync()(_scheduled_handoff(**kwargs))
    assert not _owner_sets(after)
    events = calls(after, _CROSS_SET) + calls(after, _CROSS_WAIT)
    assert events
    mode = 4 if kwargs.get("arch") == "dav-3510" else 2
    assert {int(event.args[0]) for event in events} == {mode}
    assert any(event.args[1].value == "PIPE_FIX" for event in calls(after, _CROSS_SET))
    assert any(event.args[1].value == "PIPE_MTE2" for event in calls(after, _CROSS_WAIT))
