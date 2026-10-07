"""Closed dav-2201 MAD/FIX streams selected from ordinary scheduled tile ops.

These CPU tests check typed operands, control placement and finite event
counts. They do not turn a unit check into whole-pipe completion or simulate
the undocumented native unit state/granularity.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_nz_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


_GEMM = "tl.tileop.gemm"
_COPY = "tl.tileop.ascend_copy"
_DIRECTION = "tl.ascend_set_mmad_direction"
_CONFIG = "tl.ascend_set_fixpipe_config"
_ROOT_CONFIG = "tl.ascend.root_fixpipe_config"
_SET = "tl.ascend_set_flag"
_WAIT = "tl.ascend_wait_flag"
_SCALAR_VALUES = object()


def _fixture(*, kind="ordinary", native_k=3, whole_guard=False, prepared_inputs=False,
             repeated_readback=False, arch="dav-2201"):
    """Final scheduled native-L0 tasks, without solver/lowering prerequisites."""
    epochs = 4 if whole_guard else 3
    e, owner, result, k, reader = [tirx.Var(name, "int32")
                                   for name in ("epoch", "owner", "result", "native_k", "reader")]
    read_epoch, read_owner = (tirx.Var(name, "int32") for name in ("read_epoch", "read_owner"))
    a_gm = tirx.decl_buffer((epochs, 2, 32, 16), "float16", name="A")
    b_gm = tirx.decl_buffer(a_gm.shape, "float16", name="B")
    output = tirx.decl_buffer((epochs, 2, 4, 32, 32), "float32", name="Output")
    a_l1 = tirx.decl_buffer((32, 16), "float16", name="a_l1", scope="shared.l1")
    b_l1 = tirx.decl_buffer((32, 16), "float16", name="b_l1", scope="shared.l1")
    a = tirx.decl_buffer((32, 16), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((32, 16), "float16", name="b", scope="shared.l0b")
    c_shape = (2, 32, 32) if kind == "manual-version" else (32, 32)
    c = tirx.decl_buffer(c_shape, "float32", name="accum", scope="shared.l0c")
    other = tirx.decl_buffer((32, 32), "float32", name="other", scope="shared.l0c") if kind == "interleaved-C" else None
    feedback = tirx.decl_buffer((32, 32), "float32", name="feedback", scope="shared.l1")
    counter = tirx.decl_buffer((1,), "int32", name="input_epoch", scope="local.var") if prepared_inputs else None
    assert counter is None or native_k > 1
    stage = 1 if kind == "stage" else 0

    def c_region(buffer=c):
        ranges = [tvm.ir.Range.from_min_extent(0, extent) for extent in (32, 32)]
        if len(buffer.shape) == 3:
            ranges.insert(0, tvm.ir.Range.from_min_extent(result % 2, 1))
        return tirx.BufferRegion(buffer, ranges)

    def mad(buffer=c, clear=None):
        return tirx.Evaluate(T.gemm(a, b, c_region(buffer), transpose_B=True,
                                   clear_accum=(k == 0) if clear is None else clear))

    # The native K control has the exact copy/copy/MAD shape emitted by L1
    # lowering. Its source data need not vary with K to test C completion.
    k_tasks = [unit(copy(a_l1, a), core=2, stage=stage),
               unit(copy(b_l1, b), core=2, stage=stage),
               unit(mad(clear=False if kind == "read-first" else None), core=2, stage=stage,
                    guard=k < native_k - 1 if kind == "missing-terminal" else None)]
    k_annotations = {}
    if counter is not None:
        k_tasks.append(unit(tirx.BufferStore(counter, counter[0] + 1, [0]), core=2))
        k_annotations = {
            "multi_buffer_eligible": [a.data, b.data],
            "tl.multi_buffer_counter_map": {a.data: counter, b.data: counter},
            "tl.storage_epoch_guard_map": {a.data: tirx.const(True, "bool"), b.data: tirx.const(True, "bool")},
        }
    k_body = seq(*k_tasks)
    if native_k == 1:
        producer = unit(mad(clear=False if kind == "read-first" else True), core=2, stage=stage)
    else:
        producer = unit(tirx.For(k, 0, native_k, tirx.ForKind.SERIAL, k_body, annotations=k_annotations), core=None, stage=stage)

    width = 16 if kind == "partial-C" else 32
    source_indices = [result % 2, 0, 0] if len(c.shape) == 3 else [0, 0]
    source_shape = [1, 32, width] if len(c.shape) == 3 else [32, width]
    export = unit(copy(c, output, src_indices=source_indices, src_shape=source_shape,
                       dst_indices=[e, owner, result, 0, 0], dst_shape=[1, 1, 1, 32, width],
                       annotations={_ROOT_CONFIG: tirx.IntImm("int32", 1)}),
                  core=2, stage=stage,
                  guard=result < 3 if kind == "export-guard" else None)
    transactions = [producer, export]
    if kind == "interleaved-C":
        transactions.insert(1, unit(mad(other, clear=True), core=2, stage=stage))
    result_loop = unit(tirx.For(result, 0, 4, tirx.ForKind.SERIAL, seq(*transactions)),
                       core=None, stage=stage, guard=e < 2 if kind == "result-guard" else None)

    def readback(epoch_index, owner_index):
        return unit(tirx.For(reader, 0, 4, tirx.ForKind.SERIAL,
                             unit(copy(output, feedback, src_indices=[epoch_index, owner_index, reader, 0, 0],
                                       src_shape=[1, 1, 1, 32, 32]), core=2)), core=None,
                    guard=epoch_index < 2 if kind == "result-guard" else None)

    owner_body = seq(
        unit(copy(a_gm, a_l1, src_indices=[e, owner, 0, 0], src_shape=[1, 1, 32, 16]), core=2),
        unit(copy(b_gm, b_l1, src_indices=[e, owner, 0, 0], src_shape=[1, 1, 32, 16]), core=2),
        *([unit(copy(a_l1, a), core=2), unit(copy(b_l1, b), core=2)] if native_k == 1 else []),
        result_loop, *([readback(e, owner)] if repeated_readback else []),
    )
    owners = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, owner_body), core=None,
                  guard=e < 3 if whole_guard else None)
    outer = unit(tirx.For(e, 0, epochs, tirx.ForKind.SERIAL, owners), core=None)
    # The default GM consumer is one post-producer root phase. It tests the
    # actual FIX->MTE2 data dependency without requiring a per-activation
    # binary channel to be reusable through the selected C-data protocol.
    post_readback = unit(tirx.For(read_epoch, 0, epochs, tirx.ForKind.SERIAL,
                                 unit(tirx.For(read_owner, 0, 2, tirx.ForKind.SERIAL,
                                               readback(read_epoch, read_owner)), core=None,
                                      guard=read_epoch < 3 if whole_guard else None)), core=None)
    layouts = {a_l1: make_ascend_nz_layout(a_l1), b_l1: make_ascend_nz_layout(b_l1),
               a: make_ascend_l0a_2201_layout(a), b: make_ascend_major_k_layout(b),
               c: make_ascend_l0c_layout(c),
               feedback: make_ascend_nz_layout(feedback)}
    if other is not None:
        layouts[other] = make_ascend_l0c_layout(other)
    if kind == "missing-layout":
        del layouts[c]
    # At this prepared boundary a versions-map entry requires an automatic
    # owner loop. An ordinary compiler-owned single slot has no entry: its
    # fixed-one frontend request was consumed before InsertSync.
    annotations = {"layout_map": layouts, "tl.buffer_versions_map": {}}
    if counter is not None:
        annotations["tl.buffer_versions_map"] = {a.data: 1, b.data: 1}
    if kind == "manual-version":
        annotations["tl.manual_multi_buffer"] = {c.data: 2}
    setup = unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_CONFIG), [])), core=2)
    body = [setup]
    if counter is not None:
        body.append(unit(tirx.BufferStore(counter, 0, [0]), core=2))
    body.append(outer)
    if not repeated_readback:
        body.append(post_readback)
    before = kernel(seq(*body), buffers=[a_l1, b_l1, a, b, c, feedback] + ([other] if other is not None else [])
                    + ([counter] if counter is not None else []),
                    params=[a_gm, b_gm, output], annotations=annotations)
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    before["main"] = before["main"].with_attr("target", target)
    return before, {"c": c, "output": output, "epoch": e, "owner": owner,
                    "result": result, "k": k, "native_k": native_k, "counter": counter}


def _insert(before):
    with before["main"].attrs["target"]:
        return transform.InsertSync()(before)


def _scalar_cell(buffer, indices, env):
    assert buffer.scope() == "local.var" and buffer.dtype == "int32"
    assert tuple(_value(extent, env) for extent in buffer.shape) == (1,)
    assert tuple(_value(index, env) for index in indices) == (0,)
    return buffer.data


def _value(expr, env):
    def load_scalar(node):
        if isinstance(node, tirx.BufferLoad):
            cell = _scalar_cell(node.buffer, node.indices, env)
            assert cell in env[_SCALAR_VALUES], f"Uninitialized fixture scalar: {node}"
            return tirx.const(env[_SCALAR_VALUES][cell], node.dtype)

    resolved = tirx.stmt_functor.ir_transform(tirx.Evaluate(expr), None, load_scalar).value
    bindings = {var: value for var, value in env.items() if isinstance(var, tirx.Var)}
    value = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(resolved, bindings))
    assert isinstance(value, tirx.IntImm), f"Unbound finite fixture expression: {value}"
    return int(value)


def _name(call):
    return call.op.name if isinstance(call.op, tvm.ir.Op) else ""


def _trace(ir):
    """Expand finite scalar control, retaining actual loop identity at each op."""
    body = ir["main"].body if isinstance(ir, tvm.IRModule) else ir

    def visit(stmt, env, loops):
        if isinstance(stmt, tirx.SBlockRealize):
            assert _value(stmt.predicate, env)
            yield from visit(stmt.block.body, env, loops)
        elif isinstance(stmt, tirx.SBlock):
            yield from visit(stmt.body, env, loops)
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                yield from visit(child, env, loops)
        elif isinstance(stmt, tirx.AttrStmt):
            yield from visit(stmt.body, env, loops)
        elif isinstance(stmt, tirx.For):
            start, extent = _value(stmt.min, env), _value(stmt.extent, env)
            assert stmt.kind == tirx.ForKind.SERIAL and 0 <= extent <= 4
            for index in range(start, start + extent):
                local = {**env, stmt.loop_var: tirx.const(index, stmt.loop_var.dtype)}
                yield from visit(stmt.body, local, (*loops, stmt.loop_var))
        elif isinstance(stmt, tirx.IfThenElse):
            selected = stmt.then_case if _value(stmt.condition, env) else stmt.else_case
            if selected is not None:
                yield from visit(selected, env, loops)
        elif isinstance(stmt, tirx.Bind):
            env[stmt.var] = tirx.const(_value(stmt.value, env), stmt.var.dtype)
        elif isinstance(stmt, tirx.BufferStore):
            assert stmt.predicate is None
            cell = _scalar_cell(stmt.buffer, stmt.indices, env)
            env[_SCALAR_VALUES][cell] = _value(stmt.value, env)
        elif isinstance(stmt, tirx.Evaluate):
            if isinstance(stmt.value, tirx.Call):
                # Each issue sees the actual mutable counter value at that
                # instant; later commits must not mutate earlier snapshots.
                yield stmt.value, {**env, _SCALAR_VALUES: dict(env[_SCALAR_VALUES])}, loops
            else:
                assert _value(stmt.value, env) == 0
        else:
            pytest.fail(f"Unexpected unit-stream fixture statement: {stmt}")

    return list(visit(body, {_SCALAR_VALUES: {}}, ()))


def _is_export(call, facts):
    return (_name(call) == _COPY and
            call.args[0].args[0].buffer.data.same_as(facts["c"].data))


def _matching_completion(records, event, start, end):
    sets = [(index, _value(call.args[1], env)) for index, (call, env, _) in enumerate(records)
            if start < index < end and _name(call) == _SET and call.args[0].value == event]
    waits = [(index, _value(call.args[1], env)) for index, (call, env, _) in enumerate(records)
             if start < index < end and _name(call) == _WAIT and call.args[0].value == event]
    return any(sent == acquired and notify < wait for notify, sent in sets for wait, acquired in waits)


@pytest.mark.parametrize("native_k", [1, 3], ids=["single-MAD", "native-K-chain"])
@pytest.mark.parametrize("whole_guard", [False, True], ids=["unconditional", "whole-stream-guard"])
def test_full_result_stream_selects_units_and_drains_each_owner_epoch(native_k, whole_guard):
    before, facts = _fixture(native_k=native_k, whole_guard=whole_guard)
    after = _insert(before)
    records = _trace(after)
    transactions = {}
    activations = {}
    for index, (call, env, loops) in enumerate(records):
        if facts["epoch"] in env and facts["owner"] in env:
            activation = (_value(facts["epoch"], env), _value(facts["owner"], env))
            activations.setdefault(activation, []).append(index)
        if _name(call) == _GEMM or _is_export(call, facts):
            key = (*activation, _value(facts["result"], env))
            transactions.setdefault(key, []).append((call, env))
        if (_name(call) in (_SET, _WAIT) and call.args[0].value in ("M_FIX", "FIX_M")
                and facts["result"] in loops):
            # A true stream drain or parent owner/epoch dependency can be
            # projected to the first/last result. Intermediate results must
            # use the selected C-data protocol instead of per-result flags.
            assert _value(facts["result"], env) in (0, 3)

    assert set(transactions) == {(e, owner, result) for e in range(3) for owner in range(2) for result in range(4)}
    for sequence in transactions.values():
        mads, export = sequence[:-1], sequence[-1]
        assert len(mads) == native_k and all(_name(call) == _GEMM for call, _ in mads)
        assert [_value(call.annotations["unit_flag_ctrl"], env) for call, env in mads] == [2] * (native_k - 1) + [3]
        assert [_value(call.args[9], env) for call, env in mads] == [1] + [0] * (native_k - 1)
        call, env = export
        assert _is_export(call, facts) and _value(call.annotations["unit_flag_ctrl"], env) == 3
        assert tuple(_value(extent, env) for extent in call.args[0].args[2:]) == (32, 32)

    assert len(activations) == 6
    for positions in activations.values():
        mads = [i for i in positions if _name(records[i][0]) == _GEMM]
        exports = [i for i in positions if _is_export(records[i][0], facts)]
        directions = [(i, records[i][0].args[0].value) for i in positions if _name(records[i][0]) == _DIRECTION]
        assert [direction for _, direction in directions] == ["n", "m"]
        first, last = directions[0][0], directions[-1][0]
        assert first < min(mads) < max(exports) < last
        assert _matching_completion(records, "FIX_M", max(exports), last), "Missing real full stream drain"
        assert _matching_completion(records, "S_M", first, min(mads)), "Direction state needs ordinary completion"
        assert _matching_completion(records, "S_FIX", first, min(exports)), "Entry must protect ancestor MAD/FIX completion"
        assert _matching_completion(records, "M_S", max(exports), last), "Restore must follow the completed M drain"
        assert _matching_completion(records, "S_M", last, max(positions) + 1), "Restored direction must complete before exit"

    # The same operations without the actual C layout are deliberately
    # ineligible. Compare issued C events rather than banning the first/last
    # projections of real scope boundaries inside the selected result loop.
    fallback, _ = _fixture(kind="missing-layout", native_k=native_k, whole_guard=whole_guard)
    fallback_records = _trace(_insert(fallback))
    for event in ("M_FIX", "FIX_M"):
        selected_count = sum(_name(call) == _SET and call.args[0].value == event for call, _, _ in records)
        ordinary_count = sum(_name(call) == _SET and call.args[0].value == event for call, _, _ in fallback_records)
        assert selected_count < ordinary_count, (event, selected_count, ordinary_count)

    # Count issued ordinary event pairs only. This is not a binary/FIFO model
    # of the native unit protocol or proof that source-order issue completed.
    totals = {}
    for operation in (_SET, _WAIT):
        totals[operation] = Counter((call.args[0].value, _value(call.args[1], env))
                                    for call, env, _ in records if _name(call) == operation)
    assert totals[_SET] == totals[_WAIT]


def test_selected_unit_stream_keeps_root_spr_gm_and_input_dependencies():
    before, facts = _fixture()
    after = _insert(before)
    records = _trace(after)
    setup = next(i for i, (call, _, _) in enumerate(records) if _name(call) == _CONFIG)
    first_export = next(i for i, (call, _, _) in enumerate(records) if _is_export(call, facts))
    assert _matching_completion(records, "S_FIX", setup, first_export)
    exports = [i for i, (call, _, _) in enumerate(records) if _is_export(call, facts)]
    reloads = [i for i, (call, _, _) in enumerate(records) if _name(call) == _COPY
               and call.args[0].args[0].buffer.data.same_as(facts["output"].data)]
    assert len(exports) == len(reloads) == 24
    assert _matching_completion(records, "FIX_MTE2", max(exports), min(reloads))
    for event in ("MTE2_MTE1", "MTE1_M", "M_MTE1"):
        assert any(_name(call) == _SET and call.args[0].value == event for call, _, _ in records), event
        assert any(_name(call) == _WAIT and call.args[0].value == event for call, _, _ in records), event


def test_prepared_input_counters_preserve_their_real_clock_inside_selected_stream():
    before, facts = _fixture(prepared_inputs=True)
    after = _insert(before)
    records = _trace(after)
    mads = [(call, env) for call, env, _ in records if _name(call) == _GEMM]
    exports = [(call, env) for call, env, _ in records if _is_export(call, facts)]
    assert len(mads) == 3 * 2 * 4 * 3
    assert [_value(call.annotations["unit_flag_ctrl"], env) for call, env in mads] == [2, 2, 3] * 24
    assert [env[_SCALAR_VALUES][facts["counter"].data] for _, env in mads] == list(range(72))
    assert len(exports) == 24 and all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for call, env in exports)
    assert len(calls(after, _DIRECTION)) == 2
    for epoch in range(3):
        for owner in range(2):
            positions = [i for i, (_, env, _) in enumerate(records)
                         if facts["epoch"] in env and facts["owner"] in env
                         and _value(facts["epoch"], env) == epoch and _value(facts["owner"], env) == owner]
            last_export = max(i for i in positions if _is_export(records[i][0], facts))
            restore = next(i for i in positions if _name(records[i][0]) == _DIRECTION and records[i][0].args[0].value == "m")
            assert _matching_completion(records, "FIX_M", last_export, restore)
            assert _matching_completion(records, "M_S", last_export, restore)
            assert _matching_completion(records, "S_M", restore, max(positions) + 1)
    totals = [Counter((call.args[0].value, _value(call.args[1], env))
                      for call, env, _ in records if _name(call) == operation) for operation in (_SET, _WAIT)]
    assert totals[0] == totals[1]


def test_unit_selection_rolls_back_when_repeated_gm_channel_needs_ordinary_c_completion():
    # Full completion through ordinary MAD/FIX can close this repeated GM
    # channel's reverse reuse. C-data unit protection cannot stand in for that
    # completion. An otherwise admissible candidate must roll back atomically.
    before, _ = _fixture(repeated_readback=True)
    ordinary, _ = _fixture(kind="missing-layout", repeated_readback=True)
    after, baseline = _insert(before), _insert(ordinary)
    assert not calls(after, _DIRECTION)
    assert tvm.ir.structural_equal(after["main"].body.block.body, baseline["main"].body.block.body,
                                   map_free_vars=True)
    for call in calls(after, _GEMM) + calls(after, _COPY):
        control = call.annotations.get("unit_flag_ctrl")
        assert control is None or _value(control, {}) == 0
    records = _trace(after)
    totals = [Counter((call.args[0].value, _value(call.args[1], env))
                      for call, env, _ in records if _name(call) == operation) for operation in (_SET, _WAIT)]
    assert totals[0] == totals[1]


def test_exclusive_drain_budget_rolls_back_before_ordinary_capacity_diagnostic():
    def fixture(*, canonical_c):
        a_gm = tirx.decl_buffer((32, 16), "float16", name="A")
        b_gm = tirx.decl_buffer((32, 16), "float16", name="B")
        output = tirx.decl_buffer((9, 4, 32, 32), "float32", name="Output")
        a_l1 = tirx.decl_buffer((32, 16), "float16", name="a_l1", scope="shared.l1")
        b_l1 = tirx.decl_buffer((32, 16), "float16", name="b_l1", scope="shared.l1")
        a = tirx.decl_buffer((32, 16), "float16", name="a", scope="shared.l0a")
        b = tirx.decl_buffer((32, 16), "float16", name="b", scope="shared.l0b")
        c = tirx.decl_buffer((32, 32), "float32", name="accum", scope="shared.l0c")
        layouts = {a_l1: make_ascend_nz_layout(a_l1), b_l1: make_ascend_nz_layout(b_l1),
                   a: make_ascend_l0a_2201_layout(a), b: make_ascend_major_k_layout(b)}
        if canonical_c:
            layouts[c] = make_ascend_l0c_layout(c)
        setup = unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(_CONFIG), [])), core=2)
        tasks = [setup, unit(copy(a_gm, a_l1), core=2), unit(copy(b_gm, b_l1), core=2),
                 unit(copy(a_l1, a), core=2), unit(copy(b_l1, b), core=2)]
        for index in range(9):
            result = tirx.Var(f"result{index}", "int32")
            mad = unit(tirx.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)), core=2)
            export = unit(copy(c, output, dst_indices=[index, result, 0, 0], dst_shape=[1, 1, 32, 32],
                               annotations={_ROOT_CONFIG: tirx.IntImm("int32", 1)}), core=2)
            tasks.append(unit(tirx.For(result, 0, 4, tirx.ForKind.SERIAL, seq(mad, export)), core=None))
        before = kernel(seq(*tasks), buffers=[a_l1, b_l1, a, b, c],
                        params=[a_gm, b_gm, output], annotations={"layout_map": layouts, "tl.buffer_versions_map": {}})
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
        before["main"] = before["main"].with_attr("target", target)
        return before

    # Nine sequential result loops share one actual 4-KiB C allocation.
    # Every loop is independently admissible, but the selected streams need
    # nine private drain IDs, exceeding the eight hardware FIX_M slots.
    # Selection must roll back as a whole. The ordinary allocator also cannot
    # share all of these distinct loop clocks; its existing capacity diagnostic
    # must remain intact rather than being bypassed or spilled through 3510.
    after = _insert(fixture(canonical_c=True))
    baseline = _insert(fixture(canonical_c=False))
    assert not calls(after, _DIRECTION)
    assert tvm.ir.structural_equal(after["main"].body.block.body, baseline["main"].body.block.body,
                                   map_free_vars=True)
    for call in calls(after, _GEMM) + calls(after, _COPY):
        assert "unit_flag_ctrl" not in call.annotations

    for program in (after, baseline):
        with program["main"].attrs["target"]:
            with pytest.raises(ValueError, match="requires .* local event slots.*limit is 8.*Mutex spill requires dav-3510"):
                transform.RewriteFlagToBuf()(program)
    assert not calls(after, "tl.ascend_get_buf")
    assert not calls(after, "tl.ascend_rls_buf")
    records = _trace(after)
    assert sum(_name(call) == _GEMM for call, _, _ in records) == 36
    assert sum(_name(call) == _COPY and call.args[0].args[0].buffer.scope() == "shared.l0c"
               for call, _, _ in records) == 36
    totals = []
    for operation in (_SET, _WAIT):
        events = [(call.args[0].value, _value(call.args[1], env))
                  for call, env, _ in records if _name(call) == operation]
        assert events
        totals.append(Counter(events))
    assert totals[0] == totals[1]


@pytest.mark.parametrize("kind", ["partial-C", "interleaved-C", "stage", "manual-version",
                                  "export-guard", "missing-terminal", "read-first", "missing-layout"])
def test_unclosed_result_patterns_keep_unit_zero_and_full_c_events(kind):
    before, _ = _fixture(kind=kind)
    after = _insert(before)
    assert not calls(after, _DIRECTION)
    for call in calls(after, _GEMM) + calls(after, _COPY):
        control = call.annotations.get("unit_flag_ctrl")
        assert control is None or _value(control, {}) == 0
    for event in ("M_FIX", "FIX_M"):
        assert any(call.args[0].value == event for call in calls(after, _SET)), event
        assert any(call.args[0].value == event for call in calls(after, _WAIT)), event


def test_guarded_result_loop_keeps_ordinary_binary_contract_diagnostic():
    # The result alone can skip while the surrounding input loads execute.
    # This is not a closed stream. Its ordinary MTE1->M carried event also
    # lacks the supported outermost clock; selection must not bypass that
    # existing diagnostic to make an otherwise invalid schedule compile.
    before, _ = _fixture(kind="result-guard")
    with pytest.raises(tvm.error.InternalError,
                       match="shifted carried events require an unconditional outermost lexical loop.*pipe=MTE1->M"):
        _insert(before)


def test_3510_does_not_select_the_2201_stream_protocol():
    before, _ = _fixture(arch="dav-3510")
    after = _insert(before)
    assert not calls(after, _DIRECTION)
    assert all("unit_flag_ctrl" not in call.annotations or _value(call.annotations["unit_flag_ctrl"], {}) == 0
               for call in calls(after, _GEMM) + calls(after, _COPY))
