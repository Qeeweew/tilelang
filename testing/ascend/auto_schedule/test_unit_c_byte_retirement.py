"""Full native unit results can retire C bytes without retiring whole MADs.

The completion graph contains only emitted ordinary flags/barriers. Native
unit controls are checked separately; they never become whole-command edges.
These are CPU lowering regressions, with no CANN compilation or device run.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout, make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import (
    _lower, _matching_unit_boundary, _pairs, _root, _unit_stream_activations,
)
from testing.ascend.auto_schedule.test_l0c_phase_reuse import (
    _assert_storage_contract, _completion_commands, _program,
)
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_MAD = "tl.tileop.gemm"
_COPY = "tl.tileop.ascend_copy"
_DIRECTION = "tl.ascend_set_mmad_direction"
_SET, _WAIT = "tl.ascend_set_flag", "tl.ascend_wait_flag"


def _ordinary_graph(records):
    """Audit input whole-completion while leaving native C data out of HB."""
    trace = []

    def blocks(arg, env):
        buffer, origin, extent = _region(arg, env)
        assert all(value == 1 for value in extent[:-2])
        assert all(value % 16 == 0 for value in (*origin[-2:], *extent[-2:]))
        return tuple((buffer.data, *origin[:-2], row, column)
                     for row in range(origin[-2] // 16, (origin[-2] + extent[-2]) // 16)
                     for column in range(origin[-1] // 16, (origin[-1] + extent[-1]) // 16))

    for call, env, _ in records:
        name = _name(call)
        if name in (_SET, _WAIT):
            event, flag = call.args[0].value, _value(call.args[1], env)
            assert 0 <= flag < 8
            source, destination = event.split("_")
            setting = name == _SET
            trace.append(_Command(source if setting else destination,
                                  "set" if setting else "wait", (event, flag)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name == _MAD:
            trace.append(_Command("M", "mad", reads=(*blocks(call.args[0], env), *blocks(call.args[1], env))))
        elif name == _COPY:
            src, _, _ = _region(call.args[0], env)
            if src.scope() == "shared.l0c":
                trace.append(_Command("FIX", "export"))
            else:
                dst, _, _ = _region(call.args[1], env)
                pipe = "MTE2" if src.scope() == "global" else "MTE1"
                assert dst.scope() in ("shared.l1", "shared.l0a", "shared.l0b")
                trace.append(_Command(pipe, "copy", reads=blocks(call.args[0], env), writes=blocks(call.args[1], env)))
        else:
            assert name in (_DIRECTION, "tl.ascend_set_fixpipe_config", "tirx.call_extern"), name
            trace.append(_Command("S", "scalar"))
    return trace, _prove_completions(trace)


def _assert_closed_byte_protocol(records, require, result_loops, *, barrier):
    activations = _unit_stream_activations(records, result_loops)
    directions = [(i, call.args[0].value, loops) for i, (call, _, loops) in enumerate(records)
                  if _name(call) == _DIRECTION]
    assert [(value, loops) for _, value, loops in directions] == [("n", ()), ("m", ())]
    setup, restore = directions[0][0], directions[1][0]
    first = activations[0]["mads"][0][0]
    last = activations[-1]["exports"][-1][0]
    for event in ("S_M", "S_FIX"):
        assert _matching_unit_boundary(records, event, setup, first, root=True)
    assert _matching_unit_boundary(records, "M_S", last, restore, root=True)
    barriers = [i for i, (call, _, _) in enumerate(records)
                if _name(call) == "tl.ascend_pipe_barrier" and call.args[0].value == "PIPE_M"]
    assert bool(barriers) == barrier
    for index, activation in enumerate(activations):
        mads, exports, prefix = activation["mads"], activation["exports"], activation["prefix"]
        previous = activations[index - 1] if index else None
        following = activations[index + 1] if index + 1 < len(activations) else None
        begin = previous["exports"][-1][0] if previous else setup
        end = following["mads"][0][0] if following else restore
        assert _matching_unit_boundary(records, "M_FIX", begin, exports[0][0],
                                       notify_before=mads[0][0], prefix=prefix)
        assert _matching_unit_boundary(records, "FIX_M", exports[-1][0], end, prefix=prefix)
        # Matching full exports protect C bytes across activations. No
        # whole-MAD->FIX edge or old-MAD->new-MAD edge is assumed here.
        if following:
            for old, *_ in exports:
                require(3 * old + 2, 3 * following["mads"][0][0] + 1)
        for work, *_ in (*mads, *exports):
            require(3 * setup + 2, 3 * work + 1)
            require(3 * work + 2, 3 * restore + 1)
    return activations


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_dsl_keeps_independent_c_pitches_and_ab_completion_without_owner_m_barrier(dtype):
    artifact, snapshots = _lower(_program(dtype, True))
    _assert_storage_contract(artifact, snapshots)
    lowered = snapshots["tl.LowerScheduledTIR"]
    records = list(_trace(lowered, core=2, lane=0))
    trace, _ = _completion_commands(lowered, unit_streams=True)
    require = _prove_completions(trace)  # Includes both physical A/B banks.
    activations = _assert_closed_byte_protocol(records, require, {"qk_n", "row_part"}, barrier=False)
    counts = Counter()
    for activation in activations:
        mads, exports = activation["mads"], activation["exports"]
        (storage,) = {_region(call.args[2], env)[0].name for _, call, env, _ in mads}
        counts[storage] += 1
        qk = storage == "cq"
        assert len(exports) == (4 if qk else 2)
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env, _ in mads] == (
            [3] * 4 if qk else [2, 2, 3] * 2)
        assert [_value(call.args[9], env) for _, call, env, _ in mads] == (
            [1] * 4 if qk else [1, 0, 0] * 2)
        assert all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for _, call, env, _ in exports)
    assert counts == {"cq": 4, "cp": 4}
    source = artifact.kernel_source
    assert source.count('asc_set_mmad_direction_n();') == source.count('asc_set_mmad_direction_m();') == 1
    assert 'asc_sync_pipe(PIPE_M);' not in source


def test_shifted_independent_c_results_keep_full_fix_retirement_without_m_barrier():
    before, (cq, cp) = _fixture()

    def post(node):
        if isinstance(node, tirx.For) and node.loop_var.name == "epoch":
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                            annotations={"enable_offset": True,
                                         "tl.ascend_preserve_issue_order": True})
        if (isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
                and isinstance(node.body, tirx.For) and node.body.loop_var.name == "pv_owner"):
            return tirx.AttrStmt({"stage": 1}, node.attr_key, node.value, node.body)

    before["main"] = before["main"].with_body(
        tirx.stmt_functor.ir_transform(before["main"].body, None, post))
    scheduled, _, lowered = _through_sync(before)
    # Capacity admission keeps C independent. Full-result completion still
    # retires each C lifetime without claiming whole-MAD retirement.
    assert frozenset((cq.data, cp.data)) not in _pairs(scheduled)
    records = list(_trace(lowered, core=2, lane=0))
    _, require = _ordinary_graph(records)
    activations = _assert_closed_byte_protocol(records, require,
                                               {"qk_result", "pv_result"}, barrier=False)
    assert len(activations) == 12


def _fixture(*, native_k=3, epochs=3, active_first=0, active_count=3, kind="full",
             epoch_name="epoch"):
    """Two genuine scheduled result streams; AutoSchedule chooses the pair."""
    epoch = tirx.Var(epoch_name, "int32")
    qshape = (32, 64) if kind == "small-output" else (64, 128)
    pshape = (32, 256 if kind == "prefix" else 128)
    cq = tirx.decl_buffer(qshape, "float32", name="cq", scope="shared.l0c")
    cp = tirx.decl_buffer(pshape, "float32", name="cp", scope="shared.l0c")
    buffers, layouts, phases, outputs = [cq, cp], {}, [], []
    layouts.update({c: make_ascend_l0c_layout(c) for c in (cq, cp)})
    active = tirx.And(epoch >= active_first, epoch < active_first + active_count)
    for label, c, k_size, transpose in (("qk", cq, 16, True), ("pv", cp, 32, False)):
        m, n = int(c.shape[0]), int(c.shape[1])
        if kind == "prefix" and label == "pv":
            n //= 2
        a = tirx.decl_buffer((m, k_size), "float16", name=label + "_a", scope="shared.l0a")
        b = tirx.decl_buffer((n, k_size) if transpose else (k_size, n), "float16", name=label + "_b", scope="shared.l0b")
        sa, sb = [tirx.decl_buffer(value.shape, "float16", name=value.name + "_l1", scope="shared.l1") for value in (a, b)]
        buffers.extend((a, b, sa, sb))
        layouts.update({a: make_ascend_l0a_2201_layout(a),
                        b: make_ascend_major_k_layout(b) if transpose else make_ascend_major_mn_layout(b)})
        owner, result, k = [tirx.Var(label + suffix, "int32") for suffix in ("_owner", "_result", "_k")]
        out = tirx.decl_buffer((epochs, 2, 3, m, n), "float32", name=label + "_out")
        outputs.append(out)
        c_region = tirx.BufferRegion(c, [tvm.ir.Range.from_min_extent(0, m), tvm.ir.Range.from_min_extent(0, n)])
        mad = unit(tirx.Evaluate(T.gemm(a, b, c_region, transpose_B=transpose,
                                      clear_accum=True if native_k == 1 else k == 0)), core=2, cost=(2, 1))
        if native_k != 1:
            mad = unit(tirx.For(k, 0, native_k, tirx.ForKind.SERIAL, mad), core=None)
        width = n // 2 if kind == "partial-fix" and label == "pv" else n
        export = unit(copy(c, out, src_shape=[m, width], dst_indices=[epoch, owner, result, 0, 0],
                           dst_shape=[1, 1, 1, m, width]), core=2, cost=(1, 1))
        results = unit(tirx.For(result, 0, 3, tirx.ForKind.SERIAL, seq(mad, export)), core=None)
        phase = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, seq(
            unit(copy(sa, a), core=2, cost=(1, 1)), unit(copy(sb, b), core=2, cost=(1, 1)), results)),
            core=None, guard=active)
        phases.append(phase)
    if kind == "third-C":
        third = tirx.decl_buffer((32, 16), "float32", name="third", scope="shared.l0c")
        buffers.append(third)
        layouts[third] = make_ascend_l0c_layout(third)
    body = unit(tirx.For(epoch, 0, epochs, tirx.ForKind.SERIAL, seq(*phases)), core=None)
    if kind == "opaque":
        body = seq(unit(tirx.Evaluate(tirx.call_extern("int32", "unknown_state")), core=2, cost=(1, 1)), body)
    mod = kernel(body, buffers=buffers, params=outputs, annotations={
        "layout_map": layouts, "tl.buffer_versions_map": {cq.data: 1, cp.data: 1}})
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, (cq, cp)


def _through_sync(before):
    with before["main"].attrs["target"]:
        scheduled = transform.AutoSchedule()(before)
        resolved = transform.ResolveCore()(transform.PrepareMultiBuffer()(transform.AssignCore()(scheduled)))
        inserted = transform.InsertSync()(resolved)
        lowered = transform.LowerScheduledTIR()(transform.MaterializeMultiBuffer()(inserted))
    return scheduled, inserted, lowered


@pytest.mark.parametrize("native_k,epochs,first,count", [(1, 1, 0, 1), (3, 3, 0, 3), (3, 5, 1, 3)],
                         ids=["single-result-epoch", "odd-K-and-epochs", "inactive-prefix-and-suffix"])
def test_finite_full_result_pair_retires_c_bytes_at_matching_fix(native_k, epochs, first, count):
    before, (cq, cp) = _fixture(native_k=native_k, epochs=epochs, active_first=first, active_count=count)
    scheduled, inserted, lowered = _through_sync(before)
    assert frozenset((cq.data, cp.data)) in _pairs(scheduled)
    aliases = _root(inserted).annotations["tl.buffer_alias_map"]
    assert cq.data in aliases[cp.data] and cp.data in aliases[cq.data]
    records = list(_trace(lowered, core=2, lane=0))
    _, require = _ordinary_graph(records)
    activations = _assert_closed_byte_protocol(records, require, {"qk_result", "pv_result"}, barrier=False)
    assert len(activations) == count * 4
    for activation in activations:
        mads, exports = activation["mads"], activation["exports"]
        assert len(mads) == 3 * native_k and len(exports) == 3
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env, _ in mads] == (
            [2] * (native_k - 1) + [3]) * 3
        assert [_value(call.args[9], env) for _, call, env, _ in mads] == ([1] + [0] * (native_k - 1)) * 3
        assert all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for _, call, env, _ in exports)
        for _, call, env, _ in (*mads, *exports):
            c, origin, extent = _region(call.args[2] if _name(call) == _MAD else call.args[0], env)
            assert origin == (0, 0) and extent == tuple(map(int, c.shape))
        assert all(first <= dict((var.name, value) for var, value, _ in loops)["epoch"] < first + count
                   for _, _, _, loops in (*mads, *exports))


@pytest.mark.parametrize("kind", ["prefix", "third-C", "small-output", "partial-fix", "opaque"])
def test_unproved_c_byte_domains_keep_the_old_retirement_protocol(kind):
    before, (cq, cp) = _fixture(kind=kind)
    scheduled, inserted, lowered = _through_sync(before)
    assert (frozenset((cq.data, cp.data)) in _pairs(scheduled)) == (kind == "small-output")
    records = list(_trace(lowered, core=2, lane=0))
    _, require = _ordinary_graph(records)
    directions = [(call.args[0].value, loops) for call, _, loops in records if _name(call) == _DIRECTION]
    if kind in ("prefix", "third-C", "small-output"):
        # Both native streams remain eligible, but neither has the exact
        # mandatory two-C/full-allocation byte-reuse contract.
        _assert_closed_byte_protocol(records, require, {"qk_result", "pv_result"}, barrier=True)
    else:
        # Missing matching full FIX or an opaque SPR writer cannot receive
        # the invariant root protocol. Preserve its ordinary/scoped fallback.
        assert directions != [("n", ()), ("m", ())]
    # Independent C allocations have independent retirement obligations. If
    # the allocator actually aliases them, retain the stronger shared ledger.
    aliases = _root(inserted).annotations.get("tl.buffer_alias_map", {})
    shared_c = cq.data in aliases.get(cp.data, ())
    retired = {}

    def storage_key(arg, env):
        storage = _region(arg, env)[0].data
        return frozenset((cq.data, cp.data)) if shared_c else storage

    for i, (call, env, loops) in enumerate(records):
        if _name(call) == _MAD:
            (depth,) = [depth for depth, (var, _, _) in enumerate(loops)
                        if var.name in ("qk_result", "pv_result")]
            activation = (loops[depth][0].name,
                          tuple((var.name, value) for var, value, _ in loops[:depth]))
            key = storage_key(call.args[2], env)
            previous = retired.get(key)
            if previous is not None and activation != previous["activation"]:
                assert previous["fix"] is not None
                require(3 * previous["fix"] + 2, 3 * i + 1)
                for old in previous["mads"]:
                    require(3 * old + 2, 3 * i + 1)
            if previous is None or activation != previous["activation"]:
                previous = retired[key] = {"activation": activation, "mads": [], "fix": None}
            previous["mads"].append(i)
        elif _name(call) == _COPY and _region(call.args[0], env)[0].scope() == "shared.l0c":
            retired[storage_key(call.args[0], env)]["fix"] = i


def test_runtime_empty_group_keeps_conservative_uf_and_drains_binary_flags():
    # This is a genuine runtime group guard, rather than a contradictory
    # compile-time interval. Keep the group clock outermost, as required for
    # carried binary events. A third C allocation deliberately preserves the
    # conservative UF path instead of certifying two-C byte retirement.
    before, (cq, cp) = _fixture(epochs=2, active_count=1, kind="third-C", epoch_name="group")
    scheduled, _, lowered = _through_sync(before)
    assert frozenset((cq.data, cp.data)) not in _pairs(scheduled)
    records = list(_trace(lowered, core=2, lane=0))
    _, require = _ordinary_graph(records)  # Includes both groups and root drain.
    activations = _assert_closed_byte_protocol(records, require, {"qk_result", "pv_result"}, barrier=True)
    assert len(activations) == 4
    mads, exports = Counter(), Counter()
    for call, env, loops in records:
        group = next((value for var, value, _ in loops if var.name == "group"), None)
        if _name(call) == _MAD:
            mads[group] += 1
        elif _name(call) == _COPY and _region(call.args[0], env)[0].scope() == "shared.l0c":
            exports[group] += 1
    assert mads == {0: 2 * 2 * 3 * 3}
    assert exports == {0: 2 * 2 * 3}
