"""Different canonical C pitches share bytes only after phase completion.

The DSL keeps QK C128x256 and PV C64x64 independent through native lowering.
Fixture task costs are unitless scheduler weights, not hardware measurements.
These regressions lower on the CPU; they neither compile CANN nor use a device.
"""

from collections import Counter, defaultdict
import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout, make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, region as tile_region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import (
    _assert_kernel_unit_boundaries, _bytes, _lower, _pairs, _root, _unit_stream_activations,
)
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_TARGET = {"kind": "ascend", "arch": "dav-2201"}
_MARKER = "tl.ascend_l0_phase_reuse_pairs"
_GEMM = "tl.tileop.gemm"
_COPY = "tl.tileop.ascend_copy"
_LOAD3D = "tl.tileop.ascend_cube_load3d"


def _program(dtype, unit_streams):
    @T.prim_func
    def main(Q: T.Tensor((3, 2, 128, 64), dtype),
             K: T.Tensor((3, 2, 1024, 64), dtype),
             P: T.Tensor((3, 2, 128, 768), dtype),
             V: T.Tensor((3, 2, 768, 64), dtype),
             Score: T.Tensor((3, 2, 128, 1024), "float32"),
             Result: T.Tensor((3, 2, 128, 64), "float32")):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((128, 64), dtype)
            k_l1 = T.alloc_l1((1024, 64), dtype)
            p_l1 = T.alloc_l1((64, 256), dtype)
            v_l1 = T.alloc_l1((768, 64), dtype)
            aq = T.alloc_l0a((128, 64), dtype)
            bq = T.alloc_l0b((256, 64), dtype)
            cq = T.alloc_l0c((128, 256), "float32")
            cp = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({q_l1: 1, k_l1: 1, v_l1: 1, aq: 1, bq: 1, cq: 1, cp: 1})
            if not unit_streams:
                # Isolate ordinary whole-M/FIX completion from the separately
                # checked native unit protocol; explicit unit0 is not opt-out.
                T.set_mmad_direction("m")
            for epoch in T.serial(3):
                with T.Stage(0):
                    if epoch < 2:
                        for qk_owner in T.serial(2):
                            T.copy(Q[epoch, qk_owner, :, :], q_l1)
                            T.copy(K[epoch, qk_owner, :, :], k_l1)
                            T.copy(q_l1, aq)
                            for qk_n in T.serial(4):
                                T.copy(k_l1[qk_n * 256:(qk_n + 1) * 256, :], bq)
                                T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
                                T.copy(cq, Score[epoch, qk_owner, :, qk_n * 256:(qk_n + 1) * 256])
                with T.Stage(0):
                    if epoch < 2:
                        for pv_owner in T.serial(2):
                            T.copy(V[epoch, pv_owner, :, :], v_l1)
                            for row_part in T.serial(2):
                                for pv_chunk in T.Pipelined(3, max_stages=4):
                                    T.copy(P[epoch, pv_owner, row_part * 64:(row_part + 1) * 64,
                                             pv_chunk * 256:(pv_chunk + 1) * 256], p_l1)
                                    T.gemm(p_l1, v_l1[pv_chunk * 256:(pv_chunk + 1) * 256, :],
                                           cp, transpose_B=False, clear_accum=pv_chunk == 0)
                                T.copy(cp, Result[epoch, pv_owner, row_part * 64:(row_part + 1) * 64, :])
    return main


def _assert_storage_contract(artifact, snapshots):
    before = snapshots["tl.AnnotateMultiBufferEligible"]
    scheduled, physical = snapshots["tl.AutoSchedule"], snapshots["tl.MaterializeMultiBuffer"]
    cq, cp = [allocated_buffer(scheduled, name) for name in ("cq", "cp")]
    assert not cq.data.same_as(cp.data)
    assert _bytes(cq) + _bytes(cp) == 144 * 1024
    assert frozenset((cq.data, cp.data)) in _pairs(scheduled)
    versions = _root(scheduled).annotations["tl.buffer_versions_map"]
    for c in (cq, cp):
        assert allocated_buffer(before, c.name).same_as(c)
        assert int(_root(before).annotations["tl.buffer_versions_map"][c.data]) == 1
        assert c.data not in versions
        assert all(c.data not in loop.annotations.get("multi_buffer_eligible", [])
                   for loop in nodes(scheduled, tirx.For))
        tvm.ir.assert_structural_equal(_root(scheduled).annotations["layout_map"][c],
                                       make_ascend_l0c_layout(c))
        assert tuple(map(int, allocated_buffer(physical, c.name).shape)) == tuple(map(int, c.shape))
    aliases = _root(snapshots["tl.InsertSync"]).annotations["tl.buffer_alias_map"]
    assert cp.data in aliases[cq.data] and cq.data in aliases[cp.data]
    assert _MARKER not in _root(snapshots["tl.InsertSync"]).annotations

    a, b = [allocated_buffer(scheduled, name) for name in ("gemm_l0a_0", "gemm_l0b_0")]
    assert tuple(map(int, a.shape)) == (64, 256) and tuple(map(int, b.shape)) == (256, 64)
    # The repeated PV A64 group saves more descriptors than QK A128.
    # Keep QK's original copy and select both independent PV banks.
    loads = {call.args[0].args[0].buffer.name: call
             for call in calls(scheduled, _LOAD3D)}
    assert set(loads) == {"p_l1", "v_l1"}
    assert loads["p_l1"].args[1].args[0].buffer.same_as(a)
    assert loads["v_l1"].args[1].args[0].buffer.same_as(b)
    assert tuple(map(int, loads["p_l1"].args[2:])) == (64, 256)
    assert tuple(map(int, loads["v_l1"].args[2:])) == (768, 64)
    (config,) = calls(scheduled, "tl.ascend_set_load3d_config")
    assert tuple(map(int, config.args)) == (64, 768)
    assert {call.args[1].args[0].buffer.data for call in calls(scheduled, _COPY)
            if call.args[1].args[0].buffer.scope() == "shared.l0a"} == {
                allocated_buffer(scheduled, "aq").data}
    assert all(int(versions[buffer.data]) == 2 for buffer in (a, b))
    assert all(tuple(map(int, allocated_buffer(physical, buffer.name).shape)) == (2, *map(int, buffer.shape))
               for buffer in (a, b))
    for buffer, layout in ((a, make_ascend_l0a_2201_layout), (b, make_ascend_major_mn_layout)):
        tvm.ir.assert_structural_equal(_root(scheduled).annotations["layout_map"][buffer], layout(buffer))
    for first_name, second in (("aq", a), ("bq", b)):
        first = allocated_buffer(scheduled, first_name)
        assert frozenset((first.data, second.data)) in _pairs(scheduled)
        assert second.data in aliases[first.data] and first.data in aliases[second.data]

    native = [call for function in artifact.device_mod.functions.values() for call in calls(function, "tl.ascend_mad")]
    assert len(native) == 2
    by_m = {int(call.args[3]): call for call in native}
    assert set(by_m) == {64, 128}
    assert tuple(int(by_m[128].args[index]) for index in (3, 4, 5)) == (128, 64, 256)
    assert tuple(int(by_m[64].args[index]) for index in (3, 4, 5)) == (64, 256, 64)
    c_pointers = [mad.args[0] for mad in by_m.values()]
    assert all(pointer.op.name == "tirx.tvm_access_ptr" and int(pointer.args[2]) == 0
               for pointer in c_pointers)
    assert c_pointers[0].args[1].same_as(c_pointers[1].args[1])
    fixes = [call for function in artifact.device_mod.functions.values()
             for call in calls(function, "tl.ascend_copy_matrix_cc_to_gm")]
    assert len(fixes) == 2
    assert {(int(call.args[4]), int(call.args[3]), int(call.args[6])) for call in fixes} == {
        (128, 256, 128), (64, 64, 64)}
    assert all(call.args[1].args[1].same_as(c_pointers[0].args[1]) and int(call.args[1].args[2]) == 0
               for call in fixes)
    for scope, size in (("shared.l0a.dyn", 64), ("shared.l0b.dyn", 64), ("shared.l0c.dyn", 128)):
        (bank,) = [allocation.buffer for function in artifact.device_mod.functions.values()
                   for allocation in nodes(function, tirx.AllocBuffer) if allocation.buffer.scope() == scope]
        assert _bytes(bank) == size * 1024
    return cq, cp


def _completion_commands(mod, unit_streams=False):
    """Keep C unit checks out of the whole-command completion graph."""
    trace, c_work, exports = [], [], Counter()
    banks = {"aq": "A", "gemm_l0a_0": "A", "bq": "B", "gemm_l0b_0": "B", "cq": "C", "cp": "C"}

    def blocks(arg, env):
        buffer, origin, extent = _region(arg, env)
        assert all(value == 1 for value in extent[:-2])
        if buffer.name in banks:
            # Each local endpoint is a full canonical tile/version. Both C
            # pitches occupy contiguous bytes, without identifying layouts.
            assert origin[-2:] == (0, 0) and extent[-2:] == tuple(map(int, buffer.shape[-2:]))
            count = math.prod(map(int, buffer.shape[-2:])) // 256
            start = (origin[0] if len(origin) == 3 else 0) * count
            return tuple((banks[buffer.name], block) for block in range(start, start + count))
        return tuple((buffer.data, *origin[:-2], row, column)
                     for row in range(origin[-2] // 16, (origin[-2] + extent[-2]) // 16)
                     for column in range(origin[-1] // 16, (origin[-1] + extent[-1]) // 16))

    for call, env, _ in _trace(mod, core=2, lane=0):
        name, index = _name(call), len(trace)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            flag = _value(call.args[1], env)
            assert 0 <= flag < 8
            setting = name == "tl.ascend_set_flag"
            trace.append(_Command(source if setting else destination, "set" if setting else "wait", (event, flag)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name == _GEMM:
            c, _, _ = _region(call.args[2], env)
            if not unit_streams:
                assert "unit_flag_ctrl" not in call.annotations or _value(call.annotations["unit_flag_ctrl"], env) == 0
            trace.append(_Command("M", "mad", reads=(*blocks(call.args[0], env), *blocks(call.args[1], env))))
            c_work.append((index, "mad", c.name, blocks(call.args[2], env), _value(call.args[9], env)))
        elif name in (_COPY, _LOAD3D):
            src, _, _ = _region(call.args[0], env)
            dst, dest_origin, _ = _region(call.args[1], env)
            pipe = "MTE2" if src.scope() == "global" else "FIX" if src.scope() == "shared.l0c" else "MTE1"
            if pipe == "FIX":
                assert src.name in ("cq", "cp") and dst.name in ("Score", "Result")
                epoch, owner, row, column = dest_origin
                assert epoch in (0, 1) and owner in (0, 1)
                if src.name == "cq":
                    assert row == 0 and column in (0, 256, 512, 768)
                else:
                    assert row in (0, 64) and column == 0
                exports[dst.name, epoch, owner, row, column] += 1
                trace.append(_Command(pipe, "export"))
                c_work.append((index, "export", src.name, blocks(call.args[0], env), None))
            else:
                trace.append(_Command(pipe, "copy", reads=blocks(call.args[0], env), writes=blocks(call.args[1], env)))
        else:
            assert name in ("tl.ascend_set_mmad_direction", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_cube_load_config", "tl.ascend_set_load3d_config"), name
            trace.append(_Command("S", "scalar"))
    assert exports == Counter({("Score", e, owner, 0, n * 256): 1 for e in range(2) for owner in range(2) for n in range(4)} | {
        ("Result", e, owner, row * 64, 0): 1 for e in range(2) for owner in range(2) for row in range(2)})
    return trace, c_work


def _ordinary_completion(mod):
    """Use real fences/acquires, with no cross-pipe issue-order completion."""
    trace, c_work = _completion_commands(mod)
    require = _prove_completions(trace)
    pending, old_readers, crossings = defaultdict(list), {}, Counter()
    for index, kind, storage, touched, clear in c_work:
        if kind == "mad":
            if clear:
                assert not pending[storage]
                for block in touched:
                    if block in old_readers:
                        previous, previous_storage = old_readers[block]
                        require(3 * previous + 2, 3 * index + 1)
                        if previous_storage != storage:
                            crossings[previous_storage, storage] += 1
            else:
                assert pending[storage], "Accumulation must follow this result's full clear"
            pending[storage].append(index)
        else:
            chain = pending.pop(storage)
            assert len(chain) == (1 if storage == "cq" else 3)
            # A source M fence joins the whole K chain before FIX may read C.
            # No unit check or same-M issue order is treated as full completion.
            for producer in chain:
                require(3 * producer + 2, 3 * index + 1)
            for block in touched:
                old_readers[block] = index, storage
    assert not any(pending.values())
    assert crossings["cq", "cp"] and crossings["cp", "cq"]


def _unit_completion(mod):
    # Ordinary local flags and A/B/L1 reuse still require real completion.
    # C writes/reads are omitted from this graph because a native unit check
    # supplies C-data readiness, not whole-MAD or whole-FIX completion.
    trace, _ = _completion_commands(mod, unit_streams=True)
    _prove_completions(trace)
    records = list(_trace(mod, core=2, lane=0))
    activations = _unit_stream_activations(records, {"qk_n", "row_part"})
    # This exact full-allocation pair uses matching FIX completion for C
    # bytes; the helper still checks every real FIX->next-MAD and root gate.
    _assert_kernel_unit_boundaries(records, activations, c_byte_retirement=True)
    seen = Counter()
    for item in activations:
        mads, fixes = item["mads"], item["exports"]
        storages = {_region(call.args[2], env)[0].name for _, call, env, _ in mads}
        assert len(storages) == 1
        (storage,) = storages
        seen[storage] += 1
        expected = [3] * 4 if storage == "cq" else [2, 2, 3] * 2
        assert len(fixes) == (4 if storage == "cq" else 2)
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env, _ in mads] == expected
        assert [_value(call.args[9], env) for _, call, env, _ in mads] == ([1] * 4 if storage == "cq" else [1, 0, 0] * 2)
        assert all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for _, call, env, _ in fixes)
    assert seen == {"cq": 4, "cp": 4}


@pytest.mark.parametrize("dtype,unit_streams", [("float16", False), ("bfloat16", True)], ids=["ordinary-completion", "native-unit-completion"])
def test_independent_c_pitches_share_late_bytes_with_completed_phases(dtype, unit_streams):
    artifact, snapshots = _lower(_program(dtype, unit_streams))
    _assert_storage_contract(artifact, snapshots)
    if unit_streams:
        assert Counter(call.args[0].value for call in calls(snapshots["tl.InsertSync"], "tl.ascend_set_mmad_direction")) == {"n": 1, "m": 1}
        _unit_completion(snapshots["tl.LowerScheduledTIR"])
    else:
        _ordinary_completion(snapshots["tl.LowerScheduledTIR"])


def test_manual_ancestor_reserves_the_selected_childs_full_physical_rings(monkeypatch):
    queries = []
    schedule = z3_scheduler.z3_schedule_loop_python

    def observe(*args, **kwargs):
        # Keep the real solver, including its stage/version refinements. The
        # manual parent has no variable storage of its own, but must reserve
        # the automatic PV child's two physical A/B slots and the shared C.
        result = schedule(*args, **kwargs)
        manual = bool(kwargs["manual_schedule"])
        if manual and not list(args[8]):
            groups = [tuple(map(int, group)) for group in args[9]]
            queries.append(groups)
        return result

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", observe)
    artifact, snapshots = _lower(_program("float16", False))
    _assert_storage_contract(artifact, snapshots)
    # Memory groups have the scheduler's ordered scope keys: shared, L0A,
    # L0B, L0C, L1. All three L0 banks are exactly occupied by the selected
    # maximum rings (64/64/128 KiB), never a sum of the two phase allocations.
    assert any(len(groups) == 5 and all(groups[index] == (0,) for index in (1, 2, 3))
               for groups in queries), queries


def _admission_fixture(kind="ordinary", arch="dav-2201", pressure=False, accumulation=None,
                       guarded_phases=False, direct_definition=None):
    qm, qn, pm, pn = (128, 256, 64, 64) if pressure else (32, 64, 16, 32)
    aq = tirx.decl_buffer((qm, 16), "float16", name="aq", scope="shared.l0a")
    bq = tirx.decl_buffer((qn, 16), "float16", name="bq", scope="shared.l0b")
    ap = tirx.decl_buffer((pm, 32), "float16", name="ap", scope="shared.l0a")
    bp = tirx.decl_buffer((32, pn), "float16", name="bp", scope="shared.l0b")
    cq = tirx.decl_buffer((qm, qn), "float32", name="cq", scope="shared.l0c")
    cp = tirx.decl_buffer((pm, pn), "float32", name="cp", scope="shared.l0c")
    sources = [tirx.decl_buffer(buffer.shape, buffer.dtype, name=buffer.name + "_l1", scope="shared.l1") for buffer in (aq, bq, ap, bp)]
    score = tirx.decl_buffer((3, 2, 3, qm, qn), "float32", name="Score")
    result = tirx.decl_buffer((3, 2, 3, pm, pn), "float32", name="Result")
    epoch = tirx.Var("epoch", "int32")
    phases = []
    for prefix, a, b, c, sa, sb, out, transpose in (("qk", aq, bq, cq, *sources[:2], score, True),
                                                  ("pv", ap, bp, cp, *sources[2:], result, False)):
        owner, tile = [tirx.Var(prefix + suffix, "int32") for suffix in ("_owner", "_result")]
        k = tirx.Var(prefix + "_k", "int32")
        clear = not (kind == "read-first" and prefix == "pv")
        if accumulation is not None:
            clear = k == 0
            if prefix == "pv":
                if accumulation == "missing-clear":
                    clear = False
                elif accumulation == "late-clear":
                    clear = k == 1
                elif accumulation == "result-dependent-clear":
                    clear = tirx.And(k == 0, tile == 0)
                elif accumulation == "scalar-c-clear":
                    clear = tirx.Or(k == 0, c[0, 0] > 0)
        if direct_definition is not None and prefix == "pv":
            if direct_definition == "result-clear":
                clear = tile == 0
            elif direct_definition == "owner-clear":
                clear = owner == 0
            elif direct_definition == "epoch-clear":
                clear = epoch == 0
        mad = T.gemm(a, b, c, transpose_B=transpose, clear_accum=clear)
        if direct_definition == "scalar-c-annotation" and prefix == "pv":
            mad = tirx.Call(mad.dtype, mad.op, mad.args,
                            annotations={**dict(mad.annotations), "test_scalar_effect": c[0, 0]})
        if (accumulation in ("serialized-n-ineligible", "partial-write") or
                direct_definition == "partial-write") and prefix == "pv":
            arguments = list(mad.args)
            arguments[6] = tirx.IntImm("int32", pn // 2)
            if accumulation == "partial-write" or direct_definition == "partial-write":
                # Native coverage follows participating regions. The full
                # FIX below reads columns this first clear does not define.
                arguments[1] = tile_region(b, 1, shape=[32, pn // 2])
                arguments[2] = tile_region(c, 3, shape=[pm, pn // 2])
            # serialized-n-ineligible only protects this helper's narrow
            # eligibility; it makes no claim that serialized N determines
            # native coverage when the participating regions are unchanged.
            mad = tirx.Call(mad.dtype, mad.op, arguments, annotations=mad.annotations)
        native = unit(tirx.Evaluate(mad), core=2, cost=(2, 1),
                      guard=k > 0 if accumulation == "skipped-first" and prefix == "pv" else None)
        if accumulation == "interleaved-c" and prefix == "pv":
            native = seq(
                unit(tirx.Evaluate(T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)), core=2, cost=(2, 1)),
                unit(copy(cq, score, dst_indices=[epoch, owner, tile, 0, 0], dst_shape=[1, 1, 1, qm, qn]), core=2, cost=(1, 1)),
                native,
            )
        if accumulation is not None:
            native = unit(tirx.For(k, 0, 0 if accumulation == "zero-k" and prefix == "pv" else 3,
                                   tirx.ForKind.SERIAL, native), core=None)
        export = unit(copy(c, out, src_shape=[pm, pn // 2] if kind == "partial-export" and prefix == "pv" else None,
                           dst_indices=[epoch, owner, tile, 0, 0],
                           dst_shape=[1, 1, 1, pm, pn // 2] if kind == "partial-export" and prefix == "pv" else [1, 1, 1, *c.shape]), core=2, cost=(1, 1))
        annotations = {"multi_buffer_eligible": [cp.data]} if kind == "fixed-two" and prefix == "pv" else {}
        results = unit(tirx.For(tile, 0, 3, tirx.ForKind.SERIAL, seq(native, export), annotations=annotations), core=None)
        loads = [unit(copy(sa, a), core=2, cost=(1, 1)), unit(copy(sb, b), core=2, cost=(1, 1))]
        if accumulation == "interleaved-c":
            if prefix == "qk":
                continue
            loads.extend([unit(copy(sources[0], aq), core=2, cost=(1, 1)),
                          unit(copy(sources[1], bq), core=2, cost=(1, 1))])
        phases.append(unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, seq(*loads, results)),
                           core=None, guard=epoch < 2 if guarded_phases else None))
    epoch_annotations = {}
    if accumulation is not None or direct_definition is not None:
        # Conflict hints are consumed by their own lexical clock. Keep the
        # phase handoff in epoch as well as the independent root boundary.
        epoch_annotations["conflict_hint"] = [[cq.data, cp.data, -1, tirx.const(True, "bool")]]
    body = unit(tirx.For(epoch, 0, 3, tirx.ForKind.SERIAL, seq(*phases),
                         annotations=epoch_annotations), core=None)
    if accumulation is not None:
        # Keep this focused lifetime regression on ordinary whole-command
        # completion. The full DSL cases independently check unit streams.
        body = seq(unit(tirx.Evaluate(T.set_mmad_direction("m")), core=2, cost=(1, 1)), body)
    if kind == "outside-reader":
        body = seq(body, unit(copy(cq, score, dst_indices=[0, 0, 0, 0, 0], dst_shape=[1, 1, 1, qm, qn]), core=2, cost=(1, 1)))
    buffers = [*sources, aq, bq, ap, bp, cq, cp]
    layouts = {aq: make_ascend_l0a_2201_layout(aq), ap: make_ascend_l0a_2201_layout(ap),
               bq: make_ascend_major_k_layout(bq), bp: make_ascend_major_mn_layout(bp),
               cq: make_ascend_l0c_layout(cq), cp: make_ascend_l0c_layout(cp)}
    versions = {cq.data: 1, cp.data: 2 if kind == "fixed-two" else 1}
    if accumulation is not None or direct_definition is not None:
        # Exercise the late lifetime certificate independently of early C
        # phase admission. This small pair fits without a max-budget choice.
        # A fixed single C has no automatic owner/version plan: declaring it
        # eligible just to prepare count1 would also forbid its unit stream.
        versions = {cp.data: 2} if kind == "fixed-two" else {}
    if kind == "third-allocation":
        third = tirx.decl_buffer((16, 16), "float32", name="third", scope="shared.l0c")
        buffers.append(third)
        layouts[third] = make_ascend_l0c_layout(third)
        versions[third.data] = 1
    annotations = {"layout_map": layouts, "tl.buffer_versions_map": versions}
    if accumulation is not None or direct_definition is not None:
        # An existing physical conflict hint supplies ordinary completion
        # between these potential byte aliases. The lifetime certificate must
        # still establish a closed value; no completion edge is fabricated.
        annotations["tl.root_conflict_hints"] = [[cq.data, cp.data, -1, tirx.const(True, "bool")]]
    before = kernel(body, buffers=buffers, params=[score, result], annotations=annotations)
    before["main"] = before["main"].with_attr("target", tvm.target.Target({**_TARGET, "arch": arch}))
    return before, cq, cp


@pytest.mark.parametrize("kind,arch", [("ordinary", "dav-2201"), ("partial-export", "dav-2201"),
                                      ("read-first", "dav-2201"), ("outside-reader", "dav-2201"),
                                      ("fixed-two", "dav-2201"), ("third-allocation", "dav-2201"),
                                      ("reuse-disabled", "dav-2201"), ("ordinary", "dav-3510")])
def test_unproved_c_phases_keep_independent_storage_and_conservative_charge(kind, arch):
    before, cq, cp = _admission_fixture(kind, arch)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": kind == "reuse-disabled"}):
        after = transform.AutoSchedule()(before)
    selected = frozenset((cq.data, cp.data)) in _pairs(after)
    assert selected == (kind == "ordinary" and arch == "dav-2201")
    for c in (cq, cp):
        assert allocated_buffer(after, c.name).same_as(c)
        tvm.ir.assert_structural_equal(_root(after).annotations["layout_map"][c], make_ascend_l0c_layout(c))
    physical = [buffer for buffer in _root(after).alloc_buffers if buffer.scope() == "shared.l0c"]
    assert len(physical) == (3 if kind == "third-allocation" else 2)
    versions = _root(after).annotations.get("tl.buffer_versions_map", {})
    for buffer in physical:
        if buffer.data not in versions:
            assert all(buffer.data not in loop.annotations.get("multi_buffer_eligible", []) for loop in nodes(after, tirx.For))
    charge = sum(_bytes(buffer) * (int(versions[buffer.data]) if buffer.data in versions else 1) for buffer in physical)
    assert charge == 10 * 1024 + (2 * 1024 if kind == "fixed-two" else 1024 if kind == "third-allocation" else 0)
    assert charge <= 128 * 1024


@pytest.mark.parametrize("kind", ["reuse-disabled", "partial-export", "third-allocation"])
def test_unselected_full_bank_c_phases_keep_the_capacity_diagnostic(kind):
    admitted, cq, cp = _admission_fixture(pressure=True)
    assert _bytes(cq) == 128 * 1024 and _bytes(cp) == 16 * 1024
    assert frozenset((cq.data, cp.data)) in _pairs(transform.AutoSchedule()(admitted))
    rejected, _, _ = _admission_fixture(kind, pressure=True)
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": kind == "reuse-disabled"}):
        with pytest.raises(tvm.error.InternalError, match=r"shared.l0c.*capacity|capacity.*shared.l0c"):
            transform.AutoSchedule()(rejected)


def test_forged_c_phase_choice_cannot_bypass_completion_proven_alias():
    before, cq, cp = _admission_fixture("outside-reader")
    scheduled = transform.AutoSchedule()(before)
    assert frozenset((cq.data, cp.data)) not in _pairs(scheduled)
    root = _root(scheduled)
    marked = kernel(root.body, buffers=root.alloc_buffers, params=list(scheduled["main"].buffer_map.values()),
                    annotations={**dict(root.annotations), _MARKER: [[cq.data, cp.data]]})
    marked["main"] = marked["main"].with_attr(dict(scheduled["main"].attrs))
    prepared = transform.PrepareMultiBuffer()(transform.AssignCore()(marked))
    resolved = transform.ResolveCore()(prepared)
    with pytest.raises(tvm.error.InternalError, match="Selected L0 phase reuse lacks a completion-proven physical alias"):
        transform.InsertSync()(resolved)


@pytest.mark.parametrize("kind,accumulation", [("ordinary", "first-clear"),
                                              ("ordinary", "missing-clear"),
                                              ("ordinary", "late-clear"),
                                              ("ordinary", "result-dependent-clear"),
                                              ("ordinary", "skipped-first"),
                                              ("ordinary", "scalar-c-clear"),
                                              ("ordinary", "serialized-n-ineligible"),
                                              ("ordinary", "partial-write"),
                                              ("ordinary", "zero-k"),
                                              ("ordinary", "interleaved-c"),
                                              ("outside-reader", "first-clear"),
                                              ("fixed-two", "first-clear")])
def test_only_closed_first_clear_accumulations_share_c_bytes(kind, accumulation):
    before, cq, cp = _admission_fixture(kind, accumulation=accumulation)
    expected = kind == "ordinary" and accumulation == "first-clear"
    assert frozenset((cq.data, cp.data)) not in _pairs(before)
    # This is a final scheduled native input, independent of the early choice
    # and scheduler. The full DSL tests above exercise their complete pipeline.
    resolved = transform.ResolveCore()(transform.PrepareMultiBuffer()(transform.AssignCore()(before)))
    inserted = transform.InsertSync()(resolved)
    aliases = _root(inserted).annotations["tl.buffer_alias_map"]
    assert (cp.data in aliases.get(cq.data, []) and cq.data in aliases.get(cp.data, [])) == expected
    if expected:
        _assert_closed_accumulation_completion(transform.LowerScheduledTIR()(inserted))


@pytest.mark.parametrize("definition", ["full-clear", "result-clear", "owner-clear", "epoch-clear",
                                        "scalar-c-annotation", "partial-write"])
def test_direct_full_clear_lifetimes_keep_native_units_and_both_retirement_ends(definition):
    # There is no K subtree between either direct MAD and its result loop.
    # Keep unit streams enabled: unit readiness cannot supply the whole-MAD
    # completion used by the ordinary generation builder.
    before, cq, cp = _admission_fixture(direct_definition=definition, guarded_phases=True)
    original = tvm.ir.save_json(before)
    resolved = transform.ResolveCore()(transform.PrepareMultiBuffer()(transform.AssignCore()(before)))
    inserted = transform.InsertSync()(resolved)
    assert tvm.ir.save_json(before) == original
    aliases = _root(inserted).annotations["tl.buffer_alias_map"]
    expected = definition == "full-clear"
    assert (cp.data in aliases.get(cq.data, []) and cq.data in aliases.get(cp.data, [])) == expected
    if not expected:
        return

    assert Counter(call.args[0].value for call in calls(inserted, "tl.ascend_set_mmad_direction")) == {"n": 1, "m": 1}
    records = list(_trace(transform.LowerScheduledTIR()(inserted), core=2, lane=0))
    trace = []

    def blocks(arg, env):
        buffer, origin, extent = _region(arg, env)
        assert origin == (0, 0) and extent == tuple(map(int, buffer.shape))
        return tuple((buffer.data, row, column) for row in range(extent[0] // 16)
                     for column in range(extent[1] // 16))

    for index, (call, env, _) in enumerate(records):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            flag = _value(call.args[1], env)
            assert 0 <= flag < 8
            setting = name == "tl.ascend_set_flag"
            trace.append(_Command(source if setting else destination, "set" if setting else "wait", (event, flag)))
        elif name == _GEMM:
            # A/B dependencies remain ordinary whole completion. Native C
            # unit readiness is deliberately absent from this graph.
            trace.append(_Command("M", "mad", reads=(*blocks(call.args[0], env), *blocks(call.args[1], env))))
        elif name == _COPY:
            src = _region(call.args[0], env)[0]
            trace.append(_Command("FIX", "export") if src.scope() == "shared.l0c" else
                         _Command("MTE1", "copy", reads=blocks(call.args[0], env), writes=blocks(call.args[1], env)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        else:
            assert name == "tl.ascend_set_mmad_direction", name
            trace.append(_Command("S", "scalar"))
    require = _prove_completions(trace)
    activations = _unit_stream_activations(records, {"qk_result", "pv_result"})
    _assert_kernel_unit_boundaries(records, activations)
    seen, previous_work, previous_storage = Counter(), [], None
    for item in activations:
        mads, fixes = item["mads"], item["exports"]
        assert len(mads) == len(fixes) == 3
        (storage,) = {_region(call.args[2], env)[0].name for _, call, env, _ in mads}
        seen[storage] += 1
        assert all(_value(call.args[9], env) == 1 and _value(call.annotations["unit_flag_ctrl"], env) == 3
                   for _, call, env, _ in mads)
        assert all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for _, call, env, _ in fixes)
        if previous_storage is not None and previous_storage != storage:
            # Alias handoff must retire both the old whole MADs and their
            # asynchronous FIX readers. No unit-internal C edge is assumed.
            for old in previous_work:
                require(3 * old + 2, 3 * mads[0][0] + 1)
        previous_work = [index for index, *_ in (*mads, *fixes)]
        previous_storage = storage
    assert seen == {"cq": 4, "cp": 4}


def _assert_closed_accumulation_completion(lowered, active_epochs=3):
    records = list(_trace(lowered, core=2, lane=0))
    trace, work = [], []
    for call, env, _ in records:
        name, index = _name(call), len(trace)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            trace.append(_Command(source if setting else destination, "set" if setting else "wait",
                                  (event, _value(call.args[1], env))))
        elif name == _GEMM:
            c = _region(call.args[2], env)[0]
            trace.append(_Command("M", "mad"))
            work.append((index, "mad", c.name, _value(call.args[9], env)))
        elif name == _COPY:
            src = _region(call.args[0], env)[0]
            pipe = "FIX" if src.scope() == "shared.l0c" else "MTE1"
            trace.append(_Command(pipe, "export" if pipe == "FIX" else "copy"))
            if pipe == "FIX":
                work.append((index, "export", src.name, None))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        else:
            assert name == "tl.ascend_set_mmad_direction", name
            trace.append(_Command("S", "scalar"))
    require = _prove_completions(trace)
    previous_export, pending = None, []
    for index, role, storage, clear in work:
        if role == "mad":
            if clear:
                assert not pending
                if previous_export is not None:
                    require(3 * previous_export + 2, 3 * index + 1)
            else:
                assert pending
            pending.append(index)
        else:
            assert len(pending) == 3
            for producer in pending:
                require(3 * producer + 2, 3 * index + 1)
            pending.clear()
            previous_export = index
    assert not pending
    assert Counter((role, storage) for _, role, storage, _ in work) == {
        ("mad", "cq"): active_epochs * 18, ("mad", "cp"): active_epochs * 18,
        ("export", "cq"): active_epochs * 6, ("export", "cp"): active_epochs * 6,
    }
    return records


def test_guarded_c_phase_conflicts_retire_fix_at_lexical_boundaries():
    # Guarded task endpoints and unconditional phase boundaries are distinct
    # physical cuts. An active-only FIX_M handshake cannot certify the latter.
    before, cq, cp = _admission_fixture(accumulation="first-clear", guarded_phases=True)
    resolved = transform.ResolveCore()(transform.PrepareMultiBuffer()(transform.AssignCore()(before)))
    inserted = transform.InsertSync()(resolved)
    aliases = _root(inserted).annotations["tl.buffer_alias_map"]
    assert cp.data in aliases[cq.data] and cq.data in aliases[cp.data]
    records = _assert_closed_accumulation_completion(transform.LowerScheduledTIR()(inserted), active_epochs=2)
    boundary_operations = Counter()
    notifications, acquisitions = Counter(), Counter()
    for call, env, loops in records:
        name = _name(call)
        if name not in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            continue
        flag = (call.args[0].value, _value(call.args[1], env))
        assert 0 <= flag[1] < 8
        (notifications if name == "tl.ascend_set_flag" else acquisitions)[flag] += 1
        coordinates = {var.name: value for var, value, _ in loops}
        assert len(coordinates) == len(loops)
        if flag[0] == "FIX_M" and set(coordinates) == {"epoch"}:
            boundary_operations[name, coordinates["epoch"]] += 1
    assert notifications == acquisitions
    # The inactive final epoch still consumes/publishes the ordinary boundary
    # protocol. The root seed and final drain are balanced by the graph oracle.
    for name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
        assert all(boundary_operations[name, epoch] > 0 for epoch in range(3))
