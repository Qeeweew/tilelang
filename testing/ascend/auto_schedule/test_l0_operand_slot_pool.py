"""Physical operand credits survive independent phase clocks and inactive epochs.

These CPU fixtures use the fixed Q cache, counter-based QK B ring, and shifted
PV rings together. Completion proofs use real binary events and per-pipe FIFO;
unit readiness and cross-pipe source order supply no completion edges.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T
from tvm import tirx

from testing.ascend._ir import allocated_buffer, nodes
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower, _root
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _prove_binary_protocol


def _program(dtype="float16", unit_streams=True, extra_reader=False, active_epochs=3,
             logical_stages=False, pv_native_k=256, pv_b_versions=None):
    # The unsupported-reader case must also fit without a C phase alias.
    q_rows = 64 if extra_reader or logical_stages else 128

    @T.prim_func
    def main(Q: T.Tensor((3, 2, q_rows, 64), dtype),
             K: T.Tensor((3, 1024, 64), dtype),
             P: T.Tensor((3, 2, 128, 1024), dtype),
             V: T.Tensor((3, 1024, 64), dtype),
             Score: T.Tensor((3, 2, q_rows, 1024), "float32"),
             Result: T.Tensor((3, 2, 128, 64), "float32")):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((q_rows, 64), dtype)
            k_l1 = T.alloc_l1((1024, 64), dtype)
            p_l1 = T.alloc_l1((64, 1024), dtype)
            v_l1 = T.alloc_l1((1024, 64), dtype)
            aq = T.alloc_l0a((q_rows, 64), dtype)
            bq = T.alloc_l0b((256, 64), dtype)
            ap = T.alloc_l0a((64, pv_native_k), dtype)
            bp = T.alloc_l0b((pv_native_k, 64), dtype)
            cq = T.alloc_l0c((q_rows, 256), "float32")
            cp = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({q_l1: 1, k_l1: 1, p_l1: 1, v_l1: 1,
                                        aq: 1, bq: 2, cq: 1, cp: 1})
            if pv_b_versions is not None:
                T.annotate_buffer_versions({bp: pv_b_versions})
            if not unit_streams:
                T.set_mmad_direction("m")
            for epoch in T.serial(active_epochs if logical_stages else active_epochs + 2,
                                  annotations={"enable_offset": logical_stages,
                                               "tl.ascend_preserve_issue_order": logical_stages}):
                with T.Stage(0):
                    if epoch < active_epochs:
                        T.copy(K[epoch, :, :], k_l1)
                        for qk_owner in T.serial(2):
                            T.copy(Q[epoch, qk_owner, :, :], q_l1)
                            T.copy(q_l1, aq)
                            for qk_n in T.serial(4):
                                T.copy(k_l1[qk_n * 256:(qk_n + 1) * 256, :], bq)
                                T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
                                if extra_reader:
                                    T.gemm(aq, bq, cq, transpose_B=True, clear_accum=True)
                                T.copy(cq, Score[epoch, qk_owner, :, qk_n * 256:(qk_n + 1) * 256])
                with T.Stage(1 if logical_stages else 0):
                    if (0 if logical_stages else 1) <= epoch and epoch < active_epochs + (0 if logical_stages else 1):
                        T.copy(V[epoch - (0 if logical_stages else 1), :, :], v_l1)
                        for pv_owner in T.serial(2):
                            for m_part in T.serial(2):
                                T.copy(P[epoch - (0 if logical_stages else 1), pv_owner, m_part * 64:(m_part + 1) * 64, :], p_l1)
                                for pv_k in T.Pipelined(1024 // pv_native_k, max_stages=2):
                                    with T.Task(latency=2, ii=1):
                                        T.copy(p_l1[:, pv_k * pv_native_k:(pv_k + 1) * pv_native_k], ap)
                                    with T.Task(latency=2, ii=1):
                                        T.copy(v_l1[pv_k * pv_native_k:(pv_k + 1) * pv_native_k, :], bp)
                                    with T.Task(latency=3, ii=1):
                                        T.gemm(ap, bp, cp, transpose_B=False, clear_accum=pv_k == 0)
                                T.copy(cp, Result[epoch - (0 if logical_stages else 1), pv_owner, m_part * 64:(m_part + 1) * 64, :])
    return main


@dataclass(frozen=True)
class _Command:
    pipe: str
    kind: str
    flag: tuple | None = None
    identity: tuple | None = None
    bank: str | None = None
    blocks: tuple = ()


def _physical(argument, env):
    buffer, origin, _ = _region(argument, env)
    assert buffer.scope() in ("shared.l0a", "shared.l0b")
    slot = origin[0] if len(origin) == 3 else 0
    versions = int(buffer.shape[0]) if len(origin) == 3 else 1
    assert origin[-2:] == (0, 0) and 0 <= slot < versions
    plane = math.prod(map(int, buffer.shape[-2:]))
    return buffer.scope(), tuple(range(slot * plane // 256, (slot + 1) * plane // 256))


def _check_physical_completions(mod, active, *, extra_reader=False, pv_native_k=256,
                                check_prefetch=True):
    trace, payload, readers, loaded = [], {}, defaultdict(list), {}
    native_counts, exports, phase_starts, phase_ends = Counter(), Counter(), {}, {}
    pending_reads = []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        coords = {var.name: value for var, value, _ in loops}
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            setting = name == "tl.ascend_set_flag"
            event_id = _value(call.args[1], env)
            assert 0 <= event_id < 8, "Local event exceeds the 2201 bank"
            trace.append(_Command(source if setting else destination, "set" if setting else "wait",
                                  (event, event_id)))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            src, source_min, _ = _region(call.args[0], env)
            dst, destination_min, _ = _region(call.args[1], env)
            if dst.scope() in ("shared.l0a", "shared.l0b"):
                phase = "qk" if dst.name in ("aq", "bq") else "pv"
                epoch, owner = coords["epoch"], coords[phase + "_owner"]
                part = coords.get("m_part", 0)
                tile = (None if dst.name == "aq" else source_min[-2] // 256) if phase == "qk" else (
                    source_min[-1 if dst.name == "ap" else -2] // pv_native_k)
                bank, blocks = _physical(call.args[1], env)
                identity = phase, epoch, owner, part, tile
                index = len(trace)
                for block in blocks:
                    key = bank, block
                    pending_reads.extend((reader, index) for reader in readers[key])
                    readers[key] = []
                    payload[key] = identity
                    loaded[key] = index
                trace.append(_Command("MTE1", "load", identity=identity, bank=bank, blocks=blocks))
                phase_starts.setdefault((phase, epoch), index)
            elif src.scope() == "shared.l0c":
                phase = "qk" if dst.name == "Score" else "pv"
                epoch = coords["epoch"]
                trace.append(_Command("FIX", "export", identity=(phase, epoch)))
                phase_ends[phase, epoch] = len(trace) - 1
                exports[phase] += 1
            else:
                assert src.scope() == "global" and dst.scope() == "shared.l1"
                trace.append(_Command("MTE2", "input"))
        elif name == "tl.tileop.gemm":
            a = _region(call.args[0], env)[0]
            phase = "qk" if a.name == "aq" else "pv"
            epoch, owner = coords["epoch"], coords[phase + "_owner"]
            part = coords.get("m_part", 0)
            key = phase, epoch, owner, part
            tile = native_counts[key] // (2 if phase == "qk" and extra_reader else 1)
            assert _value(call.args[9], env) == (1 if phase == "qk" else int(tile == 0))
            index = len(trace)
            for operand in call.args[:2]:
                bank, blocks = _physical(operand, env)
                fixed = phase == "qk" and bank == "shared.l0a"
                expected = phase, epoch, owner, part, None if fixed else tile
                for block in blocks:
                    physical = bank, block
                    assert payload.get(physical) == expected, (physical, payload.get(physical), expected)
                    pending_reads.append((loaded[physical], index))
                    readers[physical].append(index)
            trace.append(_Command("M", "mad", identity=(*key, tile)))
            native_counts[key] += 1
        else:
            assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_mmad_direction"), name
            trace.append(_Command("S", "configuration"))
    precedes = _prove_binary_protocol(trace)
    assert all(precedes(source, destination) for source, destination in pending_reads), (
        "Every load needs completion before MAD; every overwritten byte needs its last MAD completion")
    assert sum(native_counts.values()) == active * ((16 if extra_reader else 8) + 4 * (1024 // pv_native_k))
    assert exports == (Counter({"qk": active * 8, "pv": active * 4}) if active else Counter())
    if active >= 3 and check_prefetch:
        for epoch in range(2, active):
            first = phase_starts["qk", epoch]
            last_mad = max(index for index, item in enumerate(trace)
                           if item.kind == "mad" and item.identity[:2] == ("pv", epoch - 1))
            if extra_reader:
                assert precedes(last_mad, first), (
                    "An unqualified reader set must retain the conservative operand handoff")
            else:
                assert not precedes(last_mad, first), "Slot0 prefetch still inherits old slot1 MAD retirement"
                assert not precedes(phase_ends["pv", epoch - 1], first), "Slot0 prefetch still waits for old FIX"


@pytest.mark.parametrize("dtype,unit_streams", [("float16", True), ("bfloat16", False)])
@pytest.mark.parametrize("active", [1, 3], ids=["fill-drain", "wrap-and-inactive-epochs"])
def test_physical_pool_covers_fixed_counter_and_shifted_iteration_owners(dtype, unit_streams, active):
    artifact, snapshots = _lower(_program(dtype, unit_streams, active_epochs=active))
    prepared = snapshots["tl.PrepareMultiBuffer"]
    buffers = {name: allocated_buffer(prepared, name) for name in ("aq", "bq", "ap", "bp")}
    assert int(_root(prepared).annotations["tl.buffer_versions_map"][buffers["bq"].data]) == 2
    qk_counters = [loop.annotations.get("tl.multi_buffer_counter_map", {})
                   for loop in nodes(prepared, tirx.For) if loop.loop_var.name == "qk_n"]
    assert len(qk_counters) == 1 and buffers["bq"].data in qk_counters[0]
    versions = _root(prepared).annotations["tl.buffer_versions_map"]
    assert all(int(versions[buffers[name].data]) == 2 for name in ("ap", "bp"))
    synced = _root(snapshots["tl.InsertSync"])
    aliases = synced.annotations["tl.buffer_alias_map"]
    for first, second in (("aq", "ap"), ("bq", "bp")):
        assert buffers[second].data in aliases[buffers[first].data]
        assert buffers[first].data in aliases[buffers[second].data]
    _check_physical_completions(snapshots["tl.LowerScheduledTIR"], active)
    assert artifact.kernel_source


def test_additional_operand_reader_keeps_conservative_completion_protocol():
    _, snapshots = _lower(_program(unit_streams=False, extra_reader=True))
    _check_physical_completions(snapshots["tl.LowerScheduledTIR"], 3, extra_reader=True)


@pytest.mark.parametrize("active", [1, 3])
@pytest.mark.parametrize("pv_native_k", [128, 256], ids=["overlapping-pitches", "equal-pitches"])
def test_logical_phase_offsets_preserve_physical_operand_credits(active, pv_native_k):
    _, snapshots = _lower(_program(logical_stages=True, active_epochs=active, pv_native_k=pv_native_k))
    _check_physical_completions(snapshots["tl.LowerScheduledTIR"], active, pv_native_k=pv_native_k)


def test_automatic_phases_share_the_static_epoch_and_physical_credits():
    from tvm import tirx
    program = _program(logical_stages=True, active_epochs=3, pv_native_k=128)

    def post(node):
        if isinstance(node, tirx.For) and node.loop_var.name == "epoch":
            annotations = dict(node.annotations)
            annotations.pop("tl.ascend_preserve_issue_order")
            annotations.pop("enable_offset")  # Ordinary Pipelined's default.
            annotations.update({"tl.ascend_auto_pipeline": True, "num_stages": 4})
            return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                            annotations=annotations, step=node.step)
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_stage":
            return node.body

    program = program.with_body(tirx.stmt_functor.ir_transform(program.body, None, post))
    _, snapshots = _lower(program)
    # Automatic selection may choose another relative phase stage. Check its
    # actual payloads and all readers; the manual tests above additionally
    # require the specific next-slot prefetch allowed by their chosen stages.
    _check_physical_completions(snapshots["tl.LowerScheduledTIR"], 3,
                                pv_native_k=128, check_prefetch=False)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_three_version_b_ring_preserves_shifted_byte_credits(dtype):
    # A two-slot 32 KiB QK ring intersects a three-slot 16 KiB PV ring.
    # Three active epochs exercise both wraps and the final shifted reader.
    _, snapshots = _lower(_program(dtype=dtype, logical_stages=True,
                                   active_epochs=3, pv_native_k=128,
                                   pv_b_versions=3))
    prepared = snapshots["tl.PrepareMultiBuffer"]
    bp = allocated_buffer(prepared, "bp")
    assert int(_root(prepared).annotations["tl.buffer_versions_map"][bp.data]) == 3
    _check_physical_completions(snapshots["tl.LowerScheduledTIR"], 3,
                                pv_native_k=128)


def test_slot_pool_ring_keeps_physical_capacity_gate():
    # Five 16 KiB slots cannot fit in L0B even when phases share a base.
    with pytest.raises(tvm.error.InternalError,
                       match="L0 phase reuse exceeds scope shared.l0b capacity"):
        _lower(_program(logical_stages=True, active_epochs=3,
                        pv_native_k=128, pv_b_versions=5))
