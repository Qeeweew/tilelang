"""Independent native operand rings share L0 banks across closed phases.

Task costs below are unitless test weights, chosen to make two versions useful
to the real scheduler. They are not measured hardware latencies. Completion
checks use actual local flags and pipe ordering; no source-order edge between
MTE1 and MAD is invented, and no device is launched by this file.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout, make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import allocated_buffer, calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _prove_binary_protocol


_PAIRS = "tl.ascend_l0_phase_reuse_pairs"
_TARGET = {"kind": "ascend", "arch": "dav-2201"}
_COPY = "tl.tileop.ascend_copy"
_LOAD3D = "tl.tileop.ascend_cube_load3d"
_GEMM = "tl.tileop.gemm"


def _root(mod):
    return next(block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root")


def _bytes(buffer):
    dtype = tvm.DataType(buffer.dtype)
    return math.prod(map(int, buffer.shape)) * dtype.bits * dtype.lanes // 8


def _pairs(mod):
    return {frozenset(pair) for pair in _root(mod).annotations.get(_PAIRS, [])}


def _program(dtype="float16", guarded=False, cached_qk=False, unit_streams=False,
             pipelined_p_cache=False):
    """Independent 3/4-step K rings, or one fixed Q cache and an automatic PV."""
    qk_k_extent, qk_n_extent = (64, 1024) if cached_qk else (192, 256)

    def fixed_versions(q, k, p, v, c, aq, bq):
        result = {q: 1, k: 1, p: 1, v: 1, c: 1}
        if pipelined_p_cache:
            del result[p]
        if cached_qk:
            result.update({aq: 1, bq: 1})
        return result

    @T.prim_func
    def main(Q: T.Tensor((3, 2, 128, qk_k_extent), dtype),
             K: T.Tensor((3, 2, qk_n_extent, qk_k_extent), dtype),
             P: T.Tensor((3, 2, 128, 512), dtype),
             V: T.Tensor((3, 2, 512, 64), dtype),
             Score: T.Tensor((3, 2, 128, qk_n_extent), "float32"),
             Result: T.Tensor((3, 2, 128, 64), "float32")):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((128, qk_k_extent), dtype)
            k_l1 = T.alloc_l1((qk_n_extent, qk_k_extent), dtype)
            p_l1 = T.alloc_l1((128, 256 if pipelined_p_cache else 512), dtype)
            v_l1 = T.alloc_l1((512, 64), dtype)
            aq = T.alloc_l0a((128, 64), dtype)
            bq = T.alloc_l0b((256, 64), dtype)
            if not pipelined_p_cache:
                ap = T.alloc_l0a((128, 128), dtype)
                bp = T.alloc_l0b((128, 64), dtype)
            accum = T.alloc_l0c((128, 256), "float32")
            T.annotate_buffer_versions(fixed_versions(q_l1, k_l1, p_l1, v_l1, accum, aq, bq))
            # This regression isolates A/B phase reuse and whole-pipe C
            # completion. Automatic unit streams have their own regressions;
            # an explicit direction keeps that independent protocol unselected.
            if not unit_streams:
                T.set_mmad_direction("m")
            for epoch in T.serial(3):
                with T.Stage(0):
                    if epoch < (2 if guarded else 3):
                        for qk_owner in T.serial(2):
                            T.copy(Q[epoch, qk_owner, :, :], q_l1)
                            T.copy(K[epoch, qk_owner, :, :], k_l1)
                            if cached_qk:
                                with T.Task(latency=2, ii=1):
                                    T.copy(q_l1, aq)
                                for qk_n in T.serial(4):
                                    with T.Task(latency=2, ii=1):
                                        T.copy(k_l1[qk_n * 256:(qk_n + 1) * 256, :], bq)
                                    with T.Task(latency=3, ii=1):
                                        T.gemm(aq, bq, accum, transpose_B=True, clear_accum=True)
                                    T.copy(accum, Score[epoch, qk_owner, :, qk_n * 256:(qk_n + 1) * 256])
                            else:
                                for qk_k in T.Pipelined(3, max_stages=2):
                                    with T.Task(latency=2, ii=1):
                                        T.copy(q_l1[:, qk_k * 64:(qk_k + 1) * 64], aq)
                                    with T.Task(latency=2, ii=1):
                                        T.copy(k_l1[:, qk_k * 64:(qk_k + 1) * 64], bq)
                                    with T.Task(latency=3, ii=1):
                                        T.gemm(aq, bq, accum, transpose_B=True, clear_accum=qk_k == 0)
                                T.copy(accum, Score[epoch, qk_owner, :, :])
                with T.Stage(0):
                    if epoch < (2 if guarded else 3):
                        for pv_owner in T.serial(2):
                            if pipelined_p_cache:
                                T.copy(V[epoch, pv_owner, :, :], v_l1)
                                # Ordinary L1 GEMM generates two native K128
                                # steps. Both this P ring and that native ring
                                # remain choices of the automatic scheduler.
                                for pv_chunk in T.Pipelined(2, max_stages=4):
                                    T.copy(P[epoch, pv_owner, :, pv_chunk * 256:(pv_chunk + 1) * 256], p_l1)
                                    T.gemm(p_l1, v_l1[pv_chunk * 256:(pv_chunk + 1) * 256, :],
                                           accum[:, :64], transpose_B=False,
                                           clear_accum=pv_chunk == 0)
                            else:
                                T.copy(P[epoch, pv_owner, :, :], p_l1)
                                T.copy(V[epoch, pv_owner, :, :], v_l1)
                                for pv_k in T.Pipelined(4, max_stages=2):
                                    with T.Task(latency=2, ii=1):
                                        T.copy(p_l1[:, pv_k * 128:(pv_k + 1) * 128], ap)
                                    with T.Task(latency=2, ii=1):
                                        T.copy(v_l1[pv_k * 128:(pv_k + 1) * 128, :], bp)
                                    with T.Task(latency=3, ii=1):
                                        T.gemm(ap, bp, accum[:, :64], transpose_B=False,
                                               clear_accum=pv_k == 0)
                            T.copy(accum[:, :64], Result[epoch, pv_owner, :, :])
    return main


def _lower(program):
    snapshots = {}
    wanted = {"tl.AnnotateMultiBufferEligible", "tl.AutoSchedule", "tl.PrepareMultiBuffer",
              "tl.ResolveCore", "tl.InsertSync", "tl.MaterializeMultiBuffer", "tl.LowerScheduledTIR"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(program.with_attr("global_symbol", "main"),
                                  target=_TARGET, enable_host_codegen=False,
                                  enable_device_compile=False)
    assert set(snapshots) == wanted and artifact.kernel_source
    return artifact, snapshots


def _assert_logical_rings(snapshots, cached_qk=False, pipelined_p_cache=False):
    before, scheduled = (snapshots[name] for name in
                         ("tl.AnnotateMultiBufferEligible", "tl.AutoSchedule"))
    root = _root(scheduled)
    names = {"aq": "aq", "bq": "bq",
             "ap": "gemm_l0a_0" if pipelined_p_cache else "ap",
             "bp": "gemm_l0b_0" if pipelined_p_cache else "bp"}
    operands = {role: allocated_buffer(scheduled, name) for role, name in names.items()}
    assert _pairs(scheduled) == {frozenset(operands[name].data for name in pair)
                                for pair in (("aq", "ap"), ("bq", "bp"))}
    assert len({buffer.data for buffer in operands.values()}) == 4
    versions = root.annotations["tl.buffer_versions_map"]
    assert all(int(versions[operands[name].data]) == 2 for name in ("ap", "bp"))
    expected_layout = {"aq": make_ascend_l0a_2201_layout, "ap": make_ascend_l0a_2201_layout,
                       "bq": make_ascend_major_k_layout, "bp": make_ascend_major_mn_layout}
    for name, buffer in operands.items():
        original = allocated_buffer(before, names[name])
        assert original.same_as(buffer)
        tvm.ir.assert_structural_equal(root.annotations["layout_map"][buffer],
                                       expected_layout[name](buffer))
        owners = [loop for loop in nodes(scheduled, tirx.For)
                  if buffer.data in loop.annotations.get("multi_buffer_eligible", [])]
        if cached_qk and name.endswith("q"):
            assert not owners and buffer.data not in versions
            assert int(_root(before).annotations["tl.buffer_versions_map"][buffer.data]) == 1
        else:
            assert int(versions[buffer.data]) == 2
            expected_owner = "qk_k" if name.endswith("q") else "gemm_sub_k_0" if pipelined_p_cache else "pv_k"
            assert len(owners) == 1 and owners[0].loop_var.name == expected_owner
            assert int(owners[0].extent) == (3 if name.endswith("q") else 2 if pipelined_p_cache else 4)
            assert all(not other.data in owners[0].annotations["multi_buffer_eligible"]
                       for other_name, other in operands.items() if other_name[-1] != name[-1])
    counts = {name: 1 if cached_qk and name.endswith("q") else 2 for name in operands}
    assert sum(_bytes(operands[name]) * counts[name] for name in ("aq", "ap")) == (80 if cached_qk else 96) * 1024
    assert sum(_bytes(operands[name]) * counts[name] for name in ("bq", "bp")) == (64 if cached_qk else 96) * 1024
    for pair in (("aq", "ap"), ("bq", "bp")):
        assert max(_bytes(operands[name]) * counts[name] for name in pair) == (
            32 if cached_qk and pair[0] == "bq" else 64) * 1024

    prepared = snapshots["tl.PrepareMultiBuffer"]
    assert _pairs(prepared) == _pairs(scheduled)
    # A/B from one native K loop may share a prepared counter. The two
    # phases must retain distinct clocks, whether lexical or counter-based.
    clocks = {}
    for name, buffer in operands.items():
        if cached_qk and name.endswith("q"):
            continue
        (owner,) = [loop for loop in nodes(prepared, tirx.For)
                   if buffer.data in loop.annotations.get("multi_buffer_eligible", [])]
        counter_map = owner.annotations.get("tl.multi_buffer_counter_map", {})
        clocks[name] = counter_map[buffer.data].data if buffer.data in counter_map else owner.loop_var
    assert clocks["ap"].same_as(clocks["bp"])
    if not cached_qk:
        assert clocks["aq"].same_as(clocks["bq"])
        assert not clocks["aq"].same_as(clocks["ap"])
    synced = snapshots["tl.InsertSync"]
    assert _PAIRS not in _root(synced).annotations
    aliases = _root(synced).annotations["tl.buffer_alias_map"]
    for first, second in (("aq", "ap"), ("bq", "bp")):
        a, b = operands[first].data, operands[second].data
        assert b in aliases[a] and a in aliases[b]
    physical = snapshots["tl.MaterializeMultiBuffer"]
    for name, buffer in operands.items():
        actual = allocated_buffer(physical, names[name])
        assert actual.data.same_as(buffer.data)
        assert tuple(map(int, actual.shape)) == (
            tuple(map(int, buffer.shape)) if counts[name] == 1 else (2, *tuple(map(int, buffer.shape))))
    return operands


@dataclass(frozen=True)
class _Completion:
    pipe: str
    kind: str
    flag: tuple | None = None
    phase: str | None = None
    epoch: int | None = None
    owner: int | None = None
    tile: int | None = None
    storage: str | None = None
    blocks: tuple = ()


def _completion_trace(mod, cached_qk=False):
    trace = []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        epoch = next((index for var, index, _ in loops if var.name == "epoch"), None)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source, destination = event.split("_")
            is_set = name == "tl.ascend_set_flag"
            trace.append(_Completion(source if is_set else destination, "set" if is_set else "wait",
                                     (event, _value(call.args[1], env)), epoch=epoch))
        elif name == "tl.ascend_pipe_barrier":
            pipe = call.args[0].value.removeprefix("PIPE_")
            assert pipe != "ALL", "This oracle requires explicit per-pipe completion gates"
            trace.append(_Completion(pipe, "barrier", epoch=epoch))
        elif name in (_COPY, _LOAD3D):
            src, source_min, _ = _region(call.args[0], env)
            dst, dest_min, _ = _region(call.args[1], env)
            if dst.scope() in ("shared.l0a", "shared.l0b"):
                phase = "qk" if dst.name.endswith("q") else "pv"
                fixed = cached_qk and phase == "qk"
                if fixed:
                    assert dest_min == (0, 0)
                    slot = 0
                else:
                    assert len(dest_min) == 3 and dest_min[0] in (0, 1) and dest_min[1:] == (0, 0)
                    slot = dest_min[0]
                owner = next(index for var, index, _ in loops if var.name == phase + "_owner")
                native_k = 64 if phase == "qk" else 128
                tile = (None if dst.name == "aq" else source_min[-2] // 256) if fixed else (
                    (source_min[-2] if dst.name == "bp" else source_min[-1]) // native_k)
                plane = math.prod(map(int, dst.shape[-2:]))
                start = slot * plane // 256
                trace.append(_Completion("MTE1", "load", phase=phase, epoch=epoch, owner=owner,
                                         tile=tile, storage=dst.scope(),
                                         blocks=tuple(range(start, start + plane // 256))))
            elif src.scope() == "shared.l0c":
                phase = "qk" if dst.name == "Score" else "pv"
                owner = dest_min[1]
                tile = dest_min[-1] // 256 if cached_qk and phase == "qk" else None
                trace.append(_Completion("FIX", "export", phase=phase, epoch=epoch, owner=owner, tile=tile))
            else:
                assert src.scope() == "global" and dst.scope() == "shared.l1"
                trace.append(_Completion("MTE2", "input", epoch=epoch))
        elif name == _GEMM:
            a, a_min, _ = _region(call.args[0], env)
            b, b_min, _ = _region(call.args[1], env)
            phase = "qk" if a.name == "aq" else "pv"
            assert b.name == ("bq" if phase == "qk" else "bp")
            owner = next(index for var, index, _ in loops if var.name == phase + "_owner")
            blocks = []
            for operand, origin in ((a, a_min), (b, b_min)):
                if cached_qk and phase == "qk":
                    assert origin == (0, 0)
                    slot = 0
                else:
                    assert len(origin) == 3 and origin[0] in (0, 1) and origin[1:] == (0, 0)
                    slot = origin[0]
                plane = math.prod(map(int, operand.shape[-2:]))
                start = slot * plane // 256
                blocks.append((operand.scope(), tuple(range(start, start + plane // 256))))
            # Capture native clear semantics along with actual physical slots.
            tile = (next(index for var, index, _ in loops if var.name == "qk_n")
                    if cached_qk and phase == "qk" else None)
            trace.append(_Completion("M", "mad", flag=(_value(call.args[9], env),),
                                     phase=phase, epoch=epoch, owner=owner, tile=tile, blocks=tuple(blocks)))
        else:
            assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_mmad_direction"), name
            trace.append(_Completion("S", "configuration", epoch=epoch))
    return trace


def _assert_completion_and_prefetch(mod, active_epochs, cached_qk=False):
    trace = _completion_trace(mod, cached_qk)
    precedes = _prove_binary_protocol(trace)
    writers, readers, chains, loads, mads, exports = {}, defaultdict(list), defaultdict(list), {}, {}, Counter()
    cross_phase_overwrites = Counter()
    for index, node in enumerate(trace):
        if node.kind == "load":
            key = (node.phase, node.epoch, node.owner, node.tile, node.storage)
            assert key not in loads
            loads[key] = index
            for block in node.blocks:
                physical = node.storage, block
                for previous in readers[physical]:
                    assert precedes(previous, index), "L0 overwrite precedes an actual MAD reader completion"
                    if trace[previous].phase != node.phase:
                        cross_phase_overwrites[node.storage] += 1
                readers[physical] = []
                writers[physical] = index
        elif node.kind == "mad":
            payloads = set()
            for storage, blocks in node.blocks:
                for block in blocks:
                    physical = storage, block
                    assert physical in writers, "MAD reads an uninitialized physical L0 block"
                    writer = writers[physical]
                    produced = trace[writer]
                    assert precedes(writer, index), "MAD has no L0 load completion"
                    assert (produced.phase, produced.epoch, produced.owner) == (node.phase, node.epoch, node.owner)
                    if cached_qk and node.phase == "qk" and storage == "shared.l0a":
                        assert produced.tile is None
                    else:
                        payloads.add(produced.tile)
                    readers[physical].append(index)
            assert len(payloads) == 1
            (tile,) = payloads
            fixed = cached_qk and node.phase == "qk"
            key = (node.phase, node.epoch, node.owner, node.tile) if fixed else (
                node.phase, node.epoch, node.owner)
            chain = chains[key]
            if fixed:
                assert tile == node.tile and 0 <= tile < 4 and not chain and node.flag == (1,)
            else:
                assert tile == len(chain) and node.flag == (int(tile == 0),)
            chain.append(index)
            mads[node.phase, node.epoch, node.owner, tile] = index
        elif node.kind == "export":
            fixed = cached_qk and node.phase == "qk"
            key = (node.phase, node.epoch, node.owner, node.tile) if fixed else (
                node.phase, node.epoch, node.owner)
            chain = chains.pop(key)
            assert len(chain) == (1 if fixed else 3 if node.phase == "qk" else 4)
            assert all(precedes(mad, index) for mad in chain)
            exports[node.phase] += 1
    assert not chains
    assert exports == Counter({"qk": active_epochs * (8 if cached_qk else 2), "pv": active_epochs * 2})
    if cached_qk:
        assert sum(node.kind == "load" and node.phase == "qk" and node.storage == "shared.l0a"
                   for node in trace) == active_epochs * 2
    assert cross_phase_overwrites["shared.l0a"] and cross_phase_overwrites["shared.l0b"]
    # These two completion nodes must remain independent; merely allocating
    # two slots while retaining MAD-current -> load-next is not a pipeline.
    for phase in (("pv",) if cached_qk else ("qk", "pv")):
        independent = 0
        for epoch in range(active_epochs):
            for owner in range(2):
                for tile in range((3 if phase == "qk" else 4) - 1):
                    current = mads[phase, epoch, owner, tile]
                    for scope in ("shared.l0a", "shared.l0b"):
                        following = loads[phase, epoch, owner, tile + 1, scope]
                        independent += not precedes(current, following) and not precedes(following, current)
        assert independent, f"{phase} has no MTE1-next / MAD-current completion independence"


def _assert_late_bases(artifact, cached_qk=False):
    mads = [call for function in artifact.device_mod.functions.values() for call in calls(function, "tl.ascend_mad")]
    assert len(mads) == 2
    by_n = {int(call.args[5]): call for call in mads}
    assert set(by_n) == {64, 256}
    analyzer = tvm.arith.Analyzer()
    for operand, pitches in ((1, {256: 8192, 64: 16384}), (2, {256: 16384, 64: 8192})):
        bases = []
        for n, mad in by_n.items():
            pointer = mad.args[operand]
            assert isinstance(pointer, tirx.Call) and pointer.op.name == "tirx.tvm_access_ptr"
            bases.append(pointer.args[1])
            bounds = analyzer.const_int_bound(pointer.args[2])
            assert bounds.min_value == 0 and bounds.max_value == (0 if cached_qk and n == 256 else pitches[n])
        assert bases[0].same_as(bases[1])
    for scope in ("shared.l0a.dyn", "shared.l0b.dyn"):
        (bank,) = [allocation.buffer for function in artifact.device_mod.functions.values()
                   for allocation in nodes(function, tirx.AllocBuffer) if allocation.buffer.scope() == scope]
        assert _bytes(bank) == (32 if cached_qk and scope == "shared.l0b.dyn" else 64) * 1024


@pytest.mark.parametrize("dtype,guarded", [("float16", False), ("bfloat16", True)],
                         ids=["fp16-three-active-epochs", "bf16-inactive-tail"])
def test_ordinary_pipelined_phases_keep_independent_rings_and_share_late_banks(dtype, guarded):
    artifact, snapshots = _lower(_program(dtype, guarded))
    _assert_logical_rings(snapshots)
    _assert_completion_and_prefetch(snapshots["tl.LowerScheduledTIR"], 2 if guarded else 3)
    _assert_late_bases(artifact)


def test_guarded_fixed_q_cache_shares_with_the_automatic_pv_ring():
    artifact, snapshots = _lower(_program("float16", guarded=True, cached_qk=True))
    _assert_logical_rings(snapshots, cached_qk=True)
    _assert_completion_and_prefetch(snapshots["tl.LowerScheduledTIR"], 2, cached_qk=True)
    _assert_late_bases(artifact, cached_qk=True)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("automatic", [False, True], ids=["closed-serial-epoch", "automatic-epoch"])
def test_balanced_l1_gemms_keep_the_actual_epoch_storage_contract(dtype, automatic):
    @T.prim_func
    def main(Q: T.Tensor((16, 64), dtype), K: T.Tensor((320, 64), dtype),
             P: T.Tensor((16, 320), dtype), V: T.Tensor((320, 64), dtype),
             Score: T.Tensor((2, 16, 320), "float32"),
             Result: T.Tensor((2, 16, 64), "float32")):
        with T.Kernel(1):
            q = T.alloc_l1((16, 64), dtype)
            k = T.alloc_l1((320, 64), dtype)
            p = T.alloc_l1((16, 320), dtype)
            v = T.alloc_l1((320, 64), dtype)
            score = T.alloc_l0c((16, 320), "float32")
            result = T.alloc_l0c((16, 64), "float32")
            if automatic:
                for epoch in T.Pipelined(2, max_stages=3):
                    T.copy(Q, q)
                    T.copy(K, k)
                    T.gemm(q, k, score, transpose_B=True, clear_accum=True)
                    T.copy(score, Score[epoch, :, :])
                    T.copy(P, p)
                    T.copy(V, v)
                    T.gemm(p, v, result, clear_accum=True)
                    T.copy(result, Result[epoch, :, :])
            else:
                for epoch in T.serial(2):
                    T.copy(Q, q)
                    T.copy(K, k)
                    T.gemm(q, k, score, transpose_B=True, clear_accum=True)
                    T.copy(score, Score[epoch, :, :])
                    T.copy(P, p)
                    T.copy(V, v)
                    T.gemm(p, v, result, clear_accum=True)
                    T.copy(result, Result[epoch, :, :])

    artifact, snapshots = _lower(main)
    before = _root(snapshots["tl.AnnotateMultiBufferEligible"])
    synced = _root(snapshots["tl.InsertSync"])
    physical = _root(snapshots["tl.MaterializeMultiBuffer"])
    selected = _pairs(snapshots["tl.AutoSchedule"])
    aliases = synced.annotations["tl.buffer_alias_map"]
    for scope in ("shared.l0a", "shared.l0b"):
        operands = [buffer for buffer in before.alloc_buffers if buffer.scope() == scope]
        assert len(operands) == 2, "Equal native chunks need no separate tail owner"
        first, second = (buffer.data for buffer in operands)
        actual = [buffer for buffer in physical.alloc_buffers
                  if buffer.data.same_as(first) or buffer.data.same_as(second)]
        assert len(actual) == 2, "Logical layouts and independent version owners remain intact"
        banks = [allocation.buffer for function in artifact.device_mod.functions.values()
                 for allocation in nodes(function, tirx.AllocBuffer)
                 if allocation.buffer.scope() == scope + ".dyn"]
        (bank,) = banks
        if automatic:
            # The existing closed-phase certificate does not admit an offset
            # automatic outer epoch. Keep its independent rings and sum all
            # actual slots, rather than inferring a max-budget from syntax.
            assert frozenset((first, second)) not in selected
            assert second not in aliases[first] and first not in aliases[second]
            assert _bytes(bank) == sum(map(_bytes, actual)) <= 65536
        else:
            assert frozenset((first, second)) in selected
            assert second in aliases[first] and first in aliases[second]
            assert _bytes(bank) == max(map(_bytes, actual)) <= 65536
    assert _PAIRS not in synced.annotations


def _matching_unit_boundary(records, event, start, end, *, notify_before=None,
                            root=False, prefix=()):
    sent, received = defaultdict(list), defaultdict(list)
    for index, (call, env, _) in enumerate(records):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag") and call.args[0].value == event:
            (sent if name == "tl.ascend_set_flag" else received)[_value(call.args[1], env)].append(index)
    # Pair the actual finite generations, rather than finding any equal ID
    # inside the interval and accidentally matching different notifications.
    for flag, notifications in sent.items():
        for notify, wait in zip(notifications, received[flag]):
            if not (start < notify < wait < end) or (notify_before is not None and notify >= notify_before):
                continue
            endpoints = [tuple((var.name, value) for var, value, _ in records[index][2])
                         for index in (notify, wait)]
            if all(endpoint[:len(prefix)] == prefix and (not root or not endpoint)
                   for endpoint in endpoints):
                return True
    return False


def _unit_stream_activations(records, result_loops):
    """Group actual work by the selected result loop's enclosing activation."""
    activations, previous = {}, None
    for index, (call, env, loops) in enumerate(records):
        name = _name(call)
        if name != _GEMM and not (name == _COPY and _region(call.args[0], env)[0].scope() == "shared.l0c"):
            continue
        selected = [depth for depth, (var, _, _) in enumerate(loops) if var.name in result_loops]
        assert len(selected) == 1, "Each selected native operation belongs to one result loop"
        depth = selected[0]
        prefix = tuple((var.name, value) for var, value, _ in loops[:depth])
        key = loops[depth][0].name, prefix
        if key != previous:
            assert key not in activations, "A closed result stream cannot interleave another stream"
            activations[key] = {"prefix": prefix, "mads": [], "exports": []}
            previous = key
        activations[key]["mads" if name == _GEMM else "exports"].append((index, call, env, loops))
    assert activations and all(item["mads"] and item["exports"] for item in activations.values())
    return list(activations.values())


def _assert_kernel_unit_boundaries(records, activations, *, c_byte_retirement=False):
    """Check real init, owner drains and restoration without a C unit edge."""
    directions = [(index, call.args[0].value, loops) for index, (call, _, loops) in enumerate(records)
                  if _name(call) == "tl.ascend_set_mmad_direction"]
    assert [(direction, loops) for _, direction, loops in directions] == [("n", ()), ("m", ())]
    setup, restore = directions[0][0], directions[1][0]
    first_work = min(item["mads"][0][0] for item in activations)
    last_work = max(index for item in activations for index, *_ in (*item["mads"], *item["exports"]))
    assert setup < first_work <= last_work < restore
    for event in ("S_M", "S_FIX"):
        assert _matching_unit_boundary(records, event, setup, first_work, root=True), event
    assert _matching_unit_boundary(records, "M_S", last_work, restore, root=True)

    trace = []
    for call, env, _ in records:
        name = _name(call)
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
            trace.append(_Command("M", "mad"))
        elif name in (_COPY, _LOAD3D):
            source = _region(call.args[0], env)[0]
            pipe = "FIX" if source.scope() == "shared.l0c" else "MTE2" if source.scope() == "global" else "MTE1"
            trace.append(_Command(pipe, "copy"))
        else:
            assert name in ("tl.ascend_set_mmad_direction", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_cube_load_config", "tl.ascend_set_load3d_config"), name
            trace.append(_Command("S", "scalar"))
    require = _prove_completions(trace)
    for number, item in enumerate(activations):
        mads, exports, prefix = item["mads"], item["exports"], item["prefix"]
        previous = activations[number - 1] if number else None
        following = activations[number + 1] if number + 1 < len(activations) else None
        start = previous["exports"][-1][0] if previous else setup
        stop = following["mads"][0][0] if following else restore
        assert _matching_unit_boundary(records, "M_FIX", start, exports[0][0],
                                       notify_before=mads[0][0], prefix=prefix), prefix
        assert _matching_unit_boundary(records, "FIX_M", exports[-1][0], stop, prefix=prefix), prefix
        # The entry fence retires preceding M work before this owner's FIX;
        # it does not turn the current native unit readiness into whole MAD.
        if previous and not c_byte_retirement:
            for old, *_ in previous["mads"]:
                require(3 * old + 2, 3 * exports[0][0] + 1)
        if following:
            for old, *_ in exports:
                require(3 * old + 2, 3 * following["mads"][0][0] + 1)
        for index, *_ in (*mads, *exports):
            require(3 * setup + 2, 3 * index + 1)
            require(3 * index + 2, 3 * restore + 1)


def _assert_joint_unit_streams(snapshots, pv_operand="ap"):
    synced = snapshots["tl.InsertSync"]
    directions = calls(synced, "tl.ascend_set_mmad_direction")
    assert Counter(call.args[0].value for call in directions) == {"n": 1, "m": 1}, "Fully selected streams keep one kernel direction"
    shells = [loop for loop in nodes(synced, tirx.For) if loop.loop_var.name == "unit_activation"]
    assert shells and all(int(loop.extent) == 1 and calls(loop.body, _GEMM) for loop in shells)

    records = list(_trace(snapshots["tl.LowerScheduledTIR"], core=2, lane=0))
    totals = {"tl.ascend_set_flag": Counter(), "tl.ascend_wait_flag": Counter()}
    for index, (call, env, loops) in enumerate(records):
        name = _name(call)
        if name in totals:
            event, flag = call.args[0].value, _value(call.args[1], env)
            assert 0 <= flag < 8
            totals[name][event, flag] += 1
        if name == _GEMM or (name == _COPY and _region(call.args[0], env)[0].scope() == "shared.l0c"):
            epochs = [value for var, value, _ in loops if var.name == "epoch"]
            assert len(epochs) == 1 and epochs[0] in (0, 1), "The inactive phase must issue no native result work"
    assert totals["tl.ascend_set_flag"] == totals["tl.ascend_wait_flag"]
    # Operand completion retains ordinary finite flags. A native C unit check
    # is not inserted into a graph as whole-MAD or whole-FIX completion.
    for event in ("MTE1_M", "M_MTE1"):
        assert any(kind == event and count for (kind, _), count in totals["tl.ascend_set_flag"].items())

    seen = Counter()
    activations = _unit_stream_activations(records, {"qk_n", "pv_owner"})
    _assert_kernel_unit_boundaries(records, activations)
    for item in activations:
        mads, exports = item["mads"], item["exports"]
        phases = {_region(call.args[0], env)[0].name for _, call, env, _ in mads}
        assert len(phases) == 1
        (operand,) = phases
        phase = "qk" if operand == "aq" else "pv"
        assert operand in ("aq", pv_operand)
        seen[phase] += 1
        controls = [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env, _ in mads]
        clears = [_value(call.args[9], env) for _, call, env, _ in mads]
        if phase == "qk":
            assert len(mads) == len(exports) == 4 and controls == [3] * 4 and clears == [1] * 4
        else:
            assert len(mads) == 8 and len(exports) == 2
            assert controls == [2, 2, 2, 3] * 2 and clears == [1, 0, 0, 0] * 2
        assert all(_value(call.annotations["unit_flag_ctrl"], env) == 3 for _, call, env, _ in exports)
    assert seen == {"qk": 4, "pv": 2}
    assert any(_name(call) == _GEMM and any(var.name == "unit_activation" for var, _, _ in loops)
               for call, _, loops in records), "The one-trip activation shell must contain actual native work"


def test_guarded_cached_qk_and_pipelined_pv_keep_unit_streams_and_phase_aliases():
    artifact, snapshots = _lower(_program("float16", guarded=True, cached_qk=True, unit_streams=True))
    _assert_logical_rings(snapshots, cached_qk=True)
    _assert_late_bases(artifact, cached_qk=True)
    _assert_joint_unit_streams(snapshots)


def test_automatic_p_cache_and_shifted_native_k_keep_joint_phase_and_unit_contracts():
    artifact, snapshots = _lower(_program("float16", guarded=True, cached_qk=True,
                                          unit_streams=True, pipelined_p_cache=True))
    operands = _assert_logical_rings(snapshots, cached_qk=True, pipelined_p_cache=True)
    _assert_late_bases(artifact, cached_qk=True)
    _assert_joint_unit_streams(snapshots, pv_operand=operands["ap"].name)

    before = snapshots["tl.AnnotateMultiBufferEligible"]
    scheduled = snapshots["tl.AutoSchedule"]
    p = allocated_buffer(scheduled, "p_l1")
    assert p.data not in _root(before).annotations["tl.buffer_versions_map"]
    assert int(_root(scheduled).annotations["tl.buffer_versions_map"][p.data]) == 2
    tvm.ir.assert_structural_equal(_root(scheduled).annotations["layout_map"][p],
                                   make_ascend_major_k_layout(p))
    (p_owner,) = [loop for loop in nodes(scheduled, tirx.For)
                  if p.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert p_owner.loop_var.name == "pv_chunk" and int(p_owner.extent) == 2
    # This checks the actual automatic outcome, not hand-written T.Stage
    # offsets: the complete native K control and its MAD are both shifted.
    control_markers = [marker for marker in nodes(scheduled, tirx.AttrStmt)
                       if marker.attr_key == "tl.schedule_unit"
                       and isinstance(marker.body, tirx.For)
                       and marker.body.loop_var.name == "gemm_sub_k_0"]
    assert len(control_markers) == 1 and int(control_markers[0].node["stage"]) == 1
    mad_markers = [marker for marker in nodes(control_markers[0].body.body, tirx.AttrStmt)
                   if marker.attr_key == "tl.schedule_unit"
                   and isinstance(marker.body, tirx.AttrStmt)
                   and marker.body.attr_key == "tl.ascend_task"
                   and calls(marker.body, _GEMM)]
    assert len(mad_markers) == 1 and int(mad_markers[0].node["stage"]) == 1
    assert int(control_markers[0].body.annotations["num_stages"]) > 1

    for name in ("tl.PrepareMultiBuffer", "tl.ResolveCore"):
        mod = snapshots[name]
        root = _root(mod)
        assert int(root.annotations["tl.buffer_versions_map"][p.data]) == 2
        assert all(int(root.annotations["tl.buffer_versions_map"][operands[role].data]) == 2
                   for role in ("ap", "bp"))
        clocks = {}
        for role, storage in (("p", p), ("a", operands["ap"]), ("b", operands["bp"])):
            (owner,) = [loop for loop in nodes(mod, tirx.For)
                        if storage.data in loop.annotations.get("multi_buffer_eligible", [])]
            counters = owner.annotations.get("tl.multi_buffer_counter_map", {})
            clocks[role] = counters[storage.data].data if storage.data in counters else owner.loop_var
        assert clocks["a"].same_as(clocks["b"])
        assert not clocks["p"].same_as(clocks["a"]), "Chunk and native-K versions must retain distinct clocks"
    physical = snapshots["tl.MaterializeMultiBuffer"]
    actual_p = allocated_buffer(physical, "p_l1")
    assert actual_p.data.same_as(p.data) and tuple(map(int, actual_p.shape)) == (2, 128, 256)
    assert tuple(map(int, allocated_buffer(physical, "accum").shape)) == (128, 256)
    # P has two 64-KiB versions. Root L1 use includes every physical version
    # of Q/K/P/V, while the independent A/B phase contract checks their banks.
    assert sum(_bytes(buffer) for buffer in _root(physical).alloc_buffers
               if buffer.scope() == "shared.l1") == 336 * 1024

    fills = Counter()
    p_read_slots = Counter()
    for call, env, _ in _trace(snapshots["tl.LowerScheduledTIR"], core=2, lane=0):
        if _name(call) not in (_COPY, _LOAD3D):
            continue
        src, source_min, source_extent = _region(call.args[0], env)
        dst, dest_min, dest_extent = _region(call.args[1], env)
        if dst.name == "p_l1":
            assert src.name == "P" and source_extent == (1, 1, 128, 256)
            assert dest_extent == (1, 128, 256) and dest_min[1:] == (0, 0)
            assert dest_min[0] in (0, 1)
            epoch, owner, row, column = source_min
            assert row == 0 and column in (0, 256)
            fills[epoch, owner, column // 256] += 1
        if src.name == "p_l1":
            assert dst.name == operands["ap"].name
            assert source_extent == (1, 128, 128) and source_min[1] == 0
            assert source_min[0] in (0, 1) and source_min[2] in (0, 128)
            p_read_slots[source_min[0]] += 1
    assert fills == Counter({(epoch, owner, chunk): 1
                             for epoch in range(2) for owner in range(2) for chunk in range(2)})
    assert p_read_slots == {0: 8, 1: 8}, "Both automatic P versions must feed every native K step"


def _admission_fixture(kind="ordinary", arch="dav-2201"):
    """Small native tasks keep the conservative two-allocation budget legal.

    These inputs already have scheduling units, canonical layouts and eligible
    K loops. AutoSchedule still uses its real solver. They isolate rejection
    from the larger DSL case whose two double rings require phase reuse.
    """
    mx = kind == "mx"
    dtype, qk_k, pv_k = ("float8_e4m3fn", 128, 128) if mx else ("float16", 16, 32)
    aq = tirx.decl_buffer((32, qk_k), dtype, name="aq", scope="shared.l0a")
    bq = tirx.decl_buffer((64, qk_k), dtype, name="bq", scope="shared.l0b")
    ap = tirx.decl_buffer((32, pv_k), dtype, name="ap", scope="shared.l0a")
    bp = tirx.decl_buffer((pv_k, 16), dtype, name="bp", scope="shared.l0b")
    operands = (aq, bq, ap, bp)
    q = tirx.decl_buffer((32, 3 * qk_k), dtype, name="q_l1", scope="shared.l1")
    k = tirx.decl_buffer((64, 3 * qk_k), dtype, name="k_l1", scope="shared.l1")
    p = tirx.decl_buffer((32, 4 * pv_k), dtype, name="p_l1", scope="shared.l1")
    v = tirx.decl_buffer((4 * pv_k, 16), dtype, name="v_l1", scope="shared.l1")
    accum = tirx.decl_buffer((32, 64), "float32", name="accum", scope="shared.l0c")
    score = tirx.decl_buffer((3, 2, 32, 64), "float32", name="Score")
    result = tirx.decl_buffer((3, 2, 32, 16), "float32", name="Result")
    epoch = tirx.Var("epoch", "int32")
    buffers = [q, k, p, v, *operands, accum]
    a_layout = make_ascend_major_k_layout if mx else make_ascend_l0a_2201_layout
    layouts = {aq: a_layout(aq), ap: a_layout(ap),
               bq: make_ascend_major_k_layout(bq), bp: make_ascend_major_mn_layout(bp),
               accum: make_ascend_l0c_layout(accum)}
    annotations = {"layout_map": layouts,
                   "tl.buffer_versions_map": {buffer.data: 1 for buffer in (q, k, p, v, accum)}}
    state = None
    if kind == "mutable-activation":
        state = tirx.decl_buffer((1,), "int32", name="activation", scope="local.var")
        buffers.append(state)
    if kind == "extra-root-allocation":
        for scope in ("shared.l0a", "shared.l0b"):
            extra = tirx.decl_buffer((32, 16), "float16", name="extra_" + scope[-3:], scope=scope)
            buffers.append(extra)
            layouts[extra] = (make_ascend_l0a_2201_layout(extra) if scope.endswith("a")
                              else make_ascend_major_k_layout(extra))
    if kind == "manual-versions":
        annotations["tl.manual_multi_buffer"] = {buffer.data: 1 for buffer in operands}

    def active(buffer, rows=None, columns=None):
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(0, rows or buffer.shape[0]),
                                          tvm.ir.Range.from_min_extent(0, columns or buffer.shape[1])])

    scales = {}
    if mx:
        # This is a real block-scaled tile operator, not a renamed dense MAD.
        # Only AutoSchedule is exercised: it must not choose this 2201 reuse
        # recipe for MX; later native target legality is a separate contract.
        bindings = {}
        for operand in operands:
            sf_rows = 16 if operand.same_as(bp) else int(operand.shape[0])
            sf = tirx.decl_buffer((sf_rows, 2), "uint16", name="sf_" + operand.name,
                                 scope="shared.l0a.sf" if operand.scope().endswith("a") else "shared.l0b.sf")
            buffers.append(sf)
            scales[operand.data] = sf
            bindings[sf.data] = operand.data
        annotations["tl.l0_sf_bindings"] = bindings

    def mad(a, b, width, clear, *, transpose):
        if mx:
            return tirx.Evaluate(T.gemm_blockscaled(
                active(a), active(b), active(accum, 32, width),
                active(scales[a.data]), active(scales[b.data]),
                transpose_B=transpose, clear_accum=clear))
        return tirx.Evaluate(T.gemm(active(a), active(b), active(accum, 32, width),
                                   transpose_B=transpose, clear_accum=clear))

    phases = []
    for phase, a, b, source_a, source_b, width, trips, native_k, out in (
            ("qk", aq, bq, q, k, 64, 3, qk_k, score),
            ("pv", ap, bp, p, v, 16, 4, pv_k, result)):
        owner, tile = (tirx.Var(phase + suffix, "int32") for suffix in ("_owner", "_k"))
        tasks = [
            unit(copy(source_a, a, src_indices=[0, tile * native_k], src_shape=a.shape),
                 core=2, stage=-1, cost=(2, 1)),
            unit(copy(source_b, b,
                      src_indices=[0, tile * native_k] if phase == "qk" else [tile * native_k, 0],
                      src_shape=b.shape), core=2, stage=-1, cost=(2, 1)),
            unit(mad(a, b, width, tile == 0, transpose=phase == "qk"),
                 core=2, stage=-1, cost=(3, 1)),
        ]
        inner_annotations = {"multi_buffer_eligible": [a.data, b.data],
                             "tl.ascend_auto_pipeline": True, "num_stages": 2}
        if kind == "common-auto-owner":
            inner_annotations = {"tl.ascend_auto_pipeline": False, "num_stages": 1}
        inner = unit(tirx.For(tile, 0, trips, tirx.ForKind.SERIAL, seq(*tasks),
                             annotations=inner_annotations), core=None, stage=0)
        output_epoch = epoch
        if kind == "escaped-address":
            output_epoch = tirx.call_extern("int32", "foreign_index", aq.data, bq.data)
        export = unit(copy(accum, out, src_shape=[32, width],
                           dst_indices=[output_epoch, owner, 0, 0], dst_shape=[1, 1, 32, width]),
                      core=2, stage=0, cost=(1, 1))
        body = [inner, export]
        if kind == "overlapping-lifetime" and phase == "pv":
            body.append(unit(mad(aq, bq, 64, True, transpose=True), core=2, stage=0, cost=(3, 1)))
        guard = state[0] == 0 if state is not None else None
        phase_stage = -1 if kind == "common-auto-owner" else int(kind == "different-stage" and phase == "pv")
        phases.append(unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, seq(*body)),
                           core=None, stage=phase_stage, guard=guard))
    epoch_annotations = {}
    if kind == "common-auto-owner":
        epoch_annotations = {"multi_buffer_eligible": [buffer.data for buffer in operands],
                             "tl.ascend_auto_pipeline": True, "num_stages": 2}
    elif kind == "different-stage":
        epoch_annotations = {"enable_offset": True, "num_stages": 2,
                             "tl.ascend_auto_pipeline": False}
    if state is not None:
        phases.insert(0, unit(tirx.BufferStore(state, epoch % 2, [0]), core=2, stage=0, cost=(1, 1)))
    loop = unit(tirx.For(epoch, 0, 3, tirx.ForKind.SERIAL, seq(*phases),
                        annotations=epoch_annotations), core=None, stage=0)
    mod = kernel(loop, buffers=buffers, params=[score, result], annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({**_TARGET, "arch": arch}))
    return mod, operands


def test_native_admission_fixture_has_two_independent_automatic_owners():
    before, operands = _admission_fixture()
    after = transform.AutoSchedule()(before)
    assert _pairs(after) == {frozenset(buffer.data for buffer in operands[::2]),
                             frozenset(buffer.data for buffer in operands[1::2])}
    versions = _root(after).annotations["tl.buffer_versions_map"]
    assert all(int(versions[buffer.data]) == 2 for buffer in operands)


@pytest.mark.parametrize("kind,arch", [
    ("overlapping-lifetime", "dav-2201"),
    ("escaped-address", "dav-2201"),
    ("mutable-activation", "dav-2201"),
    ("common-auto-owner", "dav-2201"),
    ("extra-root-allocation", "dav-2201"),
    ("manual-versions", "dav-2201"),
    ("mx", "dav-2201"),
    ("reuse-disabled", "dav-2201"),
    ("ordinary", "dav-3510"),
], ids=["overlapping-lifetime", "escaped-address", "mutable-activation", "common-auto-owner",
        "extra-allocation", "manual-versions", "mx", "reuse-disabled", "3510"])
def test_unproved_phases_keep_the_conservative_bank_contract(kind, arch):
    before, operands = _admission_fixture(kind, arch)
    # The opt-out disables the reuse choice, not the complete-bank resource
    # reservation below. This otherwise admissible fixture must still charge
    # both logical allocations and their independently selected ring counts.
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": kind == "reuse-disabled"}):
        after = transform.AutoSchedule()(before)
    root = _root(after)
    assert not _pairs(after)
    aliases = root.annotations.get("tl.buffer_alias_map", {})
    assert all(second.data not in aliases.get(first.data, [])
               for first, second in ((operands[0], operands[2]), (operands[1], operands[3])))
    for operand in operands:
        actual = allocated_buffer(after, operand.name)
        assert actual.same_as(operand)
        tvm.ir.assert_structural_equal(root.annotations["layout_map"][actual],
                                       _root(before).annotations["layout_map"][operand])
    for operation in (_COPY, _GEMM, "tl.tileop.gemm_blockscaled"):
        assert len(calls(before, operation)) == len(calls(after, operation))
    # The selected map contains scheduled rings, not every root allocation.
    # A manual physical allocation is charged once from its actual shape;
    # an untouched, ineligible root allocation still reserves one full tile.
    versions = root.annotations.get("tl.buffer_versions_map", {})
    manual = root.annotations.get("tl.manual_multi_buffer", {})
    if kind == "manual-versions":
        assert not versions
        assert set(manual) == {buffer.data for buffer in operands}
        assert all(int(count) == 1 for count in manual.values())
    implicit = Counter()
    for scope in ("shared.l0a", "shared.l0b"):
        charge = 0
        for buffer in root.alloc_buffers:
            if buffer.scope() == scope:
                if buffer.data in manual:
                    assert buffer.data not in versions
                    charge += _bytes(buffer)
                elif buffer.data in versions:
                    charge += _bytes(buffer) * int(versions[buffer.data])
                else:
                    # Do not hide a missing ring count behind a default one.
                    # Only an actually unused allocation has this implicit
                    # single-slot contract at the AutoSchedule boundary.
                    assert not any(buffer.data in loop.annotations.get("multi_buffer_eligible", [])
                                   for loop in nodes(after, tirx.For))
                    assert not any(load.buffer.data.same_as(buffer.data)
                                   for load in nodes(before, tirx.BufferLoad))
                    charge += _bytes(buffer)
                    implicit[scope] += 1
        assert charge <= 64 * 1024
    assert implicit == (Counter({"shared.l0a": 1, "shared.l0b": 1})
                        if kind == "extra-root-allocation" else Counter())


def test_infeasible_cross_phase_manual_stages_keep_the_existing_diagnostic():
    # PV at stage one shares a fixed C with next-epoch stage-zero QK. This
    # hand schedule is infeasible under its ordinary dependencies. Rejecting
    # it is preferable to accepting a max-budget alias without a legal order.
    before, _ = _admission_fixture("different-stage")
    with pytest.raises(RuntimeError, match="Manual schedule constraints are infeasible"):
        transform.AutoSchedule()(before)
