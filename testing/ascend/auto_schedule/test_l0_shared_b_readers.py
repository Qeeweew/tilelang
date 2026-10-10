"""A versioned B generation retires after all finite serial MAD readers.

The CPU proof uses actual prepared slots and binary whole-pipe flags. It
does not treat unit C readiness as operand completion or model device latency.
"""
from collections import Counter, defaultdict

import pytest
from tilelang.ascend import language as T

from testing.ascend._ir import allocated_buffer
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower, _root
from testing.ascend.auto_schedule.test_l0_operand_slot_pool import _Command
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import _prove_binary_protocol


def _program(dtype, readers):
    @T.prim_func
    def main(Q: T.Tensor((3, readers, 64, 64), dtype),
             K: T.Tensor((3, 384, 64), dtype),
             P: T.Tensor((3, readers, 64, 384), dtype),
             V: T.Tensor((3, 384, 64), dtype),
             Score: T.Tensor((3, readers, 64, 384), "float32"),
             Result: T.Tensor((3, readers, 64, 64), "float32")):
        with T.Kernel(1):
            q = T.alloc_l1((readers, 64, 64), dtype)
            k = T.alloc_l1((384, 64), dtype)
            p = T.alloc_l1((64, 384), dtype)
            v = T.alloc_l1((384, 64), dtype)
            aq = T.alloc_l0a((readers, 64, 64), dtype)
            bq = T.alloc_l0b((128, 64), dtype)
            cq = T.alloc_l0c((64, 128), "float32")
            cp = T.alloc_l0c((64, 64), "float32")
            T.annotate_buffer_versions({q: 1, k: 1, p: 1, v: 1,
                                        aq: 1, bq: 2, cq: 1, cp: 1})
            for epoch in T.serial(3, annotations={"enable_offset": True,
                                                  "tl.ascend_preserve_issue_order": True}):
                with T.Stage(0):
                    T.copy(K[epoch, :, :], k)
                    for owner in T.serial(readers):
                        T.copy(Q[epoch, owner, :, :], q[owner, :, :])
                    for owner in T.serial(readers):
                        T.copy(q[owner, :, :], aq[owner, :, :])
                    for n in T.serial(3):
                        T.copy(k[n * 128:(n + 1) * 128, :], bq)
                        for owner in T.serial(readers):
                            T.gemm(aq[owner, :, :], bq, cq,
                                   transpose_B=True, clear_accum=True)
                            T.copy(cq, Score[epoch, owner, :, n * 128:(n + 1) * 128])
                with T.Stage(1):
                    T.copy(V[epoch, :, :], v)
                    for owner in T.serial(readers):
                        T.copy(P[epoch, owner, :, :], p)
                        for chunk in T.Pipelined(3):
                            T.gemm(p[:, chunk * 128:(chunk + 1) * 128],
                                   v[chunk * 128:(chunk + 1) * 128, :], cp,
                                   transpose_B=False, clear_accum=chunk == 0)
                        T.copy(cp, Result[epoch, owner, :, :])
    return main


@pytest.mark.parametrize("dtype,readers", [("float16", 2), ("bfloat16", 3)])
def test_shared_versioned_b_keeps_every_reader_and_last_mad_completion(dtype, readers):
    artifact, snapshots = _lower(_program(dtype, readers))
    synced = snapshots["tl.InsertSync"]
    bq = allocated_buffer(synced, "bq")
    bp = allocated_buffer(synced, "gemm_l0b_0")
    aliases = _root(synced).annotations["tl.buffer_alias_map"]
    assert bp.data in aliases[bq.data] and bq.data in aliases[bp.data]
    commands, payload, pending, live = [], {}, [], defaultdict(list)
    loaded, uses, loads, active_epoch = {}, Counter(), Counter(), {}
    records = list(_trace(snapshots["tl.LowerScheduledTIR"], core=2, lane=0))
    for call, env, loops in records:
        name = _name(call)
        coordinates = {var.name: value for var, value, _ in loops}
        index = len(commands)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            src, dst = event.split("_")
            flag = _value(call.args[1], env)
            assert 0 <= flag < 8
            setting = name == "tl.ascend_set_flag"
            commands.append(_Command(src if setting else dst,
                                     "set" if setting else "wait", (event, flag)))
        elif name == "tl.ascend_pipe_barrier":
            commands.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            src, origin, _ = _region(call.args[0], env)
            dst, destination, _ = _region(call.args[1], env)
            pipe = "MTE2" if src.scope() == "global" else (
                "FIX" if src.scope() == "shared.l0c" else "MTE1")
            if src.scope() == "global" and dst.name in ("k", "v"):
                active_epoch["qk" if dst.name == "k" else "pv"] = origin[0]
            if dst.scope() == "shared.l0b":
                assert len(destination) == 3 and destination[-2:] == (0, 0)
                slot = destination[0]
                assert slot in (0, 1)
                pending.extend((reader, index) for reader in live[slot])
                live[slot] = []
                phase = "qk" if dst.name == "bq" else "pv"
                identity = phase, active_epoch[phase], origin[-2] // 128
                payload[slot], loaded[slot] = identity, index
                loads[phase] += 1
            commands.append(_Command(pipe, "copy"))
        elif name == "tl.tileop.gemm":
            b, origin, _ = _region(call.args[1], env)
            assert len(origin) == 3 and origin[-2:] == (0, 0)
            phase = "qk" if b.name == "bq" else "pv"
            epoch, owner = active_epoch[phase], coordinates["owner"]
            tile = uses[phase, epoch, owner]
            slot = origin[0]
            assert payload[slot] == (phase, epoch, tile)
            pending.append((loaded[slot], index))
            live[slot].append(index)
            assert _value(call.args[9], env) == (1 if phase == "qk" else int(tile == 0))
            uses[phase, epoch, owner] += 1
            commands.append(_Command("M", "mad"))
        else:
            assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config",
                            "tl.ascend_set_mmad_direction"), name
            commands.append(_Command("S", "configuration"))
    precedes = _prove_binary_protocol(commands)
    assert all(precedes(src, dst) for src, dst in pending), (
        "Loads must finish before each MAD; every reader must finish before slot overwrite")
    assert loads == Counter(qk=9, pv=9 * readers)
    assert uses == Counter({(phase, epoch, owner): 3 for phase in ("qk", "pv")
                            for epoch in range(3) for owner in range(readers)})
    assert artifact.kernel_source
