"""Fixed operand planes retire as a phase before an independent native ring.

The leading plane axis is logical storage, not a buffer-version credit.
Actual binary completion must protect every MAD reader before overlapping
bytes are overwritten by either phase, including across outer epochs.
"""
import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import allocated_buffer, nodes
from testing.ascend.auto_schedule.test_owner_result_completion import _trace
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import (
    _assert_balanced_operand_completions, _bytes, _lower, _pairs, _root,
)


def _program(dtype, planes=2, interleaved=False, reader_planes=None,
             automatic=False, groups=1, epochs=3, balanced=False):
    readers = planes if reader_planes is None else reader_planes
    @T.prim_func
    def main(Q: T.Tensor((groups, epochs, planes, 128, 128), dtype),
             K: T.Tensor((groups, epochs, 384, 128), dtype),
             P: T.Tensor((groups, epochs, planes, 128, 384), dtype),
             V: T.Tensor((groups, epochs, 384, 128), dtype),
             Score: T.Tensor((groups, epochs, planes, 128, 384), "float32"),
             Result: T.Tensor((groups, epochs, planes, 128, 128), "float32")):
        with T.Kernel(groups) as group:
            q = T.alloc_l1((planes, 128, 128), dtype)
            k = T.alloc_l1((128, 128), dtype)
            p = T.alloc_l1((128, 128), dtype)
            v = T.alloc_l1((384, 128), dtype)
            aq = T.alloc_l0a((planes, 128, 128), dtype)
            bq = T.alloc_l0b((128, 128), dtype)
            cq = T.alloc_l0c((128, 128), "float32")
            cp = T.alloc_l0c((128, 128), "float32")
            T.annotate_buffer_versions({q: 1, k: 1, p: 1, v: 1,
                                        aq: 1, bq: 2, cq: 1, cp: 1})
            for epoch in T.serial((epochs + groups - 1 - group) // groups if balanced else epochs,
                                 annotations={"enable_offset": True,
                                                  "tl.ascend_preserve_issue_order": True}):
                with T.Stage(0):
                    for owner in T.serial(planes):
                        T.copy(Q[group, epoch, owner, :, :], q[owner, :, :])
                    for owner in T.serial(planes):
                        T.copy(q[owner, :, :], aq[owner, :, :])
                    for n in T.serial(3):
                        T.copy(K[group, epoch, n * 128:(n + 1) * 128, :], k)
                        T.copy(k, bq)
                        for owner in T.serial(readers):
                            T.gemm(aq[owner, :, :], bq, cq,
                                   transpose_B=True, clear_accum=True)
                            T.copy(cq, Score[group, epoch, owner, :, n * 128:(n + 1) * 128])
                with T.Stage(1):
                    T.copy(V[group, epoch, :, :], v)
                    for owner in T.serial(planes):
                        for chunk in T.Pipelined(3):
                            T.copy(P[group, epoch, owner, :, chunk * 128:(chunk + 1) * 128], p)
                            T.gemm(p, v[chunk * 128:(chunk + 1) * 128, :], cp,
                                   clear_accum=chunk == 0)
                        T.copy(cp, Result[group, epoch, owner, :, :])
                if interleaved:
                    with T.Stage(2):
                        for owner in T.serial(planes):
                            T.gemm(aq[owner, :, :], bq, cq,
                                   transpose_B=True, clear_accum=True)
                            T.copy(cq, Score[group, epoch, owner, :, :128])
    if not automatic:
        return main

    def automatic_epoch(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.ascend_stage":
            return node.body
        if isinstance(node, tirx.For) and node.loop_var.name == "epoch":
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            node.body, annotations={"tl.ascend_auto_pipeline": True,
                                                    "num_stages": 4})

    return main.with_body(tirx.stmt_functor.ir_transform(main.body, None,
                                                       automatic_epoch))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_fixed_planes_share_l0a_with_an_independent_pv_ring(dtype):
    artifact, snapshots = _lower(_program(dtype))
    scheduled = snapshots["tl.AutoSchedule"]
    aq = allocated_buffer(scheduled, "aq")
    ap = allocated_buffer(scheduled, "gemm_l0a_0")
    assert frozenset((aq.data, ap.data)) in _pairs(scheduled)
    assert _bytes(aq) == 65536
    physical = snapshots["tl.MaterializeMultiBuffer"]
    assert tuple(map(int, allocated_buffer(physical, "aq").shape)) == (2, 128, 128)
    assert _bytes(allocated_buffer(physical, "gemm_l0a_0")) == 65536
    aliases = _root(snapshots["tl.InsertSync"]).annotations["tl.buffer_alias_map"]
    assert ap.data in aliases[aq.data] and aq.data in aliases[ap.data]
    _assert_balanced_operand_completions(snapshots["tl.LowerScheduledTIR"], _pairs(scheduled))
    assert artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_automatic_schedule_keeps_the_resident_plane_interval_closed(dtype):
    artifact, snapshots = _lower(_program(dtype, automatic=True))
    selected = _pairs(snapshots["tl.AutoSchedule"])
    aq = allocated_buffer(snapshots["tl.AutoSchedule"], "aq")
    ap = allocated_buffer(snapshots["tl.AutoSchedule"], "gemm_l0a_0")
    assert frozenset((aq.data, ap.data)) in selected
    # This oracle checks cycles in actual binary credits and every physical
    # overwrite against the old whole-MAD readers, across fill/drain epochs.
    _assert_balanced_operand_completions(snapshots["tl.LowerScheduledTIR"], selected)
    assert artifact.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("epochs,expected", [(7, (3, 2, 2)), (2, (1, 1, 0))])
def test_group_dependent_epochs_keep_completion_for_every_group(dtype, automatic, epochs, expected):
    artifact, snapshots = _lower(_program(dtype, automatic=automatic, groups=3,
                                         epochs=epochs, balanced=True))
    selected = _pairs(snapshots["tl.AutoSchedule"])
    aq = allocated_buffer(snapshots["tl.AutoSchedule"], "aq")
    ap = allocated_buffer(snapshots["tl.AutoSchedule"], "gemm_l0a_0")
    assert frozenset((aq.data, ap.data)) in selected
    lowered = snapshots["tl.LowerScheduledTIR"]
    for group, count in enumerate(expected):
        # The actual binary flags must complete on both lengths and on an
        # empty group. This also checks all old MAD readers before overwrite.
        _assert_balanced_operand_completions(lowered, selected, group=group,
                                            allow_empty=count == 0)
        exports = [call for call, _, _ in _trace(lowered, core=2, lane=0, group=group)
                   if call.op.name == "tl.tileop.ascend_copy"
                   and call.args[1].args[0].buffer.name == "Result"]
        assert len(exports) == count * 2
    if all(expected):
        assert artifact.kernel_source.count("asc_set_mmad_direction_n();") == 1
        assert artifact.kernel_source.count("asc_set_mmad_direction_m();") == 1
    # A one-epoch stream may retain ordinary whole-MAD/FIX handshakes when
    # the independent reverse-completion certificate cannot select units.
    # Its exact data transactions and physical completions are checked above.


def test_storage_dependent_epoch_extent_cannot_claim_a_closed_phase():
    program = _program("float16", groups=3, epochs=7, balanced=True)
    query = next(buffer for buffer in program.buffer_map.values() if buffer.name == "Q")
    def mutable_extent(node):
        if isinstance(node, tirx.For) and node.loop_var.name == "epoch":
            extent = tirx.Cast("int32", query[0, 0, 0, 0, 0]) % 3
            return tirx.For(node.loop_var, node.min, extent, node.kind,
                            node.body, annotations=node.annotations)
    broken = program.with_body(tirx.stmt_functor.ir_transform(program.body, None, mutable_extent))
    with pytest.raises(tvm.error.InternalError, match="scope.*shared.l0a.*capacity"):
        _lower(broken)


@pytest.mark.parametrize("fault", ["interposed-mate", "split-cache-stage"])
def test_prepared_broken_cache_interval_cannot_claim_completion(fault):
    _, snapshots = _lower(_program("float16"))
    prepared = snapshots["tl.ResolveCore"]

    def break_interval(node):
        if isinstance(node, tirx.For) and node.loop_var.name == "epoch":
            units = list(node.body.seq)
            assert len(units) == 5
            if fault == "interposed-mate":
                # Q GM load, Q plane load, V GM load, PV, Q plane readers.
                units = [units[i] for i in (0, 1, 3, 4, 2)]
            else:
                reader = units[2]
                annotations = dict(reader.node)
                annotations["stage"] = tirx.IntImm("int32", 1)
                units[2] = tirx.AttrStmt(annotations, reader.attr_key,
                                        reader.value, reader.body)
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            tirx.SeqStmt(units), annotations=node.annotations)

    function = prepared["main"]
    broken = tvm.IRModule({"main": function.with_body(
        tirx.stmt_functor.ir_transform(function.body, None, break_interval))})
    with function.attrs["target"], pytest.raises(
            tvm.error.InternalError,
            match="uncovered multi-buffer owner exclusion|completion-proven physical alias"):
        transform.InsertSync()(broken)


def test_fixed_planes_still_obey_the_complete_l0a_byte_capacity():
    with pytest.raises(tvm.error.InternalError, match="scope.*shared.l0a.*capacity"):
        _lower(_program("float16", planes=3))


def test_interleaved_plane_lifetimes_cannot_use_a_max_budget_bank():
    with pytest.raises(tvm.error.InternalError, match="scope.*shared.l0a.*capacity"):
        _lower(_program("float16", interleaved=True))


def test_incomplete_plane_readers_cannot_manufacture_the_missing_credits():
    with pytest.raises(tvm.error.InternalError, match="uncovered multi-buffer owner exclusion|completion-proven physical alias"):
        _lower(_program("float16", reader_planes=1))
