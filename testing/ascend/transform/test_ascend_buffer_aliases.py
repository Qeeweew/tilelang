"""Manual alias inference uses completion edges on the accessing hardware pipe."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


def _call(name, *args):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl." + name), list(args)))


def _load(gm, ub):
    return _call("ascend_copy_gm_to_ubuf", ub.access_ptr("w"), gm.access_ptr("r"), 0, 1, 256, 0, 0, 0, 0, 0, 0)


def _store(ub, gm):
    return _call("ascend_copy_ubuf_to_gm", gm.access_ptr("w"), ub.access_ptr("r"), 0, 1, 256, 0, 0, 0)


def _flags(pipe, event_id=0):
    return [_call("ascend_set_flag", pipe, event_id), _call("ascend_wait_flag", pipe, event_id)]


def _module(buffers, body, gm, *, params=(), target=None):
    function = tirx.PrimFunc([gm.data, *params], tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in buffers], *body]))
    if target is not None:
        function = function.with_attr("target", target)
    return tvm.IRModule({"main": function})


def _buffers():
    return (
        tirx.decl_buffer((64,), "float32", name="gm"),
        tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn"),
        tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn"),
    )


def _reuse_body(gm, a, b, between):
    return [_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm), *between,
            _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)]


def _may_alias(after, a, b):
    (contract,) = [node.node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]
    return b.data in contract[a.data]


@pytest.mark.parametrize("release", ["none", "acquire-only", "MTE3_MTE2", "PIPE_ALL"])
def test_dma_read_requires_a_completion_edge_before_reuse(release):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    between = []
    if release == "acquire-only":
        between = [_call("ascend_wait_flag", "MTE3_MTE2", 0)]
    elif release == "MTE3_MTE2":
        between = _flags(release)
    elif release == "PIPE_ALL":
        between = [_call("ascend_pipe_barrier", "PIPE_ALL")]
    before = _module(
        [a, b], [_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm), *between, _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)], gm
    )
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release in ("MTE3_MTE2", "PIPE_ALL"))


@pytest.mark.parametrize("release_pipe", ["S_MTE2", "V_MTE2"])
def test_pad_value_operand_is_scalar_even_inside_vector_block(release_pipe):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    body = tirx.SeqStmt(
        [_load(gm, a), *_flags("MTE2_S"), tirx.Evaluate(T.ascend_set_copy_pad_value(a[0])), *_flags(release_pipe), _load(gm, b)]
    )
    before = _module([a, b], [tirx.SBlock([], [], [], "VECTOR", body)], gm)
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (release_pipe == "S_MTE2")


@pytest.mark.parametrize("mode", [2, 4], ids=["2201-collective", "unsupported-2201-mode4"])
@pytest.mark.parametrize("release_pipe", ["PIPE_MTE1", "PIPE_FIX"])
def test_cross_core_handshake_only_releases_its_local_pipe(release_pipe, mode):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.l1")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    l0a = tirx.decl_buffer((64,), "float32", name="l0a", scope="shared.l0a")
    # The Cube reader is MTE1. A FIX release cannot complete that read, even
    # if Vector returns a token that Cube acquires on MTE2 before filling b.
    cube_read = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_copy_gm_to_cbuf", a.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
                *_flags("MTE2_MTE1"),
                _call("ascend_load_cbuf_to_ca", l0a.access_ptr("w"), a.access_ptr("r")),
                _call("ascend_cross_core_set_flag", mode, release_pipe, 0),
            ]
        ),
    )
    vector = tirx.SBlock(
        [],
        [],
        [],
        "VECTOR",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", mode, "PIPE_MTE3", 0),
                _call("ascend_cross_core_set_flag", mode, "PIPE_MTE3", 1),
            ]
        ),
    )
    cube_write = tirx.SBlock(
        [],
        [],
        [],
        "CUBE",
        tirx.SeqStmt(
            [
                _call("ascend_cross_core_wait_flag", mode, "PIPE_MTE2", 1),
                _call("ascend_copy_gm_to_cbuf", b.access_ptr("w"), gm.access_ptr("r"), 0, 16, 0, 4, 16, 0, 0, 0, 4, "float32"),
            ]
        ),
    )
    lane = tirx.Var("worker", "int32")
    binding = tirx.IterVar(tvm.ir.Range(0, 2), lane, tirx.IterVar.ThreadIndex, "cthread")
    vector = tirx.AttrStmt(binding, "thread_extent", 2, vector)
    before = _module([a, b, l0a], [cube_read, vector, cube_write], gm,
                     target=tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    after = transform.InferBufferAliases()(before)
    assert _may_alias(after, a, b) == (mode == 2 and release_pipe == "PIPE_MTE1")


def test_unbound_event_parameter_does_not_become_zero():
    gm, a, b = _buffers()
    event_id = tirx.Var("event_id", "int32")
    between = [_call("ascend_set_flag", "MTE3_MTE2", event_id),
               _call("ascend_wait_flag", "MTE3_MTE2", 0)]
    before = _module([a, b], _reuse_body(gm, a, b, between), gm, params=[event_id])
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("thread_tag", ["blockIdx.x", "threadIdx.x"])
def test_thread_binding_is_not_a_loop_zero_substitution(thread_tag):
    gm, a, b = _buffers()
    worker = tirx.Var("worker", "int32")
    binding = tirx.IterVar(tvm.ir.Range(0, 2), worker, tirx.IterVar.ThreadIndex, thread_tag)
    between = [_call("ascend_set_flag", "MTE3_MTE2", worker),
               _call("ascend_wait_flag", "MTE3_MTE2", 0)]
    body = tirx.AttrStmt(binding, "thread_extent", 2,
                        tirx.SeqStmt(_reuse_body(gm, a, b, between)))
    before = _module([a, b], [body], gm)
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("bridge_core", ["CUBE", "VECTOR"])
def test_local_hard_event_ids_do_not_cross_core_banks(bridge_core):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.l1")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    l0a = tirx.decl_buffer((64,), "float32", name="l0a", scope="shared.l0a")

    def load_l1(buffer):
        return _call("ascend_copy_gm_to_cbuf", buffer.access_ptr("w"), gm.access_ptr("r"),
                     0, 16, 0, 4, 16, 0, 0, 0, 4, "float32")

    first = tirx.SBlock([], [], [], "CUBE", tirx.SeqStmt([
        load_l1(a), *_flags("MTE2_MTE1"),
        _call("ascend_load_cbuf_to_ca", l0a.access_ptr("w"), a.access_ptr("r")),
        _call("ascend_set_flag", "MTE1_S", 0)]))
    bridge = tirx.SBlock([], [], [], bridge_core, tirx.SeqStmt([
        _call("ascend_wait_flag", "MTE1_S", 0),
        _call("ascend_set_flag", "S_MTE2", 0)]))
    last = tirx.SBlock([], [], [], "CUBE", tirx.SeqStmt([
        _call("ascend_wait_flag", "S_MTE2", 0), load_l1(b)]))
    before = _module([a, b, l0a], [first, bridge, last], gm)
    # a's final access is on MTE1. Only an acquire in that same Cube bank
    # can carry its completion through Scalar to b's MTE2 overwrite.
    assert _may_alias(transform.InferBufferAliases()(before), a, b) == (bridge_core == "CUBE")


def test_nested_nonzero_loop_values_are_distinct_bindings():
    gm, a, b = _buffers()
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    first = tirx.SeqStmt([_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm),
                         _call("ascend_set_flag", "MTE3_MTE2", i),
                         _call("ascend_wait_flag", "MTE3_MTE2", j)])
    nested = tirx.For(i, 2, 1, tirx.ForKind.SERIAL,
                      tirx.For(j, 5, 1, tirx.ForKind.SERIAL, first))
    before = _module([a, b], [nested, _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)], gm)
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("name", ["ticket", "sid_named_ticket"])
def test_pure_bind_event_identity_is_independent_of_variable_name(name):
    gm, a, b = _buffers()
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    ticket = tirx.Var(name, "int32")
    first = tirx.SeqStmt([tirx.Bind(ticket, i + j), _load(gm, a),
                         *_flags("MTE2_MTE3"), _store(a, gm),
                         _call("ascend_set_flag", "MTE3_MTE2", ticket),
                         _call("ascend_wait_flag", "MTE3_MTE2", i + j)])
    nested = tirx.For(i, 2, 1, tirx.ForKind.SERIAL,
                      tirx.For(j, 5, 1, tirx.ForKind.SERIAL, first))
    before = _module([a, b], [nested, _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)], gm)
    assert _may_alias(transform.InferBufferAliases()(before), a, b)


def _loop_handoff(gm, a, b, extent, *, bridge):
    iteration = tirx.Var("iteration", "int32")
    acquire = [_call("ascend_wait_flag", "MTE3_V", 0)]
    write = [_load(gm, b), *_flags("MTE2_MTE3", 1), _store(b, gm)]
    if bridge:
        acquire.append(_call("ascend_set_flag", "V_MTE2", 2))
        write.insert(0, _call("ascend_wait_flag", "V_MTE2", 2))
    loop = tirx.For(iteration, 0, extent, tirx.ForKind.SERIAL, tirx.SeqStmt([
        tirx.IfThenElse(iteration == 0, acquire[0] if len(acquire) == 1 else tirx.SeqStmt(acquire), None),
        tirx.IfThenElse(iteration == 1, tirx.SeqStmt(write), None),
    ]), annotations={"enable_offset": True})
    # The only completion of a's MTE3 read is acquired on V. Merely entering
    # the next iteration cannot transfer that completion to the MTE2 write.
    return [_load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm),
            _call("ascend_set_flag", "MTE3_V", 0), loop]


@pytest.mark.parametrize("bridge", [False, True], ids=["independent-pipes", "real-completion-bridge"])
def test_loop_progress_does_not_complete_an_independent_pipe(bridge):
    gm, a, b = _buffers()
    before = _module([a, b], _loop_handoff(gm, a, b, 2, bridge=bridge), gm)
    assert _may_alias(transform.InferBufferAliases()(before), a, b) == bridge


@pytest.mark.parametrize("dynamic", [False, True], ids=["finite-budget-overflow", "unknown-trip"])
def test_unproved_loop_instances_do_not_create_completion(dynamic):
    gm, a, b = _buffers()
    trip = tirx.Var("trip", "int32")
    extent = trip if dynamic else 10000
    before = _module([a, b], _loop_handoff(gm, a, b, extent, bridge=False), gm,
                     params=[trip] if dynamic else [])
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("notify_all", [False, True], ids=["missing-participant", "empty-owner-still-notifies"])
def test_collective_completion_requires_the_empty_aiv_participant(notify_all):
    gm = tirx.decl_buffer((64,), "float32", name="gm")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.l1")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.l1")
    l0a = tirx.decl_buffer((64,), "float32", name="l0a", scope="shared.l0a")
    payload = tirx.decl_buffer((64,), "float32", name="payload", scope="shared.dyn")
    worker = tirx.Var("worker", "int32")

    def load_l1(buffer):
        return _call("ascend_copy_gm_to_cbuf", buffer.access_ptr("w"), gm.access_ptr("r"),
                     0, 16, 0, 4, 16, 0, 0, 0, 4, "float32")

    cube_first = tirx.SBlock([], [], [], "CUBE", tirx.SeqStmt([
        load_l1(a), *_flags("MTE2_MTE1"),
        _call("ascend_load_cbuf_to_ca", l0a.access_ptr("w"), a.access_ptr("r")),
        _call("ascend_cross_core_set_flag", 2, "PIPE_MTE1", 0),
    ]))
    notify = _call("ascend_cross_core_set_flag", 2, "PIPE_MTE3", 1)
    if not notify_all:
        notify = tirx.IfThenElse(worker == 0, notify, None)
    # AIV1 performs no DMA payload, but remains a required mode-2 participant.
    vector = tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt([
        _call("ascend_cross_core_wait_flag", 2, "PIPE_MTE3", 0),
        tirx.IfThenElse(worker == 0, tirx.SeqStmt([
            _load(gm, payload), *_flags("MTE2_MTE3"), _store(payload, gm),
        ]), None),
        notify,
    ]))
    binding = tirx.IterVar(tvm.ir.Range(0, 2), worker, tirx.IterVar.ThreadIndex, "cthread")
    vector = tirx.AttrStmt(binding, "thread_extent", 2, vector)
    cube_last = tirx.SBlock([], [], [], "CUBE", tirx.SeqStmt([
        _call("ascend_cross_core_wait_flag", 2, "PIPE_MTE2", 1), load_l1(b),
    ]))
    before = _module([a, b, l0a, payload], [cube_first, vector, cube_last], gm,
                     target=tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    assert _may_alias(transform.InferBufferAliases()(before), a, b) == notify_all


@pytest.mark.parametrize("wait_all", [False, True], ids=["one-private-bank", "both-private-banks"])
def test_local_completion_is_required_in_each_accessing_aiv_bank(wait_all):
    gm, a, b = _buffers()
    worker = tirx.Var("worker", "int32")
    acquire = _call("ascend_wait_flag", "MTE3_MTE2", 1)
    if not wait_all:
        acquire = tirx.IfThenElse(worker == 0, acquire, None)
    vector = tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt([
        _load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm),
        _call("ascend_set_flag", "MTE3_MTE2", 1), acquire,
        _load(gm, b), *_flags("MTE2_MTE3", 2), _store(b, gm),
    ]))
    binding = tirx.IterVar(tvm.ir.Range(0, 2), worker, tirx.IterVar.ThreadIndex, "cthread")
    before = _module([a, b], [tirx.AttrStmt(binding, "thread_extent", 2, vector)], gm,
                     target=tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    # UB is private, so lane0's wait neither retires lane1's read nor makes its
    # later MTE2 overwrite safe. Both lanes must separately close their cycle.
    assert _may_alias(transform.InferBufferAliases()(before), a, b) == wait_all


def test_unassigned_ub_owner_is_not_a_distinct_private_bank():
    gm, a, b = _buffers()
    before = _module([a, b], [_load(gm, a),
                     tirx.SBlock([], [], [], "VECTOR", _load(gm, b))], gm)
    # Missing ownership cannot prove that these two UB allocations are
    # physically independent. There is no completion edge in this program.
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


def test_unknown_aiv_extent_retains_the_second_lane_may_access():
    gm, a, b = _buffers()
    worker, count = tirx.Var("worker", "int32"), tirx.Var("count", "int32")
    body = _reuse_body(gm, a, b, _flags("MTE3_MTE2"))
    body.append(tirx.IfThenElse(worker == 1, _store(a, gm), None))
    vector = tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt(body))
    binding = tirx.IterVar(tvm.ir.Range(0, count), worker,
                          tirx.IterVar.ThreadIndex, "cthread")
    before = _module([a, b], [tirx.AttrStmt(binding, "thread_extent", count, vector)],
                     gm, params=[count],
                     target=tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    # The initial lifetime is closed before b, but lane1 may read a again
    # afterwards. Projecting an unknown lane domain onto lane0 drops that read.
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("release_pipe", ["S_MTE2", "V_MTE2"])
def test_if_predicate_read_requires_scalar_completion(release_pipe):
    gm, a, b = _buffers()
    condition = tirx.IfThenElse(a[0] > 0, tirx.Evaluate(0), None)
    body = tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt([
        _load(gm, a), *_flags("MTE2_S"), condition,
        *_flags(release_pipe), _load(gm, b),
    ]))
    before = _module([a, b], [body], gm)
    assert _may_alias(transform.InferBufferAliases()(before), a, b) == (release_pipe == "S_MTE2")


def test_while_condition_read_stays_live_across_its_body():
    gm, a, b = _buffers()
    loop = tirx.While(a[0] > 0, tirx.SeqStmt([
        *_flags("S_MTE2"), _load(gm, b),
        *_flags("MTE2_MTE3"), _store(b, gm),
    ]))
    body = tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt([
        _load(gm, a), *_flags("MTE2_S"), loop,
    ]))
    before = _module([a, b], [body], gm)
    # The body may finish and evaluate a[0] again. A completion of the first
    # condition read does not retire all later reads in this unknown loop.
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("control", ["break", "continue"])
def test_unexecuted_loop_release_is_not_a_completion_witness(control):
    gm, a, b = _buffers()
    iteration = tirx.Var("iteration", "int32")
    transfer = tirx.break_loop() if control == "break" else tirx.continue_loop()
    loop = tirx.For(iteration, 0, 2, tirx.ForKind.SERIAL, tirx.SeqStmt([
        tirx.Evaluate(transfer), *_flags("MTE3_MTE2"),
    ]))
    before = _module([a, b], _reuse_body(gm, a, b, [loop]), gm)
    # Neither control path can issue the apparent release below it. This
    # IR-only regression does not execute an incomplete protocol on hardware.
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


def test_sync_id_operand_read_is_part_of_the_buffer_lifetime():
    gm, a, b = _buffers()
    body = _reuse_body(gm, a, b, _flags("MTE3_MTE2"))
    body.append(_call("ascend_set_flag", "V_S", tirx.Cast("int32", a[0])))
    before = _module([a, b], [tirx.SBlock([], [], [], "VECTOR", tirx.SeqStmt(body))], gm)
    # Although the earlier DMA read of a has completed, evaluating this
    # dynamic event ID reads a after b. Unknown IDs cannot hide that access.
    # Only alias inference runs: no unknown-ID protocol is sent to a device.
    assert not _may_alias(transform.InferBufferAliases()(before), a, b)


@pytest.mark.parametrize("dynamic", [False, True], ids=["finite-budget-overflow", "unknown-trip"])
def test_external_full_pipe_cut_retires_a_loop_may_live_summary(dynamic):
    gm, a, b = _buffers()
    iteration, trip = tirx.Var("iteration", "int32"), tirx.Var("trip", "int32")
    loop = tirx.For(iteration, 0, trip if dynamic else 10000,
                    tirx.ForKind.SERIAL, tirx.SeqStmt([
                        _load(gm, a), *_flags("MTE2_MTE3"), _store(a, gm),
                        *_flags("MTE3_MTE2"),
                    ]))
    before = _module([a, b], [loop, _call("ascend_pipe_barrier", "PIPE_ALL"),
                     _load(gm, b), *_flags("MTE2_MTE3"), _store(b, gm)],
                     gm, params=[trip] if dynamic else [])
    # Internal unknown/repeated flags cannot prove a cut, but the explicit
    # whole-loop barrier retires every may-access before b starts.
    assert _may_alias(transform.InferBufferAliases()(before), a, b)
