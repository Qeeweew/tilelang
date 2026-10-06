"""Forward handoff issue preferences retain the shared completion contracts.

The fixture uses diagnostic ordering weights and two fixed GM slots to isolate
issue selection. It makes no claim about calibrated device timing.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("z3")
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


WORKSPACES = "tl.ascend.compiler_workspace_storages"


def _mixed_handoffs(*, arch="dav-2201", manual=False):
    outbound, inbound = [tirx.decl_buffer((16, 16), "float32", name=name)
                         for name in ("outbound", "inbound")]
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    received = tirx.decl_buffer((16, 16), "float32", name="received", scope="shared.dyn")
    source = tirx.decl_buffer((16, 16), "float32", name="source", scope="shared.dyn")
    produced = tirx.decl_buffer((16, 16), "float32", name="produced", scope="shared.dyn")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    ptr = lambda buffer, effect: tirx.Call(
        "handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0, 0], 256, effect]
    )
    vector = tirx.Evaluate(tirx.Call(
        "void", tvm.ir.Op.get("tl.ascend_membase.exp"),
        [ptr(produced, 2), ptr(source, 1), tirx.IntImm("uint32", 256)],
    ))
    stage = 0 if manual else -1
    # Task indices at the FFI boundary:
    # 0 AIC publisher; 1 AIV incoming reader; 2 independent Vector producer;
    # 3 AIV publisher; 4 AIC incoming reader.
    body = seq(
        unit(copy(accum, outbound), core=2, stage=stage, cost=(1, 1)),
        unit(copy(outbound, received), core=1, stage=stage, cost=(1, 1)),
        unit(vector, core=1, stage=stage, cost=(2, 1)),
        unit(copy(produced, inbound), core=1, stage=stage, cost=(1, 1)),
        unit(copy(inbound, l1), core=2, stage=stage, cost=(1, 1)),
    )
    epoch = tirx.Var("epoch", "int32")
    loop = tirx.For(
        epoch, 0, 4, tirx.ForKind.SERIAL, body,
        annotations={"tl.ascend_auto_pipeline": True, "num_stages": 1},
    )
    before = kernel(
        unit(loop, core=None, stage=-1),
        buffers=[outbound, inbound, accum, received, source, produced, l1],
        annotations={
            "vector_count": 2,
            WORKSPACES: [outbound.data, inbound.data],
            "tl.buffer_versions_map": {outbound.data: 2, inbound.data: 2},
        },
    )
    before = tvm.IRModule({"main": before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": arch}))})
    return transform.AnnotateMultiBufferEligible()(before), outbound, inbound


def _observe_schedule(before):
    solve = z3_scheduler.z3_schedule_loop_python
    seen = []

    def observe(*args, **kwargs):
        seen.append(set(tuple(pair) for pair in kwargs["issue_priority_pairs"]))
        return solve(*args, **kwargs)

    # The existing FFI wrapper looks up this Python function dynamically, so
    # this observer also works after C++ has cached the registered FFI callback.
    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=observe):
        scheduled = transform.AutoSchedule()(before)
    assert len(seen) == 1
    return scheduled, seen[0]


def _leaf_units(mod):
    return [stmt for stmt, _ in statements(mod)
            if isinstance(stmt, tirx.AttrStmt) and stmt.attr_key == "tl.schedule_unit"
            and not nodes(stmt.body, tirx.For)]


def _copy_position(units, storage, *, destination):
    for index, task in enumerate(units):
        for call in calls(task.body, "tl.tileop.ascend_copy"):
            region = call.args[1 if destination else 0]
            if region.args[0].buffer.data.same_as(storage):
                return index
    raise AssertionError("Missing fixture copy")


def _collective_events(mod, *, operation, core, pipe):
    events = []
    for stmt, parents in statements(mod):
        if not isinstance(stmt, tirx.Evaluate) or not isinstance(stmt.value, tirx.Call):
            continue
        event = stmt.value
        if (not isinstance(event.op, tvm.ir.Op)
                or event.op.name != f"tl.ascend_cross_core_{operation}_flag"
                or event.args[1].value != pipe):
            continue
        # Only the recurrent loop sites, excluding entry tokens and exit drains.
        if not any(isinstance(parent, tirx.For) for parent in parents):
            continue
        owner = next(parent for parent in reversed(parents)
                     if isinstance(parent, tirx.AttrStmt)
                     and parent.attr_key == "tl.ascend_task")
        if int(owner.node["core_mask"]) == core:
            events.append(event)
    return events


def _assert_collective_pair(mod, producer_core, producer_pipe, consumer_core, consumer_pipe):
    sets = _collective_events(mod, operation="set", core=producer_core, pipe=producer_pipe)
    waits = _collective_events(mod, operation="wait", core=consumer_core, pipe=consumer_pipe)
    assert sets and waits
    analyzer = tvm.arith.Analyzer()
    assert any(analyzer.can_prove_equal(released.args[2], acquired.args[2])
               for released in sets for acquired in waits)


@pytest.mark.parametrize(
    "arch, manual, enabled",
    [("dav-2201", False, True), ("dav-2201", True, False), ("dav-3510", False, False)],
    ids=["2201-automatic", "manual-source-order", "3510-contract"],
)
def test_mixed_handoffs_prioritize_independent_work_and_keep_completion(arch, manual, enabled):
    before, outbound, inbound = _mixed_handoffs(arch=arch, manual=manual)
    scheduled, preferences = _observe_schedule(before)
    if enabled:
        assert preferences == {(0, 4), (2, 1), (3, 1)}
        # Slot-release WAR edges also cross cores, but their destination writers
        # 0 and 3 must not be mistaken for forward RAW consumers.
        assert not any(consumer in (0, 3) for _, consumer in preferences)
        units = _leaf_units(scheduled)
        assert _copy_position(units, outbound.data, destination=True) < _copy_position(
            units, inbound.data, destination=False
        )
        vector_position = next(index for index, task in enumerate(units)
                               if calls(task.body, "tl.ascend_membase.exp"))
        assert vector_position < _copy_position(units, outbound.data, destination=False)
    else:
        assert not preferences

    # Fixed diagnostic slot counts stay fixed through the preference solve.
    versions = scheduled["main"].body.block.annotations["tl.buffer_versions_map"]
    assert int(versions[outbound.data]) == int(versions[inbound.data]) == 2
    prepared = transform.PrepareMultiBuffer()(transform.AssignCore()(scheduled))
    synced = transform.InsertSync()(transform.ResolveCore()(prepared))
    events = calls(synced, "tl.ascend_cross_core_set_flag") + calls(
        synced, "tl.ascend_cross_core_wait_flag"
    )
    assert events and {int(event.args[0]) for event in events} == ({2} if arch == "dav-2201" else {4})
    # Outbound channel: data readiness and reverse read-completion release.
    _assert_collective_pair(synced, 2, "PIPE_FIX", 1, "PIPE_MTE2")
    _assert_collective_pair(synced, 1, "PIPE_MTE2", 2, "PIPE_FIX")
    # Inbound channel: data readiness and reverse read-completion release.
    _assert_collective_pair(synced, 1, "PIPE_MTE3", 2, "PIPE_MTE2")
    _assert_collective_pair(synced, 2, "PIPE_MTE2", 1, "PIPE_MTE3")


def test_unannotated_native_ancestors_share_core_facts_and_exclude_descendants():
    """Infer native work before AssignCore, without losing a reader's own chain."""
    make = lambda name, dtype="float32", scope="global": tirx.decl_buffer(
        (16, 16), dtype, name=name, scope=scope
    )
    outbound, inbound, returned = make("outbound"), make("inbound", "float16"), make("returned")
    a, b = [make(name, "float16", scope) for name, scope in (
        ("a", "shared.l0a"), ("b", "shared.l0b"))]
    accum, dependent_accum = [make(name, scope="shared.l0c")
                              for name in ("accum", "dependent_accum")]
    received, source, intermediate, final = [make(name, scope="shared.dyn")
                                           for name in ("received", "source", "intermediate", "final")]
    produced = make("produced", "float16", "shared.dyn")
    l1 = make("l1", "float16", "shared.l1")
    dependent_a = make("dependent_a", "float16", "shared.l0a")
    ptr = lambda buffer, effect: tirx.Call(
        "handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0, 0], 256, effect]
    )
    native = lambda operation, destination, source: tirx.Evaluate(tirx.Call(
        "void", tvm.ir.Op.get("tl.ascend_membase." + operation),
        [ptr(destination, 2), ptr(source, 1), tirx.IntImm("uint32", 256)],
    ))
    # 0 is independent AIC work producing handoff 1; 3/4 are independent AIV
    # ancestors of handoff 5. Reader 6 feeds 7→8→9, so 9 cannot be promoted
    # before 6. Native GEMMs, vector work and L1→L0 copy have no core metadata.
    body = seq(
        unit(tirx.Evaluate(T.gemm(a, b, accum, transpose_B=True, clear_accum=True)),
             core=0, stage=-1, cost=(2, 1)),
        unit(copy(accum, outbound), core=2, stage=-1, cost=(1, 1)),
        unit(copy(outbound, received), core=1, stage=-1, cost=(1, 1)),
        unit(native("exp", intermediate, source), core=0, stage=-1, cost=(2, 1)),
        unit(native("float2half_rn", produced, intermediate), core=0, stage=-1, cost=(1, 1)),
        unit(copy(produced, inbound), core=1, stage=-1, cost=(1, 1)),
        unit(copy(inbound, l1), core=2, stage=-1, cost=(1, 1)),
        unit(copy(l1, dependent_a), core=0, stage=-1, cost=(1, 1)),
        unit(tirx.Evaluate(T.gemm(dependent_a, b, dependent_accum,
                                transpose_B=True, clear_accum=True)),
             core=0, stage=-1, cost=(2, 1)),
        unit(copy(dependent_accum, returned), core=2, stage=-1, cost=(1, 1)),
        unit(copy(returned, final), core=1, stage=-1, cost=(1, 1)),
    )
    epoch = tirx.Var("epoch", "int32")
    loop = tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body,
                    annotations={"tl.ascend_auto_pipeline": True, "num_stages": 1})
    before = kernel(
        unit(loop, core=None, stage=-1),
        buffers=[outbound, inbound, returned, a, b, accum, dependent_accum,
                 received, source, intermediate, final, produced, l1, dependent_a],
        annotations={"vector_count": 2, WORKSPACES: [outbound.data, inbound.data, returned.data],
                     "tl.buffer_versions_map": {outbound.data: 2, inbound.data: 2, returned.data: 2}},
    )
    before = tvm.IRModule({"main": before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    scheduled, preferences = _observe_schedule(transform.AnnotateMultiBufferEligible()(before))
    assert {(0, 6), (1, 6), (3, 2), (4, 2), (5, 2)} <= preferences
    assert not any(producer in (7, 8, 9) and consumer == 6
                   for producer, consumer in preferences)
    assert {consumer for _, consumer in preferences} <= {2, 6, 10}
    # Auto quality inference supplies no placement metadata. AssignCore and
    # ResolveCore still own final placement before the existing sync pass.
    prepared = transform.PrepareMultiBuffer()(transform.AssignCore()(scheduled))
    synced = transform.InsertSync()(transform.ResolveCore()(prepared))
    _assert_collective_pair(synced, 2, "PIPE_FIX", 1, "PIPE_MTE2")
    _assert_collective_pair(synced, 1, "PIPE_MTE2", 2, "PIPE_FIX")
    _assert_collective_pair(synced, 1, "PIPE_MTE3", 2, "PIPE_MTE2")
    _assert_collective_pair(synced, 2, "PIPE_MTE2", 1, "PIPE_MTE3")
