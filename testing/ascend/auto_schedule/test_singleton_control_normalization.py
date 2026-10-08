"""Ordinary one-trip loops disappear before scheduling owns versions/events."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq


def _normalize(body, *buffers):
    return transform.NormalizeControlFlowForSchedule()(kernel(body, buffers=buffers))


def test_singleton_substitutes_nonzero_min_and_preserves_body_order():
    output = tirx.decl_buffer((16,), "int32", name="output", scope="shared.dyn")
    sub = tirx.Var("sub", "int32")
    body = seq(tirx.BufferStore(output, sub, [sub]),
               tirx.BufferStore(output, sub + 1, [sub + 1]))
    after = _normalize(tirx.For(sub, 3, 1, tirx.ForKind.SERIAL, body), output)
    assert not nodes(after, tirx.For)
    stores = nodes(after, tirx.BufferStore)
    assert len(stores) == 2
    analyzer = tvm.arith.Analyzer()
    for offset, store in enumerate(stores):
        assert int(analyzer.simplify(store.value)) == 3 + offset
        assert int(analyzer.simplify(store.indices[0])) == 3 + offset
    tvm.ir.assert_structural_equal(transform.NormalizeControlFlowForSchedule()(after), after)


def test_singleton_minimum_is_snapshotted_before_its_source_is_overwritten():
    state = tirx.decl_buffer((1,), "int32", name="state", scope="shared.dyn")
    output = tirx.decl_buffer((16,), "int32", name="output", scope="shared.dyn")
    sub = tirx.Var("sub", "int32")
    body = seq(tirx.BufferStore(state, 0, [0]), tirx.BufferStore(output, sub, [sub]))
    after = _normalize(tirx.For(sub, state[0], 1, tirx.ForKind.SERIAL, body), state, output)
    assert not nodes(after, tirx.For)
    (snapshot,) = nodes(after, tirx.Bind)
    (load,) = nodes(after, tirx.BufferLoad)
    assert load.buffer.same_as(state)
    assert snapshot.value.same_as(load)
    assert after["main"].body.block.body.seq[0].same_as(snapshot)
    output_store = nodes(after, tirx.BufferStore)[-1]
    assert output_store.value.same_as(snapshot.var)
    assert output_store.indices[0].same_as(snapshot.var)


@pytest.mark.parametrize("annotations", [{"num_stages": 1},
                                          {"tl.ascend_preserve_issue_order": True},
                                          {"enable_offset": True}, {"custom_loop_contract": 1}])
def test_annotated_singleton_retains_its_control_domain(annotations):
    sub = tirx.Var("sub", "int32")
    loop = tirx.For(sub, 0, 1, tirx.ForKind.SERIAL, tirx.Evaluate(sub), annotations=annotations)
    (retained,) = nodes(_normalize(loop), tirx.For)
    assert retained.loop_var.same_as(sub)
    assert set(retained.annotations) == set(annotations)


@pytest.mark.parametrize("name", ["tl.loop_break", "tirx.break_loop", "tirx.continue_loop"])
def test_singleton_retains_loop_control_target(name):
    sub = tirx.Var("sub", "int32")
    body = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(name), []))
    after = _normalize(tirx.For(sub, 0, 1, tirx.ForKind.SERIAL, body))
    assert len(nodes(after, tirx.For)) == 1
    assert len(calls(after, name)) == 1


def test_stage_outside_singleton_is_kept_but_stage_inside_retains_its_owner():
    sub = tirx.Var("sub", "int32")
    body = tirx.Evaluate(sub)
    loop = tirx.For(sub, 2, 1, tirx.ForKind.SERIAL, body)
    outside = _normalize(tirx.AttrStmt(tirx.IntImm("int32", 0), "tl.ascend_stage", 1, loop))
    assert not nodes(outside, tirx.For)
    (stage,) = nodes(outside, tirx.AttrStmt)
    assert stage.attr_key == "tl.ascend_stage" and int(stage.body.value) == 2
    inside = _normalize(tirx.For(sub, 2, 1, tirx.ForKind.SERIAL,
                                tirx.AttrStmt(tirx.IntImm("int32", 0), "tl.ascend_stage", 1, body)))
    assert len(nodes(inside, tirx.For)) == 1
    assert len(nodes(inside, tirx.AttrStmt)) == 1


@pytest.mark.parametrize("wrapper", ["tl.ascend_task", "tl.ascend_per_core_task",
                                     "tl.simdvf_scope", "tl.simtvf_scope"])
def test_singleton_inside_opaque_task_is_not_rewritten(wrapper):
    sub = tirx.Var("sub", "int32")
    loop = tirx.For(sub, 0, 1, tirx.ForKind.SERIAL, tirx.Evaluate(sub))
    attribute = tirx.AttrStmt({}, wrapper, 1, loop)
    (retained,) = nodes(_normalize(attribute), tirx.For)
    assert retained.same_as(loop)


def test_singleton_around_atomic_task_keeps_task_boundary_and_substitutes_data():
    sub = tirx.Var("sub", "int32")
    task = tirx.AttrStmt({}, "tl.ascend_task", 1, tirx.Evaluate(sub))
    after = _normalize(tirx.For(sub, 4, 1, tirx.ForKind.SERIAL, task))
    assert not nodes(after, tirx.For)
    (retained,) = nodes(after, tirx.AttrStmt)
    assert retained.attr_key == "tl.ascend_task" and int(retained.body.value) == 4


@pytest.mark.parametrize("extent,step", [(2, None), (1, 2)])
def test_non_singleton_or_nonunit_step_is_not_inlined(extent, step):
    sub = tirx.Var("sub", "int32")
    loop = tirx.For(sub, 0, extent, tirx.ForKind.SERIAL, tirx.Evaluate(sub), step=step)
    assert len(nodes(_normalize(loop), tirx.For)) == 1
