"""Core ownership scopes preserve native instruction/task completion boundaries."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements, task_core_masks


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _core(kind, body, *, buffers=(), annotations=None, realized=False):
    block = tirx.SBlock([], [], [], kind, body, alloc_buffers=buffers, annotations=annotations or {})
    return tirx.SBlockRealize([], True, block) if realized else block


def _pointer(buffer, effect, count=17):
    return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], count, effect])


def _exp(source, destination, count=17):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.exp"), [_pointer(destination, 2, count), _pointer(source, 1, count), tirx.Cast("uint32", count)]))


def _sid_scope(sid, count, body):
    thread = tirx.IterVar(tvm.ir.Range(0, count), sid, tirx.IterVar.ThreadIndex, "cthread")
    return tirx.AttrStmt(thread, "thread_extent", count, body)


def _task(body, core=None):
    metadata = {} if core is None else {"core_mask": tirx.IntImm("int64", core)}
    return tirx.AttrStmt(metadata, "tl.ascend_task", 1, body)


def _materialize(mod):
    return transform.MaterializeScheduleUnits()(transform.NormalizeControlFlowForSchedule()(mod))


def _sync(mod):
    # Keep source order: these regressions isolate native classification and
    # synchronization, rather than depend on one particular solver schedule.
    def stage_zero(node):
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            return tirx.AttrStmt({"stage": tirx.IntImm("int32", 0)}, node.attr_key, node.value, node.body)

    func = mod["main"]
    body = tirx.stmt_functor.ir_transform(func.body, None, stage_zero)
    mod = tvm.IRModule({"main": func.with_body(body)})
    mod = transform.AssignCore()(mod)
    mod = transform.ResolveCore()(mod)
    return transform.InsertSync()(mod)


@pytest.mark.parametrize("realized", [False, True], ids=["direct", "realized"])
def test_cube_region_dma_uses_mte2_and_mte1_with_distinct_tasks(realized):
    source = tirx.decl_buffer((16, 16), "float16", name="source")
    l1 = tirx.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((16, 16), "float16", name="l0", scope="shared.l0a")
    before = _target(kernel(_core("CUBE", seq(copy(source, l1), copy(l1, l0)), realized=realized), buffers=[l1, l0], params=[source]))
    materialized = _materialize(before)
    assert len([attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]) == 2
    assert task_core_masks(materialized, "tl.tileop.ascend_copy") == {2}
    assert not any(block.name_hint == "CUBE" for block in nodes(materialized, tirx.SBlock))
    after = _sync(materialized)
    for operation in ("set", "wait"):
        assert [call.args[0].value for call in calls(after, "tl.ascend_" + operation + "_flag")] == ["MTE2_MTE1"]


def test_vector_region_dma_and_math_have_actual_pipe_completion_edges():
    source = tirx.decl_buffer((32,), "float32", name="source")
    output = tirx.decl_buffer((32,), "float32", name="output")
    a, b = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b")]
    before = _target(kernel(_core("VECTOR", seq(copy(source, a), _exp(a, b), copy(b, output))), buffers=[a, b], params=[source, output]))
    materialized = _materialize(before)
    tasks = [attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]
    assert len(tasks) == 3 and all(int(task.node["core_mask"]) == 1 for task in tasks)
    after = _sync(materialized)
    assert {call.args[0].value for call in calls(after, "tl.ascend_set_flag")} == {"MTE2_V", "V_MTE3"}
    assert {call.args[0].value for call in calls(after, "tl.ascend_wait_flag")} == {"MTE2_V", "V_MTE3"}


def test_cube_fix_handoff_is_not_misclassified_as_scalar_or_vector():
    accumulator = tirx.decl_buffer((16, 16), "float32", name="acc", scope="shared.l0c")
    workspace = tirx.decl_buffer((16, 16), "float32", name="workspace")
    private = tirx.decl_buffer((16, 16), "float32", name="private", scope="shared.dyn")
    body = seq(_core("CUBE", copy(accumulator, workspace)), _core("VECTOR", copy(workspace, private)))
    after = _sync(_materialize(_target(kernel(body, buffers=[accumulator, private], params=[workspace], annotations={"vector_count": 2}))))
    arrivals = calls(after, "tl.ascend_cross_core_set_flag")
    waits = calls(after, "tl.ascend_cross_core_wait_flag")
    assert arrivals and waits
    assert {call.args[1].value for call in arrivals} == {"PIPE_FIX"}
    assert {call.args[1].value for call in waits} == {"PIPE_MTE2"}
    assert all(int(call.args[0]) == 2 for call in arrivals + waits)


@pytest.mark.parametrize("count", [1, 2])
def test_cthread_scope_identity_extent_and_empty_completion_are_preserved(count):
    a, b, c = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b", "c")]
    sid = tirx.Var("sid", "int32")
    active = tirx.Select(sid == 0, 17, 0)
    body = _core("VECTOR", _sid_scope(sid, count, seq(_exp(a, b, active), _exp(b, c, active))), annotations={"vector_count": count})
    before = _target(kernel(body, buffers=[a, b, c]))
    materialized = _materialize(before)
    assert int(materialized["main"].body.block.annotations["vector_count"]) == count
    bindings = [attr for attr in nodes(materialized, tirx.AttrStmt) if attr.attr_key == "thread_extent"]
    assert len(bindings) == 2 and all(attr.node.var.same_as(sid) and int(attr.value) == count for attr in bindings)
    after = _sync(materialized)
    (barrier,) = [call for call in calls(after, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_V"]
    (_, parents) = next((stmt, parents) for stmt, parents in statements(after) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(barrier))
    assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)


def test_cthread_wrapped_scalar_bind_remains_visible_to_following_tasks():
    a, b = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b")]
    sid, active = tirx.Var("sid", "int32"), tirx.Var("active", "int32")
    body = _sid_scope(sid, 2, seq(tirx.Bind(active, tirx.Select(sid == 0, 17, 0)), _exp(a, b, active)))
    after = _sync(_materialize(_target(kernel(_core("VECTOR", body), buffers=[a, b]))))
    (bind,) = nodes(after, tirx.Bind)
    assert bind.var.same_as(active)
    (operation,) = calls(after, "tl.ascend_membase.exp")
    assert isinstance(operation.args[2], tirx.Cast) and operation.args[2].value.same_as(active)
    assert all(int(task.node["core_mask"]) == 1 for task in nodes(after, tirx.AttrStmt) if task.attr_key == "tl.ascend_task")


@pytest.mark.parametrize("frontend_layout", [False, True])
def test_core_allocations_and_layouts_move_to_same_kernel_root(frontend_layout):
    a, b = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b")]
    layout = Layout((32,), lambda i: i)
    keys = [buffer.data if frontend_layout else buffer for buffer in (a, b)]
    body = _core("VECTOR", _exp(a, b), buffers=[a, b], annotations={"layout_map": dict.fromkeys(keys, layout)})
    after = _materialize(_target(kernel(body)))
    root = after["main"].body.block
    assert len(root.alloc_buffers) == 2 and root.alloc_buffers[0].same_as(a) and root.alloc_buffers[1].same_as(b)
    assert all(root.annotations["layout_map"][key].is_equal(layout) for key in keys)


def test_core_region_loop_bounds_are_snapshotted_before_scheduling():
    extent = tirx.decl_buffer((1,), "int32", name="extent", scope="shared.dyn")
    a, b = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b")]
    row = tirx.Var("row", "int32")
    body = _core("VECTOR", tirx.For(row, 0, extent[0], tirx.ForKind.SERIAL, _exp(a, b)))
    materialized = _materialize(_target(kernel(body, buffers=[extent, a, b])))
    (bind,) = nodes(materialized, tirx.Bind)
    (loop,) = nodes(materialized, tirx.For)
    assert isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.same_as(extent)
    assert loop.extent.same_as(bind.var)
    assert all(int(task.node["core_mask"]) == 1 for task in nodes(materialized, tirx.AttrStmt) if task.attr_key == "tl.ascend_task")


@pytest.mark.parametrize("kind, expected", [("CUBE", 2), ("VECTOR", 1)])
def test_scalar_tasks_keep_explicit_core_ownership_when_assigning_resources(kind, expected):
    value = tirx.Var("scalar", "int32")
    materialized = _materialize(_target(kernel(_core(kind, tirx.Bind(value, 7)))))
    after = transform.AssignCore()(materialized)
    (task,) = [attr for attr in nodes(after, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]
    assert int(task.node["core_mask"]) == expected


def test_instruction_resources_must_match_declared_core_ownership():
    source = tirx.decl_buffer((16, 16), "float16", name="source")
    l1 = tirx.decl_buffer((16, 16), "float16", name="l1", scope="shared.l1")
    materialized = _materialize(_target(kernel(_core("VECTOR", copy(source, l1)), buffers=[l1], params=[source])))
    with pytest.raises(tvm.error.InternalError, match="resources conflict.*core ownership"):
        transform.AssignCore()(materialized)


@pytest.mark.parametrize("kind", ["SIMT_VF", "SIMD_VF"])
def test_3510_vf_retains_its_atomic_vector_operation(kind):
    source = tirx.decl_buffer((32,), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((32,), "float32", name="destination", scope="shared.dyn")
    vf = tirx.SBlock([], [], [], kind, seq(tirx.BufferStore(destination, source[0] + 1, [0]), tirx.BufferStore(destination, source[1] * 2, [1])))
    after = _materialize(_target(kernel(_core("VECTOR", vf), buffers=[source, destination]), "dav-3510"))
    tasks = [attr for attr in nodes(after, tirx.AttrStmt) if attr.attr_key == "tl.ascend_task"]
    assert len(tasks) == 1 and int(tasks[0].node["core_mask"]) == 1
    assert len([block for block in nodes(after, tirx.SBlock) if block.name_hint == kind]) == 1
    assert not calls(_sync(after), "tl.ascend_pipe_barrier")


@pytest.mark.parametrize("repeated", [False, True])
def test_explicit_task_does_not_hide_multiple_or_repeated_native_operations(repeated):
    a, b, c = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b", "c")]
    operations = tirx.For(tirx.Var("iteration", "int32"), 0, 2, tirx.ForKind.SERIAL, _exp(a, b)) if repeated else seq(_exp(a, b), _exp(b, c))
    with pytest.raises(tvm.error.InternalError, match="multiple or repeated native MemBase"):
        _materialize(_target(kernel(_core("VECTOR", _task(operations)), buffers=[a, b, c])))
    after = _materialize(_target(kernel(_core("VECTOR", _task(_exp(a, b))), buffers=[a, b])))
    assert len(calls(after, "tl.ascend_membase.exp")) == 1


@pytest.mark.parametrize("failure", ["core-mask", "vector-count", "vector-count-bool", "layout", "annotation"])
def test_conflicting_core_scope_facts_have_explicit_boundary_diagnostics(failure):
    a, b = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("a", "b")]
    body = _exp(a, b)
    annotations = {}
    root_annotations = {}
    message = ""
    if failure == "core-mask":
        body = _task(body, core=2)
        message = "core_mask conflicts"
    elif failure == "vector-count":
        annotations["vector_count"] = 1
        root_annotations["vector_count"] = 2
        message = "Conflicting Ascend core region vector_count"
    elif failure == "vector-count-bool":
        annotations["vector_count"] = True
        message = "vector_count must be the integer 1 or 2"
    elif failure == "layout":
        annotations["layout_map"] = {a: Layout((32,), lambda i: 31 - i)}
        root_annotations["layout_map"] = {a: Layout((32,), lambda i: i)}
        message = "Conflicting layout"
    else:
        annotations["unknown_core_policy"] = 1
        message = "Unsupported Ascend core ownership annotation"
    with pytest.raises(tvm.error.InternalError, match=message):
        _materialize(_target(kernel(_core("VECTOR", body, annotations=annotations), buffers=[a, b], annotations=root_annotations)))
