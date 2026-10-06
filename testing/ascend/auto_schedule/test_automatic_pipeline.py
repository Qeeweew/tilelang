"""Ordinary Ascend Pipelined intent selects stages and storage versions together."""

import pytest
import tilelang.language.common as common_language
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize(
    "num_stages, expected_bound, automatic",
    [(None, 4, True), (0, None, False), (2, 2, False)],
    ids=["automatic-default", "explicit-serial", "explicit-buffer-bound"],
)
def test_dialect_retains_pipeline_intent(num_stages, expected_bound, automatic):
    @T.prim_func
    def main(A: T.Tensor((4,), "int32")):
        with T.Kernel(1):
            for i in T.Pipelined(4, num_stages=num_stages):
                A[i] = i

    loop = next(loop for loop in nodes(main, tirx.For) if loop.kind == tirx.ForKind.SERIAL)
    bound = loop.annotations.get("num_stages")
    assert (int(bound) if bound is not None else None) == expected_bound
    assert bool(loop.annotations.get("tl.ascend_auto_pipeline", False)) == automatic


def test_common_pipeline_default_remains_serial():
    @common_language.prim_func
    def main(A: common_language.Tensor((4,), "int32")):
        for i in common_language.Pipelined(4):
            A[i] = i

    loop = nodes(main, tirx.For)[0]
    assert "num_stages" not in loop.annotations
    assert "tl.ascend_auto_pipeline" not in loop.annotations


def _ring(l1_bytes, *, automatic=True, max_stages=4, steps=4, manual_stages=None, loop_annotations=None):
    elements = l1_bytes // 4
    a = tirx.decl_buffer((steps, elements), "float32", name="A")
    l1 = tirx.decl_buffer((elements,), "float32", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((32,), "float32", name="l0", scope="shared.l0a")
    i = tirx.Var("i", "int32")
    stages = manual_stages or (-1, -1)
    body = seq(
        unit(copy(a, l1, src_indices=[i, 0], src_shape=[1, elements]), core=2, stage=stages[0], cost=(3, 1)),
        unit(copy(l1, l0, src_shape=[32]), core=2, stage=stages[1], cost=(3, 1)),
    )
    annotations = {"multi_buffer_eligible": [l1.data]}
    if automatic:
        annotations.update({"tl.ascend_auto_pipeline": True, "num_stages": max_stages})
    annotations.update(loop_annotations or {})
    loop = unit(tirx.For(i, 0, steps, tirx.ForKind.SERIAL, body, annotations=annotations), core=None, stage=-1)
    mod = kernel(loop, buffers=[l1, l0], params=[a])
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    return mod, l1


def _selected_plan(mod, ub):
    after = transform.AutoSchedule()(mod)
    tasks = [task for task in nodes(after, tirx.AttrStmt) if task.attr_key == "tl.schedule_unit" and calls(task.body, "tl.tileop.ascend_copy")]
    # Discard the enclosing loop unit; each task unit has no descendant For.
    tasks = [task for task in tasks if not nodes(task.body, tirx.For)]
    stages = [int(task.node["stage"]) for task in tasks]
    versions = int(after["main"].body.block.annotations["tl.buffer_versions_map"][ub.data])
    return after, stages, versions


def test_auto_pipeline_overlaps_with_available_storage():
    # Explicit diagnostic durations make completion longer than issue interval.
    # No hardware timing model is assumed by this scheduling regression.
    before, ub = _ring(8 * 1024)
    after, stages, versions = _selected_plan(before, ub)
    assert set(stages) == {0, 1}
    assert versions == 2
    assert all("tl.ascend_auto_pipeline" not in loop.annotations for loop in nodes(after, tirx.For))


@pytest.mark.parametrize("limited_by", ["capacity", "horizon", "single-iteration", "explicit-serial"])
def test_auto_pipeline_uses_one_version_when_required(limited_by):
    before, ub = _ring(
        300 * 1024 if limited_by == "capacity" else 8 * 1024,
        automatic=limited_by != "explicit-serial",
        max_stages=1 if limited_by == "horizon" else 4,
        steps=1 if limited_by == "single-iteration" else 4,
    )
    _, stages, versions = _selected_plan(before, ub)
    assert set(stages) == {0}
    assert versions == 1


def test_manual_stages_are_preserved_beyond_automatic_horizon():
    before, ub = _ring(8 * 1024, max_stages=1, manual_stages=(0, 1))
    _, stages, versions = _selected_plan(before, ub)
    assert set(stages) == {0, 1}
    assert versions == 2


@pytest.mark.parametrize("explicit_zero", [False, True], ids=["plain-serial", "explicit-pipeline-zero"])
def test_serial_dense_chain_keeps_original_epochs(explicit_zero):
    # Three asynchronous endpoints can fit a smaller modulo period only by
    # shifting the final store to another epoch. Plain Serial must keep stage
    # zero, including the durable intent from explicit Pipelined(num_stages=0).
    a, c = [tirx.decl_buffer((4, 32), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i = tirx.Var("epoch", "int32")
    ptr = lambda buffer, effect: tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0], 32, effect])
    vector = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.exp"), [ptr(y, 2), ptr(x, 1), tirx.IntImm("uint32", 32)]))
    body = seq(
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 32]), stage=-1, cost=(3, 1)),
        unit(vector, stage=-1, cost=(3, 1)),
        unit(copy(y, c, dst_indices=[i, 0], dst_shape=[1, 32]), stage=-1, cost=(3, 1)),
    )
    annotations = {"multi_buffer_eligible": [x.data, y.data]}
    if explicit_zero:
        annotations["tl.ascend_auto_pipeline"] = False
    loop = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations=annotations), core=None, stage=-1)
    before = kernel(loop, buffers=[x, y], params=[a, c])
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    after = transform.AutoSchedule()(before)
    stages = [int(attr.node["stage"]) for attr in nodes(after, tirx.AttrStmt)
              if attr.attr_key == "tl.schedule_unit" and not nodes(attr.body, tirx.For)]
    assert stages == [0, 0, 0]
    versions = after["main"].body.block.annotations["tl.buffer_versions_map"]
    assert int(versions[x.data]) == int(versions[y.data]) == 1


def test_solver_boundary_receives_serial_horizon_and_preserves_explicit_controls():
    import subprocess
    import sys

    # C++ retains the FFI callback after its first use. Isolate this boundary
    # check so the suite's previous solves cannot bypass its observer.
    script = r'''
import tvm_ffi
from testing.ascend.auto_schedule.test_automatic_pipeline import _ring, _selected_plan

original = tvm_ffi.get_global_func("tl.transform.z3_schedule_loop_python")
seen = []
expected = None
def observe(*args):
    assert int(args[14]) == expected, (int(args[14]), expected)
    seen.append(expected)
    return original(*args)
tvm_ffi.register_global_func("tl.transform.z3_schedule_loop_python", observe, override=True)
cases = [
    (dict(automatic=False), 1),
    (dict(automatic=False, loop_annotations={"tl.ascend_auto_pipeline": False}), 1),
    (dict(automatic=False, loop_annotations={"num_stages": 1}), 0),
    (dict(automatic=False, loop_annotations={"enable_offset": True}), 0),
    (dict(automatic=True, max_stages=4), 2),
    (dict(automatic=True, max_stages=1, manual_stages=(0, 1)), 0),
]
for settings, expected in cases:
    before, storage = _ring(8 * 1024, **settings)
    _selected_plan(before, storage)
assert seen == [1, 1, 0, 0, 2, 0], seen
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
