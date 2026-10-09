"""Each public scheduling query owns its budget without mutating the input.

Detailed byte, phase-reuse and workspace bounds live in their existing suites.
These regressions exercise distinct roots on the same pass object, including
the direct manual completion path that bypasses Python.
"""

import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_compiler_workspace_versions import _ring
from testing.ascend.auto_schedule.test_storage_footprint import _target_2201, _ub_storage


def _budget_fixture(style):
    # Root-only storage must be reserved even though no task accesses it.
    ring = _ub_storage("ring", 64 * 1024)
    epilogue = _ub_storage("epilogue", 96 * 1024)
    width = int(ring.shape[0])
    source = tirx.decl_buffer((4, width), "float32", name="source")
    output = tirx.decl_buffer((4, width), "float32", name="output")
    epoch = tirx.Var("epoch", "int32")
    stage = -1 if style == "automatic" else 0
    annotations = {"multi_buffer_eligible": [ring.data], "num_stages": 2}
    if style == "automatic":
        annotations["tl.ascend_auto_pipeline"] = True
    elif style == "complete":
        annotations["tl.ascend_preserve_issue_order"] = True
    body = seq(
        unit(copy(source, ring, src_indices=[epoch, 0], src_shape=[1, width]),
             stage=stage, cost=(1, 1)),
        unit(copy(ring, output, dst_indices=[epoch, 0], dst_shape=[1, width]),
             stage=stage, cost=(1, 1)),
    )
    owner = unit(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body,
                          annotations=annotations), core=None, stage=stage)
    module = _target_2201(kernel(owner, buffers=[ring, epilogue],
                                   params=[source, output]))
    return module, ring


def _observe_queries(monkeypatch):
    original = z3_scheduler.z3_schedule_loop_python
    queries = []

    def observe(*args, **kwargs):
        queries.append((
            tuple(map(int, args[8])),
            tuple(tuple(map(int, group)) for group in args[9]),
            tuple(map(int, kwargs["buffer_min_versions"])),
            bool(kwargs["manual_schedule"]),
        ))
        return original(*args, **kwargs)

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", observe)
    return queries


def _selected_versions(module, storage):
    return int(module["main"].body.block.annotations["tl.buffer_versions_map"][storage])


@pytest.mark.parametrize("style", ["automatic", "stage-only", "complete"])
def test_budget_queries_do_not_retain_another_roots_storage_ids_or_reservations(monkeypatch, style):
    before, ring = _budget_fixture(style)
    # B has a fixed compiler workspace instead of A's variable on-chip ring.
    # Its explicit count must survive B, without becoming A's next minimum.
    workspace, scratch = _ring(explicit_versions=2)
    workspace = transform.AnnotateMultiBufferEligible()(workspace)
    snapshots = [tvm.ir.save_json(module) for module in (before, workspace)]
    queries = _observe_queries(monkeypatch)
    schedule = transform.AutoSchedule()

    first = schedule(before)
    first_queries = list(queries)
    queries.clear()
    scheduled_workspace = schedule(workspace)
    workspace_queries = list(queries)
    queries.clear()
    second = schedule(before)

    assert queries == first_queries
    assert len(workspace_queries) == 1
    if style == "complete":
        assert first_queries == []
    else:
        assert len(first_queries) == 1
        assert first_queries[0] != workspace_queries[0]
        assert first_queries[0][3] == (style == "stage-only")
    # Only one 64 KiB ring slot fits beside the 96 KiB root allocation.
    assert _selected_versions(first, ring.data) == _selected_versions(second, ring.data) == 1
    assert _selected_versions(scheduled_workspace, scratch.data) == 2
    assert [tvm.ir.save_json(module) for module in (before, workspace)] == snapshots
