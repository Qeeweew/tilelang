"""Fixpipe copies own their NZ-to-ND configuration inside one atomic task."""

from unittest.mock import patch

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx
from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _copies(arch):
    sources = [tirx.decl_buffer((16, 16), "float32", name=f"acc{i}", scope="shared.l0c") for i in range(2)]
    outputs = [tirx.decl_buffer((16, 16), "float32", name=f"out{i}") for i in range(2)]
    operations = [copy(source, output) for source, output in zip(sources, outputs)]
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    return sources, outputs, operations, target


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_fixpipe_copy_with_internal_register_setup_is_one_task(arch):
    sources, outputs, operations, target = _copies(arch)
    before = kernel(seq(*(unit(operation, core=2).body for operation in operations)), buffers=sources, params=outputs)
    before["main"] = before["main"].with_attr("target", target)
    after = transform.MaterializeScheduleUnits()(before)
    assert len(calls(after, "tl.tileop.ascend_copy")) == 2


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("per_core", [False, True], ids=["atomic", "per-core-candidates"])
def test_independent_fixpipe_copies_share_register_ordering(arch, per_core):
    sources, outputs, operations, target = _copies(arch)
    tasks = []
    for operation in operations:
        if per_core:
            metadata = {"core_mask": tirx.IntImm("int64", 2), "latency": tirx.IntImm("int64", 1), "ii": tirx.IntImm("int64", 1)}
            candidate = unit(operation, core=2).body
            grouped = tirx.AttrStmt(metadata, "tl.ascend_per_core_task", 1, candidate)
            tasks.append(unit(grouped, core=None, stage=-1))
        else:
            tasks.append(unit(operation, core=2, stage=-1, cost=(1, 1)))
    body = seq(*tasks)
    loop = tirx.For(tirx.Var("i", "int32"), 0, 3, tirx.ForKind.SERIAL, body, annotations={"num_stages": 2})
    before = kernel(unit(loop, core=None, stage=-1), buffers=sources, params=outputs)
    before["main"] = before["main"].with_attr("target", target)
    captured = []

    def schedule(*args, **kwargs):
        data_dependencies = kwargs.get("data_deps", args[4] if len(args) > 4 else None)
        captured.extend(tuple(int(value) for value in dependency) for dependency in data_dependencies)
        return [0, 1], [], 2

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=schedule):
        transform.AutoSchedule()(before)
    # The source and destination buffers are disjoint between the two copies.
    # These directional dependencies come from the shared SPR, not memory or
    # a choice of resource exclusion order by the solver.
    edges = {dependency[:3] for dependency in captured}
    assert {(0, 1, 0), (1, 0, 1)} <= edges
