"""Explicit global source order uses ordinary storage/sync lowering.

The Stage-only API remains a partial schedule. Disabling the registered loop
scheduler's solver availability distinguishes direct completion from a hidden
performance search; all fixtures run only compiler passes on the CPU.
"""

import pytest
import tilelang

from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import allocated_buffer, copy, kernel, nodes, schedule, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value
from testing.ascend.auto_schedule.test_sync_binary_flags import (
    _Completion, _prove_binary_protocol, _prove_data_epochs,
)


def _program(*, preserve=True, fixed=True, versions=2, stage=0, nested=False,
             tile=64, num_stages=None):
    annotations = {"enable_offset": stage != 0}
    if num_stages is not None:
        annotations["num_stages"] = num_stages
    if preserve:
        annotations["tl.ascend_preserve_issue_order"] = True

    @T.prim_func
    def main(A: T.Tensor((4, tile), "float32"), B: T.Tensor((4, tile), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((tile,), "float32")
            if fixed:
                T.annotate_buffer_versions({temp: versions})
            if nested:
                for outer in T.serial(2):
                    for epoch in T.serial(4, annotations=annotations):
                        with T.Stage(0), T.Task(latency=4, ii=2):
                            T.copy(A[epoch, :], temp)
                        with T.Stage(stage), T.Task(latency=3, ii=1):
                            T.copy(temp, B[epoch, :])
            else:
                for epoch in T.serial(4, annotations=annotations):
                    with T.Stage(0), T.Task(latency=4, ii=2):
                        T.copy(A[epoch, :], temp)
                    with T.Stage(stage), T.Task(latency=3, ii=1):
                        T.copy(temp, B[epoch, :])
    return main


@pytest.mark.parametrize("stage", [0, 1])
@pytest.mark.parametrize("fixed", [False, True], ids=["completed-versions", "fixed-versions"])
def test_complete_source_order_does_not_enter_loop_solver(monkeypatch, stage, fixed):
    monkeypatch.setattr(z3_scheduler, "Z3_AVAILABLE", False)
    # The requested loop is the outermost actual control; ordinary physical
    # versions, synchronization, and fill/drain are still materialized.
    lowered = schedule(_program(stage=stage, fixed=fixed), through="LowerScheduledTIR")
    reads, writes, trace = [], [], []
    for call, env, _ in _trace(lowered, core=1, lane=0):
        name = _name(call)
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            source_pipe, destination_pipe = event.split("_")
            setting = name == "tl.ascend_set_flag"
            trace.append(_Completion(source_pipe if setting else destination_pipe,
                                     "set" if setting else "wait",
                                     (event, _value(call.args[1], env))))
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Completion(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name == "tl.tileop.ascend_copy":
            source, source_min, _ = _region(call.args[0], env)
            destination, destination_min, _ = _region(call.args[1], env)
            if source.name == "A":
                reads.append(source_min[0])
                trace.append(_Completion("MTE2", "produce", epoch=(source_min[0],),
                                         slot=(destination.data, *destination_min)))
            if destination.name == "B":
                writes.append(destination_min[0])
                trace.append(_Completion("MTE3", "consume", epoch=(destination_min[0],),
                                         slot=(source.data, *source_min)))
    assert reads == writes == list(range(4))
    _prove_data_epochs(trace, _prove_binary_protocol(trace), [(epoch,) for epoch in range(4)])
    assert not any("tl.ascend_preserve_issue_order" in loop.annotations
                   for loop in nodes(lowered, tirx.For))


@pytest.mark.parametrize("stage,expected", [(0, 1), (1, 2)])
def test_unknown_versions_are_the_least_dependency_completion(monkeypatch, stage, expected):
    monkeypatch.setattr(z3_scheduler, "Z3_AVAILABLE", False)
    after = schedule(_program(fixed=False, stage=stage), through="AutoSchedule")
    root = next(block for block in nodes(after, tirx.SBlock)
                if block.name_hint == "tilelang_root")
    temp = allocated_buffer(after, "temp")
    assert int(root.annotations["tl.buffer_versions_map"][temp.data]) == expected


def test_required_versions_above_stage_bound_are_diagnosed():
    with pytest.raises(Exception, match="source-order-version-bound"):
        schedule(_program(fixed=False, stage=1, num_stages=1), through="AutoSchedule")


def test_completed_versions_must_fit_memory_capacity():
    # One 128 KiB slot fits UB, while the required two-slot delayed ring does
    # not fit either supported target. Completion cannot silently spill it.
    with pytest.raises(Exception, match="source-order-memory-capacity"):
        schedule(_program(fixed=False, stage=1, tile=32768), through="AutoSchedule")


def test_illegal_explicit_fixed_version_count_is_not_increased():
    with pytest.raises(Exception, match="source-order-data-dependency"):
        schedule(_program(versions=1, stage=1), through="AutoSchedule")


def test_nested_explicit_order_is_diagnosed():
    with pytest.raises(Exception, match="only the outermost loop"):
        schedule(_program(nested=True), through="AutoSchedule")


def test_stage_only_keeps_partial_completion(monkeypatch):
    monkeypatch.setattr(z3_scheduler, "Z3_AVAILABLE", False)
    with pytest.raises(Exception, match="z3-solver package is required"):
        schedule(_program(preserve=False), through="AutoSchedule")


def test_illegal_fixed_source_order_is_diagnosed():
    # Stage defines logical access order: stage-zero reads the old value
    # before the stage-one write. This read-first recurrence has no automatic
    # ring owner. Complete source order must reject its unapplied fixed count;
    # it must neither reinterpret the Stage contract nor just warn and ignore it.
    @T.prim_func
    def invalid(A: T.Tensor((4, 64), "float32"), B: T.Tensor((4, 64), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({temp: 2})
            for epoch in T.serial(4, annotations={"enable_offset": True,
                                                  "tl.ascend_preserve_issue_order": True}):
                with T.Stage(1), T.Task(latency=4, ii=2):
                    T.copy(A[epoch, :], temp)
                with T.Stage(0), T.Task(latency=3, ii=1):
                    T.copy(temp, B[epoch, :])
    with pytest.raises(Exception, match="source-order-unowned-fixed-version"):
        schedule(invalid, through="AutoSchedule")


def test_child_selected_noneligible_versions_are_charged_at_parent():
    @T.prim_func
    def too_large(A: T.Tensor((2, 2, 16384), "float32"),
                  B: T.Tensor((2, 2, 16384), "float32"),
                  C: T.Tensor((2, 32768), "float32"),
                  D: T.Tensor((2, 32768), "float32")):
        with T.Kernel(1):
            child = T.alloc_shared((16384,), "float32")
            parent = T.alloc_shared((32768,), "float32")
            T.annotate_buffer_versions({child: 2, parent: 1})
            for outer in T.serial(2, annotations={"tl.ascend_preserve_issue_order": True}):
                with T.Stage(0):
                    for inner in T.Pipelined(2, num_stages=2):
                        with T.Task(latency=4, ii=2):
                            T.copy(A[outer, inner, :], child)
                        with T.Task(latency=3, ii=1):
                            T.copy(child, B[outer, inner, :])
                with T.Stage(0), T.Task(latency=4, ii=2):
                    T.copy(C[outer, :], parent)
                with T.Stage(0), T.Task(latency=3, ii=1):
                    T.copy(parent, D[outer, :])
    # Each owner fits alone. At the parent, the child's two 64 KiB slots plus
    # the 128 KiB parent tile exceed UB; charging only one child slot is wrong.
    with pytest.raises(Exception, match="source-order-memory-capacity"):
        schedule(too_large, through="AutoSchedule")


def test_generated_vector_scratch_completes_without_user_buffer_names(monkeypatch):
    @T.prim_func
    def expression(A: T.Tensor((4, 64), "float32"), B: T.Tensor((4, 64), "float16")):
        with T.Kernel(1):
            source = T.alloc_shared((64,), "float32")
            result = T.alloc_shared((64,), "float16")
            T.annotate_buffer_versions({source: 1, result: 1})
            for epoch in T.serial(4, annotations={"tl.ascend_preserve_issue_order": True}):
                with T.Stage(0):
                    T.copy(A[epoch, :], source)
                with T.Stage(0):
                    for i in T.Parallel(64):
                        result[i] = T.cast((source[i] + T.float32(1)) * source[i], "float16")
                with T.Stage(0):
                    T.copy(result, B[epoch, :])
    monkeypatch.setattr(z3_scheduler, "Z3_AVAILABLE", False)
    artifact = tilelang.lower(expression.with_attr("global_symbol", "main"),
                              target={"kind": "ascend", "arch": "dav-2201"},
                              enable_host_codegen=False, enable_device_compile=False)
    assert artifact.kernel_source


def test_manual_cost_summary_covers_issue_interval_at_last_task(monkeypatch):
    source = tirx.decl_buffer((64,), "float32", name="source")
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared.dyn")
    b = tirx.decl_buffer((64,), "float32", name="b", scope="shared.dyn")
    epoch = tirx.Var("epoch", "int32")
    # Explicit pass input isolates the summary contract. Public T.Task rejects
    # latency < II, but recovered/control timing must still bound issue span.
    owner = tirx.For(epoch, 0, 1, tirx.ForKind.SERIAL,
                     unit(copy(source, a), cost=(0, 5)),
                     annotations={"tl.ascend_preserve_issue_order": True})
    before = kernel(seq(unit(owner, core=None), unit(copy(source, b), cost=(1, 1))),
                    buffers=[a, b], params=[source])
    observed = []

    def capture(latencies, iis, *args, **kwargs):
        observed.append((latencies, iis))
        return [0, 5], [0, 1]

    # The registered FFI wrapper calls this function dynamically, even when
    # the C++ side has already cached that wrapper in an earlier test.
    monkeypatch.setattr(z3_scheduler, "z3_schedule_python", capture)
    transform.AutoSchedule()(before)
    assert observed == [([5, 1], [5, 1])]
