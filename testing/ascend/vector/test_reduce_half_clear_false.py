"""dav-2201 clear=False half max uses typed V combine, never scalar half max."""

import re

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region


TARGET = {"kind": "ascend", "arch": "dav-2201"}


def _program(count, retain_axis=False):
    pitch = (count + 15) // 16 * 16
    source = tirx.decl_buffer((3, pitch), "float16", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((3, 1) if retain_axis else (3,), "float16",
                                  name="destination", scope="shared.dyn")
    reduce = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(source, 1, [0, 0], [3, count]),
        region(destination, 3, [0, 0] if retain_axis else [0], [3, 1] if retain_axis else [3]),
        tirx.StringImm("max"), 1, tirx.const(False, "bool"),
    ])
    mod = kernel(tirx.Evaluate(reduce), buffers=[source, destination])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod, source, destination


@pytest.mark.parametrize("count", [17, 128, 130])
@pytest.mark.parametrize("retain_axis", [False, True])
def test_compact_half_max_snapshots_old_value_and_exposes_one_lane_vector_combine(count, retain_axis):
    before, source, destination = _program(count, retain_axis)
    reduced = transform.AscendLowerVectorReduce()(before)
    (maximum,) = nodes(reduced, tirx.Max)
    assert maximum.dtype == maximum.a.dtype == maximum.b.dtype == "float16"
    assert isinstance(maximum.a, tirx.BufferLoad) and isinstance(maximum.b, tirx.BufferLoad)
    assert not nodes(reduced, tirx.Cast)
    (previous,) = [bind for bind in nodes(reduced, tirx.Bind)
                  if isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.same_as(destination)]
    (snapshot,) = [store for store in nodes(reduced, tirx.BufferStore)
                  if store.buffer.same_as(maximum.b.buffer)]
    assert snapshot.value.same_as(previous.var)
    assert not maximum.b.buffer.data.same_as(destination.data)
    (combine_loop,) = [loop for loop in nodes(reduced, tirx.For) if loop.kind == tirx.ForKind.PARALLEL]
    assert int(combine_loop.extent) == 1
    (row_loop,) = [loop for loop in nodes(reduced, tirx.For) if loop.kind == tirx.ForKind.SERIAL]
    assert int(row_loop.extent) == 3
    (output,) = [store for store in nodes(reduced, tirx.BufferStore) if store.buffer.same_as(destination)]
    assert isinstance(output.value, tirx.Var) and output.value.dtype == "float16"
    # Odd compact output row1 is read/stored through S, not used as a V operand.
    analyzer = tvm.arith.Analyzer()
    assert int(analyzer.simplify(tirx.stmt_functor.substitute(
        output.indices[0], {row_loop.loop_var: tirx.IntImm(row_loop.loop_var.dtype, 1)}))) == 1
    scratch = reduced["main"].body.block.alloc_buffers[2:]
    assert len(scratch) >= 3
    assert all(buffer.scope() == "shared.dyn" and buffer.dtype == "float16"
               and int(buffer.shape[0]) * 2 % 32 == 0 for buffer in scratch)

    vector = transform.AscendLowerVectorTiles()(reduced)
    assert not nodes(vector, tirx.Max)  # No later scalar half Max can escape.
    reductions = calls(vector, "tl.ascend_membase.repeat_reduce_max_only_value")
    assert [int(call.args[-1]) for call in reductions] == ([130, 2] if count == 130 else [count])
    (combine,) = calls(vector, "tl.ascend_membase.max")
    assert int(combine.args[-1]) == 1
    assert all(int(argument.args[1]) == 1 for argument in combine.args[:3])
    assert [int(argument.args[2]) for argument in combine.args[:3]] == [2, 1, 1]
    assert all(argument.args[0].buffer.dtype == "float16" for argument in combine.args[:3])
    assert all(not argument.args[0].buffer.data.same_as(destination.data) for argument in combine.args[:3])
    assert combine.args[1].args[0].buffer.data.same_as(reductions[-1].args[0].args[0].buffer.data)
    with tvm.target.Target(TARGET):
        materialized = transform.MaterializeScheduleUnits()(vector)
    reduce_tasks = [task for task in nodes(materialized, tirx.AttrStmt)
                    if task.attr_key == "tl.ascend_task" and calls(task.body, "tl.ascend_membase.repeat_reduce_max_only_value")]
    max_tasks = [task for task in nodes(materialized, tirx.AttrStmt)
                 if task.attr_key == "tl.ascend_task" and calls(task.body, "tl.ascend_membase.max")]
    assert len(reduce_tasks) == len(reductions) and len(max_tasks) == 1
    assert all(not task.same_as(max_tasks[0]) for task in reduce_tasks)


@pytest.mark.parametrize("count", [17, 128, 130])
def test_ordinary_compact_half_clear_false_codegen_has_native_max_and_completion(count):
    pitch = (count + 15) // 16 * 16

    @T.prim_func
    def main(A: T.Tensor((3, pitch), "float16"), Initial: T.Tensor((3,), "float16"), O: T.Tensor((3,), "float16")):
        with T.Kernel(1):
            source = T.alloc_shared((3, pitch), "float16")
            output = T.alloc_shared((3,), "float16")
            T.copy(A, source)
            T.copy(Initial, output)
            T.reduce_max(source[:, 0:count], output, dim=1, clear=False)
            T.copy(output, O)

    artifact = tilelang.lower(main, target=TARGET, target_host="c",
                              enable_host_codegen=True, enable_device_compile=False)
    source = artifact.kernel_source
    assert artifact.host_mod
    assert "asc_repeat_reduce_max_only_value(" in source
    assert "asc_max(" in source
    assert not re.search(r"(?<![A-Za-z0-9_])max\s*\(", source), source
    assert "__builtin_cce_fmaxf16" not in source
    combine = source.index("asc_max(")
    reduction = source.rindex("asc_repeat_reduce_max_only_value(", 0, combine)
    assert source.rfind("asc_sync_pipe(PIPE_V)", 0, combine) > reduction, source
    (previous,) = re.finditer(r"half membase_reduce_previous_\d+ =", source)
    assert previous.start() < source.rfind("asc_sync_notify(PIPE_S, PIPE_V,", 0, combine), source
    (result,) = re.finditer(r"half membase_reduce_result_\d+ =", source)
    notify = source.index("asc_sync_notify(PIPE_V, PIPE_S,", combine)
    wait = source.index("asc_sync_wait(PIPE_V, PIPE_S,", combine)
    assert combine < notify < wait < result.start(), source
