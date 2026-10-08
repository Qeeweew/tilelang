"""Native scheduling resources retain each concrete core's pipe namespace.

These CPU pass inputs use diagnostic costs. Observing the real AutoSchedule
FFI call checks resource construction without requiring a particular optimized
order or treating model start times as hardware completion.
"""

from inspect import signature
from unittest.mock import patch

import pytest

pytest.importorskip("z3")
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_blocking_issue_order import _assert_collective_pair


_VECTOR_SHIFT = 7
_MTE2, _M, _V, _S = 2, 8, 16, 64
_WORKSPACES = "tl.ascend.compiler_workspace_storages"


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _body(tasks, kind):
    if kind == "straight":
        return seq(*tasks)
    loop = tirx.For(tirx.Var("epoch", "int32"), 0, 4, tirx.ForKind.SERIAL,
                   seq(*tasks), annotations={"tl.ascend_auto_pipeline": True, "num_stages": 1})
    return unit(loop, core=None, stage=-1)


def _observe(mod, arch="dav-2201"):
    records = []

    def observer(kind, solve):
        def invoke(*args, **kwargs):
            inputs = dict(signature(solve).bind(*args, **kwargs).arguments)
            records.append((kind, inputs))
            return solve(*args, **kwargs)
        return invoke

    straight, loop = z3_scheduler.z3_schedule_python, z3_scheduler.z3_schedule_loop_python
    # Both registered FFI adapters resolve these Python functions dynamically,
    # even when the native pass has cached the registered adapter itself.
    with patch.object(z3_scheduler, "z3_schedule_python", side_effect=observer("straight", straight)), \
            patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=observer("loop", loop)):
        scheduled = transform.AutoSchedule()(_target(mod, arch))
    return scheduled, records


def _one(records, kind, count):
    matching = [inputs for call_kind, inputs in records
                if call_kind == kind and len(inputs["resource_flags"]) == count]
    assert len(matching) == 1
    return matching[0]


def _pairs(inputs, name):
    return {tuple(pair) for pair in inputs[name]}


@pytest.mark.parametrize("kind,placement,arch", [
    (kind, placement, "dav-2201")
    for kind in ("straight", "loop") for placement in ("inferred", "manual")
] + [("loop", "inferred", "dav-3510")])
def test_mte2_conflicts_are_within_one_concrete_core(kind, placement, arch):
    source = tirx.decl_buffer((16, 16), "float32", name="source")
    l1 = [tirx.decl_buffer((16, 16), "float32", name=f"l1_{i}", scope="shared.l1") for i in range(2)]
    ub = [tirx.decl_buffer((16, 16), "float32", name=f"ub_{i}", scope="shared.dyn") for i in range(2)]
    manual = placement == "manual"
    tasks = [unit(copy(source, destination), core=core if manual else 0,
                  stage=0 if manual else -1, cost=(1, 1))
             for core, destination in [(2, l1[0]), (2, l1[1]), (1, ub[0]), (1, ub[1])]]
    _, records = _observe(kernel(_body(tasks, kind), buffers=[*l1, *ub], params=[source]), arch)
    inputs = _one(records, kind, 4)
    assert inputs["resource_flags"] == [_MTE2, _MTE2, _MTE2 << _VECTOR_SHIFT, _MTE2 << _VECTOR_SHIFT]
    # HBM affinity is also disjoint across these two cores, while each core's
    # two real loads still contend. No resource exclusion may cross the banks.
    assert _pairs(inputs, "resource_deps") == {(0, 1), (2, 3)}
    assert _pairs(inputs, "pipe_order_deps") == ({(0, 1), (2, 3)} if manual else set())


@pytest.mark.parametrize("kind", ["straight", "loop"])
@pytest.mark.parametrize("manual", [False, True])
def test_scalar_core_namespaces_and_manual_pipe_order(kind, manual):
    tasks = [unit(tirx.Bind(tirx.Var(f"scalar_{i}", "int32"), i + 1), core=core,
                  stage=0 if manual else -1, cost=(1, 1))
             for i, core in enumerate((2, 2, 1, 1))]
    _, records = _observe(kernel(_body(tasks, kind)))
    inputs = _one(records, kind, 4)
    assert inputs["resource_flags"] == [_S, _S, _S << _VECTOR_SHIFT, _S << _VECTOR_SHIFT]
    # Automatic loops retain their existing policy of excluding Scalar from
    # resource_deps. Straight lists and manual per-pipe order still distinguish
    # the two actual Scalar units.
    assert _pairs(inputs, "resource_deps") == ({(0, 1), (2, 3)} if kind == "straight" else set())
    assert _pairs(inputs, "pipe_order_deps") == ({(0, 1), (2, 3)} if manual else set())


def test_unscoped_scalar_retains_both_possible_core_resources():
    tasks = [unit(tirx.Bind(tirx.Var(f"scalar_{i}", "int32"), i + 1), core=core,
                  stage=-1, cost=(1, 1)) for i, core in enumerate((0, 2, 1))]
    _, records = _observe(kernel(seq(*tasks)))
    inputs = _one(records, "straight", 3)
    assert inputs["resource_flags"] == [_S | (_S << _VECTOR_SHIFT), _S, _S << _VECTOR_SHIFT]
    assert _pairs(inputs, "resource_deps") == {(0, 1), (0, 2)}


@pytest.mark.parametrize("kind", ["straight", "loop"])
def test_compound_resources_union_leaf_core_pipe_pairs(kind):
    source = tirx.decl_buffer((16, 16), "float32", name="source")
    l1 = tirx.decl_buffer((16, 16), "float32", name="l1", scope="shared.l1")
    received, vector_source, vector_result = [
        tirx.decl_buffer((16, 16), "float32", name=name, scope="shared.dyn")
        for name in ("received", "vector_source", "vector_result")]
    a, b = [tirx.decl_buffer((16, 16), "float16", name=name, scope=scope)
            for name, scope in (("a", "shared.l0a"), ("b", "shared.l0b"))]
    c = tirx.decl_buffer((16, 16), "float32", name="c", scope="shared.l0c")

    def pointer(buffer, effect):
        return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0, 0], 256, effect])

    vector = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase.exp"),
                                   [pointer(vector_result, 2), pointer(vector_source, 1),
                                    tirx.IntImm("uint32", 256)]))
    first = [unit(copy(source, l1), core=0, stage=-1, cost=(1, 1)),
             unit(vector, core=0, stage=-1, cost=(1, 1))]
    second = [unit(tirx.Evaluate(T.gemm(a, b, c, transpose_B=True, clear_accum=True)),
                   core=0, stage=-1, cost=(1, 1)),
              unit(copy(source, received), core=0, stage=-1, cost=(1, 1))]
    owners = [unit(tirx.For(tirx.Var(f"owner_{i}", "int32"), 0, 2,
                           tirx.ForKind.SERIAL, seq(*tasks), annotations={"num_stages": 1}),
                   core=None, stage=-1) for i, tasks in enumerate((first, second))]
    _, records = _observe(kernel(_body(owners, kind),
                                buffers=[l1, received, vector_source, vector_result, a, b, c], params=[source]))
    # Internal owner solves precede the enclosing list's solve.
    inputs = [inputs for call_kind, inputs in records if call_kind == kind][-1]
    assert inputs["resource_flags"] == [_MTE2 | (_V << _VECTOR_SHIFT), _M | (_MTE2 << _VECTOR_SHIFT)]
    assert not _pairs(inputs, "resource_deps")
    # Crossing OR(core) with OR(pipe) would incorrectly place MTE2 on both
    # cores for each owner and manufacture an exclusion between these owners.


@pytest.mark.parametrize("kind", ["straight", "loop"])
def test_core_resource_split_keeps_typed_cross_core_raw_completion(kind):
    result = tirx.decl_buffer((16, 16), "float32", name="result")
    accum = tirx.decl_buffer((16, 16), "float32", name="accum", scope="shared.l0c")
    received = tirx.decl_buffer((16, 16), "float32", name="received", scope="shared.dyn")
    tasks = [unit(copy(accum, result), core=2, stage=-1, cost=(1, 1)),
             unit(copy(result, received), core=1, stage=-1, cost=(1, 1))]
    before = kernel(_body(tasks, kind), buffers=[result, accum, received], annotations={
        "vector_count": 2, _WORKSPACES: [result.data], "tl.buffer_versions_map": {result.data: 2},
    })
    scheduled, records = _observe(transform.AnnotateMultiBufferEligible()(before))
    inputs = _one(records, kind, 2)
    assert inputs["resource_flags"] == [32, _MTE2 << _VECTOR_SHIFT]
    assert any(tuple(dep[:2]) == (0, 1) and dep[-1] > 0 for dep in inputs["data_deps"])
    assert not _pairs(inputs, "resource_deps")
    if kind == "loop":
        prepared = transform.PrepareMultiBuffer()(transform.AssignCore()(scheduled))
        synced = transform.InsertSync()(transform.ResolveCore()(prepared))
        _assert_collective_pair(synced, 2, "PIPE_FIX", 1, "PIPE_MTE2")
        _assert_collective_pair(synced, 1, "PIPE_MTE2", 2, "PIPE_FIX")
        events = calls(synced, "tl.ascend_cross_core_set_flag") + calls(synced, "tl.ascend_cross_core_wait_flag")
        assert events and {int(event.args[0]) for event in events} == {2}
