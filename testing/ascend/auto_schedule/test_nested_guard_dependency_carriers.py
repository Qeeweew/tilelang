"""A child-loop dependency query must keep its enclosing epoch fixed.

The source-order scheduling callback exposes the common dependency analysis;
it does not solve or optimize the fixture. The finite DMA completion oracle
then checks the spill/reload protocol, including the real parent-loop RAW.
"""

from collections import defaultdict
from unittest.mock import patch

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_sync_binary_flags import (
    _completion_trace,
    _lower_protocol,
    _prove_binary_protocol,
    _prove_data_epochs,
)


def _fixture(pairs=2, subtiles=2, *, alternating=True):
    shape = (pairs, subtiles, 1, 64)
    source, state, output = [tirx.decl_buffer(shape, "float32", name=name)
                             for name in ("Input", "State", "Output")]
    ub = tirx.decl_buffer((1, 64), "float32", name="ub", scope="shared.dyn")
    epoch, sub = tirx.Var("epoch", "int32"), tirx.Var("sub", "int32")
    pair = epoch // 2 if alternating else epoch

    def load(buffer):
        return copy(buffer, ub, src_indices=[pair, sub, 0, 0], src_shape=[1, 1, 1, 64])

    def store(buffer):
        return copy(ub, buffer, dst_indices=[pair, sub, 0, 0], dst_shape=[1, 1, 1, 64])

    if alternating:
        odd, even = epoch % 2 == 1, epoch % 2 == 0
        # Keep the reload before the spill in lexical task order, as in the
        # ordinary output-workspace path. They cannot both run at one epoch.
        tasks = [unit(load(state), guard=odd, cost=(1, 1)),
                 unit(load(source), guard=even, cost=(1, 1)),
                 unit(store(output), guard=odd, cost=(1, 1)),
                 unit(store(state), guard=even, cost=(1, 1))]
    else:
        tasks = [unit(load(source), cost=(1, 1)), unit(store(output), cost=(1, 1))]
    child = tirx.For(sub, 0, subtiles, tirx.ForKind.SERIAL, seq(*tasks))
    outer = tirx.For(epoch, 0, pairs * (2 if alternating else 1), tirx.ForKind.SERIAL,
                     unit(child, core=None))
    mod = kernel(unit(outer, core=None), buffers=[ub], params=[source, state, output])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod


def _observe_dependencies(before):
    observed = []

    def source_order(*args, **kwargs):
        latencies = list(map(int, args[1]))
        observed.append((len(latencies), [tuple(map(int, dep)) for dep in args[4]]))
        starts, end = [], 0
        for latency, ii in zip(latencies, map(int, args[2])):
            starts.append(end)
            end += max(latency, ii, 1)
        return starts, [1] * len(args[8]), max(end, 1)

    with patch.object(z3_scheduler, "z3_schedule_loop_python", side_effect=source_order):
        transform.AutoSchedule()(before)
    return observed


def test_opposite_outer_guards_do_not_create_a_child_carried_dependency():
    observed = _observe_dependencies(_fixture())
    assert sorted(count for count, _ in observed) == [1, 4]
    inner = next(deps for count, deps in observed if count == 4)
    edges = {(producer, consumer, distance) for producer, consumer, distance, _ in inner}
    assert (0, 2, 0) in edges and (1, 3, 0) in edges
    assert not any(producer == 3 and consumer == 0 for producer, consumer, _, _ in inner), (
        "Even-epoch spill and odd-epoch reload are mutually exclusive while sub advances"
    )
    outer = next(deps for count, deps in observed if count == 1)
    assert any(producer == consumer == 0 and distance == 1 for producer, consumer, distance, _ in outer), (
        "The workspace spill must still complete before the next outer epoch reloads it"
    )


def test_child_ub_reuse_does_not_discharge_the_parent_iteration_boundary():
    before = _fixture(3, 2, alternating=False)
    observed = _observe_dependencies(before)
    assert sorted(count for count, _ in observed) == [1, 2]
    inner = next(deps for count, deps in observed if count == 2)
    assert any(producer == 1 and consumer == 0 and distance == 1
               for producer, consumer, distance, _ in inner)
    outer = next(deps for count, deps in observed if count == 1)
    assert any(producer == consumer == 0 and distance == 1 for producer, consumer, distance, _ in outer), (
        "A child-carried edge for the same leaf task pair cannot erase the parent-carried edge"
    )
    trace = _completion_trace(_lower_protocol(before))
    _prove_data_epochs(trace, _prove_binary_protocol(trace),
                       [(epoch, sub) for epoch in range(3) for sub in range(2)])


@pytest.mark.parametrize("pairs,subtiles", [(2, 2), (3, 3)])
def test_nested_spill_reload_keeps_binary_flags_and_parent_data_completion(pairs, subtiles):
    trace = _completion_trace(_lower_protocol(_fixture(pairs, subtiles)))
    precedes = _prove_binary_protocol(trace)
    producers, consumers = defaultdict(list), defaultdict(list)
    data = [(index, node) for index, node in enumerate(trace) if node.kind in ("produce", "consume")]
    for index, node in data:
        (producers if node.kind == "produce" else consumers)[node.epoch].append(index)
    expected = {(pair, sub) for pair in range(pairs) for sub in range(subtiles)}
    assert set(producers) == set(consumers) == expected
    for epoch in expected:
        assert len(producers[epoch]) == len(consumers[epoch]) == 2
        initial_load, reload = producers[epoch]
        spill, output = consumers[epoch]
        assert initial_load < spill < reload < output
        assert precedes(initial_load, spill)
        assert precedes(spill, reload), "The real cross-outer-epoch workspace RAW was lost"
        assert precedes(reload, output)
    assert len({node.slot for _, node in data}) == 1
    for (previous, left), (current, right) in zip(data, data[1:]):
        assert left.kind != right.kind
        assert precedes(previous, current), "UB reuse needs completion across both child and parent iterations"
