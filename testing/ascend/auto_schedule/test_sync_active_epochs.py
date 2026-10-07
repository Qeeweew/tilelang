"""A guarded root DMA clock may contain one exact contiguous active span.

The child loop makes its first/last synchronization sites distinct from the
root epoch guard. The existing finite DMA oracle checks completion and binary
flag reuse; data epochs retain the actual source coordinate, including a
nonzero origin or nonunit step.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_sync_binary_flags import (
    _completion_trace,
    _lower_protocol,
    _prove_binary_protocol,
)


def _active_dma_program(guard, *, minimum=0, extent=7, step=1, first=0,
                        snapshot=False, gapped_phase=False):
    # A symbolic extent is replayed only within the fixed backing capacity.
    capacity = minimum + (extent if isinstance(extent, int) else 7) + 1
    shape = (capacity, 2, 1, 64)
    state, output = [tirx.decl_buffer(shape, "float32", name=name)
                     for name in ("State", "Output")]
    storage = tirx.decl_buffer((1, 64), "float32", name="storage", scope="shared.dyn")
    seed = tirx.decl_buffer((1, 64), "float32", name="seed", scope="shared.dyn")
    output_cast = tirx.decl_buffer((1, 64), "float32", name="output_cast", scope="shared.dyn")
    epoch, sub = tirx.Var("epoch", "int32"), tirx.Var("sub", "int32")
    active = guard(epoch)
    prefix = []
    if snapshot:
        predicate = tirx.Var("active_snapshot", "bool")
        prefix.append(unit(tirx.Bind(predicate, active), core=3))
        active = predicate
    ordinal = (epoch - minimum) // step
    phase = ordinal // 2 if gapped_phase else ordinal - first
    # Retain the initial phase's actual GM coordinate through its spill/reload.
    row = minimum + (4 * (phase // 2) if gapped_phase else first + 2 * (phase // 2)) * step
    odd, even = tirx.Var("odd_phase", "bool"), tirx.Var("even_phase", "bool")

    def load(buffer, predicate):
        return unit(copy(buffer, storage, src_indices=[row, sub, 0, 0],
                         src_shape=[1, 1, 1, 64]), guard=predicate)

    def store(buffer, predicate, source=storage):
        return unit(copy(source, buffer, dst_indices=[row, sub, 0, 0],
                         dst_shape=[1, 1, 1, 64]), guard=predicate)

    def vector_math(name, source, destination):
        def pointer(buffer, effect):
            return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[0, 0], 64, effect])

        return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase." + name),
                                     [pointer(destination, 2), pointer(source, 1), tirx.const(64, "uint32")]))

    # As in the native V pass fixtures, seed is an abstract arithmetic input;
    # this protocol test does not execute or claim numerical V results.
    initialize = vector_math("copy_bits", seed, storage)
    update = vector_math("exp", storage, storage)
    cast = vector_math("copy_bits", storage, output_cast)
    # Match the output workspace path: V initializes an even phase, MTE3
    # spills it, and MTE2 reloads the following odd phase before V updates it.
    # Using MTE2 as the initial writer would create a different, classic
    # full-ring channel and would not exercise the active-span helper.
    child = tirx.For(sub, 0, 2, tirx.ForKind.SERIAL, seq(
        unit(tirx.Bind(odd, phase % 2 == 1), core=3),
        unit(tirx.Bind(even, phase % 2 == 0), core=3),
        unit(initialize, guard=even), store(state, even),
        load(state, odd), unit(update, guard=odd), unit(cast, guard=odd),
        store(output, odd, source=output_cast),
    ))
    owner = tirx.For(epoch, minimum, extent, tirx.ForKind.SERIAL,
                     seq(*prefix, unit(child, core=None, guard=active)),
                     step=None if step == 1 else tirx.const(step, "int32"))
    params = [state, output] + ([] if isinstance(extent, int) else [extent])
    mod = kernel(unit(owner, core=None), buffers=[storage, seed, output_cast], params=params)
    mod["main"] = mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod


def _oracle_controls(mod):
    """Inline pure fixture Binds and spell stepped loops as ordinal loops.

    The shared oracle intentionally has a smaller control interpreter. This
    adapter changes control syntax and omits opaque native V arithmetic. Every DMA
    and flag expression follows the same substitutions; no completion or flag
    operation is added. The proof below covers DMA data, not V arithmetic.
    """
    bindings = {}

    def collect(node):
        if isinstance(node, tirx.Bind):
            assert node.var.dtype == "bool", "Only fixture predicate snapshots are inlined"
            impure = []
            tirx.stmt_functor.post_order_visit(
                node.value, lambda value: impure.append(value)
                if isinstance(value, (tirx.BufferLoad, tirx.Call)) else None)
            assert not impure, "A replay snapshot must be a pure arithmetic predicate"
            bindings[node.var] = tirx.stmt_functor.substitute(node.value, bindings)

    func = mod["main"]
    tirx.stmt_functor.post_order_visit(func.body, collect)
    def opaque_math(node):
        if isinstance(node, tirx.Bind):
            return tirx.Evaluate(0)
        if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call):
            call = node.value
            if isinstance(call.op, tvm.ir.Op) and call.op.name in (
                    "tl.ascend_membase.copy_bits", "tl.ascend_membase.exp"):
                return tirx.Evaluate(0)
        return None

    body = tirx.stmt_functor.ir_transform(func.body, opaque_math, None)
    body = tirx.stmt_functor.substitute(body, bindings)
    analyzer = tvm.arith.Analyzer()

    def ordinal_loop(node):
        if not isinstance(node, tirx.For) or node.step is None:
            return None
        assert isinstance(node.step, tirx.IntImm) and int(node.step) > 0
        ordinal = tirx.Var(node.loop_var.name + "_ordinal", node.loop_var.dtype)
        count = analyzer.simplify((node.extent + node.step - 1) // node.step)
        body = tirx.stmt_functor.substitute(
            node.body, {node.loop_var: node.min + ordinal * node.step})
        return tirx.For(ordinal, 0, count, node.kind, body,
                        thread_binding=node.thread_binding, annotations=node.annotations)

    body = tirx.stmt_functor.ir_transform(body, None, ordinal_loop)
    return tvm.IRModule({"main": func.with_body(body)})


def _prove_spill_reload_epochs(trace, active, *, subtiles=2):
    precedes = _prove_binary_protocol(trace)
    producers, consumers = defaultdict(list), defaultdict(list)
    data = [(index, node) for index, node in enumerate(trace) if node.kind in ("produce", "consume")]
    for index, node in data:
        (producers if node.kind == "produce" else consumers)[node.epoch].append(index)
    active = list(active)
    expected = {(active[index], sub): index + 1 < len(active)
                for index in range(0, len(active), 2) for sub in range(subtiles)}
    assert set(consumers) == set(expected)
    assert set(producers) == {epoch for epoch, complete in expected.items() if complete}
    for epoch, complete in expected.items():
        assert len(consumers[epoch]) == 1 + complete
        assert len(producers[epoch]) == complete, "Each active spill/reload DMA phase must execute exactly once"
        chain = [consumers[epoch][0]]
        if complete:
            chain.extend([producers[epoch][0], consumers[epoch][1]])
        assert chain == sorted(chain)
        assert all(precedes(left, right) for left, right in zip(chain, chain[1:])), (
            "Workspace spill, reload and output require DMA completion order"
        )
        if complete:
            assert trace[chain[0]].slot == trace[chain[1]].slot
            assert trace[chain[2]].slot != trace[chain[1]].slot, "Output DMA must read the separate cast UB"
    state_slot = data[0][1].slot if data else None
    state_data = [(index, node) for index, node in data if node.slot == state_slot]
    for (previous, left), (current, right) in zip(state_data, state_data[1:]):
        assert precedes(previous, current), "DMA UB RAW/WAR must survive child and root iteration boundaries"
    if len(active) >= 2:
        # A classic full-ring protocol seeds every ID. The parent shifted
        # channel must expose additional real producer residues; this keeps
        # these fixtures from passing while bypassing active-span rebasing.
        # Kernel seeds form the initial SetFlag-only prefix. A V producer can
        # set a body flag before the first DMA, so "before first data" would
        # misclassify that real notification as an initial token.
        seed_end = next(index for index, node in enumerate(trace) if node.kind != "set")
        seeded, used = defaultdict(list), defaultdict(set)
        for node in trace[:seed_end]:
            if node.kind == "set":
                seeded[node.flag[0]].append(node.flag)
        for node in trace:
            if node.kind == "set":
                used[node.flag[0]].add(node.flag)
        assert all(len(flags) == len(set(flags)) for flags in seeded.values())
        assert any(flags and set(flags) < used[channel] for channel, flags in seeded.items()), (
            "Fixture must exercise a shifted partial-seed channel"
        )


@pytest.mark.parametrize("guard,first,active", [
    pytest.param(lambda epoch: epoch < 4, 0, range(4), id="prefix"),
    pytest.param(lambda epoch: epoch >= 1, 1, range(1, 7), id="suffix-odd-origin"),
    pytest.param(lambda epoch: tirx.all(epoch >= 1, epoch < 5), 1, range(1, 5), id="bounded-odd-origin"),
    pytest.param(lambda epoch: epoch >= 7, 7, [], id="empty-suffix"),
    pytest.param(lambda epoch: epoch < 0, 0, [], id="empty-prefix"),
])
@pytest.mark.parametrize("snapshot", [False, True], ids=["direct-guard", "body-bind-guard"])
def test_contiguous_root_span_preserves_nested_dma_epochs(guard, first, active, snapshot):
    lowered = _lower_protocol(_active_dma_program(guard, first=first, snapshot=snapshot))
    trace = _completion_trace(_oracle_controls(lowered))
    _prove_spill_reload_epochs(trace, active)


@pytest.mark.parametrize("minimum,extent,step,lower,upper,active", [
    pytest.param(3, 8, 1, 4, 10, [4, 5, 6, 7, 8, 9], id="nonzero-origin"),
    pytest.param(3, 14, 2, 5, 16, [5, 7, 9, 11, 13, 15], id="positive-step"),
    pytest.param(2, 19, 3, 3, 16, [5, 8, 11, 14], id="cut-points-between-iterations"),
])
def test_active_span_uses_ordinal_clock_and_actual_dma_coordinates(
        minimum, extent, step, lower, upper, active):
    before = _active_dma_program(lambda epoch: tirx.all(epoch >= lower, epoch < upper),
                                 minimum=minimum, extent=extent, step=step,
                                 first=(lower - minimum + step - 1) // step, snapshot=True)
    trace = _completion_trace(_oracle_controls(_lower_protocol(before)))
    _prove_spill_reload_epochs(trace, active)


@pytest.mark.parametrize("trips", [0, 1, 2, 3, 7])
def test_symbolic_extent_can_leave_the_contiguous_span_empty(trips):
    count = tirx.Var("trip_count", "int32")
    before = _active_dma_program(lambda epoch: epoch >= 1, extent=count, first=1, snapshot=True)
    trace = _completion_trace(_oracle_controls(_lower_protocol(before)), {count: trips})
    _prove_spill_reload_epochs(trace, range(1, trips))


@pytest.mark.parametrize("guard", [
    pytest.param(lambda epoch: epoch % 2 == 0, id="parity"),
    pytest.param(lambda epoch: tirx.any(epoch < 2, epoch >= 4), id="disjoint-intervals"),
    pytest.param(lambda epoch: tirx.all(epoch >= 1, epoch < 5, epoch != 3), id="interior-hole"),
])
def test_gapped_root_span_is_rejected_without_mutating_the_input(guard):
    before = transform.PrepareMultiBuffer()(_active_dma_program(guard, snapshot=True, gapped_phase=True))
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError, match="one exact contiguous active-epoch interval with static cut points"):
        transform.InsertSync()(before)
    assert tvm.ir.save_json(before) == snapshot
