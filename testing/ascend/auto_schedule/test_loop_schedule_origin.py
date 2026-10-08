"""Equivalent absolute scheduling clocks must retain one physical pipeline.

These CPU regressions inject feasible translated witnesses into the real FFI
boundary, then inspect the emitted stages, rings, and binary completion graph.
"""

from tilelang import tvm
from tilelang.ascend import transform
from tilelang.ascend.transform import z3_scheduler
from tvm import tirx

from testing.ascend._ir import kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_l0_operand_epoch_metadata import (
    _aic_events, _cube_only, _program,
)
from testing.ascend.auto_schedule.test_l0_operand_slot_pool import _check_physical_completions
from testing.ascend.auto_schedule.test_l0ab_phase_reuse import _lower


def _assert_same_physical_ir(first, second):
    # Layout maps retain both old Buffer keys and materialized ring aliases.
    # Even two identical-origin lowerings fail whole-Map free-key matching.
    # Compare every layout/Buffer fact explicitly, then compare the complete
    # executable IR, allocations and remaining metadata without those maps.
    def layouts(mod):
        result = {}
        for block in nodes(mod, tirx.SBlock):
            for buffer, layout in block.annotations.get("layout_map", {}).items():
                key = (block.name_hint, buffer.name, str(buffer.dtype), buffer.scope(),
                       tuple(map(str, buffer.shape)), tuple(map(str, buffer.strides)))
                assert key not in result, f"Ambiguous fixture layout key: {key}"
                result[key] = (buffer, layout)
        return result

    a, b = layouts(first), layouts(second)
    assert a.keys() == b.keys() and a
    for key in a:
        for lhs, rhs in zip(a[key], b[key]):
            tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)

    def without_layout_maps(mod):
        def erase(node):
            if isinstance(node, tirx.SBlock) and "layout_map" in node.annotations:
                annotations = {key: value for key, value in node.annotations.items()
                               if key != "layout_map"}
                return tirx.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint,
                                  node.body, node.init, node.alloc_buffers,
                                  node.match_buffers, annotations)
        return tvm.IRModule({gvar: func.with_body(tirx.stmt_functor.ir_transform(func.body, None, erase))
                             for gvar, func in mod.functions.items()})

    tvm.ir.assert_structural_equal(without_layout_maps(first), without_layout_maps(second),
                                  map_free_vars=True)


def test_translated_automatic_witnesses_keep_identical_physical_rings(monkeypatch):
    original = z3_scheduler.z3_schedule_loop_python
    shift = 0
    observed = []

    def translated(*args, **kwargs):
        if (args[1] == [2, 2, 3] and args[2] == [1, 1, 1]
                and not kwargs.get("manual_schedule")):
            # Two independent operand loads, followed by MAD. Both origins
            # satisfy the same hard constraints; +1 crosses an II boundary.
            observed.append(shift)
            return [2 + shift, 1 + shift, 4 + shift], [2, 2], 3
        return original(*args, **kwargs)

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", translated)
    program = _program(active=1)
    _, first = _lower(program)
    shift = 1
    _, second = _lower(program)
    assert observed == [0, 1]
    for snapshots in (first, second):
        _check_physical_completions(_cube_only(snapshots["tl.LowerScheduledTIR"]), 1)
    assert _aic_events(first["tl.LowerScheduledTIR"]) == _aic_events(second["tl.LowerScheduledTIR"])
    _assert_same_physical_ir(first["tl.LowerScheduledTIR"], second["tl.LowerScheduledTIR"])


def test_origin_cut_keeps_let_stage_order_and_original_cost(monkeypatch):
    output = tirx.decl_buffer((4,), "float32", name="Output")
    state = tirx.decl_buffer((1,), "float32", name="state", scope="shared.dyn")
    value = tirx.Var("value", "float32")
    epoch = tirx.Var("epoch", "int32")
    body = seq(
        unit(tirx.Bind(value, state[0]), core=3, stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(output, value, [epoch]), core=3, stage=-1, cost=(1, 1)),
        unit(tirx.BufferStore(state, tirx.FloatImm("float32", 1), [0]),
             core=3, stage=-1, cost=(1, 1)),
    )
    loop = tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body,
                   annotations={"num_stages": 2, "enable_offset": True,
                                "multi_buffer_eligible": [state.data]})
    before = kernel(seq(unit(loop, core=None, stage=-1),
                        unit(tirx.BufferStore(output, tirx.FloatImm("float32", 0), [0]),
                             core=3, stage=-1, cost=(1, 1))),
                    buffers=[state], params=[output],
                    annotations={"tl.buffer_versions_map": {state.data: 2}})

    observed = []
    costs = []

    def witness(*args, **kwargs):
        assert len(args[1]) == 3
        assert kwargs["stage_order_deps"] == [(1, 2)]
        # Consumer/writer stages are both one. Cutting at time two would
        # move only the writer to stage zero and violate the clone contract.
        observed.append(1)
        return [2, 8, 6], [], 5

    def root_cost(latencies, iis, *args, **kwargs):
        costs.append((latencies, iis))
        return [0, 24], [0, 1]

    monkeypatch.setattr(z3_scheduler, "z3_schedule_loop_python", witness)
    monkeypatch.setattr(z3_scheduler, "z3_schedule_python", root_cost)
    after = transform.AutoSchedule()(before)
    assert observed == [1]
    loop = next(loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "epoch")
    units = [unit for unit in nodes(loop.body, tirx.AttrStmt)
             if unit.attr_key == "tl.schedule_unit"]
    # A cloned Bind is prepended to stage one. The original consumer and
    # writer must also remain there after rejecting the proposed new cut.
    stores = [(int(unit.node["stage"]), store.buffer.name)
              for unit in units for store in nodes(unit.body, tirx.BufferStore)]
    assert (1, "state") in stores and (1, "Output") in stores
    assert costs == [([24, 1], [24, 1])]
