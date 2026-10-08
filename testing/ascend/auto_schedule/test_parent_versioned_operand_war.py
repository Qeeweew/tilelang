"""A retained native-MAD slot ring covers a closed serial ancestor's WAR.

The oracle uses materialized operand slots, actual mutable counter stores,
and real whole-pipeline event completions. Native unit readiness supplies no
edges. The two-operand case also exercises merged MTE1 endpoints: the release
can acquire before B's load while readiness is published after A's load.
"""

from collections import Counter
from itertools import product

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout, make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_WAR = "M_MTE1"
_COPY = "tl.tileop.ascend_copy"
_LOAD3D = "tl.tileop.ascend_cube_load3d"
_GEMM = "tl.tileop.gemm"


def _blocks(argument, env):
    """Enumerate every plane and 16x16 block in this finite typed region."""
    buffer, origin, shape = _region(argument, env)
    assert len(shape) == len(origin) == len(buffer.shape) and len(shape) >= 2
    assert all(start >= 0 and extent > 0 and start + extent <= _value(capacity, env)
               for start, extent, capacity in zip(origin, shape, buffer.shape))
    assert all(value % 16 == 0 for value in (*origin[-2:], *shape[-2:])), (origin, shape)
    # The row-level GM prefill contains all K planes, unlike one native
    # operand load. Preserve each plane's distinct payload and memory epoch.
    ranges = [range(start, start + extent) for start, extent in zip(origin[:-2], shape[:-2])]
    ranges.extend(range(start // 16, (start + extent) // 16)
                  for start, extent in zip(origin[-2:], shape[-2:]))
    return tuple((buffer.data, *coordinates) for coordinates in product(*ranges))


def _fixture(side="b", trips=3, *, kind="ordinary", load3d=False, outer_stage=0,
             versions=2, runtime_epochs=False):
    e, owner, row, k = [tirx.Var(name, "int32") for name in ("epoch", "owner", "row_part", "chunk")]
    epoch_count = tirx.Var("epoch_count", "int32") if runtime_epochs else None
    selected = ("a", "b") if side == "both" else (side,)
    shapes = {"a": (32, 16), "b": (16, 32)}
    inputs = {s: tirx.decl_buffer((3, 2, 3, trips, *shape), "float16", name="Input" + s.upper())
              for s, shape in shapes.items()}
    fixed = {s: tirx.decl_buffer(shape, "float16", name="Fixed" + s.upper()) for s, shape in shapes.items()}
    caches = {s: tirx.decl_buffer((trips, *shape) if s in selected else shape,
                                 "float16", name=s + "_l1", scope="shared.l1")
              for s, shape in shapes.items()}
    operands = {s: tirx.decl_buffer(shape, "float16", name=s, scope="shared.l0" + s)
                for s, shape in shapes.items()}
    view = tirx.decl_buffer(operands[side].shape, "float16", name=side + "_view",
                           data=operands[side].data, scope="shared.l0" + side) if kind == "view" else None
    c = tirx.decl_buffer((32, 32), "float32", name="accum", scope="shared.l0c")
    readers = 2 if kind == "extra-reader" else 1
    output = tirx.decl_buffer((3, 2, 3, trips, readers, 32, 32), "float32", name="Output")
    clock = tirx.decl_buffer((1,), "int32", name="operand_clock", scope="local.var") if kind == "counter" else None
    # Keep this negative on the already-supported ordinary counter path; it
    # tests certificate rejection rather than nested shifted-counter support.
    read_stage = 0 if clock is not None or versions == 1 else 1
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    layouts = {**{cache: make_ascend_major_k_layout(cache) for cache in caches.values()},
               operands["a"]: make_ascend_l0a_2201_layout(operands["a"]),
               operands["b"]: make_ascend_major_mn_layout(operands["b"]),
               c: make_ascend_l0c_layout(c)}
    if view is not None:
        layouts[view] = (make_ascend_l0a_2201_layout(view) if side == "a"
                         else make_ascend_major_mn_layout(view))

    loads, setup = {}, []
    for s in selected:
        loader = copy(caches[s], operands[s], src_indices=[k, 0, 0], src_shape=[1, *shapes[s]])
        if load3d:
            # Select before atomic scheduling, with the real leading-plane
            # loop domain available to the recipe's bounds proof.
            native = kernel(tirx.For(k, 0, trips, tirx.ForKind.SERIAL, loader),
                            buffers=[caches[s], operands[s]], annotations={"layout_map": layouts})
            native["main"] = native["main"].with_attr("target", target)
            with target:
                native = transform.AscendSelectCubeLoads()(native)
            (chosen,) = calls(native, _LOAD3D)
            (config,) = calls(native, "tl.ascend_set_load3d_config")
            assert chosen.args[0].args[0].buffer.same_as(caches[s])
            assert chosen.args[1].args[0].buffer.same_as(operands[s])
            loader = tirx.Evaluate(chosen)
            setup.append(unit(tirx.Evaluate(config), core=2))
        loads[s] = unit(loader, core=2)

    # The actual PV path loads B then A on the same MTE1 pipeline. Keep both
    # selected when requested, so no test assumes per-buffer event endpoints.
    tasks = [loads[s] for s in ("b", "a") if s in selected]
    m, n = (16, 32) if kind == "partial-reader" and side == "a" else (
        (32, 16) if kind == "partial-reader" else (32, 32))
    for reader in range(readers):
        a, b = operands["a"], operands["b"]
        if view is not None:
            a, b = (view, b) if side == "a" else (a, view)
        mad = T.gemm(a, b, c, transpose_B=False, clear_accum=True)
        if kind == "partial-reader":
            arguments = list(mad.args)
            arguments[0] = region(a, 1, shape=[m, 16])
            arguments[1] = region(b, 1, shape=[16, n])
            arguments[2] = region(c, 3, shape=[m, n])
            arguments[5], arguments[6] = tirx.IntImm("int32", m), tirx.IntImm("int32", n)
            mad = tirx.Call(mad.dtype, mad.op, arguments, annotations=mad.annotations)
        tasks.extend([
            unit(tirx.Evaluate(mad), core=2, stage=read_stage),
            unit(copy(c, output, src_shape=[m, n],
                      dst_indices=[e, owner, row, k, reader, 0, 0], dst_shape=[1, 1, 1, 1, 1, m, n],
                      annotations={"tl.ascend.root_fixpipe_config": tirx.IntImm("int32", 1)}),
                 core=2, stage=read_stage),
        ])
    annotations = {"multi_buffer_eligible": [operands[s].data for s in selected], "num_stages": 2}
    if clock is not None:
        annotations.update({"tl.multi_buffer_counter_map": {operands[side].data: clock},
                            "tl.storage_epoch_guard_map": {operands[side].data: tirx.const(True, "bool")}})
        tasks.append(unit(tirx.BufferStore(clock, clock[0] + 1, [0]), core=2))
    outer_guard = row < 2 if kind == "guarded-child" else None
    chunks = unit(tirx.For(k, 0, trips, tirx.ForKind.SERIAL, seq(*tasks), annotations=annotations),
                  core=None, stage=outer_stage, guard=outer_guard)
    fills = [unit(copy(inputs[s], caches[s], src_indices=[e, owner, row, 0, 0, 0],
                       src_shape=[1, 1, 1, trips, *shapes[s]]), core=2,
                  stage=outer_stage, guard=outer_guard) for s in selected]
    # A stage-zero sibling makes this a real fill/drain translation of the
    # whole child invocation, rather than an all-equal-stage no-op.
    stage_zero = [unit(tirx.Evaluate(0), core=2)] if outer_stage else []
    rows = unit(tirx.For(row, 0, 3, tirx.ForKind.SERIAL, seq(*stage_zero, *fills, chunks)), core=None)
    owners = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, rows), core=None)
    outer = unit(tirx.For(e, 0, tirx.Max(epoch_count, 0) if runtime_epochs else 3,
                         tirx.ForKind.SERIAL, owners), core=None)
    prologue = [unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_set_fixpipe_config"), [])), core=2),
                # This regression needs actual MAD whole-completion edges.
                # Automatic C unit-stream selection is tested independently.
                unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_set_mmad_direction"),
                                            [tirx.StringImm("m")])), core=2), *setup]
    for s in ("a", "b"):
        if s not in selected:
            prologue.extend([unit(copy(fixed[s], caches[s]), core=2),
                             unit(copy(caches[s], operands[s]), core=2)])
    if clock is not None:
        prologue.append(unit(tirx.BufferStore(clock, 0, [0]), core=2))
    buffers = [*caches.values(), *operands.values(), c] + ([clock] if clock is not None else [])
    metadata = {"layout_map": layouts, "tl.buffer_versions_map": {operands[s].data: versions for s in selected}}
    if clock is None:
        metadata["tl.buffer_version_mode"] = {operands[s].data: "iteration" for s in selected}
    mod = kernel(seq(*prologue, outer), buffers=buffers,
                 params=[*inputs.values(), *fixed.values(), output] + ([epoch_count] if runtime_epochs else []),
                 annotations=metadata)
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        if clock is None:
            mod = transform.PrepareMultiBuffer()(mod)
        mod = transform.ResolveCore()(mod)
    return mod, {"selected": selected, "operands": operands, "shapes": shapes,
                 "read_stage": read_stage, "outer_stage": outer_stage,
                 "trips": trips, "readers": readers, "load3d": load3d,
                 "versions": versions, "rows": 2 if kind == "guarded-child" else 3,
                 "epoch_count": epoch_count}


def _lower(before):
    with before["main"].attrs["target"], tvm.transform.PassContext(
            config={"tl.disable_shared_memory_reuse": True}):
        inserted = transform.InsertSync()(before)
        physical = transform.MaterializeMultiBuffer()(inserted)
        return transform.LowerScheduledTIR()(physical)


def _check_physical_lifetime(mod, facts):
    trace, payload, counts, fills = [], {}, Counter(), {}
    mad_count = export_count = 0
    configuration, native_loads = [], []
    for call, env, loops in _trace(mod, core=2, lane=0):
        name = _name(call)
        coordinates = {var.name: value for var, value, _ in loops}
        assert len(coordinates) == len(loops), "Fixture loop names must remain unique"
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event = call.args[0].value
            src, dst = event.split("_")
            setting = name == "tl.ascend_set_flag"
            flag = (event, _value(call.args[1], env))
            assert 0 <= flag[1] < 8
            trace.append(_Command(src if setting else dst, "set" if setting else "wait", flag))
            if event == _WAR and "chunk" in coordinates:
                counts[coordinates["epoch"], coordinates["owner"], coordinates["row_part"],
                       "set" if setting else "wait"] += 1
        elif name == "tl.ascend_pipe_barrier":
            trace.append(_Command(call.args[0].value.removeprefix("PIPE_"), "barrier"))
        elif name in (_COPY, _LOAD3D):
            src, src_min, _ = _region(call.args[0], env)
            dst, dst_min, _ = _region(call.args[1], env)
            reads, writes = _blocks(call.args[0], env), _blocks(call.args[1], env)
            assert len(reads) == len(writes)
            if src.scope() == "global":
                pipe = "MTE2"
                for block in reads:
                    payload.setdefault(block, (src.name, *block[1:]))
            elif dst.scope() in ("shared.l0a", "shared.l0b"):
                assert src.scope() == "shared.l1"
                pipe = "MTE1"
                for s in facts["selected"]:
                    if dst.data.same_as(facts["operands"][s].data):
                        if facts["versions"] > 1:
                            assert tuple(map(int, dst.shape)) == (facts["versions"], *facts["shapes"][s])
                            slot, matrix_origin = dst_min[0], dst_min[1:]
                        else:
                            assert tuple(map(int, dst.shape)) == facts["shapes"][s]
                            slot, matrix_origin = 0, dst_min
                        assert matrix_origin == (0, 0)
                        logical = (coordinates["epoch"], coordinates["owner"],
                                   coordinates["row_part"] - facts["outer_stage"], coordinates["chunk"])
                        assert (s, *logical) not in fills
                        fills[(s, *logical)] = slot
                if name == _LOAD3D:
                    native_loads.append(len(trace))
            else:
                assert src.scope() == "shared.l0c" and dst.scope() == "global"
                pipe = "FIX"
                for block in reads:
                    assert payload[block] == ("result", *dst_min[:4], *block[-2:])
                export_count += 1
            for read, write in zip(reads, writes):
                assert read in payload, (name, src.name, src_min, read)
                payload[write] = payload[read]
            trace.append(_Command(pipe, "copy", reads=reads, writes=writes))
        elif name == _GEMM:
            assert _value(call.args[9], env), "Independent results must really clear C"
            assert "unit_flag_ctrl" not in call.annotations, "Unit readiness must not replace whole MAD completion"
            logical = (coordinates["epoch"], coordinates["owner"],
                       coordinates["row_part"] - facts["outer_stage"]) + (
                coordinates["chunk"] - facts["read_stage"],)
            reads = []
            for s, argument in zip(("a", "b"), call.args[:2]):
                buffer, _, _ = _region(argument, env)
                assert buffer.data.same_as(facts["operands"][s].data)
                for block in _blocks(argument, env):
                    expected = (("Input" + s.upper(), *logical, *block[-2:]) if s in facts["selected"]
                                else ("Fixed" + s.upper(), *block[-2:]))
                    assert payload[block] == expected, (s, logical, block, payload[block], expected)
                    reads.append(block)
            writes = _blocks(call.args[2], env)
            for block in writes:
                payload[block] = ("result", *logical, *block[-2:])
            trace.append(_Command("M", "mad", reads=tuple(reads), writes=writes))
            mad_count += 1
        else:
            assert name in ("tl.ascend_set_fixpipe_config", "tl.ascend_set_mmad_direction",
                            "tl.ascend_set_load3d_config"), name
            if name == "tl.ascend_set_load3d_config":
                configuration.append(len(trace))
            trace.append(_Command("S", "scalar"))
    # This checks finite balance, consume-before-reset, deadlock freedom and
    # every physical RAW/WAR/WAW, using only actually emitted event edges.
    require = _prove_completions(trace)
    if facts["load3d"]:
        assert len(configuration) == 1 and native_loads
        for load in native_loads:
            require(3 * configuration[0] + 2, 3 * load + 1)
    else:
        assert not configuration and not native_loads
    epochs = facts.get("epochs", 3)
    assert mad_count == epochs * 2 * facts["rows"] * facts["trips"] * facts["readers"]
    assert export_count == mad_count
    assert len(fills) == len(facts["selected"]) * epochs * 2 * facts["rows"] * facts["trips"]
    if epochs and facts["trips"] % 2 and facts["versions"] == 2:
        for s in facts["selected"]:
            assert fills[s, 0, 0, 0, 0] != fills[s, 0, 0, 1, 0], "An odd child cannot reset its physical slot clock"
    return counts


@pytest.mark.parametrize("side,trips,load3d", [
    ("a", 3, False), ("a", 4, False), ("b", 3, False), ("b", 4, False),
    ("both", 3, False), ("both", 4, False), ("a", 3, True), ("b", 3, True),
], ids=["a-odd", "a-even", "b-odd", "b-even", "merged-odd", "merged-even", "a-load3d", "b-load3d"])
def test_operand_slot_release_covers_closed_serial_ancestor_war(side, trips, load3d):
    before, facts = _fixture(side, trips, load3d=load3d)
    counts = _check_physical_lifetime(_lower(before), facts)
    for epoch in range(3):
        for owner in range(2):
            for row in range(3):
                assert counts[epoch, owner, row, "wait"] == trips
                assert counts[epoch, owner, row, "set"] == trips


@pytest.mark.parametrize("epochs", [0, 1, 3], ids=["empty-root", "single-root", "reentered-root"])
@pytest.mark.parametrize("trips", [3, 4], ids=["odd-inner", "even-inner"])
def test_runtime_root_count_keeps_the_actual_ancestor_slot_protocol(epochs, trips):
    before, facts = _fixture("both", trips, outer_stage=1, runtime_epochs=True)
    physical = _lower(before)
    function = physical["main"]
    physical = tvm.IRModule({"main": function.with_body(tirx.stmt_functor.substitute(
        function.body, {facts["epoch_count"]: tirx.const(epochs, "int32")}))})
    counts = _check_physical_lifetime(physical, {**facts, "epochs": epochs})
    for epoch in range(epochs):
        for owner in range(2):
            for row in range(1, 4):
                assert counts[epoch, owner, row, "wait"] == trips
                assert counts[epoch, owner, row, "set"] == trips
    root_ring = Counter(_name(call) for call, _, loops in _trace(physical, core=2, lane=0)
                        if _name(call) in ("tl.ascend_set_flag", "tl.ascend_wait_flag")
                        and call.args[0].value == _WAR and not loops)
    # Even an empty runtime parent consumes its two initial credits at the
    # final drain; no synthetic parent completion is needed or counted.
    assert root_ring == {"tl.ascend_set_flag": 2, "tl.ascend_wait_flag": 2}


@pytest.mark.parametrize("side,kind", [
    ("a", "partial-reader"), ("b", "partial-reader"), ("b", "extra-reader"),
    ("a", "view"), ("b", "counter"),
], ids=["partial-a", "partial-b", "extra-mad", "distinct-view", "counter"])
def test_unqualified_operand_parent_keeps_its_ordinary_memory_protocol(side, kind):
    before, facts = _fixture(side, kind=kind)
    counts = _check_physical_lifetime(_lower(before), facts)
    for epoch in range(3):
        for owner in range(2):
            for operation in ("wait", "set"):
                assert counts[epoch, owner, 1, operation] > facts["trips"]
