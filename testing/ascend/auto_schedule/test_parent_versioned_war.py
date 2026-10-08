"""A child's real slot release can cover its direct parent's storage WAR.

The finite oracle follows materialized slots and actual completion events.
It never substitutes a whole last-read/first-write completion edge for a
per-version release, including when the three-trip child changes slot phase.
"""

from collections import Counter, defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import make_ascend_l0a_2201_layout, make_ascend_major_k_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_nested_local_protocols import _Command, _prove_completions
from testing.ascend.auto_schedule.test_owner_result_completion import _name, _region, _trace, _value


_WAR = "MTE1_MTE2"


def _fixture(trips=3, *, kind="ordinary", load3d=False):
    e, owner, row, k = [tirx.Var(name, "int32") for name in ("epoch", "owner", "row_part", "chunk")]
    source = tirx.decl_buffer((3, 2, 3, trips, 32, 16), "float16", name="Input")
    p = tirx.decl_buffer((32, 16), "float16", name="p", scope="shared.l1")
    a = tirx.decl_buffer((32, 16), "float16", name="a", scope="shared.l0a")
    a2 = tirx.decl_buffer((32, 16), "float16", name="a2", scope="shared.l0a")
    p_view = tirx.decl_buffer(p.shape, p.dtype, name="p_view", data=p.data, scope="shared.l1")
    counter = tirx.decl_buffer((1,), "int32", name="p_clock", scope="local.var") if kind == "counter" else None
    # Keep the guarded negative on the ordinary unshifted path. A shifted
    # carried binary channel in this guarded nested domain has its own
    # existing target diagnostic, independent of parent WAR coverage.
    read_stage = 0 if trips == 1 or counter is not None or kind == "guarded-child" else 1
    read_rows = 16 if kind == "partial-reader" else 32
    reader = copy(p_view if kind == "view" else p, a,
                  src_shape=[read_rows, 16], dst_shape=[read_rows, 16])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    layouts = {p: make_ascend_major_k_layout(p), a: make_ascend_l0a_2201_layout(a)}
    setup = None
    if load3d:
        # Instruction selection precedes atomic scheduled task markers in the
        # real pipeline. Select this aligned, full-M recipe while it is still
        # ordinary typed copy IR, then schedule the actual selected endpoints.
        native = kernel(reader, buffers=[p, a], annotations={"layout_map": layouts})
        native["main"] = native["main"].with_attr("target", target)
        with target:
            native = transform.AscendSelectCubeLoads()(native)
        (selected,) = calls(native, "tl.tileop.ascend_cube_load3d")
        (config,) = calls(native, "tl.ascend_set_load3d_config")
        assert selected.args[0].args[0].buffer.same_as(p)
        assert selected.args[1].args[0].buffer.same_as(a)
        reader, setup = tirx.Evaluate(selected), unit(tirx.Evaluate(config), core=2)
    tasks = [unit(copy(source, p, src_indices=[e, owner, row, k, 0, 0],
                       src_shape=[1, 1, 1, 1, 32, 16]), core=2),
             unit(reader, core=2, stage=read_stage)]
    if kind == "extra-reader":
        tasks.append(unit(copy(p, a2), core=2, stage=read_stage))
    annotations = {"multi_buffer_eligible": [p.data], "num_stages": 2}
    if counter is not None:
        annotations.update({"tl.multi_buffer_counter_map": {p.data: counter},
                            "tl.storage_epoch_guard_map": {p.data: tirx.const(True, "bool")}})
        tasks.append(unit(tirx.BufferStore(counter, counter[0] + 1, [0]), core=2))
    if trips == 1:
        annotations["num_stages"] = 1
    chunks = unit(tirx.For(k, 0, trips, tirx.ForKind.SERIAL, seq(*tasks), annotations=annotations),
                  core=None, guard=e < 2 if kind == "guarded-child" else None)
    rows = unit(tirx.For(row, 0, 3, tirx.ForKind.SERIAL, chunks), core=None)
    owners = unit(tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, rows), core=None)
    body = unit(tirx.For(e, 0, 3, tirx.ForKind.SERIAL, owners), core=None)
    if setup is not None:
        body = seq(setup, body)
    if counter is not None:
        body = seq(unit(tirx.BufferStore(counter, 0, [0]), core=2), body)
    buffers = [p, a]
    if kind == "view":
        layouts[p_view] = make_ascend_major_k_layout(p_view)
    if kind == "extra-reader":
        buffers.append(a2)
        layouts[a2] = make_ascend_l0a_2201_layout(a2)
    if counter is not None:
        buffers.append(counter)
    metadata = {"layout_map": layouts, "tl.buffer_versions_map": {p.data: 2}}
    if counter is None:
        metadata["tl.buffer_version_mode"] = {p.data: "iteration"}
    mod = kernel(body, buffers=buffers, params=[source], annotations=metadata)
    mod["main"] = mod["main"].with_attr("target", target)
    with target:
        if counter is None:
            mod = transform.PrepareMultiBuffer()(mod)
        mod = transform.ResolveCore()(mod)
    return mod, {"p": p, "read_stage": read_stage, "trips": trips,
                 "readers": 2 if kind == "extra-reader" else 1,
                 "epochs": 2 if kind == "guarded-child" else 3,
                 "counter": counter is not None, "load3d": load3d}


def _lower(before):
    with before["main"].attrs["target"], tvm.transform.PassContext(
            config={"tl.disable_shared_memory_reuse": True}):
        inserted = transform.InsertSync()(before)
        physical = transform.MaterializeMultiBuffer()(inserted)
        lowered = transform.LowerScheduledTIR()(physical)
    return lowered


def _check_physical_lifetime(lowered, facts):
    trace, accesses, counts = [], [], Counter()
    reader_calls = 0
    for call, env, loops in _trace(lowered, core=2, lane=0):
        name, index = _name(call), len(trace)
        coordinates = {var.name: value for var, value, _ in loops}
        assert len(coordinates) == len(loops)
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
        elif name in ("tl.tileop.ascend_copy", "tl.tileop.ascend_cube_load3d"):
            src, src_min, src_shape = _region(call.args[0], env)
            dst, dst_min, dst_shape = _region(call.args[1], env)
            writing = dst.data.same_as(facts["p"].data)
            reading = src.data.same_as(facts["p"].data)
            assert writing != reading
            buf, origin, shape = (dst, dst_min, dst_shape) if writing else (src, src_min, src_shape)
            assert len(buf.shape) == 3 and tuple(map(int, buf.shape)) == (2, 32, 16)
            assert origin[1:] == (0, 0) and shape[0] == 1
            blocks = tuple((origin[0], row) for row in range(shape[1] // 16))
            trace.append(_Command("MTE2" if writing else "MTE1", "write" if writing else "read",
                                  writes=blocks if writing else (), reads=() if writing else blocks))
            if writing:
                logical = src_min[:4]
            else:
                logical = tuple(coordinates[key] for key in ("epoch", "owner", "row_part")) + (
                    coordinates["chunk"] - facts["read_stage"],)
                reader_calls += 1
            accesses.append((index, writing, blocks, logical))
        else:
            assert name == "tl.ascend_set_load3d_config", name
            trace.append(_Command("S", "scalar"))
    require = _prove_completions(trace)
    writers, readers = {}, defaultdict(set)
    payload = {}
    for index, writing, blocks, logical in accesses:
        for block in blocks:
            if writing:
                for reader in readers[block]:
                    require(3 * reader + 2, 3 * index + 1)
                if block in writers:
                    require(3 * writers[block] + 2, 3 * index + 1)
                readers[block].clear()
                writers[block], payload[block] = index, logical
            else:
                assert payload[block] == logical, (block, payload[block], logical)
                require(3 * writers[block] + 2, 3 * index + 1)
                readers[block].add(index)
    assert reader_calls == facts["epochs"] * 2 * 3 * facts["trips"] * facts["readers"]
    return counts


@pytest.mark.parametrize("trips,load3d", [(3, False), (4, False), (3, True)], ids=["odd", "even", "typed-load3d"])
def test_direct_parent_war_is_covered_by_the_real_flattened_slot_ring(trips, load3d):
    before, facts = _fixture(trips, load3d=load3d)
    counts = _check_physical_lifetime(_lower(before), facts)
    for epoch in range(3):
        for owner in range(2):
            # The middle row has only its W2 child-ring releases/acquires.
            # The wider owner and epoch protocols remain at their boundaries.
            assert counts[epoch, owner, 1, "wait"] == trips
            assert counts[epoch, owner, 1, "set"] == trips
            assert counts[epoch, owner, 0, "wait"] > trips
            assert counts[epoch, owner, 2, "set"] > trips


@pytest.mark.parametrize("trips,kind", [(1, "ordinary"), (3, "partial-reader"),
                                       (3, "extra-reader"), (3, "view"),
                                       (3, "counter")],
                         ids=["missing-full-ring", "partial", "extra-access", "distinct-view", "counter"])
def test_unqualified_parent_war_keeps_the_ordinary_protocol(trips, kind):
    before, facts = _fixture(trips, kind=kind)
    counts = _check_physical_lifetime(_lower(before), facts)
    for epoch in range(facts["epochs"]):
        for owner in range(2):
            # There is no parent-memory certificate for any of these inputs.
            for operation in ("wait", "set"):
                if trips == 1:
                    # No child full-ring release exists. This is the one
                    # ordinary parent handshake, not a deleted obligation.
                    assert counts[epoch, owner, 1, operation] == 1
                else:
                    assert counts[epoch, owner, 1, operation] > trips


def test_guarded_child_rejects_unsupported_shifted_nested_binary_protocol():
    before, _ = _fixture(kind="guarded-child")
    # The guarded child is outside the storage-WAR certificate. The ordinary
    # nested carried channel also lacks the supported unconditional domain;
    # preserve that target rejection rather than claiming a valid finite ring.
    with pytest.raises(tvm.error.InternalError, match=(
            "shifted carried events require an unconditional outermost lexical loop")):
        _lower(before)
