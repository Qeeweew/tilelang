"""Physical M/N tiles stay inside their original guard, stage and core scope.

These CPU pass tests enumerate typed regions and accumulation effects. They do
not run the scheduler, CANN compilation, or a device completion protocol.
"""

import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements


def _region(buffer, origin, shape):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent)
                                     for start, extent in zip(origin, shape)])


def _decode(expression):
    assert expression.op.name == "tl.region"
    return expression.args[0].buffer, expression.args[0].indices, expression.args[2:]


def _value(expression, mapping):
    return int(tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expression, mapping)))


def _expanded(statement, mapping=None):
    mapping = mapping or {}
    if isinstance(statement, tirx.For):
        start, count = _value(statement.min, mapping), _value(statement.extent, mapping)
        for coordinate in range(start, start + count):
            yield from _expanded(statement.body, {**mapping, statement.loop_var:
                                                  tirx.const(coordinate, statement.loop_var.dtype)})
    elif isinstance(statement, tirx.IfThenElse):
        branch = statement.then_case if _value(statement.condition, mapping) else statement.else_case
        if branch is not None:
            yield from _expanded(branch, mapping)
    elif isinstance(statement, tirx.SeqStmt):
        for child in statement.seq:
            yield from _expanded(child, mapping)
    elif isinstance(statement, tirx.SBlockRealize):
        if _value(statement.predicate, mapping):
            yield from _expanded(statement.block, mapping)
    elif isinstance(statement, (tirx.SBlock, tirx.AttrStmt)):
        yield from _expanded(statement.body, mapping)
    elif isinstance(statement, tirx.Evaluate):
        if isinstance(statement.value, tirx.Call):
            yield statement.value, mapping
        else:
            assert _value(statement.value, mapping) == 0
    else:
        pytest.fail(f"Unexpected physical-tile statement: {statement}")


def _fixture(*, scope="nested", metadata="fixed", count=1, mode=None,
             m=128, n=1024, dtype="float16", kind="ordinary", pv_chunk=None,
             epochs=3, active_epochs=2, active_owners=2):
    a = tirx.decl_buffer((2, m, 64), dtype, name="q", scope="shared.l1")
    b = tirx.decl_buffer((n, 64), dtype, name="k", scope="shared.l1")
    c = tirx.decl_buffer((m, n), "float32", name="logical_c", scope="shared.l0c")
    # The leading axis is two distinct accumulators, not two compiler versions.
    pv = tirx.decl_buffer((2, m, 64), "float32", name="pv_c", scope="shared.l0c")
    ring = tirx.decl_buffer((2, 16, 64), dtype, name="other_ring", scope="shared.l1")
    out = tirx.decl_buffer((epochs, 2, m, n + 32), "float32", name="out")
    e, owner = tirx.Var("epoch", "int32"), tirx.Var("owner", "int32")
    enabled = tirx.Var("enabled", "bool")
    gemm = tirx.Evaluate(T.gemm(_region(a, [owner, 0, 0], [1, m, 64]), b, c,
                              transpose_B=True, clear_accum=kind != "clear-false"))
    exported = m - 16 if kind == "partial-export" else m
    export = copy(c, out, src_shape=[exported, n], dst_indices=[e, owner, 0, 16],
                  dst_shape=[1, 1, exported, n])
    body = seq(gemm, export)
    if kind == "split-guard":
        body = seq(tirx.IfThenElse(enabled, gemm, None), export)
    elif kind == "two-guards":
        body = seq(tirx.IfThenElse(enabled, gemm, None),
                   tirx.IfThenElse(enabled, export, None))
    elif kind == "multi-reader":
        body = seq(body, export)
    elif kind == "uniform-guard":
        body = tirx.IfThenElse(enabled, body, None)
    if active_owners != 2:
        body = tirx.IfThenElse(owner < active_owners, body, None)
    body = tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, body)
    if scope in ("cube", "nested", "core-realize", "core-region", "core-region-realize"):
        core = tirx.SBlock([], [], [], "ascend_core_region" if scope.startswith("core-region") else "CUBE", body)
        body = tirx.SBlockRealize([], True, core) if scope.endswith("realize") else core
    if scope in ("if", "nested"):
        body = tirx.IfThenElse(e < active_epochs, body, None)
    if scope in ("stage", "nested"):
        body = tirx.AttrStmt(tirx.IntImm("int64", 0), "tl.ascend_stage", 1, body)
    if scope in ("task", "per-core-task", "schedule-unit"):
        key = {"task": "tl.ascend_task", "per-core-task": "tl.ascend_per_core_task",
               "schedule-unit": "tl.schedule_unit"}[scope]
        body = tirx.AttrStmt({"core_mask": 1, "stage": 0}, key, 1, body)
    elif scope == "while":
        body = tirx.While(enabled, body)
    elif scope in ("opaque-block", "annotated-core", "predicated-core"):
        block = tirx.SBlock([], [], [], "OPAQUE" if scope == "opaque-block" else "CUBE", body,
                            annotations={"foreign": 1} if scope == "annotated-core" else {})
        body = tirx.SBlockRealize([], enabled, block) if scope == "predicated-core" else block
    loop_annotations = {"tl_pipeline_order": [0]} if scope == "pipeline-order" else (
        {"tl_pipeline_stage": [0]} if scope == "pipeline-stage" else {})
    body = tirx.For(e, 0, epochs, tirx.ForKind.SERIAL, body, annotations=loop_annotations)
    allocations = [a, b, c, pv, ring]
    layouts = {c.data: make_ascend_l0c_layout(c), pv.data: make_ascend_l0c_layout(pv)}
    annotations = {"layout_map": layouts, "tl.buffer_versions_map": {pv.data: 1},
                   "tl.manual_multi_buffer": {ring.data: 2},
                   "tl.buffer_version_mode": {ring.data: "counter"}}
    if metadata in ("fixed", "both"):
        annotations["tl.buffer_versions_map"][c.data] = count
    if metadata in ("manual", "both"):
        annotations["tl.manual_multi_buffer"][c.data] = count
    if mode is not None:
        annotations["tl.buffer_version_mode"][c.data] = mode
    if kind == "unknown-metadata":
        annotations["foreign_metadata"] = {c.data: [1]}
    elif kind == "escape":
        body = seq(body, tirx.Evaluate(tirx.call_extern("int32", "foreign", c.data)))
    elif kind == "alias":
        allocations.append(tirx.decl_buffer(c.shape, c.dtype, name="alias", data=c.data, scope=c.scope()))
    if pv_chunk is not None:
        p = tirx.decl_buffer((m, pv_chunk), dtype, name="p", scope="shared.l1")
        v = tirx.decl_buffer((2, pv_chunk, 64), dtype, name="v", scope="shared.l1")
        pv_out = tirx.decl_buffer((2, m, 64), "float32", name="pv_out")
        chunk = tirx.Var("chunk", "int32")
        update = tirx.Evaluate(T.gemm(p, _region(v, [0, 0, 0], [1, pv_chunk, 64]),
                                     _region(pv, [owner, 0, 0], [1, m, 64]),
                                     clear_accum=chunk == 0))
        updates = tirx.For(chunk, 0, n // pv_chunk, tirx.ForKind.SERIAL,
                           tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, update))
        exports = tirx.For(owner, 0, 2, tirx.ForKind.SERIAL,
                           copy(pv, pv_out, src_indices=[owner, 0, 0], src_shape=[1, m, 64],
                                dst_indices=[owner, 0, 0], dst_shape=[1, m, 64]))
        body = seq(body, updates, exports)
        allocations += [p, v]
    params = [out, enabled] + ([pv_out] if pv_chunk is not None else [])
    mod = kernel(body, buffers=allocations, params=params, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return mod, {"a": a, "b": b, "c": c, "pv": pv, "ring": ring, "out": out,
                 "e": e, "owner": owner, "enabled": enabled, "m": m, "n": n, "scope": scope,
                 "epochs": epochs, "active_epochs": active_epochs, "active_owners": active_owners}


def _slot_geometry(scratch):
    shape = tuple(map(int, scratch.shape))
    assert len(shape) in (2, 3)
    slots = shape[0] if len(shape) == 3 else 1
    assert slots in (1, 2)
    return slots, shape[-2:]


def _assert_slot_region(scratch, origin, shape):
    slots, tile = _slot_geometry(scratch)
    assert origin[-2:] == (0, 0) and shape[-2:] == tile
    if len(scratch.shape) == 3:
        assert len(origin) == len(shape) == 3 and shape[0] == 1
        assert 0 <= origin[0] < slots
    else:
        assert len(origin) == len(shape) == 2
    return origin


def _scratch_and_metadata(after, facts):
    root = after["main"].body.block
    l0c = [buffer for buffer in root.alloc_buffers if buffer.scope() == "shared.l0c"]
    (scratch,) = [buffer for buffer in l0c if not buffer.data.same_as(facts["pv"].data)]
    assert sum(math.prod(map(int, buffer.shape)) * 4 for buffer in l0c) <= 128 * 1024
    assert not scratch.data.same_as(facts["c"].data)
    for block in nodes(after, tirx.SBlock):
        for key in ("layout_map", "tl.buffer_versions_map", "tl.manual_multi_buffer", "tl.buffer_version_mode"):
            for identity in block.annotations.get(key, {}):
                storage = identity.data if isinstance(identity, tirx.Buffer) else identity
                assert not storage.same_as(facts["c"].data), key
    slots, _ = _slot_geometry(scratch)
    assert slots == 2
    assert int(root.annotations["tl.manual_multi_buffer"][scratch.data]) == 2
    assert scratch.data not in root.annotations["tl.buffer_versions_map"]
    assert int(root.annotations["tl.buffer_versions_map"][facts["pv"].data]) == 1
    assert int(root.annotations["tl.manual_multi_buffer"][facts["ring"].data]) == 2
    assert str(root.annotations["tl.buffer_version_mode"][facts["ring"].data]) == "counter"
    assert all(not load.buffer.data.same_as(facts["c"].data) for load in nodes(after, tirx.BufferLoad))
    return scratch


def _assert_coverage(after, facts, scratch, *, enabled=True):
    rectangles, loads, pending, exported_slots = {}, {}, {}, {}
    slots, tile = _slot_geometry(scratch)
    controls = {facts["enabled"]: tirx.const(enabled, "bool")}
    for call, mapping in _expanded(after["main"].body, controls):
        if call.op.name == "tl.tileop.gemm":
            a, _, _ = _decode(call.args[0])
            b, _, _ = _decode(call.args[1])
            c, origin, shape = _decode(call.args[2])
            assert c.data.same_as(scratch.data)
            slot = _assert_slot_region(scratch, tuple(_value(index, mapping) for index in origin),
                                       tuple(_value(extent, mapping) for extent in shape))
            assert tuple(_value(value, mapping) for value in call.args[5:7]) == tile
            assert int(call.args[0].args[1]) == int(call.args[1].args[1]) == 1
            clear = _value(call.args[9], mapping)
            assert not clear or slot not in pending, "An owner cleared a C slot before its export"
            pending.setdefault(slot, []).append((loads[a.data], loads[b.data], clear,
                                                 _value(call.args[7], mapping), int(call.args[2].args[1])))
        elif call.op.name == "tl.tileop.ascend_copy":
            src, origin, shape = _decode(call.args[0])
            dst, dst_origin, dst_shape = _decode(call.args[1])
            origin = tuple(_value(index, mapping) for index in origin)
            shape = tuple(_value(extent, mapping) for extent in shape)
            if src.scope() == "shared.l1":
                assert int(call.args[0].args[1]) == 1 and int(call.args[1].args[1]) == 2
                loads[dst.data] = (src, origin, shape)
            elif src.data.same_as(scratch.data):
                assert dst.data.same_as(facts["out"].data)
                assert int(call.args[0].args[1]) == 1 and int(call.args[1].args[1]) == 2
                slot = _assert_slot_region(scratch, origin, shape)
                destination = tuple(_value(index, mapping) for index in dst_origin)
                assert tuple(_value(extent, mapping) for extent in dst_shape) == (1, 1, *tile)
                assert slot in pending, "Export selected another owner or generation's C slot"
                epoch, owner, row, column = destination
                cursor = 0
                for index, (a, b, clear, k, mask) in enumerate(pending.pop(slot)):
                    assert a[0].data.same_as(facts["a"].data) and b[0].data.same_as(facts["b"].data)
                    assert a[1] == (owner, row, cursor) and a[2] == (1, tile[0], k)
                    assert b[1] == (column - 16, cursor) and b[2] == (tile[1], k)
                    assert clear == int(index == 0)
                    assert mask in (2, 3) and (clear or mask == 3)
                    cursor += k
                assert cursor == 64
                rectangles.setdefault((epoch, owner), []).append((row, column, *tile))
                exported_slots.setdefault((epoch, owner), []).append(slot[0] if slots == 2 else 0)
    assert not pending
    active = min(facts["epochs"], facts["active_epochs"]) if facts["scope"] in ("if", "nested") else facts["epochs"]
    if not enabled:
        active = 0
    assert set(rectangles) == {(epoch, owner) for epoch in range(active) for owner in range(facts["active_owners"])}
    for coordinate, tiles in rectangles.items():
        assert sum(rows * columns for _, _, rows, columns in tiles) == facts["m"] * facts["n"]
        for row in range(facts["m"]):
            intervals = sorted((column, column + columns) for start, column, rows, columns in tiles
                               if start <= row < start + rows)
            assert intervals[0][0] == 16 and intervals[-1][1] == 16 + facts["n"]
            assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
        # Each lexical owner/epoch invocation starts its own physical ordinal.
        # An odd tile count must not shift the next owner's generation index.
        assert exported_slots[coordinate] == [ordinal % slots for ordinal in range(len(tiles))]
    return exported_slots


@pytest.mark.parametrize("scope", ["if", "stage", "cube", "core-realize", "core-region", "core-region-realize", "nested"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_adjacent_pair_expands_inside_its_original_scope(scope, dtype):
    before, facts = _fixture(scope=scope, dtype=dtype)
    after = transform.AscendLowerL1Gemm()(before)
    scratch = _scratch_and_metadata(after, facts)
    _assert_coverage(after, facts, scratch)
    for call in calls(after, "tl.tileop.gemm"):
        (parents,) = [parents for statement, parents in statements(after)
                      if isinstance(statement, tirx.Evaluate) and statement.value.same_as(call)]
        if scope in ("if", "nested"):
            (guard,) = [parent for parent in parents if isinstance(parent, tirx.IfThenElse)]
            tvm.ir.assert_structural_equal(guard.condition, facts["e"] < 2)
        if scope in ("stage", "nested"):
            (stage,) = [parent for parent in parents if isinstance(parent, tirx.AttrStmt)
                        and parent.attr_key == "tl.ascend_stage"]
            assert int(stage.node) == 0 and int(stage.value) == 1
        if scope in ("cube", "core-realize", "core-region", "core-region-realize", "nested"):
            name = "ascend_core_region" if scope.startswith("core-region") else "CUBE"
            assert any(isinstance(parent, tirx.SBlock) and parent.name_hint == name for parent in parents)


def test_nondefault_geometry_preserves_complete_owner_coverage():
    before, facts = _fixture(m=96, n=512)
    after = transform.AscendLowerL1Gemm()(before)
    _assert_coverage(after, facts, _scratch_and_metadata(after, facts))


@pytest.mark.parametrize("m,n,parity", [(16, 2064, 1), (32, 2048, 0)], ids=["odd", "even"])
def test_slot_ordinal_restarts_for_each_guarded_owner_and_epoch(m, n, parity):
    # The number of aligned 16x16 blocks is respectively odd / a power of
    # two. Every legal oversized physical tiling retains the chosen parity,
    # without constraining the tile selector's M/N tradeoff.
    before, facts = _fixture(m=m, n=n)
    after = transform.AscendLowerL1Gemm()(before)
    exports = _assert_coverage(after, facts, _scratch_and_metadata(after, facts))
    assert len(exports) == 4
    assert all(len(sequence) > 1 and len(sequence) % 2 == parity for sequence in exports.values())


def test_uniform_disabled_pair_does_not_publish_a_physical_slot():
    before, facts = _fixture(kind="uniform-guard", m=16, n=2064)
    after = transform.AscendLowerL1Gemm()(before)
    scratch = _scratch_and_metadata(after, facts)
    assert not _assert_coverage(after, facts, scratch, enabled=False)
    _assert_coverage(after, facts, scratch, enabled=True)


def test_empty_owner_keeps_active_owners_slot_clock_and_coverage():
    before, facts = _fixture(active_owners=1, m=16, n=2064)
    after = transform.AscendLowerL1Gemm()(before)
    exports = _assert_coverage(after, facts, _scratch_and_metadata(after, facts))
    assert set(exports) == {(0, 0), (1, 0)}


@pytest.mark.parametrize("metadata", ["none", "fixed", "manual", "both"])
def test_single_version_identity_metadata_is_removed_with_the_logical_accumulator(metadata):
    before, facts = _fixture(metadata=metadata)
    after = transform.AscendLowerL1Gemm()(before)
    _assert_coverage(after, facts, _scratch_and_metadata(after, facts))


@pytest.mark.parametrize("metadata", ["fixed", "manual"])
@pytest.mark.parametrize("count", [2, 3])
def test_more_than_one_accumulator_version_is_not_silently_discarded(metadata, count):
    before, _ = _fixture(metadata=metadata, count=count)
    with pytest.raises(ValueError, match="one fixed accumulator version"):
        transform.AscendLowerL1Gemm()(before)


@pytest.mark.parametrize("mode", ["auto", "iteration", "counter"])
def test_version_modes_are_not_reinterpreted_as_fixed_scratch(mode):
    before, _ = _fixture(mode=mode)
    with pytest.raises(ValueError, match="one fixed accumulator version"):
        transform.AscendLowerL1Gemm()(before)


@pytest.mark.parametrize("scope", ["task", "per-core-task", "schedule-unit", "while", "opaque-block",
                                  "annotated-core", "predicated-core", "pipeline-order", "pipeline-stage"])
def test_opaque_or_explicit_order_scopes_still_reject_physical_expansion(scope):
    before, _ = _fixture(scope=scope)
    with pytest.raises(ValueError, match="ordinary lexical ownership/completion"):
        transform.AscendLowerL1Gemm()(before)


@pytest.mark.parametrize("kind", ["split-guard", "two-guards", "multi-reader", "clear-false", "partial-export",
                                 "unknown-metadata", "escape", "alias"])
def test_pair_and_escape_requirements_remain_conservative_with_fixed_one_metadata(kind):
    before, _ = _fixture(kind=kind)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match="Ascend dav-2201 physical M/N GEMM"):
        transform.AscendLowerL1Gemm()(before)
    assert tvm.ir.save_json(before) == snapshot


@pytest.mark.parametrize("pv_chunk", [256, 512])
def test_partial_owner_accumulators_keep_their_identity_clear_and_root_bank_reservation(pv_chunk):
    before, facts = _fixture(pv_chunk=pv_chunk)
    after = transform.AscendLowerL1Gemm()(before)
    _scratch_and_metadata(after, facts)
    updates = []
    exports = []
    for call, mapping in _expanded(after["main"].body):
        if call.op.name == "tl.tileop.gemm":
            c, origin, shape = _decode(call.args[2])
            if c.data.same_as(facts["pv"].data):
                origin = tuple(_value(index, mapping) for index in origin)
                shape = tuple(_value(extent, mapping) for extent in shape)
                assert origin[1:] == (0, 0) and shape == (1, facts["m"], 64)
                updates.append((origin[0], _value(call.args[9], mapping), _value(call.args[7], mapping)))
        elif call.op.name == "tl.tileop.ascend_copy":
            source, origin, _ = _decode(call.args[0])
            if source.data.same_as(facts["pv"].data):
                assert sum(k for owner, _, k in updates if owner == _value(origin[0], mapping)) == facts["n"]
                exports.append(_value(origin[0], mapping))
    assert exports == [0, 1]
    for owner in range(2):
        steps = [(clear, k) for coordinate, clear, k in updates if coordinate == owner]
        assert steps and [clear for clear, _ in steps] == [1] + [0] * (len(steps) - 1)
        assert sum(k for _, k in steps) == facts["n"]
    root = after["main"].body.block
    for scope, limit in (("shared.l0a", 64 * 1024), ("shared.l0b", 64 * 1024)):
        assert sum(math.prod(map(int, buffer.shape)) * 2 for buffer in root.alloc_buffers if buffer.scope() == scope) <= limit
