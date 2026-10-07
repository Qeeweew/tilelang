"""Oversized logical GEMMs export complete compact physical M/N tiles.

These host regressions check coverage and instruction effects, rather than a
particular tile choice or optimizer solution. No CANN compilation is requested.
The full frontend cases reach native code generation, but structural event
checks do not prove finite publish/release counts or hardware completion.
"""

import math

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout, make_ascend_l0c_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements


def _region(buffer, origin, shape):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent)
                                     for start, extent in zip(origin, shape)])


def _decode(expression):
    assert expression.op.name == "tl.region"
    return expression.args[0].buffer, expression.args[0].indices, expression.args[2:]


def _value(expression, mapping=None):
    if mapping:
        expression = tirx.stmt_functor.substitute(expression, mapping)
    return int(tvm.arith.Analyzer().simplify(expression))


def _fixture(m=256, n=256, k=64, *, dtype="float16", transpose_b=True,
             epochs=1, extra_l0c=False, kind="ordinary", arch="dav-2201"):
    # Nonzero L1 origins and a wider GM destination make dropping a region
    # origin or using the sub-tile width as the destination pitch observable.
    prefix = (2,) if epochs > 1 else ()
    a = tirx.decl_buffer((*prefix, m + 16, k + 32), dtype, name="a", scope="shared.l1")
    b_shape = (n + 16, k + 32) if transpose_b else (k + 32, n + 16)
    b = tirx.decl_buffer((*prefix, *b_shape), dtype, name="b", scope="shared.l1")
    c = tirx.decl_buffer((m, n), "float32", name="logical_c", scope="shared.l0c")
    out_prefix = (epochs,) if epochs > 1 else ()
    out = tirx.decl_buffer((*out_prefix, m + 32, n + 64), "float32", name="out")
    epoch = tirx.Var("epoch", "int64")
    leading = [epoch % 2] if epochs > 1 else []
    leading_extent = [1] if epochs > 1 else []
    a_region = _region(a, [*leading, 16, 32], [*leading_extent, m, k])
    b_region = _region(b, [*leading, *((16, 32) if transpose_b else (32, 16))],
                       [*leading_extent, *((n, k) if transpose_b else (k, n))])
    gemm = tirx.Evaluate(T.gemm(a_region, b_region, c, transpose_B=transpose_b,
                              clear_accum=kind != "clear-false"))
    dst_origin = [epoch, 16, 32] if epochs > 1 else [16, 32]
    exported_shape = [m - 16, n] if kind == "partial-export" else [m, n]
    export = copy(c, out, src_shape=exported_shape, dst_indices=dst_origin,
                  dst_shape=[*leading_extent, *exported_shape])
    body = seq(gemm, export)
    allocations = [a, b, c]
    layouts = {c.data: make_ascend_l0c_layout(c)}
    annotations = {"layout_map": layouts}
    params = [out]
    if extra_l0c:
        allocations.append(tirx.decl_buffer((128, 64), "float32", name="other_c", scope="shared.l0c"))
    if kind == "multi-reader":
        body = seq(body, export)
    elif kind == "escape":
        body = seq(body, tirx.Evaluate(tirx.call_extern("int32", "foreign", c.data)))
    elif kind == "alias":
        alias = tirx.decl_buffer(c.shape, c.dtype, name="alias_c", data=c.data, scope=c.scope())
        allocations.append(alias)
    elif kind == "custom-layout":
        layouts[c.data] = Layout(c.shape, lambda i, j: [j // 16, i // 16, i % 16, (j + 1) % 16])
    elif kind == "manual-version":
        annotations["tl.buffer_versions_map"] = {c.data: 2}
    elif kind == "separate-conditional":
        enabled = tirx.Var("enabled", "bool")
        params.append(enabled)
        body = seq(tirx.IfThenElse(enabled, gemm, None), export)
    if epochs > 1:
        body = tirx.For(epoch, tirx.const(0, "int64"), tirx.const(epochs, "int64"),
                        tirx.ForKind.SERIAL, body,
                        annotations={"tl.ascend_auto_pipeline": True, "num_stages": 4})
    mod = kernel(body, buffers=allocations, params=params, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return mod, {"a": a, "b": b, "c": c, "out": out, "m": m, "n": n, "k": k,
                 "epochs": epochs, "transpose_b": transpose_b}


def _expanded(statement, mapping=None):
    """Enumerate static ordinary loops without assuming their names or sizes."""
    mapping = mapping or {}
    if isinstance(statement, tirx.For):
        assert statement.kind == tirx.ForKind.SERIAL
        start, count = _value(statement.min, mapping), _value(statement.extent, mapping)
        for coordinate in range(start, start + count):
            yield from _expanded(statement.body, {**mapping, statement.loop_var:
                                                  tirx.const(coordinate, statement.loop_var.dtype)})
    elif isinstance(statement, tirx.SeqStmt):
        for child in statement.seq:
            yield from _expanded(child, mapping)
    elif isinstance(statement, tirx.SBlockRealize):
        assert _value(statement.predicate, mapping)
        yield from _expanded(statement.block, mapping)
    elif isinstance(statement, (tirx.SBlock, tirx.AttrStmt)):
        yield from _expanded(statement.body, mapping)
    elif isinstance(statement, tirx.Evaluate) and isinstance(statement.value, tirx.Call):
        yield statement.value, mapping
    elif isinstance(statement, tirx.Evaluate):
        assert _value(statement.value, mapping) == 0
    else:
        pytest.fail(f"Unexpected physical-tile control: {statement}")


def _l0c(mod):
    return [buffer for block in nodes(mod, tirx.SBlock) for buffer in block.alloc_buffers
            if buffer.scope() == "shared.l0c"]


def _assert_replaced(mod, facts, *, extra_bytes=0):
    old = facts["c"]
    allocations = _l0c(mod)
    scratch = [buffer for buffer in allocations if buffer.name != "other_c"]
    assert len(scratch) == 1
    scratch = scratch[0]
    assert not scratch.data.same_as(old.data)
    rows, columns = map(int, scratch.shape)
    assert rows > 0 and columns > 0 and rows % 16 == columns % 16 == 0
    assert facts["m"] % rows == facts["n"] % columns == 0
    assert rows * columns * 4 + extra_bytes <= 128 * 1024
    assert sum(math.prod(map(int, buffer.shape)) * 4 for buffer in allocations) <= 128 * 1024
    assert not scratch.strides or tuple(map(int, scratch.strides)) == (columns, 1)
    root = mod["main"].body.block
    assert int(root.annotations["tl.buffer_versions_map"][scratch.data]) == 1
    assert all(not buffer.data.same_as(old.data) for buffer in root.alloc_buffers)
    for key in root.annotations.get("layout_map", {}):
        storage = key.data if isinstance(key, tirx.Buffer) else key
        assert not storage.same_as(old.data)
    for load in nodes(mod, tirx.BufferLoad):
        assert not load.buffer.data.same_as(old.data)
    return scratch


def _assert_coverage_and_k_order(mod, facts, scratch):
    pending, loads, rectangles = [], {}, {epoch: [] for epoch in range(facts["epochs"])}
    for call, mapping in _expanded(mod["main"].body):
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name == "tl.tileop.ascend_copy":
            source, source_origin, source_shape = _decode(call.args[0])
            destination, destination_origin, destination_shape = _decode(call.args[1])
            src_origin = tuple(_value(index, mapping) for index in source_origin)
            dst_origin = tuple(_value(index, mapping) for index in destination_origin)
            src_shape = tuple(_value(extent, mapping) for extent in source_shape)
            dst_shape = tuple(_value(extent, mapping) for extent in destination_shape)
            assert int(call.args[0].args[1]) == 1 and int(call.args[1].args[1]) == 2
            if source.scope() == "shared.l1":
                assert destination.scope() in ("shared.l0a", "shared.l0b")
                assert all(index == 0 for index in dst_origin)
                loads[destination.data] = (source, src_origin, src_shape)
            elif source.scope() == "shared.l0c":
                assert source.data.same_as(scratch.data) and destination.data.same_as(facts["out"].data)
                assert src_origin == (0, 0) and src_shape == tuple(map(int, scratch.shape))
                assert dst_shape[-2:] == src_shape and all(extent == 1 for extent in dst_shape[:-2])
                epoch = dst_origin[0] if facts["epochs"] > 1 else 0
                row, col = dst_origin[-2:]
                rectangles[epoch].append((row, col, *src_shape))
                assert pending, "A physical FIX must follow its complete accumulation"
                cursor = 0
                for index, (a_load, b_load, clear, mask, dimensions) in enumerate(pending):
                    a_buffer, a_origin, a_shape = a_load
                    b_buffer, b_origin, b_shape = b_load
                    assert a_buffer.data.same_as(facts["a"].data) and b_buffer.data.same_as(facts["b"].data)
                    assert dimensions[:2] == src_shape
                    chunk_k = dimensions[2]
                    assert a_origin[-2:] == (row, 32 + cursor)
                    assert a_shape[-2:] == (src_shape[0], chunk_k)
                    expected_b = (col - 16, 32 + cursor) if facts["transpose_b"] else (32 + cursor, col - 16)
                    assert b_origin[-2:] == expected_b
                    assert b_shape[-2:] == ((src_shape[1], chunk_k) if facts["transpose_b"] else (chunk_k, src_shape[1]))
                    if facts["epochs"] > 1:
                        assert a_origin[:-2] == b_origin[:-2] == (epoch % 2,)
                        assert a_shape[:-2] == b_shape[:-2] == (1,)
                    assert clear == int(index == 0)
                    # A dynamic first-sub-K clear may conservatively retain
                    # rw effects for the entire loop; accumulation must read.
                    assert mask in (2, 3) and (clear or mask == 3)
                    cursor += chunk_k
                assert cursor == facts["k"]
                pending = []
        elif call.op.name == "tl.tileop.gemm":
            a, a_origin, _ = _decode(call.args[0])
            b, b_origin, _ = _decode(call.args[1])
            c, c_origin, c_shape = _decode(call.args[2])
            assert c.data.same_as(scratch.data)
            assert tuple(_value(index, mapping) for index in c_origin) == (0, 0)
            dimensions = tuple(_value(value, mapping) for value in call.args[5:8])
            assert tuple(_value(extent, mapping) for extent in c_shape) == dimensions[:2]
            assert all(_value(index, mapping) == 0 for index in [*a_origin, *b_origin])
            pending.append((loads[a.data], loads[b.data], _value(call.args[9], mapping),
                            int(call.args[2].args[1]), dimensions))
    assert not pending
    for epoch, tiles in rectangles.items():
        assert tiles, epoch
        # Every logical row has disjoint intervals covering the complete N
        # domain. This detects holes, duplicate tiles and omitted tail tiles.
        for row in range(16, 16 + facts["m"]):
            intervals = sorted((col, col + columns) for r, col, rows, columns in tiles if r <= row < r + rows)
            assert intervals[0][0] == 32 and intervals[-1][1] == 32 + facts["n"]
            assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
        assert sum(rows * columns for _, _, rows, columns in tiles) == facts["m"] * facts["n"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nt_complete_export_uses_bounded_compact_tiles(dtype):
    before, facts = _fixture(dtype=dtype)
    after = transform.AscendLowerL1Gemm()(before)
    scratch = _assert_replaced(after, facts)
    _assert_coverage_and_k_order(after, facts, scratch)


def test_nn_slices_b_columns_and_preserves_every_sub_k():
    before, facts = _fixture(128, 512, 256, transpose_b=False)
    after = transform.AscendLowerL1Gemm()(before)
    _assert_coverage_and_k_order(after, facts, _assert_replaced(after, facts))


def test_m_axis_is_split_when_full_m_exceeds_even_a_minimum_n_fractal():
    # No legal N-only tile can retain M=2304: 2304*16*4 > 128KiB.
    # This exercises M slicing independently of the implementation's tie
    # breaking, while keeping L1 and both native operand banks small.
    before, facts = _fixture(2304, 32, 16)
    after = transform.AscendLowerL1Gemm()(before)
    scratch = _assert_replaced(after, facts)
    assert int(scratch.shape[0]) < facts["m"]
    _assert_coverage_and_k_order(after, facts, scratch)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_leading_epochs_nonzero_origins_and_full_gm_pitch_are_preserved(dtype):
    before, facts = _fixture(dtype=dtype, epochs=3)
    after = transform.AscendLowerL1Gemm()(before)
    _assert_coverage_and_k_order(after, facts, _assert_replaced(after, facts))
    epoch = next(loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "epoch")
    assert str(epoch.loop_var.dtype) == "int64" and int(epoch.extent) == 3
    assert int(epoch.annotations["num_stages"]) == 4
    assert bool(epoch.annotations["tl.ascend_auto_pipeline"])
    for call in calls(after, "tl.tileop.ascend_copy"):
        source, _, _ = _decode(call.args[0])
        if source.scope() == "shared.l0c":
            destination, _, _ = _decode(call.args[1])
            assert tuple(map(int, destination.shape)) == (3, facts["m"] + 32, facts["n"] + 64)


def test_other_root_accumulator_is_charged_before_selecting_physical_tile():
    before, facts = _fixture(extra_l0c=True)
    after = transform.AscendLowerL1Gemm()(before)
    scratch = _assert_replaced(after, facts, extra_bytes=32 * 1024)
    assert any(buffer.name == "other_c" for buffer in _l0c(after))
    _assert_coverage_and_k_order(after, facts, scratch)


@pytest.mark.parametrize("kind", ["clear-false", "multi-reader", "escape", "alias", "custom-layout",
                                  "manual-version", "partial-export", "separate-conditional", "matrix-tail"])
def test_unsupported_oversized_lifetime_is_diagnosed_without_partial_rewrite(kind):
    before, facts = _fixture(m=255 if kind == "matrix-tail" else 256, kind=kind)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match="Ascend dav-2201 physical M/N GEMM"):
        transform.AscendLowerL1Gemm()(before)
    assert tvm.ir.save_json(before) == snapshot
    assert any(buffer.data.same_as(facts["c"].data) for buffer in _l0c(before))
    versions = before["main"].body.block.annotations.get("tl.buffer_versions_map", {})
    assert int(versions.get(facts["c"].data, 0)) != 1


def test_accumulator_that_already_fits_keeps_its_logical_storage():
    before, facts = _fixture(64, 64, 16)
    after = transform.AscendLowerL1Gemm()(before)
    (allocation,) = _l0c(after)
    assert allocation.data.same_as(facts["c"].data)
    (export,) = [call for call in calls(after, "tl.tileop.ascend_copy")
                 if _decode(call.args[0])[0].scope() == "shared.l0c"]
    assert _decode(export.args[0])[0].data.same_as(facts["c"].data)
    assert not nodes(after, tirx.For)


def test_3510_preserves_its_existing_l1_gemm_path():
    before, _ = _fixture(arch="dav-3510")
    tvm.ir.assert_structural_equal(before, transform.AscendLowerL1Gemm()(before))


@pytest.mark.parametrize("transpose_b", [True, False], ids=["nt", "nn"])
def test_native_fix_uses_compact_padded_m_and_full_destination_row_pitch(transpose_b):
    before, facts = _fixture(128, 512, 256, transpose_b=transpose_b)
    target = before["main"].attrs["target"]
    with target:
        after = transform.AscendLowerL1Gemm()(before)
        after = transform.AscendLayoutInference()(after)
        after = transform.NormalizeAscendFractalStorage()(after)
        after = transform.AscendNormalizeCubeCopyFootprints()(after)
        after = transform.AscendLowerTileOp()(after)
    allocations = {buffer.data: buffer for buffer in _l0c(after)}
    exports = calls(after, "tl.ascend_copy_matrix_cc_to_gm")
    assert exports
    for export in exports:
        assert export.args[1].op.name == "tirx.tvm_access_ptr"
        physical = allocations[export.args[1].args[1]]
        assert len(physical.shape) == 4
        padded_m = int(physical.shape[-3]) * 16
        assert _value(export.args[4]) == _value(export.args[6]) == padded_m
        assert _value(export.args[5]) == facts["n"] + 64
        assert _value(export.args[3]) == int(physical.shape[-4]) * 16


def _lower_frontend(program):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendLowerL1Gemm", "tl.AscendLowerMixedPartition", "tl.AutoSchedule",
                             "tl.InsertSync", "tl.MaterializeMultiBuffer", "tl.AscendLowerTileOp"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(program.with_attr("global_symbol", "main"),
                                target={"kind": "ascend", "arch": "dav-2201"},
                                enable_host_codegen=False, enable_device_compile=False)
    assert set(snapshots) == {"tl.AscendLowerL1Gemm", "tl.AscendLowerMixedPartition", "tl.AutoSchedule",
                              "tl.InsertSync", "tl.MaterializeMultiBuffer", "tl.AscendLowerTileOp"}
    return result.kernel_source, snapshots


def _fixed_scratch(mod, *, explicit=False):
    (scratch,) = [buffer for buffer in _l0c(mod) if buffer.name != "other_c"]
    roots = [block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root"]
    assert len(roots) == 1
    versions = roots[0].annotations.get("tl.buffer_versions_map", {})
    # Before owner discovery the override is explicit. AutoSchedule may omit
    # an ownerless allocation from its result table; omission means one slot.
    if explicit:
        assert scratch.data in versions
    assert int(versions.get(scratch.data, 1)) == 1
    if not explicit:
        for loop in nodes(mod, tirx.For):
            assert all(not storage.same_as(scratch.data)
                       for storage in loop.annotations.get("multi_buffer_eligible", []))
    return scratch


def test_mixed_full_handoff_follows_all_subtiles_and_keeps_private_row_ownership():
    @T.prim_func
    def program(a: T.Tensor((3, 256, 64), "float16"), b: T.Tensor((3, 256, 64), "float16"),
                out: T.Tensor((3, 256, 256), "float32")):
        with T.MixedKernel(1, sids=2):
            a_l1 = T.alloc_l1((256, 64), "float16")
            b_l1 = T.alloc_l1((256, 64), "float16")
            c = T.alloc_l0c((256, 256), "float32")
            vector = T.alloc_shared((256, 256), "float32")
            for epoch in T.Pipelined(3):
                T.copy(a[epoch, :, :], a_l1)
                T.copy(b[epoch, :, :], b_l1)
                T.gemm(a_l1, b_l1, c, transpose_B=True, clear_accum=True)
                T.copy(c, vector)
                for i, j in T.Parallel(256, 256):
                    vector[i, j] = vector[i, j] * 2.0
                T.copy(vector, out[epoch, :, :])

    source, snapshots = _lower_frontend(program)
    split = snapshots["tl.AscendLowerL1Gemm"]
    _fixed_scratch(split, explicit=True)
    (workspace,) = [buffer for block in nodes(split, tirx.SBlock) for buffer in block.alloc_buffers
                    if buffer.scope() == "global" and buffer.name.startswith("gm_handoff_")]
    # The GM→UB reader consumes the whole logical matrix after the M/N
    # loops, rather than publishing or vectorizing one physical sub-tile.
    (reader,) = [call for call in calls(split, "tl.tileop.ascend_copy")
                 if _decode(call.args[0])[0].data.same_as(workspace.data)]
    assert tuple(map(int, _decode(reader.args[0])[2][-2:])) == (256, 256)
    reader_parents = next(parents for statement, parents in statements(split)
                          if isinstance(statement, tirx.Evaluate) and statement.value.same_as(reader))
    assert len([parent for parent in reader_parents if isinstance(parent, tirx.For)]) == 1
    partitioned = snapshots["tl.AscendLowerMixedPartition"]
    private = [buffer for block in nodes(partitioned, tirx.SBlock) for buffer in block.alloc_buffers
               if buffer.scope() == "shared.dyn"]
    assert any(tuple(map(int, buffer.shape[-2:])) == (128, 256) for buffer in private)
    _fixed_scratch(snapshots["tl.AutoSchedule"])
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    assert sum(math.prod(map(int, buffer.shape)) * 4 for buffer in _l0c(materialized)) <= 128 * 1024
    assert "asc_sync_wait(PIPE_FIX, PIPE_M," in source
    synced = snapshots["tl.InsertSync"]
    # These are CPU integration checks that the real synchronization pass
    # emitted both protocols. ID presence is not a proof that every dynamic
    # sub-tile/epoch consumes exactly one token. The CANN/native acceptance
    # gate must separately review finite initialization, wrap and drain.
    sets = [call for call in calls(synced, "tl.ascend_set_flag") if call.args[0].value == "FIX_M"]
    waits = [call for call in calls(synced, "tl.ascend_wait_flag") if call.args[0].value == "FIX_M"]
    assert sets and waits
    assert {str(call.args[1]) for call in sets} == {str(call.args[1]) for call in waits}
    arrivals = calls(synced, "tl.ascend_cross_core_set_flag")
    consumers = calls(synced, "tl.ascend_cross_core_wait_flag")
    assert arrivals and consumers
    assert all(int(call.args[0]) == 2 for call in [*arrivals, *consumers])


def test_independent_automatic_accumulator_owner_shares_the_full_root_bank_budget():
    @T.prim_func
    def program(a: T.Tensor((3, 256, 16), "float16"), b: T.Tensor((3, 256, 16), "float16"),
                out: T.Tensor((3, 256, 256), "float32"), other_out: T.Tensor((3, 128, 64), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((256, 16), "float16")
            b_l1 = T.alloc_l1((256, 16), "float16")
            c = T.alloc_l0c((256, 256), "float32")
            other_c = T.alloc_l0c((128, 64), "float32")
            for epoch in T.Pipelined(3):
                T.copy(a[epoch, :, :], a_l1)
                T.copy(b[epoch, :, :], b_l1)
                T.gemm(a_l1, b_l1, c, transpose_B=True, clear_accum=True)
                T.copy(c, out[epoch, :, :])
            for other_epoch in T.Pipelined(3):
                T.copy(a[other_epoch, :, :], a_l1)
                T.copy(b[other_epoch, :, :], b_l1)
                T.gemm(a_l1[0:128, :], b_l1[0:64, :], other_c, transpose_B=True, clear_accum=True)
                T.copy(other_c, other_out[other_epoch, :, :])

    _, snapshots = _lower_frontend(program)
    _fixed_scratch(snapshots["tl.AscendLowerL1Gemm"], explicit=True)
    scheduled = snapshots["tl.AutoSchedule"]
    scratch = _fixed_scratch(scheduled)
    assert math.prod(map(int, scratch.shape)) * 4 <= 96 * 1024
    assert any(buffer.name == "other_c" for buffer in _l0c(scheduled))
    materialized = snapshots["tl.MaterializeMultiBuffer"]
    allocations = _l0c(materialized)
    assert len(allocations) == 2
    assert sum(math.prod(map(int, buffer.shape)) * 4 for buffer in allocations) <= 128 * 1024
    # The second owner may use more than one slot if the common bank budget
    # permits it. Its chosen ring size is deliberately not part of this test.
    assert any(buffer.name == "other_c" for buffer in allocations)
