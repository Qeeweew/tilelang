"""Native A/B selection preserves truthful effects and one-shot root state."""

from unittest.mock import patch

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from tilelang.ascend.transform import z3_scheduler


NATIVE = "tl.tileop.ascend_cube_load3d"


def _fixture(dtype="float16", m=127, cache_k=256, load_k=128,
             k_start=128, *, m_start=0, source_m=None, leading=False, annotations=None):
    source_m = m if source_m is None else source_m
    prefix = (2,) if leading else ()
    src = tirx.decl_buffer((*prefix, source_m, cache_k), dtype, name="a_l1", scope="shared.l1")
    dst = tirx.decl_buffer((*prefix, m, load_k), dtype, name="a_l0", scope="shared.l0a")
    origins = [1] if leading else []
    extents = [1] if leading else []
    body = copy(src, dst, src_indices=[*origins, m_start, k_start],
                dst_indices=[*origins, 0, 0], src_shape=[*extents, m, load_k],
                dst_shape=[*extents, m, load_k], annotations=annotations)
    mod = kernel(body, buffers=[src, dst], annotations={"layout_map": {
        src: make_ascend_major_k_layout(src),
        dst: make_ascend_l0a_2201_layout(dst),
    }})
    return mod


def _select(mod, arch="dav-2201", *, normalize_footprint=True):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", target)})
    with target:
        mod = transform.NormalizeAscendFractalStorage()(mod)
        mod = transform.AscendInsertOOBPadding()(mod)
        if normalize_footprint:
            mod = transform.AscendNormalizeCubeCopyFootprints()(mod)
        return transform.AscendSelectCubeLoads()(mod)


def _equal(actual, expected):
    assert len(actual) == len(expected)
    analyzer = tvm.arith.Analyzer()
    for left, right in zip(actual, expected):
        assert analyzer.can_prove_equal(left, right), (left, right)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("m", [127, 128])
def test_full_padded_m_subk_has_exact_native_regions(dtype, m):
    mod = _select(_fixture(dtype, m))
    (load,) = calls(mod, NATIVE)
    read, write = load.args[:2]
    assert read.op.name == write.op.name == "tl.region"
    _equal(read.args[0].buffer.shape, [128, 256])
    _equal(write.args[0].buffer.shape, [128, 128])
    _equal(read.args[0].indices, [0, 128])
    _equal(write.args[0].indices, [0, 0])
    _equal(read.args[1:], [1, 128, 128])
    _equal(write.args[1:], [2, 128, 128])
    _equal(load.args[2:], [128, 256])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_late_lowering_keeps_versions_in_base_pointers_and_subk_in_descriptor(dtype):
    mod = _select(_fixture(dtype, leading=True))
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    with target:
        after = transform.AscendLowerTileOp()(mod)
    (load,) = calls(after, "tl.ascend_load_cbuf_to_ca_3d")
    assert not calls(after, NATIVE)
    _equal(load.args[2:], [128, 256, 128, 128, 128])
    # Source's K origin is not counted twice. Leading matrix versions use the
    # allocation-wide pitch, independently for source and destination.
    _equal([load.args[0].args[2], load.args[1].args[2]], [128 * 128, 128 * 256])


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_materialized_source_and_destination_clocks_keep_independent_base_pitches(index_dtype):
    def index(value):
        return tirx.IntImm(index_dtype, value)

    selected = _select(_fixture(m=index(32), cache_k=index(32),
                               load_k=index(16), k_start=index(16)))
    (native,) = calls(selected, NATIVE)
    src, dst = [operand.args[0].buffer for operand in native.args[:2]]
    assert all(str(dim.dtype) == index_dtype
               for buffer in (src, dst) for dim in buffer.shape)
    assert all(str(value.dtype) == index_dtype for value in native.args[2:])
    epoch = tirx.Var("epoch", index_dtype)
    loop = tirx.For(epoch, index(0), index(7), tirx.ForKind.SERIAL,
                   unit(tirx.Evaluate(native), core=2),
                   annotations={"multi_buffer_eligible": [src.data, dst.data]})
    prepared = kernel(unit(loop, core=None), buffers=[src, dst], annotations={
        "layout_map": {src: make_ascend_major_k_layout(src), dst: make_ascend_l0a_2201_layout(dst)},
        "tl.buffer_versions_map": {src.data: 2, dst.data: 3},
    })
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    prepared["main"] = prepared["main"].with_attr("target", target)
    with target:
        materialized = transform.MaterializeMultiBuffer()(prepared)
        (versioned,) = calls(materialized, NATIVE)
        _equal(versioned.args[0].args[0].buffer.shape, [2, 32, 32])
        _equal(versioned.args[1].args[0].buffer.shape, [3, 32, 16])
        # The bounded version number follows the existing int32 version axis;
        # its epoch and the matrix dimensions retain their requested width.
        src_version = tirx.Cast("int32", epoch % 2)
        dst_version = tirx.Cast("int32", epoch % 3)
        _equal(versioned.args[0].args[0].indices, [src_version, 0, 16])
        _equal(versioned.args[1].args[0].indices, [dst_version, 0, 0])
        for operand in versioned.args[:2]:
            assert all(str(dim.dtype) == index_dtype
                       for dim in operand.args[0].buffer.shape[-2:])
        lowered = transform.AscendLowerTileOp()(materialized)
    (pointer,) = calls(lowered, "tl.ascend_load_cbuf_to_ca_3d")
    assert not calls(lowered, NATIVE)
    _equal([pointer.args[0].args[2], pointer.args[1].args[2]],
           [tirx.Cast(index_dtype, dst_version) * 32 * 16,
            tirx.Cast(index_dtype, src_version) * 32 * 32])
    assert all(str(operand.args[2].dtype) == index_dtype
               for operand in pointer.args[:2])
    # The source K origin contributes only the final descriptor field. Both
    # pointers contain their own version clock and no additional sub-K offset.
    _equal(pointer.args[2:], [32, 32, 32, 16, 16])


def test_partial_logical_k_reads_the_complete_in_bounds_physical_fragment():
    mod = _select(_fixture(m=31, cache_k=32, load_k=6, k_start=16))
    (load,) = calls(mod, NATIVE)
    _equal(load.args[0].args[0].buffer.shape, [32, 32])
    _equal(load.args[0].args[0].indices, [0, 16])
    _equal(load.args[0].args[2:], [32, 16])
    _equal(load.args[1].args[0].buffer.shape, [32, 16])
    _equal(load.args[1].args[2:], [32, 16])


@pytest.mark.parametrize("kwargs", [
    {"source_m": 256},             # only half the physical source M
    {"source_m": 256, "m_start": 16},
    {"k_start": 1},               # K origin is not a complete B16 fractal
    {"cache_k": 256, "load_k": 128, "k_start": 144},  # physical OOB
    {"m": 32768, "cache_k": 16, "load_k": 16, "k_start": 0},
    {"m": 16, "cache_k": 65536, "load_k": 16, "k_start": 0},
])
def test_unproved_or_unsupported_geometry_retains_the_existing_copy(kwargs):
    # Oversized buffers isolate native field selection, not whole-kernel
    # resource legality. The established target capacity boundary remains.
    mod = _select(_fixture(**kwargs), normalize_footprint=False)
    assert not calls(mod, NATIVE)
    assert calls(mod, "tl.tileop.ascend_copy")


def test_overflowing_int32_source_origin_retains_checked_2d_boundary():
    # This aligned positive origin plus 128 wraps to -2147483536 in int32.
    # Selection must compare the physical footprint in int64, then preserve
    # the existing 2D lowering's explicit source-allocation diagnostic.
    mod = _select(_fixture(m=32, k_start=2147483632), normalize_footprint=False)
    assert not calls(mod, NATIVE)
    assert calls(mod, "tl.tileop.ascend_copy")
    with mod["main"].attrs["target"], pytest.raises(
        tvm.error.InternalError, match="source footprint exceeds"
    ):
        transform.AscendLowerTileOp()(mod)


def test_overflowing_int64_source_origin_plus_extent_retains_checked_2d_boundary():
    # The aligned positive origin is INT64_MAX-15. Adding the 128-element
    # request overflows int64, so merely widening an endpoint is insufficient.
    origin = tirx.IntImm("int64", 2**63 - 16)
    mod = _select(_fixture(m=32, k_start=origin), normalize_footprint=False)
    assert not calls(mod, NATIVE)
    assert not calls(mod, CFG)
    (original,) = calls(mod, "tl.tileop.ascend_copy")
    retained_origin = original.args[0].args[0].indices[-1]
    assert str(retained_origin.dtype) == "int64"
    _equal([retained_origin, original.args[0].args[-1]], [origin, 128])
    with mod["main"].attrs["target"], pytest.raises(
        tvm.error.InternalError, match="source footprint exceeds"
    ):
        transform.AscendLowerTileOp()(mod)


def test_3510_keeps_its_existing_copy_path():
    mod = _select(_fixture(), arch="dav-3510")
    assert not calls(mod, NATIVE)
    assert calls(mod, "tl.tileop.ascend_copy")


@pytest.mark.parametrize("key", ["transpose", "data_select", "dual_dst_ctl", "nd2nz",
                                 "unit_flag_ctrl", "sub_blockid", "l2_cache_ctrl",
                                 "pad_value", "dst_block", "coalesced_width", "unknown_hint"])
def test_annotated_copy_retains_original_path_and_annotation(key):
    # The input capability boundary does not reject every hint on L1->L0.
    # Native selection must not silently discard either a hardware control
    # or a currently unknown hint. Even explicit default hints take 2D.
    value = tirx.IntImm("int32", 0)
    mod = _select(_fixture(annotations={key: value}))
    assert not calls(mod, NATIVE)
    (original,) = calls(mod, "tl.tileop.ascend_copy")
    tvm.ir.assert_structural_equal(original.annotations[key], value)


def test_selection_is_idempotent():
    once = _select(_fixture())
    twice = transform.AscendSelectCubeLoads()(once)
    tvm.ir.assert_structural_equal(once, twice)


CFG = "tl.ascend_set_load3d_config"


def _state_fixture(*, nested=True, per_core=False, consumers=1):
    selected = _select(_fixture(m=32, cache_k=32, load_k=16, k_start=16))
    (load,) = calls(selected, NATIVE)
    (config,) = calls(selected, CFG)
    buffers = [operand.args[0].buffer for operand in load.args[:2]]
    loops = []
    for consumer in range(consumers):
        operation = tirx.Evaluate(load)
        if per_core:
            metadata = {"core_mask": tirx.IntImm("int64", 2),
                        "latency": tirx.IntImm("int64", 1),
                        "ii": tirx.IntImm("int64", 1)}
            operation = tirx.AttrStmt(metadata, "tl.ascend_per_core_task", 1,
                                      unit(operation, core=2).body)
            operation = unit(operation, core=None)
        else:
            operation = unit(operation, core=2, cost=(1, 1))
        if nested:
            j = tirx.Var(f"j{consumer}", "int32")
            operation = unit(tirx.For(j, 0, 3, tirx.ForKind.SERIAL, operation), core=None)
        i = tirx.Var(f"i{consumer}", "int32")
        loops.append(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, operation), core=None))
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = kernel(seq(unit(tirx.Evaluate(config), core=2, cost=(1, 1)), *loops), buffers=buffers)
    mod["main"] = mod["main"].with_attr("target", target)
    return mod


def test_root_raw_is_shared_with_the_issue_scheduler():
    before = _state_fixture(nested=False)
    captured = []

    def schedule(*args, **kwargs):
        captured.extend(tuple(int(x) for x in edge) for edge in args[3])
        return list(range(len(args[0]))), list(range(len(args[0])))

    with patch.object(z3_scheduler, "z3_schedule_python", side_effect=schedule):
        transform.AutoSchedule()(before)
    assert (0, 1, 1) in captured


def _sync_state(mod):
    with mod["main"].attrs["target"]:
        for factory in [transform.AutoSchedule, transform.AssignCore,
                        transform.PrepareMultiBuffer, transform.ResolveCore,
                        transform.InsertSync]:
            mod = factory()(mod)
    return mod


@pytest.mark.parametrize("per_core", [False, True])
@pytest.mark.parametrize("consumers", [1, 2])
def test_root_configuration_completion_is_consumed_once_at_each_loop_entry(per_core, consumers):
    # Decode/AutoSchedule/Prepare/Resolve/Decode preserve the exact read-only
    # SPR fact, including grouped candidates. The configuration has no buffer.
    mod = _sync_state(_state_fixture(per_core=per_core, consumers=consumers))
    sets = [c for c in calls(mod, "tl.ascend_set_flag") if c.args[0].value == "S_MTE1"]
    waits = [c for c in calls(mod, "tl.ascend_wait_flag") if c.args[0].value == "S_MTE1"]
    assert sets and waits, mod.script()
    assert len(calls(mod, CFG)) == 1
    analyzer = tvm.arith.Analyzer()
    for wait in waits:
        assert sum(analyzer.can_prove_equal(wait.args[1], s.args[1]) for s in sets) == 1
    # A single producer event must never be consumed by two separate sites.
    for setter in sets:
        assert sum(analyzer.can_prove_equal(setter.args[1], w.args[1]) for w in waits) == 1

    def visit(stmt, loops=(), guards=()):
        if isinstance(stmt, tirx.For):
            visit(stmt.body, (*loops, stmt), guards)
        elif isinstance(stmt, tirx.IfThenElse):
            visit(stmt.then_case, loops, (*guards, stmt.condition))
            if stmt.else_case is not None:
                visit(stmt.else_case, loops, (*guards, ~stmt.condition))
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, loops, guards)
        elif isinstance(stmt, tirx.AttrStmt):
            visit(stmt.body, loops, guards)
        elif isinstance(stmt, tirx.SBlockRealize):
            visit(stmt.block.body, loops, guards)
        elif isinstance(stmt, tirx.SBlock):
            visit(stmt.body, loops, guards)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "tl.ascend_wait_flag" and call.args[0].value == "S_MTE1":
                activation = tirx.all(*guards) if guards else tirx.const(True, "bool")
                with analyzer.constraint_scope(activation):
                    for loop in loops:
                        assert analyzer.can_prove(loop.loop_var == loop.min), mod.script()
    visit(mod["main"].body)


def _assert_one_shot_s_mte1_events(mod):
    sets = [c for c in calls(mod, "tl.ascend_set_flag") if c.args[0].value == "S_MTE1"]
    waits = [c for c in calls(mod, "tl.ascend_wait_flag") if c.args[0].value == "S_MTE1"]
    assert sets and waits, mod.script()
    analyzer = tvm.arith.Analyzer()
    for setter in sets:
        assert sum(analyzer.can_prove_equal(setter.args[1], w.args[1]) for w in waits) == 1
    for wait in waits:
        assert sum(analyzer.can_prove_equal(wait.args[1], s.args[1]) for s in sets) == 1


@pytest.mark.parametrize("factory", [transform.AutoSchedule, transform.InsertSync])
@pytest.mark.parametrize("shape", ["two_3d_leaves", "3d_and_2d_leaves",
                                   "two_3d_in_one_task", "internal_loop"])
def test_compound_per_core_load3d_reader_is_diagnosed_before_sync_migration(factory, shape):
    selected = _select(_fixture(m=32, cache_k=32, load_k=16, k_start=16))
    (load,) = calls(selected, NATIVE)
    (config,) = calls(selected, CFG)
    source, destination = [operand.args[0].buffer for operand in load.args[:2]]
    candidate = unit(tirx.Evaluate(load), core=2, cost=(1, 1)).body
    if shape == "two_3d_leaves":
        body = seq(candidate, candidate)
    elif shape == "3d_and_2d_leaves":
        other = copy(source, destination, src_indices=[0, 16],
                     src_shape=[32, 16], dst_shape=[32, 16])
        body = seq(candidate, unit(other, core=2, cost=(1, 1)).body)
    elif shape == "two_3d_in_one_task":
        body = unit(seq(tirx.Evaluate(load), tirx.Evaluate(load)),
                    core=2, cost=(1, 1)).body
    else:
        i = tirx.Var("internal_epoch", "int32")
        # There is only one static Load3D call. Counting calls alone would
        # miss that the migrated one-shot wait executes twice at runtime.
        body = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, candidate)
    metadata = {"core_mask": tirx.IntImm("int64", 2),
                "latency": tirx.IntImm("int64", 1),
                "ii": tirx.IntImm("int64", 1)}
    grouped = tirx.AttrStmt(metadata, "tl.ascend_per_core_task", 1, body)
    before = kernel(seq(unit(tirx.Evaluate(config), core=2, cost=(1, 1)),
                        unit(grouped, core=None)), buffers=[source, destination])
    before["main"] = before["main"].with_attr("target", selected["main"].attrs["target"])
    with before["main"].attrs["target"], pytest.raises(
        tvm.error.InternalError, match="Load3D T.PerCoreTask requires one flat T.Task leaf"
    ):
        factory()(before)


def test_two_load_leaves_in_one_nested_scope_keep_one_shot_completion():
    original = _state_fixture(nested=False)
    root = original["main"].body.block
    setup, outer_unit = root.body.seq
    outer = outer_unit.body
    (load,) = calls(outer.body, NATIVE)
    second = tirx.Call(load.dtype, load.op, load.args, annotations=load.annotations)
    j = tirx.Var("j", "int32")
    body = seq(outer.body, unit(tirx.Evaluate(second), core=2, cost=(1, 1)))
    inner = unit(tirx.For(j, 0, 3, tirx.ForKind.SERIAL, body), core=None)
    outer = tirx.For(outer.loop_var, outer.min, outer.extent, outer.kind, inner)
    mod = kernel(seq(setup, unit(outer, core=None)), buffers=root.alloc_buffers)
    mod["main"] = mod["main"].with_attr("target", original["main"].attrs["target"])
    after = transform.InsertSync()(mod)
    assert len(calls(after, NATIVE)) == 2
    assert len(calls(after, CFG)) == 1
    _assert_one_shot_s_mte1_events(after)


def test_maybe_empty_first_consumer_waits_outside_its_control():
    original = _state_fixture(nested=False, consumers=2)
    root = original["main"].body.block
    setup, first_unit, second_unit = root.body.seq
    first = first_unit.body
    count = tirx.Var("count", "int32")
    first = tirx.For(first.loop_var, first.min, count, first.kind, first.body)
    mod = kernel(seq(setup, unit(first, core=None), second_unit),
                 buffers=root.alloc_buffers, params=[count])
    mod["main"] = mod["main"].with_attr("target", original["main"].attrs["target"])
    after = transform.InsertSync()(mod)
    _assert_one_shot_s_mte1_events(after)
    # The first loop may execute zero times. Its completion wait must execute
    # before entry; an absent reader cannot discharge a later reader's edge.
    (maybe_empty,) = [loop for loop in nodes(after, tirx.For)
                      if loop.extent.same_as(count)]
    assert not [c for c in calls(maybe_empty.body, "tl.ascend_wait_flag")
                if c.args[0].value == "S_MTE1"], after.script()
    top = list(after["main"].body.block.body.seq)
    first_pos = next(i for i, stmt in enumerate(top) if nodes(stmt, tirx.For) and
                     any(loop.extent.same_as(count) for loop in nodes(stmt, tirx.For)))
    assert any(c.args[0].value == "S_MTE1"
               for stmt in top[:first_pos]
               for c in calls(stmt, "tl.ascend_wait_flag")), after.script()


def test_read_only_load_requires_a_root_initializer():
    mod = _state_fixture(nested=False)
    root = mod["main"].body.block
    # Remove only the initializer's complete schedule unit.
    root = tirx.SBlock([], [], [], "tilelang_root", root.body.seq[1], alloc_buffers=root.alloc_buffers)
    mod["main"] = mod["main"].with_body(tirx.SBlockRealize([], True, root))
    with pytest.raises(tvm.error.InternalError, match="preceding kernel-root configuration"):
        transform.InsertSync()(mod)


def test_different_kernel_local_fmatrix_values_fall_back_together():
    first, second = _fixture(m=32), _fixture(m=64)
    roots = [x["main"].body.block for x in [first, second]]
    layouts = {key: layout for root in roots for key, layout in root.annotations["layout_map"].items()}
    combined = kernel(seq(*(root.body for root in roots)),
                      buffers=[b for root in roots for b in root.alloc_buffers],
                      annotations={"layout_map": layouts})
    selected = _select(combined)
    assert not calls(selected, NATIVE)
    assert not calls(selected, CFG)
    assert len(calls(selected, "tl.tileop.ascend_copy")) == 2


def test_explicit_atomic_copy_retains_2d_path():
    original = _fixture()
    root = original["main"].body.block
    body = unit(root.body, core=2).body
    mod = kernel(body, buffers=root.alloc_buffers, annotations=root.annotations)
    selected = _select(mod)
    assert calls(selected, "tl.tileop.ascend_copy")
    assert not calls(selected, NATIVE)
    assert not calls(selected, CFG)


def test_conditionally_repeated_load_selects_with_root_state():
    original = _fixture()
    root = original["main"].body.block
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 4, tirx.ForKind.SERIAL,
                   tirx.IfThenElse(i % 2 == 0, root.body, None))
    mod = kernel(body, buffers=root.alloc_buffers, annotations=root.annotations)
    selected = _select(mod)
    assert not calls(selected, "tl.tileop.ascend_copy")
    assert len(calls(selected, NATIVE)) == len(calls(selected, CFG)) == 1
    (loop,) = nodes(selected, tirx.For)
    assert calls(loop.body, NATIVE)
    assert not calls(loop.body, CFG)


def test_gm_valid_read_and_padding_producer_survive_native_selection():
    gm = tirx.decl_buffer((127, 256), "float16", name="gm")
    l1 = tirx.decl_buffer((128, 256), "float16", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((128, 128), "float16", name="l0", scope="shared.l0a")
    body = seq(copy(gm, l1, src_shape=[128, 256]),
               copy(l1, l0, src_indices=[0, 128], src_shape=[128, 128]))
    mod = kernel(body, buffers=[l1, l0], params=[gm], annotations={"layout_map": {
        l1: make_ascend_major_k_layout(l1), l0: make_ascend_l0a_2201_layout(l0),
    }})
    selected = _select(mod)
    (gm_copy,) = calls(selected, "tl.tileop.ascend_copy")
    _equal(gm_copy.args[0].args[2:], [127, 256])
    fills = calls(selected, "tl.tileop.fill")
    assert fills, selected.script()
    (native,) = calls(selected, NATIVE)
    _equal(native.args[0].args[0].indices, [0, 128])
    _equal(native.args[0].args[2:], [128, 128])

    # The shared dependency analysis must see the native read overlap with
    # row-padding writes, then carry its MTE2->MTE1 completion to this task.
    with selected["main"].attrs["target"]:
        scheduled = selected
        for factory in [transform.NormalizeControlFlowForSchedule,
                        transform.MaterializeScheduleUnits,
                        transform.AnnotateMultiBufferEligible,
                        transform.EstimateLatency, transform.AutoSchedule,
                        transform.AssignCore, transform.PrepareMultiBuffer,
                        transform.ResolveCore, transform.InsertSync]:
            scheduled = factory()(scheduled)
    tasks = [node for node in nodes(scheduled, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    consumers = [task for task in tasks if calls(task.body, NATIVE)]
    assert consumers, scheduled.script()
    padding_tasks = [task for task in tasks if calls(task.body, "tl.tileop.fill")]
    assert padding_tasks, scheduled.script()
    # A set before the padding task would not cover that producer. Match the
    # concrete hard-event ID from after the last padding task to a wait before
    # the selected native load, independently of event allocator numbering.
    last_padding = max(i for i, task in enumerate(tasks) if calls(task.body, "tl.tileop.fill"))
    consumer = next(i for i, task in enumerate(tasks) if calls(task.body, NATIVE))
    assert last_padding < consumer, scheduled.script()
    between = tasks[last_padding + 1:consumer]
    sets = [call for task in between for call in calls(task.body, "tl.ascend_set_flag") if call.args[0].value == "MTE2_MTE1"]
    waits = [call for task in between for call in calls(task.body, "tl.ascend_wait_flag") if call.args[0].value == "MTE2_MTE1"]
    analyzer = tvm.arith.Analyzer()
    assert sets and waits and any(analyzer.can_prove_equal(left.args[1], right.args[1]) for left in sets for right in waits), scheduled.script()


def test_subk_tail_read_keeps_the_actual_explicit_k_padding_producer():
    gm = tirx.decl_buffer((32, 16), "float16", name="gm")
    l1 = tirx.decl_buffer((32, 32), "float16", name="l1", scope="shared.l1")
    l0 = tirx.decl_buffer((32, 6), "float16", name="l0", scope="shared.l0a")
    body = seq(copy(gm, l1, src_shape=[32, 32]),
               copy(l1, l0, src_indices=[0, 16], src_shape=[32, 6]))
    mod = kernel(body, buffers=[l1, l0], params=[gm], annotations={"layout_map": {
        l1: make_ascend_major_k_layout(l1), l0: make_ascend_l0a_2201_layout(l0),
    }})
    selected = _select(mod)
    (gm_copy,) = calls(selected, "tl.tileop.ascend_copy")
    _equal(gm_copy.args[0].args[2:], [32, 16])
    (fill,) = calls(selected, "tl.tileop.fill")
    (load,) = calls(selected, NATIVE)
    _equal(fill.args[0].args[0].indices, [0, 16])
    _equal(fill.args[0].args[2:], [32, 16])
    _equal(load.args[0].args[0].indices, [0, 16])
    _equal(load.args[0].args[2:], [32, 16])
    assert fill.args[0].args[0].buffer.same_as(load.args[0].args[0].buffer)


def test_host_root_and_multiple_kernel_roots_initialize_independently():
    first, second = _fixture(m=32), _fixture(m=64)
    host = tirx.SBlock([], [], [], "host_root", seq(first["main"].body, second["main"].body))
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = tvm.IRModule({"main": tirx.PrimFunc([], tirx.SBlockRealize([], True, host)).with_attr("target", target)})
    with target:
        mod = transform.NormalizeAscendFractalStorage()(mod)
        mod = transform.AscendInsertOOBPadding()(mod)
        mod = transform.AscendNormalizeCubeCopyFootprints()(mod)
        mod = transform.AscendSelectCubeLoads()(mod)
    assert len(calls(mod, CFG)) == len(calls(mod, NATIVE)) == 2
    kernels = [b for b in nodes(mod, tirx.SBlock) if b.name_hint == "tilelang_root"]
    assert [int(calls(b, CFG)[0].args[0]) for b in kernels] == [32, 64]
    for root in kernels:
        marker = root.body.seq[0]
        assert isinstance(marker, tirx.AttrStmt) and marker.attr_key == "tl.ascend_task"
        assert int(marker.node["core_mask"]) == 2


def test_repeated_configuration_inside_loop_is_diagnosed():
    mod = _state_fixture(nested=False)
    root = mod["main"].body.block
    setup, loop_unit = root.body.seq
    loop = loop_unit.body
    repeated = tirx.For(loop.loop_var, loop.min, loop.extent, loop.kind,
                        seq(setup, loop.body))
    replacement = kernel(unit(repeated, core=None), buffers=root.alloc_buffers)
    replacement["main"] = replacement["main"].with_attr("target", mod["main"].attrs["target"])
    with pytest.raises(tvm.error.InternalError, match="unconditional kernel-root S task"):
        transform.InsertSync()(replacement)



def _b_fixture(dtype="float16", *, leading=False, cache_k=1024, n=64,
               load_k=128, k_start=384, n_start=0, load_n=None):
    prefix = (2,) if leading else ()
    load_n = n if load_n is None else load_n
    src = tirx.decl_buffer((*prefix, cache_k, n), dtype, name="b_l1", scope="shared.l1")
    dst = tirx.decl_buffer((*prefix, load_k, load_n), dtype, name="b_l0", scope="shared.l0b")
    origins, extents = ([1], [1]) if leading else ([], [])
    return kernel(copy(src, dst, src_indices=[*origins, k_start, n_start],
                       dst_indices=[*origins, 0, 0], src_shape=[*extents, load_k, load_n],
                       dst_shape=[*extents, load_k, load_n]), buffers=[src, dst],
                  annotations={"layout_map": {src: make_ascend_major_k_layout(src),
                                               dst: make_ascend_major_mn_layout(dst)}})


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("leading", [False, True])
def test_full_n_b_native_load_retains_row_origin_and_version_pitch(dtype, leading):
    selected = _select(_b_fixture(dtype, leading=leading))
    (native,) = calls(selected, NATIVE)
    read, write = native.args[:2]
    _equal(read.args[0].indices, [1, 384, 0] if leading else [384, 0])
    _equal(read.args[2:], [1, 128, 64] if leading else [128, 64])
    _equal(write.args[0].indices, [1, 0, 0] if leading else [0, 0])
    _equal(native.args[2:], [1024, 64])
    _equal(calls(selected, CFG)[0].args, [1, 1024])
    with selected["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(selected)
    (load,) = calls(lowered, "tl.ascend_load_cbuf_to_cb_3d")
    assert not calls(lowered, NATIVE)
    _equal(load.args[2:], [1024, 64, 128, 64, 384])
    _equal([load.args[0].args[2], load.args[1].args[2]],
           [128 * 64, 1024 * 64] if leading else [0, 0])


@pytest.mark.parametrize("kwargs", [
    {"k_start": 1},
    {"k_start": 912},
    {"n_start": 16, "load_n": 32},
    {"load_n": 32},
    {"cache_k": 32768},
])
def test_b_unproved_geometry_keeps_existing_copy(kwargs):
    selected = _select(_b_fixture(**kwargs), normalize_footprint=False)
    assert not calls(selected, NATIVE)
    assert not calls(selected, CFG)
    assert calls(selected, "tl.tileop.ascend_copy")


def test_a_and_b_configure_independent_fmatrix_widths():
    roots = [x["main"].body.block for x in (_fixture(m=128), _b_fixture())]
    combined = kernel(seq(*(root.body for root in roots)),
                      buffers=[b for root in roots for b in root.alloc_buffers],
                      annotations={"layout_map": {b: layout for root in roots
                                                 for b, layout in root.annotations["layout_map"].items()}})
    selected = _select(combined)
    assert len(calls(selected, NATIVE)) == 2
    (config,) = calls(selected, CFG)
    _equal(config.args, [128, 1024])


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_conflicting_a_bank_keeps_invariant_b_native_regions(dtype):
    roots = [mod["main"].body.block for mod in
             (_fixture(dtype, m=32), _fixture(dtype, m=64), _b_fixture(dtype))]
    combined = kernel(seq(*(root.body for root in roots)),
                      buffers=[buffer for root in roots for buffer in root.alloc_buffers],
                      annotations={"layout_map": {buffer: layout for root in roots
                                                 for buffer, layout in root.annotations["layout_map"].items()}})
    selected = _select(combined)
    (native,) = calls(selected, NATIVE)
    (config,) = calls(selected, CFG)
    _equal(config.args, [1, 1024])
    tvm.ir.assert_structural_equal(native.args[:2], calls(roots[2], "tl.tileop.ascend_copy")[0].args)
    _equal(native.args[2:], [1024, 64])
    fallback = calls(selected, "tl.tileop.ascend_copy")
    assert len(fallback) == 2
    for actual, root in zip(fallback, roots[:2]):
        tvm.ir.assert_structural_equal(actual, calls(root, "tl.tileop.ascend_copy")[0])
    assert not calls(seq(*list(selected["main"].body.block.body.seq)[1:]), CFG)
    tvm.ir.assert_structural_equal(transform.AscendSelectCubeLoads()(selected), selected)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_conflicting_b_bank_keeps_invariant_a_native_regions(dtype):
    roots = [mod["main"].body.block for mod in
             (_fixture(dtype, m=128), _b_fixture(dtype, cache_k=512), _b_fixture(dtype, cache_k=1024))]
    combined = kernel(seq(*(root.body for root in roots)),
                      buffers=[buffer for root in roots for buffer in root.alloc_buffers],
                      annotations={"layout_map": {buffer: layout for root in roots
                                                 for buffer, layout in root.annotations["layout_map"].items()}})
    selected = _select(combined)
    (native,) = calls(selected, NATIVE)
    (config,) = calls(selected, CFG)
    _equal(config.args, [128, 1])
    tvm.ir.assert_structural_equal(native.args[:2], calls(roots[0], "tl.tileop.ascend_copy")[0].args)
    _equal(native.args[2:], [128, 256])
    fallback = calls(selected, "tl.tileop.ascend_copy")
    assert len(fallback) == 2
    for actual, root in zip(fallback, roots[1:]):
        tvm.ir.assert_structural_equal(actual, calls(root, "tl.tileop.ascend_copy")[0])
    assert not calls(seq(*list(selected["main"].body.block.body.seq)[1:]), CFG)
    tvm.ir.assert_structural_equal(transform.AscendSelectCubeLoads()(selected), selected)


@pytest.mark.parametrize("activation", ["periodic", "late", "runtime"])
def test_conditional_native_readers_acquire_root_state_once_before_all_controls(activation):
    original = _state_fixture(nested=False, consumers=2)
    root = original["main"].body.block
    setup, first, second = root.body.seq
    loop = first.body
    (load,) = calls(loop.body, NATIVE)
    enabled = tirx.Var("enabled", "bool")
    guard = (loop.loop_var % 2 == 0 if activation == "periodic" else
             loop.loop_var >= 2 if activation == "late" else enabled)
    guarded = unit(tirx.For(loop.loop_var, loop.min, loop.extent, loop.kind,
                           unit(tirx.Evaluate(load), core=2, cost=(1, 1), guard=guard)), core=None)
    before = kernel(seq(setup, guarded, second), buffers=root.alloc_buffers,
                    params=[enabled] if activation == "runtime" else [])
    before["main"] = before["main"].with_attr("target", original["main"].attrs["target"])
    after = _sync_state(before)
    _assert_one_shot_s_mte1_events(after)
    sets = [c for c in calls(after, "tl.ascend_set_flag") if c.args[0].value == "S_MTE1"]
    waits = [c for c in calls(after, "tl.ascend_wait_flag") if c.args[0].value == "S_MTE1"]
    assert len(sets) == len(waits) == 1
    assert sum(isinstance(stmt.value, tirx.Call) and isinstance(stmt.value.op, tvm.ir.Op)
               and stmt.value.op.name == NATIVE for stmt in nodes(after, tirx.Evaluate)) == 2
    # The acquire executes even when the first reader's activation is empty.
    # Its root wait precedes both loops, so all later MTE1 descriptors inherit
    # initialized state by issue order without claiming old data completed.
    top = list(after["main"].body.block.body.seq)
    first_control = next(i for i, stmt in enumerate(top) if nodes(stmt, tirx.For))
    prefix = seq(*top[:first_control])
    assert len(calls(prefix, CFG)) == 1
    assert any(c.args[0].value == "S_MTE1" for c in calls(prefix, "tl.ascend_wait_flag"))
    for control in [stmt for stmt in top if nodes(stmt, tirx.For)]:
        assert not [c for c in calls(control, "tl.ascend_wait_flag") if c.args[0].value == "S_MTE1"]



@pytest.mark.parametrize("factory", [_fixture, _b_fixture], ids=["A", "B"])
@pytest.mark.parametrize("kind", ["extern", "pure_extern", "opaque_gemm"])
def test_unknown_native_state_keeps_ordinary_copy(factory, kind):
    original = factory()
    root = original["main"].body.block
    call = (tirx.Call("void", tvm.ir.Op.get("tl.ascend_gemm_l1"), [0] * 11)
            if kind == "opaque_gemm" else
            tirx.call_pure_extern("int32", "set_fmatrix_b", 0) if kind == "pure_extern" else
            tirx.call_extern("int32", "set_fmatrix", 0))
    before = kernel(seq(tirx.Evaluate(call), root.body), buffers=root.alloc_buffers,
                    annotations=root.annotations)
    after = _select(before)
    assert not calls(after, NATIVE)
    assert not calls(after, CFG)
    assert len(calls(after, "tl.tileop.ascend_copy")) == 1


@pytest.mark.parametrize("kind", ["extern", "pure_extern"])
def test_already_selected_load_rejects_unverified_native_state(kind):
    original = _state_fixture(nested=False)
    root = original["main"].body.block
    call = (tirx.call_pure_extern("int32", "set_fmatrix_b", 0) if kind == "pure_extern" else
            tirx.call_extern("int32", "set_fmatrix", 0))
    setup, loop = root.body.seq
    before = kernel(seq(setup, unit(tirx.Evaluate(call), core=2, cost=(1, 1)), loop),
                    buffers=root.alloc_buffers)
    before["main"] = before["main"].with_attr("target", original["main"].attrs["target"])
    with before["main"].attrs["target"], pytest.raises(tvm.error.InternalError, match="Load3D configuration cannot cross an external or opaque"):
        transform.InsertSync()(before)
