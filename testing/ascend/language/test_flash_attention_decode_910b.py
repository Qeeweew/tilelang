"""Decode head packing preserves BNSD storage and ordinary attention math.

These cover frontend regions, CPU math and finite native completion. CANN
compilation and device accuracy remain separate gates.
"""

from collections import Counter, defaultdict, deque
from itertools import product

import numpy as np
import pytest

from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes, statements
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _PURE_DEFINITIONS, _SCALAR_VALUES, _region, _value,
)
from testing.ascend.language.test_flash_attention_910b import example, _grouped_frontend


def _frontend(example, *, batch=2, kv_heads=2, groups=17, block_q=16,
              variant="generic", dtype="float16", causal=False, query_start=0):
    tiling = None if block_q is None else example.FwdTiling(block_q, 16, 3)
    program = example.flash_attention_fwd(
        batch, kv_heads * groups, kv_heads, 1, 35, 16, dtype=dtype,
        variant=variant, causal=causal, query_start=query_start, tiling=tiling,
    )
    buffers = [program.buffer_map[param] for param in program.params]
    assert [tuple(map(int, buffer.shape)) for buffer in buffers] == [
        (batch, kv_heads * groups, 1, 16), (batch, kv_heads, 35, 16),
        (batch, kv_heads, 35, 16), (batch, kv_heads * groups, 1, 16),
    ], "Packing must not change the caller's four BNSD parameters"
    launch, = [loop for loop in nodes(program, tirx.For)
               if loop.thread_binding is not None
               and loop.thread_binding.thread_tag == "blockIdx.x"]
    kv_loop, = [loop for loop in nodes(program, tirx.For)
                if loop.kind == tirx.ForKind.SERIAL]
    transfers = calls(program, "tl.tileop.ascend_copy") + calls(program, "tl.tileop.copy")
    loads = [next(call for call in transfers
                  if call.args[0].args[0].buffer.data.same_as(buffer.data))
             for buffer in buffers[:3]]
    export, = [call for call in transfers
               if call.args[1].args[0].buffer.data.same_as(buffers[3].data)]
    for buffer, load in zip(buffers[:3], loads):
        assert sum(call.args[0].args[0].buffer.data.same_as(buffer.data)
                   for call in transfers) == 1
    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    return program, buffers, launch, kv_loop, loads, export, definitions


def _env(definitions, launch, block, kv_loop=None, kv=0):
    result = {_PURE_DEFINITIONS: definitions, _SCALAR_VALUES: {},
              launch.loop_var: tirx.const(block, launch.loop_var.dtype)}
    if kv_loop is not None:
        result[kv_loop.loop_var] = tirx.const(kv, kv_loop.loop_var.dtype)
    return result


def _parents(program, call):
    return next(parents for stmt, parents in statements(program)
                if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call))


def _predicate_value(expr, env):
    # Frontend boolean &/| are bitwise calls until lowering. The arithmetic
    # analyzer does not fold these calls even when both arguments are bools.
    if isinstance(expr, tirx.Call) and isinstance(expr.op, tvm.ir.Op):
        if expr.op.name == "tirx.bitwise_and":
            return _predicate_value(expr.args[0], env) & _predicate_value(expr.args[1], env)
        if expr.op.name == "tirx.bitwise_or":
            return _predicate_value(expr.args[0], env) | _predicate_value(expr.args[1], env)
    return bool(_value(expr, env))


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("batch,kv_heads,groups,block_q,expected_q", [
    (1, 1, 1, None, 16),
    (1, 1, 4, None, 16),
    (2, 2, 17, None, 32),
    (2, 2, 17, 16, 16),
    (1, 2, 4, 64, 64),
    (1, 1, 65, None, 64),
], ids=["mha-minimum", "gqa-four", "default-rounded", "head-tail",
        "explicit-capacity", "default-cap"])
def test_decode_views_cover_each_head_once_and_share_kv_per_head_tile(
        example, variant, batch, kv_heads, groups, block_q, expected_q):
    program, buffers, launch, kv_loop, loads, export, definitions = _frontend(
        example, batch=batch, kv_heads=kv_heads, groups=groups, block_q=block_q,
        variant=variant,
    )
    q_load, k_load, v_load = loads
    head_tiles = (groups + expected_q - 1) // expected_q
    assert int(launch.extent) == batch * kv_heads * head_tiles
    gemms = calls(program, "tl.tileop.gemm")
    assert len(gemms) == 2 and all(int(call.args[5]) == expected_q for call in gemms)
    assert bool(kv_loop.annotations.get("tl.ascend_auto_pipeline", False)) == (variant == "pipeline")
    if block_q is not None and variant == "pipeline":
        assert int(kv_loop.annotations["num_stages"]) == 3
    assert not any(isinstance(parent, tirx.For) and parent.same_as(kv_loop)
                   for parent in _parents(program, q_load))
    assert all(any(isinstance(parent, tirx.For) and parent.same_as(kv_loop)
                   for parent in _parents(program, load)) for load in (k_load, v_load))

    evaluations = [stmt.value for stmt, _ in statements(program)
                   if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call)]
    q_l1 = q_load.args[1].args[0].buffer
    padding, = [call for call in calls(program, "tl.tileop.fill")
                if call.args[0].args[0].buffer.data.same_as(q_l1.data)]
    assert padding.args[1].value == 0
    assert next(i for i, call in enumerate(evaluations) if call.same_as(padding)) < next(
        i for i, call in enumerate(evaluations) if call.same_as(q_load))
    assert not any(isinstance(parent, tirx.For) and parent.same_as(kv_loop)
                   for parent in _parents(program, padding))

    # Independent Cartesian ownership; no copy of the factory's div/mod map.
    expected = [(b, h, start, min(expected_q, groups - start))
                for b, h in product(range(batch), range(kv_heads))
                for start in range(0, groups, expected_q)]
    output_heads, kv_requests = Counter(), Counter()
    for block, (b, kv_head, head_offset, count) in enumerate(expected):
        env = _env(definitions, launch, block)
        head_start = kv_head * groups + head_offset
        q_view, q_origin, q_shape = _region(q_load.args[0], env)
        o_view, o_origin, o_shape = _region(export.args[1], env)
        for view, original in ((q_view, buffers[0]), (o_view, buffers[3])):
            assert view.data.same_as(original.data) and view.dtype == original.dtype
            assert tuple(map(int, view.shape)) == (batch, kv_heads * groups, 16)
            assert int(view.elem_offset) == 0
            assert not view.strides or tuple(map(int, view.strides)) == (kv_heads * groups * 16, 16, 1)
        assert q_origin == o_origin == (b, head_start, 0)
        assert q_shape == o_shape == (1, count, 16)
        assert head_start + count <= (kv_head + 1) * groups
        _, dst_origin, dst_shape = _region(q_load.args[1], env)
        assert dst_origin == (0, 0) and dst_shape[-1] == 16
        assert count <= dst_shape[0] <= expected_q
        _, pad_origin, pad_shape = _region(padding.args[0], env)
        assert pad_origin[1] == 0 and pad_shape[1] == 16
        assert pad_origin[0] <= count and pad_origin[0] + pad_shape[0] == expected_q
        _, out_origin, out_shape = _region(export.args[0], env)
        assert out_origin == (0, 0) and out_shape == (count, 16)
        for head in range(head_start, head_start + count):
            output_heads[b, head] += 1
        for kv in range(_value(kv_loop.extent, env)):
            env = _env(definitions, launch, block, kv_loop, kv)
            for label, load in (("K", k_load), ("V", v_load)):
                _, origin, shape = _region(load.args[0], env)
                width = int(gemms[0].args[6])
                assert origin == (b, kv_head, kv * width, 0)
                assert shape == (1, 1, width, 16)
                kv_requests[label, b, kv_head, kv] += 1
    assert output_heads == Counter({key: 1 for key in product(range(batch), range(kv_heads * groups))})
    trips = (35 + int(gemms[0].args[6]) - 1) // int(gemms[0].args[6])
    assert kv_requests == Counter({(label, b, h, kv): head_tiles
                                   for label, b, h, kv in product(("K", "V"), range(batch),
                                                                 range(kv_heads), range(trips))})


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("query_start", [0, 17, 34])
def test_decode_causal_predicate_and_trip_bound_use_one_position_for_every_head(
        example, variant, query_start):
    program, _, launch, kv_loop, loads, _, definitions = _frontend(
        example, variant=variant, causal=True, query_start=query_start,
    )
    masked = [(store, call) for store in nodes(program, tirx.BufferStore)
              for call in nodes(store.value, tirx.Call)
              if isinstance(call.op, tvm.ir.Op) and call.op.name == "tirx.if_then_else"]
    (store, selection), = masked
    parents = next(parents for stmt, parents in statements(program) if stmt.same_as(store))
    row, column = [loop for loop in parents if isinstance(loop, tirx.For)][-2:]
    assert int(row.extent) == int(column.extent) == 16
    expected_trips = (query_start + 1) // 16 + ((query_start + 1) % 16 != 0)
    for block in range(int(launch.extent)):
        env = _env(definitions, launch, block)
        assert _value(kv_loop.extent, env) == expected_trips
        count = _region(loads[0].args[0], env)[2][1]
        for kv, i, j in product(range(expected_trips), range(count), range(16)):
            env = _env(definitions, launch, block, kv_loop, kv)
            env.update({row.loop_var: tirx.const(i, row.loop_var.dtype),
                        column.loop_var: tirx.const(j, column.loop_var.dtype)})
            assert _predicate_value(selection.args[0], env) == (kv * 16 + j < 35
                                                           and kv * 16 + j <= query_start)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("causal,query_start", [(False, 0), (True, 0), (True, 17)])
def test_ir_gather_scatter_matches_the_original_bnsd_block_rounding_oracle(
        example, dtype, variant, causal, query_start):
    _, _, launch, kv_loop, loads, export, definitions = _frontend(
        example, dtype=dtype, variant=variant, causal=causal, query_start=query_start,
    )
    random = np.random.default_rng(421)
    q = random.normal(0, 0.5, (2, 34, 1, 16)).astype(np.float32)
    k = random.normal(0, 0.5, (2, 2, 35, 16)).astype(np.float32)
    v = random.normal(0, 0.5, k.shape).astype(np.float32)
    options = dict(dtype=dtype, block_kv=16, causal=causal,
                   query_start=query_start, softmax_scale=0.25)
    expected = example.attention_reference(q, k, v, **options)
    actual = np.full((2, 34, 16), np.nan, dtype=np.float32)
    for block in range(int(launch.extent)):
        env = _env(definitions, launch, block, kv_loop)
        _, (b, head, _), (_, count, _) = _region(loads[0].args[0], env)
        _, (kb, kv_head, _, _), _ = _region(loads[1].args[0], env)
        _, (ob, out_head, _), (_, out_count, _) = _region(export.args[1], env)
        stop = min(35, _value(kv_loop.extent, env) * 16)
        packed = example.attention_reference(
            q[b:b + 1, head:head + count], k[kb:kb + 1, kv_head:kv_head + 1, :stop],
            v[kb:kb + 1, kv_head:kv_head + 1, :stop], **options,
        )
        assert out_count == count
        actual[ob, out_head:out_head + out_count] = packed[0, :, 0]
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:, :, None], expected)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_decode_causal_oracle_has_no_head_dependent_visibility(example, dtype):
    q = np.zeros((2, 34, 1, 16), dtype=np.float32)
    k = np.zeros((2, 2, 35, 16), dtype=np.float32)
    v = np.empty_like(k)
    for b, head in product(range(2), range(2)):
        v[b, head] = (100 * b + 10 * head + np.arange(35, dtype=np.float32))[:, None]
    actual = example.attention_reference(q, k, v, dtype=dtype, block_kv=16,
                                         causal=True, query_start=17)
    labels = np.array([[100 * b + 10 * head + 8.5 for head in range(2)
                        for _ in range(17)] for b in range(2)], dtype=np.float32)
    expected = example._round_dtype(np.broadcast_to(labels[:, :, None, None], actual.shape), dtype)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("groups", [1, 4, 17, 32, 33])
def test_grouped_decode_has_exact_head_ownership_and_one_kv_import_per_epoch(example, variant, groups):
    program, buffers, launch, loop, q_loads, exports, kv_loads, gemms, definitions, enabled = _grouped_frontend(
        example, query_len=1, heads=2 * groups, variant=variant,
        causal=True, query_start=17,
    )
    assert [tuple(map(int, buffer.shape)) for buffer in buffers] == [
        (2, 2 * groups, 1, 16), (2, 2, 65, 16),
        (2, 2, 65, 16), (2, 2 * groups, 1, 16),
    ]
    # An everywhere-absent second query is folded out of the frontend. A
    # runtime-absent final query keeps its guarded nodes and zero regions.
    query_components = min(2, (groups + 15) // 16)
    assert len(q_loads) == len(exports) == query_components
    assert len(gemms) == 2 * query_components
    blocks = [(b, kv_head, first) for b, kv_head in product(range(2), range(2))
              for first in range(0, groups, 32)]
    assert int(launch.extent) == len(blocks)
    written = Counter()
    for block, (b, kv_head, first) in enumerate(blocks):
        env = _env(definitions, launch, block)
        assert _value(loop.extent, env) == 2
        for owner, (q_load, export) in enumerate(zip(q_loads, exports)):
            offset = first + owner * 16
            count = min(16, max(0, groups - offset))
            assert enabled(q_load, env) == enabled(export, env) == (count > 0)
            _, _, exported_shape = _region(export.args[0], env)
            assert exported_shape == (count, 16)
            if not count:
                continue
            _, q_origin, q_shape = _region(q_load.args[0], env)
            _, o_origin, o_shape = _region(export.args[1], env)
            assert q_origin == o_origin == (b, kv_head * groups + offset, 0)
            assert q_shape == o_shape == (1, count, 16)
            assert q_origin[1] + count <= (kv_head + 1) * groups
            written.update((b, head) for head in range(o_origin[1], o_origin[1] + count))
        for kv in range(2):
            step_env = _env(definitions, launch, block, loop, kv)
            for load in kv_loads:
                assert enabled(load, step_env)
                _, origin, shape = _region(load.args[0], step_env)
                assert origin == (b, kv_head, kv * 16, 0) and shape == (1, 1, 16, 16)
            for owner in range(query_components):
                assert enabled(gemms[owner * 2], step_env) == (first + owner * 16 < groups)
    assert written == Counter({head: 1 for head in product(range(2), range(2 * groups))})
    # Both packed query components use the same causal position. Query-head
    # offsets may select Q/O rows, but must never enter the key predicate.
    masked = [(store, selection) for store in nodes(program, tirx.BufferStore)
              for selection in nodes(store.value, tirx.Call)
              if isinstance(selection.op, tvm.ir.Op) and selection.op.name == "tirx.if_then_else"]
    assert len(masked) == query_components
    for store, selection in masked:
        parents = next(parents for stmt, parents in statements(program) if stmt.same_as(store))
        row, column = [parent for parent in parents if isinstance(parent, tirx.For)][-2:]
        for block, kv, i, j in product(range(int(launch.extent)), range(2), range(16), range(16)):
            env = _env(definitions, launch, block, loop, kv)
            env.update({row.loop_var: tirx.const(i, row.loop_var.dtype),
                        column.loop_var: tirx.const(j, column.loop_var.dtype)})
            assert _predicate_value(selection.args[0], env) == (kv * 16 + j <= 17)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("variant", ["generic", "pipeline"])
def test_grouped_decode_ir_scatter_preserves_original_bnsd_math(example, dtype, variant):
    _, _, launch, _, _, exports, _, _, definitions, enabled = _grouped_frontend(
        example, query_len=1, heads=66, variant=variant, dtype=dtype,
        causal=True, query_start=17,
    )
    q = np.zeros((2, 66, 1, 16), dtype=np.float32)
    k = np.zeros((2, 2, 65, 16), dtype=np.float32)
    v = np.empty_like(k)
    for batch, head in product(range(2), range(2)):
        v[batch, head] = (100 * batch + 10 * head + np.arange(65, dtype=np.float32))[:, None]
    options = dict(dtype=dtype, block_kv=16, causal=True, query_start=17)
    expected = example.attention_reference(q, k, v, **options)
    actual = np.full((2, 66, 16), np.nan, dtype=np.float32)
    for block in range(int(launch.extent)):
        env = _env(definitions, launch, block)
        for export in exports:
            if not enabled(export, env):
                continue
            _, (b, first, _), (_, count, _) = _region(export.args[1], env)
            kv_head = first // 33
            packed = example.attention_reference(
                q[b:b + 1, first:first + count], k[b:b + 1, kv_head:kv_head + 1],
                v[b:b + 1, kv_head:kv_head + 1], **options,
            )
            actual[b, first:first + count] = packed[0, :, 0]
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:, :, None], expected)


def test_short_prefill_keeps_query_rows_and_the_existing_default_tile(example):
    program = example.flash_attention_fwd(1, 4, 1, 2, 35, 16)
    q = program.buffer_map[program.params[0]]
    launch, = [loop for loop in nodes(program, tirx.For)
               if loop.thread_binding is not None and loop.thread_binding.thread_tag == "blockIdx.x"]
    assert int(launch.extent) == 4
    transfers = calls(program, "tl.tileop.ascend_copy") + calls(program, "tl.tileop.copy")
    q_load, = [call for call in transfers if call.args[0].args[0].buffer.data.same_as(q.data)]
    assert tuple(map(int, q_load.args[0].args[0].buffer.shape)) == (1, 4, 2, 16)
    assert all(int(call.args[5]) == 64 for call in calls(program, "tl.tileop.gemm"))


def _native_decode_lower(program):
    import tilelang

    snapshots = {}
    retained = {"tl.AscendLowerMixedPartition", "tl.InsertSync",
                "tl.LowerScheduledTIR", "tl.AscendLowerTileOp"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in retained:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(program, target={"kind": "ascend", "arch": "dav-2201"},
                                  target_host="c", enable_host_codegen=False,
                                  enable_device_compile=False)
    assert snapshots.keys() == retained
    return artifact.kernel_source, snapshots


def _decode_completion_controls(mod, *, core, block, native_hb=False):
    """Project finite copies/collectives, preserving their actual controls.

    The existing trace interprets integer control and explicit counter state.
    Omit opaque vector math and its scalar values; never add or move a copy,
    completion, guard, loop or counter update. This checks placement and finite
    fill/drain, not numerical vector execution or hardware completion timing.
    """
    from testing.ascend.auto_schedule.test_owner_result_completion import (
        _COPY, _CROSS_SET, _CROSS_WAIT,
    )

    def project(node):
        if isinstance(node, tirx.SBlock) and node.name_hint in ("CUBE", "VECTOR"):
            if node.name_hint != ("CUBE" if core == 2 else "VECTOR"):
                return tirx.SBlock([], [], [], node.name_hint, tirx.Evaluate(0))
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "thread_extent":
            if isinstance(node.node, tirx.IterVar) and node.node.thread_tag == "blockIdx.x":
                return tirx.stmt_functor.substitute(
                    node.body, {node.node.var: tirx.const(block, node.node.var.dtype)})
        if isinstance(node, tirx.Bind) and not str(node.var.dtype).startswith(("bool", "int", "uint")):
            return tirx.Evaluate(0)
        if isinstance(node, (tirx.BufferStore, tirx.AllocBuffer)) and node.buffer.scope() != "local.var":
            return tirx.Evaluate(0)
        if isinstance(node, tirx.Evaluate):
            call = node.value
            if not isinstance(call, tirx.Call) or not isinstance(call.op, tvm.ir.Op) or (
                    not native_hb and call.op.name not in (_COPY, _CROSS_SET, _CROSS_WAIT)):
                return tirx.Evaluate(0)

    func = mod["main"]
    body = tirx.stmt_functor.ir_transform(func.body, None, project)
    return tvm.IRModule({"main": func.with_body(body)})


def _grouped_native_completion_commands(mod, block):
    """Retain real commands; track only native L1/L0 storage effects.

    Vector arithmetic is not numerically interpreted. Its real commands,
    local fences and mode-2 handoffs remain in the completion graph, because
    a C retirement path can run through Score -> AIV -> P -> PV.
    """
    from testing.ascend.auto_schedule.test_nested_local_protocols import _Command
    from testing.ascend.auto_schedule.test_owner_result_completion import (
        _COPY, _CROSS_SET, _CROSS_WAIT, _name, _trace,
    )

    def cells(argument, env):
        buffer, origin, shape = _region(argument, env)
        if not buffer.scope().startswith(("shared.l1", "shared.l0a", "shared.l0b", "shared.l0c")):
            return ()
        # Exact logical elements preserve partial FIX rows and L1 padding;
        # rounding a one-row tail to a full fractal would invent overlaps.
        assert all(extent >= 0 for extent in shape) and np.prod(shape) <= 4096
        return tuple((buffer.data, *indices) for indices in product(
            *(range(start, start + extent) for start, extent in zip(origin, shape))))

    result = []
    for core, lane in ((2, 0), (1, 0), (1, 1)):
        projected = _decode_completion_controls(mod, core=core, block=block, native_hb=True)
        for call, env, _ in _trace(projected, core=core, lane=lane):
            name = _name(call)
            flag, reads, writes = None, (), ()
            if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                source, destination = call.args[0].value.split("_")
                kind = "set" if name == "tl.ascend_set_flag" else "wait"
                pipe = source if kind == "set" else destination
                flag = (call.args[0].value, _value(call.args[1], env))
                assert 0 <= flag[1] < 8
            elif name in (_CROSS_SET, _CROSS_WAIT):
                assert _value(call.args[0], env) == 2, "This grouped 2201 fixture uses collective mode 2"
                kind = "cross_set" if name == _CROSS_SET else "cross_wait"
                # A2 wait_flag_dev ignores the nominal pipe argument and
                # blocks subsequent dispatch on this core/lane, not work
                # which has already been issued to other queues.
                pipe = call.args[1].value.removeprefix("PIPE_") if kind == "cross_set" else "S"
                flag = (2, _value(call.args[2], env))
            elif name == "tl.ascend_pipe_barrier":
                pipe, kind = call.args[0].value.removeprefix("PIPE_"), "barrier"
            elif name in (_COPY, "tl.tileop.ascend_cube_load3d"):
                source, _, _ = _region(call.args[0], env)
                destination, _, _ = _region(call.args[1], env)
                pipe = ("MTE2" if source.scope() == "global" else
                        "MTE1" if destination.scope().startswith(("shared.l0a", "shared.l0b")) else
                        "FIX" if source.scope().startswith("shared.l0c") else "MTE3")
                kind, reads, writes = "copy", cells(call.args[0], env), cells(call.args[1], env)
            elif name == "tl.tileop.fill":
                buffer, _, _ = _region(call.args[0], env)
                pipe, kind, writes = ("MTE2" if buffer.scope().startswith("shared.l1") else "V"), "fill", cells(call.args[0], env)
            elif name == "tl.tileop.gemm":
                assert core == 2
                unit = call.annotations.get("unit_flag_ctrl")
                assert unit is None or _value(unit, env) == 0, "Unit readiness is not whole-MAD retirement"
                pipe, kind = "M", "mad"
                reads = (*cells(call.args[0], env), *cells(call.args[1], env))
                if not _value(call.args[9], env):
                    reads += cells(call.args[2], env)
                writes = cells(call.args[2], env)
            elif name.startswith("tl.ascend_membase."):
                assert core == 1
                pipe, kind = "V", "vector"
            else:
                assert name in ("tl.ascend_set_load3d_config", "tl.ascend_set_fixpipe_config",
                                "tl.ascend_set_mmad_direction"), name
                pipe, kind = "S", "scalar"
            result.append(((core, lane), _Command(pipe, kind, flag, reads, writes)))
    return result


def _prove_grouped_native_completions(commands):
    """Finite issue/start/done graph using the existing native oracle rules.

    Each track dispatches independently. Only S/M have implicit completion
    order; Vector, DMA and FIX need actual source fences/destination gates.
    Mode 2 pairs one Cube publication with each AIV, and both AIV releases
    with one Cube acquire. Matching counts add no extra completion edges.
    This finite fixture also proves consume-before-next-notify; that is a
    sufficient bound, not a general requirement of mode-2 counting events.
    """
    parents, pending, gates, issue_tail, implicit_tail = {}, defaultdict(set), {}, {}, {}
    local_sets, local_waits, cross_sets, cross_waits = (defaultdict(list) for _ in range(4))
    pipes = ("S", "M", "FIX", "MTE1", "MTE2", "MTE3", "V")
    for index, (track, command) in enumerate(commands):
        issue, start, done = 3 * index, 3 * index + 1, 3 * index + 2
        parents[issue] = {issue_tail[track]} if track in issue_tail else set()
        parents[start], parents[done] = {issue}, {start}
        for pipe in pipes if command.pipe == "ALL" else (command.pipe,):
            key = track, pipe
            if key in gates:
                parents[start].add(gates[key])
            if pipe in ("S", "M") and key in implicit_tail:
                parents[start].add(implicit_tail[key])
            if command.kind in ("set", "cross_set", "barrier"):
                parents[done].update(pending[key])
                pending[key].clear()
            pending[key].add(done)
            if command.kind in ("wait", "barrier"):
                gates[key] = done
            if pipe in ("S", "M"):
                implicit_tail[key] = done
        # Destination-S local waits and A2 cross-core waits stop later
        # dispatch. An ordinary asynchronous acquire does not do so.
        issue_tail[track] = done if command.kind == "cross_wait" or (
            command.kind == "wait" and command.pipe == "S") else issue
        if command.kind in ("set", "wait"):
            (local_sets if command.kind == "set" else local_waits)[track, command.flag].append(done)
        if command.kind in ("cross_set", "cross_wait"):
            (cross_sets if command.kind == "cross_set" else cross_waits)[track, command.flag].append(done)

    reuse = []
    assert local_sets.keys() == local_waits.keys()
    for key, notifications in local_sets.items():
        acquisitions = local_waits[key]
        assert len(notifications) == len(acquisitions), (key, "Unbalanced local binary event")
        for notify, acquire in zip(notifications, acquisitions):
            parents[acquire].add(notify)
        reuse.extend(zip(acquisitions, notifications[1:]))
    flags = {flag for _, flag in cross_sets.keys() | cross_waits.keys()}
    for flag in flags:
        cube_sets, cube_waits = cross_sets[(2, 0), flag], cross_waits[(2, 0), flag]
        for lane in (0, 1):
            vector_sets, vector_waits = cross_sets[(1, lane), flag], cross_waits[(1, lane), flag]
            assert len(cube_sets) == len(vector_waits) and len(vector_sets) == len(cube_waits)
            for notify, acquire in (*zip(cube_sets, vector_waits), *zip(vector_sets, cube_waits)):
                parents[acquire].add(notify)
            reuse.extend(zip(vector_waits, cube_sets[1:]))
            reuse.extend(zip(cube_waits, vector_sets[1:]))

    children, remaining = defaultdict(list), {node: len(incoming) for node, incoming in parents.items()}
    for node, incoming in parents.items():
        for predecessor in incoming:
            children[predecessor].append(node)
    ready, rank = deque(node for node, degree in remaining.items() if not degree), {}
    while ready:
        node = ready.popleft()
        rank[node] = len(rank)
        for child in children[node]:
            remaining[child] -= 1
            if not remaining[child]:
                ready.append(child)
    assert len(rank) == len(parents), "Grouped native completion graph deadlocks"
    proven = set()

    def require(before, after, *, obligation="data"):
        if (before, after) in proven:
            return
        frontier, seen = [after], set()
        while frontier:
            node = frontier.pop()
            if node == before:
                proven.add((before, after))
                return
            if node not in seen and rank[node] > rank[before]:
                seen.add(node)
                frontier.extend(parents[node])
        source_track, source = commands[before // 3]
        destination_track, destination = commands[after // 3]
        raise AssertionError(
            f"Missing grouped {obligation} completion path "
            f"{before // 3}:{source_track}:{source.kind}/{source.pipe} -> "
            f"{after // 3}:{destination_track}:{destination.kind}/{destination.pipe}")

    writers, readers, hazards = {}, defaultdict(set), Counter()
    for index, (track, command) in enumerate(commands):
        for cell in command.reads:
            if cell in writers:
                require(3 * writers[cell] + 2, 3 * index + 1)
            readers[cell].add(index)
        for cell in command.writes:
            if cell in writers:
                require(3 * writers[cell] + 2, 3 * index + 1)
            for reader in readers[cell]:
                if reader != index:
                    require(3 * reader + 2, 3 * index + 1)
                    hazards[commands[reader][1].pipe, command.pipe, cell[0]] += 1
            writers[cell] = index
            readers[cell].clear()
    for consumed, reset in reuse:
        require(consumed, reset, obligation="event reuse")
    return hazards


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
def test_grouped_decode_native_shared_storage_completion(example, variant):
    """Real grouped FA, including slot wrap, empty AIV and absent q1.

    Five causal KV epochs exceed the three-stage bound. Block 0 has both
    queries; block 1 has one row in q0 (AIV1 empty) and an absent q1. The
    graph covers actual L1/L0 regions and explicit completion, not UB math,
    final arena aliasing or numerical accuracy.
    """
    from testing.ascend.auto_schedule.test_owner_result_completion import _COPY

    program = example.flash_attention_fwd(
        1, 33, 1, 1, 130, 16, dtype="float16", variant=variant,
        causal=True, query_start=65, tiling=example.FwdTiling(16, 16, 3),
        query_grouping=2,
    )
    _, snapshots = _native_decode_lower(program)
    lowered = snapshots["tl.LowerScheduledTIR"]
    func = lowered["main"]
    q, k, v, _ = [func.buffer_map[param] for param in func.params]
    imports = [call for call in calls(lowered, _COPY)
               if call.args[0].args[0].buffer.scope() == "global"
               and call.args[1].args[0].buffer.scope().startswith("shared.l1")]
    kv_storage = {
        next(call.args[1].args[0].buffer.data for call in imports
             if call.args[0].args[0].buffer.data.same_as(source.data))
        for source in (k, v)
    }
    p_storage = {call.args[1].args[0].buffer.data for call in imports
                 if not any(call.args[0].args[0].buffer.data.same_as(source.data)
                            for source in (q, k, v))}
    assert len(kv_storage) == 2 and len(p_storage) == 1
    c_storage = {call.args[2].args[0].buffer.data for call in calls(lowered, "tl.tileop.gemm")}
    assert len(c_storage) == 2
    actual_commands = []
    for block, active_queries in ((0, 2), (1, 1)):
        commands = _grouped_native_completion_commands(lowered, block)
        hazards = _prove_grouped_native_completions(commands)
        for storage in kv_storage | p_storage:
            assert hazards["MTE1", "MTE2", storage] > 0, "Must exercise real shared-L1 reuse"
            assert sum(command.pipe == "MTE1" and any(cell[0].same_as(storage) for cell in command.reads)
                       for _, command in commands) == 5 * active_queries
        for storage in c_storage:
            assert hazards["FIX", "M", storage] > 0, "Must retire FIX before reusing shared L0C"
        assert sum(command.kind == "mad" for _, command in commands) == 10 * active_queries
        actual_commands.append(commands)

    # Retain every event/token and every data operation, but publish one
    # Score result before its FIX is issued. Matching mode-2 counts still
    # hold; the real cross-track C-retirement path must now fail.
    commands = actual_commands[0]
    notify, = [index for index, (track, command) in enumerate(commands)
               if track == (2, 0) and command.kind == "cross_set" and command.pipe == "FIX"][:1]
    fix = max(index for index in range(notify)
              if commands[index][0] == (2, 0) and commands[index][1].kind == "copy"
              and commands[index][1].pipe == "FIX")
    premature = commands[:fix] + [commands[notify]] + commands[fix:notify] + commands[notify + 1:]
    def tokens(trace):
        return Counter((track, command.kind, command.flag) for track, command in trace
                       if command.kind in ("set", "wait", "cross_set", "cross_wait"))

    assert tokens(premature) == tokens(commands)
    with pytest.raises(AssertionError, match="Missing grouped data completion path .*copy/FIX -> .*mad/M"):
        _prove_grouped_native_completions(premature)


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("case", ["default-four", "causal-head-tail", "causal-one-key-prefix",
                                  "grouped-causal-head-tail", "grouped-absent-second"])
def test_native_decode_regions_and_empty_aiv_collective_fill_drain(example, variant, case):
    from testing.ascend.auto_schedule.test_owner_result_completion import (
        _COPY, _CROSS_SET, _CROSS_WAIT, _name, _trace,
    )

    query_grouping = 2 if case.startswith("grouped-") else 1
    if case == "default-four":
        batch, kv_heads, groups, kv_len, dim, width, trips = 1, 1, 4, 4096, 64, 256, 16
        program = example.flash_attention_fwd(batch, groups, kv_heads, 1, kv_len, dim,
                                             variant=variant)
    elif case in ("causal-head-tail", "grouped-causal-head-tail", "grouped-absent-second"):
        groups = 33 if case == "grouped-absent-second" else 17
        batch, kv_heads, kv_len, dim, width, trips = 2, 2, 35, 16, 16, 2
        program = example.flash_attention_fwd(
            batch, kv_heads * groups, kv_heads, 1, kv_len, dim, dtype="float16",
            variant=variant, causal=True, query_start=17, tiling=example.FwdTiling(16, 16, 3),
            query_grouping=query_grouping,
        )
    else:
        batch, kv_heads, groups, kv_len, dim, width, trips = 2, 2, 17, 130, 64, 64, 2
        program = example.flash_attention_fwd(
            batch, kv_heads * groups, kv_heads, 1, kv_len, dim, dtype="float16",
            variant=variant, causal=True, query_start=64, tiling=example.FwdTiling(16, 64, 4),
        )
    source, snapshots = _native_decode_lower(program)
    partition = snapshots["tl.AscendLowerMixedPartition"]
    func = partition["main"]
    q, _, _, output = [func.buffer_map[param] for param in func.params]
    threads = {stmt.node.thread_tag: stmt.node.var for stmt in nodes(partition, tirx.AttrStmt)
               if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tirx.IterVar)}
    definitions = {binding.var: binding.value for binding in nodes(partition, tirx.Bind)}
    copies = calls(partition, _COPY)
    q_loads = [call for call in copies if call.args[0].args[0].buffer.data.same_as(q.data)]
    exports = [call for call in copies if call.args[1].args[0].buffer.data.same_as(output.data)]
    fixes = [call for call in copies if call.args[0].args[0].buffer.scope() == "shared.l0c"
             and call.args[1].args[0].buffer.scope() == "global"]
    p_spills = [call for call in copies if call.args[0].args[0].buffer.scope() == "shared.dyn"
                and call.args[1].args[0].buffer.scope() == "global"
                and not call.args[1].args[0].buffer.data.same_as(output.data)]
    assert len(q_loads) == len(exports) == len(p_spills) == query_grouping
    assert len(fixes) == 2 * query_grouping
    query_workspaces = [
        [call.args[1].args[0].buffer.data
         for call in (fixes[2 * index], fixes[2 * index + 1], p_spills[index])]
        for index in range(query_grouping)
    ]
    # Reusing L0C and L1 P must not merge the separate valid-row components or
    # their GM publications. Every logical handoff retains its own workspace.
    assert len({data for workspace in query_workspaces for data in workspace}) == 3 * query_grouping
    gemms = calls(partition, "tl.tileop.gemm")
    assert len(gemms) == 2 * query_grouping and all(int(call.args[5]) == 16 for call in gemms)
    native_mads = calls(snapshots["tl.AscendLowerTileOp"], "tl.ascend_mad")
    assert len(native_mads) == 2 * query_grouping and all(int(call.args[3]) == 16 for call in native_mads)
    # Keep generated-code assertions at the native API boundary, not offsets,
    # flag numbers, task order or a complete golden source.
    for api in ("asc_mmad(", "asc_copy_l0c2gm(", "asc_copy_gm2ub_align(",
                "asc_copy_ub2gm_align(", "asc_sync_block_arrive(", "asc_sync_block_wait("):
        assert api in source
    assert calls(snapshots["tl.InsertSync"], _CROSS_SET)
    assert calls(snapshots["tl.InsertSync"], _CROSS_WAIT)

    covered, nonempty_lanes, absent_queries, empty_owners = Counter(), set(), set(), set()
    expected = [(b, h, start)
                for b, h in product(range(batch), range(kv_heads))
                for start in range(0, groups, 16 * query_grouping)]
    for block, (b, kv_head, head_offset) in enumerate(expected):
        base = {_PURE_DEFINITIONS: definitions, _SCALAR_VALUES: {},
                threads["blockIdx.x"]: tirx.const(block, threads["blockIdx.x"].dtype)}
        query_counts = [min(16, max(0, groups - head_offset - index * 16))
                        for index in range(query_grouping)]
        traces = {}
        for core, lane in ((2, 0), (1, 0), (1, 1)):
            projected = _decode_completion_controls(snapshots["tl.LowerScheduledTIR"], core=core, block=block)
            traces[core, lane] = list(_trace(projected, core=core, lane=lane))
        # Simplify may use the presence guard to replace an active second
        # query's Q/FIX extent by 16. Outside that guard its raw expression
        # need not evaluate to zero. Trace the real controls first, then check
        # regions only for operations that actually execute in this block.
        cube_q_loads, cube_fixes = Counter(), Counter()
        for call, actual, _ in traces[2, 0]:
            if _name(call) != _COPY:
                continue
            src, src_origin, src_shape = _region(call.args[0], actual)
            dst, dst_origin, dst_shape = _region(call.args[1], actual)
            if src.data.same_as(q.data):
                index, = [index for index, q_load in enumerate(q_loads)
                          if dst.data.same_as(q_load.args[1].args[0].buffer.data)]
                count = query_counts[index]
                assert count > 0, "An absent query must execute no Q import"
                head_start = kv_head * groups + head_offset + index * 16
                assert src_origin == (b, head_start, 0) and src_shape == (1, count, dim)
                cube_q_loads[index] += 1
            for index, workspace in enumerate(query_workspaces):
                for handoff, columns in enumerate((width, dim)):
                    if dst.data.same_as(workspace[handoff]):
                        count = query_counts[index]
                        assert count > 0, "An absent query must execute no Cube publication"
                        assert src.scope() == "shared.l0c"
                        assert src_origin[-2:] == dst_origin[-2:] == (0, 0)
                        assert src_shape[-2:] == dst_shape[-2:] == (count, columns)
                        cube_fixes[index, handoff] += 1
        for index, count in enumerate(query_counts):
            assert cube_q_loads[index] == int(count > 0)
            for handoff in (0, 1):
                assert cube_fixes[index, handoff] == (trips if count else 0)
            if not count:
                absent_queries.add(index)
        transfers = Counter()
        for lane in (0, 1):
            env = {**base, threads["cthread"]: tirx.const(lane, threads["cthread"].dtype)}
            owner_shapes = []
            for index, (export, count) in enumerate(zip(exports, query_counts)):
                if not count:
                    # No region is evaluated under a false presence guard.
                    # Actual no-copy/no-export behavior is checked below.
                    owner_shapes.append(None)
                    continue
                head_start = kv_head * groups + head_offset + index * 16
                view, origin, shape = _region(export.args[1], env)
                owner_shapes.append(shape)
                assert view.data.same_as(output.data) and tuple(map(int, view.shape)) == (batch, kv_heads * groups, dim)
                assert shape[0] == 1 and shape[2] == dim and 0 <= shape[1] <= 8
                assert origin[0] == b and origin[2] == 0
                assert head_start <= origin[1] <= head_start + count
                assert origin[1] + shape[1] <= head_start + count
                if shape[1]:
                    nonempty_lanes.add(lane)
                elif count:
                    empty_owners.add((index, lane))
            exported = Counter()
            for call, actual, _ in traces[1, lane]:
                if _name(call) != _COPY:
                    continue
                src, _, src_shape = _region(call.args[0], actual)
                dst, _, dst_shape = _region(call.args[1], actual)
                for index, workspace in enumerate(query_workspaces):
                    for handoff, data in enumerate(workspace):
                        if src.data.same_as(data) or dst.data.same_as(data):
                            assert query_counts[index] > 0, "An absent query must execute no AIV handoff copy"
                            assert np.prod(src_shape) == np.prod(dst_shape)
                            if not owner_shapes[index][1]:
                                assert np.prod(src_shape) == 0, "An empty AIV must transfer no handoff payload"
                            transfers[index, handoff, lane] += int(np.prod(src_shape))
                if dst.data.same_as(output.data):
                    index, = [index for index, export in enumerate(exports)
                              if src.data.same_as(export.args[0].args[0].buffer.data)]
                    assert query_counts[index] > 0, "An absent query must execute no output export"
                    assert tuple(dst_shape) == tuple(owner_shapes[index])
                    _, origin, _ = _region(call.args[1], actual)
                    covered.update((origin[0], head) for head in range(origin[1], origin[1] + dst_shape[1]))
                    exported[index] += 1
                if not any(shape is not None and shape[1] for shape in owner_shapes) and (src.scope() == "global" or dst.scope() == "global"):
                    assert np.prod(src_shape) == np.prod(dst_shape) == 0
            for index, shape in enumerate(owner_shapes):
                for handoff, columns in enumerate((width, dim, width)):
                    expected_elements = shape[1] * columns * trips if shape is not None else 0
                    assert transfers[index, handoff, lane] == expected_elements
                assert exported[index] == int(query_counts[index] > 0), "Absent queries must execute no export"

        def events(core, lane, operation):
            return Counter((_value(call.args[0], actual), _value(call.args[2], actual))
                           for call, actual, _ in traces[core, lane] if _name(call) == operation)

        # A Cube notification is acquired by each AIV; an AIV collective is
        # issued even by the empty lane. Counts include actual seed/drain, so
        # a tail-specific missing arrive or an unconsumed final token fails.
        cube_sets, cube_waits = events(2, 0, _CROSS_SET), events(2, 0, _CROSS_WAIT)
        assert cube_sets and cube_waits
        for lane in (0, 1):
            assert events(1, lane, _CROSS_WAIT) == cube_sets
            assert events(1, lane, _CROSS_SET) == cube_waits
        # The finite counters above include seeds and drains on every lane.
        # Automatic T.Pipelined may select a same-epoch schedule without a
        # carried result slot; correctness does not require a root-level wait.
    assert covered == Counter({head: 1 for head in product(range(batch), range(kv_heads * groups))})
    assert nonempty_lanes == ({0} if case == "default-four" else {0, 1})
    if case == "grouped-causal-head-tail":
        assert (1, 1) in empty_owners, "The present second query must exercise its empty AIV"
    if case == "grouped-absent-second":
        assert absent_queries == {1} and (0, 1) in empty_owners
