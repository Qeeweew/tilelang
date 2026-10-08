"""Decode head packing preserves BNSD storage and ordinary attention math.

These are frontend region and CPU mathematical checks. Native mixed-core
completion, CANN compilation and device accuracy remain separate gates.
"""

from collections import Counter
from itertools import product

import numpy as np
import pytest

from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes, statements
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _PURE_DEFINITIONS, _SCALAR_VALUES, _region, _value,
)
from testing.ascend.language.test_flash_attention_910b import example


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


def _decode_completion_controls(mod, *, core, block):
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
            if not isinstance(call, tirx.Call) or not isinstance(call.op, tvm.ir.Op) or call.op.name not in (
                    _COPY, _CROSS_SET, _CROSS_WAIT):
                return tirx.Evaluate(0)

    func = mod["main"]
    body = tirx.stmt_functor.ir_transform(func.body, None, project)
    return tvm.IRModule({"main": func.with_body(body)})


@pytest.mark.parametrize("variant", ["generic", "pipeline"])
@pytest.mark.parametrize("case", ["default-four", "causal-head-tail", "causal-one-key-prefix"])
def test_native_decode_regions_and_empty_aiv_collective_fill_drain(example, variant, case):
    from testing.ascend.auto_schedule.test_owner_result_completion import (
        _COPY, _CROSS_SET, _CROSS_WAIT, _name, _trace,
    )

    if case == "default-four":
        batch, kv_heads, groups, kv_len, dim, width, trips = 1, 1, 4, 4096, 64, 256, 16
        program = example.flash_attention_fwd(batch, groups, kv_heads, 1, kv_len, dim,
                                             variant=variant)
    elif case == "causal-head-tail":
        batch, kv_heads, groups, kv_len, dim, width, trips = 2, 2, 17, 35, 16, 16, 2
        program = example.flash_attention_fwd(
            batch, kv_heads * groups, kv_heads, 1, kv_len, dim, dtype="float16",
            variant=variant, causal=True, query_start=17, tiling=example.FwdTiling(16, 16, 3),
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
    q_load, = [call for call in copies if call.args[0].args[0].buffer.data.same_as(q.data)]
    export, = [call for call in copies if call.args[1].args[0].buffer.data.same_as(output.data)]
    fixes = [call for call in copies if call.args[0].args[0].buffer.scope() == "shared.l0c"
             and call.args[1].args[0].buffer.scope() == "global"]
    p_spill, = [call for call in copies if call.args[0].args[0].buffer.scope() == "shared.dyn"
                and call.args[1].args[0].buffer.scope() == "global"
                and not call.args[1].args[0].buffer.data.same_as(output.data)]
    assert len(fixes) == 2
    workspaces = [call.args[1].args[0].buffer.data for call in (*fixes, p_spill)]
    gemms = calls(partition, "tl.tileop.gemm")
    assert len(gemms) == 2 and all(int(call.args[5]) == 16 for call in gemms)
    native_mads = calls(snapshots["tl.AscendLowerTileOp"], "tl.ascend_mad")
    assert len(native_mads) == 2 and all(int(call.args[3]) == 16 for call in native_mads)
    # Keep generated-code assertions at the native API boundary, not offsets,
    # flag numbers, task order or a complete golden source.
    for api in ("asc_mmad(", "asc_copy_l0c2gm(", "asc_copy_gm2ub_align(",
                "asc_copy_ub2gm_align(", "asc_sync_block_arrive(", "asc_sync_block_wait("):
        assert api in source
    assert calls(snapshots["tl.InsertSync"], _CROSS_SET)
    assert calls(snapshots["tl.InsertSync"], _CROSS_WAIT)

    covered, nonempty_lanes = Counter(), set()
    expected = [(b, h, start, min(16, groups - start))
                for b, h in product(range(batch), range(kv_heads))
                for start in range(0, groups, 16)]
    for block, (b, kv_head, head_offset, count) in enumerate(expected):
        head_start = kv_head * groups + head_offset
        base = {_PURE_DEFINITIONS: definitions, _SCALAR_VALUES: {},
                threads["blockIdx.x"]: tirx.const(block, threads["blockIdx.x"].dtype)}
        q_view, q_origin, q_shape = _region(q_load.args[0], base)
        assert q_view.data.same_as(q.data) and q_origin == (b, head_start, 0)
        assert q_shape == (1, count, dim)
        for fix in fixes:
            _, src_origin, src_shape = _region(fix.args[0], base)
            _, dst_origin, dst_shape = _region(fix.args[1], base)
            assert src_origin[-2:] == dst_origin[-2:] == (0, 0)
            assert src_shape[-2:] == dst_shape[-2:] == (count, width if fix.same_as(fixes[0]) else dim)

        traces = {}
        for core, lane in ((2, 0), (1, 0), (1, 1)):
            projected = _decode_completion_controls(snapshots["tl.LowerScheduledTIR"], core=core, block=block)
            traces[core, lane] = list(_trace(projected, core=core, lane=lane))
        transfers = Counter()
        for lane in (0, 1):
            env = {**base, threads["cthread"]: tirx.const(lane, threads["cthread"].dtype)}
            view, origin, shape = _region(export.args[1], env)
            assert view.data.same_as(output.data) and tuple(map(int, view.shape)) == (batch, kv_heads * groups, dim)
            assert shape[0] == 1 and shape[2] == dim and 0 <= shape[1] <= 8
            assert origin[0] == b and origin[2] == 0
            assert head_start <= origin[1] <= head_start + count
            assert origin[1] + shape[1] <= head_start + count
            for head in range(origin[1], origin[1] + shape[1]):
                covered[b, head] += 1
            if shape[1]:
                nonempty_lanes.add(lane)
            for call, actual, _ in traces[1, lane]:
                if _name(call) != _COPY:
                    continue
                src, _, src_shape = _region(call.args[0], actual)
                dst, _, dst_shape = _region(call.args[1], actual)
                for index, data in enumerate(workspaces):
                    if src.data.same_as(data) or dst.data.same_as(data):
                        assert np.prod(src_shape) == np.prod(dst_shape)
                        transfers[index, lane] += int(np.prod(src_shape))
                if dst.data.same_as(output.data):
                    assert tuple(dst_shape) == tuple(shape)
                if not shape[1] and (src.scope() == "global" or dst.scope() == "global"):
                    assert np.prod(src_shape) == np.prod(dst_shape) == 0
            for index, columns in enumerate((width, dim, width)):
                assert transfers[index, lane] == shape[1] * columns * trips

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
        if variant == "pipeline":
            assert any(_name(call) == _CROSS_WAIT and not loops
                       for call, _, loops in traces[2, 0]), "Carried result slots need a final drain"
    assert covered == Counter({head: 1 for head in product(range(batch), range(kv_heads * groups))})
    assert nonempty_lanes == ({0} if case == "default-four" else {0, 1})
