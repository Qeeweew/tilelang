"""Reuse one native A tile across N in a closed physical M/N GEMM pair.

CPU tests replay actual typed operands, guards and finite local flags. They
do not equate issuing a MAD or flag with hardware completion; native queue
completion and performance require the separate AscendC acceptance checks.
"""

from collections import Counter
import math

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0c_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq


_SCALARS = object()
_DEFINITIONS = object()


def _value(expr, env):
    def scalar_load(node):
        if isinstance(node, tirx.BufferLoad):
            assert node.buffer.scope() == "local.var"
            assert all(int(i) == 1 for i in node.buffer.shape)
            assert all(_value(i, env) == 0 for i in node.indices)
            assert node.buffer.data in env[_SCALARS]
            return tirx.const(env[_SCALARS][node.buffer.data], node.dtype)

    resolved = tirx.stmt_functor.ir_transform(tirx.Evaluate(expr), None, scalar_load).value

    def resolve(var, active=()):
        if var in env:
            return env[var]
        if var not in env[_DEFINITIONS]:
            return var
        assert all(not var.same_as(other) for other in active)
        rhs = env[_DEFINITIONS][var]
        return tirx.stmt_functor.substitute(rhs, {
            dependency: resolve(dependency, (*active, var)) for dependency in nodes(rhs, tirx.Var)})

    resolved = tirx.stmt_functor.substitute(resolved, {var: resolve(var) for var in nodes(resolved, tirx.Var)})
    result = tvm.arith.Analyzer().simplify(resolved)
    assert isinstance(result, tirx.IntImm), f"Unbound fixture expression: {result}"
    return int(result)


def _trace(stmt, env=None):
    """Execute finite scalar control; mutable counters retain actual state."""
    env = {_SCALARS: {}, _DEFINITIONS: {}} if env is None else env
    if isinstance(stmt, tvm.IRModule):
        for binding in nodes(stmt, tirx.Bind):
            if not nodes(binding.value, (tirx.BufferLoad, tirx.Call)):
                env[_DEFINITIONS][binding.var] = binding.value
        yield from _trace(stmt["main"].body, env)
    elif isinstance(stmt, tirx.SeqStmt):
        for child in stmt.seq:
            yield from _trace(child, env)
    elif isinstance(stmt, tirx.SBlockRealize):
        if _value(stmt.predicate, env):
            yield from _trace(stmt.block, env)
    elif isinstance(stmt, tirx.SBlock):
        yield from _trace(stmt.body, env)
    elif isinstance(stmt, tirx.AttrStmt):
        local = env
        if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tirx.IterVar):
            assert int(stmt.value) == 1
            local = {**env, stmt.node.var: tirx.const(0, stmt.node.var.dtype)}
        yield from _trace(stmt.body, local)
    elif isinstance(stmt, tirx.For):
        assert stmt.kind == tirx.ForKind.SERIAL
        start, count = _value(stmt.min, env), _value(stmt.extent, env)
        assert 0 <= count <= 1536
        for coordinate in range(start, start + count):
            yield from _trace(stmt.body, {**env, stmt.loop_var: tirx.const(coordinate, stmt.loop_var.dtype)})
    elif isinstance(stmt, tirx.IfThenElse):
        branch = stmt.then_case if _value(stmt.condition, env) else stmt.else_case
        if branch is not None:
            yield from _trace(branch, env)
    elif isinstance(stmt, tirx.Bind):
        env[stmt.var] = tirx.const(_value(stmt.value, env), stmt.var.dtype)
    elif isinstance(stmt, tirx.BufferStore):
        assert stmt.buffer.scope() == "local.var"
        assert all(int(i) == 1 for i in stmt.buffer.shape)
        assert all(_value(i, env) == 0 for i in stmt.indices)
        assert str(stmt.buffer.dtype).startswith(("int", "uint", "bool"))
        env[_SCALARS][stmt.buffer.data] = int(tirx.const(_value(stmt.value, env), stmt.buffer.dtype))
    elif isinstance(stmt, tirx.Evaluate):
        if isinstance(stmt.value, tirx.Call):
            yield stmt.value, {**env, _SCALARS: dict(env[_SCALARS])}
        else:
            assert _value(stmt.value, env) == 0
    else:
        pytest.fail(f"Unexpected A reuse control: {stmt}")


def _region(buffer, origin, shape):
    return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(i, s) for i, s in zip(origin, shape)])


def _decode(expr, env):
    assert expr.op.name == "tl.region"
    return (expr.args[0].buffer, tuple(_value(i, env) for i in expr.args[0].indices),
            tuple(_value(i, env) for i in expr.args[2:]))


def _fixture(k=16, *, transpose_b=True, active_owners=2, active_epochs=2,
             m=1536, n=48, arch="dav-2201", explicit=False):
    e, owner = tirx.Var("epoch", "int32"), tirx.Var("owner", "int32")
    a = tirx.decl_buffer((2, 2, m + 16, k + 32), "float16", name="a", scope="shared.l1")
    b = tirx.decl_buffer((2, *((n + 16, k + 32) if transpose_b else (k + 32, n + 16))),
                        "float16", name="b", scope="shared.l1")
    c = tirx.decl_buffer((m, n), "float32", name="c", scope="shared.l0c")
    other = tirx.decl_buffer((128, 128), "float32", name="other_c", scope="shared.l0c")
    out = tirx.decl_buffer((3, 2, m + 32, n + 64), "float32", name="out")
    a_region = _region(a, [e % 2, owner, 16, 32], [1, 1, m, k])
    b_region = _region(b, [e % 2, *((16, 32) if transpose_b else (32, 16))],
                       [1, *((n, k) if transpose_b else (k, n))])
    body = seq(tirx.Evaluate(T.gemm(a_region, b_region, c, transpose_B=transpose_b, clear_accum=True)),
               copy(c, out, dst_indices=[e, owner, 16, 32], dst_shape=[1, 1, m, n]))
    body = tirx.IfThenElse(owner < active_owners, body, None)
    body = tirx.For(owner, 0, 2, tirx.ForKind.SERIAL, body)
    body = tirx.For(e, 0, 3, tirx.ForKind.SERIAL, tirx.IfThenElse(e < active_epochs, body, None),
                    annotations={"tl_pipeline_order": [0]} if explicit else {})
    mod = kernel(body, buffers=[a, b, c, other], params=[out], annotations={
        "layout_map": {c.data: make_ascend_l0c_layout(c), other.data: make_ascend_l0c_layout(other)},
        "tl.buffer_versions_map": {other.data: 1}})
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return mod, {"m": m, "n": n, "k": k, "transpose_b": transpose_b,
                 "active_epochs": active_epochs, "active_owners": active_owners}


def _allocations(mod, scope):
    return [buffer for block in nodes(mod, tirx.SBlock) for buffer in block.alloc_buffers if buffer.scope() == scope]


def _replay(mod, facts, *, reuse):
    """Resolve each MAD's current physical writers, then check its GM export."""
    writers, pending, exports = {}, {}, {}
    count = Counter()
    for call, env in _trace(mod):
        if not isinstance(call.op, tvm.ir.Op):
            continue
        if call.op.name == "tl.tileop.ascend_copy":
            src, origin, shape = _decode(call.args[0], env)
            dst, dst_origin, dst_shape = _decode(call.args[1], env)
            if dst.scope() in ("shared.l0a", "shared.l0b"):
                assert src.scope() == "shared.l1"
                assert int(call.args[0].args[1]) == 1 and int(call.args[1].args[1]) == 2
                assert not any(dst_origin)
                writers[dst.data] = (origin, shape)
                count[dst.scope()] += 1
            elif src.scope() == "shared.l0c":
                chain = pending.pop((src.data, origin))
                epoch, owner, row, column = dst_origin
                assert dst_shape[:-2] == (1, 1)
                cursor = 32
                for index, (a, b, clear, dimensions) in enumerate(chain):
                    a_origin, a_shape = a
                    b_origin, b_shape = b
                    rows, columns, chunk_k = dimensions
                    assert a_origin == (epoch % 2, owner, row, cursor)
                    assert a_shape == (1, 1, rows, chunk_k)
                    expected_b = (epoch % 2, column - 16, cursor) if facts["transpose_b"] else (
                        epoch % 2, cursor, column - 16)
                    assert b_origin == expected_b
                    assert b_shape == ((1, columns, chunk_k) if facts["transpose_b"] else (1, chunk_k, columns))
                    assert clear == int(index == 0)
                    cursor += chunk_k
                assert cursor == 32 + facts["k"]
                exports.setdefault((epoch, owner), []).append((*dst_origin[-2:], *dst_shape[-2:]))
        elif call.op.name == "tl.tileop.gemm":
            a, a_origin, _ = _decode(call.args[0], env)
            b, b_origin, _ = _decode(call.args[1], env)
            c, origin, _ = _decode(call.args[2], env)
            assert not any((*a_origin, *b_origin))
            assert a.data in writers and b.data in writers, "MAD reads an uninitialized physical slot"
            clear = _value(call.args[9], env)
            key = (c.data, origin)
            assert not clear or key not in pending
            dimensions = tuple(_value(i, env) for i in call.args[5:8])
            pending.setdefault(key, []).append((writers[a.data], writers[b.data], clear, dimensions))
            count["mad"] += 1
    assert not pending
    expected = {(e, owner) for e in range(facts["active_epochs"]) for owner in range(facts["active_owners"])}
    assert set(exports) == expected
    for tiles in exports.values():
        for row in range(16, facts["m"] + 16):
            spans = sorted((column, column + columns) for start, column, rows, columns in tiles
                           if start <= row < start + rows)
            assert spans[0][0] == 32 and spans[-1][1] == 32 + facts["n"]
            assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        assert sum(rows * columns for _, _, rows, columns in tiles) == facts["m"] * facts["n"]
    assert count["shared.l0b"] == count["mad"]
    if not expected:
        assert not count
    elif reuse:
        assert count["shared.l0a"] < count["mad"]
        # An A fill is required once for each distinct row tile, independently
        # at every owner/epoch restart. No particular tile size is prescribed.
        assert count["shared.l0a"] == sum(len({row for row, _, _, _ in tiles}) for tiles in exports.values())
    else:
        assert count["shared.l0a"] == count["mad"]
    return count


@pytest.mark.parametrize("transpose_b", [True, False], ids=["nt", "nn"])
@pytest.mark.parametrize("k", [16, 14], ids=["aligned-k", "single-tail-k"])
def test_single_k_reuses_a_across_n_preserving_origins_owners_and_odd_tiles(transpose_b, k):
    before, facts = _fixture(k, transpose_b=transpose_b)
    after = transform.AscendLowerL1Gemm()(before)
    count = _replay(after, facts, reuse=True)
    assert count["mad"] % 4 == 0 and count["mad"] // 4 % 2 == 1
    (cached,) = _allocations(after, "shared.l0a")
    root = after["main"].body.block
    assert int(root.annotations["tl.buffer_versions_map"][cached.data]) == 1
    assert cached.data not in root.annotations.get("tl.manual_multi_buffer", {})
    for scope in ("shared.l0a", "shared.l0b"):
        assert sum(math.prod(map(int, b.shape)) * 2 for b in _allocations(after, scope)) <= 64 * 1024


@pytest.mark.parametrize("k", [64, 34], ids=["multi-k", "full-plus-tail-k"])
@pytest.mark.parametrize("transpose_b", [True, False], ids=["nt", "nn"])
def test_multiple_k_chunks_reload_a_for_every_n_and_preserve_accumulation(k, transpose_b):
    before, facts = _fixture(k, transpose_b=transpose_b)
    after = transform.AscendLowerL1Gemm()(before)
    _replay(after, facts, reuse=False)
    versions = after["main"].body.block.annotations["tl.buffer_versions_map"]
    assert all(buffer.data not in versions for buffer in _allocations(after, "shared.l0a"))


@pytest.mark.parametrize("active_epochs,active_owners", [(0, 2), (2, 0), (2, 1)])
def test_empty_and_guarded_owner_domains_do_not_execute_a_cache_load(active_epochs, active_owners):
    before, facts = _fixture(active_epochs=active_epochs, active_owners=active_owners)
    _replay(transform.AscendLowerL1Gemm()(before), facts, reuse=True)


def test_fitting_gemm_has_no_cross_n_cache_or_user_annotation_opt_in():
    before, facts = _fixture(m=128, n=128)
    # A user-authored annotation cannot confer compiler-generated closed-pair
    # ownership. The implementation uses fresh loop Var identities instead.
    body = before["main"].body.block.body
    body = tirx.For(tirx.Var("foreign_tile", "int32"), 0, 1, tirx.ForKind.SERIAL, body,
                    annotations={"tl.ascend_physical_mn_n_tiles": 8})
    root = before["main"].body.block
    before["main"] = before["main"].with_body(tirx.SBlockRealize([], True, tirx.SBlock(
        [], [], [], "tilelang_root", body, alloc_buffers=root.alloc_buffers, annotations=root.annotations)))
    after = transform.AscendLowerL1Gemm()(before)
    _replay(after, facts, reuse=False)
    (a,) = _allocations(after, "shared.l0a")
    assert a.data not in after["main"].body.block.annotations["tl.buffer_versions_map"]


def test_3510_and_explicit_schedule_boundaries_keep_existing_behavior():
    before, _ = _fixture(arch="dav-3510")
    tvm.ir.assert_structural_equal(transform.AscendLowerL1Gemm()(before), before)
    before, _ = _fixture(explicit=True)
    with pytest.raises(ValueError, match="ordinary lexical ownership/completion"):
        transform.AscendLowerL1Gemm()(before)


def _full_program(transpose_b):
    b_shape = (3, 144, 48) if transpose_b else (3, 48, 144)

    @T.prim_func
    def main(a: T.Tensor((2, 1552, 48), "float16"), b: T.Tensor(b_shape, "float16"),
             out: T.Tensor((3, 2, 1568, 112), "float32"), aux: T.Tensor((3, 2, 128, 128), "float32")):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((1552, 48), "float16")
            b_l1 = T.alloc_l1(b_shape[1:], "float16")
            c = T.alloc_l0c((1536, 48), "float32")
            other_c = T.alloc_l0c((128, 128), "float32")
            T.annotate_buffer_versions({a_l1: 1, b_l1: 1, other_c: 1})
            for epoch in T.Pipelined(3):
                if epoch < 2:
                    T.copy(b[epoch, :, :], b_l1)
                    for owner in T.serial(2):
                        T.copy(a[owner, :, :], a_l1)
                        if transpose_b:
                            T.gemm(a_l1[16:1552, 32:48], b_l1[16:64, 32:48], c,
                                   transpose_B=True, clear_accum=True)
                        else:
                            T.gemm(a_l1[16:1552, 32:48], b_l1[32:48, 16:64], c,
                                   transpose_B=False, clear_accum=True)
                        T.copy(c, out[epoch, owner, 16:1552, 32:80])
                        if transpose_b:
                            T.gemm(a_l1[0:128, 32:48], b_l1[0:128, 32:48], other_c,
                                   transpose_B=True, clear_accum=True)
                        else:
                            T.gemm(a_l1[0:128, 32:48], b_l1[32:48, 0:128], other_c,
                                   transpose_B=False, clear_accum=True)
                        T.copy(other_c, aux[epoch, owner, :, :])
    return main


def _check_synced_lifetime(mod, cached_name):
    """Check emitted IR queue order and flags, without inventing RAW edges.

    Each node inherits only the preceding operation on its own queue and an
    actual consumed flag's publisher. This checks the compiler's emitted
    dependency paths; native instruction completion is a separate obligation.
    """
    (cached,) = [buffer for buffer in _allocations(mod, "shared.l0a") if buffer.name == cached_name]
    tokens, published, consumed, queues = {}, Counter(), Counter(), {}
    ancestry = []
    writer, readers = None, []
    a_fills, a_reads, reuses = 0, 0, 0
    for call, env in _trace(mod):
        if not isinstance(call.op, tvm.ir.Op):
            continue
        name = call.op.name
        pipe, event_key, source = None, None, None
        if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
            event_key = (call.args[0].value, _value(call.args[1], env))
            src, dst = event_key[0].split("_")
            pipe = src if name == "tl.ascend_set_flag" else dst
            if name == "tl.ascend_set_flag":
                assert event_key not in tokens, f"Local flag overwritten before consume: {event_key}"
            else:
                assert event_key in tokens, f"Local flag consumed without publish: {event_key}"
                source = tokens.pop(event_key)
        elif name == "tl.tileop.gemm":
            pipe = "M"
        elif name == "tl.tileop.ascend_copy":
            src = call.args[0].args[0].buffer.scope()
            dst = call.args[1].args[0].buffer.scope()
            pipe = {("global", "shared.l1"): "MTE2", ("shared.l1", "shared.l0a"): "MTE1",
                    ("shared.l1", "shared.l0b"): "MTE1", ("shared.l0c", "global"): "FIX"}.get((src, dst))
            assert pipe is not None, f"Unclassified fixture copy: {src} -> {dst}"
        else:
            continue
        previous = [queues[pipe]] if pipe in queues else []
        if source is not None:
            previous.append(source)
        ancestors = set(previous)
        for predecessor in previous:
            ancestors.update(ancestry[predecessor])
        node = len(ancestry)
        ancestry.append(ancestors)
        queues[pipe] = node
        if name == "tl.ascend_set_flag":
            tokens[event_key] = node
            published[event_key] += 1
        elif name == "tl.ascend_wait_flag":
            consumed[event_key] += 1
        elif name == "tl.tileop.ascend_copy" and call.args[1].args[0].buffer.data.same_as(cached.data):
            # The required path must include every previous N reader, also
            # when the physical loop restarts in another owner or epoch.
            assert all(reader in ancestors for reader in readers), "A overwrite lacks its final N reader release"
            writer, readers = node, []
            a_fills += 1
        elif name == "tl.tileop.gemm" and call.args[0].args[0].buffer.data.same_as(cached.data):
            assert writer is not None and writer in ancestors, "MAD lacks A writer readiness"
            reuses += bool(readers)
            readers.append(node)
            a_reads += 1
    assert published and published == consumed and not tokens
    assert any(direction == "M_MTE1" for direction, _ in published)
    assert any(direction == "MTE1_M" for direction, _ in published)
    assert a_fills > 1 and a_reads > a_fills and reuses == a_reads - a_fills


@pytest.mark.parametrize("transpose_b", [True, False], ids=["nt", "nn"])
def test_automatic_full_pipeline_keeps_a_cache_single_version_and_balances_flags(transpose_b):
    snapshots = {}
    wanted = {"tl.AscendLowerL1Gemm", "tl.AnnotateMultiBufferEligible", "tl.AutoSchedule",
              "tl.InsertSync", "tl.MaterializeMultiBuffer"}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        result = tilelang.lower(_full_program(transpose_b).with_attr("global_symbol", "main"),
                                target={"kind": "ascend", "arch": "dav-2201"}, target_host="c",
                                enable_host_codegen=True, enable_device_compile=False)
    assert result.host_mod and set(snapshots) == wanted
    split = snapshots["tl.AscendLowerL1Gemm"]
    # Identify the cached A from its guarded write, not its generated name.
    cached = []
    for branch in nodes(split, tirx.IfThenElse):
        for call in calls(branch.then_case, "tl.tileop.ascend_copy"):
            dst = call.args[1].args[0].buffer
            if dst.scope() == "shared.l0a" and isinstance(branch.condition, tirx.EQ):
                cached.append(dst)
    assert len(cached) == 1
    cached = cached[0]
    original_bytes = math.prod(map(int, cached.shape)) * 2
    for boundary in ("tl.AscendLowerL1Gemm", "tl.AnnotateMultiBufferEligible",
                     "tl.AutoSchedule", "tl.MaterializeMultiBuffer"):
        mod = snapshots[boundary]
        (physical,) = [b for b in _allocations(mod, "shared.l0a") if b.name == cached.name]
        assert math.prod(map(int, physical.shape)) * 2 == original_bytes
        root = next(block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root")
        if boundary in ("tl.AscendLowerL1Gemm", "tl.AnnotateMultiBufferEligible"):
            assert int(root.annotations["tl.buffer_versions_map"][physical.data]) == 1
        else:
            # AutoSchedule outputs counts only for admitted multi-buffer
            # owners; fixed1 excludes this A from that admission. Materialize
            # consumes logical counts and omits unit physical-version entries.
            assert physical.data not in root.annotations.get("tl.buffer_versions_map", {})
            assert physical.data not in root.annotations.get("tl.manual_multi_buffer", {})
        for loop in nodes(mod, tirx.For):
            assert all(not storage.same_as(physical.data)
                       for storage in loop.annotations.get("multi_buffer_eligible", []))
        accesses = [load for load in nodes(mod, tirx.BufferLoad) if load.buffer.data.same_as(physical.data)]
        assert accesses
        for load in accesses:
            assert all(_value(i, {_SCALARS: {}, _DEFINITIONS: {}}) == 0 for i in load.indices)
    _check_synced_lifetime(snapshots["tl.InsertSync"], cached.name)
    assert "asc_copy_l12l0a(" in result.kernel_source and "asc_mmad(" in result.kernel_source
