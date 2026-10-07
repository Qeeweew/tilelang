"""Complete per-AIV query domains retain ordinary regions and native lowering.

These CPU tests exercise the ownership boundary and full source generation.
Source/event assertions are not a device completion or numerical proof.
"""

import math

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout
from tvm import tirx
from tvm.tirx.stmt_functor import ir_transform

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements


DECLARATION = "tl.ascend_aiv_private"
VALIDATED = "tl.ascend_aiv_private_validated"
DIAGNOSTIC = "dav-2201 AIV private domain:"
TARGET = {"kind": "ascend", "arch": "dav-2201"}


def _root(mod):
    return next(block for block in nodes(mod, tirx.SBlock)
                if block.name_hint == "tilelang_root")


def _parallel(body, rows, columns, i, j):
    return tirx.For(i, 0, rows, tirx.ForKind.PARALLEL,
                   tirx.For(j, 0, columns, tirx.ForKind.PARALLEL, body))


def _replace_root(mod, *, body=None, annotations=None):
    def replace(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            return tirx.SBlock(
                node.iter_vars, node.reads, node.writes, node.name_hint,
                node.body if body is None else body, init=node.init,
                alloc_buffers=node.alloc_buffers, match_buffers=node.match_buffers,
                annotations=dict(node.annotations) if annotations is None else annotations,
            )
    return tvm.IRModule({gvar: func.with_body(ir_transform(func.body, None, replace))
                         for gvar, func in mod.functions.items()})


def _fixture(*, declaration=True, separated=True, cthread=2, kind=None):
    rows, width = 128, 64
    sid = tirx.Var("query_owner", "int32")
    shape = (2, rows, width) if separated else (rows, width)
    a, out = [tirx.decl_buffer(shape, "float32", name=name)
              for name in ("Input", "Output")]
    x, y = [tirx.decl_buffer((rows, width), "float32", name=name, scope="shared.dyn")
            for name in ("scores", "result")]
    state = tirx.decl_buffer((2, rows), "float32", name="query_state", scope="shared.dyn")
    ca, cb = [tirx.decl_buffer((16, 16), "float16", name=name, scope="shared.l1")
              for name in ("cube_a", "cube_b")]
    cc = tirx.decl_buffer((16, 16), "float32", name="cube_c", scope="shared.l0c")
    i, j = tirx.Var("query_row", "int32"), tirx.Var("column", "int32")
    read = x
    if kind in ("identical-alias", "rank-alias", "shape-alias"):
        alias_shape = (rows * width,) if kind == "rank-alias" else (64, 128) if kind == "shape-alias" else x.shape
        read = tirx.decl_buffer(alias_shape, x.dtype, name="scores_view", data=x.data, scope=x.scope())
    if kind == "rank-alias":
        read_indices = [i * width + j]
    elif kind == "shape-alias":
        # Keep the alias access in bounds so this tests incompatible storage
        # coordinates rather than an incidental out-of-bounds diagnostic.
        read_indices = [i // 2, j * 2 + i % 2]
    else:
        read_indices = [i, j]
    state_value = state[0, i] if declaration else tirx.const(0, state.dtype)
    math = _parallel(tirx.BufferStore(y, read[tuple(read_indices)] - state_value, [i, j]),
                     rows, width, i, j)
    gm_indices = [sid, 0, 0] if separated else [0, 0]
    gm_shape = [1, rows, width] if separated else [rows, width]
    body = [tirx.Evaluate(T.gemm(ca, cb, cc, transpose_B=True))]
    if declaration:
        body.append(tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"),
                                          [region(state, 2), tirx.const(0, state.dtype)])))
    body.extend([copy(a, x, src_indices=gm_indices, src_shape=gm_shape), math,
                 copy(y, out, dst_indices=gm_indices, dst_shape=gm_shape)])
    if kind == "crossed-writes":
        body.append(copy(y, out, dst_indices=[1 - sid, 0, 0], dst_shape=gm_shape))
    elif kind == "escape":
        body.append(tirx.Evaluate(tirx.call_extern("int32", "foreign_ub_pointer", x.data)))
    elif kind == "scalar-store":
        body.append(tirx.BufferStore(state, tirx.const(1, "float32"), [0, 0]))
    members = [x.data, y.data, state.data]
    if kind == "mixed-component":
        members = [x.data, y.data]
    elif kind == "non-ub":
        members.append(ca.data)
    annotations = {DECLARATION: members} if declaration else {}
    if kind == "custom-layout":
        annotations["layout_map"] = {x.data: Layout(x.shape, lambda row, col: [col, row])}
    mod = kernel(seq(*body), buffers=[ca, cb, cc, x, y, state],
                 params=[a, out, *([sid] if cthread is None else [])],
                 annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    if cthread is not None:
        iv = tirx.IterVar(tvm.ir.Range(0, cthread), sid, tirx.IterVar.ThreadIndex, "cthread")
        mod["main"] = mod["main"].with_body(tirx.AttrStmt(iv, "thread_extent", cthread, mod["main"].body))
    if kind == "non-root":
        root = _root(mod)
        inner = tirx.SBlock([], [], [], "nested_private", root.body, annotations={DECLARATION: members})
        mod = _replace_root(mod, body=tirx.SBlockRealize([], True, inner), annotations={})
    return mod, sid


def _lower(mod):
    return transform.AscendLowerMixedPartition()(mod)


def _constant(expression, replacements=None):
    if replacements:
        expression = tirx.stmt_functor.substitute(expression, replacements)
    return int(tvm.arith.Analyzer().simplify(expression))


def _with_output_strides(mod, strides):
    func = mod["main"]
    output = next(buffer for buffer in func.buffer_map.values() if buffer.name == "Output")
    strided = tirx.decl_buffer(output.shape, output.dtype, name=output.name,
                              data=output.data, strides=strides, scope=output.scope())

    def replace_output(node):
        if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(output):
            return tirx.BufferLoad(strided, node.indices, node.predicate, node.span)

    body = ir_transform(func.body, None, replace_output)
    buffer_map = {data: strided if buffer.same_as(output) else buffer
                  for data, buffer in func.buffer_map.items()}
    return tvm.IRModule({"main": tirx.PrimFunc(
        func.params, body, ret_type=func.ret_type, buffer_map=buffer_map,
        attrs=func.attrs, span=func.span,
    )})


def test_public_declaration_uses_storage_vars_and_deduplicates_identical_members():
    @T.prim_func
    def main():
        with T.MixedKernel(1, sids=2):
            scores = T.alloc_shared((128, 128), "float32")
            alias = T.reshape(scores, (128, 128))
            T.annotate_aiv_private(scores, alias, scores)

    root = _root(main)
    (member,) = root.annotations[DECLARATION]
    assert isinstance(member, tirx.Var)
    assert member.same_as(root.alloc_buffers[0].data)
    assert "tl.manual_multi_buffer" not in root.annotations
    assert "tl.buffer_versions_map" not in root.annotations


@pytest.mark.parametrize("argument,exception", [(None, ValueError), ("scalar", TypeError), ("l1", ValueError)])
def test_public_declaration_checks_buffer_input(argument, exception):
    with pytest.raises(exception, match="annotate_aiv_private"):
        @T.prim_func
        def main():
            with T.MixedKernel(1, sids=2):
                l1 = T.alloc_l1((16, 16), "float16")
                if argument is None:
                    T.annotate_aiv_private()
                elif argument == "scalar":
                    T.annotate_aiv_private(1)
                else:
                    T.annotate_aiv_private(l1)


@pytest.mark.parametrize("kind", [None, "identical-alias"])
def test_complete_query_and_leading_state_slot_keep_shapes_indices_and_regions(kind):
    before, sid = _fixture(kind=kind)
    after = _lower(before)
    private = {buffer.name: buffer for buffer in _root(after).alloc_buffers
               if buffer.scope() == "shared.dyn"}
    assert {name: tuple(map(int, buffer.shape)) for name, buffer in private.items()} == {
        "scores": (128, 64), "result": (128, 64), "query_state": (2, 128),
    }
    assert all(_constant(loop.extent) in (128, 64) for loop in nodes(after, tirx.For))
    row = next(loop for loop in nodes(after, tirx.For) if loop.loop_var.name == "query_row")
    assert _constant(row.min) == 0 and _constant(row.extent) == 128
    math_store = next(store for store in nodes(after, tirx.BufferStore)
                      if store.buffer.name == "result")
    state_load = next(load for load in nodes(math_store.value, tirx.BufferLoad)
                      if load.buffer.name == "query_state")
    assert _constant(state_load.indices[0]) == 0
    assert state_load.indices[1].same_as(row.loop_var)
    members = _root(after).annotations[VALIDATED]
    assert all(any(buffer.data.same_as(member) for member in members) for buffer in private.values())
    exported = next(call.args[1] for call in calls(after, "tl.tileop.ascend_copy")
                    if call.args[1].args[0].buffer.name == "Output")
    assert tuple(map(int, exported.args[2:])) == (1, 128, 64)
    intervals = []
    for owner in (0, 1):
        indices = [_constant(index, {sid: tirx.const(owner, sid.dtype)})
                   for index in exported.args[0].indices]
        assert indices == [owner, 0, 0]
        intervals.extend(range(owner * 128 * 64, (owner + 1) * 128 * 64))
    assert intervals == list(range(2 * 128 * 64))
    assert len([attr for attr in nodes(after, tirx.AttrStmt)
                if attr.attr_key == "thread_extent" and attr.node.thread_tag == "cthread"]) == 1
    assert tvm.ir.save_json(_lower(after)) == tvm.ir.save_json(after)


@pytest.mark.parametrize("kind,options", [
    ("no-cthread", {"cthread": None}),
    ("one-aiv", {"cthread": 1}),
    ("non-root", {}), ("non-ub", {}), ("mixed-component", {}),
    ("unseparated-write", {"separated": False}),
    ("crossed-writes", {}), ("rank-alias", {}), ("shape-alias", {}),
    ("custom-layout", {}), ("escape", {}), ("scalar-store", {}),
])
def test_invalid_private_claim_diagnoses_before_any_partial_rewrite(kind, options):
    before, _ = _fixture(kind=kind, **options)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match=DIAGNOSTIC):
        _lower(before)
    assert tvm.ir.save_json(before) == snapshot


def test_zero_stride_owner_axis_cannot_claim_disjoint_private_gm_writes():
    before, _ = _fixture()
    before = _with_output_strides(before, [0, 64, 1])
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match="(?s)" + DIAGNOSTIC +
                       ".*nonoverlapping compact or padded buffer strides"):
        _lower(before)
    assert tvm.ir.save_json(before) == snapshot


def test_padded_injective_gm_layout_preserves_complete_private_query_writes():
    before, sid = _fixture()
    before = _with_output_strides(before, [128 * 80, 80, 1])
    after = _lower(before)
    exported = next(call.args[1] for call in calls(after, "tl.tileop.ascend_copy")
                    if call.args[1].args[0].buffer.name == "Output")
    output = exported.args[0].buffer
    assert tuple(map(int, output.strides)) == (128 * 80, 80, 1)
    assert output.same_as(next(buffer for buffer in after["main"].buffer_map.values()
                               if buffer.name == "Output"))
    assert tuple(map(int, exported.args[2:])) == (1, 128, 64)
    assert {buffer.name: tuple(map(int, buffer.shape)) for buffer in _root(after).alloc_buffers
            if buffer.scope() == "shared.dyn"} == {
        "scores": (128, 64), "result": (128, 64), "query_state": (2, 128),
    }
    addresses = []
    for owner in (0, 1):
        origin = [_constant(index, {sid: tirx.const(owner, sid.dtype)})
                  for index in exported.args[0].indices]
        assert origin == [owner, 0, 0]
        addresses.extend(sum(index * stride for index, stride in
                             zip((owner, row, col), (128 * 80, 80, 1)))
                         for row in range(128) for col in range(64))
    assert len(set(addresses)) == 2 * 128 * 64
    expected = {owner * 128 * 80 + row * 80 + col
                for owner in (0, 1) for row in range(128) for col in range(64)}
    assert set(addresses) == expected
    assert not set(addresses).intersection(
        owner * 128 * 80 + row * 80 + col
        for owner in (0, 1) for row in range(128) for col in range(64, 80)
    )


def test_private_declaration_is_diagnosed_on_dav3510():
    before, _ = _fixture()
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-3510"}))
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match=DIAGNOSTIC +
                       " AIV private ownership is supported only on dav-2201"):
        _lower(before)
    assert tvm.ir.save_json(before) == snapshot


def test_existing_sid_query_selection_requires_private_declaration():
    before, _ = _fixture(declaration=False)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match="(?s)dav-2201 mixed row partition:.*"
                       "SID-dependent GM regions already select an owner"):
        _lower(before)
    assert tvm.ir.save_json(before) == snapshot


def _gm_roundtrip_fixture(reader):
    mod, sid = _fixture()
    func, root = mod["main"], _root(mod)
    buffers = {buffer.name: buffer for buffer in root.alloc_buffers}
    source, destination = buffers["scores"], buffers["result"]
    output = next(buffer for buffer in func.buffer_map.values() if buffer.name == "Output")
    bridge = tirx.decl_buffer((2, 128, 64), "float32", name="PrivateBridge")
    extra_buffers, extra_params = [], [bridge]
    if reader == "row":
        destination = tirx.decl_buffer((128, 64), "float32", name="ordinary_row", scope="shared.dyn")
        output = tirx.decl_buffer((128, 64), "float32", name="RowOutput")
        extra_buffers.append(destination)
        extra_params.append(output)
    read_owner = sid if reader == "same" else 1 - sid if reader == "crossed" else 0
    i, j = tirx.Var("roundtrip_row", "int32"), tirx.Var("roundtrip_col", "int32")
    roundtrip = [
        copy(source, bridge, dst_indices=[sid, 0, 0], dst_shape=[1, 128, 64]),
        copy(bridge, destination, src_indices=[read_owner, 0, 0], src_shape=[1, 128, 64]),
        _parallel(tirx.BufferStore(destination, destination[i, j] * tirx.const(2, "float32"), [i, j]),
                  128, 64, i, j),
        copy(destination, output, dst_indices=[0, 0] if reader == "row" else [sid, 0, 0],
             dst_shape=[128, 64] if reader == "row" else [1, 128, 64]),
    ]

    def replace_root(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            return tirx.SBlock(
                node.iter_vars, node.reads, node.writes, node.name_hint, seq(node.body, *roundtrip),
                init=node.init, alloc_buffers=[*node.alloc_buffers, *extra_buffers],
                match_buffers=node.match_buffers, annotations=node.annotations,
            )

    body = ir_transform(func.body, None, replace_root)
    buffer_map = dict(func.buffer_map)
    buffer_map.update({buffer.data: buffer for buffer in extra_params})
    mod = tvm.IRModule({"main": tirx.PrimFunc(
        [*func.params, *(buffer.data for buffer in extra_params)], body,
        ret_type=func.ret_type, buffer_map=buffer_map, attrs=func.attrs, span=func.span,
    )})
    return mod, sid


@pytest.mark.parametrize("reader,diagnostic", [
    ("row", "private GM writes cannot share storage with row accesses or opaque escapes"),
    ("crossed", "GM private writes and reads require a common explicit SID-separated axis"),
])
def test_private_written_gm_cannot_be_read_by_the_other_aiv(reader, diagnostic):
    before, _ = _gm_roundtrip_fixture(reader)
    snapshot = tvm.ir.save_json(before)
    with pytest.raises(ValueError, match=DIAGNOSTIC + " " + diagnostic):
        _lower(before)
    assert tvm.ir.save_json(before) == snapshot


def test_private_written_gm_can_be_read_by_the_same_aiv():
    before, sid = _gm_roundtrip_fixture("same")
    after = _lower(before)
    bridge_copies = [call for call in calls(after, "tl.tileop.ascend_copy")
                     if any(call.args[index].args[0].buffer.name == "PrivateBridge"
                            for index in (0, 1))]
    assert len(bridge_copies) == 2
    exported, imported = bridge_copies
    for region_arg in (exported.args[1], imported.args[0]):
        assert tuple(map(int, region_arg.args[2:])) == (1, 128, 64)
        assert [_constant(region_arg.args[0].indices[0], {sid: tirx.const(owner, sid.dtype)})
                for owner in (0, 1)] == [0, 1]
    assert {buffer.name: tuple(map(int, buffer.shape)) for buffer in _root(after).alloc_buffers
            if buffer.scope() == "shared.dyn"} == {
        "scores": (128, 64), "result": (128, 64), "query_state": (2, 128),
    }
    assert all(_constant(loop.extent) in (128, 64) for loop in nodes(after, tirx.For))


def test_ordinary_unannotated_component_retains_automatic_row_partition():
    # This component uses ordinary row-major UB data, without the explicitly
    # private leading query-state slot tested above.
    before, sid = _fixture(declaration=False, separated=False)
    after = _lower(before)
    assert int(_root(after).annotations["tl.ascend_row_partitioned"]) == 2
    assert [int(buffer.shape[0]) for buffer in _root(after).alloc_buffers
            if buffer.name in ("scores", "result")] == [64, 64]
    exported = next(call.args[1] for call in calls(after, "tl.tileop.ascend_copy")
                    if call.args[1].args[0].buffer.name == "Output")
    assert [_constant(exported.args[0].indices[0], {sid: tirx.const(owner, sid.dtype)})
            for owner in (0, 1)] == [0, 64]


def _query_program(dtype):
    @T.prim_func
    def main(Q: T.Tensor((2, 128, 64), dtype), K: T.Tensor((128, 64), dtype),
             V: T.Tensor((128, 64), dtype), Score: T.Tensor((2, 128, 128), "float32"),
             P: T.Tensor((2, 128, 128), dtype), PV: T.Tensor((2, 128, 64), "float32"),
             O: T.Tensor((2, 128, 64), "float32")):
        with T.MixedKernel(1, sids=2) as (group, sid):
            q_l1 = T.alloc_l1((2, 128, 64), dtype)
            k_l1 = T.alloc_l1((128, 64), dtype)
            v_l1 = T.alloc_l1((128, 64), dtype)
            p_l1 = T.alloc_l1((128, 128), dtype)
            score_acc = T.alloc_l0c((128, 128), "float32")
            pv_acc = T.alloc_l0c((128, 64), "float32")
            scores = T.alloc_shared((64, 128), "float32")
            p = T.alloc_shared((64, 128), dtype)
            pv = T.alloc_shared((64, 64), "float32")
            maximum = T.alloc_shared((64,), "float32")
            saved_max = T.alloc_shared((2, 128), "float32")
            T.annotate_aiv_private(scores, p, pv, maximum, saved_max)
            with T.Cube():
                T.copy(K, k_l1)
                T.copy(V, v_l1)
                for owner in T.serial(2):
                    T.copy(Q[owner, :, :], q_l1[owner, :, :])
                    T.gemm(q_l1[owner, :, :], k_l1, score_acc,
                           transpose_B=True, clear_accum=True)
                    T.copy(score_acc, Score[owner, :, :])
            T.fill(saved_max, 0)
            for subtile in T.serial(2):
                T.copy(Score[sid, subtile * 64:(subtile + 1) * 64, :], scores)
                T.reduce_max(scores, maximum, dim=1, clear=True)
                for i in T.Parallel(64):
                    saved_max[0, subtile * 64 + i] = maximum[i]
                for i, j in T.Parallel(64, 128):
                    p[i, j] = T.cast(scores[i, j] - saved_max[0, subtile * 64 + i], dtype)
                T.copy(p, P[sid, subtile * 64:(subtile + 1) * 64, :])
            with T.Cube():
                for owner in T.serial(2):
                    T.copy(P[owner, :, :], p_l1)
                    T.gemm(p_l1, v_l1, pv_acc, transpose_B=False, clear_accum=True)
                    T.copy(pv_acc, PV[owner, :, :])
            for subtile in T.serial(2):
                T.copy(PV[sid, subtile * 64:(subtile + 1) * 64, :], pv)
                for i, j in T.Parallel(64, 64):
                    pv[i, j] = pv[i, j] * T.float32(2)
                T.copy(pv, O[sid, subtile * 64:(subtile + 1) * 64, :])
    return main


def _dynamic_native_count(mod, name):
    total = 0
    for statement, ancestors in statements(mod):
        if isinstance(statement, tirx.Evaluate):
            for call in calls(statement, name):
                total += _constant(call.args[-1]) * math.prod(
                    _constant(loop.extent) for loop in ancestors if isinstance(loop, tirx.For)
                )
    return total


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_two_query_frontend_reaches_native_subtiles_reduce_scratch_and_collective_completion(dtype):
    snapshots = {}
    retained = {"tl.AscendLowerMixedPartition", "tl.AscendLowerVectorReduce",
                "tl.AscendLowerVectorTiles", "tl.InsertSync"}
    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in retained:
                snapshots[info.name] = mod
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(_query_program(dtype), target=TARGET,
                                 enable_host_codegen=False, enable_device_compile=False)
    assert snapshots.keys() == retained
    partition = snapshots["tl.AscendLowerMixedPartition"]
    buffers = {buffer.name: buffer for buffer in _root(partition).alloc_buffers}
    assert tuple(map(int, buffers["saved_max"].shape)) == (2, 128)
    assert tuple(map(int, buffers["scores"].shape)) == (64, 128)
    gemms = calls(partition, "tl.tileop.gemm")
    assert len(gemms) == 2
    assert all(int(call.args[5]) == 128 for call in gemms)
    # Both source QK/PV owner loops remain separate; no 256-row GEMM replaces
    # their tasks. AIV copies keep the SID owner and all 64 rows of each subtile.
    stores = [call.args[1] for call in calls(partition, "tl.tileop.ascend_copy")
              if call.args[1].args[0].buffer.name in ("P", "O")]
    assert len(stores) == 2
    for store in stores:
        assert tuple(map(int, store.args[2:])) in ((1, 64, 128), (1, 64, 64))
        assert isinstance(store.args[0].indices[0], tirx.Var)
        assert store.args[0].indices[0].name == "sid"
    reduced = snapshots["tl.AscendLowerVectorReduce"]
    new_ub = [buffer for buffer in _root(reduced).alloc_buffers
              if buffer.scope() == "shared.dyn" and buffer.name not in buffers]
    assert new_ub, "the width-128 reduction must materialize compiler-owned scratch"
    members = _root(reduced).annotations[VALIDATED]
    assert all(any(buffer.data.same_as(member) for member in members) for buffer in new_ub)
    native = snapshots["tl.AscendLowerVectorTiles"]
    narrow = "tl.ascend_membase." + ("float2half_rn" if dtype == "float16" else "float2bfloat16_rn")
    assert _dynamic_native_count(native, narrow) == 128 * 128
    assert _dynamic_native_count(native, "tl.ascend_membase.mul_scalar") == 128 * 64
    assert calls(snapshots["tl.InsertSync"], "tl.ascend_cross_core_set_flag")
    assert calls(snapshots["tl.InsertSync"], "tl.ascend_cross_core_wait_flag")
    source = artifact.kernel_source
    assert "asc_get_sub_block_id()" in source
    assert "asc_copy_l0c2gm" in source and "asc_copy_ub2gm_align" in source
    assert "asc_sync_pipe(PIPE_V)" in source
