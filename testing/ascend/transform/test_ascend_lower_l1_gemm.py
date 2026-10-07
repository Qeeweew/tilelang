"""L1 GEMM exposes bounded native transfers and truthful sub-K MAD regions."""

import math

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _program(m, n, k, *, clear=True, unit_flag=None, transpose_a=False, transpose_b=True, arch="dav-2201"):
    # Offsets verify that the transform slices the operation region instead of
    # reloading the allocation's origin. Both offsets satisfy native alignment.
    a_shape = (k + 32, m + 16) if transpose_a else (m + 16, k + 32)
    b_shape = (n + 16, k + 32) if transpose_b else (k + 32, n + 16)
    a = tirx.decl_buffer(a_shape, "bfloat16", name="a", scope="shared.l1")
    b = tirx.decl_buffer(b_shape, "bfloat16", name="b", scope="shared.l1")
    c = tirx.decl_buffer((m, n), "float32", name="c", scope="shared.l0c")

    def region(buffer, origin, shape):
        return tirx.BufferRegion(buffer, [tvm.ir.Range.from_min_extent(start, extent) for start, extent in zip(origin, shape)])

    a_region = region(a, (32, 16) if transpose_a else (16, 32), (k, m) if transpose_a else (m, k))
    b_region = region(b, (16, 32) if transpose_b else (32, 16), (n, k) if transpose_b else (k, n))
    call = T.gemm(
        a_region,
        b_region,
        c,
        transpose_A=transpose_a,
        transpose_B=transpose_b,
        clear_accum=clear,
        unit_flag_ctrl=unit_flag,
    )
    root = tirx.SBlock([], [], [], "tilelang_root", tirx.Evaluate(call), alloc_buffers=[a, b, c])
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return tvm.IRModule({"main": func})


def _nodes(module, kind):
    result = []
    post_order_visit(module["main"].body, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def _calls(module, name):
    return [node for node in _nodes(module, tirx.Call) if isinstance(node.op, tvm.ir.Op) and node.op.name == name]


def _l0_buffers(module):
    return [buffer for block in _nodes(module, tirx.SBlock) for buffer in block.alloc_buffers if buffer.scope() in ("shared.l0a", "shared.l0b")]


def _region(call):
    assert call.op.name == "tl.region"
    return call.args[0].buffer, call.args[0].indices, call.args[2:]


def _constant(expression):
    return int(tvm.arith.Analyzer().simplify(expression))


@pytest.mark.parametrize("k", [1536, 24576])
def test_sub_k_trip_count_does_not_expand_compiler_ir(k):
    after = transform.AscendLowerL1Gemm()(_program(128, 96, k))
    (loop,) = _nodes(after, tirx.For)
    assert loop.kind == tirx.ForKind.SERIAL
    assert int(loop.extent) == k // 128
    assert len(_calls(after, "tl.tileop.ascend_copy")) == 2
    (mad,) = _calls(after, "tl.tileop.gemm")
    assert int(mad.args[7]) == 128
    for operand, scope in zip(mad.args[:2], ("shared.l0a", "shared.l0b")):
        buffer, origin, extents = _region(operand)
        assert buffer.scope() == scope
        assert all(int(value) == 0 for value in origin)
        assert int(extents[1]) == 128
    assert len(_l0_buffers(after)) == 2
    for buffer in _l0_buffers(after):
        assert math.prod(int(extent) for extent in buffer.shape) * 2 <= 32768

    (a_copy, b_copy) = _calls(after, "tl.tileop.ascend_copy")
    for copy in (a_copy, b_copy):
        source, origin, extents = _region(copy.args[0])
        assert source.scope() == "shared.l1"
        assert int(origin[0]) == 16
        assert tvm.arith.Analyzer().can_prove_equal(origin[1], 32 + loop.loop_var * 128)
        assert int(extents[1]) == 128


def test_static_tail_has_its_own_compact_allocation_and_terminal_completion():
    after = transform.AscendLowerL1Gemm()(_program(128, 128, 1030, unit_flag=3))
    (loop,) = _nodes(after, tirx.For)
    assert int(loop.extent) == 8
    full, tail = _calls(after, "tl.tileop.gemm")
    assert int(full.args[7]) == 128
    assert int(tail.args[7]) == 6
    assert _constant(full.annotations["unit_flag_ctrl"]) == 0
    assert _constant(tail.annotations["unit_flag_ctrl"]) == 3
    assert _constant(tail.args[9]) == 0
    assert int(tail.args[2].args[1]) == 3
    tail_a, origin, extents = _region(tail.args[0])
    assert tuple(int(extent) for extent in tail_a.shape) == (128, 16)
    assert tuple(int(extent) for extent in extents) == (128, 6)
    assert all(int(value) == 0 for value in origin)
    assert len(_l0_buffers(after)) == 4
    for scope in ("shared.l0a", "shared.l0b"):
        total_bytes = sum(math.prod(int(extent) for extent in buffer.shape) * 2 for buffer in _l0_buffers(after) if buffer.scope() == scope)
        assert total_bytes <= 65536

    tail_copy = _calls(after, "tl.tileop.ascend_copy")[-2]
    _, source_origin, source_extents = _region(tail_copy.args[0])
    assert int(source_origin[1]) == 32 + 1024
    assert int(source_extents[1]) == 6


@pytest.mark.parametrize("clear", [False, True])
def test_accumulator_clear_and_completion_apply_at_whole_gemm_boundaries(clear):
    after = transform.AscendLowerL1Gemm()(_program(128, 128, 1024, clear=clear, unit_flag=3))
    (loop,) = _nodes(after, tirx.For)
    (mad,) = _calls(after, "tl.tileop.gemm")
    assert int(mad.args[2].args[1]) == 3
    for iteration in (0, 1, 7):
        substitute = {loop.loop_var: tirx.IntImm("int32", iteration)}
        actual_clear = _constant(tirx.stmt_functor.substitute(mad.args[9], substitute))
        completion = _constant(tirx.stmt_functor.substitute(mad.annotations["unit_flag_ctrl"], substitute))
        assert actual_clear == int(clear and iteration == 0)
        assert completion == (3 if iteration == 7 else 0)


def test_logical_matrix_tails_preserve_geometry_with_padded_storage():
    after = transform.AscendLowerL1Gemm()(_program(17, 19, 130))
    assert not _nodes(after, tirx.For)
    (mad,) = _calls(after, "tl.tileop.gemm")
    a, _, a_extents = _region(mad.args[0])
    b, _, b_extents = _region(mad.args[1])
    assert tuple(int(value) for value in a.shape) == (32, 144)
    assert tuple(int(value) for value in b.shape) == (32, 144)
    assert tuple(int(value) for value in a_extents) == (17, 130)
    assert tuple(int(value) for value in b_extents) == (19, 130)
    assert int(mad.args[5]) == 17
    assert int(mad.args[6]) == 19
    assert int(mad.args[7]) == 130


def test_scheduler_sees_individual_transfers_and_mads():
    after = transform.MaterializeScheduleUnits()(transform.AscendLowerL1Gemm()(_program(128, 128, 1024)))
    tasks = [node for node in _nodes(after, tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(tasks) == 3
    assert all(isinstance(task.body, tirx.Evaluate) for task in tasks)


@pytest.mark.parametrize("transpose_a,transpose_b", [(True, True), (True, False)])
def test_unsupported_l1_layout_routes_have_an_explicit_diagnostic(transpose_a, transpose_b):
    with pytest.raises(ValueError, match="requires transpose_A=False"):
        transform.AscendLowerL1Gemm()(_program(64, 64, 128, transpose_a=transpose_a, transpose_b=transpose_b))


def test_oversized_unpaired_accumulator_is_diagnosed_before_native_expansion():
    # This accumulator exceeds L0C as well as the minimum L0A fractal.
    # The paired M/N boundary cannot remove a live C without its full export.
    with pytest.raises(ValueError, match="physical M/N GEMM requires adjacent full-clear L1 GEMM"):
        transform.AscendLowerL1Gemm()(_program(32769, 16, 16))


def test_dav3510_path_is_unchanged():
    before = _program(128, 128, 1024, arch="dav-3510")
    after = transform.AscendLowerL1Gemm()(before)
    tvm.ir.assert_structural_equal(before, after)


@pytest.mark.parametrize("automatic", [True, False], ids=["automatic", "explicit-control"])
def test_native_sub_k_loops_preserve_enclosing_pipeline_intent(automatic):
    before = _program(128, 128, 256, unit_flag=3)
    root = _nodes(before, tirx.SBlock)[0]
    outer = tirx.For(
        tirx.Var("outer_k", "int32"), 0, 4, tirx.ForKind.SERIAL, root.body,
        annotations={"tl.ascend_auto_pipeline": automatic, "num_stages": 3},
    )
    function = tirx.PrimFunc([], tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "tilelang_root", outer, alloc_buffers=root.alloc_buffers)))
    function = function.with_attr("target", before["main"].attrs["target"])
    after = transform.AscendLowerL1Gemm()(tvm.IRModule({"main": function}))
    native, enclosing = _nodes(after, tirx.For)
    assert int(native.extent) == 2
    assert bool(native.annotations.get("tl.ascend_auto_pipeline", False)) == automatic
    if automatic:
        assert int(native.annotations["num_stages"]) == 3
    else:
        assert "num_stages" not in native.annotations
    # Scheduling intent does not alter accumulation or whole-GEMM completion.
    (mad,) = _calls(after, "tl.tileop.gemm")
    for iteration in (0, 1):
        substitutions = {native.loop_var: tirx.IntImm("int32", iteration)}
        assert _constant(tirx.stmt_functor.substitute(mad.args[9], substitutions)) == int(iteration == 0)
        assert _constant(tirx.stmt_functor.substitute(mad.annotations["unit_flag_ctrl"], substitutions)) == (3 if iteration == 1 else 0)


def test_explicit_inner_pipeline_is_an_automatic_inheritance_boundary():
    before = _program(128, 128, 256)
    root = _nodes(before, tirx.SBlock)[0]
    inner = tirx.For(tirx.Var("inner", "int32"), 0, 2, tirx.ForKind.SERIAL, root.body, annotations={"tl.ascend_auto_pipeline": False})
    outer = tirx.For(tirx.Var("outer", "int32"), 0, 4, tirx.ForKind.SERIAL, inner, annotations={"tl.ascend_auto_pipeline": True, "num_stages": 4})
    function = tirx.PrimFunc([], tirx.SBlockRealize([], True, tirx.SBlock([], [], [], "tilelang_root", outer, alloc_buffers=root.alloc_buffers))).with_attr("target", before["main"].attrs["target"])
    after = transform.AscendLowerL1Gemm()(tvm.IRModule({"main": function}))
    native = _nodes(after, tirx.For)[0]
    assert "tl.ascend_auto_pipeline" not in native.annotations
    assert "num_stages" not in native.annotations


def test_nn_sub_k_slices_rows_and_keeps_native_l0b_role_geometry():
    after = transform.AscendLowerL1Gemm()(_program(128, 96, 1536, transpose_b=False))
    (loop,) = _nodes(after, tirx.For)
    assert int(loop.extent) == 12
    assert len(_calls(after, "tl.tileop.ascend_copy")) == 2
    (mad,) = _calls(after, "tl.tileop.gemm")
    assert not bool(mad.args[4])
    b_l0, b_origin, b_extents = _region(mad.args[1])
    assert tuple(int(value) for value in b_l0.shape) == (128, 96)
    assert tuple(int(value) for value in b_extents) == (128, 96)
    assert all(int(value) == 0 for value in b_origin)
    b_copy = _calls(after, "tl.tileop.ascend_copy")[1]
    _, source_origin, source_extents = _region(b_copy.args[0])
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(source_origin[0], 32 + loop.loop_var * 128)
    assert int(source_origin[1]) == 16
    assert tuple(int(value) for value in source_extents) == (128, 96)


def test_nn_tail_preserves_k_n_regions_and_compact_padded_storage():
    after = transform.AscendLowerL1Gemm()(_program(128, 97, 1030, transpose_b=False, unit_flag=3))
    full, tail = _calls(after, "tl.tileop.gemm")
    full_b, _, full_extents = _region(full.args[1])
    tail_b, tail_origin, tail_extents = _region(tail.args[1])
    assert tuple(int(value) for value in full_b.shape) == (128, 112)
    assert tuple(int(value) for value in full_extents) == (128, 97)
    assert tuple(int(value) for value in tail_b.shape) == (16, 112)
    assert tuple(int(value) for value in tail_extents) == (6, 97)
    assert all(int(value) == 0 for value in tail_origin)
    assert int(tail.args[7]) == 6
    assert int(tail.args[6]) == 97
    assert _constant(tail.args[9]) == 0
    assert int(tail.args[2].args[1]) == 3
    assert _constant(full.annotations["unit_flag_ctrl"]) == 0
    assert _constant(tail.annotations["unit_flag_ctrl"]) == 3
    _, source_origin, source_extents = _region(_calls(after, "tl.tileop.ascend_copy")[-1].args[0])
    assert tuple(int(value) for value in source_origin) == (1056, 16)
    assert tuple(int(value) for value in source_extents) == (6, 97)
    for scope in ("shared.l0a", "shared.l0b"):
        assert sum(math.prod(int(extent) for extent in buffer.shape) * 2 for buffer in _l0_buffers(after) if buffer.scope() == scope) <= 65536


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_automatic_l1_nn_pipeline_emits_classic_transposed_b16_load(dtype):
    import tilelang

    m, n, total_k, tile_k = 128, 96, 768, 384

    @T.prim_func
    def main(A: T.Tensor((m, total_k), dtype), B: T.Tensor((total_k, n), dtype), C: T.Tensor((m, n), dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_k, n), dtype)
            c_l0 = T.alloc_l0c((m, n), "float32")
            for k in T.Pipelined(total_k // tile_k):
                T.copy(A[:, k * tile_k : (k + 1) * tile_k], a_l1)
                T.copy(B[k * tile_k : (k + 1) * tile_k, :], b_l1)
                T.gemm(a_l1, b_l1, c_l0, transpose_B=False, clear_accum=k == 0)
            T.copy(c_l0, C)

    source = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}).kernel_source
    assert "load_cbuf_to_cb(" in source
    assert "asc_copy_l12l0b_transpose(" not in source
    assert "tl::ascend_gemm_l1" not in source
    assert "asc_mmad(" in source
