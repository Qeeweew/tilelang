"""Host regressions for native2201 vector intent, storage, and effects."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _parallel(body, extents):
    for var, extent in reversed(extents):
        body = tirx.For(var, 0, extent, tirx.ForKind.PARALLEL, body)
    return body


def _binary(dtype="float32", count=17):
    a, b, c = [tirx.decl_buffer((count,), dtype, name=name, scope="shared") for name in "abc"]
    i = tirx.Var("i", "int32")
    mod = kernel(_parallel(tirx.BufferStore(c, a[i] + b[i], [i]), [(i, count)]), buffers=[a, b, c])
    return _target(mod), (a, b, c)


def _ops(ir):
    return [call for call in nodes(ir, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _pointers(call):
    return [arg for arg in call.args if isinstance(arg, tirx.Call) and arg.op.name == "tl.access_ptr"]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_count_tail_has_exact_read_write_effects(dtype):
    before, (a, b, c) = _binary(dtype)
    after = transform.AscendLowerVectorTiles()(before)
    op = _ops(after)[0]
    assert op.op.name == "tl.ascend_membase.add"
    assert op.op.get_attr("TGlobalSymbol") == "asc_add"
    assert int(op.args[-1]) == 17
    assert not nodes(after, tirx.For)
    pointers = _pointers(op)
    assert [ptr.args[0].buffer.name for ptr in pointers] == [c.name, a.name, b.name]
    assert [int(ptr.args[1]) for ptr in pointers] == [17, 17, 17]
    assert [int(ptr.args[2]) for ptr in pointers] == [2, 1, 1]


def test_expression_scratch_is_allocated_before_scheduling_and_shared_dag_once():
    before, (a, b, c) = _binary(count=64)
    i = tirx.Var("i", "int32")
    intermediate = a[i] + b[i]
    expression = tirx.exp(intermediate) / (intermediate * tirx.const(2, "float32"))
    body = _parallel(tirx.BufferStore(c, expression, [i]), [(i, 64)])
    before = _target(kernel(body, buffers=[a, b, c]))
    after = transform.AscendLowerVectorTiles()(before)
    assert [op.op.name.rsplit(".", 1)[1] for op in _ops(after)] == ["add", "exp", "mul_scalar", "div"]
    scratch = [buf for buf in after["main"].body.block.alloc_buffers if buf.name.startswith("membase_tmp_")]
    assert len(scratch) == 3
    assert all(buf.scope() == "shared.dyn" and int(buf.shape[0]) == 64 for buf in scratch)
    scratch_vars = {buf.data for buf in scratch}
    pointer_vars = {ptr.args[0].buffer.data for op in _ops(after) for ptr in _pointers(op)}
    assert scratch_vars <= pointer_vars
    # The existing scheduler consumes each native operation, not one opaque
    # compound expression block. No device or timing model is needed here.
    materialized = transform.MaterializeScheduleUnits()(after)
    assert len(calls(materialized, "tl.ascend_membase.add")) == 1
    assert len([node for node in nodes(materialized, tirx.AttrStmt) if node.attr_key == "tl.schedule_unit"]) == 4


def test_frontend_scratch_belongs_to_each_kernel_and_never_the_host_root():
    @T.prim_func
    def main(A: T.Tensor((2, 64), "float32"), C: T.Tensor((2, 64), "float32")):
        with T.Kernel(1):
            first = T.alloc_shared((64,), "float32")
            first_out = T.alloc_shared((64,), "float32")
            T.copy(A[0, 0:64], first)
            for i in T.Parallel(64):
                first_out[i] = T.exp(first[i] + 1.0)
            T.copy(first_out, C[0, 0:64])
        with T.Kernel(1):
            second = T.alloc_shared((64,), "float32")
            second_out = T.alloc_shared((64,), "float32")
            T.copy(A[1, 0:64], second)
            for i in T.Parallel(64):
                second_out[i] = T.exp(second[i] + 2.0)
            T.copy(second_out, C[1, 0:64])

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    mod = tirx.transform.BindTarget(target)(tvm.IRModule.from_expr(main))
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=None,
        unsupported_annotations=["cluster_dims"],
        launch_dim_tags=["cthread"],
    )(mod)
    after = transform.AscendLowerVectorTiles()(mod)
    blocks = nodes(after, tirx.SBlock)
    roots = [block for block in blocks if block.name_hint == "tilelang_root"]
    assert len(roots) == 2
    assert any(block.name_hint != "tilelang_root" for block in blocks)
    scratch_data = []
    for root in roots:
        scratch = [buffer for buffer in root.alloc_buffers if buffer.name.startswith("membase_tmp_")]
        assert len(scratch) == 1
        referenced = {ptr.args[0].buffer.data for op in _ops(root.body) for ptr in _pointers(op)}
        assert scratch[0].data in referenced
        scratch_data.append(scratch[0].data)
    assert not scratch_data[0].same_as(scratch_data[1])
    for block in blocks:
        if block.name_hint != "tilelang_root":
            assert not any(buffer.name.startswith("membase_tmp_") for buffer in block.alloc_buffers)


@pytest.mark.parametrize("dtype,count", [("float16", 17), ("float32", 65)])
def test_frontend_lower_owns_target_context_and_emits_native_tail(dtype, count):
    @T.prim_func
    def main(A: T.Tensor((count,), dtype), C: T.Tensor((count,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((count,), dtype)
            c = T.alloc_shared((count,), dtype)
            T.copy(A, a)
            for i in T.Parallel(count):
                c[i] = T.exp(a[i] + T.cast(1, dtype))
            T.copy(c, C)

    # An unrelated caller context must not influence the selected backend.
    with tvm.target.Target({"kind": "ascend", "arch": "dav-3510"}):
        source = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}).kernel_source
    assert "asc_add_scalar(" in source and "asc_exp(" in source, source
    assert "__vector__" in source and "__simd_vf__" not in source, source


def test_dense_two_dimensional_tile_is_one_instruction():
    a, b, c = [tirx.decl_buffer((2, 32), "float32", name=name, scope="shared") for name in "abc"]
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    before = _target(kernel(_parallel(tirx.BufferStore(c, a[i, j] * b[i, j], [i, j]), [(i, 2), (j, 32)]), buffers=[a, b, c]))
    after = transform.AscendLowerVectorTiles()(before)
    assert len(_ops(after)) == 1
    assert int(_ops(after)[0].args[-1]) == 64
    assert not nodes(after, tirx.For)


def _row_broadcast(width=32):
    a, c = [tirx.decl_buffer((2, width), "float32", name=name, scope="shared") for name in "ac"]
    b = tirx.decl_buffer((2,), "float32", name="row", scope="shared")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    body = _parallel(tirx.BufferStore(c, a[i, j] + b[i], [i, j]), [(i, 2), (j, width)])
    return _target(kernel(body, buffers=[a, b, c]))


def test_row_broadcast_exposes_scalar_load_before_vector_instruction():
    after = transform.AscendLowerVectorTiles()(_row_broadcast())
    loops = nodes(after, tirx.For)
    assert len(loops) == 1 and loops[0].kind == tirx.ForKind.SERIAL
    assert int(loops[0].extent) == 2
    binds = nodes(after, tirx.Bind)
    assert len(binds) == 1 and isinstance(binds[0].value, tirx.BufferLoad)
    op = _ops(after)[0]
    assert op.op.name == "tl.ascend_membase.add_scalar"
    assert op.args[2].same_as(binds[0].var)
    assert int(op.args[-1]) == 32
    assert len(_pointers(op)) == 2


def test_unaligned_row_broadcast_reports_physical_contract():
    with pytest.raises(ValueError, match="32-byte aligned"):
        transform.AscendLowerVectorTiles()(_row_broadcast(width=17))


def test_scalar_fill_uses_duplicate_with_exact_count():
    _, (_, _, c) = _binary(count=17)
    i = tirx.Var("i", "int32")
    before = _target(kernel(_parallel(tirx.BufferStore(c, tirx.const(3, "float32"), [i]), [(i, 17)]), buffers=[c]))
    after = transform.AscendLowerVectorTiles()(before)
    assert _ops(after)[0].op.name == "tl.ascend_membase.duplicate"
    assert int(_pointers(_ops(after)[0])[0].args[1]) == 17


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_integer_fill_value_is_converted_to_the_destination_scalar_dtype(dtype):
    c = tirx.decl_buffer((17,), dtype, name="c", scope="shared.dyn")
    # Fill constructs its Cast after the preceding Simplify pass. Preserve
    # this ordinary frontend call, rather than pretyping its integer zero.
    before = _target(kernel(tirx.Evaluate(T.fill(c, 0)), buffers=[c]))
    after = transform.AscendLowerVectorTiles()(before)
    (duplicate,) = _ops(after)
    assert duplicate.op.name == "tl.ascend_membase.duplicate"
    assert duplicate.args[0].args[0].buffer.dtype == dtype
    assert duplicate.args[1].dtype == dtype and float(duplicate.args[1]) == 0.0
    assert int(duplicate.args[-1]) == int(duplicate.args[0].args[1]) == 17
    assert not nodes(after, tirx.For)
    assert not any(buffer.name.startswith("membase_tmp_") for buffer in after["main"].body.block.alloc_buffers)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_frontend_integer_zero_fill_retains_dtype_through_full_lowering(dtype):
    @T.prim_func
    def main(C: T.Tensor((17,), dtype)):
        with T.Kernel(1):
            c = T.alloc_shared((17,), dtype)
            T.fill(c, 0)
            T.copy(c, C)

    @tvm.instrument.pass_instrument
    class CaptureVector:
        def __init__(self):
            self.operations = None

        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLowerVectorTiles":
                self.operations = _ops(mod)

    capture = CaptureVector()
    with tvm.transform.PassContext(instruments=[capture]):
        source = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}).kernel_source
    (duplicate,) = capture.operations
    assert duplicate.op.name == "tl.ascend_membase.duplicate"
    assert duplicate.args[0].args[0].buffer.dtype == duplicate.args[1].dtype == dtype
    assert "asc_duplicate(" in source
    assert "asc_float2" not in source and "asc_half2float(" not in source and "asc_bfloat162float(" not in source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_infinity_fill_is_an_exact_known_scalar_constant(dtype):
    c = tirx.decl_buffer((17,), dtype, name="c", scope="shared.dyn")
    before = _target(kernel(tirx.Evaluate(T.fill(c, T.infinity(dtype))), buffers=[c]))
    after = transform.AscendLowerVectorTiles()(before)
    (duplicate,) = _ops(after)
    assert duplicate.op.name == "tl.ascend_membase.duplicate"
    assert duplicate.args[0].args[0].buffer.dtype == duplicate.args[1].dtype == dtype
    assert len(calls(after, "tl.infinity")) == 1


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_frontend_prefix_select_preserves_the_normal_infinity_expression(dtype):
    @T.prim_func
    def main(A: T.Tensor((130,), dtype), C: T.Tensor((130,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((130,), dtype)
            c = T.alloc_shared((130,), dtype)
            T.copy(A, a)
            for i in T.Parallel(130):
                c[i] = T.if_then_else(i < 17, a[i], T.cast(-T.infinity("float32"), dtype))
            T.copy(c, C)

    source = tilelang.lower(main, target={"kind": "ascend", "arch": "dav-2201"}).kernel_source
    assert "tl_ascend_membase_select_bits(" in source and "asc_duplicate(" in source
    assert "tl::limits::kFloatInf" in source or "tl::limits::kFloatNInf" in source
    assert ("asc_float2half_rn(" in source) == (dtype == "float16")


def test_unknown_uniform_call_is_not_delegated_to_scalar_codegen():
    c = tirx.decl_buffer((17,), "float32", name="c", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    value = tirx.call_pure_extern("float32", "unknown_scalar_operation")
    before = _target(kernel(_parallel(tirx.BufferStore(c, value, [i]), [(i, 17)]), buffers=[c]))
    with pytest.raises(ValueError, match="unsupported vector expression"):
        transform.AscendLowerVectorTiles()(before)


def test_strided_gather_is_unsupported_and_vector_cast_is_native():
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared")
    c = tirx.decl_buffer((32,), "float32", name="c", scope="shared")
    i = tirx.Var("i", "int32")
    before = _target(kernel(_parallel(tirx.BufferStore(c, a[2 * i] + tirx.const(1, "float32"), [i]), [(i, 32)]), buffers=[a, c]))
    with pytest.raises(ValueError, match="strided/gather"):
        transform.AscendLowerVectorTiles()(before)
    h = tirx.decl_buffer((32,), "float16", name="h", scope="shared")
    before = _target(kernel(_parallel(tirx.BufferStore(c, tirx.Cast("float32", h[i]) + c[i], [i]), [(i, 32)]), buffers=[h, c]))
    after = transform.AscendLowerVectorTiles()(before)
    assert [call.op.name for call in _ops(after)] == [
        "tl.ascend_membase.half2float", "tl.ascend_membase.add"
    ]


def test_3510_parallel_is_preserved_for_vf_lowering():
    before, _ = _binary()
    before = _target(before, "dav-3510")
    after = transform.AscendLowerVectorTiles()(before)
    tvm.ir.assert_structural_equal(after, before)


def test_shifted_overlapping_in_place_region_is_rejected():
    a = tirx.decl_buffer((64,), "float32", name="a", scope="shared")
    i = tirx.Var("i", "int32")
    body = _parallel(tirx.BufferStore(a, a[i + 8] + tirx.const(1, "float32"), [i]), [(i, 32)])
    with pytest.raises(ValueError, match="shifted overlapping"):
        transform.AscendLowerVectorTiles()(_target(kernel(body, buffers=[a])))


def test_gm_arithmetic_does_not_silently_become_scalar_element_loops():
    a = tirx.decl_buffer((32,), "float32", name="a")
    c = tirx.decl_buffer((32,), "float32", name="c", scope="shared")
    i = tirx.Var("i", "int32")
    before = _target(kernel(_parallel(tirx.BufferStore(c, a[i] + tirx.const(1, "float32"), [i]), [(i, 32)]), buffers=[c], params=[a]))
    with pytest.raises(ValueError, match="private UB"):
        transform.AscendLowerVectorTiles()(before)
