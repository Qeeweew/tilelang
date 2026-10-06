"""Ordinary low-input reductions expose typed promotion and real UB effects."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes


def _program(*, count=130, rows=3, clear=True, retain_axis=False,
             dtype="bfloat16", output_dtype=None, kind="max", annotations=None, row_pitch=None):
    element_bytes = 2 if dtype in ("float16", "bfloat16") else 4
    block = 32 // element_bytes
    pitch = row_pitch or (count + block - 1) // block * block
    src = tirx.decl_buffer((rows, pitch), dtype, "src", scope="shared.dyn")
    dst = tirx.decl_buffer((rows, 1) if retain_axis else (rows,), output_dtype or dtype, "dst", scope="shared.dyn")
    src_region = tirx.Call("handle", tvm.ir.Op.get("tl.region"), [src[0, 0], 1, rows, count])
    dst_region = tirx.Call("handle", tvm.ir.Op.get("tl.region"),
                           [dst[0, 0] if retain_axis else dst[0], 2 if clear else 3,
                            rows, *([1] if retain_axis else [])])
    reduce = tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"),
                      [src_region, dst_region, tirx.StringImm(kind), 1, tirx.const(clear, "bool")],
                      annotations=annotations)
    mod = kernel(tirx.Evaluate(reduce), buffers=[src, dst])
    mod = tvm.IRModule({"main": mod["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))})
    return mod, src, dst


def _lower(mod, *, vector=False):
    mod = transform.AscendLowerVectorReduce()(mod)
    return transform.AscendLowerVectorTiles()(mod) if vector else mod


def _native(ir):
    return [node for node in nodes(ir, tirx.Call)
            if isinstance(node.op, tvm.ir.Op) and node.op.name.startswith("tl.ascend_membase.")]


@pytest.mark.parametrize("retain_axis", [False, True])
@pytest.mark.parametrize("count,partial_counts", [(17, [17]), (130, [130, 3]), (5001, [5001, 79, 2])])
def test_bfloat_max_promotes_exact_valid_elements_and_reuses_f32_partial_plan(retain_axis, count, partial_counts):
    before, src, dst = _program(count=count, retain_axis=retain_axis)
    after = _lower(before)
    native = calls(after, "tl.ascend_membase.repeat_reduce_max_only_value")
    assert [int(call.args[-1]) for call in native] == partial_counts
    assert all(arg.args[0].buffer.dtype == "float32" for call in native for arg in call.args[:2])
    casts = nodes(after, tirx.Cast)
    assert [(str(node.value.dtype), str(node.dtype)) for node in casts] == [
        ("bfloat16", "float32"), ("float32", "bfloat16")]
    loops = nodes(after, tirx.For)
    parallel = [loop for loop in loops if loop.kind == tirx.ForKind.PARALLEL]
    assert [int(loop.extent) for loop in parallel] == [count, 1]
    assert len([loop for loop in loops if loop.kind == tirx.ForKind.SERIAL]) == 1
    # An odd output row is not a valid native max destination; only the final
    # scalar store may address compact BF16 output.
    (store,) = [store for store in nodes(after, tirx.BufferStore) if store.buffer.same_as(dst)]
    assert isinstance(store.value, tirx.Var) and store.value.dtype == "bfloat16"
    assert all(not arg.args[0].buffer.same_as(src) and not arg.args[0].buffer.same_as(dst)
               for call in native for arg in call.args[:2])
    scratch = after["main"].body.block.alloc_buffers[2:]
    assert all(buf.scope() == "shared.dyn" and int(buf.shape[0]) * (2 if buf.dtype == "bfloat16" else 4) % 32 == 0
               for buf in scratch)
    assert int(scratch[0].shape[0]) == (count + 7) // 8 * 8


def test_clear_false_keeps_narrowed_reduce_and_bfloat_combine_as_separate_nodes():
    before, _, dst = _program(clear=False)
    after = _lower(before)
    (maximum,) = nodes(after, tirx.Max)
    assert maximum.dtype == maximum.a.dtype == maximum.b.dtype == "bfloat16"
    assert isinstance(maximum.a, tirx.BufferLoad) and isinstance(maximum.b, tirx.BufferLoad)
    (narrow,) = [store for store in nodes(after, tirx.BufferStore)
                if isinstance(store.value, tirx.Cast) and store.value.dtype == "bfloat16"]
    assert maximum.a.buffer.same_as(narrow.buffer)
    (previous,) = [bind for bind in nodes(after, tirx.Bind)
                  if isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.same_as(dst)]
    (snapshot,) = [store for store in nodes(after, tirx.BufferStore) if store.buffer.same_as(maximum.b.buffer)]
    assert snapshot.value.same_as(previous.var)
    assert not maximum.b.buffer.same_as(dst)


@pytest.mark.parametrize("clear", [True, False])
def test_promotion_and_combine_lower_through_existing_vector_casts(clear):
    before, src, dst = _program(clear=clear)
    after = _lower(before, vector=True)
    native = _native(after)
    names = [op.op.name.rsplit(".", 1)[1] for op in native]
    assert names[:4] == ["bfloat162float", "repeat_reduce_max_only_value",
                         "repeat_reduce_max_only_value", "float2bfloat16_rn"]
    assert int(native[0].args[-1]) == int(native[0].args[1].args[1]) == 130
    assert native[0].args[1].args[0].buffer.same_as(src)
    assert names.count("float2bfloat16_rn") == (2 if not clear else 1)
    assert names.count("max") == (1 if not clear else 0)
    assert all(not (op.op.name.endswith(".duplicate") and op.args[0].args[0].buffer.dtype == "bfloat16") for op in native)
    assert all(not (loop.kind == tirx.ForKind.PARALLEL) for loop in nodes(after, tirx.For))
    final = [store for store in nodes(after, tirx.BufferStore) if store.buffer.same_as(dst)]
    assert len(final) == 1
    materialized = transform.MaterializeScheduleUnits()(after)
    tasks = [stmt for stmt in nodes(materialized, tirx.AttrStmt) if stmt.attr_key == "tl.ascend_task"]
    assert len(tasks) >= len(native)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("clear", [True, False])
def test_low_input_sum_uses_output_f32_accumulation_and_exact_cast_effects(dtype, clear):
    before, src, dst = _program(dtype=dtype, output_dtype="float32", kind="sum", clear=clear)
    after = _lower(before, vector=True)
    native = _native(after)
    names = [op.op.name.rsplit(".", 1)[1] for op in native]
    assert names == ["half2float" if dtype == "float16" else "bfloat162float",
                     "repeat_reduce_sum", "repeat_reduce_sum"]
    widen = native[0]
    assert int(widen.args[-1]) == 130
    assert int(widen.args[0].args[1]) == int(widen.args[1].args[1]) == 130
    assert widen.args[1].args[0].buffer.same_as(src)
    assert widen.args[0].args[0].buffer.dtype == "float32"
    assert [int(op.args[-1]) for op in native[1:]] == [130, 3]
    final = [store for store in nodes(after, tirx.BufferStore) if store.buffer.same_as(dst)]
    assert len(final) == 1 and final[0].value.dtype == "float32"
    previous = [bind for bind in nodes(after, tirx.Bind)
                if isinstance(bind.value, tirx.BufferLoad) and bind.value.buffer.same_as(dst)]
    assert len(previous) == (0 if clear else 1)
    if not clear:
        assert isinstance(final[0].value, tirx.Add)
        assert final[0].value.b.same_as(previous[0].var)
    # One row's widened input + one aligned result + the three-partial
    # intermediate. No whole widened matrix and no low-output narrow exist.
    scratch = after["main"].body.block.alloc_buffers[2:]
    assert sum(int(buf.shape[0]) * 4 for buf in scratch) == 608
    assert all(buf.dtype == "float32" and buf.scope() == "shared.dyn" for buf in scratch)


@pytest.mark.parametrize("dtype,kind", [("float32", "sum"), ("float16", "max")])
def test_existing_native_dtype_routes_do_not_gain_promotion(dtype, kind):
    before, _, _ = _program(dtype=dtype, kind=kind)
    after = _lower(before)
    assert not nodes(after, tirx.Cast)
    assert all(argument.args[0].buffer.dtype == dtype
               for call in _native(after) for argument in call.args[:2])


@pytest.mark.parametrize("dtype,kind,output_dtype", [
    ("float16", "sum", "float32"), ("bfloat16", "sum", "float32"),
    ("bfloat16", "max", "bfloat16"),
])
@pytest.mark.parametrize("clear", [True, False])
def test_ordinary_dsl_lowers_promoted_reduce_without_user_cast_or_scratch(dtype, kind, output_dtype, clear):
    @T.prim_func
    def main(A: T.Tensor((3, 130), dtype), O: T.Tensor((3,), output_dtype)):
        with T.Kernel(1):
            values = T.alloc_shared((3, 144), dtype)
            output = T.alloc_shared((3,), output_dtype)
            T.copy(A, values[:, 0:130])
            if not clear:
                T.copy(O, output)
            if kind == "sum":
                T.reduce_sum(values[:, 0:130], output, dim=1, clear=clear)
            else:
                T.reduce_max(values[:, 0:130], output, dim=1, clear=clear)
            T.copy(output, O)

    artifact = tilelang.lower(
        main, target={"kind": "ascend", "arch": "dav-2201"}, target_host="c",
        enable_host_codegen=True, enable_device_compile=False,
    )
    source = artifact.kernel_source
    widen = "asc_half2float(" if dtype == "float16" else "asc_bfloat162float("
    reduce = "asc_repeat_reduce_sum(" if kind == "sum" else "asc_repeat_reduce_max_only_value("
    assert widen in source
    assert reduce in source
    assert ("asc_float2bfloat16_rn(" in source) == (kind == "max")
    assert artifact.host_mod


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_sum_still_requires_explicit_accumulator_precision(dtype):
    before, _, _ = _program(dtype=dtype, kind="sum")
    with pytest.raises(ValueError, match="Low-precision sum output needs an explicit accumulator-precision contract"):
        _lower(before)


def test_explicit_nan_propagation_remains_diagnosed():
    before, _, _ = _program(annotations={"nan_propagate": tirx.const(True, "bool")})
    with pytest.raises(ValueError, match="nan_propagate=True has no established"):
        _lower(before)


def test_promoted_source_still_requires_aligned_rows():
    before, _, _ = _program(count=17, row_pitch=17)
    with pytest.raises(ValueError, match="32-byte aligned"):
        _lower(before)


@pytest.mark.parametrize("dtype,kind,output_dtype", [
    ("float16", "sum", "float32"), ("bfloat16", "sum", "float32"),
    ("bfloat16", "max", "bfloat16"),
])
def test_aligned_promoted_input_capacity_must_fit_int32_metadata(dtype, kind, output_dtype):
    # This creates TIR metadata only. No physical UB or host allocation occurs.
    # The logical extent is legal int32, but the widened input's aligned
    # capacity would be 2**31 and must diagnose before make_const(int32).
    before, _, _ = _program(count=(1 << 31) - 1, rows=1, dtype=dtype,
                           kind=kind, output_dtype=output_dtype)
    with pytest.raises(ValueError, match="aligned reduction scratch exceeds the int32 metadata range"):
        _lower(before)


def test_native_f32_sum_keeps_its_existing_large_partial_metadata_range():
    # The pass can represent partial counts even though the normal later
    # resource planner will reject this allocation as larger than UB.
    before, _, _ = _program(count=(1 << 31) - 1, rows=1,
                           dtype="float32", kind="sum")
    after = _lower(before)
    native = calls(after, "tl.ascend_membase.repeat_reduce_sum")
    assert [int(call.args[-1]) for call in native] == [
        (1 << 31) - 1, 33554432, 524288, 8192, 128, 2,
    ]
    assert int(native[0].args[0].args[1]) == 33554432


def test_compound_promotion_cannot_hide_in_one_opaque_task():
    before, _, _ = _program()
    block = before["main"].body.block
    block = tirx.SBlock([], [], [], block.name_hint,
                       tirx.AttrStmt({}, "tl.ascend_task", 1, block.body), alloc_buffers=block.alloc_buffers)
    before = tvm.IRModule({"main": before["main"].with_body(tirx.SBlockRealize([], True, block))})
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)
