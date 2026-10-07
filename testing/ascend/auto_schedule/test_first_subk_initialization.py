"""Common owner inference recognizes a guaranteed first sub-K definition.

These CPU tests stop at AnnotateMultiBufferEligible. They neither schedule a
kernel nor compile AscendC, and do not declare a physical completion relation.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def _region(buffer, rows=None, columns=None):
    shape = (rows, columns) if rows is not None else buffer.shape
    return tirx.BufferRegion(
        buffer, [tvm.ir.Range.from_min_extent(0, extent) for extent in shape]
    )


def _owners(module, storage):
    return [
        loop.loop_var.name
        for loop in nodes(module, tirx.For)
        if storage in loop.annotations.get("multi_buffer_eligible", [])
    ]


def _fixture(kind="ordinary", *, first=0, chunks=2, policy=None, owners=1):
    rows, width, sub_k = 128, 64, 128
    p = tirx.decl_buffer((rows, sub_k), "float16", name="p", scope="shared.l1")
    v = tirx.decl_buffer((sub_k, width), "float16", name="v", scope="shared.l1")
    a = tirx.decl_buffer(p.shape, "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer(v.shape, "float16", name="b", scope="shared.l0b")
    pv = tirx.decl_buffer((rows, width), "float32", name="pv", scope="shared.l0c")
    out = tirx.decl_buffer(pv.shape, "float32", name="out")
    enabled = tirx.Var("enabled", "bool")
    steps = tirx.Var("steps", "int32")
    loops = []
    for index in range(owners):
        epoch = tirx.Var(f"epoch{index}", "int32")
        inner = tirx.Var(f"sub_k{index}", "int32")
        clear = inner == first
        if kind == "owner-coordinate":
            clear = tirx.And(epoch == 0, clear)
        elif kind == "later-only":
            clear = inner == first + 1
        elif kind == "unknown":
            clear = tirx.And(enabled, clear)
        elif kind == "false":
            clear = tirx.const(False, "bool")
        elif kind == "scalar-clear":
            clear = tirx.Or(clear, pv[0, 0] > 0)
        elif kind == "opaque-clear":
            clear = tirx.Or(clear, tirx.call_extern("bool", "foreign_clear", pv.data))
        accumulator = (
            tirx.decl_buffer((64, width), "float32", name="view", data=pv.data, scope="shared.l0c")
            if kind == "alias-view" else pv
        )
        active_rows = 64 if kind in ("partial", "alias-view") else rows
        left = pv if kind == "input-alias" else a
        active_k = width if kind == "input-alias" else sub_k
        mad = T.gemm(
            _region(left, active_rows, active_k),
            _region(b, active_k, width),
            _region(accumulator, active_rows, width),
            clear_accum=clear,
            unit_flag_ctrl=pv[0, 0] if kind == "scalar-annotation" else None,
        )
        if kind == "predicated-region":
            args = list(mad.args)
            c_region = args[2]
            identity = tirx.BufferLoad(pv, [0, 0], predicate=enabled)
            args[2] = tirx.Call(c_region.dtype, c_region.op, [identity, *c_region.args[1:]])
            mad = tirx.Call(mad.dtype, mad.op, args, annotations=mad.annotations)
        operation = tirx.Evaluate(mad)
        if kind == "atomic-conditional":
            operation = tirx.IfThenElse(enabled, operation, None)
        chunk = seq(
            unit(copy(p, a), core=2, stage=-1),
            unit(copy(v, b), core=2, stage=-1),
            unit(operation, core=2, stage=-1,
                 guard=enabled if kind == "task-conditional" else None),
        )
        if kind == "loop-break":
            chunk = seq(unit(tirx.Evaluate(tirx.Call(
                "handle", tvm.ir.Op.get("tl.loop_break"), [])), core=2, stage=-1), chunk)
        extent = 0 if kind == "zero" else steps if kind == "dynamic" else chunks
        native = unit(
            tirx.For(inner, first, extent, tirx.ForKind.SERIAL, chunk),
            core=None, stage=-1,
            guard=enabled if kind == "conditional" else None,
        )
        if kind == "opaque-before":
            native = seq(unit(tirx.Evaluate(tirx.call_extern(
                "int32", "foreign_read", pv.data)), core=2, stage=-1), native)
        annotations = {"multi_buffer_eligible": [pv.data]} if policy == "explicit" and index == 0 else {}
        loops.append(unit(
            tirx.For(epoch, 0, 3, tirx.ForKind.SERIAL,
                     seq(native, unit(copy(pv, out), core=2, stage=-1)),
                     annotations=annotations),
            core=None, stage=-1,
        ))
    annotations = {}
    if policy in ("single", "fixed", "counter"):
        annotations["tl.buffer_versions_map"] = {pv.data: 1 if policy == "single" else 2}
    if policy == "counter":
        annotations["tl.buffer_version_mode"] = {pv.data: "counter"}
    elif policy == "manual":
        annotations["tl.manual_multi_buffer"] = {pv.data: 2}
    before = kernel(
        seq(*loops), buffers=[p, v, a, b, pv], params=[out, enabled, steps],
        annotations=annotations,
    )
    return before, pv, a, b


@pytest.mark.parametrize("first,chunks", [(0, 2), (0, 5), (3, 2)])
def test_nested_first_subk_clear_has_outer_owner_without_moving_l0(first, chunks):
    before, pv, a, b = _fixture(first=first, chunks=chunks)
    after = transform.AnnotateMultiBufferEligible()(before)
    assert _owners(after, pv.data) == ["epoch0"]
    assert _owners(after, a.data) == _owners(after, b.data) == ["sub_k0"]
    # The proof changes eligibility only. The native operations, clear predicate
    # and conservative C read-write region remain exactly as supplied.
    original = calls(before, "tl.tileop.gemm")
    actual = calls(after, "tl.tileop.gemm")
    assert len(actual) == len(original) == 1
    tvm.ir.assert_structural_equal(actual[0], original[0])
    assert int(actual[0].args[2].args[1]) == 3
    assert len(nodes(after, tirx.For)) == 2


@pytest.mark.parametrize("kind", [
    "owner-coordinate", "zero", "dynamic", "conditional", "task-conditional",
    "partial", "later-only", "unknown", "false", "alias-view", "input-alias",
    "scalar-clear", "opaque-clear", "scalar-annotation", "loop-break",
    "atomic-conditional", "opaque-before", "predicated-region",
])
def test_missing_first_definition_does_not_create_an_owner(kind):
    before, pv, _, _ = _fixture(kind)
    after = transform.AnnotateMultiBufferEligible()(before)
    assert _owners(after, pv.data) == []


@pytest.mark.parametrize("policy,expected", [
    ("single", []),
    ("manual", []),
    ("fixed", ["epoch0"]),
    ("counter", ["epoch0"]),
    ("explicit", ["epoch0"]),
])
def test_existing_version_and_explicit_owner_policies_are_preserved(policy, expected):
    before, pv, _, _ = _fixture(policy=policy, owners=2 if policy == "explicit" else 1)
    after = transform.AnnotateMultiBufferEligible()(before)
    assert _owners(after, pv.data) == expected
    root_before = before["main"].body.block
    root_after = after["main"].body.block
    for key in ("tl.buffer_versions_map", "tl.buffer_version_mode", "tl.manual_multi_buffer"):
        if key in root_before.annotations:
            tvm.ir.assert_structural_equal(root_after.annotations[key], root_before.annotations[key])


def test_independent_sibling_epochs_keep_their_common_owner_frontier():
    before, pv, a, b = _fixture(owners=2)
    after = transform.AnnotateMultiBufferEligible()(before)
    assert _owners(after, pv.data) == ["epoch0", "epoch1"]
    assert _owners(after, a.data) == _owners(after, b.data) == ["sub_k0", "sub_k1"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("reduction", [256, 640, 272])
def test_lowered_l1_gemm_exposes_initialization_to_common_owner_inference(dtype, reduction):
    p = tirx.decl_buffer((128, reduction), dtype, name="p", scope="shared.l1")
    v = tirx.decl_buffer((reduction, 64), dtype, name="v", scope="shared.l1")
    pv = tirx.decl_buffer((128, 64), "float32", name="pv", scope="shared.l0c")
    out = tirx.decl_buffer(pv.shape, "float32", name="out")
    epoch = tirx.Var("kv_epoch", "int32")
    # Start from an ordinary L1 GEMM, without supplying an owner annotation or
    # native sub-K operations. Materialization exposes the actual compiler IR.
    body = tirx.For(
        epoch, 0, 3, tirx.ForKind.SERIAL,
        seq(tirx.Evaluate(T.gemm(p, v, pv, clear_accum=True)), copy(pv, out)),
    )
    before = kernel(body, buffers=[p, v, pv], params=[out])
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    )
    lowered = transform.AscendLowerL1Gemm()(before)
    lowered = transform.MaterializeScheduleUnits()(lowered)
    after = transform.AnnotateMultiBufferEligible()(lowered)
    assert _owners(after, pv.data) == ["kv_epoch"]
    native = next(loop for loop in nodes(after, tirx.For) if loop.loop_var.name.startswith("gemm_sub_k_"))
    assert int(native.extent) == reduction // 128
    native_buffers = [
        buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers
        if buffer.scope() in ("shared.l0a", "shared.l0b")
    ]
    full = [buffer for buffer in native_buffers if not buffer.name.startswith("gemm_tail_")]
    tails = [buffer for buffer in native_buffers if buffer.name.startswith("gemm_tail_")]
    assert len(full) == 2
    assert all(_owners(after, buffer.data) == [native.loop_var.name] for buffer in full)
    assert len(tails) == (2 if reduction % 128 else 0)
    assert all(_owners(after, buffer.data) == ["kv_epoch"] for buffer in tails)
    original = calls(lowered, "tl.tileop.gemm")
    actual = calls(after, "tl.tileop.gemm")
    assert len(actual) == len(original) == 1 + bool(reduction % 128)
    for result, source in zip(actual, original):
        tvm.ir.assert_structural_equal(result, source)
