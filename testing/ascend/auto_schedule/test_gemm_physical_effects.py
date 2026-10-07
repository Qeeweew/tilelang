"""Physical GEMM generations follow clear semantics, not region RW masks.

These host-only InsertSync inputs keep native C regions and canonical layouts.
Explicit conflict hints supply real completion between the two C storages;
the tests inspect the ordinary alias proof, without changing its requirements.
"""

import pytest
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import (
    make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
    make_ascend_major_k_layout,
)
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


_GEMM = "tl.tileop.gemm"


def _fixture(clear, c_mask, *, partial=False, scalar_old_c=False, annotation_old_c=False):
    a = tirx.decl_buffer((32, 16), "float16", name="a", scope="shared.l0a")
    b = tirx.decl_buffer((32, 16), "float16", name="b", scope="shared.l0b")
    accumulator, scratch = [tirx.decl_buffer((32, 32), "float32", name=name, scope="shared.l0c")
                            for name in ("accumulator", "scratch")]
    first, middle, final = [tirx.decl_buffer((32, 32), "float32", name=name)
                            for name in ("first", "middle", "final")]
    flag = tirx.Var("clear", "bool")
    clear_value = {"true": tirx.const(True, "bool"), "false": tirx.const(False, "bool"),
                   "dynamic": flag, "old-c": accumulator[0, 0] > 0}[clear]

    def mad(c, clear_value, mask=2, width=32, scalar_read=False, annotation_read=False):
        call = T.gemm(a, b, c, transpose_B=True, clear_accum=True)
        args = list(call.args)
        args[1] = region(b, 1, shape=[width, 16])
        args[2] = region(c, mask, shape=[32, width])
        args[6], args[9] = tirx.IntImm("int32", width), clear_value
        if scalar_read:
            # A real scalar access in a non-region operand must survive even
            # when clear=True removes the accumulator-region read. Imported
            # TIR may carry such bookkeeping; this is no native-support claim
            # for a nonzero C coordinate on 2201.
            args[11] = tirx.Cast("int32", accumulator[0, 0])
        annotations = dict(call.annotations)
        if annotation_read:
            annotations["test_scalar_effect"] = tirx.Cast("int32", accumulator[0, 0])
        return tirx.Evaluate(tirx.Call(call.dtype, call.op, args, annotations=annotations))

    initial = mad(accumulator, tirx.const(True, "bool"))
    temporary = mad(scratch, tirx.const(True, "bool"))
    update = mad(accumulator, clear_value, c_mask, 16 if partial else 32,
                 scalar_read=scalar_old_c, annotation_read=annotation_old_c)
    body = seq(*(unit(statement, core=2) for statement in (
        initial, copy(accumulator, first), temporary, copy(scratch, middle),
        update, copy(accumulator, final),
    )))
    annotations = {
        "layout_map": {a: make_ascend_l0a_2201_layout(a), b: make_ascend_major_k_layout(b),
                       accumulator: make_ascend_l0c_layout(accumulator),
                       scratch: make_ascend_l0c_layout(scratch)},
        "tl.root_conflict_hints": [[accumulator.data, scratch.data, -1, tirx.const(True, "bool")]],
    }
    before = kernel(body, buffers=[a, b, accumulator, scratch],
                    params=[first, middle, final] + ([flag] if clear == "dynamic" else []),
                    annotations=annotations)
    before["main"] = before["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return before, accumulator, scratch


def _check(before, accumulator, scratch, *, reusable):
    original = calls(before, _GEMM)[-1]
    with before["main"].attrs["target"]:
        after = transform.InsertSync()(before)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    aliases = root.annotations.get("tl.buffer_alias_map", {})
    assert (scratch.data in aliases.get(accumulator.data, [])) == reusable
    assert (accumulator.data in aliases.get(scratch.data, [])) == reusable
    # The detector consumes the typed effects; it does not rewrite regions,
    # clear expressions, or physical-layout identities to obtain that result.
    final = calls(after, _GEMM)[-1]
    for index in (0, 1, 2, 9, 11, 12):
        tvm.ir.assert_structural_equal(final.args[index], original.args[index])
    tvm.ir.assert_structural_equal(final.annotations, original.annotations)
    for buffer in (accumulator, scratch):
        tvm.ir.assert_structural_equal(root.annotations["layout_map"][buffer], make_ascend_l0c_layout(buffer))
    assert calls(after, "tl.ascend_set_flag") and calls(after, "tl.ascend_wait_flag")


@pytest.mark.parametrize("c_mask", [2, 3], ids=["write-region", "rw-region"])
def test_static_clear_replaces_the_old_c_generation(c_mask):
    _check(*_fixture("true", c_mask), reusable=True)


@pytest.mark.parametrize("clear", ["false", "dynamic", "old-c"])
@pytest.mark.parametrize("c_mask", [2, 3], ids=["write-region", "rw-region"])
def test_accumulating_or_unknown_clear_keeps_old_c_live(clear, c_mask):
    _check(*_fixture(clear, c_mask), reusable=False)


@pytest.mark.parametrize("c_mask", [2, 3], ids=["write-region", "rw-region"])
def test_partial_clear_does_not_define_the_unwritten_c_columns(c_mask):
    _check(*_fixture("true", c_mask, partial=True), reusable=False)


def test_static_clear_preserves_a_real_scalar_old_c_access():
    _check(*_fixture("true", 2, scalar_old_c=True), reusable=False)


def test_static_clear_preserves_scalar_reads_in_gemm_annotations():
    _check(*_fixture("true", 2, annotation_old_c=True), reusable=False)
