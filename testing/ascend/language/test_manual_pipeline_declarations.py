"""Frontend declarations preserve existing loop and ownership contracts."""

import pytest
import tvm
from tvm import tirx
from tilelang.ascend import language as T
from testing.ascend._ir import nodes


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("start", [0, 3])
def test_pipelined_manual_annotations_match_explicit_serial_contract(preserve, start):
    def make(use_api):
        @T.prim_func
        def main(A: T.Tensor((16,), "float32")):
            with T.Kernel(1):
                for i in (
                    T.Pipelined(start, 16, num_stages=0, annotations={"enable_offset": True, "tl.ascend_preserve_issue_order": preserve})
                    if use_api
                    else T.serial(start, 16, annotations={"enable_offset": True, "tl.ascend_preserve_issue_order": preserve,
                                                        "tl.ascend_auto_pipeline": False})
                ):
                    with T.Stage(1):
                        A[i] = 1.0
        return main
    tvm.ir.assert_structural_equal(make(True), make(False))


@pytest.mark.parametrize("extra", [False, True])
def test_private_scope_keeps_allocation_order_and_conditional_members(extra):
    def make(use_api):
        @T.prim_func
        def main():
            with T.MixedKernel(1, sids=2):
                _shared = T.alloc_shared((32,), "float32")
                if use_api:
                    with T.AIVPrivate():
                        a = T.alloc_shared((32,), "float32")
                        if extra:
                            b = T.alloc_shared((2, 32), "float32")
                else:
                    a = T.alloc_shared((32,), "float32")
                    if extra:
                        b = T.alloc_shared((2, 32), "float32")
                        T.annotate_aiv_private(a, b)
                    else:
                        T.annotate_aiv_private(a)
                T.evaluate(0)
        return main
    tvm.ir.assert_structural_equal(make(True), make(False))
    root = next(b for b in nodes(make(True), tirx.SBlock) if "tl.ascend_aiv_private" in b.annotations)
    assert [v.name for v in root.annotations["tl.ascend_aiv_private"]] == (["a", "b"] if extra else ["a"])


@pytest.mark.parametrize("bad", ["empty", "l1", "nested", "explicit", "loop"])
def test_private_scope_rejects_invalid_declarations_and_does_not_leak(bad):
    with pytest.raises(ValueError):
        @T.prim_func
        def invalid():
            with T.MixedKernel(1, sids=2):
                with T.AIVPrivate():
                    if bad == "l1":
                        _a = T.alloc_l1((32,), "float32")
                    if bad == "nested":
                        with T.AIVPrivate():
                            _b = T.alloc_shared((32,), "float32")
                    if bad == "explicit":
                        c = T.alloc_shared((32,), "float32")
                        T.annotate_aiv_private(c)
                    if bad == "loop":
                        for _i in T.serial(2):
                            with T.AIVPrivate():
                                _d = T.alloc_shared((32,), "float32")
                    T.evaluate(0)

    @T.prim_func
    def valid():
        with T.MixedKernel(1, sids=2):
            with T.AIVPrivate():
                _a = T.alloc_shared((32,), "float32")
            T.evaluate(0)
    root = next(b for b in nodes(valid, tirx.SBlock) if "tl.ascend_aiv_private" in b.annotations)
    assert len(root.annotations["tl.ascend_aiv_private"]) == 1
