"""Ordinary Ascend UB reductions retain logical intent for target lowering."""

import pytest
from tilelang.ascend import language as T
from tvm import tirx

from testing.ascend._ir import calls, nodes


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("clear", [False, True])
def test_ordinary_ub_reduce_has_no_fragment_round_trip(kind, clear):
    @T.prim_func
    def main(A: T.Tensor((3, 144), "float32"), O: T.Tensor((3,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((3, 144), "float32")
            output = T.alloc_shared((3,), "float32")
            T.copy(A, source)
            T.fill(output, 1.0)
            T.reduce(source[:, 0:130], output, kind, 1, clear)
            T.copy(output, O)

    (reduce,) = calls(main, "tl.tileop.reduce")
    assert reduce.args[2].value == kind
    assert int(reduce.args[3]) == 1 and bool(reduce.args[4]) == clear
    assert tuple(int(extent) for extent in reduce.args[0].args[2:]) == (3, 130)
    assert int(reduce.args[1].args[1]) == (2 if clear else 3)
    assert all(buffer.scope() != "local.fragment" for block in nodes(main, tirx.SBlock) for buffer in block.alloc_buffers)
    assert len(calls(main, "tl.tileop.ascend_copy")) == 2
