"""Native UB region boundaries reject overflow before layout/pointer lowering."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, region


def _target(body, buffers):
    mod = kernel(body, buffers=buffers)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    return tvm.IRModule({"main": mod["main"].with_attr("target", target)})


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("operand", ["source", "destination"])
def test_ub_copy_does_not_accept_a_wrapped_region_endpoint(index_dtype, operand):
    bits = 32 if index_dtype == "int32" else 64
    maximum = (1 << (bits - 1)) - 1
    origin = tirx.IntImm(index_dtype, maximum - 7)
    source, destination = [
        tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn")
        for name in ("source", "destination")
    ]
    body = copy(
        source, destination,
        src_indices=[origin if operand == "source" else 0],
        dst_indices=[origin if operand == "destination" else 0],
        src_shape=[16], dst_shape=[16],
    )
    with pytest.raises(ValueError, match="within their private allocations"):
        transform.AscendLowerVectorTiles()(_target(body, [source, destination]))


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("operand", ["source", "destination"])
def test_reduce_does_not_accept_a_wrapped_region_endpoint(index_dtype, operand):
    bits = 32 if index_dtype == "int32" else 64
    maximum = (1 << (bits - 1)) - 1
    source = tirx.decl_buffer((1, 64), "float32", name="source", scope="shared.dyn")
    destination = tirx.decl_buffer((1,), "float32", name="destination", scope="shared.dyn")
    source_origin = tirx.IntImm(index_dtype, maximum - 7) if operand == "source" else 0
    destination_origin = tirx.IntImm(index_dtype, maximum) if operand == "destination" else 0
    call = tirx.Call(
        "handle", tvm.ir.Op.get("tl.tileop.reduce"),
        [
            region(source, 1, indices=[0, source_origin], shape=[1, 16]),
            region(destination, 2, indices=[destination_origin], shape=[1]),
            tirx.StringImm("sum"), 1, tirx.const(True, "bool"),
        ],
    )
    with pytest.raises(ValueError, match="regions must lie within their UB buffers"):
        transform.AscendLowerVectorReduce()(_target(tirx.Evaluate(call), [source, destination]))


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_wide_aligned_last_in_bounds_ub_copy_is_retained(index_dtype):
    source, destination = [
        tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn")
        for name in ("source", "destination")
    ]
    body = copy(
        source, destination, src_indices=[tirx.IntImm(index_dtype, 48)],
        src_shape=[16], dst_shape=[16],
    )
    after = transform.AscendLowerVectorTiles()(_target(body, [source, destination]))
    (operation,) = calls(after, "tl.ascend_membase.copy_bits")
    assert int(operation.args[-1]) == 16
