"""Late FP32 BlockReduce validates complete-VL pointer effects before remap."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes


TARGET = {"kind": "ascend", "arch": "dav-2201"}
PREFIX = "tl.ascend_membase.datablock_reduce_"


def _pointer(buffer, indices, extent, mask):
    return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [
        buffer[tuple(indices)], tirx.IntImm("int32", extent), tirx.IntImm("int32", mask),
    ])


def _call(source, destination, *, kind="sum", count=64,
          source_indices=(0,), destination_indices=(0,)):
    return tirx.Call("void", tvm.ir.Op.get(PREFIX + kind), [
        _pointer(destination, destination_indices, count // 8, 2),
        _pointer(source, source_indices, count, 1),
        tirx.IntImm("uint32", count),
    ])


def _module(body, source, destination, *, params=(), annotations=None):
    mod = kernel(body, buffers=[source, destination], params=params, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod


def _lower(mod):
    with tvm.target.Target(TARGET):
        return tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(mod))


def _native(mod, kind="sum"):
    (call,) = calls(mod, PREFIX + kind)
    assert len(call.args) == 3
    assert call.op.get_attr("TGlobalSymbol") == "asc_datablock_reduce_" + kind
    assert str(call.args[2].dtype) == "uint32"
    for pointer in call.args[:2]:
        assert pointer.op.name == "tirx.tvm_access_ptr"
        assert str(pointer.args[0].dtype) == "float32"
    return call


def _equal(actual, expected):
    analyzer = tvm.arith.Analyzer()
    assert len(actual) == len(expected)
    assert all(analyzer.can_prove_equal(a, b) for a, b in zip(actual, expected)), (actual, expected)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_complete_count_keeps_nonzero_row_bases_and_exact_read_write_extents(kind):
    source = tirx.decl_buffer((17, 1024), "float32", "source", scope="shared.dyn", strides=[1024, 1])
    destination = tirx.decl_buffer((17, 128), "float32", "destination", scope="shared.dyn", strides=[128, 1])
    # 256 implicit VLs exceed the explicit uint8-repeat limit. Count mode
    # consumes sixteen full rows and ends exactly at each backing's boundary.
    call = _call(source, destination, kind=kind, count=16384,
                 source_indices=(1, 0), destination_indices=(1, 0))
    with tvm.target.Target(TARGET):
        lowered = transform.AscendLowerTileOp()(_module(tirx.Evaluate(call), source, destination))
    (typed,) = calls(lowered, PREFIX + kind)
    assert all(pointer.op.name == "tl.access_ptr" and len(pointer.args) == 3 for pointer in typed.args[:2])
    _equal(typed.args[0].args[1:], [2048, 2])
    _equal(typed.args[1].args[1:], [16384, 1])
    native = _native(_lower(_module(tirx.Evaluate(call), source, destination)), kind)
    dst, src = native.args[:2]
    _equal([dst.args[2], dst.args[3], dst.args[4]], [128, 2048, 2])
    _equal([src.args[2], src.args[3], src.args[4]], [1024, 16384, 1])
    _equal([dst.args[2] + dst.args[3], src.args[2] + src.args[3]], [2176, 17408])
    assert int(native.args[2]) == 16384


def test_materialized_manual_planes_preserve_a_wide_dynamic_selector():
    epoch = tirx.Var("epoch", "int64")
    source = tirx.decl_buffer((2, 17, 1024), "float32", "source", scope="shared.dyn", strides=[17408, 1024, 1])
    destination = tirx.decl_buffer((2, 17, 128), "float32", "destination", scope="shared.dyn", strides=[2176, 128, 1])
    call = _call(source, destination, count=16384,
                 source_indices=(epoch % 2, 1, 0), destination_indices=(epoch % 2, 1, 0))
    loop = tirx.For(epoch, tirx.IntImm("int64", 1 << 40), tirx.IntImm("int64", 7),
                    tirx.ForKind.SERIAL, tirx.Evaluate(call))
    after = _lower(_module(loop, source, destination, annotations={
        "tl.manual_multi_buffer": {source.data: 2, destination.data: 2},
        "tl.ascend_aiv_private_validated": [source.data, destination.data],
    }))
    native = _native(after)
    (actual_loop,) = nodes(after, tirx.For)
    slot = actual_loop.loop_var % 2
    dst, src = native.args[:2]
    _equal([dst.args[2], dst.args[3], dst.args[4]], [slot * 2176 + 128, 2048, 2])
    _equal([src.args[2], src.args[3], src.args[4]], [slot * 17408 + 1024, 16384, 1])
    assert str(actual_loop.loop_var.dtype) == "int64"


def test_aligned_element_offsets_do_not_reduce_logical_backing_capacity():
    source = tirx.decl_buffer((64,), "float32", "source", scope="shared.dyn", elem_offset=8)
    destination = tirx.decl_buffer((8,), "float32", "destination", scope="shared.dyn", elem_offset=16)
    call = _call(source, destination)
    native = _native(_lower(_module(tirx.Evaluate(call), source, destination)))
    _equal(native.args[0].args[2:], [16, 8, 2])
    _equal(native.args[1].args[2:], [8, 64, 1])


@pytest.mark.parametrize("bad,diagnostic", [
    ("count-zero", "positive uint32 count"),
    ("count-tail", "complete 64-lane repeats"),
    ("count-type", "positive uint32 count"),
    ("count-dynamic", "positive uint32 count"),
    ("untyped-source", "typed tl.access_ptr"),
    ("four-pointer-arguments", "typed tl.access_ptr"),
    ("half-storage", "FP32 UB storage"),
    ("global-storage", "FP32 UB storage"),
    ("axis-separated", "FP32 UB storage"),
    ("read-write-mask", "access mask differs"),
    ("source-extent", "pointer extent must match"),
    ("destination-extent", "pointer extent must match"),
    ("padded-strides", "compact physical storage"),
    ("unaligned-base", "32-byte aligned bases"),
    ("negative-base", "base must lie inside"),
    ("past-end-base", "base must lie inside"),
    ("source-span", "pointer span must lie inside"),
    ("destination-span", "pointer span must lie inside"),
    ("shared-storage", "independent input/output storage"),
])
def test_native_pointer_contract_rejects_unproved_storage_or_effects(bad, diagnostic):
    source = tirx.decl_buffer((128,), "float32", "source", scope="shared.dyn")
    destination = tirx.decl_buffer((16,), "float32", "destination", scope="shared.dyn")
    source_indices, destination_indices = (0,), (0,)
    if bad in ("half-storage", "global-storage", "axis-separated", "padded-strides"):
        source = tirx.decl_buffer(
            (2, 64) if bad in ("axis-separated", "padded-strides") else (128,),
            "float16" if bad == "half-storage" else "float32", "invalid_source",
            scope="global" if bad == "global-storage" else "shared.dyn",
            strides=[72, 1] if bad == "padded-strides" else None,
            axis_separators=[1] if bad == "axis-separated" else None)
        source_indices = (0, 0) if len(source.shape) == 2 else (0,)
    elif bad in ("unaligned-base", "negative-base", "past-end-base", "source-span"):
        source_indices = ({"unaligned-base": 1, "negative-base": -8,
                           "past-end-base": 128, "source-span": 72}[bad],)
    elif bad == "destination-span":
        destination = tirx.decl_buffer((12,), "float32", "destination", scope="shared.dyn")
        destination_indices = (8,)
    elif bad == "shared-storage":
        destination = tirx.decl_buffer((16,), "float32", "alias", data=source.data, scope="shared.dyn")
    call = _call(source, destination, source_indices=source_indices, destination_indices=destination_indices)
    args = list(call.args)
    params = []
    if bad.startswith("count-"):
        args[2] = {
            "count-zero": tirx.IntImm("uint32", 0),
            "count-tail": tirx.IntImm("uint32", 65),
            "count-type": tirx.IntImm("int32", 64),
            "count-dynamic": tirx.Var("count", "uint32"),
        }[bad]
        if bad == "count-dynamic":
            params = [args[2]]
    elif bad == "untyped-source":
        args[1] = source.data
    elif bad in ("four-pointer-arguments", "read-write-mask", "source-extent", "destination-extent"):
        operand = 0 if bad in ("read-write-mask", "destination-extent") else 1
        pointer_args = list(args[operand].args)
        if bad == "four-pointer-arguments":
            pointer_args.append(tirx.IntImm("int32", 0))
        elif bad == "read-write-mask":
            pointer_args[2] = tirx.IntImm("int32", 3)
        else:
            pointer_args[1] = tirx.IntImm("int32", 9 if operand == 0 else 63)
        args[operand] = tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), pointer_args)
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        malformed = tirx.Call(call.dtype, call.op, args)
        _lower(_module(tirx.Evaluate(malformed), source, destination, params=params))
