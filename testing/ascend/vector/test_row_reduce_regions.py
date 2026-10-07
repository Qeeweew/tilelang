"""Late row-reduce regions preserve exact lanes, origins and owner planes."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, region


TARGET = {"kind": "ascend", "arch": "dav-2201"}
PREFIX = "tl.ascend_membase.repeat_reduce_"


def _compact_strides(shape):
    strides = []
    size = 1
    for extent in reversed(shape):
        strides.insert(0, size)
        size *= extent
    return strides


def _module(call, source, destination, *, annotations=None, body=None):
    mod = kernel(tirx.Evaluate(call) if body is None else body,
                 buffers=[source, destination], annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod


def _fixture(*, kind="sum", rows=3, lanes=32, width=None, source_row=0,
             source_column=0, output_row=0, retained=False, leading=(),
             selectors=None, explicit_strides=False, index_dtype="int32"):
    width = lanes if width is None else width
    selectors = [0] * len(leading) if selectors is None else list(selectors)
    source_shape = (*leading, source_row + rows, width)
    # Each output plane has physical padding, so an independent manual slot
    # begins on a 32-byte boundary even when the logical row count is three.
    destination_shape = (
        *leading, output_row + (rows + 7) // 8 * 8, *((1,) if retained else ()))

    def index(value):
        return tirx.IntImm(index_dtype, value) if isinstance(value, int) else value

    def buffer(shape, name):
        return tirx.decl_buffer(
            tuple(index(extent) for extent in shape), "float32", name,
            scope="shared.dyn",
            strides=[index(stride) for stride in _compact_strides(shape)] if explicit_strides else None)

    source = buffer(source_shape, "source")
    destination = buffer(destination_shape, "destination")
    source_indices = [*selectors, source_row, source_column]
    destination_indices = [*selectors, output_row, *((0,) if retained else ())]
    call = tirx.Call("void", tvm.ir.Op.get(PREFIX + kind + "_rows_f32"), [
        region(destination, 2, [index(value) for value in destination_indices],
               [*([index(1)] * len(leading)), index(rows), *((index(1),) if retained else ())]),
        region(source, 1, [index(value) for value in source_indices],
               [*([index(1)] * len(leading)), index(rows), index(lanes)]),
        tirx.IntImm("uint8", rows), tirx.IntImm("uint16", width // 8), tirx.IntImm("uint8", lanes),
    ])
    return call, source, destination


def _lower(mod):
    with tvm.target.Target(TARGET):
        return tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(mod))


def _native(mod, kind="sum"):
    (call,) = calls(mod, PREFIX + kind + "_rows_f32")
    assert len(call.args) == 5
    assert all(pointer.op.name == "tirx.tvm_access_ptr" for pointer in call.args[:2])
    return call


def _equal(actual, expected):
    analyzer = tvm.arith.Analyzer()
    assert len(actual) == len(expected)
    assert all(analyzer.can_prove_equal(a, b) for a, b in zip(actual, expected)), (actual, expected)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("lanes", [8, 16, 32, 48, 56, 64])
def test_compact_source_uses_its_real_lane_count_and_bounded_span(kind, lanes):
    call, source, destination = _fixture(kind=kind, lanes=lanes)
    native = _native(_lower(_module(call, source, destination)), kind)
    dst, src = native.args[:2]
    _equal([dst.args[2], dst.args[3], dst.args[4]], [0, 3, 2])
    _equal([src.args[2], src.args[3], src.args[4]], [0, 3 * lanes, 1])
    assert [str(arg.dtype) for arg in native.args[2:]] == ["uint8", "uint16", "uint8"]
    assert [int(arg) for arg in native.args[2:]] == [3, lanes // 8, lanes]


@pytest.mark.parametrize("retained", [False, True], ids=["rows", "rows-unit-axis"])
def test_full64_chunk_preserves_nonzero_origins_and_last_backing_element(retained):
    call, source, destination = _fixture(
        lanes=64, width=192, source_row=1, source_column=128, output_row=8,
        retained=retained, explicit_strides=True)
    native = _native(_lower(_module(call, source, destination)))
    dst, src = native.args[:2]
    _equal([dst.args[2], dst.args[3]], [8, 3])
    _equal([src.args[2], src.args[3]], [320, 448])
    _equal([src.args[2] + src.args[3]], [4 * 192])


@pytest.mark.parametrize("retained", [False, True], ids=["rows", "rows-unit-axis"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_private_manual_planes_keep_dynamic_selector_and_compact_output(retained, index_dtype):
    epoch = tirx.Var("epoch", index_dtype)
    call, source, destination = _fixture(
        lanes=48, source_row=1, output_row=8, retained=retained,
        leading=(2, 3), selectors=[epoch % 2, tirx.IntImm(index_dtype, 1)],
        explicit_strides=True, index_dtype=index_dtype)
    before = _module(call, source, destination, annotations={
        "tl.manual_multi_buffer": {source.data: 2, destination.data: 2},
        "tl.ascend_aiv_private_validated": [source.data, destination.data],
    }, body=tirx.For(epoch, tirx.IntImm(index_dtype, 0), tirx.IntImm(index_dtype, 7),
                    tirx.ForKind.SERIAL, tirx.Evaluate(call)))
    native = _native(_lower(before))
    dst, src = native.args[:2]
    slot = tirx.Cast("int64", epoch % 2)
    _equal([dst.args[2], dst.args[3]], [slot * 48 + 24, 3])
    _equal([src.args[2], src.args[3]], [slot * 576 + 240, 144])


@pytest.mark.parametrize("rows", [1, 255])
def test_native_repeat_endpoints_keep_exact_output_span(rows):
    call, source, destination = _fixture(rows=rows, lanes=16, retained=True)
    native = _native(_lower(_module(call, source, destination)))
    _equal([native.args[0].args[3], native.args[1].args[3]], [rows, rows * 16])


@pytest.mark.parametrize("bad,diagnostic", [
    ("four-operands", "expects five operands"),
    ("rows-zero", "uint8 rows"),
    ("rows-type", "uint8 rows"),
    ("lanes-small", "uint8 valid lanes"),
    ("lanes-nonblock", "uint8 valid lanes"),
    ("lanes-large", "uint8 valid lanes"),
    ("lanes-type", "uint8 valid lanes"),
    ("stride-small", "uint16 source row stride"),
    ("stride-type", "uint16 source row stride"),
    ("partial-padded", "compact source row stride"),
    ("full-nonvl-pitch", "whole-VL stride"),
])
def test_late_scalar_schema_rejects_unproved_lane_or_stride_contract(bad, diagnostic):
    call, source, destination = _fixture()
    args = list(call.args)
    changes = {
        "rows-zero": (2, tirx.IntImm("uint8", 0)),
        "rows-type": (2, tirx.IntImm("int32", 3)),
        "lanes-small": (4, tirx.IntImm("uint8", 0)),
        "lanes-nonblock": (4, tirx.IntImm("uint8", 12)),
        "lanes-large": (4, tirx.IntImm("uint8", 72)),
        "lanes-type": (4, tirx.IntImm("int32", 32)),
        "stride-small": (3, tirx.IntImm("uint16", 3)),
        "stride-type": (3, tirx.IntImm("int32", 4)),
        "partial-padded": (3, tirx.IntImm("uint16", 8)),
    }
    if bad == "four-operands":
        args.pop()
    elif bad == "full-nonvl-pitch":
        args[3:5] = [tirx.IntImm("uint16", 9), tirx.IntImm("uint8", 64)]
    else:
        index, value = changes[bad]
        args[index] = value
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        malformed = tirx.Call(call.dtype, call.op, args)
        _lower(_module(malformed, source, destination))


@pytest.mark.parametrize("bad,diagnostic", [
    ("source-rows", "exact \\[rows, valid_lanes\\]"),
    ("source-lanes", "exact \\[rows, valid_lanes\\]"),
    ("destination-rows", "exactly rows compact elements"),
    ("source-out-of-bounds", "inside its backing"),
    ("destination-out-of-bounds", "inside its backing"),
    ("source-width", "physical row width"),
    ("source-pitch", "physical row pitch"),
    ("source-nonunit-column", "contiguous input/output"),
    ("destination-pitch", "physical row pitch"),
    ("destination-nonunit-column", "contiguous input/output"),
    ("source-alignment", "32-byte aligned bases"),
    ("destination-alignment", "32-byte aligned bases"),
    ("source-mask", "access mask"),
    ("destination-mask", "access mask"),
    ("source-untyped", "exact typed tl.region"),
    ("source-dtype", "FP32 UB storage"),
    ("source-scope", "FP32 UB storage"),
])
def test_late_regions_reject_wrong_span_backing_or_effect_metadata(bad, diagnostic):
    call, source, destination = _fixture(retained=True)
    args = list(call.args)
    operand = 0 if bad.startswith("destination") else 1
    data = list(args[operand].args)
    buffer = destination if operand == 0 else source
    if bad == "source-rows":
        data[-2] = tirx.IntImm("int32", 2)
    elif bad == "source-lanes":
        data[-1] = tirx.IntImm("int32", 24)
    elif bad == "destination-rows":
        data[-2] = tirx.IntImm("int32", 2)
    elif bad in ("source-out-of-bounds", "destination-out-of-bounds"):
        data[0] = buffer[2, 0] if operand == 1 else buffer[6, 0]
    elif bad.endswith("-mask"):
        data[1] = tirx.IntImm("int32", 2 if operand == 1 else 1)
    elif bad == "source-untyped":
        args[1] = source[0, 0]
    else:
        shape = (3, 40) if bad == "source-width" else tuple(buffer.shape)
        strides = {
            "source-pitch": [40, 1], "source-nonunit-column": [32, 2],
            "destination-pitch": [2, 1], "destination-nonunit-column": [1, 2],
        }.get(bad)
        replacement = tirx.decl_buffer(
            shape, "float16" if bad == "source-dtype" else "float32", "invalid",
            scope="global" if bad == "source-scope" else "shared.dyn", strides=strides,
            elem_offset=1 if bad.endswith("-alignment") else 0)
        data[0] = replacement[0, 0]
        if operand == 0:
            destination = replacement
        else:
            source = replacement
    if bad != "source-untyped":
        args[operand] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), data)
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _lower(_module(tirx.Call(call.dtype, call.op, args), source, destination))


def test_full64_aligned_column_still_requires_whole_vl_origin():
    call, source, destination = _fixture(lanes=64, width=128, source_column=8)
    with pytest.raises(tvm.error.InternalError, match="start at a whole VL"):
        _lower(_module(call, source, destination))


@pytest.mark.parametrize("operand", [0, 1], ids=["destination", "source"])
def test_leading_owner_or_version_axis_must_select_one_plane(operand):
    call, source, destination = _fixture(leading=(2,))
    args = list(call.args)
    data = list(args[operand].args)
    data[2] = tirx.IntImm("int32", 2)
    args[operand] = tirx.Call("handle", tvm.ir.Op.get("tl.region"), data)
    with pytest.raises(tvm.error.InternalError, match="leading/version axes must be singleton"):
        _lower(_module(tirx.Call(call.dtype, call.op, args), source, destination))
