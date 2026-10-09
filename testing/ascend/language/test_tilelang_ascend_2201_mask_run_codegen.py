"""Only proven straight-line 2201 calls share normal/full mask entry."""

import re

import pytest
from tilelang import tvm
from tvm import tirx


FULL_MASK = "asc_set_vector_mask(~uint64_t(0), ~uint64_t(0));"
NORMAL_MASK = "asc_set_mask_norm();"


def _pointer(name, dtype="float32"):
    return tirx.Var(name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), "shared.dyn"))


def _native(name, *args):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_membase." + name), list(args)))


def _brcb(dst, src, repeats=1):
    return _native("brcb_f32", dst, src, repeats)


def _row(name, dst, src, factor, repeats=8):
    return _native(name + "_rows_f32", dst, src, factor, repeats, 0, 16)


def _reduce(name, dst, src, lanes=64, rows=8, stride=None):
    stride = (16 if lanes == 64 else lanes // 8) if stride is None else stride
    return _native("repeat_reduce_" + name + "_rows_f32", dst, src,
                   tirx.const(rows, "uint8"), tirx.const(stride, "uint16"),
                   tirx.const(lanes, "uint8"))


def _barrier(pipe="PIPE_V"):
    return tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_pipe_barrier"), [pipe]))


def _function(name, body, params, arch="dav-2201"):
    return tirx.PrimFunc(list(params), body).with_attr({
        "global_symbol": name, "calling_conv": 2,
        "target": tvm.target.Target({"kind": "ascend", "arch": arch}),
    })


def _source(body, params, arch="dav-2201"):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    build = tvm.ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    return build(tvm.IRModule({"main": _function("main", body, params, arch)}), target).inspect_source()


def _entries(source, count):
    assert source.count(NORMAL_MASK) == count, source
    assert source.count(FULL_MASK) == count, source


def test_all_native_families_share_entry_across_vector_barriers():
    dst, src, factor = [_pointer(name) for name in ["dst", "src", "factor"]]
    calls = [
        _brcb(factor, src), _row("sub", dst, src, factor),
        _row("mul", dst, dst, factor), _row("div", dst, dst, factor),
        _reduce("sum", factor, dst), _reduce("max", factor, dst),
    ]
    body = tirx.SeqStmt([item for call in calls for item in [call, _barrier()]])
    source = _source(body, [dst, src, factor])
    _entries(source, 1)
    # Every mathematical call and completion remains, in its original order.
    operations = re.findall(r"\b(asc_brcb|asc_sub|asc_mul|asc_div|asc_repeat_reduce_sum|asc_repeat_reduce_max_only_value|asc_sync_pipe)\(", source)
    expected = ["asc_brcb", "asc_sub", "asc_mul", "asc_div", "asc_repeat_reduce_sum", "asc_repeat_reduce_max_only_value"]
    assert operations == [item for call in expected for item in [call, "asc_sync_pipe"]] + ["asc_sync_pipe"], source
    assert source.count("asc_sync_pipe(PIPE_V);") == 6, source
    assert source.index(FULL_MASK) < source.index("asc_brcb("), source


@pytest.mark.parametrize("breaker", ["unknown", "count", "cast", "user_mask", "other_pipe"])
def test_other_calls_end_the_run(breaker):
    dst, src = _pointer("dst"), _pointer("src")
    half = _pointer("half", "float16")
    if breaker == "unknown":
        middle = tirx.Evaluate(tirx.call_extern("void", "unknown_vector_call", dst))
        marker = "unknown_vector_call("
    elif breaker == "count":
        middle = _native("exp", dst, src, 64)
        marker = "asc_exp("
    elif breaker == "cast":
        middle = _native("float2half_rn", half, src, 64)
        marker = "asc_float2half_rn("
    elif breaker == "user_mask":
        middle = tirx.Evaluate(tirx.call_extern("void", "asc_set_vector_mask", tirx.const(0, "uint64"), tirx.const(1, "uint64")))
        marker = "asc_set_vector_mask((uint64_t)0, (uint64_t)1)"
    else:
        middle, marker = _barrier("PIPE_MTE2"), "asc_sync_pipe(PIPE_MTE2);"
    source = _source(tirx.SeqStmt([_brcb(dst, src), middle, _brcb(dst, src)]), [dst, src, half])
    _entries(source, 2)
    assert source.count("asc_brcb(") == 2, source
    if breaker == "user_mask":
        assert source.count("asc_set_vector_mask(") == 3, source
        positions = [match.start() for match in re.finditer("asc_set_vector_mask\\(", source)]
        assert positions[0] < source.index("asc_brcb(") < positions[1] < positions[2] < source.rindex("asc_brcb("), source
    else:
        assert source.index("asc_brcb(") < source.index(marker) < source.rindex(FULL_MASK) < source.rindex("asc_brcb("), source


@pytest.mark.parametrize("boundary", ["if_singleton", "if_sequence", "for", "while"])
def test_control_boundaries_never_inherit_or_export_mask_state(boundary):
    dst, src = _pointer("dst"), _pointer("src")
    enabled = tirx.Var("enabled", "int32")
    pair = tirx.SeqStmt([_brcb(dst, src), _barrier(), _brcb(dst, src)])
    if boundary.startswith("if"):
        branch = _brcb(dst, src) if boundary == "if_singleton" else pair
        middle = tirx.IfThenElse(enabled > 0, branch, branch)
        entries, calls = 4, 4 if boundary == "if_singleton" else 6
    elif boundary == "for":
        middle = tirx.For(tirx.Var("row", "int32"), 0, enabled, tirx.ForKind.SERIAL, pair)
        entries, calls = 3, 4
    else:
        middle = tirx.While(enabled > 0, pair)
        entries, calls = 3, 4
    source = _source(tirx.SeqStmt([_brcb(dst, src), middle, _brcb(dst, src)]), [dst, src, enabled])
    _entries(source, entries)
    assert source.count("asc_brcb(") == calls, source
    assert source.rindex(FULL_MASK) < source.rindex("asc_brcb("), source


@pytest.mark.parametrize("dynamic", [False, True], ids=["zero", "dynamic_zero_possible"])
@pytest.mark.parametrize("family", ["brcb", "row"])
def test_zero_or_dynamic_repeat_still_establishes_unconditional_mask(dynamic, family):
    dst, src, factor = [_pointer(name) for name in ["dst", "src", "factor"]]
    count = tirx.Var("count", "int32")
    repeats = count if dynamic else 0
    first = _brcb(dst, src, repeats) if family == "brcb" else _row("sub", dst, src, factor, repeats)
    source = _source(tirx.SeqStmt([first, _barrier(), _brcb(dst, src)]), [dst, src, factor, count])
    _entries(source, 1)
    marker = "asc_brcb(" if family == "brcb" else "asc_sub("
    assert source.index(FULL_MASK) < source.index(marker), source
    assert "if (" not in source, source
    assert source.count("asc_brcb(") == (2 if family == "brcb" else 1), source


@pytest.mark.parametrize("family", ["brcb", "row", "sum", "max"])
def test_unknown_argument_is_evaluated_once_before_current_mask(family):
    dst, src = _pointer("dst"), _pointer("src")
    unknown_pointer = tirx.call_extern("handle", "unknown_masked_pointer", dst)
    middle = (_brcb(unknown_pointer, src) if family == "brcb" else
              _row("sub", unknown_pointer, src, dst) if family == "row" else
              _reduce(family, unknown_pointer, src))
    source = _source(tirx.SeqStmt([_brcb(dst, src), middle, _brcb(dst, src)]), [dst, src])
    _entries(source, 3)
    assert source.count("unknown_masked_pointer(") == 1, source
    temporary = re.search(r"auto (mask_arg_\w*) = unknown_masked_pointer\(dst\);", source)
    assert temporary, source
    normal_positions = [match.start() for match in re.finditer(re.escape(NORMAL_MASK), source)]
    assert normal_positions[0] < temporary.start() < normal_positions[1], source
    marker = {"brcb": "asc_brcb", "row": "asc_sub", "sum": "asc_repeat_reduce_sum",
              "max": "asc_repeat_reduce_max_only_value"}[family]
    assert temporary.group(1) in source[source.index(marker + "(", normal_positions[1]):], source


@pytest.mark.parametrize("lanes", [8, 16, 24, 32, 40, 48, 56])
def test_partial_reduce_mask_ends_full_run_and_full_call_restores_it(lanes):
    dst, src = _pointer("dst"), _pointer("src")
    source = _source(tirx.SeqStmt([_brcb(dst, src), _reduce("sum", dst, src, lanes),
                                 _barrier(), _brcb(dst, src)]), [dst, src])
    assert source.count(NORMAL_MASK) == 3, source
    assert source.count(FULL_MASK) == 2, source
    partial = f"asc_set_vector_mask(uint64_t(0), uint64_t({(1 << lanes) - 1}));"
    assert source.index("asc_brcb(") < source.index(partial) < source.index("asc_repeat_reduce_sum("), source
    assert source.index("asc_repeat_reduce_sum(") < source.rindex(FULL_MASK) < source.rindex("asc_brcb("), source


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("field", ["stride", "lanes"])
def test_direct_codegen_rejects_row_reduce_scalar_dtype_mismatch(kind, field):
    dst, src = _pointer("dst"), _pointer("src")
    args = [dst, src, tirx.const(8, "uint8"), tirx.const(16, "uint16"),
            tirx.const(64, "uint8")]
    index = 3 if field == "stride" else 4
    # The values remain legal. Only the ABI type is wrong, and this public
    # codegen entry deliberately bypasses LowerRowReduceRegions validation.
    args[index] = tirx.const(16 if field == "stride" else 64, "int32")
    diagnostic = "uint16 source row stride" if field == "stride" else "uint8 valid lanes"
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _source(_native("repeat_reduce_" + kind + "_rows_f32", *args), [dst, src])


@pytest.mark.parametrize("boundary", ["attr", "sblock", "bind"])
def test_scope_and_bind_boundaries_start_a_new_run(boundary):
    dst, src = _pointer("dst"), _pointer("src")
    pair = tirx.SeqStmt([_brcb(dst, src), _barrier(), _brcb(dst, src)])
    if boundary == "attr":
        middle = tirx.AttrStmt(dst, "mask_test_scope", 1, pair)
    elif boundary == "sblock":
        # Device codegen consumes the already-lowered SBlock statement.
        middle = tirx.SBlock([], [], [], "scope", pair)
    else:
        middle = tirx.Bind(tirx.Var("mask_scalar", "int32"), tirx.const(1, "int32"))
    source = _source(tirx.SeqStmt([_brcb(dst, src), middle, _brcb(dst, src)]), [dst, src])
    _entries(source, 2 if boundary == "bind" else 3)


def test_known_address_expression_keeps_the_run():
    dst = tirx.decl_buffer((1024,), "float32", name="dst", scope="shared.dyn")
    src = tirx.decl_buffer((1024,), "float32", name="src", scope="shared.dyn")
    destination = tirx.address_of(dst[8])
    source_pointer = tirx.address_of(src[16])
    source = _source(tirx.SeqStmt([_brcb(destination, source_pointer), _brcb(destination, source_pointer)]), [dst.data, src.data])
    _entries(source, 1)
    assert source.count("asc_brcb(") == 2, source
    assert "dst[8]" in source and "src[16]" in source, source


def test_builtin_integer_address_arithmetic_keeps_the_run_without_temporaries():
    dst = tirx.decl_buffer((1024,), "float32", name="dst", scope="shared.dyn")
    src = tirx.decl_buffer((1024,), "float32", name="src", scope="shared.dyn")
    epoch = tirx.Var("epoch", "int32")
    index = tirx.bitwise_and(
        tirx.bitwise_or(tirx.shift_left(epoch, 1), tirx.shift_right(epoch, 1)),
        tirx.bitwise_xor(tirx.bitwise_not(epoch), 7))
    destination = tirx.address_of(dst[index * 8])
    source_pointer = tirx.address_of(src[index * 8])
    source = _source(tirx.SeqStmt([_brcb(destination, source_pointer),
                                 _barrier(), _brcb(destination, source_pointer)]),
                     [dst.data, src.data, epoch])
    _entries(source, 1)
    assert "auto mask_arg_" not in source, source
    assert source.count("asc_brcb(") == 2, source


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_count_mode_path_remains_unmodified_on_both_architectures(arch):
    dst, src = _pointer("dst"), _pointer("src")
    source = _source(tirx.SeqStmt([_native("exp", dst, src, 64), _native("exp", dst, src, 64)]), [dst, src], arch)
    _entries(source, 0)
    assert source.count("asc_exp(") == 2, source


def test_each_function_starts_a_new_mask_lifetime():
    dst, src = _pointer("dst"), _pointer("src")
    body = tirx.SeqStmt([_brcb(dst, src), _brcb(dst, src)])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
    functions = {name: _function(name, body, [dst, src]) for name in ["first", "second"]}
    build = tvm.ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    source = build(tvm.IRModule(functions), target).inspect_source()
    _entries(source, 2)
    assert source.count("asc_brcb(") == 4, source
