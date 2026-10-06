"""Cloned hardware SID bindings must dominate every emitted C++ scope."""

import re

import pytest
from tilelang import tvm
from tvm import tirx

from testing.ascend.language.test_tilelang_ascend_kernel_lifecycle import _source


def _mark(name, sid):
    return tirx.Evaluate(tirx.call_extern("int32", name, sid))


def _bind(sid, body, extent=2):
    count = tirx.const(extent, sid.dtype)
    domain = tvm.ir.Range.from_min_extent(tirx.const(0, sid.dtype), count)
    iv = tirx.IterVar(domain, sid, tirx.IterVar.ThreadIndex, "cthread")
    return tirx.AttrStmt(iv, "thread_extent", count, body)


def _region(name, body):
    return tirx.SBlock([], [], [], name, body)


def _function(name, body, *, arch="dav-2201", params=()):
    root = _region("root", body)
    return tirx.PrimFunc(list(params), tirx.SBlockRealize([], True, root)).with_attr(
        {"global_symbol": name, "calling_conv": 2, "target": tvm.target.Target({"kind": "ascend", "arch": arch})}
    )


def _mixed(body):
    cube = _region("CUBE", tirx.Evaluate(tirx.call_extern("int32", "mark_cube")))
    return tirx.SeqStmt([cube, body])


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_nested_cloned_sid_is_materialized_once(arch):
    sid = tirx.Var("aiv_row_sid", "int32")
    body = _region("VECTOR", _bind(sid, _region("VECTOR", _bind(sid, _mark("mark_nested", sid)))))
    source = _source({"entry": _function("entry", _mixed(body), arch=arch)}, arch)
    assert source.count("= asc_get_sub_block_id();") == 1, source
    assert source.count("const int32_t aiv_row_sid = ") == 1, source
    assert source.index("const int32_t aiv_row_sid") < source.index("if ASC_IS_AIC"), source
    assert "mark_nested(aiv_row_sid)" in source, source


def test_sibling_if_and_loop_scopes_share_a_dominating_sid_declaration():
    sid = tirx.Var("aiv_row_sid", "int32")
    enabled = tirx.Var("enabled", "int32")
    row = tirx.Var("row", "int32")
    first = tirx.IfThenElse(enabled > 0, _region("VECTOR", _bind(sid, _mark("mark_first", sid))), None)
    second = tirx.For(row, 0, 2, tirx.ForKind.SERIAL, _region("VECTOR", _bind(sid, _mark("mark_second", sid))))
    source = _source({"entry": _function("entry", _mixed(tirx.SeqStmt([first, second])), params=[enabled])}, "dav-2201")
    assert source.count("= asc_get_sub_block_id();") == 1, source
    assert source.count("const int32_t aiv_row_sid = ") == 1, source
    declaration = source.index("const int32_t aiv_row_sid")
    assert declaration < source.index("if ASC_IS_AIC") < source.index("mark_first"), source
    assert declaration < source.index("for (") < source.index("mark_second"), source
    assert "mark_first(aiv_row_sid)" in source and "mark_second(aiv_row_sid)" in source, source


def test_distinct_ir_aliases_preserve_types_and_share_one_hardware_read():
    sid32 = tirx.Var("sid", "int32")
    sid64 = tirx.Var("sid", "int64")
    body = _region("VECTOR", tirx.SeqStmt([_bind(sid32, _mark("mark_i32", sid32)), _bind(sid64, _mark("mark_i64", sid64))]))
    source = _source({"entry": _function("entry", body)}, "dav-2201")
    assert source.count("= asc_get_sub_block_id();") == 1, source
    aliases = re.findall(r"const (int(?:32|64)_t) (sid(?:_\d+)?) = (cthread_value(?:_\d+)?);", source)
    assert {dtype for dtype, _, _ in aliases} == {"int32_t", "int64_t"}, source
    assert len({name for _, name, _ in aliases}) == 2, source
    assert len({value for _, _, value in aliases}) == 1, source
    for dtype, name, _ in aliases:
        marker = "mark_i32" if dtype == "int32_t" else "mark_i64"
        assert f"{marker}({name})" in source, source


def test_multiple_entries_materialize_their_own_same_named_sid():
    sid = tirx.Var("aiv_row_sid", "int32")
    functions = {name: _function(name, _region("VECTOR", _bind(sid, _mark("mark_" + name, sid)))) for name in ["first", "second"]}
    source = _source(functions, "dav-2201")
    assert source.count("= asc_get_sub_block_id();") == 2, source
    assert source.count("const int32_t aiv_row_sid = ") == 2, source
    assert "mark_first(aiv_row_sid)" in source and "mark_second(aiv_row_sid)" in source, source


def test_repeated_sid_with_conflicting_extent_is_diagnosed():
    sid = tirx.Var("aiv_row_sid", "int32")
    body = _region("VECTOR", tirx.SeqStmt([_bind(sid, _mark("mark_first", sid), 1), _bind(sid, _mark("mark_second", sid), 2)]))
    with pytest.raises(tvm.error.InternalError, match="cthread binding.*inconsistent extents"):
        _source({"entry": _function("entry", body)}, "dav-2201")


@pytest.mark.parametrize("vf_region", ["SIMT_VF", "SIMD_VF"])
def test_vf_local_sid_bindings_are_materialized_only_in_the_helper(vf_region):
    sid = tirx.Var("aiv_row_sid", "int32")
    body = _bind(sid, _bind(sid, _mark("mark_vf", sid)))
    source = _source({"entry": _function("entry", _region(vf_region, body), arch="dav-3510")}, "dav-3510")
    (helper,) = re.findall(r"__sim[td]_vf__[^\{]+\{(.*?)\n\}", source, flags=re.DOTALL)
    assert helper.count("= asc_get_sub_block_id();") == 1, source
    assert helper.count("const int32_t aiv_row_sid = ") == 1, source
    assert "mark_vf(aiv_row_sid)" in helper, source
    assert source.count("= asc_get_sub_block_id();") == 1, source


def test_vf_local_binding_does_not_reuse_or_replace_the_kernel_declaration():
    sid = tirx.Var("aiv_row_sid", "int32")
    helper = _region("SIMD_VF", _bind(sid, _bind(sid, _mark("mark_local", sid))))
    body = _bind(sid, tirx.SeqStmt([helper, _mark("mark_kernel", sid)]))
    source = _source({"entry": _function("entry", body, arch="dav-3510")}, "dav-3510")
    (helper_source,) = re.findall(r"__simd_vf__[^\{]+\{(.*?)\n\}", source, flags=re.DOTALL)
    assert helper_source.count("= asc_get_sub_block_id();") == 1, source
    assert helper_source.count("const int32_t aiv_row_sid = ") == 1, source
    assert source.count("= asc_get_sub_block_id();") == 2, source
    assert source.count("const int32_t aiv_row_sid = ") == 2, source
    assert "mark_local(aiv_row_sid)" in helper_source and "mark_kernel(aiv_row_sid)" in source, source


@pytest.mark.parametrize("vf_region", ["SIMT_VF", "SIMD_VF"])
def test_vf_free_sid_remains_a_capture_of_the_kernel_binding(vf_region):
    sid = tirx.Var("aiv_row_sid", "int32")
    body = _bind(sid, _region(vf_region, _mark("mark_capture", sid)))
    source = _source({"entry": _function("entry", body, arch="dav-3510")}, "dav-3510")
    (helper,) = re.findall(r"__sim[td]_vf__[^\{]+\{(.*?)\n\}", source, flags=re.DOTALL)
    assert "mark_capture(aiv_row_sid)" in helper, source
    assert "asc_get_sub_block_id()" not in helper, source
    assert source.count("= asc_get_sub_block_id();") == 1, source
    assert re.search(r"__sim[td]_vf__[^\{]+\(int32_t aiv_row_sid\)", source), source
