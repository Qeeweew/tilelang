"""Native entry completion and floating-point state are kernel boundaries."""

import re

import pytest
import tilelang
from tilelang import tvm
from tvm import tirx


def _region(name):
    body = tirx.Evaluate(tirx.call_extern("int32", "mark_" + name.lower()))
    # Core/VF regions are bare SBlocks in device TIR, nested inside its opaque
    # realized root. Realizing these named regions would erase their meaning.
    return tirx.SBlock([], [], [], name, body)


def _function(name, mode, arch, *, entry=True):
    regions = [_region(region) for region in ({"cube": ["CUBE"], "vector": ["VECTOR"], "mix": ["CUBE", "VECTOR"]}[mode])]
    body = regions[0] if len(regions) == 1 else tirx.SeqStmt(regions)
    attrs = {"global_symbol": name, "target": tvm.target.Target({"kind": "ascend", "arch": arch})}
    attrs.update({"calling_conv": 2} if entry else {"tirx.is_global_func": True})
    root = tirx.SBlock([], [], [], "root", body)
    return tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr(attrs)


def _source(functions, arch):
    target = tvm.target.Target({"kind": "ascend", "arch": arch})
    build = tvm.ffi.get_global_func("target.build.tilelang_ascend_without_compile")
    # The backend consumes device TIR, after the shared opaque-block boundary.
    # This removes SBlockRealize while retaining named core/VF SBlock regions.
    mod = tilelang.transform.LowerOpaqueBlock()(tvm.IRModule(functions))
    return build(mod, target).inspect_source()


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
@pytest.mark.parametrize("mode", ["cube", "vector", "mix"])
def test_entry_drains_after_all_core_regions(arch, mode):
    source = _source({"entry": _function("entry", mode, arch)}, arch)
    assert {"cube": "__cube__", "vector": "__vector__", "mix": "__mix__(1, 2)"}[mode] in source, source
    assert source.count("asc_sync_pipe(PIPE_ALL);") == 1, source
    assert source.rstrip().endswith("asc_sync_pipe(PIPE_ALL);\n}"), source
    for marker in {"cube": ["mark_cube()"], "vector": ["mark_vector()"], "mix": ["mark_cube()", "mark_vector()"]}[mode]:
        assert source.index(marker) < source.index("asc_sync_pipe(PIPE_ALL);"), source
    non_saturating = "asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));"
    if arch == "dav-2201":
        assert source.count(non_saturating) == 1, source
        assert source.index("asc_init();") < source.index(non_saturating) < source.index("mark_"), source
    else:
        assert non_saturating not in source, source


def test_multiple_entries_each_establish_and_drain_their_own_lifetime():
    functions = {mode: _function(mode, mode, "dav-2201") for mode in ["cube", "vector", "mix"]}
    source = _source(functions, "dav-2201")
    assert source.count("asc_init();") == 3, source
    assert source.count("asc_set_ctrl(asc_get_ctrl() | (uint64_t(1) << 48));") == 3, source
    assert source.count("asc_sync_pipe(PIPE_ALL);\n}") == 3, source


def test_call_extern_global_helper_does_not_gain_entry_drain_or_ctrl_state():
    source = _source({"helper": _function("helper", "vector", "dav-2201", entry=False)}, "dav-2201")
    assert "asc_sync_pipe(PIPE_ALL);" not in source, source
    assert "asc_set_ctrl(" not in source, source


@pytest.mark.parametrize("region", ["SIMT_VF", "SIMD_VF"])
def test_vf_helper_does_not_gain_native_entry_lifecycle(region):
    body = _region(region)
    root = tirx.SBlock([], [], [], "root", body)
    func = tirx.PrimFunc([], tirx.SBlockRealize([], True, root)).with_attr(
        {"global_symbol": "entry", "calling_conv": 2, "target": tvm.target.Target({"kind": "ascend", "arch": "dav-3510"})}
    )
    source = _source({"entry": func}, "dav-3510")
    (helper,) = re.findall(r"__sim[td]_vf__[^\{]+\{(.*?)\n\}", source, flags=re.DOTALL)
    assert "asc_init();" not in helper and "asc_sync_pipe(PIPE_ALL);" not in helper, source
    assert source.count("asc_sync_pipe(PIPE_ALL);") == 1, source
