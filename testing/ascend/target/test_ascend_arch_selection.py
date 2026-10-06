"""An architecture default is selected once, before compiler stages run."""

import pytest
from tilelang import tvm

from tilelang.backend.target import determine_target
from tilelang.contrib import bisheng

Target = tvm.target.Target


@pytest.mark.parametrize("input_kind", ["string", "dict", "target"])
def test_environment_default_is_frozen(input_kind, monkeypatch):
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-2201")
    value = {"string": "ascend", "dict": {"kind": "ascend"}, "target": Target("ascend")}[input_kind]
    target = determine_target(value, return_object=True)
    assert str(target.attrs["arch"]) == "dav-2201"
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-3510")
    assert bisheng.get_target_npu_arch(target) == "dav-2201"


@pytest.mark.parametrize("attribute", ["arch", "mcpu"])
def test_explicit_architecture_wins(attribute, monkeypatch):
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-3510")
    target = determine_target({"kind": "ascend", attribute: "dav-2201"}, return_object=True)
    assert bisheng.get_target_npu_arch(target) == "dav-2201"


def test_default_preserves_upstream(monkeypatch):
    monkeypatch.delenv("ASCEND_NPU_ARCH", raising=False)
    assert bisheng.get_target_npu_arch(determine_target("ascend", return_object=True)) == "dav-3510"
