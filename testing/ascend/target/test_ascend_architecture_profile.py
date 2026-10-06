"""Target identity fixes capabilities and per-core storage before scheduling."""

import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


def target_info(attributes):
    return tvm.ffi.get_global_func("tl.GetAscendTargetInfo")(tvm.target.Target({"kind": "ascend", **attributes}))


@pytest.mark.parametrize("attribute", ["arch", "mcpu"])
def test_2201_has_private_membase_ub_and_smaller_accumulator_storage(attribute):
    info = target_info({attribute: "dav-2201"})
    assert info["arch"] == "dav-2201"
    assert info["aiv_per_aic"] == 2
    assert info["ub_bytes"] == 192 * 1024
    assert info["l0c_bytes"] == 128 * 1024
    assert info["l1_bytes"] == 512 * 1024
    assert info["l0a_bytes"] == info["l0b_bytes"] == 64 * 1024
    assert not info["supports_simt_vf"]
    assert not info["supports_simd_vf"]
    assert not info["supports_mx_gemm"]
    assert info["simt_vf_context_bytes"] == 0


def test_default_preserves_upstream_3510_contract():
    default = target_info({})
    explicit = target_info({"arch": "dav-3510", "mcpu": "dav-3510"})
    assert dict(default) == dict(explicit)
    assert default["ub_bytes"] == 248 * 1024
    assert default["l0c_bytes"] == 256 * 1024
    assert default["supports_simt_vf"]
    assert default["supports_simd_vf"]
    assert default["supports_mx_gemm"]
    assert default["simt_vf_context_bytes"] == 32 * 1024


@pytest.mark.parametrize(
    "attributes, message",
    [
        ({"arch": "dav-2201", "mcpu": "dav-3510"}, "arch and mcpu must name the same architecture"),
        ({"arch": "dav-unknown"}, "Unsupported Ascend architecture"),
        ({"mcpu": ""}, "Unsupported Ascend architecture"),
    ],
)
def test_ambiguous_or_unknown_architecture_is_rejected(attributes, message):
    with pytest.raises(tvm.error.InternalError, match=message):
        target_info(attributes)


def _ub_ring_program(arch, ub_bytes):
    elements = ub_bytes // 4
    a = tirx.decl_buffer((4, elements), "float32", name="A")
    c = tirx.decl_buffer((4, elements), "float32", name="C")
    ub = tirx.decl_buffer((elements,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, ub, src_indices=[i, 0], src_shape=[1, elements]), stage=-1, cost=(1, 1)),
        unit(copy(ub, c, dst_indices=[i, 0], dst_shape=[1, elements]), stage=-1, cost=(1, 1)),
    )
    loop = unit(
        tirx.For(
            i,
            0,
            4,
            tirx.ForKind.SERIAL,
            body,
            annotations={"num_stages": 2, "multi_buffer_eligible": [ub.data]},
        ),
        core=None,
        stage=-1,
    )
    mod = kernel(loop, buffers=[ub], params=[a, c])
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def test_auto_schedule_uses_per_aiv_2201_capacity():
    # A single 200 KiB UB version fits on 3510 but cannot fit in either 2201
    # AIV's private UB. Pairing two AIVs must not double the storage allowance.
    program = _ub_ring_program("dav-2201", 200 * 1024)
    with pytest.raises(tvm.error.InternalError, match="insufficient for minimum buffer usage"):
        transform.AutoSchedule()(program)
    transform.AutoSchedule()(_ub_ring_program("dav-3510", 200 * 1024))


def test_2201_ring_can_use_its_full_private_ub():
    transform.AutoSchedule()(_ub_ring_program("dav-2201", 192 * 1024))
