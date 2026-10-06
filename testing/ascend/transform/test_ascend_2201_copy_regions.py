"""Native whole-fractal copies expose padding dependencies before scheduling."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_l0a_2201_layout, make_ascend_major_k_layout, make_ascend_major_mn_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, region, seq
from testing.ascend.auto_schedule._scheduled_ir import unit


TARGET = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})


def _target(mod, arch="dav-2201"):
    return tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))})


def _layouts(source, destination, transpose=False):
    destination_layout = (
        make_ascend_l0a_2201_layout(destination, transpose)
        if destination.scope() == "shared.l0a"
        else (make_ascend_major_mn_layout if transpose else make_ascend_major_k_layout)(destination)
    )
    return {source: make_ascend_major_k_layout(source), destination: destination_layout}


def _normalize(mod):
    return transform.AscendNormalizeCubeCopyFootprints()(mod)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("role,transpose", [("a", False), ("a", True), ("b", False), ("b", True)])
def test_partial_logical_tile_has_exact_native_read_and_write_regions(dtype, role, transpose):
    source = tirx.decl_buffer((32, 32), dtype, name="source", scope="shared.l1")
    destination = tirx.decl_buffer((32, 32), dtype, name="destination", scope="shared.l0" + role)
    before = _target(kernel(copy(source, destination, src_shape=[17, 17], dst_shape=[17, 17]), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination, transpose)}))
    after = _normalize(before)
    (operation,) = calls(after, "tl.tileop.ascend_copy")
    assert [int(value) for value in operation.args[0].args[2:]] == [32, 32]
    assert [int(value) for value in operation.args[1].args[2:]] == [32, 32]
    tvm.ir.assert_structural_equal(_normalize(after), after)


@pytest.mark.parametrize("role", ["a", "b"])
def test_explicit_transpose_keeps_distinct_source_destination_physical_axes(role):
    source = tirx.decl_buffer((32, 16), "bfloat16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((16, 32), "bfloat16", name="destination", scope="shared.l0" + role)
    before = _target(kernel(copy(source, destination, src_shape=[17, 16], dst_shape=[16, 17], annotations={"transpose": tirx.IntImm("int32", 1)}), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination)}))
    (operation,) = calls(_normalize(before), "tl.tileop.ascend_copy")
    assert [int(value) for value in operation.args[0].args[2:]] == [32, 16]
    assert [int(value) for value in operation.args[1].args[2:]] == [16, 32]
    assert int(operation.annotations["transpose"]) == 1


def _padding(buffer):
    return tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(buffer, 2, indices=[17, 0], shape=[15, 16]), tirx.const(0, buffer.dtype)]))


def test_padding_producer_raw_and_later_reuse_war_use_native_regions():
    source = tirx.decl_buffer((32, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((32, 16), "float16", name="destination", scope="shared.l0a")
    before = _target(kernel(seq(
        unit(_padding(source), core=2),
        unit(copy(source, destination, src_shape=[17, 16], dst_shape=[17, 16]), core=2),
        unit(_padding(source), core=2),
    ), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination)}))
    # The logical intervals are disjoint. Native LoadData reads those padding
    # rows too; that fact must exist before dependency/sync analysis.
    after = transform.InsertSync()(_normalize(before))
    for pipe in ("MTE2_MTE1", "MTE1_MTE2"):
        sets = [call for call in calls(after, "tl.ascend_set_flag") if call.args[0].value == pipe]
        waits = [call for call in calls(after, "tl.ascend_wait_flag") if call.args[0].value == pipe]
        assert sets and waits
        assert {int(call.args[1]) for call in sets} == {int(call.args[1]) for call in waits}


@pytest.mark.parametrize("failure", ["missing-fractals", "outside-allocation", "unaligned-origin", "unknown-origin"])
def test_early_regions_never_authorize_missing_or_unknown_source_storage(failure):
    source_capacity = 16 if failure in ("missing-fractals", "outside-allocation") else 32
    source = tirx.decl_buffer((source_capacity, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((32, 16), "float16", name="destination", scope="shared.l0a")
    row = tirx.Var("row", "int32")
    origin = row * 16 if failure == "unknown-origin" else (1 if failure == "unaligned-origin" else 0)
    extent = 16 if failure == "missing-fractals" else 32
    variables = [row] if failure == "unknown-origin" else []
    before = _target(kernel(copy(source, destination, src_indices=[origin, 0], src_shape=[extent, 16], dst_shape=[32, 16]), buffers=[source, destination], params=variables, annotations={"layout_map": _layouts(source, destination)}))
    diagnostic = "matching physical" if failure == "missing-fractals" else ("source.*origin" if failure == "unaligned-origin" else "physical L1 allocation")
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _normalize(before)


def test_leading_versions_and_empty_copy_semantics_survive_normalization():
    source = tirx.decl_buffer((2, 32, 16), "bfloat16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((2, 32, 16), "bfloat16", name="destination", scope="shared.l0a")
    epoch = tirx.Var("epoch", "int32")
    body = copy(source, destination, src_indices=[epoch % 2, 0, 0], dst_indices=[(epoch + 1) % 2, 0, 0], src_shape=[1, 17, 16], dst_shape=[1, 17, 16])
    before = _target(kernel(tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, body), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination)}))
    (operation,) = calls(_normalize(before), "tl.tileop.ascend_copy")
    analyzer = tvm.arith.Analyzer()
    assert analyzer.can_prove_equal(operation.args[0].args[0].indices[0], epoch % 2)
    assert analyzer.can_prove_equal(operation.args[1].args[0].indices[0], (epoch + 1) % 2)
    assert [int(value) for value in operation.args[0].args[2:]] == [1, 32, 16]
    empty = _target(kernel(copy(source, destination, src_shape=[0, 17, 16], dst_shape=[1, 17, 16]), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination)}))
    tvm.ir.assert_structural_equal(_normalize(empty), empty)


def test_dynamic_empty_leading_owner_does_not_become_an_unconditional_read():
    source = tirx.decl_buffer((1, 32, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((1, 32, 16), "float16", name="destination", scope="shared.l0a")
    active = tirx.Var("active", "bool")
    extent = tirx.Select(active, 1, 0)
    before = _target(kernel(copy(source, destination, src_shape=[extent, 17, 16], dst_shape=[extent, 17, 16]), buffers=[source, destination], params=[active], annotations={"layout_map": _layouts(source, destination)}))
    after = _normalize(before)
    (operation,) = calls(after, "tl.tileop.ascend_copy")
    for operand in operation.args[:2]:
        analyzer = tvm.arith.Analyzer()
        assert analyzer.can_prove_equal(operand.args[2], extent)
        # The exact footprint is empty when the fixed plane is inactive.
        for value in operand.args[2:]:
            assert analyzer.can_prove_equal(tirx.stmt_functor.substitute(value, {active: tirx.const(False)}), 0)
    tvm.ir.assert_structural_equal(_normalize(after), after)


def test_3510_copy_regions_are_unchanged():
    source = tirx.decl_buffer((32, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((32, 16), "float16", name="destination", scope="shared.l0b")
    before = _target(kernel(copy(source, destination, src_shape=[17, 16], dst_shape=[17, 16]), buffers=[source, destination], annotations={"layout_map": _layouts(source, destination)}), "dav-3510")
    tvm.ir.assert_structural_equal(_normalize(before), before)


@pytest.mark.parametrize("conflict", [False, True])
def test_nested_layout_maps_preserve_parent_facts_and_reject_conflicts(conflict):
    source = tirx.decl_buffer((32, 16), "float16", name="source", scope="shared.l1")
    destination = tirx.decl_buffer((32, 16), "float16", name="destination", scope="shared.l0a")
    layouts = _layouts(source, destination)
    child_layouts = {destination: layouts[destination]}
    if conflict:
        child_layouts[source] = make_ascend_major_mn_layout(source)
    child = tirx.SBlock([], [], [], "child", copy(source, destination, src_shape=[17, 16], dst_shape=[17, 16]), annotations={"layout_map": child_layouts})
    before = _target(kernel(child, buffers=[source, destination], annotations={"layout_map": {source: layouts[source]}}))
    if conflict:
        with pytest.raises(tvm.error.InternalError, match="Conflicting layouts.*source"):
            _normalize(before)
    else:
        (operation,) = calls(_normalize(before), "tl.tileop.ascend_copy")
        assert [int(value) for value in operation.args[0].args[2:]] == [32, 16]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_pipeline_retains_padding_writer_and_effective_gemm_domain(dtype):
    @T.prim_func
    def main(A: T.Tensor((17, 16), dtype), B: T.Tensor((16, 16), dtype), C: T.Tensor((17, 16), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((32, 16), dtype)
            b = T.alloc_l1((16, 16), dtype)
            c = T.alloc_l0c((17, 16), "float32")
            T.copy(A[0:32, :], a)
            T.copy(B, b)
            T.gemm(a[0:17, :], b, c, transpose_B=True, clear_accum=True)
            T.copy(c, C)

    snapshots = {}
    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendNormalizeCubeCopyFootprints":
                snapshots[info.name] = mod
    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(main, target=TARGET, target_host="c", enable_host_codegen=False, enable_device_compile=False)
    normalized = snapshots["tl.AscendNormalizeCubeCopyFootprints"]
    loads = [call for call in calls(normalized, "tl.tileop.ascend_copy") if call.args[1].args[0].buffer.scope() == "shared.l0a"]
    assert len(loads) == 1
    assert int(loads[0].args[0].args[2]) == 32
    assert int(loads[0].args[1].args[2]) == 32
    (gemm,) = calls(normalized, "tl.tileop.gemm")
    assert int(gemm.args[5]) == 17
    assert "asc_fill_l1(" in artifact.kernel_source
    assert "asc_sync_notify(PIPE_MTE2, PIPE_MTE1" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
