"""Dimensionless native Cube service weights and ordinary automatic buffering."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import make_ascend_major_k_layout, make_ascend_l0a_2201_layout
from tvm import tirx

from testing.ascend._ir import copy, kernel, nodes, region
from testing.ascend.auto_schedule._task_utils import _collect_task_metadata


TARGET = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})
MODEL = "dav-2201.service_weights.estimated.v1"


def _estimate(mod):
    if isinstance(mod, tirx.PrimFunc):
        mod = tvm.IRModule({"main": mod.with_attr("global_symbol", "main")})
    mod["main"] = mod["main"].with_attr("target", TARGET)
    with TARGET:
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.transform.Simplify()(mod)
        mod = transform.MaterializeScheduleUnits()(mod)
        after = transform.EstimateLatency()(mod)
    assert str(after["main"].attrs["tl.ascend.cost_model"]) == MODEL
    metadata = _collect_task_metadata(after)
    return list(zip(metadata["latency"], metadata["ii"]))


def _gemm(m, n, k, dtype="float16", *, dynamic_k=False):
    @T.prim_func
    def main(tail: T.int32):
        with T.Kernel(1):
            a = T.alloc_l0a((m, k), dtype)
            b = T.alloc_l0b((n, k), dtype)
            c = T.alloc_l0c((m, n), "float32")
            active_k = T.max(T.min(tail, k), 0) if dynamic_k else k
            T.gemm(a[:, :active_k], b[:, :active_k], c, transpose_B=True,
                   clear_accum=True)
    return main


@pytest.mark.parametrize("m,n,k,expected", [
    (128, 128, 128, 4), (128, 128, 64, 2), (64, 64, 64, 1),
    (127, 128, 128, 4), (256, 128, 64, 4),
])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mmad_weight_scales_physical_fractal_work(m, n, k, expected, dtype):
    assert _estimate(_gemm(m, n, k, dtype))[-1] == (expected, expected)


@pytest.mark.parametrize("dtype,dynamic_k", [("float32", False), ("float16", True), ("bfloat16", True)])
def test_uncalibrated_precision_or_dynamic_work_retains_ordinal_one(dtype, dynamic_k):
    assert _estimate(_gemm(128, 128, 128, dtype, dynamic_k=dynamic_k))[-1] == (1, 1)


def _load(which, *, n=128, k=128, cache_k=256, source_n=None,
          dtype="float16", leading=False, transpose=False, tagged=True):
    source_n = n if source_n is None else source_n
    prefix = (2,) if leading else ()
    index = [1] if leading else []
    extent = [1] if leading else []
    src = tirx.decl_buffer((*prefix, source_n, cache_k), dtype, name="cache", scope="shared.l1")
    dst = tirx.decl_buffer((*prefix, n, k), dtype, name="operand", scope="shared.l0a" if which == "a3d" else "shared.l0b")
    source = region(src, 1, [*index, 0, cache_k - k], [*extent, n, k])
    destination = region(dst, 2, [*index, 0, 0], [*extent, n, k])
    if which == "a3d":
        body = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.tileop.ascend_cube_load3d"),
                                    [source, destination, n, cache_k]))
    else:
        body = copy(src, dst, src_indices=[*index, 0, cache_k - k],
                    src_shape=[*extent, n, k], dst_indices=[*index, 0, 0],
                    dst_shape=[*extent, n, k],
                    annotations={"transpose": tirx.const(1, "int32")} if transpose else None)
    layout_map = {src: make_ascend_major_k_layout(src),
                  dst: make_ascend_l0a_2201_layout(dst) if which == "a3d" else make_ascend_major_k_layout(dst)} if tagged else {}
    return kernel(body, buffers=[src, dst], annotations={"layout_map": layout_map})


@pytest.mark.parametrize("leading", [False, True])
@pytest.mark.parametrize("cache_k", [128, 256, 1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_load_payload_ignores_source_cache_span_and_version_slots(leading, cache_k, dtype):
    assert _estimate(_load("a3d", cache_k=cache_k, leading=leading, dtype=dtype))[-1] == (1, 1)
    assert _estimate(_load("b2d", cache_k=cache_k, leading=leading, dtype=dtype))[-1] == (2, 2)


@pytest.mark.parametrize("n,k,wa,wb", [(128, 64, 1, 1), (128, 128, 1, 2), (256, 128, 2, 4)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_native_load_weight_scales_destination_payload(n, k, wa, wb, dtype):
    assert _estimate(_load("a3d", n=n, k=k, dtype=dtype))[-1] == (wa, wa)
    assert _estimate(_load("b2d", n=n, k=k, dtype=dtype))[-1] == (wb, wb)


@pytest.mark.parametrize("options", [
    {"source_n": 256}, {"transpose": True}, {"tagged": False}, {"dtype": "float32"},
])
def test_segmented_unidentified_or_unmeasured_b_load_retains_ordinal_one(options):
    assert _estimate(_load("b2d", **options))[-1] == (1, 1)


def test_frontend_var_key_layout_native_b_load_retains_uncalibrated_ordinal():
    # The calibrated native-B classification needs post-layout Buffer keys.
    # Frontend layout annotations instead map each allocation's data Var to
    # its layout; estimation must retain the uncalibrated fallback at that
    # boundary rather than attempting to cast those keys to Buffer.
    before = _load("b2d")
    assert _estimate(before)[-1] == (2, 2)

    def use_frontend_keys(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "tilelang_root":
            layouts = node.annotations["layout_map"]
            frontend_layouts = {buffer.data: layout for buffer, layout in layouts.items()}
            assert frontend_layouts and all(isinstance(key, tirx.Var) for key in frontend_layouts)
            return tirx.SBlock(
                node.iter_vars, node.reads, node.writes, node.name_hint, node.body,
                init=node.init, alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations={**dict(node.annotations), "layout_map": frontend_layouts},
            )

    func = before["main"]
    frontend = tvm.IRModule({"main": func.with_body(
        tirx.stmt_functor.ir_transform(func.body, None, use_frontend_keys)
    )})
    assert _estimate(frontend)[-1] == (1, 1)


def test_unmeasured_a3d_precision_keeps_uncalibrated_ordinal():
    assert _estimate(_load("a3d", n=256, dtype="float32", tagged=False))[-1] == (1, 1)


def test_unmeasured_gm_to_l1_leaf_retains_ordinal_one():
    @T.prim_func
    def main(A: T.Tensor((128, 256), "float16")):
        with T.Kernel(1):
            cached = T.alloc_l1((128, 256), "float16")
            T.copy(A, cached)
    assert _estimate(main) == [(1, 1)]


@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ordinary_auto_gemm_selects_two_a_and_b_slots_with_service_weights(n, dtype):
    # This is an acceptance test for the scheduling effect, not a cost-table
    # mirror. No explicit task duration, stages, buffer counts or issue order.
    @T.prim_func
    def main(A: T.Tensor((128, 1024), dtype),
             B: T.Tensor((n, 1024), dtype),
             C: T.Tensor((128, n), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((128, 256), dtype)
            b = T.alloc_l1((n, 256), dtype)
            c = T.alloc_l0c((128, n), "float32")
            for ko in T.Pipelined(4):
                T.copy(A[:, ko * 256:(ko + 1) * 256], a)
                T.copy(B[:, ko * 256:(ko + 1) * 256], b)
                T.gemm(a, b, c, transpose_B=True, clear_accum=(ko == 0))
            T.copy(c, C)

    snapshots = []

    @tvm.instrument.pass_instrument
    class Observe:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AutoSchedule":
                snapshots.append(mod)

    with tvm.transform.PassContext(instruments=[Observe()]):
        tilelang.lower(main, target=TARGET, enable_host_codegen=False,
                       enable_device_compile=False)
    assert snapshots
    root = next(block for block in nodes(snapshots[-1], tirx.SBlock)
                if block.name_hint == "tilelang_root")
    counts = root.annotations["tl.buffer_versions_map"]
    for scope in ("shared.l0a", "shared.l0b"):
        operands = [buffer for buffer in root.alloc_buffers if buffer.scope() == scope]
        assert operands, snapshots[-1].script()
        assert all(int(counts[buffer.data]) == 2 for buffer in operands), snapshots[-1].script()



@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_new_b3d_path_keeps_uncalibrated_ordinal_weight(dtype):
    from testing.ascend.transform.test_ascend_select_cube_loads import _b_fixture, _select

    # A large legal B descriptor would get a scaled A3D value if the measured
    # A-path ratio were accidentally reused. B3D needs its own evidence.
    selected = _select(_b_fixture(dtype, load_k=512, k_start=0))
    assert _estimate(selected)[-1] == (1, 1)
