"""2201 byte-granular DMA tails use native gap units and legal UB pitches."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T

from testing.ascend.language.test_tilelang_ascend_2201_cube_codegen import _call_arguments


TARGET = tvm.target.Target({"kind": "ascend", "arch": "dav-2201"})


def _tail_roundtrip(dtype, rank):
    rows = (4,) if rank == 2 else (2, 3)

    @T.prim_func
    def main(src: T.Tensor((*rows, 45), dtype), dst: T.Tensor((*rows, 21), dtype)):
        with T.Kernel(1):
            ub = T.alloc_shared((*rows, 32), dtype)
            if rank == 2:
                T.copy(src[:, :17], ub[:, :17], pad_value=0.0)
                T.copy(ub[:, :17], dst[:, :17])
            else:
                T.copy(src[:, :, :17], ub[:, :, :17], pad_value=0.0)
                T.copy(ub[:, :, :17], dst[:, :, :17])

    return main


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("rank", [2, 3])
def test_multirow_tail_uses_element_padding_and_independent_gap_units(dtype, rank):
    source = tilelang.lower(_tail_roundtrip(dtype, rank), target=TARGET).kernel_source
    load = _call_arguments(source, "asc_copy_gm2ub_align")
    store = _call_arguments(source, "asc_copy_ub2gm_align")
    size = 2 if dtype == "float16" else 4
    count = 4 if rank == 2 else 6
    right_pad = 15 if size == 2 else 7
    ub_gap = 0 if size == 2 else 1
    assert len(load) == len(store) == 8, source
    assert load[2:] == [str(count), str(17 * size), "0", str(right_pad), str(28 * size), str(ub_gap)], source
    assert store[2:] == [str(count), str(17 * size), "0", "0", str(ub_gap), str(4 * size)], source
    assert f"uint{size * 8}_t" in load[0], source
    assert source.index("asc_set_copy_pad_val(") < source.index("asc_copy_gm2ub_align("), source
    assert "asc_load_l2_cache_mode" not in source and "asc_store_l2_cache_mode" not in source


def test_single_tail_burst_uses_zero_gaps():
    @T.prim_func
    def main(src: T.Tensor((17,), "float32"), dst: T.Tensor((17,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((32,), "float32")
            T.copy(src, ub[:17])
            T.copy(ub[:17], dst)

    source = tilelang.lower(main, target=TARGET).kernel_source
    for name in ("asc_copy_gm2ub_align", "asc_copy_ub2gm_align"):
        args = _call_arguments(source, name)
        assert args[2:] == ["1", "68", "0", "0", "0", "0"], source


def test_multirow_copy_rejects_unaligned_ub_pitch():
    @T.prim_func
    def main(src: T.Tensor((4, 45), "float32"), dst: T.Tensor((4, 21), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((4, 20), "float32")
            T.copy(src[:, :17], ub[:, :17])
            T.copy(ub[:, :17], dst[:, :17])

    with pytest.raises(ValueError, match="32-byte-aligned UB row pitch"):
        tilelang.lower(main, target=TARGET)


def test_copy_rejects_unaligned_ub_origin():
    @T.prim_func
    def main(src: T.Tensor((16,), "float32"), dst: T.Tensor((16,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((32,), "float32")
            T.copy(src, ub[1:17])
            T.copy(ub[1:17], dst)

    with pytest.raises(ValueError, match="32-byte-aligned UB origin"):
        tilelang.lower(main, target=TARGET)
