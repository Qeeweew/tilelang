"""Static C256 reductions keep exact private effects and bounded scratch."""

import numpy as np
import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tvm import tirx

from testing.ascend._ir import calls, nodes, region, seq
from testing.ascend.vector.test_ascend_multi_vl_reduce import (
    TARGET, _equal, _lower, _program, _root_program, _strided,
)
from testing.ascend.vector.test_reduce_dynamic_multi_vl import _execute_rows
from testing.ascend.vector.test_reduce_manual_ring_headroom import PRIVATE, _program as _ring_program


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("retain_axis", [False, True])
def test_nonzero_rows_have_exact_pointer_spans_and_two_separate_tasks(kind, retain_axis):
    before, source, destination = _program(
        kind=kind, rows=32, columns=256, source_row=2, output_row=8,
        retain_axis=retain_axis, index_dtype="int64")
    after = _lower(before, vector=True)
    (first,) = calls(after, "tl.ascend_membase.datablock_reduce_" + kind)
    (last,) = _strided(after, kind)
    partials = first.args[0].args[0].buffer
    _equal(partials.shape, [32, 32])
    assert not partials.data.same_as(source.data)
    assert not partials.data.same_as(destination.data)
    _equal(first.args[1].args[0].indices, [2, 0])
    _equal([first.args[0].args[1], first.args[1].args[1], first.args[2]], [1024, 8192, 8192])
    assert [int(arg.args[2]) for arg in first.args[:2]] == [2, 1]
    _equal(last.args[0].args[0].indices, [8, 0] if retain_axis else [8])
    _equal(last.args[0].args[2:], [32, 1] if retain_axis else [32])
    _equal(last.args[1].args[2:], [32, 32])
    assert [int(arg) for arg in last.args[2:]] == [32, 4, 32]
    assert not nodes(after, tirx.For) and not nodes(after, tirx.Bind)
    units = transform.MaterializeScheduleUnits()(after)
    assert len([node for node in nodes(units, tirx.AttrStmt)
                if node.attr_key == "tl.ascend_task"]) == 2
    with tvm.target.Target(TARGET):
        pointers = tilelang.transform.LowerAccessPtr()(transform.AscendLowerTileOp()(after))
    (first_ptr,) = calls(pointers, first.op.name)
    (last_ptr,) = _strided(pointers, kind)
    _equal([first_ptr.args[1].args[2], first_ptr.args[1].args[3]], [512, 8192])
    _equal([last_ptr.args[0].args[2], last_ptr.args[0].args[3]], [8, 32])
    _equal([last_ptr.args[1].args[2], last_ptr.args[1].args[3]], [0, 1024])


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("fits", [False, True])
def test_private_scratch_and_independent_manual_slots_obey_exact_capacity(kind, fits):
    # 8x256 source, aligned output, two separately padded manual slots.
    owner_bytes = 8192 + 32 + 64
    filler_bytes = 196608 - owner_bytes - 1024 + (0 if fits else 32)
    before, owners = _ring_program(kind=kind, rows=8, columns=256, private=True,
                                   ring_shape=(2, 1), filler_bytes=filler_bytes)
    after = _lower(before)
    datablocks = calls(after, "tl.ascend_membase.datablock_reduce_" + kind)
    assert bool(datablocks) == fits
    # Insufficient space for the larger candidate retains the established
    # four-chunk plan. It commits no rejected allocation or ownership fact.
    assert len(_strided(after, kind)) == (1 if fits else 4)
    root = after["main"].body.block
    scratch = [buffer for buffer in root.alloc_buffers
               if not any(buffer.data.same_as(owner.data) for owner in owners)]
    assert all(any(member.same_as(buffer.data) for member in root.annotations[PRIVATE])
               for buffer in scratch)
    assert sum(np.prod([int(extent) for extent in buffer.shape]) * 4 for buffer in scratch) == (
        1024 if fits else 192)


def test_pending_scratch_changes_the_second_candidate_without_aliasing_the_first():
    before, source, _ = _program(rows=8, columns=256)
    output2 = tirx.decl_buffer((8,), "float32", "second_output", scope="shared.dyn")
    original = calls(before, "tl.tileop.reduce")[0]
    args = list(original.args)
    args[1] = region(output2, 2)
    filler_bytes = 196608 - (8192 + 2 * 32) - 1216
    filler = tirx.decl_buffer((filler_bytes // 4,), "float32", "filler", scope="shared.dyn")
    after = _lower(_root_program(before, extra_buffers=[output2, filler], body=seq(
        tirx.Evaluate(original), tirx.Evaluate(tirx.Call("handle", original.op, args)))))
    assert len(calls(after, "tl.ascend_membase.datablock_reduce_sum")) == 1
    assert len(_strided(after)) == 5
    assert len({buffer.data for buffer in after["main"].body.block.alloc_buffers}) == len(
        after["main"].body.block.alloc_buffers)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_distinct_rows_blocks_and_cancellation_match_a_fp64_oracle(kind):
    before, source, destination = _program(kind=kind, rows=8, columns=256,
                                          source_row=2, output_row=8)
    after = _lower(before)
    root = after["main"].body.block
    memory = {buffer.data: np.full(tuple(int(extent) for extent in buffer.shape), np.nan, np.float32)
              for buffer in root.alloc_buffers}
    rng = np.random.default_rng(83)
    values = rng.normal(size=tuple(int(extent) for extent in source.shape)).astype(np.float32)
    # Opposing large values and small residuals exercise the changed sum
    # tree. Use an absolute forward-error bound, not bit equality to the old
    # four-chunk reduction. Different rows and blocks expose pitch/mask bugs.
    values[2:10, 0] = 2**16
    values[2:10, 8] = -(2**16)
    values[2:10, 248:256] += np.arange(8, dtype=np.float32)[:, None]
    memory[source.data][...] = values
    memory[destination.data][...] = 519
    issued = []
    _execute_rows(root.body, {}, memory, issued)
    oracle = values[2:10].astype(np.float64)
    expected = oracle.max(axis=1) if kind == "max" else oracle.sum(axis=1)
    np.testing.assert_allclose(memory[destination.data][8:16], expected,
                               rtol=2e-6, atol=0 if kind == "max" else 0.04)
    np.testing.assert_array_equal(memory[destination.data][:8], np.full(8, 519, np.float32))
    np.testing.assert_array_equal(memory[source.data], values)
    assert len(issued) == 2


@pytest.mark.parametrize("options", [
    {"clear": False}, {"pitch": 264}, {"output_row": 1},
    {"source_column": 64, "pitch": 320}, {"rows": 0}, {"rows": 256},
    {"row_extent": tirx.Var("dynamic_rows", "int32")},
])
def test_incompatible_or_dynamic_domains_keep_the_established_plan(options):
    before, _, _ = _program(columns=256, **options)
    after = _lower(before)
    assert not calls(after, "tl.ascend_membase.datablock_reduce_sum")


def test_two_native_tasks_cannot_be_hidden_in_an_atomic_task():
    before, _, _ = _program(columns=256, task=True)
    with pytest.raises(ValueError, match="multiple dependency-bearing tasks"):
        _lower(before)


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_codegen_restores_the_partial_mask_and_keeps_real_vector_completion(kind):
    @T.prim_func
    def main(A: T.Tensor((32, 256), "float32"), O: T.Tensor((32,), "float32")):
        with T.Kernel(1):
            source = T.alloc_shared((32, 256), "float32")
            output = T.alloc_shared((32,), "float32")
            T.copy(A, source)
            if kind == "sum":
                T.reduce_sum(source, output, dim=1, clear=True)
            else:
                T.reduce_max(source, output, dim=1, clear=True)
            T.copy(output, O)

    artifact = tilelang.lower(main, target=TARGET, target_host="c",
                              enable_host_codegen=True, enable_device_compile=False)
    source = artifact.kernel_source
    first = source.index("asc_datablock_reduce_" + kind + "(")
    last = source.index("asc_repeat_reduce_" + ("sum" if kind == "sum" else "max_only_value") + "(")
    between = source[first:last]
    assert "asc_sync_pipe(PIPE_V)" in between
    assert "asc_set_mask_norm()" in between
    assert "asc_set_vector_mask(uint64_t(0), uint64_t(4294967295))" in between
    assert "uint16_t(1), uint16_t(1), (uint16_t)4" in source[last:]
