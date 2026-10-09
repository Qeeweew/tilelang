"""Reduction candidates own their actual scratch before vector expansion."""

import math

import numpy as np
import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, nodes, seq
from testing.ascend.transform.test_ascend_merge_allocations import _merge_on_target
from testing.ascend.vector.test_ascend_multi_vl_reduce import _lower, _program, _root_program
from testing.ascend.vector.test_reduce_dynamic_multi_vl import _execute_rows
from testing.ascend.vector.test_reduce_manual_ring_headroom import (
    PRIVATE,
    _program as _ring_program,
    _strided,
)


def _bytes(buffers):
    return sum((math.prod(int(extent) for extent in buffer.shape) * 4 + 31) // 32 * 32
               for buffer in buffers)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("chunks,scratch_bytes", [(2, 64), (3, 128), (5, 256)])
def test_tree_allocations_are_complete_before_vector_lowering(kind, chunks, scratch_bytes):
    before, owners = _ring_program(kind=kind, rows=8, columns=chunks * 64, private=True)
    reduced = transform.AscendLowerVectorReduce()(before)
    root = reduced["main"].body.block
    scratch = [buffer for buffer in root.alloc_buffers
               if not any(buffer.data.same_as(owner.data) for owner in owners)]
    assert len(_strided(reduced, kind)) == chunks
    assert _bytes(scratch) == scratch_bytes
    assert all(any(member.same_as(buffer.data) for member in root.annotations[PRIVATE])
               for buffer in scratch)
    # Buffer owners are complete at the reduction boundary. Lowering each
    # ordinary binary assignment must not invent another algorithm temporary.
    lowered = transform.AscendLowerVectorTiles()(reduced)
    assert {buffer.data for buffer in lowered["main"].body.block.alloc_buffers} == {
        buffer.data for buffer in root.alloc_buffers
    }
    assert len(calls(lowered, "tl.ascend_membase." + ("add" if kind == "sum" else "max"))) == chunks - 1


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("chunks,scratch_bytes", [(2, 64), (3, 128), (5, 256)])
@pytest.mark.parametrize("fits", [False, True])
def test_actual_candidate_footprint_controls_exact_capacity(kind, chunks, scratch_bytes, fits):
    # Source + compact result + two individually padded 4B manual slots.
    owner_bytes = 8 * chunks * 64 * 4 + 32 + 64
    filler_bytes = 196608 - owner_bytes - scratch_bytes + (0 if fits else 32)
    before, _ = _ring_program(kind=kind, rows=8, columns=chunks * 64,
                              ring_shape=(2, 1), filler_bytes=filler_bytes)
    reduced = transform.AscendLowerVectorReduce()(before)
    assert len(_strided(reduced, kind)) == (chunks if fits else 0)


@pytest.mark.parametrize("chunks", [3, 5])
@pytest.mark.parametrize("active", [0, 1, 8])
def test_explicit_tree_preserves_fp32_rounding_and_empty_dynamic_effects(chunks, active):
    rows = tirx.Var("owned_rows", "int32")
    before, source, destination = _program(
        rows=8, columns=chunks * 64, row_extent=rows, source_row=2, output_row=8)
    reduced = _lower(before)
    root = reduced["main"].body.block
    memory = {buffer.data: np.full(tuple(int(extent) for extent in buffer.shape), np.nan, np.float32)
              for buffer in root.alloc_buffers}
    values = np.zeros(tuple(int(extent) for extent in source.shape), np.float32)
    # The native chunks each reduce one nonzero. Balanced FP32 order yields 1;
    # a right-associated three-leaf tree or left-associated five-leaf tree
    # yields a different result because adding 1 to 2**26 rounds away.
    leaves = [2**26, -(2**26), 1] if chunks == 3 else [2**26, 1, -(2**26), 1, 1]
    for chunk, value in enumerate(leaves):
        values[:, chunk * 64] = value
    memory[source.data][...] = values
    memory[destination.data][...] = 519
    issued = []
    _execute_rows(root.body.body, {rows: active}, memory, issued)
    expected = np.full(int(destination.shape[0]), 519, np.float32)
    expected[8 : 8 + active] = 1
    np.testing.assert_array_equal(memory[destination.data], expected)
    np.testing.assert_array_equal(memory[source.data], values)
    assert len(issued) == (chunks if active else 0)
    loops = [loop for loop in nodes(root.body.body, tirx.For)
             if loop.kind == tirx.ForKind.PARALLEL]
    assert len({loop.loop_var for loop in loops}) == len(loops)


@pytest.mark.parametrize("fits", [False, True])
def test_other_parallel_scratch_remains_subject_to_the_final_capacity_gate(fits):
    before, _, destination = _program()
    row = tirx.Var("other_row", "int32")
    ordinary = tirx.For(row, 0, 3, tirx.ForKind.PARALLEL,
                        tirx.BufferStore(destination,
                            (destination[row] + tirx.const(1, "float32")) * destination[row], [row]))
    # Reduce owns 64B. The unrelated ordinary expression needs a further 32B
    # that exists only after VectorTiles. This slice does not predict it.
    available = 96 if fits else 64
    filler = tirx.decl_buffer(((196608 - (1536 + 32) - available) // 4,),
                              "float32", "filler", scope="shared.dyn")
    reduced = _lower(_root_program(before, extra_buffers=[filler],
                                   body=seq(before["main"].body.block.body, ordinary)))
    assert len(_strided(reduced)) == 2
    lowered = transform.AscendLowerVectorTiles()(reduced)
    buffers = lowered["main"].body.block.alloc_buffers
    assert len(buffers) == len(reduced["main"].body.block.alloc_buffers) + 1
    assert _bytes(buffers) == (196608 if fits else 196640)
    # No alias relation is established here: the final physical arena must
    # account for every actual owner, regardless of its lowering origin.
    if fits:
        _merge_on_target(buffers)
    else:
        with pytest.raises(tvm.error.InternalError, match="exceeding capacity"):
            _merge_on_target(buffers)
