"""Expression planning preserves assignment boundaries and native effects.

These inputs exercise the real host pass. Small count-mode evaluation checks
rounding/data flow; row and predicate helpers retain their exact native regions.
"""

import numpy as np
import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq
from testing.ascend.vector.test_vector_select import _eval


def _lower(body, buffers, params=()):
    before = kernel(body, buffers=buffers, params=params, annotations={
        "tl.ascend_aiv_private_validated": [buffer.data for buffer in buffers],
    })
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return transform.AscendLowerVectorTiles()(before)


def _parallel(body, *axes):
    for variable, extent in reversed(axes):
        body = tirx.For(variable, 0, extent, tirx.ForKind.PARALLEL, body)
    return body


def _native(after):
    return [call for call in nodes(after, tirx.Call)
            if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")]


def _load(access):
    assert isinstance(access, tirx.Call) and access.op.name in ("tl.access_ptr", "tl.region")
    return access.args[0]


def _scratch_contract(after, declared):
    root = after["main"].body.block
    declared_data = {buffer.data for buffer in declared}
    scratch = [buffer for buffer in root.alloc_buffers if buffer.data not in declared_data]
    assert scratch
    members = set(root.annotations["tl.ascend_aiv_private_validated"])
    assert members == {buffer.data for buffer in root.alloc_buffers}
    reads, writes = set(), set()
    for operation in _native(after):
        for argument in operation.args:
            if not isinstance(argument, tirx.Call) or not isinstance(argument.op, tvm.ir.Op):
                continue
            if argument.op.name not in ("tl.access_ptr", "tl.region"):
                continue
            effect = int(argument.args[2] if argument.op.name == "tl.access_ptr" else argument.args[1])
            if effect & 1:
                reads.add(_load(argument).buffer.data)
            if effect & 2:
                writes.add(_load(argument).buffer.data)
    writes.update(store.buffer.data for store in nodes(after, tirx.BufferStore))
    for buffer in scratch:
        assert buffer.scope() == "shared.dyn"
        elements = int(np.prod([int(size) for size in buffer.shape]))
        bits = tvm.DataType(buffer.dtype).bits
        assert elements * bits % (32 * 8) == 0
        assert buffer.data in reads and buffer.data in writes
    return scratch


def _bf16(values):
    values = np.array(values, dtype=np.float32, copy=True)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _run_count_program(after, initial):
    """Execute emitted typed count spans; BF16 storage uses exact FP32 values."""
    values = {buffer.data: np.array(value, dtype=np.float32, copy=True).ravel()
              for buffer, value in initial}

    def read(pointer, count):
        load = _load(pointer)
        assert len(load.indices) == 1
        start = int(load.indices[0])
        return values[load.buffer.data][start:start + count].copy()

    for operation in _native(after):
        name = operation.op.name.rsplit(".", 1)[1]
        count = int(operation.args[-1])
        destination = _load(operation.args[0])
        lhs = read(operation.args[1], count)
        if name == "bfloat162float":
            result = lhs
        elif name == "float2bfloat16_rn":
            result = _bf16(lhs)
        elif name == "exp":
            result = np.exp(lhs)
        else:
            rhs = float(operation.args[2]) if name.endswith("_scalar") else read(operation.args[2], count)
            result = {"add": np.add, "mul": np.multiply}[name.removesuffix("_scalar")](lhs, rhs)
        if destination.buffer.data not in values:
            values[destination.buffer.data] = np.zeros(int(destination.buffer.shape[0]), dtype=np.float32)
        start = int(destination.indices[0])
        values[destination.buffer.data][start:start + count] = result
    return values


def test_shared_expression_node_is_replanned_after_an_input_write():
    count = 17
    a, b, first, second = [tirx.decl_buffer((count,), "float32", name=name, scope="shared.dyn")
                           for name in ("a", "b", "first", "second")]
    i = tirx.Var("lane", "int32")
    shared = tirx.exp(a[i] + b[i])
    body = _parallel(seq(
        tirx.BufferStore(first, shared, [i]),
        tirx.BufferStore(a, a[i] + tirx.const(1, "float32"), [i]),
        tirx.BufferStore(second, shared, [i]),
    ), (i, count))
    after = _lower(body, [a, b, first, second])
    operations = _native(after)
    assert [operation.op.name.rsplit(".", 1)[1] for operation in operations] == [
        "add", "exp", "add_scalar", "add", "exp"]
    assert _load(operations[1].args[0]).buffer.data.same_as(first.data)
    assert _load(operations[2].args[0]).buffer.data.same_as(a.data)
    assert _load(operations[-1].args[0]).buffer.data.same_as(second.data)
    assert not _load(operations[0].args[0]).buffer.data.same_as(_load(operations[3].args[0]).buffer.data)
    av, bv = np.linspace(-1, 1, count, dtype=np.float32), np.linspace(0, 0.5, count, dtype=np.float32)
    values = _run_count_program(after, [(a, av), (b, bv)])
    np.testing.assert_array_equal(values[first.data], np.exp(av + bv))
    np.testing.assert_array_equal(values[second.data], np.exp((av + np.float32(1)) + bv))
    _scratch_contract(after, [a, b, first, second])


def test_repeated_bfloat_dag_keeps_each_round_and_private_scratch_effect():
    count = 17
    a, b = [tirx.decl_buffer((count,), "bfloat16", name=name, scope="shared.dyn") for name in ("a", "b")]
    output = tirx.decl_buffer((count,), "float32", name="output", scope="shared.dyn")
    i = tirx.Var("lane", "int32")
    shared = a[i] + b[i]
    after = _lower(_parallel(tirx.BufferStore(output, tirx.Cast("float32", shared * shared), [i]),
                             (i, count)), [a, b, output])
    operations = _native(after)
    assert [operation.op.name.rsplit(".", 1)[1] for operation in operations] == [
        "bfloat162float", "bfloat162float", "add", "float2bfloat16_rn",
        "bfloat162float", "mul", "float2bfloat16_rn", "bfloat162float"]
    middle_round, middle_widen, multiply = operations[3:6]
    assert _load(middle_widen.args[1]).buffer.data.same_as(_load(middle_round.args[0]).buffer.data)
    assert all(_load(pointer).buffer.data.same_as(_load(middle_widen.args[0]).buffer.data)
               for pointer in multiply.args[1:3])
    for operation in operations:
        assert int(operation.args[-1]) == count
        assert [int(pointer.args[1]) for pointer in operation.args[:2]] == [count, count]
        assert int(operation.args[0].args[2]) == 2 and int(operation.args[1].args[2]) == 1
    av = np.ones(count, dtype=np.float32)
    bv = np.resize(np.array([0.00390625, 0.01171875], dtype=np.float32), count)
    expected = _bf16(_bf16(av + bv) * _bf16(av + bv))
    assert np.any(expected != _bf16((av + bv) * (av + bv)))
    np.testing.assert_array_equal(_run_count_program(after, [(a, av), (b, bv)])[output.data], expected)
    _scratch_contract(after, [a, b, output])


@pytest.mark.parametrize("same_row", [True, False], ids=["share-row", "distinct-row-storage"])
def test_row_broadcast_cache_is_expression_local_and_storage_aware(same_row):
    rows, columns = 8, 128
    a, b, output = [tirx.decl_buffer((rows, columns), "float32", name=name, scope="shared.dyn")
                    for name in ("a", "b", "output")]
    row0, row1 = [tirx.decl_buffer((rows,), "float32", name=name, scope="shared.dyn")
                  for name in ("row0", "row1")]
    i, j = tirx.Var("row", "int32"), tirx.Var("column", "int32")
    first_row = row0[i]
    second_row = first_row if same_row else row1[i]
    body = _parallel(tirx.BufferStore(output, (a[i, j] - first_row) * (b[i, j] - second_row), [i, j]),
                     (i, rows), (j, columns))
    declared = [a, b, output, row0, row1]
    after = _lower(body, declared)
    broadcasts = calls(after, "tl.ascend_membase.brcb_f32")
    assert len(broadcasts) == (1 if same_row else 2)
    assert [_load(operation.args[1]).buffer.data for operation in broadcasts] == (
        [row0.data] if same_row else [row0.data, row1.data])
    blocks = {_load(operation.args[0]).buffer.data for operation in broadcasts}
    binaries = calls(after, "tl.ascend_membase.sub_rows_f32")
    assert len(binaries) == 4
    for operation in binaries:
        destination, matrix, row = operation.args[:3]
        assert [int(access.args[1]) for access in (destination, matrix, row)] == [2, 1, 1]
        assert [int(size) for size in destination.args[2:]] == [rows, 64]
        assert [int(size) for size in matrix.args[2:]] == [rows, 64]
        assert _load(row).buffer.data in blocks and int(row.args[2]) == rows * 8
    multiply, = calls(after, "tl.ascend_membase.mul")
    assert _load(multiply.args[0]).buffer.data.same_as(output.data)
    assert int(multiply.args[-1]) == rows * columns
    _scratch_contract(after, declared)


def test_nested_selection_uses_exact_chunks_and_registers_predicate_scratch():
    width, capacity = 130, 136
    a, b, output = [tirx.decl_buffer((capacity,), "float32", name=name, scope="shared.dyn")
                    for name in ("a", "b", "output")]
    i, threshold = tirx.Var("column", "int32"), tirx.Var("threshold", "int32")
    selected = tirx.Select(i < threshold, a[i] * tirx.const(2, "float32"), b[i] + tirx.const(1, "float32"))
    after = _lower(_parallel(tirx.BufferStore(output, tirx.exp(selected), [i]), (i, width)),
                   [a, b, output], [threshold])
    selection, = calls(after, "tl.ascend_membase.select_bits")
    exponential, = calls(after, "tl.ascend_membase.exp")
    assert _load(selection.args[0]).buffer.data.same_as(_load(exponential.args[1]).buffer.data)
    assert _load(exponential.args[0]).buffer.data.same_as(output.data)
    assert int(exponential.args[-1]) == width
    loop, = nodes(after, tirx.For)
    assert loop.kind == tirx.ForKind.SERIAL and int(loop.extent) == 3
    assert [_eval(selection.args[-1], after, chunk=chunk) for chunk in range(3)] == [64, 64, 2]
    for chunk in range(3):
        for pointer in selection.args[:3]:
            assert int(pointer.args[2]) in (1, 2)
            assert _eval(pointer.args[1], after, chunk=chunk) == min(64, width - chunk * 64)
            assert _eval(_load(pointer).indices[0], after, chunk=chunk) == chunk * 64
    predicate = _load(selection.args[3]).buffer
    assert predicate.dtype == "uint64" and int(selection.args[3].args[1]) == 2
    stores = [store for store in nodes(loop.body, tirx.BufferStore) if store.buffer.data.same_as(predicate.data)]
    assert len(stores) == 2
    for chunk in range(3):
        prefix = min(max(65 - chunk * 64, 0), 64)
        assert [(_eval(store.value, after, threshold=65, chunk=chunk) & ((1 << 64) - 1))
                for store in stores] == [(1 << prefix) - 1, 0]
    scratch = _scratch_contract(after, [a, b, output])
    assert predicate.data in {buffer.data for buffer in scratch}
