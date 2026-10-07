"""Frontend GM-region contracts for the persistent paired-query example.

These tests inspect the actual PrimFunc, without scheduling or device lowering.
They do not model vector arithmetic, synchronization, or physical capacity.
"""

import collections
import importlib.util
import itertools
from pathlib import Path
import sys

import pytest

from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import nodes


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[3] / "examples/ascend/flash_attention/persistent_910b.py"
    spec = importlib.util.spec_from_file_location("_tilelang_persistent_attention_910b", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _integer(expr, values, definitions):
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    if isinstance(expr, tirx.Var):
        return values[expr] if expr in values else _integer(definitions[expr], values, definitions)
    if isinstance(expr, tirx.Cast):
        assert str(expr.dtype).startswith(("int", "uint", "bool"))
        return _integer(expr.value, values, definitions)
    if isinstance(expr, tirx.Not):
        return int(not _integer(expr.a, values, definitions))
    operators = {
        tirx.Add: lambda a, b: a + b,
        tirx.Sub: lambda a, b: a - b,
        tirx.Mul: lambda a, b: a * b,
        tirx.FloorDiv: lambda a, b: a // b,
        tirx.FloorMod: lambda a, b: a % b,
        tirx.EQ: lambda a, b: a == b,
        tirx.NE: lambda a, b: a != b,
        tirx.LT: lambda a, b: a < b,
        tirx.LE: lambda a, b: a <= b,
        tirx.GT: lambda a, b: a > b,
        tirx.GE: lambda a, b: a >= b,
        tirx.And: lambda a, b: bool(a and b),
        tirx.Or: lambda a, b: bool(a or b),
    }
    assert type(expr) in operators, ("Unsupported address/guard expression", expr)
    return int(operators[type(expr)](_integer(expr.a, values, definitions), _integer(expr.b, values, definitions)))


def _tile_sites(stmt, loops=(), guards=(), cube=False, operation="tl.tileop.ascend_copy"):
    if isinstance(stmt, tirx.SeqStmt):
        for child in stmt.seq:
            yield from _tile_sites(child, loops, guards, cube, operation)
    elif isinstance(stmt, tirx.For):
        yield from _tile_sites(stmt.body, (*loops, stmt), guards, cube, operation)
    elif isinstance(stmt, tirx.IfThenElse):
        yield from _tile_sites(stmt.then_case, loops, (*guards, (stmt.condition, True)), cube, operation)
        if stmt.else_case is not None:
            yield from _tile_sites(stmt.else_case, loops, (*guards, (stmt.condition, False)), cube, operation)
    elif isinstance(stmt, tirx.SBlockRealize):
        yield from _tile_sites(stmt.block, loops, (*guards, (stmt.predicate, True)), cube, operation)
    elif isinstance(stmt, tirx.SBlock):
        yield from _tile_sites(stmt.body, loops, guards, cube or stmt.name_hint == "CUBE", operation)
    elif isinstance(stmt, tirx.AttrStmt):
        yield from _tile_sites(stmt.body, loops, guards, cube, operation)
    elif isinstance(stmt, tirx.Evaluate):
        call = stmt.value
        if isinstance(call, tirx.Call) and isinstance(call.op, tvm.ir.Op) and (operation is None or call.op.name == operation):
            yield call, loops, guards, cube
    else:
        assert isinstance(stmt, (tirx.Bind, tirx.BufferStore)), ("Unsupported frontend statement", stmt)


def _loop_values(loops, definitions, values=None):
    values = {} if values is None else values
    if not loops:
        yield values
        return
    loop, *rest = loops
    assert loop.kind in (tirx.ForKind.SERIAL, tirx.ForKind.THREAD_BINDING)
    start = _integer(loop.min, values, definitions)
    extent = _integer(loop.extent, values, definitions)
    assert loop.step is None or _integer(loop.step, values, definitions) == 1
    for value in range(start, start + extent):
        yield from _loop_values(rest, definitions, {**values, loop.loop_var: value})


def _gm_transfers(program, groups):
    buffers = {buffer.name: buffer for buffer in program.buffer_map.values()}
    assert set(buffers) == {"Q", "K", "V", "Score", "P", "PV", "OState", "O"}
    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    launch = {loop.thread_binding.thread_tag: loop for loop in nodes(program, tirx.For) if loop.thread_binding is not None}
    group, sid = launch["blockIdx.x"].loop_var, launch["cthread"].loop_var
    assert int(launch["blockIdx.x"].extent) == groups and int(launch["cthread"].extent) == 2
    trips = 12 // groups * 3 + 2
    (epoch,) = [loop for loop in nodes(program, tirx.For) if loop.kind == tirx.ForKind.SERIAL and int(loop.extent) == trips]
    actual = collections.Counter()
    for call, loops, guards, cube in _tile_sites(program.body):
        for values in _loop_values(loops, definitions):
            # A frontend CUBE scope is one AIC task, although it is lexically
            # inside the mixed kernel's two-SID launch loop.
            if cube and values[sid] != 0:
                continue
            if not all(bool(_integer(guard, values, definitions)) == expected for guard, expected in guards):
                continue
            for writing, region in enumerate(call.args[:2]):
                assert isinstance(region, tirx.Call) and region.op.name == "tl.region"
                load = region.args[0]
                assert isinstance(load, tirx.BufferLoad) and int(region.args[1]) == writing + 1
                name = next((name for name, buffer in buffers.items() if buffer.data.same_as(load.buffer.data)), None)
                if name is None:
                    continue
                origin = tuple(_integer(value, values, definitions) for value in load.indices)
                shape = tuple(_integer(value, values, definitions) for value in region.args[2:])
                assert len(origin) == len(shape) == len(buffers[name].shape)
                assert all(0 <= start < start + size <= int(bound) for start, size, bound in zip(origin, shape, buffers[name].shape))
                actual[name, bool(writing), values[group], values[epoch.loop_var], origin, shape] += 1
    return actual, buffers


def _expected_transfers(groups, block_kv=16, pv_chunk=None):
    expected = collections.Counter()

    def add(name, writing, group, epoch, origin, shape):
        expected[name, writing, group, epoch, origin, shape] += 1

    # Cartesian query identities distributed round robin are independent of
    # the factory's flattened-index division/modulo expressions.
    pairs = list(itertools.product(range(2), range(2), range(3)))
    for group in range(groups):
        for local_pair, (batch, head, query) in enumerate(pairs[group::groups]):
            for kv in range(3):
                task = local_pair * 3 + kv
                for owner in range(2):
                    if kv == 0:
                        add("Q", False, group, task, (batch, head, query, owner, 0, 0), (1, 1, 1, 1, 32, 16))
                    add("Score", True, group, task, (group, owner, task % 2, 0, 0), (1, 1, 1, 32, block_kv))
                    if pv_chunk is None:
                        add("P", False, group, task + 1, (group, owner, task % 2, 0, 0), (1, 1, 1, 32, block_kv))
                    else:
                        for column in range(0, block_kv, pv_chunk):
                            add("P", False, group, task + 1, (group, owner, task % 2, 0, column), (1, 1, 1, 32, pv_chunk))
                    add("PV", True, group, task + 1, (group, owner, task % 2, 0, 0), (1, 1, 1, 32, 16))
                    for row in (0, 16):
                        for name, writing in (("Score", False), ("P", True)):
                            add(name, writing, group, task + 1, (group, owner, task % 2, row, 0), (1, 1, 1, 16, block_kv))
                        add("PV", False, group, task + 2, (group, owner, task % 2, row, 0), (1, 1, 1, 16, 16))
                        # With even group counts, flattened-pair parity is
                        # constant per group. Query state must still alternate.
                        state = (group, owner, local_pair % 2, row, 0)
                        if kv:
                            add("OState", False, group, task + 2, state, (1, 1, 1, 16, 16))
                        if kv < 2:
                            add("OState", True, group, task + 2, state, (1, 1, 1, 16, 16))
                        else:
                            add("O", True, group, task + 2, (batch, head, query, owner, row, 0), (1, 1, 1, 1, 16, 16))
                add("K", False, group, task, (batch, head, kv * block_kv, 0), (1, 1, block_kv, 16))
                if pv_chunk is None:
                    add("V", False, group, task, (batch, head, kv * block_kv, 0), (1, 1, block_kv, 16))
                else:
                    for column in range(0, block_kv, pv_chunk):
                        add("V", False, group, task + 1, (batch, head, kv * block_kv + column, 0), (1, 1, pv_chunk, 16))
    return expected


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("groups", [2, 4])
def test_real_frontend_keeps_query_ownership_and_group_private_rings(example, dtype, groups):
    program = example.persistent_attention_fwd(batch=2, heads=2, query_pairs=3, kv_tiles=3,
                                             groups=groups, block_q=32, block_kv=16,
                                             head_dim=16, row_block=16, dtype=dtype)
    assert str(program.attrs["global_symbol"]) == "parallel_fa_" + dtype
    actual, buffers = _gm_transfers(program, groups)
    assert actual == _expected_transfers(groups)
    for name, buffer in buffers.items():
        assert buffer.dtype == ("float32" if name in ("Score", "PV", "OState") else dtype)
        strides, running = [], 1
        for extent in reversed(buffer.shape):
            strides.append(running)
            running *= int(extent)
        assert not buffer.strides or tuple(int(value) for value in buffer.strides) == tuple(reversed(strides))
        assert int(buffer.elem_offset) == 0
    for name in ("Score", "P", "PV", "OState"):
        assert tuple(int(value) for value in buffers[name].shape) == (groups, 2, 2, 32, 16)
    output = collections.Counter(key[4] for key, count in actual.items() if key[0] == "O" for _ in range(count))
    assert output == collections.Counter((batch, head, query, owner, row, 0)
                                         for batch, head, query, owner, row in itertools.product(range(2), range(2), range(3), range(2), (0, 16)))


@pytest.mark.parametrize("dtype,groups,block_kv,pv_chunk", [
    ("float16", 2, 32, 32),
    ("float16", 2, 64, 32),
    ("bfloat16", 4, 64, 16),
])
def test_chunked_frontend_shares_v_and_keeps_owner_accumulators_until_final_export(
        example, dtype, groups, block_kv, pv_chunk):
    program = example.persistent_attention_fwd(
        batch=2, heads=2, query_pairs=3, kv_tiles=3, groups=groups,
        block_q=32, block_kv=block_kv, head_dim=16, row_block=16,
        pv_chunk=pv_chunk, dtype=dtype)
    assert str(program.attrs["global_symbol"]) == "chunked_fa_" + dtype
    actual, buffers = _gm_transfers(program, groups)
    assert actual == _expected_transfers(groups, block_kv, pv_chunk)
    assert not buffers["P"].strides or int(buffers["P"].strides[-2]) == block_kv

    # Inspect actual typed operands and loop ancestry, not a Python copy of
    # the factory's chunk/owner address expressions.
    sites = list(_tile_sites(program.body, operation=None))
    pv_gemms = [site for site in sites if site[0].op.name == "tl.tileop.gemm" and not int(site[0].args[4])]
    (gemm, loops, guards, cube), = pv_gemms
    assert cube
    p_region, v_region, c_region = gemm.args[:3]
    p_l1, v_l1, accumulator = [region.args[0].buffer for region in (p_region, v_region, c_region)]
    assert tuple(map(int, p_l1.shape)) == (32, pv_chunk)
    assert tuple(map(int, v_l1.shape)) == (2, pv_chunk, 16)
    assert tuple(map(int, accumulator.shape)) == (2, 32, 16)
    assert tuple(map(int, c_region.args[2:])) == (1, 32, 16)
    assert all(int(value) == 0 for value in c_region.args[0].indices[1:])
    assert int(c_region.args[1]) == 3  # The first-chunk predicate is not a static write-only claim.
    (root,) = [block for block in nodes(program, tirx.SBlock) if "tl.buffer_versions_map" in block.annotations]
    assert int(root.annotations["tl.buffer_versions_map"][accumulator.data]) == 1
    assert int(root.annotations["tl.buffer_versions_map"][p_l1.data]) == 1
    assert int(root.annotations["tl.manual_multi_buffer"][v_l1.data]) == 2

    (chunk_var,) = nodes(gemm.args[9], tirx.Var)
    (chunk_loop,) = [loop for loop in loops if loop.loop_var.same_as(chunk_var)]
    owner_var = c_region.args[0].indices[0]
    (owner_loop,) = [loop for loop in loops if loop.loop_var.same_as(owner_var)]
    assert int(chunk_loop.extent) == block_kv // pv_chunk and int(owner_loop.extent) == 2
    (v_copy,) = [site for site in sites if site[0].op.name == "tl.tileop.ascend_copy"
                and site[0].args[0].args[0].buffer.data.same_as(buffers["V"].data)]
    assert v_copy[0].args[1].args[0].buffer.data.same_as(v_l1.data)
    assert any(loop.same_as(chunk_loop) for loop in v_copy[1])
    assert not any(loop.same_as(owner_loop) for loop in v_copy[1])
    (export,) = [site for site in sites if site[0].op.name == "tl.tileop.ascend_copy"
                and site[0].args[1].args[0].buffer.data.same_as(buffers["PV"].data)]
    assert export[0].args[0].args[0].buffer.data.same_as(accumulator.data)
    assert not any(loop.same_as(chunk_loop) for loop in export[1])
    assert next(i for i, site in enumerate(sites) if site is export) > next(
        i for i, site in enumerate(sites) if site is pv_gemms[0])

    definitions = {binding.var: binding.value for binding in nodes(program, tirx.Bind)}
    launch = {loop.thread_binding.thread_tag: loop for loop in loops if loop.thread_binding is not None}
    group, sid = launch["blockIdx.x"].loop_var, launch["cthread"].loop_var
    (epoch,) = [loop for loop in loops if loop.kind == tirx.ForKind.SERIAL and int(loop.extent) == 12 // groups * 3 + 2]
    accumulation = collections.defaultdict(list)
    for values in _loop_values(loops, definitions):
        if values[sid] or not all(bool(_integer(guard, values, definitions)) == expected for guard, expected in guards):
            continue
        chunk = values[chunk_var]
        assert _integer(v_region.args[0].indices[0], values, definitions) == chunk % 2
        accumulation[values[group], values[epoch.loop_var], values[owner_var]].append(
            (chunk, bool(_integer(gemm.args[9], values, definitions))))
    expected = {(group, epoch, owner): [(chunk, chunk == 0) for chunk in range(block_kv // pv_chunk)]
                for group in range(groups) for epoch in range(1, 12 // groups * 3 + 1) for owner in range(2)}
    assert dict(accumulation) == expected


@pytest.mark.parametrize("changes,message", [
    ({"batch": True}, "positive integers"),
    ({"groups": 5}, "divide evenly"),
    ({"block_q": 24}, "Aligned dense tiles"),
    ({"row_block": 48}, "complete UB row subtiles"),
    ({"dtype": "float32"}, "FP16/BF16"),
    ({"pv_chunk": True}, "pv_chunk must be a positive integer"),
    ({"pv_chunk": 0}, "pv_chunk must be a positive integer"),
    ({"pv_chunk": 8}, "pv_chunk must be 16-aligned"),
    ({"block_kv": 48, "pv_chunk": 32}, "divide block_kv"),
])
def test_invalid_contract_is_rejected_before_frontend_construction(example, changes, message):
    options = dict(batch=2, heads=2, query_pairs=3, kv_tiles=3, groups=2,
                   block_q=32, block_kv=16, head_dim=16, row_block=16)
    options.update(changes)
    with pytest.raises(ValueError, match=message):
        example.persistent_attention_fwd(**options)
