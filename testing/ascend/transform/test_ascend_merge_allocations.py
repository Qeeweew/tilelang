"""The common allocator consumes pairwise compatibility without assuming transitivity."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_packer_respects_complete_allocation_conflicts(disable_reuse):
    a, b, c = [tirx.decl_buffer((size,), "int32", name=name, scope="shared.dyn") for name, size in [("a", 64), ("b", 32), ("c", 32)]]
    # a is compatible with either b or c; b and c must remain disjoint.
    contract = {a.data: [b.data, c.data], b.data: [a.data], c.data: [a.data]}
    body = tirx.SeqStmt(
        [
            *[tirx.AllocBuffer(buffer) for buffer in (a, b, c)],
            *[tirx.BufferStore(buffer, index, [0]) for index, buffer in enumerate((a, b, c))],
        ]
    )
    before = tvm.IRModule({"main": tirx.PrimFunc([], tirx.AttrStmt(contract, "tl.buffer_alias_map", 1, body))})
    after = transform.MergeUBAllocations(disable_reuse=disable_reuse)(before)
    assert not [node for node in nodes(after, tirx.AttrStmt) if node.attr_key == "tl.buffer_alias_map"]
    offsets = {int(store.value): int(tvm.arith.Analyzer().simplify(store.indices[0])) for store in nodes(after, tirx.BufferStore)}
    assert offsets[1] + 32 <= offsets[2] or offsets[2] + 32 <= offsets[1]
    if disable_reuse:
        for left, left_size, right, right_size in [(0, 64, 1, 32), (0, 64, 2, 32)]:
            assert offsets[left] + left_size <= offsets[right] or offsets[right] + right_size <= offsets[left]
    else:
        assert any(offsets[0] < offsets[index] + 32 and offsets[index] < offsets[0] + 64 for index in (1, 2))


def _merge_on_target(buffers, *, arch="dav-2201", contract=True, aliases=None, disable_reuse=False):
    body = tirx.SeqStmt(
        [*[tirx.AllocBuffer(buffer) for buffer in buffers], *[tirx.BufferStore(buffer, tirx.const(0, buffer.dtype), [0]) for buffer in buffers]]
    )
    thread = tirx.IterVar(tvm.ir.Range(0, 1), tirx.Var("block", "int32"), tirx.IterVar.ThreadIndex, "blockIdx.x")
    body = tirx.AttrStmt(thread, "thread_extent", 1, body)
    if contract:
        body = tirx.AttrStmt(aliases or {buffer.data: [] for buffer in buffers}, "tl.buffer_alias_map", 1, body)
    func = tirx.PrimFunc([], body).with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return transform.MergeUBAllocations(align_bytes=32, disable_reuse=disable_reuse)(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("contract,disable_reuse", [(False, False), (True, False), (True, True)])
def test_final_ub_capacity_includes_root_epilogue(contract, disable_reuse):
    # Two 96 KiB slots fit the loop's budget exactly. A root-only epilogue is
    # still part of the physical arena even when it is absent from loop effects.
    ring = tirx.decl_buffer((2 * 24576,), "float32", name="ring", scope="shared.dyn")
    epilogue = tirx.decl_buffer((16,), "float32", name="epilogue", scope="shared.dyn")
    with pytest.raises(tvm.error.InternalError, match=r"UB \(per AIV\).*196672 bytes.*capacity 196608 bytes"):
        _merge_on_target([ring, epilogue], contract=contract, disable_reuse=disable_reuse)

    # Aligned versions plus the same epilogue now fit the bank exactly.
    ring = tirx.decl_buffer((2 * 24568,), "float32", name="ring", scope="shared.dyn")
    _merge_on_target([ring, epilogue], contract=contract, disable_reuse=disable_reuse)


def test_final_capacity_uses_actual_alias_plan():
    a, b = [tirx.decl_buffer((120 * 1024,), "uint8", name=name, scope="shared.dyn") for name in ("a", "b")]
    aliases = {a.data: [b.data], b.data: [a.data]}
    # A completion-proved shared slot fits; charging the sum would reject it.
    _merge_on_target([a, b], aliases=aliases)
    with pytest.raises(tvm.error.InternalError, match="exceeding capacity"):
        _merge_on_target([a, b], aliases=aliases, disable_reuse=True)


@pytest.mark.parametrize(
    "scope,capacity",
    [("shared.dyn", 192 * 1024)]
    + [(scope + suffix, capacity) for scope, capacity in [("shared.l1", 512 * 1024), ("shared.l0a", 64 * 1024), ("shared.l0b", 64 * 1024), ("shared.l0c", 128 * 1024)] for suffix in ("", ".dyn")],
)
def test_final_capacity_checks_single_allocations_and_each_bank(scope, capacity):
    at_limit = tirx.decl_buffer((capacity,), "uint8", name="at_limit", scope=scope)
    _merge_on_target([at_limit])
    over_limit = tirx.decl_buffer((capacity + 1,), "uint8", name="over_limit", scope=scope)
    with pytest.raises(tvm.error.InternalError, match="exceeding capacity"):
        _merge_on_target([over_limit])


def test_2201_and_3510_have_separate_final_capacity():
    ub = tirx.decl_buffer((208 * 1024,), "uint8", name="ub", scope="shared.dyn")
    _merge_on_target([ub], arch="dav-3510")
    with pytest.raises(tvm.error.InternalError, match="dav-2201.*exceeding capacity"):
        _merge_on_target([ub])


def test_final_banks_do_not_share_a_capacity_budget():
    ub = tirx.decl_buffer((192 * 1024,), "uint8", name="ub", scope="shared.dyn")
    l1 = tirx.decl_buffer((512 * 1024,), "uint8", name="l1", scope="shared.l1")
    _merge_on_target([ub, l1])


@pytest.mark.parametrize("contract", [False, True])
def test_plain_and_dynamic_scopes_consume_the_same_physical_bank(contract):
    a = tirx.decl_buffer((256 * 1024,), "uint8", name="a", scope="shared.l1")
    b = tirx.decl_buffer((256 * 1024 + 1,), "uint8", name="b", scope="shared.l1.dyn")
    with pytest.raises(tvm.error.InternalError, match="shared.l1.*exceeding capacity"):
        _merge_on_target([a, b], contract=contract)


def test_plain_and_dynamic_scopes_share_one_rewritten_arena():
    a = tirx.decl_buffer((16 * 1024,), "uint8", name="a", scope="shared.l1")
    b = tirx.decl_buffer((16 * 1024,), "uint8", name="b", scope="shared.l1.dyn")
    after = _merge_on_target([a, b])
    allocation, = nodes(after, tirx.AllocBuffer)
    assert allocation.buffer.scope() == "shared.l1.dyn"
    assert int(allocation.buffer.shape[0]) == 32 * 1024
    stores = nodes(after, tirx.BufferStore)
    assert all(store.buffer.data.same_as(allocation.buffer.data) for store in stores)
    assert sorted(int(tvm.arith.Analyzer().simplify(store.indices[0])) for store in stores) == [0, 16 * 1024]


@pytest.mark.parametrize("contract", [False, True])
def test_final_capacity_keeps_byte_sizes_in_int64(contract):
    ub = tirx.decl_buffer((2**32,), "uint8", name="large_ub", scope="shared.dyn")
    with pytest.raises(tvm.error.InternalError, match="4294967296 bytes.*exceeding capacity"):
        _merge_on_target([ub], contract=contract)


def test_final_unbounded_onchip_allocation_has_a_diagnostic():
    n = tirx.Var("n", "int32")
    ub = tirx.decl_buffer((n,), "uint8", name="dynamic_ub", scope="shared.dyn")
    with pytest.raises(tvm.error.InternalError, match="final allocation requires a static physical storage footprint"):
        _merge_on_target([ub])
