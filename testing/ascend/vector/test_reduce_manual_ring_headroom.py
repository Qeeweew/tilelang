"""Independent physical UB rings do not disable otherwise legal row plans."""

import pytest

from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import Layout
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, region


TARGET = {"kind": "ascend", "arch": "dav-2201"}
PRIVATE = "tl.ascend_aiv_private_validated"


def _program(*, kind="sum", rows=3, columns=128, ring_shape=(2, 128),
             filler_bytes=0, private=False, ring_extent=None,
             versioned_operand=None, automatic=False):
    source = tirx.decl_buffer((rows, columns), "float32", "source", scope="shared.dyn")
    output = tirx.decl_buffer((rows,), "float32", "output", scope="shared.dyn")
    ring = tirx.decl_buffer(ring_shape if ring_extent is None else (2, ring_extent),
                            "float32", "independent_ring", scope="shared.dyn")
    owners = [source, output, ring]
    if filler_bytes:
        assert filler_bytes % 32 == 0
        owners.append(tirx.decl_buffer((filler_bytes // 4,), "float32", "filler", scope="shared.dyn"))
    versioned = {"source": source, "output": output}.get(versioned_operand, ring)
    annotations = {
        "tl.buffer_versions_map" if automatic else "tl.manual_multi_buffer": {versioned.data: 2},
    }
    if private:
        # Prepared input after LowerMixedPartition: ownership is an established
        # member fact, not something the reduce lowerer infers from shapes.
        annotations[PRIVATE] = [buffer.data for buffer in owners]
    reduce = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [
        region(source, 1), region(output, 2), kind, 1, tirx.const(True, "bool"),
    ]))
    mod = kernel(reduce, buffers=owners, annotations=annotations,
                 params=[] if ring_extent is None else [ring_extent])
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    return mod, owners


def _strided(mod, kind="sum"):
    return calls(mod, "tl.ascend_membase.repeat_reduce_" + kind + "_rows_f32")


def _count(mod, kind="sum"):
    suffix = "sum" if kind == "sum" else "max_only_value"
    return calls(mod, "tl.ascend_membase.repeat_reduce_" + suffix)


def _datablock(mod, kind="sum"):
    return calls(mod, "tl.ascend_membase.datablock_reduce_" + kind)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("ring_shape", [(2, 128), (16,)], ids=["explicit_slots", "count_only_flat"])
def test_independent_manual_ring_keeps_static_dense_rows(kind, ring_shape):
    before, owners = _program(kind=kind, ring_shape=ring_shape)
    after = transform.AscendLowerVectorReduce()(before)
    assert len(_strided(after, kind)) == 2
    assert not _count(after, kind)
    assert not nodes(after, tirx.Bind)
    assert all(buffer.same_as(actual)
               for buffer, actual in zip(owners, after["main"].body.block.alloc_buffers))
    # The compiler adds its own partials; the unrelated user's physical ring
    # and version annotation retain the same owner, shape and count.
    assert int(after["main"].body.block.annotations["tl.manual_multi_buffer"][owners[2].data]) == 2


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("fits", [False, True])
def test_manual_slot_padding_is_charged_at_the_capacity_boundary(kind, fits):
    # Source=1536B, output=32B, two 4B manual slots occupy 2*32B.
    # The candidate owns 64B partials; its final binary node writes dst.
    # Charging the whole compact ring as 32B would incorrectly accept the
    # non-fitting case. Its old count plan still fits with <=64B scratch.
    owner_bytes, ring_bytes, candidate_bytes = 1536 + 32, 64, 64
    filler_bytes = 196608 - owner_bytes - ring_bytes - candidate_bytes
    if not fits:
        filler_bytes += 32
    before, _ = _program(kind=kind, ring_shape=(2, 1), filler_bytes=filler_bytes)
    after = transform.AscendLowerVectorReduce()(before)
    assert len(_strided(after, kind)) == (2 if fits else 0)
    assert bool(_count(after, kind)) != fits


@pytest.mark.parametrize("padded_alias", [False, True])
def test_flat_manual_owner_aliases_are_charged_once_at_their_maximum_footprint(padded_alias):
    # The exact-capacity candidate fits if identical views of one flat manual
    # owner are charged once. An alias reserving 32B more must force fallback;
    # the original <=64B count scratch still fits in that case.
    filler_bytes = 196608 - (1536 + 32 + 64) - 64
    before, owners = _program(ring_shape=(16,), filler_bytes=filler_bytes)
    ring = owners[2]
    alias = tirx.decl_buffer((16,), "float32", "ring_alias", scope="shared.dyn", data=ring.data)
    annotations = dict(before["main"].body.block.annotations.items())
    annotations["layout_map"] = {
        ring: Layout([16], lambda i: [i]),
        alias: Layout([16], lambda i: [i + (8 if padded_alias else 0)]),
    }
    mod = kernel(before["main"].body.block.body, buffers=owners, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = transform.AscendLowerVectorReduce()(mod)
    assert len(_strided(after)) == (0 if padded_alias else 2)
    assert bool(_count(after)) == padded_alias


@pytest.mark.parametrize("operand", ["source", "output"])
@pytest.mark.parametrize("columns", [128, 1024])
@pytest.mark.parametrize("automatic", [False, True])
def test_native_operand_keeps_existing_plan_until_its_layout_is_known(operand, columns, automatic):
    # A leading version axis may acquire a different row stride later. In
    # particular output[2] needs 32B per result when it is a two-slot ring.
    before, _ = _program(rows=2, columns=columns, versioned_operand=operand,
                         automatic=automatic)
    after = transform.AscendLowerVectorReduce()(before)
    assert not _strided(after)
    assert [int(call.args[-1]) for call in _count(after)] == [columns, columns // 64]


@pytest.mark.parametrize("columns", [128, 1024])
def test_independent_fixed_versions_keep_native_rows(columns):
    before, _ = _program(columns=columns, automatic=True)
    after = transform.AscendLowerVectorReduce()(before)
    assert len(_strided(after)) == (2 if columns == 128 else 1)
    assert not _count(after)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("fits", [False, True])
@pytest.mark.parametrize("padded_alias", [False, True])
def test_fixed_logical_slots_charge_aligned_owner_maximum(kind, fits, padded_alias):
    # Each logical 4B slot needs 32B; the padded alias raises it to 64B.
    # Fixed counts expand logical slots, unlike a user's physical manual ring.
    ring_bytes = (64 if padded_alias else 32) * 2
    filler_bytes = 196608 - (1536 + 32 + ring_bytes) - 64 + (0 if fits else 32)
    before, owners = _program(kind=kind, ring_shape=(1,), automatic=True,
                              filler_bytes=filler_bytes)
    ring = owners[2]
    alias = tirx.decl_buffer((1,), "float32", "ring_alias", scope="shared.dyn", data=ring.data)
    annotations = dict(before["main"].body.block.annotations.items())
    annotations["layout_map"] = {
        ring: Layout([1], lambda i: [i]),
        alias: Layout([1], lambda i: [i + (8 if padded_alias else 0)]),
    }
    mod = kernel(before["main"].body.block.body, buffers=owners, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = transform.AscendLowerVectorReduce()(mod)
    assert len(_strided(after, kind)) == (2 if fits else 0)
    assert bool(_count(after, kind)) != fits


@pytest.mark.parametrize("columns", [128, 1024])
def test_unknown_manual_owner_footprint_keeps_existing_plan(columns):
    capacity = tirx.Var("capacity", "int32")
    before, _ = _program(columns=columns, ring_extent=capacity)
    after = transform.AscendLowerVectorReduce()(before)
    assert not _strided(after)
    assert [int(call.args[-1]) for call in _count(after)] == [columns, columns // 64]


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("columns", [192, 1024])
def test_new_partials_and_vector_combine_keep_validated_private_ownership(kind, columns):
    before, owners = _program(kind=kind, columns=columns, private=True)
    reduced = transform.AscendLowerVectorReduce()(before)
    wide = columns == 1024
    assert len(_strided(reduced, kind)) == (1 if wide else 3)
    partials = _strided(reduced, kind)[0].args[1 if wide else 0].args[0].buffer
    assert tuple(int(extent) for extent in partials.shape) == ((3, 16) if wide else (3, 8))
    assert len(_datablock(reduced, kind)) == (2 if wide else 0)
    if wide:
        first, second = _datablock(reduced, kind)
        blocks = first.args[0].args[0].buffer
        assert tuple(int(extent) for extent in blocks.shape) == (3, 128)
        assert second.args[1].args[0].buffer.same_as(blocks)
        assert second.args[0].args[0].buffer.same_as(partials)
        assert not blocks.data.same_as(partials.data)
        for call, written, read in ((first, 384, 3072), (second, 48, 384)):
            assert [int(value) for value in call.args[0].args[1:]] == [written, 2]
            assert [int(value) for value in call.args[1].args[1:]] == [read, 1]
            assert int(call.args[2]) == read
    lowered = transform.AscendLowerVectorTiles()(reduced)
    assert len(_strided(lowered, kind)) == (1 if wide else 3)
    assert len(_count(lowered, kind)) == 0
    assert len(_datablock(lowered, kind)) == (2 if wide else 0)
    assert len(calls(lowered, "tl.ascend_membase." + ("add" if kind == "sum" else "max"))) == (0 if wide else 2)
    members = lowered["main"].body.block.annotations[PRIVATE]
    assert all(any(member.same_as(buffer.data) for member in members)
               for buffer in lowered["main"].body.block.alloc_buffers)
    assert len(before["main"].body.block.annotations[PRIVATE]) == len(owners)
    assert not nodes(lowered, tirx.Bind)


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("fits", [False, True])
def test_two_level_partials_and_manual_slot_padding_are_charged_exactly(kind, fits):
    # The row-major partials need 3*16*4=192B and no binary-combine reserve.
    # The unrelated two-slot ring still occupies two aligned 32B slots.
    owner_bytes, ring_bytes, partial_bytes = 3 * 1024 * 4 + 32, 64, 192
    filler_bytes = 196608 - owner_bytes - ring_bytes - partial_bytes
    if not fits:
        filler_bytes += 32
    before, owners = _program(kind=kind, columns=1024, ring_shape=(2, 1),
                              filler_bytes=filler_bytes)
    after = transform.AscendLowerVectorReduce()(before)
    assert not _datablock(after, kind)
    assert len(_strided(after, kind)) == (1 if fits else 0)
    assert [int(call.args[-1]) for call in _count(after, kind)] == ([3072] if fits else [1024, 16])
    assert int(after["main"].body.block.annotations["tl.manual_multi_buffer"][owners[2].data]) == 2


@pytest.mark.parametrize("padded_alias", [False, True])
def test_two_level_partials_keep_the_shared_manual_owner_maximum_footprint(padded_alias):
    filler_bytes = 196608 - (3 * 1024 * 4 + 32 + 64) - 192
    before, owners = _program(columns=1024, ring_shape=(16,), filler_bytes=filler_bytes)
    ring = owners[2]
    alias = tirx.decl_buffer((16,), "float32", "ring_alias", scope="shared.dyn", data=ring.data)
    annotations = dict(before["main"].body.block.annotations.items())
    annotations["layout_map"] = {
        ring: Layout([16], lambda i: [i]),
        alias: Layout([16], lambda i: [i + (8 if padded_alias else 0)]),
    }
    mod = kernel(before["main"].body.block.body, buffers=owners, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = transform.AscendLowerVectorReduce()(mod)
    assert not _datablock(after)
    assert len(_strided(after)) == (0 if padded_alias else 1)
    assert [int(call.args[-1]) for call in _count(after)] == ([1024, 16] if padded_alias else [3072])


@pytest.mark.parametrize("kind", ["sum", "max"])
@pytest.mark.parametrize("fits", [False, True])
def test_datablock_scratch_and_manual_slot_padding_select_the_exact_capacity_plan(kind, fits):
    # Three stages require disjoint 1536B datablocks and 192B partials.
    # Two manual 4B slots occupy 64B. Losing one 32B unit must retain the
    # smaller 192B two-level plan, not a scalar row fallback.
    owner_bytes, ring_bytes, scratch_bytes = 3 * 1024 * 4 + 32, 64, 1536 + 192
    filler_bytes = 196608 - owner_bytes - ring_bytes - scratch_bytes
    if not fits:
        filler_bytes += 32
    before, owners = _program(kind=kind, columns=1024, ring_shape=(2, 1),
                              filler_bytes=filler_bytes)
    after = transform.AscendLowerVectorReduce()(before)
    assert len(_strided(after, kind)) == 1
    assert len(_datablock(after, kind)) == (2 if fits else 0)
    assert [int(call.args[-1]) for call in _count(after, kind)] == ([] if fits else [3072])
    partials = _strided(after, kind)[0].args[1].args[0].buffer
    assert tuple(int(value) for value in partials.shape) == (3, 16)
    assert [int(value) for value in _strided(after, kind)[0].args[2:]] == [3, 2, 16]
    assert int(after["main"].body.block.annotations["tl.manual_multi_buffer"][owners[2].data]) == 2


@pytest.mark.parametrize("padded_alias", [False, True])
def test_datablock_headroom_charges_a_shared_manual_owner_once_at_its_maximum(padded_alias):
    filler_bytes = 196608 - (3 * 1024 * 4 + 32 + 64) - (1536 + 192)
    before, owners = _program(columns=1024, ring_shape=(16,), filler_bytes=filler_bytes)
    ring = owners[2]
    alias = tirx.decl_buffer((16,), "float32", "ring_alias", scope="shared.dyn", data=ring.data)
    annotations = dict(before["main"].body.block.annotations.items())
    annotations["layout_map"] = {
        ring: Layout([16], lambda i: [i]),
        alias: Layout([16], lambda i: [i + (8 if padded_alias else 0)]),
    }
    mod = kernel(before["main"].body.block.body, buffers=owners, annotations=annotations)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target(TARGET))
    after = transform.AscendLowerVectorReduce()(mod)
    assert len(_strided(after)) == 1
    assert len(_datablock(after)) == (0 if padded_alias else 2)
    assert [int(call.args[-1]) for call in _count(after)] == ([3072] if padded_alias else [])
