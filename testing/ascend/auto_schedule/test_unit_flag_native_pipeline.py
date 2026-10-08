"""Logical native-K unit controls survive ordinary scheduled fill/drain.

These CPU cases consume real typed MAD/copy IR and the common multibuffer,
sync and scheduled-loop passes. They do not model native unit internals or
replace operand/full-completion dependencies with unit protection.
"""

from collections import Counter

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.layout import make_ascend_nz_layout
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, seq
from testing.ascend.auto_schedule._scheduled_ir import unit
from testing.ascend.auto_schedule.test_unit_flag_activation_guards import _replace_root
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _COPY, _DIRECTION, _GEMM, _SET, _WAIT, _check_direction_boundaries, _fixture, _insert, _is_export,
    _matching_completion, _name, _trace, _value,
)


def _pipeline(*, native_k=2, mad_stage=1, versions=2, guarded=False,
              rejection=None, boxed_stage_bound=False):
    before, facts = _fixture(
        native_k=max(native_k, 2), c_backing_n=64,
        activation_guard="direct" if guarded else None,
        persistent_inputs=guarded,
        kind="missing-terminal" if rejection == "missing-terminal" else "ordinary",
    )
    root = before["main"].body.block
    a, b = [next(buffer for buffer in root.alloc_buffers if buffer.name == name)
            for name in ("a", "b")]
    chunk = tirx.Var("chunk", "int32")
    chunks = 2
    changed = Counter()

    def mutate(node):
        if isinstance(node, tirx.Call) and _name(node) == _GEMM:
            args = list(node.args)
            args[9] = tirx.And(chunk == 0, args[9])
            changed["mad"] += 1
            return tirx.Call(node.dtype, node.op, args, annotations=node.annotations)
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit":
            body = node.body
            while isinstance(body, tirx.AttrStmt):
                body = body.body
            if isinstance(body, tirx.IfThenElse):
                body = body.then_case
                while isinstance(body, tirx.AttrStmt):
                    body = body.body
            if (isinstance(body, tirx.Evaluate) and isinstance(body.value, tirx.Call)
                    and _name(body.value) == _GEMM):
                return tirx.AttrStmt({"stage": mad_stage}, node.attr_key, node.value, node.body)
            if isinstance(node.body, tirx.For) and node.body.loop_var.same_as(facts["k"]):
                changed["chunk"] += 1
                return unit(tirx.For(chunk, 0, chunks, tirx.ForKind.SERIAL, node,
                                     annotations={"num_stages": 4}), core=None)
        if isinstance(node, tirx.For) and node.loop_var.same_as(facts["k"]):
            annotations = {"multi_buffer_eligible": [a.data, b.data], "num_stages": 4}
            if rejection == "offset-disabled":
                annotations["enable_offset"] = False
            elif rejection == "unconsumed-auto":
                annotations["tl.ascend_auto_pipeline"] = True
            elif rejection == "stage-out-of-bound":
                annotations["num_stages"] = mad_stage
            elif rejection == "non-integer-bound":
                annotations["num_stages"] = tirx.FloatImm("float32", 4.0)
            elif rejection == "dynamic-bound":
                annotations["num_stages"] = facts["epoch"] + 4
            elif boxed_stage_bound:
                annotations["num_stages"] = tirx.IntImm("int32", 4)
            children = list(node.body.seq)
            # Actual AutoSchedule chooses the prior-epoch MAD before current
            # loads. The fill-only first physical tick must issue no MAD.
            children = [children[2], children[1], children[0]]
            if rejection == "multiple-MAD-stages":
                mad = calls(children[0], _GEMM)[0]
                args = list(mad.args)
                args[9] = tirx.const(False, "bool")
                extra = tirx.Evaluate(tirx.Call(mad.dtype, mad.op, args, annotations=mad.annotations))
                children.insert(1, unit(extra, core=2, stage=mad_stage + 1))
            return tirx.For(node.loop_var, node.min, native_k, node.kind,
                            seq(*children), annotations=annotations)

    body = tirx.stmt_functor.ir_transform(root.body, None, mutate)
    assert changed == {"mad": 1, "chunk": 1}
    before = _replace_root(before, body, annotations={
        **dict(root.annotations), "tl.buffer_versions_map": {a.data: versions, b.data: versions},
    })
    facts.update(chunk=chunk, chunks=chunks, a=a, b=b, mad_stage=mad_stage)
    return before, facts


def _lower(before):
    with before["main"].attrs["target"]:
        prepared = transform.PrepareMultiBuffer()(before)
        synced = _insert(prepared)
        materialized = transform.MaterializeMultiBuffer()(synced)
        after = transform.LowerScheduledTIR()(materialized)
    return prepared, synced, materialized, after


def _physical_loop_facts(module, facts, keys):
    # LowerScheduledTIR freshens bound Vars. The fixture deliberately gives
    # each loop a unique name; resolve the actual physical clocks by that
    # identity, while retaining storage identities from the original IR.
    loops = {}

    def visit(node):
        if isinstance(node, tirx.For):
            name = node.loop_var.name
            assert name not in loops, f"Ambiguous fixture loop name: {name}"
            loops[name] = node.loop_var

    tirx.stmt_functor.post_order_visit(module["main"].body, visit)
    rebound = dict(facts)
    for key in keys:
        name = facts[key].name
        assert name in loops, f"Missing lowered fixture loop: {name}"
        rebound[key] = loops[name]
    return rebound


def _single_trip_dma_parent(*, period=None):
    # This independent DMA fixture gives the child exactly one ready event.
    # The actual physical slot, not the declared three-slot backing, fixes the
    # parent WAR distance. Odd parent trips exercise ancestor clock restart.
    outers, chunks, declared_slots = 3, 3, 3
    outer, chunk, child = [tirx.Var(name, "int32") for name in
                           ("dma_outer", "dma_chunk", "dma_child")]
    source = tirx.decl_buffer((outers, chunks, 16, 16), "float16", name="Input")
    consumed = tirx.decl_buffer(source.shape, "float16", name="Consumed", scope="shared.l0a")
    storage = tirx.decl_buffer((declared_slots if period else outers * chunks, 16, 16),
                              "float16", name="storage", scope="shared.l1")
    flat = outer * chunks + chunk
    slot = flat % period if period else flat
    write = copy(source, storage, src_indices=[outer, chunk, 0, 0], src_shape=[1, 1, 16, 16],
                 dst_indices=[slot, 0, 0], dst_shape=[1, 16, 16])
    read = copy(storage, consumed, src_indices=[slot, 0, 0], src_shape=[1, 16, 16],
                dst_indices=[outer, chunk, 0, 0], dst_shape=[1, 1, 16, 16])
    body = unit(tirx.For(child, 0, 1, tirx.ForKind.SERIAL,
                         seq(unit(read, core=2, stage=1), unit(write, core=2)),
                         annotations={"num_stages": 4}), core=None)
    body = unit(tirx.For(chunk, 0, chunks, tirx.ForKind.SERIAL, body), core=None)
    body = unit(tirx.For(outer, 0, outers, tirx.ForKind.SERIAL, body), core=None)
    annotations = {"tl.manual_multi_buffer": {storage.data: declared_slots}} if period else {}
    before = kernel(body, buffers=[storage, consumed], params=[source], annotations=annotations)
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return before


@pytest.mark.parametrize("period", [1, 2])
def test_single_trip_ready_uses_real_parent_completion_and_flattened_clock(period):
    from testing.ascend.auto_schedule.test_sync_binary_flags import (
        _completion_trace, _lower_protocol, _prove_binary_protocol, _prove_data_epochs,
    )

    before = _single_trip_dma_parent(period=period)
    original = tvm.ir.save_json(before)
    trace = _completion_trace(_lower_protocol(before))
    assert tvm.ir.save_json(before) == original
    _prove_data_epochs(trace, _prove_binary_protocol(trace),
                       [(outer, chunk) for outer in range(3) for chunk in range(3)])
    ready_ids = [node.flag[1] for node in trace
                 if node.kind == "set" and node.flag[0] == "MTE2_MTE1"]
    assert len(ready_ids) == 9
    assert len(set(ready_ids)) == period, "Ready width must follow real reverse completion, not backing slots"
    assert ready_ids == [ready_ids[flat % period] for flat in range(9)]


def test_single_trip_child_cannot_skip_its_missing_parent_completion():
    before = transform.PrepareMultiBuffer()(_single_trip_dma_parent())
    original = tvm.ir.save_json(before)
    with pytest.raises(tvm.error.InternalError,
                       match="no parent reverse completion path for single-iteration MTE2->MTE1"):
        _insert(before)
    assert tvm.ir.save_json(before) == original


@pytest.mark.parametrize("native_k,mad_stage", [(1, 1), (2, 1), (3, 1), (2, 2)])
@pytest.mark.parametrize("versions", [1, 2])
def test_scheduled_single_MAD_chain_has_exact_shifted_clear_and_terminal(native_k, mad_stage, versions):
    before, facts = _pipeline(native_k=native_k, mad_stage=mad_stage, versions=versions)
    original = tvm.ir.save_json(before)
    _, synced, materialized, after = _lower(before)
    assert tvm.ir.save_json(before) == original
    assert calls(synced, _DIRECTION), "The admissible scheduled stream fell back"
    versions_map = synced["main"].body.block.annotations["tl.buffer_versions_map"]
    assert int(versions_map[facts["a"].data]) == int(versions_map[facts["b"].data]) == versions
    physical_root = materialized["main"].body.block
    physical_versions = physical_root.annotations.get("tl.manual_multi_buffer", {})
    for buffer in (facts["a"], facts["b"]):
        assert int(physical_versions.get(buffer.data, 1)) == versions
        allocation = next(item for item in physical_root.alloc_buffers if item.data.same_as(buffer.data))
        expected = (versions, 32, 16) if versions > 1 else (32, 16)
        assert tuple(int(extent) for extent in allocation.shape) == expected
    c = next(item for item in physical_root.alloc_buffers if item.data.same_as(facts["c"].data))
    assert tuple(int(extent) for extent in c.shape) == (32, 64)
    facts = _physical_loop_facts(after, facts, ("epoch", "owner", "result", "chunk", "k"))
    records = _trace(after)
    transactions, activations = {}, {}
    for index, (call, env, _) in enumerate(records):
        if facts["epoch"] in env and facts["owner"] in env:
            activation = (_value(facts["epoch"], env), _value(facts["owner"], env))
            activations.setdefault(activation, []).append(index)
        if _name(call) == _GEMM or _is_export(call, facts):
            key = (*activation, _value(facts["result"], env))
            transactions.setdefault(key, []).append((index, call, env))
    assert set(transactions) == {(e, owner, result) for e in range(3) for owner in range(2) for result in range(4)}
    for transaction in transactions.values():
        mads, export = transaction[:-1], transaction[-1]
        assert len(mads) == facts["chunks"] * native_k
        assert [(_value(facts["chunk"], env), _value(facts["k"], env)) for _, _, env in mads] == [
            (chunk, k + mad_stage) for chunk in range(facts["chunks"]) for k in range(native_k)]
        assert [_value(call.args[9], env) for _, call, env in mads] == [1] + [0] * (len(mads) - 1)
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env in mads] == [2] * (len(mads) - 1) + [3]
        _, call, env = export
        assert _is_export(call, facts) and _value(call.annotations["unit_flag_ctrl"], env) == 3
    _check_brackets(records, facts, activations)


def test_boxed_pipeline_bound_retains_the_same_selected_stream():
    ordinary, ordinary_facts = _pipeline()
    boxed, boxed_facts = _pipeline(boxed_stage_bound=True)
    _, ordinary_synced, _, ordinary_after = _lower(ordinary)
    _, boxed_synced, _, boxed_after = _lower(boxed)
    assert calls(ordinary_synced, _DIRECTION) and calls(boxed_synced, _DIRECTION)
    # Both public metadata representations describe the same logical chain.
    # Compare the consumed native trace, independently of the annotation map.
    def math_trace(module, facts):
        return [(_name(call), _value(call.annotations["unit_flag_ctrl"], env),
                 _value(call.args[9], env) if _name(call) == _GEMM else None)
                for call, env, _ in _trace(module)
                if _name(call) == _GEMM or _is_export(call, facts)]
    assert math_trace(ordinary_after, ordinary_facts) == math_trace(boxed_after, boxed_facts)


def _check_brackets(records, facts, activations):
    _check_direction_boundaries(records, facts, activations)
    for event in ("MTE1_M", "M_MTE1"):
        assert any(_name(call) == _SET and call.args[0].value == event for call, _, _ in records), event
        assert any(_name(call) == _WAIT and call.args[0].value == event for call, _, _ in records), event
    # Ordinary flags keep their own finite counts; this is not a native unit
    # completion model. Full native safety remains a separate acceptance gate.
    totals = [Counter((call.args[0].value, _value(call.args[1], env))
                      for call, env, _ in records if _name(call) == operation) for operation in (_SET, _WAIT)]
    assert totals[0] == totals[1]


def test_guarded_scheduled_result_drains_before_restoring_direction():
    before, facts = _pipeline(guarded=True)
    _, _, _, after = _lower(before)
    facts = _physical_loop_facts(after, facts, ("epoch", "result", "chunk", "k"))
    records = _trace(after)
    active = {epoch: [i for i, (_, env, _) in enumerate(records)
                      if facts["epoch"] in env and _value(facts["epoch"], env) == epoch]
              for epoch in range(4)}
    assert {epoch for epoch, positions in active.items()
            if any(_name(records[i][0]) == _GEMM for i in positions)} == {1, 2}
    _check_brackets(records, facts, {epoch: active[epoch] for epoch in (1, 2)})


@pytest.mark.parametrize("rejection", ["offset-disabled", "unconsumed-auto", "stage-out-of-bound",
                                       "multiple-MAD-stages", "missing-terminal",
                                       "non-integer-bound", "dynamic-bound"])
def test_unproved_scheduled_chain_retains_ordinary_protocol(rejection):
    before, facts = _pipeline(rejection=rejection)
    with before["main"].attrs["target"]:
        prepared = transform.PrepareMultiBuffer()(before)
        after = _insert(prepared)
    assert not calls(after, _DIRECTION)
    for call in calls(after, _GEMM):
        assert "unit_flag_ctrl" not in call.annotations
    assert any(call.args[0].value == "M_FIX" for call in calls(after, _SET))


def _shifted_control_pipeline(*, control_stage=1, mad_stage=1, native_k=2,
                             versions=2, rejection=None):
    before, facts = _pipeline(native_k=native_k, mad_stage=mad_stage, versions=versions)
    root = before["main"].body.block
    source = next(buffer for buffer in before["main"].buffer_map.values() if buffer.name == "A")
    # This is the ordinary nested P-chunk pattern: one current GM->L1 copy,
    # then a prior-chunk native-K control with its own shifted MAD and drain.
    chunk_l1 = tirx.decl_buffer((32, 16), "float16", name="chunk_l1", scope="shared.l1")
    changed = Counter()

    def mutate(node):
        if (isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call)
                and _name(node.value) == _COPY
                and node.value.args[1].args[0].buffer.data.same_as(facts["a"].data)):
            changed["A-input"] += 1
            return copy(chunk_l1, facts["a"])
        if isinstance(node, tirx.For) and node.loop_var.same_as(facts["k"]):
            extent, body = node.extent, node.body
            if rejection == "dynamic-descendant":
                extent = facts["epoch"] + 1
            elif rejection == "breaking-descendant":
                body = seq(*list(body.seq), unit(tirx.Evaluate(tirx.Call(
                    "handle", tvm.ir.Op.get("tl.loop_break"), [])), core=2))
            elif rejection == "multiple-MADs-with-shifted-control":
                mad = calls(body, _GEMM)[0]
                args = list(mad.args)
                args[9] = tirx.const(False, "bool")
                extra = tirx.Evaluate(tirx.Call(mad.dtype, mad.op, args, annotations=mad.annotations))
                body = seq(*list(body.seq), unit(extra, core=2, stage=mad_stage))
            return tirx.For(node.loop_var, node.min, extent, node.kind,
                            body, annotations=node.annotations)
        if (isinstance(node, tirx.AttrStmt) and node.attr_key == "tl.schedule_unit"
                and isinstance(node.body, tirx.For)):
            if node.body.loop_var.same_as(facts["k"]):
                changed["native-control"] += 1
                return unit(node.body, core=None, stage=control_stage,
                            guard=facts["chunk"] == 0 if rejection == "guarded-descendant" else None)
            if node.body.loop_var.same_as(facts["chunk"]) and rejection == "direct-child-shift":
                return unit(node.body, core=None, stage=1)
            if node.body.loop_var.same_as(facts["result"]) and rejection == "result-shift":
                return unit(node.body, core=None, stage=1)
        if isinstance(node, tirx.For) and node.loop_var.same_as(facts["chunk"]):
            changed["chunk"] += 1
            annotations = {"multi_buffer_eligible": [chunk_l1.data], "num_stages": 4}
            if rejection == "parent-bound-too-small":
                annotations["num_stages"] = control_stage
            elif rejection == "missing-parent-bound":
                del annotations["num_stages"]
            elif rejection == "parent-offset-disabled":
                annotations["enable_offset"] = False
            elif rejection == "parent-unconsumed-auto":
                annotations["tl.ascend_auto_pipeline"] = True
            fill = unit(copy(source, chunk_l1,
                             src_indices=[facts["epoch"], facts["owner"], 0, 0],
                             src_shape=[1, 1, 32, 16]), core=2)
            return tirx.For(node.loop_var, node.min, node.extent, node.kind,
                            seq(fill, node.body), annotations=annotations)

    body = tirx.stmt_functor.ir_transform(root.body, None, mutate)
    assert changed == {"A-input": 1, "native-control": 1, "chunk": 1}
    annotations = dict(root.annotations)
    annotations["layout_map"] = {**dict(annotations["layout_map"]),
                                 chunk_l1: make_ascend_nz_layout(chunk_l1)}
    annotations["tl.buffer_versions_map"] = {**dict(annotations["tl.buffer_versions_map"]),
                                            chunk_l1.data: 2}
    before = _replace_root(before, body, allocations=[*root.alloc_buffers, chunk_l1],
                           annotations=annotations)
    facts.update(chunk_l1=chunk_l1, control_stage=control_stage)
    return before, facts


@pytest.mark.parametrize("control_stage,mad_stage,versions", [(1, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 2)])
def test_shifted_native_control_keeps_logical_clear_terminal_and_complete_bracket(control_stage, mad_stage, versions):
    before, facts = _shifted_control_pipeline(
        control_stage=control_stage, mad_stage=mad_stage, versions=versions)
    original = tvm.ir.save_json(before)
    _, synced, materialized, after = _lower(before)
    assert tvm.ir.save_json(before) == original
    assert calls(synced, _DIRECTION), "The admissible nested stream fell back"
    physical = materialized["main"].body.block
    for storage, count in ((facts["chunk_l1"], 2), (facts["a"], versions), (facts["b"], versions)):
        buffer = next(buffer for buffer in physical.alloc_buffers if buffer.data.same_as(storage.data))
        assert tuple(int(extent) for extent in buffer.shape) == ((count, 32, 16) if count > 1 else (32, 16))
    c = next(buffer for buffer in physical.alloc_buffers if buffer.data.same_as(facts["c"].data))
    assert tuple(int(extent) for extent in c.shape) == (32, 64)
    facts = _physical_loop_facts(after, facts, ("epoch", "owner", "result", "chunk", "k"))
    records = _trace(after)
    transactions, activations = {}, {}
    for index, (call, env, _) in enumerate(records):
        if facts["epoch"] in env and facts["owner"] in env:
            activation = (_value(facts["epoch"], env), _value(facts["owner"], env))
            activations.setdefault(activation, []).append(index)
        if _name(call) == _GEMM or _is_export(call, facts):
            key = (*activation, _value(facts["result"], env))
            transactions.setdefault(key, []).append((index, call, env))
    assert set(transactions) == {(e, owner, result) for e in range(3) for owner in range(2) for result in range(4)}
    for transaction in transactions.values():
        mads, export = transaction[:-1], transaction[-1]
        assert len(mads) == 4
        assert [(_value(facts["chunk"], env), _value(facts["k"], env)) for _, _, env in mads] == [
            (chunk + control_stage, k + mad_stage) for chunk in range(2) for k in range(2)]
        assert [_value(call.args[9], env) for _, call, env in mads] == [1, 0, 0, 0]
        assert [_value(call.annotations["unit_flag_ctrl"], env) for _, call, env in mads] == [2, 2, 2, 3]
        assert _is_export(export[1], facts) and _value(export[1].annotations["unit_flag_ctrl"], export[2]) == 3
    fills = [(call, env) for call, env, _ in records if _name(call) == _COPY
             and call.args[1].args[0].buffer.data.same_as(facts["chunk_l1"].data)]
    reads = [(call, env) for call, env, _ in records if _name(call) == _COPY
             and call.args[0].args[0].buffer.data.same_as(facts["chunk_l1"].data)]
    assert len(fills) == 48 and len(reads) == 96
    assert all(_value(call.args[1].args[0].indices[0], env) == _value(facts["chunk"], env) % 2
               for call, env in fills)
    assert all(_value(call.args[0].args[0].indices[0], env) == (_value(facts["chunk"], env) - control_stage) % 2
               for call, env in reads)
    _check_brackets(records, facts, activations)


@pytest.mark.parametrize("rejection", ["parent-bound-too-small", "missing-parent-bound", "parent-offset-disabled",
                                       "parent-unconsumed-auto", "guarded-descendant", "dynamic-descendant",
                                       "breaking-descendant", "direct-child-shift", "result-shift",
                                       "multiple-MADs-with-shifted-control"])
def test_unproved_nested_control_keeps_ordinary_protocol(rejection):
    # Zero-stage MADs isolate the whole-Control single-task gate from the
    # previously covered leaf-stage restriction.
    before, _ = _shifted_control_pipeline(mad_stage=0 if rejection == "multiple-MADs-with-shifted-control" else 1,
                                         rejection=rejection)
    original = tvm.ir.save_json(before)
    prepared = transform.PrepareMultiBuffer()(before)
    unsupported_scope = {"guarded-descendant": "result", "breaking-descendant": "chunk"}.get(rejection)
    if unsupported_scope is not None:
        # Skipping or breaking the native-K activation also prevents the
        # ordinary parent's projected binary channel from proving its
        # supported clock. Unit selection cannot bypass that diagnostic.
        with pytest.raises(tvm.error.InternalError, match=(
                "shifted carried events require an unconditional outermost lexical loop"
                f".*pipe=MTE1->M.*analysis_loop={unsupported_scope}")):
            _insert(prepared)
        assert tvm.ir.save_json(before) == original
        return
    after = _insert(prepared)
    assert tvm.ir.save_json(before) == original
    assert not calls(after, _DIRECTION)
    assert all("unit_flag_ctrl" not in call.annotations for call in calls(after, _GEMM))
    assert any(call.args[0].value == "M_FIX" for call in calls(after, _SET))


def test_single_trip_shifted_control_retains_its_unproved_operand_lifetime_diagnostic():
    # Selection cannot make the unrelated MTE1->M flag binary-safe. The
    # existing single-trip parent proof excludes a shifted ancestor; keep
    # that diagnostic rather than using unit C protection as whole-MAD
    # completion or silently accepting an ordinary fallback.
    before, _ = _shifted_control_pipeline(native_k=1)
    original = tvm.ir.save_json(before)
    prepared = transform.PrepareMultiBuffer()(before)
    with pytest.raises(tvm.error.InternalError, match="no reverse completion path for MTE1->M"):
        _insert(prepared)
    assert tvm.ir.save_json(before) == original


def _shifted_m64_inputs(*, dtype="float16", load_stage=1, mad_stage=2, rejection=None):
    # Two K256 chunks keep the finite trace small while retaining the actual
    # M64/N64 native operands, private C pitch64, and P0/load1/MAD2 offsets.
    from tilelang.ascend import language as T
    from tilelang.layout import (
        make_ascend_l0a_2201_layout, make_ascend_l0c_layout,
        make_ascend_major_mn_layout,
    )

    epoch, owner, result, chunk = [tirx.Var(name, "int32") for name in
                                  ("m64_epoch", "m64_owner", "m64_result", "m64_chunk")]
    epochs, owners, results, chunks = 2, 2, 2, 2
    p_gm = tirx.decl_buffer((epochs, owners, results, chunks, 64, 256), dtype, name="P")
    v_gm = tirx.decl_buffer((epochs, chunks * 256, 64), dtype, name="V")
    output = tirx.decl_buffer((epochs, owners, results, 64, 64), "float32", name="M64Output")
    p = tirx.decl_buffer((64, 256), dtype, name="m64_p", scope="shared.l1")
    v = tirx.decl_buffer((chunks * 256, 64), dtype, name="m64_v", scope="shared.l1")
    a = tirx.decl_buffer((64, 256), dtype, name="m64_a", scope="shared.l0a")
    b = tirx.decl_buffer((256, 64), dtype, name="m64_b", scope="shared.l0b")
    c = tirx.decl_buffer((64, 64), "float32", name="m64_c", scope="shared.l0c")
    feedback = tirx.decl_buffer((64, 64), "float32", name="m64_feedback", scope="shared.l1")
    fill = unit(copy(p_gm, p, src_indices=[epoch, owner, result, chunk, 0, 0],
                     src_shape=[1, 1, 1, 1, 64, 256]), core=2)
    load_a = unit(copy(p, a), core=2, stage=load_stage,
                  guard=chunk < chunks if rejection == "own-DMA-guard" else None)
    load_b = unit(copy(v, b, src_indices=[chunk * 256, 0], src_shape=[256, 64]),
                  core=2, stage=load_stage)
    mad = unit(tirx.Evaluate(T.gemm(a, b, c, clear_accum=chunk == 0)), core=2, stage=mad_stage)
    annotations = {"num_stages": 3, "multi_buffer_eligible": [p.data, a.data, b.data]}
    if rejection == "invalid-parent-bound":
        annotations["num_stages"] = 1
    elif rejection == "parent-offset-disabled":
        annotations["enable_offset"] = False
    elif rejection == "parent-unconsumed-auto":
        annotations["tl.ascend_auto_pipeline"] = True
    children = [mad, load_b, load_a, fill]
    direct = []
    extra_allocations, extra_layouts = [], {}
    if rejection == "direct-result-DMA":
        direct = [unit(copy(v_gm, v, src_indices=[epoch, 0, 0],
                            src_shape=[1, chunks * 256, 64]), core=2, stage=1)]
    if rejection == "sibling-shift-multiple-MADs":
        # A shifted DMA-only sibling must impose the single-MAD-task limit,
        # even when the two real accumulating MAD tasks themselves stay at0.
        extra = unit(tirx.Evaluate(T.gemm(a, b, c, clear_accum=False)), core=2)
        sibling = tirx.Var("m64_dma_sibling", "int32")
        # This sibling has its own single-slot input. It must not access P/A
        # outside their annotated chunk owner, which Prepare rightly rejects.
        sibling_input = tirx.decl_buffer((64, 256), dtype, name="m64_sibling_input", scope="shared.l1")
        extra_allocations = [sibling_input]
        extra_layouts = {sibling_input: make_ascend_nz_layout(sibling_input)}
        sibling_copy = copy(p_gm, sibling_input, src_indices=[epoch, owner, result, 0, 0, 0],
                            src_shape=[1, 1, 1, 1, 64, 256])
        dma = unit(tirx.For(sibling, 0, 2, tirx.ForKind.SERIAL,
                            seq(unit(sibling_copy, core=2), unit(sibling_copy, core=2, stage=1)),
                            annotations={"num_stages": 2}), core=None)
        children = [fill, load_b, load_a, mad, extra]
        direct = [dma]
    body = unit(tirx.For(chunk, 0, chunks, tirx.ForKind.SERIAL, seq(*children),
                         annotations=annotations), core=None)
    export = unit(copy(c, output, dst_indices=[epoch, owner, result, 0, 0],
                       dst_shape=[1, 1, 1, 64, 64],
                       annotations={"tl.ascend.root_fixpipe_config": tirx.IntImm("int32", 1)}), core=2)
    body = unit(tirx.For(result, 0, results, tirx.ForKind.SERIAL,
                         seq(*direct, body, export)), core=None)
    body = unit(tirx.For(owner, 0, owners, tirx.ForKind.SERIAL, body), core=None)
    v_fill = unit(copy(v_gm, v, src_indices=[epoch, 0, 0], src_shape=[1, chunks * 256, 64]), core=2)
    body = unit(tirx.For(epoch, 0, epochs, tirx.ForKind.SERIAL, seq(v_fill, body)), core=None)
    read_epoch, read_owner, read_result = [tirx.Var(name, "int32") for name in
                                         ("m64_read_epoch", "m64_read_owner", "m64_read_result")]
    readback = unit(copy(output, feedback, src_indices=[read_epoch, read_owner, read_result, 0, 0],
                         src_shape=[1, 1, 1, 64, 64]), core=2)
    for var, extent in ((read_result, results), (read_owner, owners), (read_epoch, epochs)):
        readback = unit(tirx.For(var, 0, extent, tirx.ForKind.SERIAL, readback), core=None)
    setup = unit(tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get("tl.ascend_set_fixpipe_config"), [])), core=2)
    layouts = {p: make_ascend_nz_layout(p), v: make_ascend_nz_layout(v),
               a: make_ascend_l0a_2201_layout(a), b: make_ascend_major_mn_layout(b),
               c: make_ascend_l0c_layout(c), feedback: make_ascend_nz_layout(feedback), **extra_layouts}
    before = kernel(seq(setup, body, readback), buffers=[p, v, a, b, c, feedback, *extra_allocations],
                    params=[p_gm, v_gm, output], annotations={"layout_map": layouts,
                    "tl.buffer_versions_map": {p.data: 2, a.data: 2, b.data: 2}})
    before["main"] = before["main"].with_attr(
        "target", tvm.target.Target({"kind": "ascend", "arch": "dav-2201"}))
    return before, {"epoch": epoch, "owner": owner, "result": result, "chunk": chunk,
                    "c": c, "p": p, "a": a, "b": b, "output": output}


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_shifted_m64_input_dma_preserves_physical_offsets_and_closed_stream(dtype):
    before, facts = _shifted_m64_inputs(dtype=dtype)
    original = tvm.ir.save_json(before)
    _, synced, materialized, after = _lower(before)
    assert tvm.ir.save_json(before) == original
    assert calls(synced, _DIRECTION), "The P0/load1/MAD2 stream fell back"
    for storage, shape in ((facts["p"], (2, 64, 256)), (facts["a"], (2, 64, 256)),
                           (facts["b"], (2, 256, 64)), (facts["c"], (64, 64))):
        allocation = next(buffer for buffer in materialized["main"].body.block.alloc_buffers
                          if buffer.data.same_as(storage.data))
        assert tuple(int(extent) for extent in allocation.shape) == shape
    facts = _physical_loop_facts(after, facts, ("epoch", "owner", "result", "chunk"))
    records = _trace(after)
    activations, transactions = {}, {}
    for index, (call, env, _) in enumerate(records):
        if facts["epoch"] in env and facts["owner"] in env:
            activation = (_value(facts["epoch"], env), _value(facts["owner"], env))
            activations.setdefault(activation, []).append(index)
        if _name(call) == _GEMM or _is_export(call, facts):
            key = (*activation, _value(facts["result"], env))
            transactions.setdefault(key, []).append((call, env))
    assert set(transactions) == {(e, owner, part) for e in range(2) for owner in range(2) for part in range(2)}
    for transaction in transactions.values():
        mads, export = transaction[:-1], transaction[-1]
        assert len(mads) == 2
        assert [_value(facts["chunk"], env) for _, env in mads] == [2, 3]
        assert [_value(call.args[9], env) for call, env in mads] == [1, 0]
        assert [_value(call.annotations["unit_flag_ctrl"], env) for call, env in mads] == [2, 3]
        assert [_value(call.args[index], env) for call, env in mads for index in (5, 6, 7)] == [64, 64, 256] * 2
        for call, env in mads:
            for operand in (0, 1):
                assert _value(call.args[operand].args[0].indices[0], env) == (_value(facts["chunk"], env) - 2) % 2
        assert _is_export(export[0], facts) and _value(export[0].annotations["unit_flag_ctrl"], export[1]) == 3
    loads, fills = [], []
    for call, env, _ in records:
        if _name(call) != _COPY:
            continue
        dst = call.args[1].args[0]
        if dst.buffer.data.same_as(facts["a"].data) or dst.buffer.data.same_as(facts["b"].data):
            loads.append((call, env))
            assert _value(facts["chunk"], env) in (1, 2)
            assert _value(dst.indices[0], env) == (_value(facts["chunk"], env) - 1) % 2
            if dst.buffer.data.same_as(facts["a"].data):
                assert _value(call.args[0].args[0].indices[0], env) == (_value(facts["chunk"], env) - 1) % 2
            else:
                assert _value(call.args[0].args[0].indices[0], env) == (_value(facts["chunk"], env) - 1) * 256
        elif dst.buffer.data.same_as(facts["p"].data):
            fills.append((call, env))
            assert _value(facts["chunk"], env) in (0, 1)
            assert _value(dst.indices[0], env) == _value(facts["chunk"], env) % 2
    assert len(loads) == 32 and len(fills) == 16
    _check_brackets(records, facts, activations)


@pytest.mark.parametrize("rejection", ["own-DMA-guard", "direct-result-DMA", "invalid-parent-bound",
                                       "parent-offset-disabled", "parent-unconsumed-auto",
                                       "sibling-shift-multiple-MADs"])
def test_unproved_shifted_dma_cannot_claim_a_unit_stream(rejection):
    # Isolate DMA-stage validation from positive MAD-stage validation.
    before, _ = _shifted_m64_inputs(load_stage=0 if rejection in (
        "direct-result-DMA", "sibling-shift-multiple-MADs") else 1,
        mad_stage=0 if rejection != "own-DMA-guard" else 2, rejection=rejection)
    original = tvm.ir.save_json(before)
    prepared = transform.PrepareMultiBuffer()(before)
    after = _insert(prepared)
    assert tvm.ir.save_json(before) == original
    assert not calls(after, _DIRECTION)
    assert all("unit_flag_ctrl" not in call.annotations for call in calls(after, _GEMM))
    assert any(call.args[0].value == "M_FIX" for call in calls(after, _SET))


def test_guarded_stage_zero_dma_keeps_its_existing_unit_stream():
    before, _ = _shifted_m64_inputs(load_stage=0, mad_stage=1, rejection="own-DMA-guard")
    prepared = transform.PrepareMultiBuffer()(before)
    after = _insert(prepared)
    assert calls(after, _DIRECTION), "The unchanged stage0 guarded-DMA contract fell back"
