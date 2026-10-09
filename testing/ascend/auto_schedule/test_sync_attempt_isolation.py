"""Failed sync attempts must leave only the accepted kernel protocol.

Reuse prepared public-pass inputs that already exercise a real stream rollback
or owner-event capacity retry. These tests check repeated/serialized inputs and
independent kernel roots, rather than exposing an allocator or retry hook.
The finite event checks below do not model native unit completion.
"""

from collections import Counter

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, nodes, seq
from testing.ascend.auto_schedule.test_owner_input_completion import (
    _assert_owner_input, _completion_graph, _input_flags, _scheduled_input,
)
from testing.ascend.auto_schedule.test_owner_result_completion import (
    _assert_owner_result, _name, _owner_sets, _scheduled_handoff,
    _trace as _owner_trace, _value,
)
from testing.ascend.auto_schedule.test_unit_flag_kernel_direction import _prove_kernel
from testing.ascend.auto_schedule.test_unit_flag_stream import (
    _COPY, _DIRECTION, _GEMM, _SET, _WAIT, _fixture,
    _trace as _unit_trace, _value as _unit_value,
)


def _roots(mod):
    return [block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root"]


def _code_and_storage(root, compare_layouts=True):
    # Keep complete task code, allocations, layouts and alias metadata together.
    # Bind storage and Buffer identities before their annotation map keys.
    # Map comparison cannot establish a previously unseen correspondence.
    buffers = list(root.alloc_buffers)
    for access in nodes(root.body, (tirx.BufferLoad, tirx.BufferStore)):
        if not any(access.buffer.same_as(buffer) for buffer in buffers):
            buffers.append(access.buffer)
    storage = []
    for buffer in buffers:
        if not any(buffer.data.same_as(data) for data in storage):
            storage.append(buffer.data)
    annotations = dict(root.annotations)
    layouts = annotations.get("layout_map")
    if not compare_layouts:
        annotations.pop("layout_map", None)
    elif layouts is not None:
        # Buffer is a structural tree node, so FFI Map comparison cannot map
        # its object identity. Compare every key and value in allocation/use
        # order instead, retaining the complete Buffer and Layout contracts.
        entries = [[buffer, layouts[buffer]] for buffer in buffers if buffer in layouts]
        assert len(entries) == len(layouts)
        annotations["layout_map"] = entries
    code = tirx.SBlock(root.iter_vars, root.reads, root.writes, root.name_hint,
                      root.body, root.init, root.alloc_buffers, root.match_buffers, annotations)
    return [storage, buffers, code]


def _assert_same_kernels(actual, expected, compare_layouts=True):
    left, right = _roots(actual), _roots(expected)
    assert len(left) == len(right)
    for actual_root, expected_root in zip(left, right):
        tvm.ir.assert_structural_equal(_code_and_storage(actual_root, compare_layouts),
                                       _code_and_storage(expected_root, compare_layouts), map_free_vars=True)


def _lane_binding(mod):
    bindings = [attribute for attribute in nodes(mod, tirx.AttrStmt)
                if attribute.attr_key == "thread_extent"
                and isinstance(attribute.node, tirx.IterVar)
                and attribute.node.thread_tag == "cthread"]
    assert len(bindings) <= 1
    return bindings[0] if bindings else None


def _combine(first, second):
    """Two independent roots in one actual lane context, without shared data."""
    functions = [first["main"], second["main"]]
    bindings = [_lane_binding(first), _lane_binding(second)]
    assert (bindings[0] is None) == (bindings[1] is None)
    bodies = []
    for function, binding in zip(functions, bindings):
        body = binding.body if binding is not None else function.body
        if binding is not None and not binding.node.var.same_as(bindings[0].node.var):
            body = tirx.stmt_functor.substitute(body, {binding.node.var: bindings[0].node.var})
        bodies.append(body)
    body = seq(*bodies)
    if bindings[0] is not None:
        binding = bindings[0]
        body = tirx.AttrStmt(binding.node, binding.attr_key, binding.value, body)
    function = tirx.PrimFunc([param for function in functions for param in function.params],
                             body,
                             buffer_map={param: buffer for function in functions
                                         for param, buffer in function.buffer_map.items()},
                             attrs=functions[0].attrs)
    return tvm.IRModule({"main": function})


def _kernel_at(mod, index):
    root = _roots(mod)[index]
    body = tirx.SBlockRealize([], True, root)
    binding = _lane_binding(mod)
    if binding is not None:
        body = tirx.AttrStmt(binding.node, binding.attr_key, binding.value, body)
    return tvm.IRModule({"main": mod["main"].with_body(body)})


def _insert(pass_object, before):
    original = tvm.ir.save_json(before)
    with before["main"].attrs["target"]:
        after = pass_object(before)
    assert tvm.ir.save_json(before) == original
    return after


def _exercise(failed, selected, check_failed, check_selected):
    # Reuse the same public pass object: no attempt state may survive a call.
    insert = transform.InsertSync()
    fallback = _insert(insert, failed)
    accepted = _insert(insert, selected)
    check_failed(fallback)
    check_selected(accepted)

    for before, expected in ((failed, fallback), (selected, accepted)):
        _assert_same_kernels(_insert(insert, before), expected)
        restored = tvm.ir.load_json(tvm.ir.save_json(before))
        _assert_same_kernels(_insert(insert, restored), expected)

    # A failed root before or after an independent selected root must neither
    # consume its flags nor leave annotations/insertions in the other's tree.
    for inputs, expected in (((failed, selected), (fallback, accepted)),
                             ((selected, failed), (accepted, fallback))):
        combined = _combine(*inputs)
        for before in (combined, tvm.ir.load_json(tvm.ir.save_json(combined))):
            after = _insert(insert, before)
            assert len(_roots(after)) == 2
            for index, standalone in enumerate(expected):
                _assert_same_kernels(_kernel_at(after, index), standalone)
    return fallback


def _assert_ordinary_collective_balance(mod):
    sets, waits = {}, {}
    for core, lane in ((2, 0), (1, 0), (1, 1)):
        records = list(_owner_trace(mod, core=core, lane=lane))
        for operation, destination in (("tl.ascend_cross_core_set_flag", sets),
                                       ("tl.ascend_cross_core_wait_flag", waits)):
            destination[core, lane] = Counter(
                _value(call.args[2], env) for call, env, _ in records
                if _name(call) == operation)
        # Local banks are independent for each physical participant.
        local = [Counter((call.args[0].value, _value(call.args[1], env))
                         for call, env, _ in records if _name(call) == operation)
                 for operation in (_SET, _WAIT)]
        assert local[0] == local[1]
    assert sets[2, 0] and waits[2, 0]
    for lane in range(2):
        assert sets[2, 0] == waits[1, lane]
        assert waits[2, 0] == sets[1, lane]


def test_unit_stream_root_retry_is_repeatable_and_kernel_local():
    failed, _ = _fixture(repeated_readback=True)
    ordinary, _ = _fixture(kind="missing-layout", repeated_readback=True)
    selected, facts = _fixture()

    def check_failed(after):
        assert not calls(after, _DIRECTION)
        for call in calls(after, _GEMM) + calls(after, _COPY):
            control = call.annotations.get("unit_flag_ctrl")
            assert control is None or _unit_value(control, {}) == 0
        records = _unit_trace(after)
        totals = [Counter((call.args[0].value, _unit_value(call.args[1], env))
                          for call, env, _ in records if _name(call) == operation)
                  for operation in (_SET, _WAIT)]
        assert totals[0] and totals[0] == totals[1]
        assert len(calls(after, "tl.ascend_set_fixpipe_config")) == 1

    fallback = _exercise(failed, selected, check_failed,
                         lambda after: _prove_kernel(_unit_trace(after), facts))
    # The completion-proof rollback starts from the pre-unit snapshot. It must
    # produce the ordinary protocol, without duplicate direction boundaries.
    # This control deliberately omits the accumulator layout to deny admission;
    # only this comparison excludes that intentional metadata difference. All
    # repeated/serialized and independent-root checks above retain layouts.
    _assert_same_kernels(fallback, _insert(transform.InsertSync(), ordinary), compare_layouts=False)


def test_owner_result_capacity_retry_is_repeatable_and_kernel_local():
    failed = _scheduled_handoff(reserved_flags=12)
    selected = _scheduled_handoff()

    def check_failed(after):
        assert not _owner_sets(after)
        events = calls(after, "tl.ascend_cross_core_set_flag")
        events += calls(after, "tl.ascend_cross_core_wait_flag")
        assert events and {int(call.args[0]) for call in events} == {2}
        _assert_ordinary_collective_balance(after)

    _exercise(failed, selected, check_failed,
              lambda after: _assert_owner_result(after, rows=32))


def test_owner_input_capacity_retry_is_repeatable_and_kernel_local():
    failed = _scheduled_input(reserved_flags=12)
    selected = _scheduled_input()

    def check_failed(after):
        assert not _input_flags(after)
        _assert_ordinary_collective_balance(after)
        # Reuse the existing completion graph for finite ordinary flags and
        # both physical mode2 participants, including the final root drain.
        trace, _ = _completion_graph(after)
        assert any(command.kind == "read" for command in trace)
        assert any(command.kind == "write" for command in trace)

    _exercise(failed, selected, check_failed,
              lambda after: _assert_owner_input(after, epochs=3))
