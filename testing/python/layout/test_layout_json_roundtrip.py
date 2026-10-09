"""JSON restores layout binders, including inherited and replication maps.

These CPU tests consume restored layouts through the public layout API.  They
do not repair variables by name or rely on structural JSON equality alone.
"""

import itertools
import json

import pytest

from tilelang import _ffi_api, tvm
from tilelang.layout import (
    Fragment,
    Layout,
    PartialFragment,
    make_ascend_l0a_2201_layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_sf_layout,
    try_extract_fractal_layout,
)
from tvm import tirx


def _roundtrips(layout):
    # A second serialization catches a getter that fixes only its returned
    # expression while leaving an inconsistent stored binding graph.
    for _ in range(2):
        layout = tvm.ir.load_json(tvm.ir.save_json(layout))
        yield layout


def _constant(expr):
    value = tvm.arith.Analyzer().simplify(expr)
    assert isinstance(value, tirx.IntImm), f"unbound layout expression: {value}"
    return int(value)


def _forward(layout, indices):
    return tuple(_constant(value) for value in layout.map_forward_index(indices))


def _thread(layout, indices):
    value, = layout.map_forward_thread(indices)
    return value


def _shape(values):
    return tuple(_constant(value) for value in values)


def _assert_equal(actual, expected):
    assert actual.is_equal(expected)
    assert expected.is_equal(actual)
    tvm.ir.assert_structural_equal(actual, expected)
    assert tvm.ir.structural_hash(actual) == tvm.ir.structural_hash(expected)
    assert _shape(actual.get_input_shape()) == _shape(expected.get_input_shape())
    assert _shape(actual.get_output_shape()) == _shape(expected.get_output_shape())


@pytest.mark.parametrize("kind", ["linear", "split", "transpose"])
def test_layout_roundtrip_consumes_restored_axis_bindings(kind):
    if kind == "linear":
        original = Layout([4, 8], lambda i, j: i * 8 + j)
        expected = lambda i, j: (i * 8 + j,)
    elif kind == "split":
        original = Layout([32], lambda i: [i // 8, i % 8])
        expected = lambda i: (i // 8, i % 8)
    else:
        original = Layout([4, 8], lambda i, j: [j, i])
        expected = lambda i, j: (j, i)

    domain = itertools.product(*(range(n) for n in _shape(original.get_input_shape())))
    points = list(domain)
    for restored in _roundtrips(original):
        _assert_equal(restored, original)
        for point in points:
            assert _forward(restored, point) == expected(*point)


def test_inverse_after_json_roundtrip_retains_coordinate_roundtrip():
    original = Layout([32], lambda i: [i // 8, i % 8])
    for restored in _roundtrips(original):
        inverse = restored.inverse()
        _assert_equal(inverse, original.inverse())
        for loaded_inverse in _roundtrips(inverse):
            assert _shape(loaded_inverse.get_input_shape()) == (4, 8)
            assert _shape(loaded_inverse.get_output_shape()) == (32,)
            for i in range(32):
                assert _forward(loaded_inverse, _forward(restored, [i])) == (i,)


@pytest.mark.parametrize("operation", ["expand", "reshape", "repeat"])
def test_transforming_a_restored_layout_rebinds_new_axes(operation):
    original = Layout([4, 8], lambda i, j: i * 8 + j)
    transform = {
        "expand": lambda layout: layout.expand([2]),
        "reshape": lambda layout: layout.reshape([8, 4]),
        "repeat": lambda layout: layout.repeat(dim=1, factor=2),
    }[operation]
    expected = transform(original)
    points = list(itertools.product(*(range(n) for n in _shape(expected.get_input_shape()))))
    for restored in _roundtrips(original):
        transformed = transform(restored)
        for loaded_transform in _roundtrips(transformed):
            _assert_equal(loaded_transform, expected)
            for point in points:
                assert _forward(loaded_transform, point) == _forward(expected, point)


def _replicated_fragment():
    return Fragment(
        [4, 8],
        forward_thread_fn=lambda i, j, rep: (i * 2 + j % 2) * 4 + rep,
        forward_index_fn=lambda i, j: j // 2,
        replicate=4,
    )


def test_fragment_roundtrip_binds_replication_in_thread_map_and_inverse():
    original = _replicated_fragment()
    for restored in _roundtrips(original):
        assert isinstance(restored, Fragment)
        _assert_equal(restored, original)
        assert _constant(restored.get_thread_size()) == 32
        assert _shape(restored.get_output_shape()) == (4,)
        inverse = next(_roundtrips(restored.inverse()))
        assert _shape(inverse.get_input_shape()) == (4, 32)
        assert _shape(inverse.get_output_shape()) == (4, 8, 4)
        for rep, i, j in itertools.product(range(4), range(4), range(8)):
            # Fragment's public mapping API places replication before axes;
            # its inverse returns logical axes followed by replication.
            index = _forward(restored, [rep, i, j])
            thread = _constant(_thread(restored, [rep, i, j]))
            assert index == (j // 2,)
            assert thread == (i * 2 + j % 2) * 4 + rep
            assert _forward(inverse, [*index, thread]) == (i, j, rep)


def test_replicating_a_restored_fragment_rebinds_new_replication_domain():
    original = _replicated_fragment()
    for restored in _roundtrips(original):
        transformed = restored.replicate(2)
        for loaded_transform in _roundtrips(transformed):
            _assert_equal(loaded_transform, original.replicate(2))
            assert _constant(loaded_transform.get_thread_size()) == 64
            for rep in range(8):
                assert _forward(loaded_transform, [rep, 3, 7]) == (3,)
                assert _constant(_thread(loaded_transform, [rep, 3, 7])) == (
                    28 + rep % 4 + 32 * (rep // 4)
                )


def test_partial_fragment_roundtrip_preserves_combine_and_copy_replication():
    original = PartialFragment(
        [8],
        forward_thread_fn=lambda i, rep: (i % 4) * 8 + rep,
        forward_index_fn=lambda i: i // 4,
        replicate=8,
        combine=2,
    )
    for restored in _roundtrips(original):
        assert isinstance(restored, PartialFragment)
        _assert_equal(restored, original)
        assert _constant(restored.combine_size) == 2
        assert _constant(restored.replicate_size) == 8
        assert [tuple(map(int, step)) for step in _ffi_api.PartialFragment_combine_steps(restored)] == [(2, 1)]
        post_collective = next(_roundtrips(restored.as_post_collective()))
        assert isinstance(post_collective, Fragment)
        assert not isinstance(post_collective, PartialFragment)
        _assert_equal(post_collective, original.as_post_collective())
        for rep, i in itertools.product(range(8), range(8)):
            assert _forward(restored, [rep, i]) == (i // 4,)
            thread = _constant(_thread(restored, [rep, i]))
            assert thread == (i % 4) * 8 + rep
            assert _constant(_thread(post_collective, [rep, i])) == thread


def _free_vars(expr):
    variables = []

    def visit(node):
        if isinstance(node, tirx.Var) and not any(node.same_as(var) for var in variables):
            variables.append(node)

    tirx.stmt_functor.post_order_visit(expr, visit)
    return variables


@pytest.mark.parametrize("replication", [False, True])
def test_same_named_external_symbol_remains_free_after_roundtrip(replication):
    if replication:
        probe = Fragment([8], forward_thread_fn=lambda i, rep: i * 4 + rep, replicate=4)
        external = tirx.Var(probe.get_forward_vars()[0].name, "int32")
        original = Fragment(
            [8],
            forward_thread_fn=lambda i, rep: i * 4 + rep + external,
            forward_index_fn=lambda i: i,
            replicate=4,
        )
        consume = lambda layout: _thread(layout, [2, 3])
        expected_constant = 14
    else:
        probe = Layout([8], lambda i: i)
        external = tirx.Var(probe.get_forward_vars()[0].name, "int32")
        original = Layout([8], lambda i: i + external)
        consume = lambda layout: layout.map_forward_index([3])[0]
        expected_constant = 3

    for restored in _roundtrips(original):
        expression = consume(restored)
        free, = _free_vars(expression)
        assert free.name == external.name
        assert _constant(tirx.stmt_functor.substitute(expression, {free: 7})) == expected_constant + 7


@pytest.mark.parametrize("kind", range(6))
def test_ascend_canonical_kind_and_expansion_survive_json_roundtrip(kind):
    shape, dtype = ((32, 8), "uint8") if kind == 2 else ((32, 48), "float32" if kind == 1 else "float16")
    buffer = tirx.decl_buffer(shape, dtype, name="logical", scope="shared")
    if kind in (4, 5):
        original = make_ascend_l0a_2201_layout(buffer, transpose=kind == 5)
    else:
        original = {
            0: make_ascend_major_k_layout,
            1: make_ascend_l0c_layout,
            2: make_ascend_sf_layout,
            3: make_ascend_major_mn_layout,
        }[kind](buffer)

    def signature(layout, logical_buffer):
        info = try_extract_fractal_layout(layout, logical_buffer)
        assert info is not None
        return tuple(_constant(getattr(info, name)) for name in info.__slots__)

    expected_signature = signature(original, buffer)
    assert expected_signature[0] == kind
    points = [(0, 0), (15, shape[1] - 1), (16, 0), (shape[0] - 1, shape[1] - 1)]
    expanded_buffer = tirx.decl_buffer((2, *shape), dtype, name="expanded", scope="shared")
    for restored in _roundtrips(original):
        _assert_equal(restored, original)
        assert signature(restored, buffer) == expected_signature
        for point in points:
            assert _forward(restored, point) == _forward(original, point)
        expanded = next(_roundtrips(restored.expand([2])))
        _assert_equal(expanded, original.expand([2]))
        assert signature(expanded, expanded_buffer) == expected_signature
        for point in points:
            assert _forward(expanded, [1, *point]) == (1, *_forward(original, point))


def test_legacy_json_without_bindings_is_inspectable_but_not_consumable():
    payload = json.loads(tvm.ir.save_json(Layout([4, 8], lambda i, j: i * 8 + j)))
    fields = payload["nodes"][payload["root_index"]]["data"]
    fields.pop("placeholder_vars")
    restored = tvm.ir.load_json(json.dumps(payload))
    assert _shape(restored.get_input_shape()) == (4, 8)
    with pytest.raises(Exception, match="missing serialized placeholder bindings"):
        restored.map_forward_index([1, 2])


if __name__ == "__main__":
    pytest.main([__file__])
