"""Native2201 UB dependencies require completion, not just V issue order."""

import pytest
import tilelang
import tilelang.ascend.language as T

from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _pointer(buffer, effect, *, offset=0, count=17):
    return tirx.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [buffer[offset], count, effect])


def _exp(source, destination, *, source_offset=0, destination_offset=0, count=17):
    return tirx.Evaluate(
        tirx.Call(
            "void",
            tvm.ir.Op.get("tl.ascend_membase.exp"),
            [_pointer(destination, 2, offset=destination_offset, count=count), _pointer(source, 1, offset=source_offset, count=count), tirx.Cast("uint32", count)],
        )
    )


def _program(body, buffers, *, arch="dav-2201", params=()):
    mod = kernel(body, buffers=buffers, params=params)
    mod["main"] = mod["main"].with_attr("target", tvm.target.Target({"kind": "ascend", "arch": arch}))
    return mod


def _barriers(mod):
    return [call for call in calls(mod, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_V"]


def _insert_sync(mod, disable_reuse=True):
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        return transform.InsertSync()(mod)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("hazard", ["RAW", "WAR", "WAW"])
def test_dependent_native_vector_accesses_wait_for_completion(dtype, hazard):
    a, b, middle, output = [tirx.decl_buffer((32,), dtype, name=name, scope="shared.dyn") for name in ("a", "b", "middle", "output")]
    if hazard == "RAW":
        producer, consumer = _exp(a, middle), _exp(middle, output)
    elif hazard == "WAR":
        producer, consumer = _exp(middle, output), _exp(b, middle)
    else:
        producer, consumer = _exp(a, middle), _exp(b, middle)
    after = _insert_sync(_program(seq(unit(producer), unit(consumer)), [a, b, middle, output]))
    (barrier,) = _barriers(after)
    ordered = list(statements(after))
    operations = calls(after, "tl.ascend_membase.exp")
    positions = [next(index for index, (stmt, _) in enumerate(ordered) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(call)) for call in [operations[0], barrier, operations[1]]]
    assert positions == sorted(positions)
    assert not calls(after, "tl.ascend_set_flag") and not calls(after, "tl.ascend_wait_flag")


@pytest.mark.parametrize("shared_storage", [False, True], ids=["separate-storage", "disjoint-subregions"])
def test_independent_native_vector_accesses_do_not_serialize(shared_storage):
    source = tirx.decl_buffer((64,), "float32", name="source", scope="shared.dyn")
    first = tirx.decl_buffer((64,), "float32", name="first", scope="shared.dyn")
    second = first if shared_storage else tirx.decl_buffer((64,), "float32", name="second", scope="shared.dyn")
    operations = [unit(_exp(source, first, count=16)), unit(_exp(source, second, destination_offset=16 if shared_storage else 0, count=16))]
    after = _insert_sync(_program(seq(*operations), [source, first] + ([] if shared_storage else [second])))
    assert not _barriers(after)


def test_existing_3510_vf_order_is_not_inferred_from_2201_contract():
    source, middle, output = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("source", "middle", "output")]

    def vf(src, dst):
        return tirx.SBlock([], [], [], "SIMD_VF", tirx.BufferStore(dst, src[0] + tirx.const(1, "float32"), [0]))

    after = _insert_sync(_program(seq(unit(vf(source, middle)), unit(vf(middle, output))), [source, middle, output], arch="dav-3510"))
    assert not _barriers(after)


def test_loop_carried_native_vector_state_completes_each_iteration():
    state = tirx.decl_buffer((32,), "float32", name="state", scope="shared.dyn")
    epoch = tirx.Var("epoch", "int32")
    loop = tirx.For(epoch, 0, 4, tirx.ForKind.SERIAL, unit(_exp(state, state)))
    after = _insert_sync(_program(unit(loop, core=None), [state]))
    (barrier,) = _barriers(after)
    (_, parents) = next((stmt, parents) for stmt, parents in statements(after) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(barrier))
    assert any(isinstance(parent, tirx.For) and parent.loop_var.same_as(epoch) for parent in parents)
    assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)


def test_dynamic_count_and_empty_task_guard_preserve_completion_boundary():
    source, middle, output = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("source", "middle", "output")]
    count = tirx.Var("count", "int32")
    active = count > 0
    # Empty data work is internal to each atomic task. The synchronization
    # boundary is unconditional, just like the producer/consumer task marker.
    operations = [unit(tirx.IfThenElse(active, operation, None)) for operation in (_exp(source, middle, count=count), _exp(middle, output, count=count))]
    after = _insert_sync(_program(seq(*operations), [source, middle, output], params=[count]))
    (barrier,) = _barriers(after)
    (_, parents) = next((stmt, parents) for stmt, parents in statements(after) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(barrier))
    assert not any(isinstance(parent, tirx.IfThenElse) for parent in parents)


def test_vector_storage_alias_requires_a_real_completion_path():
    source, first, second = [tirx.decl_buffer((32,), "float32", name=name, scope="shared.dyn") for name in ("source", "first", "second")]
    # Different destinations share no dependency. V issue order cannot prove
    # that first's write has completed before second reuses its allocation.
    before = _program(seq(unit(_exp(source, first)), unit(_exp(source, second))), [source, first, second])
    after = _insert_sync(before, disable_reuse=False)
    root = after["main"].body.block
    aliases = root.annotations.get("tl.buffer_alias_map", {})
    assert not any(other.same_as(second.data) for other in aliases.get(first.data, []))
    assert not any(other.same_as(first.data) for other in aliases.get(second.data, []))
    assert not _barriers(after)


def _lower_native(program):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(
            program,
            target={"kind": "ascend", "arch": "dav-2201"},
            target_host="c",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    return artifact.kernel_source, snapshots


def _native_arithmetic_raw_edges(mod):
    operations = [
        call for call in nodes(mod, tirx.Call)
        if isinstance(call.op, tvm.ir.Op) and call.op.name.startswith("tl.ascend_membase.")
    ]
    last_write, edges = {}, []
    for index, operation in enumerate(operations):
        accesses = []
        for argument in operation.args:
            if isinstance(argument, tirx.Call) and isinstance(argument.op, tvm.ir.Op) and argument.op.name == "tl.access_ptr":
                load, count, effect = argument.args
                # This fixture has identical zero-origin dense regions. Its
                # dependency assertions use storage/effects, not operation names.
                assert isinstance(load, tirx.BufferLoad)
                assert all(int(origin) == 0 for origin in load.indices)
                assert int(count) == 130
                accesses.append((load.buffer.data, int(effect)))
        for storage, effect in accesses:
            if effect & 1 and storage in last_write:
                edges.append((last_write[storage], index))
        for storage, effect in accesses:
            if effect & 2:
                last_write[storage] = index
    return operations, set(edges)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("compound_expression", [False, True], ids=["separate-parallel", "one-parallel-expression"])
def test_full_native_arithmetic_lowering_emits_vector_completion(dtype, compound_expression):
    capacity, count = 144, 130

    @T.prim_func
    def main(A: T.Tensor((capacity,), dtype), B: T.Tensor((capacity,), dtype), O: T.Tensor((capacity,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((capacity,), dtype)
            b = T.alloc_shared((capacity,), dtype)
            added = T.alloc_shared((capacity,), dtype)
            subtracted = T.alloc_shared((capacity,), dtype)
            multiplied = T.alloc_shared((capacity,), dtype)
            output = T.alloc_shared((capacity,), dtype)
            T.copy(A, a)
            T.copy(B, b)
            if compound_expression:
                for i in T.Parallel(count):
                    # Keep a real chain after upstream algebraic cancellation.
                    output[i] = ((a[i] + b[i]) - b[i] * b[i]) * b[i] / (a[i] + T.cast(1, dtype))
            else:
                for i in T.Parallel(count):
                    added[i] = a[i] + b[i]
                for i in T.Parallel(count):
                    subtracted[i] = added[i] - b[i]
                for i in T.Parallel(count):
                    multiplied[i] = subtracted[i] * b[i]
                for i in T.Parallel(count):
                    output[i] = multiplied[i] / b[i]
            T.copy(output[0:count], O[0:count])

    source, snapshots = _lower_native(main)
    # Pass-local tests alone missed codegen silently dropping these barriers.
    for name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
        assert len(_barriers(snapshots[name])) >= 3
    operations, raw_edges = _native_arithmetic_raw_edges(snapshots["tl.LowerScheduledTIR"])
    assert len(raw_edges) >= 3
    positions, cursor = [], 0
    for operation in operations:
        function = "asc_" + operation.op.name.rsplit(".", 1)[-1] + "("
        position = source.index(function, cursor)
        positions.append(position)
        cursor = position + len(function)
    for producer, consumer in raw_edges:
        assert "asc_sync_pipe(PIPE_V);" in source[positions[producer]:positions[consumer]]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_full_native_ub_copy_chain_emits_vector_completion(dtype):
    capacity, count = 32, 17

    @T.prim_func
    def main(A: T.Tensor((capacity,), dtype), O: T.Tensor((capacity,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((capacity,), dtype)
            copied = T.alloc_shared((capacity,), dtype)
            assigned = T.alloc_shared((capacity,), dtype)
            T.copy(A, a)
            T.copy(a[0:count], copied[0:count])
            for i in T.Parallel(count):
                assigned[i] = copied[i]
            T.copy(assigned[0:count], O[0:count])

    source, snapshots = _lower_native(main)
    for name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
        assert _barriers(snapshots[name])
    copy_name = "tl_ascend_membase_copy_bits("
    first = source.index(copy_name)
    second = source.index(copy_name, first + len(copy_name))
    assert "asc_sync_pipe(PIPE_V);" in source[first:second]


def test_ordinary_vector_region_exposes_internal_cast_math_completion():
    @T.prim_func
    def main(A: T.Tensor((32,), "float16"), B: T.Tensor((32,), "float16"), O: T.Tensor((32,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((32,), "float16")
            b = T.alloc_shared((32,), "float16")
            output = T.alloc_shared((32,), "float16")
            T.copy(A, a)
            T.copy(B, b)
            with T.Vector(vector=1):
                for i in T.Parallel(17):
                    output[i] = T.cast((T.cast(a[i], "float32") + T.cast(b[i], "float32")) * T.cast(b[i], "float32"), "float16")
            T.copy(output[0:17], O[0:17])

    source, snapshots = _lower_native(main)
    for name in ("tl.InsertSync", "tl.LowerScheduledTIR"):
        assert len(_barriers(snapshots[name])) >= 3
    # The two independent widening casts can issue together. Their consumer,
    # following multiplication, and narrowing cast require completion.
    cast = source.index("asc_half2float(")
    second_cast = source.index("asc_half2float(", cast + 1)
    added = source.index("asc_add(")
    multiplied = source.index("asc_mul(")
    narrowed = source.index("asc_float2half_rn(")
    assert "asc_sync_pipe(PIPE_V);" not in source[cast:second_cast]
    for producer, consumer in ((second_cast, added), (added, multiplied), (multiplied, narrowed)):
        assert "asc_sync_pipe(PIPE_V);" in source[producer:consumer]
