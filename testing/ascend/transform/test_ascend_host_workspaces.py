"""Host storage legalization preserves native-device workspace ownership."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tvm import tirx


def _mixed_gemm(dtype):
    @T.prim_func
    def main(A: T.Tensor((16, 16), dtype), B: T.Tensor((16, 16), dtype), V: T.Tensor((16, 16), dtype), O: T.Tensor((16, 16), dtype)):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), dtype)
            b_l1 = T.alloc_l1((16, 16), dtype)
            score_l0 = T.alloc_l0c((16, 16), "float32")
            scores = T.alloc_shared((16, 16), "float32")
            probabilities = T.alloc_shared((16, 16), dtype)
            p_l1 = T.alloc_l1((16, 16), dtype)
            v_l1 = T.alloc_l1((16, 16), dtype)
            output_l0 = T.alloc_l0c((16, 16), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, score_l0, transpose_B=True, clear_accum=True)
            T.copy(score_l0, scores)
            for i, j in T.Parallel(16, 16):
                probabilities[i, j] = T.cast(scores[i, j], dtype)
            T.copy(probabilities, p_l1)
            T.copy(V, v_l1)
            T.gemm(p_l1, v_l1, output_l0, transpose_B=False, clear_accum=True)
            T.copy(output_l0, O)

    return main


def _nodes(mod, kind):
    found = []
    for function in mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            function.body,
            lambda node: found.append(node) if isinstance(node, kind) else None,
        )
    return found


def _calls(mod, name):
    return [call for call in _nodes(mod, tirx.Call) if isinstance(call.op, tvm.ir.Op) and call.op.name == name]


def _workspace_bindings(mod):
    return [
        binding
        for binding in _nodes(mod, tirx.Bind)
        if isinstance(binding.value, tirx.Call)
        and isinstance(binding.value.op, tvm.ir.Op)
        and binding.value.op.name == "tirx.TVMBackendAllocWorkspace"
    ]


def _assert_workspace_ownership(mod):
    bindings = _workspace_bindings(mod)
    assert len(bindings) == 2
    assert sorted(int(binding.value.args[2]) for binding in bindings) == [512, 1024]
    frees = _calls(mod, "tirx.TVMBackendFreeWorkspace")
    packed_arguments = _calls(mod, "tirx.tvm_struct_set")
    assert len(frees) == 2
    for binding in bindings:
        # Packed kernel arguments and release use the allocation's exact Var,
        # even when BF16StorageLegalize replaces its host pointer type.
        assert sum(free.args[2].same_as(binding.var) for free in frees) == 1
        assert sum(argument.args[3].same_as(binding.var) for argument in packed_arguments) == 1
    assert len(_calls(mod, "tirx.tvm_call_packed_lowered")) == 2  # set_device + kernel


def _assert_aligned_workspace_ownership(mod):
    bindings = _workspace_bindings(mod)
    assert len(bindings) == 2
    assert sorted(int(binding.value.args[2]) for binding in bindings) == [1023, 1535]
    aliases = [
        binding
        for binding in _nodes(mod, tirx.Bind)
        if isinstance(binding.value, tirx.Call)
        and isinstance(binding.value.op, tvm.ir.Op)
        and binding.value.op.name == "tirx.handle_add_byte_offset"
    ]
    assert len(aliases) == 2
    frees = _calls(mod, "tirx.TVMBackendFreeWorkspace")
    packed_arguments = _calls(mod, "tirx.tvm_struct_set")
    assert len(frees) == 2
    for binding in bindings:
        matching = [alias for alias in aliases if alias.value.args[0].same_as(binding.var)]
        assert len(matching) == 1
        (alias,) = matching
        assert not alias.var.same_as(binding.var)
        assert sum(free.args[2].same_as(binding.var) for free in frees) == 1
        assert not any(free.args[2].same_as(alias.var) for free in frees)
        assert sum(argument.args[3].same_as(alias.var) for argument in packed_arguments) == 1
        assert not any(argument.args[3].same_as(binding.var) for argument in packed_arguments)
    assert len(_calls(mod, "tirx.tvm_call_packed_lowered")) == 2


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_full_host_codegen_preserves_gm_workspace_bytes_and_vars(dtype):
    snapshots, order = {}, []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_before_pass(self, mod, info):
            if info.name == "tirx.LowerTVMBuiltin":
                snapshots["before-builtin"] = mod
            if info.name in (
                "tirx.LowerTVMBuiltin",
                "tl.AscendAlignWorkspaceAllocations",
                "tirx.FP8StorageLegalize",
                "tirx.BF16StorageLegalize",
            ):
                order.append(info.name)

        def run_after_pass(self, mod, info):
            if info.name in ("tirx.LowerTVMBuiltin", "tl.AscendAlignWorkspaceAllocations", "tirx.BF16StorageLegalize"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        artifact = tilelang.lower(
            _mixed_gemm(dtype),
            target={"kind": "ascend", "arch": "dav-2201"},
            target_host="c",
            enable_host_codegen=True,
            enable_device_compile=False,
        )
    assert order == ["tirx.LowerTVMBuiltin", "tl.AscendAlignWorkspaceAllocations", "tirx.FP8StorageLegalize", "tirx.BF16StorageLegalize"]
    allocations = _nodes(snapshots["before-builtin"], tirx.AllocBuffer)
    assert sorted((str(node.buffer.dtype), int(node.buffer.shape[0])) for node in allocations) == sorted(
        [
            ("float32", 256),
            (dtype, 256),
        ]
    )
    _assert_workspace_ownership(snapshots["tirx.LowerTVMBuiltin"])
    _assert_aligned_workspace_ownership(snapshots["tl.AscendAlignWorkspaceAllocations"])
    _assert_aligned_workspace_ownership(snapshots["tirx.BF16StorageLegalize"])
    assert artifact.host_mod and artifact.kernel_source

    # Only host storage carriers are legalized. The native kernel still takes
    # four input/output pointers plus its handoff workspace in the math dtype.
    native_pointer_dtypes = [
        str(parameter.type_annotation.element_type.dtype)
        for function in artifact.device_mod.functions.values()
        for parameter in function.params
        if hasattr(parameter.type_annotation, "element_type")
    ]
    assert native_pointer_dtypes.count(dtype) == 5
    assert native_pointer_dtypes.count("float32") == 1
