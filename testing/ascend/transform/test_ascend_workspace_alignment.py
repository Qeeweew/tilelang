"""Compiler-owned workspace alignment preserves allocation ownership and ABI."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend.transform.test_ascend_host_workspaces import _calls, _mixed_gemm, _nodes, _workspace_bindings


LIMIT = (1 << 63) - 1 - 511
SIZE_ERROR = "workspace byte size exceeds int64 capacity after 512-byte alignment"


def _call(dtype, name, args):
    return tirx.Call(dtype, tvm.ir.Op.get(name), args)


def _allocation_program(nbytes, *, external_free=False):
    workspace, external = tirx.Var("workspace", "handle"), tirx.Var("external", "handle")
    params = ([nbytes] if isinstance(nbytes, tirx.Var) else []) + [external]
    body = [
        tirx.Bind(workspace, _call("handle", "tirx.TVMBackendAllocWorkspace", [12, 3, nbytes, 2, 32])),
        tirx.IfThenElse(
            _call("bool", "tirx.isnullptr", [workspace]),
            tirx.Evaluate(_call("int32", "tirx.tvm_throw_last_error", [])),
            None,
        ),
        tirx.Evaluate(tirx.call_extern("int32", "observe_workspace", workspace, external, nbytes)),
        tirx.Evaluate(_call("int32", "tirx.TVMBackendFreeWorkspace", [12, 3, workspace])),
    ]
    if external_free:
        body.append(tirx.Evaluate(_call("int32", "tirx.TVMBackendFreeWorkspace", [12, 3, external])))
    function = tirx.PrimFunc(params, tirx.SeqStmt(body)).with_attr(
        {
            "global_symbol": "workspace_entry",
            "target": tvm.target.Target("c"),
        }
    )
    return tvm.IRModule({"main": function}), workspace, external


def _align(module, arch="dav-2201"):
    return transform.AscendAlignWorkspaceAllocations(tvm.target.Target({"kind": "ascend", "arch": arch}))(module)


@pytest.mark.parametrize("size", [0, 1, 511, 512, 5003, LIMIT])
def test_static_allocation_reserves_exact_slack_and_keeps_external_pointer(size):
    before, workspace, external = _allocation_program(tirx.const(size, "uint64"), external_free=True)
    after = _align(before)
    (allocation,) = _workspace_bindings(after)
    assert int(tvm.arith.Analyzer().simplify(allocation.value.args[2])) == size + 511
    assert not allocation.var.same_as(workspace)
    aliases = [node for node in _nodes(after, tirx.Bind) if node.var.same_as(workspace)]
    assert len(aliases) == 1
    assert aliases[0].value.op.name == "tirx.handle_add_byte_offset"
    assert aliases[0].value.args[0].same_as(allocation.var)
    (observe,) = _calls(after, "tirx.call_extern")
    assert observe.args[1].same_as(workspace) and observe.args[2].same_as(external)
    frees = _calls(after, "tirx.TVMBackendFreeWorkspace")
    assert sum(call.args[2].same_as(allocation.var) for call in frees) == 1
    assert sum(call.args[2].same_as(external) for call in frees) == 1
    assert not _nodes(after, tirx.AssertStmt)
    # Both the new pre-arithmetic check and the original lowered check use raw.
    assert len(_calls(after, "tirx.isnullptr")) == 2
    assert all(call.args[0].same_as(allocation.var) for call in _calls(after, "tirx.isnullptr"))


@pytest.mark.parametrize("dtype, size", [("int64", -1), ("int64", LIMIT + 1), ("uint64", (1 << 64) - 1)])
def test_static_allocation_rejects_byte_overflow_before_allocating(dtype, size):
    before, _, _ = _allocation_program(tirx.const(size, dtype))
    with pytest.raises(ValueError, match=SIZE_ERROR):
        _align(before)


def _execute_size_predicate(tmp_path, nbytes, condition, values):
    """Use C's real uint64 semantics, including TVM's large_uint_imm boundary."""
    from tvm_ffi import libinfo as ffi_libinfo

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a host C++ compiler is required")
    function = tirx.PrimFunc(
        [nbytes],
        tirx.Evaluate(_call("int32", "tirx.ret", [tirx.Cast("int32", condition)])),
        ret_type=tvm.ir.PrimType("int32"),
    ).with_attr({"global_symbol": "workspace_size_valid", "target": tvm.target.Target("c")})
    generated = tvm.ffi.get_global_func("target.build.c")(tvm.IRModule({"main": function}), tvm.target.Target("c"))
    (tmp_path / "predicate.c").write_text(generated.inspect_source())
    suffix = "ULL" if nbytes.dtype == "uint64" else "LL"
    checks = [
        f"if (workspace_size_valid({value}{suffix}) != {int(0 <= value <= LIMIT)}) return {index + 1};"
        for index, value in enumerate(values)
    ]
    source = tmp_path / "predicate_test.cc"
    source.write_text('#include "predicate.c"\nint main() {\n' + "\n".join(checks) + "\nreturn 0;\n}\n")
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / "predicate_test"
    command = [
        *compiler,
        "-std=c++17",
        "-O0",
        "-I",
        str(root / "3rdparty" / "tvm" / "include"),
        "-I",
        ffi_libinfo.find_include_path(),
        "-I",
        ffi_libinfo.find_dlpack_include_path(),
        str(source),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True, check=False, timeout=10)
    assert completed.returncode == 0, f"predicate boundary {completed.returncode}: " + completed.stdout + completed.stderr


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_dynamic_allocation_checks_signed_range_before_raw_bind(tmp_path, dtype):
    nbytes = tirx.Var("nbytes", dtype)
    before, _, _ = _allocation_program(nbytes)
    after = _align(before)
    (check,) = _nodes(after, tirx.AssertStmt)
    (allocation,) = _workspace_bindings(after)
    assert SIZE_ERROR in str(check)
    statements = list(after["main"].body.seq)
    assert next(index for index, stmt in enumerate(statements) if stmt.same_as(check)) < next(
        index for index, stmt in enumerate(statements) if stmt.same_as(allocation)
    )
    values = [0, 1, 5003, LIMIT, LIMIT + 1] + ([-1] if dtype == "int64" else [(1 << 64) - 1])
    _execute_size_predicate(tmp_path, nbytes, check.condition, values)
    for value in values:
        substitution = {nbytes: tirx.const(value, dtype)}
        if 0 <= value <= LIMIT:
            reserved = tirx.stmt_functor.substitute(allocation.value.args[2], substitution)
            assert int(tvm.arith.Analyzer().simplify(reserved)) == value + 511


def test_3510_preserves_original_workspace_abi_and_bytes():
    before, _, _ = _allocation_program(tirx.Var("nbytes", "int64"), external_free=True)
    tvm.ir.assert_structural_equal(_align(before, "dav-3510"), before)


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_host_codegen_uses_device_target_after_binding_cpu_target(arch):
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import host_codegen

    before, _, _ = _allocation_program(tirx.const(5003, "uint64"))
    captured = []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendAlignWorkspaceAllocations":
                captured.append(mod)

    context = create_backend_context({"kind": "ascend", "arch": arch}, target_host="c", execution_backend="tvm_ffi")
    with tvm.transform.PassContext(instruments=[Capture()]):
        generated = host_codegen(before, context)
    assert len(captured) == 1
    (allocation,) = _workspace_bindings(captured[0])
    assert int(tvm.arith.Analyzer().simplify(allocation.value.args[2])) == 5003 + (511 if arch == "dav-2201" else 0)
    assert str(captured[0]["main"].attrs["target"].kind.name) == "c"
    assert generated.inspect_source()


def test_padded_tail_handoff_keeps_logical_region_and_group_pitch():
    from testing.ascend.transform.test_ascend_gm_handoff import _buffer, _lower, _region, _workspace

    before, _ = _lower(_buffer((17, 33)), _buffer((17, 33), scope="shared", name="dst"), grid=(3,))
    (workspace,) = _workspace(before)
    assert tuple(int(extent) for extent in workspace.shape) == (3, 32, 48)
    for transfer in _calls(before, "tl.tileop.ascend_copy"):
        assert tuple(int(value) for value in _region(transfer.args[0])[2][-2:]) == (17, 33)
        assert tuple(int(value) for value in _region(transfer.args[1])[2][-2:]) == (17, 33)
    # Alignment adds only host allocation slack. The existing padded group
    # plane and every physical version of the whole allocation remain aligned.
    group_bytes = 32 * 48 * 4
    assert group_bytes % 512 == 0
    host, _, _ = _allocation_program(tirx.const(2 * 3 * group_bytes, "uint64"))
    (allocation,) = _workspace_bindings(_align(host))
    assert int(tvm.arith.Analyzer().simplify(allocation.value.args[2])) == 2 * 3 * group_bytes + 511
    assert tuple(int(extent) for extent in workspace.shape) == (3, 32, 48)


# Execute the actual generated host stub, replacing only its packed device
# function and allocator. This tests pointer ownership/footprints, not NPU math.
HOST_HARNESS = r"""
#include "host.c"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned char kGuard = 0xa5, kPayload = 0x3c;
struct Allocation {
  unsigned char *backing, *raw, *alias = nullptr;
  size_t bytes, total;
  bool freed = false;
};
std::vector<Allocation> allocations;
std::array<void *, 4> external{};
int kernel_calls = 0, external_seen = 0;
std::string raised;
void Require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
}

extern "C" void* TVMBackendAllocWorkspace(int device, int id, uint64_t bytes,
                                          int dtype_code, int dtype_bits) {
  Require(device == kDLExtDev && id == 3, "allocation device identity");
  Require((bytes == 1023 || bytes == 1535) &&
              ((dtype_code == kDLFloat && (dtype_bits == 16 || dtype_bits == 32)) ||
               (dtype_code == kDLBfloat && dtype_bits == 16)),
          "exact byte reservation and dtype hint");
  void *memory = nullptr;
  size_t total = static_cast<size_t>(bytes) + 1024;
  Require(posix_memalign(&memory, 512, total) == 0, "host allocator");
  auto *backing = static_cast<unsigned char *>(memory);
  std::memset(backing, kGuard, total);
  // Deliberately satisfy the runtime's 64-byte minimum but not 512 bytes.
  auto *raw = backing + 64;
  Require(reinterpret_cast<uintptr_t>(raw) % 512 == 64, "test allocator offset");
  allocations.push_back({backing, raw, nullptr, static_cast<size_t>(bytes), total});
  return raw;
}

extern "C" int TVMBackendFreeWorkspace(int device, int id, void *pointer) {
  Require(device == kDLExtDev && id == 3, "free device identity");
  for (auto &owner : allocations) {
    if (pointer != owner.raw) continue;
    Require(!owner.freed && owner.alias != nullptr, "free raw owner exactly once after launch");
    size_t start = static_cast<size_t>(owner.alias - owner.backing);
    size_t payload = owner.bytes - 511;
    for (size_t i = 0; i < owner.total; ++i)
      Require(owner.backing[i] == (start <= i && i < start + payload ? kPayload : kGuard),
              "workspace payload or surrounding allocation bytes changed");
    owner.freed = true;
    std::free(owner.backing);
    return 0;
  }
  throw std::runtime_error("free received alias instead of raw allocation");
}

extern "C" int TVMBackendGetFuncFromEnv(void *, const char *name, TVMFFIObjectHandle *out) {
  *out = reinterpret_cast<void *>(std::string(name) == "__tvm_set_device" ? 1 : 2);
  return 0;
}

extern "C" int TVMFFIFunctionCall(TVMFFIObjectHandle function, TVMFFIAny *args,
                                  int32_t count, TVMFFIAny *) {
  if (function == reinterpret_cast<void *>(1)) {
    Require(count == 2 && args[0].v_int64 == kDLExtDev && args[1].v_int64 == 3,
            "set_device arguments");
    return 0;
  }
  Require(function == reinterpret_cast<void *>(2) && ++kernel_calls == 1, "one kernel launch");
  int workspaces = 0;
  for (int index = 0; index < count; ++index) {
    if (args[index].type_index != kTVMFFIOpaquePtr) continue;
    void *pointer = args[index].v_ptr;
    bool is_external = false;
    for (void *original : external) is_external |= pointer == original;
    if (is_external) { ++external_seen; continue; }
    Allocation *matched = nullptr;
    for (auto &owner : allocations) {
      auto address = reinterpret_cast<uintptr_t>(pointer);
      auto base = reinterpret_cast<uintptr_t>(owner.raw);
      if (base <= address && address < base + owner.bytes) matched = &owner;
    }
    Require(matched && !matched->alias && reinterpret_cast<uintptr_t>(pointer) % 512 == 0,
            "kernel receives each aligned alias exactly once");
    matched->alias = static_cast<unsigned char *>(pointer);
    Require(matched->alias + matched->bytes - 511 <= matched->raw + matched->bytes,
            "complete logical footprint fits raw allocation");
    std::memset(pointer, kPayload, matched->bytes - 511);
    ++workspaces;
  }
  Require(workspaces == 2 && external_seen == 4, "exact workspace and external pointer ABI");
  return 0;
}

extern "C" void TVMFFIErrorSetRaisedFromCStr(const char *kind, const char *message) {
  raised = std::string(kind) + ": " + message;
}

int main() {
  try {
    int64_t shape[] = {16, 16};
    int64_t strides[] = {16, 1};
    std::array<DLTensor, 4> tensors{};
    std::array<TVMFFIAny, 4> args{};
    std::array<void *, 4> backing{};
    for (int index = 0; index < 4; ++index) {
      Require(posix_memalign(&backing[index], 512, 1024) == 0, "external allocator");
      external[index] = static_cast<unsigned char *>(backing[index]) + 64;
      std::memset(backing[index], kGuard, 1024);
      tensors[index].data = external[index];
      tensors[index].device = {kDLExtDev, 3};
      tensors[index].ndim = 2;
      tensors[index].dtype = {TEST_DTYPE_CODE, 16, 1};
      tensors[index].shape = shape;
      tensors[index].strides = strides;
      args[index].type_index = kTVMFFIDLTensorPtr;
      args[index].v_ptr = &tensors[index];
    }
    TVMFFIAny result{};
    Require(__tvm_ffi_main(nullptr, args.data(), 4, &result) == 0, raised.c_str());
    Require(allocations.size() == 2 && kernel_calls == 1 && external_seen == 4, "launch counts");
    for (const auto &owner : allocations) Require(owner.freed, "workspace owner leaked");
    for (void *memory : backing) {
      for (int index = 0; index < 1024; ++index)
        Require(static_cast<unsigned char *>(memory)[index] == kGuard, "external data changed");
      std::free(memory);
    }
    std::cout << "aligned aliases, exact footprints, raw frees, unchanged external ABI\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Ascend host ABI execution uses a Linux C++ compiler")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_generated_c_host_stub_aligns_non512_allocator_and_frees_raw(tmp_path, dtype):
    from tvm_ffi import libinfo as ffi_libinfo

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a host C++ compiler is required")
    artifact = tilelang.lower(
        _mixed_gemm(dtype),
        target={"kind": "ascend", "arch": "dav-2201"},
        target_host="c",
        enable_host_codegen=True,
        enable_device_compile=False,
    )
    (tmp_path / "host.c").write_text(artifact.host_mod.inspect_source())
    source = tmp_path / "harness.cc"
    source.write_text(HOST_HARNESS)
    root = Path(__file__).resolve().parents[3]
    executable = tmp_path / "host_workspace"
    command = [
        *compiler,
        "-std=c++17",
        "-O0",
        "-fno-strict-aliasing",
        "-DTEST_DTYPE_CODE=" + ("2" if dtype == "float16" else "4"),
        "-I",
        str(root / "3rdparty" / "tvm" / "include"),
        "-I",
        ffi_libinfo.find_include_path(),
        "-I",
        ffi_libinfo.find_dlpack_include_path(),
        str(source),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True, check=False, timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
