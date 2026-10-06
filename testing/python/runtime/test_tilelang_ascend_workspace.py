"""Test framework-owned Ascend workspaces with a host-only DLPack allocator."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Ascend runtime is Linux-only")


HARNESS = r"""
#include ASCEND_RUNTIME_SOURCE
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using tvm::runtime::AscendDeviceAPI;
using tvm::Device;

namespace {
int created = 0, released = 0;
bool enqueued = false, released_after_enqueue = false, reenter = false;
int variant = 0;
int64_t requested_bytes = 0;
TVMFFIStreamHandle release_stream = nullptr;
Device device{kDLExtDev, 3};
void *stream = reinterpret_cast<void *>(0x1230);

void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename F>
void Reject(F callback, const char *fragment) {
  try {
    callback();
  } catch (const std::exception &error) {
    Require(std::string(error.what()).find(fragment) != std::string::npos,
            error.what());
    return;
  }
  throw std::runtime_error(std::string("expected rejection: ") + fragment);
}

struct Owner {
  DLManagedTensorVersioned managed{};
  int64_t shape = 0, stride = 1;
  void *memory = nullptr;

  static void Delete(DLManagedTensorVersioned *managed) {
    auto *owner = static_cast<Owner *>(managed->manager_ctx);
    ++released;
    release_stream = TVMFFIEnvGetStream(device.device_type, device.device_id);
    released_after_enqueue = enqueued;
    if (reenter) {
      reenter = false;
      auto *api = AscendDeviceAPI::Global();
      void *nested = api->AllocWorkspace(device, 16);
      api->FreeWorkspace(device, nested);
    }
    std::free(owner->memory);
    delete owner;
  }
};

int Allocate(DLTensor *prototype, DLManagedTensorVersioned **out, void *error_ctx,
             void (*set_error)(void *, const char *, const char *)) {
  *out = nullptr;
  if (variant == 7) {
    set_error(error_ctx, "MemoryError", "fake allocation failure");
    return -1;
  }
  if (variant == 8) return 0;
  Require(prototype->dtype.code == kDLUInt && prototype->dtype.bits == 8 &&
              prototype->dtype.lanes == 1 && prototype->ndim == 1,
          "workspace must allocate one-dimensional bytes");
  auto *owner = new Owner;
  requested_bytes = prototype->shape[0];
  owner->shape = requested_bytes;
  Require(posix_memalign(&owner->memory, 64, static_cast<size_t>(requested_bytes) + 64) == 0,
          "fake host allocation failed");
  owner->managed.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
  owner->managed.manager_ctx = owner;
  owner->managed.deleter = Owner::Delete;
  owner->managed.dl_tensor = *prototype;
  owner->managed.dl_tensor.data = owner->memory;
  owner->managed.dl_tensor.shape = &owner->shape;
  owner->managed.dl_tensor.strides = &owner->stride;
  switch (variant) {
    case 1: ++owner->managed.dl_tensor.device.device_id; break;
    case 2: owner->managed.dl_tensor.dtype.bits = 16; break;
    case 3: owner->stride = 2; break;
    case 4: --owner->shape; break;
    case 5: owner->managed.dl_tensor.byte_offset = 1; break;
    case 6: owner->managed.dl_tensor.data = static_cast<char *>(owner->memory) + 1; break;
    case 9: owner->managed.dl_tensor.data = nullptr; break;
  }
  ++created;
  *out = &owner->managed;
  return 0;
}

void Setup() {
  Require(TVMFFIEnvSetDLPackManagedTensorAllocator(Allocate, 0, nullptr) == 0,
          "set allocator");
  Require(TVMFFIEnvSetStream(device.device_type, device.device_id, stream, nullptr) == 0,
          "set stream");
}

void Run(const std::string &mode) {
  auto *api = AscendDeviceAPI::Global();
  Setup();
  if (mode == "lifetime") {
    DLDataType hint{kDLFloat, 16, 1};
    void *ptr = api->AllocWorkspace(device, 5003, hint);
    Require(ptr != nullptr && requested_bytes == 5003 && created == 1 && released == 0,
            "byte allocation and retained owner");
    // The owner's deleter is independent of subsequent allocator replacement.
    TVMFFIEnvSetDLPackManagedTensorAllocator(nullptr, 0, nullptr);
    enqueued = true;
    api->FreeWorkspace(device, ptr);
    Require(released == 1 && released_after_enqueue && release_stream == stream,
            "release owning DLPack tensor after enqueue on the producer stream");
    Reject([&] { api->FreeWorkspace(device, ptr); }, "not a live allocation");
    api->FreeWorkspace(device, nullptr);
  } else if (mode == "device_stream") {
    void *first = api->AllocWorkspace(device, 32);
    void *second = api->AllocWorkspace(device, 32);
    Require(first != second && released == 0, "concurrent workspaces must not alias");
    Device other{kDLExtDev, device.device_id + 1};
    Reject([&] { api->FreeWorkspace(other, first); }, "allocation device");
    TVMFFIEnvSetStream(device.device_type, device.device_id,
                       reinterpret_cast<void *>(0x4560), nullptr);
    Reject([&] { api->FreeWorkspace(device, first); }, "same producer stream");
    Require(released == 0, "invalid free must retain both owners");
    TVMFFIEnvSetStream(device.device_type, device.device_id, stream, nullptr);
    api->FreeWorkspace(device, second);
    api->FreeWorkspace(device, first);
    Require(released == 2, "both owners released");
  } else if (mode == "boundary") {
    Reject([&] { api->AllocWorkspace(Device{kDLCPU, 0}, 32); }, "ext_dev");
    Reject([&] { api->AllocWorkspace(Device{kDLExtDev, -1}, 32); }, "nonnegative");
    Reject([&] { api->AllocWorkspace(device, std::numeric_limits<size_t>::max()); }, "int64");
    void *ptr = api->AllocWorkspace(device, 0);
    Require(requested_bytes == 1 && ptr != nullptr, "zero bytes need a unique nonnull handle");
    api->FreeWorkspace(device, ptr);
    Require(created == 1 && released == 1, "no allocations on invalid input");
  } else if (mode == "missing_context") {
    TVMFFIEnvSetDLPackManagedTensorAllocator(nullptr, 0, nullptr);
    Reject([&] { api->AllocWorkspace(device, 32); }, "DLPack allocator");
    Setup();
    TVMFFIEnvSetStream(device.device_type, device.device_id, nullptr, nullptr);
    Reject([&] { api->AllocWorkspace(device, 32); }, "current NPU stream");
    Require(created == 0, "reject missing context before allocation");
  } else if (mode == "malformed") {
    for (variant = 1; variant <= 6; ++variant) {
      const char *message = variant == 1 ? "wrong device" :
          variant == 6 ? "aligned pointer" : "contiguous byte tensor";
      Reject([&] { api->AllocWorkspace(device, 32); }, message);
      Require(created == released, "reject malformed tensors without leaking owner");
    }
    variant = 9;
    Reject([&] { api->AllocWorkspace(device, 32); }, "aligned pointer");
    Require(created == released, "reject null pointer without leaking owner");
  } else if (mode == "allocator_error") {
    variant = 7;
    Reject([&] { api->AllocWorkspace(device, 32); }, "fake allocation failure");
    variant = 8;
    Reject([&] { api->AllocWorkspace(device, 32); }, "failed to allocate a tensor");
    Require(created == 0, "allocator failure creates no owners");
  } else if (mode == "reentrant_release") {
    void *ptr = api->AllocWorkspace(device, 32);
    reenter = true;
    api->FreeWorkspace(device, ptr);
    Require(created == 2 && released == 2, "framework deleter runs outside the workspace mutex");
  } else {
    throw std::runtime_error("unknown mode");
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 2, "one mode required");
    Run(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
"""


@pytest.fixture(scope="module")
def workspace_harness(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a host C++ compiler is required")
    import tilelang
    from tilelang.libinfo import find_lib_path
    from tvm_ffi import libinfo as ffi_libinfo

    root = Path(__file__).resolve().parents[3]
    source = root / "src" / "ascend" / "runtime" / "ascend_device_api.cc"
    if not source.is_file():
        root = Path(tilelang.__file__).parent
        source = root / "src" / "ascend" / "runtime" / "ascend_device_api.cc"
    if not source.is_file():
        pytest.skip("Ascend runtime sources are required")

    runtime = Path(find_lib_path("tvm_runtime"))
    ffi_library = Path(ffi_libinfo.find_libtvm_ffi())
    work = tmp_path_factory.mktemp("ascend_workspace")
    input_path = work / "harness.cc"
    input_path.write_text(HARNESS)
    executable = work / "harness"
    # DT_RPATH selects these exact installed runtime libraries for this standalone
    # host executable, including on systems where CANN supplies a different TVM.
    library_dirs = sorted({str(runtime.parent), str(ffi_library.parent)})
    command = [
        *compiler, "-std=c++17", "-pthread",
        f'-DASCEND_RUNTIME_SOURCE="{source}"',
        "-I", str(root / "3rdparty" / "tvm" / "include"),
        "-I", ffi_libinfo.find_include_path(),
        "-I", ffi_libinfo.find_dlpack_include_path(),
        str(input_path), str(runtime), str(ffi_library),
        "-Wl,--disable-new-dtags", "-Wl,-rpath," + ":".join(library_dirs),
        "-o", str(executable),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.parametrize(
    "mode", ["lifetime", "device_stream", "boundary", "missing_context", "malformed", "allocator_error", "reentrant_release"]
)
def test_framework_workspace_contract(workspace_harness, mode):
    result = subprocess.run([str(workspace_harness), mode], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
