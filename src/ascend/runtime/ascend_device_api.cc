/*!
 * \file ascend_device_api.cc
 * \brief Minimal DeviceAPI for the Ascend NPU, registered as
 *        "device_api.ext_dev".
 *
 * The Ascend target reports DLPack device type kDLExtDev (12) — the same type
 * torch_npu assigns to NPU tensors. Because kDLExtDev != kDLCPU,
 * `MakePackedAPI` injects a `__tvm_set_device(12, dev_id)` call into the host
 * stub (see DeviceAPI::NeedSetDevice). That call resolves the DeviceAPI factory
 * "device_api." + DLDeviceType2Str(kDLExtDev) == "device_api.ext_dev".
 *
 * On the tvm_ffi execution path the producer framework owns the device context
 * and stream. Host-lowered alloc_global buffers use its DLPack allocator, with
 * an owning Tensor retained until TVMBackendFreeWorkspace. The allocator and
 * work stream are installed together by TVM-FFI's call context; releasing the
 * owner after kernel enqueue permits the framework's stream-ordered allocator
 * to reuse memory safely. No raw CANN allocation or device synchronization is
 * introduced here. Ordinary data allocation and copies remain unsupported.
 */
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace runtime {

class AscendDeviceAPI final : public DeviceAPI {
public:
  void SetDevice(Device dev) final {}

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any *rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void *AllocDataSpace(Device dev, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not allocate device memory; NPU buffers are "
           "managed by PyTorch (torch_npu).";
    TVM_FFI_UNREACHABLE();
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not free device memory; NPU buffers are "
           "managed by PyTorch (torch_npu).";
    TVM_FFI_UNREACHABLE();
  }

  void *AllocWorkspace(Device dev, size_t nbytes,
                       DLDataType type_hint = {}) final {
    CheckWorkspaceDevice(dev);
    TVM_FFI_CHECK(nbytes <=
                      static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                  ValueError)
        << "Ascend workspace size exceeds the DLPack int64 shape range";
    TVM_FFI_CHECK(TVMFFIEnvGetDLPackManagedTensorAllocator() != nullptr,
                  RuntimeError)
        << "Ascend workspace requires the TVM-FFI producer DLPack allocator; "
           "invoke the kernel with a framework tensor";
    TVMFFIStreamHandle stream =
        TVMFFIEnvGetStream(dev.device_type, dev.device_id);
    TVM_FFI_CHECK(stream != nullptr, RuntimeError)
        << "Ascend workspace requires the producer's current NPU stream; "
           "install the Torch NPU stream exchange before invocation";

    // Allocate bytes, not type_hint elements: LowerTVMBuiltin already computed
    // the complete byte size, including physical layout and buffer versions.
    // A one-byte allocation also gives zero-size workspaces a unique handle.
    int64_t extent = static_cast<int64_t>(std::max(nbytes, size_t{1}));
    DLDataType byte_dtype{kDLUInt, 8, 1};
    ffi::Tensor owner = ffi::Tensor::FromEnvAlloc(
        TVMFFIEnvTensorAlloc, ffi::ShapeView(&extent, 1), byte_dtype, dev);
    TVM_FFI_CHECK(owner.device().device_type == dev.device_type &&
                      owner.device().device_id == dev.device_id,
                  RuntimeError)
        << "Ascend workspace allocator returned a tensor on the wrong device";
    TVM_FFI_CHECK(owner.dtype().code == kDLUInt && owner.dtype().bits == 8 &&
                      owner.dtype().lanes == 1 && owner.ndim() == 1 &&
                      owner.shape()[0] >= extent && owner.IsContiguous() &&
                      owner.byte_offset() == 0,
                  RuntimeError)
        << "Ascend workspace allocator must return a contiguous byte tensor "
           "covering the requested size with zero byte offset";
    void *ptr = owner.data_ptr();
    TVM_FFI_CHECK(ptr != nullptr &&
                      reinterpret_cast<uintptr_t>(ptr) % kTempAllocaAlignment ==
                          0,
                  RuntimeError)
        << "Ascend workspace allocator returned a null or insufficiently "
           "aligned "
           "pointer (required alignment "
        << kTempAllocaAlignment << " bytes)";

    std::lock_guard<std::mutex> lock(workspace_mutex_);
    auto &allocations = workspaces_[dev.device_id];
    TVM_FFI_CHECK(allocations.count(ptr) == 0, RuntimeError)
        << "Ascend workspace allocator returned a pointer that is already live";
    allocations.emplace(ptr, Workspace{std::move(owner), stream});
    return ptr;
  }

  void FreeWorkspace(Device dev, void *ptr) final {
    CheckWorkspaceDevice(dev);
    if (ptr == nullptr) {
      return;
    }
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(workspace_mutex_);
      auto device = workspaces_.find(dev.device_id);
      TVM_FFI_CHECK(device != workspaces_.end(), RuntimeError)
          << "Ascend workspace must be released on its allocation device";
      auto allocation = device->second.find(ptr);
      TVM_FFI_CHECK(allocation != device->second.end(), RuntimeError)
          << "Ascend workspace pointer is not a live allocation on this device";
      TVM_FFI_CHECK(allocation->second.stream ==
                        TVMFFIEnvGetStream(dev.device_type, dev.device_id),
                    RuntimeError)
          << "Ascend workspace must be released in the same producer stream "
             "context as its allocation";
      owner = std::move(allocation->second.owner);
      device->second.erase(allocation);
    }
    // Invoke the producer's DLPack deleter outside the lock. The retained
    // framework Tensor returns memory to its stream-aware allocator, even if
    // the environment allocator has since changed. Kernel work is asynchronous.
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  static AscendDeviceAPI *Global() {
    // Explicit new to avoid exit-time destruction of global state; the OS
    // reclaims it at process exit (matches CPUDeviceAPI).
    static auto *inst = new AscendDeviceAPI();
    return inst;
  }

protected:
  void CopyDataFromTo(const void *from, size_t from_offset, void *to,
                      size_t to_offset, size_t size, Device dev_from,
                      Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    TVM_FFI_THROW(RuntimeError)
        << "AscendDeviceAPI does not implement CopyDataFromTo; copies are "
           "performed by PyTorch (torch_npu).";
  }

private:
  struct Workspace {
    ffi::Tensor owner;
    TVMFFIStreamHandle stream;
  };

  static void CheckWorkspaceDevice(Device dev) {
    TVM_FFI_CHECK(dev.device_type == kDLExtDev && dev.device_id >= 0,
                  ValueError)
        << "Ascend workspace requires an ext_dev device with a nonnegative id";
  }

  std::mutex workspace_mutex_;
  // Device pointers can have the same numeric address in different contexts.
  std::unordered_map<int32_t, std::unordered_map<void *, Workspace>>
      workspaces_;
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.ext_dev",
                               [](ffi::PackedArgs args, ffi::Any *rv) {
                                 DeviceAPI *ptr = AscendDeviceAPI::Global();
                                 *rv = static_cast<void *>(ptr);
                               });
}

} // namespace runtime
} // namespace tvm
