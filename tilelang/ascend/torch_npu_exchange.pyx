# cython: language_level=3
"""Private, zero-copy Torch NPU arguments for TVM-FFI execution.

TVM-FFI caches an argument type's Exchange API pointer. A separate argument
type gives TileLang a stable NPU-aware table even when torch.Tensor was passed
to FFI before TileLang was imported. Torch's own table is never modified.
"""

import torch
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from libc.stdint cimport int32_t, uint32_t, uintptr_t


ctypedef struct DLPackVersion:
    uint32_t major
    uint32_t minor


ctypedef struct DLPackExchangeAPIHeader:
    DLPackVersion version
    void* prev_api


ctypedef int (*DLPackFromPyObject)(void* value, void** out) except -1
ctypedef int (*DLPackTensorFromPyObject)(void* value, void* out) except -1
ctypedef int (*DLPackCurrentWorkStream)(
    int32_t device_type, int32_t device_id, void** out_stream,
) except -1


ctypedef struct DLPackExchangeAPI:
    DLPackExchangeAPIHeader header
    void* managed_tensor_allocator
    DLPackFromPyObject managed_tensor_from_py_object_no_sync
    void* managed_tensor_to_py_object_no_sync
    DLPackTensorFromPyObject dltensor_from_py_object_no_sync
    DLPackCurrentWorkStream current_work_stream


cdef int DLPACK_MAJOR_VERSION = 1
cdef int DLPACK_EXT_DEVICE = 12
cdef const char* DLPACK_EXCHANGE_CAPSULE = "dlpack_exchange_api"
cdef DLPackExchangeAPI* torch_exchange_original_api = NULL
cdef DLPackExchangeAPI torch_exchange_argument_api
cdef object torch_exchange_original_capsule = None
cdef object torch_npu_stream_getter = None


class _TorchNPUTensorArgument:
    """Keep the original producer tensor alive through zero-copy conversion."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor


cdef int torch_npu_tensor_from_py_object(
    void* value, void** out,
) except -1 with gil:
    cdef object tensor = (<object>value).tensor
    return torch_exchange_original_api.managed_tensor_from_py_object_no_sync(
        <void*>tensor, out,
    )


cdef int torch_npu_dltensor_from_py_object(
    void* value, void* out,
) except -1 with gil:
    cdef object tensor = (<object>value).tensor
    return torch_exchange_original_api.dltensor_from_py_object_no_sync(
        <void*>tensor, out,
    )


cdef int torch_npu_current_work_stream(
    int32_t device_type, int32_t device_id, void** out_stream,
) except -1 with gil:
    if device_type == DLPACK_EXT_DEVICE:
        out_stream[0] = <void*><uintptr_t>torch_npu_stream_getter(device_id)
        return 0
    return torch_exchange_original_api.current_work_stream(
        device_type, device_id, out_stream,
    )


def install_torch_npu_stream_exchange():
    """Initialize TileLang's private NPU argument Exchange API once."""
    global torch_exchange_original_api
    global torch_exchange_original_capsule
    global torch_exchange_argument_api
    global torch_npu_stream_getter

    if torch_exchange_original_api != NULL:
        return False

    import torch_npu

    cdef object capsule
    cdef object stream_getter
    cdef object current_stream
    cdef DLPackExchangeAPI* exchange_api

    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        raise RuntimeError(
            "torch.Tensor does not expose __dlpack_c_exchange_api__; "
            "load TVM-FFI's Torch DLPack extension first"
        )

    capsule = torch.Tensor.__dlpack_c_exchange_api__
    exchange_api = <DLPackExchangeAPI*>PyCapsule_GetPointer(
        capsule, DLPACK_EXCHANGE_CAPSULE,
    )
    if exchange_api.header.version.major != DLPACK_MAJOR_VERSION:
        raise RuntimeError("unsupported DLPack Exchange API major version")
    if (
        exchange_api.managed_tensor_allocator == NULL
        or exchange_api.managed_tensor_from_py_object_no_sync == NULL
        or exchange_api.managed_tensor_to_py_object_no_sync == NULL
        or exchange_api.dltensor_from_py_object_no_sync == NULL
        or exchange_api.current_work_stream == NULL
    ):
        raise RuntimeError("incomplete Torch DLPack Exchange API table")

    stream_getter = getattr(torch_npu._C, "_npu_getCurrentRawStream", None)
    if stream_getter is None:
        stream_getter = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if stream_getter is None:
        current_stream = torch_npu.npu.current_stream
        stream_getter = lambda device_id: current_stream(device_id).npu_stream

    # Preserve the producer ABI version, allocator, result conversion and
    # predecessor table. Only argument unwrapping and stream lookup differ.
    torch_exchange_argument_api = exchange_api[0]
    torch_exchange_argument_api.managed_tensor_from_py_object_no_sync = (
        torch_npu_tensor_from_py_object
    )
    torch_exchange_argument_api.dltensor_from_py_object_no_sync = (
        torch_npu_dltensor_from_py_object
    )
    torch_exchange_argument_api.current_work_stream = torch_npu_current_work_stream
    argument_capsule = PyCapsule_New(
        &torch_exchange_argument_api, DLPACK_EXCHANGE_CAPSULE, NULL,
    )
    _TorchNPUTensorArgument.__dlpack_c_exchange_api__ = argument_capsule
    torch_exchange_original_api = exchange_api
    torch_npu_stream_getter = stream_getter
    # FFI's type cache and returned containers may retain either API pointer.
    # The original callbacks and their owning capsule live with this module.
    torch_exchange_original_capsule = capsule
    return True


def is_torch_npu_stream_exchange_installed():
    """Return whether the private NPU argument bridge is initialized."""
    return torch_exchange_original_api != NULL


def _as_torch_npu_argument(tensor):
    """Internal bridge constructor; callers select NPU tensor arguments."""
    install_torch_npu_stream_exchange()
    return _TorchNPUTensorArgument(tensor)


def wrap_torch_npu_tensor_arguments(arguments):
    """Wrap each NPU tensor, including allocator anchors, without a data copy."""
    return tuple(
        _as_torch_npu_argument(value)
        if isinstance(value, torch.Tensor) and value.device.type == "npu"
        else value
        for value in arguments
    )
