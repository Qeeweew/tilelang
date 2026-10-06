"""CPU-only tests of the product Torch NPU argument/stream bridge.

The fake producer changes only DLPack device metadata and the stream getter;
all data, conversion, allocation and ownership remain real CPU Torch objects.
Each subprocess isolates FFI's process/thread lifetime argument-type cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import tilelang  # noqa: F401 - expose the bundled native-extension directory


HARNESS = r'''
import ctypes as C
import gc
import sys
import types
import weakref

import torch
import tvm_ffi

class Version(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32)]

class Header(C.Structure):
    _fields_ = [("version", Version), ("prev_api", C.c_void_p)]

class API(C.Structure):
    _fields_ = [
        ("header", Header), ("allocator", C.c_void_p),
        ("from_py", C.c_void_p), ("to_py", C.c_void_p),
        ("tensor_from_py", C.c_void_p), ("stream", C.c_void_p),
    ]

class Device(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]

class DType(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]

class Tensor(C.Structure):
    _fields_ = [
        ("data", C.c_void_p), ("device", Device), ("ndim", C.c_int),
        ("dtype", DType), ("shape", C.POINTER(C.c_int64)),
        ("strides", C.POINTER(C.c_int64)), ("byte_offset", C.c_uint64),
    ]

class Managed(C.Structure):
    _fields_ = [
        ("version", Version), ("manager_ctx", C.c_void_p),
        ("deleter", C.c_void_p), ("flags", C.c_uint64), ("tensor", Tensor),
    ]

get_ptr = C.pythonapi.PyCapsule_GetPointer
get_ptr.argtypes = [C.py_object, C.c_char_p]
get_ptr.restype = C.c_void_p
new_capsule = C.pythonapi.PyCapsule_New
new_capsule.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
new_capsule.restype = C.py_object
original_capsule = torch.Tensor.__dlpack_c_exchange_api__
original = API.from_address(get_ptr(original_capsule, b"dlpack_exchange_api"))
FROM = C.PYFUNCTYPE(C.c_int, C.py_object, C.POINTER(C.c_void_p))
original_from = FROM(original.from_py)
fake_npu = False
converted = []

@FROM
def producer_from(value, out):
    converted.append(value.data_ptr())
    result = original_from(value, out)
    if result == 0 and fake_npu:
        # No NPU access: the consumer below only reads its stream TLS.
        managed = Managed.from_address(out[0])
        managed.tensor.device = Device(12, 3)
    return result

producer = API.from_buffer_copy(original)
producer.from_py = C.cast(producer_from, C.c_void_p).value
producer_capsule = new_capsule(C.addressof(producer), b"dlpack_exchange_api", None)
echo = tvm_ffi.get_global_func("testing.echo")
warmed = sys.argv[1] == "warm"
if warmed:
    # This permanently caches the original Torch type's Exchange API.
    assert torch.equal(echo(torch.arange(4)), torch.arange(4))
# The product loader exposes its bundled native-extension directory. Import
# it after the warm-cache step to retain the cache-before-TileLang regression.
import tilelang
torch.Tensor.__dlpack_c_exchange_api__ = producer_capsule

streams = {3: 0x1230}
queries = []
def current_stream(device):
    queries.append(device)
    return streams[device]
sys.modules["torch_npu"] = types.SimpleNamespace(
    _C=types.SimpleNamespace(_npu_getCurrentRawStream=current_stream),
)
import tilelang_ascend_npu_exchange as exchange
assert exchange.install_torch_npu_stream_exchange()
assert not exchange.install_torch_npu_stream_exchange()
assert torch.Tensor.__dlpack_c_exchange_api__ is producer_capsule

x = torch.arange(17, dtype=torch.float32)
wrapped = exchange._as_torch_npu_argument(x)
bridge = API.from_address(get_ptr(type(wrapped).__dlpack_c_exchange_api__, b"dlpack_exchange_api"))
assert bridge.header.version.major == producer.header.version.major
assert bridge.header.version.minor == producer.header.version.minor
assert bridge.header.prev_api == producer.header.prev_api
assert bridge.allocator == producer.allocator and bridge.to_py == producer.to_py
result = echo(wrapped)
assert type(result) is torch.Tensor and result.data_ptr() == x.data_ptr()
assert converted == [x.data_ptr()]
assert torch.equal(result, x)
ref = weakref.ref(x)
del x
gc.collect()
assert ref() is wrapped.tensor
del wrapped
gc.collect()
assert torch.equal(result, torch.arange(17, dtype=torch.float32))

# Returned containers also retain the producer output converter, rather than
# leaking the private argument wrapper into user-visible results.
array = tvm_ffi.get_global_func("testing.make_array_with_mixed")(
    exchange._as_torch_npu_argument(result), 7,
)
assert type(array[0]) is torch.Tensor and array[0].data_ptr() == result.data_ptr()

allocator = tvm_ffi.LIB.TVMFFIEnvGetDLPackManagedTensorAllocator
allocator.restype = C.c_void_p
dev = tvm_ffi.device("ext_dev", 3)
seen = []
def observe(tensor):
    seen.append((tvm_ffi.get_raw_stream(dev), allocator()))
    return 0
observe = tvm_ffi.convert_func(observe)
fake_npu = True
argument = exchange._as_torch_npu_argument(result)
observe(argument)
streams[3] = 0x4560
observe(argument)
fake_npu = False
assert queries == [3, 3], queries
assert seen == [(0x1230, producer.allocator), (0x4560, producer.allocator)], seen
assert tvm_ffi.get_raw_stream(dev) == 0

# A scalar-only kernel's zero-element allocator anchor is still a producer
# tensor and can establish the same stream/allocator call context.
fake_npu = True
observe(exchange._as_torch_npu_argument(torch.empty(0)))
fake_npu = False
assert seen[-1] == (0x4560, producer.allocator)

# The product wrapper selector never wraps CPU/CUDA tensors or scalar values.
cpu = torch.empty(0)
selected = exchange.wrap_torch_npu_tensor_arguments((cpu, 17, None))
assert selected[0] is cpu and selected[1:] == (17, None)

class NPUTensor(torch.Tensor):
    @property
    def device(self):
        return types.SimpleNamespace(type="npu", index=3)
npu = torch.empty(0).as_subclass(NPUTensor)
selected = exchange.wrap_torch_npu_tensor_arguments((npu, cpu, npu))
assert selected[0].tensor is npu and selected[1] is cpu and selected[2].tensor is npu
assert torch.Tensor.__dlpack_c_exchange_api__ is producer_capsule
print("PRIVATE_NPU_BRIDGE_PASS", sys.argv[1])
'''


@pytest.mark.parametrize("order", ["cold", "warm"])
def test_private_npu_arguments_preserve_stream_allocator_and_ownership(order):
    pytest.importorskip("tilelang_ascend_npu_exchange")
    result = subprocess.run(
        [sys.executable, "-c", HARNESS, order],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ASCEND_RT_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRIVATE_NPU_BRIDGE_PASS " + order in result.stdout


@pytest.mark.parametrize("scalar_only", [False, True])
def test_product_adapter_prepares_input_and_zero_tensor_allocator_anchor(monkeypatch, scalar_only):
    import torch
    from tilelang.jit.adapter import tvm_ffi as adapter_module

    class NPUTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu", index=3)

    tensor = torch.empty(0).as_subclass(NPUTensor)
    device = tensor.device
    adapter = adapter_module.TVMFFIKernelAdapter.__new__(adapter_module.TVMFFIKernelAdapter)
    adapter.params, adapter.result_idx = [object(), object()], [1]
    adapter._ffi_callee_allocated_output_abi = True
    adapter.get_current_device_functor = lambda: lambda: device
    prepared, calls, allocations, installs = [], [], [], []
    expected = torch.tensor([42])
    adapter._get_executable = lambda: lambda *args: calls.append(args) or expected
    monkeypatch.setattr(adapter_module, "_install_torch_stream_exchange", lambda: installs.append(True))

    def prepare(arguments):
        prepared.append(arguments)
        return tuple(SimpleNamespace(tensor=value) if isinstance(value, NPUTensor) else value for value in arguments)

    def empty(*shape, **kwargs):
        allocations.append((shape, kwargs))
        assert shape == (0,) and kwargs == {"device": device}
        return tensor

    monkeypatch.setattr(adapter_module, "_prepare_torch_arguments", prepare)
    monkeypatch.setattr(torch, "empty", empty)
    value = 17 if scalar_only else tensor
    assert adapter._convert_torch_func()(value) is expected
    assert prepared == [(value, tensor)]
    assert calls[0][-1].tensor is tensor
    if scalar_only:
        assert calls[0][0] == 17 and len(allocations) == 1
    else:
        assert calls[0][0].tensor is tensor and not allocations
    assert installs == [True]
