"""NPU tuning context and fail-stop behavior, without launching a device."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from tilelang.autotuner import AutoTuner, set_autotune_inputs
from tilelang.autotuner.param import CompileArgs, ProfileArgs
from tilelang.autotuner.tuner import _BenchmarkWorkerState

tuner_module = importlib.import_module("tilelang.autotuner.tuner")


@dataclass(frozen=True)
class _Device:
    type: str
    index: int


class _Tensor:
    def __init__(self, device):
        self.device = device

    def to(self, *, device):
        return _Tensor(device)

    def clone(self):
        return _Tensor(self.device)


@pytest.fixture
def npu(monkeypatch):
    state = SimpleNamespace(current=5, entered=[], bound=[])

    @contextmanager
    def device_context(device):
        previous = state.current
        state.current = device.index
        state.entered.append(device)
        try:
            yield
        finally:
            state.current = previous

    def cuda_is_available():
        pytest.fail("Ascend tuning consulted CUDA availability")

    def set_device(device):
        state.current = device.index
        state.bound.append(device)

    monkeypatch.setattr(tuner_module, "torch", SimpleNamespace(
        Tensor=_Tensor,
        device=_Device,
        cuda=SimpleNamespace(is_available=cuda_is_available),
        npu=SimpleNamespace(is_available=lambda: True, current_device=lambda: state.current,
                            device_count=lambda: 8, device=device_context, set_device=set_device),
    ))
    return state


def _tuner():
    tuner = AutoTuner(lambda: None, configs=[{}])
    tuner.compile_args = CompileArgs(target="ascend")
    return tuner


def test_npu_resolution_preserves_current_or_explicit_logical_device(npu):
    tuner = _tuner()
    assert tuner._resolve_benchmark_devices(False, None, "ascend") == (False, [5])
    assert tuner._resolve_benchmark_devices(False, [3], "ascend") == (False, [3])
    with pytest.raises(ValueError, match="one explicitly selected"):
        tuner._resolve_benchmark_devices(True, [2, 3], "ascend")
    with pytest.raises(ValueError, match="Invalid logical NPU"):
        tuner._resolve_benchmark_devices(False, [8], "ascend")


@pytest.mark.parametrize("target", [
    {"kind": "ascend", "arch": "dav-2201"},
    "ascend -arch=dav-2201",
])
def test_npu_target_forms_do_not_fall_back_to_cuda(npu, target):
    tuner = _tuner()
    tuner.compile_args = CompileArgs(target=target)
    assert tuner._target_kind() == "ascend"
    assert tuner._current_benchmark_device() == _Device("npu", 5)
    assert tuner._resolve_benchmark_devices(False, [3], tuner._target_kind()) == (False, [3])


def test_captured_inputs_follow_worker_npu_and_preserve_scalars(npu):
    tuner = _tuner()
    original = _Tensor(_Device("npu", 1))
    with set_autotune_inputs(original, 0.25):
        tuner.set_profile_args()
    supply = tuner.profile_args.supply_prog
    first = supply([])
    assert first[0] is not original and first[0].device == _Device("npu", 5)
    assert first[1] == 0.25 and supply([]) is first
    with tuner_module.torch.npu.device(_Device("npu", 3)):
        second = supply([])
    assert second[0].device == _Device("npu", 3) and second is not first


@pytest.mark.parametrize("timeout", [0, 1])
def test_compile_caller_and_selected_benchmark_keep_separate_npu_bindings(npu, monkeypatch, timeout):
    tuner = _tuner()
    original = _Tensor(_Device("npu", 1))
    with set_autotune_inputs(original, 0.25):
        tuner.set_profile_args()
    assert tuner._resolve_benchmark_devices(False, [3], "ascend") == (False, [3])
    monkeypatch.setattr(tuner, "_resolve_num_compile_workers", lambda: 1)
    compiled = []

    def compile_program(**config):
        compiled.append((npu.current, config))
        return object()

    pool, futures, _, _ = tuner._prepare_compile_execution(
        config_args=[{"tile": 16}, {"tile": 32}], grouped_compile_active=False,
        group_compile_size=1, compile_func=compile_program, elaborate_func=lambda: None,
    )
    try:
        units = [future.result(timeout=1) for future in futures]
    finally:
        pool.shutdown(wait=True)
    assert compiled == [(5, {"tile": 16}), (5, {"tile": 32})]
    assert npu.bound == [_Device("npu", 5), _Device("npu", 5)]
    tasks, results = queue.Queue(), queue.Queue()
    start = threading.Event()
    for (idx, config, kernel, error), in units:
        assert error is None
        tasks.put((kernel, config, idx))
    tasks.put(None)
    observed = []

    def benchmark_target(**kwargs):
        supplied = tuner.profile_args.supply_prog([])
        observed.append((kwargs["benchmark_device"], npu.current, supplied))
        return 1.0, None

    start.set()
    tuner._benchmark_worker_loop(3, tasks, results, start, "ascend", benchmark_target,
                                 timeout, _BenchmarkWorkerState())
    assert [(device, current) for device, current, _ in observed] == [(_Device("npu", 3), 3)] * 2
    assert observed[0][2] is observed[1][2], "Captured inputs remain cached within the selected device"
    assert observed[0][2][0].device == _Device("npu", 3)
    assert observed[0][2][1] == 0.25
    assert npu.current == 5 and results.qsize() == 2
    assert all(results.get_nowait()[-2] is None for _ in range(2))


def test_npu_binding_failure_uses_the_same_fail_stop_result_path(npu, monkeypatch):
    tuner = _tuner()
    tasks, results = queue.Queue(), queue.Queue()
    start, stop = threading.Event(), threading.Event()
    tasks.put((object(), {}, 0))
    tasks.put((object(), {}, 1))

    @contextmanager
    def unavailable_scope(device):
        raise RuntimeError("NPU context rejected device")
        yield

    monkeypatch.setattr(tuner_module.torch.npu, "device", unavailable_scope)
    start.set()
    tuner._benchmark_worker_loop(3, tasks, results, start, "ascend",
                                 lambda **kwargs: pytest.fail("Benchmark ran outside a valid device scope"),
                                 0, _BenchmarkWorkerState(), stop)
    failure = results.get_nowait()
    assert failure[-2] == "error" and "NPU context rejected device" in failure[-1]
    assert stop.is_set() and tasks.qsize() == 1 and results.empty()


@pytest.mark.parametrize("timeout", [0, 1])
def test_cuda_worker_binding_warning_and_candidate_recovery_are_preserved(monkeypatch, timeout):
    state = SimpleNamespace(current=5, bound=[], entered=[])

    def set_device(ordinal):
        state.bound.append(ordinal)
        raise RuntimeError("Initial CUDA binding failed")

    @contextmanager
    def device_context(device):
        previous = state.current
        state.current = device.index
        state.entered.append(device)
        try:
            yield
        finally:
            state.current = previous

    monkeypatch.setattr(tuner_module, "torch", SimpleNamespace(
        device=_Device, cuda=SimpleNamespace(set_device=set_device, device=device_context),
    ))
    warnings = []
    monkeypatch.setattr(tuner_module.logger, "warning", lambda *args: warnings.append(args))
    tuner = AutoTuner(lambda: None, configs=[{}])
    tasks, results = queue.Queue(), queue.Queue()
    start, stop = threading.Event(), threading.Event()
    tasks.put((object(), {}, 0))
    tasks.put((object(), {}, 1))
    tasks.put(None)
    observed = []

    def benchmark_target(**kwargs):
        observed.append((kwargs["benchmark_device"], state.current))
        if len(observed) == 1:
            raise RuntimeError("CUDA candidate failed")
        return 1.0, None

    start.set()
    tuner._benchmark_worker_loop(3, tasks, results, start, "cuda", benchmark_target,
                                 timeout, _BenchmarkWorkerState(), stop)
    assert state.bound == [3] and warnings == [("Failed to bind benchmark worker to cuda:%s", 3)]
    assert observed == [(_Device("cuda", 3), 3)] * 2
    assert results.get_nowait()[-2] == "error" and results.get_nowait()[-2] is None
    assert not stop.is_set() and state.current == 5 and tasks.empty()


def test_cpu_worker_keeps_no_implicit_accelerator_binding(monkeypatch):
    monkeypatch.setattr(tuner_module, "torch", SimpleNamespace())
    tuner = AutoTuner(lambda: None, configs=[{}])
    tasks, results = queue.Queue(), queue.Queue()
    start = threading.Event()
    tasks.put((object(), {}, 0))
    tasks.put(None)

    def benchmark_target(**kwargs):
        assert kwargs["benchmark_device"] is None
        return 1.0, None

    start.set()
    tuner._benchmark_worker_loop(0, tasks, results, start, "c", benchmark_target,
                                 0, _BenchmarkWorkerState())
    assert results.get_nowait()[-2] is None


@pytest.mark.parametrize("timeout", [0, 1])
def test_npu_failure_stops_before_next_queued_candidate(npu, timeout):
    tuner = _tuner()
    tasks, results = queue.Queue(), queue.Queue()
    start, stop = threading.Event(), threading.Event()
    tasks.put((object(), {}, 0))
    tasks.put((object(), {}, 1))
    calls = []

    def benchmark_target(**kwargs):
        calls.append(kwargs["benchmark_device"])
        assert npu.current == 3
        raise RuntimeError("device execution failed")

    start.set()
    tuner._benchmark_worker_loop(3, tasks, results, start, "ascend", benchmark_target,
                                 timeout, _BenchmarkWorkerState(), stop)
    assert calls == [_Device("npu", 3)] and stop.is_set()
    result = results.get_nowait()
    assert result[0] == 0 and result[-2] == "error" and "device execution failed" in result[-1]
    assert results.empty() and tasks.qsize() == 1


def test_npu_timeout_stops_before_next_queued_candidate(npu):
    tuner = _tuner()
    tasks, results = queue.Queue(), queue.Queue()
    start, stop, release = threading.Event(), threading.Event(), threading.Event()
    tasks.put((object(), {}, 0))
    tasks.put((object(), {}, 1))

    def benchmark_target(**kwargs):
        release.wait(timeout=1)
        return 1.0, None

    start.set()
    try:
        tuner._benchmark_worker_loop(3, tasks, results, start, "ascend", benchmark_target,
                                     0.01, _BenchmarkWorkerState(), stop)
        assert stop.is_set() and results.get_nowait()[-2] == "timeout"
        assert tasks.qsize() == 1
    finally:
        release.set()


def test_benchmark_passes_explicit_device_to_supply_validation_and_both_timings(npu):
    tuner = _tuner()
    tuner.profile_args = ProfileArgs(ref_prog=lambda: None, cache_input_tensors=False)
    calls = []

    class Profiler:
        def _get_inputs(self, **kwargs):
            calls.append(("supply", kwargs["device"]))
            return []

        def assert_allclose(self, *args, **kwargs):
            calls.append(("check", kwargs["device"]))

        def do_bench(self, *args, **kwargs):
            calls.append(("bench", kwargs["device"]))
            return 1.0

    kernel = SimpleNamespace(get_profiler=lambda **kwargs: Profiler())
    device = _Device("npu", 3)
    assert tuner._benchmark_target(kernel, 1, 1, 2.0, _BenchmarkWorkerState(), device) == (1.0, 1.0)
    assert calls == [(kind, device) for kind in ("supply", "check", "bench", "supply", "bench")]


def test_explicit_cpu_supply_and_correctness_do_not_touch_cuda(monkeypatch):
    from tilelang.engine.param import KernelParam
    from tilelang import tvm
    from tilelang.profiler import Profiler, TensorSupplyType
    from tilelang.utils.tensor import get_tensor_supply
    from tilelang.utils import device as device_utils

    def unexpected_default():
        pytest.fail("Explicit input supply consulted the default device")

    monkeypatch.setattr(device_utils, "get_current_device", unexpected_default)
    tensor = get_tensor_supply(TensorSupplyType.One, device=torch.device("cpu"))(
        KernelParam(tvm.DataType("float32"), [4]))
    assert tensor.device.type == "cpu"
    profiler = Profiler([], [], TensorSupplyType.One).with_default_adapter(lambda x: x + 1)
    profiler.assert_allclose(lambda x: x + 1, input_tensors=[tensor])
