"""Tests for auto-tuner cache-key identity."""

import inspect
import threading
from dataclasses import replace

import pytest

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs, ProfileArgs
from tilelang.cache.kernel_cache import KernelCache


def _kernel(block_size=128):
    return block_size


def _cache_key(compile_args=None, profile_args=None):
    tuner = AutoTuner(_kernel, configs=[{"block_size": 128}])
    tuner.compile_args = compile_args or CompileArgs()
    tuner.profile_args = profile_args or ProfileArgs()
    return tuner.generate_cache_key(inspect.signature(_kernel).parameters, {})


def test_cache_key_includes_output_indices():
    base_args = CompileArgs(out_idx=[0])

    assert _cache_key(compile_args=base_args) != _cache_key(compile_args=replace(base_args, out_idx=[1]))


def test_cache_key_includes_profile_validation_and_input_behavior():
    base_args = ProfileArgs(skip_check=False, cache_input_tensors=False)

    variants = (
        replace(base_args, skip_check=True),
        replace(base_args, cache_input_tensors=True),
    )

    base_key = _cache_key(profile_args=base_args)
    assert all(base_key != _cache_key(profile_args=variant) for variant in variants)


def test_cache_key_is_disabled_for_profile_callbacks():
    lock = threading.Lock()

    def callback(value):
        with lock:
            return value

    for callback_field in ("ref_prog", "supply_prog", "manual_check_prog"):
        assert _cache_key(profile_args=ProfileArgs(**{callback_field: callback})) is None


def test_ascend_winner_key_tracks_native_build_at_the_same_package_version(monkeypatch):
    build = {"stamp": "libtilelang.so:build-a|libtvm_compiler.so:compiler-a"}
    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", staticmethod(lambda: build["stamp"]))
    compile_args = CompileArgs(target={"kind": "ascend", "arch": "dav-2201"}, execution_backend="tvm_ffi")

    original = _cache_key(compile_args=compile_args)
    assert original is not None
    assert _cache_key(compile_args=compile_args) == original
    build["stamp"] = "libtilelang.so:build-b|libtvm_compiler.so:compiler-a"
    assert _cache_key(compile_args=compile_args) != original
    build["stamp"] = "libtilelang.so:build-a|libtvm_compiler.so:compiler-b"
    assert _cache_key(compile_args=compile_args) != original


@pytest.mark.parametrize("missing", [None, "", OSError("unreadable native library")])
def test_ascend_unknown_native_build_disables_winner_cache(monkeypatch, missing):
    def stamp():
        if isinstance(missing, OSError):
            raise missing
        return missing

    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", staticmethod(stamp))
    assert _cache_key(compile_args=CompileArgs(target="ascend -arch=dav-2201")) is None


def test_ascend_profile_callbacks_still_disable_cache_before_build_identity(monkeypatch):
    def unexpected_stamp():
        pytest.fail("An uncacheable callback must not require native build identity")

    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", staticmethod(unexpected_stamp))
    for field in ("ref_prog", "supply_prog", "manual_check_prog"):
        assert _cache_key(compile_args=CompileArgs(target="ascend"),
                          profile_args=ProfileArgs(**{field: lambda: None})) is None


def test_other_backends_keep_existing_winner_key_without_native_stamp(monkeypatch):
    def unexpected_stamp():
        pytest.fail("This cache-key change is scoped to Ascend")

    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", staticmethod(unexpected_stamp))
    for target in ("auto", "cuda", "llvm"):
        compile_args = CompileArgs(target=target)
        assert _cache_key(compile_args=compile_args) is not None
        assert _cache_key(compile_args=compile_args) == _cache_key(compile_args=compile_args)
