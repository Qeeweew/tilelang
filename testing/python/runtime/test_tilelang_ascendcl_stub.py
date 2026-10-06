"""Exercise ACL loading and ELF interposition without a CANN install or NPU."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the ACL stub uses Linux ELF loading")


RUNTIME_SOURCE = r"""
#include "ascendcl.h"
extern "C" {
int32_t aclrtBinaryLoadFromData(const void *, size_t, const void *, void **out) {
  *out = reinterpret_cast<void *>(17);
  return 0;
}
int32_t aclrtBinaryGetFunction(void *, const char *, void **out) {
  *out = reinterpret_cast<void *>(19);
  return 0;
}
int32_t aclrtBinaryUnLoad(void *) { return 0; }
int32_t aclrtGetDevice(int32_t *out) { *out = 23; return 0; }
#ifndef OMIT_LAUNCH
int32_t aclrtLaunchKernelWithHostArgs(void *, uint32_t, void *, void *, void *,
                                    size_t, void *, size_t) { return 43; }
#endif
#ifndef OMIT_ERROR
const char *aclGetRecentErrMsg() { return "fake ACL error"; }
#endif
}
"""

PROFILER_SOURCE = r"""
#include "ascendcl.h"
extern "C" int32_t aclrtLaunchKernelWithHostArgs(
    void *, uint32_t, void *, void *, void *, size_t, void *, size_t) {
  return 87;
}
"""

HARNESS_SOURCE = r"""
#include "ascendcl.h"
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

void *Load(const char *path, int flags) {
  void *handle = dlopen(path, RTLD_NOW | flags);
  if (!handle) throw std::runtime_error(dlerror());
  return handle;
}

template <typename T> T Symbol(void *handle, const char *name) {
  void *symbol = dlsym(handle, name);
  if (!symbol) throw std::runtime_error(dlerror());
  return reinterpret_cast<T>(symbol);
}

int main(int argc, char **argv) {
  if (argc != 6) return 1;
  const std::string mode = argv[1];
  std::vector<void *> providers;
  void *stub = nullptr;
  try {
    if (mode == "preloaded" || mode == "local") {
      providers.push_back(Load(argv[3], mode == "local" ? RTLD_LOCAL : RTLD_GLOBAL));
    }
    if (mode == "profiler_first") {
      providers.push_back(Load(argv[4], RTLD_GLOBAL));
      providers.push_back(Load(argv[3], RTLD_GLOBAL));
    }
    stub = Load(argv[2], RTLD_GLOBAL);
    if (mode == "profiler_after") providers.push_back(Load(argv[4], RTLD_GLOBAL));
    if (mode == "after" || mode == "profiler_after" || mode == "missing") {
      providers.push_back(Load(argv[3], RTLD_GLOBAL));
    }

    auto device = Symbol<decltype(&aclrtGetDevice)>(stub, "aclrtGetDevice");
    auto launch = Symbol<decltype(&aclrtLaunchKernelWithHostArgs)>(stub, "aclrtLaunchKernelWithHostArgs");
    auto error = Symbol<decltype(&aclGetRecentErrMsg)>(stub, "aclGetRecentErrMsg");
    int32_t value = -1;
    if (mode == "missing") {
      try {
        device(&value);
      } catch (const std::runtime_error &failure) {
        return std::string(failure.what()).find("aclrtLaunchKernelWithHostArgs") != std::string::npos &&
               error() == nullptr ? 0 : 2;
      }
      return 3;
    }
    if (device(&value) != 0 || value != 23) return 4;

    // A framework or profiler can release its own references after the API
    // table is initialized. The stub's retained callbacks must stay callable.
    for (void *provider : providers) dlclose(provider);
    int expected = (mode == "profiler_first" || mode == "profiler_after") ? 87 : 43;
    if (launch(nullptr, 1, nullptr, nullptr, nullptr, 0, nullptr, 0) != expected) return 5;
    const char *message = error();
    bool has_error = std::string(argv[5]) == "error";
    if (has_error ? message == nullptr || std::string(message) != "fake ACL error" : message != nullptr) return 6;
  } catch (const std::exception &) {
    return 7;
  }
  return 0;
}
"""


@pytest.fixture(scope="module")
def acl_libraries(tmp_path_factory):
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a host C++ compiler is required")
    root = Path(__file__).resolve().parents[3]
    source = root / "src" / "ascend" / "stubs"
    if not (source / "ascendcl.cc").exists():
        pytest.skip("ACL stub sources are required")
    work = tmp_path_factory.mktemp("ascendcl")

    def build(name, code, flags, defines=()):
        input_path = work / f"{name}.cc"
        input_path.write_text(code)
        output = work / name
        subprocess.run(
            [*compiler, "-std=c++17", "-I", str(source), *flags, *defines, str(input_path), "-o", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        return output

    stub = work / "libstub_ascendcl.so"
    subprocess.run(
        [*compiler, "-std=c++17", "-shared", "-fPIC", str(source / "ascendcl.cc"), "-ldl", "-o", str(stub)],
        check=True,
        capture_output=True,
        text=True,
    )
    profiler = build("libprofiler.so", PROFILER_SOURCE, ["-shared", "-fPIC"])
    harness = build("harness", HARNESS_SOURCE, ["-ldl"])
    runtimes = {}
    for variant, defines in [("complete", ()), ("no_error", ("-DOMIT_ERROR",)), ("missing", ("-DOMIT_LAUNCH",))]:
        output = build(f"runtime_{variant}.so", RUNTIME_SOURCE, ["-shared", "-fPIC"], defines)
        directory = work / variant
        directory.mkdir()
        runtime = directory / "libascendcl.so"
        output.rename(runtime)
        runtimes[variant] = runtime
    return stub, profiler, harness, runtimes


@pytest.mark.parametrize("mode", ["preloaded", "after", "local", "fallback", "profiler_first", "profiler_after"])
def test_acl_load_order_and_callback_lifetime(acl_libraries, mode):
    run_acl_harness(acl_libraries, mode, "complete")


@pytest.mark.parametrize("variant", ["no_error", "missing"])
def test_acl_missing_symbols(acl_libraries, variant):
    run_acl_harness(acl_libraries, "missing" if variant == "missing" else "after", variant)


def run_acl_harness(acl_libraries, mode, variant):
    stub, profiler, harness, runtimes = acl_libraries
    runtime = runtimes[variant]
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(runtime.parent)
    result = subprocess.run(
        [str(harness), mode, str(stub), str(runtime), str(profiler), "no_error" if variant == "no_error" else "error"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
