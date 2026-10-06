/**
 * \file ascendcl.cc
 * \brief Implementation of the CANN runtime (libascendcl) stub library.
 *
 * This implements lazy loading of libascendcl.so and provides exported global
 * wrapper functions that serve as drop-in replacements for the ACL runtime
 * entrypoints used by TileLang's Ascend runtime module.
 *
 * The implementation mirrors src/rocm/stubs/hip.cc:
 * - Resolve symbols via dlopen/dlsym on first use.
 * - Prefer RTLD_DEFAULT/RTLD_NEXT when CANN is already loaded by another
 *   framework (e.g. torch_npu).
 * - Fall back to well-known CANN install roots when LD_LIBRARY_PATH is not
 *   set up (no set_env.sh sourced).
 */

#include "ascendcl.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "ascendcl_stub is POSIX-only (requires <dlfcn.h> / dlopen). "            \
    "The Ascend backend does not support Windows; configure with "             \
    "-DTILELANG_USE_ASCEND_STUBS=OFF."
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#if defined(__linux__)
#include <link.h>
#endif

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace tvm::tl::ascendcl {

namespace {

// Library handles and the resolved callbacks deliberately live until process
// exit. Module destruction can still call ACL after other Python extensions
// release their dlopen references.
struct LoadedAscendCL {
  AscendCLAPI api{};
  void *handle{nullptr};
  bool library_found{false};
  std::string missing_symbol;
  std::vector<void *> retained_libraries;
};

void *StubModuleBase() {
  Dl_info info{};
  dladdr(reinterpret_cast<void *>(&StubModuleBase), &info);
  return info.dli_fbase;
}

void *ExternalSymbol(void *scope, const char *name) {
  (void)dlerror();
  void *symbol = dlsym(scope, name);
  if (dlerror() != nullptr || symbol == nullptr) {
    return nullptr;
  }
  Dl_info info{};
  if (dladdr(symbol, &info) == 0 || info.dli_fbase == StubModuleBase()) {
    return nullptr;
  }
  // Compare DSO identities, not &aclrtGetDevice: ELF symbol preemption can
  // make the address of our wrapper refer to the real library or a profiler.
  return symbol;
}

std::vector<std::string> LibAscendCLPathCandidates() {
  std::vector<std::string> candidates;
  // Honors LD_LIBRARY_PATH (set by CANN's set_env.sh).
  candidates.emplace_back("libascendcl.so");

  std::vector<std::string> roots;
  for (const char *env : {"ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"}) {
    const char *value = std::getenv(env);
    if (value != nullptr && value[0] != '\0') {
      roots.emplace_back(value);
    }
  }
  roots.emplace_back("/usr/local/Ascend/ascend-toolkit/latest");

  for (const std::string &root : roots) {
    candidates.push_back(root + "/lib64/libascendcl.so");
    candidates.push_back(root + "/runtime/lib64/libascendcl.so");
  }
  return candidates;
}

void *LoadRuntimeLibrary() {
  for (const std::string &path : LibAscendCLPathCandidates()) {
    void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      return handle;
    }
  }
  return nullptr;
}

#if defined(__linux__)
struct LoadedSymbolSearch {
  const char *name;
  void *symbol{nullptr};
  void *retained_library{nullptr};
};

int SearchLoadedLibrary(dl_phdr_info *info, size_t, void *data) {
  if (info->dlpi_name == nullptr || info->dlpi_name[0] == '\0') {
    return 0;
  }
  auto *search = static_cast<LoadedSymbolSearch *>(data);
  void *handle = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
  if (handle == nullptr) {
    return 0;
  }
  search->symbol = ExternalSymbol(handle, search->name);
  if (search->symbol != nullptr) {
    search->retained_library = handle;
    return 1;
  }
  dlclose(handle);
  return 0;
}
#endif

template <typename T> T GetSymbol(LoadedAscendCL *library, const char *name) {
  // Resolve each entrypoint separately. A profiler may interpose only launch
  // while the other callbacks come from libascendcl; its launch hook must stay
  // in the call chain even when the stub precedes libascendcl in load order.
  void *symbol = ExternalSymbol(RTLD_DEFAULT, name);
  if (symbol == nullptr) {
    symbol = ExternalSymbol(RTLD_NEXT, name);
  }
  void *retained = nullptr;
#if defined(__linux__)
  if (symbol == nullptr) {
    // RTLD_NEXT uses the caller's lookup scope, which may exclude DSOs that
    // were dlopened afterwards. Visit already-loaded modules in loader order
    // before opening CANN directly, preserving a later profiler's launch hook.
    LoadedSymbolSearch search{name};
    dl_iterate_phdr(SearchLoadedLibrary, &search);
    symbol = search.symbol;
    retained = search.retained_library;
  }
#endif
  if (symbol == nullptr) {
    if (library->handle == nullptr) {
      library->handle = LoadRuntimeLibrary();
    }
    if (library->handle != nullptr) {
      symbol = ExternalSymbol(library->handle, name);
    }
  }
  if (symbol == nullptr) {
    return nullptr;
  }

  if (retained == nullptr) {
    Dl_info info{};
    dladdr(symbol, &info);
    retained = dlopen(info.dli_fname, RTLD_LAZY | RTLD_LOCAL);
  }
  if (retained == nullptr) {
    return nullptr;
  }
  library->retained_libraries.push_back(retained);
  return reinterpret_cast<T>(symbol);
}

LoadedAscendCL *LoadAscendCL() {
  auto *library = new LoadedAscendCL();
  // RTLD_DEFAULT is a valid lookup scope with the value nullptr on Linux.
  // Record availability explicitly instead of using the scope as a boolean.
  library->api.aclrtGetDevice_ =
      GetSymbol<decltype(library->api.aclrtGetDevice_)>(library,
                                                        "aclrtGetDevice");
  library->library_found =
      library->api.aclrtGetDevice_ != nullptr || library->handle != nullptr;
  if (!library->library_found) {
    return library;
  }

// Required symbols: record the first missing name (older CANN releases may
// lack e.g. aclrtLaunchKernelWithHostArgs) rather than leaving a usable partial
// API table. get_or_null() must remain safe on error-message paths.
#define LOOKUP(symbol)                                                         \
  if (library->api.symbol##_ == nullptr) {                                     \
    library->api.symbol##_ =                                                   \
        GetSymbol<decltype(library->api.symbol##_)>(library, #symbol);         \
  }                                                                            \
  if (library->api.symbol##_ == nullptr) {                                     \
    library->missing_symbol = #symbol;                                         \
    library->api = AscendCLAPI{};                                              \
    return library;                                                            \
  }

  LOOKUP(aclrtBinaryLoadFromData)
  LOOKUP(aclrtBinaryGetFunction)
  LOOKUP(aclrtBinaryUnLoad)
  LOOKUP(aclrtGetDevice)
  LOOKUP(aclrtLaunchKernelWithHostArgs)
#undef LOOKUP

  // Optional: wrappers return nullptr when absent.
  library->api.aclGetRecentErrMsg_ =
      GetSymbol<decltype(library->api.aclGetRecentErrMsg_)>(
          library, "aclGetRecentErrMsg");
  return library;
}

LoadedAscendCL *GetLoadedAscendCL() {
  static LoadedAscendCL *library = LoadAscendCL();
  return library;
}

} // namespace

void *AscendCLAPI::get_handle() { return GetLoadedAscendCL()->handle; }

bool AscendCLAPI::is_available() {
  LoadedAscendCL *library = GetLoadedAscendCL();
  return library->library_found && library->missing_symbol.empty();
}

AscendCLAPI *AscendCLAPI::get_or_null() {
  return is_available() ? &GetLoadedAscendCL()->api : nullptr;
}

AscendCLAPI *AscendCLAPI::get() {
  LoadedAscendCL *library = GetLoadedAscendCL();
  if (!library->library_found) {
    throw std::runtime_error(
        "CANN runtime library (libascendcl.so) not found. Install the Ascend "
        "CANN toolkit and source its set_env.sh (or import torch_npu first) "
        "before using TileLang's Ascend backend.");
  }
  if (!library->missing_symbol.empty()) {
    throw std::runtime_error(
        "libascendcl.so was found but does not provide the required symbol `" +
        library->missing_symbol +
        "`. TileLang's Ascend backend requires a newer CANN toolkit.");
  }
  return &library->api;
}

} // namespace tvm::tl::ascendcl

// ============================================================================
// Global wrapper function implementations
// ============================================================================

using tvm::tl::ascendcl::AscendCLAPI;

extern "C" {

int32_t aclrtBinaryLoadFromData(const void *data, size_t size,
                                const void *options, void **binHandle) {
  return AscendCLAPI::get()->aclrtBinaryLoadFromData_(data, size, options,
                                                      binHandle);
}

int32_t aclrtBinaryGetFunction(void *binHandle, const char *name,
                               void **funcHandle) {
  return AscendCLAPI::get()->aclrtBinaryGetFunction_(binHandle, name,
                                                     funcHandle);
}

int32_t aclrtBinaryUnLoad(void *binHandle) {
  return AscendCLAPI::get()->aclrtBinaryUnLoad_(binHandle);
}

int32_t aclrtGetDevice(int32_t *deviceId) {
  return AscendCLAPI::get()->aclrtGetDevice_(deviceId);
}

int32_t aclrtLaunchKernelWithHostArgs(void *func, uint32_t numBlocks,
                                      void *stream, void *config, void *args,
                                      size_t argsSize, void *reserved,
                                      size_t reservedSize) {
  return AscendCLAPI::get()->aclrtLaunchKernelWithHostArgs_(
      func, numBlocks, stream, config, args, argsSize, reserved, reservedSize);
}

const char *aclGetRecentErrMsg(void) {
  // Error-message path: must never throw, callers handle nullptr.
  AscendCLAPI *api = AscendCLAPI::get_or_null();
  if (api == nullptr || api->aclGetRecentErrMsg_ == nullptr) {
    return nullptr;
  }
  return api->aclGetRecentErrMsg_();
}

} // extern "C"
