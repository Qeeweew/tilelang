// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

enum class KernelExecutionKind { kVector, kCube, kMix };

struct KernelNativeUsage {
  bool gemm_l0{false};
  bool gemm_l1{false};
  bool nd2nz_copy{false};
  bool nd2nz_post_copy{false};
  bool gm_bypass_dcache{false};
  bool philox_rng{false};
  bool cooperative_groups{false};
};

// One validated final-function snapshot. SID declarations are projected into
// their emitted C++ function scopes, while all of them share this physical
// kernel's execution domain. Cube-only legacy SID extents remain unchanged.
// Borrowed VF keys expire with the input function;
// AddFunction replaces the complete value before emitting another entry.
struct KernelExecutionTopology {
  KernelExecutionKind kind{KernelExecutionKind::kVector};
  int active_aiv_count{1};
  int physical_aiv_count{1};
  std::vector<tirx::Var> kernel_sid_bindings;
  std::unordered_map<const tirx::SBlockNode *, std::vector<tirx::Var>>
      vf_sid_bindings;
  KernelNativeUsage native;

  const std::vector<tirx::Var> &
  BindingsForVf(const tirx::SBlockNode *block) const {
    auto it = vf_sid_bindings.find(block);
    TVM_FFI_ICHECK(it != vf_sid_bindings.end())
        << "Ascend VF body is outside the selected kernel execution topology";
    return it->second;
  }
};

} // namespace codegen
} // namespace tvm
