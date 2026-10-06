// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include <tvm/tirx/stmt_functor.h>

#include "ascend_pipe.h"

namespace tvm {
namespace tl {
namespace ascend {

// Execution-core intent is an existing region/pipe/storage fact. It is not a
// shape-based ownership inference. Reuse it at both sides of the partition
// boundary, including handover-only mixed kernels without a GEMM call.
inline bool HasAscendCubeIntent(const tirx::Stmt &body) {
  bool cube = false;
  tirx::PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<tirx::SBlockNode>())
      cube |= block->name_hint == "CUBE";
    if (const auto *call = node.as<tirx::CallNode>()) {
      uint16_t pipe = GetAscendCallPipeMask(ffi::GetRef<tirx::Call>(call));
      cube |= (pipe &
               (PipeMask(ResourcePipe::kCube) | PipeMask(ResourcePipe::kMTE1) |
                PipeMask(ResourcePipe::kFixpipe))) != 0;
      if (IsAscendCopyCall(call)) {
        AscendCopy copy(call->args, call->annotations);
        cube |= IsL1Buffer(copy->src) || IsL1Buffer(copy->dst) ||
                IsL0ABuffer(copy->src) || IsL0ABuffer(copy->dst) ||
                IsL0BBuffer(copy->src) || IsL0BBuffer(copy->dst) ||
                IsL0CBuffer(copy->src) || IsL0CBuffer(copy->dst);
      }
    }
  });
  return cube;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
