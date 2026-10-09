// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include <tvm/tirx/stmt_functor.h>

#include "ascend_pipe.h"
#include "attr.h"
#include "auto_schedule/task_annotations.h"

namespace tvm {
namespace tl {
namespace ascend {

namespace mixed_kernel_intent_detail {

class CoreIntentCollector : public tirx::StmtExprVisitor {
public:
  static bool HasCube(const tirx::Stmt &body) {
    CoreIntentCollector collector;
    collector(body);
    return collector.cube_;
  }

private:
  void VisitStmt_(const tirx::SBlockNode *op) final {
    CoreMask saved = owner_;
    if (op->name_hint == "CUBE") {
      owner_ = kCoreCube;
      cube_ = true;
    } else if (op->name_hint == "VECTOR") {
      owner_ = kCoreVector;
    }
    // Never prune a Vector region: a real Cube instruction remains a Cube
    // requirement, and the existing core validator diagnoses that conflict.
    tirx::StmtExprVisitor::VisitStmt_(op);
    owner_ = saved;
  }

  void VisitStmt_(const tirx::AttrStmtNode *op) final {
    CoreMask saved = owner_;
    if (op->attr_key == tl::attr::kAscendTask ||
        op->attr_key == tl::attr::kAscendPerCoreTask) {
      auto core = ParseTaskMetadata(op->node).core_mask;
      if (core.has_value() && IsConcreteCore(core.value())) {
        owner_ = core.value();
        cube_ |= owner_ == kCoreCube;
      }
    }
    tirx::StmtExprVisitor::VisitStmt_(op);
    owner_ = saved;
  }

  void VisitExpr_(const tirx::CallNode *op) final {
    tirx::Call call = ffi::GetRef<tirx::Call>(op);
    // Sync endpoint pipes describe an existing owner's protocol, rather than
    // new data execution resources. This shared classifier excludes them.
    cube_ |= HasCore(GetAscendCallCoreRequirements(call), kCoreCube);
    if (owner_ == kCoreUnassigned &&
        (op->op.same_as(ascend_cross_core_set_flag()) ||
         op->op.same_as(ascend_cross_core_wait_flag()))) {
      // AssignCore pins an unowned cross-core endpoint when its one pipe
      // identifies a core. Broad/shared pipe masks supply no such fact.
      uint16_t endpoint = GetAscendCallPipeMask(call);
      if (IsSingleResourcePipe(endpoint))
        cube_ |= GetAscendExclusivePipeCoreMask(
                     static_cast<ResourcePipe>(endpoint)) == kCoreCube;
    }
    tirx::StmtExprVisitor::VisitExpr_(op);
  }

  CoreMask owner_ = kCoreUnassigned;
  bool cube_ = false;
};

} // namespace mixed_kernel_intent_detail

// A concrete ownership region, an operation/storage requirement, or an
// unowned cross-core endpoint with a unique execution core establishes intent.
// Reuse these facts on both sides of the mixed-partition boundary.
inline bool HasAscendCubeIntent(const tirx::Stmt &body) {
  return mixed_kernel_intent_detail::CoreIntentCollector::HasCube(body);
}

} // namespace ascend
} // namespace tl
} // namespace tvm
