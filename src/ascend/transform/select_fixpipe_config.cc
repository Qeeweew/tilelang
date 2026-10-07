// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file select_fixpipe_config.cc
 * \brief Select an explicit invariant dav-2201 Fixpipe configuration.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/ascend_pipe.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace tl {
using namespace tirx;
namespace ascend {

bool UsesRootFixpipeConfig(const Map<String, ObjectRef> &annotations) {
  auto value = annotations.Get(kRootFixpipeConfig);
  if (!value)
    return false;
  const auto *flag = value->as<IntImmNode>();
  TVM_FFI_CHECK(flag && flag->value == 1, ValueError)
      << "Internal root Fixpipe configuration flag must equal integer 1";
  return true;
}

namespace {

// Only registered operations with known hardware roles and pure IR metadata
// can preserve ND_PARA. In particular, even a call_pure_extern is not a
// verified native register contract and prevents selection.
bool PreservesFixpipeConfig(const CallNode *call) {
  if (call->op.same_as(builtin::call_extern()) ||
      call->op.same_as(builtin::call_pure_extern()))
    return false;
  const auto callee = call->op.as<Op>();
  if (!callee)
    return false;
  if (call->op.same_as(ascend_gemm_l1()) ||
      call->op.same_as(ascend_blockscaled_gemm_l1()))
    return false; // Opaque library implementations may issue Basic Fixpipe.
  if (call->op.same_as(region()) || call->op.same_as(access_ptr()) ||
      call->op.same_as(builtin::tvm_access_ptr()) ||
      call->op.same_as(builtin::assume()))
    return true;
  if (GetAscendCallPipeMask(GetRef<Call>(call)) != 0)
    return true;
  static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  return effects.count(callee.value()) &&
         effects[callee.value()]->value <=
             static_cast<int>(CallEffectKind::kPure);
}

class FixpipeConfigInspector : public StmtExprVisitor {
public:
  int initializers{0};
  int copies{0};
  int selected{0};
  bool initializer_is_root{true};
  bool selected_before_initializer{false};
  bool unknown_state{false};
  bool selection_unsafe{false};

private:
  void VisitStmt_(const SBlockNode *op) final {
    CoreMask saved = core_;
    if (op->name_hint == "CUBE")
      core_ = kCoreCube;
    else if (op->name_hint == "VECTOR")
      core_ = kCoreVector;
    StmtExprVisitor::VisitStmt_(op);
    core_ = saved;
  }
  void VisitStmt_(const ForNode *op) final {
    ++control_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --control_depth_;
  }
  void VisitStmt_(const WhileNode *op) final {
    ++control_depth_;
    ++conditional_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --conditional_depth_;
    --control_depth_;
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    ++control_depth_;
    ++conditional_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --conditional_depth_;
    --control_depth_;
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    bool atomic = op->attr_key == attr::kAscendTask ||
                  op->attr_key == attr::kAscendPerCoreTask;
    atomic_depth_ += atomic;
    CoreMask saved = core_;
    if (atomic) {
      auto metadata = ParseTaskMetadata(op->node);
      if (metadata.core_mask)
        core_ = metadata.core_mask.value();
    }
    StmtExprVisitor::VisitStmt_(op);
    core_ = saved;
    atomic_depth_ -= atomic;
  }
  void VisitExpr_(const CallNode *call) final {
    bool marked = UsesRootFixpipeConfig(call->annotations);
    bool copy = false;
    if (call->op.same_as(ascend_set_fixpipe_config())) {
      ++initializers;
      initializer_is_root &=
          control_depth_ == 0 && call->args.empty() && core_ == kCoreCube;
    } else if (IsAscendCopyCall(call)) {
      AscendCopy node(call->args, call->annotations);
      copy = IsL0CBuffer(node->src) && IsGlobalBuffer(node->dst);
      if (copy) {
        ++copies;
        selection_unsafe |= conditional_depth_ != 0 || atomic_depth_ != 0;
      } else if (IsL0CBuffer(node->src)) {
        unknown_state = true;
      }
    } else if (call->op.same_as(ascend_copy_matrix_cc_to_gm())) {
      copy = true;
      ++copies;
      // Existing late native calls keep their original atomic setup contract.
      selection_unsafe = true;
    } else if (call->op.same_as(ascend_copy_matrix_cc_to_ub())) {
      unknown_state = true;
    } else if (!PreservesFixpipeConfig(call)) {
      unknown_state = true;
    }
    TVM_FFI_CHECK(!marked || copy, ValueError)
        << "Internal root Fixpipe configuration flag requires L0C->GM copy";
    if (marked) {
      TVM_FFI_CHECK(core_ != kCoreVector, ValueError)
          << "Selected Fixpipe copy must execute on the AIC";
      ++selected;
      selected_before_initializer |= initializers == 0;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
  int control_depth_{0};
  int conditional_depth_{0};
  int atomic_depth_{0};
  CoreMask core_{kCoreUnassigned};
};

void ValidateScope(const Stmt &body) {
  FixpipeConfigInspector inspector;
  inspector(body);
  if (inspector.initializers == 0 && inspector.selected == 0)
    return;
  TVM_FFI_CHECK(inspector.initializers == 1 && inspector.initializer_is_root &&
                    !inspector.selected_before_initializer,
                ValueError)
      << "Selected Fixpipe requires exactly one preceding unconditional "
         "kernel-root configuration owned by the AIC";
  TVM_FFI_CHECK(!inspector.unknown_state &&
                    inspector.selected == inspector.copies,
                ValueError)
      << "Root Fixpipe configuration cannot coexist with unknown external "
         "calls, raw register setters or unselected FIX copies";
}

class RootConfigValidator : public StmtVisitor {
public:
  int kernels{0};

private:
  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "tilelang_root") {
      ++kernels;
      ValidateScope(op->body);
    } else {
      StmtVisitor::VisitStmt_(op);
    }
  }
};

class FixpipeConfigSelector : public StmtExprMutator {
public:
  SBlock Rewrite(const SBlock &root) {
    FixpipeConfigInspector inspector;
    inspector(root->body);
    if (inspector.initializers || inspector.selected) {
      ValidateScope(root->body);
      return root;
    }
    if (!inspector.copies || inspector.unknown_state ||
        inspector.selection_unsafe)
      return root;
    SBlock rewritten = Downcast<SBlock>(operator()(root));
    TaskMetadata ownership;
    ownership.core_mask = kCoreCube;
    Stmt setup =
        Evaluate(Call(DataType::Void(), ascend_set_fixpipe_config(), {}));
    setup = AttrStmt(MergeTaskMetadata(ownership, Integer(0)),
                     attr::kAscendTask, Integer(1), setup);
    rewritten.CopyOnWrite()->body = SeqStmt({setup, rewritten->body});
    ValidateScope(rewritten->body);
    return rewritten;
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (IsAscendCopyCall(op)) {
      AscendCopy copy(op->args, op->annotations);
      if (IsL0CBuffer(copy->src) && IsGlobalBuffer(copy->dst)) {
        Call rewritten = GetRef<Call>(op);
        rewritten.CopyOnWrite()->annotations.Set(kRootFixpipeConfig,
                                                 Integer(1));
        return rewritten;
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};
} // namespace

void ValidateRootFixpipeConfig(const Stmt &body) {
  RootConfigValidator validator;
  validator(body);
  if (validator.kernels == 0)
    ValidateScope(body);
}
} // namespace ascend

namespace transform {
tirx::transform::Pass AscendSelectFixpipeConfig() {
  auto pass_func = [](PrimFunc func, IRModule,
                      tvm::transform::PassContext) -> PrimFunc {
    Target target = func->GetAttr<Target>(tvm::attr::kTarget).value();
    if (GetAscendTargetInfo(target).architecture !=
        AscendArchitecture::kDav2201)
      return func;
    return ascend::RewriteTilelangKernels(
        func, "AscendSelectFixpipeConfig",
        [](const ascend::TilelangKernelContext &context) {
          return ascend::FixpipeConfigSelector().Rewrite(context.root);
        });
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.AscendSelectFixpipeConfig", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendSelectFixpipeConfig",
                                   AscendSelectFixpipeConfig);
}
} // namespace transform
} // namespace tl
} // namespace tvm
