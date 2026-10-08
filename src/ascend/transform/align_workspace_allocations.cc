/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file align_workspace_allocations.cc
 * \brief Give compiler-owned host workspace allocations a 2201 performance
 *        alignment without changing their runtime allocation ownership.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <limits>
#include <unordered_map>

#include "ascend/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int64_t kWorkspacePerformanceAlignment = 512;
constexpr const char *kWorkspaceSizeError =
    "Ascend 2201 workspace byte size exceeds int64 capacity after "
    "512-byte alignment";

// LowerTVMBuiltin has already made host-owned global allocations explicit as
// AllocWorkspace Bind + null check + matching FreeWorkspace. External kernel
// arguments never have such a Bind. The caller selects this policy from the
// device target, since the host function itself is bound to a CPU target.
class WorkspaceAllocationAligner : public StmtExprMutator {
public:
  Stmt VisitStmt_(const BindNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !call->op.same_as(alloc_workspace_))
      return StmtExprMutator::VisitStmt_(op);

    TVM_FFI_CHECK(
        call->args.size() == 5 && op->var.dtype().is_handle() &&
            call->args[2].dtype().is_scalar() &&
            call->args[2].dtype().bits() <= 64 &&
            (call->args[2].dtype().is_int() || call->args[2].dtype().is_uint()),
        ValueError)
        << "Ascend workspace alignment expects a lowered AllocWorkspace Bind";
    PrimExpr requested = VisitExpr(call->args[2]);
    constexpr int64_t kMaximumBytes = std::numeric_limits<int64_t>::max() -
                                      (kWorkspacePerformanceAlignment - 1);
    // Reject constants before converting signed values to unsigned. Pinned
    // TIR represents uint64 constants above INT64_MAX as large_uint_imm calls,
    // which the arithmetic analyzer does not fold in a range comparison.
    if (const auto *constant = requested.as<IntImmNode>()) {
      TVM_FFI_CHECK(constant->value >= 0 && constant->value <= kMaximumBytes,
                    ValueError)
          << kWorkspaceSizeError;
    } else if (const auto *constant = requested.as<CallNode>();
               constant && constant->op.same_as(builtin::large_uint_imm())) {
      const auto *low = constant->args.size() == 2
                            ? constant->args[0].as<IntImmNode>()
                            : nullptr;
      const auto *high = constant->args.size() == 2
                             ? constant->args[1].as<IntImmNode>()
                             : nullptr;
      TVM_FFI_CHECK(low && high && low->value >= 0 && high->value >= 0 &&
                        low->value <= std::numeric_limits<uint32_t>::max() &&
                        high->value <= std::numeric_limits<uint32_t>::max(),
                    ValueError)
          << "Ascend workspace alignment expects a valid large_uint_imm";
      uint64_t value = (static_cast<uint64_t>(high->value) << 32U) |
                       static_cast<uint64_t>(low->value);
      TVM_FFI_CHECK(value <= static_cast<uint64_t>(kMaximumBytes), ValueError)
          << kWorkspaceSizeError;
    }
    // Keep a signed dynamic check signed, so negative inputs fail the runtime
    // assertion before the allocation-size cast and addition are evaluated.
    PrimExpr checked = cast(requested.dtype().is_int() ? DataType::Int(64)
                                                       : DataType::UInt(64),
                            requested);
    PrimExpr valid = checked <= make_const(checked.dtype(), kMaximumBytes);
    if (requested.dtype().is_int())
      valid = (checked >= make_zero(checked.dtype())) && valid;
    valid = analyzer_.Simplify(valid);
    TVM_FFI_CHECK(!analyzer_.CanProve(!valid), ValueError)
        << kWorkspaceSizeError;
    PrimExpr bytes = cast(DataType::UInt(64), requested);
    PrimExpr extra =
        make_const(DataType::UInt(64), kWorkspacePerformanceAlignment - 1);

    Var raw(op->var->name_hint + "_raw", op->var->type_annotation);
    TVM_FFI_CHECK(raw_allocations_.emplace(op->var, raw).second, ValueError)
        << "Ascend workspace alignment requires unique allocation Bind vars";
    ffi::Array<PrimExpr> args =
        call->args.Map([this](const PrimExpr &arg) { return VisitExpr(arg); });
    args.Set(2, bytes + extra);
    ffi::Array<Stmt> sequence;
    if (!analyzer_.CanProve(valid))
      sequence.push_back(AssertStmt(valid, StringImm("ValueError"),
                                    {StringImm(kWorkspaceSizeError)}));
    sequence.push_back(Bind(
        raw, Call(call->dtype, call->op, args, call->annotations, call->span)));
    // Check the actual owner before pointer arithmetic. The original lowered
    // null check is retained as well and continues to check this raw handle.
    sequence.push_back(
        IfThenElse(Call(DataType::Bool(), builtin::isnullptr(), {raw}),
                   Evaluate(Call(DataType::Int(32),
                                 builtin::tvm_throw_last_error(), {}))));
    PrimExpr address = reinterpret(DataType::UInt(64), raw);
    PrimExpr mask = extra;
    PrimExpr offset = bitwise_and(
        make_const(DataType::UInt(64), kWorkspacePerformanceAlignment) -
            bitwise_and(address, mask),
        mask);
    // Byte GEP keeps the pointer inside its retained allocation. Computing an
    // offset of at most 511 avoids rounding the integer address via addr+511.
    sequence.push_back(
        Bind(op->var, Call(DataType::Handle(),
                           builtin::handle_add_byte_offset(), {raw, offset})));
    return SeqStmt::Flatten(sequence);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    int argument = call->op.same_as(free_workspace_)
                       ? 2
                       : (call->op.same_as(builtin::isnullptr()) ? 0 : -1);
    if (argument < 0 || call->args.size() <= static_cast<size_t>(argument))
      return call;
    const auto *var = call->args[argument].as<VarNode>();
    if (!var)
      return call;
    auto found = raw_allocations_.find(ffi::GetRef<Var>(var));
    if (found == raw_allocations_.end())
      return call;
    call.CopyOnWrite()->args.Set(argument, found->second);
    return call;
  }

private:
  arith::Analyzer analyzer_;
  Op alloc_workspace_{Op::Get("tirx.TVMBackendAllocWorkspace")};
  Op free_workspace_{Op::Get("tirx.TVMBackendFreeWorkspace")};
  std::unordered_map<Var, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      raw_allocations_;
};

} // namespace

tvm::transform::Pass AscendAlignWorkspaceAllocations(Target target) {
  const bool align_workspaces =
      GetAscendTargetInfo(target).architecture == AscendArchitecture::kDav2201;
  return tirx::transform::CreatePrimFuncPass(
      [align_workspaces](PrimFunc func, IRModule, tvm::transform::PassContext) {
        if (!align_workspaces)
          return func;
        WorkspaceAllocationAligner aligner;
        func.CopyOnWrite()->body = aligner(func->body);
        return func;
      },
      0, "tl.AscendAlignWorkspaceAllocations", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.AscendAlignWorkspaceAllocations",
      AscendAlignWorkspaceAllocations);
}

} // namespace tl
} // namespace tvm
