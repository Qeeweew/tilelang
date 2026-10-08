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

#ifndef TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_L0AB_PHASE_REUSE_H_
#define TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_L0AB_PHASE_REUSE_H_

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./dependency_analysis.h"
#include "./resource_epoch.h"
#include "./scheduled_tir.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/cube_load.h"
#include "ascend/target_utils.h"
#include "op/gemm.h"

namespace tvm::tl::ascend {

// Root Array<Array<Var>>, with two storage Vars per selected reuse pair.
// Operand and accumulator phases share one max-budget/final-alias contract,
// without sharing logical layouts or version clocks.
constexpr const char *kL0PhaseReusePairs = "tl.ascend_l0_phase_reuse_pairs";

// An early address-reuse choice, not an alias proof or a shared version clock.
// Keep both logical allocations/layouts through native lowering. InsertSync
// must prove HasBufferAlias(first, second) before the max-budget choice can be
// used by the late allocator. Exactly two allocations in this scope make that
// positive proof sufficient for the existing clique packer to co-locate them.
struct L0PhaseReusePair {
  Var first, second;
  std::string scope;
  ControlNode *epoch{nullptr};
  IRStructure *first_phase{nullptr};
  IRStructure *second_phase{nullptr};
};

namespace l0ab_phase_reuse_detail {

using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
template <typename T>
using StorageMap =
    std::unordered_map<Var, T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

using resource_epoch_detail::Constant;
using resource_epoch_detail::PositiveStaticLoop;
using resource_epoch_detail::StageZero;

inline std::string Scope(const Var &storage) {
  std::string scope = GetPtrStorageScope(storage);
  if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
    return "shared.l0a";
  if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return "shared.l0b";
  return "";
}

inline std::string StorageScope(const Var &storage) {
  std::string scope = Scope(storage);
  if (!scope.empty())
    return scope;
  scope = GetPtrStorageScope(storage);
  return scope == "shared.l0c" || scope == "shared.l0c.dyn" ? "shared.l0c" : "";
}

inline ControlNode *ParentControl(IRStructure *node) {
  IRStructure *parent = node->GetParent();
  return parent && parent->IsControl() ? static_cast<ControlNode *>(parent)
                                       : nullptr;
}

// Activation snapshots may depend on enclosing loop values, but never on
// mutable storage or a call. Resolve their existing Bind context rather than
// treating a bare boolean Var as an immutable hardware fact.
inline bool StableGuards(const IRStructure *node, const TaskNode *context) {
  std::unordered_set<const VarNode *> outer_vars, resolving, stable;
  for (const IRStructure *parent = node->GetParent(); parent;
       parent = parent->GetParent())
    if (parent->IsControl())
      outer_vars.insert(
          static_cast<const ControlNode *>(parent)->control->loop_var.get());
  std::function<bool(const VarNode *)> stable_var;
  class PureExpression : public ExprVisitor {
  public:
    explicit PureExpression(const std::function<bool(const VarNode *)> &check)
        : check_(check) {}
    bool valid{true};

  private:
    void VisitExpr_(const VarNode *op) final { valid &= check_(op); }
    void VisitExpr_(const CallNode *) final { valid = false; }
    void VisitExpr_(const BufferLoadNode *) final { valid = false; }
    void VisitExpr_(const ProducerLoadNode *) final { valid = false; }
    void VisitExpr_(const ReduceNode *) final { valid = false; }
    const std::function<bool(const VarNode *)> &check_;
  };
  stable_var = [&](const VarNode *var) {
    if (var->dtype.lanes() != 1 ||
        (!var->dtype.is_int() && !var->dtype.is_uint() &&
         !var->dtype.is_bool()))
      return false;
    if (outer_vars.count(var) || stable.count(var))
      return true;
    if (!resolving.insert(var).second)
      return false;
    const PrimExpr *definition = nullptr;
    for (const Constr &constraint : context->outer_ctx.constrs_) {
      if (constraint.kind == Constr::kBindValue &&
          constraint.var.get() == var) {
        if (definition)
          return false;
        definition = &constraint.value;
      }
    }
    if (!definition)
      return false;
    PureExpression expression(stable_var);
    expression(*definition);
    resolving.erase(var);
    if (expression.valid)
      stable.insert(var);
    return expression.valid;
  };
  for (const auto &guard : node->GetGuards()) {
    if (!guard->IsCondition())
      return false;
    PureExpression expression(stable_var);
    expression(static_cast<const ConditionGuard *>(guard.get())->condition);
    if (!expression.valid)
      return false;
  }
  return true;
}

inline bool FullAllocation(const Buffer &buffer, const Array<Range> &ranges) {
  if (buffer->shape.size() != 2 || ranges.size() != 2)
    return false;
  for (size_t axis = 0; axis < 2; ++axis)
    if (Constant(ranges[axis]->min) != 0 ||
        Constant(ranges[axis]->extent) != Constant(buffer->shape[axis]))
      return false;
  return true;
}

inline std::optional<Call> Operation(const TaskNode *task) {
  if (task->IsPerCoreTask() || (task->GetCoreMask() != kCoreUnassigned &&
                                task->GetCoreMask() != kCoreCube))
    return std::nullopt;
  Stmt body = task->stmt;
  while (const auto *attribute = body.as<AttrStmtNode>()) {
    if (attribute->attr_key != attr::kAscendTask)
      return std::nullopt;
    body = attribute->body;
  }
  const auto *evaluate = body.as<EvaluateNode>();
  if (!evaluate || !evaluate->value.as<CallNode>())
    return std::nullopt;
  return Downcast<Call>(evaluate->value);
}

// Distinguish typed endpoint transport from scalar/pointer escape. A view
// sharing the storage Var is still a different logical Buffer and is rejected.
// This is an input audit; actual task effects remain the common region facts.
class KernelAudit : public StmtExprVisitor {
public:
  explicit KernelAudit(StorageMap<Buffer> buffers)
      : buffers_(std::move(buffers)) {
    for (const auto &[storage, _] : buffers_)
      scopes_.insert(StorageScope(storage));
  }
  StorageMap<int> allocations;
  StorageSet invalid;
  LayoutMap layouts;
  bool unsafe_metadata{false};

private:
  void VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    if (buffers_.count(var))
      invalid.insert(var);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (buffers_.count(op->buffer->data))
      invalid.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (buffers_.count(op->buffer->data))
      invalid.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    if (alloc_data && scopes_.count(StorageScope(buffer->data)) &&
        !buffers_.count(buffer->data))
      unsafe_metadata = true;
    if (alloc_data && buffers_.count(buffer->data))
      ++allocations[buffer->data];
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }
  void VisitStmt_(const SBlockNode *op) final {
    unsafe_metadata |= !op->match_buffers.empty();
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      // Earlier layout inference may still carry Var-keyed entries for other
      // buffers. Only inferred Buffer keys can establish our canonical facts.
      for (const auto &[key, layout] :
           value.value().cast<Map<ObjectRef, Layout>>()) {
        auto buffer = key.as<Buffer>();
        if (!buffer.has_value() || !buffers_.count(buffer.value()->data))
          continue;
        const Var &storage = buffer.value()->data;
        if (!buffer.value().same_as(buffers_.at(storage)))
          invalid.insert(storage);
        if (auto previous = layouts.Get(buffer.value()))
          if (!previous.value()->IsEqual(layout.get()))
            invalid.insert(storage);
        layouts.Set(buffer.value(), layout);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode *op) final {
    unsafe_metadata |= op->annotations.count(kMultiBufferCounterMap) ||
                       op->annotations.count(kStorageEpochGuardMap) ||
                       op->annotations.count("conflict_hint");
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    // Calls with unmodeled effects may address local storage without carrying
    // its Var. Pointer escape detection alone cannot certify those calls.
    if (op->op.same_as(builtin::call_extern()) ||
        op->op.same_as(builtin::call_pure_extern()) || !op->op.as<OpNode>() ||
        op->op.same_as(ascend_gemm_l1()) ||
        op->op.same_as(ascend_blockscaled_gemm_l1()))
      unsafe_metadata = true;
    const bool typed_cube = op->op.same_as(Gemm::Get()) ||
                            IsAscendCopyCall(op) ||
                            op->op.same_as(AscendCubeLoad3D::Get());
    uint16_t pipes = GetAscendCallPipeMask(ffi::GetRef<Call>(op));
    // A late M/MTE1 intrinsic can use a literal local address without exposing
    // any storage Var. Such a call cannot certify the two typed lifetimes.
    if (!typed_cube && (pipes & (PipeMask(ResourcePipe::kCube) |
                                 PipeMask(ResourcePipe::kMTE1))) != 0)
      unsafe_metadata = true;
    if (const auto *node = op->op.as<OpNode>()) {
      const Op &operation = ffi::GetRef<Op>(node);
      if (!op->op.same_as(region()) && !op->op.same_as(tl::access_ptr()) &&
          !op->op.same_as(builtin::tvm_access_ptr()) &&
          !op->op.same_as(builtin::assume()) && pipes == 0) {
        static auto effects =
            Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
        if (!effects.count(operation) ||
            effects[operation]->value > static_cast<int>(CallEffectKind::kPure))
          unsafe_metadata = true;
      }
    }
    size_t typed = op->op.same_as(Gemm::Get()) ? 3 : 0;
    if (IsAscendCopyCall(op) || op->op.same_as(AscendCubeLoad3D::Get()))
      typed = 2;
    for (size_t i = 0; i < op->args.size(); ++i) {
      const auto *region_call = op->args[i].as<CallNode>();
      const auto *load = region_call && region_call->op.same_as(region()) &&
                                 region_call->args.size() >= 2
                             ? region_call->args[0].as<BufferLoadNode>()
                             : nullptr;
      if (i >= typed || !load || load->predicate.defined() ||
          load->indices.size() != load->buffer->shape.size() ||
          region_call->args.size() != load->indices.size() + 2) {
        VisitExpr(op->args[i]);
        continue;
      }
      if (buffers_.count(load->buffer->data) &&
          !load->buffer.same_as(buffers_.at(load->buffer->data)))
        invalid.insert(load->buffer->data);
      if (scopes_.count(StorageScope(load->buffer->data)) &&
          !buffers_.count(load->buffer->data))
        unsafe_metadata = true;
      for (const PrimExpr &index : load->indices)
        VisitExpr(index);
      for (size_t j = 1; j < region_call->args.size(); ++j)
        VisitExpr(region_call->args[j]);
    }
    for (const auto &[_, value] : op->annotations)
      if (auto expression = value.as<PrimExpr>())
        VisitExpr(expression.value());
  }
  StorageMap<Buffer> buffers_;
  std::unordered_set<std::string> scopes_;
};

inline bool CanonicalAllocation(const Buffer &buffer, const LayoutMap &layouts,
                                int64_t capacity) {
  if (buffer->dtype.lanes() != 1 ||
      (buffer->dtype != DataType::Float(16) && !buffer->dtype.is_bfloat16()) ||
      buffer->shape.size() != 2 || !buffer->strides.empty() ||
      !buffer->axis_separators.empty() || Constant(buffer->elem_offset) != 0)
    return false;
  auto rows = Constant(buffer->shape[0]), cols = Constant(buffer->shape[1]);
  if (!rows || !cols || *rows <= 0 || *cols <= 0 || *rows % 16 || *cols % 16 ||
      *rows > capacity / 2 / *cols)
    return false;
  auto layout = layouts.Get(buffer);
  if (!layout.has_value() || !IsAscendFractalLayout(layout.value()))
    return false;
  AscendFractalLayoutInfo info;
  if (!TryExtractAscendFractalLayout(layout.value(), buffer, &info))
    return false;
  if (Scope(buffer->data) == "shared.l0a")
    return info.kind == AscendFractalKind::kL0AMajorK &&
           layout.value()->IsEqual(
               MakeAscendL0A2201Layout(buffer, false).get());
  return (info.kind == AscendFractalKind::kMajorK &&
          layout.value()->IsEqual(MakeAscendMajorKLayout(buffer).get())) ||
         (info.kind == AscendFractalKind::kMajorMN &&
          layout.value()->IsEqual(MakeAscendMajorMNLayout(buffer).get()));
}

inline bool DefaultAnnotations(const Map<String, ObjectRef> &annotations) {
  for (const auto &[key, value] : annotations) {
    auto expression = value.as<PrimExpr>();
    if (key != "unit_flag_ctrl" || !expression.has_value() ||
        Constant(expression.value()) != 0)
      return false;
  }
  return true;
}

inline bool AdmissibleAccess(const TaskNode *task, const Buffer &buffer,
                             bool reads, bool writes,
                             const LayoutMap &layouts) {
  auto operation = Operation(task);
  if (!operation || !StableGuards(task, task))
    return false;
  const Call &call = *operation;
  if (reads && !writes && call->op.same_as(Gemm::Get())) {
    Gemm gemm(call->args, call->annotations);
    if (!task->UsesOnlyPipe(ResourcePipe::kCube) || !IsL0ABuffer(gemm->a_) ||
        !IsL0BBuffer(gemm->b_) || !IsL0CBuffer(gemm->c_) ||
        gemm->c_->dtype != DataType::Float(32) ||
        gemm->a_->dtype != gemm->b_->dtype ||
        gemm->a_->dtype != buffer->dtype || gemm->transA_ ||
        gemm->mbar_.defined() || gemm->wgWait_ != 0 || gemm->isWgmma_ ||
        gemm->isTcgen05_ || gemm->kPack_ != 1 ||
        !DefaultAnnotations(call->annotations))
      return false;
    const BufferRegion &region =
        Scope(buffer->data) == "shared.l0a" ? gemm->aRegion_ : gemm->bRegion_;
    if (!region->buffer.same_as(buffer) ||
        !FullAllocation(buffer, region->region))
      return false;
    if (Scope(buffer->data) == "shared.l0b") {
      Layout expected = gemm->transB_ ? MakeAscendMajorKLayout(buffer)
                                      : MakeAscendMajorMNLayout(buffer);
      if (!layouts.at(buffer)->IsEqual(expected.get()))
        return false;
    }
    return true;
  }
  if (!writes || reads || !task->UsesOnlyPipe(ResourcePipe::kMTE1))
    return false;
  if (call->op.same_as(AscendCubeLoad3D::Get())) {
    if (!call->annotations.empty())
      return false;
    AscendCubeLoad3D copy(call->args, call->annotations);
    return IsL1Buffer(copy->src->buffer) &&
           copy->src->buffer->dtype == buffer->dtype &&
           copy->dst->buffer.same_as(buffer) &&
           FullAllocation(buffer, copy->dst->region);
  }
  if (!IsAscendCopyCall(call.get()) || !call->annotations.empty())
    return false;
  AscendCopy copy(call->args, call->annotations);
  return IsL1Buffer(copy->src) && copy->src->dtype == buffer->dtype &&
         copy->dst.same_as(buffer) && FullAllocation(buffer, copy->dst_range);
}

inline void
CollectControls(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                std::vector<ControlNode *> *controls) {
  for (const auto &node : nodes)
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node.get());
      controls->push_back(control);
      CollectControls(control->children, controls);
    }
}

// Locate the existing lexical phase without changing any of its activation
// predicates. Completion and final address reuse remain downstream proofs.
inline bool
LocateClosedPhase(TaskNode *task, ControlNode **epoch, IRStructure **phase,
                  bool prepared = false,
                  const ResourceEpochAnalysis *resource_epochs = nullptr,
                  const std::vector<Var> &resources = {},
                  ResourceEpochDecision *decision = nullptr) {
  ControlNode *outer = ParentControl(task);
  if (!outer)
    return false;
  outer = outer->GetOutermostControl();
  ResourceEpochAnalysis lexical_epochs;
  const ResourceEpochAnalysis &facts =
      resource_epochs ? *resource_epochs : lexical_epochs;
  ResourceEpochDecision epoch_decision =
      facts.Inspect(outer, resources, prepared);
  if (decision)
    *decision = epoch_decision;
  if (!epoch_decision.Accepted() || (*epoch && *epoch != outer))
    return false;
  IRStructure *child = task->GetChildOnPathFrom(outer);
  if (!child->IsControl() || !StageZero(child) || (*phase && *phase != child))
    return false;
  auto *phase_loop = static_cast<ControlNode *>(child);
  if (!PositiveStaticLoop(phase_loop) ||
      !StableGuards(phase_loop, phase_loop->task.get()))
    return false;
  for (ControlNode *parent = ParentControl(task); parent != outer;
       parent = parent->GetParentControl())
    if (!parent || !PositiveStaticLoop(parent) ||
        !StableGuards(parent, parent->task.get()))
      return false;
  *epoch = outer;
  *phase = child;
  return true;
}

} // namespace l0ab_phase_reuse_detail

// Called before ScheduleBuilder mutates stages/versions. This only chooses
// disjoint lexical phases: no benchmark/kernel names or source pointer tricks.
inline std::vector<L0PhaseReusePair>
AnalyzeL0ABPhaseReuse(const ScheduledTIR &scheduled,
                      const AscendTargetInfo &target) {
  using namespace l0ab_phase_reuse_detail;
  std::vector<L0PhaseReusePair> plans;
  ResourceEpochAnalysis resource_epochs(scheduled);
  const auto &metadata = scheduled.metadata;
  if (target.architecture != AscendArchitecture::kDav2201 ||
      metadata.has_buffer_aliases || !metadata.root_conflict_hints.empty() ||
      metadata.kernel_root->annotations.count(kL0PhaseReusePairs))
    return plans;
  StorageMap<Buffer> buffers;
  std::unordered_map<std::string, std::vector<Buffer>> scopes;
  for (const Buffer &buffer : metadata.kernel_root->alloc_buffers) {
    std::string scope = Scope(buffer->data);
    if (!scope.empty()) {
      if (!buffers.emplace(buffer->data, buffer).second)
        return {};
      scopes[scope].push_back(buffer);
    }
  }
  KernelAudit audit(buffers);
  audit(metadata.kernel_root);
  if (audit.unsafe_metadata)
    return plans;
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  std::vector<ControlNode *> controls;
  CollectControls(scheduled.tree, &controls);
  for (const auto &[scope, members] : scopes) {
    if (members.size() != 2 ||
        std::find(metadata.unlimit_memory_scopes.begin(),
                  metadata.unlimit_memory_scopes.end(),
                  String(scope)) != metadata.unlimit_memory_scopes.end())
      continue;
    int64_t capacity =
        scope == "shared.l0a" ? target.l0a_bytes : target.l0b_bytes;
    const Buffer &a = members[0], &b = members[1];
    // The late allocator packs each exact scope separately. Capacity-scope
    // normalization does not authorize merging a static and dynamic arena.
    bool valid = GetPtrStorageScope(a->data) == GetPtrStorageScope(b->data);
    for (const Buffer &buffer : members)
      valid &= audit.allocations[buffer->data] == 1 &&
               !audit.invalid.count(buffer->data) &&
               !metadata.manual_buffer_versions.count(buffer->data) &&
               !metadata.buffer_version_modes.count(buffer->data) &&
               CanonicalAllocation(buffer, audit.layouts, capacity);
    for (ControlNode *control : controls)
      if (control->IsMultiBufferEligible(a->data) &&
          control->IsMultiBufferEligible(b->data))
        valid = false;
    if (!valid)
      continue;
    ControlNode *epoch = nullptr;
    IRStructure *phases[2] = {nullptr, nullptr};
    for (size_t index = 0; index < 2 && valid; ++index) {
      const Buffer &buffer = members[index];
      bool has_read = false, has_write = false;
      for (TaskNode *task : tasks) {
        bool reads = false, writes = false;
        auto touches = [&](const std::vector<BufferRegion> &regions) {
          bool found = false;
          for (const BufferRegion &region : regions)
            if (region->buffer->data.same_as(buffer->data)) {
              valid &= region->buffer.same_as(buffer);
              found = true;
            }
          return found;
        };
        reads = touches(task->GetReadRegions());
        writes = touches(task->GetWriteRegions());
        if (!reads && !writes)
          continue;
        has_read |= reads;
        has_write |= writes;
        if (!valid ||
            !AdmissibleAccess(task, buffer, reads, writes, audit.layouts)) {
          valid = false;
          break;
        }
        valid &= LocateClosedPhase(task, &epoch, &phases[index], false,
                                   &resource_epochs, {a->data, b->data});
      }
      valid &= has_read && has_write;
    }
    if (!valid || !epoch || !phases[0] || !phases[1] || phases[0] == phases[1])
      continue;
    if (phases[0]->GetIndex() < phases[1]->GetIndex())
      plans.push_back({a->data, b->data, scope, epoch, phases[0], phases[1]});
    else
      plans.push_back({b->data, a->data, scope, epoch, phases[1], phases[0]});
  }
  return plans;
}

// Reuse the ordinary conflict contract at the actual common epoch. It forces
// cross-storage RAW/WAW/WAR through both AutoSchedule and InsertSync; these
// storage-free dependencies do not inherit either operand's private ring.
// Their unconditional lexical sites surround the phase Control nodes, so
// inactive data phases still participate in the finite boundary protocol.
inline void
ApplyL0PhaseReuseConflicts(const std::vector<L0PhaseReusePair> &plans,
                           ConflictHintList *root_conflicts) {
  for (const L0PhaseReusePair &plan : plans) {
    Array<ffi::Any> hint{plan.first, plan.second, Integer(-1), Bool(true)};
    auto *control = plan.epoch->control.CopyOnWrite();
    ConflictHintList hints;
    if (auto previous = control->annotations.Get("conflict_hint"))
      hints = previous.value().cast<ConflictHintList>();
    hints.push_back(hint);
    control->annotations.Set("conflict_hint", std::move(hints));
    root_conflicts->push_back(std::move(hint));
  }
}

} // namespace tvm::tl::ascend

#endif // TVM_TL_ASCEND_TRANSFORM_AUTO_SCHEDULE_L0AB_PHASE_REUSE_H_
