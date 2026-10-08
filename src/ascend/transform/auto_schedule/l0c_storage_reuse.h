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

// Memory-pressure admission for single-version, sequential L0C lifetimes.
// Buffer identities and canonical layouts stay logical; data is the shared
// physical storage identity consumed by dependencies, versions and allocation.
#pragma once

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./multi_buffer.h"
#include "./l0ab_phase_reuse.h"
#include "./scheduled_tir.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/copy.h"
#include "ascend/target_utils.h"
#include "ascend/transform/storage_footprint.h"
#include "op/gemm.h"

namespace tvm {
namespace tl {
namespace ascend {
namespace l0c_storage_reuse_detail {

using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// Layout output and allocation ownership have already been established. Do
// not infer another physical layout or accept user-created storage aliases.
class KernelFacts : public StmtExprVisitor {
public:
  explicit KernelFacts(const SBlock &root) { operator()(root); }

  LayoutMap layouts;
  StorageSet conflicting_layouts;
  StorageSet padding_values;
  StorageSet escaped_storages;
  std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> owners;
  bool has_match_buffer{false};
  bool has_conflict_hint{false};
  bool has_prepared_versions{false};

private:
  void VisitExpr_(const VarNode *op) final {
    escaped_storages.insert(ffi::GetRef<Var>(op));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    escaped_storages.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    escaped_storages.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitCallAnnotations(const CallNode *op) {
    for (const auto &[_, value] : op->annotations)
      if (auto expression = value.as<PrimExpr>())
        VisitExpr(expression.value());
  }

  bool VisitRegionIdentity(const PrimExpr &operand) {
    const auto *call = operand.as<CallNode>();
    if (call == nullptr || !call->op.same_as(region()) || call->args.size() < 2)
      return false;
    const auto *load = call->args[0].as<BufferLoadNode>();
    const int64_t *mask = as_const_int(call->args[1]);
    if (load == nullptr || load->predicate.defined() || mask == nullptr ||
        *mask < 1 || *mask > 3 ||
        load->indices.size() != load->buffer->shape.size() ||
        call->args.size() != load->indices.size() + 2)
      return false;
    // Only a direct typed operand transports a Buffer identity rather than
    // loading a scalar. Its indices/extents can still contain real scalar
    // loads or opaque pointers and must pass through the ordinary visitor.
    for (const PrimExpr &index : load->indices)
      VisitExpr(index);
    for (size_t i = 1; i < call->args.size(); ++i)
      VisitExpr(call->args[i]);
    VisitCallAnnotations(call);
    return true;
  }

  void VisitExpr_(const CallNode *op) final {
    size_t typed_operands = 0;
    if (op->op.same_as(Op::Get("tl.tileop.gemm")))
      typed_operands = 3;
    else if (IsAscendCopyCall(op))
      typed_operands = 2;
    for (size_t i = 0; i < op->args.size(); ++i)
      if (i >= typed_operands || !VisitRegionIdentity(op->args[i]))
        VisitExpr(op->args[i]);
    VisitCallAnnotations(op);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (auto expression = op->node.as<PrimExpr>())
      VisitExpr(expression.value());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    if (alloc_data)
      ++owners[buffer->data];
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }

  void VisitStmt_(const SBlockNode *op) final {
    has_match_buffer |= !op->match_buffers.empty();
    if (auto value = op->annotations.Get(attr::kLayoutMap)) {
      for (const auto &[buffer, layout] : value.value().cast<LayoutMap>()) {
        if (auto previous = layouts.Get(buffer)) {
          if (!previous.value()->IsEqual(layout.get()))
            conflicting_layouts.insert(buffer->data);
        }
        layouts.Set(buffer, layout);
      }
    }
    if (auto value = op->annotations.Get(attr::kSafeValueMap))
      for (const auto &[storage, _] : value.value().cast<Map<Var, PrimExpr>>())
        padding_values.insert(storage);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    has_prepared_versions |= op->annotations.count(kMultiBufferCounterMap) ||
                             op->annotations.count(kStorageEpochGuardMap);
    if (auto value = op->annotations.Get("conflict_hint"))
      has_conflict_hint |= !value.value().cast<Array<ffi::Any>>().empty();
    StmtExprVisitor::VisitStmt_(op);
  }
};

inline bool IsPositiveStaticControl(const ControlNode *control) {
  const For &loop = control->control;
  const int64_t *extent = as_const_int(loop->extent);
  return !control->HasConditions() && !control->BodyHasLoopBreak() &&
         as_const_int(loop->min) != nullptr && extent != nullptr &&
         *extent > 0 && (!loop->step.has_value() || is_one(loop->step.value()));
}

// Atomic core/task/block wrappers may carry layout metadata, but cannot hide
// additional operations or repeated/conditional execution in this slice.
inline const CallNode *SingleCall(Stmt statement) {
  while (true) {
    if (const auto *attribute = statement.as<AttrStmtNode>()) {
      if (attribute->attr_key != attr::kAscendTask &&
          !IsScheduleGuardAttribute(attribute))
        return nullptr;
      statement = attribute->body;
    } else if (const auto *realize = statement.as<SBlockRealizeNode>()) {
      if (!is_one(realize->predicate) || !realize->iter_values.empty())
        return nullptr;
      statement = realize->block;
    } else if (const auto *block = statement.as<SBlockNode>()) {
      if (block->init.defined() || !block->alloc_buffers.empty() ||
          !block->match_buffers.empty() || !block->iter_vars.empty())
        return nullptr;
      statement = block->body;
    } else {
      const auto *evaluate = statement.as<EvaluateNode>();
      return evaluate ? evaluate->value.as<CallNode>() : nullptr;
    }
  }
}

struct Candidate {
  Buffer buffer;
  ControlNode *owner{nullptr};
  size_t first{0};
  size_t last{0};
  int64_t rows{0};
  int64_t bytes{0};
};

inline bool FullRegion(const BufferRegion &region, const Buffer &buffer) {
  if (!region->buffer.same_as(buffer) || region->region.size() != 2)
    return false;
  arith::Analyzer analyzer;
  for (size_t axis = 0; axis < 2; ++axis) {
    if (!is_zero(region->region[axis]->min) ||
        !analyzer.CanProveEqual(region->region[axis]->extent,
                                buffer->shape[axis]))
      return false;
  }
  return true;
}

inline void CollectTasks(const std::vector<std::shared_ptr<IRStructure>> &tree,
                         std::vector<TaskNode *> *leaves,
                         std::vector<TaskNode *> *headers) {
  for (const auto &node : tree) {
    if (node->IsTask()) {
      leaves->push_back(static_cast<TaskNode *>(node.get()));
    } else {
      auto *control = static_cast<ControlNode *>(node.get());
      headers->push_back(control->task.get());
      CollectTasks(control->children, leaves, headers);
    }
  }
}

inline bool FindLifetime(Candidate *candidate,
                         const std::vector<TaskNode *> &leaves,
                         const std::vector<TaskNode *> &headers) {
  const Buffer &buffer = candidate->buffer;
  ControlNode *owner = candidate->owner;
  if (!IsPositiveStaticControl(owner))
    return false;
  for (TaskNode *header : headers) {
    if (header && header->TouchesStorage(buffer->data))
      return false;
  }
  bool seen = false;
  for (TaskNode *task : leaves) {
    // Read-vars also contains every typed region's storage identity. Actual
    // scalar/opaque uses were excluded by KernelFacts, without conflating
    // that transport metadata with a scalar read.
    for (const Var &var : task->GetWriteVars())
      if (var.same_as(buffer->data))
        return false;
    if (!task->TouchesStorage(buffer->data))
      continue;
    if (!task->IsWithin(owner) || task->IsPerCoreTask())
      return false;
    Map<Var, PrimExpr> first_iteration;
    for (const IRStructure *node = task; node != nullptr;
         node = node->GetParent()) {
      if (node->HasConditions() || node->GetStage() >= 0)
        return false;
      if (node != owner && node->IsControl() && node->IsWithin(owner)) {
        const auto *control = static_cast<const ControlNode *>(node);
        if (!IsPositiveStaticControl(control))
          return false;
        first_iteration.Set(control->control->loop_var, control->control->min);
      }
    }
    // Only the one allocated logical buffer may describe this storage. An
    // untracked view, escaped pointer or scalar access cannot prove reuse.
    for (const auto &regions :
         {task->GetReadRegions(), task->GetWriteRegions()}) {
      for (const BufferRegion &region : regions)
        if (region->buffer->data.same_as(buffer->data) &&
            !region->buffer.same_as(buffer))
          return false;
    }
    const CallNode *call = SingleCall(task->stmt);
    if (call == nullptr)
      return false;
    if (task->GetCoreMask() == kCoreVector)
      return false;
    // Unit-flag hints suppress ordinary same-core FIX events in InsertSync.
    // Their separate hardware protocol is not a storage-reuse proof here.
    if (auto unit_flag = call->annotations.Get("unit_flag_ctrl")) {
      const auto *value = unit_flag.value().as<IntImmNode>();
      if (value == nullptr || value->value != 0)
        return false;
    }
    bool writes = false;
    for (const BufferRegion &region : task->GetWriteRegions())
      writes |= region->buffer->data.same_as(buffer->data);
    if (writes) {
      if (!call->op.same_as(Op::Get("tl.tileop.gemm")) ||
          task->GetPipeMask() != PipeMask(ResourcePipe::kCube))
        return false;
      Gemm gemm(call->args, call->annotations);
      if (!FullRegion(gemm->cRegion_, buffer) ||
          !as_const_int(buffer->shape[0]) || !as_const_int(buffer->shape[1]) ||
          gemm->m_ != *as_const_int(buffer->shape[0]) ||
          gemm->n_ != *as_const_int(buffer->shape[1]) ||
          gemm->a_->data.same_as(buffer->data) ||
          gemm->b_->data.same_as(buffer->data))
        return false;
      // A full logical GEMM on a 16-aligned canonical tile writes the whole
      // physical accumulator. Its first nested iteration must clear for
      // every owner epoch; do not substitute the owner coordinate itself.
      arith::Analyzer analyzer;
      if (!seen && !is_one(analyzer.Simplify(
                       Substitute(gemm->clearAccum_, first_iteration))))
        return false;
    } else {
      if (!seen || !IsAscendCopyCall(call) ||
          task->GetPipeMask() != PipeMask(ResourcePipe::kFixpipe))
        return false;
      AscendCopy copy(call->args, call->annotations);
      if (!copy->src.same_as(buffer))
        return false;
    }
    IRStructure *child = task->GetChildOnPathFrom(owner);
    if (!seen)
      candidate->first = child->GetIndex();
    candidate->last = child->GetIndex();
    seen = true;
  }
  return seen;
}

inline bool Compatible(const Candidate &lhs, const Candidate &rhs) {
  return lhs.owner == rhs.owner && lhs.rows == rhs.rows &&
         lhs.buffer->data_alignment == rhs.buffer->data_alignment &&
         lhs.buffer->offset_factor == rhs.buffer->offset_factor &&
         lhs.buffer->buffer_type == rhs.buffer->buffer_type &&
         (lhs.last < rhs.first || rhs.last < lhs.first);
}

// A narrow remap of the already-established protocols. Logical Buffer objects
// retain their own geometry and layouts; only their physical data changes.
class StorageRewriter : public StmtExprMutator {
public:
  StorageRewriter(Map<Buffer, Buffer> buffers, Map<Var, Var> storages,
                  StorageSet removed)
      : storages_(std::move(storages)), removed_(std::move(removed)) {
    buffer_remap_ = std::move(buffers);
  }

private:
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = ffi::GetRef<Var>(op);
    return storages_.Get(var).value_or(var);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    if (auto value = loop->annotations.Get(kMultiBufferEligible)) {
      Array<Var> eligible;
      StorageSet seen;
      for (const Var &storage : value.value().cast<Array<Var>>()) {
        Var remapped = storages_.Get(storage).value_or(storage);
        if (seen.insert(remapped).second)
          eligible.push_back(remapped);
      }
      loop.CopyOnWrite()->annotations.Set(kMultiBufferEligible, eligible);
    }
    return loop;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock shell = ffi::GetRef<SBlock>(op);
    Array<Buffer> allocations;
    for (const Buffer &buffer : shell->alloc_buffers)
      if (!removed_.count(buffer->data))
        allocations.push_back(buffer);
    shell.CopyOnWrite()->alloc_buffers = std::move(allocations);
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(shell.get()));
    if (auto value = block->annotations.Get(attr::kLayoutMap)) {
      LayoutMap layouts;
      for (const auto &[buffer, layout] : value.value().cast<LayoutMap>())
        layouts.Set(buffer_remap_.Get(buffer).value_or(buffer), layout);
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
    }
    return block;
  }

  Map<Var, Var> storages_;
  StorageSet removed_;
};

} // namespace l0c_storage_reuse_detail

// Return the original shell if no completely proved capacity repair exists.
// The caller decodes a changed shell once, before scheduling. No cost model,
// source-order assumption about asynchronous completion, or late alias proof
// substitutes for the shared-data RAW/WAR/WAW edges established afterward.
inline SBlock TryCoalesceL0CStorage(const ScheduledTIR &scheduled,
                                    const AscendTargetInfo &target) {
  using namespace l0c_storage_reuse_detail;
  const auto &metadata = scheduled.metadata;
  const SBlock &root = metadata.kernel_root;
  if (target.architecture != AscendArchitecture::kDav2201 ||
      metadata.has_buffer_aliases || !metadata.root_conflict_hints.empty())
    return root;
  for (const ffi::String &scope : metadata.unlimit_memory_scopes)
    if (scope == "shared.l0c")
      return root;
  KernelFacts facts(root);
  if (facts.has_match_buffer || facts.has_conflict_hint ||
      facts.has_prepared_versions)
    return root;
  auto slots = StorageFootprintCollector::SlotBytes(
      root, metadata.manual_buffer_versions);
  int64_t usage = 0;
  for (const auto &[storage, count] : facts.owners) {
    if (GetPtrStorageScope(storage) != "shared.l0c")
      continue;
    // Explicit physical/version contracts are not reinterpreted by this
    // automatic admission choice, including unrelated L0C reservations.
    if (metadata.buffer_versions.count(storage) ||
        metadata.manual_buffer_versions.count(storage) ||
        metadata.buffer_version_modes.count(storage))
      return root;
    auto slot = slots.Get(storage);
    const int64_t *bytes = slot ? as_const_int(slot.value()) : nullptr;
    if (bytes == nullptr || *bytes < 0 ||
        *bytes > std::numeric_limits<int64_t>::max() - usage)
      return root;
    usage += *bytes;
  }
  if (usage <= target.l0c_bytes)
    return root;
  MultiBufferOwnerMap owners = CollectMultiBufferOwners(
      scheduled.tree, L0StorageGroups(CollectL0SFBindings(root)));
  std::vector<TaskNode *> leaves, headers;
  CollectTasks(scheduled.tree, &leaves, &headers);
  std::vector<Candidate> candidates;
  for (const Buffer &buffer : root->alloc_buffers) {
    const Var &storage = buffer->data;
    if (!IsL0CBuffer(buffer) || buffer->dtype != DataType::Float(32) ||
        buffer->shape.size() != 2 || !buffer->axis_separators.empty() ||
        !is_zero(buffer->elem_offset) || facts.owners[storage] != 1 ||
        facts.conflicting_layouts.count(storage) ||
        facts.padding_values.count(storage) ||
        facts.escaped_storages.count(storage) ||
        metadata.buffer_versions.count(storage) ||
        metadata.manual_buffer_versions.count(storage) ||
        metadata.buffer_version_modes.count(storage))
      continue;
    const int64_t *rows = as_const_int(buffer->shape[0]);
    const int64_t *cols = as_const_int(buffer->shape[1]);
    if (rows == nullptr || cols == nullptr || *rows <= 0 || *cols <= 0 ||
        *rows % 16 != 0 || *cols % 16 != 0)
      continue;
    arith::Analyzer analyzer;
    if (!buffer->strides.empty() &&
        (buffer->strides.size() != 2 || !is_one(buffer->strides[1]) ||
         !analyzer.CanProveEqual(buffer->strides[0], buffer->shape[1])))
      continue;
    auto layout = facts.layouts.Get(buffer);
    if (!layout || !layout.value()->IsEqual(MakeAscendL0CLayout(buffer).get()))
      continue;
    auto owner = owners.find(storage);
    if (owner == owners.end() || owner->second.size() != 1)
      continue;
    const int64_t *bytes = as_const_int(slots.at(storage));
    if (bytes == nullptr || *rows > std::numeric_limits<int64_t>::max() / 4 ||
        *cols > std::numeric_limits<int64_t>::max() / (4 * *rows) ||
        *bytes != 4 * *rows * *cols)
      continue;
    Candidate candidate{buffer, owner->second[0], 0, 0, *rows, *bytes};
    if (FindLifetime(&candidate, leaves, headers))
      candidates.push_back(std::move(candidate));
  }
  // Largest allocation first makes every accepted smaller canonical N-prefix
  // fit its surviving owner. Each group is a complete lifetime-compatibility
  // clique; compatibility is never assumed transitive.
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate &lhs, const Candidate &rhs) {
                     return lhs.bytes > rhs.bytes;
                   });
  Map<Var, Var> storages;
  Map<Buffer, Buffer> buffers;
  StorageSet removed, assigned, shared;
  for (const Candidate &largest : candidates) {
    if (assigned.count(largest.buffer->data))
      continue;
    std::vector<const Candidate *> group{&largest};
    for (const Candidate &candidate : candidates) {
      if (usage <= target.l0c_bytes)
        break;
      if (candidate.buffer.same_as(largest.buffer) ||
          candidate.bytes > largest.bytes ||
          assigned.count(candidate.buffer->data) ||
          !std::all_of(group.begin(), group.end(),
                       [&](const Candidate *member) {
                         return Compatible(*member, candidate);
                       }))
        continue;
      Buffer remapped = candidate.buffer;
      remapped.CopyOnWrite()->data = largest.buffer->data;
      buffers.Set(candidate.buffer, remapped);
      storages.Set(candidate.buffer->data, largest.buffer->data);
      removed.insert(candidate.buffer->data);
      shared.insert(largest.buffer->data);
      assigned.insert(candidate.buffer->data);
      assigned.insert(largest.buffer->data);
      usage -= candidate.bytes;
      group.push_back(&candidate);
    }
  }
  if (usage > target.l0c_bytes || buffers.empty())
    return root;
  SBlock result = Downcast<SBlock>(StorageRewriter(
      std::move(buffers), std::move(storages), std::move(removed))(root));
  BufferVersionMap versions = metadata.buffer_versions;
  for (const Var &storage : shared)
    versions.Set(storage, 1);
  result.CopyOnWrite()->annotations.Set(kBufferVersionsMap, versions);
  return result;
}

namespace l0c_phase_reuse_detail {

using namespace l0ab_phase_reuse_detail;

inline bool CanonicalC(const Buffer &buffer, const LayoutMap &layouts,
                       int64_t capacity) {
  if (buffer->dtype != DataType::Float(32) || buffer->shape.size() != 2 ||
      !buffer->strides.empty() || !buffer->axis_separators.empty() ||
      Constant(buffer->elem_offset) != 0)
    return false;
  auto rows = Constant(buffer->shape[0]), cols = Constant(buffer->shape[1]);
  if (!rows || !cols || *rows <= 0 || *cols <= 0 || *rows % 16 || *cols % 16 ||
      *rows > capacity / 4 / *cols)
    return false;
  auto layout = layouts.Get(buffer);
  return layout && layout.value()->IsEqual(MakeAscendL0CLayout(buffer).get());
}

// One full-clear native K chain followed by one full FIX per result. This is
// initialization/region admission, not a native-unit or asynchronous completion
// proof. InsertSync must still establish all cross-phase completion edges.
inline bool CompleteResult(TaskNode *mad, TaskNode *fix) {
  ControlNode *result = ParentControl(fix);
  if (!result || !mad->IsWithin(result) || mad->HasGuards() || fix->HasGuards())
    return false;
  IRStructure *producer = mad->GetChildOnPathFrom(result);
  if (producer->GetIndex() >= fix->GetIndex())
    return false;
  Map<Var, PrimExpr> first_k;
  for (ControlNode *loop = ParentControl(mad); loop != result;
       loop = loop->GetParentControl()) {
    if (!loop || loop->HasGuards() || !PositiveStaticLoop(loop))
      return false;
    first_k.Set(loop->control->loop_var, loop->control->min);
  }
  auto operation = Operation(mad);
  if (!operation)
    return false;
  Gemm gemm((*operation)->args, (*operation)->annotations);
  return arith::Analyzer().CanProve(Substitute(gemm->clearAccum_, first_k));
}

inline bool NativeCWriter(const TaskNode *task, const Buffer &buffer,
                          const Call &call) {
  if (!call->op.same_as(Gemm::Get()) ||
      !task->UsesOnlyPipe(ResourcePipe::kCube) ||
      !DefaultAnnotations(call->annotations))
    return false;
  Gemm gemm(call->args, call->annotations);
  if (!IsL0ABuffer(gemm->a_) || !IsL0BBuffer(gemm->b_) ||
      !gemm->c_.same_as(buffer) ||
      (gemm->a_->dtype != DataType::Float(16) &&
       !gemm->a_->dtype.is_bfloat16()) ||
      gemm->a_->dtype != gemm->b_->dtype || gemm->transA_ ||
      gemm->mbar_.defined() || gemm->wgWait_ != 0 || gemm->isWgmma_ ||
      gemm->isTcgen05_ || gemm->kPack_ != 1 ||
      !FullAllocation(buffer, gemm->cRegion_->region))
    return false;
  const auto &a = gemm->aRegion_->region;
  const auto &b = gemm->bRegion_->region;
  if (a.size() < 2 || b.size() < 2)
    return false;
  auto m = Constant(buffer->shape[0]), n = Constant(buffer->shape[1]);
  auto k = Constant(a.back()->extent);
  return k && *k > 0 && *k % 16 == 0 &&
         Constant(a[a.size() - 2]->extent) == m &&
         Constant(b[gemm->transB_ ? b.size() - 2 : b.size() - 1]->extent) == n &&
         Constant(b[gemm->transB_ ? b.size() - 1 : b.size() - 2]->extent) == k;
}

inline bool OrdinaryFixAnnotations(const Map<String, ObjectRef> &annotations) {
  for (const auto &[key, value] : annotations) {
    auto expression = value.as<PrimExpr>();
    if (!expression ||
        !((key == "unit_flag_ctrl" && Constant(expression.value()) == 0) ||
          (key == kRootFixpipeConfig && Constant(expression.value()) == 1)))
      return false;
  }
  return true;
}

} // namespace l0c_phase_reuse_detail

// Independent canonical C geometries keep their original Buffer, data and
// layout identities. Only late byte allocation may co-locate them after real
// FIX/MAD completion has been proved. The older same-M repair above remains
// a separate, unchanged early shared-data route for unannotated lifetimes.
inline std::vector<L0PhaseReusePair>
AnalyzeL0CPhaseReuse(const ScheduledTIR &scheduled,
                     const AscendTargetInfo &target) {
  using namespace l0c_phase_reuse_detail;
  std::vector<L0PhaseReusePair> plans;
  const auto &metadata = scheduled.metadata;
  if (target.architecture != AscendArchitecture::kDav2201 ||
      metadata.has_buffer_aliases || !metadata.root_conflict_hints.empty() ||
      metadata.kernel_root->annotations.count(kL0PhaseReusePairs) ||
      std::find(metadata.unlimit_memory_scopes.begin(),
                metadata.unlimit_memory_scopes.end(), String("shared.l0c")) !=
          metadata.unlimit_memory_scopes.end())
    return plans;
  StorageMap<Buffer> buffers;
  for (const Buffer &buffer : metadata.kernel_root->alloc_buffers)
    if (StorageScope(buffer->data) == "shared.l0c" &&
        !buffers.emplace(buffer->data, buffer).second)
      return plans;
  if (buffers.size() != 2)
    return plans;
  KernelAudit audit(buffers);
  audit(metadata.kernel_root);
  if (audit.unsafe_metadata)
    return plans;
  std::vector<Buffer> members;
  for (const Buffer &buffer : metadata.kernel_root->alloc_buffers)
    if (buffers.count(buffer->data))
      members.push_back(buffer);
  if (GetPtrStorageScope(members[0]->data) !=
      GetPtrStorageScope(members[1]->data))
    return plans;
  std::vector<ControlNode *> controls;
  CollectControls(scheduled.tree, &controls);
  for (const Buffer &buffer : members) {
    auto version = metadata.buffer_versions.Get(buffer->data);
    if (!version || version.value() != 1 ||
        metadata.manual_buffer_versions.count(buffer->data) ||
        metadata.buffer_version_modes.count(buffer->data) ||
        audit.allocations[buffer->data] != 1 ||
        audit.invalid.count(buffer->data) ||
        !CanonicalC(buffer, audit.layouts, target.l0c_bytes))
      return plans;
    if (auto padding = metadata.kernel_root->annotations.Get(attr::kSafeValueMap))
      if (padding.value().cast<Map<Var, PrimExpr>>().count(buffer->data))
        return plans;
    for (ControlNode *control : controls)
      if (control->IsMultiBufferEligible(buffer->data))
        return plans;
  }
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  ControlNode *epoch = nullptr;
  IRStructure *phases[2] = {nullptr, nullptr};
  for (size_t index = 0; index < 2; ++index) {
    const Buffer &buffer = members[index];
    TaskNode *mad = nullptr, *fix = nullptr;
    for (TaskNode *task : tasks) {
      bool reads = false, writes = false;
      for (const BufferRegion &region : task->GetReadRegions())
        if (region->buffer->data.same_as(buffer->data)) {
          if (!region->buffer.same_as(buffer))
            return {};
          reads = true;
        }
      for (const BufferRegion &region : task->GetWriteRegions())
        if (region->buffer->data.same_as(buffer->data)) {
          if (!region->buffer.same_as(buffer))
            return {};
          writes = true;
        }
      if (!reads && !writes)
        continue;
      auto operation = Operation(task);
      if (!operation || !StableGuards(task, task) ||
          !LocateClosedPhase(task, &epoch, &phases[index]))
        return {};
      const Call &call = *operation;
      if (writes) {
        if (mad || !NativeCWriter(task, buffer, call))
          return {};
        mad = task;
      } else {
        if (fix || !reads || !IsAscendCopyCall(call.get()) ||
            !OrdinaryFixAnnotations(call->annotations) ||
            !task->UsesOnlyPipe(ResourcePipe::kFixpipe))
          return {};
        AscendCopy copy(call->args, call->annotations);
        if (!copy->src.same_as(buffer) || !IsGlobalBuffer(copy->dst) ||
            copy->src->dtype != copy->dst->dtype ||
            copy->dual_dst_ctl || copy->transpose || copy->data_select ||
            (copy->sub_blockid.defined() &&
             Constant(copy->sub_blockid) != 0) || copy->dst_block.has_value() ||
            copy->pad_value.has_value() || copy->src_oob_safe_value.has_value() ||
            !FullAllocation(buffer, copy->src_range))
          return {};
        fix = task;
      }
    }
    if (!mad || !fix || !CompleteResult(mad, fix))
      return {};
  }
  if (!epoch || !phases[0] || !phases[1] || phases[0] == phases[1])
    return plans;
  if (phases[0]->GetIndex() < phases[1]->GetIndex())
    plans.push_back({members[0]->data, members[1]->data, "shared.l0c", epoch,
                     phases[0], phases[1]});
  else
    plans.push_back({members[1]->data, members[0]->data, "shared.l0c", epoch,
                     phases[1], phases[0]});
  return plans;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
