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

// Pass-local resource/epoch facts. Rebuild after owner, counter, guard, or
// task-effect rewrites; this is not a completion or physical-alias proof.
#pragma once

#include <tvm/arith/analyzer.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "./scheduled_tir.h"

namespace tvm::tl::ascend {

namespace resource_epoch_detail {
inline std::optional<int64_t> Constant(const PrimExpr &expr) {
  PrimExpr value = arith::Analyzer().Simplify(expr);
  if (const auto *integer = value.as<IntImmNode>())
    return integer->value;
  return std::nullopt;
}

inline bool StageZero(const IRStructure *node) {
  return node->GetStage() == kUnscheduledStage || node->GetStage() == 0;
}

// Use the same policy before scheduling and after intent is consumed. A
// durable explicit value overrides automatic Pipelined's default offsets.
inline bool UsesStageOffsets(const ControlNode *control) {
  bool enabled = false;
  if (auto automatic =
          control->control->annotations.Get("tl.ascend_auto_pipeline"))
    enabled = automatic.value().cast<bool>();
  if (auto explicit_value = control->control->annotations.Get("enable_offset"))
    enabled = explicit_value.value().cast<bool>();
  return enabled;
}

inline bool PositiveStaticLoop(const ControlNode *control) {
  const For &loop = control->control;
  auto extent = Constant(loop->extent);
  return loop->kind == ForKind::kSerial && Constant(loop->min) == 0 && extent &&
         *extent > 0 &&
         (!loop->step.has_value() || Constant(loop->step.value()) == 1) &&
         !control->BodyHasLoopBreak();
}

inline bool Touches(const ObjectRef &value, const Var &storage) {
  bool found = false;
  PostOrderVisit(value, [&](const ObjectRef &node) {
    if (const auto *load = node.as<BufferLoadNode>())
      found |= load->buffer->data.same_as(storage);
    if (const auto *store = node.as<BufferStoreNode>())
      found |= store->buffer->data.same_as(storage);
    if (const auto *var = node.as<VarNode>())
      found |= var == storage.get();
  });
  return found;
}
} // namespace resource_epoch_detail

enum class ResourceEpochRejection {
  kNone,
  kOpenControl,
  kUnrecognizedAnnotation,
  kUnpreparedCounter,
  kCounterAllocation,
  kCounterPlan,
  kRelatedCounter,
  kNonVectorCounterUse,
};

inline const char *ResourceEpochReason(ResourceEpochRejection reason) {
  switch (reason) {
  case ResourceEpochRejection::kNone:
    return "closed-resource-epoch";
  case ResourceEpochRejection::kOpenControl:
    return "open-epoch-control";
  case ResourceEpochRejection::kUnrecognizedAnnotation:
    return "unknown-epoch-annotation";
  case ResourceEpochRejection::kUnpreparedCounter:
    return "unprepared-epoch-counter";
  case ResourceEpochRejection::kCounterAllocation:
    return "unverified-counter-allocation";
  case ResourceEpochRejection::kCounterPlan:
    return "unverified-counter-plan";
  case ResourceEpochRejection::kRelatedCounter:
    return "resource-related-counter";
  case ResourceEpochRejection::kNonVectorCounterUse:
    return "non-vector-counter-use";
  }
  ICHECK(false);
  return "unknown";
}

struct ResourceEpochDecision {
  ResourceEpochRejection reason{ResourceEpochRejection::kNone};
  Var storage{Var()};
  bool Accepted() const { return reason == ResourceEpochRejection::kNone; }
};

// Both early phase selection and late slot selection use this query. The
// early query has no prepared plan and cannot ignore counter metadata. Late
// selection may prove that an allocated AIV UB clock is independent of the
// queried L0 resource. Keeping this narrow rule preserves the existing
// admission set; arbitrary unrelated clocks are not silently accepted.
class ResourceEpochAnalysis {
public:
  ResourceEpochAnalysis() = default;
  explicit ResourceEpochAnalysis(const ScheduledTIR &scheduled,
                                 const MultiBufferPlan *plan = nullptr)
      : plan_(plan) {
    CollectAllTaskNodes(scheduled.tree, tasks_);
    for (const Buffer &buffer : scheduled.metadata.kernel_root->alloc_buffers)
      allocations_.emplace(buffer->data, buffer);
    if (plan_)
      for (const MultiBufferInfo &info : plan_->Infos())
        if (info.UsesCounter()) {
          counter_storages_[info.counter->data].push_back(&info);
          counter_groups_[info.counter_group_id].push_back(&info);
        }
  }

  bool HasExclusiveCounter(const MultiBufferInfo &info) const {
    if (!info.UsesCounter())
      return false;
    auto group = counter_groups_.find(info.counter_group_id);
    return group != counter_groups_.end() && group->second.size() == 1 &&
           group->second.front()->storage.same_as(info.storage);
  }

  ResourceEpochDecision Inspect(const ControlNode *epoch,
                                const std::vector<Var> &resources = {},
                                bool prepared = false) const {
    using namespace resource_epoch_detail;
    if (epoch->GetParentControl() || epoch->HasGuards() || !StageZero(epoch) ||
        !PositiveStaticLoop(epoch))
      return {ResourceEpochRejection::kOpenControl, Var()};
    for (const auto &[key, value] : epoch->control->annotations) {
      if (key == kMultiBufferEligible ||
          key == "tl.ascend_preserve_issue_order")
        continue;
      if (prepared && (key == "conflict_hint" || key == kStorageEpochGuardMap ||
                       key == "tl.ascend_l0_phase_reuse_pairs"))
        continue;
      if (key == kMultiBufferCounterMap) {
        if (!prepared || !plan_)
          return {ResourceEpochRejection::kUnpreparedCounter, Var()};
        ResourceEpochDecision counters = InspectCounters(epoch, resources);
        if (!counters.Accepted())
          return counters;
        continue;
      }
      if (key == "num_stages" &&
          Constant(value.cast<PrimExpr>()).value_or(0) > 0)
        continue;
      // Before scheduling, automatic intent may select this same static
      // epoch. Prepared proofs require the request to have been consumed.
      if (key == "enable_offset" || (key == "tl.ascend_auto_pipeline" &&
                                     (!prepared || !value.cast<bool>())))
        continue;
      return {ResourceEpochRejection::kUnrecognizedAnnotation, Var()};
    }
    return {ResourceEpochRejection::kNone, Var()};
  }

private:
  bool AllocatedUb(const Var &storage) const {
    auto allocation = allocations_.find(storage);
    if (allocation == allocations_.end())
      return false;
    std::string scope = GetPtrStorageScope(storage);
    return scope == "shared" || scope == "shared.dyn";
  }

  ResourceEpochDecision
  InspectCounters(const ControlNode *epoch,
                  const std::vector<Var> &resources) const {
    auto [audit, inserted] =
        counter_audits_.emplace(epoch, ResourceEpochDecision{});
    if (inserted)
      audit->second = InspectIndependentCounters(epoch);
    if (!audit->second.Accepted())
      return audit->second;
    auto counters = epoch->control->annotations.Get(kMultiBufferCounterMap);
    for (const auto &[storage, counter] :
         counters.value().cast<MultiBufferCounterMap>()) {
      const MultiBufferInfo *info = plan_->Find(storage);
      for (const Var &resource : resources) {
        const MultiBufferInfo *related = plan_->Find(resource);
        if (resource.same_as(storage) ||
            (related && related->UsesCounter() &&
             (related->counter_group_id == info->counter_group_id ||
              related->counter->data.same_as(counter->data))))
          return {ResourceEpochRejection::kRelatedCounter, storage};
      }
    }
    return {ResourceEpochRejection::kNone, Var()};
  }

  ResourceEpochDecision
  InspectIndependentCounters(const ControlNode *epoch) const {
    using namespace resource_epoch_detail;
    auto counters = epoch->control->annotations.Get(kMultiBufferCounterMap);
    ICHECK(counters.has_value());
    for (const auto &[storage, counter] :
         counters.value().cast<MultiBufferCounterMap>()) {
      if (!AllocatedUb(storage))
        return {ResourceEpochRejection::kCounterAllocation, storage};
      const MultiBufferInfo *info = plan_->Find(storage);
      if (!info || !info->UsesCounter() || !info->counter.same_as(counter) ||
          info->owners.size() != 1 || info->owners.front().loop != epoch)
        return {ResourceEpochRejection::kCounterPlan, storage};
      auto peers = counter_storages_.find(counter->data);
      ICHECK(peers != counter_storages_.end());
      for (const MultiBufferInfo *other : peers->second)
        if (!AllocatedUb(other->storage))
          return {ResourceEpochRejection::kRelatedCounter, other->storage};
      for (TaskNode *task : tasks_) {
        bool touches = Touches(task->stmt, storage) ||
                       Touches(task->stmt, counter->data) ||
                       task->GuardsTouchStorage(storage) ||
                       task->GuardsTouchStorage(counter->data);
        if (touches && task->GetCoreMask() != kCoreVector)
          return {ResourceEpochRejection::kNonVectorCounterUse, storage};
      }
    }
    return {ResourceEpochRejection::kNone, Var()};
  }

  // This memo belongs to this immutable snapshot only. Counter metadata and
  // task effects must not be mutated while querying the analysis.
  mutable std::unordered_map<const ControlNode *, ResourceEpochDecision>
      counter_audits_;
  const MultiBufferPlan *plan_{nullptr};
  std::vector<TaskNode *> tasks_;
  std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      allocations_;
  std::unordered_map<Var, std::vector<const MultiBufferInfo *>,
                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      counter_storages_;
  std::unordered_map<int, std::vector<const MultiBufferInfo *>> counter_groups_;
};

} // namespace tvm::tl::ascend
