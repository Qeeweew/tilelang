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
 * \file auto_schedule.cc
 * \brief AutoSchedule pass and schedule builder.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/dependency_analysis.h"
#include "./auto_schedule/ir_structure.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/l0ab_phase_reuse.h"
#include "./auto_schedule/l0c_storage_reuse.h"
#include "./auto_schedule/manual_schedule_completion.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/optimization_diagnostic.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/transform/buffer_version.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "runtime/thread_storage_scope.h"
#include "storage_footprint.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/collector.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;

namespace {

// Ascend's Pipelined(None) records intent separately from the upstream maximum
// buffer-version annotation. An explicit zero never emits this marker.
constexpr const char *kAutoPipeline = "tl.ascend_auto_pipeline";

// AutoSchedule target resource limits.

// Check whether a kernel body contains any SIMT_VF blocks. SIMT_VF
// (asc_vf_call) reserves 32 KB of the physical 248 KB Unified Buffer for
// thread context, so the usable shared memory depends on its presence.
class SimtVFDetector : public tirx::StmtVisitor {
public:
  bool found = false;

  void VisitStmt_(const tirx::SBlockNode *op) override {
    if (op->name_hint == "SIMT_VF") {
      found = true;
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

inline bool HasSimtVF(const tirx::Stmt &stmt) {
  SimtVFDetector detector;
  detector(stmt);
  return detector.found;
}

inline int64_t GetSharedMemoryLimit(const tirx::Stmt &kernel_body,
                                    const AscendTargetInfo &target_info) {
  bool has_simt_vf = HasSimtVF(kernel_body);
  ICHECK(!has_simt_vf || target_info.supports_simt_vf)
      << "SimtVF is not supported on " << target_info.arch;
  return target_info.ub_bytes -
         (has_simt_vf ? target_info.simt_vf_context_bytes : 0);
}

// Scheduler.

std::unique_ptr<IRExtraInfo> MakeAutoScheduleExtraInfo();

// Check if two variables are the same
bool SameVar(const Var &a, const Var &b);

// Check if there is a variable dependency between two IRStructures
bool HasVarDependency(const IRStructure *a, const IRStructure *b);

// Check if two IRStructures have resource dependency (use same hardware
// resource)
bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask = UINT16_MAX);

// Builder that schedules TaskNode/ControlNode children in IRStructure
class ScheduleBuilder {
private:
  struct LetDependencyClosure {
    std::vector<bool> is_declaration;
    std::vector<std::set<size_t>> dependencies;
  };

  static LetDependencyClosure
  CollectLetDependencyClosure_(const std::vector<IRStructure *> &nodes);

public:
  // Recursive scheduling function
  // Directly schedules the entire IRStructure tree recursively in place
  void ScheduleRecursive(std::shared_ptr<IRStructure> &node);

  // Schedule an ordered child list in place (root list or a ControlNode body):
  // gather + recurse into each child, then Z3-schedule and write the reordered
  // list back.
  void ScheduleList(std::vector<std::shared_ptr<IRStructure>> &children);

  // Z3-based scheduler for a straight-line child list.
  std::vector<std::shared_ptr<IRStructure>>
  Z3SchedulePython(const std::vector<std::shared_ptr<IRStructure>> &children,
                   bool manual_schedule);

  // Distance-aware Z3 scheduler for a ControlNode loop body.
  void Z3SchedulePythonLoop(ControlNode *ctrl, bool manual_schedule);

  // Set memory limit for a given buffer scope (e.g. "shared", "shared.l1")
  void SetStorageGroups(L0StorageGroups groups) {
    storage_groups_ = std::move(groups);
  }

  // Only 2201 has the non-pipe-selective collective wait contract used by
  // this source-issue preference. No target timing calibration is implied.
  void SetPreferHandoffIssue(bool value) { prefer_handoff_issue_ = value; }

  // Structural GM prefetch preference for ordinary dav-2201 auto pipelines.
  void SetPreferPrefetchIssue(bool value) { prefer_prefetch_issue_ = value; }

  // Whole-kernel L0 admission for dav-2201 automatic loops. Separate phase
  // allocations remain reserved unless a selected reuse contract retires
  // every reader before the other phase may overwrite the same address.
  void SetReserveL0Storage(bool value) { reserve_l0_storage_ = value; }

  // UB has one physical arena per AIV, including buffers used only outside
  // the current loop. A loop-local version choice cannot spend their bytes.
  void SetReserveUBStorage(bool value) { reserve_ub_storage_ = value; }

  void SetL0PhaseReuse(std::vector<L0PhaseReusePair> pairs) {
    l0_phase_reuse_ = std::move(pairs);
  }

  void SetMemoryLimit(const std::string &scope, int64_t bytes) {
    memory_limits_[scope] = bytes;
  }

  void SetStorageSlotBytes(ffi::Map<Var, PrimExpr> slots) {
    storage_slot_bytes_ = std::move(slots);
  }

  void SetCompilerWorkspaces(CompilerWorkspaceStorageSet storages) {
    compiler_workspaces_ = std::move(storages);
  }

  // Set manual buffer version overrides (from annotate_buffer_versions).
  // Keys are buffer data variables, values are fixed num_versions.
  void SetBufferVersionOverrides(BufferVersionMap overrides) {
    buffer_version_overrides_ = std::move(overrides);
  }

  // Set the user-declared manual multi-buffer table (from
  // annotate_manual_multi_buffer), threaded into dependency analysis so
  // cross-iteration distances are solved per access pair.
  void SetManualBufferVersions(BufferVersionMap manual_buffer_versions) {
    manual_buffer_versions_ = std::move(manual_buffer_versions);
    dependency_cache_.clear();
  }

  void SetRootConflictHints(ConflictHintList root_conflict_hints) {
    root_conflict_hints_ = std::move(root_conflict_hints);
    dependency_cache_.clear();
  }

  const BufferVersionMap &GetSelectedBufferVersions() const {
    return selected_buffer_versions_;
  }

private:
  static bool HasRequestedStage_(
      const std::vector<std::shared_ptr<IRStructure>> &children) {
    bool has_unscheduled = false;
    bool has_requested = false;
    for (const auto &node : children) {
      int stage = node->GetStage();
      ICHECK_GE(stage, kUnscheduledStage)
          << "Schedule-unit stage must be -1 or non-negative, got " << stage;
      has_unscheduled |= stage == kUnscheduledStage;
      has_requested |= stage != kUnscheduledStage;
    }
    ICHECK(!(has_unscheduled && has_requested))
        << "AutoSchedule expects each child list to be either entirely "
           "unscheduled (stage=-1) or entirely manually staged (stage>=0); "
           "MaterializeScheduleUnits must normalize unannotated manual "
           "siblings to stage 0";
    return has_requested;
  }

  int RequestedStage_(const IRStructure *node) const {
    int stage = node->GetStage();
    if (stage == kUnscheduledStage)
      return 0;
    return stage;
  }

  void MergeSelectedBufferVersion(const Var &storage, int num_versions) {
    ICHECK_GT(num_versions, 0);
    auto existing = selected_buffer_versions_.find(storage);
    if (existing != selected_buffer_versions_.end()) {
      // Disjoint owners are scheduled independently but share one physical
      // storage ring, so keep enough versions for every owner. Automatic
      // dav-2201 UB/L0 queries reserve other root allocations and carry this
      // existing count as a hard lower bound before selecting again. Selected
      // L0 phase contracts also require this reservation in manual ancestors.
      num_versions = std::max((*existing).second, num_versions);
    }
    for (const Var &member : storage_groups_.Members(storage))
      selected_buffer_versions_.Set(member, num_versions);
  }

  bool prefer_handoff_issue_{false};
  bool prefer_prefetch_issue_{false};
  bool reserve_l0_storage_{false};
  bool reserve_ub_storage_{false};
  std::vector<L0PhaseReusePair> l0_phase_reuse_;
  L0StorageGroups storage_groups_;
  ffi::Map<Var, PrimExpr> storage_slot_bytes_;
  CompilerWorkspaceStorageSet compiler_workspaces_;
  std::map<std::string, int64_t> memory_limits_;
  BufferVersionMap buffer_version_overrides_;
  BufferVersionMap manual_buffer_versions_;
  MultiBufferOwnerMap multi_buffer_owners_;
  ConflictHintList root_conflict_hints_;
  DependencyCache dependency_cache_;
  BufferVersionMap selected_buffer_versions_;
};

// Scratch timing state used only while AutoSchedule is solving and ordering an
// IRStructure tree. Task latency and II live directly on IRStructure because
// they are ordinary T.Task metadata.
class AutoScheduleExtraInfo final : public IRExtraInfo {
public:
  std::unique_ptr<IRExtraInfo> Clone() const final {
    return std::make_unique<AutoScheduleExtraInfo>(*this);
  }

  void SetStartTime(int64_t start_time) { start_time_ = start_time; }
  int64_t GetStartTime() const { return start_time_; }
  void SetIIperIter(int64_t ii_per_iter) { ii_per_iter_ = ii_per_iter; }
  int64_t GetIIperIter() const { return ii_per_iter_; }

private:
  int64_t start_time_{0};
  int64_t ii_per_iter_{0};
};

AutoScheduleExtraInfo *GetAutoScheduleExtraInfo(IRStructure *node) {
  return node->GetExtraInfo<AutoScheduleExtraInfo>();
}

int64_t GetTaskTimestamp(IRStructure *subtree_root, TaskNode *task,
                         bool producer_side) {
  std::vector<IRStructure *> path = task->PathFrom(subtree_root);
  int64_t timestamp = 0;
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    ICHECK(path[i]->IsControl());
    auto *control = static_cast<ControlNode *>(path[i]);
    if (producer_side) {
      int64_t trip_count = control->GetTripCount();
      int64_t ii_per_iter = GetAutoScheduleExtraInfo(control)->GetIIperIter();
      timestamp += (trip_count > 0 ? trip_count - 1 : 0) * ii_per_iter;
    }
    timestamp += GetAutoScheduleExtraInfo(path[i + 1])->GetStartTime();
  }
  return timestamp;
}

int64_t GetDependencyLatency(const DepInfo &dependency, bool reverse = false) {
  int64_t latency = 0;
  for (const auto &[forward_producer, forward_consumer] :
       dependency.task_pairs) {
    TaskNode *producer = reverse ? forward_consumer : forward_producer;
    TaskNode *consumer = reverse ? forward_producer : forward_consumer;
    IRStructure *producer_root =
        reverse ? dependency.cons_node : dependency.prod_node;
    IRStructure *consumer_root =
        reverse ? dependency.prod_node : dependency.cons_node;
    int64_t producer_time = GetTaskTimestamp(producer_root, producer, true);
    int64_t consumer_time = GetTaskTimestamp(consumer_root, consumer, false);
    latency = std::max(latency,
                       producer_time + producer->GetLatency() - consumer_time);
  }
  return latency;
}

std::unique_ptr<IRExtraInfo> MakeAutoScheduleExtraInfo() {
  return std::make_unique<AutoScheduleExtraInfo>();
}

bool SameVar(const Var &a, const Var &b) { return a.same_as(b); }

bool HasVarDependency(const IRStructure *a, const IRStructure *b) {
  for (const auto &write_var_a : a->GetWriteVars()) {
    for (const auto &read_var_b : b->GetReadVars()) {
      if (SameVar(write_var_a, read_var_b))
        return true;
    }
  }
  return false;
}

bool HasResourceDependency(const IRStructure *a, const IRStructure *b,
                           uint16_t mask) {
  if (((a->GetPipeMask() & b->GetPipeMask()) & mask) != 0)
    return true;
  return (a->GetHbmMask() & b->GetHbmMask()) != 0;
}

CoreMask TaskExecutionCore(const TaskNode *task) {
  CoreMask candidate = GetTaskCandidateCoreMask(task);
  CoreMask lexical = task->GetCoreMask();
  return lexical == kCoreUnassigned ? candidate : candidate & lexical;
}

CoreMask NodeExecutionCore(IRStructure *node) {
  CoreMask result = kCoreUnassigned;
  std::vector<TaskNode *> leaves;
  CollectAllTaskNodes(node, leaves);
  for (TaskNode *task : leaves) {
    // Scalar bookkeeping with no memory or SPR effects is domain-neutral.
    if (task->GetCoreMask() == kCoreUnassigned &&
        task->UsesOnlyPipe(ResourcePipe::kScalar) &&
        task->GetReadRegions().empty() && task->GetWriteRegions().empty() &&
        task->GetSpecialWriteMask() == 0 &&
        task->GetExplicitSpecialReadMask() == 0)
      continue;
    CoreMask core = TaskExecutionCore(task);
    if (!IsConcreteCore(core))
      return kCoreBroadcast;
    result |= core;
  }
  return result;
}

// Derive an issue-quality preference from the shared leaf access facts. This
// does not add, erase or redirect a completion dependency. A forward handoff is
// a concrete-core writer -> other-core reader of the dependency storage. The
// opposite read -> write slot-release edge remains a hard constraint, but is
// not mistaken for a forward data consumer here.
std::vector<std::pair<int64_t, int64_t>>
CollectHandoffIssuePreferences(const std::vector<IRStructure *> &nodes,
                               const std::vector<DepInfo> &deps) {
  const size_t n = nodes.size();
  std::unordered_map<IRStructure *, size_t> index;
  std::vector<CoreMask> cores(n, kCoreUnassigned);
  for (size_t i = 0; i < n; ++i) {
    index[nodes[i]] = i;
    cores[i] = NodeExecutionCore(nodes[i]);
  }
  std::vector<bool> handoff(n, false), blocking_consumer(n, false);
  std::vector<std::vector<size_t>> predecessors(n), successors(n);
  for (const DepInfo &dep : deps) {
    if (dep.kind != DependencyKind::kData)
      continue;
    size_t i = index.at(dep.prod_node), j = index.at(dep.cons_node);
    if (dep.distance == 0 && i != j) {
      // A dependency through an ambiguous or other-core intermediate still
      // excludes a producer from being independent of this consumer.
      successors[i].push_back(j);
      if (IsConcreteCore(cores[i]) && cores[i] == cores[j])
        predecessors[j].push_back(i);
    }
    if (!dep.storage.has_value())
      continue;
    for (const auto &[producer, consumer] : dep.task_pairs) {
      CoreMask pcore = TaskExecutionCore(producer);
      CoreMask ccore = TaskExecutionCore(consumer);
      if (!IsConcreteCore(pcore) || !IsConcreteCore(ccore) || pcore == ccore ||
          cores[i] != pcore || cores[j] != ccore)
        continue;
      const Var &storage = dep.storage.value();
      // Do not infer a unique forward role from an ambiguous read-modify-write
      // pair. All of its original dependencies still go to Z3 and InsertSync.
      if (!producer->WritesStorage(storage) ||
          !consumer->ReadsStorage(storage) ||
          (producer->ReadsStorage(storage) && consumer->WritesStorage(storage)))
        continue;
      handoff[i] = true;
      blocking_consumer[j] = true;
    }
  }

  std::vector<bool> productive(n, false);
  for (size_t seed = 0; seed < n; ++seed) {
    if (!handoff[seed])
      continue;
    std::vector<size_t> pending{seed};
    std::vector<bool> seen(n, false);
    while (!pending.empty()) {
      size_t task = pending.back();
      pending.pop_back();
      if (seen[task] || blocking_consumer[task])
        continue;
      seen[task] = true;
      productive[task] = true;
      pending.insert(pending.end(), predecessors[task].begin(),
                     predecessors[task].end());
    }
  }

  std::vector<std::pair<int64_t, int64_t>> result;
  for (size_t consumer = 0; consumer < n; ++consumer) {
    if (!blocking_consumer[consumer])
      continue;
    std::vector<bool> depends_on_consumer(n, false);
    std::vector<size_t> pending{consumer};
    while (!pending.empty()) {
      size_t task = pending.back();
      pending.pop_back();
      if (depends_on_consumer[task])
        continue;
      depends_on_consumer[task] = true;
      pending.insert(pending.end(), successors[task].begin(),
                     successors[task].end());
    }
    for (size_t producer = 0; producer < n; ++producer) {
      if (productive[producer] && !depends_on_consumer[producer] &&
          cores[producer] == cores[consumer])
        result.emplace_back(producer, consumer);
    }
  }
  return result;
}

// A preference is a typed asynchronous GM load's forward RAW consumer, not
// its reverse slot-release edge. Shared dependency facts still establish all
// readiness, last-reader, owner, tail and SPR constraints independently.
std::vector<std::pair<int64_t, int64_t>> CollectPrefetchIssuePreferences(
    const std::vector<IRStructure *> &nodes, const std::vector<DepInfo> &deps,
    const std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
        &storage_version_id,
    const MultiBufferOwnerMap &owners, ControlNode *loop) {
  std::unordered_map<IRStructure *, size_t> index;
  std::vector<CoreMask> cores;
  for (size_t i = 0; i < nodes.size(); ++i) {
    index[nodes[i]] = i;
    cores.push_back(NodeExecutionCore(nodes[i]));
  }
  std::unordered_map<const TaskNode *,
                     std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>>
      load_storages;
  auto is_load = [&](const TaskNode *task, const Var &storage) {
    if (!task->UsesOnlyPipe(ResourcePipe::kMTE2))
      return false;
    auto [entry, inserted] = load_storages.try_emplace(task);
    if (inserted) {
      PostOrderVisit(task->stmt, [&](const ObjectRef &node) {
        const auto *call = node.as<CallNode>();
        if (!call || !IsAscendCopyCall(call))
          return;
        AscendCopy copy(call->args, call->annotations);
        if (IsGlobalBuffer(copy->src) &&
            (IsSharedBuffer(copy->dst) || IsL1Buffer(copy->dst)))
          entry->second.insert(copy->dst->data);
      });
    }
    return entry->second.count(storage) != 0;
  };
  std::set<std::pair<int64_t, int64_t>> pairs;
  for (const DepInfo &dep : deps) {
    if (dep.kind != DependencyKind::kData || dep.distance != 0 ||
        !dep.storage.has_value())
      continue;
    const Var &storage = dep.storage.value();
    auto owner = owners.find(storage);
    if (!storage_version_id.count(storage) ||
        !loop->IsMultiBufferEligible(storage) || owner == owners.end() ||
        owner->second.size() != 1 || owner->second[0] != loop)
      continue;
    size_t i = index.at(dep.prod_node), j = index.at(dep.cons_node);
    if (i == j || !IsConcreteCore(cores[i]) || cores[i] != cores[j] ||
        (nodes[i]->GetPipeMask() & nodes[j]->GetPipeMask()) != 0 ||
        nodes[i]->ReadsStorage(storage) || nodes[j]->WritesStorage(storage))
      continue;
    for (const auto &[producer, consumer] : dep.task_pairs) {
      if (TaskExecutionCore(producer) != cores[i] ||
          TaskExecutionCore(consumer) != cores[j] ||
          !producer->WritesStorage(storage) ||
          producer->ReadsStorage(storage) || !consumer->ReadsStorage(storage) ||
          consumer->WritesStorage(storage) || !is_load(producer, storage))
        continue;
      pairs.emplace(i, j);
    }
  }
  return {pairs.begin(), pairs.end()};
}

ScheduleBuilder::LetDependencyClosure
ScheduleBuilder::CollectLetDependencyClosure_(
    const std::vector<IRStructure *> &nodes) {
  LetDependencyClosure result;
  result.is_declaration.resize(nodes.size(), false);
  result.dependencies.resize(nodes.size());

  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
      declaration_by_var;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const IRStructure *node = nodes[i];
    if (!node || !node->IsTask())
      continue;
    const auto *task = static_cast<const TaskNode *>(node);
    if (ffi::Optional<Bind> bind = GetFlatTaskBind(task->stmt);
        bind.defined()) {
      result.is_declaration[i] = true;
      declaration_by_var[bind.value()->var] = i;
    }
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    if (!nodes[i])
      continue;
    std::vector<Var> pending = nodes[i]->GetReadVars();
    while (!pending.empty()) {
      Var var = std::move(pending.back());
      pending.pop_back();
      auto it = declaration_by_var.find(var);
      if (it == declaration_by_var.end())
        continue;
      size_t declaration = it->second;
      if (!result.dependencies[i].insert(declaration).second)
        continue;
      std::vector<Var> reads = nodes[declaration]->GetReadVars();
      pending.insert(pending.end(), reads.begin(), reads.end());
    }
  }
  return result;
}

/*
  Recursively schedule root node, after scheduling IRStructure satisfies the
following properties: 1) Each ordered child list (the root list or a
ControlNode body) is reordered by the Z3 scheduler and the children carry
scheduling metadata directly. 2) For each ControlNode, its `children` are the
Z3-scheduled loop body. Conditional execution is carried by each node's
`conditions` delta (no dedicated IfNode), so guarded tasks are ordinary
schedule candidates.
*/
void ScheduleBuilder::ScheduleList(
    std::vector<std::shared_ptr<IRStructure>> &children) {
  multi_buffer_owners_ = CollectMultiBufferOwners(children, storage_groups_);
  dependency_cache_.clear();
  std::vector<std::shared_ptr<IRStructure>> origin_children = children;
  bool manual_schedule = HasRequestedStage_(origin_children);
  for (size_t i = 0; i < origin_children.size(); ++i) {
    origin_children[i]->SetIndex(i);
    origin_children[i]->SetParent(nullptr);
  }
  for (auto &child : origin_children) {
    ScheduleRecursive(child);
  }
  children = Z3SchedulePython(origin_children, manual_schedule);
  for (size_t i = 0; i < children.size(); ++i) {
    children[i]->SetIndex(i);
    children[i]->SetParent(nullptr);
  }
}

void ScheduleBuilder::ScheduleRecursive(std::shared_ptr<IRStructure> &node) {
  if (!node)
    return;

  if (node->IsTask()) {
    return;
  } else if (node->IsControl()) {
    auto ctrl = static_cast<ControlNode *>(node.get());
    std::vector<std::shared_ptr<IRStructure>> origin_children = ctrl->children;
    bool manual_schedule = HasRequestedStage_(origin_children);
    for (size_t i = 0; i < origin_children.size(); ++i) {
      origin_children[i]->SetIndex(i);
      origin_children[i]->SetParent(ctrl);
    }
    for (auto &child : origin_children) {
      ScheduleRecursive(child);
    }
    Z3SchedulePythonLoop(ctrl, manual_schedule);
    for (size_t i = 0; i < ctrl->children.size(); ++i) {
      ctrl->children[i]->SetIndex(i);
      ctrl->children[i]->SetParent(ctrl);
    }
    return;
  }

  LOG(FATAL) << "[ScheduleRecursive] Unknown IRStructure type" << node.get();
}

std::vector<std::shared_ptr<IRStructure>> ScheduleBuilder::Z3SchedulePython(
    const std::vector<std::shared_ptr<IRStructure>> &children,
    bool manual_schedule) {
  size_t n = children.size();
  std::vector<std::shared_ptr<IRStructure>> scheduled = children;
  for (const auto &node : scheduled) {
    ICHECK_EQ(RequestedStage_(node.get()), 0)
        << "T.Stage with a non-zero stage must be inside a serial loop "
           "annotated with enable_offset=True";
    // MaterializeScheduleUnits uses -1 to distinguish unscheduled units. A
    // straight-line list has no software-pipeline offset, so scheduling it
    // always publishes stage 0, including all fallback paths below.
    node->SetStage(0);
    GetAutoScheduleExtraInfo(node.get())->SetStartTime(0);
  }

  if (n <= 1) {
    return scheduled;
  }

  try {
    // Get the Python-registered function using ffi::Function::GetGlobal
    static std::optional<ffi::Function> z3_schedule_func =
        ffi::Function::GetGlobal("tl.transform.z3_schedule_python");
    if (!z3_schedule_func.has_value()) {
      ICHECK(!manual_schedule)
          << "Python Z3 scheduler is required for manual AutoSchedule";
      LOG(WARNING) << "Python Z3 scheduler not registered, falling back to "
                      "topological sort";
      return scheduled;
    }

    std::vector<IRStructure *> nodes;
    nodes.reserve(n);
    for (const auto &u : scheduled) {
      nodes.push_back(u.get());
    }

    // Prepare input data
    std::vector<int64_t> latencies;
    std::vector<int64_t> iis;
    std::vector<int64_t> resource_flags;
    std::vector<std::tuple<int64_t, int64_t, int64_t>> data_deps;
    std::vector<std::pair<int64_t, int64_t>> resource_deps;
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        owner_exclusion_deps;
    std::vector<std::pair<int64_t, int64_t>> pipe_order_deps;

    latencies.reserve(n);
    iis.reserve(n);
    resource_flags.reserve(n);

    for (size_t i = 0; i < n; ++i) {
      const IRStructure *node = nodes[i];
      latencies.push_back(node->GetLatency());
      iis.push_back(node->GetII());

      // Pass pipe mask directly as resource flags
      resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
    }

    // Variable (Let-var write -> read) data dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        if (HasVarDependency(nodes[i], nodes[j])) {
          data_deps.emplace_back(i, j, latencies[i]);
        }
      }
    }

    // Special-register ordering: a node writing register R must be ordered
    // against every node reading R (RAW forward, WAR backward) and every
    // other writer of R (WAW), preserving program order. loop_break writes
    // kLoopControl, which every node reads, so it fences against all
    // siblings; hf32/pad match only their consumers.
    for (size_t i = 0; i < n; ++i) {
      // Root Load3D initialization uses the shared storage-free DepInfo edge.
      uint16_t wmask = nodes[i]->GetSpecialWriteMask() &
                       ~static_cast<uint16_t>(SpecialRegister::kLoad3DConfig);
      if (wmask == 0)
        continue;
      for (size_t j = 0; j < n; ++j) {
        if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
          continue;
        if (i < j) {
          data_deps.emplace_back(i, j, latencies[i]);
        } else {
          data_deps.emplace_back(j, i, 0);
        }
      }
      for (size_t j = i + 1; j < n; ++j) {
        if ((wmask & nodes[j]->GetSpecialWriteMask()) != 0)
          data_deps.emplace_back(i, j, latencies[i]);
      }
    }

    // Buffer-region data dependencies
    auto deps = AnalyzeDependencies(
        nodes, /*loop=*/nullptr, manual_buffer_versions_, multi_buffer_owners_,
        &dependency_cache_, root_conflict_hints_);
    std::unordered_map<IRStructure *, size_t> node_idx;
    for (size_t i = 0; i < n; ++i) {
      node_idx[nodes[i]] = i;
    }
    for (const auto &dep : deps) {
      size_t i = node_idx[dep.prod_node];
      size_t j = node_idx[dep.cons_node];
      if (dep.kind == DependencyKind::kOwnerExclusion) {
        int64_t forward = GetDependencyLatency(dep);
        int64_t reverse = GetDependencyLatency(dep, /*reverse=*/true);
        if (i > j) {
          std::swap(i, j);
          std::swap(forward, reverse);
        }
        owner_exclusion_deps.emplace_back(i, j, forward, reverse);
      } else {
        data_deps.emplace_back(i, j, GetDependencyLatency(dep));
      }
    }

    // Collect resource dependencies
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        uint16_t shared_pipes =
            nodes[i]->GetPipeMask() & nodes[j]->GetPipeMask();
        if (HasResourceDependency(nodes[i], nodes[j])) {
          resource_deps.emplace_back(i, j);
        }
        if (manual_schedule && shared_pipes != 0) {
          pipe_order_deps.emplace_back(i, j);
        }
      }
    }

    // Convert vectors to TVM containers
    ffi::Array<int64_t> tvm_latencies;
    ffi::Array<int64_t> tvm_iis;
    ffi::Array<int64_t> tvm_resource_flags;
    ffi::Array<ffi::Array<int64_t>> tvm_data_deps;
    ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;
    ffi::Array<ffi::Array<int64_t>> tvm_owner_exclusion_deps;
    ffi::Array<ffi::Array<int64_t>> tvm_pipe_order_deps;

    for (auto val : latencies) {
      tvm_latencies.push_back(val);
    }
    for (auto val : iis) {
      tvm_iis.push_back(val);
    }
    for (auto val : resource_flags) {
      tvm_resource_flags.push_back(val);
    }
    for (const auto &dep : data_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(std::get<0>(dep));
      pair.push_back(std::get<1>(dep));
      pair.push_back(std::get<2>(dep));
      tvm_data_deps.push_back(pair);
    }
    for (const auto &dep : resource_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_resource_deps.push_back(pair);
    }
    for (const auto &dep : owner_exclusion_deps) {
      ffi::Array<int64_t> tuple;
      tuple.push_back(std::get<0>(dep));
      tuple.push_back(std::get<1>(dep));
      tuple.push_back(std::get<2>(dep));
      tuple.push_back(std::get<3>(dep));
      tvm_owner_exclusion_deps.push_back(tuple);
    }
    for (const auto &dep : pipe_order_deps) {
      ffi::Array<int64_t> pair;
      pair.push_back(dep.first);
      pair.push_back(dep.second);
      tvm_pipe_order_deps.push_back(pair);
    }

    // Extract results
    // Python function returns only start_times, C++ side will sort by
    // start_time
    auto start_times =
        z3_schedule_func
            .value()(tvm_latencies, tvm_iis, tvm_resource_flags, tvm_data_deps,
                     tvm_resource_deps, tvm_owner_exclusion_deps,
                     tvm_pipe_order_deps)
            .cast<ffi::Array<int64_t>>();

    if (start_times.size() != n) {
      ICHECK(!manual_schedule)
          << "Python Z3 scheduler returned invalid manual schedule results";
      LOG(WARNING) << "Python Z3 scheduler returned invalid results (size "
                      "mismatch), falling back to topological sort";
      return scheduled;
    }

    // Write start_times onto the scheduled nodes, then sort them in place by
    // start_time (stable: ties keep original program order).
    for (size_t i = 0; i < n; ++i) {
      GetAutoScheduleExtraInfo(nodes[i])->SetStartTime(start_times[i]);
    }

    std::stable_sort(
        scheduled.begin(), scheduled.end(),
        [](const std::shared_ptr<IRStructure> &a,
           const std::shared_ptr<IRStructure> &b) {
          return GetAutoScheduleExtraInfo(a.get())->GetStartTime() <
                 GetAutoScheduleExtraInfo(b.get())->GetStartTime();
        });
    return scheduled;

  } catch (const std::exception &e) {
    ICHECK(!manual_schedule)
        << "Manual per-pipe scheduling failed: " << e.what();
    LOG(WARNING) << "Python Z3 scheduler failed with exception: " << e.what()
                 << ", falling back to topological sort";
    return scheduled;
  } catch (...) {
    ICHECK(!manual_schedule)
        << "Manual per-pipe scheduling failed with unknown exception";
    LOG(WARNING) << "Python Z3 scheduler failed with unknown exception, "
                    "falling back to topological sort";
    return scheduled;
  }
}

// Z3-based scheduler for loops that calls Python implementation via FFI
// with distance-aware dependencies
void ScheduleBuilder::Z3SchedulePythonLoop(ControlNode *ctrl,
                                           bool manual_schedule) {
  if (ctrl->children.empty()) {
    LOG(WARNING)
        << "Z3SchedulePythonLoop called on a control node without children";
    return;
  }

  std::vector<IRStructure *> nodes;
  nodes.reserve(ctrl->children.size());
  for (const auto &child : ctrl->children) {
    nodes.push_back(child.get());
  }

  size_t n = nodes.size();

  std::vector<int64_t> manual_stages;
  manual_stages.reserve(n);
  int max_manual_stage = 0;
  for (const IRStructure *node : nodes) {
    int stage = RequestedStage_(node);
    manual_stages.push_back(stage);
    max_manual_stage = std::max(max_manual_stage, stage);
  }

  auto num_stages = 1;
  auto num_stages_val = ctrl->control.get()->annotations.Get("num_stages");
  if (num_stages_val.has_value()) {
    num_stages = num_stages_val.value().cast<IntImm>()->value;
  }
  ICHECK_GT(num_stages, 0)
      << "Ascend pipeline buffer-version bound must be positive";
  bool auto_pipeline = false;
  if (auto intent = ctrl->control->annotations.Get(kAutoPipeline))
    auto_pipeline = intent.value().cast<bool>();
  int stage_horizon = 0;
  if (auto_pipeline && !manual_schedule) {
    // Bound the search by the number of active asynchronous endpoints and the
    // frontend maximum (four by default). This is a small optimization horizon,
    // not a completeness guarantee or a required depth. The joint solve still
    // chooses fewer stages/versions when dependencies or capacity favor that.
    uint16_t pipes = ctrl->GetPipeMask() & ~PipeMask(ResourcePipe::kScalar);
    int pipe_count = 0;
    while (pipes != 0) {
      pipe_count += pipes & 1;
      pipes >>= 1;
    }
    num_stages = std::min(num_stages, std::max(pipe_count, 1));
    num_stages = static_cast<int>(std::min<int64_t>(
        num_stages, std::max<int64_t>(ctrl->GetTripCount(), 1)));
    stage_horizon = num_stages;
  }

  // When false, the Z3 loop scheduler keeps each resource's tasks within a
  // single II window (max(start) - min(start) < II). Sourced from the loop's
  // "enable_offset" annotation. Automatic Pipelined intent enables offsets;
  // otherwise preserve upstream's false default (constraint applied).
  bool enable_offset = auto_pipeline;
  auto enable_offset_val =
      ctrl->control.get()->annotations.Get("enable_offset");
  if (enable_offset_val.has_value()) {
    enable_offset = enable_offset_val.value().cast<bool>();
  }
  if (!auto_pipeline && !manual_schedule && !num_stages_val.has_value() &&
      !enable_offset) {
    // A plain serial loop (including explicit Pipelined(num_stages=0))
    // has no epoch-offset request. Keep every task in its original iteration.
    // Leaving k unconstrained makes the solver search unbounded cross-pipe
    // stages even though no pipeline was requested. Explicit
    // positive version bounds and enable_offset retain their upstream policy.
    stage_horizon = 1;
  }

  if (manual_schedule) {
    ICHECK(
        !ctrl->control.get()->annotations.Get("tl_pipeline_order").has_value())
        << "Ascend manual AutoSchedule derives each pipe's issue order from "
           "source order; remove the T.Pipelined order argument";
    ICHECK(
        !ctrl->control.get()->annotations.Get("tl_pipeline_stage").has_value())
        << "Ascend manual AutoSchedule uses T.Stage scopes; remove the "
           "T.Pipelined stage argument";
    if (!enable_offset) {
      ICHECK_EQ(max_manual_stage, 0)
          << "Non-zero T.Stage values require enable_offset=True on the "
             "enclosing loop";
    }
    if (!num_stages_val.has_value() || auto_pipeline) {
      num_stages = max_manual_stage + 1;
    }
  }

  bool preserve_issue_order = false;
  if (auto intent = ctrl->control->annotations.Get(kPreserveIssueOrder))
    preserve_issue_order = intent.value().cast<bool>();
  if (preserve_issue_order) {
    ICHECK(manual_schedule)
        << kPreserveIssueOrder << " requires explicit T.Stage scopes";
    ICHECK(!ctrl->GetParentControl())
        << kPreserveIssueOrder << " currently supports only the outermost loop";
  }

  // Prepare input data
  std::vector<int64_t> latencies;
  std::vector<int64_t> iis;
  std::vector<int64_t> resource_flags;
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
      data_deps; // (i, j, distance, latency)
  std::vector<std::pair<int64_t, int64_t>> resource_deps;
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
      owner_exclusion_deps;
  std::vector<std::pair<int64_t, int64_t>> pipe_order_deps;
  std::vector<std::pair<int64_t, int64_t>> stage_order_deps;

  latencies.reserve(n);
  iis.reserve(n);
  resource_flags.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    const IRStructure *node = nodes[i];
    latencies.push_back(node->GetLatency());
    iis.push_back(node->GetII());
    resource_flags.push_back(static_cast<int64_t>(node->GetPipeMask()));
  }

  // Keep one representative Buffer per storage for shape/scope accounting.
  // Version variables and dependency distances are keyed directly by storage.
  struct StorageResource {
    Var storage;
    Buffer representative;
  };
  std::vector<int64_t> storage_sizes;
  std::vector<StorageResource> storage_resources;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> seen_storages;
  std::unordered_map<Var, size_t, ObjectPtrHash, ObjectPtrEqual>
      storage_version_id;
  // Build memory groups from memory_limits_ map.
  // std::map iteration is ordered by key: "shared" then "shared.l1".
  std::map<std::string, int> scope_to_group;
  std::vector<int64_t> group_capacities;
  std::vector<std::string> group_scopes;
  std::vector<std::vector<int64_t>> group_indices;
  // Independent int64 byte-representation bounds, never an aggregate GM
  // hardware capacity or a charge against an on-chip memory pool.
  std::vector<int64_t> workspace_version_ids;
  for (const auto &entry : memory_limits_) {
    if (entry.second > 0) {
      scope_to_group[entry.first] = (int)group_capacities.size();
      group_capacities.push_back(entry.second);
      group_scopes.push_back(entry.first);
      group_indices.push_back({});
    }
  }
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      annotated_storage_versions;
  auto get_storage_size = [&](const Var &storage) -> int64_t {
    auto slot = storage_slot_bytes_.Get(storage);
    ICHECK(slot.has_value())
        << "AutoSchedule cannot find the allocation owner for storage "
        << storage->name_hint;
    arith::Analyzer analyzer;
    PrimExpr size = analyzer.Simplify(slot.value());
    const auto *constant = size.as<IntImmNode>();
    ICHECK(constant)
        << "AutoSchedule requires a static physical storage footprint for "
        << storage->name_hint << ", got " << size;
    ICHECK_GE(constant->value, 0)
        << "AutoSchedule found a negative physical storage footprint for "
        << storage->name_hint;
    return constant->value;
  };
  auto get_buffer_size = [&](const Buffer &buffer) {
    return get_storage_size(buffer->data);
  };
  auto consume_fixed_capacity = [&](int group, int64_t slot_bytes,
                                    int versions) {
    if (preserve_issue_order) {
      __int128 bytes = __int128(slot_bytes) * versions;
      ICHECK(bytes <= group_capacities[group])
          << "Explicit Ascend source issue order is not legal: "
             "source-order-memory-capacity for fixed storage";
      group_capacities[group] -= static_cast<int64_t>(bytes);
    } else if (reserve_ub_storage_ && group_scopes[group] == "shared") {
      __int128 remaining =
          __int128(group_capacities[group]) - __int128(slot_bytes) * versions;
      ICHECK(remaining >= std::numeric_limits<int64_t>::min())
          << "Ascend UB reservation exceeds int64 byte representation";
      group_capacities[group] = static_cast<int64_t>(remaining);
    } else {
      group_capacities[group] -= slot_bytes * versions;
    }
  };
  // A selected phase pair commits the kernel to one completion-proven shared
  // bank. Every enclosing query must consume that same max-footprint contract,
  // including manual ancestors of an automatically versioned child. Otherwise
  // the ancestor either sums mutually exclusive allocations or loses the
  // child's established physical ring count. Unselected manual kernels retain
  // their existing per-query resource policy.
  const bool reserve_l0 =
      reserve_l0_storage_ && (!manual_schedule || !l0_phase_reuse_.empty());
  auto is_ub_storage = [](const Var &storage) {
    const std::string scope = GetPtrStorageScope(storage);
    return scope == "shared" || scope == "shared.dyn";
  };
  auto l0_scope = [](const Var &storage) -> std::string {
    const std::string scope = GetPtrStorageScope(storage);
    if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
      return "shared.l0a";
    if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
      return "shared.l0b";
    if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
      return "shared.l0c";
    return "";
  };
  auto reserved_versions = [&](const Var &storage) -> int {
    // The manual allocation already contains its physical version axis.
    if (manual_buffer_versions_.count(storage))
      return 1;
    int versions = selected_buffer_versions_.Get(storage).value_or(1);
    // A fixed count with an established owner will be applied when that owner
    // is scheduled; reserve it even if the owner has not been visited yet.
    if (multi_buffer_owners_.count(storage))
      versions = std::max(versions,
                          buffer_version_overrides_.Get(storage).value_or(1));
    return versions;
  };
  auto enumerate_buffers = [&](auto &&visit) {
    for (const auto &r : ctrl->GetReadRegions())
      visit(r->buffer);
    for (const auto &r : ctrl->GetWriteRegions())
      visit(r->buffer);
  };
  enumerate_buffers([&](const Buffer &buffer) {
    if (!IsSharedBuffer(buffer) && !IsL1Buffer(buffer) &&
        !IsL0CBuffer(buffer) && !IsL0ABuffer(buffer) && !IsL0BBuffer(buffer) &&
        !compiler_workspaces_.count(buffer->data)) {
      return;
    }
    if (seen_storages.insert(buffer->data).second)
      storage_resources.push_back({buffer->data, buffer});
  });

  for (const StorageResource &resource : storage_resources) {
    const Var &storage = resource.storage;
    const Buffer &buffer = resource.representative;
    if (preserve_issue_order && !manual_buffer_versions_.count(storage)) {
      auto requested = buffer_version_overrides_.Get(storage);
      // Stage order defines logical access order in the existing API. A
      // read-first recurrence therefore cannot acquire an automatic version
      // owner just because source-order completion was requested. The partial
      // scheduler warns about an unapplied override; the complete contract
      // must reject it rather than silently publish fewer physical versions.
      ICHECK(!requested.has_value() || requested.value() <= 1 ||
             multi_buffer_owners_.count(storage))
          << "Explicit Ascend source issue order is not legal: "
             "source-order-unowned-fixed-version for storage "
          << storage->name_hint
          << "; the requested ring has no eligible loop owner";
    }
    if (compiler_workspaces_.count(storage)) {
      if (!ctrl->IsMultiBufferEligible(storage))
        continue;
      int64_t slot_bytes = get_buffer_size(buffer);
      auto override_it = buffer_version_overrides_.find(storage);
      if (override_it != buffer_version_overrides_.end()) {
        int versions = (*override_it).second;
        CheckCompilerWorkspaceVersionBytes(storage, slot_bytes, versions);
        annotated_storage_versions[storage] = versions;
      } else {
        size_t id = storage_sizes.size();
        storage_version_id.emplace(storage, id);
        workspace_version_ids.push_back(static_cast<int64_t>(id));
        storage_sizes.push_back(slot_bytes);
      }
      continue;
    }
    std::string scope;
    if (IsL1Buffer(buffer)) {
      scope = "shared.l1";
    } else if (IsL0CBuffer(buffer)) {
      scope = "shared.l0c";
    } else if (IsL0ABuffer(buffer)) {
      scope = "shared.l0a";
    } else if (IsL0BBuffer(buffer)) {
      scope = "shared.l0b";
    } else {
      scope = "shared";
    }
    int group_id = scope_to_group[scope];
    bool root_l0 = reserve_l0 && !l0_scope(storage).empty();
    bool root_ub = reserve_ub_storage_ && is_ub_storage(storage);
    if (root_l0 && manual_buffer_versions_.count(storage))
      continue;
    if (root_ub && manual_buffer_versions_.count(storage)) {
      // Its allocation already includes the physical version axis.
      consume_fixed_capacity(group_id, get_buffer_size(buffer), 1);
      continue;
    }
    if (!ctrl->IsMultiBufferEligible(storage)) {
      // These buffers cannot be multi-buffered, and cannot share memory with
      // other buffers either, so we directly reduce the capacity of the
      // memory group by their size
      if (!root_l0) {
        int versions =
            preserve_issue_order || root_ub ? reserved_versions(storage) : 1;
        consume_fixed_capacity(group_id, get_buffer_size(buffer), versions);
      }
    } else {
      auto override_it = buffer_version_overrides_.find(storage);
      if (override_it != buffer_version_overrides_.end()) {
        int override_val = (*override_it).second;
        if ((root_l0 || root_ub) && preserve_issue_order)
          ICHECK_GE(override_val, reserved_versions(storage))
              << "Explicit Ascend source issue order has a fixed version count "
                 "smaller than an already selected shared ring for storage "
              << storage->name_hint;
        if (root_l0 || root_ub)
          override_val = std::max(override_val, reserved_versions(storage));
        annotated_storage_versions[storage] = override_val;
        if (!root_l0)
          consume_fixed_capacity(group_id, get_buffer_size(buffer),
                                 override_val);
      } else {
        size_t id = storage_sizes.size();
        storage_version_id.emplace(storage, id);
        group_indices[group_id].push_back((int64_t)id);
        storage_sizes.push_back(get_buffer_size(buffer));
      }
    }
  }

  std::vector<int64_t> storage_min_versions(storage_sizes.size(), 1);
  if (reserve_ub_storage_) {
    // Every root UB allocation occupies this AIV's arena, including epilogue
    // casts, normalization results and scratch absent from ctrl's accesses.
    // Accessed fixed storage was charged above; variable storage remains in
    // its memory group, with any previously selected physical ring as a lower
    // bound. Reserve only the remaining allocation owners here, once each.
    for (const auto &[storage, _] : storage_slot_bytes_) {
      if (!is_ub_storage(storage))
        continue;
      int versions = reserved_versions(storage);
      ICHECK_GT(versions, 0);
      auto variable = storage_version_id.find(storage);
      if (variable != storage_version_id.end()) {
        storage_min_versions[variable->second] = versions;
        continue;
      }
      if (seen_storages.count(storage))
        continue;
      int64_t bytes = get_storage_size(storage);
      int64_t &remaining = group_capacities[scope_to_group.at("shared")];
      ICHECK_LE(bytes, remaining / versions)
          << "Z3 loop scheduling failed: scope \"shared\" capacity "
          << remaining << " is insufficient for reserved root storage "
          << storage->name_hint << " (" << bytes << " bytes, " << versions
          << " physical versions)";
      remaining -= bytes * versions;
    }
  }
  if (reserve_l0) {
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> phase_storages;
    auto reserved_bytes = [&](const Var &storage) {
      int versions = reserved_versions(storage);
      ICHECK_GT(versions, 0);
      int64_t bytes = get_storage_size(storage);
      ICHECK_LE(bytes, std::numeric_limits<int64_t>::max() / versions);
      return bytes * versions;
    };
    for (const L0PhaseReusePair &pair : l0_phase_reuse_) {
      auto first = storage_version_id.find(pair.first);
      auto second = storage_version_id.find(pair.second);
      const bool first_variable = first != storage_version_id.end();
      const bool second_variable = second != storage_version_id.end();
      ICHECK(!(first_variable && second_variable))
          << "L0 phase reuse cannot budget two independently variable "
             "allocations in one scheduling query";
      ICHECK(phase_storages.insert(pair.first).second &&
             phase_storages.insert(pair.second).second);
      int64_t &remaining = group_capacities[scope_to_group.at(pair.scope)];
      int64_t first_bytes = reserved_bytes(pair.first);
      int64_t second_bytes = reserved_bytes(pair.second);
      if (!first_variable && !second_variable) {
        int64_t bytes = std::max(first_bytes, second_bytes);
        ICHECK_LE(bytes, remaining)
            << "L0 phase reuse exceeds scope " << pair.scope << " capacity";
        remaining -= bytes;
      } else {
        // Exactly two allocations occupy this bank. max(variable, fixed)
        // fits iff both fit independently; keep the variable's full byte
        // charge in its existing group and check its mate separately.
        ICHECK_LE(first_variable ? second_bytes : first_bytes, remaining)
            << "L0 phase reuse mate exceeds scope " << pair.scope
            << " capacity";
        const Var &variable = first_variable ? pair.first : pair.second;
        size_t id = first_variable ? first->second : second->second;
        storage_min_versions[id] = reserved_versions(variable);
      }
    }
    // Unselected root allocations retain the conservative sum reservation.
    // Chosen phase pairs retain distinct logical storage/version clocks;
    // InsertSync must certify their alias before the late packer consumes it.
    for (const auto &[storage, _] : storage_slot_bytes_) {
      std::string scope = l0_scope(storage);
      if (scope.empty() || phase_storages.count(storage))
        continue;
      int versions = reserved_versions(storage);
      ICHECK_GT(versions, 0);
      auto variable = storage_version_id.find(storage);
      if (variable != storage_version_id.end()) {
        storage_min_versions[variable->second] = versions;
        continue;
      }
      int64_t bytes = get_storage_size(storage);
      int group_id = scope_to_group.at(scope);
      int64_t &remaining = group_capacities[group_id];
      ICHECK_LE(bytes, remaining / versions)
          << "Z3 loop scheduling failed: scope \"" << scope << "\" capacity "
          << remaining << " is insufficient for reserved root storage "
          << storage->name_hint << " (" << bytes << " bytes, " << versions
          << " physical versions)";
      remaining -= bytes * versions;
    }
  }

  if (preserve_issue_order) {
    // Completion must budget the actual shared ring, including decisions
    // already made by independent nested owners. It cannot shrink that ring
    // merely because this owner's fixed order needs fewer versions.
    for (const auto &[storage, id] : storage_version_id)
      storage_min_versions[id] =
          std::max<int64_t>(storage_min_versions[id],
                            selected_buffer_versions_.Get(storage).value_or(1));
    for (const auto &[storage, fixed] : annotated_storage_versions)
      ICHECK_LE(selected_buffer_versions_.Get(storage).value_or(1), fixed)
          << "Explicit fixed buffer version count conflicts with an existing "
             "owner for "
          << storage->name_hint;
  }

  // Validate buffer capacities upfront: every eligible buffer needs at least
  // one version, or the existing shared ring count for automatic 2201 L0.
  // If that minimum exceeds the remaining capacity, Z3 cannot be feasible.
  for (size_t g = 0; g < group_capacities.size(); ++g) {
    int64_t capacity = group_capacities[g];
    const auto &idxs = group_indices[g];
    if (capacity < 0) {
      LOG(FATAL) << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
                 << "\" capacity is negative (" << capacity
                 << "). Non-eligible and fixed-version buffers already exceed"
                 << " the memory limit. Try using T.annotate_unlimit_memory(\""
                 << group_scopes[g] << "\") or reduce buffer sizes.";
    }
    if (!idxs.empty()) {
      int64_t min_usage = 0;
      for (int64_t idx : idxs) {
        ICHECK_LE(storage_sizes[idx],
                  (std::numeric_limits<int64_t>::max() - min_usage) /
                      storage_min_versions[idx]);
        min_usage += storage_sizes[idx] * storage_min_versions[idx];
      }
      if (min_usage > capacity) {
        LOG(FATAL) << "Z3 loop scheduling failed: scope \"" << group_scopes[g]
                   << "\" capacity " << capacity
                   << " is insufficient for minimum buffer usage " << min_usage
                   << ". Try using T.annotate_unlimit_memory(\""
                   << group_scopes[g] << "\") or reduce buffer sizes.";
      }
    }
  }

  // Variable (Let-var write -> read) data dependencies
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (HasVarDependency(nodes[i], nodes[j])) {
        data_deps.emplace_back(i, j, 0, latencies[i]);
      }
    }
  }

  // Special-register ordering: a node writing register R must be ordered
  // against every node reading R (RAW forward, WAR backward) and every other
  // writer of R (WAW), preserving program order. loop_break writes
  // kLoopControl, which every node reads, so it fences against all siblings;
  // hf32/pad match only their consumers. The distance bit encodes intra-iter
  // (0) vs next-iter (1) direction.
  for (size_t i = 0; i < n; ++i) {
    // Root Load3D initialization uses the shared storage-free DepInfo edge.
    uint16_t wmask = nodes[i]->GetSpecialWriteMask() &
                     ~static_cast<uint16_t>(SpecialRegister::kLoad3DConfig);
    if (wmask == 0)
      continue;
    for (size_t j = 0; j < n; ++j) {
      if (j == i || (wmask & nodes[j]->GetSpecialReadMask()) == 0)
        continue;
      if (i < j) {
        data_deps.emplace_back(i, j, 0, latencies[i]);
        data_deps.emplace_back(j, i, 1, 0);
      } else {
        data_deps.emplace_back(i, j, 1, latencies[i]);
        data_deps.emplace_back(j, i, 0, 0);
      }
    }
    for (size_t j = i + 1; j < n; ++j) {
      if ((wmask & nodes[j]->GetSpecialWriteMask()) == 0)
        continue;
      data_deps.emplace_back(i, j, 0, latencies[i]);
      data_deps.emplace_back(j, i, 1, latencies[j]);
    }
  }

  // Buffer-region data dependencies
  auto deps = AnalyzeDependencies(nodes, ctrl, manual_buffer_versions_,
                                  multi_buffer_owners_, &dependency_cache_,
                                  root_conflict_hints_);
  std::unordered_map<IRStructure *, size_t> node_idx;
  for (size_t i = 0; i < n; ++i) {
    node_idx[nodes[i]] = i;
  }
  for (const auto &dep : deps) {
    size_t i = node_idx[dep.prod_node];
    size_t j = node_idx[dep.cons_node];
    int64_t latency = GetDependencyLatency(dep);
    if (dep.kind == DependencyKind::kOwnerExclusion) {
      int64_t reverse_latency = GetDependencyLatency(dep, /*reverse=*/true);
      if (i > j) {
        std::swap(i, j);
        std::swap(latency, reverse_latency);
      }
      owner_exclusion_deps.emplace_back(i, j, latency, reverse_latency);
      continue;
    }
    if (dep.distance >= 0) {
      data_deps.emplace_back(i, j, dep.distance, latency);
    } else {
      int64_t distance = 1;
      if (dep.storage.has_value()) {
        Var storage = storage_groups_.Representative(dep.storage.value());
        auto it = storage_version_id.find(storage);
        if (it != storage_version_id.end()) {
          // Encode the storage-version id in the negative distance for the Z3
          // scheduler to recognize.
          distance = -(int64_t)it->second - 1;
        } else {
          auto ann_it = annotated_storage_versions.find(storage);
          if (ann_it != annotated_storage_versions.end()) {
            distance = ann_it->second;
          }
        }
      }
      data_deps.emplace_back(i, j, distance, latency);
    }
  }

  // Collect resource dependencies
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      uint16_t shared_pipes = nodes[i]->GetPipeMask() & nodes[j]->GetPipeMask();
      if (HasResourceDependency(nodes[i], nodes[j],
                                ~((uint16_t)ResourcePipe::kScalar))) {
        resource_deps.emplace_back(i, j);
      }
      if (manual_schedule && shared_pipes != 0) {
        pipe_order_deps.emplace_back(i, j);
      }
    }
  }

  // This is a soft issue preference after the minimum-II solve. All directed,
  // resource, owner, stage, version and SPR constraints remain hard. Compound
  // units keep their existing boundaries; no wait is moved into lane guards.
  std::vector<std::pair<int64_t, int64_t>> issue_priority_pairs;
  if (prefer_handoff_issue_ && auto_pipeline && !manual_schedule)
    issue_priority_pairs = CollectHandoffIssuePreferences(nodes, deps);

  // This refinement runs jointly over stage/residue/version variables at the
  // minimum II, before the Python scheduler fixes stages or emitted counts.
  std::vector<std::pair<int64_t, int64_t>> prefetch_priority_pairs;
  if (prefer_prefetch_issue_ && auto_pipeline && enable_offset &&
      !manual_schedule && num_stages > 1)
    prefetch_priority_pairs = CollectPrefetchIssuePreferences(
        nodes, deps, storage_version_id, multi_buffer_owners_, ctrl);

  // Stage-order constraints for copied Let variables.
  //
  // A Let task `a = buf[...]` only READS buffers (it writes a scalar var).
  // When `a` is consumed in a different pipeline stage, solve_conflict_var
  // clones the Let into that consumer's stage and prepends it to the loop
  // body. The clone re-reads `buf` at the consumer's iteration. If `buf` is
  // also WRITTEN inside the loop, that re-read must observe the value from
  // before the write — otherwise the clone reads a stale/overwritten buffer.
  //
  // AnalyzeDependencies already feeds the Let's own WAR edge to Z3, but only
  // constrains the Let's original stage, not the stages of nodes that will
  // receive a copy. Here we add: for every node U that transitively reads a
  // Let-defined var, and every WAR consumer W of that Let (a node that writes
  // the buffer the Let reads, same-iter or cross-iter), require k_U <= k_W in
  // the Z3 (k, r) model. Equality is allowed because the clone is placed at
  // the front of the loop body, so it reads before any write in that
  // iteration.
  if (enable_offset) {
    LetDependencyClosure let_closure = CollectLetDependencyClosure_(nodes);
    // Collect WAR consumers per Let node
    std::vector<std::set<size_t>> let_war_consumers(n);
    for (const auto &dep : deps) {
      if (dep.kind != DependencyKind::kData)
        continue;
      size_t p = node_idx[dep.prod_node];
      if (let_closure.is_declaration[p]) {
        let_war_consumers[p].insert(node_idx[dep.cons_node]);
      }
    }
    for (size_t u = 0; u < n; ++u) {
      for (size_t l : let_closure.dependencies[u]) {
        for (size_t w : let_war_consumers[l]) {
          if (u == w)
            continue;
          stage_order_deps.emplace_back((int64_t)u, (int64_t)w);
        }
      }
    }
  }

  // Convert vectors to TVM containers
  ffi::Array<int64_t> tvm_latencies;
  ffi::Array<int64_t> tvm_iis;
  ffi::Array<int64_t> tvm_resource_flags;
  ffi::Array<ffi::Array<int64_t>>
      tvm_data_deps; // each element is [i, j, distance, latency]
  ffi::Array<ffi::Array<int64_t>> tvm_resource_deps;
  ffi::Array<ffi::Array<int64_t>> tvm_owner_exclusion_deps;
  ffi::Array<ffi::Array<int64_t>> tvm_pipe_order_deps;
  ffi::Array<ffi::Array<int64_t>> tvm_issue_priority_pairs;
  ffi::Array<ffi::Array<int64_t>> tvm_prefetch_priority_pairs;
  ffi::Array<int64_t> tvm_storage_sizes;
  ffi::Array<int64_t> tvm_storage_min_versions;
  ffi::Array<int64_t> tvm_manual_stages;

  for (auto val : latencies) {
    tvm_latencies.push_back(val);
  }
  for (auto val : iis) {
    tvm_iis.push_back(val);
  }
  for (auto val : resource_flags) {
    tvm_resource_flags.push_back(val);
  }
  for (const auto &dep : data_deps) {
    ffi::Array<int64_t> triple;
    triple.push_back(std::get<0>(dep));
    triple.push_back(std::get<1>(dep));
    triple.push_back(std::get<2>(dep));
    triple.push_back(std::get<3>(dep));
    tvm_data_deps.push_back(triple);
  }
  for (const auto &dep : resource_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_resource_deps.push_back(pair);
  }
  for (const auto &dep : owner_exclusion_deps) {
    ffi::Array<int64_t> tuple;
    tuple.push_back(std::get<0>(dep));
    tuple.push_back(std::get<1>(dep));
    tuple.push_back(std::get<2>(dep));
    tuple.push_back(std::get<3>(dep));
    tvm_owner_exclusion_deps.push_back(tuple);
  }
  for (const auto &dep : pipe_order_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_pipe_order_deps.push_back(pair);
  }
  for (const auto &[producer, consumer] : issue_priority_pairs)
    tvm_issue_priority_pairs.push_back({producer, consumer});
  for (const auto &[producer, consumer] : prefetch_priority_pairs)
    tvm_prefetch_priority_pairs.push_back({producer, consumer});
  ffi::Array<ffi::Array<int64_t>> tvm_stage_order_deps;
  for (const auto &dep : stage_order_deps) {
    ffi::Array<int64_t> pair;
    pair.push_back(dep.first);
    pair.push_back(dep.second);
    tvm_stage_order_deps.push_back(pair);
  }
  for (int64_t size : storage_sizes) {
    tvm_storage_sizes.push_back(size);
  }
  for (int64_t versions : storage_min_versions)
    tvm_storage_min_versions.push_back(versions);
  for (int64_t stage : manual_stages) {
    tvm_manual_stages.push_back(stage);
  }

  // Build memory groups for FFI: each inner array is [capacity, idx0, idx1,
  // ...]
  ffi::Array<ffi::Array<int64_t>> tvm_memory_groups;
  for (size_t g = 0; g < group_capacities.size(); ++g) {
    ffi::Array<int64_t> group;
    group.push_back(group_capacities[g]);
    for (auto idx : group_indices[g]) {
      group.push_back(idx);
    }
    tvm_memory_groups.push_back(group);
  }
  for (int64_t id : workspace_version_ids) {
    // Z3 integers cannot overflow. Constrain the final per-allocation byte
    // extent before selecting versions, while leaving GM out of device
    // resource-capacity accounting. Different workspaces are not summed.
    tvm_memory_groups.push_back({std::numeric_limits<int64_t>::max(), id});
  }

  ffi::Array<int64_t> start_times;
  ffi::Array<int> storage_versions;
  int64_t ii = 0;
  if (preserve_issue_order) {
    std::vector<std::vector<int64_t>> completion_memory_groups;
    for (const auto &group : tvm_memory_groups)
      completion_memory_groups.emplace_back(group.begin(), group.end());
    ManualCompletionResult completion = CompleteManualSourceOrder(
        manual_stages, latencies, iis, resource_flags, data_deps, resource_deps,
        owner_exclusion_deps, pipe_order_deps, stage_order_deps, enable_offset,
        storage_sizes, storage_min_versions, completion_memory_groups,
        num_stages);
    ICHECK(completion.plan.has_value())
        << "Explicit Ascend source issue order is not legal: "
        << completion.reason;
    start_times = ffi::Array<int64_t>(completion.plan->starts);
    storage_versions = ffi::Array<int>(completion.plan->versions);
    ii = completion.plan->ii;
    OptimizationDiagnostics diagnostics;
    diagnostics.Record("manual-source-order", OptimizationStatus::kAccepted,
                       completion.reason, {}, ctrl);
  } else {
    // A Stage-only schedule leaves cross-pipe order and possibly versions
    // unspecified, so it keeps the existing partial-completion path.
    static std::optional<ffi::Function> z3_schedule_loop_func =
        ffi::Function::GetGlobal("tl.transform.z3_schedule_loop_python");
    ICHECK(z3_schedule_loop_func.has_value())
        << "Python Z3 loop scheduler is not registered";
    auto return_val =
        z3_schedule_loop_func
            .value()(num_stages, tvm_latencies, tvm_iis, tvm_resource_flags,
                     tvm_data_deps, tvm_resource_deps, tvm_owner_exclusion_deps,
                     tvm_pipe_order_deps, tvm_storage_sizes, tvm_memory_groups,
                     tvm_stage_order_deps, enable_offset, tvm_manual_stages,
                     manual_schedule, stage_horizon, tvm_issue_priority_pairs,
                     tvm_prefetch_priority_pairs, tvm_storage_min_versions)
            .cast<ffi::Tuple<ffi::Array<int64_t>, ffi::Array<int>, int64_t>>();
    start_times = return_val.get<0>();
    storage_versions = return_val.get<1>();
    ii = return_val.get<2>();
  }
  ICHECK_EQ(start_times.size(), n);
  ICHECK_GT(ii, 0);
  ICHECK_EQ(storage_versions.size(), storage_min_versions.size());
  for (size_t i = 0; i < storage_versions.size(); ++i)
    ICHECK_GE(storage_versions[i], storage_min_versions[i])
        << "AutoSchedule returned fewer versions than the shared storage ring";

  // Automatic plans have no fixed absolute origin. Use a common origin
  // for stages and timestamps: subtracting only from timestamps changes the
  // emitted physical phase. A cut at the first issue makes globally shifted
  // solver witnesses identical, provided it preserves every residue/stage
  // constraint. Otherwise remove whole periods and retain the original cut.
  if (!manual_schedule) {
    int64_t first = *std::min_element(start_times.begin(), start_times.end());
    ICHECK_GE(first, 0);
    std::vector<int64_t> relative, residues, stages;
    for (int64_t start : start_times) {
      relative.push_back(start - first);
      residues.push_back((start - first) % ii);
      stages.push_back((start - first) / ii);
    }
    const char *rejection = nullptr;
    using Wide = __int128;
    for (const auto &[u, v] : resource_deps) {
      Wide difference = Wide(residues[v]) - residues[u];
      if (!((difference >= iis[u] && Wide(ii) - difference >= iis[v]) ||
            (-difference >= iis[v] && Wide(ii) + difference >= iis[u])))
        rejection = "resource-wrap";
    }
    for (const auto &[u, v, forward, reverse] : owner_exclusion_deps) {
      Wide difference = Wide(residues[v]) - residues[u];
      if (!((difference >= forward && Wide(ii) - difference >= reverse) ||
            (-difference >= reverse && Wide(ii) + difference >= forward)))
        rejection = "owner-wrap";
    }
    for (const auto &[u, v] : pipe_order_deps)
      if (residues[u] > residues[v])
        rejection = "pipe-order";
    for (const auto &[u, v] : stage_order_deps)
      if (stages[u] > stages[v])
        rejection = "let-stage-order";
    int64_t origin = rejection ? (first / ii) * ii : first;
    if (rejection)
      for (size_t i = 0; i < n; ++i)
        relative[i] = start_times[i] - origin;
    start_times = ffi::Array<int64_t>(relative);
    OptimizationDiagnostics diagnostics;
    diagnostics.Record("automatic-schedule-origin",
                       rejection ? OptimizationStatus::kRejected
                                 : OptimizationStatus::kAccepted,
                       rejection ? rejection : "first-issue-origin", {}, ctrl);
  }

  // Derive a per-task stage from (start_time / ii). Z3 schedules
  // `start = k * ii + r` so `start_time / ii` recovers k. Producers (low k,
  // earliest in z3's absolute time) get stage 0; later tasks get positive
  // stages.
  std::vector<int> raw_k(n, 0);
  int min_k_raw = std::numeric_limits<int>::max();
  int64_t min_start_time = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < n; ++i) {
    int k = (ii > 0) ? static_cast<int>(start_times[i] / ii) : 0;
    raw_k[i] = k;
    min_k_raw = std::min(min_k_raw, k);
    min_start_time = std::min(min_start_time, start_times[i]);
  }
  for (size_t i = 0; i < n; ++i) {
    // When offset is disabled, collapse every task into a single pipeline
    // stage so LowerScheduledTIR emits a flat (non-pipelined) loop body.
    nodes[i]->SetStage(manual_schedule
                           ? static_cast<int>(manual_stages[i])
                           : (enable_offset ? (raw_k[i] - min_k_raw) : 0));
    // Automatic stage and timestamp origins were normalized together.
    // Stage-only/manual schedules retain their existing timestamp policy.
    GetAutoScheduleExtraInfo(nodes[i])->SetStartTime(
        manual_schedule ? start_times[i] - min_start_time : start_times[i]);
  }
  // Intent has now become the native task stages and kernel buffer-version
  // plan. Downstream consumers use those facts rather than selecting again.
  ctrl->control.CopyOnWrite()->annotations.erase(kAutoPipeline);
  auto phys_time = [&](const std::shared_ptr<IRStructure> &c) -> int64_t {
    return GetAutoScheduleExtraInfo(c.get())->GetStartTime() -
           c->GetStage() * ii;
  };
  std::stable_sort(ctrl->children.begin(), ctrl->children.end(),
                   [&](const std::shared_ptr<IRStructure> &a,
                       const std::shared_ptr<IRStructure> &b) {
                     // At equal physical times, larger stages select earlier
                     // logical iterations, for automatic and manual schedules.
                     // Keep source order only when stages also match.
                     if (phys_time(a) == phys_time(b)) {
                       return a->GetStage() > b->GetStage();
                     }
                     return phys_time(a) < phys_time(b);
                   });

  // Reorder & copy Let-defined variables. Recompute the closure after the
  // scheduled children have been reordered so declaration indices match the
  // final sibling list.
  std::vector<IRStructure *> scheduled_nodes;
  scheduled_nodes.reserve(n);
  for (const auto &child : ctrl->children)
    scheduled_nodes.push_back(child.get());
  LetDependencyClosure let_closure =
      CollectLetDependencyClosure_(scheduled_nodes);

  // Resolve Let-var stage conflicts
  // Later stages run earlier in a physical pipeline iteration, so prepend
  // their clones first.
  std::map<int, std::set<size_t>, std::greater<int>> stage_to_decls;
  std::map<int, int64_t> stage_base_start; // stage -> floored start time
  for (size_t u = 0; u < n; ++u) {
    int s = ctrl->children[u]->GetStage();
    for (size_t d : let_closure.dependencies[u]) {
      if (ctrl->children[d]->GetStage() == s)
        continue; // decl already in the consumer's stage: no clone needed
      stage_to_decls[s].insert(d);
      int64_t st =
          GetAutoScheduleExtraInfo(ctrl->children[u].get())->GetStartTime();
      auto bit = stage_base_start.find(s);
      if (bit == stage_base_start.end())
        stage_base_start[s] = st;
      else
        bit->second = std::min(bit->second, st);
    }
  }

  std::vector<std::shared_ptr<IRStructure>> new_clones;
  for (auto &[s, decls] : stage_to_decls) {
    int64_t base_start = (ii > 0) ? stage_base_start[s] / ii * ii : 0;
    std::vector<std::pair<Var, Var>> renames; // old var -> fresh clone var
    std::vector<std::shared_ptr<TaskNode>> stage_clones;
    for (size_t d : decls) {
      ICHECK(let_closure.is_declaration[d]);
      auto *decl_task = static_cast<TaskNode *>(ctrl->children[d].get());
      ffi::Optional<Bind> decl_bind = GetFlatTaskBind(decl_task->stmt);
      ICHECK(decl_bind.defined());
      auto new_var = decl_bind.value()->var.copy_with_suffix("");
      auto cloned_task = std::static_pointer_cast<TaskNode>(decl_task->Clone());
      cloned_task->stmt = ReplaceFlatTaskBind(
          decl_task->stmt,
          Bind(new_var, decl_bind.value()->value, decl_bind.value()->span));
      cloned_task->SetReadVars(decl_task->GetReadVars());
      auto write_vars = decl_task->GetWriteVars();
      for (auto &v : write_vars) {
        if (v.same_as(decl_bind.value()->var))
          v = new_var;
      }
      cloned_task->SetWriteVars(write_vars);
      cloned_task->SetStage(s);
      GetAutoScheduleExtraInfo(cloned_task.get())->SetStartTime(base_start);
      renames.emplace_back(decl_bind.value()->var, new_var);
      stage_clones.push_back(cloned_task);
    }
    // Rewire chained references among the clones
    for (auto &clone : stage_clones)
      for (auto &[old_var, new_var] : renames)
        clone->SubstituteVar(old_var, new_var);
    for (size_t u = 0; u < n; ++u) {
      if (ctrl->children[u]->GetStage() != s)
        continue;
      for (auto &[old_var, new_var] : renames)
        ctrl->children[u]->SubstituteVar(old_var, new_var);
    }
    // Emit the clones defs-before-uses
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> clone_writes;
    for (auto &c : stage_clones)
      for (auto &w : c->GetWriteVars())
        clone_writes.insert(w);
    std::vector<bool> emitted(stage_clones.size(), false);
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> available;
    size_t done = 0;
    while (done < stage_clones.size()) {
      bool progress = false;
      for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
        if (emitted[ci])
          continue;
        bool ready = true;
        for (auto &rv : stage_clones[ci]->GetReadVars()) {
          if (clone_writes.count(rv) && !available.count(rv)) {
            ready = false;
            break;
          }
        }
        if (!ready)
          continue;
        for (auto &w : stage_clones[ci]->GetWriteVars())
          available.insert(w);
        new_clones.push_back(stage_clones[ci]);
        emitted[ci] = true;
        ++done;
        progress = true;
      }
      if (!progress) { // defensive: emit any remainder (no cycle expected)
        for (size_t ci = 0; ci < stage_clones.size(); ++ci) {
          if (!emitted[ci]) {
            new_clones.push_back(stage_clones[ci]);
            emitted[ci] = true;
            ++done;
          }
        }
      }
    }
  }
  ctrl->children.insert(ctrl->children.begin(), new_clones.begin(),
                        new_clones.end());

  // Deterministic completion used synthetic coordinates only to preserve
  // stage and residue order. Replace all scratch timing with a conservative
  // serial cost summary before an enclosing sequence reads task timestamps.
  if (preserve_issue_order) {
    int64_t serial_ii = 0;
    for (const auto &child : ctrl->children) {
      int64_t span = std::max(child->GetLatency(), child->GetII());
      ICHECK_LE(span, std::numeric_limits<int64_t>::max() - serial_ii);
      serial_ii += span;
    }
    serial_ii = std::max<int64_t>(serial_ii, 1);
    int64_t prefix = 0;
    for (const auto &child : ctrl->children) {
      __int128 timestamp = __int128(child->GetStage()) * serial_ii + prefix;
      ICHECK(timestamp <= std::numeric_limits<int64_t>::max() -
                              std::max(child->GetLatency(), child->GetII()))
          << "Explicit manual schedule cost summary overflows int64";
      GetAutoScheduleExtraInfo(child.get())
          ->SetStartTime(static_cast<int64_t>(timestamp));
      prefix += std::max(child->GetLatency(), child->GetII());
    }
    ii = serial_ii;
    ctrl->control.CopyOnWrite()->annotations.erase(kPreserveIssueOrder);
  }

  // Estimate overall latency across loop iterations.
  int64_t overall_latency = 0;
  for (const auto &child : ctrl->children) {
    int64_t start_time = GetAutoScheduleExtraInfo(child.get())->GetStartTime();
    int64_t span = preserve_issue_order
                       ? std::max(child->GetLatency(), child->GetII())
                       : child->GetLatency();
    overall_latency = std::max(overall_latency, start_time + span);
  }
  if (preserve_issue_order) {
    __int128 total =
        __int128(overall_latency) + __int128(ii) * (ctrl->GetTripCount() - 1);
    ICHECK(total >= 0 && total <= std::numeric_limits<int64_t>::max())
        << "Explicit manual schedule loop cost summary overflows int64";
    overall_latency = static_cast<int64_t>(total);
  } else {
    overall_latency += ii * (ctrl->GetTripCount() - 1);
  }

  ctrl->SetII(overall_latency);
  ctrl->SetLatency(overall_latency);
  GetAutoScheduleExtraInfo(ctrl)->SetIIperIter(ii);
  for (const StorageResource &resource : storage_resources) {
    const Var &storage = resource.storage;
    auto annotated = annotated_storage_versions.find(storage);
    if (annotated != annotated_storage_versions.end()) {
      MergeSelectedBufferVersion(storage, annotated->second);
      continue;
    }
    auto selected = storage_version_id.find(storage);
    if (selected != storage_version_id.end()) {
      MergeSelectedBufferVersion(storage, storage_versions[selected->second]);
    }
  }
}

// Pass orchestration.

static void WarnOnUnappliedBufferVersionOverrides(
    const BufferVersionMap &buffer_version_overrides,
    const BufferVersionMap &selected_buffer_versions) {
  for (const auto &[buffer_var, num_versions] : buffer_version_overrides) {
    if (num_versions > 1 && !selected_buffer_versions.count(buffer_var)) {
      LOG(WARNING) << "Ascend AutoSchedule did not enable multi-buffering for "
                      "buffer \""
                   << buffer_var->name_hint
                   << "\" although T.annotate_buffer_versions requested "
                   << num_versions
                   << " versions. The buffer must be eligible in a scheduled "
                      "serial loop and written before it is read.";
    }
  }
}

// Build and schedule one kernel segment, leaving core assignment/resolution
// and final lowering to the downstream passes.
static void ScheduleSingleKernel(const Stmt &kernel_body, Target target,
                                 ScheduledTIR *scheduled_tir) {
  auto &ir_structure = scheduled_tir->tree;
  ScheduledTIRMetadata &metadata = scheduled_tir->metadata;
  ICHECK(!ir_structure.empty()) << "IRStructure is empty (empty body?)";

  std::vector<TaskNode *> costed_tasks;
  CollectAllTaskNodes(ir_structure, costed_tasks);
  for (TaskNode *task : costed_tasks) {
    ICHECK_GT(task->GetII(), 0)
        << "AutoSchedule found a task without latency/II annotations: "
        << task->stmt
        << ". Run tl.transform.EstimateLatency before AutoSchedule.";
  }

  // Schedule IRStructure in place with the Z3 scheduler.
  ScheduleBuilder unit_builder;
  L0StorageGroups groups(CollectL0SFBindings(metadata.kernel_root));
  metadata.buffer_versions = ExpandL0StorageGroupValues(
      metadata.buffer_versions, groups, "version counts");
  metadata.manual_buffer_versions = ExpandL0StorageGroupValues(
      metadata.manual_buffer_versions, groups, "manual version counts");
  unit_builder.SetStorageGroups(groups);
  unit_builder.SetCompilerWorkspaces(
      GetCompilerWorkspaceStorages(metadata.kernel_root));
  unit_builder.SetStorageSlotBytes(StorageFootprintCollector::SlotBytes(
      metadata.kernel_root, metadata.manual_buffer_versions));
  const AscendTargetInfo &target_info = GetAscendTargetInfo(target);
  std::vector<L0PhaseReusePair> phase_reuse;
  if (!tvm::transform::PassContext::Current()
           ->GetConfig<Bool>(kDisableSharedMemoryReuse, Bool(false))
           .value()) {
    phase_reuse = AnalyzeL0ABPhaseReuse(*scheduled_tir, target_info);
    auto c_reuse = AnalyzeL0CPhaseReuse(*scheduled_tir, target_info);
    phase_reuse.insert(phase_reuse.end(), c_reuse.begin(), c_reuse.end());
  }
  if (!phase_reuse.empty()) {
    ApplyL0PhaseReuseConflicts(phase_reuse, &metadata.root_conflict_hints);
    Array<Array<Var>> pairs;
    for (const L0PhaseReusePair &pair : phase_reuse)
      pairs.push_back({pair.first, pair.second});
    metadata.kernel_root.CopyOnWrite()->annotations.Set(kL0PhaseReusePairs,
                                                        std::move(pairs));
  }
  unit_builder.SetL0PhaseReuse(std::move(phase_reuse));
  unit_builder.SetPreferHandoffIssue(target_info.architecture ==
                                     AscendArchitecture::kDav2201);
  unit_builder.SetPreferPrefetchIssue(target_info.architecture ==
                                      AscendArchitecture::kDav2201);
  unit_builder.SetReserveL0Storage(target_info.architecture ==
                                   AscendArchitecture::kDav2201);
  unit_builder.SetReserveUBStorage(target_info.architecture ==
                                   AscendArchitecture::kDav2201);
  int64_t shared_mem_limit = GetSharedMemoryLimit(kernel_body, target_info);
  unit_builder.SetMemoryLimit("shared", shared_mem_limit);
  unit_builder.SetMemoryLimit("shared.l1", target_info.l1_bytes);
  unit_builder.SetMemoryLimit("shared.l0c", target_info.l0c_bytes);
  unit_builder.SetMemoryLimit("shared.l0a", target_info.l0a_bytes);
  unit_builder.SetMemoryLimit("shared.l0b", target_info.l0b_bytes);
  constexpr int64_t kUnlimitedMemory = 1LL << 40;
  for (const String &scope : metadata.unlimit_memory_scopes) {
    unit_builder.SetMemoryLimit(scope, kUnlimitedMemory);
  }
  unit_builder.SetBufferVersionOverrides(metadata.buffer_versions);
  unit_builder.SetManualBufferVersions(metadata.manual_buffer_versions);
  unit_builder.SetRootConflictHints(metadata.root_conflict_hints);
  unit_builder.ScheduleList(ir_structure);

  for (const auto &[sf, data] : groups.Bindings()) {
    auto requested = metadata.buffer_versions.Get(data);
    ICHECK(!requested.has_value() || requested.value() <= 1 ||
           unit_builder.GetSelectedBufferVersions().count(data))
        << "Bound L0 data/SF group " << data->name_hint
        << " has no common multi-buffer owner. Load scales and data under a "
           "common owner, or request one version for the group.";
  }
  metadata.unlimit_memory_scopes = {};
  metadata.buffer_versions = unit_builder.GetSelectedBufferVersions();
}
} // namespace

// Build and schedule each kernel, serializing the tree into ordinary TIR
// markers for the downstream scheduled-TIR passes.
tvm::transform::Pass AutoSchedule() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    // Get target from PrimFunc attribute for GemmInst determination
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    Target target;
    if (target_opt.defined()) {
      target = target_opt.value();
    }
    return RewriteTilelangKernels(
        std::move(func), "AutoSchedule",
        [&](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.root, context.outer_ctx, MakeAutoScheduleExtraInfo);
          SBlock root =
              TryCoalesceL0CStorage(scheduled_tir, GetAscendTargetInfo(target));
          if (!root.same_as(context.root)) {
            // A capacity repair changes physical storage identity. Decode the
            // changed shell once so every task/owner/dependency consumes those
            // facts, rather than retaining a parallel alias side table.
            scheduled_tir = DecodeScheduledTIR(root, context.outer_ctx,
                                               MakeAutoScheduleExtraInfo);
          }
          BufferVersionMap buffer_version_overrides =
              scheduled_tir.metadata.buffer_versions;
          ScheduleSingleKernel(root->body, target, &scheduled_tir);
          WarnOnUnappliedBufferVersionOverrides(
              buffer_version_overrides, scheduled_tir.metadata.buffer_versions);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        });
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSchedule", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AutoSchedule", AutoSchedule);
}

} // namespace tl
} // namespace tvm
