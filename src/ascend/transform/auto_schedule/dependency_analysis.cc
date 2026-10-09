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
 * \file dependency_analysis.cc
 * \brief Shared schedule and synchronization dependency analysis.
 */

#include "./dependency_analysis.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"
#include "transform/common/attr.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// A bare buffer hint matches every region using its storage. A region hint
// matches only the same logical buffer and exact footprint.
static bool HintOperandMatches(const Any &hint, const BufferRegion &query) {
  if (auto var = hint.as<Var>())
    return var.value().same_as(query->buffer->data);
  BufferRegion hint_region = Downcast<BufferRegion>(hint);
  return hint_region->buffer.same_as(query->buffer) &&
         RegionsEqual(hint_region->region, query->region);
}

static ConflictHintList
ActiveConflictHints(ControlNode *loop, const ConflictHintList &root_hints) {
  if (loop == nullptr)
    return root_hints;
  auto ann = loop->control->annotations.Get("conflict_hint");
  return ann.has_value() ? ann.value().cast<ConflictHintList>()
                         : ConflictHintList{};
}

static bool IsConflictHint(const Array<Any> &hint) {
  ICHECK_EQ(hint.size(), 4u)
      << "conflict_hint annotation expects 4-element entries";
  const auto *is_conflict = hint[3].as<IntImmNode>();
  ICHECK(is_conflict != nullptr && is_conflict->dtype.is_bool())
      << "conflict_hint annotation expects a constant boolean polarity";
  return is_conflict->value;
}

namespace {

// The symbolic environment of two task executions. Region bounds, mutable-read
// snapshots and prover state are deliberately absent: each access comparison
// instantiates those independently from these immutable iteration facts.
struct PreparedRegionContext {
  PreparedRegionContext(const ConstrSet &a_ctx, const ConstrSet &b_ctx,
                        ControlNode *loop, int offset,
                        bool independent_owner_epochs) {
    ConstrSet a_local_ctx = a_ctx;
    ConstrSet b_local_ctx = b_ctx;
    ConstrSet loop_ctx;
    // Scheduled tasks can define condition snapshots outside the constraint
    // visitor's Bind stack. Rename all variables written in the compared
    // iteration scope, including these otherwise-free predicates. Ancestor
    // definitions stay shared: opposite branches selected by one outer epoch
    // are still exclusive when comparing iterations of an inner loop.
    auto seed_iteration_vars = [&](const ControlNode *scope) {
      for (const Var &var : scope->GetWriteVars()) {
        if (!a_sub.count(var))
          a_sub.Set(var, var.copy_with_suffix("_p"));
        if (!b_sub.count(var))
          b_sub.Set(var, var.copy_with_suffix("_c"));
      }
    };
    if (offset != 0) {
      // A carried edge belongs to the loop being analyzed, not to any differing
      // coordinate in its enclosing nest. Keep ancestors fixed and instantiate
      // this loop and its descendants independently. Parent analysis retains
      // hazards that cross the outer boundary of this sequence.
      const ControlNode *rename_scope = loop;
      if (independent_owner_epochs) {
        // Distinct storage owners have independent epochs even when their
        // enclosing coordinates happen to coincide.
        rename_scope = loop->GetAncestorControls().front();
      }
      seed_iteration_vars(rename_scope);
      Var pivot = rename_scope->control->loop_var;
      a_local_ctx = a_local_ctx.RenameFrom("_p", a_sub, pivot);
      b_local_ctx = b_local_ctx.RenameFrom("_c", b_sub, pivot);
      Var iteration = loop->control->loop_var;
      PrimExpr p = Substitute(iteration, a_sub);
      PrimExpr c = Substitute(iteration, b_sub);
      if (offset > 0) {
        loop_ctx.AddConstr(c == p + IntImm(iteration.dtype(), offset));
      } else if (!independent_owner_epochs) {
        loop_ctx.AddConstr(p != c);
      }
    }
    constraints = loop_ctx.Merge(a_local_ctx).Merge(b_local_ctx);
  }

  Map<Var, PrimExpr> a_sub, b_sub;
  ConstrSet constraints;
};

// One ordered node-pair analysis, shared by its RAW/WAW/WAR region queries.
// The tasks and loop are unchanged for this lifetime. Nothing survives into
// another child list, an IR rewrite, or a synchronization fallback attempt.
class RegionConflictAnalysis {
public:
  RegionConflictAnalysis(ControlNode *loop,
                         const ConflictHintList &root_conflicts)
      : loop_(loop), root_conflicts_(root_conflicts) {}

  bool MayConflict(const TaskNode *a_task, const BufferRegion &a_region,
                   const TaskNode *b_task, const BufferRegion &b_region,
                   int offset, size_t num_storage_owners) {
    const bool cross = offset != 0;
    if (cross && loop_ == nullptr)
      return false;

    bool declared_no_conflict = false;
    for (const Any &entry : ActiveConflictHints(loop_, root_conflicts_)) {
      Array<Any> hint = Downcast<Array<Any>>(entry);
      bool is_conflict = IsConflictHint(hint);
      int64_t cross_code = Downcast<IntImm>(hint[2])->value;
      if (cross_code != -1 && (cross_code == 1) != cross)
        continue;
      bool matches = (HintOperandMatches(hint[0], a_region) &&
                      HintOperandMatches(hint[1], b_region)) ||
                     (HintOperandMatches(hint[0], b_region) &&
                      HintOperandMatches(hint[1], a_region));
      if (!matches)
        continue;
      if (is_conflict)
        return true;
      declared_no_conflict = true;
    }
    // A forced conflict wins if contradictory declarations match.
    if (declared_no_conflict)
      return false;

    if (!a_region->buffer->data.same_as(b_region->buffer->data))
      return false;

    // Same storage but different logical buffer (T.view/T.reshape): regions may
    // differ in rank/dtype/strides, so keep the dependency conservatively.
    if (!a_region->buffer.same_as(b_region->buffer))
      return true;

    const Array<Range> &a_rng = a_region->region;
    const Array<Range> &b_rng = b_region->region;
    if (a_rng.size() != b_rng.size())
      return true; // shape mismatch — be safe
    const size_t ndim = a_rng.size();

    // Independent owner epochs permit equal loop coordinates, even when the
    // loop is already outermost. Keep that relation distinct from an ordinary
    // carried edge (p != c); a rename scope alone does not identify a context.
    bool independent_owner_epochs = offset < 0 && num_storage_owners > 1;
    const PreparedRegionContext &context =
        contexts_
            .try_emplace(std::make_tuple(a_task, b_task, offset,
                                         independent_owner_epochs),
                         a_task->outer_ctx, b_task->outer_ctx, loop_, offset,
                         independent_owner_epochs)
            .first->second;
    arith::Analyzer ana;
    ana.z3_prover.SetRLimit(50000);
    context.constraints.Populate(ana);

    // Freshen mutable reads (BufferLoads etc.) inline in the region bounds, per
    // side: a read written verbatim in both bounds is not provably equal across
    // the two access points (a store may sit between them), so it must become
    // an independent unknown on each side.
    FreshenMutableReads freshen_a(FreshenMutableReads::Mode::kSnapshot);
    FreshenMutableReads freshen_b(FreshenMutableReads::Mode::kSnapshot);
    PrimExpr any_dim_disjoint = Bool(false);
    for (size_t i = 0; i < ndim; ++i) {
      PrimExpr a_min = freshen_a(Substitute(a_rng[i]->min, context.a_sub));
      PrimExpr a_max = freshen_a(
          Substitute(a_rng[i]->min + a_rng[i]->extent - 1, context.a_sub));
      PrimExpr b_min = freshen_b(Substitute(b_rng[i]->min, context.b_sub));
      PrimExpr b_max = freshen_b(
          Substitute(b_rng[i]->min + b_rng[i]->extent - 1, context.b_sub));
      any_dim_disjoint = any_dim_disjoint || (a_max < b_min) || (b_max < a_min);
    }
    any_dim_disjoint = ana.Simplify(any_dim_disjoint);
    // Raw Z3 as a fallback catches the infeasible-context case that Analyzer's
    // constant-fold short-circuit misses.
    return !(ana.CanProve(any_dim_disjoint) ||
             ana.z3_prover.CanProve(any_dim_disjoint));
  }

private:
  ControlNode *loop_;
  const ConflictHintList &root_conflicts_;
  std::map<std::tuple<const TaskNode *, const TaskNode *, int, bool>,
           PreparedRegionContext>
      contexts_;
};

} // namespace

std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes, ControlNode *loop,
                    const BufferVersionMap &manual_buffer_versions,
                    const MultiBufferOwnerMap &multi_buffer_owners,
                    DependencyCache *dependency_cache,
                    const ConflictHintList &root_conflicts) {
  DependencyCache local_cache;
  DependencyCache &cache = dependency_cache ? *dependency_cache : local_cache;
  if (loop) {
    auto cache_it = cache.find(loop);
    if (cache_it != cache.end())
      return cache_it->second;
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const IRStructure *lhs, const IRStructure *rhs) {
                       return lhs->GetStage() < rhs->GetStage();
                     });
  }

  const size_t n = nodes.size();
  std::vector<DepInfo> deps;

  struct RegionAccess {
    TaskNode *task;
    BufferRegion region;
    DependencyAccessMode mode;
  };
  using StorageAccesses =
      std::unordered_map<Var, DependencyStorageAccess, ObjectPtrHash,
                         ObjectPtrEqual>;
  struct NodeAccesses {
    std::vector<RegionAccess> reads;
    std::vector<RegionAccess> writes;
    StorageAccesses storage_accesses;
    std::unordered_map<TaskNode *, StorageAccesses> task_accesses;
  };
  auto record_access = [](StorageAccesses *accesses, const BufferRegion &region,
                          DependencyAccessMode mode) {
    DependencyStorageAccess &access = (*accesses)[region->buffer->data];
    if (mode == DependencyAccessMode::kRead)
      access.reads = true;
    else
      access.writes = true;
  };
  auto storage_access = [](const StorageAccesses &accesses,
                           const ffi::Optional<Var> &storage) {
    if (storage.has_value()) {
      auto found = accesses.find(storage.value());
      if (found != accesses.end())
        return found->second;
    }
    return DependencyStorageAccess{};
  };
  std::vector<NodeAccesses> accesses(n);
  for (size_t i = 0; i < n; ++i) {
    // Whole-node roles include a control's own header task. The leaf set below
    // deliberately excludes that task, preserving the dependency edge set.
    for (const BufferRegion &region : nodes[i]->GetReadRegions())
      record_access(&accesses[i].storage_accesses, region,
                    DependencyAccessMode::kRead);
    for (const BufferRegion &region : nodes[i]->GetWriteRegions())
      record_access(&accesses[i].storage_accesses, region,
                    DependencyAccessMode::kWrite);
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(nodes[i], tasks);
    for (TaskNode *task : tasks) {
      StorageAccesses &roles = accesses[i].task_accesses[task];
      for (const BufferRegion &region : task->GetReadRegions()) {
        record_access(&roles, region, DependencyAccessMode::kRead);
        accesses[i].reads.push_back(
            {task, region, DependencyAccessMode::kRead});
      }
      for (const BufferRegion &region : task->GetWriteRegions()) {
        record_access(&roles, region, DependencyAccessMode::kWrite);
        accesses[i].writes.push_back(
            {task, region, DependencyAccessMode::kWrite});
      }
    }
  }

  auto get_manual_versions = [&](const ffi::Optional<Var> &storage) {
    if (!storage.has_value())
      return 0;
    auto it = manual_buffer_versions.find(storage.value());
    return it == manual_buffer_versions.end() ? 0 : (*it).second;
  };
  auto get_num_storage_owners =
      [&](const ffi::Optional<Var> &storage) -> size_t {
    if (loop == nullptr || !storage.has_value())
      return 0;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() ||
        std::find(owners->second.begin(), owners->second.end(), loop) ==
            owners->second.end()) {
      return 0;
    }
    return owners->second.size();
  };
  auto find_owner = [](const std::vector<ControlNode *> &owners,
                       const TaskNode *task) -> ControlNode * {
    ControlNode *result = nullptr;
    for (ControlNode *owner : owners) {
      if (!task->IsWithin(owner))
        continue;
      // Nested owners deliberately remain data dependencies here;
      // PrepareMultiBuffer diagnoses their overlapping ownership.
      if (result != nullptr)
        return nullptr;
      result = owner;
    }
    return result;
  };
  auto is_owner_exclusion = [&](const ffi::Optional<Var> &storage,
                                const TaskNode *lhs, const TaskNode *rhs) {
    if (!storage.has_value())
      return false;
    auto owners = multi_buffer_owners.find(storage.value());
    if (owners == multi_buffer_owners.end() || owners->second.size() <= 1)
      return false;
    ControlNode *lhs_owner = find_owner(owners->second, lhs);
    ControlNode *rhs_owner = find_owner(owners->second, rhs);
    return lhs_owner != nullptr && rhs_owner != nullptr &&
           lhs_owner != rhs_owner;
  };

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (loop == nullptr && i >= j)
        continue;
      struct PendingDependency {
        ffi::Optional<Var> storage;
        int distance;
        DependencyKind kind;
        bool cross_storage_conflict;
        // Preserve the original endpoint-pair ordering and deduplication while
        // retaining every access conflict on that one record.
        std::map<std::pair<TaskNode *, TaskNode *>, DependencyTaskPair>
            task_pairs;
      };
      std::vector<PendingDependency> pending;
      RegionConflictAnalysis conflicts(loop, root_conflicts);

      auto add_pair = [&](const ffi::Optional<Var> &storage, int distance,
                          const RegionAccess &producer,
                          const RegionAccess &consumer,
                          DependencyKind kind = DependencyKind::kData,
                          bool cross_storage_conflict = false) {
        auto existing = std::find_if(
            pending.begin(), pending.end(), [&](const PendingDependency &dep) {
              return dep.storage.same_as(storage) && dep.distance == distance &&
                     dep.kind == kind &&
                     dep.cross_storage_conflict == cross_storage_conflict;
            });
        if (existing == pending.end()) {
          pending.push_back(
              {storage, distance, kind, cross_storage_conflict, {}});
          existing = std::prev(pending.end());
        }
        auto pair =
            existing->task_pairs
                .try_emplace(
                    std::make_pair(producer.task, consumer.task),
                    DependencyTaskPair{
                        producer.task, consumer.task, {}, 0,
                        storage_access(
                            accesses[i].task_accesses.at(producer.task), storage),
                        storage_access(
                            accesses[j].task_accesses.at(consumer.task), storage)})
                .first;
        pair->second.memory_accesses.push_back(
            {producer.region, consumer.region, producer.mode, consumer.mode});
      };

      auto consider = [&](const std::vector<RegionAccess> &lhs_accesses,
                          const std::vector<RegionAccess> &rhs_accesses) {
        for (const RegionAccess &lhs : lhs_accesses) {
          for (const RegionAccess &rhs : rhs_accesses) {
            const Var &lhs_storage = lhs.region->buffer->data;
            const Var &rhs_storage = rhs.region->buffer->data;
            ffi::Optional<Var> storage;
            if (lhs_storage.same_as(rhs_storage))
              storage = lhs_storage;
            bool owner_exclusion =
                is_owner_exclusion(storage, lhs.task, rhs.task);
            // The i<j visit sees every cross-owner write/read, write/write,
            // and read/write conflict once. Such conflicts are undirected:
            // the scheduler chooses one physical owner order.
            if (owner_exclusion && i >= j)
              continue;
            int manual_versions = get_manual_versions(storage);
            if (manual_versions && !owner_exclusion) {
              for (int d = (i < j ? 0 : 1); d <= manual_versions; ++d) {
                if (conflicts.MayConflict(lhs.task, lhs.region, rhs.task,
                                          rhs.region, d,
                                          /*num_storage_owners=*/0)) {
                  add_pair(storage, d, lhs, rhs);
                  break;
                }
              }
              continue;
            }
            for (int distance : {0, -1}) {
              if (distance == 0 && i >= j)
                continue;
              if (distance < 0 && loop == nullptr)
                continue;
              if (conflicts.MayConflict(lhs.task, lhs.region, rhs.task,
                                        rhs.region, distance,
                                        get_num_storage_owners(storage))) {
                add_pair(storage, owner_exclusion ? 0 : distance, lhs, rhs,
                         owner_exclusion ? DependencyKind::kOwnerExclusion
                                         : DependencyKind::kData,
                         // Different storages conflict only through a positive
                         // hint; ordinary region overlap cannot create this.
                         !lhs_storage.same_as(rhs_storage));
                break;
              }
            }
          }
        }
      };
      consider(accesses[i].writes, accesses[j].reads);
      consider(accesses[i].writes, accesses[j].writes);
      consider(accesses[i].reads, accesses[j].writes);

      // A control-node self edge crosses iterations of this parent loop.
      // The same leaf pair in a descendant dependency compares a different
      // coordinate/domain and does not discharge this boundary hazard.
      for (PendingDependency &dep : pending) {
        if (dep.task_pairs.empty())
          continue;
        std::vector<DependencyTaskPair> pairs;
        pairs.reserve(dep.task_pairs.size());
        for (auto &entry : dep.task_pairs)
          pairs.push_back(std::move(entry.second));
        DependencyStorageAccess producer_access =
            storage_access(accesses[i].storage_accesses, dep.storage);
        DependencyStorageAccess consumer_access =
            storage_access(accesses[j].storage_accesses, dep.storage);
        deps.push_back({nodes[i], nodes[j], std::move(dep.storage),
                        std::move(pairs), dep.distance, dep.kind,
                        dep.cross_storage_conflict, producer_access,
                        consumer_access});
      }
    }
  }
  // Invariant root SPR state is not a buffer hazard and cannot be versioned.
  // Both scheduling and synchronization consume these storage-free leaf RAW
  // edges. Only explicitly selected FIX readers opt into this contract; the
  // established per-copy configuration path retains its ordinary SPR ordering.
  for (auto [root_config, name] :
       {std::pair{SpecialRegister::kLoad3DConfig, "Load3D"},
        std::pair{SpecialRegister::kFixpipeNz2Nd, "Fixpipe"}}) {
    uint16_t mask = static_cast<uint16_t>(root_config);
    std::vector<std::vector<TaskNode *>> state_reads(n);
    std::vector<std::vector<TaskNode *>> tasks_by_node(n);
    bool has_explicit_read = false;
    for (size_t i = 0; i < n; ++i) {
      CollectAllTaskNodes(nodes[i], tasks_by_node[i]);
      for (TaskNode *task : tasks_by_node[i]) {
        if (task->GetExplicitSpecialReadMask() & mask) {
          state_reads[i].push_back(task);
          has_explicit_read = true;
        }
      }
    }
    if (root_config == SpecialRegister::kFixpipeNz2Nd && !has_explicit_read)
      continue;
    TaskNode *initializer = nullptr;
    size_t initializer_node = n;
    for (size_t i = 0; i < n; ++i) {
      for (TaskNode *task : tasks_by_node[i]) {
        if (!(task->GetSpecialWriteMask() & mask))
          continue;
        ICHECK(loop == nullptr && nodes[i] == task &&
               !task->HasConditions() && initializer == nullptr)
            << name << " configuration requires exactly one unconditional "
                       "kernel-root S task; repeated configuration is unsupported";
        ICHECK_EQ(task->GetPipeMask(), PipeMask(ResourcePipe::kScalar));
        ICHECK_EQ(task->GetCoreMask(), kCoreCube)
            << name << " configuration must be owned by the AIC";
        initializer = task;
        initializer_node = i;
      }
    }
    if (loop != nullptr)
      continue;
    for (size_t i = 0; i < n; ++i) {
      if (state_reads[i].empty())
        continue;
      ICHECK(initializer != nullptr && initializer_node < i)
          << "Selected " << name
          << " requires a preceding kernel-root configuration";
      std::vector<DependencyTaskPair> pairs;
      for (TaskNode *reader : state_reads[i])
        pairs.push_back({initializer, reader, {}, mask});
      deps.push_back({nodes[initializer_node], nodes[i], std::nullopt,
                      std::move(pairs), 0, DependencyKind::kData});
    }
  }
  if (loop)
    cache.emplace(loop, deps);
  return deps;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
