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
 * \file insert_sync.cc
 * \brief Synchronization analysis, flag allocation, and insertion.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/arith/pattern.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "./auto_schedule/buffer_alias_analysis.h"
#include "./auto_schedule/dependency_analysis.h"
#include "./auto_schedule/gm_read_release.h"
#include "./auto_schedule/gm_write_acquire.h"
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/l0ab_phase_reuse.h"
#include "./auto_schedule/l0_operand_slot_pool.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/owner_result.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "./auto_schedule/unit_flag_stream.h"
#include "ascend/op/builtin.h"
#include "ascend/op/cube_load.h"
#include "ascend/target_utils.h"
#include "ascend/transform/attr.h"
#include "buffer_alias.h"
#include "buffer_version.h"
#include "runtime/thread_storage_scope.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using ffi::GetRef;

namespace {

constexpr int kIntraCoreFlagLimit = 8;

// One logical version is one physical collective counter on dav-2201. On
// dav-3510, mode 4 addresses an independent Cube-side counter for each AIV.
// Allocation, steady-state issue, and finite fill/drain use this same contract.
struct CrossCoreSyncContract {
  bool collective;
  int num_aiv_subcores;

  int Mode() const { return collective ? 2 : 4; }
  int CubeChannels() const { return collective ? 1 : num_aiv_subcores; }
  int CubeFlagBase(int base_id, int channel) const {
    ICHECK_LT(channel, CubeChannels());
    return base_id + (collective ? 0 : 16 * channel);
  }
};

std::vector<Var> CollectAivLaneVariables(const Stmt &body) {
  std::vector<Var> result;
  PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    const auto *attribute = node.as<AttrStmtNode>();
    if (!attribute || attribute->attr_key != tirx::attr::thread_extent)
      return;
    const auto *binding = attribute->node.as<IterVarNode>();
    if (binding && binding->thread_tag == "cthread")
      result.push_back(binding->var);
  });
  return result;
}

// Follow existing lexical Bind/range facts; proving a guard under itself
// would conceal lane divergence. Mutable reads have no uniformity contract.
// Data predicates inside task bodies are deliberately excluded so an empty
// owner still reaches its completion task.
bool HasNonuniformCollectiveControl(const PrimExpr &expression,
                                    const ConstrSet &context,
                                    const std::vector<Var> &lane_variables) {
  std::unordered_set<const VarNode *> varying;
  for (const Var &lane : lane_variables)
    varying.insert(lane.get());
  auto is_varying = [&](const PrimExpr &value) {
    bool result = false;
    PostOrderVisit(value, [&](const ffi::ObjectRef &node) {
      if (const auto *var = node.as<VarNode>())
        result |= varying.count(var) != 0;
      if (node.as<BufferLoadNode>() || node.as<ProducerLoadNode>())
        result = true;
      if (const auto *call = node.as<CallNode>())
        result |= SideEffect(GetRef<PrimExpr>(call)) > CallEffectKind::kPure;
    });
    return result;
  };
  for (const Constr &constraint : context.constrs_) {
    if (constraint.kind == Constr::kBindValue && is_varying(constraint.value)) {
      varying.insert(constraint.var.get());
    } else if (constraint.kind == Constr::kBindRange &&
               (is_varying(constraint.range->min) ||
                is_varying(constraint.range->extent))) {
      varying.insert(constraint.var.get());
    }
  }
  return is_varying(expression);
}

// ============================================================================
// Synchronization site/edge model, collection, and lookup
// ============================================================================

struct SyncInsertionSite {
  IRStructure *path_begin;
  IRStructure *path_end;
  bool at_beginning;
  CoreMask core;
  std::string pipe;
  int domain_id;
  PrimExpr guard{Bool(true)};

  SyncInsertionSite(IRStructure *path_begin, TaskNode *task, bool at_beginning,
                    CoreMask core, const EpochDomain &domain,
                    ControlNode *participant_loop = nullptr,
                    bool active_tasks_only = false)
      : path_begin(path_begin), path_end(nullptr), at_beginning(at_beginning),
        core(core), pipe(GetResourcePipeName(task->GetPipeMask())),
        domain_id(domain.id) {
    ICHECK(path_begin != nullptr);
    ICHECK(task != nullptr);
    ICHECK(IsConcreteCore(core));
    ICHECK_GT(domain_id, 0);
    if (domain.IsCounter()) {
      ICHECK(domain.owner != nullptr && path_begin != domain.owner &&
             path_begin->IsWithin(domain.owner))
          << "Counter-domain site must stay strictly inside its owner";
    } else {
      ICHECK_EQ(path_begin->GetParent(), domain.owner)
          << "Lexical-domain site must be a direct child of its owner";
    }
    PrimExpr active_guard = domain.active_guard;
    bool active_guard_is_enclosed = is_one(active_guard);
    bool path_begin_guard_matches_active = false;
    auto append = [&](PrimExpr term) {
      guard = is_one(guard) ? std::move(term) : guard && term;
    };
    std::vector<IRStructure *> path = task->PathFrom(path_begin);
    for (size_t path_index = 0; path_index < path.size(); ++path_index) {
      path_end = path[path_index];
      bool has_conditions = path_end->HasConditions();
      if (has_conditions && active_tasks_only) {
        // The owner plan proves the exact finite source/consumer activation.
        // Unlike a seeded collective clock, it emits no inactive-epoch token.
        append(path_end->GetConditionGuard());
      } else if (has_conditions) {
        // Only a guard scope's direct child may absorb its active guard.
        if (path_end != path_begin || is_one(active_guard) ||
            path_begin->GetParent() != domain.owner)
          break;
        PrimExpr condition_guard = path_end->GetConditionGuard();
        if (!active_guard.same_as(condition_guard) &&
            !ffi::StructuralEqual()(active_guard, condition_guard)) {
          break;
        }
        path_begin_guard_matches_active = true;
      }
      if (path_index + 1 == path.size())
        break;
      ICHECK(path_end->IsControl());
      auto *control = static_cast<ControlNode *>(path_end);
      if (control->BodyHasLoopBreak())
        break;
      PrimExpr nonempty = control->control->extent >
                          make_zero(control->control->extent.dtype());
      bool can_enter = control->task->outer_ctx.CanProve(nonempty);
      if (!can_enter)
        break;
      if (path_end == path_begin && has_conditions)
        active_guard_is_enclosed = path_begin_guard_matches_active;
      const ForNode *for_node = control->control.get();
      // An owner-result cut is reached once for each complete owner plane.
      // Descendant physical-tile loops still contribute their final-iteration
      // guard. Ordinary collective sites never pass participant_loop.
      if (control == participant_loop)
        continue;
      PrimExpr loop_var = for_node->loop_var;
      PrimExpr min = for_node->min;
      PrimExpr extent = for_node->extent;
      PrimExpr step = for_node->step.has_value() ? for_node->step.value()
                                                 : IntImm(DataType::Int(32), 1);
      if (at_beginning) {
        append(loop_var == min);
      } else {
        append(((loop_var - min) / step + 1) * step >= extent);
      }
    }
    ICHECK(path_end != nullptr) << "Failed to find a sync insertion node";
    // A crossed path_begin condition physically encloses the insertion. If it
    // was not crossed, keep the domain activation predicate on the emitted
    // synchronization itself.
    if (!active_guard_is_enclosed)
      append(active_guard);
  }

  int EpochDomainId() const { return domain_id; }

  bool operator<(const SyncInsertionSite &other) const {
    int parent_order = CompareIRStructure(path_begin->GetParent(),
                                          other.path_begin->GetParent());
    if (parent_order != 0)
      return parent_order < 0;

    int path_end_order = CompareIRStructure(path_end, other.path_end);
    if (path_end_order == -2)
      return at_beginning;
    if (path_end_order == 2)
      return !other.at_beginning;
    if (path_end_order != 0)
      return path_end_order < 0;
    if (at_beginning != other.at_beginning)
      return at_beginning;
    if (core != other.core)
      return core < other.core;
    if (pipe != other.pipe)
      return pipe < other.pipe;
    // Keep domains as the final key. For collected sites, equal parent,
    // path_end, direction, core, and pipe also imply equal path_begin, so all
    // variants of one physical position remain contiguous.
    return EpochDomainId() < other.EpochDomainId();
  }

  bool operator==(const SyncInsertionSite &other) const {
    return path_begin == other.path_begin && path_end == other.path_end &&
           at_beginning == other.at_beginning && core == other.core &&
           pipe == other.pipe && EpochDomainId() == other.EpochDomainId();
  }

  bool HasSamePhysicalPosition(const SyncInsertionSite &other) const {
    return path_begin == other.path_begin && path_end == other.path_end &&
           at_beginning == other.at_beginning && core == other.core &&
           pipe == other.pipe;
  }

  const PrimExpr &GetGuard() const { return guard; }
};

using SyncInsertionSiteList = std::vector<SyncInsertionSite>;

// This private certificate releases only the typed GM storage. Its old wide
// source END must never enter generic completion or flag-lifetime closure.
struct GmReadReleaseCut {
  GmReadReleasePlan plan;
  SyncInsertionSite active;
  SyncInsertionSite inactive;
};

// This private acquire protects the GM write, not entry to its enclosing
// Cube phase. Independent input preparation can precede the physical wait.
struct GmWriteAcquireCut {
  GmWriteAcquirePlan plan;
  SyncInsertionSite active;
  SyncInsertionSite inactive;
};

// Inputs needed to look up a collected insertion site without reconstructing
// it. Domain ids are pass-local and reconstructed from canonical guards.
using SyncInsertionSiteKey =
    std::tuple<IRStructure *, TaskNode *, bool, CoreMask, int>;

struct DepEdge {
  size_t src;
  size_t dst;
  int distance;
  bool strict;

  DepEdge(size_t src, size_t dst, int distance, bool strict)
      : src(src), dst(dst), distance(distance), strict(strict) {}

  bool operator<(const DepEdge &other) const {
    // Process tighter edges first: smaller distance, later producer, then
    // earlier consumer. Site IDs preserve SyncInsertionSite's total order.
    if (distance != other.distance)
      return distance < other.distance;
    if (src != other.src)
      return src > other.src;
    if (dst != other.dst)
      return dst < other.dst;
    return strict < other.strict;
  }
};

// Fields: counter group, producer core/pipe, consumer core/pipe, versions,
// distance.
// Counter event IDs always use the owner clock, but edge distance remains in
// the iteration units of the loop where the dependency was analyzed.
using CounterSyncSignature =
    std::tuple<int, CoreMask, std::string, CoreMask, std::string, int, int>;

// The ordinal is assigned once within each owner/signature sequence. Equal
// keys in different owners identify one shared physical counter channel.
using CounterChannelKey = std::pair<CounterSyncSignature, int>;

// A local hard event is a binary flag, unlike a cross-core counter. A carried
// edge src(i) -> dst(i+d) uses producer-epoch IDs and only d initial tokens.
// A reusable ring has width d+r, where dst(i) completes before src(i+r).
// A finite non-reusing clock instead reserves one ID per real/initial epoch.
struct LocalBinaryFlagPlan {
  int distance;
  PrimExpr trip_count;
};

struct LocalBinaryEpochSpan {
  PrimExpr first;
  PrimExpr trip_count;
};

enum class UnitFlagBoundaryRole {
  kNone,
  kEntry,
  kSetup,
  kFixSetup,
  kDrain,
  kRestore,
  kExit,
  kStreamEntry,
  kStreamDrain
};

// A one-iteration child has no intra-child reuse. Its ready flag remains
// real, but its repeated-activation lifetime is checked in the direct parent
// after that parent's ordinary carried completion edges have been established.
struct SingleIterationBinaryPlan {
  ControlNode *parent;
  PrimExpr activation{Bool(true)};
  bool validated{false};
};

struct SyncPoint {
  DepEdge edge;
  DependencyKind dependency_kind;
  int num_versions;
  PrimExpr flag_iteration;
  ControlNode *flag_loop;
  int counter_group_id;
  size_t allocation_index;
  std::optional<CounterChannelKey> counter_channel;
  std::optional<LocalBinaryFlagPlan> local_binary_plan;
  std::optional<SingleIterationBinaryPlan> single_iteration_binary_plan;
  UnitFlagBoundaryRole unit_boundary{UnitFlagBoundaryRole::kNone};
  // Emitted by the selected physical pool, not this logical-storage ring.
  // Only an exact child-generation edge can enter the ordinary closure.
  bool operand_pool_release{false};
  std::optional<size_t> gm_read_release;
  std::optional<size_t> gm_write_acquire;

  bool IsPrivateGmHandoff() const {
    return gm_read_release.has_value() || gm_write_acquire.has_value();
  }

  SyncPoint(DepEdge edge, DependencyKind dependency_kind, int num_versions,
            ControlNode *flag_loop)
      : edge(std::move(edge)), dependency_kind(dependency_kind),
        num_versions(num_versions),
        flag_iteration(CalculateIterationCount(flag_loop)),
        flag_loop(flag_loop), counter_group_id(0) {}

  SyncPoint(DepEdge edge, DependencyKind dependency_kind, int num_versions,
            PrimExpr flag_iteration, ControlNode *counter_owner,
            int counter_group_id)
      : edge(std::move(edge)), dependency_kind(dependency_kind),
        num_versions(num_versions), flag_iteration(std::move(flag_iteration)),
        flag_loop(counter_owner), counter_group_id(counter_group_id) {
    ICHECK(this->flag_iteration.defined());
    ICHECK(counter_owner != nullptr);
    ICHECK_GT(counter_group_id, 0);
  }

  const CounterChannelKey &GetCounterChannel() const {
    ICHECK(counter_channel.has_value())
        << "Counter synchronization point has no channel identity";
    return counter_channel.value();
  }

  bool operator<(const SyncPoint &other) const {
    if (edge < other.edge)
      return true;
    if (other.edge < edge)
      return false;
    int loop_order = CompareIRStructure(flag_loop, other.flag_loop);
    if (loop_order != 0)
      return loop_order < 0;
    if (num_versions != other.num_versions)
      return num_versions < other.num_versions;
    return false;
  }
};

struct VisibleSyncPoint {
  size_t sync_point_id;
  std::vector<DepEdge> projected_edges;
};

class SyncSiteRegistry {
public:
  explicit SyncSiteRegistry(const EpochDomainRegistry &domains)
      : domains_(domains) {}

  void Collect(const std::vector<IRStructure *> &nodes) {
    sites_.clear();
    site_ids_.clear();
    parent_site_ids_.clear();

    std::map<SyncInsertionSite, std::map<int, SyncInsertionSite>> parent_sites;
    std::map<SyncInsertionSiteKey, SyncInsertionSite> keyed_sites;
    std::vector<TaskNode *> tasks;
    for (IRStructure *node : nodes)
      CollectAllTaskNodes(node, tasks);

    for (TaskNode *task : tasks) {
      if (task->GetCoreMask() == kCoreUnassigned)
        continue;
      for (IRStructure *path_begin : task->PathFrom()) {
        SyncInsertionSiteList inner_sites;
        IRStructure *parent = path_begin->GetParent();
        ICHECK(parent == nullptr || parent->IsControl());
        auto *scope =
            parent == nullptr ? nullptr : static_cast<ControlNode *>(parent);
        std::vector<int> domain_ids = SiteDomainsForTaskAtScope(task, scope);
        AddInsertionSites(path_begin, task, domain_ids, inner_sites,
                          keyed_sites);
        sites_.insert(sites_.end(), inner_sites.begin(), inner_sites.end());

        if (parent == nullptr)
          continue;
        auto *parent_scope =
            static_cast<ControlNode *>(parent)->GetParentControl();
        std::vector<int> parent_domain_ids =
            SiteDomainsForTaskAtScope(task, parent_scope);
        for (const SyncInsertionSite &inner_site : inner_sites) {
          for (int parent_domain_id : parent_domain_ids) {
            SyncInsertionSiteKey parent_key{parent, task,
                                            inner_site.at_beginning,
                                            inner_site.core, parent_domain_id};
            auto parent_it = keyed_sites.find(parent_key);
            ICHECK(parent_it != keyed_sites.end())
                << "Synchronization insertion site is missing its parent site";
            // One child endpoint may promote into several parent domains, but
            // each domain must select exactly one physical parent endpoint.
            auto [recorded_it, inserted] = parent_sites[inner_site].emplace(
                parent_domain_id, parent_it->second);
            ICHECK(inserted || recorded_it->second.HasSamePhysicalPosition(
                                   parent_it->second))
                << "One synchronization site has conflicting parent sites "
                   "in the same epoch domain";
          }
        }
      }
    }
    std::sort(sites_.begin(), sites_.end());
    sites_.erase(std::unique(sites_.begin(), sites_.end()), sites_.end());
    for (const auto &[key, site] : keyed_sites) {
      auto [_, inserted] = site_ids_.emplace(key, FindSiteId(site));
      ICHECK(inserted) << "Duplicate synchronization insertion-site key";
    }
    parent_site_ids_.resize(sites_.size());
    for (const auto &[site, parents] : parent_sites) {
      std::vector<size_t> &parent_ids = parent_site_ids_[FindSiteId(site)];
      for (const auto &[_, parent] : parents)
        parent_ids.push_back(FindSiteId(parent));
    }
  }

  const SyncInsertionSiteList &Sites() const { return sites_; }

  size_t Size() const { return sites_.size(); }

  const EpochDomain &Domain(const SyncInsertionSite &site) const {
    return domains_.Domain(site.EpochDomainId());
  }

  bool SameEpochDomain(const SyncInsertionSite &lhs,
                       const SyncInsertionSite &rhs) const {
    return lhs.EpochDomainId() == rhs.EpochDomainId();
  }

  template <typename Visitor>
  void ForEachPhysicalSitePair(const DepEdge &edge,
                               const Visitor &visit) const {
    ICHECK(SameEpochDomain(Src(edge), Dst(edge)))
        << "A dependency edge must have one active-epoch domain";
    std::vector<size_t> src_variants = PhysicalSiteVariants(edge.src);
    std::vector<size_t> dst_variants = PhysicalSiteVariants(edge.dst);
    ForEachMatchingDomainSitePair(
        src_variants, dst_variants, [&](size_t src, size_t dst) {
          visit(DepEdge(src, dst, edge.distance, edge.strict));
        });
  }

  template <typename Visitor>
  void ForEachParentSitePair(const DepEdge &edge, const Visitor &visit) const {
    ICHECK(SameEpochDomain(Src(edge), Dst(edge)))
        << "A dependency edge must have one active-epoch domain";
    ICHECK_LT(std::max(edge.src, edge.dst), parent_site_ids_.size());
    const std::vector<size_t> &src_parents = parent_site_ids_[edge.src];
    const std::vector<size_t> &dst_parents = parent_site_ids_[edge.dst];
    ForEachMatchingDomainSitePair(
        src_parents, dst_parents, [&](size_t src, size_t dst) {
          visit(DepEdge(src, dst, edge.distance, edge.strict));
        });
  }

  template <typename Visitor>
  void ForEachMatchingPhysicalSitePair(size_t src_site_id, size_t dst_site_id,
                                       const Visitor &visit) const {
    std::vector<size_t> src_variants = PhysicalSiteVariants(src_site_id);
    std::vector<size_t> dst_variants = PhysicalSiteVariants(dst_site_id);
    ForEachMatchingDomainSitePair(src_variants, dst_variants, visit);
  }

  size_t FindParentSiteInDomain(size_t site_id, int domain_id) const {
    ICHECK_LT(site_id, parent_site_ids_.size());
    std::optional<size_t> result;
    for (size_t parent_site_id : parent_site_ids_[site_id]) {
      if (sites_[parent_site_id].EpochDomainId() != domain_id)
        continue;
      ICHECK(!result.has_value())
          << "Synchronization insertion site has multiple parents in epoch "
             "domain "
          << domain_id;
      result = parent_site_id;
    }
    ICHECK(result.has_value())
        << "Synchronization insertion site has no parent in epoch domain "
        << domain_id;
    return result.value();
  }

  const SyncInsertionSite &Src(const DepEdge &edge) const {
    return sites_[edge.src];
  }

  const SyncInsertionSite &Dst(const DepEdge &edge) const {
    return sites_[edge.dst];
  }

  const SyncInsertionSite &Src(const SyncPoint &sync_point) const {
    return Src(sync_point.edge);
  }

  const SyncInsertionSite &Dst(const SyncPoint &sync_point) const {
    return Dst(sync_point.edge);
  }

  bool IsCrossCore(const SyncPoint &sync_point) const {
    return Src(sync_point).core != Dst(sync_point).core;
  }

  bool UsesFlag(const SyncPoint &sync_point) const {
    return IsCrossCore(sync_point) ||
           Src(sync_point).pipe != Dst(sync_point).pipe;
  }

  bool HasSameFlagKind(const SyncPoint &lhs, const SyncPoint &rhs) const {
    const SyncInsertionSite &lhs_src = Src(lhs);
    const SyncInsertionSite &lhs_dst = Dst(lhs);
    const SyncInsertionSite &rhs_src = Src(rhs);
    const SyncInsertionSite &rhs_dst = Dst(rhs);
    return lhs_src.core == rhs_src.core && lhs_src.pipe == rhs_src.pipe &&
           lhs_dst.core == rhs_dst.core && lhs_dst.pipe == rhs_dst.pipe;
  }

  bool HasCompatibleFlagKind(const SyncPoint &lhs, const SyncPoint &rhs) const {
    bool lhs_cross_core = IsCrossCore(lhs);
    bool rhs_cross_core = IsCrossCore(rhs);
    if (lhs_cross_core || rhs_cross_core)
      return lhs_cross_core && rhs_cross_core;
    return HasSameFlagKind(lhs, rhs);
  }

  CounterSyncSignature
  GetCounterSyncSignature(const SyncPoint &sync_point) const {
    ICHECK_GT(sync_point.counter_group_id, 0);
    const SyncInsertionSite &src = Src(sync_point);
    const SyncInsertionSite &dst = Dst(sync_point);
    return {sync_point.counter_group_id,
            src.core,
            src.pipe,
            dst.core,
            dst.pipe,
            sync_point.num_versions,
            sync_point.edge.distance};
  }

  size_t FindSiteId(const SyncInsertionSite &site) const {
    auto it = std::lower_bound(sites_.begin(), sites_.end(), site);
    ICHECK(it != sites_.end() && *it == site)
        << "Synchronization insertion site was not collected";
    return std::distance(sites_.begin(), it);
  }

  size_t FindSiteId(const SyncInsertionSiteKey &key) const {
    auto it = site_ids_.find(key);
    ICHECK(it != site_ids_.end())
        << "Synchronization insertion-site key was not collected";
    return it->second;
  }

  size_t FindBoundarySite(size_t site_id, bool at_beginning) const {
    SyncInsertionSite boundary = sites_[site_id];
    boundary.at_beginning = at_beginning;
    return FindSiteId(boundary);
  }

  std::vector<size_t>
  CollectSiteIds(const std::vector<IRStructure *> &nodes) const {
    std::vector<size_t> result;
    for (IRStructure *node : nodes) {
      std::vector<TaskNode *> tasks;
      CollectAllTaskNodes(node, tasks);
      IRStructure *parent = node->GetParent();
      ICHECK(parent == nullptr || parent->IsControl());
      auto *scope =
          parent == nullptr ? nullptr : static_cast<ControlNode *>(parent);
      for (TaskNode *task : tasks) {
        AddInsertionSiteIds(node, task, SiteDomainsForTaskAtScope(task, scope),
                            result);
      }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

private:
  std::vector<int> SiteDomainsForTaskAtScope(const TaskNode *task,
                                             ControlNode *scope) const {
    std::vector<int> result = domains_.DomainsForTaskAtScope(task, scope);
    int lexical_domain = domains_.UnconditionalLexicalDomain(scope);
    if (std::find(result.begin(), result.end(), lexical_domain) == result.end())
      result.push_back(lexical_domain);
    return result;
  }

  std::vector<size_t> PhysicalSiteVariants(size_t site_id) const {
    ICHECK_LT(site_id, sites_.size());
    const SyncInsertionSite &site = sites_[site_id];
    size_t begin = site_id;
    while (begin > 0 && site.HasSamePhysicalPosition(sites_[begin - 1]))
      --begin;
    size_t end = site_id + 1;
    while (end < sites_.size() && site.HasSamePhysicalPosition(sites_[end]))
      ++end;
    std::vector<size_t> result;
    result.reserve(end - begin);
    for (size_t candidate_id = begin; candidate_id < end; ++candidate_id)
      result.push_back(candidate_id);
    return result;
  }

  template <typename Visitor>
  void ForEachMatchingDomainSitePair(const std::vector<size_t> &src_sites,
                                     const std::vector<size_t> &dst_sites,
                                     const Visitor &visit) const {
    size_t src_index = 0;
    size_t dst_index = 0;
    while (src_index < src_sites.size() && dst_index < dst_sites.size()) {
      size_t src = src_sites[src_index];
      size_t dst = dst_sites[dst_index];
      int src_domain = sites_[src].EpochDomainId();
      int dst_domain = sites_[dst].EpochDomainId();
      if (src_domain < dst_domain) {
        ++src_index;
        continue;
      }
      if (src_domain > dst_domain) {
        ++dst_index;
        continue;
      }
      visit(src, dst);
      ++src_index;
      ++dst_index;
    }
  }

  void AddInsertionSites(
      IRStructure *path_begin, TaskNode *task,
      const std::vector<int> &domain_ids, SyncInsertionSiteList &sites,
      std::map<SyncInsertionSiteKey, SyncInsertionSite> &keyed_sites) const {
    for (bool at_beginning : {false, true}) {
      for (CoreMask core : kConcreteCores) {
        if (!HasCore(task->GetCoreMask(), core))
          continue;
        auto add = [&](int domain_id) {
          const EpochDomain &domain = domains_.Domain(domain_id);
          sites.emplace_back(path_begin, task, at_beginning, core, domain);
          const SyncInsertionSite &site = sites.back();
          SyncInsertionSiteKey key{path_begin, task, at_beginning, core,
                                   site.EpochDomainId()};
          auto [it, inserted] = keyed_sites.emplace(std::move(key), site);
          ICHECK(inserted || it->second == site)
              << "Synchronization insertion-site key has inconsistent sites";
        };
        for (int domain_id : domain_ids)
          add(domain_id);
      }
    }
  }

  void AddInsertionSiteIds(IRStructure *path_begin, TaskNode *task,
                           const std::vector<int> &domain_ids,
                           std::vector<size_t> &site_ids) const {
    for (bool at_beginning : {false, true}) {
      for (CoreMask core : kConcreteCores) {
        if (!HasCore(task->GetCoreMask(), core))
          continue;
        for (int domain_id : domain_ids) {
          site_ids.push_back(FindSiteId(SyncInsertionSiteKey{
              path_begin, task, at_beginning, core, domain_id}));
        }
      }
    }
  }

  const EpochDomainRegistry &domains_;
  SyncInsertionSiteList sites_;
  std::map<SyncInsertionSiteKey, size_t> site_ids_;
  std::vector<std::vector<size_t>> parent_site_ids_;
};

// ============================================================================
// Dependency analysis model and closure optimization
// ============================================================================

class DepClosure {
public:
  DepClosure() = default;

  explicit DepClosure(size_t num_sites)
      : outgoing_(num_sites), incoming_(num_sites) {}

  bool Contains(const DepEdge &edge) const { return present_.count(edge); }

  bool HasEdge(size_t src, size_t dst, int distance) const {
    return Contains(DepEdge(src, dst, distance, false)) ||
           Contains(DepEdge(src, dst, distance, true));
  }

  int MinDistance(size_t src, size_t dst, bool require_strict = false) const {
    int result = std::numeric_limits<int>::max();
    if (src >= outgoing_.size())
      return result;
    for (const DepEdge &edge : outgoing_[src]) {
      if (edge.dst == dst && (!require_strict || edge.strict))
        result = std::min(result, edge.distance);
    }
    return result;
  }

  void Insert(const DepEdge &edge) {
    if (!present_.insert(edge).second)
      return;
    ICHECK_LT(std::max(edge.src, edge.dst), outgoing_.size());
    edges_.push_back(edge);
    outgoing_[edge.src].push_back(edge);
    incoming_[edge.dst].push_back(edge);
  }

  template <typename Projector>
  void Saturate(const DepEdge &start, int max_distance,
                const Projector &project) {
    if (start.distance > max_distance)
      return;
    std::vector<DepEdge> work;
    std::set<DepEdge> pending;
    auto enqueue_one = [&](const DepEdge &edge) {
      if (edge.distance <= max_distance && !Contains(edge) &&
          pending.insert(edge).second) {
        work.push_back(edge);
      }
    };
    auto enqueue = [&](const DepEdge &edge) { project(edge, enqueue_one); };
    enqueue(start);
    while (!work.empty()) {
      DepEdge edge = work.back();
      work.pop_back();
      if (Contains(edge)) {
        pending.erase(edge);
        continue;
      }
      for (const DepEdge &other : outgoing_[edge.dst]) {
        int distance = edge.distance + other.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(edge.src, other.dst, distance,
                          edge.strict || other.strict));
        }
      }
      for (const DepEdge &other : incoming_[edge.src]) {
        int distance = other.distance + edge.distance;
        if (distance <= max_distance) {
          enqueue(DepEdge(other.src, edge.dst, distance,
                          other.strict || edge.strict));
        }
      }
      pending.erase(edge);
      Insert(edge);
    }
  }

  DepClosure DistanceZero() const {
    DepClosure result(outgoing_.size());
    for (const DepEdge &edge : edges_) {
      if (edge.distance == 0)
        result.Insert(edge);
    }
    return result;
  }

  const std::vector<DepEdge> &Edges() const { return edges_; }

private:
  std::vector<DepEdge> edges_;
  std::set<DepEdge> present_;
  std::vector<std::vector<DepEdge>> outgoing_;
  std::vector<std::vector<DepEdge>> incoming_;
};

struct SyncAnalysisResult {
  DepClosure closure;
  std::vector<VisibleSyncPoint> visible_sync_points;
};

// Ordinary completion edges at the boundary of a selected unit stream. These
// use the same sites, closure and flag allocator as data dependencies. Unit
// protection itself is deliberately never inserted into the completion graph.
struct UnitFlagBoundary {
  ControlNode *scope;
  IRStructure *producer_node;
  TaskNode *producer;
  IRStructure *consumer_node;
  TaskNode *consumer;
  UnitFlagBoundaryRole role;
};

// Check if any operation in stmt has unit_flag_ctl/ctrl != 0. This walks the
// whole statement because a T.PerCoreTask leaf may hold a compound body.
bool HasUnitFlagCtrl(const Stmt &stmt) {
  class Detector : public StmtExprVisitor {
  public:
    bool found{false};

  private:
    void VisitExpr_(const CallNode *op) final {
      if (auto ann = op->annotations.Get("unit_flag_ctrl")) {
        if (const auto *imm = ann->as<IntImmNode>()) {
          found = imm->value != 0;
        } else {
          found = true;
        }
      }
      if (!found)
        StmtExprVisitor::VisitExpr_(op);
    }
  } detector;
  detector(stmt);
  return detector.found;
}

std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe) {
  ICHECK(producer_pipe != "UNKNOWN" && consumer_pipe != "UNKNOWN");
  std::string event = producer_pipe + "_" + consumer_pipe;
  if (event == "MTE2_M")
    return "MTE2_MTE1";
  if (event == "M_MTE2")
    return "MTE1_MTE2";
  if (event == "MTE3_M")
    return "MTE3_MTE1";
  if (event == "M_MTE3")
    return "MTE1_MTE3";
  return event;
}

// These are the same-pipe dependencies the materializer satisfies without a
// barrier. On dav-2201, native MemBase vector instructions need PIPE_V
// completion for dependent UB accesses; instruction issue order is weaker.
// Leave the existing dav-3510 VF behavior unchanged; its completion contract
// needs separate evidence. MTE/FIX issue order alone does not protect
// overlapping accesses either.
// SyncInsertionSite stores raw GetResourcePipeName values; the PIPE_ prefix
// belongs only to names emitted for flags and barriers.
bool HasImplicitSamePipeOrder(const std::string &pipe,
                              AscendArchitecture architecture) {
  return pipe == "S" || pipe == "M" ||
         (pipe == "V" && architecture != AscendArchitecture::kDav2201);
}

// ============================================================================
// Recursive dependency analysis and reuse planning
// ============================================================================
class SyncAnalyzer : public BufferAliasAnalysisContext {
public:
  explicit SyncAnalyzer(
      const MultiBufferPlan &multi_buffer_plan,
      const EpochDomainRegistry &domains, AscendArchitecture architecture,
      BufferVersionMap manual = {}, ConflictHintList root_conflicts = {},
      L0StorageGroups storage_groups = L0StorageGroups(),
      const std::vector<OwnerResultPlan> *owner_results = nullptr,
      const std::vector<OwnerResultPlan> *owner_inputs = nullptr,
      const std::vector<UnitFlagStreamPlan> *unit_streams = nullptr,
      const std::vector<UnitFlagBoundary> *unit_boundaries = nullptr,
      const std::vector<L0OperandSlotPool> *operand_pools = nullptr,
      bool enable_gm_read_release = true,
      bool enable_gm_write_acquire = true,
      bool closed_c_byte_retirement = false)
      : storage_groups_(std::move(storage_groups)),
        multi_buffer_plan_(multi_buffer_plan), domains_(domains),
        architecture_(architecture), manual_(std::move(manual)),
        root_conflicts_(std::move(root_conflicts)),
        owner_results_(owner_results), owner_inputs_(owner_inputs),
        unit_streams_(unit_streams), unit_boundaries_(unit_boundaries),
        operand_pools_(operand_pools),
        enable_gm_read_release_(enable_gm_read_release),
        enable_gm_write_acquire_(enable_gm_write_acquire),
        closed_c_byte_retirement_(closed_c_byte_retirement),
        site_registry_(domains),
        trace_c_lifetimes_(
            runtime::detail::VerboseLoggingEnabled(__FILE__, 1)) {
    for (const MultiBufferInfo &info : multi_buffer_plan_.Infos()) {
      ICHECK(!info.owners.empty());
      std::vector<ControlNode *> owners;
      owners.reserve(info.owners.size());
      for (const MultiBufferOwnerInfo &owner : info.owners)
        owners.push_back(owner.loop);
      auto [_, inserted] =
          multi_buffer_owners_.emplace(info.storage, std::move(owners));
      ICHECK(inserted);
    }
  }

  SyncAnalyzer(const SyncAnalyzer &) = delete;
  SyncAnalyzer &operator=(const SyncAnalyzer &) = delete;

  bool Analyze(const std::vector<std::shared_ptr<IRStructure>> &root) {
    disable_buffer_reuse_ =
        tvm::transform::PassContext::Current()
            ->GetConfig<Bool>(kDisableSharedMemoryReuse, Bool(false))
            .value();
    std::vector<IRStructure *> nodes;
    nodes.reserve(root.size());
    for (const std::shared_ptr<IRStructure> &child : root)
      nodes.push_back(child.get());

    site_registry_.Collect(nodes);
    trace_c_sites_.clear();
    scope_closures_.clear();
    lifetime_orders_.clear();
    SyncAnalysisResult root_result = AnalyzeNodeList(nodes, /*loop=*/nullptr);
    scope_closures_.insert_or_assign(nullptr, std::move(root_result.closure));
    if (owner_fallback_required_ || unit_fallback_required_)
      return false;
    if (operand_pools_) {
      for (const auto &pool : *operand_pools_) {
        for (const auto *member : {&pool.first, &pool.second}) {
          ControlNode *scope = member->generation;
          int domain = domains_.DomainForStorage(member->buffer->data, scope);
          size_t writer = site_registry_.FindSiteId(SyncInsertionSiteKey{
              member->writer->GetChildOnPathFrom(scope), member->writer, false,
              kCoreCube, domain});
          size_t reader = site_registry_.FindSiteId(SyncInsertionSiteKey{
              member->reader->GetChildOnPathFrom(scope), member->reader, true,
              kCoreCube, domain});
          auto closure = scope_closures_.find(scope);
          if (closure == scope_closures_.end() ||
              site_registry_.Sites()[writer].path_end != member->writer ||
              site_registry_.Sites()[reader].path_end != member->reader ||
              MinCompletionDistance(closure->second, writer, reader) != 0) {
            operand_pool_fallback_required_ = true;
            return false;
          }
        }
      }
    }
    for (const SyncPoint &point : sync_points_) {
      ICHECK(!point.single_iteration_binary_plan ||
             point.single_iteration_binary_plan->validated)
          << "dav-2201 local binary flag: unresolved single-iteration "
             "parent lifetime";
    }

    // The contract's presence still enables sequential allocation downstream.
    // Synchronization and flag reuse above remain necessary in this mode.
    if (disable_buffer_reuse_) {
      buffer_aliases_ = {};
      return true;
    }

    BufferAliasAnalyzer buffer_alias_analyzer(*this, storage_groups_);
    buffer_alias_analyzer.Collect(root);
    buffer_alias_analyzer.Analyze(root);
    buffer_alias_analyzer.Validate();
    buffer_aliases_ = buffer_alias_analyzer.GetAliases();
    // Slot credits prove byte separation independently of whole-allocation
    // generation closure. Do not manufacture a phase-END -> phase-START edge:
    // an old disjoint slot and FIX may still execute after the new first load.
    if (operand_pools_)
      for (const auto &pool : *operand_pools_)
        AddBufferAlias(&buffer_aliases_, pool.first.buffer->data,
                       pool.second.buffer->data);
    return true;
  }

  const SyncSiteRegistry &SiteRegistry() const { return site_registry_; }

  std::vector<SyncPoint> &SyncPoints() { return sync_points_; }

  const std::set<std::pair<size_t, size_t>> &ReusablePairs() const {
    return reusable_pairs_;
  }

  const BufferAliasMap &BufferAliases() const { return buffer_aliases_; }
  bool NeedsUnitFallback() const { return unit_fallback_required_; }
  bool NeedsOperandPoolFallback() const {
    return operand_pool_fallback_required_;
  }
  bool HasGmReadReleases() const { return !gm_read_releases_.empty(); }
  bool HasGmWriteAcquires() const { return !gm_write_acquires_.empty(); }
  bool HasPrivateGmHandoffs() const {
    return HasGmReadReleases() || HasGmWriteAcquires();
  }
  const std::vector<GmReadReleaseCut> &GmReadReleases() const {
    return gm_read_releases_;
  }
  const std::vector<GmWriteAcquireCut> &GmWriteAcquires() const {
    return gm_write_acquires_;
  }

  bool HasOwnerResultRelease(const OwnerResultPlan &plan) const {
    for (const SyncPoint &point : sync_points_) {
      if (point.gm_read_release) {
        const auto &cut = gm_read_releases_.at(*point.gm_read_release);
        if (cut.plan.MatchesResult(plan) &&
            cut.plan.MatchesProtocol(point.flag_loop, point.edge.distance,
                                     point.num_versions,
                                     point.counter_group_id))
          return true;
      }
      if (point.gm_write_acquire) {
        const auto &cut = gm_write_acquires_.at(*point.gm_write_acquire);
        if (cut.plan.MatchesResult(plan) &&
            cut.plan.MatchesProtocol(point.flag_loop, point.edge.distance,
                                     point.num_versions,
                                     point.counter_group_id))
          return true;
      }
    }
    int domain = domains_.DomainForStorage(plan.storage, plan.epoch);
    size_t read_end = site_registry_.FindSiteId(
        SyncInsertionSiteKey{plan.consumer->GetChildOnPathFrom(plan.epoch),
                             plan.consumer, false, kCoreVector, domain});
    size_t write_begin = site_registry_.FindSiteId(
        SyncInsertionSiteKey{plan.producer->GetChildOnPathFrom(plan.epoch),
                             plan.producer, true, kCoreCube, domain});
    const DepClosure &closure = scope_closures_.at(plan.epoch);
    int distance = MinCompletionDistance(closure, read_end, write_begin);
    // The ordinary collective release must retire all reads before this GM
    // slot is written again. No owner-result edge contributes to this proof.
    return distance <= plan.num_slots - plan.distance;
  }

  bool HasOwnerInputRelease(const OwnerResultPlan &plan) const {
    for (const SyncPoint &point : sync_points_) {
      if (point.gm_read_release) {
        const auto &cut = gm_read_releases_.at(*point.gm_read_release);
        if (cut.plan.MatchesInput(plan) &&
            cut.plan.MatchesProtocol(point.flag_loop, point.edge.distance,
                                     point.num_versions,
                                     point.counter_group_id))
          return true;
      }
    }
    int domain = domains_.DomainForStorage(plan.storage, plan.epoch);
    size_t read_end = site_registry_.FindSiteId(
        SyncInsertionSiteKey{plan.consumer->GetChildOnPathFrom(plan.epoch),
                             plan.consumer, false, kCoreCube, domain});
    size_t write_begin = site_registry_.FindSiteId(
        SyncInsertionSiteKey{plan.producer->GetChildOnPathFrom(plan.epoch),
                             plan.producer, true, kCoreVector, domain});
    const DepClosure &closure = scope_closures_.at(plan.epoch);
    // This must be the ordinary paired input-consumed release, independent
    // of the new readiness protocol. It retires both owners' reads before
    // either lane may contribute a token for the next use of this GM slot.
    return MinCompletionDistance(closure, read_end, write_begin) <=
           plan.num_slots - plan.distance;
  }

private:
  bool GmHandoffNeedsFallback(bool valid) {
    if (valid || !HasPrivateGmHandoffs())
      return false;
    // The private GM cut may remove an unrelated flag's old wide witness.
    // Retry the same owner/unit/pool choices with ordinary GM release first.
    owner_fallback_required_ = true;
    return true;
  }

  void SelectGmReadRelease(SyncPoint *point, const DepInfo &dep,
                          TaskNode *reader, TaskNode *writer,
                          ControlNode *loop) {
    if (!enable_gm_read_release_ || (!owner_inputs_ && !owner_results_) ||
        architecture_ != AscendArchitecture::kDav2201 || !loop ||
        !site_registry_.IsCrossCore(*point))
      return;
    const SyncInsertionSite &src = site_registry_.Src(*point);
    const SyncInsertionSite &dst = site_registry_.Dst(*point);
    int lexical_domain = domains_.UnconditionalLexicalDomain(loop);
    const EpochDomain &domain = domains_.Domain(lexical_domain);
    if (domain.IsCounter() || domain.owner != loop ||
        !is_one(domain.active_guard) ||
        src.EpochDomainId() != lexical_domain ||
        dst.EpochDomainId() != lexical_domain || !is_one(src.GetGuard()) ||
        !is_one(dst.GetGuard()) ||
        !loop->GetLoopBodyContext().CanProve(
            point->flag_iteration == CalculateIterationCount(loop)))
      return;
    const bool input = src.core == kCoreCube && src.pipe == "MTE2" &&
                       dst.core == kCoreVector && dst.pipe == "MTE3";
    const bool result = src.core == kCoreVector && src.pipe == "MTE2" &&
                        dst.core == kCoreCube && dst.pipe == "FIX";
    if ((!input || !owner_inputs_) && (!result || !owner_results_))
      return;
    auto plan = input
                    ? AnalyzeGmReadRelease(*owner_inputs_, dep, reader, writer,
                                           loop, manual_, multi_buffer_plan_)
                    : AnalyzeGmResultReadRelease(*owner_results_, dep, reader,
                                                 writer, loop, manual_,
                                                 multi_buffer_plan_);
    if (!plan || !plan->MatchesProtocol(point->flag_loop, point->edge.distance,
                                        point->num_versions,
                                        point->counter_group_id))
      return;
    SyncInsertionSite active(plan->reader_boundary, reader, false, src.core,
                             domain, nullptr, true);
    if (active.path_end != reader)
      return;
    active.guard = active.guard && plan->active;
    SyncInsertionSite inactive = src;
    inactive.guard = inactive.guard && plan->inactive;
    point->gm_read_release = gm_read_releases_.size();
    gm_read_releases_.push_back(
        {std::move(*plan), std::move(active), std::move(inactive)});
  }

  void SelectGmWriteAcquire(SyncPoint *point, const DepInfo &dep,
                            TaskNode *reader, TaskNode *writer,
                            ControlNode *loop) {
    if (!enable_gm_write_acquire_ || !owner_results_ ||
        architecture_ != AscendArchitecture::kDav2201 || !loop ||
        !site_registry_.IsCrossCore(*point))
      return;
    const SyncInsertionSite &src = site_registry_.Src(*point);
    const SyncInsertionSite &dst = site_registry_.Dst(*point);
    int lexical_domain = domains_.UnconditionalLexicalDomain(loop);
    const EpochDomain &domain = domains_.Domain(lexical_domain);
    if (domain.IsCounter() || domain.owner != loop ||
        !is_one(domain.active_guard) ||
        src.EpochDomainId() != lexical_domain ||
        dst.EpochDomainId() != lexical_domain || !is_one(src.GetGuard()) ||
        !is_one(dst.GetGuard()) || src.core != kCoreVector || src.pipe != "MTE2" ||
        dst.core != kCoreCube || dst.pipe != "FIX" ||
        !loop->GetLoopBodyContext().CanProve(
            point->flag_iteration == CalculateIterationCount(loop)))
      return;
    auto plan = AnalyzeGmWriteAcquire(*owner_results_, dep, reader, writer, loop,
                                     manual_, multi_buffer_plan_);
    if (!plan || !plan->MatchesProtocol(point->flag_loop, point->edge.distance,
                                        point->num_versions,
                                        point->counter_group_id))
      return;
    SyncInsertionSite active(plan->writer_boundary, writer, true, kCoreCube,
                             domain, nullptr, true);
    if (active.path_end != writer)
      return;
    active.guard = active.guard && plan->active;
    SyncInsertionSite inactive = dst;
    inactive.guard = inactive.guard && plan->inactive;
    point->gm_write_acquire = gm_write_acquires_.size();
    gm_write_acquires_.push_back(
        {std::move(*plan), std::move(active), std::move(inactive)});
  }

  const L0OperandSlotMember *PoolMemoryObligationMember(
      const DepInfo &dep, TaskNode *source, TaskNode *destination,
      ControlNode *loop) const {
    if (!operand_pools_ || dep.kind != DependencyKind::kData)
      return nullptr;
    for (const auto &pool : *operand_pools_) {
      for (const auto *member : {&pool.first, &pool.second}) {
        if (dep.storage && dep.storage.value().same_as(member->buffer->data) &&
            source == member->reader && destination == member->writer)
          return member;
        // A repeated enclosing control also reports old-generation writes
        // against a later generation's reader. The unique full overwrite
        // kills that incoming value; its retained same-generation RAW ready
        // and the physical credit protect the actual read and overwrite.
        // This is an ancestor memory obligation, never a phase completion.
        if (!dep.cross_storage_conflict && dep.storage &&
            dep.storage.value().same_as(member->buffer->data) &&
            source == member->writer && destination == member->reader &&
            dep.distance < 0 && dep.prod_node == dep.cons_node && loop &&
            loop != member->generation && member->generation->IsWithin(loop))
          return member;
        const auto &other = member == &pool.first ? pool.second : pool.first;
        // An early alias hint conservatively creates RAW, WAW and WAR
        // obligations between both logical allocations. The physical pool
        // protects their intersecting bytes with the complete chain:
        // old load -> old whole MAD -> credit -> new load -> new MAD.
        // Keep each allocation's actual RAW ready edge. Only these exact
        // cross-allocation access pairs are replaced; no phase-END relation
        // follows from their independent slot lifetimes.
        if (dep.cross_storage_conflict && !dep.storage &&
            ((source == member->writer &&
              (destination == other.writer || destination == other.reader)) ||
             (source == member->reader && destination == other.writer)))
          return member;
      }
    }
    return nullptr;
  }

  static std::string LifetimeNodeName(IRStructure *node) {
    if (node == nullptr)
      return "<root>";
    std::ostringstream name;
    if (node->IsControl()) {
      name << "loop:" << static_cast<ControlNode *>(node)->control->loop_var;
    } else {
      auto *task = static_cast<TaskNode *>(node);
      Stmt body = task->stmt;
      while (const auto *attribute = body.as<AttrStmtNode>())
        body = attribute->body;
      const auto *evaluate = body.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      name << "task:";
      if (call)
        name << call->op;
      else
        name << body->GetTypeKey();
    }
    name << "[index=" << node->GetIndex() << ",stage=" << node->GetStage()
         << "]";
    return name.str();
  }

  bool TraceCStorage(const Var &storage) const {
    if (!trace_c_lifetimes_)
      return false;
    std::string scope = GetPtrStorageScope(storage);
    return scope == "shared.l0c" || scope == "shared.l0c.dyn";
  }

  void TraceLifetimeSite(const char *action, const Var &storage,
                         size_t site_id) const {
    if (!TraceCStorage(storage))
      return;
    trace_c_sites_.insert(site_id);
    TraceLifetimeSiteFields(action, site_id);
  }

  void TraceLifetimeSiteFields(const char *action, size_t site_id) const {
    const SyncInsertionSite &site = site_registry_.Sites()[site_id];
    const EpochDomain &domain = site_registry_.Domain(site);
    LOG(INFO) << "C-site action=" << action << " id=" << site_id
              << " core=" << site.core << " pipe=" << site.pipe
              << " domain=" << site.domain_id
              << " owner=" << LifetimeNodeName(domain.owner)
              << " counter=" << domain.IsCounter()
              << " active=" << domain.active_guard
              << " begin=" << site.at_beginning
              << " path_begin=" << LifetimeNodeName(site.path_begin)
              << " path_end=" << LifetimeNodeName(site.path_end)
              << " guard=" << site.GetGuard();
  }

  std::vector<size_t> ResolveLifetimeEndpoints(const Var &storage,
                                               TaskNode *task,
                                               bool at_beginning,
                                               ControlNode *scope) const final {
    IRStructure *path_begin = task;
    while (path_begin->GetParent() != scope) {
      path_begin = path_begin->GetParent();
      ICHECK(path_begin != nullptr)
          << "Task is not nested in its buffer-lifetime owner scope";
    }

    int domain_id = domains_.DomainForStorage(storage, scope);

    std::vector<size_t> endpoints;
    for (CoreMask core : kConcreteCores) {
      if (!HasCore(task->GetCoreMask(), core))
        continue;
      endpoints.push_back(site_registry_.FindSiteId(SyncInsertionSiteKey{
          path_begin, task, at_beginning, core, domain_id}));
    }
    std::sort(endpoints.begin(), endpoints.end());
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end()),
                    endpoints.end());
    for (size_t endpoint : endpoints)
      TraceLifetimeSite(at_beginning ? "resolve-begin" : "resolve-end",
                        storage, endpoint);
    return endpoints;
  }

  bool CWriterRetiredByFullExport(const Var &storage, TaskNode *mad,
                                 TaskNode *fix,
                                 ControlNode *result_scope) const final {
    if (!closed_c_byte_retirement_ || !unit_streams_)
      return false;
    for (const auto &plan : *unit_streams_)
      if (plan.c->data.same_as(storage) && plan.result_loop == result_scope &&
          plan.mads.size() == 1 && plan.mads.front() == mad && plan.fix == fix)
        return true;
    return false;
  }

  bool IsClosedCByteConflict(const DepInfo &dep, TaskNode *producer,
                            TaskNode *consumer) const {
    if (!closed_c_byte_retirement_ || !unit_streams_ ||
        dep.kind != DependencyKind::kData || !dep.cross_storage_conflict ||
        dep.storage.has_value() || (dep.distance != 0 && dep.distance != -1))
      return false;
    const UnitFlagStreamPlan *old_result = nullptr, *new_result = nullptr;
    for (const auto &plan : *unit_streams_) {
      if (plan.mads.front() == producer || plan.fix == producer)
        old_result = &plan;
      if (plan.mads.front() == consumer || plan.fix == consumer)
        new_result = &plan;
    }
    // The selected mandatory pair has a complete native C result per owner.
    // Its matching full FIX consumes the old C bytes. Existing real owner
    // drains/entries must prove FIX->next-MAD separation in the final alias
    // analysis; otherwise the untouched UF+PIPE_M program is rebuilt. A wide
    // phase event would reintroduce whole-MAD lifetime through flag reuse.
    // Do not add that event or a synthetic completion edge. A/B obligations
    // and the ordinary validation of all retained flags stay intact.
    return old_result && new_result && old_result != new_result &&
           !(producer == old_result->fix && consumer == new_result->fix);
  }

  size_t PromoteLifetimeSite(const Var &storage, size_t site_id,
                             ControlNode *control) const final {
    ICHECK_LT(site_id, site_registry_.Size());
    int parent_domain_id =
        domains_.DomainForStorage(storage, control->GetParentControl());
    if (closed_c_byte_retirement_ && unit_streams_ && unit_boundaries_ &&
        !site_registry_.Sites()[site_id].at_beginning) {
      for (const auto &plan : *unit_streams_) {
        if (!plan.c->data.same_as(storage) || plan.result_loop != control ||
            site_registry_.Sites()[site_id].path_end != plan.fix)
          continue;
        for (const auto &boundary : *unit_boundaries_) {
          if (boundary.role != UnitFlagBoundaryRole::kStreamDrain ||
              boundary.producer != plan.fix ||
              boundary.scope != control->GetParentControl())
            continue;
          size_t fix_end =
              site_registry_.FindParentSiteInDomain(site_id, parent_domain_id);
          size_t drain_end = site_registry_.FindSiteId(SyncInsertionSiteKey{
              boundary.consumer_node, boundary.consumer, false, kCoreCube,
              parent_domain_id});
          auto scope = scope_closures_.find(boundary.scope);
          if (scope != scope_closures_.end() &&
              MinCompletionDistance(scope->second, fix_end, drain_end) == 0) {
            // Widen only these C bytes to their real FIX_M acquire witness.
            // The empty M task ends after the emitted wait, not after whole
            // MAD retirement. Its parent projection retains that real drain
            // across guarded owners instead of inventing a FIX phase fence.
            TraceLifetimeSite("promote-C-drain", storage, drain_end);
            return drain_end;
          }
        }
      }
    }
    TraceLifetimeSite("promote-source", storage, site_id);
    if (!domains_.Domain(parent_domain_id).IsCounter()) {
      size_t lexical_site = site_registry_.FindParentSiteInDomain(
          site_id,
          domains_.UnconditionalLexicalDomain(control->GetParentControl()));
      TraceLifetimeSite("promote-lexical", storage, lexical_site);
      if (IsUnconditionalControlBoundary(
              lexical_site, control,
              site_registry_.Sites()[site_id].at_beginning)) {
        // Enclose the child in a wider real physical boundary. This does not
        // project its guarded closure: lifetime separation must be proved
        // again using the unconditional parent's own completion graph.
        // A storage-free completion witness need not carry this storage's
        // guarded domain. Select this conservative bound before requesting
        // that narrower parent site.
        TraceLifetimeSite("promote-selected", storage, lexical_site);
        return lexical_site;
      }
    }
    size_t parent_site =
        site_registry_.FindParentSiteInDomain(site_id, parent_domain_id);
    TraceLifetimeSite("promote-parent", storage, parent_site);
    ICHECK_EQ(site_registry_.Sites()[parent_site].path_begin, control);
    TraceLifetimeSite("promote-selected", storage, parent_site);
    return parent_site;
  }

  bool IsUnconditionalControlBoundary(size_t site_id, ControlNode *control,
                                      bool at_beginning) const final {
    ICHECK_LT(site_id, site_registry_.Size());
    const SyncInsertionSite &site = site_registry_.Sites()[site_id];
    const EpochDomain &domain = site_registry_.Domain(site);
    return site.path_begin == control && site.path_end == control &&
           site.at_beginning == at_beginning && !domain.IsCounter() &&
           domain.owner == control->GetParentControl() &&
           is_one(domain.active_guard) && is_one(site.GetGuard());
  }

  std::optional<StorageIterationPeriod>
  StoragePeriod(const Var &storage, ControlNode *scope) const final {
    if (const MultiBufferInfo *info = multi_buffer_plan_.Find(storage)) {
      // A counter epoch can skip lexical iterations or span multiple owners.
      // No implicit conversion to this query's lexical clock is valid.
      if (info->UsesCounter())
        return std::nullopt;
      if (scope != nullptr) {
        if (const auto *owner = info->FindOwner(scope)) {
          if (owner->loop == scope)
            return StorageIterationPeriod{info->num_versions, true};
        }
      }
    }
    if (auto versions = manual_.Get(storage))
      return StorageIterationPeriod{versions.value(), false};
    return StorageIterationPeriod{};
  }

  std::optional<int64_t> MinimumDistance(ControlNode *scope, size_t src,
                                         size_t dst) const final {
    if (trace_c_lifetimes_ && trace_c_sites_.count(src) &&
        trace_c_sites_.count(dst)) {
      TraceLifetimeSiteFields("minimum-source", src);
      TraceLifetimeSiteFields("minimum-destination", dst);
    }
    if (scope == nullptr)
      return SameIterationHappensBefore(scope, src, dst)
                 ? std::optional<int64_t>(0)
                 : std::nullopt;
    const auto &sites = site_registry_.Sites();
    if (sites[src].EpochDomainId() != sites[dst].EpochDomainId())
      return std::nullopt;
    const EpochDomain &domain = site_registry_.Domain(sites[src]);
    if (domain.IsCounter() || domain.owner != scope ||
        UsesVar(domain.active_guard, [&](const VarNode *var) {
          return scope->control->loop_var.same_as(GetRef<Var>(var));
        }))
      return std::nullopt;
    auto order = lifetime_orders_.find(scope);
    if (order == lifetime_orders_.end())
      return std::nullopt;
    return order->second.MinimumDistance(src, dst);
  }

  bool HappensBefore(ControlNode *scope, size_t src, size_t dst,
                     int64_t distance) const final {
    // A negative query is not the same-iteration query. The nonnegative path
    // model cannot prove it; retaining that distinction is essential when
    // subtracting two generation endpoint offsets.
    if (distance < 0)
      return false;
    if (distance == 0)
      return SameIterationHappensBefore(scope, src, dst);
    if (scope == nullptr)
      return false;
    auto minimum = MinimumDistance(scope, src, dst);
    return minimum && distance >= *minimum;
  }

  bool HasImplicitCompletionOrder(size_t src, size_t dst) const {
    const auto &sites = site_registry_.Sites();
    return sites[src].core == sites[dst].core &&
           sites[src].pipe == sites[dst].pipe &&
           HasImplicitSamePipeOrder(sites[src].pipe, architecture_);
  }

  bool HasCompletionEdge(const DepClosure &closure, size_t src,
                         size_t dst) const {
    // A composed strong edge may contain weak issue-order segments around a
    // real synchronization. A path containing only MTE/FIX issue edges cannot
    // certify that an old allocation's accesses have finished.
    return closure.Contains(DepEdge(src, dst, 0, true)) ||
           (HasImplicitCompletionOrder(src, dst) &&
            closure.Contains(DepEdge(src, dst, 0, false)));
  }

  int MinCompletionDistance(const DepClosure &closure, size_t src,
                            size_t dst) const {
    // strict records a real completion relation somewhere in the composed
    // path; weak segments around it need not each be synchronization edges.
    return closure.MinDistance(src, dst, !HasImplicitCompletionOrder(src, dst));
  }

  std::optional<LocalBinaryEpochSpan>
  LocalBinaryActiveSpan(const SyncPoint &sync_point, ControlNode *loop,
                        const PrimExpr &trip_count,
                        const PrimExpr &step) const {
    arith::Analyzer guard_analyzer;
    loop->GetLoopBodyContext().Populate(guard_analyzer);
    PrimExpr source_guard = guard_analyzer.Simplify(
        site_registry_.Domain(site_registry_.Src(sync_point)).active_guard);
    PrimExpr destination_guard = guard_analyzer.Simplify(
        site_registry_.Domain(site_registry_.Dst(sync_point)).active_guard);
    if (!guard_analyzer.CanProve(source_guard == destination_guard))
      return std::nullopt;
    return ContiguousLocalBinarySpan(source_guard, loop, trip_count, step);
  }

  std::optional<LocalBinaryEpochSpan>
  ContiguousLocalBinarySpan(const PrimExpr &source_guard, ControlNode *loop,
                            const PrimExpr &trip_count,
                            const PrimExpr &step) const {
    PrimExpr zero = make_zero(trip_count.dtype());
    if (is_one(source_guard))
      return LocalBinaryEpochSpan{zero, trip_count};
    if (is_zero(source_guard))
      return LocalBinaryEpochSpan{zero, zero};

    // Accept one interval in the lexical coordinate, never a gapped active
    // sequence. Static cut points keep root seed/drain expressions independent
    // of condition snapshots or other values defined inside this loop.
    const ForNode *for_node = loop->control.get();
    Array<PrimExpr> bounds =
        arith::DetectClipBound(source_guard, {for_node->loop_var});
    if (bounds.size() != 2)
      return std::nullopt;
    for (const PrimExpr &bound : bounds) {
      if (bound.defined() && !bound.as<IntImmNode>())
        return std::nullopt;
    }
    PrimExpr first = zero;
    PrimExpr last = trip_count;
    if (bounds[0].defined()) {
      first =
          max(zero, min(trip_count, ceildiv(bounds[0] - for_node->min, step)));
    }
    if (bounds[1].defined()) {
      last = max(
          zero, min(trip_count, indexdiv(bounds[1] - for_node->min, step) + 1));
    }
    // The span also controls kernel-root drain operations, including when
    // this loop never enters. Its expressions may use only outer facts;
    // the body's extent > 0 premise must not remove empty-span protection.
    arith::Analyzer boundary_analyzer;
    loop->task->outer_ctx.Populate(boundary_analyzer);
    first = boundary_analyzer.Simplify(first);
    last = boundary_analyzer.Simplify(last);

    // Check exact equivalence on actual loop iterations, including a nonzero
    // origin or positive step. Inferring a bound from a guard is not enough:
    // holes or an unmodeled predicate must not silently drop flag operations.
    Var epoch("local_binary_active_epoch", trip_count.dtype());
    PrimExpr at_epoch = Substitute(
        source_guard, {{for_node->loop_var, for_node->min + epoch * step}});
    arith::Analyzer proof;
    loop->task->outer_ctx.Populate(proof);
    auto exit = proof.EnterConstraint(zero <= epoch && epoch < trip_count);
    bool exact = proof.CanProve(at_epoch == (first <= epoch && epoch < last));
    exit();
    if (!exact)
      return std::nullopt;
    return LocalBinaryEpochSpan{
        first, boundary_analyzer.Simplify(max(last - first, zero))};
  }

  std::optional<int> FiniteLocalBinaryWidth(ControlNode *loop,
                                            const PrimExpr &trip_count,
                                            int distance) const {
    if (distance < 0 || distance > kIntraCoreFlagLimit)
      return std::nullopt;
    arith::Analyzer analyzer;
    loop->task->outer_ctx.Populate(analyzer);
    arith::ConstIntBound bound = analyzer.const_int_bound(trip_count);
    if (bound->max_value < 0 ||
        bound->max_value > kIntraCoreFlagLimit - distance) {
      return std::nullopt;
    }
    // All real producer epochs and the negative initial epochs have distinct
    // residues. Width one also represents an empty distance-zero stream.
    return std::max<int>(1, static_cast<int>(bound->max_value) + distance);
  }

  bool SameIterationHappensBefore(ControlNode *scope, size_t src,
                                  size_t dst) const {
    auto current = scope_closures_.find(scope);
    ICHECK(current != scope_closures_.end())
        << "Buffer alias analysis is missing a scope closure";
    const DepClosure &closure = current->second;
    if (HasCompletionEdge(closure, src, dst))
      return true;
    const SyncInsertionSiteList &sites = site_registry_.Sites();
    ICHECK_LT(std::max(src, dst), sites.size());
    const SyncInsertionSite &query_src = sites[src];
    const SyncInsertionSite &query_dst = sites[dst];
    ConstrSet proof_ctx =
        scope != nullptr ? scope->GetLoopBodyContext() : ConstrSet();
    PrimExpr query_guard = site_registry_.Domain(query_src).active_guard &&
                           site_registry_.Domain(query_dst).active_guard;
    bool proven = false;
    site_registry_.ForEachMatchingPhysicalSitePair(
        src, dst, [&](size_t physical_src, size_t physical_dst) {
          if (proven ||
              !HasCompletionEdge(closure, physical_src, physical_dst)) {
            return;
          }
          PrimExpr candidate_guard =
              site_registry_.Domain(sites[physical_src]).active_guard;
          proven = GuardImplies(query_guard, candidate_guard, proof_ctx);
        });
    return proven;
  }

  SyncAnalysisResult AnalyzeNodeList(const std::vector<IRStructure *> &nodes,
                                     ControlNode *loop) {
    SyncAnalysisResult result{DepClosure(site_registry_.Size()), {}};
    std::vector<size_t> task_site_ids = site_registry_.CollectSiteIds(nodes);

    std::vector<std::vector<VisibleSyncPoint>> child_sync_points;
    for (IRStructure *node : nodes) {
      SyncAnalysisResult child_result = AnalyzeNode(node);
      if (owner_fallback_required_ || unit_fallback_required_)
        return result;
      AddPromotedChildClosure(result.closure, node, child_result.closure, loop);
      child_sync_points.push_back(std::move(child_result.visible_sync_points));
    }

    std::vector<SyncPoint> local_sync_points = CollectSyncPoints(nodes, loop);
    int max_distance = loop == nullptr ? 0 : 1;
    for (const SyncPoint &sync_point : local_sync_points)
      max_distance = std::max(max_distance, sync_point.edge.distance);

    AddSamePipeEdges(result.closure, task_site_ids, max_distance, loop);
    RemoveNestedCoveredLocalProtocols(local_sync_points, child_sync_points,
                                      result.closure, max_distance, loop);
    OptimizeSyncPoints(local_sync_points, result.closure, max_distance, loop);
    if (loop != nullptr)
      OptimizeFlagVersions(local_sync_points, result.closure, loop);
    if (owner_fallback_required_ || unit_fallback_required_)
      return result;
    ResolveSingleIterationBinaryPoints(child_sync_points, result.closure, loop);
    if (owner_fallback_required_ || unit_fallback_required_)
      return result;

    std::vector<VisibleSyncPoint> local_points;
    local_points.reserve(local_sync_points.size());
    for (SyncPoint &sync_point : local_sync_points) {
      size_t sync_point_id = AppendSyncPoint(std::move(sync_point));
      if (sync_points_[sync_point_id].IsPrivateGmHandoff())
        continue;
      local_points.push_back(
          {sync_point_id, {sync_points_[sync_point_id].edge}});
    }

    for (size_t i = 0; i < child_sync_points.size(); ++i) {
      for (size_t j = i + 1; j < child_sync_points.size(); ++j) {
        RecordReusablePairs(
            child_sync_points[i], child_sync_points[j], result.closure,
            /*allow_cross_iter=*/true, /*allow_mixed_iter=*/false);
      }
      RecordReusablePairs(child_sync_points[i], local_points, result.closure,
                          /*allow_cross_iter=*/false,
                          /*allow_mixed_iter=*/false);
      result.visible_sync_points.insert(result.visible_sync_points.end(),
                                        child_sync_points[i].begin(),
                                        child_sync_points[i].end());
    }

    // Mixed same/cross-iteration reuse is checked only at the collection level
    // where both points still have the same ring context.
    RecordReusablePairs(local_points, local_points, result.closure,
                        /*allow_cross_iter=*/false, /*allow_mixed_iter=*/true);
    result.visible_sync_points.insert(result.visible_sync_points.end(),
                                      local_points.begin(), local_points.end());
    RecordLifetimeOrder(task_site_ids, loop, result.closure);
    if (loop != nullptr)
      scope_closures_.insert_or_assign(loop, result.closure);
    result.closure = result.closure.DistanceZero();
    return result;
  }

  SyncAnalysisResult AnalyzeNode(IRStructure *node) {
    if (node == nullptr)
      return {DepClosure(site_registry_.Size()), {}};

    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      if (control->children.empty() || !control->control.defined())
        return {DepClosure(site_registry_.Size()), {}};

      std::vector<IRStructure *> nodes;
      for (const std::shared_ptr<IRStructure> &child : control->children)
        nodes.push_back(child.get());
      SyncAnalysisResult result = AnalyzeNodeList(nodes, control);

      if (!control->BodyHasLoopBreak()) {
        std::vector<VisibleSyncPoint> promoted_sync_points;
        for (VisibleSyncPoint &visible : result.visible_sync_points) {
          if (PromoteReuseSites(visible))
            promoted_sync_points.push_back(std::move(visible));
        }
        result.visible_sync_points = std::move(promoted_sync_points);
      } else {
        result.visible_sync_points.clear();
      }
      return result;
    }

    if (node->IsTask())
      return {DepClosure(site_registry_.Size()), {}};

    LOG(FATAL) << "Unknown IRStructure kind in SyncAnalyzer";
    return {DepClosure(site_registry_.Size()), {}};
  }

  size_t CounterAccessSite(const MultiBufferInfo &info,
                           const MultiBufferOwnerInfo &owner, TaskNode *task,
                           CoreMask core, bool at_beginning) const {
    int domain_id = domains_.DomainForStorage(info.storage, owner.loop);
    return site_registry_.FindSiteId(
        SyncInsertionSiteKey{task->GetChildOnPathFrom(owner.loop), task,
                             at_beginning, core, domain_id});
  }

  size_t AppendSyncPoint(SyncPoint sync_point) {
    if (sync_point.counter_group_id > 0) {
      // Local flag-version optimization has finished, so the channel key is
      // stable for both parent-level coverage and final flag allocation.
      CounterSyncSignature signature =
          site_registry_.GetCounterSyncSignature(sync_point);
      int ordinal =
          counter_channel_occurrences_[sync_point.flag_loop][signature]++;
      sync_point.counter_channel.emplace(std::move(signature), ordinal);
    }
    size_t index = sync_points_.size();
    sync_points_.push_back(std::move(sync_point));
    return index;
  }

  bool IsFullRingCounterSync(const SyncPoint &sync_point,
                             const MultiBufferInfo &info,
                             ControlNode *owner) const {
    if (!sync_point.counter_channel.has_value())
      return false;

    // AppendSyncPoint assigns a counter channel only after validating the
    // endpoints against their counter domain. Reassert that relation here
    // because sibling-owner coverage uses flag_loop as the endpoint owner.
    const SyncInsertionSite &src = site_registry_.Src(sync_point);
    const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
    ICHECK_GT(sync_point.counter_group_id, 0);
    ICHECK_EQ(src.EpochDomainId(), dst.EpochDomainId());
    const EpochDomain &domain = site_registry_.Domain(src);
    ICHECK(domain.IsCounter());
    ICHECK_EQ(domain.owner, sync_point.flag_loop);

    return sync_point.flag_loop == owner &&
           sync_point.counter_group_id == info.counter_group_id &&
           sync_point.edge.distance == info.num_versions &&
           sync_point.num_versions == info.num_versions;
  }

  bool IsCounterDependencyCovered(const MultiBufferInfo &info,
                                  TaskNode *producer, CoreMask producer_core,
                                  TaskNode *consumer, CoreMask consumer_core,
                                  ControlNode *analysis_loop) const {
    ICHECK(info.UsesCounter());
    const MultiBufferOwnerInfo *producer_owner = info.FindOwner(producer);
    const MultiBufferOwnerInfo *consumer_owner = info.FindOwner(consumer);
    if (producer_owner == nullptr || consumer_owner == nullptr ||
        producer_owner == consumer_owner) {
      return false;
    }
    IRStructure *common_ancestor =
        producer_owner->loop->GetLowestCommonAncestor(consumer_owner->loop);
    if (common_ancestor == producer_owner->loop ||
        common_ancestor == consumer_owner->loop) {
      LOG(FATAL) << "Cannot resolve dependencies between nested multi-buffer "
                    "owners";
    }
    if (producer_owner->loop->BodyHasLoopBreak() ||
        consumer_owner->loop->BodyHasLoopBreak() ||
        common_ancestor != analysis_loop) {
      return false;
    }

    size_t producer_access = CounterAccessSite(
        info, *producer_owner, producer, producer_core, /*at_beginning=*/false);
    size_t consumer_access = CounterAccessSite(
        info, *consumer_owner, consumer, consumer_core, /*at_beginning=*/true);
    auto ordered = [&](ControlNode *owner, size_t src, size_t dst) {
      if (src == dst)
        return true;
      auto it = scope_closures_.find(owner);
      ICHECK(it != scope_closures_.end())
          << "Counter owner closure was not recorded";
      return it->second.HasEdge(src, dst, 0);
    };

    for (const SyncPoint &producer_sync : sync_points_) {
      if (!IsFullRingCounterSync(producer_sync, info, producer_owner->loop) ||
          !ordered(producer_owner->loop, producer_access,
                   producer_sync.edge.src)) {
        continue;
      }

      for (const SyncPoint &consumer_sync : sync_points_) {
        if (!IsFullRingCounterSync(consumer_sync, info, consumer_owner->loop) ||
            consumer_sync.GetCounterChannel() !=
                producer_sync.GetCounterChannel() ||
            !ordered(consumer_owner->loop, consumer_sync.edge.dst,
                     consumer_access)) {
          continue;
        }
        return true;
      }
    }
    return false;
  }

  int FindSharedDependencyDomain(TaskNode *producer, TaskNode *consumer,
                                 ControlNode *loop) const {
    std::vector<int> producer_domains =
        domains_.DomainsForTaskAtScope(producer, loop);
    std::vector<int> consumer_domains =
        domains_.DomainsForTaskAtScope(consumer, loop);
    int lexical_domain = domains_.UnconditionalLexicalDomain(loop);
    std::vector<int> candidates = producer_domains;
    candidates.insert(candidates.end(), consumer_domains.begin(),
                      consumer_domains.end());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());

    // A guarded source domain is available at both endpoints only when both
    // tasks already use it. The unconditional lexical domain needs no guard
    // value and is therefore always a safe common fallback.
    auto has_site = [&](const std::vector<int> &task_domains, int candidate) {
      return candidate == lexical_domain ||
             std::find(task_domains.begin(), task_domains.end(), candidate) !=
                 task_domains.end();
    };
    auto projects_to_task = [&](int candidate,
                                const std::vector<int> &task_domains) {
      if (task_domains.empty())
        return candidate == lexical_domain;
      return std::any_of(
          task_domains.begin(), task_domains.end(), [&](int target) {
            return domains_.CanProjectDomain(candidate, target, loop);
          });
    };
    for (int candidate : candidates) {
      if (has_site(producer_domains, candidate) &&
          has_site(consumer_domains, candidate) &&
          projects_to_task(candidate, producer_domains) &&
          projects_to_task(candidate, consumer_domains)) {
        return candidate;
      }
    }
    ICHECK(projects_to_task(lexical_domain, producer_domains) &&
           projects_to_task(lexical_domain, consumer_domains))
        << "The unconditional lexical domain must project to every task at "
           "its scope";
    return lexical_domain;
  }

  // Discharge a storage WAR, not a whole-loop completion relation. With a
  // direct T-trip iteration owner, equal physical slots across parent
  // iterations have delta = d*T + k_cons - k_prod > 0, divisible by W.
  // Consequently delta >= W, and the retained child release protects each
  // actual overwritten slot, including an odd T. Its last read need not
  // complete before the next parent's first write (those slots may differ).
  bool IsParentVersionedWarCovered(const DepInfo &dep,
                                   const MultiBufferInfo &info,
                                   TaskNode *reader, TaskNode *writer,
                                   ControlNode *parent) const {
    const bool operand =
        reader->GetPipeMask() == PipeMask(ResourcePipe::kCube) &&
        writer->GetPipeMask() == PipeMask(ResourcePipe::kMTE1);
    if (architecture_ != AscendArchitecture::kDav2201 || !parent ||
        dep.kind != DependencyKind::kData || dep.distance != -1 ||
        dep.cross_storage_conflict || info.UsesCounter() ||
        info.num_versions <= 1 || info.owners.size() != 1 ||
        reader->GetCoreMask() != kCoreCube ||
        writer->GetCoreMask() != kCoreCube ||
        (!operand &&
         (reader->GetPipeMask() != PipeMask(ResourcePipe::kMTE1) ||
          writer->GetPipeMask() != PipeMask(ResourcePipe::kMTE2))))
      return false;
    ControlNode *child = info.owners.front().loop;
    auto finite_loop = [](ControlNode *loop, bool allow_outer_stage = false) {
      const ForNode *node = loop->control.get();
      const auto *extent = node->extent.as<IntImmNode>();
      PrimExpr step = node->step.has_value() ? node->step.value()
                                             : IntImm(node->extent.dtype(), 1);
      return (loop->GetStage() == 0 ||
              (allow_outer_stage && loop->GetStage() >= 0)) &&
             loop->GetGuards().empty() &&
             !loop->BodyHasLoopBreak() && node->kind == ForKind::kSerial &&
             is_zero(node->min) && is_one(step) && extent && extent->value > 0;
    };
    if (child->GetParentControl() != parent || dep.prod_node != child ||
        dep.cons_node != child || reader->GetParent() != child ||
        writer->GetParent() != child || !finite_loop(parent) ||
        !finite_loop(child, operand) ||
        child->control->extent.as<IntImmNode>()->value < info.num_versions ||
        !reader->GetGuards().empty() || !writer->GetGuards().empty())
      return false;

    // A child's outer stage translates this entire owner invocation by the
    // same parent epoch. It does not reset its flattened physical slot clock:
    // (p - offset) * T + k advances by T at the next invocation, so equal slots
    // still differ by a positive multiple of W. LowerScheduledTIR applies
    // that translation to the retained ready/release events and both typed
    // endpoints together (offset is stage minus the minimum sibling stage).
    // Explicit child/endpoint guards remain rejected;
    // their active sequence would require a separate epoch certificate.

    auto uses_storage = [&](const ObjectRef &expression) {
      bool found = false;
      PostOrderVisit(expression, [&](const ObjectRef &node) {
        if (const auto *load = node.as<BufferLoadNode>())
          found |= load->buffer->data.same_as(info.storage);
        if (const auto *store = node.as<BufferStoreNode>())
          found |= store->buffer->data.same_as(info.storage);
        if (const auto *var = node.as<VarNode>())
          found |= var == info.storage.get();
      });
      return found;
    };
    auto storage_region = [&](TaskNode *task,
                              bool writing) -> Optional<BufferRegion> {
      Stmt body = task->stmt;
      while (const auto *attribute = body.as<AttrStmtNode>())
        body = attribute->body;
      const auto *evaluate = body.as<EvaluateNode>();
      const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
      if (!call || !(call->op.same_as(AscendCopy::Get()) ||
                     call->op.same_as(Copy::Get()) ||
                     (operand && !writing && call->op.same_as(Gemm::Get())) ||
                     ((!writing || operand) &&
                      call->op.same_as(AscendCubeLoad3D::Get()))))
        return std::nullopt;
      // Typed regions declare the only storage accesses. A scalar load or
      // data-pointer escape in an index, hint, or other argument is additional
      // work even if its region hull happens to merge into the full tile.
      for (const PrimExpr &argument : call->args) {
        const auto *region_call = argument.as<CallNode>();
        if (region_call && region_call->op.same_as(region())) {
          BufferRegion bounds =
              NormalizeToBufferRegion(GetRef<Call>(region_call));
          for (const Range &range : bounds->region)
            if (uses_storage(range->min) || uses_storage(range->extent))
              return std::nullopt;
        } else if (uses_storage(argument)) {
          return std::nullopt;
        }
      }
      for (const auto &[key, value] : call->annotations) {
        auto expression = value.as<PrimExpr>();
        if (!expression || uses_storage(*expression))
          return std::nullopt;
      }
      TileOperator operation = ParseOperator(GetRef<Call>(call));
      BufferRegion source, destination;
      if (operand && !writing) {
        const auto *gemm = operation.as<GemmNode>();
        if (!gemm)
          return std::nullopt;
        bool a = gemm->a_->data.same_as(info.storage);
        bool b = gemm->b_->data.same_as(info.storage);
        if (a == b || gemm->c_->data.same_as(info.storage))
          return std::nullopt;
        source = a ? gemm->aRegion_ : gemm->bRegion_;
        destination = gemm->cRegion_;
      } else if (const auto *copy = operation.as<CopyNode>()) {
        source = BufferRegion(copy->src, copy->src_range);
        destination = BufferRegion(copy->dst, copy->dst_range);
      } else if (const auto *load = operation.as<AscendCubeLoad3DNode>()) {
        source = load->src;
        destination = load->dst;
      } else {
        return std::nullopt;
      }
      if (operand) {
        bool l0_operand =
            (writing ? destination : source)->buffer.scope() == "shared.l0a" ||
            (writing ? destination : source)->buffer.scope() == "shared.l0b";
        if (!l0_operand ||
            (writing ? source->buffer.scope() != "shared.l1"
                     : destination->buffer.scope() != "shared.l0c"))
          return std::nullopt;
      } else if (writing ? (source->buffer.scope() != "global" ||
                     destination->buffer.scope() != "shared.l1")
                  : (source->buffer.scope() != "shared.l1" ||
                     (destination->buffer.scope() != "shared.l0a" &&
                      destination->buffer.scope() != "shared.l0b")))
        return std::nullopt;
      BufferRegion result = writing ? destination : source;
      if (!result->buffer->data.same_as(info.storage) ||
          result->region.size() != result->buffer->shape.size())
        return std::nullopt;
      for (size_t axis = 0; axis < result->region.size(); ++axis)
        if (!is_zero(result->region[axis]->min) ||
            !ffi::StructuralEqual()(result->region[axis]->extent,
                                    result->buffer->shape[axis]))
          return std::nullopt;
      return result;
    };
    auto read = storage_region(reader, false);
    auto write = storage_region(writer, true);
    if (!read || !write || !read.value()->buffer.same_as(write.value()->buffer))
      return false;
    std::function<bool(IRStructure *)> closed_accesses =
        [&](IRStructure *node) {
          if (node->GuardsTouchStorage(info.storage))
            return false;
          if (node->IsTask())
            return node == reader || node == writer ||
                   !uses_storage(static_cast<TaskNode *>(node)->stmt);
          auto *control = static_cast<ControlNode *>(node);
          if (uses_storage(control->control->min) ||
              uses_storage(control->control->extent) ||
              (control->control->step.has_value() &&
               uses_storage(control->control->step.value())))
            return false;
          return std::all_of(control->children.begin(), control->children.end(),
                             [&](const auto &nested) {
                               return closed_accesses(nested.get());
                             });
        };
    if (!closed_accesses(parent))
      return false;

    PrimExpr clock = MultiBufferIteration(info);
    auto full_ring_point = [&](const SyncPoint &point, bool release) {
      const auto &src = site_registry_.Src(point);
      const auto &dst = site_registry_.Dst(point);
      return point.flag_loop == child &&
             point.dependency_kind == DependencyKind::kData &&
             point.edge.strict && point.counter_group_id == 0 &&
             !point.local_binary_plan && !point.single_iteration_binary_plan &&
             point.unit_boundary == UnitFlagBoundaryRole::kNone &&
             point.num_versions == info.num_versions &&
             point.edge.distance == (release ? info.num_versions : 0) &&
             ffi::StructuralEqual()(point.flag_iteration, clock) &&
             (release ? src.path_end == reader : dst.path_end == reader) &&
             ((release ? dst.path_end : src.path_end) == writer ||
              (operand && (release ? dst.path_end : src.path_end)->IsTask() &&
               (release ? dst.path_end : src.path_end)->GetParent() == child)) &&
             src.path_begin == src.path_end && dst.path_begin == dst.path_end &&
             src.core == kCoreCube && dst.core == kCoreCube &&
             src.pipe == (release ? (operand ? "M" : "MTE1")
                                 : (operand ? "MTE1" : "MTE2")) &&
             dst.pipe == (release ? (operand ? "MTE1" : "MTE2")
                                 : (operand ? "M" : "MTE1")) &&
             is_one(src.GetGuard()) && is_one(dst.GetGuard()) &&
             !site_registry_.Domain(src).IsCounter() &&
             !site_registry_.Domain(dst).IsCounter() &&
             is_one(site_registry_.Domain(src).active_guard) &&
             is_one(site_registry_.Domain(dst).active_guard);
    };
    auto closure = scope_closures_.find(child);
    if (closure == scope_closures_.end())
      return false;
    bool ready = false, released = false;
    // These points were appended only after child optimization and ordinary
    // binary validation. Both real protocols (including root seed/drain)
    // remain emitted; no parent completion edge is manufactured here.
    for (const SyncPoint &point : sync_points_) {
      ready |= full_ring_point(point, false);
      if (full_ring_point(point, true) &&
          MinCompletionDistance(closure->second, point.edge.dst,
                                point.edge.src) == 0)
        released = true;
    }
    if (!ready || !released)
      return false;
    if (!operand)
      return true;
    // Child optimization may put one ready after the last operand load and
    // one release wait before the first. Consume the existing complete paths
    // for this exact Buffer's writer; MTE1 issue order alone is insufficient.
    // The strict segment must still come from the retained real hard events.
    int domain = domains_.DomainForStorage(info.storage, child);
    auto site = [&](TaskNode *task, bool beginning) {
      return site_registry_.FindSiteId(
          SyncInsertionSiteKey{task, task, beginning, kCoreCube, domain});
    };
    return MinCompletionDistance(closure->second, site(writer, false),
                                  site(reader, true)) == 0 &&
           MinCompletionDistance(closure->second, site(reader, false),
                                  site(writer, true)) <= info.num_versions;
  }

  // A repeated full operand load also has a parent WAW. Its exact slot's
  // previous load completes before MAD can read it, and MAD's real release
  // completes before the next overwrite. Reuse that same closed RAW+WAR
  // certificate instead of issuing a wider MTE1 barrier. Merely finding the
  // same writer in a descendant, or same-pipe issue order, proves nothing.
  bool IsParentOperandWriteCovered(const DepInfo &dep,
                                   const MultiBufferInfo &info,
                                   TaskNode *producer, TaskNode *consumer,
                                   ControlNode *parent) const {
    if (producer != consumer ||
        producer->GetPipeMask() != PipeMask(ResourcePipe::kMTE1) ||
        !producer->WritesStorage(info.storage) || info.owners.size() != 1)
      return false;
    ControlNode *child = info.owners.front().loop;
    TaskNode *reader = nullptr;
    for (const auto &node : child->children) {
      if (!node->IsTask() || !node->ReadsStorage(info.storage))
        continue;
      if (reader)
        return false;
      reader = static_cast<TaskNode *>(node.get());
    }
    return reader && reader->GetPipeMask() == PipeMask(ResourcePipe::kCube) &&
           IsParentVersionedWarCovered(dep, info, reader, producer, parent);
  }

  std::vector<SyncPoint>
  CollectSyncPoints(const std::vector<IRStructure *> &nodes,
                    ControlNode *loop) {
    std::vector<DepInfo> deps =
        AnalyzeDependencies(nodes, loop, manual_, multi_buffer_owners_,
                            &dependency_cache_, root_conflicts_);
    if (loop != nullptr) {
      std::vector<DepInfo> directed_deps;
      directed_deps.reserve(deps.size() * 2);
      for (DepInfo &dep : deps) {
        if (dep.kind != DependencyKind::kOwnerExclusion) {
          directed_deps.push_back(std::move(dep));
          continue;
        }

        ICHECK_EQ(dep.prod_node->GetParent(), loop);
        ICHECK_EQ(dep.cons_node->GetParent(), loop);
        ICHECK_NE(dep.prod_node->GetIndex(), dep.cons_node->GetIndex());
        bool lhs_first = dep.prod_node->GetIndex() < dep.cons_node->GetIndex();
        IRStructure *first = lhs_first ? dep.prod_node : dep.cons_node;
        IRStructure *second = lhs_first ? dep.cons_node : dep.prod_node;
        std::vector<DependencyTaskPair> first_to_second =
            std::move(dep.task_pairs);
        if (!lhs_first) {
          for (auto &[producer, consumer] : first_to_second)
            std::swap(producer, consumer);
        }
        // A stage-s task executes logical iteration (physical_iteration - s).
        // Therefore two owners ordered in one physical iteration have logical
        // distances stage(first)-stage(second) and its wraparound complement.
        int forward_distance = first->GetStage() - second->GetStage();
        std::vector<DependencyTaskPair> second_to_first;
        second_to_first.reserve(first_to_second.size());
        for (const auto &[producer, consumer] : first_to_second)
          second_to_first.emplace_back(consumer, producer);
        directed_deps.push_back({first, second, dep.storage,
                                 std::move(first_to_second), forward_distance,
                                 DependencyKind::kOwnerExclusion});
        directed_deps.push_back(
            {second, first, dep.storage, std::move(second_to_first),
             1 - forward_distance, DependencyKind::kOwnerExclusion});
      }
      deps = std::move(directed_deps);
    }

    auto expand_cores = [](CoreMask task_mask) {
      std::vector<CoreMask> result;
      for (CoreMask core : kConcreteCores) {
        if (HasCore(task_mask, core))
          result.push_back(core);
      }
      return result;
    };

    std::vector<SyncPoint> sync_points;
    for (const DepInfo &dep : deps) {
      std::string storage_scope;
      if (dep.storage.has_value()) {
        storage_scope = GetPtrStorageScope(dep.storage.value());
        if (storage_scope == "local" || storage_scope == "local.var" ||
            storage_scope == "local.fragment") {
          continue;
        }
      }

      int distance = dep.distance;
      int num_versions = 1;
      // A boundary outside every storage owner uses the analysis loop's
      // lexical clock. Storage identity alone cannot assign a descendant
      // owner's counter to this parent epoch; within-owner edges override
      // this default with their actual storage-local clock below.
      ControlNode *flag_loop = loop;
      const MultiBufferInfo *counter_info = nullptr;
      const MultiBufferOwnerInfo *counter_owner = nullptr;

      const MultiBufferInfo *info =
          dep.storage.has_value() ? multi_buffer_plan_.Find(dep.storage.value())
                                  : nullptr;
      int manual_versions = 0;
      if (dep.storage.has_value()) {
        if (auto versions = manual_.Get(dep.storage.value()))
          manual_versions = versions.value();
      }
      const MultiBufferOwnerInfo *enclosing_owner = nullptr;
      if (info != nullptr && loop != nullptr) {
        if (!info->UsesCounter())
          ICHECK_EQ(info->owners.size(), 1U);
        enclosing_owner = info->FindOwner(loop);
      }
      bool within_owner = enclosing_owner != nullptr;
      if (within_owner) {
        // Every dependency in an automatic ring owner's subtree uses that
        // owner's clock so descendant closure edges compose with owner-level
        // edges. Dependencies outside all owner subtrees remain ordinary
        // one-slot edges; sibling-owner obligations may instead be discharged
        // by matching complete counter protocols in both owner-local closures.
        num_versions = info->num_versions;
        flag_loop = enclosing_owner->loop;
        if (info->UsesCounter()) {
          counter_info = info;
          counter_owner = enclosing_owner;
        }
        // Only an unresolved dependency carried by the owner loop spans the
        // full physical ring. A descendant loop still carries by one of its
        // own iterations.
        if (loop == enclosing_owner->loop && distance < 0)
          distance = info->num_versions;
      } else if (info == nullptr) {
        if (manual_versions) {
          num_versions = dep.distance > 0 ? dep.distance : manual_versions;
          flag_loop = loop;
        }
      }
      if (distance < 0 && dep.kind == DependencyKind::kData)
        distance = 1;
      ICHECK_EQ(counter_info != nullptr, counter_owner != nullptr);
      std::optional<int> dependency_domain_id;
      if (dep.storage.has_value()) {
        dependency_domain_id =
            domains_.DomainForStorage(dep.storage.value(), loop);
      }
      if (counter_info != nullptr) {
        ICHECK(dependency_domain_id.has_value());
        ICHECK_EQ(dependency_domain_id.value(),
                  domains_.DomainForStorage(counter_info->storage,
                                            counter_owner->loop));
      }
      for (const auto &[producer, consumer] : dep.task_pairs) {
        if (IsClosedCByteConflict(dep, producer, consumer))
          continue;
        const L0OperandSlotMember *pool_member =
            PoolMemoryObligationMember(dep, producer, consumer, loop);
        // Every physical overwritten interval consumes its last reader's
        // credit. Only its own generation recurrence is also a whole-task
        // completion relation; a parent/phase envelope is deliberately not.
        bool pool_local_release =
            pool_member && !dep.cross_storage_conflict &&
            loop == pool_member->generation;
        if (pool_member && !pool_local_release)
          continue;
        if (info != nullptr) {
          if (IsParentVersionedWarCovered(dep, *info, producer, consumer, loop) ||
              IsParentOperandWriteCovered(dep, *info, producer, consumer, loop))
            continue;
        }
        // A forced conflict between distinct storages requests ordering in
        // this scope. Its phase envelope also spans inactive epochs.
        // Emit a real boundary protocol in the lexical clock rather than an
        // active-only task protocol that cannot retire the previous FIX at
        // the next phase entry. Storage-free SPR dependencies retain their
        // ordinary shared activation domain.
        bool scope_conflict =
            architecture_ == AscendArchitecture::kDav2201 && loop != nullptr &&
            dep.cross_storage_conflict;
        int pair_domain_id =
            scope_conflict
                ? domains_.UnconditionalLexicalDomain(loop)
            : dependency_domain_id.has_value()
                ? dependency_domain_id.value()
                : FindSharedDependencyDomain(producer, consumer, loop);
        std::string producer_pipe =
            GetResourcePipeName(producer->GetPipeMask());
        std::string consumer_pipe =
            GetResourcePipeName(consumer->GetPipeMask());
        std::string hard_event = GetHardEvent(producer_pipe, consumer_pipe);
        // dav-2201 requires a closed, selected stream. Numeric controls alone
        // do not prove C ownership or completion. Keep dav-3510 compatibility
        // separate: its unit protocol is not selected by this 2201 analysis.
        bool fix_unit_flag =
            dep.kind == DependencyKind::kData && dep.storage.has_value() &&
            (storage_scope == "shared.l0c" ||
             storage_scope == "shared.l0c.dyn") &&
            (hard_event == "M_FIX" || hard_event == "FIX_M") &&
            ((architecture_ == AscendArchitecture::kDav2201 && unit_streams_ &&
              CoversUnitFlagDependency(*unit_streams_, loop, dep, producer,
                                       consumer)) ||
             (architecture_ != AscendArchitecture::kDav2201 &&
              (HasUnitFlagCtrl(producer->stmt) ||
               HasUnitFlagCtrl(consumer->stmt))));
        for (CoreMask producer_core : expand_cores(producer->GetCoreMask())) {
          for (CoreMask consumer_core : expand_cores(consumer->GetCoreMask())) {
            if (owner_results_ && dep.kind == DependencyKind::kData &&
                dep.storage.has_value() && producer_core == kCoreCube &&
                consumer_core == kCoreVector &&
                std::any_of(
                    owner_results_->begin(), owner_results_->end(),
                    [&](const OwnerResultPlan &plan) {
                      return plan.storage.same_as(dep.storage.value()) &&
                             plan.producer == producer &&
                             plan.consumer == consumer && plan.epoch == loop &&
                             plan.distance == distance;
                    })) {
              // A qualified RAW is materialized as a per-owner result. Keep
              // it out of the all-participant closure: it cannot discharge an
              // unrelated GM hazard or prove a collective allocation lifetime.
              continue;
            }
            if (owner_inputs_ && dep.kind == DependencyKind::kData &&
                dep.storage.has_value() && producer_core == kCoreVector &&
                consumer_core == kCoreCube &&
                std::any_of(
                    owner_inputs_->begin(), owner_inputs_->end(),
                    [&](const OwnerResultPlan &plan) {
                      return plan.storage.same_as(dep.storage.value()) &&
                             plan.producer == producer &&
                             plan.consumer == consumer && plan.epoch == loop &&
                             plan.distance == distance;
                    })) {
              // Only this exact per-owner RAW receives staggered collective
              // readiness. It contributes no all-participant completion edge;
              // ordinary WAR/reuse and unrelated buffer proofs stay intact.
              continue;
            }
            if (producer_core == kCoreCube && consumer_core == kCoreCube &&
                fix_unit_flag)
              continue;
            if (info != nullptr && info->UsesCounter() && !within_owner &&
                IsCounterDependencyCovered(*info, producer, producer_core,
                                           consumer, consumer_core, loop)) {
              continue;
            }
            size_t producer_site = site_registry_.FindSiteId(
                SyncInsertionSiteKey{dep.prod_node, producer, false,
                                     producer_core, pair_domain_id});
            size_t consumer_site = site_registry_.FindSiteId(
                SyncInsertionSiteKey{dep.cons_node, consumer, true,
                                     consumer_core, pair_domain_id});
            DepEdge edge(producer_site, consumer_site, distance, true);
            if (pool_local_release &&
                (site_registry_.Sites()[producer_site].path_end != producer ||
                 site_registry_.Sites()[consumer_site].path_end != consumer)) {
              // No wider lexical completion follows from a slot credit.
              continue;
            }
            if (counter_info != nullptr) {
              PrimExpr iteration = BufferLoad(counter_info->counter,
                                              {IntImm(DataType::Int(32), 0)});
              sync_points.emplace_back(std::move(edge), dep.kind, num_versions,
                                       iteration, counter_owner->loop,
                                       counter_info->counter_group_id);
            } else {
              sync_points.emplace_back(std::move(edge), dep.kind, num_versions,
                                       flag_loop);
            }
            sync_points.back().operand_pool_release = pool_local_release;
            SelectGmReadRelease(&sync_points.back(), dep, producer, consumer,
                                loop);
            SelectGmWriteAcquire(&sync_points.back(), dep, producer, consumer,
                                 loop);
          }
        }
      }
    }
    if (unit_boundaries_) {
      for (const UnitFlagBoundary &boundary : *unit_boundaries_) {
        if (boundary.scope != loop)
          continue;
        int domain = domains_.UnconditionalLexicalDomain(loop);
        size_t src = site_registry_.FindSiteId(
            SyncInsertionSiteKey{boundary.producer_node, boundary.producer,
                                 false, kCoreCube, domain});
        size_t dst = site_registry_.FindSiteId(
            SyncInsertionSiteKey{boundary.consumer_node, boundary.consumer,
                                 true, kCoreCube, domain});
        sync_points.emplace_back(DepEdge(src, dst, 0, true),
                                 DependencyKind::kData, 1, loop);
        sync_points.back().unit_boundary = boundary.role;
      }
    }
    return sync_points;
  }

  void AddPromotedChildClosure(DepClosure &out, IRStructure *child,
                               const DepClosure &child_closure,
                               ControlNode *projection_scope) const {
    if (child->IsControl() &&
        static_cast<ControlNode *>(child)->BodyHasLoopBreak()) {
      return;
    }

    for (const DepEdge &edge : child_closure.Edges()) {
      // A lexical domain ends at its loop. Re-express the same physical
      // endpoints in each compatible parent domain. A wider parent guard
      // cannot inherit this proof: an inactive child endpoint could otherwise
      // become a transitive bridge between unrelated parent-only accesses.
      site_registry_.ForEachParentSitePair(edge, [&](const DepEdge &parent) {
        if (site_registry_.Src(parent).path_end !=
                site_registry_.Src(edge).path_end ||
            site_registry_.Dst(parent).path_end !=
                site_registry_.Dst(edge).path_end) {
          return;
        }
        int source_domain = site_registry_.Src(edge).EpochDomainId();
        int parent_domain = site_registry_.Src(parent).EpochDomainId();
        if (!domains_.CanProjectDomain(source_domain, parent_domain,
                                       projection_scope)) {
          return;
        }
        SaturateDependencyEdge(out, parent,
                               /*max_distance=*/0, projection_scope);
      });
    }
  }

  void RecordLifetimeOrder(const std::vector<size_t> &task_sites,
                           ControlNode *scope, const DepClosure &closure) {
    if (disable_buffer_reuse_)
      return;
    IterationOrder order(site_registry_.Size());
    const auto &sites = site_registry_.Sites();
    for (const DepEdge &edge : closure.Edges()) {
      // DepClosure models only forward iteration distances. Keep the check
      // below to detect violations of this producer invariant.
      if (sites[edge.src].EpochDomainId() == sites[edge.dst].EpochDomainId())
        ICHECK(order.AddEdge(
            edge.src, edge.dst, edge.distance,
            edge.strict || HasImplicitCompletionOrder(edge.src, edge.dst)));
    }
    if (scope != nullptr) {
      // Keep the primitive same-pipe relation even when its distance exceeds
      // the flag optimizer's bounded closure. Shortest paths can then compose
      // arbitrarily large distances without enumerating intermediate epochs.
      for (size_t src : task_sites) {
        for (size_t dst : task_sites) {
          const auto &a = sites[src];
          const auto &b = sites[dst];
          if (a.core != b.core || a.pipe != b.pipe ||
              a.EpochDomainId() != b.EpochDomainId())
            continue;
          int64_t distance = int64_t(a.path_begin->GetStage()) -
                             b.path_begin->GetStage() + (src < dst ? 0 : 1);
          ICHECK(
              order.AddEdge(src, dst, std::max<int64_t>(distance, 0),
                            HasImplicitSamePipeOrder(a.pipe, architecture_)));
        }
      }
    }
    lifetime_orders_.insert_or_assign(scope, std::move(order));
  }

  void AddSamePipeEdges(DepClosure &closure,
                        const std::vector<size_t> &task_sites, int max_distance,
                        ControlNode *projection_scope) const {
    const SyncInsertionSiteList &sites = site_registry_.Sites();
    using PipeDomainKey = std::tuple<CoreMask, std::string, int>;
    std::map<PipeDomainKey, std::vector<size_t>> by_pipe;
    for (size_t site_id : task_sites) {
      const SyncInsertionSite &site = sites[site_id];
      by_pipe[{site.core, site.pipe, site.EpochDomainId()}].push_back(site_id);
    }

    auto level_stage = [](const SyncInsertionSite &site) {
      return site.path_begin->GetStage();
    };
    for (auto &[_, pipe_sites] : by_pipe) {
      // A single pipe issues sites in physical (iteration, position) order.
      // For a(k) before b(k+d), the minimum valid distance is
      // stage(a) - stage(b), plus one when a is physically after b.
      for (size_t lhs_id : pipe_sites) {
        for (size_t rhs_id : pipe_sites) {
          if (lhs_id == rhs_id)
            continue;
          const SyncInsertionSite &lhs = sites[lhs_id];
          const SyncInsertionSite &rhs = sites[rhs_id];
          bool lhs_before_rhs = lhs_id < rhs_id;
          int min_distance =
              level_stage(lhs) - level_stage(rhs) + (lhs_before_rhs ? 0 : 1);
          min_distance = std::max(min_distance, 0);
          for (int distance = min_distance; distance <= max_distance;
               ++distance) {
            SaturateDependencyEdge(closure,
                                   DepEdge(lhs_id, rhs_id, distance, false),
                                   max_distance, projection_scope);
          }
        }
      }
    }
  }

  void SaturateDependencyEdge(DepClosure &closure, const DepEdge &edge,
                              int max_distance,
                              ControlNode *projection_scope) const {
    closure.Saturate(
        edge, max_distance, [&](const DepEdge &candidate, const auto &visit) {
          if (candidate.distance != 0 && candidate.distance != 1) {
            visit(candidate);
            return;
          }
          int source_domain = site_registry_.Src(candidate).EpochDomainId();
          site_registry_.ForEachPhysicalSitePair(
              candidate, [&](const DepEdge &projected) {
                int target_domain =
                    site_registry_.Src(projected).EpochDomainId();
                // Same-iteration ordering projects into a narrower active
                // guard. Unit-distance ordering additionally requires both
                // domains to describe the same active-epoch sequence.
                bool can_project =
                    domains_.CanProjectDomain(source_domain, target_domain,
                                              projection_scope) &&
                    (candidate.distance == 0 ||
                     domains_.CanProjectDomain(target_domain, source_domain,
                                               projection_scope));
                if (can_project) {
                  visit(projected);
                }
              });
        });
  }

  bool UnshiftedBinaryLoop(ControlNode *loop, bool static_positive) const {
    if (!loop || !loop->control.defined() || loop->GetStage() != 0 ||
        loop->control->kind != ForKind::kSerial || loop->BodyHasLoopBreak())
      return false;
    const ForNode *node = loop->control.get();
    const auto *min = node->min.as<IntImmNode>();
    const auto *extent = node->extent.as<IntImmNode>();
    PrimExpr step = node->step.has_value()
                        ? node->step.value()
                        : IntImm(node->extent.dtype(), 1);
    const auto *constant_step = step.as<IntImmNode>();
    if (!min || min->value != 0 || !constant_step ||
        constant_step->value != 1 || !node->extent.dtype().is_int() ||
        SideEffect(node->extent) != CallEffectKind::kPure ||
        (static_positive && (!extent || extent->value <= 0)))
      return false;
    for (const auto &[key, value] : node->annotations) {
      if (key == "num_stages") {
        auto expression = value.as<PrimExpr>();
        const auto *stages = expression ? expression->as<IntImmNode>() : nullptr;
        if (!stages || stages->value != 1)
          return false;
      } else if (key == "enable_offset" || key == "tl.ascend_auto_pipeline") {
        auto expression = value.as<PrimExpr>();
        auto boolean = value.try_cast<bool>();
        if (!(expression ? is_zero(*expression)
                         : boolean.has_value() && !boolean.value()))
          return false;
      }
    }
    return true;
  }

  // A parent carried edge can be the last-to-first subset of a child's
  // persistent binary protocol. Prove the whole finite protocol, rather than
  // treating a child distance of one as one parent iteration by projection.
  bool CoversNestedLocalProtocol(const SyncPoint &wide,
                                const SyncPoint &narrow,
                                ControlNode *parent) const {
    if (wide.IsPrivateGmHandoff() || narrow.IsPrivateGmHandoff())
      return false;
    ControlNode *child = wide.flag_loop;
    auto persistent = [&](const SyncPoint &point) {
      return point.edge.strict && point.edge.distance == 1 &&
             point.num_versions == 1 && point.counter_group_id == 0 &&
             !point.local_binary_plan.has_value() &&
             point.unit_boundary == UnitFlagBoundaryRole::kNone &&
             !site_registry_.IsCrossCore(point) &&
             site_registry_.UsesFlag(point);
    };
    if (architecture_ != AscendArchitecture::kDav2201 ||
        !persistent(wide) || !persistent(narrow) ||
        narrow.flag_loop != parent || !child ||
        child->GetParentControl() != parent ||
        !UnshiftedBinaryLoop(parent, false) ||
        !UnshiftedBinaryLoop(child, true) ||
        !site_registry_.HasSameFlagKind(wide, narrow))
      return false;

    const auto &ws = site_registry_.Src(wide);
    const auto &wd = site_registry_.Dst(wide);
    const auto &ns = site_registry_.Src(narrow);
    const auto &nd = site_registry_.Dst(narrow);
    if (!ws.path_end->IsTask() || !wd.path_end->IsTask() ||
        ws.path_end != ns.path_end || wd.path_end != nd.path_end ||
        ws.at_beginning || ns.at_beginning || !wd.at_beginning ||
        !nd.at_beginning || ns.path_begin != child || nd.path_begin != child ||
        ws.path_begin->GetParent() != child ||
        wd.path_begin->GetParent() != child)
      return false;
    for (const auto &[site, owner] :
         {std::pair<const SyncInsertionSite *, ControlNode *>(&ws, child),
          {&wd, child}, {&ns, parent}, {&nd, parent}}) {
      const EpochDomain &domain = site_registry_.Domain(*site);
      if (domain.IsCounter() || domain.owner != owner)
        return false;
    }
    auto child_closure = scope_closures_.find(child);
    if (child_closure == scope_closures_.end() ||
        MinCompletionDistance(child_closure->second, wide.edge.dst,
                              wide.edge.src) != 0)
      return false;

    ConstrSet proof = parent->GetLoopBodyContext();
    arith::Analyzer outer;
    proof.Populate(outer);
    if (!outer.CanProve(narrow.flag_iteration ==
                        CalculateIterationCount(parent)) ||
        !outer.CanProve(wide.flag_iteration == CalculateIterationCount(child)))
      return false;
    PrimExpr activation = outer.Simplify(
        child->GetConditionGuard() && site_registry_.Domain(ws).active_guard);
    // A pure condition snapshot must expand to immutable enclosing loop
    // coordinates. Freshened mutable Bind reads and unknown input predicates
    // do not establish a finite active sequence for this certificate.
    std::unordered_set<const VarNode *> enclosing;
    for (ControlNode *loop = parent; loop; loop = loop->GetParentControl())
      enclosing.insert(loop->control->loop_var.get());
    bool stable_activation = true;
    PostOrderVisit(activation, [&](const ffi::ObjectRef &node) {
      if (const auto *var = node.as<VarNode>())
        stable_activation &= enclosing.count(var) != 0;
      if (node.as<BufferLoadNode>() || node.as<ProducerLoadNode>())
        stable_activation = false;
      if (const auto *call = node.as<CallNode>())
        stable_activation &=
            SideEffect(GetRef<PrimExpr>(call)) == CallEffectKind::kPure;
    });
    PrimExpr parent_trip = max(parent->control->extent,
                              make_zero(parent->control->extent.dtype()));
    PrimExpr parent_step = IntImm(parent->control->extent.dtype(), 1);
    if (!stable_activation ||
        !ContiguousLocalBinarySpan(activation, parent, parent_trip, parent_step))
      return false;

    std::unordered_set<ControlNode *> bound_loops;
    auto boundary_guard = [&](IRStructure *endpoint, bool at_beginning)
        -> std::optional<PrimExpr> {
      PrimExpr guard = Bool(true);
      for (IRStructure *node : endpoint->PathFrom(child)) {
        if (node->GetStage() != 0)
          return std::nullopt;
        for (const auto &wrapper : node->GetGuards()) {
          if (!wrapper->IsCondition() || node != child)
            return std::nullopt;
        }
        if (!node->IsControl())
          continue;
        auto *loop = static_cast<ControlNode *>(node);
        if (!UnshiftedBinaryLoop(loop, true))
          return std::nullopt;
        if (bound_loops.insert(loop).second)
          proof.AddConstr(loop->control->loop_var,
                          Range::FromMinExtent(loop->control->min,
                                               loop->control->extent));
        if (loop == child)
          continue;
        PrimExpr variable = loop->control->loop_var;
        guard = guard && (at_beginning
                             ? variable == loop->control->min
                             : variable + 1 >= loop->control->extent);
      }
      return guard;
    };
    auto source_boundary = boundary_guard(ws.path_end, false);
    auto destination_boundary = boundary_guard(wd.path_end, true);
    if (!source_boundary || !destination_boundary)
      return false;
    PrimExpr first = child->control->loop_var == child->control->min;
    PrimExpr last = child->control->loop_var + 1 >= child->control->extent;
    auto emitted_guard = [&](const SyncInsertionSite &site) {
      return child->GetConditionGuard() &&
             site_registry_.Domain(site).active_guard && site.GetGuard();
    };
    // Exactly one wait/set per child iteration, and exactly the first wait
    // and last set per parent activation. Extra guards cannot be discarded.
    return GuardsEquivalent(emitted_guard(ws), activation && *source_boundary,
                            proof) &&
           GuardsEquivalent(emitted_guard(wd),
                            activation && *destination_boundary, proof) &&
           GuardsEquivalent(emitted_guard(ns),
                            activation && last && *source_boundary, proof) &&
           GuardsEquivalent(emitted_guard(nd),
                            activation && first && *destination_boundary, proof);
  }

  void RemoveNestedCoveredLocalProtocols(
      std::vector<SyncPoint> &local_points,
      const std::vector<std::vector<VisibleSyncPoint>> &child_points,
      DepClosure &closure, int max_distance, ControlNode *parent) const {
    std::vector<SyncPoint> retained;
    retained.reserve(local_points.size());
    for (SyncPoint &narrow : local_points) {
      bool covered = false;
      for (const auto &points : child_points) {
        for (const VisibleSyncPoint &visible : points) {
          if (CoversNestedLocalProtocol(sync_points_[visible.sync_point_id],
                                        narrow, parent)) {
            covered = true;
            break;
          }
        }
        if (covered)
          break;
      }
      if (!covered) {
        retained.push_back(std::move(narrow));
        continue;
      }
      // The retained child flag is seeded/drained once at kernel boundaries.
      // Each wait clears it before the next real set (the child's validated
      // reverse-completion edge). Its final child token is consumed by the
      // first child in the next active parent iteration; inactive prefixes,
      // suffixes and an empty whole loop leave the initial/final token for
      // the one root drain. This complete finite certificate establishes the
      // parent last-to-first completion edge without emitting another flag,
      // and keeps that true relation available to alias/lifetime consumers.
      SaturateDependencyEdge(closure, narrow.edge, max_distance, parent);
    }
    local_points = std::move(retained);
  }

  bool CoversLocalProtocol(const SyncPoint &wide, const SyncPoint &narrow,
                           ControlNode *loop) const {
    if (wide.IsPrivateGmHandoff() || narrow.IsPrivateGmHandoff())
      return false;
    if (architecture_ != AscendArchitecture::kDav2201 || loop == nullptr ||
        wide.counter_group_id != 0 || narrow.counter_group_id != 0 ||
        wide.flag_loop != loop || narrow.flag_loop != loop ||
        wide.edge.distance <= 0 || wide.edge.distance != narrow.edge.distance ||
        site_registry_.IsCrossCore(wide) ||
        site_registry_.IsCrossCore(narrow) || !site_registry_.UsesFlag(wide) ||
        !site_registry_.HasSameFlagKind(wide, narrow)) {
      return false;
    }
    const SyncInsertionSite &wide_src = site_registry_.Src(wide);
    const SyncInsertionSite &wide_dst = site_registry_.Dst(wide);
    const SyncInsertionSite &narrow_src = site_registry_.Src(narrow);
    const SyncInsertionSite &narrow_dst = site_registry_.Dst(narrow);
    if (!wide_src.HasSamePhysicalPosition(narrow_src) ||
        !wide_dst.HasSamePhysicalPosition(narrow_dst)) {
      return false;
    }
    for (const SyncInsertionSite *site :
         {&wide_src, &wide_dst, &narrow_src, &narrow_dst}) {
      const EpochDomain &domain = site_registry_.Domain(*site);
      if (domain.IsCounter() || domain.owner != loop)
        return false;
    }

    // This runs before active-span normalization. Equal raw lexical clocks
    // and distances preserve src(e)->dst(e+d), even when one domain skips
    // leading/trailing epochs. Do not project a carried closure edge into a
    // different active sequence or share its binary IDs: discard the whole
    // covered protocol, including its eventual seed and drain instead.
    const ConstrSet &context = loop->GetLoopBodyContext();
    arith::Analyzer analyzer;
    context.Populate(analyzer);
    PrimExpr lexical_iteration = CalculateIterationCount(loop);
    if (!analyzer.CanProve(wide.flag_iteration == lexical_iteration) ||
        !analyzer.CanProve(narrow.flag_iteration == lexical_iteration)) {
      return false;
    }
    // The context contains only the enclosing loop facts, never either
    // endpoint's activation guard. Implication is proved separately at the
    // identical physical source and destination positions.
    return GuardImplies(narrow_src.GetGuard(), wide_src.GetGuard(), context) &&
           GuardImplies(narrow_dst.GetGuard(), wide_dst.GetGuard(), context);
  }

  void RemoveCoveredLocalProtocols(std::vector<SyncPoint> &sync_points,
                                   ControlNode *loop) const {
    std::vector<bool> covered(sync_points.size(), false);
    for (size_t narrow = 0; narrow < sync_points.size(); ++narrow) {
      for (size_t wide = 0; wide < sync_points.size(); ++wide) {
        if (wide == narrow || covered[wide] ||
            !CoversLocalProtocol(sync_points[wide], sync_points[narrow],
                                 loop)) {
          continue;
        }
        // Equivalent protocols keep their first representative. Strict guard
        // inclusion instead keeps the wider protocol regardless of ordering.
        if (wide < narrow || !CoversLocalProtocol(sync_points[narrow],
                                                  sync_points[wide], loop)) {
          covered[narrow] = true;
          break;
        }
      }
    }
    std::vector<SyncPoint> retained;
    retained.reserve(sync_points.size());
    for (size_t i = 0; i < sync_points.size(); ++i) {
      if (!covered[i])
        retained.push_back(std::move(sync_points[i]));
    }
    sync_points = std::move(retained);
  }

  void OptimizeSyncPoints(std::vector<SyncPoint> &sync_points,
                          DepClosure &closure, int max_distance,
                          ControlNode *projection_scope) const {
    RemoveCoveredLocalProtocols(sync_points, projection_scope);
    std::stable_sort(sync_points.begin(), sync_points.end());
    std::vector<SyncPoint> emit;
    for (SyncPoint &sync_point : sync_points) {
      if (sync_point.IsPrivateGmHandoff()) {
        // One endpoint moves inside the phase. Retain the event, but never
        // saturate its old wide edge or use it to remove another point.
        emit.push_back(std::move(sync_point));
        continue;
      }
      if (sync_point.operand_pool_release) {
        // This exact same-generation relation is implemented by the physical
        // pool. It can close a retained load-ready flag's binary lifetime;
        // no ordinary ring/separate seed is emitted for it.
        SaturateDependencyEdge(closure, sync_point.edge, max_distance,
                               projection_scope);
        continue;
      }
      // The real boundary gates jointly establish the closed stream and its
      // terminal flag lifetime. Retain their exact blocking Scalar cuts.
      if (sync_point.unit_boundary == UnitFlagBoundaryRole::kNone &&
          closure.Contains(sync_point.edge))
        continue;
      if (sync_point.dependency_kind == DependencyKind::kOwnerExclusion &&
          sync_point.edge.distance != 0 && sync_point.edge.distance != 1) {
        LOG(FATAL) << "InsertSync cannot materialize an uncovered multi-buffer "
                      "owner exclusion at logical distance "
                   << sync_point.edge.distance << " (paired distance "
                   << 1 - sync_point.edge.distance << ")";
      }
      SaturateDependencyEdge(closure, sync_point.edge, max_distance,
                             projection_scope);
      emit.push_back(std::move(sync_point));
    }
    sync_points = std::move(emit);
  }

  std::optional<PrimExpr>
  SingleIterationBinaryActivation(const SyncPoint &point,
                                  ControlNode *loop) const {
    const SyncInsertionSite &src = site_registry_.Src(point);
    const SyncInsertionSite &dst = site_registry_.Dst(point);
    const auto *producer = static_cast<const TaskNode *>(src.path_end);
    const auto *consumer = static_cast<const TaskNode *>(dst.path_end);
    auto expand = [&](const PrimExpr &expr, const TaskNode *task) {
      // Expand definitions, not the endpoint's assumed If predicate. Merging
      // both endpoint contexts could make mismatched guards vacuously equal.
      return owner_result_detail::Expand(
          expr, owner_result_detail::TaskBindings(task));
    };
    PrimExpr source = expand(loop->GetConditionGuard() && src.GetGuard() &&
                                 site_registry_.Domain(src).active_guard,
                             producer);
    PrimExpr destination = expand(loop->GetConditionGuard() && dst.GetGuard() &&
                                      site_registry_.Domain(dst).active_guard,
                                  consumer);
    std::unordered_set<const VarNode *> iteration_vars;
    for (ControlNode *scope = loop; scope; scope = scope->GetParentControl())
      iteration_vars.insert(scope->control->loop_var.get());
    auto immutable = [&](const PrimExpr &expr) {
      bool producer_load = false;
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        producer_load |= node.as<ProducerLoadNode>() != nullptr;
      });
      return !producer_load && owner_result_detail::Immutable(expr) &&
             !UsesVar(expr, [&](const VarNode *var) {
               return iteration_vars.count(var) == 0;
             });
    };
    const ConstrSet &context = loop->GetLoopBodyContext();
    if (!immutable(source) || !immutable(destination) ||
        !GuardImplies(source, destination, context) ||
        !GuardImplies(destination, source, context) ||
        !GuardImplies(source, expand(producer->GetConditionGuard(), producer),
                      context) ||
        !GuardImplies(destination,
                      expand(consumer->GetConditionGuard(), consumer), context))
      return std::nullopt;
    return source;
  }

  ControlNode *SingleIterationBinaryParent(const SyncPoint &point,
                                           ControlNode *loop,
                                           PrimExpr *activation) const {
    if (!loop || point.flag_loop != loop || point.edge.distance != 0 ||
        point.counter_group_id != 0 || point.local_binary_plan ||
        point.unit_boundary != UnitFlagBoundaryRole::kNone)
      return nullptr;
    ControlNode *parent = loop->GetParentControl();
    if (!parent)
      return nullptr;
    // A conditional singleton may emit zero or one matched pair. Its enclosing
    // clock still has static positive, unshifted, unguarded repetition. This
    // does not prove completion between successive invocations.
    for (ControlNode *scope = loop; scope; scope = scope->GetParentControl()) {
      if ((scope != loop && scope->HasConditions()) ||
          scope->BodyHasLoopBreak() || scope->GetStage() != 0)
        return nullptr;
      const ForNode *node = scope->control.get();
      PrimExpr step = node->step.has_value() ? node->step.value()
                                             : IntImm(node->extent.dtype(), 1);
      const auto *extent = node->extent.as<IntImmNode>();
      const auto *stride = step.as<IntImmNode>();
      if (!extent || !stride || extent->value <= 0 || stride->value <= 0 ||
          SideEffect(node->min) != CallEffectKind::kPure ||
          (scope == loop && extent->value != stride->value))
        return nullptr;
    }
    const SyncInsertionSite &src = site_registry_.Src(point);
    const SyncInsertionSite &dst = site_registry_.Dst(point);
    const ConstrSet &context = loop->GetLoopBodyContext();
    if (!src.path_end->IsTask() || !dst.path_end->IsTask() ||
        src.path_begin != src.path_end || dst.path_begin != dst.path_end ||
        src.path_begin->GetParent() != loop ||
        dst.path_begin->GetParent() != loop ||
        site_registry_.Domain(src).IsCounter() ||
        site_registry_.Domain(dst).IsCounter() ||
        site_registry_.Domain(src).owner != loop ||
        site_registry_.Domain(dst).owner != loop ||
        SideEffect(point.flag_iteration) != CallEffectKind::kPure ||
        !context.CanProve(point.flag_iteration ==
                          CalculateIterationCount(parent)))
      return nullptr;
    bool unconditional =
        !loop->HasConditions() && context.CanProve(src.GetGuard()) &&
        context.CanProve(dst.GetGuard()) &&
        context.CanProve(site_registry_.Domain(src).active_guard) &&
        context.CanProve(site_registry_.Domain(dst).active_guard);
    if (unconditional) {
      *activation = Bool(true);
    } else {
      auto matched = SingleIterationBinaryActivation(point, loop);
      if (!matched)
        return nullptr;
      *activation = *matched;
    }
    return parent;
  }

  void ResolveSingleIterationBinaryPoints(
      const std::vector<std::vector<VisibleSyncPoint>> &child_points,
      const DepClosure &closure, ControlNode *parent) {
    if (!parent)
      return;
    for (const auto &points : child_points) {
      for (const VisibleSyncPoint &visible : points) {
        SyncPoint &point = sync_points_[visible.sync_point_id];
        if (!point.single_iteration_binary_plan ||
            point.single_iteration_binary_plan->parent != parent ||
            point.single_iteration_binary_plan->validated)
          continue;
        int reverse = std::numeric_limits<int>::max();
        for (const DepEdge &edge : visible.projected_edges) {
          const SyncInsertionSite &src = site_registry_.Src(edge);
          const SyncInsertionSite &dst = site_registry_.Dst(edge);
          // A wider, real parent completion may protect this activation
          // subset. Query its unchanged physical leaf endpoints; do not
          // project the child's conditional forward edge into that closure.
          if (site_registry_.Domain(src).IsCounter() ||
              site_registry_.Domain(dst).IsCounter() ||
              site_registry_.Domain(src).owner != parent ||
              site_registry_.Domain(dst).owner != parent ||
              src.path_end != site_registry_.Src(point).path_end ||
              dst.path_end != site_registry_.Dst(point).path_end ||
              src.core != site_registry_.Src(point).core ||
              dst.core != site_registry_.Dst(point).core ||
              src.pipe != site_registry_.Src(point).pipe ||
              dst.pipe != site_registry_.Dst(point).pipe)
            continue;
          const auto *producer = static_cast<const TaskNode *>(src.path_end);
          const auto *consumer = static_cast<const TaskNode *>(dst.path_end);
          PrimExpr parent_guard =
              owner_result_detail::Expand(
                  site_registry_.Domain(src).active_guard && src.GetGuard(),
                  owner_result_detail::TaskBindings(producer)) &&
              owner_result_detail::Expand(
                  site_registry_.Domain(dst).active_guard && dst.GetGuard(),
                  owner_result_detail::TaskBindings(consumer));
          if (!GuardImplies(point.single_iteration_binary_plan->activation,
                            parent_guard,
                            point.flag_loop->GetLoopBodyContext()))
            continue;
          reverse = std::min(
              reverse, MinCompletionDistance(closure, edge.dst, edge.src));
        }
        if (reverse == std::numeric_limits<int>::max()) {
          if ((owner_results_ && !owner_results_->empty()) ||
              (owner_inputs_ && !owner_inputs_->empty())) {
            owner_fallback_required_ = true;
            return;
          }
          if (unit_streams_ && !unit_streams_->empty()) {
            LOG(WARNING) << "Unit stream fallback: single-iteration local flag "
                            "has no parent reverse completion; parent="
                         << parent->control->loop_var->name_hint;
            unit_fallback_required_ = true;
            return;
          }
        }
        ICHECK_NE(reverse, std::numeric_limits<int>::max())
            << "dav-2201 local binary flag: no parent reverse completion path "
               "for single-iteration "
            << site_registry_.Src(point).pipe << "->"
            << site_registry_.Dst(point).pipe;
        ICHECK_GT(reverse, 0)
            << "dav-2201 local binary flag: nonpositive parent reverse "
               "completion distance";
        // With d=0 and identical flattened clocks, the r-wide parent ring
        // remains sufficient when matched activations skip notifications.
        // Storage width or a single child trip cannot replace this proof.
        point.num_versions = reverse;
        point.single_iteration_binary_plan->validated = true;
      }
    }
  }

  void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                            const DepClosure &closure, ControlNode *loop) {
    for (SyncPoint &sync_point : sync_points) {
      if (sync_point.IsPrivateGmHandoff())
        continue; // Preserve the original paired clock, seeds and drains.
      const SyncInsertionSite &src = site_registry_.Src(sync_point);
      const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
      if (src.pipe == dst.pipe)
        continue;
      bool is_innermost = loop == sync_point.flag_loop;
      int innermost_num_versions = is_innermost ? sync_point.num_versions : 1;
      bool local_binary = architecture_ == AscendArchitecture::kDav2201 &&
                          !site_registry_.IsCrossCore(sync_point);
      if (sync_point.unit_boundary == UnitFlagBoundaryRole::kDrain) {
        ICHECK(local_binary && src.pipe == "FIX" && dst.pipe == "M" &&
               sync_point.edge.distance == 0 && sync_point.num_versions == 1 &&
               sync_point.counter_group_id == 0 && is_innermost);
        // This is a flag-bit lifetime proof, not a C completion edge. The
        // drain's M wait clears FIX_M before the following M_S notification;
        // its PIPE_S wait blocks dispatch before the next activation can
        // issue another FIX_M notification. The complete boundary bracket
        // is constructed together and retained above. Give this point its
        // own ID; never use unit protection to manufacture a reverse whole-
        // MAD/FIX completion relation in DepClosure.
        continue;
      }
      int min_distance =
          local_binary
              ? MinCompletionDistance(closure, sync_point.edge.dst,
                                      sync_point.edge.src)
              : closure.MinDistance(sync_point.edge.dst, sync_point.edge.src);
      auto binary_context = [&]() {
        std::ostringstream detail;
        detail << " [pipe=" << src.pipe << "->" << dst.pipe
               << ", distance=" << sync_point.edge.distance
               << ", reverse_completion="
               << (min_distance == std::numeric_limits<int>::max()
                       ? "unproved"
                       : std::to_string(min_distance))
               << ", width=" << sync_point.num_versions << ", analysis_loop="
               << (loop ? loop->control->loop_var->name_hint : String("<none>"))
               << ", flag_loop="
               << (sync_point.flag_loop
                       ? sync_point.flag_loop->control->loop_var->name_hint
                       : String("<none>"))
               << ", clock=" << sync_point.flag_iteration
               << ", counter_group=" << sync_point.counter_group_id
               << ", source=" << src.path_begin->GetIndex() << ":"
               << src.path_end->GetIndex()
               << ", destination=" << dst.path_begin->GetIndex() << ":"
               << dst.path_end->GetIndex()
               << ", source_guard=" << src.GetGuard()
               << ", destination_guard=" << dst.GetGuard()
               << ", source_domain=" << src.EpochDomainId() << ":"
               << site_registry_.Domain(src).active_guard
               << ", destination_domain=" << dst.EpochDomainId() << ":"
               << site_registry_.Domain(dst).active_guard << "]";
        return detail.str();
      };

      if (local_binary && min_distance == std::numeric_limits<int>::max()) {
        PrimExpr activation;
        if (ControlNode *parent =
                SingleIterationBinaryParent(sync_point, loop, &activation)) {
          sync_point.single_iteration_binary_plan =
              SingleIterationBinaryPlan{parent, std::move(activation)};
          continue;
        }
      }

      if (local_binary && min_distance == std::numeric_limits<int>::max() &&
          ((owner_results_ && !owner_results_->empty()) ||
           (owner_inputs_ && !owner_inputs_->empty()))) {
        // The old pair-ready may have supplied this unrelated ring's only
        // reverse-completion witness. Do not use an owner edge as its proof
        // or weaken binary validation; retry the normal collective analysis.
        owner_fallback_required_ = true;
        return;
      }
      if (local_binary && min_distance == std::numeric_limits<int>::max() &&
          unit_streams_ && !unit_streams_->empty()) {
        // Unit protection cannot supply an unrelated flag's reverse whole-
        // completion witness. Retry the untouched ordinary program instead
        // of relaxing that phase contract or keeping half of the selection.
        LOG(WARNING) << "Unit stream fallback due to unrelated local flag "
                        "missing reverse completion"
                     << binary_context();
        unit_fallback_required_ = true;
        return;
      }

      if (local_binary && sync_point.edge.distance > 0) {
        // Weak MTE/FIX issue edges alone do not establish that a previous
        // WaitFlag has cleared the bit before its next SetFlag completes.
        int reverse_completion = min_distance;
        int distance = sync_point.edge.distance;
        // Preserve the existing persistent protocol when every ring slot is
        // an initial token and the consumer completes before the next set.
        if (reverse_completion == 0 && distance == innermost_num_versions)
          continue;

        // A shifted lexical phase has one root clock, with at most one
        // contiguous active interval. Re-seeding an inner loop at each entry
        // is unsafe: a local drain only blocks its destination pipe, not the
        // next source seed.
        bool shifted_scope =
            is_innermost && sync_point.counter_group_id == 0 &&
            loop != nullptr && loop->GetParentControl() == nullptr &&
            !loop->HasConditions() && !loop->BodyHasLoopBreak();
        if (GmHandoffNeedsFallback(shifted_scope))
          return;
        ICHECK(shifted_scope)
            << "dav-2201 local binary flag: shifted carried events require "
               "an unconditional outermost lexical loop"
            << binary_context();
        const ForNode *for_node = loop->control.get();
        PrimExpr step = for_node->step.has_value()
                            ? for_node->step.value()
                            : IntImm(for_node->extent.dtype(), 1);
        bool positive_step =
            loop->task->outer_ctx.CanProve(step > make_zero(step.dtype()));
        if (GmHandoffNeedsFallback(positive_step))
          return;
        ICHECK(positive_step)
            << "dav-2201 local binary flag: loop step must be positive";
        bool immutable_bounds =
            SideEffect(for_node->min) == CallEffectKind::kPure &&
            SideEffect(for_node->extent) == CallEffectKind::kPure &&
            SideEffect(step) == CallEffectKind::kPure;
        if (GmHandoffNeedsFallback(immutable_bounds))
          return;
        ICHECK(immutable_bounds)
            << "dav-2201 local binary flag: loop bounds must be immutable";
        PrimExpr trip_count = max(ceildiv(for_node->extent, step),
                                  make_zero(for_node->extent.dtype()));
        bool signed_clock = sync_point.flag_iteration.dtype().is_int() &&
                            trip_count.dtype().is_int();
        if (GmHandoffNeedsFallback(signed_clock))
          return;
        ICHECK(signed_clock)
            << "dav-2201 local binary flag: shifted epoch clocks require "
               "signed integer iteration and trip-count types";
        PrimExpr lexical_ordinal =
            indexdiv(for_node->loop_var - for_node->min, step);
        bool lexical_clock = loop->GetLoopBodyContext().CanProve(
            sync_point.flag_iteration == lexical_ordinal);
        if (GmHandoffNeedsFallback(lexical_clock))
          return;
        ICHECK(lexical_clock)
            << "dav-2201 local binary flag: shifted epoch clock must equal "
               "the root lexical iteration ordinal"
            << binary_context();
        std::optional<LocalBinaryEpochSpan> active_span =
            LocalBinaryActiveSpan(sync_point, loop, trip_count, step);
        if (GmHandoffNeedsFallback(active_span.has_value()))
          return;
        ICHECK(active_span.has_value())
            << "dav-2201 local binary flag: shifted carried events require "
               "one exact contiguous active-epoch interval with static cut "
               "points"
            << binary_context();
        sync_point.flag_iteration = lexical_ordinal - active_span->first;
        trip_count = active_span->trip_count;
        if (reverse_completion == std::numeric_limits<int>::max()) {
          std::optional<int> finite_width =
              FiniteLocalBinaryWidth(loop, trip_count, distance);
          if (GmHandoffNeedsFallback(finite_width.has_value()))
            return;
          ICHECK(finite_width.has_value())
              << "dav-2201 local binary flag: no reverse completion path for "
              << src.pipe << "->" << dst.pipe
              << "; the whole lexical clock also exceeds the finite "
                 "non-reusing event capacity";
          sync_point.num_versions = *finite_width;
        } else {
          if (GmHandoffNeedsFallback(
                  reverse_completion <=
                  std::numeric_limits<int>::max() - distance))
            return;
          ICHECK_LE(reverse_completion,
                    std::numeric_limits<int>::max() - distance);
          sync_point.num_versions = distance + reverse_completion;
        }
        sync_point.local_binary_plan =
            LocalBinaryFlagPlan{distance, trip_count};
        continue;
      }

      if (local_binary && sync_point.edge.distance == 0) {
        // A finite root lexical clock can reserve one ID per producer epoch,
        // without a reverse dependency. One inner trip-count bound does not
        // establish this for a repeated outer owner or a mutable counter.
        if (min_distance == std::numeric_limits<int>::max() && is_innermost &&
            sync_point.counter_group_id == 0 && loop != nullptr &&
            loop->GetParentControl() == nullptr) {
          const ForNode *for_node = loop->control.get();
          PrimExpr step = for_node->step.has_value()
                              ? for_node->step.value()
                              : IntImm(for_node->extent.dtype(), 1);
          bool whole_clock =
              SideEffect(for_node->min) == CallEffectKind::kPure &&
              SideEffect(for_node->extent) == CallEffectKind::kPure &&
              SideEffect(step) == CallEffectKind::kPure &&
              loop->task->outer_ctx.CanProve(step > make_zero(step.dtype()));
          if (whole_clock) {
            PrimExpr trip_count = max(ceildiv(for_node->extent, step),
                                      make_zero(for_node->extent.dtype()));
            if (std::optional<int> width =
                    FiniteLocalBinaryWidth(loop, trip_count, 0)) {
              sync_point.num_versions = *width;
              sync_point.local_binary_plan = LocalBinaryFlagPlan{0, trip_count};
              continue;
            }
          }
        }
        ICHECK_NE(min_distance, std::numeric_limits<int>::max())
            << "dav-2201 local binary flag: no reverse completion path for "
            << src.pipe << "->" << dst.pipe << binary_context();
        ICHECK_GT(min_distance, 0)
            << "dav-2201 local binary flag: nonpositive reverse completion "
               "distance for "
            << src.pipe << "->" << dst.pipe;
        if (is_innermost) {
          sync_point.num_versions = min_distance;
        } else {
          // The owner clock does not advance within a descendant loop. Its
          // repeated uses of one ID need completion before the next child
          // iteration, irrespective of the owner's physical storage width.
          if (GmHandoffNeedsFallback(min_distance <= 1))
            return;
          ICHECK_LE(min_distance, 1)
              << "dav-2201 local binary flag: reverse completion is too loose "
                 "for a descendant loop sharing its owner's clock";
        }
        continue;
      }

      // If the loop trip count is no larger than the ring, no version is
      // reused.
      if (loop != nullptr) {
        const ForNode *inner = loop->control.get();
        PrimExpr step = inner->step.has_value()
                            ? inner->step.value()
                            : IntImm(inner->extent.dtype(), 1);
        // Bound only this loop's trip count; outer context may help prove its
        // symbolic extent without flattening any enclosing iterations.
        if (loop->task->outer_ctx.CanProve(
                inner->extent <=
                IntImm(inner->extent.dtype(), innermost_num_versions) * step)) {
          continue;
        }
      }
      if (sync_point.edge.distance > 0) {
        if (sync_point.edge.distance != innermost_num_versions ||
            min_distance > 0) {
          LOG(WARNING) << "cross-iteration flag " << src.pipe << "->"
                       << dst.pipe
                       << ": the synchronization ring is malformed.";
        }
        continue;
      }
      if (min_distance > innermost_num_versions) {
        LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                     << ": reverse ordering too loose to bound flag reuse.";
        continue;
      }
      if (min_distance <= 0) {
        LOG(WARNING) << "flag " << src.pipe << "->" << dst.pipe
                     << ": dependency cycle (reverse distance <= 0).";
        continue;
      }
      if (is_innermost)
        sync_point.num_versions = min_distance;
    }
  }

  bool CanReuse(const VisibleSyncPoint &lhs, const VisibleSyncPoint &rhs,
                const DepClosure &closure, bool allow_cross_iter,
                bool allow_mixed_iter) const {
    const SyncPoint &lhs_sync = sync_points_[lhs.sync_point_id];
    const SyncPoint &rhs_sync = sync_points_[rhs.sync_point_id];
    if (lhs_sync.IsPrivateGmHandoff() || rhs_sync.IsPrivateGmHandoff())
      return false;
    if (lhs_sync.unit_boundary == UnitFlagBoundaryRole::kDrain ||
        rhs_sync.unit_boundary == UnitFlagBoundaryRole::kDrain)
      return false;
    // Existing reuse proofs compare equal event phases. Keep a shifted local
    // binary channel exclusive rather than sharing its partial seed/drain.
    if (lhs_sync.local_binary_plan || rhs_sync.local_binary_plan ||
        lhs_sync.single_iteration_binary_plan ||
        rhs_sync.single_iteration_binary_plan)
      return false;
    if (!site_registry_.UsesFlag(lhs_sync) ||
        !site_registry_.UsesFlag(rhs_sync) ||
        !site_registry_.HasCompatibleFlagKind(lhs_sync, rhs_sync)) {
      return false;
    }

    auto by_domain = [&](const VisibleSyncPoint &visible) {
      std::map<int, DepEdge> result;
      for (const DepEdge &edge : visible.projected_edges) {
        int src_domain = site_registry_.Src(edge).EpochDomainId();
        int dst_domain = site_registry_.Dst(edge).EpochDomainId();
        ICHECK_EQ(src_domain, dst_domain);
        auto [_, inserted] = result.emplace(src_domain, edge);
        ICHECK(inserted)
            << "A visible synchronization point has duplicate projections in "
               "one epoch domain";
      }
      return result;
    };
    std::map<int, DepEdge> lhs_edges = by_domain(lhs);
    std::map<int, DepEdge> rhs_edges = by_domain(rhs);
    if (lhs_edges.empty() || lhs_edges.size() != rhs_edges.size())
      return false;

    bool lhs_cross_iter = lhs_sync.edge.distance > 0;
    bool rhs_cross_iter = rhs_sync.edge.distance > 0;
    if (lhs_cross_iter != rhs_cross_iter) {
      // A mixed same/cross-iteration handshake alternates on one flag ring.
      if (!allow_mixed_iter || !site_registry_.IsCrossCore(lhs_sync) ||
          !site_registry_.IsCrossCore(rhs_sync) ||
          lhs_sync.num_versions != rhs_sync.num_versions ||
          lhs_sync.flag_loop != rhs_sync.flag_loop) {
        return false;
      }
      for (const auto &[domain_id, lhs_edge] : lhs_edges) {
        auto rhs_it = rhs_edges.find(domain_id);
        if (rhs_it == rhs_edges.end())
          return false;
        const DepEdge &rhs_edge = rhs_it->second;
        const DepEdge &cross_edge = lhs_cross_iter ? lhs_edge : rhs_edge;
        const DepEdge &same_edge = lhs_cross_iter ? rhs_edge : lhs_edge;
        if (!closure.HasEdge(cross_edge.dst, same_edge.src, 0) ||
            !closure.HasEdge(same_edge.dst, cross_edge.src, 0)) {
          return false;
        }
      }
      return true;
    }

    std::optional<bool> lhs_before;
    for (const auto &[domain_id, lhs_edge] : lhs_edges) {
      auto rhs_it = rhs_edges.find(domain_id);
      if (rhs_it == rhs_edges.end())
        return false;
      const DepEdge &rhs_edge = rhs_it->second;
      bool current_lhs_before;
      bool current_rhs_before;
      if (!lhs_cross_iter) {
        current_lhs_before = closure.HasEdge(lhs_edge.dst, rhs_edge.src, 0) &&
                             closure.HasEdge(rhs_edge.dst, lhs_edge.src, 1);
        current_rhs_before = closure.HasEdge(rhs_edge.dst, lhs_edge.src, 0) &&
                             closure.HasEdge(lhs_edge.dst, rhs_edge.src, 1);
      } else {
        if (!allow_cross_iter)
          return false;
        current_lhs_before = closure.HasEdge(lhs_edge.src, rhs_edge.dst, 0) &&
                             closure.HasEdge(rhs_edge.src, lhs_edge.dst, 1);
        current_rhs_before = closure.HasEdge(rhs_edge.src, lhs_edge.dst, 0) &&
                             closure.HasEdge(lhs_edge.src, rhs_edge.dst, 1);
      }
      if (current_lhs_before == current_rhs_before)
        return false;
      if (lhs_before.has_value() && lhs_before.value() != current_lhs_before) {
        return false;
      }
      lhs_before = current_lhs_before;
    }
    return lhs_before.has_value();
  }

  bool PromoteReuseSites(VisibleSyncPoint &visible) const {
    const SyncPoint &sync_point = sync_points_[visible.sync_point_id];
    bool cross_iter = sync_point.edge.distance > 0;
    std::set<DepEdge> promoted;
    for (const DepEdge &edge : visible.projected_edges) {
      site_registry_.ForEachParentSitePair(edge, [&](const DepEdge &parent) {
        size_t src = parent.src;
        size_t dst = parent.dst;
        if (site_registry_.Src(parent).path_end !=
            site_registry_.Src(edge).path_end) {
          src = site_registry_.FindBoundarySite(src, !cross_iter);
        }
        if (site_registry_.Dst(parent).path_end !=
            site_registry_.Dst(edge).path_end) {
          dst = site_registry_.FindBoundarySite(dst, cross_iter);
        }
        promoted.emplace(src, dst, edge.distance, edge.strict);
      });
    }
    if (promoted.empty())
      return false;
    visible.projected_edges.assign(promoted.begin(), promoted.end());
    return true;
  }

  void RecordReusablePairs(const std::vector<VisibleSyncPoint> &lhs_points,
                           const std::vector<VisibleSyncPoint> &rhs_points,
                           const DepClosure &closure, bool allow_cross_iter,
                           bool allow_mixed_iter) {
    for (const VisibleSyncPoint &lhs_visible : lhs_points) {
      for (const VisibleSyncPoint &rhs_visible : rhs_points) {
        size_t lhs = lhs_visible.sync_point_id;
        size_t rhs = rhs_visible.sync_point_id;
        if (lhs == rhs || !CanReuse(lhs_visible, rhs_visible, closure,
                                    allow_cross_iter, allow_mixed_iter)) {
          continue;
        }
        reusable_pairs_.emplace(std::min(lhs, rhs), std::max(lhs, rhs));
      }
    }
  }

  L0StorageGroups storage_groups_;
  const MultiBufferPlan &multi_buffer_plan_;
  const EpochDomainRegistry &domains_;
  AscendArchitecture architecture_;
  BufferVersionMap manual_;
  MultiBufferOwnerMap multi_buffer_owners_;
  ConflictHintList root_conflicts_;
  const std::vector<OwnerResultPlan> *owner_results_;
  const std::vector<OwnerResultPlan> *owner_inputs_;
  const std::vector<UnitFlagStreamPlan> *unit_streams_;
  const std::vector<UnitFlagBoundary> *unit_boundaries_;
  const std::vector<L0OperandSlotPool> *operand_pools_;
  const bool enable_gm_read_release_;
  std::vector<GmReadReleaseCut> gm_read_releases_;
  const bool enable_gm_write_acquire_;
  std::vector<GmWriteAcquireCut> gm_write_acquires_;
  const bool closed_c_byte_retirement_;
  SyncSiteRegistry site_registry_;
  const bool trace_c_lifetimes_;
  mutable std::unordered_set<size_t> trace_c_sites_;
  std::vector<SyncPoint> sync_points_;
  std::set<std::pair<size_t, size_t>> reusable_pairs_;
  DependencyCache dependency_cache_;
  std::map<ControlNode *, DepClosure> scope_closures_;
  std::map<ControlNode *, IterationOrder> lifetime_orders_;
  std::unordered_map<ControlNode *, std::map<CounterSyncSignature, int>>
      counter_channel_occurrences_;
  BufferAliasMap buffer_aliases_;
  bool disable_buffer_reuse_{false};
  bool owner_fallback_required_{false};
  bool unit_fallback_required_{false};
  bool operand_pool_fallback_required_{false};
};

// ============================================================================
// Flag data model, explicit reservation, and physical allocation
// ============================================================================
struct FlagInfo {
  CoreMask core_mask;
  PrimExpr flag_var;
  std::string hardevent;
  int mode;
  // Local binary rings may seed and drain different producer-epoch residues.
  PrimExpr drain_flag_var = PrimExpr();
};

using FlagInfoList = std::vector<FlagInfo>;

PrimExpr MakeFlagEventId(const PrimExpr &iteration, int num_versions,
                         int base_id);

struct CrossCoreFlagReservation {
  // Inclusive ranges of flag IDs used explicitly by the source kernel, for
  // every mode. Dynamic expressions contribute the ConstIntBound inferred
  // from their enclosing loops.
  std::vector<std::pair<int64_t, int64_t>> explicit_id_ranges;
  bool has_unbounded_id{false};
};

struct FlagAllocation {
  std::vector<size_t> sync_points;
  int num_versions;
  bool cross_iter;
  int base_id;

  FlagAllocation(size_t sync_point, int num_versions, bool cross_iter,
                 int base_id)
      : sync_points{sync_point}, num_versions(num_versions),
        cross_iter(cross_iter), base_id(base_id) {}
};

// Collect every CrossCore flag range used explicitly by the source kernel.
// Track enclosing loops, flat Bind declarations, branch guards, and
// assumptions so ConstIntBound can resolve dynamic flag expressions. Mode
// semantics are user-controlled; allocation conservatively reserves the ID
// range regardless of mode.
class ExplicitCrossCoreFlagCollector : public TaskAwareConstrVisitor {
public:
  explicit ExplicitCrossCoreFlagCollector(const CrossCoreSyncContract &contract)
      : contract_(contract) {}

  CrossCoreFlagReservation Collect(const Stmt &stmt) {
    operator()(stmt);
    return std::move(reservation_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3)
          << "ascend_cross_core_set_flag/wait_flag expects exactly 3 "
             "arguments";

      if (contract_.collective) {
        const auto *mode = op->args[0].as<IntImmNode>();
        ICHECK(mode && mode->value >= 0 && mode->value <= 2)
            << "dav-2201 cross-core synchronization supports modes 0, 1, 2; "
               "mode 4 and independent per-AIV Cube counters require dav-3510";
      }
      const PrimExpr &flag_id = op->args[2];
      arith::Analyzer analyzer;
      GetConstrSet().Populate(analyzer);
      arith::ConstIntBound bound = analyzer.const_int_bound(flag_id);
      if (contract_.collective) {
        ICHECK_GE(bound->min_value, 0)
            << "dav-2201 cross-core flag IDs must be provably in [0, 15]";
        ICHECK_LE(bound->max_value, 15)
            << "dav-2201 cross-core flag IDs must be provably in [0, 15]; "
               "per-AIV flag + 16 channels require dav-3510";
      }
      arith::Analyzer unconstrained_analyzer;
      Var unconstrained_flag_id("unconstrained_cross_core_flag_id",
                                flag_id.dtype());
      arith::ConstIntBound unconstrained_bound =
          unconstrained_analyzer.const_int_bound(unconstrained_flag_id);
      if (bound->min_value == arith::ConstIntBound::kNegInf ||
          bound->max_value == arith::ConstIntBound::kPosInf ||
          (bound->min_value == unconstrained_bound->min_value &&
           bound->max_value == unconstrained_bound->max_value)) {
        reservation_.has_unbounded_id = true;
      } else {
        ICHECK_LE(bound->min_value, bound->max_value)
            << "Ascend AutoSchedule inferred an invalid explicit cross-core "
               "flag range ["
            << bound->min_value << ", " << bound->max_value << "]";
        reservation_.explicit_id_ranges.emplace_back(bound->min_value,
                                                     bound->max_value);
      }
    }
    ConstrVisitor::VisitExpr_(op);
  }

  const CrossCoreSyncContract &contract_;
  CrossCoreFlagReservation reservation_;
};

CrossCoreFlagReservation
CollectExplicitCrossCoreFlags(const Stmt &kernel_body,
                              const CrossCoreSyncContract &contract) {
  ExplicitCrossCoreFlagCollector collector(contract);
  return collector.Collect(kernel_body);
}

class FlagAllocator {
public:
  FlagAllocator(const SyncSiteRegistry &site_registry,
                std::vector<SyncPoint> &sync_points,
                const std::set<std::pair<size_t, size_t>> &reusable_pairs,
                const CrossCoreFlagReservation &cross_core_flag_reservation,
                const CrossCoreSyncContract &cross_core_contract,
                const std::vector<L0OperandSlotPool> &operand_pools)
      : site_registry_(site_registry), sync_points_(sync_points),
        reusable_pairs_(reusable_pairs),
        cross_core_flag_reservation_(cross_core_flag_reservation),
        cross_core_contract_(cross_core_contract), operand_pools_(operand_pools) {}

  bool Plan(bool allow_fallback = false,
            bool require_intra_core_capacity = false) {
    allocations_.clear();
    flag_infos_.clear();
    GroupCounterChannels();

    std::map<std::string, int> required_ids;
    std::map<std::string, int> next_intra_core_id;
    operand_pool_bases_.clear();
    // Physical pools span the kernel. Their credits are exclusive and never
    // borrow an ordinary logical ring's binary IDs or completion closure.
    for (const auto &pool : operand_pools_) {
      operand_pool_bases_.push_back(next_intra_core_id["M_MTE1"]);
      next_intra_core_id["M_MTE1"] += pool.slots;
      required_ids["M_MTE1"] += pool.slots;
    }
    bool needs_cross_core_flag = false;
    for (size_t index = 0; index < sync_points_.size(); ++index) {
      const SyncPoint &sync_point = sync_points_[index];
      if (!site_registry_.UsesFlag(sync_point))
        continue;
      if (sync_point.counter_channel.has_value() &&
          counter_channel_representative_[index] != index) {
        continue;
      }
      std::string domain = FlagDomain(sync_point);
      required_ids[domain] += sync_point.num_versions;
      needs_cross_core_flag |= site_registry_.IsCrossCore(sync_point);
    }

    ICHECK(!needs_cross_core_flag ||
           !cross_core_flag_reservation_.has_unbounded_id)
        << "Ascend AutoSchedule cannot safely allocate cross-core flag IDs "
           "when an explicit CrossCoreSetFlag/WaitFlag uses a flag_id whose "
           "integer range cannot be bounded; constrain the flag_id with a "
           "constant-range loop or disable AutoSchedule";

    // All cross-core modes share 16 base slots. Only dav-3510 mode 4 has
    // separate Cube-side channels addressed by slot + 16 * AIV lane.
    std::array<bool, kCrossCoreFlagLimit> cross_core_slots{};
    for (const auto &[lo, hi] :
         cross_core_flag_reservation_.explicit_id_ranges) {
      ICHECK_LE(lo, hi);
      uint64_t span = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
      if (span >= static_cast<uint64_t>(kCrossCoreFlagLimit - 1)) {
        cross_core_slots.fill(true);
        continue;
      }
      for (uint64_t offset = 0; offset <= span; ++offset) {
        int64_t explicit_id = lo + static_cast<int64_t>(offset);
        int normalized_id = static_cast<int>(explicit_id % kCrossCoreFlagLimit);
        if (normalized_id < 0)
          normalized_id += kCrossCoreFlagLimit;
        cross_core_slots[normalized_id] = true;
      }
    }

    int explicit_cross_core_slots =
        std::count(cross_core_slots.begin(), cross_core_slots.end(), true);
    auto find_cross_core_range = [&](int width) {
      for (int base = 0; base + width <= kCrossCoreFlagLimit; ++base) {
        bool available = true;
        for (int offset = 0; offset < width; ++offset)
          available &= !cross_core_slots[base + offset];
        if (available)
          return base;
      }
      return -1;
    };

    for (bool cross_iter : {true, false}) {
      std::vector<size_t> candidates;
      for (size_t i = 0; i < sync_points_.size(); ++i) {
        if (site_registry_.UsesFlag(sync_points_[i]) &&
            (sync_points_[i].edge.distance > 0) == cross_iter) {
          candidates.push_back(i);
        }
      }
      std::stable_sort(candidates.begin(), candidates.end(),
                       [&](size_t lhs, size_t rhs) {
                         return sync_points_[lhs].num_versions >
                                sync_points_[rhs].num_versions;
                       });

      for (size_t index : candidates) {
        SyncPoint &sync_point = sync_points_[index];
        if (sync_point.counter_channel.has_value()) {
          size_t representative = counter_channel_representative_[index];
          if (representative != index) {
            size_t allocation_index =
                sync_points_[representative].allocation_index;
            ICHECK_LT(allocation_index, allocations_.size())
                << "Counter channel representative was not allocated first";
            allocations_[allocation_index].sync_points.push_back(index);
            sync_point.allocation_index = allocation_index;
            continue;
          }
        }
        bool compatible = false;
        std::string domain = FlagDomain(sync_point);
        bool cross_core = site_registry_.IsCrossCore(sync_point);
        int limit = cross_core ? kCrossCoreFlagLimit : kIntraCoreFlagLimit;
        int cross_core_base =
            cross_core ? find_cross_core_range(sync_point.num_versions) : -1;
        bool needs_reuse =
            cross_core ? required_ids[domain] + explicit_cross_core_slots >
                                 kCrossCoreFlagLimit ||
                             cross_core_base < 0
                       : required_ids[domain] > limit;

        // CanReuse proves the lifetime of individual sync points. This proof
        // covers the whole counter channel only when it has one owner.
        if (needs_reuse && !IsMultiOwnerCounterChannel(index)) {
          for (size_t allocation_index = 0;
               allocation_index < allocations_.size(); ++allocation_index) {
            FlagAllocation &allocation = allocations_[allocation_index];
            const SyncPoint &representative =
                sync_points_[allocation.sync_points.front()];
            if (!site_registry_.HasCompatibleFlagKind(sync_point,
                                                      representative)) {
              continue;
            }
            if (allocation.cross_iter != cross_iter &&
                (!cross_core || !site_registry_.IsCrossCore(representative))) {
              continue;
            }

            compatible = true;
            for (size_t member : allocation.sync_points) {
              if (IsMultiOwnerCounterChannel(member) ||
                  !reusable_pairs_.count(
                      {std::min(index, member), std::max(index, member)})) {
                compatible = false;
                break;
              }
            }
            if (compatible) {
              ICHECK_GE(allocation.num_versions, sync_point.num_versions);
              required_ids[domain] -= sync_point.num_versions;
              allocation.sync_points.push_back(index);
              sync_point.allocation_index = allocation_index;
              break;
            }
          }
        }
        if (compatible)
          continue;

        sync_point.allocation_index = allocations_.size();
        int base_id = 0;
        if (cross_core) {
          if (cross_core_base < 0 && allow_fallback)
            return false;
          ICHECK_GE(cross_core_base, 0)
              << "Ascend AutoSchedule cannot allocate "
              << sync_point.num_versions
              << " contiguous cross-core flag slots without overlapping "
                 "explicit cross-core flags or safely reusable auto-scheduled "
                 "flags";
          base_id = cross_core_base;
          for (int offset = 0; offset < sync_point.num_versions; ++offset)
            cross_core_slots[base_id + offset] = true;
        } else {
          base_id = next_intra_core_id[domain];
        }
        allocations_.emplace_back(index, sync_point.num_versions, cross_iter,
                                  base_id);
        if (!cross_core)
          next_intra_core_id[domain] += sync_point.num_versions;
      }
    }

    for (const auto &[domain, total] : next_intra_core_id) {
      if (total <= kIntraCoreFlagLimit)
        continue;
      // TVM_LOG_DEBUG controls this diagnostic. Report the already allocated
      // protocols without performing any additional dependency/proof query.
      if (runtime::detail::VerboseLoggingEnabled(__FILE__, 1)) {
        auto node_name = [](IRStructure *node) {
          if (node == nullptr)
            return std::string("<root>");
          std::ostringstream name;
          if (node->IsControl()) {
            name << "loop:"
                 << static_cast<ControlNode *>(node)->control->loop_var;
          } else {
            Stmt body = static_cast<TaskNode *>(node)->stmt;
            while (const auto *attribute = body.as<AttrStmtNode>())
              body = attribute->body;
            const auto *evaluate = body.as<EvaluateNode>();
            const auto *call =
                evaluate ? evaluate->value.as<CallNode>() : nullptr;
            name << "task:";
            const auto *op = call ? call->op.as<OpNode>() : nullptr;
            if (op)
              name << op->name;
            else
              name << body->GetTypeKey();
          }
          name << "[index=" << node->GetIndex() << ",stage=" << node->GetStage()
               << "]";
          return name.str();
        };
        for (size_t allocation_index = 0;
             allocation_index < allocations_.size(); ++allocation_index) {
          const FlagAllocation &allocation = allocations_[allocation_index];
          for (size_t point_index : allocation.sync_points) {
            const SyncPoint &point = sync_points_[point_index];
            if (FlagDomain(point) != domain)
              continue;
            const SyncInsertionSite &src = site_registry_.Src(point);
            const SyncInsertionSite &dst = site_registry_.Dst(point);
            LOG(INFO) << "Flag-capacity domain=" << domain << " total=" << total
                      << " allocation=" << allocation_index << " ids=["
                      << allocation.base_id << ","
                      << allocation.base_id + allocation.num_versions - 1
                      << "] point=" << point_index
                      << " versions=" << point.num_versions
                      << " distance=" << point.edge.distance
                      << " flag_loop=" << node_name(point.flag_loop)
                      << " clock=" << point.flag_iteration
                      << " counter_group=" << point.counter_group_id
                      << " unit_boundary="
                      << static_cast<int>(point.unit_boundary)
                      << " src=" << node_name(src.path_begin) << "/"
                      << node_name(src.path_end) << " src_pipe=" << src.pipe
                      << " src_domain=" << src.EpochDomainId()
                      << " src_guard=" << src.GetGuard() << " src_active="
                      << site_registry_.Domain(src).active_guard
                      << " dst=" << node_name(dst.path_begin) << "/"
                      << node_name(dst.path_end) << " dst_pipe=" << dst.pipe
                      << " dst_domain=" << dst.EpochDomainId()
                      << " dst_guard=" << dst.GetGuard() << " dst_active="
                      << site_registry_.Domain(dst).active_guard;
          }
        }
      }
      if (require_intra_core_capacity) {
        LOG(WARNING) << "Optional unit stream exceeds local hard-event "
                        "capacity after safe reuse: domain="
                     << domain << ", total=" << total
                     << ", limit=" << kIntraCoreFlagLimit;
        return false;
      }
      LOG(WARNING) << "Ascend auto-schedule reached an intra-core flag-id "
                      "high-water mark of "
                   << total << " for " << domain << " after safe reuse, "
                   << "exceeding the dav-3510 limit of " << kIntraCoreFlagLimit
                   << "; the later RewriteFlagToBuf pass will compact sparse "
                      "out-of-range ids or spill excess flag blocks to "
                      "get_buf/rls_buf when capacity is exceeded.";
    }

    for (const FlagAllocation &allocation : allocations_) {
      if (allocation.cross_iter)
        AppendCrossIterFlagInfo(allocation);
    }
    for (size_t index = 0; index < operand_pools_.size(); ++index)
      for (int slot = 0; slot < operand_pools_[index].slots; ++slot)
        flag_infos_.push_back(FlagInfo{kCoreCube,
                                      Integer(operand_pool_bases_[index] + slot),
                                      "M_MTE1", 0});

    for (const FlagAllocation &allocation : allocations_) {
      auto multi_owner = std::find_if(
          allocation.sync_points.begin(), allocation.sync_points.end(),
          [&](size_t member) { return IsMultiOwnerCounterChannel(member); });
      if (multi_owner == allocation.sync_points.end())
        continue;
      size_t representative = counter_channel_representative_[*multi_owner];
      ICHECK_EQ(allocation.sync_points.size(),
                counter_channel_size_[representative]);
      for (size_t member : allocation.sync_points) {
        ICHECK_EQ(counter_channel_representative_[member], representative)
            << "A multi-owner counter channel allocation must remain "
               "exclusive";
      }
    }
    return true;
  }

  const std::vector<FlagAllocation> &Allocations() const {
    return allocations_;
  }

  FlagInfoList TakeFlagInfos() { return std::move(flag_infos_); }
  const std::vector<int> &OperandPoolBases() const { return operand_pool_bases_; }

private:
  void GroupCounterChannels() {
    counter_channel_representative_.resize(sync_points_.size());
    std::map<CounterChannelKey, size_t> representatives;
    for (size_t index = 0; index < sync_points_.size(); ++index) {
      counter_channel_representative_[index] = index;
      const SyncPoint &sync_point = sync_points_[index];
      if (!sync_point.counter_channel.has_value() ||
          !site_registry_.UsesFlag(sync_point)) {
        continue;
      }
      auto [it, inserted] =
          representatives.emplace(sync_point.GetCounterChannel(), index);
      if (!inserted)
        counter_channel_representative_[index] = it->second;
    }

    counter_channel_size_.assign(sync_points_.size(), 0);
    for (size_t representative : counter_channel_representative_)
      ++counter_channel_size_[representative];
  }

  // GroupCounterChannels() must run first. It groups only flag-using sync
  // points, so non-flag events do not make a counter channel multi-owner.
  bool IsMultiOwnerCounterChannel(size_t index) const {
    if (!sync_points_[index].counter_channel.has_value())
      return false;
    size_t representative = counter_channel_representative_[index];
    return counter_channel_size_[representative] > 1;
  }

  std::string FlagDomain(const SyncPoint &sync_point) const {
    if (site_registry_.IsCrossCore(sync_point))
      return "Cube-Vector";
    return GetHardEvent(site_registry_.Src(sync_point).pipe,
                        site_registry_.Dst(sync_point).pipe);
  }

  void AppendCrossIterFlagInfo(const FlagAllocation &allocation) {
    const SyncPoint &sync_point = sync_points_[allocation.sync_points.front()];
    const SyncInsertionSite &src = site_registry_.Src(sync_point);
    const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
    std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
    size_t separator = hard_event.find('_');
    std::string producer_pipe = "PIPE_" + hard_event.substr(0, separator);
    std::string consumer_pipe = "PIPE_" + hard_event.substr(separator + 1);
    int base_id = allocation.base_id;

    if (!site_registry_.IsCrossCore(sync_point)) {
      if (sync_point.local_binary_plan) {
        ICHECK_EQ(allocation.sync_points.size(), 1U);
        const LocalBinaryFlagPlan &plan = *sync_point.local_binary_plan;
        for (int token = 0; token < plan.distance; ++token) {
          int epoch = token - plan.distance;
          int residue =
              (epoch % allocation.num_versions + allocation.num_versions) %
              allocation.num_versions;
          PrimExpr last_epoch =
              plan.trip_count + IntImm(plan.trip_count.dtype(), epoch);
          flag_infos_.push_back(FlagInfo{
              src.core, Integer(base_id + residue), hard_event, 0,
              MakeFlagEventId(last_epoch, allocation.num_versions, base_id)});
        }
        return;
      }
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(
            FlagInfo{src.core, Integer(base_id + version), hard_event, 0});
      }
      return;
    }

    if (src.core == kCoreVector) {
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{kCoreVector, Integer(base_id + version),
                                       producer_pipe, 1});
      }
      for (int channel = 0; channel < cross_core_contract_.CubeChannels();
           ++channel) {
        for (int version = 0; version < allocation.num_versions; ++version) {
          flag_infos_.push_back(FlagInfo{
              kCoreCube,
              Integer(cross_core_contract_.CubeFlagBase(base_id, channel) +
                      version),
              consumer_pipe, 2});
        }
      }
    } else {
      for (int channel = 0; channel < cross_core_contract_.CubeChannels();
           ++channel) {
        for (int version = 0; version < allocation.num_versions; ++version) {
          flag_infos_.push_back(FlagInfo{
              kCoreCube,
              Integer(cross_core_contract_.CubeFlagBase(base_id, channel) +
                      version),
              producer_pipe, 1});
        }
      }
      for (int version = 0; version < allocation.num_versions; ++version) {
        flag_infos_.push_back(FlagInfo{kCoreVector, Integer(base_id + version),
                                       consumer_pipe, 2});
      }
    }
  }

  static constexpr int kCrossCoreFlagLimit = 16;

  const SyncSiteRegistry &site_registry_;
  std::vector<SyncPoint> &sync_points_;
  const std::set<std::pair<size_t, size_t>> &reusable_pairs_;
  const CrossCoreFlagReservation &cross_core_flag_reservation_;
  const CrossCoreSyncContract &cross_core_contract_;
  const std::vector<L0OperandSlotPool> &operand_pools_;
  std::vector<int> operand_pool_bases_;
  std::vector<FlagAllocation> allocations_;
  FlagInfoList flag_infos_;
  std::vector<size_t> counter_channel_representative_;
  std::vector<size_t> counter_channel_size_;
};

// ============================================================================
// Synchronization statement construction and IR materialization
// ============================================================================
using SyncInsertionMap = std::map<CoreMask, std::vector<Stmt>>;

// InsertSync records synchronization here while dependency analysis still
// refers to the original tree. Build() materializes the lists as adjacent
// TaskNodes after all sync points have been planned and inserted. Same-core
// T.PerCoreTask synchronization is instead migrated beside its candidate tasks.
class SyncInsertionExtraInfo final : public IRExtraInfo {
public:
  std::unique_ptr<IRExtraInfo> Clone() const final {
    return std::make_unique<SyncInsertionExtraInfo>(*this);
  }

  SyncInsertionMap &Before() { return before_; }
  const SyncInsertionMap &Before() const { return before_; }
  SyncInsertionMap &After() { return after_; }
  const SyncInsertionMap &After() const { return after_; }

  void SubstituteVar(const Var &old_var, const Var &new_var) final {
    for (auto &[_, statements] : before_) {
      for (Stmt &statement : statements)
        statement = Substitute(statement, {{old_var, new_var}});
    }
    for (auto &[_, statements] : after_) {
      for (Stmt &statement : statements)
        statement = Substitute(statement, {{old_var, new_var}});
    }
  }

private:
  SyncInsertionMap before_;
  SyncInsertionMap after_;
};

SyncInsertionExtraInfo *GetSyncInsertionExtraInfo(IRStructure *node) {
  return node->GetExtraInfo<SyncInsertionExtraInfo>();
}

std::unique_ptr<IRExtraInfo> MakeSyncInsertionExtraInfo() {
  return std::make_unique<SyncInsertionExtraInfo>();
}

Stmt MakeSyncTask(const Stmt &body, CoreMask core_mask) {
  ICHECK(IsConcreteCore(core_mask));
  TaskMetadata metadata;
  metadata.core_mask = core_mask;
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  attr::kAscendTask, Integer(1), body, body->span);
}

std::shared_ptr<TaskNode> MakeSyncTaskNode(const Stmt &body, CoreMask core_mask,
                                           int stage) {
  auto task = std::make_shared<TaskNode>(MakeSyncInsertionExtraInfo());
  task->stmt = MakeSyncTask(body, core_mask);
  task->SetStage(stage);
  task->SetCoreMask(core_mask);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  return task;
}

// Retire invariant scalar Load3D initialization once on the AIC MTE1 queue.
// Later MTE1 descriptors inherit this readiness through issue order, even
// when their data loops skip iterations. The site has no buffer access and
// does not claim completion of any earlier MTE1 data operation.
void InsertRootLoad3DInitMarker(
    std::vector<std::shared_ptr<IRStructure>> &root, const Stmt &kernel_body) {
  constexpr uint16_t mask =
      static_cast<uint16_t>(SpecialRegister::kLoad3DConfig);
  std::vector<std::pair<size_t, TaskNode *>> writers;
  std::vector<std::pair<size_t, TaskNode *>> readers;
  for (size_t i = 0; i < root.size(); ++i) {
    std::vector<TaskNode *> tasks;
    CollectAllTaskNodes(root[i].get(), tasks);
    for (TaskNode *task : tasks) {
      if (task->GetSpecialWriteMask() & mask)
        writers.emplace_back(i, task);
      // Only actual typed or late Load3D instructions request this marker;
      // an empty task or unrelated explicit SPR bit cannot create it.
      if (AnalyzeSpecialRegisterReads(task->stmt) & mask)
        readers.emplace_back(i, task);
    }
  }
  if (readers.empty())
    return;

  TaskNode *initializer = nullptr;
  size_t initializer_index = root.size();
  for (const auto &[index, writer] : writers) {
    const auto *task_marker = writer->stmt.as<AttrStmtNode>();
    const auto *evaluate = task_marker &&
                                  task_marker->attr_key == attr::kAscendTask
                              ? task_marker->body.as<EvaluateNode>()
                              : nullptr;
    const auto *call = evaluate ? evaluate->value.as<CallNode>() : nullptr;
    ICHECK(root[index].get() == writer && !writer->HasGuards() &&
           initializer == nullptr && call &&
           call->op.same_as(ascend_set_load3d_config()))
        << "Load3D configuration requires exactly one unconditional "
           "kernel-root S task; repeated configuration is unsupported";
    ICHECK_EQ(writer->GetPipeMask(), PipeMask(ResourcePipe::kScalar));
    ICHECK_EQ(writer->GetCoreMask(), kCoreCube)
        << "Load3D configuration must be owned by the AIC";
    ICHECK_EQ(writer->GetStage(), 0)
        << "Load3D root configuration must have stage zero";
    initializer = writer;
    initializer_index = index;
  }
  for (const auto &[index, reader] : readers) {
    ICHECK(initializer != nullptr && initializer_index < index)
        << "Selected Load3D requires a preceding kernel-root configuration";
    ICHECK_EQ(reader->GetCoreMask(), kCoreCube)
        << "Selected Load3D must be owned by the AIC";
    ICHECK(reader->UsesOnlyPipe(ResourcePipe::kMTE1))
        << "Selected Load3D must issue only on MTE1";
  }
  // Inspect the complete input, including control expressions: a raw setter
  // hidden in an opaque call cannot be assumed to preserve the root SPR state.
  PostOrderVisit(kernel_body, [&](const ffi::ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      ICHECK(ascend::PreservesLoad3DConfig(call))
          << "Load3D configuration cannot cross an external or opaque "
             "operation without a modeled register-preservation contract: "
          << call->op;
    }
  });

  auto marker = MakeSyncTaskNode(Evaluate(0), kCoreCube, 0);
  marker->SetPipeMask(PipeMask(ResourcePipe::kMTE1));
  marker->SetSpecialReadMask(mask);
  marker->outer_ctx = initializer->outer_ctx;
  root.insert(root.begin() + initializer_index + 1, std::move(marker));
  for (size_t i = 0; i < root.size(); ++i) {
    root[i]->SetIndex(i);
    root[i]->SetParent(nullptr);
  }
}

class PerCoreTaskSyncMigrator : public StmtMutator {
public:
  PerCoreTaskSyncMigrator(std::vector<Stmt> before, std::vector<Stmt> after,
                          CoreMask core_mask)
      : before_(std::move(before)), after_(std::move(after)),
        core_mask_(core_mask) {
    ICHECK(IsConcreteCore(core_mask_));
  }

  Stmt Rewrite(Stmt body) {
    Stmt result = VisitStmt(std::move(body));
    ICHECK_GT(num_candidates_, 0)
        << "Cannot find a T.Task marker in normalized T.PerCoreTask";
    return result;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != attr::kAscendTask)
      return StmtMutator::VisitStmt_(op);
    Stmt candidate_body = VisitStmt(op->body);
    Stmt candidate =
        AttrStmt(op->node, op->attr_key, op->value, candidate_body, op->span);
    if (!HasTaskBufferAccess(candidate_body))
      return candidate;
    ++num_candidates_;
    Array<Stmt> sequence;
    sequence.reserve(before_.size() + 1 + after_.size());
    for (const Stmt &stmt : before_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    sequence.push_back(std::move(candidate));
    for (const Stmt &stmt : after_)
      sequence.push_back(MakeSyncTask(stmt, core_mask_));
    return SeqStmt::Flatten(sequence);
  }

  std::vector<Stmt> before_;
  std::vector<Stmt> after_;
  CoreMask core_mask_;
  int num_candidates_{0};
};

std::vector<Stmt> GetCoreStatements(const SyncInsertionMap &insertions,
                                    CoreMask core_mask) {
  auto it = insertions.find(core_mask);
  return it == insertions.end() ? std::vector<Stmt>() : it->second;
}

void AppendSyncTaskNodes(std::vector<std::shared_ptr<IRStructure>> *result,
                         const SyncInsertionMap &insertions, int stage,
                         std::optional<CoreMask> migrated_core = std::nullopt) {
  for (const auto &[core_mask, statements] : insertions) {
    if (migrated_core.has_value() && core_mask == migrated_core.value())
      continue;
    for (const Stmt &statement : statements)
      result->push_back(MakeSyncTaskNode(statement, core_mask, stage));
  }
}

inline Stmt MakeAscendSetFlag(const std::string &hard_event,
                              PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_set_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendWaitFlag(const std::string &hard_event,
                               PrimExpr event_id) {
  return Evaluate(Call(DataType::Handle(), ascend_wait_flag(),
                       {StringImm(hard_event), event_id}));
}

inline Stmt MakeAscendPipeBarrier(const std::string &pipe) {
  return Evaluate(
      Call(DataType::Handle(), ascend_pipe_barrier(), {StringImm(pipe)}));
}

// A kernel may keep CTRL51 invariant only when every operation using M/FIX
// belongs to a selected closed stream. Partial selection retains scoped
// direction transitions; an unselected operation has no such state contract.
bool HasClosedKernelUnitDirection(
    const std::vector<std::shared_ptr<IRStructure>> &root,
    const std::vector<UnitFlagStreamPlan> &plans, const Stmt &kernel_body) {
  if (plans.empty())
    return false;
  // Reuse the existing native/pure-operation admission. An unmodeled opaque
  // scalar call may change CTRL51 even though it uses neither M nor FIX.
  // Explicit direction/atomic setters were rejected before stream selection.
  bool known_state = true;
  PostOrderVisit(kernel_body, [&](const ffi::ObjectRef &node) {
    if (const auto *call = node.as<CallNode>())
      known_state &= ascend::PreservesLoad3DConfig(call);
  });
  if (!known_state)
    return false;
  std::unordered_set<TaskNode *> mads, fixes;
  for (const auto &plan : plans) {
    mads.insert(plan.mads.begin(), plan.mads.end());
    fixes.insert(plan.fix);
  }
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(root, tasks);
  for (TaskNode *task : tasks) {
    if ((task->UsesPipe(ResourcePipe::kCube) && !mads.count(task)) ||
        (task->UsesPipe(ResourcePipe::kFixpipe) && !fixes.count(task)))
      return false;
  }
  return true;
}

// The installed dav-2201 inner-product Matmul reuses one canonical C bank
// through matching complete unit results. Admit only the automatically
// selected two-allocation phase pair here. The broader unit selector also
// accepts column prefixes; those do not establish this C-byte contract.
bool HasClosedCByteRetirement(const ScheduledTIRMetadata &metadata,
                              const std::vector<UnitFlagStreamPlan> &plans) {
  if (plans.size() != 2 || plans[0].c->data.same_as(plans[1].c->data))
    return false;
  auto selected = metadata.kernel_root->annotations.Get(kL0PhaseReusePairs);
  if (!selected)
    return false;
  bool selected_pair = false;
  for (const auto &pair : selected.value().cast<Array<Array<Var>>>()) {
    if (pair.size() == 2)
      selected_pair |=
          (pair[0].same_as(plans[0].c->data) &&
           pair[1].same_as(plans[1].c->data)) ||
          (pair[1].same_as(plans[0].c->data) &&
           pair[0].same_as(plans[1].c->data));
  }
  size_t allocations = 0;
  for (const auto &buffer : metadata.kernel_root->alloc_buffers)
    if (IsL0CBuffer(buffer)) {
      ++allocations;
      if (!buffer.same_as(plans[0].c) && !buffer.same_as(plans[1].c))
        return false;
    }
  if (!selected_pair || allocations != 2)
    return false;
  auto full = [](const Buffer &buffer, const Array<Range> &region) {
    if (!unit_flag_stream_detail::AdmissibleCDomain(buffer, region))
      return false;
    for (size_t axis = 0; axis < 2; ++axis)
      if (!arith::Analyzer().CanProveEqual(region[axis]->extent,
                                           buffer->shape[axis]))
        return false;
    return true;
  };
  for (const auto &plan : plans) {
    auto version = metadata.buffer_versions.Get(plan.c->data);
    // Phase-pair selection consumes an explicit fixed-one version entry.
    // An absent entry therefore retains the same implicit single slot.
    if (plan.mads.size() != 1 || (version && version.value() != 1) ||
        metadata.manual_buffer_versions.count(plan.c->data) ||
        metadata.buffer_version_modes.count(plan.c->data))
      return false;
    auto mad = unit_flag_stream_detail::Operation(plan.mads.front(), true);
    auto fix = unit_flag_stream_detail::Operation(plan.fix, true);
    if (!mad || !fix)
      return false;
    Gemm gemm((*mad)->args, (*mad)->annotations);
    AscendCopy copy((*fix)->args, (*fix)->annotations);
    if (!full(plan.c, gemm->cRegion_->region) ||
        !full(plan.c, copy->src_range))
      return false;
    // CANN's native compute path adds PIPE_M for fewer than ten output
    // fractals. Its ungated single-bank protocol is not the contract for
    // those small results.
    auto rows = unit_flag_stream_detail::Constant(plan.c->shape[0]);
    auto columns = unit_flag_stream_detail::Constant(plan.c->shape[1]);
    if (!rows || !columns || (*rows / 16) * (*columns / 16) < 10)
      return false;
  }
  return true;
}

// Empty M tasks are insertion sites only: they do not claim a C access or
// manufacture an allocation-lifetime proof. With invariant kernel direction,
// each owner retains a real M->first FIX / last FIX->M completion handshake.
// Its two ordinary channels prove their own consume-before-reuse relation;
// they must not use the scoped drain's Scalar-blocking lifetime exception.
void InsertUnitFlagBoundaries(std::vector<std::shared_ptr<IRStructure>> &nodes,
                              ControlNode *parent,
                              const std::vector<UnitFlagStreamPlan> &plans,
                              std::vector<UnitFlagBoundary> *boundaries,
                              bool kernel_direction,
                              bool closed_c_byte_retirement) {
  std::vector<std::shared_ptr<IRStructure>> result;
  for (const auto &node : nodes) {
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node.get());
      InsertUnitFlagBoundaries(control->children, control, plans, boundaries,
                               kernel_direction, closed_c_byte_retirement);
    }
    auto plan = std::find_if(plans.begin(), plans.end(), [&](const auto &p) {
      return p.result_loop == node.get();
    });
    if (plan == plans.end()) {
      result.push_back(node);
      continue;
    }
    // An activation condition must cover the complete direction transaction,
    // including its terminal drain. Give that condition one lexical home;
    // copying it onto six siblings would re-evaluate it and split the finite
    // event domain. The selector proved that only immutable outer-loop values
    // and pure snapshots occur here. The original fallback was cloned before
    // either this shell or the selected controls were introduced.
    std::shared_ptr<ControlNode> activation;
    ControlNode *scope = parent;
    if (plan->result_loop->HasConditions()) {
      ICHECK(unit_flag_stream_detail::StableActivationGuards(plan->result_loop));
      activation =
          std::make_shared<ControlNode>(MakeSyncInsertionExtraInfo());
      activation->task =
          std::make_shared<TaskNode>(MakeSyncInsertionExtraInfo());
      activation->task->SetParent(activation.get());
      activation->SetParent(parent);
      activation->SetStage(node->GetStage());
      activation->control =
          For(Var("unit_activation", DataType::Int(32)), Integer(0), Integer(1),
              ForKind::kSerial, Evaluate(0));
      activation->task->stmt = activation->control;
      activation->task->outer_ctx = plan->result_loop->task->outer_ctx;
      GuardList guards;
      for (const auto &guard : plan->result_loop->GetGuards()) {
        ICHECK(guard->IsCondition());
        guards.push_back(guard->Clone());
        const auto *condition =
            static_cast<const ConditionGuard *>(guard.get());
        ApplyTaskAccesses(AnalyzeTaskAccesses(Evaluate(condition->condition)),
                          activation->task.get());
      }
      activation->SetGuards(std::move(guards));
      plan->result_loop->SetGuards({});
      // Guard reads now belong to the shell control, while the original
      // result control retains only its original bounds/step reads. Existing
      // task contexts still describe the same condition, now an ancestor.
      TaskNode *loop_task = plan->result_loop->task.get();
      loop_task->SetReadRegions({});
      loop_task->SetWriteRegions({});
      loop_task->SetReadVars({});
      loop_task->SetWriteVars({});
      ApplyTaskAccesses(
          AnalyzeTaskAccesses(Evaluate(plan->result_loop->control->min)),
          loop_task);
      ApplyTaskAccesses(
          AnalyzeTaskAccesses(Evaluate(plan->result_loop->control->extent)),
          loop_task);
      if (plan->result_loop->control->step.defined())
        ApplyTaskAccesses(
            AnalyzeTaskAccesses(
                Evaluate(plan->result_loop->control->step.value())),
            loop_task);
      scope = activation.get();
    }
    auto make_site = [&](ResourcePipe pipe, const Stmt &body) {
      auto task = MakeSyncTaskNode(body, kCoreCube, node->GetStage());
      task->SetPipeMask(PipeMask(pipe));
      task->outer_ctx = plan->result_loop->task->outer_ctx;
      task->SetParent(scope);
      return task;
    };
    // Generic unit readiness does not establish whole-MAD retirement. Only
    // the complete selected C-byte protocol may end that storage lifetime
    // at matching FIX completion; A/B still use their whole-MAD slot credits.
    Stmt entry_body = kernel_direction && !closed_c_byte_retirement
                          ? MakeAscendPipeBarrier("PIPE_M")
                          : Stmt(Evaluate(0));
    auto entry = make_site(ResourcePipe::kCube, entry_body);
    auto drain = make_site(ResourcePipe::kCube, Evaluate(0));
    if (kernel_direction) {
      boundaries->push_back({scope, entry.get(), entry.get(), node.get(),
                             plan->fix, UnitFlagBoundaryRole::kStreamEntry});
      boundaries->push_back({scope, node.get(), plan->fix, drain.get(),
                             drain.get(), UnitFlagBoundaryRole::kStreamDrain});
      if (activation) {
        activation->children = {entry, node, drain};
        for (size_t i = 0; i < activation->children.size(); ++i) {
          activation->children[i]->SetIndex(i);
          activation->children[i]->SetParent(activation.get());
        }
        activation->InitializeLoopBodyContext();
        result.push_back(std::move(activation));
      } else {
        result.insert(result.end(), {entry, node, drain});
      }
      continue;
    }
    auto setup =
        make_site(ResourcePipe::kScalar,
                  Evaluate(Call(DataType::Handle(), ascend_set_mmad_direction(),
                                {StringImm("n")})));
    setup->AddSpecialWrite(SpecialRegister::kMmadDirection);
    auto restore =
        make_site(ResourcePipe::kScalar,
                  Evaluate(Call(DataType::Handle(), ascend_set_mmad_direction(),
                                {StringImm("m")})));
    restore->AddSpecialWrite(SpecialRegister::kMmadDirection);
    auto exit = make_site(ResourcePipe::kCube, Evaluate(0));
    boundaries->push_back({scope, entry.get(), entry.get(), setup.get(),
                           setup.get(), UnitFlagBoundaryRole::kEntry});
    boundaries->push_back({scope, setup.get(), setup.get(), node.get(),
                           plan->mads.front(), UnitFlagBoundaryRole::kSetup});
    // The entry M_S also retires MADs from the preceding activation. Carry
    // that real completion to FIX explicitly: the selected MAD/FIX unit
    // relation is C data protection, not a whole-pipe completion witness for
    // an enclosing owner/epoch's carried RAW dependency.
    boundaries->push_back({scope, setup.get(), setup.get(), node.get(),
                           plan->fix, UnitFlagBoundaryRole::kFixSetup});
    boundaries->push_back({scope, node.get(), plan->fix, drain.get(),
                           drain.get(), UnitFlagBoundaryRole::kDrain});
    boundaries->push_back({scope, drain.get(), drain.get(), restore.get(),
                           restore.get(), UnitFlagBoundaryRole::kRestore});
    // AutoSchedule ran before this state transition was introduced. Make the
    // restored direction visible to subsequent, unselected MADs as well.
    boundaries->push_back({scope, restore.get(), restore.get(), exit.get(),
                           exit.get(), UnitFlagBoundaryRole::kExit});
    if (activation) {
      activation->children = {entry, setup, node, drain, restore, exit};
      for (size_t i = 0; i < activation->children.size(); ++i) {
        activation->children[i]->SetIndex(i);
        activation->children[i]->SetParent(activation.get());
      }
      activation->InitializeLoopBodyContext();
      result.push_back(std::move(activation));
    } else {
      result.insert(result.end(), {entry, setup, node, drain, restore, exit});
    }
  }
  for (size_t i = 0; i < result.size(); ++i) {
    result[i]->SetIndex(i);
    result[i]->SetParent(parent);
  }
  nodes = std::move(result);
}

// Root consumers are unconditional: even an empty or entirely inactive first
// stream consumes the setup flags. The terminal M site follows every active
// owner's FIX->M drain, so restore is safe for zero as well as many owners.
void InsertKernelUnitDirection(std::vector<std::shared_ptr<IRStructure>> &root,
                               std::vector<UnitFlagBoundary> *boundaries) {
  auto make_site = [](ResourcePipe pipe, const Stmt &body) {
    auto task = MakeSyncTaskNode(body, kCoreCube, 0);
    task->SetPipeMask(PipeMask(pipe));
    return task;
  };
  auto setup =
      make_site(ResourcePipe::kScalar,
                Evaluate(Call(DataType::Handle(), ascend_set_mmad_direction(),
                              {StringImm("n")})));
  setup->AddSpecialWrite(SpecialRegister::kMmadDirection);
  auto cube_start = make_site(ResourcePipe::kCube, Evaluate(0));
  auto fix_start = make_site(ResourcePipe::kFixpipe, Evaluate(0));
  fix_start->SetSpecialReadMask(
      static_cast<uint16_t>(SpecialRegister::kMmadDirection));
  auto terminal = make_site(ResourcePipe::kCube, Evaluate(0));
  auto restore =
      make_site(ResourcePipe::kScalar,
                Evaluate(Call(DataType::Handle(), ascend_set_mmad_direction(),
                              {StringImm("m")})));
  restore->AddSpecialWrite(SpecialRegister::kMmadDirection);
  // Scheduling preceded these sites. Retain both startup edges explicitly,
  // rather than expecting its cached root conflicts to include new readers.
  boundaries->push_back({nullptr, setup.get(), setup.get(), cube_start.get(),
                         cube_start.get(), UnitFlagBoundaryRole::kSetup});
  boundaries->push_back({nullptr, setup.get(), setup.get(), fix_start.get(),
                         fix_start.get(), UnitFlagBoundaryRole::kFixSetup});
  boundaries->push_back({nullptr, terminal.get(), terminal.get(), restore.get(),
                         restore.get(), UnitFlagBoundaryRole::kRestore});
  root.insert(root.begin(), {setup, cube_start, fix_start});
  root.insert(root.end(), {terminal, restore});
  for (size_t i = 0; i < root.size(); ++i) {
    root[i]->SetIndex(i);
    root[i]->SetParent(nullptr);
  }
}

inline Stmt MakeAscendCrossCoreSetFlag(int mode_id, const std::string &pipe,
                                       PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_set_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

inline Stmt MakeAscendCrossCoreWaitFlag(int mode_id, const std::string &pipe,
                                        PrimExpr flag_id) {
  return Evaluate(Call(DataType::Handle(), ascend_cross_core_wait_flag(),
                       {Integer(mode_id), StringImm(pipe), flag_id}));
}

PrimExpr MakeFlagEventId(const PrimExpr &iteration, int num_versions,
                         int base_id) {
  DataType dtype = iteration.dtype();
  PrimExpr event_id = num_versions == 1
                          ? IntImm(dtype, base_id)
                          : indexmod(iteration, IntImm(dtype, num_versions)) +
                                IntImm(dtype, base_id);
  return dtype == DataType::Int(32) ? event_id
                                    : cast(DataType::Int(32), event_id);
}

bool ShouldElidePerCoreTaskSyncGuard(IRStructure *node, CoreMask core,
                                     const PrimExpr &guard) {
  if (!node->IsTask())
    return false;
  const auto *task = static_cast<const TaskNode *>(node);
  if (!task->IsPerCoreTask() || task->GetCoreMask() != core)
    return false;

  ConstrSet outer_ctx = task->GetContextBeforeConditionGuards();
  PrimExpr task_guard = task->GetConditionGuard();
  PrimExpr sync_guard = guard.defined() ? guard : Bool(true);
  ICHECK(GuardImplies(sync_guard, task_guard, outer_ctx))
      << "Cannot migrate synchronization into guarded T.PerCoreTask: "
         "the synchronization guard "
      << sync_guard << " does not imply the T.PerCoreTask guard " << task_guard;
  return GuardsEquivalent(sync_guard, task_guard, outer_ctx);
}

void InsertStatement(IRStructure *node, Stmt stmt, bool at_beginning,
                     CoreMask core, const PrimExpr &guard = PrimExpr()) {
  ICHECK(node != nullptr);
  ICHECK(IsConcreteCore(core));
  bool elide_guard = ShouldElidePerCoreTaskSyncGuard(node, core, guard);
  if (guard.defined() && !elide_guard)
    stmt = WrapWithGuard(std::move(stmt), guard);
  SyncInsertionExtraInfo *insertions = GetSyncInsertionExtraInfo(node);
  if (at_beginning) {
    std::vector<Stmt> &before = insertions->Before()[core];
    before.insert(before.begin(), std::move(stmt));
  } else {
    insertions->After()[core].push_back(std::move(stmt));
  }
}

// Readiness has an owner selector in addition to the existing storage epoch.
// It has no all-participant closure projection or flag reuse: each owner/GM
// slot owns one ID. Ordinary reader release proves consumption before reuse.
struct OwnerCompletionEvent {
  SyncInsertionSite producer;
  SyncInsertionSite consumer;
  PrimExpr producer_id;
  PrimExpr consumer_id;
  // A V-to-C input still uses mode2: one real owner publication plus one
  // early contribution from the other physical AIV, after old-slot release.
  std::optional<SyncInsertionSite> empty_participant;
  PrimExpr empty_id;
};

std::optional<std::vector<OwnerCompletionEvent>>
AllocateOwnerCompletionEvents(const std::vector<OwnerResultPlan> &plans,
                              const std::vector<OwnerResultPlan> &inputs,
                              const SyncSiteRegistry &site_registry,
                              const std::vector<SyncPoint> &sync_points,
                              const std::vector<FlagAllocation> &allocations,
                              const CrossCoreFlagReservation &reservation,
                              const EpochDomainRegistry &domains) {
  if (plans.empty() && inputs.empty())
    return std::vector<OwnerCompletionEvent>{};
  if (reservation.has_unbounded_id)
    return std::nullopt;
  std::array<bool, 16> used{};
  for (const auto &[lo, hi] : reservation.explicit_id_ranges) {
    if (hi - lo >= 15) {
      used.fill(true);
      break;
    }
    for (int64_t id = lo; id <= hi; ++id)
      used[(id % 16 + 16) % 16] = true;
  }
  for (const FlagAllocation &allocation : allocations) {
    if (!site_registry.IsCrossCore(sync_points[allocation.sync_points.front()]))
      continue;
    for (int offset = 0; offset < allocation.num_versions; ++offset)
      used[allocation.base_id + offset] = true;
  }
  std::vector<OwnerCompletionEvent> result;
  auto reserve = [&](int width) {
    int base = -1;
    for (int candidate = 16 - width; candidate >= 0; --candidate) {
      bool free = true;
      for (int offset = 0; offset < width; ++offset)
        free &= !used[candidate + offset];
      if (free) {
        base = candidate;
        break;
      }
    }
    if (base >= 0)
      for (int offset = 0; offset < width; ++offset)
        used[base + offset] = true;
    return base;
  };
  for (const OwnerResultPlan &plan : plans) {
    int base = reserve(2 * plan.num_slots);
    if (base < 0)
      return std::nullopt;
    const EpochDomain &domain =
        domains.Domain(domains.DomainForStorage(plan.storage, plan.epoch));
    SyncInsertionSite producer(plan.producer->GetChildOnPathFrom(plan.epoch),
                               plan.producer, false, kCoreCube, domain,
                               plan.owner, true);
    SyncInsertionSite consumer(plan.consumer->GetChildOnPathFrom(plan.epoch),
                               plan.consumer, true, kCoreVector, domain,
                               nullptr, true);
    // Include activation captured inside an atomic copy task as well as its
    // enclosing control. The proof expands immutable Binds and excludes
    // descendant-loop coordinates from these predicates.
    producer.guard = producer.guard && plan.producer_activation;
    consumer.guard = consumer.guard && plan.consumer_activation;
    PrimExpr producer_id = plan.owner->control->loop_var * plan.num_slots +
                           plan.producer_slot + base;
    PrimExpr consumer_id =
        plan.sid * plan.num_slots + plan.consumer_slot + base;
    result.push_back({std::move(producer), std::move(consumer),
                      cast(DataType::Int(32), producer_id),
                      cast(DataType::Int(32), consumer_id), std::nullopt,
                      PrimExpr()});
  }
  for (const OwnerResultPlan &plan : inputs) {
    int base = reserve(2 * plan.num_slots);
    if (base < 0)
      return std::nullopt;
    const EpochDomain &domain =
        domains.Domain(domains.DomainForStorage(plan.storage, plan.epoch));
    IRStructure *producer_begin = plan.producer->GetChildOnPathFrom(plan.epoch);
    SyncInsertionSite producer(producer_begin, plan.producer, false,
                               kCoreVector, domain, nullptr, true);
    SyncInsertionSite consumer(plan.consumer->GetChildOnPathFrom(plan.epoch),
                               plan.consumer, true, kCoreCube, domain,
                               plan.owner, true);
    // Use the exact ordinary producer entry whose old-slot completion was
    // proved by HasOwnerInputRelease, rather than hoisting over that release.
    SyncInsertionSite empty =
        site_registry.Sites()[site_registry.FindSiteId(SyncInsertionSiteKey{
            producer_begin, plan.producer, true, kCoreVector, domain.id})];
    producer.guard = producer.guard && plan.producer_activation;
    consumer.guard = consumer.guard && plan.consumer_activation;
    empty.guard = empty.guard && plan.producer_activation;
    PrimExpr producer_id =
        plan.sid * plan.num_slots + plan.producer_slot + base;
    PrimExpr consumer_id = plan.owner->control->loop_var * plan.num_slots +
                           plan.consumer_slot + base;
    PrimExpr empty_id =
        (1 - plan.sid) * plan.num_slots + plan.producer_slot + base;
    result.push_back({std::move(producer), std::move(consumer),
                      cast(DataType::Int(32), producer_id),
                      cast(DataType::Int(32), consumer_id), std::move(empty),
                      cast(DataType::Int(32), empty_id)});
  }
  return result;
}

class SyncMaterializer {
public:
  SyncMaterializer(const SyncSiteRegistry &site_registry,
                   const std::vector<SyncPoint> &sync_points,
                   const std::vector<FlagAllocation> &allocations,
                   AscendArchitecture architecture,
                   const CrossCoreSyncContract &cross_core_contract,
                   const std::vector<Var> &lane_variables,
                   const std::vector<OwnerCompletionEvent> &owner_events,
                   const std::vector<L0OperandSlotPool> &operand_pools,
                   const std::vector<int> &operand_pool_bases,
                   const std::vector<GmReadReleaseCut> &gm_read_releases,
                   const std::vector<GmWriteAcquireCut> &gm_write_acquires)
      : site_registry_(site_registry), sync_points_(sync_points),
        allocations_(allocations), architecture_(architecture),
        cross_core_contract_(cross_core_contract),
        lane_variables_(lane_variables), owner_events_(owner_events),
        operand_pools_(operand_pools), operand_pool_bases_(operand_pool_bases),
        gm_read_releases_(gm_read_releases),
        gm_write_acquires_(gm_write_acquires) {}

  void Run(std::vector<std::shared_ptr<IRStructure>> &root) {
    InsertSyncPoints();
    ICHECK_EQ(operand_pools_.size(), operand_pool_bases_.size());
    for (size_t index = 0; index < operand_pools_.size(); ++index) {
      const auto &pool = operand_pools_[index];
      for (const auto *member : {&pool.first, &pool.second}) {
        PrimExpr event = member->slot + Integer(operand_pool_bases_[index]);
        // Each endpoint keeps its own task stage. MaterializeMultiBuffer and
        // LowerScheduledTIR therefore shift the operand pointer and its credit
        // by exactly the same prepared iteration/counter clock.
        InsertStatement(member->writer, MakeAscendWaitFlag("M_MTE1", event),
                        true, kCoreCube);
        InsertStatement(member->reader, MakeAscendSetFlag("M_MTE1", event),
                        false, kCoreCube, member->last_reader);
      }
    }
    for (const OwnerCompletionEvent &event : owner_events_) {
      ICHECK(architecture_ == AscendArchitecture::kDav2201);
      ICHECK_EQ(cross_core_contract_.num_aiv_subcores, 2);
      if (event.empty_participant.has_value()) {
        const SyncInsertionSite &entry = event.empty_participant.value();
        Stmt contribution =
            MakeAscendCrossCoreSetFlag(2, "PIPE_MTE3", event.empty_id);
        if (!ShouldElidePerCoreTaskSyncGuard(entry.path_end, kCoreVector,
                                             entry.GetGuard()))
          contribution =
              WrapWithGuard(std::move(contribution), entry.GetGuard());
        // InsertStatement(at_beginning=true) prepends. This contribution must
        // instead follow every ordinary release/acquire already at the entry,
        // since their completion proves reuse of this owner/slot counter.
        GetSyncInsertionExtraInfo(entry.path_end)
            ->Before()[kCoreVector]
            .push_back(std::move(contribution));
        InsertStatement(
            event.producer.path_end,
            MakeAscendCrossCoreSetFlag(2, "PIPE_MTE3", event.producer_id),
            false, kCoreVector, event.producer.GetGuard());
        InsertStatement(
            event.consumer.path_end,
            MakeAscendCrossCoreWaitFlag(2, "PIPE_MTE2", event.consumer_id),
            true, kCoreCube, event.consumer.GetGuard());
        continue;
      }
      InsertStatement(
          event.producer.path_end,
          MakeAscendCrossCoreSetFlag(2, "PIPE_FIX", event.producer_id), false,
          kCoreCube, event.producer.GetGuard());
      InsertStatement(
          event.consumer.path_end,
          MakeAscendCrossCoreWaitFlag(2, "PIPE_MTE2", event.consumer_id), true,
          kCoreVector, event.consumer.GetGuard());
    }
    MaterializeInsertions(root, nullptr);
  }

private:
  void InsertSyncPoints() {
    for (const SyncPoint &sync_point : sync_points_) {
      const SyncInsertionSite &src = site_registry_.Src(sync_point);
      const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
      std::string hard_event = GetHardEvent(src.pipe, dst.pipe);
      size_t separator = hard_event.find('_');
      std::string producer_pipe = "PIPE_" + hard_event.substr(0, separator);
      std::string consumer_pipe = "PIPE_" + hard_event.substr(separator + 1);
      PrimExpr producer_guard = src.GetGuard();
      PrimExpr consumer_guard = dst.GetGuard();

      if (!site_registry_.IsCrossCore(sync_point) &&
          producer_pipe == consumer_pipe) {
        if (HasImplicitSamePipeOrder(src.pipe, architecture_)) {
          continue;
        }
        InsertStatement(dst.path_end, MakeAscendPipeBarrier(producer_pipe),
                        true, dst.core, consumer_guard);
        continue;
      }

      const FlagAllocation &allocation =
          allocations_[sync_point.allocation_index];
      int base_id = allocation.base_id;
      PrimExpr event_id = MakeFlagEventId(sync_point.flag_iteration,
                                          sync_point.num_versions, base_id);

      if (sync_point.IsPrivateGmHandoff()) {
        ICHECK(architecture_ == AscendArchitecture::kDav2201 &&
               cross_core_contract_.collective &&
               cross_core_contract_.num_aiv_subcores == 2 &&
               site_registry_.IsCrossCore(sync_point));
        ICHECK_EQ(allocation.sync_points.size(), 1u);
        std::vector<const SyncInsertionSite *> releases{&src};
        std::vector<const SyncInsertionSite *> acquisitions{&dst};
        if (sync_point.gm_read_release) {
          const auto &cut = gm_read_releases_.at(*sync_point.gm_read_release);
          ICHECK(cut.plan.MatchesProtocol(
              sync_point.flag_loop, sync_point.edge.distance,
              sync_point.num_versions, sync_point.counter_group_id));
          releases = {&cut.active, &cut.inactive};
        }
        if (sync_point.gm_write_acquire) {
          const auto &cut = gm_write_acquires_.at(*sync_point.gm_write_acquire);
          ICHECK(cut.plan.MatchesProtocol(
              sync_point.flag_loop, sync_point.edge.distance,
              sync_point.num_versions, sync_point.counter_group_id));
          acquisitions = {&cut.active, &cut.inactive};
        }
        // Compose the two independent placement certificates on the same
        // original permission. Both physical AIV contributions, direction,
        // flag clock and finite seed/drain remain the ordinary protocol's.
        for (const SyncInsertionSite *site : releases) {
          ValidateCollectiveSite(*site);
          InsertStatement(
              site->path_end,
              MakeAscendCrossCoreSetFlag(2, producer_pipe, event_id), false,
              site->core, site->GetGuard());
        }
        for (const SyncInsertionSite *site : acquisitions) {
          ValidateCollectiveSite(*site);
          InsertStatement(
              site->path_end,
              MakeAscendCrossCoreWaitFlag(2, consumer_pipe, event_id), true,
              site->core, site->GetGuard());
        }
        continue;
      }

      if (site_registry_.IsCrossCore(sync_point)) {
        if (cross_core_contract_.collective) {
          ICHECK_EQ(cross_core_contract_.num_aiv_subcores, 2)
              << "dav-2201 mixed completion requires both physical AIV lanes; "
                 "use MixedKernel(sids=2) and keep empty-owner predicates "
                 "inside data task bodies";
          ValidateCollectiveSite(src);
          ValidateCollectiveSite(dst);
        }
        int mode = cross_core_contract_.Mode();
        if (src.core == kCoreVector) {
          InsertStatement(
              src.path_end,
              MakeAscendCrossCoreSetFlag(mode, producer_pipe, event_id), false,
              src.core, producer_guard);
          for (int channel = 0; channel < cross_core_contract_.CubeChannels();
               ++channel) {
            PrimExpr wait_id = MakeFlagEventId(
                sync_point.flag_iteration, sync_point.num_versions,
                cross_core_contract_.CubeFlagBase(base_id, channel));
            InsertStatement(
                dst.path_end,
                MakeAscendCrossCoreWaitFlag(mode, consumer_pipe, wait_id), true,
                dst.core, consumer_guard);
          }
        } else {
          for (int channel = 0; channel < cross_core_contract_.CubeChannels();
               ++channel) {
            PrimExpr set_id = MakeFlagEventId(
                sync_point.flag_iteration, sync_point.num_versions,
                cross_core_contract_.CubeFlagBase(base_id, channel));
            InsertStatement(
                src.path_end,
                MakeAscendCrossCoreSetFlag(mode, producer_pipe, set_id), false,
                src.core, producer_guard);
          }
          InsertStatement(
              dst.path_end,
              MakeAscendCrossCoreWaitFlag(mode, consumer_pipe, event_id), true,
              dst.core, consumer_guard);
        }
        continue;
      }

      InsertStatement(src.path_end, MakeAscendSetFlag(hard_event, event_id),
                      false, src.core, producer_guard);
      PrimExpr wait_id = event_id;
      if (sync_point.local_binary_plan) {
        PrimExpr producer_epoch =
            sync_point.flag_iteration -
            IntImm(sync_point.flag_iteration.dtype(),
                   sync_point.local_binary_plan->distance);
        wait_id =
            MakeFlagEventId(producer_epoch, sync_point.num_versions, base_id);
      }
      InsertStatement(dst.path_end, MakeAscendWaitFlag(hard_event, wait_id),
                      true, dst.core, consumer_guard);
    }
  }

  static ConstrSet GetSiteContext(const SyncInsertionSite &site) {
    if (site.path_end->IsTask())
      return static_cast<TaskNode *>(site.path_end)
          ->GetContextBeforeConditionGuards();
    return static_cast<ControlNode *>(site.path_end)
        ->task->GetContextBeforeConditionGuards();
  }

  void ValidateCollectiveSite(const SyncInsertionSite &site) const {
    const EpochDomain &domain = site_registry_.Domain(site);
    ICHECK(!HasNonuniformCollectiveControl(
        domain.active_guard, GetSiteContext(site), lane_variables_))
        << "dav-2201 collective epoch activation must be uniform across AIC "
           "and both AIV lanes; keep lane/empty data predicates inside tasks";
    for (IRStructure *node : site.path_end->PathFrom()) {
      ConstrSet context =
          node->IsTask()
              ? static_cast<TaskNode *>(node)->GetContextBeforeConditionGuards()
              : static_cast<ControlNode *>(node)
                    ->task->GetContextBeforeConditionGuards();
      ICHECK(!HasNonuniformCollectiveControl(node->GetConditionGuard(), context,
                                             lane_variables_))
          << "dav-2201 collective completion cannot be enclosed by a "
             "lane-dependent or mutable task guard; keep the data predicate "
             "inside T.Task/T.PerCoreTask so empty lanes still complete";
      if (node->IsControl()) {
        const For &loop = static_cast<ControlNode *>(node)->control;
        ICHECK(!HasNonuniformCollectiveControl(loop->min, context,
                                               lane_variables_) &&
               !HasNonuniformCollectiveControl(loop->extent, context,
                                               lane_variables_))
            << "dav-2201 collective completion requires a uniform owner loop; "
               "a possibly empty lane traversal belongs inside a data task";
      }
    }
    ICHECK(!HasNonuniformCollectiveControl(
        site.GetGuard(), GetSiteContext(site), lane_variables_))
        << "dav-2201 collective synchronization guard must be uniform";
  }

  void MaterializeInsertions(std::vector<std::shared_ptr<IRStructure>> &nodes,
                             ControlNode *parent) {
    std::vector<std::shared_ptr<IRStructure>> materialized;
    materialized.reserve(nodes.size());
    for (const std::shared_ptr<IRStructure> &node : nodes) {
      if (node->IsControl()) {
        auto *control = static_cast<ControlNode *>(node.get());
        MaterializeInsertions(control->children, control);
      }

      const SyncInsertionExtraInfo *insertions =
          GetSyncInsertionExtraInfo(node.get());
      const SyncInsertionMap &before_insertions = insertions->Before();
      const SyncInsertionMap &after_insertions = insertions->After();

      std::optional<CoreMask> migrated_core;
      if (node->IsTask()) {
        auto *task = static_cast<TaskNode *>(node.get());
        if (task->IsPerCoreTask()) {
          CoreMask core_mask = task->GetCoreMask();
          ICHECK(IsConcreteCore(core_mask))
              << "Scheduled T.PerCoreTask must belong to one concrete core";
          migrated_core = core_mask;
          std::vector<Stmt> before =
              GetCoreStatements(before_insertions, core_mask);
          std::vector<Stmt> after =
              GetCoreStatements(after_insertions, core_mask);
          if (!before.empty() || !after.empty()) {
            task->stmt = PerCoreTaskSyncMigrator(std::move(before),
                                                 std::move(after), core_mask)
                             .Rewrite(std::move(task->stmt));
          }
        }
      }

      AppendSyncTaskNodes(&materialized, before_insertions, node->GetStage(),
                          migrated_core);
      materialized.push_back(node);
      AppendSyncTaskNodes(&materialized, after_insertions, node->GetStage(),
                          migrated_core);
    }

    for (size_t i = 0; i < materialized.size(); ++i) {
      materialized[i]->SetIndex(i);
      materialized[i]->SetParent(parent);
    }
    nodes = std::move(materialized);
  }

  const SyncSiteRegistry &site_registry_;
  const std::vector<SyncPoint> &sync_points_;
  const std::vector<FlagAllocation> &allocations_;
  AscendArchitecture architecture_;
  const CrossCoreSyncContract &cross_core_contract_;
  const std::vector<Var> &lane_variables_;
  const std::vector<OwnerCompletionEvent> &owner_events_;
  const std::vector<L0OperandSlotPool> &operand_pools_;
  const std::vector<int> &operand_pool_bases_;
  const std::vector<GmReadReleaseCut> &gm_read_releases_;
  const std::vector<GmWriteAcquireCut> &gm_write_acquires_;
};

// ============================================================================
// Kernel-level validation and scheduled-TIR rewrite
// ============================================================================
SBlock InsertKernelSync(const Stmt &kernel_body, ScheduledTIR scheduled_tir,
                        Target target, const std::vector<Var> &lane_variables,
                        bool enable_unit_streams = true,
                        bool enable_c_byte_retirement = true,
                        OptimizationDiagnostics *diagnostics = nullptr) {
  OptimizationDiagnostics local_diagnostics;
  if (!diagnostics)
    diagnostics = &local_diagnostics;
  auto retry = [&](const char *feature, const char *reason) {
    diagnostics->Record(feature, OptimizationStatus::kRetry, reason);
  };
  std::vector<std::shared_ptr<IRStructure>> &root = scheduled_tir.tree;
  const AscendTargetInfo &target_info = GetAscendTargetInfo(target);
  CrossCoreSyncContract cross_core_contract{
      target_info.architecture == AscendArchitecture::kDav2201,
      scheduled_tir.metadata.num_aiv_subcores.value_or(
          target_info.aiv_per_aic)};
  L0StorageGroups groups(
      CollectL0SFBindings(scheduled_tir.metadata.kernel_root));
  MultiBufferPlan multi_buffer_plan =
      ReadMultiBufferPlan(root, scheduled_tir.metadata.buffer_versions, groups);
  std::vector<UnitFlagStreamPlan> unit_streams;
  std::vector<UnitFlagBoundary> unit_boundaries;
  std::optional<ScheduledTIR> unit_fallback;
  bool closed_c_byte_retirement = false;
  if (enable_unit_streams &&
      target_info.architecture == AscendArchitecture::kDav2201) {
    // An existing direction setter has an explicit state contract. Do not
    // silently replace its scope or assume the state to restore on exit.
    bool explicit_state = false;
    PostOrderVisit(kernel_body, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallNode>())
        explicit_state |= call->op.same_as(ascend_set_mmad_direction()) ||
                          call->op.same_as(ascend_set_atomic()) ||
                          call->op.same_as(ascend_set_atomic_none()) ||
                          call->op.same_as(builtin::call_extern()) ||
                          call->op.same_as(builtin::call_pure_extern());
    });
    if (!explicit_state) {
      unit_streams =
          AnalyzeUnitFlagStreams(root, scheduled_tir.metadata.kernel_root,
                                 scheduled_tir.metadata.buffer_versions,
                                 scheduled_tir.metadata.manual_buffer_versions);
      if (!unit_streams.empty()) {
        diagnostics->Record("unit-flag-stream", OptimizationStatus::kSelected,
                            "native-stream-candidates");
        unit_fallback.emplace();
        unit_fallback->metadata = scheduled_tir.metadata;
        for (const auto &node : root)
          unit_fallback->tree.push_back(node->Clone());
      }
      bool kernel_direction =
          HasClosedKernelUnitDirection(root, unit_streams, kernel_body);
      closed_c_byte_retirement =
          enable_c_byte_retirement && kernel_direction &&
          HasClosedCByteRetirement(scheduled_tir.metadata, unit_streams);
      ApplyUnitFlagStreams(unit_streams);
      InsertUnitFlagBoundaries(root, nullptr, unit_streams, &unit_boundaries,
                               kernel_direction, closed_c_byte_retirement);
      if (kernel_direction)
        InsertKernelUnitDirection(root, &unit_boundaries);
    } else {
      diagnostics->Record("unit-flag-stream", OptimizationStatus::kRejected,
                          "explicit-native-state-or-extern");
    }
  }
  // Keep this after the original unit-fallback snapshot. A recursive retry
  // rebuilds one marker from the unmodified tree; owner-analysis retries below
  // reuse this tree without inserting another initialization consumer.
  if (target_info.architecture == AscendArchitecture::kDav2201)
    InsertRootLoad3DInitMarker(root, kernel_body);
  std::vector<L0OperandSlotPool> operand_pools = AnalyzeL0OperandSlotPools(
      scheduled_tir, multi_buffer_plan, target_info, diagnostics);
  if (runtime::detail::VerboseLoggingEnabled(__FILE__, 1))
    for (const auto &pool : operand_pools)
      LOG(INFO) << "L0 operand slot pool " << pool.first.buffer->data << " / "
                << pool.second.buffer->data << " slots=" << pool.slots
                << " pitch_bytes=" << pool.slot_bytes;
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(root, tasks);
  for (TaskNode *task : tasks) {
    ICHECK_NE(task->GetCoreMask(), kCoreUnassigned)
        << "InsertSync requires ResolveCore to assign every T.Task";
  }
  EpochDomainRegistry domains(root, multi_buffer_plan);
  CrossCoreFlagReservation cross_core_flag_reservation =
      CollectExplicitCrossCoreFlags(kernel_body, cross_core_contract);
  std::vector<OwnerResultPlan> owner_results;
  std::vector<OwnerResultPlan> owner_inputs;
  if (cross_core_contract.collective &&
      cross_core_contract.num_aiv_subcores == 2) {
    owner_results = AnalyzeOwnerResults(
        root, scheduled_tir.metadata.kernel_root,
        scheduled_tir.metadata.manual_buffer_versions, lane_variables);
    owner_inputs = AnalyzeOwnerInputs(
        root, scheduled_tir.metadata.kernel_root,
        scheduled_tir.metadata.manual_buffer_versions, lane_variables);
  }
  std::unique_ptr<SyncAnalyzer> analyzer;
  std::unique_ptr<FlagAllocator> flag_allocator;
  std::optional<std::vector<OwnerCompletionEvent>> owner_events;
  bool enable_gm_read_release = true;
  bool enable_gm_write_acquire = true;
  auto disable_gm_handoff = [&](const char *reason) {
    // Restore the delayed acquire before discarding an independent early
    // input release. Each retry rebuilds the complete original proof graph.
    if (analyzer->HasGmWriteAcquires()) {
      retry("gm-write-acquire", reason);
      enable_gm_write_acquire = false;
      return true;
    }
    if (analyzer->HasGmReadReleases()) {
      retry("gm-read-release", reason);
      enable_gm_read_release = false;
      return true;
    }
    return false;
  };
  for (;;) {
    flag_allocator.reset();
    analyzer = std::make_unique<SyncAnalyzer>(
        multi_buffer_plan, domains, target_info.architecture,
        scheduled_tir.metadata.manual_buffer_versions,
        scheduled_tir.metadata.root_conflict_hints, groups, &owner_results,
        &owner_inputs, &unit_streams, &unit_boundaries, &operand_pools,
        enable_gm_read_release, enable_gm_write_acquire,
        closed_c_byte_retirement);
    if (!analyzer->Analyze(root)) {
      if (closed_c_byte_retirement) {
        retry("c-byte-retirement", "completion-proof");
        ICHECK(unit_fallback.has_value());
        return InsertKernelSync(kernel_body, std::move(unit_fallback.value()),
                                target, lane_variables, true, false,
                                diagnostics);
      }
      if (disable_gm_handoff("completion-proof")) {
        continue;
      }
      if (analyzer->NeedsOperandPoolFallback()) {
        retry("l0-operand-slot-pool", "completion-proof");
        ICHECK(!operand_pools.empty());
        operand_pools.clear();
        continue;
      }
      if (!owner_inputs.empty()) {
        retry("owner-input", "completion-proof");
        owner_inputs.clear();
        continue;
      }
      if (analyzer->NeedsUnitFallback()) {
        retry("unit-flag-stream", "completion-proof");
        ICHECK(unit_fallback.has_value());
        return InsertKernelSync(kernel_body, std::move(unit_fallback.value()),
                                target, lane_variables, false, true,
                                diagnostics);
      }
      ICHECK(!owner_results.empty());
      retry("owner-result", "completion-proof");
      owner_results.clear();
      continue;
    }
    if (closed_c_byte_retirement) {
      bool aliases_proven = true;
      for (const auto &pair : scheduled_tir.metadata.kernel_root->annotations
                                  .Get(kL0PhaseReusePairs)
                                  .value()
                                  .cast<Array<Array<Var>>>())
        aliases_proven &=
            HasBufferAlias(analyzer->BufferAliases(), pair[0], pair[1]) &&
            HasBufferAlias(analyzer->BufferAliases(), pair[1], pair[0]);
      if (!aliases_proven) {
        retry("c-byte-retirement", "mandatory-storage-alias");
        ICHECK(unit_fallback.has_value());
        return InsertKernelSync(kernel_body, std::move(unit_fallback.value()),
                                target, lane_variables, true, false,
                                diagnostics);
      }
    }
    if (analyzer->HasPrivateGmHandoffs()) {
      // A private GM cut cannot replace whole-phase alias retirement.
      // Retry before allocation/materialization if the removed wide edge
      // was needed by a mandatory physical-storage proof.
      bool aliases_proven = true;
      if (auto selected = scheduled_tir.metadata.kernel_root->annotations.Get(
              kL0PhaseReusePairs)) {
        for (const Array<Var> &pair :
             selected.value().cast<Array<Array<Var>>>()) {
          ICHECK_EQ(pair.size(), 2u);
          aliases_proven &=
              HasBufferAlias(analyzer->BufferAliases(), pair[0], pair[1]) &&
              HasBufferAlias(analyzer->BufferAliases(), pair[1], pair[0]);
        }
      }
      bool releases_proven =
          std::all_of(owner_results.begin(), owner_results.end(),
                      [&](const OwnerResultPlan &plan) {
                        return analyzer->HasOwnerResultRelease(plan);
                      }) &&
          std::all_of(owner_inputs.begin(), owner_inputs.end(),
                      [&](const OwnerResultPlan &plan) {
                        return analyzer->HasOwnerInputRelease(plan);
                      });
      if (!aliases_proven || !releases_proven) {
        ICHECK(disable_gm_handoff(!aliases_proven ? "mandatory-storage-alias"
                                                  : "owner-release-proof"));
        continue;
      }
    }
    if (!owner_inputs.empty() &&
        std::any_of(owner_results.begin(), owner_results.end(),
                    [&](const OwnerResultPlan &plan) {
                      return !analyzer->HasOwnerResultRelease(plan);
                    })) {
      // A new optional input cut must not discard an already admissible
      // result protocol by removing one of its ordinary release witnesses.
      retry("owner-input", "preserve-owner-result-release");
      owner_inputs.clear();
      continue;
    }
    size_t candidate_count = owner_results.size();
    owner_results.erase(
        std::remove_if(owner_results.begin(), owner_results.end(),
                       [&](const OwnerResultPlan &plan) {
                         return !analyzer->HasOwnerResultRelease(plan);
                       }),
        owner_results.end());
    if (owner_results.size() != candidate_count) {
      retry("owner-result", "missing-release-witness");
      continue;
    }
    candidate_count = owner_inputs.size();
    owner_inputs.erase(std::remove_if(owner_inputs.begin(), owner_inputs.end(),
                                      [&](const OwnerResultPlan &plan) {
                                        return !analyzer->HasOwnerInputRelease(
                                            plan);
                                      }),
                       owner_inputs.end());
    if (owner_inputs.size() != candidate_count) {
      retry("owner-input", "missing-release-witness");
      continue;
    }
    flag_allocator = std::make_unique<FlagAllocator>(
        analyzer->SiteRegistry(), analyzer->SyncPoints(),
        analyzer->ReusablePairs(), cross_core_flag_reservation,
        cross_core_contract, operand_pools);
    if (!flag_allocator->Plan(!owner_results.empty() || !owner_inputs.empty() ||
                                  !unit_streams.empty() ||
                                  !operand_pools.empty(),
                              !unit_streams.empty() || !operand_pools.empty() ||
                                  analyzer->HasPrivateGmHandoffs())) {
      if (closed_c_byte_retirement) {
        retry("c-byte-retirement", "flag-capacity");
        ICHECK(unit_fallback.has_value());
        return InsertKernelSync(kernel_body, std::move(unit_fallback.value()),
                                target, lane_variables, true, false,
                                diagnostics);
      }
      if (disable_gm_handoff("flag-capacity")) {
        continue;
      }
      if (!operand_pools.empty()) {
        retry("l0-operand-slot-pool", "flag-capacity");
        operand_pools.clear();
        continue;
      }
      if (!owner_inputs.empty()) {
        retry("owner-input", "flag-capacity");
        owner_inputs.clear();
        continue;
      }
      if (!owner_results.empty()) {
        retry("owner-result", "flag-capacity");
        owner_results.clear();
        continue;
      }
      // A closed stream's terminal flag is exclusive. Optional selection
      // must not force a previously legal 2201 kernel past its eight local
      // IDs: this target cannot spill hard events to the 3510 mutex API.
      ICHECK(unit_fallback.has_value());
      retry("unit-flag-stream", "flag-capacity");
      LOG(WARNING) << "Unit stream fallback due to flag allocation capacity";
      return InsertKernelSync(kernel_body, std::move(unit_fallback.value()),
                              target, lane_variables, false, true, diagnostics);
    }
    owner_events = AllocateOwnerCompletionEvents(
        owner_results, owner_inputs, analyzer->SiteRegistry(),
        analyzer->SyncPoints(), flag_allocator->Allocations(),
        cross_core_flag_reservation, domains);
    if (owner_events.has_value())
      break;
    if (disable_gm_handoff("owner-event-capacity")) {
      continue;
    }
    if (!owner_inputs.empty()) {
      retry("owner-input", "owner-event-capacity");
      owner_inputs.clear();
      continue;
    }
    // Owner specialization is optional. A kernel requiring the remaining
    // result IDs keeps the normal collective implementation instead.
    ICHECK(!owner_results.empty());
    retry("owner-result", "owner-event-capacity");
    owner_results.clear();
  }
  if (auto selected = scheduled_tir.metadata.kernel_root->annotations.Get(
          kL0PhaseReusePairs)) {
    for (const Array<Var> &pair : selected.value().cast<Array<Array<Var>>>()) {
      ICHECK_EQ(pair.size(), 2u);
      bool proven =
          HasBufferAlias(analyzer->BufferAliases(), pair[0], pair[1]) &&
          HasBufferAlias(analyzer->BufferAliases(), pair[1], pair[0]);
      if (!proven && !unit_streams.empty()) {
        std::ostringstream loops;
        for (const auto &plan : unit_streams)
          loops << " " << plan.result_loop->control->loop_var->name_hint;
        LOG(WARNING) << "L0 phase alias proof failed with "
                     << unit_streams.size()
                     << " active unit streams:" << loops.str();
      }
      ICHECK(proven)
          << "Selected L0 phase reuse lacks a completion-proven physical "
             "alias between "
          << pair[0]->name_hint << " and " << pair[1]->name_hint
          << "; max-budget scheduling cannot deploy independent allocations";
    }
    scheduled_tir.metadata.kernel_root.CopyOnWrite()->annotations.erase(
        kL0PhaseReusePairs);
  }
  for (const auto &pool : operand_pools)
    diagnostics->Record("l0-operand-slot-pool", OptimizationStatus::kAccepted,
                        "completion-and-capacity-proven",
                        {pool.first.buffer->data, pool.second.buffer->data},
                        pool.first.epoch);
  auto accepted = [&](const char *feature, bool selected) {
    if (selected)
      diagnostics->Record(feature, OptimizationStatus::kAccepted,
                          "completion-and-capacity-proven");
  };
  accepted("unit-flag-stream", !unit_streams.empty());
  accepted("c-byte-retirement", closed_c_byte_retirement);
  accepted("owner-input", !owner_inputs.empty());
  accepted("owner-result", !owner_results.empty());
  accepted("gm-read-release", analyzer->HasGmReadReleases());
  accepted("gm-write-acquire", analyzer->HasGmWriteAcquires());
  scheduled_tir.metadata.root_conflict_hints = {};
  scheduled_tir.metadata.buffer_aliases = analyzer->BufferAliases();
  scheduled_tir.metadata.has_buffer_aliases = true;
  SyncMaterializer(analyzer->SiteRegistry(), analyzer->SyncPoints(),
                   flag_allocator->Allocations(), target_info.architecture,
                   cross_core_contract, lane_variables, owner_events.value(),
                   operand_pools, flag_allocator->OperandPoolBases(),
                   analyzer->GmReadReleases(), analyzer->GmWriteAcquires())
      .Run(root);
  FlagInfoList flag_infos = flag_allocator->TakeFlagInfos();

  std::vector<std::shared_ptr<IRStructure>> prefix;
  std::vector<std::shared_ptr<IRStructure>> suffix;
  for (const FlagInfo &info : flag_infos) {
    if (info.mode == 0) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendSetFlag(info.hardevent, info.flag_var), info.core_mask, 0));
      suffix.push_back(MakeSyncTaskNode(
          MakeAscendWaitFlag(info.hardevent, info.drain_flag_var.defined()
                                                 ? info.drain_flag_var
                                                 : info.flag_var),
          info.core_mask, 0));
    } else if (info.mode == 1) {
      prefix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreSetFlag(cross_core_contract.Mode(), info.hardevent,
                                     info.flag_var),
          info.core_mask, 0));
    } else if (info.mode == 2) {
      suffix.push_back(MakeSyncTaskNode(
          MakeAscendCrossCoreWaitFlag(cross_core_contract.Mode(),
                                      info.hardevent, info.flag_var),
          info.core_mask, 0));
    }
  }
  root.insert(root.begin(), prefix.begin(), prefix.end());
  root.insert(root.end(), suffix.begin(), suffix.end());
  for (size_t i = 0; i < root.size(); ++i) {
    root[i]->SetIndex(i);
    root[i]->SetParent(nullptr);
  }
  return EncodeScheduledTIR(std::move(scheduled_tir));
}

} // namespace

// ============================================================================
// Pass registration
// ============================================================================
tvm::transform::Pass InsertSync() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) -> PrimFunc {
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target_opt.defined()) << "InsertSync requires a bound Ascend target "
                                    "before completion planning";
    Target target = target_opt.value();
    std::vector<Var> lane_variables = CollectAivLaneVariables(func->body);
    return RewriteTilelangKernels(
        std::move(func), "InsertSync",
        [&](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir = DecodeScheduledTIR(
              context.root, context.outer_ctx, MakeSyncInsertionExtraInfo);
          return InsertKernelSync(context.root->body, std::move(scheduled_tir),
                                  target, lane_variables);
        });
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InsertSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InsertSync", InsertSync);
}

} // namespace tl
} // namespace tvm
