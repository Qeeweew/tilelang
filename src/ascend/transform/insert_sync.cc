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
#include "./auto_schedule/kernel_rewriter.h"
#include "./auto_schedule/multi_buffer.h"
#include "./auto_schedule/owner_result.h"
#include "./auto_schedule/scheduled_tir.h"
#include "./auto_schedule/task_analysis.h"
#include "./auto_schedule/task_annotations.h"
#include "ascend/op/builtin.h"
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
      const std::vector<OwnerResultPlan> *owner_results = nullptr)
      : storage_groups_(std::move(storage_groups)),
        multi_buffer_plan_(multi_buffer_plan), domains_(domains),
        architecture_(architecture), manual_(std::move(manual)),
        root_conflicts_(std::move(root_conflicts)),
        owner_results_(owner_results), site_registry_(domains) {
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
    scope_closures_.clear();
    lifetime_orders_.clear();
    SyncAnalysisResult root_result = AnalyzeNodeList(nodes, /*loop=*/nullptr);
    scope_closures_.insert_or_assign(nullptr, std::move(root_result.closure));
    if (owner_fallback_required_)
      return false;

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
    return true;
  }

  const SyncSiteRegistry &SiteRegistry() const { return site_registry_; }

  std::vector<SyncPoint> &SyncPoints() { return sync_points_; }

  const std::set<std::pair<size_t, size_t>> &ReusablePairs() const {
    return reusable_pairs_;
  }

  const BufferAliasMap &BufferAliases() const { return buffer_aliases_; }

  bool HasOwnerResultRelease(const OwnerResultPlan &plan) const {
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

private:
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
    return endpoints;
  }

  size_t PromoteLifetimeSite(const Var &storage, size_t site_id,
                             ControlNode *control) const final {
    ICHECK_LT(site_id, site_registry_.Size());
    int parent_domain_id =
        domains_.DomainForStorage(storage, control->GetParentControl());
    size_t parent_site =
        site_registry_.FindParentSiteInDomain(site_id, parent_domain_id);
    ICHECK_EQ(site_registry_.Sites()[parent_site].path_begin, control);
    return parent_site;
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
      AddPromotedChildClosure(result.closure, node, child_result.closure, loop);
      child_sync_points.push_back(std::move(child_result.visible_sync_points));
    }

    std::vector<SyncPoint> local_sync_points = CollectSyncPoints(nodes, loop);
    int max_distance = loop == nullptr ? 0 : 1;
    for (const SyncPoint &sync_point : local_sync_points)
      max_distance = std::max(max_distance, sync_point.edge.distance);

    AddSamePipeEdges(result.closure, task_site_ids, max_distance, loop);
    OptimizeSyncPoints(local_sync_points, result.closure, max_distance, loop);
    if (loop != nullptr)
      OptimizeFlagVersions(local_sync_points, result.closure, loop);

    std::vector<VisibleSyncPoint> local_points;
    local_points.reserve(local_sync_points.size());
    for (SyncPoint &sync_point : local_sync_points) {
      size_t sync_point_id = AppendSyncPoint(std::move(sync_point));
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
        int pair_domain_id =
            dependency_domain_id.has_value()
                ? dependency_domain_id.value()
                : FindSharedDependencyDomain(producer, consumer, loop);
        std::string producer_pipe =
            GetResourcePipeName(producer->GetPipeMask());
        std::string consumer_pipe =
            GetResourcePipeName(consumer->GetPipeMask());
        std::string hard_event = GetHardEvent(producer_pipe, consumer_pipe);
        // Preserve the legacy unit-flag replacement only for L0C data
        // handshakes between MAD and FIX. Unit flags do not establish SPR
        // state, other storage hazards, or complete owner lifetimes.
        bool fix_unit_flag =
            dep.kind == DependencyKind::kData && dep.storage.has_value() &&
            (storage_scope == "shared.l0c" ||
             storage_scope == "shared.l0c.dyn") &&
            (hard_event == "M_FIX" || hard_event == "FIX_M") &&
            (HasUnitFlagCtrl(producer->stmt) ||
             HasUnitFlagCtrl(consumer->stmt));
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
          }
        }
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

  bool CoversLocalProtocol(const SyncPoint &wide, const SyncPoint &narrow,
                           ControlNode *loop) const {
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
      if (closure.Contains(sync_point.edge))
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

  void OptimizeFlagVersions(std::vector<SyncPoint> &sync_points,
                            const DepClosure &closure, ControlNode *loop) {
    for (SyncPoint &sync_point : sync_points) {
      const SyncInsertionSite &src = site_registry_.Src(sync_point);
      const SyncInsertionSite &dst = site_registry_.Dst(sync_point);
      if (src.pipe == dst.pipe)
        continue;
      bool is_innermost = loop == sync_point.flag_loop;
      int innermost_num_versions = is_innermost ? sync_point.num_versions : 1;
      bool local_binary = architecture_ == AscendArchitecture::kDav2201 &&
                          !site_registry_.IsCrossCore(sync_point);
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

      if (local_binary && min_distance == std::numeric_limits<int>::max() &&
          owner_results_ && !owner_results_->empty()) {
        // The old pair-ready may have supplied this unrelated ring's only
        // reverse-completion witness. Do not use an owner edge as its proof
        // or weaken binary validation; retry the normal collective analysis.
        owner_fallback_required_ = true;
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
        ICHECK(is_innermost && sync_point.counter_group_id == 0 &&
               loop != nullptr && loop->GetParentControl() == nullptr &&
               !loop->HasConditions() && !loop->BodyHasLoopBreak())
            << "dav-2201 local binary flag: shifted carried events require "
               "an unconditional outermost lexical loop"
            << binary_context();
        const ForNode *for_node = loop->control.get();
        PrimExpr step = for_node->step.has_value()
                            ? for_node->step.value()
                            : IntImm(for_node->extent.dtype(), 1);
        ICHECK(loop->task->outer_ctx.CanProve(step > make_zero(step.dtype())))
            << "dav-2201 local binary flag: loop step must be positive";
        ICHECK(SideEffect(for_node->min) == CallEffectKind::kPure &&
               SideEffect(for_node->extent) == CallEffectKind::kPure &&
               SideEffect(step) == CallEffectKind::kPure)
            << "dav-2201 local binary flag: loop bounds must be immutable";
        PrimExpr trip_count = max(ceildiv(for_node->extent, step),
                                  make_zero(for_node->extent.dtype()));
        ICHECK(sync_point.flag_iteration.dtype().is_int() &&
               trip_count.dtype().is_int())
            << "dav-2201 local binary flag: shifted epoch clocks require "
               "signed integer iteration and trip-count types";
        PrimExpr lexical_ordinal =
            indexdiv(for_node->loop_var - for_node->min, step);
        ICHECK(loop->GetLoopBodyContext().CanProve(sync_point.flag_iteration ==
                                                   lexical_ordinal))
            << "dav-2201 local binary flag: shifted epoch clock must equal "
               "the root lexical iteration ordinal"
            << binary_context();
        std::optional<LocalBinaryEpochSpan> active_span =
            LocalBinaryActiveSpan(sync_point, loop, trip_count, step);
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
          ICHECK(finite_width.has_value())
              << "dav-2201 local binary flag: no reverse completion path for "
              << src.pipe << "->" << dst.pipe
              << "; the whole lexical clock also exceeds the finite "
                 "non-reusing event capacity";
          sync_point.num_versions = *finite_width;
        } else {
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
            << src.pipe << "->" << dst.pipe;
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
    // Existing reuse proofs compare equal event phases. Keep a shifted local
    // binary channel exclusive rather than sharing its partial seed/drain.
    if (lhs_sync.local_binary_plan || rhs_sync.local_binary_plan)
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
  SyncSiteRegistry site_registry_;
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
                const CrossCoreSyncContract &cross_core_contract)
      : site_registry_(site_registry), sync_points_(sync_points),
        reusable_pairs_(reusable_pairs),
        cross_core_flag_reservation_(cross_core_flag_reservation),
        cross_core_contract_(cross_core_contract) {}

  bool Plan(bool allow_fallback = false) {
    allocations_.clear();
    flag_infos_.clear();
    GroupCounterChannels();

    std::map<std::string, int> required_ids;
    std::map<std::string, int> next_intra_core_id;
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
  auto task = std::make_shared<TaskNode>();
  task->stmt = MakeSyncTask(body, core_mask);
  task->SetStage(stage);
  task->SetCoreMask(core_mask);
  task->SetPipeMask(static_cast<uint16_t>(ResourcePipe::kScalar));
  return task;
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

// Result readiness has an owner selector in addition to the existing storage
// epoch. It deliberately has no all-participant closure projection or flag
// reuse: each owner/GM-slot pair owns one result ID. The normal collective
// reader release proves consumption before this slot and ID can be reused.
struct OwnerResultEvent {
  SyncInsertionSite producer;
  SyncInsertionSite consumer;
  PrimExpr producer_id;
  PrimExpr consumer_id;
};

std::optional<std::vector<OwnerResultEvent>>
AllocateOwnerResultEvents(const std::vector<OwnerResultPlan> &plans,
                          const SyncSiteRegistry &site_registry,
                          const std::vector<SyncPoint> &sync_points,
                          const std::vector<FlagAllocation> &allocations,
                          const CrossCoreFlagReservation &reservation,
                          const EpochDomainRegistry &domains) {
  if (plans.empty())
    return std::vector<OwnerResultEvent>{};
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
  std::vector<OwnerResultEvent> result;
  for (const OwnerResultPlan &plan : plans) {
    int width = 2 * plan.num_slots;
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
    if (base < 0)
      return std::nullopt;
    for (int offset = 0; offset < width; ++offset)
      used[base + offset] = true;
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
                      cast(DataType::Int(32), consumer_id)});
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
                   const std::vector<OwnerResultEvent> &owner_results)
      : site_registry_(site_registry), sync_points_(sync_points),
        allocations_(allocations), architecture_(architecture),
        cross_core_contract_(cross_core_contract),
        lane_variables_(lane_variables), owner_results_(owner_results) {}

  void Run(std::vector<std::shared_ptr<IRStructure>> &root) {
    InsertSyncPoints();
    for (const OwnerResultEvent &event : owner_results_) {
      ICHECK(architecture_ == AscendArchitecture::kDav2201);
      ICHECK_EQ(cross_core_contract_.num_aiv_subcores, 2);
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
  const std::vector<OwnerResultEvent> &owner_results_;
};

// ============================================================================
// Kernel-level validation and scheduled-TIR rewrite
// ============================================================================
SBlock InsertKernelSync(const Stmt &kernel_body, ScheduledTIR scheduled_tir,
                        Target target, const std::vector<Var> &lane_variables) {
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
  if (cross_core_contract.collective &&
      cross_core_contract.num_aiv_subcores == 2) {
    owner_results = AnalyzeOwnerResults(
        root, scheduled_tir.metadata.kernel_root,
        scheduled_tir.metadata.manual_buffer_versions, lane_variables);
  }
  std::unique_ptr<SyncAnalyzer> analyzer;
  std::unique_ptr<FlagAllocator> flag_allocator;
  std::optional<std::vector<OwnerResultEvent>> owner_events;
  for (;;) {
    flag_allocator.reset();
    analyzer = std::make_unique<SyncAnalyzer>(
        multi_buffer_plan, domains, target_info.architecture,
        scheduled_tir.metadata.manual_buffer_versions,
        scheduled_tir.metadata.root_conflict_hints, groups, &owner_results);
    if (!analyzer->Analyze(root)) {
      ICHECK(!owner_results.empty());
      owner_results.clear();
      continue;
    }
    size_t candidate_count = owner_results.size();
    owner_results.erase(
        std::remove_if(owner_results.begin(), owner_results.end(),
                       [&](const OwnerResultPlan &plan) {
                         return !analyzer->HasOwnerResultRelease(plan);
                       }),
        owner_results.end());
    if (owner_results.size() != candidate_count)
      continue;
    flag_allocator = std::make_unique<FlagAllocator>(
        analyzer->SiteRegistry(), analyzer->SyncPoints(),
        analyzer->ReusablePairs(), cross_core_flag_reservation,
        cross_core_contract);
    if (!flag_allocator->Plan(!owner_results.empty())) {
      owner_results.clear();
      continue;
    }
    owner_events = AllocateOwnerResultEvents(
        owner_results, analyzer->SiteRegistry(), analyzer->SyncPoints(),
        flag_allocator->Allocations(), cross_core_flag_reservation, domains);
    if (owner_events.has_value())
      break;
    // Owner specialization is optional. A kernel requiring the remaining
    // result IDs keeps the normal collective implementation instead.
    ICHECK(!owner_results.empty());
    owner_results.clear();
  }
  scheduled_tir.metadata.root_conflict_hints = {};
  scheduled_tir.metadata.buffer_aliases = analyzer->BufferAliases();
  scheduled_tir.metadata.has_buffer_aliases = true;
  SyncMaterializer(analyzer->SiteRegistry(), analyzer->SyncPoints(),
                   flag_allocator->Allocations(), target_info.architecture,
                   cross_core_contract, lane_variables, owner_events.value())
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
