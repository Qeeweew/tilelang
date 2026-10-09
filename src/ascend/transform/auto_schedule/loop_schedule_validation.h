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

// Numeric hard contracts for a concrete loop plan, independent of its producer.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

// Half-open issue spans in the existing cost model's units. These do not
// replace completion latency. kind 0 names a core/pipe bit; kind 1 an existing
// HBM contention-policy bit. All spans of a child share its issue origin.
struct IssueResourceSpan {
  int64_t kind;
  int64_t bit;
  int64_t first;
  int64_t last;
};
using IssueResourceProfile = std::vector<IssueResourceSpan>;

struct ConcreteLoopSchedule {
  std::vector<int64_t> starts;
  std::vector<int> versions;
  int64_t ii;
};

// These are the original scheduler inputs, including already reduced memory
// capacities. Validation neither discovers dependencies nor charges storage.
// The references are local to one solving/materialization boundary.
struct LoopScheduleConstraints {
  const std::vector<int64_t> &iis;
  const std::vector<int64_t> &resource_flags;
  const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> &data_deps;
  const std::vector<std::pair<int64_t, int64_t>> &resource_deps;
  const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> &owner_deps;
  const std::vector<std::pair<int64_t, int64_t>> &pipe_order_deps;
  const std::vector<std::pair<int64_t, int64_t>> &stage_order_deps;
  const std::vector<int64_t> &storage_sizes;
  const std::vector<int64_t> &storage_min_versions;
  const std::vector<std::vector<int64_t>> &memory_groups;
  int num_stages;
  bool enable_offset;
  const std::vector<int64_t> *manual_stages{nullptr};
  std::optional<int64_t> stage_horizon{std::nullopt};
  const std::vector<IssueResourceProfile> *resource_profiles{nullptr};
  const std::vector<int64_t> *resource_hbm_masks{nullptr};
};

inline const char *
ValidateLoopScheduleInputs(const LoopScheduleConstraints &inputs) {
  using Wide = __int128;
  const size_t n = inputs.iis.size();
  if (!n || inputs.resource_flags.size() != n)
    return "invalid-task-count";
  for (size_t i = 0; i < n; ++i) {
    if (inputs.iis[i] < 0)
      return "invalid-task-cost";
    if (inputs.resource_flags[i] < 0)
      return "invalid-resource-mask";
  }
  if (inputs.storage_sizes.size() != inputs.storage_min_versions.size() ||
      inputs.num_stages < 1)
    return "invalid-storage-count";
  for (size_t id = 0; id < inputs.storage_sizes.size(); ++id) {
    if (inputs.storage_sizes[id] < 0)
      return "invalid-storage-size";
    if (inputs.storage_min_versions[id] < 1 ||
        inputs.storage_min_versions[id] > std::numeric_limits<int>::max())
      return "invalid-storage-minimum";
  }
  auto valid_task = [n](int64_t id) { return id >= 0 && Wide(id) < Wide(n); };
  for (const auto &[u, v, distance, latency] : inputs.data_deps) {
    if (!valid_task(u) || !valid_task(v))
      return "invalid-task-dependency";
    if (distance < 0 &&
        -Wide(distance) - 1 >= Wide(inputs.storage_sizes.size()))
      return "invalid-storage-dependency";
  }
  for (const auto &[u, v] : inputs.resource_deps)
    if (!valid_task(u) || !valid_task(v))
      return "invalid-resource-dependency";
  for (const auto &[u, v, forward, reverse] : inputs.owner_deps)
    if (!valid_task(u) || !valid_task(v))
      return "invalid-owner-dependency";
  for (const auto &[u, v] : inputs.pipe_order_deps)
    if (!valid_task(u) || !valid_task(v))
      return "invalid-pipe-order";
  for (const auto &[u, v] : inputs.stage_order_deps)
    if (!valid_task(u) || !valid_task(v))
      return "invalid-stage-order";
  for (const auto &group : inputs.memory_groups) {
    if (group.empty() || group.front() < 0)
      return "invalid-memory-group";
    for (size_t member = 1; member < group.size(); ++member)
      if (group[member] < 0 ||
          Wide(group[member]) >= Wide(inputs.storage_sizes.size()))
        return "invalid-memory-group";
  }
  if (inputs.manual_stages) {
    if (inputs.manual_stages->size() != n)
      return "invalid-manual-stage-count";
    for (int64_t stage : *inputs.manual_stages) {
      if (stage < 0 || stage > std::numeric_limits<int>::max())
        return "invalid-manual-stage";
      if (!inputs.enable_offset && stage != 0)
        return "nonzero-stage-without-offset";
    }
  }
  if (inputs.stage_horizon && *inputs.stage_horizon < 1)
    return "invalid-stage-horizon";
  if (inputs.resource_profiles && !inputs.resource_profiles->empty()) {
    if (!inputs.enable_offset || inputs.manual_stages)
      return "hierarchical-resource-admission";
    if (inputs.resource_profiles->size() != n || !inputs.resource_hbm_masks ||
        inputs.resource_hbm_masks->size() != n)
      return "invalid-resource-profile-count";
    constexpr int64_t scalar_bits = 64 | (64 << 7);
    for (size_t i = 0; i < n; ++i) {
      int64_t pipes = 0, hbm = 0;
      std::map<std::pair<int64_t, int64_t>, bool> keys;
      for (const IssueResourceSpan &span : (*inputs.resource_profiles)[i]) {
        if ((span.kind != 0 && span.kind != 1) || span.bit <= 0 ||
            (span.bit & (span.bit - 1)) != 0 || span.first < 0 ||
            span.last <= span.first || span.last > inputs.iis[i] ||
            (span.kind == 1 && span.bit != 1 && span.bit != 2 && span.bit != 4))
          return "invalid-resource-profile";
        if (!keys.emplace(std::make_pair(span.kind, span.bit), true).second)
          return "duplicate-resource-profile";
        (span.kind == 0 ? pipes : hbm) |= span.bit;
      }
      int64_t expected_hbm = (*inputs.resource_hbm_masks)[i];
      if (expected_hbm < 0 || (expected_hbm & ~int64_t(7)) != 0)
        return "invalid-resource-hbm-mask";
      if (pipes != (inputs.resource_flags[i] & ~scalar_bits) ||
          hbm != expected_hbm)
        return "incomplete-resource-profile";
    }
  }
  return nullptr;
}

// Validate original hard constraints against actual emitted versions. A
// successful result is a numeric contract, not an optimization or HB proof.
inline const char *ValidateLoopSchedule(const LoopScheduleConstraints &inputs,
                                        const ConcreteLoopSchedule &plan) {
  using Wide = __int128;
  if (const char *reason = ValidateLoopScheduleInputs(inputs))
    return reason;
  const size_t n = inputs.iis.size();
  if (plan.starts.size() != n)
    return "invalid-task-count";
  if (plan.versions.size() != inputs.storage_min_versions.size())
    return "invalid-storage-count";
  if (plan.ii <= 0)
    return "invalid-ii";
  const Wide ii = plan.ii;
  std::vector<int64_t> stages, residues;
  stages.reserve(n);
  residues.reserve(n);
  for (int64_t start : plan.starts) {
    if (start < 0)
      return "negative-start";
    int64_t stage = start / plan.ii;
    if (stage > std::numeric_limits<int>::max())
      return "stage-overflow";
    if (!inputs.manual_stages && inputs.stage_horizon &&
        stage >= *inputs.stage_horizon)
      return "stage-horizon";
    stages.push_back(stage);
    residues.push_back(start % plan.ii);
  }
  if (inputs.manual_stages) {
    // Partial completion subtracts a common whole-period origin; complete
    // source-order completion retains the requested absolute stages.
    Wide origin = Wide(stages[0]) - (*inputs.manual_stages)[0];
    for (size_t i = 0; i < n; ++i) {
      if (Wide(stages[i]) - (*inputs.manual_stages)[i] != origin)
        return "manual-stage-origin";
      // Materialization retains requested stages, including after a partial
      // producer translates its coordinates by whole periods.
      if (Wide((*inputs.manual_stages)[i]) * ii >
          std::numeric_limits<int64_t>::max())
        return "manual-stage-coordinate-overflow";
    }
  }
  for (size_t id = 0; id < plan.versions.size(); ++id) {
    if (plan.versions[id] < inputs.storage_min_versions[id])
      return "version-minimum";
    if (plan.versions[id] >
        std::max<int64_t>(inputs.num_stages, inputs.storage_min_versions[id]))
      return "version-bound";
  }
  for (const auto &group : inputs.memory_groups) {
    Wide bytes = 0;
    for (size_t member = 1; member < group.size(); ++member) {
      size_t id = static_cast<size_t>(group[member]);
      bytes += Wide(inputs.storage_sizes[id]) * plan.versions[id];
      if (bytes > Wide(group.front()))
        return "memory-capacity";
    }
  }
  for (const auto &[u, v, distance, latency] : inputs.data_deps) {
    Wide resolved_distance =
        distance >= 0 ? Wide(distance)
                      : plan.versions[static_cast<size_t>(-Wide(distance) - 1)];
    if (Wide(plan.starts[v]) - plan.starts[u] + ii * resolved_distance <
        Wide(latency))
      return "data-dependency";
  }
  const bool profiled =
      inputs.resource_profiles && !inputs.resource_profiles->empty();
  if (profiled)
    for (const auto &profile : *inputs.resource_profiles)
      for (const IssueResourceSpan &span : profile)
        if (Wide(span.last) - span.first > ii)
          return "hierarchical-resource-self";
  for (const auto &[u, v] : inputs.resource_deps) {
    bool common = false;
    if (profiled) {
      bool before =
          residues[u] < residues[v] ||
          (residues[u] == residues[v] &&
           (stages[u] > stages[v] || (stages[u] == stages[v] && u < v)));
      for (const IssueResourceSpan &a : (*inputs.resource_profiles)[u])
        for (const IssueResourceSpan &b : (*inputs.resource_profiles)[v]) {
          if (a.kind != b.kind || a.bit != b.bit)
            continue;
          common = true;
          Wide forward =
              before ? Wide(residues[v]) + b.first - residues[u] - a.last
                     : Wide(residues[u]) + a.first - residues[v] - b.last;
          Wide wrap = before
                          ? ii + residues[u] + a.first - residues[v] - b.last
                          : ii + residues[v] + b.first - residues[u] - a.last;
          if (forward < 0 || wrap < 0)
            return "hierarchical-resource-order";
        }
    }
    // Supplied opaque conflicts without a common profile key retain the
    // original envelope; short issue spans must not erase them.
    if (!common) {
      Wide difference = Wide(residues[v]) - residues[u];
      if (!((difference >= inputs.iis[u] && ii - difference >= inputs.iis[v]) ||
            (-difference >= inputs.iis[v] && ii + difference >= inputs.iis[u])))
        return "resource-wrap";
    }
  }
  // Repeated tuples with the same ordered endpoints share one Python
  // orientation Boolean. Per-tuple OR would admit incompatible directions.
  std::map<std::pair<int64_t, int64_t>, std::pair<bool, bool>> owner_orders;
  for (const auto &[u, v, forward, reverse] : inputs.owner_deps) {
    auto entry =
        owner_orders.emplace(std::make_pair(u, v), std::make_pair(true, true))
            .first;
    Wide difference = Wide(residues[v]) - residues[u];
    entry->second.first &=
        difference >= Wide(forward) && ii - difference >= Wide(reverse);
    entry->second.second &=
        -difference >= Wide(reverse) && ii + difference >= Wide(forward);
  }
  for (const auto &entry : owner_orders)
    if (!entry.second.first && !entry.second.second)
      return "owner-wrap";
  for (const auto &[u, v] : inputs.pipe_order_deps)
    if (residues[u] > residues[v])
      return "pipe-order";
  for (const auto &[u, v] : inputs.stage_order_deps)
    if (stages[u] > stages[v])
      return "let-stage-order";
  // Validate the returned scheduling coordinates before no-offset emission
  // collapses stages. Masks do not create any new exclusion pairs.
  if (!inputs.enable_offset)
    for (size_t u = 0; u < n; ++u)
      for (size_t v = u + 1; v < n; ++v)
        if ((inputs.resource_flags[u] & inputs.resource_flags[v]) != 0 &&
            (Wide(plan.starts[u]) - plan.starts[v] >= ii ||
             Wide(plan.starts[v]) - plan.starts[u] >= ii))
          return "resource-offset";
  return nullptr;
}

} // namespace tvm::tl::ascend
