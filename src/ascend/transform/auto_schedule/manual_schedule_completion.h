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

// Deterministic completion for an explicitly fixed global issue order.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

constexpr const char *kPreserveIssueOrder = "tl.ascend_preserve_issue_order";

struct ManualScheduleCompletion {
  std::vector<int64_t> starts;
  std::vector<int> versions;
  int64_t ii;
};

struct ManualCompletionResult {
  std::optional<ManualScheduleCompletion> plan;
  const char *reason;
};

// These coordinates establish stage and residue order, not measured issue
// cycles. Fixed stages/order make each unknown ring count a one-dimensional
// integer lower bound; no II, order, or capacity search is performed. A cost
// summary must replace them before an enclosing sequence estimates
// dependencies.
inline ManualCompletionResult CompleteManualSourceOrder(
    const std::vector<int64_t> &stages, const std::vector<int64_t> &latencies,
    const std::vector<int64_t> &iis, const std::vector<int64_t> &resource_flags,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        &data_deps,
    const std::vector<std::pair<int64_t, int64_t>> &resource_deps,
    const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>
        &owner_deps,
    const std::vector<std::pair<int64_t, int64_t>> &pipe_order_deps,
    const std::vector<std::pair<int64_t, int64_t>> &stage_order_deps,
    bool enable_offset, const std::vector<int64_t> &storage_sizes,
    const std::vector<int64_t> &storage_min_versions,
    const std::vector<std::vector<int64_t>> &memory_groups, int num_stages) {
  using Wide = __int128;
  const size_t n = stages.size();
  if (!n || latencies.size() != n || iis.size() != n ||
      resource_flags.size() != n)
    return {std::nullopt, "invalid-task-count"};
  Wide quantum = 1;
  for (size_t i = 0; i < n; ++i) {
    if (stages[i] < 0 || latencies[i] < 0 || iis[i] <= 0)
      return {std::nullopt, "invalid-task-stage-or-cost"};
    if (!enable_offset && stages[i] != 0)
      return {std::nullopt, "nonzero-stage-without-offset"};
    quantum = std::max(quantum, Wide(std::max(latencies[i], iis[i])) + 1);
  }
  for (const auto &[u, v, distance, latency] : data_deps)
    quantum = std::max(quantum, Wide(latency) + 1);
  for (const auto &[u, v, forward, reverse] : owner_deps)
    quantum = std::max(quantum, Wide(std::max(forward, reverse)) + 1);
  Wide ii = (Wide(n) + 1) * quantum;
  if (ii > std::numeric_limits<int64_t>::max())
    return {std::nullopt, "coordinate-overflow"};
  ManualScheduleCompletion plan{{}, {}, static_cast<int64_t>(ii)};
  std::vector<Wide> residues;
  for (size_t i = 0; i < n; ++i) {
    Wide residue = Wide(i) * quantum;
    Wide start = Wide(stages[i]) * ii + residue;
    if (start > std::numeric_limits<int64_t>::max())
      return {std::nullopt, "coordinate-overflow"};
    plan.starts.push_back(static_cast<int64_t>(start));
    residues.push_back(residue);
  }
  if (storage_sizes.size() != storage_min_versions.size() || num_stages < 1)
    return {std::nullopt, "invalid-storage-count"};
  for (int64_t minimum : storage_min_versions) {
    if (minimum < 1 || minimum > std::numeric_limits<int>::max())
      return {std::nullopt, "invalid-storage-minimum"};
    plan.versions.push_back(static_cast<int>(minimum));
  }
  // With start and II fixed, each negative-distance edge is precisely
  // versions[id] >= ceil((latency - delta_start) / II). Taking the maximum
  // over its edges is the least legal integer completion. Existing shared
  // rings and explicit fixed counts have already entered the common inputs.
  for (const auto &[u, v, distance, latency] : data_deps) {
    if (distance >= 0)
      continue;
    Wide id = -Wide(distance) - 1;
    if (id >= Wide(plan.versions.size()))
      return {std::nullopt, "invalid-storage-dependency"};
    Wide numerator = Wide(latency) - (Wide(plan.starts[v]) - plan.starts[u]);
    Wide required = numerator / ii + (numerator > 0 && numerator % ii != 0);
    if (required > std::numeric_limits<int>::max())
      return {std::nullopt, "source-order-version-overflow"};
    if (required > plan.versions[static_cast<size_t>(id)])
      plan.versions[static_cast<size_t>(id)] = static_cast<int>(required);
  }
  for (size_t id = 0; id < plan.versions.size(); ++id)
    if (plan.versions[id] >
        std::max<int64_t>(num_stages, storage_min_versions[id]))
      return {std::nullopt, "source-order-version-bound"};
  for (const auto &group : memory_groups) {
    if (group.empty())
      return {std::nullopt, "invalid-memory-group"};
    Wide bytes = 0;
    for (size_t member = 1; member < group.size(); ++member) {
      int64_t id = group[member];
      if (id < 0 || static_cast<size_t>(id) >= plan.versions.size() ||
          storage_sizes[id] < 0)
        return {std::nullopt, "invalid-memory-group"};
      bytes += Wide(storage_sizes[id]) * plan.versions[id];
    }
    if (bytes > group.front())
      return {std::nullopt, "source-order-memory-capacity"};
  }
  // Recheck all hard edges using the completed counts, including fixed
  // positive distances. Failure never changes the requested source order.
  for (const auto &[u, v, distance, latency] : data_deps) {
    Wide resolved_distance =
        distance >= 0 ? Wide(distance)
                      : plan.versions[static_cast<size_t>(-Wide(distance) - 1)];
    if (Wide(plan.starts[v]) - plan.starts[u] <
        Wide(latency) - ii * resolved_distance)
      return {std::nullopt, "source-order-data-dependency"};
  }
  for (const auto &[u, v] : resource_deps) {
    Wide difference = residues[v] - residues[u];
    if (!((difference >= iis[u] && ii - difference >= iis[v]) ||
          (-difference >= iis[v] && ii + difference >= iis[u])))
      return {std::nullopt, "source-order-resource-wrap"};
  }
  for (const auto &[u, v, forward, reverse] : owner_deps) {
    Wide difference = residues[v] - residues[u];
    if (!((difference >= forward && ii - difference >= reverse) ||
          (-difference >= reverse && ii + difference >= forward)))
      return {std::nullopt, "source-order-owner-wrap"};
  }
  for (const auto &[u, v] : pipe_order_deps)
    if (residues[u] > residues[v])
      return {std::nullopt, "source-order-pipe-order"};
  for (const auto &[u, v] : stage_order_deps)
    if (stages[u] > stages[v])
      return {std::nullopt, "source-order-let-stage"};
  if (!enable_offset)
    for (size_t u = 0; u < n; ++u)
      for (size_t v = u + 1; v < n; ++v)
        if ((resource_flags[u] & resource_flags[v]) != 0 &&
            (Wide(plan.starts[u]) - plan.starts[v] >= ii ||
             Wide(plan.starts[v]) - plan.starts[u] >= ii))
          return {std::nullopt, "source-order-resource-offset"};
  return {std::move(plan), "fixed-source-order"};
}

} // namespace tvm::tl::ascend
