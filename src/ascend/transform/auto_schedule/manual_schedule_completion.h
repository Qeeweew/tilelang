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

#include "loop_schedule_validation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

constexpr const char *kPreserveIssueOrder = "tl.ascend_preserve_issue_order";

using ManualScheduleCompletion = ConcreteLoopSchedule;

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
  LoopScheduleConstraints constraints{iis,
                                      resource_flags,
                                      data_deps,
                                      resource_deps,
                                      owner_deps,
                                      pipe_order_deps,
                                      stage_order_deps,
                                      storage_sizes,
                                      storage_min_versions,
                                      memory_groups,
                                      num_stages,
                                      enable_offset,
                                      &stages};
  // Check all indices before the least-version completion indexes starts or
  // storage. The same input contract is used by automatic plan validation.
  if (const char *reason = ValidateLoopScheduleInputs(constraints))
    return {std::nullopt, reason};
  for (const auto &[u, v, distance, latency] : data_deps)
    quantum = std::max(quantum, Wide(latency) + 1);
  for (const auto &[u, v, forward, reverse] : owner_deps)
    quantum = std::max(quantum, Wide(std::max(forward, reverse)) + 1);
  Wide ii = (Wide(n) + 1) * quantum;
  if (ii > std::numeric_limits<int64_t>::max())
    return {std::nullopt, "coordinate-overflow"};
  ManualScheduleCompletion plan{{}, {}, static_cast<int64_t>(ii)};
  for (size_t i = 0; i < n; ++i) {
    Wide residue = Wide(i) * quantum;
    Wide start = Wide(stages[i]) * ii + residue;
    if (start > std::numeric_limits<int64_t>::max())
      return {std::nullopt, "coordinate-overflow"};
    plan.starts.push_back(static_cast<int64_t>(start));
  }
  for (int64_t minimum : storage_min_versions)
    plan.versions.push_back(static_cast<int>(minimum));
  // With start and II fixed, each negative-distance edge is precisely
  // versions[id] >= ceil((latency - delta_start) / II). Taking the maximum
  // over its edges is the least legal integer completion. Existing shared
  // rings and explicit fixed counts have already entered the common inputs.
  for (const auto &[u, v, distance, latency] : data_deps) {
    if (distance >= 0)
      continue;
    Wide id = -Wide(distance) - 1;
    Wide numerator = Wide(latency) - (Wide(plan.starts[v]) - plan.starts[u]);
    Wide required = numerator / ii + (numerator > 0 && numerator % ii != 0);
    if (required > std::numeric_limits<int>::max())
      return {std::nullopt, "source-order-version-overflow"};
    if (required > plan.versions[static_cast<size_t>(id)])
      plan.versions[static_cast<size_t>(id)] = static_cast<int>(required);
  }
  // Both producers now cross the same hard-contract boundary. Preserve the
  // existing explicit-order diagnostic vocabulary for its callers.
  if (const char *reason = ValidateLoopSchedule(constraints, plan)) {
    const std::pair<const char *, const char *> manual_reasons[] = {
        {"version-bound", "source-order-version-bound"},
        {"memory-capacity", "source-order-memory-capacity"},
        {"data-dependency", "source-order-data-dependency"},
        {"resource-wrap", "source-order-resource-wrap"},
        {"owner-wrap", "source-order-owner-wrap"},
        {"pipe-order", "source-order-pipe-order"},
        {"let-stage-order", "source-order-let-stage"},
        {"resource-offset", "source-order-resource-offset"}};
    for (const auto &[common, manual] : manual_reasons)
      if (std::string_view(reason) == common)
        return {std::nullopt, manual};
    return {std::nullopt, reason};
  }
  return {std::move(plan), "fixed-source-order"};
}

} // namespace tvm::tl::ascend
