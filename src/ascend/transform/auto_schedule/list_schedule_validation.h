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

// Numeric hard contracts for a concrete straight-line sibling schedule.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

// Original scheduler inputs. Validation does not rediscover dependencies,
// infer exclusions from masks, or repeat the producer's search reductions.
struct ListScheduleConstraints {
  const std::vector<int64_t> &latencies;
  const std::vector<int64_t> &iis;
  const std::vector<std::tuple<int64_t, int64_t, int64_t>> &data_deps;
  const std::vector<std::pair<int64_t, int64_t>> &resource_deps;
  const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> &owner_deps;
  const std::vector<std::pair<int64_t, int64_t>> &pipe_order_deps;
};

inline const char *ValidateListSchedule(const ListScheduleConstraints &inputs,
                                        const std::vector<int64_t> &starts) {
  using Wide = __int128;
  const size_t n = inputs.latencies.size();
  if (starts.size() != n || inputs.iis.size() != n)
    return "invalid-task-count";
  for (size_t i = 0; i < n; ++i) {
    if (inputs.latencies[i] < 0 || inputs.iis[i] < 0)
      return "invalid-task-cost";
    if (starts[i] < 0)
      return "negative-start";
    // Native timestamps and completion coordinates use int64_t, whereas
    // the Python producer's integer arithmetic is unbounded.
    if (Wide(starts[i]) + std::max(inputs.latencies[i], inputs.iis[i]) >
        std::numeric_limits<int64_t>::max())
      return "completion-overflow";
  }
  auto valid_task = [n](int64_t id) { return id >= 0 && Wide(id) < Wide(n); };
  for (const auto &[u, v, latency] : inputs.data_deps) {
    if (!valid_task(u) || !valid_task(v))
      return "invalid-task-dependency";
    if (Wide(starts[v]) - starts[u] < Wide(latency))
      return "data-dependency";
  }
  for (const auto &[u, v] : inputs.resource_deps) {
    if (!valid_task(u) || !valid_task(v))
      return "invalid-resource-dependency";
    Wide gap = Wide(starts[v]) - starts[u];
    if (!(gap >= inputs.iis[u] || -gap >= inputs.iis[v]))
      return "resource-overlap";
  }
  // Repeated tuples with the same ordered endpoints share one orientation
  // Boolean in the producer. One independent OR per tuple is weaker.
  std::map<std::pair<int64_t, int64_t>, std::pair<bool, bool>> owner_orders;
  for (const auto &[u, v, forward, reverse] : inputs.owner_deps) {
    if (!valid_task(u) || !valid_task(v))
      return "invalid-owner-dependency";
    auto entry =
        owner_orders.emplace(std::make_pair(u, v), std::make_pair(true, true))
            .first;
    Wide gap = Wide(starts[v]) - starts[u];
    entry->second.first &= gap >= Wide(forward);
    entry->second.second &= -gap >= Wide(reverse);
  }
  for (const auto &entry : owner_orders)
    if (!entry.second.first && !entry.second.second)
      return "owner-exclusion";
  for (const auto &[u, v] : inputs.pipe_order_deps) {
    if (!valid_task(u) || !valid_task(v))
      return "invalid-pipe-order";
    if (starts[u] > starts[v])
      return "pipe-order";
  }
  return nullptr;
}

} // namespace tvm::tl::ascend
