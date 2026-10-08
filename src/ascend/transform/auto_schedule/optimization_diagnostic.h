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

// Optional optimization diagnostics are observations, never proof inputs.
#pragma once

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "./ir_structure.h"

namespace tvm::tl::ascend {

enum class OptimizationStatus { kSelected, kRejected, kRetry, kAccepted };

inline const char *OptimizationStatusName(OptimizationStatus status) {
  switch (status) {
  case OptimizationStatus::kSelected:
    return "selected";
  case OptimizationStatus::kRejected:
    return "rejected";
  case OptimizationStatus::kRetry:
    return "retry";
  case OptimizationStatus::kAccepted:
    return "accepted";
  }
  ICHECK(false);
  return "unknown";
}

struct OptimizationDiagnostic {
  std::string feature;
  OptimizationStatus status;
  std::string reason;
  std::vector<Var> resources;
  std::string epoch;
};

// A short-lived diagnostic sink. It stores no IRStructure, sync site, alias,
// or completion certificate; a retry can discard and rebuild all such facts.
// Default compilation produces no persistent annotation or runtime behavior.
class OptimizationDiagnostics {
public:
  void Record(const char *feature, OptimizationStatus status,
              const char *reason, std::vector<Var> resources = {},
              const ControlNode *epoch = nullptr) {
    if (!runtime::detail::VerboseLoggingEnabled(
            "ascend/transform/auto_schedule/optimization_diagnostic.h", 1))
      return;
    OptimizationDiagnostic decision{
        feature, status, reason, std::move(resources),
        epoch ? std::string(epoch->control->loop_var->name_hint) : ""};
    {
      std::ostringstream text;
      text << "Ascend optimization feature=" << decision.feature
           << " status=" << OptimizationStatusName(decision.status)
           << " reason=" << decision.reason;
      if (!decision.epoch.empty())
        text << " epoch=" << decision.epoch;
      for (const Var &resource : decision.resources)
        text << " storage=" << resource->name_hint;
      LOG(INFO) << text.str();
    }
    decisions_.push_back(std::move(decision));
  }

  const std::vector<OptimizationDiagnostic> &Decisions() const {
    return decisions_;
  }

private:
  std::vector<OptimizationDiagnostic> decisions_;
};

} // namespace tvm::tl::ascend
