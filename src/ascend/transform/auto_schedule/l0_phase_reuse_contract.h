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

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <tvm/ffi/extra/structural_equal.h>

#include "../buffer_alias.h"
#include "./dependency_analysis.h"

namespace tvm::tl::ascend {

// Root Array<Array<Var>>, with two storage Vars per selected reuse pair.
// Selection is an obligation to prove physical aliasing, never its proof.
constexpr const char *kL0PhaseReusePairs = "tl.ascend_l0_phase_reuse_pairs";

// A snapshot of the early max-budget choice across scheduled-TIR rewrites.
// Storage identities and pair order survive those rewrites; early epoch/phase
// pointers do not. Keep marker presence distinct from an empty selection.
class SelectedL0PhaseReuse {
public:
  using Pair = std::array<tirx::Var, 2>;

  SelectedL0PhaseReuse() = default;
  explicit SelectedL0PhaseReuse(const ffi::Array<ffi::Array<tirx::Var>> &pairs)
      : present_(true) {
    for (const auto &pair : pairs) {
      ICHECK_EQ(pair.size(), 2u)
          << "Selected L0 phase reuse requires two storage Vars per pair";
      pairs_.push_back({pair[0], pair[1]});
      conflicts_.push_back(
          ffi::Array<ffi::Any>{pair[0], pair[1], Integer(-1), Bool(true)});
    }
  }

  static SelectedL0PhaseReuse Read(const tirx::SBlock &root) {
    if (auto pairs = root->annotations.Get(kL0PhaseReusePairs))
      return SelectedL0PhaseReuse(
          pairs.value().cast<ffi::Array<ffi::Array<tirx::Var>>>());
    return {};
  }

  bool Present() const { return present_; }
  const std::vector<Pair> &Pairs() const { return pairs_; }

  ffi::Array<ffi::Array<tirx::Var>> Encode() const {
    ICHECK(present_);
    ffi::Array<ffi::Array<tirx::Var>> result;
    for (const Pair &pair : pairs_)
      result.push_back({pair[0], pair[1]});
    return result;
  }

  bool ContainsPair(const tirx::Var &first, const tirx::Var &second) const {
    for (const Pair &pair : pairs_)
      if ((pair[0].same_as(first) && pair[1].same_as(second)) ||
          (pair[1].same_as(first) && pair[0].same_as(second)))
        return true;
    return false;
  }

  // Operand credits may replace only the exact selected scope ordering.
  // Preserve all pairs, duplicates and order, including accumulator pairs.
  bool MatchesConflicts(const ConflictHintList &root,
                        const ConflictHintList &local) const {
    return present_ && ffi::StructuralEqual()(conflicts_, root) &&
           ffi::StructuralEqual()(conflicts_, local);
  }

  // Consume one attempt's completed alias analysis. Do not infer symmetry,
  // transitive reuse, or safety from the selection itself. Returned indices
  // preserve the first-failure diagnostic and must not outlive that attempt.
  std::vector<size_t> UnprovenPairs(const BufferAliasMap &aliases) const {
    std::vector<size_t> result;
    for (size_t i = 0; i < pairs_.size(); ++i) {
      const Pair &pair = pairs_[i];
      if (!HasBufferAlias(aliases, pair[0], pair[1]) ||
          !HasBufferAlias(aliases, pair[1], pair[0]))
        result.push_back(i);
    }
    return result;
  }

private:
  bool present_{false};
  std::vector<Pair> pairs_;
  ConflictHintList conflicts_;
};

} // namespace tvm::tl::ascend
