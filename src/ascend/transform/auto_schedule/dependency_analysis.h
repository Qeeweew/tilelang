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
 * \file dependency_analysis.h
 * \brief Shared schedule and synchronization dependency analysis.
 */

#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../buffer_version.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

enum class DependencyAccessMode { kRead, kWrite };

// All logical region accesses to the parent DepInfo's storage, including
// regions that did not form a conflict. These are snapshot-local access roles,
// not coverage or completion facts. Storage-free ordering has no roles.
struct DependencyStorageAccess {
  bool reads{false};
  bool writes{false};

  bool IsReadOnly() const { return reads && !writes; }
  bool IsWriteOnly() const { return writes && !reads; }
};

// The logical may-regions used to form one conflict in shared analysis.
// They may be conservative footprints or hulls. These are access provenance,
// not exact accesses, must-access/coverage, or completion facts.
struct DependencyMemoryAccess {
  BufferRegion producer_region;
  BufferRegion consumer_region;
  DependencyAccessMode producer_mode;
  DependencyAccessMode consumer_mode;
};

// One leaf endpoint pair, with every memory conflict that formed its grouped
// dependency. Keep the endpoints and their roles together: a read-modify-write
// pair can carry more than one hazard, without creating duplicate sync points.
// A root SPR ordering has no memory accesses and names its actual register;
// it must never masquerade as a versioned buffer hazard.
struct DependencyTaskPair {
  TaskNode *producer;
  TaskNode *consumer;
  std::vector<DependencyMemoryAccess> memory_accesses;
  uint16_t special_register_mask{0};
  // Complete leaf roles on DepInfo.storage, unlike memory_accesses which
  // contains only the conflicts that formed this dependency.
  DependencyStorageAccess producer_access;
  DependencyStorageAccess consumer_access;

  bool MatchesEndpoints(const TaskNode *source,
                        const TaskNode *destination) const {
    return producer == source && consumer == destination;
  }

  bool HasMemoryHazard(DependencyAccessMode source_mode,
                       DependencyAccessMode destination_mode,
                       const Var &storage) const {
    for (const auto &access : memory_accesses)
      if (access.producer_mode == source_mode &&
          access.consumer_mode == destination_mode &&
          access.producer_region->buffer->data.same_as(storage) &&
          access.consumer_region->buffer->data.same_as(storage))
        return true;
    return false;
  }

  DependencyTaskPair Reversed() const {
    DependencyTaskPair result = *this;
    std::swap(result.producer, result.consumer);
    std::swap(result.producer_access, result.consumer_access);
    for (auto &access : result.memory_accesses) {
      std::swap(access.producer_region, access.consumer_region);
      std::swap(access.producer_mode, access.consumer_mode);
    }
    return result;
  }
};

enum class DependencyKind {
  // A directed producer-to-consumer data dependency.
  kData,
  // Two disjoint owners of one multi-buffer-eligible storage, or selected
  // physical aliases in independently shifted phases.
  // `prod_node` and `cons_node` only establish a stable pair orientation;
  // `task_pairs` stores matching accesses on those two sides. Scheduling must
  // choose either direction and keep both storage lifetimes disjoint.
  kOwnerExclusion,
};

struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  // Storage identity shared by every Buffer alias participating in this
  // dependency. Undefined for storage-free ordering (explicit conflicts or
  // invariant SPR state); those edges do not participate in buffer versioning.
  ffi::Optional<Var> storage;
  // Conflicting leaf/access records. For kData these are directed
  // producer→consumer pairs; for kOwnerExclusion they identify the two
  // unordered owner sides. Reversing the owner order must reverse each entire
  // record, including regions and access modes, not just its leaf pointers.
  std::vector<DependencyTaskPair> task_pairs;
  // Cross-iteration distance of this dependency:
  //   0   : same-iteration dependency.
  //   -1  : unresolved cross-iteration dependency. A versioned storage may
  //         resolve it to its ring width; otherwise it means one iteration
  //         of the loop whose child list is being analyzed.
  //   >=1 : a proven forward distance (or conservative lower bound), from
  //         manual ring regions or persistent-state regions and guards.
  // kOwnerExclusion leaves this as zero during shared analysis. InsertSync
  // resolves both directed distances from the final stage and physical order.
  int distance;
  DependencyKind kind{DependencyKind::kData};
  // A positive conflict hint between distinct storage identities.
  // Synchronization can preserve its requested lexical scope instead of
  // inferring either buffer's guarded epoch domain. False for invariant SPR
  // state dependencies.
  bool cross_storage_conflict{false};
  // Complete node roles on storage, including control-header accesses and
  // nonconflicting leaf regions. Empty when storage is undefined.
  DependencyStorageAccess producer_access;
  DependencyStorageAccess consumer_access;
};

// Per-phase dependency-analysis cache. A ControlNode key represents the
// dependency result for its ordered child list at that phase.
using DependencyCache = std::map<ControlNode *, std::vector<DepInfo>>;

// Conflict-hint entries. Each entry is
// [operand_a, operand_b, IntImm(cross_code), Bool(is_conflict)]. Kernel-root
// entries apply only while analyzing the outermost sequence.
using ConflictHintList = ffi::Array<ffi::Any>;

// Analyze data dependencies among scheduled TaskNode/ControlNode nodes. A loop
// child list is copied and stable-sorted by stage before dependency directions
// are derived; the kernel root list keeps its original order. `loop` identifies
// the ControlNode whose ordered children are being analyzed; null means the
// kernel root list. Ordinary carried dependencies compare different iterations
// of this loop with ancestor coordinates fixed and descendant coordinates
// independent. Parent analysis establishes outer-boundary hazards.
// `manual_buffer_versions` maps user-declared manual multi-buffer data Vars to
// their version count; empty means no manual buffers.
// `multi_buffer_owners` maps each storage treated as automatic multi-buffered
// in the current phase to its owner loops. Dependencies between distinct
// owners are returned as kOwnerExclusion; physical ring size is handled by the
// owner-local protocol.
// `dependency_cache` is shared for one scheduling or synchronization phase and
// is keyed by `loop`. An `i == j` control-node self-dependency is retained even
// if the same leaf pair occurs in a descendant dependency: the descendant
// relation does not establish completion across this parent loop's boundary.
std::vector<DepInfo>
AnalyzeDependencies(std::vector<IRStructure *> nodes,
                    ControlNode *loop = nullptr,
                    const BufferVersionMap &manual_buffer_versions = {},
                    const MultiBufferOwnerMap &multi_buffer_owners = {},
                    DependencyCache *dependency_cache = nullptr,
                    const ConflictHintList &root_conflicts = {});

} // namespace ascend
} // namespace tl
} // namespace tvm
