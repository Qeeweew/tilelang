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

#include <limits>
#include <optional>
#include <vector>

#include "./dependency_analysis.h"
#include "./multi_buffer.h"
#include "./owner_result.h"

namespace tvm::tl::ascend {

// A placement certificate for an existing paired GM WAR protocol. An input
// release follows the last typed GM->L1 copy across both Cube owners; a result
// release follows the last typed GM->UB copy on each AIV. The inactive
// notification stays at the original wide boundary. They partition the
// original lexical epoch sequence; its flag clock, distance, seeds, drains and
// two AIV participants do not change.
//
// This is deliberately NOT a completion edge for reader_boundary END. In
// particular, subsequent L1/L0 reads, MADs, FIXes or Vector work need not have
// completed.
// A selected point must stay out of generic closure, projection, alias proofs
// and flag sharing. Only this storage's exact WAR / corresponding owner
// transfer release may consume the certificate.
struct GmReadReleasePlan {
  ValidatedOwnerTransfer transfer;
  IRStructure *reader_boundary;
  PrimExpr active;
  PrimExpr inactive;
  bool MatchesInput(const ValidatedOwnerTransfer &other) const {
    return transfer.IsInput() && transfer.MatchesTransfer(other);
  }

  bool MatchesResult(const ValidatedOwnerTransfer &other) const {
    return transfer.IsResult() && transfer.MatchesTransfer(other);
  }

  bool MatchesProtocol(ControlNode *flag_loop, int distance, int width,
                       size_t counter_group) const {
    return transfer.MatchesPairedWar(flag_loop, distance, width, counter_group);
  }
};

namespace gm_read_release_detail {

// The cut crosses only unshifted, positive immutable serial loops. Bounds
// are simplified without a loop-entered or phase-active constraint: empty,
// mutable, guarded-by-a-descendant and scheduled-offset cases keep the old
// collective protocol. ValidatedOwnerTransfer proves exact region coverage,
// immutable activation, storage identity and exclusion of other GM accesses.
inline bool UnshiftedPath(TaskNode *task, ControlNode *epoch) {
  if (!task || !epoch || !task->IsWithin(epoch))
    return false;
  for (IRStructure *node : task->PathFrom(task->GetChildOnPathFrom(epoch))) {
    if (node->GetStage() != 0)
      return false;
    if (!node->IsControl())
      continue;
    auto *control = static_cast<ControlNode *>(node);
    const For &loop = control->control;
    auto low = owner_result_detail::Constant(loop->min);
    auto count = owner_result_detail::Constant(loop->extent);
    auto step = owner_result_detail::Constant(
        loop->step.value_or(make_const(loop->loop_var.dtype(), 1)));
    if (loop->kind != ForKind::kSerial || control->BodyHasLoopBreak() || !low ||
        !count || !step || *count <= 0 || *step != 1 ||
        *low > std::numeric_limits<int64_t>::max() - *count)
      return false;
  }
  return true;
}

} // namespace gm_read_release_detail

// `inputs` must be the current AnalyzeOwnerInputs result, never result plans
// or user annotations. This helper does not alter the IR, synthesize events or
// query dependency closure. Unsupported cases return no cut.
//
// Caller-side site checks remain necessary: dav-2201 mode2/two AIVs, original
// endpoints in the same unconditional lexical domain with guard=true, and an
// ordinary C/MTE2 -> V/MTE3 point matching MatchesProtocol. Construct the
// active END site with active_tasks_only=true and participant_loop=nullptr;
// require path_end==transfer.consumer, then conjoin `active`. Keeping the owner
// loop in that traversal is essential: it emits once after owner 1, not once
// per owner. Copy the original END site and conjoin `inactive` for the empty
// epoch contribution. The destination BEGIN site remains untouched.
inline std::optional<GmReadReleasePlan>
AnalyzeGmReadRelease(const std::vector<ValidatedOwnerTransfer> &inputs,
                     const DepInfo &dep, TaskNode *reader, TaskNode *writer,
                     ControlNode *epoch, const BufferVersionMap &manual,
                     const MultiBufferPlan &automatic) {
  if (!epoch || epoch->GetStage() != 0 || epoch->ContainsLoopBreak() ||
      dep.kind != DependencyKind::kData || dep.cross_storage_conflict ||
      !dep.storage || GetPtrStorageScope(*dep.storage) != "global" ||
      automatic.Find(*dep.storage) || dep.task_pairs.size() != 1 ||
      !dep.task_pairs[0].MatchesEndpoints(reader, writer) ||
      !dep.task_pairs[0].HasMemoryHazard(DependencyAccessMode::kRead,
                                         DependencyAccessMode::kWrite,
                                         *dep.storage) ||
      !gm_read_release_detail::UnshiftedPath(reader, epoch) ||
      !gm_read_release_detail::UnshiftedPath(writer, epoch) ||
      reader->GetCoreMask() != kCoreCube ||
      !reader->UsesOnlyPipe(ResourcePipe::kMTE2) ||
      writer->GetCoreMask() != kCoreVector ||
      !writer->UsesOnlyPipe(ResourcePipe::kMTE3) ||
      dep.prod_node != reader->GetChildOnPathFrom(epoch) ||
      dep.cons_node != writer->GetChildOnPathFrom(epoch))
    return std::nullopt;

  for (const ValidatedOwnerTransfer &transfer : inputs) {
    if (!transfer.IsInput() || !transfer.storage.same_as(*dep.storage) ||
        transfer.epoch != epoch || transfer.consumer != reader ||
        transfer.producer != writer || transfer.num_slots != 2 ||
        transfer.distance != 0 ||
        manual.Get(transfer.storage).value_or(0) != transfer.num_slots ||
        dep.distance != transfer.num_slots)
      continue;
    auto active =
        transfer.ActiveEpochs(ValidatedOwnerTransfer::Endpoint::kConsumer);
    if (!active)
      continue;
    return GmReadReleasePlan{transfer, dep.prod_node, *active, !*active};
  }
  return std::nullopt;
}

// `results` must be the current AnalyzeOwnerResults result. The existing owner
// proof supplies both AIVs' complete, SID-separated reads and the exact shifted
// finite activation. This placement only shortens the GM storage's read
// lifetime; it does not certify Vector phase END or any subsequent UB user.
//
// Caller checks are the same as above, except for the ordinary V/MTE2 -> C/FIX
// direction. Build the active reader END site with kCoreVector,
// active_tasks_only=true and participant_loop=nullptr; require
// path_end==transfer.consumer, then conjoin `active`. Retaining all slice loops
// makes each AIV notify only after its last GM read. The old wide END under
// `inactive` preserves one contribution from each AIV in every empty epoch.
// The destination can independently use a GmWriteAcquirePlan; neither cut may
// contribute its old wide endpoint to generic completion closure.
inline std::optional<GmReadReleasePlan> AnalyzeGmResultReadRelease(
    const std::vector<ValidatedOwnerTransfer> &results, const DepInfo &dep,
    TaskNode *reader, TaskNode *writer, ControlNode *epoch,
    const BufferVersionMap &manual, const MultiBufferPlan &automatic) {
  if (!epoch || epoch->GetStage() != 0 || epoch->ContainsLoopBreak() ||
      dep.kind != DependencyKind::kData || dep.cross_storage_conflict ||
      !dep.storage || GetPtrStorageScope(*dep.storage) != "global" ||
      automatic.Find(*dep.storage) || dep.task_pairs.size() != 1 ||
      !dep.task_pairs[0].MatchesEndpoints(reader, writer) ||
      !dep.task_pairs[0].HasMemoryHazard(DependencyAccessMode::kRead,
                                         DependencyAccessMode::kWrite,
                                         *dep.storage) ||
      !gm_read_release_detail::UnshiftedPath(reader, epoch) ||
      !gm_read_release_detail::UnshiftedPath(writer, epoch) ||
      reader->GetCoreMask() != kCoreVector ||
      !reader->UsesOnlyPipe(ResourcePipe::kMTE2) ||
      writer->GetCoreMask() != kCoreCube ||
      !writer->UsesOnlyPipe(ResourcePipe::kFixpipe) ||
      dep.prod_node != reader->GetChildOnPathFrom(epoch) ||
      dep.cons_node != writer->GetChildOnPathFrom(epoch))
    return std::nullopt;

  for (const ValidatedOwnerTransfer &transfer : results) {
    if (!transfer.IsResult() || !transfer.storage.same_as(*dep.storage) ||
        transfer.epoch != epoch || transfer.consumer != reader ||
        transfer.producer != writer || transfer.num_slots != 2 ||
        transfer.distance != 1 ||
        manual.Get(transfer.storage).value_or(0) != transfer.num_slots ||
        dep.distance != transfer.num_slots - transfer.distance)
      continue;
    auto active =
        transfer.ActiveEpochs(ValidatedOwnerTransfer::Endpoint::kConsumer);
    if (!active)
      continue;
    return GmReadReleasePlan{transfer, dep.prod_node, *active, !*active};
  }
  return std::nullopt;
}

} // namespace tvm::tl::ascend
