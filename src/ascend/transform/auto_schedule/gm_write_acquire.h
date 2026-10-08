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

#include "./gm_read_release.h"

namespace tvm::tl::ascend {

// A placement certificate for the existing paired GM-output WAR protocol.
// The active wait precedes the first typed L0C->GM copy, across *both* owners
// and every descendant slice. The inactive wait stays at the original wide
// boundary. Together they consume one aggregate permission per original
// lexical epoch; the clock, distance, seeds and drains remain unchanged. This
// certificate leaves source placement alone; an independent last-read release
// certificate may compose with it on the same original permission.
//
// This does NOT establish completion before writer_boundary BEGIN: L1/L0
// preparation and MADs may already have issued. The selected point must stay
// out of generic closure, projection, alias proofs and flag sharing. Only this
// storage's exact WAR / owner-result release may consume the certificate.
struct GmWriteAcquirePlan {
  ValidatedOwnerTransfer transfer;
  IRStructure *writer_boundary;
  PrimExpr active;
  PrimExpr inactive;

  bool MatchesResult(const ValidatedOwnerTransfer &other) const {
    return transfer.IsResult() && transfer.MatchesTransfer(other);
  }

  bool MatchesProtocol(ControlNode *flag_loop, int distance, int width,
                       size_t counter_group) const {
    return transfer.MatchesPairedWar(flag_loop, distance, width, counter_group);
  }
};

// `results` must be the current AnalyzeOwnerResults result. The established
// owner proof supplies canonical GM identity, exact owner/slice coverage,
// immutable activation, ring separation and exclusion of other GM accesses.
// This helper only selects a placement; it neither creates an owner plan nor
// changes IR, events or completion closure. Unsupported cases retain the
// ordinary protocol.
//
// Caller checks remain necessary: dav-2201 mode2/two AIVs, original endpoints
// in the same unconditional lexical domain with guard=true, and an ordinary
// V/MTE2 -> C/FIX point matching MatchesProtocol. Construct the active BEGIN
// site with active_tasks_only=true and participant_loop=nullptr; require
// path_end==transfer.producer, then conjoin `active`. Keeping every crossed
// loop in that traversal emits only at owner 0 / the first slice. Copy the
// original BEGIN site and conjoin `inactive` for an empty epoch. Source
// placement is independent, and all C/SPR/unit completion obligations remain
// separate.
inline std::optional<GmWriteAcquirePlan>
AnalyzeGmWriteAcquire(const std::vector<ValidatedOwnerTransfer> &results,
                      const DepInfo &dep, TaskNode *reader, TaskNode *writer,
                      ControlNode *epoch, const BufferVersionMap &manual,
                      const MultiBufferPlan &automatic) {
  if (!epoch || epoch->GetStage() != 0 || epoch->ContainsLoopBreak() ||
      dep.kind != DependencyKind::kData || dep.cross_storage_conflict ||
      !dep.storage || GetPtrStorageScope(*dep.storage) != "global" ||
      automatic.Find(*dep.storage) || dep.task_pairs.size() != 1 ||
      dep.task_pairs[0] != DependencyTaskPair{reader, writer} ||
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
        transfer.distance < 0 || transfer.distance >= transfer.num_slots ||
        manual.Get(transfer.storage).value_or(0) != transfer.num_slots ||
        dep.distance != transfer.num_slots - transfer.distance)
      continue;
    auto active =
        transfer.ActiveEpochs(ValidatedOwnerTransfer::Endpoint::kProducer);
    if (!active)
      continue;
    return GmWriteAcquirePlan{transfer, dep.cons_node, *active, !*active};
  }
  return std::nullopt;
}

} // namespace tvm::tl::ascend
