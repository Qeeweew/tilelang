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
  OwnerResultPlan result;
  IRStructure *writer_boundary;
  PrimExpr active;
  PrimExpr inactive;

  bool MatchesResult(const OwnerResultPlan &other) const {
    return result.storage.same_as(other.storage) &&
           result.producer == other.producer &&
           result.consumer == other.consumer && result.epoch == other.epoch &&
           result.owner == other.owner && result.sid.same_as(other.sid) &&
           result.distance == other.distance &&
           result.num_slots == other.num_slots;
  }

  bool MatchesProtocol(ControlNode *flag_loop, int distance, int width,
                       size_t counter_group) const {
    return flag_loop == result.epoch && counter_group == 0 &&
           distance == result.num_slots - result.distance && distance > 0 &&
           width == distance;
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
// path_end==result.producer, then conjoin `active`. Keeping every crossed loop
// in that traversal emits only at owner 0 / the first slice. Copy the original
// BEGIN site and conjoin `inactive` for an empty epoch. Source placement is
// independent, and all C/SPR/unit completion obligations remain separate.
inline std::optional<GmWriteAcquirePlan>
AnalyzeGmWriteAcquire(const std::vector<OwnerResultPlan> &results,
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

  const For &loop = epoch->control;
  auto low = owner_result_detail::Constant(loop->min);
  auto count = owner_result_detail::Constant(loop->extent);
  auto step = owner_result_detail::Constant(
      loop->step.value_or(make_const(loop->loop_var.dtype(), 1)));
  if (loop->kind != ForKind::kSerial || !low || !count || !step ||
      *count <= 0 || *step != 1 ||
      *low > std::numeric_limits<int64_t>::max() - *count)
    return std::nullopt;

  for (const OwnerResultPlan &result : results) {
    if (!result.storage.same_as(*dep.storage) || result.epoch != epoch ||
        result.consumer != reader || result.producer != writer ||
        result.num_slots != 2 || result.distance < 0 ||
        result.distance >= result.num_slots ||
        manual.Get(result.storage).value_or(0) != result.num_slots ||
        dep.distance != result.num_slots - result.distance)
      continue;
    int64_t first, last;
    if (!owner_result_detail::ExactInterval(result.producer_activation,
                                            loop->loop_var, *low, *low + *count,
                                            &first, &last) ||
        first >= last)
      continue;
    // Only the outer epoch variable escapes the guarded phase. Rebuild the
    // interval instead of leaking local Bind names to the inactive wide site.
    PrimExpr e = loop->loop_var;
    PrimExpr active =
        e >= make_const(e.dtype(), first) && e < make_const(e.dtype(), last);
    return GmWriteAcquirePlan{result, dep.cons_node, active, !active};
  }
  return std::nullopt;
}

} // namespace tvm::tl::ascend
