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

// A placement certificate for the existing paired GM-input WAR protocol.
// The active notification follows the last typed GM->L1 copy, across *both*
// owners and every descendant slice. The inactive notification stays at the
// original wide boundary. They partition the original lexical epoch sequence;
// its flag clock, distance, seeds, drains and two AIV recipients do not change.
//
// This is deliberately NOT a completion edge for reader_boundary END. In
// particular, subsequent L1/L0 reads, MADs and FIXes need not have completed.
// A selected point must stay out of generic closure, projection, alias proofs
// and flag sharing. Only this storage's exact WAR / owner-input release may
// consume the certificate.
struct GmReadReleasePlan {
  OwnerResultPlan input;
  IRStructure *reader_boundary;
  PrimExpr active;
  PrimExpr inactive;

  bool MatchesInput(const OwnerResultPlan &other) const {
    return input.storage.same_as(other.storage) &&
           input.producer == other.producer &&
           input.consumer == other.consumer && input.epoch == other.epoch &&
           input.owner == other.owner && input.sid.same_as(other.sid) &&
           input.distance == other.distance &&
           input.num_slots == other.num_slots;
  }

  bool MatchesProtocol(ControlNode *flag_loop, int distance, int width,
                       size_t counter_group) const {
    return flag_loop == input.epoch && counter_group == 0 &&
           distance == input.num_slots - input.distance && distance > 0 &&
           width == distance;
  }
};

namespace gm_read_release_detail {

// The cut crosses only unshifted, positive immutable serial loops. Bounds
// are simplified without a loop-entered or phase-active constraint: empty,
// mutable, guarded-by-a-descendant and scheduled-offset cases keep the old
// collective protocol. OwnerResultPlan already proves exact region coverage,
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
// require path_end==input.consumer, then conjoin `active`. Keeping the owner
// loop in that traversal is essential: it emits once after owner 1, not once
// per owner. Copy the original END site and conjoin `inactive` for the empty
// epoch contribution. The destination BEGIN site remains untouched.
inline std::optional<GmReadReleasePlan>
AnalyzeGmReadRelease(const std::vector<OwnerResultPlan> &inputs,
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
      reader->GetCoreMask() != kCoreCube ||
      !reader->UsesOnlyPipe(ResourcePipe::kMTE2) ||
      writer->GetCoreMask() != kCoreVector ||
      !writer->UsesOnlyPipe(ResourcePipe::kMTE3) ||
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

  for (const OwnerResultPlan &input : inputs) {
    if (!input.storage.same_as(*dep.storage) || input.epoch != epoch ||
        input.consumer != reader || input.producer != writer ||
        input.num_slots != 2 || input.distance != 0 ||
        manual.Get(input.storage).value_or(0) != input.num_slots ||
        dep.distance != input.num_slots)
      continue;
    int64_t first, last;
    if (!owner_result_detail::ExactInterval(input.consumer_activation,
                                            loop->loop_var, *low, *low + *count,
                                            &first, &last) ||
        first >= last)
      continue;
    // Only the outer epoch variable escapes the guarded phase. Local Bind
    // names from a task's activation must not be referenced at the wide END.
    PrimExpr e = loop->loop_var;
    PrimExpr active =
        e >= make_const(e.dtype(), first) && e < make_const(e.dtype(), last);
    return GmReadReleasePlan{input, dep.prod_node, active, !active};
  }
  return std::nullopt;
}

} // namespace tvm::tl::ascend
