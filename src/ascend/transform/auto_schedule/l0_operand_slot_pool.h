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

#include <numeric>

#include "./l0ab_phase_reuse.h"
#include "./multi_buffer.h"
#include "./optimization_diagnostic.h"

namespace tvm::tl::ascend {

// A credit protects bytes, not a phase's completion. Both logical layouts and
// prepared clocks remain intact. The late allocator must still co-locate the
// selected pair; the certificate below authorizes only that pair's memory.
struct L0OperandSlotMember {
  Buffer buffer;
  TaskNode *writer{nullptr};
  TaskNode *reader{nullptr};
  ControlNode *generation{nullptr};
  ControlNode *epoch{nullptr};
  IRStructure *phase{nullptr};
  IRStructure *phase_end{nullptr};
  const MultiBufferInfo *versions{nullptr};
  PrimExpr slot;
  PrimExpr read_slot;
  int fixed_planes{1};
  PrimExpr last_reader{Bool(true)};
  int credit_span{1};
};

struct L0OperandSlotPool {
  L0OperandSlotMember first, second;
  int slots{0};
  int64_t slot_bytes{0};
};

namespace l0_operand_slot_detail {
using namespace l0ab_phase_reuse_detail;

using resource_epoch_detail::Touches;

inline bool PureTypedArguments(const Call &call, const Buffer &buffer) {
  for (const PrimExpr &argument : call->args) {
    const auto *node = argument.as<CallNode>();
    if (!node || !node->op.same_as(region())) {
      if (Touches(argument, buffer->data))
        return false;
      continue;
    }
    BufferRegion bounds = NormalizeToBufferRegion(GetRef<Call>(node));
    if (bounds->buffer->data.same_as(buffer->data) &&
        !bounds->buffer.same_as(buffer))
      return false;
    for (const Range &range : bounds->region)
      if (Touches(range->min, buffer->data) ||
          Touches(range->extent, buffer->data))
        return false;
  }
  for (const auto &[_, annotation] : call->annotations) {
    auto value = annotation.as<PrimExpr>();
    if (!value || Touches(*value, buffer->data))
      return false;
  }
  return true;
}

inline bool IsFullEndpoint(TaskNode *task, const Buffer &buffer, bool write,
                            bool fixed_planes = false) {
  auto call = Operation(task);
  if (!call || task->HasGuards() || !PureTypedArguments(*call, buffer) ||
      task->GetCoreMask() != kCoreCube)
    return false;
  BufferRegion operand;
  if (!write && (*call)->op.same_as(Gemm::Get()) &&
      task->UsesOnlyPipe(ResourcePipe::kCube)) {
    Gemm gemm((*call)->args, (*call)->annotations);
    if (!IsL0ABuffer(gemm->a_) || !IsL0BBuffer(gemm->b_) ||
        !IsL0CBuffer(gemm->c_) || gemm->mbar_.defined() || gemm->wgWait_ ||
        gemm->isWgmma_ || gemm->isTcgen05_)
      return false;
    bool a = gemm->a_.same_as(buffer), b = gemm->b_.same_as(buffer);
    if (a == b)
      return false;
    operand = a ? gemm->aRegion_ : gemm->bRegion_;
  } else if (write && task->UsesOnlyPipe(ResourcePipe::kMTE1)) {
    if ((*call)->op.same_as(AscendCubeLoad3D::Get())) {
      AscendCubeLoad3D load((*call)->args, (*call)->annotations);
      if (!IsL1Buffer(load->src->buffer))
        return false;
      operand = load->dst;
    } else if (IsAscendCopyCall(call->get())) {
      AscendCopy load((*call)->args, (*call)->annotations);
      if (!IsL1Buffer(load->src))
        return false;
      operand = BufferRegion(load->dst, load->dst_range);
    } else {
      return false;
    }
  } else {
    return false;
  }
  return operand->buffer.same_as(buffer) &&
         (fixed_planes ? FullMatrixPlane(task, buffer, operand->region)
                       : FullAllocation(buffer, operand->region));
}

inline bool ClosedCounter(const std::vector<std::shared_ptr<IRStructure>> &root,
                          const ResourceEpochAnalysis &resource_epochs,
                          const MultiBufferInfo &info,
                          const L0OperandSlotMember &member) {
  if (info.owners.size() != 1 || member.writer->GetStage() != 0 ||
      member.reader->GetStage() != 0)
    return false;
  if (!resource_epochs.HasExclusiveCounter(info))
    return false;
  int init = 0, commit = 0;
  bool valid = true;
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (node->GuardsTouchStorage(info.counter->data))
      valid = false;
    if (node->IsControl()) {
      auto *control = static_cast<ControlNode *>(node);
      if (Touches(control->control->min, info.counter->data) ||
          Touches(control->control->extent, info.counter->data) ||
          (control->control->step.has_value() &&
           Touches(control->control->step.value(), info.counter->data)))
        valid = false;
      for (const auto &child : control->children)
        visit(child.get());
      return;
    }
    auto *task = static_cast<TaskNode *>(node);
    if (!Touches(task->stmt, info.counter->data))
      return;
    Stmt body = task->stmt;
    while (const auto *attribute = body.as<AttrStmtNode>())
      body = attribute->body;
    const auto *store = body.as<BufferStoreNode>();
    if (!store || !store->buffer.same_as(info.counter) || task->HasGuards() ||
        task->GetStage() != 0 || store->indices.size() != 1 ||
        !is_zero(store->indices[0])) {
      valid = false;
      return;
    }
    if (!task->GetParent() && is_zero(store->value) &&
        task->GetIndex() <
            member.generation->GetOutermostControl()->GetIndex()) {
      ++init;
      return;
    }
    PrimExpr next = BufferLoad(info.counter, {Integer(0)}) + 1;
    if (task->GetParent() == member.generation &&
        task->GetIndex() > member.reader->GetIndex() &&
        ffi::StructuralEqual()(store->value, next)) {
      ++commit;
      return;
    }
    valid = false;
  };
  for (const auto &node : root)
    visit(node.get());
  return valid && init == 1 && commit == 1;
}

inline bool SelectedPhaseConflicts(const ScheduledTIR &scheduled,
                                   const SelectedL0PhaseReuse &selected,
                                   const ControlNode *epoch) {
  auto local = epoch->control->annotations.Get("conflict_hint");
  if (!local)
    return false;
  // Only the exact early choice's scope ordering may be replaced by credits.
  // Do not interpret an arbitrary user conflict as operand-byte ownership.
  return selected.MatchesConflicts(scheduled.metadata.root_conflict_hints,
                                   local.value().cast<ConflictHintList>());
}

// A fixed leading axis is a collection of resident logical planes, not an
// automatic ring. One complete writer loop initializes every plane before a
// disjoint finite reader nest; the inner reader loop visits every plane once
// per repetition. Its own plane returns a whole-MAD credit only on the final
// repetition. Both endpoint indices remain in their original lexical clocks.
inline bool FixedPlaneMember(L0OperandSlotMember *member,
                              const ScheduledTIR &scheduled,
                              const MultiBufferPlan &plan,
                              const SelectedL0PhaseReuse &selected,
                              const ResourceEpochAnalysis &resource_epochs,
                              ResourceEpochDecision *epoch_decision) {
  const Buffer &buffer = member->buffer;
  auto count = Constant(buffer->shape[0]);
  if (buffer->shape.size() != 3 || !count || *count < 2 ||
      plan.Find(buffer->data))
    return false;
  auto operand = [&](TaskNode *task, bool write) -> BufferRegion {
    Call call = *Operation(task);
    if (write && call->op.same_as(AscendCubeLoad3D::Get()))
      return AscendCubeLoad3D(call->args, call->annotations)->dst;
    if (write) {
      AscendCopy copy(call->args, call->annotations);
      return BufferRegion(copy->dst, copy->dst_range);
    }
    Gemm gemm(call->args, call->annotations);
    return gemm->a_.same_as(buffer) ? gemm->aRegion_ : gemm->bRegion_;
  };
  ControlNode *writer = ParentControl(member->writer);
  ControlNode *reader = ParentControl(member->reader);
  auto complete_plane_loop = [&](ControlNode *loop, TaskNode *task,
                                  const BufferRegion &region) {
    return loop && PositiveStaticLoop(loop) && !loop->HasGuards() &&
           task->GetStage() == 0 &&
           Constant(loop->control->extent) == count &&
           region->region[0]->min.same_as(loop->control->loop_var);
  };
  BufferRegion written = operand(member->writer, true);
  BufferRegion read = operand(member->reader, false);
  if (!complete_plane_loop(writer, member->writer, written) ||
      !complete_plane_loop(reader, member->reader, read))
    return false;
  IRStructure *write_phase = nullptr, *read_phase = nullptr;
  if (!LocateClosedPhase(member->writer, &member->epoch, &write_phase, true,
                         &resource_epochs, {buffer->data}, epoch_decision) ||
      !LocateClosedPhase(member->reader, &member->epoch, &read_phase, true,
                         &resource_epochs, {buffer->data}, epoch_decision) ||
      write_phase != writer ||
      write_phase->GetIndex() >= read_phase->GetIndex() ||
      write_phase->GetStage() < 0 ||
      write_phase->GetStage() != read_phase->GetStage() ||
      !SelectedPhaseConflicts(scheduled, selected, member->epoch))
    return false;
  for (IRStructure *node : member->reader->PathFrom(member->epoch)) {
    if (node == member->epoch || node == member->reader)
      continue;
    if (!node->IsControl())
      return false;
    auto *loop = static_cast<ControlNode *>(node);
    if (!PositiveStaticLoop(loop) || loop->HasGuards() ||
        (loop != read_phase && !StageZero(loop)))
      return false;
    if (loop != reader)
      member->last_reader = member->last_reader &&
                            (loop->control->loop_var ==
                             loop->control->extent - 1);
  }
  member->generation = member->epoch;
  member->phase = write_phase;
  member->phase_end = read_phase;
  member->slot = written->region[0]->min;
  member->read_slot = read->region[0]->min;
  member->fixed_planes = static_cast<int>(*count);
  return true;
}

inline std::optional<L0OperandSlotMember> RejectMember(const Buffer &buffer,
                                                       const char *reason) {
  if (runtime::detail::VerboseLoggingEnabled(__FILE__, 1))
    LOG(INFO) << "L0 operand slot member " << buffer->data
              << " rejected: " << reason;
  return std::nullopt;
}

inline std::optional<L0OperandSlotMember>
Member(const ScheduledTIR &scheduled, const MultiBufferPlan &plan,
       const SelectedL0PhaseReuse &selected, const Buffer &buffer,
       const LayoutMap &layouts, int64_t capacity,
       const ResourceEpochAnalysis &resource_epochs,
       ResourceEpochDecision *epoch_decision, const char **rejection_reason) {
  auto reject = [&](const char *reason) {
    *rejection_reason = reason;
    return RejectMember(buffer, reason);
  };
  bool fixed_planes = buffer->shape.size() == 3;
  if (!(fixed_planes ? CanonicalPlaneAllocation(buffer, layouts, capacity)
                     : CanonicalAllocation(buffer, layouts, capacity)))
    return reject("canonical-allocation");
  if (auto physical =
          scheduled.metadata.manual_buffer_versions.Get(buffer->data))
    if (physical.value() != 1)
      return reject("manual-physical-versions");
  L0OperandSlotMember result;
  result.buffer = buffer;
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  for (TaskNode *task : tasks) {
    bool reads = false, writes = false;
    for (const auto &access : task->GetReadRegions())
      reads |= access->buffer->data.same_as(buffer->data);
    for (const auto &access : task->GetWriteRegions())
      writes |= access->buffer->data.same_as(buffer->data);
    if (!reads && !writes) {
      if (Touches(task->stmt, buffer->data))
        return reject("unmodeled-operand-access");
      continue;
    }
    if (reads == writes || !IsFullEndpoint(task, buffer, writes, fixed_planes))
      return reject("typed-full-endpoint");
    TaskNode *&endpoint = writes ? result.writer : result.reader;
    if (endpoint)
      return reject("nonunique-endpoint");
    endpoint = task;
  }
  if (!result.writer || !result.reader)
    return reject("missing-endpoint");
  if (fixed_planes) {
    if (!FixedPlaneMember(&result, scheduled, plan, selected, resource_epochs,
                          epoch_decision))
      return reject("fixed-plane-generation");
  } else {
    result.generation = ParentControl(result.writer);
    if (!result.generation || !PositiveStaticLoop(result.generation) ||
        !result.reader->IsWithin(result.generation))
      return reject("finite-generation");
    IRStructure *read_child =
        result.reader->GetChildOnPathFrom(result.generation);
    if (result.writer->GetStage() > read_child->GetStage() ||
        (result.writer->GetStage() == read_child->GetStage() &&
         result.writer->GetIndex() >= read_child->GetIndex()))
      return reject("logical-endpoint-order");
    // All intervening controls are finite. An activation may skip an entire
    // generation, never its writer or one of its readers independently.
    for (IRStructure *node : result.reader->PathFrom(result.generation)) {
      if (node == result.generation || node == result.reader)
        continue;
      if (!node->IsControl())
        return reject("reader-path");
      auto *control = static_cast<ControlNode *>(node);
      if (!PositiveStaticLoop(control) || control->HasGuards() ||
          !StageZero(control))
        return reject("finite-reader-path");
      result.last_reader = result.last_reader && (control->control->loop_var ==
                                                  control->control->extent - 1);
    }
    result.versions = plan.Find(buffer->data);
    if (result.versions) {
      const auto &info = *result.versions;
      if (info.num_versions < 1 || info.owners.size() != 1 ||
          info.owners.front().loop != result.generation ||
          (info.UsesCounter() &&
           result.reader->GetParent() != result.generation))
        return reject("prepared-version-ownership");
      // Ring size is bounded by physical byte capacity below and the shared
      // event bank during flag allocation. A legal three-slot ring needs the
      // same per-slot whole-MAD credit as a two-slot ring.
      // An iteration-owned tile can serve the already-proven finite serial
      // reader nest. All readers use the generation's slot; last_reader returns
      // its whole-MAD credit only after the final occurrence. Counter commits
      // retain their direct writer/reader/advance contract.
      if (info.UsesCounter() &&
          !ClosedCounter(scheduled.tree, resource_epochs, info, result))
        return reject("closed-counter");
      result.slot = MultiBufferVersionIndex(info);
    } else {
      if (result.writer->GetStage() != 0 || result.reader->GetStage() != 0)
        return reject("fixed-endpoint-stage");
      result.slot = Integer(0);
    }
    if (!LocateClosedPhase(result.writer, &result.epoch, &result.phase, true,
                           &resource_epochs, {buffer->data}, epoch_decision) ||
        !LocateClosedPhase(result.reader, &result.epoch, &result.phase, true,
                           &resource_epochs, {buffer->data}, epoch_decision) ||
        !SelectedPhaseConflicts(scheduled, selected, result.epoch))
      return reject("closed-prepared-phase");
    result.read_slot = result.slot;
    result.phase_end = result.phase;
  }
  std::function<bool(IRStructure *)> controls = [&](IRStructure *node) {
    if (node->GuardsTouchStorage(buffer->data))
      return false;
    if (node->IsTask())
      return true;
    auto *control = static_cast<ControlNode *>(node);
    if (Touches(control->control->min, buffer->data) ||
        Touches(control->control->extent, buffer->data) ||
        (control->control->step.has_value() &&
         Touches(control->control->step.value(), buffer->data)))
      return false;
    for (const auto &child : control->children)
      if (!controls(child.get()))
        return false;
    return true;
  };
  for (const auto &node : scheduled.tree)
    if (!controls(node.get()))
      return reject("storage-dependent-control");
  return result;
}
} // namespace l0_operand_slot_detail

inline std::vector<L0OperandSlotPool> AnalyzeL0OperandSlotPools(
    const ScheduledTIR &scheduled, const MultiBufferPlan &versions,
    const SelectedL0PhaseReuse &selected, const AscendTargetInfo &target,
    OptimizationDiagnostics *diagnostics = nullptr) {
  using namespace l0_operand_slot_detail;
  std::vector<L0OperandSlotPool> pools;
  auto raw_layouts =
      scheduled.metadata.kernel_root->annotations.Get(attr::kLayoutMap);
  if (target.architecture != AscendArchitecture::kDav2201 ||
      !selected.Present() || !raw_layouts ||
      scheduled.metadata.has_buffer_aliases)
    return pools;
  LayoutMap layouts = raw_layouts.value().cast<LayoutMap>();
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(scheduled.tree, tasks);
  ResourceEpochAnalysis resource_epochs(scheduled, &versions);
  for (const auto &pair : selected.Pairs()) {
    auto record = [&](OptimizationStatus status, const char *reason,
                      const ControlNode *epoch = nullptr,
                      const Var &cause = Var()) {
      if (!diagnostics)
        return;
      std::vector<Var> resources(pair.begin(), pair.end());
      if (cause.defined() && std::none_of(resources.begin(), resources.end(),
                                          [&](const Var &storage) {
                                            return storage.same_as(cause);
                                          }))
        resources.push_back(cause);
      diagnostics->Record("l0-operand-slot-pool", status, reason,
                          std::move(resources), epoch);
    };
    if (Scope(pair[0]).empty() || Scope(pair[0]) != Scope(pair[1])) {
      record(OptimizationStatus::kRejected, "incompatible-storage-pair");
      continue;
    }
    Buffer buffers[2];
    int allocations = 0;
    for (const Buffer &buffer : scheduled.metadata.kernel_root->alloc_buffers) {
      if (GetPtrStorageScope(buffer->data) == GetPtrStorageScope(pair[0]))
        ++allocations;
      for (size_t i = 0; i < 2; ++i)
        if (buffer->data.same_as(pair[i]))
          buffers[i] = buffer;
    }
    if (allocations != 2 || !buffers[0].defined() || !buffers[1].defined()) {
      record(OptimizationStatus::kRejected, "noncanonical-root-allocations");
      continue;
    }
    bool nested_allocation = false;
    PostOrderVisit(scheduled.metadata.kernel_root, [&](const ObjectRef &node) {
      if (const auto *allocation = node.as<AllocBufferNode>())
        nested_allocation |= Scope(allocation->buffer->data) == Scope(pair[0]);
      if (const auto *block = node.as<SBlockNode>())
        if (block != scheduled.metadata.kernel_root.get())
          for (const Buffer &allocation : block->alloc_buffers)
            nested_allocation |= Scope(allocation->data) == Scope(pair[0]);
    });
    if (nested_allocation) {
      record(OptimizationStatus::kRejected, "nested-storage-allocation");
      continue;
    }
    KernelAudit audit(
        {{buffers[0]->data, buffers[0]}, {buffers[1]->data, buffers[1]}});
    for (TaskNode *task : tasks) {
      auto operation = Operation(task);
      // Unit-stream selection inserted a real whole-M entry barrier before
      // this late audit. A pipe barrier has no operand-address semantics;
      // raw MAD/load calls and all unknown M/MTE1 effects remain rejected.
      if (operation && (*operation)->op.same_as(ascend_pipe_barrier()) &&
          (*operation)->args.size() == 1 &&
          (*operation)->args[0].as<StringImmNode>())
        continue;
      audit(task->stmt);
    }
    if (audit.unsafe_metadata || !audit.invalid.empty()) {
      record(OptimizationStatus::kRejected, "unmodeled-storage-effects");
      continue;
    }
    int64_t capacity =
        Scope(pair[0]) == "shared.l0a" ? target.l0a_bytes : target.l0b_bytes;
    ResourceEpochDecision epoch_decisions[2];
    const char *member_rejections[2] = {nullptr, nullptr};
    auto a =
        Member(scheduled, versions, selected, buffers[0], layouts, capacity,
               resource_epochs, &epoch_decisions[0], &member_rejections[0]);
    auto b =
        Member(scheduled, versions, selected, buffers[1], layouts, capacity,
               resource_epochs, &epoch_decisions[1], &member_rejections[1]);
    if (!a || !b) {
      size_t rejected = a ? 1 : 0;
      const auto &decision = epoch_decisions[rejected];
      const char *reason = decision.Accepted()
                               ? member_rejections[rejected]
                               : ResourceEpochReason(decision.reason);
      record(OptimizationStatus::kRejected, reason, nullptr, decision.storage);
      continue;
    }
    if (a->epoch != b->epoch || a->phase == b->phase) {
      record(OptimizationStatus::kRejected, "different-or-shared-phase");
      continue;
    }
    if (!(a->phase_end->GetIndex() < b->phase->GetIndex() ||
          b->phase_end->GetIndex() < a->phase->GetIndex())) {
      record(OptimizationStatus::kRejected, "interleaved-physical-phases",
             a->epoch);
      continue;
    }
    auto bytes = [](const Buffer &buffer) {
      size_t rank = buffer->shape.size();
      return *Constant(buffer->shape[rank - 2]) *
             *Constant(buffer->shape[rank - 1]) * 2;
    };
    int av = a->versions ? a->versions->num_versions : a->fixed_planes;
    int bv = b->versions ? b->versions->num_versions : b->fixed_planes;
    int64_t ab = bytes(a->buffer), bb = bytes(b->buffer);
    int64_t pitch = std::max(ab, bb);
    bool same_intervals = !((av > 1 && bv > 1 && ab != bb) ||
                            (av > 1 && bb > ab) || (bv > 1 && ab > bb));
    // Keep the existing single-credit interval for equal slots and a fixed
    // smaller prefix. Otherwise partition both physical rings at their common
    // byte boundary. A larger load acquires every intersecting interval and
    // its last whole MAD releases each one. Logical version pitches never
    // change, including when one slot overlaps two slots of the other ring.
    if (!same_intervals) {
      pitch = std::gcd(ab, bb);
      a->credit_span = ab / pitch;
      b->credit_span = bb / pitch;
    }
    int64_t physical_bytes = std::max(ab * av, bb * bv);
    if (physical_bytes > capacity) {
      record(OptimizationStatus::kRejected, "storage-capacity", a->epoch);
      continue;
    }
    int slots = (physical_bytes + pitch - 1) / pitch;
    pools.push_back({*a, *b, slots, pitch});
    record(OptimizationStatus::kSelected, "physical-slot-candidate", a->epoch);
  }
  return pools;
}

} // namespace tvm::tl::ascend
