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
 * \file materialize_schedule_units.cc
 * \brief Normalize task boundaries and materialize trivial schedule units.
 */

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ascend/op/builtin.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "ascend/transform/auto_schedule/task_annotations.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using namespace ascend;
using ffi::Array;
using ffi::GetRef;

namespace {

Stmt MakeSequence(const std::vector<Stmt> &statements) {
  if (statements.empty())
    return Evaluate(0);
  if (statements.size() == 1)
    return statements[0];
  Array<Stmt> sequence;
  sequence.reserve(statements.size());
  for (const Stmt &stmt : statements)
    sequence.push_back(stmt);
  return SeqStmt::Flatten(sequence);
}

bool IsTaskBoundary(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>())
    return loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled;
  return stmt.as<EvaluateNode>() || stmt.as<BufferStoreNode>() ||
         stmt.as<BindNode>() || stmt.as<WhileNode>() || stmt.as<SBlockNode>();
}

bool IsCoreRegion(const SBlockNode *block) {
  return block->name_hint == "CUBE" || block->name_hint == "VECTOR";
}

class TaskNormalizer : public StmtMutator {
public:
  static SBlock Rewrite(SBlock root) {
    TaskNormalizer normalizer(root);
    Stmt body = normalizer(root->body);
    auto *writer = root.CopyOnWrite();
    writer->body = std::move(body);
    writer->alloc_buffers = std::move(normalizer.allocations_);
    if (!normalizer.layouts_.empty())
      writer->annotations.Set(tl::attr::kLayoutMap, normalizer.layouts_);
    if (normalizer.vector_count_.has_value())
      writer->annotations.Set(kVectorCount, normalizer.vector_count_.value());
    return root;
  }

private:
  explicit TaskNormalizer(const SBlock &root)
      : allocations_(root->alloc_buffers) {
    for (const Buffer &buffer : allocations_)
      allocation_owners_.Set(buffer->data, buffer);
    if (auto layouts = root->annotations.Get(tl::attr::kLayoutMap))
      MergeLayouts(layouts.value());
    if (auto count = root->annotations.Get(kVectorCount))
      RecordVectorCount(count.value());
  }

  void MergeLayouts(const ffi::Any &value) {
    // Scheduling also accepts frontend IR before layout inference. Preserve
    // its Var-keyed map, or the inferred Buffer-keyed map, without inventing
    // a conversion between these two layout-inference stage contracts.
    for (const auto &[key, layout] :
         value.cast<ffi::Map<ObjectRef, Layout>>()) {
      bool buffer_key = key.as<BufferNode>() != nullptr;
      ICHECK(buffer_key || key.as<VarNode>())
          << "Ascend layout_map keys must be Buffers or Vars";
      ICHECK(!buffer_layout_keys_.has_value() ||
             buffer_layout_keys_.value() == buffer_key)
          << "Ascend ownership regions cannot mix frontend Var-keyed and "
             "inferred Buffer-keyed layout maps";
      buffer_layout_keys_ = buffer_key;
      if (auto previous = layouts_.Get(key))
        ICHECK(previous.value()->IsEqual(layout.get()))
            << "Conflicting layout on an Ascend core region storage " << key;
      layouts_.Set(key, layout);
    }
  }

  void RecordVectorCount(const ffi::Any &value) {
    auto count = value.try_cast<Integer>();
    ICHECK(!value.as<bool>().has_value() && count.has_value() &&
           !count.value()->dtype.is_bool() &&
           (count.value()->value == 1 || count.value()->value == 2))
        << "Ascend core region vector_count must be the integer 1 or 2";
    ICHECK(!vector_count_.has_value() ||
           vector_count_.value()->value == count.value()->value)
        << "Conflicting Ascend core region vector_count values";
    vector_count_ = count.value();
  }

  TaskMetadata OwnershipMetadata(const ffi::Any &existing) const {
    TaskMetadata metadata;
    if (core_mask_ == kCoreUnassigned)
      return metadata;
    TaskMetadata previous = ParseTaskMetadata(existing);
    CoreMask narrowed =
        previous.core_mask.value_or(kCoreBroadcast) & core_mask_;
    ICHECK_NE(narrowed, kCoreUnassigned)
        << "T.Task core_mask conflicts with its CUBE/VECTOR ownership region";
    metadata.core_mask = narrowed;
    return metadata;
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (!IsCoreRegion(op->block.get()))
      return StmtMutator::VisitStmt_(op);
    ICHECK(op->iter_values.empty())
        << "Ascend core ownership regions require an unbound block";
    Stmt body = VisitStmt(op->block);
    if (is_one(op->predicate))
      return body;
    return IfThenElse(op->predicate, std::move(body));
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsScheduledCthreadBinding(op)) {
      const Var &sid = Downcast<IterVar>(op->node)->var;
      ICHECK(sid.dtype().lanes() == 1 &&
             (sid.dtype().is_int() || sid.dtype().is_uint()))
          << "Ascend cthread binding requires a scalar integer variable";
      RecordVectorCount(op->value);
      if (auto previous = sid_extents_.Get(sid))
        ICHECK(analyzer_.CanProveEqual(previous.value(), op->value))
            << "Repeated Ascend cthread binding has inconsistent extents";
      sid_extents_.Set(sid, op->value);
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (!IsCoreRegion(op))
      return StmtMutator::VisitStmt_(op);
    ICHECK_EQ(task_marker_depth_, 0)
        << "CUBE/VECTOR ownership regions must enclose complete T.Task "
           "boundaries, rather than appear inside an atomic T.Task";
    ICHECK(op->iter_vars.empty() && op->match_buffers.empty() &&
           !op->init.defined() && op->reads.empty() && op->writes.empty())
        << "Ascend core ownership regions cannot carry bound iterators, "
           "init, match buffers, or explicit block access declarations";
    CoreMask requested = op->name_hint == "CUBE" ? kCoreCube : kCoreVector;
    ICHECK(core_mask_ == kCoreUnassigned || core_mask_ == requested)
        << "Nested CUBE/VECTOR regions have incompatible core ownership";

    for (const auto &[key, value] : op->annotations) {
      if (key == kVectorCount) {
        ICHECK_EQ(requested, kCoreVector)
            << "vector_count belongs to an AIV ownership region";
        RecordVectorCount(value);
      } else if (key == tl::attr::kLayoutMap) {
        // Move the existing layout fact to the root before removing the shell.
        MergeLayouts(value);
      } else {
        LOG(FATAL) << "Unsupported Ascend core ownership annotation: " << key;
      }
    }
    for (const Buffer &buffer : op->alloc_buffers) {
      // A root allocation outlives the ownership shell. Static capacity keeps
      // its size independent of control variables whose scopes are removed.
      for (const PrimExpr &extent : buffer->shape)
        ICHECK(as_const_int(extent) != nullptr)
            << "Ascend core region allocations require static capacities";
      if (auto previous = allocation_owners_.Get(buffer->data)) {
        ICHECK(previous.value().same_as(buffer))
            << "Conflicting allocation owners for Ascend core region buffer "
            << buffer->name;
      } else {
        allocations_.push_back(buffer);
        allocation_owners_.Set(buffer->data, buffer);
      }
    }
    CoreMask previous = core_mask_;
    core_mask_ = requested;
    Stmt body = VisitStmt(op->body);
    core_mask_ = previous;
    return body;
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    // Ownership shells must be opened before the ordinary atomic-block rule.
    if (const auto *block = stmt.as<SBlockNode>(); block && IsCoreRegion(block))
      return VisitStmt_(block);
    if (const auto *realize = stmt.as<SBlockRealizeNode>();
        realize && IsCoreRegion(realize->block.get()))
      return VisitStmt_(realize);
    if (const auto *attribute = stmt.as<AttrStmtNode>()) {
      if (attribute->attr_key == tl::attr::kAscendStage) {
        ICHECK_EQ(task_marker_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.Task";
        ICHECK_EQ(per_core_task_depth_, 0)
            << "T.Stage must wrap a complete scheduler task; it cannot appear "
               "inside T.PerCoreTask";
      }
      if (attribute->attr_key == tl::attr::kAscendTask) {
        ++task_marker_depth_;
        Stmt result = StmtMutator::VisitStmt(stmt);
        --task_marker_depth_;
        const auto *normalized = result.as<AttrStmtNode>();
        ICHECK(normalized && normalized->attr_key == tl::attr::kAscendTask);
        TaskMetadata metadata = OwnershipMetadata(normalized->node);
        return AttrStmt(MergeTaskMetadata(metadata, normalized->node),
                        normalized->attr_key, normalized->value,
                        normalized->body, normalized->span);
      }
      if (attribute->attr_key == tl::attr::kAscendPerCoreTask) {
        ICHECK_EQ(task_marker_depth_, 0)
            << "T.PerCoreTask cannot be nested inside T.Task. Place T.Task "
               "candidate regions inside T.PerCoreTask instead.";
        ++per_core_task_depth_;
        Stmt body = VisitStmt(attribute->body);
        --per_core_task_depth_;
        TaskMetadata metadata = OwnershipMetadata(attribute->node);
        return AttrStmt(MergeTaskMetadata(metadata, attribute->node),
                        attribute->attr_key, attribute->value, std::move(body),
                        attribute->span);
      }
    }

    if (task_marker_depth_ > 0)
      return StmtMutator::VisitStmt(stmt);

    if (!IsTaskBoundary(stmt))
      return StmtMutator::VisitStmt(stmt);

    if (per_core_task_depth_ > 0 && stmt.as<BindNode>()) {
      ICHECK(!HasTaskBufferAccess(stmt))
          << "An unmarked scalar Bind inside T.PerCoreTask may not access "
             "buffers: synchronization with statements outside the "
             "PerCoreTask is emitted only at T.Task candidate sites, so an "
             "external buffer dependency could not be synchronized at this "
             "Bind. Dependency-free scalar Bind control is allowed. Move the "
             "buffer access into a supported same-pipe T.Task candidate. Bind="
          << stmt;
    }

    TaskMetadata metadata = OwnershipMetadata(IntImm(DataType::Int(32), 0));
    return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                    tl::attr::kAscendTask, IntImm(DataType::Int(32), 1), stmt,
                    stmt->span);
  }

  int task_marker_depth_{0};
  int per_core_task_depth_{0};
  CoreMask core_mask_{kCoreUnassigned};
  Array<Buffer> allocations_;
  ffi::Map<Var, Buffer> allocation_owners_;
  ffi::Map<ObjectRef, Layout> layouts_;
  std::optional<bool> buffer_layout_keys_;
  std::optional<Integer> vector_count_;
  ffi::Map<Var, PrimExpr> sid_extents_;
  arith::Analyzer analyzer_;
};

ScheduleUnitMetadata MakeInitialScheduleMetadata() {
  ScheduleUnitMetadata metadata;
  metadata.Set(kScheduleUnitStage, Integer(kUnscheduledStage));
  return metadata;
}

Stmt MakeInitialScheduleUnit(Stmt body) {
  Span span = body->span;
  return AttrStmt(MakeInitialScheduleMetadata(), attr::kScheduleUnit,
                  Integer(1), std::move(body), span);
}

Stmt MakeNoOpTask() {
  TaskMetadata metadata;
  Stmt body = Evaluate(0);
  return AttrStmt(MergeTaskMetadata(metadata, IntImm(DataType::Int(32), 0)),
                  tl::attr::kAscendTask, IntImm(DataType::Int(32), 1),
                  std::move(body));
}

AttrStmt RequireScheduleUnit(const Stmt &stmt) {
  const auto *unit = stmt.as<AttrStmtNode>();
  ICHECK(unit != nullptr && IsScheduleUnit(unit))
      << "MaterializeScheduleUnits expected a schedule unit, got " << stmt;
  return GetRef<AttrStmt>(unit);
}

Stmt RebuildWithPrependedGuard(const AttrStmt &unit, Stmt guarded_body) {
  return AttrStmt(unit->node, unit->attr_key, unit->value,
                  std::move(guarded_body), unit->span);
}

Stmt AssignStage(const Stmt &stmt, int stage) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  ICHECK_EQ(GetScheduleUnitStage(unit.get()), kUnscheduledStage)
      << "Nested T.Stage scopes are not supported";
  ScheduleUnitMetadata metadata = unit->node.cast<ScheduleUnitMetadata>();
  metadata.Set(kScheduleUnitStage, Integer(stage));
  return AttrStmt(metadata, unit->attr_key, unit->value, unit->body,
                  unit->span);
}

Stmt PrependConditionGuard(const Stmt &stmt, const PrimExpr &condition) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  return RebuildWithPrependedGuard(unit, IfThenElse(condition, unit->body));
}

Stmt PrependAttributeGuard(const Stmt &stmt, const AttrStmtNode *attribute) {
  AttrStmt unit = RequireScheduleUnit(stmt);
  return RebuildWithPrependedGuard(
      unit, AttrStmt(attribute->node, attribute->attr_key, attribute->value,
                     unit->body, attribute->span));
}

class PerCoreTaskControlValidator : public StmtExprVisitor {
public:
  explicit PerCoreTaskControlValidator(uint16_t pipe_mask)
      : expected_pipe_("PIPE_" + GetResourcePipeName(pipe_mask)) {}

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
        op->op.same_as(tl::ascend_cross_core_wait_flag())) {
      ICHECK_EQ(op->args.size(), 3);
      const auto *mode = op->args[0].as<IntImmNode>();
      ICHECK(mode) << "Cross-core synchronization mode must be a constant "
                      "integer, got "
                   << op->args[0];
      const auto *pipe = op->args[1].as<StringImmNode>();
      ICHECK(pipe) << "Inter-core synchronization pipe must be a constant "
                      "string, got "
                   << op->args[1];
      ICHECK_EQ(pipe->value, expected_pipe_)
          << "Inter-core synchronization inside T.PerCoreTask must use its "
             "inferred pipe "
          << expected_pipe_ << ", got " << pipe->value;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::string expected_pipe_;
};

class TaskMarkerControlValidator : public StmtExprVisitor {
public:
  TaskMarkerControlValidator(bool reject_candidate_selection,
                             bool allow_cross_core_sync)
      : reject_candidate_selection_(reject_candidate_selection),
        allow_cross_core_sync_(allow_cross_core_sync) {}

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    ICHECK_NE(op->attr_key, tl::attr::kAscendTask)
        << "Nested T.Task regions are not supported";
    ICHECK_NE(op->attr_key, tl::attr::kAscendStage)
        << "T.Stage must wrap a complete scheduler task. Place T.Stage outside "
           "T.Task, T.SimtVF, T.SimdVF, SBlock, and parallel-loop task "
           "boundaries.";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    ICHECK(!reject_candidate_selection_)
        << "Control flow selecting a T.Task candidate must be outside the "
           "T.Task scope";
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    ICHECK(allow_cross_core_sync_ ||
           (!op->op.same_as(tl::ascend_cross_core_set_flag()) &&
            !op->op.same_as(tl::ascend_cross_core_wait_flag())))
        << "Inter-core synchronization inside T.Task must be the task's only "
           "statement";
    StmtExprVisitor::VisitExpr_(op);
  }

  bool reject_candidate_selection_;
  bool allow_cross_core_sync_;
};

bool IsStandaloneCrossCoreSync(const Stmt &stmt) {
  const auto *evaluate = stmt.as<EvaluateNode>();
  if (!evaluate)
    return false;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

struct ValidatedTask {
  uint16_t pipe_mask{0};
  bool has_buffer_access{false};
};

// A user Task is atomic to scheduling/synchronization. Compiler lowering may
// expand one expression into several native MemBase instructions; retaining
// the atomic wrapper would hide their completion edges. A single select_bits
// helper remains one primitive with its internal CMPMASK setup/consume.
class AtomicNativeTaskValidator : public StmtExprVisitor {
public:
  static void Validate(const Stmt &body) {
    AtomicNativeTaskValidator validator;
    validator(body);
    ICHECK(validator.native_count_ <= 1 && !validator.repeated_native_)
        << "T.Task contains multiple or repeated native MemBase operations; "
           "leave ordinary Parallel/vector expressions outside T.Task or "
           "split them into separate primitive T.Task regions so their "
           "completion dependencies can be scheduled";
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (const auto *callee = op->op.as<OpNode>();
        callee && std::string(callee->name).rfind("tl.ascend_membase.", 0) == 0) {
      ++native_count_;
      repeated_native_ |= repeat_depth_ > 0;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    bool repeated = !is_one(op->extent);
    repeat_depth_ += repeated;
    StmtExprVisitor::VisitStmt_(op);
    repeat_depth_ -= repeated;
  }

  void VisitStmt_(const WhileNode *op) final {
    ++repeat_depth_;
    StmtExprVisitor::VisitStmt_(op);
    --repeat_depth_;
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "SIMT_VF" && op->name_hint != "SIMD_VF")
      StmtExprVisitor::VisitStmt_(op);
  }

  int native_count_{0};
  int repeat_depth_{0};
  bool repeated_native_{false};
};

ValidatedTask ValidateTask(const AttrStmtNode *op, bool inside_per_core_task) {
  ICHECK_EQ(op->attr_key, tl::attr::kAscendTask);

  Stmt task_body = op->body;
  TaskMarkerControlValidator control_validator(
      inside_per_core_task, IsStandaloneCrossCoreSync(task_body));
  control_validator(task_body);
  AtomicNativeTaskValidator::Validate(task_body);

  Stmt task = GetRef<Stmt>(op);
  TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(task);
  uint16_t issue_pipe_mask = resource_usage.pipe_mask;
  // FIX copies and native predicate selection configure/consume their SPR
  // inside one atomic data operation. Independent SPR writers still add S.
  if ((AnalyzeSpecialRegisterWrites(task) &
       ~(static_cast<uint16_t>(SpecialRegister::kFixpipeNz2Nd) |
         static_cast<uint16_t>(SpecialRegister::kCmpMask))) != 0)
    issue_pipe_mask |= static_cast<uint16_t>(ResourcePipe::kScalar);
  ICHECK((issue_pipe_mask & (issue_pipe_mask - 1)) == 0)
      << "T.Task supports exactly one Ascend hardware pipe, but its body uses "
         "multiple pipes (mask="
      << issue_pipe_mask
      << "). Split the statements into separate T.Task regions. Body=" << task;
  uint16_t hbm_core_affinity = resource_usage.hbm_mask & kHbmCoreAffinityMask;
  ICHECK((hbm_core_affinity & (hbm_core_affinity - 1)) == 0)
      << "T.Task requires one Ascend core affinity, but its body uses both AIC "
         "and AIV HBM paths. Split GM-to-L1 and GM-to-UB operations into "
         "separate T.Task regions. Body="
      << task;
  return {resource_usage.pipe_mask, HasTaskBufferAccess(task)};
}

bool IsPerCoreTaskScalarControl(const AttrStmtNode *task) {
  ICHECK_EQ(task->attr_key, tl::attr::kAscendTask);
  Stmt body = task->body;
  if (body.as<BindNode>())
    return true;
  const auto *evaluate = body.as<EvaluateNode>();
  if (!evaluate)
    return false;
  if (is_zero(evaluate->value))
    return true;
  const auto *call = evaluate->value.as<CallNode>();
  return call && (call->op.same_as(tl::ascend_cross_core_set_flag()) ||
                  call->op.same_as(tl::ascend_cross_core_wait_flag()));
}

class PerCoreTaskBodyValidator : public StmtVisitor {
public:
  static void Validate(const Stmt &body, uint16_t expected_pipe_mask) {
    PerCoreTaskBodyValidator validator(expected_pipe_mask);
    validator(body);
    ICHECK(validator.has_dependency_task_)
        << "T.PerCoreTask must contain at least one dependency-bearing task "
           "after task normalization";
  }

private:
  explicit PerCoreTaskBodyValidator(uint16_t expected_pipe_mask)
      : expected_pipe_mask_(expected_pipe_mask) {}

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kAscendTask) {
      ValidatedTask task = ValidateTask(op, /*inside_per_core_task=*/true);
      if (!task.has_buffer_access) {
        ICHECK(IsPerCoreTaskScalarControl(op))
            << "T.PerCoreTask only permits scalar Bind/condition/loop control "
               "and inter-core synchronization in addition to operations on "
               "its inferred pipe; got "
            << GetRef<Stmt>(op);
        return;
      }
      ICHECK_NE(task.pipe_mask, 0)
          << "Cannot infer an Ascend pipe for T.PerCoreTask candidate "
          << GetRef<Stmt>(op);
      ICHECK_EQ(task.pipe_mask & (task.pipe_mask - 1), 0)
          << "Each dependency-bearing statement in T.PerCoreTask must use "
             "exactly one hardware pipe, but statement mask="
          << task.pipe_mask << ", statement=" << GetRef<Stmt>(op);
      ICHECK_EQ(task.pipe_mask, expected_pipe_mask_)
          << "T.PerCoreTask candidate pipe does not match its inferred pipe. "
             "Inferred mask="
          << expected_pipe_mask_ << ", statement mask=" << task.pipe_mask
          << ", statement=" << GetRef<Stmt>(op);
      has_dependency_task_ = true;
      return;
    }
    ICHECK_NE(op->attr_key, tl::attr::kAscendPerCoreTask)
        << "Nested T.PerCoreTask regions are not supported";
    ICHECK_NE(op->attr_key, attr::kScheduleUnit)
        << "T.PerCoreTask contains an already-materialized schedule unit";
    ICHECK(IsScheduleGuardAttribute(op))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside T.PerCoreTask. Unsupported attr_key: "
        << op->attr_key;
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
  }

  void VisitStmt_(const BufferStoreNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
  }

  void VisitStmt_(const BindNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
  }

  void VisitStmt_(const WhileNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "While");
  }

  void VisitStmt_(const SBlockNode *) final {
    FatalUnmarkedTask("MaterializeScheduleUnits", "block");
  }

  uint16_t expected_pipe_mask_{0};
  bool has_dependency_task_{false};
};

void ValidatePerCoreTask(const AttrStmtNode *op) {
  ICHECK_EQ(op->attr_key, tl::attr::kAscendPerCoreTask);
  TaskResourceUsage resource_usage = AnalyzeTaskResourceUsage(op->body);
  uint16_t pipe_mask = resource_usage.pipe_mask;
  ICHECK_NE(pipe_mask, 0) << "Cannot infer an Ascend pipe for T.PerCoreTask";
  ICHECK_EQ(pipe_mask & (pipe_mask - 1), 0)
      << "T.PerCoreTask requires exactly one inferred hardware pipe, but its "
         "mask is "
      << pipe_mask;
  PerCoreTaskBodyValidator::Validate(op->body, pipe_mask);
  PerCoreTaskControlValidator control_validator(pipe_mask);
  control_validator(op->body);
}

// Lower normalized Task TIR into the canonical scheduled-TIR grammar without
// making scheduling decisions. A child list without T.Stage uses stage -1
// throughout. If any child has a requested non-negative stage, unannotated
// siblings are normalized to stage 0 so AutoSchedule can identify manual mode
// from a uniform per-list invariant.
class ScheduleUnitMaterializer {
public:
  static Stmt Rewrite(const Stmt &body) {
    ScheduleUnitMaterializer materializer;
    std::vector<Stmt> units = materializer.MaterializeChildList(body);
    return MakeSequence(units);
  }

private:
  // Only kernel roots and loop bodies define child-list boundaries. Conditions
  // and attributes flatten into their enclosing list and are normalized there.
  std::vector<Stmt> MaterializeChildList(const Stmt &stmt) {
    std::vector<Stmt> units = MaterializeList(stmt);
    EnsureNonEmpty(&units);
    NormalizeRequestedStages(&units);
    return units;
  }

  std::vector<Stmt> MaterializeList(const Stmt &stmt) {
    if (!stmt.defined())
      return {};
    if (const auto *sequence = stmt.as<SeqStmtNode>()) {
      std::vector<Stmt> result;
      for (const Stmt &child : sequence->seq) {
        std::vector<Stmt> child_units = MaterializeList(child);
        result.insert(result.end(), child_units.begin(), child_units.end());
      }
      return result;
    }
    if (const auto *loop = stmt.as<ForNode>())
      return MaterializeLoop(loop);
    if (const auto *condition = stmt.as<IfThenElseNode>())
      return MaterializeCondition(condition);
    if (const auto *attribute = stmt.as<AttrStmtNode>())
      return MaterializeAttribute(attribute);
    if (stmt.as<EvaluateNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Evaluate");
    if (stmt.as<BufferStoreNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "BufferStore");
    if (stmt.as<BindNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "Bind");
    if (stmt.as<WhileNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "While");
    if (stmt.as<SBlockNode>())
      FatalUnmarkedTask("MaterializeScheduleUnits", "block");
    LOG(FATAL) << "MaterializeScheduleUnits found an unsupported statement: "
               << stmt;
  }

  std::vector<Stmt> MaterializeLoop(const ForNode *op) {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)
      FatalUnmarkedTask("MaterializeScheduleUnits", "parallel loop");
    std::vector<Stmt> body = MaterializeChildList(op->body);
    For loop = GetRef<For>(op);
    loop.CopyOnWrite()->body = MakeSequence(body);
    return {MakeInitialScheduleUnit(std::move(loop))};
  }

  std::vector<Stmt> MaterializeCondition(const IfThenElseNode *op) {
    std::vector<Stmt> result = MaterializeList(op->then_case);
    for (Stmt &unit : result)
      unit = PrependConditionGuard(unit, op->condition);
    if (!op->else_case.defined())
      return result;

    std::vector<Stmt> else_units = MaterializeList(op->else_case.value());
    PrimExpr else_condition = Not(op->condition);
    for (Stmt &unit : else_units)
      unit = PrependConditionGuard(unit, else_condition);
    result.insert(result.end(), else_units.begin(), else_units.end());
    return result;
  }

  std::vector<Stmt> MaterializeAttribute(const AttrStmtNode *op) {
    ICHECK_NE(op->attr_key, attr::kScheduleUnit)
        << "MaterializeScheduleUnits cannot run on an already-materialized "
           "schedule";
    if (op->attr_key == tl::attr::kAscendTask) {
      ValidateTask(op, /*inside_per_core_task=*/false);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (op->attr_key == tl::attr::kAscendPerCoreTask) {
      ValidatePerCoreTask(op);
      return {MakeInitialScheduleUnit(GetRef<Stmt>(op))};
    }
    if (op->attr_key == tl::attr::kAscendStage) {
      PrimExpr stage_expr = op->node.cast<PrimExpr>();
      const auto *stage = stage_expr.as<IntImmNode>();
      ICHECK(stage != nullptr && stage->value >= 0 &&
             stage->value <= std::numeric_limits<int>::max())
          << "T.Stage requires a non-negative constant integer, got "
          << stage_expr;
      std::vector<Stmt> body = MaterializeList(op->body);
      EnsureNonEmpty(&body);
      for (Stmt &unit : body)
        unit = AssignStage(unit, static_cast<int>(stage->value));
      return body;
    }
    ICHECK(IsScheduleGuardAttribute(op))
        << "MaterializeScheduleUnits only supports known scheduling AttrStmt "
           "wrappers inside the schedulable tilelang_root body. Unsupported "
           "attr_key: "
        << op->attr_key;
    std::vector<Stmt> body = MaterializeList(op->body);
    EnsureNonEmpty(&body);
    for (Stmt &unit : body)
      unit = PrependAttributeGuard(unit, op);
    return body;
  }

  void EnsureNonEmpty(std::vector<Stmt> *units) {
    ICHECK(units != nullptr);
    if (units->empty())
      units->push_back(MakeInitialScheduleUnit(MakeNoOpTask()));
  }

  void NormalizeRequestedStages(std::vector<Stmt> *units) {
    ICHECK(units != nullptr);
    bool manual_schedule =
        std::any_of(units->begin(), units->end(), [](const Stmt &unit) {
          return GetScheduleUnitStage(RequireScheduleUnit(unit).get()) !=
                 kUnscheduledStage;
        });
    if (!manual_schedule)
      return;
    for (Stmt &unit : *units) {
      if (GetScheduleUnitStage(RequireScheduleUnit(unit).get()) ==
          kUnscheduledStage) {
        unit = AssignStage(unit, 0);
      }
    }
  }
};

} // namespace

/*! \brief Normalize task boundaries and materialize unscheduled units. */
tvm::transform::Pass MaterializeScheduleUnits() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(func), "MaterializeScheduleUnits",
        [](const TilelangKernelContext &context) {
          SBlock root = TaskNormalizer::Rewrite(context.root);
          Stmt normalized = root->body;
          root.CopyOnWrite()->body =
              ScheduleUnitMaterializer::Rewrite(normalized);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MaterializeScheduleUnits", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MaterializeScheduleUnits",
                        MaterializeScheduleUnits);
}

} // namespace tl
} // namespace tvm
