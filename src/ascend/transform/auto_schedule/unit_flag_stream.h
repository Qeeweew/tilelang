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

#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "./dependency_analysis.h"
#include "./ir_structure.h"
#include "./multi_buffer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/copy.h"
#include "ascend/op/cube_load.h"
#include "ascend/op/utils.h"
#include "ascend/transform/buffer_version.h"
#include "op/gemm.h"

namespace tvm::tl::ascend {

// A proof local to InsertSync. Unit protection covers C data only; direction
// setup/restore and a real terminal FIX->M drain are separate obligations of
// the caller. No durable user annotation can manufacture this proof.
struct UnitFlagStreamPlan {
  ControlNode *result_loop{nullptr};
  Buffer c;
  std::vector<TaskNode *> mads;
  TaskNode *fix{nullptr};
  std::vector<PrimExpr> mad_controls;
};

namespace unit_flag_stream_detail {

constexpr int64_t kMaxProofIterations = 4096;

inline std::optional<int64_t> Constant(const PrimExpr &expr) {
  PrimExpr value = arith::Analyzer().Simplify(expr);
  if (const auto *integer = value.as<IntImmNode>())
    return integer->value;
  return std::nullopt;
}

inline bool DefaultUnit(const Map<String, ObjectRef> &annotations) {
  auto control = annotations.Get("unit_flag_ctrl");
  return !control.has_value() ||
         (control.value().as<PrimExprNode>() &&
          Constant(Downcast<PrimExpr>(control.value())) == 0);
}

// One typed operation, optionally enclosed in its semantic task marker.
// Compound tasks, hidden loops and per-core candidate groups are outside the
// first selection contract.
inline std::optional<Call> Operation(const TaskNode *task) {
  if (task->IsPerCoreTask() || task->GetStage() != 0 ||
      task->GetCoreMask() != kCoreCube)
    return std::nullopt;
  Stmt body = task->stmt;
  while (const auto *attribute = body.as<AttrStmtNode>()) {
    if (attribute->attr_key != "tl.ascend_task")
      return std::nullopt;
    body = attribute->body;
  }
  const auto *evaluate = body.as<EvaluateNode>();
  if (!evaluate || !evaluate->value.as<CallNode>())
    return std::nullopt;
  return Downcast<Call>(evaluate->value);
}

inline bool StaticLoopBounds(const ControlNode *loop, int64_t *count) {
  auto min = Constant(loop->control->min);
  auto extent = Constant(loop->control->extent);
  auto step = loop->control->step.has_value()
                  ? Constant(loop->control->step.value())
                  : std::optional<int64_t>(1);
  // Avoid pipeline offsets, skipped terminal iterations and counter clocks.
  if (loop->GetStage() != 0 ||
      loop->control->kind != ForKind::kSerial ||
      !min || *min != 0 || !extent || *extent <= 0 ||
      *extent > kMaxProofIterations || !step || *step != 1)
    return false;
  for (const auto &[key, value] : loop->control->annotations) {
    if (key != kMultiBufferEligible && key != kMultiBufferCounterMap &&
        key != kStorageEpochGuardMap)
      return false;
  }
  *count = *extent;
  return true;
}

inline bool StaticLoop(const ControlNode *loop, int64_t *count) {
  return !loop->HasConditions() && StaticLoopBounds(loop, count);
}

// A result's own activation guard may be moved to one lexical shell around
// the entire stream and its six real boundary gates. Only immutable outer
// loop values and their pure Bind snapshots are accepted. In particular, an
// opaque call or a captured mutable load is not an activation proof, even if
// an Analyzer could simplify a relation involving that value.
inline bool StableActivationGuards(const ControlNode *result) {
  if (!result->HasConditions())
    return false;
  std::unordered_set<const VarNode *> outer_vars;
  for (const ControlNode *parent = result->GetParentControl(); parent;
       parent = parent->GetParentControl())
    outer_vars.insert(parent->control->loop_var.get());
  std::unordered_set<const VarNode *> changing_vars;
  for (const Var &var : result->GetWriteVars())
    changing_vars.insert(var.get());
  std::function<void(const ControlNode *)> collect_loops =
      [&](const ControlNode *loop) {
        changing_vars.insert(loop->control->loop_var.get());
        for (const auto &child : loop->children)
          if (child->IsControl())
            collect_loops(static_cast<const ControlNode *>(child.get()));
      };
  collect_loops(result);

  std::unordered_set<const VarNode *> resolving;
  std::unordered_set<const VarNode *> stable_vars;
  std::function<bool(const VarNode *)> stable_var;
  class PureOuterExpression : public ExprVisitor {
  public:
    explicit PureOuterExpression(
        const std::function<bool(const VarNode *)> &stable_var)
        : stable_var_(stable_var) {}
    bool valid{true};

  private:
    void VisitExpr_(const VarNode *op) final { valid &= stable_var_(op); }
    void VisitExpr_(const CallNode *) final { valid = false; }
    void VisitExpr_(const BufferLoadNode *) final { valid = false; }
    void VisitExpr_(const ProducerLoadNode *) final { valid = false; }
    void VisitExpr_(const ReduceNode *) final { valid = false; }
    const std::function<bool(const VarNode *)> &stable_var_;
  };
  stable_var = [&](const VarNode *var) {
    if (!var->dtype.is_int() && !var->dtype.is_uint() && !var->dtype.is_bool())
      return false;
    if (var->dtype.lanes() != 1 || changing_vars.count(var))
      return false;
    if (outer_vars.count(var) || stable_vars.count(var))
      return true;
    if (!resolving.insert(var).second)
      return false;
    const PrimExpr *definition = nullptr;
    for (const Constr &constraint : result->task->outer_ctx.constrs_) {
      if (constraint.kind != Constr::kBindValue ||
          constraint.var.get() != var)
        continue;
      if (definition != nullptr)
        return false;
      definition = &constraint.value;
    }
    if (!definition)
      return false;
    PureOuterExpression expression(stable_var);
    expression(*definition);
    resolving.erase(var);
    if (expression.valid)
      stable_vars.insert(var);
    return expression.valid;
  };
  for (const auto &guard : result->GetGuards()) {
    // Do not move binding, assume or other attribute scopes with the guard.
    if (!guard->IsCondition())
      return false;
    const PrimExpr &condition =
        static_cast<const ConditionGuard *>(guard.get())->condition;
    PureOuterExpression expression(stable_var);
    expression(condition);
    if (!expression.valid)
      return false;
  }
  return true;
}

inline bool StaticResultLoop(const ControlNode *loop, int64_t *count) {
  return StaticLoopBounds(loop, count) &&
         (!loop->HasConditions() || StableActivationGuards(loop));
}

// A full-M, zero-origin column prefix of canonical NZ storage keeps base zero
// and the backing M pitch. MAD's actual M and FIX's source stride therefore
// agree even when the allocation has more columns. AnalyzeUnitFlagStreams
// additionally requires every MAD and its unique FIX to use this same domain.
inline bool AdmissibleCDomain(const Buffer &buffer, const Array<Range> &ranges) {
  if (!IsL0CBuffer(buffer) || buffer->dtype != DataType::Float(32) ||
      buffer->shape.size() != 2 || ranges.size() != 2 ||
      !buffer->strides.empty() || !buffer->axis_separators.empty() ||
      Constant(buffer->elem_offset) != 0)
    return false;
  for (size_t axis = 0; axis < 2; ++axis) {
    auto capacity = Constant(buffer->shape[axis]);
    auto extent = Constant(ranges[axis]->extent);
    if (!capacity || *capacity <= 0 || *capacity % 16 != 0 ||
        !extent || *extent <= 0 || *extent % 16 != 0 ||
        Constant(ranges[axis]->min) != 0 || *extent > *capacity ||
        (axis == 0 && *extent != *capacity))
      return false;
  }
  // dav-2201 has 128 KiB of FP32 accumulator storage. Division avoids
  // overflowing a product when a synthetic input bypasses resource planning.
  return *Constant(buffer->shape[0]) <= 32768 / *Constant(buffer->shape[1]);
}

// The ordinary L0 frontend may serialize allocation M/N for a region operand.
// GemmMAD::_l0_operation_extents emits native M/N/K from the participating
// regions. Prove its same three M/N/K equalities for the static stream domain.
inline bool NativeMadMatchesC(const Gemm &gemm,
                             const Array<Range> &domain) {
  const auto &a = gemm->aRegion_->region;
  const auto &b = gemm->bRegion_->region;
  const auto &c = gemm->cRegion_->region;
  if (a.size() < 2 || b.size() < 2 || c.size() != 2 || domain.size() != 2)
    return false;
  auto m = Constant(c[0]->extent);
  auto n = Constant(c[1]->extent);
  auto k = Constant(a.back()->extent); // transA is rejected by Collect.
  if (!m || !n || !k || *m <= 0 || *n <= 0 || *k <= 0 ||
      *m % 16 || *n % 16 || *k % 16 ||
      m != Constant(domain[0]->extent) ||
      n != Constant(domain[1]->extent) ||
      Constant(a[a.size() - 2]->extent) != m ||
      Constant(b[gemm->transB_ ? b.size() - 2 : b.size() - 1]->extent) != n ||
      Constant(b[gemm->transB_ ? b.size() - 1 : b.size() - 2]->extent) != k)
    return false;
  return true;
}

// Raw pointers, scalar C accesses and any differently-shaped Buffer alias
// invalidate selection, even when their access detector happens to produce
// the same bounding region. Ordinary typed C use outside a stream is retained
// and remains protected by real boundary dependencies/drain.
class StorageAudit : public StmtExprVisitor {
public:
  explicit StorageAudit(Buffer c) : c_(std::move(c)) {}
  bool valid{true};

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer->data.same_as(c_->data))
      valid = false;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer->data.same_as(c_->data))
      valid = false;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    std::vector<BufferRegion> regions;
    if (op->op.same_as(Gemm::Get())) {
      Gemm gemm(op->args, op->annotations);
      regions = {gemm->aRegion_, gemm->bRegion_, gemm->cRegion_};
    } else if (IsAscendCopyCall(op)) {
      AscendCopy copy(op->args, op->annotations);
      regions = {BufferRegion(copy->src, copy->src_range),
                 BufferRegion(copy->dst, copy->dst_range)};
    } else {
      bool raw_c = false;
      PostOrderVisit(GetRef<Call>(op), [&](const ObjectRef &node) {
        if (const auto *var = node.as<VarNode>())
          raw_c |= var == c_->data.get();
        if (const auto *load = node.as<BufferLoadNode>())
          raw_c |= load->buffer->data.same_as(c_->data);
      });
      if (raw_c)
        valid = false;
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    for (const BufferRegion &region : regions) {
      if (region->buffer->data.same_as(c_->data) &&
          !region->buffer.same_as(c_))
        valid = false;
      for (const Range &range : region->region) {
        VisitExpr(range->min);
        VisitExpr(range->extent);
      }
    }
    for (const auto &[key, value] : op->annotations)
      if (value.as<PrimExprNode>())
        VisitExpr(Downcast<PrimExpr>(value));
  }
  Buffer c_;
};

struct Mad {
  TaskNode *task;
  Call call;
  Gemm gemm;
  std::vector<ControlNode *> loops;
};

// PrepareMultiBuffer owns these clocks and ReadMultiBufferPlan has validated
// their owner/group contract before this analysis. Only retain counters of
// this result/native-K subtree's ordinary A/B input storage, never C clocks.
inline bool AddInputCounters(ControlNode *loop, std::vector<Buffer> *counters) {
  auto annotation = loop->control->annotations.Get(kMultiBufferCounterMap);
  if (!annotation)
    return true;
  auto map = annotation.value().cast<MultiBufferCounterMap>();
  auto guards = loop->control->annotations.Get(kStorageEpochGuardMap);
  if (!guards)
    return false;
  auto guard_map = guards.value().cast<StorageEpochGuardMap>();
  auto reads = loop->GetReadRegions();
  auto writes = loop->GetWriteRegions();
  reads.insert(reads.end(), writes.begin(), writes.end());
  for (const auto &[storage, counter] : map) {
    auto guard = guard_map.Get(storage);
    if (!guard || Constant(guard.value()) != 1 ||
        counter.scope() != "local.var" || counter->dtype != DataType::Int(32) ||
        counter->shape.size() != 1 || Constant(counter->shape[0]) != 1 ||
        !counter->strides.empty() || Constant(counter->elem_offset) != 0)
      return false;
    bool input = false;
    for (const BufferRegion &region : reads) {
      if (!region->buffer->data.same_as(storage))
        continue;
      if (!IsL1Buffer(region->buffer) && !IsL0ABuffer(region->buffer) &&
          !IsL0BBuffer(region->buffer))
        return false;
      input = true;
    }
    if (!input)
      return false;
    Buffer candidate = counter;
    if (std::none_of(counters->begin(), counters->end(), [&](const Buffer &old) {
          return old.same_as(candidate);
        }))
      counters->push_back(counter);
  }
  return true;
}

inline bool InputCounterCommit(const TaskNode *task,
                               const std::vector<Buffer> &counters) {
  if (task->IsPerCoreTask() || task->HasConditions() || task->GetStage() != 0 ||
      task->GetCoreMask() != kCoreCube ||
      !task->UsesOnlyPipe(ResourcePipe::kScalar))
    return false;
  Stmt body = task->stmt;
  while (const auto *attribute = body.as<AttrStmtNode>()) {
    if (attribute->attr_key != "tl.ascend_task")
      return false;
    body = attribute->body;
  }
  const auto *store = body.as<BufferStoreNode>();
  if (!store || store->predicate.has_value() || store->indices.size() != 1 ||
      Constant(store->indices[0]) != 0 ||
      std::none_of(counters.begin(), counters.end(), [&](const Buffer &counter) {
        return counter.same_as(store->buffer);
      }))
    return false;
  const auto *add = store->value.as<AddNode>();
  if (!add || Constant(add->b) != 1)
    return false;
  const auto *load = add->a.as<BufferLoadNode>();
  return load && load->buffer.same_as(store->buffer) &&
         load->indices.size() == 1 && Constant(load->indices[0]) == 0 &&
         !load->predicate.has_value();
}

// This walk accepts input DMA followed by a finite native-K chain and one
// complete export. It does not infer a protocol from arbitrary C accesses.
inline bool Collect(ControlNode *result, std::vector<Mad> *mads,
                    TaskNode **fix, std::optional<AscendCopy> *copy) {
  bool valid = true;
  std::vector<Buffer> result_counters;
  if (!AddInputCounters(result, &result_counters))
    return false;
  std::function<void(IRStructure *, std::vector<ControlNode *>,
                     std::vector<Buffer>)> walk =
      [&](IRStructure *node, std::vector<ControlNode *> loops,
          std::vector<Buffer> counters) {
        if (!valid)
          return;
        if (node->IsControl()) {
          auto *control = static_cast<ControlNode *>(node);
          int64_t count;
          if (control->HasConditions() || !StaticLoop(control, &count) ||
              *fix != nullptr || !AddInputCounters(control, &counters)) {
            valid = false;
            return;
          }
          loops.push_back(control);
          for (const auto &child : control->children)
            walk(child.get(), loops, counters);
          return;
        }
        auto *task = static_cast<TaskNode *>(node);
        if (InputCounterCommit(task, counters))
          return;
        auto operation = Operation(task);
        if (!operation) {
          valid = false;
          return;
        }
        const Call &call = operation.value();
        if (call->op.same_as(Gemm::Get())) {
          Gemm gemm(call->args, call->annotations);
          if (*fix || task->HasConditions() || !DefaultUnit(call->annotations) ||
              !IsL0ABuffer(gemm->a_) || !IsL0BBuffer(gemm->b_) ||
              gemm->a_->dtype != gemm->b_->dtype ||
              (gemm->a_->dtype != DataType::Float(16) &&
               !gemm->a_->dtype.is_bfloat16()) ||
              gemm->transA_ || gemm->mbar_.defined() || gemm->wgWait_ != 0 ||
              gemm->isWgmma_ || gemm->isTcgen05_ || gemm->kPack_ != 1 ||
              gemm->m_ <= 0 || gemm->n_ <= 0 ||
              gemm->k_ <= 0 || gemm->m_ % 16 || gemm->n_ % 16 ||
              gemm->k_ % 16 ||
              !AdmissibleCDomain(gemm->c_, gemm->cRegion_->region)) {
            valid = false;
            return;
          }
          for (const auto &[key, value] : call->annotations)
            if (key != "unit_flag_ctrl")
              valid = false;
          mads->push_back({task, call, gemm, std::move(loops)});
          return;
        }
        if (IsAscendCopyCall(call.get())) {
          AscendCopy typed(call->args, call->annotations);
          if (IsL0CBuffer(typed->src)) {
            if (*fix || mads->empty() || !loops.empty() || task->HasConditions() ||
                !IsGlobalBuffer(typed->dst) || !DefaultUnit(call->annotations) ||
                typed->dual_dst_ctl || typed->transpose || typed->data_select ||
                (typed->sub_blockid.defined() &&
                 Constant(typed->sub_blockid) != 0) ||
                typed->dst_block.has_value() ||
                typed->pad_value.has_value() ||
                typed->src_oob_safe_value.has_value() ||
                !AdmissibleCDomain(typed->src, typed->src_range) ||
                typed->dst_range.size() < 2 ||
                (typed->dst->dtype != DataType::Float(32) &&
                 typed->dst->dtype != DataType::Float(16) &&
                 !typed->dst->dtype.is_bfloat16())) {
              valid = false;
              return;
            }
            for (const auto &[key, value] : call->annotations)
              if (key != "unit_flag_ctrl" && key != "l2_cache_ctrl" &&
                  key != kRootFixpipeConfig)
                valid = false;
            size_t leading = typed->dst_range.size() - 2;
            arith::Analyzer analyzer;
            task->outer_ctx.Populate(analyzer);
            for (size_t i = 0; i < typed->dst_range.size(); ++i) {
              const Range &range = typed->dst_range[i];
              if (!CanProveAscendRangeInBounds(range->min, range->extent,
                                               typed->dst->shape[i], &analyzer) ||
                  (i < leading && Constant(range->extent) != 1))
                valid = false;
            }
            for (size_t i = 0; i < typed->src_range.size(); ++i)
              if (Constant(typed->src_range[i]->extent) !=
                  Constant(typed->dst_range[i + leading]->extent))
                valid = false;
            *fix = task;
            *copy = typed;
            return;
          }
        }
        // Input DMA may be early, but another Cube/FIX/Scalar operation is
        // not part of this closed result transaction.
        if (*fix ||
            (!IsAscendCopyCall(call.get()) &&
             !call->op.same_as(AscendCubeLoad3D::Get())) ||
            (!task->UsesOnlyPipe(ResourcePipe::kMTE1) &&
                     !task->UsesOnlyPipe(ResourcePipe::kMTE2)))
          valid = false;
      };
  for (const auto &child : result->children)
    walk(child.get(), {}, result_counters);
  return valid && *fix != nullptr && !mads->empty();
}

} // namespace unit_flag_stream_detail

inline std::vector<UnitFlagStreamPlan> AnalyzeUnitFlagStreams(
    const std::vector<std::shared_ptr<IRStructure>> &root,
    const SBlock &kernel_root, const BufferVersionMap &buffer_versions,
    const BufferVersionMap &manual_versions) {
  using namespace unit_flag_stream_detail;
  std::vector<UnitFlagStreamPlan> plans;
  std::function<void(IRStructure *)> visit = [&](IRStructure *node) {
    if (!node->IsControl())
      return;
    auto *result = static_cast<ControlNode *>(node);
    int64_t result_count;
    std::vector<Mad> mads;
    TaskNode *fix = nullptr;
    std::optional<AscendCopy> copy;
    bool selected = StaticResultLoop(result, &result_count) &&
                    Collect(result, &mads, &fix, &copy);
    if (selected) {
      Buffer c = mads.front().gemm->c_;
      auto version = buffer_versions.Get(c->data);
      auto manual = manual_versions.Get(c->data);
      selected = (!version || version.value() == 1) &&
                 (!manual || manual.value() == 1) &&
                 copy.value()->src.same_as(c);
      bool backing = false;
      for (const Buffer &allocation : kernel_root->alloc_buffers)
        backing |= allocation.same_as(c);
      auto maps = kernel_root->annotations.Get("layout_map");
      if (!maps || !backing) {
        selected = false;
      } else {
        auto map = maps.value().cast<Map<ObjectRef, Layout>>();
        auto layout = map.Get(c);
        selected &= layout.has_value() &&
                    layout.value()->IsEqual(MakeAscendL0CLayout(c).get());
      }
      // An eligible storage can have a prepared counter clock even at one
      // physical slot. This first stream only admits a fixed single C.
      std::function<void(IRStructure *)> check_owner = [&](IRStructure *part) {
        if (!part->IsControl())
          return;
        auto *loop = static_cast<ControlNode *>(part);
        if (auto eligible = loop->control->annotations.Get(kMultiBufferEligible))
          for (const Var &storage : eligible.value().cast<Array<Var>>())
            if (storage.same_as(c->data))
              selected = false;
        if (auto counters = loop->control->annotations.Get(kMultiBufferCounterMap))
          if (counters.value().cast<MultiBufferCounterMap>().count(c->data))
            selected = false;
        for (const auto &child : loop->children)
          check_owner(child.get());
      };
      for (const auto &part : root)
        check_owner(part.get());
      StorageAudit audit(c);
      audit(kernel_root->body);
      selected &= audit.valid;
      for (const Mad &mad : mads)
        selected &= mad.gemm->c_.same_as(c) &&
                    NativeMadMatchesC(mad.gemm, copy.value()->src_range);
      std::vector<TaskNode *> tasks;
      CollectAllTaskNodes(result, tasks);
      for (TaskNode *task : tasks) {
        bool mad = std::any_of(mads.begin(), mads.end(), [&](const Mad &item) {
          return item.task == task;
        });
        if (task != fix && !mad && task->TouchesStorage(c->data))
          selected = false;
      }

      // Enumerate only the small static native-K chain. The output loop and
      // enclosing owner/epoch are never unrolled or used as a proof clock.
      std::vector<std::pair<size_t, Map<Var, PrimExpr>>> occurrences;
      for (size_t index = 0; selected && index < mads.size(); ++index) {
        std::function<void(size_t, Map<Var, PrimExpr>)> expand =
            [&](size_t depth, Map<Var, PrimExpr> bindings) {
              if (occurrences.size() >= kMaxProofIterations) {
                selected = false;
                return;
              }
              if (depth == mads[index].loops.size()) {
                occurrences.emplace_back(index, std::move(bindings));
                return;
              }
              ControlNode *loop = mads[index].loops[depth];
              int64_t count;
              if (!StaticLoop(loop, &count)) {
                selected = false;
                return;
              }
              for (int64_t i = 0; selected && i < count; ++i) {
                auto next = bindings;
                next.Set(loop->control->loop_var,
                         make_const(loop->control->loop_var.dtype(), i));
                expand(depth + 1, std::move(next));
              }
            };
        expand(0, {});
      }
      for (size_t i = 0; selected && i < occurrences.size(); ++i) {
        const auto &[index, bindings] = occurrences[i];
        selected &= Constant(Substitute(mads[index].gemm->clearAccum_,
                                        bindings)) == (i == 0 ? 1 : 0);
      }
      if (selected) {
        UnitFlagStreamPlan plan{result, c, {}, fix, {}};
        size_t last = occurrences.back().first;
        for (size_t i = 0; i < mads.size(); ++i) {
          PrimExpr control = IntImm(DataType::Int(32), 2);
          if (i == last) {
            PrimExpr terminal = Bool(true);
            for (ControlNode *loop : mads[i].loops) {
              int64_t count;
              ICHECK(StaticLoop(loop, &count));
              terminal = terminal &&
                         (loop->control->loop_var ==
                          make_const(loop->control->loop_var.dtype(), count - 1));
            }
            control = arith::Analyzer().Simplify(
                Select(terminal, IntImm(DataType::Int(32), 3), control));
          }
          plan.mads.push_back(mads[i].task);
          plan.mad_controls.push_back(std::move(control));
        }
        plans.push_back(std::move(plan));
        return; // Never claim a nested chain as another result stream.
      }
    }
    for (const auto &child : result->children)
      visit(child.get());
  };
  for (const auto &node : root)
    visit(node.get());
  return plans;
}

inline void ApplyUnitFlagStreams(const std::vector<UnitFlagStreamPlan> &plans) {
  class SetControl : public StmtExprMutator {
  public:
    explicit SetControl(PrimExpr control) : control_(std::move(control)) {}
  private:
    PrimExpr VisitExpr_(const CallNode *op) final {
      if (!op->op.same_as(Gemm::Get()) && !IsAscendCopyCall(op))
        return StmtExprMutator::VisitExpr_(op);
      Call call = GetRef<Call>(op);
      call.CopyOnWrite()->annotations.Set("unit_flag_ctrl", control_);
      return call;
    }
    PrimExpr control_;
  };
  for (const UnitFlagStreamPlan &plan : plans) {
    ICHECK_EQ(plan.mads.size(), plan.mad_controls.size());
    for (size_t i = 0; i < plan.mads.size(); ++i)
      plan.mads[i]->stmt = SetControl(plan.mad_controls[i])(plan.mads[i]->stmt);
    plan.fix->stmt = SetControl(IntImm(DataType::Int(32), 3))(plan.fix->stmt);
  }
}

inline bool CoversUnitFlagDependency(
    const std::vector<UnitFlagStreamPlan> &plans, ControlNode *analysis_loop,
    const DepInfo &dep, TaskNode *producer, TaskNode *consumer) {
  if (dep.kind != DependencyKind::kData || !dep.storage.has_value())
    return false;
  for (const UnitFlagStreamPlan &plan : plans) {
    if (analysis_loop != plan.result_loop ||
        !dep.storage.value().same_as(plan.c->data))
      continue;
    auto is_mad = [&](TaskNode *task) {
      return std::find(plan.mads.begin(), plan.mads.end(), task) !=
             plan.mads.end();
    };
    if (dep.distance == 0 && is_mad(producer) && consumer == plan.fix)
      return true;
    if (dep.distance == -1 && producer == plan.fix && is_mad(consumer))
      return true;
  }
  return false;
}

} // namespace tvm::tl::ascend
