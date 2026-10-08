// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include <tvm/arith/pattern.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "../aiv_private.h"
#include "./ir_structure.h"
#include "ascend/op/copy.h"

namespace tvm::tl::ascend {

// A pass-local fact established by AnalyzeOwnerResults/AnalyzeOwnerInputs.
// For results the AIC producer traverses owners and the AIV consumer selects
// SID; for inputs those roles are reversed. Both directions carry the same
// canonical storage, exact coverage, finite activation, slot separation and
// two-participant proof. Consumers may select/move facts, but must not mutate
// them or retain them across changes to tasks, owners, regions or guards.
// This is not a dependency/completion edge: a caller must independently prove
// read-completion release before replacing the collective protocol.
struct ValidatedOwnerTransfer {
  enum class Direction { kInput, kResult };
  enum class Endpoint { kProducer, kConsumer };

  Var storage;
  TaskNode *producer;
  TaskNode *consumer;
  ControlNode *epoch;
  ControlNode *owner;
  Var sid;
  int distance;
  int num_slots;
  PrimExpr producer_slot;
  PrimExpr consumer_slot;
  PrimExpr producer_activation;
  PrimExpr consumer_activation;
  Direction direction;
  // Optional placement facts proved once from the original guards. A valid
  // transfer need not support either narrow GM cut.
  std::optional<PrimExpr> producer_active_epochs;
  std::optional<PrimExpr> consumer_active_epochs;

  bool IsInput() const { return direction == Direction::kInput; }
  bool IsResult() const { return direction == Direction::kResult; }

  // Identity within one immutable scheduled snapshot. Slot expressions and
  // activations belong to the unique validated endpoint pair, rather than
  // being reconstructed by each placement consumer.
  bool MatchesTransfer(const ValidatedOwnerTransfer &other) const {
    return direction == other.direction && storage.same_as(other.storage) &&
           producer == other.producer && consumer == other.consumer &&
           epoch == other.epoch && owner == other.owner &&
           sid.same_as(other.sid) && distance == other.distance &&
           num_slots == other.num_slots;
  }

  // The opposite side of this transfer is the existing paired GM WAR. This
  // only identifies its original lexical protocol; it grants no endpoint
  // placement, completion, alias or event-sharing permission.
  bool MatchesPairedWar(ControlNode *flag_loop, int war_distance, int width,
                        size_t counter_group) const {
    return flag_loop == epoch && counter_group == 0 &&
           war_distance == num_slots - distance && war_distance > 0 &&
           width == war_distance;
  }

  // Read the proven nonempty interval using only the outer epoch variable.
  // Local Bind names never escape to an inactive wide END/BEGIN site.
  const std::optional<PrimExpr> &ActiveEpochs(Endpoint endpoint) const {
    return endpoint == Endpoint::kProducer ? producer_active_epochs
                                          : consumer_active_epochs;
  }
};

namespace owner_result_detail {

constexpr size_t kProofLimit = 256;
using Bindings = ffi::Map<Var, PrimExpr>;

inline bool Uses(const PrimExpr &expr, const Var &var) {
  return UsesVar(expr, [&](const VarNode *node) { return node == var.get(); });
}

// Region addresses and activation must be immutable arithmetic. In
// particular, never turn a captured mutable read into a lasting Bind fact.
inline bool Immutable(const PrimExpr &expr) {
  bool safe = true;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (node.as<BufferLoadNode>() || node.as<CallNode>())
      safe = false;
  });
  return safe;
}

inline PrimExpr Expand(const PrimExpr &expr, const Bindings &bindings) {
  arith::Analyzer analyzer;
  return analyzer.Simplify(Substitute(expr, bindings));
}

inline Bindings TaskBindings(const TaskNode *task) {
  Bindings result;
  for (const Constr &constraint : task->outer_ctx.constrs_) {
    if (constraint.kind == Constr::kBindValue && Immutable(constraint.value))
      result.Set(constraint.var, Expand(constraint.value, result));
  }
  return result;
}

inline std::optional<int64_t> Constant(const PrimExpr &expr) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expr);
  if (const auto *value = simplified.as<IntImmNode>())
    return value->value;
  return std::nullopt;
}

// Shared finite epoch bounds for the owner proof and its placement consumers.
// Simplify before entering the loop or its activation: zero/mutable domains
// must not become positive through the endpoint's own guard.
inline std::optional<std::pair<int64_t, int64_t>>
FiniteEpochBounds(ControlNode *epoch) {
  if (!epoch || epoch->control->kind != ForKind::kSerial ||
      epoch->ContainsLoopBreak())
    return std::nullopt;
  const For &loop = epoch->control;
  auto low = Constant(loop->min);
  auto count = Constant(loop->extent);
  auto step =
      Constant(loop->step.value_or(make_const(loop->loop_var.dtype(), 1)));
  if (!low || !count || !step || *step != 1 || *count <= 0 ||
      *low > std::numeric_limits<int64_t>::max() - *count)
    return std::nullopt;
  return std::pair<int64_t, int64_t>{*low, *low + *count};
}

inline bool Multiply(int64_t a, int64_t b, int64_t *out) {
  if (a < 0 || b < 0 || (b != 0 && a > std::numeric_limits<int64_t>::max() / b))
    return false;
  *out = a * b;
  return true;
}

inline std::optional<AscendCopy> TypedCopy(const CallNode *call) {
  if (!IsAscendCopyCall(call) || call->args.size() != 2)
    return std::nullopt;
  for (const PrimExpr &arg : call->args) {
    const auto *region_call = arg.as<CallNode>();
    if (!region_call || !region_call->op.same_as(tl::region()) ||
        region_call->args.size() < 2 ||
        !region_call->args[0].as<BufferLoadNode>())
      return std::nullopt;
  }
  return AscendCopy(call->args, call->annotations);
}

struct Occurrence {
  AscendCopy copy;
  std::vector<For> loops;
  Bindings bindings;
  PrimExpr guard;
};

// A semantic copy task may contain a finite loop nest. Do not accept opaque
// computation, mutable scalar state or a control path hidden inside it.
class CopyBody {
public:
  bool valid{true};
  std::vector<Occurrence> copies;

  void Walk(const Stmt &stmt, Bindings bindings, std::vector<For> loops,
            PrimExpr guard) {
    if (!valid)
      return;
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        if (const auto *bind = child.as<BindNode>()) {
          PrimExpr value = Expand(bind->value, bindings);
          if (!Immutable(value)) {
            valid = false;
            return;
          }
          bindings.Set(bind->var, value);
        } else {
          Walk(child, bindings, loops, guard);
        }
      }
    } else if (const auto *attribute = stmt.as<AttrStmtNode>()) {
      // Task and core wrappers are lexical scopes; their body remains typed.
      if (attribute->attr_key != "tl.ascend_task" &&
          attribute->attr_key != "tl.ascend_per_core_task" &&
          attribute->attr_key != "tl.schedule_unit") {
        valid = false;
        return;
      }
      Walk(attribute->body, bindings, loops, guard);
    } else if (const auto *loop = stmt.as<ForNode>()) {
      if (loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled) {
        valid = false;
        return;
      }
      loops.push_back(ffi::GetRef<For>(loop));
      Walk(loop->body, bindings, loops, guard);
    } else if (const auto *branch = stmt.as<IfThenElseNode>()) {
      PrimExpr condition = Expand(branch->condition, bindings);
      if (!Immutable(condition)) {
        valid = false;
        return;
      }
      Walk(branch->then_case, bindings, loops, guard && condition);
      if (branch->else_case.defined())
        Walk(branch->else_case.value(), bindings, loops, guard && !condition);
    } else if (const auto *block = stmt.as<SBlockNode>()) {
      if ((block->name_hint != "CUBE" && block->name_hint != "VECTOR" &&
           block->name_hint != "ascend_core_region") ||
          !block->iter_vars.empty() || !block->reads.empty() ||
          !block->writes.empty() || !block->alloc_buffers.empty() ||
          !block->match_buffers.empty() || block->init.defined()) {
        valid = false;
        return;
      }
      Walk(block->body, bindings, loops, guard);
    } else if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      if (!realize->iter_values.empty() ||
          !is_one(Expand(realize->predicate, bindings))) {
        valid = false;
        return;
      }
      Walk(realize->block, bindings, loops, guard);
    } else if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      const auto *call = evaluate->value.as<CallNode>();
      auto copy = call ? TypedCopy(call) : std::nullopt;
      if (!copy) {
        valid = is_zero(evaluate->value);
        return;
      }
      copies.push_back({copy.value(), std::move(loops), std::move(bindings),
                        std::move(guard)});
    } else {
      valid = false;
    }
  }
};

struct Endpoint {
  TaskNode *task;
  Occurrence occurrence;
};

// Outside the two typed copies, any pointer use, alias view, scalar access or
// opaque access to this storage invalidates the proof. Typed region metadata
// is traversed as arithmetic, rather than mistaken for a scalar BufferLoad.
class StorageAudit : public StmtExprVisitor {
public:
  explicit StorageAudit(Buffer canonical) : canonical_(std::move(canonical)) {}
  bool valid{true};

private:
  void VisitExpr_(const CallNode *call) final {
    if (auto parsed = TypedCopy(call)) {
      const AscendCopy &copy = parsed.value();
      for (const Buffer &buffer : {copy->src, copy->dst}) {
        if (buffer->data.same_as(canonical_->data) &&
            !buffer.same_as(canonical_))
          valid = false;
      }
      for (const auto &ranges : {copy->src_range, copy->dst_range}) {
        for (const Range &range : ranges) {
          VisitExpr(range->min);
          VisitExpr(range->extent);
        }
      }
      for (const auto &[key, value] : call->annotations) {
        if (value.as<PrimExprNode>())
          VisitExpr(Downcast<PrimExpr>(value));
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const BufferLoadNode *load) final {
    if (load->buffer->data.same_as(canonical_->data))
      valid = false;
    StmtExprVisitor::VisitExpr_(load);
  }
  void VisitStmt_(const BufferStoreNode *store) final {
    if (store->buffer->data.same_as(canonical_->data))
      valid = false;
    StmtExprVisitor::VisitStmt_(store);
  }
  void VisitExpr_(const VarNode *var) final {
    if (var == canonical_->data.get())
      valid = false;
  }
  void VisitStmt_(const AttrStmtNode *attribute) final {
    if (auto expr = attribute->node.try_cast<PrimExpr>())
      VisitExpr(expr.value());
    StmtExprVisitor::VisitStmt_(attribute);
  }
  void VisitStmt_(const SBlockNode *block) final {
    for (const auto &match : block->match_buffers) {
      if (match->buffer->data.same_as(canonical_->data) ||
          match->source->buffer->data.same_as(canonical_->data))
        valid = false;
    }
    for (const BufferRegion &region : block->reads)
      if (region->buffer->data.same_as(canonical_->data))
        valid = false;
    for (const BufferRegion &region : block->writes)
      if (region->buffer->data.same_as(canonical_->data))
        valid = false;
    if (auto value = block->annotations.Get("layout_map")) {
      for (const auto &[key, layout] :
           value.value().cast<ffi::Map<ObjectRef, Layout>>()) {
        if (const auto *buffer = key.as<BufferNode>();
            buffer && buffer->data.same_as(canonical_->data))
          valid = false;
        if (const auto *var = key.as<VarNode>(); var == canonical_->data.get())
          valid = false;
      }
    }
    StmtExprVisitor::VisitStmt_(block);
  }
  Buffer canonical_;
};

inline bool Injective(const Buffer &buffer, std::vector<int64_t> *shape) {
  if (!buffer->axis_separators.empty() || !buffer->dtype.is_scalar() ||
      !Immutable(buffer->elem_offset))
    return false;
  auto offset = Constant(buffer->elem_offset);
  if (!offset || *offset < 0)
    return false;
  for (const PrimExpr &dimension : buffer->shape) {
    auto size = Constant(dimension);
    if (!size || *size <= 0)
      return false;
    shape->push_back(*size);
  }
  if (buffer->strides.empty())
    return true;
  if (buffer->strides.size() != shape->size())
    return false;
  int64_t required = 1;
  for (size_t i = shape->size(); i > 0; --i) {
    auto stride = Constant(buffer->strides[i - 1]);
    if (!stride || *stride < required ||
        !Multiply(*stride, (*shape)[i - 1], &required))
      return false;
  }
  return true;
}

inline PrimExpr Guard(const Endpoint &endpoint) {
  PrimExpr guard = endpoint.occurrence.guard;
  for (IRStructure *node = endpoint.task; node; node = node->GetParent())
    guard = guard && node->GetConditionGuard();
  return Expand(guard, endpoint.occurrence.bindings);
}

inline std::vector<ControlNode *> Ancestors(TaskNode *task) {
  std::vector<ControlNode *> result;
  for (IRStructure *node = task->GetParent(); node; node = node->GetParent()) {
    if (!node->IsControl())
      return {};
    result.push_back(static_cast<ControlNode *>(node));
  }
  std::reverse(result.begin(), result.end());
  return result;
}

// Only a finite, exact interval is accepted. The bounds are not derived under
// the guard itself; that would make holes and mismatched endpoints vacuous.
inline bool ExactInterval(PrimExpr guard, const Var &epoch, int64_t low,
                          int64_t high, int64_t *first, int64_t *last) {
  arith::Analyzer analyzer;
  guard = analyzer.Simplify(guard);
  if (!Immutable(guard))
    return false;
  if (is_zero(guard)) {
    *first = *last = low;
    return true;
  }
  Array<PrimExpr> bounds = arith::DetectClipBound(guard, {epoch});
  if (bounds.size() != 2)
    return false;
  *first = low;
  *last = high;
  if (bounds[0].defined()) {
    auto cut = Constant(bounds[0]);
    if (!cut)
      return false;
    *first = std::max(low, std::min(high, *cut));
  }
  if (bounds[1].defined()) {
    auto cut = Constant(bounds[1]);
    if (!cut || *cut == std::numeric_limits<int64_t>::max())
      return false;
    *last = std::max(low, std::min(high, *cut + 1));
  }
  auto scope =
      analyzer.EnterConstraint(epoch >= make_const(epoch.dtype(), low) &&
                               epoch < make_const(epoch.dtype(), high));
  bool exact =
      analyzer.CanProve(guard == (epoch >= make_const(epoch.dtype(), *first) &&
                                  epoch < make_const(epoch.dtype(), *last)));
  scope();
  return exact;
}

// Preserve the placement selector's original guard proof, rather than using
// a stronger transfer guard conjoined with epoch bounds. Failure only leaves
// this endpoint's optional cut unavailable; it does not reject the transfer.
inline std::optional<PrimExpr> ActiveEpochs(const PrimExpr &guard,
                                         const Var &epoch, int64_t low,
                                         int64_t high) {
  int64_t first, last;
  if (!ExactInterval(guard, epoch, low, high, &first, &last) || first >= last)
    return std::nullopt;
  return epoch >= make_const(epoch.dtype(), first) &&
         epoch < make_const(epoch.dtype(), last);
}

using Rect = std::vector<std::pair<int64_t, int64_t>>;

inline bool Overlap(const Rect &a, const Rect &b) {
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].second <= b[i].first || b[i].second <= a[i].first)
      return false;
  return true;
}

inline bool Contained(const Rect &inner, const Rect &outer) {
  for (size_t i = 0; i < inner.size(); ++i)
    if (inner[i].first < outer[i].first || inner[i].second > outer[i].second)
      return false;
  return true;
}

inline std::optional<int64_t> Volume(const Rect &rect) {
  int64_t volume = 1;
  for (const auto &[low, high] : rect)
    if (!Multiply(volume, high - low, &volume))
      return std::nullopt;
  return volume;
}

// Subtract read rectangles from the bounding box. This proves union coverage
// even when read rectangles overlap, without enumerating elements or emitting
// unrolled IR. The same finite work budget bounds the residual partition.
inline bool Covers(const Rect &box, const std::vector<Rect> &reads) {
  std::vector<Rect> remaining{box};
  for (const Rect &read : reads) {
    if (!Contained(read, box))
      return false;
    std::vector<Rect> next;
    for (const Rect &part : remaining) {
      if (!Overlap(part, read)) {
        next.push_back(part);
        continue;
      }
      Rect middle = part;
      for (size_t axis = 0; axis < part.size(); ++axis) {
        int64_t low = std::max(part[axis].first, read[axis].first);
        int64_t high = std::min(part[axis].second, read[axis].second);
        if (middle[axis].first < low) {
          Rect piece = middle;
          piece[axis].second = low;
          next.push_back(std::move(piece));
          middle[axis].first = low;
        }
        if (high < middle[axis].second) {
          Rect piece = middle;
          piece[axis].first = high;
          next.push_back(std::move(piece));
          middle[axis].second = high;
        }
      }
      if (next.size() > kProofLimit)
        return false;
    }
    remaining = std::move(next);
  }
  return remaining.empty();
}

inline bool Enumerate(const std::vector<For> &loops, size_t index,
                      Bindings substitutions, const Region &region,
                      const Bindings &bindings, std::vector<Region> *result) {
  if (index == loops.size()) {
    if (result->size() >= kProofLimit)
      return false;
    Region expanded;
    for (const Range &range : region) {
      PrimExpr min = Expand(range->min, bindings);
      PrimExpr extent = Expand(range->extent, bindings);
      expanded.push_back(Range::FromMinExtent(Expand(min, substitutions),
                                              Expand(extent, substitutions)));
    }
    result->push_back(std::move(expanded));
    return true;
  }
  const For &loop = loops[index];
  auto min = Constant(Expand(Expand(loop->min, bindings), substitutions));
  auto extent = Constant(Expand(Expand(loop->extent, bindings), substitutions));
  auto step = Constant(
      Expand(Expand(loop->step.value_or(make_const(loop->loop_var.dtype(), 1)),
                    bindings),
             substitutions));
  if (!min || !extent || !step || *extent <= 0 || *step <= 0 ||
      *min > std::numeric_limits<int64_t>::max() - *extent)
    return false;
  int64_t trips = *extent / *step + (*extent % *step != 0);
  if (trips > static_cast<int64_t>(kProofLimit))
    return false;
  for (int64_t i = 0; i < trips; ++i) {
    substitutions.Set(loop->loop_var,
                      make_const(loop->loop_var.dtype(), *min + i * *step));
    if (!Enumerate(loops, index + 1, substitutions, region, bindings, result))
      return false;
  }
  return true;
}

inline bool Regions(const Endpoint &endpoint, ControlNode *epoch,
                    ControlNode *owner, const Region &region,
                    std::vector<Region> *result) {
  std::vector<For> loops;
  bool inside = false;
  for (ControlNode *control : Ancestors(endpoint.task)) {
    if (control == epoch) {
      inside = true;
      continue;
    }
    if (!inside || control == owner)
      continue;
    if (control->ContainsLoopBreak())
      return false;
    loops.push_back(control->control);
  }
  loops.insert(loops.end(), endpoint.occurrence.loops.begin(),
               endpoint.occurrence.loops.end());
  PrimExpr guard = Guard(endpoint);
  for (const For &loop : loops)
    if (Uses(guard, loop->loop_var))
      return false;
  return Enumerate(loops, 0, {}, region, endpoint.occurrence.bindings, result);
}

inline std::optional<ValidatedOwnerTransfer>
Prove(const Var &storage, int slots, const Endpoint &producer,
      const Endpoint &consumer, const Var &sid, bool vector_to_cube = false) {
  const auto &write = producer.occurrence.copy;
  const auto &read = consumer.occurrence.copy;
  Buffer canonical = write->dst;
  if (slots <= 1 || slots > static_cast<int>(kProofLimit) ||
      !read->src.same_as(canonical) ||
      write->dst_range.size() != canonical->shape.size() ||
      read->src_range.size() != canonical->shape.size())
    return std::nullopt;
  std::vector<int64_t> shape;
  if (!Injective(canonical, &shape))
    return std::nullopt;
  // Both directions transfer a matrix plane. Leading singleton axes are
  // shared keys; owner and ring storage cannot alias the matrix axes.
  if (shape.size() < 4)
    return std::nullopt;

  auto producers = Ancestors(producer.task);
  auto consumers = Ancestors(consumer.task);
  ControlNode *epoch = nullptr;
  for (size_t i = 0; i < std::min(producers.size(), consumers.size()); ++i) {
    if (producers[i] != consumers[i])
      break;
    epoch = producers[i];
  }
  auto bounds = FiniteEpochBounds(epoch);
  if (!bounds)
    return std::nullopt;
  const auto [epoch_min, epoch_end] = *bounds;
  Var iteration = epoch->control->loop_var;

  const auto &owner_controls = vector_to_cube ? consumers : producers;
  for (ControlNode *owner : owner_controls) {
    if (owner == epoch || !owner->IsWithin(epoch) ||
        owner->control->kind != ForKind::kSerial ||
        owner->ContainsLoopBreak() ||
        Constant(owner->control->min) != std::optional<int64_t>(0) ||
        Constant(owner->control->extent) != std::optional<int64_t>(2) ||
        Constant(owner->control->step.value_or(make_const(
            owner->control->loop_var.dtype(), 1))) != std::optional<int64_t>(1))
      continue;
    Var owner_var = owner->control->loop_var;
    if (owner_var.same_as(iteration) || sid.same_as(iteration))
      continue;
    PrimExpr source_guard = Guard(producer);
    PrimExpr destination_guard = Guard(consumer);
    if (Uses(source_guard, owner_var) || Uses(source_guard, sid) ||
        Uses(destination_guard, owner_var) || Uses(destination_guard, sid))
      continue;

    std::vector<Region> writes, reads;
    if (!Regions(producer, epoch, vector_to_cube ? nullptr : owner,
                 write->dst_range, &writes) ||
        !Regions(consumer, epoch, vector_to_cube ? owner : nullptr,
                 read->src_range, &reads) ||
        writes.empty() || reads.empty() ||
        2 * (writes.size() + reads.size()) > kProofLimit)
      continue;
    arith::Analyzer analyzer;
    for (const Constr &constraint : producer.task->outer_ctx.constrs_)
      if (constraint.kind == Constr::kBindRange)
        constraint.Populate(analyzer);
    analyzer.Bind(owner_var, Range::FromMinExtent(0, 2), true);
    analyzer.Bind(sid, Range::FromMinExtent(0, 2), true);

    int owner_axis = -1;
    Var write_owner = vector_to_cube ? sid : owner_var;
    Var read_owner = vector_to_cube ? owner_var : sid;
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      bool matches = shape[axis] == 2;
      for (const Region &rect : writes)
        matches &= analyzer.CanProveEqual(rect[axis]->extent, 1) &&
                   analyzer.CanProveEqual(rect[axis]->min, write_owner);
      for (const Region &rect : reads)
        matches &= analyzer.CanProveEqual(rect[axis]->extent, 1) &&
                   analyzer.CanProveEqual(rect[axis]->min, read_owner);
      if (matches)
        owner_axis = owner_axis == -1 ? static_cast<int>(axis) : -2;
    }
    if (owner_axis < 0 || owner_axis >= static_cast<int>(shape.size()) - 2)
      continue;

    // Owner inputs specialize only the same finite activation. Results retain
    // their established positive producer-to-consumer epoch distance.
    int first_distance = vector_to_cube ? 0 : 1;
    int end_distance = vector_to_cube ? 1 : slots;
    for (int distance = first_distance; distance < end_distance; ++distance) {
      Bindings shift{{iteration, iteration + distance}};
      PrimExpr epoch_bounds =
          iteration >= make_const(iteration.dtype(), epoch_min) &&
          iteration < make_const(iteration.dtype(), epoch_end);
      PrimExpr source_active = source_guard && epoch_bounds;
      PrimExpr destination_active = destination_guard && epoch_bounds;
      int64_t first, last, destination_first, destination_last;
      if (!ExactInterval(source_active, iteration, epoch_min, epoch_end, &first,
                         &last) ||
          !ExactInterval(destination_active, iteration, epoch_min, epoch_end,
                         &destination_first, &destination_last) ||
          destination_first - distance != first ||
          destination_last - distance != last || first >= last)
        continue;
      arith::Analyzer proof;
      for (const Constr &constraint : producer.task->outer_ctx.constrs_)
        if (constraint.kind == Constr::kBindRange &&
            !constraint.var.same_as(iteration))
          constraint.Populate(proof);
      proof.Bind(
          iteration,
          Range::FromMinExtent(make_const(iteration.dtype(), first),
                               make_const(iteration.dtype(), last - first)),
          true);
      proof.Bind(owner_var, Range::FromMinExtent(0, 2), true);
      proof.Bind(sid, Range::FromMinExtent(0, 2), true);
      int slot_axis = -1;
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (static_cast<int>(axis) == owner_axis || shape[axis] != slots)
          continue;
        PrimExpr p = writes[0][axis]->min;
        PrimExpr c = reads[0][axis]->min;
        bool matches = Uses(p, iteration) && Uses(c, iteration) &&
                       !Uses(p, write_owner) && !Uses(c, read_owner) &&
                       proof.CanProveEqual(p, Substitute(c, shift));
        // The other core's selector cannot define an input storage clock.
        // Keep the existing result admission unchanged.
        if (vector_to_cube)
          matches &= !Uses(p, read_owner) && !Uses(c, write_owner);
        for (const Region &rect : writes)
          matches &= proof.CanProveEqual(rect[axis]->extent, 1) &&
                     proof.CanProveEqual(rect[axis]->min, p);
        for (const Region &rect : reads)
          matches &= proof.CanProveEqual(rect[axis]->extent, 1) &&
                     proof.CanProveEqual(rect[axis]->min, c);
        if (matches)
          slot_axis = slot_axis == -1 ? static_cast<int>(axis) : -2;
      }
      if (slot_axis < 0 || slot_axis >= static_cast<int>(shape.size()) - 2)
        continue;

      // Manual version count alone does not establish when a physical slot
      // can be reused. The collective release proof below uses slots-distance,
      // so exclude every earlier same-slot write in the finite active domain.
      bool separated_slots = true;
      PrimExpr slot = writes[0][slot_axis]->min;
      for (int offset = 1; offset < slots && offset < last - first; ++offset) {
        arith::Analyzer reuse;
        for (const Constr &constraint : producer.task->outer_ctx.constrs_)
          if (constraint.kind == Constr::kBindRange &&
              !constraint.var.same_as(iteration))
            constraint.Populate(reuse);
        reuse.Bind(iteration,
                   Range::FromMinExtent(
                       make_const(iteration.dtype(), first),
                       make_const(iteration.dtype(), last - first - offset)),
                   true);
        PrimExpr next_slot =
            Substitute(slot, {{iteration, iteration + offset}});
        if (!reuse.CanProve(slot != next_slot)) {
          separated_slots = false;
          break;
        }
      }
      if (!separated_slots)
        continue;

      std::vector<int> payload_axes;
      bool valid = true;
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (static_cast<int>(axis) == owner_axis ||
            static_cast<int>(axis) == slot_axis)
          continue;
        PrimExpr key = writes[0][axis]->min;
        bool singleton_key =
            axis + 2 < shape.size() && !Uses(key, owner_var) && !Uses(key, sid);
        for (const Region &rect : writes)
          singleton_key &= proof.CanProveEqual(rect[axis]->extent, 1) &&
                           proof.CanProveEqual(rect[axis]->min, key);
        for (const Region &rect : reads)
          singleton_key &=
              proof.CanProveEqual(rect[axis]->extent, 1) &&
              proof.CanProveEqual(Substitute(rect[axis]->min, shift), key);
        if (!singleton_key)
          payload_axes.push_back(static_cast<int>(axis));
      }
      auto rectangles = [&](const std::vector<Region> &regions, bool consumer,
                            std::vector<Rect> *rects) {
        for (const Region &region : regions) {
          Rect rect;
          for (size_t axis = 0; axis < shape.size(); ++axis) {
            PrimExpr min = consumer ? Substitute(region[axis]->min, shift)
                                    : region[axis]->min;
            PrimExpr extent = region[axis]->extent;
            if (!Immutable(min) || !Immutable(extent) ||
                !CanProveAscendRangeInBounds(min, extent,
                                             canonical->shape[axis], &proof))
              return false;
          }
          for (int axis : payload_axes) {
            auto min = Constant(region[axis]->min);
            auto extent = Constant(region[axis]->extent);
            if (!min || !extent || *extent <= 0 || *min < 0 ||
                *min > shape[axis] - *extent)
              return false;
            rect.emplace_back(*min, *min + *extent);
          }
          rects->push_back(std::move(rect));
        }
        return true;
      };
      std::vector<Rect> source_rects, destination_rects;
      valid &= rectangles(writes, false, &source_rects);
      valid &= rectangles(reads, true, &destination_rects);
      if (!valid)
        continue;
      Rect box = source_rects[0];
      int64_t volume = 0;
      for (size_t i = 0; i < source_rects.size(); ++i) {
        const Rect &rect = source_rects[i];
        auto size = Volume(rect);
        if (!size || volume > std::numeric_limits<int64_t>::max() - *size) {
          valid = false;
          break;
        }
        volume += *size;
        for (size_t axis = 0; axis < box.size(); ++axis) {
          box[axis].first = std::min(box[axis].first, rect[axis].first);
          box[axis].second = std::max(box[axis].second, rect[axis].second);
        }
        for (size_t j = 0; j < i; ++j)
          valid &= !Overlap(rect, source_rects[j]);
      }
      if (!valid || Volume(box) != std::optional<int64_t>(volume) ||
          !Covers(box, destination_rects))
        continue;
      return ValidatedOwnerTransfer{
          storage,
          producer.task,
          consumer.task,
          epoch,
          owner,
          sid,
          distance,
          slots,
          writes[0][slot_axis]->min,
          reads[0][slot_axis]->min,
          source_guard,
          destination_guard,
          vector_to_cube ? ValidatedOwnerTransfer::Direction::kInput
                         : ValidatedOwnerTransfer::Direction::kResult,
          ActiveEpochs(source_guard, iteration, epoch_min, epoch_end),
          ActiveEpochs(destination_guard, iteration, epoch_min, epoch_end)};
    }
  }
  return std::nullopt;
}

inline std::vector<ValidatedOwnerTransfer>
AnalyzeOwnerTransfers(const std::vector<std::shared_ptr<IRStructure>> &root,
                      const SBlock &kernel_root, const BufferVersionMap &manual,
                      const std::vector<Var> &lanes, bool vector_to_cube) {
  std::vector<ValidatedOwnerTransfer> plans;
  if (lanes.size() != 1 || !kernel_root.defined())
    return plans;
  Array<Var> private_members = ValidatedAivPrivateData(kernel_root.get());
  if (private_members.empty())
    return plans;
  std::vector<TaskNode *> tasks;
  CollectAllTaskNodes(root, tasks);
  for (const auto &[storage, slots] : manual) {
    std::vector<TaskNode *> writers, readers;
    for (TaskNode *task : tasks) {
      if (task->WritesStorage(storage))
        writers.push_back(task);
      if (task->ReadsStorage(storage))
        readers.push_back(task);
    }
    if (writers.size() != 1 || readers.size() != 1 ||
        writers[0] == readers[0] ||
        !writers[0]->UsesOnlyPipe(vector_to_cube ? ResourcePipe::kMTE3
                                                 : ResourcePipe::kFixpipe) ||
        !readers[0]->UsesOnlyPipe(ResourcePipe::kMTE2) ||
        writers[0]->GetCoreMask() !=
            (vector_to_cube ? kCoreVector : kCoreCube) ||
        readers[0]->GetCoreMask() != (vector_to_cube ? kCoreCube : kCoreVector))
      continue;
    CopyBody source, destination;
    source.Walk(writers[0]->stmt, TaskBindings(writers[0]), {}, Bool(true));
    destination.Walk(readers[0]->stmt, TaskBindings(readers[0]), {},
                     Bool(true));
    if (!source.valid || !destination.valid || source.copies.size() != 1 ||
        destination.copies.size() != 1)
      continue;
    const AscendCopy &write = source.copies[0].copy;
    const AscendCopy &read = destination.copies[0].copy;
    if (!IsGlobalBuffer(write->dst) || !write->dst->data.same_as(storage) ||
        !IsGlobalBuffer(read->src) || !read->src->data.same_as(storage))
      continue;
    if (vector_to_cube) {
      // This proof specializes ordinary private UB exports consumed by L1
      // input copies. Hints carrying another hardware route or completion
      // contract are not implicitly granted owner-input semantics.
      if (!IsSharedBuffer(write->src) ||
          !ContainsAivPrivateData(private_members, write->src->data) ||
          !IsL1Buffer(read->dst))
        continue;
      bool ordinary = true;
      for (const AscendCopy &copy : {write, read}) {
        ordinary &= copy->data_select == 0 && copy->transpose == 0 &&
                    copy->nd2nz == 0 && copy->dual_dst_ctl == 0 &&
                    !copy->mx_sf_data.defined() && !copy->root_fixpipe_config &&
                    !copy->pad_value.defined() &&
                    is_zero(copy->unit_flag_ctl) && is_zero(copy->sub_blockid);
      }
      if (!ordinary)
        continue;
    } else if (!IsL0CBuffer(write->src) || !IsSharedBuffer(read->dst) ||
               !ContainsAivPrivateData(private_members, read->dst->data) ||
               read->data_select != 0 || read->transpose != 0 ||
               read->nd2nz != 0 || write->dual_dst_ctl != 0) {
      continue;
    }
    StorageAudit audit(write->dst);
    audit(kernel_root->body);
    if (!audit.valid)
      continue;
    Endpoint producer{writers[0], std::move(source.copies[0])};
    Endpoint consumer{readers[0], std::move(destination.copies[0])};
    if (auto plan =
            Prove(storage, slots, producer, consumer, lanes[0], vector_to_cube))
      plans.push_back(std::move(plan.value()));
  }
  return plans;
}

} // namespace owner_result_detail

// Called only by the 2201 path after MixedKernel(sids=2) and private-UB
// validation. Any unsupported shape/control/storage use keeps the existing
// collective protocol: this analysis never makes a malformed proof fatal.
inline std::vector<ValidatedOwnerTransfer>
AnalyzeOwnerResults(const std::vector<std::shared_ptr<IRStructure>> &root,
                    const SBlock &kernel_root, const BufferVersionMap &manual,
                    const std::vector<Var> &lanes) {
  return owner_result_detail::AnalyzeOwnerTransfers(root, kernel_root, manual,
                                                    lanes, false);
}

// An input's SID-selected private UB writes and owner-traversed AIC reads have
// identical finite activations. No event is synthesized here. The caller must
// retain the ordinary collective read-completion/reuse relation and prove a
// once-per-activation producer-BEGIN cut after that release before adding the
// complementary empty participant of an owner-indexed mode2 publication.
inline std::vector<ValidatedOwnerTransfer>
AnalyzeOwnerInputs(const std::vector<std::shared_ptr<IRStructure>> &root,
                   const SBlock &kernel_root, const BufferVersionMap &manual,
                   const std::vector<Var> &lanes) {
  return owner_result_detail::AnalyzeOwnerTransfers(root, kernel_root, manual,
                                                    lanes, true);
}

} // namespace tvm::tl::ascend
