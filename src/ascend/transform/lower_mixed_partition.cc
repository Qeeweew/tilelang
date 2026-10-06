// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_mixed_partition.cc
 * \brief Establish private two-AIV row ownership before layout inference.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "auto_schedule/kernel_rewriter.h"
#include "layout/layout.h"
#include "mixed_kernel_intent.h"
#include "op/builtin.h"
#include "op/fill.h"
#include "op/reduce.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::GetRef;
using ffi::Optional;

namespace {

// This is a stage contract, not a second encoding of ownership. Buffer shapes,
// local indices, regions, and the enclosing cthread are the durable facts used
// by layout, task analysis, storage planning, synchronization, and codegen.
constexpr const char *kRowPartitioned = "tl.ascend_row_partitioned";

void Require(bool condition, const char *message) {
  if (!condition)
    TVM_FFI_THROW(ValueError) << "dav-2201 mixed row partition: " << message;
}

bool UsesVar(const PrimExpr &expr, const Var &var) {
  bool found = false;
  PostOrderVisit(expr,
                 [&](const ObjectRef &node) { found |= node.same_as(var); });
  return found;
}

struct ElementAccess {
  Buffer buffer;
  Array<PrimExpr> indices;
  bool write;
};

class ElementAccessCollector : public StmtExprVisitor {
public:
  std::vector<ElementAccess> accesses;

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    accesses.push_back({op->buffer, op->indices, true});
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    accesses.push_back({op->buffer, op->indices, false});
    // Indices describe an access; nested gathers are deliberately outside the
    // dense affine contract and are checked once by RowAxis below.
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::region()))
      return;
    StmtExprVisitor::VisitExpr_(op);
  }
};

struct ParallelNest {
  For outer;
  int depth;
  std::vector<ElementAccess> accesses;
};

class KernelIntentCollector : public StmtExprVisitor {
public:
  std::vector<ParallelNest> nests;
  std::vector<Call> copies, reductions;

private:
  void VisitStmt_(const ForNode *op) final {
    bool parallel = op->kind == ForKind::kParallel;
    if (parallel && parallel_depth_ == 0) {
      int depth = 1;
      const ForNode *inner = op;
      while (const auto *next = inner->body.as<ForNode>()) {
        if (next->kind != ForKind::kParallel)
          break;
        ++depth;
        inner = next;
      }
      ElementAccessCollector collector;
      collector(op->body);
      nests.push_back({GetRef<For>(op), depth, std::move(collector.accesses)});
    }
    parallel_depth_ += parallel;
    StmtExprVisitor::VisitStmt_(op);
    parallel_depth_ -= parallel;
  }
  void VisitExpr_(const CallNode *op) final {
    if (IsAscendCopyCall(op))
      copies.push_back(GetRef<Call>(op));
    else if (op->op.same_as(ReduceOp::Get()))
      reductions.push_back(GetRef<Call>(op));
    StmtExprVisitor::VisitExpr_(op);
  }
  int parallel_depth_{0};
};

// Temporary connected-component analysis over existing access relations. It
// disappears after the rewrite and does not introduce another IR or scheduler.
struct OwnedStorage {
  Buffer allocation;
  int axis;
  int64_t rows;
  int parent;
  Optional<PrimExpr> valid_rows;
};

class RowOwnership {
public:
  RowOwnership(const SBlock &root, const ConstrSet &outer_ctx) {
    outer_ctx.Populate(analyzer_);
    mixed_ = ascend::HasAscendCubeIntent(root->body);
    for (const Buffer &buffer : root->alloc_buffers) {
      if (IsSharedBuffer(buffer))
        allocations_[buffer->data.get()] = buffer;
    }
    KernelIntentCollector collector;
    collector(root->body);
    intent_ = std::move(collector);
  }

  bool IsMixed() const { return mixed_; }
  bool Empty() const { return owned_.empty(); }

  void Analyze() {
    // Seed by an actual affine row coordinate of a multidimensional Parallel
    // write. A shape match alone never establishes ownership.
    for (const auto &nest : intent_.nests) {
      if (nest.depth < 2)
        continue;
      for (const auto &access : nest.accesses) {
        if (!access.write || !IsSharedBuffer(access.buffer))
          continue;
        int axis = RowAxis(access, nest.outer->loop_var);
        if (axis >= 0)
          Mark(access.buffer, axis);
      }
    }

    // Propagate the same coordinate through elementwise reads/writes, copies,
    // and trailing-axis reductions. Every successful iteration adds storage;
    // there are at most as many iterations as referenced private allocations.
    bool changed;
    do {
      changed = false;
      for (const auto &nest : intent_.nests)
        changed |= ConnectNest(nest);
      for (const Call &call : intent_.copies)
        changed |= ConnectCopy(AscendCopy(call->args, call->annotations));
      for (const Call &call : intent_.reductions)
        changed |= ConnectReduce(ReduceOp(call->args, call->annotations));
    } while (changed);

    Require(!owned_.empty() || intent_.nests.empty(),
            "no shared affine row coordinate was found for the mixed vector "
            "component; express a dense row/column Parallel nest");
    for (const auto &nest : intent_.nests)
      ConnectNest(nest);
    // GM boundaries establish the component's actual valid rows. Validate
    // internal UB copies only after all those boundaries, independently of
    // statement order: a row-state cache update may precede the first load.
    std::vector<AscendCopy> owned_copies;
    for (const Call &call : intent_.copies) {
      AscendCopy copy(call->args, call->annotations);
      if (Find(copy->src) >= 0 && Find(copy->dst) >= 0)
        owned_copies.push_back(copy);
      else
        EstablishCopyDomain(copy);
    }
    for (const AscendCopy &copy : owned_copies)
      EstablishCopyDomain(copy);
    for (const Call &call : intent_.reductions) {
      ReduceOp reduce(call->args, call->annotations);
      int id = Find(reduce->src);
      if (id < 0)
        continue;
      int axis = owned_[id].axis;
      const auto &component = Component(id);
      PrimExpr extent = reduce->srcRegion_->region[axis]->extent;
      Require(
          analyzer_.CanProveEqual(extent,
                                  reduce->dstRegion_->region[axis]->extent) &&
              (analyzer_.CanProveEqual(extent, Integer(component.rows)) ||
               (component.valid_rows.has_value() &&
                analyzer_.CanProveEqual(extent, component.valid_rows.value()))),
          "owned reductions must use the full logical or established "
          "valid row domain on both operands");
    }
  }

  int Find(const Buffer &buffer) const {
    auto it = ids_.find(buffer->data.get());
    return it == ids_.end() ? -1 : it->second;
  }
  bool OwnsData(const Var &data) const { return ids_.count(data.get()) != 0; }
  const OwnedStorage &Storage(int id) const { return owned_[id]; }
  const OwnedStorage &Component(int id) { return owned_[Leader(id)]; }
  int NestOwner(const ForNode *loop) const {
    auto it = nest_owners_.find(loop);
    return it == nest_owners_.end() ? -1 : it->second;
  }
  int RowAxis(const ElementAccess &access, const Var &row) {
    int result = -1;
    for (size_t axis = 0; axis < access.indices.size(); ++axis) {
      if (!UsesVar(access.indices[axis], row))
        continue;
      Require(result == -1 &&
                  analyzer_.CanProveEqual(access.indices[axis], row),
              "row accesses must use one shared affine coordinate without "
              "gathers, shifts, or cross-row reads");
      result = static_cast<int>(axis);
    }
    return result;
  }

  int GMRowAxis(const AscendCopy &copy, bool ub_source, int axis) {
    const auto &ub = ub_source ? copy->src_range : copy->dst_range;
    const auto &gm = ub_source ? copy->dst_range : copy->src_range;
    Require(gm.size() >= ub.size(),
            "GM/UB row copies must preserve the trailing coordinate axes");
    size_t leading = gm.size() - ub.size();
    for (size_t i = 0; i < leading; ++i)
      Require(analyzer_.CanProve(gm[i]->extent >= 0) &&
                  analyzer_.CanProve(gm[i]->extent <= 1),
              "extra leading GM region axes must have singleton 0/1 validity");
    for (size_t i = 0; i < ub.size(); ++i)
      Require(analyzer_.CanProveEqual(ub[i]->extent, gm[leading + i]->extent),
              "GM/UB copies must carry the same valid region on both sides");
    return static_cast<int>(leading) + axis;
  }

private:
  bool Mark(const Buffer &buffer, int axis) {
    Require(IsSharedBuffer(buffer), "owned rows require private UB storage");
    Require(axis == 0,
            "the initial native row partition requires the leading UB axis; "
            "transpose or gather ownership needs a separate layout contract");
    auto owner = allocations_.find(buffer->data.get());
    Require(owner != allocations_.end(),
            "owned UB storage must have an allocation in its kernel root");
    const Buffer &allocation = owner->second;
    Require(
        buffer->shape.size() == allocation->shape.size() &&
            buffer->dtype == allocation->dtype,
        "owned UB reinterpretation or rank-changing aliases are unsupported");
    Require(buffer->strides.empty() && allocation->strides.empty() &&
                buffer->axis_separators.empty() &&
                analyzer_.CanProveEqual(buffer->elem_offset, Integer(0)),
            "owned UB must use compact pre-layout storage with zero offset");
    for (size_t i = 0; i < buffer->shape.size(); ++i)
      Require(analyzer_.CanProveEqual(buffer->shape[i], allocation->shape[i]),
              "owned UB aliases must preserve the allocation coordinates");
    const auto *rows = allocation->shape[axis].as<IntImmNode>();
    Require(rows && rows->value > 0,
            "logical row capacity must be a positive compile-time constant");
    int existing = Find(buffer);
    if (existing >= 0) {
      Require(owned_[existing].axis == axis,
              "one UB allocation has conflicting row ownership axes");
      return false;
    }
    int id = static_cast<int>(owned_.size());
    ids_[buffer->data.get()] = id;
    owned_.push_back({allocation, axis, rows->value, id, std::nullopt});
    return true;
  }

  int Leader(int id) {
    if (owned_[id].parent != id)
      owned_[id].parent = Leader(owned_[id].parent);
    return owned_[id].parent;
  }
  void Unite(int a, int b) {
    a = Leader(a);
    b = Leader(b);
    Require(owned_[a].rows == owned_[b].rows,
            "connected row operations must share one logical row capacity");
    if (a != b)
      owned_[b].parent = a;
  }

  bool ConnectNest(const ParallelNest &nest) {
    int owner = -1;
    for (const auto &access : nest.accesses) {
      if (!IsSharedBuffer(access.buffer))
        continue;
      int id = Find(access.buffer);
      if (id < 0)
        continue;
      int axis = RowAxis(access, nest.outer->loop_var);
      Require(axis == owned_[id].axis,
              "an owned buffer must use the shared row loop coordinate; "
              "cross-row or scalar accesses inside Parallel are unsupported");
      if (owner >= 0)
        Unite(owner, id);
      owner = id;
    }
    if (owner < 0)
      return false;
    Require(analyzer_.CanProveEqual(nest.outer->min, Integer(0)) &&
                analyzer_.CanProveEqual(nest.outer->extent,
                                        Integer(owned_[owner].rows)),
            "the row Parallel domain must cover the logical tile from zero");
    bool changed = false;
    for (const auto &access : nest.accesses) {
      if (!IsSharedBuffer(access.buffer))
        continue;
      int axis = RowAxis(access, nest.outer->loop_var);
      if (axis < 0)
        continue; // Column-only broadcast, replicated in private UB.
      changed |= Mark(access.buffer, axis);
      Unite(owner, Find(access.buffer));
    }
    nest_owners_[nest.outer.get()] = owner;
    return changed;
  }

  bool ConnectCopy(const AscendCopy &copy) {
    int source = Find(copy->src), destination = Find(copy->dst);
    bool changed = false;
    if (IsSharedBuffer(copy->src) && IsSharedBuffer(copy->dst) &&
        (source >= 0 || destination >= 0)) {
      Require(copy->src_range.size() == copy->dst_range.size(),
              "UB row copies must preserve rank and coordinate axes");
      int axis = source >= 0 ? owned_[source].axis : owned_[destination].axis;
      Require(analyzer_.CanProveEqual(copy->src_range[axis]->extent,
                                      copy->dst_range[axis]->extent),
              "UB row copy extents must match");
      changed |= Mark(copy->src, axis);
      changed |= Mark(copy->dst, axis);
      Unite(Find(copy->src), Find(copy->dst));
    }
    return changed;
  }

  bool ConnectReduce(const ReduceOp &reduce) {
    int source = Find(reduce->src), destination = Find(reduce->dst);
    if (source < 0 && destination < 0)
      return false;
    int axis = source >= 0 ? owned_[source].axis : owned_[destination].axis;
    Require(reduce->dim == static_cast<int>(reduce->src->shape.size()) - 1 &&
                reduce->dim != axis,
            "reductions must preserve owned rows and reduce the trailing "
            "column axis; cross-AIV reductions require a collective plan");
    Require(reduce->dst->shape.size() + 1 == reduce->src->shape.size() ||
                reduce->dst->shape.size() == reduce->src->shape.size(),
            "reduction outputs must remove the trailing axis or retain it");
    bool changed = Mark(reduce->src, axis);
    changed |= Mark(reduce->dst, axis);
    Unite(Find(reduce->src), Find(reduce->dst));
    return changed;
  }

  void EstablishCopyDomain(const AscendCopy &copy) {
    int source = Find(copy->src), destination = Find(copy->dst);
    if (source < 0 && destination < 0) {
      Require(!(IsSharedBuffer(copy->src) && IsGlobalBuffer(copy->dst)),
              "a mixed UB export has no established row ownership; a "
              "replicated column broadcast must not write the same GM tile");
      return;
    }
    Require(copy->dual_dst_ctl == 0 &&
                !copy->annotations.Get("double").has_value(),
            "ordinary mixed row ownership cannot be combined with dual_copy");
    bool ub_source = source >= 0;
    int id = ub_source ? source : destination;
    int axis = owned_[id].axis;
    const auto &ub_ranges = ub_source ? copy->src_range : copy->dst_range;
    Require(analyzer_.CanProveEqual(ub_ranges[axis]->min, Integer(0)),
            "owned copy regions must start at logical row zero");
    if (source >= 0 && destination >= 0) {
      Require(analyzer_.CanProveEqual(copy->dst_range[axis]->min, Integer(0)),
              "owned UB copy destinations must start at row zero");
      const auto &component = Component(id);
      PrimExpr extent = ub_ranges[axis]->extent;
      Require(
          analyzer_.CanProveEqual(extent, Integer(component.rows)) ||
              (component.valid_rows.has_value() &&
               analyzer_.CanProveEqual(extent, component.valid_rows.value())),
          "owned UB copies must use the full logical or established valid "
          "row domain on both operands");
      return;
    }
    const Buffer &other = ub_source ? copy->dst : copy->src;
    Require(IsGlobalBuffer(other),
            "logical L0C/UB and UB/L1 handovers must be expanded into GM "
            "copies before row partitioning");
    int gm_axis = GMRowAxis(copy, ub_source, axis);
    const auto &gm_ranges = ub_source ? copy->dst_range : copy->src_range;
    PrimExpr valid = ub_ranges[axis]->extent;
    Require(!UsesAnyRowLoop(valid),
            "whole-tile copies must remain outside the owner row traversal");
    // A frontend tile can extend past the GM tail. The valid region, rather
    // than workspace capacity, is authoritative. OOB lowering later handles
    // column padding; this pass establishes only the shared row domain.
    valid = analyzer_.Simplify(
        max(make_zero(valid.dtype()),
            min(min(valid, make_const(valid.dtype(), owned_[id].rows)),
                other->shape[gm_axis] - gm_ranges[gm_axis]->min)));
    size_t leading = gm_ranges.size() - ub_ranges.size();
    for (size_t i = 0; i < leading; ++i)
      valid =
          analyzer_.Simplify(valid * cast(valid.dtype(), gm_ranges[i]->extent));
    OwnedStorage &component = owned_[Leader(id)];
    if (component.valid_rows.has_value())
      Require(analyzer_.CanProveEqual(component.valid_rows.value(), valid),
              "connected GM copies disagree on the actual valid row domain; "
              "pass the same valid region through every handover");
    else
      component.valid_rows = valid;
  }

  bool UsesAnyRowLoop(const PrimExpr &expr) const {
    for (const auto &nest : intent_.nests)
      if (UsesVar(expr, nest.outer->loop_var))
        return true;
    return false;
  }

  arith::Analyzer analyzer_;
  bool mixed_{false};
  KernelIntentCollector intent_;
  std::unordered_map<const VarNode *, Buffer> allocations_;
  std::unordered_map<const VarNode *, int> ids_;
  std::unordered_map<const ForNode *, int> nest_owners_;
  std::vector<OwnedStorage> owned_;
};

class OwnedKernelRewriter : public StmtExprMutator {
public:
  OwnedKernelRewriter(RowOwnership *ownership, Var sid,
                      const ConstrSet &outer_ctx)
      : ownership_(ownership), sid_(std::move(sid)) {
    analyzer_.Bind(sid_, Range(0, 2));
    outer_ctx.ToConstraints().Populate(analyzer_);
  }
  SBlock Rewrite(const SBlock &root) {
    SBlock block = Downcast<SBlock>(VisitStmt(root));
    auto *writer = block.CopyOnWrite();
    writer->annotations.Set(kRowPartitioned, Integer(2));
    writer->annotations.Set("vector_count", Integer(2));
    return block;
  }

private:
  PrimExpr Capacity(int id, DataType dtype) {
    return make_const(dtype, (ownership_->Storage(id).rows + 1) / 2);
  }
  PrimExpr Origin(int id, DataType dtype) {
    return cast(dtype, sid_) * Capacity(id, dtype);
  }
  PrimExpr OwnerRows(int id, DataType dtype) {
    const auto &component = ownership_->Component(id);
    PrimExpr valid =
        component.valid_rows.value_or(make_const(dtype, component.rows));
    return analyzer_.Simplify(
        max(make_zero(dtype),
            min(Capacity(id, dtype), cast(dtype, valid) - Origin(id, dtype))));
  }
  Buffer Remap(const Buffer &buffer) {
    int id = ownership_->Find(buffer);
    if (id < 0)
      return buffer;
    auto existing = buffers_.Get(buffer);
    if (existing.has_value())
      return existing.value();
    Buffer result = buffer;
    auto *writer = result.CopyOnWrite();
    int axis = ownership_->Storage(id).axis;
    writer->shape.Set(axis, Capacity(id, buffer->shape[axis].dtype()));
    buffers_.Set(buffer, result);
    return result;
  }

  Array<PrimExpr> LocalIndices(const Buffer &buffer,
                               const Array<PrimExpr> &indices) {
    Array<PrimExpr> result;
    for (const auto &index : indices)
      result.push_back(VisitExpr(index));
    int id = ownership_->Find(buffer);
    if (id < 0)
      return result;
    int axis = ownership_->Storage(id).axis;
    Require(active_row_.has_value() &&
                analyzer_.CanProveEqual(indices[axis], active_row_.value()),
            "owned element accesses must be inside their shared row Parallel; "
            "use region-based copy/reduce/fill for whole-tile access");
    result.Set(axis, analyzer_.Simplify(result[axis] -
                                        Origin(id, result[axis].dtype())));
    return result;
  }

  Call Region(const PrimExpr &argument, int id, bool global,
              int global_axis = -1, bool capacity = false) {
    AccessRegion access = NormalizeToAccessRegion(argument, kAccessReadWrite);
    auto ranges = access.region->region;
    Buffer buffer = access.region->buffer;
    int axis = global ? global_axis : ownership_->Storage(id).axis;
    Array<PrimExpr> indices, args;
    for (size_t i = 0; i < ranges.size(); ++i) {
      PrimExpr origin = VisitExpr(ranges[i]->min);
      if (static_cast<int>(i) == axis) {
        if (global) {
          const auto &component = ownership_->Component(id);
          PrimExpr valid = component.valid_rows.value_or(
              make_const(origin.dtype(), component.rows));
          // Empty owners address the end of the valid slice, not an
          // out-of-bounds logical origin. Their zero extent still reaches the
          // semantic whole-copy task and its collective completion.
          origin = min(origin + min(Origin(id, origin.dtype()),
                                    cast(origin.dtype(), valid)),
                       buffer->shape[i]);
        } else {
          Require(analyzer_.CanProveEqual(ranges[i]->min, Integer(0)),
                  "owned tile operations must start at logical row zero");
          origin = make_zero(origin.dtype());
        }
      }
      indices.push_back(analyzer_.Simplify(origin));
    }
    args = {BufferLoad(Remap(buffer), indices), Integer(access.access_mask)};
    for (size_t i = 0; i < ranges.size(); ++i)
      args.push_back(static_cast<int>(i) == axis
                         ? (capacity ? Capacity(id, ranges[i]->extent.dtype())
                                     : OwnerRows(id, ranges[i]->extent.dtype()))
                         : VisitExpr(ranges[i]->extent));
    return Call(DataType::Handle(), tl::region(), args);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (active_row_.has_value() && active_row_.value().same_as(GetRef<Var>(op)))
      return GetRef<Var>(op) + Origin(active_owner_, op->dtype);
    return GetRef<PrimExpr>(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Optional<PrimExpr> predicate;
    if (op->predicate.has_value())
      predicate = VisitExpr(op->predicate.value());
    return BufferLoad(Remap(op->buffer), LocalIndices(op->buffer, op->indices),
                      predicate, op->span);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Optional<PrimExpr> predicate;
    if (op->predicate.has_value())
      predicate = VisitExpr(op->predicate.value());
    return BufferStore(Remap(op->buffer), VisitExpr(op->value),
                       LocalIndices(op->buffer, op->indices), predicate,
                       op->span);
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (IsAscendCopyCall(op)) {
      AscendCopy copy(op->args, op->annotations);
      int source = ownership_->Find(copy->src);
      int destination = ownership_->Find(copy->dst);
      if (source < 0 && destination < 0)
        return StmtExprMutator::VisitExpr_(op);
      Require(!active_row_.has_value(),
              "whole owned copies must stay outside the row traversal so "
              "empty AIV partitions still participate in completion");
      Array<PrimExpr> args = op->args;
      if (source >= 0 && destination >= 0) {
        args.Set(0, Region(op->args[0], source, false));
        args.Set(1, Region(op->args[1], destination, false));
      } else {
        bool ub_source = source >= 0;
        int id = ub_source ? source : destination;
        int axis = ownership_->GMRowAxis(copy, ub_source,
                                         ownership_->Storage(id).axis);
        args.Set(ub_source ? 0 : 1,
                 Region(op->args[ub_source ? 0 : 1], id, false));
        args.Set(ub_source ? 1 : 0,
                 Region(op->args[ub_source ? 1 : 0], id, true, axis));
      }
      return Call(op->dtype, op->op, args, op->annotations, op->span);
    }
    if (op->op.same_as(ReduceOp::Get())) {
      ReduceOp reduce(op->args, op->annotations);
      int source = ownership_->Find(reduce->src);
      int destination = ownership_->Find(reduce->dst);
      if (source < 0 && destination < 0)
        return StmtExprMutator::VisitExpr_(op);
      Require(!active_row_.has_value() && source >= 0 && destination >= 0,
              "owned reductions must preserve a complete row-domain relation");
      Array<PrimExpr> args = op->args;
      args.Set(0, Region(op->args[0], source, false));
      args.Set(1, Region(op->args[1], destination, false));
      return Call(op->dtype, op->op, args, op->annotations, op->span);
    }
    if (op->op.same_as(Fill::Get())) {
      Fill fill(op->args, op->annotations);
      int id = ownership_->Find(fill->dst);
      if (id >= 0) {
        Require(!active_row_.has_value(),
                "owned fill must initialize private tile storage outside rows");
        int axis = ownership_->Storage(id).axis;
        Require(analyzer_.CanProveEqual(fill->region[axis]->extent,
                                        Integer(ownership_->Storage(id).rows)),
                "partial owned fills require an explicit row-domain plan");
        // Private padding may be initialized on both lanes; only the actual
        // owner rows are computed/exported. No shared GM data is duplicated.
        Array<PrimExpr> args = op->args;
        args.Set(0, Region(op->args[0], id, false, -1, true));
        args.Set(1, VisitExpr(op->args[1]));
        return Call(op->dtype, op->op, args, op->annotations, op->span);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    int id = ownership_->NestOwner(op);
    if (id < 0)
      return StmtExprMutator::VisitStmt_(op);
    Require(!active_row_.has_value(),
            "nested owner-row components require an explicit row-domain plan");
    active_row_ = op->loop_var;
    active_owner_ = id;
    Stmt body = VisitStmt(op->body);
    active_row_ = std::nullopt;
    active_owner_ = -1;
    // Keep a 1D output row vector as a vector operation with a bounded dynamic
    // count. A multidimensional tile traverses rows but retains dense inner
    // Parallel instructions; there is no scalar elementwise fallback.
    For result = GetRef<For>(op);
    auto *writer = result.CopyOnWrite();
    writer->extent = OwnerRows(id, op->extent.dtype());
    if (op->body.as<ForNode>())
      writer->kind = ForKind::kSerial;
    writer->body = std::move(body);
    return result;
  }

  BufferRegion MetadataRegion(const BufferRegion &region) {
    int id = ownership_->Find(region->buffer);
    if (id < 0)
      return region;
    int axis = ownership_->Storage(id).axis;
    Array<Range> ranges;
    for (size_t i = 0; i < region->region.size(); ++i) {
      const Range &range = region->region[i];
      PrimExpr origin = VisitExpr(range->min);
      PrimExpr extent = VisitExpr(range->extent);
      if (static_cast<int>(i) == axis) {
        if (active_row_.has_value() && UsesVar(range->min, active_row_.value()))
          origin = analyzer_.Simplify(origin - Origin(id, origin.dtype()));
        else {
          Require(
              analyzer_.CanProveEqual(range->min, Integer(0)),
              "owned block regions must use the established row coordinate");
          if (analyzer_.CanProveEqual(range->extent,
                                      Integer(ownership_->Storage(id).rows)))
            extent = Capacity(id, extent.dtype());
        }
      }
      ranges.push_back(Range::FromMinExtent(origin, extent));
    }
    return BufferRegion(Remap(region->buffer), ranges);
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (auto annotation = op->annotations.Get(attr::kLayoutMap)) {
      // Before layout inference, frontend annotations are keyed by storage
      // Vars. Native Cube layouts survive; partitioned UB needs its new local
      // shape established before a physical layout can be assigned.
      for (const auto &[data, layout] :
           annotation.value().cast<Map<Var, Layout>>())
        Require(!ownership_->OwnsData(data),
                "row ownership must be established before owned UB layout "
                "annotations");
    }
    SBlock result = GetRef<SBlock>(op);
    auto *writer = result.CopyOnWrite();
    writer->body = VisitStmt(op->body);
    if (op->init.has_value())
      writer->init = VisitStmt(op->init.value());
    writer->alloc_buffers.clear();
    for (const Buffer &buffer : op->alloc_buffers)
      writer->alloc_buffers.push_back(Remap(buffer));
    writer->reads.clear();
    writer->writes.clear();
    for (const BufferRegion &region : op->reads)
      writer->reads.push_back(MetadataRegion(region));
    for (const BufferRegion &region : op->writes)
      writer->writes.push_back(MetadataRegion(region));
    for (const auto &match : op->match_buffers)
      Require(ownership_->Find(match->buffer) < 0 &&
                  ownership_->Find(match->source->buffer) < 0,
              "owned match-buffer aliases need an explicit pre-partition "
              "coordinate map");
    return result;
  }

  RowOwnership *ownership_;
  Var sid_;
  arith::Analyzer analyzer_;
  ffi::Map<Buffer, Buffer> buffers_;
  Optional<Var> active_row_;
  int active_owner_{-1};
};

class MixedPartitionRewriter : public StmtMutator {
public:
  explicit MixedPartitionRewriter(const Stmt &body) {
    // Reuse grid ranges and enclosing scalar binds from the upstream kernel
    // context collector; validity is not reconstructed from UB capacity.
    for (const auto &context : ascend::kernel_rewriter_detail::
             TilelangKernelContextCollector::Collect(body))
      outer_contexts_.emplace(context.root.get(), context.outer_ctx);
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    const auto *iv = op->node.as<IterVarNode>();
    bool cthread = op->attr_key == tirx::attr::thread_extent && iv &&
                   iv->thread_tag == "cthread";
    if (!cthread)
      return StmtMutator::VisitStmt_(op);
    Optional<Var> saved_sid = sid_;
    Optional<PrimExpr> saved_extent = sid_extent_;
    sid_ = iv->var;
    sid_extent_ = op->value;
    Stmt body = VisitStmt(op->body);
    sid_ = saved_sid;
    sid_extent_ = saved_extent;
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (op->block->name_hint != "tilelang_root")
      return StmtMutator::VisitStmt_(op);
    if (auto marker = op->block->annotations.Get(kRowPartitioned)) {
      Require(marker.value().cast<Integer>()->value == 2,
              "the existing row partition marker must specify two AIVs");
      return GetRef<Stmt>(op);
    }
    const ConstrSet &outer_ctx = outer_contexts_.at(op->block.get());
    RowOwnership ownership(op->block, outer_ctx);
    if (!ownership.IsMixed())
      return GetRef<Stmt>(op);
    ownership.Analyze();
    if (ownership.Empty())
      return GetRef<Stmt>(op);
    bool inner_cthread = false;
    PostOrderVisit(op->block->body, [&](const ObjectRef &node) {
      if (const auto *attr = node.as<AttrStmtNode>()) {
        if (const auto *iv = attr->node.as<IterVarNode>())
          inner_cthread |= attr->attr_key == tirx::attr::thread_extent &&
                           iv->thread_tag == "cthread";
      }
    });
    Require(!inner_cthread,
            "inner cthread scopes conflict with kernel-wide row ownership; "
            "use ordinary unscoped Parallel intent for automatic partitioning");
    if (sid_.has_value()) {
      const auto *extent = sid_extent_.value().as<IntImmNode>();
      Require(extent && extent->value == 2,
              "a mixed dav-2201 kernel requires cthread extent two");
    }
    Var sid = sid_.value_or(Var("aiv_row_sid", DataType::Int(32)));
    SBlock root =
        OwnedKernelRewriter(&ownership, sid, outer_ctx).Rewrite(op->block);
    Stmt result = SBlockRealize(op->iter_values, op->predicate, root, op->span);
    if (!sid_.has_value()) {
      IterVar iv(Range(0, 2), sid, IterVarType::kThreadIndex, "cthread");
      result = AttrStmt(iv, tirx::attr::thread_extent, Integer(2), result);
    }
    return result;
  }

  Optional<Var> sid_;
  Optional<PrimExpr> sid_extent_;
  std::unordered_map<const SBlockNode *, ConstrSet> outer_contexts_;
};

} // namespace

tvm::transform::Pass AscendLowerMixedPartition() {
  return tirx::transform::CreatePrimFuncPass(
      [](PrimFunc func, const IRModule &, const tvm::transform::PassContext &) {
        auto target = func->GetAttr<Target>(tvm::attr::kTarget);
        ICHECK(target.has_value())
            << "AscendLowerMixedPartition requires a bound target";
        if (GetAscendTargetInfo(target.value()).architecture ==
            AscendArchitecture::kDav2201)
          func.CopyOnWrite()->body =
              MixedPartitionRewriter(func->body)(func->body);
        return func;
      },
      0, "tl.AscendLowerMixedPartition", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendLowerMixedPartition",
                        AscendLowerMixedPartition);
}

} // namespace tl
} // namespace tvm
