/*!
 * \file ascend/transform/lower_gm_handoff.cc
 * \brief Expose dav-2201 mixed memory transitions as ordinary copy tasks.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/op/copy.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/storage_footprint.h"
#include "op/builtin.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
using namespace tirx;

namespace {

PrimExpr HandoffRegion(const Buffer &buffer, const Array<Range> &ranges,
                       int mask) {
  Array<PrimExpr> indices;
  for (const Range &range : ranges)
    indices.push_back(range->min);
  Array<PrimExpr> args{BufferLoad(buffer, indices), Integer(mask)};
  for (const Range &range : ranges)
    args.push_back(range->extent);
  return Call(DataType::Handle(), region(), args);
}

Stmt CopyRegion(const Buffer &src, const Array<Range> &src_ranges,
                const Buffer &dst, const Array<Range> &dst_ranges,
                const Map<String, ObjectRef> &annotations = {}) {
  return Evaluate(Call(DataType::Handle(), AscendCopy::Get(),
                       {HandoffRegion(src, src_ranges, kAccessRead),
                        HandoffRegion(dst, dst_ranges, kAccessWrite)},
                       annotations));
}

Stmt CoreRegion(const char *core, Stmt body,
                const Map<String, ffi::Any> &annotations = {}) {
  return SBlock({}, {}, {}, core, std::move(body), std::nullopt, {}, {},
                annotations);
}

bool IsCoreRegion(const SBlock &block) {
  return block->name_hint == "CUBE" || block->name_hint == "VECTOR";
}

// A logical transition can be written inside T.Vector(), although its first
// or last physical transfer executes on AIC. Distribute the existing core
// region to its leaves, allowing the explicit transfer regions to override it.
// Loops and bindings remain one control tree; no task or epoch is duplicated.
class CoreRegionDistributor : public StmtMutator {
public:
  Stmt Rewrite(Stmt body) { return VisitStmt(body); }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "CUBE" && op->name_hint != "VECTOR")
      return StmtMutator::VisitStmt_(op);
    TVM_FFI_CHECK(op->alloc_buffers.empty() && op->match_buffers.empty() &&
                      op->iter_vars.empty() && !op->init.defined() &&
                      op->reads.empty() && op->writes.empty(),
                  ValueError)
        << "Ascend dav-2201 GM handoff requires allocation-free core regions";
    for (const auto &annotation : op->annotations)
      TVM_FFI_CHECK(op->name_hint == "VECTOR" &&
                        annotation.first == "vector_count",
                    ValueError)
          << "Ascend GM handoff cannot discard core-region annotations";
    String saved = core_;
    auto saved_annotations = annotations_;
    core_ = op->name_hint;
    annotations_ = op->annotations;
    Stmt result = VisitStmt(op->body);
    core_ = saved;
    annotations_ = std::move(saved_annotations);
    return result;
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (!IsCoreRegion(op->block))
      return StmtMutator::VisitStmt_(op);
    TVM_FFI_CHECK(op->iter_values.empty(), ValueError)
        << "Ascend GM handoff requires unbound core regions";
    // A realize owns its predicate. Keep that lexical scope without retaining
    // a core label that would override the physical copies below it.
    Stmt body = VisitStmt(op->block);
    SBlock block({}, {}, {}, "ascend_core_region", std::move(body));
    return SBlockRealize({}, op->predicate, block, op->span);
  }

  Stmt Leaf(Stmt stmt) {
    return core_.empty()
               ? stmt
               : CoreRegion(core_.c_str(), std::move(stmt), annotations_);
  }
  Stmt VisitStmt_(const EvaluateNode *op) final {
    return Leaf(GetRef<Stmt>(op));
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    return Leaf(GetRef<Stmt>(op));
  }
  Stmt VisitStmt_(const ForNode *op) final {
    // Parallel is one expression domain. Moving its scalar stores into
    // separate blocks would destroy the dense native vector expression.
    if (op->kind == ForKind::kParallel)
      return Leaf(GetRef<Stmt>(op));
    return StmtMutator::VisitStmt_(op);
  }
  String core_;
  Map<String, ffi::Any> annotations_;
};

class GMHandoffLowerer : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.defined(), ValueError)
        << "AscendLowerGMHandoff requires a bound target";
    if (GetAscendTargetInfo(target.value()).architecture !=
        AscendArchitecture::kDav2201)
      return func;
    arith::Analyzer analyzer;
    GMHandoffLowerer lowerer(&analyzer);
    func.CopyOnWrite()->body = lowerer.VisitStmt(func->body);
    return func;
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  explicit GMHandoffLowerer(arith::Analyzer *analyzer) : Parent(analyzer) {}
  struct LaunchAxis {
    PrimExpr index{Integer(0)};
    PrimExpr extent{Integer(1)};
  };
  struct Root {
    std::vector<Buffer> buffers;
    std::array<LaunchAxis, 3> axes;
  };

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent)
      return Parent::VisitStmt_(op);
    const auto *iv = op->node.as<IterVarNode>();
    std::string tag = iv ? std::string(iv->thread_tag) : "";
    if (!iv || tag.size() != 10 || tag.substr(0, 9) != "blockIdx.")
      return Parent::VisitStmt_(op);
    int axis = tag[9] - 'x';
    TVM_FFI_CHECK(axis >= 0 && axis < 3, ValueError)
        << "Ascend GM handoff requires an ordinary block grid";
    LaunchAxis saved = axes_[axis];
    axes_[axis] = {iv->var, op->value};
    Stmt result = Parent::VisitStmt_(op);
    axes_[axis] = std::move(saved);
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root")
      return Parent::VisitStmt_(op);
    roots_.push_back({{}, axes_});
    SBlock result = Downcast<SBlock>(Parent::VisitStmt_(op));
    Root root = std::move(roots_.back());
    roots_.pop_back();
    if (!root.buffers.empty() ||
        result->annotations.count(attr::kCompilerWorkspaceStorages)) {
      auto *writer = result.CopyOnWrite();
      Array<Var> compiler_storages = ascend::CompilerWorkspaceStorageList(result);
      for (const Buffer &buffer : root.buffers) {
        writer->alloc_buffers.push_back(buffer);
        compiler_storages.push_back(buffer->data);
      }
      writer->annotations.Set(attr::kCompilerWorkspaceStorages,
                              compiler_storages);
      if (!root.buffers.empty())
        writer->body = CoreRegionDistributor().Rewrite(writer->body);
    }
    return result;
  }

  // Validate before constructing products or rounded constants: overflowing
  // PrimExpr arithmetic would otherwise hide an invalid workspace capacity.
  static int64_t AlignCapacity(int64_t extent) {
    TVM_FFI_CHECK(extent > 0 &&
                      extent <= std::numeric_limits<int32_t>::max() - 15,
                  ValueError)
        << "Ascend GM handoff tile capacity exceeds int32 aligned geometry";
    return (extent + 15) / 16 * 16;
  }

  std::pair<int64_t, PrimExpr> Grid(const Root &root) {
    int64_t groups = 1;
    PrimExpr group = make_zero(DataType::Int(64));
    for (int axis = 2; axis >= 0; --axis) {
      PrimExpr extent = analyzer_->Simplify(root.axes[axis].extent);
      const auto *constant = extent.as<IntImmNode>();
      TVM_FFI_CHECK(constant && constant->value > 0, ValueError)
          << "Ascend GM handoff requires a static positive block grid";
      TVM_FFI_CHECK(constant->value <=
                        std::numeric_limits<uint32_t>::max() / groups,
                    ValueError)
          << "Ascend GM handoff block grid exceeds uint32 launch capacity";
      groups *= constant->value;
      group = group * make_const(DataType::Int(64), constant->value) +
              cast(DataType::Int(64), root.axes[axis].index);
    }
    return {groups, analyzer_->Simplify(group)};
  }

  Array<Range> Bounds(const Buffer &buffer, const Array<Range> &ranges) {
    TVM_FFI_CHECK(ranges.size() == buffer->shape.size() && ranges.size() >= 2,
                  ValueError)
        << "Ascend GM handoff requires rank-matched matrix regions";
    Array<Range> result;
    for (size_t axis = 0; axis < ranges.size(); ++axis) {
      const Range &range = ranges[axis];
      bool leading = axis + 2 < ranges.size();
      TVM_FFI_CHECK(!leading || analyzer_->CanProveEqual(range->extent, 1),
                    ValueError)
          << "Ascend GM handoff requires singleton leading axes";
      PrimExpr zero = make_zero(range->extent.dtype());
      PrimExpr shape = cast(range->extent.dtype(), buffer->shape[axis]);
      PrimExpr origin = cast(range->extent.dtype(), range->min);
      if (!leading)
        TVM_FFI_CHECK(analyzer_->CanProve(origin >= zero,
                                          arith::ProofStrength::kSymbolicBound),
                      ValueError)
            << "Ascend GM handoff requires nonnegative matrix origins";
      TVM_FFI_CHECK(analyzer_->CanProve(range->extent >= zero,
                                        arith::ProofStrength::kSymbolicBound),
                    ValueError)
          << "Ascend GM handoff requires nonnegative region extents";
      PrimExpr extent = Min(range->extent, Max(shape - origin, zero));
      if (leading)
        extent = Select(origin >= zero && origin < shape, range->extent, zero);
      result.push_back(
          Range::FromMinExtent(range->min, analyzer_->Simplify(extent)));
    }
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (!call || !IsAscendCopyCall(call))
      return Parent::VisitStmt_(op);
    AscendCopy copy(call->args, call->annotations);
    bool to_ub = IsL0CBuffer(copy->src) && IsSharedBuffer(copy->dst);
    bool to_l1 = IsSharedBuffer(copy->src) && IsL1Buffer(copy->dst);
    if (!to_ub && !to_l1)
      return GetRef<Stmt>(op);
    TVM_FFI_CHECK(!roots_.empty(), ValueError)
        << "Ascend dav-2201 GM handoff requires tilelang_root";
    TVM_FFI_CHECK(copy->transpose == 0 && copy->dual_dst_ctl == 0, ValueError)
        << "Ascend dav-2201 logical GM handoff requires matching axes; "
           "use ordinary copy rather than hardware dual_copy hints";
    TVM_FFI_CHECK(copy->src->dtype == copy->dst->dtype, ValueError)
        << "Ascend dav-2201 GM handoff preserves dtype; express conversion "
           "as an ordinary UB expression";
    DataType dtype = copy->src->dtype;
    TVM_FFI_CHECK((to_ub && dtype == DataType::Float(32)) ||
                      (to_l1 && (dtype.is_float16() || dtype.is_bfloat16())),
                  ValueError)
        << "Ascend dav-2201 GM handoff supports FP32 L0C->UB and "
           "FP16/BF16 UB->L1";
    // This boundary pairs the trailing matrix axes explicitly, including a
    // one-row/one-column matrix. Bounds come from the original operands;
    // scratch capacity must never authorize an extra read.
    Array<Range> src = Bounds(copy->src, copy->src_range);
    Array<Range> dst = Bounds(copy->dst, copy->dst_range);
    Array<Range> destination = dst;
    PrimExpr leading_valid = Integer(1);
    for (const auto *ranges : {&src, &dst})
      for (size_t axis = 0; axis + 2 < ranges->size(); ++axis)
        leading_valid = Min(leading_valid,
                            cast(DataType::Int(32), (*ranges)[axis]->extent));
    leading_valid = analyzer_->Simplify(leading_valid);
    for (int tail_axis = 2; tail_axis >= 1; --tail_axis) {
      size_t s = src.size() - tail_axis;
      size_t d = dst.size() - tail_axis;
      // The smaller region is the actual transfer domain; the larger one
      // describes its operand capacity. In particular FA exports valid Q
      // rows into a full UB tile, and imports valid P rows into a full L1
      // operand. Keep that domain explicit rather than reading scratch's pad.
      PrimExpr extent = analyzer_->Simplify(
          Min(src[s]->extent, cast(src[s]->extent.dtype(), dst[d]->extent)));
      src.Set(s, Range::FromMinExtent(src[s]->min, extent));
      dst.Set(d, Range::FromMinExtent(dst[d]->min,
                                      cast(dst[d]->extent.dtype(), extent)));
    }
    PrimExpr rows = src[src.size() - 2]->extent;
    PrimExpr cols = src.back()->extent;
    // Capacity belongs to the logical matrix, while valid regions may be
    // smaller. Ownership lowering subsequently rewrites only AIV regions.
    const Buffer &capacity = to_ub ? copy->src : copy->dst;
    const auto *r =
        capacity->shape[capacity->shape.size() - 2].as<IntImmNode>();
    const auto *c = capacity->shape.back().as<IntImmNode>();
    TVM_FFI_CHECK(r && c, ValueError)
        << "Ascend dav-2201 GM handoff requires static tile capacities";
    const int64_t row_capacity = AlignCapacity(r->value);
    const int64_t column_capacity = AlignCapacity(c->value);
    auto [groups, group] = Grid(roots_.back());
    const int64_t bytes = dtype.bits() / 8;
    TVM_FFI_CHECK(column_capacity <=
                          std::numeric_limits<int64_t>::max() / bytes &&
                      row_capacity <= std::numeric_limits<int64_t>::max() /
                                          (column_capacity * bytes) &&
                      groups <= std::numeric_limits<int64_t>::max() /
                                    (row_capacity * column_capacity * bytes),
                  ValueError)
        << "Ascend dav-2201 GM handoff workspace byte size overflows int64";
    Buffer workspace = decl_buffer(
        {make_const(DataType::Int(64), groups),
         make_const(DataType::Int(32), row_capacity),
         make_const(DataType::Int(32), column_capacity)},
        dtype, "gm_handoff_" + std::to_string(next_id_++), "global");
    roots_.back().buffers.push_back(workspace);
    Array<Range> middle{Range::FromMinExtent(group, leading_valid),
                        Range::FromMinExtent(make_zero(rows.dtype()), rows),
                        Range::FromMinExtent(make_zero(cols.dtype()), cols)};
    Map<String, ObjectRef> annotations = call->annotations;
    // Logical UB->L1 hints do not configure either physical DMA route.
    annotations.erase("nd2nz");
    annotations.erase("dual_dst_ctl");
    if (to_ub) {
      return SeqStmt({CoreRegion("CUBE", CopyRegion(copy->src, src, workspace,
                                                    middle, annotations)),
                      CoreRegion("VECTOR", CopyRegion(workspace, middle,
                                                      copy->dst, dst))});
    }
    Array<Stmt> sequence{
        CoreRegion("VECTOR", CopyRegion(copy->src, src, workspace, middle))};
    // Scratch's padded capacity is in bounds, but its unwritten lanes are not
    // zero. Define the L1 operand tail explicitly before a partial import.
    size_t row_axis = destination.size() - 2;
    size_t col_axis = destination.size() - 1;
    bool source_tail =
        !analyzer_->CanProveEqual(rows, destination[row_axis]->extent) ||
        !analyzer_->CanProveEqual(cols, destination[col_axis]->extent);
    if (source_tail && !is_zero(leading_valid)) {
      // Keep the fill in the currently legal logical allocation. Storage
      // normalization extends complete axes only after their physical shape
      // has been established; no intermediate region exceeds its buffer.
      Array<Range> full = destination;
      const Range &columns = full[col_axis];
      bool complete_columns =
          analyzer_->CanProveEqual(columns->min, 0) &&
          analyzer_->CanProveEqual(columns->extent, copy->dst->shape[col_axis]);
      TVM_FFI_CHECK(
          complete_columns ||
              (analyzer_->CanProveEqual(FloorMod(columns->min, 16), 0) &&
               analyzer_->CanProveEqual(FloorMod(columns->extent, 16), 0)),
          ValueError)
          << "Ascend GM handoff L1 tail initialization requires a "
             "C0-aligned column footprint or a complete destination axis";
      // Keep empty leading owners as data predicates on the fill, never as
      // control flow around a scheduled task/collective completion.
      const Range &row = full[row_axis];
      full.Set(row_axis, Range::FromMinExtent(
                             row->min, analyzer_->Simplify(Select(
                                           leading_valid > 0, row->extent,
                                           make_zero(row->extent.dtype())))));
      for (size_t axis = 0; axis + 2 < full.size(); ++axis)
        full.Set(axis, Range::FromMinExtent(
                           full[axis]->min,
                           make_const(full[axis]->extent.dtype(), 1)));
      Stmt fill = ascend::MakeL1Fill(copy->dst, full, make_zero(dtype));
      Call call = Downcast<Call>(fill.as<EvaluateNode>()->value);
      call.CopyOnWrite()->annotations.Set(attr::kInitializeL1Padding,
                                         Integer(1));
      sequence.push_back(CoreRegion("CUBE", Evaluate(call)));
    }
    sequence.push_back(CoreRegion(
        "CUBE", CopyRegion(workspace, middle, copy->dst, dst, annotations)));
    return SeqStmt(sequence);
  }

  std::array<LaunchAxis, 3> axes_;
  std::vector<Root> roots_;
  int next_id_{0};
};
} // namespace

tvm::transform::Pass AscendLowerGMHandoff() {
  return tirx::transform::CreatePrimFuncPass(
      [](PrimFunc func, IRModule, tvm::transform::PassContext) {
        return GMHandoffLowerer::Rewrite(std::move(func));
      },
      0, "tl.AscendLowerGMHandoff", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLowerGMHandoff",
                                   AscendLowerGMHandoff);
}
} // namespace tl
} // namespace tvm
