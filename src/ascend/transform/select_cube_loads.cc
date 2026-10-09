// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file select_cube_loads.cc
 * \brief Select native Cube loads after physical layout and OOB padding.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/cube_load.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/register_state.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/ascend_pipe.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/auto_schedule/task_annotations.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
using namespace tirx;
namespace {

class CubeLoadSelector : public arith::IRMutatorWithAnalyzer {
public:
  explicit CubeLoadSelector(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  SBlock Rewrite(const SBlock &root) {
    bool existing_state = false;
    bool unknown_state = false;
    PostOrderVisit(root->body, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        existing_state |= call->op.same_as(ascend_set_load3d_config()) ||
                          call->op.same_as(AscendCubeLoad3D::Get()) ||
                          call->op.same_as(ascend_load_cbuf_to_ca_3d()) ||
                          call->op.same_as(ascend_load_cbuf_to_cb_3d());
        unknown_state |= ascend::GetRegisterStateEffect(
                             call, ascend::RegisterState::kLoad3DConfig) !=
                         ascend::RegisterStateEffect::kPreserves;
      }
    });
    if (existing_state || unknown_state)
      return root;
    SBlock rewritten = Downcast<SBlock>(operator()(root));
    Optional<PrimExpr> a_width = SelectWidth(true);
    Optional<PrimExpr> b_width = SelectWidth(false);
    const bool select_a = a_width.defined();
    const bool select_b = b_width.defined();
    if (!select_a && !select_b)
      return root;
    // FMATRIX_A and FMATRIX_B are independent banks. Each selected bank has
    // one invariant configuration; restore every other width's exact copy.
    // Geometry and effects were proved only once.
    std::unordered_map<const CallNode *, Stmt> fallbacks;
    for (const SelectedLoad &load : selected_loads_)
      if (const auto &width = load.is_l0a ? a_width : b_width;
          !width.defined() || !Equal(width.value(), load.width))
        fallbacks.emplace(load.selected.get(), load.original);
    if (!fallbacks.empty()) {
      class RestoreCopies : public StmtExprMutator {
      public:
        explicit RestoreCopies(
            const std::unordered_map<const CallNode *, Stmt> &fallbacks)
            : fallbacks_(fallbacks) {}

      private:
        Stmt VisitStmt_(const EvaluateNode *op) final {
          auto it = fallbacks_.find(op->value.as<CallNode>());
          if (it != fallbacks_.end())
            return it->second;
          return StmtExprMutator::VisitStmt_(op);
        }
        const std::unordered_map<const CallNode *, Stmt> &fallbacks_;
      };
      rewritten = Downcast<SBlock>(RestoreCopies(fallbacks)(rewritten));
    }
    ascend::TaskMetadata ownership;
    ownership.core_mask = kCoreCube;
    Stmt setup = Evaluate(Call(DataType::Void(), ascend_set_load3d_config(),
                               {select_a ? a_width.value() : Integer(1),
                                select_b ? b_width.value() : Integer(1)}));
    setup = AttrStmt(ascend::MergeTaskMetadata(ownership, Integer(0)),
                     attr::kAscendTask, Integer(1), setup);
    rewritten.CopyOnWrite()->body = SeqStmt({setup, rewritten->body});
    return rewritten;
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;

  static std::optional<int64_t> Multiply(int64_t lhs, int64_t rhs) {
    if (lhs < 0 || rhs < 0 ||
        (rhs && lhs > std::numeric_limits<int64_t>::max() / rhs))
      return std::nullopt;
    return lhs * rhs;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto saved = execution_bound_;
    // Count before entering the body: its extent>0 constraint must not erase
    // a zero-trip domain. Guards can only reduce this static upper bound.
    PrimExpr extent = analyzer_->Simplify(op->extent);
    PrimExpr step = analyzer_->Simplify(op->step.value_or(Integer(1)));
    const auto *count = extent.as<IntImmNode>();
    const auto *stride = step.as<IntImmNode>();
    if (execution_bound_ && *execution_bound_ == 0) {
      // An unreachable enclosing loop also bounds unknown descendants.
    } else if (execution_bound_ && count && count->value >= 0 && stride &&
               stride->value > 0) {
      // ceildiv without an overflowing extent+step-1 intermediate.
      int64_t trips =
          count->value / stride->value + (count->value % stride->value != 0);
      execution_bound_ = Multiply(*execution_bound_, trips);
    } else {
      execution_bound_ = std::nullopt;
    }
    Stmt result = Parent::VisitStmt_(op);
    execution_bound_ = saved;
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    auto saved = execution_bound_;
    if (!execution_bound_ || *execution_bound_ != 0)
      execution_bound_ = std::nullopt;
    Stmt result = Parent::VisitStmt_(op);
    execution_bound_ = saved;
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool atomic = op->attr_key == attr::kAscendTask ||
                  op->attr_key == attr::kAscendPerCoreTask;
    atomic_depth_ += atomic;
    Stmt result = Parent::VisitStmt_(op);
    atomic_depth_ -= atomic;
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    LayoutMap saved = layouts_;
    if (auto value = op->annotations.Get(attr::kLayoutMap))
      for (const auto &[buffer, layout] : value.value().cast<LayoutMap>())
        layouts_.Set(buffer, layout);
    Stmt result = Parent::VisitStmt_(op);
    layouts_ = saved;
    return result;
  }

  bool Prove(const PrimExpr &condition) {
    return analyzer_->CanProve(condition, arith::ProofStrength::kSymbolicBound);
  }
  bool Equal(const PrimExpr &lhs, const PrimExpr &rhs) {
    return analyzer_->CanProveEqual(lhs, rhs);
  }
  bool InBounds(const PrimExpr &origin, const PrimExpr &extent,
                const PrimExpr &capacity) {
    // Share the semantic boundary with native copy lowering. Even an int64
    // origin+extent can overflow, so prove origin<=capacity before comparing
    // the nonnegative count with capacity-origin.
    return CanProveAscendRangeInBounds(origin, extent, capacity, analyzer_);
  }
  bool LeadingUnit(const Buffer &buffer, const Array<Range> &ranges) {
    if (ranges.size() != buffer->shape.size() || ranges.size() < 2)
      return false;
    for (size_t i = 0; i + 2 < ranges.size(); ++i)
      if (!Equal(ranges[i]->extent, 1) ||
          !Prove(ranges[i]->min >= 0 && ranges[i]->min < buffer->shape[i]))
        return false;
    return true;
  }

  PrimExpr Region(const Buffer &buffer, const Array<Range> &ranges, int mask) {
    Array<PrimExpr> indices, args;
    for (const Range &range : ranges)
      indices.push_back(range->min);
    args.push_back(BufferLoad(buffer, indices));
    args.push_back(IntImm(DataType::Int(32), mask));
    for (const Range &range : ranges)
      args.push_back(range->extent);
    return Call(DataType::Handle(), region(), args);
  }

  std::optional<int64_t>
  DescriptorSaving(const AscendCopy &copy, const Layout &src_layout,
                   const Layout &dst_layout,
                   const AscendFractalLayoutInfo &src_info,
                   const AscendFractalLayoutInfo &dst_info, bool is_l0a) {
    if (!execution_bound_)
      return std::nullopt;
    AscendFractalRegionInfo src_region, dst_region;
    if (!TryExtractAscendFractalRegion(src_layout, copy->src_range,
                                       &src_region) ||
        !TryExtractAscendFractalRegion(dst_layout, copy->dst_range,
                                       &dst_region))
      return std::nullopt;
    // Match Copy::Lower's exact 2201 descriptor organization, including its
    // shared contiguous-B rule. A selected Load3D emits one descriptor.
    bool needs_transpose = src_info.c0_axis != dst_info.c0_axis;
    bool merge_b = !is_l0a && ascend::CanMerge2201L0BLoad(
                                  copy->src_range, src_info, src_region,
                                  dst_region, needs_transpose, analyzer_);
    PrimExpr descriptor_count =
        is_l0a ? dst_region.outer1->extent : dst_region.outer0->extent;
    if (merge_b)
      descriptor_count = Integer(1);
    PrimExpr count = analyzer_->Simplify(descriptor_count);
    const auto *descriptors = count.as<IntImmNode>();
    if (!descriptors || descriptors->value < 1)
      return std::nullopt;
    return Multiply(descriptors->value - 1, *execution_bound_);
  }

  Optional<PrimExpr> SelectWidth(bool is_l0a) {
    Optional<PrimExpr> first_width;
    bool conflicting = false;
    for (const auto &load : selected_loads_) {
      if (load.is_l0a != is_l0a)
        continue;
      if (first_width.defined() && !Equal(first_width.value(), load.width))
        conflicting = true;
      first_width = load.width;
    }
    // Preserve the established uniform-bank path, including dynamic loops.
    if (!conflicting)
      return first_width;
    std::unordered_map<int64_t, int64_t> savings;
    for (const auto &load : selected_loads_) {
      if (load.is_l0a != is_l0a)
        continue;
      if (!load.descriptor_saving)
        return std::nullopt;
      int64_t &total = savings[load.width.as<IntImmNode>()->value];
      if (total > std::numeric_limits<int64_t>::max() - *load.descriptor_saving)
        return std::nullopt;
      total += *load.descriptor_saving;
    }
    int64_t best = 0, best_width = 0;
    bool tied = false;
    for (const auto &[width, saving] : savings) {
      if (saving > best) {
        best = saving;
        best_width = width;
        tied = false;
      } else if (saving == best) {
        tied = true;
      }
    }
    // Descriptor savings are an upper-bound count, not a latency estimate.
    // No strict winner (or unknown/overflow above) keeps the old fallback.
    if (!best || tied)
      return std::nullopt;
    for (const auto &load : selected_loads_)
      if (load.is_l0a == is_l0a &&
          load.width.as<IntImmNode>()->value == best_width)
        return load.width;
    return std::nullopt;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (atomic_depth_ || !IsAscendCopyCall(call))
      return Parent::VisitStmt_(op);
    // The selected recipe implements ordinary copy only. Keep every annotated
    // copy on its established path: hardware controls, layout hints and
    // unknown annotations must never disappear in native instruction selection.
    if (!call->annotations.empty())
      return Parent::VisitStmt_(op);
    AscendCopy copy(call->args, call->annotations);
    const bool is_l0a = IsL0ABuffer(copy->dst);
    if (!IsL1Buffer(copy->src) || (!is_l0a && !IsL0BBuffer(copy->dst)) ||
        copy->transpose != 0 || copy->src->dtype != copy->dst->dtype ||
        copy->src->dtype.lanes() != 1 ||
        !(copy->src->dtype.is_float16() || copy->src->dtype.is_bfloat16()) ||
        !copy->src->strides.empty() || !copy->dst->strides.empty() ||
        !is_zero(copy->src->elem_offset) || !is_zero(copy->dst->elem_offset) ||
        !LeadingUnit(copy->src, copy->src_range) ||
        !LeadingUnit(copy->dst, copy->dst_range))
      return Parent::VisitStmt_(op);

    auto src_layout = ascend::FindLayoutForBuffer(layouts_, copy->src);
    auto dst_layout = ascend::FindLayoutForBuffer(layouts_, copy->dst);
    AscendFractalLayoutInfo src_info, dst_info;
    if (!src_layout.defined() || !dst_layout.defined() ||
        !IsAscendFractalLayout(src_layout.value()) ||
        !IsAscendFractalLayout(dst_layout.value()) ||
        !TryExtractAscendFractalLayout(src_layout.value(), copy->src,
                                       &src_info) ||
        !TryExtractAscendFractalLayout(dst_layout.value(), copy->dst,
                                       &dst_info) ||
        src_info.kind != AscendFractalKind::kMajorK ||
        dst_info.kind != (is_l0a ? AscendFractalKind::kL0AMajorK
                                 : AscendFractalKind::kMajorMN))
      return Parent::VisitStmt_(op);

    size_t sn = copy->src_range.size(), dn = copy->dst_range.size();
    const Range &sr = copy->src_range[sn - 2];
    const Range &sc = copy->src_range[sn - 1];
    const Range &dr = copy->dst_range[dn - 2];
    const Range &dc = copy->dst_range[dn - 1];
    PrimExpr physical_rows = copy->src->shape[sn - 2];
    PrimExpr physical_cols = copy->src->shape[sn - 1];
    PrimExpr loaded_rows = copy->dst->shape[dn - 2];
    PrimExpr loaded_cols = copy->dst->shape[dn - 1];
    // A uses (rows,cols)=(M,K), NN B uses (K,N). Both recipes copy one
    // complete compact destination tile. A covers the source's full padded M;
    // B covers its full padded N and materializes the native ZN order. Sizes
    // and origins are elements, never classic Load2D fractal counts.
    if (!Equal(dr->min, 0) || !Equal(dc->min, 0) ||
        !Equal(src_info.outer1 * src_info.row_frac, physical_rows) ||
        !Equal(src_info.outer0 * src_info.c0, physical_cols) ||
        !Equal(dst_info.outer1 * dst_info.row_frac,
               is_l0a ? loaded_rows : loaded_cols) ||
        !Equal(dst_info.outer0 * dst_info.c0,
               is_l0a ? loaded_cols : loaded_rows) ||
        !Equal(sr->extent, loaded_rows) || !Equal(dr->extent, loaded_rows) ||
        !Equal(sc->extent, loaded_cols) || !Equal(dc->extent, loaded_cols) ||
        !Equal(floormod(physical_rows, 16), 0) ||
        !Equal(floormod(physical_cols, 16), 0) ||
        !Equal(floormod(loaded_rows, 16), 0) ||
        !Equal(floormod(loaded_cols, 16), 0) ||
        !Prove(physical_rows > 0 && physical_rows <= 32767 &&
               physical_cols > 0 && physical_cols <= 65535 && loaded_rows > 0 &&
               loaded_rows <= 65535 && loaded_cols > 0 && loaded_cols <= 65535))
      return Parent::VisitStmt_(op);
    if (is_l0a ? (!Equal(sr->min, 0) || !Equal(loaded_rows, physical_rows) ||
                  !Equal(floormod(sc->min, 16), 0) ||
                  !InBounds(sc->min, loaded_cols, physical_cols))
               : (!Equal(sc->min, 0) || !Equal(loaded_cols, physical_cols) ||
                  !Equal(floormod(sr->min, 16), 0) ||
                  !InBounds(sr->min, loaded_rows, physical_rows)))
      return Parent::VisitStmt_(op);

    PrimExpr configured_width = analyzer_->Simplify(physical_rows);
    if (!configured_width.as<IntImmNode>())
      return Parent::VisitStmt_(op);
    Array<Range> reads = copy->src_range, writes = copy->dst_range;
    // Consume the shared NormalizeCubeCopyFootprints contract verbatim.
    // Selection never invents or enlarges a hardware read/write region.
    Call selected(DataType::Void(), AscendCubeLoad3D::Get(),
                  {Region(copy->src, reads, kAccessRead),
                   Region(copy->dst, writes, kAccessWrite), physical_rows,
                   physical_cols},
                  call->annotations);
    selected_loads_.push_back(
        {selected, GetRef<Stmt>(op), is_l0a, configured_width,
         DescriptorSaving(copy, src_layout.value(), dst_layout.value(),
                          src_info, dst_info, is_l0a)});
    return Evaluate(selected, op->span);
  }

  struct SelectedLoad {
    Call selected;
    Stmt original;
    bool is_l0a;
    PrimExpr width;
    std::optional<int64_t> descriptor_saving;
  };
  LayoutMap layouts_;
  std::vector<SelectedLoad> selected_loads_;
  std::optional<int64_t> execution_bound_{1};
  int atomic_depth_{0};
};
} // namespace

namespace transform {
tirx::transform::Pass AscendSelectCubeLoads() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tirx::transform::PassContext &) {
    Target target = func->GetAttr<Target>(tvm::attr::kTarget)
                        .value_or(Target::Current(true));
    ICHECK(target.defined()) << "AscendSelectCubeLoads requires a bound target";
    if (GetAscendTargetInfo(target).architecture !=
        AscendArchitecture::kDav2201)
      return func;
    return ascend::RewriteTilelangKernels(
        func, "AscendSelectCubeLoads",
        [](const ascend::TilelangKernelContext &context) {
          arith::Analyzer analyzer;
          context.outer_ctx.ToConstraints().Populate(analyzer);
          return CubeLoadSelector(&analyzer).Rewrite(context.root);
        });
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.AscendSelectCubeLoads", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendSelectCubeLoads",
                                   AscendSelectCubeLoads);
}
} // namespace transform
} // namespace tl
} // namespace tvm
