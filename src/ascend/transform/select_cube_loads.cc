// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file select_cube_loads.cc
 * \brief Select native Cube loads after physical layout and OOB padding.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/cube_load.h"
#include "ascend/op/oob_padding.h"
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
        unknown_state |= !ascend::PreservesLoad3DConfig(call);
      }
    });
    if (existing_state || unknown_state)
      return root;
    SBlock rewritten = Downcast<SBlock>(operator()(root));
    if (different_configurations_ ||
        (!fmatrix_a_width_.defined() && !fmatrix_b_width_.defined()))
      return root;
    ascend::TaskMetadata ownership;
    ownership.core_mask = kCoreCube;
    Stmt setup = Evaluate(Call(DataType::Void(), ascend_set_load3d_config(),
                               {fmatrix_a_width_.value_or(Integer(1)),
                                fmatrix_b_width_.value_or(Integer(1))}));
    setup = AttrStmt(ascend::MergeTaskMetadata(ownership, Integer(0)),
                     attr::kAscendTask, Integer(1), setup);
    rewritten.CopyOnWrite()->body = SeqStmt({setup, rewritten->body});
    return rewritten;
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;

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
    auto &side_width = is_l0a ? fmatrix_a_width_ : fmatrix_b_width_;
    if (side_width.defined() && !Equal(side_width.value(), configured_width))
      different_configurations_ = true;
    side_width = configured_width;

    Array<Range> reads = copy->src_range, writes = copy->dst_range;
    // Consume the shared NormalizeCubeCopyFootprints contract verbatim.
    // Selection never invents or enlarges a hardware read/write region.
    return Evaluate(Call(DataType::Void(), AscendCubeLoad3D::Get(),
                         {Region(copy->src, reads, kAccessRead),
                          Region(copy->dst, writes, kAccessWrite),
                          physical_rows, physical_cols},
                         call->annotations),
                    op->span);
  }

  LayoutMap layouts_;
  Optional<PrimExpr> fmatrix_a_width_, fmatrix_b_width_;
  bool different_configurations_{false};
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
