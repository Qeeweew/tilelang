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
    PostOrderVisit(root->body, [&](const ffi::ObjectRef &node) {
      if (const auto *call = node.as<CallNode>())
        existing_state |= call->op.same_as(ascend_set_load3d_config()) ||
                          call->op.same_as(AscendCubeLoad3D::Get()) ||
                          call->op.same_as(ascend_load_cbuf_to_ca_3d());
    });
    if (existing_state)
      return root;
    SBlock rewritten = Downcast<SBlock>(operator()(root));
    if (different_configurations_ || !physical_m_.defined())
      return root;
    ascend::TaskMetadata ownership;
    ownership.core_mask = kCoreCube;
    Stmt setup = Evaluate(Call(DataType::Void(), ascend_set_load3d_config(),
                               {physical_m_.value()}));
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

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // A one-shot root event must not be consumed by a condition that becomes
    // true on multiple iterations. Keep conditional copies on the established
    // 2D path until their first-consumer protocol is explicitly represented.
    ++conditional_depth_;
    Stmt result = Parent::VisitStmt_(op);
    --conditional_depth_;
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
    if (atomic_depth_ || conditional_depth_ || !IsAscendCopyCall(call))
      return Parent::VisitStmt_(op);
    // The selected recipe implements ordinary copy only. Keep every annotated
    // copy on its established path: hardware controls, layout hints and
    // unknown annotations must never disappear in native instruction selection.
    if (!call->annotations.empty())
      return Parent::VisitStmt_(op);
    AscendCopy copy(call->args, call->annotations);
    if (!IsL1Buffer(copy->src) || !IsL0ABuffer(copy->dst) ||
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
        dst_info.kind != AscendFractalKind::kL0AMajorK)
      return Parent::VisitStmt_(op);

    size_t sn = copy->src_range.size(), dn = copy->dst_range.size();
    const Range &sm = copy->src_range[sn - 2];
    const Range &sk = copy->src_range[sn - 1];
    const Range &dm = copy->dst_range[dn - 2];
    const Range &dk = copy->dst_range[dn - 1];
    PrimExpr physical_m = copy->src->shape[sn - 2];
    PrimExpr physical_k = copy->src->shape[sn - 1];
    PrimExpr loaded_m = copy->dst->shape[dn - 2];
    PrimExpr loaded_k = copy->dst->shape[dn - 1];
    // This narrow recipe selects one compact A tile, covering the source's
    // complete padded M. Sizes are elements, not 2D fractal counts. Unknown
    // bounds, partial M, transpose and noncompact destinations keep 2D.
    if (!Equal(sm->min, 0) || !Equal(dm->min, 0) || !Equal(dk->min, 0) ||
        !Equal(floormod(sk->min, 16), 0) || !Equal(loaded_m, physical_m) ||
        !Equal(src_info.outer1 * src_info.row_frac, physical_m) ||
        !Equal(src_info.outer0 * src_info.c0, physical_k) ||
        !Equal(dst_info.outer1 * dst_info.row_frac, loaded_m) ||
        !Equal(dst_info.outer0 * dst_info.c0, loaded_k) ||
        !Equal(sm->extent, loaded_m) || !Equal(dm->extent, loaded_m) ||
        !Equal(sk->extent, loaded_k) || !Equal(dk->extent, loaded_k) ||
        !Equal(floormod(physical_m, 16), 0) ||
        !Equal(floormod(physical_k, 16), 0) ||
        !Equal(floormod(loaded_k, 16), 0) ||
        !InBounds(sk->min, loaded_k, physical_k) ||
        !Prove(sm->extent > 0 && sk->extent > 0 && dm->extent > 0 &&
               dk->extent > 0 && physical_m > 0 && physical_m <= 32767 &&
               physical_k > 0 && physical_k <= 65535 && loaded_k > 0 &&
               loaded_k <= 65535))
      return Parent::VisitStmt_(op);

    PrimExpr configured_m = analyzer_->Simplify(physical_m);
    if (!configured_m.as<IntImmNode>())
      return Parent::VisitStmt_(op);
    if (physical_m_.defined() && !Equal(physical_m_.value(), configured_m))
      different_configurations_ = true;
    physical_m_ = configured_m;

    Array<Range> reads = copy->src_range, writes = copy->dst_range;
    // Consume the shared NormalizeCubeCopyFootprints contract verbatim.
    // Selection never invents or enlarges a hardware read/write region.
    return Evaluate(
        Call(DataType::Void(), AscendCubeLoad3D::Get(),
             {Region(copy->src, reads, kAccessRead),
              Region(copy->dst, writes, kAccessWrite), physical_m, physical_k},
             call->annotations),
        op->span);
  }

  LayoutMap layouts_;
  Optional<PrimExpr> physical_m_;
  bool different_configurations_{false};
  int atomic_depth_{0};
  int conditional_depth_{0};
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
