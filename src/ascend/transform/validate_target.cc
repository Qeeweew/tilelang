/*!
 * \file ascend/transform/validate_target.cc
 * \brief Validate frontend operations against the selected Ascend target.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>

#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/fill.h"
#include "op/gemm.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

// This is the frontend capability boundary. It does not infer layouts,
// schedule tasks, or repair a route unsupported by the selected hardware.
// Frontend vector candidates are checked for a supported execution domain;
// their definitive contiguous/aligned storage contract belongs to MemBase
// lowering after layout inference and padding.
class Dav2201Validator : public StmtExprVisitor {
public:
  void Validate(const PrimFunc &func) {
    for (const auto &[var, buffer] : func->buffer_map) {
      (void)var;
      ValidateBuffer(buffer);
    }
    VisitStmt(func->body);
  }

private:
  static void Reject(const std::string &operation, const std::string &reason) {
    TVM_FFI_THROW(ValueError)
        << "Ascend dav-2201: " << operation << ": " << reason;
  }

  static void ValidateDType(DataType dtype, const std::string &description) {
    // Void calls configure or issue operations; lanes=0 does not describe a
    // register vector. Their operation capability is checked separately.
    if (dtype.is_void())
      return;
    if (dtype.is_float8() || dtype.is_float4()) {
      Reject(description, "FP8/FP4 storage and arithmetic are unsupported");
    }
    if (dtype.lanes() != 1) {
      Reject(description,
             "vector-valued storage and expressions require dav-2201 "
             "MemBase lowering; VF/RegBase values are unsupported");
    }
  }

  static void ValidateBuffer(const Buffer &buffer) {
    ValidateDType(buffer->dtype, "buffer " + std::string(buffer->name));
    if (IsL0SFBuffer(buffer)) {
      Reject("buffer " + std::string(buffer->name),
             "MX scale-factor storage is unsupported");
    }
  }

  static void RequireMemBase(const std::string &operation) {
    Reject(
        operation,
        "this vector operation has no implemented dav-2201 MemBase lowering");
  }

  void ValidateVectorBuffer(const Buffer &buffer,
                            const std::string &operation) const {
    if (!IsSharedBuffer(buffer) ||
        !(buffer->dtype.is_float16() || buffer->dtype.is_bfloat16() ||
          buffer->dtype == DataType::Float(32)))
      Reject(operation,
             "the implemented MemBase vector path requires private UB "
             "FP16, BF16, or FP32 values");
  }

  void ValidateCopy(const AscendCopy &copy) const {
    const Buffer &src = copy->src;
    const Buffer &dst = copy->dst;
    ValidateBuffer(src);
    ValidateBuffer(dst);
    const std::string route =
        "copy " + std::string(src.scope()) + " -> " + std::string(dst.scope());
    if (IsSharedBuffer(src) && IsSharedBuffer(dst)) {
      ValidateVectorBuffer(src, route);
      ValidateVectorBuffer(dst, route);
      if (src->dtype != dst->dtype || copy->transpose != 0 ||
          copy->dual_dst_ctl != 0)
        Reject(route, "private UB copy must preserve the element type and "
                      "ordinary axes without dual-destination control");
      // Vector lowering establishes the physical dense region, alignment,
      // exact count and overlap contract after layout inference.
      return;
    }
    if ((IsL0CBuffer(src) && IsSharedBuffer(dst)) ||
        (IsSharedBuffer(src) && IsL1Buffer(dst))) {
      bool score = IsL0CBuffer(src);
      bool supported =
          src->dtype == dst->dtype &&
          (score ? src->dtype == DataType::Float(32)
                 : (src->dtype.is_float16() || src->dtype.is_bfloat16()));
      if (!supported || copy->transpose != 0 || copy->dual_dst_ctl != 0)
        Reject(route, "logical GM handoff requires matching FP32 L0C/UB or "
                      "matching FP16/BF16 UB/L1 with ordinary axes");
      // AscendLowerGMHandoff exposes both transfers and their workspace.
      // AscendLowerMixedPartition establishes AIV ownership before layout.
      return;
    }
    if (IsL1Buffer(src) && IsSharedBuffer(dst)) {
      Reject(route, "direct L1 -> UB transfer is unsupported");
    }
    if (IsSharedBuffer(src) || IsSharedBuffer(dst)) {
      const bool load = IsGlobalBuffer(src) && IsSharedBuffer(dst);
      const bool store = IsSharedBuffer(src) && IsGlobalBuffer(dst);
      if (!(load || store))
        Reject(route, "the implemented MemBase copy path requires GM <-> UB");
      ValidateVectorBuffer(load ? dst : src, route);
      if (src->dtype != dst->dtype || copy->transpose != 0)
        Reject(
            route,
            "MemBase GM <-> UB copy must preserve the element type and axes");
      return;
    }
    if (IsGlobalBuffer(src) && IsL1Buffer(dst)) {
      if (src->dtype != dst->dtype) {
        Reject(route, "GM -> L1 copy must preserve the element type");
      }
      if (copy->transpose != 0) {
        Reject(route, "GM -> L1 DN-to-NZ transpose is unsupported; express the "
                      "transpose using a supported L1 -> L0 load");
      }
      return;
    }
    if (IsL1Buffer(src) && (IsL0ABuffer(dst) || IsL0BBuffer(dst))) {
      if (src->dtype != dst->dtype) {
        Reject(route, "L1 -> L0 copy must preserve the element type");
      }
      if (!(src->dtype.is_float16() || src->dtype.is_bfloat16()))
        Reject(route,
               "the implemented native LoadData2D path requires FP16 or BF16");
      return;
    }
    if (IsL0CBuffer(src) && IsGlobalBuffer(dst)) {
      if (src->dtype != DataType::Float(32) ||
          !(dst->dtype.is_float16() || dst->dtype.is_bfloat16() ||
            dst->dtype == DataType::Float(32))) {
        Reject(route, "the implemented Fixpipe path requires FP32 accumulation "
                      "and FP16, BF16, or FP32 output");
      }
      if (copy->dual_dst_ctl != 0) {
        Reject(route, "dual-destination UB control is unsupported");
      }
      return;
    }
    Reject(route, "the native Cube path supports GM -> L1, L1 -> L0A/L0B, and "
                  "L0C -> GM copies");
  }

  static void ValidateGemm(const Gemm &gemm) {
    ValidateBuffer(gemm->a_);
    ValidateBuffer(gemm->b_);
    ValidateBuffer(gemm->c_);
    const DataType input = gemm->a_->dtype;
    if (input != gemm->b_->dtype ||
        !(input.is_float16() || input.is_bfloat16()) ||
        gemm->c_->dtype != DataType::Float(32)) {
      Reject("GEMM precision",
             "the implemented native path requires matching FP16 or BF16 "
             "operands and FP32 accumulation");
    }
    if (!IsL0CBuffer(gemm->c_) ||
        !((IsL1Buffer(gemm->a_) && IsL1Buffer(gemm->b_)) ||
          (IsL0ABuffer(gemm->a_) && IsL0BBuffer(gemm->b_)))) {
      Reject("GEMM storage",
             "requires L1 operand tiles or L0A/L0B operands, with an L0C "
             "accumulator");
    }
    if (IsL1Buffer(gemm->a_)) {
      if (gemm->transA_)
        Reject("L1 GEMM",
               "the implemented native expansion requires "
               "transpose_A=False (NN and NT B operands are supported)");
      if (gemm->m_ <= 0 || gemm->n_ <= 0 || gemm->k_ <= 0)
        Reject("L1 GEMM",
               "native sub-K expansion requires positive static M/N/K");
      // AscendLowerL1Gemm exposes its source regions, compact L0 buffers,
      // copies and MADs after explicit unroll and before layout/scheduling.
    }
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF") {
      Reject(std::string(op->name_hint), "VF/RegBase execution is unsupported");
    }
    for (const Buffer &buffer : op->alloc_buffers)
      ValidateBuffer(buffer);
    for (const MatchBufferRegion &match : op->match_buffers)
      ValidateBuffer(match->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    ValidateBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    ValidateBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final { StmtExprVisitor::VisitStmt_(op); }

  void VisitExpr_(const CallNode *op) final {
    ValidateDType(op->dtype, "call result");
    const auto callee = op->op.as<Op>();
    if (callee) {
      const std::string &name = callee.value()->name;
      if (name.rfind("tl.simd.", 0) == 0) {
        Reject(name, "VF/RegBase instructions are unsupported");
      }
      if (name == "tl.tileop.gemm_blockscaled" || name == "tl.ascend_mad_mx" ||
          name == "tl.ascend_blockscaled_gemm_l1" ||
          name == "tl.ascend_load_ca_sf" || name == "tl.ascend_load_cb_sf") {
        Reject(name, "block-scaled MX GEMM is unsupported");
      }
      if (IsAscendCopyCall(op)) {
        ValidateCopy(AscendCopy(op->args, op->annotations));
      } else if (name == "tl.tileop.gemm") {
        ValidateGemm(Gemm(op->args, op->annotations));
      } else if (name == "tl.tileop.fill") {
        Fill fill(op->args, op->annotations);
        if (!IsL1Buffer(fill->dst))
          ValidateVectorBuffer(fill->dst, "T.fill");
      } else if (name == "tl.ascend_copy_matrix_cc_to_ub") {
        Reject(name, "direct L0C -> UB Fixpipe is unsupported");
      } else if (name == "tl.ascend_copy_ubuf_to_cbuf" ||
                 name == "tl.ascend_nd2nz_post_copy") {
        Reject(name, "UB -> L1 requires a GM handoff");
      } else if (name == "tl.tileop.finalize_reducer" ||
                 name == "tl.ascend_nd2nz_scatter") {
        RequireMemBase(name);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    ValidateBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    ValidateBuffer(op->buffer);
    if (IsAscendOnChipBuffer(op->buffer))
      ValidateVectorBuffer(op->buffer, "element-wise on-chip BufferStore");
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CastNode *op) final {
    ValidateDType(op->dtype, "cast");
    StmtExprVisitor::VisitExpr_(op);
  }
};

} // namespace

tvm::transform::Pass AscendValidateTarget() {
  auto pass_func = [](PrimFunc func, IRModule, PassContext) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.has_value()) {
      TVM_FFI_THROW(ValueError)
          << "AscendValidateTarget requires a bound target";
    }
    if (GetAscendTargetInfo(target.value()).architecture ==
        AscendArchitecture::kDav2201) {
      Dav2201Validator().Validate(func);
    }
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendValidateTarget", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendValidateTarget",
                        AscendValidateTarget);
}

} // namespace tl
} // namespace tvm
