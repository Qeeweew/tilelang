/*!
 * \file tl/ascend/op/copy.cc
 * \brief Ascend implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"

#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/builtin.h"
#include "ascend/target_utils.h"
#include "ascend_mte_plan.h"
#include "backend/common/target_utils.h"
#include "layout/layout.h"
#include "oob_padding.h"
#include "op/utils.h"
#include "support/check.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

namespace {

bool IsDMACopy(const CopyNode &op) {
  return GetDMAPath(op.src, op.dst) != DMAPath::kNone;
}

bool IsInsideSimtVF(const LowerArgs &T) {
  // A SimtVF region binds a real threadIdx Var named "simtvf*"; other contexts
  // pass either a different thread Var or the constant-0 logical index.
  const auto *var = T.thread_index.as<VarNode>();
  if (!var) {
    return false;
  }
  return std::string(var->name_hint).rfind("simtvf", 0) == 0;
}

void CollectFragmentLayouts(const PrimExpr &expr,
                            const Map<Var, PrimExpr> &bind_var_to_expr,
                            const LayoutMap &existing_layouts,
                            PrimExpr thread_extent, Range thread_bounds,
                            Map<Buffer, Layout> &result_map) {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = GetRef<Var>(var_node);
      if (bind_var_to_expr.count(var)) {
        CollectFragmentLayouts(bind_var_to_expr[var], bind_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

// All callers use positive hardware units. Quotient/remainder rounding
// preserves the index dtype without overflowing value+unit-1 first.
PrimExpr CeilDivCopyCount(const PrimExpr &value, const PrimExpr &unit) {
  PrimExpr factor = cast(value.dtype(), unit);
  return FloorDiv(value, factor) +
         cast(value.dtype(),
              FloorMod(value, factor) != make_zero(value.dtype()));
}

PrimExpr Float4StorageBytes(const PrimExpr &elements, DataType dtype) {
  ICHECK(dtype.is_float4_e2m1fn());
  int element_bits = dtype.bits() * dtype.lanes();
  ICHECK(element_bits % 8 == 0 || 8 % element_bits == 0)
      << "FP4 element width must be byte-packable, got " << dtype;
  return FloorDiv(elements * make_const(elements.dtype(), element_bits),
                  make_const(elements.dtype(), 8));
}

DataType GetBufferStorageDType(const Buffer &buf) {
  const auto *ptr_type = buf->data->type_annotation.as<PointerTypeNode>();
  if (!ptr_type)
    return buf->dtype;
  const auto *prim_type = ptr_type->element_type.as<PrimTypeNode>();
  return prim_type ? prim_type->dtype : buf->dtype;
}

// T.view may give byte-addressable storage a logical FP4 dtype without
// changing its data Var. Genuine FP4 storage is rewritten later by
// RewriteFp4ToFp4x2; normalize byte-backed aliases before issuing the DMA.
PrimExpr NormalizeFloat4ByteViewPtr(const Buffer &buf,
                                    const PrimExpr &fallback_ptr,
                                    const PrimExpr &logical_offset,
                                    const PrimExpr &logical_extent) {
  DataType storage_dtype = GetBufferStorageDType(buf);
  if (!buf->dtype.is_float4_e2m1fn() || storage_dtype.is_float4_e2m1fn()) {
    return fallback_ptr;
  }

  int storage_bits = storage_dtype.bits() * storage_dtype.lanes();
  ICHECK_EQ(storage_bits % 8, 0)
      << "FP4 view requires byte-addressable storage, got " << storage_dtype;
  return Call(DataType::Handle(), builtin::tvm_access_ptr(),
              {TypeAnnotation(DataType::Int(8)), buf->data,
               Float4StorageBytes(logical_offset, buf->dtype),
               Float4StorageBytes(logical_extent, buf->dtype),
               make_const(DataType::Int(32), 1)});
}

// Map float32 source (L0C) + destination dtype → quant_pre hardware mode.
static int GetCCQuantPre(DataType src_dtype, DataType dst_dtype) {
  if (src_dtype.is_float() && src_dtype.bits() == 32) {
    if (dst_dtype.is_bfloat16())
      return 16; // F322BF16
    if (dst_dtype.is_float16())
      return 1; // F322F16
    if (dst_dtype.is_float() && dst_dtype.bits() == 8)
      return 13; // QF322FP8_PRE
    if (dst_dtype.is_int() && dst_dtype.bits() == 8)
      return 9; // REQ8
    if (dst_dtype.is_int() && dst_dtype.bits() == 4)
      return 22; // REQ4
    if (dst_dtype.is_int() && dst_dtype.bits() == 16)
      return 28; // DEQS16
  }
  return 0; // NoQuant
}

void CheckCompactL0Region(const Buffer &buffer, const Array<Range> &ranges,
                          arith::Analyzer *analyzer,
                          const std::string &description) {
  ICHECK_GE(ranges.size(), 2U)
      << description << " requires a rank >= 2 L0 region for buffer "
      << buffer->name << ", got ranges " << ranges;
  for (size_t i = ranges.size() - 2; i < ranges.size(); ++i) {
    const Range &range = ranges[i];
    PrimExpr zero = make_const(range->min.dtype(), 0);
    ICHECK(analyzer->CanProveEqual(range->min, zero))
        << description
        << " requires zero-origin trailing matrix dimensions; buffer "
        << buffer->name << " has range " << range << " on axis " << i
        << ". Split/non-zero-origin L0 tiles are not supported.";
  }
}

PrimExpr MakeCopyHasDataPredicate(const Array<Range> &src_ranges,
                                  const Array<Range> &dst_ranges,
                                  arith::Analyzer *analyzer) {
  // OOB clamping represents every reason to skip a DMA as a zero semantic
  // extent. This includes both an empty dynamic tile and an out-of-range
  // fixed/singleton-axis index. Reconstruct one execution predicate here so
  // neither condition becomes control flow before AutoSchedule.
  PrimExpr has_data = const_true();
  std::vector<PrimExpr> guarded_predicates;
  auto add_ranges = [&](const Array<Range> &ranges) {
    for (const Range &range : ranges) {
      PrimExpr extent = analyzer->Simplify(range->extent);
      PrimExpr zero = make_zero(extent.dtype());
      PrimExpr predicate = analyzer->Simplify(extent > zero);
      if (analyzer->CanProve(predicate, arith::ProofStrength::kSymbolicBound)) {
        continue;
      }

      bool duplicate = false;
      for (const PrimExpr &guarded_predicate : guarded_predicates) {
        if (analyzer->CanProveEqual(predicate, guarded_predicate)) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) {
        continue;
      }

      guarded_predicates.push_back(predicate);
      has_data = analyzer->Simplify(has_data && predicate);
    }
  };
  add_ranges(src_ranges);
  add_ranges(dst_ranges);
  return analyzer->Simplify(has_data);
}

Stmt LowerDMACopy(const AscendCopyNode &op, const LowerArgs &T,
                  arith::Analyzer *analyzer, DMAPath dma_path) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  int elem_bits = op.src->dtype.bits() * op.src->dtype.lanes();

  // AscendInsertOOBPadding has already clamped GM-facing regions while
  // preserving possibly-zero semantic extents. Derive both the empty-tile and
  // fixed-axis in-bounds guards only now, after scheduling and multi-buffer
  // analysis. Once the predicate is captured, normalize a dynamic 0/1 unit
  // axis to one solely for building legal DMA geometry; the final guard
  // preserves the original semantics.
  Array<Range> input_src = op.src_range, input_dst = op.dst_range;
  if ((dma_path == DMAPath::kL1ToL0A || dma_path == DMAPath::kL1ToL0B) &&
      GetAscendTargetInfo(T.target).architecture ==
          AscendArchitecture::kDav2201) {
    // Direct LowerTileOp callers use the same proof as the pre-schedule pass;
    // ordinary compilation already carries these exact instruction regions.
    std::tie(input_src, input_dst) =
        Normalize2201L1ToL0Footprint(op, T.layout_map, analyzer);
  }
  PrimExpr has_data = MakeCopyHasDataPredicate(input_src, input_dst, analyzer);
  if (is_zero(has_data)) {
    return Evaluate(0);
  }
  Array<Range> src_range = NormalizeEmptyUnitAxesForMTE(input_src, analyzer);
  Array<Range> dst_range = NormalizeEmptyUnitAxesForMTE(input_dst, analyzer);

  auto make_access_ptr = [&](const Buffer &buf, PrimExpr offset,
                             PrimExpr extent, int rw_mask) -> PrimExpr {
    return Call(
        DataType::Handle(), builtin::tvm_access_ptr(),
        {TypeAnnotation(buf->dtype), buf->data, offset, extent, I(rw_mask)});
  };
  auto region_elements = [&](const Array<Range> &ranges) {
    PrimExpr result = I(1);
    for (const Range &range : ranges) {
      result = result * range->extent;
    }
    return result;
  };
  StridedLayout src_region =
      StridedLayout::FromBufferRange(op.src, src_range, elem_bits);
  StridedLayout dst_region = StridedLayout::FromBufferRange(
      op.dst, dst_range, op.dst->dtype.bits() * op.dst->dtype.lanes());
  PrimExpr src_region_elements = region_elements(src_range);
  PrimExpr dst_region_elements = region_elements(dst_range);
  PrimExpr src_ptr =
      make_access_ptr(op.src, src_region.offset, src_region_elements, 1);
  PrimExpr dst_ptr =
      make_access_ptr(op.dst, dst_region.offset, dst_region_elements, 2);

  PrimExpr sid = I(0);
  PrimExpr zero = I(0);
  PrimExpr l2_cache_ctrl = I(op.L2CacheCtrlOr(0));

  PrimExpr call;
  if (dma_path == DMAPath::kGMToUB || dma_path == DMAPath::kUBToGM) {
    MTECopy2D plan = PlanMTECopy(src_region, dst_region, analyzer);

    PrimExpr plan_row_bytes =
        AscendMTEBytesFromElements(plan.row_elems, elem_bits);
    PrimExpr plan_src_stride_bytes =
        AscendMTEBytesFromElements(plan.src_stride_elems, elem_bits);
    PrimExpr plan_dst_stride_bytes =
        AscendMTEBytesFromElements(plan.dst_stride_elems, elem_bits);

    PrimExpr plan_total_elems = plan.row_elems * plan.n_rows;
    PrimExpr plan_src_ptr =
        make_access_ptr(op.src, plan.src_offset, plan_total_elems, 1);
    PrimExpr plan_dst_ptr =
        make_access_ptr(op.dst, plan.dst_offset, plan_total_elems, 2);
    PrimExpr plan_total_bytes = plan_row_bytes * plan.n_rows;

    // Ascend GM->UB padding (data_select): an unaligned row is right-padded up
    // to the next 32B boundary and the pad lanes are filled from the pad-value
    // register. The register must have been set by a leading
    // ascend_set_copy_pad_value op.
    auto as_int_plan = [](PrimExpr e) -> int64_t {
      if (auto *imm = e.as<IntImmNode>()) {
        return imm->value;
      }
      return -1;
    };
    int64_t rb_plan = as_int_plan(plan_row_bytes);
    bool data_select = dma_path == DMAPath::kGMToUB && op.data_select != 0;
    // Padding right-pads each row up to the next 32B boundary, so it needs a
    // statically-known row byte length to compute the pad count. A symbolic
    // row extent would silently skip padding and read garbage tail lanes.
    ICHECK(!data_select || rb_plan > 0)
        << "Ascend padded GM->UB copy (pad_value / data_select) requires a "
           "statically-known row byte length, got a symbolic row extent.";
    int64_t right_pad_elems = 0;
    if (data_select && rb_plan > 0 && (rb_plan % 32) != 0) {
      int64_t aligned_bytes = (rb_plan + 31) / 32 * 32;
      right_pad_elems = ((aligned_bytes - rb_plan) * 8) / elem_bits;
    }
    bool do_pad = right_pad_elems > 0;

    // Collapse into a single contiguous burst only when the region is truly
    // contiguous on both sides: one row, or each side's inter-row stride equals
    // the row byte length (consecutive rows back-to-back). A small row
    // (row_bytes < 32) is NOT sufficient: strided rows must still be issued as
    // separate bursts, otherwise rows 2+ are silently dropped. The align_v2 MTE
    // intrinsics support byte-granular bursts, so sub-32B rows are valid in the
    // multi-row path.
    bool plan_single_row =
        analyzer->CanProveEqual(plan.n_rows, I(1)) ||
        (analyzer->CanProveEqual(plan_src_stride_bytes, plan_row_bytes) &&
         analyzer->CanProveEqual(plan_dst_stride_bytes, plan_row_bytes));

    if (GetAscendTargetInfo(T.target).architecture ==
        AscendArchitecture::kDav2201) {
      With<arith::ConstraintContext> nonempty_copy(analyzer, has_data);
      const bool load = dma_path == DMAPath::kGMToUB;
      PrimExpr ub_offset_bytes = AscendMTEBytesFromElements(
          load ? plan.dst_offset : plan.src_offset, elem_bits);
      TVM_FFI_CHECK(analyzer->CanProveEqual(
                        FloorMod(ub_offset_bytes,
                                 make_const(ub_offset_bytes.dtype(), 32)),
                        make_zero(ub_offset_bytes.dtype())),
                    ValueError)
          << "dav-2201 GM <-> UB copy requires a 32-byte-aligned UB origin, "
             "got byte offset "
          << ub_offset_bytes;
      // One burst can carry a contiguous tile of any rank. Multiple native
      // bursts each advance the UB side by its rounded 32-byte footprint.
      const bool one_burst = analyzer->CanProveEqual(plan.n_rows, I(1)) ||
                             (plan_single_row && !do_pad);
      if (!one_burst) {
        TVM_FFI_CHECK(analyzer->CanProve(plan.n_rows <= I(4095)), ValueError)
            << "dav-2201 aligned DMA burst count must fit its 12-bit field, "
               "got " << plan.n_rows;
        PrimExpr ub_pitch = load ? plan_dst_stride_bytes : plan_src_stride_bytes;
        PrimExpr gm_pitch = load ? plan_src_stride_bytes : plan_dst_stride_bytes;
        PrimExpr padded_row_bytes = plan_row_bytes +
            I(load ? right_pad_elems * elem_bits / 8 : 0);
        PrimExpr ub_footprint =
            CeilDivCopyCount(padded_row_bytes,
                             make_const(padded_row_bytes.dtype(), 32)) *
            make_const(padded_row_bytes.dtype(), 32);
        TVM_FFI_CHECK(
            analyzer->CanProve(ub_pitch >= ub_footprint) &&
                analyzer->CanProveEqual(
                    FloorMod(ub_pitch, make_const(ub_pitch.dtype(), 32)),
                    make_zero(ub_pitch.dtype())),
            ValueError)
            << "dav-2201 multi-row DMA requires a 32-byte-aligned UB row "
               "pitch covering the rounded burst footprint, got pitch "
            << ub_pitch << " and footprint " << ub_footprint;
        TVM_FFI_CHECK(analyzer->CanProve(gm_pitch >= plan_row_bytes), ValueError)
            << "dav-2201 multi-row DMA requires a nonnegative GM byte gap";
      }
    }

    if (dma_path == DMAPath::kGMToUB) {
      if (do_pad) {
        // Padded copy: rightPadding fills the row tail from the pad-value
        // register (asc_set_copy_pad_val). The destination stride is the
        // physical UB row stride (plan_dst_stride_bytes), NOT the padded row
        // width: the dst buffer may be over-allocated wider than align32(row),
        // and rows must land at their real stride to avoid clobbering each
        // other.
        call =
            Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                 {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows, plan_row_bytes,
                  zero, I(right_pad_elems), I(1), l2_cache_ctrl,
                  plan_src_stride_bytes, plan_dst_stride_bytes});
      } else if (plan_single_row) {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, I(1), plan_total_bytes,
                     zero, zero, zero, l2_cache_ctrl, plan_total_bytes,
                     plan_total_bytes});
      } else {
        call = Call(DataType::Void(), ascend_copy_gm_to_ubuf(),
                    {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows,
                     plan_row_bytes, zero, zero, zero, l2_cache_ctrl,
                     plan_src_stride_bytes, plan_dst_stride_bytes});
      }
    } else {
      constexpr int kStoreDefaultL2CacheCtrl = 4;
      PrimExpr store_l2_cache_ctrl =
          I(op.L2CacheCtrlOr(kStoreDefaultL2CacheCtrl));
      if (plan_single_row) {
        call = Call(DataType::Void(), ascend_copy_ubuf_to_gm(),
                    {plan_dst_ptr, plan_src_ptr, sid, I(1), plan_total_bytes,
                     store_l2_cache_ctrl, plan_total_bytes, plan_total_bytes});
      } else {
        call = Call(DataType::Void(), ascend_copy_ubuf_to_gm(),
                    {plan_dst_ptr, plan_src_ptr, sid, plan.n_rows,
                     plan_row_bytes, store_l2_cache_ctrl, plan_dst_stride_bytes,
                     plan_src_stride_bytes});
      }
    }
  } else if (dma_path == DMAPath::kGMToL1) {
    StridedLayout src_layout =
        NormalizeMTE2DLayout(op.src, src_range, analyzer, "Ascend GM->L1 copy");
    StridedLayout dst_region_layout = NormalizeTrailingMTE2DLayout(
        op.dst, dst_range, analyzer, "Ascend GM->L1 destination");
    const Mode &src_inner = src_layout.modes[0];
    const Mode &src_row = src_layout.modes[1];
    PrimExpr src_row_stride_bytes = AscendMTEBytesFromElements(
        cast(DataType::Int(32), src_row.stride), elem_bits);

    AscendFractalLayoutInfo dst_layout;
    bool has_dst_layout = false;
    if (auto layout = FindLayoutForBuffer(T.layout_map, op.dst)) {
      has_dst_layout =
          TryExtractAscendFractalLayout(layout.value(), op.dst, &dst_layout);
      ICHECK(has_dst_layout)
          << "GM->L1 copy expects an Ascend fractal layout on the L1 dst "
          << op.dst->name;
    }
    bool needs_transpose = op.transpose != 0;
    PrimExpr dst_n_value;
    if (has_dst_layout) {
      if (dst_layout.kind == AscendFractalKind::kSF) {
        needs_transpose = !needs_transpose;
        dst_n_value = dst_layout.outer1;
      } else {
        dst_n_value = dst_layout.outer1 * dst_layout.row_frac;
      }
    } else {
      dst_n_value = dst_region_layout.modes[1].size;
    }

    PrimExpr valid_inner_elems = src_inner.size;
    PrimExpr valid_n_rows = src_row.size;
    const Mode &dst_inner = dst_region_layout.modes[0];
    const Mode &dst_row = dst_region_layout.modes[1];
    // The source region defines the DMA transfer geometry, while the
    // destination may be over-allocated to provide a larger NZ pitch. kSF
    // flips the hardware transpose bit for its packed physical layout; logical
    // source/destination axis matching still follows the user request.
    bool geometry_transpose = op.transpose != 0;
    bool source_exceeds_destination =
        geometry_transpose
            ? analyzer->CanProve(src_row.size > dst_inner.size) ||
                  analyzer->CanProve(src_inner.size > dst_row.size)
            : analyzer->CanProve(src_row.size > dst_row.size) ||
                  analyzer->CanProve(src_inner.size > dst_inner.size);
    ICHECK(!source_exceeds_destination)
        << "Ascend GM->L1 copy requires the mapped 2D source region to fit "
           "inside the destination (with axes reversed for transpose), got "
           "source "
        << src_row.size << "x" << src_inner.size << " and destination "
        << dst_row.size << "x" << dst_inner.size
        << ", transpose=" << geometry_transpose << ".";

    PrimExpr physical_inner_elems = valid_inner_elems;
    std::string physical_dtype;
    if (op.src->dtype.is_float4_e2m1fn() && op.dst->dtype.is_float4_e2m1fn()) {
      // dav-3510 exposes GM->L1 ND2NZ for 8/16/32-bit storage types.  FP4 is
      // physically packed as two values per byte, so issue the DMA as int8
      // and express the inner dimension in packed-byte units.
      physical_inner_elems =
          Float4StorageBytes(valid_inner_elems, op.src->dtype);
      physical_dtype = "int8_t";
      src_ptr = NormalizeFloat4ByteViewPtr(op.src, src_ptr, src_region.offset,
                                           src_region_elements);
    }

    PrimExpr transpose_flag = I(needs_transpose ? 1 : 0);
    if (GetAscendTargetInfo(T.target).architecture ==
        AscendArchitecture::kDav2201) {
      TVM_FFI_CHECK(analyzer->CanProve(src_row.stride <= I(65535)), ValueError)
          << "dav-2201 ND2NZ source row stride must fit uint16 elements, got "
          << src_row.stride;
    }
    if (GetAscendTargetInfo(T.target).architecture ==
            AscendArchitecture::kDav2201 && needs_transpose) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 GM -> L1 DN-to-NZ transpose is unsupported; "
             "keep L1 in native NZ order and transpose during L1 -> L0";
    }
    // For dn2nz (transpose), SDK expects nValue=width(cols) and
    // dValue=height(rows), which is the reverse of nd2nz.
    PrimExpr n_value = needs_transpose ? physical_inner_elems : valid_n_rows;
    PrimExpr d_value = needs_transpose ? valid_n_rows : physical_inner_elems;
    if (has_dst_layout && dst_layout.kind == AscendFractalKind::kSF &&
        op.src->dtype.bits() == 8 && op.dst->dtype.bits() == 8) {
      physical_dtype = "uint16_t";
      n_value = n_value / I(2);
    }
    call = Call(DataType::Void(), ascend_copy_gm_to_cbuf(),
                {dst_ptr, src_ptr, sid, src_row_stride_bytes, l2_cache_ctrl,
                 n_value, d_value, zero, zero, transpose_flag, dst_n_value,
                 StringImm(physical_dtype)});
    // OOB clamping and the padding fill for a padded GM->L1 copy are handled
    // before lowering by AscendInsertOOBPadding. The common epilogue below
    // applies the empty-tile guard after constructing this DMA call.

  } else if (dma_path == DMAPath::kL1ToL0A || dma_path == DMAPath::kL1ToL0B) {
    NormalizeTrailingMTE2DLayout(op.src, src_range, analyzer,
                                 "Ascend L1->L0 source");
    NormalizeTrailingMTE2DLayout(op.dst, dst_range, analyzer,
                                 "Ascend L1->L0 destination");
    CheckCompactL0Region(op.dst, dst_range, analyzer,
                         "Ascend L1->L0 compact destination");
    // M/K positions go through intrinsic args; leading dims (e.g. version)
    // through the pointer offset.
    PrimExpr src_base_ptr =
        MakeAscendLeadingDimAccessPtr(BufferRegion(op.src, src_range), 1);

    AscendFractalLayoutInfo src_info;
    AscendFractalLayoutInfo dst_info;
    Optional<Layout> src_layout = FindLayoutForBuffer(T.layout_map, op.src);
    Optional<Layout> dst_layout = FindLayoutForBuffer(T.layout_map, op.dst);
    bool have_src_layout =
        src_layout.defined() &&
        TryExtractAscendFractalLayout(src_layout.value(), op.src, &src_info);
    bool have_dst_layout =
        dst_layout.defined() &&
        TryExtractAscendFractalLayout(dst_layout.value(), op.dst, &dst_info);

    PrimExpr m_start, k_start, m_step, k_step, src_stride, dst_stride;

    ICHECK(have_src_layout && have_dst_layout)
        << "Ascend L1->L0 compact-region lowering requires canonical fractal "
           "layouts on both buffers, got source "
        << op.src->name << " layout=" << src_layout.defined()
        << " and destination " << op.dst->name
        << " layout=" << dst_layout.defined() << ".";
    AscendFractalRegionInfo src_region;
    AscendFractalRegionInfo dst_region;
    ICHECK(TryExtractAscendFractalRegion(src_layout.value(), src_range,
                                         &src_region))
        << "Failed to map L1->L0 source region through layout for buffer "
        << op.src->name;
    ICHECK(TryExtractAscendFractalRegion(dst_layout.value(), dst_range,
                                         &dst_region))
        << "Failed to map L1->L0 destination region through layout for buffer "
        << op.dst->name;
    m_start = src_region.outer1->min;
    k_start = src_region.outer0->min;
    m_step = src_region.outer1->extent;
    k_step = src_region.outer0->extent;
    src_stride = src_info.outer1;
    // L0 regions are compact MAD tiles: the destination pitch is the mapped
    // region pitch, while the L1 source keeps its allocation-wide NZ pitch.
    dst_stride = dst_region.outer1->extent;

    // The user may request an explicit transpose via T.copy(..,
    // transpose=True).  Additionally, when the L1 source and the L0
    // destination carry different fractal majors (e.g. a K-major L1 tile
    // feeding an MN-major L0 fractal, or vice versa), an implicit transpose is
    // required to reconcile the two layouts.  The effective transpose is the
    // XOR of the user request and the major mismatch: two transposes cancel.
    //
    // L1 tiles are canonically K-major (reduce-K on the C0 axis); when the L1
    // source carries no explicit layout we assume K-major so the mismatch is
    // driven purely by the inferred L0 major.
    bool user_transpose = op.transpose != 0;
    LogicalAxis src_c0_axis =
        have_src_layout ? src_info.c0_axis : LogicalAxis::kCol;
    LogicalAxis dst_c0_axis =
        have_dst_layout ? dst_info.c0_axis : LogicalAxis::kCol;
    bool major_mismatch = src_c0_axis != dst_c0_axis;
    bool needs_transpose = user_transpose ^ major_mismatch;

    if (needs_transpose) {
      // Validate the emitted load after storage normalization and OOB
      // clamping. Logical extents need not cover whole fractals (e.g. bf16
      // MN=24), but the hardware's transpose groups must be complete.
      int c0 = AscendC0(op.src->dtype.bits());
      auto require_multiple = [&](const PrimExpr &value, int multiple,
                                  const char *parameter) {
        if (multiple == 1)
          return;
        // Empty copies never issue the instruction. Scope this assumption
        // to the proof so the runtime has_data guard is preserved below.
        With<arith::ConstraintContext> nonempty_copy(analyzer, has_data);
        ICHECK(analyzer->CanProveEqual(
            FloorMod(value, make_const(value.dtype(), multiple)),
            make_zero(value.dtype())))
            << "Ascend transposed L1->L0 copy of " << op.src->name << " ("
            << op.src->dtype << ") requires " << parameter
            << " to be divisible by " << multiple << ", got " << value
            << ". Copy a padded source region covering complete transpose "
               "groups; restrict the GEMM region to the effective M/N/K.";
      };
      require_multiple(m_step, c0 > 16 ? c0 / 16 : 1, "m_step");
      require_multiple(k_step, c0 < 16 ? 16 / c0 : 1, "k_step");
      size_t ndim = src_range.size();
      require_multiple(
          src_range[ndim - 2 + (src_info.row16_axis == LogicalAxis::kCol)]->min,
          16, "source row16 origin");
      require_multiple(
          src_range[ndim - 2 + (src_info.c0_axis == LogicalAxis::kCol)]->min,
          c0, "source C0 origin");

      // Transposition exchanges the source's row16 and C0 dimensions.
      // Compare physical regions, not logical extents or allocation bytes:
      // the declared destination also describes the copy's write effects.
      PrimExpr written_outer0 =
          FloorDiv(m_step * 16, make_const(m_step.dtype(), c0));
      PrimExpr written_outer1 =
          FloorDiv(k_step * c0, make_const(k_step.dtype(), 16));
      ICHECK(!analyzer->CanProve(written_outer0 > dst_region.outer0->extent) &&
             !analyzer->CanProve(written_outer1 > dst_region.outer1->extent))
          << "Ascend transposed L1->L0 copy writes " << written_outer0 << "x"
          << written_outer1 << " fractals, but destination region "
          << op.dst->name << " only covers " << dst_region.outer0->extent << "x"
          << dst_region.outer1->extent
          << ". Pad the L0 allocation and copy destination region to cover "
             "the physical write; use the effective K only in T.gemm.";
      ICHECK(!analyzer->CanProve(written_outer0 > dst_info.outer0) &&
             !analyzer->CanProve(written_outer1 > dst_info.outer1))
          << "Ascend transposed L1->L0 copy exceeds the physical L0 allocation "
          << op.dst->name;
    }

    if (GetAscendTargetInfo(T.target).architecture ==
        AscendArchitecture::kDav2201) {
      const bool is_l0a = dma_path == DMAPath::kL1ToL0A;
      PrimExpr repeat =
          is_l0a ? dst_region.outer0->extent : dst_region.outer1->extent;
      PrimExpr loop_count =
          is_l0a ? dst_region.outer1->extent : dst_region.outer0->extent;
      const bool loop_source_rows = is_l0a != needs_transpose;
      size_t src_ndim = src_range.size();
      size_t source_row_axis =
          src_ndim - (src_info.row16_axis == LogicalAxis::kRow ? 2 : 1);
      size_t source_c0_axis =
          src_ndim - (src_info.c0_axis == LogicalAxis::kRow ? 2 : 1);
      auto native =
          is_l0a ? ascend_load_cbuf_to_ca_2d() : ascend_load_cbuf_to_cb_2d();
      // Without a fractal transpose, a full physical N slice of L1 NZ is
      // already contiguous in L0B ZB order. CANN 9.1 Matmul uses one 2D load
      // for this case. Compare the padded allocation pitch, not logical N;
      // leading versions remain in the pointers and sub-K in start_index.
      PrimExpr merged_repeat =
          cast(DataType::Int(64), loop_count) * cast(DataType::Int(64), repeat);
      bool merge_b =
          !is_l0a && !needs_transpose &&
          analyzer->CanProveEqual(src_range[source_row_axis]->min, I(0)) &&
          analyzer->CanProveEqual(
              FloorMod(
                  src_range[source_c0_axis]->min,
                  cast(src_range[source_c0_axis]->min.dtype(), src_info.c0)),
              make_zero(src_range[source_c0_axis]->min.dtype())) &&
          analyzer->CanProveEqual(m_start, I(0)) &&
          analyzer->CanProveEqual(m_step, src_stride) &&
          analyzer->CanProveEqual(repeat, m_step) &&
          analyzer->CanProveEqual(loop_count, k_step) &&
          analyzer->CanProve(merged_repeat <= I(255));
      Stmt native_copy;
      if (merge_b) {
        native_copy =
            Evaluate(Call(DataType::Void(), native,
                          {dst_ptr, src_base_ptr, k_start * src_stride,
                           merged_repeat, I(1), I(0), I(0)}));
      } else {
        Var frac("l0_fractal", loop_count.dtype());
        // L0A is M-major ZA; L0B is K-major ZB. Select the source NZ fractal
        // sequence that becomes contiguous in that native destination order.
        PrimExpr start = k_start * src_stride + m_start +
                         frac * (loop_source_rows ? I(1) : src_stride);
        PrimExpr stride = loop_source_rows ? src_stride : I(1);
        Array<Range> chunk = dst_range;
        const LogicalAxis loop_axis =
            is_l0a ? dst_info.row16_axis : dst_info.c0_axis;
        size_t axis = chunk.size() - (loop_axis == LogicalAxis::kRow ? 2 : 1);
        PrimExpr fractal_extent = make_const(frac.dtype(), 16);
        chunk.Set(axis,
                  Range::FromMinExtent(frac * fractal_extent, fractal_extent));
        PrimExpr chunk_dst =
            MakeAccessPtrFromRegion(BufferRegion(op.dst, chunk), 2);
        native_copy =
            For(frac, make_zero(frac.dtype()), loop_count, ForKind::kSerial,
                Evaluate(Call(DataType::Void(), native,
                              {chunk_dst, src_base_ptr, start, repeat, stride,
                               I(0), I(needs_transpose ? 1 : 0)})));
      }
      if (!is_one(has_data) && !analyzer->CanProve(has_data))
        native_copy = IfThenElse(has_data, native_copy);
      return native_copy;
    }

    auto intrinsic = dma_path == DMAPath::kL1ToL0A ? ascend_load_cbuf_to_ca()
                                                   : ascend_load_cbuf_to_cb();
    Array<PrimExpr> ld_args = {
        dst_ptr,    src_base_ptr, m_start,
        k_start,    m_step,       k_step,
        src_stride, dst_stride,   I(needs_transpose ? 1 : 0)};
    call = Call(DataType::Void(), intrinsic, ld_args);
  } else if (dma_path == DMAPath::kL0CToUB) {
    StridedLayout src_layout = NormalizeTrailingMTE2DLayout(
        op.src, src_range, analyzer, "Ascend L0C->UB source");
    StridedLayout dst_layout = NormalizeMTE2DLayout(
        op.dst, dst_range, analyzer, "Ascend L0C->UB destination");
    const Mode &src_inner = src_layout.modes[0];
    const Mode &src_row = src_layout.modes[1];
    const Mode &dst_inner = dst_layout.modes[0];
    const Mode &dst_row = dst_layout.modes[1];
    AscendFractalLayoutInfo info;
    AscendFractalRegionInfo compact_src_region;
    PrimExpr loop_src_stride = src_row.size;
    // The copied N extent may be a runtime tail, while adjacent destination
    // rows still follow the UB buffer's physical row pitch.  These coincide
    // for a full tile but differ for a sliced L0C->UB tail copy.  DUAL_N also
    // has a destination pitch different from the L0C source width.
    PrimExpr dst_row_stride = cast(DataType::Int(32), dst_row.stride);

    CheckCompactL0Region(op.src, src_range, analyzer,
                         "Ascend L0C->UB compact source");
    Optional<Layout> compact_src_layout =
        FindLayoutForBuffer(T.layout_map, op.src);
    ICHECK(compact_src_layout.defined() &&
           TryExtractAscendFractalLayout(compact_src_layout.value(), op.src,
                                         &info))
        << "L0C->UB copy expects a canonical Ascend L0C layout on source "
        << op.src->name;
    ICHECK(TryExtractAscendFractalRegion(compact_src_layout.value(), src_range,
                                         &compact_src_region))
        << "Failed to map L0C->UB source region through layout for buffer "
        << op.src->name;
    // NZ column groups keep the allocation's padded M pitch. Clamping the
    // logical read region (for a tail or handoff) must not compact its storage.
    loop_src_stride = info.outer1 * info.row_frac;

    int dual_dst_ctl = op.dual_dst_ctl;
    PrimExpr copy_inner = src_inner.size;
    PrimExpr copy_rows = src_row.size;
    if (dual_dst_ctl == 0) {
      bool destination_exceeds_source =
          analyzer->CanProve(dst_inner.size > src_inner.size) ||
          analyzer->CanProve(dst_row.size > src_row.size);
      ICHECK(!destination_exceeds_source)
          << "Ascend L0C->UB destination region must fit inside the source "
             "tile unless dual-destination mode is enabled, got source "
          << src_row.size << "x" << src_inner.size << " and destination "
          << dst_row.size << "x" << dst_inner.size << ".";
      copy_inner = dst_inner.size;
      copy_rows = dst_row.size;
    }
    PrimExpr dual_dst_ctl_val = I(dual_dst_ctl);
    PrimExpr unit_flag_ctl_val = cast(DataType::Int(32), op.unit_flag_ctl);
    PrimExpr sub_blockid_val = cast(DataType::Int(32), op.sub_blockid);
    call = Call(DataType::Void(), ascend_copy_matrix_cc_to_ub(),
                {dst_ptr,
                 src_ptr,
                 sid,
                 copy_inner,
                 copy_rows,
                 dst_row_stride,
                 loop_src_stride,
                 dual_dst_ctl_val,
                 sub_blockid_val,   // sub_blockid
                 zero,              // clip_relu_pre
                 unit_flag_ctl_val, // unit_flag_ctl
                 I(GetCCQuantPre(op.src->dtype, op.dst->dtype)), // quant_pre
                 zero,                                           // relu_pre
                 zero,                                           // split_en
                 I(1),                                           // NZ2ND_en
                 zero,                                           // quant_post
                 zero,                                           // relu_post
                 zero,   // clip_relu_post
                 zero,   // loop_enhance_en
                 zero,   // eltwise_op
                 zero,   // eltwise_antq_en
                 zero,   // loop_enhance_merge_en
                 zero,   // C0_pad_en
                 zero,   // wino_post_en
                 zero,   // broadcast_en
                 zero}); // NZ2DN_en
  } else if (dma_path == DMAPath::kUBToL1) {
    ICHECK(!op.nd2nz)
        << "Ascend nd2nz UB->L1 copy must be emitted directly from the Python "
           "frontend (tilelang/language/copy_op.py); it should never reach "
           "tl.tileop.copy lowering. This indicates the Python bypass was not "
           "taken; check the call site.";
    ICHECK_EQ(op.src->dtype.bits(), op.dst->dtype.bits())
        << "Ascend raw UB->L1 copy cannot convert element widths, got source "
        << op.src->dtype << " and destination " << op.dst->dtype << ".";
    StridedLayout src_layout =
        StridedLayout::FromBufferRange(op.src, src_range, elem_bits)
            .Coalesce(analyzer);
    StridedLayout dst_layout =
        StridedLayout::FromBufferRange(op.dst, dst_range, op.dst->dtype.bits())
            .Coalesce(analyzer);
    ICHECK_EQ(src_layout.modes.size(), 1U)
        << "Ascend raw UB->L1 copy requires a contiguous source region; got "
        << src_layout.modes.size() << " modes after coalescing for buffer "
        << op.src->name << " shape " << op.src->shape << " ranges " << src_range
        << ". Strided UB->L1 copies are not yet supported.";
    ICHECK_EQ(dst_layout.modes.size(), 1U)
        << "Ascend raw UB->L1 copy requires a contiguous destination region; "
        << "got " << dst_layout.modes.size()
        << " modes after coalescing for buffer " << op.dst->name << " shape "
        << op.dst->shape << " ranges " << dst_range
        << ". Strided UB->L1 copies are not yet supported.";
    bool element_count_mismatch = analyzer->CanProve(src_layout.modes[0].size !=
                                                     dst_layout.modes[0].size);
    ICHECK(!element_count_mismatch)
        << "Ascend raw UB->L1 copy requires source and destination regions "
           "with the same element count, got source "
        << src_layout.modes[0].size << " and destination "
        << dst_layout.modes[0].size << ".";
    PrimExpr raw_total_bytes =
        AscendMTEBytesFromElements(src_layout.modes[0].size, elem_bits);
    TVM_FFI_CHECK(
        analyzer->CanProveEqual(
            FloorMod(raw_total_bytes, make_const(raw_total_bytes.dtype(), 32)),
            make_zero(raw_total_bytes.dtype())),
        ValueError)
        << "Ascend raw UB->L1 copy requires a payload that is a multiple of "
           "32 bytes, but got "
        << raw_total_bytes << " bytes from source buffer " << op.src->name
        << " ranges " << src_range << ".";
    PrimExpr n_burst_val =
        raw_total_bytes / make_const(raw_total_bytes.dtype(), 32);
    call = Call(DataType::Void(), ascend_copy_ubuf_to_cbuf(),
                {dst_ptr, src_ptr, zero, n_burst_val, I(1), zero, zero});
  } else if (dma_path == DMAPath::kL0CToGM) {
    StridedLayout src_layout = NormalizeTrailingMTE2DLayout(
        op.src, src_range, analyzer, "Ascend L0C->GM source");
    StridedLayout dst_layout = NormalizeMTE2DLayout(
        op.dst, dst_range, analyzer, "Ascend L0C->GM destination");
    const Mode &src_inner = src_layout.modes[0];
    const Mode &src_row = src_layout.modes[1];
    const Mode &dst_inner = dst_layout.modes[0];
    const Mode &dst_row = dst_layout.modes[1];
    AscendFractalLayoutInfo info;
    AscendFractalRegionInfo compact_src_region;
    PrimExpr loop_src_stride = src_row.size;

    CheckCompactL0Region(op.src, src_range, analyzer,
                         "Ascend L0C->GM compact source");
    Optional<Layout> compact_src_layout =
        FindLayoutForBuffer(T.layout_map, op.src);
    ICHECK(compact_src_layout.defined() &&
           TryExtractAscendFractalLayout(compact_src_layout.value(), op.src,
                                         &info))
        << "L0C->GM copy expects a canonical Ascend L0C layout on source "
        << op.src->name;
    ICHECK(TryExtractAscendFractalRegion(compact_src_layout.value(), src_range,
                                         &compact_src_region))
        << "Failed to map L0C->GM source region through layout for buffer "
        << op.src->name;
    // The copied rows describe validity, not the distance between NZ column
    // groups. A small valid-M region still lives in the full L0C allocation.
    loop_src_stride = info.outer1 * info.row_frac;
    // L0C->GM needs the physical distance between adjacent destination rows,
    // not the logical last dimension.  They are equal for compact tensors but
    // differ for padded/runtime StridedTensor outputs.
    bool destination_exceeds_source =
        analyzer->CanProve(dst_inner.size > src_inner.size) ||
        analyzer->CanProve(dst_row.size > src_row.size);
    ICHECK(!destination_exceeds_source)
        << "Ascend L0C->GM destination region must fit inside the source tile, "
           "got source "
        << src_row.size << "x" << src_inner.size << " and destination "
        << dst_row.size << "x" << dst_inner.size << ".";
    PrimExpr dst_row_stride = cast(DataType::Int(32), dst_row.stride);
    PrimExpr unit_flag_ctl_val = cast(DataType::Int(32), op.unit_flag_ctl);
    call =
        Call(DataType::Void(), ascend_copy_matrix_cc_to_gm(),
             {dst_ptr,
              src_ptr,
              sid,
              dst_inner.size,
              dst_row.size,
              dst_row_stride,
              loop_src_stride,
              I(op.L2CacheCtrlOr(0)), // [7] l2_cache_ctl
              zero,                   // [8] clip_relu_pre
              unit_flag_ctl_val,      // [9] unit_flag_ctl
              I(GetCCQuantPre(op.src->dtype, op.dst->dtype)), // [10] quant_pre
              zero,                                           // [11] relu_pre
              zero,                                           // [12] split_en
              I(1),                                           // [13] NZ2ND_en
              zero,                                           // [14] quant_post
              zero,                                           // [15] relu_post
              zero,   // [16] clip_relu_post
              zero,   // [17] loop_enhance_en
              zero,   // [18] eltwise_op
              zero,   // [19] eltwise_antq_en
              zero,   // [20] loop_enhance_merge_en
              zero,   // [21] C0_pad_en
              zero,   // [22] wino_post_en
              zero,   // [23] broadcast_en
              zero}); // [24] NZ2DN_en
  } else {
    LOG(FATAL) << "Unknown Ascend DMA path";
  }

  Stmt copy_stmt = Evaluate(call);
  if (!is_one(has_data) && !analyzer->CanProve(has_data)) {
    copy_stmt = IfThenElse(has_data, copy_stmt);
  }
  return copy_stmt;
}

// SF has separate storage but derives its address from the bound L0 tile.
// All leading axes are ordinary tile selectors, including materialized rings.
Stmt LowerMxSfLoad(const AscendCopyNode &op, const LowerArgs &T,
                   arith::Analyzer *analyzer, bool is_l0a) {
  auto I = [](int64_t v) { return make_const(DataType::Int(32), v); };
  const Buffer &sf_buf = op.src;
  const Buffer &handle = op.dst;

  PrimExpr has_data =
      MakeCopyHasDataPredicate(op.src_range, op.dst_range, analyzer);
  if (is_zero(has_data)) {
    return Evaluate(0);
  }

  Array<Range> src_range = NormalizeEmptyUnitAxesForMTE(op.src_range, analyzer);
  Array<Range> dst_range = NormalizeEmptyUnitAxesForMTE(op.dst_range, analyzer);
  ICHECK_EQ(sf_buf->dtype.bits(), handle->dtype.bits())
      << "L0 SF copy requires matching source and destination scale widths";
  ICHECK_EQ(sf_buf->dtype.lanes(), 1);
  ICHECK_EQ(handle->dtype.lanes(), 1);
  ICHECK_EQ(op.transpose, 0) << "L0 SF copy does not support transpose";
  size_t ndim = dst_range.size();
  ICHECK_GE(ndim, 2) << "Ascend MX scale-factor copy destination " << handle
                     << " must keep a trailing 2-D SF handle shape.";
  CheckCompactL0Region(handle, dst_range, analyzer,
                       "Ascend MX scale-factor copy");

  ICHECK(op.mx_sf_data.defined())
      << "L0 SF copy destination " << handle->name
      << " has no allocation binding; use alloc_l0a_sf/alloc_l0b_sf";
  const Buffer &data_buf = op.mx_sf_data.value();
  ICHECK_EQ(data_buf->shape.size(), ndim)
      << "L0 SF and bound data allocation must preserve the same leading axes";
  AscendFractalLayoutInfo data_info;
  Optional<Layout> data_layout = FindLayoutForBuffer(T.layout_map, data_buf);
  ICHECK(
      data_layout.defined() &&
      TryExtractAscendFractalLayout(data_layout.value(), data_buf, &data_info))
      << "L0 SF's bound data buffer requires a fractal layout";
  Array<Range> data_ranges;
  for (size_t i = 0; i + 2 < ndim; ++i) {
    ICHECK(analyzer->CanProve(dst_range[i]->extent <= 1))
        << "L0 SF copy must select one tile on each leading axis";
    data_ranges.push_back(dst_range[i]);
  }
  PrimExpr mn = dst_range[ndim - 2]->extent;
  PrimExpr k = dst_range[ndim - 1]->extent * I(4 * handle->dtype.bits());
  data_ranges.push_back(Range::FromMinExtent(
      I(0), data_info.c0_axis == LogicalAxis::kCol ? mn : k));
  data_ranges.push_back(Range::FromMinExtent(
      I(0), data_info.c0_axis == LogicalAxis::kCol ? k : mn));
  PrimExpr dst_ptr =
      MakeAscendLeadingDimAccessPtr(BufferRegion(data_buf, data_ranges), 2);

  NormalizeTrailingMTE2DLayout(sf_buf, src_range, analyzer,
                               "Ascend MX scale-factor source");
  PrimExpr sf_ptr =
      MakeAscendLeadingDimAccessPtr(BufferRegion(sf_buf, src_range), 1);

  AscendFractalLayoutInfo sf_info;
  Optional<Layout> sf_layout = FindLayoutForBuffer(T.layout_map, sf_buf);
  bool have_sf_layout =
      sf_layout.defined() &&
      TryExtractAscendFractalLayout(sf_layout.value(), sf_buf, &sf_info) &&
      sf_info.kind == AscendFractalKind::kSF;

  ICHECK(have_sf_layout)
      << "L0 SF copy requires the canonical SF_K layout on its L1 source";
  AscendFractalRegionInfo sf_region;
  ICHECK(
      TryExtractAscendFractalRegion(sf_layout.value(), src_range, &sf_region))
      << "Failed to map MX scale-factor source region for " << sf_buf->name;
  // Source pitch belongs to the L1 allocation; destination pitch belongs to
  // this compact tile. SF_K packs pairs of E8M0 bytes on its C0 axis.
  PrimExpr sf_m_start = sf_region.outer0->min;
  PrimExpr sf_k_start = sf_region.outer1->min;
  PrimExpr sf_m_step = sf_region.outer0->extent;
  PrimExpr sf_k_step = sf_region.outer1->extent;
  PrimExpr sf_src_stride = sf_info.outer1;
  PrimExpr sf_dst_stride =
      CeilDivCopyCount(dst_range[ndim - 1]->extent, sf_info.c0);
  {
    With<arith::ConstraintContext> nonempty_copy(analyzer, has_data);
    size_t src_ndim = src_range.size();
    ICHECK(analyzer->CanProveEqual(
               FloorMod(src_range[src_ndim - 2]->min,
                        make_const(src_range[src_ndim - 2]->min.dtype(), 16)),
               make_zero(src_range[src_ndim - 2]->min.dtype())) &&
           analyzer->CanProveEqual(
               FloorMod(src_range[src_ndim - 1]->min,
                        cast(src_range[src_ndim - 1]->min.dtype(), sf_info.c0)),
               make_zero(src_range[src_ndim - 1]->min.dtype())))
        << "L0 SF source origin must align to 16 MN rows and a scale pair";
    ICHECK(analyzer->CanProveEqual(
               sf_m_step, CeilDivCopyCount(mn, make_const(mn.dtype(), 16))) &&
           analyzer->CanProveEqual(sf_k_step, sf_dst_stride))
        << "L0 SF source and compact destination region must cover the same "
           "physical scale groups";
    for (size_t i = ndim - 2; i < ndim; ++i) {
      ICHECK(analyzer->CanProve(data_ranges[i]->extent <= data_buf->shape[i]))
          << "L0 SF copy must fit its bound data allocation " << data_buf->name;
    }
  }

  Stmt stmt = Evaluate(Call(DataType::Void(),
                            is_l0a ? ascend_load_ca_sf() : ascend_load_cb_sf(),
                            {dst_ptr, sf_ptr, sf_m_start, sf_k_start, sf_m_step,
                             sf_k_step, sf_src_stride, sf_dst_stride}));
  if (!is_one(has_data) && !analyzer->CanProve(has_data)) {
    stmt = IfThenElse(has_data, stmt);
  }
  return stmt;
}

Stmt LowerAscendNormalCopy(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer) {
  if (!IsInsideSimtVF(T)) {
    return LowerNormalCopy(op, T, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target, T.thread_bounds, T.layout_map, analyzer,
                         T.buffer_remap, T.bind_var_to_expr},
                        level);
  }

  auto loop_layout = par_op->GetLoopLayout();
  return LowerParallelLoop(
      par_op->GetRoot(), loop_layout, T.thread_index, analyzer, T.layout_map,
      par_op->GetPredicate(T.thread_index),
      /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());
}

} // namespace

// Implementation of the AscendCopyNode behavior. Shared between the virtual
// methods (typed nodes parsed from tl.tileop.ascend_copy) and the CopyImpl
// bridge below (base nodes parsed from plain tl.tileop.copy).
struct AscendCopyImpl {
  static LayoutMap InferLayout(const AscendCopyNode &op,
                               const LayoutInferArgs &T, InferLevel level) {
    if (IsDMACopy(op)) {
      Map<Buffer, Layout> result_map;
      PrimExpr thread_extent = T.thread_bounds->extent;
      for (const auto &range : op.src_range) {
        CollectFragmentLayouts(range->min, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
        CollectFragmentLayouts(range->extent, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
      }
      for (const auto &range : op.dst_range) {
        CollectFragmentLayouts(range->min, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
        CollectFragmentLayouts(range->extent, T.bind_var_to_expr, T.layout_map,
                               thread_extent, T.thread_bounds, result_map);
      }

      if (IsL0SFBuffer(op.dst)) {
        // The L1 source carries the SF_K fractal layout. The SF handle
        // destination never gets a layout: its addressing is fixed by the
        // ISA to the bound data tile's address.
        result_map.Set(op.src, MakeAscendSFLayout(op.src));
        return result_map;
      }

      DMAPath dma_path = GetDMAPath(op.src, op.dst);
      if (dma_path == DMAPath::kGMToL1 && level == InferLevel::kFree &&
          !FindLayoutForBuffer(T.layout_map, op.dst).defined()) {
        // A standalone GM->L1 copy has no downstream L0/GEMM consumer to
        // anchor its layout. Preserve the canonical legacy K-col NZ layout as
        // a free-mode fallback.
        result_map.Set(op.dst, MakeAscendMajorKLayout(op.dst));
      } else if (dma_path == DMAPath::kL1ToL0A ||
                 dma_path == DMAPath::kL1ToL0B) {
        Optional<Layout> existing_src_layout =
            FindLayoutForBuffer(T.layout_map, op.src);
        Layout src_layout = existing_src_layout.defined()
                                ? existing_src_layout.value()
                                : MakeAscendMajorKLayout(op.src);
        if (!existing_src_layout.defined()) {
          result_map.Set(op.src, src_layout);
        }
        AscendFractalLayoutInfo src_info;
        ICHECK(TryExtractAscendFractalLayout(src_layout, op.src, &src_info))
            << "Ascend L1->L0 copy expects a fractal layout on source L1 "
            << op.src->name;

        Optional<Layout> dst_layout = FindLayoutForBuffer(T.layout_map, op.dst);
        if (!dst_layout.defined() && level == InferLevel::kFree) {
          // A standalone L1->L0 copy has no GEMM consumer to anchor the L0
          // major. Preserve the legacy default by assigning MajorK only after
          // stricter inference levels have had a chance to contribute it.
          result_map.Set(op.dst, MakeAscendMajorKLayout(op.dst));
        }
      }

      // Layout constraints for Ascend Cube-side data buffers are attached at
      // explicit annotation or consuming-operation sites (e.g. alloc_l1(...,
      // major=...) and GemmMAD). Copy inference does not overwrite them; it
      // only supplies free-mode MajorK defaults for standalone copy endpoints,
      // fragment layouts discovered in index expressions, and SF_K layouts
      // for the sources of standalone MX scale-factor loads.
      return result_map;
    }

    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const AscendCopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    ICHECK(!IsL0SFBuffer(op.src))
        << "L0 SF handles can only be read by gemm_blockscaled";
    if (IsL0SFBuffer(op.dst)) {
      DMAPath dma_path = GetDMAPath(op.src, op.dst);
      ICHECK(dma_path == DMAPath::kL1ToL0ASF || dma_path == DMAPath::kL1ToL0BSF)
          << "Ascend MX scale-factor copy expects an L1 source and an "
             "L0A/L0B SF handle destination, got "
          << op.src.scope() << " -> " << op.dst.scope();
      return LowerMxSfLoad(op, T, analyzer, dma_path == DMAPath::kL1ToL0ASF);
    }
    if (!IsInsideSimtVF(T) && !IsLocalBuffer(op.src) &&
        !IsLocalBuffer(op.dst)) {
      DMAPath dma_path = GetDMAPath(op.src, op.dst);
      if (dma_path != DMAPath::kNone) {
        return LowerDMACopy(op, T, analyzer, dma_path);
      }
    }

    return LowerAscendNormalCopy(op, T, analyzer);
  }
};

// The public helper is shared by early IR normalization and late emission.
std::pair<Array<Range>, Array<Range>>
Normalize2201L1ToL0Footprint(const AscendCopyNode &op,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer) {
  auto I = [](int64_t value) { return make_const(DataType::Int(32), value); };
  DMAPath dma_path = GetDMAPath(op.src, op.dst);
  ICHECK(dma_path == DMAPath::kL1ToL0A || dma_path == DMAPath::kL1ToL0B);
  PrimExpr has_data =
      MakeCopyHasDataPredicate(op.src_range, op.dst_range, analyzer);
  if (is_zero(has_data))
    return {op.src_range, op.dst_range};
  const bool always_has_data = analyzer->CanProve(has_data);
  With<arith::ConstraintContext> nonempty_copy(analyzer, has_data);
  Array<Range> src_range = NormalizeEmptyUnitAxesForMTE(op.src_range, analyzer);
  Array<Range> dst_range = NormalizeEmptyUnitAxesForMTE(op.dst_range, analyzer);
  NormalizeTrailingMTE2DLayout(op.src, src_range, analyzer,
                               "Ascend L1->L0 source footprint");
  NormalizeTrailingMTE2DLayout(op.dst, dst_range, analyzer,
                               "Ascend L1->L0 destination footprint");
  CheckCompactL0Region(op.dst, dst_range, analyzer,
                       "Ascend L1->L0 compact destination");
  auto src_layout = FindLayoutForBuffer(layout_map, op.src);
  auto dst_layout = FindLayoutForBuffer(layout_map, op.dst);
  AscendFractalLayoutInfo src_info, dst_info;
  AscendFractalRegionInfo src_region, dst_region;
  ICHECK(src_layout.defined() && dst_layout.defined() &&
         TryExtractAscendFractalLayout(src_layout.value(), op.src, &src_info) &&
         TryExtractAscendFractalLayout(dst_layout.value(), op.dst, &dst_info))
      << "dav-2201 LoadData2D requires canonical fractal layouts";
  size_t src_ndim = src_range.size();
  size_t source_row_axis =
      src_ndim - (src_info.row16_axis == LogicalAxis::kRow ? 2 : 1);
  size_t source_c0_axis =
      src_ndim - (src_info.c0_axis == LogicalAxis::kRow ? 2 : 1);
  ICHECK(
      analyzer->CanProveEqual(
          FloorMod(
              src_range[source_row_axis]->min,
              cast(src_range[source_row_axis]->min.dtype(), src_info.row_frac)),
          make_zero(src_range[source_row_axis]->min.dtype())) &&
      analyzer->CanProveEqual(
          FloorMod(src_range[source_c0_axis]->min,
                   cast(src_range[source_c0_axis]->min.dtype(), src_info.c0)),
          make_zero(src_range[source_c0_axis]->min.dtype())))
      << "dav-2201 LoadData2D requires a row16/C0-aligned source origin; "
      << op.src->name << " has source region " << src_range;
  auto in_source_axis = [&](const PrimExpr &origin, const PrimExpr &extent,
                            const PrimExpr &capacity) {
    return CanProveAscendRangeInBounds(origin, extent, capacity, analyzer);
  };
  // Prove bounds before MapRegionBounds forms min+extent in the input dtype.
  // A wrapped narrow interval must never turn an out-of-allocation origin
  // into a benign-looking mapped fractal. The capacity is physical padding,
  // even for direct LowerTileOp callers whose buffer still has a logical shape.
  PrimExpr row_capacity = cast(DataType::Int(64), src_info.outer1) *
                          cast(DataType::Int(64), src_info.row_frac);
  PrimExpr c0_capacity = cast(DataType::Int(64), src_info.outer0) *
                         cast(DataType::Int(64), src_info.c0);
  ICHECK(in_source_axis(src_range[source_row_axis]->min,
                        src_range[source_row_axis]->extent, row_capacity) &&
         in_source_axis(src_range[source_c0_axis]->min,
                        src_range[source_c0_axis]->extent, c0_capacity))
      << "dav-2201 LoadData2D source footprint exceeds or cannot be proven "
         "within the physical L1 allocation "
      << op.src->name;
  ICHECK(
      TryExtractAscendFractalRegion(src_layout.value(), src_range,
                                    &src_region) &&
      TryExtractAscendFractalRegion(dst_layout.value(), dst_range, &dst_region))
      << "dav-2201 LoadData2D requires canonical fractal regions";
  PrimExpr m_start = src_region.outer1->min;
  PrimExpr k_start = src_region.outer0->min;
  PrimExpr m_step = src_region.outer1->extent;
  PrimExpr k_step = src_region.outer0->extent;
  bool needs_transpose =
      (op.transpose != 0) ^ (src_info.c0_axis != dst_info.c0_axis);
  ICHECK(op.src->dtype.is_float16() || op.src->dtype.is_bfloat16())
      << "dav-2201 native LoadData2D currently requires FP16 or BF16";
  const bool is_l0a = dma_path == DMAPath::kL1ToL0A;
  ICHECK(!is_l0a || dst_info.kind == AscendFractalKind::kL0AMajorK ||
         dst_info.kind == AscendFractalKind::kL0AMajorMN)
      << "dav-2201 L0A requires its native ZA layout";
  // A native MAD tile is compact. Larger allocations may hold separate
  // versions, but the selected trailing tile has one physical pitch.
  ICHECK(analyzer->CanProveEqual(dst_region.outer0->extent, dst_info.outer0) &&
         analyzer->CanProveEqual(dst_region.outer1->extent, dst_info.outer1))
      << "dav-2201 LoadData2D destination must cover its compact padded "
         "L0 tile; allocate a separate sub-K L0 tile";
  PrimExpr repeat =
      is_l0a ? dst_region.outer0->extent : dst_region.outer1->extent;
  PrimExpr loop_count =
      is_l0a ? dst_region.outer1->extent : dst_region.outer0->extent;
  const bool loop_source_rows = is_l0a != needs_transpose;
  PrimExpr read_rows = loop_source_rows ? loop_count : repeat;
  PrimExpr read_columns = loop_source_rows ? repeat : loop_count;
  // Hardware reads complete fractals. A larger L0 destination cannot
  // authorize additional L1 reads beyond the declared source footprint.
  ICHECK(analyzer->CanProveEqual(read_rows, m_step) &&
         analyzer->CanProveEqual(read_columns, k_step))
      << "dav-2201 LoadData2D requires matching physical source/destination "
         "fractal footprints; source "
      << op.src->name << " covers " << m_step << "x" << k_step
      << " fractals, but the compact L0 tile reads " << read_rows << "x"
      << read_columns
      << ". Materialize source padding in L1 before copying a larger tile; "
         "a larger destination does not authorize extra source reads.";
  ICHECK(in_source_axis(m_start, read_rows, src_info.outer1) &&
         in_source_axis(k_start, read_columns, src_info.outer0))
      << "dav-2201 LoadData2D source footprint exceeds or cannot be proven "
         "within the physical L1 allocation "
      << op.src->name
      << ". This L1 tail requires explicit padding in a sufficient L1 "
         "allocation before native loading.";
  for (const auto &[buffer, ranges] :
       {std::make_pair(op.src, src_range), std::make_pair(op.dst, dst_range)}) {
    for (size_t axis = 0; axis + 2 < ranges.size(); ++axis) {
      ICHECK(in_source_axis(ranges[axis]->min, ranges[axis]->extent,
                            buffer->shape[axis]))
          << "dav-2201 LoadData2D leading/version axis " << axis
          << " exceeds or cannot be proven within allocation " << buffer->name
          << ": " << ranges[axis];
    }
  }
  ICHECK(analyzer->CanProve(repeat <= I(255)))
      << "dav-2201 LoadData2D repeat count must fit uint8";
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "dav-2201 LoadData2D preserves operand dtype";
  // These regions describe the instruction, including padded lanes. They are
  // established before scheduling so padding producers and later reuse have
  // the same RAW/WAR facts as native lowering. GEMM keeps its logical domain.
  auto physical = [&](const Array<Range> &original,
                      const AscendFractalLayoutInfo &info,
                      const AscendFractalRegionInfo &mapped) {
    Array<Range> ranges = original;
    for (size_t axis = original.size() - 2; axis < original.size(); ++axis) {
      bool row16 = (axis == original.size() - 2) ==
                   (info.row16_axis == LogicalAxis::kRow);
      PrimExpr extent = row16 ? mapped.outer1->extent * info.row_frac
                              : mapped.outer0->extent * info.c0;
      // Keep an empty copy empty; a data predicate must not disappear when
      // its active instruction footprint rounds to a complete fractal.
      extent = analyzer->Simplify(extent);
      if (!always_has_data)
        extent = Select(has_data, extent, make_zero(extent.dtype()));
      ranges.Set(axis, Range::FromMinExtent(original[axis]->min, extent));
    }
    return ranges;
  };
  return {physical(op.src_range, src_info, src_region),
          physical(op.dst_range, dst_info, dst_region)};
}

} // namespace ascend

namespace {

// Decode the Ascend copy hint annotations into the typed AscendCopyNode
// fields. The annotations map on the Call stays the durable encoding; both
// AscendCopy constructors funnel through here so the two parse paths cannot
// diverge.
void DecodeAscendCopyAnnotations(AscendCopyNode *node) {
  auto int_or = [node](const char *key, int default_value) {
    if (auto val = node->annotations.Get(key)) {
      if (const auto *int_val = val->as<IntImmNode>()) {
        return static_cast<int>(int_val->value);
      }
    }
    return default_value;
  };
  node->nd2nz = int_or("nd2nz", 0);
  node->dual_dst_ctl = int_or("dual_dst_ctl", 0);
  node->transpose = int_or("transpose", 0);
  node->data_select = int_or("data_select", 0);
  node->mx_sf_data = std::nullopt;
  node->l2_cache_ctrl = std::nullopt;
  if (auto val = node->annotations.Get("l2_cache_ctrl")) {
    if (const auto *int_val = val->as<IntImmNode>()) {
      node->l2_cache_ctrl = Integer(int_val->value);
    }
  }
  node->unit_flag_ctl = IntImm(DataType::Int(32), 0);
  if (auto val = node->annotations.Get("unit_flag_ctrl")) {
    node->unit_flag_ctl = Downcast<PrimExpr>(val.value());
  }
  node->sub_blockid = IntImm(DataType::Int(32), 0);
  if (auto val = node->annotations.Get("sub_blockid")) {
    node->sub_blockid = Downcast<PrimExpr>(val.value());
  }
  node->pad_value = std::nullopt;
  if (auto val = node->annotations.Get("pad_value")) {
    node->pad_value = Downcast<PrimExpr>(val.value());
  }
}

// Mirrors ApplyCopyBlockAnnotations in src/op/copy.cc: resolve the annotated
// source OOB fallback value from the enclosing block.
TileOperator
ApplyAscendCopyBlockAnnotations(TileOperator tile_op,
                                BlockAnnotations block_annotations) {
  AscendCopy copy = Downcast<AscendCopy>(tile_op);

  // Safe because this handler is invoked immediately after TLOpBuilder creates
  // a fresh AscendCopyNode, before the node escapes ParseOperator.
  auto *node = const_cast<AscendCopyNode *>(copy.get());
  ICHECK(node != nullptr);

  node->src_oob_safe_value = PrimExpr();
  auto safe_value_map_obj = block_annotations.Get(attr::kSafeValueMap);
  if (!safe_value_map_obj) {
    return copy;
  }

  auto safe_value_map =
      Downcast<Map<Var, PrimExpr>>(safe_value_map_obj.value());
  auto it = safe_value_map.find(node->src->data);
  if (it != safe_value_map.end()) {
    node->src_oob_safe_value = (*it).second;
  }
  return copy;
}

bool MatchAscendCopyTarget(Target target) { return TargetIsAscend(target); }

// Plain tl.tileop.copy calls (e.g. copies synthesized by shared passes such
// as ReducerPlanAndMaterialize) parse into the base CopyNode and reach the
// target-dispatched CopyImpl registry; upgrade them so both spellings
// converge on the AscendCopyImpl implementation.
LayoutMap BridgeAscendCopyInferLayout(const CopyNode &op,
                                      const LayoutInferArgs &layout_args,
                                      InferLevel level) {
  if (op.IsInstance<AscendCopyNode>()) {
    return ascend::AscendCopyImpl::InferLayout(
        static_cast<const AscendCopyNode &>(op), layout_args, level);
  }
  AscendCopy upgraded(op);
  return ascend::AscendCopyImpl::InferLayout(*upgraded.get(), layout_args,
                                             level);
}

Stmt BridgeAscendCopyLower(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) {
  if (op.IsInstance<AscendCopyNode>()) {
    return ascend::AscendCopyImpl::Lower(
        static_cast<const AscendCopyNode &>(op), lower_args, analyzer);
  }
  AscendCopy upgraded(op);
  return ascend::AscendCopyImpl::Lower(*upgraded.get(), lower_args, analyzer);
}

bool RegisterAscendCopy() {
  RegisterCopyImpl(CopyImpl{
      "ascend.Copy",
      MatchAscendCopyTarget,
      200,
      BridgeAscendCopyInferLayout,
      BridgeAscendCopyLower,
  });
  return true;
}

const bool ascend_copy_registered = RegisterAscendCopy();

} // namespace

// Constructs an AscendCopy operator node from tl.tileop.ascend_copy call
// arguments and annotations.
// args[0]: source region, args[1]: destination region.
AscendCopy::AscendCopy(Array<PrimExpr> args,
                       Map<String, ObjectRef> annotations) {
  ObjectPtr<AscendCopyNode> node = make_object<AscendCopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  node->annotations = annotations;
  if (auto dst_block = node->annotations.Get("dst_block")) {
    if (auto int_imm = dst_block->as<IntImmNode>()) {
      if (int_imm->value != -1) {
        node->dst_block = Integer(int_imm->value);
      }
    } else {
      node->dst_block = Downcast<PrimExpr>(dst_block.value());
    }
  }
  DecodeAscendCopyAnnotations(node.get());
  data_ = std::move(node);
}

// Upgrades a base copy parsed from a plain tl.tileop.copy call. The base
// state (regions, access regions, block-annotation results) transfers as-is.
// MX scale-factor loads also arrive base-spelled (the dialect needs no
// special emission for them); their dedicated destination scope classifies
// them, and AscendLowerTileOp resolves the allocation binding before lowering.
AscendCopy::AscendCopy(const CopyNode &base) {
  ObjectPtr<AscendCopyNode> node = make_object<AscendCopyNode>(base);
  DecodeAscendCopyAnnotations(node.get());
  data_ = std::move(node);
}

// Creates a shallow clone of this AscendCopyNode.
TileOperator AscendCopyNode::Clone() const {
  auto op = make_object<AscendCopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return AscendCopy(op);
}

LayoutMap AscendCopyNode::InferLayout(const LayoutInferArgs &layout_args,
                                      InferLevel level) const {
  return ascend::AscendCopyImpl::InferLayout(*this, layout_args, level);
}

Stmt AscendCopyNode::Lower(const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) const {
  return ascend::AscendCopyImpl::Lower(*this, lower_args, analyzer);
}

L0SFBindings CollectL0SFBindings(const Stmt &body) {
  L0SFBindings bindings, data_to_sf;
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    const auto *block = obj.as<SBlockNode>();
    if (!block)
      return;
    if (auto annotation = block->annotations.Get(kL0SFBindings)) {
      for (const auto &[sf, data] : annotation.value().cast<L0SFBindings>()) {
        if (auto existing = bindings.Get(sf)) {
          ICHECK(existing.value().same_as(data))
              << "Conflicting allocation bindings for L0 SF " << sf->name_hint;
        }
        ICHECK(!sf.same_as(data))
            << "L0 SF and data must have distinct storage";
        if (auto existing = data_to_sf.Get(data)) {
          ICHECK(existing.value().same_as(sf))
              << "L0 data " << data->name_hint
              << " has multiple SF handles; reuse the handle returned by "
                 "alloc_l0a_sf/alloc_l0b_sf";
        }
        bindings.Set(sf, data);
        data_to_sf.Set(data, sf);
      }
    }
  });
  return bindings;
}

bool IsAscendCopyCall(const CallNode *call) {
  if (call == nullptr) {
    return false;
  }
  return call->op.same_as(AscendCopy::Get()) || call->op.same_as(Copy::Get());
}

// Register the Ascend dialect copy operation. Same contract as
// tl.tileop.copy; the Ascend lowering hints ride in annotations and are
// decoded into typed AscendCopyNode fields at parse time.
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(AscendCopy, ascend_copy)
    .set_attr<OpBlockAnnotationHandlerFunc>(kTLOpBlockAnnotationHandler,
                                            ApplyAscendCopyBlockAnnotations)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { AscendCopyNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
