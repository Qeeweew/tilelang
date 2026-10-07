// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_vector_reduce.cc
 * \brief Expose native2201 trailing-axis reductions and partial UB storage.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/aiv_private.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/buffer_version.h"
#include "ascend/transform/storage_footprint.h"
#include "op/builtin.h"
#include "op/reduce.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

// The CANN9.1 npu_arch_2201 count wrappers invoke vcadd(last_acc_mode=0)
// and vcmax(ONLY_VALUE). They produce ceil(count / (256 / sizeof(T)))
// contiguous partials, not one scalar for an arbitrary count. Destination
// repeat stride is measured in elements. Each call remains one V task.
TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_sum")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "asc_repeat_reduce_sum");

TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_max_only_value")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol",
                             "asc_repeat_reduce_max_only_value");

// Complete FP32 datablocks contract eight source elements to one result.
// The bounded dense-row plan below uses only full 256B count repeats, so
// every output block and both typed access_ptr extents are exact.
TVM_REGISTER_OP("tl.ascend_membase.datablock_reduce_sum")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "asc_datablock_reduce_sum");
TVM_REGISTER_OP("tl.ascend_membase.datablock_reduce_max")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "asc_datablock_reduce_max");

// Normal-mask repeats retain exact [rows, valid_lanes] source regions.
// Unlike the count overload, the physical row stride and active lanes are
// independently represented. No inactive lane is part of the memory effect.
TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_sum_rows_f32")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_max_rows_f32")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

namespace {

class VectorReduceLowerer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.has_value())
        << "AscendLowerVectorReduce requires a bound target";
    if (GetAscendTargetInfo(target.value()).architecture !=
        AscendArchitecture::kDav2201)
      return func;
    VectorReduceLowerer lowerer;
    lowerer.ub_capacity_bytes_ = GetAscendTargetInfo(target.value()).ub_bytes;
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = lowerer.VisitStmt(func->body);
    ICHECK(lowerer.scratch_.empty())
        << "AscendLowerVectorReduce requires a root allocation block";
    return func;
  }

private:
  void Require(bool condition, const char *message) const {
    if (!condition)
      TVM_FFI_THROW(ValueError) << "dav-2201 MemBase reduce: " << message;
  }

  static Stmt MakeSeq(const Array<Stmt> &stmts) {
    return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
  }

  PrimExpr FlatOffset(const BufferLoad &load) {
    if (auto layout = ascend::FindLayoutForBuffer(layouts_, load->buffer)) {
      Require(layout.value()->InputDim() == load->indices.size(),
              "layout rank must match the UB operand");
      auto indices = layout.value()->Forward(load->indices);
      auto shape = layout.value()->OutputShape();
      PrimExpr offset = make_zero(indices[0].dtype());
      for (size_t i = 0; i < indices.size(); ++i)
        offset = offset * shape[i] + indices[i];
      return offset + load->buffer->elem_offset;
    }
    auto offsets = load->buffer->ElemOffset(load->indices);
    Require(offsets.size() == 1, "axis-separated UB storage is unsupported");
    return offsets[0];
  }

  void CheckBounds(const BufferRegion &region) {
    for (size_t i = 0; i < region->region.size(); ++i) {
      auto range = region->region[i];
      Require(CanProveAscendRangeInBounds(range->min, range->extent,
                                          region->buffer->shape[i], &analyzer_),
              "source and destination regions must lie within their UB "
              "buffers; clamp the domain or use a padded tile");
    }
  }

  PrimExpr Ptr(const BufferLoad &base, int64_t extent, int mask) {
    return Call(DataType::Handle(), tl::access_ptr(),
                {base, make_const(DataType::Int(32), extent),
                 make_const(DataType::Int(32), mask)});
  }

  BufferLoad NewScratch(int64_t elements, DataType dtype) {
    // Every intermediate is a later source, whose native contract requires
    // a 32-byte start. Round capacity, while effects retain only valid
    // partials.
    int64_t block_elements = 32 / (dtype.bits() / 8);
    int64_t padded =
        (elements + block_elements - 1) / block_elements * block_elements;
    Require(padded <= std::numeric_limits<int32_t>::max(),
            "aligned reduction scratch exceeds the int32 metadata range");
    Buffer buffer =
        decl_buffer({make_const(DataType::Int(32), padded)}, dtype,
                    "membase_reduce_tmp_" + std::to_string(next_temporary_++),
                    "shared.dyn");
    scratch_.push_back(buffer);
    return BufferLoad(buffer, {make_const(DataType::Int(32), 0)});
  }

  BufferLoad CombineCompactMax(const BufferLoad &reduced,
                               const BufferLoad &destination,
                               Array<Stmt> *tasks) {
    // Compact output rows are legal S accesses but need not be aligned V
    // operands. Snapshot the old typed value into compiler-owned aligned UB,
    // then expose the combine as an ordinary one-lane vector operation. In
    // particular, dav-2201 has no scalar half max instruction.
    DataType dtype = destination->buffer->dtype;
    Var previous("membase_reduce_previous_" + std::to_string(next_temporary_++),
                 dtype);
    tasks->push_back(Bind(previous, destination));
    BufferLoad old_output = NewScratch(1, dtype);
    tasks->push_back(BufferStore(old_output->buffer, previous, {0}));
    BufferLoad combined = NewScratch(1, dtype);
    Var combine("membase_reduce_combine_" + std::to_string(next_temporary_++),
                DataType::Int(32));
    tasks->push_back(
        For(combine, 0, 1, ForKind::kParallel,
            BufferStore(combined->buffer,
                        Max(BufferLoad(reduced->buffer, {combine}),
                            BufferLoad(old_output->buffer, {combine})),
                        {combine})));
    return combined;
  }

  Optional<Stmt> TryBatchNativeRows(const ReduceOp &reduce,
                                    const BufferLoad &src,
                                    const BufferLoad &dst,
                                    const std::vector<Var> &rows,
                                    int64_t input_count) {
    const auto *op = reduce.get();
    const DataType dtype = op->src->dtype;
    const int64_t vl = 256 / (dtype.bits() / 8);
    if (!op->clear || rows.size() != 1 || input_count != vl ||
        op->src->dtype != op->dst->dtype || op->src->shape.size() != 2 ||
        !op->src->axis_separators.empty() || !op->dst->axis_separators.empty())
      return std::nullopt;
    PrimExpr row_extent = analyzer_.Simplify(op->srcRegion_->region[0]->extent);
    const auto *row_count = row_extent.as<IntImmNode>();
    // Counter mode ignores the uint8 repeat field; the native count is
    // uint32. Retain the tighter existing int32 pointer-metadata contract.
    if (!row_count || row_count->value <= 0 ||
        row_count->value > std::numeric_limits<int32_t>::max() / vl)
      return std::nullopt;

    auto src_ranges = op->srcRegion_->region;
    auto dst_ranges = op->dstRegion_->region;
    // A flat access_ptr also describes a logical row-major run. Whole trailing
    // dimensions keep that common effect faithful to the proved physical run;
    // a layout-transposed partial column view needs the ordinary fallback.
    if (!analyzer_.CanProveEqual(op->src->shape.back(), Integer(vl)) ||
        !analyzer_.CanProveEqual(src_ranges.back()->min, Integer(0)) ||
        op->dst->shape.size() != dst_ranges.size() ||
        (dst_ranges.size() == 2 &&
         (!analyzer_.CanProveEqual(op->dst->shape.back(), Integer(1)) ||
          !analyzer_.CanProveEqual(dst_ranges.back()->min, Integer(0)))))
      return std::nullopt;
    if (auto layout = ascend::FindLayoutForBuffer(layouts_, op->dst);
        layout.has_value() && layout.value()->InputDim() != dst->indices.size())
      return std::nullopt;

    Array<PrimExpr> src_indices, dst_indices;
    for (const Range &range : src_ranges)
      src_indices.push_back(range->min);
    for (const Range &range : dst_ranges)
      dst_indices.push_back(range->min);
    BufferLoad src_base(op->src, src_indices), dst_base(op->dst, dst_indices);
    PrimExpr src_offset = FlatOffset(src_base);
    PrimExpr dst_offset = FlatOffset(dst_base);
    int64_t dst_alignment = op->type->IsMax() ? 2 : 1;
    if (!analyzer_.CanProveEqual(
            FlatOffset(src),
            src_offset + rows[0] * make_const(rows[0].dtype(), vl)) ||
        !analyzer_.CanProveEqual(FlatOffset(dst), dst_offset + rows[0]) ||
        !analyzer_.CanProveEqual(
            floormod(src_offset, Integer(32 / (dtype.bits() / 8))),
            Integer(0)) ||
        !analyzer_.CanProveEqual(floormod(dst_offset, Integer(dst_alignment)),
                                 Integer(0)))
      return std::nullopt;

    const int64_t count = row_count->value * vl;
    const Op &intrinsic = Op::Get(
        op->type->IsSum() ? "tl.ascend_membase.repeat_reduce_sum"
                          : "tl.ascend_membase.repeat_reduce_max_only_value");
    // Each 256B input repeat is one complete logical row. ONLY_VALUE dst
    // stride=1 is in elements; the conservative C API alignment applies only
    // to this call's base, not each internal result. No row scalar readback.
    return Evaluate(Call(DataType::Void(), intrinsic,
                         {Ptr(dst_base, row_count->value, kAccessWrite),
                          Ptr(src_base, count, kAccessRead),
                          make_const(DataType::UInt(32), count)}));
  }

  Optional<Stmt> TryMultiVLNativeRows(const ReduceOp &reduce,
                                      const BufferLoad &src,
                                      const BufferLoad &dst,
                                      const std::vector<Var> &rows,
                                      int64_t input_count) {
    const auto *op = reduce.get();
    // Keep the established count-mode whole-VL path and all tail/dynamic
    // cases. This is a normal-mask static repeat plan, not a count wrapper.
    if (!op->clear || op->src->dtype != DataType::Float(32) ||
        op->dst->dtype != DataType::Float(32) || rows.size() != 1 ||
        op->src->shape.size() != 2 || input_count <= 64 ||
        input_count % 64 != 0 || input_count / 8 > 65535 ||
        !op->src->axis_separators.empty() || !op->dst->axis_separators.empty())
      return std::nullopt;
    const auto &src_ranges = op->srcRegion_->region;
    const auto &dst_ranges = op->dstRegion_->region;
    PrimExpr row_extent = analyzer_.Simplify(src_ranges[0]->extent);
    const auto *row_count = row_extent.as<IntImmNode>();
    if (!row_count || row_count->value <= 0 || row_count->value > 255 ||
        !analyzer_.CanProveEqual(
            op->src->shape.back(),
            make_const(op->src->shape.back().dtype(), input_count)) ||
        !is_zero(src_ranges.back()->min) ||
        (dst_ranges.size() == 2 &&
         (!is_one(op->dst->shape.back()) || !is_zero(dst_ranges.back()->min))))
      return std::nullopt;
    // The late native boundary consumes this original Buffer before layout
    // remapping. Keep selection in that same compact physical-view domain.
    if (!op->src->strides.empty() &&
        (!is_one(op->src->strides.back()) ||
         !analyzer_.CanProveEqual(
             op->src->strides[0],
             make_const(op->src->strides[0].dtype(), input_count))))
      return std::nullopt;

    Array<PrimExpr> src_indices, dst_indices;
    for (const Range &range : src_ranges)
      src_indices.push_back(range->min);
    for (const Range &range : dst_ranges)
      dst_indices.push_back(range->min);
    BufferLoad src_base(op->src, src_indices), dst_base(op->dst, dst_indices);
    PrimExpr src_offset = FlatOffset(src_base),
             dst_offset = FlatOffset(dst_base);
    auto raw_offsets = op->src->ElemOffset(src_indices);
    if (!analyzer_.CanProveEqual(
            FlatOffset(src),
            src_offset + rows[0] * make_const(rows[0].dtype(), input_count)) ||
        !analyzer_.CanProveEqual(FlatOffset(dst), dst_offset + rows[0]) ||
        !analyzer_.CanProveEqual(
            floormod(src_offset, make_const(src_offset.dtype(), 8)),
            make_zero(src_offset.dtype())) ||
        !analyzer_.CanProveEqual(
            floormod(dst_offset, make_const(dst_offset.dtype(), 8)),
            make_zero(dst_offset.dtype())) ||
        raw_offsets.size() != 1 ||
        !analyzer_.CanProveEqual(
            floormod(raw_offsets[0], make_const(raw_offsets[0].dtype(), 8)),
            make_zero(raw_offsets[0].dtype())))
      return std::nullopt;

    const int64_t count = row_count->value;
    const int64_t chunks = input_count / 64;
    const Op &intrinsic = Op::Get(
        op->type->IsSum() ? "tl.ascend_membase.repeat_reduce_sum_rows_f32"
                          : "tl.ascend_membase.repeat_reduce_max_rows_f32");
    // This plan writes the user's destination through the native row op,
    // unlike the chunk tree's ordinary vector combine. Its raw Buffer must
    // satisfy the same compact/aligned contract as its mapped layout.
    auto raw_dst_base = op->dst->ElemOffset(dst_indices);
    auto raw_dst_row = op->dst->ElemOffset(dst->indices);
    bool native_dst_compact =
        raw_dst_base.size() == 1 && raw_dst_row.size() == 1 &&
        (op->dst->strides.empty() || (is_one(op->dst->strides.front()) &&
                                      is_one(op->dst->strides.back()))) &&
        analyzer_.CanProveEqual(raw_dst_row[0], raw_dst_base[0] + rows[0]) &&
        analyzer_.CanProveEqual(
            floormod(raw_dst_base[0], make_const(raw_dst_base[0].dtype(), 8)),
            make_zero(raw_dst_base[0].dtype()));
    // Full datablock count repeats compact each group of eight elements.
    // For these widths both count stages end on complete 256B repeats and
    // preserve row boundaries: [R,C] -> [R,C/8] -> [R,C/64]. Prefer this
    // uniform three-stage candidate only when both disjoint scratch buffers
    // fit. Otherwise retain the established whole-repeat two-level plan,
    // whose smaller partial allocation has the same final [R,C/64] view.
    const bool compact_wide_rows = input_count >= 512 && input_count <= 4096 &&
                                   input_count % 512 == 0 && native_dst_compact;
    const bool use_datablocks =
        compact_wide_rows &&
        HasMultiVLHeadroom(op->src, op->dst,
                           count * (input_count / 8 + chunks) * 4);
    if (compact_wide_rows &&
        (use_datablocks ||
         HasMultiVLHeadroom(op->src, op->dst, count * chunks * 4))) {
      Require(task_depth_ == 0,
              "this reduction expands to multiple dependency-bearing tasks; "
              "use T.Stage rather than an opaque T.Task/PerCoreTask wrapper");
      Buffer partials = decl_buffer({make_const(DataType::Int(32), count),
                                     make_const(DataType::Int(32), chunks)},
                                    DataType::Float(32),
                                    "membase_reduce_partials_" +
                                        std::to_string(next_temporary_++),
                                    "shared.dyn");
      BufferLoad partial_base(partials, {0, 0});
      Array<Stmt> tasks;
      if (use_datablocks) {
        const int64_t block_columns = input_count / 8;
        Buffer blocks = decl_buffer(
            {make_const(DataType::Int(32), count),
             make_const(DataType::Int(32), block_columns)},
            DataType::Float(32),
            "membase_reduce_datablocks_" + std::to_string(next_temporary_++),
            "shared.dyn");
        scratch_.push_back(blocks);
        BufferLoad block_base(blocks, {0, 0});
        const Op &count_intrinsic = Op::Get(
            op->type->IsSum() ? "tl.ascend_membase.datablock_reduce_sum"
                              : "tl.ascend_membase.datablock_reduce_max");
        tasks.push_back(Evaluate(
            Call(DataType::Void(), count_intrinsic,
                 {Ptr(block_base, count * block_columns, kAccessWrite),
                  Ptr(src_base, count * input_count, kAccessRead),
                  make_const(DataType::UInt(32), count * input_count)})));
        tasks.push_back(Evaluate(
            Call(DataType::Void(), count_intrinsic,
                 {Ptr(partial_base, count * chunks, kAccessWrite),
                  Ptr(block_base, count * block_columns, kAccessRead),
                  make_const(DataType::UInt(32), count * block_columns)})));
      } else {
        const Op &count_intrinsic =
            Op::Get(op->type->IsSum()
                        ? "tl.ascend_membase.repeat_reduce_sum"
                        : "tl.ascend_membase.repeat_reduce_max_only_value");
        tasks.push_back(Evaluate(
            Call(DataType::Void(), count_intrinsic,
                 {Ptr(partial_base, count * chunks, kAccessWrite),
                  Ptr(src_base, count * input_count, kAccessRead),
                  make_const(DataType::UInt(32), count * input_count)})));
      }
      scratch_.push_back(partials);
      PrimExpr partial_region =
          Call(DataType::Handle(), tl::region(),
               {partial_base, make_const(DataType::Int(32), kAccessRead),
                make_const(DataType::Int(32), count),
                make_const(DataType::Int(32), chunks)});
      Array<PrimExpr> destination_args{
          dst_base, make_const(DataType::Int(32), kAccessWrite)};
      for (const Range &range : dst_ranges)
        destination_args.push_back(range->extent);
      tasks.push_back(Evaluate(
          Call(DataType::Void(), intrinsic,
               {Call(DataType::Handle(), tl::region(), destination_args),
                partial_region, make_const(DataType::UInt(8), count),
                make_const(DataType::UInt(16), chunks / 8),
                make_const(DataType::UInt(8), chunks)})));
      return MakeSeq(tasks);
    }

    int64_t levels = 0, remaining = input_count;
    do {
      remaining = (remaining + 63) / 64;
      ++levels;
    } while (remaining > 1);
    // Bound expansion by the ordinary row plan's native instruction count.
    // This is a structural selection heuristic, not a measured time model.
    if (2 * chunks - 1 > count * levels)
      return std::nullopt;

    PrimExpr aligned_row_bytes = analyzer_.Simplify(ascend::AlignPrimExpr(
        ascend::PackedBytesFromElems(make_const(DataType::Int(64), count), 32),
        ascend::AscendScopeAlignBytes("shared.dyn")));
    const auto *row_bytes = aligned_row_bytes.as<IntImmNode>();
    ICHECK(row_bytes && row_bytes->value > 0);
    const int64_t partial_bytes = chunks * row_bytes->value;
    // VectorTiles emits at most one aligned row temporary per binary node.
    // Charge even the final node, although its output normally is dst. Reserve
    // these future allocations across subsequent reductions in the same root.
    const int64_t combine_bytes = (chunks - 1) * row_bytes->value;
    if (!HasMultiVLHeadroom(op->src, op->dst, partial_bytes + combine_bytes))
      return std::nullopt;
    Require(task_depth_ == 0,
            "this reduction expands to multiple dependency-bearing tasks; "
            "use T.Stage rather than an opaque T.Task/PerCoreTask wrapper");

    const int64_t row_capacity = (count + 7) / 8 * 8;
    Buffer partials = decl_buffer({make_const(DataType::Int(32), chunks),
                                   make_const(DataType::Int(32), row_capacity)},
                                  DataType::Float(32),
                                  "membase_reduce_partials_" +
                                      std::to_string(next_temporary_++),
                                  "shared.dyn");
    scratch_.push_back(partials);
    future_combine_bytes_ += combine_bytes;
    Array<Stmt> tasks;
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
      Array<PrimExpr> source = src_indices;
      source.Set(1, source[1] + make_const(source[1].dtype(), chunk * 64));
      PrimExpr source_region =
          Call(DataType::Handle(), tl::region(),
               {BufferLoad(op->src, source),
                make_const(DataType::Int(32), kAccessRead),
                make_const(src_ranges[0]->extent.dtype(), count),
                make_const(src_ranges[1]->extent.dtype(), 64)});
      PrimExpr destination_region =
          Call(DataType::Handle(), tl::region(),
               {BufferLoad(partials, {make_const(DataType::Int(32), chunk), 0}),
                make_const(DataType::Int(32), kAccessWrite),
                make_const(DataType::Int(32), 1),
                make_const(DataType::Int(32), count)});
      tasks.push_back(
          Evaluate(Call(DataType::Void(), intrinsic,
                        {destination_region, source_region,
                         make_const(DataType::UInt(8), count),
                         make_const(DataType::UInt(16), input_count / 8),
                         make_const(DataType::UInt(8), 64)})));
    }

    // Each chunk performs the native 64-lane tree. Combine compact row
    // vectors with an ordinary balanced tree, preserving typed FP32 nodes
    // and their separate scheduling/completion boundaries in VectorTiles.
    std::vector<PrimExpr> values;
    for (int64_t chunk = 0; chunk < chunks; ++chunk)
      values.push_back(BufferLoad(
          partials, {make_const(DataType::Int(32), chunk), rows[0]}));
    while (values.size() > 1) {
      std::vector<PrimExpr> next;
      for (size_t i = 0; i < values.size(); i += 2)
        next.push_back(i + 1 == values.size() ? values[i]
                       : op->type->IsSum()
                           ? PrimExpr(Add(values[i], values[i + 1]))
                           : PrimExpr(Max(values[i], values[i + 1])));
      values = std::move(next);
    }
    tasks.push_back(For(rows[0], make_zero(rows[0].dtype()),
                        make_const(rows[0].dtype(), count), ForKind::kParallel,
                        BufferStore(op->dst, values[0], dst->indices)));
    return MakeSeq(tasks);
  }

  bool HasMultiVLHeadroom(const Buffer &src, const Buffer &dst,
                          int64_t candidate_bytes) {
    if (!ub_owner_bytes_.has_value() || !ub_slots_.Get(src->data).has_value() ||
        !ub_slots_.Get(dst->data).has_value())
      return false;
    // The late manual-version layout can change these operands' row stride.
    // A separate manual ring does not change the native rows, but its aligned
    // physical slots must still be included in the root's UB footprint.
    for (const Var &data : {src->data, dst->data}) {
      if (auto versions = manual_versions_.Get(data);
          versions.has_value() && versions.value() > 1)
        return false;
    }
    int64_t remaining = ub_capacity_bytes_ - ub_owner_bytes_.value();
    auto charge = [&](int64_t bytes) {
      if (bytes < 0 || bytes > remaining)
        return false;
      remaining -= bytes;
      return true;
    };
    if (!charge(future_combine_bytes_) || !charge(candidate_bytes))
      return false;
    for (const Buffer &buffer : scratch_) {
      PrimExpr footprint = analyzer_.Simplify(ascend::AlignPrimExpr(
          ascend::GetBufferSizeBytes(buffer, DataType::Int(64)),
          ascend::AscendScopeAlignBytes(GetPtrStorageScope(buffer->data))));
      const auto *bytes = footprint.as<IntImmNode>();
      if (!bytes || !charge(bytes->value))
        return false;
    }
    return true;
  }

  Stmt LowerReduce(const ReduceOp &reduce) {
    const auto *op = reduce.get();
    Require(IsSharedBuffer(op->src) && IsSharedBuffer(op->dst),
            "ordinary reductions require private UB operands");
    Require(op->type->IsSum() || op->type->IsMax(),
            "only reduce_sum and reduce_max are implemented");
    Require(!op->nan_propagate,
            "nan_propagate=True has no established native2201 contract");
    DataType dtype = op->src->dtype;
    const bool low_input = dtype.is_float16() || dtype.is_bfloat16();
    const bool promote_sum =
        op->type->IsSum() && low_input && op->dst->dtype == DataType::Float(32);
    const bool promote_max =
        op->type->IsMax() && dtype.is_bfloat16() && op->dst->dtype == dtype;
    const bool native =
        op->dst->dtype == dtype && (dtype == DataType::Float(32) ||
                                    (op->type->IsMax() && dtype.is_float16()));
    Require(native || promote_sum || promote_max,
            "sum supports float32 accumulation/output, including float16 or "
            "bfloat16 input. Low-precision sum output needs an explicit "
            "accumulator-precision contract; FP32 promotion cannot preserve "
            "its typed intermediate rounding. Max supports matching "
            "float16/bfloat16/float32 input/output");
    Require(!op->src->data.same_as(op->dst->data),
            "source/destination storage aliasing requires an explicit "
            "non-overlapping temporary");

    auto src_ranges = op->srcRegion_->region;
    auto dst_ranges = op->dstRegion_->region;
    size_t rank = src_ranges.size();
    Require(rank > 0 && op->dim == static_cast<int>(rank) - 1,
            "only the trailing axis is supported; ownership-axis or "
            "strided reductions need a separate reduction plan");
    const auto *inner = src_ranges.back()->extent.as<IntImmNode>();
    Require(inner && inner->value > 0,
            "the reduced extent must be a positive compile-time constant; "
            "count zero must not issue a native instruction");
    int64_t input_count = inner->value;
    Require(input_count <= std::numeric_limits<int32_t>::max(),
            "reduced extent exceeds the supported UB pointer-metadata range");
    Require(dst_ranges.size() == rank - 1 || dst_ranges.size() == rank,
            "output region must remove the reduced axis or retain it with "
            "extent one");
    if (dst_ranges.size() == rank)
      Require(analyzer_.CanProveEqual(
                  dst_ranges.back()->extent,
                  make_const(dst_ranges.back()->extent.dtype(), 1)),
              "a retained output axis must have extent one");
    CheckBounds(op->srcRegion_);
    CheckBounds(op->dstRegion_);

    Array<PrimExpr> src_indices, dst_indices;
    std::vector<Var> rows;
    for (size_t i = 0; i + 1 < rank; ++i) {
      Require(
          analyzer_.CanProveEqual(src_ranges[i]->extent, dst_ranges[i]->extent),
          "source/output non-reduced region extents must match");
      Require(analyzer_.CanProve(src_ranges[i]->extent >= 0),
              "row extents must be nonnegative");
      Var row("membase_reduce_row_" + std::to_string(next_row_++),
              src_ranges[i]->min.dtype());
      rows.push_back(row);
      analyzer_.Bind(row, Range::FromMinExtent(make_zero(row.dtype()),
                                               src_ranges[i]->extent));
      src_indices.push_back(src_ranges[i]->min + row);
      dst_indices.push_back(dst_ranges[i]->min + row);
    }
    src_indices.push_back(src_ranges.back()->min);
    if (dst_ranges.size() == rank)
      dst_indices.push_back(dst_ranges.back()->min);
    BufferLoad src(op->src, src_indices), dst(op->dst, dst_indices);

    PrimExpr offset = FlatOffset(src);
    int source_alignment = 32 / (dtype.bits() / 8);
    Require(analyzer_.CanProveEqual(
                floormod(offset, make_const(offset.dtype(), source_alignment)),
                make_zero(offset.dtype())),
            "each source row must start at a 32-byte aligned physical UB "
            "address; pad the row layout rather than reading another row");
    Var column("membase_reduce_column", src_ranges.back()->min.dtype());
    analyzer_.Bind(column, Range::FromMinExtent(make_zero(column.dtype()),
                                                src_ranges.back()->extent));
    Array<PrimExpr> varying = src_indices;
    varying.Set(rank - 1, varying.back() + column);
    Require(analyzer_.CanProveEqual(FlatOffset(BufferLoad(op->src, varying)),
                                    offset + column),
            "the reduced row must be physically contiguous");

    if (promote_sum || promote_max) {
      // Widen input as ordinary typed Cast nodes. For sum the output dtype
      // specifies FP32 accumulation; low-precision sum output is not silently
      // assigned a different accumulator. BF16 max selects a represented
      // value in the same established comparison domain as the FP32 route.
      Require(task_depth_ == 0,
              "this reduction expands to multiple dependency-bearing tasks; "
              "use T.Stage rather than an opaque T.Task/PerCoreTask wrapper");
      Array<Stmt> tasks;
      BufferLoad wide_input = NewScratch(input_count, DataType::Float(32));
      Var column("membase_reduce_promote_" + std::to_string(next_temporary_++),
                 DataType::Int(32));
      Array<PrimExpr> promoted_indices = src_indices;
      promoted_indices.Set(rank - 1, promoted_indices.back() + column);
      tasks.push_back(
          For(column, 0, make_const(column.dtype(), input_count),
              ForKind::kParallel,
              BufferStore(wide_input->buffer,
                          Cast(DataType::Float(32),
                               BufferLoad(op->src, promoted_indices)),
                          {column})));

      BufferLoad wide_result = NewScratch(1, DataType::Float(32));
      auto scratch_region = [&](const BufferLoad &base, int64_t count,
                                int mask) {
        return Call(DataType::Handle(), tl::region(),
                    {base, make_const(DataType::Int(32), mask),
                     make_const(DataType::Int(32), count)});
      };
      ReduceOp wide_reduce(
          {scratch_region(wide_input, input_count, kAccessRead),
           scratch_region(wide_result, 1, kAccessWrite),
           StringImm(promote_sum ? "sum" : "max"),
           make_const(DataType::Int(32), 0), const_true()});
      tasks.push_back(LowerReduce(wide_reduce));

      BufferLoad final_value = wide_result;
      if (promote_max) {
        // Narrow this reduction node before the separate clear=False Max
        // node. Both remain ordinary typed operations for VectorTiles.
        BufferLoad narrowed = NewScratch(1, dtype);
        Var item("membase_reduce_narrow_" + std::to_string(next_temporary_++),
                 DataType::Int(32));
        tasks.push_back(For(
            item, 0, 1, ForKind::kParallel,
            BufferStore(narrowed->buffer,
                        Cast(dtype, BufferLoad(wide_result->buffer, {item})),
                        {item})));
        final_value = narrowed;
        if (!op->clear)
          final_value = CombineCompactMax(narrowed, dst, &tasks);
      }
      Var result("membase_reduce_result_" + std::to_string(next_temporary_++),
                 op->dst->dtype);
      tasks.push_back(Bind(result, final_value));
      PrimExpr value = result;
      if (promote_sum && !op->clear) {
        Var previous("membase_reduce_previous_" +
                         std::to_string(next_temporary_++),
                     DataType::Float(32));
        tasks.push_back(Bind(previous, dst));
        value = result + previous;
      }
      tasks.push_back(BufferStore(op->dst, value, dst_indices));
      Stmt body = MakeSeq(tasks);
      for (size_t i = rows.size(); i-- > 0;)
        body = For(rows[i], make_zero(rows[i].dtype()), src_ranges[i]->extent,
                   ForKind::kSerial, body);
      return body;
    }

    if (auto batched = TryBatchNativeRows(reduce, src, dst, rows, input_count))
      return batched.value();
    if (auto batched =
            TryMultiVLNativeRows(reduce, src, dst, rows, input_count))
      return batched.value();

    // Max needs twice the element alignment even in ONLY_VALUE mode. An
    // inferred padded output layout may satisfy that contract directly; a
    // compact multi-row output generally needs the explicit scalar path.
    bool native_destination = true;
    if (op->type->IsMax()) {
      PrimExpr dst_offset = FlatOffset(dst);
      native_destination = analyzer_.CanProveEqual(
          floormod(dst_offset, make_const(dst_offset.dtype(), 2)),
          make_zero(dst_offset.dtype()));
    }
    bool compact_store = !native_destination || !op->clear;
    int64_t values_per_repeat = 256 / (dtype.bits() / 8);
    const Op &intrinsic = Op::Get(
        op->type->IsSum() ? "tl.ascend_membase.repeat_reduce_sum"
                          : "tl.ascend_membase.repeat_reduce_max_only_value");
    Array<Stmt> tasks;
    BufferLoad current = src;
    int64_t count = input_count;
    do {
      int64_t partials = (count + values_per_repeat - 1) / values_per_repeat;
      BufferLoad next =
          partials == 1 && !compact_store ? dst : NewScratch(partials, dtype);
      tasks.push_back(Evaluate(Call(DataType::Void(), intrinsic,
                                    {Ptr(next, partials, kAccessWrite),
                                     Ptr(current, count, kAccessRead),
                                     make_const(DataType::UInt(32), count)})));
      current = next;
      count = partials;
    } while (count > 1);

    // Stable CANN max constrains its destination to 4 bytes (half) / 8 bytes
    // (float), with no only_value exception. Keep that operation on aligned
    // scratch, then expose the V->S readback and compact output store. For
    // clear=False the old output is another explicit scalar read. Half max
    // combines in V: the corresponding scalar instruction is unavailable on
    // dav-2201. Its scratch and typed Parallel are visible before scheduling.
    if (compact_store) {
      bool vector_combine =
          !op->clear && op->type->IsMax() && dtype.is_float16();
      if (vector_combine)
        current = CombineCompactMax(current, dst, &tasks);
      Var result("membase_reduce_result_" + std::to_string(next_temporary_++),
                 dtype);
      tasks.push_back(Bind(result, current));
      PrimExpr value = result;
      if (!op->clear && !vector_combine) {
        Var previous("membase_reduce_previous_" +
                         std::to_string(next_temporary_++),
                     dtype);
        tasks.push_back(Bind(previous, dst));
        value = op->type->IsSum() ? PrimExpr(result + previous)
                                  : PrimExpr(Max(result, previous));
      }
      tasks.push_back(BufferStore(op->dst, value, dst_indices));
    }
    Require(task_depth_ == 0 || tasks.size() == 1,
            "this reduction expands to multiple dependency-bearing tasks; "
            "use T.Stage rather than an opaque T.Task/PerCoreTask wrapper");
    Stmt body = MakeSeq(tasks);
    for (size_t i = rows.size(); i-- > 0;)
      body = For(rows[i], make_zero(rows[i].dtype()), src_ranges[i]->extent,
                 ForKind::kSerial, body);
    return body;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(ReduceOp::Get())) {
      ReduceOp reduce(call->args, call->annotations);
      bool private_domain =
          ascend::ContainsAivPrivateData(aiv_private_, reduce->src->data) &&
          ascend::ContainsAivPrivateData(aiv_private_, reduce->dst->data);
      size_t first_scratch = scratch_.size();
      Stmt lowered = LowerReduce(reduce);
      // Scratch is allocated by this producer, in the same local execution
      // domain as the reduction. Later vector lowering consumes that fact;
      // it must not infer ownership again from the scratch's padded shape.
      if (private_domain) {
        for (size_t i = first_scratch; i < scratch_.size(); ++i)
          ascend::AppendAivPrivateData(&aiv_private_, scratch_[i]->data);
      }
      return lowered;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool task = op->attr_key == attr::kAscendTask ||
                op->attr_key == attr::kAscendPerCoreTask;
    if (task)
      ++task_depth_;
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>())
        analyzer_.Bind(iv->var, Range::FromMinExtent(
                                    make_zero(op->value.dtype()), op->value));
    }
    Stmt body = StmtExprMutator::VisitStmt_(op);
    if (task)
      --task_depth_;
    return body;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool root = op->name_hint == "tilelang_root";
    LayoutMap saved = layouts_;
    auto saved_bytes = ub_owner_bytes_;
    auto saved_slots = ub_slots_;
    auto saved_manual_versions = manual_versions_;
    int64_t saved_combine_bytes = future_combine_bytes_;
    auto saved_private = aiv_private_;
    if (root) {
      aiv_private_ = ascend::ValidatedAivPrivateData(op);
      manual_versions_ = {};
      if (auto value = op->annotations.Get(kManualMultiBuffer))
        manual_versions_ = value.value().cast<BufferVersionMap>();
      // Reuse the common allocation/storage facts: owner aliases are charged
      // once at their maximum footprint, including layout output and alignment.
      // Manual rings already reserve their physical storage. Include their
      // per-slot alignment; only automatic future versioning remains unknown.
      ub_slots_ = ascend::StorageFootprintCollector::SlotBytes(
          GetRef<SBlock>(op), manual_versions_);
      ub_owner_bytes_ = 0;
      future_combine_bytes_ = 0;
      for (const auto &[storage, footprint] : ub_slots_) {
        if (ascend::AscendScopeAlignBytes(GetPtrStorageScope(storage)) != 32)
          continue;
        PrimExpr simplified = analyzer_.Simplify(footprint);
        const auto *bytes = simplified.as<IntImmNode>();
        if (!bytes || bytes->value < 0 ||
            bytes->value > ub_capacity_bytes_ - ub_owner_bytes_.value()) {
          ub_owner_bytes_ = std::nullopt;
          break;
        }
        ub_owner_bytes_ = ub_owner_bytes_.value() + bytes->value;
      }
      if (auto value = op->annotations.Get(kBufferVersionsMap)) {
        for (const auto &[storage, versions] :
             value.value().cast<BufferVersionMap>()) {
          if (versions > 1 && ascend::AscendScopeAlignBytes(
                                  GetPtrStorageScope(storage)) == 32)
            ub_owner_bytes_ = std::nullopt;
        }
      }
    }
    if (auto map = op->annotations.Get(attr::kLayoutMap))
      layouts_ = map.value().cast<LayoutMap>();
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    if (root) {
      SBlockNode *writer = block.CopyOnWrite();
      for (const Buffer &buffer : scratch_)
        writer->alloc_buffers.push_back(buffer);
      if (!aiv_private_.empty())
        writer->annotations.Set(ascend::kAivPrivateValidated, aiv_private_);
      scratch_.clear();
    }
    layouts_ = saved;
    if (root) {
      ub_owner_bytes_ = saved_bytes;
      ub_slots_ = saved_slots;
      manual_versions_ = saved_manual_versions;
      future_combine_bytes_ = saved_combine_bytes;
      aiv_private_ = saved_private;
    }
    return block;
  }

  arith::Analyzer analyzer_;
  LayoutMap layouts_;
  Array<Buffer> scratch_;
  Array<Var> aiv_private_;
  ffi::Map<Var, PrimExpr> ub_slots_;
  BufferVersionMap manual_versions_;
  std::optional<int64_t> ub_owner_bytes_;
  int64_t ub_capacity_bytes_{0};
  int64_t future_combine_bytes_{0};
  int next_temporary_{0};
  int next_row_{0};
  int task_depth_{0};
};

} // namespace

tvm::transform::Pass AscendLowerVectorReduce() {
  return tirx::transform::CreatePrimFuncPass(
      [](PrimFunc func, const IRModule &mod,
         const tvm::transform::PassContext &ctx) {
        return VectorReduceLowerer::Rewrite(std::move(func));
      },
      0, "tl.AscendLowerVectorReduce", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLowerVectorReduce",
                                   AscendLowerVectorReduce);
}

} // namespace tl
} // namespace tvm
