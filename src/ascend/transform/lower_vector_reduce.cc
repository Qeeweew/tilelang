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
#include <string>
#include <vector>

#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/attr.h"
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
        call && call->op.same_as(ReduceOp::Get()))
      return LowerReduce(ReduceOp(call->args, call->annotations));
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
    if (auto map = op->annotations.Get(attr::kLayoutMap))
      layouts_ = map.value().cast<LayoutMap>();
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    if (root) {
      SBlockNode *writer = block.CopyOnWrite();
      for (const Buffer &buffer : scratch_)
        writer->alloc_buffers.push_back(buffer);
      scratch_.clear();
    }
    layouts_ = saved;
    return block;
  }

  arith::Analyzer analyzer_;
  LayoutMap layouts_;
  Array<Buffer> scratch_;
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
