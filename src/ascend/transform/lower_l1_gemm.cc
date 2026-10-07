/*!
 * \file ascend/transform/lower_l1_gemm.cc
 * \brief Expose native dav-2201 L1 GEMM movement before scheduling.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/utils.h"

#include "lower_l1_gemm_mn.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

int64_t Align16(int64_t extent) { return (extent + 15) / 16 * 16; }

PrimExpr RegionCall(const Buffer &buffer, const Array<Range> &ranges,
                    int access_mask) {
  Array<PrimExpr> indices;
  for (const Range &range : ranges)
    indices.push_back(range->min);
  Array<PrimExpr> args{BufferLoad(buffer, indices),
                       IntImm(DataType::Int(32), access_mask)};
  for (const Range &range : ranges)
    args.push_back(range->extent);
  return Call(DataType::Handle(), region(), args);
}

Array<Range> L0Ranges(const PrimExpr &rows, const PrimExpr &columns) {
  return {Range::FromMinExtent(make_zero(rows.dtype()), rows),
          Range::FromMinExtent(make_zero(columns.dtype()), columns)};
}

// Keep the source BufferRegion as the durable truth, including any leading
// version indices and the original nonzero matrix origin. Only K is sliced.
Array<Range> SubKSource(const BufferRegion &region, PrimExpr k_start,
                        int64_t k_extent, bool k_is_rows = false) {
  Array<Range> ranges = region->region;
  size_t axis = ranges.size() - (k_is_rows ? 2 : 1);
  const Range &original = ranges[axis];
  ranges.Set(axis, Range::FromMinExtent(
                       original->min + k_start,
                       make_const(original->extent.dtype(), k_extent)));
  return ranges;
}

class L1GemmLowerer : public StmtMutator {
public:
  L1GemmLowerer(
      const AscendTargetInfo &target,
      const ascend::l1_gemm_mn_detail::PhysicalMNLoops &physical_mn_loops)
      : target_(target), physical_mn_loops_(physical_mn_loops) {}

private:
  struct PhysicalMNContext {
    Var tile;
    int64_t n_tiles;
  };

  Stmt VisitStmt_(const ForNode *op) final {
    Optional<PrimExpr> previous = auto_pipeline_bound_;
    auto previous_mn = physical_mn_;
    physical_mn_.reset();
    auto mn = physical_mn_loops_.find(op->loop_var);
    if (mn != physical_mn_loops_.end() && mn->second > 1)
      physical_mn_ = PhysicalMNContext{op->loop_var, mn->second};
    if (auto intent = op->annotations.Get("tl.ascend_auto_pipeline")) {
      auto_pipeline_bound_ = std::nullopt;
      if (intent.value().cast<bool>()) {
        auto bound = op->annotations.Get("num_stages");
        ICHECK(bound.has_value())
            << "Automatic Ascend pipeline requires a buffer-version bound";
        auto_pipeline_bound_ = bound.value().cast<PrimExpr>();
      }
    } else if (op->annotations.count("num_stages")) {
      // Explicit pipeline controls do not inherit an enclosing automatic
      // optimization request.
      auto_pipeline_bound_ = std::nullopt;
    }
    Stmt rewritten = StmtMutator::VisitStmt_(op);
    auto_pipeline_bound_ = previous;
    physical_mn_ = previous_mn;
    return rewritten;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root")
      return StmtMutator::VisitStmt_(op);
    pending_allocations_.emplace_back();
    pending_single_versions_.emplace_back();
    SBlock rewritten = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
    std::vector<Buffer> pending = std::move(pending_allocations_.back());
    pending_allocations_.pop_back();
    for (const Buffer &buffer : pending)
      rewritten.CopyOnWrite()->alloc_buffers.push_back(buffer);
    if (!pending_single_versions_.back().empty()) {
      BufferVersionMap versions;
      if (auto value = rewritten->annotations.Get(kBufferVersionsMap))
        versions = value.value().cast<BufferVersionMap>();
      for (const Var &storage : pending_single_versions_.back())
        versions.Set(storage, 1);
      rewritten.CopyOnWrite()->annotations.Set(kBufferVersionsMap, versions);
    }
    pending_single_versions_.pop_back();
    return rewritten;
  }

  static void RequireSupportedTranspose(const Gemm &gemm) {
    if (gemm->transA_) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM requires transpose_A=False; "
             "transposed A requires a separately supported native "
             "L1-to-L0 layout route";
    }
  }

  void ValidateGeometry(const Gemm &gemm) {
    if (gemm->m_ <= 0 || gemm->n_ <= 0 || gemm->k_ <= 0) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM requires positive static M/N/K";
    }
    if (gemm->aRegion_->region.size() < 2 ||
        gemm->bRegion_->region.size() < 2 ||
        gemm->cRegion_->region.size() < 2) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM requires matrix BufferRegions";
    }
    const auto &a = gemm->aRegion_->region;
    const auto &b = gemm->bRegion_->region;
    const auto &c = gemm->cRegion_->region;
    if (!analyzer_.CanProveEqual(a[a.size() - 2]->extent,
                                 c[c.size() - 2]->extent) ||
        !analyzer_.CanProveEqual(b[b.size() - (gemm->transB_ ? 2 : 1)]->extent,
                                 c.back()->extent) ||
        !analyzer_.CanProveEqual(c[c.size() - 2]->extent, gemm->m_) ||
        !analyzer_.CanProveEqual(c.back()->extent, gemm->n_) ||
        !analyzer_.CanProveEqual(a.back()->extent, gemm->k_) ||
        !analyzer_.CanProveEqual(b[b.size() - (gemm->transB_ ? 1 : 2)]->extent,
                                 gemm->k_)) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM operand regions must agree on M/N/K";
    }
  }

  int64_t SelectSubK(const Gemm &gemm) const {
    const int64_t bytes = gemm->a_->dtype.bits() / 8;
    const int64_t m = Align16(gemm->m_);
    const int64_t n = Align16(gemm->n_);
    auto fit = [&](int64_t divisor) {
      return std::min(target_.l0a_bytes / divisor / (m * bytes),
                      target_.l0b_bytes / divisor / (n * bytes)) /
             16 * 16;
    };
    // A sizing heuristic, in bytes: leave space for two potential versions.
    // The scheduler still chooses the actual count from dependencies and
    // target capacity; this does not declare any measured performance gain.
    int64_t tile_k = fit(2);
    if (tile_k == 0)
      tile_k = fit(1);
    if (tile_k == 0) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM cannot fit a 16-element sub-K tile in "
             "L0A/L0B (M="
          << gemm->m_ << ", N=" << gemm->n_ << ", operand bytes=" << bytes
          << ", L0A bytes=" << target_.l0a_bytes
          << ", L0B bytes=" << target_.l0b_bytes << ")";
    }
    return std::min(tile_k, Align16(gemm->k_));
  }

  PrimExpr NonterminalUnitFlag(const PrimExpr &terminal) {
    // CANN's CHECK bit protects every native K chunk. Only the terminal
    // chunk may additionally SET the result for FIX: 3 -> 2, 2 -> 2,
    // 1 -> 0, 0 -> 0. Preserve conditional CHECK without expanding Selects.
    return analyzer_.Simplify(
        bitwise_and(terminal, make_const(terminal.dtype(), 2)));
  }

  Stmt MakeChunk(const Call &original, const Gemm &gemm, const Buffer &a_l0,
                 const Buffer &b_l0, PrimExpr k_start, int64_t k_extent,
                 PrimExpr clear_accum, bool is_last,
                 Optional<PrimExpr> a_load_guard = std::nullopt) {
    const auto &a = gemm->aRegion_->region;
    const auto &b = gemm->bRegion_->region;
    PrimExpr chunk_k = make_const(DataType::Int(32), k_extent);
    Array<Range> a_dst = L0Ranges(a[a.size() - 2]->extent, chunk_k);
    Array<Range> b_dst = gemm->transB_
                             ? L0Ranges(b[b.size() - 2]->extent, chunk_k)
                             : L0Ranges(chunk_k, b.back()->extent);
    Stmt a_copy = Evaluate(Call(
        DataType::Handle(), AscendCopy::Get(),
        {RegionCall(gemm->a_, SubKSource(gemm->aRegion_, k_start, k_extent),
                    kAccessRead),
         RegionCall(a_l0, a_dst, kAccessWrite)},
        {}, original->span));
    if (a_load_guard.has_value())
      a_copy = IfThenElse(a_load_guard.value(), a_copy);
    Stmt b_copy =
        Evaluate(Call(DataType::Handle(), AscendCopy::Get(),
                      {RegionCall(gemm->b_,
                                  SubKSource(gemm->bRegion_, k_start, k_extent,
                                             !gemm->transB_),
                                  kAccessRead),
                       RegionCall(b_l0, b_dst, kAccessWrite)},
                      {}, original->span));
    Call mad = original;
    auto *writer = mad.CopyOnWrite();
    writer->args.Set(0, RegionCall(a_l0, a_dst, kAccessRead));
    writer->args.Set(1, RegionCall(b_l0, b_dst, kAccessRead));
    writer->args.Set(
        2, RegionCall(gemm->c_, gemm->cRegion_->region,
                      is_one(clear_accum) ? kAccessWrite : kAccessReadWrite));
    writer->args.Set(7, IntImm(DataType::Int(32), k_extent));
    writer->args.Set(9, clear_accum);
    // Completion belongs to the whole L1 GEMM, but each intermediate sub-K
    // retains CHECK when the original operation participates in that protocol.
    if (!is_last) {
      if (auto terminal = original->annotations.Get("unit_flag_ctrl"))
        writer->annotations.Set(
            "unit_flag_ctrl",
            NonterminalUnitFlag(Downcast<PrimExpr>(terminal.value())));
    }
    return SeqStmt({a_copy, b_copy, Evaluate(mad)});
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call_node = op->value.as<CallNode>();
    if (call_node == nullptr || !call_node->op.same_as(Gemm::Get()))
      return StmtMutator::VisitStmt_(op);
    Call original = GetRef<Call>(call_node);
    Gemm gemm(original->args, original->annotations);
    if (!IsL1Buffer(gemm->a_) || !IsL1Buffer(gemm->b_))
      return StmtMutator::VisitStmt_(op);
    RequireSupportedTranspose(gemm);
    ValidateGeometry(gemm);
    if (gemm->a_->dtype != gemm->b_->dtype ||
        !(gemm->a_->dtype.is_float16() || gemm->a_->dtype.is_bfloat16()) ||
        gemm->a_->dtype.lanes() != 1 || !IsL0CBuffer(gemm->c_) ||
        gemm->c_->dtype != DataType::Float(32)) {
      TVM_FFI_THROW(ValueError)
          << "Ascend dav-2201 L1 GEMM requires FP16/BF16 operands and an "
             "FP32 L0C accumulator";
    }
    if (pending_allocations_.empty()) {
      TVM_FFI_THROW(ValueError)
          << "AscendLowerL1Gemm requires a tilelang_root allocation scope";
    }

    int64_t tile_k = SelectSubK(gemm);
    std::string suffix = std::to_string(next_gemm_++);
    Buffer a_l0 =
        decl_buffer({IntImm(DataType::Int(32), Align16(gemm->m_)),
                     IntImm(DataType::Int(32), tile_k)},
                    gemm->a_->dtype, "gemm_l0a_" + suffix, "shared.l0a");
    Buffer b_l0 = decl_buffer(
        {IntImm(DataType::Int(32), gemm->transB_ ? Align16(gemm->n_) : tile_k),
         IntImm(DataType::Int(32), gemm->transB_ ? tile_k : Align16(gemm->n_))},
        gemm->b_->dtype, "gemm_l0b_" + suffix, "shared.l0b");
    pending_allocations_.back().push_back(a_l0);
    pending_allocations_.back().push_back(b_l0);

    int64_t full_chunks = gemm->k_ / tile_k;
    int64_t tail = gemm->k_ % tile_k;
    Optional<PrimExpr> a_load_guard;
    if (physical_mn_.has_value() && full_chunks + (tail != 0) == 1) {
      const PhysicalMNContext &mn = physical_mn_.value();
      a_load_guard =
          indexmod(mn.tile, make_const(mn.tile.dtype(), mn.n_tiles)) == 0;
      // This physical A persists across all N readers of one M tile. A
      // per-tile version selector would read uninitialized slots on N > 0.
      // Common typed-region dependency analysis retains the final reader's
      // WAR before the next M/owner/epoch overwrites this single allocation.
      pending_single_versions_.back().push_back(a_l0->data);
    }
    Array<Stmt> sequence;
    if (full_chunks > 0) {
      if (full_chunks == 1) {
        sequence.push_back(MakeChunk(original, gemm, a_l0, b_l0, 0, tile_k,
                                     gemm->clearAccum_, tail == 0,
                                     a_load_guard));
      } else {
        Var sub_k("gemm_sub_k_" + suffix, DataType::Int(32));
        Stmt body = MakeChunk(original, gemm, a_l0, b_l0,
                              sub_k * make_const(sub_k.dtype(), tile_k), tile_k,
                              And(gemm->clearAccum_, sub_k == 0), false);
        if (tail == 0 && original->annotations.count("unit_flag_ctrl")) {
          // Keep terminal completion conditional without duplicating the loop
          // body or expanding the K trip count into the IR.
          SeqStmt chunk = Downcast<SeqStmt>(body);
          Evaluate mad = Downcast<Evaluate>(chunk->seq.back());
          Call mad_call = Downcast<Call>(mad->value);
          PrimExpr terminal =
              Downcast<PrimExpr>(original->annotations.at("unit_flag_ctrl"));
          mad_call.CopyOnWrite()->annotations.Set(
              "unit_flag_ctrl",
              analyzer_.Simplify(
                  Select(sub_k == make_const(sub_k.dtype(), full_chunks - 1),
                         terminal, NonterminalUnitFlag(terminal))));
          chunk.CopyOnWrite()->seq.Set(2, Evaluate(mad_call));
          body = chunk;
        }
        ffi::Map<ffi::String, ffi::Any> annotations;
        if (auto_pipeline_bound_.has_value()) {
          // Native sub-K movement remains part of the enclosing automatic
          // pipeline. Give the existing joint planner the same bounded intent
          // instead of silently fixing these internal L0 buffers to one slot.
          annotations.Set("tl.ascend_auto_pipeline", true);
          annotations.Set("num_stages", auto_pipeline_bound_.value());
        }
        sequence.push_back(For(sub_k, 0, make_const(sub_k.dtype(), full_chunks),
                               ForKind::kSerial, body, std::nullopt,
                               annotations));
      }
    }
    if (tail > 0) {
      Buffer tail_a = a_l0;
      Buffer tail_b = b_l0;
      if (full_chunks > 0) {
        const int64_t tail_k = Align16(tail);
        const int64_t bytes = gemm->a_->dtype.bits() / 8;
        if (Align16(gemm->m_) * (tile_k + tail_k) * bytes > target_.l0a_bytes ||
            Align16(gemm->n_) * (tile_k + tail_k) * bytes > target_.l0b_bytes) {
          TVM_FFI_THROW(ValueError)
              << "Ascend dav-2201 L1 GEMM full and tail L0 tiles exceed "
                 "target capacity; reduce the logical M/N tile size";
        }
        tail_a = decl_buffer({IntImm(DataType::Int(32), Align16(gemm->m_)),
                              IntImm(DataType::Int(32), tail_k)},
                             gemm->a_->dtype, "gemm_tail_l0a_" + suffix,
                             "shared.l0a");
        tail_b = decl_buffer(
            {IntImm(DataType::Int(32),
                    gemm->transB_ ? Align16(gemm->n_) : tail_k),
             IntImm(DataType::Int(32),
                    gemm->transB_ ? tail_k : Align16(gemm->n_))},
            gemm->b_->dtype, "gemm_tail_l0b_" + suffix, "shared.l0b");
        pending_allocations_.back().push_back(tail_a);
        pending_allocations_.back().push_back(tail_b);
      }
      sequence.push_back(
          MakeChunk(original, gemm, tail_a, tail_b,
                    make_const(DataType::Int(32), full_chunks * tile_k), tail,
                    full_chunks == 0 ? gemm->clearAccum_ : const_false(), true,
                    a_load_guard));
    }
    return SeqStmt::Flatten(sequence);
  }

  const AscendTargetInfo &target_;
  arith::Analyzer analyzer_;
  Optional<PrimExpr> auto_pipeline_bound_;
  const ascend::l1_gemm_mn_detail::PhysicalMNLoops &physical_mn_loops_;
  std::optional<PhysicalMNContext> physical_mn_;
  std::vector<std::vector<Buffer>> pending_allocations_;
  std::vector<std::vector<Var>> pending_single_versions_;
  uint64_t next_gemm_{0};
};

} // namespace

tvm::transform::Pass AscendLowerL1Gemm() {
  auto pass_func = [](PrimFunc func, IRModule, PassContext) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.has_value()) {
      TVM_FFI_THROW(ValueError) << "AscendLowerL1Gemm requires a bound target";
    }
    const AscendTargetInfo &info = GetAscendTargetInfo(target.value());
    if (info.architecture != AscendArchitecture::kDav2201)
      return func;
    ascend::l1_gemm_mn_detail::PhysicalMNGemmLowerer physical_tiles(info);
    Stmt body = physical_tiles(func->body);
    L1GemmLowerer lowerer(info, physical_tiles.physical_mn_loops());
    func.CopyOnWrite()->body = lowerer(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendLowerL1Gemm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendLowerL1Gemm", AscendLowerL1Gemm);
}

} // namespace tl
} // namespace tvm
