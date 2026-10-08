// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file lower_vector_tiles.cc
 * \brief Expose 2201 MemBase arithmetic, scratch, and memory effects before
 *        scheduling. Each intrinsic is one native vector operation.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <iterator>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aiv_private.h"
#include "ascend/op/copy.h"
#include "ascend/op/oob_padding.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/storage_footprint.h"
#include "layout/layout.h"
#include "mixed_kernel_intent.h"
#include "op/builtin.h"
#include "op/fill.h"
#include "op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::GetRef;

// Keep the BufferLoad-based tl.access_ptr operands until LowerAccessPtr. The
// scheduler can therefore see source/destination regions and temporary UB
// lifetimes without reconstructing effects from emitted C++.
#define REGISTER_MEMBASE_OP(Name, Inputs)                                      \
  TVM_REGISTER_OP("tl.ascend_membase." #Name)                                  \
      .set_num_inputs(Inputs)                                                  \
      .set_attr<TCallEffectKind>("TCallEffectKind",                            \
                                 Integer(CallEffectKind::kOpaque))             \
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "asc_" #Name)

REGISTER_MEMBASE_OP(add, 4);
REGISTER_MEMBASE_OP(sub, 4);
REGISTER_MEMBASE_OP(mul, 4);
REGISTER_MEMBASE_OP(div, 4);
REGISTER_MEMBASE_OP(max, 4);
REGISTER_MEMBASE_OP(min, 4);
REGISTER_MEMBASE_OP(exp, 3);
REGISTER_MEMBASE_OP(duplicate, 3);
REGISTER_MEMBASE_OP(half2float, 3);
REGISTER_MEMBASE_OP(bfloat162float, 3);
REGISTER_MEMBASE_OP(float2half_rn, 3);
REGISTER_MEMBASE_OP(float2half_rd, 3);
REGISTER_MEMBASE_OP(float2half_ru, 3);
REGISTER_MEMBASE_OP(float2half_rz, 3);
REGISTER_MEMBASE_OP(float2bfloat16_rn, 3);
REGISTER_MEMBASE_OP(float2bfloat16_rd, 3);
REGISTER_MEMBASE_OP(float2bfloat16_ru, 3);
REGISTER_MEMBASE_OP(float2bfloat16_rz, 3);
REGISTER_MEMBASE_OP(add_scalar, 4);
REGISTER_MEMBASE_OP(sub_scalar, 4);
REGISTER_MEMBASE_OP(mul_scalar, 4);
REGISTER_MEMBASE_OP(max_scalar, 4);
REGISTER_MEMBASE_OP(min_scalar, 4);
#undef REGISTER_MEMBASE_OP

// These operations retain typed FP32 regions. Only native code generation
// bitcasts the single Brcb instruction's operands to uint32_t pointers.
TVM_REGISTER_OP("tl.ascend_membase.brcb_f32")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.ascend_membase.sub_rows_f32")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TVM_REGISTER_OP("tl.ascend_membase.mul_rows_f32")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.ascend_membase.div_rows_f32")
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.ascend_membase.select_bits")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl_ascend_membase_select_bits");

TVM_REGISTER_OP("tl.ascend_membase.copy_bits")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "tl_ascend_membase_copy_bits");

namespace {

// Storage liveness for destructive expression destinations. This is
// deliberately before vector lowering: a native instruction may overwrite one
// of its inputs only when no later statement can observe the old value. Full
// tile copies and fills are strong definitions; a scalar/partial store is not.
// Loop backedges and both control-flow arms participate, so lexical last use
// alone is never a certificate for an accumulator inside a recurring pipeline
// epoch.
class VectorExpressionLastUse {
public:
  using StorageSet = std::unordered_set<const VarNode *>;

  void Analyze(const Stmt &body) {
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<SBlockNode>())
        for (const Buffer &buffer : block->alloc_buffers)
          if (IsSharedBuffer(buffer)) {
            local_.insert(buffer->data.get());
            local_buffers_.emplace(buffer->data.get(), buffer);
          }
    });
    PostOrderVisit(body, [&](const ObjectRef &node) {
      Optional<Buffer> view;
      if (const auto *load = node.as<BufferLoadNode>())
        view = load->buffer;
      if (const auto *store = node.as<BufferStoreNode>())
        view = store->buffer;
      if (view.has_value()) {
        auto owner = local_buffers_.find(view.value()->data.get());
        if (owner != local_buffers_.end() &&
            !owner->second.same_as(view.value()))
          aliased_definitions_.insert(owner->first);
      }
    });
    EscapeCollector escapes(&local_, &escaped_);
    escapes(body);
    Analyze(body, {}, true);
  }

  const StorageSet &DeadInputs(const BufferStoreNode *store) const {
    auto found = dead_inputs_.find(store);
    return found == dead_inputs_.end() ? empty_ : found->second;
  }

private:
  class EscapeCollector : public StmtExprVisitor {
  public:
    EscapeCollector(const StorageSet *local, StorageSet *escaped)
        : local_(local), escaped_(escaped) {}

  private:
    void MarkPointerStorages(const PrimExpr &expr) {
      PostOrderVisit(expr, [&](const ObjectRef &node) {
        if (const auto *load = node.as<BufferLoadNode>();
            load && local_->count(load->buffer->data.get()))
          escaped_->insert(load->buffer->data.get());
      });
    }

    void VisitExpr_(const VarNode *op) final {
      if (local_->count(op))
        escaped_->insert(op);
    }

    void VisitExpr_(const CallNode *op) final {
      // Compiler-owned regions/copies do not retain pointers. Unknown handle
      // consumers may do so even if there is no later syntactic buffer read.
      bool known = op->op.same_as(tl::access_ptr()) ||
                   op->op.same_as(tl::region()) || IsAscendCopyCall(op) ||
                   op->op.same_as(Fill::Get());
      if (!known) {
        for (const PrimExpr &arg : op->args) {
          if (arg.dtype().is_handle() || op->dtype.is_handle())
            MarkPointerStorages(arg);
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const BindNode *op) final {
      if (op->value.dtype().is_handle())
        MarkPointerStorages(op->value);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      if (op->value.dtype().is_handle())
        MarkPointerStorages(op->value);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitStmt_(const SBlockNode *op) final {
      // Match-buffer aliases have a different data Var. This narrow analysis
      // must not infer that the source allocation is dead from that identity.
      for (const MatchBufferRegion &match : op->match_buffers)
        if (local_->count(match->source->buffer->data.get()))
          escaped_->insert(match->source->buffer->data.get());
      StmtExprVisitor::VisitStmt_(op);
    }

    const StorageSet *local_;
    StorageSet *escaped_;
  };

  static void AddReads(const ObjectRef &node, StorageSet *live) {
    PostOrderVisit(node, [&](const ObjectRef &child) {
      if (const auto *load = child.as<BufferLoadNode>())
        live->insert(load->buffer->data.get());
      if (const auto *var = child.as<VarNode>())
        live->insert(var);
    });
  }

  static void Union(const StorageSet &other, StorageSet *live) {
    live->insert(other.begin(), other.end());
  }

  bool FullDefinition(const Buffer &buffer, const Array<Range> &region) {
    auto allocation = local_buffers_.find(buffer->data.get());
    // A full write of a smaller same-data view does not define all values of
    // the allocation. Only the actual local allocation is a strong kill.
    if (allocation == local_buffers_.end() ||
        aliased_definitions_.count(buffer->data.get()) ||
        !allocation->second.same_as(buffer))
      return false;
    if (region.size() != buffer->shape.size())
      return false;
    for (size_t i = 0; i < region.size(); ++i)
      if (!analyzer_.CanProveEqual(region[i]->min, 0) ||
          !analyzer_.CanProveEqual(region[i]->extent, buffer->shape[i]))
        return false;
    return true;
  }

  StorageSet AnalyzeLoop(const Stmt &body, const StorageSet &after,
                         bool record) {
    StorageSet live = after;
    while (true) {
      StorageSet next = Analyze(body, live, false);
      Union(after, &next); // The loop may have zero iterations.
      if (next == live)
        break;
      live = std::move(next);
    }
    if (record)
      Analyze(body, live, true);
    return live;
  }

  StorageSet Analyze(const Stmt &stmt, StorageSet live, bool record) {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (size_t i = seq->seq.size(); i-- > 0;)
        live = Analyze(seq->seq[i], std::move(live), record);
      return live;
    }
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      StorageSet skipped = live;
      live = Analyze(realize->block, std::move(live), record);
      if (!is_one(realize->predicate))
        Union(skipped, &live);
      AddReads(realize->predicate, &live);
      for (const PrimExpr &value : realize->iter_values)
        AddReads(value, &live);
      return live;
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      live = Analyze(block->body, std::move(live), record);
      // Reduction init can be conditional on reduction iterators. It is not
      // an unconditional definition at every block realization.
      if (block->init.has_value())
        AddReads(block->init.value(), &live);
      return live;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      // Parallel lanes become one dense native region. The lowerer separately
      // checks that this region has no repeated/broadcast source coordinates.
      if (loop->step.has_value())
        AddReads(loop->step.value(), &live);
      if (loop->kind == ForKind::kParallel || is_one(loop->extent)) {
        live = Analyze(loop->body, std::move(live), record);
      } else {
        AddReads(loop->min, &live);
        AddReads(loop->extent, &live);
        live = AnalyzeLoop(loop->body, live, record);
      }
      AddReads(loop->min, &live);
      AddReads(loop->extent, &live);
      return live;
    }
    if (const auto *loop = stmt.as<WhileNode>()) {
      AddReads(loop->condition, &live);
      return AnalyzeLoop(loop->body, live, record);
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      StorageSet otherwise =
          branch->else_case.has_value()
              ? Analyze(branch->else_case.value(), live, record)
              : live;
      live = Analyze(branch->then_case, std::move(live), record);
      Union(otherwise, &live);
      AddReads(branch->condition, &live);
      return live;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      live = Analyze(attr->body, std::move(live), record);
      AddReads(attr->value, &live);
      return live;
    }
    if (const auto *store = stmt.as<BufferStoreNode>()) {
      if (record) {
        StorageSet reads;
        AddReads(store->value, &reads);
        StorageSet dead;
        for (const VarNode *data : reads)
          if (local_.count(data) && !escaped_.count(data) && !live.count(data))
            dead.insert(data);
        auto [found, inserted] = dead_inputs_.emplace(store, dead);
        if (!inserted)
          for (auto it = found->second.begin(); it != found->second.end();)
            it = dead.count(*it) ? std::next(it) : found->second.erase(it);
      }
      AddReads(store->value, &live);
      for (const PrimExpr &index : store->indices)
        AddReads(index, &live);
      if (store->predicate.has_value())
        AddReads(store->predicate.value(), &live);
      return live;
    }
    if (const auto *evaluate = stmt.as<EvaluateNode>()) {
      const auto *call = evaluate->value.as<CallNode>();
      if (IsAscendCopyCall(call)) {
        Copy copy(call->args, call->annotations);
        if (FullDefinition(copy->dst, copy->dst_range))
          live.erase(copy->dst->data.get());
        live.insert(copy->src->data.get());
        for (const Range &range : copy->src_range) {
          AddReads(range->min, &live);
          AddReads(range->extent, &live);
        }
        for (const Range &range : copy->dst_range) {
          AddReads(range->min, &live);
          AddReads(range->extent, &live);
        }
        for (const auto &[key, value] : call->annotations)
          if (value.as<PrimExprNode>())
            AddReads(value, &live);
        return live;
      }
      if (call && call->op.same_as(Fill::Get())) {
        Fill fill(call->args, call->annotations);
        if (FullDefinition(fill->dst, fill->region))
          live.erase(fill->dst->data.get());
        AddReads(fill->value, &live);
        for (const Range &range : fill->region) {
          AddReads(range->min, &live);
          AddReads(range->extent, &live);
        }
        for (const auto &[key, value] : call->annotations)
          if (value.as<PrimExprNode>())
            AddReads(value, &live);
        return live;
      }
    }
    // Binds, unknown operators and unsupported control forms cannot establish
    // a strong definition. Reading their entire subtree is conservative.
    AddReads(stmt, &live);
    return live;
  }

  arith::Analyzer analyzer_;
  StorageSet local_, escaped_, aliased_definitions_, empty_;
  std::unordered_map<const VarNode *, Buffer> local_buffers_;
  std::unordered_map<const BufferStoreNode *, StorageSet> dead_inputs_;
};

class VectorTileLowerer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "AscendLowerVectorTiles requires a bound target";
    if (GetAscendTargetInfo(target.value()).architecture !=
        AscendArchitecture::kDav2201)
      return func;

    VectorTileLowerer lowerer;
    lowerer.last_use_.Analyze(func->body);
    PrimFuncNode *writer = func.CopyOnWrite();
    writer->body = lowerer.VisitStmt(func->body);
    ICHECK(lowerer.scratch_.empty())
        << "AscendLowerVectorTiles requires a root allocation block";
    return func;
  }

private:
  bool IsPrivateBody(const Stmt &body) const {
    if (aiv_private_.empty())
      return false;
    bool found = false, complete = true;
    auto buffer = [&](const Buffer &buffer) {
      if (IsSharedBuffer(buffer)) {
        found = true;
        complete &= ascend::ContainsAivPrivateData(aiv_private_, buffer->data);
      }
    };
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>())
        buffer(load->buffer);
      if (const auto *store = node.as<BufferStoreNode>())
        buffer(store->buffer);
    });
    return found && complete;
  }

  class PrivateOperationScope {
  public:
    PrivateOperationScope(bool *state, bool value)
        : state_(state), saved_(*state) {
      *state_ = value;
    }
    ~PrivateOperationScope() { *state_ = saved_; }

  private:
    bool *state_;
    bool saved_;
  };

  void RegisterScratch(const Buffer &buffer) {
    scratch_.push_back(buffer);
    if (private_operation_)
      ascend::AppendAivPrivateData(&aiv_private_, buffer->data);
  }

  struct Value {
    Optional<BufferLoad> tile;
    PrimExpr scalar;
  };
  // Buffer.data is a definition in TVM structural equality. Two independent
  // same-shaped buffers can therefore compare structurally equal. Expression
  // CSE must additionally preserve the identities of the storage it reads.
  class ReadStorageCollector : public ExprVisitor {
  public:
    std::vector<const VarNode *> data;

  private:
    void VisitExpr_(const BufferLoadNode *op) final {
      data.push_back(op->buffer->data.get());
      ExprVisitor::VisitExpr_(op);
    }
  };
  struct ValueHash {
    size_t operator()(const PrimExpr &expr) const {
      size_t hash = ffi::StructuralHash()(expr);
      ReadStorageCollector collector;
      collector(expr);
      for (const VarNode *data : collector.data) {
        size_t identity = std::hash<const VarNode *>()(data);
        hash ^= identity + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };
  struct ValueEqual {
    bool operator()(const PrimExpr &a, const PrimExpr &b) const {
      ReadStorageCollector lhs, rhs;
      lhs(a);
      rhs(b);
      return lhs.data == rhs.data && ffi::StructuralEqual()(a, b);
    }
  };
  using ValueMap = std::unordered_map<PrimExpr, Value, ValueHash, ValueEqual>;

  static Stmt MakeSeq(const Array<Stmt> &stmts) {
    if (stmts.empty())
      return Evaluate(0);
    return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
  }

  void Require(bool condition, const std::string &reason) const {
    if (!condition)
      TVM_FFI_THROW(ValueError) << "dav-2201 MemBase Parallel: " << reason;
  }

  void CheckBuffer(const Buffer &buffer) const {
    Require(IsSharedBuffer(buffer),
            "operands must use private UB (shared/shared.dyn), got " +
                std::string(buffer->name) + " in " +
                std::string(buffer.scope()));
    Require(buffer->dtype == DataType::Float(16) ||
                buffer->dtype == DataType::Float(32) ||
                buffer->dtype.is_bfloat16(),
            "MemBase storage supports float16/bfloat16/float32; explicit "
            "conversion is "
            "required for " +
                std::string(buffer->name));
  }

  PrimExpr FlatOffset(const BufferLoad &load, bool physical) {
    Array<PrimExpr> indices = load->indices;
    Array<PrimExpr> shape = load->buffer->shape;
    if (physical) {
      if (auto layout = ascend::FindLayoutForBuffer(layouts_, load->buffer)) {
        Require(layout.value()->InputDim() == indices.size(),
                "layout rank must match the UB operand");
        indices = layout.value()->Forward(indices);
        shape = layout.value()->OutputShape();
      }
    }
    if (!physical || !ascend::FindLayoutForBuffer(layouts_, load->buffer)) {
      Array<PrimExpr> offsets = load->buffer->ElemOffset(indices);
      Require(offsets.size() == 1, "axis-separated UB storage is unsupported");
      return offsets[0];
    }
    PrimExpr offset = make_zero(indices[0].dtype());
    for (size_t i = 0; i < indices.size(); ++i)
      offset = offset * shape[i] + indices[i];
    return offset + load->buffer->elem_offset;
  }

  bool Invariant(const PrimExpr &expr) {
    // Simplify has already normalized the frontend. Uniformity is a local
    // variable-use question, not another global proof over a math expression.
    return !UsesVar(expr, [&](const VarNode *var) {
      return base_map_.count(GetRef<Var>(var)) != 0;
    });
  }

  bool DenseOrScalar(const BufferLoad &load, bool allow_scalar) {
    PrimExpr logical = FlatOffset(load, false);
    if (allow_scalar && Invariant(logical))
      return true;
    PrimExpr physical = FlatOffset(load, true);
    return analyzer_.CanProveEqual(logical, Substitute(logical, base_map_) +
                                                linear_index_) &&
           analyzer_.CanProveEqual(physical, Substitute(physical, base_map_) +
                                                 linear_index_);
  }

  BufferLoad Base(const BufferLoad &load) {
    Array<PrimExpr> indices;
    for (const PrimExpr &index : load->indices)
      indices.push_back(Substitute(index, base_map_));
    return BufferLoad(load->buffer, indices);
  }

  void CheckAligned(const BufferLoad &base) {
    PrimExpr offset = FlatOffset(base, true);
    int alignment = 32 / (base->buffer->dtype.bits() / 8);
    Require(analyzer_.CanProveEqual(
                floormod(offset, make_const(offset.dtype(), alignment)),
                make_zero(offset.dtype())),
            "vector operand starts must be 32-byte aligned in physical UB "
            "storage; use an aligned row layout for row broadcasts");
  }

  void CheckInBounds(const BufferLoad &load) {
    Require(!load->predicate.has_value() || is_one(load->predicate.value()),
            "predicated loads need an explicit supported vector mask");
    for (size_t i = 0; i < load->indices.size(); ++i) {
      Require(analyzer_.CanProve(load->indices[i] >= 0) &&
                  analyzer_.CanProve(load->indices[i] < load->buffer->shape[i]),
              "the vector region must lie within " +
                  std::string(load->buffer->name) +
                  "; clamp the Parallel domain or pad the UB tile");
    }
  }

  BufferLoad NewScratch(DataType dtype, int64_t elements = -1) {
    if (elements < 0)
      elements = count_;
    PrimExpr bytes = ascend::PackedBytesFromElems(
        make_const(DataType::Int(64), elements), dtype.bits() * dtype.lanes());
    PrimExpr capacity = analyzer_.Simplify(ascend::PackedElemsFromBytes(
        ascend::AlignPrimExpr(bytes,
                              ascend::AscendScopeAlignBytes("shared.dyn")),
        dtype.bits() * dtype.lanes()));
    Buffer buffer = decl_buffer(
        {capacity}, dtype, "membase_tmp_" + std::to_string(next_temporary_++),
        "shared.dyn");
    RegisterScratch(buffer);
    return BufferLoad(buffer, {make_const(DataType::Int(32), 0)});
  }

  PrimExpr Ptr(const BufferLoad &load, int mask,
               Optional<PrimExpr> elements = std::nullopt) {
    return Call(DataType::Handle(), tl::access_ptr(),
                {load, elements.has_value() ? elements.value() : active_count_,
                 make_const(DataType::Int(32), mask)});
  }

  void Emit(const char *name, Array<PrimExpr> args, bool append_count = true) {
    // Count-mode arithmetic permits exact in-place operands. A shifted
    // overlapping source is not an in-place operation and would introduce
    // instruction-order-dependent reads within the vector tile.
    BufferLoad dst = Downcast<BufferLoad>(Downcast<Call>(args[0])->args[0]);
    for (size_t i = 1; i < args.size(); ++i) {
      const auto *ptr = args[i].as<CallNode>();
      if (!ptr || (!ptr->op.same_as(tl::access_ptr()) &&
                   !ptr->op.same_as(tl::region())))
        continue;
      BufferLoad src = Downcast<BufferLoad>(ptr->args[0]);
      if (!src->buffer->data.same_as(dst->buffer->data))
        continue;
      // Offsets and spans are in bits: cast operands have different element
      // widths even when their tl.access_ptr counts are equal.
      int dst_bits = dst->buffer->dtype.bits() * dst->buffer->dtype.lanes();
      int src_bits = src->buffer->dtype.bits() * src->buffer->dtype.lanes();
      PrimExpr a = cast(DataType::Int(64), FlatOffset(dst, true)) *
                   make_const(DataType::Int(64), dst_bits);
      PrimExpr b = cast(DataType::Int(64), FlatOffset(src, true)) *
                   make_const(DataType::Int(64), src_bits);
      bool exact_in_place = dst->buffer->dtype == src->buffer->dtype &&
                            analyzer_.CanProveEqual(a, b);
      Require(exact_in_place ||
                  analyzer_.CanProve(
                      a + make_const(a.dtype(), count_ * dst_bits) <= b) ||
                  analyzer_.CanProve(
                      b + make_const(b.dtype(), count_ * src_bits) <= a),
              "shifted overlapping source/destination regions are "
              "unsupported; use a separate UB tile");
    }
    if (append_count)
      args.push_back(cast(DataType::UInt(32), active_count_));
    emitted_.push_back(Evaluate(
        Call(DataType::Void(),
             Op::Get(std::string("tl.ascend_membase.") + name), args)));
  }

  void EmitBitCopy(const BufferLoad &dst, const BufferLoad &src) {
    Require(dst->buffer->dtype == src->buffer->dtype,
            "UB bit copy requires matching element types; use T.cast for "
            "datatype conversion");
    CheckAligned(src);
    if (dst->buffer->data.same_as(src->buffer->data) &&
        analyzer_.CanProveEqual(FlatOffset(dst, true), FlatOffset(src, true)))
      return;
    // Keep typed regions in TIR. Only the one-instruction native primitive
    // reinterprets FP16/BF16 as one UInt16 lane or FP32 as two UInt16 lanes.
    // Emit also checks shifted overlap against the bounded active capacity.
    Emit("copy_bits", {Ptr(dst, kAccessWrite), Ptr(src, kAccessRead)});
  }

  Value Scalar(const PrimExpr &expr) {
    PrimExpr value = Substitute(expr, base_map_);
    bool reads_memory = false;
    PostOrderVisit(value, [&](const ObjectRef &node) {
      reads_memory |= node.as<BufferLoadNode>() != nullptr;
    });
    // UB scalar loads execute on S, not V. Make their completion and the
    // scalar-to-vector value dependency visible to the shared task analyses.
    if (reads_memory) {
      if (analyzer_.const_int_bound(active_count_)->min_value == 0) {
        // A COUNTER=0 native vector call is a documented NOP. A uniform UB
        // broadcast load on S must likewise avoid reading inactive data, while
        // the unconditional Bind still provides its task/completion site.
        value = Select(active_count_ > make_zero(active_count_.dtype()), value,
                       make_zero(value.dtype()));
      }
      Var scalar("membase_scalar_" + std::to_string(next_temporary_++),
                 value.dtype());
      emitted_.push_back(Bind(scalar, value));
      value = scalar;
    }
    return {std::nullopt, value};
  }

  Value MaterializeScalar(Value value, DataType dtype) {
    if (value.tile.defined())
      return value;
    BufferLoad dst = NewScratch(dtype);
    Emit("duplicate", {Ptr(dst, kAccessWrite), cast(dtype, value.scalar)});
    return {dst, PrimExpr()};
  }

  // Compact row state is proved against logical and physical storage.
  bool IsAligned32(const BufferLoad &base) {
    return analyzer_.CanProveEqual(
        floormod(FlatOffset(base, true),
                 make_const(FlatOffset(base, true).dtype(),
                            32 / (base->buffer->dtype.bits() / 8))),
        0);
  }

  bool IsRowBroadcast(const BufferLoad &load) {
    if (batch_rows_ == 0 || load->buffer->dtype != DataType::Float(32))
      return false;
    PrimExpr row = batch_row_var_.value() - batch_row_min_;
    PrimExpr logical = FlatOffset(load, false);
    PrimExpr physical = FlatOffset(load, true);
    return analyzer_.CanProveEqual(logical,
                                   Substitute(logical, base_map_) + row) &&
           analyzer_.CanProveEqual(physical,
                                   Substitute(physical, base_map_) + row);
  }

  // A strided native row instruction consumes a real logical rectangle.
  // Keep that rectangle through scheduling and version materialization; only
  // AscendLowerTileOp converts it to a pointer after effects have been used.
  bool IsRowMatrix(const BufferLoad &load) {
    if (load->indices.size() != 2 || !batch_row_var_.has_value() ||
        !batch_column_var_.has_value())
      return false;
    BufferLoad base = Base(load);
    return analyzer_.CanProveEqual(load->indices[0],
                                   base->indices[0] + batch_row_var_.value() -
                                       batch_row_min_) &&
           analyzer_.CanProveEqual(
               load->indices[1], base->indices[1] + batch_column_var_.value() -
                                     batch_column_min_) &&
           analyzer_.CanProveEqual(
               load->buffer->shape[1],
               make_const(load->buffer->shape[1].dtype(), batch_columns_));
  }

  BufferLoad NewRowScratch() {
    Buffer buffer = decl_buffer(
        {make_const(DataType::Int(32), batch_rows_),
         make_const(DataType::Int(32), batch_columns_)},
        DataType::Float(32), "membase_tmp_" + std::to_string(next_temporary_++),
        "shared.dyn");
    RegisterScratch(buffer);
    return BufferLoad(buffer, {make_const(DataType::Int(32), 0),
                               make_const(DataType::Int(32), 0)});
  }

  PrimExpr RowChunkRegion(const BufferLoad &base, int mask, int64_t column) {
    ICHECK_EQ(base->indices.size(), 2U);
    Array<PrimExpr> indices = base->indices;
    indices.Set(1, indices[1] + make_const(indices[1].dtype(), column));
    return Call(DataType::Handle(), tl::region(),
                {BufferLoad(base->buffer, indices),
                 make_const(DataType::Int(32), mask),
                 make_const(indices[0].dtype(), batch_rows_),
                 make_const(indices[1].dtype(), 64)});
  }

  struct RowBinary {
    BufferLoad matrix;
    BufferLoad row;
    const char *name;
    bool row_lhs;
  };

  std::optional<RowBinary> MatchRowBinary(const PrimExpr &expr) {
    if (batch_rows_ == 0 || expr.dtype() != DataType::Float(32))
      return std::nullopt;
    PrimExpr lhs, rhs;
    const char *name;
    if (const auto *sub = expr.as<SubNode>()) {
      lhs = sub->a;
      rhs = sub->b;
      name = "sub_rows_f32";
    } else if (const auto *mul = expr.as<MulNode>()) {
      lhs = mul->a;
      rhs = mul->b;
      name = "mul_rows_f32";
    } else if (const auto *div = expr.as<DivNode>()) {
      lhs = div->a;
      rhs = div->b;
      name = "div_rows_f32";
    } else {
      return std::nullopt;
    }
    const auto *a = lhs.as<BufferLoadNode>();
    const auto *b = rhs.as<BufferLoadNode>();
    if (!a || !b || a->buffer->dtype != DataType::Float(32) ||
        b->buffer->dtype != DataType::Float(32))
      return std::nullopt;
    BufferLoad aa = GetRef<BufferLoad>(a), bb = GetRef<BufferLoad>(b);
    if (IsRowBroadcast(aa) && DenseOrScalar(bb, false) && IsRowMatrix(bb))
      return RowBinary{bb, aa, name, true};
    if (IsRowBroadcast(bb) && DenseOrScalar(aa, false) && IsRowMatrix(aa))
      return RowBinary{aa, bb, name, false};
    return std::nullopt;
  }

  Value LowerRowBinary(const RowBinary &binary, Optional<BufferLoad> output) {
    BufferLoad matrix = Base(binary.matrix);
    BufferLoad row = Base(binary.row);
    auto found = row_blocks_.find(binary.row);
    BufferLoad blocks;
    PrimExpr rows = make_const(DataType::Int(32), batch_rows_);
    PrimExpr block_elements = make_const(DataType::Int(32), batch_rows_ * 8);
    if (found == row_blocks_.end()) {
      blocks = NewScratch(DataType::Float(32), batch_rows_ * 8);
      // One repeat reads eight compact FP32 values and writes eight 32B
      // blocks. R is a multiple of eight; neither the read nor write is rounded
      // beyond the declared regions. The ordinary V RAW dependency supplies
      // completion before a row-binary consumer.
      Emit("brcb_f32",
           {Ptr(blocks, kAccessWrite, block_elements),
            Ptr(row, kAccessRead, rows),
            make_const(DataType::UInt(8), batch_rows_ / 8)},
           false);
      row_blocks_.emplace(binary.row, blocks);
    } else {
      blocks = found->second;
    }
    BufferLoad dst = output.has_value() ? output.value() : NewRowScratch();
    PrimExpr row_region = Call(
        DataType::Handle(), tl::region(),
        {blocks, make_const(DataType::Int(32), kAccessRead), block_elements});
    // Each instruction processes 64 columns of all owned rows. Its exact
    // multidimensional effects exclude the gaps between chunks and never
    // describe a fictitious contiguous R*C span from a nonzero column origin.
    for (int64_t column = 0; column < batch_columns_; column += 64) {
      PrimExpr matrix_region = RowChunkRegion(matrix, kAccessRead, column);
      Emit(binary.name,
           {RowChunkRegion(dst, kAccessWrite, column),
            binary.row_lhs ? row_region : matrix_region,
            binary.row_lhs ? matrix_region : row_region,
            make_const(DataType::UInt(8), batch_rows_),
            make_const(DataType::Bool(), binary.row_lhs),
            make_const(DataType::UInt(8), batch_columns_ / 8)},
           false);
    }
    return {dst, PrimExpr()};
  }

  bool CollectRowAssignments(Stmt body, std::vector<BufferStore> *stores) {
    while (const auto *realize = body.as<SBlockRealizeNode>()) {
      if (!is_one(realize->predicate) || !realize->iter_values.empty() ||
          !realize->block->alloc_buffers.empty() ||
          !realize->block->match_buffers.empty() ||
          !realize->block->annotations.empty() ||
          realize->block->init.defined())
        return false;
      body = realize->block->body;
    }
    if (const auto *sequence = body.as<SeqStmtNode>()) {
      for (const Stmt &statement : sequence->seq)
        if (!CollectRowAssignments(statement, stores))
          return false;
      return true;
    }
    const auto *store = body.as<BufferStoreNode>();
    if (!store ||
        (store->predicate.has_value() && !is_one(store->predicate.value())))
      return false;
    stores->push_back(GetRef<BufferStore>(store));
    return true;
  }

  std::optional<Stmt> TryCoalescedRows(const ForNode *op) {
    // Mixed ownership turns its row Parallel into a private Serial loop. Only
    // unconditional independent row/column assignments can be flattened here;
    // arbitrary serial controls and cross-row state are never reordered.
    if (mixed_cube_ && !row_partitioned_ && !private_operation_)
      return std::nullopt;
    if (op->kind != ForKind::kParallel &&
        !(op->kind == ForKind::kSerial && row_partitioned_))
      return std::nullopt;
    const auto *column = op->body.as<ForNode>();
    if (!column || column->kind != ForKind::kParallel)
      return std::nullopt;
    arith::ConstIntBound row_bound = analyzer_.const_int_bound(op->extent);
    arith::ConstIntBound column_bound =
        analyzer_.const_int_bound(column->extent);
    int64_t rows = row_bound->min_value;
    int64_t columns = column_bound->min_value;
    if (rows <= 0 || rows != row_bound->max_value || columns <= 0 ||
        columns != column_bound->max_value ||
        rows > std::numeric_limits<int32_t>::max() / columns)
      return std::nullopt;
    Stmt body = column->body;
    std::vector<BufferStore> stores;
    if (!CollectRowAssignments(body, &stores) || stores.empty())
      return std::nullopt;

    analyzer_.Bind(column->loop_var,
                   Range::FromMinExtent(column->min, column->extent));
    Map<Var, PrimExpr> saved_base = base_map_;
    PrimExpr saved_linear = linear_index_, saved_active = active_count_;
    int64_t saved_count = count_, saved_rows = batch_rows_;
    int64_t saved_columns = batch_columns_;
    Optional<Var> saved_var = batch_row_var_;
    PrimExpr saved_min = batch_row_min_;
    Optional<Var> saved_column_var = batch_column_var_;
    PrimExpr saved_column_min = batch_column_min_;
    bool saved_dense_reuse = dense_reuse_region_;
    auto restore = [&]() {
      base_map_ = saved_base;
      linear_index_ = saved_linear;
      active_count_ = saved_active;
      count_ = saved_count;
      batch_rows_ = saved_rows;
      batch_columns_ = saved_columns;
      batch_column_var_ = saved_column_var;
      batch_column_min_ = saved_column_min;
      batch_row_var_ = saved_var;
      batch_row_min_ = saved_min;
      dense_reuse_region_ = saved_dense_reuse;
    };
    base_map_.clear();
    base_map_.Set(op->loop_var, op->min);
    base_map_.Set(column->loop_var, column->min);
    linear_index_ = (op->loop_var - op->min) * column->extent +
                    (column->loop_var - column->min);
    active_count_ = make_const(DataType::Int(32), rows * columns);
    count_ = rows * columns;
    // Dense independent assignments need only the ordinary count-mode
    // contract. Compact row operands additionally require Brcb's exact
    // eight-row input blocks and 64-lane normal binary chunks. The row repeat
    // stride is in 32B blocks and must fit the native uint8_t operand.
    batch_rows_ =
        columns % 64 == 0 && columns / 8 <= 255 && rows % 8 == 0 && rows <= 255
            ? rows
            : 0;
    batch_columns_ = columns;
    batch_row_var_ = op->loop_var;
    batch_row_min_ = op->min;
    batch_column_var_ = column->loop_var;
    batch_column_min_ = column->min;

    std::vector<BufferLoad> reads, writes;
    std::unordered_set<const BufferLoadNode *> row_reads;
    bool predicates = false;
    for (const BufferStore &store : stores) {
      writes.push_back(BufferLoad(store->buffer, store->indices));
      PostOrderVisit(store->value, [&](const ObjectRef &node) {
        if (const auto *load = node.as<BufferLoadNode>())
          reads.push_back(GetRef<BufferLoad>(load));
        if (const auto *expr = node.as<PrimExprNode>()) {
          if (auto binary = MatchRowBinary(GetRef<PrimExpr>(expr)))
            row_reads.insert(binary->row.get());
        }
        if (node.as<SelectNode>())
          predicates = true;
        if (const auto *call = node.as<CallNode>();
            call && call->op.same_as(Op::Get("tirx.if_then_else")))
          predicates = true;
      });
    }
    bool valid = !predicates;
    // Replacing per-lane statement order by whole-tile statement order is
    // safe only for independent lanes. Same-storage matrix views therefore
    // require identical typed physical starts; shifted/disjoint views fall
    // back instead of adding another alias-range contract here.
    auto exact_or_distinct = [&](const BufferLoad &a, const BufferLoad &b) {
      return !a->buffer->data.same_as(b->buffer->data) ||
             (a->buffer->dtype == b->buffer->dtype &&
              analyzer_.CanProveEqual(FlatOffset(Base(a), true),
                                      FlatOffset(Base(b), true)));
    };
    for (const BufferLoad &write : writes) {
      valid &= DenseOrScalar(write, false) && IsAligned32(Base(write));
      if (!row_reads.empty())
        valid &= IsRowMatrix(write);
      for (const BufferLoad &other : writes)
        valid &= exact_or_distinct(write, other);
    }
    for (const BufferLoad &load : reads) {
      if (IsRowBroadcast(load)) {
        valid &= row_reads.count(load.get()) != 0 && IsAligned32(Base(load));
        for (const BufferLoad &write : writes)
          valid &= !load->buffer->data.same_as(write->buffer->data);
        for (const BufferLoad &other : reads)
          if (!IsRowBroadcast(other))
            valid &= !load->buffer->data.same_as(other->buffer->data);
      } else {
        bool dense = DenseOrScalar(load, false);
        valid &= dense || DenseOrScalar(load, true);
        if (dense)
          valid &= IsAligned32(Base(load));
        for (const BufferLoad &write : writes) {
          valid &= exact_or_distinct(load, write);
          // A uniform scalar read cannot alias a matrix written by another
          // lane or a preceding whole-tile assignment.
          valid &= dense || !load->buffer->data.same_as(write->buffer->data);
        }
      }
    }
    if (!valid) {
      restore();
      return std::nullopt;
    }
    for (const BufferLoad &write : writes) {
      CheckBuffer(write->buffer);
      CheckInBounds(write);
    }
    for (const BufferLoad &load : reads) {
      CheckBuffer(load->buffer);
      CheckInBounds(load);
    }
    dense_reuse_region_ = true;
    Stmt result = LowerBody(body);
    restore();
    return result;
  }

  // Intersections/unions remain one scalar Min/Max cutoff, rather than a
  // per-lane condition tree. Other predicates have a single boundary
  // diagnostic.
  Optional<PrimExpr> PrefixThreshold(const PrimExpr &condition) {
    PrimExpr normalized = analyzer_.Simplify(condition);
    if (Invariant(normalized))
      return Select(normalized, active_count_,
                    make_zero(active_count_.dtype()));
    if (const auto *op = normalized.as<AndNode>()) {
      auto a = PrefixThreshold(op->a), b = PrefixThreshold(op->b);
      if (a.has_value() && b.has_value())
        return analyzer_.Simplify(min(a.value(), b.value()));
      return std::nullopt;
    }
    if (const auto *op = normalized.as<OrNode>()) {
      auto a = PrefixThreshold(op->a), b = PrefixThreshold(op->b);
      if (a.has_value() && b.has_value())
        return analyzer_.Simplify(max(a.value(), b.value()));
      return std::nullopt;
    }
    // Python '&'/'|' preserve bool operands as bitwise Calls in upstream TIR.
    // For bool dtype they have the same prefix intersection/union semantics
    // as logical And/Or; integer bit masks remain outside this contract.
    if (const auto *op = normalized.as<CallNode>();
        op && op->dtype.is_bool() && op->args.size() == 2 &&
        (op->op.same_as(Op::Get("tirx.bitwise_and")) ||
         op->op.same_as(Op::Get("tirx.bitwise_or")))) {
      auto a = PrefixThreshold(op->args[0]), b = PrefixThreshold(op->args[1]);
      if (a.has_value() && b.has_value())
        return analyzer_.Simplify(op->op.same_as(Op::Get("tirx.bitwise_and"))
                                      ? min(a.value(), b.value())
                                      : max(a.value(), b.value()));
      return std::nullopt;
    }
    if (base_map_.size() != 1)
      return std::nullopt;
    PrimExpr a, b;
    int inclusive = 0;
    if (const auto *op = normalized.as<LTNode>()) {
      a = op->a;
      b = op->b;
    } else if (const auto *op = normalized.as<LENode>()) {
      a = op->a;
      b = op->b;
      inclusive = 1;
    } else if (const auto *op = normalized.as<GTNode>()) {
      a = op->b;
      b = op->a;
    } else if (const auto *op = normalized.as<GENode>()) {
      a = op->b;
      b = op->a;
      inclusive = 1;
    } else {
      return std::nullopt;
    }
    if (!a.dtype().is_int() || !b.dtype().is_int())
      return std::nullopt;
    const auto &[var, origin] = *base_map_.begin();
    PrimExpr cutoff = analyzer_.Simplify(b - a + var - origin +
                                         make_const(a.dtype(), inclusive));
    if (!Invariant(cutoff))
      return std::nullopt;
    return cutoff;
  }

  bool SupportedPredicate(const PrimExpr &predicate) {
    PrimExpr normalized = analyzer_.Simplify(predicate);
    return Invariant(normalized) || PrefixThreshold(normalized).has_value();
  }

  Buffer NewPredicateScratch() {
    Buffer buffer = decl_buffer(
        {Integer(4)}, DataType::UInt(64),
        "membase_predicate_" + std::to_string(next_temporary_++), "shared.dyn");
    RegisterScratch(buffer); // 32 B capacity, 16 B actual CMPMASK input.
    return buffer;
  }

  PrimExpr PredicatePtr(const Buffer &buffer) {
    return Call(
        DataType::Handle(), tl::access_ptr(),
        {BufferLoad(buffer, {Integer(0)}), Integer(2), Integer(kAccessRead)});
  }

  PrimExpr PrefixWord(PrimExpr prefix, int word) {
    PrimExpr bits = analyzer_.Simplify(
        max(make_zero(prefix.dtype()),
            min(make_const(prefix.dtype(), 64),
                prefix - make_const(prefix.dtype(), word * 64))));
    PrimExpr one = make_const(DataType::UInt(64), 1);
    // Both C Select operands may be printed before the condition. Keep the
    // shift itself within [0,63], including a full 64-bit prefix.
    PrimExpr shift =
        cast(DataType::UInt(64), min(bits, make_const(bits.dtype(), 63)));
    return analyzer_.Simplify(Select(bits == make_const(bits.dtype(), 64),
                                     ~make_zero(DataType::UInt(64)),
                                     (one << shift) - one));
  }

  BufferLoad ChunkBase(const BufferLoad &base, const PrimExpr &offset) {
    Require(!base->indices.empty() &&
                analyzer_.CanProve(
                    base->buffer->shape.back() >=
                    make_const(base->buffer->shape.back().dtype(), count_)),
            "prefix select requires its vector coordinate in the trailing UB "
            "dimension; use a row Parallel expression");
    Array<PrimExpr> indices = base->indices;
    indices.Set(indices.size() - 1,
                indices.back() + cast(indices.back().dtype(), offset));
    BufferLoad result(base->buffer, indices);
    CheckAligned(result);
    return result;
  }

  Value LowerSelect(const PrimExpr &condition, const PrimExpr &true_value,
                    const PrimExpr &false_value, Optional<BufferLoad> output) {
    DataType dtype = true_value.dtype();
    Require(dtype == false_value.dtype() &&
                (dtype == DataType::Float(16) || dtype == DataType::Float(32)),
            "native predicate selection supports float16/float32; promote "
            "BF16 values explicitly for this slice");
    PrimExpr predicate = analyzer_.Simplify(condition);
    bool uniform = Invariant(predicate);
    auto threshold =
        uniform ? Optional<PrimExpr>() : PrefixThreshold(predicate);
    Require(
        uniform || threshold.has_value(),
        "supported predicates are row-independent scalar conditions or "
        "affine column prefixes (including intersected tail/causal bounds)");
    Value control = Scalar(uniform ? predicate : threshold.value());
    Value a = MaterializeScalar(LowerExpr(true_value), dtype);
    Value b = MaterializeScalar(LowerExpr(false_value), dtype);
    BufferLoad dst = output.has_value() ? output.value() : NewScratch(dtype);
    Buffer mask = NewPredicateScratch();
    if (uniform) {
      PrimExpr word = Select(control.scalar, ~make_zero(DataType::UInt(64)),
                             make_zero(DataType::UInt(64)));
      emitted_.push_back(BufferStore(mask, word, {Integer(0)}));
      emitted_.push_back(BufferStore(mask, word, {Integer(1)}));
      Emit("select_bits",
           {Ptr(dst, kAccessWrite), Ptr(a.tile.value(), kAccessRead),
            Ptr(b.tile.value(), kAccessRead), PredicatePtr(mask)});
      return {dst, PrimExpr()};
    }
    int vl = 256 / (dtype.bits() / 8);
    Var chunk("membase_select_chunk_" + std::to_string(next_temporary_++),
              DataType::Int(32));
    int64_t chunks = (count_ + vl - 1) / vl;
    analyzer_.Bind(chunk, Range(0, make_const(chunk.dtype(), chunks)));
    PrimExpr offset = chunk * make_const(chunk.dtype(), vl);
    PrimExpr remaining = control.scalar - cast(control.scalar.dtype(), offset);
    Array<Stmt> chunk_body{
        BufferStore(mask, PrefixWord(remaining, 0), {Integer(0)}),
        BufferStore(mask,
                    dtype.bits() == 16 ? PrefixWord(remaining, 1)
                                       : make_zero(DataType::UInt(64)),
                    {Integer(1)})};
    PrimExpr saved_count = active_count_;
    active_count_ = analyzer_.Simplify(
        max(make_zero(saved_count.dtype()),
            min(make_const(saved_count.dtype(), vl), saved_count - offset)));
    // Reuse the normal overlap/effect emitter. Multiple VLs retain one compact
    // Serial loop, so code size does not grow with the vector extent.
    size_t before = emitted_.size();
    Emit("select_bits", {Ptr(ChunkBase(dst, offset), kAccessWrite),
                         Ptr(ChunkBase(a.tile.value(), offset), kAccessRead),
                         Ptr(ChunkBase(b.tile.value(), offset), kAccessRead),
                         PredicatePtr(mask)});
    chunk_body.push_back(emitted_.back());
    emitted_.resize(before);
    active_count_ = saved_count;
    if (chunks == 1) {
      // A single VL has no independent reuse epoch. Keep the scalar mask
      // writes and vector read in their actual enclosing owner, whose reverse
      // completion protects reuse across rows and outer loop iterations.
      emitted_.push_back(
          Substitute(MakeSeq(chunk_body), {{chunk, make_zero(chunk.dtype())}}));
    } else {
      emitted_.push_back(For(chunk, Integer(0),
                             make_const(chunk.dtype(), chunks),
                             ForKind::kSerial, MakeSeq(chunk_body)));
    }
    return {dst, PrimExpr()};
  }

  DataType ValueDType(const Value &value) const {
    return value.tile.has_value() ? value.tile.value()->buffer->dtype
                                  : value.scalar.dtype();
  }

  std::string CastRounding(const CastNode *op) {
    std::string rounding;
    for (const auto &[key, value] : op->annotations) {
      if (key == "round") {
        auto annotation = value.try_cast<PrimExpr>();
        Require(annotation.has_value(), "cast round must be a string");
        const auto *mode = annotation.value().as<StringImmNode>();
        Require(mode != nullptr, "cast round must be a string");
        rounding = mode->value;
      } else if (key == "sat") {
        auto annotation = value.try_cast<PrimExpr>();
        Require(annotation.has_value(), "cast sat must be a boolean constant");
        const auto *sat = annotation.value().as<IntImmNode>();
        // Upstream's ordinary half/bfloat casts have no satfinite operation:
        // the absent default is the ordinary datatype conversion, as in its
        // CUDA path. An explicit finite-saturation request has no lowering.
        Require(sat && sat->value == 0,
                "explicit finite-saturating cast is unsupported on dav-2201");
      } else {
        Require(
            false,
            "unsupported cast annotation (stochastic rounding is unsupported)");
      }
    }
    Require(rounding.empty() || rounding == "rn" || rounding == "rz" ||
                rounding == "rp" || rounding == "rm",
            "cast supports default/rn/rz/rp/rm; stochastic rounding is "
            "unsupported");
    return rounding;
  }

  Value Convert(Value source, DataType dtype,
                Optional<BufferLoad> output = std::nullopt,
                const std::string &rounding = "") {
    DataType from = ValueDType(source);
    if (from == dtype)
      return source;
    Require(from == DataType::Float(16) || from.is_bfloat16() ||
                from == DataType::Float(32),
            "cast source must be float16/bfloat16/float32");
    Require(dtype == DataType::Float(16) || dtype.is_bfloat16() ||
                dtype == DataType::Float(32),
            "cast destination must be float16/bfloat16/float32");
    source = MaterializeScalar(source, from);
    if (from != DataType::Float(32) && dtype != DataType::Float(32)) {
      // There is no direct 2201 half<->bfloat instruction. Exact widening is
      // a distinct native task with its own visible FP32 UB lifetime.
      source = Convert(source, DataType::Float(32));
      from = DataType::Float(32);
    }
    BufferLoad dst = output.has_value() ? output.value() : NewScratch(dtype);
    CheckAligned(dst);
    const char *name = nullptr;
    if (dtype == DataType::Float(32)) {
      name = from.is_bfloat16() ? "bfloat162float" : "half2float";
    } else {
      bool bf16 = dtype.is_bfloat16();
      if (rounding.empty() || rounding == "rn")
        name = bf16 ? "float2bfloat16_rn" : "float2half_rn";
      else if (rounding == "rm")
        name = bf16 ? "float2bfloat16_rd" : "float2half_rd";
      else if (rounding == "rp")
        name = bf16 ? "float2bfloat16_ru" : "float2half_ru";
      else
        name = bf16 ? "float2bfloat16_rz" : "float2half_rz";
    }
    Emit(name, {Ptr(dst, kAccessWrite), Ptr(source.tile.value(), kAccessRead)});
    return {dst, PrimExpr()};
  }

  Optional<BufferLoad> DestructiveCastSource(const CastNode *cast) {
    // Reuse only an FP32 arithmetic intermediate immediately consumed by a
    // narrowing cast. Conversion retains its distinct typed destination and
    // rounding point. A declared arithmetic destination (for example a DMA
    // input followed by a separate work tile) is never replaced here.
    if (!dense_reuse_region_ || !expression_reuse_allowed_ ||
        cast->value.dtype() != DataType::Float(32) ||
        (cast->dtype != DataType::Float(16) && !cast->dtype.is_bfloat16()))
      return std::nullopt;
    PrimExpr lhs, rhs;
    if (const auto *op = cast->value.as<AddNode>()) {
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = cast->value.as<SubNode>()) {
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = cast->value.as<MulNode>()) {
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = cast->value.as<DivNode>()) {
      lhs = op->a;
      rhs = op->b;
    } else {
      return std::nullopt;
    }
    for (const PrimExpr &operand : {lhs, rhs}) {
      const auto *load = operand.as<BufferLoadNode>();
      if (!load || load->buffer->dtype != DataType::Float(32))
        continue;
      const VarNode *data = load->buffer->data.get();
      // A second occurrence can require the old value in a sibling expression,
      // even when the buffer is dead after the enclosing assignment.
      if (!dead_inputs_.count(data) || expression_reads_[data] != 1)
        continue;
      BufferLoad region = GetRef<BufferLoad>(load);
      if (!DenseOrScalar(region, false))
        continue;
      // Use the exact same BufferLoad, owner and version coordinates, layout
      // and physical start. Emit checks the native same-dtype exact-overlap
      // contract against every other operand, including row broadcasts.
      return Base(region);
    }
    return std::nullopt;
  }

  class VectorComputationDetector : public ExprVisitor {
  public:
    bool found{false};

  private:
    // Address/index calculations are scalar address work, not vector value
    // operations. A uniform row-scalar UB load must remain a one-value S read.
    void VisitExpr_(const BufferLoadNode *) final {}
    void VisitExpr_(const CastNode *op) final {
      found = true;
      ExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const CallNode *op) final {
      // This is an upstream scalar constant, lowered by LowerIntrin to a
      // typed FloatImm. Do not confuse it with an unknown math operation.
      if (op->op.same_as(Op::Get("tl.infinity")) && op->args.size() == 1) {
        const auto *dtype = op->args[0].as<StringImmNode>();
        if (dtype &&
            ((dtype->value == "float16" && op->dtype == DataType::Float(16)) ||
             (dtype->value == "float32" && op->dtype == DataType::Float(32)) ||
             (dtype->value == "bfloat16" && op->dtype.is_bfloat16())))
          return;
      }
      found = true;
      ExprVisitor::VisitExpr_(op);
    }
  };

  Value LowerExpr(const PrimExpr &expr,
                  Optional<BufferLoad> output = std::nullopt) {
    // Fill adds a datatype conversion after frontend Simplify. A uniform
    // integer scalar (for example the ordinary fill value 0) belongs to S
    // expression codegen, followed by one typed native duplicate. This does
    // not provide integer lane conversion or delegate unknown calls to S.
    if (const auto *op = expr.as<CastNode>();
        op && (op->value.dtype().is_int() || op->value.dtype().is_uint()) &&
        (op->dtype == DataType::Float(16) || op->dtype == DataType::Float(32) ||
         op->dtype.is_bfloat16()) &&
        Invariant(op->value)) {
      VectorComputationDetector source_detector;
      source_detector(op->value);
      if (!source_detector.found) {
        Require(CastRounding(op).empty(),
                "uniform integer-to-float scalar conversion supports default "
                "rounding only");
        Require(!output.has_value() ||
                    output.value()->buffer->dtype == op->dtype,
                "expression dtype must match its output buffer");
        return Scalar(analyzer_.Simplify(expr));
      }
    }
    // Cast annotations are never silently delegated to scalar codegen. Math
    // calls and floating casts remain vector operations for uniform arguments.
    bool requires_vector =
        expr.dtype().is_bfloat16() &&
        (expr.as<AddNode>() || expr.as<SubNode>() || expr.as<MulNode>() ||
         expr.as<DivNode>() || expr.as<MaxNode>() || expr.as<MinNode>());
    VectorComputationDetector detector;
    detector(expr);
    requires_vector |= detector.found;
    if (!requires_vector && Invariant(expr))
      return Scalar(expr);
    if (!output.has_value()) {
      auto found = values_.find(expr);
      if (found != values_.end())
        return found->second;
    }
    if (auto binary = MatchRowBinary(expr)) {
      Value result = LowerRowBinary(binary.value(), output);
      values_.emplace(expr, result);
      return result;
    }
    if (const auto *load = expr.as<BufferLoadNode>()) {
      BufferLoad base = Base(GetRef<BufferLoad>(load));
      CheckAligned(base);
      return {base, PrimExpr()};
    }
    if (const auto *select = expr.as<SelectNode>()) {
      Value result = LowerSelect(select->condition, select->true_value,
                                 select->false_value, output);
      values_.emplace(expr, result);
      return result;
    }
    if (const auto *call = expr.as<CallNode>();
        call && call->op.same_as(Op::Get("tirx.if_then_else")) &&
        call->args.size() == 3) {
      Value result =
          LowerSelect(call->args[0], call->args[1], call->args[2], output);
      values_.emplace(expr, result);
      return result;
    }
    DataType dtype = expr.dtype();
    Require(!output.has_value() || output.value()->buffer->dtype == dtype,
            "expression dtype must match its output buffer");
    if (const auto *op = expr.as<CastNode>()) {
      Value result = Convert(LowerExpr(op->value, DestructiveCastSource(op)),
                             dtype, output, CastRounding(op));
      values_.emplace(expr, result);
      return result;
    }
    Require(dtype == DataType::Float(16) || dtype == DataType::Float(32) ||
                dtype.is_bfloat16(),
            "arithmetic supports float16/bfloat16/float32");
    const char *name = nullptr;
    const char *scalar_name = nullptr;
    PrimExpr lhs, rhs;
    bool unary_exp = false;
    if (const auto *op = expr.as<AddNode>()) {
      name = "add";
      scalar_name = "add_scalar";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = expr.as<SubNode>()) {
      name = "sub";
      scalar_name = "sub_scalar";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = expr.as<MulNode>()) {
      name = "mul";
      scalar_name = "mul_scalar";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = expr.as<DivNode>()) {
      name = "div";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = expr.as<MaxNode>()) {
      name = "max";
      scalar_name = "max_scalar";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *op = expr.as<MinNode>()) {
      name = "min";
      scalar_name = "min_scalar";
      lhs = op->a;
      rhs = op->b;
    } else if (const auto *call = expr.as<CallNode>();
               call && call->op.same_as(Op::Get("tirx.exp")) &&
               call->args.size() == 1) {
      name = "exp";
      unary_exp = true;
      lhs = call->args[0];
    } else {
      TVM_FFI_THROW(ValueError)
          << "dav-2201 MemBase Parallel: unsupported vector expression " << expr
          << "; supported operations are cast/add/sub/mul/div/exp/max/min";
    }
    // BF16 has no ordinary MemBase vadd/vmul/etc. Preserve each TIR node's
    // BF16 rounding point by widening its inputs, computing in FP32, then
    // narrowing this node's result. This is visible instruction-level IR.
    DataType compute_dtype = dtype.is_bfloat16() ? DataType::Float(32) : dtype;
    BufferLoad dst = output.has_value() && !dtype.is_bfloat16()
                         ? output.value()
                         : NewScratch(compute_dtype);
    auto operand = [&](const PrimExpr &value) {
      Value lowered = LowerExpr(value);
      Require(ValueDType(lowered) == dtype, "arithmetic input dtype mismatch");
      // Use a normal typed Cast expression as the CSE key for compiler-added
      // promotions; all consumers of the same BF16 value share its widening.
      return dtype.is_bfloat16() ? LowerExpr(Cast(compute_dtype, value))
                                 : lowered;
    };
    Value a = operand(lhs);
    if (unary_exp) {
      a = MaterializeScalar(a, compute_dtype);
      Emit(name, {Ptr(dst, kAccessWrite), Ptr(a.tile.value(), kAccessRead)});
    } else {
      Value b = operand(rhs);
      if (scalar_name && b.scalar.defined() && a.tile.has_value()) {
        Emit(scalar_name,
             {Ptr(dst, kAccessWrite), Ptr(a.tile.value(), kAccessRead),
              cast(compute_dtype, b.scalar)});
      } else if (scalar_name && a.scalar.defined() && b.tile.has_value() &&
                 std::string(name) != "sub") {
        Emit(scalar_name,
             {Ptr(dst, kAccessWrite), Ptr(b.tile.value(), kAccessRead),
              cast(compute_dtype, a.scalar)});
      } else {
        a = MaterializeScalar(a, compute_dtype);
        b = MaterializeScalar(b, compute_dtype);
        Emit(name, {Ptr(dst, kAccessWrite), Ptr(a.tile.value(), kAccessRead),
                    Ptr(b.tile.value(), kAccessRead)});
      }
    }
    Value result{dst, PrimExpr()};
    if (dtype.is_bfloat16())
      result = Convert(result, dtype, output);
    values_.emplace(expr, result);
    return result;
  }

  Stmt LowerBody(const Stmt &body) {
    if (const auto *seq = body.as<SeqStmtNode>()) {
      Array<Stmt> stmts;
      for (const Stmt &stmt : seq->seq)
        stmts.push_back(LowerBody(stmt));
      return MakeSeq(stmts);
    }
    if (const auto *realize = body.as<SBlockRealizeNode>()) {
      Require(is_one(realize->predicate) && realize->iter_values.empty(),
              "predicated or iter-bound blocks are unsupported");
      return LowerBody(realize->block);
    }
    if (const auto *block = body.as<SBlockNode>()) {
      Require(block->iter_vars.empty() && block->alloc_buffers.empty() &&
                  block->match_buffers.empty() && !block->init.defined() &&
                  block->annotations.empty(),
              "Parallel blocks with local allocation or annotations are "
              "unsupported");
      return LowerBody(block->body);
    }
    if (const auto *conditional = body.as<IfThenElseNode>()) {
      auto unwrap = [&](Stmt stmt) {
        while (const auto *realize = stmt.as<SBlockRealizeNode>()) {
          Require(is_one(realize->predicate) && realize->iter_values.empty(),
                  "lane predicates require opaque assignment blocks");
          stmt = realize->block->body;
        }
        return stmt;
      };
      Stmt then_body = unwrap(conditional->then_case);
      const auto *store = then_body.as<BufferStoreNode>();
      Require(store != nullptr,
              "lane-predicated statements require one UB assignment");
      Require(!store->predicate.has_value() || is_one(store->predicate.value()),
              "combine lane conditions into one affine prefix predicate");
      PrimExpr alternative = BufferLoad(store->buffer, store->indices);
      if (conditional->else_case.has_value()) {
        Stmt else_body = unwrap(conditional->else_case.value());
        const auto *other = else_body.as<BufferStoreNode>();
        Require(other && other->buffer.same_as(store->buffer) &&
                    ffi::StructuralEqual()(other->indices, store->indices),
                "both lane-predicate branches must assign the same UB region");
        Require(!other->predicate.has_value() ||
                    is_one(other->predicate.value()),
                "combine lane conditions into one affine prefix predicate");
        alternative = other->value;
      }
      return LowerBody(
          BufferStore(store->buffer,
                      Select(conditional->condition, store->value, alternative),
                      store->indices));
    }
    const auto *store = body.as<BufferStoreNode>();
    Require(store != nullptr,
            "Parallel bodies must contain unconditional UB assignments; "
            "conditional/strided updates need an explicit supported primitive");
    if (store->predicate.has_value() && !is_one(store->predicate.value()))
      return LowerBody(
          BufferStore(store->buffer,
                      Select(store->predicate.value(), store->value,
                             BufferLoad(store->buffer, store->indices)),
                      store->indices));
    output_dtype_ = store->buffer->dtype;
    emitted_.clear();
    values_.clear();
    row_blocks_.clear();
    dead_inputs_ = last_use_.DeadInputs(store);
    expression_reads_.clear();
    ReadStorageCollector reads;
    reads(store->value);
    for (const PrimExpr &index : store->indices)
      reads(index);
    if (store->predicate.has_value())
      reads(store->predicate.value());
    for (const VarNode *data : reads.data)
      ++expression_reads_[data];
    expression_reuse_allowed_ =
        !store->predicate.has_value() || is_one(store->predicate.value());
    PostOrderVisit(store->value, [&](const ObjectRef &node) {
      if (node.as<SelectNode>())
        expression_reuse_allowed_ = false;
      if (const auto *call = node.as<CallNode>();
          call && call->op.same_as(Op::Get("tirx.if_then_else")))
        expression_reuse_allowed_ = false;
    });
    BufferLoad dst = Base(BufferLoad(store->buffer, store->indices));
    CheckAligned(dst);
    Value result = LowerExpr(store->value, dst);
    if (result.scalar.defined()) {
      Emit("duplicate",
           {Ptr(dst, kAccessWrite), cast(output_dtype_, result.scalar)});
    } else if (!result.tile.value().same_as(dst)) {
      EmitBitCopy(dst, result.tile.value());
    }
    return MakeSeq(emitted_);
  }

  Stmt LowerUBCopy(const Copy &copy) {
    Require(
        !mixed_cube_ || row_partitioned_ ||
            (ascend::ContainsAivPrivateData(aiv_private_, copy->src->data) &&
             ascend::ContainsAivPrivateData(aiv_private_, copy->dst->data)),
        "mixed Cube/Vector kernels require explicit two-AIV partition "
        "semantics");
    CheckBuffer(copy->src);
    CheckBuffer(copy->dst);
    Require(copy->src->dtype == copy->dst->dtype,
            "UB bit copy requires matching element types; use T.cast for "
            "datatype conversion");
    for (const auto &[key, value] : copy->annotations) {
      // Coalescing is an optimization hint for SIMT copies; count-mode native
      // copy already issues the entire contiguous region. Layout and DMA
      // annotations are not interchangeable with this private-UB operation.
      Require(key == attr::kCoalescedWidth,
              "UB bit copy does not accept DMA or parallel-loop layout "
              "annotations");
    }
    auto validate_ranges = [&](const Buffer &buffer,
                               const Array<Range> &ranges) {
      Require(ranges.size() == buffer->shape.size(),
              "UB copy region rank must match its buffer");
      for (size_t i = 0; i < ranges.size(); ++i) {
        Require(CanProveAscendRangeInBounds(ranges[i]->min, ranges[i]->extent,
                                            buffer->shape[i], &analyzer_),
                "UB copy regions must lie within their private allocations; "
                "clamp the valid region or fill padding explicitly");
      }
    };
    validate_ranges(copy->src, copy->src_range);
    validate_ranges(copy->dst, copy->dst_range);
    // MakeSIMTLoop skips singleton axes. Match the remaining extents first:
    // the generic helper must not silently use a source-sized loop when a
    // destination region describes a different logical domain.
    Array<PrimExpr> src_extents, dst_extents;
    for (const Range &range : copy->src_range)
      if (!is_one(range->extent))
        src_extents.push_back(range->extent);
    for (const Range &range : copy->dst_range)
      if (!is_one(range->extent))
        dst_extents.push_back(range->extent);
    Require(src_extents.size() == dst_extents.size(),
            "UB copy source and destination must use the same region domain");
    for (size_t i = 0; i < src_extents.size(); ++i)
      Require(analyzer_.CanProveEqual(src_extents[i], dst_extents[i]),
              "UB copy source and destination must use the same region domain");
    For loop = copy->MakeSIMTLoop(&analyzer_);
    // The shared helper spells an all-singleton copy as a serial one-element
    // loop whose body does not use its iterator. Preserve that exact region
    // as a native one-element bit copy rather than a scalar UB read/broadcast.
    if (loop->kind == ForKind::kSerial && is_one(loop->extent)) {
      const auto *store = loop->body.as<BufferStoreNode>();
      ICHECK(store != nullptr);
      const auto *load = store->value.as<BufferLoadNode>();
      ICHECK(load != nullptr);
      count_ = 1;
      active_count_ = make_const(DataType::Int(32), 1);
      emitted_.clear();
      BufferLoad dst(store->buffer, store->indices);
      CheckAligned(dst);
      EmitBitCopy(dst, GetRef<BufferLoad>(load));
      return MakeSeq(emitted_);
    }
    return VisitStmt(loop);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    PrivateOperationScope private_scope(&private_operation_,
                                        IsPrivateBody(op->body));
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    if (auto batched = TryCoalescedRows(op))
      return batched.value();
    if (op->kind != ForKind::kParallel)
      return StmtExprMutator::VisitStmt_(op);
    Require(!mixed_cube_ || row_partitioned_ || private_operation_,
            "mixed Cube/Vector kernels require explicit two-AIV partition "
            "semantics, which this pure-vector slice does not yet provide");
    std::vector<const ForNode *> loops;
    std::vector<int64_t> capacities;
    Stmt body = GetRef<Stmt>(op);
    while (const auto *loop = body.as<ForNode>()) {
      if (loop->kind != ForKind::kParallel)
        break;
      arith::ConstIntBound extent = analyzer_.const_int_bound(loop->extent);
      if (extent->min_value == 0 && extent->max_value == 0)
        return Evaluate(0);
      Require(
          extent->min_value >= 0 && extent->max_value > 0 &&
              extent->max_value <= std::numeric_limits<int32_t>::max(),
          "Parallel extents require a nonnegative bounded private UB domain");
      capacities.push_back(extent->max_value);
      analyzer_.Bind(loop->loop_var,
                     Range::FromMinExtent(loop->min, loop->extent));
      loops.push_back(loop);
      body = loop->body;
    }
    std::vector<BufferLoad> reads, writes;
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>())
        reads.push_back(GetRef<BufferLoad>(load));
      if (const auto *store = node.as<BufferStoreNode>())
        writes.push_back(BufferLoad(store->buffer, store->indices));
    });
    Require(!writes.empty(), "Parallel must write a UB tile");
    for (const auto &load : reads) {
      CheckBuffer(load->buffer);
      CheckInBounds(load);
    }
    for (const auto &load : writes) {
      CheckBuffer(load->buffer);
      CheckInBounds(load);
    }
    size_t first = 0;
    bool had_dense_access = false;
    for (; first < loops.size(); ++first) {
      base_map_.clear();
      count_ = 1;
      active_count_ = make_const(DataType::Int(32), 1);
      linear_index_ = make_const(op->loop_var.dtype(), 0);
      for (size_t i = first; i < loops.size(); ++i) {
        int64_t extent = capacities[i];
        Require(count_ <= std::numeric_limits<int32_t>::max() / extent,
                "vector count exceeds the supported compile-time UB range");
        count_ *= extent;
        active_count_ =
            active_count_ * cast(DataType::Int(32), loops[i]->extent);
        linear_index_ = linear_index_ * loops[i]->extent +
                        (loops[i]->loop_var - loops[i]->min);
        base_map_.Set(loops[i]->loop_var, loops[i]->min);
      }
      bool dense = true;
      for (const auto &load : writes)
        dense &= DenseOrScalar(load, false);
      for (const auto &load : reads)
        dense &= DenseOrScalar(load, true);
      had_dense_access |= dense;
      bool predicates = true;
      PostOrderVisit(body, [&](const ObjectRef &node) {
        if (const auto *select = node.as<SelectNode>())
          predicates &= SupportedPredicate(select->condition);
        if (const auto *conditional = node.as<IfThenElseNode>())
          predicates &= SupportedPredicate(conditional->condition);
        if (const auto *store = node.as<BufferStoreNode>();
            store && store->predicate.has_value())
          predicates &= SupportedPredicate(store->predicate.value());
        if (const auto *call = node.as<CallNode>();
            call && call->op.same_as(Op::Get("tirx.if_then_else")) &&
            call->args.size() == 3)
          predicates &= SupportedPredicate(call->args[0]);
      });
      if (dense && predicates)
        break;
    }
    Require(
        first < loops.size() || !had_dense_access,
        "supported predicates are row-independent scalar conditions or "
        "affine column prefixes (including intersected tail/causal bounds)");
    Require(first < loops.size(),
            "the innermost Parallel axis must be contiguous in both logical "
            "and physical UB layout; strided/gather accesses are unsupported");
    active_count_ = analyzer_.Simplify(active_count_);
    bool saved_dense_reuse = dense_reuse_region_;
    // When outer Parallel axes need serial traversal, a source may broadcast
    // across those axes. Do not overwrite it without a cross-row last-use
    // proof; the coalesced route above already establishes a dense rectangle.
    dense_reuse_region_ = first == 0;
    Stmt result = LowerBody(body);
    dense_reuse_region_ = saved_dense_reuse;
    for (size_t i = first; i-- > 0;) {
      // One native instruction processes a contiguous row/tile. Outer row
      // traversal remains a control loop, never a scalar element-wise fallback.
      result = For(loops[i]->loop_var, loops[i]->min, loops[i]->extent,
                   ForKind::kSerial, result);
    }
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>())
        analyzer_.Bind(iv->var, Range::FromMinExtent(
                                    make_zero(op->value.dtype()), op->value));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Prior simplification may use a fill/drain guard to remove a slot modulo.
    // Retain that same scoped fact when checking the resulting UB accesses.
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case;
    {
      With<arith::ConstraintContext> context(&analyzer_, condition);
      then_case = VisitStmt(op->then_case);
    }
    Optional<Stmt> else_case;
    if (op->else_case.has_value()) {
      With<arith::ConstraintContext> context(&analyzer_, Not(condition));
      else_case = VisitStmt(op->else_case.value());
    }
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (IsAscendCopyCall(call)) {
      Copy copy(call->args, call->annotations);
      if (IsSharedBuffer(copy->src) && IsSharedBuffer(copy->dst))
        return LowerUBCopy(copy);
    }
    if (call && call->op.same_as(Fill::Get())) {
      Fill fill(call->args, call->annotations);
      if (IsSharedBuffer(fill->dst))
        return VisitStmt(fill->MakeSIMTLoop(&analyzer_));
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool root = op->name_hint == "tilelang_root";
    bool saved_mixed = mixed_cube_;
    bool saved_partitioned = row_partitioned_;
    auto saved_private = aiv_private_;
    if (root) {
      auto partition = op->annotations.Get("tl.ascend_row_partitioned");
      row_partitioned_ = partition.has_value() &&
                         partition.value().cast<Integer>()->value == 2;
      mixed_cube_ = ascend::HasAscendCubeIntent(op->body);
      aiv_private_ = ascend::ValidatedAivPrivateData(op);
    }
    LayoutMap saved = layouts_;
    if (auto map = op->annotations.Get(attr::kLayoutMap))
      layouts_ = map.value().cast<LayoutMap>();
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    if (root) {
      SBlockNode *writer = block.CopyOnWrite();
      for (const Buffer &buffer : scratch_)
        writer->alloc_buffers.push_back(buffer);
      scratch_.clear();
      if (!aiv_private_.empty())
        writer->annotations.Set(ascend::kAivPrivateValidated, aiv_private_);
    }
    layouts_ = saved;
    mixed_cube_ = saved_mixed;
    row_partitioned_ = saved_partitioned;
    if (root)
      aiv_private_ = saved_private;
    return block;
  }

  arith::Analyzer analyzer_;
  VectorExpressionLastUse last_use_;
  VectorExpressionLastUse::StorageSet dead_inputs_;
  std::unordered_map<const VarNode *, size_t> expression_reads_;
  bool dense_reuse_region_{false};
  bool expression_reuse_allowed_{false};
  LayoutMap layouts_;
  bool mixed_cube_{false};
  bool row_partitioned_{false};
  bool private_operation_{false};
  Array<Var> aiv_private_;
  Array<Buffer> scratch_;
  Array<Stmt> emitted_;
  Map<Var, PrimExpr> base_map_;
  PrimExpr linear_index_;
  PrimExpr active_count_;
  DataType output_dtype_;
  ValueMap values_;
  std::unordered_map<PrimExpr, BufferLoad, ValueHash, ValueEqual> row_blocks_;
  int64_t batch_rows_{0};
  int64_t batch_columns_{0};
  Optional<Var> batch_row_var_;
  PrimExpr batch_row_min_;
  Optional<Var> batch_column_var_;
  PrimExpr batch_column_min_;
  int64_t count_{0};
  int next_temporary_{0};
};

} // namespace

tvm::transform::Pass AscendLowerVectorTiles() {
  return tirx::transform::CreatePrimFuncPass(
      [](PrimFunc func, const IRModule &mod,
         const tvm::transform::PassContext &ctx) {
        return VectorTileLowerer::Rewrite(std::move(func));
      },
      0, "tl.AscendLowerVectorTiles", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLowerVectorTiles",
                                   AscendLowerVectorTiles);
}

} // namespace tl
} // namespace tvm
