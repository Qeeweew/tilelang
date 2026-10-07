/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ascend/transform/lower_l1_gemm_mn.h
 * \brief Pair oversized L1 GEMM and its complete GM export before K lowering.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "ascend/transform/attr.h"
#include "ascend/transform/buffer_version.h"
#include "ascend/transform/storage_footprint.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
namespace ascend {
namespace l1_gemm_mn_detail {

using namespace tirx;

using StorageSet =
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// Decode only the explicit typed transport, never a scalar BufferLoad or an
// opaque pointer. The ordinary operator constructors consume the same region.
inline std::optional<AccessRegion> TypedRegion(const PrimExpr &expression) {
  const auto *call = expression.as<CallNode>();
  if (!call || !call->op.same_as(region()) || call->args.size() < 2 ||
      !call->annotations.empty())
    return std::nullopt;
  const auto *load = call->args[0].as<BufferLoadNode>();
  const int64_t *mask = as_const_int(call->args[1]);
  if (!load || load->predicate.defined() || !mask || *mask < 1 || *mask > 3 ||
      load->indices.size() != load->buffer->shape.size() ||
      call->args.size() != load->indices.size() + 2)
    return std::nullopt;
  return NormalizeToAccessRegion(expression, kAccessReadWrite);
}

struct Access {
  const CallNode *call;
  size_t operand;
  int mask;
};

// One root-wide inventory proves that replacing this allocation removes all
// its uses. Layout keys are handled separately because they transport storage
// identity; ordinary scalar/pointer/container references are real escapes.
class KernelFacts : public StmtExprVisitor {
public:
  explicit KernelFacts(const SBlock &root) { operator()(root); }

  std::vector<Buffer> l0c_allocations;
  std::unordered_map<Var, int, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> owners;
  std::unordered_map<Var, std::vector<Buffer>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      buffers;
  std::unordered_map<Var, std::vector<Access>, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      accesses;
  std::vector<std::pair<ObjectRef, Layout>> layouts;
  BufferVersionMap fixed_versions;
  BufferVersionMap manual_versions;
  StorageSet escaped;
  bool has_match_buffer{false};
  bool unknown_metadata{false};

private:
  void BufferGeometry(const Buffer &buffer) {
    VisitBufferUse(buffer);
    for (const PrimExpr &extent : buffer->shape)
      VisitExpr(extent);
    for (const PrimExpr &stride : buffer->strides)
      VisitExpr(stride);
    VisitExpr(buffer->elem_offset);
  }

  void Metadata(const ffi::Any &value) {
    if (auto expression = value.as<PrimExpr>()) {
      VisitExpr(expression.value());
    } else if (auto buffer = value.as<Buffer>()) {
      escaped.insert(buffer.value()->data);
      BufferGeometry(buffer.value());
    } else if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      for (const auto &item : array.value())
        Metadata(item);
    } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto &[key, item] : map.value()) {
        Metadata(key);
        Metadata(item);
      }
    } else if (!value.as<String>().has_value() &&
               value.as<ObjectRef>().has_value()) {
      // Do not assume that an unfamiliar object cannot retain the removed C.
      unknown_metadata = true;
    }
  }

  void VisitBufferUse(const Buffer &buffer) final {
    auto &views = buffers[buffer->data];
    for (const Buffer &previous : views)
      if (previous.same_as(buffer))
        return;
    views.push_back(buffer);
  }

  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    VisitBufferUse(buffer);
    if (alloc_data) {
      ++owners[buffer->data];
      if (IsL0CBuffer(buffer))
        l0c_allocations.push_back(buffer);
    }
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }

  void VisitExpr_(const VarNode *op) final {
    escaped.insert(ffi::GetRef<Var>(op));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    escaped.insert(op->buffer->data);
    BufferGeometry(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    escaped.insert(op->buffer->data);
    BufferGeometry(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.as<PrimExprNode>())
      VisitExpr(Downcast<PrimExpr>(op->op));
    size_t typed = op->op.same_as(Gemm::Get()) ? 3
                   : IsAscendCopyCall(op)      ? 2
                                               : 0;
    for (size_t i = 0; i < op->args.size(); ++i) {
      std::optional<AccessRegion> access =
          i < typed ? TypedRegion(op->args[i]) : std::nullopt;
      if (!access.has_value()) {
        VisitExpr(op->args[i]);
        continue;
      }
      const BufferRegion &region = access.value().region;
      accesses[region->buffer->data].push_back(
          {op, i, access.value().access_mask});
      BufferGeometry(region->buffer);
      for (const Range &range : region->region) {
        VisitExpr(range->min);
        VisitExpr(range->extent);
      }
    }
    for (const auto &[_, value] : op->annotations)
      Metadata(value);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (const auto *axis = op->node.as<IterVarNode>()) {
      VisitExpr(axis->var);
      if (axis->dom.defined()) {
        VisitExpr(axis->dom->min);
        VisitExpr(axis->dom->extent);
      }
    } else {
      Metadata(op->node);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    has_match_buffer |= !op->match_buffers.empty();
    for (const BufferRegion &access : op->reads)
      escaped.insert(access->buffer->data);
    for (const BufferRegion &access : op->writes)
      escaped.insert(access->buffer->data);
    for (const auto &[key, value] : op->annotations) {
      if (key == attr::kLayoutMap) {
        for (const auto &[buffer, layout] :
             value.cast<Map<ObjectRef, Layout>>()) {
          layouts.emplace_back(buffer, layout);
          if (const auto *view = buffer.as<BufferNode>())
            BufferGeometry(ffi::GetRef<Buffer>(view));
          else if (!buffer.as<VarNode>())
            unknown_metadata = true;
          for (const PrimExpr &extent : layout->InputShape())
            VisitExpr(extent);
          for (const PrimExpr &index : layout->GetForwardIndex())
            VisitExpr(index);
        }
      } else {
        if (key == kBufferVersionsMap || key == kManualMultiBuffer) {
          BufferVersionMap versions = value.cast<BufferVersionMap>();
          auto &destination =
              key == kBufferVersionsMap ? fixed_versions : manual_versions;
          for (const auto &[storage, count] : versions) {
            TVM_FFI_CHECK(count > 0, ValueError)
                << "Ascend dav-2201 physical M/N GEMM requires positive "
                   "version counts";
            destination.Set(
                storage, std::max(destination.Get(storage).value_or(1), count));
          }
        }
        Metadata(value);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    for (const auto &[_, value] : op->annotations)
      Metadata(value);
    StmtExprVisitor::VisitStmt_(op);
  }
};

inline Var LayoutStorage(const ObjectRef &key) {
  if (const auto *buffer = key.as<BufferNode>())
    return buffer->data;
  if (const auto *storage = key.as<VarNode>())
    return ffi::GetRef<Var>(storage);
  return Var();
}

// This boundary precedes layout inference. Construct the existing canonical
// L0C factory's physical shape instead of casting a frontend Var-keyed map to
// an inferred Buffer-keyed map or charging only the compact logical shape.
inline std::optional<int64_t> CanonicalBytes(const Buffer &buffer,
                                             arith::Analyzer *analyzer) {
  if (buffer->shape.size() < 2 || buffer->dtype != DataType::Float(32) ||
      !buffer->axis_separators.empty() ||
      !is_zero(analyzer->Simplify(buffer->elem_offset)) ||
      (!buffer->strides.empty() &&
       buffer->strides.size() != buffer->shape.size()))
    return std::nullopt;
  // Bound the products before constructing shared footprint expressions. The
  // canonical layout pads the final two dimensions to complete 16x16 blocks.
  constexpr int64_t max_elements =
      (std::numeric_limits<int64_t>::max() - 1023) / 4;
  int64_t padded_elements = 1;
  std::vector<int64_t> extents;
  for (size_t axis = 0; axis < buffer->shape.size(); ++axis) {
    PrimExpr simplified = analyzer->Simplify(buffer->shape[axis]);
    const auto *constant = simplified.as<IntImmNode>();
    if (!constant || constant->value <= 0 ||
        constant->value > std::numeric_limits<int32_t>::max() - 15)
      return std::nullopt;
    extents.push_back(constant->value);
    int64_t extent = constant->value;
    if (axis + 2 >= buffer->shape.size())
      extent = (extent + 15) / 16 * 16;
    if (extent > max_elements / padded_elements)
      return std::nullopt;
    padded_elements *= extent;
  }
  if (!buffer->strides.empty()) {
    int64_t inner_span = 1;
    for (size_t axis = buffer->shape.size(); axis-- > 0;) {
      PrimExpr simplified = analyzer->Simplify(buffer->strides[axis]);
      const auto *stride = simplified.as<IntImmNode>();
      if (!stride || stride->value < inner_span ||
          stride->value > max_elements / extents[axis])
        return std::nullopt;
      inner_span = stride->value * extents[axis];
    }
  }
  Layout layout = MakeAscendL0CLayout(buffer);
  PrimExpr physical =
      max(GetBufferSizeBytes(buffer, DataType::Int(64)),
          PackedBytesFromElems(ShapeElements(layout->OutputShape()), 32));
  PrimExpr bytes = analyzer->Simplify(AlignPrimExpr(physical, 1024));
  const auto *constant = bytes.as<IntImmNode>();
  if (!constant || constant->value <= 0)
    return std::nullopt;
  return constant->value;
}

inline bool CanonicalLayouts(const KernelFacts &facts, const Buffer &owner) {
  Layout expected = MakeAscendL0CLayout(owner);
  for (const auto &[key, layout] : facts.layouts) {
    if (!LayoutStorage(key).same_as(owner->data))
      continue;
    if ((key.as<BufferNode>() && !key.same_as(owner)) ||
        !layout->IsEqual(expected.get()))
      return false;
  }
  return true;
}

inline const CallNode *LeafCall(const Stmt &statement) {
  if (const auto *evaluate = statement.as<EvaluateNode>())
    return evaluate->value.as<CallNode>();
  if (const auto *realize = statement.as<SBlockRealizeNode>()) {
    if (!is_one(realize->predicate) || !realize->iter_values.empty())
      return nullptr;
    return LeafCall(realize->block);
  }
  if (const auto *block = statement.as<SBlockNode>()) {
    if (block->name_hint != "CUBE" || block->init.defined() ||
        !block->alloc_buffers.empty() || !block->match_buffers.empty() ||
        !block->iter_vars.empty() || !block->reads.empty() ||
        !block->writes.empty() || !block->annotations.empty())
      return nullptr;
    return LeafCall(block->body);
  }
  return nullptr;
}

inline Stmt ReplaceLeaf(Stmt statement, Stmt replacement) {
  if (statement.as<EvaluateNode>())
    return replacement;
  if (const auto *node = statement.as<SBlockRealizeNode>()) {
    SBlockRealize result = ffi::GetRef<SBlockRealize>(node);
    result.CopyOnWrite()->block =
        Downcast<SBlock>(ReplaceLeaf(node->block, std::move(replacement)));
    return result;
  }
  SBlock result = Downcast<SBlock>(statement);
  result.CopyOnWrite()->body =
      ReplaceLeaf(result->body, std::move(replacement));
  return result;
}

inline bool OrdinaryCompletion(const CallNode *call) {
  if (auto value = call->annotations.Get("unit_flag_ctrl")) {
    const auto *flag = value.value().as<IntImmNode>();
    if (!flag || flag->value != 0)
      return false;
  }
  return true;
}

inline PrimExpr MakeRegion(const Buffer &buffer, const Array<Range> &ranges,
                           int mask) {
  Array<PrimExpr> indices;
  for (const Range &range : ranges)
    indices.push_back(range->min);
  Array<PrimExpr> args{BufferLoad(buffer, indices), Integer(mask)};
  for (const Range &range : ranges)
    args.push_back(range->extent);
  return Call(DataType::Handle(), region(), args);
}

inline Array<Range> SliceAxis(Array<Range> ranges, size_t axis,
                              const Var &index, int64_t tile) {
  const Range &original = ranges[axis];
  PrimExpr offset = cast(original->min.dtype(), index) *
                    make_const(original->min.dtype(), tile);
  ranges.Set(axis,
             Range::FromMinExtent(original->min + offset,
                                  make_const(original->extent.dtype(), tile)));
  return ranges;
}

inline std::vector<int64_t> AlignedDivisors(int64_t extent) {
  std::vector<int64_t> divisors;
  int64_t units = extent / 16;
  for (int64_t divisor = 1; divisor <= units / divisor; ++divisor) {
    if (units % divisor)
      continue;
    divisors.push_back(divisor * 16);
    if (divisor != units / divisor)
      divisors.push_back(units / divisor * 16);
  }
  return divisors;
}

class PairRewriter : public arith::IRMutatorWithAnalyzer {
public:
  PairRewriter(arith::Analyzer *analyzer, const KernelFacts &facts,
               Buffer accumulator, int64_t budget, uint64_t suffix)
      : IRMutatorWithAnalyzer(analyzer), facts_(facts),
        accumulator_(std::move(accumulator)), budget_(budget), suffix_(suffix) {
  }

  Buffer scratch;

private:
  using Parent = arith::IRMutatorWithAnalyzer;

  bool MatrixRegion(const BufferRegion &region) {
    if (region->region.size() < 2 ||
        region->region.size() != region->buffer->shape.size())
      return false;
    for (size_t axis = 0; axis + 2 < region->region.size(); ++axis)
      if (!analyzer_->CanProveEqual(region->region[axis]->extent, 1))
        return false;
    return true;
  }

  bool FullAccumulator(const BufferRegion &region) {
    if (!region->buffer.same_as(accumulator_) || !MatrixRegion(region))
      return false;
    for (size_t axis = 0; axis < region->region.size(); ++axis)
      if (!is_zero(analyzer_->Simplify(region->region[axis]->min)) ||
          !analyzer_->CanProveEqual(region->region[axis]->extent,
                                    accumulator_->shape[axis]))
        return false;
    return true;
  }

  bool Pair(const CallNode *producer, const CallNode *consumer) {
    if (!producer || !producer->op.same_as(Gemm::Get()) || !consumer ||
        !IsAscendCopyCall(consumer) || producer->args.size() != 13 ||
        consumer->args.size() != 2 || !OrdinaryCompletion(producer) ||
        !OrdinaryCompletion(consumer))
      return false;
    for (size_t i = 0; i < 3; ++i) {
      auto access = TypedRegion(producer->args[i]);
      if (!access.has_value() || (i < 2 && access->access_mask != kAccessRead))
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
      auto access = TypedRegion(consumer->args[i]);
      if (!access.has_value() ||
          access->access_mask != (i == 0 ? kAccessRead : kAccessWrite))
        return false;
    }
    Gemm gemm(producer->args, producer->annotations);
    AscendCopy copy(consumer->args, consumer->annotations);
    if (!gemm->c_.same_as(accumulator_) || !copy->src.same_as(accumulator_) ||
        !IsL1Buffer(gemm->a_) || !IsL1Buffer(gemm->b_) || gemm->transA_ ||
        gemm->a_->dtype != gemm->b_->dtype ||
        !(gemm->a_->dtype.is_float16() || gemm->a_->dtype.is_bfloat16()) ||
        gemm->a_->dtype.lanes() != 1 || !IsGlobalBuffer(copy->dst) ||
        copy->dst->dtype != DataType::Float(32) ||
        !is_one(analyzer_->Simplify(gemm->clearAccum_)) ||
        !is_zero(producer->args[10]) || !is_zero(producer->args[11]) ||
        !is_zero(producer->args[12]) || !FullAccumulator(gemm->cRegion_) ||
        !FullAccumulator(BufferRegion(copy->src, copy->src_range)) ||
        !MatrixRegion(gemm->aRegion_) || !MatrixRegion(gemm->bRegion_) ||
        !MatrixRegion(BufferRegion(copy->dst, copy->dst_range)) ||
        gemm->m_ <= 0 || gemm->n_ <= 0 || gemm->k_ <= 0 || gemm->m_ % 16 ||
        gemm->n_ % 16)
      return false;
    const auto &a = gemm->aRegion_->region;
    const auto &b = gemm->bRegion_->region;
    const auto &d = copy->dst_range;
    if (!analyzer_->CanProveEqual(
            accumulator_->shape[accumulator_->shape.size() - 2], gemm->m_) ||
        !analyzer_->CanProveEqual(accumulator_->shape.back(), gemm->n_) ||
        !analyzer_->CanProveEqual(a[a.size() - 2]->extent, gemm->m_) ||
        !analyzer_->CanProveEqual(a.back()->extent, gemm->k_) ||
        !analyzer_->CanProveEqual(b[b.size() - (gemm->transB_ ? 2 : 1)]->extent,
                                  gemm->n_) ||
        !analyzer_->CanProveEqual(b[b.size() - (gemm->transB_ ? 1 : 2)]->extent,
                                  gemm->k_) ||
        !analyzer_->CanProveEqual(d[d.size() - 2]->extent, gemm->m_) ||
        !analyzer_->CanProveEqual(d.back()->extent, gemm->n_))
      return false;
    const auto &uses = facts_.accesses.at(accumulator_->data);
    if (uses.size() != 2)
      return false;
    bool write = false, read = false;
    for (const Access &use : uses) {
      write |= use.call == producer && use.operand == 2 &&
               (use.mask == kAccessWrite || use.mask == kAccessReadWrite);
      read |=
          use.call == consumer && use.operand == 0 && use.mask == kAccessRead;
    }
    return write && read;
  }

  Stmt RewritePair(const Stmt &producer_stmt, const Stmt &consumer_stmt) {
    const CallNode *producer = LeafCall(producer_stmt);
    const CallNode *consumer = LeafCall(consumer_stmt);
    if (!Pair(producer, consumer))
      return Stmt();
    Gemm gemm(producer->args, producer->annotations);
    AscendCopy copy(consumer->args, consumer->annotations);
    int64_t tile_m = 0, tile_n = 0, area = 0;
    for (int64_t m : AlignedDivisors(gemm->m_))
      for (int64_t n : AlignedDivisors(gemm->n_)) {
        if (m > budget_ / 4 / n)
          continue;
        int64_t candidate = m * n;
        if (candidate > area || (candidate == area && m > tile_m)) {
          tile_m = m;
          tile_n = n;
          area = candidate;
        }
      }
    TVM_FFI_CHECK(area > 0, ValueError)
        << "Ascend dav-2201 physical M/N GEMM cannot fit a 16x16 L0C tile "
           "after reserving other root allocations (remaining bytes="
        << budget_ << ")";
    std::string suffix = std::to_string(suffix_);
    scratch =
        decl_buffer({Integer(tile_m), Integer(tile_n)}, DataType::Float(32),
                    "gemm_mn_l0c_" + suffix, "shared.l0c");
    Var mi("gemm_m_" + suffix, DataType::Int(32));
    Var ni("gemm_n_" + suffix, DataType::Int(32));
    Array<Range> c{Range::FromMinExtent(0, Integer(tile_m)),
                   Range::FromMinExtent(0, Integer(tile_n))};
    Call sliced = ffi::GetRef<Call>(producer);
    auto *writer = sliced.CopyOnWrite();
    writer->args.Set(
        0, MakeRegion(gemm->a_,
                      SliceAxis(gemm->aRegion_->region,
                                gemm->aRegion_->region.size() - 2, mi, tile_m),
                      kAccessRead));
    writer->args.Set(1, MakeRegion(gemm->b_,
                                   SliceAxis(gemm->bRegion_->region,
                                             gemm->bRegion_->region.size() -
                                                 (gemm->transB_ ? 2 : 1),
                                             ni, tile_n),
                                   kAccessRead));
    writer->args.Set(2, MakeRegion(scratch, c, kAccessWrite));
    writer->args.Set(5, Integer(tile_m));
    writer->args.Set(6, Integer(tile_n));
    writer->args.Set(9, const_true());
    Array<Range> destination =
        SliceAxis(copy->dst_range, copy->dst_range.size() - 2, mi, tile_m);
    destination = SliceAxis(destination, destination.size() - 1, ni, tile_n);
    Call exported = ffi::GetRef<Call>(consumer);
    exported.CopyOnWrite()->args.Set(0, MakeRegion(scratch, c, kAccessRead));
    exported.CopyOnWrite()->args.Set(
        1, MakeRegion(copy->dst, destination, kAccessWrite));
    Stmt body = SeqStmt({ReplaceLeaf(producer_stmt, Evaluate(sliced)),
                         ReplaceLeaf(consumer_stmt, Evaluate(exported))});
    body = For(ni, 0, Integer(gemm->n_ / tile_n), ForKind::kSerial, body);
    return For(mi, 0, Integer(gemm->m_ / tile_m), ForKind::kSerial, body);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> result;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      if (!blocked_ && !scratch.defined() && i + 1 < op->seq.size()) {
        Stmt rewritten = RewritePair(op->seq[i], op->seq[i + 1]);
        if (rewritten.defined()) {
          result.push_back(rewritten);
          ++i;
          continue;
        }
      }
      result.push_back(VisitStmt(op->seq[i]));
    }
    return SeqStmt::Flatten(result);
  }

  template <typename Node> Stmt Blocked(const Node *op) {
    ++blocked_;
    Stmt result = Parent::VisitStmt_(op);
    --blocked_;
    return result;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final { return Blocked(op); }
  Stmt VisitStmt_(const WhileNode *op) final { return Blocked(op); }
  Stmt VisitStmt_(const SBlockNode *op) final { return Blocked(op); }
  Stmt VisitStmt_(const SBlockRealizeNode *op) final { return Blocked(op); }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kAscendTask ||
        op->attr_key == attr::kAscendPerCoreTask ||
        op->attr_key == attr::kAscendStage ||
        op->attr_key == attr::kScheduleUnit)
      return Blocked(op);
    return Parent::VisitStmt_(op);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kSerial ||
        op->annotations.count("tl_pipeline_order") ||
        op->annotations.count("tl_pipeline_stage"))
      return Blocked(op);
    return Parent::VisitStmt_(op);
  }

  const KernelFacts &facts_;
  Buffer accumulator_;
  int64_t budget_;
  uint64_t suffix_;
  int blocked_{0};
};

class LayoutRemover : public StmtMutator {
public:
  explicit LayoutRemover(Var storage) : storage_(std::move(storage)) {}

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(StmtMutator::VisitStmt_(op));
    if (auto value = block->annotations.Get(attr::kLayoutMap)) {
      Map<ObjectRef, Layout> layouts;
      for (const auto &[key, layout] :
           value.value().cast<Map<ObjectRef, Layout>>())
        if (!LayoutStorage(key).same_as(storage_))
          layouts.Set(key, layout);
      block.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
    }
    return block;
  }
  Var storage_;
};

inline SBlock RewriteRoot(const SBlock &root, const AscendTargetInfo &target,
                          uint64_t suffix) {
  arith::Analyzer analyzer;
  Buffer oversized;
  for (const Buffer &buffer : root->alloc_buffers) {
    if (!IsL0CBuffer(buffer))
      continue;
    auto bytes = CanonicalBytes(buffer, &analyzer);
    if (!bytes.has_value() || bytes.value() <= target.l0c_bytes)
      continue;
    TVM_FFI_CHECK(!oversized.defined(), ValueError)
        << "Ascend dav-2201 physical M/N GEMM supports one oversized "
           "accumulator per kernel root";
    oversized = buffer;
  }
  if (!oversized.defined())
    return root;
  KernelFacts facts(root);
  const Var &storage = oversized->data;
  TVM_FFI_CHECK(
      !facts.has_match_buffer && !facts.unknown_metadata &&
          facts.owners[storage] == 1 && facts.buffers[storage].size() == 1 &&
          !facts.escaped.count(storage) && CanonicalLayouts(facts, oversized) &&
          oversized->strides.empty() && is_zero(oversized->elem_offset),
      ValueError)
      << "Ascend dav-2201 physical M/N GEMM requires a unique canonical "
         "L0C allocation without aliases, escapes or manual version metadata";
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      slots;
  for (const Buffer &buffer : facts.l0c_allocations) {
    if (buffer->data.same_as(storage))
      continue;
    auto bytes = CanonicalBytes(buffer, &analyzer);
    TVM_FFI_CHECK(bytes.has_value() && CanonicalLayouts(facts, buffer),
                  ValueError)
        << "Ascend dav-2201 physical M/N GEMM requires static canonical "
           "physical footprints for other root L0C allocations";
    slots[buffer->data] = std::max(slots[buffer->data], bytes.value());
  }
  int64_t budget = target.l0c_bytes;
  for (const auto &[data, bytes] : slots) {
    int count = facts.manual_versions.count(data)
                    ? 1
                    : facts.fixed_versions.Get(data).value_or(1);
    TVM_FFI_CHECK(bytes <= budget / count, ValueError)
        << "Ascend dav-2201 physical M/N GEMM has no L0C capacity after "
           "reserving other root allocations";
    budget -= bytes * count;
  }
  PairRewriter rewriter(&analyzer, facts, oversized, budget, suffix);
  Stmt body = rewriter(root->body);
  TVM_FFI_CHECK(rewriter.scratch.defined(), ValueError)
      << "Ascend dav-2201 physical M/N GEMM requires adjacent full-clear "
         "L1 GEMM and unique full L0C-to-GM export, static aligned M/N, "
         "and ordinary unguarded ownership/completion";
  SBlock result = root;
  Array<Buffer> allocations;
  for (const Buffer &buffer : root->alloc_buffers)
    if (!buffer->data.same_as(storage))
      allocations.push_back(buffer);
  allocations.push_back(rewriter.scratch);
  result.CopyOnWrite()->alloc_buffers = allocations;
  result.CopyOnWrite()->body = body;
  result = Downcast<SBlock>(LayoutRemover(storage)(result));
  // AscendLayoutInference consumes frontend storage-Var layout anchors. This
  // new allocation replaces the annotated logical C and needs its own anchor;
  // typed GEMM/export regions alone do not infer the accumulator's layout.
  Map<ObjectRef, Layout> layouts;
  if (auto value = result->annotations.Get(attr::kLayoutMap))
    layouts = value.value().cast<Map<ObjectRef, Layout>>();
  layouts.Set(rewriter.scratch->data, MakeAscendL0CLayout(rewriter.scratch));
  result.CopyOnWrite()->annotations.Set(attr::kLayoutMap, layouts);
  BufferVersionMap versions;
  if (auto value = result->annotations.Get(kBufferVersionsMap))
    versions = value.value().cast<BufferVersionMap>();
  versions.Set(rewriter.scratch->data, 1);
  result.CopyOnWrite()->annotations.Set(kBufferVersionsMap, versions);
  return result;
}

class PhysicalMNGemmLowerer : public StmtMutator {
public:
  explicit PhysicalMNGemmLowerer(const AscendTargetInfo &target)
      : target_(target) {}

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint == "tilelang_root")
      return RewriteRoot(ffi::GetRef<SBlock>(op), target_, next_root_++);
    return StmtMutator::VisitStmt_(op);
  }
  const AscendTargetInfo &target_;
  uint64_t next_root_{0};
};

} // namespace l1_gemm_mn_detail
} // namespace ascend
} // namespace tl
} // namespace tvm
