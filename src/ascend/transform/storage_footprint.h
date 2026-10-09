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

/*! \file storage_footprint.h
 *  \brief Shared physical storage size and slot alignment for Ascend.
 */
#pragma once

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>

#include "ascend/transform/attr.h"
#include "layout/layout.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
namespace ascend {

using CompilerWorkspaceStorageSet =
    std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

inline ffi::Array<tirx::Var>
CompilerWorkspaceStorageList(const tirx::SBlock &root) {
  auto annotation = root->annotations.Get(attr::kCompilerWorkspaceStorages);
  if (!annotation.has_value())
    return {};
  if (auto storages = annotation.value().try_cast<ffi::Array<tirx::Var>>())
    return storages.value();
  // Public allocator calls use a mergeable map in the script builder. The
  // handoff boundary publishes the same canonical list as generated storage.
  ffi::Array<tirx::Var> storages;
  for (const auto &[storage, enabled] :
       annotation.value().cast<ffi::Map<tirx::Var, PrimExpr>>()) {
    ICHECK(tirx::is_one(enabled)) << "Invalid compiler workspace declaration";
    storages.push_back(storage);
  }
  return storages;
}

// Provenance is checked against the allocation owner once at each scheduling
// contract boundary. Names and the first accessed Buffer view are not owners.
inline CompilerWorkspaceStorageSet
GetCompilerWorkspaceStorages(const tirx::SBlock &root) {
  CompilerWorkspaceStorageSet result;
  for (const tirx::Var &storage : CompilerWorkspaceStorageList(root)) {
    bool has_owner = false;
    for (const tirx::Buffer &buffer : root->alloc_buffers) {
      has_owner |= buffer->data.same_as(storage) && buffer.scope() == "global";
    }
    ICHECK(has_owner) << "Ascend compiler workspace " << storage->name_hint
                      << " must have a root global allocation owner";
    result.insert(storage);
  }
  return result;
}

inline void CheckCompilerWorkspaceVersionBytes(const tirx::Var &storage,
                                               int64_t slot_bytes,
                                               int versions) {
  ICHECK_GE(slot_bytes, 0);
  ICHECK_GT(versions, 0);
  // The framework owns a one-dimensional DLPack byte Tensor, whose extent is
  // int64. This is a representation bound, not an estimate of GM capacity.
  ICHECK(slot_bytes == 0 ||
         versions <= std::numeric_limits<int64_t>::max() / slot_bytes)
      << "Ascend compiler workspace " << storage->name_hint
      << " versioned byte size overflows int64";
}

// The same address alignment governs allocator starts and version-slot starts.
// UB alignment was previously supplied only by MergeUBAllocations' caller.
inline int AscendScopeAlignBytes(const std::string &scope) {
  if (scope == "shared" || scope == "shared.dyn")
    return 32;
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return 1024;
  if (scope == "shared.l1" || scope == "shared.l1.dyn" ||
      scope == "shared.l0a" || scope == "shared.l0a.dyn" ||
      scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return 512;
  return 0;
}

inline int AscendScopeAlignBytes(const tirx::AllocBufferNode *alloc) {
  return alloc ? AscendScopeAlignBytes(GetPtrStorageScope(alloc->buffer->data))
               : 0;
}

inline int64_t PackedBytesFromElems(int64_t elements, int elem_bits) {
  return (elem_bits % 8 == 0) ? elements * (elem_bits / 8)
                              : (elements * elem_bits + 7) / 8;
}

inline PrimExpr PackedBytesFromElems(PrimExpr elements, int elem_bits) {
  if (elem_bits % 8 == 0)
    return elements * tirx::make_const(elements.dtype(), elem_bits / 8);
  return tirx::FloorDiv(elements *
                                tirx::make_const(elements.dtype(), elem_bits) +
                            tirx::make_const(elements.dtype(), 7),
                        tirx::make_const(elements.dtype(), 8));
}

inline PrimExpr PackedElemsFromBytes(PrimExpr bytes, int elem_bits) {
  return tirx::FloorDiv(bytes * tirx::make_const(bytes.dtype(), 8),
                        tirx::make_const(bytes.dtype(), elem_bits));
}

inline PrimExpr AlignPrimExpr(const PrimExpr &value, int alignment) {
  if (alignment <= 1)
    return value;
  DataType dtype = value.dtype();
  ICHECK(dtype.is_int() || dtype.is_uint())
      << "Expected integer dtype for alignment, but got " << dtype;
  PrimExpr align = tirx::make_const(dtype, alignment);
  return indexdiv(value + tirx::make_const(dtype, alignment - 1), align) *
         align;
}

inline PrimExpr ShapeElements(const ffi::Array<PrimExpr> &shape,
                              DataType dtype = DataType::Int(64)) {
  PrimExpr elements = tirx::make_const(dtype, 1);
  for (const PrimExpr &extent : shape)
    elements =
        elements * (extent.dtype() == dtype ? extent : cast(dtype, extent));
  return elements;
}

// This is an allocation footprint, not a region's element count. Explicit
// outer strides include their trailing padding, as FlattenBuffer does. The
// final padding is conservative; it does not authorize accessing extra lanes.
inline PrimExpr GetBufferSizeBytes(const tirx::Buffer &buffer,
                                   DataType result_dtype) {
  ICHECK(buffer->axis_separators.empty())
      << "Ascend storage footprint requires one physical address space: "
      << buffer->name;
  PrimExpr elements = ShapeElements(buffer->shape);
  if (!buffer->strides.empty()) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size());
    ICHECK(!buffer->shape.empty());
    elements = cast(DataType::Int(64), buffer->strides[0]) *
               cast(DataType::Int(64), buffer->shape[0]);
  }
  int bits = buffer->dtype.bits() * buffer->dtype.lanes();
  return cast(result_dtype, PackedBytesFromElems(elements, bits));
}

inline PrimExpr GetBufferSizeBytes(const tirx::AllocBufferNode *alloc,
                                   DataType result_dtype) {
  return GetBufferSizeBytes(alloc->buffer, result_dtype);
}

// Key by allocation storage, never by the first read/write view. Layout output
// is already the established physical shape; aliases only provide additional
// physical footprint information. No new layout inference takes place here.
// The maximum also preserves an allocation owner's reserved trailing storage.
class StorageFootprintCollector : public tirx::StmtExprVisitor {
public:
  static ffi::Map<tirx::Var, PrimExpr>
  SlotBytes(const tirx::SBlock &root,
            const ffi::Map<tirx::Var, int> &manual_versions = {}) {
    StorageFootprintCollector collector;
    collector(root);
    ffi::Map<tirx::Var, PrimExpr> slots;
    arith::Analyzer analyzer;
    for (const auto &[data, bytes] : collector.allocations_) {
      PrimExpr physical = bytes;
      if (auto layout = collector.layout_bytes_.Get(data))
        physical = max(physical, layout.value());
      int alignment = AscendScopeAlignBytes(GetPtrStorageScope(data));
      PrimExpr footprint = AlignPrimExpr(physical, alignment);
      // Manual UB with an explicit leading version axis is realigned by the
      // late version-layout pass. Charge every slot, including its padding,
      // rather than rounding the whole compact ring only once. Count-only
      // annotations on a flat allocation preserve the user's physical layout.
      if (auto versions = manual_versions.Get(data);
          versions.has_value() && versions.value() > 1 && alignment == 32) {
        bool carries_versions = false;
        for (const tirx::Buffer &buffer : collector.owners_[data]) {
          if (!buffer->shape.empty() &&
              analyzer.CanProveEqual(buffer->shape[0], versions.value()))
            carries_versions = true;
        }
        if (carries_versions) {
          PrimExpr count = tirx::make_const(physical.dtype(), versions.value());
          PrimExpr per_slot = indexdiv(physical + count - 1, count);
          footprint = AlignPrimExpr(per_slot, alignment) * count;
        }
      }
      slots.Set(data, analyzer.Simplify(footprint));
    }
    return slots;
  }

private:
  void VisitBufferDef(const tirx::Buffer &buffer, bool alloc_data) final {
    if (alloc_data) {
      PrimExpr size = GetBufferSizeBytes(buffer, DataType::Int(64));
      if (auto previous = allocations_.Get(buffer->data))
        size = max(previous.value(), size);
      allocations_.Set(buffer->data, size);
      auto owners =
          owners_.Get(buffer->data).value_or(ffi::Array<tirx::Buffer>{});
      owners.push_back(buffer);
      owners_.Set(buffer->data, owners);
    }
    StmtExprVisitor::VisitBufferDef(buffer, alloc_data);
  }

  void VisitStmt_(const tirx::SBlockNode *op) final {
    if (auto value = op->annotations.Get(tl::attr::kLayoutMap)) {
      for (const auto &[buffer, layout] :
           value.value().cast<ffi::Map<tirx::Buffer, Layout>>()) {
        PrimExpr bytes =
            PackedBytesFromElems(ShapeElements(layout->OutputShape()),
                                 buffer->dtype.bits() * buffer->dtype.lanes());
        if (auto previous = layout_bytes_.Get(buffer->data))
          bytes = max(previous.value(), bytes);
        layout_bytes_.Set(buffer->data, bytes);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  ffi::Map<tirx::Var, PrimExpr> allocations_;
  ffi::Map<tirx::Var, ffi::Array<tirx::Buffer>> owners_;
  ffi::Map<tirx::Var, PrimExpr> layout_bytes_;
};

} // namespace ascend
} // namespace tl
} // namespace tvm
