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
 * \file normalize_cube_copy_footprints.cc
 * \brief Expose dav-2201 native L1-to-L0 instruction regions before scheduling.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {
using namespace tirx;
namespace {

class CubeCopyFootprintNormalizer : public arith::IRMutatorWithAnalyzer {
public:
  explicit CubeCopyFootprintNormalizer(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

private:
  using Parent = arith::IRMutatorWithAnalyzer;
  Stmt VisitStmt_(const SBlockNode *op) final {
    LayoutMap saved = layouts_;
    if (auto layout = op->annotations.Get(attr::kLayoutMap)) {
      for (const auto &[buffer, value] : layout.value().cast<LayoutMap>()) {
        if (auto previous = layouts_.Get(buffer))
          ICHECK(previous.value()->IsEqual(value.get()))
              << "Conflicting layouts for native Cube copy buffer "
              << buffer->name;
        layouts_.Set(buffer, value);
      }
    }
    Stmt result = Parent::VisitStmt_(op);
    layouts_ = std::move(saved);
    return result;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(Parent::VisitExpr_(op));
    if (!IsAscendCopyCall(call.get()))
      return call;
    AscendCopy copy(call->args, call->annotations);
    if (!IsL1Buffer(copy->src) ||
        !(IsL0ABuffer(copy->dst) || IsL0BBuffer(copy->dst)))
      return call;
    auto [source, destination] =
        ascend::Normalize2201L1ToL0Footprint(*copy.get(), layouts_, analyzer_);
    auto region_call = [](const Buffer &buffer, const Array<Range> &ranges,
                          int mask) {
      Array<PrimExpr> indices;
      for (const Range &range : ranges)
        indices.push_back(range->min);
      Array<PrimExpr> args{BufferLoad(buffer, indices), Integer(mask)};
      for (const Range &range : ranges)
        args.push_back(range->extent);
      return Call(DataType::Handle(), region(), args);
    };
    // Only the copy's access regions change. In particular the logical GEMM
    // domain and its clear/completion controls remain independent of padding.
    auto *writer = call.CopyOnWrite();
    writer->args.Set(0, region_call(copy->src, source, kAccessRead));
    writer->args.Set(1, region_call(copy->dst, destination, kAccessWrite));
    return call;
  }

  LayoutMap layouts_;
};
} // namespace

tvm::transform::Pass AscendNormalizeCubeCopyFootprints() {
  return tirx::transform::CreatePrimFuncPass(
      [](PrimFunc function, IRModule, tvm::transform::PassContext) {
        auto target = function->GetAttr<Target>(tvm::attr::kTarget);
        TVM_FFI_CHECK(target.defined(), ValueError)
            << "AscendNormalizeCubeCopyFootprints requires a bound target";
        if (GetAscendTargetInfo(target.value()).architecture !=
            AscendArchitecture::kDav2201)
          return function;
        arith::Analyzer analyzer;
        function.CopyOnWrite()->body =
            CubeCopyFootprintNormalizer(&analyzer)(function->body);
        return function;
      },
      0, "tl.AscendNormalizeCubeCopyFootprints", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.AscendNormalizeCubeCopyFootprints",
      AscendNormalizeCubeCopyFootprints);
}
} // namespace tl
} // namespace tvm
