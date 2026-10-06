// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ascend/op/cube_load.h
 * \brief Selected dav-2201 native Cube loads with explicit region effects.
 */
#ifndef TVM_TL_ASCEND_OP_CUBE_LOAD_H_
#define TVM_TL_ASCEND_OP_CUBE_LOAD_H_

#include "op/operator.h"

namespace tvm {
namespace tl {

// Created only after layout normalization and OOB padding. The regions cover
// the actual padded hardware transfer; the GEMM keeps its effective domain.
class AscendCubeLoad3DNode : public TileOperatorNode {
public:
  tirx::BufferRegion src, dst;
  PrimExpr physical_m, physical_k;

  tirx::Stmt Lower(const LowerArgs &args,
                   arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &, InferLevel) const final {
    return {};
  }
  TileOperator Clone() const final;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.AscendCubeLoad3D", AscendCubeLoad3DNode,
                                    TileOperatorNode);
  static void RegisterReflection() {
    ffi::reflection::ObjectDef<AscendCubeLoad3DNode>()
        .def_ro("src", &AscendCubeLoad3DNode::src)
        .def_ro("dst", &AscendCubeLoad3DNode::dst)
        .def_ro("physical_m", &AscendCubeLoad3DNode::physical_m)
        .def_ro("physical_k", &AscendCubeLoad3DNode::physical_k);
  }
};

class AscendCubeLoad3D : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(AscendCubeLoad3D, TileOperator,
                                             AscendCubeLoad3DNode);
  AscendCubeLoad3D(ffi::Array<PrimExpr> args,
                   ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

} // namespace tl
} // namespace tvm
#endif
