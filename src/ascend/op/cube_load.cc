// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ascend/op/cube_load.cc
 * \brief Mechanical late pointer lowering of selected native Cube loads.
 */
#include "cube_load.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op_attr_types.h>

#include "ascend/op/builtin.h"
#include "ascend/op/register_state.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
using namespace tirx;

AscendCubeLoad3D::AscendCubeLoad3D(
    ffi::Array<PrimExpr> args,
    ffi::Map<ffi::String, ffi::ObjectRef> annotations) {
  ICHECK_EQ(args.size(), 4U);
  ICHECK(annotations.empty())
      << "Selected 2201 Load3Dv2 does not accept copy annotations";
  auto node = ffi::make_object<AscendCubeLoad3DNode>();
  node->src = NormalizeToBufferRegion(args[0]);
  node->dst = NormalizeToBufferRegion(args[1]);
  node->physical_rows = args[2];
  node->physical_cols = args[3];
  node->SetAccessRegions({{node->src, kAccessRead}, {node->dst, kAccessWrite}});
  data_ = std::move(node);
}

TileOperator AscendCubeLoad3DNode::Clone() const {
  return AscendCubeLoad3D(ffi::make_object<AscendCubeLoad3DNode>(*this));
}

Stmt AscendCubeLoad3DNode::Lower(const LowerArgs &args,
                                 arith::Analyzer *) const {
  ICHECK(GetAscendTargetInfo(args.target).architecture ==
         AscendArchitecture::kDav2201);
  ICHECK_GE(src->region.size(), 2U);
  ICHECK_GE(dst->region.size(), 2U);
  const bool is_l0a = IsL0ABuffer(dst->buffer);
  ICHECK(is_l0a || IsL0BBuffer(dst->buffer));
  const auto &rows = src->region[src->region.size() - 2];
  const auto &cols = src->region[src->region.size() - 1];
  // Native selection already established source NZ and destination ZA/ZN
  // contracts. Materialized versions alter only leading indices. These late
  // flat pointers are addresses, never the pre-schedule memory-effect facts.
  PrimExpr dst_ptr = MakeAccessPtrFromRegion(dst, kAccessWrite);
  PrimExpr src_base = MakeAscendLeadingDimAccessPtr(src, kAccessRead);
  return Evaluate(
      Call(DataType::Void(),
           is_l0a ? ascend_load_cbuf_to_ca_3d() : ascend_load_cbuf_to_cb_3d(),
           {dst_ptr, src_base, physical_rows, physical_cols, rows->extent,
            cols->extent, is_l0a ? cols->min : rows->min}));
}

TIR_REGISTER_TL_TILE_OP(AscendCubeLoad3D, ascend_cube_load3d)
    .set_num_inputs(4)
    .set_attr<ascend::TRegisterStatePreservation>(
        ascend::kRegisterStatePreservationAttr,
        Integer(ascend::kPreserveCubeRegisterStates))
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TVM_FFI_STATIC_INIT_BLOCK() { AscendCubeLoad3DNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
