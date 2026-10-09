// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ascend/op/register_state.cc
 * \brief Explicit preservation of kernel-invariant Cube register states.
 */
#include "register_state.h"

#include <vector>

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/gemm_blockscaled.h"

namespace tvm {
namespace tl {
namespace ascend {
using namespace tirx;
namespace {

// These neutral operations have native Ascend implementations which do not
// define either tracked state. Keep this an explicit identity list: a pipe,
// symbol, or namespace prefix is not a register-preservation contract.
bool IsPreservingNeutralOperation(const Op &op) {
  static const std::vector<Op> operations{Op::Get("tl.tileop.fill"),
                                          Op::Get("tl.tileop.reduce"),
                                          Op::Get("tl.tileop.finalize_reducer"),
                                          sync_warp(),
                                          ballot_sync(),
                                          ballot(),
                                          activemask(),
                                          warp_reduce_sum(),
                                          warp_reduce_max(),
                                          warp_reduce_min(),
                                          warp_reduce_bitand(),
                                          warp_reduce_bitor(),
                                          atomic_add_elem_op(),
                                          atomic_add_ret_elem_op(),
                                          atomic_addx2_elem_op(),
                                          atomic_addx2_ret_elem_op(),
                                          atomic_addx4_elem_op(),
                                          atomic_addx4_ret_elem_op(),
                                          atomic_load_elem_op(),
                                          atomic_store_elem_op(),
                                          atomic_or_elem_op(),
                                          atomic_max_elem_op(),
                                          atomic_max_ret_elem_op(),
                                          atomic_min_elem_op(),
                                          atomic_min_ret_elem_op(),
                                          rng_init(),
                                          rng_rand(),
                                          rng_rand_float(),
                                          loop_break(),
                                          device_assert(),
                                          device_assert_with_msg()};
  for (const Op &known : operations)
    if (op.same_as(known))
      return true;
  return false;
}

Optional<Buffer> RegionBuffer(const PrimExpr &argument) {
  if (const auto *load = argument.as<BufferLoadNode>())
    return load->buffer;
  const auto *call = argument.as<CallNode>();
  if (!call || call->args.empty() ||
      (!call->op.same_as(region()) && !call->op.same_as(access_ptr())))
    return {};
  const auto *load = call->args[0].as<BufferLoadNode>();
  return load ? Optional<Buffer>(load->buffer) : Optional<Buffer>();
}

bool IsNativeL0Gemm(const CallNode *call) {
  // The whole tileop.gemm identity includes L1 library GEMMs which perform
  // their own register setup. Only its typed L0 operands select native MAD.
  if (call->args.size() < 3)
    return false;
  auto a = RegionBuffer(call->args[0]);
  auto b = RegionBuffer(call->args[1]);
  auto c = RegionBuffer(call->args[2]);
  if (!a.defined() || !b.defined() || !c.defined() || !IsL0ABuffer(a.value()) ||
      !IsL0BBuffer(b.value()) || !IsL0CBuffer(c.value()))
    return false;
  if (call->op.same_as(GemmBlockScaled::Get())) {
    if (call->args.size() != 16)
      return false;
    auto sfa = RegionBuffer(call->args[13]);
    auto sfb = RegionBuffer(call->args[14]);
    return sfa.defined() && sfb.defined() && IsL0ASFBuffer(sfa.value()) &&
           IsL0BSFBuffer(sfb.value());
  }
  return true;
}

} // namespace

RegisterStateEffect GetRegisterStateEffect(const CallNode *call,
                                           RegisterState state) {
  if (!call || call->op.same_as(builtin::call_extern()) ||
      call->op.same_as(builtin::call_pure_extern()))
    return RegisterStateEffect::kUnknown;
  const auto callee = call->op.as<Op>();
  if (!callee || call->op.same_as(ascend_gemm_l1()) ||
      call->op.same_as(ascend_blockscaled_gemm_l1()))
    return RegisterStateEffect::kUnknown;

  int bit;
  switch (state) {
  case RegisterState::kLoad3DConfig:
    if (call->op.same_as(ascend_set_load3d_config()))
      return RegisterStateEffect::kWrites;
    bit = kPreserveLoad3DConfig;
    break;
  case RegisterState::kMmadDirection:
    if (call->op.same_as(ascend_set_mmad_direction()))
      return RegisterStateEffect::kWrites;
    bit = kPreserveMmadDirection;
    break;
  default:
    return RegisterStateEffect::kUnknown;
  }

  if (call->op.same_as(Gemm::Get()) || call->op.same_as(GemmBlockScaled::Get()))
    return IsNativeL0Gemm(call) ? RegisterStateEffect::kPreserves
                                : RegisterStateEffect::kUnknown;

  static auto preservation = Op::GetAttrMap<TRegisterStatePreservation>(
      kRegisterStatePreservationAttr);
  if (preservation.count(callee.value()) &&
      (preservation[callee.value()]->value & bit) != 0)
    return RegisterStateEffect::kPreserves;

  if (call->op.same_as(region()) || call->op.same_as(access_ptr()) ||
      call->op.same_as(builtin::tvm_access_ptr()) ||
      call->op.same_as(builtin::assume()) || IsAscendCopyCall(call) ||
      IsPreservingNeutralOperation(callee.value()))
    return RegisterStateEffect::kPreserves;

  // Preserve the existing pure-IR admission. An external call marked pure
  // still has no hardware-state contract and was rejected above.
  static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  return effects.count(callee.value()) &&
                 effects[callee.value()]->value <=
                     static_cast<int>(CallEffectKind::kPure)
             ? RegisterStateEffect::kPreserves
             : RegisterStateEffect::kUnknown;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
