// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include "ascend/op/vector_reduce.h"

#include "ascend/op/register_state.h"

#include "support/check.h"

#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
namespace ascend {

using namespace tirx;

// Normal-mask repeats retain exact [rows, valid_lanes] source regions until
// late lowering. Inactive lanes are not part of their memory effects.
TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_sum_rows_f32")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TRegisterStatePreservation>(kRegisterStatePreservationAttr,
                                          Integer(kPreserveCubeRegisterStates));
TVM_REGISTER_OP("tl.ascend_membase.repeat_reduce_max_rows_f32")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TRegisterStatePreservation>(kRegisterStatePreservationAttr,
                                          Integer(kPreserveCubeRegisterStates));

const Op &RepeatRowReduceF32Op(RepeatRowReduceF32Kind kind) {
  static const Op sum = Op::Get("tl.ascend_membase.repeat_reduce_sum_rows_f32");
  static const Op maximum =
      Op::Get("tl.ascend_membase.repeat_reduce_max_rows_f32");
  return kind == RepeatRowReduceF32Kind::kSum ? sum : maximum;
}

bool IsRepeatRowReduceF32(const CallNode *call) {
  return call &&
         (call->op.same_as(
              RepeatRowReduceF32Op(RepeatRowReduceF32Kind::kSum)) ||
          call->op.same_as(RepeatRowReduceF32Op(RepeatRowReduceF32Kind::kMax)));
}

RepeatRowReduceF32 DecodeRepeatRowReduceF32(const CallNode *call) {
  ICHECK(IsRepeatRowReduceF32(call)) << "Expected an FP32 row reduce call";
  ICHECK_EQ(call->args.size(), 5U) << "FP32 row reduce expects five operands";
  const auto *rows = call->args[2].as<IntImmNode>();
  const auto *stride = call->args[3].as<IntImmNode>();
  const auto *lanes = call->args[4].as<IntImmNode>();
  ICHECK(call->args[2].dtype() == DataType::UInt(8) &&
         (!rows || (rows->value > 0 && rows->value <= 255)))
      << "FP32 row reduce requires uint8 rows in [1, 255]";
  ICHECK(lanes && lanes->value >= 8 && lanes->value <= 64 &&
         lanes->value % 8 == 0 && call->args[4].dtype() == DataType::UInt(8))
      << "FP32 row reduce requires static uint8 valid lanes in [8, 64], "
         "in multiples of eight";
  ICHECK(stride && stride->value >= lanes->value / 8 &&
         stride->value <= 65535 && call->args[3].dtype() == DataType::UInt(16))
      << "FP32 row reduce requires a uint16 source row stride in "
         "[valid_lanes / 8, 65535]";
  ICHECK(lanes->value == 64 ? stride->value % 8 == 0
                            : stride->value == lanes->value / 8)
      << "FP32 row reduce requires a whole-VL stride for 64 lanes or "
         "a compact source row stride for partial lanes";
  const auto kind =
      call->op.same_as(RepeatRowReduceF32Op(RepeatRowReduceF32Kind::kSum))
          ? RepeatRowReduceF32Kind::kSum
          : RepeatRowReduceF32Kind::kMax;
  return {kind,          call->args[0], call->args[1], call->args[2],
          call->args[3], stride->value, lanes->value};
}

const char *RepeatRowReduceF32::NativeSymbol() const {
  // The six-argument overload uses destination repeat stride in elements and
  // source strides in 32-byte datablocks. Count-mode overloads are separate
  // Ops.
  return kind == RepeatRowReduceF32Kind::kSum
             ? "asc_repeat_reduce_sum"
             : "asc_repeat_reduce_max_only_value";
}

Call MakeRepeatRowReduceF32(RepeatRowReduceF32Kind kind,
                            const PrimExpr &destination, const PrimExpr &source,
                            const PrimExpr &rows, const PrimExpr &source_stride,
                            const PrimExpr &valid_lanes) {
  Call call(DataType::Void(), RepeatRowReduceF32Op(kind),
            {destination, source, rows, source_stride, valid_lanes});
  DecodeRepeatRowReduceF32(call.get());
  return call;
}

} // namespace ascend
} // namespace tl
} // namespace tvm
