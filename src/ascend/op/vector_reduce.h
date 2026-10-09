// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ascend/op/vector_reduce.h
 * \brief Scalar ABI of dav-2201 FP32 normal-mask row reductions.
 */
#ifndef TVM_TL_ASCEND_OP_VECTOR_REDUCE_H_
#define TVM_TL_ASCEND_OP_VECTOR_REDUCE_H_

#include <cstdint>

#include <tvm/ir/op.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {
namespace ascend {

enum class RepeatRowReduceF32Kind { kSum, kMax };

const Op &RepeatRowReduceF32Op(RepeatRowReduceF32Kind kind);
bool IsRepeatRowReduceF32(const tirx::CallNode *call);

// A transient view of the existing five-operand Call, not proof metadata.
// Scalar decoding does not prove dynamic rows nonempty or validate memory.
// The late region boundary must prove the original (unnarrowed) active rows,
// guard, typed access roles, backing geometry and alignment before pointers.
struct RepeatRowReduceF32 {
  RepeatRowReduceF32Kind kind;
  PrimExpr destination;
  PrimExpr source;
  PrimExpr rows;          // uint8 repeat count, possibly dynamic
  PrimExpr source_stride; // uint16, measured in 32-byte datablocks
  int64_t source_stride_blocks;
  int64_t valid_lanes; // static uint8 FP32 lanes, in multiples of eight

  // Fixed operands of the six-argument C API overload.
  static constexpr int kDestinationRepeatStrideElements = 1;
  static constexpr int kSourceBlockStrideDatablocks = 1;

  const char *NativeSymbol() const;
  int64_t SourceRowStrideElements() const { return source_stride_blocks * 8; }
  bool UsesFullNormalMask() const { return valid_lanes == 64; }
  uint64_t NormalMaskLowBits() const {
    // A full mask must not shift a uint64 by 64.
    return UsesFullNormalMask() ? ~uint64_t(0)
                                : (uint64_t(1) << valid_lanes) - 1;
  }
};

// Recognized calls share this one operand-count/type/range/stride contract.
// Destination/source stay opaque here: regions and pointers are stage-specific.
RepeatRowReduceF32 DecodeRepeatRowReduceF32(const tirx::CallNode *call);

tirx::Call MakeRepeatRowReduceF32(RepeatRowReduceF32Kind kind,
                                  const PrimExpr &destination,
                                  const PrimExpr &source, const PrimExpr &rows,
                                  const PrimExpr &source_stride,
                                  const PrimExpr &valid_lanes);

} // namespace ascend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_OP_VECTOR_REDUCE_H_
