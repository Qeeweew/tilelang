// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/ascend/op/register_state.h
 * \brief Preservation contracts for the two kernel-invariant Cube states.
 */
#ifndef TVM_TL_ASCEND_OP_REGISTER_STATE_H_
#define TVM_TL_ASCEND_OP_REGISTER_STATE_H_

#include <tvm/tirx/expr.h>

namespace tvm {
namespace tl {
namespace ascend {

enum class RegisterState { kLoad3DConfig, kMmadDirection };
enum class RegisterStateEffect { kUnknown, kPreserves, kWrites };

// An explicit property of a native operation, independent of its issue pipe.
// No property (or a missing bit) grants no preservation permission. Writes
// are classified separately, so a configuration setter is never a preserver
// of the state it defines.
using TRegisterStatePreservation = Integer;
inline constexpr const char *kRegisterStatePreservationAttr =
    "TAscendRegisterStatePreservation";
inline constexpr int kPreserveLoad3DConfig = 1 << 0;
inline constexpr int kPreserveMmadDirection = 1 << 1;
inline constexpr int kPreserveCubeRegisterStates =
    kPreserveLoad3DConfig | kPreserveMmadDirection;

RegisterStateEffect GetRegisterStateEffect(const tirx::CallNode *call,
                                           RegisterState state);

} // namespace ascend
} // namespace tl
} // namespace tvm
#endif
