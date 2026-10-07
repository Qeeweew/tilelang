// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
namespace ascend {

inline constexpr const char *kAivPrivate = "tl.ascend_aiv_private";
inline constexpr const char *kAivPrivateValidated =
    "tl.ascend_aiv_private_validated";

// LowerMixedPartition establishes this member fact. Subsequent vector/reduce
// lowerers may append their own UB scratch in the same execution domain; the
// user's declaration remains unchanged and is never a kernel-wide exemption.
inline ffi::Array<tirx::Var>
ValidatedAivPrivateData(const tirx::SBlockNode *block) {
  auto members = block->annotations.Get(kAivPrivateValidated);
  return members.has_value() ? members.value().cast<ffi::Array<tirx::Var>>()
                             : ffi::Array<tirx::Var>{};
}

inline bool ContainsAivPrivateData(const ffi::Array<tirx::Var> &members,
                                   const tirx::Var &data) {
  for (const tirx::Var &member : members)
    if (member.same_as(data))
      return true;
  return false;
}

inline void AppendAivPrivateData(ffi::Array<tirx::Var> *members,
                                 const tirx::Var &data) {
  if (!ContainsAivPrivateData(*members, data))
    members->push_back(data);
}

} // namespace ascend
} // namespace tl
} // namespace tvm
