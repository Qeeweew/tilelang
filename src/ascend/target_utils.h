/*!
 * \file tl/ascend/target_utils.h
 * \brief Ascend target attribute helpers.
 */

#ifndef TVM_TL_ASCEND_TARGET_UTILS_H_
#define TVM_TL_ASCEND_TARGET_UTILS_H_

#include <cstdint>

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsAscend(Target target);
bool IsAscendVectorizableFP8(DataType dtype);

enum class AscendArchitecture { kDav2201, kDav3510 };

// Architectural capabilities and physical storage capacity, in bytes. These
// are legality constraints, not measured bandwidths or scheduling costs.
// UB capacity is per AIV; the paired AIVs do not share one UB allocation.
struct AscendTargetInfo {
  AscendArchitecture architecture;
  const char *arch;
  int64_t ub_bytes;
  int64_t l1_bytes;
  int64_t l0a_bytes;
  int64_t l0b_bytes;
  int64_t l0c_bytes;
  int aiv_per_aic;
  bool supports_simt_vf;
  bool supports_simd_vf;
  bool supports_mx_gemm;
  int64_t simt_vf_context_bytes;
};

// Resolve explicit arch/mcpu consistently. Both attributes, when present,
// must name the same supported architecture. A target with neither retains
// upstream's dav-3510 default. Environment selection belongs at target
// construction, so every compiler stage consumes the same fixed target.
const AscendTargetInfo &GetAscendTargetInfo(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TARGET_UTILS_H_
