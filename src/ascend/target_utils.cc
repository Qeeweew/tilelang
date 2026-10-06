/*!
 * \file tl/ascend/target_utils.cc
 * \brief Ascend target attribute helpers.
 */

#include "ascend/target_utils.h"

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <string>

#include "support/check.h"

namespace tvm {
namespace tl {

bool TargetIsAscend(Target target) {
  return target.defined() && target->kind->name == "ascend";
}

bool IsAscendVectorizableFP8(DataType dtype) {
  // NOTE: E8M0 is a special type of FP8 which is not handled here.
  // We only handle FP8 types which can be represented with
  // __asc_fp8_interpretation_t here.
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

const AscendTargetInfo &GetAscendTargetInfo(Target target) {
  ICHECK(TargetIsAscend(target))
      << "Ascend architecture information requires an Ascend target";
  auto arch = target->GetAttr<ffi::String>("arch");
  auto mcpu = target->GetAttr<ffi::String>("mcpu");
  ICHECK(!arch.has_value() || !mcpu.has_value() || arch.value() == mcpu.value())
      << "Ascend target arch and mcpu must name the same architecture, got "
      << arch.value() << " and " << mcpu.value();
  std::string name = arch.has_value()   ? std::string(arch.value())
                     : mcpu.has_value() ? std::string(mcpu.value())
                                        : "dav-3510";

  // CANN 9.1 dav-2201 storage capacities. In mixed kernels each AIV owns
  // its private UB while the AIC owns L1/L0A/L0B/L0C.
  static constexpr AscendTargetInfo dav2201{AscendArchitecture::kDav2201,
                                            "dav-2201",
                                            192 * 1024,
                                            512 * 1024,
                                            64 * 1024,
                                            64 * 1024,
                                            128 * 1024,
                                            2,
                                            false,
                                            false,
                                            false,
                                            0};
  // Preserve the existing upstream dav-3510 capacities and VF reservation.
  static constexpr AscendTargetInfo dav3510{AscendArchitecture::kDav3510,
                                            "dav-3510",
                                            248 * 1024,
                                            512 * 1024,
                                            64 * 1024,
                                            64 * 1024,
                                            256 * 1024,
                                            2,
                                            true,
                                            true,
                                            true,
                                            32 * 1024};
  if (name == dav2201.arch)
    return dav2201;
  ICHECK(name == dav3510.arch)
      << "Unsupported Ascend architecture '" << name
      << "'; supported architectures are dav-2201 and dav-3510";
  return dav3510;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsAscend",
           [](Target target) { return TargetIsAscend(target); })
      .def("tl.GetAscendTargetInfo", [](Target target) {
        const AscendTargetInfo &info = GetAscendTargetInfo(target);
        ffi::Map<ffi::String, ffi::Any> result;
        result.Set("arch", ffi::String(info.arch));
        result.Set("ub_bytes", info.ub_bytes);
        result.Set("l1_bytes", info.l1_bytes);
        result.Set("l0a_bytes", info.l0a_bytes);
        result.Set("l0b_bytes", info.l0b_bytes);
        result.Set("l0c_bytes", info.l0c_bytes);
        result.Set("aiv_per_aic", info.aiv_per_aic);
        result.Set("supports_simt_vf", info.supports_simt_vf);
        result.Set("supports_simd_vf", info.supports_simd_vf);
        result.Set("supports_mx_gemm", info.supports_mx_gemm);
        result.Set("simt_vf_context_bytes", info.simt_vf_context_bytes);
        return result;
      });
}

} // namespace tl
} // namespace tvm
