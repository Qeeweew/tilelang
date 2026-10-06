// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include "c_api/asc_simd.h"
#include <cstdint>

#if __NPU_ARCH__ == 2201

// Predicate storage and generation are explicit in TIR. This primitive owns
// the nonescaping CMPMASK setup/consume sequence, as one atomic V task. The
// installed CANN9.1 API requires V ordering after loading CMPMASK. A prefix
// selection issues at most one VL per call; a uniform all-zero/all-one mask
// may be reused by a larger count. COUNTER=0 is a documented native NOP.
template <typename T>
__aicore__ inline void
tl_ascend_membase_select_bits(__ubuf__ T *dst, __ubuf__ T *src0,
                              __ubuf__ T *src1, __ubuf__ uint64_t *predicate,
                              uint32_t count) {
  asc_set_cmp_mask(reinterpret_cast<__ubuf__ void *>(predicate));
  pipe_barrier(pipe_t::PIPE_V);
  asc_select(dst, src0, src1, count);
}

#endif
