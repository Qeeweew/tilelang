// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#pragma once

#include "c_api/asc_simd.h"
#include <cstdint>

#if __NPU_ARCH__ == 2201

// One bit-preserving native V instruction. In installed CANN9.1, UB DMA
// truncates its byte count to 32-byte blocks. UInt16 count-mode OR(x, x)
// instead preserves every source bit and honors exact FP16/BF16/FP32 tails.
// Typed read/write regions, alignment, and alias legality are established in
// TIR before scheduling; COUNTER=0 remains a native NOP with its task site.
template <typename T>
__aicore__ inline void
tl_ascend_membase_copy_bits(__ubuf__ T *dst, __ubuf__ T *src, uint32_t count) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4,
                "MemBase copy requires FP16, BF16, or FP32 storage");
  __ubuf__ uint16_t *source = reinterpret_cast<__ubuf__ uint16_t *>(src);
  constexpr uint32_t lanes = sizeof(T) / sizeof(uint16_t);
  asc_or(reinterpret_cast<__ubuf__ uint16_t *>(dst), source, source,
         count * lanes);
}

#endif
