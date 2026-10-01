#pragma once
#include "reader.hpp"
#include "cute/swizzle.hpp"

namespace quactlize::dequant {

// Exchange four codes in one uint32 cell, not four expanded BF16 results
// in four uint32 cells. Q5 keeps the selected four high bits at [19:16].
// The canonical global layout is unchanged; this is CTA-local storage.
template<KType T>
struct PackedCodes {
    static_assert(T == KType::Q4_K || T == KType::Q5_K);
    CUTLASS_HOST_DEVICE static uint32_t combine(uint16_t low, uint16_t high, int n, int k32) {
        uint32_t word = low;
        if constexpr (T == KType::Q5_K) {
            int const shift = Reader<T>::HighMap::word_slot(n, k32);
            word |= uint32_t((high >> shift) & 15) << 16;
        }
        return word;
    }
    CUTLASS_HOST_DEVICE static int code(uint32_t word, int slot) {
        int value = (word >> (4 * slot)) & 15;
        if constexpr (T == KType::Q5_K) value |= ((word >> (16 + slot)) & 1) << 4;
        return value;
    }
};

template<int TileK>
struct PackedExchangeLayout {
    static_assert(TileK == 128 || TileK == 256);
    static constexpr int kWords = TileK / 4;
    static constexpr int kGroups = TileK / 32;
    static constexpr int kCells = 32 * kWords;
    // Integer words indexed by (N, group*8+K_residue). Preserve aligned
    // four-word reads. Fold N8 cohorts into distinct banks for producers;
    // fold the second consumer row (K128), or K128 half (K256), into bit2.
    CUTLASS_HOST_DEVICE static constexpr int offset(int row, int word) {
        int address = int(cute::Swizzle<1,2,3>{}(row * kWords + word));
        if constexpr (TileK == 256) address = int(cute::Swizzle<1,2,4>{}(address));
        return address ^ (row & 24);
    }
    // float2 affine values. One producer per (N,K32), then four consumers
    // share the same value. The N8 xor separates producer cohorts.
    CUTLASS_HOST_DEVICE static constexpr int affine_offset(int row, int group) {
        return int(cute::Swizzle<2,2,(TileK == 128 ? 3 : 4)>{}(row * kGroups + group));
    }
};

} // namespace quactlize::dequant
