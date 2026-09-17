#pragma once
#include <stdint.h>
#include "cutlass/cutlass.h"

namespace quactlize::fusion {
struct PairedN4 {
    CUTLASS_HOST_DEVICE static constexpr int gate(int n) { return (n / 4) * 8 + n % 4; }
    CUTLASS_HOST_DEVICE static constexpr int up(int n) { return gate(n) + 4; }
    CUTLASS_HOST_DEVICE static constexpr int channel(int physical_n) { return physical_n / 8 * 4 + physical_n % 4; }
    CUTLASS_HOST_DEVICE static constexpr bool is_up(int physical_n) { return (physical_n & 4) != 0; }
};

// A virtual raw GGUF matrix consumed by the existing word/metadata packer.
// No FP decode, requantization, or intermediate concatenation is needed.
struct PairedRawRows {
    uint8_t const* gate;
    uint8_t const* up;
    uint64_t row_bytes;
    int n;
    CUTLASS_HOST_DEVICE uint8_t const* operator+(uint64_t offset) const {
        uint64_t row = offset / row_bytes, column = row % (2 * uint64_t(n));
        uint64_t source_row = row / (2 * uint64_t(n)) * n + PairedN4::channel(int(column));
        return (PairedN4::is_up(int(column)) ? up : gate) + source_row * row_bytes + offset % row_bytes;
    }
    CUTLASS_HOST_DEVICE uint8_t operator[](uint64_t offset) const { return *(operator+(offset)); }
};
}
