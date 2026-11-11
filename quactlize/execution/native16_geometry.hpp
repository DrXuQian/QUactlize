#pragma once
#include "../compat/reader.cuh"

namespace quactlize::execution::native16 {
// Exact PPU0010 m16n16k16 register ownership, tested against actlize traits.
// A is [M,K], B is [N,K], C is [M,N]; these are NOT Hopper's m16n8 maps.
struct Fragment {
    CUTLASS_HOST_DEVICE static constexpr int row(int lane, int v) { return lane / 4 + 8 * (v / 4); }
    CUTLASS_HOST_DEVICE static constexpr int k(int lane, int v) {
        return 2 * (lane % 4) + v % 2 + 8 * ((v / 2) % 2);
    }
    CUTLASS_HOST_DEVICE static constexpr int column(int lane, int v) { return lane % 4 + 4 * (v % 4); }
};
template <class T>
CUTLASS_HOST_DEVICE T q5_weight(uint16_t low, uint16_t high, int low_slot, int high_slot, uint32_t dm) {
    int code = ((low >> (4 * low_slot)) & 15) | (((high >> high_slot) & 1) << 4);
    float d = float(cutlass::half_t::bitcast(uint16_t(dm)));
    float minimum = float(cutlass::half_t::bitcast(uint16_t(dm >> 16)));
    return T(compat::difference(compat::product(d, float(code)), -minimum));
}
} // namespace quactlize::execution::native16
