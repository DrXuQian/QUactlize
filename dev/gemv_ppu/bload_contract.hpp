#pragma once
#include <cstdint>

// Development-only SIMT consumer of the existing PPU0010 transposed b16
// delivery. A b32 register pairs two K residues of ONE N, not adjacent N.
// The host oracle composes these expressions with the actual CuTe B layout.
namespace q4_bload {
#if defined(__CUDACC__) || defined(__HGGCC__) || defined(__HGGC__)
#define Q4_BLOAD_HD __host__ __device__
#else
#define Q4_BLOAD_HD
#endif
Q4_BLOAD_HD constexpr unsigned word_n(unsigned lane, unsigned reg) {
    return lane / 4 + 8 * (reg / 2);
}
Q4_BLOAD_HD constexpr unsigned word_kg(unsigned lane, unsigned reg, unsigned half) {
    return 2 * (lane % 4) + 8 * (reg % 2) + half;
}
Q4_BLOAD_HD constexpr unsigned code_k(unsigned kg, unsigned nibble) {
    return (kg / 8) * 32 + kg % 8 + nibble * 8;
}
constexpr bool recipe(int stage_k, int warps) {
    return (stage_k == 256 && warps == 4) ||
           (stage_k == 512 && warps == 8) ||
           (stage_k == 1024 && warps == 8);
}
constexpr bool shape(int n, int k) {
    return (n == 512 && k == 2048) || (n == 1024 && k == 5120) ||
           (n == 4096 && (k == 2048 || k == 4096)) ||
           (n == 5120 && k == 8192) || (n == 8192 && k == 5120);
}
#undef Q4_BLOAD_HD
} // namespace q4_bload
