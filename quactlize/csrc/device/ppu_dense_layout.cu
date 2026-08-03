// Offline layout half of SCALE_FIRST x DENSE. This is deliberately separate from ppu_dense_backend.cu: the reorder
// is host code and can be built by nvcc for its local seam test, while the fpA tensor-core launcher is hgcc-only.
// Both are linked into libquactlize_ppu.so on the PPU build.
#include <cstdint>
#include <algorithm>
#include <vector>

#include "xplane_offline.hpp"

namespace {

template <int Bits>
void unpack_native(uint8_t const* packed, std::vector<uint8_t>& q, int n, int k) {
  constexpr int Mask = (1 << Bits) - 1;
  q.resize(size_t(n) * k);
  for (int col = 0; col < n; ++col)
    for (int kk = 0; kk < k; ++kk) {
      int64_t const bit = (int64_t(col) * k + kk) * Bits;
      q[size_t(kk) * n + col] = uint8_t((packed[bit >> 3] >> (bit & 7)) & Mask);
    }
}

template <int Bits>
void pack_native(std::vector<uint8_t> const& q, uint8_t* packed, int n, int k) {
  std::fill(packed, packed + size_t(n) * k * Bits / 8, uint8_t(0));
  constexpr int Mask = (1 << Bits) - 1;
  for (int col = 0; col < n; ++col)
    for (int kk = 0; kk < k; ++kk) {
      int64_t const bit = (int64_t(col) * k + kk) * Bits;
      packed[bit >> 3] |= uint8_t((q[size_t(kk) * n + col] & Mask) << (bit & 7));
    }
}

template <int LowBits, int HighBits, int TileK = 256>
int prepare(uint8_t const* low_native, uint8_t const* high_native,
            uint8_t* low_layout, uint8_t* high_layout, int n, int k) {
  // One conservative tactic for every format. At TK=256 every 1/2/4-bit plane supplies at least one 32-byte AIU
  // delivery, so F1=F2=1 and the dense launcher never needs a folded stride.
  std::vector<uint8_t> low;
  unpack_native<LowBits>(low_native, low, n, k);
  xplane::place_derived<LowBits, 64, 64, TileK, 32, 32, 1>(
      reinterpret_cast<int8_t*>(low_layout), low, n, k);
  if constexpr (HighBits != 0) {
    if (!high_native || !high_layout) return 21;
    std::vector<uint8_t> high;
    unpack_native<HighBits>(high_native, high, n, k);
    xplane::place_hi<LowBits, HighBits, 64, 64, TileK, 32, 32, 1, 1>(
        reinterpret_cast<int8_t*>(high_layout), high, n, k);
  }
  return 0;
}

template <int LowBits, int HighBits, int TileK = 256>
int recover(uint8_t const* low_layout, uint8_t const* high_layout,
            uint8_t* low_native, uint8_t* high_native, int n, int k) {
  std::vector<uint8_t> low;
  xplane::recover_derived<LowBits, 64, 64, TileK, 32, 32, 1>(
      reinterpret_cast<int8_t const*>(low_layout), low, n, k);
  pack_native<LowBits>(low, low_native, n, k);
  if constexpr (HighBits != 0) {
    if (!high_layout || !high_native) return 21;
    std::vector<uint8_t> high;
    xplane::recover_hi<LowBits, HighBits, 64, 64, TileK, 32, 32, 1, 1>(
        reinterpret_cast<int8_t const*>(high_layout), high, n, k);
    pack_native<HighBits>(high, high_native, n, k);
  }
  return 0;
}

}  // namespace

extern "C" int quactlize_ppu_prepare_dense(uint8_t const* low_native, uint8_t const* high_native,
                                             uint8_t* low_layout, uint8_t* high_layout,
                                             int n, int k, int qtype) {
  if (!low_native || !low_layout || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  switch (qtype) {
    case 10: return prepare<2, 0>(low_native, high_native, low_layout, high_layout, n, k);
    case 11: return prepare<2, 1>(low_native, high_native, low_layout, high_layout, n, k);
    case 12: return prepare<4, 0>(low_native, high_native, low_layout, high_layout, n, k);
    case 13: return prepare<4, 1>(low_native, high_native, low_layout, high_layout, n, k);
    // Q6's int2 high-plane delivery is complete at TK=128. TK=256 covers only half the logical K slots in the
    // two-plane map (the new inverse caught this); use the already box-validated Q6 tactic instead.
    case 14: return prepare<4, 2, 128>(low_native, high_native, low_layout, high_layout, n, k);
    default: return 22;
  }
}

extern "C" int quactlize_ppu_recover_dense(uint8_t const* low_layout, uint8_t const* high_layout,
                                             uint8_t* low_native, uint8_t* high_native,
                                             int n, int k, int qtype) {
  if (!low_layout || !low_native || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  switch (qtype) {
    case 10: return recover<2, 0>(low_layout, high_layout, low_native, high_native, n, k);
    case 11: return recover<2, 1>(low_layout, high_layout, low_native, high_native, n, k);
    case 12: return recover<4, 0>(low_layout, high_layout, low_native, high_native, n, k);
    case 13: return recover<4, 1>(low_layout, high_layout, low_native, high_native, n, k);
    case 14: return recover<4, 2, 128>(low_layout, high_layout, low_native, high_native, n, k);
    default: return 22;
  }
}
