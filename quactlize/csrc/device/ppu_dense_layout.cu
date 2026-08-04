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

template <int LowBits, int HighBits, int TileK = 256, int LowFold = 1, int HighFold = 1>
int prepare(uint8_t const* low_native, uint8_t const* high_native,
            uint8_t* low_layout, uint8_t* high_layout, int n, int k) {
  static_assert(LowFold * TileK * LowBits % 256 == 0,
                "low xplane row must contain a whole number of 32-byte deliveries");
  static_assert(HighBits == 0 || HighFold * TileK * HighBits % 256 == 0,
                "high xplane row must contain a whole number of 32-byte deliveries");
  std::vector<uint8_t> low;
  unpack_native<LowBits>(low_native, low, n, k);
  xplane::place_derived<LowBits, 64, 64, TileK, 32, 32, LowFold>(
      reinterpret_cast<int8_t*>(low_layout), low, n, k);
  if constexpr (HighBits != 0) {
    if (!high_native || !high_layout) return 21;
    std::vector<uint8_t> high;
    unpack_native<HighBits>(high_native, high, n, k);
    xplane::place_hi<LowBits, HighBits, 64, 64, TileK, 32, 32, HighFold, LowFold>(
        reinterpret_cast<int8_t*>(high_layout), high, n, k);
  }
  return 0;
}

template <int LowBits, int HighBits, int TileK = 256, int LowFold = 1, int HighFold = 1>
int recover(uint8_t const* low_layout, uint8_t const* high_layout,
            uint8_t* low_native, uint8_t* high_native, int n, int k) {
  static_assert(LowFold * TileK * LowBits % 256 == 0,
                "low xplane row must contain a whole number of 32-byte deliveries");
  static_assert(HighBits == 0 || HighFold * TileK * HighBits % 256 == 0,
                "high xplane row must contain a whole number of 32-byte deliveries");
  std::vector<uint8_t> low;
  xplane::recover_derived<LowBits, 64, 64, TileK, 32, 32, LowFold>(
      reinterpret_cast<int8_t const*>(low_layout), low, n, k);
  pack_native<LowBits>(low, low_native, n, k);
  if constexpr (HighBits != 0) {
    if (!high_layout || !high_native) return 21;
    std::vector<uint8_t> high;
    xplane::recover_hi<LowBits, HighBits, 64, 64, TileK, 32, 32, HighFold, LowFold>(
        reinterpret_cast<int8_t const*>(high_layout), high, n, k);
    pack_native<HighBits>(high, high_native, n, k);
  }
  return 0;
}

// The arrangement surface is deliberately FINITE. Every row below is either a shipped layout or a tactic already
// built by the fold sweeps; accepting an arbitrary integer tuple would turn a template instantiation into a runtime
// promise. Fold factors are the minimum that makes each plane's contiguous run a whole 32-byte AIU delivery. F>Fmin
// remains an unmeasured tuning axis and is refused until a consumer has actually established it.
template <int LowBits, int HighBits, int TileK, int LowFold, int HighFold>
int prepare_arranged(uint8_t const* low_native, uint8_t const* high_native,
                     uint8_t* low_layout, uint8_t* high_layout, int n, int k,
                     int low_fold, int high_fold) {
  if (low_fold != LowFold || high_fold != HighFold) return 24;
  return prepare<LowBits, HighBits, TileK, LowFold, HighFold>(
      low_native, high_native, low_layout, high_layout, n, k);
}

template <int LowBits, int HighBits, int TileK, int LowFold, int HighFold>
int recover_arranged(uint8_t const* low_layout, uint8_t const* high_layout,
                     uint8_t* low_native, uint8_t* high_native, int n, int k,
                     int low_fold, int high_fold) {
  if (low_fold != LowFold || high_fold != HighFold) return 24;
  return recover<LowBits, HighBits, TileK, LowFold, HighFold>(
      low_layout, high_layout, low_native, high_native, n, k);
}

#define QUACTLIZE_ARRANGED_TILES(CALL, LB, HB)                                                    \
  case 32:  return CALL<LB, HB, 32,  256 / (32  * LB),                                           \
                       (HB == 0 ? 1 : 256 / (32  * HB))>(                                        \
      low_in, high_in, low_out, high_out, n, k, low_fold, high_fold);                            \
  case 64:  return CALL<LB, HB, 64,  (64  * LB >= 256 ? 1 : 256 / (64  * LB)),                   \
                       (HB == 0 || 64  * HB >= 256 ? 1 : 256 / (64  * HB))>(                     \
      low_in, high_in, low_out, high_out, n, k, low_fold, high_fold);                            \
  case 128: return CALL<LB, HB, 128, 1, (HB == 0 || 128 * HB >= 256 ? 1 : 256 / (128 * HB))>(    \
      low_in, high_in, low_out, high_out, n, k, low_fold, high_fold);                            \
  case 256: return CALL<LB, HB, 256, 1, 1>(                                                      \
      low_in, high_in, low_out, high_out, n, k, low_fold, high_fold)

template <bool Recover>
int arranged_dispatch(uint8_t const* low_in, uint8_t const* high_in,
                      uint8_t* low_out, uint8_t* high_out,
                      int n, int k, int qtype, int low_fold, int tile_k, int high_fold) {
  if (!low_in || !low_out || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  if (tile_k <= 0 || k % tile_k) return 23;
#define CALL (Recover ? recover_arranged : prepare_arranged)
  // A conditional expression cannot name function templates, so select the operation once around the format/tile
  // ladder while keeping each legal row an explicit instantiation.
  if constexpr (Recover) {
    switch (qtype) {
      case 10: switch (tile_k) { QUACTLIZE_ARRANGED_TILES(recover_arranged, 2, 0); default: return 23; }
      case 11: switch (tile_k) { case 64: case 128: case 256: QUACTLIZE_ARRANGED_TILES(recover_arranged, 2, 1); default: return 23; }
      case 12: switch (tile_k) { QUACTLIZE_ARRANGED_TILES(recover_arranged, 4, 0); default: return 23; }
      case 13: switch (tile_k) { case 64: case 128: case 256: QUACTLIZE_ARRANGED_TILES(recover_arranged, 4, 1); default: return 23; }
      case 14: switch (tile_k) { case 32: case 64: case 128: QUACTLIZE_ARRANGED_TILES(recover_arranged, 4, 2); default: return 23; }
      default: return 22;
    }
  } else {
    switch (qtype) {
      case 10: switch (tile_k) { QUACTLIZE_ARRANGED_TILES(prepare_arranged, 2, 0); default: return 23; }
      case 11: switch (tile_k) { case 64: case 128: case 256: QUACTLIZE_ARRANGED_TILES(prepare_arranged, 2, 1); default: return 23; }
      case 12: switch (tile_k) { QUACTLIZE_ARRANGED_TILES(prepare_arranged, 4, 0); default: return 23; }
      case 13: switch (tile_k) { case 64: case 128: case 256: QUACTLIZE_ARRANGED_TILES(prepare_arranged, 4, 1); default: return 23; }
      case 14: switch (tile_k) { case 32: case 64: case 128: QUACTLIZE_ARRANGED_TILES(prepare_arranged, 4, 2); default: return 23; }
      default: return 22;
    }
  }
#undef CALL
}

#undef QUACTLIZE_ARRANGED_TILES

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
