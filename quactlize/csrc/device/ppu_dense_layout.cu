// Offline layout half of SCALE_FIRST x DENSE. This is deliberately separate from ppu_dense_backend.cu: the reorder
// is host code and can be built by nvcc for its local seam test, while the fpA tensor-core launcher is hgcc-only.
// Both are linked into libquactlize_ppu.so on the PPU build.
#include <cstdint>
#include <algorithm>
#include <vector>

#include "fold_traits.hpp"
#include "ppu_format_config.hpp"
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

// The producer is asked for ArtifactTileK, never for a fold. Both folds come from the same helper the consumer uses,
// and the canonical WON=2 geometry is copied from the measured grouped fixtures. Narrow planes raise WN (and TN with
// it) until the delivery is legal: this is the difference between the complete int1 TK64 map at WN64 and the all-zero
// or half-covered maps that a nominally well-formed WN32 instantiation can produce.
template <int LowBits, int HighBits, int ArtifactTileK>
struct ProducerConfig {
  static constexpr int LowFold = fold::delivery_fold_v<LowBits, ArtifactTileK>;
  static constexpr int HighFold = [] {
    if constexpr (HighBits == 0) return 1;
    else return fold::delivery_fold_v<HighBits, ArtifactTileK>;
  }();
  static constexpr int MaxFold = LowFold > HighFold ? LowFold : HighFold;
  static constexpr int WarpN = MaxFold > 2 ? 16 * MaxFold : 32;
  static constexpr int TileN = 2 * WarpN;
  static_assert(WarpN <= 64, "this producer exposes only consumer-validated warp widths");
};

template <int LowBits, int HighBits, int ArtifactTileK = 256>
int prepare(uint8_t const* low_native, uint8_t const* high_native,
            uint8_t* low_layout, uint8_t* high_layout, int n, int k) {
  using C = ProducerConfig<LowBits, HighBits, ArtifactTileK>;
  constexpr int LowFold = C::LowFold, HighFold = C::HighFold;
  static_assert(LowFold * ArtifactTileK * LowBits % 256 == 0,
                "low xplane row must contain a whole number of 32-byte deliveries");
  static_assert(HighBits == 0 || HighFold * ArtifactTileK * HighBits % 256 == 0,
                "high xplane row must contain a whole number of 32-byte deliveries");
  std::vector<uint8_t> low;
  unpack_native<LowBits>(low_native, low, n, k);
  xplane::place_derived<LowBits, 64, C::TileN, ArtifactTileK, 32, C::WarpN, LowFold>(
      reinterpret_cast<int8_t*>(low_layout), low, n, k);
  if constexpr (HighBits != 0) {
    if (!high_native || !high_layout) return 21;
    std::vector<uint8_t> high;
    unpack_native<HighBits>(high_native, high, n, k);
    xplane::place_hi<LowBits, HighBits, 64, C::TileN, ArtifactTileK, 32, C::WarpN, HighFold, LowFold>(
        reinterpret_cast<int8_t*>(high_layout), high, n, k);
  }
  return 0;
}

template <int LowBits, int HighBits, int ArtifactTileK = 256>
int recover(uint8_t const* low_layout, uint8_t const* high_layout,
            uint8_t* low_native, uint8_t* high_native, int n, int k) {
  using C = ProducerConfig<LowBits, HighBits, ArtifactTileK>;
  constexpr int LowFold = C::LowFold, HighFold = C::HighFold;
  static_assert(LowFold * ArtifactTileK * LowBits % 256 == 0,
                "low xplane row must contain a whole number of 32-byte deliveries");
  static_assert(HighBits == 0 || HighFold * ArtifactTileK * HighBits % 256 == 0,
                "high xplane row must contain a whole number of 32-byte deliveries");
  std::vector<uint8_t> low;
  xplane::recover_derived<LowBits, 64, C::TileN, ArtifactTileK, 32, C::WarpN, LowFold>(
      reinterpret_cast<int8_t const*>(low_layout), low, n, k);
  pack_native<LowBits>(low, low_native, n, k);
  if constexpr (HighBits != 0) {
    if (!high_layout || !high_native) return 21;
    std::vector<uint8_t> high;
    xplane::recover_hi<LowBits, HighBits, 64, C::TileN, ArtifactTileK, 32, C::WarpN, HighFold, LowFold>(
        reinterpret_cast<int8_t const*>(high_layout), high, n, k);
    pack_native<HighBits>(high, high_native, n, k);
  }
  return 0;
}

// The runtime surface is deliberately FINITE. Each case is a real template instantiation; arbitrary integers would
// turn "the template parses" into an artifact capability. Q3/Q5 TK32 have no viable int1 consumer, and Q6 TK256's
// cross-plane map is known incomplete, so neither row exists here.
#define QUACTLIZE_TILE_CASES(CALL, LB, HB)                                   \
  case 32:  return CALL<LB, HB, 32>(low_in, high_in, low_out, high_out, n, k); \
  case 64:  return CALL<LB, HB, 64>(low_in, high_in, low_out, high_out, n, k); \
  case 128: return CALL<LB, HB, 128>(low_in, high_in, low_out, high_out, n, k); \
  case 256: return CALL<LB, HB, 256>(low_in, high_in, low_out, high_out, n, k)
#define QUACTLIZE_TILE_CASES_NO32(CALL, LB, HB)                               \
  case 64:  return CALL<LB, HB, 64>(low_in, high_in, low_out, high_out, n, k); \
  case 128: return CALL<LB, HB, 128>(low_in, high_in, low_out, high_out, n, k); \
  case 256: return CALL<LB, HB, 256>(low_in, high_in, low_out, high_out, n, k)
#define QUACTLIZE_TILE_CASES_NO256(CALL, LB, HB)                              \
  case 32:  return CALL<LB, HB, 32>(low_in, high_in, low_out, high_out, n, k); \
  case 64:  return CALL<LB, HB, 64>(low_in, high_in, low_out, high_out, n, k); \
  case 128: return CALL<LB, HB, 128>(low_in, high_in, low_out, high_out, n, k)

template <bool Recover>
int tile_dispatch(uint8_t const* low_in, uint8_t const* high_in,
                  uint8_t* low_out, uint8_t* high_out,
                  int n, int k, int qtype, int artifact_tile_k) {
  if (!low_in || !low_out || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  if (artifact_tile_k <= 0 || k % artifact_tile_k) return 23;
  if ((qtype == 11 || qtype == 13) && artifact_tile_k == 32) return 23;
  if (qtype == 14 && artifact_tile_k == 256) return 23;
  // A conditional expression cannot name function templates, so select the operation once around the format/tile
  // ladder while keeping each legal row an explicit instantiation.
  if constexpr (Recover) {
    switch (qtype) {
      case 10: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES(recover, 2, 0); default: return 23; }
      case 11: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO32(recover, 2, 1); default: return 23; }
      case 12: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES(recover, 4, 0); default: return 23; }
      case 13: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO32(recover, 4, 1); default: return 23; }
      case 14: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO256(recover, 4, 2); default: return 23; }
      default: return 22;
    }
  } else {
    switch (qtype) {
      case 10: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES(prepare, 2, 0); default: return 23; }
      case 11: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO32(prepare, 2, 1); default: return 23; }
      case 12: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES(prepare, 4, 0); default: return 23; }
      case 13: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO32(prepare, 4, 1); default: return 23; }
      case 14: switch (artifact_tile_k) { QUACTLIZE_TILE_CASES_NO256(prepare, 4, 2); default: return 23; }
      default: return 22;
    }
  }
}

#undef QUACTLIZE_TILE_CASES
#undef QUACTLIZE_TILE_CASES_NO32
#undef QUACTLIZE_TILE_CASES_NO256

}  // namespace

extern "C" int quactlize_ppu_prepare_dense(uint8_t const* low_native, uint8_t const* high_native,
                                             uint8_t* low_layout, uint8_t* high_layout,
                                             int n, int k, int qtype) {
  if (!low_native || !low_layout || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  switch (qtype) {
    case 10: return prepare<2, 0, ppu_formats::for_qtype(10).scale_first_tile_k>(
        low_native, high_native, low_layout, high_layout, n, k);
    case 11: return prepare<2, 1, ppu_formats::for_qtype(11).scale_first_tile_k>(
        low_native, high_native, low_layout, high_layout, n, k);
    case 12: return prepare<4, 0, ppu_formats::for_qtype(12).scale_first_tile_k>(
        low_native, high_native, low_layout, high_layout, n, k);
    case 13: return prepare<4, 1, ppu_formats::for_qtype(13).scale_first_tile_k>(
        low_native, high_native, low_layout, high_layout, n, k);
    case 14: return prepare<4, 2, ppu_formats::for_qtype(14).scale_first_tile_k>(
        low_native, high_native, low_layout, high_layout, n, k);
    default: return 22;
  }
}

extern "C" int quactlize_ppu_recover_dense(uint8_t const* low_layout, uint8_t const* high_layout,
                                             uint8_t* low_native, uint8_t* high_native,
                                             int n, int k, int qtype) {
  if (!low_layout || !low_native || n <= 0 || k <= 0 || n % 256 || k % 256) return 20;
  switch (qtype) {
    case 10: return recover<2, 0, ppu_formats::for_qtype(10).scale_first_tile_k>(
        low_layout, high_layout, low_native, high_native, n, k);
    case 11: return recover<2, 1, ppu_formats::for_qtype(11).scale_first_tile_k>(
        low_layout, high_layout, low_native, high_native, n, k);
    case 12: return recover<4, 0, ppu_formats::for_qtype(12).scale_first_tile_k>(
        low_layout, high_layout, low_native, high_native, n, k);
    case 13: return recover<4, 1, ppu_formats::for_qtype(13).scale_first_tile_k>(
        low_layout, high_layout, low_native, high_native, n, k);
    case 14: return recover<4, 2, ppu_formats::for_qtype(14).scale_first_tile_k>(
        low_layout, high_layout, low_native, high_native, n, k);
    default: return 22;
  }
}

// Artifact-aware ABI. Kept as distinct symbols so an extension cannot pass the new integer to an older fixed-layout
// library. The integer is ArtifactTileK: it fixes the bytes and derives both ArtifactLowFold and ArtifactHighFold.
// A consumer's independent TacticTileK never crosses this producer boundary.
extern "C" int quactlize_ppu_prepare_dense_for_tile(
    uint8_t const* low_native, uint8_t const* high_native,
    uint8_t* low_layout, uint8_t* high_layout,
    int n, int k, int qtype, int artifact_tile_k) {
  return tile_dispatch<false>(low_native, high_native, low_layout, high_layout,
                              n, k, qtype, artifact_tile_k);
}

extern "C" int quactlize_ppu_recover_dense_for_tile(
    uint8_t const* low_layout, uint8_t const* high_layout,
    uint8_t* low_native, uint8_t* high_native,
    int n, int k, int qtype, int artifact_tile_k) {
  return tile_dispatch<true>(low_layout, high_layout, low_native, high_native,
                             n, k, qtype, artifact_tile_k);
}
