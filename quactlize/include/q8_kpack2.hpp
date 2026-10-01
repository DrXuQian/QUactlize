#pragma once

#include "kquant_kpack_offline.hpp"
#include "quactlize_ppu_config.h"

// Q8_0 is not a K-quant packed-unit format. Each raw block contains one FP16
// d and 32 signed bytes. The resident planes are byte-neutral, per expert:
//   low:   b16 [K/2,N], each word gathers {q[k]+128,q[k+8]+128}
//   scale: FP16 [K/32,N], preserving d's original bits (including signed zero).
// No high or zero plane. Consumers keep A FP16 and convert B in registers.
namespace q8_kpack2 {
using Map = kquant_kpack::PlaneMap<8,32>;
inline constexpr int kQtype = 8;
inline constexpr uint64_t kMappingId = QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID;
struct Traits { int low_bits = 8, high_bits = 0, group_size = 32; };

constexpr quactlize_ppu_placed_arrangement_v2 arrangement() {
  return {2, QUACTLIZE_PPU_LAYOUT_Q8_KPACK2_TRANSPOSE_V1, 8, 0, 0, 32, 32, 0, kMappingId};
}
constexpr bool matches(quactlize_ppu_placed_arrangement_v2 const* a) {
  auto b = arrangement();
  return a && a->version == b.version && a->layout == b.layout &&
      a->bits == b.bits && a->high_bits == 0 && a->artifact_tile_k == 0 &&
      a->transport_tile_k == 32 && a->group_size == 32 && a->reserved == 0 &&
      a->mapping_id == b.mapping_id;
}
constexpr bool shape_supported(int n, int k, int experts) {
  return n > 0 && k > 0 && experts > 0 && n % 256 == 0 && k % 256 == 0;
}
} // namespace q8_kpack2
