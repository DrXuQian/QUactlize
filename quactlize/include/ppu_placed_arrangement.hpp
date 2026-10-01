#pragma once

#include "ppu_format_config.hpp"
#include "kquant_kpack_offline.hpp"
#include "q4_kpack4_offline.hpp"
#include "quactlize_ppu_config.h"

namespace ppu_arrangements {

static_assert(q4_kpack4::kLayoutId == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 &&
              q4_kpack4::kMappingId == QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID);
static_assert(kquant_kpack::kLayoutId == QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1 &&
              kquant_kpack::kMappingId == QUACTLIZE_PPU_KQUANT_KPACK_MAPPING_ID);

constexpr quactlize_ppu_placed_arrangement_v2 q4_kpack4_transpose_v1() {
  auto const& format = ppu_formats::for_qtype(12);
  return {QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
          QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1,
          format.low_bits, format.high_bits, 0, q4_kpack4::kTransportK,
          format.group_size, 0, q4_kpack4::kMappingId};
}

constexpr quactlize_ppu_placed_arrangement_v2 kquant_kpack_transpose_v1(int qtype) {
  auto const& format = ppu_formats::for_qtype(qtype);
  return {QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V2,
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1,
          format.low_bits, format.high_bits, 0,
          kquant_kpack::transport_k(format.low_bits, format.high_bits),
          format.group_size, 0, kquant_kpack::kMappingId};
}

constexpr bool same_descriptor(quactlize_ppu_placed_arrangement_v2 const& lhs,
                               quactlize_ppu_placed_arrangement_v2 const& rhs) {
  return lhs.version == rhs.version && lhs.layout == rhs.layout &&
         lhs.bits == rhs.bits && lhs.high_bits == rhs.high_bits &&
         lhs.artifact_tile_k == rhs.artifact_tile_k &&
         lhs.transport_tile_k == rhs.transport_tile_k &&
         lhs.group_size == rhs.group_size && lhs.reserved == rhs.reserved &&
         lhs.mapping_id == rhs.mapping_id;
}

// Byte-map admission must not depend on a particular compute tile.
constexpr bool matches_canonical_kpack(quactlize_ppu_placed_arrangement_v2 const* a,
                                        int qtype) {
  if (!a || qtype < 10 || qtype > 14) return false;
  auto const canonical = qtype == 12 ? q4_kpack4_transpose_v1()
                                     : kquant_kpack_transpose_v1(qtype);
  return same_descriptor(*a, canonical);
}

constexpr bool matches_compiled_tactic(quactlize_ppu_placed_arrangement_v2 const* a,
                                       int qtype, int k, int tile_k) {
  return matches_canonical_kpack(a, qtype) &&
         k > 0 && k % a->group_size == 0 &&
         tile_k >= a->transport_tile_k && tile_k % a->transport_tile_k == 0;
}

constexpr bool packed_tensor_reader_supported(quactlize_ppu_placed_arrangement_v2 const* a,
                                               int qtype, int k, int tile_k) {
  return matches_compiled_tactic(a, qtype, k, tile_k);
}
} // namespace ppu_arrangements
