#pragma once

#include "kquant_kpack_offline.hpp"
#include "quactlize_ppu_config.h"

// Q5_1 is NOT Q5_K. Keep its unsigned 5-bit codes and original FP16 d/min
// bits, without arithmetic, rebiasing, padding or requantization. Per expert:
// low: b16 [K/4,N], four K8-spaced low nibbles per word;
// high: b16 [K/16,N], sixteen K8-spaced high bits per word (NOT Q5_K's
// converter-specific N/K high-plane transpose);
// units: original {d,min} bytes in [K/32,N,4]. Total bytes equal raw GGUF.
// A complete high-word delivery covers K128, so K640 needs no padding.
namespace q5_1_kpack {
using Low = kquant_kpack::PlaneMap<4, 32>;
using High = kquant_kpack::PlaneMap<1, 32>;
constexpr quactlize_ppu_placed_arrangement_v2 arrangement() {
    return {2, QUACTLIZE_PPU_LAYOUT_Q5_1_KPACK_TRANSPOSE_V1, 4, 1, 0, 256, 32,
            0, QUACTLIZE_PPU_Q5_1_KPACK_MAPPING_ID};
}
constexpr bool matches(quactlize_ppu_placed_arrangement_v2 const *a) {
    auto b = arrangement();
    return a && a->version == b.version && a->layout == b.layout && a->bits == b.bits &&
           a->artifact_tile_k == b.artifact_tile_k && a->high_bits == b.high_bits &&
           a->transport_tile_k == b.transport_tile_k && a->group_size == b.group_size && a->reserved == 0 &&
           a->mapping_id == b.mapping_id;
}
constexpr bool shape_supported(int n, int k, int experts) {
    return n > 0 && n % 16 == 0 && k > 0 && k % High::kLogicalKPerDelivery == 0 && experts > 0;
}
} // namespace q5_1_kpack
