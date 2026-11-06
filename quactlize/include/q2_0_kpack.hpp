#pragma once
#include "kquant_kpack_offline.hpp"
#include "quactlize_ppu_config.h"

// GGML Q2_0 (ID42), NOT Q2_K: 64 values = FP16 d + 16 code bytes.
// A code is unsigned 0..3; its value is (code - 1) * d. No rebiasing,
// requantization or scale arithmetic occurs during packing.
// low: b16 [E,K/8,N], eight K8-spaced codes/word; units: FP16 bits [E,K/64,N].
namespace q2_0_kpack {
using Low = kquant_kpack::PlaneMap<2, 64>;
inline bool shape_supported(int n, int k, int experts) {
    return experts > 0 && Low::shape_supported(n, k);
}
inline quactlize_ppu_placed_arrangement_v2 arrangement() {
    return {2, QUACTLIZE_PPU_LAYOUT_Q2_0_KPACK_TRANSPOSE_V1, 2, 0, 0,
            Low::kTransportK, 64, 0, QUACTLIZE_PPU_Q2_0_KPACK_MAPPING_ID};
}
inline bool matches(quactlize_ppu_placed_arrangement_v2 const *a) {
    auto b = arrangement();
    return a && a->version == b.version && a->layout == b.layout && a->bits == b.bits &&
           a->high_bits == b.high_bits && a->artifact_tile_k == b.artifact_tile_k &&
           a->transport_tile_k == b.transport_tile_k && a->group_size == b.group_size &&
           a->reserved == b.reserved && a->mapping_id == b.mapping_id;
}
} // namespace q2_0_kpack
