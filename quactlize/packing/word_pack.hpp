#pragma once

#include "gguf_unit_pack.hpp"
#include "kquant_kpack_offline.hpp"
#include "q8_kpack2.hpp"
#include <type_traits>

namespace quactlize_pack {

using gguf_scale::KType;

// Virtual raw [E,2N,K], gate rows before up rows within each expert.
// Both sources retain their original GGUF block bytes; no intermediate copy.
struct PairedRows {
  uint8_t const* gate;
  uint8_t const* up;
  uint64_t expert_bytes;
  CUTLASS_HOST_DEVICE uint8_t const* operator+(uint64_t offset) const {
    uint64_t const expert = offset / (2 * expert_bytes);
    uint64_t const local = offset % (2 * expert_bytes);
    return (local < expert_bytes ? gate : up) + expert * expert_bytes + local % expert_bytes;
  }
  CUTLASS_HOST_DEVICE uint8_t operator[](uint64_t offset) const { return *(operator+(offset)); }
};

struct Q8 {
  template<class Source>
  CUTLASS_HOST_DEVICE static uint16_t word(Source raw, int n, int k, uint64_t index) {
    uint64_t const words = uint64_t(n) * (k / 2);
    uint64_t const expert = index / words, local = index % words;
    int const col = int(local % n), kg = int(local / n);
    uint16_t result = 0;
    for (int slot = 0; slot < 2; ++slot) {
      int const kk = q8_kpack2::Map::logical_k(kg, slot);
      uint64_t const block = (expert * n + col) * (k / 32) + kk / 32;
      result |= uint16_t(raw[block * 34 + 2 + kk % 32] ^ 0x80) << (8 * slot);
    }
    return result;
  }
  template<class Source>
  CUTLASS_HOST_DEVICE static void metadata(Source raw, uint8_t* scale,
                                           int n, int k, uint64_t index) {
    uint64_t const groups = k / 32, outer = index / n;
    uint64_t const block = ((outer / groups) * n + index % n) * groups + outer % groups;
    scale[2*index] = raw[34*block];
    scale[2*index+1] = raw[34*block+1];
  }
};

// Ownership is one entire output word, not an atomic nibble scatter.
template <KType T, bool High>
struct Plane {
  using C = gguf_scale::CodeTraits<T>;
  using Field = std::conditional_t<High, typename C::Hi, typename C::Lo>;
  static constexpr int low_bits = C::Lo::kWidth;
  static constexpr int bits = Field::kWidth;
  static constexpr int group = gguf_scale::Traits<T>::kGroupSize;
  static constexpr int pack = 16 / bits;
  using Map = std::conditional_t<High,
      kquant_kpack::HighPlaneMap<low_bits, bits, group>,
      kquant_kpack::PlaneMap<bits, group>>;

  template<class Source>
  CUTLASS_HOST_DEVICE static uint16_t word(
      Source raw, int n, int k, uint64_t index) {
    using R = gguf_scale::unit_pack::Raw<T>;
    uint64_t const words = uint64_t(n) * (k / pack);
    uint64_t const expert = index / words;
    uint64_t const local = index % words;
    int const pn = int(local % n), kg = int(local / n);
    uint16_t result = 0;
    for (int slot = 0; slot < pack; ++slot) {
      int const col = Map::logical_n(pn, kg, slot);
      int kk;
      if constexpr (High && T == KType::Q5_K)
        kk = Map::logical_k(pn, kg, slot);
      else
        kk = Map::logical_k(kg, slot);
      uint8_t const* block = raw +
          ((expert * n + col) * (k / 256) + kk / 256) * R::kBytes;
      int const j = kk % 256;
      int const coord = C::word_coord(j / group, (j % group) / 4);
      // Q3/Q6 canonical planes carry offset-binary codes. Raw low/high
      // fields already are those bits; no signed dequant/requant step.
      int const code = Field::extract_byte(
          block + (High ? C::kHiOffset : C::kLoOffset), coord, j % 4);
      result |= uint16_t(code << (slot * bits));
    }
    return result;
  }
};

template <KType T, class Source>
CUTLASS_HOST_DEVICE void metadata(
    Source raw, uint8_t* units, int n, int k, uint64_t index) {
  using U = gguf_scale::packed_unit::Unit<T>;
  using R = gguf_scale::unit_pack::Raw<T>;
  int const ku = k / (256 * U::kSbPerUnit);
  int const col = int(index % n);
  uint64_t const outer = index / n;
  uint64_t const expert = outer / ku;
  int const unit = int(outer % ku);
  for (int sb = 0; sb < U::kSbPerUnit; ++sb) {
    uint8_t const* block = raw +
        ((expert * n + col) * (k / 256) + unit * U::kSbPerUnit + sb) * R::kBytes;
    uint16_t const d = uint16_t(block[R::kDOffset]) |
        (uint16_t(block[R::kDOffset + 1]) << 8);
    uint16_t dm = 0;
    if constexpr (U::kHasMin)
      dm = uint16_t(block[R::kDminOffset]) |
          (uint16_t(block[R::kDminOffset + 1]) << 8);
    gguf_scale::packed_unit::pack_unit_sb<T>(
        block + R::kScaleOffset, cutlass::half_t::bitcast(d),
        cutlass::half_t::bitcast(dm), sb, units + index * U::kUnitTotal);
  }
}

} // namespace quactlize_pack
