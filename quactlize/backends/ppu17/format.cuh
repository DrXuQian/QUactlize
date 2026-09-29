#pragma once

#include <type_traits>
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "gguf_scale_layout.hpp"
#include "kquant_kpack_offline.hpp"
#include "ppu_format_config.hpp"
#include "api.h"

namespace quactlize::ppu17 {

template <class T> struct Affine { T scale, zero; };

// Byte-only reader for the existing packed-unit ABI. Keep the bit layout
// checked against gguf_packed_unit.hpp in host tests; that header's optional
// AIU re-exports must not enter a PPU1.7 translation unit.
template <int Q> struct Format {
  static constexpr auto config = ppu_formats::for_qtype(Q);
  static_assert(config.qtype == Q);
  static constexpr auto type = gguf_scale::KType(Q - 10);
  using Traits = gguf_scale::Traits<type>;
  static constexpr int low_bits = config.low_bits, high_bits = config.high_bits;
  static constexpr int group_size = config.group_size;
  static constexpr int header_bytes = Traits::kHasMin ? 4 : 2;
  static constexpr int sb_bytes = header_bytes + Traits::kBlockBytes;
  static constexpr int sb_per_unit = sb_bytes % 4 ? 2 : 1;
  static constexpr int run_groups = Traits::kHasMin ? Traits::kGroups / 2 : Traits::kGroups;
  static constexpr int run_bits = run_groups * (Traits::kScaleBits + Traits::kMinBits);
  static constexpr int converter_bias = low_bits == 4 ? 8 : 0;
  static constexpr int quant_bias = Q == 11 ? 4 : Q == 14 ? 32 : 0;
  static constexpr int correction = converter_bias - quant_bias;
  using Low = kquant_kpack::PlaneMap<low_bits, group_size>;
  using High = kquant_kpack::HighPlaneMap<low_bits, high_bits, group_size>;

  CUTLASS_HOST_DEVICE static constexpr size_t unit_bytes(int n, int k) {
    return size_t(n) * (k / 256) * sb_bytes;
  }
  CUTLASS_HOST_DEVICE static constexpr size_t unit_offset(int n, int col, int group) {
    int sb = group / Traits::kGroups;
    return (size_t(sb / sb_per_unit) * n + col) * (sb_per_unit * sb_bytes)
        + (sb % sb_per_unit) * sb_bytes;
  }
  CUTLASS_HOST_DEVICE static constexpr int field_bit(int group, int which) {
    return header_bytes * 8 + (group / run_groups) * run_bits
        + (which ? run_groups * Traits::kScaleBits : 0)
        + (group % run_groups) * (which ? Traits::kMinBits : Traits::kScaleBits);
  }
  CUTLASS_HOST_DEVICE static int field(uint8_t const *p, int group, int which) {
    int bits = which ? Traits::kMinBits : Traits::kScaleBits;
    int bit = field_bit(group, which), shift = bit & 7;
    unsigned word = p[bit / 8];
    if (shift + bits > 8) word |= unsigned(p[bit / 8 + 1]) << 8;
    return (word >> shift) & ((1u << bits) - 1);
  }
  CUTLASS_HOST_DEVICE static int decode_words(uint16_t low, uint16_t high, int col, int k) {
    int value = (low >> (low_bits * Low::word_slot(k)))
        & ((1 << low_bits) - 1);
    if constexpr (high_bits != 0) {
      int slot;
      if constexpr (Q == 13) slot = High::word_slot(col, k);
      else slot = High::word_slot(k);
      value |= ((high >> (high_bits * slot))
          & ((1 << high_bits) - 1)) << low_bits;
    }
    return value - converter_bias;
  }
  CUTLASS_HOST_DEVICE static int code(uint16_t const *low, uint16_t const *high,
                                    int n, int col, int k) {
    uint16_t h = 0;
    if constexpr (high_bits) h = high[High::word_index(col, k, n)];
    return decode_words(low[Low::word_index(col, k, n)], h, col, k);
  }
  template <class T>
  CUTLASS_HOST_DEVICE static Affine<T> affine(uint8_t const *units, int n, int col, int group) {
    auto p = units + unit_offset(n, col, group);
    auto half_value = [](uint8_t const *v) {
      return float(cutlass::half_t::bitcast(uint16_t(v[0]) | uint16_t(v[1]) << 8));
    };
    int g = group % Traits::kGroups;
    int sc = field(p, g, 0);
    if constexpr (Traits::kSigned) sc = (sc ^ 128) - 128;
    sc -= Traits::kScaleBias;
    Affine<T> result{T(half_value(p) * float(sc)), T(0.f)};
    if constexpr (Traits::kHasMin)
      result.zero = T(-half_value(p + 2) * float(field(p, g, 1)));
    if constexpr (correction != 0) {
      if constexpr (std::is_same_v<T, cutlass::bfloat16_t>) {
        // Direct BF16 multiply then add, matching typed FQ/SF metadata.
        T adjusted(float(correction) * float(result.scale));
        result.zero = T(float(result.zero) + float(adjusted));
      } else {
        result.zero = T(float(result.zero) + float(correction) * float(result.scale));
      }
    }
    return result;
  }
};

template <> struct Format<8> {
  static constexpr int low_bits = 8, high_bits = 0, group_size = 32;
  using Low = kquant_kpack::PlaneMap<8, 32>;
  using High = kquant_kpack::PlaneMap<1, 32>; // unused descriptor type, never loaded
  CUTLASS_HOST_DEVICE static constexpr size_t unit_bytes(int n, int k) {
    return size_t(n) * (k / 32) * 2;
  }
  CUTLASS_HOST_DEVICE static int code(uint16_t const *low, uint16_t const *,
                                    int n, int col, int k) {
    return decode_words(low[Low::word_index(col, k, n)], 0, col, k);
  }
  CUTLASS_HOST_DEVICE static int decode_words(uint16_t low, uint16_t, int, int k) {
    return int((low >> (8 * Low::word_slot(k))) & 255) - 128;
  }
  template <class T>
  CUTLASS_HOST_DEVICE static Affine<T> affine(uint8_t const *units, int n, int col, int group) {
    auto p = units + 2 * (size_t(group) * n + col);
    return {T(float(cutlass::half_t::bitcast(uint16_t(p[0]) | uint16_t(p[1]) << 8))), T(0.f)};
  }
};

template <int Q, class T, bool ScaleFirst> struct WeightReader {
  using F = Format<Q>;
  uint16_t const *low, *high;
  uint8_t const *units;
  T const *scale, *zero;
  int n, k;

  CUTLASS_HOST_DEVICE explicit WeightReader(q17_call_v1 const &c, int expert)
      : low(static_cast<uint16_t const *>(c.low) + size_t(expert) * c.problem.n * c.problem.k * F::low_bits / 16),
        high(nullptr), units(nullptr), scale(nullptr), zero(nullptr), n(c.problem.n), k(c.problem.k) {
    if constexpr (F::high_bits)
      high = static_cast<uint16_t const *>(c.high) + size_t(expert) * n * k * F::high_bits / 16;
    if constexpr (ScaleFirst) {
      size_t offset = size_t(expert) * n * (k / F::group_size);
      scale = static_cast<T const *>(c.scale) + offset;
      zero = static_cast<T const *>(c.zero) + offset;
    } else {
      units = static_cast<uint8_t const *>(c.units) + size_t(expert) * F::unit_bytes(n, k);
    }
  }
  CUTLASS_HOST_DEVICE T dequantize(int code, int col, int kk) const {
    int group = kk / F::group_size;
    Affine<T> a;
    if constexpr (ScaleFirst) a = {scale[size_t(group) * n + col], zero[size_t(group) * n + col]};
    else a = F::template affine<T>(units, n, col, group);
    T product(float(code) * float(a.scale));
    return T(float(product) + float(a.zero));
  }
  CUTLASS_HOST_DEVICE T operator()(int col, int kk) const {
    return dequantize(F::code(low, high, n, col, kk), col, kk);
  }
};

} // namespace quactlize::ppu17
