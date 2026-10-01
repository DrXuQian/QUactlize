#pragma once

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "gguf_scale_layout.hpp"
// Packed metadata helpers require the SDK half types. The byte/scale decode
// remains available when only the host-side CUTLASS types are present.
#if defined(__has_include)
#  if __has_include(<hggc_fp16.h>)
#    define GGUF_SCALE_HAVE_PACKED 1
#    include "actlize_extensions/cutlass/gguf_packed_scale.h"
#  endif
#endif

namespace gguf_scale {

using cutlass::half_t;

CUTLASS_HOST_DEVICE half_t int_to_half_small(int v) {
  uint16_t const bits = uint16_t(0x6400u | uint32_t((v + 128) & 0x3FF));
  return cutlass::half_t::bitcast(bits) - half_t(1152.f);
}

// The code's own centre is a per-format constant (Q3_K's -32; zero for the others) and lives in Traits::kScaleBias,
// read off dump_real_weights.py. Subtracting it BEFORE the conversion keeps one magic constant instead of two.
template <KType T>
CUTLASS_HOST_DEVICE half_t scale_code_to_half(int sc) {
  return int_to_half_small(sc - Traits<T>::kScaleBias);
}

// The two products, named once so a caller cannot form them a second way. `zero` carries the SIGN the mainloop's
// `plus{}` pass expects -- it adds `zero`, and a k-quant's affine term is -dmin*mn.
struct GroupScale {
  half_t scale;
  half_t zero;
};

template <KType T>
CUTLASS_HOST_DEVICE GroupScale make_group_scale(int sc, int mn, half_t d, half_t dmin) {
  GroupScale g;
  g.scale = d * scale_code_to_half<T>(sc);
  g.zero  = -(dmin * int_to_half_small(mn));      // the min code has no centre in any k-quant that has a min
  return g;
}

// Scale only (Q3_K/Q6_K after phase 1, and any symmetric format): no min channel at all, and the code's own centre is
// the converter's Bias, so nothing is left for a zero to carry.
template <KType T>
CUTLASS_HOST_DEVICE half_t make_group_scale_only(int sc, half_t d) {
  return d * scale_code_to_half<T>(sc);
}

// ---------------------------------------------------------------------------------------------------------------
// ONE SUPERBLOCK, ONE COLUMN: the object that replaces eight strided fp16 reads with one contiguous read plus decode.
// `bytes` is the format's scale block (12 B for Q4_K/Q3_K, 16 B for Q2_K/Q6_K) as it sits in memory -- no repacking,
// which is the point of phase 2: smem holds the native bytes.
template <KType T>
struct Superblock {
  static constexpr int kGroups     = Traits<T>::kGroups;
  static constexpr int kBlockBytes = Traits<T>::kBlockBytes;
  static constexpr bool kHasMin    = Traits<T>::kHasMin;

  GroupScale g[kGroups];

  CUTLASS_HOST_DEVICE void decode(uint8_t const* bytes, half_t d, half_t dmin) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kGroups; ++i) {
      int const sc = scale_of<T>(bytes, i);
      if constexpr (kHasMin) {
        g[i] = make_group_scale<T>(sc, min_of<T>(bytes, i), d, dmin);
      } else {
        g[i].scale = make_group_scale_only<T>(sc, d);
        g[i].zero  = half_t(0.f);
      }
    }
  }
};

// The traffic this replaces, as an expression rather than a comment, so a shape change cannot silently invalidate the
// claim: fp16 scale (+ zero) per group per column against the native bytes per superblock per column.
template <KType T>
CUTLASS_HOST_DEVICE constexpr float bytes_per_group_per_col() {
  return float(Traits<T>::kBlockBytes) / float(Traits<T>::kGroups);
}
template <KType T>
CUTLASS_HOST_DEVICE constexpr float bytes_per_group_per_col_fp16() {
  return Traits<T>::kHasMin ? 4.f : 2.f;   // scale + zero, or scale alone
}


#if defined(GGUF_SCALE_HAVE_PACKED)
using cutlass::gguf_packed::PackBits;
using cutlass::gguf_packed::kUnitBytes;
static constexpr int kPackBitBase = cutlass::gguf_packed::kBitBase;
CUTLASS_HOST_DEVICE constexpr int packed_bit_of(int g, int which) { return cutlass::gguf_packed::bit_of(g, which); }
CUTLASS_HOST_DEVICE int  packed_code(uint8_t const* u, int g, int which) { return cutlass::gguf_packed::code_of(u, g, which); }
CUTLASS_HOST_DEVICE void packed_put (uint8_t* u, int g, int which, int v) { cutlass::gguf_packed::put_code(u, g, which, v); }
template <KType T>
CUTLASS_HOST_DEVICE GroupScale packed_group(uint8_t const* unit, int g) {
  auto const g2 = cutlass::gguf_packed::group_of<Traits<T>::kScaleBias, Traits<T>::kHasMin>(unit, g);
  return GroupScale{g2.scale, g2.zero};
}

#endif  // GGUF_SCALE_HAVE_PACKED

}  // namespace gguf_scale
