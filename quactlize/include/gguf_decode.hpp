#pragma once
// Raw GGUF field decoding shared by GEMV and fully expanded GEMM.
// No device-kernel or PPU converter dependency. Arithmetic is unchanged.
#include <cstdint>
#include "cutlass/numeric_types.h"
#include "gguf_scale_layout.hpp"

namespace gguf_scale {
namespace vecdot {

using cutlass::half_t;

// The superblock is 256 elements in every k-quant. Group counts and sizes differ and come from Traits.
static constexpr int kQK = 256;

// One group's contribution, named once so the five formats cannot each invent their own rounding order.
struct GroupAcc {
  float sumqx;      // sum of code * activation
  float sumx;       // sum of activation, which the affine term multiplies
};

CUTLASS_HOST_DEVICE float apply_group(GroupAcc a, float dl, float ml) { return dl * a.sumqx - ml * a.sumx; }

// ---------------------------------------------------------------------------------------------------------------
// Q4_K -- 144 B: d[2] dmin[2] scales[12] qs[128]. Eight groups of 32.
// Official ordering: qs.reshape(-1,1,32) >> [0,4] & 0xF -> (4 chunks, 2 halves, 32 bytes), flattened (-1,32).
// So element i takes chunk i/64, nibble half (i%64)/32, byte i%32. Writing that out is the ONLY way the element
// ORDER gets checked; a decoder that had the right scales and the wrong order would pass every per-group test.
CUTLASS_HOST_DEVICE int q4k_code(uint8_t const* qs, int i) {
  using C = CodeTraits<KType::Q4_K>;
  int const g = i / 32, j = i % 32;
  return C::Lo::extract_byte(qs, C::word_coord(g, j / 4), j % 4);
}
// Q5_K -- 176 B: d dmin scales[12] qh[32] qs[128]. Same low nibbles as Q4_K plus one high bit per element, and the
// high plane has a SINGLE 32-byte chunk indexed by bit position, not by chunk.
CUTLASS_HOST_DEVICE int q5k_code(uint8_t const* qs, uint8_t const* qh, int i) {
  using C = CodeTraits<KType::Q5_K>;
  int const g = i / 32, j = i % 32, c = C::word_coord(g, j / 4);
  return C::Lo::extract_byte(qs, c, j % 4) | (C::Hi::extract_byte(qh, c, j % 4) << 4);
}
// Q2_K -- 84 B: scales[16] qs[64] d[2] dmin[2]. Sixteen groups of 16, two bits per element.
CUTLASS_HOST_DEVICE int q2k_code(uint8_t const* qs, int i) {
  using C = CodeTraits<KType::Q2_K>;
  int const g = i / 16, j = i % 16;
  return C::Lo::extract_byte(qs, C::word_coord(g, j / 4), j % 4);
}
// Q6_K -- 210 B: ql[128] qh[64] scales[16] d[2]. Four low bits plus two high, centred at -32, SIGNED scales.
CUTLASS_HOST_DEVICE int q6k_code(uint8_t const* ql, uint8_t const* qh, int i) {
  using C = CodeTraits<KType::Q6_K>;
  int const g = i / 16, j = i % 16, c = C::word_coord(g, j / 4);
  int const lo = C::Lo::extract_byte(ql, c, j % 4);
  int const hi = C::Hi::extract_byte(qh, c, j % 4);
  return (lo | (hi << 4)) - 32;
}
// Q3_K -- 110 B: hmask[32] qs[64] scales[12] d[2]. Two bits plus ONE INVERTED high bit: the official comment says
// "strangely, the offset is zero when the bitmask is 1", i.e. the mask being CLEAR subtracts 4. Getting that
// backwards shifts a third of the codes by 4 and still produces plausible weights.
CUTLASS_HOST_DEVICE int q3k_code(uint8_t const* qs, uint8_t const* hmask, int i) {
  using C = CodeTraits<KType::Q3_K>;
  int const g = i / 16, j = i % 16, c = C::word_coord(g, j / 4);
  int const lo = C::Lo::extract_byte(qs, c, j % 4);
  int const hbit = C::Hi::extract_byte(hmask, c, j % 4);
  return lo - ((hbit ^ 1) << 2);
}

// Q3_K's six-bit scales, SHIFT-MAJOR: lscales.reshape(1,8) >> [0,4] flattens to (2,8), hscales.reshape(1,4) >>
// [0,2,4,6] flattens to (4,4). So group g takes low nibble from byte g%8 at shift 4*(g/8) and high two bits from
// byte 8 + g%4 at shift 2*(g/4). Centre is -32.
CUTLASS_HOST_DEVICE int q3k_scale(uint8_t const* sc, int g) {
  return scale_of<KType::Q3_K>(sc, g) - Traits<KType::Q3_K>::kScaleBias;
}

// ---------------------------------------------------------------------------------------------------------------
// ONE TRAVERSAL, TWO CONSUMERS. visit<T> walks a superblock and hands each element its index, its GROUP index, its
// integer code, and that group's (dl, ml). vecdot_block and dequantize_block are both thin users of it.
//
// THE ALTERNATIVE IS TWO TRANSCRIPTIONS OF THE SAME BIT LAYOUT, and this file's whole claim is that the layout was
// transcribed once, from the official dequantiser, and checked. A second copy for the dequantise path would be a
// second thing to be wrong, and wrong in the way that is hardest to catch: each would be tested against the
// reference separately and could disagree with the other only on a format nobody exercised that week.
//
// The GROUP INDEX is passed rather than inferred. A first version detected group boundaries by watching (dl, ml)
// change, which is correct only by accident -- two adjacent groups with equal scale and min would not flush, and it
// happens to still give the right answer because the scalars are equal. Correct-by-accident in a traversal that two
// paths depend on is not worth the four bytes it saves.
template <class F> CUTLASS_HOST_DEVICE void visit_q4k(uint8_t const* b, F f) {
  float const d = float(half_t::bitcast(uint16_t(b[0] | (b[1] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[2] | (b[3] << 8))));
  uint8_t const* sc = b + 4; uint8_t const* qs = b + 16;
  for (int g = 0; g < 8; ++g) {
    float const dl = d * float(scale_of<KType::Q4_K>(sc, g));
    float const ml = dmin * float(min_of<KType::Q4_K>(sc, g));
    for (int j = 0; j < 32; ++j) { int const i = g * 32 + j; f(i, g, q4k_code(qs, i), dl, ml); }
  }
}
template <class F> CUTLASS_HOST_DEVICE void visit_q5k(uint8_t const* b, F f) {
  float const d = float(half_t::bitcast(uint16_t(b[0] | (b[1] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[2] | (b[3] << 8))));
  uint8_t const* sc = b + 4; uint8_t const* qh = b + 16; uint8_t const* qs = b + 48;
  for (int g = 0; g < 8; ++g) {
    float const dl = d * float(scale_of<KType::Q5_K>(sc, g));
    float const ml = dmin * float(min_of<KType::Q5_K>(sc, g));
    for (int j = 0; j < 32; ++j) { int const i = g * 32 + j; f(i, g, q5k_code(qs, qh, i), dl, ml); }
  }
}
template <class F> CUTLASS_HOST_DEVICE void visit_q2k(uint8_t const* b, F f) {
  uint8_t const* sc = b; uint8_t const* qs = b + 16;
  float const d = float(half_t::bitcast(uint16_t(b[80] | (b[81] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[82] | (b[83] << 8))));
  for (int g = 0; g < 16; ++g) {
    float const dl = d * float(scale_of<KType::Q2_K>(sc, g));
    float const ml = dmin * float(min_of<KType::Q2_K>(sc, g));
    for (int j = 0; j < 16; ++j) { int const i = g * 16 + j; f(i, g, q2k_code(qs, i), dl, ml); }
  }
}
template <class F> CUTLASS_HOST_DEVICE void visit_q6k(uint8_t const* b, F f) {
  uint8_t const* ql = b; uint8_t const* qh = b + 128;
  uint8_t const* sc = b + 192;
  float const d = float(half_t::bitcast(uint16_t(b[208] | (b[209] << 8))));
  for (int g = 0; g < 16; ++g) {
    float const dl = d * float(scale_of<KType::Q6_K>(sc, g)); // no min channel; the centre rides in the code
    for (int j = 0; j < 16; ++j) { int const i = g * 16 + j; f(i, g, q6k_code(ql, qh, i), dl, 0.f); }
  }
}
template <class F> CUTLASS_HOST_DEVICE void visit_q3k(uint8_t const* b, F f) {
  uint8_t const* hmask = b; uint8_t const* qs = b + 32; uint8_t const* sc = b + 96;
  float const d = float(half_t::bitcast(uint16_t(b[108] | (b[109] << 8))));
  for (int g = 0; g < 16; ++g) {
    float const dl = d * float(q3k_scale(sc, g));
    for (int j = 0; j < 16; ++j) { int const i = g * 16 + j; f(i, g, q3k_code(qs, hmask, i), dl, 0.f); }
  }
}

// RANDOM ACCESS, NOT A TRAVERSAL. visit<T> walks all 256 elements, which is right for a host loop and wrong for a
// warp: a lane that runs the whole traversal and keeps its 32nd share replicates the loads and the arithmetic 32
// times. Measured, that costs the entire win -- 775.9 us for one thread per block against 644.3 for a replicating
// warp, 1.2x, when the store pattern alone should be worth an order of magnitude.
//
// These give element i and group g directly, so a warp can PARTITION the block instead.
template <KType T>
CUTLASS_HOST_DEVICE int code_at(uint8_t const* b, int i) {
  if constexpr      (T == KType::Q2_K) return q2k_code(b + 16, i);
  else if constexpr (T == KType::Q3_K) return q3k_code(b + 32, b, i);
  else if constexpr (T == KType::Q4_K) return q4k_code(b + 16, i);
  else if constexpr (T == KType::Q5_K) return q5k_code(b + 48, b + 16, i);
  else                                 return q6k_code(b, b + 128, i);
}

// WHOLE-GROUP RANDOM ACCESS. The cooperative GEMV assigns group g to one lane, so spelling its (g,j) addressing
// directly both removes division from the inner loop and makes the new memory pattern reviewable. For every format
// j walks a contiguous run in each physical code plane. Adjacent groups either cover the next run or revisit the
// same packed bytes at another bit position:
//
//   Q2/Q3 low: 16-byte run; groups 0/1 cover 32 B, then groups 2/3 revisit it at the next two-bit shift.
//   Q4/Q5 low: 32-byte run; each odd group revisits the preceding even group's bytes at the high nibble.
//   Q3 hmask:  16-byte run at bit g/2.       Q5 qh: one 32-byte run at bit g.
//   Q6 ql/qh:  16-byte runs at respectively nibble (g/4)%2 and two-bit field (g/2)%4.
//
// Thus every plane stays inside the original compact superblock footprint; the separate high-bit planes are not
// inferred from low-plane indexing. CUDA correctness checks exercise all 256 (g,j) pairs against code_at/vecdot_block.
template <KType T>
CUTLASS_HOST_DEVICE int code_at_group(uint8_t const* b, int g, int j) {
  using C = CodeTraits<T>;
  int const c = C::word_coord(g, j / 4);
  int const lo = C::Lo::extract_byte(b + C::kLoOffset, c, j % 4);
  if constexpr (T == KType::Q2_K) {
    return lo;
  } else if constexpr (T == KType::Q3_K) {
    int const hbit = C::Hi::extract_byte(b + C::kHiOffset, c, j % 4);
    return lo - ((hbit ^ 1) << 2);
  } else if constexpr (T == KType::Q4_K) {
    return lo;
  } else if constexpr (T == KType::Q5_K) {
    return lo | (C::Hi::extract_byte(b + C::kHiOffset, c, j % 4) << 4);
  } else {
    int const hi = C::Hi::extract_byte(b + C::kHiOffset, c, j % 4);
    return (lo | (hi << 4)) - 32;
  }
}

// The block multipliers and group codes are split so a cooperative consumer can load d/dmin once per block and
// decode one scale pair per group. The serial traversal naturally hoists the header loads; repeatedly calling the
// former monolithic helper from every lane would have put that work back into the inner loop.
template <KType T>
CUTLASS_HOST_DEVICE void block_d_dmin(uint8_t const* b, float& d, float& dmin) {
  if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
    d = float(half_t::bitcast(uint16_t(b[0] | (b[1] << 8))));
    dmin = float(half_t::bitcast(uint16_t(b[2] | (b[3] << 8))));
  } else if constexpr (T == KType::Q2_K) {
    d = float(half_t::bitcast(uint16_t(b[80] | (b[81] << 8))));
    dmin = float(half_t::bitcast(uint16_t(b[82] | (b[83] << 8))));
  } else if constexpr (T == KType::Q3_K) {
    d = float(half_t::bitcast(uint16_t(b[108] | (b[109] << 8))));
    dmin = 0.f;
  } else {
    d = float(half_t::bitcast(uint16_t(b[208] | (b[209] << 8))));
    dmin = 0.f;
  }
}

template <KType T>
CUTLASS_HOST_DEVICE void group_dl_ml_from_base(uint8_t const* b, int g, float d, float dmin,
                                                float& dl, float& ml) {
  if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
    dl = d * float(scale_of<T>(b + 4, g));
    ml = dmin * float(min_of<T>(b + 4, g));
  } else if constexpr (T == KType::Q2_K) {
    dl = d * float(scale_of<T>(b, g));
    ml = dmin * float(min_of<T>(b, g));
  } else if constexpr (T == KType::Q3_K) {
    dl = d * float(q3k_scale(b + 96, g));
    ml = 0.f;
  } else {
    dl = d * float(scale_of<T>(b + 192, g));
    ml = 0.f;
  }
}

// The group's (dl, ml), read straight from the block's own header and scale field.
template <KType T>
CUTLASS_HOST_DEVICE void group_dl_ml(uint8_t const* b, int g, float& dl, float& ml) {
  float d, dmin;
  block_d_dmin<T>(b, d, dmin);
  group_dl_ml_from_base<T>(b, g, d, dmin, dl, ml);
}

template <KType T>
CUTLASS_HOST_DEVICE constexpr int group_size() {
  return (T == KType::Q4_K || T == KType::Q5_K) ? 32 : 16;
}

template <KType T, class F>
CUTLASS_HOST_DEVICE void visit(uint8_t const* b, F f) {
  if constexpr      (T == KType::Q2_K) visit_q2k(b, f);
  else if constexpr (T == KType::Q3_K) visit_q3k(b, f);
  else if constexpr (T == KType::Q4_K) visit_q4k(b, f);
  else if constexpr (T == KType::Q5_K) visit_q5k(b, f);
  else                                 visit_q6k(b, f);
}

} // namespace vecdot
} // namespace gguf_scale
