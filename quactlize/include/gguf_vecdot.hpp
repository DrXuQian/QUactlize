#pragma once
// PURE CUDA-CORE GGUF DECODE: one superblock's dot product, scales extracted straight into registers, no fp16 planes
// anywhere. This is the decode-side answer that the scale pre-pass cannot give.
//
// WHY BOTH EXIST AND NEITHER REPLACES THE OTHER. gguf_scale_prepass.hpp removes the STORAGE objection -- the fp16
// planes live in a workspace, so a k-quant checkpoint stops costing +9..52% stored bytes. It does nothing about
// RESIDENCY: at decode the planes would have to be rebuilt every token or kept resident, and keeping them is the
// forbidden storage again. schemes.py records exactly that for every k-quant GEMV cell -- "the GEMV kernels read
// fp16 scale planes; at decode those must be resident, and for a k-quant that is the stored-byte increase the
// constraint forbids". A native decode is the only thing that reaches that band.
//
// THE SHAPE IS llama.cpp's MMVQ, and the reason is arithmetic rather than taste. Writing the dot as
//
//     sum_i  W_i x_i  =  sum_g [ dl_g * (sum_{i in g} q_i x_i)  -  ml_g * (sum_{i in g} x_i) ]
//
// pulls both per-group scalars OUT of the inner loop: the inner loop touches only the integer code and the
// activation, and each group costs two multiplies at the end. The scale pair is therefore consumed where it is
// produced, in registers, which is the whole point -- a shared-memory publication would put back the channel the
// pre-pass exists to avoid, and at GEMV there is no M reuse to amortise it over.
//
// EVERY BIT LAYOUT BELOW IS TRANSCRIBED FROM gguf.quants.<T>.dequantize_blocks, not from memory and not from our own
// dump_real_weights.py -- that parser is where the tree's k-quant constants came from, so it cannot also be their
// witness. tests/test_gguf_golden.py checks this file against the official dequantiser element by element, which
// covers the ORDERING as well as the arithmetic; the pre-pass tests could not, because they only ever looked at
// per-group scalars.
// WHAT THIS IS NOT YET, AND IT MATTERS FOR BOTH CORRECTNESS AND PRECISION. These take the RAW GGUF block, in the
// checkpoint's own element order. The shipping GEMV consumes the OFFLINE-REORDERED weight -- quactlize's layout
// registry exists precisely because the kernel wants a different arrangement -- so two things follow that a
// raw-order test cannot see:
//
//   * a decoder validated only in raw order is validated on a path nobody runs. The reorder is a permutation, so it
//     preserves VALUES; what it changes is which element each lane owns, and an indexing mistake in the reordered
//     decode is invisible here.
//   * the accumulation ORDER changes with it. Each lane sums a different subset in a different sequence, so the fp32
//     result differs in the last bits from this one BY CONSTRUCTION. The tolerance for the reordered path has to be
//     a summation-order tolerance, not the ~1e-7 exactness the raw path reaches, and tightening it would produce a
//     test that fails for the one reason that is not a bug.
//
// So this file is the reference decode -- the thing the reordered kernel must agree with after permuting the
// reference the same way -- and not the kernel itself.
//
// THE SHIPPING GEMV'S SHAPE, decided and not yet built: express each format's REORDERED weight arrangement as a cute
// Layout and index the decode through it, one layout per format. That keeps the arrangement in the same place every
// other reorder in this tree lives -- the layout registry -- instead of as index arithmetic duplicated per kernel,
// and it is the same discipline that made the sub-byte B path work (pi = frag.layout()^-1, derived rather than
// written). The code extraction functions above stay as they are; only the element index they are handed changes.
#include <cstdint>
#include "cutlass/numeric_types.h"
#include "gguf_scale_layout.hpp"

#ifndef CUTLASS_HOST_DEVICE
#  define CUTLASS_HOST_DEVICE inline
#endif

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
  return (qs[(i / 64) * 32 + (i % 32)] >> (4 * ((i % 64) / 32))) & 0xF;
}
// Q5_K -- 176 B: d dmin scales[12] qh[32] qs[128]. Same low nibbles as Q4_K plus one high bit per element, and the
// high plane has a SINGLE 32-byte chunk indexed by bit position, not by chunk.
CUTLASS_HOST_DEVICE int q5k_code(uint8_t const* qs, uint8_t const* qh, int i) {
  return q4k_code(qs, i) | (((qh[i % 32] >> (i / 32)) & 1) << 4);
}
// Q2_K -- 84 B: scales[16] qs[64] d[2] dmin[2]. Sixteen groups of 16, two bits per element.
CUTLASS_HOST_DEVICE int q2k_code(uint8_t const* qs, int i) {
  return (qs[(i / 128) * 32 + (i % 32)] >> (2 * ((i % 128) / 32))) & 3;
}
// Q6_K -- 210 B: ql[128] qh[64] scales[16] d[2]. Four low bits plus two high, centred at -32, SIGNED scales.
CUTLASS_HOST_DEVICE int q6k_code(uint8_t const* ql, uint8_t const* qh, int i) {
  int const lo = (ql[(i / 128) * 64 + (i % 64)] >> (4 * ((i % 128) / 64))) & 0xF;
  int const hi = (qh[(i / 128) * 32 + (i % 32)] >> (2 * ((i % 128) / 32))) & 3;
  return (lo | (hi << 4)) - 32;
}
// Q3_K -- 110 B: hmask[32] qs[64] scales[12] d[2]. Two bits plus ONE INVERTED high bit: the official comment says
// "strangely, the offset is zero when the bitmask is 1", i.e. the mask being CLEAR subtracts 4. Getting that
// backwards shifts a third of the codes by 4 and still produces plausible weights.
CUTLASS_HOST_DEVICE int q3k_code(uint8_t const* qs, uint8_t const* hmask, int i) {
  int const lo = (qs[(i / 128) * 32 + (i % 32)] >> (2 * ((i % 128) / 32))) & 3;
  int const hbit = (hmask[i % 32] >> (i / 32)) & 1;
  return lo - ((hbit ^ 1) << 2);
}

// Q3_K's six-bit scales, SHIFT-MAJOR: lscales.reshape(1,8) >> [0,4] flattens to (2,8), hscales.reshape(1,4) >>
// [0,2,4,6] flattens to (4,4). So group g takes low nibble from byte g%8 at shift 4*(g/8) and high two bits from
// byte 8 + g%4 at shift 2*(g/4). Centre is -32.
CUTLASS_HOST_DEVICE int q3k_scale(uint8_t const* sc, int g) {
  int const lo = (sc[g % 8] >> (4 * (g / 8))) & 0xF;
  int const hi = (sc[8 + (g % 4)] >> (2 * (g / 4))) & 3;
  return (lo | (hi << 4)) - 32;
}

// ---------------------------------------------------------------------------------------------------------------
// ONE SUPERBLOCK, ONE DOT PRODUCT. `block` is the format's RAW GGUF block, exactly as it sits in the file: no
// repacking, which is what makes this usable at decode with the checkpoint's own bytes.
template <KType T>
CUTLASS_HOST_DEVICE float vecdot_block(uint8_t const* block, float const* x);

template <>
CUTLASS_HOST_DEVICE float vecdot_block<KType::Q4_K>(uint8_t const* b, float const* x) {
  float const d = float(half_t::bitcast(uint16_t(b[0] | (b[1] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[2] | (b[3] << 8))));
  uint8_t const* sc = b + 4;
  uint8_t const* qs = b + 16;
  float acc = 0.f;
  for (int g = 0; g < 8; ++g) {
    GroupAcc a{0.f, 0.f};
    for (int j = 0; j < 32; ++j) {
      int const i = g * 32 + j;
      a.sumqx += float(q4k_code(qs, i)) * x[i];
      a.sumx += x[i];
    }
    acc += apply_group(a, d * float(scale_of<KType::Q4_K>(sc, g)), dmin * float(min_of<KType::Q4_K>(sc, g)));
  }
  return acc;
}

template <>
CUTLASS_HOST_DEVICE float vecdot_block<KType::Q5_K>(uint8_t const* b, float const* x) {
  float const d = float(half_t::bitcast(uint16_t(b[0] | (b[1] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[2] | (b[3] << 8))));
  uint8_t const* sc = b + 4;
  uint8_t const* qh = b + 16;
  uint8_t const* qs = b + 48;
  float acc = 0.f;
  for (int g = 0; g < 8; ++g) {
    GroupAcc a{0.f, 0.f};
    for (int j = 0; j < 32; ++j) {
      int const i = g * 32 + j;
      a.sumqx += float(q5k_code(qs, qh, i)) * x[i];
      a.sumx += x[i];
    }
    acc += apply_group(a, d * float(scale_of<KType::Q5_K>(sc, g)), dmin * float(min_of<KType::Q5_K>(sc, g)));
  }
  return acc;
}

template <>
CUTLASS_HOST_DEVICE float vecdot_block<KType::Q2_K>(uint8_t const* b, float const* x) {
  uint8_t const* sc = b;
  uint8_t const* qs = b + 16;
  float const d = float(half_t::bitcast(uint16_t(b[80] | (b[81] << 8))));
  float const dmin = float(half_t::bitcast(uint16_t(b[82] | (b[83] << 8))));
  float acc = 0.f;
  for (int g = 0; g < 16; ++g) {
    GroupAcc a{0.f, 0.f};
    for (int j = 0; j < 16; ++j) {
      int const i = g * 16 + j;
      a.sumqx += float(q2k_code(qs, i)) * x[i];
      a.sumx += x[i];
    }
    acc += apply_group(a, d * float(sc[g] & 0xF), dmin * float(sc[g] >> 4));
  }
  return acc;
}

template <>
CUTLASS_HOST_DEVICE float vecdot_block<KType::Q6_K>(uint8_t const* b, float const* x) {
  uint8_t const* ql = b;
  uint8_t const* qh = b + 128;
  int8_t const* sc = reinterpret_cast<int8_t const*>(b + 192);
  float const d = float(half_t::bitcast(uint16_t(b[208] | (b[209] << 8))));
  float acc = 0.f;
  for (int g = 0; g < 16; ++g) {
    GroupAcc a{0.f, 0.f};
    for (int j = 0; j < 16; ++j) {
      int const i = g * 16 + j;
      a.sumqx += float(q6k_code(ql, qh, i)) * x[i];
      a.sumx += x[i];
    }
    acc += apply_group(a, d * float(sc[g]), 0.f);      // no min channel; the centre rides in the code
  }
  return acc;
}

template <>
CUTLASS_HOST_DEVICE float vecdot_block<KType::Q3_K>(uint8_t const* b, float const* x) {
  uint8_t const* hmask = b;
  uint8_t const* qs = b + 32;
  uint8_t const* sc = b + 96;
  float const d = float(half_t::bitcast(uint16_t(b[108] | (b[109] << 8))));
  float acc = 0.f;
  for (int g = 0; g < 16; ++g) {
    GroupAcc a{0.f, 0.f};
    for (int j = 0; j < 16; ++j) {
      int const i = g * 16 + j;
      a.sumqx += float(q3k_code(qs, hmask, i)) * x[i];
      a.sumx += x[i];
    }
    acc += apply_group(a, d * float(q3k_scale(sc, g)), 0.f);
  }
  return acc;
}

}  // namespace vecdot
}  // namespace gguf_scale
