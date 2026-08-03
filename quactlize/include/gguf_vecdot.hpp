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
// THE SOURCE CONTRACT IS RAW GGUF. This route intentionally reads the checkpoint's native k-quant blocks in their
// own element order and materialises neither weights nor fp16 scale/zero planes. The SCALE_FIRST route is different:
// its packed code tensor can be reordered for gemv_lowbit/, but that representation is not an input to this kernel.
// The cooperative reduction below changes only fp32 summation order: every format combines its group/block owners
// in one final row butterfly. Its device golden therefore uses an explicit summation-order tolerance against this
// scalar traversal.
#include <cstdint>
#include "cutlass/numeric_types.h"
#include "gguf_scale_layout.hpp"   // brings cute/tensor.hpp, so cute::Layout is available as a destination
#include "cute/atom/copy_atom.hpp"

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

// THE GEMV CONSUMER. Grouping the dot as dl*sum(q x) - ml*sum(x) keeps both scalars out of the inner loop, which is
// the whole reason this shape is worth having at decode.
template <KType T>
CUTLASS_HOST_DEVICE float vecdot_block(uint8_t const* b, float const* x) {
  float acc = 0.f, sumqx = 0.f, sumx = 0.f, dl = 0.f, ml = 0.f;
  int cur = -1;
  visit<T>(b, [&](int i, int g, int q, float gdl, float gml) {
    if (g != cur) { if (cur >= 0) acc += dl * sumqx - ml * sumx; cur = g; dl = gdl; ml = gml; sumqx = 0.f; sumx = 0.f; }
    sumqx += float(q) * x[i];
    sumx += x[i];
  });
  return acc + dl * sumqx - ml * sumx;
}

// THE FALLBACK CONSUMER: full fp16 weights, which is what a cuBLAS or DeepGemm path multiplies. NO ZMul here -- that
// is the mixed-input CONVERTER's centre correction, and these are the actual weight values. Adding it would shift
// every weight by 8*scale and still look entirely plausible, which is the failure mode that parameter exists for.
//
// THE DESTINATION IS A cute TENSOR, NOT A CALLABLE THAT HAPPENS TO LOOK LIKE A LAYOUT. The distinction is operational:
// a copy partition can be applied to a tensor and to its coordinate tensor together, which lets the device kernel
// assign consecutive PHYSICAL output addresses to consecutive lanes. Calling `layout(logical_i)` after lanes have
// already striped logical indices loses that property on every nontrivial layout and measured 196.6 us versus 65.5.
//
// The block-level host primitive takes a tensor for the same reason the kernel does: shape, layout and storage remain
// one object, and a caller cannot pass address arithmetic whose claimed layout is invisible to cute.
template <KType T, class Engine, class Layout>
CUTLASS_HOST_DEVICE void dequantize_block_to(uint8_t const* b, cute::Tensor<Engine, Layout> dst) {
  visit<T>(b, [&](int i, int, int q, float dl, float ml) { dst(i) = half_t(dl * float(q) - ml); });
}
template <KType T>
CUTLASS_HOST_DEVICE void dequantize_block(uint8_t const* b, half_t* out) {
  auto dst = cute::make_tensor(out, cute::make_layout(cute::make_shape(cute::Int<kQK>{})));
  dequantize_block_to<T>(b, dst);
}

// THE SPLIT CONSUMER: codes, scale and zero as SEPARATE arrays, which is what every offline packer in this tree
// takes. It is the piece that lets a GGUF checkpoint reach the existing kernels at all -- they consume a packed
// low-bit weight plus fp16 planes, and until now nothing turned a k-quant block into that triple.
//
// The split is exact by construction: W = code * scale + zero with zero = -dmin*mn, so the reconstruction is a test
// that can actually fail, and does if the traversal hands out the wrong element for a code.
template <KType T>
CUTLASS_HOST_DEVICE void unpack_block(uint8_t const* b, int8_t* codes, half_t* scale, half_t* zero) {
  visit<T>(b, [&](int i, int g, int q, float dl, float ml) {
    codes[i] = int8_t(q);
    scale[g] = half_t(dl);
    zero[g] = half_t(-ml);
  });
}

// ---------------------------------------------------------------------------------------------------------------
// DECIDED: NO dp4a. llama.cpp's MMVQ gets its speed by quantising the ACTIVATION to int8 per 32-block and
// accumulating with __dp4a, which is a CUDA-core instruction doing four int8 products into an int32 in one issue.
// That branch is NOT taken here, for two reasons that both stand on their own.
//
// The numerical one: it quantises the activation, and this path exists to serve decode, where there is no second
// chance to recover the error. TRT-LLM's weight-only GEMV takes the other branch -- fp16 activations, weights
// dequantised into registers, fma -- and introduces no activation error at all. vecdot_block above is that shape.
//
// The hardware one, which should be checked before anyone revisits this: PPU is confirmed to have an int8 TENSOR
// core (ppu.mma.m16n16k32.s32.s8.s8.s32) and is NOT confirmed to have a cuda-core four-way int8 dot at all. If it
// does not, the MMVQ shape has no advantage to trade the error against.
//
// dp4a would also force four codes into one 32-bit register lined up with four quantised activation bytes. Keeping
// fp32 activation loads and register-dequantised weights avoids both that packing constraint and its numerical error.

#if defined(__CUDACC__) || defined(__HGGCCC__)
// ---------------------------------------------------------------------------------------------------------------
// THIS IS THE NATIVE-GGUF GEMV, distinct from gemv_lowbit/. That validated launcher consumes pre-materialised fp16
// scale/zero planes (SCALE_FIRST); this path keeps raw k-quant blocks resident and decodes scale codes in registers
// (FULLY_QUANTIZED). Both routes are useful and neither substitutes for the other at decode.
//
// The implementation stays in the CUDA/hgcc common subset so the same ownership and arithmetic can run on PPU and
// in the local CUDA probe. The probe is host-CUDA-only; the kernel itself is not.
//
// THE DEVICE ENTRY POINTS. Everything above is CUTLASS_HOST_DEVICE, which in a host-only build degrades to `inline`
// -- so until these existed there was no kernel at all, only arithmetic that COULD be compiled as device code. That
// distinction is worth stating because the torch ops that validate all of this are CPU loops: they establish that
// the arithmetic matches llama.cpp and nothing whatever about a device path.
//
// The old GEMV is retained as a named timing baseline. One thread owns an entire output row, so a warp's raw-weight
// loads are blocks_per_row*block_bytes apart (1152 B for Q4_K at eight superblocks). That is the source-side analogue
// of dequantize_kernel_warp's measured store pathology below.
template <KType T>
__global__ void vecdot_rows_kernel_serial(uint8_t const* blocks, int64_t block_bytes, float const* x,
                                          float* out, int rows, int blocks_per_row) {
  int const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float acc = 0.f;
  for (int b = 0; b < blocks_per_row; ++b) {
    acc += vecdot_block<T>(blocks + (int64_t(r) * blocks_per_row + b) * block_bytes, x + int64_t(b) * kQK);
  }
  out[r] = acc;
}

// SUBGROUP-COOPERATIVE GEMV. A warp owns RowsPerWarp rows and each active lane consumes contiguous words from the raw
// code planes. Q2/Q3 use eight lanes per row and lane L owns groups L and L+8. Q6 uses sixteen lanes and one group per
// lane. Q4/Q5 use four lanes; lane L owns the adjacent pair (2L,2L+1), hence both nibbles of qs[32L..32L+31].
//
// Every code word supplies four adjacent elements. The separate high plane is merged while the four codes are still
// byte-packed; conversion and multiplication remain scalar CUDA-core fp32 operations. One butterfly at row end
// combines the owners. This changes fp32 summation order but introduces no activation/weight quantisation; dp4a and
// tensor-core MMA remain forbidden.
template <int Width>
__device__ __forceinline__ float vecdot_subgroup_sum(float v) {
  static_assert(Width >= 1 && Width <= 32 && (Width & (Width - 1)) == 0, "subgroup width must be a warp divisor");
  CUTLASS_PRAGMA_UNROLL
  for (int off = Width / 2; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}

// Measurement-only NOPs used by the rows=131072 cost probes. A constant nonzero code removes packed-code loads and
// extraction while every lane still consumes x, scale and the final reduction. Unit scale/min codes remove the
// packed per-group field loads and extraction while preserving header loads, both accumulators and every live lane.
template <KType T>
__device__ __forceinline__ int vecdot_kernel_code_at(uint8_t const* b, int g, int j) {
#if defined(GGUF_VECDOT_CODE_NOP)
  (void)b; (void)g; (void)j;
  return 1;
#else
  return code_at_group<T>(b, g, j);
#endif
}

// One aligned 32-bit load supplies four adjacent packed bytes. Raw k-quant blocks are compact: Q2/Q4/Q5 strides
// (84/144/176 B) preserve four-byte alignment, but Q3/Q6 strides (110/210 B) make every odd block two-byte aligned.
// Test the actual address rather than assuming cudaMalloc's base alignment propagates through the block stride.
// Two aligned 16-bit loads handle that real case; byte assembly remains the fully general fallback. The group loop
// separately retains a scalar-code tail so a future non-multiple-of-four group cannot read past its plane.
__device__ __forceinline__ uint32_t vecdot_load_u32_le(uint8_t const* p) {
  if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0)
    return *reinterpret_cast<uint32_t const*>(p);
  if ((reinterpret_cast<uintptr_t>(p) & 1u) == 0) {
    uint16_t const lo = *reinterpret_cast<uint16_t const*>(p);
    uint16_t const hi = *reinterpret_cast<uint16_t const*>(p + 2);
    return uint32_t(lo) | (uint32_t(hi) << 16);
  }
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

struct VecdotCode4 {
  float q0, q1, q2, q3;
};

struct VecdotCodePair4 {
  VecdotCode4 lo, hi;
};

template <KType T>
__device__ __forceinline__ VecdotCode4 vecdot_code4_from_bytes(uint32_t bytes) {
  constexpr float kOffset = T == KType::Q3_K ? -4.f : T == KType::Q6_K ? -32.f : 0.f;
  // A guarded CUDA __byte_perm / PPU ppu.prmt.b32 A/B produced the same Q2/Q3/Q5 time and cost Q4 one event tick.
  // Keep ordinary exact integer-to-float conversion and no private copy of the selector/magic constants. Actlize's
  // MixGemmNumericArrayConverter<half_t,int8_t,4> is not a fit here: it emits fp16 operand fragments with PPU-only
  // asm, whereas this CUDA/hgcc-common GEMV immediately needs four fp32 values for fp32 activation products.
  return {float( bytes        & 0xFFu) + kOffset,
          float((bytes >>  8) & 0xFFu) + kOffset,
          float((bytes >> 16) & 0xFFu) + kOffset,
          float((bytes >> 24) & 0xFFu) + kOffset};
}

// Four-code word decode for one whole-group owner. Each physical plane is loaded once; two-plane formats merge the
// high bits while they are still byte-packed. Q3 returns q+4 and Q6 q+32 here so every byte stays unsigned; the
// format offset is applied by vecdot_code4_from_bytes after extraction.
template <KType T>
__device__ __forceinline__ VecdotCode4 vecdot_kernel_code4(uint8_t const* b, int g, int j) {
#if defined(GGUF_VECDOT_CODE_NOP)
  (void)b; (void)g; (void)j;
  return {1.f, 1.f, 1.f, 1.f};
#else
  using C = CodeTraits<T>;
  int const c = C::word_coord(g, j / 4);
  uint8_t const* lo_ptr = b + C::kLoOffset + C::Lo::byte_of(c);
  uint32_t const lo = C::Lo::extract_word(vecdot_load_u32_le(lo_ptr), c);
  uint32_t codes;
  if constexpr (T == KType::Q2_K || T == KType::Q4_K) {
    codes = lo;
  } else {
    uint8_t const* hi_ptr = b + C::kHiOffset + C::Hi::byte_of(c);
    uint32_t const high = C::Hi::extract_word(vecdot_load_u32_le(hi_ptr), c);
    if constexpr (T == KType::Q3_K) codes = lo | (high << 2);  // q+4
    else                            codes = lo | (high << 4);  // Q5 raw, Q6 q+32
  }
  return vecdot_code4_from_bytes<T>(codes);
#endif
}

template <KType T>
__device__ __forceinline__ VecdotCodePair4 vecdot_kernel_q45_pair4(uint8_t const* b, int pair, int j) {
  static_assert(T == KType::Q4_K || T == KType::Q5_K, "paired word decode is only Q4/Q5");
#if defined(GGUF_VECDOT_CODE_NOP)
  (void)b; (void)pair; (void)j;
  VecdotCode4 const one{1.f, 1.f, 1.f, 1.f};
  return {one, one};
#else
  using C = CodeTraits<T>;
  int const g0 = 2 * pair;
  int const c0 = C::word_coord(g0, j / 4);
  int const c1 = C::word_coord(g0 + 1, j / 4);
  uint8_t const* lo_ptr = b + C::kLoOffset + C::Lo::byte_of(c0);
  uint32_t const packed = vecdot_load_u32_le(lo_ptr);
  uint32_t lo = C::Lo::extract_word(packed, c0);
  uint32_t hi = C::Lo::extract_word(packed, c1);
  if constexpr (T == KType::Q5_K) {
    uint8_t const* high_ptr = b + C::kHiOffset + C::Hi::byte_of(c0);
    uint32_t const high = vecdot_load_u32_le(high_ptr);
    lo |= C::Hi::extract_word(high, c0) << 4;
    hi |= C::Hi::extract_word(high, c1) << 4;
  }
  return {vecdot_code4_from_bytes<T>(lo), vecdot_code4_from_bytes<T>(hi)};
#endif
}

template <KType T>
__device__ __forceinline__ void vecdot_kernel_group_dl_ml(uint8_t const* b, int g, float d, float dmin,
                                                          float& dl, float& ml) {
#if defined(GGUF_VECDOT_SCALE_NOP)
  (void)b; (void)g;
  dl = d;
  if constexpr (T == KType::Q2_K || T == KType::Q4_K || T == KType::Q5_K) ml = dmin;
  else ml = 0.f;
#else
  group_dl_ml_from_base<T>(b, g, d, dmin, dl, ml);
#endif
}

// Retuned at rows=131072 after packed-word extraction. Q2/Q3 select four rows per warp, Q4/Q5 eight, and Q6 two.
template <KType T>
CUTLASS_HOST_DEVICE constexpr int vecdot_preferred_rows_per_warp() {
  if constexpr (T == KType::Q4_K || T == KType::Q5_K) return 8;
  else if constexpr (T == KType::Q6_K) return 2;
  else return 4;
}

template <KType T, int RowsPerWarp = vecdot_preferred_rows_per_warp<T>()>
CUTLASS_HOST_DEVICE constexpr int vecdot_grid_size(int rows, int threads) {
  int const warps = (rows + RowsPerWarp - 1) / RowsPerWarp;
  int const warps_per_cta = threads / 32;
  return (warps + warps_per_cta - 1) / warps_per_cta;
}

template <KType T, int RowsPerWarp = vecdot_preferred_rows_per_warp<T>()>
__global__ void vecdot_rows_kernel(uint8_t const* blocks, int64_t block_bytes, float const* x,
                                   float* out, int rows, int blocks_per_row) {
  static_assert(RowsPerWarp >= 1 && RowsPerWarp <= 32 && (RowsPerWarp & (RowsPerWarp - 1)) == 0,
                "rows per warp must be a power of two");
  constexpr int kLanesPerRow = 32 / RowsPerWarp;
  constexpr int kGroups = (T == KType::Q4_K || T == KType::Q5_K) ? 8 : 16;
  constexpr int kGroupSize = group_size<T>();
  constexpr bool kHasMin = T == KType::Q2_K || T == KType::Q4_K || T == KType::Q5_K;
  static_assert(kGroups % kLanesPerRow == 0 || kLanesPerRow % kGroups == 0,
                "lane subgroup and group count must divide one another");

  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  int const row_in_warp = lane / kLanesPerRow;
  int const row_lane = lane & (kLanesPerRow - 1);
  int const r = warp * RowsPerWarp + row_in_warp;
  bool const active = r < rows;

  // Q4/Q5 pair adjacent groups so one lane owns both nibbles of each byte. At the intended RowsPerWarp=8 point,
  // four lanes own one row and lane L consumes groups (2L,2L+1): its low-plane run is qs[32L..32L+31], loaded
  // exactly once and decoded into both groups. Q5's qh[j] is likewise loaded once for the pair, then the two group
  // bit positions are selected from that byte. Other sweep points preserve the same pair-strided ownership.
  if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
    float lane_acc0 = 0.f, lane_acc1 = 0.f;
    for (int block = 0; block < blocks_per_row; ++block) {
      uint8_t const* blk = active
          ? blocks + (int64_t(r) * blocks_per_row + block) * block_bytes
          : blocks;
      float d = 0.f, dmin = 0.f;
      bool const owns_pair = row_lane < kGroups / 2;
      if (active && owns_pair) block_d_dmin<T>(blk, d, dmin);

      CUTLASS_PRAGMA_UNROLL
      for (int pair = row_lane; pair < kGroups / 2; pair += kLanesPerRow) {
        int const g0 = 2 * pair;
        int const g1 = g0 + 1;
        float dl0, ml0, dl1, ml1;
        vecdot_kernel_group_dl_ml<T>(blk, g0, d, dmin, dl0, ml0);
        vecdot_kernel_group_dl_ml<T>(blk, g1, d, dmin, dl1, ml1);
        float qx00 = 0.f, qx01 = 0.f, qx10 = 0.f, qx11 = 0.f;
        float sx00 = 0.f, sx01 = 0.f, sx10 = 0.f, sx11 = 0.f;
        if (active) {
          int j = 0;
          CUTLASS_PRAGMA_UNROLL
          for (; j + 3 < kGroupSize; j += 4) {
            VecdotCodePair4 const q = vecdot_kernel_q45_pair4<T>(blk, pair, j);
            float const x00 = x[int64_t(block) * kQK + g0 * kGroupSize + j];
            float const x01 = x[int64_t(block) * kQK + g0 * kGroupSize + j + 1];
            float const x02 = x[int64_t(block) * kQK + g0 * kGroupSize + j + 2];
            float const x03 = x[int64_t(block) * kQK + g0 * kGroupSize + j + 3];
            float const x10 = x[int64_t(block) * kQK + g1 * kGroupSize + j];
            float const x11 = x[int64_t(block) * kQK + g1 * kGroupSize + j + 1];
            float const x12 = x[int64_t(block) * kQK + g1 * kGroupSize + j + 2];
            float const x13 = x[int64_t(block) * kQK + g1 * kGroupSize + j + 3];
            qx00 += q.lo.q0*x00; qx01 += q.lo.q1*x01; qx00 += q.lo.q2*x02; qx01 += q.lo.q3*x03;
            qx10 += q.hi.q0*x10; qx11 += q.hi.q1*x11; qx10 += q.hi.q2*x12; qx11 += q.hi.q3*x13;
            sx00 += x00; sx01 += x01; sx00 += x02; sx01 += x03;
            sx10 += x10; sx11 += x11; sx10 += x12; sx11 += x13;
          }
          CUTLASS_PRAGMA_UNROLL
          for (; j < kGroupSize; ++j) {
            float const x0 = x[int64_t(block) * kQK + g0 * kGroupSize + j];
            float const x1 = x[int64_t(block) * kQK + g1 * kGroupSize + j];
            float const q0 = float(vecdot_kernel_code_at<T>(blk, g0, j));
            float const q1 = float(vecdot_kernel_code_at<T>(blk, g1, j));
            if (j & 1) { qx01 += q0*x0; qx11 += q1*x1; sx01 += x0; sx11 += x1; }
            else       { qx00 += q0*x0; qx10 += q1*x1; sx00 += x0; sx10 += x1; }
          }
        }
        lane_acc0 += apply_group({qx00 + qx01, sx00 + sx01}, dl0, ml0);
        lane_acc1 += apply_group({qx10 + qx11, sx10 + sx11}, dl1, ml1);
      }
    }
    float lane_acc = vecdot_subgroup_sum<kLanesPerRow>(lane_acc0 + lane_acc1);
    if (active && row_lane == 0) out[r] = lane_acc;
    return;
  }

  float lane_acc = 0.f;

  for (int block = 0; block < blocks_per_row; ++block) {
    uint8_t const* blk = active
        ? blocks + (int64_t(r) * blocks_per_row + block) * block_bytes
        : blocks;
    float d = 0.f, dmin = 0.f;
    bool const owns_group = row_lane < kGroups;
    if (active && owns_group) block_d_dmin<T>(blk, d, dmin);

    CUTLASS_PRAGMA_UNROLL
    for (int g = row_lane; g < kGroups; g += kLanesPerRow) {
      float dl, ml;
      vecdot_kernel_group_dl_ml<T>(blk, g, d, dmin, dl, ml);
      float sum0 = 0.f, sum1 = 0.f;
#if !defined(GGUF_VECDOT_PER_ELEMENT_AFFINE)
      float sumx0 = 0.f, sumx1 = 0.f;
#endif
      if (active) {
#if !defined(GGUF_VECDOT_PER_ELEMENT_AFFINE)
        int j = 0;
        CUTLASS_PRAGMA_UNROLL
        for (; j + 3 < kGroupSize; j += 4) {
          VecdotCode4 const q = vecdot_kernel_code4<T>(blk, g, j);
          float const x0 = x[int64_t(block) * kQK + g * kGroupSize + j];
          float const x1 = x[int64_t(block) * kQK + g * kGroupSize + j + 1];
          float const x2 = x[int64_t(block) * kQK + g * kGroupSize + j + 2];
          float const x3 = x[int64_t(block) * kQK + g * kGroupSize + j + 3];
          if constexpr (T == KType::Q2_K) {
            // Preserve Q2's measured single dependency chain; two accumulators lost one event tick at both tested K.
            sum0 += q.q0 * x0; sum0 += q.q1 * x1; sum0 += q.q2 * x2; sum0 += q.q3 * x3;
            if constexpr (kHasMin) { sumx0 += x0; sumx0 += x1; sumx0 += x2; sumx0 += x3; }
          } else {
            sum0 += q.q0 * x0; sum1 += q.q1 * x1; sum0 += q.q2 * x2; sum1 += q.q3 * x3;
            if constexpr (kHasMin) { sumx0 += x0; sumx1 += x1; sumx0 += x2; sumx1 += x3; }
          }
        }
        // Current k-quant groups are multiples of four. Keep the scalar tail explicit so widening the helper does not
        // silently make a future odd group read past its plane.
        CUTLASS_PRAGMA_UNROLL
        for (; j < kGroupSize; ++j) {
          float const xv = x[int64_t(block) * kQK + g * kGroupSize + j];
          float const qv = float(vecdot_kernel_code_at<T>(blk, g, j));
          if constexpr (T == KType::Q2_K) {
            sum0 += qv * xv;
            if constexpr (kHasMin) sumx0 += xv;
          } else if (j & 1) {
            sum1 += qv * xv;
            if constexpr (kHasMin) sumx1 += xv;
          } else {
            sum0 += qv * xv;
            if constexpr (kHasMin) sumx0 += xv;
          }
        }
#else
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kGroupSize; j += (T == KType::Q2_K ? 1 : 2)) {
          float const x0 = x[int64_t(block) * kQK + g * kGroupSize + j];
          int const q0 = vecdot_kernel_code_at<T>(blk, g, j);
          sum0 += (dl * float(q0) - ml) * x0;
          if constexpr (T != KType::Q2_K) {
            float const x1 = x[int64_t(block) * kQK + g * kGroupSize + j + 1];
            int const q1 = vecdot_kernel_code_at<T>(blk, g, j + 1);
            // Counterfactual: keep group ownership and one final reduction, but apply affine arithmetic per element.
            sum1 += (dl * float(q1) - ml) * x1;
          }
        }
#endif
      }
#if defined(GGUF_VECDOT_PER_ELEMENT_AFFINE)
      lane_acc += sum0 + sum1;
#else
      lane_acc += apply_group({sum0 + sum1, sumx0 + sumx1}, dl, ml);
#endif
    }
  }
  lane_acc = vecdot_subgroup_sum<kLanesPerRow>(lane_acc);
  if (active && row_lane == 0) out[r] = lane_acc;
}

// ONE WARP PER 256 PHYSICAL DESTINATIONS. This is the shape that matters and the numbers say why:
// measured on a 5090 at 8 experts x 2048 columns x 8 superblocks, one thread per block runs 786.5 us and this runs
// 61.4 us -- 12.8x, 1.399 TB/s, 78.1% of peak and 94.5% of a measured streaming-copy roof.
//
// The loss it removes is entirely in the store pattern. With one thread per block a single store instruction has
// lane addresses 512 bytes apart, so a warp touches 32 separate 32-byte sectors for 64 useful bytes: 6.25% sector
// utilisation, exactly 16x worse than lanes writing consecutive elements. The baseline was never DRAM-bound --
// making its data warm in L2 moved it 2.3% -- while this version goes 61.4 us cold to 25.5 us warm, which is what
// being bandwidth-bound looks like.
//
// THE OWNERSHIP IS DERIVED FROM THE DESTINATION. `dst_layout` maps logical (block,element) coordinates to physical
// offsets. We partition a compact physical tensor and the matching identity-coordinate tensor with the SAME copy;
// composing the coordinate tensor with right_inverse(dst_layout) then tells each physical owner which source block
// and element it must decode. That is the measured 3x distinction: partition physical addresses first, derive source
// coordinates second.
//
// right_inverse needs a static compact bijection. The separately named `_logical` kernel below is the general path
// for dynamic, padded or aliased layouts; it stripes logical elements and costs 196.6 us against 65.5 us here on the
// measured nontrivial reorder. Keeping the slow path explicit is preferable to treating right_inverse as meaningful
// on holes, where it can return a coordinate for an address the layout never owns.
template <KType T, class DstLayout>
__global__ void dequantize_kernel_warp(uint8_t const* blocks, int64_t block_bytes, half_t* out,
                                       int n_blocks, DstLayout dst_layout) {
  static_assert(cute::is_static_v<DstLayout>,
                "the physical-partition path needs a static layout; use dequantize_kernel_warp_logical otherwise");
  static_assert(cute::rank_v<decltype(cute::shape(DstLayout{}))> == 2,
                "destination shape must be (source block, element in its 256-element superblock)");
  static_assert(cute::size<1>(DstLayout{}) == kQK, "the destination's element mode must have extent 256");
  static_assert(cute::cosize(DstLayout{}) == cute::size(DstLayout{}),
                "the fast path needs a compact bijection; padded/aliased layouts use the logical fallback");
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  if (warp >= n_blocks) return;

  auto dst_physical = cute::make_tensor(cute::make_gmem_ptr(out),
      cute::make_layout(cute::make_shape(cute::size(dst_layout))));
  auto logical_id = cute::make_identity_tensor(cute::shape(dst_layout));
  auto source_for_physical = cute::composition(logical_id, cute::right_inverse(dst_layout));
  auto source_flat = cute::group_modes<0, cute::rank(source_for_physical)>(source_for_physical);
  auto dst_tile = cute::local_tile(dst_physical, cute::make_shape(cute::Int<kQK>{}), warp);
  auto source_tile = cute::local_tile(source_flat, cute::make_shape(cute::Int<kQK>{}), warp);

  auto tiled_copy = cute::make_tiled_copy(
      cute::Copy_Atom<cute::UniversalCopy<half_t>, half_t>{},
      cute::Layout<cute::Shape<cute::_32>>{}, cute::Layout<cute::Shape<cute::_1>>{});
  auto thr_copy = tiled_copy.get_thread_slice(lane);
  auto thr_dst = thr_copy.partition_D(dst_tile);
  auto thr_source = thr_copy.partition_D(source_tile);
  constexpr int kGs = group_size<T>();
  CUTLASS_PRAGMA_UNROLL
  for (int v = 0; v < cute::size(thr_dst); ++v) {
    auto const coord = thr_source(v);
    int const b = int(cute::get<0>(coord));
    int const i = int(cute::crd2idx(cute::get<1>(coord), cute::shape<1>(dst_layout)));
    uint8_t const* blk = blocks + int64_t(b) * block_bytes;
    float dl, ml;
    group_dl_ml<T>(blk, i / kGs, dl, ml);
    thr_dst(v) = half_t(dl * float(code_at<T>(blk, i)) - ml);
  }
}

// GENERAL WARP FALLBACK. It is correct for any cute Layout, including a runtime shape, padding (cosize > size), and
// aliasing. Logical striping means stores need not be coalesced; that cost is the reason this is not the default.
template <KType T, class DstLayout>
__global__ void dequantize_kernel_warp_logical(uint8_t const* blocks, int64_t block_bytes, half_t* out,
                                               int n_blocks, DstLayout dst_layout) {
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const lane = threadIdx.x & 31;
  if (warp >= n_blocks) return;
  auto dst = cute::make_tensor(cute::make_gmem_ptr(out), dst_layout);
  uint8_t const* blk = blocks + int64_t(warp) * block_bytes;
  constexpr int kGs = group_size<T>();
  CUTLASS_PRAGMA_UNROLL
  for (int v = 0; v < kQK / 32; ++v) {
    int const i = lane + 32 * v;
    float dl, ml;
    group_dl_ml<T>(blk, i / kGs, dl, ml);
    dst(cute::make_coord(warp, cute::idx2crd(i, cute::shape<1>(dst_layout)))) =
        half_t(dl * float(code_at<T>(blk, i)) - ml);
  }
}

// SERIAL DEVICE REFERENCE, also expressed against a cute tensor rather than an opaque address callable.
template <KType T, class DstLayout>
__global__ void dequantize_kernel(uint8_t const* blocks, int64_t block_bytes, half_t* out,
                                  int n_blocks, DstLayout dst_layout) {
  int const b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  auto dst = cute::make_tensor(cute::make_gmem_ptr(out), dst_layout);
  visit<T>(blocks + int64_t(b) * block_bytes, [&](int i, int, int q, float dl, float ml) {
    dst(cute::make_coord(b, cute::idx2crd(i, cute::shape<1>(dst_layout)))) = half_t(dl * float(q) - ml);
  });
}

#endif

}  // namespace vecdot
}  // namespace gguf_scale
