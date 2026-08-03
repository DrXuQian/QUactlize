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
// The cooperative reduction below changes only fp32 summation order inside each group, so its device golden uses an
// explicit summation-order tolerance against this scalar traversal.
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
using Q4CodeShape = cute::Shape<cute::_32, cute::_2, cute::_4>;  // (byte-in-chunk, nibble, chunk)
using Q4CodeByteLayout = cute::Layout<Q4CodeShape, cute::Stride<cute::_1, cute::_0, cute::_32>>;
using Q4CodeShiftLayout = cute::Layout<Q4CodeShape, cute::Stride<cute::_0, cute::_4, cute::_0>>;
CUTLASS_HOST_DEVICE int q4k_code(uint8_t const* qs, int i) {
  return (qs[Q4CodeByteLayout{}(i)] >> Q4CodeShiftLayout{}(i)) & 0xF;
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

// SUBGROUP-COOPERATIVE GEMV. A warp owns RowsPerWarp rows; each power-of-two lane subgroup walks one row's k axis.
// Consecutive lanes therefore read consecutive code bytes, and the work is genuinely partitioned: each logical
// element is decoded by exactly one lane. The subgroup leader alone decodes d/dmin and the group's packed scale
// pair, after which a butterfly reduction supplies sum(q*x) and (for affine formats) sum(x).
//
// This deliberately reassociates each group reduction relative to vecdot_block's scalar left fold. It introduces no
// weight or activation quantisation (dp4a remains forbidden); correctness uses an explicit fp32 summation-order
// tolerance. The outer block and group order remains unchanged so the only reassociation is inside a 16/32-element
// group.
template <int Width>
__device__ __forceinline__ float vecdot_subgroup_sum(float v) {
  static_assert(Width >= 1 && Width <= 32 && (Width & (Width - 1)) == 0, "subgroup width must be a warp divisor");
  CUTLASS_PRAGMA_UNROLL
  for (int off = Width / 2; off > 0; off >>= 1) v += __shfl_xor_sync(~0u, v, off);
  return v;
}

// The cold-data sweep at blocks_per_row={1,2,4,8} selected the same point at every k depth: the formats with a
// separate high-bit plane and no affine min (Q3_K/Q6_K) prefer two rows, while Q2/Q4/Q5 amortise reductions best at
// four. Keeping this format trait explicit makes a plain vecdot_rows_kernel<T> launch use the measured variant.
template <KType T>
CUTLASS_HOST_DEVICE constexpr int vecdot_preferred_rows_per_warp() {
  return (T == KType::Q3_K || T == KType::Q6_K) ? 2 : 4;
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
  static_assert(kGroupSize % kLanesPerRow == 0 || kLanesPerRow % kGroupSize == 0,
                "lane subgroup and quant group must divide one another");

  // Grid-stride row ownership keeps an old launcher correct if it still uses the serial kernel's smaller grid.
  // vecdot_grid_size supplies the measured one-pass grid; either way blockDim must be a warp multiple.
  int const warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int const warp_stride = (gridDim.x * blockDim.x) >> 5;
  int const lane = threadIdx.x & 31;
  int const row_in_warp = lane / kLanesPerRow;
  int const row_lane = lane & (kLanesPerRow - 1);
  for (int row0 = warp * RowsPerWarp; row0 < rows; row0 += warp_stride * RowsPerWarp) {
    int const r = row0 + row_in_warp;
    bool const active = r < rows;
    float acc = 0.f;

    for (int block = 0; block < blocks_per_row; ++block) {
      uint8_t const* blk = active
          ? blocks + (int64_t(r) * blocks_per_row + block) * block_bytes
          : blocks;
      float d = 0.f, dmin = 0.f;
      if (active && row_lane == 0) block_d_dmin<T>(blk, d, dmin);
      float block_acc = 0.f;

      CUTLASS_PRAGMA_UNROLL
      for (int g = 0; g < kGroups; ++g) {
        float sumqx = 0.f, sumx = 0.f;
        if (active) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = row_lane; j < kGroupSize; j += kLanesPerRow) {
            int const i = g * kGroupSize + j;
            float const xv = x[int64_t(block) * kQK + i];
            sumqx += float(code_at<T>(blk, i)) * xv;
            if constexpr (kHasMin) sumx += xv;
          }
        }
        sumqx = vecdot_subgroup_sum<kLanesPerRow>(sumqx);
        if constexpr (kHasMin) sumx = vecdot_subgroup_sum<kLanesPerRow>(sumx);
        if (active && row_lane == 0) {
          float dl, ml;
          group_dl_ml_from_base<T>(blk, g, d, dmin, dl, ml);
          block_acc += apply_group({sumqx, sumx}, dl, ml);
        }
      }
      if (active && row_lane == 0) acc += block_acc;
    }
    if (active && row_lane == 0) out[r] = acc;
  }
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
