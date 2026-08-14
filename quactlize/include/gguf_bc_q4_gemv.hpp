// DENSE Q4_K GEMV FROM THE SHIPPING XPLANE A64 + PACKED-UNIT ARTIFACT.
//
// This is the PDF Q4_K SIMT topology applied to the *shipping* decode operand:
// xplane A64 codes plus byte-neutral packed units.  It neither defines nor
// changes a resident format.  For A64, the producer map proves that one logical
// 32-code group is one contiguous uint4 at
//
//   low[((sb * N + n) * 8 + group) * 16]
//
// and the only remaining difference from native GGUF is the fixed P4x32 nibble
// permutation inside that uint4.  Each lane owns one group, exactly as the PDF
// kernel's eight lanes own the eight metadata groups of one superblock.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gguf_bc_q4_reader.hpp"
#include "gguf_packed_unit.hpp"
#include "gemv_lowbit/gemv_common.hpp"

namespace gguf_scale::bc_q4_gemv {

constexpr int kQK = 256;
constexpr int kGroups = 8;
constexpr int kGroupK = 32;
constexpr int kThreadsPerSb = 8;
constexpr int kSbPerWarp = 4;
constexpr int kElemsPerWarp = kQK * kSbPerWarp;
constexpr int kUnitBytes = 16;
constexpr int kCodeBytesPerSb = 128;

static_assert(packed_unit::Unit<KType::Q4_K>::kUnitTotal == kUnitBytes,
              "Q4_K packed-unit ABI changed");
static_assert(bc_vecdot::q4_reader::Q4WordPlan<64>::kWords == 4,
              "A64 Q4 group is no longer four words");

using ScaleZero = bc_vecdot::q4_reader::Q4NativeScaleZero;

// Decode one lane's scale/min pair from the already-loaded 16-byte packed
// unit.  The 96 payload bits are two self-contained 48-bit runs of four groups.
// Keeping the uint4 as the input is important: all eight lanes issue the same
// 16-byte metadata address and the hardware can broadcast/merge that load.
__device__ __forceinline__ ScaleZero decode_scale_zero(uint4 m, unsigned group) {
  return bc_vecdot::q4_reader::decode_scale_zero({m.x, m.y, m.z, m.w}, group);
}

template <int Word, int Pair>
__device__ __forceinline__ half2 activation_pair(half2 const (&logical)[16]) {
  using Plan = bc_vecdot::q4_reader::Q4WordPlan<64>;
  constexpr int p0 = 8 * Word + Plan::physical_nibble_from_pair_lane(Pair, 0);
  constexpr int p1 = 8 * Word + Plan::physical_nibble_from_pair_lane(Pair, 1);
  constexpr int k0 = Plan::logical_k_from_physical_nibble(p0);
  constexpr int k1 = Plan::logical_k_from_physical_nibble(p1);
  half const a0 = (k0 & 1) ? logical[k0 / 2].y : logical[k0 / 2].x;
  half const a1 = (k1 & 1) ? logical[k1 / 2].y : logical[k1 / 2].x;
  return __halves2half2(a0, a1);
}

template <int Word>
__device__ __forceinline__ void dot_word(std::uint32_t packed,
                                         half2 const (&logical)[16],
                                         half2 scale, half2 zero,
                                         half2& d0, half2& d1,
                                         half2& d2, half2& d3) {
  auto const q = bc_vecdot::q4_reader::dequantize_word(packed);
  d0 = __hfma2(__hfma2(q.pair[0], scale, zero), activation_pair<Word, 0>(logical), d0);
  d1 = __hfma2(__hfma2(q.pair[1], scale, zero), activation_pair<Word, 1>(logical), d1);
  d2 = __hfma2(__hfma2(q.pair[2], scale, zero), activation_pair<Word, 2>(logical), d2);
  d3 = __hfma2(__hfma2(q.pair[3], scale, zero), activation_pair<Word, 3>(logical), d3);
}

template <int CTA_N, int WARPS_N, int WARPS_K = 1>
__global__ void __launch_bounds__(WARPS_N * WARPS_K * 32,
                                  1024 / (WARPS_N * WARPS_K * 32))
kernel(half const* act, std::uint8_t const* low, std::uint8_t const* units,
       float* out, unsigned m, unsigned n, unsigned k) {
  static_assert(CTA_N >= 1 && CTA_N <= 16, "CTA_N out of range");
  static_assert(WARPS_N >= 1, "WARPS_N must be positive");
  static_assert(WARPS_K == 1,
                "benchmark-first A64 topology intentionally proves the PDF winning Wk1 path first");
  constexpr int Threads = WARPS_N * WARPS_K * 32;

  unsigned const tid = threadIdx.x;
  unsigned const warp_id = tid >> 5;
  unsigned const lane = tid & 31u;
  unsigned const sb_in_warp = lane / kThreadsPerSb;
  unsigned const group = lane & (kThreadsPerSb - 1);
  unsigned const n_group = warp_id / WARPS_K;
  unsigned const k_warp = warp_id % WARPS_K;
  unsigned const row = blockIdx.x;
  unsigned const col_start = blockIdx.y * (CTA_N * WARPS_N) + n_group * CTA_N;
  extern __shared__ float4 shared_raw[];
  half* const shared_act = reinterpret_cast<half*>(shared_raw);
  unsigned const n_vec = k / 8;
  float4 const* src = reinterpret_cast<float4 const*>(act + row * k);
  for (unsigned i = tid; i < n_vec; i += Threads) shared_raw[i] = src[i];
  __syncthreads();

  float acc[CTA_N];
#pragma unroll
  for (int ii = 0; ii < CTA_N; ++ii) acc[ii] = 0.0f;

  unsigned const chunks = k / kElemsPerWarp;
  for (unsigned chunk = k_warp; chunk < chunks; chunk += WARPS_K) {
    unsigned const sb = chunk * kSbPerWarp + sb_in_warp;
    unsigned const group_k = sb * kQK + group * kGroupK;

    // The P4x32 register permutation is fixed, so load logical activation order
    // once and let compile-time pair maps select registers below.
    half2 logical_x[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
      logical_x[j] = *reinterpret_cast<half2 const*>(shared_act + group_k + 2 * j);

    uint4 metadata[CTA_N], quant[CTA_N];
    uint4 const* mp = reinterpret_cast<uint4 const*>(units) + std::size_t(sb) * n + col_start;
    uint4 const* qp = reinterpret_cast<uint4 const*>(low)
                    + (std::size_t(sb) * n + col_start) * kGroups + group;
#pragma unroll
    for (int ii = 0; ii < CTA_N; ++ii) {
      metadata[ii] = *mp++;
      quant[ii] = *qp;
      qp += kGroups;
    }

#pragma unroll
    for (int ii = 0; ii < CTA_N; ++ii) {
      ScaleZero const sz = decode_scale_zero(metadata[ii], group);
      half2 const scale = __half2half2(sz.scale);
      half2 const zero = __half2half2(sz.zero);
      half2 d0 = __float2half2_rn(0.0f);
      half2 d1 = __float2half2_rn(0.0f);
      half2 d2 = __float2half2_rn(0.0f);
      half2 d3 = __float2half2_rn(0.0f);
      uint4 const q = quant[ii];
      dot_word<0>(q.x, logical_x, scale, zero, d0, d1, d2, d3);
      dot_word<1>(q.y, logical_x, scale, zero, d0, d1, d2, d3);
      dot_word<2>(q.z, logical_x, scale, zero, d0, d1, d2, d3);
      dot_word<3>(q.w, logical_x, scale, zero, d0, d1, d2, d3);
      half2 const sum = __hadd2(__hadd2(d0, d1), __hadd2(d2, d3));
      acc[ii] += __half2float(__low2half(sum)) + __half2float(__high2half(sum));
    }
  }

#pragma unroll
  for (int ii = 0; ii < CTA_N; ++ii) {
    float v = acc[ii];
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    if (lane == 0 && col_start + unsigned(ii) < n)
      out[row * n + col_start + unsigned(ii)] = v;
  }
}

template <int CTA_N, int WARPS_N, int WARPS_K = 1>
inline void launch(half const* act, std::uint8_t const* low,
                   std::uint8_t const* units, float* out,
                   unsigned m, unsigned n, unsigned k,
                   gemv_stream_t stream = nullptr) {
  constexpr int Threads = WARPS_N * WARPS_K * 32;
  dim3 const grid(m, (n + CTA_N * WARPS_N - 1) / (CTA_N * WARPS_N), 1);
  std::size_t const shared_bytes = std::size_t(k) * sizeof(half);
  kernel<CTA_N, WARPS_N, WARPS_K><<<grid, Threads, shared_bytes, stream>>>(
      act, low, units, out, m, n, k);
}

// The first shipping point is selected from the complete same-binary 5090
// sweep, rather than copied from the raw-GGUF PDF kernel.  Keep the admission
// predicate beside it: the current K loop owns four superblocks per warp and
// therefore must not silently drop a non-1024 tail.  Unsupported shapes remain
// on the arrangement-aware generic BC reader.
constexpr bool default_admits(unsigned m, unsigned n, unsigned k) {
  // This is a measured specialization, not a universal replacement.  Its
  // vector metadata/code loads have no tail predicate and its one-CTA A stage
  // intentionally stays below the portable default dynamic-smem floor.
  // Every rejected shape retains the generic arrangement-aware reader.
  constexpr unsigned kOutputColumns = 2 * 4;
  constexpr unsigned kLargestAdmittedK = 8192;
  return m == 1 && n != 0 && (n % kOutputColumns) == 0 &&
         k >= kElemsPerWarp && k <= kLargestAdmittedK &&
         (k % kElemsPerWarp) == 0;
}

static_assert(default_admits(1, 4096, 4096), "measured Q4/A64 shape must remain admitted");
static_assert(!default_admits(2, 4096, 4096) &&
              !default_admits(1, 4097, 4096) &&
              !default_admits(1, 4096, 4352) &&
              !default_admits(1, 4096, 32768),
              "Q4/A64 production specialization must fail closed outside its proved domain");

inline bool launch_default(half const* act, std::uint8_t const* low,
                           std::uint8_t const* units, float* out,
                           unsigned m, unsigned n, unsigned k,
                           gemv_stream_t stream = nullptr) {
  if (!default_admits(m, n, k)) return false;
  launch<2, 4, 1>(act, low, units, out, m, n, k, stream);
  return true;
}

}  // namespace gguf_scale::bc_q4_gemv
