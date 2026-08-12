// Q4_K GEMV reconstructed from the 22-page document supplied for INBOX 132.
//
// SOURCE BOUNDARY -- this is not an upstream/exact source file.  Pages 1--11
// contain the helpers and kernel body.  Page 12 truncates launch_q4k_gemv()
// after two assertions; page 14 gives the missing grid/block/dynamic-smem
// launch geometry, which is restored below.  The PDF contains no source URL,
// attachment, host packer, golden, allocation code, or timer.
//
// The document also shows two metadata-conversion bodies: the main listing
// uses scalar __ushort2half_rn (pages 9--10), while the explanatory listing
// uses a paired magic-number conversion (pages 18--20).  PairMetadata is kept
// as an explicit arm so measurements never silently mix those two versions.
#pragma once

#include <cassert>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace q4k_pdf_reconstruction {

struct block_q4_K {
  half d;
  half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};

static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q4_K) % 16 == 0,
              "the PDF's uint4 loads require a 16-byte block stride");

inline constexpr int QK_K = 256;
inline constexpr int THREADS_PER_SB = 8;
inline constexpr int BLOCKS_PER_WARP = 4;
inline constexpr int ELEMS_PER_WARP = 1024;

__device__ __forceinline__ std::uint32_t byte_of(std::uint32_t w, unsigned i) {
  return (w >> (8 * i)) & 0xffu;
}

__device__ __forceinline__ void get_scale_min_k4_w(
    unsigned j, std::uint32_t s0, std::uint32_t s1, std::uint32_t s2,
    std::uint8_t& d, std::uint8_t& m) {
  unsigned const i = j & 3u;
  std::uint32_t const b0 = byte_of(s0, i);
  std::uint32_t const b1 = byte_of(s1, i);
  std::uint32_t const b2 = byte_of(s2, i);
  std::uint32_t const d_lo = b0 & 63u;
  std::uint32_t const m_lo = b1 & 63u;
  std::uint32_t const d_hi = (b2 & 0x0fu) | ((b0 >> 6) << 4);
  std::uint32_t const m_hi = (b2 >> 4) | ((b1 >> 6) << 4);
  bool const hi = j >= 4;
  d = static_cast<std::uint8_t>(hi ? d_hi : d_lo);
  m = static_cast<std::uint8_t>(hi ? m_hi : m_lo);
}

struct q_h2x4 {
  half2 q0;
  half2 q1;
  half2 q2;
  half2 q3;
};

__device__ __forceinline__ q_h2x4 lop3_convert_to_h2(std::uint32_t raw) {
  std::uint32_t i4s;
  asm volatile("prmt.b32 %0, %1, %1, 0x3120;\n" : "=r"(i4s) : "r"(raw));
  std::uint32_t h[4];
  std::uint32_t const top = i4s >> 8;
  // The PDF's comment says 0xe4, but the expression and required truth table
  // are 0xea.  Preserve the source expression, not the stale prose.
  constexpr std::uint32_t imm_lut = (0xf0u & 0xccu) | 0xaau;
  constexpr std::uint32_t bottom_mask = 0x000f000fu;
  constexpr std::uint32_t top_mask = 0x00f000f0u;
  constexpr std::uint32_t magic = 0x64006400u;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0]) : "r"(i4s), "n"(bottom_mask), "n"(magic), "n"(imm_lut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1]) : "r"(i4s), "n"(top_mask), "n"(magic), "n"(imm_lut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2]) : "r"(top), "n"(bottom_mask), "n"(magic), "n"(imm_lut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3]) : "r"(top), "n"(top_mask), "n"(magic), "n"(imm_lut));

  // The PDF spells these three inputs with an immediate ("n") inline-asm
  // constraint.  CUDA 12.8 emits literal PTX operands which ptxas rejects for
  // sub/fma.f16x2.  Register constraints preserve the documented operations
  // and are the minimum sm_120 reconstruction needed to make them legal.
  std::uint32_t const top_magic = 0x64086408u;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(top_magic));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(top_magic));
  std::uint32_t const one_sixteenth = 0x2c002c00u;
  std::uint32_t const neg_72 = 0xd480d480u;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[1]) : "r"(h[1]), "r"(one_sixteenth), "r"(neg_72));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(h[3]) : "r"(h[3]), "r"(one_sixteenth), "r"(neg_72));
  union Bits { std::uint32_t u; half2 h2; } u0{}, u1{}, u2{}, u3{};
  u0.u = h[0]; u1.u = h[1]; u2.u = h[2]; u3.u = h[3];
  return {u0.h2, u1.h2, u2.h2, u3.h2};
}

__device__ __forceinline__ half2 u6_pair_to_half2(std::uint32_t lo, std::uint32_t hi) {
  union Bits { std::uint32_t u; half2 h2; } v{}, bias{};
  v.u = 0x64006400u | (lo & 0xffu) | ((hi & 0xffu) << 16);
  bias.u = 0x64006400u;
  return __hsub2(v.h2, bias.h2);
}

__device__ __forceinline__ void q4k_dot_word(
    std::uint32_t qword,
    half2 a_lo0, half2 a_lo1, half2 a_hi0, half2 a_hi1,
    half2 scale_lo2, half2 zero_lo2, half2 scale_hi2, half2 zero_hi2,
    half2& dq0, half2& dq1, half2& dq2, half2& dq3) {
  q_h2x4 const qm = lop3_convert_to_h2(qword);
  dq0 = __hfma2(__hfma2(qm.q0, scale_lo2, zero_lo2), a_lo0, dq0);
  dq1 = __hfma2(__hfma2(qm.q1, scale_hi2, zero_hi2), a_hi0, dq1);
  dq2 = __hfma2(__hfma2(qm.q2, scale_lo2, zero_lo2), a_lo1, dq2);
  dq3 = __hfma2(__hfma2(qm.q3, scale_hi2, zero_hi2), a_hi1, dq3);
}

template <int CTA_N, int WARPS_N, int WARPS_K, bool PairMetadata>
__global__ void __launch_bounds__(WARPS_N * WARPS_K * 32,
                                  1024 / (WARPS_N * WARPS_K * 32))
q4k_gemv_kernel(half const* act, block_q4_K const* w, half* out,
                 unsigned m, unsigned n, unsigned k) {
  static_assert(CTA_N >= 1 && CTA_N <= 32, "CTA_N out of range");
  static_assert(WARPS_N >= 1 && WARPS_K >= 1, "warp factors must be positive");
  static_assert(WARPS_N * WARPS_K * 32 <= 1024, "CTA exceeds 1024 threads");
  (void)m;
  constexpr int threads = WARPS_N * WARPS_K * 32;
  unsigned const tid = threadIdx.x;
  unsigned const warp_id = tid / 32;
  unsigned const lane = tid % 32;
  unsigned const sb_in_warp = lane / THREADS_PER_SB;
  unsigned const t_in_sb = lane % THREADS_PER_SB;
  unsigned const n_group = warp_id / WARPS_K;
  unsigned const k_warp = warp_id % WARPS_K;
  unsigned const row = blockIdx.x;
  unsigned const col_start = blockIdx.y * (CTA_N * WARPS_N) + n_group * CTA_N;
  unsigned const blocks_per_row = k / QK_K;

  extern __shared__ float4 sh_raw[];
  half* const sh_act = reinterpret_cast<half*>(sh_raw);
  unsigned const n_vec = k / 8;
  float4 const* src = reinterpret_cast<float4 const*>(act + row * k);
  for (unsigned i = tid; i < n_vec; i += threads) sh_raw[i] = src[i];
  __syncthreads();

  float acc[CTA_N];
#pragma unroll
  for (int i = 0; i < CTA_N; ++i) acc[i] = 0.0f;

  unsigned const chunks = k / ELEMS_PER_WARP;
  for (unsigned chunk = k_warp; chunk < chunks; chunk += WARPS_K) {
    unsigned const block_idx = chunk * BLOCKS_PER_WARP + sb_in_warp;
    unsigned const k_lo = chunk * ELEMS_PER_WARP + sb_in_warp * QK_K
                        + (t_in_sb / 2) * 64 + (t_in_sb % 2) * 16;
    unsigned const k_hi = k_lo + 32;

    half2 act_lo2[8], act_hi2[8];
    float4 const* src_lo = reinterpret_cast<float4 const*>(sh_act + k_lo);
    float4 const* src_hi = reinterpret_cast<float4 const*>(sh_act + k_hi);
    reinterpret_cast<float4*>(act_lo2)[0] = src_lo[0];
    reinterpret_cast<float4*>(act_lo2)[1] = src_lo[1];
    reinterpret_cast<float4*>(act_hi2)[0] = src_hi[0];
    reinterpret_cast<float4*>(act_hi2)[1] = src_hi[1];

    uint4 m4[CTA_N], qs4[CTA_N];
    block_q4_K const* bq = w + col_start * blocks_per_row + block_idx;
#pragma unroll
    for (int ii = 0; ii < CTA_N; ++ii, bq += blocks_per_row) {
      m4[ii] = *reinterpret_cast<uint4 const*>(bq);
      qs4[ii] = *reinterpret_cast<uint4 const*>(bq->qs + t_in_sb * 16);
    }

#pragma unroll
    for (int ii = 0; ii < CTA_N; ++ii) {
      std::uint32_t const meta0 = m4[ii].x;
      std::uint32_t const meta1 = m4[ii].y;
      std::uint32_t const meta2 = m4[ii].z;
      std::uint32_t const meta3 = m4[ii].w;
      half const d = __ushort_as_half(static_cast<unsigned short>(meta0 & 0xffffu));
      half const dminn = __ushort_as_half(static_cast<unsigned short>((meta0 >> 16) ^ 0x8000u));
      unsigned const j_lo = (t_in_sb / 2) * 2;
      unsigned const j_hi = j_lo + 1;
      std::uint8_t sc_lo, m_lo, sc_hi, m_hi;
      get_scale_min_k4_w(j_lo, meta1, meta2, meta3, sc_lo, m_lo);
      get_scale_min_k4_w(j_hi, meta1, meta2, meta3, sc_hi, m_hi);
      half const d8 = __hmul(d, __ushort_as_half(static_cast<unsigned short>(0x4800)));

      half2 scale_lo2, scale_hi2, zero_lo2, zero_hi2;
      if constexpr (PairMetadata) {
        half2 const sc2 = u6_pair_to_half2(sc_lo, sc_hi);
        half2 const mn2 = u6_pair_to_half2(m_lo, m_hi);
        half2 const scale2 = __hmul2(__half2half2(d), sc2);
        half2 const zero2 = __hfma2(__half2half2(d8), sc2,
                                    __hmul2(__half2half2(dminn), mn2));
        scale_lo2 = __half2half2(__low2half(scale2));
        scale_hi2 = __half2half2(__high2half(scale2));
        zero_lo2 = __half2half2(__low2half(zero2));
        zero_hi2 = __half2half2(__high2half(zero2));
      } else {
        half const sc_lo_h = __ushort2half_rn(sc_lo);
        half const mn_lo_h = __ushort2half_rn(m_lo);
        half const sc_hi_h = __ushort2half_rn(sc_hi);
        half const mn_hi_h = __ushort2half_rn(m_hi);
        scale_lo2 = __half2half2(__hmul(d, sc_lo_h));
        scale_hi2 = __half2half2(__hmul(d, sc_hi_h));
        zero_lo2 = __half2half2(__hfma(d8, sc_lo_h, __hmul(dminn, mn_lo_h)));
        zero_hi2 = __half2half2(__hfma(d8, sc_hi_h, __hmul(dminn, mn_hi_h)));
      }

      uint4 const qs = qs4[ii];
      half2 dq0 = __float2half2_rn(0.f), dq1 = dq0, dq2 = dq0, dq3 = dq0;
      q4k_dot_word(qs.x, act_lo2[0], act_lo2[1], act_hi2[0], act_hi2[1],
                   scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
      q4k_dot_word(qs.y, act_lo2[2], act_lo2[3], act_hi2[2], act_hi2[3],
                   scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
      q4k_dot_word(qs.z, act_lo2[4], act_lo2[5], act_hi2[4], act_hi2[5],
                   scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
      q4k_dot_word(qs.w, act_lo2[6], act_lo2[7], act_hi2[6], act_hi2[7],
                   scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
      half2 const sum = __hadd2(__hadd2(dq0, dq1), __hadd2(dq2, dq3));
      acc[ii] += __half2float(__low2half(sum)) + __half2float(__high2half(sum));
    }
  }

#pragma unroll
  for (int col = 0; col < CTA_N; ++col) {
    float v = acc[col];
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 16);
    acc[col] = v;
  }

  if constexpr (WARPS_K == 1) {
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < CTA_N; ++col)
        out[row * n + col_start + col] = __float2half(acc[col]);
    }
  } else {
    __shared__ float red[WARPS_N][WARPS_K][CTA_N];
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < CTA_N; ++col) red[n_group][k_warp][col] = acc[col];
    }
    __syncthreads();
    if (k_warp == 0) {
      for (unsigned col = lane; col < CTA_N; col += 32) {
        float sum = 0.f;
#pragma unroll
        for (int kw = 0; kw < WARPS_K; ++kw) sum += red[n_group][kw][col];
        out[row * n + col_start + col] = __float2half(sum);
      }
    }
  }
}

template <int CTA_N, int WARPS_N = 8, int WARPS_K = 1, bool PairMetadata = false>
inline void launch_q4k_gemv(half const* act, block_q4_K const* w, half* out,
                             int m, int n, int k, cudaStream_t stream = nullptr) {
  constexpr int threads = WARPS_N * WARPS_K * 32;
  constexpr int cols_per_cta = CTA_N * WARPS_N;
  static_assert(threads <= 1024, "CTA exceeds 1024 threads");
  assert(k % ELEMS_PER_WARP == 0);
  assert(n % cols_per_cta == 0);
  dim3 const grid(m, n / cols_per_cta, 1);
  dim3 const block(threads, 1, 1);
  std::size_t const smem = std::size_t(k) * sizeof(half);
  q4k_gemv_kernel<CTA_N, WARPS_N, WARPS_K, PairMetadata>
      <<<grid, block, smem, stream>>>(act, w, out, unsigned(m), unsigned(n), unsigned(k));
}

}  // namespace q4k_pdf_reconstruction
