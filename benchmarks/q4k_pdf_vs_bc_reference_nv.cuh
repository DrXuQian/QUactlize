// BENCHMARK-ONLY NVIDIA REFERENCE — copied from the user-supplied Q4_K SIMT GEMV
// in .coord/ref_q4k_gemv_simt.cuh (handover 2026-08-14). This is not a
// production route. The only semantic edits are the four NVIDIA arithmetic asm
// sites: sub.f16x2/fma.rn.f16x2 constants use register operands because ptxas
// rejects immediate operands for these instructions. The lop3 truth table, loads,
// accumulation, launch geometry, and output type remain source-identical.


#pragma once
#include <cassert>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace q4k_gemv
{

struct block_q4_K
{
    half     d;
    half     dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");
static_assert(sizeof(block_q4_K) % 16 == 0,
              "block stride must be a multiple of 16B: the uint4 loads of the "
              "header and of qs rely on every block being 16B aligned");

constexpr int QK_K               = 256;   // weights per super-block
constexpr int K_SCALE_SIZE       = 12;    // packed (sc, m) bytes per super-block
constexpr int THREADS_PER_SB     = 8;     // lanes cooperating on one super-block
constexpr int BLOCKS_PER_WARP    = 4;     // super-blocks per warp per iteration
constexpr int ELEMS_PER_WARP     = 1024;  // BLOCKS_PER_WARP * QK_K

// ----------------------------------------------------------------------------
// get_scale_min_k4_w: unpack the 6-bit (sc, m) pair of group j.
//
// Same bit layout as get_scale_min_k4() in llama.cpp, but scales[12] arrives as
// three uint32 words and the byte selection is a register shift instead of an
// index into a byte array. Indexing a local byte array with a runtime j would
// force the metadata to memory.
//
// Branch-free: both the j < 4 and the j >= 4 form share the byte index
// i = j & 3 (equal to j when j < 4, to j - 4 otherwise), so the three bytes are
// fetched once, both candidate results are computed, and one is selected.
// A data-dependent branch here would diverge, since lanes 0..3 and 4..7 of a
// super-block group take opposite sides.
// ----------------------------------------------------------------------------
__device__ __forceinline__
uint32_t byte_of(uint32_t w, unsigned i)
{
    return (w >> (8 * i)) & 0xFF;
}

__device__ __forceinline__
void get_scale_min_k4_w(unsigned j, uint32_t s0, uint32_t s1, uint32_t s2,
                        uint8_t& d, uint8_t& m)
{
    // s0 = scales[0..3], s1 = scales[4..7], s2 = scales[8..11]
    unsigned const i  = j & 3;
    uint32_t const b0 = byte_of(s0, i);
    uint32_t const b1 = byte_of(s1, i);
    uint32_t const b2 = byte_of(s2, i);

    uint32_t const d_lo = b0 & 63;                                // j < 4
    uint32_t const m_lo = b1 & 63;
    uint32_t const d_hi = (b2 & 0x0F) | ((b0 >> 6) << 4);         // j >= 4
    uint32_t const m_hi = (b2 >> 4)   | ((b1 >> 6) << 4);

    bool const hi = (j >= 4);
    d = (uint8_t)(hi ? d_hi : d_lo);
    m = (uint8_t)(hi ? m_hi : m_lo);
}

// Four half2 values as named scalars rather than an array: an array written
// through a pointer parameter blocks register promotion and ends up in local
// memory.
struct q_h2x4
{
    half2 q0;   // low  nibbles, elements (0, 1)
    half2 q1;   // high nibbles, elements (0, 1) + 32
    half2 q2;   // low  nibbles, elements (2, 3)
    half2 q3;   // high nibbles, elements (2, 3) + 32
};

// ----------------------------------------------------------------------------
// lop3_convert_to_h2: 32-bit quant word (8 nibbles) -> 4 half2 holding (q - 8),
// with no integer-to-float conversion instructions.
//
// The trick (magic constants from CUTLASS FastInterleavedAndBiased-
// NumericArrayConverter): 0x6400 is the fp16 encoding of 1024, whose mantissa
// occupies the low 10 bits. OR-ing a 4-bit value into that mantissa yields
// fp16(1024 + q) exactly, so one lop3 masks a nibble and inserts the exponent
// in a single instruction, and a following subtract of 1032 = 1024 + 8 leaves
// fp16(q - 8).
//
// High nibbles sit in bits 4..7, i.e. the inserted mantissa is 16*q, so their
// path multiplies by 1/16 and subtracts 72 = (1024 + 8*16)/16 instead, fused
// into one fma.
//
// The leading prmt permutes the bytes [b0,b1,b2,b3] -> [b0,b2,b1,b3] so that
// each output half2 holds two *adjacent* elements. That matches the register
// layout of the float4 activation loads, so the activation side needs no
// packing at all.
// ----------------------------------------------------------------------------
__device__ __forceinline__
q_h2x4 lop3_convert_to_h2(uint32_t i4s_raw)
{
    uint32_t i4s;
    asm volatile("prmt.b32 %0, %1, %1, 0x3120;\n" : "=r"(i4s) : "r"(i4s_raw));

    uint32_t h[4];
    uint32_t const top_i4s = i4s >> 8;

    constexpr uint32_t immLut       = (0xf0 & 0xcc) | 0xaa; // 0xe4 = (A & B) | C
    constexpr uint32_t BOTTOM_MASK  = 0x000f000f;
    constexpr uint32_t TOP_MASK     = 0x00f000f0;
#if defined(Q4K_PDF_PLANT_WRONG_MAGIC)
    // Benchmark-only negative control: int8's +128 mantissa anchor cannot
    // decode unsigned Q4 nibbles. The independent raw-Q4 oracle must reject it.
    constexpr uint32_t I4_MAGIC     = 0x64806480;
#else
    constexpr uint32_t I4_MAGIC     = 0x64006400;           // half2{1024, 1024}
#endif

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s),     "n"(BOTTOM_MASK), "n"(I4_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s),     "n"(TOP_MASK),    "n"(I4_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4_MAGIC), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK),    "n"(I4_MAGIC), "n"(immLut));

    // low nibbles: fp16(1024 + q) - 1032 = fp16(q - 8)
    constexpr uint32_t FP16_TOP_MAGIC = 0x64086408; // half2{1032, 1032}
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC));

    // high nibbles: mantissa holds 16*q, so scale by 1/16 and offset by -72
    constexpr uint32_t ONE_SIXTEENTH  = 0x2c002c00; // half2{1/16, 1/16}
    constexpr uint32_t NEG_72         = 0xd480d480; // half2{-72, -72}
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

    // uint32 -> half2 reinterpretation: register renaming, no instruction
    union { uint32_t u; half2 h2; } u0, u1, u2, u3;
    u0.u = h[0]; u1.u = h[1]; u2.u = h[2]; u3.u = h[3];
    return { u0.h2, u1.h2, u2.h2, u3.h2 };
}

// ----------------------------------------------------------------------------
// q4k_dot_word: dequantize one 32-bit quant word (8 weights) and accumulate
// its contribution. All parameters are scalars so the call site can unroll by
// hand; the four accumulators are independent chains.
// ----------------------------------------------------------------------------
__device__ __forceinline__
void q4k_dot_word(uint32_t qword,
                  half2 a_lo0, half2 a_lo1, half2 a_hi0, half2 a_hi1,
                  half2 scale_lo2, half2 zero_lo2,
                  half2 scale_hi2, half2 zero_hi2,
                  half2& dq0, half2& dq1, half2& dq2, half2& dq3)
{
    q_h2x4 const qm = lop3_convert_to_h2(qword);
    // inner hfma2: w = (q - 8) * scale + zero;  outer hfma2: acc += w * a
    dq0 = __hfma2(__hfma2(qm.q0, scale_lo2, zero_lo2), a_lo0, dq0);
    dq1 = __hfma2(__hfma2(qm.q1, scale_hi2, zero_hi2), a_hi0, dq1);
    dq2 = __hfma2(__hfma2(qm.q2, scale_lo2, zero_lo2), a_lo1, dq2);
    dq3 = __hfma2(__hfma2(qm.q3, scale_hi2, zero_hi2), a_hi1, dq3);
}

// ----------------------------------------------------------------------------
// kernel
//   CTA_N    columns per warp
//   WARPS_N  warp groups along the column axis
//   WARPS_K  warps splitting the k axis inside a group
// ----------------------------------------------------------------------------
template <int CTA_N, int WARPS_N, int WARPS_K>
__global__ void __launch_bounds__(WARPS_N * WARPS_K * 32,
                                  1024 / (WARPS_N * WARPS_K * 32))
q4k_gemv_kernel(
    const half*        act,
    const block_q4_K*  w,
    half*              out,
    unsigned m, unsigned n, unsigned k)
{
    static_assert(CTA_N >= 1 && CTA_N <= 32, "CTA_N out of range");
    static_assert(WARPS_N >= 1 && WARPS_K >= 1, "WARPS_N/WARPS_K must be >= 1");
    static_assert(WARPS_N * WARPS_K * 32 <= 1024, "CTA exceeds 1024 threads");
    constexpr int WARPS   = WARPS_N * WARPS_K;
    constexpr int THREADS = WARPS * 32;

    // Unsigned throughout: signed / and % expand into a sign-correcting
    // sequence, unsigned ones become a single shift or and.
    unsigned const tid        = threadIdx.x;
    unsigned const warp_id    = tid / 32;
    unsigned const lane_id    = tid % 32;
    unsigned const sb_in_warp = lane_id / THREADS_PER_SB;
    unsigned const t_in_sb    = lane_id % THREADS_PER_SB;

    unsigned const n_group    = warp_id / WARPS_K;   // which column group
    unsigned const k_warp     = warp_id % WARPS_K;   // which k slice

    unsigned const row        = blockIdx.x;
    unsigned const col_start  = blockIdx.y * (CTA_N * WARPS_N) + n_group * CTA_N;
    unsigned const blocks_per_row = k / QK_K;

    // Stage the whole activation row in dynamic shared memory. Declared as
    // float4 so the base is 16-byte aligned for the float4 reads below.
    extern __shared__ float4 sh_raw[];
    half* const sh_act = reinterpret_cast<half*>(sh_raw);
    {
        // Cooperative copy, 16 bytes per thread, fully coalesced.
        // k % ELEMS_PER_WARP == 0 implies k is divisible by 8, and the row base
        // is 16-byte aligned, so the float4 view is valid.
        unsigned const n_vec = k / 8;
        float4 const* src = reinterpret_cast<float4 const*>(act + row * k);
        for (unsigned i = tid; i < n_vec; i += THREADS)
            sh_raw[i] = src[i];
    }
    __syncthreads();   // only barrier in the kernel; sh_act is read-only after

    // fp32 accumulator, one per column
    float acc[CTA_N];
#pragma unroll
    for (int ii = 0; ii < CTA_N; ++ii)
        acc[ii] = 0.0f;

    // Strided chunk loop; one chunk is ELEMS_PER_WARP elements of k.
    // WARPS_K need not divide chunks: warps of a group then differ by at most
    // one chunk of work.
    unsigned const chunks = k / ELEMS_PER_WARP;
    for (unsigned chunk = k_warp; chunk < chunks; chunk += WARPS_K)
    {
        unsigned const block_idx = chunk * BLOCKS_PER_WARP + sb_in_warp;

        // ---- stage 1: issue every load first, no arithmetic in between ----

        // Activation: two 16-element runs, 32 elements apart, read from shared
        // memory with absolute k indices (the whole row is resident).
        unsigned const k_lo = chunk * ELEMS_PER_WARP + sb_in_warp * QK_K
                            + (t_in_sb / 2) * 64 + (t_in_sb % 2) * 16;
        unsigned const k_hi = k_lo + 32;

        // Element pairs (2j, 2j+1) already sit in the right register lanes for
        // the half2 arithmetic below, so no shuffling or packing is needed.
        half2 act_lo2[8], act_hi2[8];
        {
            float4 const* src_lo = reinterpret_cast<float4 const*>(sh_act + k_lo);
            float4 const* src_hi = reinterpret_cast<float4 const*>(sh_act + k_hi);
            reinterpret_cast<float4*>(act_lo2)[0] = src_lo[0];
            reinterpret_cast<float4*>(act_lo2)[1] = src_lo[1];
            reinterpret_cast<float4*>(act_hi2)[0] = src_hi[0];
            reinterpret_cast<float4*>(act_hi2)[1] = src_hi[1];
        }

        // Weights: header + quant bytes for every column, back to back.
        // These arrays are only ever indexed by the unrolled loop counter so
        // they stay in registers; spilling them would add local-memory traffic
        // in the hot loop.
        uint4 m4[CTA_N], qs4[CTA_N];
        {
            // Pointer increment instead of a multiply per column.
            block_q4_K const* bq = w + col_start * blocks_per_row + block_idx;
#pragma unroll
            for (int ii = 0; ii < CTA_N; ++ii, bq += blocks_per_row)
            {
                // Header is exactly 16 bytes: d(2) + dmin(2) + scales(12).
                // All 8 lanes of the group read the same address; the hardware
                // merges that into one transaction.
                m4[ii]  = *reinterpret_cast<uint4 const*>(bq);
                qs4[ii] = *reinterpret_cast<uint4 const*>(bq->qs + t_in_sb * 16);
            }
        }

        // ---- stage 2: dequantize and accumulate, no loads ----
#pragma unroll
        for (int ii = 0; ii < CTA_N; ++ii)
        {
            // d, dmin and scales are per (column, super-block). Kept as plain
            // scalars: taking the address of a local would force it to memory.
            uint32_t const meta0 = m4[ii].x;   // d | dmin << 16
            uint32_t const meta1 = m4[ii].y;   // scales[0..3]
            uint32_t const meta2 = m4[ii].z;   // scales[4..7]
            uint32_t const meta3 = m4[ii].w;   // scales[8..11]

            half const d     = __ushort_as_half((unsigned short)(meta0 & 0xFFFF));
            // -dmin by flipping the sign bit, no negate instruction
            half const dminn = __ushort_as_half((unsigned short)((meta0 >> 16) ^ 0x8000));

            unsigned const j_lo = (t_in_sb / 2) * 2;   // group of the low nibbles
            unsigned const j_hi = j_lo + 1;            // group of the high nibbles
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            get_scale_min_k4_w(j_lo, meta1, meta2, meta3, sc_lo, m_lo);
            get_scale_min_k4_w(j_hi, meta1, meta2, meta3, sc_hi, m_hi);

            // scale and zero in fp16. d8 = 8*d only changes the exponent, so it
            // is exact. zero is a fused multiply-add from (d8, sc): deriving it
            // from the already-rounded scale would multiply that rounding error
            // by 8 and break the (q - 8) anchor cancellation.
            half const d8 = __hmul(d, __ushort_as_half((unsigned short)0x4800));  // 8.0h

            half const sc_lo_h = __ushort2half_rn(sc_lo);   // sc, m <= 63: exact in fp16
            half const m_lo_h  = __ushort2half_rn(m_lo);
            half const sc_hi_h = __ushort2half_rn(sc_hi);
            half const m_hi_h  = __ushort2half_rn(m_hi);

            half2 const scale_lo2 = __half2half2(__hmul(d, sc_lo_h));
            half2 const scale_hi2 = __half2half2(__hmul(d, sc_hi_h));
            half2 const zero_lo2  = __half2half2(__hfma(d8, sc_lo_h, __hmul(dminn, m_lo_h)));
            half2 const zero_hi2  = __half2half2(__hfma(d8, sc_hi_h, __hmul(dminn, m_hi_h)));

            uint4 const qs = qs4[ii];   // 16 bytes = 4 quant words, already loaded

            // Four independent accumulator chains, one per quant word. Written
            // as straight-line code: a loop body containing asm may silently
            // fail to unroll, which would turn the array indices dynamic and
            // push everything into local memory.
            half2 dq0 = __float2half2_rn(0.f);
            half2 dq1 = __float2half2_rn(0.f);
            half2 dq2 = __float2half2_rn(0.f);
            half2 dq3 = __float2half2_rn(0.f);
            q4k_dot_word(qs.x, act_lo2[0], act_lo2[1], act_hi2[0], act_hi2[1],
                         scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
            q4k_dot_word(qs.y, act_lo2[2], act_lo2[3], act_hi2[2], act_hi2[3],
                         scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
            q4k_dot_word(qs.z, act_lo2[4], act_lo2[5], act_hi2[4], act_hi2[5],
                         scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);
            q4k_dot_word(qs.w, act_lo2[6], act_lo2[7], act_hi2[6], act_hi2[7],
                         scale_lo2, zero_lo2, scale_hi2, zero_hi2, dq0, dq1, dq2, dq3);

            // Fold the four chains into fp32 once per super-block.
            half2 const s = __hadd2(__hadd2(dq0, dq1), __hadd2(dq2, dq3));
            acc[ii] += __half2float(__low2half(s)) + __half2float(__high2half(s));
        }
    }

    // ---- epilogue: fp32 reduction ----
    // Every lane of the warp holds a partial sum for the same columns, so
    // reduce across all 32 lanes first.
#pragma unroll
    for (int col = 0; col < CTA_N; ++col)
    {
        float v = acc[col];
        v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
        v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
        v += __shfl_xor_sync(0xFFFFFFFF, v, 4);
        v += __shfl_xor_sync(0xFFFFFFFF, v, 8);
        v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
        acc[col] = v;   // lane 0 now holds this warp's column sums
    }

    if constexpr (WARPS_K == 1)
    {
        // One warp owns the column group outright: store directly, no shared
        // memory and no barrier.
        if (lane_id == 0)
        {
#pragma unroll
            for (int col = 0; col < CTA_N; ++col)
                out[row * n + col_start + col] = __float2half(acc[col]);
        }
    }
    else
    {
        // Combine the WARPS_K k-slices of each column group.
        __shared__ float red[WARPS_N][WARPS_K][CTA_N];
        if (lane_id == 0)
        {
#pragma unroll
            for (int col = 0; col < CTA_N; ++col)
                red[n_group][k_warp][col] = acc[col];
        }
        __syncthreads();

        if (k_warp == 0)
        {
            for (unsigned col = lane_id; col < CTA_N; col += 32)
            {
                float sum = 0.f;
#pragma unroll
                for (int kw = 0; kw < WARPS_K; ++kw)
                    sum += red[n_group][kw][col];
                out[row * n + col_start + col] = __float2half(sum);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// launcher
//   Requires: k % ELEMS_PER_WARP == 0,  n % (CTA_N * WARPS_N) == 0,
//             WARPS_N * WARPS_K * 32 <= 1024
//   Shared memory: k * sizeof(half) bytes for the staged activation row.
// ----------------------------------------------------------------------------
template <int CTA_N, int WARPS_N = 8, int WARPS_K = 1>
void launch_q4k_gemv(
    const half* act, const block_q4_K* w, half* out,
    int m, int n, int k, cudaStream_t stream = 0)
{
    constexpr int WARPS        = WARPS_N * WARPS_K;
    constexpr int THREADS      = WARPS * 32;
    constexpr int cols_per_cta = CTA_N * WARPS_N;
    static_assert(THREADS <= 1024, "CTA exceeds 1024 threads");

    assert(k % ELEMS_PER_WARP == 0);
    assert(n % cols_per_cta == 0);

    dim3 grid(m, n / cols_per_cta);
    dim3 block(THREADS);
    size_t const smem_bytes = (size_t)k * sizeof(half);
    q4k_gemv_kernel<CTA_N, WARPS_N, WARPS_K><<<grid, block, smem_bytes, stream>>>(
        act, w, out, (unsigned)m, (unsigned)n, (unsigned)k);
}

} // namespace q4k_gemv
