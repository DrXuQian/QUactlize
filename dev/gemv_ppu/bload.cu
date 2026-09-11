// Compile-only experiment: canonical Q4 K-pack -> AIU/swzl/transpose ->
// registers -> SIMT FP32. No Tensor Core MMA, no offline-format change.
#include <hggc_runtime.h>
#include <hggc_fp16.h>
#include "cutlass/half.h"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/arch/copy_ppu.hpp"
#include "bload_contract.hpp"
#include "q4_aligned.cuh"

namespace q4_bload {
using namespace quactlize::dev::q4_native;

template<int BK> struct Transport {
    static constexpr int H = BK / 4;
    using Write = cute::PPU0010_AIU_LOAD<cute::C<H * 16 * 16>, cutlass::half_t, true, true>;
    using Read = cute::PPU0010_TSM_LD_SWZL<cutlass::half_t, H, 16, true, true, 1>;
    __device__ static void load(uint16_t* shared, uint16_t const* low,
                                int n, int k, int col, int kg) {
        cute::AiuDesc desc{};
        desc.gmem_ptr = reinterpret_cast<uint8_t const*>(low);
        desc.dim_h = k / 4; desc.dim_w = n;
        desc.cube_h = H; desc.cube_w = 16;
        if (threadIdx.x == 0) Write::copy(shared, low, desc, kg, col);
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads(); // publish the one-thread DMA to ALL consumer warps
    }
    __device__ static void read(uint32_t (&regs)[4], uint16_t* shared, int micro) {
        Read::copy(regs, shared, micro * 16, 0);
    }
};

// Same fragment/compute topology as the AIU arm, with direct scalar b16 loads.
// This control separates a transport change from the old N4 topology change.
__device__ __forceinline__ void global_fragment(uint32_t (&regs)[4],
        uint16_t const* low, int n, unsigned col, unsigned kg) {
    unsigned lane = threadIdx.x & 31;
    #pragma unroll
    for (unsigned v = 0; v < 4; ++v) {
        unsigned const nn = col + word_n(lane, v);
        uint32_t const lo = low[size_t(kg + word_kg(lane, v, 0)) * n + nn];
        uint32_t const hi = low[size_t(kg + word_kg(lane, v, 1)) * n + nn];
        regs[v] = lo | (hi << 16);
    }
}

template<int Slot>
__device__ __forceinline__ void dot(uint32_t word, __half2 a, ScaleZero sz,
                                  float& x, float& y) {
    __half2 const w = __hfma2(codes<Slot>(word), __half2half2(sz.scale), __half2half2(sz.zero));
    float2 const wf = __half22float2(w), af = __half22float2(a);
    x = fmaf(wf.x, af.x, x);
    y = fmaf(wf.y, af.y, y);
}

template<bool Aiu, int BK, int WK>
__global__ void gemv(__half const* a, uint16_t const* low, uint8_t const* units,
                     float* output, unsigned n, unsigned k) {
    // B staging is part of the AIU transport cost; the compiler may remove
    // its unused reservation in the direct control. Compare actual resources.
    struct alignas(32) Storage { uint16_t b[BK / 4 * 16]; float partial[WK * 16]; };
    __shared__ Storage shared;
    unsigned const lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    unsigned const col = blockIdx.x * 16;
    float x[2][4]{}, y[2][4]{};
    for (unsigned kb = 0; kb < k; kb += BK) {
        if constexpr (Aiu) Transport<BK>::load(shared.b, low, n, k, col, kb / 4);
        #pragma unroll
        for (unsigned micro = warp; micro < BK / 64; micro += WK) {
            unsigned const k0 = kb + micro * 64;
            uint32_t regs[4];
            if constexpr (Aiu) Transport<BK>::read(regs, shared.b, micro);
            else global_fragment(regs, low, n, col, k0 / 4);
            uint4 meta[2];
            #pragma unroll
            for (unsigned ni = 0; ni < 2; ++ni)
                meta[ni] = aligned_unit(units + (size_t(k0 / 256) * n + col + lane / 4 + ni * 8) * 16);
            #pragma unroll
            for (unsigned group = 0; group < 2; ++group) {
                __half2 av[4];
                #pragma unroll
                for (unsigned slot = 0; slot < 4; ++slot)
                    av[slot] = *reinterpret_cast<__half2 const*>(a + k0 + group * 32 + slot * 8 + 2 * (lane % 4));
                #pragma unroll
                for (unsigned ni = 0; ni < 2; ++ni) {
                    auto sz = aligned_scale_zero(meta[ni], (k0 / 32 + group) & 7);
                    uint32_t const word = regs[ni * 2 + group];
                    dot<0>(word, av[0], sz, x[ni][0], y[ni][0]);
                    dot<1>(word, av[1], sz, x[ni][1], y[ni][1]);
                    dot<2>(word, av[2], sz, x[ni][2], y[ni][2]);
                    dot<3>(word, av[3], sz, x[ni][3], y[ni][3]);
                }
            }
        }
        if constexpr (Aiu) __syncthreads(); // all reads finish before overwrite
    }
    #pragma unroll
    for (unsigned ni = 0; ni < 2; ++ni) {
        float v = ((x[ni][0] + x[ni][1]) + (x[ni][2] + x[ni][3])) +
                  ((y[ni][0] + y[ni][1]) + (y[ni][2] + y[ni][3]));
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        if ((lane & 3) == 0) shared.partial[warp * 16 + lane / 4 + ni * 8] = v;
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0;
        #pragma unroll
        for (unsigned w = 0; w < WK; ++w) sum += shared.partial[w * 16 + threadIdx.x];
        output[col + threadIdx.x] = sum;
    }
}

// Raw b16 coordinate gate: multiple N offsets and repeated K stages. The
// output records physical (microtile,lane,reg), independently checked on host.
template<int BK, int WK>
__global__ void transport_gate(uint16_t const* input, uint32_t* out) {
    __shared__ __align__(32) uint16_t shared[BK / 4 * 16];
    constexpr unsigned N = 64, K = 2048;
    unsigned const tile = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    for (unsigned kb = 0; kb < K; kb += BK) {
        Transport<BK>::load(shared, input, N, K, tile * 16, kb / 4);
        #pragma unroll
        for (unsigned micro = warp; micro < BK / 64; micro += WK) {
            uint32_t regs[4];
            Transport<BK>::read(regs, shared, micro);
            #pragma unroll
            for (unsigned v = 0; v < 4; ++v)
                out[((tile * (K / 64) + kb / 64 + micro) * 32 + lane) * 4 + v] = regs[v];
        }
        __syncthreads();
    }
}
} // namespace q4_bload

extern "C" int q4_bload_run(int aiu, int bk, int wk, int n, int k,
        void const* a, void const* low, void const* units, void* out, void* stream) {
    if (!q4_bload::shape(n, k) || !q4_bload::recipe(bk, wk) || (aiu != 0 && aiu != 1)) return -1;
    if (!a || !low || !units || !out || (uintptr_t(a) & 15) || (uintptr_t(low) & 31) ||
        (uintptr_t(units) & 15) || (uintptr_t(out) & 3)) return -2;
    auto s = static_cast<hggcStream_t>(stream);
    #define Q4_BLOAD_ARM(BK,WK) if (bk == BK && wk == WK) { \
        if (aiu) q4_bload::gemv<true,BK,WK><<<n/16,WK*32,0,s>>>( \
            static_cast<__half const*>(a),static_cast<uint16_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out),n,k); \
        else q4_bload::gemv<false,BK,WK><<<n/16,WK*32,0,s>>>( \
            static_cast<__half const*>(a),static_cast<uint16_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out),n,k); \
        return int(hggcGetLastError()); }
    Q4_BLOAD_ARM(256,4)
    Q4_BLOAD_ARM(512,8)
    Q4_BLOAD_ARM(1024,8)
    #undef Q4_BLOAD_ARM
    return -1;
}

extern "C" int q4_bload_transport(int bk, int wk, void const* input, void* output, void* stream) {
    if (!input || !output || (uintptr_t(input) & 31) || (uintptr_t(output) & 3)) return -2;
    #define Q4_GATE_ARM(BK,WK) if (bk == BK && wk == WK) { \
        q4_bload::transport_gate<BK,WK><<<4,WK*32,0,static_cast<hggcStream_t>(stream)>>>( \
            static_cast<uint16_t const*>(input),static_cast<uint32_t*>(output)); return int(hggcGetLastError()); }
    Q4_GATE_ARM(256,4)
    Q4_GATE_ARM(512,8)
    Q4_GATE_ARM(1024,8)
    #undef Q4_GATE_ARM
    return -1;
}
