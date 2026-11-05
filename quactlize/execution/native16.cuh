#pragma once
#include "native16_geometry.hpp"
#include "../compat/contract.hpp"
#include "cute/arch/mma_ppu0010.hpp"
#include <hggc_runtime.h>
#include <type_traits>

namespace quactlize::execution::native16 {
using compat::Weight;
__device__ __forceinline__ int expert(qpm_problem_v1 const &p, qpm_buffers_v1 const &b, int row) {
    return p.mode == QPM_DENSE ? 0 : b.ids[int64_t(row / p.topk) * p.ids_stride + row % p.topk];
}
__device__ __forceinline__ int64_t input_offset(qpm_problem_v1 const &p, int row) {
    return int64_t(row / p.topk) * p.input_token_stride +
           (p.channels == 1 ? 0 : int64_t(row % p.topk) * p.input_row_stride);
}

// N32/CTA, four adjacent columns per lane. Each group of eight lanes loads
// contiguous 64B code / 128B affine packets; A is shared with width-8 exchanges.
// K/group is a multiple of four, so each warp's K tail is uniformly inactive.
template <class T> __global__ void packed_simt(qpm_problem_v1 p, qpm_buffers_v1 b, Weight w) {
    unsigned tid = threadIdx.x, lane = tid % 32, worker = tid / 8;
    int tiles = (uint64_t(p.n) + 31) / 32;
    int row = blockIdx.x / tiles, first = (blockIdx.x % tiles) * 32;
    int e = expert(p, b, row);
    if (e < 0 || e >= p.experts) {
        if (tid == 0 && e != -1)
            atomicOr(b.device_error, QPM_BAD_ID);
        if (tid < 32 && first + int(tid) < p.n)
            static_cast<T *>(b.output)[int64_t(row) * p.output_stride + first + tid] = T(0.f);
        return;
    }
    int col = first + (tid % 8) * 4;
    auto low = reinterpret_cast<uint16_t const *>(w.low + uint64_t(e) * w.low_stride);
    auto high = reinterpret_cast<uint16_t const *>(w.high + uint64_t(e) * w.high_stride);
    auto units = reinterpret_cast<uint32_t const *>(w.units + uint64_t(e) * w.units_stride);
    auto a = static_cast<T const *>(b.input) + input_offset(p, row);
    float sum[4]{};
    for (unsigned g = worker; g < unsigned(p.k / 32); g += 16) {
        uint4 dm = col < p.n ? *reinterpret_cast<uint4 const *>(units + uint64_t(g) * p.n + col)
                             : make_uint4(0, 0, 0, 0);
        uint32_t metadata[4] = {dm.x, dm.y, dm.z, dm.w};
        float activation[4];
#pragma unroll
        for (int v = 0; v < 4; ++v)
            activation[v] = float(a[g * 32 + (lane % 8) * 4 + v]);
#pragma unroll
        for (int residue = 0; residue < 8; ++residue) {
            uint2 lo = col < p.n
                           ? *reinterpret_cast<uint2 const *>(low + (uint64_t(g) * 8 + residue) * p.n + col)
                           : make_uint2(0, 0);
            uint2 hi =
                col < p.n
                    ? *reinterpret_cast<uint2 const *>(high + (uint64_t(g / 4) * 8 + residue) * p.n + col)
                    : make_uint2(0, 0);
            uint16_t lows[4] = {uint16_t(lo.x), uint16_t(lo.x >> 16), uint16_t(lo.y), uint16_t(lo.y >> 16)};
            uint16_t highs[4] = {uint16_t(hi.x), uint16_t(hi.x >> 16), uint16_t(hi.y), uint16_t(hi.y >> 16)};
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                int offset = slot * 8 + residue;
                float av = __shfl_sync(0xffffffffu, activation[offset % 4], (lane & ~7u) + offset / 4);
#pragma unroll
                for (int v = 0; v < 4; ++v)
                    sum[v] = fmaf(
                        av, float(q5_weight<T>(lows[v], highs[v], slot, (g % 4) * 4 + slot, metadata[v])),
                        sum[v]);
            }
        }
    }
#pragma unroll
    for (int delta = 8; delta <= 16; delta *= 2) {
#pragma unroll
        for (int v = 0; v < 4; ++v)
            sum[v] += __shfl_xor_sync(0xffffffffu, sum[v], delta);
    }
    __shared__ float partial[4][32];
    if (lane < 8) {
#pragma unroll
        for (int v = 0; v < 4; ++v)
            partial[tid / 32][lane * 4 + v] = sum[v];
    }
    __syncthreads();
    if (tid < 32 && first + int(tid) < p.n) {
        float value = ((partial[0][tid] + partial[1][tid]) + partial[2][tid]) + partial[3][tid];
        static_cast<T *>(b.output)[int64_t(row) * p.output_stride + first + tid] = T(value);
    }
}

__global__ void prefix_tiles(int *counts, int *cursors, int *tiles, int experts) {
    if (threadIdx.x)
        return;
    int rows = 0, tasks = 0;
    for (int e = 0; e < experts; ++e) {
        int n = counts[e];
        counts[e] = rows;
        cursors[e] = 0;
        tiles[e] = tasks;
        rows += n;
        tasks += (n + 15) / 16;
    }
    counts[experts] = rows;
    tiles[experts] = tasks;
}
template <class T>
__global__ void route_slots(qpm_problem_v1 p, qpm_buffers_v1 b, int rows, int const *counts, int *cursors,
                            int *maps) {
    int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows)
        return;
    int e = expert(p, b, row);
    if (e >= 0 && e < p.experts) {
        if (lane == 0)
            maps[counts[e] + atomicAdd(cursors + e, 1)] = row;
    } else {
        if (lane == 0 && e != -1)
            atomicOr(b.device_error, QPM_BAD_ID);
        for (int n = lane; n < p.n; n += 32)
            static_cast<T *>(b.output)[int64_t(row) * p.output_stride + n] = T(0.f);
    }
}

template <class T>
__device__ __forceinline__ void mma(uint32_t const (&a)[4], uint32_t const (&b)[4], float (&c)[8]) {
    using Atom =
        std::conditional_t<std::is_same_v<T, cutlass::half_t>, cute::PPU0010_16x16x16_F32F16F16F32_TN,
                           cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
    Atom::fma(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
              c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}

// M16/N64/K128 staging. Global loads own complete physical packed words,
// including one high-plane/affine copy per K128 (not per K32/N task).
// B is decoded straight into native MMA registers: no global W16 expansion.
// Shared is plain scalar-produced storage, not an AIU.swzl layout.
template <class T>
__global__ void packed_tc(qpm_problem_v1 p, qpm_buffers_v1 b, Weight w, int const *counts, int const *tiles,
                          int const *maps) {
    int nt = (uint64_t(p.n) + 63) / 64, task = blockIdx.x / nt, first = (blockIdx.x % nt) * 64;
    int e = 0, row0 = task * 16, row_end = p.m;
    if (p.mode == QPM_INDEXED) {
        if (task >= tiles[p.experts])
            return;
        int lo = 0, hi = p.experts;
        while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            if (tiles[mid] <= task)
                lo = mid;
            else
                hi = mid;
        }
        e = lo;
        row0 = counts[e] + (task - tiles[e]) * 16;
        row_end = counts[e + 1];
    }
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    __shared__ int slots[16];
    __shared__ __align__(16) uint16_t a[16 * 128], low[32 * 64], high[8 * 64];
    __shared__ __align__(16) uint32_t dm[4 * 64];
    if (tid < 16)
        slots[tid] = row0 + tid < row_end ? (p.mode == QPM_INDEXED ? maps[row0 + tid] : row0 + tid) : -1;
    __syncthreads();
    auto gl = reinterpret_cast<uint16_t const *>(w.low + uint64_t(e) * w.low_stride);
    auto gh = reinterpret_cast<uint16_t const *>(w.high + uint64_t(e) * w.high_stride);
    auto gm = reinterpret_cast<uint32_t const *>(w.units + uint64_t(e) * w.units_stride);
    float accum[8]{};
    for (int start = 0; start < p.k; start += 128) {
        for (int i = tid * 8; i < 32 * 64; i += 128 * 8)
            *reinterpret_cast<uint4 *>(low + i) =
                first + i % 64 < p.n ? *reinterpret_cast<uint4 const *>(
                                           gl + int64_t(start / 4 + i / 64) * p.n + first + i % 64)
                                     : make_uint4(0, 0, 0, 0);
        if (tid * 8 < 8 * 64) {
            int i = tid * 8;
            *reinterpret_cast<uint4 *>(high + i) =
                first + i % 64 < p.n ? *reinterpret_cast<uint4 const *>(
                                           gh + int64_t(start / 16 + i / 64) * p.n + first + i % 64)
                                     : make_uint4(0, 0, 0, 0);
        }
        if (tid * 4 < 4 * 64) {
            int i = tid * 4;
            *reinterpret_cast<uint4 *>(dm + i) =
                first + i % 64 < p.n ? *reinterpret_cast<uint4 const *>(
                                           gm + int64_t(start / 32 + i / 64) * p.n + first + i % 64)
                                     : make_uint4(0, 0, 0, 0);
        }
        for (int i = tid; i < 16 * 128; i += 128) {
            int slot = slots[i / 128];
            a[i] = slot < 0 ? 0
                            : static_cast<uint16_t const *>(b.input)[input_offset(p, slot) + start + i % 128];
        }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < 4; ++group) {
            uint32_t metadata[2] = {dm[group * 64 + warp * 16 + lane / 4],
                                    dm[group * 64 + warp * 16 + lane / 4 + 8]};
#pragma unroll
            for (int atom = 0; atom < 2; ++atom) {
                uint32_t ra[4], rb[4];
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    int km = group * 32 + atom * 16 + Fragment::k(lane, 2 * v);
                    int m = Fragment::row(lane, 2 * v), n = warp * 16 + m;
                    ra[v] = uint32_t(a[m * 128 + km]) | (uint32_t(a[m * 128 + km + 1]) << 16);
                    uint16_t x = q5_weight<T>(low[(km / 32 * 8 + km % 8) * 64 + n], high[(km % 8) * 64 + n],
                                              (km % 32) / 8, km / 8, metadata[v / 2])
                                     .raw();
                    uint16_t y =
                        q5_weight<T>(low[(km / 32 * 8 + (km + 1) % 8) * 64 + n],
                                     high[((km + 1) % 8) * 64 + n], (km % 32) / 8, km / 8, metadata[v / 2])
                            .raw();
                    rb[v] = uint32_t(x) | (uint32_t(y) << 16);
                }
                mma<T>(ra, rb, accum);
            }
        }
        // All MMA operands are in registers; retire every shared reader before
        // any warp overwrites the stage. No asynchronous copy is in flight.
        __syncthreads();
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        int slot = slots[Fragment::row(lane, v)];
        int n = first + warp * 16 + Fragment::column(lane, v);
        if (slot >= 0 && n < p.n)
            static_cast<T *>(b.output)[int64_t(slot) * p.output_stride + n] = T(accum[v]);
    }
}
} // namespace quactlize::execution::native16
