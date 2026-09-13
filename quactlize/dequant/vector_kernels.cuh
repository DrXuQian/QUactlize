#pragma once
#include "unit16.hpp"

namespace quactlize::dequant {

template<KType T, int Threads>
__global__ __launch_bounds__(Threads) void sf_unit16(
        uint8_t const* __restrict__ units, cutlass::half_t* __restrict__ scale,
        cutlass::half_t* __restrict__ zero, int n, int k) {
    int const col = blockIdx.x * Threads + threadIdx.x;
    if (col >= n) return;
    int const sb = blockIdx.y, e = blockIdx.z;
    auto const v = Unit16<T>::load(units + ((int64_t(e) * (k / 256) + sb) * n + col) * 16);
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        auto const sz = v.scale(g);
        int64_t const out = ((int64_t(e) * (k / 256) + sb) * 8 + g) * n + col;
        scale[out] = sz.scale;
        zero[out] = sz.zero;
    }
}

// The K128 variants keep the output rows at a complete 256-byte span per
// tile. Compare scalar B/pair store, scalar B/vector store, and vector B/
// vector store independently. A uint4 transports bits; it never performs
// float4 arithmetic or changes BF16 rounding.
template<KType T, bool VectorB, bool VectorStore>
__global__ __launch_bounds__(128) void full_wide(
        uint16_t const* __restrict__ low, uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units, uint16_t* __restrict__ output, int n, int k) {
    using R = Reader<T>;
    __shared__ uint32_t tile[32][129];
    int const e = blockIdx.z, n0 = blockIdx.x * 32, k0 = blockIdx.y * 128;
    int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int64_t const nk = int64_t(n) * k;
    auto const l = low + int64_t(e) * nk / (16 / R::lo_bits);
    uint16_t const* h = nullptr;
    if constexpr (R::hi_bits) h = high + int64_t(e) * nk / (16 / R::hi_bits);
    auto const u = units + int64_t(e) * R::metadata_bytes(nk);
    if constexpr (!VectorB) {
        #pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
            int const kb = k0 + 32 * chunk, col = n0 + lane;
            auto const first = R::affine(u, col, kb / R::group, n);
            auto second = first;
            if constexpr (R::group == 16) second = R::affine(u, col, kb / R::group + 1, n);
            #pragma unroll
            for (int r = warp; r < 8; r += 4) {
                uint16_t const word = l[R::LowMap::word_index(col, kb + r, n)];
                uint16_t hw = 0;
                if constexpr (R::hi_bits) hw = h[R::HighMap::word_index(col, kb + r, n)];
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    int const kk = kb + r + 8 * s;
                    tile[lane][32 * chunk + r + 8 * s] = R::weight(
                        R::raw_from_words(word, hw, col, kk), s < 2 ? first : second);
                }
            }
        }
    } else {
        // Four consecutive N8 vectors per residue cover N32. A warp owns
        // K32; only the four residue-zero lanes load metadata, then broadcast
        // scale/min to the eight K residues using warp shuffles.
        int const col0 = n0 + (lane % 4) * 8, residue = lane / 4, kb = k0 + warp * 32;
        uint4 const lv = *reinterpret_cast<uint4 const*>(l + R::LowMap::word_index(col0, kb + residue, n));
        uint4 hv{};
        if constexpr (R::hi_bits) hv = *reinterpret_cast<uint4 const*>(h + R::HighMap::word_index(col0, kb + residue, n));
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            int const col = col0 + v;
            typename R::Affine first{}, second{};
            if (residue == 0) {
                if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
                    auto const meta = Unit16<T>::load(u + (int64_t(kb / 256) * n + col) * 16);
                    first = second = meta.affine((kb / 32) % 8);
                } else {
                    first = R::affine(u, col, kb / R::group, n);
                    second = R::affine(u, col, kb / R::group + (R::group == 16), n);
                }
            }
            first.scale = __shfl_sync(0xffffffff, first.scale, lane % 4);
            first.minimum = __shfl_sync(0xffffffff, first.minimum, lane % 4);
            second.scale = __shfl_sync(0xffffffff, second.scale, lane % 4);
            second.minimum = __shfl_sync(0xffffffff, second.minimum, lane % 4);
            uint32_t const lw = v / 2 == 0 ? lv.x : v / 2 == 1 ? lv.y : v / 2 == 2 ? lv.z : lv.w;
            uint32_t const hw = v / 2 == 0 ? hv.x : v / 2 == 1 ? hv.y : v / 2 == 2 ? hv.z : hv.w;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                int const kk = kb + residue + 8 * s;
                tile[col - n0][warp * 32 + residue + 8 * s] = R::weight(
                    R::raw_from_words(uint16_t(lw >> (16 * (v % 2))), uint16_t(hw >> (16 * (v % 2))), col, kk),
                    s < 2 ? first : second);
            }
        }
    }
    __syncthreads();
    if constexpr (VectorStore) {
        #pragma unroll
        for (int i = threadIdx.x; i < 512; i += 128) {
            int const row = i / 16, kk = (i % 16) * 8;
            uint4 const v{tile[row][kk] | tile[row][kk+1] << 16,
                          tile[row][kk+2] | tile[row][kk+3] << 16,
                          tile[row][kk+4] | tile[row][kk+5] << 16,
                          tile[row][kk+6] | tile[row][kk+7] << 16};
            *reinterpret_cast<uint4*>(output + (int64_t(e) * n + n0 + row) * k + k0 + kk) = v;
        }
    } else {
        #pragma unroll
        for (int i = threadIdx.x; i < 2048; i += 128) {
            int const row = i / 64, kk = (i % 64) * 2;
            reinterpret_cast<uint32_t*>(output)[((int64_t(e) * n + n0 + row) * k + k0 + kk) / 2]
                = tile[row][kk] | tile[row][kk+1] << 16;
        }
    }
}
} // namespace quactlize::dequant
