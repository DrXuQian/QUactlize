#pragma once
#include "unit16.hpp"
#include "transpose_layout.hpp"

namespace quactlize::dequant {

// Additive Q4/Q5 experiment. The global B mapping and FP32 arithmetic are
// identical to full_wide<T,true,true>. Only the shared exchange, metadata
// lifetime and number of K32 groups per CTA differ. Nothing reads A.
template<KType T, int TileK, int StageK, bool CacheMetadata>
__global__ __launch_bounds__(128) void full_shared(
        uint16_t const* __restrict__ low, uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units, uint16_t* __restrict__ output, int n, int k) {
    static_assert(T == KType::Q4_K || T == KType::Q5_K);
    static_assert(TileK == 128 || TileK == 256);
    static_assert(TileK % StageK == 0);
    using R = Reader<T>;
    using Layout = FullTransposeLayout<StageK>;
    __shared__ __align__(16) uint32_t tile[Layout::kCells];
    // This allocation is dead and removed for CacheMetadata=false; inspect
    // the native shared size rather than assuming the compiler did so.
    __shared__ uint4 metadata[32];
    int const e = blockIdx.z, n0 = blockIdx.x * 32, k0 = blockIdx.y * TileK;
    int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int const col0 = (lane % 4) * 8, residue = lane / 4;
    int64_t const nk = int64_t(n) * k;
    auto const l = low + int64_t(e) * nk / 4;
    uint16_t const* h = nullptr;
    if constexpr (R::hi_bits) h = high + int64_t(e) * nk / (16 / R::hi_bits);
    auto const u = units + (int64_t(e) * (k / 256) + k0 / 256) * n * 16;

    if constexpr (CacheMetadata) {
        // One complete unit per N, one contiguous 512-byte warp request.
        if (threadIdx.x < 32)
            metadata[threadIdx.x] = *reinterpret_cast<uint4 const*>(u + (n0 + threadIdx.x) * 16);
        __syncthreads();
    }
    #pragma unroll
    for (int stage = 0; stage < TileK / StageK; ++stage) {
        constexpr int Parts = StageK / 128;
        uint4 lv[Parts], hv[Parts]{};
        #pragma unroll
        for (int part = 0; part < Parts; ++part) {
            int const kb = k0 + stage * StageK + part * 128 + warp * 32;
            lv[part] = *reinterpret_cast<uint4 const*>(l + R::LowMap::word_index(n0 + col0, kb + residue, n));
            if constexpr (R::hi_bits)
                hv[part] = *reinterpret_cast<uint4 const*>(h + R::HighMap::word_index(n0 + col0, kb + residue, n));
        }
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            int const col = col0 + v;
            Unit16<T> meta{};
            if (residue == 0) {
                if constexpr (CacheMetadata) {
                    uint4 const packed = metadata[col];
                    meta = {packed.x, packed.y, packed.z, packed.w};
                } else {
                    meta = Unit16<T>::load(u + (n0 + col) * 16);
                }
            }
            #pragma unroll
            for (int part = 0; part < Parts; ++part) {
                int const kb = k0 + stage * StageK + part * 128 + warp * 32;
                typename R::Affine affine{};
                if (residue == 0) affine = meta.affine((kb / 32) % 8);
                affine.scale = __shfl_sync(0xffffffff, affine.scale, lane % 4);
                affine.minimum = __shfl_sync(0xffffffff, affine.minimum, lane % 4);
                uint32_t const lw = v / 2 == 0 ? lv[part].x : v / 2 == 1 ? lv[part].y : v / 2 == 2 ? lv[part].z : lv[part].w;
                uint32_t const hw = v / 2 == 0 ? hv[part].x : v / 2 == 1 ? hv[part].y : v / 2 == 2 ? hv[part].z : hv[part].w;
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    int const kk = kb + residue + 8 * s;
                    int const local_k = part * 128 + warp * 32 + residue + 8 * s;
                    tile[Layout::offset(col, local_k)] = R::weight(
                        R::raw_from_words(uint16_t(lw >> (16 * (v % 2))),
                                          uint16_t(hw >> (16 * (v % 2))), n0 + col, kk), affine);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int i = threadIdx.x; i < Layout::kCells / 8; i += 128) {
            int const row = i / (StageK / 8), kk = (i % (StageK / 8)) * 8;
            uint4 const a = *reinterpret_cast<uint4 const*>(tile + Layout::offset(row, kk));
            uint4 const b = *reinterpret_cast<uint4 const*>(tile + Layout::offset(row, kk + 4));
            uint4 const packed{a.x | a.y << 16, a.z | a.w << 16, b.x | b.y << 16, b.z | b.w << 16};
            *reinterpret_cast<uint4*>(output + (int64_t(e) * n + n0 + row) * k + k0 + stage * StageK + kk) = packed;
        }
        // Protect the shared exchange before its next K128 stage reuses it.
        // The terminal stage needs no additional CTA barrier.
        if constexpr (TileK > StageK) {
            if (stage + 1 < TileK / StageK) __syncthreads();
        }
    }
}

} // namespace quactlize::dequant
