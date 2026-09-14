#pragma once
#include "unit16.hpp"
#include "packed_exchange.hpp"

namespace quactlize::dequant {

// Keep N-fast uint4 global loads and K-fast uint4 BF16 stores. Transpose
// packed codes BEFORE the FP32 affine/BF16 conversion. No expanded output
// crosses shared memory and no scale broadcast shuffle is required.
template<KType T, int TileK, bool KMajor>
__global__ __launch_bounds__(128) void full_packed_exchange(
        uint16_t const* __restrict__ low, uint16_t const* __restrict__ high,
        uint8_t const* __restrict__ units, uint16_t* __restrict__ output, int n, int k) {
    using R = Reader<T>;
    using Codes = PackedCodes<T>;
    using Layout = PackedExchangeLayout<TileK>;
    __shared__ __align__(16) uint32_t tile[Layout::kCells];
    __shared__ float2 affine[32 * Layout::kGroups];
    int const e = blockIdx.z;
    int const n0 = (KMajor ? blockIdx.y : blockIdx.x) * 32;
    int const k0 = (KMajor ? blockIdx.x : blockIdx.y) * TileK;
    int const tid = threadIdx.x & 127;
    int const lane = tid & 31, warp = tid >> 5;
    int const col0 = (lane % 4) * 8, residue = lane / 4;
    int64_t const nk = int64_t(n) * k;
    auto l = low + int64_t(e) * nk / 4;
    uint16_t const* h = nullptr;
    if constexpr (R::hi_bits) h = high + int64_t(e) * nk / 16;
    auto u = units + (int64_t(e) * (k / 256) + k0 / 256) * n * 16;
    constexpr int Parts = TileK / 128;
    uint4 lv[Parts], hv[Parts]{};
    #pragma unroll
    for (int part = 0; part < Parts; ++part) {
        int const kb = k0 + part * 128 + warp * 32;
        lv[part] = *reinterpret_cast<uint4 const*>(l + R::LowMap::word_index(n0 + col0, kb + residue, n));
        if constexpr (R::hi_bits)
            hv[part] = *reinterpret_cast<uint4 const*>(h + R::HighMap::word_index(n0 + col0, kb + residue, n));
    }
    #pragma unroll
    for (int v = 0; v < 8; ++v) {
        int const col = col0 + v;
        if (residue == 0) {
            auto meta = Unit16<T>::load(u + (n0 + col) * 16);
            #pragma unroll
            for (int part = 0; part < Parts; ++part) {
                int const g = part * 4 + warp;
                auto const a = meta.affine(((k0 >> 5) + g) & 7);
                affine[Layout::affine_offset(col, g)] = {a.scale, a.minimum};
            }
        }
        #pragma unroll
        for (int part = 0; part < Parts; ++part) {
            uint32_t const lw = v/2 == 0 ? lv[part].x : v/2 == 1 ? lv[part].y : v/2 == 2 ? lv[part].z : lv[part].w;
            uint32_t const hw = v/2 == 0 ? hv[part].x : v/2 == 1 ? hv[part].y : v/2 == 2 ? hv[part].z : hv[part].w;
            int const g = part * 4 + warp;
            tile[Layout::offset(col, g * 8 + residue)] = Codes::combine(
                uint16_t(lw >> (16 * (v % 2))), uint16_t(hw >> (16 * (v % 2))), n0 + col, k0 + g * 32);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int iteration = 0; iteration < TileK / 32; ++iteration) {
        int const i = tid + iteration * 128;
        int const row = i / (TileK / 8), kk = (i % (TileK / 8)) * 8;
        int const g = kk / 32, slot = (kk / 8) % 4;
        uint4 const a = *reinterpret_cast<uint4 const*>(tile + Layout::offset(row, g * 8));
        uint4 const b = *reinterpret_cast<uint4 const*>(tile + Layout::offset(row, g * 8 + 4));
        auto const f = affine[Layout::affine_offset(row, g)];
        typename R::Affine const s{f.x, f.y};
        uint4 const result{
            uint32_t(R::weight(Codes::code(a.x, slot), s)) | uint32_t(R::weight(Codes::code(a.y, slot), s)) << 16,
            uint32_t(R::weight(Codes::code(a.z, slot), s)) | uint32_t(R::weight(Codes::code(a.w, slot), s)) << 16,
            uint32_t(R::weight(Codes::code(b.x, slot), s)) | uint32_t(R::weight(Codes::code(b.y, slot), s)) << 16,
            uint32_t(R::weight(Codes::code(b.z, slot), s)) | uint32_t(R::weight(Codes::code(b.w, slot), s)) << 16};
        *reinterpret_cast<uint4*>(output + (int64_t(e) * n + n0 + row) * k + k0 + kk) = result;
    }
}

} // namespace quactlize::dequant
