#include <hggc_runtime.h>
#include "api.h"
#include "reader.hpp"
#include "vector_kernels.cuh"
#include "shared_kernels.cuh"
#include "packed_kernels.cuh"
#include "../execution/validation.hpp"
#include "gguf_scale_prepass.hpp"

namespace quactlize::dequant {

template<KType T, int Columns>
__global__ void sf_columns(uint8_t const* __restrict__ units, cutlass::half_t* __restrict__ scale, cutlass::half_t* __restrict__ zero,
                           int n, int k, int experts) {
    using R = execution::Reader<T>;
    using U = typename R::U;
    constexpr int LanesPerColumn = 32 / Columns;
    int const lane = threadIdx.x % 32;
    int const e = blockIdx.z, sb = blockIdx.y;
    int const col = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * Columns + lane % Columns;
    if (col >= n) return;
    uint8_t const* u = units + int64_t(e) * R::metadata_bytes(int64_t(n) * k);
    #pragma unroll
    for (int g = lane / Columns; g < U::kGroups; g += LanesPerColumn) {
        auto const v = R::scale(u, col, sb * U::kGroups + g, n);
        int64_t const out = (int64_t(e) * (k / R::group) + sb * U::kGroups + g) * n + col;
        scale[out] = v.scale;
        zero[out] = v.zero;
    }
}

template<KType T>
__global__ void full_direct(uint16_t const* low, uint16_t const* high, uint8_t const* units,
                            uint16_t* output, int n, int k, int experts) {
    using R = Reader<T>;
    int64_t const i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const nk = int64_t(n) * k;
    if (i >= nk * experts) return;
    int const e = i / nk, col = (i % nk) / k, kk = i % k;
    auto const l = low + int64_t(e) * nk / (16 / R::lo_bits);
    uint16_t const* h = nullptr;
    if constexpr (R::hi_bits) h = high + int64_t(e) * nk / (16 / R::hi_bits);
    auto const u = units + int64_t(e) * R::metadata_bytes(nk);
    int const raw = R::code(l, h, col, kk, n) + (R::lo_bits == 4 ? 8 : 0);
    output[i] = R::weight(raw, R::affine(u, col, kk / R::group, n));
}

template<KType T, int Threads>
__global__ __launch_bounds__(Threads) void full_transpose(uint16_t const* __restrict__ low, uint16_t const* __restrict__ high,
                               uint8_t const* __restrict__ units, uint16_t* __restrict__ output, int n, int k, int experts) {
    using R = Reader<T>;
    // N-fast load: one warp requests 32 consecutive b16 words (64 bytes).
    // A thread owns one N and K residue, expands the four K+8 slots, then
    // shared transpose permits K-fast paired BF16 output stores. 32-bit
    // padded shared cells avoid half-word bank aliasing on the transpose.
    __shared__ uint32_t tile[32][33];
    int const e = blockIdx.z;
    int const n0 = blockIdx.x * 32, k0 = blockIdx.y * 32;
    int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int64_t const nk = int64_t(n) * k;
    auto const l = low + int64_t(e) * nk / (16 / R::lo_bits);
    uint16_t const* h = nullptr;
    if constexpr (R::hi_bits) h = high + int64_t(e) * nk / (16 / R::hi_bits);
    auto const u = units + int64_t(e) * R::metadata_bytes(nk);
    int const col = n0 + lane;
    auto const first = R::affine(u, col, k0 / R::group, n);
    auto second = first;
    if constexpr (R::group == 16) second = R::affine(u, col, k0 / R::group + 1, n);
    #pragma unroll
    for (int residue = warp; residue < 8; residue += Threads / 32) {
        uint16_t const word = l[R::LowMap::word_index(col, k0 + residue, n)];
        uint16_t high_word = 0;
        if constexpr (R::hi_bits) high_word = h[R::HighMap::word_index(col, k0 + residue, n)];
        #pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            int const kk = k0 + residue + 8 * slot;
            int const raw = R::raw_from_words(word, high_word, col, kk);
            tile[lane][residue + 8 * slot] = R::weight(raw, slot < 2 ? first : second);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = threadIdx.x; i < 512; i += Threads) {
        int const row = i / 16, kk = (i % 16) * 2;
        uint32_t const pair = tile[row][kk] | tile[row][kk + 1] << 16;
        reinterpret_cast<uint32_t*>(output)[(int64_t(e) * n + n0 + row) * (k / 2) + (k0 + kk) / 2] = pair;
    }
}

template<KType T>
int launch(qzd_call_v1 const& c) {
    auto stream = static_cast<hggcStream_t>(c.stream);
    auto low = static_cast<uint16_t const*>(c.low);
    auto high = static_cast<uint16_t const*>(c.high);
    auto units = static_cast<uint8_t const*>(c.units);
    if (c.operation == 0) {
        auto scale = static_cast<cutlass::half_t*>(c.output);
        auto zero = static_cast<cutlass::half_t*>(c.zero);
        if (c.config == 0) {
            using namespace gguf_scale;
            prepass::UnitPlaneDesc dst{scale,zero,int64_t(c.n)*(c.k/Traits<T>::kGroupSize),c.n,1};
            auto args = prepass::make_unit_prepass_kernel_args(units,dst,c.experts,c.n,c.k/256);
            int const grid = prepass::prepass_unit_grid_size<T>(c.experts,c.n,c.k/256,256);
            prepass::prepass_unit_kernel<T,packed_unit::kCanonicalPlacedZMul<T>><<<grid,256,0,stream>>>(args);
        } else if (c.config >= 4) {
            if (c.config > 5) return QKG_INVALID;
            if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
                int const threads = c.config == 4 ? 128 : 256;
                dim3 const grid(c.n / threads, c.k / 256, c.experts);
                if (threads == 128) sf_unit16<T,128><<<grid,128,0,stream>>>(units,scale,zero,c.n,c.k);
                else sf_unit16<T,256><<<grid,256,0,stream>>>(units,scale,zero,c.n,c.k);
            } else return QKG_FORMAT;
        } else {
            int const columns = c.config == 1 ? 16 : 32, threads = c.config == 3 ? 128 : 256;
            dim3 const grid((c.n + columns*(threads/32) - 1)/(columns*(threads/32)),c.k/256,c.experts);
            if (columns == 16) sf_columns<T,16><<<grid,threads,0,stream>>>(units,scale,zero,c.n,c.k,c.experts);
            else sf_columns<T,32><<<grid,threads,0,stream>>>(units,scale,zero,c.n,c.k,c.experts);
        }
    } else {
        auto output = static_cast<uint16_t*>(c.output);
        if (c.config == 0) {
            int const grid = (int64_t(c.experts)*c.n*c.k+255)/256;
            full_direct<T><<<grid,256,0,stream>>>(low,high,units,output,c.n,c.k,c.experts);
        } else if (c.config >= 10) {
            if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
                if (c.config == 10) full_packed_exchange<T,128,false><<<dim3(c.n/32,c.k/128,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k);
                else if (c.config == 11) full_packed_exchange<T,256,false><<<dim3(c.n/32,c.k/256,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k);
                else full_packed_exchange<T,256,true><<<dim3(c.k/256,c.n/32,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k);
            } else return QKG_FORMAT;
        } else if (c.config >= 6) {
            if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
                int const tile_k = c.config >= 8 ? 256 : 128;
                dim3 const grid(c.n/32,c.k/tile_k,c.experts);
                if (c.config == 6) full_shared<T,128,128,false><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
                else if (c.config == 7) full_shared<T,128,128,true><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
                else if (c.config == 8) full_shared<T,256,256,true><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
                else full_shared<T,256,128,true><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
            } else return QKG_FORMAT;
        } else if (c.config >= 3) {
            dim3 const grid(c.n/32,c.k/128,c.experts);
            if (c.config == 3) full_wide<T,false,false><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
            else if (c.config == 4) full_wide<T,false,true><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
            else full_wide<T,true,true><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k);
        } else {
            dim3 const grid(c.n/32,c.k/32,c.experts);
            if (c.config == 1) full_transpose<T,256><<<grid,256,0,stream>>>(low,high,units,output,c.n,c.k,c.experts);
            else full_transpose<T,128><<<grid,128,0,stream>>>(low,high,units,output,c.n,c.k,c.experts);
        }
    }
    return hggcGetLastError() == hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
} // namespace quactlize::dequant

extern "C" int quactlize_kpack_dequant_v1(qzd_call_v1 const* call,
        quactlize_ppu_placed_arrangement_v2 const* arrangement) {
    using namespace quactlize::execution;
    using namespace quactlize::dequant;
    if (!call || call->version != 1 || call->size != sizeof(*call)) return QKG_INVALID;
    auto const& c = *call;
    if (c.operation < 0 || c.operation > 1 || c.config < 0 || c.config > 12) return QKG_INVALID;
    qkg_sizes_v1 s{};
    int rc = sizes(c.qtype,c.n,c.k,c.experts,arrangement,s);
    if (rc) return rc;
    if (c.qtype == 8) return QKG_FORMAT;
    if (c.experts > 65535 || c.k/32 > 65535) return QKG_SHAPE;
    if (c.operation == 1 && c.config == 12 && c.n/32 > 65535) return QKG_SHAPE;
    uint64_t const count = uint64_t(c.n)*c.k*c.experts;
    if ((count+255)/256 > INT32_MAX || count > UINT64_MAX/2) return QKG_OVERFLOW;
    // Production SF prepass forms a signed 32-bit linear thread index.
    if (count/64 > INT32_MAX) return QKG_OVERFLOW;
    uint64_t const out = c.operation ? count*2 : s.sf_plane_bytes;
    if (c.unit_bytes < s.units_bytes || c.output_bytes < out ||
        (c.operation && (c.low_bytes < s.low_bytes || c.high_bytes < s.high_bytes))) return QKG_CAPACITY;
    uintptr_t ptr[] = {uintptr_t(c.units),uintptr_t(c.output),uintptr_t(c.zero),uintptr_t(c.low),uintptr_t(c.high)};
    uint64_t len[] = {s.units_bytes,out,c.operation ? 0 : out,c.operation ? s.low_bytes : 0,c.operation ? s.high_bytes : 0};
    if (c.operation && c.zero) return QKG_INVALID;
    for (int i=0;i<5;++i) if (len[i]) {
        if (!ptr[i] || (ptr[i]&15)) return QKG_INVALID;
        if (!span(ptr[i],len[i])) return QKG_OVERFLOW;
        for(int j=0;j<i;++j) if(overlap(ptr[i],len[i],ptr[j],len[j])) return QKG_INVALID;
    }
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using gguf_scale::KType;
    switch(c.qtype) {
        case 10:return launch<KType::Q2_K>(c);
        case 11:return launch<KType::Q3_K>(c);
        case 12:return launch<KType::Q4_K>(c);
        case 13:return launch<KType::Q5_K>(c);
        case 14:return launch<KType::Q6_K>(c);
        default:return QKG_FORMAT;
    }
}

extern "C" int quactlize_kpack_dequant_probe_v1(int* l2, int* sm, int* warp) {
    if (!l2 || !sm || !warp) return QKG_INVALID;
    int device=0;
    if (hggcGetDevice(&device)!=hggcSuccess ||
        hggcDeviceGetAttribute(l2,hggcDevAttrL2CacheSize,device)!=hggcSuccess ||
        hggcDeviceGetAttribute(sm,hggcDevAttrMultiProcessorCount,device)!=hggcSuccess ||
        hggcDeviceGetAttribute(warp,hggcDevAttrWarpSize,device)!=hggcSuccess) return QKG_RUNTIME;
    return QKG_OK;
}
