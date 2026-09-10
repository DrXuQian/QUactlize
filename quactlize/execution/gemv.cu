#include <hggc_runtime.h>
#include "api.h"
#include "reader.hpp"

#define QKG_CONCAT_IMPL(A,B) A##B
#define QKG_CONCAT(A,B) QKG_CONCAT_IMPL(A,B)
// Each format is a separate translation unit. Explicit namespaces prevent
// same-source host stubs/device registrations from aliasing across objects.
namespace QKG_CONCAT(kpack_q,QKG_QTYPE) {
using namespace quactlize::execution;
#if QKG_QTYPE == 8
using R = Q8Reader;
#else
constexpr KType type = KType(QKG_QTYPE - 10);
using R = Reader<type>;
#endif

__device__ int expert_for(qkg_call_v1 const& c, int row) {
    if (c.mode == QKG_DENSE) return 0;
    if (c.mode == QKG_INDEXED) return c.ids[int64_t(row / c.topk) * c.ids_stride + row % c.topk];
    int lo = 0, hi = c.experts;
    while (lo < hi) {
        int const mid = lo + (hi - lo) / 2;
        if (c.offsets[mid + 1] <= row) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// The pair reader keeps one b16 word from each plane live across all K8
// slots in this metadata group. It uses half2 only for exact code conversion
// and fused affine dequantization; the dot still accumulates in FP32.
template<class Reader>
__device__ __forceinline__ float pair_dot(qkg_call_v1 const& c, int64_t a_base,
    uint16_t const* low, uint16_t const* high, uint8_t const* units,
    int col, int worker, int workers, int partition, int split) {
    using R=Reader;
    float even=0.f, odd=0.f;
    for (int g=partition*workers+worker; g<c.k/R::group; g+=split*workers) {
        auto const s=R::scale(units,col,g,c.n);
        #pragma unroll
        for (int r=0; r<8; ++r) {
            int const begin=g*R::group+r;
            uint16_t lo=low[R::LowMap::word_index(col,begin,c.n)];
            uint16_t hi=0;
            if constexpr (R::hi_bits!=0) hi=high[R::HighMap::word_index(col,begin,c.n)];
            #pragma unroll
            for (int slot=0; slot<R::group/8; slot+=2) {
                int const k0=begin+8*slot, k1=k0+8;
                // Q8 has two slots per word, so the upper K16 half of this
                // scale group needs a second word. K-quants reuse one word.
                if constexpr (R::lo_bits==8) lo=low[R::LowMap::word_index(col,k0,c.n)];
                // Remove the magic integer before multiplying. Folding it
                // into zero would cause FP16 cancellation and change values.
                uint32_t const weight=R::weight_pair(R::raw_from_words(lo,hi,col,k0),
                    R::raw_from_words(lo,hi,col,k1),s);
                float a0=c.input_type==QKG_F32 ? static_cast<float const*>(c.a)[a_base+k0]
                    : float(static_cast<Half const*>(c.a)[a_base+k0]);
                float a1=c.input_type==QKG_F32 ? static_cast<float const*>(c.a)[a_base+k1]
                    : float(static_cast<Half const*>(c.a)[a_base+k1]);
                even=fmaf(float(Half(a0)),float(cutlass::gguf_packed::lo_h2(weight)),even);
                odd=fmaf(float(Half(a1)),float(cutlass::gguf_packed::hi_h2(weight)),odd);
            }
        }
    }
    return even+odd;
}

template<int Columns, int Warps, bool Pair = false>
__global__ void kpack_gemv(qkg_call_v1 c, int split) {
    constexpr int Workers = Warps * 32 / Columns;
    int const tiles = c.n / Columns;
    int const tile = int(blockIdx.x) % tiles;
    int const outer = int(blockIdx.x) / tiles;
    int const partition = outer % split, row = outer / split;
    int const col = tile * Columns + threadIdx.x % Columns;
    int const worker = threadIdx.x / Columns;
    int const expert = expert_for(c, row);
    if (expert < 0 || expert >= c.experts) {
        if (threadIdx.x < Columns) {
            if (split == 1) c.output[int64_t(row) * c.out_row_stride + col] = __int_as_float(0x7fc00000);
            else static_cast<float*>(c.workspace)[(int64_t(row) * split + partition) * c.n + col]
                = __int_as_float(0x7fc00000);
        }
        return;
    }
    int64_t const a_base = c.mode == QKG_INDEXED
        ? int64_t(row / c.topk) * c.a_token_stride + (row % c.topk % c.channels) * c.a_row_stride
        : int64_t(row) * c.a_row_stride;
    int64_t const nk = int64_t(c.n) * c.k;
    auto low = reinterpret_cast<uint16_t const*>(c.low + expert * (nk / 8 * R::lo_bits));
    uint16_t const* high = nullptr;
    if constexpr (R::hi_bits != 0)
        high = reinterpret_cast<uint16_t const*>(c.high + expert * (nk / 8 * R::hi_bits));
    auto units = c.units + expert * R::metadata_bytes(nk);
    float accum = 0.f;
    if constexpr (Pair) {
        accum=pair_dot<R>(c,a_base,low,high,units,col,worker,Workers,partition,split);
    } else {
    for (int g = partition * Workers + worker; g < c.k / R::group; g += split * Workers) {
        auto const s = R::scale(units, col, g, c.n);
        // Eight b16 word addresses per group. Slots are K8-spaced in each
        // word, so the compiler reuses them across the short slot loop.
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            #pragma unroll
            for (int slot = 0; slot < R::group / 8; ++slot) {
                int const kk = g * R::group + r + 8 * slot;
                float a = c.input_type == QKG_F32
                    ? static_cast<float const*>(c.a)[a_base + kk]
                    : float(static_cast<Half const*>(c.a)[a_base + kk]);
                // The mixed-input GEMM boundary converts activations to FP16.
                a = float(Half(a));
                accum = fmaf(a, float(R::weight(R::code(low, high, col, kk, c.n), s)), accum);
            }
        }
    }
    }
    __shared__ float partial[Warps * 32];
    partial[threadIdx.x] = accum;
    __syncthreads();
    if (threadIdx.x < Columns) {
        float value = 0.f;
        #pragma unroll
        for (int w = 0; w < Workers; ++w) value += partial[w * Columns + threadIdx.x];
        if (split == 1) c.output[int64_t(row) * c.out_row_stride + col] = value;
        else static_cast<float*>(c.workspace)[(int64_t(row) * split + partition) * c.n + col] = value;
    }
}

__global__ void kpack_gemv_reduce(qkg_call_v1 c, int split) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < int64_t(c.rows) * c.n; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t const row = i / c.n, col = i % c.n;
        float value = 0.f;
        for (int s = 0; s < split; ++s)
            value += static_cast<float const*>(c.workspace)[(row * split + s) * c.n + col];
        c.output[row * c.out_row_stride + col] = value;
    }
}

template<int Columns, int Warps, bool Pair = false> int launch(qkg_call_v1 const& c, int split) {
    auto stream = static_cast<hggcStream_t>(c.stream);
    unsigned const grid = unsigned(uint64_t(c.rows) * split * (c.n / Columns));
    kpack_gemv<Columns,Warps,Pair><<<grid,Warps*32,0,stream>>>(c,split);
    if (hggcGetLastError() != hggcSuccess) return QKG_RUNTIME;
    if (split > 1) {
        uint64_t const count = (uint64_t(c.rows)*c.n+255)/256;
        kpack_gemv_reduce<<<unsigned(count < 65535 ? count : 65535),256,0,stream>>>(c,split);
    }
    return hggcGetLastError() == hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
}

extern "C" int QKG_CONCAT(qkg_launch_,QKG_QTYPE)(qkg_call_v1 const& c, qkg_config_v1 const& f) {
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
    if (hggcGetLastError() != hggcSuccess) return QKG_RUNTIME;
#if QKG_QTYPE == 8
    if (f.columns == 16) return f.warps == 4 ? launch<16,4,true>(c,f.split) : launch<16,8,true>(c,f.split);
    return f.warps == 4 ? launch<32,4,true>(c,f.split) : launch<32,8,true>(c,f.split);
#else
    if (f.columns == 16) return f.warps == 4 ? launch<16,4>(c,f.split) : launch<16,8>(c,f.split);
    return f.warps == 4 ? launch<32,4>(c,f.split) : launch<32,8>(c,f.split);
#endif
}

extern "C" int QKG_CONCAT(qkg_pair_launch_,QKG_QTYPE)(qkg_call_v1 const& c, qkg_config_v1 const& f) {
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
    if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if (f.columns==16) {
        if (f.warps==2) return launch<16,2,true>(c,f.split);
        return f.warps==4 ? launch<16,4,true>(c,f.split) : launch<16,8,true>(c,f.split);
    }
    if (f.warps==2) return launch<32,2,true>(c,f.split);
    return f.warps==4 ? launch<32,4,true>(c,f.split) : launch<32,8,true>(c,f.split);
}
