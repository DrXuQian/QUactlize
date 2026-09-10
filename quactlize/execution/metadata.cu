#include <hggc_runtime.h>
#include "validation.hpp"
#include "gguf_scale_prepass.hpp"

namespace {
using namespace quactlize::execution;
template<gguf_scale::KType T>
int launch(uint8_t const* units, uint16_t* scale, uint16_t* zero,
           int n, int k, int experts, hggcStream_t stream) {
    using namespace gguf_scale;
    prepass::UnitPlaneDesc dst{reinterpret_cast<cutlass::half_t*>(scale),
        reinterpret_cast<cutlass::half_t*>(zero),int64_t(n)*(k/Traits<T>::kGroupSize),n,1};
    auto args = prepass::make_unit_prepass_kernel_args(units,dst,experts,n,k/256);
    int grid = prepass::prepass_unit_grid_size<T>(experts,n,k/256,256);
    prepass::prepass_unit_kernel<T,packed_unit::kCanonicalPlacedZMul<T>>
        <<<grid,256,0,stream>>>(args);
    return hggcGetLastError() == hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
}

extern "C" int quactlize_kpack_sf_prepare_v1(int q, int n, int k, int experts,
        uint8_t const* units, uint64_t unit_bytes, uint16_t* scale, uint16_t* zero,
        uint64_t plane_bytes, quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream) {
    using namespace quactlize::execution;
    if (q==8) return QKG_FORMAT;  // Q8 uses resident d, not a K-quant prepass.
    qkg_sizes_v1 s{};
    int const rc = sizes(q,n,k,experts,arrangement,s);
    if (rc) return rc;
    // The production kernel forms its linear thread index in 32 bits.
    if (uint64_t(experts)*(k/256)*(uint64_t(n)/8) > uint64_t(UINT32_MAX)/32)
        return QKG_OVERFLOW;
    if (unit_bytes < s.units_bytes || plane_bytes < s.sf_plane_bytes) return QKG_CAPACITY;
    uintptr_t const p[] = {uintptr_t(units),uintptr_t(scale),uintptr_t(zero)};
    uint64_t const b[] = {s.units_bytes,s.sf_plane_bytes,s.sf_plane_bytes};
    for (int i = 0; i < 3; ++i) {
        if (!p[i] || (p[i] & 15)) return QKG_INVALID;
        if (!span(p[i],b[i])) return QKG_OVERFLOW;
        for (int j = 0; j < i; ++j)
            if (overlap(p[i],b[i],p[j],b[j])) return QKG_INVALID;
    }
    if (hggcGetLastError() != hggcSuccess) return QKG_RUNTIME;
    using gguf_scale::KType;
    auto st = static_cast<hggcStream_t>(stream);
    switch (q) {
        case 10: return launch<KType::Q2_K>(units,scale,zero,n,k,experts,st);
        case 11: return launch<KType::Q3_K>(units,scale,zero,n,k,experts,st);
        case 12: return launch<KType::Q4_K>(units,scale,zero,n,k,experts,st);
        case 13: return launch<KType::Q5_K>(units,scale,zero,n,k,experts,st);
        case 14: return launch<KType::Q6_K>(units,scale,zero,n,k,experts,st);
        default: return QKG_FORMAT;
    }
}
