#include <hggc_runtime.h>
#include "indexed.h"
#include "reader.hpp"
#include "vector_kernels.cuh"
#include "packed_kernels.cuh"
#include "../execution/validation.hpp"

namespace quactlize::dequant {
template<KType T>
int launch_indexed(qzd_call_v1 const& c, ExpertSelection selection) {
    auto stream = static_cast<hggcStream_t>(c.stream);
    auto low = static_cast<uint16_t const*>(c.low);
    auto high = static_cast<uint16_t const*>(c.high);
    auto units = static_cast<uint8_t const*>(c.units);
    auto output = static_cast<uint16_t*>(c.output);
    if (c.config == 4)
        full_wide<T,false,true,true><<<dim3(c.n/32,c.k/128,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k,selection);
    else if (c.config == 5)
        full_wide<T,true,true,true><<<dim3(c.n/32,c.k/128,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k,selection);
    else if constexpr (T == KType::Q4_K || T == KType::Q5_K) {
        if (c.config == 10)
            full_packed_exchange<T,128,false,true><<<dim3(c.n/32,c.k/128,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k,selection);
        else if (c.config == 11)
            full_packed_exchange<T,256,false,true><<<dim3(c.n/32,c.k/256,c.experts),128,0,stream>>>(low,high,units,output,c.n,c.k,selection);
        else return QKG_INVALID;
    } else return QKG_FORMAT;
    return hggcGetLastError() == hggcSuccess ? QKG_OK : QKG_RUNTIME;
}
}

// All buffers retain their E-axis allocation/strides. Only IDs[0:*count]
// are read and written. Unselected BF16 experts remain untouched. IDs/count
// are resident GPU inputs; no allocation, synchronization, or host readback.
extern "C" int quactlize_kpack_dequant_indexed_v1(qzd_call_v1 const* call,
        quactlize_ppu_placed_arrangement_v2 const* arrangement,
        int const* ids, int const* count, int* device_error) {
    using namespace quactlize::execution;
    using namespace quactlize::dequant;
    if (!call || call->version != 1 || call->size != sizeof(*call) ||
        call->operation != 1 || call->zero || !ids || !count || !device_error) return QKG_INVALID;
    auto const& c = *call;
    qkg_sizes_v1 s{};
    int rc = sizes(c.qtype,c.n,c.k,c.experts,arrangement,s);
    if (rc) return rc;
    if (c.qtype == 8) return QKG_FORMAT;
    if (c.experts > 65535 || c.k/128 > 65535) return QKG_SHAPE;
    uint64_t out = uint64_t(c.n)*c.k*c.experts*2;
    if (c.low_bytes < s.low_bytes || c.high_bytes < s.high_bytes ||
        c.unit_bytes < s.units_bytes || c.output_bytes < out) return QKG_CAPACITY;
    uintptr_t ptr[] = {uintptr_t(c.low),uintptr_t(c.high),uintptr_t(c.units),uintptr_t(c.output),
                      uintptr_t(ids),uintptr_t(count),uintptr_t(device_error)};
    uint64_t len[] = {s.low_bytes,s.high_bytes,s.units_bytes,out,uint64_t(c.experts)*4,4,4};
    for (int i=0;i<7;++i) if (len[i]) {
        if (!ptr[i] || ptr[i] % (i<4 ? 16 : 4)) return QKG_INVALID;
        if (!span(ptr[i],len[i])) return QKG_OVERFLOW;
        for (int j=0;j<i;++j) if (overlap(ptr[i],len[i],ptr[j],len[j])) return QKG_INVALID;
    }
    if (hggcGetLastError() != hggcSuccess) return QKG_RUNTIME;
    ExpertSelection selection{ids,count,device_error,c.experts};
    switch (c.qtype) {
        case 10:return launch_indexed<KType::Q2_K>(c,selection);
        case 11:return launch_indexed<KType::Q3_K>(c,selection);
        case 12:return launch_indexed<KType::Q4_K>(c,selection);
        case 13:return launch_indexed<KType::Q5_K>(c,selection);
        case 14:return launch_indexed<KType::Q6_K>(c,selection);
        default:return QKG_FORMAT;
    }
}
