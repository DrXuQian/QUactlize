// Cache experiments only. This file is not part of the production runtime.
#include <hggc_runtime.h>
#include <cute/arch/copy.hpp>
#include <cstdint>

namespace {
struct Range { uint8_t const* pointer; uint64_t bytes; };
static_assert(sizeof(Range) == 16, "range ABI is pointer plus byte count");

__device__ uint32_t read_cached(uint8_t const* pointer) {
    uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
    return value;
}

template <bool Hint>
__global__ void weight_prefetch(Range const* ranges, int count, uint64_t* receipt) {
    uint64_t const tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t const stride = uint64_t(gridDim.x) * blockDim.x * 32;
    uint32_t checksum = 0, touches = 0;
    #pragma unroll 1
    for (int r = 0; r < count; ++r) {
        Range const range = ranges[r];
        for (uint64_t offset = tid * 32; offset < range.bytes; offset += stride) {
            if constexpr (Hint) cute::prefetch(range.pointer + offset);
            else checksum ^= read_cached(range.pointer + offset);
            ++touches;
        }
    }
    // Makes the load control observable. A hint receipt proves issuance, not
    // completion or residency. The blocking-load control distinguishes them.
    receipt[tid] = (uint64_t(touches) << 32) | checksum;
}

__global__ void cache_pressure(uint8_t const* input, uint64_t bytes, uint32_t* output) {
    uint64_t const tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t value = 0;
    for (uint64_t offset = tid * 32; offset < bytes;
         offset += uint64_t(gridDim.x) * blockDim.x * 32)
        value ^= read_cached(input + offset);
    output[tid] = value;
}
}

extern "C" int kpack_prefetch_device_v1(int64_t* values, int count) {
    if (!values || count != 8) return -1;
    int ordinal = 0;
    auto rc = hggcGetDevice(&ordinal);
    if (rc != hggcSuccess) return int(rc);
    // Use the same scalar CU query as KernelHardwareInfo. Do not infer it
    // from a device-property aggregate whose ABI differs between wrappers.
    hggcDeviceAttr const attributes[] = {
        hggcDevAttrMultiProcessorCount, hggcDevAttrWarpSize, hggcDevAttrL2CacheSize,
        hggcDevAttrMaxSharedMemoryPerMultiprocessor,
        hggcDevAttrMaxRegistersPerMultiprocessor,
        hggcDevAttrMaxThreadsPerMultiProcessor, hggcDevAttrConcurrentKernels};
    for (int i = 0; i < 7; ++i) {
        int value = 0;
        rc = hggcDeviceGetAttribute(&value, attributes[i], ordinal);
        if (rc != hggcSuccess) return int(rc);
        values[i] = value;
    }
    values[7] = ordinal;
    return 0;
}

extern "C" int kpack_prefetch_v1(void const* ranges, int count, int hint,
        int blocks, void* receipt, void* stream) {
    if (!ranges || !receipt || count < 1 || count > 24 ||
        blocks < 1 || blocks > 72 || (hint != 0 && hint != 1)) return -1;
    auto prior = hggcGetLastError();
    if (prior != hggcSuccess) return int(prior);
    if (hint) weight_prefetch<true><<<blocks, 128, 0, static_cast<hggcStream_t>(stream)>>>(
        static_cast<Range const*>(ranges), count, static_cast<uint64_t*>(receipt));
    else weight_prefetch<false><<<blocks, 128, 0, static_cast<hggcStream_t>(stream)>>>(
        static_cast<Range const*>(ranges), count, static_cast<uint64_t*>(receipt));
    return int(hggcGetLastError());
}

extern "C" int kpack_cache_pressure_v1(void const* input, uint64_t bytes,
        void* output, void* stream) {
    if (!input || !output || bytes < 32 || bytes % 32) return -1;
    auto prior = hggcGetLastError();
    if (prior != hggcSuccess) return int(prior);
    cache_pressure<<<288, 256, 0, static_cast<hggcStream_t>(stream)>>>(
        static_cast<uint8_t const*>(input), bytes, static_cast<uint32_t*>(output));
    return int(hggcGetLastError());
}
