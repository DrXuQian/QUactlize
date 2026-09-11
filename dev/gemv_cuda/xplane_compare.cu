// Development-only wrappers around the unchanged historical Q4 Xplane reader.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#undef CUTLASS_HOST_DEVICE
#undef CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#include "gguf_bc_q4_gemv.hpp"
#include "gguf_unit_pack.hpp"
#include "xplane_offline.hpp"
#include "q4k_pdf_ab_fixture.hpp"
#include <vector>

extern "C" int q4_device_l2_bytes() {
    int value = 0;
    return cudaDeviceGetAttribute(&value,cudaDevAttrL2CacheSize,0) == cudaSuccess
        ? value : -1;
}

extern "C" int q4_xplane_pack(int n, int k, void const* raw,
                              void* low, void* units) {
    if (!raw || !low || !units || n <= 0 || n % 256 || k <= 0 || k % 1024)
        return -1;
    using Block = q4k_pdf_ab::block_q4_K;
    auto blocks = static_cast<Block const*>(raw);
    std::vector<uint8_t> logical(size_t(n) * k), recovered;
    for (int col = 0; col < n; ++col)
        for (int kk = 0; kk < k; ++kk)
            logical[size_t(kk) * n + col] = q4k_pdf_ab::get_q(
                blocks[size_t(col) * (k / 256) + kk / 256], kk % 256);
    xplane::place_derived<4,64,64,64,32,32,1,64>(
        static_cast<int8_t*>(low), logical, n, k);
    xplane::recover_derived<4,64,64,64,32,32,1,64>(
        static_cast<int8_t const*>(low), recovered, n, k);
    if (logical != recovered) return -2;
    gguf_scale::unit_pack::pack<gguf_scale::KType::Q4_K>(
        static_cast<uint8_t const*>(raw), static_cast<uint8_t*>(units), n, k, 1);
    return 0;
}

template<int C, int W>
int launch_q4(int n, int k, void const* a, void const* low, void const* units,
              void* output, void* stream) {
    gguf_scale::bc_q4_gemv::launch<C,W,1>(static_cast<half const*>(a),
        static_cast<uint8_t const*>(low), static_cast<uint8_t const*>(units),
        static_cast<float*>(output), 1, n, k, static_cast<cudaStream_t>(stream));
    return int(cudaGetLastError());
}

extern "C" int q4_xplane_run(int c, int w, int n, int k, void const* a,
    void const* low, void const* units, void* output, void* stream) {
    if (n <= 0 || k < 1024 || k > 8192 || k % 1024 ||
        (c != 1 && c != 2 && c != 4 && c != 8) ||
        (w != 2 && w != 4 && w != 8) || n % (c*w)) return -1;
    for (auto p : {a, low, units})
        if (!p || (reinterpret_cast<uintptr_t>(p) & 15)) return -1;
    if (!output) return -1;
#define ARM(C,W) if (c == C && w == W) return launch_q4<C,W>(n,k,a,low,units,output,stream)
    ARM(1,2); ARM(2,2); ARM(4,2); ARM(8,2);
    ARM(1,4); ARM(2,4); ARM(4,4); ARM(8,4);
    ARM(1,8); ARM(2,8); ARM(4,8); ARM(8,8);
#undef ARM
    return -1;
}
