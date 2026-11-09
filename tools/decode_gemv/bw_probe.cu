// Achievable streaming-read bandwidth probe (not a GEMV). Each CTA reads a contiguous chunk
// with 16-byte loads and writes one word, so DRAM traffic is the read size.
#include <hggc_runtime_api.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

template<int Unroll>
__global__ void read_kernel(uint4 const* __restrict__ p, size_t words_per_cta, uint32_t* out) {
    uint4 const* base = p + size_t(blockIdx.x) * words_per_cta;
    uint32_t acc = 0;
    for (size_t i = threadIdx.x; i < words_per_cta; i += blockDim.x * Unroll) {
        uint4 v[Unroll];
        #pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            size_t j = i + size_t(u) * blockDim.x;
            v[u] = j < words_per_cta ? base[j] : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int u = 0; u < Unroll; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    size_t bytes = size_t(argc > 1 ? atoll(argv[1]) : 1024) << 20;
    int ring = argc > 2 ? atoi(argv[2]) : 4;
    void* buf = nullptr;
    uint32_t* out = nullptr;
    if (hggcMalloc(&buf, bytes * ring) || hggcMalloc((void**)&out, 1 << 20)) { printf("alloc failed\n"); return 1; }
    hggcMemset(buf, 1, bytes * ring);
    hggcEvent_t a, b;
    hggcEventCreate(&a); hggcEventCreate(&b);
    int ctas[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int threads[] = {128, 256, 512, 1024};
    for (int t : threads) for (int g : ctas) {
        size_t words = bytes / 16 / g;
        std::vector<float> ms;
        for (int rep = 0; rep < 3 * ring; ++rep) {
            uint4 const* p = reinterpret_cast<uint4 const*>(static_cast<char*>(buf) + (rep % ring) * bytes);
            hggcEventRecord(a, 0);
            read_kernel<4><<<g, t>>>(p, words, out);
            hggcEventRecord(b, 0);
            hggcEventSynchronize(b);
            float x; hggcEventElapsedTime(&x, a, b);
            if (rep >= ring) ms.push_back(x);
        }
        std::sort(ms.begin(), ms.end());
        float med = ms[ms.size() / 2];
        printf("threads=%4d ctas=%5d bytes=%zu median_ms=%.4f GBps=%.1f\n", t, g, bytes, med, bytes / med / 1e6);
    }
    return 0;
}
