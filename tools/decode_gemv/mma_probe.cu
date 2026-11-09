// Device probe for the PPU0010 m8n16k16 f16/bf16 MMA register layout used by the
// decode kernels (CuTe MMA_Traits<PPU0010_8x16x16_F32F16F16F32_TN>):
//   A (8x16, row): lane holds m = lane/4, k = 2*(lane%4) + v0 + 8*v1   (2 regs: v1)
//   B (16x16,col): lane holds n = lane/4 + 8*v2, k = 2*(lane%4) + v0 + 8*v1 (4 regs: v1 + 2*v2)
//   C (8x16):      lane holds m = lane/4, n = lane%4 + 4*v          (4 floats)
#include <hggc_runtime_api.h>
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

__device__ inline uint32_t pack_h(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ inline uint32_t pack_b(float x, float y) {
    __ppu_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

template<int BF16>
__global__ void probe(float const* A, float const* B, float* C) {
    int lane = threadIdx.x, m = lane / 4, t = lane % 4;
    uint32_t a[2], b[4];
    for (int v1 = 0; v1 < 2; ++v1) {
        int k = 2 * t + 8 * v1;
        a[v1] = BF16 ? pack_b(A[m * 16 + k], A[m * 16 + k + 1]) : pack_h(A[m * 16 + k], A[m * 16 + k + 1]);
    }
    for (int v2 = 0; v2 < 2; ++v2)
        for (int v1 = 0; v1 < 2; ++v1) {
            int n = m + 8 * v2, k = 2 * t + 8 * v1;
            b[v1 + 2 * v2] = BF16 ? pack_b(B[n * 16 + k], B[n * 16 + k + 1]) : pack_h(B[n * 16 + k], B[n * 16 + k + 1]);
        }
    float d[4] = {0, 0, 0, 0};
    if (BF16)
        asm volatile("ppu.mma.sync.aligned.m8n16k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                       "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
    else
        asm volatile("ppu.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
                     : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
                       "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
    for (int v = 0; v < 4; ++v) C[m * 16 + t + 4 * v] = d[v];
}

int main() {
    float hA[8 * 16], hB[16 * 16], hC[8 * 16], ref[8 * 16];
    for (int i = 0; i < 8 * 16; ++i) hA[i] = float((i * 7) % 13 - 6);
    for (int i = 0; i < 16 * 16; ++i) hB[i] = float((i * 5) % 11 - 5);
    for (int m = 0; m < 8; ++m)
        for (int n = 0; n < 16; ++n) {
            float s = 0;
            for (int k = 0; k < 16; ++k) s += hA[m * 16 + k] * hB[n * 16 + k];
            ref[m * 16 + n] = s;
        }
    float *A, *B, *C;
    hggcMalloc((void**)&A, sizeof hA); hggcMalloc((void**)&B, sizeof hB); hggcMalloc((void**)&C, sizeof hC);
    hggcMemcpy(A, hA, sizeof hA, hggcMemcpyHostToDevice);
    hggcMemcpy(B, hB, sizeof hB, hggcMemcpyHostToDevice);
    int bad = 0;
    for (int bf = 0; bf < 2; ++bf) {
        hggcMemset(C, 0, sizeof hC);
        if (bf) probe<1><<<1, 32>>>(A, B, C); else probe<0><<<1, 32>>>(A, B, C);
        if (hggcDeviceSynchronize() != hggcSuccess) { printf("launch failed\n"); return 2; }
        hggcMemcpy(hC, C, sizeof hC, hggcMemcpyDeviceToHost);
        int e = 0;
        for (int i = 0; i < 8 * 16; ++i) if (std::fabs(hC[i] - ref[i]) > 1e-3f) ++e;
        printf("%s m8n16k16 layout: %s (%d mismatches)\n", bf ? "bf16" : "f16", e ? "FAIL" : "PASS", e);
        bad += e;
    }
    return bad ? 1 : 0;
}
