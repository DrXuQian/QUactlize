// Keep CUDA compatibility headers separate from native HGGCRT headers.
#include <cublas_v2.h>

namespace {
using BlasGemm = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType, int, const void*, cudaDataType, int,
    const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
// Check against the actual SDK declaration without linking a second provider.
static_assert(sizeof(static_cast<BlasGemm>(&cublasGemmEx)) == sizeof(void*), "cuBLAS ABI");

int ordinary_blas(void* entry, void* handle, const void* a, const void* weight, void* out,
                  int m, int n, int k) {
    if (!entry || !handle || !a || !weight || !out || m <= 0 || n <= 0 || k <= 0) return 1;
    const float alpha = 1.f, beta = 0.f;
    // Same installed handle, algorithm, workspace and BF16/FP32 contract as Python.
    return int(reinterpret_cast<BlasGemm>(entry)(static_cast<cublasHandle_t>(handle),
        CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight, CUDA_R_16BF, k,
        a, CUDA_R_16BF, k, &beta, out, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
}

extern "C" int strategy_prefill_cublas_launch(void* entry, void* handle, const void* a,
        const void* weight, void* out, int m, int n, int k) {
    return ordinary_blas(entry, handle, a, weight, out, m, n, k);
}
