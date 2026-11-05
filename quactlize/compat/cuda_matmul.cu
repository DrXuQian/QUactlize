// NVIDIA compatibility only. Same native16/F32-accumulation contract as the
// existing CUDA bridge, but bounded expert/column expansion instead of one
// [N,K] weight copy per token/slot. This is NOT a PPU selector or perf claim.
#include "contract.hpp"
#include "reader.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <string>

namespace quactlize::compat::cuda {
constexpr uint64_t blas_bytes = 32ull << 20;
thread_local std::string error;
void checked(cudaError_t rc) {
    if (rc != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(rc));
}
void checked(cublasStatus_t rc) {
    if (rc != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS status=" + std::to_string(rc));
}
template <class F> int guard(F f) {
    try {
        f();
        error.clear();
        return QPM_OK;
    } catch (std::invalid_argument const &e) {
        error = e.what();
        return QPM_INVALID;
    } catch (std::exception const &e) {
        error = e.what();
        return QPM_RUNTIME;
    }
}
Layout query(qpm_problem_v1 const &p, quactlize_ppu_placed_arrangement_v2 const *a) {
    auto l = layout(p, a);
    auto &s = l.sizes;
    s.route = QPM_GROUPED_TC;
    s.expert_tile = 1;
    s.column_tile = std::min(l.padded_n, 512);
    s.grouped_input_bytes = mul(mul(l.rows, p.k), 2);
    s.grouped_output_bytes = mul(mul(l.rows, s.column_tile), 2);
    s.expanded_weight_bytes = mul(mul(p.k, s.column_tile), 2);
    s.routing_bytes = 0;
    l.a = 0;
    l.c = align(s.grouped_input_bytes);
    l.w = add(l.c, align(s.grouped_output_bytes));
    l.metadata = add(l.w, align(s.expanded_weight_bytes));
    s.workspace_bytes = p.m ? add(l.metadata, blas_bytes) : 0;
    return l;
}
struct Prepared {
    qpm_problem_v1 p;
    Layout l;
    cublasHandle_t blas = nullptr;
    ~Prepared() {
        if (blas)
            cublasDestroy(blas);
    }
};
__device__ int selected(qpm_problem_v1 const &p, qpm_buffers_v1 const &b, int r) {
    return p.mode == QPM_DENSE ? 0 : b.ids[int64_t(r / p.topk) * p.ids_stride + r % p.topk];
}
template <class T> __global__ void gather(qpm_problem_v1 p, qpm_buffers_v1 b, T *a) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.m) * p.topk * p.k;
         i += int64_t(gridDim.x) * blockDim.x) {
        int r = i / p.k, k = i % p.k, e = selected(p, b, r);
        T v = T(0.f);
        if (e >= 0 && e < p.experts)
            v = static_cast<T const *>(b.input)[int64_t(r / p.topk) * p.input_token_stride +
                                                (p.channels == 1 ? 0 : r % p.topk) * p.input_row_stride + k];
        else if (e != -1 && !k)
            atomicOr(b.device_error, QPM_BAD_ID);
        a[i] = v;
    }
}
template <class T, int Q, bool Packed>
__global__ void expand(Weight w, T *out, int e, int first, int columns) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(columns) * w.k;
         i += int64_t(gridDim.x) * blockDim.x)
        out[i] = T(value<Q, Packed>(w, e, first + i / w.k, i % w.k));
}
template <class T>
__global__ void scatter(qpm_problem_v1 p, qpm_buffers_v1 b, T const *c, int expert, int first, int cols,
                        int ld) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(p.m) * p.topk * cols;
         i += int64_t(gridDim.x) * blockDim.x) {
        int r = i / cols, n = i % cols, e = selected(p, b, r);
        if (e == expert)
            static_cast<T *>(b.output)[int64_t(r) * p.output_stride + first + n] = c[int64_t(r) * ld + n];
        else if (expert == 0 && (e < 0 || e >= p.experts))
            static_cast<T *>(b.output)[int64_t(r) * p.output_stride + first + n] = T(0.f);
    }
}
int grid(uint64_t n) { return int(std::max<uint64_t>(1, std::min<uint64_t>(4096, (n + 255) / 256))); }
template <class T, int Q, bool Packed> void run_typed(Prepared &h, qpm_buffers_v1 const &b) {
    auto &p = h.p;
    auto &l = h.l;
    auto &s = l.sizes;
    auto stream = static_cast<cudaStream_t>(b.stream);
    auto *base = static_cast<uint8_t *>(b.workspace);
    auto *a = reinterpret_cast<T *>(base + l.a), *c = reinterpret_cast<T *>(base + l.c),
         *w = reinterpret_cast<T *>(base + l.w);
    Weight weight{static_cast<uint8_t const *>(b.raw),
                  static_cast<uint8_t const *>(b.low),
                  static_cast<uint8_t const *>(b.high),
                  static_cast<uint8_t const *>(b.units),
                  s.raw_bytes / p.experts,
                  s.low_bytes / p.experts,
                  s.high_bytes / p.experts,
                  s.units_bytes / p.experts,
                  p.n,
                  p.k};
    gather<T><<<grid(uint64_t(l.rows) * p.k), 256, 0, stream>>>(p, b, a);
    checked(cublasSetStream(h.blas, stream));
    checked(cublasSetWorkspace(h.blas, base + l.metadata, blas_bytes));
    auto type = p.dtype == QPM_F16 ? CUDA_R_16F : CUDA_R_16BF;
    float alpha = 1, beta = 0;
    for (int e = 0; e < p.experts; ++e)
        for (int first = 0; first < p.n; first += s.column_tile) {
            int cols = std::min(s.column_tile, p.n - first);
            expand<T, Q, Packed><<<grid(uint64_t(cols) * p.k), 256, 0, stream>>>(weight, w, e, first, cols);
            checked(cublasGemmEx(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, cols, l.rows, p.k, &alpha, w, type, p.k, a,
                                 type, p.k, &beta, c, type, s.column_tile, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT));
            scatter<T>
                <<<grid(uint64_t(l.rows) * cols), 256, 0, stream>>>(p, b, c, e, first, cols, s.column_tile);
        }
    checked(cudaGetLastError());
}
template <class T, int Q> void typed(Prepared &h, qpm_buffers_v1 const &b) {
    if (h.p.layout == QPM_RAW_GGUF)
        run_typed<T, Q, false>(h, b);
    else
        run_typed<T, Q, true>(h, b);
}
void run(Prepared &h, qpm_buffers_v1 const &b) {
    validate(h.p, h.l, b);
    require(b.stream, "explicit stream required");
    int device = -1;
    checked(cudaGetDevice(&device));
    require(device == h.p.device, "matmul on wrong device");
    if (!h.p.m)
        return;
#define CASE(Q)                                                                                              \
    case Q:                                                                                                  \
        if (h.p.dtype == QPM_F16)                                                                            \
            typed<cutlass::half_t, Q>(h, b);                                                                 \
        else                                                                                                 \
            typed<cutlass::bfloat16_t, Q>(h, b);                                                             \
        break
    switch (h.p.qtype) {
        CASE(7);
        CASE(8);
        CASE(10);
        CASE(11);
        CASE(12);
        CASE(13);
        CASE(14);
    }
#undef CASE
}
} // namespace quactlize::compat::cuda
extern "C" {
int quactlize_cuda_matmul_query_v1(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a,
                                   qpm_sizes_v1 *s) {
    using namespace quactlize::compat;
    namespace nv = quactlize::compat::cuda;
    return nv::guard([&] {
        require(p && s, "null query");
        *s = nv::query(*p, a).sizes;
    });
}
int quactlize_cuda_matmul_prepare_v1(qpm_problem_v1 const *p, quactlize_ppu_placed_arrangement_v2 const *a,
                                     qpm_buffers_v1 const *b, void **out) {
    using namespace quactlize::compat;
    namespace nv = quactlize::compat::cuda;
    if (out)
        *out = nullptr;
    return nv::guard([&] {
        require(p && b && out && b->stream, "null prepare/stream");
        cudaStreamCaptureStatus status;
        nv::checked(cudaStreamIsCapturing(static_cast<cudaStream_t>(b->stream), &status));
        require(status == cudaStreamCaptureStatusNone, "prepare before capture");
        auto h = std::make_unique<nv::Prepared>();
        h->p = *p;
        h->l = nv::query(*p, a);
        int device = -1;
        nv::checked(cudaGetDevice(&device));
        require(device == p->device, "prepare on wrong device");
        validate(*p, h->l, *b);
        nv::checked(cublasCreate(&h->blas));
        nv::checked(cublasSetMathMode(h->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        nv::run(*h, *b);
        nv::checked(cudaStreamSynchronize(static_cast<cudaStream_t>(b->stream)));
        *out = h.release();
    });
}
int quactlize_cuda_matmul_run_v1(void *h, qpm_buffers_v1 const *b) {
    using namespace quactlize::compat;
    namespace nv = quactlize::compat::cuda;
    return nv::guard([&] {
        require(h && b, "null matmul call");
        nv::run(*static_cast<nv::Prepared *>(h), *b);
    });
}
void quactlize_cuda_matmul_destroy_v1(void *h) { delete static_cast<quactlize::compat::cuda::Prepared *>(h); }
char const *quactlize_cuda_matmul_error_v1(void) { return quactlize::compat::cuda::error.c_str(); }
}
