#include "api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "../../execution/validation.hpp"
#include "../../dequant/reader.hpp"
#include "gguf_decode.hpp"

namespace {
constexpr uint64_t blas_workspace_bytes = 32ull << 20;
thread_local std::string last_error;
void require(bool value, char const* message) { if (!value) throw std::invalid_argument(message); }
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
template<class F> int guarded(F function) {
    try { function(); last_error.clear(); return 0; }
    catch (std::invalid_argument const& error) { last_error = error.what(); return QKG_INVALID; }
    catch (std::exception const& error) { last_error = error.what(); return QKG_RUNTIME; }
}
uint64_t multiply(uint64_t a, uint64_t b) {
    require(!b || a <= uint64_t(INT64_MAX) / b, "tensor byte extent overflow");
    return a * b;
}
uint64_t aligned(uint64_t n) {
    require(n <= uint64_t(INT64_MAX) - 255, "workspace extent overflow");
    return (n + 255) & ~uint64_t(255);
}
int block_elements(int q) {
    if (q == 7 || q == 8 || q == 20) return 32;
    if (q >= 10 && q <= 14) return 256;
    throw std::invalid_argument("unsupported GGUF quantization type");
}
int block_bytes(int q) {
    switch (q) {
        case 7: return 24; case 8: return 34; case 20: return 18;
        case 10: return 84; case 11: return 110; case 12: return 144;
        case 13: return 176; case 14: return 210;
        default: throw std::invalid_argument("unsupported GGUF quantization type");
    }
}
qcu_sizes_v1 sizes(qcu_problem_v1 const& p, quactlize_ppu_placed_arrangement_v2 const* arrangement) {
    require(p.version == 1 && p.size == sizeof(p), "invalid problem ABI");
    require(p.m > 0 && p.n > 0 && p.k > 0 && p.experts > 0 && p.device >= 0, "invalid matrix geometry");
    require(p.compute_type == QCU_F16 || p.compute_type == QCU_BF16, "expected FP16 or BF16 compute");
    require(p.mode == QCU_DENSE || p.mode == QCU_INDEXED, "unsupported execution mode");
    require(p.topk > 0 && p.topk <= p.experts && (p.channels == 1 || p.channels == p.topk), "invalid indexed geometry");
    require(p.mode != QCU_DENSE || (p.experts == 1 && p.topk == 1 && p.channels == 1), "dense requires one weight");
    require(p.k % block_elements(p.qtype) == 0, "incomplete GGUF block in a row");
    qcu_sizes_v1 s{};
    uint64_t nk = multiply(p.n, p.k), count = multiply(nk, p.experts);
    if (p.layout == QCU_RAW_GGUF) {
        require(arrangement == nullptr, "raw GGUF must not carry a K-pack descriptor");
        s.raw_bytes = multiply(count / block_elements(p.qtype), block_bytes(p.qtype));
    } else {
        require(p.layout == QCU_CANONICAL_KPACK, "unknown weight layout");
        qkg_sizes_v1 packed{};
        require(quactlize::execution::sizes(p.qtype, p.n, p.k, p.experts, arrangement, packed) == 0,
                "shape or descriptor is not supported by the canonical producer");
        s.low_bytes = packed.low_bytes; s.high_bytes = packed.high_bytes; s.units_bytes = packed.units_bytes;
    }
    uint64_t slots = p.mode == QCU_DENSE ? 1 : multiply(p.m, p.topk);
    require(slots <= INT32_MAX, "too many indexed slots");
    s.expanded_weight_bytes = multiply(multiply(slots, nk), 2);
    s.expanded_input_bytes = p.mode == QCU_INDEXED ? multiply(multiply(slots, p.k), 2) : 0;
    s.input_bytes = multiply(multiply(multiply(p.m, p.channels), p.k), 2);
    s.output_bytes = multiply(multiply(multiply(p.m, p.topk), p.n), 2);
    s.ids_bytes = p.mode == QCU_INDEXED ? multiply(slots, 4) : 0;
    uint64_t w = aligned(s.expanded_weight_bytes), a = aligned(s.expanded_input_bytes);
    require(w <= uint64_t(INT64_MAX) - a - blas_workspace_bytes, "workspace size overflow");
    s.workspace_bytes = w + a + blas_workspace_bytes;
    return s;
}
struct Prepared {
    qcu_problem_v1 p;
    qcu_sizes_v1 s;
    cublasHandle_t blas = nullptr;
    ~Prepared() { if (blas) cublasDestroy(blas); }
};

__device__ float half_value(uint8_t const* p) {
    return __half2float(__ushort_as_half(uint16_t(p[0]) | uint16_t(p[1]) << 8));
}
template<int Q> __device__ float raw_value(uint8_t const* p, int i) {
    if constexpr (Q >= 10 && Q <= 14) {
        constexpr auto type = gguf_scale::KType(Q - 10);
        int const group = i / gguf_scale::Traits<type>::kGroupSize;
        float scale, minimum;
        gguf_scale::vecdot::group_dl_ml<type>(p, group, scale, minimum);
        float code = float(gguf_scale::vecdot::code_at<type>(p, i));
        return __fsub_rn(__fmul_rn(code, scale), minimum);
    } else if constexpr (Q == 8) {
        return __fmul_rn(half_value(p), float(int8_t(p[2 + i])));
    } else if constexpr (Q == 7) {
        uint32_t high = uint32_t(p[4]) | uint32_t(p[5]) << 8 | uint32_t(p[6]) << 16 | uint32_t(p[7]) << 24;
        int code = ((p[8 + i % 16] >> (4 * (i / 16))) & 15) | (((high >> i) & 1) << 4);
        return __fadd_rn(__fmul_rn(half_value(p), float(code)), half_value(p + 2));
    } else {
        constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        int code = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
        return __fmul_rn(half_value(p), float(values[code]));
    }
}
template<int Q> __device__ float packed_value(uint8_t const* low, uint8_t const* high,
                                             uint8_t const* units, int n, int col, int kk) {
    if constexpr (Q == 8) {
        using R = quactlize::execution::Q8Reader;
        int code = R::code(reinterpret_cast<uint16_t const*>(low), nullptr, col, kk, n);
        return __fmul_rn(float(code), half_value(units + 2 * (int64_t(kk / 32) * n + col)));
    } else if constexpr (Q >= 10 && Q <= 14) {
        constexpr auto type = gguf_scale::KType(Q - 10);
        using R = quactlize::dequant::Reader<type>;
        int raw = R::code(reinterpret_cast<uint16_t const*>(low), reinterpret_cast<uint16_t const*>(high), col, kk, n)
            + (R::lo_bits == 4 ? 8 : 0);
        constexpr int bias = Q == 11 ? 4 : Q == 14 ? 32 : 0;
        auto affine = R::affine(units, col, kk / R::group, n);
        return __fsub_rn(__fmul_rn(float(raw - bias), affine.scale), affine.minimum);
    } else return 0.f; // Host admission rejects non-K-pack formats.
}
template<class T> __device__ T rounded(float value);
template<> __device__ __half rounded(float value) { return __float2half_rn(value); }
template<> __device__ __nv_bfloat16 rounded(float value) { return __float2bfloat16_rn(value); }

template<int Q, class T, bool Packed>
__global__ void expand(uint8_t const* raw, uint8_t const* low, uint8_t const* high, uint8_t const* units,
                       int32_t const* ids, T const* input, T* weight, T* gathered, int32_t* error,
                       int n, int k, int experts, int slots, int topk, int channels,
                       uint64_t raw_stride, uint64_t low_stride, uint64_t high_stride, uint64_t units_stride) {
    constexpr int block = Q >= 10 && Q <= 14 ? 256 : 32;
    constexpr int bytes = Q == 7 ? 24 : Q == 8 ? 34 : Q == 20 ? 18 : Q == 10 ? 84 :
                          Q == 11 ? 110 : Q == 12 ? 144 : Q == 13 ? 176 : 210;
    int64_t nk = int64_t(n) * k, total = int64_t(slots) * nk;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(blockDim.x) * gridDim.x) {
        int slot = i / nk;
        int64_t position = i % nk;
        int expert = ids ? ids[slot] : 0;
        float value = 0.f;
        if (expert >= 0 && expert < experts) {
            if constexpr (Packed)
                value = packed_value<Q>(low + uint64_t(expert) * low_stride,
                    high ? high + uint64_t(expert) * high_stride : nullptr,
                    units + uint64_t(expert) * units_stride, n, position / k, position % k);
            else value = raw_value<Q>(raw + uint64_t(expert) * raw_stride + (position / block) * bytes, position % block);
        } else if (expert != -1) atomicExch(error, 1);
        weight[i] = rounded<T>(value);
    }
    if (gathered) {
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(slots) * k; i += int64_t(blockDim.x) * gridDim.x) {
            int slot = i / k, expert = ids[slot];
            int64_t row = int64_t(slot / topk) * channels + (channels == 1 ? 0 : slot % topk);
            gathered[i] = expert >= 0 && expert < experts ? input[row * k + i % k] : rounded<T>(0.f);
        }
    }
}

void validate(Prepared const& h, qcu_buffers_v1 const& b) {
    require(b.version == 1 && b.size == sizeof(b), "invalid buffers ABI");
    auto const& s = h.s;
    void const* pointers[] = {b.raw,b.low,b.high,b.units,b.input,b.output,b.ids,b.workspace,b.device_error};
    uint64_t lengths[] = {s.raw_bytes,s.low_bytes,s.high_bytes,s.units_bytes,s.input_bytes,
                          s.output_bytes,s.ids_bytes,s.workspace_bytes,4};
    uint64_t capacities[] = {b.raw_bytes,b.low_bytes,b.high_bytes,b.units_bytes,b.input_bytes,
                             b.output_bytes,b.ids_bytes,b.workspace_bytes,4};
    for (int i = 0; i < 9; ++i) {
        require(capacities[i] >= lengths[i], "buffer capacity is too small");
        if (!lengths[i]) { require(pointers[i] == nullptr, "unexpected unused buffer"); continue; }
        uintptr_t address = reinterpret_cast<uintptr_t>(pointers[i]);
        require(quactlize::execution::span(address, lengths[i]), "invalid buffer extent");
        int alignment = i == 7 ? 256 : i >= 6 ? 4 : (i == 1 || i == 2 || i >= 4) ? 2 : 1;
        require(address % alignment == 0, "buffer alignment differs from contract");
        for (int j = 0; j < i; ++j)
            require(!quactlize::execution::overlap(address, lengths[i], reinterpret_cast<uintptr_t>(pointers[j]), lengths[j]),
                    "input, output, selection and scratch buffers must be disjoint");
    }
    int device = -1; cuda_check(cudaGetDevice(&device));
    require(device == h.p.device, "prepared CUDA device differs from current device");
}
template<int Q, class T, bool Packed> void launch_layout(Prepared const& h, qcu_buffers_v1 const& b) {
    auto const& p = h.p; auto const& s = h.s;
    auto weight = static_cast<T*>(b.workspace);
    auto gathered = p.mode == QCU_INDEXED ? reinterpret_cast<T*>(static_cast<uint8_t*>(b.workspace) + aligned(s.expanded_weight_bytes)) : nullptr;
    int slots = p.mode == QCU_INDEXED ? p.m * p.topk : 1;
    int grid = int(std::min<uint64_t>((s.expanded_weight_bytes / 2 + 255) / 256, 65535));
    expand<Q,T,Packed><<<grid,256,0,static_cast<cudaStream_t>(b.stream)>>>(static_cast<uint8_t const*>(b.raw),
            static_cast<uint8_t const*>(b.low), static_cast<uint8_t const*>(b.high), static_cast<uint8_t const*>(b.units),
            static_cast<int32_t const*>(b.ids), static_cast<T const*>(b.input), weight, gathered, b.device_error,
            p.n,p.k,p.experts,slots,p.topk,p.channels,s.raw_bytes/p.experts,s.low_bytes/p.experts,s.high_bytes/p.experts,s.units_bytes/p.experts);
}
template<int Q, class T> void launch(Prepared const& h, qcu_buffers_v1 const& b) {
    if (h.p.layout == QCU_RAW_GGUF) launch_layout<Q,T,false>(h,b);
    else launch_layout<Q,T,true>(h,b);
}
template<class T> void expand_weights(Prepared const& h, qcu_buffers_v1 const& b) {
    switch (h.p.qtype) {
        case 7: launch<7,T>(h,b); break; case 8: launch<8,T>(h,b); break;
        case 10: launch<10,T>(h,b); break; case 11: launch<11,T>(h,b); break;
        case 12: launch<12,T>(h,b); break; case 13: launch<13,T>(h,b); break;
        case 14: launch<14,T>(h,b); break; case 20: launch<20,T>(h,b); break;
    }
}
void run(Prepared& h, qcu_buffers_v1 const& b, bool gemm) {
    validate(h,b);
    cuda_check(cudaMemsetAsync(b.device_error, 0, 4, static_cast<cudaStream_t>(b.stream)));
    if (h.p.compute_type == QCU_F16) expand_weights<__half>(h,b);
    else expand_weights<__nv_bfloat16>(h,b);
    cuda_check(cudaGetLastError());
    if (!gemm) return;
    auto const& p = h.p; auto const& s = h.s;
    auto scratch = static_cast<uint8_t*>(b.workspace);
    void* blas_workspace = scratch + aligned(s.expanded_weight_bytes) + aligned(s.expanded_input_bytes);
    blas_check(cublasSetStream(h.blas, static_cast<cudaStream_t>(b.stream)));
    blas_check(cublasSetWorkspace(h.blas, blas_workspace, blas_workspace_bytes));
    cudaDataType_t type = p.compute_type == QCU_F16 ? CUDA_R_16F : CUDA_R_16BF;
    float one = 1.f, zero = 0.f;
    if (p.mode == QCU_DENSE)
        blas_check(cublasGemmEx(h.blas, CUBLAS_OP_T, CUBLAS_OP_N, p.n,p.m,p.k,
            &one,scratch,type,p.k,b.input,type,p.k,&zero,b.output,type,p.n,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    else
        blas_check(cublasGemmStridedBatchedEx(h.blas,CUBLAS_OP_T,CUBLAS_OP_N,p.n,1,p.k,
            &one,scratch,type,p.k,int64_t(p.n)*p.k,scratch+aligned(s.expanded_weight_bytes),type,p.k,p.k,
            &zero,b.output,type,p.n,p.n,p.m*p.topk,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
}

extern "C" char const* quactlize_cuda_error_v1() { return last_error.c_str(); }
extern "C" int quactlize_cuda_query_v1(qcu_problem_v1 const* p,
        quactlize_ppu_placed_arrangement_v2 const* arrangement, qcu_sizes_v1* output) {
    return guarded([&] { require(p && output,"null query argument"); *output = sizes(*p,arrangement); });
}
extern "C" int quactlize_cuda_prepare_v1(qcu_problem_v1 const* p,
        quactlize_ppu_placed_arrangement_v2 const* arrangement, void* stream, void** output) {
    return guarded([&] {
        require(p && output,"null preparation argument"); *output = nullptr;
        auto s = sizes(*p,arrangement);
        int device; cuda_check(cudaGetDevice(&device)); require(device == p->device,"current device differs");
        cudaStreamCaptureStatus status; cuda_check(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status));
        require(status == cudaStreamCaptureStatusNone, "preparation must precede graph capture");
        auto h = std::make_unique<Prepared>(); h->p = *p; h->s = s;
        blas_check(cublasCreate(&h->blas));
        blas_check(cublasSetMathMode(h->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        *output = h.release();
    });
}
extern "C" int quactlize_cuda_run_v1(void* handle, qcu_buffers_v1 const* buffers) {
    return guarded([&] { require(handle && buffers,"null execution argument"); run(*static_cast<Prepared*>(handle),*buffers,true); });
}
extern "C" int quactlize_cuda_dequant_v1(void* handle, qcu_buffers_v1 const* buffers) {
    return guarded([&] { require(handle && buffers,"null expansion argument"); run(*static_cast<Prepared*>(handle),*buffers,false); });
}
extern "C" void quactlize_cuda_destroy_v1(void* handle) { delete static_cast<Prepared*>(handle); }
