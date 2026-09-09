// Diagnostic launch adapters; kernel arithmetic is extracted from llama.cpp.
#include <cstdarg>

// The standalone diagnostic supplies only error reporting and the real CUDA
// device query; it does not replace any arithmetic or launch implementation.
extern "C" void ggml_abort(char const* file, int line, char const* fmt, ...) {
    std::fprintf(stderr,"%s:%d: ",file,line);
    va_list ap; va_start(ap,fmt); std::vfprintf(stderr,fmt,ap); va_end(ap);
    std::abort();
}
void ggml_cuda_error(char const* stmt, char const* func, char const* file, int line, char const* msg) {
    ggml_abort(file,line,"%s: %s: %s",func,stmt,msg);
}
int ggml_cuda_get_device() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

template<ggml_type Type, bool SmallK>
int reference_mmvq(int n, int k, int rows, int channels, void const* raw,
                  int const* ids, float* out, void* quantized, cudaStream_t stream) {
    constexpr int warps = calc_nwarps(Type, 1, MMVQ_PARAMETERS_GENERIC);
    constexpr int rpb = calc_rows_per_block(1, MMVQ_PARAMETERS_GENERIC, SmallK, warps);
    auto params = ggml_cuda_kernel_launch_params(dim3((n+rpb-1)/rpb,rows,1),dim3(32,warps,1),0,stream);
    ggml_cuda_mm_fusion_args_device fusion{};
    ggml_cuda_kernel_launch(mul_mat_vec_q<Type,1,false,SmallK>,params,
        raw,quantized,ids,fusion,out,uint32_t(k),init_fastdiv_values(channels),
        uint32_t(k/256),uint32_t(channels*k/32),uint32_t(n),init_fastdiv_values(1),
        uint32_t(n*k/256),uint32_t(k/32),uint32_t(n),init_fastdiv_values(1),
        uint32_t(0),uint32_t(0),uint32_t(0),uint32_t(rows));
    return int(cudaGetLastError());
}

template<ggml_type Type>
int reference_mmvq_switch(int n, int k, int rows, int channels, void const* raw,
                         int const* ids, float* out, void* quantized, cudaStream_t stream) {
    constexpr int warps = calc_nwarps(Type,1,MMVQ_PARAMETERS_GENERIC);
    constexpr int step = get_vdr_mmvq(Type)*32/ggml_cuda_type_traits<Type>::qi;
    // The current should_use_small_k rule, for Q4_K/Q5_K on NVIDIA >= Turing.
    if (k/256 < warps*step)
        return reference_mmvq<Type,true>(n,k,rows,channels,raw,ids,out,quantized,stream);
    return reference_mmvq<Type,false>(n,k,rows,channels,raw,ids,out,quantized,stream);
}

template<int Q>
__global__ void historical_dmmv_indexed(void const* raw, float const* a, int const* ids,
                                      float* out, int n, int k, int channels) {
    int slot = blockIdx.y;
    int expert = ids ? ids[slot] : 0;
    constexpr int bytes = Q == 12 ? sizeof(block_q4_K) : sizeof(block_q5_K);
    auto weight = static_cast<unsigned char const*>(raw) + int64_t(expert)*n*(k/256)*bytes;
    auto input = a + int64_t(slot%channels)*k;
    if constexpr (Q == 12)
        dmmv_legacy::dequantize_mul_mat_vec_q4_k(weight,input,out+int64_t(slot)*n,k,n);
    else
        dmmv_legacy::dequantize_mul_mat_vec_q5_k(weight,input,out+int64_t(slot)*n,k);
}

// method: 0=current MMVQ including Q8_1, 1=historical DMMV without A quantization.
extern "C" int llama_reference_run(int method, int q, int n, int k, int rows, int channels,
    void const* raw, float const* a, int const* ids, float* out, void* quantized, void* stream_ptr) {
    if ((q!=12 && q!=13) || n<=0 || k<=0 || k%256 || rows<=0 || channels<=0 ||
        (rows!=1 && !ids) || !raw || !a || !out || !quantized || (method!=0 && method!=1)) return 1;
    auto stream = static_cast<cudaStream_t>(stream_ptr);
    if (method == 1) {
        if (q==12)
            historical_dmmv_indexed<12><<<dim3(n,rows,1),32,0,stream>>>(raw,a,ids,out,n,k,channels);
        else
            historical_dmmv_indexed<13><<<dim3(n,rows,1),32,0,stream>>>(raw,a,ids,out,n,k,channels);
        return int(cudaGetLastError());
    }
    auto params = ggml_cuda_kernel_launch_params(dim3((k+CUDA_QUANTIZE_BLOCK_SIZE-1)/CUDA_QUANTIZE_BLOCK_SIZE,channels,1),
        dim3(CUDA_QUANTIZE_BLOCK_SIZE,1,1),0,stream);
    ggml_cuda_kernel_launch(quantize_q8_1,params,a,quantized,int64_t(k),int64_t(k),
        int64_t(channels)*k,int64_t(channels)*k,int64_t(k),uint32_t(channels),init_fastdiv_values(1));
    int error = int(cudaGetLastError());
    if (error) return error;
    return q==12 ? reference_mmvq_switch<GGML_TYPE_Q4_K>(n,k,rows,channels,raw,ids,out,quantized,stream)
                 : reference_mmvq_switch<GGML_TYPE_Q5_K>(n,k,rows,channels,raw,ids,out,quantized,stream);
}
