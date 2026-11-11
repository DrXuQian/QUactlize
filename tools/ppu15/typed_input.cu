// Test-only native AIU reference vs the shipping typed-A publisher.
#include "native.cu"
#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_decode_input.hpp"

template<int H,int K> struct TypedSource {
    using value_type=float;
    float const* data;
    CUTLASS_DEVICE float const& operator()(int row,int k,int tile) const {
        return data[tile*H*K+row*K+k];
    }
};

template<bool Bf16,int H,int K> __global__ void typed_input_delivery(
        float const* input,uint16_t const* rounded,uint16_t* output,int rows) {
    using E=cute::conditional_t<Bf16,cutlass::bfloat16_t,cutlass::half_t>;
    constexpr int Stages=H==16 ? 3 : 2, Count=H*K*Stages, Threads=128;
    __shared__ __align__(128) E shared[Count];
    auto dump=[&](int arm) {
        __syncthreads();
        for(int i=threadIdx.x;i<Count;i+=Threads)
            output[arm*Count+i]=reinterpret_cast<uint16_t const*>(shared)[i];
        __syncthreads();
    };
    // Hardware reference: same padded cube and matched AIU writer consumed
    // by production's PPU15 A reader. No software swizzle on this arm.
    if(threadIdx.x==0) {
        cute::AiuDesc desc{};
        desc.init(reinterpret_cast<uint8_t const*>(rounded),rows,K,K);
        for(int stage=0;stage<Stages;++stage) for(int cube=0;cube<K/64;++cube)
            cute::PPU0015_AIU_LOAD<cute::Int<H*64*16>,E,false,H,64>::copy(
                shared+stage*H*K+cube*H*64,rounded+stage*H*K,desc,cube*64,0);
        cute::cp_async_fence();cute::cp_async_wait<0>();
    }
    dump(0);
    for(int stage=0;stage<Stages;++stage)
        cutlass::gemm::collective::detail::copy_decode_a<H,K,Threads,false,0,0,0>(
            TypedSource<H,K>{input},shared,stage,stage,threadIdx.x,rows);
    dump(1); // exact pre-fix PPU10 publisher, not a fabricated zero output
    for(int stage=0;stage<Stages;++stage)
        cutlass::gemm::collective::detail::copy_decode_a<H,K,Threads,false,0,0,0,cutlass::arch::PPU0015>(
            TypedSource<H,K>{input},shared,stage,stage,threadIdx.x,rows);
    dump(2);
}

extern "C" int qz15_typed_input(void const* input,void const* rounded,void* output,
        int bf16,int height,int k,int rows,void* stream) {
    if(!input || !rounded || !output || rows<1 || rows>height || (bf16!=0 && bf16!=1)) return -1;
#define QZ_TYPED(H,K) if(height==H && k==K) { \
    if(bf16) typed_input_delivery<true,H,K><<<1,128,0,static_cast<hggcStream_t>(stream)>>>( \
        static_cast<float const*>(input),static_cast<uint16_t const*>(rounded),static_cast<uint16_t*>(output),rows); \
    else typed_input_delivery<false,H,K><<<1,128,0,static_cast<hggcStream_t>(stream)>>>( \
        static_cast<float const*>(input),static_cast<uint16_t const*>(rounded),static_cast<uint16_t*>(output),rows); \
    return int(hggcGetLastError()); }
    QZ_TYPED(16,64);QZ_TYPED(16,128);QZ_TYPED(16,256);QZ_TYPED(64,128);
#undef QZ_TYPED
    return -1;
}
