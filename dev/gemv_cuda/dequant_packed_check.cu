#include <cuda_runtime.h>
#include "quactlize/dequant/vector_kernels.cuh"
#include "quactlize/dequant/packed_kernels.cuh"

// Development CUDA carrier of the exact headers. Not a shipping backend,
// not a replacement dequant implementation, and not PPU performance proof.
template<gguf_scale::KType T>
int launch(int config,uint16_t const* low,uint16_t const* high,uint8_t const* units,
           uint16_t* out,int n,int k,int experts,cudaStream_t stream) {
    using namespace quactlize::dequant;
    if(config==4)full_wide<T,false,true><<<dim3(n/32,k/128,experts),128,0,stream>>>(low,high,units,out,n,k);
    else if(config==5)full_wide<T,true,true><<<dim3(n/32,k/128,experts),128,0,stream>>>(low,high,units,out,n,k);
    else if(config==10)full_packed_exchange<T,128,false><<<dim3(n/32,k/128,experts),128,0,stream>>>(low,high,units,out,n,k);
    else if(config==11)full_packed_exchange<T,256,false><<<dim3(n/32,k/256,experts),128,0,stream>>>(low,high,units,out,n,k);
    else if(config==12)full_packed_exchange<T,256,true><<<dim3(k/256,n/32,experts),128,0,stream>>>(low,high,units,out,n,k);
    else return -1;
    return int(cudaGetLastError());
}
extern "C" int dequant_packed_cuda(int q,int config,uint16_t const* low,uint16_t const* high,
        uint8_t const* units,uint16_t* out,int n,int k,int experts,void* stream) {
    if(n<=0 || n%256 || k<=0 || k%256 || experts<=0)return -1;
    if(q==12)return launch<gguf_scale::KType::Q4_K>(config,low,high,units,out,n,k,experts,static_cast<cudaStream_t>(stream));
    if(q==13)return launch<gguf_scale::KType::Q5_K>(config,low,high,units,out,n,k,experts,static_cast<cudaStream_t>(stream));
    return -1;
}
