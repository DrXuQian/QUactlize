// DeepGEMM's small-row provider may consume per-row expert IDs even when
// m_rows is provided. Derive them from the actual GPU permutation each call.
#include <hggc_runtime.h>
#include <cstdint>
namespace quactlize::moe {
__global__ void provider_expert_ids(int const* ids,int const* destination,int* experts,int rows) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<rows;i+=int64_t(gridDim.x)*blockDim.x)
        experts[i]=ids[destination[i]];
}
}
extern "C" int qzm_test_provider_rows(int const* ids,int const* destination,int* experts,int rows,void* stream) {
    if(!ids || !destination || !experts || rows<=0) return 1;
    int blocks=int((int64_t(rows)+255)/256);
    if(blocks>65535) blocks=65535;
    quactlize::moe::provider_expert_ids<<<blocks,256,0,static_cast<hggcStream_t>(stream)>>>(ids,destination,experts,rows);
    return int(hggcGetLastError());
}
