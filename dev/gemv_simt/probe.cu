#include <cuda_runtime.h>
#include <cstdint>

__global__ void simt_candidate_marker(uint32_t* out) {
    if (threadIdx.x==0) *out=0x514b504b;
}

extern "C" int simt_candidate_probe(int* info,char* name) {
    int device=0;
    auto rc=cudaGetDevice(&device);
    if (rc!=cudaSuccess) return int(rc);
    cudaDeviceProp p{};
    rc=cudaGetDeviceProperties(&p,device);
    if (rc!=cudaSuccess) return int(rc);
    info[0]=device;info[1]=p.multiProcessorCount;info[2]=p.l2CacheSize;
    info[3]=p.warpSize;info[4]=p.major;info[5]=p.minor;
    for (int i=0;i<256;++i) name[i]=p.name[i];
    uint32_t *ptr=nullptr,value=0;
    rc=cudaMalloc(&ptr,sizeof(value));
    if (rc!=cudaSuccess) return int(rc);
    rc=cudaMemset(ptr,0,sizeof(value));
    if (rc==cudaSuccess) {
        simt_candidate_marker<<<1,32>>>(ptr);
        rc=cudaGetLastError();
    }
    if (rc==cudaSuccess) rc=cudaDeviceSynchronize();
    if (rc==cudaSuccess) rc=cudaMemcpy(&value,ptr,sizeof(value),cudaMemcpyDeviceToHost);
    auto released=cudaFree(ptr);
    if (rc!=cudaSuccess) return int(rc);
    if (released!=cudaSuccess) return int(released);
    return value==0x514b504b ? 0 : -1;
}
