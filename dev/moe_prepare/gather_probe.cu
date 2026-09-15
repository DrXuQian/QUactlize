#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "quactlize/include/actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
namespace quactlize::runtime {using Half=cutlass::half_t;}
#include "quactlize/runtime/indexed.cuh"
#include "quactlize/runtime/moe_chain.cuh"

struct Plan {float const* source;__half* gate;__half* up;int k;};
__global__ void plain(Plan p) {
    if(threadIdx.x>=32) for(int c=int(threadIdx.x)-32;c<p.k;c+=224)
        for(int r=0;r<8;++r) {p.gate[r*p.k+c]=__float2half(p.source[c]);p.up[r*p.k+c]=__float2half(p.source[c]);}
}
template<class T>
__global__ void flat(float const* a,T* gate,T* up,int k) {
    if(threadIdx.x>=32) for(int c=int(threadIdx.x)-32;c<k;c+=224)
        for(int r=0;r<8;++r) {gate[r*k+c]=T(a[c]);up[r*k+c]=T(a[c]);}
}
__global__ void full(quactlize::runtime::ComputeMoePlan<cutlass::half_t> p,int mode) {
    if(mode==6) {
        auto const& pp=p.gate;uint32_t simt=quactlize::runtime::moe_simt_mask(p);
        if(threadIdx.x>=32 && !((simt&1) && (p.merged || (simt&2)))) {
          for(int64_t col=int(threadIdx.x)-32;col<pp.k;col+=224) {
            cutlass::half_t value=cutlass::half_t(pp.io.a[col]);
            #pragma unroll
            for(int r=0;r<8;++r) {
              if(!(simt&1)) static_cast<cutlass::half_t*>(pp.a)[int64_t(r)*pp.k+col]=value;
              if(!p.merged && !(simt&2)) static_cast<cutlass::half_t*>(p.up.a)[int64_t(r)*pp.k+col]=value;
            }
          }
        }
    } else if(mode==4) {
        if(threadIdx.x>=32) for(int64_t c=int(threadIdx.x)-32;c<p.gate.k;c+=224)
            for(int r=0;r<8;++r) {
                static_cast<cutlass::half_t*>(p.gate.a)[int64_t(r)*p.gate.k+c]=cutlass::half_t(p.gate.io.a[c]);
                static_cast<cutlass::half_t*>(p.up.a)[int64_t(r)*p.gate.k+c]=cutlass::half_t(p.gate.io.a[c]);
            }
    } else if(mode==5) {
        int lane=int(threadIdx.x)-32,stride=224;
        auto const& pp=p.gate;
        if(threadIdx.x>=32) for(int64_t col=lane;col<pp.k;col+=stride) {
            cutlass::half_t value= cutlass::half_t(pp.io.a[col]);
            #pragma unroll
            for(int r=0;r<8;++r) {
                static_cast<cutlass::half_t*>(pp.a)[int64_t(r)*pp.k+col]=value;
                static_cast<cutlass::half_t*>(p.up.a)[int64_t(r)*pp.k+col]=value;
            }
        }
    } else if(mode==2) {
        if(threadIdx.x>=32) for(int c=int(threadIdx.x)-32;c<p.gate.k;c+=224)
            for(int r=0;r<8;++r) {
                static_cast<cutlass::half_t*>(p.gate.a)[r*p.gate.k+c]=cutlass::half_t(p.gate.io.a[c]);
                static_cast<cutlass::half_t*>(p.up.a)[r*p.gate.k+c]=cutlass::half_t(p.gate.io.a[c]);
            }
    } else if(threadIdx.x>=32) quactlize::runtime::moe_m1_gather(p,int(threadIdx.x)-32,224);
}
int main(int argc,char** argv) {
    int mode=argc>1?std::atoi(argv[1]):0;
    float* source;__half *gate,*up;cudaMalloc(&source,512*4);cudaMalloc(&gate,8*512*2);cudaMalloc(&up,8*512*2);
    cudaMemset(source,0,512*4);
    quactlize::runtime::ComputeMoePlan<cutlass::half_t> p{};
    p.gate.k=512;p.gate.a=gate;p.up.a=up;p.gate.io.a=source;
    if(mode==0)plain<<<1,256>>>({source,gate,up,512});
    else if(mode==1)flat<<<1,256>>>(source,reinterpret_cast<cutlass::half_t*>(gate),reinterpret_cast<cutlass::half_t*>(up),512);
    else full<<<1,256>>>(p,mode);
    printf("GATHER_PROBE mode=%d status=%s sizeof_plan=%zu\n",mode,cudaGetErrorString(cudaDeviceSynchronize()),sizeof(p));
    cudaFree(source);cudaFree(gate);cudaFree(up);
}
