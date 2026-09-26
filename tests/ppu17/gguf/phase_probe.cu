// Single-stage ordinary-shared publication next to a real bulk TMA copy.
// No GEMM, stage reuse or metadata decoding. Volatile prevents uniform-value
// propagation from deleting the very shared read/write being investigated.
#include <cuda_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using Pipeline=cutlass::PipelineTmaAsync<4>;
template<bool Fence> __global__ void phase(int* out,int const* input) {
  __shared__ alignas(16) char storage_bytes[sizeof(typename Pipeline::SharedStorage)];
  auto& storage=*reinterpret_cast<typename Pipeline::SharedStorage*>(storage_bytes);
  __shared__ volatile int value;
  __shared__ alignas(16) int data[4];
  typename Pipeline::Params p;
  p.transaction_bytes=16;p.num_consumers=256;p.is_leader=threadIdx.x==0;
  p.role=threadIdx.x<128?Pipeline::ThreadCategory::Producer:Pipeline::ThreadCategory::Consumer;
  Pipeline pipeline(storage,p,cute::make_shape(cute::_1{},cute::_1{},cute::_1{}));
  __syncthreads();
  if(threadIdx.x==0) {
    auto state=cutlass::make_producer_start_state<Pipeline>();
    auto token=pipeline.producer_try_acquire(state);
    while(token!=cutlass::BarrierStatus::WaitDone)token=pipeline.producer_try_acquire(state);
    if constexpr(Fence)asm volatile("":::"memory");
    value=37;
    if constexpr(Fence)asm volatile("":::"memory");
    pipeline.producer_acquire(state,token);
    cute::SM90_BULK_COPY_G2S::copy(input,pipeline.producer_get_barrier(state),data,16);
  } else if(threadIdx.x>=128) {
    typename Pipeline::PipelineState state;
    pipeline.consumer_wait(state);
    if constexpr(Fence)asm volatile("":::"memory");
    out[threadIdx.x-128]=value+data[0]-77;
  }
}
__global__ void invalid_store(volatile int* output){output[300]=1;}
int main(int argc,char** argv) {
  int* out=nullptr;int* input=nullptr;int host[4]={77,0,0,0};
  if(cudaMalloc(&out,256*sizeof(int))!=cudaSuccess)return 2;
  if(cudaMalloc(&input,sizeof(host))!=cudaSuccess)return 2;
  if(cudaMemcpy(input,host,sizeof(host),cudaMemcpyHostToDevice)!=cudaSuccess)return 2;
  if(argc==4&&std::string(argv[3])=="scalar-negative")invalid_store<<<1,1>>>(out);
  else if(argc>=2&&std::atoi(argv[1]))phase<true><<<1,384>>>(out,input);
  else phase<false><<<1,384>>>(out,input);
  auto rc=cudaDeviceSynchronize();
  if(rc!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(rc));return 2;}
  int values[256];cudaMemcpy(values,out,sizeof(values),cudaMemcpyDeviceToHost);cudaFree(out);cudaFree(input);
  for(int v:values)if(v!=37)return 1;
  std::puts("GGUF_Q4_SMOKE PASS scope=ONE_STAGE_PHASE_RELEASE_ACQUIRE_NOT_GEMM");
}
