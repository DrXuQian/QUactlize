#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#if !defined(__NVCC__) || defined(__HGGCCC_VER_MAJOR__)
#error This probe requires NVIDIA nvcc, not a PPU compiler wrapper.
#endif
__global__ void half_probe(__half* output) {
  int i=int(blockIdx.x)*int(blockDim.x)+int(threadIdx.x);
  output[i]=__hadd(__int2half_rn(i),__float2half_rn(1.f));
}
int main() {
  __half *device=nullptr, host[32];
  if (cudaMalloc(&device,sizeof(host))!=cudaSuccess) return 1;
  if (cudaMemset(device,0xa5,sizeof(host))!=cudaSuccess) return 2;
  half_probe<<<1,32>>>(device);
  auto launch=cudaGetLastError(), completion=cudaDeviceSynchronize();
  if (launch!=cudaSuccess || completion!=cudaSuccess) return 3;
  if (cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost)!=cudaSuccess) return 4;
  int bad=0;
  for (int i=0;i<32;++i) bad+=__half2float(host[i])!=float(i+1);
  cudaFree(device);
  std::printf("CUDA_HALF_PROBE bad=%d cells=32 runtime=REAL_CUDA\n",bad);
  return bad!=0;
}
