// Direct sanitizer target. No subprocess, Torch, timing or hidden GPU oracle.
#include "api.h"
#include <cuda_runtime.h>
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

void gpu(cudaError_t rc) { if (rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
void api(int rc) { if (rc) throw std::runtime_error(q17_gguf_error()); }
struct Device {
  void* ptr=nullptr;
  Device(size_t bytes,void const* host) { gpu(cudaMalloc(&ptr,bytes));gpu(cudaMemcpy(ptr,host,bytes,cudaMemcpyHostToDevice)); }
  ~Device() { if(ptr)cudaFree(ptr); }
};
__global__ void invalid_store(volatile int* output) { output[8]=1; }
template<class T> void run(int dtype,int config,bool negative) {
  constexpr int M=65,N=256,K=1024;
  std::vector<uint8_t> codes(N*K),packed(N*K/2);
  std::vector<T> a(M*K,T(0)),s(N*K/32),z(s.size()),c(M*N+128,T(-1232));
  for(int k=0;k<K;++k)for(int n=0;n<N;++n)codes[k*N+n]=(k*7+n*3)&15;
  for(int g=0;g<K/32;++g)for(int n=0;n<N;++n) {
    s[g*N+n]=T(float(1+(n+g*5)%63)/4096.f);
    T original_zero(-float((n+g*3)%63)/8192.f);
    z[g*N+n]=T(float(original_zero)+8*float(s[g*N+n]));
  }
  for(int m=0;m<M;++m)a[m*K+(m*37+31)%K]=T(1);
  api(q17_gguf_pack(dtype,N,K,codes.data(),packed.data()));
  Device ad(a.size()*2,a.data()),bd(packed.size(),packed.data()),
         sd(s.size()*2,s.data()),zd(z.size()*2,z.data()),
         cd(negative?16:c.size()*2,c.data());
  q17_gguf_call call{dtype,config,M,N,K,ad.ptr,bd.ptr,sd.ptr,zd.ptr,
                    negative?cd.ptr:static_cast<T*>(cd.ptr)+64,nullptr};
  void* plan=nullptr;api(q17_gguf_prepare(&call,&plan));
  api(q17_gguf_execute(plan));gpu(cudaDeviceSynchronize());
  q17_gguf_destroy(plan);
  if(negative)throw std::runtime_error("sanitizer negative was not rejected");
  gpu(cudaMemcpy(c.data(),cd.ptr,c.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<64;++i)if(float(c[i])!=-1232||float(c[64+M*N+i])!=-1232)
    throw std::runtime_error("output guard overwritten");
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    int k=(m*37+31)%K;
    T product(float(int(codes[k*N+n])-8)*float(s[(k/32)*N+n]));
    T expected(float(product)+float(z[(k/32)*N+n]));
    if(c[64+m*N+n].raw()!=expected.raw())throw std::runtime_error("one-hot output differs");
  }
  std::printf("GGUF_Q4_SMOKE PASS dtype=%d config=%d onehot_raw=PASS guards=PASS target_calls=1\n",dtype,config);
}
int main(int argc,char** argv) {
  try {
    if(argc<3||argc>4)throw std::runtime_error("usage: smoke DTYPE CONFIG [negative]");
    int dtype=std::stoi(argv[1]),config=std::stoi(argv[2]);
    if(argc==4&&std::string(argv[3])=="scalar-negative") {
      int zero=0; Device out(sizeof(zero),&zero);
      invalid_store<<<1,1>>>(static_cast<int*>(out.ptr));
      gpu(cudaDeviceSynchronize());
      throw std::runtime_error("scalar sanitizer negative was not rejected");
    }
    bool negative=argc==4&&std::string(argv[3])=="negative";
    if(dtype==0)run<cutlass::half_t>(dtype,config,negative);
    else if(dtype==1)run<cutlass::bfloat16_t>(dtype,config,negative);
    else throw std::runtime_error("dtype");
    return 0;
  }catch(std::exception const& e){std::fprintf(stderr,"GGUF_Q4_SMOKE FAIL %s\n",e.what());return 1;}
}
