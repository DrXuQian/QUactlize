// Direct packed-unit sanitizer target. CPU oracle never calls the FQ decoder.
#include "api.h"
#include <cuda_runtime.h>
#include "cutlass/half.h"
#include "cutlass/bfloat16.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

void gpu(cudaError_t rc) { if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc)); }
void api(int rc) { if(rc)throw std::runtime_error(q17_gguf_error()); }
struct Device {
  void* ptr=nullptr;
  Device(size_t bytes,void const* host) {
    gpu(cudaMalloc(&ptr,bytes));gpu(cudaMemcpy(ptr,host,bytes,cudaMemcpyHostToDevice));
  }
  ~Device(){if(ptr)cudaFree(ptr);}
};
__global__ void invalid_store(volatile int* output){output[8]=1;}

template<class T> void run(int dtype,int config) {
  constexpr int M=65,N=256,K=1024;
  std::vector<uint8_t> codes(N*K),packed(N*K/2),units(N*K/256*16);
  std::vector<T> a(M*K,T(0)),scale(N*K/32),zero(scale.size()),out(M*N+128,T(-1232));
  for(int k=0;k<K;++k)for(int n=0;n<N;++n)codes[k*N+n]=(k*7+n*3)&15;
  for(int sb=0;sb<K/256;++sb)for(int n=0;n<N;++n) {
    auto d=cutlass::half_t(.000173f*(1+(n+sb*3)%23));
    auto dm=cutlass::half_t(.000091f*(1+(n+sb*7)%19));
    auto* raw=&units[(sb*N+n)*16];
    raw[0]=d.raw();raw[1]=d.raw()>>8;raw[2]=dm.raw();raw[3]=dm.raw()>>8;
    for(int g=0;g<8;++g) {
      int sc=(n+g*5+sb*11)&63,mi=(n+g*3+sb*7)&63;
      for(int minimum=0;minimum<2;++minimum) {
        int bit=32+(g/4)*48+minimum*24+(g%4)*6;
        int field=minimum?mi:sc;
        for(int b=0;b<6;++b)raw[(bit+b)/8]|=((field>>b)&1)<<((bit+b)%8);
      }
      T s(float(d)*sc),z(-float(dm)*mi);
      float correction=8*float(s);
      if constexpr(std::is_same_v<T,cutlass::bfloat16_t>)correction=float(T(correction));
      scale[(sb*8+g)*N+n]=s;zero[(sb*8+g)*N+n]=T(float(z)+correction);
    }
  }
  for(int m=0;m<M;++m)a[m*K+(m*37+31)%K]=T(1);
  api(q17_gguf_pack(dtype,N,K,codes.data(),packed.data()));
  Device ad(a.size()*2,a.data()),bd(packed.size(),packed.data()),
         ud(units.size(),units.data()),cd(out.size()*2,out.data());
  q17_gguf_fq_call call{dtype,config,M,N,K,ad.ptr,bd.ptr,ud.ptr,
                        static_cast<T*>(cd.ptr)+64,nullptr};
  void* plan=nullptr;api(q17_gguf_prepare_fq(&call,&plan));
  api(q17_gguf_execute(plan));gpu(cudaDeviceSynchronize());q17_gguf_destroy(plan);
  gpu(cudaMemcpy(out.data(),cd.ptr,out.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<64;++i)if(float(out[i])!=-1232||float(out[64+M*N+i])!=-1232)
    throw std::runtime_error("output guard overwritten");
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    int k=(m*37+31)%K;
    T product(float(int(codes[k*N+n])-8)*float(scale[(k/32)*N+n]));
    T expected(float(product)+float(zero[(k/32)*N+n]));
    if(out[64+m*N+n].raw()!=expected.raw())throw std::runtime_error("one-hot FQ output differs");
  }
  std::printf("GGUF_Q4_SMOKE PASS metadata=PACKED_UNITS dtype=%d config=%d onehot_raw=PASS guards=PASS target_calls=1\n",dtype,config);
}
int main(int argc,char** argv) {
  try {
    if(argc<3||argc>4)throw std::runtime_error("usage: fq-smoke DTYPE CONFIG [scalar-negative]");
    if(argc==4) {
      if(std::string(argv[3])!="scalar-negative")throw std::runtime_error("unknown mode");
      int zero=0;Device d(sizeof(zero),&zero);invalid_store<<<1,1>>>(static_cast<int*>(d.ptr));
      gpu(cudaDeviceSynchronize());throw std::runtime_error("scalar fault was not detected");
    }
    int dtype=std::stoi(argv[1]),config=std::stoi(argv[2]);
    if(dtype==0)run<cutlass::half_t>(dtype,config);
    else if(dtype==1)run<cutlass::bfloat16_t>(dtype,config);
    else throw std::runtime_error("dtype");
    return 0;
  }catch(std::exception const& e){std::fprintf(stderr,"GGUF_Q4_SMOKE FAIL %s\n",e.what());return 1;}
}
