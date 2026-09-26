// Diagnostic C boundary over unmodified vLLM Marlin device templates
// (Apache-2.0). All three upstream small/large geometry choices are retained.
#include "common.cuh"
#include "libtorch_stable/quantization/marlin/marlin_template.h"
#include <array>
#include <vector>

namespace q17_reference {
struct Config { void const *kernel; int threads, tn, tk, tm; };

template<vllm::ScalarTypeId D, int TM, int TN, int TK, int Threads, bool M8>
Config config() {
  return {reinterpret_cast<void const *>(
      marlin::Marlin<D,vllm::kU4B8.id(),D,D,Threads,TM,TN/16,TK/16,M8,4,8,false>),
      Threads,TN,TK,TM};
}
template<vllm::ScalarTypeId D> Config configuration(q17_ref_call const &c) {
  if (c.m <= 8) {
    switch(c.config) {
      case 0: return config<D,1,128,128,256,true>();
      case 1: return config<D,1,128,64,128,true>();
      case 2: return config<D,1,64,128,128,true>();
    }
  } else if (c.m >= 64 && c.m % 64 == 0) {
    switch(c.config) {
      case 0: return config<D,4,256,64,256,false>();
      case 1: return config<D,4,128,64,128,false>();
      case 2: return config<D,4,64,128,128,false>();
    }
  }
  throw std::runtime_error("Marlin diagnostic scope is M1..8 or multiples of64, configs0..2");
}

struct Prepared : Plan {
  q17_ref_call call;
  Config config;
  int sms=0, shared=0;
  size_t partial_bytes=0;
  Prepared(q17_ref_call const &c, Config cfg) : call(c), config(cfg) {
    int device=0;
    checked(cudaGetDevice(&device),"device");
    checked(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"SM count");
    checked(cudaDeviceGetAttribute(&shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device),"shared limit");
    if (c.n%cfg.tn || c.k%cfg.tk) throw std::runtime_error("Marlin tile divisibility");
    // Upstream scale-only cache-size formula, stages4/group128, no zeros.
    int a=4*cfg.tm*16*cfg.tk*2, b=4*cfg.tk*cfg.tn/2;
    int reduction=cfg.tm*16*(cfg.tn+8)*2, bias=cfg.tn*2;
    int scales=4*((cfg.tk+127)/128)*cfg.tn*2;
    int need=a+std::max({b,reduction,std::min(b,reduction)+bias})+scales;
    if (need>shared) throw std::runtime_error("Marlin shared footprint does not fit");
    checked(cudaFuncSetAttribute(cfg.kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared),"Marlin opt-in shared");
    int scratch_m=std::min(64, ((c.m+15)/16)*16);
    partial_bytes=size_t(sms)*scratch_m*256*sizeof(float);
    allocate(partial_bytes+size_t(sms)*sizeof(int));
    checked(cudaMemsetAsync(workspace,0,bytes,static_cast<cudaStream_t>(c.stream)),"initial Marlin scratch");
  }
  void execute() override {
    auto a=static_cast<int4 const *>(call.a), b=static_cast<int4 const *>(call.b);
    auto c=static_cast<int4 *>(call.output), tmp=static_cast<int4 *>(workspace);
    int4 const *bias=nullptr, *scales=static_cast<int4 const *>(call.scales),
               *zeros=static_cast<int4 const *>(call.zeros);
    float const *a_scales=nullptr, *global_scale=nullptr;
    int const *g_idx=nullptr;
    int groups=call.k/128, m=call.m, n=call.n, k=call.k, lda=k;
    auto locks=reinterpret_cast<int *>(static_cast<uint8_t *>(workspace)+partial_bytes);
    bool has_bias=false, atomic=false, fp32_reduce=true;
    void *args[]={&a,&b,&c,&tmp,&bias,&a_scales,&scales,&global_scale,&zeros,&g_idx,
                  &groups,&m,&n,&k,&lda,&locks,&has_bias,&atomic,&fp32_reduce,&shared};
    // All internal cross-CTA FP32 reduction and lock reset are in this kernel.
    checked(cudaLaunchKernel(config.kernel,dim3(sms),dim3(config.threads),args,
                             shared,static_cast<cudaStream_t>(call.stream)),"Marlin full call");
  }
};
}

Q17_REF_EXPORT int q17_ref_config_count() { return 3; }
Q17_REF_EXPORT char const *q17_ref_config_name(int config) {
  constexpr char const *names[]={"upstream-choice0", "upstream-choice1", "upstream-choice2"};
  return config>=0 && config<3 ? names[config] : "INVALID";
}
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const *c, void **out) {
  return q17_reference::guard([&] {
    if (!c || !out) throw std::runtime_error("null call/plan output");
    *out=nullptr; q17_reference::validate(*c);
    auto cfg=q17_reference::configuration<vllm::kFloat16.id()>(*c);
    *out=new q17_reference::Prepared(*c,cfg);
  });
}
Q17_REF_EXPORT int q17_ref_pack(int dtype,int n,int k,uint8_t const *input,uint8_t *out) {
  return q17_reference::guard([&] {
    if (!input || !out || n<=0 || n%128 || k<=0 || k%128 || dtype!=0)
      throw std::runtime_error("invalid Marlin packing geometry");
    // The pinned get_weight_perm(4,is_a_8bit=False), followed by its 16x64
    // tile permutation. No activation quantization or reference kernel edits.
    std::vector<int> perm;
    for(int i=0;i<32;++i) {
      std::vector<int> one;
      for(int block: {0,1})
        for(int row: {2*(i%4),2*(i%4)+1,2*(i%4+4),2*(i%4+4)+1})
          one.push_back(16*row+i/4+8*block);
      for(int j=0;j<4;++j) for(int p:one) perm.push_back(p+256*j);
    }
    constexpr int ilv[]={0,2,4,6,1,3,5,7};
    std::array<int,1024> interleaved{};
    for(int i=0;i<1024;++i) interleaved[i]=perm[(i/8)*8+ilv[i%8]];
    std::memset(out,0,size_t(n)*k/2);
    for(size_t dst=0;dst<size_t(n)*k;++dst) {
      size_t src=(dst/1024)*1024+interleaved[dst%1024];
      int kb=src/(size_t(n)*16), rem=src%(size_t(n)*16);
      int col=(rem/256)*16+rem%16, kk=kb*16+(rem%256)/16;
      uint8_t value=input[size_t(kk)*n+col];
      if(value>15) throw std::runtime_error("not a uint4 code");
      out[dst/2]|=value<<((dst%2)*4);
    }
  });
}
