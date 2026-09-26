// Force non-superblock-aligned K starts through the real Stream-K scheduler.
#include "api.h"
#include "quactlize/backends/ppu17/gguf_prepacked.cuh"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checked(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
struct Device {
  void* p=nullptr;
  explicit Device(size_t bytes) { checked(cudaMalloc(&p,bytes)); }
  ~Device() { if(p)cudaFree(p); }
  void put(void const* src,size_t bytes) { checked(cudaMemcpy(p,src,bytes,cudaMemcpyHostToDevice)); }
};
__global__ void invalid_store(volatile int* output) { output[8]=1; }

template<class T,int Stages> void run(int dtype) {
  using namespace cute;
  using Types=quactlize::ppu17::GgufQ4Fq<T,128,16,Stages>;
  using Gemm=typename Types::Gemm;
  constexpr int M=65,N=256,K=1280;
  std::vector<uint8_t> codes(N*K),packed(N*K/2),units(N*K/256*16);
  std::vector<T> a(M*K,T(0)),out(M*N+128,T(-1232));
  for(int k=0;k<K;++k)for(int n=0;n<N;++n)codes[k*N+n]=(k*7+n*3)&15;
  // Exact power-of-two header and distinct six-bit groups avoid ambiguous
  // floating-point reductions; the oracle is independent of affine<T>().
  auto d=cutlass::half_t(0.125f);
  for(int sb=0;sb<K/256;++sb)for(int n=0;n<N;++n) {
    auto* raw=&units[(sb*N+n)*16];raw[0]=d.raw();raw[1]=d.raw()>>8;
    for(int g=0;g<8;++g) {
      int bit=32+(g/4)*48+(g%4)*6,field=1+(sb*11+g*3+n)%63;
      for(int b=0;b<6;++b)raw[(bit+b)/8]|=((field>>b)&1)<<((bit+b)%8);
    }
  }
  for(int m=0;m<M;++m)a[m*K+(m*37+31)%K]=T(1);
  if(q17_gguf_pack(dtype,N,K,codes.data(),packed.data()))throw std::runtime_error(q17_gguf_error());
  Device ad(a.size()*2),bd(packed.size()),ud(units.size()),cd(out.size()*2);
  ad.put(a.data(),a.size()*2);bd.put(packed.data(),packed.size());
  ud.put(units.data(),units.size());cd.put(out.data(),out.size()*2);
  using UnusedStride=cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
  typename Types::Kernel::MainloopArguments ml{
    static_cast<typename Types::Code const*>(bd.p),UnusedStride{},
    static_cast<T const*>(ad.p),make_stride(int64_t(K),_1{},int64_t(0))};
  ml.group_size=32;ml.ptr_S=static_cast<uint64_t const*>(ud.p);
  ml.dS=make_stride(_1{},int64_t(2)*N,int64_t(0));
  typename Types::Kernel::EpilogueArguments ep{
    {},nullptr,{},static_cast<T*>(cd.p)+64,make_stride(_1{},int64_t(N),int64_t(0))};
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{N,M,K,1},ml,ep};
  checked(cudaGetDevice(&args.hw_info.device_id));
  args.hw_info.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
  args.scheduler.splits=3;
  args.scheduler.decomposition_mode=Types::Kernel::TileScheduler::DecompositionMode::SplitK;
  if(Gemm::can_implement(args)!=cutlass::Status::kSuccess)throw std::runtime_error("forced split rejected");
  Device workspace(Gemm::get_workspace_size(args));
  auto params=Types::Kernel::to_underlying_arguments(args,workspace.p);
  if(params.scheduler.divmod_splits_.divisor!=3 ||
     params.scheduler.divmod_k_tiles_per_sk_unit_.divisor!=6 ||
     params.scheduler.big_units_!=2)
    throw std::runtime_error("scheduler no longer gives K starts 0,7,14");
  Gemm gemm;
  if(gemm.initialize(args,workspace.p)!=cutlass::Status::kSuccess ||
     gemm.run()!=cutlass::Status::kSuccess)throw std::runtime_error("forced split launch failed");
  checked(cudaDeviceSynchronize());
  checked(cudaMemcpy(out.data(),cd.p,out.size()*2,cudaMemcpyDeviceToHost));
  for(int i=0;i<64;++i)if(float(out[i])!=-1232 || float(out[64+M*N+i])!=-1232)
    throw std::runtime_error("forced split guard overwritten");
  for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
    int k=(m*37+31)%K,g=k/32,sb=k/256;
    T scale(0.125f*(1+(sb*11+(g%8)*3+n)%63));
    T product((int(codes[k*N+n])-8)*float(scale));
    T want(float(product)+float(T(8*float(scale))));
    if(out[64+m*N+n].raw()!=want.raw())throw std::runtime_error("forced split onehot mismatch");
  }
  std::printf("GGUF_Q4_SMOKE PASS fixture=split-start dtype=%d stages=%d M=%d N=%d K=%d splits=3 k_starts=0,7,14 onehot_raw=PASS guards=PASS\n",dtype,Stages,M,N,K);
}
}
int main(int argc,char** argv) {
  try {
    if(argc<3 || argc>4)throw std::runtime_error("usage: split-start DTYPE CONFIG [scalar-negative]");
    if(argc==4) {
      if(std::string(argv[3])!="scalar-negative")throw std::runtime_error("unknown mode");
      Device d(sizeof(int));invalid_store<<<1,1>>>(static_cast<int*>(d.p));
      checked(cudaDeviceSynchronize());throw std::runtime_error("scalar fault was not detected");
    }
    int dtype=std::stoi(argv[1]),config=std::stoi(argv[2]);
    if(dtype==0 && config==0)run<cutlass::half_t,4>(0);
    else if(dtype==0 && config==3)run<cutlass::half_t,8>(0);
    else if(dtype==1 && config==0)run<cutlass::bfloat16_t,4>(1);
    else if(dtype==1 && config==3)run<cutlass::bfloat16_t,8>(1);
    else throw std::runtime_error("only dtype0/1 config0/3");
    return 0;
  }catch(std::exception const& e){std::fprintf(stderr,"GGUF_Q4_SPLIT_START FAIL %s\n",e.what());return 1;}
}
