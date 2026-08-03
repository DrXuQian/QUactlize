// Local-CUDA resident-kernel timing for the four CUDA-core decode launches filled in the support matrix. Allocations,
// initialisation and transfers are outside every event pair. `rows` is the output dimension N per expert; MoE uses
// one gathered token for each of eight experts, so its logical work and resident weight bytes are eight times N*K.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GEMV_GS_LIST(EMIT) EMIT(16) EMIT(32)
#define GEMV_QUANT_LIST(EMIT, G) EMIT(QuantOp::FinegrainedScaleZero, G)
#include "gemv_lowbit/gemv_launcher.hpp"
#include "gguf_vecdot.hpp"

namespace {

using gguf_scale::KType;
using gguf_scale::vecdot::VecdotActivation;

int ok(cudaError_t e) { return e == cudaSuccess ? 0 : int(e); }

__global__ void flush_l2(uint8_t const* p, size_t bytes, unsigned long long* sink) {
  size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128;
  unsigned long long v = 0;
  for (; i < bytes; i += size_t(gridDim.x) * blockDim.x * 128) v += p[i];
  if (v) atomicAdd(sink, v);
}

struct Buf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit Buf(size_t n = 0) : bytes(n) { if (n && ok(cudaMalloc(&p, n))) p = nullptr; }
  ~Buf() { cudaFree(p); }
  Buf(Buf const&) = delete;
  template <class T> T* as() const { return static_cast<T*>(p); }
  bool good() const { return bytes == 0 || p; }
};

template <KType T> struct Spec;
#define SPEC(T, F, STEP, THREADS, GS, RAW) \
  template <> struct Spec<KType::T> { \
    static constexpr ppu_gemv::WFormat format = ppu_gemv::WFormat::F; \
    static constexpr int step = STEP, threads = THREADS, gs = GS, raw = RAW; \
    using D = ppu_gemv::KernelDetails<ppu_gemv::FP16DetailsA, format, ppu_gemv::WLayout::Native, step, threads>; \
  }
SPEC(Q2_K, Int2,  16, 128, 16,  84);
SPEC(Q3_K, Q3_21, 32,  64, 16, 110);
SPEC(Q4_K, Int4,  16, 128, 32, 144);
SPEC(Q5_K, Q5_41, 32,  64, 32, 176);
SPEC(Q6_K, Q6_42, 16, 128, 16, 210);
#undef SPEC

template <KType T, int RPW, bool Grouped>
void launch_native_fixed(uint8_t const* w, VecdotActivation const* x, int const* offsets, float* out,
                         int n, int bpr, int experts) {
  constexpr int threads = 256;
  dim3 grid(gguf_scale::vecdot::vecdot_grid_size<T, RPW>(n, threads), 1, Grouped ? experts : 1);
  gguf_scale::vecdot::vecdot_rows_kernel<T, RPW, Grouped><<<grid, threads>>>(w, Spec<T>::raw, x, out, n, bpr, offsets);
}

template <KType T, bool Grouped>
void launch_native(uint8_t const* w, VecdotActivation const* x, int const* offsets, float* out,
                   int n, int bpr, int experts) {
  switch (gguf_scale::vecdot::vecdot_rows_per_warp<T>(n, bpr)) {
    case 1: launch_native_fixed<T, 1, Grouped>(w,x,offsets,out,n,bpr,experts); break;
    case 2: launch_native_fixed<T, 2, Grouped>(w,x,offsets,out,n,bpr,experts); break;
    case 4: launch_native_fixed<T, 4, Grouped>(w,x,offsets,out,n,bpr,experts); break;
    case 8: launch_native_fixed<T, 8, Grouped>(w,x,offsets,out,n,bpr,experts); break;
  }
}

template <class Launch>
int time_launch(Launch launch, uint8_t const* flush, size_t flush_bytes, unsigned long long* sink,
                int rows, int reps, float* cold_us, float* warm_us) {
  cudaEvent_t begin{}, end{};
  if (ok(cudaEventCreate(&begin)) || ok(cudaEventCreate(&end))) return 1;
  auto measure = [&](bool cold, float* result) -> int {
    std::vector<float> sample;
    sample.reserve(reps);
    if (!cold) {
      for (int i = 0; i < 100; ++i) launch();
      if (ok(cudaDeviceSynchronize())) return 2;
    }
    // A single small launch is below the 2.048-us event quantum. Batch warm launches so N=2048 spans hundreds of
    // ticks; a cold sample remains exactly one launch after a same-stream L2 flush.
    int const batch = cold ? 1 : std::max(1, std::min(128, (131072 + rows - 1) / rows));
    for (int r = 0; r < reps + 5; ++r) {
      if (cold) flush_l2<<<4096,256>>>(flush, flush_bytes, sink);
      cudaEventRecord(begin);
      for (int i = 0; i < batch; ++i) launch();
      cudaEventRecord(end);
      if (ok(cudaGetLastError()) || ok(cudaEventSynchronize(end))) return 3;
      float ms = 0.f;
      if (ok(cudaEventElapsedTime(&ms, begin, end))) return 4;
      if (r >= 5) sample.push_back(ms * 1000.f / batch);
    }
    std::sort(sample.begin(), sample.end());
    *result = sample[sample.size()/2];
    return 0;
  };
  int rc = measure(true, cold_us);
  if (!rc) rc = measure(false, warm_us);
  cudaEventDestroy(begin); cudaEventDestroy(end);
  return rc;
}

template <KType T, bool Grouped>
int bench_native(int n, int k, int experts, int reps, float* cold, float* warm, double* traffic) {
  int const ecount = Grouped ? experts : 1, bpr = k / 256, total_rows = Grouped ? experts : 1;
  size_t const wb = size_t(ecount) * n * bpr * Spec<T>::raw;
  size_t const xb = size_t(total_rows) * k * sizeof(VecdotActivation);
  size_t const ob = size_t(total_rows) * n * sizeof(float);
  cudaDeviceProp prop{}; int dev = 0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
  size_t const fb = std::max<size_t>(size_t(prop.l2CacheSize) * 2, size_t(128) << 20);
  Buf w(wb), x(xb), out(ob), offsets(Grouped ? size_t(experts+1)*4 : 0), flush(fb), sink(8);
  if (!w.good() || !x.good() || !out.good() || !offsets.good() || !flush.good() || !sink.good()) return 10;
  cudaMemset(w.p,1,wb); cudaMemset(x.p,0x3f,xb); cudaMemset(out.p,0,ob); cudaMemset(flush.p,1,fb); cudaMemset(sink.p,0,8);
  if constexpr (Grouped) {
    std::vector<int> h(experts+1); for (int i=0;i<=experts;++i) h[i]=i;
    cudaMemcpy(offsets.p,h.data(),h.size()*4,cudaMemcpyHostToDevice);
  }
  auto launch = [&] { launch_native<T,Grouped>(w.as<uint8_t>(),x.as<VecdotActivation>(),offsets.as<int>(),
                                               out.as<float>(),n,bpr,experts); };
  launch(); if (ok(cudaGetLastError()) || ok(cudaDeviceSynchronize())) return 11;
  int rc = time_launch(launch,flush.as<uint8_t>(),fb,sink.as<unsigned long long>(),n,reps,cold,warm);
  *traffic = double(wb + xb + ob);
  return rc ? 20+rc : 0;
}

template <KType T, bool Grouped>
int bench_scale(int n, int k, int experts, int reps, float* cold, float* warm, double* traffic) {
  using S = Spec<T>; using D = typename S::D;
  constexpr int lo_bits = D::kLoBits, hi_bits = D::kHiBits;
  int const ecount = Grouped ? experts : 1, total_rows = Grouped ? experts : 1;
  int64_t const lo_per = int64_t(n)*k*lo_bits/8, hi_per = int64_t(n)*k*hi_bits/8;
  int64_t const scale_per = int64_t(k/S::gs)*n;
  size_t const xb=size_t(total_rows)*k*2, lob=size_t(ecount)*lo_per, hib=size_t(ecount)*hi_per;
  size_t const sb=size_t(ecount)*scale_per*2, ob=size_t(total_rows)*n*2;
  cudaDeviceProp prop{}; int dev=0; cudaGetDevice(&dev); cudaGetDeviceProperties(&prop,dev);
  size_t const fb=std::max<size_t>(size_t(prop.l2CacheSize)*2,size_t(128)<<20);
  Buf x(xb),lo(lob),hi(hib),scale(sb),zero(sb),out(ob),offsets(Grouped?size_t(experts+1)*4:0),flush(fb),sink(8);
  if(!x.good()||!lo.good()||!hi.good()||!scale.good()||!zero.good()||!out.good()||!offsets.good()||!flush.good()||!sink.good()) return 30;
  cudaMemset(x.p,0x3c,xb); cudaMemset(lo.p,0x55,lob); if(hib) cudaMemset(hi.p,0x55,hib);
  cudaMemset(scale.p,0x3c,sb); cudaMemset(zero.p,0,sb); cudaMemset(out.p,0,ob); cudaMemset(flush.p,1,fb); cudaMemset(sink.p,0,8);
  if constexpr(Grouped){ std::vector<int> h(experts+1); for(int i=0;i<=experts;++i)h[i]=i;
    cudaMemcpy(offsets.p,h.data(),h.size()*4,cudaMemcpyHostToDevice); }
  ppu_gemv::Params p{};
  p.act=x.p; p.weight=lo.p; p.weight_hi=hi.p; p.scales=scale.p; p.zeros=zero.p; p.out=out.p;
  p.m=total_rows; p.n=n; p.k=k; p.groupsize=S::gs; p.format=S::format;
  p.quant=ppu_gemv::QuantOp::FinegrainedScaleZero; p.layout=ppu_gemv::WLayout::Native;
  if constexpr(Grouped){ p.num_experts=experts; p.row_offsets=offsets.as<int>(); p.max_rows=1;
    p.w_bytes_per_expert=lo_per; p.w_hi_bytes_per_expert=hi_per; p.scale_elems_per_expert=scale_per; }
  auto launch=[&]{ ppu_gemv::launch_gemv<D,8,2>(p,0); };
  int before=ppu_gemv::gemv_fail_count(); launch();
  if(ppu_gemv::gemv_fail_count()!=before||ok(cudaGetLastError())||ok(cudaDeviceSynchronize())) return 31;
  int rc=time_launch(launch,flush.as<uint8_t>(),fb,sink.as<unsigned long long>(),n,reps,cold,warm);
  *traffic=double(xb+lob+hib+2*sb+ob);
  return rc?40+rc:0;
}

template <KType T>
int run_format(char const* name,int n,int k,int experts,int reps){
  for(int route=0;route<4;++route){ float c=0,w=0; double bytes=0; int rc=0;
    if(route==0)rc=bench_native<T,false>(n,k,experts,reps,&c,&w,&bytes);
    if(route==1)rc=bench_native<T,true >(n,k,experts,reps,&c,&w,&bytes);
    if(route==2)rc=bench_scale <T,false>(n,k,experts,reps,&c,&w,&bytes);
    if(route==3)rc=bench_scale <T,true >(n,k,experts,reps,&c,&w,&bytes);
    if(rc){std::fprintf(stderr,"%s route %d failed: %d\n",name,route,rc);return rc;}
    char const* rn[]={"native_dense","native_moe","scale_dense","scale_moe"};
    int mult=(route&1)?experts:1;
    std::printf("%s,%s,%d,%d,%.6f,%.6f,%.0f,%.0f\n",name,rn[route],n,mult,c,w,double(mult)*n*k,bytes);
    std::fflush(stdout);
  } return 0;
}

} // namespace

int main(int argc,char**argv){
  int reps=argc>1?std::atoi(argv[1]):11, experts=argc>2?std::atoi(argv[2]):8, k=2048;
  std::puts("format,route,rows,experts,cold_us,warm_us,elements,bytes");
  for(int n: {131072,2048}){
    if(run_format<KType::Q2_K>("Q2_K",n,k,experts,reps)||run_format<KType::Q3_K>("Q3_K",n,k,experts,reps)||
       run_format<KType::Q4_K>("Q4_K",n,k,experts,reps)||run_format<KType::Q5_K>("Q5_K",n,k,experts,reps)||
       run_format<KType::Q6_K>("Q6_K",n,k,experts,reps)) return 1;
  }
  return 0;
}
