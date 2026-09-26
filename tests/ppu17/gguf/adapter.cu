#include "api.h"
#include <cstring>
#include <stdexcept>
#include <string>
#include "quactlize/backends/ppu17/gguf_prepacked.cuh"

namespace {
thread_local std::string last_error;
void check(cudaError_t rc, char const* where) {
  if (rc != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(rc));
}
template<class F> int guard(F&& f) {
  try { f(); last_error.clear(); return 0; }
  catch (std::exception const& e) { last_error=e.what(); return 1; }
}
struct Plan {
  void* workspace=nullptr;
  size_t bytes=0;
  int device=0;
  virtual ~Plan() { if (workspace) cudaFree(workspace); }
  virtual void execute()=0;
};
template<class T, int M, int Stages=4, bool PackedUnits=false> struct TypedPlan : Plan {
  using Types=quactlize::ppu17::GgufQ4Prepacked<T,128,M,1,Stages,PackedUnits>;
  using Gemm=typename Types::Gemm;
  typename Gemm::Arguments args;
  cudaStream_t stream;
  explicit TypedPlan(q17_gguf_call const& c, void const* units=nullptr)
      : stream(static_cast<cudaStream_t>(c.stream)) {
    using namespace cute;
    auto a_stride=make_stride(int64_t(c.k),_1{},int64_t(0));
    auto s_stride=make_stride(_1{},int64_t(c.n),int64_t(0));
    using UnusedStride=cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
    typename Types::Kernel::MainloopArguments mainloop{
        static_cast<typename Types::Code const*>(c.b),UnusedStride{},
        static_cast<T const*>(c.a),a_stride};
    mainloop.group_size=32;
    if constexpr(PackedUnits) {
      mainloop.ptr_S=static_cast<uint64_t const*>(units);
      mainloop.dS=make_stride(_1{},int64_t(2)*c.n,int64_t(0));
    } else {
      mainloop.ptr_S=static_cast<T const*>(c.scales);
      mainloop.dS=s_stride;
      mainloop.ptr_Z=static_cast<T const*>(c.zeros);
    }
    typename Types::Kernel::EpilogueArguments epilogue{
        {},nullptr,{},static_cast<T*>(c.output),s_stride};
    args=typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
        {c.n,c.m,c.k,1},mainloop,epilogue};
    check(cudaGetDevice(&device),"device");
    args.hw_info.device_id=device;
    args.hw_info.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    if (Gemm::can_implement(args)!=cutlass::Status::kSuccess)
      throw std::runtime_error("Q4_K group32 can_implement rejected");
    bytes=Gemm::get_workspace_size(args);
    if (bytes) check(cudaMalloc(&workspace,bytes),"workspace");
  }
  void execute() override {
    int current; check(cudaGetDevice(&current),"execute device");
    if (current!=device) throw std::runtime_error("prepared device changed");
    Gemm gemm;
    // Include Stream-K lock clearing and all fixup in every measured call.
    if (gemm.initialize(args,workspace,stream)!=cutlass::Status::kSuccess ||
        gemm.run(stream)!=cutlass::Status::kSuccess)
      throw std::runtime_error("Q4_K SF initialize/launch rejected");
    check(cudaGetLastError(),"Q4_K SF full call");
  }
};
template<class T, bool PackedUnits=false> Plan* make(q17_gguf_call const& c,void const* units=nullptr) {
  switch(c.config) {
    case 0:return new TypedPlan<T,16,4,PackedUnits>(c,units);
    case 1:return new TypedPlan<T,64,4,PackedUnits>(c,units);
    case 2:return new TypedPlan<T,128,4,PackedUnits>(c,units);
    case 3:return new TypedPlan<T,16,8,PackedUnits>(c,units);
    case 4:return new TypedPlan<T,64,8,PackedUnits>(c,units);
    case 5:return new TypedPlan<T,128,8,PackedUnits>(c,units);
  }
  throw std::runtime_error("unknown Q4_K SF config");
}
template<class T> void pack(int n,int k,uint8_t const* input,uint8_t* out) {
  using Map=typename quactlize::ppu17::GgufQ4Sf<T,128,16>::Packing;
  auto inner=Map::ppblock_ilvd_NK_to_offset();
  auto blocks=Map::bNbKL_to_offset(cute::make_shape(n,k,1));
  std::memset(out,0,size_t(n)*k/2);
  for (int kk=0;kk<k;++kk) for (int col=0;col<n;++col) {
    uint8_t q=input[size_t(kk)*n+col];
    if (q>15) throw std::runtime_error("invalid original uint4 code");
    size_t offset=blocks(cute::make_coord(col/128,kk/64,0))+inner(cute::make_coord(col%128,kk%64));
    out[offset/2] |= q<<((offset%2)*4);
  }
}
}
Q17_GGUF_EXPORT char const* q17_gguf_error() { return last_error.c_str(); }
Q17_GGUF_EXPORT char const* q17_gguf_contract() { return "GGUF_Q4_K_G32_SF_PRIVATE_MMA_V1"; }
Q17_GGUF_EXPORT char const* q17_gguf_fq_contract() { return "GGUF_Q4_K_G32_PACKED_UNITS_PRIVATE_MMA_V1"; }
Q17_GGUF_EXPORT int q17_gguf_config_count() { return 6; }
Q17_GGUF_EXPORT int q17_gguf_prepare(q17_gguf_call const* c,void** out) {
  return guard([&] {
    if (!c || !out) throw std::runtime_error("null call/plan");
    *out=nullptr;
    if (c->m<1 || c->n<128 || c->n%128 || c->k<256 || c->k%256 ||
        !c->a || !c->b || !c->scales || !c->zeros || !c->output)
      throw std::runtime_error("requires complete Q4_K group32 SF planes and aligned shape");
    int device; check(cudaGetDevice(&device),"device");
    cudaDeviceProp p{}; check(cudaGetDeviceProperties(&p,device),"properties");
    if (p.major!=9 || p.minor!=0) throw std::runtime_error("requires physical SM90");
    if (c->dtype==0) *out=make<cutlass::half_t>(*c);
    else if (c->dtype==1) *out=make<cutlass::bfloat16_t>(*c);
    else throw std::runtime_error("only FP16/BF16");
  });
}
Q17_GGUF_EXPORT int q17_gguf_execute(void* plan) {
  return guard([&] { if (!plan) throw std::runtime_error("null plan"); static_cast<Plan*>(plan)->execute(); });
}
Q17_GGUF_EXPORT int q17_gguf_prepare_fq(q17_gguf_fq_call const* c,void** out) {
  return guard([&] {
    if (!c || !out) throw std::runtime_error("null FQ call/plan");
    *out=nullptr;
    if (c->m<1 || c->n<128 || c->n%128 || c->k<256 || c->k%256 ||
        !c->a || !c->b || !c->units || !c->output || uintptr_t(c->units)%16)
      throw std::runtime_error("requires aligned Q4 packed units and private codes");
    int device; check(cudaGetDevice(&device),"device");
    cudaDeviceProp p{}; check(cudaGetDeviceProperties(&p,device),"properties");
    if (p.major!=9 || p.minor!=0) throw std::runtime_error("requires physical SM90");
    q17_gguf_call common{c->dtype,c->config,c->m,c->n,c->k,c->a,c->b,
                         nullptr,nullptr,c->output,c->stream};
    if (c->dtype==0) *out=make<cutlass::half_t,true>(common,c->units);
    else if (c->dtype==1) *out=make<cutlass::bfloat16_t,true>(common,c->units);
    else throw std::runtime_error("only FP16/BF16");
  });
}
Q17_GGUF_EXPORT void q17_gguf_destroy(void* plan) { delete static_cast<Plan*>(plan); }
Q17_GGUF_EXPORT size_t q17_gguf_workspace(void* plan) { return plan?static_cast<Plan*>(plan)->bytes:0; }
Q17_GGUF_EXPORT int q17_gguf_pack(int dtype,int n,int k,uint8_t const* input,uint8_t* out) {
  return guard([&] {
    if (!input || !out || n<128 || n%128 || k<256 || k%256)
      throw std::runtime_error("invalid GGUF packing geometry");
    if (dtype==0) pack<cutlass::half_t>(n,k,input,out);
    else if (dtype==1) pack<cutlass::bfloat16_t>(n,k,input,out);
    else throw std::runtime_error("unsupported packing dtype");
  });
}
