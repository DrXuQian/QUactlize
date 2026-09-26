// Diagnostic prepared-call boundary for the actual PPU1.7 GPTQ scale-only
// collective. This TU uses only PPU CUTLASS3.6.0, never vLLM's include graph.
#include "common.cuh"
#include "quactlize/backends/ppu17/gptq.cuh"

namespace q17_reference {
using namespace cute;

template<int N, int M, int K, class Scheduler = cutlass::gemm::StreamKScheduler, int Stages = 0> struct Prepared : Plan {
  using Types = quactlize::ppu17::GptqScaleOnly<N,M,K,Scheduler,Stages>;
  using Gemm = typename Types::Gemm;
  typename Gemm::Arguments args;
  cudaStream_t stream;
  explicit Prepared(q17_ref_call const &c) : stream(static_cast<cudaStream_t>(c.stream)) {
    auto a_stride = make_stride(int64_t(c.k), _1{}, int64_t(0));
    auto s_stride = make_stride(_1{}, int64_t(c.n), int64_t(0));
    typename Types::Kernel::MainloopArguments mainloop{
        static_cast<cutlass::int4b_t const *>(c.b), a_stride,
        static_cast<cutlass::half_t const *>(c.a), a_stride,
        static_cast<cutlass::half_t const *>(c.scales), s_stride, 128};
    typename Types::Kernel::EpilogueArguments epilogue{
        {}, nullptr, {}, static_cast<cutlass::half_t *>(c.output), s_stride};
    args = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
        {c.n,c.m,c.k,1}, mainloop, epilogue};
    checked(cudaGetDevice(&args.hw_info.device_id), "GPTQ device");
    args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
      throw std::runtime_error("GPTQ scale-only can_implement rejected");
    allocate(Gemm::get_workspace_size(args));
  }
  void execute() override {
    Gemm gemm;
    if (gemm.initialize(args,workspace,stream) != cutlass::Status::kSuccess)
      throw std::runtime_error("GPTQ scale-only initialize rejected");
    if (gemm.run(stream) != cutlass::Status::kSuccess)
      throw std::runtime_error("GPTQ scale-only run rejected");
    checked(cudaGetLastError(),"GPTQ full call");
  }
};

#define Q17_GPTQ_CONFIGS(X) \
  X(0,128,16,64) X(1,128,32,64) X(2,128,64,64) \
  X(3,128,128,64) X(4,128,64,128) X(5,256,64,64)
}

Q17_REF_EXPORT int q17_ref_config_count() { return 12; }
Q17_REF_EXPORT char const *q17_ref_config_name(int config) {
  switch (config) {
#define NAME(I,N,M,K) case I: return #N "x" #M "x" #K "_gptq_scaleonly_cooperative_streamk";
    Q17_GPTQ_CONFIGS(NAME)
#undef NAME
    case 6: return "128x16x64_gptq_scaleonly_cooperative_marlin";
    case 7: return "128x64x64_gptq_scaleonly_cooperative_marlin";
    case 8: return "128x16x64_gptq_scaleonly_cooperative_streamk_s4";
    case 9: return "128x16x64_gptq_scaleonly_cooperative_streamk_s8";
    case 10: return "128x16x64_gptq_scaleonly_cooperative_marlin_s4";
    case 11: return "128x16x64_gptq_scaleonly_cooperative_marlin_s8";
  }
  return "INVALID";
}
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const *c, void **out) {
  return q17_reference::guard([&] {
    if (!c || !out) throw std::runtime_error("null call/plan");
    *out=nullptr; q17_reference::validate(*c);
    switch(c->config) {
#define CONFIG(I,N,M,K) case I: *out=new q17_reference::Prepared<N,M,K>(*c); return;
      Q17_GPTQ_CONFIGS(CONFIG)
#undef CONFIG
      case 6: *out=new q17_reference::Prepared<128,16,64,quactlize::ppu17::MarlinScheduler>(*c); return;
      case 7: *out=new q17_reference::Prepared<128,64,64,quactlize::ppu17::MarlinScheduler>(*c); return;
      case 8: *out=new q17_reference::Prepared<128,16,64,cutlass::gemm::StreamKScheduler,4>(*c); return;
      case 9: *out=new q17_reference::Prepared<128,16,64,cutlass::gemm::StreamKScheduler,8>(*c); return;
      case 10: *out=new q17_reference::Prepared<128,16,64,quactlize::ppu17::MarlinScheduler,4>(*c); return;
      case 11: *out=new q17_reference::Prepared<128,16,64,quactlize::ppu17::MarlinScheduler,8>(*c); return;
    }
    throw std::runtime_error("unknown GPTQ config");
  });
}
Q17_REF_EXPORT int q17_ref_pack(int dtype, int n, int k, uint8_t const *input, uint8_t *out) {
  return q17_reference::guard([&] {
    if (!input || !out || dtype!=0 || n<=0 || n%128 || k<=0 || k%128)
      throw std::runtime_error("invalid GPTQ packing geometry");
    std::memset(out,0,size_t(n)*k/2);
    for (int col=0;col<n;++col) for (int kk=0;kk<k;++kk) {
      uint8_t q=input[size_t(kk)*n+col];
      if (q>15) throw std::runtime_error("not a uint4 code");
      size_t offset=size_t(col)*k+kk;
      out[offset/2] |= ((int(q)-8)&15) << ((offset%2)*4);
    }
  });
}
