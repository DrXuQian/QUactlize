// PPU CUTLASS4.3 prepared-call boundary. Reuses the upstream reader and
// optionally our existing Marlin integer decomposition/native SM90 fixup.
#include "common.cuh"
#include "quactlize/backends/ppu17/gptq_prepacked.cuh"

namespace q17_reference {
using namespace cute;
template<int N, int M, int ClusterN, class Scheduler, bool PrepareOnce = false, int Stages = 0> struct PrepackedPlan : Plan {
  using Types = quactlize::ppu17::PrepackedGptq<N,M,ClusterN,Scheduler,Stages>;
  using Gemm = typename Types::Gemm;
  typename Gemm::Arguments args;
  cudaStream_t stream;
  explicit PrepackedPlan(q17_ref_call const& c) : stream(static_cast<cudaStream_t>(c.stream)) {
    auto a_stride = make_stride(int64_t(c.k), _1{}, int64_t(0));
    auto s_stride = make_stride(_1{}, int64_t(c.n), int64_t(0));
    using UnusedStride = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
    typename Types::Kernel::MainloopArguments mainloop{
        static_cast<typename Types::Code const*>(c.b), UnusedStride{},
        static_cast<cutlass::half_t const*>(c.a), a_stride,
        static_cast<cutlass::half_t const*>(c.scales), s_stride, 128};
    typename Types::Kernel::EpilogueArguments epilogue{
        {}, nullptr, {}, static_cast<cutlass::half_t*>(c.output), s_stride};
    args = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
        {c.n,c.m,c.k,1}, mainloop, epilogue};
    checked(cudaGetDevice(&args.hw_info.device_id), "GPTQ device");
    args.hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    if (Gemm::can_implement(args) != cutlass::Status::kSuccess) throw std::runtime_error("prepacked GPTQ can_implement rejected");
    allocate(Gemm::get_workspace_size(args));
    if constexpr (PrepareOnce) {
      static_assert(std::is_same_v<Scheduler,quactlize::ppu17::MarlinScheduler>);
      // This scheduler resets both group locks in its final peer. Initialize
      // once while preparing the serial workspace, just like native Marlin.
      // Stream-K does not have that contract and must keep per-call clearing.
      Gemm gemm;
      if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Marlin workspace preparation rejected");
    }
  }
  void execute() override {
    Gemm gemm;
    auto status = PrepareOnce ? gemm.update(args, workspace) : gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("prepacked GPTQ initialize/update rejected");
    if (gemm.run(stream) != cutlass::Status::kSuccess) throw std::runtime_error("prepacked GPTQ launch rejected");
    checked(cudaGetLastError(), "prepacked GPTQ full call");
  }
};
#define Q17_PREPACKED_CONFIGS(X) \
  X(0,128,16,1,cutlass::gemm::StreamKScheduler,streamk) \
  X(1,128,64,2,cutlass::gemm::StreamKScheduler,streamk) \
  X(2,128,128,2,cutlass::gemm::StreamKScheduler,streamk) \
  X(3,128,16,1,quactlize::ppu17::MarlinScheduler,marlin) \
  X(4,128,64,1,quactlize::ppu17::MarlinScheduler,marlin) \
  X(5,128,128,1,quactlize::ppu17::MarlinScheduler,marlin)
}
// Remaining upstream default-heuristic geometries. Keep every reference
// geometry in the candidate inventory, not only last screen's three winners.
#define Q17_PREPACKED_EXTRA(X) \
  X(9,128,32,2) X(10,128,256,2) X(11,256,16,1) \
  X(12,256,32,2) X(13,256,64,2) X(14,256,128,2)
Q17_REF_EXPORT int q17_ref_config_count() { return 19; }
Q17_REF_EXPORT char const* q17_ref_config_name(int config) {
  switch(config) {
#define NAME(I,N,M,C,S,TAG) case I: return #N "x" #M "x64_cluster" #C "_prepacked_cooperative_" #TAG;
    Q17_PREPACKED_CONFIGS(NAME)
#undef NAME
    case 6: return "128x16x64_cluster1_prepacked_marlin_self_reset";
    case 7: return "128x64x64_cluster1_prepacked_marlin_self_reset";
    case 8: return "128x128x64_cluster1_prepacked_marlin_self_reset";
#define NAME(I,N,M,C) case I: return #N "x" #M "x64_cluster" #C "_prepacked_cooperative_streamk";
    Q17_PREPACKED_EXTRA(NAME)
#undef NAME
    case 15: return "128x16x64_cluster1_prepacked_streamk_s4";
    case 16: return "128x16x64_cluster1_prepacked_streamk_s8";
    case 17: return "128x16x64_cluster1_prepacked_marlin_self_reset_s4";
    case 18: return "128x16x64_cluster1_prepacked_marlin_self_reset_s8";
  }
  return "INVALID";
}
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const* c, void** out) {
  return q17_reference::guard([&] {
    if (!c || !out) throw std::runtime_error("null call/plan");
    *out = nullptr; q17_reference::validate(*c);
    switch(c->config) {
#define MAKE(I,N,M,C,S,TAG) case I: *out=new q17_reference::PrepackedPlan<N,M,C,S>(*c); return;
      Q17_PREPACKED_CONFIGS(MAKE)
#undef MAKE
      case 6: *out=new q17_reference::PrepackedPlan<128,16,1,quactlize::ppu17::MarlinScheduler,true>(*c); return;
      case 7: *out=new q17_reference::PrepackedPlan<128,64,1,quactlize::ppu17::MarlinScheduler,true>(*c); return;
      case 8: *out=new q17_reference::PrepackedPlan<128,128,1,quactlize::ppu17::MarlinScheduler,true>(*c); return;
#define MAKE(I,N,M,C) case I: *out=new q17_reference::PrepackedPlan<N,M,C,cutlass::gemm::StreamKScheduler>(*c); return;
      Q17_PREPACKED_EXTRA(MAKE)
#undef MAKE
      case 15: *out=new q17_reference::PrepackedPlan<128,16,1,cutlass::gemm::StreamKScheduler,false,4>(*c); return;
      case 16: *out=new q17_reference::PrepackedPlan<128,16,1,cutlass::gemm::StreamKScheduler,false,8>(*c); return;
      case 17: *out=new q17_reference::PrepackedPlan<128,16,1,quactlize::ppu17::MarlinScheduler,true,4>(*c); return;
      case 18: *out=new q17_reference::PrepackedPlan<128,16,1,quactlize::ppu17::MarlinScheduler,true,8>(*c); return;
    }
    throw std::runtime_error("unknown prepacked GPTQ config");
  });
}
Q17_REF_EXPORT int q17_ref_pack(int dtype, int n, int k, uint8_t const* input, uint8_t* out) {
  return q17_reference::guard([&] {
    if (!input || !out || dtype != 0 || n <= 0 || n % 128 || k <= 0 || k % 128) throw std::runtime_error("invalid prepacked GPTQ geometry");
    using Map = quactlize::ppu17::PrepackedGptq<128,16>::Packing;
    auto inner = Map::ppblock_ilvd_NK_to_offset();
    auto blocks = Map::bNbKL_to_offset(cute::make_shape(n,k,1));
    std::memset(out, 0, size_t(n)*k/2);
    for (int kk=0; kk<k; ++kk) for (int col=0; col<n; ++col) {
      uint8_t q = input[size_t(kk)*n+col];
      if (q > 15) throw std::runtime_error("not a uint4 code");
      size_t offset = blocks(cute::make_coord(col/128,kk/64,0)) + inner(cute::make_coord(col%128,kk%64));
      out[offset/2] |= q << ((offset%2)*4);
    }
  });
}
