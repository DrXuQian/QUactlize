// Diagnostic standalone instantiation of vLLM Machete (Apache-2.0).
// Kernel/collective/prepacked layout headers are pinned and never edited.
// Types below match upstream MacheteKernelTemplate's W4A16 affine branch;
// only the torch::Tensor host wrapper is replaced by the small C/POD boundary.
#include "common.cuh"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "libtorch_stable/cutlass_extensions/vllm_numeric_conversion.cuh"
#include "libtorch_stable/quantization/machete/machete_prepacked_layout.cuh"

namespace q17_reference {
using namespace cute;

template<class T, int NTile, int MTile, int ClusterN> struct Machete {
  using Schedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using Tile = Shape<Int<NTile>, Int<MTile>, _64>;
  using Cluster = Shape<Int<ClusterN>, _1, _1>;
  using Packing = machete::PrepackedLayoutBTemplate<T, cutlass::uint4b_t, T, float,
      cutlass::layout::ColumnMajor, Schedule>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, cutlass::layout::ColumnMajor, 0,
      T, cutlass::layout::ColumnMajor, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90AccFetch>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::VLLMCollectiveBuilder<
      cutlass::gemm::collective::MacheteKernelTag, cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp, tuple<cutlass::uint4b_t, T, T>, Packing, 32,
      T, cutlass::layout::ColumnMajor, 8, float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop,
      Epilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  struct Prepared : Plan {
    typename Gemm::Arguments args;
    cudaStream_t stream;
    explicit Prepared(q17_ref_call const &c) : stream(static_cast<cudaStream_t>(c.stream)) {
      using UnusedBStride = cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>;
      auto stride_a = make_stride(int64_t(c.k), _1{}, int64_t(0));
      auto stride_s = make_stride(_1{}, int64_t(c.n), int64_t(0));
      typename Kernel::MainloopArguments mainloop{
          static_cast<cutlass::uint4b_t const *>(c.b), UnusedBStride{},
          static_cast<T const *>(c.a), stride_a,
          static_cast<T const *>(c.scales), stride_s, 64,
          static_cast<T const *>(c.zeros)};
      typename Kernel::EpilogueArguments epilogue{{}, nullptr, {}, static_cast<T *>(c.output), stride_s};
      args = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
          {c.n, c.m, c.k, 1}, mainloop, epilogue};
      if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
        throw std::runtime_error("Machete can_implement rejected");
      allocate(Gemm::get_workspace_size(args));
    }
    void execute() override {
      // Preserve upstream per-call workspace initialization. Any required
      // memset/reduction is captured and INCLUDED in the measured call.
      Gemm gemm;
      auto status = gemm.initialize(args, workspace, stream);
      if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("Machete initialize rejected");
      if (gemm.run(stream) != cutlass::Status::kSuccess)
        throw std::runtime_error("Machete run rejected");
      checked(cudaGetLastError(), "Machete launch");
    }
  };
};

// Complete unique default_heuristic schedule inventory in pinned generate.py.
#define Q17_MACHETE_SCHEDULES(X) \
  X(0,128,16,1) X(1,128,32,2) X(2,128,64,2) \
  X(3,128,128,2) X(4,128,256,2) X(5,256,16,1) \
  X(6,256,32,2) X(7,256,64,2) X(8,256,128,2)

template<class T> Plan *prepare(q17_ref_call const &c) {
  switch(c.config) {
#define CONFIG(I,N,M,C) case I: return new typename Machete<T,N,M,C>::Prepared(c);
    Q17_MACHETE_SCHEDULES(CONFIG)
#undef CONFIG
  }
  throw std::runtime_error("unknown Machete schedule");
}
template<class T> void pack(int n, int k, uint8_t const *input, uint8_t *output) {
  using Map = typename Machete<T,128,16,1>::Packing;
  auto inner = Map::ppblock_ilvd_NK_to_offset();
  auto blocks = Map::bNbKL_to_offset(make_shape(n,k,1));
  std::memset(output, 0, size_t(n)*k/2);
  for (int kk=0; kk<k; ++kk) for (int col=0; col<n; ++col) {
    uint8_t value = input[size_t(kk)*n+col];
    if (value > 15) throw std::runtime_error("not a uint4 code");
    size_t offset = blocks(make_coord(col/128,kk/64,0)) + inner(make_coord(col%128,kk%64));
    output[offset/2] |= value << ((offset%2)*4);
  }
}
}

Q17_REF_EXPORT int q17_ref_config_count() { return 9; }
Q17_REF_EXPORT char const *q17_ref_config_name(int config) {
  switch(config) {
#define NAME(I,N,M,C) case I: return #N "x" #M "x64_cluster" #C "_cooperative_streamk";
    Q17_MACHETE_SCHEDULES(NAME)
#undef NAME
  }
  return "INVALID";
}
Q17_REF_EXPORT int q17_ref_prepare(q17_ref_call const *c, void **out) {
  return q17_reference::guard([&] {
    if (!c || !out) throw std::runtime_error("null call/plan output");
    *out = nullptr; q17_reference::validate(*c);
    *out = c->dtype == 0 ? q17_reference::prepare<cutlass::half_t>(*c)
                        : q17_reference::prepare<cutlass::bfloat16_t>(*c);
  });
}
Q17_REF_EXPORT int q17_ref_pack(int dtype, int n, int k, uint8_t const *input, uint8_t *out) {
  return q17_reference::guard([&] {
    if (!input || !out || n<=0 || n%128 || k<=0 || k%64 || (dtype!=0 && dtype!=1))
      throw std::runtime_error("invalid Machete packing geometry");
    if (dtype==0) q17_reference::pack<cutlass::half_t>(n,k,input,out);
    else q17_reference::pack<cutlass::bfloat16_t>(n,k,input,out);
  });
}
