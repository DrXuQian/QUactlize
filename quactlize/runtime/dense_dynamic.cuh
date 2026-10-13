#pragma once
#include "dynamic.hpp"

namespace quactlize::runtime {
// Uses the exact legacy adapters and reducer, but parameters are local to
// this invocation. Resource attributes were set by query during prepare.
template<class G> int launch_dynamic(typename G::Arguments const& args,hggcStream_t stream) {
  if (G::can_implement(args)!=cutlass::Status::kSuccess || G::get_workspace_size(args)!=0)
    return QK_UNSUPPORTED;
  auto params=G::GemmKernel::to_underlying_arguments(args,nullptr);
  return G::run(params,stream)==cutlass::Status::kSuccess ? QK_OK : QK_RUNTIME_ERROR;
}

template<class T> int run_dense_dynamic(qk_call_v1 const& c,qk_recipe_v1 const& r,int occupancy) {
  using Prepared=typename T::Prepared;
  using Shipping=typename Prepared::ShippingGemm;
  using Split=typename Prepared::SplitGemm;
  using Persistent=typename T::PersistentGemm;
  auto stream=static_cast<hggcStream_t>(c.stream);
  auto output=static_cast<Half*>(c.output);
  auto ml=Prepared::make_mainloop_arguments(static_cast<Half const*>(c.a),
      static_cast<typename T::Low const*>(c.low),c.metadata,c.zero,c.m,c.n,c.k,c.group_size,
      static_cast<typename T::High const*>(c.high));
  if (r.split>1) {
    auto partials=static_cast<float*>(c.workspace);
    auto stride=cutlass::gemm::kernel::detail::make_compact_fp32_partial_stride<typename Split::GemmKernel::StrideD>(c.m,c.n);
    typename Split::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
        {partials,stride,partials,stride},r.split};
    typename Prepared::Reduction reduction;
    typename Prepared::Reduction::Arguments ra{c.m,c.n,r.split,partials,c.workspace_bytes,output,c.n};
    // Reducer initialization only constructs host parameters; it enqueues no work.
    if (reduction.initialize(ra)!=cutlass::Status::kSuccess) return QK_INVALID;
    int rc=launch_dynamic<Split>(args,stream);
    if (rc) return rc;
    return reduction.run(stream)==cutlass::Status::kSuccess ? QK_OK : QK_RUNTIME_ERROR;
  }
  auto sc=cutlass::make_cute_packed_stride(typename Shipping::GemmKernel::StrideC{},cute::make_shape(c.m,c.n,1));
  auto sd=cutlass::make_cute_packed_stride(typename Shipping::GemmKernel::StrideD{},cute::make_shape(c.m,c.n,1));
  if (r.algorithm==QK_PERSISTENT) {
    typename Persistent::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
        {{1.f,0.f},static_cast<Half*>(nullptr),sc,output,sd},
        cutlass::KernelHardwareInfo{c.device,c.compute_units},{},occupancy,uint32_t(r.grid)};
    return launch_dynamic<Persistent>(args,stream);
  }
  typename Shipping::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
      {{1.f,0.f},static_cast<Half*>(nullptr),sc,output,sd},1};
  return launch_dynamic<Shipping>(args,stream);
}
} // namespace quactlize::runtime
