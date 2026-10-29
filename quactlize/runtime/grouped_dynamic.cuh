#pragma once
#include "grouped_dynamic.hpp"
#include "gguf_packed_unit.hpp"

// Included after the legacy module/query implementation. Reuse its exact
// Group<> specializations, metadata kernels, directory and ordered reducer.
// Only host parameter binding and reducer ownership differ from bound v4.
namespace quactlize::runtime {
template<int Q> inline uint64_t grouped_packed_metadata_bytes(uint64_t count) {
  if constexpr(Q==8) return count/32*2;
  else {
    using Unit=gguf_scale::packed_unit::Unit<static_cast<gguf_scale::KType>(Q-10)>;
    return count/256*Unit::kSbBytes;
  }
}
template<bool Persistent,bool Split,bool Compact> struct DynamicGroupedHandle final : Handle {
  static constexpr bool Directory=Persistent || Compact;
  using Bound=GroupedHandle<Persistent,Split,Compact>;
  using T=typename Bound::T;
  using K=typename Bound::K;
  using G=typename Bound::G;
  using Output=typename Bound::Output;
  using Shape=typename Bound::Shape;
  using DStride=typename Bound::DStride;
  qk_compute_device_call_v4 shape{};
  qk_recipe_v1 recipe{};
  qk_resources_v1 resources{};
  GroupedWorkspace layout{};
  uint64_t head=0,low_bytes=0,high_bytes=0,meta_bytes=0;

  int prepare(qk_compute_device_call_v4 const& d,qk_recipe_v1 const& r,qk_resources_v1 const& z) {
    shape=d; shape.device_call.call.stream=nullptr; recipe=r; resources=z;
    auto const& c=d.device_call.call;
    head=scheduler_bytes(d.device_call.max_rows,c.experts,Directory,c.m);
    if(!grouped_workspace(c.experts,r.split,c.m,c.n,head,sizeof(Shape),sizeof(DStride),true,layout) ||
       layout.total!=resources.workspace_bytes) return QK_INVALID;
    uint64_t count=uint64_t(c.n)*c.k;
    if(count>uint64_t(INT64_MAX)/uint64_t(c.experts)) return QK_INVALID;
    count*=c.experts;
    low_bytes=count/8*F::spec.low_bits; high_bytes=count/8*F::spec.high_bits;
    if constexpr(packed) meta_bytes=grouped_packed_metadata_bytes<qtype>(count);
    else meta_bytes=count/F::spec.group_size*2;
    return QK_OK;
  }
  int run(hggcStream_t) override { return QK_INVALID; }
  int grouped_output(qk_compute_device_call_v4 const& d,int mode,qk_grouped_result_v1& out) const override {
    out={};
    if(!same_grouped_shape(d,shape) ||
       !grouped_buffers(d.device_call.call,layout.total,F::spec.high_bits!=0,needs_zero,
                        low_bytes,high_bytes,meta_bytes)) return QK_INVALID;
    return grouped_result(d,recipe.split,mode,layout,out);
  }
  int run_grouped_dynamic(qk_compute_device_call_v4 const& d,int mode) const override {
    return launch(d,mode,-1);
  }
#ifdef QK_GROUPED_DYNAMIC_TESTING
  int grouped_stage(qk_compute_device_call_v4 const& d,int mode,int stage) const override {
    if(stage<0 || stage>3 || (stage==1 && !Directory) ||
       (stage==3 && (!Split || mode!=QK_GROUPED_COMPLETE))) return QK_INVALID;
    return launch(d,mode,stage);
  }
#endif
  int launch(qk_compute_device_call_v4 const& d,int mode,int stage) const {
    qk_grouped_result_v1 result{};
    int rc=grouped_output(d,mode,result); if(rc) return rc;
    auto const& c=d.device_call.call;
    int const maximum=d.device_call.max_rows;
    auto stream=static_cast<hggcStream_t>(c.stream);
    auto base=static_cast<char*>(c.workspace);
    auto ds=reinterpret_cast<Shape*>(base+layout.shapes);
    auto dp=reinterpret_cast<Output**>(base+layout.outputs);
    auto dd=reinterpret_cast<DStride*>(base+layout.strides);
    auto rows=reinterpret_cast<int*>(base+layout.rows);
    auto partials=reinterpret_cast<float*>(base+layout.partials);
    auto sa=cutlass::make_cute_packed_stride(typename K::StrideA{},cute::make_shape(maximum,c.k,c.experts));
    auto sb=cutlass::make_cute_packed_stride(typename K::StrideB{},cute::make_shape(c.n,c.k,c.experts));
    auto ss=cutlass::make_cute_packed_stride(typename T::Mainloop::StrideScale{},
                                          cute::make_shape(c.n,c.k/c.group_size,c.experts));
    moe_grouped_ppu::GroupProblemShape problem;
    problem.num_groups=c.experts; problem.problem_shapes=ds; problem.host_problem_shapes=nullptr;
    typename G::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,problem,
      {static_cast<GroupCompute const*>(c.a),sa,static_cast<typename T::Low const*>(c.low),sb,
       static_cast<GroupMetadata const*>(c.metadata),ss,c.group_size,
       static_cast<GroupMetadata const*>(c.zero),c.offsets_device},
      {{},static_cast<Output const**>(nullptr),typename T::Epilogue::StrideC{},dp,dd},
      cutlass::KernelHardwareInfo{c.device,c.compute_units}};
    args.representative_m=maximum; args.representative_n=c.n; args.representative_k=c.k;
    if constexpr(!std::is_void_v<typename T::High>)
      args.mainloop.ptr_B2=static_cast<typename T::High const*>(c.high);
    quactlize::moe_directory::View directory{};
    if constexpr(Directory) {
      directory=quactlize::moe_directory::make_bounded_view(base,head,c.m,maximum,c.experts,tm);
      args.directory_header=directory.header; args.directory_entries=directory.entries;
      args.logical_work_upper=uint64_t(directory.capacity)*uint64_t((c.n+tn-1)/tn)*recipe.split;
      args.ctas_per_cu=resources.occupancy; args.grid_ctas_override=recipe.grid; args.splitk=recipe.split;
    } else {
      args.group_M=rows; args.mtiles_uniform=(maximum+tm-1)/tm; args.splitk=recipe.split;
    }
    // All rejection checks precede the first launch. The selected grouped
    // kernels require no private initialized workspace (the directory is ours).
    if(G::can_implement(args)!=cutlass::Status::kSuccess || G::get_workspace_size(args)>head)
      return QK_UNSUPPORTED;
    typename Bound::Reduction reduction;
    if constexpr(Split) {
      typename Bound::Reduction::Arguments ra{c.m,c.n,recipe.split,partials,
          layout.partial_bytes,static_cast<GroupCompute*>(c.output),c.n};
      if(mode==QK_GROUPED_COMPLETE && reduction.initialize(ra)!=cutlass::Status::kSuccess)
        return QK_INVALID;
    }
    auto params=K::to_underlying_arguments(args,base);
    if(stage<0 || stage==0) {
      if constexpr(Split)
        grouped_splitk_device_metadata<<<(c.experts+127)/128,128,0,stream>>>(c.offsets_device,
            partials,ds,dp,dd,rows,c.m,c.n,c.k,c.experts,recipe.split);
      else
        grouped_device_metadata<<<(c.experts+127)/128,128,0,stream>>>(c.offsets_device,
            static_cast<GroupCompute*>(c.output),ds,dp,dd,rows,c.n,c.k,c.experts);
      if(hggcGetLastError()!=hggcSuccess) return QK_RUNTIME_ERROR;
    }
    if constexpr(Directory)
      if((stage<0 || stage==1) &&
         !quactlize::moe_directory::launch_build<tm>(rows,c.offsets_device,maximum,c.experts,directory,stream))
        return QK_RUNTIME_ERROR;
    if((stage<0 || stage==2) && G::run(params,stream)!=cutlass::Status::kSuccess) return QK_RUNTIME_ERROR;
    if constexpr(Split)
      if((stage<0 || stage==3) && mode==QK_GROUPED_COMPLETE && reduction.run(stream)!=cutlass::Status::kSuccess)
        return QK_RUNTIME_ERROR;
    return QK_OK;
  }
};
} // namespace quactlize::runtime

extern "C" int quactlize_kpack_grouped_prepare_dynamic_v1(qk_compute_device_call_v4 const* d,
    qk_recipe_v1 const* r,void** handle) {
  using namespace quactlize::runtime;
  if(!handle) return QK_INVALID;
  *handle=nullptr;
  if(!qk_compute_matches(d) || !address_free(d->device_call.call)) return QK_INVALID;
  qk_resources_v1 resources{};
  int rc=qk_grouped_query(&d->device_call,r,&resources); if(rc) return rc;
  hggcStreamCaptureStatus capture=hggcStreamCaptureStatusNone;
  if(hggcStreamIsCapturing(static_cast<hggcStream_t>(d->device_call.call.stream),&capture)!=hggcSuccess)
    return QK_RUNTIME_ERROR;
  if(capture!=hggcStreamCaptureStatusNone) return QK_UNSUPPORTED;
  if constexpr(grouped) {
    try {
      auto make=[&](auto* type) {
        using H=std::remove_pointer_t<decltype(type)>;
        auto p=std::make_unique<H>();
        int status=p->prepare(*d,*r,resources);
        if(status==QK_OK) *handle=p.release();
        return status;
      };
      if(r->algorithm==QK_PERSISTENT)
        return r->split>1?make(static_cast<DynamicGroupedHandle<true,true,false>*>(nullptr)):
                         make(static_cast<DynamicGroupedHandle<true,false,false>*>(nullptr));
      if(compact_device_call(d->device_call.max_rows,d->device_call.call.experts))
        return r->split>1?make(static_cast<DynamicGroupedHandle<false,true,true>*>(nullptr)):
                         make(static_cast<DynamicGroupedHandle<false,false,true>*>(nullptr));
      return r->split>1?make(static_cast<DynamicGroupedHandle<false,true,false>*>(nullptr)):
                       make(static_cast<DynamicGroupedHandle<false,false,false>*>(nullptr));
    } catch(...) { return QK_RUNTIME_ERROR; }
  }
  return QK_UNSUPPORTED;
}
extern "C" int quactlize_kpack_grouped_run_dynamic_v1(void* handle,qk_compute_device_call_v4 const* d,int32_t mode) {
  if(!handle || !d) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle const*>(handle)->run_grouped_dynamic(*d,mode);
}
extern "C" int quactlize_kpack_grouped_output_dynamic_v1(void* handle,qk_compute_device_call_v4 const* d,
    int32_t mode,qk_grouped_result_v1* out) {
  if(out) *out={};
  if(!handle || !d || !out) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle const*>(handle)->grouped_output(*d,mode,*out);
}
#ifdef QK_GROUPED_DYNAMIC_TESTING
extern "C" int quactlize_kpack_test_grouped_dynamic_stage(void* h,qk_compute_device_call_v4 const* d,int mode,int stage) {
  if(!h || !d) return QK_INVALID;
  return static_cast<quactlize::runtime::Handle const*>(h)->grouped_stage(*d,mode,stage);
}
#endif
