#pragma once
#include "api.h"
#include "types.cuh"
#include "reducer.cuh"
#include <memory>
#include <cstring>

namespace quactlize::decode {
#ifndef QKD_USE_BF16_COMPUTE
#define QKD_USE_BF16_COMPUTE 0
#endif
static_assert(QKD_USE_BF16_COMPUTE==0 || QKD_USE_BF16_COMPUTE==1);
constexpr int compute_type=QKD_USE_BF16_COMPUTE ? QKD_COMPUTE_BF16 : QKD_COMPUTE_F16;
using Compute=std::conditional_t<QKD_USE_BF16_COMPUTE,cutlass::bfloat16_t,cutlass::half_t>;
using Core=runtime::DenseTypes<QK_QTYPE,QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,QK_AP,QK_DN,Compute>;
using F=runtime::Format<QK_QTYPE>;
constexpr bool packed=QK_ROUTE==QK_DENSE_FQ;
constexpr uint64_t mapping=QK_QTYPE==8?q8_kpack2::kMappingId:QK_QTYPE==12?
    UINT64_C(0x51344b5034540001):UINT64_C(0x514b504b54000001);
static_assert(QK_ROUTE==QK_DENSE_FQ || QK_ROUTE==QK_DENSE_SF);

inline int validate(qkd_dense_call_v1 const& d,qk_recipe_v1 const& r) {
  auto const& c=d.call;
  if (d.version!=1 || d.size!=sizeof(d) || c.version!=1 || c.size!=sizeof(c) ||
      d.input_type!=d.output_type || (d.input_type!=QKD_F32 && d.input_type!=QKD_BF16) ||
      c.m<1 || c.m>8 || c.experts!=1 || c.n<=0 || c.n%256 || c.k<=0 ||
      c.k%((QK_QTYPE==11||QK_QTYPE==14)?512:256) || c.mapping_id!=mapping ||
      c.group_size!=F::spec.group_size || c.rows_host || c.rows_device || c.offsets_device ||
      r.version!=1 || r.size!=sizeof(r) || r.algorithm<0 || r.algorithm>1 ||
      (r.split!=1 && r.split!=2 && r.split!=4 && r.split!=8) ||
      (r.algorithm==QK_PERSISTENT ? (r.grid<=0 || r.split!=1) : r.grid!=0) ||
      c.compute_units<=0 || c.n>INT32_MAX/8 || c.k>INT32_MAX/8) return QK_INVALID;
  if (c.k%(QK_TK*r.split) || c.k/(QK_TK*r.split)<QK_STAGES-1 ||
      (QK_AP && c.m!=1) || (packed && r.algorithm!=QK_ORDINARY)) return QK_UNSUPPORTED;
  return QK_OK;
}

struct Handle {
  virtual int run(hggcStream_t)=0;
  virtual ~Handle()=default;
};
template<class Scalar> struct TypedHandle:Handle {
  using T=DenseTypes<Core,Scalar,Scalar>;
  using Shipping=typename T::Shipping::Gemm;
  using Split=typename T::Split::Gemm;
  using Persistent=typename T::PersistentGemm;
  Shipping shipping;Split split;Persistent persistent;
  Scalar* output=nullptr;float* partials=nullptr;
  int slices=1,count=0;bool use_persistent=false;

  static int query(qkd_dense_call_v1 const& d,qk_recipe_v1 const& r,qk_resources_v1& out) {
    out={1,sizeof(out),0,0,0,0,0};
    if(r.split>1) {
      out.workspace_bytes=uint64_t(d.call.m)*d.call.n*r.split*4;
      out.shared_bytes=Split::GemmKernel::SharedStorageSize;
      out.occupancy=Split::maximum_active_blocks();
    } else if(r.algorithm==QK_PERSISTENT) {
      out.shared_bytes=Persistent::GemmKernel::SharedStorageSize;
      out.occupancy=Persistent::maximum_active_blocks();
    } else {
      out.shared_bytes=Shipping::GemmKernel::SharedStorageSize;
      out.occupancy=Shipping::maximum_active_blocks();
    }
    if(!ppu_tactics::fits_block_smem(out.shared_bytes))return QK_UNSUPPORTED;
    if(out.occupancy<=0)return QK_RUNTIME_ERROR;
    if(r.algorithm==QK_PERSISTENT && uint64_t(r.grid)>uint64_t(d.call.compute_units)*out.occupancy)return QK_UNSUPPORTED;
    return QK_OK;
  }

  int prepare(qkd_dense_call_v1 const& d,qk_recipe_v1 const& r,int occupancy) {
    auto const& c=d.call;auto stream=static_cast<hggcStream_t>(c.stream);
    typename T::Mainloop::Arguments ml{};
    ml.ptr_A=static_cast<Scalar const*>(c.a);
    ml.dA=cutlass::make_cute_packed_stride(typename T::Mainloop::StrideA{},cute::make_shape(c.m,c.k,1));
    ml.ptr_B=static_cast<typename Core::Low const*>(c.low);
    ml.dB=cutlass::make_cute_packed_stride(typename T::Mainloop::StrideB{},cute::make_shape(c.n,c.k,1));
    ml.ptr_S=static_cast<typename T::Mainloop::ElementScale const*>(c.metadata);
    ml.dS=cutlass::make_cute_packed_stride(typename T::Mainloop::StrideScale{},cute::make_shape(c.n,c.k/c.group_size,1));
    ml.group_size=c.group_size;ml.ptr_Z=static_cast<typename T::Mainloop::ElementZero const*>(c.zero);
    if constexpr(!std::is_void_v<typename Core::High>) ml.ptr_B2=static_cast<typename Core::High const*>(c.high);
    output=static_cast<Scalar*>(c.output);slices=r.split;count=c.m*c.n;
    if(r.split>1) {
      partials=static_cast<float*>(c.workspace);
      auto stride=cutlass::gemm::kernel::detail::make_compact_fp32_partial_stride<typename Split::GemmKernel::StrideD>(c.m,c.n);
      typename Split::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
          {partials,stride,partials,stride},r.split};
      if(Split::can_implement(args)!=cutlass::Status::kSuccess)return QK_UNSUPPORTED;
      return split.initialize(args,nullptr,stream)==cutlass::Status::kSuccess?QK_OK:QK_INITIALIZE_ERROR;
    }
    auto sc=cutlass::make_cute_packed_stride(typename Shipping::GemmKernel::StrideC{},cute::make_shape(c.m,c.n,1));
    auto sd=cutlass::make_cute_packed_stride(typename Shipping::GemmKernel::StrideD{},cute::make_shape(c.m,c.n,1));
    if(r.algorithm==QK_PERSISTENT) {
      typename Persistent::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
          {{1.f,0.f},static_cast<Scalar*>(nullptr),sc,output,sd},
          cutlass::KernelHardwareInfo{c.device,c.compute_units},{},occupancy,uint32_t(r.grid)};
      if(Persistent::can_implement(args)!=cutlass::Status::kSuccess)return QK_UNSUPPORTED;
      use_persistent=true;
      return persistent.initialize(args,nullptr,stream)==cutlass::Status::kSuccess?QK_OK:QK_INITIALIZE_ERROR;
    }
    typename Shipping::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{c.m,c.n,c.k,1},ml,
        {{1.f,0.f},static_cast<Scalar*>(nullptr),sc,output,sd},1};
    if(Shipping::can_implement(args)!=cutlass::Status::kSuccess)return QK_UNSUPPORTED;
    return shipping.initialize(args,nullptr,stream)==cutlass::Status::kSuccess?QK_OK:QK_INITIALIZE_ERROR;
  }

  int run(hggcStream_t stream) override {
    if(hggcGetLastError()!=hggcSuccess)return QK_RUNTIME_ERROR;
    auto rc=slices>1?split.run(stream):use_persistent?persistent.run(stream):shipping.run(stream);
    if(rc!=cutlass::Status::kSuccess)return QK_RUNTIME_ERROR;
#define QKD_REDUCE(S) case S: reduce_decode<S><<<(count+63)/64,32,0,stream>>>(partials,output,count); break
    switch(slices) { QKD_REDUCE(2);QKD_REDUCE(4);QKD_REDUCE(8);default:break; }
#undef QKD_REDUCE
    return hggcGetLastError()==hggcSuccess?QK_OK:QK_RUNTIME_ERROR;
  }
};
} // namespace quactlize::decode

extern "C" qk_identity_v1 const* quactlize_kpack_decode_dense_identity_v1() {
  static qk_identity_v1 const identity{1,sizeof(qk_identity_v1),QK_QTYPE,QK_ROUTE,
      QK_TM,QK_TN,QK_TK,QK_WM,QK_WN,QK_STAGES,QK_AP,QK_DN,
      quactlize::decode::mapping,QK_PARENT,QK_BUILD_KEY};
  return &identity;
}
extern "C" int quactlize_kpack_decode_dense_device_v1(char* name,int capacity,int32_t* device,int32_t* cu) {
  if (!name || capacity<=0 || !device || !cu) return QK_INVALID;
  hggcDeviceProp prop{};
  if (hggcGetDevice(device)!=hggcSuccess || hggcGetDeviceProperties(&prop,*device)!=hggcSuccess)return QK_RUNTIME_ERROR;
  if (std::strlen(prop.name)>=size_t(capacity))return QK_INVALID;
  *cu=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(*device);
  if (*cu<=0) return QK_RUNTIME_ERROR;
  std::strcpy(name,prop.name);return QK_OK;
}
static int qkd_query(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
  using namespace quactlize::decode;
  if(!d||!r||!out)return QK_INVALID;
  int rc=validate(*d,*r);if(rc)return rc;
  int device=-1;
  if(hggcGetDevice(&device)!=hggcSuccess)return QK_RUNTIME_ERROR;
  if(device!=d->call.device || cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device)!=d->call.compute_units)return QK_INVALID;
  return d->input_type==QKD_F32?TypedHandle<float>::query(*d,*r,*out):TypedHandle<cutlass::bfloat16_t>::query(*d,*r,*out);
}
static int qkd_prepare(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,void** handle) {
  using namespace quactlize::decode;
  if(!handle)return QK_INVALID;*handle=nullptr;
  qk_resources_v1 res{};int rc=qkd_query(d,r,&res);if(rc)return rc;
  auto const& c=d->call;
  if(!c.a||!c.low||!c.metadata||!c.output||((F::spec.high_bits!=0)!=(c.high!=nullptr))||
      ((!packed && QK_QTYPE!=8)?c.zero==nullptr:c.zero!=nullptr)||
      (uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.high)|uintptr_t(c.metadata)|uintptr_t(c.zero)|uintptr_t(c.output))%16 ||
      c.workspace_bytes<res.workspace_bytes || (res.workspace_bytes && (!c.workspace || uintptr_t(c.workspace)%16)))return QK_INVALID;
  hggcStreamCaptureStatus capture=hggcStreamCaptureStatusNone;
  if (hggcStreamIsCapturing(static_cast<hggcStream_t>(c.stream),&capture)!=hggcSuccess)return QK_RUNTIME_ERROR;
  if (capture!=hggcStreamCaptureStatusNone)return QK_UNSUPPORTED;
  try {
    std::unique_ptr<Handle> h;
    if(d->input_type==QKD_F32) {auto p=std::make_unique<TypedHandle<float>>();rc=p->prepare(*d,*r,res.occupancy);h=std::move(p);}
    else {auto p=std::make_unique<TypedHandle<cutlass::bfloat16_t>>();rc=p->prepare(*d,*r,res.occupancy);h=std::move(p);}
    if(rc)return rc;*handle=h.release();return QK_OK;
  } catch(...) {return QK_RUNTIME_ERROR;}
}
extern "C" int quactlize_kpack_decode_dense_query_v1(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
  if constexpr(QKD_USE_BF16_COMPUTE) return QK_UNSUPPORTED;
  return qkd_query(d,r,out);
}
extern "C" int quactlize_kpack_decode_dense_prepare_v1(qkd_dense_call_v1 const* d,qk_recipe_v1 const* r,void** handle) {
  if constexpr(QKD_USE_BF16_COMPUTE) {if(handle)*handle=nullptr;return QK_UNSUPPORTED;}
  return qkd_prepare(d,r,handle);
}
extern "C" qkd_compute_identity_v2 const* quactlize_kpack_decode_dense_identity_v2() {
  static qkd_compute_identity_v2 const identity{2,sizeof(identity),
      quactlize_kpack_decode_dense_identity_v1(),quactlize::decode::compute_type};
  return &identity;
}
static bool qkd_compute_matches(qkd_dense_call_v2 const* d) {
  return d && d->version==2 && d->size==sizeof(*d) &&
      d->compute_type==quactlize::decode::compute_type;
}
extern "C" int quactlize_kpack_decode_dense_query_v2(qkd_dense_call_v2 const* d,qk_recipe_v1 const* r,qk_resources_v1* out) {
  if(!qkd_compute_matches(d)) return QK_INVALID;
  return qkd_query(&d->dense,r,out);
}
extern "C" int quactlize_kpack_decode_dense_prepare_v2(qkd_dense_call_v2 const* d,qk_recipe_v1 const* r,void** handle) {
  if(handle)*handle=nullptr;
  if(!qkd_compute_matches(d)) return QK_INVALID;
  return qkd_prepare(&d->dense,r,handle);
}
extern "C" int quactlize_kpack_decode_dense_run_v1(void* h,void* stream) {
  return h?static_cast<quactlize::decode::Handle*>(h)->run(static_cast<hggcStream_t>(stream)):QK_INVALID;
}
extern "C" void quactlize_kpack_decode_dense_destroy_v1(void* h) {delete static_cast<quactlize::decode::Handle*>(h);}
