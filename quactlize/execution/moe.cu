#include <hggc_runtime.h>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"
namespace quactlize::runtime { using Half=cutlass::half_t; }
#include "../runtime/indexed.cuh"
#include "../runtime/moe_chain.cuh"
#include "moe.h"
#include <algorithm>
#include <cstring>

namespace {
using Shape=cute::Shape<int,int,int>;
using Stride=cute::Stride<int64_t,cute::_1,cute::_0>;
namespace md=quactlize::moe_directory;
struct Layout {
  uint64_t a,output,offsets,row_ids,workspace,header,entries,shapes,outputs,strides,rows,total;
};
bool layout(qkg_call_v1 const& c,Layout& l) {
  if (c.version!=1 || c.size!=sizeof(c) || c.mode!=QKG_INDEXED || c.input_type!=QKG_F32 ||
      c.experts!=256 || c.topk!=8 || c.rows<8 || c.rows>64 || c.rows%8 ||
      (c.channels!=1 && c.channels!=8) || c.n<=0 || c.k<=0 || c.n%256 || c.k%256 ||
      c.n>INT32_MAX-8192 || c.k>INT32_MAX-8192 || c.a_row_stride<c.k ||
      c.a_row_stride>INT64_MAX/8 || c.a_token_stride<c.channels*c.a_row_stride ||
      c.a_token_stride>INT64_MAX/32 || c.out_row_stride<c.n ||
      c.out_row_stride>INT64_MAX/256 || c.ids_stride<8 || c.ids_stride>INT64_MAX/32) return false;
  uint64_t end=0;
  auto take=[&](uint64_t bytes) { uint64_t begin=end; end+=(bytes+255)&~UINT64_C(255); return begin; };
  l.a=take(uint64_t(c.rows)*c.k*4);l.output=take(uint64_t(c.rows)*c.n*4);
  l.offsets=take(257*4);l.row_ids=take(uint64_t(c.rows)*4);l.workspace=end;
  l.header=take(sizeof(md::Header));l.entries=take(uint64_t(c.rows)*sizeof(md::BlockEntry));
  l.shapes=take(256*sizeof(Shape));l.outputs=take(256*sizeof(void*));
  l.strides=take(256*sizeof(Stride));l.rows=take(256*4);l.total=end;
  return true;
}
void tuple_abi(qk_moe_projection_v1& p) {
  Shape s{};Stride d{};
  auto off=[](auto const& tuple,auto const& v) { return uint32_t(reinterpret_cast<char const*>(&v)-reinterpret_cast<char const*>(&tuple)); };
  p.shape_size=sizeof(s);p.stride_size=sizeof(d);
  p.shape_offsets[0]=off(s,cute::get<0>(s));p.shape_offsets[1]=off(s,cute::get<1>(s));
  p.shape_offsets[2]=off(s,cute::get<2>(s));p.stride_offset=off(d,cute::get<0>(d));
}
}
extern "C" int quactlize_kpack_moe_simt_query_v1(qkg_call_v1 const* call,uint64_t* bytes) {
  if (!call || !bytes) return QKG_INVALID;
  *bytes=0;Layout l{};
  if (!layout(*call,l)) return QKG_SHAPE;
  *bytes=l.total;return QKG_OK;
}
extern "C" int quactlize_kpack_moe_simt_bind_v1(qkg_call_v1 const* call,int device,void* scratch,
    uint64_t bytes,qk_moe_projection_v1* out) {
  if (!out) return QKG_INVALID;
  *out={};Layout l{};
  if (!call || !layout(*call,l) || !scratch || uintptr_t(scratch)%256 ||
      bytes<l.total || uintptr_t(scratch)>UINTPTR_MAX-l.total || !call->a || !call->output || !call->ids)
    return QKG_INVALID;
  int current=-1,cu=0;hggcDeviceProp prop{};hggcStreamCaptureStatus capture{};
  if (hggcGetDevice(&current)!=hggcSuccess || hggcGetDeviceProperties(&prop,current)!=hggcSuccess ||
      hggcDeviceGetAttribute(&cu,hggcDevAttrMultiProcessorCount,current)!=hggcSuccess ||
      hggcStreamIsCapturing(static_cast<hggcStream_t>(call->stream),&capture)!=hggcSuccess) return QKG_RUNTIME;
  if ((device>=0 && current!=device) || std::strcmp(prop.name,"PPU-ZW810") || cu!=72 ||
      capture!=hggcStreamCaptureStatusNone) return QKG_SHAPE;
  device=current;
  auto const& c=*call;auto base=static_cast<char*>(scratch);
  auto& p=*out;p.version=1;p.size=sizeof(p);tuple_abi(p);
  p.m=c.rows;p.n=c.n;p.k=c.k;p.experts=c.experts;p.tile_m=8;p.splits=1;p.device=device;
  p.a=base+l.a;p.output=base+l.output;p.offsets=reinterpret_cast<int*>(base+l.offsets);
  p.shapes=base+l.shapes;p.outputs=base+l.outputs;p.strides=base+l.strides;
  p.rows=reinterpret_cast<int*>(base+l.rows);p.directory_header=base+l.header;
  p.directory_entries=base+l.entries;p.directory_capacity=c.rows;
  p.workspace=base+l.workspace;p.workspace_bytes=l.total-l.workspace;
  p.io={1,sizeof(p.io),c.rows/8,8,c.channels,0,c.ids_stride,c.a_row_stride,c.a_token_stride,
    c.out_row_stride,c.ids,static_cast<float const*>(c.a),c.output,reinterpret_cast<int*>(base+l.row_ids)};
  return QKG_OK;
}
extern "C" int quactlize_kpack_moe_mixed_stage_v1(qk_moe_plan_v1 const* source,uint32_t mask,int phase,void* opaque) {
  using namespace quactlize::runtime;
  if (!source || source->version!=1 || source->size!=sizeof(*source) || !mask || mask>7 ||
      (source->merged && (mask&2)) || source->gate.experts!=256 || source->gate.m<8 || source->gate.m>64)
    return QKG_INVALID;
  MixedMoePlan plan;static_cast<qk_moe_plan_v1&>(plan)=*source;plan.simt_mask=mask;
  auto stream=static_cast<hggcStream_t>(opaque);
  if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
  if (phase==QK_MOE_PREPARE) {
    if (moe_prepare_m1_supported(plan)) moe_chain_prepare_m1<Shape,Stride><<<1,256,0,stream>>>(plan);
    else if (plan.gate.m>32) moe_chain_prepare<Shape,Stride,64><<<moe_prepare_blocks(256,plan.gate.m),256,0,stream>>>(plan);
    else moe_chain_prepare<Shape,Stride><<<moe_prepare_blocks(256,plan.gate.m),256,0,stream>>>(plan);
  } else if (phase==QK_MOE_ACTIVATE) {
    moe_chain_swiglu_mixed<<<dim3(std::min((plan.down.k+255)/256,32),plan.gate.m),256,0,stream>>>(plan);
  } else return QKG_INVALID;
  return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}

extern "C" int quactlize_kpack_moe_weighted_finish_v1(qk_moe_plan_v1 const* plan,uint32_t mask,
    qk_llama_moe_finish_v1 const* finish,void* opaque) {
  using namespace quactlize::runtime;
  if (!plan || !finish || plan->version!=1 || plan->size!=sizeof(*plan) || mask>7 ||
      finish->version!=1 || finish->size!=sizeof(*finish) || !finish->weights || !finish->output ||
      plan->down.io.tokens<1 || plan->down.io.tokens>8 || plan->down.io.topk!=8 ||
      plan->down.m!=plan->down.io.tokens*8 || plan->down.n<=0 ||
      finish->weights_stride<8 || finish->output_stride<plan->down.n) return QKG_INVALID;
  auto stream=static_cast<hggcStream_t>(opaque);
  if (hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
  dim3 grid((plan->down.n+127)/128,plan->down.io.tokens);
  if (mask&4) moe_weighted_finish<true><<<grid,128,0,stream>>>(plan->down,*finish);
  else moe_weighted_finish<false><<<grid,128,0,stream>>>(plan->down,*finish);
  return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
