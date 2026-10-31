#include "quactlize/runtime/grouped_dynamic.hpp"
#include "quactlize/moe/grouped.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>

using namespace quactlize::runtime;
static void* ptr(uintptr_t n) { return reinterpret_cast<void*>(n); }
int main() {
  static_assert(sizeof(qk_grouped_result_v1)==48);
  static_assert(offsetof(qk_grouped_result_v1,partial_stride)==40);
  for(int tokens:{1,8,9,16,128,2048,5120,5121,65537}) for(int split:{1,2,4,8}) {
    qk_compute_device_call_v4 shape{};
    shape.version=4;shape.size=sizeof(shape);shape.compute_type=QK_COMPUTE_BF16;shape.metadata_type=1;
    auto& d=shape.device_call;d.version=2;d.size=sizeof(d);d.max_rows=tokens;
    auto& c=d.call;c.version=1;c.size=sizeof(c);c.m=tokens*8;c.n=3072;c.k=512;
    c.experts=256;c.group_size=32;c.device=1;c.compute_units=72;c.mapping_id=42;
    assert(address_free(c));
    auto call=shape;
    auto& b=call.device_call.call;
    GroupedWorkspace layout;
    assert(grouped_workspace(256,split,c.m,c.n,4096,12,8,true,layout));
    constexpr uintptr_t step=UINT64_C(1)<<40;
    b.a=ptr(step);b.low=ptr(2*step);b.high=ptr(3*step);b.metadata=ptr(4*step);
    b.output=ptr(5*step);b.offsets_device=static_cast<int32_t*>(ptr(6*step));
    b.workspace=ptr(7*step);b.workspace_bytes=layout.total;b.stream=ptr(123);
    assert(same_grouped_shape(call,shape));
    assert(!address_free(b));
    assert(grouped_buffers(b,layout.total,true,false,1024,256,512));
    for(int mode:{QK_GROUPED_COMPLETE,QK_GROUPED_PRODUCER}) {
      qk_grouped_result_v1 out{};
      assert(grouped_result(call,split,mode,layout,out)==0);
      bool partial=mode==QK_GROUPED_PRODUCER && split>1;
      assert(out.version==1 && out.size==sizeof(out) && out.reserved==0);
      assert(out.compute_type==QK_COMPUTE_BF16 && out.row_stride==3072);
      assert(out.partial_count==(partial?split:1));
      assert(out.dtype==(partial?QK_RESULT_F32:QK_RESULT_BF16));
      assert(out.partial_stride==(partial?int64_t(c.m)*c.n:0));
      assert(out.data==(partial?static_cast<char*>(b.workspace)+layout.partials:b.output));
      qzm_value_v1 value{};
      assert(quactlize::moe::grouped_value(out,value)==0);
      assert(value.state==(partial?QZM_PARTIALS:QZM_PROJECTED));
      assert(value.dtype==(partial?QZM_F32:QZM_BF16) && value.round_type==QZM_BF16);
      out.reserved=1;assert(quactlize::moe::grouped_value(out,value)==QZR_INVALID && !value.data);
    }
    qk_grouped_result_v1 out{};out.data=b.output;
    assert(grouped_result(call,split,99,layout,out)==QK_INVALID && out.data==nullptr);
    auto bad=call;bad.compute_type=0;assert(!same_grouped_shape(bad,shape));
    bad=call;bad.metadata_type=0;assert(!same_grouped_shape(bad,shape));
    bad=call;bad.device_call.max_rows++;assert(!same_grouped_shape(bad,shape));
    bad=call;bad.device_call.call.device++;assert(!same_grouped_shape(bad,shape));
    bad=call;bad.device_call.call.rows_host=static_cast<int*>(ptr(1));assert(!same_grouped_shape(bad,shape));
    bad=call;bad.device_call.call.rows_device=static_cast<int*>(ptr(1));assert(!same_grouped_shape(bad,shape));
    auto no=b;no.workspace_bytes--;assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.output=const_cast<void*>(b.a);assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.workspace=const_cast<void*>(b.metadata);assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.output=static_cast<char*>(b.workspace)+16;assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.offsets_device=nullptr;assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.high=nullptr;assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.metadata=ptr(UINTPTR_MAX-15);assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
    no=b;no.output=ptr(step+2);assert(!grouped_buffers(no,layout.total,true,false,1024,256,512));
  }
  std::cout<<"GROUPED_DYNAMIC_HOST PASS arbitrary_M shape precision partials live_alias\n";
}
