#pragma once
#include "dynamic.hpp"
#include "grouped_workspace.hpp"

namespace quactlize::runtime {
inline bool same_grouped_shape(qk_compute_device_call_v4 const& a,
                               qk_compute_device_call_v4 const& b) {
  auto const& x=a.device_call; auto const& y=b.device_call;
  auto const& c=x.call; auto const& d=y.call;
  return a.version==4 && a.size==sizeof(a) && a.compute_type==b.compute_type &&
      a.metadata_type==b.metadata_type && x.version==2 && x.size==sizeof(x) && !x.reserved &&
      x.max_rows==y.max_rows && c.version==1 && c.size==sizeof(c) &&
      c.m==d.m && c.n==d.n && c.k==d.k && c.experts==d.experts &&
      c.group_size==d.group_size && c.device==d.device && c.compute_units==d.compute_units &&
      c.mapping_id==d.mapping_id && !c.rows_host && !c.rows_device;
}

// Extents come from the validated shape/format, never device data. This guards
// live writes against every readable plane, including reused changing IDs.
inline bool grouped_buffers(qk_call_v1 const& c,uint64_t scratch,bool high,bool zero,
                            uint64_t low_bytes,uint64_t high_bytes,uint64_t meta_bytes) {
  if(!dense_buffers(c,scratch,high,zero) || !c.offsets_device ||
      (uintptr_t(c.offsets_device)&3)) return false;
  void const* ptrs[]={c.a,c.low,c.high,c.metadata,c.zero,c.offsets_device,c.output,c.workspace};
  uint64_t sizes[]={uint64_t(c.m)*c.k*2,low_bytes,high_bytes,meta_bytes,zero?meta_bytes:0,
      uint64_t(c.experts+1)*4,uint64_t(c.m)*c.n*2,scratch};
  for(int i=0;i<8;++i) {
    if(!sizes[i]) continue;
    uintptr_t p=uintptr_t(ptrs[i]);
    if(!p || sizes[i]>uint64_t(INT64_MAX) || sizes[i]>UINTPTR_MAX-p) return false;
    for(int j=0;j<i;++j) {
      if(!sizes[j] || (i<6 && j<6)) continue;
      uintptr_t q=uintptr_t(ptrs[j]);
      if(p<q+sizes[j] && q<p+sizes[i]) return false;
    }
  }
  return true;
}

inline int grouped_result(qk_compute_device_call_v4 const& d,int split,int mode,
                          GroupedWorkspace const& layout,qk_grouped_result_v1& out) {
  out={};
  if(mode!=QK_GROUPED_COMPLETE && mode!=QK_GROUPED_PRODUCER) return QK_INVALID;
  auto const& c=d.device_call.call;
  bool partial=mode==QK_GROUPED_PRODUCER && split>1;
  out={1,sizeof(out),partial?static_cast<char*>(c.workspace)+layout.partials:c.output,
       partial?QK_RESULT_F32:d.compute_type==QK_COMPUTE_BF16?QK_RESULT_BF16:QK_RESULT_F16,
       partial?split:1,d.compute_type,0,c.n,partial?int64_t(c.m)*c.n:0};
  return QK_OK;
}
} // namespace quactlize::runtime
