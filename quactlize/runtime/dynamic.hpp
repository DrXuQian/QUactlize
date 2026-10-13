#pragma once
#include "abi.h"
#include <limits>

namespace quactlize::runtime {
// Host-only checks shared by native and typed dense plans. Shape-only plans
// never retain caller addresses, including an obsolete workspace generation.
inline bool address_free(qk_call_v1 const& c) {
  return !c.a && !c.low && !c.high && !c.metadata && !c.zero && !c.output &&
      !c.rows_host && !c.rows_device && !c.offsets_device && !c.workspace && !c.workspace_bytes;
}
inline bool same_dense_shape(qk_call_v1 const& a,qk_call_v1 const& b) {
  return a.version==1 && a.size==sizeof(a) && a.m==b.m && a.n==b.n && a.k==b.k &&
      a.experts==1 && b.experts==1 && a.group_size==b.group_size &&
      a.device==b.device && a.compute_units==b.compute_units && a.mapping_id==b.mapping_id &&
      !a.rows_host && !a.rows_device && !a.offsets_device;
}
inline bool dense_buffers(qk_call_v1 const& c,uint64_t workspace_bytes,bool high,bool zero) {
  return c.a && c.low && c.metadata && c.output && high==(c.high!=nullptr) && zero==(c.zero!=nullptr) &&
      !((uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.high)|uintptr_t(c.metadata)|
         uintptr_t(c.zero)|uintptr_t(c.output))&15) && c.workspace_bytes>=workspace_bytes &&
      (!workspace_bytes || (c.workspace && !(uintptr_t(c.workspace)&15)));
}
} // namespace quactlize::runtime
