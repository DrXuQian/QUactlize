#pragma once
#include "api.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace q17_reference {
inline thread_local std::string last_error;
inline void checked(cudaError_t code, char const *where) {
  if (code != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}
struct Plan {
  void *workspace = nullptr;
  size_t bytes = 0;
  bool owns_workspace = true;
  virtual ~Plan() { if (workspace && owns_workspace) cudaFree(workspace); }
  virtual void execute() = 0;
  void allocate(size_t count) {
    bytes = count;
    if (bytes) checked(cudaMalloc(&workspace, bytes), "workspace allocation");
  }
};
inline void validate(q17_ref_call const &c) {
  if ((c.dtype != 0 && c.dtype != 1) || c.m < 1 || c.n < 128 || c.n % 128 ||
      c.k < 256 || c.k % 256 || !c.a || !c.b || !c.scales || !c.zeros || !c.output)
    throw std::runtime_error("invalid common W4A16 group64 fixture");
  int device = 0;
  checked(cudaGetDevice(&device), "current device");
  cudaDeviceProp properties{};
  checked(cudaGetDeviceProperties(&properties, device), "device properties");
  if (properties.major != 9 || properties.minor != 0)
    throw std::runtime_error("this experiment requires physical SM90");
}
template<class F> int guard(F &&f) {
  try { f(); last_error.clear(); return 0; }
  catch (std::exception const &error) { last_error = error.what(); return 1; }
}
}

Q17_REF_EXPORT char const *q17_ref_error() { return q17_reference::last_error.c_str(); }
Q17_REF_EXPORT int q17_ref_execute(void *plan) {
  return q17_reference::guard([&] {
    if (!plan) throw std::runtime_error("null reference plan");
    static_cast<q17_reference::Plan *>(plan)->execute();
  });
}
Q17_REF_EXPORT void q17_ref_destroy(void *plan) { delete static_cast<q17_reference::Plan *>(plan); }
Q17_REF_EXPORT size_t q17_ref_workspace(void *plan) {
  return plan ? static_cast<q17_reference::Plan *>(plan)->bytes : 0;
}
Q17_REF_EXPORT int q17_ref_device(int *l2_bytes, int *sm_count) {
  return q17_reference::guard([&] {
    if (!l2_bytes || !sm_count) throw std::runtime_error("null device output");
    int device=0;
    q17_reference::checked(cudaGetDevice(&device), "device");
    q17_reference::checked(cudaDeviceGetAttribute(l2_bytes,cudaDevAttrL2CacheSize,device), "L2");
    q17_reference::checked(cudaDeviceGetAttribute(sm_count,cudaDevAttrMultiProcessorCount,device), "SM count");
    if (*l2_bytes<=0 || *sm_count<=0) throw std::runtime_error("missing device geometry");
  });
}
Q17_REF_EXPORT int q17_ref_share_workspace(void *plan, void *owner) {
  return q17_reference::guard([&] {
    if (!plan || !owner || plan==owner) throw std::runtime_error("invalid workspace sharing");
    auto &dst=*static_cast<q17_reference::Plan *>(plan);
    auto &src=*static_cast<q17_reference::Plan *>(owner);
    if (dst.bytes!=src.bytes || !src.owns_workspace)
      throw std::runtime_error("workspace sizes/ownership differ");
    if (dst.owns_workspace && dst.workspace)
      q17_reference::checked(cudaFree(dst.workspace),"release redundant scratch");
    dst.workspace=src.workspace;
    dst.owns_workspace=false;
  });
}
