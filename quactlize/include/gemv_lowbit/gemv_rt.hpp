// Minimal device-runtime shim, so ONE source builds and runs both locally under nvcc and on the box.
//
// Runtime failures are latched, never process-fatal.  In particular this header is used by
// libquactlize_ppu.so: an allocation, copy, or synchronize failure must cross its C ABI as a non-zero return
// instead of terminating the process which loaded the library.  Callers start one operation with
// rt_clear_error(), check rt_ok() before launching, and translate rt_status() at their ABI boundary.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(PPU_GEMV_RT_FAKE)
// Host-only fault injection used by l108_rt_error_contract.cpp.  It deliberately has no CUDA dependency.
#elif defined(__HGGCCC__)
#include <hggc_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace ppu_gemv {

enum class RtStatus : int {
  Ok = 0,
  Allocation = 1,
  HostToDevice = 2,
  DeviceToHost = 3,
  Synchronize = 4,
  HostAllocation = 5,
};

// Each public entry is synchronous and owns all of its temporary buffers.  Thread-local state keeps two callers on
// different host threads independent without making the C ABI carry a context object.
inline thread_local RtStatus g_rt_status = RtStatus::Ok;

inline void rt_clear_error() { g_rt_status = RtStatus::Ok; }
inline RtStatus rt_status() { return g_rt_status; }
inline bool rt_ok() { return g_rt_status == RtStatus::Ok; }

inline bool rt_record(RtStatus status, char const* operation, char const* detail) {
  if (g_rt_status == RtStatus::Ok) {
    g_rt_status = status;
    std::fprintf(stderr, "[gemv] %s: %s\n", operation, detail ? detail : "device runtime failure");
  }
  return false;
}

#if defined(PPU_GEMV_RT_FAKE)

inline thread_local RtStatus g_rt_fake_failure = RtStatus::Ok;
inline void rt_test_fail_next(RtStatus status) { g_rt_fake_failure = status; }
inline bool rt_test_should_fail(RtStatus status) {
  if (g_rt_fake_failure != status) return false;
  g_rt_fake_failure = RtStatus::Ok;
  return true;
}

inline void* rt_malloc(size_t bytes) {
  if (rt_test_should_fail(RtStatus::Allocation)) {
    rt_record(RtStatus::Allocation, "allocation", "injected failure");
    return nullptr;
  }
  void* p = std::malloc(bytes);
  if (!p && bytes) rt_record(RtStatus::Allocation, "allocation", "host fake backend returned null");
  return p;
}
inline void rt_free(void* p) { std::free(p); }
inline bool rt_h2d(void* dst, void const* src, size_t bytes) {
  if (rt_test_should_fail(RtStatus::HostToDevice))
    return rt_record(RtStatus::HostToDevice, "H2D", "injected failure");
  if (bytes) std::memcpy(dst, src, bytes);
  return true;
}
inline bool rt_d2h(void* dst, void const* src, size_t bytes) {
  if (rt_test_should_fail(RtStatus::DeviceToHost))
    return rt_record(RtStatus::DeviceToHost, "D2H", "injected failure");
  if (bytes) std::memcpy(dst, src, bytes);
  return true;
}
inline bool rt_sync(char const*) {
  if (rt_test_should_fail(RtStatus::Synchronize))
    return rt_record(RtStatus::Synchronize, "synchronize", "injected failure");
  return true;
}

#elif defined(__HGGCCC__)

inline void* rt_malloc(size_t bytes) {
  void* p = nullptr;
  hggcError_t const e = hggcMalloc(&p, bytes);
  if (e != hggcSuccess) {
    rt_record(RtStatus::Allocation, "hggcMalloc", hggcGetErrorString(e));
    return nullptr;
  }
  return p;
}
inline void rt_free(void* p) { if (p) (void)hggcFree(p); }
inline bool rt_h2d(void* dst, void const* src, size_t bytes) {
  hggcError_t const e = hggcMemcpy(dst, src, bytes, hggcMemcpyHostToDevice);
  return e == hggcSuccess || rt_record(RtStatus::HostToDevice, "H2D", hggcGetErrorString(e));
}
inline bool rt_d2h(void* dst, void const* src, size_t bytes) {
  hggcError_t const e = hggcMemcpy(dst, src, bytes, hggcMemcpyDeviceToHost);
  return e == hggcSuccess || rt_record(RtStatus::DeviceToHost, "D2H", hggcGetErrorString(e));
}
inline bool rt_sync(char const* where) {
  hggcError_t const e = hggcDeviceSynchronize();
  return e == hggcSuccess || rt_record(RtStatus::Synchronize, where, hggcGetErrorString(e));
}

#else

inline void* rt_malloc(size_t bytes) {
  void* p = nullptr;
  cudaError_t const e = cudaMalloc(&p, bytes);
  if (e != cudaSuccess) {
    rt_record(RtStatus::Allocation, "cudaMalloc", cudaGetErrorString(e));
    return nullptr;
  }
  return p;
}
inline void rt_free(void* p) { if (p) (void)cudaFree(p); }
inline bool rt_h2d(void* dst, void const* src, size_t bytes) {
  cudaError_t const e = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
  return e == cudaSuccess || rt_record(RtStatus::HostToDevice, "H2D", cudaGetErrorString(e));
}
inline bool rt_d2h(void* dst, void const* src, size_t bytes) {
  cudaError_t const e = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
  return e == cudaSuccess || rt_record(RtStatus::DeviceToHost, "D2H", cudaGetErrorString(e));
}
inline bool rt_sync(char const* where) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaGetLastError();
  return e == cudaSuccess || rt_record(RtStatus::Synchronize, where, cudaGetErrorString(e));
}

#endif

inline bool rt_memset0(void* p, size_t bytes) {
  // No portable memset wrapper exists across both runtimes, and this is only used by test harnesses.
  void* z = std::calloc(bytes, 1);
  if (!z && bytes) return rt_record(RtStatus::HostAllocation, "host allocation", "calloc returned null");
  bool const ok = rt_h2d(p, z, bytes);
  std::free(z);
  return ok;
}

// RAII, so a harness with twenty allocations does not leak one per early return.
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(size_t n) : p(n ? rt_malloc(n) : nullptr), bytes(n) {}
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) { if (p) rt_free(p); p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0; }
    return *this;
  }
  DevBuf(DevBuf const&) = delete;
  DevBuf& operator=(DevBuf const&) = delete;
  ~DevBuf() { if (p) rt_free(p); }
  template <typename T> T* as() const { return static_cast<T*>(p); }
  bool from_host(void const* src) { return !bytes || (p && rt_h2d(p, src, bytes)); }
  bool to_host(void* dst) const { return !bytes || (p && rt_d2h(dst, p, bytes)); }
};

// Reserved by every libquactlize_ppu C entry for an allocation/copy/synchronize failure.  Stage D2H through a
// private host buffer so the documented `out is untouched on error` rule also holds when the failing operation is
// the output copy itself.
inline constexpr int kRuntimeError = 41;

template <class T>
int rt_copy_output(DevBuf const& src, T* out, size_t count) {
  size_t const bytes = count * sizeof(T);
  void* tmp = std::malloc(bytes);
  if (!tmp && bytes) {
    rt_record(RtStatus::HostAllocation, "host output staging", "malloc returned null");
    return kRuntimeError;
  }
  bool const copied = src.to_host(tmp);
  if (copied && rt_ok() && bytes) std::memcpy(out, tmp, bytes);
  std::free(tmp);
  return copied && rt_ok() ? 0 : kRuntimeError;
}

template <class T>
int rt_copy_two_outputs(DevBuf const& src0, T* out0, DevBuf const& src1, T* out1, size_t count) {
  size_t const bytes = count * sizeof(T);
  void* tmp = std::malloc(bytes * 2);
  if (!tmp && bytes) {
    rt_record(RtStatus::HostAllocation, "host output staging", "malloc returned null");
    return kRuntimeError;
  }
  auto* tmp0 = static_cast<unsigned char*>(tmp);
  auto* tmp1 = tmp0 + bytes;
  bool const copied0 = src0.to_host(tmp0);
  bool const copied1 = copied0 && src1.to_host(tmp1);
  if (copied1 && rt_ok() && bytes) {
    std::memcpy(out0, tmp0, bytes);
    std::memcpy(out1, tmp1, bytes);
  }
  std::free(tmp);
  return copied1 && rt_ok() ? 0 : kRuntimeError;
}

}  // namespace ppu_gemv
