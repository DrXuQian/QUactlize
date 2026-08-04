// The shared library's runtime shim must be testable without a device and, above all, must not terminate its host
// process.  Inject every operation class which used to call exit(1), then prove the status is latched and output
// staging preserves the caller's buffer on a failed D2H.
#define PPU_GEMV_RT_FAKE 1
#include "gemv_lowbit/gemv_rt.hpp"

#include <array>
#include <cstdio>

using ppu_gemv::DevBuf;
using ppu_gemv::RtStatus;

int main() {
  int bad = 0;

  ppu_gemv::rt_clear_error();
  ppu_gemv::rt_test_fail_next(RtStatus::Allocation);
  DevBuf failed(16);
  bad += failed.p != nullptr || ppu_gemv::rt_status() != RtStatus::Allocation;

  ppu_gemv::rt_clear_error();
  DevBuf buf(16);
  std::array<unsigned char, 16> src{};
  ppu_gemv::rt_test_fail_next(RtStatus::HostToDevice);
  bad += buf.from_host(src.data()) || ppu_gemv::rt_status() != RtStatus::HostToDevice;

  ppu_gemv::rt_clear_error();
  ppu_gemv::rt_test_fail_next(RtStatus::Synchronize);
  bad += ppu_gemv::rt_sync("injected") || ppu_gemv::rt_status() != RtStatus::Synchronize;

  ppu_gemv::rt_clear_error();
  std::array<unsigned char, 16> out;
  out.fill(0xa5);
  ppu_gemv::rt_test_fail_next(RtStatus::DeviceToHost);
  int const rc = ppu_gemv::rt_copy_output(buf, out.data(), out.size());
  bad += rc != ppu_gemv::kRuntimeError || ppu_gemv::rt_status() != RtStatus::DeviceToHost;
  for (auto x : out) bad += x != 0xa5;

  ppu_gemv::rt_clear_error();
  src.fill(0x3c);
  bad += !buf.from_host(src.data());
  out.fill(0);
  bad += ppu_gemv::rt_copy_output(buf, out.data(), out.size()) != 0 || out != src;

  std::printf("rt-error-contract bad=%d rc=%d output_untouched=%d\n", bad,
              ppu_gemv::kRuntimeError, out == src);
  return bad ? 1 : 0;
}
