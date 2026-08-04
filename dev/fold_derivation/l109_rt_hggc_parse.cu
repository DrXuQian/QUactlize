// Parse and execute the exact __HGGCCC__ branch used by libquactlize_ppu.so against the local runtime stubs.
#include "gemv_lowbit/gemv_rt.hpp"

#include <array>
#include <cstdio>

int main() {
  ppu_gemv::rt_clear_error();
  std::array<unsigned char, 8> host{};
  ppu_gemv::DevBuf device(host.size());
  bool const h2d = device.from_host(host.data());
  bool const sync = ppu_gemv::rt_sync("hggc parse gate");
  bool const d2h = device.to_host(host.data());
  int const bad = !h2d || !sync || !d2h || !ppu_gemv::rt_ok();
  std::printf("rt-hggc-parse bad=%d\n", bad);
  return bad;
}
