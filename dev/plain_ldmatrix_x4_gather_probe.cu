// PPU001 plain ldmatrix.x4 provider-map probe.
//
// Each lane points at a distinct aligned 16-byte shared row whose eight b16
// values encode (provider lane, word).  The output therefore reveals whether
// the address operand is lane-local and gives the exact provider/word map for
// every (output lane, register).  A collapsed-address control makes all lanes
// point at provider row zero and must lose the unique-provider fingerprint.

#include <cstdint>
#include <cstdio>

#include "hggc_runtime.h"

namespace {

constexpr int kWarp = 32;
constexpr int kWordsPerProvider = 8;

struct alignas(16) SharedRows {
  std::uint16_t values[kWarp][kWordsPerProvider];
};

__device__ constexpr std::uint16_t tag(int provider, int word) {
  return std::uint16_t((provider << 8) | word);
}

__global__ void plain_x4_probe(int collapse_addresses) {
  __shared__ SharedRows rows;
  int const lane = int(threadIdx.x) & 31;
  for (int word = 0; word < kWordsPerProvider; ++word)
    rows.values[lane][word] = tag(lane, word);
  __syncthreads();

  int const provider = collapse_addresses ? 0 : lane;
  void const* row_ptr = static_cast<void const*>(&rows.values[provider][0]);
  std::uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile(
      "ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(row_ptr)
      : "memory");
#endif

  printf(
      "PLAIN_X4_CELL arm=%s lane=%d "
      "r0=%04x,%04x r1=%04x,%04x r2=%04x,%04x r3=%04x,%04x\n",
      collapse_addresses ? "collapsed" : "lane-local", lane,
      unsigned(r0 & 0xffffu), unsigned(r0 >> 16),
      unsigned(r1 & 0xffffu), unsigned(r1 >> 16),
      unsigned(r2 & 0xffffu), unsigned(r2 >> 16),
      unsigned(r3 & 0xffffu), unsigned(r3 >> 16));
}

}  // namespace

int main() {
  std::puts("PLAIN_X4_BEGIN opcode=ppu.ldmatrix.m8n8.x4.shared.b16 "
            "rows=32x16B tags=provider:word");
  plain_x4_probe<<<1, kWarp>>>(0);
  if (hggcDeviceSynchronize() != hggcSuccess) return 2;
  plain_x4_probe<<<1, kWarp>>>(1);
  if (hggcDeviceSynchronize() != hggcSuccess) return 3;
  std::puts("PLAIN_X4_END rows=64");
  return 0;
}
