// PPU001 shipping x4.swzl lane-base probe.
//
// Each lane supplies a distinct aligned 512-byte cube base. Every b16 word in
// a cube encodes (base-provider lane, physical word), so the real tc01 swzl
// output reveals whether [%4] is lane-local without changing the opcode or
// any swizzle parameter used by the mixed-input kernel. A collapsed-base
// control makes all lanes point at cube zero.

#include <cstdint>
#include <cstdio>

#include "hggc_runtime.h"

namespace {

constexpr int kWarp = 32;
constexpr int kCubeH = 16;
constexpr int kCubeWBytes = 32;
constexpr int kHalfwordsPerCube = kCubeH * kCubeWBytes / 2;

struct alignas(32) SharedCubes {
  std::uint16_t values[kWarp][kHalfwordsPerCube];
};

__device__ constexpr std::uint16_t tag(int provider, int word) {
  return std::uint16_t((provider << 8) | word);
}

__global__ void swzl_x4_probe(int collapse_bases) {
  __shared__ SharedCubes cubes;
  int const lane = int(threadIdx.x) & 31;
  for (int word = 0; word < kHalfwordsPerCube; ++word)
    cubes.values[lane][word] = tag(lane, word);
  __syncthreads();

  int const provider = collapse_bases ? 0 : lane;
  auto* stage_base = reinterpret_cast<std::int8_t*>(
      &cubes.values[provider][0]);
  std::uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
  asm volatile(
      "ppu.tc01.ldmatrix.sync.aligned.m8n8.x4.swzl.shared.b16 "
      "{%0,%1,%2,%3}, [%4], {%5,%6,%7,%8,%9,%10};"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(stage_base), "r"(0), "r"(0), "r"(1), "r"(kCubeH),
        "r"(1), "r"(0)
      : "memory");
#endif

  printf(
      "SWZL_X4_CELL arm=%s lane=%d "
      "r0=%04x,%04x r1=%04x,%04x r2=%04x,%04x r3=%04x,%04x\n",
      collapse_bases ? "collapsed" : "lane-local", lane,
      unsigned(r0 & 0xffffu), unsigned(r0 >> 16),
      unsigned(r1 & 0xffffu), unsigned(r1 >> 16),
      unsigned(r2 & 0xffffu), unsigned(r2 >> 16),
      unsigned(r3 & 0xffffu), unsigned(r3 >> 16));
}

}  // namespace

int main() {
  std::puts("SWZL_X4_BEGIN opcode=tc01.m8n8.x4.swzl.shared.b16 "
            "bases=32x512B tags=provider:physical-word");
  swzl_x4_probe<<<1, kWarp>>>(0);
  if (hggcDeviceSynchronize() != hggcSuccess) return 2;
  swzl_x4_probe<<<1, kWarp>>>(1);
  if (hggcDeviceSynchronize() != hggcSuccess) return 3;
  std::puts("SWZL_X4_END rows=64");
  return 0;
}
