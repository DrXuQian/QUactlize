// L99 -- DO THE BENCH'S TWO SCALE REPRESENTATIONS DECODE TO THE SAME NUMBERS?
//
// test_moe_splitk_bench used to fill its fp16 planes with scale = +0.0625 and zero = -0.0625 and hand the SAME
// allocation to the packed path, which reinterprets it as 16-byte gguf units. The two representations therefore
// decoded to different values, and every pack-against-base timing quoted from that bench compared two kernels doing
// different arithmetic. That is not a small confound: it is the central measurement of the whole native-format
// question.
//
// The fix picks x = 0.0625, sc = 1, mn = 7, so scale = d*sc = x and zero = 8*scale - dmin*mn = 8x - 7x = x, making
// both planes the constant +x. This file checks that identity through the SAME decode functions the kernel uses --
// both of them, the scalar group_of_words and the packed group_pair_of_words -- for all eight groups, rather than
// leaving it asserted in a comment. x = 0x2c00 is also well clear of the subnormal range, so nothing here wanders
// into the denormal path in either direction.
//
//   nvcc -std=c++17 -x cu -arch=sm_80 -w -I stub_inc -I <actlize>/include -o /tmp/l99 l99_bench_like_for_like.cu
#include <cstdio>
#include <cstring>
#include "quactlize_extensions/cutlass/gguf_packed_scale.h"
using cutlass::half_t; namespace gp = cutlass::gguf_packed;
int main() {
  half_t const x(0.0625f);
  uint8_t u8[16] = {}; uint16_t xb = x.raw();
  std::memcpy(u8 + 0, &xb, 2); std::memcpy(u8 + 2, &xb, 2);
  for (int g = 0; g < 8; ++g) { gp::put_code(u8, g, 0, 1); gp::put_code(u8, g, 1, 7); }
  std::printf("unit bytes:"); for (int i=0;i<16;++i) std::printf(" %02x", u8[i]); std::printf("\n");
  uint32_t u[4]; std::memcpy(u, u8, 16);
  auto h = gp::head_of_words(u); uint32_t m2 = gp::mul2_of_words(u);
  int bad = 0;
  for (int g = 0; g < 8; ++g) {
#define ONE(G) if (g==G) { auto a = gp::group_of_words<G,0,true,8>(u,h); auto b = gp::group_pair_of_words<G,8,0>(u,m2); \
    if (a.scale != x || a.zero != x || b.scale != x || b.zero != x) { ++bad; \
      std::printf("  g%d scalar(%g,%g) packed(%g,%g) want (%g,%g)\n", G, float(a.scale),float(a.zero),float(b.scale),float(b.zero),float(x),float(x)); } }
    ONE(0) ONE(1) ONE(2) ONE(3) ONE(4) ONE(5) ONE(6) ONE(7)
#undef ONE
  }
  std::printf("%s: all 8 groups decode to scale=zero=%g in BOTH the scalar and packed paths -> %d bad\n",
              bad?"FAIL":"PASS", float(x), bad);
  return bad?1:0;
}
