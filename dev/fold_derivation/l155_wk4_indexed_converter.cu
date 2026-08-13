// L155 -- factor the classic-aligned WK4 int4 converter into data selection.
//
// The old production implementation selected one of four compile-time
// MixGemmChunkEmit arms with a runtime switch.  This oracle exhausts the exact
// production layout and proves that the same owner map is obtained with:
//
//   vreg = 2*vi + (wk >> 1), pair = 2*(wk & 1) + ti,
//   destination = source*16 + 4*NI + 2*vi + ti.
//
// It also evaluates the fixed-register/mask select used by production.  The
// three tempting independent mistakes -- swapping the two wk bits for the
// vreg, swapping them for the byte phase, and transposing the destination --
// are separately compilable red controls.

#include <array>
#include <cstdint>
#include <cstdio>

#include "cute/tensor.hpp"
#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"

namespace {
using namespace cute;

using Ranked = Layout<Shape<Shape<_2, _2, _2>, _1, _4>,
                      Stride<Stride<_1, _2, _4>, _0, _8>>;

struct Map {
  std::array<int, 32> slot{};
};

constexpr int token(int source, int ni, int vreg, int pair) {
  return 1 + pair + 4 * (vreg + 4 * (ni + 4 * source));
}

template <int WK>
constexpr Map old_template_map() {
  using Emit = cutlass::MixGemmChunkEmit<4, WK, 4, true, Ranked>;
  Map out{};
  for (int& x : out.slot) x = -1;
  for (int source = 0; source < 2; ++source)
    for (int ni = 0; ni < 4; ++ni)
      for (int v = 0; v < 4; ++v)
        for (int t = 0; t < 4; ++t)
          if (Emit::keep(t, v)) {
            int const dst = source * 16 + ni * 4 + Emit::at(t, v);
            if (out.slot[dst] != -1) out.slot[dst] = -2;
            else out.slot[dst] = token(source, ni, v, t);
          }
  return out;
}

enum class Variant { Good, BadHighBit, BadPhaseBit, BadDestination };

constexpr Map indexed_map(int wk, Variant variant) {
  Map out{};
  for (int& x : out.slot) x = -1;
  int const high = variant == Variant::BadHighBit ? (wk & 1) : (wk >> 1);
  int const phase = variant == Variant::BadPhaseBit ? (wk >> 1) : (wk & 1);
  for (int source = 0; source < 2; ++source)
    for (int ni = 0; ni < 4; ++ni)
      for (int vi = 0; vi < 2; ++vi)
        for (int ti = 0; ti < 2; ++ti) {
          int const v = 2 * vi + high;
          int const t = 2 * phase + ti;
          int const local = variant == Variant::BadDestination
              ? 2 * ti + vi : 2 * vi + ti;
          int const dst = source * 16 + 4 * ni + local;
          if (out.slot[dst] != -1) out.slot[dst] = -2;
          else out.slot[dst] = token(source, ni, v, t);
        }
  return out;
}

constexpr std::array<Map, 4> kOld = {
    old_template_map<0>(), old_template_map<1>(),
    old_template_map<2>(), old_template_map<3>()};

constexpr int map_diff(Variant variant) {
  int bad = 0;
  for (int wk = 0; wk < 4; ++wk) {
    auto const indexed = indexed_map(wk, variant);
    for (int i = 0; i < 32; ++i)
      bad += indexed.slot[i] != kOld[wk].slot[i];
  }
  return bad;
}

constexpr uint32_t word(int source, int ni, int vreg) {
  return 0x10203040u ^ uint32_t(source * 0x01010101u)
       ^ uint32_t(ni * 0x11111111u) ^ uint32_t(vreg * 0x0305070bu);
}

constexpr uint32_t select_word(uint32_t even, uint32_t odd, int high) {
  uint32_t const mask = 0u - uint32_t(high & 1);
  return even ^ ((even ^ odd) & mask);
}

constexpr int selected_word_bad() {
  int bad = 0;
  for (int wk = 0; wk < 4; ++wk)
    for (int source = 0; source < 2; ++source)
      for (int ni = 0; ni < 4; ++ni)
        for (int vi = 0; vi < 2; ++vi) {
          uint32_t const got = select_word(
              word(source, ni, 2 * vi), word(source, ni, 2 * vi + 1), wk >> 1)
              >> (8 * (wk & 1));
          uint32_t const want = word(source, ni, 2 * vi + (wk >> 1))
              >> (8 * (wk & 1));
          bad += got != want;
        }
  return bad;
}

static_assert(map_diff(Variant::Good) == 0,
              "indexed WK4 converter must equal all compile-time Chunk arms");
static_assert(selected_word_bad() == 0,
              "fixed-register mask select must equal the closed-form source");
static_assert(map_diff(Variant::BadHighBit) == 64);
static_assert(map_diff(Variant::BadPhaseBit) == 64);
static_assert(map_diff(Variant::BadDestination) == 64);

} // namespace

int main() {
  constexpr int good = map_diff(Variant::Good);
  constexpr int bad_high = map_diff(Variant::BadHighBit);
  constexpr int bad_phase = map_diff(Variant::BadPhaseBit);
  constexpr int bad_destination = map_diff(Variant::BadDestination);
  std::printf("L155 indexed-vs-template outputs=128 bad=%d select-word-bad=%d\n",
              good, selected_word_bad());
  std::printf("L155 planted high-bit=%d phase-bit=%d destination=%d EXPECTED-RED\n",
              bad_high, bad_phase, bad_destination);

#if defined(L155_BAD_HIGH_BIT)
  std::puts("L155 EXPECTED-FAIL: wk low bit selected the vreg column");
  return bad_high == 64 ? 1 : 2;
#elif defined(L155_BAD_PHASE_BIT)
  std::puts("L155 EXPECTED-FAIL: wk high bit selected the byte phase");
  return bad_phase == 64 ? 1 : 2;
#elif defined(L155_BAD_DESTINATION)
  std::puts("L155 EXPECTED-FAIL: vi/ti destination axes were transposed");
  return bad_destination == 64 ? 1 : 2;
#else
  std::puts("L155 PASS: cohort is a register/phase index, not control flow");
  return good == 0 && selected_word_bad() == 0 ? 0 : 1;
#endif
}
