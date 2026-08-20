// L219 -- close the two host-provable seams in the exact Q4_K/A32 folded
// reader without pretending that a host model proves PPU register codegen.
//
// 1. The production chunk emitter writes half2 values.  Its destination is a
//    CuTe (half2-in-group, delivery, fold, instance) layout and must agree
//    exactly with the independently shared artifact scatter.
// 2. The historical cadence prepared next-tile delivery zero before consuming
//    current-tile delivery three.  A constructive register-alias plant must
//    corrupt that order and must not corrupt consume-before-prepare.

// The second plant models the device lifetime failure under investigation; it
// does not claim that nvcc allocates the real PPU registers.  The exact PPU row
// remains the final verdict.

#include <array>
#include <cstdio>

#include "quactlize_extensions/cutlass/quactlize_mix_gemm_convert.h"

namespace {

using Scatter = cutlass::MixGemmArtifactScatter<4, 2, 4>;
using Typed = Scatter::Half2Layout<1>;

struct MapResult {
  int mismatch = 0;
  int duplicate = 0;
  int missing = 0;
  int wrap_overlap = 0;
};

MapResult check_typed_map() {
  std::array<int, 64> owner{};
  owner.fill(-1);
  MapResult out;
  for (int fold = 0; fold < 2; ++fold) {
    for (int delivery = 0; delivery < 4; ++delivery) {
      for (int h2 = 0; h2 < Scatter::Group / 2; ++h2) {
        int const typed = int(Typed{}(h2, delivery, fold, 0));
        int const independent = Scatter::group_base(fold, delivery, 0) / 2 + h2;
        out.mismatch += typed != independent;
        if (typed < 0 || typed >= int(owner.size())) {
          ++out.mismatch;
          continue;
        }
        out.duplicate += owner[typed] >= 0;
        owner[typed] = 8 * fold + h2;
      }
    }
  }
  for (int value : owner) out.missing += value < 0;
  for (int fold = 0; fold < 2; ++fold)
    for (int h2 = 0; h2 < 8; ++h2)
      out.wrap_overlap +=
          int(Typed{}(h2, 3, fold, 0)) == int(Typed{}(h2, 0, fold, 0));
  return out;
}

// A deliberately invalid destination map: erasing the delivery stride makes
// all four deliveries name the same 16 half2 slots.  This proves the exact-once
// check can turn red; it is not a second spelling of the production layout.
int wrong_delivery_stride_duplicates() {
  std::array<int, 64> seen{};
  seen.fill(0);
  int duplicate = 0;
  for (int fold = 0; fold < 2; ++fold)
    for (int delivery = 0; delivery < 4; ++delivery)
      for (int h2 = 0; h2 < 8; ++h2) {
        int const planted = h2 + 32 * fold;
        duplicate += seen[planted]++ != 0;
      }
  return duplicate;
}

struct CadenceResult {
  int legacy_bad = 0;
  int corrected_bad = 0;
  int next_bad = 0;
};

CadenceResult check_cadence_alias_plant() {
  std::array<int, 16> physical{};
  std::array<int, 16> current{};
  std::array<int, 16> next{};
  std::array<int, 16> consumed{};
  for (int i = 0; i < 16; ++i) {
    current[i] = 0x3000 + i;
    next[i] = 0x4000 + i;
  }

  // Historical order under the planted allocation: next delivery zero is
  // prepared into the same physical registers while current delivery three
  // is still live, then MMA observes the overwrite.
  physical = current;
  physical = next;
  consumed = physical;
  CadenceResult out;
  for (int i = 0; i < 16; ++i) out.legacy_bad += consumed[i] != current[i];

  // Corrected order consumes the current delivery first.  Preparing the next
  // delivery may reuse those registers only after the current value is dead.
  physical = current;
  consumed = physical;
  physical = next;
  for (int i = 0; i < 16; ++i) {
    out.corrected_bad += consumed[i] != current[i];
    out.next_bad += physical[i] != next[i];
  }
  return out;
}

}  // namespace

int main() {
  static_assert(cute::size(Typed{}) == 64,
                "Q4/A32 destination must contain 64 half2 values");
  static_assert(cute::cosize(Typed{}) == 64,
                "Q4/A32 typed destination must be compact and exact-once");
  auto const map = check_typed_map();
  int const wrong_stride = wrong_delivery_stride_duplicates();
  auto const cadence = check_cadence_alias_plant();
  bool const ok = map.mismatch == 0 && map.duplicate == 0 &&
                  map.missing == 0 && map.wrap_overlap == 0 &&
                  wrong_stride > 0 && cadence.legacy_bad == 16 &&
                  cadence.corrected_bad == 0 && cadence.next_bad == 0;
  std::printf(
      "L219 Q4/A32 typed-half2 map_bad=%d duplicate=%d missing=%d "
      "d3_d0_overlap=%d wrong-stride-duplicate=%d "
      "alias-plant legacy_bad=%d corrected_bad=%d next_bad=%d result=%s\n",
      map.mismatch, map.duplicate, map.missing, map.wrap_overlap,
      wrong_stride, cadence.legacy_bad, cadence.corrected_bad,
      cadence.next_bad, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
