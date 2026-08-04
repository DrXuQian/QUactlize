// L105 -- WHICH CONFIGURATIONS ACTUALLY PRODUCE DIFFERENT LOW-PLANE BYTES, and can a format pin ONE F?
//
// WHY THIS EXISTS. Two written claims in this repo contradict each other about place_derived, and both are cited:
//
//   layouts.py xplane()                    "TK changes it ... TN,WN through TN/max(WN,16)"
//   unfused_weight_dequantize.hpp (l61)    "the unfolded placement is TILE-INVARIANT ... any (TN, TK) dividing
//                                           (N, K) within the delivery bound gives the same buffer"
//
// l104 settled the HIGH plane and reported in passing that "TM, WM and WN do not survive into the bytes anywhere
// in this grid", which agrees with neither of the above about WN. So the low plane -- the whole of a single-plane
// format -- has three claims and no measurement. This file measures it.
//
// WHAT IT ANSWERS, in the order the answers are wanted:
//   1. group every legal configuration by its STORED BYTES, and report which parameters actually split the groups
//   2. from that, which configurations are redundant and can be pruned
//   3. whether ONE F per bit width suffices -- the state where a format's layout is a function of the format
//      alone, so an artifact header needs no descriptor and the online tactic search is "anything keeping that F"
//
// RESULT, AFTER ATTACKING THE ORIGINAL GRID. F=1 is byte-invariant for every complete tactic tested with TK<=256,
// now including atom-sized TM/TN=16, TN=32, WN=16, both orientations of a non-square warp, a 32-warp CTA, and TM
// below 32. But it is NOT an unrestricted theorem: int4 (TM64,TN64,TK512,w32x32,F1) currently instantiates and
// produces different bytes. A bits-only F=1 token therefore needs TK<=256 as an explicit tactic-domain boundary.
//
// THE ORIGINAL "LEGALITY BOUND" WAS INCOMPLETE. A whole 32-byte physical row is necessary, but a tactic also needs
// a valid atom/warp tiling, enough per-thread fragment slots, and at least one N iteration (NI=WN/(16*F)). The first
// run counted F=4/WN=32 outputs as layout classes even though NI=0 left plane_map entirely unset and emitted an
// all-zero buffer. Those rows are now printed as skips and WN=64 supplies the legal F=4 controls. A sweep that hides
// or misclassifies its coverage gaps is how an invariance claim ends up broader than its evidence.
//
//   nvcc -std=c++17 -O2 -I ../../include -I ../../../third_party/actlize/include \
//        l105_low_plane_config_classes.cu -o /tmp/l105 && /tmp/l105
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "xplane_offline.hpp"

namespace {

constexpr int kN = 256, kK = 512;      // divisible by every TN/TK in the grid

std::vector<uint8_t> source(int bits) {
  // Deterministic xorshift, and NOT the old multiply-and-shift pattern: a periodic low-bit fixture can make a
  // displacement invisible (the exact l52 failure recorded in xplane_offline.hpp). Classes below are keyed by the
  // COMPLETE stored bytes, not this buffer's 64-bit report hash, so a hash collision cannot manufacture equality.
  std::vector<uint8_t> q(size_t(kN) * kK);
  uint32_t s = 0x9e3779b9u;
  for (size_t i = 0; i < q.size(); ++i) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    q[i] = uint8_t(s & ((1u << bits) - 1u));
  }
  return q;
}

uint64_t fnv(std::vector<int8_t> const& v) {
  uint64_t h = 1469598103934665603ull;
  for (int8_t b : v) { h ^= uint8_t(b); h *= 1099511628211ull; }
  return h;
}

template <int Bits, int TM, int TN, int TK, int WM, int WN, int F>
void run(std::map<std::vector<int8_t>, std::vector<std::string>>& classes,
         std::vector<std::string>& skipped) {
  // THE GUARD IS if constexpr, NOT AN if. xplane_offline.hpp asserts "row must be a whole number of 32B
  // deliveries" at COMPILE time, so an illegal combination cannot be skipped at run time -- it has to not be
  // instantiated. A complete tactic needs more than that physical-row condition: the atom/warp tiling, per-thread
  // delivery capacity, and at least one N instance must all exist. The first l105 counted F=4/WN=32 rows as legal;
  // Ng/RPI is zero there, plane_map is all -1, and the resulting all-zero buffer is not a layout class.
  constexpr bool kWholeRow = (F * TK * Bits >= 256) && (F * TK * Bits) % 256 == 0;
  constexpr bool kAtom = TM % 16 == 0 && TN % 16 == 0 && TK % 16 == 0 && TN % F == 0;
  constexpr bool kWarp = WM > 0 && WN > 0 && TM % WM == 0 && TN % WN == 0
                      && (TM / WM) * (TN / WN) <= 32;
  constexpr bool kThreadDelivery = WN * TK * Bits >= 4096;
  constexpr bool kNIteration = WN >= 16 * F;  // Ng/RPI = WN/(16*F)
  constexpr bool kFitsFixture = kN % TN == 0 && kK % TK == 0;
  constexpr bool kLegal = kWholeRow && kAtom && kWarp && kThreadDelivery && kNIteration && kFitsFixture;
  char tag[128];
  std::snprintf(tag, sizeof tag, "b%d TM%-3d TN%-3d TK%-3d WM%-2d WN%-2d F%d", Bits, TM, TN, TK, WM, WN, F);
  if constexpr (!kLegal) {
    char why[64];
    std::snprintf(why, sizeof why, "%s", !kWholeRow ? "incomplete 32B row" : !kAtom ? "non-atom tile"
        : !kWarp ? "invalid warp tiling" : !kThreadDelivery ? "over-delivery"
        : !kNIteration ? "NI=0" : "does not divide fixture");
    skipped.push_back(std::string(tag) + " -- " + why);
    return;
  }
  else {
  auto q = source(Bits);
  std::vector<int8_t> out(size_t(kN) * kK * Bits / 8, 0);
  xplane::place_derived<Bits, TM, TN, TK, WM, WN, F>(out.data(), q, kN, kK);
  classes[std::move(out)].push_back(tag);
  }
}

// The grid. TM/WM are included precisely because l104 says they do not survive; a sweep that omits the
// parameters it expects to be irrelevant cannot report that they are.
template <int Bits>
void sweep(std::map<std::vector<int8_t>, std::vector<std::string>>& c, std::vector<std::string>& sk) {
  // Unfolded boundary coverage the original l61/l105 grids omitted: atom-sized M/N tiles, WN=16, TM<32,
  // and both orientations of a non-square warp. All use TK=256, which is legal for every live width.
  run<Bits, 16,  16, 256, 16, 16, 1>(c, sk);  run<Bits, 32,  32, 256, 16, 16, 1>(c, sk);
  run<Bits, 64,  32, 256, 32, 16, 1>(c, sk);  run<Bits, 32,  64, 256, 16, 32, 1>(c, sk);
  run<Bits, 64,  64, 256, 64, 32, 1>(c, sk);  run<Bits,128,  64, 256, 16, 16, 1>(c, sk); // 32 warps
  if constexpr (Bits >= 2) {
    run<Bits, 16,  16, 128, 16, 16, 1>(c, sk);  run<Bits, 32,  32, 128, 16, 16, 1>(c, sk);
    run<Bits, 64,  32, 128, 32, 16, 1>(c, sk);  run<Bits, 32,  64, 128, 16, 32, 1>(c, sk);
  }
  if constexpr (Bits == 4) {
    run<Bits, 16,  16,  64, 16, 16, 1>(c, sk);  run<Bits, 32,  32,  64, 16, 16, 1>(c, sk);
    run<Bits, 64,  32,  64, 32, 16, 1>(c, sk);  run<Bits, 32,  64,  64, 16, 32, 1>(c, sk);
  }
  run<Bits, 32,  64,  64, 32, 32, 1>(c, sk);  run<Bits, 64,  64,  64, 32, 32, 1>(c, sk);
  run<Bits, 64,  64, 128, 32, 32, 1>(c, sk);  run<Bits, 64,  64, 256, 32, 32, 1>(c, sk);
  run<Bits, 64, 128, 128, 32, 32, 1>(c, sk);  run<Bits, 64, 128, 256, 32, 64, 1>(c, sk);
  run<Bits,128, 128, 256, 64, 64, 1>(c, sk);  run<Bits, 64, 256, 256, 32, 64, 1>(c, sk);
  run<Bits, 64,  64,  64, 32, 32, 2>(c, sk);  run<Bits, 64,  64, 128, 32, 32, 2>(c, sk);
  run<Bits, 64, 128, 128, 32, 64, 2>(c, sk);  run<Bits, 64,  64,  64, 32, 32, 4>(c, sk);
  run<Bits, 64,  64,  64, 32, 64, 4>(c, sk);  // legal F=4 control: WN=64 makes NI=1
  // Boundary attack: the unfolded writer is documented as interleave-256, but the old sweep stopped at TK=256.
  // int4/TK=512 is the only wider row that currently gets through place_derived's RPS static_assert.
  if constexpr (Bits == 4) run<Bits, 64, 64, 512, 32, 32, 1>(c, sk);
}

void report(char const* what, std::map<std::vector<int8_t>, std::vector<std::string>> const& c,
            std::vector<std::string> const& skipped) {
  std::printf("\n== %s: %zu distinct layout(s) over %d built configurations (%d skipped as unbuildable)\n",
              what, c.size(), [&]{ int n = 0; for (auto& kv : c) n += int(kv.second.size()); return n; }(),
              int(skipped.size()));
  int i = 0;
  for (auto const& kv : c) {
    std::printf("  class %d  hash %016llx  %zu config(s)\n", i++, (unsigned long long)fnv(kv.first), kv.second.size());
    for (auto const& t : kv.second) std::printf("      %s\n", t.c_str());
  }
  for (auto const& t : skipped) std::printf("  SKIP  %s\n", t.c_str());
}

}  // namespace

int main() {
  std::printf("L105 -- do different configurations give different LOW-plane bytes?\n");
  std::printf("Legality: complete 32B row + atom/warp tiling + per-thread delivery + NI>=1; exact skips below.\n");
  for (int bits : {1, 2, 4}) {
    std::map<std::vector<int8_t>, std::vector<std::string>> c;
    std::vector<std::string> skipped;
    if (bits == 1) sweep<1>(c, skipped);
    else if (bits == 2) sweep<2>(c, skipped);
    else sweep<4>(c, skipped);
    char label[32]; std::snprintf(label, sizeof label, "int%d", bits);
    report(label, c, skipped);
  }
  std::printf("\nRead the classes, not the count: a parameter that never differs BETWEEN classes does not reach\n"
              "the stored bytes, and configurations inside one class are interchangeable -- one artifact serves\n"
              "them all, which is what makes an online tactic search possible without repacking.\n");
  return 0;
}
