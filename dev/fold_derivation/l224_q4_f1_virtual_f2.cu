// L224 -- CAN A TILE-FREE Q4/A64/F1 ARTIFACT FEED THE FOLD-2 MMA FRAGMENT?
//
// This is the proof boundary for the experimental "F1-native, fold2-style compute" reader.  Raw byte equality is
// deliberately NOT the question: A64/F1 and A32/F2 have different resident byte maps.  The admissible operation is
// a register-only permutation after each complete 32 B F1 delivery.  Therefore this witness composes the real F1
// AIU/swizzle delivery map with the real F2 TiledMMA fragment and asks three stronger questions:
//
//   1. every F1 physical code has a destination in the SAME thread's F2 fragment;
//   2. the destination is a stable compile-time function of (converter output, delivery, N instance), not data;
//   3. every F2 fragment slot is reached exactly once, with no extra global delivery.
//
// The first production slice is intentionally TacticTileK >= ArtifactTileK=64.  T=32 needs a two-consume macrostep
// so one A64 delivery is retained across two K32 computes; treating it as this proof would either read 16 B with a
// 32 B atom or double weight traffic.  That separate lifetime proof is required before T32 can be admitted.

#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

#include "xplane_offline.hpp"

namespace {

using namespace cute;

template <int TM, int TN, int TK, int WM, int WN>
struct VirtualRow {
  static constexpr int Bits = 4;
  static constexpr int ArtifactFold = 1;
  static constexpr int ComputeFold = 2;
  static constexpr int ArtifactTileK = 64;
  static constexpr int CPW = 32 / Bits;
  static constexpr int DL = (ArtifactFold * TK * Bits / 8) / 32;
  static constexpr int VEC = cutlass::MixGemmEmit<Bits>::kNumOutputs;

  using ShadowM = xplane::BShadowMShape<TM, WM>;
  static constexpr int WarpN = WN > 16 ? WN : 16;
  static constexpr int WOM = ShadowM::MainWarpCount;
  static constexpr int WON = TN / WarpN;
  static constexpr int RPI = WON * 16;
  static constexpr int Threads = 32 * WOM * WON;
  static constexpr int PermK = cutlass::MixGemmMmaPermK<Bits, TK, ComputeFold>::value;
  using Mma = TiledMMA<MMA_Atom<PPU0015_16x16x16_F32F16F16F32_TN>,
                       Layout<Shape<Int<WOM>, Int<WON>, _1>>,
                       Tile<Int<ShadowM::ShadowPermutationM>, Int<WON * 16>, Int<PermK>>>;
  using CTV = xplane::CubeTV<Bits, TM, TN, TK, WM, WN, ArtifactFold, ArtifactTileK>;

  static_assert(TK >= ArtifactTileK && TK % ArtifactTileK == 0,
                "the first virtual-fold slice does not admit sub-artifact K tactics");
  static_assert(DL >= 1 && DL * VEC * (TN / RPI) == TN * TK / Threads,
                "each thread fragment must be covered by complete F1 deliveries");

  struct Stats {
    int events = 0;
    int missing_same_thread = 0;
    int unstable = 0;
    int duplicate_fragment = 0;
    int missing_fragment = 0;
    int physical_duplicate = 0;
    int physical_missing = 0;
    int delivery_reads = 0;
    int expected_delivery_reads = 0;
    int native_f2_scatter_diff = 0;
    int identity_scatter_diff = 0;
    std::vector<int> placement;
  };

  static Stats run(bool wrong_thread = false, bool drop_last_delivery = false) {
    auto const physical_owner =
        xplane::plane_map<Bits, TM, TN, TK, WM, WN, ArtifactFold, ArtifactTileK>();
    constexpr int PhysicalCodes = TN * TK;
    constexpr int FragPerThread = PhysicalCodes / Threads;
    Stats out;
    out.placement.assign(std::size_t(DL * VEC), -1);
    std::vector<int> physical_hits(std::size_t(PhysicalCodes), 0);
    std::vector<int> logical_hits(std::size_t(PhysicalCodes), 0);

    Mma mma;
    auto identity = make_identity_tensor(make_shape(Int<TN>{}, Int<TK>{}));
    for (int t = 0; t < Threads; ++t) {
      int const consumer_t = wrong_thread ? ((t + 1) % Threads) : t;
      auto frag = Mma{}.get_thread_slice(consumer_t).partition_fragment_B(
          make_tensor(make_smem_ptr((cutlass::half_t*)nullptr),
                      make_layout(Shape<Int<TN>, Int<TK>>{}, Stride<Int<TK>, _1>{})));
      auto pi = right_inverse(frag.layout());
      auto part = Mma{}.get_thread_slice(consumer_t).partition_B(identity);
      std::vector<int> flat_owner(std::size_t(size(frag)), -1);
      for (int flat = 0; flat < int(size(frag)); ++flat) {
        auto c = part(pi(flat));
        flat_owner[std::size_t(flat)] = int(get<0>(c)) * TK + int(get<1>(c));
      }

      int const lane = t % 32;
      int const warp = t / 32;
      for (int dl0 = 0; dl0 < DL; ++dl0) {
        if (drop_last_delivery && dl0 == DL - 1) continue;
        int const dl = dl0;
        for (int inst = 0; inst < TN / RPI; ++inst) {
          ++out.delivery_reads;
          for (int v = 0; v < 4; ++v) {
            int const row = CTV::base_row(warp, inst, dl / CTV::SlicesPerInst)
                          + CTV::cube_row(lane, v);
            int const wd = CTV::cube_wd(lane, v, dl);
            for (int j = 0; j < CPW; ++j) {
              int const physical = (((row * DL + dl) * 8 + wd) * CPW + j);
              if (physical < 0 || physical >= int(physical_owner.size())) {
                ++out.missing_same_thread;
                continue;
              }
              ++physical_hits[std::size_t(physical)];
              int const owner = physical_owner[std::size_t(physical)];
              int flat = -1;
              for (int f = 0; f < int(flat_owner.size()); ++f)
                if (flat_owner[std::size_t(f)] == owner) { flat = f; break; }
              if (flat < 0) {
                ++out.missing_same_thread;
                continue;
              }
              ++logical_hits[std::size_t(owner)];
              int const emit = cutlass::MixGemmEmit<Bits>::index(j, v);
              int const key = dl0 * VEC + emit;
              // Remove the N-instance offset.  The converter is invoked once per instance, so a stable local
              // placement is exactly what can be emitted without a runtime branch or a thread-dependent table.
              int const local_flat = flat - inst * DL * VEC;
              int& prior = out.placement[std::size_t(key)];
              if (prior < 0) prior = local_flat;
              else if (prior != local_flat) ++out.unstable;
              ++out.events;
            }
          }
        }
      }
      (void)FragPerThread;
    }

    for (int h : physical_hits) {
      out.physical_duplicate += h > 1;
      out.physical_missing += h == 0;
    }
    for (int h : logical_hits) {
      out.duplicate_fragment += h > 1;
      out.missing_fragment += h == 0;
    }
    out.expected_delivery_reads = Threads * DL * (TN / RPI);
    using VirtualScatter = cutlass::MixGemmArtifactScatter<Bits, ComputeFold, DL>;
    for (int dl = 0; dl < DL; ++dl)
      for (int emit = 0; emit < VEC; ++emit) {
        int const got = out.placement[std::size_t(dl * VEC + emit)];
        out.native_f2_scatter_diff += got != VirtualScatter::flat(emit, dl);
        out.identity_scatter_diff += got != dl * VEC + emit;
      }
    return out;
  }
};

template <int TM, int TN, int TK, int WM, int WN>
bool positive(char const* tag) {
  auto s = VirtualRow<TM, TN, TK, WM, WN>::run();
  bool const ok = s.events == TN * TK && s.missing_same_thread == 0 && s.unstable == 0 &&
                  s.duplicate_fragment == 0 && s.missing_fragment == 0 &&
                  s.physical_duplicate == 0 && s.physical_missing == 0 &&
                  s.delivery_reads == s.expected_delivery_reads && s.identity_scatter_diff == 0 &&
                  (TK == VirtualRow<TM, TN, TK, WM, WN>::ArtifactTileK ||
                   s.native_f2_scatter_diff > 0);
  std::printf("L224_ROW tag=%s tile=%dx%dx%d warp=%dx%d events=%d same_thread_missing=%d "
              "unstable=%d fragment_duplicate=%d fragment_missing=%d physical_duplicate=%d physical_missing=%d "
              "delivery_reads=%d/%d native_f2_scatter_diff=%d identity_scatter_diff=%d verdict=%s\n",
              tag, TM, TN, TK, WM, WN, s.events, s.missing_same_thread, s.unstable,
              s.duplicate_fragment, s.missing_fragment, s.physical_duplicate, s.physical_missing,
              s.delivery_reads, s.expected_delivery_reads, s.native_f2_scatter_diff,
              s.identity_scatter_diff, ok ? "PASS" : "FAIL");
  return ok;
}

template <bool WrongThread, bool SwapDelivery>
bool negative(char const* tag) {
  using R = VirtualRow<64, 128, 128, 64, 64>;
  auto s = R::run(WrongThread, SwapDelivery);
  bool const fired = s.missing_same_thread != 0 || s.unstable != 0 ||
                     s.duplicate_fragment != 0 || s.missing_fragment != 0;
  std::printf("L224_NEGATIVE tag=%s same_thread_missing=%d unstable=%d fragment_duplicate=%d "
              "fragment_missing=%d fired=%d\n", tag, s.missing_same_thread, s.unstable,
              s.duplicate_fragment, s.missing_fragment, int(fired));
  return fired;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= positive<64, 64, 64, 64, 32>("t64-w32");
  ok &= positive<64, 128, 64, 64, 64>("t64-w64");
  ok &= positive<64, 128, 128, 64, 64>("t128");
  ok &= positive<64, 128, 256, 64, 64>("t256");
  ok &= negative<true, false>("wrong-thread");
  ok &= negative<false, true>("missing-delivery");
  std::printf("L224_Q4_F1_VIRTUAL_F2 verdict=%s positives=4 negatives=2 artifact=A64/F1 compute=F2 "
              "weight_byte_multiplier=1 runtime_branches=0 t32=DEFERRED_MACROSTEP\n",
              ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
