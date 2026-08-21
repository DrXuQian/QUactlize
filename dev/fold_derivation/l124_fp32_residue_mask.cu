// L124 -- host oracle for Marlin's effective-row guard on the FP32 fixup path.
//
// The generated case list contains every distinct (TM,TN,WM,WN,atom) found in
// the shipping and benchmark tactic tables.  For each real PPU0010 C-fragment
// layout this probe proves three contracts without a device:
//   * (thread,fragment-index) covers every accumulator (m,n) exactly once;
//   * the scalar-striped predicate accepts exactly m_valid*n_valid slots for
//     every possible rectangular residue, including the m8 atom family;
//   * S=1..4 predicated store/add/load_add is bit-identical on valid outputs,
//     while qNaN poison in invalid registers is never accessed or leaked when
//     the same workspace is reused.
// Allocation, cache-line traffic, atomics, locks, and device timing are not
// modeled here; those remain box gates.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_accumulator_residue_mask.hpp"
#include "l124_cases.inc"

namespace {
using namespace cute;

template <int AtomM> struct AtomFor;
template <> struct AtomFor<8> { using type = PPU0010_8x16x16_F32F16F16F32_TN; };
template <> struct AtomFor<16> { using type = PPU0010_16x16x16_F32F16F16F32_TN; };

uint32_t bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

float from_bits(uint32_t u) {
  float x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

constexpr uint32_t kPoison = 0x7fc12345u;

struct Totals {
  uint64_t cases = 0, slots = 0, residues = 0, simulations = 0;
  uint64_t coverage_bad = 0, helper_bad = 0, mask_bad = 0, semantic_bad = 0;
  // Index by exact CTA thread count.  1024 is the tactic-space ceiling.
  std::array<uint64_t, 1025> cohort_cases{};
  std::array<uint64_t, 1025> cohort_slots{};
  std::array<uint64_t, 1025> cohort_simulations{};
};

float value(int epoch, int peer, int logical) {
  // Finite, non-symmetric values.  Both arms execute the same FP32 sequence;
  // bit equality therefore detects a predicate/address error, not tolerance.
  int const sign = ((logical + 3 * peer + epoch) & 1) ? -1 : 1;
  return float(sign * (1 + ((17 * logical + 11 * peer + 29 * epoch) & 127))) / 32.0f;
}

template <int TM, int TN, int WM, int WN, int AtomM>
struct Geometry {
  static_assert(TM % WM == 0 && TN % WN == 0);
  static_assert((AtomM == 8) == (TM == 8 && WM == 8),
                "case generator and production atom selector disagree");
  using Atom = typename AtomFor<AtomM>::type;
  static constexpr int IM = size<0>(typename MMA_Traits<Atom>::Shape_MNK{});
  static constexpr int IN = size<1>(typename MMA_Traits<Atom>::Shape_MNK{});
  using Mma = TiledMMA<MMA_Atom<Atom>,
      Layout<Shape<Int<TM / WM>, Int<TN / WN>, _1>>,
      Tile<Int<(TM / WM) * IM>, Int<(TN / WN) * IN>, _16>>;
  static constexpr int Threads = size(Mma{});
  using Fragment = decltype(make_fragment_like<float>(partition_fragment_C(
      Mma{}, make_shape(Int<TM>{}, Int<TN>{}))));
  static constexpr int FragmentSlots = size(Fragment{});
  static constexpr int TileSlots = TM * TN;
  static_assert(Threads > 0 && Threads % 32 == 0 && Threads <= 1024,
                "real tactic CTA cohort must be warp-aligned and fit one CTA");
  static_assert(Threads * FragmentSlots == TileSlots,
                "FP32 scalar stripes must span one complete output tile");
};

template <class G, int TM, int TN>
bool simulate(std::vector<int> const& lm, std::vector<int> const& ln,
              int mv, int nv, int peers, bool wrong_store_address = false) {
  int constexpr E = G::TileSlots;
  std::vector<float> full_ws(E, from_bits(kPoison));
  std::vector<float> pred_ws(E, from_bits(kPoison));
  auto run = [&](int run_mv, int run_nv, int epoch) {
    std::vector<float> full_out(E, from_bits(kPoison));
    std::vector<float> pred_out(E, from_bits(kPoison));
    std::vector<uint8_t> pred_touched(E, 0);
    for (int peer = 0; peer < peers; ++peer) {
      bool const last = peer + 1 == peers;
      for (int slot = 0; slot < E; ++slot) {
        bool const valid = lm[slot] < run_mv && ln[slot] < run_nv;
        float const x = valid
          ? value(epoch, peer, lm[slot] * TN + ln[slot]) : from_bits(kPoison);
        int const store_slot = wrong_store_address ? (slot + 1) % E : slot;
        if (peers == 1) {
          full_out[slot] = x;
          if (valid) pred_out[slot] = x;
        } else if (!last) {
          if (peer == 0) full_ws[slot] = x;
          else full_ws[slot] += x;
          if (valid) {
            ++pred_touched[store_slot];
            if (peer == 0) pred_ws[store_slot] = x;
            else pred_ws[store_slot] += x;
          }
        } else {
          full_out[slot] = x + full_ws[slot];
          if (valid) {
            ++pred_touched[slot];
            pred_out[slot] = x + pred_ws[slot];
          }
        }
      }
    }
    bool run_ok = true;
    for (int slot = 0; slot < E; ++slot) {
      bool const valid = lm[slot] < run_mv && ln[slot] < run_nv;
      if (valid) run_ok &= bits(pred_out[slot]) == bits(full_out[slot]);
      else {
        run_ok &= bits(pred_out[slot]) == kPoison;
        run_ok &= pred_touched[slot] == 0;
      }
    }
    return run_ok;
  };

  // Deliberately retain both workspaces between a complementary first residue
  // and the requested second residue.  A newly-valid slot must be overwritten
  // by peer0 before use; a newly-invalid slot must not be touched at all.
  bool const first = run(TM - mv, TN - nv, 0);
  bool const second = run(mv, nv, 1);
  return first && second;
}

template <int TM, int TN, int WM, int WN, int AtomM>
bool check_case(Totals& z) {
  using G = Geometry<TM, TN, WM, WN, AtomM>;
  int constexpr E = G::TileSlots;
  std::vector<int> lm(E, -1), ln(E, -1), cover(E, 0);
  auto identity = make_identity_tensor(make_shape(Int<TM>{}, Int<TN>{}));
  bool ok = true;

  for (int t = 0; t < G::Threads; ++t) {
    auto part = typename G::Mma{}.get_thread_slice(t).partition_C(identity);
    typename G::Fragment accum;
    auto physical_to_fragment = right_inverse(accum.layout());
    ok &= int(size(part)) == G::FragmentSlots;
    for (int i = 0; i < G::FragmentSlots; ++i) {
      // This is BlockStripedReduce's scalar workspace address.
      int const slot = i * G::Threads + t;
      auto const c = part(physical_to_fragment(i));
      int const m = int(get<0>(c)), n = int(get<1>(c));
      ok &= 0 <= m && m < TM && 0 <= n && n < TN;
      if (0 <= m && m < TM && 0 <= n && n < TN) {
        lm[slot] = m; ln[slot] = n; ++cover[m * TN + n];
      }
    }
  }
  for (int x : cover) ok &= x == 1;
  if (!ok) ++z.coverage_bad;

  // Prefix sums make every residue exhaustive without an O(TM^2*TN^2) walk.
  std::vector<int> prefix((TM + 1) * (TN + 1), 0);
  for (int m = 0; m < TM; ++m) for (int n = 0; n < TN; ++n) {
    prefix[(m + 1) * (TN + 1) + n + 1] = cover[m * TN + n]
      + prefix[m * (TN + 1) + n + 1] + prefix[(m + 1) * (TN + 1) + n]
      - prefix[m * (TN + 1) + n];
  }
  for (int mv = 0; mv <= TM; ++mv) for (int nv = 0; nv <= TN; ++nv) {
    ++z.residues;
    if (prefix[mv * (TN + 1) + nv] != mv * nv) {
      ++z.mask_bad; ok = false;
    }
  }

  // Edges plus two asymmetric interiors exercise poison and workspace reuse.
  int const ms[] = {0, 1, TM / 2, TM - 1, TM};
  int const ns[] = {0, 1, TN / 2, TN - 1, TN};
  // Bind the oracle to the production helper on every layout.  The exhaustive
  // prefix proof above covers all residues; these asymmetric edges prove that
  // the helper's physical-register order is the same order BlockStriped sees.
  for (int j = 0; j < 5; ++j) for (int t = 0; t < G::Threads; ++t) {
    typename G::Fragment accum;
    auto mask = cutlass::gemm::kernel::detail::make_accumulator_residue_mask(
        typename G::Mma{}, accum, make_shape(Int<TM>{}, Int<TN>{}),
        make_coord(ms[j], ns[4 - j]), t);
    for (int i = 0; i < G::FragmentSlots; ++i) {
      int const slot = i * G::Threads + t;
      bool const expected = lm[slot] < ms[j] && ln[slot] < ns[4 - j];
      if (mask(i) != expected) { ++z.helper_bad; ok = false; }
    }
  }
  for (int peers = 1; peers <= 4; ++peers) {
    for (int j = 0; j < 5; ++j) {
      ++z.simulations;
      if (!simulate<G, TM, TN>(lm, ln, ms[j], ns[4 - j], peers)) {
        ++z.semantic_bad; ok = false;
      }
    }
  }

  ++z.cases; z.slots += E;
  ++z.cohort_cases[G::Threads];
  z.cohort_slots[G::Threads] += E;
  z.cohort_simulations[G::Threads] += 20;
  return ok;
}

struct StripeCoverage {
  uint64_t visits = 0;
  uint64_t holes = 0;
  uint64_t duplicate_visits = 0;
  uint64_t out_of_tile = 0;

  bool exact(uint64_t expected) const {
    return visits == expected && holes == 0 && duplicate_visits == 0 &&
           out_of_tile == 0;
  }
};

// Replay the exact scalar address seam used by Marlin's fixup:
//   local = stripe * Cohort + threadIdx.x.
// The green arm is already exercised above with Cohort == G::Threads.  This
// helper exists for the independent red arm: silently falling back to the old
// 128-thread cohort must create holes/aliases for every newly admitted cohort.
template <class G, int Cohort>
StripeCoverage striped_coverage() {
  std::vector<uint16_t> visits(G::TileSlots, 0);
  StripeCoverage out;
  for (int thread = 0; thread < G::Threads; ++thread) {
    for (int stripe = 0; stripe < G::FragmentSlots; ++stripe) {
      ++out.visits;
      int const local = stripe * Cohort + thread;
      if (local < 0 || local >= G::TileSlots) {
        ++out.out_of_tile;
      } else {
        ++visits[local];
      }
    }
  }
  for (uint16_t count : visits) {
    out.holes += count == 0;
    out.duplicate_visits += count > 1 ? uint64_t(count - 1) : 0;
  }
  return out;
}

} // namespace

int main() {
  Totals z;
  bool ok = true;
#define L124_RUN(TM,TN,WM,WN,ATOM) \
  ok &= check_case<TM,TN,WM,WN,ATOM>(z);
  L124_FOR_EACH_CASE(L124_RUN)
#undef L124_RUN

  uint64_t cohort_case_sum = 0;
  int nonempty_cohorts = 0;
#define L124_CHECK_COHORT(WARPS,THREADS,LAYOUTS)                           \
  do {                                                                    \
    static_assert((THREADS) == 32 * (WARPS));                             \
    ok &= (LAYOUTS) > 0;                                                  \
    ok &= z.cohort_cases[THREADS] == uint64_t(LAYOUTS);                   \
    ok &= z.cohort_slots[THREADS] > 0;                                    \
    ok &= z.cohort_simulations[THREADS] == uint64_t(LAYOUTS) * 20;        \
    cohort_case_sum += z.cohort_cases[THREADS];                           \
    nonempty_cohorts += z.cohort_cases[THREADS] != 0;                     \
  } while (false);
  L124_FOR_EACH_COHORT(L124_CHECK_COHORT)
#undef L124_CHECK_COHORT
  ok &= nonempty_cohorts == L124_COHORT_COUNT;
  ok &= cohort_case_sum == z.cases;

  // Two planted controls.  Swapping the predicate axes must fail on this
  // rectangular tile, and shifting only workspace stores must break the
  // store/load address seam.  A green probe unable to see either is invalid.
  using Red = Geometry<8, 16, 8, 16, 8>;
  std::vector<int> lm(Red::TileSlots), ln(Red::TileSlots);
  auto id = make_identity_tensor(make_shape(_8{}, _16{}));
  for (int t = 0; t < Red::Threads; ++t) {
    auto p = typename Red::Mma{}.get_thread_slice(t).partition_C(id);
    typename Red::Fragment accum;
    auto physical_to_fragment = right_inverse(accum.layout());
    for (int i = 0; i < Red::FragmentSlots; ++i) {
      int const s = i * Red::Threads + t; auto c = p(physical_to_fragment(i));
      lm[s] = int(get<0>(c)); ln[s] = int(get<1>(c));
    }
  }
  int good = 0, swapped = 0;
  for (int s = 0; s < Red::TileSlots; ++s) {
    good += lm[s] < 3 && ln[s] < 11;
    swapped += ln[s] < 3 && lm[s] < 11;
  }
  bool const coordinate_red = good == 33 && swapped != good;
  bool const address_red = !simulate<Red, 8, 16>(lm, ln, 7, 13, 4, true);

  // These are the exact four real layouts carried by L131.  A stale fixed
  // 128-thread fixup cohort has a distinct, deterministic failure shape for
  // every one: 32 threads run out of tile; wider CTAs alias the prefix and
  // leave the rest untouched.
  using G32 = Geometry<8, 16, 8, 16, 8>;
  using G256 = Geometry<8, 128, 8, 16, 8>;
  using G512 = Geometry<8, 256, 8, 16, 8>;
  using G1024 = Geometry<32, 256, 16, 16, 16>;
  StripeCoverage const wrong32 = striped_coverage<G32, 128>();
  StripeCoverage const wrong256 = striped_coverage<G256, 128>();
  StripeCoverage const wrong512 = striped_coverage<G512, 128>();
  StripeCoverage const wrong1024 = striped_coverage<G1024, 128>();
  bool const cohort_red =
      wrong32.visits == 128 && wrong32.holes == 96 &&
      wrong32.duplicate_visits == 0 && wrong32.out_of_tile == 96 &&
      wrong256.visits == 1024 && wrong256.holes == 384 &&
      wrong256.duplicate_visits == 384 && wrong256.out_of_tile == 0 &&
      wrong512.visits == 2048 && wrong512.holes == 1152 &&
      wrong512.duplicate_visits == 1152 && wrong512.out_of_tile == 0 &&
      wrong1024.visits == 8192 && wrong1024.holes == 6272 &&
      wrong1024.duplicate_visits == 6272 && wrong1024.out_of_tile == 0;
  ok &= z.cases == L124_CASE_COUNT && coordinate_red && address_red && cohort_red;

  std::printf("L124 cases=%llu/%d source_rows=%d slots=%llu exhaustive_residues=%llu "
              "S1-4_runs=%llu coverage_bad=%llu helper_bad=%llu mask_bad=%llu semantic_bad=%llu\n",
      static_cast<unsigned long long>(z.cases), L124_CASE_COUNT,
      L124_SOURCE_ROW_COUNT,
      static_cast<unsigned long long>(z.slots),
      static_cast<unsigned long long>(z.residues),
      static_cast<unsigned long long>(z.simulations),
      static_cast<unsigned long long>(z.coverage_bad),
      static_cast<unsigned long long>(z.helper_bad),
      static_cast<unsigned long long>(z.mask_bad),
              static_cast<unsigned long long>(z.semantic_bad));
#define L124_PRINT_COHORT(WARPS,THREADS,LAYOUTS)                           \
  std::printf(" w%d/t%d=%llu", WARPS, THREADS,                           \
      static_cast<unsigned long long>(z.cohort_cases[THREADS]));
  std::printf("L124 cohort-layouts");
  L124_FOR_EACH_COHORT(L124_PRINT_COHORT)
#undef L124_PRINT_COHORT
  std::printf(" nonempty=%d/%d\n", nonempty_cohorts, L124_COHORT_COUNT);
  std::printf("L124 planted-coordinate=%s planted-address=%s "
              "planted-fixed128-cohort=%s result=%s\n",
      coordinate_red ? "EXPECTED_RED" : "UNEXPECTED_GREEN",
      address_red ? "EXPECTED_RED" : "UNEXPECTED_GREEN",
      cohort_red ? "EXPECTED_RED" : "UNEXPECTED_GREEN",
      ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
