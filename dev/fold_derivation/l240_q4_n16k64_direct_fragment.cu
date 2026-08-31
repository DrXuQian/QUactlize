// L240 -- derive the offline nibble map for the Q4 N16 x K64 direct reader
// from the complete, real CuTe chain.
//
//   plain shared [K/16=4, 2*N=32] uint32
//     -> 32-lane UniversalCopy<uint128_t>
//     -> four u32 registers/lane -> recast<int4>
//     -> the shipping int4 fast-converter emission map
//     -> the actual PPU0010 m8 fp16 partition_fragment_B destination
//
// The result is a logical (n,k) -> physical shared nibble permutation.  The
// same N16 atom must be reusable inside WN16, WN32 and WN64 compute fragments;
// a tactic-dependent answer would make it an invalid offline format.

#if defined(L240_COMPILER_PROBE)

#include <cuda_fp16.h>

__global__ void l240_compiler_probe(half const* x, half* y) {
  int const i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) y[0] = __hadd(x[0], x[0]);
}

int main() { return 0; }

#else

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "cute/atom/copy_atom.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "actlize_extensions/cutlass/quactlize_mix_gemm_convert.h"
#include "actlize_extensions/cutlass/gemm/collective/detail/quactlize_q4_n16k64_delivery.hpp"
#include "q4_n16k64_direct_offline.hpp"
#include "q4_kpack4_offline.hpp"

namespace {
using namespace cute;
namespace direct =
    cutlass::gemm::collective::detail::quactlize_q4_n16k64_delivery;
namespace offline = q4_n16k64_direct;

constexpr int kN = 16;
constexpr int kK = 64;
constexpr int kLanes = 32;
constexpr int kWordsPerLane = 4;
constexpr int kCodesPerWord = 8;
constexpr int kCodesPerLane = kWordsPerLane * kCodesPerWord;
constexpr int kPhysicalWords = 4 * 32;
constexpr int kCodes = kN * kK;

static_assert(kPhysicalWords * int(sizeof(std::uint32_t)) == 512);
static_assert(kCodes / 2 == 512);
static_assert(kLanes * kCodesPerLane == kCodes);

using Physical = direct::PhysicalShared<kN, kK>;
using PhysicalLayout = typename Physical::Layout;
using DirectProvider = direct::AiuPlainProvider<kN, kK>;
using ProductionReader = typename DirectProvider::ReaderType;
using UniversalAtom = typename ProductionReader::CopyAtom;
using UniversalReader = typename ProductionReader::Copy;

static_assert(UniversalAtom::NumValSrc == kWordsPerLane &&
              UniversalAtom::NumValDst == kWordsPerLane);
static_assert(std::is_same_v<typename ProductionReader::CopyInst,
                             UniversalCopy<cute::uint128_t>>);
static_assert(int(size(UniversalReader{})) == kLanes);
static_assert(cosize_v<PhysicalLayout> == kPhysicalWords);
static_assert(Physical::physical_k_rows == 4 &&
              Physical::physical_n_words == 32 &&
              Physical::stage_bytes == 512);

struct Mapping {
  std::array<int, kCodes> logical_to_physical{};
  int source_bad = 0;
  int destination_bad = 0;
  int physical_denominator_bad = 0;
  int logical_denominator_bad = 0;
  int cohort_mismatch = 0;
  int noncontiguous_destination = 0;
  int cohorts = 0;
};

// The semantic forward/inverse formula now belongs to the host-only artifact
// header.  L240 derives the mapping independently from the real CuTe chain and
// compares against that ABI; it must not keep a second private formula which
// could drift in parallel with a producer.
static_assert(offline::kNAtom == kN && offline::kKAtom == kK);
static_assert(offline::kCodesPerWord == kCodesPerWord);
static_assert(offline::kAtomMappingFingerprint ==
              UINT64_C(0x74443aed0cce4083));

// The actual UniversalCopy source partition, not a hand-written lane formula.
std::array<std::array<int, kWordsPerLane>, kLanes> universal_word_map(
    int* bad) {
  std::array<std::array<int, kWordsPerLane>, kLanes> out{};
  std::array<int, kPhysicalWords> hits{};
  // UniversalCopy is a one-dimensional vector copy.  The reader presents
  // the exact 512 B physical stage through its flat 128-word alias; the
  // PhysicalLayout below remains the authoritative [4,32] shared address.
  auto identity = make_identity_tensor(Shape<_128>{});
  typename UniversalReader::TiledLayout_TV tiled_tv{};
  PhysicalLayout physical{};
  for (int lane = 0; lane < kLanes; ++lane) {
    auto src = UniversalReader{}.get_slice(lane).partition_S(identity);
    if (int(size(src)) != kWordsPerLane) ++*bad;
    for (int vreg = 0; vreg < kWordsPerLane; ++vreg) {
      int const word = int(get<0>(src(vreg)));
      int const tv_word = int(tiled_tv(make_coord(lane, vreg)));
      out[std::size_t(lane)][std::size_t(vreg)] = word;
      *bad += word != tv_word;
      *bad += word < 0 || word >= kPhysicalWords;
      if (word >= 0 && word < kPhysicalWords) {
        int const row = word / 32;
        int const column = word % 32;
        *bad += int(physical(make_coord(row, column))) != word;
      }
      if (word >= 0 && word < kPhysicalWords)
        ++hits[std::size_t(word)];
    }
  }
  for (int h : hits) *bad += h != 1;
  return out;
}

// Bind the register representation to the same recast used by production.
// Four UniversalCopy u32 destinations are exactly 32 int4 converter inputs.
bool prove_register_recast() {
  auto words = make_tensor<std::uint32_t>(make_layout(Shape<_4>{}));
  auto codes = recast<cutlass::int4b_t>(words);
  bool exact = int(size(words)) == kWordsPerLane &&
               int(size(codes)) == kCodesPerLane;
  for (int i = 0; i < kCodesPerLane; ++i)
    exact &= int(codes.layout()(i)) == i;
  return exact;
}

template <int WN>
using ComputeMma = TiledMMA<
    MMA_Atom<PPU0010_8x16x16_F32F16F16F32_TN>,
    Layout<Shape<_1, _1, _1>>, Tile<_8, Int<WN>, Int<kK>>>;

template <int WN>
Mapping derive_mapping() {
  static_assert(WN == 16 || WN == 32 || WN == 64);
  static_assert(WN % kN == 0);
  constexpr int kNCohorts = WN / kN;

  Mapping result{};
  result.logical_to_physical.fill(-1);
  int universal_bad = 0;
  auto const source_words = universal_word_map(&universal_bad);
  result.source_bad += universal_bad;

  auto logical_tensor = make_tensor(
      make_smem_ptr((cutlass::half_t*)nullptr),
      make_layout(Shape<Int<WN>, Int<kK>>{}, Stride<Int<kK>, _1>{}));
  auto logical_identity = make_identity_tensor(Shape<Int<WN>, Int<kK>>{});

  std::array<int, kCodes> reference{};
  reference.fill(-1);
  for (int cohort = 0; cohort < kNCohorts; ++cohort) {
    std::array<int, kCodes> cohort_map{};
    std::array<int, kCodes> physical_hits{};
    std::array<int, kCodes> logical_hits{};
    cohort_map.fill(-1);

    for (int lane = 0; lane < kLanes; ++lane) {
      auto thr = ComputeMma<WN>{}.get_thread_slice(lane);
      auto fragment = thr.partition_fragment_B(logical_tensor);
      auto partition = thr.partition_B(logical_identity);
      auto pi = right_inverse(fragment.layout());

      std::array<int, kCodesPerLane> cohort_raw{};
      int cohort_raw_count = 0;
      for (int raw = 0; raw < int(size(fragment)); ++raw) {
        auto const nk = partition(pi(raw));
        int const n = int(get<0>(nk));
        if (n / kN == cohort) {
          if (cohort_raw_count < kCodesPerLane)
            cohort_raw[std::size_t(cohort_raw_count)] = raw;
          ++cohort_raw_count;
        }
      }
      if (cohort_raw_count != kCodesPerLane) {
        ++result.destination_bad;
        continue;
      }
      for (int i = 0; i < kCodesPerLane; ++i)
        for (int j = i + 1; j < kCodesPerLane; ++j)
          if (cohort_raw[std::size_t(j)] < cohort_raw[std::size_t(i)]) {
            int const tmp = cohort_raw[std::size_t(i)];
            cohort_raw[std::size_t(i)] = cohort_raw[std::size_t(j)];
            cohort_raw[std::size_t(j)] = tmp;
          }
      int const destination_base = cohort_raw[0];
      for (int i = 0; i < kCodesPerLane; ++i)
        result.noncontiguous_destination +=
            cohort_raw[std::size_t(i)] != destination_base + i;

      for (int vreg = 0; vreg < kWordsPerLane; ++vreg) {
        int const word =
            source_words[std::size_t(lane)][std::size_t(vreg)];
        for (int code = 0; code < kCodesPerWord; ++code) {
          int const physical_nibble = kCodesPerWord * word + code;
          int const emitted = cutlass::MixGemmEmit<4>::index(code, vreg);
          int const raw = destination_base + emitted;
          if (raw < 0 || raw >= int(size(fragment))) {
            ++result.destination_bad;
            continue;
          }
          auto const nk = partition(pi(raw));
          int const local_n = int(get<0>(nk)) - cohort * kN;
          int const k = int(get<1>(nk));
          int const logical = local_n * kK + k;
          if (local_n < 0 || local_n >= kN || k < 0 || k >= kK ||
              logical < 0 || logical >= kCodes ||
              physical_nibble < 0 || physical_nibble >= kCodes) {
            ++result.destination_bad;
            continue;
          }
          if (cohort_map[std::size_t(logical)] >= 0)
            ++result.logical_denominator_bad;
          cohort_map[std::size_t(logical)] = physical_nibble;
          ++logical_hits[std::size_t(logical)];
          ++physical_hits[std::size_t(physical_nibble)];
        }
      }
    }

    for (int i = 0; i < kCodes; ++i) {
      result.logical_denominator_bad += logical_hits[std::size_t(i)] != 1;
      result.physical_denominator_bad += physical_hits[std::size_t(i)] != 1;
    }
    if (cohort == 0) {
      reference = cohort_map;
    } else {
      for (int i = 0; i < kCodes; ++i)
        result.cohort_mismatch +=
            cohort_map[std::size_t(i)] != reference[std::size_t(i)];
    }
    ++result.cohorts;
  }
  result.logical_to_physical = reference;
  return result;
}

std::uint64_t mapping_id(std::array<int, kCodes> const& map) {
  // FNV-1a over a fixed little-endian uint16 representation.
  std::uint64_t h = UINT64_C(1469598103934665603);
  for (int x : map) {
    std::uint16_t const v = std::uint16_t(x);
    h ^= std::uint8_t(v & 0xffu);
    h *= UINT64_C(1099511628211);
    h ^= std::uint8_t(v >> 8);
    h *= UINT64_C(1099511628211);
  }
  return h;
}

bool structurally_exact(Mapping const& m) {
  return m.source_bad == 0 && m.destination_bad == 0 &&
         m.physical_denominator_bad == 0 &&
         m.logical_denominator_bad == 0 && m.cohort_mismatch == 0 &&
         m.noncontiguous_destination == 0;
}

int formula_bad(Mapping const& m) {
  int bad = 0;
  for (int n = 0; n < kN; ++n) {
    for (int k = 0; k < kK; ++k) {
      int const logical = n * kK + k;
      int const p = m.logical_to_physical[std::size_t(logical)];
      bad += p != offline::atom_physical_nibble(n, k);
      offline::LogicalCoordinate const inverse =
          offline::atom_logical_coordinate(p);
      bad += inverse.n != n || inverse.k != k;
    }
  }
  return bad;
}

struct OuterAbiComparison {
  int direct_denominator_bad = 0;
  int layout1_denominator_bad = 0;
  int header_mismatch = 0;
  int equal = 0;
  int total = 0;
};

// Repeat the proved N16xK64 atom through the production plain-shared layout
// at N32xK128, then compare every nibble with the already-shipping Layout1
// K-pack4 checkpoint ABI.  Equal byte counts are not enough: this is the
// pointwise offline address comparison.
OuterAbiComparison compare_existing_layout1(Mapping const& atom) {
  constexpr int N = 32;
  constexpr int K = 128;
  constexpr int Total = N * K;
  using OuterPhysical = direct::PhysicalShared<N, K>;
  typename OuterPhysical::Layout outer{};
  std::array<int, Total> direct_hits{};
  std::array<int, Total> layout1_hits{};
  OuterAbiComparison out{};

  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      int const local_n = n % kN;
      int const local_k = k % kK;
      int const local_p = atom.logical_to_physical[
          std::size_t(local_n * kK + local_k)];
      int const local_word = local_p / kCodesPerWord;
      int const local_nibble = local_p % kCodesPerWord;
      int const local_n_word = local_word % 32;
      int const local_k_row = local_word / 32;
      int const n_word = (n / kN) * 32 + local_n_word;
      int const k_row = (k / kK) * 4 + local_k_row;
      int const direct_p =
          kCodesPerWord * int(outer(make_coord(k_row, n_word))) +
          local_nibble;
      int const header_p =
          int(offline::physical_nibble(n, k, N));
      int const layout1_p =
          q4_kpack4::kPack *
              int(q4_kpack4::word_index(n, k, N)) +
          q4_kpack4::nibble(k);
      if (direct_p >= 0 && direct_p < Total)
        ++direct_hits[std::size_t(direct_p)];
      else
        ++out.direct_denominator_bad;
      if (layout1_p >= 0 && layout1_p < Total)
        ++layout1_hits[std::size_t(layout1_p)];
      else
        ++out.layout1_denominator_bad;
      out.header_mismatch += direct_p != header_p;
      out.equal += direct_p == layout1_p;
      ++out.total;
    }
  }
  for (int h : direct_hits) out.direct_denominator_bad += h != 1;
  for (int h : layout1_hits) out.layout1_denominator_bad += h != 1;
  return out;
}

std::array<int, kCodes> candidate_offline_map(Mapping const& actual) {
  auto candidate = actual.logical_to_physical;
#if defined(L240_PLANT_SOURCE_EQUALS_DEST) && L240_PLANT_SOURCE_EQUALS_DEST
  // Wrong but tempting: make physical register order equal to destination
  // order, erasing the converter's nonidentity MixGemmEmit permutation.
  candidate.fill(-1);
  for (int lane = 0; lane < kLanes; ++lane) {
    for (int vreg = 0; vreg < kWordsPerLane; ++vreg) {
      for (int code = 0; code < kCodesPerWord; ++code) {
        int const source = kCodesPerWord * (kWordsPerLane * lane + vreg) + code;
        int const emitted = cutlass::MixGemmEmit<4>::index(code, vreg);
        int const actual_logical = [&] {
          for (int logical = 0; logical < kCodes; ++logical)
            if (actual.logical_to_physical[std::size_t(logical)] == source)
              return logical;
          return -1;
        }();
        int const wrong_physical = kCodesPerLane * lane + emitted;
        if (actual_logical >= 0)
          candidate[std::size_t(actual_logical)] = wrong_physical;
      }
    }
  }
#elif defined(L240_PLANT_ROTATE_COHORT) && L240_PLANT_ROTATE_COHORT
  // A bijective, byte-count-preserving one-vector rotation.  This is a RED
  // for importing an unrelated reader's cohort permutation.
  for (int& physical : candidate)
    physical = (physical + kCodesPerLane) % kCodes;
#endif
  return candidate;
}

int fixture_bad(Mapping const& actual) {
  auto const candidate = candidate_offline_map(actual);
  std::array<int, kCodes> physical{};
  std::array<int, kCodes> output{};
  std::array<int, kCodes> candidate_hits{};
  physical.fill(-1);
  output.fill(-1);
  int bad = 0;
  for (int logical = 0; logical < kCodes; ++logical) {
    int const p = candidate[std::size_t(logical)];
    if (p < 0 || p >= kCodes) {
      ++bad;
      continue;
    }
    ++candidate_hits[std::size_t(p)];
    physical[std::size_t(p)] = logical;
  }
  for (int h : candidate_hits) bad += h != 1;
  for (int logical = 0; logical < kCodes; ++logical) {
    int const p = actual.logical_to_physical[std::size_t(logical)];
    if (p < 0 || p >= kCodes) {
      ++bad;
      continue;
    }
    output[std::size_t(logical)] = physical[std::size_t(p)];
    bad += output[std::size_t(logical)] != logical;
  }
  return bad;
}

template <int WN>
bool report(Mapping const& m, std::uint64_t reference_id) {
  std::uint64_t const id = mapping_id(m.logical_to_physical);
  int const numeric_bad = fixture_bad(m);
  int const reversible_bad = formula_bad(m);
  bool const ok = structurally_exact(m) && id == reference_id &&
                  numeric_bad == 0 && reversible_bad == 0;
  std::printf(
      "L240 FRAGMENT WN=%d cohorts=%d mapping=0x%016llx "
      "source_bad=%d destination_bad=%d physical_denominator_bad=%d "
      "logical_denominator_bad=%d cohort_mismatch=%d "
      "noncontiguous_destination=%d fixture_bad=%d reversible_bad=%d "
      "result=%s\n",
      WN, m.cohorts, static_cast<unsigned long long>(id), m.source_bad,
      m.destination_bad, m.physical_denominator_bad,
      m.logical_denominator_bad, m.cohort_mismatch,
      m.noncontiguous_destination, numeric_bad, reversible_bad,
      ok ? "PASS" : "FAIL");
  return ok;
}

void print_anchors(Mapping const& m) {
  std::printf("L240 MAP_ANCHOR n0_k0_15=");
  for (int k = 0; k < 16; ++k)
    std::printf("%s%d", k == 0 ? "" : ",",
                m.logical_to_physical[std::size_t(k)]);
  std::printf(" n15_k48_63=");
  for (int k = 48; k < 64; ++k)
    std::printf("%s%d", k == 48 ? "" : ",",
                m.logical_to_physical[std::size_t(15 * kK + k)]);
  std::putchar('\n');
#if defined(L240_DUMP_MAP) && L240_DUMP_MAP
  for (int n = 0; n < kN; ++n) {
    std::printf("L240 MAP n=%d values=", n);
    for (int k = 0; k < kK; ++k)
      std::printf("%s%d", k == 0 ? "" : ",",
                  m.logical_to_physical[std::size_t(n * kK + k)]);
    std::putchar('\n');
  }
#endif
}

}  // namespace

int main() {
  Mapping const w16 = derive_mapping<16>();
  Mapping const w32 = derive_mapping<32>();
  Mapping const w64 = derive_mapping<64>();
  std::uint64_t const id = mapping_id(w16.logical_to_physical);
  bool const recast_exact = prove_register_recast();
  OuterAbiComparison const outer = compare_existing_layout1(w16);
  bool ok = recast_exact;
  ok &= id == offline::kAtomMappingFingerprint;
  ok &= report<16>(w16, offline::kAtomMappingFingerprint);
  ok &= report<32>(w32, offline::kAtomMappingFingerprint);
  ok &= report<64>(w64, offline::kAtomMappingFingerprint);
  ok &= outer.direct_denominator_bad == 0 &&
        outer.layout1_denominator_bad == 0 && outer.header_mismatch == 0 &&
        outer.total == 4096 &&
        outer.equal != outer.total;
  print_anchors(w16);
  std::printf(
      "L240 REVERSIBLE_MAP bitperm="
      "p0:k3,p1:k4,p2:k0,p3:k5,p4:n3,p5:k1,p6:k2,p7:n0,p8:n1,p9:n2 "
      "inverse=EXACT\n");
  std::printf(
      "L240 OFFLINE_ABI shape=32x128 direct_denominator_bad=%d "
      "layout1_denominator_bad=%d header_mismatch=%d equal=%d/%d "
      "verdict=%s\n",
      outer.direct_denominator_bad, outer.layout1_denominator_bad,
      outer.header_mismatch, outer.equal, outer.total,
      outer.equal == outer.total ? "REUSES_LAYOUT1" :
                                   "DISTINCT_REQUIRES_NEW_MAPPING");
  std::printf(
      "L240 Q4_N16K64_DIRECT_FRAGMENT %s mapping=0x%016llx "
      "bijection=1024/1024 recast=%s outer=32x128 geometries=3 reds=2\n",
      ok ? "PASS" : "FAIL", static_cast<unsigned long long>(id),
      recast_exact ? "EXACT" : "BAD");
  return ok ? 0 : 1;
}

#endif
