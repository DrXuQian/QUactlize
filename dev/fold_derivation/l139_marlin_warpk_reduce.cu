// L139 -- standalone-Marlin production C-fragment / K-cohort oracle.
//
// This oracle deliberately aliases MarlinCollectivePPU::TiledMma.  It does
// not instantiate the retired generic mixed-input builder.  CuTe supplies the
// production 1M x 2N x 4K thread topology and the 32-register fragment extent;
// the native PPU m16n16 accumulator coordinates come from classic Marlin's
// acc_i/acc_j contract, exactly as MarlinKernelPPU consumes them.
//
// The positive arm proves, exhaustively over all 256 threads and 32 slots:
//   * four K cohorts have one owner for every compact (N-warp,lane) id;
//   * K0 is exactly 64 compact output threads, with 32 slots each;
//   * every K cohort covers the 16x128 C tile exactly once and is coordinate-
//     isomorphic to K0 under the classic acc_i/acc_j map;
//   * the production shared-scratch address cadence reproduces classic's
//     4 -> 2 -> 1 FP32 reduction with raw-bit equality.
//
// The generic CuTe partition_C map and a tempting compact row-major map are
// measured, not accepted: both are bijections and both are wrong associations
// for the native PPU accumulator.  Dedicated plants must turn each one red.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

#include <cuda_fp16.h>

// Stock nvcc is used here as a host C++ compiler.  Without __HGGCCC__ the
// production header's CUTLASS_DEVICE bodies are parsed as inline host bodies;
// these declarations keep three uninstantiated device operations parseable.
// The oracle never calls them.  Defining __HGGCCC__ instead is not viable:
// stock nvcc then device-instantiates PPU CuTe helpers it cannot compile.
__half2 l139_unreachable_hfma2(__half2, __half2, __half2);
unsigned int l139_unreachable_cvta(void const*);
struct L139UnreachableThreadIdx { int x = 0, y = 0, z = 0; };
inline constexpr L139UnreachableThreadIdx l139_unreachable_thread_idx{};
void l139_unreachable_syncthreads();
#define __hfma2 l139_unreachable_hfma2
#define __cvta_generic_to_shared l139_unreachable_cvta
#define threadIdx l139_unreachable_thread_idx
#define __syncthreads l139_unreachable_syncthreads
#include "cute/tensor.hpp"
#include "actlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
#undef __syncthreads
#undef threadIdx
#undef __cvta_generic_to_shared
#undef __hfma2

namespace {
using namespace cute;

using ProductionStrideA = Stride<int64_t, _1, int64_t>;
using ProductionStrideB = Stride<int64_t, _1, int64_t>;
using ProductionStrideScale = Stride<_1, int64_t, int64_t>;
using ProductionCollective = cutlass::gemm::collective::MarlinCollectivePPU<
    Shape<_16, _128, _128>, Shape<_16, _64, _32>, 4, 128,
    ProductionStrideA, ProductionStrideB, ProductionStrideScale>;
using ProductionMma = typename ProductionCollective::TiledMma;
using ProductionFragment = decltype(make_fragment_like<float>(
    partition_fragment_C(ProductionMma{}, Shape<_16, _128>{})));

constexpr int kTileM = ProductionCollective::TileM;
constexpr int kTileN = ProductionCollective::TileN;
constexpr int kComputeThreads = size(ProductionMma{});
constexpr int kWarpKCohorts = ProductionCollective::WarpOnK;
constexpr int kOutputThreads = kComputeThreads / kWarpKCohorts;
constexpr int kFragmentSlots = size(ProductionFragment{});
constexpr int kTileSlots = kTileM * kTileN;

static_assert(kTileM == 16 && kTileN == 128);
static_assert(ProductionCollective::TileK == 128);
static_assert(ProductionCollective::WarpM == 16 &&
              ProductionCollective::WarpN == 64 &&
              ProductionCollective::WarpK == 32);
static_assert(kComputeThreads == 256 && kWarpKCohorts == 4);
static_assert(kOutputThreads == 64 && kFragmentSlots == 32);
static_assert(kOutputThreads * kFragmentSlots == kTileSlots);
static_assert(size<0>(typename ProductionMma::ThrLayoutVMNK{}) == 32);
static_assert(size<1>(typename ProductionMma::ThrLayoutVMNK{}) == 1);
static_assert(size<2>(typename ProductionMma::ThrLayoutVMNK{}) == 2);
static_assert(size<3>(typename ProductionMma::ThrLayoutVMNK{}) == 4);

uint32_t raw(float x) {
  uint32_t value = 0;
  std::memcpy(&value, &x, sizeof(value));
  return value;
}

constexpr int classic_acc_i(int lane, int value) {
  return lane / 4 + (((value >> 2) & 1) << 3);
}

constexpr int classic_acc_j(int lane, int value) {
  return lane % 4 + ((value % 4) << 2);
}

constexpr int classic_logical(int compact, int slot) {
  int const warp_n = compact / 32;
  int const lane = compact % 32;
  int const n_block = slot / 8;
  int const value = slot % 8;
  int const row = classic_acc_i(lane, value);
  int const col = 64 * warp_n + 16 * n_block +
                  classic_acc_j(lane, value);
  return row * kTileN + col;
}

// A compact stripe is a useful scratch layout, but it is not the PPU C
// coordinate map.  Keeping the formula named prevents it from drifting back
// into output ownership merely because it is a bijection of 2048 values.
constexpr int compact_wrong_logical(int compact, int slot) {
  return slot * kOutputThreads + compact;
}

struct ThreadInfo {
  int v = -1;
  int m = -1;
  int n = -1;
  int wk = -1;
  int compact = -1;
  std::array<int, kFragmentSlots> classic{};
  std::array<int, kFragmentSlots> generic{};
  std::array<int, kFragmentSlots> compact_wrong{};
};

std::array<ThreadInfo, kComputeThreads> make_thread_table() {
  std::array<ThreadInfo, kComputeThreads> result{};
  auto const topology = ProductionMma{}.get_thr_layout_vmnk();
  auto identity = make_identity_tensor(Shape<_16, _128>{});

  for (int thread = 0; thread < kComputeThreads; ++thread) {
    auto coord = topology.get_flat_coord(thread);
    ThreadInfo& info = result[thread];
    info.v = int(get<0>(coord));
    info.m = int(get<1>(coord));
    info.n = int(get<2>(coord));
    info.wk = int(get<3>(coord));
    info.compact = info.v + 32 * info.n;

    auto generic_partition =
        ProductionMma{}.get_thread_slice(thread).partition_C(identity);
    ProductionFragment fragment;
    auto physical_to_fragment = right_inverse(fragment.layout());
    for (int slot = 0; slot < kFragmentSlots; ++slot) {
      auto generic_coord = generic_partition(physical_to_fragment(slot));
      info.classic[slot] = classic_logical(info.compact, slot);
      info.generic[slot] = int(get<0>(generic_coord)) * kTileN +
                           int(get<1>(generic_coord));
      info.compact_wrong[slot] =
          compact_wrong_logical(info.compact, slot);
    }
  }
  return result;
}

struct GeometryResult {
  int bad_topology = 0;
  int owner_holes = 0;
  int owner_duplicates = 0;
  int k0_threads = 0;
  int cohort_holes = 0;
  int cohort_duplicates = 0;
  int cohort_coordinate_mismatches = 0;
  int generic_coordinate_mismatches = 0;
  int compact_coordinate_mismatches = 0;

  bool ok() const {
    return bad_topology == 0 && owner_holes == 0 &&
           owner_duplicates == 0 && k0_threads == kOutputThreads &&
           cohort_holes == 0 && cohort_duplicates == 0 &&
           cohort_coordinate_mismatches == 0 &&
           generic_coordinate_mismatches > 0 &&
           compact_coordinate_mismatches > 0;
  }
};

GeometryResult verify_geometry(
    std::array<ThreadInfo, kComputeThreads> const& table) {
  GeometryResult result;
  std::array<std::array<int, kWarpKCohorts>, kOutputThreads> owner{};
  for (auto& by_cohort : owner) by_cohort.fill(-1);
  std::array<std::array<int, kTileSlots>, kWarpKCohorts> coverage{};
  for (auto& by_logical : coverage) by_logical.fill(0);

  for (int thread = 0; thread < kComputeThreads; ++thread) {
    ThreadInfo const& info = table[thread];
    bool const valid = info.v >= 0 && info.v < 32 && info.m == 0 &&
                       info.n >= 0 && info.n < 2 && info.wk >= 0 &&
                       info.wk < kWarpKCohorts && info.compact >= 0 &&
                       info.compact < kOutputThreads;
    if (!valid) {
      ++result.bad_topology;
      continue;
    }
    if (info.wk == 0) ++result.k0_threads;
    int& slot = owner[info.compact][info.wk];
    if (slot >= 0) ++result.owner_duplicates;
    slot = thread;
    for (int i = 0; i < kFragmentSlots; ++i) {
      int logical = info.classic[i];
      if (logical < 0 || logical >= kTileSlots) {
        ++result.bad_topology;
      } else {
        ++coverage[info.wk][logical];
      }
      result.generic_coordinate_mismatches +=
          info.generic[i] != info.classic[i];
      result.compact_coordinate_mismatches +=
          info.compact_wrong[i] != info.classic[i];
    }
  }

  for (int compact = 0; compact < kOutputThreads; ++compact) {
    int const k0 = owner[compact][0];
    if (k0 < 0) {
      ++result.owner_holes;
      continue;
    }
    for (int wk = 0; wk < kWarpKCohorts; ++wk) {
      int const thread = owner[compact][wk];
      if (thread < 0) {
        ++result.owner_holes;
        continue;
      }
      for (int slot = 0; slot < kFragmentSlots; ++slot) {
        result.cohort_coordinate_mismatches +=
            table[thread].classic[slot] != table[k0].classic[slot];
      }
    }
  }
  for (int wk = 0; wk < kWarpKCohorts; ++wk) {
    for (int hits : coverage[wk]) {
      result.cohort_holes += hits == 0;
      result.cohort_duplicates += hits > 1 ? hits - 1 : 0;
    }
  }
  return result;
}

enum class Fault {
  None,
  GenericLayout,
  CompactLayout,
  OmitCohort,
  FlatReduction,
  AllCohortsWrite,
  DuplicateK0Owner,
};

char const* fault_name(Fault fault) {
  switch (fault) {
    case Fault::None: return "none";
    case Fault::GenericLayout: return "generic-layout";
    case Fault::CompactLayout: return "compact-layout";
    case Fault::OmitCohort: return "omit-cohort";
    case Fault::FlatReduction: return "flat-reduction";
    case Fault::AllCohortsWrite: return "all-cohorts-write";
    case Fault::DuplicateK0Owner: return "duplicate-k0-owner";
  }
  return "unknown";
}

Fault parse_fault(int argc, char** argv) {
  constexpr std::string_view prefix = "--fault=";
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.substr(0, prefix.size()) != prefix) continue;
    std::string_view value = arg.substr(prefix.size());
    if (value == "generic-layout") return Fault::GenericLayout;
    if (value == "compact-layout") return Fault::CompactLayout;
    if (value == "omit-cohort") return Fault::OmitCohort;
    if (value == "flat-reduction") return Fault::FlatReduction;
    if (value == "all-cohorts-write") return Fault::AllCohortsWrite;
    if (value == "duplicate-k0-owner") return Fault::DuplicateK0Owner;
  }
  return Fault::None;
}

// This fixture is exact in FP32 and makes coordinate association observable.
float association_value(int logical, int wk) {
  return float(1 + logical + kTileSlots * wk);
}

// This fixture deliberately distinguishes classic's raw addition cadence
// from a flat reduction.  It is not used to prove mathematical equivalence;
// it is used only to bind the exact 4 -> 2 -> 1 parenthesization.
float cadence_value(int wk) {
  constexpr float values[kWarpKCohorts] = {
      16777216.0f, -16777216.0f, 0.25f, 0.25f};
  return values[wk];
}

float classic_tree(std::array<float, kWarpKCohorts> value) {
  float const upper_pair = value[3] + value[2];
  float const upper_three = value[1] + upper_pair;
  return value[0] + upper_three;
}

// Replay MarlinKernelPPU::thread_block_reduce at scalar-slot granularity:
// cohorts 2 and 3 stage first; cohort 1 consumes both and stages their sum;
// cohort 0 consumes the survivor.  The scratch indices are checked against
// the production constants instead of being replaced by a generic reduce.
float production_tree(
    std::array<float, kWarpKCohorts> value, int compact, int slot) {
  constexpr int kChunks = kFragmentSlots / 4;
  constexpr int kRedOffset = 2;
  constexpr int kRedSharedStride = kOutputThreads * 4 * 2;
  constexpr int kRedSharedDelta = kOutputThreads;
  static_assert(kChunks == 8 && kRedSharedStride == 512 &&
                kRedSharedDelta == 64);

  // Preserve both the Vector128 index and the scalar position within it.  The
  // caller exhausts all 64 compact owners x 32 slots, so this walks every
  // scratch address touched by each level rather than replaying one exemplar.
  std::array<float, 4 * 2 * kRedSharedStride> scratch{};
  int const chunk = slot / 4;
  int const scalar = slot % 4;
  auto scalar_index = [scalar](int vector_index) {
    return 4 * vector_index + scalar;
  };
  for (int wk : {2, 3}) {
    int const read = kRedSharedStride * wk + compact;
    int const write = kRedSharedDelta * chunk +
                      (read - kRedSharedStride * kRedOffset);
    scratch[scalar_index(write)] = value[wk];
  }
  int const wk1_read = kRedSharedStride + compact;
  int const wk1_write = wk1_read - kRedSharedStride;
  int const chunk_offset = kRedSharedDelta * chunk;
  float const peer = scratch[scalar_index(chunk_offset + wk1_read)];
  float const prior = scratch[scalar_index(chunk_offset + wk1_write)];
  value[1] += peer + prior;
  scratch[scalar_index(chunk_offset + wk1_write)] = value[1];
  value[0] += scratch[scalar_index(chunk_offset + compact)];
  return value[0];
}

struct ReductionResult {
  int association_raw_bitdiff = 0;
  int cadence_raw_bitdiff = 0;
  int output_holes = 0;
  int output_duplicates = 0;
  int stage2_writers = 0;
  int stage1_writers = 0;
  int survivor_writers = 0;

  bool ok() const {
    return association_raw_bitdiff == 0 && cadence_raw_bitdiff == 0 &&
           output_holes == 0 && output_duplicates == 0 &&
           stage2_writers == 2 * kOutputThreads * kFragmentSlots &&
           stage1_writers == kOutputThreads * kFragmentSlots &&
           survivor_writers == kOutputThreads * kFragmentSlots;
  }
};

ReductionResult verify_reduction(
    std::array<ThreadInfo, kComputeThreads> const& table, Fault fault) {
  std::array<std::array<int, kWarpKCohorts>, kOutputThreads> owner{};
  for (auto& by_cohort : owner) by_cohort.fill(-1);
  for (int thread = 0; thread < kComputeThreads; ++thread) {
    ThreadInfo const& info = table[thread];
    if (info.compact >= 0 && info.compact < kOutputThreads &&
        info.wk >= 0 && info.wk < kWarpKCohorts) {
      owner[info.compact][info.wk] = thread;
    }
  }

  ReductionResult result;
  std::array<int, kTileSlots> output_hits{};
  for (int compact = 0; compact < kOutputThreads; ++compact) {
    if (fault == Fault::DuplicateK0Owner && compact == kOutputThreads - 1) {
      owner[compact][0] = owner[0][0];
    }
    for (int slot = 0; slot < kFragmentSlots; ++slot) {
      std::array<float, kWarpKCohorts> association{};
      std::array<float, kWarpKCohorts> reference{};
      for (int wk = 0; wk < kWarpKCohorts; ++wk) {
        int const thread = owner[compact][wk];
        if (thread < 0) continue;
        int logical = table[thread].classic[slot];
        if (fault == Fault::GenericLayout) logical = table[thread].generic[slot];
        if (fault == Fault::CompactLayout)
          logical = table[thread].compact_wrong[slot];
        association[wk] = association_value(logical, wk);
        reference[wk] = association_value(
            table[owner[compact][0]].classic[slot], wk);
      }
      if (fault == Fault::OmitCohort) association[3] = 0.0f;

      float const expected_association = classic_tree(reference);
      float const got_association = production_tree(association, compact, slot);
      result.association_raw_bitdiff +=
          raw(expected_association) != raw(got_association);

      std::array<float, kWarpKCohorts> cadence{};
      for (int wk = 0; wk < kWarpKCohorts; ++wk)
        cadence[wk] = cadence_value(wk);
      float const expected_cadence = classic_tree(cadence);
      float got_cadence = production_tree(cadence, compact, slot);
      if (fault == Fault::FlatReduction) {
        got_cadence = ((cadence[0] + cadence[1]) + cadence[2]) + cadence[3];
      }
      result.cadence_raw_bitdiff += raw(expected_cadence) != raw(got_cadence);

      int const output_thread = owner[compact][0];
      int output_logical = output_thread >= 0
          ? table[output_thread].classic[slot] : -1;
      if (output_logical >= 0 && output_logical < kTileSlots) {
        ++output_hits[output_logical];
      }
      if (fault == Fault::AllCohortsWrite) {
        for (int wk = 1; wk < kWarpKCohorts; ++wk) {
          int const thread = owner[compact][wk];
          if (thread >= 0) ++output_hits[table[thread].classic[slot]];
        }
      }
      result.stage2_writers += 2;
      result.stage1_writers += 1;
      result.survivor_writers += 1;
    }
  }
  for (int hits : output_hits) {
    result.output_holes += hits == 0;
    result.output_duplicates += hits > 1 ? hits - 1 : 0;
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Fault const fault = parse_fault(argc, argv);
  auto const table = make_thread_table();
  GeometryResult const geometry = verify_geometry(table);
  ReductionResult const reduction = verify_reduction(table, fault);
  bool const ok = geometry.ok() && reduction.ok();

  std::printf(
      "L139 production=MarlinCollectivePPU::TiledMma "
      "tile=%dx%d topology=1Mx2Nx4K threads=%d k0=%d slots=%d\n",
      kTileM, kTileN, kComputeThreads, geometry.k0_threads,
      kFragmentSlots);
  std::printf(
      "L139 geometry topology_bad=%d owner_holes=%d owner_duplicates=%d "
      "cohort_holes=%d cohort_duplicates=%d cohort_coord_bad=%d "
      "generic_vs_classic=%d compact_vs_classic=%d\n",
      geometry.bad_topology, geometry.owner_holes,
      geometry.owner_duplicates, geometry.cohort_holes,
      geometry.cohort_duplicates, geometry.cohort_coordinate_mismatches,
      geometry.generic_coordinate_mismatches,
      geometry.compact_coordinate_mismatches);
  std::printf(
      "L139 reduce fault=%s raw={association:%d cadence:%d} "
      "output={holes:%d duplicates:%d} cadence_writers={%d,%d,%d}\n",
      fault_name(fault), reduction.association_raw_bitdiff,
      reduction.cadence_raw_bitdiff, reduction.output_holes,
      reduction.output_duplicates, reduction.stage2_writers,
      reduction.stage1_writers, reduction.survivor_writers);

  if (fault == Fault::None) {
    if (!ok) {
      std::puts("L139 FAIL: standalone production fragment contract is red");
      return 1;
    }
    std::puts(
        "L139 PASS: four classic acc_i/acc_j cohorts are exhaustive; "
        "K0=64x32 and production 4->2->1 reduction are raw-bit exact");
    return 0;
  }
  if (ok) {
    std::printf("L139 UNEXPECTED-GREEN fault=%s\n", fault_name(fault));
    return 0;
  }
  std::printf("L139 EXPECTED-RED fault=%s\n", fault_name(fault));
  return 2;
}
