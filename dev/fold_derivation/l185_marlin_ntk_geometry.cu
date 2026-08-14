// L185 -- device-free exhaustive N/K geometry oracle for standalone Marlin.
//
// This binds the production collective/TiledMma/output-map types.  It proves
// the seven admitted (TN,TK,WN,WK) geometries over both PPU M atoms and every stage:
// A vector producers, B/A logical K agreement, output-cohort ownership,
// shared ledgers, and gs128 scale cadence across non-zero phases/ring wraps.

// No device kernel is launched.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>

__half2 l185_unreachable_hfma2(__half2, __half2, __half2);
unsigned int l185_unreachable_cvta(void const*);
struct L185UnreachableThreadIdx { int x = 0, y = 0, z = 0; };
inline constexpr L185UnreachableThreadIdx l185_unreachable_thread_idx{};
void l185_unreachable_syncthreads();
#define __hfma2 l185_unreachable_hfma2
#define __cvta_generic_to_shared l185_unreachable_cvta
#define threadIdx l185_unreachable_thread_idx
#define __syncthreads l185_unreachable_syncthreads
#include "marlin_tactic_space_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/marlin_collective_ppu.hpp"
#include "quactlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp"
#undef __syncthreads
#undef threadIdx
#undef __cvta_generic_to_shared
#undef __hfma2

namespace mt = marlin_tactics_ppu;
namespace cg = cutlass::gemm::collective;
namespace kg = cutlass::gemm::kernel::marlin_ppu_detail;

namespace {

enum class Plant {
  None,
  OldM8KCadence,
  DropLastARound,
  ZeroScalePhase,
  LocalOutputTile,
};

Plant parse_plant(char const* value) {
  if (value == nullptr || !std::strcmp(value, "none")) return Plant::None;
  if (!std::strcmp(value, "old-m8-k-cadence")) return Plant::OldM8KCadence;
  if (!std::strcmp(value, "drop-last-a-round")) return Plant::DropLastARound;
  if (!std::strcmp(value, "zero-scale-phase")) return Plant::ZeroScalePhase;
  if (!std::strcmp(value, "local-output-tile")) return Plant::LocalOutputTile;
  return static_cast<Plant>(-1);
}

char const* plant_name(Plant plant) {
  switch (plant) {
    case Plant::None: return "none";
    case Plant::OldM8KCadence: return "old-m8-k-cadence";
    case Plant::DropLastARound: return "drop-last-a-round";
    case Plant::ZeroScalePhase: return "zero-scale-phase";
    case Plant::LocalOutputTile: return "local-output-tile";
  }
  return "unknown";
}

template <int TM, int TN, int TK, int WN, int WK, int Stages>
using Main = cg::MarlinCollectivePPU<
    cute::Shape<cute::Int<TM>, cute::Int<TN>, cute::Int<TK>>,
    cute::Shape<cute::Int<TM>, cute::Int<WN>, cute::Int<WK>>, Stages, 128,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<int64_t, cute::_1, int64_t>,
    cute::Stride<cute::_1, int64_t, int64_t>>;

template <class M>
using ExpectedMma = cute::TiledMMA<
    typename M::MmaAtom,
    cute::Layout<cute::Shape<
        cute::_1, cute::Int<M::WarpOnN>, cute::Int<M::WarpOnK>>>,
    cute::Tile<
        cute::Int<M::InstructionM>, cute::Int<16 * M::WarpOnN>,
        cute::Int<16 * M::WarpOnK>>>;

struct Totals {
  uint64_t types = 0;
  uint64_t a_cases = 0;
  uint64_t a_vectors = 0;
  uint64_t k_pairs = 0;
  uint64_t output_cells = 0;
  uint64_t scale_cases = 0;
  uint64_t scale_tiles = 0;
};

struct Failure {
  bool failed = false;
  char const* category = "";
  int tm = 0, tn = 0, tk = 0, stages = 0;
  int a = 0, b = 0, c = 0;

  void set(char const* why, int tm_, int tn_, int tk_, int stages_,
           int a_ = 0, int b_ = 0, int c_ = 0) {
    if (failed) return;
    failed = true;
    category = why;
    tm = tm_; tn = tn_; tk = tk_; stages = stages_;
    a = a_; b = b_; c = c_;
  }
};

template <class M>
void verify_a_producer(Plant plant, Totals& totals, Failure& failure) {
  int const max_m = M::InstructionM == 8 ? 1 : M::TileM;
  for (int problem_m = 1; problem_m <= max_m; ++problem_m) {
    ++totals.a_cases;
    std::vector<int> logical(std::size_t(M::ASharedStage), 0);
    std::vector<int> physical(std::size_t(M::ASharedStage), 0);
    int rounds = M::ASharedWriteIters;
    if (plant == Plant::DropLastARound && M::TileN == 64 &&
        M::TileK == 128 && M::InstructionM == 16) {
      --rounds;
    }
    for (int round = 0; round < rounds; ++round) {
      for (int tid = 0; tid < M::Threads; ++tid) {
        int const linear = M::a_producer_linear(round, tid);
        if (!M::a_producer_active(linear, problem_m)) continue;
        if (linear < 0 || linear >= M::ASharedStage) {
          failure.set("a-logical-range", M::TileM, M::TileN, M::TileK,
                      M::Stages, problem_m, round, tid);
          return;
        }
        int const transformed = M::transform_a_index(linear);
        if (transformed < 0 || transformed >= M::ASharedStage) {
          failure.set("a-physical-range", M::TileM, M::TileN, M::TileK,
                      M::Stages, problem_m, linear, transformed);
          return;
        }
        ++logical[std::size_t(linear)];
        ++physical[std::size_t(transformed)];
        ++totals.a_vectors;
      }
    }
    int const live = problem_m * M::AGlobalOuter;
    int physical_live = 0;
    for (int i = 0; i < M::ASharedStage; ++i) {
      int const expected = i < live ? 1 : 0;
      physical_live += physical[std::size_t(i)];
      if (logical[std::size_t(i)] != expected ||
          physical[std::size_t(i)] > 1) {
        failure.set("a-exact-once", M::TileM, M::TileN, M::TileK,
                    M::Stages, problem_m, i,
                    100 * logical[std::size_t(i)] +
                        physical[std::size_t(i)]);
        return;
      }
    }
    if (physical_live != live) {
      failure.set("a-physical-cardinality", M::TileM, M::TileN, M::TileK,
                  M::Stages, problem_m, physical_live, live);
      return;
    }
  }
}

template <class M>
void verify_ab_k(Plant plant, Totals& totals, Failure& failure) {
  for (int tid = 0; tid < M::Threads; ++tid) {
    int const warp_k = (tid / 32) / M::WarpOnN;
    for (int inner = 0; inner < M::BInnerIters; ++inner) {
      int const k_inner = inner / M::BLoadsPerKInner;
      int const b_block = k_inner * M::WarpOnK + warp_k;
      int a_block = k_inner * M::WarpOnK + warp_k;
      if (plant == Plant::OldM8KCadence && M::InstructionM == 8) {
        a_block = warp_k * M::KInnerIters + k_inner;
      }
      ++totals.k_pairs;
      if (b_block != a_block) {
        failure.set("a-b-k-slice", M::TileM, M::TileN, M::TileK,
                    M::Stages, tid, inner, 100 * a_block + b_block);
        return;
      }
    }
  }
}

template <class M>
void verify_output(Plant plant, Totals& totals, Failure& failure) {
  constexpr int output_threads = M::Threads / M::WarpOnK;
  std::vector<int> seen(std::size_t(M::TileM * M::TileN), 0);
  int constexpr q = 3;
  for (int tid = 0; tid < output_threads; ++tid) {
    int const lane = tid % 32;
    for (int n_block = 0; n_block < M::NBlocksPerWarp; ++n_block) {
      int n_base = kg::output_n_base<
          M::TileN, M::NBlocksPerWarp>(q, tid, n_block);
      if (plant == Plant::LocalOutputTile) n_base -= q * M::TileN;
      for (int value = 0; value < M::AccumulatorValues; ++value) {
        int const row = kg::output_row<M::InstructionM>(lane, value);
        int const col = n_base +
            kg::output_col_offset<M::InstructionM>(lane, value);
        int const local_col = col - q * M::TileN;
        ++totals.output_cells;
        if (row < 0 || row >= M::TileM || local_col < 0 ||
            local_col >= M::TileN) {
          failure.set("output-range", M::TileM, M::TileN, M::TileK,
                      M::Stages, tid, row, col);
          return;
        }
        ++seen[std::size_t(row * M::TileN + local_col)];
      }
    }
  }
  for (int i = 0; i < M::TileM * M::TileN; ++i) {
    if (seen[std::size_t(i)] != 1) {
      failure.set("output-exact-once", M::TileM, M::TileN, M::TileK,
                  M::Stages, i, seen[std::size_t(i)]);
      return;
    }
  }
}

template <class M>
void verify_scale_ring(Plant plant, Totals& totals, Failure& failure) {
  constexpr int max_tiles = 64;
  for (int begin = 0; begin < max_tiles; ++begin) {
    for (int count = 1; count <= max_tiles - begin; ++count) {
      ++totals.scale_cases;
      std::array<int, M::Stages> slot{};
      slot.fill(-1);
      int phase = begin % M::ScaleTilesPerGroup;
      if (plant == Plant::ZeroScalePhase) phase = 0;
      int const group_base = M::scale_group_index(begin);
      auto copy = [&](int ring_slot, int tile_base, int tile_offset,
                      int local_tile) {
        int const group = group_base +
            M::scale_group_offset(phase, tile_base, tile_offset);
        int const expected = M::scale_group_index(begin + local_tile);
        if (group != expected) {
          failure.set("scale-source-group", M::TileM, M::TileN, M::TileK,
                      M::Stages, begin, local_tile,
                      100 * group + expected);
          return;
        }
        slot[std::size_t(ring_slot)] = group;
      };

      for (int i = 0; i < M::Stages - 1 && i < count; ++i) {
        copy(i, 0, i, i);
        if (failure.failed) return;
      }
      int remaining = count;
      for (int tile = 0; tile < count; ++tile) {
        int const pipe = tile % M::Stages;
        int const expected = M::scale_group_index(begin + tile);
        ++totals.scale_tiles;
        if (slot[std::size_t(pipe)] != expected) {
          failure.set("scale-ring-read", M::TileM, M::TileN, M::TileK,
                      M::Stages, begin, tile,
                      100 * slot[std::size_t(pipe)] + expected);
          return;
        }
        if (remaining >= M::Stages) {
          int const outer_base =
              (M::Stages - 1) + (tile / M::Stages) * M::Stages;
          int const candidate = M::Stages - 1 + tile;
          copy((pipe + M::Stages - 1) % M::Stages,
               outer_base, pipe, candidate);
          if (failure.failed) return;
        }
        --remaining;
      }
    }
  }
}

template <class M>
void verify_type(Plant plant, Totals& totals, Failure& failure) {
  static_assert(std::is_same_v<typename M::TiledMma, ExpectedMma<M>>,
                "production TiledMma permutation does not scale with N/K cohorts");
  static_assert(cute::size(typename M::TiledMma{}) == M::Threads);
  static_assert(cute::size<1>(typename M::TiledMma::ThrLayoutVMNK{}) == 1);
  static_assert(cute::size<2>(typename M::TiledMma::ThrLayoutVMNK{}) == M::WarpOnN);
  static_assert(cute::size<3>(typename M::TiledMma::ThrLayoutVMNK{}) == M::WarpOnK);
  constexpr mt::MarlinTacticPPU tactic{
      M::TileM, M::TileN, M::TileK, M::WarpM, M::WarpN, M::WarpK,
      M::Stages, mt::MarlinLoadKindPPU::CpAsync};
  static_assert(mt::admitted(tactic));
  static_assert(sizeof(typename M::SharedStorage) ==
                mt::mainloop_shared_bytes(tactic));
  ++totals.types;
  verify_a_producer<M>(plant, totals, failure);
  if (failure.failed) return;
  verify_ab_k<M>(plant, totals, failure);
  if (failure.failed) return;
  verify_output<M>(plant, totals, failure);
  if (failure.failed) return;
  verify_scale_ring<M>(plant, totals, failure);
}

template <int TM, int TN, int TK, int WN, int WK>
void verify_stages(Plant plant, Totals& totals, Failure& failure) {
  verify_type<Main<TM, TN, TK, WN, WK, 2>>(plant, totals, failure);
  if (failure.failed) return;
  verify_type<Main<TM, TN, TK, WN, WK, 3>>(plant, totals, failure);
  if (failure.failed) return;
  verify_type<Main<TM, TN, TK, WN, WK, 4>>(plant, totals, failure);
  if (failure.failed) return;
  verify_type<Main<TM, TN, TK, WN, WK, 5>>(plant, totals, failure);
  if (failure.failed) return;
  verify_type<Main<TM, TN, TK, WN, WK, 6>>(plant, totals, failure);
}

template <int TM>
void verify_geometries(Plant plant, Totals& totals, Failure& failure) {
  verify_stages<TM, 64, 128, 64, 32>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 128, 64, 64, 32>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 128, 64, 128, 16>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 128, 128, 64, 32>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 128, 128, 128, 16>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 256, 64, 64, 32>(plant, totals, failure);
  if (failure.failed) return;
  verify_stages<TM, 256, 64, 128, 16>(plant, totals, failure);
}

}  // namespace

int main(int argc, char** argv) {
  Plant plant = Plant::None;
  if (argc == 2 && !std::strncmp(argv[1], "--plant=", 8)) {
    plant = parse_plant(argv[1] + 8);
  } else if (argc != 1) {
    std::fprintf(stderr, "usage: %s [--plant=NAME]\n", argv[0]);
    return 2;
  }
  if (plant == static_cast<Plant>(-1)) {
    std::fprintf(stderr, "L185 FAIL: unknown plant\n");
    return 2;
  }

  Totals totals;
  Failure failure;
  verify_geometries<8>(plant, totals, failure);
  if (!failure.failed) verify_geometries<16>(plant, totals, failure);

  if (plant == Plant::None) {
    if (failure.failed) {
      std::fprintf(stderr,
          "L185 FAIL category=%s cfg=%dx%dx%d s%d detail=%d/%d/%d\n",
          failure.category, failure.tm, failure.tn, failure.tk,
          failure.stages, failure.a, failure.b, failure.c);
      return 1;
    }
    std::printf(
        "L185 PASS types=%llu A={cases:%llu vectors:%llu} "
        "AB_K_pairs=%llu output_cells=%llu scale={cases:%llu tiles:%llu} "
        "geometries=7 M_atoms=2 stages=5\n",
        static_cast<unsigned long long>(totals.types),
        static_cast<unsigned long long>(totals.a_cases),
        static_cast<unsigned long long>(totals.a_vectors),
        static_cast<unsigned long long>(totals.k_pairs),
        static_cast<unsigned long long>(totals.output_cells),
        static_cast<unsigned long long>(totals.scale_cases),
        static_cast<unsigned long long>(totals.scale_tiles));
    return 0;
  }
  if (!failure.failed) {
    std::fprintf(stderr, "L185 UNEXPECTED-GREEN plant=%s\n",
                 plant_name(plant));
    return 2;
  }
  std::printf(
      "L185 EXPECTED-RED plant=%s category=%s cfg=%dx%dx%d s%d detail=%d/%d/%d\n",
      plant_name(plant), failure.category, failure.tm, failure.tn,
      failure.tk, failure.stages, failure.a, failure.b, failure.c);
  return 1;
}
