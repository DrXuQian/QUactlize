// L225 -- compiled-type closure for the opt-in Q4 A64/F1 -> compute-F2 policy.
//
// L224 proves the byte/delivery/fragment map.  This file proves the production builder consumes exactly that
// contract: physical copy/storage types remain the ordinary F1 types, T64 collapses to the exact ordinary collective,
// and T128 changes only the TiledMma K permutation.  A T32 plant must fail because sub-artifact reuse has no proved
// lifetime yet.

#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "ppu_group_schedule.hpp"

namespace {
using namespace cute;
using QM = ppu_mixed_policy::QuantMode;
using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;

template <int TK>
struct Cell {
  using Tile = Shape<_64, _128, Int<TK>>;
  using Scale = Shape<_128, Int<(TK + 31) / 32>>;
  using Warp = Shape<_64, _64, Int<TK>>;
  using Ordinary = ppu_mixed_policy::MainloopPolicy<
      QM::FinegrainedScaleOnly, Schedule, Tile, Scale, Warp, 3, true,
      cutlass::int4b_t, void, 64>;
  using Virtual = ppu_mixed_policy::VirtualFoldMainloopPolicy<
      2, QM::FinegrainedScaleOnly, Schedule, Tile, Scale, Warp, 3, true,
      cutlass::int4b_t, 64>;
  using ExpectedOrdinarySchedule = cutlass::gemm::KernelAiuFold<1, Schedule, 1, 64>;
  using ExpectedVirtualSchedule = cutlass::gemm::KernelAiuVirtualFold<
      2, ExpectedOrdinarySchedule>;

  static_assert(std::is_same_v<typename Ordinary::KernelSchedule, ExpectedOrdinarySchedule>,
                "the opt-in addition must not wrap the default schedule");
  static_assert(std::is_same_v<typename Virtual::KernelSchedule, ExpectedVirtualSchedule>,
                "the virtual schedule must be one outer logical wrapper over the unchanged artifact ABI");
  static_assert(Ordinary::CollectiveBuilderType::ArtifactLowFold == 1 &&
                Virtual::CollectiveBuilderType::ArtifactLowFold == 1 &&
                Ordinary::CollectiveBuilderType::ArtifactTileK == 64 &&
                Virtual::CollectiveBuilderType::ArtifactTileK == 64,
                "ordinary and virtual readers must name identical physical A64/F1 bytes");
  static_assert(Ordinary::CollectiveBuilderType::ComputeLowFold == 1 &&
                Virtual::CollectiveBuilderType::ComputeLowFold == 2,
                "only the virtual reader may change the logical compute fold");
  using O = typename Ordinary::CollectiveOp;
  using V = typename Virtual::CollectiveOp;
  static_assert(std::is_same_v<typename O::DispatchPolicy, typename V::DispatchPolicy> &&
                std::is_same_v<typename O::GmemTiledCopyB, typename V::GmemTiledCopyB> &&
                std::is_same_v<typename O::SmemLayoutB, typename V::SmemLayoutB> &&
                sizeof(typename O::SharedStorage) == sizeof(typename V::SharedStorage),
                "virtual F2 must retain F1 global copy, shared layout and shared-memory footprint");
};

using T64 = Cell<64>;
using T128 = Cell<128>;
using T256 = Cell<256>;
static_assert(std::is_same_v<typename T64::O, typename T64::V>,
              "Q4 T64 has the same PermK=64 and must collapse to the exact ordinary collective");
static_assert(!std::is_same_v<typename T128::O, typename T128::V> &&
              !std::is_same_v<typename T256::O, typename T256::V>,
              "larger tactics must receive a distinct compute-F2 TiledMma");
static_assert(cutlass::MixGemmMmaPermK<4, 128, 1>::value == 64 &&
              cutlass::MixGemmMmaPermK<4, 128, 2>::value == 128 &&
              cutlass::MixGemmMmaPermK<4, 256, 1>::value == 64 &&
              cutlass::MixGemmMmaPermK<4, 256, 2>::value == 256,
              "the exact logical permutation delta must remain explicit");

using DenseOrdinary128 = fpa_intb_ppu::DenseKernelTypes<
    QM::FinegrainedScaleOnly, Schedule, typename T128::Tile, typename T128::Scale,
    typename T128::Warp, 3, true, cutlass::int4b_t, void, 64>;
using DenseVirtual128 = fpa_intb_ppu::DenseVirtualFoldKernelTypes<
    2, QM::FinegrainedScaleOnly, Schedule, typename T128::Tile, typename T128::Scale,
    typename T128::Warp, 3, true, cutlass::int4b_t, 64>;
static_assert(std::is_same_v<typename DenseOrdinary128::CollectiveMainloop, typename T128::O> &&
              std::is_same_v<typename DenseVirtual128::CollectiveMainloop, typename T128::V> &&
              std::is_same_v<typename DenseOrdinary128::CollectiveEpilogue,
                             typename DenseVirtual128::CollectiveEpilogue> &&
              DenseOrdinary128::SharedStorageSize == DenseVirtual128::SharedStorageSize,
              "the ScaleFirst kernel authority must preserve epilogue/smem and select the proved mainloop");

#if defined(L225_NEG_T32)
using RejectedT32 = Cell<32>;
static_assert(sizeof(RejectedT32) != 0, "force the rejected policy to instantiate");
#endif
}  // namespace

int main() {
  std::printf("L225_Q4_F1_VIRTUAL_F2_TYPE PASS default=UNCHANGED physical=F1 compute=F2 "
              "t64=TYPE_IDENTICAL t128_t256=MMA_ONLY smem_delta=0 runtime_branch_delta=0\n");
  return 0;
}
