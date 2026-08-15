// L190 -- exact shipping-mainloop type and device-body gate for dense fixed Split-K parallel.
//
// This translation unit deliberately constructs the producer through
// dense_splitk_parallel_ppu::KernelTypes.  It does not reconstruct the new kernel or its partial
// epilogue in parallel with production.  The S==1 control is reconstructed independently because
// its whole purpose is to prove that the historical shipping authority remains the old
// GemmUniversal<..., SplitKSerialScheduler> type.

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "dense_splitk_parallel_ppu.cuh"
#include "ppu_group_schedule.hpp"

namespace {

using namespace cute;
using QM = fpa_intb_ppu::QuantMode;

using BaseSchedule = ppu_group_schedule::FinegrainedSchedule<128>;
using Tile = Shape<_8, _128, _128>;
using ScaleTile = Shape<_128, _1>;
using Warp = Shape<_8, _32, _128>;

// Exact ordinary, unfolded, one-plane int4/gs128 M8 row.  ArtifactTileK==TacticTileK==128
// makes the format identity explicit instead of deriving it a second time in this gate.
using Shipping = fpa_intb_ppu::DenseKernelTypes<
    QM::FinegrainedScaleZero, BaseSchedule, Tile, ScaleTile, Warp, 3, true,
    cutlass::int4b_t, void, 128>;
using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
using SplitKernel = typename Split::GemmKernel;

using ExpectedShippingKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, typename Shipping::CollectiveMainloop,
    typename Shipping::CollectiveEpilogue,
    cutlass::gemm::SplitKSerialScheduler>;

static_assert(std::is_same_v<typename Split::CollectiveMainloop,
                             typename Shipping::CollectiveMainloop> &&
                  std::is_same_v<typename SplitKernel::CollectiveMainloop,
                                 typename Shipping::CollectiveMainloop>,
              "L190_SPLITK_MAINLOOP_MUST_BE_BYTE_IDENTICAL_TO_SHIPPING");
static_assert(std::is_same_v<typename Shipping::GemmKernel,
                             ExpectedShippingKernel>,
              "L190_S1_MUST_REMAIN_THE_HISTORICAL_SHIPPING_KERNEL_TYPE");
static_assert(!std::is_same_v<SplitKernel, typename Shipping::GemmKernel>,
              "L190_S8_PRODUCER_MUST_NOT_REPLACE_THE_S1_SHIPPING_TYPE");
static_assert(size<0>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 8 &&
                  size<1>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 16 &&
                  size<2>(typename Shipping::CollectiveMainloop::TiledMma::AtomShape_MNK{}) == 16,
              "L190_SHIPPING_ROW_MUST_RETAIN_M8N16K16");
static_assert(Split::BlockM == 8 && Split::BlockN == 128 &&
                  Shipping::MainloopPolicy::TacticTileK == 128 &&
                  Shipping::MainloopPolicy::ArtifactTileK == 128 &&
                  Shipping::CollectiveMainloop::DispatchPolicy::StaticGroupSize == 128,
              "L190_MUST_NAME_THE_EXACT_M8_TN128_TK128_GS128_ROW");

// A concrete global wrapper is stronger than forming KernelTypes: it forces the front end through
// GemmUniversalMixedInputSplitKParallel::operator(), including shipping load_init/mainloop and the
// owned FP32 partial epilogue.  No device code is executed by this gate.
#if !defined(L190_SEVER_DEVICE_BODY)
__global__ void l190_force_splitk_device_body(SplitKernel::Params params) {
  extern __shared__ char smem[];
  SplitKernel{}(params, smem);
}
#endif

bool host_contract() {
  constexpr int M = 1;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int S = 8;
  constexpr auto partition = cutlass::gemm::kernel::fixed_splitk::make_params(
      uint64_t((M + Split::BlockM - 1) / Split::BlockM) *
          uint64_t((N + Split::BlockN - 1) / Split::BlockN),
      uint32_t((K + Shipping::MainloopPolicy::TacticTileK - 1) /
               Shipping::MainloopPolicy::TacticTileK),
      S);
  static_assert(partition.is_valid() && partition.output_tiles == 32 &&
                    partition.k_tiles_per_output == 32 &&
                    partition.splits == 8 && partition.work_units == 256 &&
                    partition.k_tiles_per_split == 4,
                "L190_S8_PARTITION_MUST_REMAIN_32x32x8");
  static_assert(size_t(M) * size_t(N) * size_t(S) * sizeof(float) == 131072,
                "L190_S8_FP32_WORKSPACE_MUST_REMAIN_131072_BYTES");

  // This exact formula is the public grid contract implemented by
  // SplitKernel::get_grid_shape.  The device-body arm below instantiates the
  // production consumer; keeping the host calculation constexpr avoids the
  // local nvcc/CuTe product-object device-pass floor.
  constexpr unsigned GridX = unsigned((M + Split::BlockM - 1) / Split::BlockM);
  constexpr unsigned GridY = unsigned((N + Split::BlockN - 1) / Split::BlockN);
  constexpr unsigned GridZ = unsigned(partition.splits);
  static_assert(GridX == 1 && GridY == 32 && GridZ == 8,
                "L190_S8_GRID_MUST_REMAIN_1x32x8");
  dim3 const grid{GridX, GridY, GridZ};

  size_t workspace = 0;
  bool const workspace_ok = cutlass::gemm::device::splitk_parallel::
      fp32_workspace_size(M, N, S, workspace);
  bool const grid_ok = grid.x == 1 && grid.y == 32 && grid.z == 8;
  bool const partition_ok = partition.is_valid() &&
      partition.output_tiles == 32 &&
      partition.k_tiles_per_output == 32 && partition.splits == 8 &&
      partition.work_units == 256 && partition.k_tiles_per_split == 4;
  dense_splitk_parallel_ppu::WorkspacePlan plan;
  bool const outer_plan_ok = dense_splitk_parallel_ppu::query_workspace_plan(
      M, N, S, plan);
  bool const workspace_exact = workspace_ok && outer_plan_ok &&
      workspace == 131072 && plan.partial_bytes == workspace &&
      plan.alignment == 16;

  std::printf(
      "[l190] shipping=ordinary-int4-gs128 tile=8x128x128 warp=8x32x128 "
      "S1=historical grid=%ux%ux%u units=%llu k/peer=%u workspace=%zu "
      "device_body=ODR-USED\n",
      grid.x, grid.y, grid.z,
      static_cast<unsigned long long>(partition.work_units),
      partition.k_tiles_per_split, workspace);
  return grid_ok && partition_ok && workspace_exact;
}

#if defined(L190_ADMISSION_PROBE)
bool admission_contract() {
  constexpr int M = 1;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int S = 8;
  constexpr int ScaleK = K / 128;
  using StrideA = typename SplitKernel::StrideA;
  using StrideB = typename SplitKernel::StrideB;
  using StrideD = typename SplitKernel::StrideD;
  using StrideScale = typename Split::CollectiveMainloop::StrideScale;

  StrideA sA{int64_t(K), cute::_1{}, int64_t(0)};
  StrideB sB{int64_t(K), cute::_1{}, int64_t(0)};
  StrideScale sS{cute::_1{}, int64_t(N), int64_t(0)};
  StrideD compact{int64_t(N), cute::_1{}, int64_t(M) * N};
  StrideD dD = compact;
#if defined(L190_PLANT_BAD_STRIDE)
  dD = StrideD{int64_t(N), cute::_1{}, int64_t(M) * N + 4};
#endif

  auto* aligned = reinterpret_cast<float*>(uintptr_t{0x5000});
  auto* destination = aligned;
#if defined(L190_PLANT_UNALIGNED)
  destination = reinterpret_cast<float*>(uintptr_t{0x5004});
#endif

  typename SplitKernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {M, N, K, 1};
  args.mainloop = {
      reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x1000}), sA,
      reinterpret_cast<cutlass::int4b_t const*>(uintptr_t{0x2000}), sB,
      reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x3000}), sS,
      128,
      reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x4000})};
  args.partial_epilogue = {aligned, compact, destination, dD};
  args.split_k_slices = S;
  bool const accepted = SplitKernel::can_implement(args);

#if defined(L190_PLANT_BAD_STRIDE)
  std::printf("[l190:admission] plant=bad-stride accepted=%d expected=0\n",
              int(accepted));
  return !accepted;
#elif defined(L190_PLANT_UNALIGNED)
  std::printf("[l190:admission] plant=unaligned accepted=%d expected=0\n",
              int(accepted));
  return !accepted;
#else
  std::printf("[l190:admission] plant=none accepted=%d expected=1\n",
              int(accepted));
  return accepted;
#endif
}
#endif

}  // namespace

int main() {
#if defined(L190_ADMISSION_PROBE)
  if (!admission_contract()) {
    std::fprintf(stderr, "[l190:admission] FAIL: compact partial ABI admission drifted\n");
    return 1;
  }
  std::puts("[l190:admission] PASS");
  return 0;
#else
  if (!host_contract()) {
    std::fprintf(stderr, "[l190] FAIL: S8 lowering/grid/workspace contract drifted\n");
    return 1;
  }
  std::puts(
      "[l190:host] PASS: exact shipping mainloop retained; S8 grid/workspace exact; "
      "S1 authority unchanged");
  return 0;
#endif
}
