// L198 compiled-type witness for the one-plane fixed Split-K family.
//
// Compile this TU once with PPU_B_CHUNK=0 and once with PPU_B_CHUNK=1.
// BC0 covers the shipping i4/i2/i1 buckets; BC1 covers the shipping i2/i1
// buckets (the tactic authority intentionally emits no i4/BC1 rows).

#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "dense_splitk_parallel_ppu.cuh"
#include "ppu_group_schedule.hpp"

#ifndef PPU_B_CHUNK
#error "L198 requires an explicit PPU_B_CHUNK=0/1 compile bucket"
#endif
#if PPU_B_CHUNK != 0 && PPU_B_CHUNK != 1
#error "L198 recognizes only the shipping boolean BChunk axis"
#endif

namespace {

using namespace cute;
using QM = fpa_intb_ppu::QuantMode;

template <class ElementB, int ExpectedBits, int ArtifactTileK,
          int ExpectedArtifactTileK, QM ActualMode, QM ExpectedMode,
          int ExpectedBChunk, int GroupSize = 32>
struct OnePlaneCell {
  static constexpr int Bits = cutlass::sizeof_bits<ElementB>::value;
  using Schedule = ppu_group_schedule::FinegrainedSchedule<GroupSize>;
  using Tile = Shape<_8, _128, Int<ArtifactTileK>>;
  using ScaleTile = Shape<_128,
      Int<ppu_group_schedule::scale_groups_v<ArtifactTileK, GroupSize>>>;
  using Warp = Shape<_8, _32, Int<ArtifactTileK>>;

  using Shipping = fpa_intb_ppu::DenseKernelTypes<
      ActualMode, Schedule, Tile, ScaleTile, Warp, 3, true,
      ElementB, void, ArtifactTileK>;
  using DispatchPolicy =
      typename Shipping::CollectiveMainloop::DispatchPolicy;
  using PackedShipping = fpa_intb_ppu::DensePackedAKernelTypes<
      1, ActualMode, Schedule, Tile, ScaleTile, Warp, 3, true,
      ElementB, ArtifactTileK>;
  using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
  using PackedSplit = dense_splitk_parallel_ppu::KernelTypes<
      PackedShipping, Tile, Warp>;
  using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
      PackedShipping, Tile, Warp>;

  using ExpectedShippingKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, typename Shipping::CollectiveMainloop,
      typename Shipping::CollectiveEpilogue,
      cutlass::gemm::SplitKSerialScheduler>;
  using ExpectedPackedShippingKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, typename PackedShipping::CollectiveMainloop,
      typename PackedShipping::CollectiveEpilogue,
      cutlass::gemm::SplitKSerialScheduler>;

  static_assert(Bits == ExpectedBits, "L198_FORMAT_BITS_SEAM");
  static_assert(ArtifactTileK == ExpectedArtifactTileK,
                "L198_FORMAT_ARTIFACT_SEAM");
  static_assert(ActualMode == ExpectedMode, "L198_FORMAT_MODE_SEAM");
  static_assert(PPU_B_CHUNK == ExpectedBChunk, "L198_FORMAT_BCHUNK_SEAM");
  static_assert(
      DispatchPolicy::StaticGroupSize == GroupSize,
      "L198_STATIC_GROUP_SIZE_TYPE_SEAM");
  static_assert(Shipping::MainloopPolicy::HighBits == 0 &&
                    Shipping::MainloopPolicy::ArtifactLowFold == 1 &&
                    Shipping::MainloopPolicy::ArtifactHighFold == 1,
                "L198_FORMAT_ONE_PLANE_FOLD_SEAM");
  static_assert(Shipping::MainloopPolicy::Descriptor::quant_mode ==
                    ExpectedMode &&
                    Shipping::MainloopPolicy::Descriptor::low_bits ==
                    ExpectedBits &&
                    Shipping::MainloopPolicy::Descriptor::artifact_tile_k ==
                    ExpectedArtifactTileK,
                "L198_FORMAT_DESCRIPTOR_SEAM");
  static_assert(std::is_same_v<
                    typename Shipping::MainloopPolicy::Descriptor::
                        BProviderType,
                    ppu_mixed_policy::OrdinaryBProvider> &&
                    Shipping::MainloopPolicy::CollectiveBuilderType::
                            ArtifactTileK == ExpectedArtifactTileK &&
                    cutlass::gemm::fold_schedule_traits<
                        typename Shipping::MainloopPolicy::KernelSchedule>::
                            ArtifactTileK == ExpectedArtifactTileK &&
                    cutlass::gemm::fold_schedule_traits<
                        typename Shipping::MainloopPolicy::KernelSchedule>::
                            ArtifactLowFold == 1,
                "L198_PRODUCTION_RESIDENT_ARTIFACT_READER_SEAM");

  // S=1 remains the exact historical GemmUniversal type for both the ordinary
  // and independently selected M1 packed-A providers.
  static_assert(std::is_same_v<typename Shipping::GemmKernel,
                               ExpectedShippingKernel> &&
                    std::is_same_v<typename PackedShipping::GemmKernel,
                                   ExpectedPackedShippingKernel>,
                "L198_S1_SHIPPING_TYPE_IDENTITY");

  // S>1 changes only scheduling/output.  In particular neither provider may
  // rebuild B/S/Z conversion or reinterpret the resident artifact.
  static_assert(std::is_same_v<typename Split::CollectiveMainloop,
                               typename Shipping::CollectiveMainloop> &&
                    std::is_same_v<typename PackedSplit::CollectiveMainloop,
                                   typename PackedShipping::CollectiveMainloop>,
                "L198_S_GT_1_EXACT_COLLECTIVE_REUSE");
  static_assert(std::is_same_v<typename Split::GemmKernel::CompletionPolicy,
                               cutlass::gemm::kernel::fixed_splitk::
                                   SeparateKernelCompletion> &&
                    std::is_same_v<
                        typename PackedSplit::GemmKernel::CompletionPolicy,
                        cutlass::gemm::kernel::fixed_splitk::
                            SeparateKernelCompletion>,
                "L198_SEPARATE_REDUCER_IS_DEFAULT");
  static_assert(std::is_same_v<typename Split::GemmKernel::ElementD, float> &&
                    std::is_same_v<
                        typename PackedSplit::GemmKernel::ElementD, float> &&
                    sizeof(typename Split::GemmKernel::ElementD) == 4,
                "L198_PARTIAL_ABI_MUST_REMAIN_FP32");
  static_assert(PackedShipping::MainloopPolicy::PackedARows == 1 &&
                    PackedShipping::MainloopPolicy::ArtifactTileK ==
                    ExpectedArtifactTileK &&
                    PackedShipping::MainloopPolicy::ArtifactLowFold == 1,
                "L198_M1_PROVIDER_MUST_RETAIN_RESIDENT_ARTIFACT");
  static_assert(size<0>(
                    typename Shipping::CollectiveMainloop::TiledMma::
                        AtomShape_MNK{}) == 8 &&
                    size<0>(
                        typename PackedShipping::CollectiveMainloop::TiledMma::
                            AtomShape_MNK{}) == 8,
                "L198_TABLE_WITNESS_MUST_RETAIN_M8_INSTRUCTION");

  static constexpr size_t prepared_bytes = sizeof(Prepared);
  static constexpr bool value = prepared_bytes != 0;
  static constexpr bool atom_at_a_time =
      Shipping::MainloopPolicy::Descriptor::atom_at_a_time;
};

template <class ElementB, int Bits, int ArtifactTileK, QM Mode,
          int GroupSize = 32>
using PositiveCell = OnePlaneCell<ElementB, Bits, ArtifactTileK,
                                  ArtifactTileK, Mode, Mode, PPU_B_CHUNK,
                                  GroupSize>;

using I2ScaleOnly = PositiveCell<cutlass::uint2b_t, 2, 128,
                                 QM::FinegrainedScaleOnly>;
using I2ScaleZero = PositiveCell<cutlass::uint2b_t, 2, 128,
                                 QM::FinegrainedScaleZero>;
using I1ScaleOnly = PositiveCell<cutlass::uint1b_t, 1, 256,
                                 QM::FinegrainedScaleOnly>;
using I1ScaleZero = PositiveCell<cutlass::uint1b_t, 1, 256,
                                 QM::FinegrainedScaleZero>;
using I2ScaleOnlyGs16 = PositiveCell<cutlass::uint2b_t, 2, 128,
                                     QM::FinegrainedScaleOnly, 16>;
using I2ScaleZeroGs16 = PositiveCell<cutlass::uint2b_t, 2, 128,
                                     QM::FinegrainedScaleZero, 16>;
static_assert(I2ScaleOnly::value && I2ScaleZero::value &&
              I1ScaleOnly::value && I1ScaleZero::value &&
              I2ScaleOnlyGs16::value && I2ScaleZeroGs16::value);

#if PPU_B_CHUNK == 0
using I4ScaleOnly = PositiveCell<cutlass::int4b_t, 4, 64,
                                 QM::FinegrainedScaleOnly>;
using I4ScaleZero = PositiveCell<cutlass::int4b_t, 4, 64,
                                 QM::FinegrainedScaleZero>;
static_assert(I4ScaleOnly::value && I4ScaleZero::value);
#endif

struct ArgumentSeamResult {
  bool scale_only_null = false;
  bool scale_only_nonnull = false;
  bool scale_zero_nonnull = false;
  bool scale_zero_null = false;
  bool static_group_size_match = false;
  bool static_group_size_mismatch = false;
  bool kernels_accept_well_formed = false;
};

template <class Cell>
bool real_split_arguments_accept(cutlass::half_t const* zeros,
                                 int runtime_group_size,
                                 bool& kernel_accept) {
  constexpr int M = 1;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int S = 8;
  using Shipping = typename Cell::PackedShipping;
  using Kernel = typename Cell::PackedSplit::GemmKernel;
  using ElementB = typename Kernel::ElementB;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;
  using StrideScale = typename Shipping::CollectiveMainloop::StrideScale;
  using DispatchPolicy =
      typename Shipping::CollectiveMainloop::DispatchPolicy;
  constexpr int StaticGroupSize =
      DispatchPolicy::StaticGroupSize;
  static_assert(StaticGroupSize > 0);

  StrideA sA = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(M, K, 1));
  StrideB sB = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(N, K, 1));
  StrideScale sS = cutlass::make_cute_packed_stride(
      StrideScale{}, make_shape(N, K / StaticGroupSize, 1));
  StrideD sP = cutlass::make_cute_packed_stride(
      StrideD{}, make_shape(M, N, S));
  auto* partials = reinterpret_cast<float*>(uintptr_t{0x5000});
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x1000}), sA,
       reinterpret_cast<ElementB const*>(uintptr_t{0x2000}), sB,
       reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x3000}), sS,
       runtime_group_size, zeros},
      {partials, sP, partials, sP},
      S};
  kernel_accept = Kernel::can_implement(args);
  return kernel_accept &&
      dense_splitk_parallel_ppu::
          one_plane_metadata_arguments_valid<Shipping>(args.mainloop) &&
      dense_splitk_parallel_ppu::
          shipping_group_size_arguments_valid<Shipping>(args.mainloop);
}

template <class ScaleOnlyCell, class ScaleZeroCell>
ArgumentSeamResult check_argument_seam() {
  using Shipping = typename ScaleOnlyCell::PackedShipping;
  using DispatchPolicy =
      typename Shipping::CollectiveMainloop::DispatchPolicy;
  constexpr int StaticGroupSize =
      DispatchPolicy::StaticGroupSize;
  constexpr int WrongGroupSize = StaticGroupSize == 16 ? 32 : 16;
  auto* z = reinterpret_cast<cutlass::half_t const*>(uintptr_t{0x4000});
  bool so_good_kernel = false, so_bad_kernel = false;
  bool sz_good_kernel = false, sz_bad_kernel = false;
  bool gs_good_kernel = false, gs_bad_kernel = false;
  ArgumentSeamResult result;
  result.scale_only_null =
      real_split_arguments_accept<ScaleOnlyCell>(
          nullptr, StaticGroupSize, so_good_kernel);
  result.scale_only_nonnull =
      real_split_arguments_accept<ScaleOnlyCell>(
          z, StaticGroupSize, so_bad_kernel);
  result.scale_zero_nonnull =
      real_split_arguments_accept<ScaleZeroCell>(
          z, StaticGroupSize, sz_good_kernel);
  result.scale_zero_null =
      real_split_arguments_accept<ScaleZeroCell>(
          nullptr, StaticGroupSize, sz_bad_kernel);
  result.static_group_size_match =
      real_split_arguments_accept<ScaleOnlyCell>(
          nullptr, StaticGroupSize, gs_good_kernel);
  result.static_group_size_mismatch =
      real_split_arguments_accept<ScaleOnlyCell>(
          nullptr, WrongGroupSize, gs_bad_kernel);
  result.kernels_accept_well_formed =
      so_good_kernel && sz_good_kernel && gs_good_kernel;
  return result;
}

// Optional compile-time plants name the exact format seam they violate.
#if defined(L198_PLANT_BITS)
using Planted = OnePlaneCell<cutlass::int4b_t, 2, 128, 128,
                             QM::FinegrainedScaleOnly,
                             QM::FinegrainedScaleOnly, PPU_B_CHUNK>;
static_assert(Planted::value);
#elif defined(L198_PLANT_MODE)
using Planted = OnePlaneCell<cutlass::uint2b_t, 2, 128, 128,
                             QM::FinegrainedScaleOnly,
                             QM::FinegrainedScaleZero, PPU_B_CHUNK>;
static_assert(Planted::value);
#elif defined(L198_PLANT_ARTIFACT)
using Planted = OnePlaneCell<cutlass::uint2b_t, 2, 256, 128,
                             QM::FinegrainedScaleOnly,
                             QM::FinegrainedScaleOnly, PPU_B_CHUNK>;
static_assert(Planted::value);
#elif defined(L198_PLANT_BCHUNK)
using Planted = OnePlaneCell<cutlass::uint2b_t, 2, 128, 128,
                             QM::FinegrainedScaleOnly,
                             QM::FinegrainedScaleOnly, 1 - PPU_B_CHUNK>;
static_assert(Planted::value);
#endif

}  // namespace

int main() {
  dense_splitk_parallel_ppu::WorkspacePlan s1;
  dense_splitk_parallel_ppu::WorkspacePlan s8;
  bool const workspace =
      dense_splitk_parallel_ppu::query_workspace_plan(1, 4096, 1, s1) &&
      dense_splitk_parallel_ppu::query_workspace_plan(1, 4096, 8, s8) &&
      s1.partial_bytes == 0 && s8.partial_bytes == 131072;
  constexpr int format_cells = PPU_B_CHUNK == 0 ? 6 : 4;
  ArgumentSeamResult const i2_args =
      check_argument_seam<I2ScaleOnly, I2ScaleZero>();
  ArgumentSeamResult const i1_args =
      check_argument_seam<I1ScaleOnly, I1ScaleZero>();
  ArgumentSeamResult const i2_gs16_args =
      check_argument_seam<I2ScaleOnlyGs16, I2ScaleZeroGs16>();
#if PPU_B_CHUNK == 0
  ArgumentSeamResult const i4_args =
      check_argument_seam<I4ScaleOnly, I4ScaleZero>();
  bool const i4_ok = i4_args.scale_only_null &&
      !i4_args.scale_only_nonnull && i4_args.scale_zero_nonnull &&
      !i4_args.scale_zero_null && i4_args.static_group_size_match &&
      !i4_args.static_group_size_mismatch &&
      i4_args.kernels_accept_well_formed;
#else
  bool const i4_ok = true;
#endif
  auto argument_pair_ok = [](ArgumentSeamResult const& result) {
    return result.scale_only_null && !result.scale_only_nonnull &&
        result.scale_zero_nonnull && !result.scale_zero_null &&
        result.static_group_size_match &&
        !result.static_group_size_mismatch &&
        result.kernels_accept_well_formed;
  };
  bool const arguments = argument_pair_ok(i2_args) &&
      argument_pair_ok(i1_args) && argument_pair_ok(i2_gs16_args) && i4_ok;
  bool const pass = workspace && arguments && I2ScaleOnly::value &&
      I2ScaleZero::value && I1ScaleOnly::value && I1ScaleZero::value;
  std::printf(
      "[l198:type] %s bc=%d format_cells=%d bits=%s "
      "modes=ScaleOnly/ScaleZero "
      "S1=EXACT_SHIPPING_TYPE SGT1=EXACT_COLLECTIVE partial=FP32 "
      "completion=SEPARATE artifact_reader=EXACT "
      "arguments=SO:null+/-nonnull- SZ:nonnull+/null- "
      "static_gs=16/32:match+/mismatch- "
      "workspace_s8=%zu atom_i2=%d atom_i1=%d\n",
      pass ? "PASS" : "FAIL", PPU_B_CHUNK, format_cells,
      PPU_B_CHUNK == 0 ? "4/2/1" : "2/1", s8.partial_bytes,
      int(I2ScaleOnly::atom_at_a_time), int(I1ScaleOnly::atom_at_a_time));
  return pass ? 0 : 1;
}
