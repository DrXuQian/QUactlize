// L199 -- exhaustive local type/ABI census for dense multiformat fixed Split-K.
//
// One invocation selects a shipping qtype, metadata ABI and PPU_B_CHUNK build.
// The runner crosses all five qtypes, both metadata ABIs and both B-chunk modes.
// Within an invocation this source mechanically crosses the four registered
// artifact TileK candidates, every QUACTLIZE_PPU_DENSE_CONFIGS row and
// S={1,2,4,8}.  Rejected cells receive a named reason; admitted cells bind the
// exact S==1 shipping type and the exact S>1 CollectiveMainloop, then exercise
// the real mainloop/split admission predicates with non-dereferenced pointers.

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "dense_splitk_multiformat_ppu.cuh"
#include "ppu_dense_shipping_policy.hpp"
#include "ppu_format_config.hpp"
#include "ppu_group_schedule.hpp"
#include "ppu_placed_arrangement.hpp"

#ifndef L199_QTYPE
#error "L199_QTYPE must select one shipping qtype"
#endif

#ifndef L199_PACKED_METADATA
#define L199_PACKED_METADATA 0
#endif

#ifndef PPU_B_CHUNK
#define PPU_B_CHUNK 0
#endif

static_assert(PPU_B_CHUNK == 0 || PPU_B_CHUNK == 1,
              "L199 crosses only the two production B-chunk modes");
static_assert(L199_PACKED_METADATA == 0 || L199_PACKED_METADATA == 1);

namespace {

using QM = fpa_intb_ppu::QuantMode;
namespace fs = cutlass::gemm::kernel::fixed_splitk;

constexpr int kM = 1;
constexpr int kN = 4096;
constexpr int kK = 4096;
constexpr std::array<int, 4> kArtifactTileK{{32, 64, 128, 256}};
constexpr std::array<int, 4> kSplits{{1, 2, 4, 8}};

template <int QType>
struct Format;

template <>
struct Format<10> {
  using Low = cutlass::uint2b_t;
  using High = void;
  static constexpr int GroupSize = 16;
  static constexpr int PackedFormat = 2;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::I2;
};

template <>
struct Format<11> {
  using Low = cutlass::uint2b_t;
  using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 16;
  static constexpr int PackedFormat = 3;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q3_K;
};

template <>
struct Format<12> {
  using Low = cutlass::int4b_t;
  using High = void;
  static constexpr int GroupSize = 32;
  static constexpr int PackedFormat = 0;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::I4;
};

template <>
struct Format<13> {
  using Low = cutlass::int4b_t;
  using High = cutlass::uint1b_t;
  static constexpr int GroupSize = 32;
  static constexpr int PackedFormat = 1;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q5_K;
};

template <>
struct Format<14> {
  using Low = cutlass::int4b_t;
  using High = cutlass::uint2b_t;
  static constexpr int GroupSize = 16;
  static constexpr int PackedFormat = 4;
  static constexpr ppu_tactics::Format TacticFormat = ppu_tactics::Format::Q6_K;
};

using Selected = Format<L199_QTYPE>;
using Low = typename Selected::Low;
using High = typename Selected::High;
constexpr bool kPackedMetadata = L199_PACKED_METADATA != 0;
constexpr auto kFormat = ppu_formats::for_qtype(L199_QTYPE);
static_assert(kFormat.qtype == L199_QTYPE &&
              kFormat.low_bits == ppu_mixed_policy::element_bits_v<Low> &&
              kFormat.high_bits == ppu_mixed_policy::element_bits_v<High> &&
              kFormat.group_size == Selected::GroupSize,
              "L199 type mapping must be derived from the shipping format registry");

#if L199_PACKED_METADATA
#if !defined(PPU_PACKED_SCALE) || PPU_PACKED_SCALE == 0
#error "packed L199 arm must enable the shipping packed metadata collective"
#endif
#if !defined(PPU_PACKED_FORMAT)
#error "packed L199 arm must select the format-specific packed unit"
#endif
static_assert(PPU_PACKED_FORMAT == Selected::PackedFormat,
              "packed L199 arm must select the qtype's shipping packed unit");
#else
#if defined(PPU_PACKED_SCALE) && PPU_PACKED_SCALE != 0
#error "scale-first L199 arm must compile the fp16 scale/zero collective"
#endif
#endif

#if defined(L199_FORCE_DEVICE_BODY)
constexpr int witness_artifact_tile_k() {
  if constexpr (!kPackedMetadata) return kFormat.scale_first_tile_k;
  if constexpr (L199_QTYPE == 10) return 128;
  if constexpr (L199_QTYPE == 12) return 64;
  return 64;  // Q3/Q5/Q6: independently folded high-plane witness.
}
constexpr int kWitnessArtifactTileK = witness_artifact_tile_k();
constexpr int kWitnessTacticTileK = kPackedMetadata
    ? kFormat.fully_quantized_tile_k : kFormat.scale_first_tile_k;
constexpr int kWitnessScaleGroups =
    ppu_group_schedule::scale_groups_v<kWitnessTacticTileK, Selected::GroupSize>;
using WitnessSchedule = ppu_group_schedule::FinegrainedSchedule<Selected::GroupSize>;
using WitnessTile = cute::Shape<cute::_8, cute::_128, cute::C<kWitnessTacticTileK>>;
using WitnessScaleTile = cute::Shape<cute::_128, cute::C<kWitnessScaleGroups>>;
using WitnessWarp = cute::Shape<cute::_8, cute::_32, cute::C<kWitnessTacticTileK>>;
// Production gives exactly one M==1/config combination an independent A
// provider: ShortWideM8S3 with an ordinary unfolded one-plane B.  The packed
// Q2/Q4 witnesses deliberately name that exact backend branch; the two-plane
// witnesses remain ordinary DenseKernelTypes.
constexpr bool kWitnessBackendUsesPackedA =
    kPackedMetadata && std::is_void_v<High> &&
    ppu_mixed_policy::element_bits_v<Low> != 0 &&
    fold::delivery_fold_v<ppu_mixed_policy::element_bits_v<Low>,
                          kWitnessArtifactTileK> == 1;
constexpr int kWitnessStages = kWitnessBackendUsesPackedA ? 3 : 2;
using WitnessOrdinaryShipping = fpa_intb_ppu::DenseKernelTypes<
    QM::FinegrainedScaleZero, WitnessSchedule, WitnessTile, WitnessScaleTile,
    WitnessWarp, kWitnessStages, true, Low, High, kWitnessArtifactTileK>;
using WitnessPackedAShipping = fpa_intb_ppu::DensePackedAKernelTypes<
    1, QM::FinegrainedScaleZero, WitnessSchedule, WitnessTile,
    WitnessScaleTile, WitnessWarp, kWitnessStages, true, Low,
    kWitnessArtifactTileK>;
#if defined(L199_PLANT_DEVICE_DECODE_DEFAULT_ORDINARY)
constexpr bool kWitnessSelectsPackedA = false;
#else
constexpr bool kWitnessSelectsPackedA = kWitnessBackendUsesPackedA;
#endif
using WitnessShipping = std::conditional_t<
    kWitnessSelectsPackedA, WitnessPackedAShipping,
    WitnessOrdinaryShipping>;
constexpr bool kWitnessActualPackedA = std::is_same_v<
    typename WitnessShipping::MainloopPolicy::Descriptor::AProviderType,
    ppu_mixed_policy::PackedRowAProvider>;
static_assert(kWitnessActualPackedA == kWitnessBackendUsesPackedA,
              "L199_DEVICE_DECODE_DEFAULT_PACKED_A_SEAM");
using WitnessSplit = dense_splitk_parallel_ppu::KernelTypes<
    WitnessShipping, WitnessTile, WitnessWarp>;
using WitnessKernel = typename WitnessSplit::GemmKernel;
static_assert(std::is_same_v<typename WitnessKernel::CollectiveMainloop,
                             typename WitnessShipping::CollectiveMainloop>);
static_assert(dense_splitk_parallel_ppu::MainloopUsesPackedMetadata<
                  typename WitnessShipping::CollectiveMainloop>::value ==
              kPackedMetadata);
#ifndef L199_EXPECT_EFFECTIVE_BCHUNK
#error "device-body witness must name the expected effective BChunk state"
#endif
constexpr bool kWitnessEffectiveBChunk =
    ppu_mixed_policy::AtomAtATimeConversion<
        typename WitnessShipping::CollectiveMainloop>::value;
static_assert(kWitnessEffectiveBChunk == bool(L199_EXPECT_EFFECTIVE_BCHUNK),
              "L199_DEVICE_BCHUNK_EFFECTIVE_SEAM");

__global__ void l199_force_multiformat_device_body(
    WitnessKernel::Params params) {
  extern __shared__ char smem[];
  WitnessKernel{}(params, smem);
}
#endif

enum class Reject {
  None,
  FormatArtifactUnsupported,
  ScaleFirstArtifactMismatch,
  PackedReaderUnsupported,
  Kernel,
  Producer,
  ShippingSharedStorage,
  SplitSharedStorage,
  SplitPartition,
  PipelineDepth,
  RealAdmission,
  ExactCoverage,
  ExactFp16,
  ArgumentSeam,
};

char const* reject_name(Reject reject, ppu_tactics::Exclusion exclusion) {
  switch (reject) {
    case Reject::None: return "ADMITTED";
    case Reject::FormatArtifactUnsupported: return "FORMAT_ARTIFACT_UNSUPPORTED";
    case Reject::ScaleFirstArtifactMismatch: return "SCALE_FIRST_ARTIFACT_MISMATCH";
    case Reject::PackedReaderUnsupported: return "PACKED_READER_UNSUPPORTED";
    case Reject::Kernel: return ppu_tactics::exclusion_clause(exclusion);
    case Reject::Producer: return ppu_tactics::exclusion_clause(exclusion);
    case Reject::ShippingSharedStorage: return "SHIPPING_SHARED_STORAGE";
    case Reject::SplitSharedStorage: return "SPLIT_SHARED_STORAGE";
    case Reject::SplitPartition: return "SPLIT_PARTITION";
    case Reject::PipelineDepth: return "INADMISSIBLE_PIPELINE_DEPTH";
    case Reject::RealAdmission: return "REAL_KERNEL_ADMISSION";
    case Reject::ExactCoverage: return "EXACT_ONCE_COVERAGE";
    case Reject::ExactFp16: return "RAW_FP16_MISMATCH";
    case Reject::ArgumentSeam: return "FORMAT_ARGUMENT_SEAM";
  }
  return "UNKNOWN";
}

struct Census {
  uint64_t cells = 0;
  uint64_t admitted = 0;
  uint64_t rejected = 0;
  uint64_t typed = 0;
  uint64_t exact_qk = 0;
  uint64_t b2_reds = 0;
  uint64_t fold_reds = 0;
  uint64_t metadata_reds = 0;
  uint64_t group_reds = 0;
  uint64_t bchunk_effective_on = 0;
  uint64_t bchunk_effective_off = 0;
  uint64_t bchunk_named_rejects = 0;
  uint64_t failures = 0;
};

template <int ArtifactTileK>
constexpr Reject artifact_reject() {
  if (!ppu_formats::artifact_tile_k_supported(kFormat, ArtifactTileK)) {
    return Reject::FormatArtifactUnsupported;
  }
  if constexpr (!kPackedMetadata) {
    if (ArtifactTileK != kFormat.scale_first_tile_k) {
      return Reject::ScaleFirstArtifactMismatch;
    }
  } else {
    constexpr quactlize_ppu_placed_arrangement_v1 arrangement{
        QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
        kFormat.low_bits, ArtifactTileK, kFormat.high_bits};
    if (!ppu_arrangements::packed_tensor_reader_supported(
            &arrangement, L199_QTYPE, kK, kFormat.fully_quantized_tile_k)) {
      return Reject::PackedReaderUnsupported;
    }
  }
  return Reject::None;
}

template <int TM, int TN, int TacticTileK>
uint64_t logical_fixture_checksum() {
  uint64_t checksum = 1469598103934665603ull;
  int const output_tiles = ((kM + TM - 1) / TM) * ((kN + TN - 1) / TN);
  int const k_tiles = kK / TacticTileK;
  for (int q = 0; q < output_tiles; ++q) {
    float sum = 0.f;
    for (int kt = 0; kt < k_tiles; ++kt) {
      // Small integers make every partial and every reducer addition exact in
      // FP32; the fixture therefore tests decomposition/order, not tolerance.
      sum += float(((q * 3 + kt * 5 + L199_QTYPE) % 7) - 3);
    }
    uint16_t const bits = cutlass::half_t(sum).raw();
    checksum ^= uint64_t(bits) | (uint64_t(q) << 16);
    checksum *= 1099511628211ull;
  }
  return checksum;
}

template <int TM, int TN, int TacticTileK>
bool exact_fixture(int splits, uint64_t& qk_cells) {
  int const output_tiles = ((kM + TM - 1) / TM) * ((kN + TN - 1) / TN);
  int const k_tiles = kK / TacticTileK;
  fs::Params const params = fs::make_params(output_tiles, k_tiles, splits);
  if (!params.is_valid()) return false;
  std::vector<uint8_t> visits(size_t(output_tiles) * k_tiles, uint8_t(0));
  for (int q = 0; q < output_tiles; ++q) {
    float reference = 0.f;
    for (int kt = 0; kt < k_tiles; ++kt) {
      reference += float(((q * 3 + kt * 5 + L199_QTYPE) % 7) - 3);
    }
    float reduced = 0.f;
    for (int s = 0; s < splits; ++s) {
      auto const work = fs::work_for(params, uint64_t(q), uint32_t(s));
      if (!fs::work_matches_params(params, work)) return false;
      float partial = 0.f;
      for (uint32_t kt = work.k_begin; kt < work.k_begin + work.k_count; ++kt) {
        size_t const index = size_t(q) * k_tiles + kt;
        if (visits[index]++ != 0) return false;
        partial += float(((q * 3 + int(kt) * 5 + L199_QTYPE) % 7) - 3);
      }
      reduced += partial;
    }
    if (cutlass::half_t(reference).raw() != cutlass::half_t(reduced).raw()) {
      return false;
    }
  }
  for (uint8_t visits_for_cell : visits) {
    if (visits_for_cell != 1) return false;
  }
  qk_cells += visits.size();
  return true;
}

template <int ArtifactTileK, ppu_dense_shipping::ConfigId ConfigId,
          int TM, int TN, int WM, int WN, int Stages>
void typed_config(Census& census, char const* config_name) {
  constexpr int TacticTileK = kPackedMetadata
      ? kFormat.fully_quantized_tile_k : kFormat.scale_first_tile_k;
  constexpr ppu_tactics::FormatSpec spec{
      Selected::TacticFormat, kFormat.name, kFormat.low_bits, kFormat.high_bits};
  constexpr ppu_tactics::Candidate candidate{
      spec, TM, TN, TacticTileK, WM, WN, ArtifactTileK, PPU_B_CHUNK};
  constexpr auto kernel_exclusion =
      ppu_tactics::DenseSpace::kernel_exclusion(candidate);
  constexpr auto producer_exclusion =
      ppu_tactics::common_producer_exclusion(candidate);

  if constexpr (kernel_exclusion != ppu_tactics::Exclusion::None ||
                producer_exclusion != ppu_tactics::Exclusion::None) {
    Reject const reject = kernel_exclusion != ppu_tactics::Exclusion::None
        ? Reject::Kernel : Reject::Producer;
    auto const exclusion = kernel_exclusion != ppu_tactics::Exclusion::None
        ? kernel_exclusion : producer_exclusion;
    if constexpr (kernel_exclusion ==
                  ppu_tactics::Exclusion::BChunkUnsupportedBits) {
      census.bchunk_named_rejects += kSplits.size();
    }
    for (int split : kSplits) {
      ++census.cells;
      ++census.rejected;
      std::printf("[l199:cell] q=%d metadata=%s A=%d config=%s "
                  "bchunk=%d/REJECT S=%d REJECT %s\n",
                  L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",
                  ArtifactTileK, config_name, PPU_B_CHUNK, split,
                  reject_name(reject, exclusion));
    }
  } else {
    constexpr int ScaleGroups =
        ppu_group_schedule::scale_groups_v<TacticTileK, Selected::GroupSize>;
    using Tile = cute::Shape<cute::C<TM>, cute::C<TN>, cute::C<TacticTileK>>;
    using ScaleTile = cute::Shape<cute::C<TN>, cute::C<ScaleGroups>>;
    using Warp = cute::Shape<cute::C<WM>, cute::C<WN>, cute::C<TacticTileK>>;
    using Schedule = ppu_group_schedule::FinegrainedSchedule<Selected::GroupSize>;
    using OrdinaryShipping = fpa_intb_ppu::DenseKernelTypes<
        QM::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp, Stages,
        true, Low, High, ArtifactTileK>;
    using PackedAShipping = fpa_intb_ppu::DensePackedAKernelTypes<
        1, QM::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp, Stages,
        true, Low, ArtifactTileK>;
    constexpr bool backend_uses_packed_a =
        ConfigId == ppu_dense_shipping::kDecodeDefault && kM == 1 &&
        std::is_void_v<High> &&
        ppu_tactics::artifact_low_fold(candidate) == 1;
#if defined(L199_PLANT_DECODE_DEFAULT_ORDINARY)
    constexpr bool select_packed_a = false;
#else
    constexpr bool select_packed_a = backend_uses_packed_a;
#endif
    using Shipping = std::conditional_t<
        select_packed_a, PackedAShipping, OrdinaryShipping>;
    constexpr bool actual_packed_a = std::is_same_v<
        typename Shipping::MainloopPolicy::Descriptor::AProviderType,
        ppu_mixed_policy::PackedRowAProvider>;
    static_assert(actual_packed_a == backend_uses_packed_a,
                  "L199_DECODE_DEFAULT_PACKED_A_SHIPPING_SEAM");
    using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
    using ExpectedShippingKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, typename Shipping::CollectiveMainloop,
        typename Shipping::CollectiveEpilogue,
        cutlass::gemm::SplitKSerialScheduler>;

    static_assert(std::is_same_v<typename Shipping::GemmKernel,
                                 ExpectedShippingKernel>,
                  "S=1 must remain the exact historical shipping type");
    static_assert(std::is_same_v<typename Split::CollectiveMainloop,
                                 typename Shipping::CollectiveMainloop> &&
                  std::is_same_v<typename Split::GemmKernel::CollectiveMainloop,
                                 typename Shipping::CollectiveMainloop>,
                  "S>1 must reuse the exact shipping mainloop");
    static_assert(std::is_same_v<typename Shipping::ElementA, cutlass::half_t> &&
                  std::is_same_v<typename Shipping::ElementAccumulator, float> &&
                  std::is_same_v<typename Shipping::ElementD, cutlass::half_t> &&
                  std::is_same_v<typename Split::GemmKernel::ElementD, float>,
                  "audited A/accumulator/D/partial types changed");
    static_assert(Shipping::MainloopPolicy::ArtifactLowFold ==
                      ppu_tactics::artifact_low_fold(candidate) &&
                  Shipping::MainloopPolicy::ArtifactHighFold ==
                      ppu_tactics::artifact_high_fold(candidate),
                  "split path must retain the resident artifact folds");
    static_assert(
        dense_splitk_parallel_ppu::MainloopUsesPackedMetadata<
            typename Shipping::CollectiveMainloop>::value == kPackedMetadata,
        "metadata ABI must match the selected shipping collective");
    constexpr bool effective_bchunk =
        ppu_mixed_policy::AtomAtATimeConversion<
            typename Shipping::CollectiveMainloop>::value;

    constexpr bool shipping_fits =
        Shipping::SharedStorageSize <= ppu_tactics::kBlockSmemBytes;
    constexpr bool split_fits =
        Split::GemmKernel::SharedStorageSize <= ppu_tactics::kBlockSmemBytes;

    if constexpr (!shipping_fits) {
      for (int split : kSplits) {
        ++census.cells;
        ++census.rejected;
        std::printf("[l199:cell] q=%d metadata=%s A=%d config=%s "
                    "bchunk=%d/%d S=%d REJECT %s\n",
                    L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",
                    ArtifactTileK, config_name, PPU_B_CHUNK,
                    int(effective_bchunk), split,
                    reject_name(Reject::ShippingSharedStorage,
                                ppu_tactics::Exclusion::None));
      }
    } else {
    for (int split : kSplits) {
      ++census.cells;
      Reject reject = Reject::None;
      if (split > 1 && !split_fits) {
        reject = Reject::SplitSharedStorage;
      } else {
        int const output_tiles = ((kM + TM - 1) / TM) * ((kN + TN - 1) / TN);
        int const k_tiles = kK / TacticTileK;
        fs::Params const partition = fs::make_params(output_tiles, k_tiles, split);
        if (!partition.is_valid()) {
          reject = Reject::SplitPartition;
        } else if (split > 1 && int(partition.k_tiles_per_split) < Stages - 1) {
          reject = Reject::PipelineDepth;
        }
      }

      if (reject != Reject::None) {
        ++census.rejected;
        std::printf("[l199:cell] q=%d metadata=%s A=%d config=%s "
                    "bchunk=%d/%d S=%d REJECT %s\n",
                    L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",
                    ArtifactTileK, config_name, PPU_B_CHUNK,
                    int(effective_bchunk), split,
                    reject_name(reject, ppu_tactics::Exclusion::None));
        continue;
      }

      // Forming the prepared type triggers all public static guards.  Plants
      // alter exactly one typed seam and are compiled only by the runner's RED arms.
#if defined(L199_PLANT_OMIT_B2_TYPE)
      using LauncherHigh = void;
#else
      using LauncherHigh = High;
#endif
#if defined(L199_PLANT_METADATA_MODE)
      constexpr bool LauncherPacked = !kPackedMetadata;
#else
      constexpr bool LauncherPacked = kPackedMetadata;
#endif
      using Prepared = dense_splitk_parallel_ppu::PreparedMultiformatLauncher<
          Shipping, Tile, Warp, LauncherHigh, LauncherPacked>;
      static_assert(sizeof(Prepared) > 0);
      ++census.typed;

      constexpr uintptr_t AAddress = 0x10000;
      constexpr uintptr_t BAddress = 0x20000;
      constexpr uintptr_t MetadataAddress = 0x30000;
      constexpr uintptr_t ZeroAddress = 0x40000;
      constexpr uintptr_t HighAddress = 0x50000;
      constexpr uintptr_t WorkspaceAddress = 0x80000;
      constexpr uintptr_t DAddress = 0x100000;
      auto const* A = reinterpret_cast<cutlass::half_t const*>(AAddress);
      auto const* B = reinterpret_cast<Low const*>(BAddress);
      void const* metadata = reinterpret_cast<void const*>(MetadataAddress);
      void const* zeros = kPackedMetadata
          ? nullptr : reinterpret_cast<void const*>(ZeroAddress);
      auto* D = reinterpret_cast<cutlass::half_t*>(DAddress);
      auto* workspace = reinterpret_cast<char*>(WorkspaceAddress);
      High const* B2 = nullptr;
      if constexpr (!std::is_void_v<High>) {
        B2 = reinterpret_cast<High const*>(HighAddress);
      }
      dense_splitk_parallel_ppu::WorkspacePlan plan;
      bool const plan_ok = dense_splitk_parallel_ppu::query_workspace_plan(
          kM, kN, split, plan);
      size_t const workspace_bytes = split == 1 ? 0 : plan.partial_bytes;
      auto const issue = Prepared::inspect_arguments(
          A, B, metadata, zeros, D, kM, kN, kK, Selected::GroupSize,
          split, workspace, workspace_bytes, B2);
      bool seam_ok = plan_ok && issue ==
          dense_splitk_parallel_ppu::MultiformatArgumentIssue::None;

      constexpr int WrongGroupSize = Selected::GroupSize == 16 ? 32 : 16;
      auto const wrong_group = Prepared::inspect_arguments(
          A, B, metadata, zeros, D, kM, kN, kK, WrongGroupSize,
          split, workspace, workspace_bytes, B2);
      seam_ok = seam_ok && wrong_group ==
          dense_splitk_parallel_ppu::MultiformatArgumentIssue::
              StaticGroupSizeMismatch;
      ++census.group_reds;

      auto mainloop = Prepared::make_mainloop_arguments(
          A, B, metadata, zeros, kM, kN, kK, Selected::GroupSize, B2);
      bool real_admission = false;
      if (split == 1) {
        using ShippingKernel = typename Shipping::GemmKernel;
        using ShippingGemm = typename Shipping::Gemm;
        using StrideC = typename ShippingKernel::StrideC;
        using StrideD = typename ShippingKernel::StrideD;
        StrideC sC = cutlass::make_cute_packed_stride(
            StrideC{}, cute::make_shape(kM, kN, 1));
        StrideD sD = cutlass::make_cute_packed_stride(
            StrideD{}, cute::make_shape(kM, kN, 1));
        typename ShippingGemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, {kM, kN, kK, 1},
            mainloop,
            {{1.f, 0.f}, static_cast<cutlass::half_t*>(nullptr), sC, D, sD}, 1};
        real_admission = ShippingGemm::can_implement(args) == cutlass::Status::kSuccess;
      } else {
        using SplitKernel = typename Split::GemmKernel;
        using StrideD = typename SplitKernel::StrideD;
        StrideD sP = cutlass::make_cute_packed_stride(
            StrideD{}, cute::make_shape(kM, kN, split));
        auto* partials = reinterpret_cast<float*>(workspace);
        typename SplitKernel::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, {kM, kN, kK, 1},
            mainloop, {partials, sP, partials, sP}, split};
        real_admission = SplitKernel::can_implement(args);
      }

      // Per-family seam REDs.  The broad underlying Arguments types do not
      // validate null data pointers, so the owned launcher must do it.
      if constexpr (!std::is_void_v<High>) {
        auto const missing = Prepared::inspect_arguments(
            A, B, metadata, zeros, D, kM, kN, kK, Selected::GroupSize,
            split, workspace, workspace_bytes, nullptr);
        seam_ok = seam_ok && missing ==
            dense_splitk_parallel_ppu::MultiformatArgumentIssue::MissingHighPlane;
        ++census.b2_reds;

        auto bad_fold = mainloop;
        using StrideB = typename Shipping::CollectiveMainloop::StrideB;
        constexpr int HighFold = Shipping::MainloopPolicy::ArtifactHighFold;
        constexpr int BadFold = HighFold == 1 ? 2 : 1;
        bad_fold.dB2 = cutlass::make_cute_packed_stride(
            StrideB{}, cute::make_shape(kN / BadFold, kK * BadFold, 1));
        bad_fold.dB2_valid = true;
        bool const fold_red = !Shipping::CollectiveMainloop::can_implement(
            cute::make_shape(kM, kN, kK, 1), bad_fold);
        seam_ok = seam_ok && fold_red;
        ++census.fold_reds;
      } else {
        auto const unexpected = Prepared::inspect_arguments(
            A, B, metadata, zeros, D, kM, kN, kK, Selected::GroupSize,
            split, workspace, workspace_bytes,
            reinterpret_cast<void const*>(HighAddress));
        seam_ok = seam_ok && unexpected ==
            dense_splitk_parallel_ppu::MultiformatArgumentIssue::UnexpectedHighPlane;
        ++census.b2_reds;
      }
      if constexpr (kPackedMetadata) {
        auto const bad_zero = Prepared::inspect_arguments(
            A, B, metadata, reinterpret_cast<void const*>(ZeroAddress), D,
            kM, kN, kK, Selected::GroupSize, split, workspace,
            workspace_bytes, B2);
        auto const missing_units = Prepared::inspect_arguments(
            A, B, nullptr, nullptr, D, kM, kN, kK, Selected::GroupSize,
            split, workspace, workspace_bytes, B2);
        seam_ok = seam_ok &&
            bad_zero == dense_splitk_parallel_ppu::MultiformatArgumentIssue::
                PackedMetadataHasSeparateZero &&
            missing_units == dense_splitk_parallel_ppu::MultiformatArgumentIssue::
                MissingMetadata;
        census.metadata_reds += 2;
      } else {
        auto const missing_zero = Prepared::inspect_arguments(
            A, B, metadata, nullptr, D, kM, kN, kK, Selected::GroupSize,
            split, workspace, workspace_bytes, B2);
        seam_ok = seam_ok && missing_zero ==
            dense_splitk_parallel_ppu::MultiformatArgumentIssue::ScaleZeroMissingZero;
        ++census.metadata_reds;
      }

      uint64_t qk = 0;
      bool const exact = exact_fixture<TM, TN, TacticTileK>(split, qk);
      if (!real_admission) reject = Reject::RealAdmission;
      else if (!seam_ok) reject = Reject::ArgumentSeam;
      else if (!exact) reject = Reject::ExactCoverage;
      if (reject != Reject::None) {
        ++census.failures;
        std::printf("[l199:cell] q=%d metadata=%s A=%d config=%s "
                    "bchunk=%d/%d S=%d FALSE_GREEN %s\n",
                    L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",
                    ArtifactTileK, config_name, PPU_B_CHUNK,
                    int(effective_bchunk), split,
                    reject_name(reject, ppu_tactics::Exclusion::None));
      } else {
        ++census.admitted;
        if constexpr (effective_bchunk) ++census.bchunk_effective_on;
        else ++census.bchunk_effective_off;
        census.exact_qk += qk;
        std::printf("[l199:cell] q=%d metadata=%s A=%d folds=%d/%d config=%s "
                    "bchunk=%d/%d S=%d checksum=%llu ADMITTED\n",
                    L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",
                    ArtifactTileK, Shipping::MainloopPolicy::ArtifactLowFold,
                    Shipping::MainloopPolicy::ArtifactHighFold, config_name,
                    PPU_B_CHUNK,
                    int(effective_bchunk), split,
                    static_cast<unsigned long long>(
                        logical_fixture_checksum<TM, TN, TacticTileK>()));
      }
    }
    }
  }
}

template <int ArtifactTileK>
void artifact_census(Census& census) {
  constexpr Reject artifact = artifact_reject<ArtifactTileK>();
  if constexpr (artifact != Reject::None) {
#define L199_REJECT_CONFIG(ID, NAME, TM, TN, WM, WN, STAGES)                 \
    for (int split : kSplits) {                                              \
      ++census.cells; ++census.rejected;                                     \
      std::printf("[l199:cell] q=%d metadata=%s A=%d config=%s "          \
                  "bchunk=%d/REJECT "                                    \
                  "S=%d REJECT %s\n",                                      \
                  L199_QTYPE, kPackedMetadata ? "packed" : "scale-zero",   \
                  ArtifactTileK, NAME, PPU_B_CHUNK, split,                   \
                  reject_name(artifact, ppu_tactics::Exclusion::None));      \
    }
    QUACTLIZE_PPU_DENSE_CONFIGS(L199_REJECT_CONFIG)
#undef L199_REJECT_CONFIG
  } else {
#define L199_TYPED_CONFIG(ID, NAME, TM, TN, WM, WN, STAGES)                 \
    typed_config<ArtifactTileK, ppu_dense_shipping::ConfigId::ID,           \
                 TM, TN, WM, WN, STAGES>(census, NAME);
    QUACTLIZE_PPU_DENSE_CONFIGS(L199_TYPED_CONFIG)
#undef L199_TYPED_CONFIG
  }
}

}  // namespace

int main() {
  Census census;
  artifact_census<32>(census);
  artifact_census<64>(census);
  artifact_census<128>(census);
  artifact_census<256>(census);

  constexpr uint64_t expected_cells =
      kArtifactTileK.size() * ppu_dense_shipping::kConfigs.size() * kSplits.size();
  constexpr int expected_effective_bchunk = PPU_B_CHUNK == 0 ? 0 :
      (L199_QTYPE == 10 ? 0 : (L199_QTYPE == 12 ? -1 : 1));
  bool const bchunk_state_ok = expected_effective_bchunk < 0
      ? census.admitted == 0 && census.bchunk_named_rejects != 0
      : census.admitted != 0 &&
          (expected_effective_bchunk != 0
               ? census.bchunk_effective_on == census.admitted &&
                     census.bchunk_effective_off == 0
               : census.bchunk_effective_off == census.admitted &&
                     census.bchunk_effective_on == 0);
  char const* effective_bchunk_name = census.admitted == 0 ? "REJECT" :
      (census.bchunk_effective_on == census.admitted ? "1" :
       census.bchunk_effective_off == census.admitted ? "0" : "MIXED");
  bool const pass = census.cells == expected_cells &&
      census.cells == census.admitted + census.rejected &&
      census.admitted == census.typed &&
      bchunk_state_ok &&
      census.failures == 0;
  std::printf(
      "[l199] %s q=%d metadata=%s bchunk=%d effective=%s "
      "cells=%llu admitted=%llu "
      "rejected=%llu typed=%llu exact_qk=%llu b2_reds=%llu fold_reds=%llu "
      "metadata_reds=%llu group_reds=%llu bchunk_named_rejects=%llu "
      "failures=%llu "
      "A=fp16 Acc=fp32 D=fp16\n",
      pass ? "PASS" : "FAIL", L199_QTYPE,
      kPackedMetadata ? "packed" : "scale-zero", PPU_B_CHUNK,
      effective_bchunk_name,
      static_cast<unsigned long long>(census.cells),
      static_cast<unsigned long long>(census.admitted),
      static_cast<unsigned long long>(census.rejected),
      static_cast<unsigned long long>(census.typed),
      static_cast<unsigned long long>(census.exact_qk),
      static_cast<unsigned long long>(census.b2_reds),
      static_cast<unsigned long long>(census.fold_reds),
      static_cast<unsigned long long>(census.metadata_reds),
      static_cast<unsigned long long>(census.group_reds),
      static_cast<unsigned long long>(census.bchunk_named_rejects),
      static_cast<unsigned long long>(census.failures));
  return pass ? 0 : 1;
}
