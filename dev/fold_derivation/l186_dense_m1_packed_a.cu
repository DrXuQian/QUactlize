// L186 -- local compiled-type oracle for the non-Marlin dense M==1 packed-A shipping path.
//
// This probe instantiates the exact ordinary Q4_K ShortWideM8S3 type used by production and the complete ordinary
// unfolded one-plane M==1 enable matrix that production can reach for Q2/Q4.  It proves:
//   * the old/default schedule and GemmKernel remain exactly the direct pre-packed construction;
//   * the new schedule is a distinct Rows=1 wrapper over that unchanged resident-B contract;
//   * allocation/read geometry is bound to one physical 16-row cube and the same Rows=1 pitch authority.
//
// Runtime source/destination/output coverage lives in l186_dense_m1_packed_a_geometry.cu. Keeping runtime CuTe
// tensor iteration out of this full shipping-type TU matters: the PPU macro stack intentionally forces the full
// device bodies through nvcc's device pass, where host-side coordinate products are not a valid execution oracle.
// Neither half is a PPU execution result; the device opcode remains a box postcondition.

#include <cstdio>
#include <cstddef>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "ppu_format_config.hpp"
#include "ppu_group_schedule.hpp"

namespace {
using namespace cute;
using QM = fpa_intb_ppu::QuantMode;

#ifndef L186_PACK_ROWS
#define L186_PACK_ROWS 1
#endif
#ifndef L186_TILE_M
#define L186_TILE_M 8
#endif

using BaseSchedule = ppu_group_schedule::FinegrainedSchedule<32>;
using Tile = Shape<Int<L186_TILE_M>, _128, _64>;
using Scale = Shape<_128, _2>;
using Warp = Shape<Int<L186_TILE_M>, _32, _64>;
using ExpectedOrdinarySchedule = cutlass::gemm::KernelAiuFold<1, BaseSchedule, 1, 64>;
using ExpectedPackedSchedule = cutlass::gemm::KernelAiuPackedA<L186_PACK_ROWS, ExpectedOrdinarySchedule>;

using Ordinary = fpa_intb_ppu::DenseKernelTypes<
    QM::FinegrainedScaleZero, BaseSchedule, Tile, Scale, Warp, 3, true,
    cutlass::int4b_t, void, 64>;
using Packed = fpa_intb_ppu::DensePackedAKernelTypes<
    L186_PACK_ROWS, QM::FinegrainedScaleZero, BaseSchedule, Tile, Scale, Warp, 3, true,
    cutlass::int4b_t, 64>;

// Reconstruct the pre-packed direct builder rather than comparing two aliases of DenseKernelTypes.  If the default
// MainloopPolicy is ever silently wrapped, this equality fails even if the new packed type still compiles.
using DirectBuilder = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cute::tuple<cutlass::int4b_t, cutlass::half_t, cutlass::half_t>,
    cutlass::layout::ColumnMajorInterleaved<256>, 32,
    float, cute::tuple<Tile, Scale>, Warp, cute::Int<3>, ExpectedOrdinarySchedule>;
using DirectMainloop = typename DirectBuilder::CollectiveOp;
using DirectKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, DirectMainloop, typename Ordinary::CollectiveEpilogue,
    cutlass::gemm::SplitKSerialScheduler>;

static_assert(std::is_same_v<typename Ordinary::MainloopPolicy::KernelSchedule,
                             ExpectedOrdinarySchedule>,
              "M>1/default policy must retain the unwrapped historical schedule type");
static_assert(std::is_same_v<typename Ordinary::CollectiveMainloop, DirectMainloop> &&
              std::is_same_v<typename Ordinary::GemmKernel, DirectKernel>,
              "M>1/default compiled type must remain exactly the direct historical construction");
static_assert(std::is_same_v<typename Packed::MainloopPolicy::KernelSchedule,
                             ExpectedPackedSchedule>,
              "M==1 must carry the independent typed Rows=1 A provider");
static_assert(!std::is_same_v<typename Packed::CollectiveMainloop,
                              typename Ordinary::CollectiveMainloop>,
              "packed and ordinary A providers must not silently collapse to one collective type");

using Mainloop = typename Packed::CollectiveMainloop;
using Mma = typename Mainloop::TiledMma;
static_assert(size<0>(typename Mma::AtomShape_MNK{}) == 8 &&
              size<1>(typename Mma::AtomShape_MNK{}) == 16 &&
              size<2>(typename Mma::AtomShape_MNK{}) == 16,
              "the exact shipping packed-A type must use ppu001 m8n16k16");
static_assert(L186_TILE_M == 8 && L186_PACK_ROWS == 1,
              "the positive control is the exact TM8/Rows1 shipping route");
static_assert(Mainloop::LogicalTileM == 8 && Mainloop::PhysicalATileM == 16 &&
              Mainloop::kACubeH == 16,
              "logical m8 must retain one physical 16-row read cube authority");
static_assert(Mainloop::kPackedA && Mainloop::kAPackRows == 1 &&
              Mainloop::kAPackPitch ==
                  cutlass::gemm::collective::detail::aPackPitchForRows(1),
              "typed writer, allocation and read atom must share the Rows=1 pitch authority");
static_assert(!Ordinary::CollectiveMainloop::kPackedA,
              "the established M>1/default collective must retain the ordinary A provider");
static_assert(Mainloop::kAPackSpan <=
                  cute::cosize_v<typename Mainloop::SmemLayoutAPhysical> &&
              (Mainloop::kACubes == 1 ||
               Mainloop::kAPackSpan <
                   cute::cosize_v<typename Mainloop::SmemLayoutAPhysical>),
              "packed A must cover the full x4 stage footprint and reduce smem when a stage has multiple cubes");
static_assert(offsetof(typename Mainloop::SharedStorage, smem_b) % 32 == 0,
              "packed A must leave the following PPU0010 B shared-memory provider 32-byte aligned");

// PRODUCTION ENABLE MATRIX, not a synthetic cross product.  The backend enables packed A exactly when the low
// plane's *artifact* fold is one.  Q4 therefore enters at A64 and Q2 at A128. Scale-first uses the registry's
// minimum-delivery tactic; fully-quantized uses its TK256 tactic and the versioned arrangement admits each unfolded
// ArtifactTileK divisor below. Q4/A32 and Q2/A32,A64 are intentionally absent because those resident artifacts are
// folded; two-plane formats are intentionally absent because the shipping guard rejects High != void.
template <int QType, class ElementB, int GroupSize, int TacticTileK,
          int ArtifactTileK, int PipelineStages = 3>
struct ProductionPackedACell {
  static constexpr auto Format = ppu_formats::for_qtype(QType);
  using CellTile = Shape<_8, _128, Int<TacticTileK>>;
  using CellScale = Shape<_128, Int<(TacticTileK + GroupSize - 1) / GroupSize>>;
  using CellWarp = Shape<_8, _32, Int<TacticTileK>>;
  using CellOrdinary = fpa_intb_ppu::DenseKernelTypes<
      QM::FinegrainedScaleZero, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      CellTile, CellScale, CellWarp, PipelineStages, true, ElementB, void,
      ArtifactTileK>;
  using CellPacked = fpa_intb_ppu::DensePackedAKernelTypes<
      1, QM::FinegrainedScaleZero, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      CellTile, CellScale, CellWarp, PipelineStages, true, ElementB,
      ArtifactTileK>;
  using CellMainloop = typename CellPacked::CollectiveMainloop;
  using CellPolicy = typename CellPacked::MainloopPolicy;
  using CellMma = typename CellMainloop::TiledMma;

  static_assert(Format.qtype == QType && Format.group_size == GroupSize &&
                Format.low_bits == cutlass::sizeof_bits<ElementB>::value && Format.high_bits == 0,
                "matrix cell must name the actual one-plane shipping registry row");
  static_assert((TacticTileK == Format.scale_first_tile_k ||
                 TacticTileK == Format.fully_quantized_tile_k) &&
                ppu_formats::artifact_tile_k_supported(Format, ArtifactTileK) &&
                ArtifactTileK <= TacticTileK && TacticTileK % ArtifactTileK == 0,
                "matrix cell must be reachable through a production scale-first or arrangement-aware route");
  static_assert(fold::delivery_fold_v<cutlass::sizeof_bits<ElementB>::value, ArtifactTileK> == 1,
                "packed-A matrix admits only the backend's ordinary unfolded artifact branch");
  static_assert(CellPolicy::ArtifactTileK == ArtifactTileK && CellPolicy::ArtifactLowFold == 1 &&
                CellPolicy::HighBits == 0,
                "the compiled policy must retain the production artifact identity");
  static_assert(CellMainloop::kPackedA && CellMainloop::kAPackRows == 1 &&
                CellMainloop::LogicalTileM == 8 && CellMainloop::PhysicalATileM == 16 &&
                CellMainloop::kACubes == TacticTileK / 64 &&
                CellMainloop::kAWrThreads == (TacticTileK / 64) * 8,
                "every admitted cell must instantiate the exact Rows1 physical-m16 provider");
  static_assert(size<0>(typename CellMma::AtomShape_MNK{}) == 8 &&
                size<1>(typename CellMma::AtomShape_MNK{}) == 16 &&
                size<2>(typename CellMma::AtomShape_MNK{}) == 16,
                "every admitted cell must retain the shipping m8n16k16 instruction");
  static_assert(!CellOrdinary::CollectiveMainloop::kPackedA &&
                !std::is_same_v<typename CellOrdinary::CollectiveMainloop, CellMainloop>,
                "matrix must compare a distinct packed provider to the unchanged ordinary type");
  static_assert(offsetof(typename CellMainloop::SharedStorage, smem_b) % 32 == 0,
                "every admitted packed-A cell must preserve B shared-memory alignment");
  static constexpr bool value = true;
};

// Exact type that exposed the rare direct-partial mismatch on the box.  L186
// originally covered only Stages=3; that left the actual s2 symbol outside the
// purported packed-A proof denominator.
using ExactSplitKFailureCell = ProductionPackedACell<
    12, cutlass::int4b_t, 32, 256, 64, 2>;
using ExactSplitKFailureSmemCopy = Copy_Atom<
    PPU0010_TSM_LD_SWZL_M8<cutlass::half_t, 16, 64, true, false,
                            4, 64, 1216>,
    cutlass::half_t>;
static_assert(ExactSplitKFailureCell::CellMainloop::DispatchPolicy::Stages == 2 &&
              ExactSplitKFailureCell::CellMainloop::kACubes == 4 &&
              ExactSplitKFailureCell::CellMainloop::kAPackPitch == 64 &&
              ExactSplitKFailureCell::CellMainloop::kAPackStagePitch == 1216 &&
              std::is_same_v<
                  typename ExactSplitKFailureCell::CellMainloop::SmemCopyAtomA,
                  ExactSplitKFailureSmemCopy>,
              "L186 must instantiate the exact TM8/TK256/Stages2 packed-A mainloop");


constexpr auto kQ2 = ppu_formats::for_qtype(10);
constexpr auto kQ4 = ppu_formats::for_qtype(12);
static_assert(kQ2.scale_first_tile_k == 128 && kQ2.fully_quantized_tile_k == 256 &&
              kQ4.scale_first_tile_k == 64 && kQ4.fully_quantized_tile_k == 256,
              "L186 matrix must follow the shipping registry rather than parallel TileK literals");

// Q4: scale-first A64; FQ tactic TK256 with each supported unfolded arrangement A64/A128/A256.
static_assert(ProductionPackedACell<12, cutlass::int4b_t, 32, kQ4.scale_first_tile_k, 64>::value);
static_assert(ProductionPackedACell<12, cutlass::int4b_t, 32, kQ4.fully_quantized_tile_k, 64>::value);
static_assert(ProductionPackedACell<12, cutlass::int4b_t, 32, kQ4.fully_quantized_tile_k, 128>::value);
static_assert(ProductionPackedACell<12, cutlass::int4b_t, 32, kQ4.fully_quantized_tile_k, 256>::value);
// Q2: scale-first A128; FQ tactic TK256 with each supported unfolded arrangement A128/A256.
static_assert(ProductionPackedACell<10, cutlass::uint2b_t, 16, kQ2.scale_first_tile_k, 128>::value);
static_assert(ProductionPackedACell<10, cutlass::uint2b_t, 16, kQ2.fully_quantized_tile_k, 128>::value);
static_assert(ProductionPackedACell<10, cutlass::uint2b_t, 16, kQ2.fully_quantized_tile_k, 256>::value);

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
using SelectedPackedMetadataCell =
    ProductionPackedACell<10, cutlass::uint2b_t, 16, kQ2.fully_quantized_tile_k, 256>;
#else
using SelectedPackedMetadataCell =
    ProductionPackedACell<12, cutlass::int4b_t, 32, kQ4.fully_quantized_tile_k, 256>;
#endif
static_assert(SelectedPackedMetadataCell::CellMainloop::is_packed_scale,
              "the matching fully-quantized build must consume packed metadata through the same packed-A type");
#endif

}  // namespace

int main() {
  std::printf("[l186:type] ordinary_type_identity=1 packed_distinct=1 "
              "atom=m8n16k16 physical/logical=%d/%d rows=%d pitch=%d span=%d natural=%d "
              "exact_s2_tk256_stage_pitch=%d exact_span=%d exact_natural=%d matrix=7(q4=4,q2=3)\n",
              Mainloop::PhysicalATileM, Mainloop::LogicalTileM, Mainloop::kAPackRows,
              Mainloop::kAPackPitch, Mainloop::kAPackSpan,
              int(cute::cosize_v<typename Mainloop::SmemLayoutAPhysical>),
              ExactSplitKFailureCell::CellMainloop::kAPackStagePitch,
              ExactSplitKFailureCell::CellMainloop::kAPackSpan,
              int(cute::cosize_v<typename ExactSplitKFailureCell::CellMainloop::SmemLayoutAPhysical>));
  std::puts("[l186:type] PASS: exact shipping M1 type is packed; default/M>1 compiled type is identical to the direct historical construction");
  return 0;
}
