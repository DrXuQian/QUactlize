// L237 -- local production-type closure for the canonical Q2/Q3/Q5/Q6
// per-plane K-pack format.  Compile this source once per L237_QTYPE together
// with the matching PPU_PACKED_FORMAT.  It deliberately instantiates both the
// dense and grouped launch adapters over one shared KPackMainloopPolicy.

#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "fpA_intB_ppu.cuh"
#include "kquant_kpack_offline.hpp"
#include "moe_grouped_ppu.cuh"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

#ifndef L237_QTYPE
#define L237_QTYPE 10
#endif

using namespace cute;
namespace b_delivery =
    cutlass::gemm::collective::detail::quactlize_b_delivery;

#if L237_QTYPE == 10
using Low = cutlass::uint2b_t;
using High = void;
constexpr int kGroup = 16;
constexpr int kTileK = 256;
constexpr int kLowBits = 2;
constexpr int kHighBits = 0;
constexpr int kPackedFormat = 2;
#elif L237_QTYPE == 11
using Low = cutlass::uint2b_t;
using High = cutlass::uint1b_t;
constexpr int kGroup = 16;
constexpr int kTileK = 256;
constexpr int kLowBits = 2;
constexpr int kHighBits = 1;
constexpr int kPackedFormat = 3;
#elif L237_QTYPE == 13
using Low = cutlass::int4b_t;
using High = cutlass::uint1b_t;
constexpr int kGroup = 32;
constexpr int kTileK = 256;
constexpr int kLowBits = 4;
constexpr int kHighBits = 1;
constexpr int kPackedFormat = 1;
#elif L237_QTYPE == 14
using Low = cutlass::int4b_t;
using High = cutlass::uint2b_t;
constexpr int kGroup = 16;
constexpr int kTileK = 128;
constexpr int kLowBits = 4;
constexpr int kHighBits = 2;
constexpr int kPackedFormat = 4;
#else
#error "L237_QTYPE must be Q2_K(10), Q3_K(11), Q5_K(13) or Q6_K(14)"
#endif

#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT != kPackedFormat
// The preprocessor cannot compare against a constexpr.  The C++ assertion
// below is the actual fail-closed binding; this branch only documents why the
// command always supplies PPU_PACKED_FORMAT explicitly.
#endif
static_assert(PPU_PACKED_FORMAT == kPackedFormat,
              "L237 qtype must be compiled with its registry packed format");

using Schedule = ppu_group_schedule::FinegrainedSchedule<kGroup>;

template <int TM, int TN, int WM, int WN, int Stages>
struct ProductionTypes {
  using Tile = Shape<Int<TM>, Int<TN>, Int<kTileK>>;
  using Scale = Shape<Int<TN>, Int<kTileK / kGroup>>;
  using Warp = Shape<Int<WM>, Int<WN>, Int<kTileK>>;
  using Dense = fpa_intb_ppu::DenseKPackKernelTypes<
      ppu_mixed_policy::QuantMode::FinegrainedScaleZero,
      Schedule, Tile, Scale, Warp, Stages, true, Low, High>;
  using Policy = typename Dense::MainloopPolicy;
  using Mainloop = typename Dense::CollectiveMainloop;
  using Builder = typename Policy::CollectiveBuilderType;
  using DefaultOperandB = typename Builder::DefaultOperandB;
  using BDeliveryBinding = typename DefaultOperandB::BDeliveryBinding;
  using BDeliveryShared = typename BDeliveryBinding::Shared;
  using MainloopStorage = typename Mainloop::SharedStorage;
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                ppu_tactics::DenseSpace, Policy>);
  static_assert(ppu_mixed_policy::kernel_policy_valid_v<
                ppu_tactics::GroupedSpace, Policy>);
  static_assert(Dense::SharedStorageSize > 0 &&
                Dense::SharedStorageSize <= ppu_tactics::kBlockSmemBytes);
  static_assert(std::is_same_v<typename Mainloop::BDeliveryPolicy,
                               b_delivery::ProductionBDelivery>);
  static_assert(std::is_same_v<typename DefaultOperandB::BDeliveryTags,
                               b_delivery::ProductionBDelivery>);
  static_assert(std::is_same_v<typename BDeliveryBinding::G2S,
                               typename b_delivery::ProductionBDelivery::G2S>);
  static_assert(std::is_same_v<typename BDeliveryBinding::S2R,
                               typename b_delivery::ProductionBDelivery::S2R>);
  static_assert(std::is_same_v<
                BDeliveryShared,
                typename DefaultOperandB::PhysicalSharedContract>);
  static_assert(std::is_same_v<typename BDeliveryShared::Layout,
                               typename DefaultOperandB::StageSmemLayout>);
  static_assert(std::is_same_v<typename BDeliveryShared::Element,
                               typename Mainloop::BTransportElement>);
  static_assert(BDeliveryShared::bytes_per_stage ==
                    kquant_kpack::PlaneMap<kLowBits, kGroup>::placed_bytes(
                        TN, kTileK),
                "K-pack B-delivery bytes must be one compact low-plane tile");
  static_assert(BDeliveryShared::n_atom == kquant_kpack::kTransportN);
  static_assert(BDeliveryShared::logical_k_atom ==
                kquant_kpack::kReaderPhysicalK * (16 / kLowBits));
  static_assert(BDeliveryShared::alignment_bytes == 32,
                "PPU0010 AIU shared base is a 32-byte contract");
  static_assert(offsetof(MainloopStorage, smem_b) %
                        BDeliveryShared::alignment_bytes == 0,
                "K-quant storage must realize the declared B alignment");
  static_assert(DefaultOperandB::CodesPerWord == 16 / kLowBits &&
                DefaultOperandB::PhysicalK *
                        DefaultOperandB::CodesPerWord ==
                    kTileK);
  static_assert(cute::cosize_v<typename Mainloop::SmemLayoutB> *
                    int(sizeof(typename Mainloop::BTransportElement)) ==
                TN * kTileK * kLowBits / 8 * Stages,
                "K-pack low shared bytes must equal the compact code plane");
  static constexpr bool high_storage_exact() {
    if constexpr (kHighBits == 0) {
      return true;
    } else {
      return cute::cosize_v<typename Mainloop::SmemLayoutB2> *
                 int(sizeof(typename Mainloop::B2TransportElement)) ==
             TN * kTileK * kHighBits / 8 * Stages;
    }
  }
  static_assert(high_storage_exact(),
                "K-pack high shared bytes must equal the compact code plane");
};

// The three shape-selected production defaults cross both PPU0010 atom-M
// families and both dense/grouped schedule inventories.  l236 independently
// covers every admitted N/WN B-fragment geometry.
using Decode = ProductionTypes<8, 128, 8, 32, 3>;
using DenseShipping = ProductionTypes<64, 64, 32, 32, 3>;
using GroupedShipping = ProductionTypes<16, 128, 16, 16, 2>;
using Policy = typename Decode::Policy;
using Mainloop = typename GroupedShipping::Mainloop;
using GroupedTile = typename GroupedShipping::Tile;
using GroupedWarp = typename GroupedShipping::Warp;
using GroupedEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::PPU0010, cutlass::arch::OpClassTensorOp,
    GroupedTile, GroupedWarp,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    cutlass::half_t, cutlass::layout::RowMajor*, 8,
    cutlass::half_t, cutlass::layout::RowMajor*, 8,
    cutlass::epilogue::EpiloguePtrArraySimtVectorized,
    cutlass::epilogue::fusion::LinearCombination<cutlass::half_t, float>
    >::CollectiveOp;
using GroupedKernel = cutlass::gemm::kernel::GemmUniversal<
    moe_grouped_ppu::GroupProblemShape, Mainloop, GroupedEpilogue>;

static_assert(Policy::LowBits == kLowBits && Policy::HighBits == kHighBits);
static_assert(Policy::LowPack == 16 / kLowBits);
static_assert(Policy::HighPack == (kHighBits ? 16 / kHighBits : 0));
static_assert(Policy::ArtifactTileK == 0 &&
              Policy::ArtifactLowFold == 1 &&
              Policy::ArtifactHighFold == 1);
static_assert(Policy::Descriptor::kpack_transpose);
static_assert(Policy::Descriptor::kpack_low == 16 / kLowBits);
static_assert(Policy::Descriptor::kpack_high ==
              (kHighBits ? 16 / kHighBits : 0));
static_assert(Policy::Descriptor::transport_tile_k ==
              kquant_kpack::transport_k(kLowBits, kHighBits));
static_assert(std::is_same_v<
    typename Policy::Descriptor::WeightLayoutType,
    ppu_mixed_policy::KQuantKPackTransposeWeightLayout<
        16 / kLowBits, (kHighBits ? 16 / kHighBits : 0)>>);
static_assert(Mainloop::is_packed_scale);
static_assert(ppu_mixed_policy::kernel_policy_valid_v<
              ppu_tactics::DenseSpace, Policy>);
static_assert(ppu_mixed_policy::kernel_policy_valid_v<
              ppu_tactics::GroupedSpace, Policy>);
static_assert(cute::size<0>(typename GroupedEpilogue::SmemLayout{}) ==
                  cute::size<0>(typename Mainloop::TiledMma::AtomShape_MNK{}) *
                      cute::size<1>(typename Mainloop::TiledMma::ThrLayoutVMNK{}),
              "grouped ptr-array epilogue must bind the same K-pack mainloop");
static_assert(GroupedKernel::SharedStorageSize > 0 &&
              GroupedKernel::SharedStorageSize <= ppu_tactics::kBlockSmemBytes);

int main() {
  std::printf("L237 KQUANT_KPACK production-type PASS qtype=%d "
              "low=%d/pack%d high=%d/pack%d dense=BOUND grouped=BOUND\n",
              L237_QTYPE, kLowBits, 16 / kLowBits, kHighBits,
              kHighBits ? 16 / kHighBits : 0);
}
