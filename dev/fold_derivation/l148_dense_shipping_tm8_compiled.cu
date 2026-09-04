// Exact compiled-type census for INBOX 127's TM8 family.
//
// Build once without PPU_PACKED_SCALE for all five scale-first types, then once per packed format for the matching
// fully-quantized type. DenseKernelTypes is the production generic_launcher's own type authority; reading its
// SharedStorageSize here therefore includes packed metadata staging without instantiating a second formula or the
// launch function's device body. This is the authority missing from the broader host tactic-space census in l147.
#include <cstdio>

#include "fpA_intB_ppu.cuh"
#include "ppu_dense_shipping_policy.hpp"
#include "ppu_format_config.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

namespace {

using QM = fpa_intb_ppu::QuantMode;

template <int Stages, class Low, class High, int GroupSize, int TacticTileK,
          int ArtifactTileK, bool Packed>
bool query(size_t& shared_bytes) {
  constexpr int ScaleGroups = ppu_group_schedule::scale_groups_v<TacticTileK, GroupSize>;
  using Tile = cute::Shape<cute::C<8>, cute::C<128>, cute::C<TacticTileK>>;
  using Scale = cute::Shape<cute::C<128>, cute::C<ScaleGroups>>;
  using Warp = cute::Shape<cute::C<8>, cute::C<32>, cute::C<TacticTileK>>;
  using Kernel = fpa_intb_ppu::DenseKernelTypes<QM::FinegrainedScaleZero,
      ppu_group_schedule::FinegrainedSchedule<GroupSize>, Tile, Scale, Warp, Stages, true,
      Low, High, ArtifactTileK>;
  if constexpr (Packed) {
    static_assert(Kernel::CollectiveMainloop::is_packed_scale,
                  "fully-quantized census must instantiate the production packed-scale collective");
  }
  shared_bytes = Kernel::SharedStorageSize;
  return ppu_tactics::fits_block_smem(Kernel::SharedStorageSize);
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK, bool Packed>
int emit() {
  constexpr auto registered = ppu_formats::for_qtype(QType);
  constexpr char const* mode = Packed ? "fully-quantized" : "scale-first";
  constexpr int HighBits = std::is_void_v<High> ? 0 : cutlass::sizeof_bits<High>::value;
  static_assert(registered.low_bits == cutlass::sizeof_bits<Low>::value &&
                registered.high_bits == HighBits && registered.group_size == GroupSize,
                "l148 format types/group size must come from the production registry row");
  static_assert((Packed ? registered.fully_quantized_tile_k : registered.scale_first_tile_k) == TacticTileK &&
                ArtifactTileK == TacticTileK,
                "l148 tactic/artifact TileK must follow the production reader-default registry");
  int legal = 0;
#define L148_CONFIG(ID, NAME, TM, TN, WM, WN, STAGE) do { \
    static_assert(TM == 8 && TN == 128 && WM == 8 && WN == 32, \
                  "the l148 family macro must remain the exact TM8 ShortWide geometry"); \
    size_t shared_bytes = 0; \
    bool const ok = query<STAGE, Low, High, GroupSize, TacticTileK, ArtifactTileK, Packed>(shared_bytes); \
    std::printf("compiled format=%s mode=%s tk=%d config=%s shared_bytes=%zu verdict=%s", \
                registered.name, mode, TacticTileK, NAME, shared_bytes, ok ? "LEGAL" : "ILLEGAL"); \
    if (!ok) std::printf(" reason=compiled SharedStorageSize exceeds the 256KB block limit"); \
    std::printf("\n"); \
    legal += ok; \
  } while (0);
  QUACTLIZE_PPU_DENSE_M8_SHORTWIDE_CONFIGS(L148_CONFIG);
#undef L148_CONFIG
  return legal;
}

}  // namespace

int main() {
  int legal = 0, cells = 0;
#if defined(L148_SCALE_FIRST)
  legal += emit<10, cutlass::uint2b_t, void, 16, 128, 128, false>(); cells += 6;
  legal += emit<11, cutlass::uint2b_t, cutlass::uint1b_t, 16, 256, 256, false>(); cells += 6;
  legal += emit<12, cutlass::int4b_t, void, 32, 64, 64, false>(); cells += 6;
  legal += emit<13, cutlass::int4b_t, cutlass::uint1b_t, 32, 256, 256, false>(); cells += 6;
  legal += emit<14, cutlass::int4b_t, cutlass::uint2b_t, 16, 128, 128, false>(); cells += 6;
#elif !defined(PPU_PACKED_SCALE) || !defined(PPU_PACKED_FORMAT)
#error "fully-quantized l148 requires PPU_PACKED_SCALE and PPU_PACKED_FORMAT"
#elif PPU_PACKED_FORMAT == 0
  legal += emit<12, cutlass::int4b_t, void, 32, 256, 256, true>(); cells += 6;
#elif PPU_PACKED_FORMAT == 1
  legal += emit<13, cutlass::int4b_t, cutlass::uint1b_t, 32, 256, 256, true>(); cells += 6;
#elif PPU_PACKED_FORMAT == 2
  legal += emit<10, cutlass::uint2b_t, void, 16, 256, 256, true>(); cells += 6;
#elif PPU_PACKED_FORMAT == 3
  legal += emit<11, cutlass::uint2b_t, cutlass::uint1b_t, 16, 256, 256, true>(); cells += 6;
#elif PPU_PACKED_FORMAT == 4
  legal += emit<14, cutlass::int4b_t, cutlass::uint2b_t, 16, 128, 128, true>(); cells += 6;
#else
#error "unknown PPU_PACKED_FORMAT"
#endif
  std::printf("compiled-summary cells=%d legal=%d illegal=%d\n", cells, legal, cells - legal);
  return 0;
}
