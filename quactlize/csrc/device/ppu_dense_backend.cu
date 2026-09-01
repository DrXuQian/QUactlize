// hgcc half of SCALE_FIRST x DENSE: raw host pointers -> the dedicated fpA mixed-input launcher. The offline weight
// reorder lives in ppu_dense_layout.cu so the resident artifact crosses this ABI already in the kernel's layout.
//
// NARROWING THE BUILD TO ONE FORMAT, which is what QUACTLIZE_DENSE_ONLY below is for. Every `#if !defined(
// QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == <qtype>` guard drops the formats that do not match, and the
// qtype ids are the registry's (quactlize/include/ppu_format_config.inc): Q2_K 10, Q3_K 11, Q4_K 12, Q5_K 13,
// Q6_K 14. This translation unit is the slowest thing in the tree to compile, so iterating on ONE format is the
// difference between a change costing a minute and costing five.
//
//     PPU_DEFS=QUACTLIZE_DENSE_ONLY=12 TARGET=ppu_dense_backend ./build.sh      # Q4_K only
//
// THE INVOCATION IS WRITTEN DOWN BECAUSE IT WAS NOT. It discharged this macro from check_switch_macros.py's
// temporary ALLOWED debt: ci/check_format_table_buildable.py's docstring had cited a capability nobody could
// invoke. Measured 2026-08-11 with the
// syntax gate's own flags on this file, which is the evidence that the guards actually drop work rather than
// merely compiling: 10368 diagnostics with no defs against 2232 with QUACTLIZE_DENSE_ONLY=12, a ratio of 4.6
// against the 5 formats it selects between. (Neither run produces an artifact -- the stub's cute:: noise
// prevents that for every file in this tree; see dev/fold_derivation/syntax_check.sh.)
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fpA_intB_ppu.cuh"
#include "moe_grouped_ppu.cuh"
#include "gemv_lowbit/gemv_rt.hpp"
#include "ppu_dense_shipping_policy.hpp"
#include "ppu_q4_kpack4_shipping_policy.hpp"
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 1
#include "ppu_dense_w4_splitk_launch.cuh"
#include "scalefirst_persistent_policy.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_aiu_gemm_mixed_input_persistent.hpp"
#else
#define QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED 0
#endif
#include "ppu_format_config.hpp"
#include "ppu_grouped_configs.inc"
#include "ppu_placed_arrangement.hpp"
#include "quactlize_ppu_device.h"

// The optional collectives this backend INSTANTIATES: it ships Q3_K (uint2+uint1, two planes) and folded
// artifacts alongside the single-plane path. quactlize_actlize.hpp carries the base only.
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "actlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

namespace {

using ppu_gemv::DevBuf;
using half_t = cutlass::half_t;
using QM = fpa_intb_ppu::QuantMode;
using GQM = moe_grouped_ppu::QuantMode;
using GS = moe_grouped_ppu::GroupShape;
using DS = moe_grouped_ppu::DStride;
using Q4PackedUnit = cutlass::gguf_packed::Unit<cutlass::gguf_packed::Fmt::Q4K>;
static_assert(Q4PackedUnit::kUnitBytes == 16, "Q4_K's byte-neutral packed unit is the shipped 16-byte unit");
#if defined(PPU_PACKED_FORMAT)
static constexpr auto kSelectedPackedFmt = cutlass::gguf_packed::Fmt(PPU_PACKED_FORMAT);
static constexpr int kSelectedPackedFormatId = PPU_PACKED_FORMAT;
#else
static constexpr auto kSelectedPackedFmt = cutlass::gguf_packed::Fmt::Q4K;
static constexpr int kSelectedPackedFormatId = 0;
#endif
static constexpr auto kSelectedFormat = ppu_formats::for_packed_format(kSelectedPackedFormatId);
static_assert(kSelectedFormat.qtype >= 0, "PPU_PACKED_FORMAT must name a row in ppu_format_config.inc");
using SelectedPackedUnit = cutlass::gguf_packed::Unit<kSelectedPackedFmt>;

using DenseConfigId = ppu_dense_shipping::ConfigId;
using Kpack4ConfigId = ppu_q4_kpack4_shipping::ConfigId;

constexpr quactlize_ppu_config_v1 kDenseConfigs[] = {
#define QUACTLIZE_PPU_DENSE_CONFIG_ROW(ID, NAME, TM, TN, WM, WN, STAGES) \
  {false, NAME, TM, TN, WM, WN, STAGES},
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_ROW)
#undef QUACTLIZE_PPU_DENSE_CONFIG_ROW
  // The scale-first CUDA-core family consumes the same logical planes but has no tensor tile geometry.
  {true, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME, 0, 0, 0, 0, 0},
};
constexpr DenseConfigId kDefaultDenseConfig = ppu_dense_shipping::kLegacyDefault;
constexpr DenseConfigId kDecodeDefaultDenseConfig = ppu_dense_shipping::kDecodeDefault;
static_assert(sizeof(kDenseConfigs) / sizeof(kDenseConfigs[0]) > 1,
              "libquactlize_ppu must compile a config set, not one frozen tactic");
static_assert(sizeof(kDenseConfigs) / sizeof(kDenseConfigs[0]) == size_t(DenseConfigId::Count) + 1,
              "the dense inventory must contain every tensor-core config followed by its one CUDA tactic");

constexpr int minimum_dense_tile_m() { return ppu_dense_shipping::minimum_tile_m(); }
constexpr int minimum_dense_tile_n() { return ppu_dense_shipping::minimum_tile_n(); }

bool find_dense_config(char const* name, int m, DenseConfigId& config) {
  return ppu_dense_shipping::find_config(name, m, config);
}

DenseConfigId resolve_dense_config(char const* name, int m) {
  DenseConfigId config{};
  if (find_dense_config(name, m, config)) return config;
  std::fprintf(stderr, "[quactlize_ppu] dense config '%s' is not compiled in; declining to default '%s'\n",
               name, kDenseConfigs[0].name);
  return kDefaultDenseConfig;
}

enum class GroupedConfigId {
#define QUACTLIZE_PPU_GROUPED_CONFIG_ID(ID, NAME, TM, TN, WM, WN, STAGES) ID,
  QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_CONFIG_ID)
#undef QUACTLIZE_PPU_GROUPED_CONFIG_ID
  Count,
};

constexpr quactlize_ppu_config_v1 kGroupedConfigs[] = {
#define QUACTLIZE_PPU_GROUPED_CONFIG_ROW(ID, NAME, TM, TN, WM, WN, STAGES) \
  {false, NAME, TM, TN, WM, WN, STAGES},
  QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_CONFIG_ROW)
#undef QUACTLIZE_PPU_GROUPED_CONFIG_ROW
  // The CUDA-core MoE GEMV is one family-level tactic. Its tile fields deliberately carry no meaning.
  {true, QUACTLIZE_PPU_GROUPED_CUDA_CONFIG_NAME, 0, 0, 0, 0, 0},
};
constexpr GroupedConfigId kDefaultGroupedConfig = GroupedConfigId::Default;
static_assert(int(GroupedConfigId::Count) > 1,
              "libquactlize_ppu must compile a grouped config set, not one frozen tactic");
static_assert(sizeof(kGroupedConfigs) / sizeof(kGroupedConfigs[0]) == size_t(GroupedConfigId::Count) + 1,
              "the grouped inventory must contain every tensor-core config followed by its one CUDA tactic");

bool find_grouped_tensor_config(char const* name, GroupedConfigId& config) {
  if (!name || !name[0]) { config = kDefaultGroupedConfig; return true; }
#define QUACTLIZE_PPU_GROUPED_CONFIG_MATCH(ID, NAME, TM, TN, WM, WN, STAGES) \
  if (std::strcmp(name, NAME) == 0) { config = GroupedConfigId::ID; return true; }
  QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_CONFIG_MATCH)
#undef QUACTLIZE_PPU_GROUPED_CONFIG_MATCH
  return false;
}

GroupedConfigId resolve_grouped_config(char const* name) {
  GroupedConfigId config{};
  if (find_grouped_tensor_config(name, config)) return config;
  std::fprintf(stderr,
               "[quactlize_ppu] grouped tensor config '%s' is not compiled in; declining to default '%s'\n",
               name, kGroupedConfigs[0].name);
  return kDefaultGroupedConfig;
}

constexpr size_t align16(size_t value) { return (value + 15) & ~size_t(15); }

bool selected_fully_quantized_qtype(int qtype, int k) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
  bool const paired_unit = kSelectedPackedFormatId == 3 || kSelectedPackedFormatId == 4;
  return qtype == kSelectedFormat.qtype && (!paired_unit || k % 512 == 0);
#else
  (void)qtype; (void)k;
  return false;
#endif
}

size_t dense_workspace_bytes(int m, int n) {
  // One query serves every compiled tactic, so size for the largest possible CTA grid rather than for either
  // shape-selected default. TM8 still stages a physical 16-row A cube; this bound counts output CTAs, not A-smem.
  return size_t(cutlass::ceil_div(m, minimum_dense_tile_m()))
       * cutlass::ceil_div(n, minimum_dense_tile_n()) * sizeof(int);
}

struct GroupedWorkspaceLayout {
  size_t shapes;
  size_t out_ptrs;
  size_t out_strides;
  size_t rows;
  size_t kernel;
  size_t kernel_bytes;
  size_t total;
};

GroupedWorkspaceLayout grouped_workspace_layout(int max_rows, int n, int experts) {
  GroupedWorkspaceLayout layout{};
  size_t cursor = 0;
  layout.shapes = cursor; cursor = align16(cursor + sizeof(GS) * size_t(experts));
  layout.out_ptrs = cursor; cursor = align16(cursor + sizeof(half_t*) * size_t(experts));
  layout.out_strides = cursor; cursor = align16(cursor + sizeof(DS) * size_t(experts));
  layout.rows = cursor; cursor = align16(cursor + sizeof(int) * size_t(experts));
  layout.kernel = cursor;
  size_t const legacy_scheduler_bytes =
      size_t(cutlass::ceil_div(max_rows, 16)) * cutlass::ceil_div(n, 64)
      * size_t(experts) * 64;
  // The selected grouped config is not known at the public workspace-query ABI.
  // TileM=8 is the smallest admitted grouped tile, hence the largest possible
  // directory.  Keep the old workspace bound as well: the non-persistent path
  // continues to use it for its ragged prefix and existing callers see one ABI.
  size_t const persistent_directory_bytes =
      quactlize::moe_directory::workspace_bytes(max_rows, experts, 8);
  layout.kernel_bytes = std::max(legacy_scheduler_bytes, persistent_directory_bytes);
  layout.total = layout.kernel + layout.kernel_bytes;
  return layout;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false,
          int ArtifactTileK = TileK, class MainloopPolicyOverride = void>
int launch_grouped_tactic(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    void const* scale, uint16_t const* zero,
    half_t** out_ptrs, DS* out_strides, int const* rows,
    int max_rows, int n, int k, int experts, GS* shapes, GS const* shapes_host, int const* offsets,
    char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  // The dense side's shared legality guard, applied to the grouped config list before instantiating a collective.
  // Scale-copy slot count is deliberately absent here: the collective caps its logical copy layout to the concrete
  // CTA and statically proves full metadata coverage. Other kernel exclusions still fail closed through return 31.
  constexpr ppu_tactics::Candidate kTactic{
      {ppu_tactics::Format::I4, "grouped-backend",
       ppu_mixed_policy::element_bits_v<Low>, ppu_mixed_policy::element_bits_v<High>},
      TileM, TileN, TileK, WarpM, WarpN,
      ArtifactTileK > 0 ? ArtifactTileK : TileK};
  if constexpr (ppu_tactics::GroupedSpace::kernel_exclusion(kTactic) != ppu_tactics::Exclusion::None) {
    return 31;
  } else {
  constexpr int ScaleGroups = ppu_group_schedule::scale_groups_v<TileK, GroupSize>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>, cute::C<TileK>>;
  using Scale = cute::Shape<cute::C<TileN>, cute::C<ScaleGroups>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>, cute::C<TileK>>;
  bool const launched = moe_grouped_ppu::launch<QuantOp,
                          ppu_group_schedule::FinegrainedSchedule<GroupSize>,
                          Tile, Scale, Warp, Stages, true, Low, High, PackedScale,
                          QueryOnly, RequireUniversalFallback, ArtifactTileK,
                          moe_grouped_ppu::kPersistentBuild,
                          MainloopPolicyOverride>(
      reinterpret_cast<half_t const*>(act), reinterpret_cast<Low const*>(low),
      reinterpret_cast<half_t const*>(scale), reinterpret_cast<half_t const*>(zero),
      out_ptrs, out_strides, rows, max_rows, n, k, experts, GroupSize,
      shapes, shapes_host, offsets, workspace, workspace_bytes, stream,
      [&]() {
        if constexpr (std::is_void_v<High>) return static_cast<High const*>(nullptr);
        else return reinterpret_cast<High const*>(high);
      }(), k, false, 1, false);
  return launched ? 0 : 31;
  }
}

#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
// Grouped K-pack4 keeps the established ragged scheduler, pointer-array
// epilogue and workspace contract.  Only the physical B/metadata mainloop is
// selected here.  Keeping ArtifactTileK=0 is essential: layout=1 bytes have a
// fixed K64 transport but no Xplane artifact-TileK identity.
template <int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_grouped_q4_kpack4_tactic(
    uint16_t const* act, uint8_t const* low, uint8_t const* units,
    half_t** out_ptrs, DS* out_strides, int const* rows,
    int max_rows, int n, int k, int experts, GS* shapes,
    GS const* shapes_host, int const* offsets, char* workspace,
    size_t workspace_bytes, hggcStream_t stream) {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>, cute::C<256>>;
  using Scale = cute::Shape<cute::C<TileN>, cute::C<8>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>, cute::C<256>>;
  using Kpack4Policy = ppu_mixed_policy::Q4KPack4MainloopPolicy<
      GQM::FinegrainedScaleZero, Schedule, Tile, Scale, Warp, Stages,
      true, 0, 0>;
  return launch_grouped_tactic<
      GQM::FinegrainedScaleZero, true, cutlass::int4b_t, void, 32,
      256, TileM, TileN, WarpM, WarpN, Stages, QueryOnly,
      RequireUniversalFallback, 0, Kpack4Policy>(
          act, low, nullptr, units, nullptr, out_ptrs, out_strides, rows,
          max_rows, n, k, experts, shapes, shapes_host, offsets, workspace,
          workspace_bytes, stream);
}

template <bool QueryOnly = false>
int launch_grouped_q4_kpack4_config(
    GroupedConfigId config, uint16_t const* act, uint8_t const* low,
    uint8_t const* units, half_t** out_ptrs, DS* out_strides,
    int const* rows, int max_rows, int n, int k, int experts, GS* shapes,
    GS const* shapes_host, int const* offsets, char* workspace,
    size_t workspace_bytes, hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_GROUPED_KPACK4_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case GroupedConfigId::ID: \
      return launch_grouped_q4_kpack4_tactic< \
          TM, TN, WM, WN, STAGES, QueryOnly, \
          GroupedConfigId::ID == kDefaultGroupedConfig>( \
              act, low, units, out_ptrs, out_strides, rows, max_rows, n, k, \
              experts, shapes, shapes_host, offsets, workspace, \
              workspace_bytes, stream);
    QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_KPACK4_CONFIG_CASE)
#undef QUACTLIZE_PPU_GROUPED_KPACK4_CONFIG_CASE
    case GroupedConfigId::Count: break;
  }
  return 31;
}
#endif

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
// Canonical Q2/Q3/Q5/Q6 K-pack grouped path.  The ragged scheduler,
// ptr-array epilogue and packed-metadata channel are unchanged; the override
// names only the physical per-plane b16 reader.  ArtifactTileK remains zero
// because layout=2 is tactic-independent.
template <class Low, class High, int GroupSize, int TileK,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_grouped_kpack_tactic(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, half_t** out_ptrs, DS* out_strides,
    int const* rows, int max_rows, int n, int k, int experts, GS* shapes,
    GS const* shapes_host, int const* offsets, char* workspace,
    size_t workspace_bytes, hggcStream_t stream) {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<GroupSize>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>, cute::C<TileK>>;
  using Scale = cute::Shape<
      cute::C<TileN>,
      cute::C<ppu_group_schedule::scale_groups_v<TileK, GroupSize>>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>, cute::C<TileK>>;
  using Policy = ppu_mixed_policy::KPackMainloopPolicy<
      GQM::FinegrainedScaleZero, Schedule, Tile, Scale, Warp, Stages,
      true, Low, High, 0, 0>;
  return launch_grouped_tactic<
      GQM::FinegrainedScaleZero, true, Low, High, GroupSize, TileK,
      TileM, TileN, WarpM, WarpN, Stages, QueryOnly,
      RequireUniversalFallback, 0, Policy>(
          act, low, high, units, nullptr, out_ptrs, out_strides, rows,
          max_rows, n, k, experts, shapes, shapes_host, offsets, workspace,
          workspace_bytes, stream);
}

template <class Low, class High, int GroupSize, int TileK,
          bool QueryOnly = false>
int launch_grouped_kpack_config(
    GroupedConfigId config, uint16_t const* act, uint8_t const* low,
    uint8_t const* high, uint8_t const* units, half_t** out_ptrs,
    DS* out_strides, int const* rows, int max_rows, int n, int k,
    int experts, GS* shapes, GS const* shapes_host, int const* offsets,
    char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_GROUPED_KPACK_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case GroupedConfigId::ID: \
      return launch_grouped_kpack_tactic< \
          Low, High, GroupSize, TileK, TM, TN, WM, WN, STAGES, QueryOnly, \
          GroupedConfigId::ID == kDefaultGroupedConfig>( \
              act, low, high, units, out_ptrs, out_strides, rows, max_rows, \
              n, k, experts, shapes, shapes_host, offsets, workspace, \
              workspace_bytes, stream);
    QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_KPACK_CONFIG_CASE)
#undef QUACTLIZE_PPU_GROUPED_KPACK_CONFIG_CASE
    case GroupedConfigId::Count: break;
  }
  return 31;
}
#endif

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK,
          bool QueryOnly = false>
int launch_grouped_config(
    GroupedConfigId config,
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    void const* scale, uint16_t const* zero,
    half_t** out_ptrs, DS* out_strides, int const* rows,
    int max_rows, int n, int k, int experts, GS* shapes, GS const* shapes_host, int const* offsets,
    char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_GROUPED_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case GroupedConfigId::ID: \
      return launch_grouped_tactic<QuantOp, PackedScale, Low, High, GroupSize, TileK, \
                                   TM, TN, WM, WN, STAGES, QueryOnly, \
                                   GroupedConfigId::ID == kDefaultGroupedConfig>( \
          act, low, high, scale, zero, out_ptrs, out_strides, rows, max_rows, n, k, experts, \
          shapes, shapes_host, offsets, workspace, workspace_bytes, stream);
    QUACTLIZE_PPU_GROUPED_CONFIGS(QUACTLIZE_PPU_GROUPED_CONFIG_CASE)
#undef QUACTLIZE_PPU_GROUPED_CONFIG_CASE
    case GroupedConfigId::Count: break;
  }
  return 31;
}

template <class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK, bool PackedScale,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool UseM1PackedA = false,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_dense_tactic(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                        void const* scale, uint16_t const* zero, uint16_t* out,
                        int m, int n, int k, void* workspace, size_t workspace_bytes,
                        hggcStream_t stream) {
  // QUACTLIZE_PPU_DENSE_CONFIGS is crossed with each TileK used by callers. Keep the shared legality guard here for
  // constraints that genuinely make a kernel uninstantiable. Scale-copy capacity is no longer one of them: the
  // mainloop caps its logical copy layout to the concrete CTA and proves full metadata coverage at compile time.
  constexpr ppu_tactics::Candidate kTactic{
      {ppu_tactics::Format::I4, "dense-backend",
       ppu_mixed_policy::element_bits_v<Low>, ppu_mixed_policy::element_bits_v<High>},
      TileM, TileN, TacticTileK, WarpM, WarpN, ArtifactTileK};
  constexpr auto kKernelExclusion = ppu_tactics::DenseSpace::kernel_exclusion(kTactic);
  constexpr auto kProducerExclusion = ppu_tactics::common_producer_exclusion(kTactic);
  static_assert(!RequireUniversalFallback ||
                    (kKernelExclusion == ppu_tactics::Exclusion::None &&
                     kProducerExclusion == ppu_tactics::Exclusion::None),
                "a compiled dense default must survive kernel and producer legality before exact type proof");
  if constexpr (kKernelExclusion != ppu_tactics::Exclusion::None ||
                kProducerExclusion != ppu_tactics::Exclusion::None) {
    // 31 is this file's existing "did not launch". A separate code would be a new ABI meaning for callers that only
    // test truthiness, so every remaining tactic-space exclusion keeps the established result.
    return 31;
  } else {
  constexpr int ScaleGroups = ppu_group_schedule::scale_groups_v<TacticTileK, GroupSize>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>, cute::C<TacticTileK>>;
  using ScaleTile = cute::Shape<cute::C<TileN>, cute::C<ScaleGroups>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>, cute::C<TacticTileK>>;
  constexpr bool kOrdinaryOnePlane =
      std::is_void_v<High> &&
      fold::delivery_fold_v<ppu_mixed_policy::element_bits_v<Low>, ArtifactTileK> == 1;
  if constexpr (UseM1PackedA && kOrdinaryOnePlane) {
    // Runtime M selects between two compile-time kernel types.  The list-valid query and the real launch both enter
    // this exact function, so neither can silently inspect the ordinary type and launch the packed one (or vice
    // versa).  M=2..7 falls through to the unchanged DenseKernelTypes instantiation below.
    if (m == 1) {
      using PackedKernelTypes = fpa_intb_ppu::DensePackedAKernelTypes<1,
          QM::FinegrainedScaleZero, ppu_group_schedule::FinegrainedSchedule<GroupSize>,
          Tile, ScaleTile, Warp, Stages, true, Low, ArtifactTileK>;
      // This branch is an exact M==1 route, even when its containing config is
      // also the ordinary universal fallback. Keep the Rows1 type exact and
      // let the Rows0 call below carry that fallback proof.
      bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero,
          ppu_group_schedule::FinegrainedSchedule<GroupSize>,
          Tile, ScaleTile, Warp, Stages, true,
          Low, High, PackedScale, QueryOnly, false, ArtifactTileK,
          PackedKernelTypes>(
              reinterpret_cast<half_t const*>(act), reinterpret_cast<Low const*>(low),
              reinterpret_cast<half_t const*>(scale), reinterpret_cast<half_t const*>(zero),
              reinterpret_cast<half_t*>(out),
              m, n, k, GroupSize, 1, static_cast<char*>(workspace), workspace_bytes, stream);
      return launched ? 0 : 31;
    }
  }
  bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero,
      ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, ScaleTile, Warp, Stages, true,
      Low, High, PackedScale, QueryOnly, RequireUniversalFallback, ArtifactTileK>(
          reinterpret_cast<half_t const*>(act), reinterpret_cast<Low const*>(low),
          reinterpret_cast<half_t const*>(scale), reinterpret_cast<half_t const*>(zero),
          reinterpret_cast<half_t*>(out),
          m, n, k, GroupSize, 1, static_cast<char*>(workspace), workspace_bytes, stream,
          [&]() {
            if constexpr (std::is_void_v<High>) return static_cast<High const*>(nullptr);
            else return reinterpret_cast<High const*>(high);
          }());
  return launched ? 0 : 31;
  }
}

template <class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK,
          bool PackedScale, bool QueryOnly = false>
int launch_dense_config(DenseConfigId config, uint16_t const* act, uint8_t const* low, uint8_t const* high,
                        void const* scale, uint16_t const* zero, uint16_t* out,
                        int m, int n, int k, void* workspace, size_t workspace_bytes,
                        hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_DENSE_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case DenseConfigId::ID: \
      return launch_dense_tactic<Low, High, GroupSize, TacticTileK, ArtifactTileK, PackedScale, \
                                 TM, TN, WM, WN, STAGES, \
                                 (DenseConfigId::ID == kDecodeDefaultDenseConfig), QueryOnly, \
                                 (DenseConfigId::ID == kDefaultDenseConfig || \
                                  DenseConfigId::ID == kDecodeDefaultDenseConfig) && \
                                     ArtifactTileK == TacticTileK>( \
          act, low, high, scale, zero, out, m, n, k, workspace, workspace_bytes, stream);
    QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_CASE)
#undef QUACTLIZE_PPU_DENSE_CONFIG_CASE
    case DenseConfigId::Count: break;
  }
  return 31;
}

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
template <class Low, class High, int GroupSize, int TacticTileK,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_dense_kpack_tactic(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, uint16_t* out, int m, int n, int k,
    void* workspace, size_t workspace_bytes, hggcStream_t stream) {
  constexpr ppu_tactics::Candidate kTactic{
      {ppu_tactics::Format::I4, "dense-kpack",
       ppu_mixed_policy::element_bits_v<Low>,
       ppu_mixed_policy::element_bits_v<High>},
      TileM, TileN, TacticTileK, WarpM, WarpN, TacticTileK};
  constexpr auto kKernelExclusion =
      ppu_tactics::DenseSpace::kernel_exclusion(kTactic);
  constexpr auto kProducerExclusion =
      ppu_tactics::common_producer_exclusion(kTactic);
  static_assert(!RequireUniversalFallback ||
                    (kKernelExclusion == ppu_tactics::Exclusion::None &&
                     kProducerExclusion == ppu_tactics::Exclusion::None),
                "the default dense K-pack tactic must be statically legal");
  if constexpr (kKernelExclusion != ppu_tactics::Exclusion::None ||
                kProducerExclusion != ppu_tactics::Exclusion::None) {
    return 31;
  } else {
    using Schedule = ppu_group_schedule::FinegrainedSchedule<GroupSize>;
    using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>,
                             cute::C<TacticTileK>>;
    using Scale = cute::Shape<
        cute::C<TileN>, cute::C<ppu_group_schedule::scale_groups_v<
                               TacticTileK, GroupSize>>>;
    using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>,
                             cute::C<TacticTileK>>;
    using Types = fpa_intb_ppu::DenseKPackKernelTypes<
        QM::FinegrainedScaleZero, Schedule, Tile, Scale, Warp, Stages,
        true, Low, High, 0, 0>;
    bool const launched = fpa_intb_ppu::generic_launcher<
        QM::FinegrainedScaleZero, Schedule, Tile, Scale, Warp, Stages, true,
        Low, High, true, QueryOnly, RequireUniversalFallback, 0, Types>(
            reinterpret_cast<half_t const*>(act),
            reinterpret_cast<Low const*>(low),
            reinterpret_cast<half_t const*>(units), nullptr,
            reinterpret_cast<half_t*>(out), m, n, k, GroupSize, 1,
            static_cast<char*>(workspace), workspace_bytes, stream,
            [&]() {
              if constexpr (std::is_void_v<High>)
                return static_cast<High const*>(nullptr);
              else
                return reinterpret_cast<High const*>(high);
            }());
    return launched ? 0 : 31;
  }
}

template <class Low, class High, int GroupSize, int TacticTileK,
          bool QueryOnly = false>
int launch_dense_kpack_config(
    DenseConfigId config, uint16_t const* act, uint8_t const* low,
    uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, void* workspace, size_t workspace_bytes,
    hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_DENSE_KPACK_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case DenseConfigId::ID: \
      return launch_dense_kpack_tactic< \
          Low, High, GroupSize, TacticTileK, TM, TN, WM, WN, STAGES, \
          QueryOnly, \
          DenseConfigId::ID == kDefaultDenseConfig || \
              DenseConfigId::ID == kDecodeDefaultDenseConfig>( \
                  act, low, high, units, out, m, n, k, workspace, \
                  workspace_bytes, stream);
    QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_KPACK_CONFIG_CASE)
#undef QUACTLIZE_PPU_DENSE_KPACK_CONFIG_CASE
    case DenseConfigId::Count: break;
  }
  return 31;
}
#endif

// K-pack4 is a physical sibling of the Xplane collective, not an
// ArtifactTileK value.  Keep its exact kernel type and logical stride
// construction in a separate path so the established v1 dispatcher cannot
// accidentally reinterpret layout=1 bytes through a fold-derived reader.
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
template <int APackRows, int TileM, int TileN, int TacticTileK,
          int WarpM, int WarpN, int Stages, bool QueryOnly = false>
int launch_dense_q4_kpack4_exact(
    uint16_t const* act, uint8_t const* low, uint8_t const* units,
    uint16_t* out, int m, int n, int k, void* workspace,
    size_t workspace_bytes, hggcStream_t stream, int split_k_slices) {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>,
                           cute::C<TacticTileK>>;
  using ScaleTile = cute::Shape<
      cute::C<TileN>, cute::C<ppu_group_schedule::scale_groups_v<
                           TacticTileK, 32>>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>,
                           cute::C<TacticTileK>>;
  using Shipping = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
      QM::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      Stages, true, APackRows, 0>;
  using Mainloop = typename Shipping::CollectiveMainloop;
  using GemmKernel = typename Shipping::GemmKernel;
  using Gemm = typename Shipping::Gemm;
  using SplitTypes = dense_splitk_parallel_ppu::KernelTypes<Shipping, Tile, Warp>;
  using SplitKernel = typename SplitTypes::GemmKernel;
  using SplitGemm = typename SplitTypes::Gemm;
  using Reduction = typename SplitTypes::Reduction;

  static_assert(APackRows == 0 || APackRows == 1,
                "shipping K-pack4 A provider is ordinary or one-row packed");
  static_assert(Mainloop::is_packed_scale,
                "fully-quantized K-pack4 must retain packed Q4_K metadata");
  static_assert(
      std::is_same_v<typename Shipping::MainloopPolicy::Descriptor::BProviderType,
                     ppu_mixed_policy::KPack4TransposedBProvider>,
      "arrangement-v2 layout=K-pack4 must select the transposed provider");
  if constexpr (GemmKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
    return 31;
  }
  if constexpr (SplitKernel::SharedStorageSize > ppu_tactics::kBlockSmemBytes) {
    return 31;
  }
  if constexpr (APackRows == 1) {
    if (m != 1) return 31;
  }
  if (split_k_slices != 1 && split_k_slices != 2 &&
      split_k_slices != 4 && split_k_slices != 8) {
    return 31;
  }
  if (split_k_slices > 1) {
    auto const partition = cutlass::gemm::kernel::fixed_splitk::make_params(
        cutlass::ceil_div(m, TileM) * cutlass::ceil_div(n, TileN),
        k / TacticTileK, split_k_slices);
    if (!partition.is_valid() ||
        int(partition.k_tiles_per_split) < Stages - 1) {
      return 31;
    }
  }
  if constexpr (QueryOnly) return 0;
  else {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using StrideS = typename Mainloop::StrideScale;
    StrideA sA = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    // The K-pack4 provider interprets this logical (N,K) stride together with
    // its descriptor-owned physical map.  ArtifactLowFold deliberately stays
    // one; folding this stride would create a second, contradictory format.
    StrideB sB = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n, k, 1));
    StrideS sS = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(n, k / 32, 1));
    StrideC sC = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(m, n, 1));
    StrideD sD = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(m, n, 1));
    typename Mainloop::Arguments mainloop{
        reinterpret_cast<half_t const*>(act), sA,
        reinterpret_cast<cutlass::int4b_t const*>(low), sB,
        reinterpret_cast<half_t const*>(units), sS, 32,
        static_cast<half_t const*>(nullptr)};
    if (split_k_slices == 1) {
      typename Gemm::Arguments args{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k, 1}, mainloop,
          {{1.f, 0.f}, static_cast<half_t*>(nullptr), sC,
           reinterpret_cast<half_t*>(out), sD},
          1};
      Gemm gemm;
      if (Gemm::can_implement(args) != cutlass::Status::kSuccess ||
          Gemm::get_workspace_size(args) > workspace_bytes ||
          gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess ||
          gemm.run(stream) != cutlass::Status::kSuccess) {
        return 31;
      }
      return 0;
    }
    dense_splitk_parallel_ppu::WorkspacePlan plan;
    if (!dense_splitk_parallel_ppu::query_workspace_plan(
            m, n, split_k_slices, plan) || !workspace ||
        workspace_bytes < plan.partial_bytes) {
      return 31;
    }
    using PartialStride = typename SplitKernel::StrideD;
    PartialStride sP = cutlass::gemm::kernel::detail::
        make_compact_fp32_partial_stride<PartialStride>(m, n);
    float* partials = reinterpret_cast<float*>(workspace);
    typename SplitGemm::Arguments producer_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1}, mainloop, {partials, sP, partials, sP},
        split_k_slices};
    typename Reduction::Arguments reducer_args{
        m, n, split_k_slices, partials, workspace_bytes,
        reinterpret_cast<half_t*>(out), n};
    SplitGemm producer;
    Reduction reducer;
    if (SplitGemm::can_implement(producer_args) != cutlass::Status::kSuccess ||
        SplitGemm::get_workspace_size(producer_args) != 0 ||
        producer.initialize(producer_args, nullptr, stream) != cutlass::Status::kSuccess ||
        Reduction::can_implement(reducer_args) != cutlass::Status::kSuccess ||
        reducer.initialize(reducer_args) != cutlass::Status::kSuccess) {
      return 31;
    }
    return cutlass::gemm::device::splitk_parallel::
               launch_main_then_reduce_same_stream(
                   [&](hggcStream_t launch_stream) {
                     return producer.run(launch_stream);
                   }, reducer, stream) == cutlass::Status::kSuccess
        ? 0 : 31;
  }
}

template <bool QueryOnly = false>
int launch_dense_q4_kpack4_config(
    Kpack4ConfigId config, uint16_t const* act, uint8_t const* low,
    uint8_t const* units, uint16_t* out, int m, int n, int k,
    void* workspace, size_t workspace_bytes, hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_KPACK4_CONFIG_CASE(ID, NAME, TM, TN, TK, WM, WN, STAGES, SPLIT) \
    case Kpack4ConfigId::ID: \
      return launch_dense_q4_kpack4_exact< \
          0, TM, TN, TK, WM, WN, STAGES, QueryOnly>( \
          act, low, units, out, m, n, k, workspace, workspace_bytes, stream, SPLIT);
    QUACTLIZE_PPU_Q4_KPACK4_CONFIGS(QUACTLIZE_PPU_KPACK4_CONFIG_CASE)
#undef QUACTLIZE_PPU_KPACK4_CONFIG_CASE
    case Kpack4ConfigId::Count: break;
  }
  return 31;
}
#endif

#if !defined(PPU_PACKED_SCALE) || (PPU_PACKED_SCALE == 0)
template <bool QueryOnly = false>
int launch_scalefirst_q4_kpack4_persistent(
    uint16_t const* act, uint8_t const* low, uint16_t const* scale,
    uint16_t const* zero, uint16_t* out, int m, int n, int k,
    hggcStream_t stream) {
  using Schedule = ppu_group_schedule::FinegrainedSchedule<32>;
  using Tile = cute::Shape<cute::_64, cute::_64, cute::_64>;
  using ScaleTile = cute::Shape<cute::_64, cute::_2>;
  using Warp = cute::Shape<cute::_64, cute::_32, cute::_64>;
  using Shipping = fpa_intb_ppu::DenseQ4KPack4KernelTypes<
      QM::FinegrainedScaleZero, Schedule, Tile, ScaleTile, Warp,
      3, true, 0, 0>;
  using Mainloop = typename Shipping::CollectiveMainloop;
  using PersistentKernel = cutlass::gemm::kernel::PersistentMixedInputKernel<
      cute::Shape<int, int, int, int>, Mainloop,
      typename Shipping::CollectiveEpilogue>;
  using PersistentGemm = cutlass::gemm::device::GemmUniversalAdapter<
      PersistentKernel>;
  static_assert(!Mainloop::is_packed_scale,
                "ScaleFirst K-pack4 must consume fp16 scale/zero planes, not packed units");
  static_assert(PersistentKernel::SharedStorageSize <=
                    ppu_tactics::kBlockSmemBytes,
                "shipping ScaleFirst K-pack4 persistent kernel must fit one block");
  if (m < 64 || n <= 0 || k <= 0 || n % 256 || k % 256) return 31;
  if constexpr (QueryOnly) return 0;
  else {
    using StrideA = typename PersistentKernel::StrideA;
    using StrideB = typename PersistentKernel::StrideB;
    using StrideC = typename PersistentKernel::StrideC;
    using StrideD = typename PersistentKernel::StrideD;
    using StrideS = typename Mainloop::StrideScale;
    StrideA sA = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    StrideB sB = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n, k, 1));
    StrideS sS = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(n, k / 32, 1));
    StrideC sC = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(m, n, 1));
    StrideD sD = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(m, n, 1));
    typename Mainloop::Arguments mainloop{
        reinterpret_cast<half_t const*>(act), sA,
        reinterpret_cast<cutlass::int4b_t const*>(low), sB,
        reinterpret_cast<half_t const*>(scale), sS, 32,
        reinterpret_cast<half_t const*>(zero)};
    int const occupancy = PersistentGemm::maximum_active_blocks();
    int const device_id = 0;
    int const cu = cutlass::KernelHardwareInfo::
        query_device_multiprocessor_count(device_id);
    std::uint64_t const q =
        std::uint64_t(cutlass::ceil_div(m, 64)) *
        std::uint64_t(cutlass::ceil_div(n, 64));
    int const blocks_per_cu = std::min(occupancy, 8);
    std::uint64_t const grid = quactlize::scalefirst_policy::
        capacity_grid(q, cu, blocks_per_cu);
    if (occupancy <= 0 || cu <= 0 || blocks_per_cu <= 0 || grid == 0 ||
        grid > std::uint64_t(INT32_MAX)) return 31;
    typename PersistentGemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1}, mainloop,
        {{1.f, 0.f}, static_cast<half_t*>(nullptr), sC,
         reinterpret_cast<half_t*>(out), sD},
        cutlass::KernelHardwareInfo{device_id, cu}, {}, occupancy,
        std::uint32_t(grid)};
    PersistentGemm gemm;
    if (PersistentGemm::can_implement(args) != cutlass::Status::kSuccess ||
        PersistentGemm::get_workspace_size(args) != 0 ||
        gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess ||
        gemm.run(stream) != cutlass::Status::kSuccess) {
      return 31;
    }
    return 0;
  }
}
#endif
#endif

constexpr int qtype_group_size(int qtype) {
  return ppu_formats::for_qtype(qtype).group_size;
}

bool tensor_problem_domain(int m, int n, int k, int group_size, int qtype) {
  // These are the public entries' resident-artifact constraints, not tile-size constraints. M/N tails are
  // predicated by both kernels, so a TileN larger than N is legal; the current producer/consumer ABI nevertheless
  // admits only the interleaved artifact selected by N,K multiples of 256.
  return m > 0 && n > 0 && k > 0 && n % 256 == 0 && k % 256 == 0 &&
         group_size == qtype_group_size(qtype);
}

template <class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK, bool PackedScale>
bool dense_config_type_valid(DenseConfigId config, int m, int n, int k) {
  return launch_dense_config<Low, High, GroupSize, TacticTileK, ArtifactTileK, PackedScale, true>(
      config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      m, n, k, nullptr, 0, nullptr) == 0;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK>
bool grouped_config_type_valid(GroupedConfigId config, int max_rows, int n, int k, int experts) {
  return launch_grouped_config<QuantOp, PackedScale, Low, High, GroupSize, TileK, true>(
      config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      max_rows, n, k, experts, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == 0;
}

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
template <class Low, class High, int GroupSize, int TileK>
bool dense_kpack_config_type_valid(
    DenseConfigId config, int m, int n, int k) {
  return launch_dense_kpack_config<Low, High, GroupSize, TileK, true>(
             config, nullptr, nullptr, nullptr, nullptr, nullptr,
             m, n, k, nullptr, 0, nullptr) == 0;
}

template <class Low, class High, int GroupSize, int TileK>
bool grouped_kpack_config_type_valid(
    GroupedConfigId config, int max_rows, int n, int k, int experts) {
  return launch_grouped_kpack_config<Low, High, GroupSize, TileK, true>(
             config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
             nullptr, max_rows, n, k, experts, nullptr, nullptr, nullptr,
             nullptr, 0, nullptr) == 0;
}
#endif

#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
bool grouped_q4_kpack4_config_type_valid(
    GroupedConfigId config, int max_rows, int n, int k, int experts) {
  return launch_grouped_q4_kpack4_config<true>(
      config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, max_rows,
      n, k, experts, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == 0;
}
#endif

bool dense_lowbit_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype) {
  if (!tensor_problem_domain(m, n, k, group_size, qtype)) return false;
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    case 10: return dense_config_type_valid<cutlass::uint2b_t, void, 16,
        ppu_formats::for_qtype(10).scale_first_tile_k,
        ppu_formats::for_qtype(10).scale_first_tile_k, false>(config, m, n, k);
#else
    case 10: return dense_config_type_valid<cutlass::uint2b_t, void, 16,
        ppu_formats::for_qtype(10).scale_first_tile_k,
        ppu_formats::for_qtype(10).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    case 11: return false;
#else
    case 11: return dense_config_type_valid<cutlass::uint2b_t, cutlass::uint1b_t, 16,
        ppu_formats::for_qtype(11).scale_first_tile_k,
        ppu_formats::for_qtype(11).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
    case 12: return dense_config_type_valid<cutlass::int4b_t, void, 32,
        ppu_formats::for_qtype(12).scale_first_tile_k,
        ppu_formats::for_qtype(12).scale_first_tile_k, false>(config, m, n, k);
#else
    case 12: return dense_config_type_valid<cutlass::int4b_t, void, 32,
        ppu_formats::for_qtype(12).scale_first_tile_k,
        ppu_formats::for_qtype(12).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    case 13: return false;
#else
    case 13: return dense_config_type_valid<cutlass::int4b_t, cutlass::uint1b_t, 32,
        ppu_formats::for_qtype(13).scale_first_tile_k,
        ppu_formats::for_qtype(13).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    case 14: return false;
#else
    case 14: return dense_config_type_valid<cutlass::int4b_t, cutlass::uint2b_t, 16,
        ppu_formats::for_qtype(14).scale_first_tile_k,
        ppu_formats::for_qtype(14).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
    default: return false;
  }
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK>
bool dense_fully_quantized_config_type_valid(DenseConfigId config, int m, int n, int k) {
  if constexpr (!ppu_arrangements::static_packed_tensor_reader_supported<
                    QType, TacticTileK, ArtifactTileK>()) {
    return false;
  } else {
    return dense_config_type_valid<Low, High, GroupSize, TacticTileK, ArtifactTileK, true>(
        config, m, n, k);
  }
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK>
bool dense_fully_quantized_config_for_artifact_valid(
    DenseConfigId config, int m, int n, int k, int artifact_tile_k) {
  switch (artifact_tile_k) {
    case 32: return dense_fully_quantized_config_type_valid<
        QType, Low, High, GroupSize, TacticTileK, 32>(config, m, n, k);
    case 64: return dense_fully_quantized_config_type_valid<
        QType, Low, High, GroupSize, TacticTileK, 64>(config, m, n, k);
    case 128: return dense_fully_quantized_config_type_valid<
        QType, Low, High, GroupSize, TacticTileK, 128>(config, m, n, k);
    case 256: return dense_fully_quantized_config_type_valid<
        QType, Low, High, GroupSize, TacticTileK, 256>(config, m, n, k);
    default: return false;
  }
}

bool dense_fully_quantized_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement) {
  if (!tensor_problem_domain(m, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, ppu_formats::for_qtype(qtype).fully_quantized_tile_k)) return false;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return dense_fully_quantized_config_for_artifact_valid<12, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
          config, m, n, k, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 2
  return dense_fully_quantized_config_for_artifact_valid<10, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
          config, m, n, k, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 1
  return dense_fully_quantized_config_for_artifact_valid<13, cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
          config, m, n, k, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 3
  return dense_fully_quantized_config_for_artifact_valid<11, cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
          config, m, n, k, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 4
  return dense_fully_quantized_config_for_artifact_valid<14, cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
          config, m, n, k, arrangement->artifact_tile_k);
#else
  return false;
#endif
#else
  (void)config;
  return false;
#endif
}

bool dense_fully_quantized_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype) {
  auto const arrangement = ppu_arrangements::legacy_fully_quantized_default(qtype);
  return dense_fully_quantized_config_valid(
      config, m, n, k, group_size, qtype, &arrangement);
}

bool dense_fully_quantized_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (!tensor_problem_domain(m, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return false;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    quactlize_ppu_placed_arrangement_v1 const legacy{
        QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
        arrangement->bits, arrangement->artifact_tile_k,
        arrangement->high_bits};
    return dense_fully_quantized_config_valid(
        config, m, n, k, group_size, qtype, &legacy);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
#if PPU_PACKED_FORMAT == 2
    return qtype == 10 && dense_kpack_config_type_valid<
        cutlass::uint2b_t, void, 16, 256>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 3
    return qtype == 11 && dense_kpack_config_type_valid<
        cutlass::uint2b_t, cutlass::uint1b_t, 16, 256>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 1
    return qtype == 13 && dense_kpack_config_type_valid<
        cutlass::int4b_t, cutlass::uint1b_t, 32, 256>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 4
    return qtype == 14 && dense_kpack_config_type_valid<
        cutlass::int4b_t, cutlass::uint2b_t, 16, 128>(config, m, n, k);
#else
    return false;
#endif
#else
    return false;
#endif
  }
  if (arrangement->layout !=
          QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 || qtype != 12) {
    return false;
  }
  (void)config;
  return false;
}

bool q4_kpack4_config_valid(
    Kpack4ConfigId config, int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (!tensor_problem_domain(m, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k,
          ppu_formats::for_qtype(qtype).fully_quantized_tile_k) ||
      arrangement->layout != QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1 ||
      qtype != 12) {
    return false;
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  return launch_dense_q4_kpack4_config<true>(
             config, nullptr, nullptr, nullptr, nullptr, m, n, k,
             nullptr, 0, nullptr) == 0;
#else
  (void)config;
  return false;
#endif
}

bool grouped_lowbit_config_valid(
    GroupedConfigId config, int total_rows, int n, int k, int group_size,
    int experts, int max_rows, int qtype) {
  if (experts <= 0 || max_rows <= 0 ||
      !tensor_problem_domain(total_rows, n, k, group_size, qtype)) return false;
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    case 10: return false;
#else
    case 10: return grouped_config_type_valid<GQM::FinegrainedScaleOnly, false,
        cutlass::uint2b_t, void, 16, ppu_formats::for_qtype(10).scale_first_tile_k>(
            config, max_rows, n, k, experts);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    case 11: return false;
#else
    case 11: return grouped_config_type_valid<GQM::FinegrainedScaleOnly, false,
        cutlass::uint2b_t, cutlass::uint1b_t, 16, ppu_formats::for_qtype(11).scale_first_tile_k>(
            config, max_rows, n, k, experts);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
    case 12: return grouped_config_type_valid<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, void, 32, ppu_formats::for_qtype(12).scale_first_tile_k>(
            config, max_rows, n, k, experts);
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    case 13: return false;
#else
    case 13: return grouped_config_type_valid<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, cutlass::uint1b_t, 32, ppu_formats::for_qtype(13).scale_first_tile_k>(
            config, max_rows, n, k, experts);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    case 14: return false;
#else
    case 14: return grouped_config_type_valid<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, cutlass::uint2b_t, 16, ppu_formats::for_qtype(14).scale_first_tile_k>(
            config, max_rows, n, k, experts);
#endif
#endif
    default: return false;
  }
}

bool grouped_fully_quantized_config_valid(
    GroupedConfigId config, int total_rows, int n, int k, int group_size,
    int experts, int max_rows, int qtype) {
  if (experts <= 0 || max_rows <= 0 ||
      !tensor_problem_domain(total_rows, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k)) return false;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return grouped_config_type_valid<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 2
  return grouped_config_type_valid<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 1
  return grouped_config_type_valid<GQM::FinegrainedScaleZero, true,
      cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 3
  return grouped_config_type_valid<GQM::FinegrainedScaleZero, true,
      cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 4
  return grouped_config_type_valid<GQM::FinegrainedScaleZero, true,
      cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(config, max_rows, n, k, experts);
#else
  return false;
#endif
#else
  (void)config;
  return false;
#endif
}

bool grouped_fully_quantized_config_valid(
    GroupedConfigId config, int total_rows, int n, int k, int group_size,
    int experts, int max_rows, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (experts <= 0 || max_rows <= 0 || max_rows > total_rows ||
      !tensor_problem_domain(total_rows, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return false;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    // The legacy grouped kernel already owns the exact shipping Xplane reader
    // in this same binary. Unlike dense, grouped has not instantiated the four
    // artifact-TileK reader variants, so admit precisely that compiled type.
    return arrangement->artifact_tile_k == tactic_tile_k &&
        grouped_fully_quantized_config_valid(
            config, total_rows, n, k, group_size, experts, max_rows, qtype);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
#if PPU_PACKED_FORMAT == 2
    return qtype == 10 && grouped_kpack_config_type_valid<
        cutlass::uint2b_t, void, 16, 256>(
            config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 3
    return qtype == 11 && grouped_kpack_config_type_valid<
        cutlass::uint2b_t, cutlass::uint1b_t, 16, 256>(
            config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 1
    return qtype == 13 && grouped_kpack_config_type_valid<
        cutlass::int4b_t, cutlass::uint1b_t, 32, 256>(
            config, max_rows, n, k, experts);
#elif PPU_PACKED_FORMAT == 4
    return qtype == 14 && grouped_kpack_config_type_valid<
        cutlass::int4b_t, cutlass::uint2b_t, 16, 128>(
            config, max_rows, n, k, experts);
#else
    return false;
#endif
#else
    return false;
#endif
  }
  if (arrangement->layout !=
          QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
    return false;
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  return grouped_q4_kpack4_config_type_valid(
      config, max_rows, n, k, experts);
#else
  (void)config;
  return false;
#endif
}

quactlize_ppu_config_v2 config_v2(quactlize_ppu_config_v1 const& config, int tile_k) {
  return {config.enable_cuda_kernel, config.name, config.tile_m, config.tile_n, tile_k,
          config.warp_m, config.warp_n, config.stages};
}

quactlize_ppu_config_v3 config_v3(
    quactlize_ppu_config_v1 const& config, int tactic_tile_k, int artifact_tile_k) {
  return {config.enable_cuda_kernel, config.name, config.tile_m, config.tile_n,
          tactic_tile_k, artifact_tile_k, config.warp_m, config.warp_n, config.stages};
}

int32_t list_valid_dense_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype, bool fully_quantized) {
  auto const& format = ppu_formats::for_qtype(qtype);
  int const tile_k = fully_quantized ? format.fully_quantized_tile_k : format.scale_first_tile_k;
  int32_t count = 0;
  for (int i = 0; i < int(DenseConfigId::Count); ++i) {
    auto const id = static_cast<DenseConfigId>(i);
    bool const valid = fully_quantized
        ? dense_fully_quantized_config_valid(id, m, n, k, group_size, qtype)
        : dense_lowbit_config_valid(id, m, n, k, group_size, qtype);
    if (!valid) continue;
    if (configs && count < capacity) configs[count] = config_v2(kDenseConfigs[i], tile_k);
    ++count;
  }
  if (!fully_quantized && quactlize_ppu_gemv_lowbit_config_valid_v1(
          m, n, k, group_size, qtype, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME)) {
    if (configs && count < capacity) {
      configs[count] = {true, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME, 0, 0, 0, 0, 0, 0};
    }
    ++count;
  }
  return count;
}

int32_t list_valid_dense_fully_quantized_configs_for_arrangement_v1(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement) {
  if (!ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, ppu_formats::for_qtype(qtype).fully_quantized_tile_k)) return 0;
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  int32_t count = 0;
  for (int i = 0; i < int(DenseConfigId::Count); ++i) {
    auto const id = static_cast<DenseConfigId>(i);
    if (!dense_fully_quantized_config_valid(
            id, m, n, k, group_size, qtype, arrangement))
      continue;
    if (configs && count < capacity) {
      configs[count] = config_v3(kDenseConfigs[i], tactic_tile_k, arrangement->artifact_tile_k);
    }
    ++count;
  }
  return count;
}

int32_t list_valid_dense_fully_quantized_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (!ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return 0;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
    int32_t count = 0;
    for (int i = 0; i < int(Kpack4ConfigId::Count); ++i) {
      auto const id = static_cast<Kpack4ConfigId>(i);
      if (!q4_kpack4_config_valid(
              id, m, n, k, group_size, qtype, arrangement)) {
        continue;
      }
      auto const& row = ppu_q4_kpack4_shipping::row(id);
      if (configs && count < capacity) {
        configs[count] = {false, row.name, row.tile_m, row.tile_n,
                          row.tile_k, 0, row.warp_m, row.warp_n,
                          row.stages};
      }
      ++count;
    }
    return count;
  }
  int32_t count = 0;
  for (int i = 0; i < int(DenseConfigId::Count); ++i) {
    auto const id = static_cast<DenseConfigId>(i);
    if (!dense_fully_quantized_config_valid(
            id, m, n, k, group_size, qtype, arrangement)) {
      continue;
    }
    if (configs && count < capacity) {
      configs[count] = config_v3(
          kDenseConfigs[i], tactic_tile_k,
          arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1
              ? arrangement->artifact_tile_k : 0);
    }
    ++count;
  }
  return count;
}

int32_t list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
    quactlize_ppu_config_v4* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (!ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return 0;
  }
  int32_t count = 0;
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
    for (int i = 0; i < int(Kpack4ConfigId::Count); ++i) {
      auto const id = static_cast<Kpack4ConfigId>(i);
      if (!q4_kpack4_config_valid(
              id, m, n, k, group_size, qtype, arrangement)) {
        continue;
      }
      auto const& row = ppu_q4_kpack4_shipping::row(id);
      if (configs && count < capacity) {
        configs[count] = {false, row.name, row.tile_m, row.tile_n,
                          row.tile_k, 0, row.warp_m, row.warp_n,
                          row.stages, row.split};
      }
      ++count;
    }
    return count;
  }
  for (int i = 0; i < int(DenseConfigId::Count); ++i) {
    auto const id = static_cast<DenseConfigId>(i);
    if (!dense_fully_quantized_config_valid(
            id, m, n, k, group_size, qtype, arrangement)) {
      continue;
    }
    auto const& row = kDenseConfigs[i];
    if (configs && count < capacity) {
      configs[count] = {row.enable_cuda_kernel, row.name,
                        row.tile_m, row.tile_n, tactic_tile_k,
                        arrangement->artifact_tile_k, row.warp_m,
                        row.warp_n, row.stages, 1};
    }
    ++count;
  }
  return count;
}

int32_t list_valid_grouped_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype) {
  int const tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  int32_t count = 0;
  for (int i = 0; i < int(GroupedConfigId::Count); ++i) {
    auto const id = static_cast<GroupedConfigId>(i);
    if (!grouped_fully_quantized_config_valid(
            id, total_rows, n, k, group_size, experts, max_rows, qtype))
      continue;
    if (configs && count < capacity) configs[count] = config_v2(kGroupedConfigs[i], tile_k);
    ++count;
  }
  return count;
}

int32_t list_valid_grouped_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int const tactic_tile_k =
      ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  int32_t count = 0;
  for (int i = 0; i < int(GroupedConfigId::Count); ++i) {
    auto const id = static_cast<GroupedConfigId>(i);
    if (!grouped_fully_quantized_config_valid(
            id, total_rows, n, k, group_size, experts, max_rows, qtype,
            arrangement)) {
      continue;
    }
    auto const& row = kGroupedConfigs[i];
    if (configs && count < capacity) {
      configs[count] = {false, row.name, row.tile_m, row.tile_n,
                        tactic_tile_k,
                        arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1
                            ? arrangement->artifact_tile_k : 0,
                        row.warp_m, row.warp_n, row.stages};
    }
    ++count;
  }
  return count;
}

template <class Low, class High = void, int GroupSize = 16,
          int TacticTileK = 256, int ArtifactTileK = TacticTileK>
int dense_fully_quantized_device(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                                 uint8_t const* units, uint16_t* out, int m, int n, int k,
                                 DenseConfigId config, void* workspace, size_t workspace_bytes,
                                 hggcStream_t stream) {
  size_t const need = dense_workspace_bytes(m, n);
  if (!workspace || workspace_bytes < need) return 37;
  int const launch_rc = launch_dense_config<Low, High, GroupSize, TacticTileK, ArtifactTileK, true>(
      config, act, low, high, units, nullptr, out, m, n, k, workspace, workspace_bytes, stream);
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch("fully-quantized dense GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
template <class Low, class High, int GroupSize, int TacticTileK>
int dense_kpack_device(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, uint16_t* out, int m, int n, int k,
    DenseConfigId config, void* workspace, size_t workspace_bytes,
    hggcStream_t stream) {
  size_t const need = dense_workspace_bytes(m, n);
  if (!workspace || workspace_bytes < need) return 37;
  int const launch_rc = launch_dense_kpack_config<
      Low, High, GroupSize, TacticTileK>(
          config, act, low, high, units, out, m, n, k, workspace,
          workspace_bytes, stream);
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch(
             "fully-quantized dense K-pack GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

template <class Low, class High, int GroupSize, int TacticTileK>
int dense_kpack(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, uint16_t* out, int m, int n, int k,
    DenseConfigId config) {
  ppu_gemv::rt_clear_error();
  constexpr int LowBits = cutlass::sizeof_bits<Low>::value;
  constexpr int HighBits = std::is_void_v<High>
      ? 0 : cutlass::sizeof_bits<High>::value;
  size_t const low_bytes = size_t(n) * k * LowBits / 8;
  size_t const high_bytes = size_t(n) * k * HighBits / 8;
  size_t const unit_bytes =
      size_t(k / (256 * SelectedPackedUnit::kSbPerUnit)) * n *
      SelectedPackedUnit::kUnitTotal;
  DevBuf da(size_t(m) * k * 2), dl(low_bytes), dh(high_bytes),
      ds(unit_bytes), dout(size_t(m) * n * 2);
  size_t const ws_bytes = dense_workspace_bytes(m, n);
  DevBuf ws(ws_bytes);
  da.from_host(act);
  dl.from_host(low);
  ds.from_host(units);
  if constexpr (HighBits != 0) {
    if (!high) return 33;
    dh.from_host(high);
  }
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  int const launch_rc = dense_kpack_device<
      Low, High, GroupSize, TacticTileK>(
          reinterpret_cast<uint16_t const*>(da.p),
          reinterpret_cast<uint8_t const*>(dl.p),
          reinterpret_cast<uint8_t const*>(dh.p),
          reinterpret_cast<uint8_t const*>(ds.p),
          reinterpret_cast<uint16_t*>(dout.p), m, n, k, config,
          ws.p, ws_bytes, nullptr);
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync("fully-quantized dense K-pack GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(m) * n);
}
#endif

#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
int dense_q4_kpack4_device(
    uint16_t const* act, uint8_t const* low, uint8_t const* units,
    uint16_t* out, int m, int n, int k, Kpack4ConfigId config,
    void* workspace, size_t workspace_bytes, hggcStream_t stream) {
  dense_splitk_parallel_ppu::WorkspacePlan partial_plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(m, n, 4, partial_plan)) return 37;
  size_t const need = std::max(dense_workspace_bytes(m, n), partial_plan.partial_bytes);
  if (!workspace || workspace_bytes < need) return 37;
  int const launch_rc = launch_dense_q4_kpack4_config(
      config, act, low, units, out, m, n, k,
      workspace, workspace_bytes, stream);
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch(
             "fully-quantized dense Q4 K-pack4 GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}
#endif

template <class Low, class High = void, int GroupSize = 16,
          int TacticTileK = 256, bool PackedScale = false, int ArtifactTileK = TacticTileK>
int dense(uint16_t const* act, uint8_t const* low, uint8_t const* high,
          void const* scale, uint16_t const* zero, uint16_t* out,
          int m, int n, int k, int group_size, DenseConfigId config) {
  ppu_gemv::rt_clear_error();
  constexpr int LowBits = cutlass::sizeof_bits<Low>::value;
  constexpr int HighBits = std::is_void_v<High> ? 0 : cutlass::sizeof_bits<High>::value;
  size_t const low_bytes = size_t(n) * k * LowBits / 8;
  size_t const high_bytes = size_t(n) * k * HighBits / 8;
  size_t const plane_elems = size_t(k / GroupSize) * n;
  // A packed unit is byte-neutral metadata for one or two superblocks of a column, selected at build time. It is not
  // a half tensor: the shared mainloop reinterprets ptr_S as raw bytes when packed and does not read ptr_Z.
  // Keep the allocation distinction explicit so this entry cannot grow a second decoder beside the collective.
  size_t const scale_bytes = PackedScale
      ? size_t(k / (256 * SelectedPackedUnit::kSbPerUnit)) * n * SelectedPackedUnit::kUnitTotal
      : plane_elems * 2;
  DevBuf da(size_t(m) * k * 2), dl(low_bytes), dh(high_bytes), ds(scale_bytes),
         dz(PackedScale ? 0 : plane_elems * 2),
         dout(size_t(m) * n * 2);
  // SplitKSerial's semaphore workspace is one int per output CTA. The shared query covers every compiled config.
  size_t const ws_bytes = dense_workspace_bytes(m, n);
  DevBuf ws(ws_bytes);
  da.from_host(act); dl.from_host(low); ds.from_host(scale);
  if constexpr (!PackedScale) dz.from_host(zero);
  if constexpr (HighBits != 0) dh.from_host(high);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;

  int launch_rc = 0;
  if constexpr (PackedScale) {
    launch_rc = dense_fully_quantized_device<Low, High, GroupSize, TacticTileK, ArtifactTileK>(
        reinterpret_cast<uint16_t const*>(da.p), reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p), reinterpret_cast<uint8_t const*>(ds.p),
        reinterpret_cast<uint16_t*>(dout.p), m, n, k, config, ws.p, ws_bytes, nullptr);
  } else {
    launch_rc = launch_dense_config<Low, High, GroupSize, TacticTileK, ArtifactTileK, false>(
        config, reinterpret_cast<uint16_t const*>(da.p), reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p), ds.p, reinterpret_cast<uint16_t const*>(dz.p),
        reinterpret_cast<uint16_t*>(dout.p), m, n, k, ws.p, ws_bytes, nullptr);
  }
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync(PackedScale ? "fully-quantized dense GEMM" : "scale-first dense GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(m) * n);
}

int dense_q4_kpack4(
    uint16_t const* act, uint8_t const* low, uint8_t const* units,
    uint16_t* out, int m, int n, int k, Kpack4ConfigId config) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  ppu_gemv::rt_clear_error();
  size_t const low_bytes = size_t(n) * k / 2;
  size_t const unit_bytes = size_t(k / 256) * n * Q4PackedUnit::kUnitBytes;
  DevBuf da(size_t(m) * k * 2), dl(low_bytes), ds(unit_bytes),
      dout(size_t(m) * n * 2);
  dense_splitk_parallel_ppu::WorkspacePlan partial_plan;
  if (!dense_splitk_parallel_ppu::query_workspace_plan(m, n, 4, partial_plan)) return 37;
  size_t const ws_bytes = std::max(dense_workspace_bytes(m, n), partial_plan.partial_bytes);
  DevBuf ws(ws_bytes);
  da.from_host(act);
  dl.from_host(low);
  ds.from_host(units);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  int const launch_rc = dense_q4_kpack4_device(
      reinterpret_cast<uint16_t const*>(da.p),
      reinterpret_cast<uint8_t const*>(dl.p),
      reinterpret_cast<uint8_t const*>(ds.p),
      reinterpret_cast<uint16_t*>(dout.p), m, n, k, config,
      ws.p, ws_bytes, nullptr);
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync("fully-quantized dense Q4 K-pack4 GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(m) * n);
#else
  (void)act; (void)low; (void)units; (void)out;
  (void)m; (void)n; (void)k; (void)config;
  return 34;
#endif
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK>
int dense_fully_quantized_for_artifact(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, DenseConfigId config) {
  if constexpr (!ppu_arrangements::static_packed_tensor_reader_supported<
                    QType, TacticTileK, ArtifactTileK>()) {
    return 38;
  } else {
    return dense<Low, High, GroupSize, TacticTileK, true, ArtifactTileK>(
        act, low, high, units, nullptr, out, m, n, k, GroupSize, config);
  }
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK>
int dense_fully_quantized_for_arrangement(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, DenseConfigId config, int artifact_tile_k) {
  switch (artifact_tile_k) {
    case 32: return dense_fully_quantized_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 32>(act, low, high, units, out, m, n, k, config);
    case 64: return dense_fully_quantized_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 64>(act, low, high, units, out, m, n, k, config);
    case 128: return dense_fully_quantized_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 128>(act, low, high, units, out, m, n, k, config);
    case 256: return dense_fully_quantized_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 256>(act, low, high, units, out, m, n, k, config);
    default: return 38;
  }
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK, int ArtifactTileK>
int dense_fully_quantized_device_for_artifact(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, DenseConfigId config, void* workspace, size_t workspace_bytes,
    hggcStream_t stream) {
  if constexpr (!ppu_arrangements::static_packed_tensor_reader_supported<
                    QType, TacticTileK, ArtifactTileK>()) {
    return 38;
  } else {
    return dense_fully_quantized_device<Low, High, GroupSize, TacticTileK, ArtifactTileK>(
        act, low, high, units, out, m, n, k, config, workspace, workspace_bytes, stream);
  }
}

template <int QType, class Low, class High, int GroupSize, int TacticTileK>
int dense_fully_quantized_device_for_arrangement(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, DenseConfigId config, void* workspace, size_t workspace_bytes,
    hggcStream_t stream, int artifact_tile_k) {
  switch (artifact_tile_k) {
    case 32: return dense_fully_quantized_device_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 32>(
            act, low, high, units, out, m, n, k, config, workspace, workspace_bytes, stream);
    case 64: return dense_fully_quantized_device_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 64>(
            act, low, high, units, out, m, n, k, config, workspace, workspace_bytes, stream);
    case 128: return dense_fully_quantized_device_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 128>(
            act, low, high, units, out, m, n, k, config, workspace, workspace_bytes, stream);
    case 256: return dense_fully_quantized_device_for_artifact<
        QType, Low, High, GroupSize, TacticTileK, 256>(
            act, low, high, units, out, m, n, k, config, workspace, workspace_bytes, stream);
    default: return 38;
  }
}

}  // namespace

extern "C" int32_t quactlize_ppu_list_configs(quactlize_ppu_config_v1 const** configs) {
  if (configs) *configs = kDenseConfigs;
  return int32_t(sizeof(kDenseConfigs) / sizeof(kDenseConfigs[0]));
}

extern "C" int32_t quactlize_ppu_list_grouped_configs(quactlize_ppu_config_v1 const** configs) {
  if (configs) *configs = kGroupedConfigs;
  return int32_t(sizeof(kGroupedConfigs) / sizeof(kGroupedConfigs[0]));
}

extern "C" int32_t quactlize_ppu_dense_lowbit_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  DenseConfigId config{};
  return find_dense_config(config_name, m, config) &&
         dense_lowbit_config_valid(config, m, n, k, group_size, qtype);
}

extern "C" int32_t quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  bool const name_ok = !config_name || !config_name[0] ||
      std::strcmp(config_name,
                  ppu_q4_kpack4_shipping::kScaleFirstPersistentName) == 0;
  if (!name_ok || m < 64 || group_size != 32 || qtype != 12 ||
      !ppu_arrangements::matches_compiled_tactic(
          arrangement, qtype, k, 64) ||
      arrangement->layout != QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
    return 0;
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    (!defined(PPU_PACKED_SCALE) || (PPU_PACKED_SCALE == 0))
  return launch_scalefirst_q4_kpack4_persistent<true>(
             nullptr, nullptr, nullptr, nullptr, nullptr,
             m, n, k, nullptr) == 0;
#else
  return 0;
#endif
}

extern "C" int32_t quactlize_ppu_dense_fully_quantized_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  DenseConfigId config{};
  return find_dense_config(config_name, m, config) &&
         dense_fully_quantized_config_valid(config, m, n, k, group_size, qtype);
}

extern "C" int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v1(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, char const* config_name) {
  DenseConfigId config{};
  return find_dense_config(config_name, m, config) &&
         dense_fully_quantized_config_valid(
             config, m, n, k, group_size, qtype, arrangement);
}

extern "C" int32_t quactlize_ppu_dense_fully_quantized_config_valid_for_arrangement_v2(
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  if (arrangement &&
      arrangement->layout == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
    Kpack4ConfigId config{};
    return ppu_q4_kpack4_shipping::find_config(
               config_name, m, n, k, config) &&
           q4_kpack4_config_valid(
               config, m, n, k, group_size, qtype, arrangement);
  }
  DenseConfigId config{};
  return find_dense_config(config_name, m, config) &&
         dense_fully_quantized_config_valid(
             config, m, n, k, group_size, qtype, arrangement);
}

extern "C" int32_t quactlize_ppu_grouped_lowbit_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name) {
  GroupedConfigId config{};
  return find_grouped_tensor_config(config_name, config) &&
         grouped_lowbit_config_valid(
             config, total_rows, n, k, group_size, experts, max_rows, qtype);
}

extern "C" int32_t quactlize_ppu_grouped_fully_quantized_config_valid_v1(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, char const* config_name) {
  GroupedConfigId config{};
  return find_grouped_tensor_config(config_name, config) &&
         grouped_fully_quantized_config_valid(
             config, total_rows, n, k, group_size, experts, max_rows, qtype);
}

extern "C" int32_t quactlize_ppu_grouped_fully_quantized_config_valid_for_arrangement_v2(
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  GroupedConfigId config{};
  return find_grouped_tensor_config(config_name, config) &&
         grouped_fully_quantized_config_valid(
             config, total_rows, n, k, group_size, experts, max_rows, qtype,
             arrangement);
}

extern "C" int32_t quactlize_ppu_list_valid_dense_lowbit_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype) {
  return list_valid_dense_configs_v2(
      configs, capacity, m, n, k, group_size, qtype, false);
}

extern "C" int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype) {
  return list_valid_dense_configs_v2(
      configs, capacity, m, n, k, group_size, qtype, true);
}

extern "C" int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v1(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement) {
  return list_valid_dense_fully_quantized_configs_for_arrangement_v1(
      configs, capacity, m, n, k, group_size, qtype, arrangement);
}

extern "C" int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  return list_valid_dense_fully_quantized_configs_for_arrangement_v2(
      configs, capacity, m, n, k, group_size, qtype, arrangement);
}

extern "C" int32_t quactlize_ppu_list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
    quactlize_ppu_config_v4* configs, int32_t capacity,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  return list_valid_dense_fully_quantized_configs_for_arrangement_v2_v4(
      configs, capacity, m, n, k, group_size, qtype, arrangement);
}

extern "C" int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype) {
  return list_valid_grouped_configs_v2(
      configs, capacity, total_rows, n, k, group_size, experts, max_rows, qtype);
}

extern "C" int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_for_arrangement_v2(
    quactlize_ppu_config_v3* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows,
    int qtype, quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  return list_valid_grouped_configs_for_arrangement_v2(
      configs, capacity, total_rows, n, k, group_size, experts, max_rows,
      qtype, arrangement);
}

extern "C" int64_t quactlize_ppu_dense_w4_splitk_workspace_bytes_v1(
    int m, int n, int k,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED
  return ppu_dense_w4_splitk::query_workspace_bytes(m, n, k, profile);
#else
  (void)m; (void)n; (void)k; (void)profile;
  return -1;
#endif
}

extern "C" int quactlize_ppu_dense_w4_splitk_dev_v1(
    uint16_t const* act, uint8_t const* weight_xplane,
    uint16_t const* scales, uint16_t* out,
    int m, int n, int k, void* workspace, int64_t workspace_bytes,
    void* stream,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED
  if (!act || !weight_xplane || !scales || !out ||
      !ppu_dense_w4_splitk::problem_is_in_fixed_abi(m, n, k)) {
    return 30;
  }
  std::size_t const usable_workspace_bytes = workspace_bytes > 0
      ? static_cast<std::size_t>(workspace_bytes) : 0;
  auto const selected = ppu_dense_w4_splitk::select_profile(
      m, n, k, reinterpret_cast<std::uintptr_t>(workspace),
      usable_workspace_bytes, profile);

  ppu_gemv::rt_clear_error();
  hggcStream_t const launch_stream = static_cast<hggcStream_t>(stream);
  ppu_dense_w4_splitk::Prepared prepared;
  if (!ppu_dense_w4_splitk::prepare_selected<>(
          selected, prepared,
          reinterpret_cast<half_t const*>(act),
          reinterpret_cast<cutlass::int4b_t const*>(weight_xplane),
          reinterpret_cast<half_t const*>(scales),
          reinterpret_cast<half_t*>(out), m, n, k,
          static_cast<char*>(workspace), usable_workspace_bytes,
          launch_stream)) {
    return 31;
  }
  if (prepared.run(launch_stream) != cutlass::Status::kSuccess) return 31;
  return ppu_gemv::rt_check_launch("dense W4 fixed Split-K enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
#else
  (void)act; (void)weight_xplane; (void)scales; (void)out;
  (void)m; (void)n; (void)k; (void)workspace; (void)workspace_bytes;
  (void)stream; (void)profile;
  return 33;
#endif
}

extern "C" int quactlize_ppu_dense_lowbit_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, void* stream,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  if (!act || !low || high || !scale || !zero || !out ||
      quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2(
          m, n, k, group_size, qtype, arrangement, config_name) != 1) {
    return 38;
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    (!defined(PPU_PACKED_SCALE) || (PPU_PACKED_SCALE == 0))
  ppu_gemv::rt_clear_error();
  int const rc = launch_scalefirst_q4_kpack4_persistent(
      act, low, scale, zero, out, m, n, k,
      static_cast<hggcStream_t>(stream));
  if (rc) return rc;
  return ppu_gemv::rt_check_launch(
             "ScaleFirst dense Q4 K-pack4 persistent enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
#else
  (void)stream;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  if (!act || !low || !scale || !zero || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  DenseConfigId const config = resolve_dense_config(config_name, m);
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    // The selected Q2 unit has 16 groups, so TileK=256 would reinterpret this ABI's fp16 scale plane as raw units.
    // Its single unfolded code plane is tile-invariant; TileK=128 keeps the established scale-first contract live.
    case 10: return group_size == 16 ? dense<cutlass::uint2b_t,void,16,
        ppu_formats::for_qtype(10).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#else
    case 10: return group_size == 16 ? dense<cutlass::uint2b_t,void,16,
        ppu_formats::for_qtype(10).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    // Q3's fixed TK256 high-plane descriptor is also the packed type in a format-3 library.
    case 11: return 36;
#else
    case 11: return group_size == 16 ? dense<cutlass::uint2b_t,cutlass::uint1b_t,16,
        ppu_formats::for_qtype(11).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
    // The fully-quantized TileK selects packed units in this build, so the SCALE_FIRST contract must use its own
    // registry value. Its single low plane is tile-invariant; the smaller scale-first run keeps kPackedScaleOn false
    // and lets one flagged library run the independent scale-first oracle beside the packed entry below.
    case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32,
        ppu_formats::for_qtype(12).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#else
    case 12: return group_size == 32 ? dense<cutlass::int4b_t,void,32,
        ppu_formats::for_qtype(12).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    // Q5's high-plane placement is tied to this TileK=256 tactic, while this exact type selects raw packed metadata
    // in a format-1 build. Refuse the fp16 scale-first contract explicitly instead of reinterpreting its scale plane
    // as units; the default build and every differently selected packed-format build retain the established path.
    case 13: return 36;
#else
    case 13: return group_size == 32 ? dense<cutlass::int4b_t,cutlass::uint1b_t,32,
        ppu_formats::for_qtype(13).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    // Q6 must retain TK128; in a format-4 library that exact type consumes paired raw units, not fp16 planes.
    case 14: return 36;
#else
    case 14: return group_size == 16 ? dense<cutlass::int4b_t,cutlass::uint2b_t,16,
        ppu_formats::for_qtype(14).scale_first_tile_k>(act,low,high,scale,zero,out,m,n,k,group_size,config) : 32;
#endif
#endif
    default: return 33;
  }
}

extern "C" int quactlize_ppu_dense_lowbit(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                                            uint16_t const* scale, uint16_t const* zero, uint16_t* out,
                                            int m, int n, int k, int group_size, int qtype) {
  return quactlize_ppu_dense_lowbit_config_v1(
      act, low, high, scale, zero, out, m, n, k, group_size, qtype, nullptr);
}

extern "C" int quactlize_ppu_dense_lowbit_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  if (!act || !low || high || !scale || !zero || !out ||
      quactlize_ppu_dense_lowbit_config_valid_for_arrangement_v2(
          m, n, k, group_size, qtype, arrangement, config_name) != 1) {
    return 38;
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    (!defined(PPU_PACKED_SCALE) || (PPU_PACKED_SCALE == 0))
  ppu_gemv::rt_clear_error();
  size_t const plane_bytes = size_t(n) * size_t(k / group_size) * 2;
  DevBuf da(size_t(m) * k * 2), dl(size_t(n) * k / 2),
      ds(plane_bytes), dz(plane_bytes), dout(size_t(m) * n * 2);
  da.from_host(act); dl.from_host(low); ds.from_host(scale); dz.from_host(zero);
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  int const rc = launch_scalefirst_q4_kpack4_persistent(
      reinterpret_cast<uint16_t const*>(da.p),
      reinterpret_cast<uint8_t const*>(dl.p),
      reinterpret_cast<uint16_t const*>(ds.p),
      reinterpret_cast<uint16_t const*>(dz.p),
      reinterpret_cast<uint16_t*>(dout.p), m, n, k, nullptr);
  if (rc) return rc;
  ppu_gemv::rt_sync("ScaleFirst dense Q4 K-pack4 persistent GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(m) * n);
#else
  (void)config_name;
  return 34;
#endif
}

// FULLY_QUANTIZED x DENSE, format-selected k-quants. This is only a second ABI contract: it instantiates the SAME
// dense() wrapper and CollectiveBuilder as scale-first. Q2/Q3/Q4/Q5 use tactic TileK=256; Q6 keeps tactic TileK=128.
// The legacy resident-byte arrangement remains fully_quantized_tile_k, exactly as before the versioned descriptor;
// new Python artifacts use the arrangement-aware entry even at their no-tile scale-first default. The two-plane
// collective's scale channel stages paired units without silently replacing an explicit artifact identity.
// Builds without PPU_PACKED_SCALE retain the symbol but return 34.
extern "C" int quactlize_ppu_dense_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, char const* config_name) {
  if (!act || !low || !units || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  DenseConfigId const config = resolve_dense_config(config_name, m);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  return dense<cutlass::int4b_t, void, 32, ppu_formats::for_qtype(12).fully_quantized_tile_k, true,
               ppu_formats::for_qtype(12).fully_quantized_tile_k>(
      act, low, nullptr, units, nullptr, out, m, n, k, 32, config);
#elif PPU_PACKED_FORMAT == 2
  if (qtype != 10) return 33;
  return dense<cutlass::uint2b_t, void, 16, ppu_formats::for_qtype(10).fully_quantized_tile_k, true,
               ppu_formats::for_qtype(10).fully_quantized_tile_k>(
      act, low, nullptr, units, nullptr, out, m, n, k, 16, config);
#elif PPU_PACKED_FORMAT == 1
  if (qtype != 13 || !high) return 33;
  return dense<cutlass::int4b_t, cutlass::uint1b_t, 32,
               ppu_formats::for_qtype(13).fully_quantized_tile_k, true,
               ppu_formats::for_qtype(13).fully_quantized_tile_k>(
      act, low, high, units, nullptr, out, m, n, k, 32, config);
#elif PPU_PACKED_FORMAT == 3
  if (qtype != 11 || !high || k % 512) return 33;
  return dense<cutlass::uint2b_t, cutlass::uint1b_t, 16,
               ppu_formats::for_qtype(11).fully_quantized_tile_k, true,
               ppu_formats::for_qtype(11).fully_quantized_tile_k>(
      act, low, high, units, nullptr, out, m, n, k, 16, config);
#elif PPU_PACKED_FORMAT == 4
  if (qtype != 14 || !high || k % 512) return 33;
  return dense<cutlass::int4b_t, cutlass::uint2b_t, 16,
               ppu_formats::for_qtype(14).fully_quantized_tile_k, true,
               ppu_formats::for_qtype(14).fully_quantized_tile_k>(
      act, low, high, units, nullptr, out, m, n, k, 16, config);
#else
  (void)qtype;
  return 35;  // this binary's packed-scale format has no entry here yet
#endif
#else
  (void)config;
  (void)qtype;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_fully_quantized(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype) {
  return quactlize_ppu_dense_fully_quantized_config_v1(
      act, low, high, units, out, m, n, k, qtype, nullptr);
}

extern "C" int quactlize_ppu_dense_fully_quantized_for_arrangement_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v1 const* arrangement, char const* config_name) {
  if (!act || !low || !units || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  if (!selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, ppu_formats::for_qtype(qtype).fully_quantized_tile_k)) return 38;
  DenseConfigId config{};
  if (!find_dense_config(config_name, m, config)) return 39;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  return dense_fully_quantized_for_arrangement<12, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
          act, low, nullptr, units, out, m, n, k, config, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 2
  if (qtype != 10) return 33;
  return dense_fully_quantized_for_arrangement<10, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
          act, low, nullptr, units, out, m, n, k, config, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 1
  if (qtype != 13 || !high) return 33;
  return dense_fully_quantized_for_arrangement<13, cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 3
  if (qtype != 11 || !high || k % 512) return 33;
  return dense_fully_quantized_for_arrangement<11, cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 4
  if (qtype != 14 || !high || k % 512) return 33;
  return dense_fully_quantized_for_arrangement<14, cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, arrangement->artifact_tile_k);
#else
  return 35;
#endif
#else
  (void)high; (void)config;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_fully_quantized_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, uint16_t* out, int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  if (!act || !low || !units || !out || m <= 0 || n <= 0 || k <= 0 ||
      n % 256 || k % 256) {
    return 30;
  }
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (!selected_fully_quantized_qtype(qtype, k) ||
      !ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return 38;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    quactlize_ppu_placed_arrangement_v1 const legacy{
        QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
        arrangement->bits, arrangement->artifact_tile_k,
        arrangement->high_bits};
    return quactlize_ppu_dense_fully_quantized_for_arrangement_v1(
        act, low, high, units, out, m, n, k, qtype,
        &legacy, config_name);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
    DenseConfigId config{};
    if (!find_dense_config(config_name, m, config)) return 39;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
#if PPU_PACKED_FORMAT == 2
    if (qtype != 10 || high) return 33;
    return dense_kpack<cutlass::uint2b_t, void, 16, 256>(
        act, low, nullptr, units, out, m, n, k, config);
#elif PPU_PACKED_FORMAT == 3
    if (qtype != 11 || !high) return 33;
    return dense_kpack<cutlass::uint2b_t, cutlass::uint1b_t, 16, 256>(
        act, low, high, units, out, m, n, k, config);
#elif PPU_PACKED_FORMAT == 1
    if (qtype != 13 || !high) return 33;
    return dense_kpack<cutlass::int4b_t, cutlass::uint1b_t, 32, 256>(
        act, low, high, units, out, m, n, k, config);
#elif PPU_PACKED_FORMAT == 4
    if (qtype != 14 || !high) return 33;
    return dense_kpack<cutlass::int4b_t, cutlass::uint2b_t, 16, 128>(
        act, low, high, units, out, m, n, k, config);
#else
    return 35;
#endif
#else
    (void)high; (void)config;
    return 34;
#endif
  }
  Kpack4ConfigId config{};
  if (!ppu_q4_kpack4_shipping::find_config(
          config_name, m, n, k, config)) return 39;
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  if (qtype != 12 || high) return 33;
  return dense_q4_kpack4(
      act, low, units, out, m, n, k, config);
#else
  (void)high; (void)config;
  return 34;
#endif
}

extern "C" int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(
    int m, int n, int k, int qtype) {
  if (m <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 || !selected_fully_quantized_qtype(qtype, k)) return -1;
  return int64_t(dense_workspace_bytes(m, n));
}

extern "C" int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2(
    int m, int n, int k, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (m <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 ||
      !selected_fully_quantized_qtype(qtype, k)) {
    return -1;
  }
  if (arrangement &&
      arrangement->layout == QUACTLIZE_PPU_LAYOUT_Q4_KPACK4_TRANSPOSE_V1) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED
    Kpack4ConfigId config{};
    if (!ppu_q4_kpack4_shipping::find_config(nullptr, m, n, k, config) ||
        !q4_kpack4_config_valid(
            config, m, n, k, qtype_group_size(qtype), qtype, arrangement)) {
      return -1;
    }
    dense_splitk_parallel_ppu::WorkspacePlan partial_plan;
    if (!dense_splitk_parallel_ppu::query_workspace_plan(
            m, n, 4, partial_plan)) {
      return -1;
    }
    return int64_t(std::max(
        dense_workspace_bytes(m, n), partial_plan.partial_bytes));
#else
    return -1;
#endif
  }
  DenseConfigId config{};
  if (!find_dense_config(nullptr, m, config) ||
      !dense_fully_quantized_config_valid(
          config, m, n, k, qtype_group_size(qtype), qtype,
          arrangement)) {
    return -1;
  }
  return int64_t(dense_workspace_bytes(m, n));
}

extern "C" int quactlize_ppu_dense_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name) {
  int64_t const need = quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(m, n, k, qtype);
  if (!act || !low || !units || !out || !workspace || need < 0 || workspace_bytes < need) return 30;
  ppu_gemv::rt_clear_error();
  hggcStream_t const s = static_cast<hggcStream_t>(stream);
  DenseConfigId const config = resolve_dense_config(config_name, m);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return dense_fully_quantized_device<cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
      act, low, nullptr, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 2
  return dense_fully_quantized_device<cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
      act, low, nullptr, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 1
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
      act, low, high, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 3
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
      act, low, high, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 4
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
      act, low, high, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#else
  return 35;
#endif
#else
  (void)high; (void)stream; (void)config;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name, quactlize_ppu_placed_arrangement_v1 const* arrangement) {
  int64_t const need = quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(m, n, k, qtype);
  if (!act || !low || !units || !out || !workspace || need < 0 || workspace_bytes < need) return 30;
  if (!ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, ppu_formats::for_qtype(qtype).fully_quantized_tile_k)) return 38;
  DenseConfigId config{};
  if (!find_dense_config(config_name, m, config)) return 39;
  ppu_gemv::rt_clear_error();
  hggcStream_t const s = static_cast<hggcStream_t>(stream);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return dense_fully_quantized_device_for_arrangement<12, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
          act, low, nullptr, units, out, m, n, k, config, workspace, size_t(workspace_bytes), s,
          arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 2
  return dense_fully_quantized_device_for_arrangement<10, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
          act, low, nullptr, units, out, m, n, k, config, workspace, size_t(workspace_bytes), s,
          arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 1
  if (!high) return 33;
  return dense_fully_quantized_device_for_arrangement<13, cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, workspace, size_t(workspace_bytes), s,
          arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 3
  if (!high) return 33;
  return dense_fully_quantized_device_for_arrangement<11, cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, workspace, size_t(workspace_bytes), s,
          arrangement->artifact_tile_k);
#elif PPU_PACKED_FORMAT == 4
  if (!high) return 33;
  return dense_fully_quantized_device_for_arrangement<14, cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
          act, low, high, units, out, m, n, k, config, workspace, size_t(workspace_bytes), s,
          arrangement->artifact_tile_k);
#else
  return 35;
#endif
#else
  (void)high; (void)stream; (void)config;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, uint16_t* out, int m, int n, int k, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  int64_t const need =
      quactlize_ppu_dense_fully_quantized_workspace_bytes_for_arrangement_v2(
          m, n, k, qtype, arrangement);
  if (!act || !low || !units || !out || !workspace || need < 0 ||
      workspace_bytes < need) {
    return 30;
  }
  int const tactic_tile_k = ppu_formats::for_qtype(qtype).fully_quantized_tile_k;
  if (!ppu_arrangements::packed_tensor_reader_supported(
          arrangement, qtype, k, tactic_tile_k)) {
    return 38;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    quactlize_ppu_placed_arrangement_v1 const legacy{
        QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
        arrangement->bits, arrangement->artifact_tile_k,
        arrangement->high_bits};
    return quactlize_ppu_dense_fully_quantized_dev_for_arrangement_v1(
        act, low, high, units, out, m, n, k, qtype,
        workspace, workspace_bytes, stream, config_name, &legacy);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
    DenseConfigId config{};
    if (!find_dense_config(config_name, m, config)) return 39;
    ppu_gemv::rt_clear_error();
    hggcStream_t const s = static_cast<hggcStream_t>(stream);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
#if PPU_PACKED_FORMAT == 2
    if (qtype != 10 || high) return 33;
    return dense_kpack_device<cutlass::uint2b_t, void, 16, 256>(
        act, low, nullptr, units, out, m, n, k, config,
        workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 3
    if (qtype != 11 || !high) return 33;
    return dense_kpack_device<
        cutlass::uint2b_t, cutlass::uint1b_t, 16, 256>(
            act, low, high, units, out, m, n, k, config,
            workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 1
    if (qtype != 13 || !high) return 33;
    return dense_kpack_device<
        cutlass::int4b_t, cutlass::uint1b_t, 32, 256>(
            act, low, high, units, out, m, n, k, config,
            workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 4
    if (qtype != 14 || !high) return 33;
    return dense_kpack_device<
        cutlass::int4b_t, cutlass::uint2b_t, 16, 128>(
            act, low, high, units, out, m, n, k, config,
            workspace, size_t(workspace_bytes), s);
#else
    return 35;
#endif
#else
    (void)high; (void)config; (void)s;
    return 34;
#endif
  }
  Kpack4ConfigId config{};
  if (!ppu_q4_kpack4_shipping::find_config(
          config_name, m, n, k, config)) return 39;
  ppu_gemv::rt_clear_error();
  hggcStream_t const s = static_cast<hggcStream_t>(stream);
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  if (qtype != 12 || high) return 33;
  return dense_q4_kpack4_device(
      act, low, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#else
  (void)high; (void)config; (void)s;
  return 34;
#endif
}

extern "C" int quactlize_ppu_dense_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, void* workspace, int64_t workspace_bytes, void* stream) {
  return quactlize_ppu_dense_fully_quantized_dev_v2(
      act, low, high, units, out, m, n, k, qtype, workspace, workspace_bytes, stream, nullptr);
}

namespace {

static __global__ void grouped_metadata(
    int const* offsets, half_t* out, GS* shapes, half_t** out_ptrs, DS* out_strides, int* rows,
    int n, int k, int experts) {
  int const e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= experts) return;
  int const begin = offsets[e];
  int const count = offsets[e + 1] - begin;
  rows[e] = count;
  shapes[e] = cute::make_shape(count, n, k);
  out_ptrs[e] = out + int64_t(begin) * n;
  out_strides[e] = cutlass::make_cute_packed_stride(DS{}, cute::make_shape(count, n, 1));
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize,
          int TileK, bool Q4Kpack4 = false, bool KQuantKpack = false>
int grouped_device(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, void const* scale,
    int const* offsets, uint16_t* out, int total_rows, int n, int k, int experts, int max_rows,
    GroupedConfigId config, void* workspace, size_t workspace_bytes, hggcStream_t stream) {
  GroupedWorkspaceLayout const layout = grouped_workspace_layout(max_rows, n, experts);
  if (!workspace || workspace_bytes < layout.total) return 37;
  auto* base = static_cast<uint8_t*>(workspace);
  auto* shapes = reinterpret_cast<GS*>(base + layout.shapes);
  auto* out_ptrs = reinterpret_cast<half_t**>(base + layout.out_ptrs);
  auto* out_strides = reinterpret_cast<DS*>(base + layout.out_strides);
  auto* rows = reinterpret_cast<int*>(base + layout.rows);
  auto* kernel_workspace = reinterpret_cast<char*>(base + layout.kernel);
  auto* out_half = reinterpret_cast<half_t*>(out);

  grouped_metadata<<<(experts + 127) / 128, 128, 0, stream>>>(
      offsets, out_half, shapes, out_ptrs, out_strides, rows, n, k, experts);
  if (!ppu_gemv::rt_check_launch(PackedScale ? "fully-quantized grouped metadata enqueue"
                                             : "scale-first grouped metadata enqueue"))
    return ppu_gemv::kRuntimeError;

  int launch_rc = 31;
  static_assert(!(Q4Kpack4 && KQuantKpack),
                "a grouped launch selects exactly one physical K-pack ABI");
  if constexpr (Q4Kpack4) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
    static_assert(QuantOp == GQM::FinegrainedScaleZero && PackedScale &&
                      std::is_same_v<Low, cutlass::int4b_t> &&
                      std::is_void_v<High> && GroupSize == 32 && TileK == 256,
                  "grouped K-pack4 must retain the canonical Q4_K packed-metadata contract");
    launch_rc = launch_grouped_q4_kpack4_config(
        config, act, low, reinterpret_cast<uint8_t const*>(scale), out_ptrs,
        out_strides, rows, max_rows, n, k, experts, shapes, nullptr, offsets,
        kernel_workspace, layout.kernel_bytes, stream);
#endif
  } else if constexpr (KQuantKpack) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
    static_assert(QuantOp == GQM::FinegrainedScaleZero && PackedScale,
                  "generic grouped K-pack consumes packed metadata");
    launch_rc = launch_grouped_kpack_config<Low, High, GroupSize, TileK>(
        config, act, low, high, reinterpret_cast<uint8_t const*>(scale),
        out_ptrs, out_strides, rows, max_rows, n, k, experts, shapes,
        nullptr, offsets, kernel_workspace, layout.kernel_bytes, stream);
#endif
  } else {
    launch_rc = launch_grouped_config<QuantOp, PackedScale,
        Low, High, GroupSize, TileK>(
        config, act, low, high, scale, nullptr, out_ptrs, out_strides, rows,
        max_rows, n, k, experts, shapes, nullptr, offsets, kernel_workspace,
        layout.kernel_bytes, stream);
  }
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch(PackedScale ? "fully-quantized grouped GEMM enqueue"
                                               : "scale-first grouped GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize,
          int TileK, bool Q4Kpack4 = false, bool KQuantKpack = false>
int grouped(uint16_t const* act, uint8_t const* low, uint8_t const* high, void const* scale,
            int const* rows_per_expert, uint16_t* out,
            int total_rows, int n, int k, int experts,
            GroupedConfigId config = kDefaultGroupedConfig) {
  ppu_gemv::rt_clear_error();
  using GS = moe_grouped_ppu::GroupShape;
  using DS = moe_grouped_ppu::DStride;
  constexpr int LowBits = cutlass::sizeof_bits<Low>::value;
  constexpr int HighBits = std::is_void_v<High> ? 0 : cutlass::sizeof_bits<High>::value;
  constexpr int ScaleGroups = ppu_group_schedule::scale_groups_v<TileK, GroupSize>;
  if constexpr (PackedScale) {
    static_assert(SelectedPackedUnit::kGroups % ScaleGroups == 0,
                  "the fixed grouped TileK must cover an integral group run of its packed superblock");
  }

  std::vector<int> rows(static_cast<size_t>(experts)), offsets(static_cast<size_t>(experts));
  int sum = 0, max_rows = 0;
  for (int e = 0; e < experts; ++e) {
    if (rows_per_expert[e] < 0) return 36;
    rows[size_t(e)] = rows_per_expert[e];
    offsets[size_t(e)] = sum;
    sum += rows_per_expert[e];
    max_rows = std::max(max_rows, rows_per_expert[e]);
  }
  if (sum != total_rows || max_rows <= 0) return 36;

  size_t const low_bytes = size_t(experts) * n * k * LowBits / 8;
  size_t const scale_bytes = PackedScale
      ? size_t(experts) * (k / (256 * SelectedPackedUnit::kSbPerUnit)) * n * SelectedPackedUnit::kUnitTotal
      : size_t(experts) * (k / GroupSize) * n * sizeof(half_t);
  size_t const high_bytes = size_t(experts) * n * k * HighBits / 8;
  DevBuf da(size_t(total_rows) * k * 2), dl(low_bytes), dh(high_bytes), ds(scale_bytes),
         dout(size_t(total_rows) * n * 2);
  da.from_host(act); dl.from_host(low); ds.from_host(scale);
  if constexpr (HighBits != 0) {
    if (!high) return 33;
    dh.from_host(high);
  }
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;

  std::vector<GS> shapes(static_cast<size_t>(experts));
  std::vector<half_t*> out_ptrs(static_cast<size_t>(experts));
  std::vector<DS> out_strides(static_cast<size_t>(experts));
  for (int e = 0; e < experts; ++e) {
    shapes[size_t(e)] = cute::make_shape(rows[size_t(e)], n, k);
    out_ptrs[size_t(e)] = dout.as<half_t>() + size_t(offsets[size_t(e)]) * n;
    out_strides[size_t(e)] = cutlass::make_cute_packed_stride(
        DS{}, cute::make_shape(rows[size_t(e)], n, 1));
  }
  DevBuf d_shapes(sizeof(GS) * size_t(experts)), d_out_ptrs(sizeof(half_t*) * size_t(experts)),
         d_out_strides(sizeof(DS) * size_t(experts)), d_rows(sizeof(int) * size_t(experts)),
         d_offsets(sizeof(int) * size_t(experts));
  d_shapes.from_host(shapes.data()); d_out_ptrs.from_host(out_ptrs.data());
  d_out_strides.from_host(out_strides.data()); d_rows.from_host(rows.data()); d_offsets.from_host(offsets.data());
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;

  size_t const ws_bytes = size_t(cutlass::ceil_div(max_rows, 16)) * cutlass::ceil_div(n, 64)
                        * size_t(experts) * 64;
  DevBuf ws(ws_bytes);
  // Call the fixed group-size instantiations directly. filter_and_run's runtime ladder instantiates several SK values
  // together, so asserting packed selection there would correctly fail on its non-selected control branches.
  int launch_rc = 31;
  static_assert(!(Q4Kpack4 && KQuantKpack),
                "a grouped host launch selects exactly one physical K-pack ABI");
  if constexpr (Q4Kpack4) {
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
    static_assert(QuantOp == GQM::FinegrainedScaleZero && PackedScale &&
                      std::is_same_v<Low, cutlass::int4b_t> &&
                      std::is_void_v<High> && GroupSize == 32 && TileK == 256,
                  "host grouped K-pack4 must match the device reader's exact Q4_K policy");
    launch_rc = launch_grouped_q4_kpack4_config(
        config, reinterpret_cast<uint16_t const*>(da.p),
        reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(ds.p), d_out_ptrs.as<half_t*>(),
        d_out_strides.as<DS>(), d_rows.as<int>(), max_rows, n, k, experts,
        d_shapes.as<GS>(), shapes.data(), d_offsets.as<int>(), ws.as<char>(),
        ws_bytes, nullptr);
#endif
  } else if constexpr (KQuantKpack) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT) && (PPU_PACKED_FORMAT != 0)
    static_assert(QuantOp == GQM::FinegrainedScaleZero && PackedScale,
                  "host generic grouped K-pack consumes packed metadata");
    launch_rc = launch_grouped_kpack_config<Low, High, GroupSize, TileK>(
        config, reinterpret_cast<uint16_t const*>(da.p),
        reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p),
        reinterpret_cast<uint8_t const*>(ds.p), d_out_ptrs.as<half_t*>(),
        d_out_strides.as<DS>(), d_rows.as<int>(), max_rows, n, k, experts,
        d_shapes.as<GS>(), shapes.data(), d_offsets.as<int>(), ws.as<char>(),
        ws_bytes, nullptr);
#endif
  } else {
    launch_rc = launch_grouped_config<QuantOp, PackedScale,
        Low, High, GroupSize, TileK>(
        config, reinterpret_cast<uint16_t const*>(da.p),
        reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p), ds.p, nullptr,
        d_out_ptrs.as<half_t*>(), d_out_strides.as<DS>(), d_rows.as<int>(),
        max_rows, n, k, experts, d_shapes.as<GS>(), shapes.data(),
        d_offsets.as<int>(), ws.as<char>(), ws_bytes, nullptr);
  }
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync(PackedScale ? "fully-quantized grouped GEMM" : "scale-first grouped GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(total_rows) * n);
}

}  // namespace

// SCALE_FIRST x GROUPED. The resident artifact is the placed low/high code planes plus one fp16 scale plane;
// there is deliberately no zero pointer because this ABI names the already validated FinegrainedScaleOnly
// collective used by test_scalefirst_bench. Provider selection happened when the artifact was packed. Tactics
// selected here may change tile geometry, never reinterpret that artifact as fully-quantized packed units.
extern "C" int quactlize_ppu_grouped_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int qtype, char const* config_name) {
  if (!act || !low || !scale || !rows_per_expert || !out || total_rows <= 0 || n <= 0 || k <= 0 ||
      experts <= 0 || n % 256 || k % 256) return 30;
  GroupedConfigId const config = resolve_grouped_config(config_name);
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    // This format-selected builder redirects the grouped type to packed metadata; its scale copy does not cover
    // every slot at the scale-first TileK. Refuse the fp16 provider instead of compiling a truncated copy.
    case 10: return 36;
#else
    case 10: return group_size == 16
        ? grouped<GQM::FinegrainedScaleOnly, false, cutlass::uint2b_t, void, 16,
                  ppu_formats::for_qtype(10).scale_first_tile_k>(
              act, low, nullptr, scale, rows_per_expert, out, total_rows, n, k, experts, config)
        : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    case 11: return 36;
#else
    case 11: return group_size == 16 && high
        ? grouped<GQM::FinegrainedScaleOnly, false, cutlass::uint2b_t, cutlass::uint1b_t, 16,
                  ppu_formats::for_qtype(11).scale_first_tile_k>(
              act, low, high, scale, rows_per_expert, out, total_rows, n, k, experts, config)
        : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
    case 12: return group_size == 32
        ? grouped<GQM::FinegrainedScaleOnly, false, cutlass::int4b_t, void, 32,
                  ppu_formats::for_qtype(12).scale_first_tile_k>(
              act, low, nullptr, scale, rows_per_expert, out, total_rows, n, k, experts, config)
        : 32;
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    case 13: return 36;
#else
    case 13: return group_size == 32 && high
        ? grouped<GQM::FinegrainedScaleOnly, false, cutlass::int4b_t, cutlass::uint1b_t, 32,
                  ppu_formats::for_qtype(13).scale_first_tile_k>(
              act, low, high, scale, rows_per_expert, out, total_rows, n, k, experts, config)
        : 32;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    case 14: return 36;
#else
    case 14: return group_size == 16 && high
        ? grouped<GQM::FinegrainedScaleOnly, false, cutlass::int4b_t, cutlass::uint2b_t, 16,
                  ppu_formats::for_qtype(14).scale_first_tile_k>(
              act, low, high, scale, rows_per_expert, out, total_rows, n, k, experts, config)
        : 32;
#endif
#endif
    default: return 33;
  }
}

extern "C" int quactlize_ppu_grouped_lowbit(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int qtype) {
  return quactlize_ppu_grouped_lowbit_config_v1(
      act, low, high, scale, rows_per_expert, out,
      total_rows, n, k, group_size, experts, qtype, nullptr);
}

extern "C" int64_t quactlize_ppu_grouped_lowbit_workspace_bytes_v1(
    int max_rows, int n, int k, int group_size, int experts, int qtype) {
  if (!grouped_lowbit_config_valid(kDefaultGroupedConfig, max_rows, n, k, group_size,
                                   experts, max_rows, qtype)) return -1;
  return int64_t(grouped_workspace_layout(max_rows, n, experts).total);
}

extern "C" int quactlize_ppu_grouped_lowbit_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name) {
  GroupedConfigId const config = resolve_grouped_config(config_name);
  int64_t const need = quactlize_ppu_grouped_lowbit_workspace_bytes_v1(
      max_rows, n, k, group_size, experts, qtype);
  if (!act || !low || !scale || !offsets || !out || !workspace || total_rows <= 0 ||
      need < 0 || workspace_bytes < need ||
      !grouped_lowbit_config_valid(config, total_rows, n, k, group_size, experts, max_rows, qtype)) return 30;
  ppu_gemv::rt_clear_error();
  hggcStream_t const s = static_cast<hggcStream_t>(stream);
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    case 10: return 36;
#else
    case 10: return grouped_device<GQM::FinegrainedScaleOnly, false,
        cutlass::uint2b_t, void, 16, ppu_formats::for_qtype(10).scale_first_tile_k>(
            act, low, nullptr, scale, offsets, out, total_rows, n, k, experts, max_rows,
            config, workspace, size_t(workspace_bytes), s);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    case 11: return 36;
#else
    case 11: return high ? grouped_device<GQM::FinegrainedScaleOnly, false,
        cutlass::uint2b_t, cutlass::uint1b_t, 16, ppu_formats::for_qtype(11).scale_first_tile_k>(
            act, low, high, scale, offsets, out, total_rows, n, k, experts, max_rows,
            config, workspace, size_t(workspace_bytes), s) : 33;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
    case 12: return grouped_device<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, void, 32, ppu_formats::for_qtype(12).scale_first_tile_k>(
            act, low, nullptr, scale, offsets, out, total_rows, n, k, experts, max_rows,
            config, workspace, size_t(workspace_bytes), s);
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    case 13: return 36;
#else
    case 13: return high ? grouped_device<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, cutlass::uint1b_t, 32, ppu_formats::for_qtype(13).scale_first_tile_k>(
            act, low, high, scale, offsets, out, total_rows, n, k, experts, max_rows,
            config, workspace, size_t(workspace_bytes), s) : 33;
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    case 14: return 36;
#else
    case 14: return high ? grouped_device<GQM::FinegrainedScaleOnly, false,
        cutlass::int4b_t, cutlass::uint2b_t, 16, ppu_formats::for_qtype(14).scale_first_tile_k>(
            act, low, high, scale, offsets, out, total_rows, n, k, experts, max_rows,
            config, workspace, size_t(workspace_bytes), s) : 33;
#endif
#endif
    default: return 33;
  }
}

extern "C" int quactlize_ppu_grouped_lowbit_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint16_t const* scale,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream) {
  return quactlize_ppu_grouped_lowbit_dev_v2(
      act, low, high, scale, offsets, out, total_rows, n, k, group_size, experts, max_rows, qtype,
      workspace, workspace_bytes, stream, nullptr);
}

// FULLY_QUANTIZED x GROUPED. The artifact is low codes, an optional high code plane, and format-shaped paired units;
// activations and
// output are concatenated in expert order and rows_per_expert supplies the ragged boundaries. All tensor-core work
// remains in moe_grouped_ppu's existing grouped scheduler and shared packed-scale collective. This wrapper only
// materialises the raw-pointer arrays that the grouped CUTLASS interface requires.
extern "C" int quactlize_ppu_grouped_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* rows_per_expert,
    uint16_t* out, int total_rows, int n, int k, int experts, int qtype, char const* config_name) {
  if (!act || !low || !units || !rows_per_expert || !out || total_rows <= 0 || n <= 0 || k <= 0 ||
      experts <= 0 || n % 256 || k % 256) return 30;
  GroupedConfigId const config = resolve_grouped_config(config_name);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
      act, low, nullptr, units, rows_per_expert, out, total_rows, n, k, experts, config);
#elif PPU_PACKED_FORMAT == 2
  if (qtype != 10) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
      act, low, nullptr, units, rows_per_expert, out, total_rows, n, k, experts, config);
#elif PPU_PACKED_FORMAT == 1
  if (qtype != 13 || !high) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
      act, low, high, units, rows_per_expert, out, total_rows, n, k, experts, config);
#elif PPU_PACKED_FORMAT == 3
  if (qtype != 11 || !high || k % 512) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
      act, low, high, units, rows_per_expert, out, total_rows, n, k, experts, config);
#elif PPU_PACKED_FORMAT == 4
  if (qtype != 14 || !high || k % 512) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
      act, low, high, units, rows_per_expert, out, total_rows, n, k, experts, config);
#else
  (void)qtype;
  return 35;
#endif
#else
  (void)config;
  (void)qtype;
  return 34;
#endif
}

extern "C" int quactlize_ppu_grouped_fully_quantized_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, int const* rows_per_expert, uint16_t* out,
    int total_rows, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    char const* config_name) {
  if (!act || !low || !units || !rows_per_expert || !out ||
      total_rows <= 0 || n <= 0 || k <= 0 || experts <= 0) {
    return 30;
  }
  GroupedConfigId config{};
  if (!find_grouped_tensor_config(config_name, config)) return 39;
  int max_rows = 0;
  int64_t row_sum = 0;
  for (int e = 0; e < experts; ++e) {
    if (rows_per_expert[e] < 0) return 36;
    max_rows = std::max(max_rows, rows_per_expert[e]);
    row_sum += rows_per_expert[e];
  }
  if (row_sum != total_rows ||
      !grouped_fully_quantized_config_valid(
          config, total_rows, n, k, qtype_group_size(qtype),
          experts, max_rows, qtype,
          arrangement)) {
    return 30;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    // The v1 grouped entry is the exact Xplane implementation; the v2
    // descriptor guard above proves that its compiled artifact TileK matches.
    return quactlize_ppu_grouped_fully_quantized_config_v1(
        act, low, high, units, rows_per_expert, out, total_rows, n, k,
        experts, qtype, config_name);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
#if PPU_PACKED_FORMAT == 2
    if (qtype != 10 || high) return 33;
    return grouped<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t,
                   void, 16, 256, false, true>(
        act, low, nullptr, units, rows_per_expert, out, total_rows, n, k,
        experts, config);
#elif PPU_PACKED_FORMAT == 3
    if (qtype != 11 || !high) return 33;
    return grouped<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t,
                   cutlass::uint1b_t, 16, 256, false, true>(
        act, low, high, units, rows_per_expert, out, total_rows, n, k,
        experts, config);
#elif PPU_PACKED_FORMAT == 1
    if (qtype != 13 || !high) return 33;
    return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t,
                   cutlass::uint1b_t, 32, 256, false, true>(
        act, low, high, units, rows_per_expert, out, total_rows, n, k,
        experts, config);
#elif PPU_PACKED_FORMAT == 4
    if (qtype != 14 || !high) return 33;
    return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t,
                   cutlass::uint2b_t, 16, 128, false, true>(
        act, low, high, units, rows_per_expert, out, total_rows, n, k,
        experts, config);
#else
    return 35;
#endif
#else
    (void)high;
    return 34;
#endif
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  if (qtype != 12 || high) return 33;
  return grouped<GQM::FinegrainedScaleZero, true, cutlass::int4b_t,
                 void, 32, 256, true>(
      act, low, nullptr, units, rows_per_expert, out, total_rows, n, k,
      experts, config);
#else
  return 34;
#endif
}

extern "C" int quactlize_ppu_grouped_fully_quantized(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* rows_per_expert,
    uint16_t* out, int total_rows, int n, int k, int experts, int qtype) {
  return quactlize_ppu_grouped_fully_quantized_config_v1(
      act, low, high, units, rows_per_expert, out, total_rows, n, k, experts, qtype, nullptr);
}

extern "C" int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1(
    int max_rows, int n, int k, int experts, int qtype) {
  if (max_rows <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 || experts <= 0 ||
      !selected_fully_quantized_qtype(qtype, k)) return -1;
  return int64_t(grouped_workspace_layout(max_rows, n, experts).total);
}

extern "C" int64_t quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
    int total_rows, int max_rows, int n, int k, int experts, int qtype,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  if (!grouped_fully_quantized_config_valid(
          kDefaultGroupedConfig, total_rows, n, k, qtype_group_size(qtype),
          experts, max_rows,
          qtype, arrangement)) {
    return -1;
  }
  return int64_t(grouped_workspace_layout(max_rows, n, experts).total);
}

extern "C" int quactlize_ppu_grouped_fully_quantized_dev_for_arrangement_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint8_t const* units, int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream,
    char const* config_name,
    quactlize_ppu_placed_arrangement_v2 const* arrangement) {
  GroupedConfigId config{};
  if (!find_grouped_tensor_config(config_name, config)) return 39;
  int64_t const need =
      quactlize_ppu_grouped_fully_quantized_workspace_bytes_for_arrangement_v2(
          total_rows, max_rows, n, k, experts, qtype, arrangement);
  if (!act || !low || !units || !offsets || !out || !workspace ||
      total_rows <= 0 || need < 0 || workspace_bytes < need ||
      !grouped_fully_quantized_config_valid(
          config, total_rows, n, k, qtype_group_size(qtype),
          experts, max_rows, qtype,
          arrangement)) {
    return 30;
  }
  if (arrangement->layout == QUACTLIZE_PPU_LAYOUT_XPLANE_V1) {
    // Reuse the already-instantiated exact Xplane device path. It consumes the
    // same device offsets and workspace; only v2 descriptor admission was
    // missing.
    return quactlize_ppu_grouped_fully_quantized_dev_v2(
        act, low, high, units, offsets, out, total_rows, n, k, experts,
        max_rows, qtype, workspace, workspace_bytes, stream, config_name);
  }
  if (arrangement->layout ==
          QUACTLIZE_PPU_LAYOUT_KQUANT_KPACK_TRANSPOSE_V1) {
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    defined(PPU_PACKED_FORMAT)
    ppu_gemv::rt_clear_error();
    hggcStream_t const s = static_cast<hggcStream_t>(stream);
#if PPU_PACKED_FORMAT == 2
    if (qtype != 10 || high) return 33;
    return grouped_device<GQM::FinegrainedScaleZero, true,
        cutlass::uint2b_t, void, 16, 256, false, true>(
            act, low, nullptr, units, offsets, out, total_rows, n, k,
            experts, max_rows, config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 3
    if (qtype != 11 || !high) return 33;
    return grouped_device<GQM::FinegrainedScaleZero, true,
        cutlass::uint2b_t, cutlass::uint1b_t, 16, 256, false, true>(
            act, low, high, units, offsets, out, total_rows, n, k,
            experts, max_rows, config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 1
    if (qtype != 13 || !high) return 33;
    return grouped_device<GQM::FinegrainedScaleZero, true,
        cutlass::int4b_t, cutlass::uint1b_t, 32, 256, false, true>(
            act, low, high, units, offsets, out, total_rows, n, k,
            experts, max_rows, config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 4
    if (qtype != 14 || !high) return 33;
    return grouped_device<GQM::FinegrainedScaleZero, true,
        cutlass::int4b_t, cutlass::uint2b_t, 16, 128, false, true>(
            act, low, high, units, offsets, out, total_rows, n, k,
            experts, max_rows, config, workspace, size_t(workspace_bytes), s);
#else
    return 35;
#endif
#else
    (void)high; (void)stream;
    return 34;
#endif
  }
#if QUACTLIZE_PPU_DENSE_W4_SPLITK_ENABLED && \
    defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && \
    (!defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0)
  if (qtype != 12 || high) return 33;
  ppu_gemv::rt_clear_error();
  return grouped_device<GQM::FinegrainedScaleZero, true,
                        cutlass::int4b_t, void, 32, 256, true>(
      act, low, nullptr, units, offsets, out, total_rows, n, k, experts,
      max_rows, config, workspace, size_t(workspace_bytes),
      static_cast<hggcStream_t>(stream));
#else
  (void)stream;
  return 34;
#endif
}

extern "C" int quactlize_ppu_grouped_fully_quantized_dev_v2(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream, char const* config_name) {
  int64_t const need = quactlize_ppu_grouped_fully_quantized_workspace_bytes_v1(
      max_rows, n, k, experts, qtype);
  if (!act || !low || !units || !offsets || !out || !workspace || total_rows <= 0 ||
      need < 0 || workspace_bytes < need) return 30;
  ppu_gemv::rt_clear_error();
  hggcStream_t const s = static_cast<hggcStream_t>(stream);
  GroupedConfigId const config = resolve_grouped_config(config_name);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return grouped_device<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
      act, low, nullptr, units, offsets, out, total_rows, n, k, experts, max_rows,
      config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 2
  return grouped_device<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
      act, low, nullptr, units, offsets, out, total_rows, n, k, experts, max_rows,
      config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 1
  if (!high) return 33;
  return grouped_device<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
      act, low, high, units, offsets, out, total_rows, n, k, experts, max_rows,
      config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 3
  if (!high) return 33;
  return grouped_device<GQM::FinegrainedScaleZero, true, cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
      act, low, high, units, offsets, out, total_rows, n, k, experts, max_rows,
      config, workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 4
  if (!high) return 33;
  return grouped_device<GQM::FinegrainedScaleZero, true, cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k>(
      act, low, high, units, offsets, out, total_rows, n, k, experts, max_rows,
      config, workspace, size_t(workspace_bytes), s);
#else
  return 35;
#endif
#else
  (void)high; (void)stream; (void)config;
  return 34;
#endif
}

extern "C" int quactlize_ppu_grouped_fully_quantized_dev_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units,
    int const* offsets, uint16_t* out,
    int total_rows, int n, int k, int experts, int max_rows, int qtype,
    void* workspace, int64_t workspace_bytes, void* stream) {
  return quactlize_ppu_grouped_fully_quantized_dev_v2(
      act, low, high, units, offsets, out, total_rows, n, k, experts, max_rows, qtype,
      workspace, workspace_bytes, stream, nullptr);
}
