// hgcc half of SCALE_FIRST x DENSE: raw host pointers -> the dedicated fpA mixed-input launcher. The offline weight
// reorder lives in ppu_dense_layout.cu so the resident artifact crosses this ABI already in the kernel's layout.
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fpA_intB_ppu.cuh"
#include "moe_grouped_ppu.cuh"
#include "gemv_lowbit/gemv_rt.hpp"
#include "ppu_dense_configs.inc"
#include "ppu_format_config.hpp"
#include "ppu_grouped_configs.inc"
#include "quactlize_ppu_device.h"

// The optional collectives this backend INSTANTIATES: it ships Q3_K (uint2+uint1, two planes) and folded
// artifacts alongside the single-plane path. quactlize_actlize.hpp carries the base only.
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_fold.hpp"
#include "quactlize_extensions/cutlass/gemm/collective/ppu_mma_aiu_mixed_input_2plane.hpp"

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

enum class DenseConfigId {
#define QUACTLIZE_PPU_DENSE_CONFIG_ID(ID, NAME, TM, TN, WM, WN, STAGES) ID,
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_ID)
#undef QUACTLIZE_PPU_DENSE_CONFIG_ID
  Count,
};

constexpr quactlize_ppu_config_v1 kDenseConfigs[] = {
#define QUACTLIZE_PPU_DENSE_CONFIG_ROW(ID, NAME, TM, TN, WM, WN, STAGES) \
  {false, NAME, TM, TN, WM, WN, STAGES},
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_ROW)
#undef QUACTLIZE_PPU_DENSE_CONFIG_ROW
  // The scale-first CUDA-core family consumes the same logical planes but has no tensor tile geometry.
  {true, QUACTLIZE_PPU_DENSE_CUDA_CONFIG_NAME, 0, 0, 0, 0, 0},
};
constexpr DenseConfigId kDefaultDenseConfig = DenseConfigId::Default;
static_assert(sizeof(kDenseConfigs) / sizeof(kDenseConfigs[0]) > 1,
              "libquactlize_ppu must compile a config set, not one frozen tactic");
static_assert(sizeof(kDenseConfigs) / sizeof(kDenseConfigs[0]) == size_t(DenseConfigId::Count) + 1,
              "the dense inventory must contain every tensor-core config followed by its one CUDA tactic");

constexpr int minimum_dense_tile_m() {
  int value = kDenseConfigs[0].tile_m;
  for (int i = 0; i < int(DenseConfigId::Count); ++i) value = std::min(value, kDenseConfigs[i].tile_m);
  return value;
}

constexpr int minimum_dense_tile_n() {
  int value = kDenseConfigs[0].tile_n;
  for (int i = 0; i < int(DenseConfigId::Count); ++i) value = std::min(value, kDenseConfigs[i].tile_n);
  return value;
}

bool find_dense_config(char const* name, DenseConfigId& config) {
  if (!name || !name[0]) { config = kDefaultDenseConfig; return true; }
#define QUACTLIZE_PPU_DENSE_CONFIG_MATCH(ID, NAME, TM, TN, WM, WN, STAGES) \
  if (std::strcmp(name, NAME) == 0) { config = DenseConfigId::ID; return true; }
  QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_MATCH)
#undef QUACTLIZE_PPU_DENSE_CONFIG_MATCH
  return false;
}

DenseConfigId resolve_dense_config(char const* name) {
  DenseConfigId config{};
  if (find_dense_config(name, config)) return config;
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
  // One query serves every compiled tactic, so size for the largest possible CTA grid rather than for the default.
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
  layout.kernel_bytes = size_t(cutlass::ceil_div(max_rows, 16)) * cutlass::ceil_div(n, 64)
                      * size_t(experts) * 64;
  layout.total = layout.kernel + layout.kernel_bytes;
  return layout;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_grouped_tactic(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    void const* scale, uint16_t const* zero,
    half_t** out_ptrs, DS* out_strides, int const* rows,
    int max_rows, int n, int k, int experts, GS* shapes, GS const* shapes_host, int const* offsets,
    char* workspace, size_t workspace_bytes, hggcStream_t stream) {
  // The dense side's guard, for the grouped config list. Same two rows are illegal at TileK=256 -- 16x128 w16x32
  // and 32x128 w32x32 both want 256 scale-copy slots against 128 threads -- and for the same reason: GroupSize is
  // a compile-time 16 here, so ScaleCopyCoverage is exact rather than conservative. Grouped's own Default
  // (16x128 w16x16) survives at 8 warps: 256 slots against 256 threads, passing on equality.
  constexpr ppu_tactics::Candidate kTactic{
      {ppu_tactics::Format::I4, "grouped-backend",
       ppu_mixed_policy::element_bits_v<Low>, ppu_mixed_policy::element_bits_v<High>},
      TileM, TileN, TileK, WarpM, WarpN, TileK};
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
                          QueryOnly, RequireUniversalFallback, TileK>(
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
  }
  return 31;
}

template <class Low, class High, int GroupSize, int TileK, bool PackedScale,
          int TileM, int TileN, int WarpM, int WarpN, int Stages,
          bool QueryOnly = false, bool RequireUniversalFallback = false>
int launch_dense_tactic(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                        void const* scale, uint16_t const* zero, uint16_t* out,
                        int m, int n, int k, void* workspace, size_t workspace_bytes,
                        hggcStream_t stream) {
  // A SHIPPED CONFIG IS NOT LEGAL AT EVERY TileK, and the cross product is what this switch instantiates.
  //
  // QUACTLIZE_PPU_DENSE_CONFIGS names five tile/warp shapes and says nothing about TileK, because it was written
  // when TileK was one number for the whole binary. It is now a template parameter here, so each row is
  // instantiated at every TileK the callers use -- and two of the five are illegal at 256:
  //
  //   ShortWide 16x128 w16x32   Scale_CopyThreadSlots = (128/8) * ceil(256/16) = 256  >  32 * 4 warps = 128
  //   MidWide   32x128 w32x32   same arithmetic, same 256 > 128
  //
  // That is ScaleCopyCoverage, and here it is EXACT rather than conservative: GroupSize is a compile-time
  // parameter of this instantiation and the callers pass 16, so gs=16 is not a hypothetical this row must also
  // survive -- it is the row. ppu_mixed_policy's static_assert caught it as soon as the local tier was run in
  // full, which is what the assert is for.
  //
  // GUARDING RATHER THAN DELETING THE ROWS: both are legal and useful at TileK 64 and 128, so removing them
  // would cost coverage the sweep can use. Guarding costs a compile-time branch and makes the excluded pair
  // report "did not launch", which dense_config_type_valid already surfaces as unsupported.
  constexpr ppu_tactics::Candidate kTactic{
      {ppu_tactics::Format::I4, "dense-backend",
       ppu_mixed_policy::element_bits_v<Low>, ppu_mixed_policy::element_bits_v<High>},
      TileM, TileN, TileK, WarpM, WarpN, TileK};
  if constexpr (ppu_tactics::DenseSpace::kernel_exclusion(kTactic) != ppu_tactics::Exclusion::None) {
    // 31 is this file's existing "did not launch". A separate code would be more informative and would also be a
    // new ABI meaning for every caller that only tests truthiness, so the distinction stays in this comment.
    return 31;
  } else {
  constexpr int ScaleGroups = ppu_group_schedule::scale_groups_v<TileK, GroupSize>;
  using Tile = cute::Shape<cute::C<TileM>, cute::C<TileN>, cute::C<TileK>>;
  using Warp = cute::Shape<cute::C<WarpM>, cute::C<WarpN>, cute::C<TileK>>;
  bool const launched = fpa_intb_ppu::generic_launcher<QM::FinegrainedScaleZero,
      ppu_group_schedule::FinegrainedSchedule<GroupSize>,
      Tile, cute::Shape<cute::C<TileN>, cute::C<ScaleGroups>>, Warp, Stages, true,
      Low, High, PackedScale, QueryOnly, RequireUniversalFallback, TileK>(
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

template <class Low, class High, int GroupSize, int TileK, bool PackedScale, bool QueryOnly = false>
int launch_dense_config(DenseConfigId config, uint16_t const* act, uint8_t const* low, uint8_t const* high,
                        void const* scale, uint16_t const* zero, uint16_t* out,
                        int m, int n, int k, void* workspace, size_t workspace_bytes,
                        hggcStream_t stream) {
  switch (config) {
#define QUACTLIZE_PPU_DENSE_CONFIG_CASE(ID, NAME, TM, TN, WM, WN, STAGES) \
    case DenseConfigId::ID: \
      return launch_dense_tactic<Low, High, GroupSize, TileK, PackedScale, TM, TN, WM, WN, STAGES, QueryOnly, \
                                 DenseConfigId::ID == kDefaultDenseConfig>( \
          act, low, high, scale, zero, out, m, n, k, workspace, workspace_bytes, stream);
    QUACTLIZE_PPU_DENSE_CONFIGS(QUACTLIZE_PPU_DENSE_CONFIG_CASE)
#undef QUACTLIZE_PPU_DENSE_CONFIG_CASE
  }
  return 31;
}

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

template <class Low, class High, int GroupSize, int TileK, bool PackedScale>
bool dense_config_type_valid(DenseConfigId config, int m, int n, int k) {
  return launch_dense_config<Low, High, GroupSize, TileK, PackedScale, true>(
      config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      m, n, k, nullptr, 0, nullptr) == 0;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK>
bool grouped_config_type_valid(GroupedConfigId config, int max_rows, int n, int k, int experts) {
  return launch_grouped_config<QuantOp, PackedScale, Low, High, GroupSize, TileK, true>(
      config, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      max_rows, n, k, experts, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == 0;
}

bool dense_lowbit_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype) {
  if (!tensor_problem_domain(m, n, k, group_size, qtype)) return false;
  switch (qtype) {
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 10
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 2
    case 10: return dense_config_type_valid<cutlass::uint2b_t, void, 16,
        ppu_formats::for_qtype(10).scale_first_tile_k, false>(config, m, n, k);
#else
    case 10: return dense_config_type_valid<cutlass::uint2b_t, void, 16,
        ppu_formats::for_qtype(10).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 11
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 3
    case 11: return false;
#else
    case 11: return dense_config_type_valid<cutlass::uint2b_t, cutlass::uint1b_t, 16,
        ppu_formats::for_qtype(11).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 12
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
    case 12: return dense_config_type_valid<cutlass::int4b_t, void, 32,
        ppu_formats::for_qtype(12).scale_first_tile_k, false>(config, m, n, k);
#else
    case 12: return dense_config_type_valid<cutlass::int4b_t, void, 32,
        ppu_formats::for_qtype(12).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 13
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 1
    case 13: return false;
#else
    case 13: return dense_config_type_valid<cutlass::int4b_t, cutlass::uint1b_t, 32,
        ppu_formats::for_qtype(13).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
#if !defined(QUACTLIZE_DENSE_ONLY) || QUACTLIZE_DENSE_ONLY == 14
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0) && defined(PPU_PACKED_FORMAT) && PPU_PACKED_FORMAT == 4
    case 14: return false;
#else
    case 14: return dense_config_type_valid<cutlass::int4b_t, cutlass::uint2b_t, 16,
        ppu_formats::for_qtype(14).scale_first_tile_k, false>(config, m, n, k);
#endif
#endif
    default: return false;
  }
}

bool dense_fully_quantized_config_valid(
    DenseConfigId config, int m, int n, int k, int group_size, int qtype) {
  if (!tensor_problem_domain(m, n, k, group_size, qtype) ||
      !selected_fully_quantized_qtype(qtype, k)) return false;
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return dense_config_type_valid<cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k, true>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 2
  return dense_config_type_valid<cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k, true>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 1
  return dense_config_type_valid<cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k, true>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 3
  return dense_config_type_valid<cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k, true>(config, m, n, k);
#elif PPU_PACKED_FORMAT == 4
  return dense_config_type_valid<cutlass::int4b_t, cutlass::uint2b_t, 16,
      ppu_formats::for_qtype(14).fully_quantized_tile_k, true>(config, m, n, k);
#else
  return false;
#endif
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

quactlize_ppu_config_v2 config_v2(quactlize_ppu_config_v1 const& config, int tile_k) {
  return {config.enable_cuda_kernel, config.name, config.tile_m, config.tile_n, tile_k,
          config.warp_m, config.warp_n, config.stages};
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

template <class Low, class High = void, int GroupSize = 16, int TileK = 256>
int dense_fully_quantized_device(uint16_t const* act, uint8_t const* low, uint8_t const* high,
                                 uint8_t const* units, uint16_t* out, int m, int n, int k,
                                 DenseConfigId config, void* workspace, size_t workspace_bytes,
                                 hggcStream_t stream) {
  size_t const need = dense_workspace_bytes(m, n);
  if (!workspace || workspace_bytes < need) return 37;
  int const launch_rc = launch_dense_config<Low, High, GroupSize, TileK, true>(
      config, act, low, high, units, nullptr, out, m, n, k, workspace, workspace_bytes, stream);
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch("fully-quantized dense GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

template <class Low, class High = void, int GroupSize = 16, int TileK = 256, bool PackedScale = false>
int dense(uint16_t const* act, uint8_t const* low, uint8_t const* high,
          void const* scale, uint16_t const* zero, uint16_t* out,
          int m, int n, int k, int group_size, DenseConfigId config = kDefaultDenseConfig) {
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
    launch_rc = dense_fully_quantized_device<Low, High, GroupSize, TileK>(
        reinterpret_cast<uint16_t const*>(da.p), reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p), reinterpret_cast<uint8_t const*>(ds.p),
        reinterpret_cast<uint16_t*>(dout.p), m, n, k, config, ws.p, ws_bytes, nullptr);
  } else {
    launch_rc = launch_dense_config<Low, High, GroupSize, TileK, false>(
        config, reinterpret_cast<uint16_t const*>(da.p), reinterpret_cast<uint8_t const*>(dl.p),
        reinterpret_cast<uint8_t const*>(dh.p), ds.p, reinterpret_cast<uint16_t const*>(dz.p),
        reinterpret_cast<uint16_t*>(dout.p), m, n, k, ws.p, ws_bytes, nullptr);
  }
  if (launch_rc) return launch_rc;
  ppu_gemv::rt_sync(PackedScale ? "fully-quantized dense GEMM" : "scale-first dense GEMM");
  if (!ppu_gemv::rt_ok()) return ppu_gemv::kRuntimeError;
  return ppu_gemv::rt_copy_output(dout, out, size_t(m) * n);
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
  return find_dense_config(config_name, config) &&
         dense_lowbit_config_valid(config, m, n, k, group_size, qtype);
}

extern "C" int32_t quactlize_ppu_dense_fully_quantized_config_valid_v1(
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  DenseConfigId config{};
  return find_dense_config(config_name, config) &&
         dense_fully_quantized_config_valid(config, m, n, k, group_size, qtype);
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

extern "C" int32_t quactlize_ppu_list_valid_grouped_fully_quantized_configs_v2(
    quactlize_ppu_config_v2* configs, int32_t capacity,
    int total_rows, int n, int k, int group_size, int experts, int max_rows, int qtype) {
  return list_valid_grouped_configs_v2(
      configs, capacity, total_rows, n, k, group_size, experts, max_rows, qtype);
}

extern "C" int quactlize_ppu_dense_lowbit_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high,
    uint16_t const* scale, uint16_t const* zero, uint16_t* out,
    int m, int n, int k, int group_size, int qtype, char const* config_name) {
  if (!act || !low || !scale || !zero || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  DenseConfigId const config = resolve_dense_config(config_name);
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

// FULLY_QUANTIZED x DENSE, format-selected k-quants. This is only a second ABI contract: it instantiates the SAME
// dense() wrapper and CollectiveBuilder as scale-first. Q2/Q3/Q4/Q5 use TileK=256; Q6 keeps the required TileK=128
// high-plane placement and selects one of two group runs from each superblock. The two-plane collective's scale
// channel calls the same packed helpers and stages paired units without changing weight-plane placement.
// Builds without PPU_PACKED_SCALE retain the symbol but return 34.
extern "C" int quactlize_ppu_dense_fully_quantized_config_v1(
    uint16_t const* act, uint8_t const* low, uint8_t const* high, uint8_t const* units, uint16_t* out,
    int m, int n, int k, int qtype, char const* config_name) {
  if (!act || !low || !units || !out || m <= 0 || n <= 0 || k <= 0 || n % 256 || k % 256) return 30;
  DenseConfigId const config = resolve_dense_config(config_name);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  if (qtype != 12) return 33;
  return dense<cutlass::int4b_t, void, 32, ppu_formats::for_qtype(12).fully_quantized_tile_k, true>(
      act, low, nullptr, units, nullptr, out, m, n, k, 32, config);
#elif PPU_PACKED_FORMAT == 2
  if (qtype != 10) return 33;
  return dense<cutlass::uint2b_t, void, 16, ppu_formats::for_qtype(10).fully_quantized_tile_k, true>(
      act, low, nullptr, units, nullptr, out, m, n, k, 16, config);
#elif PPU_PACKED_FORMAT == 1
  if (qtype != 13 || !high) return 33;
  return dense<cutlass::int4b_t, cutlass::uint1b_t, 32,
               ppu_formats::for_qtype(13).fully_quantized_tile_k, true>(
      act, low, high, units, nullptr, out, m, n, k, 32, config);
#elif PPU_PACKED_FORMAT == 3
  if (qtype != 11 || !high || k % 512) return 33;
  return dense<cutlass::uint2b_t, cutlass::uint1b_t, 16,
               ppu_formats::for_qtype(11).fully_quantized_tile_k, true>(
      act, low, high, units, nullptr, out, m, n, k, 16, config);
#elif PPU_PACKED_FORMAT == 4
  if (qtype != 14 || !high || k % 512) return 33;
  return dense<cutlass::int4b_t, cutlass::uint2b_t, 16,
               ppu_formats::for_qtype(14).fully_quantized_tile_k, true>(
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

extern "C" int64_t quactlize_ppu_dense_fully_quantized_workspace_bytes_v1(
    int m, int n, int k, int qtype) {
  if (m <= 0 || n <= 0 || n % 256 || k <= 0 || k % 256 || !selected_fully_quantized_qtype(qtype, k)) return -1;
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
  DenseConfigId const config = resolve_dense_config(config_name);
#if defined(PPU_PACKED_SCALE) && (PPU_PACKED_SCALE != 0)
#if !defined(PPU_PACKED_FORMAT) || PPU_PACKED_FORMAT == 0
  return dense_fully_quantized_device<cutlass::int4b_t, void, 32,
      ppu_formats::for_qtype(12).fully_quantized_tile_k>(
      act, low, nullptr, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 2
  return dense_fully_quantized_device<cutlass::uint2b_t, void, 16,
      ppu_formats::for_qtype(10).fully_quantized_tile_k>(
      act, low, nullptr, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 1
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::int4b_t, cutlass::uint1b_t, 32,
      ppu_formats::for_qtype(13).fully_quantized_tile_k>(
      act, low, high, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 3
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::uint2b_t, cutlass::uint1b_t, 16,
      ppu_formats::for_qtype(11).fully_quantized_tile_k>(
      act, low, high, units, out, m, n, k, config,
      workspace, size_t(workspace_bytes), s);
#elif PPU_PACKED_FORMAT == 4
  if (!high) return 33;
  return dense_fully_quantized_device<cutlass::int4b_t, cutlass::uint2b_t, 16,
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

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK>
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

  int const launch_rc = launch_grouped_config<QuantOp, PackedScale,
      Low, High, GroupSize, TileK>(
      config, act, low, high, scale, nullptr, out_ptrs, out_strides, rows, max_rows, n, k, experts,
      shapes, nullptr, offsets, kernel_workspace, layout.kernel_bytes, stream);
  if (launch_rc) return launch_rc;
  return ppu_gemv::rt_check_launch(PackedScale ? "fully-quantized grouped GEMM enqueue"
                                               : "scale-first grouped GEMM enqueue")
      ? 0 : ppu_gemv::kRuntimeError;
}

template <GQM QuantOp, bool PackedScale, class Low, class High, int GroupSize, int TileK>
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
  int const launch_rc = launch_grouped_config<QuantOp, PackedScale,
      Low, High, GroupSize, TileK>(
      config, reinterpret_cast<uint16_t const*>(da.p), reinterpret_cast<uint8_t const*>(dl.p),
      reinterpret_cast<uint8_t const*>(dh.p), ds.p, nullptr,
      d_out_ptrs.as<half_t*>(), d_out_strides.as<DS>(), d_rows.as<int>(), max_rows, n, k, experts,
      d_shapes.as<GS>(), shapes.data(), d_offsets.as<int>(), ws.as<char>(), ws_bytes, nullptr);
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
