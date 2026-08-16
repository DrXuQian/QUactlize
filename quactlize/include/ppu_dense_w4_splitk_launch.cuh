// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Exact production authority behind quactlize_ppu_dense_w4_splitk_*_v1.
// The C ABI wrapper in ppu_dense_backend.cu is intentionally thin so local
// type/call-edge gates instantiate this same implementation, not an oracle
// reconstruction.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "dense_splitk_parallel_ppu.cuh"
#include "ppu_dense_splitk_shipping_policy.hpp"
#include "ppu_group_schedule.hpp"
#include "quactlize_ppu_device.h"

namespace ppu_dense_w4_splitk {

namespace selector = ppu_dense_splitk_shipping;
using namespace cute;

using ProductionSchedule = ppu_group_schedule::FinegrainedSchedule<128>;
// This is the committed exact-warm TN64 row, not an inferred winner from the
// larger sweep: tactic 8x64x128 / warp 8x16x128 / s2 consumes the independently
// identified resident ArtifactTK64 bytes.  A profile chooses only S.
using ProductionTile = Shape<_8, _64, _128>;
using ProductionScaleTile = Shape<_64, _1>;
using ProductionWarp = Shape<_8, _16, _128>;
using ProductionShipping = fpa_intb_ppu::DensePackedAKernelTypes<
    1, fpa_intb_ppu::QuantMode::FinegrainedScaleOnly,
    ProductionSchedule, ProductionTile, ProductionScaleTile, ProductionWarp,
    2, true, cutlass::int4b_t, 64>;
using ProductionSplit = dense_splitk_parallel_ppu::KernelTypes<
    ProductionShipping, ProductionTile, ProductionWarp>;
using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
    ProductionShipping, ProductionTile, ProductionWarp>;
using TypeContract = selector::DispatchTypeContract<
    ProductionShipping, ProductionSplit>;

static_assert(std::is_same_v<typename TypeContract::S1Gemm,
                             typename ProductionShipping::Gemm> &&
                  std::is_same_v<typename TypeContract::S1Kernel,
                                 typename ProductionShipping::GemmKernel> &&
                  std::is_same_v<typename Prepared::ShippingGemm,
                                 typename ProductionShipping::Gemm>,
              "production S1 must remain the exact shipping Gemm type");
static_assert(std::is_same_v<typename ProductionShipping::CollectiveMainloop,
                             typename ProductionSplit::CollectiveMainloop>,
              "parallel production must retain the shipping mainloop");
static_assert(
    ProductionShipping::MainloopPolicy::Descriptor::quant_mode ==
            fpa_intb_ppu::QuantMode::FinegrainedScaleOnly &&
        ProductionShipping::MainloopPolicy::LowBits == 4 &&
        ProductionShipping::MainloopPolicy::HighBits == 0 &&
        ProductionShipping::MainloopPolicy::TacticTileK == 128 &&
        ProductionShipping::MainloopPolicy::ArtifactTileK == 64 &&
        ProductionShipping::MainloopPolicy::ArtifactLowFold == 1 &&
        ProductionShipping::MainloopPolicy::ArtifactHighFold == 1 &&
        ProductionShipping::MainloopPolicy::PackedARows == 1 &&
        ProductionShipping::CollectiveMainloop::DispatchPolicy::
                StaticGroupSize == 128,
    "production type must remain M1 W4 ScaleOnly gs128 tactic-TK128/artifact-TK64");
static_assert(std::is_same_v<typename ProductionShipping::ElementAccumulator,
                             float> &&
                  std::is_same_v<typename ProductionShipping::ElementD,
                                 cutlass::half_t>,
              "production Split-K ABI is FP32 partial to FP16 output");
static_assert(
    int(selector::QuantSemantics::FinegrainedScaleOnly) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY &&
        int(selector::QuantSemantics::FinegrainedScaleZero) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ZERO &&
        int(selector::MetadataStorage::Fp16Planes) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_FP16_PLANES &&
        int(selector::MetadataStorage::PackedUnits) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_PACKED_UNITS &&
        int(selector::ArtifactLayout::ResidentXPlane) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_RESIDENT_XPLANE &&
        int(selector::ArtifactLayout::Other) ==
            QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_OTHER,
    "C profile enum values must match the selector authority");

constexpr bool problem_is_in_fixed_abi(int m, int n, int k) {
  return m == 1 && n > 0 && k > 0 && n % 256 == 0 && k % 256 == 0;
}

constexpr selector::Key production_key(int m, int n, int k) {
  return {
      {m, n, k},
      {4, 0, 128, selector::QuantSemantics::FinegrainedScaleOnly,
       selector::MetadataStorage::Fp16Planes, false},
      {selector::ArtifactLayout::ResidentXPlane, 64, 1, 1, 0},
      {8, 64, 128, 8, 16, 2, 1, true},
  };
}

// Bind every non-problem profile coordinate back to the concrete production
// type.  The profile is an admission record, not a second kernel registry: a
// tactic or artifact edit must fail compilation until this identity changes
// with the type that is actually launched.
using ProductionDescriptor =
    typename ProductionShipping::MainloopPolicy::Descriptor;
inline constexpr selector::Key kProductionTypeIdentity =
    production_key(1, 256, 256);
static_assert(
    kProductionTypeIdentity.format.low_bits ==
            ProductionShipping::MainloopPolicy::LowBits &&
        kProductionTypeIdentity.format.high_bits ==
            ProductionShipping::MainloopPolicy::HighBits &&
        kProductionTypeIdentity.format.group_size ==
            ProductionShipping::CollectiveMainloop::DispatchPolicy::
                StaticGroupSize &&
        kProductionTypeIdentity.format.quant ==
            selector::QuantSemantics::FinegrainedScaleOnly &&
        kProductionTypeIdentity.format.metadata ==
            (ProductionDescriptor::packed_metadata
                 ? selector::MetadataStorage::PackedUnits
                 : selector::MetadataStorage::Fp16Planes) &&
        kProductionTypeIdentity.format.has_zero_plane ==
            ppu_mixed_policy::has_zero(ProductionDescriptor::quant_mode) &&
        kProductionTypeIdentity.artifact.layout ==
            selector::ArtifactLayout::ResidentXPlane &&
        kProductionTypeIdentity.artifact.tile_k ==
            ProductionShipping::MainloopPolicy::ArtifactTileK &&
        kProductionTypeIdentity.artifact.low_fold ==
            ProductionShipping::MainloopPolicy::ArtifactLowFold &&
        kProductionTypeIdentity.artifact.high_fold ==
            ProductionShipping::MainloopPolicy::ArtifactHighFold &&
        kProductionTypeIdentity.artifact.b_chunk ==
            int(ProductionDescriptor::atom_at_a_time) &&
        kProductionTypeIdentity.tactic.tile_m ==
            int(cute::size<0>(ProductionTile{})) &&
        kProductionTypeIdentity.tactic.tile_n ==
            int(cute::size<1>(ProductionTile{})) &&
        kProductionTypeIdentity.tactic.tile_k ==
            int(cute::size<2>(ProductionTile{})) &&
        kProductionTypeIdentity.tactic.warp_m ==
            int(cute::size<0>(ProductionWarp{})) &&
        kProductionTypeIdentity.tactic.warp_n ==
            int(cute::size<1>(ProductionWarp{})) &&
        kProductionTypeIdentity.tactic.stages ==
            ProductionDescriptor::stages &&
        kProductionTypeIdentity.tactic.packed_a_rows ==
            ProductionShipping::MainloopPolicy::PackedARows &&
        kProductionTypeIdentity.tactic.aiu_interleaved ==
            ProductionDescriptor::interleaved,
    "production profile identity must be derived from the launched kernel type");

inline bool decode_profile(
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile,
    selector::ProfileRow& decoded) {
  if (profile == nullptr) return false;
  auto const& key = profile->key;
  if ((key.quant_semantics !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY &&
       key.quant_semantics !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ZERO) ||
      (key.metadata_storage !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_FP16_PLANES &&
       key.metadata_storage !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_PACKED_UNITS) ||
      (key.artifact_layout !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_RESIDENT_XPLANE &&
       key.artifact_layout !=
           QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_OTHER) ||
      (key.has_zero_plane != 0 && key.has_zero_plane != 1) ||
      (key.aiu_interleaved != 0 && key.aiu_interleaved != 1)) {
    return false;
  }

  decoded.schema_version = profile->schema_version;
  decoded.key.problem = {key.rows, key.columns, key.inner};
  decoded.key.format = {
      key.low_bits, key.high_bits, key.group_size,
      static_cast<selector::QuantSemantics>(key.quant_semantics),
      static_cast<selector::MetadataStorage>(key.metadata_storage),
      key.has_zero_plane != 0};
  decoded.key.artifact = {
      static_cast<selector::ArtifactLayout>(key.artifact_layout),
      key.artifact_tile_k, key.artifact_low_fold,
      key.artifact_high_fold, key.artifact_b_chunk};
  decoded.key.tactic = {
      key.tactic_tile_m, key.tactic_tile_n, key.tactic_tile_k,
      key.tactic_warp_m, key.tactic_warp_n, key.tactic_stages,
      key.packed_a_rows, key.aiu_interleaved != 0};
  decoded.selected_s = profile->selected_s;
  return true;
}

inline selector::Selection select_profile(
    int m, int n, int k, std::uintptr_t workspace_address,
    std::size_t workspace_bytes,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile) {
  selector::ProfileRow decoded{};
  selector::ProfileRow const* decoded_ptr =
      decode_profile(profile, decoded) ? &decoded : nullptr;
  selector::Request const request{
      production_key(m, n, k), workspace_address, workspace_bytes};
  return selector::select(request, decoded_ptr);
}

inline std::int64_t query_workspace_bytes(
    int m, int n, int k,
    quactlize_ppu_dense_w4_splitk_profile_v1 const* profile) {
  if (!problem_is_in_fixed_abi(m, n, k)) return -1;
  selector::Selection const selected = select_profile(
      m, n, k, selector::kMeasuredWorkspaceAlignment,
      (std::numeric_limits<std::size_t>::max)(), profile);
  if (!selected.parallel_selected()) return 0;
  std::size_t const required = selector::required_partial_bytes(
      production_key(m, n, k).problem, selected.split_k_slices());
  if (required == 0 || required >
          static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)())) {
    return -1;
  }
  return static_cast<std::int64_t>(required);
}

// Two production call edges, both into the concrete Prepared type.  The S1
// arm carries a literal one and therefore reaches Prepared::ShippingGemm; only
// selector::ProfileSelectsParallel can forward a profile split to the FP32
// partial producer/reducer arm.
template <class SourceBindingOnly = void>
bool prepare_selected(
    selector::Selection selection, Prepared& prepared,
    cutlass::half_t const* act, cutlass::int4b_t const* weight_xplane,
    cutlass::half_t const* scales, cutlass::half_t* out,
    int m, int n, int k, char* workspace, std::size_t workspace_bytes,
    hggcStream_t stream) {
#if defined(QUACTLIZE_W4_SPLITK_SEVER_PREPARE_EDGE)
  (void)selection;
  (void)prepared;
  (void)act;
  (void)weight_xplane;
  (void)scales;
  (void)out;
  (void)m;
  (void)n;
  (void)k;
  (void)workspace;
  (void)workspace_bytes;
  (void)stream;
  return false;
#else
  return selector::dispatch_selected(
      selection,
      [&]() {
        return prepared.initialize(
            act, weight_xplane, scales, nullptr, out, m, n, k, 128, 1,
            workspace, workspace_bytes, stream);
      },
      [&](int selected_s) {
        return prepared.initialize(
            act, weight_xplane, scales, nullptr, out, m, n, k, 128,
            selected_s, workspace, workspace_bytes, stream);
      });
#endif
}

}  // namespace ppu_dense_w4_splitk
