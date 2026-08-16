// Copyright (c) 2026, quactlize contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Fail-closed selector authority for the measured dense fixed Split-K route.
//
// This policy deliberately does not reinterpret the GGUF Q4_K entry.  That
// route is FinegrainedScaleZero/gs32, while the only measured fixed Split-K
// shipping candidate is a separate W4 FinegrainedScaleOnly/gs128 resident
// xplane.  A profile row must match the complete problem, format, artifact and
// tactic key before it may select S>1.  Missing, stale or malformed profile
// data always returns S=1, which the integration hook dispatches through the
// caller's historical shipping callable verbatim.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

namespace ppu_dense_splitk_shipping {

inline constexpr std::uint32_t kProfileSchemaVersion = 1;
inline constexpr std::size_t kMeasuredWorkspaceAlignment = 128;

enum class QuantSemantics : std::uint8_t {
  FinegrainedScaleOnly,
  FinegrainedScaleZero,
};

enum class MetadataStorage : std::uint8_t {
  Fp16Planes,
  PackedUnits,
};

enum class ArtifactLayout : std::uint8_t {
  ResidentXPlane,
  Other,
};

struct ProblemKey {
  int rows = 0;
  int columns = 0;
  int inner = 0;
};

struct FormatKey {
  int low_bits = 0;
  int high_bits = 0;
  int group_size = 0;
  QuantSemantics quant = QuantSemantics::FinegrainedScaleZero;
  MetadataStorage metadata = MetadataStorage::PackedUnits;
  bool has_zero_plane = true;
};

struct ArtifactKey {
  ArtifactLayout layout = ArtifactLayout::Other;
  int tile_k = 0;
  int low_fold = 0;
  int high_fold = 0;
  int b_chunk = -1;
};

struct TacticKey {
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int warp_m = 0;
  int warp_n = 0;
  int stages = 0;
  int packed_a_rows = 0;
  bool aiu_interleaved = false;
};

struct Key {
  ProblemKey problem{};
  FormatKey format{};
  ArtifactKey artifact{};
  TacticKey tactic{};
};

constexpr bool operator==(ProblemKey const& lhs, ProblemKey const& rhs) {
  return lhs.rows == rhs.rows && lhs.columns == rhs.columns &&
      lhs.inner == rhs.inner;
}

constexpr bool operator==(FormatKey const& lhs, FormatKey const& rhs) {
  return lhs.low_bits == rhs.low_bits && lhs.high_bits == rhs.high_bits &&
      lhs.group_size == rhs.group_size && lhs.quant == rhs.quant &&
      lhs.metadata == rhs.metadata &&
      lhs.has_zero_plane == rhs.has_zero_plane;
}

constexpr bool operator==(ArtifactKey const& lhs, ArtifactKey const& rhs) {
  return lhs.layout == rhs.layout && lhs.tile_k == rhs.tile_k &&
      lhs.low_fold == rhs.low_fold && lhs.high_fold == rhs.high_fold &&
      lhs.b_chunk == rhs.b_chunk;
}

constexpr bool operator==(TacticKey const& lhs, TacticKey const& rhs) {
  return lhs.tile_m == rhs.tile_m && lhs.tile_n == rhs.tile_n &&
      lhs.tile_k == rhs.tile_k && lhs.warp_m == rhs.warp_m &&
      lhs.warp_n == rhs.warp_n && lhs.stages == rhs.stages &&
      lhs.packed_a_rows == rhs.packed_a_rows &&
      lhs.aiu_interleaved == rhs.aiu_interleaved;
}

constexpr bool operator==(Key const& lhs, Key const& rhs) {
  return lhs.problem == rhs.problem && lhs.format == rhs.format &&
      lhs.artifact == rhs.artifact && lhs.tactic == rhs.tactic;
}

struct ProfileRow {
  std::uint32_t schema_version = 0;
  Key key{};
  int selected_s = 1;
};

struct Request {
  Key key{};
  std::uintptr_t workspace_address = 0;
  std::size_t workspace_bytes = 0;
};

enum class DecisionReason : std::uint8_t {
  NoProfile,
  UnsupportedDomain,
  StaleProfileSchema,
  StaleProfileKey,
  InvalidProfileSplit,
  ProfileSelectsS1,
  InadmissiblePartition,
  InsufficientWorkspace,
  ProfileSelectsParallel,
};

class Selection;
constexpr Selection select(
    Request const& request, ProfileRow const* profile = nullptr);

class Selection {
 private:
  int split_k_slices_ = 1;
  DecisionReason reason_ = DecisionReason::NoProfile;

  constexpr Selection(int split_k_slices, DecisionReason reason)
      : split_k_slices_(split_k_slices), reason_(reason) {}
  friend constexpr Selection select(
      Request const& request, ProfileRow const* profile);

 public:
  Selection() = delete;
  constexpr int split_k_slices() const { return split_k_slices_; }
  constexpr DecisionReason reason() const { return reason_; }
  constexpr bool parallel_selected() const {
    return reason_ == DecisionReason::ProfileSelectsParallel &&
        split_k_slices_ > 1;
  }
};

constexpr bool is_profile_split(int split_k_slices) {
  return split_k_slices == 1 || split_k_slices == 2 ||
      split_k_slices == 4 || split_k_slices == 8;
}

constexpr bool is_stage_axis_value(int stages) {
  return stages == 2 || stages == 3 || stages == 4 || stages == 6 ||
      stages == 8 || stages == 12;
}

// This is intentionally narrower than the reusable kernel type.  It is the
// measured/profile-eligible W4 slice, not a claim that every type which happens
// to compile should enter production ranking.
constexpr bool is_proven_w4_domain(Key const& key) {
  auto const& p = key.problem;
  auto const& f = key.format;
  auto const& a = key.artifact;
  auto const& t = key.tactic;
  return p.rows == 1 && p.columns > 0 && p.inner > 0 &&
      p.columns % 256 == 0 && p.inner % 256 == 0 &&
      f.low_bits == 4 && f.high_bits == 0 && f.group_size == 128 &&
      f.quant == QuantSemantics::FinegrainedScaleOnly &&
      f.metadata == MetadataStorage::Fp16Planes && !f.has_zero_plane &&
      a.layout == ArtifactLayout::ResidentXPlane && a.tile_k == 64 &&
      a.low_fold == 1 && a.high_fold == 1 && a.b_chunk == 0 &&
      t.tile_m == 8 && t.tile_n > 0 && t.tile_k >= a.tile_k &&
      t.tile_k % a.tile_k == 0 && t.warp_m == 8 && t.warp_n > 0 &&
      t.tile_n % t.warp_n == 0 && t.tile_n % 16 == 0 &&
      t.warp_n % 16 == 0 && is_stage_axis_value(t.stages) &&
      t.packed_a_rows == 1 && t.aiu_interleaved;
}

constexpr std::size_t required_partial_bytes(
    ProblemKey const& problem, int split_k_slices) {
  if (problem.rows <= 0 || problem.columns <= 0 ||
      split_k_slices <= 1) {
    return 0;
  }
  std::size_t const rows = static_cast<std::size_t>(problem.rows);
  std::size_t const columns = static_cast<std::size_t>(problem.columns);
  std::size_t const splits = static_cast<std::size_t>(split_k_slices);
  constexpr std::size_t limit = (std::numeric_limits<std::size_t>::max)();
  if (rows > limit / columns) return 0;
  std::size_t const elements = rows * columns;
  if (elements > limit / splits) return 0;
  std::size_t const partial_elements = elements * splits;
  if (partial_elements > limit / sizeof(float)) return 0;
  return partial_elements * sizeof(float);
}

constexpr bool partition_is_admissible(Key const& key, int split_k_slices) {
  int const tactic_k = key.tactic.tile_k;
  if (tactic_k <= 0 || key.problem.inner % tactic_k != 0 ||
      !is_profile_split(split_k_slices)) {
    return false;
  }
  int const k_tiles = key.problem.inner / tactic_k;
  return k_tiles % split_k_slices == 0 &&
      k_tiles / split_k_slices >= key.tactic.stages - 1;
}

constexpr bool workspace_is_admissible(
    Request const& request, int split_k_slices) {
  std::size_t const required =
      required_partial_bytes(request.key.problem, split_k_slices);
  return required != 0 && request.workspace_address != 0 &&
      request.workspace_address % kMeasuredWorkspaceAlignment == 0 &&
      request.workspace_bytes >= required;
}

constexpr Selection select(
    Request const& request, ProfileRow const* profile) {
  if (!is_proven_w4_domain(request.key)) {
    return {1, DecisionReason::UnsupportedDomain};
  }
  if (profile == nullptr) return {1, DecisionReason::NoProfile};
  if (profile->schema_version != kProfileSchemaVersion) {
    return {1, DecisionReason::StaleProfileSchema};
  }
  if (!(profile->key == request.key)) {
    return {1, DecisionReason::StaleProfileKey};
  }
  if (!is_profile_split(profile->selected_s)) {
    return {1, DecisionReason::InvalidProfileSplit};
  }
  if (profile->selected_s == 1) {
    return {1, DecisionReason::ProfileSelectsS1};
  }
  if (!partition_is_admissible(request.key, profile->selected_s)) {
    return {1, DecisionReason::InadmissiblePartition};
  }
  if (!workspace_is_admissible(request, profile->selected_s)) {
    return {1, DecisionReason::InsufficientWorkspace};
  }
  return {profile->selected_s, DecisionReason::ProfileSelectsParallel};
}

constexpr char const* reason_name(DecisionReason reason) {
  switch (reason) {
    case DecisionReason::NoProfile: return "no-profile";
    case DecisionReason::UnsupportedDomain: return "unsupported-domain";
    case DecisionReason::StaleProfileSchema: return "stale-profile-schema";
    case DecisionReason::StaleProfileKey: return "stale-profile-key";
    case DecisionReason::InvalidProfileSplit: return "invalid-profile-split";
    case DecisionReason::ProfileSelectsS1: return "profile-selects-s1";
    case DecisionReason::InadmissiblePartition: return "inadmissible-partition";
    case DecisionReason::InsufficientWorkspace: return "insufficient-workspace";
    case DecisionReason::ProfileSelectsParallel: return "profile-selects-parallel";
  }
  return "unknown";
}

// One clearly named integration hook.  The S==1 callable is supplied by the
// existing shipping dispatch and is invoked directly for every fallback.  The
// parallel callable receives the validated split count only after select()
// has produced ProfileSelectsParallel.
template <class ShippingLaunch, class ParallelLaunch>
auto dispatch_selected(
    Selection selection, ShippingLaunch&& shipping_launch,
    ParallelLaunch&& parallel_launch)
    -> std::invoke_result_t<ShippingLaunch> {
  using ShippingResult = std::invoke_result_t<ShippingLaunch>;
  using ParallelResult = std::invoke_result_t<ParallelLaunch, int>;
  static_assert(std::is_same_v<ShippingResult, ParallelResult>,
                "S1 and fixed Split-K dispatch arms must return one ABI type");
  if (selection.parallel_selected() &&
      is_profile_split(selection.split_k_slices())) {
    return std::invoke(std::forward<ParallelLaunch>(parallel_launch),
                       selection.split_k_slices());
  }
  return std::invoke(std::forward<ShippingLaunch>(shipping_launch));
}

// Compile-time integration witness: S==1 names ShippingTypes::Gemm and
// ShippingTypes::GemmKernel directly; S>1 may change only the output-phase
// kernel while retaining the exact shipping collective/mainloop.
template <class ShippingTypes, class SplitTypes>
struct DispatchTypeContract {
  using S1Gemm = typename ShippingTypes::Gemm;
  using S1Kernel = typename ShippingTypes::GemmKernel;
  using ParallelGemm = typename SplitTypes::Gemm;
  using ParallelKernel = typename SplitTypes::GemmKernel;
  static_assert(std::is_same_v<typename ShippingTypes::CollectiveMainloop,
                               typename SplitTypes::CollectiveMainloop>,
                "fixed Split-K selector must reuse the exact shipping mainloop");
};

}  // namespace ppu_dense_splitk_shipping
