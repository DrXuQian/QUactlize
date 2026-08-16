// L200: executable C-profile/workspace contract for the real W4 Split-K
// production launch authority.  Device pointers are never dereferenced here;
// the runner separately forces prepare_selected into Prepared::initialize.
#include <cstddef>
#include <cstdio>
#include <type_traits>

#include "ppu_dense_w4_splitk_launch.cuh"

namespace production = ppu_dense_w4_splitk;
namespace selector = ppu_dense_splitk_shipping;

static_assert(std::is_standard_layout_v<
                  quactlize_ppu_dense_w4_splitk_key_v1> &&
              std::is_standard_layout_v<
                  quactlize_ppu_dense_w4_splitk_profile_v1>);
static_assert(sizeof(quactlize_ppu_dense_w4_splitk_key_v1) == 22 * 4);
static_assert(sizeof(quactlize_ppu_dense_w4_splitk_profile_v1) == 24 * 4);
static_assert(std::is_same_v<typename production::TypeContract::S1Gemm,
                             typename production::ProductionShipping::Gemm>);
static_assert(std::is_same_v<
                  typename production::ProductionShipping::CollectiveMainloop,
                  typename production::ProductionSplit::CollectiveMainloop>);

constexpr quactlize_ppu_dense_w4_splitk_profile_v1 profile(
    int m, int n, int k, int selected_s) {
  return {
      QUACTLIZE_PPU_DENSE_W4_SPLITK_PROFILE_SCHEMA_V1,
      {m, n, k,
       4, 0, 128,
       QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ONLY,
       QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_FP16_PLANES,
       0,
       QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_RESIDENT_XPLANE,
       64, 1, 1, 0,
       8, 64, 128, 8, 16, 2, 1, 1},
      selected_s};
}

#if defined(L200_FORCE_PRODUCTION_PREPARE)
[[maybe_unused]] auto const kProductionPrepareInstantiation =
    &production::prepare_selected<>;
#endif

int main() {
  constexpr int M = 1;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr std::uintptr_t Workspace = 0x1000;
  constexpr std::size_t S2Bytes = std::size_t(M) * N * 2 * sizeof(float);
  constexpr std::size_t S4Bytes = std::size_t(M) * N * 4 * sizeof(float);
  constexpr std::size_t S8Bytes = std::size_t(M) * N * 8 * sizeof(float);

  int errors = 0;
  int controls = 0;
  int shipping_calls = 0;
  int parallel_calls = 0;

  auto check = [&](char const* label, int m, int n, int k,
                   quactlize_ppu_dense_w4_splitk_profile_v1 const* row,
                   std::uintptr_t workspace_address,
                   std::size_t workspace_bytes,
                   selector::DecisionReason expected_reason, int expected_s,
                   bool expected_parallel, std::int64_t expected_query) {
    ++controls;
    selector::Selection const selected = production::select_profile(
        m, n, k, workspace_address, workspace_bytes, row);
    int const shipping_before = shipping_calls;
    int const parallel_before = parallel_calls;
    int const dispatched = selector::dispatch_selected(
        selected,
        [&]() {
          ++shipping_calls;
          return 101;
        },
        [&](int splits) {
          ++parallel_calls;
          return 200 + splits;
        });
    std::int64_t const query =
        production::query_workspace_bytes(m, n, k, row);
    bool const routed_parallel = parallel_calls == parallel_before + 1;
    bool const routed_shipping = shipping_calls == shipping_before + 1;
    bool const ok = selected.reason() == expected_reason &&
        selected.split_k_slices() == expected_s &&
        selected.parallel_selected() == expected_parallel &&
        routed_parallel == expected_parallel &&
        routed_shipping == !expected_parallel &&
        dispatched == (expected_parallel ? 200 + expected_s : 101) &&
        query == expected_query;
    std::printf(
        "[l200:case] name=%s reason=%s selected_s=%d route=%s "
        "query=%lld -> %s\n",
        label, selector::reason_name(selected.reason()),
        selected.split_k_slices(),
        routed_parallel ? "fixed-splitk" : "literal-shipping-s1",
        static_cast<long long>(query), ok ? "PASS" : "FAIL");
    errors += !ok;
  };

  auto s2 = profile(M, N, K, 2);
  auto s4 = profile(M, N, K, 4);
  auto s8 = profile(M, N, K, 8);
  check("exact-s2", M, N, K, &s2, Workspace, S2Bytes,
        selector::DecisionReason::ProfileSelectsParallel, 2, true, S2Bytes);
  check("exact-s4", M, N, K, &s4, Workspace, S4Bytes,
        selector::DecisionReason::ProfileSelectsParallel, 4, true, S4Bytes);
  check("exact-s8", M, N, K, &s8, Workspace, S8Bytes,
        selector::DecisionReason::ProfileSelectsParallel, 8, true, S8Bytes);

  auto changed = profile(M, N, K, 1);
  check("explicit-s1", M, N, K, &changed, 0, 0,
        selector::DecisionReason::ProfileSelectsS1, 1, false, 0);
  check("null-profile", M, N, K, nullptr, 0, 0,
        selector::DecisionReason::NoProfile, 1, false, 0);

  changed = s8;
  ++changed.schema_version;
  check("stale-schema", M, N, K, &changed, Workspace, S8Bytes,
        selector::DecisionReason::StaleProfileSchema, 1, false, 0);
  changed = s8;
  changed.selected_s = 3;
  check("invalid-s3", M, N, K, &changed, Workspace, S8Bytes,
        selector::DecisionReason::InvalidProfileSplit, 1, false, 0);

  auto stale_key = [&](char const* label,
                       quactlize_ppu_dense_w4_splitk_profile_v1 const& row) {
    check(label, M, N, K, &row, Workspace, S8Bytes,
          selector::DecisionReason::StaleProfileKey, 1, false, 0);
  };

  changed = s8; ++changed.key.rows; stale_key("key-rows", changed);
  changed = s8; changed.key.columns += 256; stale_key("key-columns", changed);
  changed = s8; changed.key.inner += 256; stale_key("key-inner", changed);
  changed = s8; --changed.key.low_bits; stale_key("key-low-bits", changed);
  changed = s8; ++changed.key.high_bits; stale_key("key-high-bits", changed);
  changed = s8; changed.key.group_size = 32; stale_key("key-group-size", changed);
  changed = s8;
  changed.key.quant_semantics =
      QUACTLIZE_PPU_DENSE_W4_SPLITK_QUANT_SCALE_ZERO;
  stale_key("key-quant", changed);
  changed = s8;
  changed.key.metadata_storage =
      QUACTLIZE_PPU_DENSE_W4_SPLITK_METADATA_PACKED_UNITS;
  stale_key("key-metadata", changed);
  changed = s8; changed.key.has_zero_plane = 1; stale_key("key-zero", changed);
  changed = s8;
  changed.key.artifact_layout =
      QUACTLIZE_PPU_DENSE_W4_SPLITK_ARTIFACT_OTHER;
  stale_key("key-artifact-layout", changed);
  changed = s8; changed.key.artifact_tile_k = 128; stale_key("key-artifact-tk", changed);
  changed = s8; changed.key.artifact_low_fold = 2; stale_key("key-low-fold", changed);
  changed = s8; changed.key.artifact_high_fold = 2; stale_key("key-high-fold", changed);
  changed = s8; changed.key.artifact_b_chunk = 1; stale_key("key-bchunk", changed);
  changed = s8; changed.key.tactic_tile_m = 16; stale_key("key-tile-m", changed);
  changed = s8; changed.key.tactic_tile_n = 128; stale_key("key-tile-n", changed);
  changed = s8; changed.key.tactic_tile_k = 64; stale_key("key-tile-k", changed);
  changed = s8; changed.key.tactic_warp_m = 16; stale_key("key-warp-m", changed);
  changed = s8; changed.key.tactic_warp_n = 32; stale_key("key-warp-n", changed);
  changed = s8; changed.key.tactic_stages = 3; stale_key("key-stages", changed);
  changed = s8; changed.key.packed_a_rows = 0; stale_key("key-packed-a", changed);
  changed = s8; changed.key.aiu_interleaved = 0; stale_key("key-interleaved", changed);

  check("short-workspace", M, N, K, &s8, Workspace, S8Bytes - 1,
        selector::DecisionReason::InsufficientWorkspace, 1, false, S8Bytes);
  check("misaligned-workspace", M, N, K, &s8, Workspace + 16, S8Bytes,
        selector::DecisionReason::InsufficientWorkspace, 1, false, S8Bytes);

  changed = s8;
  changed.key.quant_semantics = 7;
  check("malformed-enum", M, N, K, &changed, Workspace, S8Bytes,
        selector::DecisionReason::NoProfile, 1, false, 0);

  check("outside-m", 2, N, K, &s8, Workspace, S8Bytes,
        selector::DecisionReason::UnsupportedDomain, 1, false, -1);
  check("outside-n", M, N - 1, K, &s8, Workspace, S8Bytes,
        selector::DecisionReason::UnsupportedDomain, 1, false, -1);
  check("outside-k", M, N, K - 1, &s8, Workspace, S8Bytes,
        selector::DecisionReason::UnsupportedDomain, 1, false, -1);

  bool const type_ok =
      std::is_same_v<typename production::TypeContract::S1Gemm,
                     typename production::ProductionShipping::Gemm> &&
      std::is_same_v<
          typename production::ProductionShipping::CollectiveMainloop,
          typename production::ProductionSplit::CollectiveMainloop>;
  std::printf(
      "[l200:type] s1_exact=%d same_mainloop=%d mode=ScaleOnly gs=128 "
      "artifact=xplane-tk64 tactic=8x64x128/8x16/s2 profile_bytes=%zu -> %s\n",
      int(std::is_same_v<typename production::TypeContract::S1Gemm,
                         typename production::ProductionShipping::Gemm>),
      int(std::is_same_v<
          typename production::ProductionShipping::CollectiveMainloop,
          typename production::ProductionSplit::CollectiveMainloop>),
      sizeof(quactlize_ppu_dense_w4_splitk_profile_v1),
      type_ok ? "PASS" : "FAIL");
  errors += !type_ok;

  std::printf(
      "[l200] %s controls=%d shipping_calls=%d parallel_calls=%d "
      "full_key_fields=22 profile_axis={1,2,4,8}\n",
      errors == 0 ? "PASS" : "FAIL", controls, shipping_calls,
      parallel_calls);
  return errors == 0 ? 0 : 1;
}
