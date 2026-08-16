// L197: bind the fail-closed profile selector to the real W4 shipping type.
#include <cstdio>
#include <type_traits>

#define PPU_B_CHUNK 0
#include "dense_splitk_parallel_ppu.cuh"
#include "ppu_dense_splitk_shipping_policy.hpp"
#include "ppu_group_schedule.hpp"

namespace selector = ppu_dense_splitk_shipping;

using namespace cute;
using Schedule = ppu_group_schedule::FinegrainedSchedule<128>;
using TileShape = Shape<_8, _64, _64>;
using ScaleTile = Shape<_64, _1>;
using Warp = Shape<_8, _16, _64>;
using Shipping = fpa_intb_ppu::DensePackedAKernelTypes<
    1, fpa_intb_ppu::QuantMode::FinegrainedScaleOnly, Schedule,
    TileShape, ScaleTile, Warp, 2, true, cutlass::int4b_t, 64>;
using Split = dense_splitk_parallel_ppu::KernelTypes<Shipping, TileShape, Warp>;
using Prepared = dense_splitk_parallel_ppu::PreparedOnePlaneLauncher<
    Shipping, TileShape, Warp>;
using Contract = selector::DispatchTypeContract<Shipping, Split>;

static_assert(std::is_same_v<typename Contract::S1Gemm,
                             typename Shipping::Gemm>);
static_assert(std::is_same_v<typename Contract::S1Kernel,
                             typename Shipping::GemmKernel>);
static_assert(std::is_same_v<typename Prepared::ShippingGemm,
                             typename Shipping::Gemm>);
static_assert(std::is_same_v<typename Contract::ParallelGemm,
                             typename Split::Gemm>);
static_assert(std::is_same_v<typename Shipping::CollectiveMainloop,
                             typename Split::CollectiveMainloop>);
static_assert(Shipping::MainloopPolicy::Descriptor::quant_mode ==
              fpa_intb_ppu::QuantMode::FinegrainedScaleOnly);
static_assert(Shipping::MainloopPolicy::LowBits == 4 &&
              Shipping::MainloopPolicy::HighBits == 0 &&
              Shipping::MainloopPolicy::TacticTileK == 64 &&
              Shipping::MainloopPolicy::ArtifactTileK == 64 &&
              Shipping::MainloopPolicy::ArtifactLowFold == 1 &&
              Shipping::MainloopPolicy::ArtifactHighFold == 1 &&
              Shipping::MainloopPolicy::PackedARows == 1);
static_assert(
    Shipping::CollectiveMainloop::DispatchPolicy::StaticGroupSize == 128);
static_assert(std::is_same_v<typename Shipping::ElementAccumulator, float> &&
              std::is_same_v<typename Shipping::ElementD, cutlass::half_t>);

constexpr selector::Key kMeasuredKey{
    {1, 4096, 4096},
    {4, 0, 128, selector::QuantSemantics::FinegrainedScaleOnly,
     selector::MetadataStorage::Fp16Planes, false},
    {selector::ArtifactLayout::ResidentXPlane, 64, 1, 1, 0},
    {8, 64, 64, 8, 16, 2, 1, true},
};
constexpr std::size_t kS8Bytes = 1u * 4096u * 8u * sizeof(float);
constexpr selector::Request kMeasuredRequest{kMeasuredKey, 0x1000, kS8Bytes};
constexpr selector::ProfileRow kS8Profile{
    selector::kProfileSchemaVersion, kMeasuredKey, 8};

static_assert(selector::is_proven_w4_domain(kMeasuredKey));
static_assert(selector::required_partial_bytes(kMeasuredKey.problem, 8) ==
              kS8Bytes);
static_assert(selector::select(kMeasuredRequest, &kS8Profile)
                  .parallel_selected());
static_assert(selector::select(kMeasuredRequest, &kS8Profile)
                  .split_k_slices() == 8);
static_assert(selector::select(kMeasuredRequest).split_k_slices() == 1);

// The production integration edge.  This is intentionally a real, compiled
// call to PreparedOnePlaneLauncher rather than a second mock selector.  Its S1
// lambda supplies the literal historical split count, which enters
// Prepared::ShippingGemm; its S>1 lambda forwards only select()'s admitted
// profile count, which enters Prepared::SplitGemm + the permanent reducer.
// The host oracle below exercises the same dispatch_selected control edge with
// observable call counters; device pointers are not dereferenced locally.
template <class SourceBindingOnly = void>
bool prepare_selected_production(
    selector::Selection selection, Prepared& prepared,
    cutlass::half_t const* a, cutlass::int4b_t const* b,
    cutlass::half_t const* scales, cutlass::half_t* d,
    int m, int n, int k, char* workspace, std::size_t workspace_bytes,
    hggcStream_t stream) {
#if defined(L197_SEVER_PRODUCTION_EDGE)
  (void)selection;
  (void)prepared;
  (void)a;
  (void)b;
  (void)scales;
  (void)d;
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
            a, b, scales, nullptr, d, m, n, k, 128, 1,
            workspace, workspace_bytes, stream);
      },
      [&](int selected_s) {
        return prepared.initialize(
            a, b, scales, nullptr, d, m, n, k, 128, selected_s,
            workspace, workspace_bytes, stream);
      });
#endif
}

// Taking this address instantiates the concrete function-template body.  The
// runner enables it against a temporary header overlay carrying a dependent
// marker inside PreparedOnePlaneLauncher::initialize.  A route-severed control
// must compile cleanly against the same overlay before the executable is
// certified with L197_PRODUCTION_EDGE_WITNESSED=1.
#if defined(L197_FORCE_PRODUCTION_EDGE)
[[maybe_unused]] auto const kProductionEdgeInstantiation =
    &prepare_selected_production<>;
#endif

#if !defined(L197_PRODUCTION_EDGE_WITNESSED)
#define L197_PRODUCTION_EDGE_WITNESSED 0
#endif
static_assert(L197_PRODUCTION_EDGE_WITNESSED == 0 ||
                  L197_PRODUCTION_EDGE_WITNESSED == 1,
              "L197 production-edge witness must be boolean");

int main() {
  int errors = 0;
  int controls = 0;
  int shipping_calls = 0;
  int parallel_calls = 0;

  auto check = [&](char const* label, selector::Request const& request,
                   selector::ProfileRow const* profile,
                   selector::DecisionReason expected_reason,
                   int expected_s, bool expected_parallel) {
    ++controls;
    selector::Selection const selected = selector::select(request, profile);
    int const shipping_before = shipping_calls;
    int const parallel_before = parallel_calls;
    int const result = selector::dispatch_selected(
        selected,
        [&]() {
          ++shipping_calls;
          return 101;
        },
        [&](int splits) {
          ++parallel_calls;
          return 200 + splits;
        });
    bool const routed_parallel = parallel_calls == parallel_before + 1;
    bool const routed_shipping = shipping_calls == shipping_before + 1;
    bool const ok = selected.reason() == expected_reason &&
        selected.split_k_slices() == expected_s &&
        selected.parallel_selected() == expected_parallel &&
        routed_parallel == expected_parallel &&
        routed_shipping == !expected_parallel &&
        result == (expected_parallel ? 200 + expected_s : 101);
    std::printf(
        "[l197:case] name=%s selected_s=%d reason=%s route=%s -> %s\n",
        label, selected.split_k_slices(), selector::reason_name(selected.reason()),
        routed_parallel ? "fixed-splitk" : "shipping-s1",
        ok ? "PASS" : "FAIL");
    errors += !ok;
  };

  selector::ProfileRow profile = kS8Profile;
  for (int splits : {2, 4, 8}) {
    profile = kS8Profile;
    profile.selected_s = splits;
    check(splits == 2 ? "profile-s2" :
          splits == 4 ? "profile-s4" : "profile-s8",
          kMeasuredRequest, &profile,
          selector::DecisionReason::ProfileSelectsParallel, splits, true);
  }

  profile = kS8Profile;
  profile.selected_s = 1;
  selector::Request no_workspace = kMeasuredRequest;
  no_workspace.workspace_address = 0;
  no_workspace.workspace_bytes = 0;
  check("profile-s1-exact-path", no_workspace, &profile,
        selector::DecisionReason::ProfileSelectsS1, 1, false);
  check("absent-profile", kMeasuredRequest, nullptr,
        selector::DecisionReason::NoProfile, 1, false);

  profile = kS8Profile;
  profile.schema_version = selector::kProfileSchemaVersion + 1;
  check("stale-schema", kMeasuredRequest, &profile,
        selector::DecisionReason::StaleProfileSchema, 1, false);

  profile = kS8Profile;
  profile.key.format.group_size = 32;
  check("stale-semantic-key", kMeasuredRequest, &profile,
        selector::DecisionReason::StaleProfileKey, 1, false);

  profile = kS8Profile;
  profile.selected_s = 3;
  check("invalid-s3", kMeasuredRequest, &profile,
        selector::DecisionReason::InvalidProfileSplit, 1, false);

  auto domain_control = [&](char const* label, selector::Request request) {
    selector::ProfileRow matching = kS8Profile;
    matching.key = request.key;
    check(label, request, &matching,
          selector::DecisionReason::UnsupportedDomain, 1, false);
  };

  selector::Request request = kMeasuredRequest;
  request.key.problem.rows = 2;
  domain_control("m2", request);
  request = kMeasuredRequest;
  request.key.format.quant = selector::QuantSemantics::FinegrainedScaleZero;
  domain_control("scale-zero", request);
  request = kMeasuredRequest;
  request.key.format.group_size = 32;
  domain_control("gguf-q4-gs32", request);
  request = kMeasuredRequest;
  request.key.format.low_bits = 2;
  domain_control("w2", request);
  request = kMeasuredRequest;
  request.key.format.high_bits = 1;
  domain_control("two-plane", request);
  request = kMeasuredRequest;
  request.key.format.metadata = selector::MetadataStorage::PackedUnits;
  domain_control("fully-quantized-metadata", request);
  request = kMeasuredRequest;
  request.key.format.has_zero_plane = true;
  domain_control("zero-plane-present", request);
  request = kMeasuredRequest;
  request.key.artifact.tile_k = 128;
  domain_control("artifact-tk128", request);
  request = kMeasuredRequest;
  request.key.artifact.low_fold = 2;
  domain_control("folded-artifact", request);
  request = kMeasuredRequest;
  request.key.artifact.b_chunk = 1;
  domain_control("bchunk1", request);
  request = kMeasuredRequest;
  request.key.tactic.packed_a_rows = 0;
  domain_control("ordinary-a", request);
  request = kMeasuredRequest;
  request.key.tactic.tile_m = 16;
  domain_control("non-m8-tactic", request);
  request = kMeasuredRequest;
  request.key.tactic.aiu_interleaved = false;
  domain_control("non-xplane-consumer", request);

  request = kMeasuredRequest;
  request.workspace_bytes = kS8Bytes - 1;
  check("short-workspace", request, &kS8Profile,
        selector::DecisionReason::InsufficientWorkspace, 1, false);
  request = kMeasuredRequest;
  request.workspace_address += 16;
  check("weak-workspace-alignment", request, &kS8Profile,
        selector::DecisionReason::InsufficientWorkspace, 1, false);

  request = kMeasuredRequest;
  request.key.tactic.tile_k = 256;
  request.key.tactic.stages = 12;
  profile = kS8Profile;
  profile.key = request.key;
  check("shallow-s8-pipeline", request, &profile,
        selector::DecisionReason::InadmissiblePartition, 1, false);

  bool const type_ok =
      std::is_same_v<typename Contract::S1Gemm, typename Shipping::Gemm> &&
      std::is_same_v<typename Prepared::ShippingGemm,
                     typename Shipping::Gemm> &&
      std::is_same_v<typename Shipping::CollectiveMainloop,
                     typename Split::CollectiveMainloop>;
  // This certificate is supplied only after the runner's compiler marker
  // proves that &prepare_selected_production<> reached the concrete
  // PreparedOnePlaneLauncher::initialize body, and its severed twin did not.
  constexpr bool production_edge_ok =
      L197_PRODUCTION_EDGE_WITNESSED == 1;
  std::printf(
      "[l197:type] s1_exact=%d same_mainloop=%d mode=ScaleOnly "
      "bits=4+0 gs=128 artifact=xplane-tk64 packed_a_rows=1 bc=0 "
      "production_edge=%d -> %s\n",
      int(std::is_same_v<typename Contract::S1Gemm,
                         typename Shipping::Gemm>),
      int(std::is_same_v<typename Shipping::CollectiveMainloop,
                         typename Split::CollectiveMainloop>),
      int(production_edge_ok),
      type_ok && production_edge_ok ? "PASS" : "FAIL");
  errors += !(type_ok && production_edge_ok);

  std::printf(
      "[l197] %s controls=%d shipping_calls=%d parallel_calls=%d "
      "default=shipping-s1 profile_axis={1,2,4,8}\n",
      errors == 0 ? "PASS" : "FAIL", controls, shipping_calls,
      parallel_calls);
  return errors == 0 ? 0 : 1;
}
