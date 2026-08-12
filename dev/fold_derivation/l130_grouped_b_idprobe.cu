// L130 -- B-side companion to #112/G5's zero-plane IDPROBE.
//
// The device probe made B numerically inert (q == 8), so L125 could only
// constrain metadata addressing.  Here zero=0 and scale=1, while expert e has
// exactly e unit int4 deviations (q=9; every other code is q=8) in every
// column.  With A=1 and K=256 the exact output is therefore e.  No tolerance
// and no floating-point oracle is involved.
//
// The host arm exhausts e=0..255 through the resident B byte map.  Its model
// is not self-anchored:
//   (1) place_derived must be byte-identical to the deleted, gate-only legacy
//       five-step packer; and
//   (2) place_derived -> recover_derived must reproduce every canonical code.
// xplane::plane_map is the production AIU/swzl/converter composition; the
// type-only arm below proves that its geometry is exactly the G5 collective.

#if defined(L130_TYPE_ONLY)

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "m8n16_g5_contract.hpp"

using Shipping = m8n16_g5_contract::M8;
using Mainloop = typename Shipping::Mainloop;
using Descriptor = typename Shipping::Policy::Descriptor;
using ExpectedOperand = m8n16_g5_contract::ExpectedBOperand;

#ifndef L130_SELECTED_WN
#define L130_SELECTED_WN 32
#endif
using SelectedWarp = cute::Shape<cute::_8, cute::Int<L130_SELECTED_WN>, cute::_64>;
using SelectedPolicy = ppu_mixed_policy::MainloopPolicy<
    Shipping::QuantMode, typename Shipping::BaseSchedule,
    typename Shipping::Tile, typename Shipping::Scale, SelectedWarp,
    Shipping::Stages, Shipping::AiuInterleaved, typename Shipping::ElementB>;

static_assert(std::is_same_v<typename SelectedPolicy::CollectiveOp, Mainloop>,
              "L130 selected policy is not the shipping G5 B type");
static_assert(Descriptor::quant_mode == ppu_mixed_policy::QuantMode::FinegrainedScaleZero);
static_assert(Descriptor::tactic_tile_k == 64 && Descriptor::artifact_tile_k == 64 &&
              Descriptor::artifact_low_fold == 1 && Descriptor::artifact_high_fold == 1 &&
              Descriptor::stages == 3 && !Descriptor::interleaved);
static_assert(std::is_same_v<typename Descriptor::BProviderType,
                             ppu_mixed_policy::OrdinaryBProvider>);
static_assert(std::is_same_v<typename Mainloop::DispatchPolicy::kContinous, cute::_1>,
              "G5 must take load_init_B's dB-backed non-interleaved arm");
static_assert(std::is_same_v<typename Mainloop::ElementB, cutlass::int4b_t>);
static_assert(std::is_same_v<typename Mainloop::GmemTiledCopyB,
                             typename ExpectedOperand::GmemTiledCopy>);
static_assert(std::is_same_v<typename Mainloop::SmemCopyAtomB,
                             typename ExpectedOperand::SmemCopyAtom>);
static_assert(std::is_same_v<typename Mainloop::StrideB,
                             cute::Stride<int64_t, cute::_1, int64_t>>);
static_assert(int(cute::size(typename Mainloop::TiledMma{})) == 32);
static_assert(m8n16_g5_contract::kN == 32 && m8n16_g5_contract::kK == 256 &&
              m8n16_g5_contract::kTacticK == 64 &&
              m8n16_g5_contract::kExperts == 256 &&
              m8n16_g5_contract::kGroupSize == 32);

int main() {
  // Replay the exact pointer construction used by load_init_B's ordinary
  // (kContinuous == 1) arm.  dB is expressed in logical int4 elements, but
  // make_gmem_ptr(typed_pointer) selects the generic Iterator overload and
  // retains raw C++ pointer arithmetic.  This is the layer L130's original
  // byte-map model skipped.
  alignas(16) std::uint8_t storage[3 * m8n16_g5_contract::kN *
                                    m8n16_g5_contract::kK]{};
  auto const* typed = reinterpret_cast<cutlass::int4b_t const*>(storage);
  auto const shape = cute::make_shape(
      m8n16_g5_contract::kN, m8n16_g5_contract::kK,
      m8n16_g5_contract::kExperts);
  typename Mainloop::StrideB const dB{
      int64_t(m8n16_g5_contract::kK), cute::_1{},
      int64_t(m8n16_g5_contract::kN * m8n16_g5_contract::kK)};

  auto const raw_nkl = cute::make_tensor(cute::make_gmem_ptr(typed), shape, dB);
  auto const raw_e1 = raw_nkl(cute::_, cute::_, 1);
  auto const sub_nkl = cute::make_tensor(
      cute::make_gmem_ptr<cutlass::int4b_t>(
          static_cast<void const*>(typed)),
      shape, dB);
  auto const sub_e1 = sub_nkl(cute::_, cute::_, 1);
  auto const raw_delta = reinterpret_cast<std::uintptr_t>(
      cute::raw_pointer_cast(raw_e1.data())) -
      reinterpret_cast<std::uintptr_t>(storage);
  auto const subbyte_delta = reinterpret_cast<std::uintptr_t>(
      cute::raw_pointer_cast(sub_e1.data())) -
      reinterpret_cast<std::uintptr_t>(storage);
  constexpr std::uintptr_t kLogicalCodes =
      m8n16_g5_contract::kN * m8n16_g5_contract::kK;
  constexpr std::uintptr_t kPhysicalBytes = kLogicalCodes / 2;
  bool const pointer_replay = raw_delta == kLogicalCodes &&
                              subbyte_delta == kPhysicalBytes;

  std::puts("[l130:type] exact G5 B: FinegrainedScaleZero gs32 "
            "tile=8x32x64 warp=8x32x64 stages=3 int4 CTA32; "
            "ordinary dB-backed kContinuous=1; dB2/interleaved=NOT-SELECTED PASS");
  std::printf("[l130:production-slice] typed-int4 generic make_gmem_ptr "
              "expert-delta=%zu B (observed bug); explicitly subbyte-aware "
              "delta=%zu B (artifact=%zu B) -> %s\n",
              std::size_t(raw_delta), std::size_t(subbyte_delta),
              std::size_t(kPhysicalBytes), pointer_replay ? "EXPECTED-RED" : "FAIL");
  return pointer_replay ? 0 : 1;
}

#else

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "legacy_pipeline.hpp"
#include "m8n16_g5_layout_spec.hpp"
#include "xplane_offline.hpp"

namespace spec = m8n16_g5_layout_spec;

namespace {

constexpr int kBits = 4;
constexpr int kCodesPerByte = 2;
constexpr int kArtifactBytes = spec::kN * spec::kK * kBits / 8;
constexpr int kCodesPerExpert = spec::kN * spec::kK;
constexpr int kKTiles = spec::kK / spec::kTacticK;

static_assert(spec::kExperts == 256 && spec::kN == 32 && spec::kK == 256 &&
              spec::kTacticK == 64 && spec::kStoredRowK == 256 &&
              !spec::kAiuInterleaved);
static_assert(kArtifactBytes == 4096 && kKTiles == 4);

std::vector<std::uint8_t> probe_codes(int expert) {
  std::vector<std::uint8_t> q(kCodesPerExpert, std::uint8_t(8));
  // Each output column sees exactly `expert` copies of (q-8)==1.
  for (int n = 0; n < spec::kN; ++n)
    for (int k = 0; k < expert; ++k)
      q[std::size_t(k) * spec::kN + n] = std::uint8_t(9);
  return q;
}

std::vector<std::int8_t> legacy_pack(
    std::vector<std::uint8_t> const& stored_q, int N, int K) {
  // The legacy entry consumes signed int4 and adds the +8 bias itself.  Feed
  // q-8, packed [N][K], so its output is directly comparable to place_derived.
  std::vector<std::int8_t> nk(std::size_t(N) * K / 2, 0), out(nk.size(), 0);
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      int const signed_q = int(stored_q[std::size_t(k) * N + n]) - 8;
      std::size_t const i = std::size_t(n) * K + k;
      nk[i / kCodesPerByte] |= std::int8_t((signed_q & 15) << (4 * (i % 2)));
    }
  }
  legacy::preprocess_weights_for_mixed_gemm<false, 256, 0>(
      out.data(), nk.data(), {std::size_t(K), std::size_t(N)},
      QuantTypeClass::PACKED_INT4_WEIGHT_ONLY);
  return out;
}

int exact_output(std::vector<std::uint8_t> const& recovered, int n) {
  int sum = 0;
  for (int k = 0; k < spec::kK; ++k)
    sum += int(recovered[std::size_t(k) * spec::kN + n]) - 8;
  return sum;
}

struct Census {
  long long legacy_byte_diff = 0;
  long long roundtrip_code_diff = 0;
  long long address_bad = 0;
  long long value_bad = 0;
  long long output_bad = 0;
  long long map_holes = 0;
  long long map_dups = 0;
};

int sum_dequantized_column(std::vector<std::uint8_t> const& q, int n,
                           int scale_denominator, int zero_sum) {
  int qsum = 0;
  for (int k = 0; k < spec::kK; ++k)
    qsum += int(q[std::size_t(k) * spec::kN + n]) - 8;
  return qsum / scale_denominator + zero_sum;
}

}  // namespace

int main() {
  auto const tile_map = xplane::plane_map<4, 8, 32, 64, 8, 32, 1>();
  std::vector<int> map_hits(spec::kN * spec::kTacticK, 0);
  for (int logical : tile_map) {
    if (logical >= 0 && logical < int(map_hits.size())) ++map_hits[logical];
  }

  Census total;
  for (int hits : map_hits) {
    total.map_holes += hits == 0;
    total.map_dups += hits > 1 ? hits - 1 : 0;
  }

  std::vector<std::int8_t> all(std::size_t(spec::kExperts) * kArtifactBytes);
  std::vector<std::vector<std::uint8_t>> recovered(spec::kExperts);
  int min_source = int(all.size()), max_source = -1;

  // The deleted packer's PPU interleave primitive requires both matrix byte
  // extents to be 32-B multiples; G5's N=32 int4 plane is only 16 B wide.
  // Calibrate the exact same TN32/TK64/WN32 byte map on the smallest legal
  // N=64 companion, then keep the G5-sized place/recover anchor below.  This
  // is a constraint of the legacy reference, not a widening of the model.
  constexpr int kLegacyN = 64;
  std::vector<std::uint8_t> anchor_q(std::size_t(spec::kK) * kLegacyN);
  for (int k = 0; k < spec::kK; ++k)
    for (int n = 0; n < kLegacyN; ++n)
      anchor_q[std::size_t(k) * kLegacyN + n] =
          std::uint8_t((((11 * k + 7 * n + (k ^ n)) & 15) + 8) & 15);
  std::vector<std::int8_t> anchor_placed(std::size_t(kLegacyN) * spec::kK / 2);
  xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
      anchor_placed.data(), anchor_q, kLegacyN, spec::kK);
  auto const anchor_legacy = legacy_pack(anchor_q, kLegacyN, spec::kK);
  for (std::size_t i = 0; i < anchor_placed.size(); ++i)
    total.legacy_byte_diff +=
        std::uint8_t(anchor_placed[i]) != std::uint8_t(anchor_legacy[i]);

  for (int e = 0; e < spec::kExperts; ++e) {
    auto const q = probe_codes(e);
    auto* resident = all.data() + std::size_t(e) * kArtifactBytes;
    xplane::place_derived<4, 8, 32, 64, 8, 32, 1>(
        resident, q, spec::kN, spec::kK);

    xplane::recover_derived<4, 8, 32, 64, 8, 32, 1>(
        resident, recovered[e], spec::kN, spec::kK);
    for (int i = 0; i < kCodesPerExpert; ++i)
      total.roundtrip_code_diff += recovered[e][i] != q[i];

    std::vector<int> physical_hits(kCodesPerExpert, 0);
    for (int kt = 0; kt < kKTiles; ++kt) {
      for (int p = 0; p < int(tile_map.size()); ++p) {
        int const logical = tile_map[p];
        if (logical < 0 || logical >= spec::kN * spec::kTacticK) {
          ++total.address_bad;
          continue;
        }
        int const row = p / (spec::kTacticK);
        int const within_run = p % spec::kTacticK;
        // F=1/interleave-256 resident address: each physical N row owns a
        // 256-code run and the four TK64 artifacts occupy consecutive runs.
        int const resident_code = row * spec::kStoredRowK +
                                  kt * spec::kTacticK + within_run;
        int const global_code = e * kCodesPerExpert + resident_code;
        int const global_byte = global_code / kCodesPerByte;
        min_source = std::min(min_source, global_byte);
        max_source = std::max(max_source, global_byte);
        total.address_bad += resident_code < 0 || resident_code >= kCodesPerExpert;
        total.address_bad += global_byte < e * kArtifactBytes ||
                             global_byte >= (e + 1) * kArtifactBytes;
        if (resident_code < 0 || resident_code >= kCodesPerExpert) continue;
        ++physical_hits[resident_code];

        int const n = logical / spec::kTacticK;
        int const k = kt * spec::kTacticK + logical % spec::kTacticK;
        std::uint8_t const byte = std::uint8_t(all[global_byte]);
        std::uint8_t const got = (byte >> (4 * (global_code & 1))) & 15;
        total.value_bad += got != q[std::size_t(k) * spec::kN + n];
      }
    }
    for (int hits : physical_hits) {
      total.address_bad += hits != 1;
    }

    int expert_output_bad = 0;
    for (int n = 0; n < spec::kN; ++n)
      expert_output_bad += exact_output(recovered[e], n) != e;
    total.output_bad += expert_output_bad;
    std::printf("[l130:e] e=%3d scheduler=%3d dB-code-base=%7d "
                "resident-bytes=[%7d,%7d] output=%3d columns=%s\n",
                e, e, e * kCodesPerExpert, e * kArtifactBytes,
                (e + 1) * kArtifactBytes - 1, exact_output(recovered[e], 0),
                expert_output_bad ? "BAD" : "32/32");
  }

  // Reproduce the retained device observation with the missing production
  // assumption made explicit.  The raw typed-int4 L slice advances 8192
  // BYTES per expert, twice the 4096-byte artifact pitch.  Therefore the B
  // probe reads payload(2e) below the allocation midpoint.  At e=128 the
  // same bad slice starts one-past the 1 MiB B allocation.  In the observed
  // allocation sequence the next 128 KiB is the scale plane: sixteen bad
  // 8192-byte strides, exactly e=128..143.  Its repeated fp16(1/32) bytes
  // 00 28, interpreted as packed int4, contribute -44.  A later zero-filled
  // workspace contributes -64.  These are OOB contents, not two metadata
  // remaps and not a dequantization gain of two.
  int b_replay_bad = 0;
  for (int e : {1, 3}) {
    int const got = exact_output(recovered[2 * e], 0);
    b_replay_bad += got != 2 * e;
    std::printf("[l130:observed-B] e=%d raw-byte-pitch source-expert=%d "
                "got=%d want=%d -> %s\n",
                e, 2 * e, got, 2 * e, got == 2 * e ? "REPRODUCED" : "FAIL");
  }

  std::vector<std::int8_t> scale_bytes(kArtifactBytes);
  for (int i = 0; i < kArtifactBytes; i += 2) {
    scale_bytes[i] = std::int8_t(0x00);
    scale_bytes[i + 1] = std::int8_t(0x28);  // fp16(1/32), little-endian
  }
  std::vector<std::uint8_t> scale_as_q, zero_as_q;
  xplane::recover_derived<4, 8, 32, 64, 8, 32, 1>(
      scale_bytes.data(), scale_as_q, spec::kN, spec::kK);
  std::vector<std::int8_t> zero_bytes(kArtifactBytes, 0);
  xplane::recover_derived<4, 8, 32, 64, 8, 32, 1>(
      zero_bytes.data(), zero_as_q, spec::kN, spec::kK);

  int zero_replay_bad = 0;
  for (auto const target : {std::pair<int, int>{128, 84},
                            std::pair<int, int>{129, 85},
                            std::pair<int, int>{143, 99}}) {
    int const got = sum_dequantized_column(scale_as_q, 0, 32, target.first);
    bool all_columns = true;
    for (int n = 1; n < spec::kN; ++n)
      all_columns &= sum_dequantized_column(scale_as_q, n, 32, target.first) == got;
    zero_replay_bad += got != target.second || !all_columns;
    std::printf("[l130:observed-zero] e=%d OOB-source=fp16(1/32)-plane "
                "got=%d want=%d columns=%s -> %s\n",
                target.first, got, target.second, all_columns ? "uniform" : "SPLIT",
                got == target.second && all_columns ? "REPRODUCED" : "FAIL");
  }
  for (auto const target : {std::pair<int, int>{190, 126},
                            std::pair<int, int>{201, 137},
                            std::pair<int, int>{255, 191}}) {
    int const got = sum_dequantized_column(zero_as_q, 0, 32, target.first);
    bool all_columns = true;
    for (int n = 1; n < spec::kN; ++n)
      all_columns &= sum_dequantized_column(zero_as_q, n, 32, target.first) == got;
    zero_replay_bad += got != target.second || !all_columns;
    std::printf("[l130:observed-zero] e=%d OOB-source=zero-filled-region "
                "got=%d want=%d columns=%s -> %s\n",
                target.first, got, target.second, all_columns ? "uniform" : "SPLIT",
                got == target.second && all_columns ? "REPRODUCED" : "FAIL");
  }

  // Required red control: replay the exact arithmetic with the historical
  // high-half remap.  It must reject exactly experts 128..255, and their
  // observed ID must be exactly e-64 in every column.
  int red_mismatched_experts = 0;
  int red_low_bad = 0;
  int red_high_inexact = 0;
  for (int e = 0; e < spec::kExperts; ++e) {
    int const planted = e < 128 ? e : e - 64;
    int const got = exact_output(recovered[planted], 0);
    bool columns_same = true;
    for (int n = 1; n < spec::kN; ++n)
      columns_same &= exact_output(recovered[planted], n) == got;
    bool const mismatch = !columns_same || got != e;
    red_mismatched_experts += mismatch;
    red_low_bad += e < 128 && mismatch;
    red_high_inexact += e >= 128 && (!mismatch || !columns_same || got != e - 64);
  }

  // Two small controls prove the anchors have load-bearing comparisons.
  auto corrupt = anchor_placed;
  corrupt[0] ^= 1;
  int corrupt_legacy_diff = 0;
  for (std::size_t i = 0; i < corrupt.size(); ++i)
    corrupt_legacy_diff +=
        std::uint8_t(corrupt[i]) != std::uint8_t(anchor_legacy[i]);
  std::vector<std::uint8_t> corrupt_recovered;
  xplane::recover_derived<4, 8, 32, 64, 8, 32, 1>(
      corrupt.data(), corrupt_recovered, kLegacyN, spec::kK);
  int corrupt_roundtrip_diff = 0;
  for (std::size_t i = 0; i < anchor_q.size(); ++i)
    corrupt_roundtrip_diff += corrupt_recovered[i] != anchor_q[i];

  bool const positive = total.legacy_byte_diff == 0 &&
      total.roundtrip_code_diff == 0 && total.address_bad == 0 &&
      total.value_bad == 0 && total.output_bad == 0 &&
      total.map_holes == 0 && total.map_dups == 0 &&
      min_source == 0 && max_source == int(all.size()) - 1;
  bool const red = red_mismatched_experts == 128 && red_low_bad == 0 &&
                   red_high_inexact == 0;
  bool const controls = corrupt_legacy_diff == 1 && corrupt_roundtrip_diff == 1;
  bool const observed_replay = b_replay_bad == 0 && zero_replay_bad == 0;

  std::printf("[l130] e0..255 legacy-byte-diff=%lld place/recover-code-diff=%lld "
              "map-holes=%lld map-dups=%lld address-bad=%lld value-bad=%lld "
              "output-bad=%lld source=[%d,%d] -> %s\n",
              total.legacy_byte_diff, total.roundtrip_code_diff,
              total.map_holes, total.map_dups, total.address_bad,
              total.value_bad, total.output_bad, min_source, max_source,
              positive ? "PASS" : "FAIL");
  std::printf("[l130:anchor] legacy-five-step byte identity at legal N64 "
              "companion + G5-N32 place/recover identity; planted-byte "
              "controls legacy=%d "
              "roundtrip=%d expected=1/1 -> %s\n",
              corrupt_legacy_diff, corrupt_roundtrip_diff,
              controls ? "PASS" : "FAIL");
  std::printf("[l130:red] e>=128->e-64 mismatched-experts=%d low-bad=%d "
              "high-inexact=%d expected=128/0/0 -> %s\n",
              red_mismatched_experts, red_low_bad, red_high_inexact,
              red ? "EXPECTED-RED" : "FAIL");
  std::printf("[l130:scope] B-low-plane-only; G5 selects ordinary dB. "
              "interleaved dB and dB2 are NOT SELECTED by this kernel type; "
              "zero/scale addressing is covered by L125, not inferred here. result=%s\n",
              positive && red && controls && observed_replay ? "PASS" : "FAIL");
  std::printf("[l130:observed-replay] B-bad=%d zero-bad=%d -> %s; "
              "the e>=128 values are consequences of the observed OOB "
              "allocation contents, not a stable expert mapping\n",
              b_replay_bad, zero_replay_bad,
              observed_replay ? "PASS" : "FAIL");
  return positive && red && controls && observed_replay ? 0 : 1;
}

#endif
