#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <utility>

#include "benchmarks/gemv_perf_fixture.hpp"

int main() {
  int errors = 0;
  int const token1_ids[] = {7, 11, 35, 77, 127, 128, 218, 224};
  for (int tokens : {1, 2, 4}) {
    auto const r = gemv_perf_fixture::make_route(256, tokens, 8);
    int const want_active = tokens == 1 ? 8 : tokens == 2 ? 15 : 30;
    int const want_max = tokens == 1 ? 1 : tokens == 2 ? 2 : 3;
    errors += !r.valid || r.total_rows != tokens * 8 ||
              int(r.active_ids.size()) != want_active || r.max_rows != want_max;
    for (int e = 0; e < 256; ++e) {
      errors += r.row_offsets[std::size_t(e + 1)] - r.row_offsets[std::size_t(e)] !=
                r.rows_per_expert[std::size_t(e)];
    }
    if (tokens == 1)
      for (int i = 0; i < 8; ++i) errors += r.active_ids[std::size_t(i)] != token1_ids[i];
  }

  auto const r = gemv_perf_fixture::make_route(256, 1, 8);
  std::set<std::uint32_t> active_seeds;
  for (int e : r.active_ids)
    active_seeds.insert(gemv_perf_fixture::plane_seed(e, true, false));
  errors += active_seeds.size() != r.active_ids.size();
  for (int e = 0; e < 256; ++e) {
    bool const active = r.active_slot_for_expert[std::size_t(e)] >= 0;
    if (!active)
      errors += active_seeds.count(gemv_perf_fixture::plane_seed(e, false, false)) != 0;
  }

  // Re-enact the shipped grouped sub-byte pitch bug against the new fixture,
  // not merely against a byte-count formula. The old seam advanced N*K
  // logical int4 codes through a uint8_t pointer, so an in-bounds expert e
  // reads physical expert 2e; e>=128 runs past the plane. Every one of the
  // three device witnesses per active expert must therefore either fail the
  // harness's own 2% numerical tolerance or be provably OOB.
  int old_pitch_wrong_witnesses = 0;
  int old_pitch_total_witnesses = 0;
  constexpr int n4 = 512, k4 = 2048;
  int const witness_columns[] = {0, n4 / 2, n4 - 1};
  for (int e : r.active_ids) {
    for (int n : witness_columns) {
      ++old_pitch_total_witnesses;
      int const source_e = 2 * e;
      if (source_e >= 256) {
        ++old_pitch_wrong_witnesses;
        continue;
      }
      bool const source_active = r.active_slot_for_expert[std::size_t(source_e)] >= 0;
      auto const want_seed = gemv_perf_fixture::plane_seed(e, true, false);
      auto const got_seed = gemv_perf_fixture::plane_seed(source_e, source_active, false);
      float want = 0.0f, got = 0.0f;
      float const a = gemv_perf_fixture::activation_value(e, 0);
      for (int k = 0; k < k4; ++k) {
        int const g = k / 32;
        float const s = gemv_perf_fixture::scale_value(e, g, n, true);
        float const z = gemv_perf_fixture::zero_value(e, g, n, true);
        want += a * (float(gemv_perf_fixture::plane_code(4, n, k, want_seed)) * s + z);
        got += a * (float(gemv_perf_fixture::plane_code(4, n, k, got_seed)) * s + z);
      }
      float const tol = 0.02f * std::max(1.0f, want < 0 ? -want : want);
      float const diff = got > want ? got - want : want - got;
      old_pitch_wrong_witnesses += diff > tol;
    }
  }
  errors += old_pitch_wrong_witnesses != old_pitch_total_witnesses;

  std::uint64_t pitch_checks = 0;
  for (int bits : {1, 2, 4, 6})
    for (auto [n, k] : {std::pair<int, int>{512, 2048}, {512, 3072},
                        {2048, 512}, {3072, 512}}) {
      std::uint64_t const want = std::uint64_t(n) * k * bits / 8u;
      errors += gemv_perf_fixture::packed_plane_bytes(n, k, bits) != want;
      for (int e = 0; e < 256; ++e) {
        errors += gemv_perf_fixture::packed_plane_expert_offset(e, n, k, bits) !=
                  std::uint64_t(e) * want;
        ++pitch_checks;
      }
    }

  std::printf("[l135] routes tokens=1/2/4 active=8/15/30 max=1/2/3 "
              "token1_ids=7,11,35,77,127,128,218,224 pitch_checks=%llu "
              "old_pitch_wrong_witnesses=%d/%d %s\n",
              (unsigned long long)pitch_checks, old_pitch_wrong_witnesses,
              old_pitch_total_witnesses, errors ? "FAIL" : "PASS");
  return errors ? 1 : 0;
}
