#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "ppu_placed_arrangement.hpp"

int main() {
  using A = quactlize_ppu_placed_arrangement_v1;
  static_assert(std::is_standard_layout_v<A> && std::is_trivially_copyable_v<A>);
  static_assert(sizeof(A) == 16 && alignof(A) == alignof(int32_t));
  static_assert(offsetof(A, version) == 0 && offsetof(A, bits) == 4 &&
                offsetof(A, artifact_tile_k) == 8 && offsetof(A, high_bits) == 12);

  using C = quactlize_ppu_config_v3;
  static_assert(std::is_standard_layout_v<C> && std::is_trivially_copyable_v<C>);
  static_assert(sizeof(void*) != 8 || (sizeof(C) == 48 && alignof(C) == 8),
                "config_v3 LP64 inventory ABI changed");
  static_assert(offsetof(C, enable_cuda_kernel) == 0 && offsetof(C, name) == 8 &&
                offsetof(C, tile_m) == 16 && offsetof(C, tile_n) == 20 &&
                offsetof(C, tactic_tile_k) == 24 && offsetof(C, artifact_tile_k) == 28 &&
                offsetof(C, warp_m) == 32 && offsetof(C, warp_n) == 36 &&
                offsetof(C, stages) == 40,
                "config_v3 fields or padding changed; bump the public ABI rather than drifting it");

  constexpr auto legacy_q2 = ppu_arrangements::legacy_fully_quantized_default(10);
  constexpr auto legacy_q4 = ppu_arrangements::legacy_fully_quantized_default(12);
  static_assert(legacy_q2.artifact_tile_k == 256 && legacy_q4.artifact_tile_k == 256,
                "the unversioned dense FQ ABI historically consumes A256 Q2/Q4 bytes; do not unify it with the "
                "no-tile Python producer's scale-first A128/A64 defaults");

  uint64_t accepted = 0, rejected = 0;
  for (auto const& f : ppu_formats::kConfigs) {
    int const tactic_k = f.fully_quantized_tile_k;
    for (int artifact_k : {32, 64, 128, 256}) {
      A const a{QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1,
                f.low_bits, artifact_k, f.high_bits};
      bool const want = ppu_formats::artifact_tile_k_supported(f, artifact_k) &&
                        artifact_k <= tactic_k && tactic_k % artifact_k == 0;
      bool const got = ppu_arrangements::matches_compiled_tactic(&a, f.qtype, 4096, tactic_k);
      assert(got == want);
      (got ? accepted : rejected)++;

      A bad = a;
      bad.version++;
      assert(!ppu_arrangements::matches_compiled_tactic(&bad, f.qtype, 4096, tactic_k));
      bad = a;
      bad.bits++;
      assert(!ppu_arrangements::matches_compiled_tactic(&bad, f.qtype, 4096, tactic_k));
      bad = a;
      bad.high_bits++;
      assert(!ppu_arrangements::matches_compiled_tactic(&bad, f.qtype, 4096, tactic_k));
      assert(!ppu_arrangements::matches_compiled_tactic(nullptr, f.qtype, 4096, tactic_k));
    }
  }

  A const f2{QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 2, 64, 1};
  assert((ppu_arrangements::packed_tensor_matches_exact_reader<11, 256, 64>(&f2, 4096)));
  assert((!ppu_arrangements::packed_tensor_matches_exact_reader<11, 256, 256>(&f2, 4096)));
  A const unsupported_single_plane{QUACTLIZE_PPU_PLACED_ARRANGEMENT_VERSION_V1, 4, 32, 0};
  assert(!ppu_arrangements::packed_tensor_reader_supported(
      &unsupported_single_plane, 12, 4096, 256));
  std::cout << "L138 dense placed-arrangement ABI accepted=" << accepted
            << " rejected=" << rejected
            << " F2_to_F1=EXPECTED_RED single_plane_F2=FAIL_CLOSED PASS\n";
}
