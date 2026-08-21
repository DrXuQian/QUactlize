// Host oracle for the shipping MoE persistent block directory.
//
// This is intentionally independent of a PPU runtime.  It exercises the exact
// production ABI and swizzle helpers, then emulates the device grid-stride walk.
// A scheduler mapping is admitted only when every real
// (expert, local-M-tile, N-tile) coordinate appears exactly once.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <tuple>
#include <vector>

#include "actlize_extensions/cutlass/gemm/kernel/ppu_moe_block_directory.hpp"

namespace md = quactlize::moe_directory;
using Key = std::tuple<int, int, int>;

struct Fixture {
  std::vector<int> rows;
  std::vector<int> offsets;
  std::vector<md::BlockEntry> entries;
};

template <int TileM>
Fixture make_fixture(std::vector<int> rows) {
  Fixture out;
  out.rows = std::move(rows);
  out.offsets.resize(out.rows.size());
  int row_prefix = 0;
  int block_prefix = 0;
  for (int e = 0; e < int(out.rows.size()); ++e) {
    out.offsets[e] = row_prefix;
    int const blocks = md::ceil_div_nonnegative(out.rows[e], TileM);
    md::BlockEntry const entry =
        md::make_entry(e, out.rows[e], block_prefix, row_prefix);
    for (int m = 0; m < blocks; ++m) out.entries.push_back(entry);
    block_prefix += blocks;
    row_prefix += out.rows[e];
  }
  return out;
}

template <int TileM>
std::set<Key> expected(Fixture const& fixture, int n_blocks) {
  std::set<Key> out;
  for (int e = 0; e < int(fixture.rows.size()); ++e) {
    int const m_blocks = md::ceil_div_nonnegative(fixture.rows[e], TileM);
    for (int m = 0; m < m_blocks; ++m)
      for (int n = 0; n < n_blocks; ++n) out.emplace(e, m, n);
  }
  return out;
}

template <int TileM, int MBlocksPerGroup = (TileM == 16 ? 1 : 2)>
bool classify(Fixture const& fixture, int n_blocks, int physical_grid,
              bool verbose = false) {
  std::vector<Key> observed;
  uint64_t const total = uint64_t(fixture.entries.size()) * uint64_t(n_blocks);
  for (int cta = 0; cta < physical_grid; ++cta) {
    for (uint64_t iter = 0;; ++iter) {
      uint64_t const linear = iter * uint64_t(physical_grid) + uint64_t(cta);
      if (linear >= total) break;
      int const directory_index = int(linear / uint64_t(n_blocks));
      auto const& entry = fixture.entries.at(size_t(directory_index));
      int const expert_m_blocks =
          md::ceil_div_nonnegative(entry.expert_rows, TileM);
      int const linear_in_expert =
          int(linear) - entry.expert_block_begin * n_blocks;
      if (entry.expert < 0 || entry.expert >= int(fixture.rows.size()) ||
          expert_m_blocks <= 0 || linear_in_expert < 0 ||
          linear_in_expert >= expert_m_blocks * n_blocks) {
        if (verbose) {
          std::printf("invalid entry at linear=%llu directory=%d expert=%d local=%d\n",
                      static_cast<unsigned long long>(linear), directory_index,
                      entry.expert, linear_in_expert);
        }
        return false;
      }
      auto const decoded = md::decode_swizzled(
          linear_in_expert, expert_m_blocks, n_blocks, MBlocksPerGroup);
      if (decoded.m_tile < 0 || decoded.m_tile >= expert_m_blocks ||
          decoded.n_tile < 0 || decoded.n_tile >= n_blocks) {
        if (verbose) std::printf("decoded coordinate outside expert domain\n");
        return false;
      }
      observed.emplace_back(entry.expert, decoded.m_tile, decoded.n_tile);
    }
  }

  auto const want = expected<TileM>(fixture, n_blocks);
  std::set<Key> unique(observed.begin(), observed.end());
  bool const exact = observed.size() == want.size() &&
                     unique.size() == observed.size() && unique == want;
  if (!exact && verbose) {
    std::printf("coverage mismatch observed=%zu unique=%zu expected=%zu\n",
                observed.size(), unique.size(), want.size());
  }
  return exact;
}

template <int TileM>
bool positive_suite() {
  auto const fixture = make_fixture<TileM>(
      {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65});
  for (int n_blocks : {1, 3, 8}) {
    uint64_t const total = uint64_t(fixture.entries.size()) * n_blocks;
    for (int grid : {1, 2, 7, 19, int(total), int(total + 13)}) {
      if (!classify<TileM>(fixture, n_blocks, grid, true)) return false;
    }
  }
  return true;
}

bool negative_suite() {
  auto fixture = make_fixture<8>({1, 17, 0, 33});
  if (!classify<8>(fixture, 3, 5)) return false;

  auto missing = fixture;
  missing.entries.erase(missing.entries.begin() + 1);
  if (classify<8>(missing, 3, 5)) {
    std::printf("negative missing-entry was incorrectly green\n");
    return false;
  }

  auto duplicate = fixture;
  duplicate.entries.insert(duplicate.entries.begin() + 1,
                           duplicate.entries.front());
  if (classify<8>(duplicate, 3, 5)) {
    std::printf("negative duplicate-entry was incorrectly green\n");
    return false;
  }

  auto wrong_prefix = fixture;
  wrong_prefix.entries.back().expert_block_begin -= 1;
  if (classify<8>(wrong_prefix, 3, 5)) {
    std::printf("negative wrong-prefix was incorrectly green\n");
    return false;
  }
  return true;
}

int main() {
  static_assert(sizeof(md::Header) == 16);
  static_assert(sizeof(md::BlockEntry) == 16);
  static_assert(md::workspace_bytes(65, 12, 8) >=
                md::entries_offset() + 12u * 9u * sizeof(md::BlockEntry));
  // Literal order witnesses for DeepGEMM's two-M-block swizzle.  Coverage
  // alone would accept any bijection and therefore could not prove that the
  // intended L2 order survived a refactor.
  static_assert(md::decode_swizzled(0, 3, 3, 2).m_tile == 0 &&
                md::decode_swizzled(0, 3, 3, 2).n_tile == 0);
  static_assert(md::decode_swizzled(1, 3, 3, 2).m_tile == 1 &&
                md::decode_swizzled(1, 3, 3, 2).n_tile == 0);
  static_assert(md::decode_swizzled(5, 3, 3, 2).m_tile == 1 &&
                md::decode_swizzled(5, 3, 3, 2).n_tile == 2);
  static_assert(md::decode_swizzled(6, 3, 3, 2).m_tile == 2 &&
                md::decode_swizzled(6, 3, 3, 2).n_tile == 0);

  bool const positive = positive_suite<8>() && positive_suite<16>() &&
                        positive_suite<32>();
  bool const negative = negative_suite();
  if (!positive || !negative) {
    std::printf("[l215] FAIL positive=%d negative=%d\n", int(positive), int(negative));
    return 1;
  }
  std::printf("[l215] PASS: TileM=8/16/32 ragged+zero+tail coverage exact-once across arbitrary grids; "
              "missing/duplicate/wrong-prefix negatives red\n");
  return 0;
}
