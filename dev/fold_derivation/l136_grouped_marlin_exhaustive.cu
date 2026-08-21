// L136 -- exhaustive composition proof for ragged grouped Marlin.
//
// The manifest names every committed grouped tactic row and every declared
// MoE shape/router input.  Routing uses the same pinned C++ fixture as the
// benchmark.  Prefix construction and q decoding call the exact host/device
// helper used by the production grouped kernel; stripes call the exact Marlin
// core.  Deduplication occurs only after both production lowerings.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "moe_router_fixture.hpp"
#include "cutlass/gemm/kernel/ppu_tile_scheduler_marlin_core.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/ppu_grouped_ragged_geometry.hpp"

using Core = cutlass::gemm::kernel::detail::MarlinStripeSchedulerCore;
using Geometry = cutlass::gemm::kernel::detail::GroupedRaggedOutputTiles;
using Params = Core::Params;
using Work = Core::WorkTileInfo;

struct Key {
  std::vector<int> group_tiles;
  uint64_t nt = 0, kt = 0, cu = 0;
  bool operator<(Key const& x) const {
    return std::tie(group_tiles, nt, kt, cu) <
           std::tie(x.group_tiles, x.nt, x.kt, x.cu);
  }
};

struct Totals {
  uint64_t raw = 0, unique = 0, checked = 0;
  uint64_t groups = 0, zero_groups = 0, outputs = 0;
  uint64_t segments = 0, cells = 0, handoffs = 0;
  uint64_t cross_group = 0, protected_rows = 0, protected_handoffs = 0;
  uint64_t errors = 0;
};

static int value(uint64_t q, uint64_t k) {
  return int((q + k) % 3) - 1;
}

static int range_sum(uint64_t q, uint64_t k, uint64_t count) {
  int out = 0;
  for (uint64_t i = 0; i < count % 3; ++i) out += value(q, k + i);
  return out;
}

static void fail(Totals& t, char const* what, Key const& x,
                 uint64_t got = 0, uint64_t want = 0) {
  if (t.errors < 8) {
    std::fprintf(stderr,
        "[l136] mismatch=%s E=%zu Nt=%llu Kt=%llu CU=%llu got=%llu want=%llu\n",
        what, x.group_tiles.size(), (unsigned long long)x.nt,
        (unsigned long long)x.kt, (unsigned long long)x.cu,
        (unsigned long long)got, (unsigned long long)want);
  }
  ++t.errors;
}

static bool make_prefix(Key const& x, std::vector<int>& prefix, Totals& t) {
  prefix.assign(x.group_tiles.size() + 1, 0);
  uint64_t q = 0;
  for (size_t e = 0; e < x.group_tiles.size(); ++e) {
    // The manifest stores M-tile counts.  Feeding m=mt, TM=1 through the
    // production helper preserves every integer state, including zero rows.
    uint64_t next = 0;
    if (!Geometry::append_group(q, x.group_tiles[e], int(x.nt),
                                1, int(x.nt), next) ||
        next > uint64_t(INT32_MAX)) {
      fail(t, "prefix-overflow", x, next, INT32_MAX);
      return false;
    }
    q = next;
    prefix[e + 1] = int(q);
  }
  return q != 0;
}

struct Peer { uint64_t block; uint32_t count; uint32_t index; };

static void check(Key const& x, uint64_t multiplicity, Totals& t) {
  std::vector<int> prefix;
  if (!make_prefix(x, prefix, t)) return;
  uint64_t const q_count = uint64_t(prefix.back());
  Params p = Core::make_params_for_tiles(q_count, 1, 1, x.kt, x.cu);
  if (!p.valid_ || p.output_tiles_ != q_count || p.tiles_m_ != q_count ||
      p.tiles_n_ != 1 || p.tiles_l_ != 1) {
    fail(t, "synthetic-q-lowering", x, p.output_tiles_, q_count);
    return;
  }
  std::vector<uint8_t> visits(size_t(q_count * x.kt), 0);
  std::vector<int> sums(size_t(q_count), 0);
  std::vector<std::vector<Peer>> peers{size_t(q_count)};

  for (uint64_t block = 0; block < p.grid_blocks_; ++block) {
    Work w = Core::get_work_for_block(p, block);
    uint64_t const stripe_begin = block * p.iters_per_block_;
    uint64_t const stripe_end = std::min(
        stripe_begin + p.iters_per_block_, p.total_k_tiles_);
    uint64_t cursor = stripe_begin;
    int previous_expert = -1;
    while (w.is_valid()) {
      uint64_t const q = w.output_tile_idx;
      if (q >= q_count || uint64_t(w.M_idx) != q || w.N_idx != 0 ||
          w.L_idx != 0 || w.lock_idx != q || w.K_idx >= x.kt ||
          uint64_t(w.K_idx) + w.k_tile_count > x.kt) {
        fail(t, "global-q-work", x, q, q_count);
        return;
      }
      int const expert = Geometry::decode_expert(prefix.data(),
                                                  int(x.group_tiles.size()),
                                                  int(q));
      if (expert < 0 || x.group_tiles[size_t(expert)] == 0 ||
          !(prefix[size_t(expert)] <= int(q) &&
            int(q) < prefix[size_t(expert) + 1])) {
        fail(t, "zero-group-or-decode", x, q, uint64_t(expert));
        return;
      }
      int m_idx = -1, n_idx = -1;
      if (!Geometry::decode_local_mn(
              int(q), prefix[size_t(expert)], x.group_tiles[size_t(expert)],
              1, m_idx, n_idx) || m_idx < 0 ||
          m_idx >= x.group_tiles[size_t(expert)] || n_idx < 0 ||
          uint64_t(n_idx) >= x.nt) {
        fail(t, "expert-local-mn", x, q, uint64_t(expert));
        return;
      }
      if (previous_expert >= 0 && expert != previous_expert) ++t.cross_group;
      previous_expert = expert;
      peers[size_t(q)].push_back({block, w.slice_count, w.slice_idx});
      sums[size_t(q)] += range_sum(q, w.K_idx, w.k_tile_count);
      for (uint64_t k = w.K_idx; k < uint64_t(w.K_idx) + w.k_tile_count; ++k) {
        ++visits[size_t(q * x.kt + k)];
      }
      ++t.segments;
      t.cells += w.k_tile_count;
      cursor = w.linear_next;
      w = Core::fetch_next_work(p, w);
    }
    if (cursor < p.total_k_tiles_ && cursor != stripe_end) {
      fail(t, "stripe-not-consumed", x, cursor, stripe_end);
      return;
    }
  }

  for (uint64_t cell = 0; cell < q_count * x.kt; ++cell) {
    if (visits[size_t(cell)] != 1) {
      fail(t, "cell-exact-once", x, visits[size_t(cell)], 1);
      return;
    }
  }
  std::set<uint64_t> locks;
  uint64_t local_handoffs = 0;
  for (uint64_t q = 0; q < q_count; ++q) {
    auto const& ps = peers[size_t(q)];
    if (ps.empty() || !locks.insert(q).second) {
      fail(t, "global-q-lock", x, q, 1);
      return;
    }
    for (size_t i = 0; i < ps.size(); ++i) {
      uint32_t const reverse = uint32_t(ps.size() - 1 - i);
      if (ps[i].count != ps.size() || ps[i].index != reverse) {
        fail(t, "peer-protocol", x, ps[i].index, reverse);
        return;
      }
    }
    local_handoffs += ps.size() - 1;
    if (sums[size_t(q)] != range_sum(q, 0, x.kt)) {
      fail(t, "exact-fixture", x, sums[size_t(q)], range_sum(q, 0, x.kt));
      return;
    }
  }
  if (q_count >= x.cu) {
    ++t.protected_rows;
    if (local_handoffs != 0 || p.grid_blocks_ != q_count ||
        p.iters_per_block_ != x.kt) {
      fail(t, "Q>=CU-protection", x, local_handoffs, 0);
      return;
    }
    t.protected_handoffs += local_handoffs;
  }
  t.handoffs += local_handoffs;
  t.groups += x.group_tiles.size();
  for (int mt : x.group_tiles) t.zero_groups += uint64_t(mt == 0);
  t.outputs += q_count;
  (void)multiplicity;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s manifest.tsv expected-rows\n", argv[0]);
    return 2;
  }
  unsigned long long expected_raw = 0;
  if (std::sscanf(argv[2], "%llu", &expected_raw) != 1) return 2;
  uint64_t const expected = uint64_t(expected_raw);
  std::ifstream in(argv[1]);
  if (!in) return 2;

  Totals t;
  std::map<Key, uint64_t> unique;
  int n, k, tokens, experts, topk, tm, tn, tk, cu;
  uint64_t multiplicity = 0;
  while (in >> n >> k >> tokens >> experts >> topk >> tm >> tn >> tk >> cu
            >> multiplicity) {
    moe_router_fixture::Rows routed;
    if (!moe_router_fixture::route(tokens, topk, experts, routed)) return 2;
    Key x;
    x.nt = Geometry::ceil_div_u64(uint64_t(n), uint64_t(tn));
    x.kt = Geometry::ceil_div_u64(uint64_t(k), uint64_t(tk));
    x.cu = uint64_t(cu);
    x.group_tiles.reserve(size_t(experts));
    for (int rows : routed.per_expert) {
      x.group_tiles.push_back(int(Geometry::ceil_div_u64(
          uint64_t(rows), uint64_t(tm))));
    }
    unique[std::move(x)] += multiplicity;
    t.raw += multiplicity;
  }
  if (!in.eof() || t.raw != expected) return 2;
  t.unique = unique.size();
  for (auto const& [x, mult] : unique) {
    check(x, mult, t);
    ++t.checked;
    if (t.errors) break;
  }
  std::printf("[l136] raw=%llu/%llu remaining=%llu unique=%llu/%llu remaining=%llu\n",
      (unsigned long long)t.raw, (unsigned long long)expected,
      (unsigned long long)(expected - t.raw),
      (unsigned long long)t.checked, (unsigned long long)t.unique,
      (unsigned long long)(t.unique - t.checked));
  std::printf("[l136] groups=%llu zero-groups=%llu outputs=%llu segments=%llu logical-(q,k)=%llu handoffs=%llu cross-group=%llu\n",
      (unsigned long long)t.groups, (unsigned long long)t.zero_groups,
      (unsigned long long)t.outputs, (unsigned long long)t.segments,
      (unsigned long long)t.cells, (unsigned long long)t.handoffs,
      (unsigned long long)t.cross_group);
  std::printf("[l136] Q>=CU classes=%llu handoffs=%llu criterion=raw-integer-equality-before-run fixture=grouped-marlin-cell-exact ORDER-INDEPENDENT+FP16-EXACT\n",
      (unsigned long long)t.protected_rows,
      (unsigned long long)t.protected_handoffs);
  std::printf("[l136] proposition-A grouped-ragged exact-once=>DP result=%s failures=%llu\n",
      t.errors == 0 ? "PASS" : "FAIL", (unsigned long long)t.errors);
  return t.errors == 0 ? 0 : 1;
}
