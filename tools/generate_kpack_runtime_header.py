"""SDK-free C++17 exact-table selector; generated data, no fitted model."""

import json

import kpack_runtime_policy as runtime


def generate(model):
    runtime.Selector(model)
    configs = sorted(model["configurations"])
    indices = {c: i for i, c in enumerate(configs)}
    modes = {"implicit": 0, "ordinary": 1, "capacity": 2, "balanced": 3}
    config_lines = []
    for cid in configs:
        c = model["configurations"][cid]
        ints = [c["qtype"], runtime.ROUTES.index(c["route"])] + [
            c[k]
            for k in (
                "tm",
                "tn",
                "tk",
                "wm",
                "wn",
                "stages",
                "ap",
                "dn",
                "persistent",
                "split",
            )
        ]
        ints += [modes[c["grid_mode"]], c["grid_b"], c["occupancy"]]
        config_lines.append(
            "  {"
            + ", ".join(
                [json.dumps(cid), json.dumps(c["symbol"]), json.dumps(c["algorithm"])]
                + list(map(str, ints))
                + [c["mapping_id"] + "ULL"]
            )
            + "},"
        )
    flat, vectors = [], []
    for row in model["row_vectors"]:
        vectors.append(f"  {{{len(flat)}, {len(row)}}},")
        flat.extend(row)
    entries = []
    for e in model["entries"]:
        status = (
            "MeasuredWithin5" if e["status"] == runtime.WITHIN else "MeasuredException"
        )
        entries.append(
            "  {{"
            + ", ".join(str(x) + "ULL" for x in e["key"])
            + "}, "
            + f"{e['row_vector']}, {indices[e['config_id']]}, Status::{status}, {e['max_measured_regret_pct']:.17g}, {e['max_measured_spread_pct']:.17g}"
            + "},"
        )
    binding = model["required_binding"]
    return (
        """// Generated measured data; do not edit. No kernel inventory or launch code.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace quactlize_kpack_runtime_v1 {
enum class Route { FqDense, SfDense, FqGrouped, SfGrouped };
enum class Status { InvalidQuery, BindingMismatch, FallbackRequired,
                    MeasuredWithin5, MeasuredException };
struct Query {
  int qtype = 0, n = 0, k = 0, group_size = 0;
  Route route = Route::FqDense;
  int m = 0, experts = 1, total_rows = 0, max_rows = 0;
  int const* rows = nullptr;
  int rows_count = 0;
  char const* device_name = nullptr;
  int compute_units = 0;
  std::uint64_t mapping_id = 0;
  char const* kernel_source = nullptr;
  char const* sdk_digest = nullptr;
};
struct Config {
  char const *id, *symbol, *algorithm;
  int qtype, route, tm, tn, tk, wm, wn, stages, ap, dn;
  int parent_persistent, split, grid_mode, grid_b, occupancy;
  std::uint64_t mapping_id;
};
struct Selection {
  Status status = Status::InvalidQuery;
  Config const* config = nullptr;
  std::int64_t grid = 0;
  double max_measured_regret_pct = 0, max_measured_spread_pct = 0;
};
"""
        + f"""inline constexpr char kPolicyDigest[] = "{model['policy_digest']}";
inline constexpr char kKernelSource[] = "{binding['kernel_source']}";
inline constexpr char kSdkDigest[] = "{binding['sdk_digest']}";
"""
        + """namespace detail {
using Key = std::array<std::uint64_t, 8>;
struct RowVector { int offset, count; };
struct Entry { Key key; int rows, config; Status status; double regret, spread; };
inline constexpr Config kConfigs[] = {
"""
        + "\n".join(config_lines)
        + """
};
inline constexpr std::int32_t kRows[] = {
"""
        + ",".join(map(str, flat or [0]))
        + """
};
inline constexpr RowVector kRowVectors[] = {
"""
        + "\n".join(vectors or ["  {0, 0},"])
        + """
};
inline constexpr Entry kEntries[] = {
"""
        + "\n".join(entries)
        + """
};
inline bool equal(char const* a, char const* b) { return a && std::strcmp(a, b) == 0; }
inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }
} // namespace detail

// FP16 full-output calibration only. Caller must bind the complete returned
// identity to its inventory and run can_implement. Neither a measured lookup
// nor a timing exception is a device-launch authorization by itself.
inline Selection select(Query const& q) {
  using namespace detail;
  int route = static_cast<int>(q.route);
  if (q.qtype < 10 || q.qtype > 14 || route < 0 || route > 3 ||
      q.n <= 0 || q.n % 16 || q.k <= 0 || q.k % 256 ||
      q.group_size != ((q.qtype == 12 || q.qtype == 13) ? 32 : 16)) return {};
  bool grouped = route >= 2;
  std::uint64_t hash = 0;
  if (grouped) {
    if (q.experts <= 0 || !q.rows || q.rows_count != q.experts ||
        q.total_rows <= 0 || q.max_rows <= 0) return {};
    std::int64_t total = 0;
    int maximum = 0;
    hash = 14695981039346656037ULL;
    for (int i = 0; i < q.experts; ++i) {
      if (q.rows[i] < 0) return {};
      total += q.rows[i];
      if (q.rows[i] > maximum) maximum = q.rows[i];
      auto row = static_cast<std::uint32_t>(q.rows[i]);
      for (int shift = 0; shift < 32; shift += 8)
        hash = (hash ^ ((row >> shift) & 255)) * 1099511628211ULL;
    }
    if (total != q.total_rows || maximum != q.max_rows) return {};
  } else if (q.m <= 0 || q.rows || q.rows_count) return {};
  std::uint64_t mapping = q.qtype == 12 ? 0x51344b5034540001ULL : 0x514b504b54000001ULL;
  if (!equal(q.device_name, "PPU-ZW810") || q.compute_units != 72 ||
      q.mapping_id != mapping || !equal(q.kernel_source, kKernelSource) ||
      !equal(q.sdk_digest, kSdkDigest)) return {Status::BindingMismatch};
  Key key{static_cast<std::uint64_t>(q.qtype), static_cast<std::uint64_t>(route),
          static_cast<std::uint64_t>(q.n), static_cast<std::uint64_t>(q.k),
          static_cast<std::uint64_t>(grouped ? q.experts : 1),
          static_cast<std::uint64_t>(grouped ? q.total_rows : q.m),
          static_cast<std::uint64_t>(grouped ? q.max_rows : 0), hash};
  std::size_t lo = 0, hi = sizeof(kEntries) / sizeof(kEntries[0]);
  auto count = hi;
  while (lo < hi) {
    auto mid = lo + (hi - lo) / 2;
    if (kEntries[mid].key < key) lo = mid + 1; else hi = mid;
  }
  for (; lo < count && kEntries[lo].key == key; ++lo) {
    auto const& e = kEntries[lo];
    if (grouped) {
      auto v = kRowVectors[e.rows];
      bool matches = v.count == q.experts;
      for (int i = 0; matches && i < v.count; ++i)
        matches = kRows[v.offset + i] == q.rows[i];
      if (!matches) continue; // Hash is an accelerator, never equality proof.
    }
    auto const& c = kConfigs[e.config];
    std::int64_t grid = 0;
    if (c.grid_mode) {
      std::int64_t mt = 0;
      if (grouped) for (int i = 0; i < q.experts; ++i) mt += ceil_div(q.rows[i], c.tm);
      else mt = ceil_div(q.m, c.tm);
      auto tiles = mt * ceil_div(q.n, c.tn);
      if (c.grid_mode == 1) grid = tiles;
      else {
        auto capacity = 72LL * c.grid_b;
        grid = c.grid_mode == 2 ? (tiles < capacity ? tiles : capacity)
                               : ceil_div(tiles, ceil_div(tiles, capacity));
      }
    }
    return {e.status, &c, grid, e.regret, e.spread};
  }
  return {Status::FallbackRequired};
}
} // namespace quactlize_kpack_runtime_v1
"""
    )
