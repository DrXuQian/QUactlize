"""Generate an SDK-free single-choice selector; data reuse, no fitted tree."""

import json
import re

import kpack_heuristic as heuristic
import kpack_runtime_policy as runtime


def generate(model, base, base_header="kpack_zw810_runtime_v1.hpp"):
    selector = heuristic.Selector(model, base)
    if not re.fullmatch(r"[A-Za-z_0-9.-]+\.hpp", base_header):
        raise ValueError("invalid base header name")
    old_ids = {cid: i for i, cid in enumerate(sorted(base["configurations"]))}
    extra_ids = {cid: i for i, cid in enumerate(sorted(model["configurations"]))}
    modes = {"implicit": 0, "ordinary": 1, "capacity": 2, "balanced": 3}
    extra = []
    for cid in extra_ids:
        c = model["configurations"][cid]
        numbers = [c["qtype"], runtime.ROUTES.index(c["route"])] + [
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
        numbers += [modes[c["grid_mode"]], c["grid_b"], c["occupancy"]]
        extra.append(
            "  {"
            + ",".join(
                [json.dumps(cid), json.dumps(c["symbol"]), json.dumps(c["algorithm"])]
                + list(map(str, numbers))
                + [c["mapping_id"] + "ULL"]
            )
            + "},"
        )

    def config_ptr(cid):
        return (
            f"&base::detail::kConfigs[{old_ids[cid]}]"
            if cid in old_ids
            else f"&kExtraConfigs[{extra_ids[cid]}]"
        )

    flat, row_offsets, recent = [], {}, []
    for e in model["entries"]:
        rows = tuple(e["context"]["rows"])
        if rows not in row_offsets:
            row_offsets[rows] = len(flat)
            flat.extend(rows)
        recent.append(
            "  {{"
            + ",".join(str(v) + "ULL" for v in e["key"])
            + "},"
            + config_ptr(e["config_id"])
            + f",{row_offsets[rows]},{len(rows)}}},"
        )
    refs = {}
    for i, e in enumerate(base["entries"]):
        rows = base["row_vectors"][e["row_vector"]] if e["row_vector"] >= 0 else []
        refs[(tuple(e["key"]), tuple(rows))] = (i, False)
    for i, e in enumerate(model["entries"]):
        refs[(tuple(e["key"]), tuple(e["context"]["rows"]))] = (i, True)
    preferences = []
    ordered = sorted(
        selector.entries.items(),
        key=lambda kv: (kv[1]["source"] != heuristic.RECENT, kv[1]["config_id"], kv[0]),
    )
    for identity, e in ordered:
        index, is_recent = refs[identity]
        preferences.append(f"  {{{index},{str(is_recent).lower()},{e['profile'][2]}}},")
    b = model["required_binding"]
    return (
        "// Generated data. Selection is not device-launch admission.\n#pragma once\n"
        f"#include {json.dumps(base_header)}\n#include <string_view>\n"
        "namespace quactlize_kpack_heuristic_v1 {\n"
        "namespace base = quactlize_kpack_runtime_v1;\n"
        "using Query = base::Query;\nusing Config = base::Config;\n"
        "enum class Status { InvalidQuery, BindingMismatch, FallbackRequired,\n"
        "                    MeasuredRecent, MeasuredHistorical, HeuristicUnvalidated };\n"
        "struct Selection { Status status = Status::InvalidQuery; Config const* config = nullptr; std::int64_t grid = 0; };\n"
        f"inline constexpr char kModelDigest[] = {json.dumps(model['model_digest'])};\n"
        f"inline constexpr char kKernelSource[] = {json.dumps(b['kernel'])};\n"
        f"inline constexpr char kSdkDigest[] = {json.dumps(b['sdk'])};\n"
        f"inline constexpr char kModuleContract[] = {json.dumps(heuristic.common.digest(model['authority']['receipt']['compiler']))};\n"
        f"static_assert(std::string_view(base::kPolicyDigest) == {json.dumps(base['policy_digest'])},\n"
        '              "heuristic and measured policy differ");\n'
        "namespace detail {\nusing Key = base::detail::Key;\n"
        "inline constexpr Config kExtraConfigs[] = {\n"
        + "\n".join(extra or ["  {},"])
        + "\n};\n"
        "struct Recent { Key key; Config const* config; int offset, count; };\n"
        "inline constexpr int kRows[] = {" + ",".join(map(str, flat or [0])) + "};\n"
        "inline constexpr Recent kRecent[] = {\n"
        + "\n".join(recent or ["  {},"])
        + "\n};\n"
        f"inline constexpr int kRecentCount = {len(recent)};\n"
        "struct Preference { int index; bool recent; int active; };\n"
        "inline constexpr Preference kPreferences[] = {\n"
        + "\n".join(preferences)
        + "\n};\n"
        + r"""
inline bool eligible(Config const& c, Query const& q) {
  int route = static_cast<int>(q.route);
  bool grouped = route >= 2;
  int m = grouped ? q.total_rows : q.m;
  return c.route == route && c.qtype == q.qtype && q.k % (c.tk * c.split) == 0 &&
         q.k / (c.tk * c.split) >= c.stages - 1 &&
         (!c.ap || (!grouped && m == 1)) &&
         (grouped || c.tm != 8 || m <= (route == 0 ? 64 : 7)) &&
         (c.split == 1 || (!grouped && m < 64)) &&
         (c.grid_mode < 2 || (c.grid_b >= 1 && c.grid_b <= c.occupancy));
}
inline std::int64_t grid(Config const& c, Query const& q) {
  if (c.grid_mode < 2) return 0;
  std::int64_t mt = 0;
  if (static_cast<int>(q.route) >= 2)
    for (int i = 0; i < q.experts; ++i) mt += base::detail::ceil_div(q.rows[i], c.tm);
  else mt = base::detail::ceil_div(q.m, c.tm);
  auto tiles = mt * base::detail::ceil_div(q.n, c.tn);
  auto cap = 72LL * c.grid_b;
  return c.grid_mode == 1 ? tiles : c.grid_mode == 2 ? (tiles < cap ? tiles : cap)
       : base::detail::ceil_div(tiles, base::detail::ceil_div(tiles, cap));
}
inline std::int64_t distance(std::int64_t a, std::int64_t b) {
  auto largest = a > b ? a : b;
  return 1000000 * (a > b ? a - b : b - a) / (largest ? largest : 1);
}
} // namespace detail

// No allocation, JIT, profiling, module loading or launch. Query.kernel_source
// and sdk_digest must use the runtime module's identity hash scope, not the
// historical sweep's differently scoped hashes. Prediction is explicitly opt-in.
inline Selection select(Query const& q, bool allow_prediction = false) {
  // Reuse artifact/query validation and historical lookup. These are calibration
  // data, not current-module binding: that separate check is immediately below.
  auto old_query = q;
  old_query.device_name = "PPU-ZW810"; old_query.compute_units = 72;
  old_query.mapping_id = q.qtype == 12 ? 0x51344b5034540001ULL : 0x514b504b54000001ULL;
  old_query.kernel_source = base::kKernelSource; old_query.sdk_digest = base::kSdkDigest;
  auto old = base::select(old_query);
  if (old.status == base::Status::InvalidQuery || q.n % 256 ||
      q.k % ((q.qtype == 11 || q.qtype == 14) ? 512 : 256)) return {};
  if (!base::detail::equal(q.device_name, "PPU-ZW810") || q.compute_units != 72 ||
      q.mapping_id != old_query.mapping_id || !base::detail::equal(q.kernel_source, kKernelSource) ||
      !base::detail::equal(q.sdk_digest, kSdkDigest)) return {Status::BindingMismatch};
  bool grouped = static_cast<int>(q.route) >= 2;
  int experts = grouped ? q.experts : 1;
  int total = grouped ? q.total_rows : q.m;
  int maximum = grouped ? q.max_rows : 0;
  auto family_matches = [&](detail::Key const& key) {
    return key[0] == std::uint64_t(q.qtype) && key[1] == std::uint64_t(q.route) &&
           key[2] == std::uint64_t(q.n) && key[3] == std::uint64_t(q.k) && key[4] == std::uint64_t(experts);
  };
  for (int j = 0; j < detail::kRecentCount; ++j) {
    auto const& e = detail::kRecent[j];
    if (!family_matches(e.key) || e.key[5] != std::uint64_t(total) || e.key[6] != std::uint64_t(maximum)) continue;
    bool equal = e.count == (grouped ? experts : 0);
    for (int i = 0; equal && i < e.count; ++i) equal = detail::kRows[e.offset + i] == q.rows[i];
    if (equal) {
      if (!detail::eligible(*e.config, q)) return {Status::FallbackRequired};
      return {Status::MeasuredRecent, e.config, detail::grid(*e.config, q)};
    }
  }
  if (old.config) {
    if (!detail::eligible(*old.config, q)) return {Status::FallbackRequired};
    return {Status::MeasuredHistorical, old.config, detail::grid(*old.config, q)};
  }
  if (!allow_prediction) return {Status::FallbackRequired};
  int active = 0;
  if (grouped) for (int i = 0; i < experts; ++i) active += q.rows[i] > 0;
  Config const* best = nullptr;
  std::int64_t best_distance = 4000000;
  // Preferences contain only effective entries: superseded old entries cannot
  // accidentally reappear as nearest-neighbor seeds. Deterministic tie order.
  for (auto const& ref : detail::kPreferences) {
    auto const& key = ref.recent ? detail::kRecent[ref.index].key : base::detail::kEntries[ref.index].key;
    if (!family_matches(key)) continue;
    auto const* c = ref.recent ? detail::kRecent[ref.index].config
                              : &base::detail::kConfigs[base::detail::kEntries[ref.index].config];
    if (!detail::eligible(*c, q)) continue;
    auto score = detail::distance(total, key[5]) + detail::distance(maximum, key[6])
               + detail::distance(active, ref.active);
    if (score < best_distance) { best_distance = score; best = c; }
  }
  if (!best) return {Status::FallbackRequired};
  return {Status::HeuristicUnvalidated, best, detail::grid(*best, q)};
}
} // namespace quactlize_kpack_heuristic_v1
"""
    )
