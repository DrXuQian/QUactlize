#pragma once
#include "smallm_matched.hpp"
#include "../../policies/kpack_q8_vector_v1.hpp"
#include <array>

namespace quactlize::dispatch::effective {
namespace data = matched::data;

constexpr bool equal(char const* a, char const* b) {
    if (!a || !b) return a == b;
    while (*a && *a == *b) { ++a; ++b; }
    return *a == *b;
}

// Fold measured replacements once at compile time. The original exact rows,
// exclusions, donor order and *original* TC eligibility remain in matched;
// changing that eligibility after TC->SIMT replacement would change policy.
constexpr qkg_simt_config_v1 replacement(data::Row const& r) {
    auto const& choice = data::kChoices[r.choice];
    if (r.q != 8) return {};
    if (choice.kind == QKS_SMALLM_SIMT) {
        for (auto const& v : quactlize::q8_vector_data::kRows) {
            if (r.mode != v.mode || r.n != v.n || r.k != v.k || r.experts != v.experts ||
                r.topk != v.topk || r.channels != v.channels || r.tokens != v.tokens ||
                r.compute != v.compute) continue;
            auto const& f = choice.reader;
            if (f.variant != v.baseline[0] || f.columns != v.baseline[1] ||
                f.warps != v.baseline[2] || f.values != v.baseline[3] ||
                f.split != v.baseline[4]) return {};
            auto a = v.candidate;
            return {1, sizeof(qkg_simt_config_v1), a[0], a[1], a[2], a[3], a[4]};
        }
    } else if (choice.kind == QKS_SMALLM_TC && r.compute == QK_COMPUTE_F16 &&
               r.mode == QKG_DENSE && r.experts == 1 && r.topk == 1 &&
               r.channels == 1 && r.tokens == 1) {
        auto const& f = choice.tc;
        if (f.qtype != 8 || f.route != 1 || f.parent_persistent != -1 ||
            f.grid_mode != 0 || f.grid_b != 0 || !f.symbol) return {};
        for (auto const& v : quactlize::q8_vector_data::kTcRows) {
            if (r.n != v.n || r.k != v.k || !equal(f.symbol, v.symbol) ||
                f.tm != v.tm || f.tn != v.tn || f.tk != v.tk || f.wm != v.wm ||
                f.wn != v.wn || f.stages != v.stages || f.ap != v.ap ||
                f.dn != v.dn || f.split != v.split) continue;
            auto a = v.candidate;
            return {1, sizeof(qkg_simt_config_v1), a[0], a[1], a[2], a[3], a[4]};
        }
    }
    return {};
}

constexpr auto replacements() {
    std::array<qkg_simt_config_v1, sizeof(data::kExact) / sizeof(data::kExact[0])> out{};
    for (size_t i = 0; i < out.size(); ++i) out[i] = replacement(data::kExact[i]);
    return out;
}
inline constexpr auto kReplacements = replacements();
} // namespace quactlize::dispatch::effective
