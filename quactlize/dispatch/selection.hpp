#pragma once
#include "compute.hpp"
#include "decode.hpp"
#include "effective.hpp"

namespace quactlize::dispatch {

// Internal identity, not an ABI enum or a persisted cache key. Channels and
// selection profile are separate dimensions instead of encoded integers.
enum class SelectionProfile { General, Decode, LegacySmallM, MatchedSmallM };
struct SelectionContext {
    SelectionProfile profile = SelectionProfile::General;
    int channels = 0;
};

struct TcDecision {
    Config value{};
    std::string symbol;
    int policy = 0;
    explicit operator bool() const { return !symbol.empty(); }
    Config config() const { auto c = value; c.symbol = symbol.c_str(); return c; }
};

// Pure selection shared by runtime and the fixed-route build planner.
// No DSO, JIT, device access or resource allocation belongs here.
inline TcDecision select_tc(qks_request_v1 const& r, bool decode, int compute,
                            Selected preferred = {}) {
    TcDecision out;
    if (!valid(r) || !compute_valid(compute)) return out;
    auto selected = preferred.config ? preferred : decode ? select_decode_tc(r) : select(r);
    bool matched = preferred.config && (preferred.policy == QKS_MATCHED_EXACT ||
        preferred.policy == QKS_MATCHED_BUCKET || preferred.policy == QKS_MATCHED_ROUTER);
    if (compute == QK_COMPUTE_BF16 && !matched) {
        if (!selected.config) selected = select(r);
        out.value = compute_proposal(r, selected, out.symbol);
        if (!out.value.symbol) out.symbol.clear();
        out.policy = QKS_COMPUTE_INITIAL;
    } else if (selected.config) {
        out.value = *selected.config;
        out.symbol = selected.config->symbol;
        out.policy = selected.policy;
    }
    return out;
}

struct SmallMDecision {
    int status = QKS_MISS;
    qks_smallm_choice_v2 choice{2, sizeof(qks_smallm_choice_v2)};
    Config const* tc = nullptr;
};

// The complete v3 host decision. Module materialization is a separate step;
// a SIMT decision must never cause a TC parent to load or compile.
inline SmallMDecision select_smallm(qkg_simt_call_v2 const& typed,
        quactlize_ppu_placed_arrangement_v2 const& arrangement) {
    SmallMDecision out;
    auto selected = matched::select(typed);
    if (!selected.row) return out;
    auto const& row = *selected.row;
    auto const& original = matched::data::kChoices[row.choice];
    auto const& replacement = effective::kReplacements[size_t(selected.row - matched::data::kExact)];
    auto& result = out.choice;
    auto& base = result.base;
    result.compute_type = typed.compute_type;
    base.version = 1; base.size = sizeof(base); base.kind = original.kind;
    base.policy = selected.policy;
    base.source_n = row.n; base.source_k = row.k; base.source_tokens = row.tokens;
    qkg_simt_config_v1 control{1, sizeof(control), 0, 4, 4, 4, 1};
    if (execution::simt::query_v2(typed, control, &arrangement, base.sizes) != QKG_OK) {
        out.status = QKS_INVALID;
        return out;
    }
    if (replacement.version) {
        base.kind = QKS_SMALLM_SIMT;
        base.simt = replacement;
        if (selected.policy != QKS_MATCHED_BUCKET) base.policy = QKS_Q8_VECTOR_MEASURED;
    } else if (base.kind == QKS_SMALLM_SIMT) {
        auto f = original.reader;
        base.simt = {1, sizeof(base.simt), f.variant, f.columns, f.warps, f.values, f.split};
    }
    if (base.kind == QKS_SMALLM_SIMT) {
        if (execution::simt::query_v2(typed, base.simt, &arrangement, base.sizes) != QKG_OK) return out;
    } else if (base.kind == QKS_SMALLM_TC) {
        out.tc = &original.tc;
    } else {
        auto f = original.reader;
        result.q4 = {1, sizeof(result.q4), f.reader, f.variant, f.warps, f.values, f.columns};
    }
    out.status = QKS_OK;
    return out;
}
} // namespace quactlize::dispatch
