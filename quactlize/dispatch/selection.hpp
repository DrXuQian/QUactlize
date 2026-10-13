#pragma once
#include "compute.hpp"
#include "decode.hpp"
#include "smallm_matched.hpp"
#include "../include/ppu_format_config.hpp"
#include "../include/ppu_kquant_measured_policy.hpp"
#include "../include/ppu_q4_kpack4_shipping_policy.hpp"

namespace quactlize::dispatch {

// Internal identity, not an ABI enum or a persisted cache key. Channels and
// selection profile are separate dimensions instead of encoded integers.
enum class SelectionProfile { General, Decode, LegacySmallM, MatchedSmallM, DenseDynamic, DenseNative };
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

// The old dense device API's canonical K-pack fallback, expressed as a JIT
// parent rather than loading the six legacy DSOs. Reuse its authoritative
// policies, including auto delivery (DN0), not a similar-looking new tile.
inline TcDecision select_legacy_dense(qks_request_v1 const& r) {
    if (!valid(r) || r.route!=QK_DENSE_FQ || r.qtype==8) return {};
    TcDecision out;
    auto& c=out.value;
    c={"legacy-dense",nullptr,"LEGACY_DENSE",r.qtype,r.route,
        0,0,0,0,0,0,0,0,-1,1,0,0,0,r.mapping_id};
    if (r.qtype==12) {
        ppu_q4_kpack4_shipping::ConfigId id{};
        if (!ppu_q4_kpack4_shipping::find_config(nullptr,r.m,r.n,r.k,id)) return {};
        auto const& row=ppu_q4_kpack4_shipping::row(id);
        c.tm=row.tile_m;c.tn=row.tile_n;c.tk=row.tile_k;
        c.wm=row.warp_m;c.wn=row.warp_n;c.stages=row.stages;c.split=row.split;
    } else {
        ppu_dense_shipping::ConfigId id{};
        if (!ppu_kquant_measured_policy::find_dense_config(nullptr,r.qtype,r.m,r.n,r.k,id)) return {};
        auto const& row=ppu_dense_shipping::kConfigs[static_cast<int>(id)];
        c.tm=row.tile_m;c.tn=row.tile_n;c.tk=ppu_formats::for_qtype(r.qtype).fully_quantized_tile_k;
        c.wm=row.warp_m;c.wn=row.warp_n;c.stages=row.stages;
    }
    if (r.k%(c.tk*c.split) || r.k/(c.tk*c.split)<c.stages-1 || (c.tm==8 && r.m>64)) return {};
    out.symbol="fqk_tc_q"+std::to_string(r.qtype)+"_l"+(r.qtype==12?"1":"2")+
        "_a0_tm"+std::to_string(c.tm)+"_tn"+std::to_string(c.tn)+"_tk"+std::to_string(c.tk)+
        "_wm"+std::to_string(c.wm)+"_wn"+std::to_string(c.wn)+"_s"+std::to_string(c.stages)+"_bc0_ap0_dn0";
    out.policy=QKS_LEGACY_DENSE;
    return out;
}

// Preserve the existing native policy, then the old canonical FQ fallback.
// Only a shape the old path cannot cover gets the new resource-bounded S1.
inline TcDecision select_dense_dynamic(qks_request_v1 const& r,int compute) {
    if (r.route>=2 || !valid(r) || int64_t(r.m)*r.n>INT32_MAX) return {};
    auto out=select_tc(r,false,compute);
    if (out) return out;
    if (!compute_valid(compute)) return {};
    if (compute==QK_COMPUTE_F16) {
        out=select_legacy_dense(r);
        if (out) return out;
    }
    out.value=compute_proposal(r,{},out.symbol);
    if (!out.value.symbol) {out.symbol.clear();return out;}
    out.policy=QKS_DENSE_FALLBACK;
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
    if (base.kind == QKS_SMALLM_SIMT) {
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
