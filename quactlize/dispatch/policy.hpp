#pragma once
#include "api.h"
#include "../../policies/kpack_zw810_heuristic_v1.hpp"
#include <algorithm>
#include <limits>

namespace quactlize::dispatch {
namespace policy = quactlize_kpack_heuristic_v1;
using Config = policy::Config;
struct Selected { Config const* config = nullptr; int policy = 0; };

inline bool valid(qks_request_v1 const& r) {
    return r.version == 1 && r.size == sizeof(r) && r.qtype >= 10 && r.qtype <= 14 &&
        r.route >= 0 && r.route <= 3 && r.m > 0 && r.m <= INT32_MAX-256 &&
        r.n > 0 && r.n <= INT32_MAX-256 && r.k > 0 && r.k <= INT32_MAX-256 &&
        r.n % 256 == 0 && r.k % ((r.qtype == 11 || r.qtype == 14) ? 512 : 256) == 0 &&
        r.experts > 0 && r.experts <= INT32_MAX-256 &&
        r.mapping_id == (r.qtype == 12 ? UINT64_C(0x51344b5034540001) : UINT64_C(0x514b504b54000001)) &&
        (r.route < 2 ? r.experts == 1 && r.max_rows == r.m :
            r.max_rows > 0 && r.max_rows <= r.m && int64_t(r.max_rows)*r.experts >= r.m);
}

inline Selected select(qks_request_v1 const& r) {
    if (!valid(r)) return {};
    policy::Query q;
    q.qtype=r.qtype; q.n=r.n; q.k=r.k; q.m=r.m; q.total_rows=r.m;
    q.experts=r.experts; q.max_rows=r.max_rows;
    q.route=static_cast<policy::base::Route>(r.route);
    q.group_size=(r.qtype == 12 || r.qtype == 13) ? 32 : 16;
    q.device_name="PPU-ZW810"; q.compute_units=72; q.mapping_id=r.mapping_id;
    q.kernel_source=policy::kKernelSource; q.sdk_digest=policy::kSdkDigest;
    if (r.route < 2) {
        auto s=policy::select(q,true);
        int kind=s.status == policy::Status::MeasuredRecent ? QKS_RECENT :
            s.status == policy::Status::MeasuredHistorical ? QKS_HISTORICAL : QKS_PREDICTED;
        return {s.config,kind};
    }
    // No router readback or fabricated exact row vector. Rank the existing
    // same-family measurements using total work and the caller's row bound.
    // This is a bound-based proposal, never an exact grouped measurement.
    Config const* best=nullptr;
    int64_t distance=std::numeric_limits<int64_t>::max();
    for (auto const& ref : policy::detail::kPreferences) {
        auto const& key=ref.recent ? policy::detail::kRecent[ref.index].key :
            policy::base::detail::kEntries[ref.index].key;
        if (key[0]!=uint64_t(r.qtype) || key[1]!=uint64_t(r.route) ||
            key[2]!=uint64_t(r.n) || key[3]!=uint64_t(r.k) || key[4]!=uint64_t(r.experts)) continue;
        auto c=ref.recent ? policy::detail::kRecent[ref.index].config :
            &policy::base::detail::kConfigs[policy::base::detail::kEntries[ref.index].config];
        if (!policy::detail::eligible(*c,q)) continue;
        int64_t d=policy::detail::distance(r.m,key[5])+policy::detail::distance(r.max_rows,key[6]);
        if (d<distance) { best=c; distance=d; }
    }
    return {best,QKS_DEVICE_BOUNDS};
}

inline qk_recipe_v1 recipe(Config const& c, qks_request_v1 const& r, int occupancy) {
    qk_recipe_v1 out{1,sizeof(out),c.grid_mode >= 2 ? QK_PERSISTENT : QK_ORDINARY,c.split,0};
    if (out.algorithm == QK_PERSISTENT) {
        int b=std::max(1,std::min(c.grid_b,occupancy));
        int64_t mt=r.route >= 2 ? int64_t(r.experts)*((int64_t(r.max_rows)+c.tm-1)/c.tm) :
            (int64_t(r.m)+c.tm-1)/c.tm;
        int64_t tiles=mt*((int64_t(r.n)+c.tn-1)/c.tn), cap=72LL*b;
        out.grid=int(c.grid_mode == 3 ? (tiles+(tiles+cap-1)/cap-1)/((tiles+cap-1)/cap) :
            std::min(tiles,cap));
    }
    return out;
}
} // namespace quactlize::dispatch
