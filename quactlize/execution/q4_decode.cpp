#include "q4_decode.h"
#include "q4_s1_validation.hpp"
#include "../../policies/kpack_q4_decode_v1.hpp"

extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&, qkg_q4_decode_config_v1 const&);

extern "C" int quactlize_kpack_q4_decode_select_v1(qkg_call_v1 const* call,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    qkg_q4_decode_config_v1* out, qkg_sizes_v1* sizes) {
    if (!call || !out || !sizes) return QKG_INVALID;
    *out={}; *sizes={};
    auto const& c=*call;
    qkg_config_v1 base{1,sizeof(base),16,4,1};
    qkg_sizes_v1 s{};
    int rc=quactlize::execution::query(c,base,arrangement,s);
    if (rc) return rc;
    if (c.qtype!=12) return QKG_FORMAT;
    if (c.input_type!=QKG_F32 || c.a_row_stride%4 ||
        (c.mode==QKG_INDEXED && c.a_token_stride%4) ||
        ((uintptr_t(c.a)|uintptr_t(c.low)|uintptr_t(c.units))&15)) return QKG_SHAPE;
    int tokens=0;
    if (c.mode==QKG_DENSE && c.experts==1 && c.rows<=8) tokens=c.rows;
    else if (c.mode==QKG_INDEXED && c.experts==256 && c.topk==8 && c.rows<=64 &&
             (c.channels==1 || c.channels==8)) tokens=c.rows/8;
    if (!tokens) return QKG_SHAPE;
    auto choice=quactlize::decode_policy::select(c.mode,c.n,c.k,tokens);
    if (!choice || !choice->simt) return QKG_SHAPE;
    auto f=choice->config;
    *out={1,sizeof(*out),f.reader,f.variant,f.warps,f.values,f.columns};
    *sizes=s;
    return QKG_OK;
}

extern "C" int quactlize_kpack_q4_decode_run_v1(qkg_call_v1 const* call,
    qkg_q4_decode_config_v1 const* config, quactlize_ppu_placed_arrangement_v2 const* arrangement) {
    if (!config || config->version!=1 || config->size!=sizeof(*config)) return QKG_INVALID;
    qkg_q4_decode_config_v1 selected{};
    qkg_sizes_v1 sizes{};
    int rc=quactlize_kpack_q4_decode_select_v1(call,arrangement,&selected,&sizes);
    if (rc) return rc;
    if (config->reader!=selected.reader || config->variant!=selected.variant ||
        config->warps!=selected.warps || config->values!=selected.values ||
        config->columns!=selected.columns) return QKG_INVALID;
    rc=quactlize::execution::q4_s1::buffers(*call,sizes);
    return rc ? rc : qkg_q4_decode_launch(*call,*config);
}
