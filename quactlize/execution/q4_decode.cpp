#include "q4_decode.h"
#include "q4_s1_validation.hpp"
#include "../../policies/kpack_q4_decode_v1.hpp"

extern "C" int qkg_q4_decode_launch(qkg_call_v1 const&, qkg_q4_decode_config_v1 const&);
extern "C" int qkg_q4_decode_launch_bf16(qkg_call_v1 const&, qkg_q4_decode_config_v1 const&);

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

extern "C" int quactlize_kpack_q4_decode_select_v2(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* arrangement,
    qkg_q4_decode_config_v1* out,qkg_sizes_v1* sizes) {
    if (!d || !out || !sizes || !quactlize::execution::q4_s1::typed_valid(*d)) return QKG_INVALID;
    if (d->compute_type==QKG_COMPUTE_F16)
        return quactlize_kpack_q4_decode_select_v1(&d->call,arrangement,out,sizes);
    if (d->call.input_type!=QKG_F32 && d->call.input_type!=QKG_SIMT_BF16) return QKG_SHAPE;
    auto storage=quactlize::execution::q4_s1::storage_call(*d);
    qkg_config_v1 base{1,sizeof(base),16,4,1};
    int rc=quactlize::execution::query(storage,base,arrangement,*sizes);
    if (rc) return rc;
    int alignment=storage.input_type==QKG_F16 ? 8 : 4;
    if (storage.a_row_stride%alignment ||
        (storage.mode==QKG_INDEXED && storage.a_token_stride%alignment)) return QKG_SHAPE;
    // The policy key contains geometry, not storage width. Validate the real
    // storage above, then use the unchanged selector without reading A.
    auto proposal=d->call;proposal.input_type=QKG_F32;
    return quactlize_kpack_q4_decode_select_v1(&proposal,arrangement,out,sizes);
}

extern "C" int quactlize_kpack_q4_decode_run_v2(qkg_simt_call_v2 const* d,
    qkg_q4_decode_config_v1 const* config,quactlize_ppu_placed_arrangement_v2 const* arrangement) {
    if (!d || !config || config->version!=1 || config->size!=sizeof(*config)) return QKG_INVALID;
    if (d->compute_type==QKG_COMPUTE_F16) {
        if (!quactlize::execution::q4_s1::typed_valid(*d)) return QKG_INVALID;
        return quactlize_kpack_q4_decode_run_v1(&d->call,config,arrangement);
    }
    qkg_q4_decode_config_v1 selected{};qkg_sizes_v1 sizes{};
    int rc=quactlize_kpack_q4_decode_select_v2(d,arrangement,&selected,&sizes);
    if (rc) return rc;
    if (config->reader!=selected.reader || config->variant!=selected.variant ||
        config->warps!=selected.warps || config->values!=selected.values ||
        config->columns!=selected.columns) return QKG_INVALID;
    rc=quactlize::execution::q4_s1::buffers_v2(*d,sizes);
    return rc ? rc : qkg_q4_decode_launch_bf16(d->call,*config);
}
