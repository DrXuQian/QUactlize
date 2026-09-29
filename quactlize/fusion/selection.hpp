#pragma once
#include "gate_up.h"

namespace quactlize::fusion {
enum Reader { GenericReader=0, Q8Tile16Reader=1, Q4Paired32Reader=2, Q4Paired16Reader=3 };

#define QGU_CATALOG(Id) inline constexpr uint64_t recipe_catalog_id=Id;
#include "recipes.inc"
#undef QGU_CATALOG

#define QGU_RECIPE(Name,Id,B,S,TM,W,R,V,C,P,Changes,Hoist,N,K) \
inline constexpr qkg_gate_up_recipe_v2 Name{2,sizeof(qkg_gate_up_recipe_v2),Id,recipe_catalog_id, \
    {1,sizeof(qkg_gate_up_config_v1),B,S,TM,W},R,V,C,P,Changes,Hoist,N,K};
#include "recipes.inc"
#undef QGU_RECIPE

inline qkg_gate_up_recipe_v2 const* recipe(uint64_t id) {
    switch(id) {
#define QGU_RECIPE(Name,Id,...) case Id:return &Name;
#include "recipes.inc"
#undef QGU_RECIPE
    default:return nullptr;
    }
}

inline bool same_config(qkg_gate_up_config_v1 const& a,qkg_gate_up_config_v1 const& b) {
    return a.version==b.version && a.size==b.size && a.backend==b.backend &&
        a.split==b.split && a.tile_m==b.tile_m && a.warps==b.warps;
}

inline bool valid_recipe(qkg_gate_up_recipe_v2 const& r) {
    auto p=recipe(r.recipe_id);
    return p && r.version==p->version && r.size==p->size && r.catalog_id==p->catalog_id &&
        same_config(r.config,p->config) && r.reader==p->reader && r.variant==p->variant &&
        r.columns==p->columns && r.values==p->values && r.changes==p->changes &&
        r.hoist==p->hoist && r.fixed_n==p->fixed_n && r.fixed_k==p->fixed_k;
}

inline qkg_gate_up_request_v2 request(qkg_gate_up_call_v1 const& d,int row_order,
    int consumer,int consumer_qtype=0,int consumer_n=0) {
    auto const& c=d.input.call;
    int tokens=c.topk>0 && c.rows%c.topk==0 ? c.rows/c.topk : 0;
    return {2,sizeof(qkg_gate_up_request_v2),QKG_GATE_UP_N4_V1,c.qtype,c.n,c.k,c.experts,
        tokens,c.mode,c.topk,c.channels,c.input_type,d.input.compute_type,d.output_type,
        d.round_projection,row_order,consumer,consumer_qtype,consumer_n};
}

inline int select(qkg_gate_up_request_v2 const& r,qkg_gate_up_recipe_v2* out) {
    if(!out) return QKG_INVALID;
    *out={};
    if(r.version!=2 || r.size!=sizeof(r) || r.input_type<0 || r.input_type>2 ||
       r.compute_type<0 || r.compute_type>1 || r.output_type<0 || r.output_type>2 ||
       r.round_projection<0 || r.round_projection>1 || r.row_order<0 || r.row_order>1 ||
       r.consumer<0 || r.consumer>2) return QKG_INVALID;
    if(r.layout_id!=QKG_GATE_UP_N4_V1) return QKG_ARRANGEMENT;
#define QGU_RULE(Name,Q,N,K,E,T,Mode,Top,Ch,In,Compute,Out,Round,Rows,Consumer,DQ,DN) \
    if(r.qtype==Q && r.n==N && r.k==K && r.experts==E && r.tokens==T && r.mode==Mode && \
       r.topk==Top && r.channels==Ch && r.input_type==In && r.compute_type==Compute && \
       r.output_type==Out && r.round_projection==Round && r.row_order==Rows && r.consumer==Consumer && \
       (DQ<0 || r.consumer_qtype==DQ) && (DN<0 || r.consumer_n==DN)) { *out=Name;return QKG_OK; }
#include "recipes.inc"
#undef QGU_RULE
    return QKG_SHAPE;
}

inline int select(int q,int n,int k,int experts,int tokens,int compute,qkg_gate_up_config_v1* out) {
    if(!out) return QKG_INVALID;
    *out={};
#define QGU_LEGACY(Name,Q,N,K,E,T,Compute) \
    if(q==Q && n==N && k==K && experts==E && tokens==T && compute==Compute) { \
        *out=Name.config;return QKG_OK; }
#include "recipes.inc"
#undef QGU_LEGACY
    return QKG_SHAPE;
}

// Compatibility only: preserve the actual implementations selected by old
// config_v1 calls, including overrides that config_v1 could not describe.
inline qkg_gate_up_recipe_v2 const* legacy_recipe(qkg_gate_up_call_v1 const& d,
    qkg_gate_up_config_v1 const& f) {
    auto const& c=d.input.call;
#define QGU_OVERRIDE(Name,Q,N,K,E,T,Mode,Top,Ch,In,Compute,Out,Round) \
    if(c.qtype==Q && c.n==N && c.k==K && c.experts==E && c.rows==T*Top && c.mode==Mode && \
       c.topk==Top && c.channels==Ch && c.input_type==In && d.input.compute_type==Compute && \
       d.output_type==Out && d.round_projection==Round && same_config(f,Name.config)) return &Name;
#include "recipes.inc"
#undef QGU_OVERRIDE
    return nullptr;
}

inline bool recipe_capable(qkg_gate_up_call_v1 const& d,qkg_gate_up_recipe_v2 const& r) {
    auto const& c=d.input.call;
    if((r.fixed_n && int64_t(c.n)*2!=r.fixed_n) || (r.fixed_k && c.k!=r.fixed_k)) return false;
    if(r.reader==GenericReader) return r.config.backend==QKG_GATE_UP_TC ||
        ((r.variant==5)==(c.qtype==8));
    if(c.input_type!=QKG_F32 || d.output_type!=QKG_F32) return false;
    if(r.reader==Q8Tile16Reader) return c.qtype==8 && d.input.compute_type==QKG_COMPUTE_F16 &&
        d.round_projection==0 && c.mode==QKG_DENSE && c.experts==1 && c.channels==1 && c.topk==1;
    return c.qtype==12 && d.input.compute_type==QKG_COMPUTE_BF16 && d.round_projection==1 &&
        c.mode==QKG_INDEXED && c.experts==256 && c.channels==1 && c.topk==8;
}
}
