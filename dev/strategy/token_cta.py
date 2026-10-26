"""Explicit Q6 CTA-local M candidate, not an implicit production override."""
from dataclasses import asdict

from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.tuning import digest

GEOMETRIES=((4,8,8),(4,2,8),(8,4,4))


def augment(point):
    from dev.strategy.vector import spec
    w=Workload(**point['workload'])
    if w.qtype!=14 or w.operation!='dense' or w.compute!='f16' or not 1<=w.m<=8:
        raise ValueError('not the Q6 dense token cohort')
    # M1 retains the existing implementation and serves as an unchanged control.
    if w.m==1:return point
    arms=list(point['arms'])
    for c,warps,p in GEOMETRIES:
        for split in (1,2,4,8):
            r=Recipe.make('simt','fq',variant=3,columns=c,warps=warps,values=p,split=split,cta_tokens=w.m)
            s=spec(w,r)
            arms.append(dict(id=r.key,recipe=asdict(r),roles=['rule'],aliases=[],required=True,
                reasons=['CTA_LOCAL_TOKEN_REUSE_CANDIDATE_NOT_ADMITTED'],spec=s,module=digest(s)))
    if len({a['id'] for a in arms})!=len(arms):raise ValueError('duplicate token candidate')
    return dict(point,arms=arms)


def source(item):
    from dev.strategy.vector import spec
    tokens=item['cta_tokens'];c=item['config']
    w=Workload(item['q'],item['n'] or 1024,item['k'] or 5120,tokens,compute=item['compute'])
    if spec(w,Recipe.make('simt','fq',**c,split=1,cta_tokens=tokens))!=item:
        raise ValueError('noncanonical token specialization')
    config='qkg_simt_config_v1 f{1,sizeof(f),'+','.join(str(c[f]) for f in ('variant','columns','warps','values'))+',split};'
    return f'''#include "simt_token_cta.cuh"
#include "simt_validation.hpp"
extern "C" char const* strategy_vector_identity() {{ return "{digest(item)}"; }}
extern "C" int strategy_vector_query(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split,qkg_sizes_v1* out) {{
    if(!d || !a || !out)return QKG_INVALID;
    *out={{}};
    if(d->call.qtype!=14 || d->call.mode!=QKG_DENSE || d->call.rows!={tokens} ||
       d->call.input_type!=QKG_F32 || d->compute_type!=QKG_COMPUTE_F16)return QKG_SHAPE;
    {config}
    return quactlize::execution::simt::query_v2(*d,f,a,*out);
}}
extern "C" int strategy_vector_run(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split) {{
    qkg_sizes_v1 sizes{{}};
    int rc=strategy_vector_query(d,a,split,&sizes);if(rc)return rc;
    rc=quactlize::execution::simt::buffers_v2(*d,sizes);if(rc)return rc;
    return quactlize::execution::simt::token_cta_launch<{tokens},{c['columns']},{c['warps']},{c['values']}>(*d,split);
}}
'''
