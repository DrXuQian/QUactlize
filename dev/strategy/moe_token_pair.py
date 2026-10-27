"""Explicit indexed BF16 two-row candidate; never a production override."""
from quactlize.runtime.tuning import digest

GEOMETRIES=((4,2,8),(4,8,8),(8,4,4))


def source(item):
    from dev.strategy.vector import spec
    from quactlize.dispatch.table import Workload,Recipe
    c=item['config'];q=item['q']
    w=Workload(q,1024,5120,2,operation='indexed',compute='bf16',experts=8,topk=8)
    if spec(w,Recipe.make('simt','fq',**c,split=1,moe_pair_rows=item['moe_pair_rows']))!=item:
        raise ValueError('noncanonical indexed pair specialization')
    config='qkg_simt_config_v1 f{1,sizeof(f),'+','.join(str(c[f]) for f in ('variant','columns','warps','values'))+',split};'
    # Explicit causal control, separate from the selected production body.
    pair='true' if item['moe_pair_rows']==2 else 'false'
    return f'''#include "dev/strategy/moe_token_pair.cuh"
#include "simt_validation.hpp"
extern "C" char const* strategy_vector_identity() {{ return "{digest(item)}"; }}
extern "C" int strategy_vector_query(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split,qkg_sizes_v1* out) {{
    if(!d || !a || !out)return QKG_INVALID;
    *out={{}};
    if(d->call.qtype!={q} || d->call.mode!=QKG_INDEXED || d->call.topk<1 || d->call.topk>8 ||
       d->call.input_type!=QKG_F32 || d->compute_type!=QKG_COMPUTE_BF16)return QKG_SHAPE;
    {config}
    return quactlize::execution::simt::query_v2(*d,f,a,*out);
}}
extern "C" int strategy_vector_run(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split) {{
    qkg_sizes_v1 sizes{{}};
    int rc=strategy_vector_query(d,a,split,&sizes);if(rc)return rc;
    rc=quactlize::execution::simt::buffers_v2(*d,sizes);if(rc)return rc;
    return quactlize::execution::simt::moe_token_pair_launch<{q},{c['columns']},{c['warps']},{c['values']},{pair}>(*d,split);
}}
'''
