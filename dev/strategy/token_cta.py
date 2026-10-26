"""Explicit CTA-local M candidates, not an implicit production override."""
from dataclasses import asdict
from pathlib import Path

from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.tuning import digest

GEOMETRIES=((4,8,8),(4,2,8),(8,4,4))
Q8_GEOMETRIES=((4,8,4),(8,4,4))


def q8_control_entry(w,c,split):
    """Outer entry of the existing vector carrier, not a new body selector.

    Host tests compare every model/config prediction against the production
    C++ predicates. Do not replace fixed/measured controls with generic Q8.
    """
    prefix='quactlize::execution::simt::'
    if w.qtype!=8 or w.operation!='dense' or w.compute!='f16':
        raise ValueError('Q8 control entry requires a dense F16 model workload')
    if w.m==1:
        fields=(w.qtype,0,w.n,w.k,1,1,1,0,c['variant'],c['columns'],c['warps'],c['values'],split)
        path=Path(__file__).resolve().parents[2]/'quactlize/execution/measured_decode.inc'
        for line in path.read_text().splitlines():
            if line.startswith('QDM('):
                values=line[4:-1].split(',')
                if tuple(map(int,values[1:14]))==fields:
                    return prefix+'measured_decode_kernel'
        if c['variant']==5 and c['values']==4:
            if (c['columns'],c['warps'])==(8,4) and (
                    (split,w.n,w.k) in ((8,2048,4096),(1,8192,2048))):
                return prefix+'q8_vector::kernel_model'
            if c['columns']==4 and split==1 and (c['warps']==2 or
                    (c['warps']==8 and (w.n,w.k)==(4096,2048))):
                return prefix+'q8_vector::kernel_s1'
    return prefix+'q8_vector::kernel'


def retain_tc_controls(point):
    """A reader experiment, not another TC search. Never drop an incumbent.

    Same-point old winners/near-winners remain freshly measured controls;
    transferred TC geometries and their Split-K neighbors are out of scope.
    Existing receipts are not relabelled as current-device timings.
    """
    arms=[];excluded=list(point['excluded'])
    for a in point['arms']:
        if (a['recipe']['kind']!='tc' or a['required'] or
                'EXACT_OLD_WINNER_OR_WITHIN_FIVE_PERCENT' in a['reasons']):
            arms.append(a)
        else:
            excluded.append(dict(recipe=a['recipe'],reason='UNCHANGED_TC_NEIGHBOR_NOT_TOKEN_REUSE_CONTROL',
                                 previous_reasons=a['reasons']))
    sf=sorted({max(0,dict(a['recipe']['parameters']).get('dequant_config',-1)) for a in arms
               if a['recipe']['route']=='sf' and point['workload']['qtype']!=8})
    return dict(point,arms=arms,excluded=excluded,sf_configs=sf)


def augment(point):
    from dev.strategy.vector import spec
    w=Workload(**point['workload'])
    if w.qtype not in (8,10,11,12,13,14) or w.operation!='dense' or w.compute!='f16' or not 1<=w.m<=8:
        raise ValueError('not a supported dense token cohort')
    # M1 retains the existing implementation and serves as an unchanged control.
    if w.m==1:return point
    arms=list(point['arms'])
    for c,warps,p in (Q8_GEOMETRIES if w.qtype==8 else GEOMETRIES):
        for split in (1,2,4,8):
            for window in ((1,2) if w.qtype==8 else (None,)):
                extra={} if window is None else dict(cta_window=window)
                r=Recipe.make('simt','fq',variant=5 if w.qtype==8 else 3,
                              columns=c,warps=warps,values=p,split=split,cta_tokens=w.m,**extra)
                s=spec(w,r)
                if any(a['recipe']==asdict(r) for a in arms):
                    continue  # An admitted incumbent is already this exact body.
                arms.append(dict(id=r.key,recipe=asdict(r),roles=['rule'],aliases=[],required=True,
                    reasons=['CTA_LOCAL_TOKEN_REUSE_CANDIDATE_NOT_ADMITTED'],spec=s,module=digest(s)))
    if len({a['id'] for a in arms})!=len(arms):raise ValueError('duplicate token candidate')
    return dict(point,arms=arms)


def source(item):
    from dev.strategy.vector import spec
    tokens=item['cta_tokens'];c=item['config']
    w=Workload(item['q'],item['n'] or 1024,item['k'] or 5120,tokens,compute=item['compute'])
    extra={k:item[k] for k in ('cta_window',) if k in item}
    if spec(w,Recipe.make('simt','fq',**c,split=1,cta_tokens=tokens,**extra))!=item:
        raise ValueError('noncanonical token specialization')
    q=item['q'];q8=q==8
    header='simt_q8_token_cta.cuh' if q8 else 'simt_token_cta.cuh'
    launch='q8_token_cta_launch' if q8 else 'token_cta_launch'
    arguments=[tokens,c['columns'],c['warps'],c['values']]
    if q8:arguments.append(item['cta_window'])
    elif q!=14:arguments.append(q) # Keep the original Q6 carrier instantiation.
    config='qkg_simt_config_v1 f{1,sizeof(f),'+','.join(str(c[f]) for f in ('variant','columns','warps','values'))+',split};'
    return f'''#include "{header}"
#include "simt_validation.hpp"
extern "C" char const* strategy_vector_identity() {{ return "{digest(item)}"; }}
extern "C" int strategy_vector_query(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split,qkg_sizes_v1* out) {{
    if(!d || !a || !out)return QKG_INVALID;
    *out={{}};
    if(d->call.qtype!={q} || d->call.mode!=QKG_DENSE || d->call.rows!={tokens} ||
       d->call.input_type!=QKG_F32 || d->compute_type!=QKG_COMPUTE_F16)return QKG_SHAPE;
    {config}
    return quactlize::execution::simt::query_v2(*d,f,a,*out);
}}
extern "C" int strategy_vector_run(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int split) {{
    qkg_sizes_v1 sizes{{}};
    int rc=strategy_vector_query(d,a,split,&sizes);if(rc)return rc;
    rc=quactlize::execution::simt::buffers_v2(*d,sizes);if(rc)return rc;
    return quactlize::execution::simt::{launch}<{','.join(map(str,arguments))}>(*d,split);
}}
'''
