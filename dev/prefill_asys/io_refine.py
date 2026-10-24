"""Reuse casts/reducer while preserving measured SF/Full mainloops and native control."""
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path

from .finalists import case_identity, counts, full_case
from dev.strategy.prescreen import rejection
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

SCHEMA='quactlize.dense-contiguous-io.v1'
HELPER_SCHEMA='quactlize.dense-helper-reuse.v1'


def converted(case):
    if case['arm']=='full-fp16':return full_case(case['helper_options']['dequant_variant'],4)
    result=copy.deepcopy(case)
    result['helper_options']=dict(vector_io=4)
    return result


def compact_case(case):
    if case['arm']!='sf':raise ValueError('compact reducer comparison requires SF')
    r=Recipe.from_dict(case['recipe']);p=dict(r.parameters)
    if p.get('dense_reducer') or p.get('split') not in (2,4,8) or p.get('grid_mode',0)!=0:
        raise ValueError('compact reducer requires an ordinary Split-K control')
    result=copy.deepcopy(case)
    p.update(symbol=p['symbol']+'_rc2',dense_reducer=2)
    result['recipe']=json.loads(json.dumps(asdict(Recipe.make(r.kind,r.route,**p))))
    return result


def make_plan(review, *, compact=False):
    from tools.review_prefill_cohort import eligible
    if (review.get('schema')!='quactlize.prefill-component-review.v1' or
            review.get('review_sha256')!=digest({k:v for k,v in review.items() if k!='review_sha256'}) or
            review.get('pending_points') or review.get('incomplete')):
        raise ValueError('complete sealed review required for IO-only comparison')
    points=[]
    for key,old in sorted(review['points'].items()):
        w=Workload(**old['workload'])
        if w.qtype!=8:continue
        if key!=w.key:raise ValueError('review workload differs')
        native=[r for r in old['measurements'] if r['arm']=='native-fp16' and r['status']=='PASS']
        if len(native)!=1 or len(native[0].get('confirmation_captures',[]))!=2:
            raise ValueError('previous native confirmation missing')
        cases=[];controls={}
        for route in ('sf','full-fp16'):
            rows=[r for r in old['measurements'] if r['arm']==route and eligible(r)]
            if not rows:raise ValueError('missing measured route: '+route)
            for r in rows:
                if (r['candidate']!=digest(review['candidates'][r['candidate']]) or
                        len(r.get('confirmation_captures',[]))!=2 or
                        not math.isfinite(r['gpu_sum_us']) or r['gpu_sum_us']<=0):
                    raise ValueError('previous recipe/timing identity differs')
            best=min(rows,key=lambda r:(r['gpu_sum_us'],r['candidate']))
            case=review['candidates'][best['candidate']]
            controls[route]=best['candidate']
            for c,role in ((case,'prior-route-winner:'+route),(converted(case),'contiguous-io:'+route)):
                cases.append(dict(copy.deepcopy(c),id=digest(c),roles=[role]))
            if compact and route=='sf' and dict(case['recipe']['parameters'])['split']>1:
                for base,role in ((case,'compact-reducer:sf'),(converted(case),'contiguous-io+compact-reducer:sf')):
                    c=compact_case(base)
                    cases.append(dict(c,id=digest(c),roles=[role]))
        points.append(dict(id=key,workload=asdict(w),consumers=copy.deepcopy(old['consumers']),tensors=[],
            cases=cases,prior_route_winners=controls,evidence_sha256=digest(old),
            candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],
            champion_controls=[],fixed_schedules=True,confirmation_captures=2))
    result=dict(schema=HELPER_SCHEMA if compact else SCHEMA,points=points,reuse=[],confirmation_captures=2,
        inputs=dict(review_sha256=review['review_sha256'],
                    helper_policy_sha256=digest(json.loads((Path(__file__).resolve().parents[2]/
                        'policies/dense_full_helpers_20261023.json').read_text()))),counts=counts(points),
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        scope=('FIXED_GEMM_IO_AND_COMPACT_REDUCER_FACTORIAL_NOT_MODEL_TPOT' if compact else
               'FIXED_SF_AND_FULL_GEMM_ONLY_IO_CHANGES_NOT_MODEL_TPOT'),
        historical_timing_used='RECIPE_SELECTION_ONLY_ALL_ARMS_REMEASURED',
        production_changed=False,performance_admitted=False)
    result['sha256']=digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema') not in (SCHEMA,HELPER_SCHEMA) or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'})
            or plan.get('confirmation_captures')!=2 or plan.get('production_changed') is not False
            or plan.get('performance_admitted') is not False or plan['reuse']):
        raise ValueError('IO-only plan identity/contract differs')
    seen=set()
    for p in plan['points']:
        w=Workload(**p['workload'])
        if (w.key!=p['id'] or w.key in seen or w.qtype!=8 or w.operation!='dense' or
                w.storage!='f32' or w.compute!='f16' or w.m not in (128,256,512,1024,2048) or
                len(p['cases']) not in (4,6) or p['confirmation_captures']!=2 or not p['fixed_schedules']):
            raise ValueError('IO-only workload/budget differs')
        seen.add(w.key);by_id={c['id']:c for c in p['cases']};required=set()
        if len(by_id)!=len(p['cases']):raise ValueError('duplicate IO-only candidate')
        for c in p['cases']:
            if c['id']!=digest(case_identity(c)):raise ValueError('changed IO candidate identity')
        for route in ('sf','full-fp16'):
            old=by_id[p['prior_route_winners'][route]]
            if old['roles']!=['prior-route-winner:'+route] or old['arm']!=route:
                raise ValueError('IO-only control lost')
            r=Recipe.from_dict(old['recipe']);o=old['helper_options']
            if route=='sf':
                if (r.kind!='tc' or r.route!='sf' or rejection(r,w) or old['helper_mode']!='cohort-current'
                        or o not in [dict(vector_io=i) for i in (-1,1,2,3)]):
                    raise ValueError('invalid SF control')
            elif (o['dequant_variant'] not in (0,22,23,36,37,38,41) or o['vector_io'] not in range(4)
                  or case_identity(old)!=full_case(o['dequant_variant'],o['vector_io'])):
                raise ValueError('invalid Full control')
            new=converted(case_identity(old))
            if (digest(new) not in by_id or case_identity(by_id[digest(new)])!=new or
                    by_id[digest(new)]['roles']!=['contiguous-io:'+route]):
                raise ValueError('IO-only candidate changed GEMM/dequant or lacks matching control')
            required.update((old['id'],digest(new)))
            if plan['schema']==HELPER_SCHEMA and route=='sf' and dict(r.parameters)['split']>1:
                for base,role in ((case_identity(old),'compact-reducer:sf'),(new,'contiguous-io+compact-reducer:sf')):
                    candidate=compact_case(base);cid=digest(candidate)
                    if (cid not in by_id or case_identity(by_id[cid])!=candidate or
                            by_id[cid]['roles']!=[role] or rejection(Recipe.from_dict(candidate['recipe']),w)):
                        raise ValueError('compact candidate changed mainloop or lost its independent control')
                    required.add(cid)
        if required!=by_id.keys():raise ValueError('helper factorial inventory differs')
        if (p['champion_controls'] or p['candidates']!=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc']):
            raise ValueError('IO-only resource inventory differs')
    if not seen or plan['counts']!=counts(plan['points']):raise ValueError('IO-only denominator differs')
    return plan
