"""Factorized helper search: measured shortlist, immutable old controls, no routing."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
MS=(128,256,512,1024,2048)


def variants(q):
    if q in (8,10,11): return (0,22,23)
    if q in (12,13,14): return (0,8,20,21)
    raise ValueError('unsupported helper format')


def cohort(path=None):
    value=json.loads(Path(path or Path(__file__).with_name('cohort.json')).read_text())
    if value['schema']!='quactlize.dense-helper-cohort.v1' or value['ms']!=list(MS):
        raise ValueError('helper cohort contract differs')
    if not value['families']:raise ValueError('empty helper cohort')
    seen=set()
    for f in value['families']:
        key=(f['q'],f['n'],f['k'])
        if key in seen: raise ValueError('duplicate helper family')
        seen.add(key); variants(f['q']); Workload(*key,128)
        if len({r[0] for r in f['previous_f16_helpers']})!=len(f['previous_f16_helpers']):
            raise ValueError('duplicate historical control M')
        for m,dq,io in f['previous_f16_helpers']:
            if m not in MS or dq not in variants(f['q']) or io not in range(4):
                raise ValueError('illegal historical helper control')
    return value


def jobs(families):
    result={}
    for f in families:
        for storage in (0,1):
            rows=[dict(kind='dequant',q=f['q'],n=f['n'],k=f['k'],storage=storage)]
            rows += [dict(kind='io',m=m,width=width,input=input,storage=storage)
                     for m in MS for width in sorted({f['n'],f['k']}) for input in (0,1)]
            for row in rows: result[digest(row)]=row
    return result


def validate_result(row,job,authority):
    if (row.get('status')!='PASS' or row.get('job')!=job or row.get('authority')!=authority or
            row.get('record_sha256')!=digest({k:v for k,v in row.items() if k!='record_sha256'})):
        raise ValueError('helper cell identity or checksum differs')
    traffic=row.get('traffic',{});copies=row.get('copies',0)
    if (copies<2 or row.get('replays')!=2*copies or
            any(traffic.get(k,0)*copies<2.25*row.get('l2_bytes',0) for k in ('reads','writes')) or
            row.get('l2_bytes',0)<=0):
        raise ValueError('helper cell cache regime incomplete')
    expected=variants(job['q']) if job['kind']=='dequant' else range(4)
    if set(row['arms'])!={str(v) for v in expected}: raise ValueError('helper cell inventory incomplete')
    for key,arm in row['arms'].items():
        values=arm['samples_us']
        if (len(values)!=30 or any(not math.isfinite(v) or v<=0 for v in values) or
                arm['median_us']!=statistics.median(values) or arm['proof']['bad'] or
                arm['proof']['guard']!='PASS' or not arm['proof']['control_raw_bits_equal']):
            raise ValueError('helper cell numerical/sample evidence incomplete: '+key)
    if not row['negative_detected']: raise ValueError('helper cell lacks negative control')
    return row


def helper_options(f,m,storage,measurements,policy):
    def get(job):
        row=measurements[digest(job)]
        return {int(k):a['median_us'] for k,a in row['arms'].items()}
    dq=get(dict(kind='dequant',q=f['q'],n=f['n'],k=f['k'],storage=storage))
    inp=get(dict(kind='io',m=m,width=f['k'],input=1,storage=storage))
    out=get(dict(kind='io',m=m,width=f['n'],input=0,storage=storage))
    # At most two joint finalists. This sum only prunes; complete Asys calls
    # decide the result. Never equate isolated sums with measured E2E.
    ranked=sorted((dq[d]+inp[i]+out[i],d,i) for d in dq for i in inp)
    selected={}
    def add(d,i,role):
        roles=selected.setdefault((d,i),[])
        if role not in roles: roles.append(role)
    add(0,0,'typed-direct-control')
    for old_m,d,i in f['previous_f16_helpers']:
        if old_m==m: add(d,i,'previous-f16-helper-control' if not storage else 'previous-config-bf16-control')
    current=policy(f['q'],m,f['n'],f['k'],storage)
    if current is not None: add(*current,'production-helper-control')
    for _,d,i in ranked[:2]: add(d,i,'component-finalist')
    return [dict(dequant_variant=d,vector_io=i,roles=roles,
                 isolated_sum_us=dq[d]+inp[i]+out[i],cost_scope='PRUNING_NOT_E2E')
            for (d,i),roles in sorted(selected.items())]


def full_plan(families,measurements,policy,authority):
    points=[];pending=[]
    for f in families:
        for m in MS:
            w=Workload(f['q'],f['n'],f['k'],m);cases=[]
            for storage,arm in ((0,'full-fp16'),(1,'full-bf16-typed')):
                try: options=helper_options(f,m,storage,measurements,policy)
                except KeyError as e:
                    pending.append(dict(workload=asdict(w),storage=storage,missing=str(e)));continue
                for o in options:
                    h={k:o[k] for k in ('dequant_variant','vector_io')}
                    r=Recipe.make('full','full',provider='typed-bf16' if storage else 'typed-fp16',**h)
                    cases.append(dict(id=r.key,arm=arm,recipe=asdict(r),helper_mode='experimental',
                                      helper_options=h,roles=o['roles'],component_sum_us=o['isolated_sum_us']))
            if cases: points.append(dict(id=w.key,workload=asdict(w),cases=cases,
                consumers=[dict(model=name) for name in f['models']],tensors=[],call_count=None,
                coverage_scope=f['scope'],confirmation_captures=2))
    plan=dict(schema='quactlize.dense-helper-finalists.v1',points=points,pending=pending,
        component_authority=authority,production_changed=False,performance_admitted=False,
        scope='HELPER_FULL_CALL_CONFIRMATION_NOT_GEMM_ROUTE_SEARCH',
        counts=dict(points=len(points),full_arms=sum(len(p['cases']) for p in points),tc_modules=0))
    plan['sha256']=digest(plan)
    return plan


def validate_full_plan(value):
    if (value.get('schema')!='quactlize.dense-helper-finalists.v1' or
            value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'})):
        raise ValueError('helper finalist plan identity differs')
    if not value['points'] or value['pending']:
        raise ValueError('helper finalist component coverage incomplete')
    seen=set()
    for p in value['points']:
        w=Workload(**p['workload'])
        if (w.key!=p['id'] or w.key in seen or w.m not in MS or w.operation!='dense' or
                p.get('confirmation_captures')!=2 or not 4<=len(p['cases'])<=10 or
                {c['arm'] for c in p['cases']}!={'full-fp16','full-bf16-typed'}):
            raise ValueError('invalid helper finalist workload')
        seen.add(w.key);keys=set()
        for c in p['cases']:
            r=Recipe.from_dict(c['recipe']);o=c['helper_options']
            parameters=dict(r.parameters)
            if (c['id']!=r.key or r.key in keys or r.kind!='full' or r.route!='full' or
                    c['arm'] not in ('full-fp16','full-bf16-typed') or c.get('module') or
                    o['dequant_variant'] not in variants(w.qtype) or o['vector_io'] not in range(4) or
                    set(o)!={'dequant_variant','vector_io'}):
                raise ValueError('invalid helper finalist candidate')
            if parameters!=dict(o,provider='typed-fp16' if c['arm']=='full-fp16' else 'typed-bf16'):
                raise ValueError('helper recipe precision/options differ')
            keys.add(r.key)
        for arm in ('full-fp16','full-bf16-typed'):
            if not any(c['arm']==arm and c['helper_options']==dict(dequant_variant=0,vector_io=0)
                       and 'typed-direct-control' in c['roles'] for c in p['cases']):
                raise ValueError('helper finalist lacks typed control')
    return value
