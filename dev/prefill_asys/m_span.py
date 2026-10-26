"""Bounded M extrapolation experiment; no production buckets are broadened.

Rules see geometry, not timing/winner labels. Current and nearby winners are
separate required controls. Heavy SF parents remain in the candidate basis.
"""
from collections import defaultdict
from dataclasses import asdict
import copy
import json
import math

from .finalists import case_identity, counts, full_case
from .supplement import execution_case, policy_lookup
from dev.strategy.prescreen import rejection
from dev.strategy.prefill_plan import bind_original
from quactlize.dispatch.dense_prefill import validate as validate_policy
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

SCHEMA='quactlize.dense-m-span.v1'
MS=(128,256,512,1024,2048,3072,4096,5120,8192)
MAX_CASES=48
CAPTURE_BATCH_LIMIT=16


def retry_plan(original, review):
    """New immutable run for incomplete points; retain complete old points.

    Keep every original candidate and both capture directions for each retried
    point. This is not a source-hash bypass or a mixture of old/new component
    timings. The final reviewer replaces each completed point atomically.
    """
    validate(original)
    if (review.get('schema')!='quactlize.prefill-component-review.v1' or
            review.get('review_sha256')!=digest({k:v for k,v in review.items() if k!='review_sha256'})):
        raise ValueError('retry requires an unchanged all-worker component review')
    if original.get('retry'):
        raise ValueError('supply the original complete M-span plan, not an earlier retry plan')
    specs={p['id']:p for p in original['points']}
    complete=review['points'];partial={}
    for p in review['partial_points'].values():
        key=p['point_id']
        if key in complete:continue
        if key in partial:raise ValueError('ambiguous partial authorities for point '+key)
        partial[key]=p
    if (set(complete)|set(partial)!=set(specs) or
            set(review['pending_points'])!=set(partial)):
        raise ValueError('retry review does not cover the original all-worker workload')
    if not partial:raise ValueError('no incomplete points to retry')
    for key,p in {**complete,**partial}.items():
        spec=specs[key]
        if p['workload']!=spec['workload'] or p['authority'] not in review['authorities']:
            raise ValueError('retry workload/authority changed: '+key)
        if digest(review['authorities'][p['authority']])!=p['authority']:
            raise ValueError('retry original authority changed: '+key)
        expected={c['id']:c for c in spec['cases']}
        rows=[r for r in p['measurements'] if r['case_id']!='native-fp16']
        measured={r['case_id'] for r in rows}
        rejected={r['case']['id'] for r in p.get('resource_unavailable',[])}
        missing=set(p.get('missing',[]))-{'native-fp16'}
        if (len(rows)!=len(measured) or measured&missing or rejected&(measured|missing) or
                measured|missing|rejected!=set(expected)):
            raise ValueError('retry original config denominator changed: '+key)
        for row in rows:
            identity=case_identity(expected[row['case_id']])
            if row['candidate']!=digest(identity) or review['candidates'].get(row['candidate'])!=identity:
                raise ValueError('retry original recipe/helper identity changed: '+key)
        if key in complete and any(r['status']!='PASS' for r in p['measurements']):
            raise ValueError('unpassed row cannot be retained as complete: '+key)
    result=copy.deepcopy(original)
    result['points']=[p for p in result['points'] if p['id'] in partial]
    for p in result['points']:
        p['native_comparison']=partial[p['id']].get('native_comparison',True)
    result['reuse']=[dict(id=key,authority=p['authority'],evidence_sha256=digest(p),
                          status='RETAIN_ORIGINAL_COMPLETE_POINT') for key,p in sorted(complete.items())]
    result['retry']=dict(original_plan_sha256=original['sha256'],review_sha256=review['review_sha256'],
        original_points=len(specs),retained_points=len(complete),retried_points=len(partial),
        scope='INCOMPLETE_POINTS_ONLY_ALL_ORIGINAL_CONFIGS_BOTH_CAPTURES',
        old_measurements_unchanged=True)
    result['counts']=counts(result['points'])
    result['sha256']=digest({k:v for k,v in result.items() if k!='sha256'})
    return validate(result)


def tc_case(recipe, io=3):
    return dict(arm=recipe.route,helper_mode='cohort-current',
                helper_options=dict(vector_io=io),recipe=json.loads(json.dumps(asdict(recipe))))


def sf_rules(w, recipes, compute_units):
    """Unique observed parent geometries, ordinary + bounded persistent/Split-K.

    This function has no measured costs. It does not select a nearest winner
    and call that a blind rule. No cross-format performance transfer.
    """
    pool={}
    for value in recipes.values():
        if value['route']!=1 or value['tc']['qtype']!=w.qtype:
            continue
        p=dict(value['tc'])
        p.pop('dense_reducer',None)
        # Preserve real symbols for DN/warp/stage geometry; compact reduction
        # is an execution choice, not a distinct S1 mainloop.
        p['symbol']=p['symbol'].split('_rc')[0]
        p.update(split=1,grid_mode=0,grid_b=0,dequant_config=4 if w.qtype==12 else 0)
        r=Recipe.make('tc','sf',**p)
        if not rejection(r,w):pool[r.key]=r
    if w.qtype in range(10,15) and not any(dict(r.parameters)['tm']>=64 for r in pool.values()):
        # A missing large-tile observation (notably Q3) must not limit the
        # search to the old light SF parent. Re-instantiate two generic SF
        # K-quant geometries, with TARGET format and no borrowed latency.
        for tm,tn in ((64,128),(128,256)):
            p=dict(qtype=w.qtype,tm=tm,tn=tn,tk=256,wm=64,wn=32,stages=2,
                ap=0,dn=16,persistent=-1,split=1,grid_mode=0,grid_b=0,dequant_config=0)
            p['symbol']=f'sf_q{w.qtype}_a0_tm{tm}_tn{tn}_tk256_wm64_wn32_s2_bc0_ap0_dn16'
            r=Recipe.make('tc','sf',**p)
            if not rejection(r,w):pool[r.key]=r
    groups=defaultdict(list)
    for r in pool.values():
        p=dict(r.parameters)
        # Keep M/N/K tile, warp, stages AND delivery distinctions. Only exact
        # parent aliases were removed above; large/heavy is not pruned away.
        groups[(p['tm'],p['tn'])].append(r)
    result=[]
    for key,rs in sorted(groups.items()):
        # At most two K/pipeline alternatives for a tile; deterministic and
        # geometry-only. Record the untested inventory below for later recall.
        rs=sorted(rs,key=lambda r:(dict(r.parameters)['tk'],dict(r.parameters)['stages'],r.key))
        chosen=rs if len(rs)<=2 else [rs[0],rs[-1]]
        for r in chosen:result.append((tc_case(r),'rule:sf-geometry-basis'))
        representative=chosen[-1]
        p=dict(representative.parameters)
        work=math.ceil(w.m/p['tm'])*math.ceil(w.n/p['tn'])
        # Both schedules use probed occupancy at bind time, never a fixed grid.
        for mode in (2,3):
            r=Recipe.make('tc','sf',**(p|dict(grid_mode=mode,grid_b=2)))
            result.append((tc_case(r),'rule:sf-occupancy-grid'))
        if work<2*compute_units:
            for split in (2,4,8):
                r=Recipe.make('tc','sf',**(p|dict(split=split,grid_mode=0,grid_b=0)))
                if not rejection(r,w):result.append((tc_case(r),'rule:sf-underfilled-split'))
    return result, dict(legal_parents=len(pool),tile_strata=len(groups),
        proposed=len(result),screen='GEOMETRY_ONLY_NO_COST_OR_WINNER_INPUT',
        actual_resources='REQUIRED_COMPILED_QUERY_NOT_ESTIMATED')


def make_plan(policy, helper_sha, output, *, ms=MS, compute_units=72):
    validate_policy(policy)
    if (not ms or len(set(ms))!=len(ms) or any(type(m)is not int or not 128<=m<=8192 for m in ms)
            or type(compute_units)is not int or compute_units<=0):
        raise ValueError('distinct M128..8192 and positive device CU count required')
    families=sorted({(p['shape'][0],*p['shape'][2:]) for p in policy['points']})
    points=[dict(id=w.key,workload=asdict(w),consumers=[],tensors=[])
            for q,n,k in families for m in ms for w in [Workload(q,n,k,m)]]
    bind_original(points,output)
    for point in points:
        w=Workload(**point['workload']);pool={}
        def add(case, role, *, required=False):
            r=Recipe.from_dict(case['recipe'])
            if rejection(r,w):
                if required:raise ValueError('required M-span control illegal: '+w.key+' '+r.key)
                return
            cid=digest(case)
            c=pool.setdefault(cid,dict(copy.deepcopy(case),id=cid,roles=[]))
            if role not in c['roles']:c['roles'].append(role)
            if required and 'required-control' not in c['roles']:c['roles'].append('required-control')
        original=Recipe.from_dict(point['original_recipe'])
        if original.kind=='tc':
            legacy=point['original_selector'].get('policy')==17
            p=dict(original.parameters)
            if legacy:p['legacy_adapters']=True
            # The actual table-miss control uses its unchanged row/flat casts,
            # not an improved helper attached to an old tile.
            add(tc_case(Recipe.make('tc',original.route,**p),-1),'old-selector-control',required=True)
        else:
            add(dict(arm='full-bf16',helper_mode='legacy',recipe=asdict(original)),
                'old-selector-control',required=True)
        coverage,incumbent=policy_lookup(policy,w)
        if incumbent:
            add(execution_case(policy['recipes'][incumbent]),'current-whole-call-control',required=True)
            add(execution_case(policy['recipes'][incumbent]),'model-config',required=True)
        for c in pool.values():
            if 'old-selector-control' in c['roles']:
                c['roles'].append('legacy-model-config')
                if not incumbent:c['roles'].append('model-config')
        donors=sorted((p for p in policy['points'] if p['shape'][0]==w.qtype and
            p['shape'][2:]==[w.n,w.k]),key=lambda p:(abs(math.log2(p['shape'][1]/w.m)),p['key']))
        for p in donors[:2]:
            add(execution_case(policy['recipes'][p['winner']]),'nearby-M-winner-control',required=True)
            # Protect the closest per-route champions even if the blind
            # geometric shortlist has missed their stage/warp/DN combination.
            if p is donors[0]:
                for route in (0,1,2):
                    rows=[(rid,us) for rid,us in p['costs'] if policy['recipes'][rid]['route']==route]
                    if rows:
                        rid=min(rows,key=lambda x:(x[1],x[0]))[0]
                        add(execution_case(policy['recipes'][rid]),'nearby-route-control',required=True)
        proposed,report=sf_rules(w,policy['recipes'],compute_units)
        for c,role in proposed:add(c,role)
        variants=(38,41) if w.qtype==8 else (22,23) if w.qtype==11 else (20,21) if w.qtype>=12 else (0,)
        for dq in variants:add(full_case(dq,3),'rule:full-f16')
        add(full_case(0,0),'full-f16-functional-floor',required=True)
        if len(pool)>MAX_CASES:
            raise ValueError(f'{w}: {len(pool)} cases exceed {MAX_CASES}; no controls silently dropped')
        cases=sorted(pool.values(),key=lambda c:digest([w.key,c['id']]))
        point.update(cases=cases,candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],
            champion_controls=[],fixed_schedules=True,confirmation_captures=2,m_span=True,
            capture_batch_limit=CAPTURE_BATCH_LIMIT,
            table_coverage=coverage,proposal_compute_units=compute_units,sf_screen=report,
            # Resources come from cases (which include the exact control), not
            # a duplicate unwrapped legacy-native endpoint.
            original_recipe=None)
    result=dict(schema=SCHEMA,points=points,reuse=[],confirmation_captures=2,max_cases=MAX_CASES,
        inputs=dict(policy_sha256=policy['sha256'],helper_policy_sha256=helper_sha),
        ms=list(ms),counts=counts(points),production_changed=False,performance_admitted=False,
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION; LEGACY_FULL_BF16_CONTROL_LABELLED_SEPARATELY',
        scope='DENSE_MODEL_AND_TP_LOCAL_FAMILIES_M_EXTENSION_NOT_MOE_OR_MODEL_TPOT',
        historical_timing_used='PROPOSALS_AND_CONTROLS_ONLY_NOT_IMPORTED_LATENCY',
        pending=['M9_TO_127_NATIVE_PRECISION_CONTRACT','BUCKET_HOLDOUT_VALIDATION','MODEL_AB'],
        no_fq_fallback='FQ_RETAINED_AS_CONTROL_ONLY; NO_PRODUCTION_POLICY_EDIT')
    result['sha256']=digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'})
            or plan.get('production_changed') is not False or plan.get('performance_admitted') is not False
            or plan.get('confirmation_captures')!=2 or plan.get('max_cases')!=MAX_CASES):
        raise ValueError('M-span identity/contract changed')
    seen=set()
    for p in plan['points']:
        w=Workload(**p['workload'])
        if (w.key!=p['id'] or w.key in seen or w.operation!='dense' or w.compute!='f16' or w.storage!='f32'
                or w.m not in plan['ms'] or not 128<=w.m<=8192 or not p.get('m_span')
                or p.get('confirmation_captures')!=2 or not p.get('fixed_schedules')
                or p.get('capture_batch_limit')!=CAPTURE_BATCH_LIMIT):
            raise ValueError('M-span workload changed')
        seen.add(w.key);ids=set();roles=set()
        for c in p['cases']:
            r=Recipe.from_dict(c['recipe'])
            if c['id']!=digest(case_identity(c)) or c['id'] in ids or rejection(r,w):
                raise ValueError('M-span case changed or illegal')
            ids.add(c['id']);roles.update(c['roles'])
            if r.kind=='tc':
                if c['helper_mode']!='cohort-current' or c['helper_options'].get('vector_io') not in (-1,1,2,3,4):
                    raise ValueError('M-span TC adapters changed')
            elif c['arm']=='full-fp16':
                o=c['helper_options']
                if case_identity(c)!=full_case(o['dequant_variant'],o['vector_io'],
                        dequant_first=dict(r.parameters).get('dequant_first',False)):
                    raise ValueError('M-span Full-F16 arithmetic changed')
            elif (c['arm']!='full-bf16' or c['helper_mode']!='legacy' or
                    'old-selector-control' not in c['roles'] or dict(r.parameters).get('provider')!='bf16-default'):
                raise ValueError('unexpected M-span route')
        if (not 3<=len(ids)<=MAX_CASES or not {'old-selector-control','full-f16-functional-floor'}<=roles or
                not any(c['arm']=='sf' for c in p['cases']) or
                p['candidates']!=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc']):
            raise ValueError('M-span required baseline/config coverage changed')
    if not seen or plan['counts']!=counts(plan['points']):raise ValueError('M-span denominator changed')
    return plan
