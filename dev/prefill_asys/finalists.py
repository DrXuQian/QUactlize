"""Bounded whole-call confirmation from measured recipes, not another sweep.

Historical measurements prune candidates; they never supply a new timing.
Keep native, the prior per-route winners and the current Full helper. Q8's
two accepted helper finalists remain separate until measured in full context.
"""
from collections import defaultdict
from dataclasses import asdict
import copy
import json
import math
import statistics

from dev.strategy.resources import inventory, module_spec, probe_spec
from dev.strategy.space import resolved_grid
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.prefill-route-finalists.v1'
Q8_EVIDENCE = '3b06a3a626d56e78938e5689554bef6c2c498b8b97dee202fdb916e6b7571af9'


def case_identity(case):
    return {k:v for k,v in case.items() if k not in ('id','roles','module','launch')}


def full_case(dequant, io):
    options = dict(dequant_variant=dequant, vector_io=io)
    recipe = Recipe.make('full', 'full', provider='typed-fp16', **options)
    return dict(arm='full-fp16', helper_mode='experimental', helper_options=options,
                recipe=json.loads(json.dumps(asdict(recipe))))


def spread(row):
    values = row['rounds_us']
    if len(values)<3 or any(not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('invalid historical round samples')
    return 100*(max(values)-min(values))/statistics.median(values)


def make_plan(review, helpers, *, noise_pct=5.0):
    from tools.review_prefill_cohort import eligible
    if (review.get('schema')!='quactlize.prefill-component-review.v1' or
            review.get('review_sha256')!=digest({k:v for k,v in review.items() if k!='review_sha256'})):
        raise ValueError('historical review schema/digest differs')
    if review.get('pending_points') or review.get('incomplete'):
        raise ValueError('resolve incomplete historical points before finalist closeout')
    if not math.isfinite(noise_pct) or noise_pct<=0:
        raise ValueError('positive noise threshold required')
    policy = {}
    for r in helpers['rows']:
        if r['storage']!=0: continue
        key = (r['q'],r['n'],r['k'],r['m'])
        if key in policy: raise ValueError('duplicate Full-F16 helper policy key')
        policy[key] = full_case(r['dequant'],r['io'])
    points, reuse = [], []
    for key, old in sorted(review['points'].items()):
        w = Workload(**old['workload'])
        if key!=w.key or w.operation!='dense' or w.compute!='f16' or w.storage!='f32':
            raise ValueError('not a dense F32/F16 cohort point')
        current = policy[(w.qtype,w.n,w.k,w.m)]
        valid = [r for r in old['measurements'] if eligible(r)]
        native = [r for r in old['measurements'] if r['arm']=='native-fp16' and r['status']=='PASS']
        if not valid or len(native)!=1: raise ValueError('missing historical native/config evidence')
        for row in valid:
            if (row['candidate']!=digest(review['candidates'][row['candidate']]) or
                    not math.isfinite(row['gpu_sum_us']) or row['gpu_sum_us']<=0):
                raise ValueError('invalid historical candidate identity/timing')
        best = min(valid,key=lambda r:(r['gpu_sum_us'],r['candidate']))
        reasons = []
        if digest(current) not in {r['candidate'] for r in valid}:
            reasons.append('CURRENT_FULL_HELPER_NOT_IN_ROUTE_COHORT')
        if w.qtype==8: reasons.append('Q8_ACCEPTED_HELPERS_NEED_FULL_CONTEXT')
        if max(spread(best),spread(native[0]))>noise_pct: reasons.append('PRIOR_ROUND_SPREAD')
        if not reasons:
            reuse.append(dict(id=key,workload=asdict(w),previous_winner=best['candidate'],
                              evidence_sha256=digest(old),status='RETAIN_HISTORICAL_WHOLE_CALL'))
            continue
        pool = {}
        def add(value, role):
            identity = copy.deepcopy(value); cid=digest(identity)
            c = pool.setdefault(cid,dict(identity,id=cid,roles=[]))
            if role not in c['roles']: c['roles'].append(role)
        add(review['candidates'][best['candidate']], 'prior-point-winner')
        controls = []
        routes = ('sf','full-fp16') if w.qtype==8 else ('fq','sf','full-fp16')
        for arm in routes:
            rows = [r for r in valid if r['arm']==arm]
            if not rows: raise ValueError('missing historical route: '+arm)
            winner = min(rows,key=lambda r:(r['gpu_sum_us'],r['candidate']))
            add(review['candidates'][winner['candidate']], 'prior-route-winner:'+arm)
            controls.append(winner['candidate'])
        add(current, 'current-full-helper')
        if w.qtype==8:
            for v in (38,41): add(full_case(v,current['helper_options']['vector_io']), 'q8-helper-finalist')
        cases=sorted(pool.values(),key=lambda c:digest([key,c['id']]))
        points.append(dict(id=key,workload=asdict(w),consumers=old['consumers'],tensors=[],
            reasons=reasons,cases=cases,confirmation_captures=2,fixed_schedules=True,
            candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],champion_controls=[],
            prior_winner=best['candidate'],prior_route_winners=controls,
            absent_routes=['fq: NOT_PRESENT_IN_ORIGINAL_Q8_COHORT'] if w.qtype==8 else [],
            evidence_sha256=digest(old)))
    result=dict(schema=SCHEMA,points=points,reuse=reuse,
        inputs=dict(review_sha256=review['review_sha256'],helper_policy_sha256=digest(helpers),
                    q8_helper_archive_sha256=Q8_EVIDENCE),
        noise_pct=noise_pct,confirmation_captures=2,
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        scope='AFFECTED_OR_NOISY_DENSE_PREFILL_FINALISTS_NOT_MODEL_OR_MOE_OR_TP_HOLDOUT',
        historical_timing_used='PRUNING_ONLY; NO_IMPORTED_TIMING_IN_NEW_MEASUREMENTS',
        user_acceptance='OBSERVED_Q8_HELPER_GAPS_ONLY; NOT_A_GLOBAL_REGRESSION_TOLERANCE',
        production_changed=False,performance_admitted=False)
    result['counts']=counts(points)|dict(retained_points=len(reuse),original_points=len(review['points']))
    result['sha256']=digest(result)
    return validate(result)


def counts(points):
    return dict(points=len(points),families=len({(p['workload']['qtype'],p['workload']['n'],
        p['workload']['k']) for p in points}),configs=sum(len(p['cases']) for p in points),
        captures=4*len(points),timed_arms=2*sum(len(p['cases'])+1 for p in points),
        **inventory(dict(points=points))['counts'])


def snapshot(plan):
    """Deduplicate recipes/consumer lists in the handoff, without timing data."""
    validate(plan)
    document=copy.deepcopy(plan);recipes={};consumers={}
    for p in document['points']:
        key=digest(p['consumers']);consumers[key]=p.pop('consumers');p['consumer_key']=key
        for c in p['cases']:
            recipes[c['id']]=case_identity(c)
        p['cases']=[dict(id=c['id'],roles=c['roles']) for c in p['cases']]
        p.pop('candidates')  # Reconstructed exactly, not chosen again on Box.
    return dict(schema=SCHEMA+'.snapshot',plan=document,recipes=recipes,consumers=consumers)


def load(document):
    if document.get('schema') == 'quactlize.q8-tp-prefill-refine.v1':
        from .q8_refine import validate as validate_refinement
        return validate_refinement(document)
    if document.get('schema') == 'quactlize.dense-tp-supplement.v1':
        from .supplement import validate as validate_supplement
        return validate_supplement(document)
    if document.get('schema')==SCHEMA+'.snapshot':
        plan=copy.deepcopy(document['plan'])
        for p in plan['points']:
            p['consumers']=document['consumers'][p.pop('consumer_key')]
            p['cases']=[dict(document['recipes'][c['id']],**c) for c in p['cases']]
            p['candidates']=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc']
        return validate(plan)
    return validate(document)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'})
            or plan.get('confirmation_captures')!=2 or plan.get('production_changed') is not False
            or plan.get('performance_admitted') is not False):
        raise ValueError('finalist plan contract/digest differs')
    keys=set()
    for p in plan['points']:
        w=Workload(**p['workload'])
        if (w.key!=p['id'] or w.key in keys or w.operation!='dense' or w.compute!='f16'
                or w.storage!='f32' or not 128<=w.m<=4096 or not p.get('fixed_schedules')
                or p.get('confirmation_captures')!=2 or not 3<=len(p['cases'])<=6):
            raise ValueError('finalist workload/confirmation/budget differs')
        keys.add(w.key);seen=set();roles=set()
        for c in p['cases']:
            if c['id']!=digest(case_identity(c)) or c['id'] in seen or not c['roles']:
                raise ValueError('duplicate/changed finalist config')
            seen.add(c['id']);roles.update(c['roles']);r=Recipe.from_dict(c['recipe'])
            if r.kind=='tc':
                from dev.strategy.prescreen import rejection
                if c['arm']!=r.route or c['helper_mode']!='cohort-current' or rejection(r,w):
                    raise ValueError('invalid fixed TC finalist')
            elif r.kind=='full':
                o=c['helper_options'];allowed=(0,22,23,36,37,38,41) if w.qtype==8 else (0,8,20,21,22,23)
                if (o['dequant_variant'] not in allowed or o['vector_io'] not in range(4)
                        or case_identity(c)!=full_case(o['dequant_variant'],o['vector_io'])):
                    raise ValueError('Full finalist precision/helper differs')
            else: raise ValueError('only measured prefill TC/Full finalists allowed')
        required={'prior-point-winner','current-full-helper','prior-route-winner:sf','prior-route-winner:full-fp16'}
        if w.qtype!=8: required.add('prior-route-winner:fq')
        if not required<=roles or not {p['prior_winner'],*p['prior_route_winners']}<=seen:
            raise ValueError('a required historical/current control was dropped')
        if p['candidates']!=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc'] or p['champion_controls']:
            raise ValueError('fixed resource inventory differs from planned finalists')
        if w.qtype==8 and not {38,41}<={c.get('helper_options',{}).get('dequant_variant') for c in p['cases']}:
            raise ValueError('Q8 context-dependent finalist missing')
    reused=[p['id'] for p in plan['reuse']]
    if len(set(reused))!=len(reused) or keys & set(reused):raise ValueError('duplicate retained/finalist point')
    if plan['counts']!=counts(plan['points'])|dict(retained_points=len(reused),original_points=len(keys)+len(reused)):
        raise ValueError('finalist denominator/resource counts differ')
    return plan


def partition(points, shard, shards):
    if not 0<=shard<shards<=64:raise ValueError('invalid finalist shard')
    families=defaultdict(list)
    for p in points:
        w=p['workload'];families[(w['qtype'],w['n'],w['k'])].append(p)
    if len(families)<shards:raise ValueError('more GPUs than finalist families')
    def cost(rows):
        return sum((len(p['cases'])+1)*(1+p['workload']['m']*p['workload']['n']*p['workload']['k']/1e10)
                   for p in rows)
    loads=[0.]*shards;owned=[]
    for key,rows in sorted(families.items(),key=lambda item:(-cost(item[1]),item[0])):
        i=min(range(shards),key=lambda i:(loads[i],i));loads[i]+=cost(rows)
        if i==shard:owned.extend(sorted(rows,key=lambda p:p['id']))
    return owned


def bind_cases(point, queries):
    """Resolve only these fixed schedules; unavailable controls are a failure."""
    w=Workload(**point['workload']);cases=copy.deepcopy(point['cases'])
    for c in cases:
        r=Recipe.from_dict(c['recipe'])
        if r.kind!='tc':continue
        spec=probe_spec(w,r,int(dict(r.parameters).get('grid_mode',0)>=2))
        receipt=queries[digest(spec)]
        if receipt.get('request')!=spec or receipt.get('status')!='READY':
            raise ValueError('required finalist resource unavailable: '+c['id'])
        c['module']=digest(module_spec(w,r))
        c['launch']=resolved_grid(w,r,receipt['device']['compute_units'],receipt['occupancy'])
    return cases
