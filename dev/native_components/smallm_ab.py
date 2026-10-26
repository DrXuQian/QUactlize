"""Fresh selected-recipe/native comparison, not the historical component screen."""
from collections import Counter
from dataclasses import asdict
import json
import math
import statistics

from dev.strategy.m_coverage import points
from dev.strategy.refresh import original_recipe, request, family
from dev.strategy.resources import module_spec
from dev.strategy.vector import spec as vector_spec
from dev.strategy.followup import validate_confirmation
from dev.native_components.accounting import checked_component
from quactlize.dispatch.table import Workload, Recipe
from quactlize.dispatch.planning import plan_smallm
from quactlize.runtime.tuning import digest

SCHEMA='quactlize.dense-smallm-native-ab.v1'
ORDER=('native','candidate','candidate','native')
SCOPE='FRESH_SAME_DEVICE_ABBA_ISOLATED_COMPONENT_SUM_NOT_MODEL_TPOT'


def plan(output):
    ps=points()
    rows=plan_smallm(output,[request(Workload(**p['workload'])) for p in ps],dense_fallbacks=True)['requests']
    tc,vectors={},{}
    for p,row in zip(ps,rows):
        if row['status']:raise ValueError('production selection unavailable: '+p['id'])
        w=Workload(**p['workload']);r=original_recipe(row)
        spec=module_spec(w,r) if r.kind=='tc' else vector_spec(w,r)
        key=digest(spec);(tc if r.kind=='tc' else vectors)[key]=spec
        p.update(selection=row,arms=[dict(id='original',recipe=asdict(r),spec=spec,module=key,
            roles=['original'],required=True)],sf_configs=[max(0,dict(r.parameters).get('dequant_config',-1))]
            if r.route=='sf' and w.qtype!=8 else [])
    result=dict(schema=SCHEMA,points=ps,tc=tc,vectors=vectors,
        counts=dict(points=len(ps),families=len({family(Workload(**p['workload'])) for p in ps}),
            selected_configs=len(ps),tc_modules=len(tc),vector_modules=len(vectors),
            native_passes=2,candidate_passes=2),
        protocol=dict(order=list(ORDER),rounds=6,samples=5,requested_clock_mhz=1700,
            clock_deviation='ADVISORY_NOT_FAILURE',producer_ring='AT_LEAST_2P25_L2',
            helpers='ISOLATED_SINGLE_KERNEL_TIMERS',cuBLAS='ORDINARY_PROVIDER_EVENTS_NO_GRAPH'),
        historical_times_reused=False,production_changed=False,scope=SCOPE)
    result['sha256']=digest(result)
    return json.loads(json.dumps(result))


def candidate_components(data,point):
    validate_confirmation(data)
    if data['point']!=point['id'] or data['workload']!=point['workload'] or set(data['arms'])!={'original'}:
        raise ValueError('candidate workload/selected arm differs')
    keys=data['arm_components']['original'];records=data['component_records']
    if not keys or len(keys)!=len(set(keys)) or set(keys)!=set(records):
        raise ValueError('candidate stage inventory incomplete/duplicated')
    return [dict(role=records[k]['kind'],node=records[k]['identity']['node'],
        samples_us=records[k]['samples_us']) for k in keys]


def native_components(data,point):
    if (data.get('correctness')!='PASS' or data['id']!=point['id'] or data['workload']!=point['workload'] or
            data.get('linked_backend_comparison')!='PASS_TWO_INPUTS' or
            data.get('omitted_producer')!='DETECTED' or
            data.get('last_weight_copy_negative')!='DETECTED_COPY0_UNCHANGED'):
        raise ValueError('native numerical/extraction proof incomplete')
    components=data['components']
    if not components or any('quactlize' in c.get('name','').lower() for c in components):
        raise ValueError('native arm missing or executed a Quactlize kernel')
    for c in components:checked_component(c)
    return [dict(role=c['role'],node={k:v for k,v in c.items() if k in
        ('symbol','name','grid','block','shared_bytes','entry','arguments')},samples_us=c['samples_us'])
        for c in components]


def costs(components):
    """Sum independently measured component medians; never a fabricated E2E."""
    if not components:raise ValueError('empty component costs')
    rounds=[0.]*6;terms=[]
    for c in components:
        values=c['samples_us']
        if (len(values)!=6 or any(len(v)<5 for v in values) or
                any(not math.isfinite(x) or x<=0 for v in values for x in v)):
            raise ValueError('need six rounds of at least five finite positive samples')
        medians=[statistics.median(v) for v in values]
        rounds=[x+y for x,y in zip(rounds,medians)]
        terms.append(dict(role=c['role'],node=c['node'],median_us=statistics.median(medians),rounds_us=medians))
    return dict(sum_us=sum(t['median_us'] for t in terms),rounds_us=rounds,components=terms)


def verdict(passes):
    if len(passes)!=4 or tuple(p['arm'] for p in passes)!=ORDER:
        raise ValueError('native/candidate comparison requires complete ABBA, no historical substitution')
    summaries=[costs(p['components']) for p in passes]
    for i,j in ((0,3),(1,2)):
        signature=lambda s:[(c['role'],c['node']) for c in s['components']]
        if signature(summaries[i])!=signature(summaries[j]):
            raise ValueError('kernel/launch inventory changed between paired passes')
    native=statistics.median([summaries[i]['sum_us'] for i in (0,3)])
    candidate=statistics.median([summaries[i]['sum_us'] for i in (1,2)])
    ratios=[summaries[j]['rounds_us'][r]/summaries[i]['rounds_us'][r]
            for i,j in ((0,1),(3,2)) for r in range(6)]
    drift=max(abs(summaries[i]['sum_us']-summaries[j]['sum_us'])/
              min(summaries[i]['sum_us'],summaries[j]['sum_us']) for i,j in ((0,3),(1,2)))
    spread=max((max(s['rounds_us'])/min(s['rounds_us'])-1) for s in summaries)
    status=('INCONCLUSIVE' if max(drift,spread)>.05 else 'PASS' if max(ratios)<=1 and candidate<=native else
            'REGRESSION' if min(ratios)>1 and candidate>native else 'INCONCLUSIVE')
    phases={}
    for label,indices in (('native',(0,3)),('quactlize',(1,2))):
        for name,roles in (('main',{'producer','provider'}),('reducer',{'reducer','reducer_finish'}),
                           ('helper',None)):
            phase=lambda role:role not in {'producer','provider','reducer','reducer_finish'} if roles is None else role in roles
            phases[label+'_'+name+'_us']=statistics.median([
                sum(c['median_us'] for c in summaries[i]['components'] if phase(c['role'])) for i in indices])
    return dict(status=status,native_us=native,quactlize_us=candidate,delta_pct=100*(candidate/native-1),
        paired_round_ratios=ratios,order_drift_pct=100*drift,round_spread_pct=100*spread,
        passes=summaries,allowed_slowdown_pct=0,scope=SCOPE,
        component_no_regression=status=='PASS',model_tpot_admitted=False,production_changed=False,**phases)


def aggregate(plan,records):
    expected={p['id'] for p in plan['points']}
    if set(records)-expected:raise ValueError('unknown comparison point')
    counts=Counter(r['status'] for r in records.values())
    measured=sum(counts[s] for s in ('PASS','REGRESSION','INCONCLUSIVE'))
    return dict(status='COMPLETE' if measured==len(expected) else 'INCOMPLETE',
        expected=len(expected),measured=measured,counts=dict(counts),missing=sorted(expected-set(records)),
        component_no_regression=counts['PASS']==len(expected),model_tpot_admitted=False,
        production_changed=False,scope=SCOPE)
