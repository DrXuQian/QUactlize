"""Current-body small-M recovery. Historical timings never select a winner."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import statistics
import subprocess

from dev.strategy.refresh import proposed, historical, original_recipe, request
from dev.strategy.components import statistics_checked
from dev.strategy.vector import execution_contract
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[2]
METHOD='ASYS_GPU_ACTIVITY'


def workloads():
    return [Workload(q,1024,5120,m) for q,ms in ((13,range(1,5)),(14,range(2,8))) for m in ms]


def make_plan(originals):
    ws=workloads()
    if [r['request'] for r in originals]!=[request(w) for w in ws]:
        raise ValueError('regression C++ selection inventory differs')
    prior={tuple(r['key']):r for r in json.loads((ROOT/'policies/kpack_component_decode_v1.json').read_bytes())['rows']}
    history=historical();points=[];tc={};vectors={}
    for w,old in zip(ws,originals):
        if old['status']:raise ValueError('current production control unavailable')
        p=proposed(dict(id=w.key,workload=asdict(w),cohort='decode',consumers=[]),
                   original_recipe(old),history,prior,allow_new_family=True)
        p.update(selection=old,execution_contract=execution_contract(w))
        for arm in p['arms']:
            (tc if arm['recipe']['kind']=='tc' else vectors)[arm['module']]=arm['spec']
        points.append(p)
    result=dict(schema='quactlize.smallm-asys.v1',phase='regression-asys',points=points,tc=tc,vectors=vectors,
        counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),tc_modules=len(tc),
                    vector_modules=len(vectors),families=2),
        timing_method=METHOD,selection_metric='MEAN_COMPONENT_SUM',noise_policy='DIAGNOSTIC_NOT_VETO',
        historical_times_reused=False,production_changed=False,native_comparison=False,
        scope='TEN_DENSE_REGRESSION_POINTS_CURRENT_CONFIG_SWEEP_NOT_NATIVE_OR_MODEL_AB',
        sources={p:sha(ROOT/p) for p in ('policies/kpack_component_decode_v1.json',
                  'policies/kpack_smallm_effective_v1.hpp','dev/strategy/vector.py')})
    result['sha256']=digest(result)
    return json.loads(json.dumps(result))


def revision(root):
    root=Path(root)
    receipt=root/'.quactlize-source.json'
    if receipt.exists():
        value=json.loads(receipt.read_bytes())
        if set(value)!={'quactlize_commit','quactlize_tree','actlize_commit','snapshot'} or value['snapshot']!='COMMITTED_ARCHIVE':
            raise ValueError('invalid frozen source receipt')
    else:
        def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
        if Path(git('rev-parse','--show-toplevel')).resolve()!=root.resolve():
            raise ValueError('source has neither its own Git checkout nor a frozen commit receipt')
        if git('status','--porcelain','--untracked-files=normal'):
            raise ValueError('commit the sweep sources before a Box measurement')
        value=dict(quactlize_commit=git('rev-parse','HEAD'),quactlize_tree=git('rev-parse','HEAD^{tree}'),
                   actlize_commit=git('rev-parse','HEAD:third_party/actlize'),snapshot='COMMITTED_ARCHIVE')
    if any(not re.fullmatch('[0-9a-f]{40}',value[k]) for k in ('quactlize_commit','quactlize_tree','actlize_commit')):
        raise ValueError('source revision is not a full commit/tree ID')
    return value


def mean_decision(costs,roles):
    if 'original' not in costs or set(costs)!=set(roles):raise ValueError('missing paired current control')
    values={}
    for aid,rounds in costs.items():
        stats=statistics_checked(rounds)
        flat=[v for r in rounds for v in r]
        stats.update(mean_us=statistics.fmean(flat),min_us=min(flat),max_us=max(flat),
                     noise_advisory=stats['spread_pct']>5)
        values[aid]=stats
    winner=min(values,key=lambda a:(values[a]['mean_us'],a!='original',a))
    return dict(status='MEASURED_MEAN_SELECTION',winner=winner,statistics=values,
        metric='ARITHMETIC_MEAN_OF_SEPARATE_COMPONENT_SUMS',noise_policy='DIAGNOSTIC_NOT_VETO',
        delta_pct=100*(values[winner]['mean_us']/values['original']['mean_us']-1),
        native_no_regression_admitted=False,model_tpot_admitted=False)


def check_producers(point,data):
    from dev.strategy.refresh_gaps import producer_entry
    arms={a['id']:a for a in point['arms']}
    unavailable=data.get('resource_unavailable',{})
    if set(data['arms'])&set(unavailable) or set(data['arms'])|set(unavailable)!=set(arms):
        raise ValueError('requested config inventory incomplete')
    for aid,query in unavailable.items():
        from dev.strategy.resources import probe_spec
        from quactlize.dispatch.table import Recipe
        arm=arms[aid];r=Recipe.from_dict(arm['recipe'])
        spec=probe_spec(Workload(**point['workload']),r,int(dict(r.parameters).get('grid_mode',0)>=2))
        if (arm['required'] or r.kind!='tc' or query.get('status')!='UNAVAILABLE' or
                query.get('rc')!=1 or query.get('runtime_status')!=0 or query.get('request')!=spec or
                not query.get('module_sha256')):
            raise ValueError('unavailable recipe lacks a resource rejection proof')
    for aid,measured in data['arms'].items():
        spec=arms[aid]['spec'];recipe=arms[aid]['recipe']
        nodes=[data['component_records'][k]['identity']['node'] for k in data['arm_components'][aid]
               if data['component_records'][k]['kind']=='producer']
        if len(nodes)!=1:raise ValueError('missing/duplicate compute kernel')
        n=producer_entry(nodes[0])
        if recipe['kind']=='tc':
            if n not in ('cutlass::device_kernel','cutlass::Kernel'):
                raise ValueError('TC carrier did not execute CUTLASS compute')
        else:
            c=spec['config']
            common=c['variant']==3 and ((c['columns']==4 and c['values']==8 and c['warps'] in (2,8)) or
                                       (c['columns']==8 and c['values']==4 and c['warps']==4))
            expected='quactlize::execution::simt::'+('common_reader' if common else 'register_reuse')
            if expected!=n:raise ValueError('candidate executed the wrong production reader: '+n)
            if measured['receipt'].get('execution_contract')!=point['execution_contract']:
                raise ValueError('candidate did not submit the planned dense ABI')


def validate(data):
    from dev.strategy.followup import validate_confirmation
    if data.get('timing_method')!=METHOD:raise ValueError('event/host timing is not admitted to Asys selection')
    validate_confirmation(data,decision_verifier=mean_decision)
    intervals=[]
    for r in data['component_records'].values():
        p=r.get('latency_provenance',{})
        if (p!=data.get('latency_provenance') or p.get('timing_method')!=METHOD or
                any(not re.fullmatch('[0-9a-f]{40}',p.get(k,'')) for k in
                    ('quactlize_commit','quactlize_tree','actlize_commit')) or
                any(not re.fullmatch('[0-9a-f]{64}',p.get(k,'')) for k in
                    ('trace_sha256','sqlite_sha256','images_sha256','harness_sha256','source_sha256','asys_sha256')) or
                not p.get('sdk') or not p.get('device')):
            raise ValueError('latency lacks commit/profiler provenance')
        seen=set()
        for item in r.get('trace_slices',[]):
            slot=item['round'],item['sample'];ns=item['durations_ns']
            if (slot in seen or not 0<=slot[0]<6 or not 0<=slot[1]<5 or
                    item['count']!=r['calls_per_graph'] or len(ns)!=item['count'] or
                    any(type(x)!=int or x<=0 for x in ns)):
                raise ValueError('Asys raw component sample coverage differs')
            us=[x/1000 for x in ns];mean=statistics.fmean(us)
            if (mean!=item['mean_us'] or min(us)!=item['min_us'] or max(us)!=item['max_us'] or
                    r['samples_us'][slot[0]][slot[1]]!=mean):
                raise ValueError('component mean differs from raw GPU activity')
            seen.add(slot);intervals.append((item['first_activity'],item['count']))
        if len(seen)!=30 or r.get('mean_us')!=statistics.fmean(x for row in r['samples_us'] for x in row):
            raise ValueError('missing raw Asys samples or invalid mean')
    cursor=0
    for first,count in sorted(intervals):
        if first!=cursor:raise ValueError('Asys activity ranges overlap or have gaps')
        cursor+=count
