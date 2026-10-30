"""Selected projection vs native, same-card fresh ABBA and separate components."""
from collections import Counter
import csv
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from dev.native_components import selected_gate as gate
from dev.native_components.asys import parse as parse_native
from dev.native_components.build import build_adapter
from dev.native_components.fixture import activation,weights
from dev.strategy.journal import atomic_json
from dev.strategy.moe_prefill_search import compose
from dev.strategy.prefill_fixture import routing
from dev.strategy.fixture_identity import expected as expected_fixture
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base,run_moe_prefill_sweep as engine
from tools.run_kpack_llama_ab import point_request
from tools.run_native_components import carrier,environment,validated
from tools.run_prefill_model_asys import profile_command
from tools.run_smallm_regressions_asys import execute,sealed

ROOT=Path(__file__).resolve().parents[2]
ENTRY=ROOT/'tools/run_moe_confirmation.py'


def request(p,l2_bytes):
    # This entry compares MoE prefill projections. A candidate recipe does
    # not carry the native carrier's fixture metadata (notably Full arms).
    row=point_request(dict(p,cohort='prefill'),l2_bytes,routing_repeat=0)
    row['input_fixture']=activation(row)
    return row


def retry_plan(previous,route='selected'):
    """Import confirmed recipes only; all native/candidate timings are fresh."""
    from tools.run_moe_confirmation import completed
    from dev.moe_confirm import plan as confirmation
    if route not in ('selected','sf','full'):raise ValueError('unknown native comparison route')
    previous=Path(previous).resolve(strict=True)
    old=sealed(previous/'run.json');original=sealed(previous/'plan.json')
    chosen=sealed(previous/'selection.json')
    finals=sealed(previous/'finalists.json')
    if (old['mode']!='refine-prefill' or old['plan']!=original['sha256'] or
            chosen['run']!=old['sha256'] or finals['run']!=old['sha256'] or not old['native_comparison']):
        raise ValueError('native continuation requires a completed refinement selection')
    ids={p['id'] for p in original['points']}
    if set(chosen['points'])!=ids or set(finals['points'])!=ids:raise ValueError('selected point coverage differs')
    profiles={};groups={}
    for worker,pids in enumerate(old['assignments']):
        for pid in pids:
            path=previous/('gpu-'+old['cards'][worker])/'points'/(pid+'.json')
            value=completed(path,old)
            if value is None:raise ValueError('missing confirmed point: '+str(path))
            profiles[pid]=value
            groups.setdefault(confirmation.key(value['workload']),[]).append(value)
    if set(profiles)!=ids:raise ValueError('confirmed point coverage differs')
    replacements={}
    for group in groups.values():
        winner=confirmation.choose(group)
        for value in group:
            p=chosen['points'][value['point']]
            if (len(p['arms'])!=1 or p['arms'][0]['id']!='original' or
                    p['workload']!=value['workload'] or
                    digest(p['arms'][0]['recipe'])!=digest(confirmation.canonical(value['bindings'][winner])) or
                    p.get('selection_source')!='MEASURED_REFINEMENT_NOT_PRODUCTION'):
                raise ValueError('native continuation recipe differs from confirmation')
        if route!='selected':
            restricted=[dict(v,decision=dict(statistics={aid:s for aid,s in v['decision']['statistics'].items()
                if aid=='original' or v['bindings'][aid]['recipe']['route']==route})) for v in group]
            winner=confirmation.choose(restricted)
            bindings=[v['bindings'][winner] for v in group]
            if any(b!=bindings[0] for b in bindings) or bindings[0]['recipe']['route']!=route:
                raise ValueError('no common confirmed non-regressing recipe for route '+route)
            for v in group:
                p=finals['points'][v['point']];binding=v['bindings'][winner]
                arm=next(a for a in p['arms'] if a['id']==winner)
                replacements[p['id']]=dict(p,
                    arms=[dict(arm,id='original',roles=['original'],recipe=confirmation.canonical(binding))],
                    sf_configs=[binding['helper_config']] if route=='sf' else [],
                    full_configs=[binding['helper_config']] if route=='full' else [],
                    selection_source='MEASURED_REFINEMENT_NOT_PRODUCTION')
    points=[replacements.get(p['id'],chosen['points'][p['id']]) for p in original['points']]
    modules={a['module']:a['spec'] for p in points for a in p['arms'] if a['recipe']['kind']=='tc'}
    return dict(schema=original['schema'],points=points,tc=modules,vectors={},native_only=True,
        native_comparison=True,historical_times_reused=False,production_changed=False,
        counts=dict(points=len(points),arms=len(points),maximum_candidates=1,tc_modules=len(modules),
                    vector_modules=0,native_passes=2*len(points),candidate_passes=2*len(points)),
        origin=dict(output=str(previous),run=old['sha256'],plan=original['sha256'],
                    selection=chosen['sha256'],source_revision=old['source_revision'],requested_route=route),
        protocol=dict(order=list(gate.ORDER),rounds=6,samples=5,helper_reuse=False),
        scope='CONFIRMED_RECIPES_FRESH_NATIVE_ABBA_NO_RESWEEP_OR_PRODUCTION_CHANGE')


def run_retry(a,plan,run):
    from dev.moe_confirm.refine import spawn
    base.sealed(a.output/'selection.json',dict(run=run['sha256'],origin=plan['origin'],
        points={p['id']:p for p in plan['points']},historical_times_reused=False))
    codes=spawn(a,run,'native',time.monotonic()+a.budget_hours*3600)
    rc=review(a,plan,run,codes)
    result=base.read(a.output/'native-summary.json')
    atomic_json(a.output/'summary.json',dict(result,origin=plan['origin'],
        scope=plan['scope'],historical_times_reused=False,production_changed=False))
    return rc


def build(a,output):
    a.llama_source=a.llama_source.resolve(strict=True);a.llama_build=a.llama_build.resolve(strict=True)
    output.mkdir()
    adapter,receipt=build_adapter(a.llama_source,a.llama_build,a.sdk,output)
    binary=carrier(a,output)
    runtime=next((p.resolve() for p in (a.sdk/'lib/libhggc_wrapper.so',
        a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if p.is_file()),None)
    if runtime is None:raise ValueError('native SDK runtime missing')
    return dict(adapter=str(adapter),adapter_sha256=sha(adapter),carrier=str(binary),carrier_sha256=sha(binary),
        runtime=str(runtime),llama_head=receipt['identity']['llama_head'],backend=receipt['identity']['libraries'],
        llama_build=str(a.llama_build),
        immutable_files={str(p):sha(p) for p in (adapter,binary,runtime,output/'adapter.json',output/'carrier.json')}|
                        receipt['identity']['libraries'])


def native_pass(a,run,p,folder,block):
    n=run['native'];attempt=folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    auth=dict(run=run['sha256'],device=run['devices'][a.worker],adapter=n['adapter_sha256'],
        carrier=n['carrier_sha256'],backend=n['backend'],source_revision=run['source_revision'],
        harness=run['harness']['sha256'],timing_method='ASYS_GPU_ACTIVITY')
    row=request(p,run['l2_bytes'])
    w=p['workload'];family=digest([w[k] for k in ('qtype','n','k','experts')])
    raw,receipt=weights(w,folder/'fixtures'/family)
    atomic_json(attempt/'request.json',dict(runtime_library=n['runtime'],adapter_library=n['adapter'],
        authority=auth,rounds=6,samples=5,timing_method='ASYS_GPU_ACTIVITY',points=[row],
        raw_fixture=str(raw),raw_fixture_receipt=receipt,timing_input_repeat=0))
    # Preserve the worker's masked physical card, not the coordinator's env.
    env=environment(SimpleNamespace(sdk=a.sdk,llama_build=Path(n['llama_build'])))
    env.pop('GGML_CUDA_CUBLAS_ACCUMULATOR',None)
    pci=run['devices'][a.worker]['device']['pci']
    before=base.observe(Path(run['smi']),pci,1700,advisory=True)
    execute(profile_command(run['asys'],attempt/'trace',[n['carrier'],str(attempt/'request.json'),str(attempt)]),
            attempt/'profile.log',env=env)
    after=base.observe(Path(run['smi']),pci,1700,advisory=True)
    report,db=attempt/'trace.asysrep',attempt/'trace.sqlite'
    execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
    capture=attempt/(p['id']+'.json')
    provenance=dict(run['source_revision'],source_sha256=run['source']['sha256'],
        harness_sha256=run['harness']['sha256'],device=run['devices'][a.worker],sdk=run['sdk'],
        asys_sha256=run['asys_sha256'],timing_method='ASYS_GPU_ACTIVITY')
    data=parse_native(base.read(capture),p,auth,db,report,provenance)
    parsed=attempt/'parsed'/(p['id']+'.json');atomic_json(parsed,data);data=validated(parsed,auth)
    gate.validate_native_samples(data)
    return dict(arm='native',block=block,data=data,components=[dict(c,node={k:c[k] for k in
        ('symbol','grid','block','shared_bytes')}) for c in data['components']],
        raw_sha256=data['fixture']['raw_sha256'],timed_router_sha256=row['ring']['router_sha256'],
        timing_input_repeat=data['timing_input_repeat'],timed_fixture=data['timed_fixture'],
        clocks=dict(before=before,after=after),
        evidence={str(path):sha(path) for path in (capture,parsed,report,db,attempt/'request.json')})


def candidate_pass(a,run,p,queries,block):
    _,value=engine.invocation(a,run,dict(operation='measure',point=p,phase='confirm',
        arms=['original'],queries=queries,fresh_components=True),profile=True,entrypoint=ENTRY)
    data=value['data'];composed=compose(p,[data],explicit_helpers=True)
    parts=[composed['component_records'][cid] for cid in composed['arm_components']['original']]
    core=data['arms']['original']
    refined=run.get('mode')=='refine-prefill'
    if not refined and core['receipt'].get('selection')!='PUBLIC_CPP_POLICY22':
        raise ValueError('candidate bypassed public selector')
    if refined and p.get('selection_source')!='MEASURED_REFINEMENT_NOT_PRODUCTION':
        raise ValueError('unbound refinement candidate')
    components=[dict(c,role=c['kind'],node=c['identity']['node']) for c in parts]
    if composed['reset'] is not None:
        components.append(dict(composed['reset'],role='error_reset',node=dict(kind='memset',bytes=4,value=0)))
    return dict(arm='quactlize',block=block,data=value,components=components,
        raw_sha256=core['fixture']['fixture']['raw_sha256'],timed_router_sha256=core['timed_router_sha256'],
        timing_input_repeat=core['timing_input_repeat'],timed_fixture=core['timed_fixture'],
        evidence=value['evidence'])


def comparison(p,passes):
    if len(passes)!=4 or any(v['block']!=i for i,v in enumerate(passes)):
        raise ValueError('incomplete ABBA point')
    if any(v['raw_sha256']!=passes[0]['raw_sha256'] for v in passes):raise ValueError('native/selected GGUF bytes differ')
    expected=routing(Workload(**p['workload']),0)['sha256']
    if any(v['timed_router_sha256']!=expected for v in passes):raise ValueError('native/selected IDs differ')
    fixture=expected_fixture(Workload(**p['workload']),0)
    if any(v['timing_input_repeat']!=0 or v['timed_fixture']!=fixture for v in passes):
        raise ValueError('native/selected F32 input or IDs differ')
    if passes[0]['data']['native_activation_contract']!=passes[3]['data']['native_activation_contract']:
        raise ValueError('native arithmetic changed within ABBA')
    result=gate.verdict(passes)
    result.update(point=p['id'],workload=p['workload'],scope='SELECTED_MOE_PROJECTION_COMPONENT_AB_NOT_FUSED_FFN_OR_MODEL',
        candidate_arithmetic='F32_IO_BF16_OPERANDS_FP32_ACC',arithmetic_equivalence_claimed=False,
        native_activation_contract=passes[0]['data']['native_activation_contract'],model_tpot_admitted=False)
    if p.get('selection_source')=='MEASURED_REFINEMENT_NOT_PRODUCTION':
        result.update(scope='EXPLICIT_REFINEMENT_NATIVE_AB_NOT_PRODUCTION_OR_MODEL',
                      candidate_recipe=p['arms'][0]['recipe'],production_changed=False)
    return result


def saved(path,run):
    if not path.exists():return None
    value=sealed(path)
    if value['run']!=run['sha256']:raise ValueError('comparison belongs to another run')
    if value['status'] not in ('PASS','REGRESSION'):return None
    for name,want in value['evidence'].items():
        if sha(name)!=want:raise ValueError('comparison evidence changed: '+name)
    return value


def point(a,run,p,deadline):
    folder=a.output/('gpu-'+run['cards'][a.worker])
    directory='native-points' if run.get('mode')=='refine-prefill' else 'points'
    path=folder/directory/(p['id']+'.json')
    prior=saved(path,run)
    if prior:
        if directory=='native-points' and prior['candidate_recipe']!=p['arms'][0]['recipe']:
            raise ValueError('confirmed native candidate changed')
        return True  # A valid regression is retained, not sampled until lucky.
    if time.monotonic()>=deadline:return False
    print('MOE_SELECTED_NATIVE_START '+json.dumps(dict(point=p['id'],workload=p['workload'],order=gate.ORDER)),flush=True)
    _,resource=engine.invocation(a,run,dict(operation='resources',point=p,phase='confirm'),profile=False,entrypoint=ENTRY)
    p=resource['point'];passes=[];evidence={}
    for block,arm in enumerate(gate.ORDER):
        destination=folder/'abba'/p['id']/(str(block)+'.json')
        old=engine.saved(destination,run)
        if old is None:
            if time.monotonic()>=deadline:return False
            print('MOE_SELECTED_NATIVE_PASS '+json.dumps(dict(point=p['id'],block=block+1,total=4,arm=arm)),flush=True)
            value=(native_pass(a,run,p,folder,block) if arm=='native' else
                   candidate_pass(a,run,p,resource['queries'],block))
            base.sealed(destination,dict(value,status='PASS',run=run['sha256'],point=p['id'],
                                       candidate_recipe=p['arms'][0]['recipe']))
            old=sealed(destination)
        if old['arm']!=arm or old['block']!=block or old['point']!=p['id']:raise ValueError('saved ABBA binding differs')
        if directory=='native-points' and old['candidate_recipe']!=p['arms'][0]['recipe']:
            raise ValueError('saved ABBA candidate changed')
        passes.append(old);evidence[str(destination)]=sha(destination);evidence.update(old['evidence'])
    result=comparison(p,passes)
    base.sealed(path,dict(result,run=run['sha256'],evidence=evidence))
    print('MOE_SELECTED_NATIVE_POINT '+json.dumps({k:v for k,v in result.items() if k!='passes'}),flush=True)
    return True


def review(a,plan,run,codes):
    refined=run.get('mode')=='refine-prefill'
    rows=[]
    for worker,ids in enumerate(run['assignments']):
        for pid in ids:
            value=saved(a.output/('gpu-'+run['cards'][worker])/('native-points' if refined else 'points')/(pid+'.json'),run)
            rows.append(value or dict(point=pid,status='MISSING_OR_FAILED'))
    counts=Counter(r['status'] for r in rows)
    status='INCOMPLETE' if any(codes) or counts['MISSING_OR_FAILED'] else 'REGRESSION' if counts['REGRESSION'] else 'PASS'
    summary=dict(status=status,counts=dict(counts),expected=len(rows),returncodes=codes,
        source_revision=run['source_revision'],native_no_regression_admitted=status=='PASS',
        model_tpot_admitted=False,scope='SELECTED_MOE_PROJECTION_COMPONENT_AB_NOT_FUSED_FFN_OR_MODEL')
    if refined:summary['scope']='EXPLICIT_REFINEMENT_NATIVE_AB_NOT_PRODUCTION_OR_MODEL'
    atomic_json(a.output/('native-summary.json' if refined else 'summary.json'),summary)
    fields=['point','status','native_us','quactlize_us','delta_pct','native_main_us','quactlize_main_us',
            'native_helper_us','quactlize_helper_us','native_reducer_us','quactlize_reducer_us']
    with (a.output/('native-comparison.csv' if refined else 'comparison.csv')).open('w') as stream:
        writer=csv.DictWriter(stream,fields+['workload']);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row[k]) if k=='workload' and k in row else row.get(k,'') for k in fields+['workload']})
    print('MOE_SELECTED_NATIVE_DONE '+json.dumps(summary),flush=True)
    return int(status!='PASS')
