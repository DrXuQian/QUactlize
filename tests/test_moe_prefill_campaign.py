"""Host-only bounded search/parse/ownership tests; not device evidence."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import json

import pytest

from dev.strategy import moe_prefill_campaign as campaign
from dev.strategy import moe_prefill_search as search
from dev.strategy.asys_components import Collector
from dev.strategy.resources import probe_spec
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest
from test_moe_prefill_plan import large_inventory
from test_moe_model_plan import synthetic_inventory


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    inventory=large_inventory(synthetic_inventory(shapes=[(12,13,2048,512,256,8),
        (12,13,3072,1024,256,8),(11,8,4096,2048,128,6)]))
    _,value=campaign.make_plan(tmp_path_factory.mktemp('moe-prefill-sweep'),inventory)
    return value


def fake_resources(point,occupancy=8):
    images={'tc':{a['module']:dict(key='build-'+a['module'],sha256='image-'+a['module'])
                    for a in point['arms'] if a['recipe']['kind']=='tc'}}
    queries={qid:dict(request=req['request'],status='READY',rc=0,runtime_status=0,
        build_key=images['tc'][req['module']]['key'],module_sha256=images['tc'][req['module']]['sha256'],
        device=dict(compute_units=72,ordinal=0),occupancy=occupancy,workspace_bytes=128,shared_bytes=1024)
             for qid,req in campaign.query_requests(point).items()}
    return images,queries


def test_bounded_generation_keeps_actual_m_route_coverage_and_historical_recall(plan):
    assert plan['schema']==campaign.SCHEMA and len(plan['points'])==200
    assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
    assert not plan['champion_injection'] and not plan['historical_times_reused']
    assert not plan['production_changed']
    signatures={}
    for p in plan['points']:
        w=p['workload'];key=tuple(w[k] for k in ('qtype','n','k','m','channels'))
        candidates=[a['recipe'] for a in p['arms'] if 'rule' in a['roles']]
        assert signatures.setdefault(key,candidates)==candidates
        assert p['rule_reasons']['blind'] and not p['rule_recall']['missing']
        assert p['rule_reasons']['parents']<=32
        assert p['arms'][0]['id']=='original'
        assert {'fq','sf'}<={r['route'] for r in candidates}
        assert bool(p['full_configs'])==(128<=w['m']<=8192)
        assert p['sf_configs']==[0,4,5]


@pytest.mark.parametrize('occupancy',[1,4,8,12])
def test_resources_keep_historical_actual_launches_and_runtime_grids(plan,occupancy):
    # Check all geometry/M keys once; candidate generation is router blind.
    for p in plan['points']:
        if p['workload']['router']!='spread':continue
        images,queries=fake_resources(p,occupancy)
        bound=campaign.resolve(p,queries,images)
        assert bound['arms'][0]['id']=='original'
        assert len(bound['arms'])<=384
        for a in bound['arms']:
            if a['recipe']['kind']=='tc':
                launch=a['binding']['launch']
                assert launch['grid']<=72*occupancy
        assert bound['seed_sha256']==digest(p)


@pytest.mark.parametrize('fault',['missing','image','occupancy','context','control'])
def test_bad_resource_evidence_cannot_become_a_win(plan,fault):
    point=plan['points'][0];images,queries=fake_resources(point)
    if fault=='missing':queries.pop(next(iter(queries)))
    elif fault=='image':next(iter(queries.values()))['module_sha256']='stale'
    elif fault=='occupancy':next(iter(queries.values()))['occupancy']=0
    elif fault=='context':next(iter(queries.values()))['device']={'compute_units':64,'ordinal':0}
    else:
        old=point['arms'][0];r=Recipe.from_dict(old['recipe']);w=Workload(**point['workload'])
        qid=digest(probe_spec(w,r,int(dict(r.parameters)['grid_mode']>=2)))
        queries[qid].update(status='UNAVAILABLE',rc=1)
    with pytest.raises(ValueError):campaign.resolve(point,queries,images)


def test_assignment_covers_each_point_once_and_never_splits_arms(plan):
    for count in (1,3,5,7):
        groups=campaign.assignments(plan,count)
        flat=[x for g in groups for x in g]
        assert len(flat)==len(set(flat))==len(plan['points'])
        assert all(groups)
        assert set(flat)=={p['id'] for p in plan['points']}


def test_resource_resume_is_stable_after_json_recipe_tuple_roundtrip(plan):
    point=plan['points'][0];images,queries=fake_resources(point)
    resolved=campaign.resolve(point,queries,images)
    saved=json.loads(json.dumps(dict(point=resolved,queries=queries)))
    assert resolved==saved['point']
    assert saved['point']==campaign.resolve(json.loads(json.dumps(point)),saved['queries'],images)


def test_finalists_keep_route_strata_and_variable_round_candidate():
    point=dict(arms=[dict(id=i,recipe=dict(route=r,parameters=[['persistent',s]]))
                    for i,r,s in [('original','fq',0),('best','fq',1),('variable','fq',1),('sf','sf',-1),('full','full',-1)]])
    stats={i:dict(mean_us=v,samples_us=[[v]*3]*4) for i,v in
           [('original',12),('best',10),('variable',15),('sf',20),('full',30)]}
    stats['variable']['samples_us'][0]=[10,10,10]
    assert set(campaign.finalists(point,stats))==set(stats)
    with pytest.raises(ValueError,match='incomplete'):campaign.finalists(point,dict(list(stats.items())[:-1]))


def test_memset_is_gpu_timing_not_api_latency_and_missing_samples_fail():
    c=dict(primer=16,rounds=4,samples=3,repeats=16)
    count=16+4*3*16
    apis=[dict(name='hggcMemsetAsync',return_value=0,correlation_id=i,start=i,end=i+999999) for i in range(count)]
    rows=[dict(kind='memset',bytes=4,value=0,correlation_id=i,start=i*1100,end=i*1100+1000)
          for i in range(1,count)]
    result=search.parse_reset(c,rows,apis)
    assert result['mean_us']==1 and result['samples_us']==[[1]*3]*4
    assert len(result['durations_ns'])==192
    for broken in (rows[:-1],rows+[rows[-1]],rows+[dict(rows[-1],correlation_id=9999)],
                   [dict(r,bytes=8) for r in rows]):
        with pytest.raises(ValueError):search.parse_reset(c,broken,apis)
    with pytest.raises(ValueError):search.parse_reset(None,rows,apis)
    assert search.parse_reset(None,[],[]) is None


def test_main_stage_timers_alternate_arms_and_keep_capture_owner(monkeypatch):
    w=Workload(12,512,2048,128,'indexed','bf16',experts=256,topk=8,router='spread')
    nodes=[dict(name=n,symbol=n,grid=[1,1,1],block=[32,1,1],shared_bytes=0)
           for n in ('mm_ids_helper<8>','quactlize::prefill::gather_bf16','cutlass::device_kernel<GemmUniversal>','quactlize::prefill::finish_bf16')]
    active=[False];events=[];clock=[0];rows=[]
    class Graph:
        rt=SimpleNamespace(sync=lambda:None);count=16
        def __init__(self,n):self.node=n
        def run(self):
            assert active[0]
            events.append(self.node['symbol'])
            for _ in range(16):
                rows.append(dict(self.node,kind='kernel',start=clock[0],end=clock[0]+1000));clock[0]+=1100
        def close(self):assert not active[0]
        def sample(self):pytest.fail('event fallback')
    class Capture:
        groups=[[i] for i in range(4)];descriptions=nodes;rt=Graph.rt
        def isolate(self,index,**kwargs):return Graph(self.descriptions[index])
    monkeypatch.setattr(Collector,'start',lambda self,rt:(setattr(self,'rt',rt),active.__setitem__(0,True)))
    monkeypatch.setattr(Collector,'stop',lambda self:(setattr(self,'rt',None),active.__setitem__(0,False)))
    monkeypatch.setattr(search,'replay_extracted',lambda capture,proof:proof())
    t=search.AlternatingStages(rounds=4,samples=3)
    def proof():assert not active[0]
    for tile in (16,32):
        recipe=Recipe.make('tc','fq',tm=tile,split=1)
        t.collect(Capture(),w,recipe,'test-only','source',proof)
    assert len(t.graphs)==5 # shared prepare/input/output; two main configs
    t.run();t.close()
    assert len(t.collector.schedule)==5*4*3
    got=search.parse_component_rows(t.records,t.collector.schedule,rows,dict(host_test=True),4,3)
    assert all(r['mean_us']==1 for r in got.values())
    # Shared routing remains first; it is the two producers whose order
    # must alternate, not the identical helper shared by both arms.
    producers={k for k,v in t.records.items() if v['kind']=='producer'}
    order=lambda r:[x['component'] for x in t.collector.schedule
                    if x['round']==r and x['sample']==0 and x['component'] in producers]
    assert order(0)==list(reversed(order(1))) and len(order(0))==2


def test_legacy_pilot_default_and_short_screen_are_both_explicit():
    from dev.strategy.moe_prefill_native import AsysStages
    assert (AsysStages().rounds,AsysStages().samples)==(6,5)
    assert AsysStages(rounds=4,samples=3).rounds==4
    with pytest.raises(ValueError):AsysStages(rounds=1,samples=1)


def test_pilot_setup_shared_but_no_automatic_native_or_model_admission():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    text=(root/'tools/run_moe_prefill_sweep.py').read_text()
    assert 'prepare_run(a,plan,script=__file__,assign=campaign.assignments)' in text
    assert "native_no_regression_admitted=False" in text and 'model_tpot_admitted=False' in text
    assert 'deadline=time.monotonic()+a.budget_hours*3600' in text
    assert 'if time.monotonic()>=deadline:return None' in text
    assert 'QUACTLIZE_DENSE' not in text


def test_full_preflight_is_a_real_singleton_batch_and_all_arms_stay_covered():
    from tools.run_moe_prefill_sweep import batches
    point=dict(arms=[dict(id=str(i),recipe=dict(kind='full' if i==5 else 'tc')) for i in range(10)])
    groups=list(batches(point))
    assert groups==[['0','1','2','3'],['4'],['5'],['6','7','8','9']]
    assert [i for g in groups for i in g]==list(map(str,range(10)))


def test_same_sf_producer_keeps_rule_helper_choice_separate_from_control(plan):
    point=deepcopy(plan['points'][0])
    rule=next(a for a in point['arms'] if a['recipe']['route']=='sf' and a['roles']==['rule'])
    control=dict(rule,id='original',roles=['original'],required=True)
    point.update(arms=[control,rule],historical_references=[])
    original=Recipe.from_dict(control['recipe'])
    images,queries=fake_resources(point)
    bound=campaign.resolve(point,queries,images)
    control=bound['arms'][0]
    assert any(a['id']!='original' and a['binding']['launch']==control['binding']['launch'] and
        campaign.scheduled_key(Recipe.from_dict(a['recipe']))==campaign.scheduled_key(original)
        for a in bound['arms'])


def compose_fixture():
    w=Workload(12,512,2048,128,'indexed','bf16',experts=256,topk=8)
    recipes=[Recipe.make('tc','sf',dequant_config=0),Recipe.make('tc','sf',dequant_config=0),
             Recipe.make('full','full',dequant_config=4)]
    point=dict(id=w.key,workload=asdict(w),arms=[dict(id=aid,recipe=asdict(r),roles=[role]) for aid,r,role in
        zip(('original','rule-sf','rule-full'),recipes,('original','rule','rule'))])
    def record(key,v):return dict(component_id=key,mean_us=float(v),samples_us=[[float(v)]*3]*4,
        status='PASS',timing_method='ASYS_GPU_ACTIVITY')
    rows={k:record(k,v) for k,v in [('main',10),('full',9),('sf0',5),('sf4',1),('sf5',2),('dq',3)]}
    value=dict(status='PASS',point=w.key,authority='same-run/card/phase',protocol=dict(rounds=4,samples=3),
        component_records=rows,arms={aid:dict(components=[k]) for aid,k in
        [('original','main'),('rule-sf','main'),('rule-full','full')]},
        expansions=dict(sf={str(i):dict(components=['sf'+str(i)]) for i in (0,4,5)},
                        full={'4':dict(components=['dq'])}),reset=record('reset',1))
    return point,value


def test_composition_counts_helpers_and_reset_once_and_uses_fresh_mean():
    point,value=compose_fixture();result=search.compose(point,[value])
    assert result['decision']['winner']=='rule-sf'
    stats=result['decision']['statistics']
    assert {k:v['mean_us'] for k,v in stats.items()}==dict(original=15,**{'rule-sf':11,'rule-full':13})
    assert result['bindings']['original']['helper_config']==0
    assert result['bindings']['rule-sf']['helper_config']==4
    assert not result['native_no_regression_admitted'] and not result['model_tpot_admitted']
    broken=deepcopy(value);broken['reset']=None
    with pytest.raises(ValueError,match='reset'):search.compose(point,[broken])
    broken=deepcopy(value);broken['component_records']['main']['mean_us']=1
    with pytest.raises(ValueError,match='mean'):search.compose(point,[broken])
    broken=deepcopy(value);broken['arms'].pop('rule-sf')
    with pytest.raises(ValueError,match='partial'):search.compose(point,[broken])


def test_composition_rejects_cross_phase_and_corrupt_samples():
    point,value=compose_fixture()
    other=deepcopy(value);other['authority']='different-card'
    with pytest.raises(ValueError,match='phase/run/device'):search.compose(point,[value,other])
    for bad in (0,float('nan'),float('inf')):
        broken=deepcopy(value);broken['component_records']['main']['samples_us'][0][0]=bad
        with pytest.raises(ValueError):search.compose(point,[broken])
    broken=deepcopy(value);broken['component_records']['main']['samples_us']=[[10.]*5]*6
    with pytest.raises(ValueError,match='sample inventory'):search.compose(point,[broken])


def capture_fixture():
    numeric=dict(relative_l1_error=.004,tolerance=.005,cells=1024,nonfinite=0,
                 fixture='ORIGINAL_GGUF_COMMON_DYADIC_F64_FACTORIZED_DOT')
    part=dict(correctness='PASS',components=['main'],stages=[dict(id='main')],
        fixture=dict(copies=2,l2_bytes=64,active_ring_bytes=144),numerical=[numeric]*6,
        omitted_producer='DETECTED',changed_input_and_router='PASS',zero_a='PASS',pointer_negatives=['low'],
        receipt=dict(row_maps='GPU_MM_IDS_HELPER_EVERY_CALL_ACTUAL_CONSUMER_POINTERS'))
    return dict(status='CAPTURED_NOT_TIMED',timing_method='ASYS_GPU_ACTIVITY',
        protocol=dict(rounds=4,samples=3),arms={'original':part},expansions=dict(sf={},full={}))


def test_numeric_and_cold_failures_never_become_timing_admission():
    value=capture_fixture();search.validate_capture(value)
    for key,val in [('omitted_producer','PASS'),('pointer_negatives',[]),('correctness','PENDING')]:
        broken=deepcopy(value);broken['arms']['original'][key]=val
        with pytest.raises(ValueError):search.validate_capture(broken)
    for error in (float('nan'),.005,-1):
        broken=deepcopy(value);broken['arms']['original']['numerical'][0]['relative_l1_error']=error
        with pytest.raises(ValueError):search.validate_capture(broken)
    broken=deepcopy(value);broken['arms']['original']['fixture']['active_ring_bytes']=143
    with pytest.raises(ValueError):search.validate_capture(broken)


def test_resume_checks_run_screen_and_retained_batch_evidence(tmp_path):
    from tools.run_moe_prefill_sweep import completed,base
    from quactlize.runtime.compiler import sha
    run=dict(sha256='frozen-run');evidence=tmp_path/'raw.log';evidence.write_text('original')
    batch=tmp_path/'batch.json';base.sealed(batch,dict(run=run['sha256'],status='PASS',
                                                   evidence={str(evidence):sha(evidence)}))
    screen=tmp_path/'screen.json';base.sealed(screen,dict(run=run['sha256']))
    point=tmp_path/'pid.json'
    from tools.run_smallm_regressions_asys import sealed
    base.sealed(point,dict(run=run['sha256'],status='PASS',point='pid',screen=str(screen),
        screen_sha256=sealed(screen)['sha256'],batch_receipts={str(batch):sha(batch)}))
    assert completed(point,run)['status']=='PASS'
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='evidence changed'):completed(point,run)
    with pytest.raises(ValueError,match='another run'):completed(point,dict(sha256='different'))


def test_later_prepare_failure_closes_isolated_timers_before_capture_owners(monkeypatch,tmp_path):
    from contextlib import contextmanager
    from dev.strategy import cold_decode,prefill_fixture,prefill_measure,moe_prefill_native
    from quactlize.runtime import native
    state=dict(closed=False,exited=False)
    w=Workload(12,512,2048,128,'indexed','bf16',experts=256,topk=8)
    monkeypatch.setattr(native,'SDK',lambda path:None)
    monkeypatch.setattr(cold_decode,'Ring',lambda *a,**kw:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(prefill_fixture,'Helpers',lambda *a:None)
    monkeypatch.setattr(moe_prefill_native,'RoutedContext',lambda *a:SimpleNamespace(close=lambda:None,check=lambda:None))
    monkeypatch.setattr(moe_prefill_native,'RoutedTC',lambda *a:SimpleNamespace(close=lambda:None))
    count=[0]
    @contextmanager
    def prepared(*a,**kw):
        count[0]+=1
        if count[0]==2:raise ValueError('injected later prepare failure')
        try:yield None,dict(receipt={})
        finally:
            assert state['closed']
            state['exited']=True
    monkeypatch.setattr(prefill_measure,'prepared',prepared)
    class Timer:
        def __init__(self,**kw):pass
        def stages(self,*a):return [],[]
        def close(self):state['closed']=True
    monkeypatch.setattr(search,'AlternatingStages',Timer)
    a=SimpleNamespace(sdk=tmp_path,worker=0,l2_bytes=64,attempt=tmp_path)
    run=dict(sha256='run',devices=[{}],source=dict(sha256='source'),images={},adapter_sha256='adapter',
             helpers=str(tmp_path),dequant=str(tmp_path),adapter='unused')
    arms=[dict(id=str(i),recipe=asdict(Recipe.make('tc','fq',split=1)),module='m',binding=dict(query='q')) for i in range(2)]
    with pytest.raises(ValueError,match='injected later prepare'):
        search.measure(a,dict(id=w.key,workload=asdict(w)),arms,dict(tc={'m':{}}),{'q':{}},run,'screen',{})
    assert state['closed'] and state['exited']


def test_worker_retries_failed_batch_without_remeasuring_passed_arms(monkeypatch,tmp_path):
    from collections import Counter
    from tools import run_moe_prefill_sweep as runner
    from quactlize.runtime.compiler import sha
    point,value=compose_fixture();point=campaign.seal(point)
    plan=dict(points=[point]);runner.base.sealed(tmp_path/'plan.json',plan)
    runner.base.sealed(tmp_path/'run.json',dict(cards=['1'],devices=[dict(device=dict(pci='host-test-prefill'))],
        assignments=[[point['id']]]))
    run=runner.sealed(tmp_path/'run.json')
    monkeypatch.setattr(runner,'verify_run',lambda *a:None)
    monkeypatch.setattr(runner.fcntl,'flock',lambda *a:None)
    monkeypatch.setattr(runner,'resource',lambda *a:dict(point=point,queries={}))
    calls=Counter();fail=[True]
    def invocation(a,run,job,**kwargs):
        key=(job['phase'],tuple(job['arms']));calls[key]+=1
        if job['phase']=='screen' and job['arms']==['rule-full'] and fail[0]:
            fail[0]=False;raise ValueError('test-only profile failure')
        data=deepcopy(value);data['arms']={k:v for k,v in data['arms'].items() if k in job['arms']}
        if 'rule-full' not in data['arms']:data['reset']=None
        for kind,ids in [('sf',{'original','rule-sf'}),('full',{'rule-full'})]:
            if not ids.intersection(data['arms']):data['expansions'][kind]={}
        rounds,samples=(4,3) if job['phase']=='screen' else (6,5)
        data['protocol']=dict(rounds=rounds,samples=samples)
        data['authority']=job['phase']
        for row in [*data['component_records'].values(),*([data['reset']] if data['reset'] else [])]:
            row['samples_us']=[[row['mean_us']]*samples for _ in range(rounds)]
        raw=tmp_path/(digest(key)+'.trace');raw.write_text('fake host evidence, never GPU admission')
        return tmp_path,dict(status='PASS',run=run['sha256'],point=point['id'],phase=job['phase'],
                             data=data,evidence={str(raw):sha(raw)})
    monkeypatch.setattr(runner,'invocation',invocation)
    a=SimpleNamespace(output=tmp_path,worker=0,budget_hours=1)
    assert runner.worker(a)==1
    assert runner.worker(a)==0
    before=dict(calls)
    assert runner.worker(a)==0 and dict(calls)==before
    assert calls['screen',('original','rule-sf')]==1
    assert calls['screen',('rule-full',)]==2
    final=runner.sealed(tmp_path/'gpu-1/points'/(point['id']+'.json'))
    assert final['decision']['winner']=='rule-sf' and len(final['batch_receipts'])==4
    other=dict(point,id='budget-test')
    assert runner.phase(a,run,other,{},'screen',float('-inf')) is None
