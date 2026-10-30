"""Missing-geometry coverage and resumable coordination; no GPU evidence."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,replace
from types import SimpleNamespace as NS
import time

import pytest

from dev.moe_confirm import plan as confirmation,refine,refine_plan,native_ab
from dev.strategy import moe_prefill_campaign as campaign
from quactlize.dispatch.table import Recipe,Workload
from tools import run_moe_confirmation as runner
from test_moe_prefill_campaign import fake_resources,compose_fixture


@pytest.fixture(scope='module')
def plan():return confirmation.make_plan('refine-prefill',native_compare=True)


def test_40_key_audit_and_bounded_two_shape_trial(plan):
    assert len(plan['coverage']['host_keys'])==40
    assert plan['coverage']['keys_with_gaps']==40
    assert len(plan['points'])==10 and plan['counts']['maximum_candidates']<=48
    assert plan['counts']['tc_modules']<=32 and not plan['vectors']
    assert not plan['production_changed'] and not plan['champion_injection']
    assert not plan['historical_times_reused'] and plan['native_comparison']
    groups=defaultdict(list)
    for p in plan['points']:
        w=p['workload'];groups[confirmation.key(w)].append(p)
        assert (w['qtype'],w['k'],w['channels'],w['m'])==(13,512,8,5120)
        assert p['arms'][0]['id']=='original' and p['arms'][0]['recipe']['route']=='sf'
        assert any('old-control' in a['roles'] and a['recipe']['route']=='fq' for a in p['arms'])
        assert sum(a['recipe']['kind']=='full' for a in p['arms'])==1
        # This prior M4096 winner is independently recovered by the blind
        # geometry rule at M5120, not appended from the winners table.
        assert any(a['roles']==['rule'] and a['recipe']['route']=='sf' and
            all(dict(a['recipe']['parameters'])[k]==v for k,v in
                dict(tm=64,tn=128,tk=256,wm=64,wn=16,stages=2,dn=64).items()) for a in p['arms'])
    assert len(groups)==2
    for group in groups.values():
        assert {p['workload']['router'] for p in group}==set(confirmation.ROUTERS)
        assert all(p['arms']==group[0]['arms'] for p in group)


def test_transfer_withholds_target_family_and_never_reads_latency(monkeypatch,plan):
    p=plan['points'][0];w=Workload(**p['workload'])
    seed=next(Recipe.from_dict(a['recipe']) for a in p['arms'] if a['roles']==['rule'] and
              a['recipe']['route']=='sf' and dict(a['recipe']['parameters'])['tm']==64)
    # No old pool in this unit test. The ONLY legal donor is deliberately
    # more than 2x away in M; exact-target N/K must still be excluded.
    class Observation:
        def __init__(self,w,r):self.workload=w;self.recipe=r
        @property
        def latency(self):pytest.fail('candidate generation consumed timing')
    donor=Observation(replace(w,m=2048,n=4096,output_stride=4096),seed)
    poison=Observation(w,Recipe.make(seed.kind,seed.route,**(dict(seed.parameters)|dict(tn=256))))
    monkeypatch.setattr(campaign,'parents',lambda *a:([],{}))
    _,out=refine_plan.parents(w,NS(observations=[donor,poison]),None)
    assert out and {dict(r.parameters)['tn'] for r in out}=={dict(seed.parameters)['tn']}
    _,reverse=refine_plan.parents(w,NS(observations=[poison,donor]),None)
    assert out==reverse


@pytest.mark.parametrize('occupancy',[1,2,8])
def test_bound_resource_queries_keep_controls_and_common_router_inventory(plan,occupancy):
    groups=defaultdict(list)
    for p in plan['points']:
        images,queries=fake_resources(p,occupancy)
        bound=campaign.resolve(p,queries,images)
        groups[confirmation.key(p['workload'])].append(bound['arms'])
        assert bound['arms'][0]['id']=='original'
        assert any('old-control' in a['roles'] for a in bound['arms'])
        assert any(a['recipe']['kind']=='full' for a in bound['arms'])
        for a in bound['arms']:
            if a['recipe']['kind']=='tc':assert a['binding']['launch']['grid']<=72*occupancy
    for arms in groups.values():assert all(a==arms[0] for a in arms)


def finalist_fixture():
    w=Workload(13,2048,512,5120,'indexed','bf16',experts=256,topk=8,channels=8)
    recipes=[Recipe.make('tc','sf',tm=128,persistent=-1,dequant_config=5),
        Recipe.make('tc','fq',tm=128,persistent=0,dequant_config=-1),
        Recipe.make('tc','sf',tm=64,persistent=-1,dequant_config=0),
        Recipe.make('tc','sf',tm=32,persistent=-1,dequant_config=0),
        Recipe.make('full','full',dequant_config=4)]
    arms=[dict(id=aid,recipe=asdict(r),roles=[role]) for aid,r,role in zip(
        ('original','old','a','b','full'),recipes,('original','old-control','rule','rule','rule'))]
    points=[];screens={}
    for i,router in enumerate(confirmation.ROUTERS):
        rw=replace(w,router=router)
        p=dict(id=rw.key,workload=asdict(rw),arms=deepcopy(arms));points.append(p)
        times=dict(original=20,old=30,a=10 if i%2 else 15,b=15 if i%2 else 10,full=25)
        # Config0 wins only one router; common config5 is consistently good.
        records={k:dict(mean_us=float(v)) for k,v in
            [('s0',1 if i==0 else 20),('s5',2),('f4',2),('f11',3)]}
        screens[p['id']]=dict(decision=dict(statistics={k:dict(mean_us=v,samples_us=[[v]*3]*4)
            for k,v in times.items()}),component_records=records,
            helper_components=dict(sf={'0':['s0'],'5':['s5']},full={'4':['f4'],'11':['f11']}))
    return points,screens


def test_finalists_union_preserves_both_router_winners_and_common_helpers():
    points,screens=finalist_fixture();got=refine_plan.finalists(points,screens)
    for p in got.values():
        assert len(p['arms'])==5 and p['sf_configs']==[5] and p['full_configs']==[4]
        assert {dict(a['recipe']['parameters']).get('tm') for a in p['arms']}=={None,32,64,128}
        assert [a['recipe'] for a in p['arms']]==[a['recipe'] for a in next(iter(got.values()))['arms']]
    broken=deepcopy(points);broken[0]['arms']=broken[0]['arms'][:-1]
    with pytest.raises(ValueError):refine_plan.finalists(broken,screens)


@pytest.mark.parametrize('stage',['screen','confirm','native'])
def test_refinement_worker_routes_before_native_flag(tmp_path,monkeypatch,stage):
    p=dict(id='point',workload={})
    runner.base.sealed(tmp_path/'plan.json',dict(points=[p]))
    runner.base.sealed(tmp_path/'run.json',dict(mode='refine-prefill',native_comparison=True,
        cards=['0'],assignments=[['point']],devices=[dict(device=dict(pci='test-refine-worker'))]))
    monkeypatch.setattr(runner,'verify',lambda *a:None)
    monkeypatch.setattr(runner,'point',lambda *a:pytest.fail('ordinary confirmation'))
    monkeypatch.setattr(native_ab,'point',lambda *a:pytest.fail('bypassed refine dispatcher'))
    seen=[];monkeypatch.setattr(refine,'point',lambda a,*_:seen.append(a.refine_stage) or True)
    assert runner.worker(NS(output=tmp_path,worker=0,preflight=None,budget_hours=1,refine_stage=stage))==0
    assert seen==[stage]


def test_refine_incomplete_native_is_not_confirmed_success(tmp_path,monkeypatch):
    p=dict(id='p',workload=dict(n=2048,router='spread'),arms=[])
    run=dict(sha256='run',cards=['0'],assignments=[['p']],native_comparison=True)
    a=NS(output=tmp_path,budget_hours=1)
    monkeypatch.setattr(refine,'completed',lambda *a:True)
    monkeypatch.setattr(refine,'collect',lambda a,p,r,stage: {} if stage=='native' else {'p':{}})
    monkeypatch.setattr(refine,'sealed',lambda *a:dict(point=p))
    monkeypatch.setattr(refine_plan,'finalists',lambda *a:{'p':p})
    monkeypatch.setattr(refine,'choose',lambda *a:[dict(host_test=True)])
    monkeypatch.setattr(refine,'spawn',lambda *a:[1])
    monkeypatch.setattr(native_ab,'review',lambda *a:pytest.fail('incomplete native admitted'))
    assert refine.run(a,dict(points=[p],coverage=dict(host_keys=[{}],keys_with_gaps=1)),run)==1
    result=runner.base.read(tmp_path/'summary.json')
    assert result['status']=='INCOMPLETE' and not result['native_no_regression_admitted']


def test_refine_candidate_reset_counted_once_and_not_public_selector_claim(monkeypatch):
    p,value=compose_fixture();full=p['arms'][-1];full['id']='original'
    p.update(arms=[full],selection_source='MEASURED_REFINEMENT_NOT_PRODUCTION')
    core=dict(value['arms']['rule-full'],receipt={},fixture=dict(fixture=dict(raw_sha256=['raw'])),
        timed_router_sha256='ids',timing_input_repeat=0,timed_fixture={'fixture':'host-test'})
    value['arms']={'original':core}
    for r in value['component_records'].values():
        r.update(kind='provider' if r['component_id']=='full' else 'expansion',identity=dict(node={}))
    monkeypatch.setattr(native_ab.engine,'invocation',lambda *a,**k:(None,dict(data=value,evidence={})))
    result=native_ab.candidate_pass(NS(),dict(mode='refine-prefill'),p,{},0)
    assert [c['role'] for c in result['components']]==['provider','expansion','error_reset']
    assert sum(c['mean_us'] for c in result['components'])==13
    p['selection_source']='unconfirmed'
    with pytest.raises(ValueError,match='unbound'):native_ab.candidate_pass(NS(),dict(mode='refine-prefill'),p,{},0)


def test_refine_native_regression_is_retained_but_recipe_cannot_change(tmp_path,monkeypatch):
    p=dict(id='p',arms=[dict(recipe=dict(kind='full',route='full',parameters=[]))])
    run=dict(mode='refine-prefill',sha256='run',cards=['0']);a=NS(output=tmp_path,worker=0)
    runner.base.sealed(tmp_path/'gpu-0/native-points/p.json',dict(status='REGRESSION',run='run',
        evidence={},candidate_recipe=p['arms'][0]['recipe']))
    monkeypatch.setattr(native_ab.engine,'invocation',lambda *a,**k:pytest.fail('valid loss resampled'))
    assert native_ab.point(a,run,p,time.monotonic()+1)
    p['arms'][0]['recipe']['parameters']=[['changed',1]]
    with pytest.raises(ValueError,match='candidate changed'):native_ab.point(a,run,p,time.monotonic()+1)


@pytest.mark.parametrize('stage',['screen','confirm'])
def test_completed_refinement_stage_never_remeasures(tmp_path,monkeypatch,stage):
    run=dict(cards=['0']);a=NS(output=tmp_path,worker=0,refine_stage=stage,preflight=None)
    monkeypatch.setattr(refine,'completed',lambda path,r: {'status':'PASS'})
    monkeypatch.setattr(refine.engine,'resource',lambda *a,**k:pytest.fail('completed stage remeasured'))
    assert refine.point(a,run,dict(id='p'),time.monotonic()+1)


def test_refinement_archive_locator_keeps_raw_reports_on_box(tmp_path):
    from tools.pack_simt_split_review import result_root,exclusion
    from pathlib import Path
    runner.base.sealed(tmp_path/'results/plan.json',dict(schema='quactlize.moe-prefill-refine.v1'))
    assert result_root(tmp_path)==tmp_path/'results'
    for name in ('trace.asysrep','trace.sqlite','kernel.so'):
        assert exclusion(Path(name)) is not None
    for name in ('summary.json','native-summary.json','native-comparison.csv','selection.json'):
        assert exclusion(Path(name)) is None
