"""Host-only contracts; these tests do not establish PPU performance."""
from collections import defaultdict
from copy import deepcopy
import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dev.moe_confirm import plan
from dev.moe_confirm.backend import Context,Full,prepare_receipt
from dev.moe_scaffold.protocol import Prepare,PreparePlanV2
from dev.strategy import moe_prefill_search as search
from dev.strategy.prefill_fixture import routing
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('worker_id',[0,1])
@pytest.mark.parametrize('preflight',[None,'representative-point'])
def test_worker_entry_dispatches_before_coordinator_locals(tmp_path,monkeypatch,worker_id,preflight):
    from tools import run_moe_confirmation as runner
    args=SimpleNamespace(plan=False,probe=None,output=tmp_path,job=None,helper_gate=False,
                         worker=worker_id,preflight=preflight)
    seen=[]
    monkeypatch.setattr(runner,'worker',lambda a:seen.append(a) or 7)
    assert runner.main(args)==7
    assert seen==[args] and seen[0].worker==worker_id and seen[0].preflight==preflight


@pytest.mark.parametrize('changed',['coordinator','reused-coordinator','kernel','plan','artifact'])
def test_build_reuse_keeps_old_results_and_rejects_unrelated_changes(tmp_path,monkeypatch,changed):
    from tools import run_moe_confirmation as runner
    from dev.strategy import prefill_build
    from quactlize.runtime.compiler import sha
    previous=tmp_path/'old';previous.mkdir()
    artifact=previous/'library.so';artifact.write_bytes(b'unchanged-test-library')
    toy_plan=dict(points=[dict(id='p',arms=[dict(module='tc',recipe=dict(kind='tc')),
                                          dict(recipe=dict(kind='full'))])])
    runner.base.sealed(previous/'plan.json',toy_plan);planned=runner.sealed(previous/'plan.json')
    files={'tools/run_moe_confirmation.py':'before','quactlize/kernel.cuh':'same'}
    before=dict(plan=planned['sha256'],source=dict(files=files),harness=dict(files=files),
                immutable_files={str(artifact):sha(artifact)},moe_helpers=dict(test=True),
                helpers=str(previous/'helpers'),dequant=None,sdk=dict(test=True))
    cache=previous
    if changed=='reused-coordinator':
        cache=tmp_path/'original-build'
        runner.base.sealed(cache/'run.json',dict(reused_build=None))
        before['reused_build']=dict(output=str(cache))
    runner.base.sealed(previous/'run.json',before)
    (previous/'images.json').write_text('{"tc":{},"vectors":{}}')
    old_bytes=(previous/'run.json').read_bytes()
    current=files|{'tools/run_moe_confirmation.py':'after'}
    if changed=='kernel':current['quactlize/kernel.cuh']='changed'
    if changed=='plan':planned=planned|dict(sha256='different')
    if changed=='artifact':artifact.write_bytes(b'changed-library')
    monkeypatch.setattr(runner,'source_identity',lambda root:dict(files=current))
    monkeypatch.setattr(runner,'harness_identity',lambda root:dict(files=current))
    monkeypatch.setattr(runner,'verify_moe',lambda value:None)
    monkeypatch.setattr(prefill_build,'verify',lambda sdk,folder:None)
    verified=[]
    def images(a,p,records,dq,*,points):
        assert a.output==cache
        assert all(r['recipe']['kind']!='full' for pt in points for r in pt['arms'])
        verified.append(True)
    monkeypatch.setattr(runner.base,'verify_images',images)
    args=SimpleNamespace(reuse_build=previous,output=tmp_path/'new',sdk=Path('/mock-sdk'))
    if changed not in ('coordinator','reused-coordinator'):
        with pytest.raises(ValueError):runner.reuse_build(args,planned)
        assert not verified
    else:
        *_,reference=runner.reuse_build(args,planned)
        assert verified and reference['compiled_artifacts_only'] and not reference['previous_measurements_reused']
        assert reference['cache_output']==str(cache)
    assert (previous/'run.json').read_bytes()==old_bytes
    assert args.output==tmp_path/'new' and not args.output.exists()


def continuation_fixture(tmp_path):
    from tools import run_moe_confirmation as runner
    from quactlize.runtime.compiler import sha
    old=tmp_path/'old'
    runner.base.sealed(old/'run.json',dict(cards=['3'],assignments=[['a','b','c']],
                                         source_revision=dict(quactlize_commit='old-source')))
    run=runner.sealed(old/'run.json')
    batch=old/'gpu-3/batches/complete.json'
    runner.base.sealed(batch,dict(run=run['sha256'],status='PASS',evidence={},
                                 numerical=dict(host_test=True)))
    good=old/'gpu-3/points/a.json'
    w=asdict(Workload(12,512,2048,1,'indexed','bf16',experts=256,topk=8))
    value=dict(point='a',run=run['sha256'],status='PASS',workload=w,
               decision=dict(statistics=dict(original=dict(mean_us=1.))),bindings=dict(original={}),
               batch_receipts={str(batch):sha(batch)})
    runner.base.sealed(good,value)
    # One partial point and one unstarted point must be wholly remeasured.
    runner.base.sealed(old/'gpu-3/points/b.json',dict(point='b',run=run['sha256'],status='FAIL'))
    plan=dict(points=[dict(id=i,workload=w,arms=[{}],full_configs=[]) for i in ('a','b','c')],scope='HOST_TEST')
    return runner,old,run,plan


def test_gpu_change_keeps_complete_pairs_but_not_partial_timings(tmp_path):
    runner,old,old_run,plan=continuation_fixture(tmp_path)
    before={p:p.read_bytes() for p in old.rglob('*.json')}
    args=SimpleNamespace(continue_from=old,output=tmp_path/'new')
    keep=runner.retain_complete(args,plan)
    assert set(keep)=={'a'}
    value=runner.retained_point(keep['a'])
    assert value['run']==old_run['sha256']
    payload=runner.sealed(Path(keep['a']['path']))
    assert payload['origin']['source_revision']['quactlize_commit']=='old-source'
    assert payload['batches'][0]['numerical']==dict(host_test=True)
    assert {p:p.read_bytes() for p in old.rglob('*.json')}==before
    assigned=runner.assign_remaining(plan,4,keep)
    assert len(assigned)==4 and sorted(i for ids in assigned for i in ids)==['b','c']
    assert sum(not ids for ids in assigned)==2
    assert runner.assign_remaining(plan,4,dict.fromkeys(('a','b','c')))==[[],[],[],[]]


def test_second_gpu_change_preserves_original_provenance_and_review(tmp_path):
    runner,old,old_run,plan=continuation_fixture(tmp_path)
    middle=tmp_path/'middle'
    keep=runner.retain_complete(SimpleNamespace(continue_from=old,output=middle),plan)
    runner.base.sealed(middle/'run.json',dict(cards=['0','1'],assignments=[['b'],['c']],
        retained_points=keep,source_revision=dict(quactlize_commit='middle-source')))
    args=SimpleNamespace(continue_from=middle,output=tmp_path/'last')
    keep=runner.retain_complete(args,plan)
    assert set(keep)=={'a'} and runner.retained_point(keep['a'])['run']==old_run['sha256']
    runner.base.sealed(args.output/'run.json',dict(cards=['2','3'],assignments=[['b'],['c']],
        retained_points=keep,source_revision=dict(quactlize_commit='new-source')))
    run=runner.sealed(args.output/'run.json')
    assert runner.review(args,plan,run,[0,0])==1
    summary=runner.base.read(args.output/'summary.json')
    assert (summary['retained'],summary['passed'],summary['expected'])==(1,1,3)
    assert summary['points'][0]['source_run']==old_run['sha256']
    assert not summary['native_no_regression_admitted']


def test_existing_output_has_actionable_resume_or_gpu_change_message(tmp_path):
    from tools import run_moe_confirmation as runner
    args=SimpleNamespace(plan=False,probe=None,output=tmp_path,job=None,helper_gate=False,
                         worker=None,confirm_idle=True,resume=False)
    with pytest.raises(ValueError,match='--continue-from OLD --output NEW'):runner.main(args)


@pytest.fixture(scope='module')
def inventory():return plan.make_plan()


def test_frozen_shortlist_counts_no_new_grid_search_or_historical_times(inventory):
    assert inventory['counts']==dict(points=344,arms=955,tc_modules=21,vector_modules=19,maximum_candidates=6)
    assert not inventory['historical_times_reused'] and not inventory['production_changed']
    assert not inventory['native_comparison']
    data=json.loads(plan.SEED.read_text())
    assert len(data['families'])==40
    assert sum(len(f['champions']) for f in data['families'])==199
    assert sum(len(f['historical_missing']) for f in data['families'])==1
    assert 'mean_us' not in plan.SEED.read_text()
    for f in data['families']:
        assert 2<=len(f['candidates'])<=6
        assert f['original'] in f['candidates']
        assert all(r['winner'] in f['candidates'] for r in f['champions'])
        assert all(digest(r)==cid for cid,r in f['candidates'].items())
        assert sum(r['kind']=='full' for r in f['candidates'].values())<=1


def test_host_keys_not_router_labels_choose_the_candidate_inventory(inventory):
    groups=defaultdict(list)
    for p in inventory['points']:groups[plan.key(p['workload'])].append(p)
    assert len(groups)==72
    for host,points in groups.items():
        assert len(points)==(1 if host[-1]==1 else 5)
        ids={a['id'] for a in points[0]['arms']}
        assert 'original' in ids
        for p in points:
            assert p['fixed_candidates'] and {a['id'] for a in p['arms']}==ids
            assert p['workload']['compute']=='bf16' and p['workload']['storage']=='f32'
            assert p['workload']['experts']==256 and p['workload']['topk']==8
            if host[-1]<=8:
                assert len(ids)<=2
                assert all(a['recipe']['kind'] in ('q4','simt') for a in p['arms'])


def test_preflight_covers_every_implementation_and_reuses_real_points(inventory):
    ids=plan.preflight_points(inventory)
    assert len(ids)==4 and len(set(ids))==4
    by_id={p['id']:p for p in inventory['points']}
    def kinds(points):return {(a['recipe']['kind'],a['recipe']['route']) for p in points for a in p['arms']}
    assert kinds([by_id[i] for i in ids])==kinds(inventory['points'])
    assert max(by_id[i]['workload']['m'] for i in ids)==128
    assert all(by_id[i]['workload']['router']=='spread' for i in ids)


@pytest.mark.parametrize('mode,count',[('decode',144),('prefill',200),('all',344)])
def test_plan_entrypoint_needs_no_device_sdk_or_llama(mode,count):
    output=subprocess.check_output([sys.executable,str(ROOT/'tools/run_moe_confirmation.py'),
                                   '--plan','--mode',mode,'--sdk','/nonexistent'],text=True)
    assert json.loads(output)['points']==count


def stats(**costs):return dict(decision=dict(statistics={k:dict(mean_us=v) for k,v in costs.items()}))


def test_common_choice_rejects_router_specific_winner_and_ignores_tiny_tie():
    profiles=[stats(original=100,fast=90,fragile=70),stats(original=100,fast=95,fragile=101)]
    assert plan.choose(profiles)=='fast'
    assert plan.choose([stats(original=100,tie=99.5)])=='original'
    assert plan.choose([stats(original=20,tie=19.9)])=='original'
    assert plan.choose([stats(original=20,win=19)])=='win'
    assert plan.choose([stats(original=20,slow=20.1)])=='original'
    with pytest.raises(ValueError,match='incomplete'):
        plan.choose([profiles[0],stats(original=100,fast=95)])
    for bad in (0,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='invalid'):plan.choose([stats(original=20,bad=bad)])
    with pytest.raises(ValueError):plan.choose([])


def test_exact_schedule_binding_does_not_expand_or_recompile(monkeypatch,inventory):
    point=deepcopy(next(p for p in inventory['points'] if p['workload']['m']==512))
    calls=[]
    class Probe:
        def __init__(self,spec,image):calls.append(('load',image))
        def query(self,request):
            calls.append(('query',request));return dict(status='READY',device=dict(compute_units=72),occupancy=2)
    monkeypatch.setattr(plan,'NativeProbe',Probe)
    monkeypatch.setattr(plan,'resolved_grid',lambda w,r,cu,occ:dict(recipe=asdict(r),cu=cu,occupancy=occ))
    images=dict(tc={a['module']:a['module'] for a in point['arms'] if a['recipe']['kind']=='tc'})
    bound,queries=plan.bind(point,images)
    assert len(bound['arms'])==len(point['arms'])
    assert len([x for x in calls if x[0]=='load'])==len(images['tc'])
    assert len([x for x in calls if x[0]=='query'])==len(queries)
    assert all(a['recipe']==b['recipe'] for a,b in zip(point['arms'],bound['arms']))


def test_prepare_rebinds_outputs_without_changing_plan_or_call():
    obj=Context.__new__(Context);obj.prepare_plan=PreparePlanV2()
    obj.prepare_call=Prepare(input=11,output=12,expert_rows=13,active_experts=14,active_count=15)
    obj.prepare_call.route.stream=16
    observed=[]
    def execute(call,selected):
        c=call._obj
        observed.append((c.input,c.output,c.expert_rows,c.active_experts,c.active_count,c.route.stream,selected._obj))
        return 0
    obj.qz=SimpleNamespace(prepare_execute_v2=execute)
    assert obj.prepare(21)==0
    assert obj.prepare(31,rows=32,active_ids=33,active_count=34)==0
    assert observed[0][:-1]==(11,21,13,14,15,16)
    assert observed[1][:-1]==(11,31,32,33,34,16)
    assert observed[0][-1] is observed[1][-1] is obj.prepare_plan
    assert obj.prepare_call.output==12 and obj.prepare_call.expert_rows==13
    assert set(prepare_receipt(obj.prepare_plan))=={'version','route','helper'}


def test_full_call_generates_provider_ids_on_gpu_before_actual_provider():
    events=[]
    def record(name):return lambda *args,**kw:events.append((name,args,kw)) or 0
    core=Full.__new__(Full)
    core.b=SimpleNamespace(launch_route=record('route'),rows=128,ids=SimpleNamespace(ptr=1),
        destination=SimpleNamespace(ptr=2),r=SimpleNamespace(stream=C.c_void_p(3),fill=record('poison')),
        prepare=record('prepare'),provider_rows=record('ids'),finish=record('finish'))
    core.active=record('active');core.matmul=record('main')
    core.a=SimpleNamespace(data_ptr=lambda:4);core.out=SimpleNamespace(data_ptr=lambda:5)
    core.rows=SimpleNamespace(data_ptr=lambda:6);core.indices=SimpleNamespace(data_ptr=lambda:7)
    core.ids=SimpleNamespace(ptr=8);core.count=SimpleNamespace(ptr=9)
    core.update();assert not core.ids_ready
    assert [e[0] for e in events]==['route','active','poison']
    events.clear();core.launch(2)
    assert [e[0] for e in events]==['prepare','ids','main','finish']
    assert events[0][2]==dict(rows=6,active_ids=8,active_count=9)
    assert events[1][1][:4]==(1,2,7,128) and events[2][1]==(2,)
    assert core.ids_ready


@pytest.mark.parametrize('m',[1,7,16,128,5120])
@pytest.mark.parametrize('channels',[1,8])
def test_provider_expert_ids_follow_gpu_permutation_not_source_rows(m,channels):
    w=Workload(12,512,2048,m,'indexed','bf16',experts=256,topk=8,channels=channels,router='repeat2')
    r=routing(w,1);order=r['order'].copy()
    for start,stop in zip(r['offsets'][:-1],r['offsets'][1:]):order[start:stop]=order[start:stop][::-1]
    np.testing.assert_array_equal(r['owners'][order],r['sorted_owners'])
    text=(ROOT/'dev/moe_confirm/provider_rows.cu').read_text()
    assert 'experts[i]=ids[destination[i]]' in text and 'int64_t' in text


def test_separate_component_inventory_allows_direct_simt_and_public_prepare(monkeypatch):
    monkeypatch.setattr(search,'replay_extracted',lambda capture,proof:None)
    w=Workload(13,2048,512,4,'indexed','bf16',experts=256,topk=8,channels=8)
    class Graph:
        count=16
        def close(self):pass
    def capture(names):
        nodes=[dict(name=n,symbol=n,grid=[1,1,1],block=[256,1,1],shared_bytes=0) for n in names]
        return SimpleNamespace(groups=[[i] for i in range(len(names))],descriptions=nodes,
                               isolate=lambda *a,**kw:Graph())
    timer=search.AlternatingStages(quactlize_prepare=True)
    vector=Recipe.make('simt','fq',split=1)
    got,_=timer.collect(capture(['quactlize::execution::simt::register_reuse']),w,vector,'a','s',lambda:None)
    assert [s['role'] for s in got]==['producer']
    tc=Recipe.make('tc','fq',split=1)
    names=['quactlize::routing::histogram_rank<8,512>', 'quactlize::moe::prepare_prefix<true>',
           'quactlize::moe::prepare_gather<8,512,true>', 'quactlize::moe::provider_expert_ids',
           'cutlass::device_kernel<cutlass::gemm::GemmUniversal>', 'quactlize::prefill::finish_bf16']
    got,_=timer.collect(capture(names),w,tc,'a','s',lambda:None)
    assert [s['role'] for s in got]==['prepare','prepare','adapter','prepare','producer','postprocess']
    with pytest.raises(ValueError,match='inventory'):
        timer.collect(capture(names[:2]+names[3:]),w,tc,'a','s',lambda:None)
    with pytest.raises(ValueError,match='inventory'):
        timer.collect(capture(names),w,Recipe.make('tc','fq',split=4),'a','s',lambda:None)
    timer.close()


def test_explicit_helpers_do_not_sneak_in_profile_specific_retuning():
    from test_moe_prefill_campaign import compose_fixture
    p,data=compose_fixture()
    result=search.compose(p,[data],explicit_helpers=True)
    assert result['bindings']['rule-sf']['helper_config']==0
    assert result['decision']['statistics']['rule-sf']['mean_us']==15


def test_bridge_is_sdk_only_and_preserves_driver_function_api():
    text=(ROOT/'dev/moe_confirm/bridge.cpp').read_text()
    assert 'kernel_bridge.hpp' in text and 'nc_error' in text and 'ggml' not in text
    assert 'cuGraphKernelNodeGetParams' in (ROOT/'dev/native_components/kernel_node.hpp').read_text()
