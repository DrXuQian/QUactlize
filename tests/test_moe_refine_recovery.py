"""Host resource/recovery tests only; synthetic resources are not Box evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

import pytest

from dev.moe_confirm import staged_grid as staged,refine_plan
from dev.strategy import moe_prefill_campaign as campaign,moe_prefill_search as search
from dev.strategy.moe_plan import load_catalog
from dev.strategy.prescreen import Prescreen,rejection
from dev.strategy.resources import module_spec
from dev.strategy.space import split_variants
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest
from tools import retry_moe_refine as retry,run_moe_confirmation as driver
from tools import run_moe_prefill_sweep as engine,run_decode_refresh as base
from test_moe_prefill_campaign import fake_resources,compose_fixture


@pytest.fixture(scope='module')
def oversize():
    w=Workload(12,1024,2048,16,'indexed','bf16',experts=256,topk=8,router='cluster')
    table,_,_=load_catalog()
    _,parents=refine_plan.parents(w,table,Prescreen(table))
    arms=[];seen=set()
    for parent in parents:
        for r in split_variants(w,parent):
            if r.kind!='tc' or rejection(r,w) or r.key in seen:continue
            seen.add(r.key);spec=module_spec(w,r)
            arms.append(dict(id=r.key,recipe=asdict(r),roles=['rule'],module=digest(spec),spec=spec))
    control=dict(arms[0],id='original',roles=['original'],required=True)
    return dict(id=w.key,workload=asdict(w),arms=[control,*arms],historical_references=[],
                sf_configs=[0,4,5],full_configs=[])


@pytest.mark.parametrize('occupancy',[1,2,4,8,12])
def test_every_geometry_split_and_algorithm_survives_first_screen(oversize,occupancy):
    images,queries=fake_resources(oversize,occupancy)
    domain=campaign.resolve(oversize,queries,images,inventory_only=True)
    first=staged.geometry_screen(domain)
    assert len(first['arms'])<=384
    assert {staged.group(a) for a in first['arms']}=={staged.group(a) for a in domain['arms']}
    assert first['arms'][0]['id']=='original'
    assert {a['id'] for a in first['arms']}<={a['id'] for a in domain['arms']}
    if occupancy>=2:
        with pytest.raises(ValueError,match='flat screen exceeds arm limit:.*resolved=.*limit=384'):
            campaign.resolve(oversize,queries,images)


def tiny_domain(router):
    w=Workload(12,1024,2048,16,'indexed','bf16',experts=256,topk=8,router=router)
    arms=[]
    for tile in (16,32,64):
        for grid in (72,144,216):
            r=Recipe.make('tc','sf',tm=tile,tn=64,tk=256,wm=16,wn=16,split=1,
                          persistent=-1,grid_mode=2,grid_b=grid//72)
            arms.append(dict(id=r.key,recipe=asdict(r),module=str(tile),roles=['rule'],
                binding=dict(launch=dict(algorithm=1,split=1,grid=grid,work=1000))))
    return dict(id=w.key,workload=asdict(w),arms=[dict(arms[0],id='original',roles=['original']),*arms])


def test_grid_refinement_unions_router_competitors_and_keeps_full_ladder():
    domains={};anchors={};records={}
    for router,best in (('cluster',16),('spread',32)):
        p=tiny_domain(router);pid=p['id'];domains[pid]=p
        chosen=[a['id'] for a in p['arms'] if staged.control(a) or a['binding']['launch']['grid']==216]
        anchors[pid]=staged.subset(p,chosen)
        records[pid]=dict(decision=dict(statistics={a['id']:dict(mean_us=
            10 if dict(a['recipe']['parameters'])['tm']==best else 30) for a in anchors[pid]['arms']}))
    result=staged.grid_screens(domains,anchors,records)
    ids=[]
    for p in result.values():
        assert {dict(a['recipe']['parameters'])['tm'] for a in p['arms']}=={16,32}
        assert {a['binding']['launch']['grid'] for a in p['arms']}=={72,144,216}
        ids.append({a['id'] for a in p['arms']})
    assert ids[0]==ids[1]


def test_missing_geometry_samples_are_not_a_candidate_rejection():
    p=tiny_domain('cluster')
    with pytest.raises(ValueError,match='complete measured screen'):
        staged.groups_to_expand(p,dict(decision=dict(statistics={})))


def test_completed_flat_screens_do_not_get_pruned_or_remeasured():
    ps=[tiny_domain(r) for r in ('cluster','spread')]
    domains={p['id']:p for p in ps}
    results={p['id']:dict(decision=dict(statistics={a['id']:dict(mean_us=10 if
        staged.control(a) else 99) for a in p['arms']})) for p in ps}
    selected=staged.grid_screens(domains,domains,results)
    for pid,p in selected.items():assert p['arms']==domains[pid]['arms']


def test_historical_recall_failure_never_appends_champion(oversize):
    images,queries=fake_resources(oversize,4)
    domain=campaign.resolve(oversize,queries,images,inventory_only=True)
    ref=next(a for a in domain['arms'] if not staged.control(a))
    seed=dict(oversize,historical_references=[dict(recipe=ref['recipe'])])
    assert staged.historical_coverage(seed,domain,queries)['references']==1
    before=deepcopy(domain)
    with pytest.raises(ValueError,match='omitted historical launch'):
        staged.historical_coverage(seed,staged.subset(domain,['original']),queries)
    assert before==domain


def test_retained_samples_are_first_valid_not_best_and_phase_bound(tmp_path):
    p,_=compose_fixture();run=dict(sha256='run');paths=[]
    for i,cost in enumerate((10,1)):
        path=tmp_path/f'{i}.json';paths.append(path)
        base.sealed(path,dict(status='PASS',run='run',point=p['id'],phase='screen',evidence={},
            data=dict(arms={'original':dict(cost=cost)},component_records={},expansions={},reset=None)))
    rows,proof,missing=staged.available_batches(paths,p,run,engine,phase='screen')
    assert rows[0]['arms']['original']['cost']==10 and len(proof)==1
    assert missing=={'rule-sf','rule-full'}
    with pytest.raises(ValueError,match='another timing phase'):
        staged.available_batches(paths,p,run,engine,phase='confirm')


def test_recovery_only_launches_missing_arms_and_keeps_frozen_child(tmp_path,monkeypatch):
    p,value=compose_fixture();run=dict(sha256='run',cards=['1']);calls=[]
    a=NS(output=tmp_path,worker=0,source=tmp_path/'frozen')
    folder=tmp_path/'gpu-1';old=folder/'batches/screen/old.json'
    retained=dict(value,arms={'original':value['arms']['original']})
    base.sealed(old,dict(status='PASS',run='run',point=p['id'],phase='screen',evidence={},data=retained))
    before=old.read_bytes()
    monkeypatch.setattr(retry,'controller',lambda d:dict(commit='host-controller'))
    def invoke(a,r,job,**kwargs):
        assert kwargs['entrypoint']==a.source/'tools/run_moe_confirmation.py'
        assert job['recovery_controller']==dict(commit='host-controller')
        calls.extend(job['arms'])
        return None,dict(status='PASS',run='run',point=p['id'],phase='screen',evidence={},
            data=dict(value,arms={k:value['arms'][k] for k in job['arms']}))
    monkeypatch.setattr(engine,'invocation',invoke)
    assert retry.measure(a,run,p,{},driver,staged,time.monotonic()+10,'geometry')
    assert calls==['rule-sf','rule-full'] and old.read_bytes()==before
    assert retry.measure(a,run,p,{},driver,staged,time.monotonic()+10,'geometry')
    assert calls==['rule-sf','rule-full']
    record=driver.sealed(folder/'geometry'/(p['id']+'.json'))
    assert record['decision']['statistics']['original']['mean_us']==15
    assert record['decision']['statistics']['rule-sf']['mean_us']==11


def test_clone_does_not_copy_failed_batches_or_touch_old_outputs(tmp_path):
    old=tmp_path/'old';out=tmp_path/'new';old.mkdir();out.mkdir()
    run=dict(sha256='run',cards=['1'])
    for name in ('run.json','plan.json','images.json'):(old/name).write_text('{}')
    base.sealed(old/'helper-gate.json',dict(status='PASS',run='run'))
    passed=old/'gpu-1/batches/screen/pass.json'
    failed=old/'gpu-1/batches/screen/fail.json'
    base.sealed(passed,dict(status='PASS',run='run',evidence={}))
    base.sealed(failed,dict(status='FAIL',run='run',error='resource limit'))
    before={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    counts=retry.clone_receipts(old,out,run,driver)
    assert counts['batches']==1 and (out/passed.relative_to(old)).is_file()
    assert not (out/failed.relative_to(old)).exists()
    assert before=={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}


def test_controller_archive_is_small_and_does_not_touch_submodules():
    text=(Path(__file__).resolve().parents[1]/'tools/retry_moe_refine_box.sh').read_text()
    assert 'git -C "$ROOT" archive "$REV" tools/retry_moe_refine.py' in text
    assert 'submodule update' not in text and 'build.sh' not in text
    assert 'MEASUREMENT_SOURCE' in text and 'RESUME_RETRY' in text
