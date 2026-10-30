"""Native request serialization and continuation, without device execution."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from dev.moe_confirm import native_ab,plan
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest
from tools import run_moe_confirmation as runner


@pytest.mark.parametrize('n',[2048,3072])
@pytest.mark.parametrize('router',plan.ROUTERS)
def test_real_native_request_reaches_carrier_with_full_recipe(tmp_path,monkeypatch,n,router):
    """Keep point_request, activation and JSON serialization real: the failed seam."""
    w=Workload(13,n,512,5120,'indexed','bf16',experts=256,topk=8,channels=8,router=router)
    p=dict(id=w.key,workload=asdict(w),arms=[dict(id='original',recipe=asdict(
        Recipe.make('full','full',dequant_config=11,provider='bf16-default')))])
    original=deepcopy(p)
    a=NS(sdk=tmp_path,worker=0)
    run=dict(sha256='run',native=dict(runtime='runtime',adapter='adapter',adapter_sha256='adapter-hash',
        carrier='carrier',carrier_sha256='carrier-hash',backend={},llama_build=str(tmp_path)),
        devices=[dict(device=dict(pci='host-test'))],source_revision={},harness=dict(sha256='harness'),
        l2_bytes=67108864,smi='smi',asys='asys')
    monkeypatch.setattr(native_ab,'weights',lambda *a:(tmp_path/'raw',{}))
    monkeypatch.setattr(native_ab.base,'observe',lambda *a,**kw:{})
    class StopBeforeGPU(RuntimeError):pass
    seen=[]
    def execute(command,log,**kw):
        request=json.loads((log.parent/'request.json').read_text())
        row=request['points'][0];seen.append(row)
        assert row['cohort']=='prefill' and row['workload']==p['workload']
        assert len(row['owners'])==5120*8
        assert len(row['input_fixture']['categories'])==512
        assert all(len(c)==5120*8*4 for c in row['input_fixture']['coefficients'])
        assert row['ring']['active_read_bytes']>=2.25*run['l2_bytes']
        assert request['timing_input_repeat']==0 and request['rounds']==6 and request['samples']==5
        raise StopBeforeGPU('request is ready')
    monkeypatch.setattr(native_ab,'execute',execute)
    with pytest.raises(StopBeforeGPU,match='ready'):native_ab.native_pass(a,run,p,tmp_path,0)
    assert len(seen)==1 and p==original  # Do not mutate the frozen selection.


def confirmed_fixture(tmp_path):
    points=[];records={};selected={}
    for router in plan.ROUTERS:
        w=Workload(13,2048,512,5120,'indexed','bf16',experts=256,topk=8,channels=8,router=router)
        arms=[dict(id=aid,recipe=asdict(Recipe.make(kind,route,dequant_config=config,tm=tm)),
                   roles=['original' if aid=='original' else 'rule'],module='host-only-'+aid,spec={'host_test':True})
              for aid,kind,route,config,tm in [('original','tc','sf',5,128),('new-sf','tc','sf',5,64),
                                               ('full','full','full',11,0)]]
        p=dict(id=w.key,workload=asdict(w),arms=arms);points.append(p)
        costs=dict(original=20.,**{'new-sf':9. if router=='cluster' else 8.,
                                  'full':5. if router=='cluster' else 10.})
        records[p['id']]=dict(status='PASS',point=p['id'],workload=p['workload'],batch_receipts={},
            decision=dict(statistics={aid:dict(mean_us=v) for aid,v in costs.items()}),
            bindings={a['id']:dict(recipe=a['recipe'],helper_config=dict(a['recipe']['parameters'])['dequant_config']) for a in arms})
        selected[p['id']]=dict(p,arms=[dict(arms[-1],id='original')],sf_configs=[],full_configs=[11],
            selection_source='MEASURED_REFINEMENT_NOT_PRODUCTION')
    runner.base.sealed(tmp_path/'plan.json',dict(schema='quactlize.moe-prefill-refine.v1',points=points))
    original=runner.sealed(tmp_path/'plan.json')
    runner.base.sealed(tmp_path/'run.json',dict(mode='refine-prefill',plan=original['sha256'],native_comparison=True,
        cards=['0'],assignments=[[p['id'] for p in points]],source_revision={'host_test':True}))
    run=runner.sealed(tmp_path/'run.json')
    for pid,value in records.items():runner.base.sealed(tmp_path/'gpu-0/points'/(pid+'.json'),dict(value,run=run['sha256']))
    runner.base.sealed(tmp_path/'finalists.json',dict(run=run['sha256'],points={p['id']:p for p in points}))
    runner.base.sealed(tmp_path/'selection.json',dict(run=run['sha256'],points=selected))
    return run


@pytest.mark.parametrize('route,kind',[('selected','full'),('sf','tc'),('full','full')])
def test_retry_is_one_common_confirmed_recipe_without_sweep_or_old_timings(tmp_path,route,kind):
    old=confirmed_fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    result=native_ab.retry_plan(tmp_path,route)
    assert result['native_only'] and result['native_comparison'] and not result['historical_times_reused']
    assert result['origin']['run']==old['sha256'] and result['origin']['requested_route']==route
    assert result['counts']['arms']==result['counts']['points']==5
    assert result['counts']['candidate_passes']==result['counts']['native_passes']==10
    assert {p['arms'][0]['recipe']['kind'] for p in result['points']}=={kind}
    assert len({digest(p['arms'][0]['recipe']) for p in result['points']})==1
    assert all(len(p['arms'])==1 and p['arms'][0]['id']=='original' for p in result['points'])
    assert 'mean_us' not in json.dumps(result)
    assert all(p.read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('fault',['missing','failed','selection','origin'])
def test_unconfirmed_or_changed_selection_cannot_be_reused(tmp_path,fault):
    confirmed_fixture(tmp_path)
    file=next((tmp_path/'gpu-0/points').glob('*.json'))
    if fault=='missing':file.rename(file.with_suffix('.saved'))
    elif fault=='failed':
        value=runner.sealed(file);runner.base.sealed(file,dict(value,status='FAIL'))
    else:
        file=tmp_path/'selection.json';value=runner.sealed(file)
        if fault=='origin':value['run']='other-run'
        else:next(iter(value['points'].values()))['arms'][0]['recipe']['route']='sf'
        runner.base.sealed(file,value)
    with pytest.raises(ValueError):native_ab.retry_plan(tmp_path,'sf')


@pytest.mark.parametrize('changed',[None,'quactlize/moe/scaffold.cu','dev/strategy/prefill_fixture.py'])
def test_native_build_reuse_allows_only_request_coordinator_fix(tmp_path,monkeypatch,changed):
    from dev.strategy import prefill_build
    old=confirmed_fixture(tmp_path)
    sources={'quactlize/moe/scaffold.cu':'kernel','tools/run_moe_confirmation.py':'old-runner'}
    harness={'dev/strategy/prefill_fixture.py':'fixture','dev/moe_confirm/native_ab.py':'old-request'}
    old.update(source=dict(files=sources),harness=dict(files=harness),immutable_files={},
        moe_helpers={},helpers=str(tmp_path/'helpers'),dequant=None,sdk='sdk',native={'host_test':'existing-native'})
    runner.base.sealed(tmp_path/'run.json',old);newold=runner.sealed(tmp_path/'run.json')
    # Rebind host-only test receipts after adding build metadata.
    for name in ('selection.json','finalists.json'):
        value=runner.sealed(tmp_path/name);runner.base.sealed(tmp_path/name,dict(value,run=newold['sha256']))
    for file in (tmp_path/'gpu-0/points').glob('*.json'):
        value=runner.sealed(file);runner.base.sealed(file,dict(value,run=newold['sha256']))
    runner.base.atomic_json(tmp_path/'images.json',{'tc':{},'vectors':{}})
    p=native_ab.retry_plan(tmp_path)
    current_source=dict(sources,**{'tools/run_moe_confirmation.py':'new-runner'})
    current_harness=dict(harness,**{'dev/moe_confirm/native_ab.py':'new-request'})
    if changed in current_source:current_source[changed]='changed'
    if changed in current_harness:current_harness[changed]='changed'
    monkeypatch.setattr(runner,'source_identity',lambda *a:dict(files=current_source))
    monkeypatch.setattr(runner,'harness_identity',lambda *a:dict(files=current_harness))
    monkeypatch.setattr(runner,'verify_moe',lambda *a:None)
    monkeypatch.setattr(prefill_build,'verify',lambda *a:None)
    calls=[];monkeypatch.setattr(runner.base,'verify_images',lambda *a,**k:calls.append('verified'))
    a=NS(native_from=tmp_path,reuse_build=None,sdk=tmp_path,output=tmp_path/'new')
    if changed:
        with pytest.raises(ValueError,match='non-coordinator'):runner.reuse_build(a,p)
    else:
        result=runner.reuse_build(a,p)
        assert calls==['verified'] and result[-1]['native']==old['native']
        assert result[-1]['compiled_artifacts_only'] and not result[-1]['previous_measurements_reused']


def test_native_only_runs_no_screen_or_confirmation(tmp_path,monkeypatch):
    from dev.moe_confirm import refine
    p=dict(points=[dict(id='p')],origin={'source':'old-selection'},scope='native-only')
    run=dict(sha256='new-run');a=NS(output=tmp_path,budget_hours=1)
    seen=[];monkeypatch.setattr(refine,'spawn',lambda a,r,stage,deadline:seen.append(stage) or [0])
    def review(a,p,r,codes):
        runner.base.atomic_json(a.output/'native-summary.json',dict(status='PASS'));return 0
    monkeypatch.setattr(native_ab,'review',review)
    assert native_ab.run_retry(a,p,run)==0 and seen==['native']
    assert runner.sealed(tmp_path/'selection.json')['run']=='new-run'
    assert runner.base.read(tmp_path/'summary.json')['origin']==p['origin']
