"""Host coverage/accounting/recovery only; no fabricated PPU performance proof."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.native_components.smallm_ab import plan, verdict, aggregate, costs, ORDER, SCOPE
from dev.strategy.refresh import assign, family, request
from dev.strategy.m_coverage import points
from dev.strategy.decode_plan import original_recipe
from quactlize.dispatch.table import Workload, Recipe
from quactlize.dispatch.planning import plan_smallm
from quactlize.runtime.tuning import digest
from quactlize.runtime.compiler import sha

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def current(tmp_path_factory):return plan(tmp_path_factory.mktemp('smallm-native-plan'))


def test_complete_dense_scope_current_selector_not_gap_or_history(current,tmp_path):
    assert current['counts']['points']==360 and current['counts']['families']==45
    assert current['counts']['selected_configs']==360
    assert current['sha256']==digest({k:v for k,v in current.items() if k!='sha256'})
    assert {p['id'] for p in current['points']}=={p['id'] for p in points()}
    rows=plan_smallm(tmp_path,[request(Workload(**p['workload'])) for p in current['points']],dense_fallbacks=True)['requests']
    for p,r in zip(current['points'],rows):
        assert p['selection']==r and len(p['arms'])==1
        arm=p['arms'][0];assert arm['required'] and arm['id']=='original'
        selected=Recipe.from_dict(arm['recipe'])
        assert selected.kind in ('simt','q4','tc') and p['workload']['storage']=='f32'
        assert 1<=p['workload']['m']<=8 and p['workload']['operation']=='dense'
        if r.get('implementation',{}).get('q8_body'):
            assert dict(selected.parameters)['q8_body']==r['implementation']['q8_body']
    assert not current['historical_times_reused'] and not current['production_changed']


def test_four_cards_keep_entire_families_and_every_m_together(current):
    assignments=assign(current,4)
    assert len(assignments)==4 and sum(map(len,assignments))==45
    assert len({f for fs in assignments for f in fs})==45
    for fs in assignments:
        for fid in fs:
            assert {p['workload']['m'] for p in current['points'] if family(Workload(**p['workload']))==fid}==set(range(1,9))


def stages(main,helper=0):
    result=[dict(role='producer',node=dict(name='native_main',grid=[32,1,1]),samples_us=[[main]*5 for _ in range(6)])]
    if helper:result.append(dict(role='adapter',node=dict(name='cast'),samples_us=[[helper]*5 for _ in range(6)]))
    return result


def paired(native,candidate):return [dict(arm=a,components=deepcopy(native if a=='native' else candidate)) for a in ORDER]


def test_helpers_and_reducer_costs_cannot_disappear():
    rows=paired(stages(10,1),stages(8,4))
    rows[1]['components'][1]['role']=rows[2]['components'][1]['role']='reducer'
    result=verdict(rows)
    assert result['status']=='REGRESSION' and result['quactlize_us']==12 and result['native_us']==11
    assert result['quactlize_main_us']==8 and result['quactlize_reducer_us']==4
    assert result['native_helper_us']==1 and result['native_reducer_us']==0
    assert not result['component_no_regression'] and result['allowed_slowdown_pct']==0
    assert result['scope']==SCOPE and not result['model_tpot_admitted']


@pytest.mark.parametrize('candidate,status',[(9,'PASS'),(10,'PASS'),(10.01,'REGRESSION')])
def test_no_slowdown_allowance(candidate,status):
    assert verdict(paired(stages(10),stages(candidate)))['status']==status


@pytest.mark.parametrize('fault',['incomplete','order','kernel','zero','nan','rounds'])
def test_bad_measurement_is_not_a_performance_pass(fault):
    rows=paired(stages(10),stages(9))
    if fault=='incomplete':rows.pop()
    elif fault=='order':rows[0]['arm']='candidate'
    elif fault=='kernel':rows[2]['components'][0]['node']['name']='other'
    else:
        values=rows[1]['components'][0]['samples_us']
        if fault=='rounds':values.pop()
        else:values[0][0]=0 if fault=='zero' else float('nan')
    with pytest.raises(ValueError):verdict(rows)


def test_noisy_or_mixed_pair_cannot_close_regression():
    rows=paired(stages(10),stages(9));rows[2]['components']=stages(12)
    assert verdict(rows)['status']=='INCONCLUSIVE'
    rows=paired(stages(10),stages(9.99));rows[1]['components'][0]['samples_us'][0]=[10.01]*5
    assert verdict(rows)['status']=='INCONCLUSIVE'


def test_denominator_includes_regressions_and_missing_cells(current):
    records={p['id']:dict(status='PASS') for p in current['points']}
    assert aggregate(current,records)['component_no_regression']
    key=current['points'][0]['id'];records[key]['status']='REGRESSION'
    result=aggregate(current,records)
    assert result['status']=='COMPLETE' and not result['component_no_regression']
    records.pop(key);result=aggregate(current,records)
    assert result['status']=='INCOMPLETE' and result['measured']==359 and result['missing']==[key]


def test_plan_only_cli_needs_no_gpu_sdk_or_llama(tmp_path):
    out=tmp_path/'plan'
    run=subprocess.run([sys.executable,str(ROOT/'tools/run_dense_smallm_ab.py'),'--output',str(out),
        '--devices','0,1,2,3','--plan-only','--sdk','/missing','--llama-source','/missing'],
        capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'SMALLM_NATIVE_PLAN' in run.stdout
    assert json.loads((out/'plan.json').read_text())['counts']['points']==360
    assert not (out/'run.json').exists() and not (out/'native').exists()


def test_native_default_input_unchanged_but_paired_input_can_match_qz():
    source=(ROOT/'dev/native_components/carrier.cpp').read_text()
    assert 'request.value("timing_input_repeat",0)' in source
    assert 'timing_input==0||timing_input==1' in source
    assert 'point.set_input(timing_input);' in source
    # The existing prefill diagnostic source seam must still generate.
    from dev.prefill_point.build import carrier_source
    assert 'result["timing_input_repeat"]=timing_input' in carrier_source()


def test_frozen_comparison_keeps_unmodified_native_backend_environment(monkeypatch):
    from tools.run_dense_smallm_ab import environment
    monkeypatch.setenv('GGML_CUDA_FORCE_MMQ','1');monkeypatch.setenv('GGML_QUACTLIZE_DENSE','1')
    monkeypatch.setenv('GGML_CUDA_CUBLAS_ACCUMULATOR','f16')
    env=environment(SimpleNamespace(sdk=Path('/sdk'),llama_build=Path('/llama')))
    assert 'GGML_CUDA_FORCE_MMQ' not in env and 'GGML_CUDA_CUBLAS_ACCUMULATOR' not in env
    assert env['GGML_QUACTLIZE_DENSE']=='0' and env['GGML_CUDA_DISABLE_GRAPHS']=='1'


def test_candidate_process_failure_preserves_finished_points_and_restarts_rest(tmp_path,monkeypatch):
    from tools import run_dense_smallm_ab as runner
    rows=[dict(id=str(i)) for i in range(3)];calls=[]
    monkeypatch.setattr(runner,'command',lambda a:['test-only'])
    def execute(cmd,log,env):
        ids=[cmd[i+1] for i,v in enumerate(cmd) if v=='--point'];calls.append(ids)
        if len(calls)==1:
            runner.publish(tmp_path/'0.json',dict(status='MEASURED'))
            runner.publish(tmp_path/'1.json',dict(status='FAIL'))
            return 1
        runner.publish(tmp_path/'2.json',dict(status='MEASURED'));return 0
    monkeypatch.setattr(runner,'execute',execute)
    data=runner.candidate_pass(SimpleNamespace(shard=0),{},rows,tmp_path,'pass',{})
    assert calls==[['0','1','2'],['2']]
    assert [data[str(i)]['status'] for i in range(3)]==['MEASURED','FAIL','MEASURED']


def fake_pair(tmp_path,monkeypatch):
    # Stub only arithmetic proof parsing to focus on orchestration identity.
    # Real native/QZ proof validators have independent coverage and are reused.
    from tools import run_dense_smallm_ab as runner
    monkeypatch.setattr(runner,'candidate_components',lambda data,point:stages(9))
    monkeypatch.setattr(runner,'native_components',lambda data,point:stages(10))
    authority=dict(run='run',shard=0,device=dict(pci='host-double'))
    point=dict(id='point',workload={},selection={'policy':'current'})
    passes=[]
    for block,arm in enumerate(ORDER):
        path=tmp_path/f'{block}.json'
        if arm=='candidate':
            bound=authority|dict(pass_id=str(block))
            data=dict(arms={'original':dict(fixture=dict(fixture=dict(raw_sha256=['same'])))},
                      component_records={'stage':dict(identity=dict(authority=bound))})
            record=dict(status='MEASURED',arm=arm,block=block,authority=bound,
                        selection=point['selection'],data=data)
            runner.publish(path,record)
            record.update(raw_receipt=str(path),raw_sha256=sha(path))
        else:
            data=dict(authority=authority,timing_input_repeat=1,fixture=dict(raw_sha256=['same']),
                      components=[dict(identity=authority)])
            runner.publish(path,data)
            record=dict(status='MEASURED',arm=arm,block=block,data=data,path=str(path),sha256=sha(path))
        passes.append(record)
    return runner,point,passes,authority


@pytest.mark.parametrize('fault',['device','source','recipe','weight','input','stage-authority','order','duplicate'])
def test_comparison_rejects_cross_device_or_unmatched_receipts(tmp_path,monkeypatch,fault):
    runner,point,ps,authority=fake_pair(tmp_path,monkeypatch)
    if fault=='device':ps[1]['authority']=ps[1]['authority']|dict(device={'pci':'other'})
    elif fault=='source':ps[0]['data']['authority']=authority|dict(run='old')
    elif fault=='recipe':ps[1]['selection']={'policy':'old'}
    elif fault=='weight':ps[0]['data']['fixture']['raw_sha256']=['other']
    elif fault=='input':ps[0]['data']['timing_input_repeat']=0
    elif fault=='stage-authority':ps[0]['data']['components'][0]['identity']={'run':'other'}
    elif fault=='order':ps[1]['block']=0
    elif fault=='duplicate':ps[3]['path']=ps[0]['path']
    with pytest.raises(ValueError):runner.comparison(point,ps,authority)


@pytest.mark.parametrize('slow',[False,True])
def test_resume_recomputes_and_keeps_completed_even_regressions(tmp_path,monkeypatch,slow):
    runner,point,ps,authority=fake_pair(tmp_path,monkeypatch)
    if slow:monkeypatch.setattr(runner,'candidate_components',lambda *_:stages(11))
    value=runner.comparison(point,ps,authority)
    assert value['status']==('REGRESSION' if slow else 'PASS')
    path=tmp_path/'final.json';runner.publish(path,value);before=path.read_bytes()
    assert runner.completed(path,point,authority)['status']==value['status']
    assert path.read_bytes()==before
    (tmp_path/'0.json').write_text('changed')
    with pytest.raises(ValueError,match='raw paired measurement changed'):runner.completed(path,point,authority)


@pytest.mark.parametrize('bad_card',[None,'2'])
def test_coordinator_probes_assigned_cards_before_build_and_sets_worker_visibility(current,tmp_path,monkeypatch,bad_card):
    from tools import run_dense_smallm_ab as runner
    events=[];visibility=[];smi=tmp_path/'ppu-smi';smi.touch();binary=tmp_path/'carrier';binary.touch()
    a=SimpleNamespace(output=tmp_path,sdk=Path('/sdk'),llama_source=Path('/llama'),
        llama_build=Path('/llama/build'),devices='0,1,2,3',jobs=8,l2_bytes=67108864,
        plan_only=False,resume=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5,6,7')
    monkeypatch.setattr(runner,'plan',lambda _:current)
    monkeypatch.setattr(runner.refresh,'selected_smi',lambda _:smi)
    def execute(cmd,log,env):
        card=env['CUDA_VISIBLE_DEVICES'];events.append('probe-'+card)
        if card==bad_card:return 1
        runner.atomic_json(Path(cmd[cmd.index('--probe')+1]),dict(sdk='test-sdk',l2={'bytes':67108864},
            device=dict(host='host-double',pci=card,compute_units=72,warp_size=32,max_threads=1024,
                        DriverGetVersion=13000,RuntimeGetVersion=13000)))
        return 0
    monkeypatch.setattr(runner,'execute',execute)
    monkeypatch.setattr(runner.refresh,'compile_all',lambda *_:events.append('compile') or ({'tc':{},'vectors':{}},None))
    monkeypatch.setattr(runner.native,'build_adapter',lambda *_:(binary,{}))
    monkeypatch.setattr(runner.native,'carrier',lambda *_:binary)
    monkeypatch.setattr(runner.refresh,'lock_clocks',lambda a,devices:events.append('lock'))
    class Finished:
        returncode=0
        def poll(self):return 0
        def wait(self,timeout):return 0
    def popen(cmd,**kwargs):
        visibility.append(kwargs['env']['CUDA_VISIBLE_DEVICES']);return Finished()
    monkeypatch.setattr(runner.subprocess,'Popen',popen)
    monkeypatch.setattr(runner,'review',lambda a,p,r,c:1)  # no device measurement was made
    if bad_card:
        with pytest.raises(ValueError,match='device precheck failed'):runner.coordinate(a,['0','1','2','3'])
        assert events==['probe-0','probe-1','probe-2'] and not visibility
    else:
        assert runner.coordinate(a,['0','1','2','3'])==1
        assert events==['probe-0','probe-1','probe-2','probe-3','compile','lock']
        assert visibility==['0','1','2','3']
