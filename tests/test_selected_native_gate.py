"""Host/synthetic control tests only; never presented as PPU timing evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.native_components import selected_gate as gate
from dev.native_components.asys import parse_rows as native_parse
from dev.strategy.smallm_asys import METHOD
from quactlize.runtime.tuning import digest
from quactlize.runtime.compiler import sha
from tools import run_smallm_native_gate as runner
from tools.run_smallm_regressions_asys import sealed
from tools.run_smallm_native_asys import authority
from test_smallm_native_asys import fixture as native_fixture
from test_smallm_asys import capture_fixture,provenance
from dev.strategy.asys_components import parse_rows as candidate_parse

ROOT=Path(__file__).resolve().parents[1]


def stages(main,helper=0):
    cs=[dict(role='producer',node=dict(symbol='producer'),timing_method=METHOD,
             samples_us=[[main]*5 for _ in range(6)])]
    if helper:cs.append(dict(role='adapter',node=dict(symbol='cast'),timing_method=METHOD,
                            samples_us=[[helper]*5 for _ in range(6)]))
    return cs


def paired(native,candidate):
    return [dict(arm=a,components=deepcopy(native if a=='native' else candidate)) for a in gate.ORDER]


@pytest.fixture(scope='module')
def plan(tmp_path_factory):return gate.plan(tmp_path_factory.mktemp('selected-native'))


def test_complete_current_selected_config_denominator(plan):
    assert plan['counts']['points']==360 and plan['counts']['families']==45
    assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'})
    assert plan['timing_method']==METHOD and not plan['historical_times_reused'] and not plan['config_sweep']
    assert {p['workload']['m'] for p in plan['points']}==set(range(1,9))
    assert all(len(p['arms'])==1 and p['arms'][0]['id']=='original' and p['arms'][0]['required'] for p in plan['points'])
    groups=runner.assignments(plan,4)
    assert len({pid for group in groups for pid in group})==sum(map(len,groups))==360
    from dev.strategy.refresh import family
    from quactlize.dispatch.table import Workload
    owners={}
    for i,group in enumerate(groups):
        for p in plan['points']:
            if p['id'] in group:
                key=family(Workload(**p['workload']))
                assert owners.setdefault(key,i)==i


def test_separate_helper_and_reducer_can_make_faster_main_a_regression():
    rows=paired(stages(10,1),stages(8,4))
    for i in (1,2):rows[i]['components'][1]['role']='reducer'
    result=gate.verdict(rows)
    assert result['status']=='REGRESSION'
    assert (result['native_us'],result['quactlize_us'])==(11,12)
    assert result['quactlize_main_us']==8 and result['quactlize_reducer_us']==4
    assert result['native_helper_us']==1 and result['allowed_slowdown_pct']==0


def test_no_noise_veto_no_minima_no_average_across_shapes(plan):
    rows=paired(stages(10),stages(9))
    rows[1]['components'][0]['samples_us']=[[5,12,12,12,12]]*6
    rows[2]['components'][0]['samples_us']=[[5,12,12,12,12]]*6
    assert gate.verdict(rows)['status']=='REGRESSION'  # min is faster, mean is slower
    rows=paired(stages(10),stages(9))
    rows[2]['components'][0]['samples_us']=[[10.5]*5]*6
    assert gate.verdict(rows)['status']=='PASS'  # noisy but mean is faster
    records={p['id']:dict(status='PASS') for p in plan['points']}
    assert gate.aggregate(plan,records)['component_no_regression']
    pid=plan['points'][0]['id'];records[pid]['status']='REGRESSION'
    assert gate.aggregate(plan,records)['status']=='REGRESSION'
    del records[pid]
    assert gate.aggregate(plan,records)['status']=='INCOMPLETE'


@pytest.mark.parametrize('fault',['order','missing','event','nan','rounds','inventory'])
def test_invalid_abba_cannot_pass(fault):
    rows=paired(stages(10),stages(9))
    if fault=='order':rows[0]['arm']='quactlize'
    elif fault=='missing':rows.pop()
    elif fault=='inventory':rows[2]['components'][0]['node']['symbol']='wrong'
    elif fault=='event':rows[1]['components'][0]['timing_method']='CUDA_EVENTS'
    elif fault=='nan':rows[1]['components'][0]['samples_us'][0][0]=float('nan')
    else:rows[1]['components'][0]['samples_us'].pop()
    with pytest.raises(ValueError):gate.verdict(rows)


def evidence():
    native,point,_,nrows=native_fixture()
    point.update(selection=dict(test='synthetic-selected-TC'),arms=[dict(id='original',
        recipe=dict(kind='tc'),spec={})])
    run=dict(sha256='f'*64,source_revision=dict(quactlize_commit='a'*40,quactlize_tree='b'*40,
        actlize_commit='c'*40),source=dict(sha256='d'*64),harness=dict(sha256='d'*64),
        images={},devices=[dict(pci='host-synthetic')],sdk=dict(release='test'),asys_sha256='d'*64,
        adapter_sha256='d'*64,carrier_sha256='d'*64,backend={'lib':'d'*64},llama_head='e'*40)
    passes=[]
    for block,arm in enumerate(gate.ORDER):
        p=provenance()|dict(device=run['devices'][0],images_sha256=digest(run['images']),
            trace_sha256=digest(['trace',block]),sqlite_sha256=digest(['sqlite',block]),timing_method=METHOD)
        if arm=='native':
            auth=authority(run,0);data=deepcopy(native);data['authority']=auth
            for c in data['components']:c['identity']=auth
            p.update(backend=run['backend'],llama_head=run['llama_head'],
                native_adapter_sha256=run['adapter_sha256'],native_carrier_sha256=run['carrier_sha256'])
            data=native_parse(data,point,auth,nrows,p)
            data.update(timing_input_repeat=1,fixture=dict(raw_sha256=['weights']))
        else:
            data,rows=capture_fixture();data['point']=point['id']
            data['arms'].pop('fast');data['roles'].pop('fast');data['arm_components'].pop('fast')
            auth=dict(run=run['sha256'],shard=0,device=run['devices'][0]);keys={}
            old=data['component_records'];keep=set(data['arm_components']['original'])
            data['component_records']={}
            for cid,c in old.items():
                if cid not in keep:continue
                c['identity']['authority']=auth
                if c['kind']=='producer':
                    c['identity']['node'].update(name='void cutlass::Kernel<Fake>(call)',symbol='cutlass::Kernel<Fake>')
                key=digest(c['identity']);keys[cid]=key;data['component_records'][key]=c
            data['components']=list(keys.values())
            data['arm_components']['original']=[keys[k] for k in data['arm_components']['original']]
            data['trace_schedule']=[dict(s,component=keys[s['component']]) for s in data['trace_schedule'] if s['component'] in keys]
            data['arms']['original']['fixture']['fixture']=dict(raw_sha256=['weights'])
            rows=[r for r in rows if r['name']!='fast']
            for row in rows:
                if row['name']=='original':row['symbol']='cutlass::Kernel<Fake>'
            data=candidate_parse(data,rows,p)
        passes.append(dict(status='MEASURED',point=point['id'],arm=arm,block=block,
                           selection=point['selection'],data=data))
    return point,passes,run


def test_real_parsers_and_gate_bind_all_four_receipts():
    point,passes,run=evidence()
    result=gate.comparison(point,passes,run,0)
    assert result['status']=='PASS' and result['native_us']==result['quactlize_us']==10
    assert result['native_activation_contract']=='Q8_1_QUANTIZED_FROM_F32'
    assert not result['arithmetic_equivalence_claimed'] and not result['model_tpot_admitted']


@pytest.mark.parametrize('fault',['raw','source','device','binary','weights','trace','selection','helper','input'])
def test_comparison_rejects_corrupt_or_mismatched_evidence(fault):
    point,passes,run=evidence()
    if fault=='raw':passes[0]['data']['components'][0]['trace_slices'][0]['durations_ns'][0]+=1
    elif fault=='source':passes[1]['data']['latency_provenance']['quactlize_commit']='e'*40
    elif fault=='device':passes[3]['data']['latency_provenance']['device']={}
    elif fault=='binary':passes[3]['data']['latency_provenance']['backend']={}
    elif fault=='weights':passes[3]['data']['fixture']['raw_sha256']=['other']
    elif fault=='trace':
        for name in ('trace_sha256','sqlite_sha256'):
            passes[3]['data']['latency_provenance'][name]=passes[0]['data']['latency_provenance'][name]
    elif fault=='selection':passes[1]['selection']={}
    elif fault=='helper':passes[0]['data']['components'].pop(0)
    else:passes[0]['data']['timing_input_repeat']=0
    with pytest.raises(ValueError):gate.comparison(point,passes,run,0)


def test_resume_retains_regression_and_detects_changed_raw_files(tmp_path):
    point,passes,run=evidence();paths=[];evidence_hashes={}
    # Slower selected result on BOTH passes; persist and recheck, never retry.
    for i in (0,3):
        for c in passes[i]['data']['components']:
            for row in c['trace_slices']:
                row['durations_ns']=[x//2 for x in row['durations_ns']]
                for key in ('mean_us','min_us','max_us'):row[key]/=2
            c['samples_us']=[[x/2 for x in row] for row in c['samples_us']];c['mean_us']/=2
    result=gate.comparison(point,passes,run,0);assert result['status']=='REGRESSION'
    for i,data in enumerate(passes):
        path=tmp_path/f'pass-{i}.json';runner.base.sealed(path,data);paths.append(str(path));evidence_hashes[str(path)]=sha(path)
    path=tmp_path/'point.json'
    runner.base.sealed(path,dict(status='REGRESSION',point=point['id'],run=run['sha256'],
        evidence=evidence_hashes,passes=paths,comparison=result))
    assert runner.load_saved(path,point,run,0)['status']=='REGRESSION'
    Path(paths[0]).write_text('{}')
    with pytest.raises(ValueError,match='changed'):runner.load_saved(path,point,run,0)


def test_plan_only_needs_no_sdk_or_llama(tmp_path):
    out=tmp_path/'gate'
    result=subprocess.run([sys.executable,str(ROOT/'tools/run_smallm_native_gate.py'),
        '--output',str(out),'--sdk','/missing','--plan-only'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert sealed(out/'plan.json')['counts']['points']==360
    assert not (out/'build').exists() and not (out/'run.json').exists()
    from tools.pack_simt_split_review import result_root
    assert result_root(out)==out


def test_csv_keeps_missing_points_and_records_commit_recipe(tmp_path,plan):
    ps=plan['points'][:2];path=tmp_path/'comparison.csv'
    runner.comparison_csv(path,ps,{ps[0]['id']:dict(status='REGRESSION',native_us=10,quactlize_us=11)},'a'*40)
    with path.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==2 and [r['status'] for r in rows]==['REGRESSION','MISSING']
    assert all(r['quactlize_commit']=='a'*40 for r in rows)
    assert json.loads(rows[0]['recipe'])==ps[0]['arms'][0]['recipe']
    assert rows[1]['native_us']=='' and rows[1]['quactlize_us']==''


def test_native_request_matches_input_one_and_keys_fixture_by_weight(monkeypatch,tmp_path):
    point,_,run=evidence();run['l2_bytes']=67108864;run['runtime']='runtime';run['adapter']='adapter'
    folders=[]
    def fake_weights(w,folder):
        folders.append(folder)
        return folder/'weights.raw',dict(raw_sha256=['host-only'])
    monkeypatch.setattr(runner,'weights',fake_weights)
    a=runner.native_request(point,run,0,tmp_path)
    b=runner.native_request(dict(point,workload=dict(point['workload'],n=2048,output_stride=2048)),run,0,tmp_path)
    assert folders[0]!=folders[1]
    assert a['timing_input_repeat']==b['timing_input_repeat']==1
    assert a['timing_method']==METHOD and (a['rounds'],a['samples'])==(6,5)
    assert a['authority']==authority(run,0) and len(a['points'])==1
    assert len(a['points'][0]['input_fixture']['coefficients'])==2
