"""Host contracts only: no new device timing or production admission."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from dev.prefill_asys.finalists import full_case, case_identity, load, bind_cases, partition
from dev.prefill_asys.q8_refine import make_plan, validate, proposals, sf_case, dequant_diagnostics
from dev.strategy.resources import inventory
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
from tools.run_q8_tp_refine import helper_environment, package
from tools import run_dense_dequant as dq

ROOT = Path(__file__).resolve().parents[1]


def seal(row, key='sha256'):
    row[key]=digest({k:v for k,v in row.items() if k!=key})
    return row


def review():
    r=dict(schema='quactlize.prefill-component-review.v1',points={},candidates={},
        authorities={'device':dict(device=dict(device=dict(compute_units=72)))},incomplete=[],pending_points=[])
    for n,k in ((256,3072),(1024,3072),(4096,2048),(8192,3072)):
        for m in (128,512,2048):
            w=Workload(8,n,k,m)
            sf=sf_case((64,64,64,3));sf['helper_options']=dict(vector_io=-1)
            rows=[]
            for case,us in ((sf,40.),(full_case(38,1),30.),(full_case(41,1),31.)):
                cid=digest(case);r['candidates'][cid]=case
                rows.append(dict(candidate=cid,arm=case['arm'],helper_mode=case['helper_mode'],
                    status='PASS',gpu_sum_us=us,confirmation_captures=[{},{}],phases_us=dict(expansion=n/512)))
            rows.append(dict(candidate='native',arm='native-fp16',status='PASS',gpu_sum_us=32.,
                             confirmation_captures=[{},{}],phases_us=dict(expansion=4.)))
            r['points'][w.key]=dict(workload=asdict(w),consumers=[],authority='device',measurements=rows)
    return seal(r,'review_sha256')


def test_bounded_basis_is_blind_but_all_measured_controls_survive():
    r=review();p=make_plan(r)
    assert load(p)==p and p['counts']['points']==12 and p['counts']['modules']==10
    assert max(len(x['cases']) for x in p['points'])<=16
    assert not p['production_changed'] and not p['performance_admitted']
    for point in p['points']:
        cases={c['id']:c for c in point['cases']}
        assert point['prior_winner'] in cases
        assert set(point['prior_route_winners'].values())<=cases.keys()
        assert {38,41}<={c['helper_options'].get('dequant_variant') for c in cases.values()}
        assert all('gpu_sum_us' not in c for c in cases.values())
    changed=copy.deepcopy(r)
    for old in changed['points'].values():
        old['measurements'][0]['gpu_sum_us']=1.  # different champion, same blind proposal IDs
    other=make_plan(seal(changed,'review_sha256'))
    for a,b in zip(p['points'],other['points']):
        rule_ids=lambda x:{c['id'] for c in x['cases'] if any(s.startswith('rule:') for s in c['roles'])}
        assert rule_ids(a)==rule_ids(b)
        assert a['prior_winner']!=b['prior_winner']


def test_split_is_bounded_and_persistent_grid_uses_actual_probed_resources():
    small=Workload(8,256,3072,128);large=Workload(8,8192,3072,2048)
    assert {dict(c['recipe']['parameters'])['split'] for c,_ in proposals(small,72)}=={1,2,4}
    assert {dict(c['recipe']['parameters'])['split'] for c,_ in proposals(large,72)}=={1}
    p=make_plan(review());inv=inventory(p)
    # Changing the runtime CU count changes persistent launch geometry, not
    # the stored recipe. No grid hardcoded from the historical ZW810 count.
    def bound(cu):
        receipts={key:dict(status='READY',request=v['request'],device=dict(compute_units=cu),occupancy=1)
                  for key,v in inv['queries'].items()}
        point=next(x for x in p['points'] if x['workload']['n']==8192 and x['workload']['m']==2048)
        return [c for c in bind_cases(point,receipts) if dict(c['recipe']['parameters']).get('grid_mode')==2]
    assert bound(64)[0]['launch']['grid']==64 and bound(72)[0]['launch']['grid']==72
    parts=[partition(p['points'],i,4) for i in range(4)]
    assert sorted(x['id'] for part in parts for x in part)==sorted(x['id'] for x in p['points'])


@pytest.mark.parametrize('fault',['rule','winner','precision','helper','confirmation','resources','budget'])
def test_resealed_bad_plan_is_rejected(fault):
    p=make_plan(review());point=p['points'][0]
    if fault=='rule':point['cases']=[c for c in point['cases'] if not any(s.startswith('rule:') for s in c['roles'])]
    elif fault=='winner':point['prior_winner']='missing'
    elif fault=='precision':point['workload']['compute']='bf16'
    elif fault=='helper':point['cases'][0]['helper_options']['vector_io']=99
    elif fault=='confirmation':point['confirmation_captures']=1
    elif fault=='resources':point['candidates']=[]
    elif fault=='budget':p['max_cases']=99
    with pytest.raises(ValueError):validate(seal(p))


def test_old_partial_or_unconfirmed_review_is_not_reused():
    r=review();r['pending_points']=['missing']
    with pytest.raises(ValueError):make_plan(seal(r,'review_sha256'))
    r=review();next(iter(r['points'].values()))['measurements'][0]['confirmation_captures']=[]
    with pytest.raises(ValueError):make_plan(seal(r,'review_sha256'))


def test_whole_call_and_helper_plans_run_host_only(tmp_path):
    r=review();path=tmp_path/'review.json';path.write_text(json.dumps(r));output=tmp_path/'out'
    subprocess.run([sys.executable,str(ROOT/'tools/run_q8_tp_refine.py'),'--review',str(path),
        '--output',str(output),'--sdk','/no-sdk','--llama-source','/no-llama'],
        check=True,capture_output=True)
    p=load(json.loads((output/'plan.json').read_text()))
    diagnostics=json.loads((output/'dequant.json').read_text())
    assert diagnostics==dequant_diagnostics(r,p) and len(diagnostics['points'])<=6
    assert len({(x['n'],x['k']) for x in diagnostics['points']})==len(diagnostics['points'])
    assert diagnostics['points'][-1]['role']=='comparison-control'
    subprocess.run([sys.executable,str(ROOT/'tools/run_prefill_model_asys.py'),
        '--finalist-plan',str(output/'plan.json'),'--output',str(tmp_path/'dry'),
        '--plan-only','--sdk','/no-sdk','--llama-source','/no-llama'],check=True,capture_output=True)
    assert not (output/'build').exists()


def test_custom_q8_shapes_do_not_change_old_shape_indices_or_profile_binding(tmp_path):
    old=dq.POINTS
    shapes=dq.q8_shapes('8192x3072,4096x2048')
    p=dq.plan([0,1],67108864,q8=True,native=True,shapes=shapes)
    assert p['points'][0]==dict(index=0,q=8,n=8192,k=3072) and dq.POINTS==old
    assert p['variants']==[-1,0,22,23,38,41]
    symbol='_ZN9quactlize7dequant22dense_dequant_q8_pairedILb0ELj128EEEvPKtS3_Ptii'
    config=dq.profile_entry(dict(kernels={symbol:{}}),0,41,0,shapes=shapes)
    assert config==dict(symbol=symbol,grid=[128,24,1],block=[128,1,1])
    native=tmp_path/'libggml-cuda.so';native.write_bytes(b'host-only plan, not loaded')
    output=tmp_path/'diagnostic'
    subprocess.run([sys.executable,dq.__file__,'--output',str(output),'--plan-only',
        '--q8','--native-library',str(native),'--q8-shapes','8192x3072,4096x2048'],
        check=True,capture_output=True)
    assert json.loads((output/'plan.json').read_text())==dq.plan([0,1],0,q8=True,native=True,shapes=shapes)
    assert dq.q8_shapes('256x256')==((8,256,256),)
    for text in ('8192,3072','0x128','65x128','64x127','128x512','512x128',
                 '8192x16384','64x128,64x128','64x128;exit'):
        with pytest.raises(ValueError):dq.q8_shapes(text)


def test_environment_cannot_redirect_helper_or_inherit_old_point_filter(tmp_path):
    old=dict(RUN='/old',RESUME_RUN='/old',POINTS='7',PROFILE_POINTS='7',NATIVE_LIBRARY='/old.so',Q8_EXCHANGE='1',Q8_FAST_TAIL='1',Q8_HALF2='1',Q8_PAIRED='1')
    d=dict(shapes='8192x3072',profile_points=[0])
    env=helper_environment(old,tmp_path,d,'7','revision',True)
    assert env['GPU']=='7' and env['Q8_SHAPES']==d['shapes'] and env['SOURCE_REV']=='revision'
    assert env['PROFILE_POINTS']=='0' and env['Q8_NATIVE']=='1' and env['Q8']=='1'
    assert not {'RUN','RESUME_RUN','POINTS','NATIVE_LIBRARY','Q8_EXCHANGE','Q8_FAST_TAIL','Q8_HALF2','Q8_PAIRED'}&env.keys()


def test_package_excludes_sources_binaries_and_raw_profiles(tmp_path):
    (tmp_path/'request.json').write_text('{}')
    (tmp_path/'timing').mkdir();(tmp_path/'timing/review-1.json.xz').write_bytes(b'compact')
    (tmp_path/'timing/trace.asysrep').write_bytes(b'large')
    (tmp_path/'kernel.so').write_bytes(b'binary')
    path=package(tmp_path)
    with tarfile.open(path) as archive:
        assert archive.getnames()==['request.json','timing/review-1.json.xz']


@pytest.mark.parametrize('sf_rc',[0,1])
def test_coordinator_waits_for_sf_before_helper_and_does_not_hide_failures(tmp_path,monkeypatch,sf_rc):
    from tools import run_q8_tp_refine as runner
    path=tmp_path/'review.json';path.write_text(json.dumps(review()))
    output=tmp_path/'run';calls=[]
    def run(command,env):
        calls.append((command,env))
        if len(calls)==1:
            assert 'run_prefill_cohort_multi.py' in command[2]
            assert command[command.index('--gpus')+1]=='1,2,3,4'
            return sf_rc
        assert len(calls)==2 and command[0]=='bash'
        assert env['GPU']=='1' and env['PROFILE_ACU']=='1'
        assert 'NATIVE_LIBRARY' not in env  # uses the same explicit llama build
        return 0
    monkeypatch.setattr(runner,'run_coordinator',run)
    monkeypatch.setattr(sys,'argv',[runner.__file__,'--review',str(path),'--output',str(output),
        '--gpus','1,2,3,4,5,7','--run','--confirm-idle','--profile-acu'])
    assert runner.main()==sf_rc and len(calls)==2
    result=json.loads((output/'status.json').read_text())
    assert result['phases']==dict(sf=sf_rc,dequant=0)
    assert result['status']==('INCOMPLETE' if sf_rc else 'MEASURED_NOT_ADMITTED')
    assert result['not_measured_in_this_invocation']==[]
