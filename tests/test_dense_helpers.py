"""Helper closeout coverage, pruning, resume and profile boundaries; no GPU needed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from dev.dense_helpers.plan import cohort,jobs,variants,helper_options,full_plan,validate_full_plan,validate_result
from quactlize.runtime.tuning import digest
from tools.run_dense_helpers import assign,package
from tools.run_prefill_model_asys import confirmed_capture,summarize,helper_review


def measurements(families):
    result={}
    for key,job in jobs(families).items():
        vs=variants(job['q']) if job['kind']=='dequant' else range(4)
        arms={str(v):dict(median_us=100+v) for v in vs}
        winner=22 if job['kind']=='dequant' and job['q'] in (8,10,11) else 20 if job['kind']=='dequant' else 2
        arms[str(winner)]['median_us']=5
        result[key]=dict(arms=arms)
    return result


def test_cohort_all_formats_no_fake_tp_shapes_and_unique_shards():
    c=cohort();families=c['families']
    assert len(families)==35 and {f['q'] for f in families}=={8,10,11,12,13,14}
    assert sum(bool(f['models']) for f in families)==33
    assert all('deepseek' not in model for f in families for model in f['models'])
    b=assign(families,['1','2','3','4','5','6','7'])
    assert sum(map(len,b.values()))==35 and all(b.values())
    assert {digest(f) for rows in b.values() for f in rows}=={digest(f) for f in families}
    assert assign(list(reversed(families)),list(b))==b
    inventory=jobs(families)
    assert sum(j['kind']=='dequant' for j in inventory.values())==70
    assert len(inventory)<35*5*2*5  # No DQ x M x IO Cartesian timing.


def test_shortlist_keeps_old_and_production_helpers_even_if_components_lose():
    f=next(f for f in cohort()['families'] if f['q']==12)
    data=measurements([f]);policy=lambda *args:(8,1)
    rows=helper_options(f,2048,0,data,policy)
    old=next(r for r in f['previous_f16_helpers'] if r[0]==2048)
    assert any((r['dequant_variant'],r['vector_io'])==tuple(old[1:]) for r in rows)
    assert any('production-helper-control' in r['roles'] for r in rows)
    assert sum('component-finalist' in r['roles'] for r in rows)==2
    assert len(rows)<=5 and all(r['cost_scope']=='PRUNING_NOT_E2E' for r in rows)


def test_full_plan_preserves_storage_and_never_imports_tc():
    families=cohort()['families'];data=measurements(families)
    p=full_plan(families,data,lambda *args:None,'bound-authority')
    assert len(p['points'])==175 and not p['pending']
    assert validate_full_plan(p) is p
    for point in p['points']:
        assert point['confirmation_captures']==2
        assert {c['arm'] for c in point['cases']}=={'full-fp16','full-bf16-typed'}
        assert all(c['recipe']['kind']=='full' for c in point['cases'])
    bad=copy.deepcopy(p);bad['points'][0]['cases'][0]['helper_options']['vector_io']=9
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_full_plan(bad)
    missing=copy.deepcopy(data);missing.pop(next(iter(missing)))
    assert full_plan(families,missing,lambda *args:None,'bound-authority')['pending']


def test_resume_requires_raw_numerics_samples_and_cold_ring():
    job=dict(kind='dequant',q=10,n=512,k=2048,storage=1)
    d=dict(status='PASS',job=job,authority='a',negative_detected=True,l2_bytes=64,
           copies=2,replays=4,traffic=dict(reads=128,writes=256),
           arms={str(v):dict(samples_us=[10.]*30,median_us=10.,
                 proof=dict(bad=0,guard='PASS',control_raw_bits_equal=True)) for v in variants(10)})
    def seal(d):d['record_sha256']=digest({k:v for k,v in d.items() if k!='record_sha256'});return d
    assert validate_result(seal(d),job,'a') is d
    for mutate in (lambda r:r['arms']['22']['samples_us'].pop(),
                   lambda r:r['arms']['22']['proof'].update(bad=1),
                   lambda r:r.update(copies=1),lambda r:r.update(negative_detected=False),
                   lambda r:r.update(l2_bytes=10000)):
        bad=copy.deepcopy(d);mutate(bad)
        with pytest.raises(ValueError):validate_result(seal(bad),job,'a')


def test_paired_full_capture_cannot_pass_with_one_direction_missing():
    f=cohort()['families'][0];plan=full_plan([f],measurements([f]),lambda *args:None,'a')
    p=plan['points'][0]
    rows={}
    for cid in ['native-fp16']+[c['id'] for c in p['cases']]:
        r=dict(case_id=cid,gpu_sum_us=20.,phases_us=dict(main=15.,input=2.,output=3.))
        rows[cid]=confirmed_capture(None,r)
    assert summarize([p],{p['id']:rows})['status']=='INCOMPLETE'
    for cid in rows:
        rows[cid]=confirmed_capture(rows[cid],dict(case_id=cid,gpu_sum_us=18.,phases_us=dict(main=13.,input=2.,output=3.)))
        assert rows[cid]['gpu_sum_us']==19.
    assert summarize([p],{p['id']:rows})['status']=='SCREEN_COMPLETE'
    review=helper_review([p],{p['id']:rows})['rows']
    assert {r['workload']['compute'] for r in review}=={'f16','bf16'}
    assert len({r['selection_key'] for r in review})==2
    assert all('native_us' not in r for r in review if r['workload']['compute']=='bf16')


def test_plan_only_no_sdk_model_or_cuda(tmp_path):
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'tools/run_dense_helpers.py'),
             '--sdk','/no-sdk','--output',str(tmp_path/'run'),'--plan-only','--gpus','1,2,3,4,5,6,7']
    subprocess.run(command,check=True,capture_output=True)
    p=json.loads((tmp_path/'run/plan.json').read_text())
    assert p['counts']['model_points']==165 and p['counts']['extension_points']==10
    assert not (tmp_path/'run/build').exists()


def test_finalist_plan_cli_never_enters_gemm_resource_sweep(tmp_path):
    f=cohort()['families'][0]
    plan=full_plan([f],measurements([f]),lambda *args:None,'a')
    source=tmp_path/'helpers.json';source.write_text(json.dumps(plan))
    output=tmp_path/'asys'
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/run_prefill_model_asys.py'),
        '--helper-plan',str(source),'--output',str(output),'--sdk','/no-sdk','--plan-only'],
        check=True,capture_output=True)
    actual=json.loads((output/'plan.json').read_text())
    assert actual['helper_plan_sha256']==plan['sha256'] and actual['counts']['tc_modules']==0
    assert not (output/'probe').exists() and not (output/'build').exists()


def test_recipe_cannot_relabel_bf16_as_f16():
    f=cohort()['families'][0]
    p=full_plan([f],measurements([f]),lambda *args:None,'a')
    c=p['points'][0]['cases'][0]
    c['arm']='full-bf16-typed' if c['arm']=='full-fp16' else 'full-fp16'
    p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    with pytest.raises(ValueError,match='precision'):validate_full_plan(p)


@pytest.mark.parametrize('change',[
    lambda p:p['points'][0].update(confirmation_captures=1),
    lambda p:p['points'][0].update(cases=[]),
    lambda p:p.update(pending=[{'missing':'component'}]),
    lambda p:p['points'][0]['cases'][0].update(roles=[]),
])
def test_plan_cannot_drop_confirmation_or_control(change):
    f=cohort()['families'][0]
    p=full_plan([f],measurements([f]),lambda *args:None,'a')
    change(p);p['sha256']=digest({k:v for k,v in p.items() if k!='sha256'})
    with pytest.raises(ValueError):validate_full_plan(p)


def test_compact_package_excludes_weights_and_reports(tmp_path):
    worker=tmp_path/'gpu-1';worker.mkdir()
    for name in ['worker-a.log','screen-summary.json','weight.bin','profile.acurep','trace.asysrep','trace.sqlite']:
        (worker/name).write_text('evidence')
    result=package(tmp_path)
    with tarfile.open(result) as t:
        assert set(t.getnames())=={'gpu-1/worker-a.log','gpu-1/screen-summary.json'}
