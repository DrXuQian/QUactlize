"""CPU scope/control/serialization checks, not PPU correctness or timing."""
from copy import deepcopy
import ctypes as C
import json
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace

import pytest

from dev.strategy.m_coverage import points, coverage, make_decode_plan
from dev.strategy.refresh import request, assign
from dev.strategy.resources import module_spec, probe_spec
from dev.prefill_asys.m_span import make_plan, sf_rules, validate, MS
from dev.prefill_asys.finalists import load, bind_cases
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.run_prefill_model_asys import confirmation_jobs

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def originals(tmp_path_factory):
    return plan_smallm(tmp_path_factory.mktemp('dense-m-host'),
        [request(Workload(**p['workload'])) for p in points()],dense_fallbacks=True)['requests']


def test_small_m_complete_denominator_and_real_controls(originals):
    audit=coverage(originals)
    assert (audit['families'],audit['expected'],audit['exact'],audit['missing'])==(45,360,117,243)
    assert len(audit['points'])+len(audit['retained_exact'])==360
    assert {p['workload']['m'] for p in audit['points']}==set(range(1,9))
    plan=make_decode_plan(originals)
    assert plan['counts']['points']==243 and plan['counts']['max_arms']<=48
    assert not plan['native_comparison'] and not plan['production_changed']
    assert len({k for ks in assign(plan,6) for k in ks})==plan['counts']['families']
    legacy=[]
    for p in plan['points']:
        old=next(a for a in p['arms'] if a['id']=='original')
        assert old['required'] and p['original_recipe']==old['recipe']
        assert {a['recipe']['kind'] for a in p['arms']}<={'simt','q4','tc'}
        if old['spec'].get('legacy_adapters'):
            legacy.append(p)
            assert old['spec']['endpoint']=='native'
            assert dict(Recipe.from_dict(old['recipe']).parameters)['legacy_adapters'] is True
    assert len(legacy)==3


@pytest.fixture(scope='module')
def span(tmp_path_factory):
    policy=json.loads(EVIDENCE.read_text())
    helpers=digest(json.loads((ROOT/'policies/dense_full_helpers_20261023.json').read_text()))
    return make_plan(policy,helpers,tmp_path_factory.mktemp('dense-span'))


def test_span_covers_5120_and_heavy_sf_without_promoting_fallback(span):
    assert load(span)==span and len(span['points'])==45*len(MS)
    assert 5120 in span['ms'] and not span['production_changed']
    assert span['counts']['modules']<96
    for p in span['points']:
        assert len(p['cases'])<=48 and p['original_recipe'] is None
        assert any('old-selector-control' in c['roles'] for c in p['cases'])
        sf=[dict(Recipe.from_dict(c['recipe']).parameters) for c in p['cases'] if c['arm']=='sf']
        assert len(sf)>3
        # Heavy/large tiles are retained, not one light fallback per route.
        assert any(c['tm']>=64 for c in sf)
        for c in p['cases']:
            if c['arm']=='fq':assert not any(r.startswith('rule:') for r in c['roles'])


def test_rule_proposals_do_not_depend_on_winner_or_cost_labels():
    policy=json.loads(EVIDENCE.read_text());w=Workload(12,5120,8192,5120)
    a=sf_rules(w,policy['recipes'],72)
    for p in policy['points']:
        p['winner']='deliberately-invalid'
        p['costs']=[]
    assert sf_rules(w,dict(reversed(list(policy['recipes'].items()))),72)==a


def test_span_resource_rejections_preserve_controls_and_denominator(span):
    point=deepcopy(span['points'][0]);w=Workload(**point['workload']);queries={}
    for c in point['cases']:
        r=Recipe.from_dict(c['recipe'])
        if r.kind!='tc':continue
        spec=probe_spec(w,r,int(dict(r.parameters).get('grid_mode',0)>=2))
        queries[digest(spec)]=dict(request=spec,status='READY',occupancy=2,device=dict(compute_units=72))
    # An optional probe can be unavailable only with an explicit resource
    # rejection. Missing/error records still fail, as do unavailable controls.
    optional=next(c for c in point['cases'] if c['recipe']['kind']=='tc' and 'required-control' not in c['roles'])
    r=Recipe.from_dict(optional['recipe']);key=digest(probe_spec(w,r,int(dict(r.parameters)['grid_mode']>=2)))
    # One resource query can cover a required arm with different grid values.
    required={digest(probe_spec(w,Recipe.from_dict(c['recipe']),int(dict(c['recipe']['parameters'])['grid_mode']>=2)))
              for c in point['cases'] if c['recipe']['kind']=='tc' and 'required-control' in c['roles']}
    if key not in required:
        queries[key].update(status='UNAVAILABLE',rc=1,runtime_status=0)
        ready=bind_cases(point,queries)
        assert len(ready)<len(point['cases']) and point['resource_unavailable']
    key=next(iter(required));queries[key].update(status='UNAVAILABLE',rc=1,runtime_status=0)
    with pytest.raises(ValueError,match='required finalist'):bind_cases(point,queries)


def test_span_tampering_and_unsupported_precision_fail(span):
    for change in ('missing-control','m','dtype'):
        bad=deepcopy(span);p=bad['points'][0]
        if change=='missing-control':p['cases']=[c for c in p['cases'] if 'old-selector-control' not in c['roles']]
        elif change=='m':p['workload']['m']=64
        else:p['workload']['compute']='bf16'
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with pytest.raises(ValueError):validate(bad)


def test_span_captures_bound_processes_and_keep_both_directions(span):
    captures=0
    for point in span['points']:
        jobs=confirmation_jobs(point);captures+=len(jobs)
        assert jobs[0]==([dict(id='native-fp16',arm='native-fp16')],0)
        assert jobs[-1]==([dict(id='native-fp16',arm='native-fp16')],1)
        assert all(len(cases)<=16 for cases,_ in jobs)
        forward=[c['id'] for cases,direction in jobs[1:-1] if direction==0 for c in cases]
        reverse=[c['id'] for cases,direction in jobs[1:-1] if direction==1 for c in cases]
        assert forward==[c['id'] for c in point['cases']] and reverse==list(reversed(forward))
    assert captures==span['counts']['captures']


def test_legacy_native_core_times_shipping_casts_in_order():
    from dev.strategy.pilot_native import DenseCore
    core=object.__new__(DenseCore);calls=[]
    core.legacy_adapters=True
    core.b=SimpleNamespace(rows=1,p={'k':5120,'n':6144},a=SimpleNamespace(ptr=1),
        out=SimpleNamespace(ptr=4),r=SimpleNamespace(stream=5))
    core.a=SimpleNamespace(ptr=2);core.out=SimpleNamespace(ptr=3);core.handles=[6]
    core.convert=lambda *args: calls.append(('convert',args)) or 0
    core.run_fn=lambda *args: calls.append(('main',args)) or 0
    assert core.launch()==0 and [c[0] for c in calls]==['convert','main','convert']
    assert calls[0][1]==(1,1,2,1,5120,5120,1,5)
    assert calls[-1][1]==(0,3,4,1,6144,6144,1,5)
    calls.clear();core.run_fn=lambda *args: 1
    assert core.launch()==1 and len(calls)==1


@pytest.mark.parametrize('q,tm,tn',[(12,8,64),(12,8,128),(11,64,128),(11,128,256)])
def test_changed_control_and_new_heavy_sf_compile_only(tmp_path,q,tm,tn):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:pytest.skip('opt-in HGCC object compilation; no GPU performance claim')
    from quactlize.runtime.compiler import Compiler, FLAGS
    parent=dict(qtype=q,route='fq-dense' if q==12 else 'sf-dense',tm=tm,tn=tn,tk=256,
        wm=8 if q==12 else 64,wn=16 if q==12 else 32,stages=2,ap=0,dn=0 if q==12 else 16,
        persistent=-1,symbol='test_m_span_parent')
    source=Compiler.source(object.__new__(Compiler),parent,'0'*64)
    if q==12:source+='\n'+(ROOT/'quactlize/dense/copy.cu').read_text()
    src=tmp_path/'kernel.cu';obj=tmp_path/'kernel.o';src.write_text(source)
    command=[str(Path(sdk)/'bin/hgcc'),*FLAGS,f'-DPPU_PACKED_SCALE={int(q==12)}',
        f'-DPPU_PACKED_FORMAT={0 if q==12 else 3}',*['-I'+str(ROOT/p) for p in (
            'quactlize/include','quactlize/runtime','quactlize/integrations','quactlize/decode',
            'third_party/actlize/include','third_party/actlize/tools/util/include',
            'third_party/actlize/examples/common')],'-c',str(src),'-o',str(obj)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert obj.stat().st_size>0
