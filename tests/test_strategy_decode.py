"""Cold-screen admission, component isolation and bounded/resumable planning."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from dev.strategy.components import component_key
from dev.strategy.decode_plan import original_requests,make_decode_plan,execution_gate
from dev.strategy.journal import atomic_json,ArmJournal
from dev.strategy.resource_pilot import smoke_cases,module_record
from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.tuning import digest
from tools.probe_kpack_strategy import write
from test_strategy_resource_pilot import example,passed_result
from test_strategy_prescreen import tc

ROOT=Path(__file__).resolve().parents[1]


def original(probe):
    return [dict(request=original_requests(probe)[0],status=0,kind=2,
                 config=dict(reader=2,variant=7,columns=4,warps=8,values=8))]


def test_decode_preserves_blind_rules_and_separate_original(tmp_path):
    probe=example(tmp_path)
    before=deepcopy(probe['resolved'])
    plan=make_decode_plan(probe,original(probe))
    assert probe['resolved']==before
    p=plan['points'][0]
    assert p['candidates']==before['points'][0]['candidates']
    assert p['arms'][0]['id']=='original' and p['arms'][0]['roles']==['original']
    assert p['arms'][0]['recipe']['kind']=='q4'
    assert all(a['roles']==['rule'] for a in p['arms'][1:])
    assert plan['counts']['vector_modules']==1 and plan['counts']['reused_tc_modules']==1
    assert not plan['production_changed'] and not plan['full_campaign']
    assert plan==json.loads(json.dumps(plan))
    assert plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'})


def test_plan_rejects_missing_original_and_unprobed_tc(tmp_path):
    probe=example(tmp_path)
    for rows in ([],[dict(original(probe)[0],status=1)],[dict(original(probe)[0],request=[0]*9)]):
        with pytest.raises(ValueError):make_decode_plan(probe,rows)
    probe['receipts'].clear()
    with pytest.raises(ValueError,match='resource cache'):make_decode_plan(probe,original(probe))


def test_full_smallm_keeps_unavailable_points_in_requested_denominator(tmp_path):
    probe=example(tmp_path)
    for row in probe['receipts'].values():row['status']='UNAVAILABLE'
    plan=make_decode_plan(probe,original(probe),pilot=False)
    assert plan['points']==[] and plan['vectors']=={} and plan['tc_modules']==[]
    assert plan['counts']['requested_points']==plan['counts']['blocked_points']==1
    assert plan['blocked'][0]['id']==probe['resolved']['points'][0]['id']
    with pytest.raises(ValueError,match='resource cache'):make_decode_plan(probe,original(probe))


def gate_runs(tmp_path):
    probe=example(tmp_path/'probe');plan=smoke_cases(probe)
    plan['plan_sha256']=digest(plan)
    cases=plan['cases'];paths=[]
    for number,chosen in enumerate(([cases[0]],[cases[1]])):
        path=tmp_path/str(number);p=dict(plan,cases=chosen)
        p['plan_sha256']=digest({k:v for k,v in p.items() if k!='plan_sha256'})
        authority=dict(plan=p['plan_sha256'],resource=probe['authority'],harness='old-'+str(number))
        atomic_json(path/'plan.json',p);atomic_json(path/'authority.json',authority)
        for c in chosen:
            result=passed_result(c)
            if c['split_kind']=='split':
                result['components'].append(dict(result['components'][0],role='reducer',
                                                  node=dict(name='reduce_decode<2>(Params)')))
            module=module_record(probe,c)
            result['receipt']=dict(module_sha256=module['sha256'],build_key=module['key'])
            write(path/'cases'/(c['id']+'.json'),dict(authority=authority,case=c),result)
        paths.append(path)
    return probe,paths


def test_prior_smoke_union_does_not_require_repeating_passed_cells(tmp_path):
    probe,paths=gate_runs(tmp_path)
    before={p:p.read_bytes() for d in paths for p in d.rglob('*.json')}
    gate=execution_gate(probe,paths)
    assert len(gate['passed'])==2
    assert gate['scope']=='PRIOR_TC_MECHANISM_GATE_NOT_PERFORMANCE'
    assert all(p.read_bytes()==v for p,v in before.items())
    with pytest.raises(ValueError,match='coverage 1/2'):execution_gate(probe,paths[:1])


@pytest.mark.parametrize('fault',['seal','module','resource'])
def test_prior_smoke_requires_exact_receipts_and_payload(tmp_path,fault):
    probe,paths=gate_runs(tmp_path)
    if fault=='resource':
        p=paths[0]/'authority.json';value=json.loads(p.read_text());value['resource']['sdk']='wrong'
    else:
        p=next((paths[0]/'cases').glob('*.json'));value=json.loads(p.read_text())
        value['data']['receipt']['module_sha256']='wrong'
        if fault=='module':value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
    atomic_json(p,value)
    with pytest.raises(ValueError):execution_gate(probe,paths)


def test_plan_only_box_needs_no_device_and_preserves_shell(tmp_path):
    probe=example(tmp_path/'probe',Workload(12,512,2048,1))
    env=dict(os.environ,PROBE_RUN=str(probe['root']),RUN_ROOT=str(tmp_path),PLAN_ONLY='1',PYTHON=sys.executable)
    result=subprocess.run(['bash','tools/run_kpack_strategy_decode_box.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"points": 1' in result.stdout and '"production_changed": false' in result.stdout
    assert 'Current Docker shell is preserved' in result.stdout
    runs=[p for p in tmp_path.glob('kpack-rule-decode.*') if p.is_dir()]
    assert len(runs)==1 and not (runs[0]/'decode/authority.json').exists()
    result=subprocess.run(['bash','tools/run_kpack_strategy_decode_box.sh'],cwd=ROOT,
        env=dict(env,RESUME_DIR=str(runs[0])),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_helper_dedup_keeps_runtime_split_and_exact_symbol():
    w=Workload(12,1024,2048,1,'indexed','bf16',experts=256,topk=8)
    kw=dict(authority={},source='same-body',cache='same-scratch')
    node=dict(name='indexed_prepare<8>(Params)',symbol='_prepare8',grid=[8,8,1],block=[256,1,1])
    key=lambda r,n=node:component_key(w,r,n,**kw)[0]
    assert key(tc())==key(tc(tn=128))
    assert key(tc(split=2))!=key(tc(split=4))
    assert key(tc())!=key(tc(),dict(node,symbol='_prepare16',name='indexed_prepare<16>(Params)'))
    finish=dict(node,name='indexed_finish<2>(Params)',symbol='_finish2')
    assert component_key(w,tc(split=2),finish,**kw)[1]['kind']=='reducer_finish'


def test_sf_legacy_alias_matches_exact_kernel_launch_not_a_guessed_time():
    old=(ROOT/'quactlize/execution/metadata.cu').read_text()
    new=(ROOT/'quactlize/dequant/kernels.cu').read_text()
    for token in ('prepass::prepass_unit_kernel<T,packed_unit::kCanonicalPlacedZMul<T>>',
                  'prepass::UnitPlaneDesc','prepass::make_unit_prepass_kernel_args',
                  'prepass::prepass_unit_grid_size<T>','<<<grid,256,0,stream>>>'):
        assert token in old and token in new
    assert 'if (c.config == 0)' in new


def test_sf_ring_uses_metadata_and_indexed_ring_uses_active_experts():
    pytest.importorskip('torch')
    from dev.strategy.cold_decode import ring_geometry
    w=Workload(12,512,2048,1)
    sf=ring_geometry(w,dict(low=512*2048//2,high=0,units=512*2048//16),64*1024**2,True)
    assert sf['copies']==2304 and sf['minimum_read_bytes']==65536
    assert sf['active_ring_bytes']==int(2.25*64*1024**2)
    w=Workload(12,1024,2048,1,'indexed','bf16',experts=256,topk=8)
    moe=ring_geometry(w,dict(low=256*1024*2048//2,high=0,units=256*1024*2048//16),64*1024**2,False)
    assert moe['minimum_active']==8 and moe['copies']==16
    assert moe['packed_resident_bytes']>moe['active_ring_bytes']
    with pytest.raises(ValueError):ring_geometry(w,dict(low=16,units=0),64,True)
    with pytest.raises(ValueError):ring_geometry(w,dict(low=16,units=16),0,False)


@pytest.mark.parametrize('late_failure',[False,True])
def test_sf_output_poison_and_late_guard_before_publishing(monkeypatch,tmp_path,late_failure):
    pytest.importorskip('torch')
    import dev.strategy.cold_decode as c
    events=[]
    def guard():
        events.append('guard')
        if late_failure and events.count('guard')==2:raise ValueError('redzone')
    sf=SimpleNamespace(buffers=[SimpleNamespace(guard=guard)],poison=lambda:events.append('poison'),
        launch=lambda i,config:events.append('launch') or 0,check=lambda i:events.append('check') or {})
    b=SimpleNamespace(ring=SimpleNamespace(copies=2,buffers=[]))
    capture=SimpleNamespace(close=lambda:events.append('close'))
    monkeypatch.setattr(c,'RuntimeView',lambda b:None)
    monkeypatch.setattr(c,'Captured',lambda rt,calls:capture)
    row=dict(id='f'*64)
    monkeypatch.setattr(c,'component_capture',lambda *a:([row],[('path','value')]))
    monkeypatch.setattr(c,'publish_components',lambda pending:events.append('published'))
    if late_failure:
        with pytest.raises(ValueError,match='redzone'):c.measure_expansion(b,None,sf,4,{},'source',tmp_path)
        assert 'published' not in events
    else:
        result=c.measure_expansion(b,None,sf,4,{},'source',tmp_path)
        assert result['correctness']=='PASS' and events[-2:]==['close','published']
    assert events[0]=='poison'


def test_cold_capture_never_publishes_before_callers_final_guard(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    import dev.strategy.cold_decode as c
    graph=SimpleNamespace(sample=lambda:1.,close=lambda:None,count=2)
    capture=SimpleNamespace(groups=[[0]],descriptions=[dict(name='simt::register_reuse<12>',symbol='kernel')],
                            isolate=lambda *a,**kw:graph)
    proofs=[]
    w=Workload(12,1024,2048,1)
    r=Recipe.make('simt','fq',variant=3,columns=4,warps=4,values=4,split=1)
    rows,pending=c.component_capture(None,capture,w,r,{},'source',tmp_path,lambda:proofs.append(1))
    assert len(rows)==len(pending)==1 and len(proofs)==2
    assert not (tmp_path/'components').exists()
    c.publish_components(pending)
    assert c.load_component(*[pending[0][0],pending[0][1]['identity']])['status']=='PASS'


def test_decode_journal_retries_only_failed_jobs(tmp_path):
    from tools.run_kpack_strategy_decode import jobs
    probe=example(tmp_path/'probe');point=make_decode_plan(probe,original(probe))['points'][0]
    plan=jobs(point);journal=ArmJournal(tmp_path/'arms',{},'1'*32)
    for i,job in enumerate(plan):
        journal.start(point,job)
        journal.finish(point,job,'PASS' if i==0 else 'FAIL',
            data=dict(correctness='PASS',components=['a'*64]) if i==0 else None,error=None if i==0 else 'failed')
    assert not any(journal.due(point,j) for j in plan)
    retry=ArmJournal(tmp_path/'arms',{},'2'*32)
    assert [j['id'] for j in plan if retry.due(point,j)]==[j['id'] for j in plan[1:]]


def test_large_ring_linear_order_and_cached_demangling(monkeypatch):
    import dev.strategy.isolated as isolated
    nodes=list(range(10000))
    assert isolated.linear_order(nodes,list(zip(nodes,nodes[1:])))==nodes
    calls=[];isolated.readable_symbol.cache_clear()
    monkeypatch.setattr(isolated.subprocess,'check_output',lambda *a,**kw:calls.append(a) or 'name\n')
    assert all(isolated.readable_symbol('_same')=='name' for _ in range(1000))
    assert len(calls)==1


def test_l2_uses_scalar_attribute_when_properties_are_zero(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    import ctypes as C
    import quactlize.dequant.native as native
    from tools.run_kpack_strategy_decode import l2_authority
    def probe(l2,cu,warp):
        for pointer,value in ((l2,67108864),(cu,72),(warp,32)):
            C.cast(pointer,C.POINTER(C.c_int))[0]=value
        return 0
    monkeypatch.setattr(native,'bind',lambda p:(None,None,probe))
    args=SimpleNamespace(output=tmp_path,l2_bytes=0)
    result=l2_authority(args)
    assert result['bytes']==67108864 and result['source']=='SDK_COMPILED_ATTRIBUTE'
    assert 'hggcDeviceGetAttribute(l2,hggcDevAttrL2CacheSize,device)' in (ROOT/'quactlize/dequant/kernels.cu').read_text()
    args.l2_bytes=32*1024**2
    with pytest.raises(ValueError,match='differs'):l2_authority(args)
