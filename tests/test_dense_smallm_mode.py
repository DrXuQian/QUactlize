"""CPU query/contract tests; no fabricated device performance evidence."""
from copy import deepcopy
import ctypes as C
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from dev.native_components.smallm_ab import plan,fixture_point,checked_fixture
from dev.native_components.smallm_retry import vector_retry,OLD_ENTRY,OLD_FIXTURE,OLD_SCHEMA
from dev.strategy.vector import source,checked_call,execution_contract
from dev.strategy.refresh import assign,family
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def current(tmp_path_factory):
    return plan(tmp_path_factory.mktemp('smallm-mode-plan'))


def test_every_dense_fixture_has_no_router_request(current):
    for p in current['points']:
        w=Workload(**p['workload']);f=fixture_point(w)
        assert f['mode']==0 and f['experts']==f['topk']==f['channels']==1
        assert p['execution_contract']==execution_contract(w)
        b=SimpleNamespace(p=f,rows=w.m,ids=None)
        checked_fixture(b,w)
        b.p=dict(f,mode=2)
        with pytest.raises(ValueError,match='mode0'):checked_fixture(b,w)
        b.p=f;b.ids=object()
        with pytest.raises(ValueError,match='no expert IDs'):checked_fixture(b,w)


@pytest.mark.parametrize('field,value',[('mode',2),('qtype',12),('n',256),('rows',7),
    ('input_type',0),('compute_type',1),('ids',256),('a_row_stride',8192),('experts',2)])
def test_actual_call_rejects_semantic_mismatch(field,value):
    w=Workload(8,512,4096,6);expected=execution_contract(w)
    fields={k:v for k,v in expected.items() if k not in ('compute_type','ids_present')}
    d=SimpleNamespace(call=SimpleNamespace(**fields,ids=None),compute_type=0)
    assert checked_call(w,d)==expected
    setattr(d if field=='compute_type' else d.call,field,value)
    with pytest.raises(ValueError,match='execution contract differs'):checked_call(w,d)


def test_indexed_vectors_keep_their_own_contract():
    w=Workload(12,512,2048,3,'indexed','bf16',experts=32,topk=8,channels=8)
    expected=execution_contract(w)
    fields={k:v for k,v in expected.items() if k not in ('compute_type','ids_present')}
    d=SimpleNamespace(call=SimpleNamespace(**fields,ids=256),compute_type=1)
    assert checked_call(w,d)==expected
    with pytest.raises(ValueError,match='dense small-M'):fixture_point(w)
    d.call.ids=None
    with pytest.raises(ValueError,match='ids_present'):checked_call(w,d)


def query_source(item,name='strategy_vector_query'):
    text=source(item)
    start=text.index('extern "C" int strategy_vector_query(')
    end=text.index('extern "C" int strategy_vector_run(',start)
    return text[start:end].replace('strategy_vector_query',name)


def test_generated_queries_for_all_105_vector_points(current,tmp_path):
    rows=[p for p in current['points'] if p['arms'][0]['recipe']['kind']!='tc']
    assert len(rows)==105
    code='#include "quactlize/execution/simt_validation.hpp"\n'
    code+='#include "quactlize/execution/q4_s1_validation.hpp"\n#include <cassert>\n'
    units={};calls=[];q8=0
    for p in rows:
        arm=p['arms'][0];item=arm['spec'];key=digest(item)
        if key not in units:
            units[key]='query_'+str(len(units))
            code+=query_source(item,units[key])
        w=Workload(**p['workload']);f=fixture_point(w)
        split=dict(arm['recipe']['parameters']).get('split',1)
        calls.append(f'''{{qkg_simt_call_v2 d{{2,sizeof(d)}};
            auto& c=d.call;c.version=1;c.size=sizeof(c);c.qtype={w.qtype};
            c.n={w.n};c.k={w.k};c.rows={w.m};c.mode={f['mode']};c.input_type=QKG_F32;
            c.experts=c.topk=c.channels=1;c.a_row_stride=c.a_token_stride=c.k;
            c.out_row_stride=c.n;c.ids_stride=1;
            auto a=c.qtype==8?q8_kpack2::arrangement():c.qtype==12?
                ppu_arrangements::q4_kpack4_transpose_v1():ppu_arrangements::kquant_kpack_transpose_v1(c.qtype);
            qkg_sizes_v1 s{{}};assert({units[key]}(&d,&a,{split},&s)==0);''')
        if 'q8_body' in item:
            calls.append(f'c.mode=2;assert({units[key]}(&d,&a,1,&s)==20);')
            q8+=1
        calls.append('}')
    assert q8==56
    code+='int main(){\n'+ '\n'.join(calls)+'\n}\n'
    path=tmp_path/'query.cpp';path.write_text(code)
    exe=tmp_path/'query'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(path),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_real_ctypes_vector_constructor_uses_dense_request(current,tmp_path,monkeypatch):
    pytest.importorskip('torch')
    from dev.strategy.vector import Vector
    from quactlize.execution.native import arrangement
    from dev.smallm_closure import bench
    from tools import run_kpack_gemv_gate
    p=next(p for p in current['points'] if p['arms'][0]['spec'].get('q8_body')==5)
    w=Workload(**p['workload']);item=p['arms'][0]['spec']
    unit=tmp_path/'query.cpp';lib=tmp_path/'query.so'
    unit.write_text('#include "quactlize/execution/simt_validation.hpp"\n'+query_source(item)+
        f'extern "C" char const* strategy_vector_identity(){{return "{digest(item)}";}}\n'+
        'extern "C" int strategy_vector_run(qkg_simt_call_v2 const*,quactlize_ppu_placed_arrangement_v2 const*,int){return -99;}\n')
    subprocess.run(['g++','-std=c++17','-O2','-fPIC','-shared','-I'+str(ROOT),
                    str(unit),'-o',str(lib)],check=True)
    class HostResources:
        def __init__(self,_):pass
        def close(self):pass
    class Scratch:
        def __init__(self,r,size):self.ptr,self.size=4096,size
    monkeypatch.setattr(run_kpack_gemv_gate,'Resources',HostResources)
    monkeypatch.setattr(bench,'Buffer',Scratch)
    b=SimpleNamespace(p=fixture_point(w),rows=w.m,ids=None,arr=arrangement(w.qtype),
        a=SimpleNamespace(ptr=256),out=SimpleNamespace(ptr=512),r=SimpleNamespace(stream=C.c_void_p()),
        sdk=SimpleNamespace(synchronize=lambda _:None),ring=SimpleNamespace(planes=[{}]))
    record=dict(path=str(lib),sha256=sha(lib),spec=item,key='HOST_QUERY_ONLY')
    v=Vector(b,record,item,workload=w)
    assert v.receipt['execution_contract']==execution_contract(w)
    assert v.calls[0].call.mode==0 and not v.calls[0].call.ids
    v.close()
    b.p=dict(b.p,mode=2)
    with pytest.raises(ValueError,match='mode'):Vector(b,record,item,workload=w)


def write_sealed(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    v={k:v for k,v in value.items() if k!='sha256'};v['sha256']=digest(v)
    path.write_text(json.dumps(v));return v


def previous_run(current,root):
    old=deepcopy(current);old['schema']=OLD_SCHEMA
    for p in old['points']:p.pop('execution_contract')
    old=write_sealed(root/'plan.json',old)
    files={'tools/run_dense_smallm_ab.py':OLD_ENTRY,'dev/strategy/indexed.py':OLD_FIXTURE}
    run=write_sealed(root/'run.json',dict(plan=old['sha256'],cards=['0','1','2','3'],
        assignments=assign(old,4),devices=[{'host':'TEST_ONLY','pci':str(i)} for i in range(4)],
        harness=dict(files=files,sha256=digest(files))))
    by_family={f:i for i,fs in enumerate(run['assignments']) for f in fs}
    tc=0;retained_counts={}
    for p in old['points']:
        i=by_family[family(Workload(**p['workload']))]
        if p['arms'][0]['recipe']['kind']=='tc':
            status='REGRESSION' if tc<6 else 'INCONCLUSIVE' if tc<12 else 'PASS';tc+=1
            retained_counts[status]=retained_counts.get(status,0)+1
        else:status='FAIL' if 'q8_body' in p['arms'][0]['spec'] else 'PASS'
        write_sealed(root/('gpu-'+run['cards'][i])/'points'/(p['id']+'.json'),
            dict(point=p['id'],status=status,authority=dict(run=run['sha256'],shard=i,device=run['devices'][i])))
    return old,run,retained_counts


def test_targeted_recovery_includes_successful_wrong_mode_points_and_keeps_tc_losses(current,tmp_path):
    old,run,counts=previous_run(current,tmp_path)
    checked=[]
    def check(path,p,authority):
        checked.append(p['id']);return json.loads(path.read_bytes())
    before={str(p):sha(p) for p in tmp_path.rglob('*.json')}
    retry=vector_retry(current,tmp_path,check)
    assert retry['sha256']==digest({k:v for k,v in retry.items() if k!='sha256'})
    assert retry['counts']['points']==105 and retry['counts']['full_points']==360
    assert retry['counts']['retained_prior_tc']==255 and len(checked)==255
    assert not retry['tc'] and not any(p['sf_configs'] for p in retry['points'])
    assert retry['retry']['retained_counts']==counts=={'REGRESSION':6,'INCONCLUSIVE':6,'PASS':243}
    assert len(retry['retry']['invalidated'])==105
    assert {r['status'] for r in retry['retry']['invalidated']}=={'PASS','FAIL'}
    assert before=={str(p):sha(p) for p in tmp_path.rglob('*.json')}
    assert {p['id'] for p in retry['points']}=={p['id'] for p in current['points']
        if p['arms'][0]['recipe']['kind'] in ('simt','q4')}


@pytest.mark.parametrize('fault',['recipe','workload','protocol','source','missing-tc','authority'])
def test_retry_rejects_changed_or_incomplete_evidence(current,tmp_path,fault):
    old,run,_=previous_run(current,tmp_path);new=deepcopy(current)
    check=lambda path,p,a:json.loads(path.read_bytes())
    if fault=='recipe':new['points'][0]['selection']['policy']=-1
    elif fault=='workload':new['points'][0]['workload']['m']=9
    elif fault=='protocol':new['protocol']['samples']=3
    elif fault=='source':
        run['harness']['files']['tools/run_dense_smallm_ab.py']='0'*64
        run['harness']['sha256']=digest(run['harness']['files']);write_sealed(tmp_path/'run.json',run)
    elif fault=='missing-tc':check=lambda *_:None
    else:
        file=next((tmp_path/'gpu-0/points').glob('*.json'))
        r=json.loads(file.read_bytes());r['authority']['run']='other';write_sealed(file,r)
    with pytest.raises(ValueError):vector_retry(new,tmp_path,check)


def test_wrapper_separates_recovery_from_resume():
    text=(ROOT/'tools/run_dense_smallm_ab_box.sh').read_text()
    assert 'args+=(--retry-from "$SMALLM_RETRY_FROM")' in text
    assert 'SMALLM_RETRY_FROM for a new recovery' in text
    assert 'git -C "$ROOT" archive "$REV"' in text
    assert 'git -C "$ROOT/third_party/actlize" archive "$ACTLIZE_REV"' in text
    subprocess.run(['bash','-n',str(ROOT/'tools/run_dense_smallm_ab_box.sh')],check=True)
