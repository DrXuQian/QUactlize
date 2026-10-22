"""CPU mock resource queries. No PPU occupancy/performance claims."""
import ctypes as C
from dataclasses import asdict
import subprocess

import pytest

from dev.strategy.resources import (inventory,module_spec,probe_spec,request_abi,NativeProbe,resolve_point)
from dev.strategy.space import resolved_grid
from quactlize.dispatch.table import Workload,Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from test_strategy_prescreen import tc


def fixture():
    w=Workload(12,4096,2048,512)
    r=Recipe.make('tc','sf',**(dict(tc(tm=64).parameters)|dict(wm=16,grid_mode=0,grid_b=0)))
    p=dict(id=w.key,workload=asdict(w),candidates=[asdict(r)],champion_controls=[],grid_search={r.key:{}})
    return w,r,p


def receipt(w,r,algorithm,occupancy,cu=64):
    request=probe_spec(w,r,algorithm)
    return dict(request=request,status='READY',occupancy=occupancy,device=dict(ordinal=0,compute_units=cu))


def test_inventory_deduplicates_modules_and_queries_not_kernel_schedules():
    w,r,p=fixture()
    other=Recipe.make('tc','sf',**(dict(r.parameters)|dict(grid_mode=2,grid_b=32)))
    p['candidates'].append(asdict(other));p['champion_controls']=[asdict(other)]
    data=inventory(dict(points=[p]))
    assert data['counts']==dict(modules=1,queries=2)
    assert {q['request']['algorithm'] for q in data['queries'].values()}=={0,1}
    assert 'rule' in {u['role'] for q in data['queries'].values() for u in q['uses']}
    assert 'control' in {u['role'] for q in data['queries'].values() for u in q['uses']}


def test_resource_materialization_uses_persistent_not_ordinary_occupancy():
    w,r,p=fixture();records={}
    for alg,occ in ((0,12),(1,2)):
        row=receipt(w,r,alg,occ);records[digest(row['request'])]=row
    result=resolve_point(p,records)
    assert result['resource_resolution']['status']=='COMPLETE'
    bindings=result['resource_resolution']['bindings']
    for value in result['candidates']:
        chosen=Recipe.from_dict(value);binding=bindings[chosen.key]
        alg=binding['launch']['algorithm']
        assert binding['launch']==resolved_grid(w,chosen,64,2 if alg else 12)
        if alg:assert binding['launch']['grid']<=128
    assert len(result['candidates'])<=5
    assert result['resource_resolution']['original_dynamic_query']=='STILL_REQUIRED'


def test_missing_and_rejected_control_are_visible_not_rule_success():
    w,r,p=fixture()
    p['champion_controls']=[asdict(r)]
    row=receipt(w,r,0,3)
    result=resolve_point(p,{digest(row['request']):row})
    assert result['resource_resolution']['status']=='INCOMPLETE'
    assert result['resource_resolution']['missing']
    row['status']='UNAVAILABLE'
    records={digest(row['request']):row}
    other=receipt(w,r,1,2);records[digest(other['request'])]=other
    result=resolve_point(p,records)
    assert result['resource_resolution']['status']=='COMPLETE'
    assert not result['champion_controls']
    assert any(v['role']=='control' for v in result['resource_resolution']['rejected'])
    assert all(dict(Recipe.from_dict(v).parameters)['grid_mode']>=2 for v in result['candidates'])


def test_control_is_not_expanded_or_promoted_to_rule():
    w,r,p=fixture()
    control=Recipe.make('tc','sf',**(dict(r.parameters)|dict(grid_mode=3,grid_b=32)))
    p['champion_controls']=[asdict(control)]
    records={}
    for alg in (0,1):
        row=receipt(w,r,alg,2);records[digest(row['request'])]=row
    result=resolve_point(p,records)
    assert result['champion_controls']==[asdict(control)] or Recipe.from_dict(result['champion_controls'][0])==control
    assert len(result['champion_controls'])==1


def test_bounded_cohort_keeps_proposed_schedules_without_cartesian_expansion():
    w,r,p=fixture();p['fixed_schedules']=True
    data=inventory(dict(points=[p]))
    assert data['counts']==dict(modules=1,queries=1)
    row=receipt(w,r,0,3)
    resolved=resolve_point(p,{digest(row['request']):row})
    assert resolved['resource_resolution']['status']=='COMPLETE'
    assert [Recipe.from_dict(v) for v in resolved['candidates']]==[r]
    assert resolved['resource_resolution']['bindings'][r.key]['launch']==resolved_grid(w,r,64,3)


@pytest.mark.parametrize('q,compute,metadata',[(12,'bf16',1),(8,'bf16',0),(14,'f16',0)])
def test_grouped_request_is_full_v4_abi_without_data_pointers(q,compute,metadata):
    from quactlize.decode.native_compute import GroupedMetadataCall
    w=Workload(q,1024,2048,8,'indexed',compute,experts=256,topk=8)
    r=Recipe.make('tc','sf',**(dict(tc().parameters)|dict(qtype=q,persistent=-1)))
    d=request_abi(probe_spec(w,r,1),dict(ordinal=0,compute_units=64))
    assert isinstance(d,GroupedMetadataCall)
    assert (d.version,d.size,d.metadata_type)==(4,C.sizeof(GroupedMetadataCall),metadata)
    assert d.device_call.max_rows==8 and d.device_call.call.m==64
    assert d.device_call.call.compute_units==64
    assert not any(getattr(d.device_call.call,k) for k in ('a','low','high','metadata','output','workspace','rows_host','rows_device'))


def test_native_and_typed_dense_are_different_compile_resources():
    from dataclasses import replace
    w,r,_=fixture();small=replace(w,m=1)
    assert module_spec(w,r)['endpoint']=='native'
    assert module_spec(small,r)['endpoint']=='typed'
    assert module_spec(replace(w,compute='bf16'),r)['endpoint']=='typed'


def test_real_ctypes_query_reads_mock_compiled_abi(tmp_path):
    from pathlib import Path
    w=Workload(12,1024,2048,8,'indexed','bf16',experts=256,topk=8)
    r=Recipe.make('tc','sf',**(dict(tc().parameters)|dict(persistent=-1)))
    spec=module_spec(w,r);p=spec['parent'];source=tmp_path/'mock.cpp';library=tmp_path/'mock.so'
    source.write_text('''
#include "quactlize/runtime/abi.h"
#include <cstdio>
extern "C" {
qk_identity_v1 const* quactlize_kpack_identity_v1() {
  static qk_identity_v1 v{1,sizeof(v),12,3,16,64,64,16,16,2,0,16,
      UINT64_C(0x51344b5034540001),"PARENT","mock-key"};return &v;
}
qk_compute_identity_v4 const* quactlize_kpack_compute_identity_v4() {
  static qk_compute_identity_v4 v{4,sizeof(v),quactlize_kpack_identity_v1(),1,1};return &v;
}
int quactlize_kpack_device_v1(char* name,int size,int32_t* ordinal,int32_t* cu) {
  std::snprintf(name,size,"PPU-MOCK-NO-GPU");*ordinal=0;*cu=64;return 0;
}
int quactlize_kpack_grouped_query_v4(qk_compute_device_call_v4 const* c,qk_recipe_v1 const* r,qk_resources_v1* out) {
  if(c->version!=4 || c->size!=sizeof(*c) || c->metadata_type!=1 || c->device_call.call.compute_units!=64 ||
     c->device_call.call.m!=64 || c->device_call.max_rows!=8 || c->device_call.call.a) return QK_INVALID;
  *out={1,sizeof(*out),4096,8192,r->algorithm?7:2,0,0};return QK_OK;
}
}
'''.replace('PARENT',p['symbol']))
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I'+str(Path(__file__).resolve().parents[1]),
                    str(source),'-o',str(library)],check=True,capture_output=True)
    record=dict(path=str(library),sha256=sha(library),parent=p,key='mock-key')
    probe=NativeProbe(spec,record)
    for alg,occ in ((0,2),(1,7)):
        result=probe.query(probe_spec(w,r,alg))
        assert result['status']=='READY' and result['occupancy']==occ
        assert result['device']['compute_units']==64
    # A different precision must not reuse this resource evidence.
    with pytest.raises(ValueError,match='compute/metadata'):
        NativeProbe(spec|dict(compute='f16'),record)


def test_sealed_query_receipts_reject_wrong_sdk_or_tampered_results(tmp_path):
    import json
    from tools.probe_kpack_strategy import read,write
    path=tmp_path/'query.json';identity=dict(sdk='SDK_A',query='mock')
    write(path,identity,dict(status='READY',occupancy=3))
    assert read(path,identity)['occupancy']==3
    with pytest.raises(ValueError,match='identity/digest'):read(path,dict(identity,sdk='SDK_B'))
    row=json.loads(path.read_text());row['data']['occupancy']=99;path.write_text(json.dumps(row))
    frozen=path.read_bytes()
    with pytest.raises(ValueError,match='identity/digest'):read(path,identity)
    assert path.read_bytes()==frozen


def test_probe_plan_only_never_needs_sdk_or_launches_compiler(tmp_path):
    import json,sys
    from pathlib import Path
    from dev.strategy.provenance import source_identity
    _,_,point=fixture();root=Path(__file__).resolve().parents[1]
    plan=dict(points=[point],source_identity=source_identity(root))
    plan['plan_sha256']=digest(plan)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    result=subprocess.run([sys.executable,str(root/'tools/probe_kpack_strategy.py'),'--plan',str(path),
          '--output',str(tmp_path/'probe'),'--plan-only','--sdk',str(tmp_path/'NO_SDK')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"kernel_launches": 0' in result.stdout
    assert json.loads((tmp_path/'probe/inventory.json').read_text())['counts']==dict(modules=1,queries=2)
    assert not (tmp_path/'probe/jit-cache').exists()
    assert not (tmp_path/'probe/authority.json').exists()
