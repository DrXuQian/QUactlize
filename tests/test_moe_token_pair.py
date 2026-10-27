"""Host routing/ABI/footprint gates and opt-in native codegen, not GPU proof."""
from collections import Counter
import ctypes as C
from dataclasses import replace
import os
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from dev.strategy.moe_token_pair import GEOMETRIES
from dev.strategy.vector import spec,source
from quactlize.dispatch.table import Workload,Recipe

ROOT=Path(__file__).resolve().parents[1]


def workload(q=12,m=4,**kw):
    return Workload(q,512,2048,m,operation='indexed',compute='bf16',experts=256,topk=8,**kw)


def recipe(g=GEOMETRIES[0],**kw):
    c,w,p=g
    return Recipe.make('simt','fq',variant=3,columns=c,warps=w,values=p,split=1,moe_pair_rows=2,**kw)


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    folder=tmp_path_factory.mktemp('moe-pair-host')
    cpp,so=folder/'host.cpp',folder/'host.so'
    cpp.write_text('''#include "dev/strategy/moe_token_pair.hpp"
extern "C" int partner(uint64_t mask,unsigned row) {
    auto p=quactlize::execution::simt::moe_pair::locate(mask,row);
    return p.leader ? p.peer : -2;
}
extern "C" int64_t activation(int r,int topk,int channels,int64_t ts,int64_t rs) {
    return quactlize::execution::simt::moe_pair::activation_row(r,topk,channels,ts,rs);
}
''')
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT),
                    '-I'+str(ROOT/'third_party/actlize/include'),str(cpp),'-o',str(so)],check=True)
    lib=C.CDLL(str(so));lib.partner.argtypes=[C.c_uint64,C.c_uint];lib.partner.restype=C.c_int
    lib.activation.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int64,C.c_int64];lib.activation.restype=C.c_int64
    return lib


@pytest.mark.parametrize('m',range(1,9))
@pytest.mark.parametrize('topk',(1,3,8))
def test_dynamic_pair_ownership_and_padded_stride(host,m,topk):
    rng=random.Random(27+m*topk)
    # Include duplicates within a token, odd experts, sparse routing, row63,
    # invalid IDs, and changed IDs on each replay. No router-uniqueness shortcut.
    for ids in ([9]*(m*topk),list(range(m*topk)),
                *[[rng.randrange(-1,17) for _ in range(m*topk)] for _ in range(20)]):
        owned=[];pairs=[]
        for row,expert in enumerate(ids):
            if not 0<=expert<16:
                owned.append(row);continue  # existing NaN writer, no weight access
            matches=sum(1<<r for r,e in enumerate(ids) if e==expert)
            peer=host.partner(matches,row)
            if peer==-2:continue
            owned.append(row)
            if peer>=0:
                assert peer>row and ids[peer]==expert
                owned.append(peer);pairs.append((row,peer))
        assert Counter(owned)==Counter(range(m*topk))
        for channels in (1,topk):
            stride=2056;token_stride=channels*stride+16
            for row in owned:
                token,slot=divmod(row,topk)
                assert host.activation(row,topk,channels,token_stride,stride)==token*token_stride+(slot%channels)*stride
            if channels>1 and pairs:
                # A must NOT be reused with weights: a different slot/token has
                # a distinct input row even when its expert ID happens to match.
                for a,b in pairs:
                    assert host.activation(a,topk,channels,token_stride,stride)!=host.activation(b,topk,channels,token_stride,stride)


@pytest.mark.parametrize('field,value',[
    ('operation','dense'),('compute','f16'),('storage','bf16'),('qtype',8),
    ('m',9),('topk',9),('fusion','swiglu')])
def test_reject_undeclared_workload(field,value):
    w=workload()
    if field=='operation':w=Workload(12,512,2048,4)
    else:w=replace(w,**{field:value})
    with pytest.raises(ValueError):spec(w,recipe())


@pytest.mark.parametrize('extra',[{'cta_tokens':4},{'q8_body':0},{'cta_window':1}])
def test_reject_combining_experiments(extra):
    with pytest.raises(ValueError):spec(workload(),recipe(**extra))


def test_plan_keeps_shipping_and_matched_controls(tmp_path):
    from dev.strategy.moe_token_pair_plan import make_plan,workloads
    from dev.strategy.refresh import request,original_recipe
    from quactlize.dispatch.planning import plan_smallm
    originals=plan_smallm(tmp_path,[request(w) for w in workloads()])['requests']
    plan=make_plan(originals)
    assert plan['counts']==dict(points=8,arms=64,tc_modules=0,vector_modules=13,families=2)
    assert plan['timing_method']=='ASYS_GPU_ACTIVITY' and not plan['production_changed']
    for point,old in zip(plan['points'],originals):
        assert Recipe.from_dict(point['arms'][0]['recipe'])==original_recipe(old)
        assert point['arms'][0]['id']=='original' and not point['sf_configs']
        if old['kind']==2:
            assert point['arms'][0]['expected_producer']=='quactlize::execution::q4_decode::kernel_bf16'
        for arm in point['arms'][1:]:
            p=dict(arm['recipe']['parameters'])
            if p['moe_pair_rows']==2:
                assert point['workload']['m']>1
                control=Recipe.make('simt','fq',**(p|{'moe_pair_rows':1}))
                assert any(Recipe.from_dict(a['recipe'])==control for a in point['arms'])
                assert arm['expected_producer'].endswith('moe_token_pair_reader')
    # Poison historical latency-like fields: no old time may affect this
    # causal plan or be imported into new Asys samples.
    poisoned=[dict(o,median_us=1e-9) for o in originals]
    assert [p['arms'] for p in make_plan(poisoned)['points']]==[p['arms'] for p in plan['points']]
    # A fresh Q4-only run preserves the exact four requested workloads/arms.
    # It does not rerun or relabel the Q5 receipts from the original campaign.
    subset=make_plan(originals[:4],'moe-token-pair-q4')
    assert subset['points']==plan['points'][:4]
    assert subset['counts']==dict(points=4,arms=32,tc_modules=0,vector_modules=7,families=1)
    assert subset['cohort']=='moe-token-pair-q4' and not subset['historical_times_reused']
    with pytest.raises(ValueError,match='control inventory'):make_plan(originals,'moe-token-pair-q4')


@pytest.mark.parametrize('cohort,count',[('moe-token-pair',8),('moe-token-pair-q4',4)])
def test_plan_only_cli(tmp_path,cohort,count):
    command=[sys.executable,str(ROOT/'tools/run_smallm_regressions_asys.py'),
             '--output',str(tmp_path/'plan'),'--cohort',cohort,'--plan-only']
    for extra in ([],['--resume']):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert f'"points": {count}' in result.stdout and 'ASYS_GPU_ACTIVITY' in result.stdout
    assert not (tmp_path/'plan/run.json').exists()


def test_producer_and_routed_grid_gate(tmp_path):
    from copy import deepcopy
    from dataclasses import asdict
    from dev.strategy.smallm_asys import check_producers
    from dev.strategy.vector import execution_contract
    from dev.strategy.refresh_gaps import producer_entry
    w=workload();r=recipe();item=spec(w,r);entry='quactlize::execution::simt::moe_token_pair_reader'
    symbol='_ZN'+''.join(str(len(x))+x for x in entry.split('::'))+'ILi12EEEv11qkg_call_v1i'
    node=dict(name='void '+entry+'<12,4,2,8>(call,int)',symbol=symbol,grid=[16,32,1],block=[64,1,1])
    assert producer_entry(node)==entry
    point=dict(workload=asdict(w),same_expert_pair=True,execution_contract=execution_contract(w),
        arms=[dict(id='pair',recipe=asdict(r),spec=item,expected_producer=entry)])
    data=dict(arms=dict(pair=dict(receipt=dict(execution_contract=execution_contract(w)))),
        arm_components=dict(pair=['p']),component_records=dict(p=dict(kind='producer',identity=dict(node=node))))
    check_producers(point,data)
    bad=deepcopy(data);bad['component_records']['p']['identity']['node']['grid'][1]=16
    with pytest.raises(ValueError,match='routed-row ownership'):check_producers(point,bad)
    bad=deepcopy(data);wrong='quactlize::execution::simt::common_reader'
    bad['component_records']['p']['identity']['node'].update(name=wrong,symbol=wrong)
    with pytest.raises(ValueError,match='wrong production reader'):check_producers(point,bad)


@pytest.mark.parametrize('q',range(10,15))
def test_paired_global_footprint_does_not_change(q):
    from dev.gemv_simt.general_access import pattern
    from quactlize.execution.simt_codegen import Config
    for c,w,p in GEOMETRIES:
        # Both tasks keep the old group/lane/column mapping. Only their expert
        # base is shared; A remains a distinct stream per routed row.
        for split in (1,4):
            f=Config(3,c,w,p,split)
            records=pattern(q,f,512,2048)['candidate']['streams']
            for stream in records:
                for sector in ('32','64','128'):
                    foot=stream['footprint'][sector]
                    old_requests=2*foot['lane_bytes']
                    paired_requests=old_requests if stream['name'].startswith('A-') else foot['lane_bytes']
                    assert paired_requests==(old_requests if stream['name'].startswith('A-') else old_requests//2)
                    assert 0<=foot['unique_sector_utilization']<=1
    # The count is source requests, NOT compulsory HBM bytes or measured MBU.


@pytest.mark.parametrize('q',range(10,15))
def test_actual_abi_query_not_just_planner(tmp_path,q):
    from quactlize.execution.native import Call,SimtCallV2,Sizes,Arrangement,arrangement
    text=source(spec(workload(q),recipe())).split('extern "C" int strategy_vector_run',1)[0]
    text=text.replace('#include "dev/strategy/moe_token_pair.cuh"\n','')
    cpp,so=tmp_path/'query.cpp',tmp_path/'query.so';cpp.write_text(text)
    subprocess.run(['g++','-std=c++17','-shared','-fPIC','-I'+str(ROOT/'quactlize/execution'),
                    str(cpp),'-o',str(so)],check=True)
    fn=C.CDLL(str(so)).strategy_vector_query
    fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.POINTER(Sizes)];fn.restype=C.c_int
    a=arrangement(q)
    def call(m=4):
        return SimtCallV2(Call(version=1,size=C.sizeof(Call),qtype=q,n=512,k=2048,
            mode=2,experts=256,topk=8,channels=8,rows=m*8,input_type=1,
            a_row_stride=2056,a_token_stride=8*2056+16,ids_stride=11,out_row_stride=520),1)
    for m in range(1,9):
        for split in (1,2,4,8):
            d=call(m);out=Sizes()
            assert fn(C.byref(d),C.byref(a),split,C.byref(out))==0
            assert out.workspace_bytes==(0 if split==1 else m*8*split*512*4)
    for f,v in [('qtype',8),('mode',0),('rows',72),('rows',31),('rows',0),
                ('topk',0),('topk',9),('input_type',0),('input_type',2),('ids_stride',7),
                ('a_token_stride',2048),('a_row_stride',2057),('out_row_stride',511),('n',513)]:
        d=call();setattr(d.call,f,v)
        assert fn(C.byref(d),C.byref(a),1,C.byref(Sizes()))!=0
    d=call();d.compute_type=0
    assert fn(C.byref(d),C.byref(a),1,C.byref(Sizes()))!=0
    for split in (0,3,16):
        d=call();assert fn(C.byref(d),C.byref(a),split,C.byref(Sizes()))!=0


@pytest.mark.parametrize('g',GEOMETRIES)
@pytest.mark.parametrize('split',(1,2,4,8))
def test_row_split_write_coverage(host,g,split):
    c,w,p=g;n=512;rows=64;stride=n+8;tile=c*p
    ids=[(i*7)%11 for i in range(rows)]
    stores=[]
    for row,e in enumerate(ids):
        peer=host.partner(sum(1<<r for r,x in enumerate(ids) if x==e),row)
        if peer==-2:continue
        for r in ([row,peer] if peer>=0 else [row]):
            for s in range(split):
                for t in range(n//tile):
                    stores.extend(r*stride+t*tile+j if split==1 else (r*split+s)*n+t*tile+j for j in range(tile))
    expected=[r*stride+j for r in range(rows) for j in range(n)] if split==1 else list(range(rows*split*n))
    assert Counter(stores)==Counter(expected)


def test_no_format_precision_or_production_selector_change():
    body=(ROOT/'dev/strategy/moe_token_pair.cuh').read_text()
    assert 'Activation<QKG_F32,1>' in body and 'activation_pair<1,' in body
    assert body.count('common_codes<Q>')==1
    assert 'float2 total[2][Pairs]' in body and 'fmaf(values[t][residue],code.x,dot[t][p].x)' in body
    assert 'if constexpr(Changes&2)' in body
    assert 'launch_reduction<Q>(c,split,stream)' in body
    assert '[(int64_t(rows[t])*split+partition)*call.n+n]' in body
    assert 'if(Pair && c.rows!=c.topk)' in body and 'else common_reader<QKG_INDEXED,Q,C,W,P,1>' in body
    assert body.count('common_reuse_body<')==1
    assert not any('moe_token_pair' in p.read_text() for p in (ROOT/'quactlize/execution').glob('*') if p.is_file())


@pytest.mark.parametrize('q',range(10,15))
@pytest.mark.parametrize('g',GEOMETRIES)
def test_native_compile(tmp_path,q,g):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:pytest.skip('native compile opt-in; GPU proof is separate')
    from quactlize.runtime.compiler import FLAGS
    cpp,obj=tmp_path/'kernel.cu',tmp_path/'kernel.o'
    cpp.write_text(source(spec(workload(q),recipe(g))))
    result=subprocess.run([str(Path(sdk)/'bin/hgcc'),*FLAGS,
        *['-I'+str(ROOT/p) for p in ('.','quactlize/execution','quactlize/include',
             'third_party/actlize/include','third_party/actlize/tools/util/include')],
        '-c',str(cpp),'-o',str(obj)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert obj.stat().st_size>0
