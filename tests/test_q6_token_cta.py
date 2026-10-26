"""Host ownership/contracts and opt-in native compilation, NOT GPU admission."""
from collections import Counter
from copy import deepcopy
import ctypes as C
from dataclasses import asdict, replace
import os
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy.smallm_asys import workloads, make_plan, check_producers
from dev.strategy.token_cta import GEOMETRIES
from dev.strategy.vector import spec, source, execution_contract
from quactlize.dispatch.table import Workload, Recipe

ROOT=Path(__file__).resolve().parents[1]


def recipe(m, geometry=(4,8,8), split=1):
    c,w,p=geometry
    return Recipe.make('simt','fq',variant=3,columns=c,warps=w,values=p,split=split,cta_tokens=m)


def test_bounded_plan_keeps_current_controls_and_all_m(tmp_path):
    from quactlize.dispatch.planning import plan_smallm
    from dev.strategy.refresh import request, original_recipe
    from dev.strategy.prescreen import semantic_config
    ws=workloads('q6-token-cta')
    assert [(w.qtype,w.n,w.k,w.m) for w in ws]==[(14,1024,5120,m) for m in range(1,9)]
    originals=plan_smallm(tmp_path,[request(w) for w in ws],dense_fallbacks=True)['requests']
    plan=make_plan(originals,'q6-token-cta')
    assert plan['counts']==dict(points=8,arms=246,tc_modules=2,vector_modules=25,families=1)
    assert not plan['production_changed'] and not plan['native_comparison']
    for point,old in zip(plan['points'],originals):
        control,=[a for a in point['arms'] if a['id']=='original']
        assert semantic_config(Recipe.from_dict(control['recipe']))==semantic_config(original_recipe(old))
        m=point['workload']['m']
        candidates=[a for a in point['arms'] if 'cta_tokens' in a['spec']]
        assert len(candidates)==(0 if m==1 else 12)
        if m==1: continue
        assert {(a['spec']['config']['columns'],a['spec']['config']['warps'],
                 a['spec']['config']['values'],dict(a['recipe']['parameters'])['split'])
                for a in candidates}=={(*g,s) for g in GEOMETRIES for s in (1,2,4,8)}
        assert all(a['required'] and a['spec']['cta_tokens']==m for a in candidates)
        ordinary={semantic_config(Recipe.from_dict(a['recipe'])) for a in point['arms']
                  if 'cta_tokens' not in a['spec']}
        for arm in candidates:
            params=dict(arm['recipe']['parameters']);params.pop('cta_tokens')
            assert semantic_config(Recipe.make('simt','fq',**params)) in ordinary


@pytest.mark.parametrize('change',['q','bf16','mode','tokens','m1','m9','geometry','storage'])
def test_explicit_candidate_scope(change):
    w=Workload(14,1024,5120,5);r=recipe(5)
    if change=='q':w=replace(w,qtype=8)
    elif change=='bf16':w=replace(w,compute='bf16')
    elif change=='mode':w=replace(w,operation='indexed',experts=4,topk=2)
    elif change=='tokens':r=recipe(4)
    elif change=='m1':w=replace(w,m=1);r=recipe(1)
    elif change=='m9':w=replace(w,m=9);r=recipe(9)
    elif change=='geometry':r=recipe(5,(4,4,4))
    else:w=replace(w,storage='f16')
    with pytest.raises(ValueError):spec(w,r)


def test_cohort_cli_and_resume_are_frozen_without_gpu(tmp_path):
    command=[sys.executable,str(ROOT/'tools/run_smallm_regressions_asys.py'),
             '--output',str(tmp_path/'plan'),'--cohort','q6-token-cta','--plan-only']
    for extra in ([],['--resume']):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert '"points": 8' in result.stdout and 'ASYS_GPU_ACTIVITY' in result.stdout
    plan=json.loads((tmp_path/'plan/plan.json').read_bytes())
    assert plan['cohort']=='q6-token-cta'
    bad=command[:];bad[bad.index('q6-token-cta')]='regressions'
    result=subprocess.run(bad+['--resume'],capture_output=True,text=True)
    assert result.returncode!=0 and 'resume cohort differs' in result.stderr


@pytest.mark.parametrize('geometry',GEOMETRIES)
def test_actual_generated_query_guards_on_host(tmp_path,geometry):
    pytest.importorskip('torch')
    from quactlize.execution.native import Call,SimtCallV2,Sizes,Arrangement,arrangement
    text=source(spec(Workload(14,1024,5120,6),recipe(6,geometry)))
    # Compile the actual carrier query, not a Python reimplementation.
    text=text.split('extern "C" int strategy_vector_run',1)[0]
    text=text.replace('#include "simt_token_cta.cuh"\n','')
    cpp,so=tmp_path/'query.cpp',tmp_path/'query.so';cpp.write_text(text)
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-shared','-fPIC',
                    '-I'+str(ROOT/'quactlize/execution'),str(cpp),'-o',str(so)],check=True)
    fn=C.CDLL(str(so)).strategy_vector_query
    fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.POINTER(Sizes)]
    fn.restype=C.c_int
    a=arrangement(14)
    def call():
        return SimtCallV2(Call(version=1,size=C.sizeof(Call),qtype=14,n=1024,k=5120,
            experts=1,rows=6,mode=0,input_type=1,channels=1,topk=1,
            a_row_stride=5128,out_row_stride=1032),0)
    for split in (1,2,4,8):
        d=call();out=Sizes()
        assert fn(C.byref(d),C.byref(a),split,C.byref(out))==0
        assert out.workspace_bytes==(0 if split==1 else 6*split*1024*4)
    for field,value in [('rows',5),('rows',7),('qtype',13),('mode',1),('experts',2),
                        ('input_type',0),('k',5136),('n',1025),('a_row_stride',5122)]:
        d=call();setattr(d.call,field,value);out=Sizes()
        assert fn(C.byref(d),C.byref(a),1,C.byref(out))!=0
    d=call();d.compute_type=1
    assert fn(C.byref(d),C.byref(a),1,C.byref(Sizes()))!=0
    for split in (0,3,16):
        d=call();assert fn(C.byref(d),C.byref(a),split,C.byref(Sizes()))!=0
    d=call();a.mapping_id+=1
    assert fn(C.byref(d),C.byref(a),1,C.byref(Sizes()))!=0


@pytest.mark.parametrize('geometry',GEOMETRIES)
@pytest.mark.parametrize('split',[1,2,4,8])
def test_k_ownership_full_warp_tails_and_reducer_layout(geometry,split):
    columns,warps,values=geometry;workers=warps*32//columns
    for k in (512,1024,5120,8192):
        owned=[]
        for partition in range(split):
            for worker in range(workers):
                owned.extend(range(partition*workers+worker,k//16,split*workers))
            # Full-mask shuffles are legal even with K tails/empty partitions.
            for start in range(partition*workers,k//16,split*workers):
                for warp in range(warps):
                    live=[start+warp*32//columns+lane//columns<k//16 for lane in range(32)]
                    assert all(live) or not any(live)
        assert Counter(owned)==Counter(range(k//16))
    for m in range(2,9):
        n=256;stride=n+12
        indices=[t*stride+tile*32+lane if split==1 else (t*split+s)*n+tile*32+lane
                 for tile in range(n//32) for s in range(split) for t in range(m) for lane in range(32)]
        expected=[t*stride+j for t in range(m) for j in range(n)] if split==1 else list(range(m*split*n))
        assert Counter(indices)==Counter(expected)
    assert sorted((lane%columns)*values+lane//columns for lane in range(32))==list(range(32))


@pytest.mark.parametrize('geometry',GEOMETRIES)
def test_lane_addresses_and_activation_owner_do_not_change(geometry):
    c,w,p=geometry;n=1024
    for warp in range(w):
        for base in (0,32,288):
            for half in range(2):
                for residue in range(4):
                    for bits in (4,2):
                        addresses=[]
                        for lane in range(32):
                            g=base+warp*32//c+lane//c;k=g*16+half*4+residue
                            word=((k//(8*(16//bits)))*8+k%8)*n+(lane%c)*p
                            addresses.extend(range(2*word,2*word+2*p))
                        unique=len(set(addresses));sectors={b//32 for b in addresses}
                        assert unique>0 and unique<=len(sectors)*32
                        assert all((a%(2*p))==0 for a in addresses[::2*p])
            for lane in range(32):
                for offset in range(0,16,2):
                    count=16//c;owner=(lane&~(c-1))+offset//count
                    assert owner//c==lane//c
                    assert (owner%c)*count+(offset%count)==offset


def test_reused_reader_math_and_split_contract_remain_explicit():
    body=(ROOT/'quactlize/execution/simt_token_cta.cuh').read_text()
    assert 'blockIdx.y' not in body
    assert 'dim3(c.n/(Columns*P),1,split)' in body
    assert 'common_codes<Q>' in body and body.count('common_codes<Q>')==1
    assert body.index('float2 code=common_codes<Q>')<body.index('dot[t][p].x=fmaf')
    assert 'Activation<QKG_F32,0>' in body and 'float2 total[Tokens][Pairs]' in body
    assert 'dot[t][p].x=fmaf(values[t][residue],code.x,dot[t][p].x)' in body
    assert 'total[t][p].x+=fmaf(metadata[2*p].x,dot[t][p].x,metadata[2*p].y*a_sum[t])' in body
    assert 'launch_reduction<Q>(c,split,stream)' in body
    assert '[(int64_t(t)*split+partition)*call.n+n]' in body
    assert body.count('__syncthreads()')==1


def test_real_entry_and_grid_checked_not_lookalike():
    w=Workload(14,1024,5120,7);r=recipe(7,split=4);s=spec(w,r)
    contract=execution_contract(w)
    point=dict(workload=asdict(w),execution_contract=contract,
               arms=[dict(id='candidate',spec=s,recipe=asdict(r))])
    node=dict(name='void quactlize::execution::simt::token_cta_reader<7,4,8,8>(qkg_call_v1,int)',grid=[32,1,4],block=[256,1,1])
    data=dict(arms=dict(candidate=dict(receipt=dict(execution_contract=contract))),
              arm_components=dict(candidate=['producer']),
              component_records=dict(producer=dict(kind='producer',identity=dict(node=node))))
    check_producers(point,data)
    for mutation in ('grid','symbol'):
        bad=deepcopy(data);node=bad['component_records']['producer']['identity']['node']
        if mutation=='grid':node['grid'][1]=7
        else:node['name']=node['name'].replace('token_cta_reader','common_reader')
        with pytest.raises(ValueError):check_producers(point,bad)


@pytest.mark.parametrize('m',range(2,9))
@pytest.mark.parametrize('geometry',GEOMETRIES)
def test_native_token_cta_compile_only(tmp_path,m,geometry):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:pytest.skip('opt-in actual HGCC compilation; no GPU claims')
    from quactlize.runtime.compiler import FLAGS
    item=spec(Workload(14,1024,5120,m),recipe(m,geometry))
    cpp,obj=tmp_path/'kernel.cu',tmp_path/'kernel.o';cpp.write_text(source(item))
    command=[str(Path(sdk)/'bin/hgcc'),*FLAGS,
        *['-I'+str(ROOT/p) for p in ('.','quactlize/execution','quactlize/include',
            'third_party/actlize/include','third_party/actlize/tools/util/include')],'-c',str(cpp),'-o',str(obj)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert obj.stat().st_size>0
