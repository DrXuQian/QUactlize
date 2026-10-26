"""CPU contracts/maps and bounded inventory, not GPU accuracy/performance."""
from collections import Counter
from copy import deepcopy
import ctypes as C
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from dev.strategy.smallm_asys import workloads,make_plan,check_producers
from dev.strategy.token_cta import GEOMETRIES,Q8_GEOMETRIES
from dev.strategy.vector import spec,source,execution_contract
from quactlize.dispatch.table import Workload,Recipe

ROOT=Path(__file__).resolve().parents[1]
QTYPES=(10,11,12,13,14,8)


def recipe(q,m,geometry=None,split=1,window=1):
    c,w,p=geometry or (Q8_GEOMETRIES[0] if q==8 else GEOMETRIES[0])
    extra=dict(cta_window=window) if q==8 else {}
    return Recipe.make('simt','fq',variant=5 if q==8 else 3,columns=c,warps=w,
                       values=p,split=split,cta_tokens=m,**extra)


def test_bounded_formats_keep_m1_and_all_existing_winners(tmp_path):
    from quactlize.dispatch.planning import plan_smallm
    from dev.strategy.refresh import request
    ws=workloads('token-cta-formats')
    assert [(w.qtype,w.m) for w in ws]==[(q,m) for q in (10,11,12,13,8) for m in range(1,9)]
    old=plan_smallm(tmp_path,[request(w) for w in ws],dense_fallbacks=True)['requests']
    p=make_plan(old,'token-cta-formats')
    assert p['counts']['points']==40 and p['counts']['arms']<1200
    assert not p['production_changed'] and not p['historical_times_reused']
    candidates=[]
    for point in p['points']:
        q,m=(point['workload'][f] for f in ('qtype','m'))
        assert sum(a['id']=='original' for a in point['arms'])==1
        new=[a for a in point['arms'] if 'cta_tokens' in a['spec']]
        assert len(new)==(0 if m==1 else 16 if q==8 else 12)
        assert all(a['required'] for a in new)
        for a in point['arms']:
            if a['recipe']['kind']=='tc':
                assert a['required'] or 'EXACT_OLD_WINNER_OR_WITHIN_FIVE_PERCENT' in a['reasons']
        assert len(point['sf_configs'])<=1
        candidates.extend(new)
    assert len(candidates)==448
    assert {a['spec']['q'] for a in candidates}=={8,10,11,12,13}
    from dev.strategy.token_cta import retain_tc_controls
    base=dict(excluded=[],workload=asdict(ws[0]),arms=[dict(id=str(i),required=required,
        recipe=asdict(Recipe.make('tc','fq',split=1)),reasons=reasons)
        for i,(required,reasons) in enumerate([(True,[]),(False,['EXACT_OLD_WINNER_OR_WITHIN_FIVE_PERCENT']),
                                               (False,['TC_SINGLE_AXIS_SPLIT_NEIGHBOR'])])])
    filtered=retain_tc_controls(base)
    assert [a['id'] for a in filtered['arms']]==['0','1']
    assert filtered['excluded'][0]['reason']=='UNCHANGED_TC_NEIGHBOR_NOT_TOKEN_REUSE_CONTROL'


@pytest.mark.parametrize('q',QTYPES)
def test_real_generated_query_all_formats(tmp_path,q):
    pytest.importorskip('torch')
    from quactlize.execution.native import Call,SimtCallV2,Sizes,Arrangement,arrangement
    text=source(spec(Workload(q,1024,5120,8),recipe(q,8)))
    text=text.split('extern "C" int strategy_vector_run',1)[0]
    text='\n'.join(line for line in text.splitlines() if 'token_cta.cuh' not in line)
    cpp,so=tmp_path/'query.cpp',tmp_path/'query.so';cpp.write_text(text)
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(ROOT/'quactlize/execution'),
                    str(cpp),'-o',str(so)],check=True)
    fn=C.CDLL(str(so)).strategy_vector_query
    fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.c_int,C.POINTER(Sizes)];fn.restype=C.c_int
    a=arrangement(q)
    d=SimtCallV2(Call(version=1,size=C.sizeof(Call),qtype=q,n=1024,k=5120,experts=1,rows=8,
                     mode=0,input_type=1,channels=1,topk=1,a_row_stride=5128,out_row_stride=1032),0)
    for s in (1,2,4,8):
        out=Sizes();assert fn(C.byref(d),C.byref(a),s,C.byref(out))==0
        assert out.workspace_bytes==(0 if s==1 else 8*s*1024*4)
    for f,v in [('qtype',14 if q!=14 else 13),('rows',7),('rows',9),('mode',2),('input_type',2),
                ('experts',2),('n',1025),('k',5136)]:
        bad=SimtCallV2.from_buffer_copy(d);setattr(bad.call,f,v)
        assert fn(C.byref(bad),C.byref(a),1,C.byref(Sizes()))!=0
    d.compute_type=1;assert fn(C.byref(d),C.byref(a),1,C.byref(Sizes()))!=0


@pytest.mark.parametrize('q',QTYPES)
def test_loaded_words_cover_exact_gguf_codes_including_q5_shuffle(q):
    import numpy as np
    from dev.gemv_simt.general_fixture import raw_weight
    from dev.bf16_compute.fixture import planes
    from dev.gemv_simt.access import plane_word
    from reference.gguf_kpack import SPECS,_raw_code_planes
    n,k=256,512;raw,_=raw_weight(q,n,k,1821+q);packed=planes(raw,q)
    lo=packed['low'].view('<u2').ravel();hi=packed['high'].view('<u2').ravel()
    group=32 if q in (8,12,13) else 16
    bits=8 if q==8 else SPECS[q].low_bits
    high=0 if q==8 else SPECS[q].high_bits
    bias={8:128,11:4,14:32}.get(q,0)
    raw_bytes=raw.tobytes();wrong_q5=0
    # Model the actual half/slot delivery, NOT an unpack-all-weights shortcut.
    for col in (*range(32),*range(n-16,n)):
        for kk in range(k):
            g,local=divmod(kk,group);slot,tail=divmod(local,8)
            half,residue=divmod(tail,4);segment=slot//(16//bits)
            loaded=g*group+segment*8*(16//bits)+half*4+residue
            word=int(lo[plane_word(q,False,col,loaded,n)])
            code=(word>>(bits*((kk%(8*(16//bits)))//8)))&((1<<bits)-1)
            if high:
                loaded_hi=g*group+half*4+residue
                index=plane_word(q,True,col,loaded_hi,n)
                hslot=(((col>>3)&1)<<3)|((kk>>3)&7) if q==13 else (kk%(8*(16//high)))//8
                code|=((int(hi[index])>>(high*hslot))&((1<<high)-1))<<bits
                if q==13:
                    naive=((loaded_hi//128)*8+loaded_hi%8)*n+col
                    wrong_q5+=naive!=index
            if q==8:
                want=int(raw[col,kk//32,2+kk%32])^128
            else:
                f=SPECS[q];base=(col*(k//256)+kk//256)*f.raw_bytes
                a,b=_raw_code_planes(raw_bytes,base,f,kk%256);want=a|(b<<bits)
            assert code==want
            # Packed-mantissa half subtraction is exact for every code.
            value=np.array([0x6400|code],dtype='<u2').view('<f2')[0]-np.float16(1024+bias)
            assert float(value)==want-bias
    assert bool(wrong_q5)==(q==13)
    assert bool(hi.size)==bool(high)


@pytest.mark.parametrize('q',QTYPES)
def test_group_tails_and_request_models(q):
    from dev.gemv_simt.general_access import pattern
    from quactlize.execution.simt_codegen import Config
    group=32 if q in (8,12,13) else 16
    for c,w,p in (Q8_GEOMETRIES if q==8 else GEOMETRIES):
        for s in (1,2,4,8):
            workers=w*32//c
            for k in (256,512,5120,8192):
                groups=[]
                for partition in range(s):
                    for thread_group in range(workers):
                        groups.extend(range(partition*workers+thread_group,k//group,s*workers))
                    for first in range(partition*workers,k//group,s*workers):
                        for warp in range(w):
                            live=[first+(warp*32+lane)//c<k//group for lane in range(32)]
                            assert all(live) or not any(live)
                assert Counter(groups)==Counter(range(k//group))
            f=Config(5 if q==8 else 3,c,w,p,s)
            for unit_base in ((0,2) if q==8 else (0,)):
                model=pattern(q,f,1024,5120,bases=dict(A=0,low=0,high=0,units=unit_base))['candidate']
                assert model['streams']
                for stream in model['streams']:
                    assert set(stream['footprint'])=={'32','64','128'}
                    for sector,record in stream['footprint'].items():
                        assert 0<=record['unique_sector_utilization']<=1
                        assert record['lane_bytes']>=record['unique_bytes']
                    # One unchanged CTA footprint reused M times, never a
                    # claim that duplicate lane bytes equal HBM traffic.
                    for m in range(2,9):
                        old_requests=m*stream['footprint']['32']['lane_bytes']
                        new_requests=stream['footprint']['32']['lane_bytes']
                        if stream['name'].startswith('A-'):new_requests*=m
                        assert new_requests==(old_requests if stream['name'].startswith('A-') else old_requests//m)


def test_q8_fast_decode_and_half_window_order_preserved():
    body=(ROOT/'quactlize/execution/simt_q8_token_cta.cuh').read_text()
    assert 'common_metadata' not in body and 'common_codes' not in body
    assert 'q8_vector::scales<P>' in body and 'q8_vector::code_pair<0>' in body
    assert 'q8_vector::code_pair<1>' in body
    assert 'float2 total[Tokens][Pairs]' in body and 'launch_reduction<8>' in body
    assert body.count('__syncthreads()')==1 and 'blockIdx.y' not in body
    for window in (1,2):
        order=[segment*16+slot*8+(first+half)*4+r
               for first in range(0,2,window) for half in range(window)
               for segment in range(2) for slot in range(2) for r in range(4)]
        old=[segment*16+slot*8+half*4+r for half in range(2)
             for segment in range(2) for slot in range(2) for r in range(4)]
        assert order==old and sorted(order)==list(range(32))
    for extra in ({},{'cta_window':0},{'cta_window':True},{'cta_window':3},{'cta_window':1,'q8_body':3}):
        r=Recipe.make('simt','fq',variant=5,columns=4,warps=8,values=4,split=1,cta_tokens=3,**extra)
        with pytest.raises(ValueError):spec(Workload(8,1024,5120,3),r)


@pytest.mark.parametrize('q',QTYPES)
def test_exact_candidate_entry_and_fused_grid(q):
    w=Workload(q,1024,5120,4);r=recipe(q,4,split=4);item=spec(w,r);contract=execution_contract(w)
    entry='quactlize::execution::simt::'+('q8_token_cta_reader' if q==8 else 'token_cta_reader')
    c=item['config'];node=dict(name='void '+entry+'<4,4,8,4>(qkg_call_v1,int)',
                             grid=[w.n//(c['columns']*c['values']),1,4],block=[c['warps']*32,1,1])
    point=dict(workload=asdict(w),execution_contract=contract,arms=[dict(id='candidate',spec=item,recipe=asdict(r))])
    data=dict(arms=dict(candidate=dict(receipt=dict(execution_contract=contract))),
              arm_components=dict(candidate=['p']),component_records=dict(p=dict(kind='producer',identity=dict(node=node))))
    check_producers(point,data)
    bad=deepcopy(data);bad['component_records']['p']['identity']['node']['grid'][1]=4
    with pytest.raises(ValueError):check_producers(point,bad)
    bad=deepcopy(data);bad['component_records']['p']['identity']['node']['block'][0]=32
    with pytest.raises(ValueError):check_producers(point,bad)


def test_other_reader_symbols_are_not_mistaken_for_candidates():
    from dev.strategy.refresh_gaps import producer_entry
    from dev.strategy.components import role
    for entry,args in [('quactlize::execution::q4_decode::kernel','IiEv'),
                       ('quactlize::execution::simt::q8_vector::kernel','IiEv'),
                       ('quactlize::execution::simt::q8_address_0_5','Ev')]:
        symbol='_ZN'+''.join(str(len(x))+x for x in entry.split('::'))+args
        assert producer_entry(dict(name=symbol,symbol=symbol))==entry
        assert role('void '+entry+'<1>(call)')=='producer'
        with pytest.raises(ValueError):producer_entry(dict(name='void wrong::helper<1>(call)',symbol=symbol))


def test_cli_cohort_and_resume_without_gpu(tmp_path):
    command=[sys.executable,str(ROOT/'tools/run_smallm_regressions_asys.py'),'--output',str(tmp_path/'plan'),
             '--cohort','token-cta-formats','--plan-only']
    for extra in ([],['--resume']):
        result=subprocess.run(command+extra,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert '"points": 40' in result.stdout
    plan=json.loads((tmp_path/'plan/plan.json').read_bytes())
    assert not (tmp_path/'plan/run.json').exists()
    assert 'quactlize/execution/simt_q8_token_cta.cuh' in plan['sources']


@pytest.mark.parametrize('q,m,geometry,window',[
    (q,m,g,window) for q in (10,11,12,13,8) for m in range(2,9)
    for g in (Q8_GEOMETRIES if q==8 else GEOMETRIES) for window in ((1,2) if q==8 else (1,))])
def test_native_compile_new_format_bodies(tmp_path,q,m,geometry,window):
    sdk=os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:pytest.skip('opt-in actual HGCC compilation; device gate remains separate')
    from quactlize.runtime.compiler import FLAGS
    item=spec(Workload(q,1024,5120,m),recipe(q,m,geometry,window=window))
    cpp,obj=tmp_path/'kernel.cu',tmp_path/'kernel.o';cpp.write_text(source(item))
    result=subprocess.run([str(Path(sdk)/'bin/hgcc'),*FLAGS,
        *['-I'+str(ROOT/p) for p in ('.','quactlize/execution','quactlize/include',
             'third_party/actlize/include','third_party/actlize/tools/util/include')],
        '-c',str(cpp),'-o',str(obj)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert obj.stat().st_size>0
