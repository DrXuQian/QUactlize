import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from quactlize.dequant.native import Call, traffic
from tools import kpack_dequant_fixture as fixture

ROOT=Path(__file__).resolve().parents[1]
SDK=Path('/root/ppu-sdk/2.1.1')


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    if not (SDK/'bin/hgcc').is_file():pytest.skip('PPU SDK host compiler required')
    out=tmp_path_factory.mktemp('dequant-host')/'host.so'
    command=['g++','-std=c++17','-O2','-shared','-fPIC',
        '-ffp-contract=off',f'-I{ROOT}',f'-I{ROOT}/quactlize/include',
        f'-I{ROOT}/third_party/actlize/include',f'-I{SDK}/include',str(ROOT/'tests/dequant_host.cpp'),'-o',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lib=C.CDLL(str(out));lib.dequant_host.argtypes=[C.c_int]+[C.c_void_p]*4+[C.c_int]*2
    lib.dequant_tiled_host.argtypes=lib.dequant_host.argtypes
    assert lib.dequant_call_size()==C.sizeof(Call)
    return lib


@pytest.mark.parametrize('q',range(10,15))
def test_actual_host_decoder_matches_original_gguf_bf16(host,q):
    planes,gold,_=fixture.expert(q,256,512,3)
    out=np.empty_like(gold)
    ptr=lambda name:planes[name].ctypes.data if planes[name].size else None
    assert host.dequant_host(q,ptr('low'),ptr('high'),ptr('units'),out.ctypes.data,256,512)==0
    assert fixture.compare(out,gold)['bad']==0
    assert host.dequant_tiled_host(q,ptr('low'),ptr('high'),ptr('units'),out.ctypes.data,256,512)==0
    assert fixture.compare(out,gold)['bad']==0
    bad=planes['low'].copy();bad[:]=0
    assert host.dequant_host(q,bad.ctypes.data,ptr('high'),ptr('units'),out.ctypes.data,256,512)==0
    with pytest.raises(ValueError):fixture.compare(out,gold)


@pytest.mark.parametrize('q',range(10,15))
def test_byte_counts_are_stage_specific(q):
    p,gold,_=fixture.expert(q,256,512,0)
    sf=traffic(q,256,512,1,0);full=traffic(q,256,512,1,1)
    assert sf['reads']==p['units'].nbytes and sf['writes']==2*p['scale'].nbytes
    assert full['reads']==sum(p[x].nbytes for x in ('low','high','units'))
    assert full['writes']==gold.nbytes
    assert sf['useful_bytes']!=full['useful_bytes']


def test_transpose_tile_has_exact_unique_ownership_and_banks():
    for threads in (128,256):
        written=[];read=[]
        for t in range(threads):
            lane,warp=t%32,t//32
            for r in range(warp,8,threads//32):
                written.extend((lane,r+8*s) for s in range(4))
            for i in range(t,512,threads):
                row,kk=i//16,i%16*2
                read.extend([(row,kk),(row,kk+1)])
        assert len(written)==len(set(written))==1024
        assert set(written)==set(read)
        for residue in range(8):
            assert len({(lane*33+residue)%32 for lane in range(32)})==32
        # Each paired output instruction touches two 64-byte contiguous rows.
        addresses=[(i//16*512+i%16*2)*2 for i in range(32)]
        assert len({a//32 for a in addresses})==4


def test_nonfinite_or_incomplete_output_cannot_pass():
    with pytest.raises(ValueError):fixture.bf16(np.array([np.nan],dtype='f4'))
    with pytest.raises(ValueError):fixture.compare(np.zeros((1,),dtype='u2'),np.ones((2,),dtype='u2'))
    with pytest.raises(ValueError):fixture.compare(np.full(64,0x7fff,dtype='u2'),np.full(64,0x3f80,dtype='u2'))


def test_bounded_plan_and_load_footprints():
    from tools.run_kpack_dequant_gate import work,pattern
    plan=work([12,13])
    assert len(plan)==len({w['id'] for w in plan})==78
    assert sum(3 if w['operation'] else 4 for w in plan if not w['smoke'])==238
    assert {w['q'] for w in plan if w['smoke']}==set(range(10,15))
    w=dict(q=12,n=512,k=2048,operation=1)
    direct=pattern(w,0)['fields'];trans=pattern(w,1)['fields']
    assert direct['low_word']['sectors']['32']==8
    assert direct['low_word']['unique_bytes']==16
    assert trans['low_word']['sectors']['32']==2
    assert trans['low_word']['unique_bytes']==64
    assert trans['bf16_output_pairs']['sectors']['32']==4
    for q in range(10,15):
        for op in (0,1):
            for c in range(3 if op else 4):assert pattern(w|dict(q=q,operation=op),c)['fields']


def test_sf_warp_ownership_is_complete_unique():
    for groups in (8,16):
        for cols in (8,16,32):
            observed=[]
            for lane in range(32):
                if cols==8: col,g0,step=lane//4,lane%4,4
                else: col,g0,step=lane%cols,lane//cols,32//cols
                observed.extend((col,g) for g in range(g0,groups,step))
            assert len(observed)==len(set(observed))==cols*groups


@pytest.mark.parametrize('fault',['scope','device','missing','nan','time','bytes','ring','proof'])
def test_invalid_result_cannot_be_resumed(fault):
    import copy
    from tools.run_kpack_dequant_gate import validate_result
    w=dict(id='one',q=12,n=512,k=2048,experts=1,operation=1,smoke=False)
    b=traffic(12,512,2048,1,1);dev=dict(l2_bytes=1024);peak=2700.
    rows=[dict(config=c,proof=dict(bad=0,guard='PASS',negative_bad=99,signed_zero_differences=0),
               samples_us=[10.]*15,median_us=10.,round_medians_us=[10.]*3,
               effective_gbps=b['useful_bytes']/10/1000,effective_pct=b['useful_bytes']/10/1000/peak*100) for c in range(3)]
    record=dict(status='PASS',workload=w,device=dev,gemm_calls=0,
        scope='DEQUANT_ONLY_NOT_COMBINED_OR_REUSABLE_CACHE',peak_gbps=peak,bytes=b,rows=rows,
        copies=2,calls_per_graph=4,input_ring_bytes=b['reads']*2,output_ring_bytes=b['writes']*2)
    assert validate_result(record,w,dev,peak)==record
    r=copy.deepcopy(record)
    if fault=='scope':r['gemm_calls']=1
    if fault=='device':r['device']['l2_bytes']=2048
    if fault=='missing':r['rows'].pop()
    if fault=='nan':r['rows'][0]['samples_us'][0]=float('nan')
    if fault=='time':r['rows'][0]['median_us']=8.
    if fault=='bytes':r['bytes']['writes']=0
    if fault=='ring':r['calls_per_graph']=3
    if fault=='proof':r['rows'][0]['proof']['bad']=1
    with pytest.raises(ValueError):validate_result(r,w,dev,peak)
