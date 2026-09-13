"""Host execution contracts and unchanged-reader transplant; no device admission."""
import copy
import ctypes as C
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
import pytest

from dev.gemv_ppu import moe_s1 as spec, q4_s1_port as port, run_moe_s1 as runner
from dev.gemv_ppu.moe_s1_bench import Config,fixture,a_model
from quactlize.execution.native import Call,Sizes,Arrangement,arrangement

ROOT=spec.ROOT


@pytest.fixture(scope='module')
def host(tmp_path_factory):
    path=tmp_path_factory.mktemp('moe-s1')/'host.so'
    result=subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC',
        '-I'+str(ROOT/'quactlize/execution'),'-I'+str(ROOT/'quactlize/include'),
        ROOT/'tests/q4_moe_s1_host.cpp','-o',path],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lib=C.CDLL(str(path))
    lib.host_query.argtypes=[C.POINTER(Call),C.POINTER(Config),C.POINTER(Arrangement),C.POINTER(Sizes)]
    lib.host_query.restype=C.c_int
    lib.host_buffers.argtypes=[C.POINTER(Call),C.POINTER(Sizes)]; lib.host_buffers.restype=C.c_int
    lib.host_locate.argtypes=[C.POINTER(Call),C.c_int,C.POINTER(C.c_int64)]
    return lib


def call():
    return Call(version=1,size=C.sizeof(Call),qtype=12,n=512,k=2048,experts=256,rows=64,
        mode=2,input_type=1,channels=8,topk=8,a_row_stride=2056,a_token_stride=16448,
        ids_stride=11,out_row_stride=520,a=0x10000000,low=0x20000000,units=0x30000000,
        output=0x40000000,ids=0x50000000)


def query(lib,c,f=None):
    cfg=f or Config(spec.inventory()[0]); a=arrangement(12); s=Sizes()
    return lib.host_query(C.byref(c),C.byref(cfg),C.byref(a),C.byref(s)),s


def test_port_is_mechanical_and_shipping_header_has_no_dev_dependency():
    for path,body in [('q4_s1_helpers.cuh',port.helpers()),('q4_s1_readers.cuh',port.bodies())]:
        assert (ROOT/'quactlize/execution'/path).read_text().rstrip()==body.rstrip()
        assert '#include "dev/' not in body and 'quactlize::dev' not in body
    body=(ROOT/'quactlize/execution/q4_s1_kernel.cuh').read_text()
    assert 'locate(c,row)' in body and 'uint64_t(r.expert)*N*K/2' in body
    assert 'uint64_t(r.expert)*N*K/16' in body
    assert 'hggcMemcpy' not in body and 'hggcDeviceSynchronize' not in body
    assert 'c.workspace' not in body


def test_shape_and_recipe_denominator():
    assert len(spec.SHAPES)==6 and spec.TOKENS==tuple(range(1,9))
    assert len(spec.inventory())==len({r.key for r in spec.inventory()})==16
    assert len(runner.cases())==18
    assert spec.Recipe(0,0,16,1) in spec.inventory()
    for n,k in spec.SHAPES:
        text=spec.source(n,k)
        assert text.count('return launch<')==16
        for r in spec.inventory():
            g=r.geometry(n,k,64)
            assert g['grid']*g['tile_n']==64*n and g['threads']<=1024
            assert g['split']==1


def test_native_fast_code_parser_accepts_two_real_lowerings_not_empty_or_scalar():
    from collections import Counter
    from dev.gemv_ppu.build_moe_s1 import code_lowering
    assert code_lowering(Counter({'v.lop3.b32':1,'v.add.f16x2':1}),'')=='LOP3_HALF2'
    words='0xf000f 0x64006400'
    assert code_lowering(Counter({'v.and.b32':1,'v.or.b32':1,'v.add.f16x2':1}),words)=='AND_OR_HALF2'
    for ops,text in [(Counter(),''),(Counter({'v.lop3.b32':1}),''),
                     (Counter({'v.and.b32':1,'v.or.b32':1,'v.add.f16x2':1}),'')]:
        assert code_lowering(ops,text) is None


@pytest.mark.parametrize('recipe',spec.inventory())
def test_query_is_host_only_sizes_and_no_workspace(host,recipe):
    c=call(); c.a=c.low=c.units=c.ids=c.output=None
    rc,s=query(host,c,Config(recipe))
    assert rc==0 and (s.low_bytes,s.high_bytes,s.units_bytes,s.workspace_bytes)==(256*512*2048//2,0,256*512*2048//16,0)
    c=call(); assert host.host_buffers(C.byref(c),C.byref(s))==0


@pytest.mark.parametrize('field,value',[
    ('qtype',8),('version',2),('size',0),('rows',0),('mode',3),('input_type',2),
    ('channels',3),('topk',0),('ids_stride',7),('a_row_stride',2057),('a_token_stride',16449),
    ('out_row_stride',511),('n',513),('k',2050),('a_row_stride',2**63-1),('rows',2**31-8)])
def test_bad_calls_fail_before_launch(host,field,value):
    c=call(); setattr(c,field,value)
    assert query(host,c)[0]!=0


@pytest.mark.parametrize('field,value',[
    ('a',0x10000001),('units',0x30000002),('low',None),('high',0x1234),('output',0x20000000),
    ('output',0x50000000),('offsets',0x12340),('ids',None),('units',2**64-16)])
def test_bad_buffers_alignment_alias_null_span(host,field,value):
    c=call(); rc,s=query(host,c); assert rc==0
    setattr(c,field,value); assert host.host_buffers(C.byref(c),C.byref(s))!=0


def locate(host,c,row):
    out=(C.c_int64*3)(); host.host_locate(C.byref(c),row,out); return tuple(out)


@pytest.mark.parametrize('tokens',range(1,9))
@pytest.mark.parametrize('channels',[1,2,4,8])
def test_actual_indexed_locator_gate_up_down_and_padding(host,tokens,channels):
    c=call(); c.rows=tokens*8; c.channels=channels; c.a_token_stride=c.a_row_stride*channels
    ids=np.full((tokens,11),-99,dtype='<i4')
    for t in range(tokens): ids[t,:8]=(np.arange(8)*17+t*13+3)%256
    c.ids=ids.ctypes.data
    for row in range(c.rows):
        t,slot=divmod(row,8)
        assert locate(host,c,row)==(ids[t,slot],t*c.a_token_stride+(slot%channels)*c.a_row_stride,row*c.out_row_stride)


def test_actual_grouped_locator_empty_experts_and_large_local_m(host):
    counts=np.array([0,2,0,9,17,0,0,129,1,0],dtype='<i4')
    offsets=np.r_[0,counts.cumsum()].astype('<i4')
    c=call(); c.mode=1; c.experts=len(counts); c.rows=int(counts.sum()); c.topk=c.channels=1
    c.ids=None; c.offsets=offsets.ctypes.data
    assert query(host,c)[0]==0
    for row,e in enumerate(np.repeat(np.arange(len(counts)),counts)):
        assert locate(host,c,row)==(e,row*c.a_row_stride,row*c.out_row_stride)


def test_access_models_count_source_float_width_and_all_recipes():
    for n,k in spec.SHAPES:
        for r in spec.inventory():
            spec.access(r,n,k,8,1,dict(A=32,B=0,metadata=0))
            a=a_model(r,1,32)
            assert a['element_bytes']==4
            assert set(a['granules'])=={'32','64','128'}
            if r.reader==0: assert a['width_bytes']==4
            else: assert a['width_bytes']==16


def result():
    n,k,t,ch,cl=512,2048,1,1,False
    frozen=json.loads((ROOT/'prebuilt/ppu0010/q4-smallm-v1/manifest.json').read_text())['payloads']['libq4_smallm_n512_k2048.so']['sha256']
    proof={p:'PASS' for p in ('fp16_fp32_bits','indexed_grouped_bits','indexed_dense_bits','zero_a','zero_codes',
                           'invalid_ids','output_guard','gpu_ids_replay')}
    proof.update(error=1e-5,immutable_dense=frozen)
    screen=[dict(key=r.key,correctness=copy.deepcopy(proof),samples_us=[2.+i]*5,median_us=2.+i) for i,r in enumerate(spec.inventory())]
    keys=[x['key'] for x in screen[:2]]
    conf=[dict(key=key,round=rr,samples_us=[2.+i]*15,median_us=2.+i) for rr in range(6) for i,key in enumerate(keys)]
    return dict(identity={'x':1},case=runner.case_id(n,k,t,ch,cl),status='PASS',screen=screen,confirmation=conf,
        medians_us={key:2.+i for i,key in enumerate(keys)},selected=keys[0],copies=32,calls_per_graph=64,
        device=dict(l2_bytes=67108864),launches_per_call=1,scope='INDEXED_S1_NO_GATHER_SCATTER_NOT_CHAIN',
        shape=[8,n,k],tokens=t,channels=ch,topk=8,experts=256,router='spread')


def validate(row): return runner.validate(row,{'x':1},512,2048,1,1,False)


def test_result_denominators_and_ten_failure_plants():
    row=result(); assert validate(row)==row
    def missing_recipe(x): x['screen'].pop()
    def duplicate_recipe(x): x['screen'][-1]=x['screen'][0]
    def missing_round(x): x['confirmation'].pop()
    def duplicate_round(x): x['confirmation'][-1]=x['confirmation'][0]
    def nan_sample(x): x['confirmation'][0]['samples_us'][0]=float('nan')
    def wrong_median(x): x['confirmation'][0]['median_us']=99.
    def missing_proof(x): del x['screen'][0]['correctness']['gpu_ids_replay']
    def wrong_winner(x): x['selected']=x['screen'][1]['key']
    def warm_ring(x): x['copies']=2
    def altered_identity(x): x['identity']={'x':2}
    for plant in (missing_recipe,duplicate_recipe,missing_round,duplicate_round,nan_sample,wrong_median,
                  missing_proof,wrong_winner,warm_ring,altered_identity):
        bad=copy.deepcopy(row); plant(bad)
        with pytest.raises((ValueError,KeyError)): validate(bad)


def test_real_fixture_fp16_boundary_and_official_oracle():
    from tools.kpack_execution_fixture import IndexedWeights
    from gguf.quants import dequantize
    from gguf import GGMLQuantizationType
    w=IndexedWeights(12,256,512,16)
    for t in (1,3,8):
        for ch in (1,8):
            d=fixture(w,t,ch)
            assert np.isnan(d['a'][:,:,512:]).all()
            assert np.all(d['ids'][:,8:]==-99)
            assert np.any(d['a'][:,:,:512].reshape(-1)!=d['ah'].astype('f4').reshape(-1))
            for i in (0,t*8-1):
                e=int(d['expert'][i]); rng=np.random.default_rng(np.random.SeedSequence([81923,12,256,512,e]))
                raw=rng.integers(0,256,(512,144),dtype='u1')
                for off in (0,2):
                    raw[:,off:off+2]=rng.uniform(.005,.03,len(raw)).astype('<f2').view('u1').reshape(-1,2)
                official=dequantize(raw.reshape(-1),GGMLQuantizationType.Q4_K).reshape(256,512).astype('f8')
                a=d['ah'][d['arows'][i]].astype('f8')
                np.testing.assert_allclose(a@official.T,d['golden'][i],atol=1e-10,rtol=1e-12)
                np.testing.assert_allclose(np.abs(a)@np.abs(official).T,d['denom'][i],atol=1e-10,rtol=1e-12)
