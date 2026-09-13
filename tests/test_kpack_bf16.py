import ctypes as C
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.kpack_bf16_fixture import Oracle,as_float,row_domain
from tools.kpack_dequant_fixture import bf16
from tools.run_kpack_bf16_gate import families,cost_record,prefill_candidates,validate_gemm_result,profile_evidence


def test_bf16_factorized_oracle_matches_an_independent_dot():
    rng=np.random.default_rng(847)
    b=bf16(rng.normal(0,.2,(3,16,64)).astype('f4'))
    oracle=Oracle(b);a,coeff=oracle.activations(9)
    ids=np.array([0,0,0,2,2,2,2,2,2])
    out=np.stack([as_float(b[e]).astype('f8') @ as_float(a[i]).astype('f8') for i,e in enumerate(ids)])
    assert oracle.error(bf16(out),coeff,ids)<.005
    assert oracle.error(np.zeros(out.shape,'u2'),coeff,ids)>.005
    with pytest.raises(ValueError):oracle.error(np.full(out.shape,0x7fc0,'u2'),coeff,ids)
    wrong=np.stack([as_float(b[0]).astype('f8') @ as_float(a[i]).astype('f8') for i in range(9)])
    assert oracle.error(bf16(wrong),coeff,ids)>.005


@pytest.mark.parametrize('m',[512,2048,4096])
def test_provider_rows_retain_the_previous_real_router(m):
    rows,ids,h=row_domain(m,256)
    assert len(rows)==256 and len(ids)==m*8 and rows.sum()==m*8
    assert np.all(ids[:-1]<=ids[1:]) and len(h)==64
    assert np.array_equal(np.bincount(ids,minlength=256),rows)
    assert np.count_nonzero(rows)==256


def test_plan_and_component_sum_never_invent_unmeasured_fq_sf_costs():
    assert len(families([12,13]))==22
    assert cost_record(20,5)['sum_estimate_us']==25
    assert prefill_candidates(full_dequant=5,bf16_gemm=20)['fq']['status']=='UNMEASURED'
    board=prefill_candidates(fq_gemm=22,sf_dequant=5,sf_gemm=18,full_dequant=9,bf16_gemm=11)
    assert {k:v['cost_us'] for k,v in board.items()}=={'fq':22,'sf':23,'full-bf16':20}
    with pytest.raises(ValueError):cost_record(float('nan'),5)
    with pytest.raises(ValueError):prefill_candidates(fq_gemm=-1)


def test_bf16_receipt_rejects_changed_expert_cache_timing_or_admission():
    w=next(w for w in families([12]) if w['experts']==1);m=2048
    rows,_,h=row_domain(m,1);l2=64*1024**2;copies=math.ceil(2.25*l2/(2*w['n']*w['k']))
    ring=2*w['n']*w['k']*copies
    good=dict(status='PASS',workload=w,tokens=m,samples_us=[20.]*15,median_us=20.,error=.001,
        scope='GEMM_PROVIDER_ONLY_DEQUANT_NEVER_INSIDE_TIMED_GRAPH',cost=cost_record(20,5),
        guards='PASS',zero_a='PASS',changed_a_graph='PASS',rows=rows.tolist(),routes_sha256=h,
        total_rows=m,expanded_experts=1,active_experts=1,max_rows=m,topk=1,round_medians_us=[20.]*3,
        identity=dict(copies=copies,weight_ring_bytes=ring,device=dict(l2_bytes=l2)),
        active_weight_ring_bytes=ring,calls_per_graph=copies*2,
        cache='ROTATING_BF16_WEIGHT_COMPLETE_RING_TRAVERSALS',first_use_excluded_seconds=1.,
        production_changed=False,selection_admission='PENDING_MATCHED_FQ_SF_GEMM',sf_dequant_us=3.,
        candidates=prefill_candidates(sf_dequant=3.,full_dequant=5.,bf16_gemm=20.))
    validate_gemm_result(good,w,m)
    for key,value in [('error',float('nan')),('active_experts',8),('expanded_experts',256),
                      ('samples_us',[20.]*14),('median_us',19.),('max_rows',1024),('topk',8),
                      ('calls_per_graph',1),('active_weight_ring_bytes',ring//2),('cache','WARM'),
                      ('first_use_excluded_seconds',float('inf')),('production_changed',True),
                      ('selection_admission','SHIP'),('changed_a_graph','SKIP'),
                      ('candidates',prefill_candidates(fq_gemm=1.,full_dequant=5.,bf16_gemm=20.))]:
        bad=copy.deepcopy(good);bad[key]=value
        with pytest.raises(ValueError,match='.'):
            validate_gemm_result(bad,w,m)


def test_bf16_profile_requires_the_exact_workload_and_numeric_receipt():
    w=families([12])[0];dev=dict(pci='0000:08:00.0');provider='CUBLAS_PPU_SDK'
    r=dict(workload=w,tokens=2048,device=dev,provider=provider,error=.001)
    line='KPACK_BF16_PROFILE '+json.dumps(r)+'\n'
    assert profile_evidence('profiler setup\n'+line,w,2048,dev,provider)==r
    for text in ('',line+line,line.replace('2048','4096'),line.replace('.001','NaN'),
                 line.replace('CUBLAS_PPU_SDK','TORCH'),line.replace('08:00.0','09:00.0')):
        with pytest.raises(ValueError):profile_evidence(text,w,2048,dev,provider)


def test_cublas_abi_uses_nt_bf16_fp32_and_propagates_failures(monkeypatch,tmp_path):
    from tools import kpack_bf16_providers as p
    image=tmp_path/'CUDA_SDK/targets/x86_64-linux/lib/libcublas.so'
    image.parent.mkdir(parents=True);image.touch()
    calls=[]
    class Fn:
        def __init__(self,name):self.name=name;self.status=0
        def __call__(self,*args):
            calls.append((self.name,args))
            if self.name=='cublasCreate_v2':C.cast(args[0],C.POINTER(C.c_void_p))[0]=101
            return self.status
    class Lib:
        def __init__(self):self.funcs={}
        def __getattr__(self,name):return self.funcs.setdefault(name,Fn(name))
    lib=Lib();monkeypatch.setattr(p.C,'CDLL',lambda *a,**k:lib)
    obj=p.Cublas(tmp_path,29)
    class Tensor:
        def __init__(self,shape,ptr):self.shape=shape;self.ptr=ptr
        def data_ptr(self):return self.ptr
    obj(Tensor((2048,5120),11),Tensor((1024,5120),12),Tensor((2048,1024),13))
    args=calls[-1][1]
    assert args[1:6]==(1,0,1024,2048,5120)
    assert args[7:13]==(12,14,5120,11,14,5120)
    assert args[14:]==(13,14,1024,68,-1)
    lib.cublasGemmEx.status=7
    with pytest.raises(RuntimeError):obj(Tensor((8,512),11),Tensor((256,512),12),Tensor((8,256),13))
    obj.close()
