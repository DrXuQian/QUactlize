import ctypes as C
from pathlib import Path

import numpy as np
import pytest
import torch

from reference import gguf_kpack as ref
from tools.run_kpack_moe_gate import raw_weight, cpu_planes, dot, check, chain_requests, CHAIN_CASES
from tests.test_kpack_native_dispatch import probe, query
from tools.run_kpack_batched_bench import command, sequence, validate_plan, parse_row
from quactlize.dispatch.native import IndexedIO, Router


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
def test_pair_gate_uses_independent_bytes(q):
    a=raw_weight(q,2,256,512,30);b=raw_weight(q,2,256,512,31)
    joined=np.concatenate([a,b],axis=1)
    lo,hi,units=cpu_planes(joined,q)
    assert lo.nbytes+hi.nbytes+units.nbytes==joined.nbytes
    if q!=8:
        expected=ref.prepare_grouped(torch.from_numpy(joined.reshape(-1,joined.shape[-1])),512,512,q,2)
        for got,want in zip((lo,hi,units),(expected.low,expected.high,expected.units)):
            assert got.tobytes()==want.numpy().tobytes()
    ids=np.array([[0,1]],dtype='i4')
    act=np.random.default_rng(4).uniform(-.1,.1,(2,512)).astype('f4')
    gold,denom=dot(joined,q,ids,act)
    assert check(gold.astype('<f2').astype('f4'),gold,denom)<.005
    with pytest.raises(ValueError):check(np.zeros_like(gold),gold,denom)
    with pytest.raises(ValueError):check(np.full_like(gold,np.nan),gold,denom)


def test_bindings_match_c_abi():
    assert C.sizeof(IndexedIO)==88
    assert C.sizeof(Router)==56


def test_every_real_chain_request_has_a_policy_choice(probe):
    for merged,tokens,_ in CHAIN_CASES:
        requests=chain_requests(merged,tokens)
        rows=[tuple(r[key] for key in ('q','route','m','n','k','experts','max_rows')) for r in requests]
        results=query(probe,rows)
        for request,line in zip(requests,results):
            assert line!='MISS',request
            fields=line.split()
            assert int(fields[1])==request['q'] and int(fields[2])==2
            if merged and request['projection']=='gate':
                assert int(fields[-1])==3  # A prediction, not a measured doubled-N winner.
                assert request['n']==1024


def test_model_benchmark_axes_and_excluded_first_pass():
    plan=dict(prompts=[512,1024],generations=[512],parallel=1,batch=5120,ubatch=5120,
        models=[dict(name='test',path='/model.gguf',devices='0',split='none')])
    validate_plan(plan)
    assert sequence(plan,1)==[(512,512,0),(512,512,1),(1024,512,0),(1024,512,1)]
    cmd=command(Path('/bench'),plan['models'][0],plan,1,['blk.0.attn.weight'],'kpack',Path('/cache'))
    assert cmd[cmd.index('-npp')+1]=='512,512,1024,1024'
    assert cmd[cmd.index('-npl')+1]=='1'
    assert cmd[cmd.index('-ot')+1].endswith('=CUDA0_KPACK')
    import json
    row=dict(n_kv_max=1536,pp=512,tg=512,pl=1,n_batch=5120,n_ubatch=5120,flash_attn=1,
        is_pp_shared=0,n_kv=1024,t_pp=1,t_tg=2,speed_pp=512,speed_tg=256)
    assert parse_row(json.dumps(row),(512,512,0),plan)['phase']=='warmup'
    assert parse_row(json.dumps(row),(512,512,1),plan)['phase']=='measured'
    row['pp']=128
    with pytest.raises(ValueError):parse_row(json.dumps(row),(512,512,1),plan)
