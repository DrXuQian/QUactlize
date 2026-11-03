from collections import Counter
from pathlib import Path
import subprocess

import numpy as np
import pytest

from dev.moe_prefill_helpers.bench import workload_plan, BASELINE
from dev.moe_route.protocol import ids_for, oracle, tiled_model
from dev.moe_scaffold.models import assignments
from dev.moe_scaffold.protocol import Case, weighted

ROOT=Path(__file__).resolve().parents[1]


def test_real_host_predicates(tmp_path):
    target=tmp_path/'contract'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                    '-fsanitize=undefined','-I'+str(ROOT),str(ROOT/'tests/moe_prefill_helpers_host.cpp'),
                    '-o',str(target)],check=True)
    assert 'PREFILL_HELPER_HOST PASS shapes=77' in subprocess.check_output([target],text=True)


def test_bounded_inventory_and_workers():
    cases,coverage=workload_plan()
    assert len(cases)==len({c.key for c in cases})==20
    assert Counter(c.phase for c in cases)=={'prepare':8,'transition':4,'finish':8}
    assert {c.tokens for c in cases}=={512,2048,5120}
    assert {c.hidden for c in cases if c.phase=='transition'}=={512}
    assert {c.columns for c in cases if c.phase=='prepare'}=={2048,3072}
    assert {c.output for c in cases if c.phase=='finish'}=={2048,3072}
    assert coverage['baseline']==BASELINE and not coverage['production_changed']
    assert len(workload_plan(True)[0])==3
    for phase,count in (('prepare',8),('transition',4),('finish',8)):
        assert len(workload_plan(phase=phase)[0])==count
    parts=assignments(cases,4)
    assert sum(map(len,parts))==20
    assert Counter(c['phase'] for p in parts for c in p)==Counter(c.phase for c in cases)


@pytest.mark.parametrize('profile',['random','cluster'])
@pytest.mark.parametrize('tokens,padding',[(1,0),(65,4),(8193,0)])
def test_broadcast_destination_matches_independent_stable_sort(profile,tokens,padding):
    c=Case(tokens,profile=profile,padding=padding).route_case
    ids=ids_for(c,19);reference=oracle(ids,c)
    for got,want in zip(tiled_model(ids,c,512),reference):np.testing.assert_array_equal(got,want)
    # Reconstruct prefix + local ranks, then address them by (token, slot)
    # exactly as the broadcast candidate. No map produced by another CTA is read.
    hist=np.zeros(((tokens*8+511)//512,c.experts),np.int64)
    ranks=np.empty(tokens*8,np.int64)
    for token in range(tokens):
        for slot in range(8):
            expert=ids[token,slot];tile=token//64;r=token*8+slot
            assert tile==r//512
            ranks[r]=hist[tile,expert];hist[tile,expert]+=1
    prefix=np.cumsum(hist,axis=0)-hist;offsets=reference[2]
    destinations=np.array([offsets[ids[t,s]]+prefix[t//64,ids[t,s]]+ranks[t*8+s]
                           for t in range(tokens) for s in range(8)],np.int64)
    inverse=np.empty(tokens*8,np.int64);inverse[reference[1]]=np.arange(tokens*8)
    np.testing.assert_array_equal(destinations,inverse)
    assert np.unique(destinations).size==tokens*8


@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('rows,cols',[(1,4),(17,516),(2048,3072),(65536,4)])
def test_vec4_launch_exact_ownership(threads,rows,cols):
    gy=min(rows,65535)
    row_ids=[r for b in range(gy) for r in range(b,rows,gy)]
    col_ids=[b*threads*4+t*4+j for b in range((cols+threads*4-1)//(threads*4))
             for t in range(threads) if b*threads*4+t*4<cols for j in range(4)]
    assert sorted(row_ids)==list(range(rows)) and col_ids==list(range(cols))


def test_ordered_finish_and_snapshot_contract():
    values=np.array([1e20,1,-1e20,3,0,0,0,0],np.float32).reshape(1,8,1)
    assert weighted(values,np.ones((1,8),np.float32))[0,0]==3
    assert weighted(values[:,[0,2,1,3,4,5,6,7]],np.ones((1,8),np.float32))[0,0]==4
    src=(ROOT/'dev/moe_prefill_helpers/kernels.cuh').read_text()
    assert '__fmul_rn' in src and '__fadd_rn' in src and 'sum=slot?' in src
    assert 'blockIdx.x==0 && threadIdx.x==0' in src
    assert '__floats2bfloat162_rn' in src and '__bfloat1622float2' in src
    assert 'return unpack(pack(unpack(x)))' in src
    assert 'expf(-g)' in src


def test_existing_runner_and_no_production_route_edit():
    runner=(ROOT/'tools/run_moe_scaffold.py').read_text()
    assert "scaffold_source='dev/moe_prefill_helpers/native.cu'" in runner
    assert "extra_sources=('dev/moe_prefill_helpers','dev/moe_decode_helpers')" in runner
    native=(ROOT/'dev/moe_prefill_helpers/native.cu').read_text()
    assert 'if(arm==0 || !prefill_helpers::prepare_fast(*c)) return run_prepare(c,stage)' in native
    assert '!prefill_helpers::finish_fast(*c)' in native
    assert '!prefill_helpers::transition_fast(*c)' in native
    assert 'prefill_helpers' not in (ROOT/'quactlize/moe/scaffold.cu').read_text()
