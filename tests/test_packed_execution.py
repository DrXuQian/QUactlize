"""Direct packed contract/ownership proofs; not GPU performance evidence."""
import ctypes as C
import numpy as np
import pytest

from tests.test_ppu_compat import host
from quactlize.compat.native import problem, Sizes
from quactlize.packing.native import Arrangement
from ci.support.compat import canonical_planes, raw_blocks, weight, encode


def query(host,p,route=0,arr=None):
    arr=arr or Arrangement(2,5,4,1,0,256,32,0,0x5135314b50000001)
    host.packed_query.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
    s=Sizes();rc=host.packed_query(C.byref(p),C.byref(arr),route,C.byref(s))
    return rc,s


@pytest.mark.parametrize('m',[0,*range(1,10),16,128,5120])
@pytest.mark.parametrize('mode',[0,1])
@pytest.mark.parametrize('route',[0,2,3])
def test_workspace_contract(host,m,mode,route):
    e,t=(512,10) if mode else (1,1)
    rc,s=query(host,problem(7,m,2560,640,experts=e,topk=t,channels=t,layout=1,padding=1),route)
    assert rc==0 and s.route==(route or (2 if m<=8 else 3))
    assert s.column_tile==(32 if s.route==2 else 64)
    assert s.raw_bytes==s.expanded_weight_bytes==s.grouped_input_bytes==s.grouped_output_bytes==0
    expected=0
    if m and mode and s.route==3:
        aligned=lambda x:(x+255)//256*256
        expected=3*aligned(4*(e+1))+aligned(4*m*t)
    assert s.workspace_bytes==s.routing_bytes==expected
    assert s.low_bytes+s.high_bytes+s.units_bytes==e*2560*640//32*24


@pytest.mark.parametrize('fault',['q','raw','k','n','route','descriptor'])
def test_explicit_admission(host,fault):
    p=problem(7,8,2560,640,experts=512,topk=10,layout=1)
    route=0;arr=Arrangement(2,5,4,1,0,256,32,0,0x5135314b50000001)
    if fault=='q':p.qtype=13
    elif fault=='raw':p.layout=0
    elif fault=='k':p.k=608
    elif fault=='n':p.n=2559
    elif fault=='route':route=1
    else:arr.mapping_id^=1
    assert query(host,p,route,arr)[0]!=0


def test_native_mma_fragment_ownership(host):
    for operand in range(3):
        cells=[host.packed_fragment_map(operand,lane,v) for lane in range(32) for v in range(8)]
        assert sorted(cells)==list(range(256))
        # Transposing the logical fragment is NOT an interchangeable map.
        assert sum(i!=((i%16)*16+i//16) for i in cells)==240


@pytest.mark.parametrize('dtype',[0,1])
def test_q5_word_decode_and_tc_staging(host,dtype):
    raw=raw_blocks(7,2,80,640,seed=71)
    p=canonical_planes(raw,7);n,k=80,640
    low=p['low'].view('<u2').reshape(2,k//4,n)
    high=p['high'].view('<u2').reshape(2,k//16,n)
    units=p['units'].view('<u4').reshape(2,k//32,n)
    gold=encode(weight(raw,7,dtype),dtype)
    fn=host.packed_q5_weight
    fn.argtypes=[C.c_uint16,C.c_uint16,C.c_int,C.c_int,C.c_uint32,C.c_int]
    # Simulate coalesced physical-word staging and every native B fragment,
    # including K128 wrap, expert stride and N64 tail (16 live columns).
    recovered=np.zeros_like(gold);visits=np.zeros_like(gold,dtype='u1')
    for e in range(2):
        for first in range(0,n,64):
            width=min(64,n-first)
            for start in range(0,k,128):
                lo=np.zeros((32,64),'<u2');hi=np.zeros((8,64),'<u2');dm=np.zeros((4,64),'<u4')
                lo[:,:width]=low[e,start//4:start//4+32,first:first+width]
                hi[:,:width]=high[e,start//16:start//16+8,first:first+width]
                dm[:,:width]=units[e,start//32:start//32+4,first:first+width]
                for warp in range(4):
                    for atom in range(8):
                        for lane in range(32):
                            for v in range(8):
                                owned=host.packed_fragment_map(1,lane,v)
                                col=warp*16+owned%16;kk=atom*16+owned//16
                                val=fn(lo[kk//32*8+kk%8,col],hi[kk%8,col],kk%32//8,kk//8,dm[kk//32,col],dtype)
                                if first+col<n:
                                    recovered[e,first+col,start+kk]=val
                                    visits[e,first+col,start+kk]+=1
                                else:assert val==0
    assert np.all(visits==1)
    np.testing.assert_array_equal(recovered,gold)


def test_indexed_tile_bound_and_prefix_dispatch():
    rng=np.random.default_rng(531)
    for rows in (1,8,9,80,90,1280,51200):
        for experts in (1,3,512,1024):
            for concentrated in (False,True):
                ids=rng.integers(-2,experts+1,rows)
                if concentrated:ids[:rows//2]=experts-1
                counts=np.bincount(ids[(ids>=0)&(ids<experts)],minlength=experts)
                tiles=np.concatenate(([0],np.cumsum((counts+15)//16)))
                assert tiles[-1]<=((rows+15)//16+min(rows,experts)-1)
                for task in range(int(tiles[-1])):
                    e=np.searchsorted(tiles,task,side='right')-1
                    assert counts[e]>0 and (task-tiles[e])*16<counts[e]


@pytest.mark.parametrize('n,k',[(16,128),(80,640),(2560,640),(256,1024)])
def test_simt_group_column_ownership_and_shuffle_tail(n,k):
    visits=np.zeros((n,k//32),'u1')
    for first in range(0,n,32):
        for tid in range(128):
            worker,lane=tid//8,tid%32
            for g in range(worker,k//32,16):
                for v in range(4):
                    col=first+(tid%8)*4+v
                    if col<n:visits[col,g]+=1
                for offset in range(32):
                    owner=(lane&~7)+offset//4
                    assert owner//8==lane//8
                    assert (owner%8)*4+offset%4==offset
    assert np.all(visits==1)
    for iteration in range((k//32+15)//16):
        for warp in range(4):
            active=[iteration*16+warp*4+lane//8<k//32 for lane in range(32)]
            assert not any(active) or all(active) # full-mask warp exchanges are legal
