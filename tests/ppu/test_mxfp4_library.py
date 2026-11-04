"""Native GGUF MXFP4 pack + Sail W4FA16 public ABI. Real PPU required."""
import ctypes as C

import numpy as np
import pytest

from reference import mxfp4 as ref
from ci.support.moe import Params, Info, Public as MoE
from ci.support.fixture import matmul_reference, compare
from ci.support.graph import Replay
from .support import Arena, Public, Arrangement, Sizes


def packed(arena, public, raw, *, pair=None):
    e,n,kb,_=raw.shape
    arr=Arrangement();sizes=Sizes()
    assert public.canonical(39,C.byref(arr))==0
    assert arr.mapping_id==ref.MAPPING_ID
    assert public.sizes(n*(1+(pair is not None)),kb*32,e,39,C.byref(arr),C.byref(sizes))==0
    lo=arena.buffer(sizes.low_bytes);sf=arena.buffer(sizes.units_bytes)
    source=arena.upload(raw);stream=arena.stream()
    if pair is None:
        arena.sync()
        rc=public.pack(source.ptr,lo.ptr,None,sf.ptr,n,kb*32,e,39,C.byref(arr),stream)
    else:
        up=arena.upload(pair)
        arena.sync()
        rc=public.pack_pair(source.ptr,up.ptr,lo.ptr,None,sf.ptr,n,kb*32,e,39,C.byref(arr),stream)
        raw=np.concatenate((raw,pair),axis=1)
    assert rc==0
    arena.sync()
    a,b=ref.pack(raw)
    assert lo.bytes()==a.tobytes() and sf.bytes()==b.tobytes()
    np.testing.assert_array_equal(ref.recover(np.frombuffer(lo.bytes(),'<u4').reshape(a.shape),
        np.frombuffer(sf.bytes(),'u1').reshape(b.shape)),raw)
    return arr,sizes,lo,sf


@pytest.mark.parametrize('partition',['whole','n','k','ep','gate-up'])
def test_mxfp4_pack_bit_exact(package,sdk,partition):
    raw=np.random.default_rng(39).integers(0,256,(4,256,16,17),dtype='u1')
    raw[...,0]=np.arange(256,dtype='u1')[None,:,None]
    raw[...,1:]=np.arange(16,dtype='u1')*17
    if partition=='n':raw=raw[:,:128].copy()
    elif partition=='k':raw=raw[:,:,4:12].copy()
    elif partition=='ep':raw=raw[1:3].copy()
    with Arena(sdk) as arena:
        packed(arena,Public(package),raw,pair=np.flip(raw,axis=1).copy() if partition=='gate-up' else None)


@pytest.mark.parametrize('m',[1,2,3,4,5,6,7,8,9,128,2048,5120])
@pytest.mark.parametrize('channels',[1,6])
def test_mxfp4_moe_dynamic_projection(package,sdk,m,channels,record_property):
    # 256 experts / top-6 matches DeepSeek routing. Small local N/K also
    # exercises the TP4 down-shard domain without allocating a whole model.
    e,n,k,topk=256,256,256,6
    rng=np.random.default_rng(39)
    raw=rng.integers(0,256,(e,n,k//32,17),dtype='u1')
    raw[...,0]=rng.integers(118,126,raw.shape[:-1],dtype='u1')
    weights=ref.dequantize(raw)
    public=Public(package);api=MoE(public.lib);ctx=C.c_void_p()
    api.check(api.create(0,C.byref(ctx)))
    try:
        with Arena(sdk) as arena:
            arr,sizes,lo,sf=packed(arena,public,raw)
            p=Params(1,C.sizeof(Params),39,n,k,e,m,topk,channels,0,k,k*channels,topk,n,arr,sizes)
            assert api.supported(C.byref(p))==1
            plan=C.c_void_p();info=Info(1,C.sizeof(Info))
            api.check(api.prepare(ctx,C.byref(p),lo.ptr,None,sf.ptr,C.byref(plan),C.byref(info)))
            assert info.route==3 and info.split==1 and info.compute_type==1
            record_property('selected_config',info.config.decode())
            lanes=[(arena.stream(),arena.buffer(m*channels*k*4),arena.buffer(m*topk*4),
                    arena.buffer(m*topk*n*4),arena.buffer(info.workspace_bytes)) for _ in range(2)]
            graphs=[]
            try:
                for repeat in range(3):
                    wants=[]
                    for lane,(_,a,ids,out,_) in enumerate(lanes):
                        # One-hot inputs give an exact BF16 projection oracle;
                        # change all addresses and IDs between concurrent lanes.
                        js=(np.arange(m)[:,None]*31+np.arange(channels)[None,:]*17+repeat*13)%k
                        amplitude=262144. if repeat==2 else (-1. if lane else 1.)
                        x=np.zeros((m,channels,k),'f4')
                        x[np.arange(m)[:,None],np.arange(channels)[None,:],js]=amplitude
                        routes=((np.arange(m)[:,None]*(0 if repeat==0 else 31)+
                            np.arange(topk)[None,:]+lane+repeat)%e).astype('i4')
                        cols=js[:,np.arange(topk)%channels]
                        wants.append(amplitude*weights[routes[:,:,None],np.arange(n),cols[:,:,None]])
                        a.write(x);ids.write(routes);out.poison()
                    arena.sync()
                    for lane,(stream,a,ids,out,scratch) in enumerate(lanes):
                        def run(a=a,ids=ids,out=out,scratch=scratch,stream=stream):
                            return api.run(plan,a.ptr,ids.ptr,out.ptr,scratch.ptr,scratch.size,stream)
                        if repeat==1:graphs.append(Replay(sdk,stream,run,1))
                        api.check(graphs[lane]() if graphs else run())
                    arena.sync()
                    for lane,(_,a,ids,out,scratch) in enumerate(lanes):
                        np.testing.assert_array_equal(np.frombuffer(out.bytes(),'f4').reshape(m,topk,n),wants[lane])
                        for b in (a,ids,scratch):b.guard()
            finally:
                arena.sync()
                for graph in graphs:graph.close()
    finally:api.destroy(ctx)


@pytest.mark.parametrize('n,k',[(512,4096),(1024,2048),(4096,512)])
def test_mxfp4_moe_dot_tp_shapes(package,sdk,n,k):
    # Gate/up N-shard, smaller K/N pair, down K-shard. Dense dot oracle
    # checks accumulation, not only one-hot packing/permutation witnesses.
    e,m,topk=8,9,6
    rng=np.random.default_rng(n+k)
    raw=rng.integers(0,256,(e,n,k//32,17),dtype='u1')
    raw[...,0]=rng.integers(118,124,raw.shape[:-1],dtype='u1')
    weights=ref.dequantize(raw)
    x=rng.uniform(-.1,.1,(m,k)).astype('f4')
    ids=((np.arange(m)[:,None]+np.arange(topk)[None,:])%e).astype('i4')
    public=Public(package);api=MoE(public.lib);ctx=C.c_void_p()
    api.check(api.create(0,C.byref(ctx)))
    try:
        with Arena(sdk) as arena:
            arr,sizes,lo,sf=packed(arena,public,raw)
            p=Params(1,C.sizeof(Params),39,n,k,e,m,topk,1,0,k,k,topk,n,arr,sizes)
            plan=C.c_void_p();info=Info(1,C.sizeof(Info))
            api.check(api.prepare(ctx,C.byref(p),lo.ptr,None,sf.ptr,C.byref(plan),C.byref(info)))
            a=arena.upload(x);route=arena.upload(ids);out=arena.buffer(m*topk*n*4)
            scratch=arena.buffer(info.workspace_bytes);stream=arena.stream();arena.sync()
            api.check(api.run(plan,a.ptr,route.ptr,out.ptr,scratch.ptr,scratch.size,stream));arena.sync()
            got=np.frombuffer(out.bytes(),'f4').reshape(m,topk,n)
            for token in range(m):
                for slot in range(topk):
                    gold,denom=matmul_reference(x[token:token+1],weights[ids[token,slot]],'bf16',output_compute='bf16')
                    compare(got[token,slot:slot+1],gold,denom)
    finally:api.destroy(ctx)


def test_mxfp4_zero_exponent_is_not_zero_scale(package,sdk):
    # Make tiny weights observable with a large, exactly representable BF16
    # input. A device flushing the scale/product must fail, not silently pass
    # an absolute tolerance designed for ordinary model weights.
    e,n,k=8,256,256
    raw=np.zeros((e,n,k//32,17),'u1');raw[...,1:]=0x66  # E2M1 code 6 = 4
    public=Public(package);api=MoE(public.lib);ctx=C.c_void_p()
    api.check(api.create(0,C.byref(ctx)))
    try:
        with Arena(sdk) as arena:
            arr,sizes,lo,sf=packed(arena,public,raw)
            p=Params(1,C.sizeof(Params),39,n,k,e,1,6,1,0,k,k,6,n,arr,sizes)
            plan=C.c_void_p();info=Info(1,C.sizeof(Info))
            api.check(api.prepare(ctx,C.byref(p),lo.ptr,None,sf.ptr,C.byref(plan),C.byref(info)))
            x=np.zeros((1,k),'f4');x[0,0]=2.**120
            a=arena.upload(x);ids=arena.upload(np.arange(6,dtype='i4'))
            out=arena.buffer(6*n*4);scratch=arena.buffer(info.workspace_bytes)
            stream=arena.stream();arena.sync()
            api.check(api.run(plan,a.ptr,ids.ptr,out.ptr,scratch.ptr,scratch.size,stream));arena.sync()
            np.testing.assert_array_equal(np.frombuffer(out.bytes(),'f4'),np.full(6*n,2.**-5,'f4'))
    finally:api.destroy(ctx)
