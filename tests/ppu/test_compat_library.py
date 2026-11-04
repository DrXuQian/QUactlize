"""Real PPU qualification for typed raw/canonical GGUF and changing routes.

These tests consume the installed public library, never a source-built test
kernel. Host tests/compilation cannot stand in for this device gate.
"""
import ctypes as C
from contextlib import contextmanager

import numpy as np
import pytest

from quactlize.compat.native import Library, Sizes, Buffers, Router, SwiGLU, Merge, problem
from quactlize.runtime.native import checked
from ci.support.compat import FORMATS, DENSE_Q8, EXPERT_SHAPES, raw_blocks, weight, encode, decode, assert_dot
from .support import Arena, Public, packed


@pytest.fixture(scope='module')
def lib(package):
    return Library(package/'libncp_quactlize.so')


class Call:
    def __init__(self, arena, lib, p, raw=None, canonical=None, workspace=None):
        self.arena, self.lib, self.p = arena, lib, p
        self.s = Sizes()
        self.arr = canonical[0] if canonical else None
        lib.check(lib.query(C.byref(p), C.byref(self.arr) if self.arr else None, C.byref(self.s)))
        self.stream = arena.stream()
        self.x = np.zeros((p.m,p.channels,p.input_row_stride), dtype='<u2')
        self.a = arena.upload(self.x)
        self.output = arena.buffer(p.m*p.topk*p.output_stride*2)
        self.err = arena.upload(np.zeros(1, dtype='i4'))
        self.ids_host = np.zeros((p.m,p.ids_stride), dtype='i4') if p.mode else None
        self.ids = arena.upload(self.ids_host) if p.mode else None
        self.workspace = workspace or (arena.buffer(self.s.workspace_bytes) if self.s.workspace_bytes else None)
        self.raw = arena.upload(raw) if raw is not None else None
        b = self.b = Buffers(); b.version=1; b.size=C.sizeof(b)
        for name, buf in (('input',self.a), ('output',self.output), ('ids',self.ids),
                          ('workspace',self.workspace), ('raw',self.raw)):
            setattr(b,name,buf.ptr if buf else None)
            setattr(b,name+'_bytes',buf.size if buf else 0)
        if canonical:
            for name,ptr in zip(('low','high','units'),canonical[2]):
                setattr(b,name,ptr); setattr(b,name+'_bytes',getattr(canonical[1],name+'_bytes'))
        b.device_error=self.err.ptr; b.stream=self.stream.value
        self.handle=C.c_void_p()

    def input(self, seed=0):
        rng=np.random.default_rng(seed+43)
        self.x[..., :self.p.k]=encode(rng.uniform(-.12,.12,(self.p.m,self.p.channels,self.p.k)),self.p.dtype)
        self.a.write(self.x)

    def selection(self, shift=0, invalid=False, distributed=False):
        if self.ids is None:return
        p=self.p
        # Repeated experts, inactive slots, first/last experts, skewed/empty groups.
        ids=(np.arange(p.m*p.topk).reshape(p.m,p.topk)*37+shift) % (p.experts if distributed else min(p.experts,3))
        if p.topk>=4:ids[:,:4]=[p.experts-1,p.experts-1,0,-1]
        if invalid:ids[:,0]=p.experts;ids[:,1]=-2
        self.ids_host[:,:p.topk]=ids
        self.ids.write(self.ids_host)

    def prepare(self):
        self.arena.sync()
        self.lib.check(self.lib.prepare(C.byref(self.p), C.byref(self.arr) if self.arr else None,
                                       C.byref(self.b), C.byref(self.handle)))

    def run(self):
        self.lib.check(self.lib.run(self.handle,C.byref(self.b)))

    def reset(self):
        checked(self.arena.sdk.lib.hggcMemsetAsync(self.err.ptr,0,4,self.stream), 'clear device status')

    def result(self):
        self.arena.sync()
        return decode(np.frombuffer(self.output.bytes(),'<u2').reshape(self.p.m,self.p.topk,self.p.output_stride),self.p.dtype)[...,:self.p.n]

    def check(self, raw=None, golden=None):
        p=self.p;got=self.result();x=decode(self.x,p.dtype)[...,:p.k]
        ids=self.ids_host[:,:p.topk] if p.mode else np.zeros((p.m,1),dtype='i4')
        for e in np.unique(ids):
            at=np.argwhere(ids==e)
            if e<0 or e>=p.experts:
                assert np.all(got[at[:,0],at[:,1]]==0)
                continue
            w=golden[e] if golden is not None else weight(raw[e:e+1],p.qtype,p.dtype)[0]
            assert_dot(got[at[:,0],at[:,1]],x[at[:,0],at[:,1] if p.channels>1 else 0],w,p.dtype)
        status=int(np.frombuffer(self.err.bytes(),'i4')[0])
        assert status == (1 if ((ids < -1)|(ids>=p.experts)).any() else 0)

    def close(self):
        self.arena.sync()
        self.lib.destroy(self.handle);self.handle=C.c_void_p()


@contextmanager
def graph(call, body=None):
    sdk=call.arena.sdk;body=body or call.run
    source=C.c_void_p(); executable=C.c_void_p()
    call.arena.sync()
    checked(sdk.lib.hggcStreamBeginCapture(call.stream,1),'begin compat capture')
    try:
        call.reset();body()
    except BaseException:
        sdk.lib.hggcStreamEndCapture(call.stream,C.byref(source))
        if source.value:sdk.lib.hggcGraphDestroy(source)
        raise
    checked(sdk.lib.hggcStreamEndCapture(call.stream,C.byref(source)),'end compat capture')
    try:
        checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(executable),source,0),'instantiate compat graph')
        yield lambda:checked(sdk.lib.hggcGraphLaunch(executable,call.stream),'replay compat graph')
    finally:
        call.arena.sync()
        if executable.value:checked(sdk.lib.hggcGraphExecDestroy(executable),'destroy compat graph exec')
        checked(sdk.lib.hggcGraphDestroy(source),'destroy compat graph')


@pytest.mark.parametrize('q',FORMATS)
@pytest.mark.parametrize('dtype',[0,1])
@pytest.mark.parametrize('m',[1,2,8,17])
def test_formats_and_typed_dequant(lib,sdk,q,dtype,m):
    n,k=37,512 if q>=10 else 640
    raw=raw_blocks(q,1,n,k)
    with Arena(sdk) as arena:
        call=Call(arena,lib,problem(q,m,n,k,dtype=dtype,padding=3),raw)
        try:
            call.input();call.prepare();call.check(raw)
            out=arena.buffer(n*k*2)
            lib.check(lib.dequant(call.handle,C.byref(call.b),0,0,n,out.ptr,out.size))
            arena.sync()
            np.testing.assert_array_equal(np.frombuffer(out.bytes(),'<u2').reshape(1,n,k),encode(weight(raw,q,dtype),dtype))
            assert lib.dequant(call.handle,C.byref(call.b),0,n-1,2,out.ptr,out.size)!=0
        finally:call.close()


@pytest.mark.parametrize('n,k',DENSE_Q8)
@pytest.mark.parametrize('m',[1,16])
@pytest.mark.parametrize('dtype',[0,1])
def test_q8_true_dense_shapes(lib,sdk,n,k,m,dtype):
    raw=raw_blocks(8,1,n,k)
    with Arena(sdk) as arena:
        call=Call(arena,lib,problem(8,m,n,k,dtype=dtype),raw)
        try:
            call.input();call.prepare()
            # The vocabulary matrix is large; compare independent row ranges
            # without materializing its whole FP64 dequantized matrix on CPU.
            if n>16384:
                got=call.result()[:,0];x=decode(call.x,dtype)[:,0]
                for start in (0,n//2,n-64):
                    assert_dot(got[:,start:start+64],x,weight(raw[:,start:start+64],8,dtype)[0],dtype)
            else:call.check(raw)
        finally:call.close()


@pytest.mark.parametrize('q,n,k',EXPERT_SHAPES)
@pytest.mark.parametrize('m,channels',[(1,1),(2,10),(4,1),(8,10),(64,1),(512,10)])
def test_true_experts_dynamic_graph(lib,sdk,q,n,k,m,channels):
    raw=raw_blocks(q,512,n,k)
    with Arena(sdk) as arena:
        p=problem(q,m,n,k,experts=512,topk=10,channels=channels,padding=1)
        call=Call(arena,lib,p,raw)
        try:
            call.input();call.selection();call.prepare();call.check(raw)
            with graph(call) as replay:
                for shift,invalid in ((13,False),(61,True),(7,False)):
                    call.input(shift);call.selection(shift,invalid,distributed=(shift==13));arena.sync()
                    replay();call.check(raw)
                call.ids_host[:,:p.topk]=-1
                call.ids.write(call.ids_host);arena.sync();replay();call.check(raw)
        finally:call.close()


@pytest.mark.parametrize('q',(8,10,11,12,13,14))
@pytest.mark.parametrize('dtype',[0,1])
def test_canonical_bytes_and_native_consumer(package,lib,sdk,q,dtype):
    with Arena(sdk) as arena:
        # packed() checks every byte against an independent CPU producer.
        canonical=packed(arena,Public(package),q,n=256,k=512,experts=4)
        p=problem(q,17,256,512,experts=4,topk=4,channels=4,dtype=dtype,layout=1)
        call=Call(arena,lib,p,canonical=canonical)
        try:
            call.input();call.selection();call.prepare();call.check(golden=canonical[3])
            with graph(call) as replay:
                call.input(91);call.selection(37);arena.sync();replay();call.check(golden=canonical[3])
        finally:call.close()


def test_router_softmax_ties_and_nonfinite(lib,sdk):
    m,e,t=4,512,10
    with Arena(sdk) as arena:
        rng=np.random.default_rng(451)
        x=rng.normal(size=(m,e)).astype('f4');x[0]=0;x[1,:20]=5
        logits=arena.upload(x);ids=arena.buffer(m*t*4);weights=arena.buffer(m*t*4)
        error=arena.upload(np.zeros(1,'i4'));stream=arena.stream()
        p=Router(1,C.sizeof(Router),m,e,t,0,e,t,t,1.,logits.ptr,ids.ptr,weights.ptr,error.ptr,stream.value)
        arena.sync();lib.check(lib.router(C.byref(p)));arena.sync()
        got=np.frombuffer(ids.bytes(),'i4').reshape(m,t)
        gold=np.argsort(-x,axis=1,kind='stable')[:,:t]
        np.testing.assert_array_equal(got,gold)
        prob=np.exp(x-x.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
        w=np.take_along_axis(prob,gold,axis=1);w/=w.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(np.frombuffer(weights.bytes(),'f4').reshape(m,t),w,rtol=2e-6,atol=1e-7)
        x[2,9]=np.nan;logits.write(x);arena.sync();lib.check(lib.router(C.byref(p)));arena.sync()
        assert np.frombuffer(error.bytes(),'i4')[0]==2
        assert np.all(np.frombuffer(ids.bytes(),'i4').reshape(m,t)[2]==-1)
        assert np.all(np.frombuffer(weights.bytes(),'f4').reshape(m,t)[2]==0)


@pytest.mark.parametrize('dtype',[0,1])
@pytest.mark.parametrize('m',[1,8,64])
def test_ffn_chain_shared_scratch(lib,sdk,dtype,m):
    # Same 512/top10 addressing contract, smaller hidden width keeps the full
    # FP64 chain oracle cheap; true 640/2560 projection coverage is above.
    e,t,k,h=512,10,256,64
    gate=raw_blocks(12,e,h,k,1);up=raw_blocks(13,e,h,k,2);down=raw_blocks(7,e,k,h,3)
    with Arena(sdk) as arena:
        calls=[]
        try:
            params=[problem(12,m,h,k,experts=e,topk=t,dtype=dtype),
                    problem(13,m,h,k,experts=e,topk=t,dtype=dtype),
                    problem(7,m,k,h,experts=e,topk=t,channels=t,dtype=dtype)]
            capacities=[]
            for p in params:
                s=Sizes();lib.check(lib.query(C.byref(p),None,C.byref(s)));capacities.append(s.workspace_bytes)
            scratch=arena.buffer(max(capacities)) if max(capacities) else None
            for p,raw in zip(params,(gate,up,down)):
                c=Call(arena,lib,p,raw,workspace=scratch);calls.append(c);c.input();c.selection();c.prepare()
            g,u,d=calls
            # Sequential layers share ONLY scratch, never their live values.
            u.b.input=g.a.ptr;u.b.ids=g.ids.ptr;d.b.ids=g.ids.ptr
            u.b.stream=d.b.stream=g.stream.value
            u.b.device_error=d.b.device_error=g.err.ptr
            activation=SwiGLU(1,C.sizeof(SwiGLU),m*t,h,dtype,0,h,h,h,
                              g.output.ptr,u.output.ptr,d.a.ptr,g.stream.value)
            route=np.full((m,t),1/t,'f4');routing=arena.upload(route);out=arena.buffer(m*k*4)
            merge=Merge(1,C.sizeof(Merge),m,t,k,dtype,k,t,k,d.output.ptr,routing.ptr,out.ptr,g.stream.value)
            def chain():
                g.run();u.run();lib.check(lib.swiglu(C.byref(activation)));d.run();lib.check(lib.merge(C.byref(merge)))
            arena.sync();g.reset();chain();arena.sync()
            with graph(g,chain) as replay:
                for seed in (7,19):
                    g.input(seed);g.selection(seed);arena.sync();replay();arena.sync()
                    # Validate each projection against its ACTUAL input. This
                    # separates matmul error from activation/merge arithmetic.
                    g.check(gate)
                    u.x=g.x;u.ids_host=g.ids_host;u.check(up)
                    gg=g.result();uu=u.result()
                    expected=encode((gg/(1+np.exp(-gg)))*uu,dtype)
                    actual=np.frombuffer(d.a.bytes(),'<u2').reshape(m,t,h)
                    # expf implementations can straddle one native RNE tie.
                    np.testing.assert_allclose(decode(actual,dtype),decode(expected,dtype),rtol=.008 if dtype else .0011,atol=2**-24)
                    d.x=actual;d.ids_host=g.ids_host;d.check(down)
                    want=np.sum(d.result()*route[...,None],axis=1,dtype='f4')
                    np.testing.assert_allclose(np.frombuffer(out.bytes(),'f4').reshape(m,k),want,rtol=2e-6,atol=1e-8)
        finally:
            for c in reversed(calls):c.close()


def test_concurrent_streams_and_changing_addresses(lib,sdk):
    raw=raw_blocks(8,4,320,640)
    with Arena(sdk) as arena:
        p=problem(8,17,320,640,experts=4,topk=4)
        a,b=(Call(arena,lib,p,raw) for _ in range(2))
        try:
            for i,c in enumerate((a,b)):c.input(i);c.selection(i);c.prepare()
            # One immutable plan; independent live outputs/scratch/status.
            arena.sync();lib.check(lib.run(a.handle,C.byref(a.b)));lib.check(lib.run(a.handle,C.byref(b.b)))
            a.check(raw);b.check(raw)
        finally:a.close();b.close()


@pytest.mark.parametrize('dtype',[0,1])
@pytest.mark.parametrize('m',[1,8,64])
@pytest.mark.parametrize('e,t',[(1,1),(512,10)])
def test_merged_gate_up_and_separate_swiglu_agree(lib,sdk,dtype,m,e,t):
    h,k=640,2560
    raw=raw_blocks(12,e,2*h,k)
    with Arena(sdk) as arena:
        call=Call(arena,lib,problem(12,m,2*h,k,dtype=dtype,experts=e,topk=t),raw)
        try:
            call.input();call.selection();call.prepare();call.check(raw)
            merged=arena.buffer(m*t*h*2);separate=arena.buffer(m*t*h*2)
            p=SwiGLU(1,C.sizeof(SwiGLU),m*t,h,dtype,1,2*h,0,h,
                     call.output.ptr,None,merged.ptr,call.stream.value)
            lib.check(lib.swiglu(C.byref(p)))
            p.merged=0;p.up=call.output.ptr+h*2;p.up_stride=2*h;p.output=separate.ptr
            lib.check(lib.swiglu(C.byref(p)));arena.sync()
            assert merged.bytes()==separate.bytes()
        finally:call.close()


def test_zero_rows_and_fail_before_launch(lib,sdk):
    raw=raw_blocks(8,1,4,320)
    with Arena(sdk) as arena:
        call=Call(arena,lib,problem(8,0,4,320),raw)
        try:
            call.prepare();call.run();arena.sync()
            assert int(np.frombuffer(call.err.bytes(),'i4')[0])==0
            call.b.raw_bytes-=1
            assert lib.run(call.handle,C.byref(call.b))!=0
            assert b'capacity' in lib.error()
        finally:call.close()


@pytest.mark.parametrize('q,n,k',[(7,1,32),(8,4,32),(11,19,256),(14,19,256)])
@pytest.mark.parametrize('m',[1,17])
def test_short_blocks_and_gemm_tails(lib,sdk,q,n,k,m):
    raw=raw_blocks(q,1,n,k)
    with Arena(sdk) as arena:
        call=Call(arena,lib,problem(q,m,n,k),raw)
        try:
            call.input();call.prepare();call.check(raw)
        finally:call.close()
