"""Packed-format/native16 qualification. NOT tuned SIMT/TC performance admission."""
import ctypes as C
import json
import numpy as np
import pytest

from ci.support.compat import raw_blocks, canonical_planes
from ci.support.packing import Arrangement, Sizes
from quactlize.compat.native import Library, problem, Sizes as ComputeSizes
from quactlize.runtime.native import checked
from .support import Arena, Public
from .test_compat_library import Call, graph, lib


def pack(arena,public,raw,q,up=None):
    e,n,blocks,_=raw.shape;k=blocks*(64 if q==42 else 32 if q<10 else 256)
    final_n=n*(2 if up is not None else 1)
    arr,s=Arrangement(),Sizes()
    checked(public.canonical(q,C.byref(arr)),'canonical identity')
    if q in (7,42):
        # A hand-built N4 descriptor must not bypass the unsupported-layout query.
        n4=public.lib.quactlize_gate_up_pack_v1
        n4.argtypes=[C.c_void_p]*5+[C.c_int]*4+[C.c_void_p]*2
        assert n4(None,None,None,None,None,n,k,e,q,None,None)==22
    checked(public.sizes(final_n,k,e,q,C.byref(arr),C.byref(s)),'canonical sizes')
    source=arena.upload(raw);other=arena.upload(up) if up is not None else None
    outputs=[arena.buffer(size) if size else None for size in (s.low_bytes,s.high_bytes,s.units_bytes)]
    pointers=[b.ptr if b else None for b in outputs];stream=arena.stream()
    arena.sync()
    if other:
        checked(public.pack_pair(source.ptr,other.ptr,*pointers,n,k,e,q,C.byref(arr),stream),'paired pack')
        expected=canonical_planes(np.concatenate((raw,up),axis=1),q)
    else:
        checked(public.pack(source.ptr,*pointers,n,k,e,q,C.byref(arr),stream),'pack')
        expected=canonical_planes(raw,q)
    arena.sync()
    for name,buf in zip(('low','high','units'),outputs):
        got=np.frombuffer(buf.bytes(),'u1') if buf else np.empty(0,'u1')
        np.testing.assert_array_equal(got,expected[name],err_msg=f'q={q} {name} canonical bytes')
    assert s.raw_bytes==s.low_bytes+s.high_bytes+s.units_bytes
    # Producer has drained. Prove consumers never retain/read the raw source.
    for buf in (source,other):
        if buf is not None:
            buf.guard();arena.sdk.free(buf.base);arena.buffers.remove(buf)
    return arr,s,pointers,None


@pytest.mark.parametrize('q',[7,8,12,13])
def test_pair_640_device_bytes(package,sdk,q):
    # Independent raw bit patterns exercise metadata preservation, not arithmetic.
    block,width=(32,24) if q==7 else (32,34) if q==8 else (256,144 if q==12 else 176)
    rng=np.random.default_rng(512+q)
    gate=rng.integers(0,256,(3,640,2560//block,width),dtype='u1')
    up=rng.integers(0,256,gate.shape,dtype='u1')
    with Arena(sdk) as arena:pack(arena,Public(package),gate,q,up)


@pytest.mark.parametrize('q,route',[(7,None),(8,None),(7,0)])
def test_down_k640_native_graph(package,lib,sdk,q,route):
    if route is not None:lib=Library(package/'libncp_quactlize.so',packed_route=route)
    e,t,n,k=512,10,2560,640
    raw=raw_blocks(q,e,n,k)
    with Arena(sdk) as arena:
        canonical=pack(arena,Public(package),raw,q)
        max_p=problem(q,128,n,k,experts=e,topk=t,channels=t,dtype=1,layout=1,padding=1)
        capacity=ComputeSizes()
        lib.check(lib.query(C.byref(max_p),C.byref(canonical[0]),C.byref(capacity)))
        scratch=arena.buffer(capacity.workspace_bytes)
        for m in (*range(1,9),9,16,128):
            p=problem(q,m,n,k,experts=e,topk=t,channels=t,dtype=1,layout=1,padding=1)
            call=Call(arena,lib,p,canonical=canonical,workspace=scratch)
            try:
                if route is not None:
                    assert call.s.route==(2 if m<=8 else 3)
                    assert call.s.expanded_weight_bytes==call.s.grouped_input_bytes==call.s.grouped_output_bytes==0
                call.input();call.selection();call.prepare();call.check(raw)
                call.err.write(np.array([16],'i4'))
                with graph(call,clear_status=False) as replay:
                    sticky=16
                    for seed,invalid in ((13,False),(61,True),(7,False)):
                        call.input(seed);call.selection(seed,invalid,distributed=(seed==13))
                        inactive=(call.ids_host[:,:t]<0)|(call.ids_host[:,:t]>=e)
                        call.x[inactive,:k]=0x7fc0 # BF16 NaN: inactive slots must not consume A.
                        call.a.write(call.x);arena.sync();replay();arena.sync()
                        sticky|=int(invalid);call.check(raw,sticky_error=sticky)
                    call.ids_host[:,:t]=-1;call.ids.write(call.ids_host)
                    call.x[:]=0x7fc0;call.a.write(call.x)
                    arena.sync();replay();arena.sync();call.check(raw,sticky_error=sticky)
                print('PACKED_NATIVE_PASS '+json.dumps(dict(qtype=q,m=m,n=n,k=k,experts=e,topk=t,
                    route=call.s.route,workspace_bytes=call.s.workspace_bytes,
                    expanded_weight_bytes=call.s.expanded_weight_bytes,
                    scope='DIRECT_PACKED_CANDIDATE' if route is not None else 'CANONICAL_COMPATIBILITY',
                    performance_admitted=False)),flush=True)
            finally:call.close()


@pytest.mark.parametrize('route',[2,3],ids=['simt','tc'])
@pytest.mark.parametrize('dtype',[0,1],ids=['f16','bf16'])
@pytest.mark.parametrize('m',[1,9,17])
@pytest.mark.parametrize('experts,channels',[(1,1),(3,1),(3,3)])
def test_direct_q5_tails(package,sdk,route,dtype,m,experts,channels):
    lib=Library(package/'libncp_quactlize.so',packed_route=route)
    raw=raw_blocks(7,experts,80,128,seed=m)
    with Arena(sdk) as arena:
        canonical=pack(arena,Public(package),raw,7)
        p=problem(7,m,80,128,experts=experts,topk=experts,channels=channels,
                  dtype=dtype,layout=1,padding=3)
        call=Call(arena,lib,p,canonical=canonical)
        try:
            call.input();call.selection(invalid=True);call.prepare();call.check(raw)
            with graph(call) as replay:
                call.input(125);call.selection(37);arena.sync();replay();call.check(raw)
        finally:call.close()


@pytest.mark.parametrize('route',[2,3])
def test_direct_q5_zero_rows(package,sdk,route):
    lib=Library(package/'libncp_quactlize.so',packed_route=route)
    raw=raw_blocks(7,1,16,128)
    with Arena(sdk) as arena:
        canonical=pack(arena,Public(package),raw,7)
        call=Call(arena,lib,problem(7,0,16,128,layout=1),canonical=canonical)
        try:
            assert call.s.workspace_bytes==0
            call.prepare();call.run();call.check(raw)
        finally:call.close()


@pytest.mark.parametrize('route',[2,3])
def test_direct_q5_dynamic_addresses_and_streams(package,sdk,route):
    lib=Library(package/'libncp_quactlize.so',packed_route=route)
    raw=raw_blocks(7,3,16,640,seed=71)
    with Arena(sdk) as arena:
        canonical=pack(arena,Public(package),raw,7)
        p=problem(7,17,16,640,experts=3,topk=3,channels=3,layout=1,padding=1)
        first=Call(arena,lib,p,canonical=canonical)
        second=Call(arena,lib,p,canonical=canonical)
        try:
            first.input();first.selection();first.prepare();first.check(raw)
            second.input(17);second.selection(invalid=True)
            arena.sync()
            # Same handle, different dynamic A/C/IDs/status/scratch AND stream.
            run_second=lambda:lib.check(lib.run(first.handle,C.byref(second.b)))
            run_second();second.check(raw)
            with graph(first) as a,graph(second,body=run_second) as b:
                for seed in (17,91,3):
                    first.input(seed);first.selection(seed)
                    second.input(seed+1);second.selection(seed,invalid=True)
                    arena.sync();a();b();arena.sync()
                    first.check(raw);second.check(raw)
        finally:
            second.close();first.close()
