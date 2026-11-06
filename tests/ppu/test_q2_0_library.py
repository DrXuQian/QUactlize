"""Actual PPU Q2_0 gates, including the installed-wheel path. No timing admission."""
import ctypes as C
from contextlib import ExitStack
import json

import numpy as np
import pytest

from ci.support.compat import raw_blocks, encode, weight
from ci.support.serving import row_blocks, dequant_rows
from quactlize.compat.native import problem, Sizes
from quactlize.compat.serving_native import Library as Serving, Rows
from .support import Arena, Public
from .test_compat_library import Call, graph, lib
from .test_packed_native import pack

MS = (1,2,4,6,7,8,9,16,128)


@pytest.mark.parametrize('n,k,channels',[(640,2560,1),(1280,2560,1),(2560,640,10)])
@pytest.mark.parametrize('dtype',[0,1],ids=['f16','bf16'])
def test_q2_0_model_routes(package,lib,sdk,n,k,channels,dtype):
    raw = raw_blocks(42,512,n,k)
    with Arena(sdk) as arena:
        canonical = pack(arena,Public(package),raw,42)
        p = problem(42,max(MS),n,k,experts=512,topk=10,channels=channels,dtype=dtype,padding=3)
        size = Sizes();lib.check(lib.query(C.byref(p),None,C.byref(size)))
        scratch = arena.buffer(size.workspace_bytes)
        source = arena.upload(raw)
        for m in MS:
            calls = []
            try:
                for layout in (0,1):
                    p = problem(42,m,n,k,experts=512,topk=10,channels=channels,dtype=dtype,
                                layout=layout,padding=3)
                    call = Call(arena,lib,p,canonical=canonical if layout else None,workspace=scratch)
                    calls.append(call)
                    if not layout:call.b.raw=source.ptr;call.b.raw_bytes=source.size
                    else:assert not call.b.raw and call.s.raw_bytes==0
                    if len(calls)>1:call.stream=calls[0].stream;call.b.stream=call.stream.value
                    call.input();call.selection();call.prepare();call.check(raw)
                    call.err.write(np.array([16],'i4'))
                    assert call.s.expanded_weight_bytes <= 2*4*512*k
                    # Exact native-16 weight bits, independently of GEMM order.
                    decoded = arena.buffer(3*k*2)
                    lib.check(lib.dequant(call.handle,C.byref(call.b),511,n-3,3,decoded.ptr,decoded.size))
                    arena.sync()
                    np.testing.assert_array_equal(np.frombuffer(decoded.bytes(),'<u2').reshape(1,3,k),
                                                  encode(weight(raw[511:512,n-3:],42,dtype),dtype))
                def body():
                    for call in calls:call.run()  # Same scratch, same stream, ordered projections.
                with graph(calls[0],body=body,clear_status=False) as replay:
                    sticky=16
                    for seed,invalid in ((13,False),(61,True),(7,False)):
                        for call in calls:
                            call.input(seed);call.selection(seed,invalid)
                            if channels==10:
                                inactive=(call.ids_host[:,:10]<0)|(call.ids_host[:,:10]>=512)
                                call.x[inactive,:k]=0x7fc0 if dtype else 0x7e00
                                call.a.write(call.x)
                        arena.sync();replay();arena.sync();sticky|=int(invalid)
                        for call in calls:call.check(raw,sticky_error=sticky)
                    for call in calls:
                        call.ids_host[:,:10]=-1;call.ids.write(call.ids_host)
                        call.x[:]=0x7fc0 if dtype else 0x7e00;call.a.write(call.x)
                    arena.sync();replay();arena.sync()
                    for call in calls:call.check(raw,sticky_error=sticky)
                print('Q2_0_MODEL_PASS '+json.dumps(dict(m=m,n=n,k=k,experts=512,topk=10,
                    channels=channels,dtype=dtype,route=calls[-1].s.route,
                    workspace_bytes=calls[-1].s.workspace_bytes,raw_canonical=True,
                    scope='PPU_NUMERICS_AND_GRAPH_NOT_PERFORMANCE')),flush=True)
            finally:
                for call in reversed(calls):call.close()


@pytest.mark.parametrize('dtype',[0,1])
def test_q2_0_dense_alternating_graphs(package,lib,sdk,dtype):
    with Arena(sdk) as arena, ExitStack() as stack:
        specs=[(1,4,640),(7,37,192),(9,320,640),(128,64,256)]
        sizes=[]
        for m,n,k in specs:
            s=Sizes();p=problem(42,m,n,k,dtype=dtype)
            lib.check(lib.query(C.byref(p),None,C.byref(s)));sizes.append(s.workspace_bytes)
        scratch=arena.buffer(max(sizes));stream=arena.stream()
        entries=[]
        for m,n,k in specs:
            raw=raw_blocks(42,1,n,k)
            canonical=pack(arena,Public(package),raw,42)
            call=Call(arena,lib,problem(42,m,n,k,dtype=dtype,layout=1,padding=1),
                      canonical=canonical,workspace=scratch)
            stack.callback(call.close)
            call.stream=stream;call.b.stream=stream.value
            call.input();call.prepare();call.check(raw)
            replay=stack.enter_context(graph(call))
            entries.append((call,replay,raw))
        for seed in (11,67,43):
            for call,replay,raw in reversed(entries):
                call.input(seed);arena.sync();replay();call.check(raw)


@pytest.mark.parametrize('dtype',[0,1])
def test_q2_0_zero_rows(package,lib,sdk,dtype):
    with Arena(sdk) as arena:
        raw=raw_blocks(42,1,4,64);canonical=pack(arena,Public(package),raw,42)
        call=Call(arena,lib,problem(42,0,4,64,layout=1,dtype=dtype),canonical=canonical)
        try:
            assert call.s.workspace_bytes==0
            call.prepare();call.run();call.check(raw)
        finally:call.close()


@pytest.mark.parametrize('dtype',[0,1])
def test_q2_0_pair_bytes(package,sdk,dtype):
    # All code/scale bits, including signed zero, subnormal, Inf/NaN scale payloads.
    rng=np.random.default_rng(dtype+42)
    gate=rng.integers(0,256,(3,640,40,18),dtype='u1')
    up=rng.integers(0,256,gate.shape,dtype='u1')
    with Arena(sdk) as arena:pack(arena,Public(package),gate,42,up)


@pytest.mark.parametrize('dtype',[0,1])
@pytest.mark.parametrize('columns',[640,2560])
def test_q2_0_serving_rows(package,sdk,dtype,columns):
    serving=Serving(package/'libncp_quactlize.so')
    with Arena(sdk) as arena:
        stream=arena.stream();serving.check(serving.prepare(0,stream))
        raw=row_blocks(42,7,columns);table=arena.upload(raw)
        selection=np.array([0,6,6,-1,-2,7,2],'i8');ids=arena.upload(selection)
        out=arena.buffer(7*columns*2);status=arena.upload(np.array([16],'i4'))
        count=arena.upload(np.array([0],'i4'))
        p=Rows(1,C.sizeof(Rows),42,dtype,7,columns,7,raw.nbytes,56,out.size,
               table.ptr,ids.ptr,out.ptr,count.ptr,status.ptr,stream.value)
        from types import SimpleNamespace
        call=SimpleNamespace(arena=arena,stream=stream,
                             run=lambda:serving.check(serving.rows(C.byref(p))))
        with graph(call,clear_status=False) as replay:
            for live in (0,3,7,-1,8,7):
                selection=np.roll(selection,1);ids.write(selection)
                count.write(np.array([live],'i4'));arena.sync();replay();arena.sync()
                gold=np.zeros((7,columns),'f4')
                if 0<=live<=7:
                    for row,index in enumerate(selection[:live]):
                        if 0<=index<7:gold[row]=dequant_rows(raw,42)[index]
                np.testing.assert_array_equal(np.frombuffer(out.bytes(),'<u2').reshape(7,columns),encode(gold,dtype))
            assert int(np.frombuffer(status.bytes(),'i4')[0])==21  # sticky BAD_ID | BAD_COUNT | caller bit
