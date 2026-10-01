"""Device regression coverage for the canonical runtime, not model admission."""
import ctypes as C
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.native import SDK, Call, Recipe, Resources, checked
from quactlize.decode.compiler import DecodeCompiler
from quactlize.dispatch.native import DenseIO, DenseCompute
from tools.scan_configs import Fixture, simt_runner, tc_runner

pytestmark=pytest.mark.device


@pytest.fixture(scope='module')
def context():
    from tools.verify_kpack_dispatch import verify
    sdk=Path(os.environ['QUACTLIZE_SDK']).resolve(strict=True)
    bundle=Path(os.environ['QUACTLIZE_BUNDLE']).resolve(strict=True)
    cache=Path(os.environ['QUACTLIZE_JIT_CACHE']).resolve()
    verify(bundle,sdk=sdk)
    return SDK(sdk),sdk,bundle,cache


def parent(q, grouped=False, sf=False):
    route=('sf' if q==8 or sf else 'fq')+('-grouped' if grouped else '-dense')
    tm=8 if q==12 else 16
    return dict(symbol=f'product_q{q}_{route.replace("-","_")}_tm{tm}',qtype=q,route=route,
                tm=tm,tn=64,tk=256,wm=tm,wn=16,stages=2,ap=0,dn=16,
                persistent=0 if grouped and route.startswith('fq') else -1)


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
def test_gpu_pack_exact(context,q):
    import torch
    from reference import gguf_kpack as ref
    sdk,_,bundle,_=context
    f=Fixture(sdk,bundle,q,256,512,9,3,0)
    try:
        if q==8:
            codes=f.raw[...,2:].copy().view('i1').reshape(3,256,32,16).astype('i2')+128
            words=(codes[...,:8]|(codes[...,8:]<<8)).astype('<u2')
            low=words.transpose(0,2,3,1).tobytes()
            units=f.raw[...,:2].transpose(0,2,1,3).tobytes()
            expected=(low,b'',units)
        else:
            raw=torch.from_numpy(f.raw.reshape(-1,f.raw.shape[-1]))
            artifact=ref.prepare_grouped(raw,256,512,q,3)
            expected=tuple(x.numpy().tobytes() for x in (artifact.low,artifact.high,artifact.units))
        for pointer,want in zip((f.low,f.high,f.units),expected):
            assert (sdk.download(pointer,len(want)) if want else b'')==want
    finally:f.close()


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
@pytest.mark.parametrize('compute',[0,1])
@pytest.mark.parametrize('experts,m',[(1,1),(1,8),(3,9)])
def test_simt(context,q,compute,experts,m):
    sdk,_,bundle,_=context
    f=Fixture(sdk,bundle,q,256,512,m,experts,compute)
    try:
        cfg=dict(variant=5 if q==8 else 3,columns=4,warps=4,values=4,split=1)
        for split in (1,2,4):
            run,_,dtype=simt_runner(f,bundle,cfg|dict(split=split))
            sdk.fill(f.output,0xff,m*256*4)
            run();f.check(dtype)
    finally:f.close()


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
@pytest.mark.parametrize('compute,sf',[(0,False),(1,False),(1,True)])
def test_grouped_tc(context,q,compute,sf):
    sdk,sdk_path,bundle,cache=context
    f=Fixture(sdk,bundle,q,256,512,29,3,compute)
    try:
        for split in (1,2):
            destroy=lambda:None
            try:
                run,destroy,dtype=tc_runner(f,sdk_path,cache,parent(q,True,sf),
                    dict(algorithm=0,split=split,grid=0),bundle)
                sdk.fill(f.output,0xff,29*256*4)
                run();f.check(dtype)
            finally:sdk.synchronize(None);destroy()
    finally:f.close()


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
def test_dense_tc_second_tile(context,q):
    sdk,sdk_path,bundle,cache=context
    f=Fixture(sdk,bundle,q,256,512,17,1,0)
    try:
        for split in (1,2):
            destroy=lambda:None
            try:
                run,destroy,dtype=tc_runner(f,sdk_path,cache,parent(q),dict(algorithm=0,split=split,grid=0),bundle)
                sdk.fill(f.output,0xff,17*256*4)
                run();f.check(dtype)
            finally:sdk.synchronize(None);destroy()
    finally:f.close()


@pytest.mark.parametrize('q',[8,10,11,12,13,14])
@pytest.mark.parametrize('compute,input_scale',[(0,1),(1,1),(1,1048576)])
def test_dense_f32_endpoints(context,q,compute,input_scale):
    sdk,sdk_path,bundle,cache=context
    f=Fixture(sdk,bundle,q,256,512,3,1,compute,input_scale)
    handle=C.c_void_p();destroy=None
    try:
        record=DecodeCompiler(sdk_path,cache,compute_type='bf16' if compute else 'f16').build(parent(q))
        lib=C.CDLL(record['path'],mode=C.RTLD_LOCAL)
        device=lib.quactlize_kpack_decode_dense_device_v1
        device.argtypes=[C.c_char_p,C.c_int,C.POINTER(C.c_int32),C.POINTER(C.c_int32)];device.restype=C.c_int
        name=C.create_string_buffer(256);ordinal=C.c_int32();cu=C.c_int32()
        checked(device(name,len(name),C.byref(ordinal),C.byref(cu)),'device')
        c=Call(1,C.sizeof(Call),3,256,512,1,f.arr.group_size,ordinal.value,cu.value,f.arr.mapping_id,
               f.a_f32,f.low,f.high,f.units,None,f.output,None,None,None,None,0,None)
        d=DenseIO(1,C.sizeof(DenseIO),c,1,1)
        typed=DenseCompute(2,C.sizeof(DenseCompute),d,compute)
        query=lib.quactlize_kpack_decode_dense_query_v2
        prepare=lib.quactlize_kpack_decode_dense_prepare_v2
        query.argtypes=[C.POINTER(DenseCompute),C.POINTER(Recipe),C.POINTER(Resources)];query.restype=C.c_int
        prepare.argtypes=[C.POINTER(DenseCompute),C.POINTER(Recipe),C.POINTER(C.c_void_p)];prepare.restype=C.c_int
        run=lib.quactlize_kpack_decode_dense_run_v1;run.argtypes=[C.c_void_p,C.c_void_p];run.restype=C.c_int
        destroy=lib.quactlize_kpack_decode_dense_destroy_v1;destroy.argtypes=[C.c_void_p];destroy.restype=None
        recipe=Recipe(1,C.sizeof(Recipe),0,2,0);sizes=Resources()
        checked(query(C.byref(typed),C.byref(recipe),C.byref(sizes)),'dense F32 query')
        typed.dense.call.workspace=f.allocate(sizes.workspace_bytes)
        typed.dense.call.workspace_bytes=sizes.workspace_bytes
        checked(prepare(C.byref(typed),C.byref(recipe),C.byref(handle)),'dense F32 prepare')
        sdk.fill(f.output,0xff,3*256*4)
        checked(run(handle,None),'dense F32 run');f.check('f32')
    finally:
        sdk.synchronize(None)
        if handle.value and destroy:destroy(handle)
        f.close()
