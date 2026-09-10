#!/usr/bin/env python3
"""Q8_0 GPU pack -> selected W8A16 dense/grouped -> graph replay gate."""
import argparse
import ctypes as C
import json
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.runtime.native import SDK, Call, checked
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import Arrangement, Sizes, bind, device_identity
from tools.verify_kpack_dispatch import verify


def fixture(n,k,experts):
    rng=np.random.default_rng(83008)
    raw=rng.integers(0,256,(experts,n,k//32,34),dtype=np.uint8)
    d=(rng.integers(1,5,(experts,n,k//32))*0.03125).astype('<f2')
    d[...,::3]*=-1
    raw[...,:2]=d.view('u1').reshape(experts,n,k//32,2)
    codes=raw[...,2:].copy().reshape(experts,n,k)
    kk=np.arange(k)
    low=np.empty((experts,k//2,n,2),dtype=np.uint8)
    low[:,(kk//16)*8+kk%8,:,kk%16//8]=(codes^128).transpose(2,0,1)
    scale=raw[...,:2].transpose(0,2,1,3).copy().reshape(-1)
    # GGUF meaning, independent of placement/reader/converter. W is rounded
    # to FP16 just as the register dequantizer does before FP32 accumulation.
    weight=(codes.view('i1').astype('f4')*np.repeat(d.astype('f4'),32,axis=2)).astype('<f2')
    return raw,low.reshape(-1),scale,weight


def check(got,gold,denom,exact):
    if not np.isfinite(got).all(): raise ValueError('Q8 output is nonfinite')
    if exact:
        # Signed zero may differ after FP32 accumulation; compare its value.
        bad=int(np.count_nonzero(got!=gold.astype('<f2')))
        if bad: raise ValueError(f'Q8 exact one-hot output differs: {bad}/{got.size}')
        return 0.0
    err=float(np.max(np.abs(got.astype('f8')-gold)/np.maximum(denom,1e-30)))
    if err>=0.005: raise ValueError(f'Q8 W8A16 dot differs: {err}')
    return err


def run(args,sdk,packlib,n,k,rows):
    experts=len(rows); route=1 if experts==1 else 3; m=sum(rows); maximum=max(rows)
    raw,low,scale,weight=fixture(n,k,experts)
    arr=Arrangement()
    canonical=packlib.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes=[C.c_int,C.POINTER(Arrangement)]; canonical.restype=C.c_int
    checked(canonical(8,C.byref(arr)),'Q8 canonical descriptor')
    if tuple(getattr(arr,key) for key,_ in arr._fields_)!=(2,4,8,0,0,32,32,0,0x51384B5032540001):
        raise ValueError('Q8 descriptor differs')
    query,pack=bind(sdk,packlib)
    sizes=Sizes()
    checked(query(n,k,experts,8,C.byref(arr),C.byref(sizes)),'Q8 size query')
    if (sizes.raw_bytes,sizes.low_bytes,sizes.high_bytes,sizes.units_bytes)!=(raw.nbytes,low.nbytes,0,scale.nbytes):
        raise ValueError('Q8 plane sizes differ')
    r=Resources(sdk)
    dispatch=Dispatch(args.bundle,jit=dict(python=sys.executable,helper=ROOT/'tools/kpack_jit.py',sdk=args.sdk,cache=args.jit_cache))
    graph,instance=C.c_void_p(),C.c_void_p()
    try:
        source=r.upload(raw); b=r.alloc(low.nbytes); sc=r.alloc(scale.nbytes)
        r.fill(b,0xA5,low.nbytes); r.fill(sc,0xA5,scale.nbytes)
        sdk.synchronize(None)
        checked(pack(source,b,None,sc,n,k,experts,8,C.byref(arr),r.stream),'Q8 GPU producer')
        sdk.synchronize(r.stream)
        if sdk.download(b,low.nbytes)!=low.tobytes() or sdk.download(sc,scale.nbytes)!=scale.tobytes():
            raise ValueError('Q8 GPU pack differs from independent logical scatter')
        choice=dispatch.query(8,route,m,n,k,experts,maximum,arr.mapping_id)
        if choice is None: raise ValueError('Q8 selected parent missing')
        a=r.alloc(m*k*2); out=r.alloc(m*n*2+32)
        bounds=r.alloc((experts+1)*4) if experts>1 else None
        call=Call(1,C.sizeof(Call),m,n,k,experts,32,choice.device,choice.compute_units,arr.mapping_id,
            a,b,None,sc,None,out+16,None,None,bounds,r.alloc(max(16,choice.workspace_bytes)),choice.workspace_bytes,r.stream.value)
        launch=dispatch.prepare(choice,call)
        errors=[]
        for iteration in range(3):
            current=np.roll(np.array(rows,dtype='i4'),iteration)
            offsets=np.r_[0,current.cumsum()].astype('i4')
            owner=np.repeat(np.arange(experts),current)
            act=np.zeros((m,k),dtype='<f2')
            columns=(np.arange(m)*31+iteration*17)%k
            act[np.arange(m),columns]=1
            if iteration==2: act=np.random.default_rng(18).uniform(-0.1,0.1,(m,k)).astype('<f2')
            golden=np.stack([act[i].astype('f8')@weight[e].astype('f8').T for i,e in enumerate(owner)])
            denom=np.stack([np.abs(act[i].astype('f8'))@np.abs(weight[e].astype('f8')).T for i,e in enumerate(owner)])
            checked(sdk.lib.hggcMemcpy(a,act.ctypes.data,act.nbytes,1),'Q8 A upload')
            if bounds: checked(sdk.lib.hggcMemcpy(bounds,offsets.ctypes.data,offsets.nbytes,1),'Q8 rows upload')
            sdk.synchronize(None)
            r.fill(out,0xA5,m*n*2+32)
            if iteration==0:
                checked(launch(),'Q8 eager launch'); sdk.synchronize(r.stream)
                got=np.frombuffer(sdk.download(out+16,m*n*2),dtype='<f2').reshape(m,n)
                check(got,golden,denom,True)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream,0),'Q8 capture begin')
                checked(launch(),'Q8 capture run')
                checked(sdk.lib.hggcStreamEndCapture(r.stream,C.byref(graph)),'Q8 capture end')
                checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance),graph,0),'Q8 instantiate')
                r.fill(out,0xA5,m*n*2+32)
            checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'Q8 replay'); sdk.synchronize(r.stream)
            image=sdk.download(out,m*n*2+32)
            if image[:16]!=b'\xa5'*16 or image[-16:]!=b'\xa5'*16: raise ValueError('Q8 output guard changed')
            got=np.frombuffer(image[16:-16],dtype='<f2').reshape(m,n)
            errors.append(check(got,golden,denom,iteration!=2))
        # Run a finite missing-code negative through the exact same graph.
        r.fill(b,128,low.nbytes); r.fill(out,0xA5,m*n*2+32)
        checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'Q8 planted zero-code'); sdk.synchronize(r.stream)
        got=np.frombuffer(sdk.download(out+16,m*n*2),dtype='<f2').reshape(m,n)
        try: check(got,golden,denom,False)
        except ValueError: pass
        else: raise ValueError('Q8 oracle missed zeroed codes')
        checked(sdk.lib.hggcMemcpy(b,low.ctypes.data,low.nbytes,1),'Q8 restore codes'); sdk.synchronize(None)
        checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'Q8 excluded timing warmup'); sdk.synchronize(r.stream)
        times=r.samples(lambda:sdk.lib.hggcGraphLaunch(instance,r.stream),args.samples)
        result=dict(status='PASS',q=8,activation='FP16',n=n,k=k,rows=list(rows),route=route,
            selection=receipt(choice),pack='GPU_BYTE_EXACT',negative='ZERO_CODE_RED',graph_replays=3,
            errors=errors,samples_us=times,median_us=statistics.median(times),first_launch_excluded=True,
            scope='SELECTED_W8A16_CALL_NOT_LLAMA_ADAPTERS',scale='RESIDENT_RAW_FP16_NO_PREPASS')
        print('Q8_KPACK2_RESULT '+json.dumps(result),flush=True)
        return result
    finally:
        sdk.synchronize(r.stream)
        if instance: checked(sdk.lib.hggcGraphExecDestroy(instance),'destroy graph executable')
        if graph: checked(sdk.lib.hggcGraphDestroy(graph),'destroy graph')
        dispatch.close(); r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('sdk','bundle','pack-library','jit-cache','output'): p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--samples',type=int,default=11)
    args=p.parse_args()
    if args.samples<3: p.error('at least three samples required')
    verify(args.bundle)
    args.output.mkdir(parents=True,exist_ok=False)
    sdk=SDK(args.sdk); graph_bind(sdk)
    library=C.CDLL(str(args.pack_library.resolve()),mode=C.RTLD_LOCAL)
    result=dict(status='INCOMPLETE',device=device_identity(sdk),results=[],failures=[])
    try:
        for rows in ([1],[7],[9],[64],[128],[2,0,3,1],[65,0,8,7]):
            try: result['results'].append(run(args,sdk,library,256,512,rows))
            except Exception as e:
                traceback.print_exc()
                result['failures'].append(dict(rows=rows,error=str(e)))
        result['status']='FAIL' if result['failures'] else 'PASS'
    finally: (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"Q8_KPACK2_GATE status={result['status']} contexts={len(result['results'])}/7 activation=FP16",flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__': sys.exit(main())
