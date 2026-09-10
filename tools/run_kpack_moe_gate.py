#!/usr/bin/env python3
"""GPU pair packing and real selected MoE producer chains; no CPU routed launch.

Independent GGUF dots validate each standalone projection and the composed
chain. The standalone reference's CPU SwiGLU is an untimed test oracle only.
Measured/captured candidate work is entirely on device, with changing IDs/A.
"""
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
from reference import gguf_kpack as ref
from quactlize.dispatch.native import Dispatch, IndexedIO, Router, receipt
from quactlize.runtime.native import SDK, Call, checked
from tools.kpack_warmup_fixture import prepare_expert
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import Arrangement, Sizes, bind, device_identity
from tools.verify_kpack_dispatch import verify


def raw_weight(q,e,n,k,seed):
    block,width=(32,34) if q==8 else (256,ref.SPECS[q].raw_bytes)
    raw=np.random.default_rng(seed).integers(0,256,(e,n,k//block,width),dtype='u1')
    offsets=[0] if q==8 else [ref.SPECS[q].d_offset,ref.SPECS[q].dmin_offset]
    for off in offsets:
        if off>=0: raw[...,off:off+2]=np.array([1/4096],dtype='<f2').view('u1')
    return raw


def cpu_planes(raw,q):
    e,n,sb,width=raw.shape
    if q==8:
        k=sb*32; kk=np.arange(k)
        codes=raw[...,2:].copy().reshape(e,n,k)
        low=np.empty((e,k//2,n,2),dtype='u1')
        low[:,(kk//16)*8+kk%8,:,kk%16//8]=(codes^128).transpose(2,0,1)
        return [low,np.empty(0,dtype='u1'),raw[...,:2].transpose(0,2,1,3).copy()]
    planes=[prepare_expert(r,q,n,sb*256) for r in raw]
    return [np.stack([p[key] for p in planes]) for key in ('low','high','units')]


def pack(r,lib,raw,q,up=None,verify_bytes=True):
    e,n,sb,width=raw.shape; k=sb*(32 if q==8 else 256)
    a=Arrangement()
    canonical=lib.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes=[C.c_int,C.POINTER(Arrangement)];canonical.restype=C.c_int
    checked(canonical(q,C.byref(a)),'canonical arrangement')
    query,one=bind(r.sdk,lib)
    size=Sizes(); checked(query(n*(2 if up is not None else 1),k,e,q,C.byref(a),C.byref(size)),'pack sizes')
    lengths=[size.low_bytes,size.high_bytes,size.units_bytes]
    guards=[r.alloc(b+32) if b else None for b in lengths]
    pointers=[p+16 if p else None for p in guards]
    for g,b in zip(guards,lengths):
        if g: r.fill(g,0xA5,b+32)
    source=r.upload(raw); second=r.upload(up) if up is not None else None
    r.sdk.synchronize(None)
    if up is None: rc=one(source,*pointers,n,k,e,q,C.byref(a),r.stream)
    else:
        pair=lib.quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1
        pair.argtypes=[C.c_void_p]*5+[C.c_int]*4+[C.POINTER(Arrangement),C.c_void_p];pair.restype=C.c_int
        rc=pair(source,second,*pointers,n,k,e,q,C.byref(a),r.stream)
    checked(rc,'GPU pair pack' if up is not None else 'GPU pack');r.sdk.synchronize(r.stream)
    if verify_bytes:
        gold=cpu_planes(np.concatenate([raw,up],axis=1) if up is not None else raw,q)
        for g,b,want in zip(guards,lengths,gold):
            if not b: continue
            image=r.sdk.download(g,b+32)
            if image[:16]!=b'\xa5'*16 or image[-16:]!=b'\xa5'*16 or image[16:-16]!=want.tobytes():
                raise ValueError('GPU pack bytes/guard differ from independent placement')
    return a,pointers,lengths


def dequant(raw,q,expert):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    return dequantize(raw[expert].reshape(-1),GGMLQuantizationType(q)).reshape(raw.shape[1],-1).astype('f8')


def dot(raw,q,ids,act):
    golden=[];denom=[]; cache={}
    for i,e in enumerate(ids.reshape(-1)):
        if int(e) not in cache: cache[int(e)]=dequant(raw,q,int(e))
        w=cache[int(e)];a=act[i].astype('<f2').astype('f8')
        golden.append(w@a);denom.append(np.abs(w)@np.abs(a))
    return np.array(golden),np.array(denom)


def check(got,gold,denom):
    err=float(np.max(np.abs(got-gold)/np.maximum(denom,1e-20)))
    if not np.isfinite(got).all() or not np.isfinite(err) or err>=0.005:
        raise ValueError(f'independent GGUF dot failed err={err}')
    return err


def chain_case(args,sdk,lib,merged,tokens,router_enabled):
    e,n,k,topk=256,512,2048,8;m=tokens*topk
    r=Resources(sdk);d=Dispatch(args.bundle,jit=dict(python=sys.executable,helper=ROOT/'tools/kpack_jit.py',sdk=args.sdk,cache=args.jit_cache))
    graph,instance=C.c_void_p(),C.c_void_p()
    try:
        gate=raw_weight(12,e,n,k,18);up=raw_weight(12,e,n,k,27);down=raw_weight(13,e,k,n,39)
        ids=r.alloc(m*4);source=r.alloc(tokens*k*4);activation=r.alloc(m*n*4)
        route_logits=r.alloc(tokens*e*4);route_weights=r.alloc(m*4)
        parts=[]
        weights=[(gate,12,source,k,1,up if merged else None)]
        if not merged: weights.append((up,12,source,k,1,None))
        weights.append((down,13,activation,n,topk,None))
        for raw,q,inp,kk,channels,paired in weights:
            arr,planes,lengths=pack(r,lib,raw,q,up=paired,verify_bytes=False)
            nn=raw.shape[1]*(2 if paired is not None else 1)
            choice=d.query(q,2,m,nn,kk,e,tokens,arr.mapping_id)
            if choice is None: raise ValueError('selected grouped parent unavailable')
            out=r.alloc(m*nn*4+32);r.fill(out,0xA5,m*nn*4+32)
            call=Call(1,C.sizeof(Call),m,nn,kk,e,arr.group_size,choice.device,choice.compute_units,arr.mapping_id,
                r.alloc(m*kk*2),*planes[:2],planes[2],None,r.alloc(m*nn*2),None,None,
                r.alloc((e+1)*4),r.alloc(max(16,choice.workspace_bytes)),choice.workspace_bytes,r.stream.value)
            io=IndexedIO(1,C.sizeof(IndexedIO),tokens,topk,channels,0,topk,kk,channels*kk,nn,
                         ids,inp,out+16,r.alloc(m*4))
            run=d.prepare(choice,call,indexed=io)
            parts.append(dict(run=run,handle=d.handles[-1],out=out,n=nn,choice=receipt(choice)))
        router=Router(1,C.sizeof(Router),0,1,0,0,1e-8,1,route_logits,None,route_weights) if router_enabled else None
        launch=d.chain(parts[0]['handle'],None if merged else parts[1]['handle'],parts[-1]['handle'],r.stream.value,router)
        print('KPACK_MOE_SELECTED '+json.dumps(dict(merged=merged,tokens=tokens,router=router_enabled,choices=[p['choice'] for p in parts])),flush=True)
        errors=[]
        for replay in range(3):
            rng=np.random.default_rng(432+replay)
            logits=rng.uniform(-1,1,(tokens,e)).astype('f4')
            chosen=np.argsort(-logits,axis=1,kind='stable')[:,:topk].astype('i4')
            act=rng.uniform(-.1,.1,(tokens,k)).astype('f4')
            for dst,val in [(ids,chosen),(source,act),(route_logits,logits)]:
                checked(sdk.lib.hggcMemcpy(dst,val.ctypes.data,val.nbytes,1),'fixture upload')
            sdk.synchronize(None)
            # Independent baseline uses the actual selected indexed projections;
            # CPU copies and SwiGLU below are never in candidate timing/capture.
            for part in parts[:-1]: checked(part['run'](),'standalone projection')
            sdk.synchronize(r.stream)
            go=np.frombuffer(sdk.download(parts[0]['out']+16,m*parts[0]['n']*4),dtype='f4').reshape(m,-1).copy()
            if merged: go,uo=go[:,:n].copy(),go[:,n:].copy()
            else: uo=np.frombuffer(sdk.download(parts[1]['out']+16,m*n*4),dtype='f4').reshape(m,n).copy()
            aa=np.repeat(act,topk,axis=0)
            for value,raw in ((go,gate),(uo,up)):
                gold,denom=dot(raw,12,chosen,aa);errors.append(check(value,gold,denom))
            activated=(go/(1+np.exp(-go))*uo).astype('f4')
            checked(sdk.lib.hggcMemcpy(activation,activated.ctypes.data,activated.nbytes,1),'oracle activation upload');sdk.synchronize(None)
            checked(parts[-1]['run'](),'standalone down');sdk.synchronize(r.stream)
            baseline=np.frombuffer(sdk.download(parts[-1]['out']+16,m*k*4),dtype='f4').reshape(m,k).copy()
            gold,denom=dot(down,13,chosen,activated);errors.append(check(baseline,gold,denom))
            r.fill(parts[-1]['out'],0xA5,m*k*4+32)
            if replay==0:
                checked(launch(),'excluded eager chain');sdk.synchronize(r.stream)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream,0),'chain capture')
                checked(launch(),'captured chain')
                checked(sdk.lib.hggcStreamEndCapture(r.stream,C.byref(graph)),'capture end')
                checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance),graph,0),'instantiate')
            # Poison original down input: chain must produce it on device.
            r.fill(activation,0x7F,m*n*4)
            checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'chain replay');sdk.synchronize(r.stream)
            image=sdk.download(parts[-1]['out'],m*k*4+32)
            if image[:16]!=b'\xa5'*16 or image[-16:]!=b'\xa5'*16: raise ValueError('chain output guard changed')
            got=np.frombuffer(image[16:-16],dtype='f4').reshape(m,k)
            errors.append(check(got,gold,denom))
            check(got,baseline,denom)
            if router:
                observed=np.frombuffer(sdk.download(ids,m*4),dtype='i4').reshape(tokens,topk)
                if not np.array_equal(observed,chosen): raise ValueError('fused router IDs differ')
                expected=np.exp(np.take_along_axis(logits.astype('f8'),chosen,axis=1))
                expected/=expected.sum(1,keepdims=True)
                actual=np.frombuffer(sdk.download(route_weights,m*4),dtype='f4').reshape(tokens,topk)
                if not np.allclose(actual,expected,rtol=2e-5,atol=1e-7): raise ValueError('router weights differ')
        checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'excluded graph warmup');sdk.synchronize(r.stream)
        samples=r.samples(lambda:sdk.lib.hggcGraphLaunch(instance,r.stream),args.samples)
        result=dict(status='PASS',merged=merged,tokens=tokens,router=router_enabled,graph_replays=3,
            choices=[p['choice'] for p in parts],errors=errors,samples_us=samples,median_us=statistics.median(samples),
            scope='REAL_SELECTED_GPU_CHAIN',first_launch_excluded=True)
        print('KPACK_MOE_RESULT '+json.dumps(result),flush=True)
        return result
    finally:
        sdk.synchronize(r.stream)
        if instance: checked(sdk.lib.hggcGraphExecDestroy(instance),'graph destroy')
        if graph: checked(sdk.lib.hggcGraphDestroy(graph),'capture destroy')
        d.close();r.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sdk','bundle','pack-library','jit-cache','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--samples',type=int,default=11)
    args=parser.parse_args()
    if args.samples<3:parser.error('at least 3 samples')
    verify(args.bundle);args.output.mkdir(parents=True,exist_ok=False)
    sdk=SDK(args.sdk);graph_bind(sdk);lib=C.CDLL(str(args.pack_library.resolve()),mode=C.RTLD_LOCAL)
    result=dict(status='INCOMPLETE',device=device_identity(sdk),pairs=[],chains=[],failures=[])
    try:
        for q in (8,10,11,12,13,14):
            r=Resources(sdk)
            try:
                raw=raw_weight(q,3,256,512,q);up=raw_weight(q,3,256,512,q+100)
                pack(r,lib,raw,q,up=up)
                result['pairs'].append(q);print(f'KPACK_PAIR_GPU q={q} status=PASS guard=PASS',flush=True)
            except Exception as error:
                traceback.print_exc();result['failures'].append(dict(q=q,error=str(error)))
            finally:r.close()
        for merged,tokens,router in ((False,1,False),(True,1,False),(False,4,True),(True,4,True)):
            try:result['chains'].append(chain_case(args,sdk,lib,merged,tokens,router))
            except Exception as error:
                traceback.print_exc();result['failures'].append(dict(merged=merged,tokens=tokens,router=router,error=str(error)))
        result['status']='FAIL' if result['failures'] else 'PASS'
    finally:(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"KPACK_MOE_GATE status={result['status']} pair_formats={len(result['pairs'])}/6 chains={len(result['chains'])}/4",flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__':sys.exit(main())
