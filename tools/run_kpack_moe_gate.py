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
from quactlize.dispatch.native import Dispatch, IndexedIO, Router, MoeEndpoint, MoeEndpointV3, MoeEndpointV4, Choice, Request, receipt
from quactlize.execution.native import Call as GemvCall, Sizes as GemvSizes, SimtCallV2, bind_simt, bind_simt_compute
from quactlize.runtime.native import SDK, Call, checked
from tools.kpack_warmup_fixture import prepare_expert
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import Arrangement, Sizes, bind, device_identity
from quactlize.runtime.compiler import sha
from tools.verify_kpack_dispatch import verify

CHAIN_CASES=tuple((merged,tokens,router) for tokens in (1,2,3,4)
                  for merged in (False,True) for router in (False,True))


PACK_SYMBOLS = (
    'quactlize_ppu_kpack_canonical_arrangement_v1',
    'quactlize_ppu_kpack_sizes_for_arrangement_v1',
    'quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2',
    'quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1',
)


def load_pack_library(path):
    """Validate the actual paired producer before allocating any fixtures.

    The original kpack-pack-v1 DSO only supports the single-source API. It
    cannot supply this gate's canonical query or fused gate/up producer.
    These host queries establish ABI compatibility, not GPU correctness.
    """
    from quactlize.execution.native import arrangement
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    if (manifest.get('schema') != 'quactlize.kpack-device-pack-build.v1' or
            manifest.get('library') != path.name or sha(path) != manifest.get('sha256')):
        raise ValueError(f'pack library differs from its build manifest: {path}')
    lib = C.CDLL(str(path), mode=C.RTLD_LOCAL)
    missing = [name for name in PACK_SYMBOLS if not hasattr(lib, name)]
    if missing:
        raise ValueError(f'pack library lacks the MoE producer ABI: {path}; missing={missing}; '
                         'use kpack-fusion-v1/libquactlize_ppu_pack.so, not kpack-pack-v1')
    canonical = lib.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes, canonical.restype = [C.c_int, C.POINTER(Arrangement)], C.c_int
    query = lib.quactlize_ppu_kpack_sizes_for_arrangement_v1
    query.argtypes, query.restype = [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(Sizes)], C.c_int
    formats = [8, 10, 11, 12, 13, 14]
    for q in formats:
        actual = Arrangement()
        checked(canonical(q, C.byref(actual)), 'pack canonical query')
        if bytes(actual) != bytes(arrangement(q)):
            raise ValueError(f'pack canonical arrangement differs from consumer: q={q}')
        for n in (256, 512):  # Single and paired N, with more than one expert.
            e, k = 2, 512
            block, width = (32, 34) if q == 8 else (256, ref.SPECS[q].raw_bytes)
            raw = e * n * (k // block) * width
            low, high = (e * n * k * b // 8 for b in (actual.bits, actual.high_bits))
            expected = (raw, low, high, raw - low - high)
            sizes = Sizes()
            checked(query(n, k, e, q, C.byref(actual), C.byref(sizes)), 'pack plane sizes')
            if tuple(getattr(sizes, name) for name, _ in Sizes._fields_) != expected:
                raise ValueError(f'pack plane sizes differ from contract: q={q} n={n}')
        wrong = Arrangement.from_buffer_copy(actual)
        wrong.mapping_id ^= 1
        if query(256, 512, 2, q, C.byref(wrong), C.byref(Sizes())) != 38:
            raise ValueError(f'pack producer accepts a wrong mapping: q={q}')
    record = dict(path=str(path), sha256=manifest['sha256'], symbols=list(PACK_SYMBOLS),
                  formats=formats, scope='HOST_ABI_ONLY', status='PASS')
    print('KPACK_MOE_PACK_LIBRARY ' + json.dumps(record), flush=True)
    return lib, record


def chain_requests(merged,tokens,down_q=13):
    """Actual selector inputs, shared by the host coverage test and device gate."""
    weights=[('gate',12,1024 if merged else 512,2048)]
    if not merged: weights.append(('up',12,512,2048))
    weights.append(('down',down_q,2048,512))
    return [dict(projection=name,q=q,route=2,m=tokens*8,n=n,k=k,experts=256,max_rows=tokens)
            for name,q,n,k in weights]


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


def dot(raw,q,ids,act,compute=0):
    from dev.bf16_compute.fixture import round_compute
    golden=[];denom=[]; cache={}
    for i,e in enumerate(ids.reshape(-1)):
        if int(e) not in cache: cache[int(e)]=dequant(raw,q,int(e))
        w=cache[int(e)];a=round_compute(act[i],'bf16' if compute else 'f16').astype('f8')
        golden.append(w@a);denom.append(np.abs(w)@np.abs(a))
    return np.array(golden),np.array(denom)


def check(got,gold,denom):
    err=float(np.max(np.abs(got-gold)/np.maximum(denom,1e-20)))
    if not np.isfinite(got).all() or not np.isfinite(err) or err>=0.005:
        raise ValueError(f'independent GGUF dot failed err={err}')
    return err


def automatic_smallm(dispatch, call, arrangement, compute):
    """Use the same priority as the caller; never overwrite a matched TC ticket."""
    matched=dispatch.query_smallm_matched(call,arrangement,compute)
    if matched is not None:
        return matched.base,matched.q4 if matched.base.kind==2 else None
    legacy=dispatch.query_smallm(call,arrangement,compute if compute else None) if call.qtype!=12 else None
    return legacy,None


def chain_case(args,sdk,lib,merged,tokens,router_enabled,down_q=13):
    e,n,k,topk=256,512,2048,8;m=tokens*topk
    compute=int(getattr(args,'compute','fp16')=='bf16')
    jit=dict(python=sys.executable,helper=ROOT/'tools/kpack_jit.py',sdk=args.sdk,cache=args.jit_cache) if args.jit_cache else None
    r=Resources(sdk);d=Dispatch(args.bundle,jit=jit)
    graph,instance=C.c_void_p(),C.c_void_p()
    try:
        gate=raw_weight(12,e,n,k,18);up=raw_weight(12,e,n,k,27);down=raw_weight(down_q,e,k,n,39)
        ids=r.alloc(m*4);source=r.alloc(tokens*k*4);activation=r.alloc(m*n*4)
        route_logits=r.alloc(tokens*e*4);route_weights=r.alloc(m*4)
        parts=[]
        weights=[(gate,12,source,k,1,up if merged else None)]
        if not merged: weights.append((up,12,source,k,1,None))
        weights.append((down,down_q,activation,n,topk,None))
        requests=chain_requests(merged,tokens,down_q)
        assert len(weights)==len(requests)
        for (raw,q,inp,kk,channels,paired),request in zip(weights,requests):
            arr,planes,lengths=pack(r,lib,raw,q,up=paired,verify_bytes=False)
            if paired is not None:
                canonical_pair=(arr,planes,lengths)
            nn=raw.shape[1]*(2 if paired is not None else 1)
            assert (q,m,nn,kk,e,tokens)==tuple(request[key] for key in ('q','m','n','k','experts','max_rows'))
            print('KPACK_MOE_QUERY '+json.dumps(dict(request,merged=merged,router=router_enabled)),flush=True)
            out=r.alloc(m*nn*4+32);r.fill(out,0xA5,m*nn*4+32)
            mixed=getattr(args,'mixed',False)
            table=getattr(args,'smallm_table',False)
            endpoint_type=MoeEndpointV3 if table else MoeEndpoint
            version=3 if table else 2
            def wrap(endpoint):
                return MoeEndpointV4(4,C.sizeof(MoeEndpointV4),endpoint,compute) if compute else endpoint
            choice=None
            selected=q4_selected=None
            if mixed and table:
                gc=GemvCall(version=1,size=C.sizeof(GemvCall),qtype=q,n=nn,k=kk,experts=e,rows=m,mode=2,
                    input_type=1,channels=channels,topk=topk,a_row_stride=kk,a_token_stride=channels*kk,
                    ids_stride=topk,out_row_stride=nn,a=inp,low=planes[0],high=planes[1],units=planes[2],ids=ids,
                    output=out+16,stream=r.stream.value)
                selected,q4_selected=automatic_smallm(d,gc,arr,compute)
                if selected is not None and selected.kind==1:
                    execution=C.CDLL(str(args.bundle/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
                    query_simt,run=bind_simt_compute(execution);cfg=selected.simt
                    sizes=GemvSizes();typed=SimtCallV2(gc,compute)
                    checked(query_simt(C.byref(typed),C.byref(cfg),C.byref(arr),C.byref(sizes)),'selected SIMT query')
                    gc.workspace_bytes=sizes.workspace_bytes
                    gc.workspace=r.alloc(sizes.workspace_bytes) if sizes.workspace_bytes else None
                    typed=SimtCallV2(gc,compute)
                    count=d.simt_scratch(gc);scratch=r.alloc(count+512);r.fill(scratch,0xA5,count+512)
                    endpoint=endpoint_type(version,C.sizeof(endpoint_type),None,C.addressof(gc),None,None,
                        C.addressof(arr),scratch+256,count,C.addressof(cfg))
                    parts.append(dict(run=lambda run=run,typed=typed,cfg=cfg,arr=arr:run(C.byref(typed),C.byref(cfg),C.byref(arr)),
                        endpoint=wrap(endpoint),keep=(gc,typed,cfg,arr,execution),out=out,n=nn,scratch=(scratch,count),
                        choice=dict(route='SIMT',reader='register-reuse',policy=selected.policy,
                            recipe=[getattr(cfg,x) for x in ('variant','columns','warps','values','split')])))
                    continue
                if selected is not None and selected.kind==0:choice=selected.tc
            if mixed and q==12 and choice is None:
                from tools.run_q4_decode_policy_gate import Config as Q4Config
                execution=C.CDLL(str(args.bundle/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
                select=execution.quactlize_kpack_q4_decode_select_v2
                select.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Arrangement),C.POINTER(Q4Config),C.POINTER(GemvSizes)]
                select.restype=C.c_int
                gc=GemvCall(version=1,size=C.sizeof(GemvCall),qtype=q,n=nn,k=kk,experts=e,rows=m,mode=2,
                    input_type=1,channels=channels,topk=topk,a_row_stride=kk,a_token_stride=channels*kk,
                    ids_stride=topk,out_row_stride=nn,a=inp,low=planes[0],high=planes[1],units=planes[2],ids=ids,
                    output=out+16,stream=r.stream.value)
                typed=SimtCallV2(gc,compute)
                cfg=Q4Config();sizes=GemvSizes();rc=select(C.byref(typed),C.byref(arr),C.byref(cfg),C.byref(sizes))
                if q4_selected is not None and (rc or bytes(cfg)!=bytes(q4_selected)):
                    raise ValueError('matched Q4 recipe changed at the execution boundary')
                if rc==0:
                    run=execution.quactlize_kpack_q4_decode_run_v2
                    run.argtypes=[C.POINTER(SimtCallV2),C.POINTER(Q4Config),C.POINTER(Arrangement)];run.restype=C.c_int
                    count=d.simt_scratch(gc);scratch=r.alloc(count+512);r.fill(scratch,0xA5,count+512)
                    endpoint=endpoint_type(version,C.sizeof(endpoint_type),None,C.addressof(gc),C.addressof(cfg),None,
                        C.addressof(arr),scratch+256,count)
                    parts.append(dict(run=lambda run=run,typed=typed,cfg=cfg,arr=arr:run(C.byref(typed),C.byref(cfg),C.byref(arr)),
                        endpoint=wrap(endpoint),keep=(gc,typed,cfg,arr,execution),out=out,n=nn,scratch=(scratch,count),
                        choice=dict(route='SIMT',policy=selected.policy if selected is not None else ('INITIAL_BF16' if compute else 'Q4_AUTO'),
                            recipe=[getattr(cfg,x) for x in ('reader','variant','warps','values','columns')])))
                    continue
                if rc!=24:raise ValueError(f'automatic Q4 SIMT selection rejected: {rc}')
                req=Request(1,C.sizeof(Request),q,2,m,nn,kk,e,tokens,arr.mapping_id)
                choice=d.query_compute(req,compute,decode=True)
            if choice is None:
                req=Request(1,C.sizeof(Request),*[request[key] for key in ('q','route','m','n','k','experts','max_rows')],arr.mapping_id)
                choice=d.query_compute(req,compute) if compute else d.query(*[request[key] for key in ('q','route','m','n','k','experts','max_rows')],arr.mapping_id)
            if choice is None:
                raise ValueError('selected grouped parent unavailable: '+json.dumps(dict(request,reason=d.last_miss)))
            call=Call(1,C.sizeof(Call),m,nn,kk,e,arr.group_size,choice.device,choice.compute_units,arr.mapping_id,
                r.alloc(m*kk*2),*planes[:2],planes[2],None,r.alloc(m*nn*2),None,None,
                r.alloc((e+1)*4),r.alloc(max(16,choice.workspace_bytes)),choice.workspace_bytes,r.stream.value)
            io=IndexedIO(1,C.sizeof(IndexedIO),tokens,topk,channels,0,topk,kk,channels*kk,nn,
                         ids,inp,out+16,r.alloc(m*4))
            run=d.prepare_compute(choice,call,tokens,compute,indexed=io) if compute else d.prepare(choice,call,indexed=io)
            endpoint=endpoint_type(version,C.sizeof(endpoint_type),d.handles[-1],None,None,None,None,None,0)
            parts.append(dict(run=run,handle=d.handles[-1],endpoint=wrap(endpoint),out=out,n=nn,choice=receipt(choice)))
        router=Router(1,C.sizeof(Router),0,1,0,0,1e-8,1,route_logits,None,route_weights) if router_enabled else None
        field='endpoint' if getattr(args,'mixed',False) else 'handle'
        launch=d.chain(parts[0][field],None if merged else parts[1][field],parts[-1][field],r.stream.value,router,mixed=getattr(args,'mixed',False))
        paired_choice=None
        if getattr(args,'paired',False):
            from quactlize.fusion.native import Library as FusionLibrary, integration_entries, Repack, MoeBinding, FusionCall, Config as FusionConfig
            fusion=FusionLibrary(args.bundle/'libquactlize_ppu_gate_up.so');entries=integration_entries(fusion)
            layout=fusion.arrangement(12);config=FusionConfig()
            checked(entries['select'](12,n,k,e,tokens,compute,C.byref(config)),'paired selection')
            arr,canonical,lengths=canonical_pair
            auxiliary=[r.alloc(size) if size else None for size in lengths]
            repack=Repack(1,C.sizeof(Repack),12,n,k,e,1,canonical[0],canonical[2],None,None,auxiliary[0],auxiliary[2])
            checked(entries['repack'](C.byref(repack),C.byref(layout),r.stream),'paired canonical repack')
            call=GemvCall(version=1,size=C.sizeof(GemvCall),qtype=12,n=n,k=k,experts=e,rows=m,mode=2,
                input_type=1,channels=1,topk=8,a_row_stride=k,a_token_stride=k,ids_stride=8,out_row_stride=n)
            typed=FusionCall(call,compute,1,1);size=GemvSizes()
            checked(fusion.query(C.byref(typed),C.byref(config),C.byref(layout),C.byref(size)),'paired workspace')
            scratch=r.alloc(size.workspace_bytes) if size.workspace_bytes else None
            binding=MoeBinding(1,C.sizeof(MoeBinding),*auxiliary,scratch,size.workspace_bytes,layout,config)
            fn=d.lib.quactlize_kpack_dispatch_moe_bind_gate_up_v1
            fn.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(MoeBinding)];fn.restype=C.c_int
            checked(fn(d.runtime,d.chains[-1],C.byref(binding)),'paired chain binding')
            sdk.synchronize(r.stream)
            paired_choice={name:getattr(config,name) for name in ('backend','split','tile_m','warps')}
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
                gold,denom=dot(raw,12,chosen,aa,compute);errors.append(check(value,gold,denom))
            if mixed:
                from dev.bf16_compute.fixture import round_compute
                go,uo=(round_compute(x,'bf16' if compute else 'f16') for x in (go,uo))
            activated=(go/(1+np.exp(-go))*uo).astype('f4')
            checked(sdk.lib.hggcMemcpy(activation,activated.ctypes.data,activated.nbytes,1),'oracle activation upload');sdk.synchronize(None)
            checked(parts[-1]['run'](),'standalone down');sdk.synchronize(r.stream)
            baseline=np.frombuffer(sdk.download(parts[-1]['out']+16,m*k*4),dtype='f4').reshape(m,k).copy()
            gold,denom=dot(down,down_q,chosen,activated,compute);errors.append(check(baseline,gold,denom))
            r.fill(parts[-1]['out'],0xA5,m*k*4+32)
            sdk.synchronize(None)  # Untimed fixture poison must precede the nonblocking stream.
            if replay==0:
                checked(launch(),'excluded eager chain');sdk.synchronize(r.stream)
                checked(sdk.lib.hggcStreamBeginCapture(r.stream,0),'chain capture')
                checked(launch(),'captured chain')
                checked(sdk.lib.hggcStreamEndCapture(r.stream,C.byref(graph)),'capture end')
                checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(instance),graph,0),'instantiate')
            # Poison original down input: chain must produce it on device.
            r.fill(activation,0x7F,m*n*4)
            sdk.synchronize(None)
            checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'chain replay');sdk.synchronize(r.stream)
            image=sdk.download(parts[-1]['out'],m*k*4+32)
            if image[:16]!=b'\xa5'*16 or image[-16:]!=b'\xa5'*16: raise ValueError('chain output guard changed')
            got=np.frombuffer(image[16:-16],dtype='f4').reshape(m,k)
            errors.append(check(got,gold,denom))
            check(got,baseline,denom)
            for part in parts:
                if 'scratch' in part:
                    base,count=part['scratch']
                    if sdk.download(base,256)!=b'\xa5'*256 or sdk.download(base+256+count,256)!=b'\xa5'*256:
                        raise ValueError('mixed SIMT scratch guard changed')
            if router:
                observed=np.frombuffer(sdk.download(ids,m*4),dtype='i4').reshape(tokens,topk)
                if not np.array_equal(observed,chosen): raise ValueError('fused router IDs differ')
                expected=np.exp(np.take_along_axis(logits.astype('f8'),chosen,axis=1))
                expected/=expected.sum(1,keepdims=True)
                actual=np.frombuffer(sdk.download(route_weights,m*4),dtype='f4').reshape(tokens,topk)
                if not np.allclose(actual,expected,rtol=2e-5,atol=1e-7): raise ValueError('router weights differ')
        checked(sdk.lib.hggcGraphLaunch(instance,r.stream),'excluded graph warmup');sdk.synchronize(r.stream)
        samples=r.samples(lambda:sdk.lib.hggcGraphLaunch(instance,r.stream),args.samples)
        result=dict(status='PASS',compute='bf16' if compute else 'f16',merged=merged,tokens=tokens,router=router_enabled,down_q=down_q,
            paired_gate_up=paired_choice,
            mixed=getattr(args,'mixed',False),simt_projections=sum(p['choice'].get('route')=='SIMT' for p in parts),graph_replays=3,
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
    parser.add_argument('--mixed',action='store_true',help='exercise automatic SIMT/TC mixed chains, not forced TC')
    parser.add_argument('--smallm-table',action='store_true',help='bounded merged-chain check of the new automatic decode table')
    parser.add_argument('--compute',choices=('fp16','bf16'),default='fp16')
    parser.add_argument('--paired',action='store_true',help='paired-N4 gate/up activation with the existing selected down/router')
    args=parser.parse_args()
    if args.samples<3:parser.error('at least 3 samples')
    if args.smallm_table and not args.mixed:parser.error('--smallm-table requires --mixed')
    if args.compute=='bf16' and not args.smallm_table:parser.error('BF16 selected-chain check requires --mixed --smallm-table')
    if args.paired and (args.compute!='bf16' or not args.smallm_table):parser.error('--paired requires BF16 mixed small-M table')
    verify(args.bundle);args.output.mkdir(parents=True,exist_ok=False)
    lib,pack_identity=load_pack_library(args.pack_library)
    sdk=SDK(args.sdk);graph_bind(sdk)
    result=dict(status='INCOMPLETE',device=device_identity(sdk),pack_library=pack_identity,pairs=[],chains=[],failures=[])
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
        cases=[(merged,tokens,router,q) for q in ((13,12) if args.mixed else (13,))
            for tokens in ((1,4,8) if args.mixed else (1,2,3,4))
            for merged in (False,True) for router in (False,True)]
        if args.smallm_table:
            cases=[(True,tokens,router,13) for tokens in (1,2,8) for router in (False,True)]
        if args.paired:
            cases=[(True,tokens,router,13) for tokens in range(1,9) for router in (False,True)]
        for merged,tokens,router,q in cases:
            try:result['chains'].append(chain_case(args,sdk,lib,merged,tokens,router,q))
            except Exception as error:
                traceback.print_exc();result['failures'].append(dict(merged=merged,tokens=tokens,router=router,q=q,error=str(error)))
        if args.mixed and not any(x['simt_projections'] for x in result['chains']):
            result['failures'].append(dict(error='mixed gate executed no SIMT projection'))
        if args.smallm_table and not any(p.get('reader')=='register-reuse' for x in result['chains'] for p in x['choices']):
            result['failures'].append(dict(error='small-M gate executed no register-reuse projection'))
        result['status']='FAIL' if result['failures'] else 'PASS'
    finally:(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"KPACK_MOE_GATE status={result['status']} pair_formats={len(result['pairs'])}/6 chains={len(result['chains'])}/{len(cases)}",flush=True)
    return int(result['status']!='PASS')


if __name__=='__main__':sys.exit(main())
