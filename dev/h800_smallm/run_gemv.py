#!/usr/bin/env python3
"""CUDA event comparison: dense M=1..7 and indexed, one vector per expert.

All readers consume exactly the same source GGUF bytes and FP16 activations.
The supplied reader and Xplane retain FP16 weight reconstruction; larger
K-pack candidates use the previously admitted FP32 group-affine expression.
Timing includes one complete multi-row launch, never a CPU loop over rows.
"""
import argparse
import ast
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.h800_smallm.build_gemv import SHAPES

# Reuse the exact production ctypes Call definition without importing the
# unrelated torch/PPU gate helpers in its module on a CUDA-only host.
call_tree=ast.parse((ROOT/'quactlize/execution/native.py').read_text())
call_class=next(x for x in call_tree.body if isinstance(x,ast.ClassDef) and x.name=='Call')
exec(compile(ast.Module(body=[call_class],type_ignores=[]),'<production Call>','exec'))


def checked(rc,what):
    if rc: raise RuntimeError(f'{what}: CUDA status {rc}')


class CUDA:
    def __init__(self,root):
        self.lib=C.CDLL(str(root/'lib64/libcudart.so'))
        vp=C.c_void_p; pp=C.POINTER(vp); ip=C.POINTER(C.c_int)
        bindings={
            'Malloc':[pp,C.c_size_t], 'Free':[vp], 'Memcpy':[vp,vp,C.c_size_t,C.c_int],
            'Memset':[vp,C.c_int,C.c_size_t], 'DeviceSynchronize':[],
            'DeviceGetAttribute':[ip,C.c_int,C.c_int], 'StreamCreateWithFlags':[pp,C.c_uint],
            'StreamDestroy':[vp], 'StreamSynchronize':[vp],
            'StreamBeginCapture':[vp,C.c_int], 'StreamEndCapture':[vp,pp],
            'GraphInstantiate':[pp,vp,vp,vp,C.c_size_t], 'GraphLaunch':[vp,vp],
            'GraphExecDestroy':[vp], 'GraphDestroy':[vp],
            'EventCreate':[pp], 'EventDestroy':[vp], 'EventRecord':[vp,vp],
            'EventSynchronize':[vp], 'EventElapsedTime':[C.POINTER(C.c_float),vp,vp],
        }
        for name,args in bindings.items():
            fn=getattr(self.lib,'cuda'+name); fn.argtypes=args;fn.restype=C.c_int
            setattr(self,name,fn)
        self.pointers=[];self.stream=vp();checked(self.StreamCreateWithFlags(C.byref(self.stream),1),'stream')
        l2=C.c_int(); checked(self.DeviceGetAttribute(C.byref(l2),38,0),'L2')
        if l2.value<=0: raise ValueError('missing L2 capacity')
        self.l2=l2.value

    def alloc(self,n):
        ptr=C.c_void_p();checked(self.Malloc(C.byref(ptr),n),'malloc');self.pointers.append(ptr.value)
        return ptr.value

    def put(self,ptr,data):
        a=np.ascontiguousarray(data);checked(self.Memcpy(ptr,a.ctypes.data,a.nbytes,1),'upload')
        # Setup/negative fixtures use the default copy stream; the measured
        # graph is on a nonblocking stream. Close uploads explicitly here,
        # entirely outside event timing, rather than rely on implicit ordering.
        checked(self.StreamSynchronize(None),'upload complete')

    def get(self,ptr,n,dtype):
        a=np.empty(n,dtype=dtype);checked(self.Memcpy(a.ctypes.data,ptr,a.nbytes,2),'download');return a

    def close(self):
        checked(self.StreamSynchronize(self.stream),'final sync')
        for ptr in self.pointers: checked(self.Free(ptr),'free')
        self.pointers=[];checked(self.StreamDestroy(self.stream),'stream destroy')


class Replay:
    def __init__(self,gpu,launch,copies):
        self.gpu=gpu;self.graph=C.c_void_p();self.instance=C.c_void_p()
        self.begin=C.c_void_p();self.end=C.c_void_p()
        self.calls=max(1,math.ceil(32/copies))*copies
        checked(gpu.StreamBeginCapture(gpu.stream,0),'capture')
        for i in range(self.calls): launch(i%copies)
        checked(gpu.StreamEndCapture(gpu.stream,C.byref(self.graph)),'end capture')
        checked(gpu.GraphInstantiate(C.byref(self.instance),self.graph,None,None,0),'instantiate')
        checked(gpu.EventCreate(C.byref(self.begin)),'event');checked(gpu.EventCreate(C.byref(self.end)),'event')

    def sample(self):
        gpu=self.gpu
        checked(gpu.EventRecord(self.begin,gpu.stream),'event start')
        checked(gpu.GraphLaunch(self.instance,gpu.stream),'replay')
        checked(gpu.EventRecord(self.end,gpu.stream),'event end');checked(gpu.EventSynchronize(self.end),'event sync')
        ms=C.c_float();checked(gpu.EventElapsedTime(C.byref(ms),self.begin,self.end),'duration')
        us=ms.value*1000/self.calls
        if not math.isfinite(us) or us<=0: raise ValueError('invalid timing')
        return us

    def close(self):
        g=self.gpu
        checked(g.GraphExecDestroy(self.instance),'destroy instance');checked(g.GraphDestroy(self.graph),'destroy graph')
        checked(g.EventDestroy(self.begin),'destroy event');checked(g.EventDestroy(self.end),'destroy event')


def read_fixture(path,n,k):
    data=path.read_bytes();header=struct.Struct('<Q8i8iQ8Q');h=header.unpack_from(data)
    if h[0]!=0x3146584D5647514B or h[1:9]!=(1,12,n,k,1,1,1,0): raise ValueError('source fixture identity')
    position=header.size;chunks=[]
    for length in h[18:]: chunks.append(data[position:position+length]);position+=length
    if position!=len(data): raise ValueError('source fixture length')
    raw=np.frombuffer(chunks[0],dtype='u1').reshape(n,k//256,144)
    low=np.frombuffer(chunks[1],dtype='<u2').reshape(k//4,n)
    units=np.frombuffer(chunks[3],dtype='u1').reshape(k//256,n,16)
    weights=dequantize(raw.reshape(-1),GGMLQuantizationType.Q4_K).reshape(n,k).astype('f8')
    if not np.isfinite(weights).all(): raise ValueError('nonfinite GGUF weights')
    return raw,low,units,weights,hashlib.sha256(data).hexdigest()


def case(args,n,k,scope,m,mode,bundle,source,selected):
    raw,low,units,weights,fixture_sha=source
    gpu=CUDA(args.cuda);replays=[]
    indexed=scope!='dense'; rows=8 if indexed else m
    channels=1 if scope=='indexed-shared-a' else rows
    experts=args.experts if indexed else 1
    # Unsorted, nonadjacent GPU IDs; all unselected weights stay zero.
    ids=np.array(([255,2,137,0,89,4,201,6] if experts==256 else [13,2,11,0,9,4,15,6])
                 if indexed else [0]*rows,dtype='<i4')
    shifts=[int(e)*13 for e in ids] if indexed else [0]
    act=np.random.default_rng(np.random.SeedSequence([99173,n,k,rows,channels])).normal(0,.3,(channels,k)).astype('<f2')
    gold=np.stack([np.roll(weights,shifts[r] if indexed else 0,axis=0)@act[r%channels].astype('f8') for r in range(rows)])
    denom=np.stack([np.roll(np.abs(weights),shifts[r] if indexed else 0,axis=0)@np.abs(act[r%channels].astype('f8')) for r in range(rows)])
    # Padding exercises row/slot strides. The 16-byte input alignment belongs
    # to these vectorized reader contracts, not to every production endpoint.
    ah=np.full((channels,k+8),np.float16(-77),dtype='<f2');ah[:,:k]=act
    a=gpu.alloc(ah.nbytes);gpu.put(a,ah)
    idh=np.full(rows+3,-123,dtype='<i4');idh[:rows]=ids
    idptr=gpu.alloc(idh.nbytes);gpu.put(idptr,idh)
    out_count=rows*(n+8)+8;out=gpu.alloc(out_count*4)
    call=Call();call.version=1;call.size=C.sizeof(Call);call.qtype=12;call.n=n;call.k=k
    call.experts=experts;call.rows=rows;call.mode=2 if indexed else 0;call.input_type=0
    call.channels=channels if indexed else 1;call.topk=rows if indexed else 1
    call.a_row_stride=k+8;call.a_token_stride=channels*(k+8);call.ids_stride=rows+3
    call.out_row_stride=n+8;call.a=a;call.ids=idptr;call.output=out+16;call.stream=gpu.stream.value
    per_expert=n*k*9//16;active=rows if indexed else 1
    copies=1 if mode=='warm' else max(2,math.ceil(2.25*gpu.l2/(active*per_expert)))
    family=(selected or {}).get('_implementation',
        args.small_reader if n==512 else args.medium_reader if n==1024 else 'large')
    libraries={};graphs={};correctness={};records=[]
    try:
        xlib=C.CDLL(str(args.bundle/'xplane/kernel.so'),mode=C.RTLD_LOCAL)
        pack=xlib.q4_xplane_pack;pack.argtypes=[C.c_int,C.c_int]+[C.c_void_p]*3;pack.restype=C.c_int
        xp=np.empty(n*k//2,dtype='u1');xu=np.empty(units.size,dtype='u1')
        checked(pack(n,k,raw.ctypes.data,xp.ctypes.data,xu.ctypes.data),'Xplane pack/inverse')
        if not np.array_equal(xu,units.reshape(-1)): raise ValueError('control metadata differs')
        xp=xp.reshape(k//256,n,128)
        for arm,key in (('xplane','xplane'),('reference','reference'),('kpack',family)):
            lib=xlib if arm=='xplane' else C.CDLL(str(args.bundle/key/'kernel.so'),mode=C.RTLD_LOCAL)
            fn=lib.q4_smallm_run;fn.argtypes=[C.POINTER(Call),C.c_int,C.c_int,C.c_int];fn.restype=C.c_int
            libraries[arm]=lib
            recipes=bundle['arms'][key]['recipes']
            if arm=='kpack': recipes=recipes[f'{n}x{k}']
            if selected is not None: recipes=selected[arm]
            source_plane=raw if arm=='reference' else xp if arm=='xplane' else low
            axis=0 if arm=='reference' else 1
            stride=source_plane.nbytes*experts;ustride=units.nbytes*experts
            bp=gpu.alloc(stride*copies);up=gpu.alloc(ustride*copies)
            checked(gpu.Memset(bp,0,stride*copies),'zero unselected B');checked(gpu.Memset(up,0,ustride*copies),'zero unselected units')
            for copy in range(copies):
                for r,e in enumerate(ids if indexed else [0]):
                    gpu.put(bp+copy*stride+int(e)*source_plane.nbytes,np.roll(source_plane,shifts[r],axis=axis))
                    gpu.put(up+copy*ustride+int(e)*units.nbytes,np.roll(units,shifts[r],axis=1))
            def validate(positive=True,expected=gold):
                checked(gpu.StreamSynchronize(gpu.stream),'oracle sync')
                got=gpu.get(out,out_count,'<f4')
                view=got[4:-4].reshape(rows,n+8)
                if not (np.isnan(got[:4]).all() and np.isnan(got[-4:]).all() and np.isnan(view[:,n:]).all()):
                    raise ValueError('output guards overwritten')
                if not np.isfinite(view[:,:n]).all(): raise ValueError('nonfinite/unwritten output')
                error=float(np.max(np.abs(view[:,:n].astype('f8')-expected)/np.maximum(denom,1e-30)))
                if positive and error>=.005: raise ValueError(f'independent GGUF dot failed: {error}')
                return error
            for recipe in recipes:
                # Capture independent copies of the struct/closure: graph
                # creation bakes pointers, including each physical ring slice.
                current=Call.from_buffer_copy(call)
                def launch(copy,fn=fn,c=current,recipe=recipe,bp=bp,up=up,stride=stride,ustride=ustride):
                    c.low=bp+copy*stride;c.units=up+copy*ustride
                    checked(fn(C.byref(c),*recipe),'kernel launch')
                checked(gpu.Memset(out,255,out_count*4),'output poison');launch(0)
                error=validate()
                # Negative route probe is on GPU IDs, not on the host packing
                # order. A fake expert 0 decoder cannot pass nonadjacent IDs.
                negative=0.
                if indexed:
                    wrong=idh.copy();wrong[:rows]=np.roll(ids,1);gpu.put(idptr,wrong);launch(0)
                    negative=validate(False);gpu.put(idptr,idh)
                    if negative<=.005: raise ValueError('wrong-expert oracle failed to reject')
                # Zero-A confirms the launch consumes the caller's activation.
                gpu.put(a,np.zeros_like(ah));launch(0)
                checked(gpu.StreamSynchronize(gpu.stream),'zero-A sync')
                zero=gpu.get(out,out_count,'<f4')[4:-4].reshape(rows,n+8)[:,:n]
                if not np.isfinite(zero).all() or np.any(zero!=0): raise ValueError('zero-A control')
                gpu.put(a,ah);launch(0);error=max(error,validate())
                graph=Replay(gpu,launch,copies);replays.append(graph)
                # Reuse the *same captured graph* after changing input. This
                # rejects a fixture-specific/stale activation or cached result.
                changed=ah.copy();changed[:,:k]=-act;gpu.put(a,changed)
                graph.sample();error=max(error,validate(expected=-gold))
                gpu.put(a,ah)
                if indexed:
                    gpu.put(idptr,wrong);graph.sample()
                    if validate(False)<=.005: raise ValueError('captured graph ignored changed GPU IDs')
                    gpu.put(idptr,idh)
                graph.sample();error=max(error,validate())
                graphs[arm,tuple(recipe)]=graph;correctness[arm,tuple(recipe)]=(error,negative)
        # Rotate ordering across rounds; compare whole-call events in the same
        # process. No allocation, conversion, CPU route, or JIT is timed.
        choices=list(graphs)
        for turn in range(args.rounds):
            order=list(choices);np.random.default_rng(60013+turn).shuffle(order)
            for arm,recipe in order:
                graph=graphs[arm,recipe]
                for _ in range(5): graph.sample()
                samples=[graph.sample() for _ in range(args.samples)]
                err,neg=correctness[arm,recipe]
                err=max(err,validate())
                records.append(dict(arm=arm,recipe=list(recipe),round=turn,samples_us=samples,
                    median_us=statistics.median(samples),error=err,wrong_expert_error=neg))
        results={}
        for arm in libraries:
            candidates={recipe:statistics.median(r['median_us'] for r in records if r['arm']==arm and tuple(r['recipe'])==recipe)
                        for aa,recipe in graphs if aa==arm}
            best=min(candidates,key=candidates.get)
            results[arm]=dict(recipe=list(best),median_us=candidates[best])
        kp=results['kpack']['median_us']
        gaps={arm:100*(kp/results[arm]['median_us']-1) for arm in ('xplane','reference')}
        result=dict(scope=scope,M=m if not indexed else 1,rows=rows,experts=experts,channels=channels,n=n,k=k,
            kpack_implementation=family,
            weight_arithmetic='PER_WEIGHT_FP16' if family=='small' or family.startswith('rows') else 'FP32_GROUP_AFFINE',
            mode=mode,copies=copies,L2_bytes=gpu.l2,fixture_sha256=fixture_sha,ids=ids.tolist(),
            input='F16_RANDOM',output='F32_PADDED',oracle='OFFICIAL_GGUF_FP64_DOT',
            rounds=args.rounds,samples=args.samples,status='PASS',best=results,delta_pct=gaps,records=records,
            parity='WITHIN_5PCT' if max(gaps.values())<=5 else 'OPEN',
            launches_per_call=1,reducer=False,CPU_routing=False,
            metric_scope='RESIDENT_F16_A_F32_OUTPUT_NO_LLAMA_ADAPTERS')
        print('H800_SMALLM_RESULT '+json.dumps({x:y for x,y in result.items() if x!='records'}),flush=True)
        return result
    finally:
        for graph in replays: graph.close()
        gpu.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--scopes',default='dense,indexed-shared-a,indexed-independent-a')
    p.add_argument('--m',type=int,nargs='+',default=list(range(1,8)))
    p.add_argument('--shapes',default=','.join(f'{n}x{k}' for n,k in SHAPES))
    p.add_argument('--modes',default='warm,rotating');p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--samples',type=int,default=7);p.add_argument('--selection',type=Path)
    p.add_argument('--experts',type=int,choices=[16,256],default=16,
                   help='resident experts for indexed M1; eight nonadjacent experts execute')
    p.add_argument('--medium-reader',choices=['medium','affine2-medium','affine4-medium'],default='medium')
    p.add_argument('--small-reader',choices=['small','affine2-small','affine4-small','affine8-small','rows2-small','rows4-small','rows2w2-small','rows4w2-small','scalar-small'],default='small')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if any(m<1 or m>7 for m in a.m) or min(a.rounds,a.samples)<1: raise ValueError('bounded positive test arguments')
    bundle=json.loads((a.bundle/'manifest.json').read_text())
    for arm,row in bundle['arms'].items():
        if hashlib.sha256((a.bundle/arm/'kernel.so').read_bytes()).hexdigest()!=row['sha256']: raise ValueError('payload changed')
    choices=json.loads(a.selection.read_text()) if a.selection else None
    started=time.monotonic();complete=[];failures=[]
    for shape in a.shapes.split(','):
        n,k=map(int,shape.split('x'))
        if (n,k) not in SHAPES: raise ValueError('unsupported shape')
        source=read_fixture(a.fixtures/f'q12-n{n}-k{k}-e1-c1.bin',n,k)
        for scope in a.scopes.split(','):
            if scope not in ('dense','indexed-shared-a','indexed-independent-a'): raise ValueError('unknown scope')
            for m in (a.m if scope=='dense' else [1]):
                for mode in a.modes.split(','):
                    if mode not in ('warm','rotating'): raise ValueError('unknown mode')
                    key=f'{scope}-m{m}-{shape}-{mode}'
                    print('H800_SMALLM_PROGRESS '+key,flush=True)
                    try:
                        result=case(a,n,k,scope,m,mode,bundle,source,choices[key] if choices else None)
                        (a.output/(key+'.json')).write_text(json.dumps(result,indent=2)+'\n');complete.append(key)
                    except Exception as e:
                        import traceback
                        traceback.print_exc();failures.append(dict(key=key,error=str(e)))
                        print('H800_SMALLM_FAILURE '+json.dumps(failures[-1]),flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(complete=complete,failures=failures,seconds=time.monotonic()-started),indent=2)+'\n')
    print(f'H800_SMALLM_DONE passed={len(complete)} failed={len(failures)} seconds={time.monotonic()-started:.1f}',flush=True)
    return bool(failures)


if __name__=='__main__': raise SystemExit(main())
