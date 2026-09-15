#!/usr/bin/env python3
"""Q8 baseline/vector gate: independent GGUF oracle, changed graphs, cold calls."""
import argparse
import ctypes as C
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.bf16_compute.fixture import round_compute
from dev.gemv_simt import fixture
from dev.gemv_simt.native import Runtime,Graph,checked,NativeConfig
from dev.gemv_simt.run import Bench as BaseBench
from dev.gemv_simt.q8_vector_access import pattern
from quactlize.execution.native import SimtCallV2,Call
from quactlize.execution.simt_codegen import Config,SPLITS
from tools.kpack_warmup_fixture import activation_values


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def l2_identity(identity, verified_bytes=0):
    """A positive runtime query wins; an explicit receipt fills missing SDK data."""
    actual=identity['l2_bytes']
    if verified_bytes<0 or (actual>0 and verified_bytes and actual!=verified_bytes):
        raise ValueError('verified L2 bytes conflict with the runtime query')
    return identity | dict(l2_bytes=actual or verified_bytes,
        l2_source='RUNTIME_QUERY' if actual>0 else 'OPERATOR_VERIFIED_BYTES' if verified_bytes else 'UNAVAILABLE')


class Library:
    def __init__(self,path,arm,compute):
        self.manifest=json.loads((path/'manifest.json').read_text())
        lib=path/self.manifest['library']
        if self.manifest['schema']!='quactlize.q8-vector.v1' or sha(lib)!=self.manifest['library_sha256']:
            raise ValueError('Q8 experiment image identity differs')
        self.arm,self.compute=arm,compute
        self.lib=C.CDLL(str(lib.resolve()),mode=C.RTLD_LOCAL)
        self.run=self.lib.q8_vector_run
        self.run.argtypes=[C.POINTER(SimtCallV2),C.POINTER(NativeConfig),C.c_int];self.run.restype=C.c_int

    def configs(self):
        return [Config(**{k:r[k] for k in Config.__dataclass_fields__}) for r in self.manifest['configs']]

    def prepare(self,call,config):
        d=SimtCallV2(call,self.compute);f=NativeConfig(config)
        return lambda:self.run(C.byref(d),C.byref(f),self.arm)

    def probe(self):
        fn=self.lib.simt_candidate_probe;fn.argtypes=[C.POINTER(C.c_int),C.c_char_p];fn.restype=C.c_int
        v=(C.c_int*6)();name=C.create_string_buffer(256)
        checked(fn(v,name),'same-image device probe')
        return dict(zip(('ordinal','sm','l2_bytes','warp','major','minor'),list(v)))|dict(name=name.value.decode())


class Bench(BaseBench):
    def __init__(self,rt,lib,w,tokens,mode,channels,compute,copies=1,weak=False):
        self.compute=compute
        super().__init__(rt,lib,w,tokens,mode,channels,1,copies)
        # Test the actual ABI minimum: two-byte-aligned FP16 scale bases.
        if weak:
            for c in self.calls:
                raw=np.ascontiguousarray(w.planes['units'] if mode else w.planes['units'][:1])
                ptr=rt.allocate(raw.nbytes+32);rt.fill(ptr,raw.nbytes+32)
                rt.copy(ptr+2,raw);c.units=ptr+2
        self.update(0)

    def update(self,repeat):
        self.data=fixture.inputs(self.w,self.tokens,self.mode,self.channels,repeat)
        d=self.data;acount=d['a'].shape[0]
        values=activation_values(np.arange(acount)+repeat*137).astype('f4')
        if repeat==4 and self.compute: values[0,0]=243383.484375
        rounded=round_compute(values,'bf16' if self.compute else 'f16')
        d['a'][:,:self.w.k]=values[:,self.w.categories[0]]
        if self.mode==2:
            owner=d['ids'][:,:8].reshape(-1);arows=np.arange(d['rows'])//8*self.channels+np.arange(d['rows'])%8%self.channels
        elif self.mode==1:
            owner=np.arange(d['rows'])*self.w.experts//d['rows'];arows=np.arange(d['rows'])
        else: owner=np.zeros(d['rows'],dtype=int);arows=np.arange(d['rows'])
        d['gold']=np.stack([rounded[arows[r]].astype('f8')@self.w.sums[e] for r,e in enumerate(owner)])
        d['denom']=np.stack([np.abs(rounded[arows[r]].astype('f8'))@self.w.abs_sums[e] for r,e in enumerate(owner)])
        self.rt.copy(self.a,d['a'])
        if self.ids:self.rt.copy(self.ids,d['ids'])
        self.rt.sync()

    def invalid_id_negative(self,config):
        if self.mode!=2:return
        ids=self.data['ids'].copy()
        try:
            for value in (-1,self.w.experts):
                bad=ids.copy();bad[0,0]=value;self.rt.copy(self.ids,bad)
                self.poison();checked(self.lib.prepare(self.call,config)(),'invalid ID call');self.rt.sync()
                raw=self.rt.download(self.output,self.output_bytes).view('<u4')
                body=raw[4:-4].reshape(self.data['rows'],self.w.n+8)
                if not np.isnan(body[0,:self.w.n].view('<f4')).all():raise ValueError('invalid expert was not rejected')
                if not (np.all(raw[:4]==0xa5a5a5a5) and np.all(raw[-4:]==0xa5a5a5a5) and np.all(body[:,self.w.n:]==0xa5a5a5a5)):
                    raise ValueError('invalid-ID output guard differs')
        finally:self.rt.copy(self.ids,ids)


def numeric(a,rt,libs):
    w=fixture.weights(8,256,512,16);records=[]
    shapes=[(m,mode,ch) for m in range(1,9) for mode,ch in ((0,1),(2,1),(2,8))]+[(7,1,1),(8,1,1)]
    configs=[replace(c,split=s) for c in libs[0].configs() for s in SPLITS]
    expected=len(shapes)*2*len(configs)*2
    started=time.monotonic()
    for compute in range(2):
        for tokens,mode,ch in shapes:
            b=Bench(rt,libs[0],w,tokens,mode,ch,compute)
            try:
                for lib in libs:
                    lib.compute=compute;b.lib=lib
                    for cfg in configs:
                        _,err=b.correctness(cfg)
                        records.append(dict(tokens=tokens,mode=mode,channels=ch,compute=compute,arm=lib.arm,config=cfg.key,error=err))
                for weak in (False,True):
                    if weak:
                        b.close();b=Bench(rt,libs[0],w,tokens,mode,ch,compute,weak=True)
                    for lib in libs:
                        lib.compute=compute;b.lib=lib
                        for cfg in (configs[0],configs[-1]):
                            graph=Graph(rt,[lib.prepare(b.call,cfg)])
                            try:
                                for repeat in (1,2,4):
                                    b.update(repeat);b.poison();graph.sample();b.output_check()
                            finally:graph.close()
                            if mode==2:
                                b.replay_and_negative(cfg)
                                b.invalid_id_negative(cfg)
                print(f'Q8_VECTOR_NUMERIC completed={len(records)}/{expected} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            finally:b.close()
    return dict(status='PASS',expected=expected,records=records,range_value=243383.484375,
                controls='output/workspace guards, wrong IDs/zero A/changed graphs, two-byte scale bases',
                scope='F32_STORAGE_F16_AND_BF16_COMPUTE_GGUF_FACTOR_DOT')


def performance(a,rt,libs,identity):
    mode=0 if a.channels==0 else 2;channels=max(1,a.channels)
    w=fixture.weights(8,a.n,a.k,1 if not mode else 16)
    ids=fixture.inputs(w,a.tokens,mode,channels)['ids']
    active=1 if not mode else len(np.unique(ids[:,:8]))
    useful=a.n*a.k*34//32*active
    l2=identity['l2_bytes']
    if l2<=0:raise ValueError('positive verified L2 size required')
    copies=math.ceil(2.25*l2/useful)
    b=Bench(rt,libs[0],w,a.tokens,mode,channels,a.compute,copies)
    traversals=max(1,math.ceil(32/copies));pool=[];results=[]
    started=time.monotonic()
    for lib in libs:
        lib.compute=a.compute
        for cfg in (replace(c,split=s) for c in lib.configs() for s in SPLITS): pool.append((lib,cfg))
    try:
        if a.config:
            selected=[(lib,cfg) for lib,cfg in pool if cfg.key==a.config and lib.arm==a.arm]
            if len(selected)!=1:raise ValueError('profile needs one exact arm/config')
            lib,cfg=selected[0];b.lib=lib;b.correctness(cfg)
            invoke=lib.prepare(b.call,cfg)
            for _ in range(5):checked(invoke(),'excluded warmup')
            rt.sync()
            prefix='cuda' if libs[0].manifest['platform']=='cuda' else 'hggc'
            start,stop=(getattr(rt.lib,prefix+n) for n in ('ProfilerStart','ProfilerStop'))
            start.argtypes=stop.argtypes=[];start.restype=stop.restype=C.c_int
            checked(start(),'profile start');checked(invoke(),'profile call');rt.sync();checked(stop(),'profile stop');b.output_check()
            return dict(status='PASS',config=cfg.key,arm=lib.arm,scope='PROFILE_COMPLETE_CALL_FORCED_CACHE_POLICY_CONTROLLED_BY_PROFILER')
        screen={0:[],1:[]}
        for lib,cfg in pool:
            b.lib=lib;_,err=b.correctness(cfg)
            graph=Graph(rt,[lib.prepare(c,cfg) for c in b.calls]*traversals)
            try:samples=[graph.sample() for _ in range(5)]
            finally:graph.close()
            screen[lib.arm].append((statistics.median(samples),cfg,samples,err))
        selected=[]
        for lib in libs:
            for _,cfg,_,_ in sorted(screen[lib.arm],key=lambda r:r[0])[:2]:
                selected.append((lib,cfg,Graph(rt,[lib.prepare(c,cfg) for c in b.calls]*traversals)))
        timing={(lib.arm,cfg.key):[] for lib,cfg,_ in selected}
        try:
            for round in range(6):
                for lib,cfg,g in (selected if round%2==0 else list(reversed(selected))):
                    timing[lib.arm,cfg.key].append([g.sample() for _ in range(15)])
        finally:
            for _,_,g in selected:g.close()
        for lib,cfg,_ in selected:
            rounds=timing[lib.arm,cfg.key]
            results.append(dict(arm=lib.arm,config=cfg.key,rounds=rounds,
                median_us=statistics.median(x for r in rounds for x in r),
                pattern=pattern(cfg,a.n,a.k,bases=dict(A=b.a%128,low=b.call.low%128,high=0,units=b.call.units%128))))
        best={arm:min((r for r in results if r['arm']==arm),key=lambda r:r['median_us']) for arm in (0,1)}
        return dict(status='PASS',shape=[a.tokens,a.n,a.k],channels=a.channels,compute=a.compute,
                    ring_copies=copies,l2_bytes=l2,active_experts=active,useful_bytes=useful,
                    screen=[dict(arm=arm,config=c.key,samples=s,error=e) for arm,rows in screen.items() for _,c,s,e in rows],
                    confirmed=results,best=best,delta_pct=(best[1]['median_us']/best[0]['median_us']-1)*100,
                    elapsed_s=time.monotonic()-started,scope='ROTATING_WEIGHTS_COMPLETE_CALL_WITH_REDUCER_NOT_TC_OR_MODEL')
    finally:b.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('numeric','perf','profile'),default='numeric')
    p.add_argument('--n',type=int,default=512);p.add_argument('--k',type=int,default=2048)
    p.add_argument('--tokens',type=int,choices=range(1,9),default=1);p.add_argument('--channels',type=int,choices=(0,1,8),default=0)
    p.add_argument('--compute',type=int,choices=(0,1),default=0);p.add_argument('--config');p.add_argument('--arm',type=int,choices=(0,1),default=1)
    p.add_argument('--l2-bytes',type=int,default=0,help='verified physical L2 bytes only when the SDK query returns zero')
    a=p.parse_args();libs=[Library(a.bundle,i,a.compute) for i in range(2)]
    rt=Runtime(a.sdk,libs[0].manifest['platform'])
    try:
        identity=l2_identity(libs[0].probe(),a.l2_bytes)
        prefix='cuda' if libs[0].manifest['platform']=='cuda' else 'hggc'
        pci_fn=getattr(rt.lib,prefix+'DeviceGetPCIBusId')
        pci_fn.argtypes=[C.c_char_p,C.c_int,C.c_int];pci_fn.restype=C.c_int
        pci=C.create_string_buffer(64);checked(pci_fn(pci,len(pci),identity['ordinal']),'PCI identity')
        identity['pci']=pci.value.decode()
        result=numeric(a,rt,libs) if a.phase=='numeric' else performance(a,rt,libs,identity)
        result.update(identity=identity,library_sha256=libs[0].manifest['library_sha256'],timing_idle_admission='EXTERNAL_LOAD_AUDIT_REQUIRED')
        harness=[Path(__file__),ROOT/'dev/gemv_simt/fixture.py',ROOT/'dev/gemv_simt/native.py',
                 ROOT/'dev/gemv_simt/run.py',ROOT/'dev/bf16_compute/fixture.py',ROOT/'quactlize/execution/native.py']
        result['harness_sha256']={str(f.relative_to(ROOT)):sha(f) for f in harness}
        result['runtime_sha256']=sha(a.sdk/('lib64/libcudart.so' if libs[0].manifest['platform']=='cuda' else 'lib/libhggc_wrapper.so'))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('Q8_VECTOR_RESULT',json.dumps({k:v for k,v in result.items() if k in ('status','shape','compute','delta_pct','elapsed_s','scope')}),flush=True)
    finally:rt.close()


if __name__=='__main__':main()
