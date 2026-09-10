#!/usr/bin/env python3
"""Bounded Q8 SIMT selection against the actual selected W8A16 parent.

Admission is conservative: SIMT reads F32/writes F32, while the TC incumbent
gets resident FP16 endpoints without its llama conversion kernels. A SIMT
win here needs no estimated adapter credit. No model policy is guessed.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import Call, Config, Sizes, arrangement, bind
from quactlize.runtime.native import SDK, Call as TC, checked
from tools.run_kpack_gemv_gate import Resources, CONFIGS
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_pack_gate import device_identity
from tools.run_q8_kpack2_gate import fixture
from tools.export_kpack_gemv_policy import export
from tools.verify_kpack_dispatch import verify

SHAPES=((512,2048),(2048,512),(2048,4096),(4096,2048),(8192,2048),(1024,5120))
SCOPE='SIMT_F32_ENDPOINTS_VS_SELECTED_TC_FP16_CORE_CONSERVATIVE'


class Graph:
    def __init__(self,sdk,r,fn):
        self.sdk,self.r=sdk,r
        self.graph,self.instance=C.c_void_p(),C.c_void_p()
        checked(fn(),'eager before capture'); sdk.synchronize(r.stream)
        checked(sdk.lib.hggcStreamBeginCapture(r.stream,0),'begin capture')
        for _ in range(32): checked(fn(),'capture selected call')
        checked(sdk.lib.hggcStreamEndCapture(r.stream,C.byref(self.graph)),'end capture')
        checked(sdk.lib.hggcGraphInstantiateWithFlags(C.byref(self.instance),self.graph,0),'instantiate')
        for _ in range(5): checked(self.run(),'excluded graph warmup')
        sdk.synchronize(r.stream)

    def run(self): return self.sdk.lib.hggcGraphLaunch(self.instance,self.r.stream)

    def samples(self,count): return [x/32 for x in self.r.samples(self.run,count)]

    def close(self):
        self.sdk.synchronize(self.r.stream)
        checked(self.sdk.lib.hggcGraphExecDestroy(self.instance),'destroy graph instance')
        checked(self.sdk.lib.hggcGraphDestroy(self.graph),'destroy graph')


def dot_error(got,gold,denom):
    if not np.isfinite(got).all(): raise ValueError('Q8 nonfinite output')
    value=float(np.max(np.abs(got.astype('f8')-gold)/np.maximum(denom,1e-30)))
    if value>=.005: raise ValueError(f'Q8 independent dot error: {value}')
    return value


def run(args,sdk,lib,n,k,m):
    _,low,units,weight=fixture(n,k,1)
    a=np.random.default_rng(908+m).uniform(-.3,.3,(m,k)).astype('f4')
    ah=a.astype('<f2'); oracle=ah.astype('f8')@weight[0].astype('f8').T
    denom=np.abs(ah.astype('f8'))@np.abs(weight[0].astype('f8')).T
    dispatch=Dispatch(args.bundle,jit=dict(python=sys.executable,helper=ROOT/'tools/kpack_jit.py',sdk=args.sdk,cache=args.jit_cache))
    r=Resources(sdk); graphs=[]
    try:
        b,d,ap,hp=[r.upload(x) for x in (low,units,a,ah)]
        sdk.synchronize(None)
        out=r.alloc(m*n*4+32); hout=r.alloc(m*n*2)
        r.fill(hout,0xFF,m*n*2)
        arr=arrangement(8); choice=dispatch.query(8,1,m,n,k,1,m,arr.mapping_id)
        if choice is None: raise ValueError('Q8 selected TC parent missing')
        tc=TC(1,C.sizeof(TC),m,n,k,1,32,choice.device,choice.compute_units,arr.mapping_id,
            hp,b,None,d,None,hout,None,None,None,r.alloc(max(16,choice.workspace_bytes)),choice.workspace_bytes,r.stream.value)
        handle=dispatch.prepare(choice,tc)
        incumbent=Graph(sdk,r,handle); graphs.append(incumbent)
        dot_error(np.frombuffer(sdk.download(hout,m*n*2),dtype='<f2').reshape(m,n),oracle,denom)
        query,launch,_=bind(lib); calls=[]; errors=[]; timings={}; candidates={}
        for columns,warps,split in CONFIGS:
            cfg=Config(columns,warps,split)
            c=Call(version=1,size=C.sizeof(Call),qtype=8,n=n,k=k,experts=1,rows=m,
                mode=0,input_type=1,channels=1,topk=1,a_row_stride=k,out_row_stride=n,
                a=ap,low=b,units=d,output=out+16,stream=r.stream.value)
            sizes=Sizes(); checked(query(C.byref(c),C.byref(cfg),C.byref(arr),C.byref(sizes)),'SIMT query')
            if sizes.sf_plane_bytes or sizes.units_bytes!=units.nbytes: raise ValueError('Q8 scale contract')
            workspace=r.alloc(sizes.workspace_bytes+32)
            c.workspace=workspace+16 if sizes.workspace_bytes else None; c.workspace_bytes=sizes.workspace_bytes
            r.fill(workspace,0xFF,sizes.workspace_bytes+32); r.fill(out,0xFF,m*n*4+32)
            fn=lambda c=c,cfg=cfg: launch(C.byref(c),C.byref(cfg),C.byref(arr))
            graph=Graph(sdk,r,fn); graphs.append(graph)
            image=sdk.download(out,m*n*4+32)
            if image[:16]!=b'\xff'*16 or image[-16:]!=b'\xff'*16: raise ValueError('Q8 output guard')
            ws=sdk.download(workspace,sizes.workspace_bytes+32)
            if ws[:16]!=b'\xff'*16 or ws[-16:]!=b'\xff'*16: raise ValueError('Q8 workspace guard')
            errors.append(dot_error(np.frombuffer(image[16:-16],dtype='<f4').reshape(m,n),oracle,denom))
            r.fill(b,128,low.nbytes); checked(graph.run(),'zero-code negative'); sdk.synchronize(r.stream)
            planted=np.frombuffer(sdk.download(out+16,m*n*4),dtype='<f4').reshape(m,n)
            if np.any(planted!=0): raise ValueError('Q8 zero-code output is not zero')
            try: dot_error(planted,oracle,denom)
            except ValueError: pass
            else: raise ValueError('Q8 missing-code negative insensitive')
            checked(sdk.lib.hggcMemcpy(b,low.ctypes.data,low.nbytes,1),'restore codes'); sdk.synchronize(None)
            name=f'{columns}-{warps}-{split}'; candidates[name]=graph; timings[name]=[]
            calls.append((c,cfg))
        tc_times=[]
        for round_ in range(3):
            names=list(candidates)
            if round_%2: names.reverse()
            order=['TC',*names] if round_%2==0 else [*names,'TC']
            for name in order:
                graph=incumbent if name=='TC' else candidates[name]
                for _ in range(3): checked(graph.run(),'excluded switch warmup')
                samples=graph.samples(args.samples)
                (tc_times if name=='TC' else timings[name]).append(samples)
            print(f'Q8_SIMT_PROGRESS shape={m}x{n}x{k} round={round_+1}/3',flush=True)
        med={name:statistics.median(x for r_ in rounds for x in r_) for name,rounds in timings.items()}
        winner=min(med,key=lambda name:(med[name],name)); baseline=statistics.median(x for r_ in tc_times for x in r_)
        result=dict(q=8,n=n,k=k,experts=1,input_type=1,case=dict(mode=0,rows=m,channels=1,topk=1),
            correctness='PASS',zero_low_negative='DETECTED_FINITE',errors=errors,
            gemv_samples_us=timings,gemm_samples_us=tc_times,winner=winner,gemv_median_us=med[winner],
            gemm_median_us=baseline,incumbent_selection=receipt(choice),scope=SCOPE)
        print('Q8_SIMT_RESULT '+json.dumps(dict(n=n,k=k,m=m,winner=winner,simt_us=med[winner],
            tc_core_us=baseline,selected='SIMT' if med[winner]<=baseline else 'RETAIN_TC',scope=SCOPE)),flush=True)
        return result
    finally:
        for graph in reversed(graphs): graph.close()
        dispatch.close()
        r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','jit-cache','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,default=11)
    args=p.parse_args()
    if args.samples<3: p.error('at least three timing samples')
    verify(args.bundle); args.output.mkdir(parents=True,exist_ok=False)
    sdk=SDK(args.sdk); graph_bind(sdk)
    image=args.bundle/'libquactlize_ppu_execution.so'; lib=C.CDLL(str(image.resolve()),mode=C.RTLD_LOCAL)
    summary=dict(status='INCOMPLETE',device=device_identity(sdk),results=[],failures=[],
        plan=[dict(q=8,n=n,k=k,e=1,cases=[dict(mode=0,rows=m,channels=1,topk=1) for m in (1,4)]) for n,k in SHAPES],
        execution_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        native_manifest_sha256=hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest())
    try:
        for n,k in SHAPES:
            records=[]
            for m in (1,4):
                try: records.append(run(args,sdk,lib,n,k,m))
                except Exception as error:
                    traceback.print_exc(); summary['failures'].append(dict(n=n,k=k,m=m,error=str(error)))
            summary['results'].append(dict(records=records))
        summary['status']='FAIL' if summary['failures'] else 'PASS'
        if summary['status']=='PASS':
            policy,receipts=export(summary)
            (args.output/'gemv-policy.tsv').write_text(policy)
            (args.output/'decisions.json').write_text(json.dumps(receipts,indent=2)+'\n')
    finally:
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Q8_SIMT_GATE '+summary['status'],flush=True)
    return int(summary['status']!='PASS')


if __name__=='__main__': sys.exit(main())
