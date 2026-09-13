#!/usr/bin/env python3
"""Validate the production-selected Q4 call, without another candidate sweep."""
import argparse
import ctypes as C
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_ppu import decode_sweep as sweep
from dev.gemv_ppu.decode_bench import Base, error, truth, probe
from quactlize.dispatch.native import Dispatch, Request, Choice, IndexedIO
from quactlize.execution.native import Call as VecCall, Arrangement, Sizes
from quactlize.runtime.native import Call, SDK, checked
from quactlize.runtime.compiler import sha
from tools.run_kpack_grouped_decode_probe import Replay
from tools.verify_kpack_dispatch import verify


class Config(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[
        (name,C.c_int32) for name in ('reader','variant','warps','values','columns')]


def control_identity(bundle, production):
    """Check only the immutable SIMT controls we load, not unused TC payloads."""
    manifest=json.loads((bundle/'manifest.json').read_text())
    if sha(bundle/'manifest.json')!=production['decode_policy']['measured_manifest_sha256']:
        raise ValueError('immutable sweep manifest differs')
    if manifest['plan']['workloads']!=sweep.workloads():
        raise ValueError('immutable workload/recipe interpretation differs')
    for shape, recipes in manifest['simt_recipes'].items():
        n,k=map(int,shape.split('x'))
        if recipes!=[asdict(c)|{'key':c.key} for c in sweep.simt_inventory(n,k)]:
            raise ValueError('immutable SIMT index ordering differs: '+shape)
    selected={f'libq4_decode_n{w["n"]}_k{w["k"]}.so' for w in sweep.workloads()}
    for name in selected:
        if sha(bundle/name)!=manifest['payloads'][name]:
            raise ValueError('immutable SIMT payload differs: '+name)
    return {name:manifest['payloads'][name] for name in sorted(selected)}


def validate_result(record, workload, expected, device):
    if (record.get('status')!='PASS' or record.get('workload')!=workload or
        record.get('selected')!=expected or record.get('device')!=device or
        record.get('scope')!='PRODUCTION_F32_CALL_NOT_MODEL' or
        record.get('first_launch')!='EXCLUDED'):
        raise ValueError('selected result identity differs: '+workload['id'])
    values=record['samples_us']
    if (len(values)!=15 or any(not math.isfinite(v) or v<=0 for v in values) or
        not math.isfinite(record['error']) or not 0<=record['error']<.005 or
        record['median_us']!=statistics.median(values) or record['copies']<2 or
        record['calls_per_graph']%record['copies']):
        raise ValueError('selected result samples/proof differ: '+workload['id'])
    return record


def workloads(policy):
    """Every public M boundary, both A conventions and every token8 router."""
    ids=set()
    all_rows=sweep.workloads()
    for r in policy['ranges']:
        if r['role']!='auto':continue
        for w in all_rows:
            if (w['operator'],w['n'],w['k'])==(r['operator'],r['n'],r['k']) and w['tokens'] in (r['first'],r['last']):
                if w['operator']=='dense' or w['router']=='real' or w['tokens']==8:ids.add(w['id'])
    return [w for w in all_rows if w['id'] in ids]


class Selected:
    def __init__(self,args,base):
        self.b=base;self.r=base.case_r;self.handles=[];self.d=Dispatch(args.production)
        self.lib=C.CDLL(str(args.production/'libquactlize_ppu_execution.so'),mode=C.RTLD_LOCAL)
        types={'select':([C.POINTER(VecCall),C.POINTER(Arrangement),C.POINTER(Config),C.POINTER(Sizes)]),
               'run':([C.POINTER(VecCall),C.POINTER(Config),C.POINTER(Arrangement)]),
               'cast':([C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p]),
               'indexed_prepare':([C.POINTER(VecCall),C.c_void_p,C.c_void_p,C.c_void_p]),
               'indexed_finish':([C.POINTER(VecCall),C.c_void_p,C.c_void_p])}
        self.fn={}
        for name,params in types.items():
            f=getattr(self.lib,'quactlize_kpack_q4_decode_'+name+'_v1');f.argtypes=params;f.restype=C.c_int;self.fn[name]=f
        self.f=Config();s=Sizes()
        rc=self.fn['select'](C.byref(base.base),C.byref(base.arr),C.byref(self.f),C.byref(s))
        if rc not in (0,24):checked(rc,'production SIMT query')
        self.simt=rc==0
        if self.simt:
            f=self.f;self.key=f'simt:r{f.reader}-v{f.variant}-w{f.warps}-p{f.values}-c{f.columns}'
            return
        w=base.workload
        req=Request(1,C.sizeof(Request),12,2 if base.ids else 0,w['rows'],base.n,base.k,w['experts'],w['tokens'],base.arr.mapping_id)
        query=self.d.lib.quactlize_kpack_dispatch_query_decode_v1
        query.argtypes=[C.c_void_p,C.POINTER(Request),C.POINTER(Choice)];query.restype=C.c_int
        self.choice=Choice();checked(query(self.d.runtime,C.byref(req),C.byref(self.choice)),'production TC query')
        if self.choice.compute_units!=72 or self.choice.policy!=7:raise ValueError('production choice device/policy differs')
        policy=json.loads((args.production/'kpack_q4_decode_v1.json').read_text())
        expected=next(r for r in policy['replay'] if r['case']==w['id'])['recipe']
        if self.choice.parent.decode()!=expected.split(':')[1] or self.choice.split!=int(expected.split(':')[2][1:]):
            raise ValueError('production TC parent/split differs from replay')
        self.key=expected
        self.a=self.r.alloc(w['rows']*base.k*2);self.out=self.r.alloc(w['rows']*base.n*2)
        self.offsets=self.r.alloc(257*4) if base.ids else None
        self.rows=self.r.alloc(w['rows']*4) if base.ids else None
        self.workspace=self.r.alloc(self.choice.workspace_bytes) if self.choice.workspace_bytes else None
        self.fused=bool(base.ids and w['rows']<=32)
        self.io=IndexedIO(1,C.sizeof(IndexedIO),w['tokens'],8,w['channels'],0,
            base.base.ids_stride,base.base.a_row_stride,base.base.a_token_stride,base.base.out_row_stride,
            base.ids,base.a,base.out,self.rows) if self.fused else None
        for plane in [*base.planes,base.planes[0]|{'low':base.zero_low}]:
            c=Call(version=1,size=C.sizeof(Call),m=w['rows'],n=base.n,k=base.k,experts=w['experts'],
                group_size=32,device=self.choice.device,compute_units=72,mapping_id=base.arr.mapping_id,
                a=self.a,low=plane['low'],metadata=plane['units'],output=self.out,offsets_device=self.offsets,
                workspace=self.workspace,workspace_bytes=self.choice.workspace_bytes,stream=self.r.stream.value)
            self.handles.append(self.d.prepare(self.choice,c,indexed=self.io))

    def run(self,index=0,negative=False):
        b=self.b;c=VecCall.from_buffer_copy(b.base);plane=b.planes[index%b.copies]
        c.low=b.zero_low if negative else plane['low'];c.units=plane['units']
        if self.simt:return self.fn['run'](C.byref(c),C.byref(self.f),C.byref(b.arr))
        if not self.fused:
            rc=self.fn['indexed_prepare'](C.byref(c),self.a,self.offsets,self.rows) if b.ids else self.fn['cast'](
                1,b.a,self.a,b.workload['rows'],b.k,c.a_row_stride,c.stream)
            if rc:return rc
        rc=self.handles[-1 if negative else index%b.copies]()
        if rc or self.fused:return rc
        return self.fn['indexed_finish'](C.byref(c),self.out,self.rows) if b.ids else self.fn['cast'](
            0,self.out,b.out,b.workload['rows'],b.n,c.out_row_stride,c.stream)

    def close(self):
        self.b.sdk.synchronize(self.r.stream);self.d.close()


def child(args,workload):
    base=selected=graph=None
    try:
        base=Base(args,workload);selected=Selected(args,base)
        if base.device['name']!='PPU-ZW810':raise ValueError('measured policy device differs')
        base.poison();checked(selected.run(),'selected eager');got=base.read();err=error(got,base.data)
        if err>=.005:raise ValueError('independent GGUF error '+str(err))
        if selected.simt:
            base.poison();checked(base.invoke(selected.key[5:]),'immutable sweep control')
            if not np.array_equal(got.view('u4'),base.read().view('u4')):raise ValueError('production reader differs from measured image')
        base.poison();checked(selected.run(negative=True),'zero codes')
        if error(base.read(),base.data)<=.005:raise ValueError('zero-code oracle failed')
        index=0
        def run():
            nonlocal index
            rc=selected.run(index);index+=1;return rc
        graph=Replay(base.sdk,base.case_r.stream,run,base.calls_per_graph)
        checked(graph(),'graph first launch excluded');base.sdk.synchronize(base.case_r.stream)
        for factor in (0,-1,1):
            data=np.array(base.data['a']*factor,dtype='<f4')
            checked(base.sdk.lib.hggcMemcpy(base.a,data.ctypes.data,data.nbytes,1),'untimed changed A')
            base.poison();checked(graph(),'changed-A graph replay')
            target=base.data|dict(golden=base.data['golden']*factor)
            if error(base.read(),target)>=.005:raise ValueError('changed-A graph failed')
        if base.ids:
            ids=base.data['ids'].copy();ids[:,:8]=np.roll(ids[:,:8],1,axis=1)
            checked(base.sdk.lib.hggcMemcpy(base.ids,ids.ctypes.data,ids.nbytes,1),'untimed changed IDs')
            base.poison();checked(graph(),'changed-ID graph replay')
            if error(base.read(),truth(base.w,base.data,ids))>=.005:raise ValueError('changed-ID graph failed')
            checked(base.sdk.lib.hggcMemcpy(base.ids,base.data['ids'].ctypes.data,ids.nbytes,1),'restore IDs')
        checked(graph(),'restored graph warmup');base.sdk.synchronize(base.case_r.stream)
        samples=[x/base.calls_per_graph for x in base.case_r.samples(graph,15)]
        if error(base.read(),base.data)>=.005:raise ValueError('timed output differs')
        result=dict(status='PASS',workload=workload,selected=selected.key,error=err,samples_us=samples,
            median_us=statistics.median(samples),device=base.device,copies=base.copies,
            calls_per_graph=base.calls_per_graph,first_launch='EXCLUDED',scope='PRODUCTION_F32_CALL_NOT_MODEL')
        (args.output/(workload['id']+'.pending.json')).write_text(json.dumps(result,indent=2)+'\n')
        print('Q4_DECODE_SELECTED',json.dumps(result),flush=True)
    finally:
        if base:base.sdk.synchronize(base.case_r.stream)
        if graph:graph.close()
        if selected:selected.close()
        if base:base.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production',type=Path,default=ROOT/'prebuilt/ppu0010/q4-decode-policy-v1')
    p.add_argument('--bundle',type=Path,default=sweep.BUNDLE)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    production=verify(a.production,sdk=a.sdk)
    policy=json.loads((a.production/'kpack_q4_decode_v1.json').read_text())
    work=workloads(policy)
    if a.plan_only:print('Q4_DECODE_SELECTED_PLAN cases='+str(len(work)));return
    a.output.mkdir(parents=True,exist_ok=True)
    if a.probe_only:
        lib=C.CDLL(str(a.bundle/'libq4_decode_n512_k2048.so'),mode=C.RTLD_LOCAL)
        device=probe(SDK(a.sdk),lib,a.l2_bytes)
        if device['name']!='PPU-ZW810':raise ValueError('measured device differs')
        print('Q4_DECODE_SELECTED_DEVICE '+json.dumps(device));return
    if a.case:
        child(a,next(w for w in work if w['id']==a.case));return
    controls=control_identity(a.bundle,production)
    command=[sys.executable,'-u',__file__,'--production',str(a.production),'--bundle',str(a.bundle),
             '--sdk',str(a.sdk),'--output',str(a.output),'--l2-bytes',str(a.l2_bytes)]
    probe_run=subprocess.run(command+['--probe-only'],capture_output=True,text=True,check=True)
    rows=[line.split(' ',1)[1] for line in probe_run.stdout.splitlines() if line.startswith('Q4_DECODE_SELECTED_DEVICE ')]
    if len(rows)!=1:raise ValueError('missing/duplicate device identity')
    device=json.loads(rows[0]);expected={r['case']:r['recipe'] for r in policy['replay']}
    identity=dict(schema='quactlize.q4-decode-selected-gate.v1',production=sha(a.production/'manifest.json'),
                  runner=sha(__file__),workloads=work,visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  device=device,controls=controls,
                  runtime={x:sha(a.sdk/'lib'/x) for x in production['execution_receipt']['runtime']})
    target=a.output/'authority.json'
    if target.exists() and json.loads(target.read_text())!=identity:raise ValueError('resume authority differs')
    target.write_text(json.dumps(identity,indent=2)+'\n')
    start=time.monotonic();failed=[];executed=0
    for i,w in enumerate(work):
        target=a.output/(w['id']+'.json')
        if target.exists():
            validate_result(json.loads(target.read_text()),w,expected[w['id']],device)
            continue
        with (a.output/(w['id']+'.log')).open('w') as log:
            rc=subprocess.run(command+['--case',w['id']],stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:failed.append(w['id'])
        else:
            pending=a.output/(w['id']+'.pending.json')
            validate_result(json.loads(pending.read_text()),w,expected[w['id']],device)
            pending.replace(target)
        executed+=1;elapsed=time.monotonic()-start
        remaining=elapsed/executed*(len(work)-i-1)/60
        print(f'Q4_DECODE_SELECTED_PROGRESS completed={i+1}/{len(work)} failed={len(failed)} elapsed_s={elapsed:.1f} remaining_minutes={remaining:.1f} eta=OBSERVED_CASE_AVERAGE_NOT_GUARANTEE',flush=True)
    result=dict(status='FAIL' if failed else 'PASS',expected=len(work),failed=failed,seconds=time.monotonic()-start,
                files={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name not in ('result.json','console.log')})
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Q4_DECODE_SELECTED_COMPLETE',json.dumps(result),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise SystemExit(1)
