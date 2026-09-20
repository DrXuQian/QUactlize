#!/usr/bin/env python3
"""Production recipe entry versus the frozen complete-chain incumbent."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.tp2_chain.bench import Bench, Buffer
from dev.tp2_chain.plan import POINTS, ROUNDS, SAMPLES
from dev.tp2_chain.run import numerical, summarize, source_receipt, save
from dev.gemv_simt.native import Runtime, Graph, checked
from dev.gemv_simt.q8_vector_run import l2_identity
from quactlize.fusion.native import Library, MappedCall, FusionCall, Layout, Sizes, Request, Recipe, recipe_entries, integration_entries
from quactlize.runtime.compiler import sha, LIBRARIES
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify


class ProductionBench(Bench):
    def __init__(self,rt,bundle,manifest,point,l2):
        super().__init__(rt,bundle,manifest,point,l2,candidates=False)
        self.entries=recipe_entries(self.paired)
        self.control=Library(bundle/manifest['paired_gate_up']['control']['path'])
        self.paired_run=integration_entries(self.control)['run']
        self.selected=Recipe()
        self.recipe_workspace=None

        def invoke(unused,mapped,layout,index):
            d=MappedCall.from_buffer_copy(C.cast(mapped,C.POINTER(MappedCall)).contents)
            if self.recipe_workspace:
                d.call.input.call.workspace=self.recipe_workspace.ptr
                d.call.input.call.workspace_bytes=self.recipe_workspace.size
            return self.entries['run_v3'](C.byref(d),C.byref(self.selected),layout)
        self.candidate_run=invoke

    def configure(self,tokens,router=True,alias=True):
        super().configure(tokens,router,alias)
        p=self.point
        request=Request(qtype=12,n=p.n,k=p.k,experts=256,tokens=tokens,mode=2,topk=8,channels=1,
            input_type=1,compute_type=1,output_type=1,round_projection=1,row_order=0,
            consumer=1,consumer_qtype=13,consumer_n=p.k)
        status=self.entries['select_v2'](C.byref(request),C.byref(self.selected))
        self.automatic=status==0
        if status==24 and tokens>1 and p.k==3072:
            # Explicit capability check, not a promotion of unmeasured M2..8.
            request.tokens=1
            status=self.entries['select_v2'](C.byref(request),C.byref(self.selected))
        checked(status,'production paired recipe selection')
        c=type(self.calls[0]).from_buffer_copy(self.calls[0])
        c.n=p.n;c.out_row_stride=p.n
        mapped=MappedCall(FusionCall(c,1,1,1))
        size=Sizes()
        checked(self.entries['query_v2'](C.byref(mapped),C.byref(self.selected),C.byref(self.layout),C.byref(size)),
                'production paired workspace')
        self.recipe_workspace=Buffer(self.rt,size.workspace_bytes) if size.workspace_bytes else None
        if self.recipe_workspace:self.scratch.append(self.recipe_workspace)


def measure(graphs,point):
    # Graph.sample() already returns microseconds per complete captured call.
    # Never divide again by repetitions or weight copies.
    samples={a:[] for a in graphs}
    for round_index in range(ROUNDS):
        order=list(graphs) if round_index%2==0 else list(reversed(graphs))
        for arm in order:
            samples[arm].append([graphs[arm].sample() for _ in range(SAMPLES)])
        print(f'PAIRED_PRODUCTION_TIMING point={point} round={round_index+1}/{ROUNDS}',flush=True)
    return {a:summarize(v) for a,v in samples.items()}


def child(args):
    manifest=verify(args.bundle,sdk=args.sdk)
    if manifest.get('paired_gate_up',{}).get('schema')!='quactlize.paired-model.v2':
        raise ValueError('complete paired recipe package required')
    point=next(p for p in POINTS if p.name==args.point)
    result=dict(status='FAIL',scope='PRODUCTION_ENTRY_COMPLETE_CHAIN_NOT_MODEL_TPOT',point=point.name,
        manifest_sha256=sha(args.bundle/'manifest.json'),numerics={},summary={})
    rt=None;graphs={}
    try:
        rt=Runtime(args.sdk,'ppu')
        l2=l2_identity(dict(l2_bytes=rt.attribute(38)),args.l2_bytes)
        if l2['l2_bytes']<=0:raise ValueError('positive verified L2 capacity required')
        result.update(device=device_identity(rt),l2=l2,operator_idle_required=True)
        bench=ProductionBench(rt,args.bundle,manifest,point,l2['l2_bytes'])
        result['fixture']=bench.record
        incumbent='unfused' if point.k==3072 else 'paired-shipping'
        for arm in (incumbent,'paired-1'):
            result['numerics'][arm]=numerical(bench,arm)
            print(f'PAIRED_PRODUCTION_NUMERIC point={point.name} arm={arm} controls=30 status=PASS',flush=True)
        bench.configure(1);bench.update(0)
        result['selected']=dict(recipe_id=hex(bench.selected.recipe_id),catalog_id=hex(bench.selected.catalog_id),
            reader=bench.selected.reader,warps=bench.selected.config.warps,values=bench.selected.values,
            numerical_tokens=list(range(1,9)),automatic_policy_tokens=[1] if point.k==3072 else list(range(1,9)))
        loops=max(1,(32+bench.copies-1)//bench.copies)
        for arm in (incumbent,'paired-1'):
            graphs[arm]=Graph(rt,[bench.launch(arm,i) for _ in range(loops) for i in range(bench.copies)])
            graphs[arm].sample();bench.check()
        result['captured_complete_calls']=loops*bench.copies
        result['summary']=measure(graphs,point.name)
        baseline=result['summary'][incumbent]['median_us'];selected=result['summary']['paired-1']['median_us']
        result.update(status='PASS',incumbent=incumbent,delta_pct=100*(selected/baseline-1),
            admission='PENDING_DEVICE_RESULT_REVIEW',first_use_excluded=True)
        print('PAIRED_PRODUCTION_RESULT '+json.dumps(dict(point=point.name,baseline_us=baseline,
            selected_us=selected,delta_pct=result['delta_pct'],admission=result['admission'])),flush=True)
        return 0
    except Exception as error:
        result['error']=str(error);traceback.print_exc();return 1
    finally:
        try:
            for graph in graphs.values():graph.close()
            if rt:rt.close()
        except Exception as error:
            result.update(status='FAIL',cleanup_error=str(error))
            raise
        finally:save(args.output,result)


def worker(args,point,device,identity):
    dest=args.output/(point.name+'.json')
    receipt=args.output/(point.name+'.identity.json')
    if receipt.exists():
        if not args.resume or json.loads(receipt.read_text())!=identity:raise ValueError('production resume identity differs')
        if dest.exists() and json.loads(dest.read_text()).get('status')=='PASS':return json.loads(dest.read_text())
    else:save(receipt,identity)
    log=args.output/(point.name+'.'+str(time.time_ns())+'.log')
    command=[sys.executable,'-u',Path(__file__).resolve(),'--sdk',args.sdk,'--bundle',args.bundle,
             '--point',point.name,'--output',dest,'--l2-bytes',str(args.l2_bytes)]
    start=time.monotonic()
    with log.open('x') as stream:
        process=subprocess.Popen(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES=device))
        try:
            while process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'PAIRED_PRODUCTION_WAIT point={point.name} device={device} seconds={time.monotonic()-start:.0f} log={log}',flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    result=json.loads(dest.read_text()) if dest.exists() else dict(status='FAIL',error='no child receipt')
    if process.returncode:result.update(status='FAIL',process_rc=process.returncode)
    return result


def main(args):
    if args.point:return child(args)
    verify(args.bundle,sdk=args.sdk)
    devices=args.devices.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or not all(x.isdigit() for x in devices):
        raise ValueError('two different device ordinals required')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    physical=[]
    for device in devices:
        command=[sys.executable,ROOT/'dev/tp2_chain/run.py','--sdk',args.sdk,'--probe']
        text=subprocess.check_output(list(map(str,command)),env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),text=True)
        values=[json.loads(l.split(' ',1)[1]) for l in text.splitlines() if l.startswith('PAIRED_CHAIN_DEVICE ')]
        if len(values)!=1:raise ValueError('missing physical device identity')
        physical.append(values[0])
    if physical[0]['pci']==physical[1]['pci']:raise ValueError('workers resolve to the same device')
    identity=dict(manifest=sha(args.bundle/'manifest.json'),l2=args.l2_bytes,
        harness=source_receipt()|{'dev/tp2_chain/production.py':sha(Path(__file__).resolve())},
        runtime={f'lib{x}.so':sha(args.sdk/'lib'/f'lib{x}.so') for x in LIBRARIES})
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,args,p,d,identity|dict(device=i)) for p,d,i in zip(POINTS,devices,physical)]
        for p,future in zip(POINTS,futures):
            try:records.append(future.result())
            except Exception as error:records.append(dict(status='FAIL',point=p.name,error=str(error)));traceback.print_exc()
    complete=all(r['status']=='PASS' for r in records)
    save(args.output/'summary.json',dict(status='PASS' if complete else 'INCOMPLETE',records=records,
        manifest_sha256=identity['manifest'],device_admission='PENDING_REVIEW'))
    print(f'PAIRED_PRODUCTION_DONE complete={int(complete)} results={args.output}',flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--point',choices=[p.name for p in POINTS]);parser.add_argument('--devices',default='6,7')
    parser.add_argument('--l2-bytes',type=int,default=67108864);parser.add_argument('--resume',action='store_true')
    raise SystemExit(main(parser.parse_args()))
