#!/usr/bin/env python3
"""Arbitrary-M MoE helpers: exact device controls and separate-component Asys."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.moe_scaffold.build import build,verify
from dev.moe_scaffold.protocol import Case,workloads,parse
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def read(path):return json.loads(Path(path).read_text())


def run(command,log):
    atomic_json(log.with_suffix('.command.json'),command)
    with log.open('w') as out:
        rc=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT).returncode
    if rc:
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-10000));tail=stream.read().decode(errors='replace')
        raise RuntimeError(f'rc={rc} log={log}\n{tail}')


def worker(a):
    from dev.prefill_asys.trace import activities,runtime_activities
    receipt=read(a.output/'build/receipt.json');verify(receipt)
    folder=a.output/('gpu-'+a.devices.split(',')[a.worker]);folder.mkdir(exist_ok=True)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
        '--output',str(a.output),'--devices',a.devices,'--child',str(a.worker),
        '--rounds',str(a.rounds),'--samples',str(a.samples)]
    profile=[str(a.asys),'profile','-t','hggc','--sample','none','--python-sampling','false',
        '--capture-range','hggcProfilerApi','--capture-range-end','merge','--kill','none','--wait','primary',
        '-f','false','-o',str(folder/'trace'),*command]
    try:
        run(profile,folder/'profile.log')
        run([str(a.asys),'export','-o',str(folder/'trace.sqlite'),str(folder/'trace.asysrep')],folder/'export.log')
        capture=read(folder/'capture.json')
        result=parse(capture,activities(folder/'trace.sqlite',launch_shared=True),runtime_activities(folder/'trace.sqlite'))
        verify(receipt)
        result.update(cases=capture['cases'],latency_provenance=capture['latency_provenance'],
                      trace=str(folder/'trace.asysrep'),trace_sha256=sha(folder/'trace.asysrep'))
        atomic_json(folder/'result.json',result)
        return 0
    except Exception as exc:
        atomic_json(folder/'result.json',dict(status='FAIL',error=str(exc)))
        print('MOE_SCAFFOLD_FAIL '+str(exc),flush=True)
        return 1


def main(a):
    cases=workloads(a.smoke)
    if a.plan:
        print(json.dumps(dict(cases=[asdict(c) for c in cases],count=len(cases),
            scope='HELPERS_ONLY_NO_GEMM_OR_MODEL',production_changed=False),indent=2));return 0
    if a.output is None:
        a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-scaffold.',
            dir=os.environ.get('RUN_ROOT','/workspace' if Path('/workspace').is_dir() else '/tmp')))
    a.output=a.output.resolve()
    if a.worker is not None:return worker(a)
    if a.child is not None:
        from dev.moe_scaffold.bench import measure
        receipt=read(a.output/'build/receipt.json');verify(receipt)
        plan=read(a.output/'run.json');selected=plan['cases'][a.child::len(plan['devices'])]
        data=measure(a.sdk,receipt,[Case(**c) for c in selected],a.rounds,a.samples)
        verify(receipt)
        data['latency_provenance']=dict(quactlize=receipt['quactlize'],build_sha256=sha(a.output/'build/receipt.json'))
        atomic_json(a.output/('gpu-'+plan['devices'][a.child])/'capture.json',data)
        return 0
    if not a.confirm_idle:raise ValueError('--confirm-idle requires free assigned GPUs')
    devices=[str(int(v)) for v in a.devices.split(',')]
    if len(set(devices))!=len(devices) or any(int(d)<0 for d in devices):raise ValueError('invalid devices')
    if len(devices)>len(cases)-1:raise ValueError('more workers than nonempty test cases')
    a.devices=','.join(devices)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run.json').exists():raise ValueError('use a new output directory; previous results are preserved')
    print('MOE_SCAFFOLD_BUILD '+str(a.output),flush=True)
    build(a.sdk,a.output/'build')
    atomic_json(a.output/'run.json',dict(cases=[asdict(c) for c in cases],devices=devices,
        rounds=a.rounds,samples=a.samples,asys=str(a.asys),asys_sha256=sha(a.asys)))
    processes=[];start=time.monotonic()
    for i,device in enumerate(devices):
        folder=a.output/('gpu-'+device);folder.mkdir(exist_ok=True)
        log=(folder/'console.log').open('w')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
            '--output',str(a.output),'--devices',a.devices,'--worker',str(i),'--asys',str(a.asys),
            '--rounds',str(a.rounds),'--samples',str(a.samples)]
        process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
        processes.append((process,log))
    print('MOE_SCAFFOLD_START '+json.dumps(dict(output=str(a.output),points=len(cases),devices=devices)),flush=True)
    while any(p.poll() is None for p,_ in processes):
        time.sleep(10)
        print('MOE_SCAFFOLD_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in processes),
            elapsed_minutes=(time.monotonic()-start)/60,logs=str(a.output/'gpu-*/console.log'))),flush=True)
    codes=[p.returncode for p,_ in processes]
    for _,log in processes:log.close()
    results=[]
    for device in devices:
        path=a.output/('gpu-'+device)/'result.json'
        results.append(read(path) if path.is_file() else dict(status='FAIL',error='worker produced no result',device=device))
    measured=sum(len(r.get('cases',[])) for r in results)
    complete=not any(codes) and measured==len(cases) and all(r['status']=='PASS' for r in results)
    report=dict(status='PASS' if complete else 'INCOMPLETE',expected=len(cases),measured=measured,
        returncodes=codes,results=results,elapsed_seconds=time.monotonic()-start,
        baseline='SAME_MATH_UNFUSED_HELPERS_NOT_LLAMA_GEMM',production_changed=False,
        performance_admitted=False,scope='PREPARE_TRANSITION_FINISH_NOT_TC_DIRECTORY_OR_GEMM_OR_MODEL')
    atomic_json(a.output/'summary.json',report)
    with tarfile.open(a.output/'review.tgz','w:gz') as tar:
        paths=[a.output/'summary.json',a.output/'run.json',a.output/'build/receipt.json',
               *a.output.glob('gpu-*/result.json'),*a.output.glob('gpu-*/*.log')]
        for path in paths:tar.add(path,arcname=str(path.relative_to(a.output)))
    print('MOE_SCAFFOLD_DONE '+json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    print('results='+str(a.output/'review.tgz'),flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    parser.add_argument('--output',type=Path);parser.add_argument('--devices',default='0')
    parser.add_argument('--asys',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--plan',action='store_true');parser.add_argument('--confirm-idle',action='store_true')
    parser.add_argument('--rounds',type=int,default=4);parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--worker',type=int);parser.add_argument('--child',type=int)
    args=parser.parse_args()
    if args.rounds<2 or args.samples<1:parser.error('need >=2 alternating rounds and >=1 sample')
    try:raise SystemExit(main(args))
    except Exception:traceback.print_exc();raise SystemExit(1)
