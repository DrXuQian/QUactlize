#!/usr/bin/env python3
"""Arbitrary-M MoE helpers: exact device controls and separate-component Asys."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import shutil
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
from dev.moe_scaffold.models import model_plan,assignments
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def read(path):return json.loads(Path(path).read_text())


def selected_smi(a):
    if a.smi:return a.smi.resolve(strict=True)
    for path in (a.sdk/'ppu-smi/bin/ppu-smi',a.sdk/'bin/ppu-smi'):
        if path.is_file():return path.resolve()
    binary=shutil.which('ppu-smi')
    if binary:return Path(binary).resolve()
    raise ValueError('PPU-SMI not found; supply --smi or PPU_SMI')


def workload_plan(a):
    if a.cohort=='qwen-models':return model_plan(a.smoke,a.phase)
    cases=workloads(a.smoke)
    if a.phase!='all':cases=[replace(c,phase=a.phase) for c in cases]
    return cases,dict(cohort='mechanisms',cases=len(cases),scope='SYNTHETIC_BOUNDARIES_NOT_MODEL_COVERAGE')


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
    plan=read(a.output/'run.json');cases=plan['assignments'][a.worker]
    folder=a.output/('gpu-'+plan['devices'][a.worker]);folder.mkdir(exist_ok=True)
    size=plan['batch_size'];count=(len(cases)+size-1)//size;failures=0
    for i in range(count):
        selected=cases[i*size:(i+1)*size]
        batch=folder/f'batch-{i}';batch.mkdir(exist_ok=True);saved=batch/'result.json'
        if a.resume and saved.exists() and reusable(read(saved),selected,a.output):
            print(f'MOE_SCAFFOLD_REUSE batch={i+1}/{count} cases={len(selected)}',flush=True);continue
        attempt=batch/('attempt-'+str(time.time_ns()));attempt.mkdir()
        atomic_json(attempt/'job.json',dict(cases=selected,batch=i+1,batches=count,progress=str(folder/'progress.json')))
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
            '--output',str(a.output),'--batch',str(attempt)]
        profile=[str(a.asys),'profile','-t','hggc','--sample','none','--python-sampling','false',
            '--capture-range','hggcProfilerApi','--capture-range-end','merge','--kill','none','--wait','primary',
            '-f','false','-o',str(attempt/'trace'),*command]
        print(f'MOE_SCAFFOLD_BATCH batch={i+1}/{count} cases={len(selected)} log={attempt/"profile.log"}',flush=True)
        try:
            run(profile,attempt/'profile.log')
            run([str(a.asys),'export','-o',str(attempt/'trace.sqlite'),str(attempt/'trace.asysrep')],attempt/'export.log')
            capture=read(attempt/'capture.json')
            result=parse(capture,activities(attempt/'trace.sqlite',launch_shared=True),runtime_activities(attempt/'trace.sqlite'))
            verify(receipt)
            if capture['cases']!=selected:raise ValueError('captured helper cases differ from assigned batch')
            result.update(cases=capture['cases'],latency_provenance=capture['latency_provenance'],clock=capture['clock'],
                          trace=str(attempt/'trace.asysrep'),trace_sha256=sha(attempt/'trace.asysrep'))
            atomic_json(saved,result)
            print(f'MOE_SCAFFOLD_BATCH_PASS batch={i+1}/{count}',flush=True)
        except Exception as exc:
            failures+=1
            atomic_json(saved,dict(status='FAIL',error=str(exc),attempt=str(attempt),cases=selected))
            print('MOE_SCAFFOLD_FAIL '+str(exc),flush=True)
    return int(bool(failures))


def reusable(result,cases,output):
    if result.get('status')!='PASS':return False
    if result.get('cases')!=cases or result.get('latency_provenance',{}).get('build_sha256')!=sha(output/'build/receipt.json'):
        raise ValueError('saved helper batch does not match this run')
    return True


def main(a):
    if a.plan:
        cases,coverage=workload_plan(a)
        print(json.dumps(dict(cases=[asdict(c) for c in cases],count=len(cases),coverage=coverage,
            scope='HELPERS_ONLY_NO_GEMM_OR_MODEL',production_changed=False),indent=2));return 0
    if a.output is None:
        a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-scaffold.',
            dir=os.environ.get('RUN_ROOT','/workspace' if Path('/workspace').is_dir() else '/tmp')))
    a.output=a.output.resolve()
    if a.worker is not None:return worker(a)
    if a.batch is not None:
        from dev.moe_scaffold.bench import measure
        receipt=read(a.output/'build/receipt.json');verify(receipt)
        plan=read(a.output/'run.json');job=read(a.batch/'job.json')
        def progress(done,total,case):
            atomic_json(Path(job['progress']),dict(batch=job['batch'],batches=job['batches'],completed=done,total=total,
                current=dict(key=case.key,phase=case.phase,m=case.tokens) if case else None,updated_ns=time.time_ns()))
        data=measure(a.sdk,receipt,[Case(**c) for c in job['cases']],plan['rounds'],plan['samples'],
                     smi=plan['smi'],clock_mhz=plan['clock_mhz'],progress=progress)
        verify(receipt)
        data['latency_provenance']=dict(quactlize=receipt['quactlize'],build_sha256=sha(a.output/'build/receipt.json'),
                                        batch_sha256=sha(a.batch/'job.json'))
        atomic_json(a.batch/'capture.json',data)
        return 0
    cases,coverage=workload_plan(a)
    if not a.confirm_idle:raise ValueError('--confirm-idle requires free assigned GPUs')
    devices=[str(int(v)) for v in a.devices.split(',')]
    if len(set(devices))!=len(devices) or any(int(d)<0 for d in devices):raise ValueError('invalid devices')
    if len(devices)>sum(c.tokens>0 for c in cases):raise ValueError('more workers than nonempty test cases')
    a.devices=','.join(devices)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    a.smi=selected_smi(a)
    a.output.mkdir(parents=True,exist_ok=True)
    plan=dict(cases=[asdict(c) for c in cases],devices=devices,assignments=assignments(cases,len(devices)),coverage=coverage,
        rounds=a.rounds,samples=a.samples,asys=str(a.asys),asys_sha256=sha(a.asys),
        smi=str(a.smi),smi_sha256=sha(a.smi),clock_mhz=a.clock_mhz,sdk=str(a.sdk.resolve()),batch_size=a.batch_size)
    if a.resume:
        if read(a.output/'run.json')!=plan:raise ValueError('resume cases/devices/protocol differ')
        verify(read(a.output/'build/receipt.json'))
    else:
        if (a.output/'run.json').exists():raise ValueError('use --resume or a new output directory')
        print('MOE_SCAFFOLD_BUILD '+str(a.output),flush=True)
        build(a.sdk,a.output/'build')
        atomic_json(a.output/'run.json',plan)
    processes=[];start=time.monotonic()
    for i,device in enumerate(devices):
        folder=a.output/('gpu-'+device);folder.mkdir(exist_ok=True)
        log=(folder/('console-'+str(time.time_ns())+'.log')).open('w')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
            '--output',str(a.output),'--devices',a.devices,'--worker',str(i),'--asys',str(a.asys),
            '--rounds',str(a.rounds),'--samples',str(a.samples)]
        if a.resume:command.append('--resume')
        process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT)
        processes.append((process,log))
    print('MOE_SCAFFOLD_START '+json.dumps(dict(output=str(a.output),points=len(cases),devices=devices,
        cohort=coverage['cohort'],phases=coverage.get('phases'),batch_size=a.batch_size)),flush=True)
    while any(p.poll() is None for p,_ in processes):
        time.sleep(10)
        print('MOE_SCAFFOLD_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in processes),
            elapsed_minutes=(time.monotonic()-start)/60,logs=str(a.output/'gpu-*/console-*.log'),
            progress={p.parent.name:read(p) for p in sorted(a.output.glob('gpu-*/progress.json'))})),flush=True)
    codes=[p.returncode for p,_ in processes]
    for _,log in processes:log.close()
    results=[read(p) for p in sorted(a.output.glob('gpu-*/batch-*/result.json'))]
    measured=sum(len(r.get('cases',[])) for r in results if r['status']=='PASS')
    complete=not any(codes) and measured==len(cases) and all(r['status']=='PASS' for r in results)
    report=dict(status='PASS' if complete else 'INCOMPLETE',expected=len(cases),measured=measured,
        returncodes=codes,results=results,coverage=coverage,elapsed_seconds=time.monotonic()-start,
        baseline='SAME_MATH_UNFUSED_HELPERS_NOT_LLAMA_GEMM',production_changed=False,
        performance_admitted=False,scope='PREPARE_TRANSITION_FINISH_NOT_TC_DIRECTORY_OR_GEMM_OR_MODEL')
    atomic_json(a.output/'summary.json',report)
    with tarfile.open(a.output/'review.tgz','w:gz') as tar:
        paths=[a.output/'summary.json',a.output/'run.json',a.output/'build/receipt.json',
               *a.output.glob('gpu-*/batch-*/result.json'),*a.output.glob('gpu-*/*.log')]
        for result in results:
            if result['status']!='PASS' and result.get('attempt'):paths.extend(Path(result['attempt']).glob('*.log'))
        for path in paths:tar.add(path,arcname=str(path.relative_to(a.output)))
    print('MOE_SCAFFOLD_DONE '+json.dumps({k:v for k,v in report.items() if k not in ('results','coverage')}),flush=True)
    print('results='+str(a.output/'review.tgz'),flush=True)
    return 0 if complete else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    parser.add_argument('--output',type=Path);parser.add_argument('--devices',default='0')
    parser.add_argument('--cohort',choices=('mechanisms','qwen-models'),default='mechanisms')
    parser.add_argument('--phase',choices=('all','prepare','transition','finish'),default='all')
    parser.add_argument('--asys',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smi',type=Path,default=os.environ.get('PPU_SMI'))
    parser.add_argument('--clock-mhz',type=int,default=1700)
    parser.add_argument('--plan',action='store_true');parser.add_argument('--confirm-idle',action='store_true')
    parser.add_argument('--rounds',type=int,default=4);parser.add_argument('--samples',type=int,default=2)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--worker',type=int);parser.add_argument('--batch',type=Path)
    args=parser.parse_args()
    if args.rounds<2 or args.samples<1:parser.error('need >=2 alternating rounds and >=1 sample')
    if not 0<args.clock_mhz<10000:parser.error('--clock-mhz must be a positive integer below 10000')
    if not 1<=args.batch_size<=128:parser.error('--batch-size must be between 1 and 128')
    try:raise SystemExit(main(args))
    except Exception:traceback.print_exc();raise SystemExit(1)
