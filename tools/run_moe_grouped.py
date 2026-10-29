#!/usr/bin/env python3
"""Bounded grouped TC binding/real-helper-chain gate, with small review output."""
import argparse
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
from dev.moe_grouped.build import build,verify
from dev.moe_route.build import revision
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_moe_scaffold import selected_smi,run


def read(p):return json.loads(Path(p).read_text())


def plan(smoke):
    ms=(1,8,9,16,128,5120) if smoke else (*range(1,10),16,32,64,128,256,512,1024,2048,4096,5120)
    rows=[]
    for k in (2048,3072):
        for m in ms:
            for profile in ('spread','cluster'):
                row=dict(k=k,m=m,profile=profile,gate_split=1 if m%2 else 4,down_split=1 if m%2 else 2)
                row['id']=digest(row)[:20];rows.append(row)
    return rows


def worker(a):
    from dev.prefill_asys.trace import activities,runtime_activities
    from dev.moe_grouped.bench import parse
    receipt=read(a.output/'build/receipt.json');verify(receipt)
    folder=a.output/f'gpu-{a.worker}';folder.mkdir(exist_ok=True)
    jobs=read(a.output/'run.json')['assignments'][str(a.worker)]
    # A fresh process per batch avoids reusing a poisoned device context.
    failures=0
    for start in range(0,len(jobs),4):
        batch=folder/f'batch-{start//4}';batch.mkdir(exist_ok=True)
        selected=jobs[start:start+4];saved=batch/'result.json'
        if a.resume and saved.exists() and read(saved).get('status')=='PASS':continue
        attempt=batch/('attempt-'+str(time.time_ns()));attempt.mkdir()
        atomic_json(attempt/'jobs.json',selected)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
                 '--output',str(a.output),'--batch',str(attempt)]
        profile=[str(a.asys),'profile','-t','hggc','--sample','none','--python-sampling','false',
            '--capture-range','hggcProfilerApi','--capture-range-end','merge','--kill','none','--wait','primary',
            '-f','false','-o',str(attempt/'trace'),*command]
        print(f'MOE_GROUPED_BATCH first={start} total={len(jobs)} log={attempt/"profile.log"}',flush=True)
        try:
            run(profile,attempt/'profile.log')
            run([str(a.asys),'export','-o',str(attempt/'trace.sqlite'),str(attempt/'trace.asysrep')],attempt/'export.log')
            capture=read(attempt/'capture.json')
            result=parse(capture,activities(attempt/'trace.sqlite',launch_shared=True),runtime_activities(attempt/'trace.sqlite'))
            result.update(trace=str(attempt/'trace.asysrep'),trace_sha256=sha(attempt/'trace.asysrep'))
            atomic_json(saved,result)
        except Exception as exc:
            failures+=1;atomic_json(saved,dict(status='FAIL',jobs=selected,error=str(exc),attempt=str(attempt)))
            print('MOE_GROUPED_FAIL '+str(exc),flush=True)
    return int(bool(failures))


def main(a):
    jobs=plan(a.smoke)
    if a.plan:
        print(json.dumps(dict(jobs=jobs,points=len(jobs),modules=6,contract_points=28,
            config_search=False,full_provider=False,simt_policy_changed=False,production_changed=False),indent=2));return 0
    if a.output is None:a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-grouped.',dir=os.environ.get('RUN_ROOT','/workspace')))
    a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    if a.worker is not None:return worker(a)
    if a.batch:
        from dev.moe_grouped.bench import measure
        receipt=read(a.output/'build/receipt.json');verify(receipt);config=read(a.output/'run.json')
        capture=measure(a.sdk,receipt['helper'],receipt['modules'],read(a.batch/'jobs.json'),
                        Path(config['smi']),config['clock_mhz'])
        capture['provenance']=dict(commit=revision(ROOT),build_commit=receipt['helper']['quactlize'],
            build_sha256=sha(a.output/'build/receipt.json'))
        atomic_json(a.batch/'capture.json',capture);return 0
    if a.gate:
        from dev.moe_grouped.bench import contract_gate
        receipt=read(a.output/'build/receipt.json');verify(receipt)
        atomic_json(a.output/'contract.json',contract_gate(a.sdk,receipt['helper'],receipt['modules']));return 0
    if not a.confirm_idle:raise ValueError('use --confirm-idle only on idle assigned GPUs')
    devices=[str(int(x)) for x in a.devices.split(',')]
    if not devices or len(set(devices))!=len(devices) or any(int(x)<0 for x in devices):raise ValueError('invalid devices')
    a.output.mkdir(parents=True,exist_ok=True)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True);a.smi=selected_smi(a)
    assignments={d:[] for d in devices};loads={d:0 for d in devices}
    for job in sorted(jobs,key=lambda j:j['m']*j['k'],reverse=True):
        device=min(devices,key=lambda d:loads[d]);assignments[device].append(job);loads[device]+=job['m']*job['k']
    config=dict(assignments=assignments,devices=devices,smi=str(a.smi),clock_mhz=a.clock_mhz,
                sdk=str(a.sdk),asys=str(a.asys),asys_sha256=sha(a.asys),points=len(jobs))
    if a.resume:
        if read(a.output/'run.json')!=config:raise ValueError('resume inventory/device configuration differs')
        verify(read(a.output/'build/receipt.json'))
    else:
        if (a.output/'run.json').exists():raise ValueError('use --resume or a new output')
        atomic_json(a.output/'run.json',config)
        if a.reuse_build:
            from dev.moe_grouped.reuse import reuse
            reuse(a.sdk,a.reuse_build,a.output/'build')
        else:
            print('MOE_GROUPED_BUILD '+str(a.output),flush=True)
            build(a.sdk,a.output/'build',a.jobs)
    gate=a.output/'contract.json'
    if not a.resume or not gate.exists() or read(gate).get('status')!='PASS':
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),'--gate']
        print('MOE_GROUPED_CONTRACT_START log='+str(a.output/'contract.log'),flush=True)
        with (a.output/'contract.log').open('w') as log:
            process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[0]),stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                print('MOE_GROUPED_CONTRACT_WAIT log='+str(a.output/'contract.log'),flush=True)
            rc=process.returncode
        if rc:raise RuntimeError('real producer contract failed; see '+str(a.output/'contract.log'))
    active=[];start=time.monotonic()
    for device in devices:
        folder=a.output/f'gpu-{device}';folder.mkdir(exist_ok=True)
        log=(folder/('console-'+str(time.time_ns())+'.log')).open('w')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                 '--worker',device,'--asys',str(a.asys)]
        if a.resume:command.append('--resume')
        active.append((subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdout=log,stderr=subprocess.STDOUT),log))
    while any(p.poll() is None for p,_ in active):
        time.sleep(10)
        print('MOE_GROUPED_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in active),
            minutes=(time.monotonic()-start)/60,logs=str(a.output/'gpu-*/console-*.log'))),flush=True)
    codes=[p.returncode for p,_ in active]
    for _,log in active:log.close()
    results=[read(p) for p in sorted(a.output.glob('gpu-*/batch-*/result.json'))]
    points=sum(len(r.get('points',[])) for r in results if r.get('status')=='PASS')
    status='PASS' if not any(codes) and points==len(jobs) else 'INCOMPLETE'
    summary=dict(status=status,passed=points,expected=len(jobs),returncodes=codes,seconds=time.monotonic()-start,
        production_changed=False,no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',summary)
    print('MOE_GROUPED_DONE '+json.dumps(summary),flush=True)
    return int(status!='PASS')


def review(output):
    if not output or not output.is_dir():return
    target=output/'review.tgz'
    paths=[p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json','.log') and
           not p.is_relative_to(output/'build/cache')]
    paths+=list((output/'build/cache').glob('*/build.log'))
    with tarfile.open(target,'w:gz') as archive:
        for p in sorted(paths):archive.add(p,arcname=str(p.relative_to(output)))
    print('results='+str(target),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    parser.add_argument('--output',type=Path);parser.add_argument('--devices',default='3,5,6,7')
    parser.add_argument('--reuse-build',type=Path,help='verified unchanged device build; new results and harness provenance')
    parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--asys',type=Path)
    parser.add_argument('--smi',type=Path);parser.add_argument('--clock-mhz',type=int,default=1700)
    for flag in ('smoke','resume','confirm-idle','plan','gate'):parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--worker',type=int);parser.add_argument('--batch',type=Path)
    a=parser.parse_args();rc=1
    try:rc=main(a)
    except Exception:traceback.print_exc()
    finally:
        if a.worker is None and a.batch is None and not a.gate and not a.plan:review(a.output)
    raise SystemExit(rc)
