#!/usr/bin/env python3
"""Dense IO/Q6/Full-FP16 gate, component timers, then bounded one-call ACU."""
import argparse
import csv
import ctypes as C
from dataclasses import asdict
import fcntl
import json
import io
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dev.prefill_dense.spec import workloads,arms,STAGES,profile_jobs
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

def read(p):return json.loads(p.read_text())

def protocol_for(a):
    from dev.prefill_dense import round2,round3,round4
    return round4 if a.round_four else round3 if a.round_three else round2 if a.round_two else None

def harness():
    from dev.strategy.provenance import harness_identity
    local={str(p.relative_to(ROOT)):sha(p)for p in (ROOT/'dev/prefill_dense').rglob('*')
           if p.is_file()and p.suffix in ('.py','.cpp')}
    local['runner']=sha(__file__);local['transitive']=harness_identity(ROOT)['sha256']
    return digest(local)

def check_build(a):
    m=read(a.output/'build/manifest.json');t=read(a.output/'build/timer.json')
    if sha(a.sdk/'bin/hgcc')!=m['compiler_sha256']:raise ValueError('SDK compiler changed')
    if sha(a.output/'build'/m['library'])!=m['sha256']or sha(a.output/'build/libdense-opt-timer.so')!=t['sha256']:
        raise ValueError('candidate binary changed')
    if sha(ROOT/'dev/prefill_dense/timer.cpp')!=t['source']:raise ValueError('timer source changed')
    for path,want in m['source_hashes'].items():
        if sha(ROOT/path)!=want:raise ValueError('candidate source changed: '+path)
    for path,want in m['runtime'].items():
        if sha(a.sdk/'lib'/path)!=want:raise ValueError('SDK runtime changed')
    return m

def child(a):
    from quactlize.runtime.native import SDK,checked
    from dev.prefill_dense.bench import Bench,io_gate,dequant_gate,dynamic_gate
    from tools.run_kpack_llama_ab import device
    authority=read(a.output/'authority.json')
    if authority['harness']!=harness():raise ValueError('measurement harness changed')
    if device(a)!=authority['device']:raise ValueError('physical device/SDK differs')
    if sha(authority['provider']['path'])!=authority['provider']['sha256']:
        raise ValueError('cuBLAS provider binary changed')
    m=check_build(a);sdk=SDK(a.sdk);w=workloads()[a.point]
    lib=a.output/'build'/m['library'];timer=a.output/'build/libdense-opt-timer.so'
    protocol=protocol_for(a)
    if protocol:
        from dev.prefill_dense.round2 import measure
        return measure(a,sdk,w,lib,timer,m,authority,protocol=protocol)
    profile=a.arm is not None
    if not profile and a.point==0:
        atomic_json(a.output/'io-gate.json',io_gate(sdk,a.sdk,lib))
        atomic_json(a.output/'dequant-gate.json',dequant_gate(sdk,lib))
        atomic_json(a.output/'dynamic-gate.json',dynamic_gate(sdk,a.sdk,lib,timer,a.l2_bytes))
    b=None
    try:
        b=Bench(sdk,a.sdk,lib,timer,w,a.l2_bytes,profile=profile)
        pool=[r for r in arms()if not profile or r['name']==a.arm]
        handles={r['name']:b.handle(r)for r in pool};proofs={};timings={r['name']:{s:[]for s in STAGES}for r in pool}
        print(f'DENSE_OPT_FIXTURE point={a.point} copies={b.copies} profile={int(profile)}',flush=True)
        if profile:
            prior=read(a.output/f'point-{a.point}.json')
            if (prior['status']!='PASS'or prior['library_sha256']!=m['sha256']or
                    prior['authority']!=digest(authority)):
                raise ValueError('prior numerical/timing admission missing')
            if a.arm not in {r['arm']['name']for r in prior['arms']}:raise ValueError('profile arm not admitted')
        for arm in pool:
            h=handles[arm['name']];proof=[]
            for repeat in (0,1):
                b.update(repeat)
                for i in sorted({0,b.copies-1}):b.launch(h,i);proof.append(b.check())
            b.check_weights(arm['storage'],b.copies-1)
            b.update(0,zero=True);b.launch(h);b.check(zero=True)
            b.update(0);b.launch(h);proof.append(b.check());proofs[arm['name']]=proof
            print('DENSE_OPT_NUMERIC '+json.dumps(dict(point=a.point,arm=arm['name'],status='PASS')),flush=True)
        if profile:
            h=handles[a.arm]
            for _ in range(3):b.launch(h)
            sdk.synchronize(b.r.stream)
            start,stop=sdk.lib.hggcProfilerStart,sdk.lib.hggcProfilerStop
            for fn in (start,stop):fn.argtypes=[];fn.restype=C.c_int
            print(f'DENSE_OPT_ACU_BEGIN point={a.point} arm={a.arm} stage={a.stage} calls=1 cold_ring=0',flush=True)
            checked(start(),'profiler start')
            try:b.launch(h,stage=a.stage);sdk.synchronize(b.r.stream)
            finally:checked(stop(),'profiler stop')
            b.launch(h);b.check()
            result=dict(status='PASS',point=a.point,arm=a.arm,stage=a.stage,calls=1,cold_ring=False,
                library_sha256=m['sha256'],numerical=proofs,authority=digest(authority),profiled=True)
        else:
            for rnd in range(3):
                order=pool if rnd%2==0 else list(reversed(pool))
                for arm in order:
                    for stage in STAGES:timings[arm['name']][stage]+=b.samples(handles[arm['name']],stage)
                    print(f'DENSE_OPT_PROGRESS point={a.point} round={rnd+1}/3 arm={arm["name"]}',flush=True)
            result=dict(status='PASS',point=a.point,receipt=b.receipt,library_sha256=m['sha256'],authority=digest(authority),
                arms=[dict(arm=arm,proof=proofs[arm['name']],samples_us={STAGES[k]:v for k,v in timings[arm['name']].items()},
                    median_us={STAGES[k]:statistics.median(v)for k,v in timings[arm['name']].items()})for arm in pool],
                timing='ORDINARY_EVENT_COMPONENT_INTERVALS_AND_SEPARATE_COMPLETE_CALL_NOT_ACU_CYCLES',
                production_changed=False,performance_admitted=False)
        atomic_json(a.result,result);return 0
    finally:
        if b:b.close()

def execute(command,log):
    atomic_json(log.with_suffix('.command.json'),command)
    with log.open('x')as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        from tools.run_prefill_point import wait
        try:wait(p,'dense-opt',log)
        except RuntimeError:
            with log.open('rb')as f:f.seek(max(0,log.stat().st_size-8192));print(f.read().decode(errors='replace'),flush=True)
            return p.wait()
    return p.returncode

def package(root):
    path=root/('review-'+str(time.time_ns())+'.tgz')
    with tarfile.open(path,'w:gz')as tar:
        for p in root.rglob('*'):
            if p.is_file()and not p.is_symlink()and p.suffix in ('.json','.log','.csv')and p.stat().st_size<=16<<20:
                tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    print('results='+str(path),flush=True)

def check_profile(path,job):
    text=path.read_text(errors='replace');start=text.find('"ID"')
    if start<0:raise ValueError('ACU export has no kernel table')
    rows=[r for r in csv.DictReader(io.StringIO(text[start:]))if r.get('Kernel Name')]
    launches=sorted({(r['ID'],r['Kernel Name'])for r in rows})
    if not launches:raise ValueError('ACU captured no GPU kernels')
    if job['stage']!=2:
        if len(launches)!=1:raise ValueError('one component produced unexpected launches')
        if job['stage']==0:expected='gather_bf16'if job['arm']=='bf16-old'else'dense_convert_vec8'
        elif job['stage']==3:expected='finish_bf16'if job['arm']=='bf16-old'else'dense_convert_vec8'
        elif job['arm']in ('bf16-old','bf16-vec'):expected='full_wide'if job['point']==5 else'full_transpose'
        else:expected='dense_dequant_packed'
        if expected not in launches[0][1]:raise ValueError('ACU kernel differs from selected component')
    return dict(launches=launches,raw_sha256=sha(path),scope='FORCED_COLD_ACU_ALL_COMPONENT_KERNELS')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='1');p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--confirm-idle',action='store_true');p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--point',type=int,choices=range(7));p.add_argument('--result',type=Path)
    p.add_argument('--arm');p.add_argument('--stage',type=int,choices=range(4))
    rounds=p.add_mutually_exclusive_group()
    rounds.add_argument('--round-two',action='store_true');rounds.add_argument('--round-three',action='store_true')
    rounds.add_argument('--round-four',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--no-acu',action='store_true');a=p.parse_args()
    a.sdk=a.sdk.resolve(strict=True);a.output=a.output.resolve();os.environ['CUDA_VISIBLE_DEVICES']=a.device
    protocol=protocol_for(a)
    if a.point is not None:
        if a.result is None or (a.arm is not None and a.stage is None):p.error('child result/profile stage missing')
        candidates=protocol.pool(a.point)if protocol else arms()
        if a.arm is not None and a.arm not in {r['name']for r in candidates}:
            p.error('unknown arm for this point/round')
        return child(a)
    if not a.confirm_idle:p.error('confirm assigned GPU idle')
    if a.resume:
        if not protocol:p.error('resume is only supported for rounds two, three and four')
        if not (a.output/'authority.json').is_file():p.error('resume requires an existing frozen output')
    else:a.output.mkdir(parents=True,exist_ok=False)
    from tools.run_kpack_llama_ab import device
    from dev.strategy.refresh_clock import request_lock,observe
    d=device(a);a.l2_bytes=d['l2']['bytes'];smi=Path(shutil.which('ppu-smi')or a.sdk/'bin/ppu-smi')
    pci=d['device']['pci'];rows=[];reports=[]
    provider=(a.sdk/'CUDA_SDK/targets/x86_64-linux/lib/libcublas.so').resolve(strict=True)
    authority=dict(device=d,harness=harness(),workloads=[asdict(w)for w in workloads()],arms=arms(),
        provider=dict(path=str(provider),sha256=sha(provider),compute='CUBLAS_COMPUTE_32F',
            math_mode=16,workspace_bytes=4<<20,algorithm_controls=[-1,99],no_graph=True))
    if protocol:
        authority.update(round_two=a.round_two,round_three=a.round_three,arms={str(i):protocol.pool(i)for i in range(7)})
        if a.round_four:authority['round_four']=True
        authority['provider']['algorithm_controls']=[-1]
    if a.resume:
        if read(a.output/'authority.json')!=authority:raise ValueError('resume device/source/provider contract changed')
    else:atomic_json(a.output/'authority.json',authority)
    lock=Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')
    try:
        with lock.open('a')as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            frequency=request_lock(smi,pci,1700);atomic_json(a.output/'clock-lock.json',frequency)
            if frequency['returncode']:raise ValueError('1700MHz request failed')
            from dev.prefill_dense.build import build
            if not a.resume:build(a.sdk,a.output/'build',a.jobs)
            manifest=check_build(a)
            base=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                  '--device',a.device,'--l2-bytes',str(a.l2_bytes)]
            if a.round_two:base+=['--round-two']
            if a.round_three:base+=['--round-three']
            if a.round_four:base+=['--round-four']
            def admitted(path):
                if not path.is_file():return False
                row=read(path)
                return row.get('status')=='PASS'and row.get('authority')==digest(authority)and row.get('library_sha256')==manifest['sha256']
            for point in range(7):
                result=a.output/f'point-{point}.json'
                if a.resume and admitted(result):
                    rc=0;print(f'DENSE_OPT_REUSE point={point} unchanged=1',flush=True)
                else:
                    suffix='-'+str(time.time_ns())if a.resume else''
                    rc=execute(base+['--point',str(point),'--result',str(result)],a.output/f'point-{point}{suffix}.log')
                row=read(result)if not rc and result.is_file()else dict(status='FAIL',point=point,rc=rc)
                rows.append(row);atomic_json(a.output/'screen.json',dict(points=rows))
                atomic_json(a.output/f'clock-{point}.json',observe(smi,pci,1700,advisory=True))
                print('DENSE_OPT_POINT '+json.dumps(dict(point=point,status=row['status'])),flush=True)
            if not a.no_acu and all(r['status']=='PASS'for r in rows):
                jobs=protocol.profile_jobs(rows)if protocol else profile_jobs(rows)
                checker=protocol.check_profile if protocol else check_profile
                for i,job in enumerate(jobs):
                    saved=a.output/f'acu-{i}.receipt.json'
                    if a.resume and saved.is_file():
                        prior=read(saved);raw=Path(prior['raw']);result=Path(prior['result'])
                        if prior['job']==job and admitted(result)and raw.is_file()and prior['status']=='PASS':
                            if digest(checker(raw,job))==digest(prior['evidence']):
                                reports.append(prior);print(f'DENSE_OPT_REUSE profile={i} unchanged=1',flush=True);continue
                    suffix='-'+str(time.time_ns())if a.resume else''
                    target=a.output/f'profile-{i}{suffix}';result=a.output/f'profile-{i}{suffix}.json'
                    # One explicitly ranged component; keep ALL provider subkernels.
                    cmd=[str(a.sdk/'asight/bin/acu'),'--set','full','--profile-from-start','no',
                         '--graph-profiling','none','--replay-mode','kernel','--cache-control','all',
                         '--kill','no','--check-exit-code','yes','--export',str(target),*base,
                         '--point',str(job['point']),'--arm',job['arm'],'--stage',str(job['stage']),'--result',str(result)]
                    rc=execute(cmd,a.output/f'acu-{i}{suffix}.log');report=target.with_suffix('.acurep')
                    raw=a.output/f'profile-{i}{suffix}.raw.csv'
                    if not rc and report.is_file():
                        rc=execute([str(a.sdk/'asight/bin/acu'),'--import',str(report),'--page','raw','--csv'],raw)
                    else:rc=rc or 1
                    evidence={}
                    if not rc:
                        try:
                            if not admitted(result):raise ValueError('ACU numerical receipt not admitted')
                            evidence=checker(raw,job)
                        except (ValueError,KeyError)as exc:rc=1;evidence=dict(error=str(exc))
                    reports.append(job|dict(job=job,rc=rc,report=str(report),raw=str(raw),result=str(result),
                        status='PASS'if not rc else'FAIL',evidence=evidence))
                    atomic_json(saved,reports[-1])
                    print('DENSE_OPT_ACU '+json.dumps(job|dict(status=reports[-1]['status'],rc=rc,
                        cycles=evidence.get('cycles'),report=str(report))),flush=True)
            status='PASS'if all(r['status']=='PASS'for r in rows+reports)else'INCOMPLETE'
            if protocol:
                pairs=protocol.cycle_pairs(reports);atomic_json(a.output/'cycle-pairs.json',pairs)
                for pair in pairs:print(f'DENSE_R{protocol.ROUND}_CYCLES '+json.dumps(pair),flush=True)
            atomic_json(a.output/'summary.json',dict(status=status,points=rows,acu=reports,production_changed=False,
                performance_admitted=False,round_two=a.round_two,round_three=a.round_three,round_four=a.round_four,acu_skipped=a.no_acu,
                scope='DENSE_PREFILL_CANDIDATES_NOT_MODEL_E2E'))
            print('DENSE_OPT_DONE '+status,flush=True);return int(status!='PASS')
    finally:package(a.output)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
