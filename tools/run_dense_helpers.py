#!/usr/bin/env python3
"""One resumable, factorized dense-helper closeout; never changes production tables."""
import argparse
import ctypes as C
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.dense_helpers.plan import cohort,jobs,full_plan,validate_result
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_dense_dequant import build,verify_receipt
from tools.run_prefill_cohort_multi import gpu_ordinals


def read(path):return json.loads(Path(path).read_text())


def assign(families,gpus):
    bins=[[] for _ in gpus];loads=[0]*len(gpus)
    for f in sorted(families,key=lambda f:(-f['n']*f['k'],f['q'],f['n'],f['k'])):
        i=min(range(len(gpus)),key=lambda i:(loads[i],i))
        bins[i].append(f);loads[i]+=f['n']*f['k']+8_000_000
    return dict(zip(gpus,bins))


def execute(command,log,env=None):
    atomic_json(log.with_suffix('.command.json'),command)
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        start=time.monotonic()
        try:
            while process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'DENSE_HELPERS_WAIT minutes={(time.monotonic()-start)/60:.1f} log={log}',flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        atomic_json(log.with_suffix('.process.json'),dict(returncode=process.returncode,seconds=time.monotonic()-start))
        return process.returncode


def child(a):
    from tools.run_kpack_llama_ab import device
    from dev.dense_helpers.measure import dequant,io as io_measure
    record=verify_receipt(a);authority=read(a.output/'authority.json')
    if device(a)!=authority['device']:raise ValueError('component child device/runtime differs')
    job=read(a.job);key=digest(job)
    a.profile_variant=None
    value=dequant(a,job) if job['kind']=='dequant' else io_measure(a,job,C.CDLL(record['library'],mode=C.RTLD_LOCAL))
    value.update(status='PASS',job=job,authority=digest(authority),
        l2_bytes=a.l2_bytes,
        timing='ROTATING_SINGLE_KERNEL_GRAPH_EVENTS_INCLUDES_NODE_GAPS',production_changed=False)
    value['record_sha256']=digest(value);validate_result(value,job,digest(authority))
    atomic_json(a.output/'cells'/(key+'.json'),value)
    print('DENSE_HELPERS_CELL '+json.dumps(dict(job=job,median_us={k:v['median_us'] for k,v in value['arms'].items()})),flush=True)
    return 0


def worker(a):
    from tools.run_kpack_llama_ab import device
    from dev.strategy.refresh_clock import request_lock,observe
    families=read(a.families);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'build').mkdir(exist_ok=True)
    receipt=read(a.probe/'build/receipt.json')
    local=a.output/'build/receipt.json'
    if local.exists() and read(local)!=receipt:raise ValueError('worker probe receipt changed')
    if not local.exists():atomic_json(local,receipt)
    verify_receipt(a)
    lib=C.CDLL(receipt['library'],mode=C.RTLD_LOCAL)
    name=C.create_string_buffer(256);device_name=lib.dense_helper_device_name
    device_name.argtypes=[C.c_char_p,C.c_int];device_name.restype=C.c_int
    if device_name(name,len(name)):raise ValueError('device name query failed')
    identity=device(a);a.l2_bytes=identity['l2']['bytes'];pci=identity['device']['pci']
    inventory=jobs(families)
    authority=dict(device=identity,library_sha256=receipt['library_sha256'],
        device_name=name.value.decode(),
        source=receipt['sources'],jobs=digest(inventory),frequency_normalization=False,
        timing='ROTATING_SINGLE_COMPONENT_SCREEN_THEN_ORDINARY_ASYS_FULL_CALL',requested_mhz=1700)
    path=a.output/'authority.json'
    if path.exists() and read(path)!=authority:raise ValueError('worker source/device/inventory changed')
    if not path.exists():atomic_json(path,authority)
    (a.output/'jobs').mkdir(exist_ok=True);(a.output/'logs').mkdir(exist_ok=True)
    results={};failed=[];started=time.monotonic()
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        clock=request_lock(a.smi,pci,1700)
        atomic_json(a.output/f'clock-lock-{time.time_ns()}.json',clock)
        if clock['returncode']:raise ValueError('1700MHz clock request failed')
        for count,(key,job) in enumerate(sorted(inventory.items()),1):
            result=a.output/'cells'/(key+'.json')
            if result.exists():results[key]=validate_result(read(result),job,digest(authority));continue
            request=a.output/'jobs'/(key+'.json');atomic_json(request,job)
            log=a.output/'logs'/(key+'-'+str(time.time_ns())+'.log')
            before=observe(a.smi,pci,1700,advisory=True)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(request),
                '--output',str(a.output),'--sdk',str(a.sdk),'--device',a.device,'--l2-bytes',str(a.l2_bytes)]
            rc=execute(command,log)
            after=observe(a.smi,pci,1700,advisory=True)
            atomic_json(log.with_suffix('.clock.json'),dict(before=before,after=after))
            if rc==0 and result.exists():results[key]=validate_result(read(result),job,digest(authority))
            else:failed.append(dict(key=key,job=job,returncode=rc,log=str(log)))
            print('DENSE_HELPERS_PROGRESS '+json.dumps(dict(completed=count,total=len(inventory),failed=len(failed),
                elapsed_minutes=(time.monotonic()-started)/60,job=job)),flush=True)
    atomic_json(a.output/'screen-summary.json',dict(passed=len(results),expected=len(inventory),failed=failed,
                production_changed=False,status='PASS' if not failed else 'INCOMPLETE'))
    if failed:
        # Keep every successful cell; a later resume retries only missing cells.
        # Other physical-device workers continue to their complete-call phase.
        return 1
    query=lib.dense_helper_control
    query.argtypes=[C.c_int]*5+[C.POINTER(C.c_int)]*2;query.restype=None
    def policy(q,m,n,k,storage):
        if name.value!=b'PPU-ZW810' or identity['device']['compute_units']!=72:return None
        dq,vec=C.c_int(),C.c_int();query(q,m,n,k,storage,C.byref(dq),C.byref(vec))
        return (dq.value,vec.value) if dq.value>=0 else None
    plan=full_plan(families,results,policy,digest(authority))
    planfile=a.output/'helper-finalists.json'
    if planfile.exists() and read(planfile)!=plan:raise ValueError('finalist plan changed during resume')
    if not planfile.exists():atomic_json(planfile,plan)
    command=[sys.executable,'-u',str(ROOT/'tools/run_prefill_model_asys.py'),
        '--helper-plan',str(planfile),'--output',str(a.output/'full'),'--sdk',str(a.sdk),
        '--llama-source',str(a.llama_source),'--llama-build',str(a.llama_build),
        '--device',a.device,'--confirm-idle','--l2-bytes',str(a.l2_bytes),'--jobs','2']
    if (a.output/'full/plan.json').exists():command.append('--resume')
    return execute(command,a.output/f'full-{time.time_ns()}.log')


def package(output):
    paths=set(output.glob('*.json'))
    for worker in output.glob('gpu-*'):
        for pattern in ('*.json','worker-*.log','cells/*.json','logs/*.log','logs/*.process.json','logs/*.clock.json',
                        'full-*.log','full/*.json','full/*.log','full/build/**/*.log',
                        'full/build-failed-*/**/*.log',
                        'full/build/receipt.json','full/build/dense/manifest.json',
                        'full/batches/*/pass.json','full/batches/*/attempt-*/cases.json',
                        'full/batches/*/attempt-*/*.log','full/batches/*/attempt-*/*.process.json'):
            paths.update(worker.glob(pattern))
    for directory in [output/'build',*output.glob('build-failed-*')]:
        paths.update(directory.glob('*.json'));paths.update(directory.glob('*.log'))
    target=output/f'review-{time.time_ns()}.tgz'
    with tarfile.open(target,'x:gz') as tar:
        for path in sorted(paths):
            if not path.is_file() or path.is_symlink():continue
            with path.open('rb') as src:
                if path.suffix=='.log':src.seek(max(0,path.stat().st_size-16384))
                data=src.read()
            info=tarfile.TarInfo(str(path.relative_to(output)));info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    print('results='+str(target),flush=True)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--gpus',default='1');p.add_argument('--device',default='0')
    p.add_argument('--cohort',type=Path);p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--llama-source',type=Path,default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--compile-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--families',type=Path)
    p.add_argument('--probe',type=Path);p.add_argument('--job',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    a.llama_source=a.llama_source.resolve();a.llama_build=(a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve()
    if a.job:return child(a)
    if a.worker:return worker(a)
    if not (a.plan_only or a.compile_only or a.confirm_idle):p.error('--confirm-idle is required')
    document=cohort(a.cohort);gpus=gpu_ordinals(a.gpus)
    if len(gpus)>len(document['families']):p.error('more GPUs than families')
    assignments=assign(document['families'],gpus)
    model_families=sum(bool(f['models']) for f in document['families'])
    plan=dict(cohort=document,gpus=gpus,assignments=assignments,source_sha256=sha(__file__),
              stages=['DEDUPLICATED_SINGLE_KERNEL_COMPONENTS','SHORTLIST','ASYS_FULL_CALL_ABBA'],
              counts=dict(families=len(document['families']),model_families=model_families,
                  model_points=5*model_families,extension_points=5*(len(document['families'])-model_families),
                  component_cells=sum(len(jobs(f)) for f in assignments.values())),production_changed=False)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume and read(a.output/'plan.json')!=plan:raise ValueError('resume assignment or harness changed')
    if not a.resume:atomic_json(a.output/'plan.json',plan)
    print('DENSE_HELPERS_PLAN '+json.dumps(plan['counts']),flush=True)
    if a.plan_only:return 0
    with (a.output/'.coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            receipt=a.output/'build/receipt.json'
            old=read(receipt) if receipt.exists() else None
            if old and all(k in old for k in ('library_sha256','runtime','object_sha256','seconds')):
                verify_receipt(a)
            else:
                partial=a.output/'build'
                if partial.exists():
                    if partial.is_symlink() or not partial.is_dir():raise ValueError('unexpected probe build path')
                    if old and (old['compiler_sha256']!=sha(a.sdk/'bin/hgcc') or
                                any(sha(ROOT/p)!=h for p,h in old['sources'].items())):
                        raise ValueError('failed build belongs to another source/SDK; use a new run')
                    partial.rename(a.output/('build-failed-'+str(time.time_ns())))
                build(a,('quactlize/prefill/dense_io.cu','dev/dense_helpers/policy.cpp'))
        except BaseException:
            atomic_json(a.output/f'build-error-{time.time_ns()}.json',dict(traceback=traceback.format_exc()))
            package(a.output);raise
        if a.compile_only:return 0
        a.smi=a.smi or next((p for p in (Path(shutil.which('ppu-smi') or '/nonexistent'),
                            a.sdk/'bin/ppu-smi') if p.is_file()),None)
        if a.smi is None or not a.smi.is_file():raise ValueError('set --smi to the installed ppu-smi')
        processes=[];streams=[];start=time.monotonic()
        try:
            for gpu in gpus:
                root=a.output/('gpu-'+gpu);root.mkdir(exist_ok=a.resume)
                family=root/'families.json'
                if not family.exists():atomic_json(family,assignments[gpu])
                log=root/f'worker-{time.time_ns()}.log';stream=log.open('x');streams.append(stream)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--output',str(root),
                    '--families',str(family),'--probe',str(a.output),'--sdk',str(a.sdk),'--device',gpu,
                    '--smi',str(a.smi),'--l2-bytes',str(a.l2_bytes),'--llama-source',str(a.llama_source),
                    '--llama-build',str(a.llama_build)]
                proc=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                processes.append(proc)
            while any(proc.poll() is None for proc in processes):
                print('DENSE_HELPERS_MULTI_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in processes),
                    workers=len(gpus),elapsed_minutes=(time.monotonic()-start)/60,logs=str(a.output/'gpu-*/worker-*.log'))),flush=True)
                time.sleep(30)
            codes=[p.returncode for p in processes]
            result=dict(status='COMPLETE' if not any(codes) else 'INCOMPLETE',returncodes=codes,
                production_changed=False,model_admission=False,scope='HELPERS_NOT_NEW_TC_GEMM_SWEEP')
            atomic_json(a.output/'summary.json',result);print('DENSE_HELPERS_DONE '+json.dumps(result),flush=True)
            return int(any(codes))
        finally:
            for proc in processes:
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for proc in processes:
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            for stream in streams:stream.close()
            package(a.output)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
