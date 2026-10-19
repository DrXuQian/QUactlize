#!/usr/bin/env python3
"""Bounded numeric/JIT extraction diagnosis. Never overwrite or time a failed row."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.build import build_adapter
from dev.native_components.fixture import weights,activation
from dev.native_components.plan import plan
from dev.llama_ab.contract import family
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_native_components import carrier,environment,device,point_request


def failures(previous,rows):
    by={r['id']:r for r in rows};found=[];counts=Counter()
    for path in sorted(previous.glob('gpu-*/results/*.json')):
        record=json.loads(path.read_text());counts[record['status']]+=1
        if record['status']!='FAIL':continue
        if path.stem not in by:raise ValueError('unknown previous point: '+path.stem)
        worker=path.parent.parent
        saved=Path(record['log'])
        # Compact review archives retain the original absolute Box path.
        # Resolve only the same worker's attempts/<uuid>/run.log in this run.
        if len(saved.parts)<4 or saved.parts[-4]!=worker.name or saved.parts[-3]!='attempts' or saved.name!='run.log':
            raise ValueError('unexpected failure log path: '+str(saved))
        log=(worker/Path(*saved.parts[-3:])).resolve(strict=True)
        if not log.is_relative_to(previous):raise ValueError('failure log outside previous run')
        text=log.read_text(errors='replace')
        if 'native kernel params:'in text:kind='jit-params';error=None
        else:
            errors=re.findall(r'independent GGUF spot dot error native=([\d.]+)',text)
            if not errors:raise ValueError('unreviewed previous failure: '+str(log))
            kind='q8-activation'if by[path.stem]['workload']['m']<=8 else'provider-precision'
            error=float(errors[-1])
        found.append(dict(row=by[path.stem],kind=kind,error=error,worker=worker,
                          record_sha256=sha(path),log_sha256=sha(log)))
    return found,dict(counts)


def select(found):
    # At most three JIT routes, two MMVQ formats and four dense precisions.
    selected=[]
    for kind,types in (('jit-params',(10,11,12)),('q8-activation',(12,13)),
                       ('provider-precision',(10,11,12,13))):
        for q in types:
            pool=[r for r in found if r['kind']==kind and r['row']['workload']['qtype']==q]
            if not pool:continue
            if kind=='jit-params':
                pick=min(pool,key=lambda r:(r['row']['workload']['m']*r['row']['workload']['n']*
                                            r['row']['workload']['k'],r['row']['id']))
            else:pick=max(pool,key=lambda r:(r['error'],r['row']['id']))
            selected.append(pick)
    return selected


def check_binding(old,new,dev):
    # The test adapter changes; the tested production source, libraries and
    # compiler options must not. Do not attribute a caller/SDK upgrade to this fix.
    for key in ('binding','headers','flags','response_files','compiler_sha256','llama_head'):
        if old['identity'][key]!=new['identity'][key]:
            raise ValueError('production binding changed: '+key)
    before,after=old['identity']['libraries'],new['identity']['libraries']
    if any(after.get(k)!=v for k,v in before.items()) or any(
            not Path(k).name.startswith('libcuda.so')for k in after.keys()-before.keys()):
        raise ValueError('production binding changed: libraries')
    if dev['sdk'] is None:raise ValueError('missing diagnostic SDK identity')


def pack(output):
    target=output/'review.tgz'
    with tarfile.open(target,'w:gz')as tar:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json','.log') and 'fixtures'not in path.relative_to(output).parts:
                tar.add(path,arcname=str(path.relative_to(output)),recursive=False)
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--llama-source',type=Path,required=True)
    p.add_argument('--llama-build',type=Path,required=True)
    p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();a.previous=a.previous.resolve(strict=True);a.output=a.output.resolve()
    if a.output.exists() and any(a.output.iterdir()):p.error('use a new diagnostic output; previous results stay untouched')
    if a.output==a.previous or a.output.is_relative_to(a.previous):p.error('diagnostic output must be separate from previous run')
    a.output.mkdir(parents=True,exist_ok=True)
    found,counts=failures(a.previous,plan()['points']);selected=select(found)
    if not selected:raise ValueError('no reviewed failures to diagnose')
    jobs=[(r,mode)for r in selected for mode in
          (('native','bf16','f32')if r['kind']=='provider-precision'else('native',))]
    frozen=dict(previous=str(a.previous),previous_summary_sha256=sha(a.previous/'summary.json'),
        counts=counts,failure_counts=dict(Counter(r['kind']for r in found)),
        points=[dict(id=r['row']['id'],kind=r['kind'],old_error=r['error'],workload=r['row']['workload'],
                     previous_record_sha256=r['record_sha256'],previous_log_sha256=r['log_sha256'])for r in selected],
        jobs=len(jobs),timing=False,production_changed=False,previous_results_modified=False)
    atomic_json(a.output/'plan.json',frozen)
    print('NATIVE_DIAGNOSTIC_PLAN '+json.dumps(frozen),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:p.error('--confirm-idle required')
    if ',' in os.environ.get('CUDA_VISIBLE_DEVICES',''):p.error('one visible device only')
    for name in ('llama_source','llama_build','sdk'):setattr(a,name,getattr(a,name).resolve(strict=True))
    env=environment(a)
    os.environ.update({k:v for k,v in env.items()if k in ('LD_LIBRARY_PATH','PPU_SDK','CUDA_HOME')})
    dev=device(a);results=[];start=time.monotonic()
    lock=Path('/tmp')/('quactlize-components-'+digest(dev['device']['pci'])[:24]+'.lock')
    try:
        with lock.open('a')as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            adapter,receipt=build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'build')
            binary=carrier(a,a.output/'build')
            for r in selected:
                old=json.loads((r['worker']/'build/adapter.json').read_text())
                check_binding(old,receipt,dev)
                old_dev=json.loads((r['worker']/'authority.json').read_text())['device']
                if old_dev['sdk']!=dev['sdk'] or old_dev['device']['compute_units']!=dev['device']['compute_units']:
                    raise ValueError('diagnostic SDK/device class differs from failed run')
            for index,(r,mode)in enumerate(jobs,1):
                row=point_request(r['row'],dev['l2']['bytes'])
                # No performance here: one replayed copy and one control allocation.
                row.update(copies=2,ring={'scope':'TWO_COPY_DIAGNOSTIC_NOT_COLD_TIMING'},input_fixture=activation(row))
                directory=a.output/(str(index)+'-'+mode);directory.mkdir()
                prior=r['worker']/'fixtures'/family(row)
                folder=prior if (prior/'weights.raw').is_file() and (prior/'weights.json').is_file() else a.output/'fixtures'/family(row)
                raw,raw_receipt=weights(row['workload'],folder)
                authority=dict(device=dev,adapter=receipt['binary_sha256'],carrier=sha(binary),
                    previous_record_sha256=r['record_sha256'],scope='DIAGNOSTIC_ONLY')
                request=dict(runtime_library=str(a.sdk/'lib/libhggc_wrapper.so'),adapter_library=str(adapter),
                    points=[row],raw_fixture=str(raw),raw_fixture_receipt=raw_receipt,authority=authority,
                    diagnostic=True,compute_override=mode)
                atomic_json(directory/'request.json',request)
                child_env=environment(a)
                if mode!='native':child_env['GGML_CUDA_CUBLAS_COMPUTE_TYPE']=mode
                log=directory/'run.log'
                print(f'NATIVE_DIAGNOSTIC_START job={index}/{len(jobs)} kind={r["kind"]} mode={mode} log={log}',flush=True)
                with log.open('x')as stream:
                    child=subprocess.Popen([str(binary),str(directory/'request.json'),str(directory)],
                        env=child_env,stdout=stream,stderr=subprocess.STDOUT)
                    last=time.monotonic()
                    while child.poll() is None:
                        time.sleep(1)
                        if time.monotonic()-last>=30:
                            print(f'NATIVE_DIAGNOSTIC_WAIT job={index}/{len(jobs)} log={log}',flush=True);last=time.monotonic()
                path=directory/(row['id']+'.diagnostic.json')
                result=dict(id=row['id'],kind=r['kind'],mode=mode,rc=child.returncode,log=str(log))
                if path.is_file():
                    data=json.loads(path.read_text());result.update(status=data['status'],
                        raw_f32_status=[v['raw_f32_status']for v in data['proof']],
                        arithmetic_status=[v['status']for v in data['proof']],
                        raw_errors=[v['raw_spot_normalized_error']for v in data['proof']],
                        replay=data['linked_backend_comparison'],
                        phase=data.get('phase'),comparisons=data.get('comparisons',{}))
                    if 'error'in data:result['error']=data['error']
                else:result['status']='INFRASTRUCTURE_FAIL'
                results.append(result);print('NATIVE_DIAGNOSTIC_RESULT '+json.dumps(result),flush=True)
    finally:
        atomic_json(a.output/'summary.json',dict(results=results,expected=len(jobs),elapsed_seconds=time.monotonic()-start,
            status='DIAGNOSTIC_COMPLETE'if len(results)==len(jobs) and all(r['rc']==0 for r in results) else'INCOMPLETE',
            performance_admitted=False,production_changed=False,previous_results_modified=False))
        print('results='+str(pack(a.output)),flush=True)
    return 0 if len(results)==len(jobs) and all(r['rc']==0 for r in results)else 1


if __name__=='__main__':raise SystemExit(main())
