#!/usr/bin/env python3
"""Run the unchanged component experiments on independent same-model PPUs.

One subprocess at a time per physical card. A weight (all M/routes/profiles)
is the work unit, so fixture generation and dequant results are reused locally.
No compilation, cross-card normalization, or replacement of failed timings.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_kpack_cost_supplement import verify, keys, validate, same_device_class, summarize, measurement_sources
from tools.run_kpack_dequant_gate import save

RUNNER=ROOT/'tools/run_kpack_cost_supplement.py'


def work_items(points):
    grouped={}
    for p in points:grouped.setdefault(p['weight_id'],[]).append(p)
    # Long weights first; subsequent assignment is dynamic, not static modulo.
    # This is scheduling only, not a prediction used in the heuristic.
    def cost(pair):
        _,rows=pair;p=rows[0]
        return p['n']*p['k']*(p['experts']+sum(sum(r['rows']) for r in rows)/32)
    return [name for name,_ in sorted(grouped.items(),key=lambda p:(-cost(p),p[0]))]


def verify_devices(probes):
    seen=set();reference=probes[0]
    for p in probes:
        d=p['device']
        if not d.get('pci') or d['pci'] in seen:raise ValueError('workers resolve to the same/unknown physical PPU')
        if not same_device_class(reference['device'],d):raise ValueError('worker cache/CU/warp class differs')
        if p['runtime']!=reference['runtime'] or p['packages']!=reference['packages']:
            raise ValueError('worker SDK/Python packages differ')
        seen.add(d['pci'])


def command(a,output):
    return [sys.executable,'-u',str(RUNNER),'--sdk',str(a.sdk),'--bundle',str(a.bundle),
            '--output',str(output),'--phase','all','--allow-equivalent-device','--no-reuse']


def environment(device):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')


def probe(a,device):
    path=a.output/'logs'/f'probe-{device}.log'
    result=subprocess.run(command(a,a.output/'unused-probe')+['--probe-device'],
                          env=environment(device),capture_output=True,text=True)
    path.write_text(result.stdout+result.stderr)
    rows=[json.loads(s[len('COST_DEVICE '):]) for s in result.stdout.splitlines() if s.startswith('COST_DEVICE ')]
    if result.returncode or len(rows)!=1:raise ValueError(f'device {device} precheck failed; log={path}')
    return rows[0]


def collect(output,plan,assignment,root_authority):
    complete=0;missing=[];noisy=[];costs=[];origins={};devices={}
    expected={key for p in plan['points'] for key in keys(p)}|{'dense-m1-n4096-s4-recheck'}
    for name,task in assignment.items():
        directory=output/'weights'/name
        ap=directory/'authority.json'
        if not ap.exists():continue
        authority=json.loads(ap.read_text())
        if (authority['plan_sha256']!=root_authority['plan_sha256'] or
            authority['bundle_sha256']!=root_authority['bundle_sha256'] or
            authority['device']!=root_authority['probes'][str(task['device'])]['device'] or
            authority['sources']!=root_authority['sources'] or
            authority['python_packages']!=root_authority['probes'][str(task['device'])]['packages'] or
            authority['runtime']!=root_authority['runtime']):
            raise ValueError('weight provenance differs: '+name)
        ah=digest(authority)
        weight_points=[p for p in plan['points'] if p['weight_id']==name]
        # Recompute costs from checksum-bound raw component receipts. Partial
        # successes count too; a missing child summary never discards them.
        result=summarize(directory,weight_points,ah,include_reducer=name=='reducer-recheck')
        allowed=({'dense-m1-n4096-s4-recheck'} if name=='reducer-recheck' else
                 {key for p in plan['points'] if p['weight_id']==name for key in keys(p)})
        for path in sorted((directory/'components').glob('*.json')):
            key=path.stem
            if key not in allowed or key in origins:raise ValueError('foreign/duplicate component: '+key)
            row=validate(json.loads(path.read_text()),key,ah)
            if row.get('device')!=authority['device']:raise ValueError('component physical PPU differs: '+key)
            origins[key]=dict(path=str(path.relative_to(output)),sha256=sha(path),
                              authority_sha256=ah,device=authority['device'])
        devices[name]=authority['device'];noisy.extend(result['timing_unstable'])
        for row in result['costs']:
            if not row['point'].startswith(name+'-t'):raise ValueError('foreign point in weight summary')
            costs.append(row|dict(device=authority['device']))
    missing=sorted(expected-set(origins));complete=len(origins)
    result=dict(schema='quactlize.cost-supplement-multidevice.v1',
        status='COMPLETE' if not missing else 'INCOMPLETE',expected_components=len(expected),
        complete=complete,missing=missing,timing_unstable=sorted(set(noisy)),costs=costs,
        components=origins,devices=devices,assignment=assignment,
        authority_sha256=digest(root_authority),production_changed=False,
        comparison='SAME_MODEL_CROSS_CARD_ALLOWED_NO_NORMALIZATION',
        fresh_measurements=True,scope='SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E')
    save(output/'result.json',result)
    return result


def run(a):
    if len(set(a.devices))!=len(a.devices) or not a.devices or any(d<0 for d in a.devices):
        raise ValueError('devices must be distinct nonnegative ordinals')
    a.sdk=a.sdk.resolve(strict=True);a.bundle=a.bundle.resolve(strict=True);a.output=a.output.resolve()
    manifest,plan=verify(a.bundle,a.sdk)
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('logs','weights'):(a.output/name).mkdir(exist_ok=True)
    assignment_path=a.output/'assignment.json'
    assignment=json.loads(assignment_path.read_text()) if assignment_path.exists() else {}
    if a.summarize_only:
        result=collect(a.output,plan,assignment,json.loads((a.output/'authority.json').read_text()))
        return int(result['status']!='COMPLETE')
    with ThreadPoolExecutor(len(a.devices)) as pool:
        probes=list(pool.map(lambda d:probe(a,d),a.devices))
    verify_devices(probes)
    authority=dict(schema='quactlize.cost-supplement-multidevice.v1',
        bundle=str(a.bundle),bundle_sha256=sha(a.bundle/'manifest.json'),plan_sha256=plan['plan_sha256'],
        runtime=manifest['runtime'],probes={str(d):p for d,p in zip(a.devices,probes)},
        sources=measurement_sources(),
        fresh_measurements=True,production_changed=False,
        comparison='SAME_MODEL_CROSS_CARD_ALLOWED_NO_NORMALIZATION')
    ap=a.output/'authority.json'
    if ap.exists() and json.loads(ap.read_text())!=authority:raise ValueError('parallel resume authority differs')
    save(ap,authority);save(a.output/'plan.json',plan)
    names=['reducer-recheck',*work_items(plan['points'])]
    if any(n not in names or t['device'] not in a.devices for n,t in assignment.items()):
        raise ValueError('previous assignment outside this campaign')
    lock=threading.Lock();stop=threading.Event();processes={};active={};done=set();failed=[]
    pending=[n for n in names if n not in assignment]
    retained={d:[n for n in names if n in assignment and assignment[n]['device']==d] for d in a.devices}
    start=time.monotonic()

    def execute(device):
        while not stop.is_set():
            with lock:
                if retained[device]:name=retained[device].pop(0)
                elif pending:name=pending.pop(0)
                else:return
                assignment[name]=dict(device=device,status='RUNNING');active[device]=name
                save(assignment_path,assignment)
            directory=a.output/'weights'/name;log=a.output/'logs'/f'{name}.{time.time_ns()}.log'
            cmd=command(a,directory)+(['--reducer-only'] if name=='reducer-recheck' else ['--child-weight',name])
            print(f'COST_MULTI_START device={device} weight={name} log={log}',flush=True)
            before=time.monotonic();rc=-1
            try:
                with log.open('w') as stream:
                    with lock:
                        if stop.is_set():return
                        child=subprocess.Popen(cmd,env=environment(device),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                        processes[device]=child
                    rc=child.wait()
            except Exception:
                traceback.print_exc()
            finally:
                elapsed=time.monotonic()-before
                with lock:
                    processes.pop(device,None);active.pop(device,None);done.add(name)
                    if rc:failed.append(name)
                    assignment[name]=dict(device=device,status='PASS' if rc==0 else 'FAIL',rc=rc,seconds=elapsed,log=str(log.relative_to(a.output)))
                    save(assignment_path,assignment)
                print(f'COST_MULTI_WEIGHT device={device} weight={name} rc={rc} minutes={elapsed/60:.1f} remaining_continue=1',flush=True)

    def interrupt(signum,frame):
        stop.set()
        with lock:
            for child in processes.values():
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        print('COST_MULTI_STOP requested; completed components remain resumable',flush=True)

    prior={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    print(f'COST_MULTI_START workers={len(a.devices)} weights={len(names)-1} reducer=1 compilation=NONE output={a.output}',flush=True)
    try:
        with ThreadPoolExecutor(len(a.devices)) as pool:
            futures=[pool.submit(execute,d) for d in a.devices]
            while not all(f.done() for f in futures):
                wait(futures,timeout=30)
                with lock:
                    elapsed=time.monotonic()-start
                    count=sum(n!='reducer-recheck' for n in done);left=len(names)-1-count
                    # Advisory observed throughput only. Large weights run first.
                    eta=(elapsed/count*left/60) if count>=min(4,len(names)-1) else None
                    detail=','.join(f'{d}:{n}' for d,n in sorted(active.items()))
                    print(f'COST_MULTI_PROGRESS workers={len(active)}/{len(a.devices)} weights={sum(n!="reducer-recheck" for n in done)}/{len(names)-1} '
                          f'failed={len(failed)} elapsed_minutes={elapsed/60:.1f} '
                          f'remaining_minutes={eta:.1f} eta=OBSERVED_CAMPAIGN_THROUGHPUT_NOT_GUARANTEE current={detail}'
                          if eta is not None else
                          f'COST_MULTI_PROGRESS workers={len(active)}/{len(a.devices)} weights={count}/{len(names)-1} '
                          f'elapsed_minutes={elapsed/60:.1f} remaining_minutes=UNKNOWN current={detail}',flush=True)
            for f in futures:f.result()
    finally:
        for s,handler in prior.items():signal.signal(s,handler)
    result=collect(a.output,plan,assignment,authority)
    print(f'COST_MULTI_DONE status={result["status"]} components={result["complete"]}/{result["expected_components"]} '
          f'failed_weights={len(failed)} results={a.output}',flush=True)
    return int(result['status']!='COMPLETE' or stop.is_set())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--devices',type=int,nargs='+',default=list(range(8)))
    parser.add_argument('--summarize-only',action='store_true')
    try:sys.exit(run(parser.parse_args()))
    except Exception:traceback.print_exc();sys.exit(1)
