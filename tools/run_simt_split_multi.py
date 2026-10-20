#!/usr/bin/env python3
"""Disjoint same-device A/B families; compile once, never compare cross-card arms."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.split_plan import plan, inventory, configs, SPLITS
from dev.gemv_simt.build_split import build
from dev.gemv_simt.build_general import inputs
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.runtime.compiler import sha
from tools.run_simt_split import cells
from tools.pack_simt_split_review import package


def family_work(spec):
    """Scheduling weight only, not a measured duration or throughput claim."""
    w=spec['workload'];nk=w['n']*w['k']/2**20
    work=1+nk*min(w['experts'],17)  # independent GGUF fixture preparation
    recipes=len(configs(w['qtype']))*len(SPLITS)
    for m in spec['numeric_m']:
        rows=m*(1 if w['operation']=='dense' else w['topk'])
        active=1 if w['operation']=='dense' else min(w['experts'],w['topk'] if w['router']=='cluster' else rows)
        work+=recipes*(1+rows*w['n']/2**16)
        if m in spec['timing_m']:work+=recipes*(1+nk*active)
    return work


def assign_families(workers):
    points=inventory()
    if not 1<=workers<=len(points):raise ValueError('invalid worker count')
    result=[[] for _ in range(workers)];loads=[0.]*workers
    for p in sorted(points,key=lambda p:(-family_work(p),p['point'])):
        shard=min(range(workers),key=lambda i:(loads[i],i))
        result[shard].append(p['point']);loads[shard]+=family_work(p)
    return result


def check_devices(probes):
    physical=[(p['device']['host'],p['device']['pci']) for p in probes]
    if not probes or len(set(physical))!=len(probes):
        raise ValueError('requested workers refer to the same physical device')
    def cohort(p):
        d=p['device']
        return (d['host'],d['compute_units'],d['warp_size'],d['max_threads'],
                d['DriverGetVersion'],d['RuntimeGetVersion'],p['l2']['bytes'],p['sdk'])
    if len({cohort(p) for p in probes})!=1:raise ValueError('mixed SDK/device capability cohort')
    return physical


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('resume binding changed: '+str(path))
    else:atomic_json(path,value)


def shared_identity(sdk,root):
    from quactlize.runtime.native import sdk_identity
    top=json.loads((root/'manifest.json').read_text());frozen=inputs();sdk_hash=sdk_identity(sdk)
    if top['object_only']:raise ValueError('shared build contains objects, not runnable libraries')
    images={}
    for q in plan()['configs']:
        path=root/q/'manifest.json';record=json.loads(path.read_text())
        if record['object_only'] or record['sources']!=frozen or record['sdk']!=sdk_hash:
            raise ValueError('shared compiler source/SDK differs')
        if {r['compute'] for r in record['records']}!={'f16','bf16'} or len(record['records'])!=2:
            raise ValueError('shared precision inventory differs')
        for r in record['records']:
            image=root/q/r['image']
            if r['configs']!=configs(int(q)) or sha(image)!=r['sha256']:raise ValueError('shared image/config differs')
            images[q+'-'+r['compute']]=r['sha256']
    return dict(manifest_sha256=sha(root/'manifest.json'),images=images)


def worker_command(a,points,output):
    cmd=[sys.executable,'-u',str(ROOT/'tools/run_simt_split.py'),'--sdk',str(a.sdk),
         '--output',str(output),'--build-root',str(a.output/'build'),
         '--points',','.join(map(str,points)),'--l2-bytes',str(a.l2_bytes),'--confirm-idle','--no-pack']
    if output.exists():cmd.append('--resume')
    if a.acu:cmd+=['--acu',str(a.acu)]
    return cmd


def aggregate(assignments,summaries,returncodes,profile):
    expected=plan();timings=[];point_ids=[];completed=0;failures=[];profiles=[]
    for shard,(ids,summary,rc) in enumerate(zip(assignments,summaries,returncodes)):
        sub=plan(ids);completed+=summary.get('completed',0)
        rows=summary.get('points',[]);point_ids.extend(p['point'] for p in rows)
        timed=summary.get('timings',[]);timings.extend(dict(t,shard=shard) for t in timed)
        want={(p['point'],m,c,s) for p in sub['points'] for m,c,s in cells(p) if m in p['timing_m']}
        actual=[(t['point'],t['m'],t['config'],t['split']) for t in timed]
        ps=summary.get('profiles',[]);profiles.extend(dict(p,shard=shard) for p in ps)
        profile_want={(p['point'],s) for p in sub['points'] for s in SPLITS
                      if profile and p['kind']=='all-format-control' and p['workload']['compute']=='bf16'
                      and p['workload']['operation']=='indexed'}
        profile_actual=[(p.get('identity',p).get('point'),p.get('identity',p).get('split')) for p in ps]
        if (rc or summary.get('status')!='PASS' or summary.get('expected')!=sub['numeric_pairs'] or
                summary.get('completed')!=sub['numeric_pairs'] or len(rows)!=len(ids) or
                {r['point'] for r in rows}!=set(ids) or len(actual)!=len(want) or set(actual)!=want or
                len(profile_actual)!=len(profile_want) or set(profile_actual)!=profile_want or
                any(p.get('status')!='PASS' for p in ps)):
            failures.append(shard)
    if len(assignments)!=len(summaries) or len(assignments)!=len(returncodes):raise ValueError('missing worker summaries')
    if len(point_ids)!=len(set(point_ids)):raise ValueError('duplicate family ownership in worker results')
    if not failures and set(point_ids)!={p['point'] for p in expected['points']}:raise ValueError('global family coverage differs')
    return dict(status='PASS' if not failures else 'INCOMPLETE',completed=completed,expected=expected['numeric_pairs'],
                timing_pairs=len(timings),expected_timing_pairs=expected['timing_pairs'],failed_shards=failures,
                returncodes=returncodes,timings=timings,profiles=profiles,
                production_changed=False,performance_admitted=False,
                scope='DISJOINT_FAMILIES_SAME_PHYSICAL_DEVICE_WITHIN_EVERY_AB_PAIR')


def run(a,cards):
    a.output.mkdir(parents=True,exist_ok=a.resume)
    assignments=assign_families(len(cards))
    frozen=plan()
    request=dict(schema='simt-split-multigpu-v1',cards=cards,assignments=assignments,
                 harness=harness_identity(ROOT),plan=frozen,acu=bool(a.acu),
                 scheduling='DETERMINISTIC_GREEDY_ESTIMATED_WORK_NOT_MEASURED_ETA')
    freeze(a.output/'multi.json',request)
    print('SPLIT_MULTI_PLAN '+json.dumps(dict(workers=len(cards),families=len(frozen['points']),
         numeric_pairs=frozen['numeric_pairs'],timing_pairs=frozen['timing_pairs'],
         shared_build=True,shards=[dict(device=card,points=ids) for card,ids in zip(cards,assignments)])),flush=True)
    if a.plan_only:return 0
    atomic_json(a.output/'summary.json',dict(status='RUNNING',production_changed=False,performance_admitted=False))
    probes=[]
    for card in cards:
        directory=a.output/('probe-'+card);directory.mkdir(exist_ok=True)
        print('SPLIT_MULTI_PROBE device='+card,flush=True)
        with (directory/'probe.log').open('a') as log:
            try:
                subprocess.run([sys.executable,str(ROOT/'tools/run_simt_split.py'),'--sdk',str(a.sdk),
                    '--output',str(directory),'--l2-bytes',str(a.l2_bytes),'--probe-only'],
                    cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'device {card} precheck failed; see {directory}/probe.log') from exc
        probes.append(json.loads((directory/'device.json').read_text()))
    check_devices(probes);freeze(a.output/'devices.json',probes)
    if not (a.output/'build').exists():
        print('SPLIT_MULTI_BUILD copies=1 workers_wait=1',flush=True)
        build(a.sdk,a.output/'build')
    if not (a.output/'build/manifest.json').is_file():raise ValueError('shared build incomplete; inspect build logs')
    freeze(a.output/'images.json',shared_identity(a.sdk,a.output/'build'))
    attempt=a.output/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
    children=[];outputs=[a.output/('gpu-'+card) for card in cards];start=time.monotonic()
    try:
        for card,ids,output in zip(cards,assignments,outputs):
            cmd=worker_command(a,ids,output);atomic_json(attempt/('gpu-'+card+'.command.json'),cmd)
            log=(attempt/('gpu-'+card+'.log')).open('x')
            try:
                child=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card,
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            finally:log.close()
            children.append(child)
        last=0
        while any(c.poll() is None for c in children):
            now=time.monotonic()
            if now-last>=30:
                completed=sum(sum(1 for _ in (o/'points').glob('*/*.json')) for o in outputs)
                print(f'SPLIT_MULTI_WAIT alive={sum(c.poll() is None for c in children)}/{len(cards)} '
                      f'pairs={completed}/{frozen["numeric_pairs"]} elapsed_minutes={(now-start)/60:.1f} logs={attempt}/gpu-*.log',flush=True)
                for card,child in zip(cards,children):
                    if child.poll() is not None:continue
                    path=attempt/('gpu-'+card+'.log')
                    with path.open('rb') as f:
                        f.seek(max(0,path.stat().st_size-8192));tail=f.read().decode(errors='replace').splitlines()
                    rows=[line for line in tail if line.startswith(('SPLIT_PROGRESS','SPLIT_ACU','SPLIT_START'))]
                    if rows:print('  device='+card+' '+rows[-1],flush=True)
                last=now
            time.sleep(1)
    except BaseException:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:child.wait()
        raise
    rc=[c.returncode for c in children];summaries=[]
    for card,output,code,probe in zip(cards,outputs,rc,probes):
        if code:
            path=attempt/('gpu-'+card+'.log')
            print('SPLIT_MULTI_WORKER_FAIL '+json.dumps(dict(device=card,rc=code,log=str(path))),flush=True)
        summary=json.loads((output/'summary.json').read_text()) if (output/'summary.json').exists() else {}
        if summary:
            authority=json.loads((output/'authority.json').read_text())
            if authority['device']!=probe or authority['harness']!=request['harness']:
                raise ValueError('worker physical device/source binding differs')
        summaries.append(summary)
    result=aggregate(assignments,summaries,rc,bool(a.acu))
    result['elapsed_seconds']=time.monotonic()-start
    atomic_json(a.output/'summary.json',result)
    print('SPLIT_MULTI_DONE '+json.dumps({k:v for k,v in result.items() if k not in ('timings','profiles')}),flush=True)
    return int(result['status']!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',default='1,2,3,4,5,6,7');p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--acu',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-idle',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();cards=a.devices.split(',')
    if any(not c.isdigit() or str(int(c))!=c for c in cards) or len(set(cards))!=len(cards):p.error('distinct device ordinals required')
    if not a.confirm_idle and not a.plan_only:p.error('confirm all requested GPUs are otherwise idle')
    def interrupted(signum,frame):
        raise KeyboardInterrupt('coordinator received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    a.sdk=a.sdk.resolve();a.output=a.output.resolve()
    # Serialize coordinators before shared build/resume, without GPU access.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(a,cards)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
