#!/usr/bin/env python3
"""Bounded dense cohorts, current controls, Asys-only component means."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.smallm_asys import workloads,make_plan,revision,validate,check_producers,METHOD
from dev.strategy.provenance import harness_identity,source_identity
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload
from quactlize.dispatch.planning import plan_smallm
from quactlize.runtime.tuning import digest
from quactlize.runtime.compiler import sha
from tools import run_decode_refresh as base
from tools.pack_simt_split_review import package


def sealed(path):
    value=base.read(path)
    if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('receipt seal differs: '+str(path))
    return value


def child(a):
    from dev.smallm_closure.bench import Context
    from dev.strategy.cold_decode import Ring,Expansion
    from dev.native_components.smallm_ab import fixture_point,checked_fixture
    from dev.strategy.confirm import confirm_point
    from dev.strategy.asys_components import Collector
    from quactlize.runtime.native import SDK
    run=sealed(a.output/'run.json');plan=sealed(a.output/'plan.json')
    point,=[p for p in plan['points'] if p['id']==a.point]
    shard=next(i for i,ids in enumerate(run['assignments']) if a.point in ids)
    if (run['plan']!=plan['sha256'] or revision(ROOT)!=run['source_revision'] or
            source_identity(ROOT)!=run['source'] or
            harness_identity(ROOT)!=run['harness'] or base.device(a)!=run['devices'][shard]):
        raise ValueError('source/device/plan changed before capture')
    images=base.read(a.output/'images.json')
    dq=Path(base.read(a.output/'dequant.json')['directory']) if (a.output/'dequant.json').exists() else None
    if base.image_identity(images,dq)!=run['images']:raise ValueError('device image binding differs')
    base.verify_images(a,plan,images,dq,[point])
    sdk=SDK(a.sdk);w=Workload(**point['workload']);auth=dict(run=run['sha256'],shard=shard,device=run['devices'][shard])
    with ExitStack() as stack:
        ring=Ring(sdk,w,a.l2_bytes);stack.callback(ring.close)
        b=Context(sdk,ring,fixture_point(w));stack.callback(b.close);checked_fixture(b,w)
        mb=msf=sf=None
        if point['sf_configs']:
            # The bounded recovery currently has one measured SF geometry/helper.
            # Never choose a helper using an event timer under the Asys policy.
            if len(point['sf_configs'])!=1:raise ValueError('declare SF helper arms separately before extending this recovery')
            mr=Ring(sdk,w,a.l2_bytes,weights=ring.w,metadata_only=True);stack.callback(mr.close)
            mb=Context(sdk,mr,fixture_point(w));stack.callback(mb.close);checked_fixture(mb,w)
            msf=Expansion(mb,dq);stack.callback(msf.close)
            sf=Expansion(b,dq,gold=msf.gold);stack.callback(sf.close)
        bound,bindings=base.bind_point(point,images)
        collector=Collector()
        data=confirm_point(b,mb,sf,msf,bound,
            dict(best_sf=point['sf_configs'][0] if sf else None),
            dict(status='READY',arms=[x['id'] for x in bound['arms']],aliases={}),
            lambda arm:bindings[arm['id']],auth,run['source']['sha256'],trace_collector=collector)
        data['resource_unavailable']=bound['resource_unavailable']
        check_producers(point,data)
        for buf in ring.buffers:buf.guard()
    base.sealed(a.attempt/'capture.json',dict(point=point['id'],authority=auth,data=data))
    return 0


def done(path,point,run):
    if not path.exists():return False
    row=sealed(path)
    if row['point']!=point['id'] or row['run']!=run['sha256']:raise ValueError('saved point belongs to another run')
    if row['status']!='PASS':return False
    validate(row['data']);check_producers(point,row['data'])
    for name,expected in row['evidence'].items():
        if sha(Path(name))!=expected:raise ValueError('trace evidence changed: '+name)
    return True


def execute(command,log):
    atomic_json(log.with_suffix('.command.json'),command)
    with log.open('x') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(log.with_suffix('.process.json'),dict(pid=p.pid,returncode=None))
        try:
            while p.poll() is None:
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:print('SMALLM_ASYS_WAIT log='+str(log),flush=True)
        finally:
            stop_process(p)
            atomic_json(log.with_suffix('.process.json'),dict(pid=p.pid,returncode=p.returncode))
    if p.returncode:raise RuntimeError(f'child rc={p.returncode}; see {log}')


def stop_process(p):
    """Only our newly created process group; never touch other GPU workloads."""
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait(timeout=15)


def worker(a):
    from dev.strategy.asys_components import parse
    from tools.run_prefill_model_asys import profile_command
    plan=sealed(a.output/'plan.json');run=sealed(a.output/'run.json')
    if not 0<=a.worker<len(run['cards']):raise ValueError('invalid worker index')
    card=run['cards'][a.worker]
    folder=a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
    if (sha(Path(run['asys']))!=run['asys_sha256'] or sha(Path(run['smi']))!=run['smi_sha256'] or
            base.device(a)!=run['devices'][a.worker] or harness_identity(ROOT)!=run['harness']):
        raise ValueError('worker profiler/source/device differs')
    failures=0
    pci=run['devices'][a.worker]['device']['pci']
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in plan['points']:
            if point['id'] not in run['assignments'][a.worker]:continue
            path=folder/'points'/(point['id']+'.json')
            if done(path,point,run):continue
            attempt=folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            started=time.monotonic()
            print('SMALLM_ASYS_POINT_START '+json.dumps(dict(point=point['id'],workload=point['workload'],
                  configs=len(point['arms']),log=str(attempt/'profile.log'))),flush=True)
            try:
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(a.output),
                     '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--point',point['id'],'--attempt',str(attempt)]
                before=base.observe(Path(run['smi']),pci,1700,advisory=True)
                execute(profile_command(run['asys'],attempt/'trace',cmd),attempt/'profile.log')
                after=base.observe(Path(run['smi']),pci,1700,advisory=True)
                report=attempt/'trace.asysrep';db=attempt/'trace.sqlite'
                execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
                captured=sealed(attempt/'capture.json')
                auth=dict(run=run['sha256'],shard=a.worker,device=run['devices'][a.worker])
                if captured['authority']!=auth or captured['point']!=point['id']:raise ValueError('capture authority differs')
                provenance=dict(run['source_revision'],harness_sha256=run['harness']['sha256'],
                    source_sha256=run['source']['sha256'],
                    images_sha256=digest(run['images']),sdk=run['sdk'],device=run['devices'][a.worker],
                    asys_sha256=run['asys_sha256'],python_version=run['python_version'])
                data=parse(captured['data'],db,report,provenance);check_producers(point,data)
                value=dict(status='PASS',point=point['id'],run=run['sha256'],data=data,
                    clocks=dict(before=before,after=after),elapsed_seconds=time.monotonic()-started,
                    evidence={str(p):sha(p) for p in (attempt/'capture.json',report,db)})
                base.sealed(path,value)
                decision=data['decision'];stats=decision['statistics']
                winning_recipe=next(x['recipe'] for x in point['arms'] if x['id']==decision['winner'])
                print('SMALLM_ASYS_POINT '+json.dumps(dict(point=point['id'],status='PASS',
                    qtype=point['workload']['qtype'],shape=[point['workload'][k] for k in ('m','n','k')],
                    original_us=stats['original']['mean_us'],best_us=stats[decision['winner']]['mean_us'],
                    delta_pct=decision['delta_pct'],winner=decision['winner'],
                    recipe=winning_recipe,
                    quactlize_commit=run['source_revision']['quactlize_commit'])),flush=True)
            except Exception as exc:
                traceback.print_exc();failures+=1
                failure=dict(status='FAIL',point=point['id'],run=run['sha256'],error=str(exc),attempt=str(attempt))
                base.sealed(attempt/'failure.json',failure);base.sealed(path,failure)
    return int(bool(failures))


def run(a):
    from quactlize.runtime.native import sdk_identity
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if (a.output/'plan.json').exists():plan=sealed(a.output/'plan.json')
    else:
        originals=plan_smallm(a.output,[base.request(w) for w in workloads(a.cohort or 'regressions')],dense_fallbacks=True)
        atomic_json(a.output/'cpp-originals.json',originals)
        plan=make_plan(originals['requests'],a.cohort or 'regressions');atomic_json(a.output/'plan.json',plan)
    if a.cohort and a.cohort!=plan.get('cohort','regressions'):raise ValueError('resume cohort differs')
    if (plan['schema']!='quactlize.smallm-asys.v1' or
            any(sha(ROOT/p)!=value for p,value in plan['sources'].items())):
        raise ValueError('frozen plan source changed; start a new experiment')
    print('SMALLM_ASYS_PLAN '+json.dumps(dict(plan['counts'],timing_method=METHOD,
        scope=plan['scope'],selection='MEAN_NOT_MIN',native_comparison=False)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('confirm the assigned cards are otherwise idle')
    rev=revision(ROOT);a.smi=base.selected_smi(a)
    asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    cards=a.devices.split(',');devices=[]
    for card in cards:
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),
            '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--probe',str(path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),check=True)
        devices.append(base.read(path))
    base.check_devices(devices)
    inputs=dict(plan=plan['sha256'],source_revision=rev,harness=harness_identity(ROOT),source=source_identity(ROOT),
        sdk=sdk_identity(a.sdk),cards=cards,devices=devices,l2_bytes=a.l2_bytes,
        assignments=[[p['id'] for j,p in enumerate(plan['points']) if j%len(cards)==i] for i in range(len(cards))],
        asys=str(asys),asys_sha256=sha(asys),smi=str(a.smi),smi_sha256=sha(a.smi),
        python_version=sys.version,timing_method=METHOD)
    base.freeze(a.output/'inputs.json',inputs)
    images,dq=base.compile_all(a,plan)
    run=dict(inputs,images=base.image_identity(images,dq));run['sha256']=digest(run)
    base.freeze(a.output/'run.json',run)
    a.expected_ce_ghz=1.7;base.lock_clocks(a,devices)
    workers=[];started=time.monotonic();logs=a.output/'workers'/uuid.uuid4().hex;logs.mkdir(parents=True)
    try:
        for i,card in enumerate(cards):
            with (logs/('gpu-'+card+'.log')).open('x') as log:
                workers.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(i)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in workers):
            print('SMALLM_ASYS_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in workers),
                  elapsed_minutes=(time.monotonic()-started)/60,logs=str(logs/'gpu-*.log'))),flush=True)
            for p in workers:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p in workers:stop_process(p)
    rows=[]
    for i,ids in enumerate(run['assignments']):
        for pid in ids:
            path=a.output/('gpu-'+cards[i])/'points'/(pid+'.json')
            row=sealed(path) if path.exists() else dict(status='MISSING',point=pid)
            if row['status']=='PASS':done(path,next(p for p in plan['points'] if p['id']==pid),run)
            rows.append(dict(point=pid,status=row['status'],decision=row.get('data',{}).get('decision'),error=row.get('error')))
    result=dict(status='COMPLETE' if all(r['status']=='PASS' for r in rows) and all(p.returncode==0 for p in workers) else 'INCOMPLETE',
        expected=len(plan['points']),points=rows,returncodes=[p.returncode for p in workers],timing_method=METHOD,
        source_revision=rev,production_changed=False,native_no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',result)
    print('SMALLM_ASYS_DONE '+json.dumps({k:v for k,v in result.items() if k!='points'}),flush=True)
    return int(result['status']!='COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--devices',default='0,1,2,3');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--cohort',choices=('regressions','q6-token-cta'))
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int);p.add_argument('--point');p.add_argument('--attempt',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve();cards=a.devices.split(',')
    if not 1<=len(cards)<=10 or len(set(cards))!=len(cards) or any(not c.isdecimal() or str(int(c))!=c for c in cards) or not 1<=a.jobs<=64:
        p.error('invalid GPU ordinals or jobs (1..64)')
    if a.l2_bytes<=0:p.error('--l2-bytes must be a verified positive capacity')
    if sum((bool(a.point),a.worker is not None,bool(a.probe)))>1:p.error('internal child modes are exclusive')
    if a.point:
        if a.attempt is None:p.error('--point requires --attempt')
        a.attempt=a.attempt.resolve(strict=True)
        if not a.attempt.is_relative_to(a.output) or not a.attempt.is_dir():p.error('attempt must be inside output')
    elif a.attempt:p.error('--attempt is only valid with --point')
    def stop(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.point:return child(a)
    if a.worker is not None:return worker(a)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(a)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
