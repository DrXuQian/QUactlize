#!/usr/bin/env python3
"""Fresh bounded model-local MoE prefill Asys screen/confirmation.

Reuses the passed pilot's general BF16 endpoints and GPU routing. It never
changes production routing or claims native/model parity from this sweep.
"""
import argparse
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
from dev.strategy import moe_prefill_campaign as campaign,moe_model_plan as models
from dev.strategy.moe_prefill_search import measure,parse,compose
from dev.strategy.journal import atomic_json
from dev.prefill_asys.trace import activities,runtime_activities
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.run_smallm_regressions_asys import sealed,execute,stop_process
from tools.run_prefill_model_asys import profile_command
from tools.run_moe_prefill_pilot import prepare_run,verify_run
from tools.pack_simt_split_review import package


def cache_folder(folder,phase,point):
    w=point['workload']
    family=digest([w[k] for k in ('qtype','n','k','experts','channels','compute','storage')])
    return folder/'components'/phase/family


def component_cache(folder,run,shard,phase,point):
    out={}
    for path in cache_folder(folder,phase,point).glob('*.json'):
        value=sealed(path);row=value['record'];cid=row['component_id']
        auth=dict(run=run['sha256'],shard=shard,device=run['devices'][shard],phase=phase)
        if (path.stem!=cid or digest(row['identity'])!=cid or row['identity']['authority']!=auth or
                row['latency_provenance']['quactlize_commit']!=run['source_revision']['quactlize_commit'] or
                row['latency_provenance']['source_sha256']!=run['source']['sha256'] or
                row['latency_provenance']['harness_sha256']!=run['harness']['sha256']):
            raise ValueError('shared component source/device/phase changed')
        out[cid]=row
    return out


def child(a):
    from dev.strategy.isolated import check_sdk_graph_abi
    run=sealed(a.output/'run.json');verify_run(a,run,a.worker)
    job=sealed(a.job);point=job['point'];phase=job['phase']
    if (job['run']!=run['sha256'] or point['id'] not in run['assignments'][a.worker] or
            phase not in ('screen','confirm')):raise ValueError('batch job authority differs')
    check_sdk_graph_abi(a.sdk)
    if job['operation']=='resources':
        resolved,queries=campaign.bind(point,base.read(a.output/'images.json'))
        base.sealed(a.attempt/'resources.json',dict(run=run['sha256'],point=resolved,queries=queries))
        return 0
    if job['operation']!='measure':raise ValueError('unknown prefill job')
    folder=a.output/('gpu-'+run['cards'][a.worker])
    known=component_cache(folder,run,a.worker,phase,point)
    selected={arm['id']:arm for arm in point['arms']}
    arms=[selected[i] for i in job['arms']]
    captured=measure(a,point,arms,base.read(a.output/'images.json'),job['queries'],run,phase,known)
    base.sealed(a.attempt/'capture.json',dict(captured,job=job['sha256']))
    return 0


def saved(path,run):
    if not path.exists():return None
    value=sealed(path)
    if value.get('run')!=run['sha256']:raise ValueError('saved batch belongs to another run')
    if value.get('status')!='PASS':return None
    for name,want in value['evidence'].items():
        if sha(Path(name))!=want:raise ValueError('saved batch evidence changed: '+name)
    return value


def batches(point):
    # Full owns a provider graph pool and a separate reset timer. Keeping it
    # alone also makes the fail-fast gate an exact reusable normal batch.
    pending=[]
    for arm in point['arms']:
        if arm['recipe']['kind']=='full':
            if pending:yield pending;pending=[]
            yield [arm['id']]
        else:
            pending.append(arm['id'])
            if len(pending)==campaign.PROTOCOL['live_arms']:yield pending;pending=[]
    if pending:yield pending


def batch_path(folder,pid,phase,arms):
    return folder/'batches'/phase/(digest([pid,phase,arms])+'.json')


def completed(path,run):
    if not path.exists():return None
    value=sealed(path)
    if value['run']!=run['sha256'] or value['point']!=path.stem:
        raise ValueError('completed point belongs to another run/workload')
    if value['status']!='PASS':return None
    for name,want in value['batch_receipts'].items():
        if sha(name)!=want or saved(Path(name),run) is None:
            raise ValueError('completed point lost its raw batch evidence: '+name)
    screen=sealed(Path(value['screen']))
    if screen['sha256']!=value['screen_sha256'] or screen['run']!=run['sha256']:
        raise ValueError('completed point screen changed')
    return value


def invocation(a,run,job,*,profile,entrypoint=None):
    folder=a.output/('gpu-'+run['cards'][a.worker]);attempt=folder/'attempts'/uuid.uuid4().hex
    attempt.mkdir(parents=True);base.sealed(attempt/'job.json',dict(job,run=run['sha256']))
    command=[sys.executable,'-u',str(Path(entrypoint or __file__).resolve()),'--output',str(a.output),
        '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(a.worker),
        '--job',str(attempt/'job.json'),'--attempt',str(attempt)]
    pci=run['devices'][a.worker]['device']['pci'];before=base.observe(Path(run['smi']),pci,1700,advisory=True)
    try:
        execute(profile_command(run['asys'],attempt/'trace',command) if profile else command,attempt/'profile.log')
    except RuntimeError as exc:
        if entrypoint:
            detail='\n'.join((attempt/'profile.log').read_text(errors='replace').splitlines()[-25:])
            raise RuntimeError(str(exc)+'\n'+detail) from exc
        raise
    after=base.observe(Path(run['smi']),pci,1700,advisory=True)
    if not profile:return attempt,sealed(attempt/'resources.json')
    report,db=attempt/'trace.asysrep',attempt/'trace.sqlite'
    execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
    capture=sealed(attempt/'capture.json');request=sealed(attempt/'job.json')
    if (capture['job']!=request['sha256'] or capture['point']!=job['point']['id'] or
            capture['phase']!=job['phase'] or set(capture['arms'])!=set(job['arms']) or
            capture['authority']!=dict(run=run['sha256'],shard=a.worker,
                device=run['devices'][a.worker],phase=job['phase'])):
        raise ValueError('profile capture job differs')
    known=component_cache(folder,run,a.worker,job['phase'],job['point'])
    provenance=dict(run['source_revision'],source_sha256=run['source']['sha256'],
        harness_sha256=run['harness']['sha256'],images_sha256=digest(run['images']),
        device=run['devices'][a.worker],sdk=run['sdk'],asys_sha256=run['asys_sha256'],
        trace_sha256=sha(report),sqlite_sha256=sha(db),timing_method='ASYS_GPU_ACTIVITY')
    data=parse(capture,known,activities(db,launch_shared=True),runtime_activities(db),provenance)
    proof={str(p):sha(p) for p in (attempt/'job.json',attempt/'capture.json',report,db)}
    # Timings become reusable only after the child has closed all owners and
    # passed its final precision/guard checks and all raw intervals parsed.
    for cid,row in data['component_records'].items():
        if cid in known or row['kind']=='producer':continue
        base.sealed(cache_folder(folder,job['phase'],job['point'])/(cid+'.json'),
                    dict(record=row,run=run['sha256'],evidence=proof))
    return attempt,dict(status='PASS',run=run['sha256'],point=job['point']['id'],phase=job['phase'],
        data=data,clocks=dict(before=before,after=after),evidence=proof)


def resource(a,run,point):
    folder=a.output/('gpu-'+run['cards'][a.worker]);path=folder/'resources'/(point['id']+'.json')
    if path.exists():
        value=sealed(path)
        if value['run']!=run['sha256'] or value['point']!=campaign.resolve(point,value['queries'],base.read(a.output/'images.json')):
            raise ValueError('saved resource resolution changed')
        return value
    _,value=invocation(a,run,dict(operation='resources',point=point,phase='screen'),profile=False)
    base.sealed(path,value)
    return value


def phase(a,run,point,queries,which,deadline):
    folder=a.output/('gpu-'+run['cards'][a.worker]);results=[]
    first=0
    for arms in batches(point):
        path=batch_path(folder,point['id'],which,arms)
        value=saved(path,run)
        if value is not None and (value['point']!=point['id'] or value['phase']!=which or
                                  set(value['data']['arms'])!=set(arms)):
            raise ValueError('saved batch requested arms differ')
        if value is None:
            if time.monotonic()>=deadline:return None
            start=time.monotonic()
            print('MOE_PREFILL_BATCH_START '+json.dumps(dict(point=point['id'],phase=which,
                first=first,arms=len(arms),total=len(point['arms']))),flush=True)
            try:
                _,value=invocation(a,run,dict(operation='measure',phase=which,point=point,arms=arms,queries=queries),profile=True)
                value['elapsed_seconds']=time.monotonic()-start;base.sealed(path,value)
            except Exception as exc:
                # Preserve this failure; only an explicit resume retries it.
                base.sealed(path,dict(status='FAIL',run=run['sha256'],point=point['id'],error=str(exc)))
                raise
            print('MOE_PREFILL_BATCH_DONE '+json.dumps(dict(point=point['id'],phase=which,
                completed=min(first+len(arms),len(point['arms'])),total=len(point['arms']),
                seconds=value['elapsed_seconds'])),flush=True)
        results.append(value['data'])
        first+=len(arms)
    return compose(point,results)


def worker(a):
    plan,run=(sealed(a.output/n) for n in ('plan.json','run.json'));verify_run(a,run,a.worker)
    folder=a.output/('gpu-'+run['cards'][a.worker]);folder.mkdir(exist_ok=True)
    pci=run['devices'][a.worker]['device']['pci'];failed=0;done=0;budget=False
    deadline=time.monotonic()+a.budget_hours*3600
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for pid in run['assignments'][a.worker]:
            p=next(p for p in plan['points'] if p['id']==pid);path=folder/'points'/(pid+'.json')
            if completed(path,run):done+=1;continue
            if time.monotonic()>=deadline:budget=True;break
            try:
                resolved=resource(a,run,p);point=resolved['point'];queries=resolved['queries']
                screen=phase(a,run,point,queries,'screen',deadline)
                if screen is None:budget=True;break
                screen_path=folder/'screens'/(pid+'.json')
                base.sealed(screen_path,dict(screen,run=run['sha256']))
                ids=campaign.finalists(point,screen['decision']['statistics'])
                final=dict(point,arms=[next(x for x in point['arms'] if x['id']==aid) for aid in ids])
                confirmed=phase(a,run,final,queries,'confirm',deadline)
                if confirmed is None:budget=True;break
                receipts=[batch_path(folder,pid,which,ids) for which,p in
                          (('screen',point),('confirm',final)) for ids in batches(p)]
                base.sealed(path,dict(confirmed,run=run['sha256'],resolved_sha256=point['sha256'],
                    screen=str(screen_path),screen_sha256=sealed(screen_path)['sha256'],finalists=ids,
                    batch_receipts={str(p):sha(p) for p in receipts}))
                done+=1
                d=confirmed['decision'];s=d['statistics']
                print('MOE_PREFILL_POINT_DONE '+json.dumps(dict(point=pid,workload=p['workload'],
                    screen_arms=len(point['arms']),finalists=len(ids),original_us=s['original']['mean_us'],
                    best_us=s[d['winner']]['mean_us'],delta_pct=d['delta_pct'],winner=d['winner'],
                    completed=done,total=len(run['assignments'][a.worker]))),flush=True)
            except Exception as exc:
                failed+=1;traceback.print_exc()
                base.sealed(path,dict(status='FAIL',run=run['sha256'],point=pid,error=str(exc)))
                # Infrastructure can poison a context; each batch is a new
                # process. Do not stop independent points after one failure.
    atomic_json(folder/'status.json',dict(completed=done,failed=failed,budget_reached=budget,
                status='INCOMPLETE' if failed or budget else 'PASS'))
    return int(bool(failed or budget))


def run(a):
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if (a.output/'plan.json').exists():plan=sealed(a.output/'plan.json')
    else:
        audit,plan=campaign.make_plan(a.output,models.model_inventory(a,ms=campaign.audit_plan.MS))
        base.freeze(a.output/'audit.json',audit);base.freeze(a.output/'plan.json',plan)
    models.verify_inventory(a.output,dict(cohort=campaign.audit_plan.COHORT,coverage=plan['coverage']),a.moe_models)
    print('MOE_PREFILL_SWEEP_PLAN '+json.dumps(dict(plan['counts'],protocol=plan['protocol'],
          budget_hours=a.budget_hours,production_changed=False)),flush=True)
    if a.plan_only:return 0
    run=prepare_run(a,plan,script=__file__,assign=campaign.assignments)
    # Fail fast on the new Full reset-activity accounting before launching
    # hundreds of profiles. Retain this successful screen batch for the run.
    probe_point=next(p for p in plan['points'] if p['workload']['m']==128 and p['workload']['router']=='spread')
    shard=next(i for i,ids in enumerate(run['assignments']) if probe_point['id'] in ids)
    gate=a.output/'preflight.json'
    if gate.exists():
        value=sealed(gate)
        if value['status']!='PASS' or value['run']!=run['sha256']:
            raise ValueError('preflight run identity differs')
    else:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),
            '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(shard),'--preflight',probe_point['id']],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=run['cards'][shard]),check=True)
    logs=a.output/'workers'/uuid.uuid4().hex;logs.mkdir(parents=True);workers=[];started=time.monotonic()
    try:
        for i,card in enumerate(run['cards']):
            with (logs/('gpu-'+card+'.log')).open('x') as log:
                workers.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),
                    '--worker',str(i),'--budget-hours',str(a.budget_hours)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in workers):
            print('MOE_PREFILL_SWEEP_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in workers),
                minutes=(time.monotonic()-started)/60,logs=str(logs/'gpu-*.log'))),flush=True)
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
            path=a.output/('gpu-'+run['cards'][i])/'points'/(pid+'.json')
            value=sealed(path) if path.exists() else dict(status='MISSING',point=pid)
            rows.append(dict(point=pid,status=value['status'],decision=value.get('decision'),
                             bindings=value.get('bindings'),error=value.get('error')))
    result=dict(status='COMPLETE' if all(r['status']=='PASS' for r in rows) and
                all(p.returncode==0 for p in workers) else 'INCOMPLETE',expected=len(rows),
                passed=sum(r['status']=='PASS' for r in rows),returncodes=[p.returncode for p in workers],
                points=rows,scope=plan['scope'],source_revision=run['source_revision'],
                production_changed=False,native_no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',result)
    print('MOE_PREFILL_SWEEP_DONE '+json.dumps({k:v for k,v in result.items() if k!='points'}),flush=True)
    return int(result['status']!='COMPLETE')


def preflight(a):
    plan,run=(sealed(a.output/n) for n in ('plan.json','run.json'));verify_run(a,run,a.worker)
    point=next(p for p in plan['points'] if p['id']==a.preflight)
    pci=run['devices'][a.worker]['device']['pci']
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resolved=resource(a,run,point);p=resolved['point']
        # Include the exact normal singleton batch containing Full, so no
        # expensive Full precision/timer job is duplicated after preflight.
        aid=next(arm['id'] for arm in p['arms'] if arm['recipe']['kind']=='full')
        arms=next(ids for ids in batches(p) if aid in ids)
        _,value=invocation(a,run,dict(operation='measure',point=p,arms=arms,
            queries=resolved['queries'],phase='screen'),profile=True)
        folder=a.output/('gpu-'+run['cards'][a.worker])
        base.sealed(batch_path(folder,p['id'],'screen',arms),value)
        base.sealed(a.output/'preflight.json',dict(status='PASS',run=run['sha256'],
            point=p['id'],reset=value['data']['reset'],scope='FULL_RESET_CAPTURE_GATE_NOT_CAMPAIGN'))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--llama-source',type=Path);p.add_argument('--llama-build',type=Path)
    p.add_argument('--model-root',type=Path,default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    p.add_argument('--moe-models');p.add_argument('--qwen35-model',type=Path);p.add_argument('--qwen122-model',type=Path)
    p.add_argument('--deepseek-model',type=Path);p.add_argument('--devices',default='3,4,5,6,7');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--budget-hours',type=float,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int);p.add_argument('--job',type=Path)
    p.add_argument('--attempt',type=Path);p.add_argument('--preflight')
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    cards=a.devices.split(',')
    if (not 1<=len(cards)<=8 or len(set(cards))!=len(cards) or any(not c.isdecimal() or str(int(c))!=c for c in cards)
            or not 1<=a.jobs<=64 or a.l2_bytes<=0 or not 0<a.budget_hours<=24):p.error('invalid GPU/jobs/L2/budget')
    if bool(a.job)!=bool(a.attempt) or a.job and a.worker is None:p.error('incomplete batch child arguments')
    if a.attempt and not a.attempt.resolve().is_relative_to(a.output):p.error('attempt outside run')
    def stop(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,stop)
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.job:return child(a)
    if a.preflight:return preflight(a)
    if a.worker is not None:return worker(a)
    if not a.resume and a.moe_models is None:a.moe_models='qwen35-35b-a3b,qwen35-122b-a10b'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(a)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
