#!/usr/bin/env python3
"""Bounded dense/indexed cohorts, current controls, Asys-only component means."""
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
from dev.strategy.smallm_asys import workloads,make_plan,revision,validate,check_producers,assignments,METHOD,COHORTS,MOE_COHORTS,MODEL_MOE_COHORTS
from dev.strategy.moe_model_plan import COHORT as MOE_MODEL_COHORT
from dev.strategy.moe_sweep_plan import COHORT as MOE_SWEEP_COHORT
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
    screen=None
    if point.get('config_sweep'):
        if (a.phase=='confirm') != bool(a.screen):raise ValueError('confirmation requires its fresh screen')
        if a.screen:
            screen=sealed(a.screen)
            checked_result(screen,point,run,images,screen_only=True)
    elif a.screen or a.phase!='confirm':raise ValueError('screen mode is only for the MoE config sweep')
    sdk=SDK(a.sdk);w=Workload(**point['workload']);auth=dict(run=run['sha256'],shard=shard,device=run['devices'][shard])
    with ExitStack() as stack:
        ring=Ring(sdk,w,a.l2_bytes);stack.callback(ring.close)
        if w.operation=='indexed':
            from dev.strategy.indexed import point as indexed_point
            b=Context(sdk,ring,indexed_point(w));stack.callback(b.close)
            if b.rows!=w.m*w.topk or b.ids is None:raise ValueError('indexed fixture lost routed rows/IDs')
        else:
            b=Context(sdk,ring,fixture_point(w));stack.callback(b.close);checked_fixture(b,w)
        mb=msf=sf=None
        if point['sf_configs']:
            # The bounded recovery currently has one measured SF geometry/helper.
            # Never choose a helper using an event timer under the Asys policy.
            if len(point['sf_configs'])!=1:raise ValueError('declare SF helper arms separately before extending this recovery')
            mr=Ring(sdk,w,a.l2_bytes,weights=ring.w,metadata_only=True);stack.callback(mr.close)
            mb=Context(sdk,mr,indexed_point(w) if w.operation=='indexed' else fixture_point(w));stack.callback(mb.close)
            if w.operation!='indexed':checked_fixture(mb,w)
            msf=Expansion(mb,dq);stack.callback(msf.close)
            sf=Expansion(b,dq,gold=msf.gold);stack.callback(sf.close)
        extra={}
        if point.get('config_sweep'):
            from dev.strategy import moe_sweep
            bound,bindings,queries=moe_sweep.bind(point,images)
            if screen:
                if queries!=screen['resource_queries']:raise ValueError('resources changed after screen')
                bound=moe_sweep.subset(bound,moe_sweep.finalists(bound,screen['data']))
            print('MOE_SWEEP_EXEC '+json.dumps(dict(point=point['id'],phase=a.phase,
                  arms=len(bound['arms']),resource_rejections=len(bound['resource_resolution']['rejected']))),flush=True)
            data=moe_sweep.measure(b,mb,sf,msf,bound,bindings,auth,run['source']['sha256'],phase=a.phase)
            extra=dict(execution_point=bound,resource_queries=queries,
                       screen_sha256=screen['sha256'] if screen else None)
        else:
            bound,bindings=base.bind_point(point,images)
            collector=Collector()
            data=confirm_point(b,mb,sf,msf,bound,
                dict(best_sf=point['sf_configs'][0] if sf else None),
                dict(status='READY',arms=[x['id'] for x in bound['arms']],aliases={}),
                lambda arm:bindings[arm['id']],auth,run['source']['sha256'],trace_collector=collector)
            data['resource_unavailable']=bound['resource_unavailable']
        for buf in ring.buffers:buf.guard()
    # Keep the unadmitted capture after correctness/guards/cleanup, even when
    # a host symbol check rejects it. Only the worker can publish a timed PASS.
    base.sealed(a.attempt/'capture.json',dict(point=point['id'],authority=auth,data=data,**extra))
    check_producers(bound if point.get('config_sweep') else point,data)
    return 0


def done(path,point,run):
    if not path.exists():return False
    row=sealed(path)
    if row['point']!=point['id'] or row['run']!=run['sha256']:raise ValueError('saved point belongs to another run')
    if row['status']!='PASS':return False
    images=base.read(path.parents[2]/'images.json') if point.get('config_sweep') else None
    checked_result(row,point,run,images)
    return True


def checked_result(row,point,run,images=None,*,screen_only=False):
    if row.get('status')!='PASS' or row.get('point')!=point['id'] or row.get('run')!=run['sha256']:
        raise ValueError('result is not a complete point in the current campaign')
    validate(row['data'])
    execution=point
    if point.get('config_sweep'):
        from dev.strategy.moe_sweep import verify_capture
        shard=next(i for i,ids in enumerate(run['assignments']) if point['id'] in ids)
        if row.get('shard')!=shard:raise ValueError('screen/finalists must remain on the assigned device')
        if any(base.image_identity(images,None)[kind]!=run['images'][kind] for kind in ('tc','vectors')):
            raise ValueError('sweep image receipt changed')
        screen=None
        if not screen_only:
            screen=sealed(Path(row['screen']))
            checked_result(screen,point,run,images,screen_only=True)
        execution=verify_capture(point,row,images,screen=screen)
        cu=run['devices'][shard]['device']['compute_units']
        if any(q['device']['compute_units']!=cu or q['device']['ordinal']!=0 for q in row['resource_queries'].values()):
            raise ValueError('resource resolution differs from assigned device')
    check_producers(execution,row['data'])
    for name,expected in row['evidence'].items():
        if sha(Path(name))!=expected:raise ValueError('trace evidence changed: '+name)


def measure_once(a,point,run,attempt,*,phase='confirm',screen=None):
    from dev.strategy.asys_components import parse
    from dev.strategy.asys_reparse import measurement_provenance
    from tools.run_prefill_model_asys import profile_command
    attempt.mkdir(parents=True)
    started=time.monotonic()
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(a.output),
         '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--point',point['id'],
         '--attempt',str(attempt),'--phase',phase]
    if screen:cmd+=['--screen',str(screen)]
    pci=run['devices'][a.worker]['device']['pci']
    before=base.observe(Path(run['smi']),pci,1700,advisory=True)
    execute(profile_command(run['asys'],attempt/'trace',cmd),attempt/'profile.log')
    after=base.observe(Path(run['smi']),pci,1700,advisory=True)
    report=attempt/'trace.asysrep';db=attempt/'trace.sqlite'
    execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
    captured=sealed(attempt/'capture.json')
    auth=dict(run=run['sha256'],shard=a.worker,device=run['devices'][a.worker])
    if captured['authority']!=auth or captured['point']!=point['id']:raise ValueError('capture authority differs')
    data=parse(captured['data'],db,report,measurement_provenance(run,a.worker))
    value=dict(status='PASS',point=point['id'],run=run['sha256'],shard=a.worker,data=data,
        clocks=dict(before=before,after=after),elapsed_seconds=time.monotonic()-started,
        evidence={str(p):sha(p) for p in (attempt/'capture.json',report,db)})
    if point.get('config_sweep'):
        value.update({key:captured[key] for key in ('resource_queries','execution_point','screen_sha256')})
        if screen:value['screen']=str(screen)
    checked_result(value,point,run,base.read(a.output/'images.json'),screen_only=phase=='screen')
    return value


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
            attempt=folder/'attempts'/uuid.uuid4().hex
            print('SMALLM_ASYS_POINT_START '+json.dumps(dict(point=point['id'],workload=point['workload'],
                  configs=len(point['arms']),log=str(attempt/'profile.log'))),flush=True)
            try:
                screen=None
                if point.get('config_sweep'):
                    screen=folder/'screens'/(point['id']+'.json')
                    if screen.exists():
                        checked_result(sealed(screen),point,run,base.read(a.output/'images.json'),screen_only=True)
                    else:
                        value=measure_once(a,point,run,attempt,phase='screen')
                        base.sealed(screen,value)
                    screened=sealed(screen)
                    from dev.strategy.moe_sweep import finalists
                    selected=finalists(screened['execution_point'],screened['data'])
                    print('MOE_SWEEP_SCREEN '+json.dumps(dict(point=point['id'],
                        measured=len(screened['data']['arms']),finalists=len(selected),
                        seconds=screened['elapsed_seconds'],screen=str(screen))),flush=True)
                    attempt=folder/'attempts'/uuid.uuid4().hex
                value=measure_once(a,point,run,attempt,screen=screen)
                base.sealed(path,value)
                data=value['data']
                decision=data['decision'];stats=decision['statistics']
                execution=value.get('execution_point',point)
                winning_recipe=next(x['recipe'] for x in execution['arms'] if x['id']==decision['winner'])
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
    elif a.cohort in MODEL_MOE_COHORTS:
        from dev.strategy.moe_model_plan import model_inventory, make_plan as model_plan
        if a.cohort==MOE_SWEEP_COHORT:
            from dev.strategy.moe_sweep_plan import make_plan as model_plan
        plan=model_plan(a.output,model_inventory(a));atomic_json(a.output/'plan.json',plan)
    else:
        originals=plan_smallm(a.output,[base.request(w) for w in workloads(a.cohort or 'regressions')],
                             dense_fallbacks=a.cohort not in MOE_COHORTS)
        atomic_json(a.output/'cpp-originals.json',originals)
        plan=make_plan(originals['requests'],a.cohort or 'regressions');atomic_json(a.output/'plan.json',plan)
    if a.cohort and a.cohort!=plan.get('cohort','regressions'):raise ValueError('resume cohort differs')
    if (plan['schema']!='quactlize.smallm-asys.v1' or
            any(sha(ROOT/p)!=value for p,value in plan['sources'].items())):
        raise ValueError('frozen plan source changed; start a new experiment')
    from dev.strategy.moe_model_plan import verify_inventory
    verify_inventory(a.output,plan)
    print('SMALLM_ASYS_PLAN '+json.dumps(dict(plan['counts'],timing_method=METHOD,
        scope=plan['scope'],selection='MEAN_NOT_MIN',native_comparison=False)),flush=True)
    if plan.get('cohort') in MODEL_MOE_COHORTS:
        for model in plan['coverage']['models']:
            print('MOE_MODEL_COVERAGE '+json.dumps(model),flush=True)
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
        assignments=assignments(plan,len(cards)),
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
    if plan.get('cohort') in MODEL_MOE_COHORTS:
        passed={r['point'] for r in rows if r['status']=='PASS'}
        result['models']=[dict(model=m['model'],tp=m['tp'],expected=m['points'],
            passed=sum(p['id'] in passed for p in plan['points'] if any(c['model']==m['model'] for c in p['consumers'])))
            for m in plan['coverage']['models']]
    atomic_json(a.output/'summary.json',result)
    print('SMALLM_ASYS_DONE '+json.dumps({k:v for k,v in result.items() if k!='points'}),flush=True)
    return int(result['status']!='COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--devices',default='0,1,2,3');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--cohort',choices=COHORTS)
    p.add_argument('--model-root',type=Path,default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    p.add_argument('--qwen35-model',type=Path);p.add_argument('--qwen122-model',type=Path);p.add_argument('--deepseek-model',type=Path)
    p.add_argument('--llama-source',type=Path,help='CPU-only caller TP split rules for model MoE cohorts; no binary required')
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--reparse-only',action='store_true',help='CPU only: recover timestamp-only failures without overwriting old results')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int);p.add_argument('--point');p.add_argument('--attempt',type=Path)
    p.add_argument('--phase',choices=('screen','confirm'),default='confirm');p.add_argument('--screen',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve();cards=a.devices.split(',')
    if not 1<=len(cards)<=10 or len(set(cards))!=len(cards) or any(not c.isdecimal() or str(int(c))!=c for c in cards) or not 1<=a.jobs<=64:
        p.error('invalid GPU ordinals or jobs (1..64)')
    if a.l2_bytes<=0:p.error('--l2-bytes must be a verified positive capacity')
    if sum((bool(a.point),a.worker is not None,bool(a.probe),a.reparse_only))>1:p.error('internal child modes are exclusive')
    if a.reparse_only and (a.resume or a.plan_only or a.cohort):p.error('--reparse-only uses the old sealed plan, not a new cohort or GPU resume')
    if a.point:
        if a.attempt is None:p.error('--point requires --attempt')
        a.attempt=a.attempt.resolve(strict=True)
        if not a.attempt.is_relative_to(a.output) or not a.attempt.is_dir():p.error('attempt must be inside output')
    elif a.attempt:p.error('--attempt is only valid with --point')
    if (a.screen or a.phase!='confirm') and not a.point:p.error('--phase/--screen are internal point options')
    if a.screen:
        a.screen=a.screen.resolve(strict=True)
        if not a.screen.is_relative_to(a.output):p.error('screen must belong to this experiment')
    def stop(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.point:return child(a)
    if a.worker is not None:return worker(a)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.reparse_only:
                from dev.strategy.asys_reparse import reparse
                target,result=reparse(a.output,ROOT)
                print('SMALLM_ASYS_REPARSE '+json.dumps(dict(status=result['status'],expected=result['expected'],
                    output=str(target),gpu_work=False,old_receipts_modified=False)),flush=True)
                return int(result['status']!='COMPLETE')
            return run(a)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
