#!/usr/bin/env python3
"""Fixed MoE shortlist, public Quactlize preparation, isolated Asys confirmation.

No llama build, new config expansion, historical latency reuse or model claim.
Run --plan first for the exact denominator. --resume retains successful batches.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.moe_confirm import plan as plans
from dev.moe_confirm.build import build as build_moe,verify as verify_moe
from dev.strategy.journal import atomic_json
from dev.strategy.moe_prefill_campaign import assignments
from dev.strategy.moe_prefill_search import measure,compose
from dev.strategy.provenance import source_identity,harness_identity
from dev.strategy.smallm_asys import revision
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base,run_moe_prefill_sweep as engine
from tools.run_smallm_regressions_asys import sealed,stop_process
from tools.pack_simt_split_review import package


def verify(a,run,worker=None):
    if (run['source_revision']!=revision(ROOT) or run['source']!=source_identity(ROOT) or
            run['harness']!=harness_identity(ROOT) or run['plan']!=sealed(a.output/'plan.json')['sha256'] or
            run['l2_bytes']!=a.l2_bytes):raise ValueError('confirmation source/plan changed')
    for name,want in run['immutable_files'].items():
        if sha(name)!=want:raise ValueError('confirmation artifact changed: '+name)
    verify_moe(run['moe_helpers'])
    if worker is not None and run['devices'][worker]!=base.device(a):
        raise ValueError('assigned physical device/SDK changed')


def child(a):
    from dev.moe_confirm import backend
    from dev.strategy.isolated import check_sdk_graph_abi
    run=sealed(a.output/'run.json');verify(a,run,a.worker)
    if a.job.resolve().parent!=a.attempt.resolve() or a.output not in a.attempt.resolve().parents:
        raise ValueError('child job must stay inside its run directory')
    job=sealed(a.job);point=job['point']
    if job['run']!=run['sha256'] or point['id'] not in run['assignments'][a.worker]:
        raise ValueError('confirmation job is outside its assigned point set')
    images=base.read(a.output/'images.json')
    if job['operation']=='resources':
        point,queries=plans.bind(point,images)
        base.sealed(a.attempt/'resources.json',dict(run=run['sha256'],point=point,queries=queries));return 0
    if job['operation']!='measure' or job['phase']!='confirm':raise ValueError('invalid confirmation job')
    check_sdk_graph_abi(a.sdk)
    folder=a.output/('gpu-'+run['cards'][a.worker])
    known=engine.component_cache(folder,run,a.worker,'confirm',point)
    by_id={r['id']:r for r in point['arms']}
    captured=measure(a,point,[by_id[i] for i in job['arms']],images,job['queries'],run,'confirm',known,backend=backend)
    base.sealed(a.attempt/'capture.json',dict(captured,job=job['sha256']));return 0


def completed(path,run):
    if not path.exists():return None
    value=sealed(path)
    if value['run']!=run['sha256'] or value['point']!=path.stem:raise ValueError('point binding changed')
    if value['status']!='PASS':return None
    for name,want in value['batch_receipts'].items():
        if sha(name)!=want or engine.saved(Path(name),run) is None:raise ValueError('point lost batch evidence')
    return value


def point(a,run,p,deadline):
    folder=a.output/('gpu-'+run['cards'][a.worker]);path=folder/'points'/(p['id']+'.json')
    if completed(path,run):return True
    if time.monotonic()>=deadline:return False
    start=time.monotonic();resource=folder/'resources'/(p['id']+'.json')
    if resource.exists():resolved=sealed(resource)
    else:
        _,resolved=engine.invocation(a,run,dict(operation='resources',point=p,phase='confirm'),
                                     profile=False,entrypoint=__file__)
        base.sealed(resource,resolved)
    if resolved['run']!=run['sha256']:raise ValueError('resource belongs to a different confirmation')
    p=resolved['point'];results=[];receipts=[];done=0
    for ids in engine.batches(p):
        where=engine.batch_path(folder,p['id'],'confirm',ids);value=engine.saved(where,run)
        if value is None:
            if time.monotonic()>=deadline:return False
            print('MOE_CONFIRM_BATCH '+json.dumps(dict(point=p['id'],m=p['workload']['m'],
                first=done,arms=len(ids),total=len(p['arms']))),flush=True)
            _,value=engine.invocation(a,run,dict(operation='measure',point=p,phase='confirm',
                arms=ids,queries=resolved['queries']),profile=True,entrypoint=__file__)
            base.sealed(where,value)
        if value['point']!=p['id'] or set(value['data']['arms'])!=set(ids):raise ValueError('saved arms differ')
        results.append(value['data']);receipts.append(where);done+=len(ids)
    data=compose(p,results,explicit_helpers=True)
    base.sealed(path,dict(data,run=run['sha256'],elapsed_seconds=time.monotonic()-start,
        batch_receipts={str(x):sha(x) for x in receipts}))
    s=data['decision']['statistics'];win=data['decision']['winner']
    print('MOE_CONFIRM_POINT '+json.dumps(dict(point=p['id'],workload=p['workload'],arms=len(p['arms']),
        original_us=s['original']['mean_us'],best_us=s[win]['mean_us'],winner=win,
        seconds=time.monotonic()-start)),flush=True)
    return True


def worker(a):
    plan,run=(sealed(a.output/p) for p in ('plan.json','run.json'));verify(a,run,a.worker)
    folder=a.output/('gpu-'+run['cards'][a.worker]);folder.mkdir(exist_ok=True)
    points={p['id']:p for p in plan['points']};deadline=time.monotonic()+a.budget_hours*3600
    if a.preflight and a.preflight not in run['assignments'][a.worker]:
        raise ValueError('preflight point is not assigned to this device')
    failed=0;done=0;budget=False;pci=run['devices'][a.worker]['device']['pci']
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for pid in run['assignments'][a.worker]:
            if a.preflight and pid!=a.preflight:continue
            try:
                if not point(a,run,points[pid],deadline):budget=True;break
                done+=1
            except Exception as exc:
                failed+=1;traceback.print_exc()
                base.sealed(folder/'points'/(pid+'.json'),dict(status='FAIL',point=pid,run=run['sha256'],error=str(exc)))
                print('MOE_CONFIRM_FAIL '+json.dumps(dict(point=pid,error=str(exc))),flush=True)
    atomic_json(folder/('preflight-status.json' if a.preflight else 'status.json'),
        dict(completed=done,failed=failed,budget_reached=budget))
    return int(bool(failed or budget))


def build(a,plan):
    from dev.strategy.prefill_build import build as build_prefill
    images,dq=base.compile_all(a,plan)
    build_prefill(a.sdk,a.output/'helpers')
    helpers=build_moe(a.sdk,a.output/'moe-helpers')
    devices=[]
    for card in a.devices.split(','):
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,__file__,'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),
            '--probe',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),check=True)
        devices.append(base.read(path))
    base.check_devices(devices)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True);a.smi=base.selected_smi(a)
    files=[a.asys,a.smi,Path(helpers['bridge']),Path(helpers['scaffold']['candidate']),
           a.output/'helpers/manifest.json',a.output/'helpers/libstrategy_prefill.so',
           a.output/'images.json',a.output/'moe-helpers/receipt.json']
    files += [Path(r['path']) for pool in images.values() for r in pool.values()]
    if dq:files += [dq/'manifest.json',dq/'libquactlize_ppu_dequant.so']
    run=dict(plan=plan['sha256'],source_revision=revision(ROOT),source=source_identity(ROOT),
        harness=harness_identity(ROOT),l2_bytes=a.l2_bytes,cards=a.devices.split(','),devices=devices,
        assignments=assignments(plan,len(devices)),adapter_sha256=helpers['sha256'],moe_helpers=helpers,
        helpers=str(a.output/'helpers'),dequant=str(dq) if dq else None,images=base.image_identity(images,dq),
        asys=str(a.asys),asys_sha256=sha(a.asys),smi=str(a.smi),sdk=devices[0]['sdk'],
        immutable_files={str(p):sha(p) for p in files},mode=a.mode,native_comparison=False,
        scope=plan['scope'])
    base.sealed(a.output/'run.json',run);return sealed(a.output/'run.json')


def helper_gate(a):
    from dev.moe_scaffold.bench import measure as helper_measure
    run=sealed(a.output/'run.json');verify(a,run,0)
    # Untimed existing two-stream/changed-ID graph oracle, not another sweep.
    pci=run['devices'][0]['device']['pci']
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=helper_measure(a.sdk,run['moe_helpers']['scaffold'],[],smi=Path(run['smi']))
    base.sealed(a.output/'helper-gate.json',dict(status='PASS',run=run['sha256'],device=data['device'],
        scope='UNTIMED_PUBLIC_HELPER_DYNAMIC_BINDING_AND_GRAPH_GATE',performance_admitted=False));return 0


def checked_child(command,path,card):
    """Fail before the campaign, with the useful error in the top-level log."""
    with path.open('w') as log:
        p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
    started=time.monotonic()
    try:
        while p.poll() is None:
            print('MOE_CONFIRM_PREFLIGHT_WAIT '+json.dumps(dict(minutes=round((time.monotonic()-started)/60,1),
                log=str(path))),flush=True)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
    finally:stop_process(p)
    if p.returncode:
        raise RuntimeError(f'preflight rc={p.returncode}; {path}\n'+
                           '\n'.join(path.read_text(errors='replace').splitlines()[-25:]))


def review(a,plan,run,codes):
    rows=[];groups=defaultdict(list)
    for worker,ids in enumerate(run['assignments']):
        for pid in ids:
            path=a.output/('gpu-'+run['cards'][worker])/'points'/(pid+'.json')
            value=completed(path,run)
            rows.append(dict(point=pid,status='PASS' if value else 'MISSING_OR_FAILED'))
            if value:groups[plans.key(value['workload'])].append(value)
    selected=[];needed=defaultdict(int)
    for p in plan['points']:needed[plans.key(p['workload'])]+=1
    for host,count in sorted(needed.items()):
        profiles=groups[host]
        if len(profiles)!=count:continue
        chosen=plans.choose(profiles)
        bindings=[p['bindings'][chosen] for p in profiles]
        if any(b!=bindings[0] for b in bindings):raise ValueError('chosen configuration depends on router')
        selected.append(dict(key=list(host),binding=bindings[0],choice=chosen,
            profiles=[dict(router=p['workload']['router'],us=p['decision']['statistics'][chosen]['mean_us'],
                           original_us=p['decision']['statistics']['original']['mean_us']) for p in profiles]))
    passed=sum(r['status']=='PASS' for r in rows)
    summary=dict(status='PASS' if not any(codes) and passed==len(rows) else 'INCOMPLETE',
        passed=passed,expected=len(rows),points=rows,selections=selected,host_keys=len(needed),returncodes=codes,
        source_revision=run['source_revision'],scope=plan['scope'],production_changed=False,
        native_no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',summary)
    print('MOE_CONFIRM_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ('points','selections')}),flush=True)
    return int(summary['status']!='PASS')


def main(a):
    if a.plan:
        p=plans.make_plan(a.mode);print(json.dumps(dict(p['counts'],scope=p['scope'],mode=a.mode),indent=2));return 0
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.output is None:
        if a.resume:raise ValueError('--resume requires --output')
        a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-confirm.',dir=os.environ.get('RUN_ROOT','/workspace')))/'results'
    a.output=a.output.resolve()
    if a.job:return child(a)
    if a.helper_gate:return helper_gate(a)
    if a.worker is not None:return worker(a)
    if not a.confirm_idle:raise ValueError('use --confirm-idle only on idle assigned GPUs')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        plan,run=(sealed(a.output/n) for n in ('plan.json','run.json'));verify(a,run)
        if run['cards']!=a.devices.split(',') or run['mode']!=a.mode:raise ValueError('resume GPU/mode differs')
    else:
        base.sealed(a.output/'plan.json',plans.make_plan(a.mode));plan=sealed(a.output/'plan.json')
        print('MOE_CONFIRM_PLAN '+json.dumps(plan['counts']),flush=True)
        run=build(a,plan)
    a.smi=Path(run['smi'])
    a.expected_ce_ghz=1.7;base.lock_clocks(a,run['devices'])
    if not (a.output/'helper-gate.json').exists():
        command=[sys.executable,__file__,'--output',str(a.output),'--sdk',str(a.sdk),
                 '--l2-bytes',str(a.l2_bytes),'--helper-gate']
        checked_child(command,a.output/'helper-gate.log',run['cards'][0])
    gate=sealed(a.output/'helper-gate.json')
    if gate.get('status')!='PASS' or gate.get('run')!=run['sha256']:
        raise ValueError('helper gate does not belong to this run')
    for pid in plans.preflight_points(plan):
        worker=next(i for i,ids in enumerate(run['assignments']) if pid in ids)
        card=run['cards'][worker];folder=a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
        if completed(folder/'points'/(pid+'.json'),run):continue
        print('MOE_CONFIRM_PREFLIGHT '+json.dumps(dict(point=pid,device=card,reused_in_campaign=True)),flush=True)
        command=[sys.executable,'-u',__file__,'--output',str(a.output),'--sdk',str(a.sdk),
                 '--l2-bytes',str(a.l2_bytes),'--worker',str(worker),'--preflight',pid,
                 '--budget-hours',str(a.budget_hours)]
        checked_child(command,folder/('preflight-'+pid+'.log'),card)
    active=[];started=time.monotonic()
    try:
        for worker,card in enumerate(run['cards']):
            folder=a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
            path=folder/('console-'+str(time.time_ns())+'.log')
            command=[sys.executable,'-u',__file__,'--output',str(a.output),'--sdk',str(a.sdk),
                     '--l2-bytes',str(a.l2_bytes),'--worker',str(worker),'--budget-hours',str(a.budget_hours)]
            with path.open('w') as log:
                p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            active.append((p,path))
        while any(p.poll() is None for p,_ in active):
            counts={}
            for worker,card in enumerate(run['cards']):
                done=[base.read(p) for p in (a.output/('gpu-'+card)/'points').glob('*.json')]
                counts[card]=dict(passed=sum(r['status']=='PASS' for r in done),failed=sum(r['status']!='PASS' for r in done),
                                  expected=len(run['assignments'][worker]))
            print('MOE_CONFIRM_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in active),
                minutes=round((time.monotonic()-started)/60,1),points=counts,logs=str(a.output/'gpu-*/console-*.log'))),flush=True)
            time.sleep(30)
    finally:
        for p,_ in active:stop_process(p)
    for p,path in active:
        if p.returncode:print('MOE_CONFIRM_WORKER_FAIL '+json.dumps(dict(rc=p.returncode,log=str(path),
            diagnostic='\n'.join(path.read_text(errors='replace').splitlines()[-10:]))),flush=True)
    return review(a,plan,run,[p.returncode for p,_ in active])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--devices',default='3,5,6,7');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--mode',choices=('all','prefill','decode'),default='all')
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--budget-hours',type=float,default=8)
    p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path);p.add_argument('--probe',type=Path)
    p.add_argument('--worker',type=int);p.add_argument('--job',type=Path);p.add_argument('--attempt',type=Path)
    p.add_argument('--preflight')
    for flag in ('plan','resume','confirm-idle','helper-gate'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();a.sdk=a.sdk.resolve();cards=a.devices.split(',')
    if (not cards or len(cards)>8 or len(set(cards))!=len(cards) or any(not x.isdecimal() for x in cards) or
        not 1<=a.jobs<=64 or a.l2_bytes<=0 or not 0<a.budget_hours<=24):p.error('invalid GPU/jobs/L2/budget')
    if bool(a.job)!=bool(a.attempt) or a.job and a.worker is None:p.error('incomplete child arguments')
    def stop(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,stop);rc=1
    try:
        if a.worker is not None or a.probe or a.plan or a.helper_gate:rc=main(a)
        else:
            if a.output is None:
                if a.resume:raise ValueError('--resume requires --output')
                a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-confirm.',
                    dir=os.environ.get('RUN_ROOT','/workspace')))/'results'
            a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
            print('MOE_CONFIRM_RUN output='+str(a.output),flush=True)
            with a.output.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rc=main(a)
    except Exception:traceback.print_exc()
    finally:
        if a.worker is None and not (a.probe or a.plan or a.helper_gate) and a.output and a.output.is_dir():package(a.output)
    raise SystemExit(rc)
