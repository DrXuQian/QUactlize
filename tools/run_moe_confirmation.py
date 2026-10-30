#!/usr/bin/env python3
"""Fixed MoE shortlist, public Quactlize preparation, isolated Asys confirmation.

No new config expansion, historical latency reuse or model claim. By default
no llama dependency; --native-compare uses an existing llama build as control.
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
    if run['mode']=='wired-prefill':
        from dev.moe_confirm import selected as backend
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
    known={} if job.get('fresh_components') else engine.component_cache(folder,run,a.worker,'confirm',point)
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
                if run.get('native_comparison'):
                    from dev.moe_confirm.native_ab import point as compare_point
                    complete=compare_point(a,run,points[pid],deadline)
                else:complete=point(a,run,points[pid],deadline)
                if not complete:budget=True;break
                done+=1
            except Exception as exc:
                failed+=1;traceback.print_exc()
                base.sealed(folder/'points'/(pid+'.json'),dict(status='FAIL',point=pid,run=run['sha256'],error=str(exc)))
                print('MOE_CONFIRM_FAIL '+json.dumps(dict(point=pid,error=str(exc))),flush=True)
    atomic_json(folder/('preflight-status.json' if a.preflight else 'status.json'),
        dict(completed=done,failed=failed,budget_reached=budget))
    return int(bool(failed or budget))


def reuse_build(a,plan):
    """Reuse binaries after a coordinator fix, never copy old measurements."""
    from copy import copy
    from dev.strategy.prefill_build import verify as verify_prefill
    previous=a.reuse_build.resolve(strict=True)
    old=sealed(previous/'run.json');old_plan=sealed(previous/'plan.json')
    if old['plan']!=plan['sha256'] or old_plan['sha256']!=plan['sha256']:
        raise ValueError('build reuse requires the same workload/config plan')
    coordinator='tools/run_moe_confirmation.py'
    for field,current in (('source',source_identity(ROOT)),('harness',harness_identity(ROOT))):
        before,after=old[field]['files'],current['files']
        changed={p for p in before.keys()|after.keys() if before.get(p)!=after.get(p)}
        if changed-{coordinator}:
            raise ValueError('build reuse changed non-coordinator sources: '+','.join(sorted(changed)))
    for name,want in old['immutable_files'].items():
        if sha(name)!=want:raise ValueError('previous build artifact changed: '+name)
    helpers=old['moe_helpers'];verify_moe(helpers)
    helper_dir=Path(old['helpers']);verify_prefill(a.sdk,helper_dir)
    images=base.read(previous/'images.json');dq=Path(old['dequant']) if old['dequant'] else None
    # A continued run can itself reference an older build. Follow only the
    # build lineage; measurement ownership always stays with its original run.
    cache,owner=previous,old;seen={cache}
    while owner.get('reused_build'):
        ref=owner['reused_build'];cache=Path(ref.get('cache_output',ref['output'])).resolve(strict=True)
        if cache in seen:raise ValueError('cyclic build reuse')
        seen.add(cache);owner=sealed(cache/'run.json')
    check=copy(a);check.output=cache
    module_points=[dict(p,arms=[r for r in p['arms'] if r['recipe']['kind']!='full']) for p in plan['points']]
    base.verify_images(check,plan,images,dq,points=module_points)
    reference=dict(output=str(previous),cache_output=str(cache),run=old['sha256'],sdk=old['sdk'],
                   compiled_artifacts_only=True,previous_measurements_reused=False)
    return images,dq,helpers,helper_dir,reference


def retained_point(binding):
    payload=sealed(Path(binding['path']));value=payload['point']
    if (payload['sha256']!=binding['sha256'] or payload['schema']!='quactlize.moe-confirm-retained.v1' or
            value['point']!=binding['point'] or value['status']!='PASS' or
            value['run']!=payload['origin']['run_sha256']):
        raise ValueError('retained complete-point evidence changed')
    return value


def retain_complete(a,plan):
    """Carry whole paired points; partial points restart all arms on a new GPU."""
    if not a.continue_from:return {}
    previous=a.continue_from.resolve(strict=True);old=sealed(previous/'run.json')
    where={pid:previous/('gpu-'+old['cards'][i])/'points'/(pid+'.json')
           for i,ids in enumerate(old['assignments']) for pid in ids}
    retained={}
    for index,p in enumerate(plan['points']):
        if index%16==0:
            print('MOE_CONFIRM_CONTINUE_CHECK '+json.dumps(dict(checked=index,total=len(plan['points']),
                retained=len(retained),gpu_work=False)),flush=True)
        pid=p['id']
        if pid in old.get('retained_points',{}):
            binding=old['retained_points'][pid];retained_point(binding)
            payload=sealed(Path(binding['path']))
        else:
            path=where.get(pid)
            value=completed(path,old) if path is not None else None
            if value is None:continue
            # Keep original numerical proofs, samples and commit/device tags in
            # the compact upload too. Never rewrite them as a new-card result.
            payload=dict(schema='quactlize.moe-confirm-retained.v1',point=value,
                batches=[sealed(Path(b)) for b in value['batch_receipts']],
                origin=dict(run_sha256=old['sha256'],source_revision=old['source_revision'],
                            point_path=str(path),point_sha256=sha(path)))
        dest=a.output/'retained'/(pid+'.json');base.sealed(dest,payload)
        retained[pid]=dict(point=pid,path=str(dest),sha256=sealed(dest)['sha256'])
    print('MOE_CONFIRM_CONTINUE '+json.dumps(dict(retained=len(retained),
        pending=len(plan['points'])-len(retained),partial_points_restarted=True,
        cross_device_partial_timing_reused=False)),flush=True)
    return retained


def assign_remaining(plan,count,retained):
    pending=[p for p in plan['points'] if p['id'] not in retained]
    active=min(len(pending),count)
    assigned=assignments(dict(plan,points=pending),active) if active else []
    return assigned+[[] for _ in range(count-active)]


def build(a,plan):
    from dev.strategy.prefill_build import build as build_prefill
    reused=None
    if a.reuse_build:
        images,dq,helpers,helper_dir,reused=reuse_build(a,plan)
        atomic_json(a.output/'images.json',images)
        print('MOE_CONFIRM_REUSE_BUILD '+json.dumps(reused),flush=True)
    else:
        images,dq=base.compile_all(a,plan)
        helper_dir=a.output/'helpers';build_prefill(a.sdk,helper_dir)
        helpers=build_moe(a.sdk,a.output/'moe-helpers')
        if a.mode=='wired-prefill':
            from dev.moe_confirm.selected import build as build_selected
            helpers['selected_dispatch']=build_selected(a.output/'selected-dispatch',images)
    devices=[]
    for card in a.devices.split(','):
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,__file__,'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),
            '--probe',str(path)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),check=True)
        devices.append(base.read(path))
    base.check_devices(devices)
    if reused and reused['sdk']!=devices[0]['sdk']:raise ValueError('reused build SDK differs')
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True);a.smi=base.selected_smi(a)
    files=[a.asys,a.smi,Path(helpers['bridge']),Path(helpers['scaffold']['candidate']),
           helper_dir/'manifest.json',helper_dir/'libstrategy_prefill.so',
           a.output/'images.json',Path(helpers['bridge']).parent/'receipt.json']
    files += [Path(r['path']) for pool in images.values() for r in pool.values()]
    if 'selected_dispatch' in helpers:files += list(map(Path,helpers['selected_dispatch']['files']))
    native={}
    if getattr(a,'native_compare',False):
        from dev.moe_confirm.native_ab import build as build_native
        native=build_native(a,a.output/'native-build')
        files+=list(map(Path,native['immutable_files']))
    if dq:files += [dq/'manifest.json',dq/'libquactlize_ppu_dequant.so']
    retained=retain_complete(a,plan)
    run=dict(plan=plan['sha256'],source_revision=revision(ROOT),source=source_identity(ROOT),
        harness=harness_identity(ROOT),l2_bytes=a.l2_bytes,cards=a.devices.split(','),devices=devices,
        assignments=assign_remaining(plan,len(devices),retained),retained_points=retained,
        adapter_sha256=helpers['sha256'],moe_helpers=helpers,
        helpers=str(helper_dir),dequant=str(dq) if dq else None,images=base.image_identity(images,dq),
        asys=str(a.asys),asys_sha256=sha(a.asys),smi=str(a.smi),sdk=devices[0]['sdk'],
        immutable_files={str(p):sha(p) for p in files},mode=a.mode,native_comparison=bool(native),
        native=native,scope=plan['scope'],reused_build=reused)
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
    if run.get('native_comparison'):
        from dev.moe_confirm.native_ab import review as review_native
        return review_native(a,plan,run,codes)
    rows=[];groups=defaultdict(list)
    owners={pid:i for i,ids in enumerate(run['assignments']) for pid in ids}
    for p in plan['points']:
        pid=p['id'];binding=run.get('retained_points',{}).get(pid)
        value=(retained_point(binding) if binding else
            completed(a.output/('gpu-'+run['cards'][owners[pid]])/'points'/(pid+'.json'),run))
        rows.append(dict(point=pid,status='PASS' if value else 'MISSING_OR_FAILED',
                         retained=bool(binding),source_run=value['run'] if value else run['sha256']))
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
        passed=passed,expected=len(rows),retained=sum(r['retained'] for r in rows),
        points=rows,selections=selected,host_keys=len(needed),returncodes=codes,
        source_revision=run['source_revision'],scope=plan['scope'],production_changed=False,
        native_no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',summary)
    print('MOE_CONFIRM_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ('points','selections')}),flush=True)
    return int(summary['status']!='PASS')


def main(a):
    if a.plan:
        p=plans.make_plan(a.mode,native_compare=getattr(a,'native_compare',False))
        print(json.dumps(dict(p['counts'],scope=p['scope'],mode=a.mode),indent=2));return 0
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.output is None:
        if a.resume:raise ValueError('--resume requires --output')
        a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-confirm.',dir=os.environ.get('RUN_ROOT','/workspace')))/'results'
    a.output=a.output.resolve()
    if a.job:return child(a)
    if a.helper_gate:return helper_gate(a)
    if a.worker is not None:return worker(a)
    if not a.confirm_idle:raise ValueError('use --confirm-idle only on idle assigned GPUs')
    try:a.output.mkdir(parents=True,exist_ok=a.resume)
    except FileExistsError:
        raise ValueError('output exists: use --resume on the same GPUs, or --continue-from OLD --output NEW to change GPUs') from None
    if a.resume:
        plan,run=(sealed(a.output/n) for n in ('plan.json','run.json'));verify(a,run)
        if run['cards']!=a.devices.split(',') or run['mode']!=a.mode:raise ValueError('resume GPU/mode differs')
        if bool(run.get('native_comparison'))!=bool(getattr(a,'native_compare',False)):
            raise ValueError('resume native comparison scope differs')
    else:
        if a.continue_from:a.reuse_build=a.continue_from
        base.sealed(a.output/'plan.json',plans.make_plan(a.mode,native_compare=getattr(a,'native_compare',False)))
        plan=sealed(a.output/'plan.json')
        print('MOE_CONFIRM_PLAN '+json.dumps(plan['counts']),flush=True)
        run=build(a,plan)
    if not any(run['assignments']):return review(a,plan,run,[])
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
        if pid in run.get('retained_points',{}):continue
        worker_id=next(i for i,ids in enumerate(run['assignments']) if pid in ids)
        card=run['cards'][worker_id];folder=a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
        if run.get('native_comparison'):
            from dev.moe_confirm.native_ab import saved as point_completed
        else:point_completed=completed
        if point_completed(folder/'points'/(pid+'.json'),run):continue
        print('MOE_CONFIRM_PREFLIGHT '+json.dumps(dict(point=pid,device=card,reused_in_campaign=True)),flush=True)
        command=[sys.executable,'-u',__file__,'--output',str(a.output),'--sdk',str(a.sdk),
                 '--l2-bytes',str(a.l2_bytes),'--worker',str(worker_id),'--preflight',pid,
                 '--budget-hours',str(a.budget_hours)]
        checked_child(command,folder/('preflight-'+pid+'.log'),card)
    active=[];started=time.monotonic()
    try:
        for worker_id,card in enumerate(run['cards']):
            if not run['assignments'][worker_id]:continue
            folder=a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
            path=folder/('console-'+str(time.time_ns())+'.log')
            command=[sys.executable,'-u',__file__,'--output',str(a.output),'--sdk',str(a.sdk),
                     '--l2-bytes',str(a.l2_bytes),'--worker',str(worker_id),'--budget-hours',str(a.budget_hours)]
            with path.open('w') as log:
                p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
            active.append((p,path))
        while any(p.poll() is None for p,_ in active):
            counts={}
            for worker_id,card in enumerate(run['cards']):
                done=[base.read(p) for p in (a.output/('gpu-'+card)/'points').glob('*.json')]
                counts[card]=dict(passed=sum(r['status']=='PASS' for r in done),failed=sum(r['status']!='PASS' for r in done),
                                  expected=len(run['assignments'][worker_id]))
            print('MOE_CONFIRM_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in active),
                minutes=round((time.monotonic()-started)/60,1),retained=len(run.get('retained_points',{})),
                points=counts,logs=str(a.output/'gpu-*/console-*.log'))),flush=True)
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
    p.add_argument('--mode',choices=('all','prefill','decode','wired-prefill'),default='all')
    p.add_argument('--reuse-build',type=Path,help='reuse a prior confirmation build, not its timings; use a new output')
    p.add_argument('--native-compare',action='store_true',help='wired-prefill only: fresh Asys native/selected/selected/native')
    p.add_argument('--llama-source',type=Path,default=Path(os.environ['LLAMA_CI_DIR']) if os.environ.get('LLAMA_CI_DIR') else None)
    p.add_argument('--llama-build',type=Path,default=Path(os.environ['LLAMA_CI_BUILD_DIR']) if os.environ.get('LLAMA_CI_BUILD_DIR') else None)
    p.add_argument('--continue-from',type=Path,help='new GPU assignment/output, retaining complete paired points and build')
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--budget-hours',type=float,default=8)
    p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path);p.add_argument('--probe',type=Path)
    p.add_argument('--worker',type=int);p.add_argument('--job',type=Path);p.add_argument('--attempt',type=Path)
    p.add_argument('--preflight')
    for flag in ('plan','resume','confirm-idle','helper-gate'):p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();a.sdk=a.sdk.resolve();cards=a.devices.split(',')
    if (not cards or len(cards)>8 or len(set(cards))!=len(cards) or any(not x.isdecimal() for x in cards) or
        not 1<=a.jobs<=64 or a.l2_bytes<=0 or not 0<a.budget_hours<=24):p.error('invalid GPU/jobs/L2/budget')
    if bool(a.job)!=bool(a.attempt) or a.job and a.worker is None:p.error('incomplete child arguments')
    if a.continue_from and (a.resume or a.reuse_build):p.error('--continue-from uses a new output; do not combine with --resume/--reuse-build')
    if a.native_compare and (a.mode!='wired-prefill' or not a.llama_source or not a.llama_build):
        p.error('--native-compare requires --mode wired-prefill and --llama-source/--llama-build')
    if a.native_compare and (a.continue_from or a.reuse_build):
        p.error('native comparison uses a fresh build/output or --resume; not confirmation timing reuse')
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
