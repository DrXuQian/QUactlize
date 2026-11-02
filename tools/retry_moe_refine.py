#!/usr/bin/env python3
"""Recover the same-card MoE sweep with staged geometry/grid selection.

The original frozen measurement child, its numerical checks, binaries and
Asys protocol execute unchanged. Only host candidate scheduling is new. Raw
samples retain their original commit/run/device; the controller has a separate
commit/file receipt. No old output is overwritten and no compiler is invoked.
"""
import argparse
from copy import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CONTROLLER_FILES=('tools/retry_moe_refine.py','dev/moe_confirm/staged_grid.py',
                  'dev/strategy/moe_prefill_campaign.py')


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def frozen(source):
    sys.path.insert(0,str(source))
    from tools import run_moe_confirmation as driver
    from dev.moe_confirm import refine,refine_plan
    if Path(driver.__file__).resolve()!=source/'tools/run_moe_confirmation.py':
        raise ValueError('measurement imports escaped the frozen source')
    campaign=load_file('recovery_resource_domain',ROOT/CONTROLLER_FILES[2])
    selection=load_file('recovery_grid_selection',ROOT/CONTROLLER_FILES[1])
    return driver,refine,refine_plan,campaign,selection


def controller(driver):
    return dict(revision=driver.revision(ROOT),files={p:driver.sha(ROOT/p) for p in CONTROLLER_FILES},
                scope='HOST_SELECTION_ONLY_ORIGINAL_DEVICE_IMAGES_AND_MEASUREMENT_CHILD')


def clone_receipts(previous,output,run,driver):
    """Copy small receipts only; raw evidence and binaries stay at old paths."""
    for name in ('run.json','plan.json','images.json','helper-gate.json'):
        source=previous/name
        if not source.is_file():raise ValueError('missing original '+str(source))
        shutil.copyfile(source,output/name)
    gate=driver.sealed(output/'helper-gate.json')
    if gate.get('status')!='PASS' or gate.get('run')!=run['sha256']:
        raise ValueError('original helper gate did not pass')
    counts=dict(batches=0,components=0,resources=0)
    for card in run['cards']:
        root=previous/('gpu-'+card)
        print('MOE_RETRY_COPY '+json.dumps(dict(gpu=card,retained=counts,gpu_work=False)),flush=True)
        for name in ('batches','components','resources'):
            for path in sorted((root/name).rglob('*.json')):
                if name=='batches' and driver.engine.saved(path,run) is None:continue
                value=driver.sealed(path)
                if value.get('run')!=run['sha256']:raise ValueError('old receipt run differs: '+str(path))
                target=output/path.relative_to(previous);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target);counts[name]+=1
    return counts


def initialize(a,driver):
    previous=a.previous_run
    run=driver.sealed(previous/'run.json');plan=driver.sealed(previous/'plan.json')
    if run['mode']!='refine-prefill' or run['native_comparison']:
        raise ValueError('recovery requires the indexed-prefill screen, not native/model AB')
    if a.devices and a.devices.split(',')!=run['cards']:
        raise ValueError('recovery keeps the original physical card assignment')
    old=copy(a);old.output=previous;driver.verify(old,run)
    identity=dict(previous=str(previous),source=str(a.source),run=run['sha256'],controller=controller(driver))
    if a.resume:
        receipt=driver.sealed(a.output/'recovery.json')
        if any(receipt[k]!=v for k,v in identity.items()):raise ValueError('recovery binding changed')
    else:
        a.output.mkdir(parents=True,exist_ok=False)
        counts=clone_receipts(previous,a.output,run,driver)
        driver.base.sealed(a.output/'recovery.json',dict(identity,retained=counts,
            original_results_unchanged=True,production_changed=False))
        print('MOE_RETRY_RETAINED '+json.dumps(counts),flush=True)
    return plan,run


def directory(a,run,index=None):
    return a.output/('gpu-'+run['cards'][a.worker if index is None else index])


def domain(a,run,seed,driver,campaign):
    from dev.strategy import moe_prefill_campaign as original
    folder=directory(a,run);retained=folder/'resources'/(seed['id']+'.json')
    path=folder/'domains'/(seed['id']+'.json')
    if path.exists():return driver.sealed(path)
    if retained.exists():
        row=driver.sealed(retained)
        want=original.resolve(seed,row['queries'],driver.base.read(a.output/'images.json'))
        if row['run']!=run['sha256'] or row['point']!=want:
            raise ValueError('retained original resource plan changed')
        # Keep the exact flat pool already partly/fully measured. Do not
        # introduce extra grid jobs into a successfully resolved old point.
        value=dict(row,retained_flat_domain=True)
    else:
        images=driver.base.read(a.output/'images.json');probes={};queries={}
        requests=campaign.query_requests(seed)
        print('MOE_RETRY_RESOURCES '+json.dumps(dict(point=seed['id'],m=seed['workload']['m'],
            queries=len(requests),kernel_launches=0)),flush=True)
        for qid,req in requests.items():
            mid=req['module']
            if mid not in probes:probes[mid]=campaign.NativeProbe(req['spec'],images['tc'][mid])
            queries[qid]=probes[mid].query(req['request'])
        try:
            bound=original.resolve(seed,queries,images);flat=True
        except ValueError as exc:
            if not str(exc).startswith(('empty/missing-control/unbounded resource-resolved',
                    'prefill flat screen exceeds arm limit:',
                    'bounded grid/Split-K rule omitted historical launch;')):
                raise
            bound=campaign.resolve(seed,queries,images,inventory_only=True);flat=False
        value=dict(run=run['sha256'],point=bound,queries=queries,retained_flat_domain=flat)
    value['controller']=controller(driver)
    driver.base.sealed(path,value)
    print('MOE_RETRY_DOMAIN '+json.dumps(dict(point=seed['id'],arms=len(value['point']['arms']),
        flat=value['retained_flat_domain'])),flush=True)
    return driver.sealed(path)


def measure(a,run,point,queries,driver,selection,deadline,stage):
    folder=directory(a,run);phase='confirm' if stage=='confirm' else 'screen'
    target=folder/('points' if stage=='confirm' else 'screens' if stage=='grid' else 'geometry')/(point['id']+'.json')
    if driver.completed(target,run):return True
    paths=list((folder/'batches'/phase).glob('*.json'))
    values,receipts,missing=selection.available_batches(paths,point,run,driver.engine,phase=phase)
    ready=len(point['arms'])-len(missing)
    progress=dict(stage=stage,point=point['id'],m=point['workload']['m'],
                  completed=ready,total=len(point['arms']))
    driver.base.atomic_json(folder/'progress.json',progress)
    print('MOE_RETRY_POINT '+json.dumps(dict(stage=stage,point=point['id'],
        m=point['workload']['m'],total=len(point['arms']),retained=ready,pending=len(missing))),flush=True)
    todo=dict(point,arms=[x for x in point['arms'] if x['id'] in missing])
    for ids in driver.engine.batches(todo):
        if time.monotonic()>=deadline:return False
        start=time.monotonic()
        print('MOE_RETRY_BATCH_START '+json.dumps(dict(stage=stage,point=point['id'],
            completed=ready,arms=len(ids),total=len(point['arms']))),flush=True)
        _,value=driver.engine.invocation(a,run,dict(operation='measure',phase=phase,point=point,
            arms=ids,queries=queries,recovery_controller=controller(driver)),profile=True,
            entrypoint=a.source/'tools/run_moe_confirmation.py')
        path=driver.engine.batch_path(folder,point['id'],phase,ids)
        value['recovery_controller']=controller(driver)
        value['elapsed_seconds']=time.monotonic()-start
        driver.base.sealed(path,value)
        values.append(value['data']);receipts[str(path)]=driver.sha(path);ready+=len(ids)
        progress.update(completed=ready,last_batch_seconds=value['elapsed_seconds'])
        driver.base.atomic_json(folder/'progress.json',progress)
        print('MOE_RETRY_BATCH_DONE '+json.dumps(dict(stage=stage,point=point['id'],
            completed=ready,total=len(point['arms']),seconds=value['elapsed_seconds'])),flush=True)
    data=driver.compose(point,values,explicit_helpers=stage=='confirm')
    driver.base.sealed(target,dict(data,run=run['sha256'],batch_receipts=receipts,
        recovery_controller=controller(driver),screen_scope=point.get('screen_scope','CONFIRMATION')))
    return True


def worker(a,driver,campaign,selection):
    run=driver.sealed(a.output/'run.json');plan=driver.sealed(a.output/'plan.json')
    driver.verify(a,run,a.worker)
    receipt=driver.sealed(a.output/'recovery.json')
    if receipt['controller']!=controller(driver):raise ValueError('controller changed during recovery')
    points={p['id']:p for p in plan['points']};folder=directory(a,run);folder.mkdir(exist_ok=True)
    deadline=time.monotonic()+a.budget_hours*3600;failed=0
    pci=run['devices'][a.worker]['device']['pci']
    from quactlize.runtime.tuning import digest
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for pid in run['assignments'][a.worker]:
            try:
                bound=domain(a,run,points[pid],driver,campaign)
                if a.stage=='geometry':
                    point=bound['point'] if bound['retained_flat_domain'] else selection.geometry_screen(bound['point'])
                    driver.base.sealed(folder/'geometry-plans'/(pid+'.json'),point)
                else:
                    name='grid-plan.json' if a.stage=='grid' else 'finalists.json'
                    point=driver.sealed(a.output/name)['points'][pid]
                if not measure(a,run,point,bound['queries'],driver,selection,deadline,a.stage):
                    return 1
            except Exception as exc:
                failed+=1;traceback.print_exc()
                driver.base.sealed(folder/('recovery-failures-'+a.stage)/(pid+'.json'),
                    dict(point=pid,error=str(exc),run=run['sha256'],controller=controller(driver)))
    return int(bool(failed))


def spawn(a,run,stage,deadline,driver):
    active=[];start=time.monotonic()
    if start>=deadline:return [1]
    try:
        for index,card in enumerate(run['cards']):
            if not run['assignments'][index]:continue
            folder=directory(a,run,index);folder.mkdir(exist_ok=True)
            path=folder/f'retry-{stage}-{time.time_ns()}.log'
            argv=[sys.executable,'-u',str(Path(__file__).resolve()),'--previous-run',str(a.previous_run),
                '--source',str(a.source),'--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),
                '--worker',str(index),'--stage',stage,'--budget-hours',str((deadline-start)/3600)]
            with path.open('w') as log:
                p=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((p,path))
        while any(p.poll() is None for p,_ in active):
            name={'geometry':'geometry','grid':'screens','confirm':'points'}[stage]
            counts={card:dict(done=len(list((directory(a,run,i)/name).glob('*.json'))),
                expected=len(run['assignments'][i])) for i,card in enumerate(run['cards'])}
            for i,card in enumerate(run['cards']):
                progress=directory(a,run,i)/'progress.json'
                if progress.exists():
                    value=driver.base.read(progress)
                    if value['stage']==stage:counts[card]['current']=value
            print('MOE_RETRY_WAIT '+json.dumps(dict(stage=stage,minutes=round((time.monotonic()-start)/60,1),
                points=counts,logs=str(a.output/f'gpu-*/retry-{stage}-*.log'))),flush=True)
            for p,_ in active:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p,_ in active:driver.stop_process(p)
    for p,path in active:
        if p.returncode:
            print('MOE_RETRY_WORKER_FAIL '+json.dumps(dict(stage=stage,log=str(path),rc=p.returncode,
                diagnostic='\n'.join(path.read_text(errors='replace').splitlines()[-25:]))),flush=True)
    return [p.returncode for p,_ in active]


def collect(a,run,name,driver,complete=False):
    out={}
    for i,pids in enumerate(run['assignments']):
        for pid in pids:
            path=directory(a,run,i)/name/(pid+'.json')
            value=driver.completed(path,run) if complete else driver.sealed(path) if path.exists() else None
            if value is not None:out[pid]=value
    return out


def coordinate(a,driver,refine,refine_plan,campaign,selection):
    plan,run=initialize(a,driver)
    deadline=time.monotonic()+a.budget_hours*3600;codes={};status='INCOMPLETE';chosen=[]
    print('MOE_RETRY_PLAN '+json.dumps(dict(points=len(plan['points']),devices=run['cards'],
        output=str(a.output),measurement_commit=run['source_revision']['quactlize_commit'],
        controller=controller(driver),new_compiles=0,budget_hours=a.budget_hours)),flush=True)
    a.expected_ce_ghz=1.7;a.smi=Path(run['smi']);driver.base.lock_clocks(a,run['devices'])
    for stage,name in (('geometry','geometry'),('grid','screens'),('confirm','points')):
        records=collect(a,run,name,driver,complete=True)
        if len(records)<len(plan['points']):
            codes[stage]=spawn(a,run,stage,deadline,driver)
            records=collect(a,run,name,driver,complete=True)
        if len(records)!=len(plan['points']) or any(codes.get(stage,[])):break
        if stage=='geometry':
            domains=collect(a,run,'domains',driver)
            anchors=collect(a,run,'geometry-plans',driver)
            selected=selection.grid_screens({k:v['point'] for k,v in domains.items()},anchors,records)
            for seed in plan['points']:
                pid=seed['id'];selection.historical_coverage(seed,selected[pid],domains[pid]['queries'])
            driver.base.sealed(a.output/'grid-plan.json',dict(run=run['sha256'],points=selected,
                controller=controller(driver)))
            print('MOE_RETRY_GRID_PLAN '+json.dumps(dict(points=len(selected),
                selected_arms=sum(len(p['arms']) for p in selected.values()),
                maximum_point_arms=max(len(p['arms']) for p in selected.values()))),flush=True)
        elif stage=='grid':
            points=driver.sealed(a.output/'grid-plan.json')['points']
            finals=refine_plan.finalists(list(points.values()),records)
            driver.base.sealed(a.output/'finalists.json',dict(run=run['sha256'],points=finals,
                controller=controller(driver)))
        else:
            chosen=refine.choose(a,plan,run,records);status='CONFIRMED_NATIVE_PENDING'
    summary=dict(status=status,selections=chosen,expected=len(plan['points']),stage_returncodes=codes,
        native_no_regression_admitted=False,production_changed=False,model_tpot_admitted=False,
        measurement_source_revision=run['source_revision'],controller=controller(driver),
        previous_run=str(a.previous_run),scope='SAME_CARD_STAGED_SCREEN_RECOVERY_NOT_MODEL_AB')
    driver.base.sealed(a.output/'summary.json',summary)
    print('MOE_RETRY_DONE '+json.dumps(summary),flush=True)
    return int(status!='CONFIRMED_NATIVE_PENDING')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-run',type=Path,required=True,help='original results/ directory')
    p.add_argument('--source',type=Path,required=True,help='unchanged original source/ directory')
    p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--devices',help='optional check, not a reassignment')
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--budget-hours',type=float,default=12)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    p.add_argument('--stage',choices=('geometry','grid','confirm'),help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 0<a.budget_hours<=24 or a.l2_bytes<=0:p.error('invalid budget/L2')
    a.previous_run=a.previous_run.resolve(strict=True);a.source=a.source.resolve(strict=True)
    a.sdk=a.sdk.resolve(strict=True)
    if not (a.source/'.quactlize-source.json').is_file():p.error('use the original frozen source, not a live checkout')
    if a.output is None:
        if a.resume or a.worker is not None:p.error('--resume/worker requires --output')
        a.output=Path(tempfile.mkdtemp(prefix='kpack-moe-retry.',dir=os.environ.get('OUTPUT_ROOT','/workspace')))/'results'
    a.output=a.output.resolve()
    if a.output==a.previous_run or a.previous_run in a.output.parents or a.output in a.previous_run.parents:
        p.error('recovery output must be separate from original results')
    driver,refine,refine_plan,campaign,selection=frozen(a.source)
    def stop(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,stop)
    if a.worker is not None:
        if a.stage is None:p.error('worker requires a stage')
        return worker(a,driver,campaign,selection)
    if not a.confirm_idle:p.error('confirm the original assigned cards are idle')
    # The original coordinator and its worker locks must no longer be held.
    with a.previous_run.with_suffix('.lock').open('a') as original_lock:
        try:fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('original coordinator is still running; stop it first')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:p.error('another recovery owns this output')
            try:return coordinate(a,driver,refine,refine_plan,campaign,selection)
            finally:
                if a.output.is_dir():driver.package(a.output)


if __name__=='__main__':raise SystemExit(main())
