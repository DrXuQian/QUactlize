"""Bounded screen -> common confirmation -> native ABBA, using existing timers."""
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dev.moe_confirm import plan as confirmation,refine_plan,native_ab
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from tools import run_decode_refresh as base,run_moe_prefill_sweep as engine
from tools.run_smallm_regressions_asys import sealed,stop_process

ENTRY=Path(__file__).resolve().parents[2]/'tools/run_moe_confirmation.py'


def folder(a,run,worker=None):
    return a.output/('gpu-'+run['cards'][a.worker if worker is None else worker])


def completed(path,run):
    from tools.run_moe_confirmation import completed as valid_point
    return valid_point(path,run)


def point(a,run,p,deadline):
    work=folder(a,run);stage=a.refine_stage
    if stage=='native':
        chosen=sealed(a.output/'selection.json')
        if chosen['run']!=run['sha256']:raise ValueError('native selection belongs to another run')
        return native_ab.point(a,run,chosen['points'][p['id']],deadline)
    target=work/('screens' if stage=='screen' else 'points')/(p['id']+'.json')
    if not a.preflight and completed(target,run):return True
    if time.monotonic()>=deadline:return False
    resolved=engine.resource(a,run,p,entrypoint=ENTRY);p=resolved['point']
    if a.preflight:
        full=next(a['id'] for a in p['arms'] if a['recipe']['kind']=='full')
        ids=next(ids for ids in engine.batches(p) if full in ids)
        where=engine.batch_path(work,p['id'],'screen',ids)
        if engine.saved(where,run) is None:
            _,value=engine.invocation(a,run,dict(operation='measure',phase='screen',point=p,
                arms=ids,queries=resolved['queries']),profile=True,entrypoint=ENTRY)
            base.sealed(where,value)
        base.sealed(work/'full-gate'/(p['id']+'.json'),dict(status='PASS',point=p['id'],run=run['sha256'],
            scope='M5120_SHIPPING_LAYOUT_PROVIDER_NUMERICS_CHANGED_IDS_AND_GRAPH_REPLAY',
            batch_receipts={str(where):sha(where)}))
        return True
    if stage=='confirm':
        finalists=sealed(a.output/'finalists.json')
        if finalists['run']!=run['sha256']:raise ValueError('finalists belong to another run')
        p=finalists['points'][p['id']]
    value=engine.phase(a,run,p,resolved['queries'],stage,deadline,entrypoint=ENTRY,
                       explicit_helpers=stage=='confirm')
    if value is None:return False
    receipts=[engine.batch_path(work,p['id'],stage,ids) for ids in engine.batches(p)]
    base.sealed(target,dict(value,run=run['sha256'],batch_receipts={str(x):sha(x) for x in receipts}))
    best=value['decision']['winner'];stats=value['decision']['statistics']
    print('MOE_REFINE_POINT '+json.dumps(dict(stage=stage,point=p['id'],workload=p['workload'],
        arms=len(p['arms']),current_us=stats['original']['mean_us'],best_us=stats[best]['mean_us'])),flush=True)
    return True


def collect(a,plan,run,stage):
    paths={'screen':'screens','confirm':'points','native':'native-points'}
    records={}
    for worker,ids in enumerate(run['assignments']):
        for pid in ids:
            path=folder(a,run,worker)/paths[stage]/(pid+'.json')
            value=native_ab.saved(path,run) if stage=='native' else completed(path,run)
            if value is not None:records[pid]=value
    return records


def spawn(a,run,stage,deadline):
    active=[];start=time.monotonic()
    remaining=(deadline-start)/3600
    if remaining<=0:return [1]
    try:
        for index,card in enumerate(run['cards']):
            if not run['assignments'][index]:continue
            path=folder(a,run,index)/f'{stage}-{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
            argv=[sys.executable,'-u',str(ENTRY),'--output',str(a.output),'--sdk',str(a.sdk),
                '--l2-bytes',str(a.l2_bytes),'--worker',str(index),'--refine-stage',stage,
                '--budget-hours',str(remaining)]
            with path.open('w') as log:
                child=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((child,path))
        while any(p.poll() is None for p,_ in active):
            directory={'screen':'screens','confirm':'points','native':'native-points'}[stage]
            counts={card:dict(done=len(list((folder(a,run,i)/directory).glob('*.json'))),
                             expected=len(run['assignments'][i])) for i,card in enumerate(run['cards'])}
            print('MOE_REFINE_WAIT '+json.dumps(dict(stage=stage,alive=sum(p.poll() is None for p,_ in active),
                minutes=round((time.monotonic()-start)/60,1),points=counts,
                logs=str(a.output/f'gpu-*/{stage}-*.log'))),flush=True)
            for p,_ in active:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p,_ in active:stop_process(p)
    for p,path in active:
        if p.returncode:
            print('MOE_REFINE_WORKER_FAIL '+json.dumps(dict(stage=stage,rc=p.returncode,log=str(path),
                diagnostic='\n'.join(path.read_text(errors='replace').splitlines()[-20:]))),flush=True)
    return [p.returncode for p,_ in active]


def choose(a,plan,run,records):
    groups=defaultdict(list);points={};selections=[]
    finals=sealed(a.output/'finalists.json')
    for p in plan['points']:groups[confirmation.key(p['workload'])].append(records[p['id']])
    for key,profiles in groups.items():
        chosen=confirmation.choose(profiles)
        bindings=[p['bindings'][chosen] for p in profiles]
        if any(b!=bindings[0] for b in bindings):raise ValueError('refinement selected router-dependent helper')
        selections.append(dict(key=list(key),choice=chosen,binding=bindings[0],
            profiles=[dict(router=p['workload']['router'],current_us=p['decision']['statistics']['original']['mean_us'],
                           selected_us=p['decision']['statistics'][chosen]['mean_us']) for p in profiles]))
        for p in profiles:
            original=finals['points'][p['point']]
            arm=next(r for r in original['arms'] if r['id']==chosen)
            r=bindings[0]['recipe'];config=bindings[0]['helper_config']
            points[p['point']]=dict(original,arms=[dict(arm,id='original',roles=['original'],recipe=r)],
                sf_configs=[config] if r['route']=='sf' else [],full_configs=[config] if r['kind']=='full' else [],
                selection_source='MEASURED_REFINEMENT_NOT_PRODUCTION')
    base.sealed(a.output/'selection.json',dict(run=run['sha256'],selections=selections,points=points,
        production_changed=False,scope='ONE_CONFIRMED_RECIPE_PER_HOST_KEY_NOT_PRODUCTION'))
    return selections


def run(a,plan,run):
    from tools.run_moe_confirmation import checked_child
    deadline=time.monotonic()+a.budget_hours*3600
    # The two newly legal provider shapes fail early, before the TC sweep.
    for n in sorted({p['workload']['n'] for p in plan['points']}):
        p=next(p for p in plan['points'] if p['workload']['n']==n and p['workload']['router']=='spread')
        worker=next(i for i,ids in enumerate(run['assignments']) if p['id'] in ids)
        where=folder(a,run,worker);where.mkdir(exist_ok=True)
        if completed(where/'full-gate'/(p['id']+'.json'),run):continue
        command=[sys.executable,'-u',str(ENTRY),'--output',str(a.output),'--sdk',str(a.sdk),
            '--l2-bytes',str(a.l2_bytes),'--worker',str(worker),'--preflight',p['id'],
            '--refine-stage','screen','--budget-hours',str(a.budget_hours)]
        checked_child(command,where/('full-preflight-'+p['id']+'.log'),run['cards'][worker])
    stage_codes={};selections=[];status='INCOMPLETE'
    for stage in ('screen','confirm','native'):
        if stage=='native' and not run['native_comparison']:break
        records=collect(a,plan,run,stage)
        if len(records)<len(plan['points']):
            stage_codes[stage]=spawn(a,run,stage,deadline)
            records=collect(a,plan,run,stage)
        if len(records)!=len(plan['points']) or any(stage_codes.get(stage,[])):
            status='INCOMPLETE';break
        if stage=='screen':
            resolved=[sealed(folder(a,run,i)/'resources'/(pid+'.json'))['point']
                      for i,ids in enumerate(run['assignments']) for pid in ids]
            points=refine_plan.finalists(resolved,records)
            base.sealed(a.output/'finalists.json',dict(run=run['sha256'],points=points))
            print('MOE_REFINE_FINALISTS '+json.dumps(dict(points=len(points),
                arms=sum(len(p['arms']) for p in points.values()),common_across_profiles=True)),flush=True)
        elif stage=='confirm':selections=choose(a,plan,run,records);status='CONFIRMED_NATIVE_PENDING'
        else:
            native_ab.review(a,plan,run,stage_codes.get(stage,[]))
            status='REGRESSION' if any(p['status']=='REGRESSION' for p in records.values()) else 'PASS'
    summary=dict(status=status,selections=selections,expected=len(plan['points']),stage_returncodes=stage_codes,
        native_no_regression_admitted=status=='PASS',production_changed=False,model_tpot_admitted=False,
        scope='EXPLICIT_REFINEMENT_COMPONENT_AB_NOT_PRODUCTION_OR_MODEL',
        coverage_audit_keys=len(plan['coverage']['host_keys']),
        coverage_keys_with_additional_untested_geometries=plan['coverage']['keys_with_gaps'])
    atomic_json(a.output/'summary.json',summary)
    print('MOE_REFINE_DONE '+json.dumps(summary),flush=True)
    return int(status not in ('PASS','CONFIRMED_NATIVE_PENDING'))
