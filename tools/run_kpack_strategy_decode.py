#!/usr/bin/env python3
"""Five-point cold small-M screen, not the all-shape campaign or admission.

Reuse all ready TC modules from the resource pilot. Compile only explicit
vector closures and independent expansion code; profile ONE kernel at a time.
Keep original, blind rule and historical control roles separate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.journal import ArmJournal,atomic_json,planned_arms
from dev.strategy.resource_pilot import load_probe,module_record,current_module
from dev.strategy.resources import compiler as tc_compiler
from dev.strategy.decode_plan import original_requests,make_decode_plan,execution_gate
from dev.strategy.provenance import harness_identity
from dev.strategy.vector import Compiler as VectorCompiler
from quactlize.dispatch.planning import plan_smallm
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.probe_kpack_strategy import device_probe
from tools.run_kpack_strategy_pilot import oracle_identity


def jobs(point):
    helpers=[dict(id=digest(['sf-expansion',c]),recipe=None,roles=['expansion'],config=c)
             for c in point['sf_configs']]
    arms=planned_arms(point)
    # A reproducible screen ordering, NOT an alternating finalist confirmation.
    arms.sort(key=lambda a:digest(['screen-order-v1',point['id'],a['id']]))
    return helpers+arms


def records_for(probe,plan,sdk):
    records={}; compilers={}
    for point in plan['points']:
        for arm in point['arms']:
            if arm['recipe']['kind']!='tc' or arm['module'] in records: continue
            record=module_record(probe,arm)
            key=(arm['spec']['endpoint'],arm['spec']['compute'])
            if key not in compilers: compilers[key]=tc_compiler(arm['spec'],sdk,probe['root']/'cache')
            current_module(record,arm['spec'],compilers[key]); records[arm['module']]=record
    return records


def dequant_checked(folder,sdk):
    from quactlize.dequant.native import verify
    m=verify(folder,sdk)
    if sha(sdk/'bin/hgcc')!=m['compiler'] or any(sha(ROOT/k)!=v for k,v in m['source_hashes'].items()):
        raise ValueError('expansion library source/compiler differs')
    return m


def prepare_builds(args,plan):
    compiler=VectorCompiler(args.sdk,args.output/'vector-cache')
    records={}
    with ThreadPoolExecutor(max_workers=min(args.jobs,max(1,len(plan['vectors'])))) as pool:
        pending={pool.submit(compiler.build,item):key for key,item in plan['vectors'].items()}
        for task in as_completed(pending):
            key=pending[task];records[key]=task.result()
            print(f'STRATEGY_DECODE_BUILD vector={len(records)}/{len(pending)} cache_hit={int(records[key]["cache_hit"])}',flush=True)
    dequant=args.output/'dequant'
    if not (dequant/'manifest.json').exists():
        if dequant.exists():
            raise ValueError('incomplete expansion build retained; use a new output, do not overwrite its diagnostics')
        from tools.build_kpack_dequant import build
        # Failed compiler attempts remain as diagnostics; their existence must
        # not require recompiling successful vector peers on --resume.
        attempts=args.output/'expansion-builds';attempts.mkdir(exist_ok=True)
        stage=attempts/uuid.uuid4().hex
        build(args.sdk,stage)
        dequant_checked(stage,args.sdk)
        if dequant.exists():raise ValueError('another expansion build appeared; refusing overwrite')
        stage.rename(dequant)
    dequant_checked(dequant,args.sdk)
    # Stable across cache-hit flags, not builder elapsed time or invocation.
    for row in records.values(): row.pop('cache_hit',None)
    atomic_json(args.output/'vectors.json',records)
    return records


def l2_authority(args):
    import ctypes as C
    from quactlize.dequant.native import bind
    _,_,probe=bind(args.output/'dequant')
    l2,cu,warp=C.c_int(),C.c_int(),C.c_int()
    if probe(C.byref(l2),C.byref(cu),C.byref(warp)): raise ValueError('same-image L2/resource query failed')
    # The compiled probe uses scalar DeviceGetAttribute, NOT the properties
    # struct which has reported a misleading zero in older harnesses.
    if min(l2.value,args.l2_bytes)<0:raise ValueError('negative L2 capacity')
    if l2.value>0 and args.l2_bytes not in (0,l2.value):raise ValueError('supplied L2 differs from SDK query')
    size=l2.value or args.l2_bytes
    if size<=0:raise ValueError('SDK reports no L2; provide verified --l2-bytes, not an inferred board-name value')
    return dict(bytes=size,reported=l2.value,source='SDK_COMPILED_ATTRIBUTE' if l2.value else 'OPERATOR_VERIFIED',
                compute_units=cu.value,warp_size=warp.value)


def child(args,probe,plan,authority,tc_records,vectors):
    from dev.strategy.cold_decode import Ring,Expansion,measure_arm,measure_expansion
    from dev.smallm_closure.bench import Context
    from dev.strategy.indexed import point as request_point
    from quactlize.runtime.native import SDK
    from quactlize.dispatch.table import Workload
    p=next(p for p in plan['points'] if p['id']==args.point)
    journal=ArmJournal(args.output/'arms',dict(sha256=digest(authority)),args.invocation)
    w=Workload(**p['workload']);sdk=SDK(args.sdk)
    ring=b=sf=None
    try:
        started=time.monotonic();ring=Ring(sdk,w,authority['l2']['bytes'],bool(p['sf_configs']))
        req=request_point(w);req['mode']=2 if w.operation=='indexed' else 0
        b=Context(sdk,ring,req)
        if p['sf_configs']: sf=Expansion(b,args.output/'dequant')
        print('STRATEGY_DECODE_FIXTURE '+json.dumps(dict(point=p['id'],seconds=time.monotonic()-started,**ring.receipt)),flush=True)
        by_id={a['id']:a for a in p['arms']}
        for job in jobs(p):
            if not journal.due(p,job):continue
            journal.start(p,job);started=time.monotonic()
            try:
                if job['recipe'] is None:
                    result=measure_expansion(b,w,sf,job['config'],dict(sha256=digest(authority)),
                                            authority['body_source'],args.output)
                else:
                    arm=by_id[job['id']]
                    record=(tc_records if arm['recipe']['kind']=='tc' else vectors)[arm['module']]
                    query=probe['receipts'][arm['binding']['query']] if arm['recipe']['kind']=='tc' else None
                    result=measure_arm(b,w,arm,record,query,sf,dict(sha256=digest(authority)),
                                       authority['body_source'],args.output)
                journal.finish(p,job,'PASS',data=result)
                print(f'STRATEGY_DECODE_JOB point={p["id"][:12]} job={job["id"][:12]} status=PASS seconds={time.monotonic()-started:.3f}',flush=True)
            except Exception as exc:
                traceback.print_exc();journal.finish(p,job,'FAIL',error=repr(exc))
                return 1
        return 0
    finally:
        if sf:sf.close()
        if b:b.close()
        if ring:ring.close()


def summarize(args,plan,journal):
    from dev.strategy.cold_decode import load_component
    from dev.strategy.components import recommendation
    results=[]
    for point in plan['points']:
        records={};arms={};roles={};missing=[];expansions={};receipts={}
        for job in jobs(point):
            row=journal.read(point,job)
            if not row or row['status']!='PASS':
                missing.append(dict(id=job['id'],status=row['status'] if row else 'NOT_ATTEMPTED'));continue
            data=row['data'];receipts[job['id']]=data['receipt'] if 'receipt' in data else data.get('scope')
            for cid in data['components']:
                path=args.output/'components'/(cid+'.json');raw=json.loads(path.read_text())
                if digest(raw['identity'])!=cid or raw['identity']['authority']!=journal.authority:
                    raise ValueError('component authority differs from arm')
                records[cid]=load_component(path,raw['identity'])
            if job['recipe'] is None:expansions[job['config']]=data['components']
            else:
                arms[job['id']]=data['components'];roles[job['id']]=job['roles']
        # The helper search is a separate measured axis, not a product with
        # every GEMM. A historical-only control retains its declared helper.
        best_sf=min(expansions,key=lambda c:sum(records[k]['median_us'] for k in expansions[c]),default=None)
        for arm in point['arms']:
            aid=arm['id']
            roles.setdefault(aid,arm['roles']);arms.setdefault(aid,[])
            if arm['recipe']['route']=='sf' and point['workload']['qtype']!=8 and arms[aid]:
                from quactlize.dispatch.table import Recipe
                cfg=max(0,dict(Recipe.from_dict(arm['recipe']).parameters).get('dequant_config',-1))
                cfg=best_sf if 'rule' in roles[aid] else cfg
                if cfg not in expansions:arms[aid]=[];continue
                arms[aid]=[k for k in arms[aid] if records[k]['kind']!='expansion']+expansions[cfg]
        selection=recommendation(arms,records,roles=roles)
        if missing:selection['recommended']='original'
        selection['admission']='SCREEN_ONLY_ALTERNATING_FINALIST_AND_MODEL_AB_REQUIRED'
        result=dict(point=point['id'],workload=point['workload'],status='PASS' if not missing else 'FAIL',
            selection=selection,missing=missing,expansion_configs=expansions,best_sf=best_sf,receipts=receipts)
        atomic_json(args.output/'cases'/(point['id']+'.json'),result);results.append(result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--jobs',type=int,default=16);p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--execution-run',type=Path,action='append',default=[],help='prior pilot directory; repeat for failed-only retry')
    p.add_argument('--max-vector-modules',type=int,default=64);p.add_argument('--max-arms',type=int,default=512)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--idle-confirmed',action='store_true');p.add_argument('--point',help=argparse.SUPPRESS)
    p.add_argument('--invocation',default=uuid.uuid4().hex,help=argparse.SUPPRESS)
    args=p.parse_args();args.output=args.output.resolve();args.sdk=args.sdk.resolve()
    if min(args.jobs,args.max_vector_modules,args.max_arms)<=0:p.error('positive build/arm bounds required')
    probe=load_probe(args.resource_run)
    if args.point:
        plan=json.loads((args.output/'plan.json').read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=args.resume)
        original=plan_smallm(args.output,original_requests(probe))
        plan=make_decode_plan(probe,original['requests'])
        destination=args.output/'plan.json'
        if destination.exists() and json.loads(destination.read_text())!=plan:raise ValueError('decode resume plan changed')
        atomic_json(destination,plan)
        print('STRATEGY_DECODE_PLAN '+json.dumps(dict(**plan['counts'],full_campaign=False,production_changed=False)),flush=True)
    if plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):raise ValueError('decode plan digest differs')
    if plan['counts']['vector_modules']>args.max_vector_modules or plan['counts']['arms']>args.max_arms:
        raise ValueError('bounded decode pilot exceeds compile/arm budget; no device timing started')
    if args.plan_only:return 0
    if not args.idle_confirmed:p.error('confirm the same single PPU is idle')
    gate=execution_gate(probe,args.execution_run)
    from quactlize.runtime.native import sdk_identity
    from dev.strategy.isolated import check_sdk_graph_abi
    if sdk_identity(args.sdk)!=probe['authority']['sdk'] or device_probe(args.sdk)!=probe['authority']['device']:
        raise ValueError('same physical device/SDK as resource probe required')
    tc_records=records_for(probe,plan,args.sdk)
    vectors=(json.loads((args.output/'vectors.json').read_text()) if args.point else prepare_builds(args,plan))
    vc=VectorCompiler(args.sdk,args.output/'vector-cache')
    if set(vectors)!=set(plan['vectors']):raise ValueError('vector build inventory differs')
    for key,item in plan['vectors'].items():vc.verify(item,vectors[key])
    dm=dequant_checked(args.output/'dequant',args.sdk)
    authority=dict(plan=plan['plan_sha256'],resource=probe['authority'],harness=harness_identity(ROOT),
        oracle=oracle_identity(),graph_abi=check_sdk_graph_abi(args.sdk),l2=l2_authority(args),
        tc={k:r['sha256'] for k,r in tc_records.items()},vectors={k:r['sha256'] for k,r in vectors.items()},
        dequant=dm['sha256'],body_source=digest([vc.identity,dm['source_hashes']]),
        protocol=dict(rounds=3,samples=3,replays_min=16,screen_order='HASH_SEEDED_NOT_FINALIST_CONFIRMATION'),
        idle='OPERATOR_CONFIRMED',execution_gate=gate,scope='FIVE_POINT_COLD_SMALLM_SCREEN_NOT_FULL_CAMPAIGN')
    if (authority['l2']['compute_units']!=probe['authority']['device']['compute_units'] or authority['l2']['warp_size']!=32):
        raise ValueError('expansion image sees a different device')
    destination=args.output/'authority.json'
    if destination.exists() and json.loads(destination.read_text())!=authority:raise ValueError('decode resume authority changed')
    atomic_json(destination,authority)
    if args.point:return child(args,probe,plan,authority,tc_records,vectors)
    journal=ArmJournal(args.output/'arms',dict(sha256=digest(authority)),args.invocation)
    lock=Path('/tmp')/('quactlize-components-'+digest(probe['authority']['device']['pci'])[:24]+'.lock')
    started=time.monotonic()
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in plan['points']:
            point_start=time.monotonic()
            expected=sum(not (row and row['status']=='PASS') for job in jobs(point)
                         for row in [journal.read(point,job)])
            while any(journal.due(point,j) for j in jobs(point)):
                def finished_here():
                    return sum(bool(row and row['invocation']==args.invocation and row['status']!='RUNNING')
                               for job in jobs(point) for row in [journal.read(point,job)])
                before=finished_here()
                log=args.output/'logs'/(point['id']+'.'+uuid.uuid4().hex+'.log');log.parent.mkdir(exist_ok=True)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--resource-run',str(probe['root']),
                    '--output',str(args.output),'--sdk',str(args.sdk),'--l2-bytes',str(args.l2_bytes),
                    '--max-arms',str(args.max_arms),'--max-vector-modules',str(args.max_vector_modules),
                    '--point',point['id'],'--invocation',args.invocation,'--idle-confirmed']
                for path in args.execution_run:command+=['--execution-run',str(path.resolve())]
                with log.open('x') as stream:
                    proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(held.fileno(),))
                    while True:
                        try:rc=proc.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            rows=[journal.read(point,j) for j in jobs(point)]
                            done=sum(bool(r and r['status'] in ('PASS','FAIL','CRASH')) for r in rows)
                            fresh=finished_here()
                            eta=f'{(time.monotonic()-point_start)/fresh*max(0,expected-fresh)/60:.1f}' if fresh>=3 else 'UNKNOWN'
                            print(f'STRATEGY_DECODE_PROGRESS point={point["id"][:12]} completed={done}/{len(rows)} elapsed_minutes={(time.monotonic()-started)/60:.1f} remaining_minutes={eta} eta_scope=CURRENT_POINT_ADVISORY_ONLY log={log}',flush=True)
                for job in jobs(point):
                    row=journal.read(point,job)
                    if row and row['invocation']==args.invocation:
                        if row['status']=='RUNNING':journal.finish(point,job,'CRASH',error=f'child rc={rc}; log={log}')
                if finished_here()==before:
                    print(f'STRATEGY_DECODE_SETUP_FAIL point={point["id"]} rc={rc} log={log} remaining_points_continue=1',flush=True)
                    break
                if rc:print(f'STRATEGY_DECODE_CHILD_FAILED point={point["id"]} rc={rc} log={log} fresh_child_continues=1',flush=True)
    results=summarize(args,plan,journal)
    summary=dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
        points=results,counts=plan['counts'],elapsed_seconds=time.monotonic()-started,
        production_changed=False,performance_admitted=False,full_campaign=False,
        scope=authority['scope'],remaining=['ALTERNATING_FINALIST_CONFIRMATION','LARGE_M_FULL_DEQUANT','ORIGINAL_WIRING_MODEL_AB'])
    atomic_json(args.output/'summary.json',summary)
    print('STRATEGY_DECODE_DONE '+json.dumps({k:v for k,v in summary.items() if k!='points'}),flush=True)
    return int(summary['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
