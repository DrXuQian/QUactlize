#!/usr/bin/env python3
"""Bounded all-small-M supplementation and alternating finalists, offline only.

No production selector writes. Large-M/full-dequant and model A/B are separate
remaining gates, never implied by COMPLETE_SMALLM. Old five-point screens are
read-only inputs, not fresh measurements under the new cold-ring protocol.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.journal import ArmJournal,atomic_json
from dev.strategy.followup import jobs,families,load_screen,reusable,read_point,finalists,validate_confirmation
from dev.strategy.plan import make_plan
from dev.strategy.subset import point_filter,filtered_plan,completion_status
from dev.strategy.provenance import source_identity,verify_unchanged,harness_identity
from dev.strategy.resource_pilot import load_probe
from dev.strategy.resources import inventory,compiler as tc_compiler
from dev.strategy.resource_pilot import current_module
from dev.strategy.decode_plan import make_decode_plan,original_requests,original_recipe
from dev.strategy.vector import Compiler as VectorCompiler,spec as vector_spec
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Recipe,Workload
from quactlize.runtime.tuning import digest
from tools.run_kpack_strategy_decode import records_for,dequant_checked,l2_authority
from tools.run_kpack_strategy_pilot import oracle_identity
from tools.probe_kpack_strategy import device_probe


def confirmation_job(point):
    return dict(id=digest(['alternating-finalists-v1',point['id']]),recipe=None,roles=['confirmation'])


def all_jobs(point):return jobs(point)+[confirmation_job(point)]


def wait_process(command,log,*,held=None,label='SETUP',progress=None):
    with log.open('x') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                           pass_fds=(held.fileno(),) if held else ())
        start=time.monotonic()
        while True:
            try:return p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f'STRATEGY_CAMPAIGN_WAIT phase={label} minutes={(time.monotonic()-start)/60:.1f} log={log}',flush=True)
                if progress:progress()


def write_family_manifests(args,probe,plan,authority,records,vectors):
    """Do not parse/re-hash 31k resource queries in each of 104 children."""
    for key,points in families(plan['points']).items():
        arms=[a for p in points for a in p['arms']]
        tc={a['module']:records[a['module']] for a in arms if a['recipe']['kind']=='tc'}
        vec={a['module']:vectors[a['module']] for a in arms if a['recipe']['kind']!='tc'}
        queries={a['binding']['query']:probe['receipts'][a['binding']['query']] for a in arms if 'binding' in a}
        value=dict(family=key,authority_sha256=digest(authority),points=points,tc=tc,vectors=vec,queries=queries)
        value['sha256']=digest(value)
        atomic_json(args.output/'families'/(key+'.json'),value)


def family_child(args,previous):
    if args.deadline and time.monotonic()>=args.deadline:return 75
    authority=json.loads((args.output/'authority.json').read_text())
    path=args.output/'families'/(args.family+'.json');manifest=json.loads(path.read_text())
    if (manifest['family']!=args.family or manifest['authority_sha256']!=digest(authority) or
            manifest['sha256']!=digest({k:v for k,v in manifest.items() if k!='sha256'})):
        raise ValueError('family manifest identity/seal differs')
    from quactlize.runtime.native import sdk_identity
    if (harness_identity(ROOT)!=authority['harness'] or device_probe(args.sdk)!=authority['resource']['device'] or
            sdk_identity(args.sdk)!=authority['resource']['sdk']):
        raise ValueError('family child source/SDK/physical device changed')
    dequant_checked(args.dequant,args.sdk)
    vc=VectorCompiler(args.sdk,args.vector_cache);compilers={};seen=set()
    for point in manifest['points']:
        for arm in point['arms']:
            mid=arm['module']
            if mid in seen:continue
            seen.add(mid);spec=arm['spec']
            if arm['recipe']['kind']=='tc':
                key=(spec['endpoint'],spec['compute'])
                if key not in compilers:compilers[key]=tc_compiler(spec,args.sdk,args.resource_cache)
                current_module(manifest['tc'][mid],spec,compilers[key])
            else:vc.verify(spec,manifest['vectors'][mid])
    return child(args,dict(receipts=manifest['queries']),dict(points=manifest['points']),authority,
                 manifest['tc'],manifest['vectors'],previous)


def freeze_plan(args):
    path=args.output/'plan/plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan.get('point_filter',{}) != args.point_filter:
            raise ValueError('campaign point filter changed; use a new output directory')
        verify_unchanged(plan['source_identity'],ROOT)
        if plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
            raise ValueError('campaign frozen plan seal differs')
        return plan
    before=source_identity(ROOT);plan=filtered_plan(make_plan('decode'),args.point_filter)
    # Query the real selector AFTER blind rule generation. Original and
    # historical controls are separate pools in the resource inventory too.
    requests=original_requests(dict(resolved=plan))
    rows=plan_smallm(args.output,requests)['requests']
    from dataclasses import asdict
    for p,row in zip(plan['points'],rows):
        p['original_selector']=row;p['original_recipe']=asdict(original_recipe(row))
    plan['source_identity']=verify_unchanged(before,ROOT)
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    atomic_json(path,plan)
    return plan


def build_vectors(args,plan):
    compiler=VectorCompiler(args.sdk,args.vector_cache);records={}
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    slots=min(args.jobs,max(1,len(plan['vectors'])),max(1,(available-4*2**30)//2**30))
    start=time.monotonic();hits=0
    with ThreadPoolExecutor(max_workers=slots) as pool:
        pending={pool.submit(compiler.build,s):k for k,s in plan['vectors'].items()}
        total=len(pending)
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for task in done:
                key=pending.pop(task);row=task.result();hits+=int(row.pop('cache_hit',False));records[key]=row
            elapsed=time.monotonic()-start
            eta=elapsed/len(records)*len(pending) if records else None
            print('STRATEGY_CAMPAIGN_BUILD '+json.dumps(dict(phase='vectors',completed=len(records),total=total,
                  cache_hits=hits,slots=slots,remaining_seconds=eta,scope='COMPILE_ONLY_ADVISORY')),flush=True)
    atomic_json(args.output/'vectors.json',records)
    return records


def import_points(previous,plan,authority):
    imported={}
    for p in plan['points']:
        if reusable(previous,p,authority):
            imported[p['id']]=dict(previous['points'][p['id']]['result'],measurement_origin='PRIOR_SCREEN',
                previous_root=previous['root'],previous_receipt=previous['receipt_sha256'])
    return imported


def child(args,probe,plan,authority,records,vectors,previous):
    from dev.smallm_closure.bench import Context
    from dev.strategy.cold_decode import Ring,Expansion,measure_arm,measure_expansion
    from dev.strategy.confirm import confirm_point
    from dev.strategy.indexed import point as request_point
    from quactlize.runtime.native import SDK
    points=families(plan['points'])[args.family]
    imported=json.loads((args.output/'imported.json').read_text())
    journal=ArmJournal(args.output/'arms',dict(sha256=digest(authority)),args.invocation)
    sdk=SDK(args.sdk);ring=metadata_ring=b=metadata_b=sf=metadata_sf=None
    helpers={}
    try:
        if args.deadline and time.monotonic()>=args.deadline:return 75
        # Use the minimum guaranteed active slice count in this weight family.
        # No timing is transferred between distinct token/router requests.
        first=min(points,key=lambda p:p['workload']['topk']);w0=Workload(**first['workload'])
        start=time.monotonic();ring=Ring(sdk,w0,authority['l2']['bytes'])
        if any(p['sf_configs'] for p in points):
            metadata_ring=Ring(sdk,w0,authority['l2']['bytes'],weights=ring.w,metadata_only=True)
            req=request_point(w0);req['mode']=2 if w0.operation=='indexed' else 0
            metadata_b=Context(sdk,metadata_ring,req);metadata_sf=Expansion(metadata_b,args.dequant)
        print('STRATEGY_CAMPAIGN_FIXTURE '+json.dumps(dict(family=args.family,seconds=time.monotonic()-start,
              producer=ring.receipt,expansion=metadata_ring.receipt if metadata_ring else None)),flush=True)
        for point in points:
            if not any(journal.due(point,j) for j in all_jobs(point)
                       if point['id'] not in imported or j['roles']==['confirmation']):
                continue
            w=Workload(**point['workload']);req=request_point(w);req['mode']=2 if w.operation=='indexed' else 0
            b=Context(sdk,ring,req)
            if point['sf_configs']:
                if sf:sf.rebind(b)
                else:sf=Expansion(b,args.dequant,gold=metadata_sf.gold)
            by_id={a['id']:a for a in point['arms']}
            def record_for(arm):
                return ((records[arm['module']],probe['receipts'][arm['binding']['query']])
                    if arm['recipe']['kind']=='tc' else (vectors[arm['module']],None))
            if point['id'] not in imported:
                for job in jobs(point):
                    if not journal.due(point,job):continue
                    if args.deadline and time.monotonic()>=args.deadline:return 75
                    journal.start(point,job);start=time.monotonic()
                    try:
                        if job['recipe'] is None:
                            config=job['config']
                            if config not in helpers:
                                helpers[config]=measure_expansion(metadata_b,w,metadata_sf,config,
                                    journal.authority,authority['body_source'],args.output,separate_expansion=True)
                            result=dict(helpers[config],reuse_scope='SAME_WEIGHT_FAMILY_ALL_EXPERT_METADATA')
                        else:
                            arm=by_id[job['id']];record,query=record_for(arm)
                            result=measure_arm(b,w,arm,record,query,sf,journal.authority,
                                authority['body_source'],args.output,separate_expansion=True)
                        journal.finish(point,job,'PASS',data=result)
                        print(f'STRATEGY_CAMPAIGN_JOB point={point["id"]} job={job["id"]} seconds={time.monotonic()-start:.3f} status=PASS',flush=True)
                    except Exception as exc:
                        traceback.print_exc();journal.finish(point,job,'FAIL',error=repr(exc));return 1
                result,components,screen_data=read_point(args.output,point,authority)
            else:
                result=imported[point['id']];components=previous['records']
                screen_data=previous['points'][point['id']]['data']
            shortlist=finalists(point,result,components,max_arms=args.max_finalists)
            atomic_json(args.output/'screens'/(point['id']+'.json'),dict(result,finalists=shortlist))
            job=confirmation_job(point)
            if journal.due(point,job):
                if args.deadline and time.monotonic()>=args.deadline:return 75
                journal.start(point,job)
                try:
                    if shortlist['status']=='INCUMBENT_UNCHANGED':
                        data=dict(correctness='PASS',components=result['component_arms']['original'],
                            status='INCUMBENT_UNCHANGED',production_changed=False,performance_admitted=False,
                            workload=point['workload'],original_proof=screen_data['original'],
                            measurement_origin=result.get('measurement_origin','CURRENT_SCREEN'),
                            scope='EXACT_SAME_TIMED_COMPONENTS_NO_INVENTED_AB',winner='original')
                    elif shortlist['status']=='READY':
                        data=confirm_point(b,metadata_b,sf,metadata_sf,point,result,shortlist,
                                           record_for,journal.authority,authority['body_source'])
                    else:raise ValueError('finalist confirmation blocked: '+repr(shortlist))
                    validate_confirmation(data)
                    journal.finish(point,job,'PASS',data=data)
                    print(f'STRATEGY_CAMPAIGN_CONFIRM point={point["id"]} status={data["status"]}',flush=True)
                except Exception as exc:
                    traceback.print_exc();journal.finish(point,job,'FAIL',error=repr(exc));return 1
            if sf:sf.detach()
            b.close();b=None
        return 0
    finally:
        if sf:sf.close()
        if b:b.close()
        if metadata_sf:metadata_sf.close()
        if metadata_b:metadata_b.close()
        if metadata_ring:metadata_ring.close()
        if ring:ring.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-screen',type=Path,required=True)
    p.add_argument('--resource-cache',type=Path,required=True)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--jobs',type=int,default=16)
    p.add_argument('--max-tc-modules',type=int,default=768)
    p.add_argument('--max-vector-modules',type=int,default=768)
    p.add_argument('--max-arms',type=int,default=160000)
    p.add_argument('--max-finalists',type=int,default=16)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--hours',type=float,default=10.,help='stop BETWEEN arms/families at budget; retain receipts for resume')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--idle-confirmed',action='store_true')
    p.add_argument('--point-filter',type=point_filter,default={},
                   help='JSON qtype/operation/n/k filter; a fresh subset run never relabels prior results')
    p.add_argument('--family',help=argparse.SUPPRESS);p.add_argument('--invocation',default=uuid.uuid4().hex,help=argparse.SUPPRESS)
    p.add_argument('--deadline',type=float,default=0.,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.family and not re.fullmatch('[0-9a-f]{64}',args.family):p.error('invalid family ID')
    if min(args.jobs,args.max_tc_modules,args.max_vector_modules,args.max_arms,args.max_finalists,args.hours)<=0:
        p.error('positive build/measurement limits required')
    args.output=args.output.resolve();args.sdk=args.sdk.resolve();args.resource_cache=args.resource_cache.resolve()
    previous=load_screen(args.previous_screen)
    args.previous_screen=Path(previous['root']);args.vector_cache=args.previous_screen/'vector-cache'
    args.dequant=args.previous_screen/'dequant'
    args.output.mkdir(parents=True,exist_ok=args.resume or bool(args.family))
    if args.family:return family_child(args,previous)
    if not args.family:
        started=time.monotonic();plan=freeze_plan(args)
        inv=inventory(plan);vector_ids={}
        for point in plan['points']:
            for job in jobs(dict(point,sf_configs=[])):
                r=Recipe.from_dict(job['recipe'])
                if r.kind in ('q4','simt'):
                    s=vector_spec(Workload(**point['workload']),r);vector_ids[digest(s)]=s
        counts=dict(points=len(plan['points']),weight_families=len(families(plan['points'])),
                    tc_modules=inv['counts']['modules'],queries=inv['counts']['queries'],
                    vector_modules=len(vector_ids),unexpanded_arms=sum(1+len(x['candidates'])+len(x['champion_controls']) for x in plan['points']))
        atomic_json(args.output/'counts.json',counts)
        print('STRATEGY_CAMPAIGN_PLAN '+json.dumps(counts|dict(hours=args.hours,
              scope='FILTERED_SMALLM_NOT_FULL_COHORT' if args.point_filter else 'ALL_KNOWN_SMALLM_NOT_LARGE_M',
              point_filter=args.point_filter,
              duration_guaranteed=False,production_changed=False)),flush=True)
        if args.plan_only:return 0
        if not args.idle_confirmed:p.error('--idle-confirmed required; no timed job may share this card')
        if counts['tc_modules']>args.max_tc_modules or counts['vector_modules']>args.max_vector_modules:
            raise ValueError('declared compile inventory exceeds explicit bound; no compilation started')
        args.resource_cache.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(ROOT/'tools/probe_kpack_strategy.py'),'--plan',str(args.output/'plan/plan.json'),
                 '--output',str(args.output/'resources'),'--sdk',str(args.sdk),'--cache',str(args.resource_cache),
                 '--jobs',str(args.jobs),'--max-modules',str(args.max_tc_modules)]
        if (args.output/'resources').exists():command.append('--resume')
        if wait_process(command,args.output/('resource-'+str(time.time_ns())+'.log'),label='COMPILE_RESOURCE'):
            raise ValueError('resource compilation/query incomplete; resume preserves successful modules')
    probe=load_probe(args.output,pilot=False)
    raw=probe['resolved'];originals=[p['original_selector'] for p in raw['points']]
    plan=make_decode_plan(probe,originals,pilot=False)
    if plan['counts']['arms']>args.max_arms:raise ValueError('resource-expanded arms exceed declared bound')
    print('STRATEGY_CAMPAIGN_RESOLVED '+json.dumps(plan['counts']),flush=True)
    destination=args.output/'decode-plan.json'
    if destination.exists() and json.loads(destination.read_text())!=plan:raise ValueError('campaign decode plan changed')
    atomic_json(destination,plan)
    from quactlize.runtime.native import sdk_identity
    from dev.strategy.isolated import check_sdk_graph_abi
    if device_probe(args.sdk)!=probe['authority']['device'] or sdk_identity(args.sdk)!=probe['authority']['sdk']:
        raise ValueError('resource and timing SDK/device differ')
    tc_records=records_for(probe,plan,args.sdk)
    vectors=json.loads((args.output/'vectors.json').read_text()) if args.family else build_vectors(args,plan)
    vc=VectorCompiler(args.sdk,args.vector_cache)
    for key,spec in plan['vectors'].items():vc.verify(spec,vectors[key])
    dm=dequant_checked(args.dequant,args.sdk)
    from types import SimpleNamespace
    l2=l2_authority(SimpleNamespace(output=args.previous_screen,l2_bytes=args.l2_bytes))
    authority=dict(plan=plan['plan_sha256'],resource=probe['authority'],harness=harness_identity(ROOT),
        oracle=oracle_identity(),graph_abi=check_sdk_graph_abi(args.sdk),l2=l2,
        tc={k:r['sha256'] for k,r in tc_records.items()},vectors={k:r['sha256'] for k,r in vectors.items()},
        dequant=dm['sha256'],body_source=digest([vc.identity,dm['source_hashes']]),idle='OPERATOR_CONFIRMED',
        protocol=dict(rounds=3,samples=3,replays_min=16,screen_order='HASH_SEEDED_NOT_FINALIST_CONFIRMATION',
                      rings='SEPARATE_ALL_METADATA_AND_ACTIVE_PRODUCER',confirmation='6x5_ALTERNATING'),
        scope='FILTERED_SMALLM_COMPONENTS_NOT_FULL_COHORT_OR_MODEL_AB' if args.point_filter else
              'ALL_KNOWN_SMALLM_COMPONENTS_NOT_LARGE_M_OR_MODEL_AB')
    path=args.output/'authority.json'
    if path.exists() and json.loads(path.read_text())!=authority:raise ValueError('campaign authority changed')
    atomic_json(path,authority)
    if not args.family:atomic_json(args.output/'imported.json',import_points(previous,plan,authority))
    write_family_manifests(args,probe,plan,authority,tc_records,vectors)
    journal=ArmJournal(args.output/'arms',dict(sha256=digest(authority)),args.invocation)
    imported=json.loads((args.output/'imported.json').read_text());groups=families(plan['points'])
    def due(point,job):
        return not (point['id'] in imported and job['roles']!=['confirmation']) and journal.due(point,job)
    lock=Path('/tmp')/('quactlize-components-'+digest(probe['authority']['device']['pci'])[:24]+'.lock')
    budget=False
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for number,(key,points) in enumerate(groups.items()):
            if time.monotonic()-started>=args.hours*3600:budget=True;break
            while any(due(p,j) for p in points for j in all_jobs(p)):
                before=sum(due(p,j) for p in points for j in all_jobs(p))
                log=args.output/(f'family-{key}-{time.time_ns()}.log')
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(args.output),
                     '--previous-screen',str(args.previous_screen),'--resource-cache',str(args.resource_cache),
                     '--sdk',str(args.sdk),'--family',key,'--invocation',args.invocation,'--max-arms',str(args.max_arms),
                     '--max-finalists',str(args.max_finalists),'--l2-bytes',str(args.l2_bytes),
                     '--deadline',str(started+args.hours*3600)]
                def progress():
                    rows=[journal.read(p,j) for p in points for j in all_jobs(p)
                          if not (p['id'] in imported and j['roles']!=['confirmation'])]
                    done=sum(bool(r and r['status'] in ('PASS','FAIL','CRASH')) for r in rows)
                    print(f'STRATEGY_CAMPAIGN_ARMS weight={number+1}/{len(groups)} completed={done}/{len(rows)} '
                          f'budget_remaining_minutes={max(0,args.hours*60-(time.monotonic()-started)/60):.1f}',flush=True)
                rc=wait_process(cmd,log,held=held,label=f'WEIGHT_{number+1}/{len(groups)}',progress=progress)
                for point in points:
                    for job in all_jobs(point):
                        row=journal.read(point,job)
                        if row and row['status']=='RUNNING' and row['invocation']==args.invocation:
                            journal.finish(point,job,'CRASH',error=f'child rc={rc}; log={log}')
                after=sum(due(p,j) for p in points for j in all_jobs(p))
                if rc==75:budget=True;break
                if after==before:
                    print(f'STRATEGY_CAMPAIGN_SETUP_FAIL family={key} rc={rc} log={log} remaining_continue=1',flush=True);break
                if time.monotonic()-started>=args.hours*3600:budget=True;break
            elapsed=time.monotonic()-started
            print('STRATEGY_CAMPAIGN_PROGRESS '+json.dumps(dict(weights=number+1,total=len(groups),
                  elapsed_minutes=elapsed/60,remaining_minutes=elapsed/(number+1)*(len(groups)-number-1)/60,
                  eta='WEIGHT_WALL_AVERAGE_ADVISORY_NOT_GUARANTEE',budget_remaining_minutes=max(0,args.hours*60-elapsed/60))),flush=True)
            if budget:break
    points=[]
    for point in plan['points']:
        row=journal.read(point,confirmation_job(point))
        if row and row['status']=='PASS':validate_confirmation(row['data'])
        status=row['status'] if row else 'PENDING'
        decision=row.get('data',{}).get('decision',{}).get('status') if row else None
        points.append(dict(id=point['id'],workload=point['workload'],status=status,decision=decision,
                           previous_screen=point['id'] in imported,
                           result=str(journal.location(point,confirmation_job(point))/'current.json')))
    points.extend(dict(p,status='BLOCKED_RESOURCE',decision=None,previous_screen=False) for p in plan['blocked'])
    passed=sum(p['status']=='PASS' for p in points);unresolved=sum(p['decision'] in ('RULE_MISS','UNSTABLE_RETAIN_ORIGINAL') for p in points)
    summary=dict(status=completion_status(passed,len(points),unresolved,args.point_filter),
        points=points,passed=passed,expected=len(points),unresolved_selection=unresolved,
        point_filter=args.point_filter,full_smallm_cohort=not bool(args.point_filter),
        prior_screen_reused=len(imported),budget_reached=budget,elapsed_seconds=time.monotonic()-started,
        production_changed=False,performance_admitted=False,full_campaign=False,
        remaining=['LARGE_M_FULL_DEQUANT_PROVIDER','ORIGINAL_WIRING_MODEL_AB'])
    atomic_json(args.output/'summary.json',summary)
    print('STRATEGY_CAMPAIGN_DONE '+json.dumps({k:v for k,v in summary.items() if k!='points'}),flush=True)
    return int(summary['status']=='INCOMPLETE')


if __name__=='__main__':raise SystemExit(main())
