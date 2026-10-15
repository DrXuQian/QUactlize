#!/usr/bin/env python3
"""Resumable, bounded PPU component sweep; original wiring remains untouched.

Use --plan-only first. Default pilot validates the measurement mechanism; it
does not start an overnight Cartesian product. Each timed graph contains one
kernel stage only. Successful component records survive a later failed case.
"""
import argparse
from collections import defaultdict
import ctypes as C
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from dev.strategy.plan import make_plan
from dev.strategy.coverage import execution_gaps, require_executable
from dev.strategy.components import component_key, role, recommendation, statistics_checked, component_statistics
from dev.strategy.isolated import Captured, checked,check_sdk_graph_abi
from dev.strategy.journal import ArmJournal, planned_arms, execute_point, atomic_json
from dev.strategy.provenance import harness_identity
from tools.audit_kpack_strategy import audit


def save(path, value):
    atomic_json(path,value)


def record_load(path, identity):
    row=json.loads(path.read_text())
    if (row.get('identity')!=identity or row.get('record_sha256') != digest(
            {k:v for k,v in row.items() if k!='record_sha256'})):
        raise ValueError('component evidence identity/digest differs: '+str(path))
    stats=component_statistics(row)
    if any(row.get(k)!=v for k,v in stats.items()):
        raise ValueError('component statistic or scope differs: '+str(path))
    return row


def family(point):
    w=Workload(**point['workload'])
    return digest([w.qtype,w.n,w.k,w.operation,w.compute,w.experts])[:24]


def bind_original(plan, output):
    """Record the real pure C++ selector, including specialized body identity."""
    from quactlize.dispatch.planning import plan_smallm
    from tools.export_kpack_strategy import recipe
    points=[p for p in plan['points'] if p['workload']['operation']=='indexed' and p['workload']['m']<=8]
    if not points:
        return
    requests=[]
    for p in points:
        w=Workload(**p['workload'])
        requests.append([w.qtype,2,w.n,w.k,w.experts,w.topk,w.channels,w.m,int(w.compute=='bf16')])
    results=plan_smallm(output,requests)['requests']
    for p,row in zip(points,results):
        p['original_selector']=row
        if row['status']:
            p['deferred']='ORIGINAL_INDEXED_SELECTOR_MISS_REQUIRES_CALLER_FALLBACK_REPLAY'
            continue
        if row['kind']==0:
            cfg=dict(row['parent'],kind='tc',split=row['split'],grid_mode=row['grid_mode'],grid_b=row['grid_b'])
        else:
            cfg=dict(row['config'],kind='simt' if row['kind']==1 else 'q4')
        p['original_recipe']=asdict(recipe(cfg))


def module_for(args,w,r):
    from quactlize.decode.grouped_compiler import GroupedComputeCompiler
    from dev.strategy.indexed import config
    c=config(r,w)
    if c['kind']!='tc':
        return None
    # JIT/cache lookup is setup only. Compile failure is a failed candidate,
    # never an invitation to time an unrelated fallback.
    compiler=GroupedComputeCompiler(args.sdk,args.cache,compute_type=w.compute)
    return compiler.build(c['parent'])


def isolated_measure(args, rt, provider, w, recipe, copies, authority, source, records, poison, proof):
    # All actual selected launches execute for correctness / buffer setup,
    # WITHOUT timing the combined call. Mutable producer data is resident.
    for i in range(copies):
        checked(provider.launch(i),'untimed production setup')
    rt.sync()
    capture=Captured(rt,[lambda i=i:provider.launch(i) for i in range(copies)])
    stages=[];pending={}
    try:
        # An extracted sequence must really overwrite poisoned output. Merely
        # rerunning the original call afterwards would hide a broken clone.
        poison();rt.sync()
        for stage,group in enumerate(capture.groups):
            node=dict(capture.descriptions[group[0]])
            node['split']=provider.receipt.get('split',1)
            kind=role(node['name'])
            weight_ring=kind in ('producer','expansion','unclassified')
            cache=('ROTATING_WEIGHT_AND_DERIVED_WORKSPACE' if weight_ring else 'REPEATED_RESIDENT_OR_OVERSIZED_SCRATCH')
            key,identity=component_key(w,recipe,node,authority=authority,source=source,cache=cache)
            path=args.output/'components'/f'{key}.json'
            if path.is_file():
                value=record_load(path,identity)
                records[key]=value
                graph=capture.isolate(stage,weight_ring=weight_ring,validation_only=True)
                graph.close()
                print(f'STRATEGY_COMPONENT id={key[:12]} role={kind} status=REUSED',flush=True)
            else:
                graph=capture.isolate(stage,weight_ring=weight_ring,repeats=args.replays)
                try:
                    values=[]
                    for _ in range(args.rounds):
                        values.append([graph.sample() for _ in range(args.samples)])
                    value=statistics_checked(values)|dict(identity=identity,component_id=key,
                        first_use_excluded=True,host_launch_cost_added=False,
                        timed_kernel_types=1,calls_per_graph=graph.count,
                        numerical='WHOLE_CALL_ORACLE_REQUIRED_BY_CASE',kind=kind)
                    value['record_sha256']=digest(value)
                    pending[key]=(path,value);records[key]=value
                    print(f'STRATEGY_COMPONENT id={key[:12]} role={kind} status=PASS us={value["median_us"]:.4f}',flush=True)
                finally:
                    graph.close()
            stages.append(key)
        rt.sync();proof()
        # Publish only after the extracted serial stages pass the independent
        # output oracle. A failed profile cannot leave reusable PASS timings.
        for path,value in pending.values():
            save(path,value)
        return stages,capture.descriptions[:len(capture.groups)]
    finally:
        capture.close()


def dense_family(args, points, manifest, authority):
    from dev.gemv_simt.native import Runtime
    from dev.strategy.fixture import Dense
    from dev.strategy.dense import Provider
    rt=Runtime(args.sdk,'ppu')
    w=Workload(**points[0]['workload'])
    bench=Dense(rt,w,max(p['workload']['m'] for p in points),args.l2_bytes)
    args._rt=rt
    records={}
    journal=ArmJournal(args.output/'arms',authority,args.invocation)
    try:
        for point in points:
            if not any(journal.due(point,arm) for arm in planned_arms(point)):
                continue
            w=Workload(**point['workload'])
            def measure(arm):
                recipe=Recipe.from_dict(arm['recipe']) if arm['recipe'] else None
                before=len(rt.allocations);provider=None
                try:
                    bench.update(w)
                    provider=Provider(args.package,bench,w,recipe)
                    # Independent correctness, changed A and zero-A control.
                    numeric=[]
                    for repeat in (0,1):
                        bench.update(w,repeat)
                        for i in (0,bench.copies-1):
                            checked(provider.launch(i),'numerical call');numeric.append(bench.check())
                    bench.update(w,zero=True);checked(provider.launch(),'zero-A control');bench.check(zero=True)
                    bench.update(w)
                    staged,descriptions=isolated_measure(args,rt,provider,w,recipe,bench.copies,authority,
                        manifest['jit_source_contract'],records,bench.output.poison,bench.check)
                    # Recheck complete arithmetic after isolated extraction.
                    checked(provider.launch(),'post-profile numerical');numeric.append(bench.check())
                    for scratch in provider.scratch:
                        scratch.check_guard()
                    return dict(correctness='PASS',components=staged,
                        receipt=provider.receipt|dict(nodes=descriptions),error=max(numeric),fixture=bench.record)
                finally:
                    if provider:
                        provider.close()
                    rt.release_after(before)
            if not execute_point(point,journal,measure):return 1
        return 0
    finally:
        rt.close()


def indexed_family(args, points, manifest, authority):
    from quactlize.runtime.native import SDK
    from tools.run_kpack_grouped_device_gate import graph_bind
    from dev.smallm_closure.bench import WeightRing,Context,correctness
    from dev.strategy.indexed import Provider,RuntimeView,point as indexed_point
    sdk=SDK(args.sdk);graph_bind(sdk)
    ring=WeightRing(sdk,indexed_point(Workload(**points[0]['workload'])),args.l2_bytes)
    records={}
    journal=ArmJournal(args.output/'arms',authority,args.invocation)
    try:
        for point in points:
            if not any(journal.due(point,arm) for arm in planned_arms(point)):continue
            if point.get('deferred') or point['workload']['m']>8:
                raise ValueError(point.get('deferred','GROUPED_PREFILL_DRIVER_PENDING'))
            w=Workload(**point['workload']);b=Context(sdk,ring,indexed_point(w));rt=RuntimeView(b)
            try:
                def measure(arm):
                    if not arm['recipe']:raise ValueError('indexed original recipe was not resolved')
                    recipe=Recipe.from_dict(arm['recipe'])
                    provider=None
                    try:
                        provider=Provider(args.package,b,w,recipe,module_for(args,w,recipe))
                        correctness(provider.impl,full=True)
                        b.update(0)
                        staged,nodes=isolated_measure(args,rt,provider,w,recipe,ring.copies,authority,
                            manifest['jit_source_contract'],records,b.out.poison,b.check)
                        checked(provider.launch(),'post-profile numerical');proof=b.check()
                        return dict(correctness='PASS',components=staged,
                            receipt=provider.receipt|dict(nodes=nodes),error=proof,fixture=ring.receipt)
                    finally:
                        if provider:provider.close()
                if not execute_point(point,journal,measure):return 1
            finally:
                b.close()
        return 0
    finally:
        ring.close()


def summarize_point(output, point, journal):
    arms={};records={};receipts={};errors={};failures=[];roles={}
    for arm in planned_arms(point):
        key=arm['id'];roles[key]=arm['roles'];arms[key]=[]
        row=journal.read(point,arm)
        if not row or row['status']!='PASS':
            failures.append(dict(arm=key,status=row['status'] if row else 'NOT_ATTEMPTED',
                                 error=row.get('error') if row else None))
            continue
        data=row['data'];arms[key]=data['components'];receipts[key]=data['receipt'];errors[key]=data['error']
        for cid in data['components']:
            path=output/'components'/f'{cid}.json'
            raw=json.loads(path.read_text())
            if digest(raw['identity'])!=cid or raw['identity']['authority']!=journal.authority:
                raise ValueError('reused arm refers to a different component/authority')
            records[cid]=record_load(path,raw['identity'])
    result=dict(status='FAIL' if failures else 'PASS',authority=journal.authority,point=point,
        arms=arms,receipts=receipts,errors=errors,roles=roles,failures=failures,
        selection=recommendation(arms,records,roles=roles))
    save(output/'cases'/f'{point["id"]}.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package',type=Path)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--cache',type=Path)
    p.add_argument('--cohort',choices=('pilot','decode','prefill','all'),default='pilot')
    p.add_argument('--limit',type=int,default=8,
                   help='initial pool target, NOT a cap on old winners or family/Split-K coverage')
    p.add_argument('--neighbors',type=int,default=1,
                   help='additional local geometry proposals after winner/family coverage')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--samples',type=int,default=5)
    p.add_argument('--replays',type=int,default=32)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--idle-confirmed',action='store_true',help='operator confirms no competing GPU job; no automated idle proof')
    p.add_argument('--child-family')
    p.add_argument('--invocation',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.rounds<3 or a.samples<3 or a.replays<1:p.error('need >=3 rounds, >=3 samples, positive replays')
    a.output=a.output.resolve();a.cache=(a.cache or a.output/'jit-cache').resolve()
    if a.child_family:
        if not a.invocation:p.error('child requires a parent invocation ID')
        plan=json.loads((a.output/'plan.json').read_text())
    else:
        a.output.mkdir(parents=True,exist_ok=a.resume)
        plan=make_plan(a.cohort,a.limit,a.neighbors)
        bind_original(plan,a.output)
        plan=json.loads(json.dumps(plan))
        plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        path=a.output/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise ValueError('resume plan differs; keep the old run and use a new directory')
        if not path.exists():save(path,plan);save(a.output/'audit.json',audit())
        counts=dict(points=len(plan['points']),candidates=sum(1+len(x['candidates']) for x in plan['points']),
                    weight_families=len({family(x) for x in plan['points']}),cohort=a.cohort,
                    production_changed=False,
                    execution_preflight=execution_gaps(plan)['status'])
        print('STRATEGY_PLAN '+json.dumps(counts),flush=True)
        if not (a.output/'execution-preflight.json').exists():
            save(a.output/'execution-preflight.json',execution_gaps(plan))
        if a.plan_only:return 0
    require_executable(plan)
    if not a.package or not a.idle_confirmed:
        p.error('device measurement requires --package and --idle-confirmed; plan-only needs neither')
    from tools.build_ppu_library import verify
    manifest=verify(a.package)
    graph_abi=check_sdk_graph_abi(a.sdk)
    from tools.run_smallm_closure import probe
    physical=probe(a.sdk,a.l2_bytes)
    detected=physical['l2_attribute']['bytes']
    if a.l2_bytes<=0:a.l2_bytes=detected
    if a.l2_bytes<=0:raise ValueError('SDK has no positive L2; provide verified --l2-bytes')
    physical['l2_override']=a.l2_bytes
    os.environ['PPU_SDK']=str(a.sdk.resolve())
    os.environ['QUACTLIZE_PPU_CACHE_DIR']=str(a.cache)
    os.environ['QUACTLIZE_PPU_PYTHON']=sys.executable
    from quactlize.runtime.native import sdk_identity
    authority=dict(plan_sha256=plan['plan_sha256'],device=physical,sdk=sdk_identity(a.sdk),package_sha256=sha(a.package/'manifest.json'),
        kernel=manifest['jit_source_contract'],l2_bytes=a.l2_bytes,graph_abi=graph_abi,
        protocol=dict(rounds=a.rounds,samples=a.samples,replays=a.replays,scope='ISOLATED_SINGLE_KERNEL'),
        idle='OPERATOR_CONFIRMED_NOT_INDEPENDENTLY_AUDITED',
        harness=harness_identity(ROOT))
    authority=json.loads(json.dumps(authority))
    groups=defaultdict(list)
    for point in plan['points']:groups[family(point)].append(point)
    if a.child_family:
        rows=groups[a.child_family]
        return (dense_family if rows[0]['workload']['operation']=='dense' else indexed_family)(a,rows,manifest,authority)
    identity_path=a.output/'authority.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=authority:
        raise ValueError('SDK/device/package/protocol changed; cannot resume these timings')
    if not identity_path.exists():save(identity_path,authority)
    # One local lock for the physical card. This prevents another instance of
    # this runner, but never pretends to detect unrelated inference jobs.
    lock=Path('/tmp')/('quactlize-components-'+digest(physical['pci'])[:24]+'.lock')
    start=time.monotonic();failed=0
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.invocation=uuid.uuid4().hex
        journal=ArmJournal(a.output/'arms',authority,a.invocation)
        for i,(key,points) in enumerate(groups.items()):
            log=None
            while pending:=journal.pending(points):
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child-family',key,
                    '--output',str(a.output),'--package',str(a.package.resolve()),'--sdk',str(a.sdk.resolve()),
                    '--cache',str(a.cache),'--l2-bytes',str(a.l2_bytes),'--rounds',str(a.rounds),
                    '--samples',str(a.samples),'--replays',str(a.replays),'--idle-confirmed',
                    '--invocation',a.invocation]
                log=a.output/(key+f'.{time.time_ns()}.log')
                with log.open('x') as stream:
                    # The lock survives even if the parent is interrupted
                    # while its owned GPU child is still alive.
                    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                           pass_fds=(held.fileno(),))
                    while True:
                        try:rc=child.wait(timeout=30);break
                        except subprocess.TimeoutExpired:
                            print(f'STRATEGY_WAIT family={key} pending_arms={len(pending)} elapsed_minutes={(time.monotonic()-start)/60:.1f} log={log}',flush=True)
                journal.reap(points,rc,log)
                failed+=int(rc!=0)
                if len(journal.pending(points))>=len(pending):
                    # Setup failed before any arm began. Do not spin forever
                    # on the same broken library/fixture; keep other families.
                    save(a.output/'family-failures'/f'{key}.{a.invocation}.json',
                         dict(reason='NO_ARM_PROGRESS',rc=rc,log=str(log)))
                    break
            for point in points:summarize_point(a.output,point,journal)
            elapsed=time.monotonic()-start
            remaining=elapsed/(i+1)*(len(groups)-i-1)/60
            print(f'STRATEGY_PROGRESS families={i+1}/{len(groups)} failed={failed} remaining_minutes={remaining:.1f} eta=OBSERVED_FAMILY_WALL_NOT_GUARANTEE log={log}',flush=True)
    statuses={};missing=[]
    for point in plan['points']:
        path=a.output/'cases'/f'{point["id"]}.json'
        if not path.exists():missing.append(point['id']);continue
        row=json.loads(path.read_text());statuses[point['id']]=row['status']
    result=dict(status='COMPLETE' if not missing and all(s=='PASS' for s in statuses.values()) else 'INCOMPLETE',
        cases=statuses,missing=missing,elapsed_seconds=time.monotonic()-start,
        production_changed=False,admission='ORIGINAL_WIRING_MODEL_AB_REQUIRED')
    save(a.output/'summary.json',result)
    print('STRATEGY_DONE '+json.dumps(result),flush=True)
    return int(result['status']!='COMPLETE')


if __name__=='__main__':
    raise SystemExit(main())
