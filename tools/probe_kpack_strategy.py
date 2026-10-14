#!/usr/bin/env python3
"""Compile/probe a sealed rule plan; no weights, GEMM launch or benchmark.

Start with the pilot. Full compile inventory requires an explicit raised
--max-modules budget. This resolves distinct persistent grids BEFORE timing,
but is not correctness, performance, or admission of a new device policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import verify_unchanged
from dev.strategy.resources import inventory,compiler,NativeProbe,resolve_point
from quactlize.runtime.tuning import digest
from quactlize.runtime.compiler import sha


def write(path, identity, data):
    value=dict(identity=identity,data=data);value['sha256']=digest(value)
    atomic_json(path,value)


def read(path, identity):
    if not path.exists():return None
    row=json.loads(path.read_text())
    if row.get('identity')!=identity or row.get('sha256')!=digest({k:v for k,v in row.items() if k!='sha256'}):
        raise ValueError('resource resume identity/digest differs: '+str(path))
    return row['data']


def device_probe(sdk):
    """Capability attributes, not a guessed name -> CU/L2 table."""
    import ctypes as C
    from quactlize.runtime.native import SDK,checked
    lib=SDK(sdk).lib
    def call(name,args,*values):
        fn=getattr(lib,'hggc'+name);fn.argtypes,fn.restype=args,C.c_int
        checked(fn(*values),'resource '+name)
    count,ordinal=C.c_int(),C.c_int()
    call('GetDeviceCount',[C.POINTER(C.c_int)],C.byref(count))
    if count.value!=1:raise ValueError('resource probe requires exactly one CUDA_VISIBLE_DEVICES card')
    call('GetDevice',[C.POINTER(C.c_int)],C.byref(ordinal))
    pci=C.create_string_buffer(64)
    call('DeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int],pci,len(pci),ordinal.value)
    values={}
    for name,attribute in (('compute_units',16),('warp_size',10),('max_threads',1)):
        value=C.c_int()
        call('DeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int],C.byref(value),attribute,ordinal.value)
        if value.value<=0:raise ValueError('missing device attribute: '+name)
        values[name]=value.value
    if values['warp_size']!=32:raise ValueError('current PPU1.0 templates require 32 lanes')
    if not pci.value:raise ValueError('empty physical device identity')
    for name in ('DriverGetVersion','RuntimeGetVersion'):
        value=C.c_int()
        call(name,[C.POINTER(C.c_int)],C.byref(value))
        if value.value<=0:raise ValueError('missing runtime identity: '+name)
        values[name]=value.value
    # A PCI address alone can repeat on another host. A driver change must
    # also invalidate resource receipts even if the SDK files are unchanged.
    return dict(host=socket.gethostname(),ordinal=ordinal.value,pci=pci.value.decode(),**values)


def query_child(a,inv,authority):
    from quactlize.runtime.native import sdk_identity
    if authority['sdk']!=sdk_identity(a.sdk) or authority['device']!=device_probe(a.sdk):
        raise ValueError('resource child SDK/device differs')
    mid=a.query_module;record=read(a.output/'modules'/f'{mid}.json',dict(authority=authority,module=mid))
    if not record or record['status']!='PASS':raise ValueError('module was not compiled')
    probe=NativeProbe(inv['modules'][mid],record['record'])
    if any(probe.device[k]!=authority['device'][k] for k in ('ordinal','compute_units')):
        raise ValueError('loaded module sees another device')
    for key,query in inv['queries'].items():
        if query['module']!=mid:continue
        identity=dict(authority=authority,query=key);path=a.output/'queries'/f'{key}.json'
        old=read(path,identity)
        if old and old['status'] in ('READY','UNAVAILABLE'):continue
        # First query error leaves this child. Failed/unqueried cells remain
        # pending for --resume; the parent continues other module processes.
        row=probe.query(query['request'])
        write(path,identity,row)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--cache',type=Path)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--max-modules',type=int,default=48)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--query-module',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.jobs<1 or a.max_modules<1:p.error('positive jobs/module budget required')
    a.output=a.output.resolve();a.sdk=a.sdk.resolve();a.plan=a.plan.resolve()
    a.cache=(a.cache or a.output/'jit-cache').resolve()
    plan=json.loads(a.plan.read_text())
    if plan.get('plan_sha256')!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise ValueError('rule plan digest differs')
    before=plan.get('source_identity')
    if not before:raise ValueError('use audit_kpack_rules.py to seal the plan source before resource probing')
    verify_unchanged(before,ROOT)
    inv=inventory(plan)
    if a.query_module:
        authority=json.loads((a.output/'authority.json').read_text())
        return query_child(a,inv,authority)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    print('STRATEGY_RESOURCES_PLAN '+json.dumps(inv['counts']|dict(max_modules=a.max_modules,
          kernel_launches=0,timing=False,production_changed=False)),flush=True)
    inventory_path=a.output/'inventory.json'
    if inventory_path.exists() and json.loads(inventory_path.read_text())!=inv:
        raise ValueError('resource inventory differs from previous run')
    if not inventory_path.exists():atomic_json(inventory_path,inv)
    if a.plan_only:return 0
    if inv['counts']['modules']>a.max_modules:
        p.error('compile pool exceeds explicit module budget; start a pilot or raise --max-modules after wall-time calibration')
    from quactlize.runtime.native import sdk_identity
    device=device_probe(a.sdk)
    authority=dict(plan=plan['plan_sha256'],source=before['sha256'],sdk=sdk_identity(a.sdk),device=device,
                   cache=str(a.cache))
    authority_path=a.output/'authority.json'
    if authority_path.exists() and json.loads(authority_path.read_text())!=authority:
        raise ValueError('resource source/plan/SDK/device changed; use a new run directory')
    if not authority_path.exists():atomic_json(authority_path,authority)
    lock=Path('/tmp')/('quactlize-components-'+digest(device['pci'])[:24]+'.lock')
    start=time.monotonic();build_wall=query_wall=0.
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        compilers={};todo=[]
        for mid,spec in inv['modules'].items():
            identity=dict(authority=authority,module=mid)
            old=read(a.output/'modules'/f'{mid}.json',identity)
            if old and old['status']=='PASS':
                if sha(old['record']['path'])!=old['record']['sha256']:raise ValueError('cached resource DSO differs')
                continue
            key=(spec['endpoint'],spec['compute'])
            if key not in compilers:compilers[key]=compiler(spec,a.sdk,a.cache)
            todo.append((mid,spec,compilers[key]))
        available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        slots=min(a.jobs,max(1,len(todo)),max(1,(available-4*2**30)//(3*2**30)))
        def build(item):
            mid,spec,c=item;t=time.monotonic()
            try:result=dict(status='PASS',record=c.build(spec['parent']))
            except Exception as exc:result=dict(status='FAIL',error=repr(exc))
            result['wall_seconds']=time.monotonic()-t
            write(a.output/'modules'/f'{mid}.json',dict(authority=authority,module=mid),result)
            return result
        t=time.monotonic();done=0
        print(f'STRATEGY_RESOURCES_BUILD pending={len(todo)} slots={slots} requested_jobs={a.jobs}',flush=True)
        with ThreadPoolExecutor(max_workers=slots) as pool:
            pending={pool.submit(build,item) for item in todo}
            while pending:
                ready,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for f in ready:f.result();done+=1
                elapsed=time.monotonic()-t
                remaining=elapsed/done*len(pending) if done else None
                print('STRATEGY_RESOURCES_PROGRESS '+json.dumps(dict(phase='compile',completed=done,total=len(todo),
                      elapsed_seconds=elapsed,remaining_seconds=remaining,eta_scope='COMPILE_ONLY_OBSERVED_WALL')),flush=True)
        build_wall=time.monotonic()-t
        for i,(mid,spec) in enumerate(inv['modules'].items()):
            record=read(a.output/'modules'/f'{mid}.json',dict(authority=authority,module=mid))
            if record['status']!='PASS':continue
            keys=[k for k,v in inv['queries'].items() if v['module']==mid]
            if all((row:=read(a.output/'queries'/f'{key}.json',dict(authority=authority,query=key))) and
                   row['status'] in ('READY','UNAVAILABLE') for key in keys):continue
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--plan',str(a.plan),'--output',str(a.output),
                     '--sdk',str(a.sdk),'--query-module',mid]
            log=a.output/f'probe-{mid}-{time.time_ns()}.log';t=time.monotonic()
            with log.open('x') as out:
                child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,pass_fds=(held.fileno(),))
                while True:
                    try:rc=child.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        print(f'STRATEGY_RESOURCES_WAIT module={mid} log={log}',flush=True)
            query_wall+=time.monotonic()-t
            print(f'STRATEGY_RESOURCES_QUERY module={i+1}/{len(inv["modules"])} rc={rc} log={log}',flush=True)
    verify_unchanged(before,ROOT)
    receipts={};missing=[]
    for key in inv['queries']:
        row=read(a.output/'queries'/f'{key}.json',dict(authority=authority,query=key))
        if row and row['status'] in ('READY','UNAVAILABLE'):receipts[key]=row
        else:missing.append(key)
    resolved=dict(plan);resolved['points']=[resolve_point(point,receipts) for point in plan['points']]
    resolved['resource_authority']=authority
    resolved['resource_receipts_sha256']=digest(receipts)
    resolved['plan_sha256']=digest({k:v for k,v in resolved.items() if k!='plan_sha256'})
    atomic_json(a.output/'resolved-plan.json',resolved)
    summary=dict(status='INCOMPLETE' if missing else 'COMPLETE',missing=missing,counts=inv['counts'],
                 ready=sum(r['status']=='READY' for r in receipts.values()),
                 unavailable=sum(r['status']=='UNAVAILABLE' for r in receipts.values()),
                 build_wall_seconds=build_wall,query_wall_seconds=query_wall,total_seconds=time.monotonic()-start,
                 resolved_rule_arms=sum(len(p['candidates']) for p in resolved['points']),
                 kernel_launches=0,performance_proven=False,production_changed=False,
                 scope='COMPILE_AND_RESOURCE_QUERY_ONLY_FULL_TIMING_RUNNER_STILL_PENDING')
    atomic_json(a.output/'summary.json',summary)
    print('STRATEGY_RESOURCES_DONE '+json.dumps(summary),flush=True)
    return int(bool(missing))


if __name__=='__main__':raise SystemExit(main())
