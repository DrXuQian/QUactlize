#!/usr/bin/env python3
"""Native llama small-M Asys baseline; no candidate execution or caller edit."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.native_components.asys import plan, parse, METHOD
from dev.native_components.build import build_adapter
from dev.native_components.fixture import weights, activation
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity, source_identity
from dev.strategy.smallm_asys import revision
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.run_native_components import carrier, environment, validated
from tools.run_kpack_llama_ab import point_request
from tools.run_smallm_regressions_asys import sealed, execute, stop_process
from tools.run_prefill_model_asys import profile_command
from tools.pack_simt_split_review import package


def authority(run, shard):
    return dict(run=run['sha256'], device=run['devices'][shard], adapter=run['adapter_sha256'],
                carrier=run['carrier_sha256'], backend=run['backend'], harness=run['harness']['sha256'],
                source_revision=run['source_revision'], timing_method=METHOD)


def checked_saved(path, point, auth):
    if not path.exists():return None
    value = sealed(path)
    if value['point'] != point['id'] or value['authority'] != auth:
        raise ValueError('saved native point belongs to another measurement')
    if value['status'] != 'PASS':return None
    for name, expected in value['evidence'].items():
        if sha(Path(name)) != expected:raise ValueError('native trace/receipt changed: '+name)
    return value


def worker(a):
    frozen, run = (sealed(a.output/name) for name in ('plan.json','run.json'))
    if not 0 <= a.worker < len(run['cards']):raise ValueError('invalid worker')
    if (run['plan'] != frozen['sha256'] or base.device(a) != run['devices'][a.worker] or
            harness_identity(ROOT) != run['harness'] or source_identity(ROOT) != run['source']):
        raise ValueError('native worker source/device/plan changed')
    for name, expected in run['immutable_files'].items():
        if sha(Path(name)) != expected:raise ValueError('native build/profiler changed: '+name)
    card = run['cards'][a.worker]
    folder = a.output/('gpu-'+card);folder.mkdir(exist_ok=True)
    pci = run['devices'][a.worker]['device']['pci']
    auth = authority(run,a.worker)
    failures = 0
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in frozen['points']:
            if point['id'] not in run['assignments'][a.worker]:continue
            saved = folder/'points'/(point['id']+'.json')
            if checked_saved(saved,point,auth):continue
            attempt = folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            print('NATIVE_SMALLM_ASYS_START '+json.dumps(dict(workload=point['workload'],log=str(attempt/'profile.log'))),flush=True)
            try:
                row = point_request(point,run['l2_bytes'])
                row['input_fixture'] = activation(row)
                raw, receipt = weights(point['workload'],folder/'fixtures'/str(point['workload']['qtype']))
                request = dict(runtime_library=run['runtime'], adapter_library=run['adapter'], authority=auth,
                    rounds=6,samples=5,timing_method=METHOD,points=[row],raw_fixture=str(raw),raw_fixture_receipt=receipt)
                atomic_json(attempt/'request.json',request)
                before = base.observe(Path(run['smi']),pci,1700,advisory=True)
                execute(profile_command(run['asys'],attempt/'trace',
                    [run['carrier'],str(attempt/'request.json'),str(attempt)]),attempt/'profile.log')
                after = base.observe(Path(run['smi']),pci,1700,advisory=True)
                report, db = attempt/'trace.asysrep', attempt/'trace.sqlite'
                execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
                capture = attempt/(point['id']+'.json')
                provenance = dict(run['source_revision'],harness_sha256=run['harness']['sha256'],
                    source_sha256=run['source']['sha256'],device=run['devices'][a.worker],
                    sdk=run['sdk'],asys_sha256=run['immutable_files'][run['asys']],
                    llama_head=run['llama_head'],native_adapter_sha256=run['adapter_sha256'],
                    native_carrier_sha256=run['carrier_sha256'],backend=run['backend'])
                data = parse(base.read(capture),point,auth,db,report,provenance)
                parsed = attempt/'parsed'/(point['id']+'.json')
                atomic_json(parsed,data)
                # Same linked-vs-extracted and loaded-DSO checks as the existing
                # native harness; this path never calls its event measurement.
                data = validated(parsed,auth);data['status'] = 'PASS'
                base.sealed(saved,dict(status='PASS',point=point['id'],authority=auth,data=data,
                    clocks=dict(before=before,after=after),
                    evidence={str(p):sha(p) for p in (capture,parsed,report,db,attempt/'request.json')}))
                print('NATIVE_SMALLM_ASYS_POINT '+json.dumps(dict(status='PASS',qtype=point['workload']['qtype'],
                    m=point['workload']['m'],sum_mean_us=data['sum_mean_us'],
                    stages=[dict(role=c['role'],name=c['name'],mean_us=c['mean_us']) for c in data['components']],
                    activation=data['native_activation_contract'])),flush=True)
            except Exception as exc:
                traceback.print_exc();failures += 1
                base.sealed(saved,dict(status='FAIL',point=point['id'],authority=auth,error=str(exc),attempt=str(attempt)))
    return int(bool(failures))


def run(a):
    a.output.mkdir(parents=True,exist_ok=a.resume)
    frozen = plan(a.qtypes,a.n,a.k)
    base.freeze(a.output/'plan.json',frozen)
    print('NATIVE_SMALLM_ASYS_PLAN '+json.dumps(dict(points=len(frozen['points']),qtypes=a.qtypes,n=a.n,k=a.k,
        m=list(range(1,9)),timing_method=METHOD,config_sweep=False,production_changed=False)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('--confirm-idle required')
    if not a.llama_source or not a.llama_build:raise ValueError('--llama-source and --llama-build required')
    a.sdk=a.sdk.resolve(strict=True);a.llama_source=a.llama_source.resolve(strict=True);a.llama_build=a.llama_build.resolve(strict=True)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True);a.smi=base.selected_smi(a)
    rev=revision(ROOT);harness=harness_identity(ROOT);source=source_identity(ROOT)
    # The SDK/linked backend environment is also used for workers/carrier.
    clean=environment(a)
    for key in list(os.environ):
        if key not in clean:os.environ.pop(key)
    os.environ.update(clean)
    adapter, receipt=build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'build')
    binary=carrier(a,a.output/'build')
    runtime=next((p.resolve() for p in (a.sdk/'lib/libhggc_wrapper.so',a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so')
                  if p.is_file()),None)
    if runtime is None:raise ValueError('SDK runtime wrapper missing')
    cards=a.devices.split(',');devices=[]
    for card in cards:
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),
            '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--probe',str(path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),check=True)
        devices.append(base.read(path))
    base.check_devices(devices)
    run=dict(plan=frozen['sha256'],cards=cards,devices=devices,l2_bytes=a.l2_bytes,
        source_revision=rev,harness=harness,source=source,sdk=devices[0]['sdk'],
        assignments=[[p['id'] for j,p in enumerate(frozen['points']) if j%len(cards)==i] for i in range(len(cards))],
        adapter=str(adapter),adapter_sha256=sha(adapter),carrier=str(binary),carrier_sha256=sha(binary),
        runtime=str(runtime),llama_head=receipt['identity']['llama_head'],backend=receipt['identity']['libraries'],
        asys=str(a.asys),smi=str(a.smi),python_version=sys.version,
        immutable_files={str(p):sha(p) for p in (adapter,binary,runtime,a.asys,a.smi,
            a.output/'build/adapter.json',a.output/'build/carrier.json')})
    run['sha256']=digest(run);base.freeze(a.output/'run.json',run)
    a.expected_ce_ghz=1.7;base.lock_clocks(a,devices)
    logs=a.output/'workers'/uuid.uuid4().hex;logs.mkdir(parents=True)
    workers=[];start=time.monotonic()
    try:
        for i,card in enumerate(cards):
            with (logs/('gpu-'+card+'.log')).open('x') as log:
                workers.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(i)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in workers):
            print('NATIVE_SMALLM_ASYS_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in workers),
                minutes=(time.monotonic()-start)/60,logs=str(logs/'gpu-*.log'))),flush=True)
            for p in workers:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p in workers:stop_process(p)
    for card,p in zip(cards,workers):
        if p.returncode:
            log=logs/('gpu-'+card+'.log')
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-8192))
                diagnostic=stream.read().decode(errors='replace')
            print('NATIVE_SMALLM_ASYS_WORKER_FAIL '+json.dumps(dict(card=card,rc=p.returncode,
                  log=str(log),diagnostic=diagnostic)),flush=True)
    rows=[]
    for shard, group in enumerate(run['assignments']):
        for pid in group:
            path=a.output/('gpu-'+cards[shard])/'points'/(pid+'.json')
            value=sealed(path) if path.exists() else dict(status='MISSING')
            if value['status']=='PASS':checked_saved(path,next(p for p in frozen['points'] if p['id']==pid),authority(run,shard))
            data=value.get('data',{})
            rows.append(dict(point=pid,status=value['status'],error=value.get('error'),
                workload=next(p['workload'] for p in frozen['points'] if p['id']==pid),
                sum_mean_us=data.get('sum_mean_us'),receipt=str(path),
                activation=data.get('native_activation_contract'),
                stages=[dict(role=c['role'],name=c['name'],mean_us=c['mean_us']) for c in data.get('components',[])]))
    result=dict(status='COMPLETE' if all(r['status']=='PASS' for r in rows) and all(p.returncode==0 for p in workers) else 'INCOMPLETE',
        expected=len(frozen['points']),points=rows,scope=frozen['scope'],timing_method=METHOD,
        returncodes=[p.returncode for p in workers],production_changed=False,no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',result)
    print('NATIVE_SMALLM_ASYS_DONE '+json.dumps({k:v for k,v in result.items() if k!='points'}),flush=True)
    return int(result['status']!='COMPLETE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--llama-source',type=Path);p.add_argument('--llama-build',type=Path)
    p.add_argument('--qtypes',default='13,14');p.add_argument('--n',type=int,default=1024);p.add_argument('--k',type=int,default=5120)
    p.add_argument('--devices',default='0,1,2,3');p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    try:a.qtypes=[int(q) for q in a.qtypes.split(',')]
    except ValueError:p.error('invalid qtypes')
    cards=a.devices.split(',')
    if not cards or len(cards)>10 or len(set(cards))!=len(cards) or any(not c.isdecimal() or str(int(c))!=c for c in cards):p.error('invalid GPU ordinals')
    if a.l2_bytes<=0:p.error('--l2-bytes must be verified positive capacity')
    if a.probe and a.worker is not None:p.error('child modes exclusive')
    def stop(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.worker is not None:return worker(a)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(a)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
