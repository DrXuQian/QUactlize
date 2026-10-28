#!/usr/bin/env python3
"""Final selected dense/indexed recipes vs linked native llama: fresh Asys ABBA."""
import argparse
import csv
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
from dev.native_components import selected_gate as gate
from dev.native_components.asys import parse as parse_native
from dev.native_components.build import build_adapter
from dev.native_components.fixture import weights,activation
from dev.strategy.asys_components import parse as parse_candidate
from dev.strategy.asys_reparse import measurement_provenance
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity,source_identity
from dev.strategy.refresh import assign,family
from dev.strategy.smallm_asys import revision,METHOD
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.run_kpack_llama_ab import point_request
from tools.run_native_components import carrier,environment,validated
from tools.run_smallm_native_asys import authority as native_authority
from tools.run_smallm_regressions_asys import sealed,execute,stop_process
from tools.run_prefill_model_asys import profile_command
from tools.pack_simt_split_review import package


def assignments(plan,workers):
    if plan.get('cohort') == 'moe-native-models':
        from dev.strategy.smallm_asys import assignments as balanced_points
        return balanced_points(plan,workers,balanced=True)
    groups=assign(plan,workers)
    return [[p['id'] for p in plan['points'] if family(Workload(**p['workload'])) in group]
            for group in groups]


def comparison_csv(path,points,records,commit):
    metrics=('native_us','quactlize_us','delta_pct','native_main_us','quactlize_main_us',
             'native_helper_us','quactlize_helper_us','native_reducer_us','quactlize_reducer_us')
    with path.open('w',newline='') as f:
        out=csv.writer(f)
        dims=('qtype','m','n','k','operation','experts','topk','channels','router')
        out.writerow(('point',*dims,'status',*metrics,'quactlize_commit','recipe','error'))
        for p in points:
            w=p['workload'];r=records.get(p['id'],dict(status='MISSING'))
            out.writerow((p['id'],*[w[k] for k in dims],r['status'],
                *[r.get(k,'') for k in metrics],commit,json.dumps(p['arms'][0]['recipe'],sort_keys=True),r.get('error','')))


def load_saved(path,point,run,shard):
    """Retain valid regressions too; resume is not resampling until a pass."""
    if not path.exists():return None
    value=sealed(path)
    if value['point']!=point['id'] or value['run']!=run['sha256']:
        raise ValueError('saved point belongs to another native comparison')
    if value['status'] not in ('PASS','REGRESSION'):return None
    for name,want in value['evidence'].items():
        if sha(Path(name))!=want:raise ValueError('saved measurement changed: '+name)
    passes=[sealed(Path(p)) for p in value['passes']]
    result=gate.comparison(point,passes,run,shard)
    if result!=value['comparison'] or result['status']!=value['status']:
        raise ValueError('saved comparison no longer matches its raw component receipts')
    return value


def native_request(point,run,shard,folder):
    # The candidate's independent changed-input test leaves repeat1 resident
    # for timing. MoE must match both A and IDs, not just the router label.
    row=point_request(point,run['l2_bytes'],routing_repeat=1 if point['workload']['operation']=='indexed' else 0)
    row['input_fixture']=activation(row)
    raw,receipt=weights(point['workload'],folder/'fixtures'/family(Workload(**point['workload'])))
    return dict(runtime_library=run['runtime'],adapter_library=run['adapter'],
        authority=native_authority(run,shard),rounds=6,samples=5,timing_method=METHOD,
        points=[row],raw_fixture=str(raw),raw_fixture_receipt=receipt,timing_input_repeat=1)


def measure_pass(a,point,run,shard,folder,block):
    arm=gate.ORDER[block]
    attempt=folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    if arm=='native':
        atomic_json(attempt/'request.json',native_request(point,run,shard,folder))
        command=[run['carrier'],str(attempt/'request.json'),str(attempt)]
    else:
        # Reuse the already tested candidate fixture, exact runtime bodies,
        # independent oracle and isolated-stage Asys collector, not event code.
        command=[sys.executable,'-u',str(ROOT/'tools/run_smallm_regressions_asys.py'),
            '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),
            '--point',point['id'],'--attempt',str(attempt)]
    pci=run['devices'][shard]['device']['pci']
    before=base.observe(Path(run['smi']),pci,1700,advisory=True)
    execute(profile_command(run['asys'],attempt/'trace',command),attempt/'profile.log')
    after=base.observe(Path(run['smi']),pci,1700,advisory=True)
    report,db=attempt/'trace.asysrep',attempt/'trace.sqlite'
    execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
    provenance=measurement_provenance(run,shard)
    if arm=='native':
        provenance.update(llama_head=run['llama_head'],native_adapter_sha256=run['adapter_sha256'],
                          native_carrier_sha256=run['carrier_sha256'],backend=run['backend'])
        capture=attempt/(point['id']+'.json');auth=native_authority(run,shard)
        data=parse_native(base.read(capture),point,auth,db,report,provenance)
        parsed=attempt/'parsed'/(point['id']+'.json');atomic_json(parsed,data)
        data=validated(parsed,auth)
        evidence=[capture,parsed,report,db,attempt/'request.json']
    else:
        capture=attempt/'capture.json';value=sealed(capture)
        auth=dict(run=run['sha256'],shard=shard,device=run['devices'][shard])
        if value['point']!=point['id'] or value['authority']!=auth:
            raise ValueError('selected capture authority/point changed')
        data=parse_candidate(value['data'],db,report,provenance)
        evidence=[capture,report,db]
    path=attempt/'pass.json'
    base.sealed(path,dict(status='MEASURED',arm=arm,block=block,point=point['id'],
        selection=point['selection'],data=data,clocks=dict(before=before,after=after),
        evidence={str(p):sha(p) for p in evidence}))
    return path


def worker(a):
    plan,run=(sealed(a.output/name) for name in ('plan.json','run.json'))
    i=a.worker
    if not 0<=i<len(run['cards']):raise ValueError('invalid worker index')
    if (run['plan']!=plan['sha256'] or base.device(a)!=run['devices'][i] or
            revision(ROOT)!=run['source_revision'] or harness_identity(ROOT)!=run['harness'] or
            source_identity(ROOT)!=run['source']):
        raise ValueError('paired worker source/device/plan changed')
    for name,want in run['immutable_files'].items():
        if sha(Path(name))!=want:raise ValueError('native binary or profiler changed: '+name)
    folder=a.output/('gpu-'+run['cards'][i]);folder.mkdir(exist_ok=True)
    pci=run['devices'][i]['device']['pci'];failures=0
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in plan['points']:
            if point['id'] not in run['assignments'][i]:continue
            saved=folder/'points'/(point['id']+'.json')
            if load_saved(saved,point,run,i):continue
            print('SELECTED_NATIVE_START '+json.dumps(dict(point=point['id'],workload=point['workload'])),flush=True)
            paths=[]
            try:
                for block in range(4):
                    paths.append(measure_pass(a,point,run,i,folder,block))
                passes=[sealed(path) for path in paths]
                result=gate.comparison(point,passes,run,i)
                evidence={str(path):sha(path) for path in paths}
                for value in passes:evidence.update(value['evidence'])
                base.sealed(saved,dict(status=result['status'],point=point['id'],run=run['sha256'],
                    passes=list(map(str,paths)),evidence=evidence,comparison=result))
                print('SELECTED_NATIVE_POINT '+json.dumps({k:v for k,v in result.items() if k!='passes'}),flush=True)
            except Exception as exc:
                traceback.print_exc();failures+=1
                base.sealed(saved,dict(status='FAIL',point=point['id'],run=run['sha256'],
                    error=str(exc),passes=list(map(str,paths))))
                print('SELECTED_NATIVE_FAIL '+json.dumps(dict(point=point['id'],error=str(exc),remaining_continue=1)),flush=True)
    return int(bool(failures))


def run(a):
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if (a.output/'plan.json').exists():
        frozen=sealed(a.output/'plan.json')
    elif getattr(a,'cohort',None) == 'moe-native-models':
        from dev.strategy.moe_model_plan import model_inventory
        frozen=gate.plan(a.output,model_inventory(a))
    else:
        frozen=gate.plan(a.output)
    if getattr(a,'cohort',None) and a.cohort!=frozen.get('cohort','dense'):
        raise ValueError('resume native comparison cohort differs')
    from dev.strategy.moe_model_plan import verify_inventory
    verify_inventory(a.output,frozen,getattr(a,'moe_models',None))
    if frozen['schema']!=gate.SCHEMA:raise ValueError('not a selected/native Asys plan')
    base.freeze(a.output/'plan.json',frozen)
    print('SELECTED_NATIVE_PLAN '+json.dumps(dict(frozen['counts'],timing_method=METHOD,
        order=gate.ORDER,allowed_slowdown_pct=0,config_sweep=False)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('--confirm-idle required')
    if not a.llama_source or not a.llama_build:raise ValueError('native llama source and build required')
    a.llama_source=a.llama_source.resolve(strict=True);a.llama_build=a.llama_build.resolve(strict=True)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True);a.smi=base.selected_smi(a)
    clean=environment(a)
    # Test the specified native binary's default route, not an inherited
    # environment override. Native activation quantization stays visible.
    clean.pop('GGML_CUDA_CUBLAS_ACCUMULATOR',None)
    for key in list(os.environ):
        if key not in clean:os.environ.pop(key)
    os.environ.update(clean)
    rev=revision(ROOT);harness=harness_identity(ROOT);source=source_identity(ROOT)
    adapter,receipt=build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'build')
    binary=carrier(a,a.output/'build')
    runtime=next((p.resolve() for p in (a.sdk/'lib/libhggc_wrapper.so',
        a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if p.is_file()),None)
    if runtime is None:raise ValueError('SDK runtime wrapper missing')
    cards=a.devices.split(',');devices=[]
    for card in cards:
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),
            '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--probe',str(path)],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),check=True)
        devices.append(base.read(path))
    base.check_devices(devices)
    inputs=dict(plan=frozen['sha256'],cards=cards,devices=devices,l2_bytes=a.l2_bytes,
        source_revision=rev,harness=harness,source=source,sdk=devices[0]['sdk'],
        assignments=assignments(frozen,len(cards)),adapter=str(adapter),adapter_sha256=sha(adapter),
        carrier=str(binary),carrier_sha256=sha(binary),runtime=str(runtime),
        llama_head=receipt['identity']['llama_head'],backend=receipt['identity']['libraries'],
        asys=str(a.asys),asys_sha256=sha(a.asys),smi=str(a.smi),smi_sha256=sha(a.smi),
        python_version=sys.version,timing_method=METHOD,
        immutable_files={str(p):sha(p) for p in (adapter,binary,runtime,a.asys,a.smi,
            a.output/'build/adapter.json',a.output/'build/carrier.json')} | receipt['identity']['libraries'])
    base.freeze(a.output/'inputs.json',inputs)
    images,dq=base.compile_all(a,frozen)
    run=dict(inputs,images=base.image_identity(images,dq));run['sha256']=digest(run)
    base.freeze(a.output/'run.json',run)
    a.expected_ce_ghz=1.7;base.lock_clocks(a,devices)
    logs=a.output/'workers'/uuid.uuid4().hex;logs.mkdir(parents=True)
    workers=[];started=time.monotonic()
    try:
        for i,card in enumerate(cards):
            with (logs/('gpu-'+card+'.log')).open('x') as log:
                workers.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(i)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in workers):
            print('SELECTED_NATIVE_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in workers),
                minutes=(time.monotonic()-started)/60,logs=str(logs/'gpu-*.log'))),flush=True)
            for p in workers:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p in workers:stop_process(p)
    records={};points={p['id']:p for p in frozen['points']}
    for i,ids in enumerate(run['assignments']):
        for pid in ids:
            path=a.output/('gpu-'+cards[i])/'points'/(pid+'.json')
            value=load_saved(path,points[pid],run,i)
            if value:records[pid]=value['comparison']
            elif path.exists():records[pid]=dict(status='FAIL',point=pid,error=sealed(path)['error'])
    result=gate.aggregate(frozen,records)
    if any(p.returncode for p in workers):result.update(status='INCOMPLETE',component_no_regression=False)
    result.update(source_revision=rev,points=records,returncodes=[p.returncode for p in workers],
                  timing_method=METHOD,run=run['sha256'])
    comparison_csv(a.output/'comparison.csv',frozen['points'],records,rev['quactlize_commit'])
    atomic_json(a.output/'summary.json',result)
    print('SELECTED_NATIVE_DONE '+json.dumps({k:v for k,v in result.items() if k!='points'}),flush=True)
    return 0 if result['status']=='PASS' else 2 if result['status']=='REGRESSION' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--llama-source',type=Path);p.add_argument('--llama-build',type=Path)
    p.add_argument('--cohort',choices=('dense','moe-native-models'))
    p.add_argument('--model-root',type=Path,default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    p.add_argument('--moe-models');p.add_argument('--qwen35-model',type=Path)
    p.add_argument('--qwen122-model',type=Path);p.add_argument('--deepseek-model',type=Path)
    p.add_argument('--devices',default='0,1,2,3');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve();cards=a.devices.split(',')
    if not 1<=len(cards)<=10 or len(set(cards))!=len(cards) or any(not c.isdecimal() or str(int(c))!=c for c in cards):
        p.error('invalid distinct GPU ordinals')
    if not 1<=a.jobs<=64 or a.l2_bytes<=0:p.error('jobs 1..64 and verified positive L2 capacity required')
    if a.probe and a.worker is not None:p.error('child modes are exclusive')
    def stop(*_):raise KeyboardInterrupt('termination requested')
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
