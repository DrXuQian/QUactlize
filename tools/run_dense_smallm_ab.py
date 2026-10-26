#!/usr/bin/env python3
"""360 dense M1..8 selected-recipe/native comparisons; no sweep or caller patch."""
import argparse
from contextlib import ExitStack
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
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.smallm_ab import (plan, ORDER, SCOPE, candidate_components,
    native_components, verdict, aggregate, fixture_point, checked_fixture)
from dev.strategy.refresh import assign, family
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_native_components as native
from tools import run_decode_refresh as refresh
from tools.run_prefill_cohort_multi import gpu_ordinals
from tools.run_simt_split_multi import check_devices


def read(path):return json.loads(Path(path).read_bytes())


def publish(path,value):
    value={k:v for k,v in value.items() if k!='sha256'}
    atomic_json(path,value|dict(sha256=digest(value)))


def sealed(path):
    value=read(path)
    if value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('result/run seal differs: '+str(path))
    return value


def environment(a):
    env=native.environment(a)
    # The linked native implementation is the reference, not an inherited
    # forced algorithm/accumulator. Its actual stages/arguments are recorded.
    env.pop('GGML_CUDA_CUBLAS_ACCUMULATOR',None)
    return env


def execute(command,log,env):
    with log.open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        try:
            while child.poll() is None:
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('SMALLM_NATIVE_WAIT log='+str(log),flush=True)
        finally:
            if child.poll() is None:child.terminate();child.wait()
    return child.returncode


def command(a):
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(a.output),
        '--sdk',str(a.sdk),'--llama-source',str(a.llama_source),'--llama-build',str(a.llama_build),
        '--devices',a.devices,'--l2-bytes',str(a.l2_bytes),'--jobs',str(a.jobs),'--confirm-idle']


def verify_run(a,run):
    if a.shard is None or not 0<=a.shard<len(run['cards']):raise ValueError('invalid worker shard')
    if harness_identity(ROOT)!=run['harness']:raise ValueError('source changed during comparison')
    if refresh.device(a)!=run['devices'][a.shard]:raise ValueError('physical device/SDK changed')
    payload=read(a.output/'images.json');p=sealed(a.output/'plan.json')
    if p['sha256']!=run['plan']:raise ValueError('worker plan differs from frozen selection')
    dq=Path(read(a.output/'dequant.json')['directory']) if (a.output/'dequant.json').exists() else None
    if refresh.image_identity(payload,dq)!=run['images']:raise ValueError('device payload manifest changed')
    if sha(a.output/'native/native-components-carrier')!=run['native']['carrier']:
        raise ValueError('native carrier changed')
    # Recheck adapter/header/library contents; an existing receipt is a check,
    # never permission to rebuild or overwrite a live payload.
    _,receipt=native.build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'native')
    if receipt!=run['native']['receipt']:raise ValueError('native build binding changed')
    return p,payload,dq


def candidate(a):
    from dev.smallm_closure.bench import Context
    from dev.strategy.cold_decode import Ring, Expansion
    from dev.strategy.confirm import confirm_point
    from quactlize.runtime.native import SDK
    run=sealed(a.output/'run.json');p,images,dq=verify_run(a,run)
    rows=[v for v in p['points'] if v['id'] in a.point]
    if not rows or len(rows)!=len(a.point) or len({family(Workload(**r['workload'])) for r in rows})!=1:
        raise ValueError('candidate child needs unique points from one weight family')
    if family(Workload(**rows[0]['workload'])) not in run['assignments'][a.shard]:
        raise ValueError('candidate family belongs to another worker')
    authority=dict(run=run['sha256'],shard=a.shard,device=run['devices'][a.shard],pass_id=a.pass_id)
    sdk=SDK(a.sdk);w0=Workload(**rows[0]['workload']);current=None
    try:
        with ExitStack() as stack:
            ring=Ring(sdk,w0,a.l2_bytes);stack.callback(ring.close)
            raw=read(a.output/'fixtures'/family(w0)/'weights.json')
            if ring.w.record['raw_sha256']!=raw['raw_sha256']:raise ValueError('candidate/native GGUF bytes differ')
            metadata_b=metadata_sf=None
            if any(r['sf_configs'] for r in rows):
                mr=Ring(sdk,w0,a.l2_bytes,weights=ring.w,metadata_only=True);stack.callback(mr.close)
                metadata_b=Context(sdk,mr,fixture_point(w0));stack.callback(metadata_b.close)
                checked_fixture(metadata_b,w0)
                metadata_sf=Expansion(metadata_b,dq);stack.callback(metadata_sf.close)
            pci=run['devices'][a.shard]['device']['pci']
            for current in rows:
                w=Workload(**current['workload']);bound,bindings=refresh.bind_point(current,images)
                refresh.verify_images(a,p,images,dq,[current])
                with ExitStack() as local:
                    b=Context(sdk,ring,fixture_point(w));local.callback(b.close)
                    checked_fixture(b,w)
                    sf=Expansion(b,dq,gold=metadata_sf.gold) if current['sf_configs'] else None
                    if sf:local.callback(sf.close)
                    clock=lambda:refresh.observe(Path(run['smi']),pci,1700,advisory=True)
                    data=confirm_point(b,metadata_b,sf,metadata_sf,bound,dict(best_sf=None),
                        dict(status='READY',arms=['original'],aliases={}),lambda arm:bindings[arm['id']],
                        authority,run['harness']['sha256'],round_observer=clock)
                candidate_components(data,current)
                publish(a.pass_output/(current['id']+'.json'),dict(status='MEASURED',arm='candidate',
                    point=current['id'],selection=current['selection'],authority=authority,data=data))
                print('SMALLM_NATIVE_CANDIDATE point='+current['id']+' status=PASS',flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc()
        # Never continue in a potentially poisoned GPU context. Parent starts
        # a fresh child for remaining points, preserving completed receipts.
        for r in ([current] if current else rows):
            publish(a.pass_output/(r['id']+'.json'),dict(status='FAIL',point=r['id'],
                arm='candidate',authority=authority,error=repr(exc)))
        return 1


def candidate_pass(a,run,rows,directory,pass_id,env):
    pending=list(rows);result={};attempt=0
    while pending:
        log=directory/f'candidate-{attempt}.log';attempt+=1
        cmd=command(a)+['--shard',str(a.shard),'--candidate','--pass-output',str(directory),'--pass-id',pass_id]
        for r in pending:cmd+=['--point',r['id']]
        rc=execute(cmd,log,env)
        done=[]
        for r in pending:
            path=directory/(r['id']+'.json')
            if path.is_file():
                result[r['id']]=sealed(path)|dict(raw_receipt=str(path),raw_sha256=sha(path));done.append(r['id'])
        if not done:
            for r in pending:
                result[r['id']]=dict(status='FAIL',error='candidate setup failed',rc=rc,log=str(log))
            break
        pending=[r for r in pending if r['id'] not in done]
    return result


def comparison(point,passes,authority):
    if len(passes)!=4 or any(p.get('status')!='MEASURED' for p in passes):
        return dict(status='FAIL',point=point['id'],authority=authority,passes=passes,
                    error='missing or invalid arm; no performance verdict')
    rows=[];raw_hashes=[]
    for block,(arm,p) in enumerate(zip(ORDER,passes)):
        if p.get('arm')!=arm or p.get('block')!=block:raise ValueError('measured ABBA pass order differs')
        data=p['data']
        bound=p.get('authority') if arm=='candidate' else data.get('authority')
        if not bound or any(bound.get(k)!=v for k,v in authority.items()):
            raise ValueError('paired source/physical-device authority differs')
        if arm=='candidate':
            if p['selection']!=point['selection']:raise ValueError('measured production recipe changed')
            cs=candidate_components(data,point)
            if any(r['identity']['authority']!=bound for r in data['component_records'].values()):
                raise ValueError('candidate component authority differs')
            raw_hashes.append(data['arms']['original']['fixture']['fixture']['raw_sha256'])
        else:
            if data.get('timing_input_repeat')!=1:raise ValueError('native timing input differs from candidate')
            cs=native_components(data,point);raw_hashes.append(data['fixture']['raw_sha256'])
            if any(c['identity']!=bound for c in data['components']):
                raise ValueError('native component authority differs')
        rows.append(dict(arm=arm,components=cs))
    if any(h!=raw_hashes[0] for h in raw_hashes):raise ValueError('paired raw GGUF bytes differ')
    if (passes[0]['path']==passes[3]['path'] or
            passes[1]['authority']['pass_id']==passes[2]['authority']['pass_id']):
        raise ValueError('duplicated arm pass is not fresh ABBA')
    return dict(verdict(rows),point=point['id'],workload=point['workload'],selection=point['selection'],
        authority=authority,measurements=passes,native_arithmetic='UNCHANGED_LINKED_NATIVE_INCLUDING_Q8_1_IF_SELECTED',
        candidate_arithmetic='F32_IO_F16_COMPUTE_FP32_ACC_CURRENT_SELECTED_RECIPE',
        timing_input_repeat=1)


def completed(path,point,authority):
    if not path.is_file():return None
    row=sealed(path)
    if row['authority']!=authority or row['point']!=point['id']:raise ValueError('point authority changed')
    if row['status'] not in ('PASS','REGRESSION','INCONCLUSIVE'):return None
    for p in row['measurements']:
        path,expected=(p['raw_receipt'],p['raw_sha256']) if p['arm']=='candidate' else (p['path'],p['sha256'])
        if sha(path)!=expected:raise ValueError('raw paired measurement changed')
    expected=comparison(point,row['measurements'],authority)
    if {k:v for k,v in row.items() if k!='sha256'}!=expected:raise ValueError('saved comparison differs from raw samples')
    return row


def worker(a):
    from dev.native_components.fixture import weights
    run=sealed(a.output/'run.json');p,_,_=verify_run(a,run)
    env=environment(a);root=a.output/('gpu-'+run['cards'][a.shard]);root.mkdir(exist_ok=True)
    authority=dict(run=run['sha256'],shard=a.shard,device=run['devices'][a.shard])
    pci=run['devices'][a.shard]['device']['pci']
    native_auth=dict(authority,backend=run['native']['receipt']['identity']['libraries'],
                     adapter=run['native']['receipt']['binary_sha256'],rounds=6,samples=5)
    runtime=next(path for path in (a.sdk/'lib/libhggc_wrapper.so',a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if path.is_file())
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,fid in enumerate(run['assignments'][a.shard],1):
            rows=[r for r in p['points'] if family(Workload(**r['workload']))==fid]
            rows=[r for r in rows if not completed(root/'points'/(r['id']+'.json'),r,authority)]
            if not rows:continue
            attempt=root/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
            print(f'SMALLM_NATIVE_FAMILY index={index}/{len(run["assignments"][a.shard])} points={len(rows)} log={attempt}',flush=True)
            try:
                weights(rows[0]['workload'],a.output/'fixtures'/fid)
                results={r['id']:[] for r in rows}
                for block,arm in enumerate(ORDER):
                    directory=attempt/f'{block}-{arm}';directory.mkdir()
                    stamp=dict(arm=arm,block=block,before=refresh.observe(Path(run['smi']),pci,1700,advisory=True))
                    if arm=='native':
                        args=SimpleNamespace(**(vars(a)|dict(output=directory,reuse_fixtures=a.output/'fixtures')))
                        request=dict(runtime_library=str(runtime),adapter_library=str(a.output/'native/libnative-components.so'),
                                     authority=native_auth,rounds=6,samples=5,timing_input_repeat=1)
                        measured=native.run_family(args,a.output/'native/native-components-carrier',request,
                            [native.point_request(r,a.l2_bytes) for r in rows],native_auth)
                    else:measured=candidate_pass(a,run,rows,directory,str(attempt)+':'+str(block),env)
                    stamp['after']=refresh.observe(Path(run['smi']),pci,1700,advisory=True)
                    atomic_json(directory/'clock.json',stamp)
                    for r in rows:results[r['id']].append(measured.get(r['id'],dict(status='FAIL',error='missing point'))|
                                                         dict(arm=arm,block=block))
                for point in rows:
                    try:value=comparison(point,results[point['id']],authority)
                    except Exception as exc:value=dict(status='FAIL',point=point['id'],authority=authority,
                        error=repr(exc),passes=results[point['id']])
                    publish(root/'points'/(point['id']+'.json'),value)
                    print('SMALLM_NATIVE_POINT '+json.dumps({k:value[k] for k in
                        ('point','status','native_us','quactlize_us','delta_pct','error') if k in value}),flush=True)
            except Exception as exc:
                traceback.print_exc()
                for point in rows:
                    publish(root/'points'/(point['id']+'.json'),dict(status='FAIL',point=point['id'],
                        authority=authority,error=repr(exc),attempt=str(attempt)))
    return 0


def review(a,p,run,codes):
    records={};family_shards={f:i for i,fs in enumerate(run['assignments']) for f in fs}
    for point in p['points']:
        i=family_shards[family(Workload(**point['workload']))]
        path=a.output/('gpu-'+run['cards'][i])/'points'/(point['id']+'.json')
        if not path.is_file():continue
        authority=dict(run=run['sha256'],shard=i,device=run['devices'][i])
        records[point['id']]=completed(path,point,authority) or sealed(path)
    result=aggregate(p,records)|dict(worker_returncodes=codes)
    if 'retry' in p:
        result['retained_prior_tc']=p['retry']['retained_counts']
        result['retry_scope']='VECTOR_MODE_REPAIR_ONLY_NOT_FULL_COHORT_ADMISSION'
        result['full_cohort_no_regression']=False
    if any(codes):result.update(status='INCOMPLETE',component_no_regression=False)
    atomic_json(a.output/'summary.json',result)
    with (a.output/'comparison.csv').open('w',newline='') as out:
        fields=['qtype','n','k','m','status','native_us','quactlize_us','delta_pct',
                'native_main_us','native_helper_us','native_reducer_us',
                'quactlize_main_us','quactlize_helper_us','quactlize_reducer_us']
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for point in p['points']:
            row=point['workload']|records.get(point['id'],dict(status='MISSING'))
            writer.writerow({k:row.get(k) for k in fields})
    print('SMALLM_NATIVE_DONE '+json.dumps(result),flush=True)
    return 0 if result['component_no_regression'] and not any(codes) else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    parser.add_argument('--llama-source',type=Path,default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    parser.add_argument('--llama-build',type=Path)
    parser.add_argument('--devices',default='0,1,2,3')
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--l2-bytes',type=int,default=67108864)
    parser.add_argument('--confirm-idle',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-from',type=Path,help='New run for all vectors from the affected dense/native run; TC evidence remains historical')
    parser.add_argument('--probe',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--shard',type=int,help=argparse.SUPPRESS)
    parser.add_argument('--candidate',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--point',action='append',default=[],help=argparse.SUPPRESS)
    parser.add_argument('--pass-output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--pass-id',help=argparse.SUPPRESS)
    a=parser.parse_args()
    if a.retry_from and (a.resume or a.shard is not None or a.candidate or a.probe):
        parser.error('--retry-from creates a new coordinator run, not an old-run resume')
    cards=gpu_ordinals(a.devices)
    if not 1<=a.jobs<=64 or a.l2_bytes<=0:parser.error('jobs 1..64 and verified positive L2 required')
    if not a.plan_only and not a.confirm_idle:parser.error('--confirm-idle required')
    a.output=a.output.resolve();a.sdk=a.sdk.resolve();a.llama_source=a.llama_source.resolve()
    a.llama_build=(a.llama_build or a.llama_source/'build-ci').resolve()
    if a.probe:atomic_json(a.probe,refresh.device(a));return 0
    if a.candidate:return candidate(a)
    if a.shard is not None:return worker(a)
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    with (a.output/'.coordinator.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return coordinate(a,cards)
        finally:
            if not a.plan_only:
                from tools.pack_simt_split_review import package
                package(a.output)


def coordinate(a,cards):
    p=sealed(a.output/'plan.json') if a.resume else plan(a.output)
    if not a.resume and getattr(a,'retry_from',None):
        from dev.native_components.smallm_retry import vector_retry
        p=vector_retry(p,a.retry_from,completed)
    if not a.resume:atomic_json(a.output/'plan.json',p)
    print('SMALLM_NATIVE_PLAN '+json.dumps(p['counts']|dict(devices=cards,scope=SCOPE)),flush=True)
    if a.plan_only:return 0
    env=environment(a);env.pop('CUDA_VISIBLE_DEVICES',None)
    a.smi=refresh.selected_smi(SimpleNamespace(smi=None,sdk=a.sdk));a.expected_ce_ghz=1.7
    bindings=dict(cards=cards,sdk=str(a.sdk),llama_source=str(a.llama_source),
        llama_build=str(a.llama_build),l2_bytes=a.l2_bytes,plan=p['sha256'],harness=harness_identity(ROOT),
        smi=str(a.smi),smi_sha256=sha(a.smi),jobs=a.jobs)
    if a.resume:
        run=sealed(a.output/'run.json')
        if any(run.get(k)!=v for k,v in bindings.items()):raise ValueError('resume source/protocol/paths changed')
    probes=[]
    for card in cards:
        path=a.output/f'device-{card}-{time.time_ns()}.json'
        rc=execute(command(a)+['--probe',str(path)],path.with_suffix('.log'),env|dict(CUDA_VISIBLE_DEVICES=card))
        if rc:raise ValueError('device precheck failed: '+str(path.with_suffix('.log')))
        probes.append(read(path))
    check_devices(probes)
    if a.resume and probes!=run['devices']:raise ValueError('resume physical devices changed')
    # Build once on the host; workers share verified read-only payloads.
    images,dq=refresh.compile_all(a,p)
    (a.output/'native').mkdir(exist_ok=True)
    _,receipt=native.build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'native')
    binary=native.carrier(a,a.output/'native')
    value=bindings|dict(devices=probes,assignments=assign(p,len(cards)),images=refresh.image_identity(images,dq),
        native=dict(receipt=receipt,carrier=sha(binary)),production_changed=False)
    if a.resume:
        if run!={**value,'sha256':digest(value)}:raise ValueError('resume compiled payload changed')
    else:publish(a.output/'run.json',value);run=sealed(a.output/'run.json')
    refresh.lock_clocks(a,probes)
    children=[];logs=[];finished=set();start=time.monotonic()
    try:
        for i,card in enumerate(cards):
            directory=a.output/('gpu-'+card);directory.mkdir(exist_ok=a.resume)
            log=directory/f'console-{time.time_ns()}.log';stream=log.open('x');logs.append(stream)
            child=subprocess.Popen(command(a)+['--shard',str(i)],env=env|dict(CUDA_VISIBLE_DEVICES=card),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,log))
        while any(proc.poll() is None for proc,_ in children):
            for i,(proc,log) in enumerate(children):
                if proc.poll() is not None and i not in finished:
                    print(f'SMALLM_NATIVE_WORKER_EXIT gpu={cards[i]} rc={proc.returncode} log={log}',flush=True);finished.add(i)
            print('SMALLM_NATIVE_WAIT '+json.dumps(dict(alive=sum(c.poll() is None for c,_ in children),
                workers=len(children),elapsed_minutes=(time.monotonic()-start)/60,output=str(a.output))),flush=True)
            time.sleep(30)
        return review(a,p,run,[c.returncode for c,_ in children])
    finally:
        for child,_ in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child,_ in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        for log in logs:log.close()


if __name__=='__main__':raise SystemExit(main())
