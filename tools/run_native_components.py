#!/usr/bin/env python3
"""Native llama independent-component benchmark; any matching caller branch.

This does not run K-pack through llama or change its source/build. It compiles
a narrow test adapter using the existing target flags. No GEMM config sweep.
Full coverage is 1544 workloads. --plan-only requires no SDK/GPU/caller.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.plan import plan
from dev.native_components.build import build_adapter
from dev.native_components.accounting import checked_component
from dev.native_components.history import check_cohort
from dev.llama_ab.contract import family, pilot
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_kpack_llama_ab import point_request, device, identities


def helper_jobs(rows):
    from dev.strategy.prefill_fixture import routing
    from quactlize.dispatch.table import Workload
    jobs={};bindings={}
    for row in rows:
        ids=[]
        if 'MEASURE_GPU_ROW_MAP_CREATION'in row['pending']:
            w=Workload(**row['workload']);route=routing(w,0)
            contract={k:row['workload'][k]for k in ('m','experts','topk','channels','router')}
            identity=dict(kind='routing',contract=contract,router_sha256=route['sha256'],
                          input_ids_stride=w.topk,input_token_stride=w.channels,write_inverse=False)
            key=digest(identity)
            jobs.setdefault(key,dict(identity,id=key,workload=contract,owners=list(map(int,route['owners']))))
            ids.append(key)
        if 'MEASURE_FOUR_BYTE_ERROR_RESET'in row['pending']:
            identity=dict(kind='reset',bytes=4,alignment=256,value=0,scope='FULL_PREFILL_PER_CALL_ERROR_STATUS')
            key=digest(identity);jobs.setdefault(key,dict(identity,id=key));ids.append(key)
        bindings[row['id']]=ids
    return list(jobs.values()),bindings


def measure_helpers(args,binary,request,rows):
    jobs,bindings=helper_jobs(rows);records={}
    for job in jobs:
        saved=args.output/'helpers'/(job['id']+'.json')
        if saved.exists():
            previous=json.loads(saved.read_text())
            if previous.get('authority')!=request['authority'] or previous.get('contract')!=job:
                raise ValueError('helper source/device/shape changed')
            if 'component' in previous:
                if previous.get('record_sha256')!=digest({k:v for k,v in previous.items()if k!='record_sha256'}):
                    raise ValueError('saved helper record changed')
                checked_component(previous['component']);records[job['id']]=previous;continue
        attempt=args.output/'helper-attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        atomic_json(attempt/'request.json',dict(request,helper_points=[job]))
        with (attempt/'run.log').open('x') as log:
            result=subprocess.run([str(binary),str(attempt/'request.json'),str(attempt)],
                                  env=environment(args),stdout=log,stderr=subprocess.STDOUT)
        path=attempt/(job['id']+'.json')
        if result.returncode or not path.is_file():
            records[job['id']]=dict(status='FAIL',rc=result.returncode,log=str(attempt/'run.log'),
                                    authority=request['authority'],contract=job)
            atomic_json(saved,records[job['id']])
        else:
            value=json.loads(path.read_text())
            if value['authority']!=request['authority'] or value['contract']!=job:
                raise ValueError('helper receipt identity differs')
            checked_component(value['component'])
            value['record_sha256']=digest(value)
            atomic_json(saved,value);records[job['id']]=value
    return records,bindings


def comparison_rows(rows, records, helpers, bindings, history):
    historical={p['id']:p for p in history['points']};result=[]
    for row in rows:
        prior=historical[row['id']];record=records.get(row['id'],{})
        missing=[k for k in bindings[row['id']] if 'component'not in helpers.get(k,{})]
        terms=[dict(id=k,**checked_component(helpers[k]['component']))for k in bindings[row['id']]if k not in missing]
        total=prior['sum_us']+sum(c['median_us']for c in terms)if not missing else None
        native=record.get('sum_us')
        result.append(dict(id=row['id'],workload=row['workload'],
            status='HISTORICAL_SCREEN_ONLY'if native and total is not None else 'INCOMPLETE',
            quactlize_core_components=prior['components'],quactlize_supplement=terms,
            quactlize_sum_us=total,native_sum_us=native,
            delta_pct=100*(total/native-1)if native and total is not None else None,
            missing_helpers=missing,native_status=record.get('status','MISSING'),
            no_regression_admitted=False,model_tpot_admitted=False))
    return result


def write_comparison(output, rows):
    atomic_json(output/'comparison.json',dict(points=rows,
        scope='HISTORICAL_COMPONENT_SUM_SCREEN_NOT_PAIRED_OR_MODEL_TPOT'))
    fields=['id','operation','qtype','m','n','k','experts','topk','channels','compute','router',
            'quactlize_sum_us','native_sum_us','delta_pct','status']
    with (output/'comparison.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            values=dict(row['workload'],**{k:row[k]for k in ('id','quactlize_sum_us','native_sum_us','delta_pct','status')})
            writer.writerow({k:values.get(k)for k in fields})


def environment(args):
    env=dict(os.environ)
    forbidden=('GGML_CUDA_FORCE_', 'GGML_QUACTLIZE_', 'QUACTLIZE_KPACK_', 'QUACTLIZE_PPU_')
    for k in list(env):
        if k.startswith(forbidden):env.pop(k)
    env.pop('GGML_CUDA_CUBLAS_COMPUTE_TYPE',None)
    env.update(GGML_CUDA_DISABLE_GRAPHS='1',GGML_QUACTLIZE_DENSE='0',GGML_CUDA_PDL='0',
        PPU_SDK=str(args.sdk),CUDA_HOME=str(args.sdk/'CUDA_SDK'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    folders=[args.llama_build/'bin',args.sdk/'lib',args.sdk/'targets/x86_64-linux/lib',
             args.sdk/'CUDA_SDK/targets/x86_64-linux/lib']
    env['LD_LIBRARY_PATH']=':'.join(map(str,folders))+':'+env.get('LD_LIBRARY_PATH','')
    return env


def carrier(args, output):
    binary=output/'native-components-carrier'
    source=ROOT/'dev/native_components/carrier.cpp'
    command=['g++','-std=c++17','-O2','-Wall','-Wextra',
        '-I'+str(args.llama_source/'ggml/include'),'-I'+str(args.llama_source/'ggml/src'),
        '-I'+str(args.llama_source/'vendor'),str(source),'-L'+str(args.llama_build/'bin'),
        '-Wl,-rpath,'+str(args.llama_build/'bin'),'-Wl,-rpath-link,'+str(args.llama_build/'bin'),
        '-Wl,-rpath-link,'+str(args.sdk/'lib'),
        '-Wl,-rpath-link,'+str(args.sdk/'targets/x86_64-linux/lib'),
        '-Wl,-rpath-link,'+str(args.sdk/'CUDA_SDK/targets/x86_64-linux/lib'),
        '-lggml-cuda','-lggml-base','-ldl','-pthread','-o',str(binary)]
    identity=dict(command=command,source=sha(source),fixture=sha(ROOT/'dev/llama_ab/carrier.cpp'))
    receipt=output/'carrier.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['identity']!=identity or old['binary_sha256']!=sha(binary):
            raise ValueError('native fixture changed; use a new output')
        return binary
    if binary.exists():raise ValueError('unreceipted native carrier')
    with (output/'carrier-build.log').open('x') as log:
        subprocess.run(command,env=environment(args),stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(receipt,dict(identity=identity,binary_sha256=sha(binary)))
    return binary


def validated(path, authority):
    data=json.loads(path.read_text())
    if data.get('authority')!=authority or data.get('correctness')!='PASS':
        raise ValueError('native measurement authority/correctness differs')
    if data.get('id')!=path.stem:
        raise ValueError('native measurement point differs from filename')
    if (data.get('linked_backend_comparison')!='PASS_TWO_INPUTS' or data.get('omitted_producer')!='DETECTED'
            or data.get('last_weight_copy_negative')!='DETECTED_COPY0_UNCHANGED'):
        raise ValueError('native extraction numerical controls missing')
    if not data.get('components'):raise ValueError('empty component inventory')
    for component in data['components']:checked_component(component)
    expected=authority['backend']
    actual=identities([Path(x) for x in data['loaded_images']])
    if any(expected[p]!=h for p,h in actual.items()if p in expected):
        raise ValueError('loaded library changed during measurement')
    adapters=[h for p,h in actual.items()if Path(p).name=='libnative-components.so']
    if adapters!=[authority['adapter']]:
        raise ValueError('loaded native recording adapter differs')
    for stem in ('libggml-cuda.so','libggml-base.so'):
        matching={p:h for p,h in actual.items()if Path(p).name.startswith(stem)}
        if not matching or any(expected.get(p)!=h for p,h in matching.items()):
            raise ValueError('loaded native backend differs from build receipt')
    data['loaded_image_sha256']=actual
    return data


def run_family(args,binary,request,rows,authority):
    from dev.native_components.fixture import weights,activation
    try:
        raw,receipt=weights(rows[0]['workload'],args.output/'fixtures'/family(rows[0]))
        request=dict(request,raw_fixture=str(raw),raw_fixture_receipt=receipt)
        rows=[dict(r,input_fixture=activation(r))for r in rows]
    except Exception as exc:
        records={}
        for row in rows:
            record=dict(status='FAIL',reason='FAMILY_FIXTURE_FAILED',error=str(exc),authority=authority)
            records[row['id']]=record
            atomic_json(args.output/'results'/(row['id']+'.json'),record)
        print(f'NATIVE_COMPONENT_FAIL family={family(rows[0])} error={exc} remaining_continue=1',flush=True)
        return records
    pending=list(rows);records={}
    # Abort/assert invalidates a process, never the remaining family/points.
    while pending:
        attempt=args.output/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        current=dict(request,points=pending)
        atomic_json(attempt/'request.json',current)
        started=time.monotonic()
        with (attempt/'run.log').open('x') as log:
            child=subprocess.Popen([str(binary),str(attempt/'request.json'),str(attempt)],
                                   env=environment(args),stdout=log,stderr=subprocess.STDOUT)
            last=started
            while child.poll() is None:
                time.sleep(1)
                if time.monotonic()-last>=30:
                    print(f'NATIVE_COMPONENT_WAIT remaining_points={len(pending)} '
                          f'minutes={(time.monotonic()-started)/60:.1f} log={attempt / "run.log"}',flush=True)
                    last=time.monotonic()
        completed=[]
        for row in pending:
            path=attempt/(row['id']+'.json')
            if path.is_file():
                data=validated(path,authority)
                if data['workload']!=row['workload']:
                    raise ValueError('native workload differs from frozen point')
                component_sum=sum(checked_component(c)['median_us'] for c in data['components'])
                records[row['id']]=dict(status='MEASURED',data=data,sum_us=component_sum,
                    scope='SUM_OF_INDEPENDENT_COMPONENTS_NOT_MEASURED_E2E_OR_TPOT',
                    path=str(path),sha256=sha(path))
                atomic_json(args.output/'results'/(row['id']+'.json'),records[row['id']])
                completed.append(row['id'])
        pending=[r for r in pending if r['id'] not in completed]
        if pending:
            text=(attempt/'run.log').read_text(errors='replace')
            if 'NATIVE_COMPONENT_START'not in text:
                for failed in pending:
                    record=dict(status='FAIL',rc=child.returncode,log=str(attempt/'run.log'),
                                reason='FAMILY_SETUP_FAILED',authority=authority)
                    records[failed['id']]=record
                    atomic_json(args.output/'results'/(failed['id']+'.json'),record)
                break
            failed=pending.pop(0)
            record=dict(status='FAIL',rc=child.returncode,log=str(attempt/'run.log'),
                        reason='CHILD_FAILED_OR_OMITTED_POINT',authority=authority)
            records[failed['id']]=record
            atomic_json(args.output/'results'/(failed['id']+'.json'),record)
            print(f'NATIVE_COMPONENT_FAIL point={failed["id"]} rc={child.returncode} remaining_continue=1',flush=True)
    return records


def load_measured(path,authority,row):
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value.get('status')!='MEASURED':return None
    raw=Path(value['path'])
    if value['sha256']!=sha(raw):raise ValueError('saved native record changed')
    data=validated(raw,authority)
    if data['id']!=row['id'] or data['workload']!=row['workload']:
        raise ValueError('saved native point/workload changed')
    total=sum(checked_component(c)['median_us']for c in data['components'])
    if value['sum_us']!=total:raise ValueError('saved native sum differs from raw samples')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-source',type=Path,default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build',type=Path)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--phase',choices=('all','pilot'),default='all')
    p.add_argument('--cohort',choices=('all','decode','prefill'),default='all')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--l2-bytes',type=int,default=0)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--history',type=Path,default=ROOT/'docs/measurements/kpack_native_history_20261019.json')
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.rounds<3 or a.samples<3:p.error('invalid shard/sample count')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    work=plan()
    rows=work['points']
    if a.cohort!='all':rows=[r for r in rows if r['cohort']==a.cohort]
    if a.phase=='pilot':rows=pilot(rows)
    families=sorted({family(r)for r in rows})
    selected=set(families[a.shard::a.shards]);rows=[r for r in rows if family(r)in selected]
    frozen=dict(plan_sha256=work['sha256'],points=rows,phase=a.phase,cohort=a.cohort,
                shard=a.shard,shards=a.shards,full_denominator=1544)
    path=a.output/'plan.json'
    if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('frozen native inventory changed')
    if not path.exists():atomic_json(path,frozen)
    print('NATIVE_COMPONENT_PLAN '+json.dumps(dict(work['summary'],scheduled=len(rows),
        phase=a.phase,shard=a.shard,shards=a.shards,production_changed=False)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:p.error('--confirm-idle required; do not time alongside other GPU jobs')
    a.sdk=a.sdk.resolve(strict=True)
    a.llama_build=(a.llama_build or a.llama_source/'build-ci').resolve()
    env=environment(a)
    os.environ.update({k:v for k,v in env.items() if k in ('LD_LIBRARY_PATH','PPU_SDK','CUDA_HOME')})
    dev=device(a)
    cohort=check_cohort(dev)
    if a.probe_only:
        atomic_json(a.output/'device.json',dev)
        return 0
    a.llama_source=a.llama_source.resolve(strict=True)
    a.llama_build=(a.llama_build or a.llama_source/'build-ci').resolve(strict=True)
    a.sdk=a.sdk.resolve(strict=True)
    # No torch/DeepGEMM initialization in a multi-GPU parent; each process
    # sees exactly the one physical GPU selected by CUDA_VISIBLE_DEVICES.
    if ',' in os.environ.get('CUDA_VISIBLE_DEVICES',''):p.error('one visible GPU per shard process')
    env=environment(a)
    os.environ.update({k:v for k,v in env.items() if k in ('LD_LIBRARY_PATH','PPU_SDK','CUDA_HOME')})
    adapter,receipt=build_adapter(a.llama_source,a.llama_build,a.sdk,a.output/'build')
    binary=carrier(a,a.output/'build')
    harness=harness_identity(ROOT)
    authority=dict(plan=digest(frozen),device=dev,adapter=receipt['binary_sha256'],harness=harness['sha256'],
                   backend=receipt['identity']['libraries'],carrier=sha(binary),
                   rounds=a.rounds,samples=a.samples,scope='NATIVE_COMPONENTS_ONLY')
    authority['historical_cohort']=cohort
    authority['sha256']=digest(authority)
    auth=a.output/'authority.json'
    if auth.exists() and json.loads(auth.read_text())!=authority:raise ValueError('resume device/source/protocol differs')
    if not auth.exists():atomic_json(auth,authority)
    lock=Path('/tmp')/('quactlize-components-'+digest(dev['device']['pci'])[:24]+'.lock')
    records={};groups=defaultdict(list)
    for r in rows:
        old=a.output/'results'/(r['id']+'.json')
        v=load_measured(old,authority,r)
        if v is not None:
            records[r['id']]=v;continue
        groups[family(r)].append(point_request(r,dev['l2']['bytes']))
    with lock.open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=next((p for p in (a.sdk/'lib/libhggc_wrapper.so',a.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so') if p.is_file()),None)
        if runtime is None:raise ValueError('matching SDK runtime wrapper missing')
        request=dict(runtime_library=str(runtime),adapter_library=str(adapter),authority=authority,
                     rounds=a.rounds,samples=a.samples)
        helpers,bindings=measure_helpers(a,binary,request,rows)
        for index,group in enumerate(groups.values(),1):
            if harness_identity(ROOT)!=harness:raise ValueError('source changed during measurement')
            print(f'NATIVE_COMPONENT_FAMILY index={index}/{len(groups)} points={len(group)}',flush=True)
            records.update(run_family(a,binary,request,group,authority))
    if harness_identity(ROOT)!=harness:raise ValueError('source changed during measurement')
    history=json.loads(a.history.read_text())
    if history['sha256']!=digest({k:v for k,v in history.items()if k!='sha256'}) or history['source_plan']!=work['sha256']:
        raise ValueError('historical component baseline changed')
    compared=comparison_rows(rows,records,helpers,bindings,history)
    write_comparison(a.output,compared)
    summary=dict(status='COMPLETE' if len(records)==len(rows) and all(r['status']=='HISTORICAL_SCREEN_ONLY'for r in compared) else 'INCOMPLETE',
        expected=len(rows),measured=sum(v['status']=='MEASURED'for v in records.values()),
        results={k:{a:b for a,b in v.items() if a!='data'}for k,v in records.items()},
        helper_jobs=len(helpers),failed_helper_jobs=sum('component'not in v for v in helpers.values()),
        no_regression_admitted=False,model_tpot_admitted=False)
    atomic_json(a.output/'summary.json',summary)
    print('NATIVE_COMPONENT_DONE '+json.dumps({k:v for k,v in summary.items()if k!='results'}),flush=True)
    return 0 if summary['status']=='COMPLETE'else 1


if __name__=='__main__':raise SystemExit(main())
