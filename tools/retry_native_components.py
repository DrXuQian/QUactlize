#!/usr/bin/env python3
"""Retry the reviewed 268 native failures; preserve 1178 results, defer 98 FP16 issues.

No caller rebuild, configuration sweep, provider override or helper remeasurement.
Use a NEW output root. Reusing that retry root subsequently resumes only its
failed/missing points. The original run is always read-only.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.plan import plan
from dev.native_components.retry import (inspect_previous,verify_previous,verify_diagnostic,
    validate_current,same_record,comparison_summary)
from dev.native_components.history import check_cohort
from dev.llama_ab.contract import family
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_native_components as native
from tools.run_native_components_multi import pack,worker_failure


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('retry input/result changed: '+str(path))
    else:atomic_json(path,value)


def worker(args,audit,probe,work):
    previous=audit['workers'][args.worker];old=args.previous/args.worker
    rows=[r for r in work['points']if r['id']in previous['summary']['results']]
    history=json.loads((ROOT/'docs/measurements/kpack_native_history_20261019.json').read_text())
    if history['sha256']!=digest({k:v for k,v in history.items()if k!='sha256'}) or history['source_plan']!=work['sha256']:
        raise ValueError('historical component baseline changed')
    env=native.environment(args)
    os.environ.update({k:v for k,v in env.items()if k in ('LD_LIBRARY_PATH','PPU_SDK','CUDA_HOME')})
    dev=native.device(args);cohort=check_cohort(dev)
    if dev!=previous['authority']['device']:raise ValueError('retry must use the original physical GPU/SDK')
    args.output.mkdir(parents=True,exist_ok=True)
    harness=harness_identity(ROOT)
    lock=Path('/tmp')/('quactlize-components-'+digest(dev['device']['pci'])[:24]+'.lock')
    with lock.open('a')as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        adapter,receipt=native.build_adapter(args.llama_source,args.llama_build,args.sdk,args.output/'build')
        validate_current(previous,receipt,dev,probe)
        binary=native.carrier(args,args.output/'build')
        frozen=dict(previous=str(old),previous_receipts_sha256=audit['files_sha256'],
            diagnostic_sha256=probe['archive_sha256'],points=rows,
            retry=previous['retry'],reused=previous['reused'],deferred=previous['deferred'])
        freeze(args.output/'retry.json',frozen)
        authority=dict(plan=digest(frozen),device=dev,adapter=receipt['binary_sha256'],
            harness=harness['sha256'],backend=receipt['identity']['libraries'],carrier=sha(binary),
            rounds=3,samples=3,scope='NATIVE_COMPONENTS_FAILED_ONLY_RETRY',
            historical_cohort=cohort,previous_authority=previous['authority']['sha256'],
            diagnostic_sha256=probe['archive_sha256'])
        authority['sha256']=digest(authority);freeze(args.output/'authority.json',authority)
        helpers,bindings=native.helper_jobs(rows)
        if any(previous['helpers'].get(h['id'],{}).get('contract')!=h for h in helpers):
            raise ValueError('previous helper contracts differ from current workload')
        helpers=previous['helpers']
        for key,value in helpers.items():freeze(args.output/'helpers'/(key+'.json'),value)
        records={};groups=defaultdict(list)
        for row in rows:
            pid=row['id'];target=args.output/'results'/(pid+'.json')
            if pid in previous['reused']:
                old_path=old/'results'/(pid+'.json')
                value=json.loads(old_path.read_text());same_record(value,previous['summary']['results'][pid])
                value=native.load_measured(old_path,previous['authority'],row)
                if value is None:raise ValueError('reviewed successful result disappeared')
                raw=native.validated(Path(value['path']),previous['authority'])
                if value.get('data')!=raw:raise ValueError('embedded previous result differs from original raw receipt')
                freeze(target,value);records[pid]=value
            elif pid in previous['deferred']:
                value=dict(status='DEFERRED_NATIVE_FP16_ACCURACY',
                    reason='USER_DEFERRED_NATIVE_LIBRARY_ISSUE_NOT_A_PASS',
                    previous=previous['summary']['results'][pid],production_changed=False)
                freeze(target,value);records[pid]=value
            else:
                value=native.load_measured(target,authority,row)
                if value is not None:records[pid]=value
                else:groups[family(row)].append(native.point_request(row,dev['l2']['bytes']))
        args.reuse_fixtures=old/'fixtures'
        runtime=next((p for p in (args.sdk/'lib/libhggc_wrapper.so',args.sdk/'targets/x86_64-linux/lib/libhggc_wrapper.so')if p.is_file()),None)
        if runtime is None:raise ValueError('matching runtime wrapper missing')
        request=dict(runtime_library=str(runtime),adapter_library=str(adapter),authority=authority,rounds=3,samples=3)
        print('NATIVE_RETRY_WORKER '+json.dumps(dict(worker=args.worker,reused=len(previous['reused']),
            deferred=len(previous['deferred']),pending=sum(map(len,groups.values())),helpers_reused=len(helpers))),flush=True)
        start=time.monotonic()
        for i,group in enumerate(groups.values(),1):
            if harness_identity(ROOT)!=harness:raise ValueError('retry source changed during measurement')
            print(f'NATIVE_RETRY_FAMILY worker={args.worker} index={i}/{len(groups)} points={len(group)} elapsed_minutes={(time.monotonic()-start)/60:.1f}',flush=True)
            records.update(native.run_family(args,binary,request,group,authority))
        if harness_identity(ROOT)!=harness:raise ValueError('retry source changed during measurement')
    compared=native.comparison_rows(rows,records,helpers,bindings,history)
    for row in compared:
        if row['id']in previous['deferred']:row['status']='DEFERRED_NATIVE_FP16_ACCURACY'
    native.write_comparison(args.output,compared)
    summary=comparison_summary(records,compared,previous)
    summary['results']={k:{a:b for a,b in v.items()if a!='data'}for k,v in records.items()}
    atomic_json(args.output/'summary.json',summary)
    print('NATIVE_RETRY_DONE '+json.dumps({k:v for k,v in summary.items()if k not in('results','deferred_ids')}),flush=True)
    return 0 if summary['status']=='COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16'else 1


def coordinator(args,audit,probe):
    cards=args.devices.split(',');expected=[k.removeprefix('gpu-')for k in audit['workers']]
    if cards!=expected:raise ValueError('use the original devices/order: '+','.join(expected))
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--previous',str(args.previous),
        '--diagnostic',str(args.diagnostic),'--llama-source',str(args.llama_source),
        '--llama-build',str(args.llama_build),'--sdk',str(args.sdk),
        '--l2-bytes',str(args.l2_bytes),'--confirm-idle']
    freeze(args.output/'run.json',dict(command=command,devices=cards,
        previous_receipts_sha256=audit['files_sha256'],diagnostic_sha256=probe['archive_sha256'],
        harness=harness_identity(ROOT),mode='FAILED_ONLY_NEW_AUTHORITY_OLD_VALID_RESULTS_UNCHANGED'))
    # Probe physical cards before launching workers; never initialize GPU0 in parent.
    for card in cards:
        directory=args.output/('probe-'+card);directory.mkdir(exist_ok=True)
        with (directory/'probe.log').open('a')as log:
            subprocess.run([sys.executable,str(ROOT/'tools/run_native_components.py'),
                '--probe-only','--output',str(directory),'--llama-source',str(args.llama_source),
                '--llama-build',str(args.llama_build),'--sdk',str(args.sdk),
                '--l2-bytes',str(args.l2_bytes),'--confirm-idle'],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,check=True)
        dev=json.loads((directory/'device.json').read_text())
        if dev!=audit['workers']['gpu-'+card]['authority']['device']:
            raise ValueError('physical card/SDK changed at ordinal '+card)
    children=[];outputs=[];start=time.monotonic();aborted=False
    try:
        for card in cards:
            output=args.output/('gpu-'+card);output.mkdir(exist_ok=True);outputs.append(output)
            log=(output/'console.log').open('a')
            child=subprocess.Popen([*command,'--worker','gpu-'+card,'--output',str(output)],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((child,log))
        last=0
        while any(child.poll()is None for child,_ in children):
            now=time.monotonic()
            if now-last>=30:
                measured=sum(json.loads(p.read_text()).get('status')=='MEASURED'
                    for out in outputs for p in (out/'results').glob('*.json')
                    if p.stem in audit['workers'][out.name]['retry'])
                print(f'NATIVE_RETRY_WAIT alive={sum(c.poll()is None for c,_ in children)}/{len(cards)} '
                    f'retried_measured={measured}/268 elapsed_minutes={(now-start)/60:.1f} '
                    f'logs={args.output}/gpu-*/console.log',flush=True)
                last=now
            time.sleep(1)
    except BaseException:
        aborted=True
        for child,_ in children:
            if child.poll()is None:os.killpg(child.pid,signal.SIGTERM)
        for child,_ in children:child.wait()
        raise
    finally:
        for _,log in children:log.close()
        if aborted:print('results='+str(pack(args.output,outputs)),flush=True)
    returncodes=[child.returncode for child,_ in children]
    summaries=[json.loads((out/'summary.json').read_text())if(out/'summary.json').is_file()else{}for out in outputs]
    complete=not any(returncodes)and all(
        s.get('status')=='COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16' and
        s.get('retried_measured')==len(audit['workers'][out.name]['retry']) and
        s.get('measured')==len(audit['workers'][out.name]['retry'])+len(audit['workers'][out.name]['reused']) and
        s.get('deferred')==len(audit['workers'][out.name]['deferred'])
        for out,s in zip(outputs,summaries))
    result=dict(status='COMPLETE_EXCEPT_DEFERRED_NATIVE_FP16'if complete else'INCOMPLETE',
        returncodes=returncodes,reused=1178,retry_expected=268,
        retried_measured=sum(s.get('retried_measured',0)for s in summaries),deferred_native_accuracy=98,
        measured=sum(s.get('measured',0)for s in summaries),expected=1544,
        full_coverage=False,no_regression_admitted=False,production_changed=False,
        scope='HISTORICAL_COMPONENT_SCREEN_NOT_MODEL_TPOT',
        failures=[worker_failure(out,rc)for out,rc in zip(outputs,returncodes)if rc])
    atomic_json(args.output/'summary.json',result)
    print('NATIVE_RETRY_MULTI_DONE '+json.dumps(result),flush=True)
    print('results='+str(pack(args.output,outputs)),flush=True)
    return 0 if complete else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker')
    p.add_argument('--devices',default='1,2,3,4,5,7')
    p.add_argument('--llama-source',type=Path,default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build',type=Path);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--confirm-idle',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();a.previous=a.previous.resolve(strict=True);a.diagnostic=a.diagnostic.resolve(strict=True);a.output=a.output.resolve()
    if a.output.is_relative_to(a.previous) or a.previous.is_relative_to(a.output):
        p.error('use a separate retry output, never the previous run or its parent')
    work=plan();audit=inspect_previous(a.previous,work['points']);verify_previous(audit)
    probe=verify_diagnostic(a.diagnostic)
    print('NATIVE_RETRY_PLAN '+json.dumps(dict(audit['counts'],
        scope='268_FAILED_ONLY_98_NATIVE_ACCURACY_DEFERRED',gpu_work=not a.plan_only)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:p.error('--confirm-idle required')
    a.llama_source=a.llama_source.resolve(strict=True)
    a.llama_build=(a.llama_build or a.llama_source/'build-ci').resolve(strict=True);a.sdk=a.sdk.resolve(strict=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.worker:
        if a.worker not in audit['workers']:p.error('unknown previous worker')
        return worker(a,audit,probe,work)
    return coordinator(a,audit,probe)


if __name__=='__main__':raise SystemExit(main())
