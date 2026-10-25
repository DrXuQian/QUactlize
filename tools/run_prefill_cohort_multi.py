#!/usr/bin/env python3
"""Run disjoint dense Asys families on explicitly idle GPUs; retain every run."""
import argparse
from contextlib import ExitStack
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
from dev.prefill_asys.cohort import validate
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_native_components_multi import worker_failure


def gpu_ordinals(value):
    values=value.split(',')
    if not 1<=len(values)<=64 or any(not g.isascii() or not g.isdigit() for g in values):
        raise ValueError('one to 64 GPU ordinals required')
    gpus=[str(int(g)) for g in values]
    if len(set(gpus))!=len(gpus):raise ValueError('distinct physical GPU ordinals required')
    return gpus


def worker_limits(jobs,configs_per_batch):
    for flag,value,maximum in (('--jobs',jobs,64),('--configs-per-batch',configs_per_batch,32)):
        if type(value) is not int or not 1<=value<=maximum:
            raise ValueError(f'{flag} must be in 1..{maximum}, got {value!r}')
    return dict(jobs=jobs,configs_per_batch=configs_per_batch)


def worker_environment(output,gpu,index,count,revision,*,finalists=False,jobs=8,configs_per_batch=16,no_native=False):
    limits=worker_limits(jobs,configs_per_batch)
    env=dict(os.environ,GPU=gpu,SHARD=str(index),SHARDS=str(count),
        OUTPUT_ROOT=str(output/('gpu-'+gpu)),TC_CACHE=str(output/'tc-cache'),SOURCE_REV=revision,IDLE_CONFIRMED='1',
        JOBS=str(limits['jobs']),CONFIGS_PER_BATCH=str(limits['configs_per_batch']),
        NATIVE_REFERENCE='0' if no_native else '1')
    for key in ('RUN','RESUME_RUN','POINT_ID','PREVIOUS_RUN','PLAN_ONLY','COHORT','FINALIST_PLAN','MODEL'):
        env.pop(key,None)
    env['FINALIST_PLAN' if finalists else 'COHORT']=str(output/'cohort.json')
    return env


def retry_setup(previous,cohort,gpus,limits,no_native,locks):
    """Retry only the known pre-timing gate failure; never rewrite old evidence."""
    from dev.prefill_asys.finalists import load, partition
    previous=previous.resolve(strict=True)
    held=locks.enter_context((previous/'.coordinator.lock').open('r'))
    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('previous coordinator is still running') from None
    before=json.loads((previous/'coordinator.json').read_text())
    old=load(json.loads((previous/'cohort.json').read_text()))
    if (cohort['schema']!='quactlize.dense-m-span.v1' or digest(old)!=digest(cohort) or
            before['cohort']!=cohort['sha256'] or before['gpus']!=gpus or
            before.get('worker_limits')!=limits or before.get('native_reference',True)!=(not no_native)):
        raise ValueError('setup retry cohort/GPU assignment/limits/reference mode changed')
    if before.get('setup_retry') or not list(previous.glob('summary-*.json')):
        raise ValueError('setup retry requires a finished original coordinator; resume its retry separately')
    retained={};failures={}
    for index,gpu in enumerate(gpus):
        runs=[r for r in (previous/('gpu-'+gpu)).glob('qwen32-prefill-asys.*') if r.is_dir()]
        if len(runs)!=1:raise ValueError('setup retry needs exactly one retained run for GPU '+gpu)
        run=runs[0];output=run/'results';plan=json.loads((output/'plan.json').read_text())
        expected=partition(cohort['points'],index,len(gpus))
        if no_native:expected=[dict(p,native_comparison=False) for p in expected]
        identity=dict(finalist_plan_sha256=cohort['sha256'],shard=index,shards=len(gpus),points=[])
        if no_native:identity['native_reference']=False
        if (plan.get('model_identity')!=identity or digest(plan['points'])!=digest(expected) or
                plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'})):
            raise ValueError('setup retry original worker plan changed for GPU '+gpu)
        summary_path=output/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        if summary.get('status')=='SCREEN_COMPLETE':
            authority=json.loads((output/'authority.json').read_text())
            execution=json.loads((output/'execution-plan.json').read_text())
            expected_ids={p['id'] for p in expected}
            if (authority['plan']!=plan['plan_sha256'] or execution['authority']!=digest(authority) or
                    {p['id'] for p in execution['points']}!=expected_ids or
                    {p['id'] for p in summary['points']}!=expected_ids or
                    any(p['status']!='SCREEN_COMPLETE' for p in summary['points'])):
                raise ValueError('completed worker coverage/authority differs for GPU '+gpu)
            # The final review still validates every raw PASS/numerical receipt.
            retained[gpu]=dict(run=str(run),authority=digest(authority),summary_sha256=sha(summary_path))
            continue
        if (any((output/name).exists() for name in ('authority.json','execution-plan.json','batches')) or summary):
            raise ValueError('worker has partial measurements; use its frozen resume, not setup retry: '+str(run))
        logs=[p for p in output.glob('reducer-gate-*.log')
              if 'ValueError: reducer gate workload differs from native dense prefill' in p.read_text(errors='replace')]
        if not logs:raise ValueError('not the known reducer setup failure: '+str(run))
        failures[gpu]=dict(run=str(run),logs={str(p):sha(p) for p in sorted(logs)})
    if not failures:raise ValueError('no reducer setup failures to retry')
    return dict(previous=str(previous),revision=before['revision'],retained=retained,failures=failures)


def main():
    def terminate(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser(description=__doc__)
    plans=p.add_mutually_exclusive_group(required=True)
    plans.add_argument('--cohort',type=Path)
    plans.add_argument('--finalist-plan',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=8,help='compile jobs per worker; does not inherit shell JOBS')
    p.add_argument('--configs-per-batch',type=int,default=16,help='explicit per-capture budget, not shell CONFIGS_PER_BATCH')
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--no-native',action='store_true',help='Quactlize M-span selection without llama reference')
    recovery=p.add_mutually_exclusive_group()
    recovery.add_argument('--resume',action='store_true')
    recovery.add_argument('--retry-setup-from',type=Path,
                          help='new source for pre-timing reducer M-limit failures; retain complete shards')
    a=p.parse_args()
    try:
        gpus=gpu_ordinals(a.gpus)
        limits=worker_limits(a.jobs,a.configs_per_batch)
    except ValueError as e:p.error(str(e))
    if not a.confirm_idle:p.error('--confirm-idle required')
    if a.finalist_plan:
        from dev.prefill_asys.finalists import load as validate_finalists, partition
        cohort=validate_finalists(json.loads(a.finalist_plan.read_text()))
        partition(cohort['points'],0,len(gpus))  # Reject excess GPUs before starting workers.
    else:cohort=validate(json.loads(a.cohort.read_text()))
    if a.no_native and cohort['schema']!='quactlize.dense-m-span.v1':
        p.error('--no-native requires a dense M-span plan')
    if a.retry_setup_from and not a.finalist_plan:p.error('--retry-setup-from requires --finalist-plan')
    with ExitStack() as locks:
        return run_cohort(a,cohort,gpus,limits,locks)


def run_cohort(a,cohort,gpus,limits,locks):
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=a.resume)
    held=locks.enter_context((a.output/'.coordinator.lock').open('a'))
    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('this cohort already has a running coordinator') from None
    receipt=a.output/'coordinator.json'
    retry=None
    if a.resume:
        before=json.loads(receipt.read_text())
        if before['gpus']!=gpus or before['cohort']!=cohort['sha256']:
            raise ValueError('resume GPU assignment/cohort changed')
        if before.get('worker_limits')!=limits:
            raise ValueError('resume worker limits changed or were not recorded; do not relabel old timings')
        if before.get('native_reference',True)!=(not a.no_native):
            raise ValueError('resume native reference mode changed; use a new output directory')
        rev=before['revision']
        if before.get('setup_retry'):
            retry=retry_setup(Path(before['setup_retry']['previous']),cohort,gpus,limits,a.no_native,locks)
            if retry!=before['setup_retry']:raise ValueError('setup retry original evidence changed')
    else:
        if a.retry_setup_from:
            retry=retry_setup(a.retry_setup_from,cohort,gpus,limits,a.no_native,locks)
        rev=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        atomic_json(a.output/'cohort.json',cohort)
        atomic_json(receipt,dict(gpus=gpus,cohort=cohort['sha256'],revision=rev,worker_limits=limits,
                                native_reference=not a.no_native,**(dict(setup_retry=retry) if retry else {})))
    retained=retry['retained'] if retry else {}
    if retry:
        print('PREFILL_COHORT_SETUP_RETRY '+json.dumps(dict(previous=retry['previous'],
            rerun=list(retry['failures']),retained=list(retained),old_evidence_unchanged=True)),flush=True)
    print('PREFILL_COHORT_START '+json.dumps(dict(gpus=gpus,worker_limits=limits,revision=rev,
                                               native_reference=not a.no_native)),flush=True)
    children=[];streams=[];start=time.monotonic()
    try:
        for index,gpu in enumerate(gpus):
            if gpu in retained:continue
            directory=a.output/('gpu-'+gpu);directory.mkdir(exist_ok=a.resume)
            env=worker_environment(a.output,gpu,index,len(gpus),rev,finalists=bool(a.finalist_plan),
                                   no_native=a.no_native,**limits)
            if retry and (Path(retry['previous'])/'tc-cache').is_dir():
                # Content-addressed modules validate source/SDK identities;
                # reuse binaries, never import a previous timing as a new one.
                env['TC_CACHE']=str(Path(retry['previous'])/'tc-cache')
            if a.resume:
                runs=[r for r in directory.glob('qwen32-prefill-asys.*') if r.is_dir()]
                if len(runs)!=1:raise ValueError('resume needs exactly one retained run in '+str(directory))
                env['RESUME_RUN']=str(runs[0])
            log=directory/('console-'+str(time.time_ns())+'.log')
            stream=log.open('x');streams.append(stream)
            process=subprocess.Popen(['bash',str(ROOT/'tools/run_prefill_model_asys_box.sh')],
                env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((process,log))
        while any(proc.poll() is None for proc,_ in children):
            print('PREFILL_COHORT_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p,_ in children),
                workers=len(children),elapsed_minutes=(time.monotonic()-start)/60,output=str(a.output))),flush=True)
            time.sleep(30)
        codes=[proc.returncode for proc,_ in children]
        failures=[worker_failure(log.parent,proc.returncode,log=log)
                  for proc,log in children if proc.returncode]
        for failure in failures:
            print('PREFILL_COHORT_WORKER_FAIL '+json.dumps(failure),flush=True)
        runs=[r for gpu in gpus for r in (a.output/('gpu-'+gpu)).glob('qwen32-prefill-asys.*')
              if (r/'results/execution-plan.json').is_file()]
        runs.extend(Path(row['run']) for row in retained.values())
        review=a.output/('review-'+str(time.time_ns())+'.json.xz')
        command=[sys.executable,str(ROOT/'tools/review_prefill_cohort.py'),'--output',str(review)]
        for run in runs:command+=['--run',str(run)]
        rc=subprocess.run(command).returncode if runs else 1
        result=dict(returncodes=codes,worker_failures=failures,review_rc=rc,review=str(review),
                    review_written=review.is_file(),production_changed=False,
                    native_comparison=not a.no_native,no_regression_admitted=False,
                    retained_workers=list(retained),expected_workers=len(gpus),measured_workers=len(runs),
                    status='SCREEN_COMPLETE' if not any(codes) and rc==0 and len(runs)==len(gpus) else 'INCOMPLETE')
        atomic_json(a.output/('summary-'+str(time.time_ns())+'.json'),result)
        print('PREFILL_COHORT_DONE '+json.dumps(result),flush=True)
        return int(result['status']!='SCREEN_COMPLETE')
    finally:
        # Only our own new process groups. Never kill by executable name.
        for proc,_ in children:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass  # may exit between poll and kill
        for proc,_ in children:
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        for stream in streams:stream.close()


if __name__=='__main__':raise SystemExit(main())
