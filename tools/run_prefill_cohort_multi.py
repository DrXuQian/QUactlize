#!/usr/bin/env python3
"""Run disjoint dense Asys families on explicitly idle GPUs; retain every run."""
import argparse
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


def gpu_ordinals(value):
    values=value.split(',')
    if not 1<=len(values)<=64 or any(not g.isascii() or not g.isdigit() for g in values):
        raise ValueError('one to 64 GPU ordinals required')
    gpus=[str(int(g)) for g in values]
    if len(set(gpus))!=len(gpus):raise ValueError('distinct physical GPU ordinals required')
    return gpus


def worker_environment(output,gpu,index,count,revision,*,finalists=False):
    env=dict(os.environ,GPU=gpu,SHARD=str(index),SHARDS=str(count),
        OUTPUT_ROOT=str(output/('gpu-'+gpu)),TC_CACHE=str(output/'tc-cache'),SOURCE_REV=revision,IDLE_CONFIRMED='1')
    for key in ('RUN','RESUME_RUN','POINT_ID','PREVIOUS_RUN','PLAN_ONLY','COHORT','FINALIST_PLAN','MODEL'):
        env.pop(key,None)
    env['FINALIST_PLAN' if finalists else 'COHORT']=str(output/'cohort.json')
    return env


def main():
    def terminate(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser(description=__doc__)
    plans=p.add_mutually_exclusive_group(required=True)
    plans.add_argument('--cohort',type=Path)
    plans.add_argument('--finalist-plan',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:gpus=gpu_ordinals(a.gpus)
    except ValueError as e:p.error(str(e))
    if not a.confirm_idle:p.error('--confirm-idle required')
    if a.finalist_plan:
        from dev.prefill_asys.finalists import load as validate_finalists, partition
        cohort=validate_finalists(json.loads(a.finalist_plan.read_text()))
        partition(cohort['points'],0,len(gpus))  # Reject excess GPUs before starting workers.
    else:cohort=validate(json.loads(a.cohort.read_text()))
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=a.resume)
    held=(a.output/'.coordinator.lock').open('a')
    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('this cohort already has a running coordinator') from None
    receipt=a.output/'coordinator.json'
    if a.resume:
        before=json.loads(receipt.read_text())
        if before['gpus']!=gpus or before['cohort']!=cohort['sha256']:
            raise ValueError('resume GPU assignment/cohort changed')
        rev=before['revision']
    else:
        rev=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        atomic_json(a.output/'cohort.json',cohort)
        atomic_json(receipt,dict(gpus=gpus,cohort=cohort['sha256'],revision=rev))
    children=[];streams=[];start=time.monotonic()
    try:
        for index,gpu in enumerate(gpus):
            directory=a.output/('gpu-'+gpu);directory.mkdir(exist_ok=a.resume)
            env=worker_environment(a.output,gpu,index,len(gpus),rev,finalists=bool(a.finalist_plan))
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
        runs=[r for gpu in gpus for r in (a.output/('gpu-'+gpu)).glob('qwen32-prefill-asys.*')
              if (r/'results/execution-plan.json').is_file()]
        review=a.output/('review-'+str(time.time_ns())+'.json.xz')
        command=[sys.executable,str(ROOT/'tools/review_prefill_cohort.py'),'--output',str(review)]
        for run in runs:command+=['--run',str(run)]
        rc=subprocess.run(command).returncode if runs else 1
        result=dict(returncodes=codes,review_rc=rc,review=str(review),production_changed=False,
                    status='SCREEN_COMPLETE' if not any(codes) and rc==0 else 'INCOMPLETE')
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
        held.close()


if __name__=='__main__':raise SystemExit(main())
