#!/usr/bin/env python3
"""Q8 TP-local SF refinement + isolated dequant diagnosis, no model loading.

Consumes the previous compact whole-call review. Independent existing runners
retain raw evidence; this orchestrator never changes production selection.
"""
import argparse
import fcntl
import json
import lzma
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.q8_refine import make_plan, dequant_diagnostics, validate
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest
from tools.run_prefill_cohort_multi import gpu_ordinals, worker_limits
from tools.run_tp_prefill_supplement import run_coordinator


def read_review(path):
    opener = lzma.open if path.suffix=='.xz' else open
    with opener(path,'rt') as stream:
        return json.load(stream)


def helper_environment(env, output, diagnostics, gpu, revision, profile):
    result = dict(env, GPU=gpu, OUTPUT_ROOT=str(output), SOURCE_REV=revision,
        Q8='1',Q8_NATIVE='1',
        Q8_SHAPES=diagnostics['shapes'], PROFILE_ACU=str(int(profile)),IDLE_CONFIRMED='1')
    for key in ('RUN','RESUME_RUN','POINTS','PROFILE_POINTS','NATIVE_LIBRARY',
                'Q8_EXCHANGE','Q8_FAST_TAIL','Q8_HALF2','Q8_PAIRED'):
        result.pop(key,None)
    result['PROFILE_POINTS'] = ','.join(map(str,diagnostics['profile_points']))
    return result


def package(output):
    """Only compact reviews/diagnostics, never source, cache, SO or acurep."""
    paths = [p for name in ('request.json','plan.json','dequant.json','status.json')
             if (p:=output/name).is_file()]
    paths += sorted((output/'timing').glob('review-*.json.xz'))
    paths += sorted((output/'timing').glob('summary-*.json'))
    paths += sorted((output/'timing').glob('coordinator.json'))
    paths += sorted((output/'timing').glob('gpu-*/console-*.log'))
    for suffix in ('json','log'):
        for gate in ('io','reducer'):
            paths += sorted((output/'timing').glob(f'gpu-*/qwen32-prefill-asys.*/results/{gate}-gate-*.'+suffix))
    paths += sorted((output/'dequant').glob('kpack-dense-dequant.*/results/review-*.tgz'))
    paths += sorted((output/'dequant').glob('kpack-dense-dequant.*/console-*.log'))
    path = output/f'review-{time.time_ns()}.tgz'
    with tarfile.open(path,'w:gz') as archive:
        for p in paths:
            if p.is_symlink():
                raise ValueError('refusing symlink in compact result package')
            archive.add(p,arcname=str(p.relative_to(output)),recursive=False)
    print('results='+str(path),flush=True)
    return path


def main():
    def interrupted(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True,help='previous TP timing/review-*.json.xz')
    p.add_argument('--phase',choices=('all','sf','dequant','io','helpers','followup'),default='all',
                   help='io/helpers: fixed winners with cast-only or cast+compact-reducer comparisons')
    p.add_argument('--gpus',default=os.environ.get('GPUS','1,2,3,4,5,7'))
    p.add_argument('--jobs',type=int,default=8,help='compile jobs per worker; shell JOBS is ignored')
    p.add_argument('--configs-per-batch',type=int,default=16,help='configs per capture; shell CONFIGS_PER_BATCH is ignored')
    p.add_argument('--llama-source',type=Path,default=Path(os.environ.get('LLAMA_CI_DIR','/sim/eec/shared/junfu.qx/llama.cpp.ncp')))
    p.add_argument('--llama-build',type=Path,default=os.environ.get('LLAMA_CI_BUILD_DIR'))
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--output',type=Path)
    p.add_argument('--output-root',type=Path,default=Path('/workspace'))
    p.add_argument('--run',action='store_true',help='default only validates and writes the bounded plan')
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--profile-acu',action='store_true',help='profile largest helper gap and largest slow matrix after timing; one direct launch, no cold loops')
    a = p.parse_args()
    if a.profile_acu and a.phase in ('io','helpers','followup'):
        p.error('--profile-acu covers dequant only; IO/reducer phases collect Asys component timings')
    try:
        gpus = gpu_ordinals(a.gpus)
        limits=worker_limits(a.jobs,a.configs_per_batch)
    except ValueError as e:p.error(str(e))
    if a.run and not a.confirm_idle:p.error('--confirm-idle required for device work')
    if a.resume and not a.output:p.error('--resume requires --output')
    review = read_review(a.review)
    if a.phase == 'followup':
        from dev.prefill_asys.helper_followup import make_plan as make_followup, validate as validate_followup
        plan=make_followup(review)
        diagnostics=dict(points=[],scope='GRAIN_AND_ORDER_NO_MAINLOOP_SEARCH')
        validate_plan=validate_followup
    elif a.phase in ('io','helpers'):
        from dev.prefill_asys.io_refine import make_plan as make_io_plan, validate as validate_io
        plan=make_io_plan(review,compact=a.phase=='helpers')
        diagnostics=dict(points=[],scope='FIXED_MAINLOOP_HELPERS_NO_DEQUANT_SEARCH')
        validate_plan=validate_io
    else:
        plan=make_plan(review)
        diagnostics=dequant_diagnostics(review,plan)
        validate_plan=validate
    a.llama_source = a.llama_source.resolve()
    a.llama_build = (a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve()
    a.sdk = a.sdk.resolve()
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    request = dict(review=review['review_sha256'],plan=plan['sha256'],dequant=digest(diagnostics),
        revision=revision,gpus=gpus,sdk=str(a.sdk),llama_source=str(a.llama_source),
        llama_build=str(a.llama_build),profile_acu=a.profile_acu,worker_limits=limits)
    if a.output:
        a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=a.resume)
    else:
        a.output=Path(tempfile.mkdtemp(prefix='kpack-q8-tp-refine.',dir=a.output_root))
    held=(a.output/'.refine.lock').open('a')
    try:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('this refinement already has a running coordinator') from None
    if a.resume:
        if json.loads((a.output/'request.json').read_text())!=request:
            raise ValueError('resume source, review, native path or GPU assignment differs')
        validate_plan(json.loads((a.output/'plan.json').read_text()))
        if json.loads((a.output/'dequant.json').read_text())!=diagnostics:
            raise ValueError('resume helper cohort differs')
    else:
        for name,value in (('request',request),('plan',plan),('dequant',diagnostics)):
            atomic_json(a.output/(name+'.json'),value)
    print('Q8_TP_REFINE_PLAN '+json.dumps(dict(output=str(a.output),**plan['counts'],
        helper_families=len(diagnostics['points']),max_cases=max(len(x['cases']) for x in plan['points']),
        scope=plan['scope'],worker_limits=limits,production_changed=False,performance_admitted=False)),flush=True)
    if not a.run:
        print('Plan only; add --run --resume --output '+str(a.output)+' with the same arguments.',flush=True)
        return 0
    env=dict(os.environ,PPU_SDK=str(a.sdk),LLAMA_CI_DIR=str(a.llama_source),
             LLAMA_CI_BUILD_DIR=str(a.llama_build),L2_BYTES=os.environ.get('L2_BYTES','67108864'))
    for key in ('RUN','RESUME_RUN','PREVIOUS_RUN','FINALIST_PLAN','COHORT','MODEL','PLAN_ONLY','POINT_ID'):
        env.pop(key,None)
    status=dict(status='INCOMPLETE',phases={},requested_phases=['sf','dequant'] if a.phase=='all' else [a.phase],
                production_changed=False,performance_admitted=False)
    try:
        if a.phase in ('all','sf','io','helpers','followup'):
            workers=gpus[:min(len(gpus),plan['counts']['families'])]
            command=[sys.executable,'-u',str(ROOT/'tools/run_prefill_cohort_multi.py'),
                '--finalist-plan',str(a.output/'plan.json'),'--gpus',','.join(workers),
                '--output',str(a.output/'timing'),'--jobs',str(limits['jobs']),
                '--configs-per-batch',str(limits['configs_per_batch']),'--confirm-idle']
            if (a.output/'timing').exists():command.append('--resume')
            status['phases'][a.phase if a.phase in ('io','helpers','followup') else 'sf']=run_coordinator(command,env)
            atomic_json(a.output/'status.json',status)
        if a.phase in ('all','dequant'):
            helper=a.output/'dequant';helper.mkdir(exist_ok=a.resume)
            helper_env=helper_environment(env,helper,diagnostics,gpus[0],revision,a.profile_acu)
            runs=[r for r in helper.glob('kpack-dense-dequant.*') if r.is_dir()]
            if runs:
                if not a.resume or len(runs)!=1:raise ValueError('ambiguous retained helper run')
                helper_env['RESUME_RUN']=str(runs[0])
            # SF workers have all exited before this starts, including failure
            # paths. Never profile alongside another timed job on this GPU.
            status['phases']['dequant']=run_coordinator(
                ['bash',str(ROOT/'tools/run_dense_dequant_box.sh')],helper_env)
        status['status']='MEASURED_NOT_ADMITTED' if not any(status['phases'].values()) else 'INCOMPLETE'
        phases=(a.phase,) if a.phase in ('io','helpers','followup') else ('sf','dequant')
        status['not_measured_in_this_invocation']=[stage for stage in phases if stage not in status['phases']]
        print('Q8_TP_REFINE_DONE '+json.dumps(status),flush=True)
        return int(status['status']=='INCOMPLETE')
    finally:
        atomic_json(a.output/'status.json',status)
        package(a.output)
        print('Asys: '+str(a.output/'timing/gpu-*/qwen32-prefill-asys.*/results/batches/*/attempt-*/trace.asysrep'),flush=True)
        if a.phase in ('all','dequant'):
            print('ACU: '+str(a.output/'dequant/kpack-dense-dequant.*/results/profile-*.acurep'),flush=True)


if __name__=='__main__':raise SystemExit(main())
