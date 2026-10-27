#!/usr/bin/env python3
"""Twenty seeded production-bucket/native spot checks using the existing Asys runner."""
import argparse
import csv
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.random_gate import make_plan, validate, compare, SCOPE
from dev.prefill_asys.finalists import partition
from dev.strategy.journal import atomic_json
from dev.strategy.smallm_asys import revision
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_prefill_cohort_multi import gpu_ordinals


def read(path):
    return json.loads(path.read_text())


def csv_result(path, summary):
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('qtype','m','n','k','status','native_us','quactlize_us','delta_us','delta_pct',
                         'native_main_us','native_helper_us','native_reducer_us',
                         'quactlize_main_us','quactlize_helper_us','quactlize_reducer_us','recipe','quactlize_commit'))
        for row in summary.get('points',[]):
            parts=[]
            for arm in ('native','quactlize'):
                cs=row.get('components',{}).get(arm,[])
                parts.extend(sum(c['mean_us'] for c in cs if (
                    c['phase']==phase if phase!='helper' else c['phase'] not in ('main','reducer')))
                    if cs else '' for phase in ('main','helper','reducer'))
            writer.writerow((*[row['workload'][k] for k in ('qtype','m','n','k')],row['status'],
                *[row.get(k,'') for k in ('native_us','quactlize_us','delta_us','delta_pct')],
                *parts,row['selection']['recipe'],row.get('quactlize_commit','')))


def finish(output, plan):
    paths=sorted((output/'run').glob('review-*.json.xz'))
    review=None;error=None
    try:
        if not paths:raise ValueError('no component review produced; inspect worker logs')
        with lzma.open(paths[-1],'rt') as f:review=json.load(f)
        if plan.get('schema')=='quactlize.dense-model-native.v1':
            from dev.prefill_asys.model_gate import compare as compare_model
            summary=compare_model(plan,review)
        else:
            summary=compare(plan,review)
    except Exception as exc:
        error=repr(exc)
        summary=dict(status='INCOMPLETE',error=error,expected=len(plan['points']),scope=plan.get('scope',SCOPE),
                     production_changed=False,model_tpot_admitted=False)
    coordinator=sorted((output/'run').glob('summary-*.json'))
    state=read(coordinator[-1]) if coordinator else None
    if state is None or state.get('status')!='SCREEN_COMPLETE':
        summary['status']='INCOMPLETE'
    summary['coordinator']=state
    atomic_json(output/'summary.json',summary);csv_result(output/'comparison.csv',summary)
    model=plan.get('schema')=='quactlize.dense-model-native.v1'
    packet=dict(schema='quactlize.dense-model-native-review.v1' if model else 'quactlize.dense-random-native-review.v1',
                plan=plan,summary=summary,evidence=review,
                raw_reports='RETAINED_ON_BOX_NOT_EMBEDDED',review_sha256=sha(paths[-1]) if paths else None)
    packet['sha256']=digest(packet)
    path=output/('review-'+str(time.time_ns())+'.json.xz')
    with lzma.open(path,'xt') as f:json.dump(packet,f,separators=(',',':'),sort_keys=True)
    print(('MODEL_PREFILL_NATIVE_DONE ' if model else 'RANDOM_DENSE_DONE ')+
          json.dumps({k:v for k,v in summary.items() if k not in ('points','coordinator')}),flush=True)
    print('results='+str(path),flush=True)
    return 0 if summary['status']=='PASS' else 2 if summary['status']=='REGRESSION' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='new directory; default /workspace/kpack-random-dense.*/results')
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--gpus',default='0,1,2,3');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();gpus=gpu_ordinals(a.gpus)
    if not 1<=a.jobs<=64:p.error('--jobs must be in 1..64')
    if a.resume and not a.output:p.error('--resume requires the original --output')
    if not a.plan_only:
        if not a.confirm_idle:p.error('--confirm-idle required')
        revision(ROOT,require_clean=False)  # Workers execute HEAD archives, not working-tree edits.
    if a.output is None:
        parent=Path('/workspace') if Path('/workspace').is_dir() else Path(tempfile.gettempdir())
        a.output=Path(tempfile.mkdtemp(prefix='kpack-random-dense.',dir=parent))/'results'
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=a.resume)
    path=a.output/'plan.json'
    if a.resume:
        plan=validate(read(path))
        if plan['seed']!=a.seed:p.error('resume must keep the original seed')
    else:
        plan=make_plan(a.output,a.seed);atomic_json(path,plan)
    partition(plan['points'],0,len(gpus))
    print('RANDOM_DENSE_PLAN '+json.dumps(dict(plan['counts'],seed=plan['seed'],qtypes=plan['qtypes'],
        m_bands=plan['m_bands'],output=str(a.output),config_sweep=False,production_changed=False)),flush=True)
    for point in plan['points']:
        print('RANDOM_DENSE_POINT '+json.dumps(dict(workload=point['workload'],selection=point['selection'])),flush=True)
    if a.plan_only:return 0
    command=[sys.executable,'-u',str(ROOT/'tools/run_prefill_cohort_multi.py'),'--finalist-plan',str(path),
        '--output',str(a.output/'run'),'--gpus',','.join(gpus),'--jobs',str(a.jobs),
        '--configs-per-batch','1','--confirm-idle']
    if a.resume:command.append('--resume')
    atomic_json(a.output/'command.json',command)
    completed=subprocess.run(command)
    result=finish(a.output,plan)
    print('Asys='+str(a.output/'run/gpu-*/qwen32-prefill-asys.*/results/batches/*/attempt-*/trace.asysrep'),flush=True)
    print('Resume: python3 tools/run_dense_random_gate.py --output '+str(a.output)+
          ' --gpus '+','.join(gpus)+' --seed '+str(a.seed)+' --resume --confirm-idle',flush=True)
    return result if completed.returncode==0 else 1


if __name__=='__main__':raise SystemExit(main())
