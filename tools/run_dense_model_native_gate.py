#!/usr/bin/env python3
"""Fixed production/native prefill A/B for Qwen3.5-35B and Qwen3-32B M5120."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.cohort import MODEL_PATHS, headers
from dev.prefill_asys.finalists import partition
from dev.prefill_asys.model_gate import MODEL_MS, make_plan, validate
from dev.strategy.journal import atomic_json
from dev.strategy.smallm_asys import revision
from tools.run_dense_random_gate import finish
from tools.run_prefill_cohort_multi import gpu_ordinals


def coverage_csv(path, plan):
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('model','tensor','qtype','m','n','k','selection','recipe','point'))
        for p in plan['points']:
            w=p['workload']
            for consumer in p['consumers']:
                writer.writerow((consumer['model'],consumer['tensor'],w['qtype'],w['m'],w['n'],w['k'],
                    p['selection']['coverage'],p['selection']['recipe'],p['id']))


def check_models(plan, paths):
    """Resume reads only headers again; never substitute another model or M."""
    for model in plan['models']:
        files,_=headers(paths[model['name']])
        if files!=model['files']:raise ValueError('resume GGUF header/path differs: '+model['name'])


def detail_csvs(output, summary):
    with (output/'components.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(('models','qtype','m','n','k','arm','phase','kernel','mean_us','quactlize_commit'))
        for p in summary.get('points',[]):
            w=p['workload'];models=','.join(sorted({c['model'] for c in p['consumers']}))
            for arm,components in p.get('components',{}).items():
                for c in components:
                    writer.writerow((models,w['qtype'],w['m'],w['n'],w['k'],arm,c['phase'],c['name'],
                                     c['mean_us'],p['quactlize_commit']))
    with (output/'model-summary.csv').open('w',newline='') as f:
        fields=('model','m','expected','counts','native_dense_matrix_ms','quactlize_dense_matrix_ms','delta_ms','scope')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(summary.get('models',[]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    zoo=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo')
    p.add_argument('--qwen35-model',type=Path,default=zoo/MODEL_PATHS['qwen35-35b-a3b'][0])
    p.add_argument('--qwen32-model',type=Path,default=zoo/MODEL_PATHS['qwen3-32b'][0])
    p.add_argument('--llama-source',type=Path,
                   default=Path(os.environ.get('LLAMA_CI_DIR','/sim/eec/shared/junfu.qx/llama.cpp.ncp')))
    p.add_argument('--llama-build',type=Path,help='existing FP32-accumulator native build; never rebuilt')
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--gpus',default='7');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--output',type=Path,help='new directory; default /workspace/kpack-model-prefill-native.*/results')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();gpus=gpu_ordinals(a.gpus)
    if not 1<=a.jobs<=64 or a.l2_bytes<=0:p.error('jobs must be 1..64 and verified L2 bytes positive')
    if a.resume and not a.output:p.error('--resume requires the original --output')
    if not a.plan_only:
        if not a.confirm_idle:p.error('--confirm-idle required; no concurrent GPU workloads')
        revision(ROOT)
        a.sdk=a.sdk.resolve(strict=True);a.llama_source=a.llama_source.resolve(strict=True)
        a.llama_build=(a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve(strict=True)
        for path in (a.sdk/'bin/hgcc',a.sdk/'asight/bin/asys',a.llama_build/'bin/libggml-cuda.so'):
            if not path.is_file():p.error('missing '+str(path))
    paths={'qwen35-35b-a3b':a.qwen35_model,'qwen3-32b':a.qwen32_model}
    if a.output is None:
        parent=Path('/workspace') if Path('/workspace').is_dir() else Path(tempfile.gettempdir())
        a.output=Path(tempfile.mkdtemp(prefix='kpack-model-prefill-native.',dir=parent))/'results'
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=a.resume)
    path=a.output/'plan.json'
    if a.resume:
        plan=validate(json.loads(path.read_text()));check_models(plan,paths)
    else:
        plan=make_plan(paths,a.output);atomic_json(path,plan);coverage_csv(a.output/'coverage.csv',plan)
    partition(plan['points'],0,len(gpus))
    print('MODEL_PREFILL_NATIVE_PLAN '+json.dumps(dict(plan['counts'],models=MODEL_MS,
        excluded=len(plan['excluded']),output=str(a.output),config_sweep=False,
        precision=plan['precision'],production_changed=False)),flush=True)
    for point in plan['points']:
        print('MODEL_PREFILL_NATIVE_POINT '+json.dumps(dict(workload=point['workload'],
              selection=point['selection'],models=sorted({c['model'] for c in point['consumers']}))),flush=True)
    if a.plan_only:return 0
    command=[sys.executable,'-u',str(ROOT/'tools/run_prefill_cohort_multi.py'),
        '--finalist-plan',str(path),'--output',str(a.output/'run'),'--gpus',','.join(gpus),
        '--jobs',str(a.jobs),'--configs-per-batch','1','--confirm-idle']
    if a.resume:command.append('--resume')
    env=dict(os.environ,LLAMA_CI_DIR=str(a.llama_source),LLAMA_CI_BUILD_DIR=str(a.llama_build),
             PPU_SDK=str(a.sdk),L2_BYTES=str(a.l2_bytes))
    binding=dict(llama_source=str(a.llama_source),llama_build=str(a.llama_build),sdk=str(a.sdk),
        gpus=gpus,jobs=a.jobs,l2_bytes=a.l2_bytes,precision=plan['precision'],
        source=revision(ROOT),installed_quactlize_library_used=False,
        candidate='FROZEN_REPO_PRODUCTION_RECIPE_AND_STAGE_IMPLEMENTATIONS')
    binding_path=a.output/'binding.json'
    if a.resume:
        if json.loads(binding_path.read_text())!=binding:raise ValueError('resume source/build/device binding changed')
    else:atomic_json(binding_path,binding)
    atomic_json(a.output/'command.json',command)
    rc=1
    try:
        rc=subprocess.run(command,env=env).returncode
    finally:
        result=finish(a.output,plan)
        detail_csvs(a.output,json.loads((a.output/'summary.json').read_text()))
        print('Asys='+str(a.output/'run/gpu-*/qwen32-prefill-asys.*/results/batches/*/attempt-*/trace.asysrep'),flush=True)
        print('Resume with the same command plus --output '+str(a.output)+' --resume',flush=True)
    return result if rc==0 else 1


if __name__=='__main__':raise SystemExit(main())
