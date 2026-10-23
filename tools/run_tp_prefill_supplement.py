#!/usr/bin/env python3
"""Qwen3.5-122B TP2 + DeepSeek-V4-Flash TP4: observed local dense shapes.

Default: native short inventory passes and a bounded plan. --run also starts
the existing multi-GPU Asys component comparison. No caller/model/policy edits.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_asys.cohort import collect
from dev.prefill_asys.supplement import make_plan, MS
from dev.strategy.journal import atomic_json
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_prefill_cohort_multi import gpu_ordinals

MODELS = (
    ('qwen35-122b-tp2', 2, 'Qwen3.5-122B-A10B-GGUF-Q4_K_M',
     'Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf'),
    ('deepseek4-flash-tp4', 4, 'DeepSeek-V4-Flash-0731-UD-Q4_K_XL',
     'DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf'),
)


def model_path(root, directory, filename, override):
    if override:
        path = override.resolve(strict=True)
        if not path.is_file():
            raise ValueError('model override must be an exact first GGUF shard')
        return path
    found = set()
    for base in (root/directory, root/'bench_model_zoo'/directory):
        if base.is_dir():
            found.update(p.resolve() for p in base.rglob(filename) if p.is_file())
    if len(found) != 1:
        raise ValueError(f'expected one {filename} under {root}; found {sorted(map(str,found))}; '
                         'use --qwen-model/--deepseek-model with an exact first-shard file')
    return found.pop()


def probe_command(binary, model, world):
    # One small native benchmark row, with the binary's default warmup.
    # --no-warmup is not exposed by every batched-bench build, and inventory
    # duration is never admitted as performance. TP extents are observed.
    return [str(binary), '-m', str(model), '-npp', '64', '-ntg', '1', '-npl', '1',
            '-b', '64', '-ub', '64', '-c', '128', '-ngl', 'all', '-sm', 'tensor',
            '-ts', ','.join(['1']*world), '-fa', '1', '--fit', 'off', '--mmap',
            '--output-format', 'jsonl', '--log-colors', 'off', '--verbosity', '5']


def probe(a, model, name, world, directory, gpus):
    from tools.run_dense_model_ab import environment, execute
    binary = (a.llama_build/'bin/llama-batched-bench').resolve(strict=True)
    args = SimpleNamespace(sdk=a.sdk, device=','.join(gpus[:world]), jit_cache=None)
    env = environment(args, a.llama_build, 'reference')
    env.update(GGML_META_DEBUG='1', GGML_QUACTLIZE_DENSE='0')
    directory.mkdir()
    help_result = subprocess.run([str(binary), '--help'], env=env, capture_output=True, text=True,
                                 errors='replace', timeout=30, check=True)
    for flag in ('--split-mode', '--tensor-split', '--verbosity'):
        if flag not in help_result.stdout+help_result.stderr:
            raise ValueError(f'{binary} lacks required TP inventory option {flag}; inspect this binary\'s --help')
    atomic_json(directory/'identity.json', dict(binary=str(binary), sha256=sha(binary),
        model=str(model), world=world, devices=gpus[:world], GGML_META_DEBUG=1,
        warmup='NATIVE_DEFAULT_NOT_TIMED', scope='NATIVE_PLACEMENT_INVENTORY_NOT_PERFORMANCE'))
    execute(probe_command(binary, model, world), env, directory, name+'-tp-inventory')
    return directory/'run.log'


def run_coordinator(command, env):
    process = subprocess.Popen(command,env=env,start_new_session=True)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            # Give the coordinator time to stop only its own worker groups.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def main():
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, default=Path('/sim/eec/shared/AI_workspace/llm-models'))
    p.add_argument('--qwen-model', type=Path)
    p.add_argument('--deepseek-model', type=Path)
    p.add_argument('--qwen-log', type=Path, help='existing actual TP2 local-shape or GGML_META_DEBUG log')
    p.add_argument('--deepseek-log', type=Path, help='existing actual TP4 local-shape or GGML_META_DEBUG log')
    p.add_argument('--names', nargs='+', choices=[m[0] for m in MODELS], default=[m[0] for m in MODELS])
    p.add_argument('--m', nargs='+', type=int, default=list(MS))
    p.add_argument('--gpus', default=os.environ.get('GPUS', '1,2,3,4,5,6,7'))
    p.add_argument('--llama-source', type=Path,
                   default=Path(os.environ.get('LLAMA_CI_DIR','/sim/eec/shared/junfu.qx/llama.cpp.ncp')))
    p.add_argument('--llama-build', type=Path, default=os.environ.get('LLAMA_CI_BUILD_DIR'))
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--output-root', type=Path, default=Path('/workspace'))
    p.add_argument('--output', type=Path, help='new retained directory; --resume reuses this exact directory')
    p.add_argument('--max-cases', type=int, default=32)
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--run', action='store_true', help='after inventory, run all local-shape comparisons')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    gpus = gpu_ordinals(a.gpus)
    if len(set(a.names)) != len(a.names):
        p.error('duplicate model names')
    if len(set(a.m)) != len(a.m) or not set(a.m) <= set(MS):
        p.error('M must be a distinct subset of 128,256,512,1024,2048')
    if not a.confirm_idle and (a.run or any(not getattr(a, 'qwen_log' if world==2 else 'deepseek_log')
                                          for name,world,_,_ in MODELS if name in a.names)):
        p.error('--confirm-idle required before native inventory or timing')
    if a.resume and not a.output:
        p.error('--resume requires the retained --output directory')
    a.sdk = a.sdk.resolve()
    a.llama_source = a.llama_source.resolve()
    a.llama_build = (a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve()
    policy = json.loads(EVIDENCE.read_text())
    helper_sha = digest(json.loads((ROOT/'policies/dense_full_helpers_20261023.json').read_text()))
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    requested = []
    for name, world, directory, filename in MODELS:
        if name not in a.names:
            continue
        prefix = 'qwen' if world==2 else 'deepseek'
        log = getattr(a, prefix+'_log')
        if not log and len(gpus) < world:
            p.error(f'{name} inventory needs {world} assigned GPUs or an existing --{prefix}-log')
        path = model_path(a.model_root,directory,filename,getattr(a,prefix+'_model'))
        requested.append(dict(name=name,tp=world,path=str(path),
                              tp_logs=[str(log.resolve(strict=True))] if log else []))
    request = dict(models=requested,ms=a.m,gpus=gpus,revision=revision,policy=policy['sha256'],
                   helper_policy=helper_sha,max_cases=a.max_cases,sdk=str(a.sdk),
                   llama_source=str(a.llama_source),llama_build=str(a.llama_build))
    if a.output:
        a.output = a.output.resolve()
        a.output.mkdir(parents=True,exist_ok=a.resume)
    else:
        a.output = Path(tempfile.mkdtemp(prefix='kpack-tp-prefill.',dir=a.output_root))
    receipt = a.output/'request.json'
    print('TP_PREFILL_START output='+str(a.output), flush=True)
    if a.resume:
        if json.loads(receipt.read_text()) != request:
            raise ValueError('resume source/models/GPU assignment changed; retain the old run')
    else:
        atomic_json(receipt, request)
    inventory_path, plan_path = a.output/'cohort.json', a.output/'supplement.json'
    if not plan_path.exists():
        models = json.loads(json.dumps(requested))
        for model in models:
            if model['tp_logs']:
                continue
            directory = a.output/(model['name']+'-inventory')
            if a.resume and (directory/'process.json').exists() and json.loads((directory/'process.json').read_text())['rc']==0:
                log = directory/'run.log'
            else:
                if directory.exists():
                    # Never overwrite a failed inventory attempt.
                    import time
                    directory = a.output/(model['name']+'-inventory-'+str(time.time_ns()))
                log = probe(a,Path(model['path']),model['name'],model['tp'],directory,gpus)
            model['tp_logs'] = [str(log)]
        cohort = collect(models,a.m)
        atomic_json(inventory_path,cohort)
        if cohort['status'] != 'READY':
            print('TP_PREFILL_BLOCKED '+json.dumps(cohort['blocked'][:12]), flush=True)
            raise ValueError('TP local inventory incomplete; inspect cohort.json; no guessed N/K admitted')
        plan = make_plan(cohort,policy,helper_sha,max_cases=a.max_cases)
        atomic_json(plan_path,plan)
    else:
        from dev.prefill_asys.supplement import validate
        plan = validate(json.loads(plan_path.read_text()))
    counts = dict(plan['counts'], table_coverage=dict(Counter(p['table_coverage'] for p in plan['points'])),
        excluded=dict(Counter(r['reason'].split(':')[0] for r in plan['exclusions'])),
        production_changed=False,scope=plan['scope'],duration_guaranteed=False)
    print('TP_PREFILL_PLAN '+json.dumps(counts),flush=True)
    if not a.run:
        print('Plan ready. Use the same arguments with --run --resume --output '+str(a.output),flush=True)
        return 0
    # M values of a weight family stay on one GPU. Dedup can leave fewer than
    # seven families; do not start empty shards merely to occupy all cards.
    workers = gpus[:min(len(gpus),plan['counts']['families'])]
    env = dict(os.environ,PPU_SDK=str(a.sdk),LLAMA_CI_DIR=str(a.llama_source),
               LLAMA_CI_BUILD_DIR=str(a.llama_build),L2_BYTES=os.environ.get('L2_BYTES','67108864'))
    command = [sys.executable,'-u',str(ROOT/'tools/run_prefill_cohort_multi.py'),
        '--finalist-plan',str(plan_path),'--gpus',','.join(workers),'--output',str(a.output/'timing'),'--confirm-idle']
    if (a.output/'timing').exists():
        command.append('--resume')
    rc = run_coordinator(command,env)
    print(f'TP_PREFILL_DONE rc={rc} review={a.output}/timing/review-*.json.xz '
          'asys=timing/gpu-*/qwen32-prefill-asys.*/results/batches/*/attempt-*/trace.asysrep',flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
