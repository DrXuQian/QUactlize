#!/usr/bin/env python3
"""Continue a model run stopped by the obsolete dense/BF16 evidence check."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from tools.run_kpack_batched_bench import save
from tools.run_kpack_model_validation import run
from tools.verify_kpack_dispatch import verify


def inputs(previous, llama, sdk):
    results = previous / 'results'
    stopped = (results / 'runner-status.txt').read_text()
    if not re.search(r'runner_rc=[1-9]\d* stage=model-numerical\b', stopped):
        raise ValueError('this continuation requires a run stopped at model-numerical')
    status = json.loads((results / 'numerical/status.json').read_text())
    failures = [r for r in status if r['status'] != 'PASS']
    if not failures or any(r.get('error') != 'explicit BF16 request fell back to FP16 compute' for r in failures):
        raise ValueError('original failure is not the repaired compute-scope check')
    if any((results / name).exists() for name in ('benchmark', 'trace', 'acu')):
        raise ValueError('later-stage results already exist; preserve them for a separate continuation')
    pin = json.loads((ROOT / 'tools/kpack_q4_model_artifact.json').read_text())
    bundle = previous.parent / ('quactlize-model-artifact-' + pin['commit'][:10]) / pin['path']
    if sha(bundle / 'manifest.json') != pin['manifest_sha256'] or sha(bundle / 'manifest.json') != sha(results / 'bundle-manifest.json'):
        raise ValueError('original and selected runtime packages differ')
    verify(bundle, sdk=sdk)
    ci = json.loads((results / 'caller-ci-build.json').read_text())
    required = {'bin/' + name for name in ('llama-server', 'llama-batched-bench', 'llama-perplexity',
        'libggml-cuda.so', 'libncp_fa.so', 'libncp_moe.so')}
    if ci.get('entry') != '.aoneci/scripts/build.sh' or not required <= ci.get('files', {}).keys():
        raise ValueError('original caller build receipt is incomplete')
    build = Path(ci['build']).resolve(strict=True)
    for name, expected in ci['files'].items():
        file = (build / name).resolve(strict=True)
        if not file.is_relative_to(build) or sha(file) != expected:
            raise ValueError('original caller build changed: ' + name)
    checker = (llama / 'tests/quactlize_native.py').read_text()
    if 'compute scope mismatch:' not in checker or 'or int(p["rows"]) <= 8' in checker:
        raise ValueError('pull the grouped-only BF16 checker in the supplied llama checkout')
    gates = ('production-q8.json', 'bf16-metadata/summary.json', 'bf16/summary.json',
        'bf16-selected-q4/summary.json', 'bf16-matched-prefill/summary.json', 'mixed-chain/summary.json')
    for name in gates:
        if json.loads((results / name).read_text()).get('status') != 'PASS':
            raise ValueError('earlier component gate did not pass: ' + name)
    return bundle, build, {name: sha(results / name) for name in gates}


def continuation(args, output):
    previous, llama, sdk = args.previous.resolve(strict=True), args.llama.resolve(strict=True), args.sdk.resolve(strict=True)
    bundle, build, gates = inputs(previous, llama, sdk)
    result = output / 'results'
    if any(p.is_symlink() for p in (previous / 'results').rglob('*')):
        raise ValueError('original results contain links; no recursive copy started')
    shutil.copytree(previous / 'results', result / 'prior', ignore=shutil.ignore_patterns('*.asysrep', '*.sqlite*', '*.tgz'))
    save(result / 'continuation.json', dict(previous=str(previous), caller_build=str(build),
        runtime_manifest_sha256=sha(bundle / 'manifest.json'), prior_gates=gates,
        validator_sha256=sha(llama / 'tests/quactlize_native.py'), compile='NONE',
        original_results='UNMODIFIED', model_admission='PENDING_RECHECK'))
    env = dict(os.environ)
    for name in ('DG_LIBRARY_ROOT', 'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB',
                 'QUACTLIZE_KPACK_PREFILL_POLICY', 'QUACTLIZE_KPACK_GEMV_POLICY',
                 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION'):
        env.pop(name, None)
    base = previous.parent
    cache = Path(env.get('CACHE_DIR', str(base / 'kpack-model-cache'))).resolve(strict=True)
    jit = Path(env.get('JIT_CACHE', str(base / 'kpack-model-jit-cache'))).resolve(strict=True)
    env.update(PPU_SDK=str(sdk), CUDA_HOME=str(sdk / 'CUDA_SDK'), CUDA_VISIBLE_DEVICES=args.device,
        LD_LIBRARY_PATH=str(build / 'bin') + ':' + env.get('LD_LIBRARY_PATH', ''),
        DG_JIT_CACHE_DIR=str(previous / 'ci/ncp-jit-cache'),
        QUACTLIZE_KPACK_COMPUTE='bf16', QUACTLIZE_KPACK_ROUTE='auto', QUACTLIZE_KPACK_PAIR_WEIGHTS='1',
        QUACTLIZE_PPU_PACK_LIBRARY=str(bundle / 'pack/libquactlize_ppu_pack.so'),
        QUACTLIZE_KPACK_EXECUTION=str(bundle), QUACTLIZE_KPACK_JIT_HELPER=str(ROOT / 'tools/kpack_jit.py'),
        QUACTLIZE_KPACK_JIT_PYTHON=sys.executable, QUACTLIZE_KPACK_JIT_CACHE=str(jit),
        QUACTLIZE_KPACK_DEEPGEMM_HELPER=str(bundle / 'kpack_deepgemm_prewarm.py'))
    asys, acu = sdk / 'asight/bin/asys', sdk / 'asight/bin/acu'
    if not all(p.is_file() and os.access(p, os.X_OK) for p in (asys, acu)):
        raise ValueError('Asys/ACU executable is missing')
    model_plan = previous / 'results/model-plan.json'
    common = ['--llama', llama, '--build', build, '--bundle', bundle, '--plan', model_plan,
        '--cache', cache, '--jit-cache', jit, '--logits', previous / 'logits', '--corpus', args.corpus,
        '--asys', asys, '--inspector', sdk / 'bin/hgobjdump']
    validation = [sys.executable, '-u', ROOT / 'tools/run_kpack_model_validation.py']
    phases = [
        ('numerical', validation + common + ['--phase', 'numerical', '--output', result / 'numerical',
            '--reuse-from', previous / 'results/numerical']),
        ('benchmark', [sys.executable, '-u', ROOT / 'tools/run_kpack_batched_bench.py',
            '--binary', build / 'bin/llama-batched-bench', '--llama-dir', llama, '--bundle', bundle,
            '--jit-cache', jit, '--cache-root', cache, '--output-root', base, '--output', result / 'benchmark',
            '--plan', model_plan, '--device', args.device, '--order', 'abba', '--require-selected', '--repeats', '2']),
        ('trace', validation + common + ['--phase', 'trace', '--output', result / 'trace']),
        ('acu', [sys.executable, '-u', ROOT / 'tools/profile_kpack_model_decode.py',
            '--sdk', sdk, '--bundle', bundle, '--llama', llama, '--trace', result / 'trace',
            '--acu', acu, '--output', result / 'acu']),
    ]
    for phase, command in phases:
        save(result / 'status.json', dict(status='RUNNING', phase=phase))
        print(f'KPACK_MODEL_CONTINUE phase={phase} compile=NONE log={result / (phase + ".log")}', flush=True)
        run(command, result / (phase + '.log'), env)
        save(result / 'status.json', dict(status='PASS', phase=phase))
    print(f'KPACK_MODEL_CONTINUE COMPLETE Asys={result / "trace"} ACU={result / "acu"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'llama', 'sdk', 'corpus'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    if not re.fullmatch(r'\d+', a.device):
        p.error('one physical device ordinal is required')
    previous = a.previous.resolve(strict=True)
    output = Path(tempfile.mkdtemp(prefix='kpack-q4-resume.', dir=previous.parent))
    (output / 'results').mkdir()
    print(f'KPACK_MODEL_CONTINUE run={output} previous={previous}', flush=True)
    rc = 1
    try:
        continuation(a, output)
        rc = 0
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        status_file = output / 'results/status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else dict(phase='precheck')
        save(status_file, dict(status, status='FAIL', error=str(error)))
        print('KPACK_MODEL_CONTINUE FAIL: ' + str(error), flush=True)
    finally:
        with tarfile.open(str(output) + '.results.tgz', 'w:gz') as archive:
            def include(info):
                return None if info.name.endswith(('.asysrep', '.tgz')) or '.sqlite' in Path(info.name).name else info
            archive.add(output / 'results', arcname='results', filter=include)
        print(f'results={output}.results.tgz\nrunner_rc={rc}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
