#!/usr/bin/env python3
"""Replay one failed model command, then localize its first nonfinite tensor."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def option(argv, key):
    if argv.count(key) != 1 or argv.index(key) + 1 >= len(argv):
        raise ValueError(f'expected exactly one {key} in the original command')
    return argv[argv.index(key) + 1]


def command(argv, reference=False):
    argv = list(argv)
    for key, expected in (('-b', '1'), ('-ub', '1'), ('-c', '256'), ('--chunks', '2')):
        if option(argv, key) != expected:
            raise ValueError(f'original {key} differs from the failing B1/256/two-chunk case')
    if '--no-warmup' not in argv or '--kl-divergence' not in argv or '--save-all-logits' in argv:
        raise ValueError('expected the read-only KL command with no warmup')
    if not option(argv, '-ot').endswith('=CUDA0_KPACK'):
        raise ValueError('original command does not select the K-pack buffer')
    if reference:
        index = argv.index('-ot') + 1
        argv[index] = argv[index].removesuffix('_KPACK')
        index = argv.index('--kpack-cache')
        del argv[index:index+2]
    return argv


def result(text, rc, mode):
    if f'LLAMA_NUMERICAL_DEBUG mode={mode} callback={int(mode == "tensors")} timing_valid=0' not in text:
        raise ValueError('binary did not enter the requested diagnostic mode')
    stops = re.findall(r'^LLAMA_NUMERICAL_STOP (.+)$', text, re.M)
    complete = re.findall(r'^LLAMA_NUMERICAL_COMPLETE (.+)$', text, re.M)
    if rc == 86 and len(stops) == 1 and not complete:
        reason = 'NONFINITE_TENSOR' if mode == 'tensors' else 'NONFINITE_LOGITS'
        if f'reason={reason}' not in stops[0]:
            raise ValueError('diagnostic stop reason differs')
        return dict(verdict='NONFINITE_FOUND', stop=stops[0],
                    tensors=re.findall(r'^LLAMA_NUMERICAL_(?:TENSOR|SOURCE|LOGITS) (.+)$', text, re.M))
    if rc == 0 and len(complete) == 1 and not stops:
        # The original command requests 128 output rows per chunk, including
        # the final row that the 127-token KL scorer does not consume.
        if not re.search(r'\blogits=256\b', complete[0]):
            raise ValueError('diagnostic did not inspect both complete chunks')
        if mode == 'tensors' and not re.search(r'\bnodes=[1-9][0-9]*\b', complete[0]):
            raise ValueError('tensor callback inspected no nodes')
        return dict(verdict='NO_NONFINITE_OBSERVED', completion=complete[0])
    raise ValueError(f'incomplete diagnostic rc={rc}; inspect the full log')


def run(argv, env, log):
    start = time.monotonic()
    save(log.with_suffix('.command.json'), dict(argv=argv))
    with log.open('x') as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                try:
                    rc = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'KPACK_FIRST_NONFINITE_WAIT arm={log.stem} minutes={(time.monotonic()-start)/60:.1f} log={log}', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    save(log.with_suffix('.process.json'), dict(rc=rc, seconds=time.monotonic()-start, timing_valid=False))
    return log.read_text(errors='replace'), rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'llama', 'sdk', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = args.output / 'results'
    results.mkdir()
    previous = args.previous.resolve(strict=True)
    receipt_path = previous / 'results/caller-ci-build.json'
    receipt = json.loads(receipt_path.read_text())
    original_path = previous / 'results/numerical/qwen3-32b-q4km/b1-kpack-reference.command.json'
    original = command(json.loads(original_path.read_text())['argv'])
    build = Path(os.environ.get('LLAMA_CI_BUILD_DIR', receipt['build'])).resolve(strict=True)
    llama = args.llama.resolve(strict=True)
    if 'LLAMA_NUMERICAL_SELF_TEST PASS' not in (llama / 'tools/perplexity/perplexity.cpp').read_text():
        raise ValueError('update the llama dev/quactlize-v0.3.0 branch first')
    for key in ('-m', '-f', '--kl-divergence-base'):
        if not Path(option(original, key)).is_file():
            raise ValueError(f'original input {key} is missing; keep the previous run directory')
    ncp = Path(os.environ.get('NCP_CI_DIR', receipt['ncp_directory'])).resolve(strict=True)
    pin = json.loads((ROOT / 'tools/kpack_q4_model_artifact.json').read_text())
    bundle = Path(os.environ.get('EXECUTION_BUNDLE', str(previous.parent / (
        'quactlize-model-artifact-' + pin['commit'][:10]) / pin['path']))).resolve(strict=True)
    if digest(bundle / 'manifest.json') != digest(previous / 'results/bundle-manifest.json'):
        raise ValueError('execution bundle differs from the failing run')
    original[0] = str(build / 'bin/llama-perplexity')
    save(results / 'original-command.json', dict(argv=original, source=str(original_path)))
    save(results / 'inputs.json', dict(corpus_sha256=digest(Path(option(original, '-f'))),
        logits_sha256=digest(Path(option(original, '--kl-divergence-base'))),
        model=option(original, '-m'), model_bytes=Path(option(original, '-m')).stat().st_size,
        previous=str(previous), previous_caller=receipt['llama_source_commit']))

    with (results / 'verify.log').open('x') as log:
        subprocess.run([sys.executable, str(ROOT / 'tools/verify_kpack_dispatch.py'), str(bundle),
            '--sdk', str(args.sdk)], check=True, stdout=log, stderr=subprocess.STDOUT)
    print(f'KPACK_FIRST_NONFINITE_BUILD caller=INCREMENTAL_AONECI runtime=UNCHANGED jobs={args.jobs}', flush=True)
    _, build_rc = run([sys.executable, str(ROOT / 'tools/build_kpack_model_ci.py'),
        '--llama', str(llama), '--ncp', str(ncp), '--sdk', str(args.sdk), '--local-llama',
        '--reuse-llama-build', str(build), '--reuse-ncp-build', str(ncp),
        '--output', str(args.output / 'ci'), '--jobs', str(args.jobs),
        '--receipt', str(results / 'caller-ci-build.json')], os.environ.copy(), results / 'build.log')
    if build_rc:
        raise ValueError(f'incremental caller build failed rc={build_rc}; log={results / "build.log"}')

    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'QUACTLIZE_KPACK_'))}
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION', 'DG_LIBRARY_ROOT', 'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB'):
        env.pop(key, None)
    env.update(LD_LIBRARY_PATH=str(build / 'bin') + ':' + env.get('LD_LIBRARY_PATH', ''),
        CUDA_HOME=str(args.sdk / 'CUDA_SDK'),
        DG_JIT_CACHE_DIR=str(previous / 'ci/ncp-jit-cache'),
        QUACTLIZE_PPU_PACK_LIBRARY=str(bundle / 'pack/libquactlize_ppu_pack.so'),
        QUACTLIZE_PPU_BUNDLE=os.environ.get('QUACTLIZE_PPU_BUNDLE', str(previous.parent /
            'quactlize-runtime-artifact-2826cf1-46fc3096e1a1/prebuilt/ppu0010/2826cf1/runtime6-46fc3096e1a1/bundle')))
    if not Path(env['QUACTLIZE_PPU_BUNDLE'], 'manifest.json').is_file():
        raise ValueError('the existing consumer bundle is missing')
    with (results / 'self-test.log').open('x') as log:
        subprocess.run([original[0]], env=env | {'LLAMA_NUMERICAL_DEBUG': 'self-test'},
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    if 'no_sum_overflow=1' not in (results / 'self-test.log').read_text():
        raise ValueError('loaded diagnostic tool is stale; inspect the library search path')
    native_env = env | dict(QUACTLIZE_KPACK_EXECUTION=str(bundle), QUACTLIZE_KPACK_ROUTE='auto',
        QUACTLIZE_KPACK_PAIR_WEIGHTS='1', QUACTLIZE_KPACK_JIT_HELPER=str(ROOT / 'tools/kpack_jit.py'),
        QUACTLIZE_KPACK_JIT_PYTHON=sys.executable,
        QUACTLIZE_KPACK_JIT_CACHE=os.environ.get('JIT_CACHE', str(previous.parent / 'kpack-model-jit-cache')),
        QUACTLIZE_KPACK_DEEPGEMM_HELPER=str(bundle / 'kpack_deepgemm_prewarm.py'))
    save(results / 'environment.json', {k: v for k, v in native_env.items() if k.startswith(
        ('QUACTLIZE_', 'DG_JIT_', 'CUDA_VISIBLE_', 'PPU_SDK'))})
    records = []
    for arm, mode, reference in (('native-logits', 'logits', False),
                                  ('reference-tensors', 'tensors', True), ('native-tensors', 'tensors', False)):
        print(f'KPACK_FIRST_NONFINITE_START arm={arm} token_source=ORIGINAL_CORPUS token_batch=1 callback={int(mode == "tensors")}', flush=True)
        log = results / (arm + '.log')
        dump = results / (arm + '-tensors')
        dump.mkdir()
        text, rc = run(command(original, reference), (env if reference else native_env) |
                       {'LLAMA_NUMERICAL_DEBUG': mode, 'LLAMA_NUMERICAL_DUMP_DIR': str(dump)}, log)
        record = dict(arm=arm, mode=mode, log=str(log), process_rc=rc)
        try:
            record.update(result(text, rc, mode))
            if not reference and '[quactlize-plan]' not in text:
                raise ValueError('native run has no selected-plan receipt')
        except ValueError as error:
            record.update(verdict='INFRASTRUCTURE_OR_COVERAGE_FAIL', error=str(error))
        records.append(record)
        save(results / (arm + '.json'), record)
        print('KPACK_FIRST_NONFINITE_RESULT ' + json.dumps(record), flush=True)
    save(results / 'summary.json', dict(records=records, scope='FIRST_NONFINITE_LOCALIZATION_NOT_ROOT_CAUSE_OR_PERFORMANCE',
        callback_changes_graph_partition=True, source_snapshots='AFTER_NODE_INPLACE_ALIAS_NOT_EXCLUDED'))
    print(f'KPACK_FIRST_NONFINITE_DONE results={results} timing_valid=0', flush=True)
    return int(any(r['verdict'] == 'INFRASTRUCTURE_OR_COVERAGE_FAIL' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
