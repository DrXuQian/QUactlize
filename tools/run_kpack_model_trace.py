#!/usr/bin/env python3
"""Capture a warmed model A/B trace using an existing diagnostic's runtime.

No build, numerical callback, correctness gate or config sweep is run. The
existing model trace helper excludes the first complete request in each
process and gives both arms the same prompt token IDs.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_kpack_first_nonfinite import digest, option
from tools.run_kpack_model_validation import inventory, save, traces
from tools.verify_kpack_dispatch import verify


def inputs(diagnostic, model_name, llama, sdk):
    results = diagnostic / 'diagnostic/results'
    receipt = json.loads((results / 'caller-ci-build.json').read_text())
    environment = json.loads((results / 'environment.json').read_text())
    previous = Path(json.loads((results / 'inputs.json').read_text())['previous'])
    models = json.loads((previous / 'results/model-plan.json').read_text())['models']
    matches = [m for m in models if m['name'] == model_name]
    if len(matches) != 1:
        raise ValueError('select exactly one model from the previous model-plan.json')
    model = matches[0]
    command = json.loads((previous / 'results/numerical' / model_name /
                          'b1-kpack-reference.command.json').read_text())['argv']
    if option(command, '-m') != model['path']:
        raise ValueError('model plan and previous native command differ')
    cache = Path(option(command, '--kpack-cache'))
    if cache.name != model_name:
        raise ValueError('previous model cache is not named for the selected model')
    build = Path(receipt['build'])
    bundle = Path(environment['QUACTLIZE_KPACK_EXECUTION'])
    if digest(bundle / 'manifest.json') != digest(previous / 'results/bundle-manifest.json'):
        raise ValueError('runtime bundle differs from the previous model run')
    if Path(environment['PPU_SDK']).resolve() != sdk.resolve():
        raise ValueError('SDK path differs from the recorded diagnostic')
    args = SimpleNamespace(llama=llama, build=build, bundle=bundle, cache=cache.parent,
        jit_cache=Path(environment['QUACTLIZE_KPACK_JIT_CACHE']),
        asys=sdk / 'asight/bin/asys', inspector=sdk / 'bin/hgobjdump')
    for path in (Path(model['path']), build / 'bin/llama-server',
                 llama / 'tests/quactlize_native.py', args.asys, args.inspector):
        if not path.is_file():
            raise ValueError(f'missing existing trace input: {path}')
    if not cache.is_dir() or not args.jit_cache.is_dir():
        raise ValueError('keep the existing model cache and JIT cache')
    for path in (build / 'bin/llama-server', args.asys, args.inspector):
        if not os.access(path, os.X_OK):
            raise ValueError(f'trace executable is not executable: {path}')
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('LLAMA_ARG_', 'LLAMA_NUMERICAL_', 'QUACTLIZE_'))}
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION',
                'DG_LIBRARY_ROOT', 'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB'):
        env.pop(key, None)
    env.update(environment)
    env.update(CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', environment['CUDA_VISIBLE_DEVICES']),
        LD_LIBRARY_PATH=str(build / 'bin') + ':' + env.get('LD_LIBRARY_PATH', ''),
        CUDA_HOME=str(sdk / 'CUDA_SDK'),
        QUACTLIZE_KPACK_JIT_HELPER=str(ROOT / 'tools/kpack_jit.py'),
        QUACTLIZE_KPACK_JIT_PYTHON=sys.executable)
    if not env['CUDA_VISIBLE_DEVICES'].isdigit():
        raise ValueError('select one visible device ordinal')
    return args, model, env, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--model', default='qwen35-35b-q4km')
    parser.add_argument('--result-root', type=Path, default=Path('/workspace'))
    cli = parser.parse_args()
    args, model, env, previous = inputs(cli.diagnostic, cli.model, cli.llama, cli.sdk)
    verify(args.bundle)
    inv = inventory(Path(model['path']))
    if not inv['eligible']:
        raise ValueError('selected model has no supported matrices')
    directory = Path(tempfile.mkdtemp(prefix='kpack-model-asys.', dir=cli.result_root))
    output = directory / 'results'
    output.mkdir()
    save(output / 'inventory.json', inv)
    save(output / 'inputs.json', dict(model=model, previous=str(previous),
        diagnostic=str(cli.diagnostic), build=str(args.build), bundle=str(args.bundle),
        runtime_manifest_sha256=digest(args.bundle / 'manifest.json'),
        first_request='EXCLUDED_IN_EACH_PROCESS', request_batch=1, prefill=2048, decode=16,
        numerical_callback=False, accuracy_admission='NOT_RETESTED',
        timing_scope='PROFILE_DIAGNOSTIC_NOT_PERFORMANCE_ADMISSION'))
    print(f'KPACK_MODEL_ASYS run={directory} model={cli.model} build=NONE numerical=NONE', flush=True)
    original_env = os.environ.copy()
    status = dict(status='INCOMPLETE', accuracy_admission='NOT_RETESTED',
                  performance_admission='NOT_ADMITTED_BY_PROFILER')
    try:
        os.environ.clear()
        os.environ.update(env)
        traces(args, model, output, output / 'inventory.json')
        status['status'] = 'TRACE_PAIR_COMPLETE'
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        save(output / 'status.json', status)
        archive = Path(str(directory) + '.results.tgz')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith(('.asysrep', '.sqlite', '.sqlite-wal', '.sqlite-shm')):
                    tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
        print(f'results={archive}', flush=True)
        print(f'asys_reference={output}/reference/proof.asysrep', flush=True)
        print(f'asys_kpack={output}/native/proof.asysrep', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
