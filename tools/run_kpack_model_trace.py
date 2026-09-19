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
import re
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_kpack_first_nonfinite import digest, option
from tools.run_kpack_model_validation import inventory, save, traces, run
from tools.resume_kpack_q4_model import completed_benchmark_controls
from tools.run_selected_decode_box import compatibility
from tools.verify_kpack_dispatch import verify


def resolve_paths(cli):
    """Default to recorded inputs, not another checkout or an inherited SDK."""
    if cli.previous:
        cli.previous = cli.previous.resolve(strict=True)
        results = cli.previous / 'results'
        if cli.llama is None:
            receipt = json.loads((results / 'caller-ci-build.json').read_text())
            directory = receipt.get('llama_worktree', {}).get('directory')
            if not directory:
                raise ValueError('previous receipt has no caller worktree; supply --llama')
            cli.llama = Path(directory)
        if cli.sdk is None:
            receipt = json.loads((results / 'compatibility-bundle-manifest.json').read_text())
            directory = receipt.get('sdk')
            if not directory:
                raise ValueError('previous receipt has no SDK directory; supply --sdk')
            cli.sdk = Path(directory)
        if cli.result_root is None:
            cli.result_root = cli.previous.parent
    if cli.llama is None or cli.sdk is None:
        raise ValueError('--diagnostic requires --llama and --sdk')
    cli.sdk = cli.sdk.resolve(strict=True)
    cli.llama = cli.llama.resolve(strict=True)
    cli.result_root = (cli.result_root or Path('/workspace')).resolve(strict=True)


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


def benchmark_inputs(previous, model_name, llama, sdk):
    """Reuse the exact completed benchmark's binaries, model and cache."""
    results = previous / 'results'
    compute, _ = completed_benchmark_controls(results)
    protocol = json.loads((results / 'benchmark/results/protocol.json').read_text())
    models = protocol['plan']['models']
    matches = [m for m in models if m['name'] == model_name]
    if len(matches) != 1:
        raise ValueError('select exactly one model from the completed benchmark')
    model = matches[0]
    devices = model['devices']
    if not re.fullmatch(r'\d+(?:,\d+)?', devices) or len(set(devices.split(','))) != len(devices.split(',')):
        raise ValueError('invalid saved device ordinals')
    if ('tensor_split' in model) != (len(devices.split(',')) == 2):
        raise ValueError('saved tensor split and devices differ')
    if os.environ.get('CUDA_VISIBLE_DEVICES', devices) != devices:
        raise ValueError('trace must use the completed benchmark device ordinals: ' + devices)
    receipt = json.loads((results / 'caller-ci-build.json').read_text())
    build = Path(receipt['build']).resolve(strict=True)
    required = {'bin/' + name for name in ('llama-server', 'llama-batched-bench', 'libggml-cuda.so',
                                          'libncp_fa.so', 'libncp_moe.so')}
    if not required <= receipt.get('files', {}).keys():
        raise ValueError('incomplete caller build receipt')
    for name, expected in receipt['files'].items():
        path = (build / name).resolve(strict=True)
        if not path.is_relative_to(build) or digest(path) != expected:
            raise ValueError('caller binary changed: ' + name)
    command = json.loads((results / 'trace' / model_name / 'reference.command.json').read_text())['argv']
    if option(command, '--model') != model['path'] or Path(option(command, '--binary')).resolve() != (build / 'bin/llama-server').resolve():
        raise ValueError('saved trace command and completed model build differ')
    bundle = Path(option(command, '--bundle')).resolve(strict=True)
    if digest(bundle / 'manifest.json') != digest(results / 'bundle-manifest.json'):
        raise ValueError('runtime package differs from completed benchmark')
    legacy = compatibility(previous, os.environ.get('QUACTLIZE_PPU_BUNDLE'))
    cache = Path(option(command, '--cache')).resolve(strict=True)
    jit = Path(option(command, '--jit-cache')).resolve(strict=True)
    if cache.name != model_name or not (cache / 'manifest.json').is_file() or not jit.is_dir():
        raise ValueError('preserve the completed model cache and JIT cache')
    args = SimpleNamespace(llama=llama, build=build, bundle=bundle, cache=cache.parent,
        jit_cache=jit, asys=sdk / 'asight/bin/asys', inspector=sdk / 'bin/hgobjdump')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_ARG_', 'LLAMA_NUMERICAL_', 'QUACTLIZE_'))}
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION', 'DG_LIBRARY_ROOT',
                'GGML_NCP_FA_LIB', 'GGML_NCP_MOE_LIB', 'GGML_NCP_GDN_LIB'):
        env.pop(key, None)
    env.update(PPU_SDK=str(sdk), CUDA_HOME=str(sdk / 'CUDA_SDK'), CUDA_VISIBLE_DEVICES=devices,
        DG_JIT_HGCC_COMPILER=str(sdk / 'bin/hgcc'),
        QUACTLIZE_PPU_BUNDLE=str(legacy), QUACTLIZE_PPU_PACK_LIBRARY=str(bundle / 'pack/libquactlize_ppu_pack.so'),
        QUACTLIZE_KPACK_EXECUTION=str(bundle), QUACTLIZE_KPACK_ROUTE='auto', QUACTLIZE_KPACK_PAIR_WEIGHTS='1',
        QUACTLIZE_KPACK_COMPUTE=compute,
        QUACTLIZE_KPACK_GATE_UP=str(int(protocol['production_fusion'] == 'MOE_CHAIN_AND_GPU_GATE_UP_PAIR')),
        QUACTLIZE_KPACK_JIT_HELPER=str(ROOT / 'tools/kpack_jit.py'), QUACTLIZE_KPACK_JIT_PYTHON=sys.executable,
        QUACTLIZE_KPACK_JIT_CACHE=str(jit), QUACTLIZE_KPACK_DEEPGEMM_HELPER=str(bundle / 'kpack_deepgemm_prewarm.py'))
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, (build / 'bin', sdk / 'CUDA_SDK/targets/x86_64-linux/lib',
        sdk / 'targets/x86_64-linux/lib', sdk / 'lib'))) + ':' + env.get('LD_LIBRARY_PATH', '')
    return args, model, env, previous


def session_creation_failed(output):
    receipts = list(output.glob('*/asys-preflight/summary.json'))
    return any(r.get('status') == 'FAIL' and r.get('attempts') and
               all(a.get('session_creation_failure') for a in r['attempts'])
               for r in (json.loads(p.read_text()) for p in receipts))


def private_trace(args, model, directory, env):
    """Do not reuse or stop the host's shared profiler daemons."""
    from tools.kpack_asys_scope import private_command
    args.output = directory / 'private'
    plan = directory / 'trace-plan.json'
    save(plan, dict(models=[model]))
    command = [sys.executable, ROOT / 'tools/run_kpack_model_validation.py', '--phase', 'trace',
        '--llama', args.llama, '--build', args.build, '--bundle', args.bundle, '--plan', plan,
        '--cache', args.cache, '--jit-cache', args.jit_cache, '--output', args.output,
        '--logits', directory / 'unused-logits', '--corpus', directory / 'unused-corpus',
        '--asys', args.asys, '--inspector', args.inspector]
    command = private_command(command, directory / 'profiler-tmp',
        [args.llama, args.build, args.bundle, args.cache, args.jit_cache, ROOT, Path(model['path']),
         args.asys, args.inspector, Path(sys.executable)] +
        [Path(env[k]) for k in ('QUACTLIZE_PPU_BUNDLE', 'DG_JIT_CACHE_DIR') if env.get(k)])
    private_env = dict(env, TMPDIR='/tmp')
    for key in ('PERFETTO_PRODUCER_SOCK_NAME', 'PERFETTO_CONSUMER_SOCK_NAME', 'ASIGHT_SESSION_FOLDER_PATH'):
        private_env.pop(key, None)
    print('KPACK_MODEL_ASYS private_service=1 host_services=UNTOUCHED build=NONE', flush=True)
    log = directory / 'private-profiler.log'
    try:
        run(command, log, private_env)
    except ValueError as error:
        if 'unshare failed: Operation not permitted' in log.read_text(errors='replace'):
            raise ValueError('private Asys namespace is not permitted by this container; '
                'no host service was stopped. Inspect shared services-before/after.json and '
                'use a permitted isolated profiler container or a compatible idle service. Log: ' + str(log)) from error
        raise
    return args.output / model['name']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--diagnostic', type=Path)
    source.add_argument('--previous', type=Path, help='completed model benchmark; TP1 or TP2, no rebuild')
    parser.add_argument('--llama', type=Path, help='default: previous caller worktree receipt')
    parser.add_argument('--sdk', type=Path, help='default: previous compatibility SDK receipt')
    parser.add_argument('--model', default='qwen35-35b-q4km')
    parser.add_argument('--result-root', type=Path, help='default: previous run parent, or /workspace for a diagnostic')
    parser.add_argument('--profiler-scope', choices=('auto', 'shared', 'private'), default='auto',
                        help='auto isolates /tmp and helper PIDs only after session creation fails')
    cli = parser.parse_args()
    resolve_paths(cli)
    args, model, env, previous = (benchmark_inputs(cli.previous, cli.model, cli.llama, cli.sdk) if cli.previous else
                                  inputs(cli.diagnostic, cli.model, cli.llama, cli.sdk))
    verify(args.bundle, sdk=cli.sdk)
    inv = inventory(Path(model['path']))
    if not inv['eligible']:
        raise ValueError('selected model has no supported matrices')
    directory = Path(tempfile.mkdtemp(prefix='kpack-model-asys.', dir=cli.result_root))
    output = directory / 'results'
    output.mkdir()
    save(output / 'inventory.json', inv)
    save(output / 'inputs.json', dict(model=model, previous=str(previous),
        diagnostic=str(cli.diagnostic) if cli.diagnostic else None, build=str(args.build), bundle=str(args.bundle),
        profiler_scope=cli.profiler_scope, devices=env['CUDA_VISIBLE_DEVICES'],
        runtime_manifest_sha256=digest(args.bundle / 'manifest.json'),
        first_request='EXCLUDED_IN_EACH_PROCESS', request_batch=1, prefill=2048, decode=16,
        numerical_callback=False, accuracy_admission='NOT_RETESTED',
        timing_scope='PROFILE_DIAGNOSTIC_NOT_PERFORMANCE_ADMISSION'))
    print(f'KPACK_MODEL_ASYS run={directory} model={cli.model} build=NONE numerical=NONE', flush=True)
    original_env = os.environ.copy()
    status = dict(status='INCOMPLETE', accuracy_admission='NOT_RETESTED',
                  performance_admission='NOT_ADMITTED_BY_PROFILER')
    reports = output
    try:
        os.environ.clear()
        os.environ.update(env)
        if cli.profiler_scope == 'private':
            reports = output / 'private' / model['name']
            reports = private_trace(args, model, output, env)
        else:
            try:
                traces(args, model, output, output / 'inventory.json')
            except ValueError:
                if cli.profiler_scope != 'auto' or not session_creation_failed(output):
                    raise
                reports = output / 'private' / model['name']
                reports = private_trace(args, model, output, env)
        status['status'] = 'TRACE_PAIR_COMPLETE'
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        status.update(status='FAIL', error=str(error))
        raise
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        save(output / 'status.json', status | dict(reports=str(reports)))
        archive = Path(str(directory) + '.results.tgz')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in sorted(output.rglob('*')):
                if (path.is_file() and not path.is_symlink() and
                        'profiler-tmp' not in path.relative_to(output).parts and
                        not path.name.endswith(('.asysrep', '.sqlite', '.sqlite-wal', '.sqlite-shm'))):
                    tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
        print(f'results={archive}', flush=True)
        for label, arm in (('reference', 'reference'), ('kpack', 'native')):
            report = reports / arm / 'proof.asysrep'
            print(f'asys_{label}={report if report.is_file() else "NOT_CAPTURED"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
