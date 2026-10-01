#!/usr/bin/env python3
"""Build the K-pack product from source. No binary downloads or device execution."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import LIBRARIES, sha
from quactlize.runtime.identity import digest
from tools.verify_kpack_dispatch import verify


def source_identity():
    suffixes = {'.py', '.h', '.hpp', '.cuh', '.cu', '.cpp', '.inc', '.inl', '.json'}
    paths = sorted(p for folder in ('quactlize', 'tools', 'policies', 'third_party/actlize/include',
                                   'third_party/actlize/tools/util/include', 'third_party/actlize/examples/common')
                   for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    return digest([(str(p.relative_to(ROOT)), sha(p)) for p in paths])


def sdk_identity(sdk):
    files = ['bin/hgcc', 'bin/hgobjdump', *[f'lib/lib{x}.so' for x in LIBRARIES]]
    for name in files:
        if not (sdk / name).is_file():
            raise ValueError(f'missing SDK input: {sdk / name}; use the SDK root, not CUDA_SDK')
    return {name: sha(sdk / name) for name in files}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def build(args):
    sdk = args.sdk.resolve(strict=True)
    if args.jobs < 1:
        raise ValueError('--jobs must be positive')
    if not (ROOT / 'third_party/actlize/include/ppu_include.hpp').is_file():
        raise ValueError('initialize third_party/actlize with git submodule update --init --recursive')
    identity = dict(source=source_identity(), sdk=sdk_identity(sdk))
    output = args.output.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('output must be a dedicated build directory, not the repository or its parent')
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = output / 'build.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['identity'] != identity:
                raise ValueError('source or SDK changed; use a new output directory')
        else:
            if args.resume and any(p.name != 'build.lock' for p in output.iterdir()):
                raise ValueError('cannot resume an output without build.json')
            receipt = dict(schema='quactlize.source-build.v1', identity=identity, stages={},
                           device_validated=False)
            write_json(receipt_path, receipt)

        def stage(name, script, extra=(), finalize=None):
            previous = receipt['stages'].get(name)
            if previous:
                path = output / previous['path']
                if path.is_symlink() or not path.resolve(strict=True).is_relative_to(output):
                    raise ValueError('build receipt escapes output directory: ' + name)
                manifest = path / 'manifest.json'
                if sha(manifest) != previous['manifest_sha256']:
                    raise ValueError('completed build receipt changed: ' + name)
                built = read_json(manifest)
                if 'library' in built and sha(path / built['library']) != built['sha256']:
                    raise ValueError('completed build payload changed: ' + name)
                print(f'BUILD stage={name} status=REUSED', flush=True)
                return path
            attempt = 1
            while (output / f'{name}-{attempt}').exists():
                attempt += 1
            path = output / f'{name}-{attempt}'
            print(f'BUILD stage={name} status=START jobs={args.jobs} path={path}', flush=True)
            command = [sys.executable, str(ROOT / 'tools' / script), '--sdk', str(sdk),
                       '--output', str(path), '--jobs', str(args.jobs), *map(str, extra)]
            subprocess.run(command, check=True)
            if finalize is not None:
                finalize(path)
            receipt['stages'][name] = dict(path=path.name, manifest_sha256=sha(path / 'manifest.json'))
            write_json(receipt_path, receipt)
            return path

        pack = stage('pack', 'build_kpack_pack.py')
        execution = stage('execution', 'build_kpack_execution.py')
        prefill = stage('prefill', 'build_kpack_prefill_runtime.py')
        fusion = stage('fusion', 'build_kpack_gate_up.py')
        def finalize_runtime(runtime):
            manifest = read_json(runtime / 'manifest.json')
            (runtime / 'pack').mkdir(exist_ok=True)
            for name in ('libquactlize_ppu_pack.so', 'manifest.json'):
                shutil.copy2(pack / name, runtime / 'pack' / name)
            manifest['pack'] = dict(schema='quactlize.kpack-producer-package.v1', files={
                'pack/' + name: sha(runtime / 'pack' / name)
                for name in ('libquactlize_ppu_pack.so', 'manifest.json')})
            from tools import paired_decode_selection as paired
            selection = read_json(paired.EVIDENCE)
            if paired.header(selection) != paired.OUTPUT.read_text():
                raise ValueError('paired recipe table differs from its policy')
            fusion_build = read_json(fusion / 'manifest.json')
            library = fusion_build['library']
            shutil.copy2(fusion / library, runtime / library)
            shutil.copy2(fusion / 'manifest.json', runtime / 'gate-up-runtime.json')
            manifest['paired_gate_up'] = dict(schema='quactlize.paired-model.v2', library=library,
                sha256=sha(runtime/library), receipt='gate-up-runtime.json',
                receipt_sha256=sha(runtime/'gate-up-runtime.json'), layout_id=fusion_build['layout_id'],
                q8_shared='F16_N512_K2048_T1_8_N1024_K3072_T1', q4_routed='GENERATED_PAIRED_SELECTION_V2',
                canonical_retained=True, device_validated=False, selection=selection,
                cost_authority=selection['archive_sha256'])
            write_json(runtime / 'manifest.json', manifest)
            verify(runtime, sdk=sdk)

        runtime = stage('runtime', 'build_kpack_dispatch.py', (
            '--jit-only', '--execution-bundle', execution, '--prefill-runtime', prefill),
            finalize=finalize_runtime)
        verify(runtime, sdk=sdk)
        if source_identity() != identity['source']:
            raise ValueError('sources changed during build')
        receipt['stages']['runtime']['manifest_sha256'] = sha(runtime/'manifest.json')
        receipt['runtime'] = runtime.name
        receipt['complete'] = True
        write_json(receipt_path, receipt)
        print(f'BUILD status=COMPLETE runtime={runtime} device_validated=0', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--resume', action='store_true', help='reuse complete stages with identical source and SDK')
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'BUILD FAIL: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
