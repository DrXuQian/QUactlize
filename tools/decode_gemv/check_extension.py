#!/usr/bin/env python3
"""Bounded compile-only gate for the actlize exact-code extension; no GPU launches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=os.environ.get('PPU_SDK'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, help='unchanged b23397ed prototype worktree, optional')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if args.sdk is None or not (args.sdk/'bin/hgcc').is_file():
        parser.error('--sdk or PPU_SDK must name a PPU SDK root')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if not (ROOT/'third_party/actlize/include/cute/arch/mma_ppu0010.hpp').is_file():
        parser.error('initialize the pinned third_party/actlize first')
    if args.baseline and not (args.baseline/'quactlize/execution/mma_staged.cuh').is_file():
        parser.error('--baseline lacks the prototype headers')
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='exact-code-compile-'))
    output.mkdir(parents=True, exist_ok=True)
    if (output/'receipt.json').exists():
        parser.error(f'output already has a receipt: {output}')
    sdk = args.sdk.resolve()
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(sdk/'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    vendor = ROOT/'third_party/actlize'
    jobs = [(kind, staged) for kind in ('standalone', 'adapter') for staged in (False, True)]
    if args.baseline:
        jobs += [('baseline', staged) for staged in (False, True)]

    def source_hashes():
        sources = {}
        for pattern in ('quactlize/execution/mma_*.cuh',
                        'quactlize/include/actlize_extensions/cutlass/gemm/collective/*exact_code*.hpp',
                        'quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/*exact_code*.hpp',
                        'tests/ppu/exact_code_*_compile.cu'):
            for path in ROOT.glob(pattern):
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return sources

    sources = source_hashes()
    def compile_one(job):
        kind, staged = job
        label = kind + ('-staged' if staged else '-direct')
        source_root = args.baseline.resolve() if kind == 'baseline' else ROOT
        fixture = ROOT/'tests/ppu'/('exact_code_extension_compile.cu' if kind == 'standalone'
                                    else 'exact_code_adapter_compile.cu')
        target = output/(label+'.o')
        command = [str(sdk/'bin/hgcc'), *FLAGS,
                   '-I'+str(source_root/'quactlize/include'),
                   '-I'+str(vendor/'include'), '-I'+str(vendor/'tools/util/include')]
        if kind != 'standalone':
            command += ['-I'+str(source_root/'quactlize/execution')]
        if staged:
            command += ['-DEXACT_CODE_STAGED=1']
        command += ['-c', str(fixture), '-o', str(target)]
        log = output/(label+'.log')
        print('EXACT_CODE_COMPILE_START '+label, flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{label}: rc={result.returncode}; {log}\n'+
                               '\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
        for name, flag in (('resources', '--dump-resource-usage=all'), ('isa', '--dump-isa')):
            with (output/(label+'.'+name)).open('w') as stream:
                subprocess.run([str(sdk/'bin/hgobjdump'), flag, str(target)], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('EXACT_CODE_COMPILE_PASS '+label, flush=True)
        return dict(label=label, command=command, object_sha256=hashlib.sha256(target.read_bytes()).hexdigest())

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_one, jobs))
    if sources != source_hashes():
        raise RuntimeError('extension sources changed during compilation; no receipt written')
    receipt = dict(scope='COMPILE_ONLY_NOT_PPU_NUMERICS_OR_PERFORMANCE',
                   source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                   source_dirty=bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)),
                   actlize_commit=subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip(),
                   compiler_version=subprocess.check_output([str(sdk/'bin/hgcc'), '--version'], env=env, text=True),
                   sources=sources, sdk=str(sdk), jobs=results, kernel_launches=0,
                   production_changed=False)
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('EXACT_CODE_COMPILE_DONE '+str(output), flush=True)


if __name__ == '__main__':
    main()
