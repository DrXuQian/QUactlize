#!/usr/bin/env python3
"""Build five native PPU1.5 TC modules and check 60 dense FQ/SF cases. No sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_zw890_bringup as base
from tools.ppu15.tc_plan import parents, plan, verdict
from quactlize.runtime.compiler import Compiler, FLAGS, LIBRARIES


def source_identity():
    result = base.identity()
    files = [ROOT/'tools/run_zw890_tc.py', *sorted((ROOT/'tools/ppu15').glob('*.py'))]
    # Include all Quactlize source/header inputs, not just generated wrappers.
    files += sorted(p for p in (ROOT/'quactlize').rglob('*')
                    if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh', '.inl', '.cu', '.cpp', '.py'))
    result['sources'].update({str(p.relative_to(ROOT)): base.sha(p) for p in files})
    result['actlize_diff_sha256'] = hashlib.sha256(subprocess.check_output(
        ['git', '-C', str(ROOT/'third_party/actlize'), 'diff', 'HEAD', '--'])).hexdigest()
    return result


def module_commands(sdk, parent, work):
    includes = [ROOT/'quactlize'/name for name in ('include', 'runtime', 'integrations', 'decode')]
    includes += [ROOT/'third_party/actlize'/name
                 for name in ('include', 'tools/util/include', 'examples/common')]
    flags = [f for f in FLAGS if not f.startswith('-arch=')]
    q = parent['qtype']
    return [
        [str(sdk/'bin/hgcc'), *flags, '-arch=ppu_15', '-DQUACTLIZE_PPU_ARCH=150',
         f'-DPPU_PACKED_SCALE={int(parent["route"].startswith("fq"))}',
         f'-DPPU_PACKED_FORMAT={ {8: 0, 12: 0, 14: 4}[q] }',
         *['-I'+str(p) for p in includes], '-c', str(work/'kernel.cu'), '-o', str(work/'kernel.o')],
        ['g++', '-shared', '-Wl,-Bsymbolic', str(work/'kernel.o'), '-o', str(work/'kernel.so'),
         '-L'+str(sdk/'lib'), *['-l'+lib for lib in LIBRARIES]],
    ]


def build_module(sdk, output, parent, identity, env):
    work = output/'modules'/parent['symbol']
    work.mkdir(parents=True)
    key = hashlib.sha256(json.dumps(dict(source=identity, parent=parent, target=150),
                                    sort_keys=True).encode()).hexdigest()
    # Use the same code generator and module ABI as the existing runtime.
    source = Compiler.source(None, parent, key) + '''
static_assert(cute::is_same_v<quactlize::runtime::Dense::Mainloop::ArchTag,
                              cutlass::arch::PPU0015>);
static_assert(cute::size<0>(quactlize::runtime::Dense::Mainloop::TiledMma::AtomShape_MNK{}) == 16);
'''
    (work/'kernel.cu').write_text(source)
    commands = module_commands(sdk, parent, work)
    receipt = dict(parent=parent, key=key, target='ppu_15', commands=commands,
                   source_sha256=base.sha(work/'kernel.cu'), path=str(work/'kernel.so'))
    path = work/'receipt.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print('ZW890_TC_BUILD_START '+parent['symbol'], flush=True)
    for index, command in enumerate(commands):
        base.command(command, work/f'build-{index}.log', env)
    receipt.update(sha256=base.sha(work/'kernel.so'), object_sha256=base.sha(work/'kernel.o'))
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    print('ZW890_TC_BUILD_PASS '+parent['symbol'], flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--device', type=int, default=int(os.environ.get('DEVICE', '0')))
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--output', type=Path)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--build-only', action='store_true')
    a = p.parse_args()
    if a.plan_only:
        print('ZW890_TC_PLAN '+json.dumps(plan())); return 0
    if a.device < 0 or a.jobs < 1: p.error('device must be nonnegative and jobs positive')
    sdk = a.sdk.resolve()
    if not (sdk/'bin/hgcc').is_file(): p.error('--sdk/PPU_SDK must contain bin/hgcc')
    if a.output:
        output = a.output.resolve()
        try: output.mkdir(parents=True, exist_ok=False)
        except FileExistsError: p.error(f'--output must be a new directory: {output}')
    else: output = Path(tempfile.mkdtemp(prefix='qz-zw890-tc-')).resolve()
    env = dict(os.environ, PPU_SDK=str(sdk), CUDA_VISIBLE_DEVICES=str(a.device),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH', '')
    summary = dict(plan(), status='FAIL', requested_device=a.device, sdk=str(sdk), modules={})
    started = time.monotonic()
    print(f'ZW890_TC_START output={output} device={a.device}', flush=True)
    try:
        summary['source'] = source_identity()
        (output/'plan.json').write_text(json.dumps(summary, indent=2)+'\n')
        if not a.build_only:
            base.command([sys.executable, '-c', 'import numpy, torch, gguf'], output/'dependencies.log', env)
        # Build only the small established device/packer support library; none
        # of stage1's 192 cases are rerun by this command.
        base.build(sdk, output, a.jobs, env)
        with ThreadPoolExecutor(max_workers=min(a.jobs, len(parents()))) as pool:
            records = list(pool.map(lambda parent: build_module(
                sdk, output, parent, summary['source'], env), parents()))
        (output/'modules.json').write_text(json.dumps(records, indent=2)+'\n')
        if a.build_only:
            summary['status'] = 'BUILD_ONLY'; return 0
        for record in records:
            name = record['parent']['symbol']; log = output/(name+'.log')
            argv = [sys.executable, str(ROOT/'tools/ppu15/tc_check.py'), '--sdk', str(sdk),
                    '--output', str(output), '--parent', name]
            print(f'ZW890_TC_SUITE_START parent={name} expected=12 log={log}', flush=True)
            with log.open('w') as stream:
                rc = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            result = output/(name+'.jsonl')
            rows = [json.loads(x) for x in result.read_text().splitlines()] if result.is_file() else []
            summary['modules'][name] = verdict(record['parent'], rows, rc)
            print('ZW890_TC_SUITE_DONE '+json.dumps(dict(parent=name, **summary['modules'][name])), flush=True)
            if summary['modules'][name]['status'] != 'PASS':
                print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]), flush=True)
        summary['status'] = 'PASS' if all(r['status'] == 'PASS' for r in summary['modules'].values()) else 'FAIL'
        return 0 if summary['status'] == 'PASS' else 1
    except Exception as exc:
        summary['error'] = str(exc)
        print('ZW890_TC_FAIL '+str(exc), flush=True)
        return 1
    finally:
        summary['elapsed_seconds'] = time.monotonic()-started
        if (output/'device.json').is_file(): summary['device'] = json.loads((output/'device.json').read_text())
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        archive = base.review(output)
        print('ZW890_TC_DONE '+json.dumps(dict(status=summary['status'], results=str(archive),
              production_enabled=False, performance_admitted=False)), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
