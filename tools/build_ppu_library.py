#!/usr/bin/env python3
"""Build the complete public K-pack library and its private runtime closure.

No device or model is needed to build. This is not a device/performance gate.
ncp_flash_lib calls this builder; it must not generate selectors or providers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import (Compiler, FLAGS, LIBRARIES, HEADER_SUFFIXES,
                                       sha, source_contract, validate_actlize_headers)
from tools.build_kpack_pack import build as build_pack
from tools.build_kpack_execution import build as build_execution
from tools.build_kpack_prefill_runtime import build as build_prefill
from tools.build_kpack_dispatch import catalog
from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.dispatch.prefill_policy import check_receipt, receipt_fields

PUBLIC_EXPORTS = (
    'quactlize_ppu_kpack_canonical_arrangement_v1',
    'quactlize_ppu_kpack_sizes_for_arrangement_v1',
    'quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2',
    'quactlize_ppu_prepare_gate_up_dev_for_arrangement_v1',
    *('quactlize_ppu_dense_' + suffix for suffix in (
        'supported_v1', 'create_v1', 'create_with_options_v1', 'prepare_v1', 'prepare_candidate_v1',
        'run_v1', 'destroy_v1', 'error_v1')),
)
JIT_PYTHON = (
    'quactlize/__init__.py', 'quactlize/runtime/__init__.py',
    'quactlize/runtime/compiler.py', 'quactlize/runtime/tuning.py',
    'quactlize/runtime/dispatch.py', 'quactlize/runtime/candidates.py',
    'quactlize/decode/compiler.py', 'quactlize/decode/grouped_compiler.py',
    'tools/kpack_jit.py',
)


def copy_relative(source, destination):
    target = destination / source.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def stage_jit(compiler, destination):
    """Preserve the compiler's exact include closure/source contract."""
    destination.mkdir(parents=True, exist_ok=False)
    files = {ROOT / name for name in JIT_PYTHON}
    for directory in compiler.includes:
        files.update(p for p in directory.rglob('*') if p.is_file() and
                     p.suffix in HEADER_SUFFIXES)
    for directory in (ROOT, ROOT / 'third_party/actlize'):
        files.update(p for p in directory.iterdir() if p.is_file() and
                     p.name.lower().startswith(('license', 'copyright', 'notice')))
    for source in sorted(files):
        copy_relative(source, destination)
    return files


def validate_actlize_checkout(root):
    """No nested fetch: ncp/CI owns the authenticated dependency checkout."""
    root = Path(root)
    dependency = root / 'third_party/actlize'
    if (root / '.git').exists():
        expected = subprocess.check_output(
            ['git', '-C', str(root), 'ls-tree', 'HEAD', 'third_party/actlize'], text=True).split()
        if len(expected) != 4 or expected[:2] != ['160000', 'commit']:
            raise ValueError('Quactlize HEAD has no third_party/actlize gitlink')
        actual = subprocess.run(['git', '-C', str(dependency), 'rev-parse', 'HEAD'],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if actual.returncode or actual.stdout.strip() != expected[2]:
            raise ValueError('actlize checkout differs from Quactlize gitlink: '
                f'want={expected[2]} got={actual.stdout.strip() or "MISSING"}; '
                'checkout the pinned commit with CI SSH credentials before building; '
                'the library builder does not fetch or modify submodules')
        dirty = subprocess.run(['git', '-C', str(dependency), 'diff', '--quiet', 'HEAD', '--',
                                'include', 'tools/util/include', 'examples/common'])
        if dirty.returncode:
            raise ValueError('actlize JIT headers differ from its checked-out commit; '
                             'use a clean pinned checkout, do not package modified vendor headers')
    validate_actlize_headers(root)


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    if manifest.get('schema') != 'quactlize.ppu-library.v1':
        raise ValueError('unknown public library manifest')
    for name, expected in manifest['payloads'].items():
        path = output / name
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError('library payload differs: ' + name)
    exports = {line.split()[-1] for line in subprocess.check_output(
        ['nm', '-D', '--defined-only', str(output / 'libquactlize_ppu.so')], text=True).splitlines() if line.split()}
    if set(PUBLIC_EXPORTS) - exports:
        raise ValueError('public exports missing: ' + repr(sorted(set(PUBLIC_EXPORTS) - exports)))
    if 'final_selection' in manifest:
        receipt = manifest['final_selection']
        if receipt['path'] != 'quactlize-runtime/final-selection.json' or receipt['path'] not in manifest['payloads']:
            raise ValueError('invalid library selection receipt path')
        actual = validate_inventory(json.loads((output/receipt['path']).read_text()),
                                    manifest['execution_receipt'], jit=True)
        if json.loads(json.dumps(actual)) != receipt['capabilities']:
            raise ValueError('public library selection capability closure differs')
        header = manifest['source_hashes'].get('policies/kpack_smallm_effective_v1.hpp')
        if not header or manifest['execution_receipt'].get('smallm_effective_sha256') != header:
            raise ValueError('public provider and Q4 execution selectors differ')
    if 'prefill_confirmed' in manifest:
        receipt=manifest['prefill_confirmed']
        if (receipt.get('path')!='quactlize-runtime/prefill-confirmed.json' or
                receipt['path'] not in manifest['payloads']):
            raise ValueError('public provider confirmed prefill receipt differs')
        check_receipt(receipt,manifest['source_hashes'],manifest['payloads'][receipt['path']])
    if 'dense_helpers' in manifest:
        helper=manifest['dense_helpers']
        path=helper['runtime_receipt']
        if (path!='quactlize-runtime/prefill-runtime.json' or path not in manifest['payloads'] or
                helper['policy_sha256']!=manifest['source_hashes'].get('quactlize/prefill/dense_helpers.hpp')):
            raise ValueError('public provider helper policy binding differs')
        runtime=json.loads((output/path).read_text())
        if (runtime['source_hashes'].get('quactlize/prefill/dense_helpers.hpp')!=helper['policy_sha256'] or
                runtime['sha256']!=manifest['payloads'].get('quactlize-runtime/libquactlize_ppu_prefill.so')):
            raise ValueError('public provider and dense helper runtime differ')
        if 'full_table_sha256' in helper:
            table='quactlize/prefill/dense_full_helpers.inc'
            if (helper['full_table_sha256']!=manifest['source_hashes'].get(table) or
                    helper['full_table_sha256']!=runtime['source_hashes'].get(table) or
                    helper['full_evidence_sha256']!=manifest['source_hashes'].get('policies/dense_full_helpers_20261023.json')):
                raise ValueError('full-only helper table/runtime evidence differs')
    return manifest


def build(sdk, output, jobs):
    # --library bypasses the legacy shell builder, including its gitlink guard.
    # Fail here before any output directory or expensive device compilation.
    validate_actlize_checkout(ROOT)
    sdk = sdk.resolve(strict=True)
    output = output.resolve()
    if jobs < 1:
        raise ValueError('jobs must be positive')
    for path in (sdk / 'bin/hgcc', sdk / 'bin/hgobjdump',
                 *(sdk / 'lib' / f'lib{name}.so' for name in LIBRARIES)):
        if not path.is_file():
            raise ValueError(f'incomplete PPU SDK: {path}')
    output.mkdir(parents=True, exist_ok=False)
    # Intermediates must not enter the installable runtime directory.
    work = output.parent / (output.name + '.build')
    work.mkdir()
    runtime = output / 'quactlize-runtime'
    runtime.mkdir()
    start = time.monotonic()
    public_sources = {p: sha(p) for directory in ('quactlize/dense', 'quactlize/dispatch')
                      for p in (ROOT / directory).iterdir() if p.is_file()}
    public_sources.update({p: sha(p) for p in (ROOT/'policies').iterdir() if p.suffix in ('.hpp', '.json')})
    helper_policy=ROOT/'quactlize/prefill/dense_helpers.hpp'
    public_sources[helper_policy]=sha(helper_policy)
    from tools.generate_dense_full_helpers import EVIDENCE as FULL_HELPERS, HEADER as FULL_HELPERS_HEADER, render as full_helpers_header
    if full_helpers_header(json.loads(FULL_HELPERS.read_text()))!=FULL_HELPERS_HEADER.read_text():
        raise ValueError('stale full-only helper table; regenerate before building')
    public_sources[FULL_HELPERS_HEADER]=sha(FULL_HELPERS_HEADER)
    from tools.generate_decode_selection import EVIDENCE, OUTPUT, effective, header
    if header(effective(json.loads(EVIDENCE.read_text()))) != OUTPUT.read_text():
        raise ValueError('stale effective decode catalog; regenerate before building')
    from tools.fit_prefill_refresh import EVIDENCE as PREFILL_EVIDENCE, HEADER as PREFILL_HEADER, header as prefill_header
    confirmed=json.loads(PREFILL_EVIDENCE.read_text())
    if prefill_header(confirmed)!=PREFILL_HEADER.read_text():
        raise ValueError('stale confirmed prefill catalog; regenerate before building')
    shutil.copy2(PREFILL_EVIDENCE,runtime/'prefill-confirmed.json')
    final_selection = plan_smallm(work)
    (runtime/'final-selection.json').write_text(json.dumps(final_selection, indent=2)+'\n')
    # The existing device builders retain the exact production flags and
    # SIMT recipe inventory. No sweep, regenerated heuristic or fake backend.
    build_pack(sdk, work / 'packing')
    execution = build_execution(sdk, work / 'execution', jobs)
    capabilities = validate_inventory(final_selection, execution, jit=True)
    build_prefill(sdk, work / 'prefill', jobs)
    for component, name in (('execution', 'libquactlize_ppu_execution.so'),
                            ('prefill', 'libquactlize_ppu_prefill.so')):
        shutil.copy2(work / component / name, runtime / name)
    shutil.copy2(work/'prefill/manifest.json',runtime/'prefill-runtime.json')
    compiler = Compiler(sdk, work / 'modules', jobs)
    contract = source_contract(compiler.identity)
    (work / 'catalog.inc').write_text(catalog([], contract))
    stage_jit(compiler, runtime / 'jit')
    # Verify relocation before linking a dispatcher that pins this snapshot.
    actual = subprocess.check_output([sys.executable, '-c',
        'from quactlize.runtime.compiler import Compiler, source_contract; '
        'import sys; from pathlib import Path; '
        'print(source_contract(Compiler(Path(sys.argv[1]),Path(sys.argv[2])).identity))',
        str(sdk), str(work / 'snapshot-check')], cwd=runtime / 'jit', text=True).strip()
    if actual != contract:
        raise ValueError('installed JIT closure differs from dispatcher sources')
    env = dict(os.environ, PATH=str(sdk / 'bin') + ':' + os.environ.get('PATH', ''),
               LD_LIBRARY_PATH=str(sdk / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    commands = []

    def run(name, command):
        commands.append(command)
        with (work / (name + '.log')).open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed rc={result.returncode}; see {work / (name + ".log")}')
        print('QUACTLIZE_PPU_BUILD phase=' + name, flush=True)

    run('dispatch', ['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread', '-Wl,-Bsymbolic',
        '-I' + str(work), str(ROOT / 'quactlize/dispatch/binding.cpp'), '-ldl',
        '-o', str(runtime / 'libquactlize_kpack_dispatch.so')])
    include = [ROOT / 'quactlize/include', ROOT / 'third_party/actlize/include', sdk / 'include']

    def compile_one(source):
        obj = work / (Path(source).stem + '.o')
        command = ([str(sdk / 'bin/hgcc'), *FLAGS] if source.endswith('.cu') else
                   ['g++', '-std=c++17', '-O2', '-fPIC', '-pthread'])
        run(Path(source).stem, [*command, *['-I' + str(p) for p in include],
            '-c', str(ROOT / source), '-o', str(obj)])
        return obj

    with ThreadPoolExecutor(max_workers=min(jobs, 2)) as pool:
        objects = list(pool.map(compile_one, ('quactlize/dense/runtime.cpp', 'quactlize/dense/copy.cu')))
    objects += [work / 'packing' / (name + '.o') for name in ('sizes.cpp', 'ppu_pack.cu')]
    run('public', ['g++', '-shared', '-pthread', '-Wl,-Bsymbolic', '-Wl,-z,defs',
        '-Wl,-soname,libquactlize_ppu.so', '-Wl,-rpath,$ORIGIN/quactlize-runtime',
        *map(str, objects), '-L' + str(runtime), '-lquactlize_kpack_dispatch',
        '-lquactlize_ppu_execution', '-lquactlize_ppu_prefill', '-L' + str(sdk / 'lib'),
        *['-l' + name for name in LIBRARIES], '-ldl', '-o', str(output / 'libquactlize_ppu.so')])
    headers = {
        'quactlize/packing/api.h': 'quactlize/packing/api.h',
        'quactlize/dense/api.h': 'quactlize/dense/api.h',
        'quactlize/include/quactlize_ppu_config.h': 'quactlize_ppu_config.h',
    }
    for source, target in headers.items():
        path = output / 'include' / target
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, path)
    compiler.check_inputs()
    if any(sha(p) != digest for p, digest in public_sources.items()):
        raise ValueError('provider/dispatcher sources changed during build')
    payloads = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                if p.is_file() and not p.is_relative_to(work)}
    manifest = dict(schema='quactlize.ppu-library.v1', library='libquactlize_ppu.so',
        payloads=payloads, commands=commands, jit_source_contract=contract,
        sdk=compiler.sdk_identity, formats=[8, 10, 11, 12, 13, 14],
        source_hashes={str(p.relative_to(ROOT)): digest for p, digest in public_sources.items()},
        execution_receipt=execution,
        final_selection=dict(path='quactlize-runtime/final-selection.json', capabilities=capabilities),
        prefill_confirmed=dict(path='quactlize-runtime/prefill-confirmed.json',
            header_sha256=sha(PREFILL_HEADER), **receipt_fields()),
        default_dense_compute='F16_POC', device_admission='PENDING',
        dense_helpers=dict(policy_sha256=sha(helper_policy),runtime_receipt='quactlize-runtime/prefill-runtime.json',
                           full_table_sha256=sha(FULL_HELPERS_HEADER),full_evidence_sha256=sha(FULL_HELPERS),
                           scope='ZW810_FULL_350_TYPED_POINTS; TC_SF_UNCHANGED; NO_GEMM_ROUTE_OR_DTYPE_CHANGE'),
        performance_admission='NOT_MEASURED', seconds=time.monotonic() - start)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output)
    print(f'QUACTLIZE_PPU_BUILD COMPLETE library={output / "libquactlize_ppu.so"} device=PENDING', flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK') or '/usr/local/PPU_SDK'))
    parser.add_argument('--output', type=Path, required=True, help='fresh package directory')
    parser.add_argument('--jobs', type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument('--verify', action='store_true', help='verify an existing package without compilation')
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
        print('QUACTLIZE_PPU_PACKAGE VERIFIED device_admission=UNCHANGED')
    else:
        build(args.sdk, args.output, args.jobs)


if __name__ == '__main__':
    main()
