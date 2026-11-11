#!/usr/bin/env python3
"""Build or reuse a native PPU1.5 public package and run its bounded numerical gate."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_zw890_tc import source_identity
from quactlize.runtime.compiler import sha

EXPECTED = 174


def run_phase(argv, stage, output, env):
    path = output/(stage+'.log')
    started = time.monotonic()
    print(f'ZW890_LIBRARY_START stage={stage} log={path}', flush=True)
    with path.open('x') as log:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                rc = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'ZW890_LIBRARY_WAIT stage={stage} minutes={(time.monotonic()-started)/60:.1f} log={path}', flush=True)
    if rc:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-80:]), flush=True)
        raise RuntimeError(f'{stage} rc={rc}; see {path}')


def review(output):
    archive = output.with_name(output.name+'.review.tgz')
    paths = [output/p for p in ('summary.json', 'source.json', 'build.log', 'accuracy.log',
                              'accuracy.xml', 'package/manifest.json', 'package-manifest.json')]
    # Include compilation diagnostics but no binaries, caches or JIT headers.
    paths += sorted((output/'package.build').rglob('*.log'))
    with tarfile.open(archive, 'x:gz') as tar:
        for path in paths:
            if not path.is_file() or path.is_symlink(): continue
            name = str(path.relative_to(output))
            if path.stat().st_size > 1024*1024 and path.suffix == '.log':
                with path.open('rb') as f:
                    f.seek(-1024*1024, 2); data = f.read()
                info = tarfile.TarInfo(name+'.tail'); info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else: tar.add(path, arcname=name, recursive=False)
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK') or '/usr/local/PPU_SDK'))
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--package', type=Path, help='test an existing package without rebuilding or modifying it')
    a = p.parse_args()
    if a.device < 0 or a.jobs < 1: p.error('nonnegative device and positive jobs required')
    sdk = a.sdk.resolve()
    if not (sdk/'bin/hgcc').is_file(): p.error('SDK must contain bin/hgcc')
    if a.package and not (a.package/'manifest.json').is_file():
        p.error('--package must contain the built public manifest.json')
    if a.output:
        output = a.output.resolve()
        try: output.mkdir(parents=True, exist_ok=False)
        except FileExistsError: p.error('--output must be a new directory')
    else: output = Path(tempfile.mkdtemp(prefix='qz-zw890-library-'))
    package = a.package.resolve() if a.package else output/'package'
    env = dict(os.environ, PPU_SDK=str(sdk), CUDA_VISIBLE_DEVICES=str(a.device),
               QUACTLIZE_CI_PACKAGE=str(package), QUACTLIZE_CI_SUITE='accuracy',
               QUACTLIZE_PPU_PYTHON=sys.executable,
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONOPTIMIZE'): env.pop(key, None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env['LD_LIBRARY_PATH'] = str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH', '')
    summary = dict(status='FAIL', expected=EXPECTED, arch='ppu_15', requested_device=a.device,
                   sdk=str(sdk), package=str(package), package_reused=bool(a.package), performance_admitted=False,
                   production_default_changed=False, scope='PUBLIC_PACKAGE_NUMERICS_NOT_MODEL_OR_PERFORMANCE')
    start = time.monotonic()
    try:
        identity = source_identity()
        for name in ('tools/run_zw890_library.py', 'tools/build_ppu_library.py',
                     'tools/build_kpack_pack.py', 'tools/build_kpack_execution.py',
                     'tools/build_kpack_prefill_runtime.py', 'tests/ppu/test_zw890_library.py',
                     'tests/ppu/test_public_library.py', 'tests/ppu/test_moe_library.py',
                     'tests/ppu/test_ffn_library.py', 'tests/ppu/support.py', 'tests/ppu/conftest.py', 'tests/conftest.py',
                     'ci/support/fixture.py', 'ci/support/graph.py'):
            identity['sources'][name] = sha(ROOT/name)
        (output/'source.json').write_text(json.dumps(identity, indent=2)+'\n')
        if a.package:
            print(f'ZW890_LIBRARY_REUSE package={package} build=SKIPPED verification=PUBLIC_FIXTURE', flush=True)
        else:
            run_phase([sys.executable, str(ROOT/'tools/build_ppu_library.py'), '--arch', 'ppu_15',
                       '--sdk', str(sdk), '--output', str(package), '--jobs', str(a.jobs)], 'build', output, env)
        # Preserve the receipt even when the package lives outside this run.
        # The normal public fixture still verifies every payload, exports and
        # SDK identity; only the unrelated legacy Torch _C guard is bypassed.
        manifest = (package/'manifest.json').read_bytes()
        (output/'package-manifest.json').write_bytes(manifest)
        if json.loads(manifest).get('arch') != 'ppu_15':
            raise ValueError('this gate requires a native ppu_15 public package')
        run_phase([sys.executable, '-m', 'pytest', '-q', '-s', '--tb=short',
                   'tests/ppu/test_zw890_library.py', '--junitxml='+str(output/'accuracy.xml')], 'accuracy', output, env)
        cases = list(ET.parse(output/'accuracy.xml').getroot().iter('testcase'))
        passed = sum(not any(c.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for c in cases)
        summary['passed'] = passed
        if len(cases) != passed or passed != EXPECTED:
            raise ValueError(f'accuracy coverage differs: passed={passed} cases={len(cases)} expected={EXPECTED}')
        summary['status'] = 'PASS'
        return 0
    except Exception as exc:
        summary['error'] = str(exc)
        print('ZW890_LIBRARY_FAIL '+str(exc), flush=True)
        return 1
    finally:
        summary['seconds'] = time.monotonic()-start
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        archive = review(output)
        print('ZW890_LIBRARY_DONE '+json.dumps(dict(summary, results=str(archive))), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
