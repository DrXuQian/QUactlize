#!/usr/bin/env python3
"""Run with python -I, outside the checkout, after pip install --no-deps WHEEL.

This script deliberately never adds the source repository to sys.path.
Host queries are not PPU numerical or performance qualification.
"""
import argparse
import ctypes as C
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def host(target):
    import quactlize
    from quactlize.compat.native import Library, Problem, Sizes, Buffers, problem
    from quactlize.compat.serving_native import Library as ServingLibrary
    from quactlize.packing.native import Arrangement, Sizes as PackedSizes
    from quactlize._library import PUBLIC_EXPORTS
    # Test discovery before Torch import and before loading any device library.
    if 'torch' in sys.modules:
        raise RuntimeError('pure C ABI imports unexpectedly imported Torch')
    original_load, original_process = C.CDLL, subprocess.Popen
    def forbidden(*args, **kwargs):
        raise RuntimeError('import/discovery attempted dlopen, device probing or compilation')
    C.CDLL, subprocess.Popen = forbidden, forbidden
    try:
        info = quactlize.get_library_info(target)
        path = quactlize.get_library_path(target)
        json.dumps(info)
    finally:
        C.CDLL, subprocess.Popen = original_load, original_process
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Torch import initialized the device')
    original_init = torch.cuda._lazy_init
    torch.cuda._lazy_init = forbidden
    C.CDLL = forbidden
    try:
        from quactlize.serving import Serving
        assert Serving is not None
    finally:
        torch.cuda._lazy_init, C.CDLL = original_init, original_load
    if torch.cuda.is_initialized():
        raise RuntimeError('serving import initialized the device')
    files = {}
    for name in ('quactlize', 'quactlize.compat.native', 'quactlize.compat.serving_native',
                 'quactlize.packing.native', 'quactlize.runtime.native', 'quactlize.serving'):
        module = importlib.import_module(name)
        file = Path(module.__file__).resolve()
        if not file.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError(f'{name} is not installed in this environment: {file}')
        files[name] = str(file)
    library = Library(target=target)
    packed = Library(target=target, packed_route=0)
    ServingLibrary(target=target)  # Binds helpers only, never calls prepare.
    for name in PUBLIC_EXPORTS:
        getattr(library.lib, name)
    canonical = library.lib.quactlize_ppu_kpack_canonical_arrangement_v1
    canonical.argtypes, canonical.restype = [C.c_int, C.POINTER(Arrangement)], C.c_int
    sizes = library.lib.quactlize_ppu_kpack_sizes_for_arrangement_v1
    sizes.argtypes, sizes.restype = [C.c_int] * 4 + [C.POINTER(Arrangement), C.POINTER(PackedSizes)], C.c_int
    records = []
    for q, n, k in ((7, 2560, 640), (8, 2560, 640), (8, 1280, 2560),
                    (12, 1280, 2560), (13, 1280, 2560)):
        arr, planes = Arrangement(), PackedSizes()
        library.check(canonical(q, C.byref(arr)))
        library.check(sizes(n, k, 512, q, C.byref(arr), C.byref(planes)))
        assert planes.raw_bytes == planes.low_bytes + planes.high_bytes + planes.units_bytes
        consumer = packed if q == 7 else library
        for m in (1, 8, 9, 128):
            for dtype in (0, 1):
                p = problem(q, m, n, k, experts=512, topk=10,
                            channels=10 if k == 640 else 1, dtype=dtype, layout=1)
                query = Sizes()
                consumer.check(consumer.query(C.byref(p), C.byref(arr), C.byref(query)))
                # Canonical consumers do not retain the raw GGUF allocation.
                assert query.raw_bytes == 0
                for name in ('low_bytes', 'high_bytes', 'units_bytes'):
                    assert getattr(query, name) == getattr(planes, name), (q, m, dtype, name)
                if q == 7:
                    assert query.route == (2 if m <= 8 else 3) and query.expanded_weight_bytes == 0
                for field, _ in arr._fields_:
                    wrong = Arrangement.from_buffer_copy(arr)
                    setattr(wrong, field, getattr(wrong, field) ^ 1)
                    assert consumer.query(C.byref(p), C.byref(wrong), C.byref(Sizes())) != 0, field
                records.append(dict(q=q, m=m, n=n, k=k, dtype=dtype,
                                    route=query.route, workspace=query.workspace_bytes))
    print('QUACTLIZE_INSTALLED_HOST_PASS ' + json.dumps(dict(
        library=str(path), version=quactlize.__version__, source_revision=info['source_revision'],
        target=info['target'], files=files, torch=torch.__version__,
        abi_sizes=dict(problem=C.sizeof(Problem), sizes=C.sizeof(Sizes), buffers=C.sizeof(Buffers),
                       arrangement=C.sizeof(Arrangement)), queries=records,
        scope='INSTALLED_WHEEL_HOST_ABI_NOT_PPU_NUMERICS_OR_PERFORMANCE')), flush=True)
    return path


def pytest_gate(tests, *, cwd, env, pythonpath=None, minimum_tests=1):
    """Pytest exit zero can mean all skipped; only executed cases qualify."""
    with tempfile.TemporaryDirectory(prefix='quactlize-installed-result-') as temporary:
        report = Path(temporary) / 'junit.xml'
        environment = dict(env)
        environment.pop('PYTEST_ADDOPTS', None)
        command = [sys.executable, '-I', '-m', 'pytest', '-q', '-s', '-o', 'addopts=',
                   '--junitxml=' + str(report)]
        if pythonpath is not None:
            command.extend(('-o', 'pythonpath=' + str(pythonpath)))
        subprocess.run([*command, *tests], cwd=cwd, env=environment, check=True)
        root = ET.parse(report).getroot()
        suites = [root] if root.tag == 'testsuite' else root.findall('.//testsuite')
        counts = {name: sum(int(s.get(name, 0)) for s in suites)
                  for name in ('tests', 'failures', 'errors', 'skipped')}
        if counts['tests'] < minimum_tests or any(counts[k] for k in ('failures', 'errors', 'skipped')):
            raise RuntimeError(f'device gate did not execute all required cases: {counts}; '
                               f'require at least {minimum_tests} tests and no skips')
        return counts


def device_tests(source, path, flare):
    if not source:
        raise ValueError('--device-tests requires --test-source (test/oracle files only)')
    with tempfile.TemporaryDirectory(prefix='quactlize-installed-tests-') as temporary:
        root = Path(temporary)
        # Copy only tests/oracles, NEVER the production Python package or a DSO.
        for name in ('tests/ppu', 'tests/serving', 'ci/support', 'reference'):
            shutil.copytree(source / name, root / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.o'))
        (root / 'tests/__init__.py').touch()
        (root / 'ci/__init__.py').touch()
        env = {k: v for k, v in os.environ.items() if k not in (
            'PYTHONPATH', 'QUACTLIZE_CI_PACKAGE', 'QUACTLIZE_SERVING_LIBRARY')}
        # Secondary-owner tests must reopen the same verified installed DSO.
        env['QUACTLIZE_SERVING_LIBRARY'] = str(path)
        tests = ('tests/ppu/test_packed_native.py', 'tests/ppu/test_compat_library.py',
                 'tests/serving/test_serving_device.py')
        results = {'operators': pytest_gate(tests, cwd=root, env=env, pythonpath=root)}
    if flare:
        env = dict(os.environ, FLARE_TEST_PPU_MATMUL_LIBRARY=str(path), FLARE_RUN_CUDA_TESTS='1')
        env.pop('PYTHONPATH', None)
        results['flare'] = pytest_gate(('tests/preparation/test_gguf_packing_cuda.py',),
                                      cwd=flare, env=env, minimum_tests=10)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default=None)
    p.add_argument('--device-tests', action='store_true')
    p.add_argument('--test-source', type=Path)
    p.add_argument('--flare', type=Path, help='installed Flare checkout, feat/qwen38-gguf >= 3f2083b')
    p.add_argument('--deny-root', type=Path, action='append', default=[],
                   help='deny Python file access to the build/source directories during host qualification')
    args = p.parse_args()
    roots = [root.resolve() for root in args.deny_root]
    if roots:
        def audit(event, arguments):
            if event == 'open' and isinstance(arguments[0], str):
                path = Path(arguments[0]).resolve()
                if any(path.is_relative_to(root) for root in roots):
                    raise RuntimeError('unexpected access to hidden build/source directory: ' + str(path))
        sys.addaudithook(audit)
    path = host(args.target)
    if args.device_tests:
        counts = device_tests(args.test_source, path, args.flare)
        print('QUACTLIZE_INSTALLED_DEVICE_PASS performance=NOT_MEASURED ' + json.dumps(counts))
    else:
        print('QUACTLIZE_INSTALLED_DEVICE_NOT_RUN')


if __name__ == '__main__':
    main()
