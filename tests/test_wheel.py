"""Packaging/loader tests use explicit host ELF fixtures, NOT PPU correctness evidence."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from quactlize import _library as discovery
from tools.wheel_support import source_files, wheel_metadata

ROOT = Path(__file__).resolve().parents[1]


def test_sdk_release_is_not_runtime_api_marker(tmp_path):
    from tools.wheel_support import sdk_release
    (tmp_path / 'VERSION.txt').write_text('ppu_sdk_detection_magic\nhggcrt_version:v3\n')
    (tmp_path / 'release.yaml').write_text('version: 2.2.1-184ee6\n')
    assert sdk_release(tmp_path) == '2.2.1-184ee6'


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'host-fixture'
    root.mkdir()
    c = tmp_path / 'fixture.c'
    c.write_text('\n'.join(f'int {name}(void) {{ return 0; }}' for name in discovery.PUBLIC_EXPORTS))
    for name in discovery.REQUIRED_PAYLOADS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.so'):
            subprocess.run(['cc', '-shared', '-fPIC', '-Wl,-soname,' + path.name,
                            str(c), '-o', str(path)], check=True)
        else:
            path.write_text('HOST PACKAGING FIXTURE ONLY\n')
    provenance = dict(version=discovery.__version__, source_revision='a' * 40,
                      source_hashes=source_files(ROOT), supported_targets=['ppu10'], default_target='ppu10',
                      device_arch='ppu_10', sdk_release='HOST FIXTURE; NOT FOR RELEASE')
    manifest = dict(schema='quactlize.ppu-library.v1', library=discovery.PUBLIC_LIBRARY,
                    build_info=provenance, sdk='HOST FIXTURE',
                    payloads={name: discovery.file_sha256(root / name) for name in discovery.REQUIRED_PAYLOADS})
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


@pytest.fixture
def installed(bundle, monkeypatch):
    native = bundle.parent / 'quactlize/_native'
    shutil.copytree(bundle, native)
    for name in source_files(ROOT):
        if name.startswith('quactlize/') and name.endswith('.py'):
            destination = native.parent.parent / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
    (native / 'wheel.json').write_text(json.dumps(wheel_metadata(bundle, ROOT)))
    monkeypatch.setattr(discovery, '__file__', str(native.parent / '_library.py'))
    return native


def test_discovery_is_file_only(installed, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('discovery loaded a DSO, probed a device or invoked a compiler')
    monkeypatch.setattr(ctypes, 'CDLL', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    path = discovery.get_library_path()
    assert isinstance(path, Path) and path == installed / discovery.PUBLIC_LIBRARY
    info = discovery.get_library_info('ppu10')
    assert info['source_revision'] == 'a' * 40
    json.dumps(info)
    with pytest.raises(discovery.LibraryError, match='ppu15.*supports'):
        discovery.get_library_path('ppu15')


@pytest.mark.parametrize('explicit', [True, False])
@pytest.mark.parametrize('compatible', [True, False])
def test_device_package_sdk_check_matches_wheel_contract(installed, tmp_path, monkeypatch, explicit, compatible):
    from tests.ppu import conftest as gate
    expected = discovery.get_library_info()['sdk_requirements']['build_sdk']
    current = expected if compatible else 'different SDK release'
    sdk = tmp_path / 'runtime-sdk'
    sdk.mkdir()
    (sdk / 'release.yaml').write_text('version: ' + current + '\n')
    monkeypatch.setenv('PPU_SDK', str(sdk))
    if explicit:
        monkeypatch.setenv('QUACTLIZE_CI_PACKAGE', str(installed))
    else:
        monkeypatch.delenv('QUACTLIZE_CI_PACKAGE', raising=False)
    fingerprints = []
    def compiler_fingerprint(path):
        fingerprints.append(path)
        return 'same SDK release with different build-host compiler ELF'
    monkeypatch.setattr(gate, 'sdk_identity', compiler_fingerprint)
    if compatible:
        assert gate.package.__wrapped__() == installed
    else:
        with pytest.raises(pytest.fail.Exception) as error:
            gate.package.__wrapped__()
        assert expected in str(error.value) and current in str(error.value)
        assert str(sdk) in str(error.value)
    assert not fingerprints, 'public runtime compatibility must not hash build-host tools'


@pytest.mark.parametrize('explicit', [True, False])
def test_device_package_still_checks_payloads(installed, monkeypatch, explicit):
    from tests.ppu import conftest as gate
    if explicit:
        monkeypatch.setenv('QUACTLIZE_CI_PACKAGE', str(installed))
    else:
        monkeypatch.delenv('QUACTLIZE_CI_PACKAGE', raising=False)
    (installed / 'quactlize-runtime/jit/tools/kpack_jit.py').write_text('changed payload')
    with pytest.raises((ValueError, discovery.LibraryError), match='payload'):
        gate.package.__wrapped__()


@pytest.mark.parametrize('compatible', [True, False])
def test_legacy_package_without_release_keeps_exact_sdk_check(bundle, monkeypatch, compatible):
    from tests.ppu import conftest as gate
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['build_info'].pop('sdk_release')
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv('QUACTLIZE_CI_PACKAGE', str(bundle))
    monkeypatch.setattr(gate, 'sdk_identity', lambda sdk: manifest['sdk'] if compatible else 'different')
    if compatible:
        assert gate.package.__wrapped__() == bundle
    else:
        with pytest.raises(pytest.fail.Exception, match='SDK'):
            gate.package.__wrapped__()


@pytest.mark.parametrize('missing', ['libncp_quactlize.so', 'quactlize-runtime/libquactlize_ppu_execution.so',
                                    'quactlize-runtime/jit/tools/kpack_jit.py', 'manifest.json'])
def test_missing_resource_never_falls_back(installed, missing, monkeypatch):
    monkeypatch.setenv('QUACTLIZE_PPU_LIBRARY', '/some/other/libncp_quactlize.so')
    (installed / missing).unlink()
    with pytest.raises(discovery.LibraryError, match='Missing.*resource'):
        discovery.get_library_path('ppu10')


def test_changed_payload_and_manifest(installed):
    path = installed / 'quactlize-runtime/jit/tools/kpack_jit.py'
    path.write_text('changed')
    with pytest.raises(discovery.LibraryError, match='payload hash mismatch'):
        discovery.get_library_info()
    (installed / 'manifest.json').write_text('{}')
    with pytest.raises(discovery.LibraryError, match='manifest hash mismatch'):
        discovery.get_library_info()


def test_no_ambiguous_default(installed):
    path = installed / 'wheel.json'
    value = json.loads(path.read_text())
    value.pop('default_target')
    path.write_text(json.dumps(value))
    with pytest.raises(discovery.LibraryError, match='target None'):
        discovery.get_library_path()


def test_bindings_optional_path_and_old_abi_diagnostic(installed, tmp_path):
    from quactlize.compat.native import Library
    from quactlize.compat.serving_native import Library as Helpers
    assert Library(target='ppu10', packed_route=0).lib._name == str(installed / discovery.PUBLIC_LIBRARY)
    assert Helpers(target='ppu10').lib._name == str(installed / discovery.PUBLIC_LIBRARY)
    assert Library(installed / discovery.PUBLIC_LIBRARY).lib
    with pytest.raises(ValueError, match='either path or target'):
        Library(installed / discovery.PUBLIC_LIBRARY, target='ppu10')
    source, library = tmp_path / 'old.c', tmp_path / 'old.so'
    source.write_text('\n'.join(f'int {name}(void) {{return 0;}}' for name in discovery.PUBLIC_EXPORTS
                                if 'matmul_packed_' not in name))
    subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
    with pytest.raises(discovery.LibraryError, match='matmul_packed_query_v1.*old.so'):
        Library(library, packed_route=0)


@pytest.mark.parametrize('fault', ['source', 'missing-export', 'missing-payload', 'runtime-injection'])
def test_bundle_reuse_rejects_mixed_or_incomplete_inputs(bundle, fault):
    path = bundle / 'manifest.json'
    manifest = json.loads(path.read_text())
    if fault == 'source':
        manifest['build_info']['source_hashes']['quactlize/compat/native.py'] = '0' * 64
    elif fault == 'missing-payload':
        manifest['payloads'].pop('include/quactlize/compat/api.h')
    elif fault == 'runtime-injection':
        (bundle / 'quactlize-runtime/libcuda.so').write_bytes(b'must not ship')
    else:
        c = bundle.parent / 'old.c'
        c.write_text('int not_the_required_library(void) {return 0;}')
        library = bundle / discovery.PUBLIC_LIBRARY
        subprocess.run(['cc', '-shared', '-fPIC', str(c), '-o', str(library)], check=True)
        manifest['payloads'][discovery.PUBLIC_LIBRARY] = discovery.file_sha256(library)
    path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, discovery.LibraryError)):
        wheel_metadata(bundle, ROOT)


def test_real_noneditable_wheel_two_prefixes(bundle, tmp_path):
    """Actual wheel/install mechanics, but the fixture DSO is explicitly host-only."""
    work, dist = tmp_path / 'work', tmp_path / 'dist'
    log = tmp_path / 'wheel.log'
    with log.open('w') as out:
        subprocess.run([sys.executable, str(ROOT / 'tools/build_wheel.py'), '--bundle', str(bundle),
                        '--work', str(work), '--output', str(dist)], cwd=tmp_path, check=True, stdout=out,
                       stderr=subprocess.STDOUT)
    wheels = list(dist.glob('*.whl'))
    assert len(wheels) == 1 and wheels[0].name.endswith('-py3-none-linux_x86_64.whl')
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        for name in ('compat/native.py', 'compat/formats.py', 'compat/serving_native.py', 'packing/native.py',
                     'runtime/native.py', 'serving.py', '_native/manifest.json', '_native/wheel.json'):
            assert 'quactlize/' + name in names
        assert not any('/_C.' in name or name.endswith('/libcuda.so') for name in names)
    environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'QUACTLIZE_PPU_BUNDLE')}
    for index in range(2):
        prefix = tmp_path / f'environment-{index}'
        subprocess.run([sys.executable, '-m', 'venv', str(prefix)], check=True)
        python = prefix / 'bin/python'
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', str(wheels[0])], check=True,
                       cwd=tmp_path, env=environment, stdout=subprocess.PIPE)
        # Audit hooks also deny file access to both source and original bundle (without deleting them).
        code = '''
import sys, pathlib, ctypes, subprocess
forbidden = [pathlib.Path(p).resolve() for p in sys.argv[1:3]]
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if any(path == root or root in path.parents for root in forbidden):
            raise RuntimeError('source/bundle access: ' + str(path))
sys.addaudithook(audit)
import quactlize, quactlize.compat.native, quactlize.compat.serving_native
import quactlize.packing.native, quactlize.runtime.native
assert 'torch' not in sys.modules
original = ctypes.CDLL
def no_load(*a, **kw): raise RuntimeError('discovery must not dlopen')
ctypes.CDLL = no_load
path = quactlize.get_library_path(target='ppu10')
assert path.is_relative_to(pathlib.Path(sys.prefix))
for name, module in sys.modules.items():
    if name.startswith('quactlize') and getattr(module, '__file__', None):
        assert pathlib.Path(module.__file__).is_relative_to(pathlib.Path(sys.prefix)), name
ctypes.CDLL = original
quactlize.compat.native.Library(packed_route=0)
quactlize.compat.serving_native.Library()
print('HOST_FIXTURE_WHEEL_RELOCATION_PASS', sys.prefix)
'''
        subprocess.run([str(python), '-I', '-c', code, str(ROOT), str(bundle)], check=True,
                       cwd=tmp_path, env=environment)
