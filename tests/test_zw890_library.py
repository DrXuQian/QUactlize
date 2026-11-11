"""Host-only target/recipe isolation. Device accuracy has a separate runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from quactlize.runtime.compiler import FLAGS, Compiler, LIBRARIES, source_contract
from quactlize.runtime.target import definitions, flags_for_arch, selected_arch

ROOT = Path(__file__).resolve().parents[1]


def test_arch_is_explicit_and_default_flags_are_unchanged():
    assert selected_arch() == 'ppu_10'
    assert flags_for_arch(FLAGS, 'ppu_10') == FLAGS
    native = flags_for_arch(FLAGS, 'ppu_15')
    assert [x for x in native if x.startswith('-arch=')] == ['-arch=ppu_15']
    assert '-DQUACTLIZE_PPU_ARCH=150' in native
    with pytest.raises(ValueError): selected_arch('ppu_17')


def test_typed_input_publisher_matches_native_layout(tmp_path):
    sdk = Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK'))
    exe = tmp_path/'typed-input'
    subprocess.run(['g++', '-std=c++17', '-O2', f'-I{ROOT}/quactlize/include',
        f'-I{ROOT}/third_party/actlize/include', f'-I{sdk}/include',
        f'-I{sdk}/targets/x86_64-linux/include', ROOT/'tests/zw890_typed_input_host.cpp',
        '-o', exe], check=True)
    assert 'ZW890_TYPED_INPUT_HOST PASS' in subprocess.check_output([exe], text=True)


@pytest.mark.parametrize('arch', ['ppu_10', 'ppu_15'])
def test_compiled_device_and_recipe_contract(tmp_path, arch):
    exe = tmp_path/'target'
    subprocess.run(['g++', '-std=c++17', '-O2', f'-I{ROOT}', *definitions(arch),
                    ROOT/'tests/zw890_target_host.cpp', '-o', exe], check=True)
    assert subprocess.check_output([exe], text=True).strip() == 'TARGET_CONTRACT_PASS'


def test_jit_cache_contract_separates_architectures(tmp_path):
    sdk = tmp_path/'sdk'
    for name in ('bin/hgcc', 'bin/hgobjdump', *('lib/lib'+n+'.so' for n in LIBRARIES)):
        path = sdk/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'HOST IDENTITY FIXTURE ONLY')
    old, native = [Compiler(sdk, tmp_path/'cache', arch=a) for a in ('ppu_10', 'ppu_15')]
    assert source_contract(old.identity) != source_contract(native.identity)
    assert old.identity['arch'] == 'ppu_10' and native.identity['arch'] == 'ppu_15'
    assert old.flags == FLAGS


def test_package_cannot_lose_its_native_jit_target(tmp_path):
    from tools.build_ppu_library import PUBLIC_LIBRARY, verify
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        schema='quactlize.ppu-library.v1', library=PUBLIC_LIBRARY,
        payloads={PUBLIC_LIBRARY:'unused'}, arch='ppu_15')))
    with pytest.raises(ValueError, match='JIT targets differ'):
        verify(tmp_path)


def test_public_accuracy_collection_is_complete_without_a_device():
    from tools.run_zw890_library import EXPECTED
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q',
                             'tests/ppu/test_zw890_library.py'], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('tests/ppu/test_zw890_library.py::') == EXPECTED == 174


def test_typed_bf16_retry_keeps_exact_reported_scope():
    from tools.run_zw890_library import accuracy_nodes
    nodes = accuracy_nodes('typed-bf16')
    assert len(nodes) == len(set(nodes)) == 18
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', *nodes],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('tests/ppu/test_zw890_library.py::') == 18
    assert all('bf16-' in n and 'test_dense_auto' in n for n in nodes)


def test_review_keeps_diagnostics_without_binary_payloads(tmp_path):
    from tools.run_zw890_library import review
    output = tmp_path/'results'; output.mkdir()
    (output/'summary.json').write_text('{}')
    work = output/'package.build'; work.mkdir()
    (work/'kernel.so').write_bytes(b'NOT_FOR_UPLOAD')
    (work/'compile.log').write_bytes(b'x'*(1024*1024+1))
    archive = review(output)
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == ['package.build/compile.log.tail', 'summary.json']
        assert tar.getmember('package.build/compile.log.tail').size == 1024*1024


@pytest.mark.parametrize('reuse', [False, True])
def test_runner_public_suite_ignores_only_unrelated_stale_extension(tmp_path, monkeypatch, reuse):
    from tools import run_zw890_library as runner
    # A real pytest session with the real legacy guard, but no device/library
    # calls. The deliberately stale extension must never be loaded.
    checkout = tmp_path/'guard-checkout'
    testdir = checkout/'tests'; testdir.mkdir(parents=True)
    (testdir/'conftest.py').write_bytes((ROOT/'tests/conftest.py').read_bytes())
    testfile = testdir/'test_guard.py'
    testfile.write_text('def test_host_only():\n    assert True\n')
    csrc = checkout/'quactlize/csrc'; csrc.mkdir(parents=True)
    stale = csrc.parent/'_C.host-fixture.so'; stale.write_bytes(b'NEVER_LOAD')
    source = csrc/'ppu_backend.cpp'; source.write_text('// host fixture')
    os.utime(stale, (100, 100)); os.utime(source, (200, 200))
    sdk = tmp_path/'sdk'; (sdk/'bin').mkdir(parents=True)
    (sdk/'bin/hgcc').write_bytes(b'NO_COMPILATION')
    monkeypatch.setattr(runner, 'source_identity', lambda: {'sources': {}})
    monkeypatch.setattr(runner, 'EXPECTED', 1)
    monkeypatch.delenv('QUACTLIZE_CI_SUITE', raising=False)
    phases = []

    def phase(argv, stage, output, env):
        phases.append(stage)
        if stage == 'typed-input': return # Device proof is not executed by this host fixture.
        if stage == 'build':
            package = output/'package'; package.mkdir()
            (package/'manifest.json').write_text('{"arch":"ppu_15"}')
            return
        # Outside the public-suite entrypoint the same stale extension must
        # still be rejected. Do not remove or globally disable its guard.
        command = [sys.executable, '-m', 'pytest', '-q', str(testfile)]
        legacy = dict(env); legacy.pop('QUACTLIZE_CI_SUITE', None)
        result = subprocess.run(command, cwd=checkout, env=legacy, text=True, capture_output=True)
        assert result.returncode == 4
        assert 'THE BUILT EXTENSION IS OLDER' in result.stderr
        result = subprocess.run(command+['--junitxml='+str(output/'accuracy.xml')],
                                cwd=checkout, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stdout+result.stderr
        assert env['QUACTLIZE_CI_SUITE'] == 'accuracy'

    monkeypatch.setattr(runner, 'run_phase', phase)
    out = tmp_path/'result'
    argv = ['run_zw890_library.py', '--sdk', str(sdk), '--output', str(out)]
    package = tmp_path/'existing-package' if reuse else out/'package'
    if reuse:
        package.mkdir()
        (package/'manifest.json').write_text('{"arch":"ppu_15"}')
        argv += ['--package', str(package)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert runner.main() == 0
    assert phases == (['typed-input', 'accuracy'] if reuse else ['typed-input', 'build', 'accuracy'])
    assert (out/'package-manifest.json').read_bytes() == (package/'manifest.json').read_bytes()
    assert sorted(p.name for p in package.iterdir()) == ['manifest.json']
    summary = json.loads((out/'summary.json').read_text())
    assert summary['package_reused'] == reuse and summary['package'] == str(package)
    with tarfile.open(out.with_name(out.name+'.review.tgz')) as tar:
        assert 'package-manifest.json' in tar.getnames()


def test_reuse_missing_package_does_not_build_or_create_results(tmp_path, monkeypatch):
    from tools import run_zw890_library as runner
    sdk = tmp_path/'sdk'; (sdk/'bin').mkdir(parents=True)
    (sdk/'bin/hgcc').write_bytes(b'NO_COMPILATION')
    out = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['run_zw890_library.py', '--sdk', str(sdk),
        '--output', str(out), '--package', str(tmp_path/'absent')])
    with pytest.raises(SystemExit) as exc: runner.main()
    assert exc.value.code == 2 and not out.exists()
