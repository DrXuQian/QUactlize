"""Host-only target/recipe isolation. Device accuracy has a separate runner."""
import json
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
