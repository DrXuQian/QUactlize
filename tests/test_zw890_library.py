"""Host-only target/recipe isolation. Device accuracy has a separate runner."""
import json
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

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


def test_native_package_cannot_be_labelled_as_ppu10_wheel(tmp_path):
    from tools.build_ppu_library import PUBLIC_LIBRARY, verify
    (tmp_path/'manifest.json').write_text(json.dumps(dict(
        schema='quactlize.ppu-library.v1', library=PUBLIC_LIBRARY,
        payloads={PUBLIC_LIBRARY:'unused'}, arch='ppu_15', build_info=dict(device_arch='ppu_10'))))
    with pytest.raises(ValueError, match='build metadata targets differ'):
        verify(tmp_path)


def test_public_accuracy_collection_is_complete_without_a_device():
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q',
                             'tests/ppu/test_zw890_library.py'], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('tests/ppu/test_zw890_library.py::') == 174


def test_native15_does_not_compile_flashnext_ppu10_kernels(tmp_path):
    from tools.build_ppu_library import compatibility_sources
    from quactlize.compat.native import Library, problem, Sizes, Buffers
    from quactlize.compat.serving_native import Library as ServingLibrary
    assert compatibility_sources('ppu_10') == (
        'quactlize/compat/runtime.cu', 'quactlize/compat/helpers.cu', 'quactlize/compat/serving.cu')
    source, = compatibility_sources('ppu_15')
    library = tmp_path/'unsupported.so'
    subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', *definitions('ppu_15'),
                    '-I'+str(ROOT/'quactlize/include'), ROOT/source, '-o', library], check=True)
    for route in (None, 0, 2, 3):
        native = Library(library, packed_route=route)
        p = problem(7, 1, 2560, 640, experts=512, topk=10, channels=10)
        sizes = Sizes()
        C.memset(C.byref(sizes), 0x5a, C.sizeof(sizes))
        assert native.query(C.byref(p), None, C.byref(sizes)) == 20
        assert bytes(sizes) == bytes(C.sizeof(sizes))
        handle = C.c_void_p(123)
        assert native.prepare(C.byref(p), None, C.byref(Buffers()), C.byref(handle)) == 20
        assert handle.value is None
        assert native.run(None, None) == 20
        assert b'ppu_10 package' in native.error()
    serving = ServingLibrary(library)
    assert serving.prepare(0, None) == 20
    assert b'ppu_10 package' in serving.error()


def test_fast_parent_index_cannot_cross_targets(tmp_path):
    from quactlize.runtime.jit_cache import ParentCache
    sdk = tmp_path/'sdk'
    for name in ('bin/hgcc', 'bin/hgobjdump', *('lib/lib'+n+'.so' for n in LIBRARIES)):
        path = sdk/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'HOST FILE IDENTITY ONLY')
    indices = [ParentCache(sdk, tmp_path/'cache', source_contract(
        Compiler(sdk, tmp_path/'cache', arch=a).identity)) for a in ('ppu_10', 'ppu_15')]
    assert indices[0].source != indices[1].source
    assert indices[0].environment != indices[1].environment
