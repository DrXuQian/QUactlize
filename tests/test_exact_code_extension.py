"""Structure/ownership tests; these do not claim PPU numerical or timing coverage."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'quactlize/include/actlize_extensions/cutlass/gemm/collective'


def test_exact_code_geometry(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('host C++17 compiler unavailable')
    binary = tmp_path / 'geometry'
    subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I'+str(ROOT/'quactlize/include'),
                    str(ROOT/'tests/exact_code_geometry_host.cpp'), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_exact_code_collectives_do_not_depend_on_execution_abi():
    files = list(EXT.glob('ppu_mma_exact_code*.hpp')) + list((EXT/'detail').glob('ppu_exact_code*.hpp'))
    assert len(files) == 4
    for path in files:
        source = path.read_text()
        for forbidden in ('qkg_call_v1', 'QKG_', 'simt::', '#include "simt',
                          '#include "api.h"', '<<<', 'hggcMalloc'):
            assert forbidden not in source, (path, forbidden)
        for line in source.splitlines():
            if line.startswith('#include'):
                assert 'execution/' not in line, path
    reader = (EXT/'detail/ppu_exact_code_reader.hpp').read_text()
    assert 'MmaOperation<Compute>::fma(' in reader
    assert 'ppu.mma.sync' not in reader  # use the pinned actlize atom, not another asm copy


@pytest.mark.parametrize('header', ['mma_reader.cuh', 'mma_staged.cuh'])
def test_runtime_adapters_have_no_duplicate_math(header):
    source = (ROOT/'quactlize/execution'/header).read_text()
    assert 'ExactCodeCollectivePPU<' in source
    for old_math in ('code_pair(', 'fmaf(', 'cp_async16(', 'static void load_win('):
        assert old_math not in source


def test_existing_tc_umbrella_does_not_implicitly_enable_exact_code():
    source = (ROOT/'quactlize/include/quactlize_actlize.hpp').read_text()
    assert 'ppu_mma_exact_code' not in source


@pytest.mark.parametrize('invalid', [
    'Geometry<10,4,4,1,1>',   # Q2 not implemented by this reader
    'Geometry<12,3,4,1,1>',   # unsupported vector width
    'Geometry<12,4,0,1,1>',   # empty CTA
    'Geometry<12,4,4,3,1>',   # unsupported K-warp geometry
    'Geometry<12,4,8,8,1>',   # more than 1024 threads
    'Rot<3>',                # template capacity, not the runtime active row count
    'StageLayout<14,128,1>',  # would split a paired Q6 metadata unit
])
def test_invalid_geometry_fails_compilation(tmp_path, invalid):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('host C++17 compiler unavailable')
    source = tmp_path/'negative.cpp'
    source.write_text(
        '#include "actlize_extensions/cutlass/gemm/collective/detail/ppu_exact_code_geometry.hpp"\n'
        'using namespace cutlass::gemm::collective::exact_code_ppu_detail;\n'
        f'static_assert(sizeof({invalid})>0);\n')
    result = subprocess.run([compiler, '-std=c++17', '-fsyntax-only',
                             '-I'+str(ROOT/'quactlize/include'), str(source)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert 'static assertion failed' in result.stderr
