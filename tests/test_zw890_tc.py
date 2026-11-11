from pathlib import Path

import pytest

from quactlize.runtime.compiler import Compiler, validate_parent
from tools.ppu15.tc_plan import MS, SPLITS, case_keys, parents, plan, verdict
from tools.run_zw890_tc import module_commands

ROOT = Path(__file__).resolve().parents[1]


def test_exact_bounded_scope():
    p = plan()
    assert p['expected'] == 60
    assert len(parents()) == 5
    assert not p['production_enabled'] and not p['performance_admitted']
    assert 'GROUPED_BF16' in p['remaining'] and 'Q2_Q3_Q5_TC' in p['remaining']
    assert set(MS) == {1, 8, 9, 16, 65, 128} and SPLITS == (1, 2)


@pytest.mark.parametrize('parent', parents(), ids=lambda p: p['symbol'])
def test_parent_uses_existing_generator_and_explicit_target(parent, tmp_path):
    validate_parent(parent)
    source = Compiler.source(None, parent, 'host-test')
    assert '#include "module.cuh"' in source
    assert '#define QK_TM 64' in source and '#define QK_AP 0' in source
    compile_cmd, link_cmd = module_commands(Path('/example/sdk'), parent, tmp_path)
    assert [x for x in compile_cmd if x.startswith('-arch=')] == ['-arch=ppu_15']
    assert '-DQUACTLIZE_PPU_ARCH=150' in compile_cmd
    assert '-DPPU_PACKED_SCALE='+str(int(parent['route'].startswith('fq'))) in compile_cmd
    assert '-Wl,-Bsymbolic' in link_cmd


def test_incomplete_duplicate_wrong_shape_or_failed_rows_do_not_pass():
    parent = parents()[0]
    rows = [dict(parent=p, m=m, split=s, status='PASS') for p, m, s in sorted(case_keys(parent))]
    assert verdict(parent, rows, 0)['status'] == 'PASS'
    for bad, rc in ((rows[:-1], 0), (rows[:-1]+rows[:1], 0), (rows, 1),
                    ([dict(rows[0], m=7)]+rows[1:], 0),
                    ([dict(rows[0], status='FAIL')]+rows[1:], 0)):
        assert verdict(parent, bad, rc)['status'] == 'FAIL'


def test_target_is_explicit_and_default_remains_ppu10():
    text = (ROOT/'quactlize/include/ppu_target.hpp').read_text()
    assert '#define QUACTLIZE_PPU_ARCH 100' in text
    assert '__HGGC_ARCH__ != QUACTLIZE_PPU_ARCH' in text
    assert 'using Arch = cutlass::arch::PPU0015' in text
    # The product compiler/device policy must not be enabled by this probe.
    compiler = (ROOT/'quactlize/runtime/compiler.py').read_text()
    assert '"-arch=ppu_10"' in compiler
    assert 'QUACTLIZE_PPU_ARCH' not in compiler


def test_native_writer_reader_and_a_swap_are_changed_together():
    builder = (ROOT/'quactlize/include/actlize_extensions/cutlass/gemm/collective/builders/quactlize_mma_builder.inl').read_text()
    assert 'PPU0015_AIU_LOAD<cute::C<bits_per_aiu>' in builder
    assert 'PPU0015_TSM_LD_SWZL<TransportElement' in builder
    assert 'PPU0015, Element, false, M, K, false>' in builder
    assert '(HasKPack && ScheduledAPackRows == 0)' in builder
    assert '!Native15 || DeliveryN >= 32' in builder
    check = (ROOT/'tools/ppu15/tc_check.py').read_text()
    assert 'np.testing.assert_array_equal(got, want' in check
    assert 'quactlize_kpack_prepare_dynamic_v1' in check
    assert 'quactlize_kpack_run_dynamic_v1' in check
    assert 'Replay(sdk, stream, launch, 1)' in check
