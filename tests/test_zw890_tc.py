from pathlib import Path

import pytest

from quactlize.runtime.compiler import Compiler, validate_parent
from tools.ppu15.tc_plan import EXPERTS, MS, SPLITS, case_keys, parents, plan, row_counts, verdict
from tools.run_zw890_tc import module_commands, module_source

ROOT = Path(__file__).resolve().parents[1]


def test_exact_bounded_scope():
    p = plan()
    assert p['expected'] == 60
    assert len(parents()) == 5
    assert not p['production_enabled'] and not p['performance_admitted']
    assert 'GROUPED_BF16' in p['remaining'] and 'Q2_Q3_Q5_TC' in p['remaining']
    assert set(MS) == {1, 8, 9, 16, 65, 128} and SPLITS == (1, 2)


def test_supplement_only_adds_missing_formats_and_grouped_precision():
    p = plan('supplement')
    cohort = p['parents']
    dense = [x for x in cohort if x['route'].endswith('dense')]
    grouped = [x for x in cohort if x['route'].endswith('grouped')]
    assert p['expected'] == 204 and len(cohort) == 17
    assert len(dense) == 6 and len(grouped) == 11
    assert set(x['qtype'] for x in dense) == {10, 11, 13}
    assert set(x['qtype'] for x in grouped) == {8, 10, 11, 12, 13, 14}
    assert not ({x['symbol'] for x in parents()} & {x['symbol'] for x in cohort})
    assert all(x['route'] == 'sf-grouped' for x in grouped if x['qtype'] == 8)
    assert 'NOT logical MoE tokens' in p['m_semantics']
    assert 'INDEXED_ROUTER_FFN' in p['remaining']
    assert not p['production_enabled'] and not p['performance_admitted']


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


@pytest.mark.parametrize('parent', parents('supplement'), ids=lambda p: p['symbol'])
def test_supplement_module_generation(parent, tmp_path):
    validate_parent(parent)
    source = module_source(parent, 'host-test')
    assert 'cutlass::arch::PPU0015' in source
    assert 'AtomShape_MNK{}) == 16' in source
    grouped = parent['route'].endswith('grouped')
    assert ('#define QK_USE_BF16_COMPUTE 1' in source) == grouped
    if grouped:
        persistent = parent['route'].startswith('fq')
        assert parent['persistent'] == (1 if persistent else -1)
        assert f'Group<{str(persistent).lower()}>::Mainloop' in source
    else:
        assert parent['persistent'] == -1 and 'Dense::Mainloop' in source
    compile_cmd, _ = module_commands(Path('/example/sdk'), parent, tmp_path)
    fmt = {8: 0, 10: 2, 11: 3, 12: 0, 13: 1, 14: 4}[parent['qtype']]
    assert f'-DPPU_PACKED_FORMAT={fmt}' in compile_cmd


@pytest.mark.parametrize('m', MS)
def test_grouped_replay_changes_device_routes_without_changing_capacity(m):
    counts = [row_counts(m, replay) for replay in range(3)]
    assert all(len(c) == EXPERTS and sum(c) == m and min(c) >= 0 for c in counts)
    assert counts[0] == (m, 0, 0) and counts[2] == (0, 0, m)
    assert counts[1] != counts[2]
    assert max(counts[1]) - min(counts[1]) <= 1


def test_grouped_worker_uses_explicit_bf16_abi_and_live_device_offsets():
    text = (ROOT/'tools/ppu15/tc_check.py').read_text()
    assert 'bind_compute(module.lib, grouped=True)' in text
    assert 'version=4, size=C.sizeof(GroupedMetadataCall), compute_type=1' in text
    assert 'GroupedDeviceCall(2, C.sizeof(GroupedDeviceCall), call, m, 0)' in text
    assert 'offsets.write(bounds)' in text and 'source *= 2**20' in text
    assert 'matmul_reference(source[begin:end], gold[expert]' in text
    assert 'metadata_type = 0 if parent' in text
    assert "device['compute_units'] if persistent else 0" in text


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
