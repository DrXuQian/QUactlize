"""Actual high producer maps: uniqueness, fanout, unchanged consumers/rounding."""
import ctypes as C
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import test_prefill_dense_r4 as r4

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def reader():
    r4.RoundFourHost.setUpClass()
    lib = r4.RoundFourHost.lib
    lib.dense_r5_high.argtypes = [C.c_uint] * 4
    lib.dense_r5_high.restype = C.c_uint
    yield lib
    r4.RoundFourHost.tearDownClass()


def address(q, col, kk, n):
    if q == 13:
        pn = (col & ~15) | (col & 7) | (((kk >> 7) & 1) << 3)
        kg = (kk // 256) * 16 | (((kk >> 6) & 1) << 3) | (kk & 7)
        return (kg * n + pn) * 2
    return ((kk // 64 * 8 + kk % 8) * n + col) * 2


@pytest.mark.parametrize('q', [13, 14])
@pytest.mark.parametrize('tk', [128, 256])
def test_unique_vectors_and_exact_single_writer_fanout(reader, q, tk):
    fans = 4 if q == 13 else 2
    old_loads = tk // 32 * 8 * 8
    loads = old_loads // fans
    for n in (256, 512, 5120):
        for n0 in (0, 64, n - 64):
            for k0 in range(0, 1024, tk):
                physical = []
                logical = Counter()
                for i in range(loads):
                    at = None
                    for fan in range(fans):
                        p = reader.dense_r5_high(q, tk, i, fan)
                        col, residue, g = p & 255, (p >> 8) & 255, p >> 16
                        assert col % 8 == 0 and col < 64 and residue < 8 and g < tk // 32
                        this = address(q, n0 + col, k0 + g * 32 + residue, n)
                        if at is None:
                            at = this
                            physical.append(at)
                        assert this == at, 'fanout refers to a different physical word'
                        for part in range(4):
                            logical[col // 2 + part, g * 8 + residue] += 1
                    assert at % 16 == 0 and at + 16 <= n * 1024 * (1 if q == 13 else 2) // 8
                assert len(set(physical)) == loads
                assert logical == Counter({(p, w): 1 for p in range(32) for w in range(tk // 4)})
                # Static 32x4B model, NOT measured PPU bank admission. Prove
                # each publication instruction has different word owners.
                for phase in (16, 32):
                    for first in range(0, loads, phase):
                        for fan in range(fans):
                            for part in range(4):
                                words = []
                                for i in range(first, min(first + phase, loads)):
                                    p = reader.dense_r5_high(q, tk, i, fan)
                                    col, residue, g = p & 255, (p >> 8) & 255, p >> 16
                                    index = reader.dense_r4_layout(64, tk, 1, col // 2 + part, g * 8 + residue)
                                    words.append(index // 2)
                                assert len(set(words)) == len(words)
                                assert len({w % 32 for w in words}) == len(words)
                # Per-instruction footprint, not one union across multiple loads.
                # Q6 supplies four full128B rows; Q5 two physical64B N cohorts
                # per128B sector. Neither path multiplies compulsory HBM bytes.
                for first in range(0, loads, 32):
                    active = physical[first:first + 32]
                    for sector in (32, 64, 128):
                        counts = Counter((a + j) // sector for a in active for j in range(16))
                        assert max(counts.values()) <= sector
                        assert sum(counts.values()) == len(active) * 16


@pytest.mark.parametrize('q', [13, 14])
@pytest.mark.parametrize('bf', [0, 1])
@pytest.mark.parametrize('n,k', [(256, 512), (256, 1024), (512, 512)])
def test_raw_gguf_bits_and_mapping_faults(reader, q, bf, n, k):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from reference.gguf_kpack import SPECS
    from tools.kpack_warmup_fixture import prepare_expert
    from dev.bf16_compute.fixture import bf16_bits
    spec = SPECS[q]
    rng = np.random.default_rng(q + n + k + bf)
    raw = rng.integers(0, 256, (n * k // 256, spec.raw_bytes), dtype='u1')
    for offset in (spec.d_offset, spec.dmin_offset):
        if offset >= 0:
            raw[:, offset:offset + 2] = rng.uniform(-.009, .009, len(raw)).astype('<f2').view('u1').reshape(-1, 2)
            raw[:8, offset:offset + 2] = np.array([0, 0x8000, 1, 0x8001, 0x3555, 0xb555, 0x0400, 0x8400], '<u2').view('u1').reshape(-1, 2)
    packed = prepare_expert(raw, q, n, k)
    gold = dequantize(raw.reshape(-1), GGMLQuantizationType(q)).reshape(n, k)
    want = bf16_bits(gold) if bf else gold.astype('<f2').view('<u2')
    for variant in (22, 23):
        images = []
        for v in (variant - 2, variant):
            out = np.full_like(want, 0x7fff)
            args = (q, bf, v, *[packed[x].ctypes.data for x in ('low', 'high', 'units')], out.ctypes.data, n, k)
            assert reader.dense_r4_host(*args, 0) == 0
            assert not np.any((out != want) & (((out | want) & 0x7fff) != 0))
            images.append(out.copy())
        assert np.array_equal(*images), 'pure delivery change must retain all output bits'
        assert reader.dense_r4_host(*args, 6) != 0, 'omitted unique load must fail coverage'
        assert reader.dense_r4_host(*args, 3) == 0
        assert np.any(out != want), 'wrong K32 high-half selection must fail numerics'


def test_production_selector_and_output_store_are_not_changed():
    source = (ROOT / 'quactlize/dequant/dense.cu').read_text()
    assert 'void dense_dequant_unique_high(' in source
    assert 'for(unsigned i=tid;i<H::Loads;i+=128)' in source
    policy = (ROOT / 'quactlize/prefill/dense_helpers.hpp').read_text()
    assert '22' not in policy and '23' not in policy
    consumer = source.split('for(unsigned i=tid;i<(TileN/2)*(TileK/8);i+=128)', 1)[1]
    assert '*reinterpret_cast<uint4*>(dst)={out0[0],out0[1],out0[2],out0[3]}' in consumer
    old, candidate = source.split('void dense_dequant_cooperative(', 1)[1].split('// Keep the admitted cooperative body', 1)
    candidate = candidate.split('template<KType T,bool BF16>\nint dense_launch', 1)[0]
    start = '    __syncthreads();'
    assert old[old.index(start):].strip() == candidate[candidate.index(start):].strip(), 'consumer/metadata rounding/store changed'


def test_bounded_round5_inventory_and_matched_profile_pairs():
    from dev.prefill_dense import round5
    from tools.run_prefill_dense_opt import workloads_for
    from types import SimpleNamespace
    rows = []
    assert workloads_for(SimpleNamespace(round_five=True)) == round5.workloads()
    assert [(w.qtype, w.n, w.k) for w in round5.workloads()] == [
        (14, 5120, 25600), (14, 1024, 5120), (13, 5120, 25600), (13, 1024, 5120)]
    for i in range(4):
        pool = round5.pool(i)
        assert len(pool) == 8
        assert {a['dequant_variant'] for a in pool} == {20, 21, 22, 23}
        assert {s for a in pool for s in round5.stages(i, a)} == {'dequant'}
        cells = [dict(arm=a, median_us=dict(dequant=30 - a['dequant_variant'])) for a in pool]
        assert len(round5.composed(i, cells)) == 4
        rows.append(dict(status='PASS', point=i, arms=cells))
    jobs = round5.profile_jobs(rows)
    assert len(jobs) == 16 and {j['stage'] for j in jobs} == {1}
    for i in range(4):
        cohort = [j for j in jobs if j['point'] == i]
        assert [j['role'] for j in cohort] == ['control', 'candidate', 'candidate', 'control']
        arms = {a['name']: a for a in round5.pool(i)}
        a, b = [arms[j['arm']] for j in cohort[:2]]
        assert b['dequant_variant'] == a['dequant_variant'] + 2
        assert {k: v for k, v in a.items() if k not in ('name', 'dequant_variant')} == {
            k: v for k, v in b.items() if k not in ('name', 'dequant_variant')}


@pytest.mark.parametrize('point', range(4))
@pytest.mark.parametrize('variant', (20, 21, 22, 23))
def test_profile_rejects_wrong_high_ownership_or_geometry(tmp_path, point, variant):
    import csv
    from dev.prefill_dense import round5
    w = round5.workloads()[point]
    tk = 128 if variant in (20, 22) else 256
    if variant < 22:
        symbol = f'dense_dequant_cooperative<(gguf_scale::KType){w.qtype-10}, true, 64u, {tk}u, true>'
    else:
        symbol = f'dense_dequant_unique_high<(gguf_scale::KType){w.qtype-10}, true, {tk}u>'
    row = {'ID': '0', 'Kernel Name': symbol, 'launch__grid_dim_x': str(w.n // 64),
           'launch__grid_dim_y': str(w.k // tk), 'launch__grid_dim_z': '1', 'launch__block_size': '128',
           'ce__cycles_elapsed.avg': '12345', 'ce__cycles_elapsed.avg.per_second': '1700000000'}
    path = tmp_path / 'raw.csv'
    def write():
        with path.open('w') as out:
            writer = csv.DictWriter(out, fieldnames=list(row), quoting=csv.QUOTE_ALL)
            writer.writeheader(); writer.writerow(row)
    job = dict(point=point, stage=1, arm=f'r5-bf16-dq{variant}')
    write()
    assert round5.check_profile(path, job)['variant'] == variant
    row['Kernel Name'] = symbol.replace('true', 'false', 1); write()
    with pytest.raises(ValueError):
        round5.check_profile(path, job)
    row['Kernel Name'] = symbol; row['launch__grid_dim_x'] = '1'; write()
    with pytest.raises(ValueError):
        round5.check_profile(path, job)


def test_round5_orchestration_keeps_components_separate(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from dev.prefill_dense import round2, round5
    samples = []
    class Bench:
        copies = 2
        receipt = dict(scope='HOST_HARNESS_ONLY')
        def __init__(self, *a, **kw): pass
        def handle(self, arm): return arm['name']
        def update(self, *a, **kw): pass
        def launch(self, *a, **kw): pass
        def check(self, *a, **kw): return dict(relative_l1_error=0)
        def check_weights(self, *a, **kw): return dict(bad=0)
        def samples(self, h, stage): samples.append((h, stage)); return [1., 2., 3.]
        def close(self): pass
    monkeypatch.setattr('dev.prefill_dense.bench.Bench', Bench)
    a = SimpleNamespace(arm=None, point=1, output=tmp_path, sdk=tmp_path, l2_bytes=64 << 20,
                        result=tmp_path / 'point-1.json')
    assert round2.measure(a, None, round5.workloads()[1], None, None, dict(sha256='TEST'),
                          dict(host_only=True), protocol=round5) == 0
    result = json.loads(a.result.read_text())
    assert len(result['arms']) == 8 and len(result['complete']) == 16
    assert len(samples) == 48 and {s for _, s in samples} == {1, -1}
    assert result['round_five'] and not result['production_changed'] and not result['performance_admitted']
