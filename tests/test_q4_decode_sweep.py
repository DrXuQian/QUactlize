"""CPU contracts for the incremental SIMT/TC sweep; PPU measurements are separate."""
import copy
import csv
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from dev.gemv_ppu import decode_sweep as spec, run_decode_sweep as runner
from dev.gemv_ppu.decode_access import access
from dev.gemv_ppu.decode_bench import DenseTC, fixture, truth
from dev.gemv_ppu.moe_compare_bench import Unsupported
from tools.gguf_internal_shape_inventory import _routing_fixture, _identity_sha256


@pytest.fixture(scope='module')
def manifest():
    return spec.verify()


def test_complete_prior_families_and_small_m_denominator():
    workloads = spec.workloads()
    dense = [w for w in workloads if w['operator'] == 'dense']
    grouped = [w for w in workloads if w['operator'] == 'grouped']
    assert len(workloads) == len({w['id'] for w in workloads}) == 372
    assert len(dense) == 96 and len(grouped) == 276
    assert len(spec.DENSE) == 12 and set(spec.source_families()[0]) < set(spec.DENSE)
    assert (4096, 4096) in spec.DENSE
    assert len(spec.GROUPED) == 6
    assert {w['tokens'] for w in dense} == set(range(1, 9))
    assert {w['tokens'] for w in grouped} == set(range(1, 9))
    assert sum(runner.profile_workload(w) for w in workloads) == 18
    assert spec.plan()['production_changed'] is False and spec.plan()['prefill_changed'] is False


@pytest.mark.parametrize('tokens', range(1, 9))
def test_actual_weighted_router_bytes_match_old_inventory(tokens):
    ids = spec.real_ids(tokens)
    old = _routing_fixture(256, 8, tokens)
    assert _identity_sha256(ids.tolist()) == old['token_routes_sha256']
    assert all(len(set(r)) == 8 for r in ids)
    assert ids.min() >= 0 and ids.max() < 256


@pytest.mark.parametrize('repeat', range(2, 8))
def test_intermediate_expert_rows_not_just_extreme_routers(repeat):
    ids = spec.routed_ids(8, f'repeat{repeat}')
    assert int(np.bincount(ids.ravel()).max()) == repeat
    assert all(len(set(r)) == 8 for r in ids)
    assert ids.shape == (8, 8)


def test_prior_simt_winners_and_tc_tuples_are_in_catalog(manifest):
    for row in csv.DictReader(spec.DENSE_REVIEW.open(), delimiter='\t'):
        n, k = int(row['N']), int(row['K'])
        old = spec.old_dense_recipe(n, k, row['kpack_key'])
        assert old in spec.simt_inventory(n, k)
        w = next(w for w in spec.workloads() if w['operator'] == 'dense' and (w['n'], w['k']) == (n, k))
        names = {c['key'] for c in spec.catalog(manifest, w)}
        for f in ('tc_current_key', 'tc_best_key'):
            symbol, split = row[f].rsplit(':s', 1)
            assert f'tc:{symbol}:s{split}:b0:g0' in names
    parents, families, runtime = spec.historical_tc()
    assert any(p['ap'] == 1 for p in parents.values())
    for w in spec.workloads():
        entries = spec.catalog(manifest, w)
        assert len(entries) == len({r['key'] for r in entries})
        assert runner.policy_key(manifest, w) in {r['key'] for r in entries}
        for symbol, s, b, g in runtime[spec.family(w)]:
            assert f'tc:{symbol}:s{s}:b{b}:g{g}' in {r['key'] for r in entries}
        for r in entries:
            if r['arm'] == 'tc' and parents.get(r['parent'], {}).get('ap') == 1 and w['rows'] != 1:
                assert r['reason'] == 'PACKED_ROW_A_REQUIRES_M1'


def test_ported_native_tuple_inventory_and_fast_arithmetic(manifest):
    runner.verify_inventory(manifest)
    for shape in set(spec.DENSE) | set(spec.GROUPED):
        n, k = shape
        for c in spec.simt_inventory(n, k):
            assert c.columns * c.values <= 32 and n % c.tile_n == 0
            geometry = c.geometry(n, 64)
            assert geometry['grid'] * c.tile_n == 64 * n and geometry['split'] == 1
    body = (spec.ROOT / 'dev/gemv_ppu/decode_kernel.cuh').read_text()
    assert 'row_reuse<Input,Variant,Columns,Warps,P,N,K>' in body
    assert 'row_medium<Input,Variant&1,(Variant>>1)&1' in body
    assert 'row_meta<Input,Variant,1,1,Warps,N,K>' in body
    assert 'new' not in body and '__shfl' not in body


@pytest.mark.parametrize('fault', ('recipes', 'native', 'arithmetic', 'payload', 'current'))
def test_manifest_inventory_plants_rejected(manifest, fault):
    m = copy.deepcopy(manifest)
    if fault == 'recipes': m['simt_recipes']['512x2048'][0]['warps'] += 1
    elif fault == 'native': m['simt_native']['libq4_decode_n512_k2048.so'].popitem()
    elif fault == 'arithmetic':
        next(iter(m['simt_native']['libq4_decode_n512_k2048.so'].values()))['code_lowering'] = None
    elif fault == 'payload': m['modules'][0]['sha256'] = 'bad'
    else: m['selection'].pop()
    with pytest.raises(ValueError): runner.verify_inventory(m)


@pytest.mark.parametrize('shape', sorted(set(spec.DENSE) | set(spec.GROUPED)))
def test_address_models_include_actual_f32_strides_and_valid_requests(shape):
    n, k = shape
    base = SimpleNamespace(n=n, k=k, a=16, base=SimpleNamespace(low=32, units=64),
                           data={'arows': np.array([0, 7])}, workload={'rows': 8})
    for c in spec.simt_inventory(n, k):
        m = access(c, base)
        assert m['f32_a_row_stride_bytes'] == (k + 8) * 4
        assert m['f32_output_row_stride_bytes'] == (n + 8) * 4
        stream = m['f32_A_first_instruction'][0]
        assert len(stream['lane_byte_addresses']) == 32
        assert stream['granules']['32']['lane_bytes'] == 32 * stream['width_bytes']
        assert min(stream['lane_byte_addresses']) >= 16
        assert max(stream['lane_byte_addresses']) + stream['width_bytes'] <= 16 + k * 4
        if c.reader == 2 and c.variant & 1:
            assert stream['additional_load_offset_bytes'] == (16 if c.columns == 4 else None)


@pytest.fixture(scope='module')
def fake_weights():
    # Explicit small logical matrix, so the categorized dot is independently testable.
    e, n, k = 256, 32, 256
    categories = np.tile(np.arange(k) % 4, (e, 1))
    weights = np.random.default_rng(43).normal(0, .2, (e, n, k)).astype('<f2').astype('f8')
    sums = np.stack([weights[:, :, categories[0] == g].sum(2) for g in range(4)], axis=1)
    absolute = np.stack([np.abs(weights[:, :, categories[0] == g]).sum(2) for g in range(4)], axis=1)
    return SimpleNamespace(n=n, k=k, experts=e, categories=categories, sums=sums, abs_sums=absolute), weights


@pytest.mark.parametrize('router', ['real', 'spread', 'cluster'] + [f'repeat{i}' for i in range(2, 8)])
@pytest.mark.parametrize('channels', (1, 8))
def test_expert_channel_and_padding_oracle(fake_weights, router, channels):
    w, weights = fake_weights
    data = fixture(w, dict(tokens=8, channels=channels, operator='grouped', router=router))
    expected = np.stack([weights[e] @ data['ah'][data['arows'][i]].astype('f8') for i, e in enumerate(data['expert'])])
    np.testing.assert_allclose(data['golden'], expected, atol=1e-10, rtol=1e-10)
    assert np.isnan(data['a'][:, :, w.k:]).all()
    assert (data['ids'][:, 8:] == -99).all()
    changed = data['ids'].copy(); changed[:, :8] = np.roll(changed[:, :8], 1, axis=1)
    assert not np.array_equal(truth(w, data, changed)['golden'], data['golden'])


def test_dense_oracle_and_both_casts_inside_timed_launch(fake_weights):
    w, weights = fake_weights
    data = fixture(w, dict(tokens=7, operator='dense'))
    np.testing.assert_allclose(data['golden'], data['ah'].astype('f8') @ weights[0].T, atol=1e-10, rtol=1e-10)
    assert np.isnan(data['a'][:, w.k:]).all()
    order = []
    tc = DenseTC.__new__(DenseTC)
    tc.r = SimpleNamespace(stream=9); tc.handles = [11]; tc.negative = 12
    tc.call = SimpleNamespace(a=7, output=8)
    tc.base = SimpleNamespace(workload={'rows': 7}, a=1, out=2, k=256, n=32, output_stride=40)
    tc.cast = lambda mode, *a: order.append('F32_TO_F16' if mode else 'F16_TO_F32') or 0
    tc.module = SimpleNamespace(run=lambda *a: order.append('TC_AND_REDUCER') or 0)
    assert tc.launch() == 0 and order == ['F32_TO_F16', 'TC_AND_REDUCER', 'F16_TO_F32']


def proof():
    return dict(error=1e-5, zero_codes='PASS', zero_a='PASS', output_guard='PASS',
                mutable_input_replay='PASS', eager_graph_bits='PASS', f16_f32_bits='PASS',
                immutable_reader='EXACT_BITS')


def test_failed_cell_retry_preserves_prior_success_and_validates_result(tmp_path, monkeypatch):
    w = next(w for w in spec.workloads() if w['id'] == 'dense-n512-k2048-m1')
    recipes = spec.simt_inventory(w['n'], w['k'])[:2]
    configs = [dict(key='simt:' + c.key, arm='simt', recipe=c.key, reason=None) for c in recipes]
    configs += [dict(key='tc:' + k, arm='tc', parent=k, split=1, reason=None) for k in ('p0', 'p1', 'p2')]
    counts = {c['key']: 0 for c in configs}; planted = [False]; ident = {'test': 1}
    class FakeBase:
        def __init__(self, *args):
            self.n = 512; self.k = 2048; self.workload = w
            self.device = {'l2_bytes': 67108864}; self.ids = None
            self.data = {'expert': np.array([0])}; self.copies = 256; self.calls_per_graph = 512
            self.weight_sha256 = {'low': 'fixture', 'units': 'fixture'}
        def correctness(self, key): counts['simt:' + key] += 1; return proof()
        def measure(self, key, count): return [1.0 if key == recipes[0].key else 1.1] * count
        def close(self): pass
    class FakeTC:
        def __init__(self, b, bundle, record, candidate): self.key = candidate['key']
        def correctness(self):
            counts[self.key] += 1
            if self.key == 'tc:p0' and not planted[0]:
                planted[0] = True; raise ValueError('planted numeric failure')
            return proof()
        def measure(self, count): return [2.0 + int(self.key[-1])] * count
        def receipt(self): return dict(scope='DENSE_F32_CAST_TC_REAL_REDUCER_F32_CAST', split=1)
        def close(self): pass
    monkeypatch.setattr(runner, 'Base', FakeBase); monkeypatch.setattr(runner, 'DenseTC', FakeTC)
    monkeypatch.setattr(runner, 'identity', lambda _: ident)
    monkeypatch.setattr(runner, 'policy_key', lambda *a: 'tc:p2')
    monkeypatch.setattr(runner, 'access', lambda *a: {'model': 'test'})
    monkeypatch.setattr(spec, 'catalog', lambda *a: configs)
    args = SimpleNamespace(output=tmp_path, profile=False, retry_failed=False, bundle=tmp_path)
    manifest = dict(modules=[dict(parent={'symbol': p}) for p in ('p0', 'p1', 'p2')],
                    simt_native={'libq4_decode_n512_k2048.so': {c.key + ':a1': {} for c in recipes}})
    with pytest.raises(ValueError, match='planted'): runner.child(args, manifest, w)
    assert runner.child(args, manifest, w) == 2
    assert counts == {k: 1 for k in counts}
    args.retry_failed = True
    assert runner.child(args, manifest, w) == 0
    assert counts == {k: 2 if k == 'tc:p0' else 1 for k in counts}
    p = json.loads((tmp_path / (w['id'] + '.json')).read_text())
    runner.validate_result(p, ident, w, configs, 'tc:p2')
    assert len(p['confirmation']) == 30  # top2 + top2 + today's slower policy.
    assert p['current_policy_us'] == 4 and p['simt_vs_tc_pct'] == -50
    for change in ({'best': {}}, {'simt_vs_tc_pct': 0}, {'copies': 2}, {'current_policy_us': None},
                   {'confirmation': p['confirmation'][:-1]}, {'first_launch': 'INCLUDED'}):
        with pytest.raises(ValueError): runner.validate_result(p | change, ident, w, configs, 'tc:p2')
    for fault in ('nan', 'guard', 'scope', 'duplicate', 'failure', 'unknown-structural'):
        q = copy.deepcopy(p)
        if fault == 'nan': q['screen'][0]['samples_us'][0] = float('nan')
        elif fault == 'guard': q['screen'][0]['correctness']['output_guard'] = 'MISSING'
        elif fault == 'scope': next(r for r in q['screen'] if r['arm'] == 'tc')['execution']['scope'] = 'PRODUCER_ONLY'
        elif fault == 'duplicate': q['screen'].append(q['screen'][0])
        elif fault == 'failure': q['failed']['tc:p0'] = {'error': 'numeric'}
        else: q['screen'][0].update(status='STRUCTURAL', reason='WRONG_OUTPUT', samples_us=[])
        with pytest.raises(ValueError): runner.validate_result(q, ident, w, configs, 'tc:p2')


def test_no_gpu_in_controller_and_no_compilation_in_runner():
    main = inspect.getsource(runner.main)
    assert "common + ['--probe']" in main
    assert 'Compiler(' not in main and 'jit' not in main.lower().replace("jit='none'", '')


def test_box_shell_and_help_are_non_device_checks():
    root = spec.ROOT
    shell = root / 'tools/run_q4_decode_sweep_ppu_box.sh'
    if shell.exists():
        subprocess.run(['bash', '-n', str(shell)], check=True)
        assert '\n(\n' in shell.read_text()
    subprocess.run(['python', str(root / 'dev/gemv_ppu/run_decode_sweep.py'), '--help'], check=True, capture_output=True)
