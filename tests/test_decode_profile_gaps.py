"""Host-only proof of the bounded supplement; these are not device results."""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from dev.strategy.refresh import assign, family, request
from dev.strategy.refresh_gaps import PHASE, make_plan, query_points, validate_producers
from dev.strategy.vector import source
from quactlize.dispatch.planning import plan_smallm
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    path = tmp_path_factory.mktemp('profile-gaps')
    cpp = plan_smallm(path, [request(Workload(**p['workload'])) for p in query_points()])
    return make_plan(cpp['requests']), cpp


def test_only_reviewed_missing_cells_and_two_body_rechecks(evidence):
    plan, cpp = evidence
    assert len(cpp['requests']) == 70  # Host queries, not 70 device points.
    assert plan['counts'] == dict(host_keys=14, points=49, full_points=1114, families=10,
        candidate_profile_cells=69, missing_profile_cells=57, recheck_profile_cells=12,
        controls=49, arms=118, max_arms=4, tc_modules=7, vector_modules=6, operations=dict(indexed=49))
    assert plan['sha256'] == digest({k: v for k, v in plan.items() if k != 'sha256'})
    assert not plan['production_changed'] and not plan['historical_times_reused']
    cells = plan['profile_gaps']['cells']
    assert len({(c['point'], c['candidate']) for c in cells}) == len(cells)
    keys = plan['profile_gaps']['body_recheck_keys']
    assert keys == [[14, 2, 512, 2048, 256, 8, 8, 8, 1],
                    [14, 2, 512, 3072, 256, 8, 8, 8, 1]]
    assert all(c['key'] in keys for c in cells if c['previous_status'] == 'ELIGIBLE')
    for k in keys:
        profiles = [p for p in plan['points'] if request(Workload(**p['workload'])) == k]
        assert len(profiles) == len({p['workload']['router'] for p in profiles}) == 5
        assert all(p['arms'][0]['expected_producer'] == 'common-packet-3d' for p in profiles)
    for p in plan['points']:
        assert p['sf_configs'] == [] and p['profile_gap']
        assert p['workload']['m'] == 8 and p['workload']['storage'] == 'f32'
        assert p['workload']['compute'] == 'bf16'
        assert p['arms'][0]['id'] == 'original'
        assert all(a['required'] and a['recipe']['route'] == 'fq' for a in p['arms'])
        assert len(p['arms']) == 1 + sum(c['point'] == p['id'] for c in cells)
        for a in p['arms']:
            pool = plan['tc'] if a['recipe']['kind'] == 'tc' else plan['vectors']
            assert pool[a['module']] == a['spec'] and digest(a['spec']) == a['module']


@pytest.mark.parametrize('mutation', ['order', 'recipe', 'body', 'status'])
def test_wrong_cpp_incumbent_cannot_become_a_control(evidence, mutation):
    _, cpp = evidence
    rows = deepcopy(cpp['requests'])
    if mutation == 'order':
        rows.reverse()
    else:
        row = next(r for r in rows if r['kind'] == 1)
        if mutation == 'recipe':
            row['config']['split'] = 8
        elif mutation == 'body':
            row['implementation']['producer'] = 'not-production'
        else:
            row['status'] = 1
    with pytest.raises(ValueError):
        make_plan(rows)


def test_all_new_simt_sources_match_production_dispatch_order(evidence):
    plan, _ = evidence
    assert {s['q'] for s in plan['vectors'].values()} == {10, 11, 14}
    for s in plan['vectors'].values():
        text = source(s)
        assert '#include "simt_common.cuh"' in text
        assert text.index('common_reader_scope') < text.index('measured_decode(*d,f)')
        assert text.index('measured_decode_launch') < text.rindex('::launch_v2<')
        c = s['config']
        args = ','.join(str(v) for v in (s['q'], c['variant'], c['columns'], c['warps'], c['values']))
        assert f'kCommonReader<{args}>' in text
        assert digest(s) in text


def captured(point):
    """Fixture for inventory checking only; deliberately contains no timings."""
    entries = {'common-packet-3d': 'quactlize::execution::simt::common_reader',
               'register-reuse': 'quactlize::execution::simt::register_reuse',
               'tc': 'cutlass::device_kernel'}
    return dict(arms={a['id']: {} for a in point['arms']},
        arm_components={a['id']: [a['id']] for a in point['arms']},
        component_records={a['id']: dict(kind='producer', identity=dict(node=dict(
            name=f'void {entries[a["expected_producer"]]}<2, 14>(qkg_call_v1)'))) for a in point['arms']})


def test_captured_inventory_rejects_historical_lookalike_or_missing_arm(evidence):
    plan, _ = evidence
    for p in plan['points']:
        validate_producers(p, captured(p))
    p = next(p for p in plan['points'] if p['arms'][0]['expected_producer'] == 'common-packet-3d')
    d = captured(p)
    d['component_records']['original']['identity']['node']['name'] = \
        'void quactlize::execution::simt::register_reuse<14, 1, 3, 4, 8, 8, 1, 0>(qkg_call_v1, int)'
    with pytest.raises(ValueError, match='current production'):
        validate_producers(p, d)
    d = captured(p)
    del d['arms'][p['arms'][-1]['id']]
    with pytest.raises(ValueError, match='every required pairing'):
        validate_producers(p, d)
    d = captured(p)
    d['resource_unavailable'] = {'candidate': dict(status='UNAVAILABLE')}
    with pytest.raises(ValueError, match='every required pairing'):
        validate_producers(p, d)


def test_missing_candidate_is_not_optional_resource_success(evidence, monkeypatch):
    import dev.strategy.resources as resources
    from tools.run_decode_refresh import bind_point
    p = next(p for p in evidence[0]['points'] if p['arms'][0]['recipe']['kind'] == 'simt')
    class Unavailable:
        def __init__(self, *args):
            pass
        def query(self, spec):
            return dict(status='UNAVAILABLE', runtime_status=0)
    monkeypatch.setattr(resources, 'NativeProbe', Unavailable)
    images = {k: {mid: {} for mid in evidence[0][k]} for k in ('tc', 'vectors')}
    with pytest.raises(ValueError, match='required or failed TC'):
        bind_point(p, images)


def test_seven_worker_families_do_not_split_a_profile_or_ab_pair(evidence):
    plan = evidence[0]
    jobs = assign(plan, 7)
    flat = [f for j in jobs for f in j]
    assert len(flat) == len(set(flat)) == 10 and all(jobs)
    assert jobs == assign(deepcopy(plan), 7)
    assert Counter(len(j) for j in jobs) == {1: 4, 2: 3}
    for p in plan['points']:
        assert sum(family(Workload(**p['workload'])) in j for j in jobs) == 1
    with pytest.raises(ValueError):
        assign(plan, 11)


def test_empty_or_failed_supplement_cannot_close_coverage(evidence, tmp_path):
    from tools.run_decode_refresh import aggregate, authority_for, sealed
    plan = evidence[0]
    run = dict(assignments=assign(plan, 7), cards=list(map(str, range(1, 8))),
               devices=[dict(device=dict(pci=f'gap-fault-{i}')) for i in range(7)],
               sha256='host-fault-test', cu_mhz=1700, memory_mhz=1800)
    p = plan['points'][0]
    i = next(i for i, j in enumerate(run['assignments']) if family(Workload(**p['workload'])) in j)
    target = tmp_path/('gpu-'+run['cards'][i])/'points'/(p['id']+'.json')
    sealed(target, dict(status='FAIL', error='host injected failure', point=p['id'],
                        authority=authority_for(run, i)))
    before = target.read_bytes()
    r = aggregate(plan, run, tmp_path)
    assert r['expected'] == 49 and r['completed'] == 0 and r['status'] == 'INCOMPLETE'
    assert r['counts'] == dict(FAIL=1, MISSING=48)
    assert r['profile_gap_coverage']['expected'] == 69
    assert r['profile_gap_coverage']['completed'] == 0
    assert target.read_bytes() == before and not r['no_regression_admitted']


def test_real_cli_plan_resume_and_lightweight_pack(tmp_path):
    out = tmp_path/'results'
    command = [sys.executable, ROOT/'tools/run_decode_refresh.py', '--phase', PHASE,
               '--plan-only', '--output', out]
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout+p.stderr
    assert '"points": 49' in p.stdout and '"arms": 118' in p.stdout
    initial = {name: (out/name).read_bytes() for name in ('plan.json', 'cpp-originals.json')}
    assert not (out/'run.json').exists() and not (out/'cache').exists()
    p = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout+p.stderr
    assert all((out/n).read_bytes() == b for n, b in initial.items())
    with tarfile.open(sorted(out.glob('review-lite-*.tar.xz'))[-1]) as t:
        assert {'plan.json', 'cpp-originals.json', 'transfer-manifest.json'} <= set(t.getnames())
        assert not any(n.endswith(('.so', '.o', '.acurep')) for n in t.getnames())
    p = subprocess.run(command + ['--clock-retry-from', str(tmp_path/'prior')],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and 'invalid worker' in p.stderr
    p = subprocess.run([str(c) if c != PHASE else 'all' for c in command] + ['--resume'],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0 and 'frozen plan/phase changed' in p.stderr


@pytest.mark.parametrize('q', [10, 11, 14])
def test_hgcc_carrier_compile_only(tmp_path, evidence, q):
    sdk = os.environ.get('STRATEGY_COMPILE_SDK')
    if not sdk:
        pytest.skip('opt-in compiler-only check; no device available locally')
    from quactlize.runtime.compiler import FLAGS
    items = [s for s in evidence[0]['vectors'].values() if s['q'] == q]
    for i, item in enumerate(items):
        src, obj = tmp_path/f'{i}.cu', tmp_path/f'{i}.o'
        src.write_text(source(item))
        cmd = [str(Path(sdk)/'bin/hgcc'), *FLAGS, *[
            '-I'+str(ROOT/p) for p in ('.', 'quactlize/execution', 'quactlize/include',
                                      'third_party/actlize/include', 'third_party/actlize/tools/util/include')],
            '-c', str(src), '-o', str(obj)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
        assert obj.stat().st_size > 0
