"""Sealed component evidence -> one executable catalog, no online tuning."""
import copy
import ctypes as C
import json
import math
from pathlib import Path
import subprocess

import pytest

from dev.strategy import production as fit
from dev.strategy.followup import paired_verdict
from quactlize.dispatch.planning import plan_smallm, validate_inventory
from quactlize.execution.native import Call, SimtCallV2, Sizes, arrangement
from tools import generate_decode_selection as gen

ROOT = gen.ROOT


@pytest.fixture(scope='module')
def evidence():
    return json.loads(gen.COMPONENTS.read_text())


@pytest.fixture(scope='module')
def baseline():
    return gen.effective(json.loads(gen.EVIDENCE.read_text()), include_components=False)


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return plan_smallm(tmp_path_factory.mktemp('component-plan'))


def test_every_reviewed_point_and_unaffected_key(evidence, baseline, plan):
    assert json.loads(fit.serialize(evidence)) == evidence
    assert gen.COMPONENTS.read_text() == fit.serialize(evidence)
    assert evidence['summary'] == dict(points=1114, host_keys=930, changed=448, retained=482)
    assert sum(len(r['profiles']) for r in evidence['rows']) == 1114
    effective = fit.apply(evidence, baseline)
    expected = {tuple(r['key']): r for r in effective['exact']}
    declared = {tuple(r['key']): r for r in evidence['rows'] if r['changed']}
    for before, after in zip(baseline['exact'], effective['exact']):
        if tuple(before['key']) not in declared:
            assert before == after
    assert effective['buckets'] == baseline['buckets']
    assert effective['excluded'] == baseline['excluded']
    for actual in plan['requests']:
        row = expected[tuple(actual['request'])]; c = row['config']
        assert actual['status'] == 0 and actual['policy'] == row['policy']
        assert actual['dequant_config'] == row.get('dequant_config', -1)
        if c['kind'] == 'tc':
            assert actual['kind'] == 0
            assert actual['parent']['symbol'] == c['symbol']
            for f in ('route', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn'):
                assert actual['parent'][f] == c[f]
            for f in ('split', 'grid_mode', 'grid_b'):
                assert actual[f] == c[f]
        else:
            assert actual['kind'] == (1 if c['kind'] == 'simt' else 2)
            assert actual['config'] == {k: v for k, v in c.items() if k != 'kind'}
    assert sum(r['policy'] == 18 for r in plan['requests']) == 448


@pytest.mark.parametrize('fault', ['config', 'samples', 'scope', 'incumbent'])
def test_edited_evidence_cannot_change_production(evidence, baseline, fault):
    e = copy.deepcopy(evidence); b = copy.deepcopy(baseline)
    if fault == 'config':
        e['rows'][0]['config']['split'] = 100
    elif fault == 'samples':
        e['rows'][0]['profiles'][0]['record_sha256'] = '0'*64
    elif fault == 'scope':
        e['runtime_router_readback'] = True
    else:
        b['exact'][0]['config']['split'] = 100
    with pytest.raises(ValueError):
        fit.apply(e, b)


def test_sum_roundoff_is_not_a_performance_tolerance():
    # Three positive components exhibit the Python 3.10/3.12 sum distinction.
    terms = [1e16, 1., 1.]
    components = {str(i): dict(samples_us=[[t]*5 for _ in range(6)]) for i, t in enumerate(terms)}
    arms = {'original': list(components), 'candidate': list(components)}
    roles = {'original': ['original'], 'candidate': ['rule']}
    samples = {k: [[math.fsum(terms)]*5 for _ in range(6)] for k in arms}
    decision = paired_verdict(samples, roles)
    fit.validate_component_decision(components, arms, roles, decision)
    for delta in (100., 1e12):
        bad = copy.deepcopy(samples); bad['candidate'][0][0] += delta
        with pytest.raises(ValueError, match='roundoff'):
            fit.validate_component_decision(components, arms, roles, paired_verdict(bad, roles))
    bad = copy.deepcopy(decision); bad['winner'] = 'candidate'
    if bad != decision:
        with pytest.raises(ValueError, match='decision differs'):
            fit.validate_component_decision(components, arms, roles, bad)


def test_router_profiles_need_a_common_no_regression_choice():
    w = dict(qtype=12, operation='indexed', n=512, k=2048, experts=256, topk=8,
             channels=1, m=1, compute='bf16', router='spread')
    cfg = dict(kind='simt', variant=0, columns=4, warps=4, values=4, split=1)
    baseline = dict(exact=[dict(key=list(fit.key(w)), config=cfg)])
    def candidate(variant, cost):
        return dict(config=dict(cfg, variant=variant, dequant_config=-1), roles=['rule'],
            median_us=cost, rounds=[cost]*6, spread_pct=0, components=[], error=0)
    points = [dict(point=str(i), workload=dict(w, router=router), source='same-device',
        record_sha256=str(i), status='PASS', candidates={'original': candidate(0, 10),
        'local': candidate(i+1, 5), 'common': candidate(3, 8)})
        for i, router in enumerate(('spread', 'concentrated'))]
    result = fit.fit(points, [], baseline)['rows'][0]
    assert result['changed'] and result['config']['variant'] == 3
    assert all(p['selected_arm'] == 'common' for p in result['profiles'])
    points[1]['candidates']['common']['median_us'] = 11
    result = fit.fit(points, [], baseline)['rows'][0]
    assert not result['changed'] and result['config'] == cfg
    points[1]['status'] = 'UNSTABLE_RETAIN_ORIGINAL'
    points[1]['candidates']['common']['median_us'] = 8
    assert not fit.fit(points, [], baseline)['rows'][0]['changed']


def test_new_q4_specializations_and_compute_are_compiled(plan):
    from quactlize.execution import q4_decode_codegen, simt_codegen
    execution = dict(simt_configs={str(q): [c.record() for c in simt_codegen.runtime_inventory(q)]
        for q in simt_codegen.QTYPES}, simt_compute_v2=dict(compute=['f16', 'bf16'], formats=list(simt_codegen.QTYPES)),
        q4_decode_policy_sha256='fixture', q4_decode_compute_v2=dict(compute=['f16', 'bf16']),
        q4_decode_configs={f'{n}x{k}': rs for (n, k), rs in q4_decode_codegen.recipes().items()})
    validate_inventory(plan, execution, jit=True)
    for fault in ('missing-tuple', 'bf16'):
        bad = copy.deepcopy(execution)
        if fault == 'missing-tuple':
            for rs in bad['q4_decode_configs'].values():
                rs.clear()
        else:
            bad['q4_decode_compute_v2']['compute'] = ['f16']
        with pytest.raises(ValueError, match='Q4'):
            validate_inventory(plan, bad, jit=True)


def test_actual_q4_c_abi_agrees_with_typed_catalog(tmp_path, plan):
    libpath = tmp_path/'q4.so'
    subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', '-I'+str(ROOT),
        ROOT/'quactlize/execution/q4_decode.cpp', ROOT/'tests/q4_decode_policy_host.cpp', '-o', libpath], check=True)
    from test_q4_decode_policy import Config
    lib = C.CDLL(str(libpath)); fn = lib.quactlize_kpack_q4_decode_select_v2
    fn.argtypes = [C.POINTER(SimtCallV2), C.POINTER(type(arrangement(12))), C.POINTER(Config), C.POINTER(Sizes)]
    checked = 0
    for row in plan['requests']:
        if row['kind'] != 2:
            continue
        q, mode, n, k, e, top, ch, m, compute = row['request']
        call = Call(version=1, size=C.sizeof(Call), qtype=q, mode=mode, n=n, k=k,
            experts=e, topk=top, channels=ch, rows=m*top, input_type=1,
            a_row_stride=k, a_token_stride=k*ch, out_row_stride=n, ids_stride=top)
        typed = SimtCallV2(call, compute)
        config = Config(); sizes = Sizes(); a = arrangement(12)
        assert fn(C.byref(typed), C.byref(a), C.byref(config), C.byref(sizes)) == 0
        assert {f: getattr(config, f) for f in row['config']} == row['config']
        checked += 1
    assert checked > 118
