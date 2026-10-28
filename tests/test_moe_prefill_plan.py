"""Large-M audit is host evidence, never a completed timing campaign."""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from dev.strategy import moe_model_plan as models, moe_prefill_plan as prefill
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
from test_moe_model_plan import synthetic_inventory


def large_inventory(source=None):
    source = source or synthetic_inventory()
    source['points'] = [dict(p, id=w.key, workload=asdict(w)) for p in source['points']
                        if p['workload']['m'] == 1 and p['workload']['experts'] == 256
                        for m in prefill.MS for w in [replace(Workload(**p['workload']), m=m)]]
    source['models'] = source['models'][:2]
    source['requested_models'] = [m['name'] for m in source['models']]
    source['tokens'] = list(prefill.MS)
    return source


@pytest.fixture(scope='module')
def plan(tmp_path_factory):
    return prefill.make_plan(tmp_path_factory.mktemp('moe-prefill'), large_inventory())


def test_all_m_models_routers_have_generated_tc_and_live_control(plan):
    assert plan['status'] == 'HOST_PLAN_ONLY'
    assert plan['counts']['points'] == 200 and plan['counts']['families'] == 4
    assert plan['coverage']['tokens'] == [16,32,64,128,256,512,1024,2048,4096,5120]
    assert plan['sha256'] == digest({k:v for k,v in plan.items() if k != 'sha256'})
    assert not plan['full_campaign_ready'] and not plan['performance_admitted']
    assert not plan['production_changed']
    assert plan['counts']['modules'] > 0
    for point in plan['points']:
        assert point['consumers'] and point['routed_rows'] == point['workload']['m']*8
        assert point['maximum_rows_per_expert'] == point['workload']['m']
        assert not point['champion_injection'] and not point['historical_times_reused']
        assert not point['rule_recall']['missing']
        assert point['original_recipe'] or point['original_missing']
        assert any(r['kind'] == 'tc' for r in point['candidates'])
        assert all(r['kind'] in ('tc','full') for r in point['candidates'])
    assert len(plan['execution_pilot']['points']) == 24


def test_current_full_limits_are_actual_cpp_rejections_not_token_m_guess(plan):
    for point in plan['points']:
        m = point['workload']['m']
        legal = [r for r in point['full_capability'] if r['rc']==0]
        assert bool(legal) == (128 <= m <= 4096)
        assert all(r['rows']==m*8 and bool(r['workspace_bytes'])==(r['rc']==0)
                   for r in point['full_capability'])
        if m == 5120:
            assert not any(r['kind']=='full' for r in point['candidates'])
            assert point['rejected_candidates']
        if legal:
            assert any(r['kind']=='full' for r in point['candidates'])
    assert plan['counts']['full_unavailable_points']==80


def test_router_never_changes_blind_candidate_generation(plan):
    groups = {}
    for point in plan['points']:
        w = replace(Workload(**point['workload']), router='')
        signature = digest(point['proposed_candidates'])
        assert groups.setdefault(w.key, signature) == signature


def test_execution_pilot_is_bounded_and_separate_from_selection(plan):
    pilot=prefill.pilot_plan(plan)
    assert pilot['counts']['points']==8 and pilot['counts']['arms']==20
    assert pilot['counts']['modules']==len(pilot['tc'])
    assert not pilot['performance_admitted'] and not pilot['production_changed']
    assert {'NATIVE_LARGE_M_AB','FULL_CONFIG_SCREEN_CONFIRM'}<=set(pilot['pending'])
    assert {p['workload']['m'] for p in pilot['points']}=={16,128,5120}
    assert {p['workload']['channels'] for p in pilot['points']}=={1,8}
    for point in pilot['points']:
        assert {a['recipe']['route'] for a in point['arms']}==(
            {'fq','sf','full'} if point['workload']['m']==128 else {'fq','sf'})
        assert all(dict(a['recipe']['parameters']).get('split',1)==1 for a in point['arms'])


@pytest.mark.parametrize('fault',['missing_m','wrong_ms','missing_model'])
def test_large_m_inventory_cannot_hide_coverage_holes(fault):
    inventory = large_inventory()
    if fault == 'missing_m':
        inventory['points'] = [p for p in inventory['points'] if p['workload']['m'] != 5120]
    elif fault == 'wrong_ms':
        inventory['tokens'] = list(range(1,9))
    else:
        inventory['models'].pop()
    with pytest.raises(ValueError):
        models.expanded_points(inventory, ms=prefill.MS)


def test_decode_inventory_still_refuses_large_m():
    with pytest.raises(ValueError):
        models.expanded_points(large_inventory())
