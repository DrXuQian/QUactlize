"""Host-only refresh admission: same-point evidence, no hidden body substitution."""
import copy
import json
from pathlib import Path

import pytest

from dev.strategy import refresh_selection as selection
from dev.strategy.followup import paired_verdict
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT/'policies/kpack_decode_refresh_v1.json'


def candidate(cost, *, variant=3, body=None, spread=0, rounds=None, arm='candidate'):
    return dict(arm=arm, median_us=cost, rounds=rounds or [cost]*6, spread_pct=spread,
        roles=['original'] if arm == 'original' else ['rule'],
        config=dict(kind='simt', variant=variant, columns=4, warps=8, values=4, split=1),
        dequant_config=-1,
        implementation=dict(kind='shipping') if body is None else dict(kind='q8-address', body=body))


def profile(candidates, status='PASS'):
    return dict(status=status, candidates=candidates)


def test_router_profiles_require_one_common_non_regressing_body():
    old = candidate(10, arm='original')
    rows = [profile(dict(original=old, common=candidate(8, variant=1),
                         local=candidate(5, variant=i+4))) for i in range(2)]
    chosen, n = selection.selected_profiles(rows)
    assert n == 2 and [c['median_us'] for c in chosen] == [8, 8]
    rows[1]['candidates']['common']['median_us'] = 11
    chosen, _ = selection.selected_profiles(rows)
    assert all(c['arm'] == 'original' for c in chosen)


def test_router_gap_is_not_automatically_a_regression():
    old = candidate(10, arm='original')
    local = candidate(5, variant=1)
    rows = [profile(dict(original=old, local=local)), profile(dict(original=old))]
    for i, r in enumerate(rows):
        r.update(point=str(i), workload=dict(router=str(i)), raw_winner='local' if i == 0 else 'original')
    gap = selection.profile_tradeoffs(rows, old)[0]
    assert [p['status'] for p in gap['profiles']] == ['ELIGIBLE', 'NOT_MEASURED']
    rows[1]['candidates']['local'] = candidate(11, variant=1)
    gap = selection.profile_tradeoffs(rows, old)[0]
    assert [p['status'] for p in gap['profiles']] == ['ELIGIBLE', 'MEASURED_REGRESSION']


def test_raw_unstable_decision_is_not_silently_promoted():
    row = profile(dict(original=candidate(10, arm='original'),
                       fast=candidate(5)), status='UNSTABLE_RETAIN_ORIGINAL')
    chosen, n = selection.selected_profiles([row])
    assert n == 1 and chosen[0]['arm'] == 'original'


def test_filter_before_ranking_and_retain_incumbent_on_exact_ties():
    row = profile(dict(original=candidate(10, arm='original'),
                       jitter=candidate(4, spread=6),
                       stable=candidate(8, variant=2)))
    chosen, _ = selection.selected_profiles([row])
    assert chosen[0]['config']['variant'] == 2
    row['candidates']['stable'] = candidate(10, variant=2)
    assert selection.selected_profiles([row])[0][0]['arm'] == 'original'
    # A lower median cannot override the paired-round requirement.
    row['candidates']['stable'] = candidate(9, variant=2, rounds=[8, 8, 8, 11, 11, 11])
    assert selection.selected_profiles([row])[0][0]['arm'] == 'original'


def test_q8_body_identity_is_not_replaced_by_equal_geometry():
    a, b = candidate(5, variant=5, body=0), candidate(5, variant=5, body=4)
    original = candidate(10, variant=5, arm='original')
    assert len({selection.identity(c) for c in (a, b, original)}) == 3
    rows = [profile(dict(original=original, local=c)) for c in (a, b)]
    assert all(c['arm'] == 'original' for c in selection.selected_profiles(rows)[0])


def test_new_body_cannot_leak_to_indexed_split_or_dtype():
    from dataclasses import asdict
    from quactlize.dispatch.table import Recipe, Workload
    w = asdict(Workload(8, 1024, 2048, 1))
    r = Recipe.make('simt', 'fq', variant=5, columns=4, warps=8, values=4, split=1, q8_body=4)
    arm = dict(recipe=asdict(r))
    assert selection.measured_recipe(arm, w)['implementation'] == dict(kind='q8-address', body=4)
    for change in (dict(operation='indexed', experts=256, topk=8),
                   dict(storage='f16'), dict(qtype=12)):
        with pytest.raises(ValueError):
            selection.measured_recipe(arm, w | change)
    for change in (dict(split=2), dict(q8_body=2), dict(q8_body=True), dict(values=8)):
        params = dict(r.parameters) | change
        with pytest.raises(ValueError):
            selection.measured_recipe(dict(recipe=asdict(Recipe.make('simt', 'fq', **params))), w)


def test_dequant_is_part_of_selection_and_ties_are_not_cross_epoch():
    a, b = candidate(10), candidate(10)
    b['dequant_config'] = 5
    assert selection.identity(a) != selection.identity(b)
    # Measured components are positive; a faster producer with a slower
    # complete call is not a winner.
    row = profile(dict(original=candidate(10, arm='original'), producer_only=candidate(11)))
    assert selection.selected_profiles([row])[0][0]['arm'] == 'original'


def test_component_sum_validation_keeps_original_decision_and_samples(monkeypatch):
    import math
    samples = {str(i): dict(samples_us=[[x]*5 for _ in range(6)])
               for i, x in enumerate((1e16, 1., 1.))}
    arms = {'original': list(samples), 'other': list(samples)}
    roles = {'original': ['original'], 'other': ['rule']}
    recorded = {k: [[math.fsum((1e16, 1., 1.))]*5 for _ in range(6)] for k in arms}
    data = dict(component_records=samples, arm_components=arms, roles=roles,
                decision=paired_verdict(recorded, roles))
    saved = copy.deepcopy(data)
    monkeypatch.setattr(selection, 'validate_confirmation', lambda checked: None)
    selection.timing_validation(data)
    assert data == saved
    data['decision']['statistics']['other']['samples_us'][0][0] += 100
    with pytest.raises(ValueError, match='roundoff'):
        selection.timing_validation(data)


def test_checked_in_table_is_inactive_complete_and_unscaled():
    table = json.loads(TABLE.read_text())
    selection.verify(table)
    assert table['summary']['points'] == 1114 and table['summary']['host_keys'] == 930
    assert sum(len(r['profiles']) for r in table['rows']) == 1114
    assert len(table['pending']) == 22
    assert sum(p['potential_reduction_pct'] >= 1 for p in table['pending']) == 8
    assert table['frequency_normalization'] is False
    for fault in ('config', 'body', 'evidence', 'active'):
        edited = copy.deepcopy(table)
        if fault == 'config':
            edited['rows'][0]['config']['split'] = 99
        elif fault == 'body':
            edited['rows'][0]['implementation'] = dict(kind='q8-address', body=4)
        elif fault == 'evidence':
            edited['rows'][0]['profiles'][0]['selected_us'] *= .1
        else:
            edited['active'] = True
        with pytest.raises(ValueError):
            selection.verify(edited)


def test_table_and_new_q8_bodies_are_not_silently_activated():
    from tools.generate_decode_selection import EVIDENCE, effective
    table = json.loads(TABLE.read_text())
    current = effective(json.loads(EVIDENCE.read_text()))
    assert digest(current) == table['baseline_sha256']
    report = selection.readiness(table, current)
    assert report['kernel_launches'] == 0 and report['production_changed'] is False
    assert report['status'] == 'IMPLEMENTATION_REQUIRED'
    bodies = [r for r in table['rows'] if r['implementation']['kind'] == 'q8-address']
    assert len(bodies) == table['summary']['q8_body_keys'] > 0
    pending = {tuple(r['key']) for r in report['pending'] if r['reason'].startswith('Q8_BODY')}
    assert pending == {tuple(r['key']) for r in bodies}


def test_resealing_does_not_allow_a_false_retained_or_broader_body():
    table = json.loads(TABLE.read_text())
    row_index = next(i for i, r in enumerate(table['rows']) if r['implementation']['kind'] == 'q8-address')
    for fault in ('changed', 'body', 'compute'):
        edited = copy.deepcopy(table)
        row = edited['rows'][row_index]
        if fault == 'changed':
            row.update(changed=False, status='RETAIN_INCUMBENT')
        elif fault == 'body':
            row['implementation']['body'] = 99
        else:
            row['key'][-1] = 1
        edited['evidence_sha256'] = digest({k: v for k, v in edited.items() if k != 'evidence_sha256'})
        with pytest.raises(ValueError):
            selection.verify(edited)


def test_supplement_is_bounded_and_does_not_invent_old_time_controls():
    table = json.loads(TABLE.read_text())
    plan = selection.supplement_plan(table)
    assert plan['counts']['host_keys'] == 14 and plan['counts']['candidate_profile_cells'] == 57
    assert plan['contemporaneous_incumbent_required'] and not plan['executable']
    assert not plan['production_changed'] and plan['kernel_launches'] == 0
    assert len({(c['point'], digest(c['candidate'])) for c in plan['cells']}) == 57
    assert all(c['key'][1] == 2 and c['key'][7] == 8 for c in plan['cells'])
