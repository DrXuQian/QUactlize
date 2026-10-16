"""Synthetic host checks; none of these costs are PPU measurements."""
from dataclasses import asdict

import pytest

from dev.strategy.prefill_selection import control_helper_config, compose, rule_selection
from quactlize.dispatch.table import Recipe, Workload


def recipe(config=1, route='sf'):
    return Recipe.make('full' if route == 'full' else 'tc', route,
                       dequant_config=config, split=1)


@pytest.mark.parametrize('q', [10, 11, 12, 13, 14])
@pytest.mark.parametrize('config', [-1, 0, 1, 2, 3, 4, 5])
def test_legacy_bf16_metadata_mapping_is_not_an_fp16_helper(q, config):
    w = Workload(q, 256, 512, 128, 'indexed', 'bf16', experts=256, topk=8)
    assert control_helper_config(w, recipe(config)) == (config if config in (4, 5) else 0)
    assert control_helper_config(Workload(q, 256, 512, 128), recipe(config)) == max(0, config)
    assert control_helper_config(w, recipe(config, 'full')) == max(0, config)


def component(value, spread=0):
    return dict(status='PASS', scope='ISOLATED_SINGLE_KERNEL', median_us=value, spread_pct=spread)


def selection(old=10, new=8, *, original_noise=False, winner_noise=False, loser_noise=False):
    roles = {'original':['original'], 'rule':['rule'], 'loser':['rule']}
    records = {'o':component(old, 10 if original_noise else 0),
               'r':component(new, 10 if winner_noise else 0),
               'l':component(100, 10 if loser_noise else 0)}
    return rule_selection({'original':['o'], 'rule':['r'], 'loser':['l']}, records, roles)


def test_slower_rule_is_an_open_gap_never_an_original_winner():
    out = selection(new=11)
    assert out['recommended'] is None
    assert out['best_rule'] == 'rule' and out['decision'] == 'RULE_GAP'
    assert out['legacy_fallback'] is False and not out['performance_admitted']


def test_same_kernel_tie_selects_rule_identity_not_original_dispatch():
    out = selection(new=10)
    assert out['recommended'] == 'rule' and out['decision'] == 'CONFIRM_REQUIRED'
    assert out['baseline_role'] == 'REFERENCE_ONLY'


def test_noisy_loser_does_not_block_a_stable_rule_and_baseline():
    out = selection(loser_noise=True)
    assert out['recommended'] == 'rule' and out['decision'] == 'CONFIRM_REQUIRED'
    assert out['noisy'] == ['loser']


@pytest.mark.parametrize('field', ['original_noise', 'winner_noise'])
def test_unstable_comparison_cannot_fall_back_to_original(field):
    out = selection(**{field:True})
    assert out['recommended'] is None and out['decision'] == 'UNSTABLE_COMPARISON'


def test_missing_original_or_rule_is_not_success():
    for arms, records, roles in (
        ({'original':[], 'rule':['r']}, {'r':component(1)},
         {'original':['original'], 'rule':['rule']}),
        ({'original':['o']}, {'o':component(1)}, {'original':['original']}),
    ):
        out = rule_selection(arms, records, roles)
        assert out['recommended'] is None
        assert out['decision'] in ('INCOMPLETE', 'NO_RULE_CANDIDATE')


def fixture(config=1):
    w = Workload(14, 256, 512, 128, 'indexed', 'bf16', experts=256, topk=8)
    original = asdict(recipe(config)); candidate = asdict(recipe(-1))
    p = dict(id=w.key, workload=asdict(w), arms=[
        dict(id='original', recipe=original, roles=['original']),
        dict(id='rule', recipe=candidate, roles=['rule'])])
    data = {'original':['o'], 'rule':['r']}
    helpers = {'sf':{0:['s']}, 'full_dequant':{}}
    records = {'o':component(10), 'r':component(8), 's':component(2)}
    return p, data, helpers, records


@pytest.mark.parametrize('config', [1, 2, 3])
def test_saved_bf16_original_can_be_recomposed_without_changing_receipts(config):
    p, data, helpers, records = fixture(config)
    before = repr((p, data, helpers, records))
    out = compose(p, data, helpers, [], records, [])
    assert out['status'] == 'PASS' and out['selection']['costs_us']['original'] == 12
    assert out['selection']['costs_us']['rule'] == 10
    assert out['control_helper_bindings']['original'] == dict(requested=config, effective=0, route='sf')
    assert repr((p, data, helpers, records)) == before


def test_empty_composition_is_incomplete_even_when_every_job_passed():
    p, data, helpers, records = fixture(5)
    out = compose(p, data, helpers, [], records, [])
    assert out['status'] == 'INCOMPLETE'
    assert out['selection']['missing'] == {'original':['EMPTY_CALL']}
    assert out['selection']['recommended'] is None


def test_missing_helper_must_not_be_replaced_with_a_faster_different_config():
    p, data, helpers, records = fixture(0)
    helpers['sf'] = {5:['s']}
    out = compose(p, data, helpers, [], records, [])
    assert out['status'] == 'INCOMPLETE' and not out['components']['original']


def test_missing_receipt_blocks_completion_even_with_composable_other_arms():
    p, data, helpers, records = fixture()
    out = compose(p, data, helpers, [], records, [dict(id='unused',status='PENDING')])
    assert out['status'] == 'INCOMPLETE' and out['selection']['recommended'] is None
