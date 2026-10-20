"""Host evidence checks; no synthetic timings are admitted as device results."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from dev.strategy.refresh_gap_review import RECEIPT, close_supplement, comparison
from dev.strategy.refresh_gaps import TABLE
from dev.strategy.refresh_selection import supplement_plan
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    return json.loads((ROOT/TABLE).read_bytes()), json.loads(RECEIPT.read_bytes())


def reseal(report):
    report['evidence_sha256'] = digest({k: v for k, v in report.items() if k != 'evidence_sha256'})


def test_completed_pairs_are_not_pending_and_do_not_rewrite_historical_table():
    table, report = evidence()
    original = deepcopy(table)
    pending = close_supplement(table, report)
    assert table == original
    assert supplement_plan(table)['counts'] == dict(host_keys=14, points=43, candidate_profile_cells=57)
    assert pending['counts'] == dict(host_keys=0, points=0, candidate_profile_cells=0)
    assert pending['closed_counts']['candidate_profile_cells'] == 57
    assert pending['remaining_gpu_work'] is False and pending['production_changed'] is False
    assert report['counts']['promoted_keys'] == 0
    assert table['summary']['changed'] == 187 and len(table['pending']) == 22


@pytest.mark.parametrize('fault', [
    'missing-alternative', 'duplicate-alternative', 'candidate-config', 'missing-profile',
    'duplicate-profile', 'router', 'status', 'median', 'rounds', 'delta', 'run', 'selection',
    'promoted-count', 'body-rechecks', 'production', 'normalize', 'relabel-old'])
def test_resealing_cannot_hide_incomplete_or_mislabelled_comparisons(fault):
    table, report = evidence()
    row = report['rows'][0]
    p = next(p for p in row['profiles'] if p['evidence'] == 'FRESH_PAIRED_RESULT')
    if fault == 'missing-alternative':
        report['rows'].pop()
    elif fault == 'duplicate-alternative':
        report['rows'][-1] = deepcopy(row)
    elif fault == 'candidate-config':
        row['config']['split'] = 99
    elif fault == 'missing-profile':
        row['profiles'].pop()
    elif fault == 'duplicate-profile':
        row['profiles'][-1] = deepcopy(row['profiles'][0])
    elif fault == 'router':
        p['router'] = 'other'
    elif fault == 'status':
        p['status'] = 'ELIGIBLE'
    elif fault == 'median':
        p['candidate']['median_us'] *= .5
    elif fault == 'rounds':
        p['candidate']['rounds'][0] = 0
    elif fault == 'delta':
        p['delta_pct'] = -1
    elif fault == 'run':
        report['run'] = 'other'
    elif fault == 'selection':
        report['selection'] = 'other'
    elif fault == 'promoted-count':
        report['counts']['promoted_keys'] = 14
    elif fault == 'body-rechecks':
        report['body_recheck_keys'] = []
    elif fault == 'production':
        report['production_changed'] = True
    elif fault == 'normalize':
        report['frequency_normalization'] = True
    elif fault == 'relabel-old':
        p = next(p for p in row['profiles'] if p['evidence'] != 'FRESH_PAIRED_RESULT')
        p['delta_pct'] = 0
    reseal(report)
    with pytest.raises(ValueError):
        close_supplement(table, report)


def test_each_candidate_has_a_new_loss_not_just_an_unmeasured_or_noisy_profile():
    table, report = evidence()
    for row in report['rows']:
        losses = [p for p in row['profiles']
                  if p['evidence'] == 'FRESH_PAIRED_RESULT' and p['status'] == 'MEASURED_REGRESSION']
        assert losses
        assert all(p['candidate']['median_us'] > p['original']['median_us'] for p in losses)
    # Synthetic fault: remove every actual loss of one candidate. Keep all
    # other profiles/candidates valid and recompute the altered statistics.
    row = report['rows'][0]
    for p in row['profiles']:
        if p['evidence'] == 'FRESH_PAIRED_RESULT':
            p['candidate'] = deepcopy(p['original'])
            p.update(status='ELIGIBLE', delta_pct=0)
    reseal(report)
    with pytest.raises(ValueError, match='fresh measured regression'):
        close_supplement(table, report)


def test_both_changed_body_keys_use_only_fresh_profile_controls():
    _, report = evidence()
    for row in report['rows']:
        if row['key'] in report['body_recheck_keys']:
            assert len(row['profiles']) == 5
            assert all(p['evidence'] == 'FRESH_PAIRED_RESULT' for p in row['profiles'])
    assert sum(p['evidence'] == 'FRESH_PAIRED_RESULT'
               for r in report['rows'] for p in r['profiles']) == 69


def test_slow_noisy_and_unmeasured_are_different_decisions():
    old = dict(median_us=10, rounds=[10]*6, spread_pct=0)
    assert comparison(dict(old, median_us=11), old) == 'MEASURED_REGRESSION'
    assert comparison(dict(old, median_us=9, spread_pct=6), old) == 'CONFIRMATION_GATE_NOT_MET'
    assert comparison(old, old) == 'ELIGIBLE'
    with pytest.raises(KeyError):
        comparison({}, old)
