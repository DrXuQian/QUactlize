"""Audited Asys mean table and bounded exceptions; no simulated GPU results."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from quactlize.dispatch.dense_prefill import EVIDENCE, bucket_costs, validate
from quactlize.dispatch.dense_prefill_mean import STATISTIC, capture_mean, accepted_exception
from quactlize.runtime.tuning import digest


def catalog():
    return json.loads(EVIDENCE.read_text())


def test_all_measured_m_families_and_exception_are_exact():
    data = catalog(); validate(data)
    assert Counter(p['shape'][1] for p in data['points']) == dict.fromkeys(
        [16,32,64,128,256,512,1024,2048,3072,4096,5120,8192], 45)
    assert Counter(data['recipes'][p['winner']]['route'] for p in data['points']) == {0:17,1:264,2:259}
    exceptions = [p for p in data['points'] if p['mean_evidence']['delta_pct'] > 0]
    assert len(exceptions) == 1
    p = exceptions[0]
    assert p['shape'] == [12,3072,25600,5120]
    assert p['us'] == pytest.approx(2423.9159814814816)
    assert p['mean_evidence']['control']['mean_us'] == pytest.approx(2410.623111111111)
    assert p['mean_evidence']['delta_pct'] == pytest.approx(.5514288114595978)
    assert p['mean_evidence']['accepted_regression']['decision'].startswith('USER_ACCEPTED')
    assert all(p['mean_evidence']['statistic'] == STATISTIC for p in data['points'])
    assert all(bucket_costs(p) for p in data['points'])
    assert not data['model_performance_admitted']


def test_old_native_and_bucket_evidence_is_not_relabelled():
    data = catalog(); update = data['mean_updates'][0]
    assert update['imported'] == 405 and update['exact_only'] and not update['native_comparison']
    assert [s['complete_points'] for s in update['sources']] == [312,93]
    assert all(p['native_us'] is None for p in data['points'])
    assert len(data['superseded_points']) == 315
    history = data['bucket_history'][0]
    assert len(history['buckets']) == 147
    original = {p['key']:p for p in data['superseded_points'] if p['native_us'] is not None}
    for p in history['training']:
        assert dict(p['costs']) == bucket_costs(original[p['key']])
    assert update['admission']['source_commit'] is None
    assert not update['admission']['native_source_commit_proven']
    assert all(data['authorities'][p['authority']]['receipt']['harness'] for p in data['points'])


@pytest.mark.parametrize('fault', ['capture','cost','native','exception','commit','control','fq','duplicate'])
def test_mean_admission_fails_closed_on_changed_contract(fault):
    data = catalog()
    p = next(p for p in data['points'] if p['mean_evidence']['delta_pct'] > 0)
    e = p['mean_evidence']
    if fault == 'capture': e['capture_means'][p['winner']].pop()
    elif fault == 'cost': e['capture_means'][p['winner']][0][0] *= .5
    elif fault == 'native': p['native_us'] = 10.
    elif fault == 'exception': e.pop('accepted_regression')
    elif fault == 'commit': e['source_commit'] = 'f'*40
    elif fault == 'control': e['control']['mean_us'] *= .5
    elif fault == 'fq': data['recipes'][p['winner']]['route'] = 0
    elif fault == 'duplicate': data['points'].append(copy.deepcopy(p))
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError): validate(data)


def test_mean_keeps_slow_samples_and_both_directions():
    captures = [dict(confirmation=i,complete_samples_us=ts,omitted_producer='DETECTED')
                for i,ts in enumerate(([1.,2.,3.], [4.,5.,60.]))]
    row = dict(confirmation_captures=captures,complete_samples_us=[1.,2.,3.,4.,5.,60.])
    mean, pairs = capture_mean(row)
    assert mean == 12.5 and pairs == [[2.,3],[23.,3]]
    row['complete_samples_us'].pop()
    with pytest.raises(ValueError,match='aggregate samples'): capture_mean(row)


def test_exception_is_not_a_blanket_point_or_candidate_tolerance():
    admission = catalog()['mean_updates'][0]['admission']
    e = admission['accepted_regressions'][0]
    assert accepted_exception(e['shape'],e['candidate'],e['control'],.5515,admission) == e
    for shape, candidate, control, delta in (
        ([12,2048,25600,5120],e['candidate'],e['control'],.1),
        (e['shape'],'another-candidate',e['control'],.1),
        (e['shape'],e['candidate'],'another-control',.1),
        (e['shape'],e['candidate'],e['control'],.6)):
        with pytest.raises(ValueError,match='unapproved mean regression'):
            accepted_exception(shape,candidate,control,delta,admission)
