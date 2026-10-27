"""Measured middle-M admission and interval safety; no simulated GPU timings."""
from collections import Counter
import copy
import json
from pathlib import Path

import pytest

from quactlize.dispatch.dense_prefill import (
    EVIDENCE, audit_buckets, bucket_interval, bucket_training_costs, fit, validate)
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    return json.loads(EVIDENCE.read_text())


def test_middle_winners_keep_all_sample_means_and_real_source_commit():
    data = catalog(); update = data['mean_updates'][-1]
    assert update['imported'] == 225 and not update['exact_only']
    assert update['bucket_predictions_changed'] and not update['native_comparison']
    assert update['admission'] == json.loads((ROOT/'policies/dense_prefill_middle_admission.json').read_text())
    points = [p for p in data['points'] if p['shape'][1] in (16,32,64,128,256)]
    assert len(points) == 225
    assert sum(p['mean_evidence']['measured_candidates'] for p in points) == 5897
    expected = {16:{0:12,1:33},32:{0:3,1:42},64:{0:2,1:43},128:{1:38,2:7},256:{1:35,2:10}}
    for m, counts in expected.items():
        rows = [p for p in points if p['shape'][1] == m]
        assert Counter(data['recipes'][p['winner']]['route'] for p in rows) == counts
    for p in points:
        e = p['mean_evidence']
        assert e['source_commit'] == 'd3537c36f495fde486700a568fff2703372bbe67'
        assert p['us'] == min(dict(p['costs']).values())
        assert e['delta_pct'] <= 0 and p['native_us'] is None
        assert e['review_sha256'] == update['admission']['reviews'][0]
        assert all(len(pair) == 2 for pair in e['capture_means'].values())


def test_large_exact_recipes_and_measurements_are_unchanged():
    # Digest taken from the pre-admission catalog 1562b901...; only its bucket
    # training was refreshed. No remeasurement or new large-M win is claimed.
    rows = [{k:v for k,v in p.items() if k not in ('bucket_costs','bucket_evidence')}
            for p in catalog()['points'] if p['shape'][1] >= 512]
    assert len(rows) == 315
    assert digest(rows) == '4bca9f6ab1291326e679a98489ee667d9af701676f325ea934c1c2a3fc03728a'


def test_buckets_use_active_means_and_only_interval_safe_recipes():
    data = catalog()
    assert data['bucket_policy'] == 'ACTIVE_MEANS_MNK_INTERVALS_V2'
    assert len(data['buckets']) == 391
    for p in data['points']:
        assert dict(p['bucket_costs']) == bucket_training_costs(p, data['recipes'])
        assert p['bucket_evidence'] == dict(authority=p['authority'], evidence_sha256=p['evidence_sha256'])
    for b in data['buckets']:
        r = data['recipes'][b['recipe']]; lo,hi = b['m_bounds']
        assert lo <= hi and b['worst_regret_pct'] <= 5
        if r['route'] == 2:
            assert lo >= 128
        else:
            assert not r['tc']['ap'] and not r['tc'].get('legacy_adapters')
            if r['tc']['tm'] == 8 or r['route'] == 0:
                assert hi <= 64
        if hi >= 128:
            assert r['route'] in (1,2)
    assert bucket_interval([12,16,5120,5120]) == [9,31]
    assert bucket_interval([12,64,5120,5120]) == [64,127]


def test_conflicting_m_at_same_nk_split_instead_of_slow_shared_recipe():
    points = [dict(key=str(m), shape=[12,m,5120,5120], costs=costs)
              for m,costs in [(2048,[['a',10.],['b',20.]]),(3072,[['a',20.],['b',10.]])]]
    with pytest.raises(ValueError,match='cannot subdivide'):
        fit(points)
    buckets = fit(points, split_m=True)
    assert [(b['m_bounds'],b['recipe']) for b in buckets] == [([2048,2560],'a'),([2561,4095],'b')]


def test_fast_exact_tm8_cannot_escape_into_m65_127_bucket():
    recipes = {'light':dict(route=0,tc=dict(tm=8,ap=0)),
               'sf':dict(route=1,tc=dict(tm=32,ap=0)), 'full':dict(route=2,tc=None)}
    p = dict(shape=[12,64,5120,5120], costs=[['light',1.],['sf',2.],['full',3.]])
    assert bucket_training_costs(p,recipes) == {'sf':2.}
    assert dict(p['costs'])['light'] == 1.  # exact evidence is not edited
    recipes['sf8'] = dict(route=1,tc=dict(tm=8,ap=0))
    p = dict(shape=[12,16,5120,5120],costs=[['light',1.],['sf8',.5],['sf',2.]])
    assert bucket_training_costs(p,recipes) == {'light':1.,'sf':2.}


def test_later_exact_only_update_can_retain_its_prior_bucket_observation():
    # Synthetic authority only: exercise exporter validation, not admission of
    # a real new measurement or a performance claim.
    data = catalog(); p = data['points'][0]; old = copy.deepcopy(p)
    wrapper = copy.deepcopy(data['authorities'][p['authority']])
    wrapper['receipt']['host_test_fixture'] = True
    aid = digest(wrapper['receipt']); wrapper['receipt_sha256'] = aid
    data['authorities'][aid] = wrapper
    data['superseded_points'].append(old)
    p['authority'] = aid; p['evidence_sha256'] = digest('host-test-only')
    # The exact update cannot retrain a bucket with unreviewed timings. Its
    # older interval-safe mean costs are checked against preserved evidence.
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    validate(data)
    data['superseded_points'].pop()
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError,match='training observation'):
        validate(data)


def test_cross_card_confirmation_is_not_relabelled_as_one_physical_gpu():
    data = catalog()
    p = next(p for p in data['points'] if p['shape'] == [8,128,1024,4096])
    joint = data['authorities'][p['authority']]['receipt']
    assert joint['authority_kind'] == 'USER_AUTHORIZED_EQUIVALENT_DEVICE_COHORT'
    assert 'pci' not in joint['device']['device']
    assert len(joint['source_authorities']) == len(joint['physical_devices']) == 2
    sources = [data['authorities'][a]['receipt'] for a in joint['source_authorities']]
    assert sources[1]['confirmation_retry']['equivalent_cards_user_authorized'] is True
    assert sources[0]['device']['device']['pci'] != sources[1]['device']['device']['pci']


@pytest.mark.parametrize('fault',['bounds','training','authority','source'])
def test_resealed_new_bucket_or_source_corruption_fails(fault):
    data = catalog(); p = data['points'][0]
    if fault == 'bounds': data['buckets'][0]['m_bounds'][1] += 1
    elif fault == 'training': p['bucket_costs'][0][1] *= .5
    elif fault == 'authority': p['bucket_evidence']['evidence_sha256'] = '0'*64
    else: data['authorities'][p['authority']]['receipt']['source_revision']['quactlize_commit'] = 'f'*40
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    with pytest.raises(ValueError): validate(data)


def test_unseen_family_bucket_audit_keeps_losses_and_missing_measurements():
    audit = audit_buckets(catalog())
    assert sum(audit['counts'].values()) == 540
    assert audit['counts']['comparable'] == 116
    assert audit['counts']['candidate_not_measured_on_held_family'] == 76
    assert len(audit['over_5pct']) == 17 and 33 < audit['worst_regret_pct'] < 34
    assert not audit['new_model_validated']


def test_q8_exception_is_specific_and_does_not_change_raw_native_verdict():
    evidence = json.loads((ROOT/'policies/dense_smallm_native_admission.json').read_text())
    assert evidence['measured'] == 360 and evidence['raw_counts'] == {'PASS':359,'REGRESSION':1}
    assert evidence['m'] == list(range(1,9)) and evidence['families'] == 45
    assert len(evidence['accepted_regressions']) == 1
    e = evidence['accepted_regressions'][0]
    assert e['shape_q_m_n_k'] == [8,1,512,4096] and e['raw_status'] == 'REGRESSION'
    assert e['delta_us'] == pytest.approx(e['quactlize_us']-e['native_us'])
    assert 0 < e['delta_us'] < .072
    assert not evidence['global_tolerance_changed']
    assert not evidence['future_measurements_automatically_accepted']
    assert not evidence['strict_zero_regression'] and not evidence['model_tpot_admitted']
