"""Reviewed concat recipes reach production selectors; no host timing claims."""
import copy
import json
import pytest

from quactlize.dispatch import catalog_confirmation as confirmation
from quactlize.dispatch.dense_prefill import EVIDENCE as PREFILL, validate as validate_prefill
from quactlize.dispatch.planning import plan_smallm
from quactlize.execution import moe_decode_policy, token_cta_policy
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as generate


@pytest.fixture(scope='module')
def data():
    small = token_cta_policy.load()
    large = json.loads(PREFILL.read_bytes())
    return small, large


def confirmations(data):
    small, large = data
    return [r['selected']['confirmation'] for r in small['rows'] if r['cohort'] in ('concat','dense-concat')] + [
        p['model_confirmation'] for p in large['points'] if 'model_confirmation' in p]


def test_complete_exact_projection_scope_and_mean_components(data):
    rows = confirmations(data)
    ms = [*range(1,9),16,32,64,128,256,512,1024,2048,4096,5120]
    assert len(rows) == 125
    families = [(8,1024,2048),(8,2048,3072),(12,51200,5120),*[(q,34816,5120) for q in (11,12,13,14)]]
    assert {(r['workload']['qtype'],r['workload']['m'],r['workload']['n'],r['workload']['k']) for r in rows} == {
        (q,m,n,k) for m in ms for q,n,k in families} - {(14,64,34816,5120)}
    for value in rows:
        confirmation.validate(value)
        assert value['mean_us'] <= value['baseline_us']
        assert not value['contract']['activation_included']
    large = data[1]
    validate_prefill(large)
    assert sum('model_confirmation' in p for p in large['points']) == 69
    assert all(p['bucket_costs'] == [] for p in large['points'] if 'model_confirmation' in p)
    assert all(not b['holdout_validated'] for b in large['buckets'])


def test_smallm_changes_only_exact_entries_not_previous_donors(data, monkeypatch):
    current = generate.effective(json.loads(generate.EVIDENCE.read_bytes()))
    old = copy.deepcopy(data[0]); old['rows'] = [r for r in old['rows'] if r['cohort'] not in ('concat','dense-concat')]
    moe = moe_decode_policy.load()
    with monkeypatch.context() as m:
        # Replay the prior admitted token rows, without rewriting frozen
        # provenance or requiring the earlier Git checkout to exist in CI.
        m.setattr(token_cta_policy, 'load', lambda:old)
        m.setattr(moe_decode_policy, 'load', lambda:moe)
        previous = generate.effective(json.loads(generate.EVIDENCE.read_bytes()))
    expected = {tuple(r['key']) for r in data[0]['rows'] if r['cohort'] in ('concat','dense-concat') and
                not r['selected']['config'].get('legacy_adapters')}
    before = {tuple(r['key']):r for r in previous['exact']}
    after = {tuple(r['key']):r for r in current['exact']}
    assert len(before) == 2095 and len(after) == 2135
    assert {k for k in after if before.get(k) != after[k]} == expected
    assert current['buckets'] == previous['buckets']
    donor = lambda r:r.get('bucket_config',r['guard'] if r['config']['kind'] == 'q4' else r['config'])
    for key,row in before.items():
        assert donor(row) == donor(after[key])
        assert row['guard'] == after[key]['guard']
    assert {k:v for k,v in current.items() if k != 'exact'} == {
        k:v for k,v in previous.items() if k != 'exact'}


def test_real_cpp_smallm_selects_measured_geometry_and_body(tmp_path, data):
    rows = [r for r in data[0]['rows'] if r['cohort'] in ('concat','dense-concat')]
    result = plan_smallm(tmp_path, [r['key'] for r in rows], dense_fallbacks=True)['requests']
    counts = dict(simt=0,tc=0,token=0,address=0,shipping=0)
    for measured, actual in zip(rows,result,strict=True):
        chosen = measured['selected']; config = chosen['config']; impl = chosen['implementation']
        native = config.get('legacy_adapters',False)
        assert actual['status'] == 0 and actual['policy'] == (17 if native else 18)
        assert actual['native_f16'] == native
        assert actual['donor'] == ([0,0,0] if native else measured['key'][2:4]+[measured['key'][7]])
        if config['kind'] == 'tc':
            counts['tc'] += 1
            assert actual['kind'] == 0 and actual['split'] == config['split']
            assert actual['parent']['symbol'] == config['symbol']
            for k in ('qtype','route','tm','tn','tk','wm','wn','stages','ap','dn'):
                assert actual['parent'][k] == config[k]
        else:
            counts['simt'] += 1
            assert actual['kind'] == 1
            assert actual['config'] == {k:config[k] for k in token_cta_policy.FIELDS}
            if impl['kind'] == 'token-cta':
                counts['token'] += 1
                assert actual['implementation']['producer'] == 'token-cta'
                assert actual['implementation']['cta_tokens'] == measured['key'][7]
                assert actual['implementation']['cta_window'] == impl['window']
            elif impl['kind'] == 'q8-address':
                counts['address'] += 1
                assert actual['implementation']['producer'] == 'q8-address'
                assert actual['implementation']['q8_body'] == impl['body']
            else:
                counts['shipping'] += 1
                assert impl['kind'] == 'shipping'
    assert counts == dict(simt=21,tc=35,token=15,address=3,shipping=3)


def test_partial_q6_point_retains_confirmed_current_choice(data):
    from ci.support.policy_lookup import policy_lookup
    from quactlize.dispatch.table import Workload
    large = data[1]
    assert not any(p['shape'] == [14,64,34816,5120] for p in large['points'])
    coverage, rid = policy_lookup(large, Workload(14,34816,5120,64))
    assert coverage == 'BUCKET_PREDICTION'
    chosen = large['recipes'][rid]
    assert chosen['route'] == 1 and chosen['io'] == 3 and chosen['dequant'] == 0
    assert chosen['tc']['symbol'] == 'sf_q14_a0_tm64_tn64_tk128_wm64_wn32_s2_bc0_ap0_dn16'
    assert chosen['tc']['split'] == 1 and chosen['tc']['grid_mode'] == 0
    retained = large['model_tuning_updates'][-1]['retained']
    assert retained[0]['shape'] == [14,64,34816,5120]
    assert retained[0]['pending_confirmation_configs'] == 7


@pytest.mark.parametrize('fault', ['scope','dtype','source','mean','helper','proof','seal'])
@pytest.mark.parametrize('cohort', ['concat','dense-concat'])
def test_edited_compact_confirmation_is_rejected(data, fault, cohort):
    value = copy.deepcopy(next(r['selected']['confirmation'] for r in data[0]['rows'] if r['cohort'] == cohort))
    if fault == 'scope': value['workload']['n'] *= 4
    elif fault == 'dtype': value['workload']['compute'] = 'bf16'
    elif fault == 'source': value['provenance']['quactlize_commit'] = '0'*40
    elif fault == 'mean': value['mean_us'] /= 2
    elif fault == 'helper': value['components'][0]['mean_us'] /= 2
    elif fault == 'proof': value['proof']['sha256'] = ''
    if fault != 'seal': value['sha256'] = digest({k:v for k,v in value.items() if k != 'sha256'})
    else: value['sha256'] = '0'*64
    with pytest.raises(ValueError): confirmation.validate(value)
