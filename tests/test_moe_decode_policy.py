"""Fresh cross-router MoE admission and actual host selectors; no GPU timing."""
import copy
import ctypes as C
import json
import shutil
import subprocess

import pytest

from quactlize.execution import moe_decode_policy as binding
from quactlize.runtime.tuning import digest
from tools import generate_decode_selection as gen

ROOT = gen.ROOT


@pytest.fixture(scope='module')
def table():
    return binding.load()


def baseline():
    return gen.effective(json.loads(gen.EVIDENCE.read_bytes()), include_moe=False)


def measurement_catalog(table):
    """Reconstruct the exact measured controls without Git history/network."""
    result = baseline()
    exact = {tuple(r['key']):r for r in result['exact']}
    for row in table['rows']:
        old = exact[tuple(row['key'])]
        old['bucket_config'] = copy.deepcopy(old.get('bucket_config',
            old['guard'] if old['config']['kind'] == 'q4' else old['config']))
        old['bucket_dequant_config'] = old.get('bucket_dequant_config',
            -1 if old['config']['kind'] == 'q4' else old.get('dequant_config', -1))
        old['config'] = copy.deepcopy(row['original_config'])
        old['policy'] = row['original_policy']
        if old['policy'] == 18:
            old['status'] = 'MEASURED_MOE_SINGLE_ROW'
        if old['config']['kind'] != 'q4':
            old['guard'] = copy.deepcopy(old['config'])
    return result


def test_complete_model_scope_and_no_router_specific_winner_leak(table):
    assert len(table['rows']) == 32
    assert sum(len(r['profiles']) for r in table['rows']) == 144
    assert sum(r['measurement_changed'] for r in table['rows']) == 9
    assert sum(r['config']['kind'] == 'q4' for r in table['rows']) == 16
    assert sum(r['config']['kind'] == 'simt' for r in table['rows']) == 16
    assert not table['runtime_router_readback'] and not table['bucket_changes']
    assert not table['bf16_storage_admitted'] and not table['model_tpot_admitted']
    assert not table['native_no_regression_admitted']
    tc_wins = 0
    for row in table['rows']:
        assert row['selected'] == binding.choose(row['profiles'])
        for p in row['profiles']:
            selected = p['candidates'][row['selected']]
            original = p['candidates'][p['original']]
            assert selected['mean_us'] <= original['mean_us']
            assert binding.configuration(selected['recipe']) == row['config']
            best = min(p['candidates'].values(), key=lambda c:c['mean_us'])
            tc_wins += best['recipe']['kind'] == 'tc'
    assert tc_wins == 5  # kept visible, not turned into five router-aware rows


def test_refresh_only_measured_keys_and_preserve_every_bucket_donor(table):
    before = baseline()
    after = gen.effective(json.loads(gen.EVIDENCE.read_bytes()))
    keys = {tuple(r['key']) for r in table['rows']}
    assert len(before['exact']) == len(after['exact']) == 2095
    observed, config_changes = set(), set()
    for old, new in zip(before['exact'], after['exact']):
        assert old['key'] == new['key']
        if old != new:
            observed.add(tuple(old['key']))
        if old['config'] != new['config']:
            config_changes.add(tuple(old['key']))
        donor = lambda r:r.get('bucket_config', r['guard'] if r['config']['kind'] == 'q4' else r['config'])
        assert donor(old) == donor(new)
    assert observed == keys
    assert config_changes == {tuple(r['key']) for r in table['rows'] if r['changed']}
    assert {k:v for k,v in before.items() if k != 'exact'} == {k:v for k,v in after.items() if k != 'exact'}
    assert gen.header(after) == gen.OUTPUT.read_text()


def test_common_confirmation_and_full_mean_not_best_router_or_min_sample():
    def profile(original, fast, common, extra=False):
        values = {'old':dict(mean_us=original), 'tc':dict(mean_us=fast), 'simt':dict(mean_us=common)}
        if extra:
            values['unconfirmed_elsewhere'] = dict(mean_us=.1)
        return dict(original='old', candidates=values)
    points = [profile(10, 4, 9, True), profile(20, 25, 18)]
    assert binding.choose(points) == 'simt'  # TC loses another router.
    points[1]['candidates'].pop('simt')
    assert binding.choose(points) == 'old'   # Missing confirmation is not a win.
    points = [profile(10, 10, 10), profile(20, 20, 20)]
    assert binding.choose(points) == 'old'
    points[1]['original'] = 'tc'
    with pytest.raises(ValueError, match='same confirmed current control'):
        binding.choose(points)


def test_tc_is_not_excluded_by_type_and_router_regret_is_not_erased(table):
    tc = next(c['recipe'] for r in table['rows'] for p in r['profiles']
              for c in p['candidates'].values() if c['recipe']['kind'] == 'tc')
    config = binding.configuration(tc)
    assert config['kind'] == 'tc' and config['route'] == 2
    assert config['parent_persistent'] == dict(tc['parameters'])['persistent']
    profiles = [dict(original='simt', candidates={'simt':dict(mean_us=20), 'tc':dict(mean_us=15)}),
                dict(original='simt', candidates={'simt':dict(mean_us=30), 'tc':dict(mean_us=20)})]
    assert binding.choose(profiles) == 'tc'
    after = gen.effective(json.loads(gen.EVIDENCE.read_bytes()))
    rows = {tuple(r['key']):r for r in after['exact']}
    sensitive = [r for r in table['rows'] if r['worst_regret_pct'] > 5]
    assert len(sensitive) == 7
    assert all(rows[tuple(r['key'])]['status'] == 'ROUTER_SENSITIVE_PARETO' for r in sensitive)


@pytest.mark.parametrize('field', ['shape','config','mean','provenance','body','missing-profile','missing-candidate'])
def test_edited_or_resealed_evidence_is_rejected(table, field):
    bad = copy.deepcopy(table)
    row = next(r for r in bad['rows'] if r['measurement_changed'] and r['key'][7] > 1)
    p = row['profiles'][0]
    if field == 'shape': row['key'][7] = 9
    elif field == 'config': row['config']['columns'] *= 2
    elif field == 'mean': p['candidates'][row['selected']]['mean_us'] *= .5
    elif field == 'provenance': p['latency_provenance']['quactlize_commit'] = '0'*40
    elif field == 'body': bad['body_hashes']['quactlize/execution/simt_common.cuh'] = '0'*64
    elif field == 'missing-profile': row['profiles'].pop()
    else: del row['profiles'][0]['candidates'][row['selected']]
    bad['evidence_sha256'] = digest({k:v for k,v in bad.items() if k != 'evidence_sha256'})
    with pytest.raises(ValueError): binding.verify(bad)
    with pytest.raises((ValueError, KeyError)): binding.validate(bad)


def test_changed_base_cannot_silently_rebase_device_evidence(table):
    old = baseline()
    key = table['rows'][0]['key']
    next(r for r in old['exact'] if r['key'] == key)['config']['warps'] += 1
    with pytest.raises(ValueError, match='base catalog'): binding.apply(old)


def test_actual_cpp_transition_and_unseen_m_shape_guards(tmp_path, table):
    base = tmp_path/'base'
    base.mkdir()
    old = measurement_catalog(table)
    shutil.copytree(ROOT/'quactlize', base/'quactlize', ignore=shutil.ignore_patterns('__pycache__'))
    (base/'policies').mkdir()
    for header in (ROOT/'policies').glob('*.hpp'):
        shutil.copyfile(header, base/'policies'/header.name)
    (base/'policies'/gen.OUTPUT.name).write_text(gen.header(old))
    measured = {tuple(r['key']):r for r in table['rows']}
    keys = {tuple(r['key']) for r in old['exact']}
    for key in measured:
        for axis, values in [(0,[8,10,11,12,13,14]), (1,[0,1]), (2,[256,768,1024,1792,2304,4096]),
                             (3,[256,512,1024,2048,3072,4096]), (4,[128,512]), (5,[4,16]),
                             (6,[1,8]), (7,[0,1,2,3,4,5,6,7,8,9,16]), (8,[0])]:
            for value in values:
                modified = list(key)
                modified[axis] = value
                keys.add(tuple(modified))
    keys = sorted(keys)
    stdin = ''.join(' '.join(map(str,key))+'\n' for key in keys)
    results = []
    for i, root in enumerate((base, ROOT)):
        executable = tmp_path/str(i)
        subprocess.run(['g++','-std=c++17','-O2','-I'+str(root), ROOT/'tools/kpack_selection.cpp',
                        '-o', executable], check=True)
        results.append([json.loads(s) for s in subprocess.check_output([executable], input=stdin, text=True).splitlines()])
    changes = set()
    for key, a, b in zip(keys, *results):
        if key not in measured:
            assert a == b, (key,a,b)
            continue
        row = measured[key]
        assert b['status'] == 0 and b['policy'] == 18 and b['workspace_bytes'] == 0
        assert b['kind'] == (2 if row['config']['kind'] == 'q4' else 1)
        assert b['config'] == {k:v for k,v in row['config'].items() if k != 'kind'}
        if a['config'] != b['config']:
            changes.add(key)
    assert changes == {tuple(r['key']) for r in table['rows'] if r['measurement_changed']}


def test_actual_public_v3_and_q4_selectors_use_the_same_catalog(tmp_path, table):
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char kJitSource[]="";\n')
    binary = tmp_path/'public-control'
    subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(tmp_path),
        ROOT/'tests/moe_model_control_host.cpp','-ldl','-pthread','-o',binary], check=True)
    text = ''.join(' '.join(map(str,r['key']))+'\n' for r in table['rows'])
    values = [json.loads(s) for s in subprocess.check_output([binary], input=text, text=True).splitlines()]
    assert len(values) == 32
    for row, value in zip(table['rows'], values):
        config = row['config']
        fields = ('reader','variant','columns','warps','values') if config['kind'] == 'q4' else (
            'variant','columns','warps','values','split')
        assert value['status'] == 0 and value['kind'] == (2 if config['kind'] == 'q4' else 1)
        assert value['config'] == [config[f] for f in fields]


def test_q4_production_module_closure_retains_original_and_new_configs(table):
    from quactlize.execution.q4_decode_codegen import recipes, sources
    available = recipes()
    generated = sources()
    for row in table['rows']:
        if row['config']['kind'] != 'q4': continue
        n,k = row['key'][2:4]
        for c in (row['config'], row['original_config']):
            entry = tuple(c[f] for f in ('reader','variant','warps','values','columns'))
            assert entry in available[n,k]
            assert 'launch<'+','.join(map(str, (*entry,n,k,1)))+'>' in generated[f'q4decode_{n}_{k}.cu']


def test_stale_execution_manifest_cannot_hide_a_new_q4_specialization(tmp_path, table):
    from quactlize.dispatch.planning import plan_smallm, validate_inventory
    from quactlize.execution.q4_decode_codegen import recipes
    from quactlize.execution.simt_codegen import Config
    plan = plan_smallm(tmp_path, [r['key'] for r in table['rows']])
    simt = {tuple(r['config'][f] for f in gen.FIELDS) for r in table['rows'] if r['config']['kind'] == 'simt'}
    execution = dict(simt_configs={'13':[vars(Config(*c)) for c in sorted(simt)]},
        simt_compute_v2=dict(compute=['bf16'],formats=[13]), q4_decode_policy_sha256='host-test-only',
        q4_decode_configs={f'{n}x{k}':values for (n,k),values in recipes().items()},
        q4_decode_compute_v2=dict(compute=['bf16']))
    assert validate_inventory(plan,execution)['jit_required'] == []
    stale = copy.deepcopy(execution)
    stale['q4_decode_configs']['512x2048'].remove((2,7,4,4,8))
    with pytest.raises(ValueError, match='Q4 specialization absent'):
        validate_inventory(plan,stale)


def test_public_matched_abi_does_not_admit_bf16_storage(tmp_path, table):
    from quactlize.dispatch.native import Dispatch
    from quactlize.execution.native import Call, arrangement
    (tmp_path/'catalog.inc').write_text('static std::vector<Image> kImages; static char const kJitSource[]="";\n')
    subprocess.run(['g++','-std=c++17','-O2','-pthread','-shared','-fPIC','-Wl,-Bsymbolic',
        '-I'+str(tmp_path), ROOT/'quactlize/dispatch/binding.cpp','-ldl',
        '-o', tmp_path/'libquactlize_kpack_dispatch.so'], check=True)
    dispatch = Dispatch(tmp_path)
    try:
        for row in table['rows']:
            q,mode,n,k,e,top,ch,m,compute = row['key']
            call = Call(version=1,size=C.sizeof(Call),qtype=q,mode=mode,n=n,k=k,experts=e,
                topk=top,channels=ch,rows=m*top,input_type=2,a_row_stride=k,
                a_token_stride=k*ch,ids_stride=top,out_row_stride=n)
            assert dispatch.query_smallm_matched(call, arrangement(q), compute) is None
    finally:
        dispatch.close()
