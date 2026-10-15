"""Host-only evidence/retry tests; synthetic timings are not device results."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import lzma
from pathlib import Path

import pytest

from dev.strategy.components import component_key, statistics_checked
from dev.strategy.journal import ArmJournal, atomic_json
from dev.strategy.prefill_retry import (
    CLASSIFIER_BEFORE, PrefillJournal, import_passes, previous_probe, prior_root,
    read, reuse_build, same_inputs,
)
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.run_kpack_strategy_prefill import jobs, summarize

ROOT = Path(__file__).resolve().parents[1]


def sealed(path, value):
    value = deepcopy(value)
    value['record_sha256'] = digest(value)
    atomic_json(path, value)
    return value


def component(root, w, recipe, authority):
    node = dict(name='void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<X>>(Params)',
                symbol='native-test-symbol', grid=[32, 1, 1], block=[128, 1, 1])
    cid, identity = component_key(w, recipe, node, authority=dict(sha256=digest(authority)),
                                 source='host-test-only', cache='cold')
    row = statistics_checked([[3.0]*3]*3)
    row.update(identity=identity, component_id=cid, kind='producer', timed_kernel_types=1,
               calls_per_graph=16, first_use_excluded=True, host_launch_cost_added=False)
    sealed(root/'components'/(cid+'.json'), row)
    return dict(correctness='PASS', components=[cid], numerical=[dict(
        bad=0, nonfinite=0, relative_l1_error=0., cells=w.m*w.n, tolerance=.005)])


def fixture(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    w = Workload(12, 256, 512, 128)
    r = Recipe.make('tc', 'fq', split=1)
    arms = [dict(id=aid, recipe=asdict(r), roles=[role], aliases=[])
            for aid, role in [('original', 'original'), ('b'*64, 'rule'), ('c'*64, 'control')]]
    p = dict(id=w.key, workload=asdict(w), arms=arms, sf_configs=[], full_configs=[])
    plan = json.loads(json.dumps(dict(points=[p], blocked=[], counts=dict(requested_points=1),
                                      scope='HOST_TEST', excluded=[])))
    plan['plan_sha256'] = digest(plan)
    authority = dict(plan=plan['plan_sha256'], resource=dict(device='same-card', sdk='same-sdk', cache='cache'),
        harness=dict(sha256='old-harness'), body_source='body', tc={'module':'image'},
        oracle='oracle', helpers='helpers', dequant='dequant', l2=dict(bytes=64),
        protocol='3x3_ONE_KERNEL_PER_TIMER_NO_LAUNCH_PENALTY', scope='HOST_TEST')
    atomic_json(old/'authority.json', authority)
    atomic_json(old/'execution-plan.json', plan)
    atomic_json(old/'plan/plan.json', dict(plan_sha256='unused-in-this-fixture'))
    atomic_json(old/'summary.json', dict(status='INCOMPLETE'))
    old_journal = ArmJournal(old/'arms', dict(sha256=digest(authority)), 'a'*32)
    data = component(old, w, r, authority)
    original, failed, pending = plan['points'][0]['arms']
    old_journal.start(p, original); old_journal.finish(p, original, 'PASS', data=data)
    old_journal.start(p, failed); old_journal.finish(p, failed, 'FAIL', error='unknown reducer')
    current = dict(authority, harness=dict(sha256='new-harness'))
    atomic_json(new/'authority.json', current)
    return old, new, plan, authority, current


def snapshot(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_retry_preserves_old_receipts_samples_and_authority_and_skips_pass(tmp_path):
    old, new, plan, before, current = fixture(tmp_path)
    untouched = snapshot(old)
    index = import_passes(old, new, plan, current, jobs)
    assert index['passed_jobs'] == 1 and snapshot(old) == untouched
    journal = PrefillJournal(new/'arms', dict(sha256=digest(current)), 'b'*32)
    p = plan['points'][0]
    original, failed, pending = p['arms']
    assert not journal.due(p, original)
    assert journal.due(p, failed) and journal.due(p, pending)
    assert journal.read(p, original)['identity']['authority'] == {'sha256':digest(before)}
    for arm in (failed, pending):
        journal.start(p, arm)
        journal.finish(p, arm, 'PASS', data=component(new, Workload(**p['workload']),
                                                   Recipe.from_dict(arm['recipe']), current))
    assert all(not journal.due(p, arm) for arm in p['arms'])
    rows = summarize(new, plan, journal)
    assert rows[0]['status'] == 'PASS'
    assert set(rows[0]['selection']['costs_us']) == {'original', 'b'*64, 'c'*64}
    assert snapshot(old) == untouched
    assert import_passes(old, new, plan, current, jobs) == index
    resumed = PrefillJournal(new/'arms', dict(sha256=digest(current)), 'c'*32)
    assert all(not resumed.due(p, arm) for arm in p['arms'])


@pytest.mark.parametrize('key', ['resource', 'body_source', 'tc', 'oracle', 'helpers', 'dequant', 'l2', 'protocol'])
def test_retry_rejects_changed_implementation_or_device_authority(tmp_path, key):
    old, new, plan, _, current = fixture(tmp_path)
    bad = dict(current, **{key:'changed'})
    with pytest.raises(ValueError, match='authority changed'):
        import_passes(old, new, plan, bad, jobs)
    assert not (new/'prior/index.json').exists()


@pytest.mark.parametrize('damage', ['samples', 'seal', 'kind', 'missing', 'symlink'])
def test_retry_does_not_import_invalid_evidence(tmp_path, damage):
    old, new, plan, _, current = fixture(tmp_path)
    path = next((old/'components').glob('*.json'))
    row = json.loads(path.read_text())
    if damage == 'samples': row['samples_us'][0][0] = -1
    if damage == 'seal': row['median_us'] = 7
    if damage == 'kind': row['kind'] = 'prepare'
    if damage in ('samples', 'kind'):
        row.pop('record_sha256'); sealed(path, row)
    elif damage == 'seal': atomic_json(path, row)
    elif damage == 'missing': path.rename(path.with_suffix('.missing'))
    elif damage == 'symlink':
        raw = path.read_bytes(); path.rename(path.with_suffix('.saved'))
        other = tmp_path/'outside.json'; other.write_bytes(raw); path.symlink_to(other)
    with pytest.raises((ValueError, FileNotFoundError)):
        import_passes(old, new, plan, current, jobs)
    assert not (new/'prior/index.json').exists()


def test_imported_pass_cannot_be_remeasured_or_silently_disappear(tmp_path):
    old, new, plan, _, current = fixture(tmp_path)
    import_passes(old, new, plan, current, jobs)
    journal = PrefillJournal(new/'arms', dict(sha256=digest(current)), 'b'*32)
    p = plan['points'][0]; arm = p['arms'][0]
    with pytest.raises(ValueError, match='already complete'): journal.start(p, arm)
    path = new/'prior/arms'/p['id']/'original/current.json'
    path.rename(path.with_suffix('.held'))
    with pytest.raises(ValueError, match='imported PASS'): journal.read(p, arm)


def test_retry_resume_rechecks_copied_component_bytes_before_new_measurements(tmp_path):
    old, new, plan, _, current = fixture(tmp_path)
    import_passes(old, new, plan, current, jobs)
    path = next((new/'components').glob('*.json'))
    row = json.loads(path.read_text()); row['samples_us'][0][0] = 99.
    atomic_json(path, row)
    with pytest.raises(ValueError, match='digest differs'):
        import_passes(old, new, plan, current, jobs)


def closure(files):
    return dict(files=files, sha256=digest(files))


def test_only_exact_classifier_repair_and_orchestration_changes_are_allowed():
    source = (ROOT/'dev/strategy/components.py').read_bytes()
    old = closure({'dev/strategy/components.py':CLASSIFIER_BEFORE, 'timer.py':'same'})
    current = closure(dict(old['files'], **{'dev/strategy/components.py':hashlib.sha256(source).hexdigest(),
        'dev/strategy/prefill_retry.py':'new-orchestration'}))
    same_inputs(old, current, classifier_source=source)
    with pytest.raises(ValueError, match='measurement inputs changed'):
        same_inputs(old, closure(dict(current['files'], **{'timer.py':'different'})), classifier_source=source)
    changed = source.replace(b"'reducer'", b"'prepare'", 1)
    with pytest.raises(ValueError, match='reviewed BF16 reducer'):
        same_inputs(old, closure(dict(current['files'], **{
            'dev/strategy/components.py':hashlib.sha256(changed).hexdigest()})), classifier_source=changed)
    with pytest.raises(ValueError, match='closure digest'):
        same_inputs(dict(old, sha256='bad'), current, classifier_source=source)


def test_previous_probe_compares_plans_and_transitive_harness(tmp_path, monkeypatch):
    old, _, _, authority, _ = fixture(tmp_path)
    source = (ROOT/'dev/strategy/components.py').read_bytes()
    inputs = closure({'dev/strategy/components.py':CLASSIFIER_BEFORE})
    new_inputs = closure({'dev/strategy/components.py':hashlib.sha256(source).hexdigest()})
    authority['harness'] = inputs
    atomic_json(old/'authority.json', authority)
    old_plan = dict(points=['unchanged'], source_identity=inputs, plan_sha256='old')
    frozen = dict(old_plan, source_identity=new_inputs, plan_sha256='new')
    probe = dict(plan=old_plan, authority=authority['resource'])
    monkeypatch.setattr('dev.strategy.prefill_retry.load_probe', lambda *a, **k: probe)
    assert previous_probe(old, frozen, new_inputs, ROOT) == probe
    with pytest.raises(ValueError, match='inventory changed'):
        previous_probe(old, dict(frozen, points=['different']), new_inputs, ROOT)
    with pytest.raises(ValueError, match='measurement inputs changed'):
        previous_probe(old, frozen, closure(dict(new_inputs['files'], **{'oracle.py':'different'})), ROOT)


def test_helpers_are_verified_and_copied_not_recompiled(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    (old/'helpers').mkdir(parents=True)
    (old/'helpers/manifest.json').write_text('{}')
    (old/'helpers/libstrategy_prefill.so').write_bytes(b'HOST_TEST_NOT_A_REAL_LIBRARY')
    checked = []
    def verify(path, sdk):
        checked.append(path)
        assert (path/'libstrategy_prefill.so').read_bytes() == b'HOST_TEST_NOT_A_REAL_LIBRARY'
        return {'verified':'host-test-only'}
    reuse_build(old, new, 'helpers', tmp_path, verify)
    assert checked == [old/'helpers', new/'helpers']
    assert snapshot(old) == snapshot(new)


def test_retry_review_exports_old_and_new_authorities_without_relabelling(tmp_path):
    from tools.pack_kpack_strategy_review import main
    old, new, plan, before, current = fixture(tmp_path)
    import_passes(old, new, plan, current, jobs)
    journal = PrefillJournal(new/'arms', dict(sha256=digest(current)), 'b'*32)
    p = plan['points'][0]
    for arm in p['arms'][1:]:
        journal.start(p, arm)
        journal.finish(p, arm, 'FAIL', error='host-test remaining failure')
    summarize(new, plan, journal)
    atomic_json(new/'summary.json', dict(status='INCOMPLETE', prior_pass_jobs=1))
    output = tmp_path/'review.xz'
    assert main([str(new), '--output', str(output)]) == 0
    records = [json.loads(line) for line in lzma.decompress(output.read_bytes()).splitlines()]
    assert records[-1]['export_status'] == 'COMPLETE'
    assert next(r for r in records if r['type'] == 'prior_run')['authority_sha256'] == digest(before)
    original = next(r for r in records if r['type'] == 'arm' and r['id'] == 'original')
    assert original['measurement_origin'] == 'PRIOR_PASS_UNCHANGED'
    assert original['authority'] == {'sha256':digest(before)}
    assert next(r for r in records if r['type'] == 'run')['authority_sha256'] == digest(current)
    assert next(r for r in records if r['type'] == 'component')['samples_us'] == [[3.]*3]*3


def test_stopped_unfinished_arm_is_retried_and_chained_import_is_explicitly_declined(tmp_path):
    old, new, plan, _, current = fixture(tmp_path)
    assert prior_root(old) == old
    authority = json.loads((old/'authority.json').read_text())
    journal = ArmJournal(old/'arms', dict(sha256=digest(authority)), 'b'*32)
    point = plan['points'][0]; arm = point['arms'][2]
    journal.start(point, arm)
    (old/'summary.json').rename(old/'not-finished.json')
    assert prior_root(old) == old
    import_passes(old, new, plan, current, jobs)
    resumed = PrefillJournal(new/'arms', dict(sha256=digest(current)), 'c'*32)
    assert resumed.due(point, arm)
    with pytest.raises(ValueError, match='chained parser'): prior_root(new)
