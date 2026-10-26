#!/usr/bin/env python3
"""Retry missing Asys confirmation batches on an explicitly equivalent card.

Runs the ORIGINAL frozen harness and binaries, not the current checkout's
measurement code. Original receipts stay immutable; each new capture names
its real device. Requires an explicit user equivalence assertion. This is
component-screen recovery, not a new no-regression or model-performance gate.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import lzma
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


def read(path):
    return json.loads(Path(path).read_text())


def read_review(path):
    from quactlize.runtime.tuning import digest
    from tools.review_prefill_cohort import SCHEMA
    with lzma.open(path, 'rt') as stream:
        value = json.load(stream)
    if (value.get('schema') != SCHEMA or value.get('review_sha256') !=
            digest({k: v for k, v in value.items() if k != 'review_sha256'})):
        raise ValueError('previous compact review schema/hash differs')
    return value


def missing_jobs(point, partial, jobs):
    """Only whole missing batches; never silently remeasure valid captures."""
    if point.get('confirmation_captures') != 2 or partial.get('missing'):
        raise ValueError('confirmation recovery requires screened configs and a two-capture plan')
    expected = {c['id'] for c in point['cases']}
    if point.get('native_comparison', True):
        expected.add('native-fp16')
    rows = {r['case_id']: r for r in partial['measurements']}
    if len(rows) != len(partial['measurements']) or set(rows) != expected:
        raise ValueError('partial point/config inventory differs')
    absent = set()
    for key, row in rows.items():
        if row['status'] not in ('PASS', 'MISSING_CONFIRMATION'):
            raise ValueError('not a missing-confirmation failure: ' + key)
        captures = row.get('confirmation_captures', [])
        indices = {r['confirmation'] for r in captures}
        if (len(indices) != len(captures) or not indices <= {0, 1} or
                any(r['status'] != 'PASS' for r in captures) or
                (row['status'] == 'PASS') != (indices == {0, 1})):
            raise ValueError('invalid prior confirmation coverage')
        absent.update((key, i) for i in {0, 1} - indices)
    selected, scheduled = [], set()
    for cases, index in jobs:
        keys = {(c['id'], index) for c in cases}
        if not keys & absent:
            continue
        if not keys <= absent or keys & scheduled:
            raise ValueError('partial/duplicate batch; refusing to repeat valid confirmation samples')
        selected.append((cases, index))
        scheduled |= keys
    if not absent or scheduled != absent:
        raise ValueError('frozen batch plan does not exactly cover missing confirmations')
    return selected


def equivalent_authorities(old, new):
    from tools.review_prefill_cohort import domain
    # Physical identity is intentionally excluded, not source/SDK/geometry.
    if domain(old) != domain(new):
        raise ValueError('equivalent-card SDK/geometry/L2 domain differs')
    for key in ('name', 'compute_capability', 'total_memory'):
        if old['device']['device'].get(key) != new['device']['device'].get(key):
            raise ValueError('equivalent-card model attributes differ: ' + key)
    strip = lambda a: {k: v for k, v in a.items() if k not in ('device', 'confirmation_retry')}
    if strip(old) != strip(new):
        raise ValueError('confirmation source/payload/build/clock protocol differs')


def combine(result, original, retry, point_id):
    """Merge only missing directions; preserve every raw capture's authority."""
    from quactlize.runtime.tuning import digest
    from tools.review_prefill_cohort import confirmations, tables
    if len(original['authorities']) != 1 or len(retry['authorities']) != 1:
        raise ValueError('recovery expects one original worker and one retry worker')
    old_id, old = next(iter(original['authorities'].items()))
    new_id, new = next(iter(retry['authorities'].items()))
    equivalent_authorities(old, new)
    policy = new.get('confirmation_retry', {})
    if policy.get('parent_authority') != old_id or policy.get('equivalent_cards_user_authorized') is not True:
        raise ValueError('cross-device confirmation merge was not explicitly authorized')
    key = digest([old_id, point_id])
    prior = original['partial_points'][key]
    if result['partial_points'].get(key) != prior or point_id in result['points']:
        raise ValueError('previous review differs from original partial point')
    fresh = retry['partial_points'].get(digest([new_id, point_id])) or retry['points'].get(point_id)
    if not fresh or fresh['workload'] != prior['workload']:
        raise ValueError('retry workload differs')
    components = dict(original['components'], **retry['components'])
    candidates = dict(original['candidates'])
    for cid, definition in retry['candidates'].items():
        if cid in candidates and candidates[cid] != definition:
            raise ValueError('retry recipe differs')
        candidates[cid] = definition
    # This is a logical measurement cohort, NOT a pretend physical device.
    joint = copy.deepcopy(old)
    for field in ('host', 'ordinal', 'pci'):
        joint['device']['device'].pop(field, None)
    joint.update(authority_kind='USER_AUTHORIZED_EQUIVALENT_DEVICE_COHORT',
                 source_authorities=[old_id, new_id],
                 physical_devices=[old['device'], new['device']])
    joint_id = digest(joint)
    rows = {r['case_id']: copy.deepcopy(r['confirmation_captures']) for r in prior['measurements']}
    for captures in rows.values():
        for capture in captures:
            capture['authority'] = old_id
    for row in fresh['measurements']:
        if row['case_id'] not in rows:
            raise ValueError('unplanned retry config')
        captures = rows[row['case_id']]
        for capture in row['confirmation_captures']:
            if capture['confirmation'] in {c['confirmation'] for c in captures}:
                raise ValueError('retry duplicates a valid confirmation')
            captures.append(dict(capture, authority=new_id))
    merged = []
    for cid, captures in rows.items():
        normalized = {}
        for capture in captures:
            ids = []
            for component in capture['components']:
                definition = dict(components[component])
                if definition['authority'] != capture['authority']:
                    raise ValueError('raw capture/component authority differs')
                definition['authority'] = joint_id
                identity = digest(definition)
                components[identity] = definition
                ids.append(identity)
            normalized[capture['confirmation']] = dict(capture, components=ids)
        row = confirmations(normalized, 2)
        if row['status'] != 'PASS':
            raise ValueError('retry kernel/arithmetic/coverage differs: ' + cid)
        row.update(authority=joint_id,
                   confirmation_captures=sorted(captures, key=lambda r: r['confirmation']))
        merged.append(row)
    # Commit the in-memory merge only after all recipes/captures passed.
    point = {k: copy.deepcopy(v) for k, v in prior.items() if k not in
             ('point_id', 'missing', 'unconfirmed', 'measurements', 'authority')}
    point.update(authority=joint_id, measurements=merged,
                 cross_device_confirmations=True, source_authorities=[old_id, new_id])
    result['authorities'].update({old_id: old, new_id: new, joint_id: joint})
    result['components'].update(components)
    result['candidates'].update(candidates)
    result['points'][point_id] = point
    del result['partial_points'][key]
    result['diagnostics'].extend(retry['diagnostics'])
    result['history'].append(dict(point=point_id, action='FILL_MISSING_CONFIRMATIONS',
                                  previous_authority=old_id, retry_authority=new_id,
                                  replacement=joint_id, previous_partial=prior))
    result['pending_points'] = sorted({p['point_id'] for p in result['partial_points'].values()} - set(result['points']))
    result['incomplete'] = list(result['pending_points'])
    coverage = dict(result['points'])
    for p in result['partial_points'].values():
        coverage.setdefault(p['point_id'], dict(p, measurements=[]))
    result['exact'], result['buckets'] = tables(coverage)
    result.update(production_changed=False, performance_admitted=False,
                  no_regression_admitted=False, bucket_status='PENDING_USER_FEEDBACK')
    result['review_sha256'] = digest({k: v for k, v in result.items() if k != 'review_sha256'})
    return result


def frozen_runner(source):
    # Import the unchanged old harness and its transitive dependencies. In
    # particular profile_batch launches that file's ORIGINAL --worker entry.
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('original_prefill_asys', source / 'tools/run_prefill_model_asys.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--previous-review', type=Path, required=True)
    parser.add_argument('--point', required=True)
    parser.add_argument('--device', default='1')
    parser.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--equivalent-cards', action='store_true', help='user asserts same-model cards are performance-equivalent')
    parser.add_argument('--confirm-idle', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    a = parser.parse_args()
    if not a.device.isascii() or not a.device.isdigit():
        parser.error('--device requires one physical GPU ordinal')
    if not a.equivalent_cards or not (a.plan_only or a.confirm_idle):
        parser.error('requires --equivalent-cards and --confirm-idle (unless --plan-only)')
    root = a.previous_run.resolve(strict=True)
    if (root / 'results/authority.json').is_file():
        root /= 'results'
    source = (root.parent / 'source').resolve(strict=True)
    runner = frozen_runner(source)
    from quactlize.runtime.tuning import digest
    from quactlize.runtime.compiler import sha
    from dev.strategy.journal import atomic_json
    from tools.review_prefill_cohort import ingest
    old = read(root / 'authority.json')
    plan = read(root / 'plan.json')
    execution = read(root / 'execution-plan.json')
    if (runner.source_seal() != old['harness'] or plan['plan_sha256'] != old['plan'] or
            plan['plan_sha256'] != digest({k: v for k, v in plan.items() if k != 'plan_sha256'}) or
            execution['authority'] != digest(old)):
        raise ValueError('original frozen measurement source/plan changed')
    specs = [p for p in execution['points'] if p['id'] == a.point]
    if len(specs) != 1 or specs[0].get('native_comparison', True):
        raise ValueError('select exactly one Quactlize-only confirmation point')
    point = specs[0]
    original = ingest(root)
    result = read_review(a.previous_review)
    partial_key = digest([digest(old), a.point])
    partial = original['partial_points'].get(partial_key)
    if not partial or result['partial_points'].get(partial_key) != partial:
        raise ValueError('previous review is not the retained worker partial')
    jobs = missing_jobs(point, partial, runner.confirmation_jobs(point))
    print('PREFILL_CONFIRMATION_PLAN ' + json.dumps(dict(point=a.point, workload=point['workload'],
        batches=len(jobs), configs=sum(len(cs) for cs, _ in jobs), device=a.device,
        retained_points=len(result['points']), original_source=str(source),
        original_source_revision=old.get('source_revision'), production_changed=False)), flush=True)
    if a.plan_only:
        return 0
    if a.output is None:
        a.output = Path(tempfile.mkdtemp(prefix='kpack-confirmation-', dir='/workspace'))
    else:
        a.output = a.output.resolve()
        a.output.mkdir(parents=True, exist_ok=False)
    a.sdk = a.sdk.resolve(strict=True)
    a.llama_source = Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp')  # unused: Quactlize-only
    a.llama_build = a.llama_source / 'build-ci-32F-cublas'
    a.no_native = True
    a.reuse_build = Path(old['payload']['path'])
    a.cache = Path(old['cache'])
    a.l2_bytes = old['device']['l2']['bytes']
    a.asys = (a.sdk / 'asight/bin/asys').resolve(strict=True)
    from tools.run_native_components import environment
    from tools.run_kpack_llama_ab import device
    from dev.strategy.refresh_clock import request_lock, observe
    env = environment(a)
    env.update(CUDA_VISIBLE_DEVICES=a.device, GGML_CUDA_CUBLAS_ACCUMULATOR='f32')
    os.environ.update(env)
    new = copy.deepcopy(old)
    new['device'] = device(a)
    new['confirmation_retry'] = dict(parent_authority=digest(old), equivalent_cards_user_authorized=True,
                                    point=a.point, tool_sha256=sha(__file__))
    equivalent_authorities(old, new)
    if sha(a.asys) != old['asys_sha256']:
        raise ValueError('Asys binary differs from previous capture')
    # Copy only the tiny runner payload, not caches/weights/profiler reports.
    dense = a.output / 'build/dense'
    dense.mkdir(parents=True)
    for name in ('manifest.json', 'libquactlize_ppu_prefill.so'):
        shutil.copy2(root / 'build/dense' / name, dense / name)
    shutil.copy2(root / 'build/libpoint-timer.so', a.output / 'build/libpoint-timer.so')
    if (runner.validate_payload(a) != old['payload'] or
            sha(a.output / 'build/libpoint-timer.so') != old['build']['timer_sha256']):
        raise ValueError('original helper/timer payload changed')
    for name in ('plan.json', 'modules.json'):
        shutil.copy2(root / name, a.output / name)
    atomic_json(a.output / 'authority.json', new)
    atomic_json(a.output / 'execution-plan.json', dict(points=[point], authority=digest(new)))
    atomic_json(a.output / 'recovery.json', dict(previous_run=str(root), previous_review=str(a.previous_review),
                previous_review_sha256=result['review_sha256'], point=a.point,
                batches=[dict(confirmation=i, cases=[c['id'] for c in cs]) for cs, i in jobs]))
    smi = next((p for p in (Path(shutil.which('ppu-smi') or '/nonexistent'), a.sdk / 'bin/ppu-smi') if p.is_file()), None)
    if smi is None:
        raise ValueError('ppu-smi required for 1700MHz request')
    pci = new['device']['device']['pci']
    lock_path = Path('/tmp') / ('quactlize-components-' + digest(pci)[:24] + '.lock')
    print('PREFILL_CONFIRMATION_START output=' + str(a.output), flush=True)
    with lock_path.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock = request_lock(smi, pci, 1700)
        atomic_json(a.output / 'clock-lock.json', clock)
        if clock['returncode']:
            raise ValueError('1700MHz lock request failed')
        for cases, index in jobs:
            before = observe(smi, pci, 1700, advisory=True)
            runner.profile_batch(a, point, cases, new, env, index)
            atomic_json(a.output / ('clock-' + str(index) + '-' + cases[0]['id'] + '.json'),
                        dict(before=before, after=observe(smi, pci, 1700, advisory=True)))
    merged = combine(result, original, ingest(a.output), a.point)
    output = a.output / ('review-' + str(time.time_ns()) + '.json.xz')
    with lzma.open(output, 'xt', preset=6) as stream:
        json.dump(merged, stream, separators=(',', ':'))
    print('PREFILL_CONFIRMATION_DONE ' + json.dumps(dict(points=len(merged['points']),
        pending=merged['pending_points'], production_changed=False, output=str(output))), flush=True)
    print('results=' + str(output), flush=True)
    return int(bool(merged['pending_points']))


if __name__ == '__main__':
    raise SystemExit(main())
