"""Narrow migration from the strict-clock experiment; keep its raw run immutable."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from dev.strategy.refresh import family
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest

# The completed 50c168d experiment. No kernel, oracle, timer, fixture or
# candidate-generator changes are allowed by this clock-policy migration.
OLD_FILES = {
    'tools/run_decode_refresh.py': '95d874c4ea4360c9c00842a9d8a77f00c87c9ea25505c230f3de56318245bbbd',
    'dev/strategy/refresh_clock.py': 'a51e3b35ac7ab8949b918a64255449835007039f438daddb5ce8fd08f88d3d72',
}
NEW_FILES = {'dev/strategy/refresh_retry.py'}


def check_transition(old, new):
    for value in (old, new):
        if value['sha256'] != digest(value['files']):
            raise ValueError('clock retry harness seal differs')
    before, after = old['files'], new['files']
    if any(before.get(k) != v for k, v in OLD_FILES.items()):
        raise ValueError('clock retry requires the reviewed strict-clock source')
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    if not changed <= OLD_FILES.keys() | NEW_FILES:
        raise ValueError('clock retry changed non-clock measurement sources: '+','.join(sorted(changed)))
    if not OLD_FILES.keys() <= after.keys() or not NEW_FILES <= after.keys():
        raise ValueError('clock retry source closure is incomplete')


def sealed_read(path):
    value = json.loads(Path(path).read_bytes())
    if value['sha256'] != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('previous run/plan seal differs: '+str(path))
    return value


def make_retry(previous, full, current_harness, saved, authority_for):
    """Re-run incomplete points only, on their original physical-card ownership."""
    previous = Path(previous).resolve(strict=True)
    run, old = (sealed_read(previous/name) for name in ('run.json', 'plan.json'))
    if old != full or run['plan'] != full['sha256'] or 'clock_retry' in old:
        raise ValueError('clock retry workload/config inventory differs')
    check_transition(run['harness'], current_harness)
    by_family = {key: i for i, keys in enumerate(run['assignments']) for key in keys}
    pending, prior = [], []
    for point in full['points']:
        shard = by_family[family(Workload(**point['workload']))]
        path = previous/('gpu-'+run['cards'][shard])/'points'/(point['id']+'.json')
        row = saved(path, authority_for(run, shard), point)
        if row and row['status'] not in ('PASS', 'FAIL'):
            raise ValueError('unknown previous point status')
        prior.append(dict(point=point['id'], shard=shard,
                          status=row['status'] if row else 'MISSING',
                          sha256=row['sha256'] if row else None))
        if not row or row['status'] != 'PASS':
            pending.append(point)
    if not pending:
        raise ValueError('previous run has no incomplete points; no GPU retry needed')
    plan = deepcopy(full)
    plan['points'] = pending
    needed = {a['module'] for p in pending for a in p['arms']}
    for kind in ('tc', 'vectors'):
        plan[kind] = {k: v for k, v in plan[kind].items() if k in needed}
    plan['counts'].update(
        points=len(pending), families=len({family(Workload(**p['workload'])) for p in pending}),
        arms=sum(len(p['arms']) for p in pending), max_arms=max(len(p['arms']) for p in pending),
        tc_modules=len(plan['tc']), vector_modules=len(plan['vectors']),
        operations=dict(Counter(p['workload']['operation'] for p in pending)))
    plan['clock_retry'] = dict(
        previous_output=str(previous), previous_run=run['sha256'], previous_plan=old['sha256'],
        previous_points=prior, previous_passed=len(full['points'])-len(pending),
        original_expected=len(full['points']),
        raw_results_modified=False, previous_passes_remeasured=False,
        previous_clock_policy='STRICT_OBSERVATIONS_NO_LOCK_REQUEST',
        scope='INCOMPLETE_POINTS_ONLY_NEW_CLOCK_POLICY_NOT_A_NEW_FULL_SWEEP')
    plan['sha256'] = digest({k: v for k, v in plan.items() if k != 'sha256'})
    return plan


def previous_run(plan):
    retry = plan.get('clock_retry')
    if not retry:
        return None
    run = sealed_read(Path(retry['previous_output'])/'run.json')
    if run['sha256'] != retry['previous_run'] or run['plan'] != retry['previous_plan']:
        raise ValueError('clock retry parent run changed')
    return run


def cache_root(output, plan):
    retry = plan.get('clock_retry')
    return Path(retry['previous_output']) if retry else Path(output)
