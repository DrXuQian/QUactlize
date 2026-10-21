#!/usr/bin/env python3
"""Replay the two reviewed 1700-MHz cohorts and emit the active exact table.

No timings are transferred between cards. The six unresolved keys retain
their old recipe AND old cost estimates; their fresh observations are kept
separately, never promoted to a stable performance claim.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.prefill_closure import read_compact, require
from dev.strategy.prefill_refresh import bind, load_seed, same_recipe
from dev.strategy.prefill_review import audit_point, choose, component
from quactlize.dispatch.prefill_policy import EVIDENCE, HEADER, LEGACY
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.fit_prefill_confirmation import header as legacy_header

ARCHIVES = ('2d8dfa8970a7bbb28afdb58d30404712fc81684a5836ce9d6a3d5419e4ac17c5',
            'b446a32693f78d3e81a009e5d17296ed6324eac4286a4001bc22d7dd8bb51803')


def decide(point, measured):
    qualified = [a for a, v in point['arms'].items() if v['qualified']]
    try:
        return choose(point, measured, qualified), 'CONFIRMED'
    except ValueError:
        stats, original = measured['statistics'], point['original']
        s = stats[original]
        others = [v for a, v in stats.items() if a != original]
        if (point['arms'][original]['recipe']['kind'] == 'full' and
                measured['status'] == 'UNSTABLE' and
                all(max(s['rounds_us']) < min(v['rounds_us']) for v in others)):
            reason = 'RETAIN_CURRENT_FULL_TIMING_STABILITY_OPEN'
        else:
            gain = 100 * (1 - min(v['median_us'] for v in others) / s['median_us'])
            wins = [sum(x < y for x, y in zip(v['rounds_us'], s['rounds_us'])) for v in others]
            require(point['workload']['operation'] == 'indexed' and abs(gain) < .1 and wins == [3],
                    'unreviewed unresolved selection')
            reason = 'RETAIN_CURRENT_SUBPERCENT_TIE_PAIRED_WINS_3_OF_6'
        return dict(winner=original, eligible=[], qualified=qualified), reason


def cohort(path, expected):
    require(sha(path) == expected, 'unreviewed archive')
    rows, authorities = {}, []
    with tarfile.open(path) as tar:
        names = set()
        for member in tar:
            p = PurePosixPath(member.name)
            require(member.isfile() and not p.is_absolute() and '..' not in p.parts and
                    member.name not in names, 'unsafe/duplicate archive member')
            names.add(member.name)

        def read(name):
            return json.load(tar.extractfile(name))

        plan, source, summary = map(read, ('finalists.json', 'source.json', 'summary.json'))
        require(plan['plan_sha256'] == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
                'plan seal differs')
        require(summary['status'] == 'COMPLETE' and summary['measured'] == len(plan['points']),
                'measurement coverage incomplete')
        # Routing may change here, but not the implementation that was timed.
        prefixes = ('quactlize/include/', 'quactlize/runtime/', 'quactlize/decode/',
                    'quactlize/execution/', 'quactlize/prefill/', 'quactlize/integrations/')
        for name, checksum in source['files'].items():
            if name.startswith(prefixes):
                require(sha(ROOT/name) == checksum, 'measured implementation changed: ' + name)
        for shard in range(len(plan['cards'])):
            auth = read(f'shard-{shard}/authority.json')
            resources = read(f'shard-{shard}/resources.json')
            auth_sha = digest(auth)
            require(resources['authority'] == auth_sha and auth['plan'] == plan['plan_sha256'] and
                    auth['device']['compute_units'] == 72 and auth['l2']['bytes'] == 67108864 and
                    auth['source'] == source['sha256'] and not auth['previous_timings_reused'],
                    'device/source authority differs')
            compact, = [n for n in names if n.startswith(f'shard-{shard}/review-') and n.endswith('.xz')]
            tables, groups = read_compact(tar.extractfile(compact))
            components = {c['id']: component(c, tables) for c in groups['component']}
            require(len(components) == len(groups['component']), 'duplicate component')
            rounds = groups['prefill_confirmation_round']
            for original in plan['points']:
                if original['shard'] != shard:
                    continue
                point = bind(original, resources['queries'])
                pid = point['id']
                measured = read(f'shard-{shard}/results/{pid}.json')
                rr = sorted((r for r in rounds if r['point'] == pid), key=lambda r: r['round'])
                accuracy = audit_point(point, measured, rr, components, auth_sha)
                for r in rr:
                    require(set(r['jobs']) == set(point['jobs']), 'job denominator differs')
                    for aid, arm in point['arms'].items():
                        require([c for j in arm['jobs'] for c in r['jobs'][j]['components']] ==
                                [c['id'] for c in measured['components'][aid][r['round']]],
                                'helper/producer/reducer composition changed')
                rows[pid] = (point, measured, accuracy, [r['record_sha256'] for r in rr], auth_sha,
                             resources['queries'])
            authorities.append(dict(archive_sha256=expected, shard=shard, sha256=auth_sha,
                device=auth['device'], sdk=auth['sdk'], helpers=auth['helpers'], dequant=auth['dequant'],
                body_source=auth['body_source'], providers_sha256=digest(auth['providers'])))
    require(len(rows) == len(plan['points']), 'duplicate/missing points')
    return plan, rows, authorities


def build(old_path, new_path):
    old_plan, old, authorities = cohort(old_path, ARCHIVES[0])
    new_plan, new, aa = cohort(new_path, ARCHIVES[1])
    authorities += aa
    retained = set(new_plan['retained_point_ids'])
    require(len(old) == 430 and len(new) == 123 and len(retained) == 307 and
            not set(new) & retained and set(new) | retained == set(old), 'cohort coverage differs')
    legacy = json.loads((ROOT/(LEGACY+'.json')).read_text())
    prior = {p['point']: p for p in legacy['points']}
    points = []
    for pid in sorted(old):
        point, measured, accuracy, receipts, auth, queries = new[pid] if pid in new else old[pid]
        if pid not in new:
            require(old_plan['cards'][point['shard']] not in ('6', '7'), 'suspect card reused')
        decision, status = decide(point, measured)
        require(pid in new or status == 'CONFIRMED', 'unconfirmed old observation reused')
        aid = decision['winner']
        recipe = point['arms'][aid]['recipe']
        resource, launch = None, None
        if recipe['kind'] == 'tc':
            job, = [point['jobs'][j] for j in point['arms'][aid]['jobs'] if point['jobs'][j]['role']=='tc']
            resource = queries[job['arm']['binding']['query']]
            launch = job['arm']['binding']['launch']
        parts = measured['components'][aid]
        expansion = [math.fsum(c['median_us'] for c in r if c['kind'] == 'expansion') for r in parts]
        core = [math.fsum(c['median_us'] for c in r if c['kind'] != 'expansion') for r in parts]
        confirmed = status == 'CONFIRMED'
        points.append(dict(point=pid, workload=point['workload'], original=point['original'],
            selected=aid, recipe=recipe, decision=decision, refresh_status=status, timing_confirmed=confirmed,
            changed=aid != point['original'], raw_status=measured['status'], raw_winner=measured['winner'],
            statistics=measured['statistics'], arms={a: {k: v[k] for k in ('original', 'roles', 'qualified')}
                for a, v in point['arms'].items()}, component_rounds=parts, accuracy=accuracy,
            gemm_us=statistics.median(core) if confirmed else prior[pid]['gemm_us'],
            dequant_us=statistics.median(expansion) if confirmed else prior[pid]['dequant_us'],
            cost_basis='REFRESH_CONFIRMED_COMPONENTS' if confirmed else 'LEGACY_CONFIRMED_UNCHANGED_RECIPE',
            observed_core_us=statistics.median(core), observed_dequant_us=statistics.median(expansion),
            selected_resource=resource, selected_launch=launch,
            archive_sha256=ARCHIVES[int(pid in new)], authority=auth,
            result_sha256=digest(measured), round_receipts=receipts))
    result = dict(schema='quactlize.prefill-refresh-policy.v1', archives=list(ARCHIVES),
        baseline_sha256=sha(ROOT/(LEGACY+'.json')), seed_sha256=load_seed()['sha256'],
        authorities=authorities, points=points, timing='SEPARATE_COMPONENTS_CUBLAS_ORDINARY_EVENTS',
        clock='1700_MHZ_REQUESTED_THROTTLING_ADVISORY_NOT_SAMPLE_FILTER',
        scope='REFRESH_COMPONENT_SELECTION_SIX_CURRENT_RETAINED_NOT_MODEL_AB',
        model_performance_admitted=False, legacy_fallback=False,
        summary=dict(points=430, confirmed=sum(p['timing_confirmed'] for p in points),
            retained_pending=sum(not p['timing_confirmed'] for p in points),
            changes=sum(p['changed'] for p in points),
            routes=dict(Counter(p['recipe']['route'] for p in points)),
            raw_status=dict(Counter(p['raw_status'] for p in points)),
            max_relative_l1_error=max(p['accuracy']['relative_l1_error_max'] for p in points)))
    result['sha256'] = digest(result)
    validate(result)
    return result


def validate(evidence):
    require(evidence['sha256'] == digest({k: v for k, v in evidence.items() if k != 'sha256'}),
            'refresh evidence seal differs')
    require(evidence['archives'] == list(ARCHIVES) and
            evidence['baseline_sha256'] == sha(ROOT/(LEGACY+'.json')), 'baseline/archive identity differs')
    seed = load_seed()
    require(evidence['seed_sha256'] == seed['sha256'], 'shortlist changed')
    seeds = {p['id']: p for p in seed['points']}
    legacy = {p['point']: p for p in json.loads((ROOT/(LEGACY+'.json')).read_text())['points']}
    require(len(evidence['points']) == len({p['point'] for p in evidence['points']}) == 430 and
            {p['point'] for p in evidence['points']} == set(seeds), 'exact domain differs')
    for p in evidence['points']:
        pid, aid = p['point'], p['selected']
        require(Workload(**p['workload']).key == pid and p['workload'] == seeds[pid]['workload'] and
                p['original'] == seeds[pid]['incumbent'], 'workload/incumbent differs')
        require(set(p['arms']) == set(p['statistics']) == set(seeds[pid]['arms']), 'arm denominator differs')
        bound = {}
        for arm, value in p['arms'].items():
            seed_arm=seeds[pid]['arms'][arm]
            original=arm==p['original']
            roles=['rule'] if seed_arm['qualified'] else []
            if original or arm==seeds[pid]['historical_original'] or 'control' in seed_arm['historical_roles']:
                roles+=['control']
            if original:roles+=['original']
            require(value==dict(original=original,roles=roles,qualified=seed_arm['qualified']),
                    'rule qualification/control roles changed')
            bound[arm] = dict(value, recipe=seed['recipes'][seed_arm['recipe']])
        decision, status = decide(dict(p, arms=bound), dict(p, status=p['raw_status']))
        require(decision == p['decision'] and aid == decision['winner'] and status == p['refresh_status'] and
                p['timing_confirmed'] == (status == 'CONFIRMED'), 'selection gate differs')
        require(p['recipe'] == seed['recipes'][seeds[pid]['arms'][aid]['recipe']], 'selected recipe changed')
        if p['recipe']['kind']=='tc':
            from dev.strategy.resources import module_spec
            from dev.strategy.space import resolved_grid
            resource=p['selected_resource'];r=Recipe.from_dict(p['recipe']);w=Workload(**p['workload'])
            require(resource['status']=='READY' and resource['rc']==0 and resource['occupancy']>0 and
                    resource['device']['compute_units']==72 and resource['request']['module']==module_spec(w,r),
                    'selected resource identity differs')
            require(p['selected_launch']==resolved_grid(w,r,72,resource['occupancy']),
                    'measured launch differs from selected recipe')
        else:
            require(p['selected_resource'] is None and p['selected_launch'] is None, 'FULL cannot carry a TC ticket')
        require(p['changed'] == (aid != p['original']) and (not p['changed'] or p['timing_confirmed']),
                'unconfirmed recipe promotion')
        proof = p['accuracy']
        require(proof['checks'] > 0 and proof['bad'] == proof['nonfinite'] == 0 and
                0 <= proof['relative_l1_error_max'] < .005, 'numerical proof differs')
        parts = p['component_rounds']
        require(len(parts) == len(p['round_receipts']) == 6 and all(parts), 'rounds missing')
        costs = [math.fsum(c['median_us'] for c in r) for r in parts]
        require(costs == p['statistics'][aid]['rounds_us'], 'selected composition costs differ')
        for kind, observed, cost in (('expansion', 'observed_dequant_us', 'dequant_us'),
                                    ('core', 'observed_core_us', 'gemm_us')):
            values = [math.fsum(c['median_us'] for c in r if (c['kind'] == 'expansion') == (kind == 'expansion'))
                      for r in parts]
            require(p[observed] == statistics.median(values), 'component cost differs')
            require(p[cost] == (p[observed] if p['timing_confirmed'] else legacy[pid][cost]),
                    'unconfirmed timing promoted')
        if not p['timing_confirmed']:
            require(same_recipe(Recipe.from_dict(p['recipe']), Recipe.from_dict(legacy[pid]['recipe'])),
                    'pending point changed current recipe')
    s = evidence['summary']
    require(s['confirmed'] == sum(p['timing_confirmed'] for p in evidence['points']) == 424 and
            s['changes'] == sum(p['changed'] for p in evidence['points']) == 94 and
            s['retained_pending'] == 6 and s['routes'] == {'fq': 21, 'sf': 236, 'full': 173},
            'reviewed decision denominator differs')
    require(s['routes']==dict(Counter(p['recipe']['route'] for p in evidence['points'])) and
            s['raw_status']==dict(Counter(p['raw_status'] for p in evidence['points'])) and
            s['max_relative_l1_error']==max(p['accuracy']['relative_l1_error_max'] for p in evidence['points']),
            'summary differs from point evidence')


def header(evidence):
    validate(evidence)
    # Keep Config/ABI/lookup layout unchanged. Timing admission is metadata,
    # not an extra runtime route or a reason to choose an unmeasured recipe.
    return legacy_header(evidence).replace('tools/fit_prefill_confirmation.py', 'tools/fit_prefill_refresh.py').replace(
        'quactlize_prefill_confirmation_v1', 'quactlize_prefill_refresh_v1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', nargs=2, type=Path)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    evidence = build(*args.archives) if args.archives else json.loads(EVIDENCE.read_text())
    generated = header(evidence)
    text = json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n'
    if args.check:
        require(EVIDENCE.read_text() == text and HEADER.read_text() == generated, 'generated refresh differs')
    else:
        EVIDENCE.write_text(text)
        HEADER.write_text(generated)
    print('PREFILL_REFRESH_POLICY ' + json.dumps(evidence['summary']))


if __name__ == '__main__':
    main()
