"""Sealed production catalog validation/generation; no device experiments."""
from collections import Counter


import json


import math


from pathlib import Path, PurePosixPath


import statistics


ROOT = Path(__file__).resolve().parents[1]


from quactlize.dispatch.catalog_components import require


from quactlize.dispatch.catalog_prefill import load_seed, same_recipe


from quactlize.dispatch.catalog_prefill_decision import choose


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
            from quactlize.dispatch.catalog_module import module_spec
            from quactlize.dispatch.catalog_geometry import resolved_grid
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
