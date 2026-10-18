"""Bounded confirmation plan and no-regression verdict, without legacy fallback."""
from dataclasses import asdict
import math
import statistics

from dev.strategy.components import statistics_checked
from dev.strategy.prefill_closure import require
from dev.strategy.space import producer_key
from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest

ROUNDS = 6


def bind_point(selected, previous):
    require(selected['id'] == previous['id'] and selected['workload'] == previous['workload'],
            'finalist workload differs from resource plan')
    old = {a['id']:a for a in previous['arms']}
    jobs, arms = {}, {}
    for aid, candidate in selected['arms'].items():
        source = old[candidate['source_arm']]
        recipe = Recipe.from_dict(candidate['recipe'])
        require(producer_key(recipe) == producer_key(Recipe.from_dict(source['recipe'])),
                'finalist body/schedule differs from measured source arm')
        p = dict(recipe.parameters)
        role = 'bf16' if recipe.kind == 'full' else 'tc'
        core_id = digest(['bf16']) if role == 'bf16' else digest([
            'tc', source['module'], source['binding']['launch']])
        core = dict(id=core_id, role=role, recipe=None if role == 'bf16' else asdict(recipe))
        if role == 'tc':
            core['arm'] = source
        jobs.setdefault(core_id, core)
        bindings = [core_id]
        helper = ('full_dequant' if recipe.kind == 'full' else
                  'sf' if recipe.route == 'sf' and selected['workload']['qtype'] != 8 else None)
        if helper:
            config = p['dequant_config']
            require(config >= 0, 'selected expansion config is not explicit')
            jid = digest([helper, config])
            jobs[jid] = dict(id=jid, role=helper, config=config, recipe=None)
            bindings.append(jid)
        arms[aid] = dict(candidate, jobs=bindings)
    require(0 < len(arms) <= 4 and len(jobs) <= 8, 'confirmation job bound exceeded')
    return dict(selected, arms=arms, jobs=jobs)


def ordered_jobs(point, round_index):
    require(0 <= round_index < ROUNDS, 'invalid confirmation round')
    keys = sorted(point['jobs'])
    return [point['jobs'][k] for k in (keys if round_index % 2 == 0 else reversed(keys))]


def verdict(point, rounds):
    """Six alternating rounds, separately measured stages, no old timings.

    Identical stages may share a measurement within a round. They are never
    copied into another round or converted to a whole-model latency claim.
    """
    require(len(rounds) == ROUNDS, 'six complete rounds required')
    costs = {a:[] for a in point['arms']}
    components = {}
    for index, row in enumerate(rounds):
        require(row['point'] == point['id'] and row['round'] == index and row['status'] == 'PASS',
                'missing/failed confirmation round')
        require(set(row['jobs']) == set(point['jobs']), 'confirmation job denominator differs')
        require(row['order'] == [j['id'] for j in ordered_jobs(point, index)],
                'finalists were not measured in alternating order')
        for job in row['jobs'].values():
            require(job['correctness'] == 'PASS' and job['components'], 'confirmation correctness missing')
            from tools.pack_kpack_strategy_review import accuracy
            proof = accuracy(job)
            require(proof['checks'] > 0 and proof.get('bad_max', 0) == 0 and
                    proof.get('nonfinite_max', 0) == 0 and
                    math.isfinite(proof.get('relative_l1_error_max', 0)) and
                    0 <= proof.get('relative_l1_error_max', 0) < .005,
                    'independent numerical proof failed')
        for cid, component in row['components'].items():
            require(component['status'] == 'PASS', 'component measurement failed')
            require(cid == digest(component['identity']), 'component identity changed')
            require(component['record_sha256'] == digest({k:v for k,v in component.items()
                    if k != 'record_sha256'}), 'component receipt changed')
            stats = statistics_checked(component['samples_us'])
            require(all(component[k] == v for k,v in stats.items() if k in
                        ('median_us', 'round_medians_us', 'spread_pct')),
                    'component samples differ from statistics')
            require(component['first_use_excluded'] and component['host_launch_cost_added'] is False,
                    'warmup/launch timing contract differs')
            require(component['identity']['authority'] == row['authority'] and
                    row['authority']['round'] == index and row['authority']['point'] == point['id'],
                    'old round timing cannot be reused')
        for aid, arm in point['arms'].items():
            ids = [c for j in arm['jobs'] for c in row['jobs'][j]['components']]
            require(len(ids) == len(set(ids)), 'component double counted')
            values = [row['components'][c]['median_us'] for c in ids]
            require(values and all(math.isfinite(x) and x > 0 for x in values), 'invalid component cost')
            costs[aid].append(math.fsum(values))
            components.setdefault(aid, []).append([
                dict(kind=row['components'][c]['kind'], median_us=row['components'][c]['median_us'], id=c)
                for c in ids])
    stats = {a:dict(median_us=statistics.median(v), rounds_us=v,
                   spread_pct=100*(max(v)/min(v)-1)) for a,v in costs.items()}
    rules = [a for a,v in point['arms'].items() if 'rule' in v['roles']]
    controls = [a for a,v in point['arms'].items() if 'control' in v['roles'] or v['original']]
    require(rules and controls and point['original'] in controls, 'confirmation roles missing')
    best = min(rules, key=lambda a:(stats[a]['median_us'], a))
    control = min(controls, key=lambda a:(stats[a]['median_us'], a))
    ratios = {a:[x/y for x,y in zip(costs[best],costs[a])] for a in {control,point['original']}}
    noisy = any(stats[a]['spread_pct'] > 5 for a in {best,control,point['original']})
    no_regression = all(stats[best]['median_us'] <= stats[a]['median_us']*(1+1e-12) and
                        sum(r <= 1+1e-12 for r in v) >= 4 for a,v in ratios.items())
    status = 'UNSTABLE' if noisy else 'PASS' if no_regression else 'RULE_REGRESSION'
    return dict(point=point['id'], workload=point['workload'], status=status,
                winner=best if status == 'PASS' else None, best_rule=best, best_control=control,
                statistics=stats, components=components, ratios=ratios,
                recipes={a:v['recipe'] for a,v in point['arms'].items()},
                legacy_fallback=False, production_changed=False, performance_admitted=False,
                scope='ALTERNATING_COMPONENT_SUM_MODEL_AB_PENDING')
