"""Measured exact choices and cross-validated bucket proposals, from one table."""
from collections import defaultdict
from dataclasses import asdict
import math
import re
import statistics

from dev.model_tuning.inventory import sealed
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.model-tuning-policy.v1'


def host_key(point):
    # Routing profiles are evidence, never a host-readable runtime selector.
    return digest(dict(workload=point['workload'] | {'router':''}, contract=point['contract']))


def samples(values):
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError('confirmed timing needs repeated samples')
    flat = [v for row in values for v in row] if isinstance(values[0], list) else values
    if len(flat) < 3 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in flat):
        raise ValueError('invalid timing samples')
    return flat


def hard_key(point):
    w, c = point['workload'], point['contract']
    return dict(qtype=w['qtype'], operation=w['operation'], compute=w['compute'], storage=w['storage'],
        experts=w['experts'], topk=w['topk'], channels=w['channels'],
        layout=c['layout'], epilogue=c['epilogue'], form=c['form'],
        output_type=c['output_type'], input_type=c['input_type'], accumulator=c['accumulator'],
        round_projection=c['round_projection'], row_order=c['row_order'], scope=c['scope'],
        input_contiguous=w['input_stride']==w['k'], output_contiguous=w['output_stride']==w['n'])


def bucket_key(point):
    w = point['workload']
    m, n, k = (w[x] for x in ('m', 'n', 'k'))
    return dict(hard_key(point), m_band=['decode', m] if m <= 8 else ['prefill', m.bit_length()-1],
        n_band=n.bit_length()-1, k_band=k.bit_length()-1,
        aspect='wide-n' if n >= 2*k else 'wide-k' if k >= 2*n else 'balanced')


def legal(binding, point):
    """Cheap exclusions only; the real backend still queries compiled resources."""
    w = Workload(**point['workload'])
    recipe = Recipe.from_dict(binding['recipe'])
    p = dict(recipe.parameters)
    return (recipe.eligible(w) and
        p.get('fixed_n', 0) in (0, point['contract']['weight_n']) and
        p.get('fixed_k', 0) in (0, w.k) and
        not (w.operation == 'dense' and w.m >= 128 and recipe.route == 'fq'))


def record(point, configs, baseline_us, provenance, *, proof):
    if (provenance.get('timing_method') != 'ASYS_GPU_ACTIVITY' or
            not re.fullmatch('[0-9a-f]{40}', provenance.get('quactlize_commit', '')) or
            not provenance.get('source_identity') or not provenance.get('target_identity') or not proof):
        raise ValueError('missing Asys/commit/implementation/device/evidence identity')
    if not math.isfinite(baseline_us) or baseline_us <= 0:
        raise ValueError('missing finite incumbent cost')
    costs, definitions = {}, {}
    for value in configs:
        if value.get('correctness') != 'PASS':
            continue
        values = samples(value['samples_us'])
        cid = digest(value['binding'])
        if not legal(value['binding'], point):
            continue
        if cid in definitions:
            # A control and rule may resolve to the same executable binding.
            # Keep both raw captures; do not turn aliases into two choices.
            values = samples(definitions[cid]['samples_us']) + values
            value = dict(value, samples_us=values,
                         alias_captures=definitions[cid].get('alias_captures',1)+1)
        definitions[cid] = dict(value, mean_us=statistics.fmean(values))
        costs[cid] = definitions[cid]['mean_us']
    if not costs:
        raise ValueError('no valid confirmed configuration')
    winner = min(costs, key=lambda key: (costs[key], key))
    return dict(point=point['id'], workload=point['workload'], contract=point['contract'],
        configurations=definitions, winner=winner, mean_us=costs[winner], baseline_us=baseline_us,
        incumbent_no_regression=costs[winner] <= baseline_us*(1+1e-9),
        native_no_regression_admitted=False, provenance=provenance, proof=proof)


def _choose(rows):
    common = set.intersection(*(set(r['configurations']) for r in rows))
    if not common:
        return None
    return min(common, key=lambda key: (
        max(r['configurations'][key]['mean_us']/r['mean_us'] for r in rows),
        statistics.fmean(r['configurations'][key]['mean_us']/r['mean_us'] for r in rows), key))


def fit(rows, *, max_regret_pct=1.0):
    """Leave one N/K family out; one-family buckets can validate only M transfer."""
    descriptor = bucket_key(rows[0])
    if any(bucket_key(r) != descriptor for r in rows):
        raise ValueError('bucket mixes arithmetic, layout or geometry strata')
    value = dict(bucket=descriptor, points=[r['point'] for r in rows],
                 status='INSUFFICIENT_HOLDOUT_COVERAGE', holdout_validated=False)
    families = defaultdict(list)
    for row in rows:
        w = row['workload']
        families[(w['n'], w['k'])].append(row)
    mode = 'NK_TRANSFER'
    if len(families) == 1:
        mode = 'M_INTERPOLATION_ONLY'
        families = defaultdict(list)
        for row in rows:
            families[row['workload']['m']].append(row)
    value['transfer_scope'] = mode
    if len(families) < 3:
        return value
    folds = []
    for held_out, test in sorted(families.items()):
        training = [r for key, group in families.items() if key != held_out for r in group]
        selected = _choose(training)
        if selected is None or any(selected not in r['configurations'] for r in test):
            value.update(status='MISSING_COMMON_CONFIG_MEASUREMENTS', held_out=held_out)
            return value
        regrets = [100*(r['configurations'][selected]['mean_us']/r['mean_us']-1) for r in test]
        no_regression = all(r['configurations'][selected]['mean_us'] <= r['baseline_us']*(1+1e-9) for r in test)
        folds.append(dict(held_out=held_out, candidate=selected, max_regret_pct=max(regrets),
                          incumbent_no_regression=no_regression))
    chosen = _choose(rows)
    if chosen is None:
        return dict(value, status='NO_COMMON_CONFIG', folds=folds)
    regrets = [100*(r['configurations'][chosen]['mean_us']/r['mean_us']-1) for r in rows]
    admissible = (all(f['max_regret_pct'] <= max_regret_pct and f['incumbent_no_regression'] for f in folds)
                  and max(regrets) <= max_regret_pct and all(
                      r['configurations'][chosen]['mean_us'] <= r['baseline_us']*(1+1e-9) for r in rows))
    value.update(status='VALIDATED_PROPOSAL' if admissible else 'EXACT_ONLY_REGRET_TOO_LARGE',
        holdout_validated=admissible, folds=folds, candidate=chosen, max_regret_pct=max(regrets),
        binding=rows[0]['configurations'][chosen]['binding'])
    if mode == 'M_INTERPOLATION_ONLY':
        value['nk'] = [rows[0]['workload'][k] for k in ('n','k')]
    return value


def export(records, *, expected=(), pending=(), max_regret_pct=1.0):
    if not math.isfinite(max_regret_pct) or not 0 <= max_regret_pct <= 10:
        raise ValueError('bucket regret must be between 0 and 10 percent')
    observations, groups, definitions = {}, defaultdict(list), {}
    host_groups = defaultdict(list)
    for row in records:
        key = row['point']
        if key in observations:
            raise ValueError('duplicate point; replace a complete point explicitly')
        observations[key] = row
        host_groups[host_key(row)].append(row)
        for cid, value in row['configurations'].items():
            definitions[cid] = value['binding']
        # Device/SDK and implementation identity are never pooled implicitly.
        domain = {k:row['provenance'][k] for k in ('target_identity','source_identity')}
        groups[digest([bucket_key(row), domain])].append(row)
    exact, routing_gaps = {}, []
    for key, profiles in host_groups.items():
        domains={digest({k:r['provenance'][k] for k in ('target_identity','source_identity')}) for r in profiles}
        if len(domains)!=1:
            raise ValueError('one exact key has incompatible source/target measurements')
        common=set.intersection(*(set(r['configurations']) for r in profiles))
        common={cid for cid in common if all(r['configurations'][cid]['mean_us']<=r['baseline_us']*(1+1e-9)
                                            for r in profiles)}
        if not common:
            routing_gaps.append(dict(key=key,reason='NO_COMMON_NONREGRESSING_ROUTER_CONFIG',
                                     points=[r['point'] for r in profiles]))
            continue
        chosen=min(common,key=lambda cid:(max(r['configurations'][cid]['mean_us']/r['mean_us'] for r in profiles),
                    statistics.fmean(r['configurations'][cid]['mean_us']/r['mean_us'] for r in profiles),cid))
        exact[key]=dict(workload=profiles[0]['workload'] | {'router':''}, contract=profiles[0]['contract'],
            winner=chosen,binding=definitions[chosen],provenance=profiles[0]['provenance'],
            profiles=[dict(point=r['point'],router=r['workload']['router'],
                mean_us=r['configurations'][chosen]['mean_us'],baseline_us=r['baseline_us'],proof=r['proof'])
                for r in profiles],selection='ONE_STATIC_CONFIG_ACROSS_ROUTING_PROFILES',
            incumbent_no_regression=True)
    buckets = {}
    for key, rows in groups.items():
        value = fit(rows, max_regret_pct=max_regret_pct)
        value['domain'] = {k:rows[0]['provenance'][k] for k in ('target_identity','source_identity')}
        buckets[key] = value
    missing = sorted(set(expected)-set(observations))
    regressions = sorted(key for key, r in observations.items() if not r['incumbent_no_regression'])
    return sealed(dict(schema=SCHEMA, exact=exact, buckets=buckets, recipes=definitions,
        observations=observations,missing=missing, pending=[*pending,*routing_gaps], incumbent_regressions=regressions,
        status='READY_FOR_INTEGRATION' if exact and not missing and not pending and not routing_gaps and not regressions else 'INCOMPLETE',
        bucket_max_regret_pct=max_regret_pct, statistic='ARITHMETIC_MEAN_COMPONENT_SUM',
        production_changed=False, native_no_regression_admitted=False, model_tpot_admitted=False,
        scope='OFFLINE_POLICY_ARTIFACT_NOT_AUTOMATIC_RUNTIME_INSTALL'))


def lookup(policy, point, *, target_identity, source_identity):
    """Host replay of precedence. No compilation, timing, mutation or fake costs."""
    if policy.get('schema') != SCHEMA or policy.get('sha256') != digest(
            {k:v for k,v in policy.items() if k != 'sha256'}):
        raise ValueError('policy digest/schema differs')
    row = policy['exact'].get(host_key(point))
    domain = dict(target_identity=target_identity, source_identity=source_identity)
    if row and all(row['provenance'][k] == v for k,v in domain.items()) and row['incumbent_no_regression']:
        binding = row['binding']
        if legal(binding, point):
            profile=next((p for p in row['profiles'] if p['router']==point['workload']['router']),None)
            return dict(kind='EXACT', binding=binding, measured_us=profile['mean_us'] if profile else None)
    descriptor = bucket_key(point)
    for bucket in policy['buckets'].values():
        if (bucket['domain'] == domain and bucket['bucket'] == descriptor and bucket['holdout_validated'] and
                ('nk' not in bucket or bucket['nk'] == [point['workload'][k] for k in ('n','k')]) and
                legal(bucket['binding'], point)):
            return dict(kind='BUCKET_PROPOSAL_REQUIRES_RESOURCE_QUERY', binding=bucket['binding'], measured_us=None)
    return dict(kind='MISS_REQUIRES_OFFLINE_SCAN_OR_VALIDATED_GENERAL_PATH', binding=None, measured_us=None)
