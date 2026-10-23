"""Bounded TP-local dense proposals from the current whole-call evidence.

Donor timing only prunes candidates. Every target, including table hits, gets
new native and candidate measurements on the same card. No model/TPOT claim.
"""
from dataclasses import asdict
import copy
import json
import math

from dev.prefill_asys.cohort import validate as validate_cohort
from dev.prefill_asys.finalists import case_identity, counts, full_case
from dev.strategy.prescreen import rejection
from dev.strategy.space import rebind_template
from quactlize.dispatch.dense_prefill import bucket_key, validate as validate_policy
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.dense-tp-supplement.v1'
MS = (128, 256, 512, 1024, 2048)


def execution_case(value):
    if value['route'] == 2:
        return full_case(value['dequant'], value['io'])
    recipe = Recipe.make('tc', 'sf' if value['route'] else 'fq',
                         **dict(value['tc'], dequant_config=value['dequant']))
    return dict(arm=recipe.route, helper_mode='cohort-current',
                helper_options=dict(vector_io=value['io']),
                recipe=json.loads(json.dumps(asdict(recipe))))


def legal(value, w):
    if value['route'] == 2:
        return True
    return rejection(Recipe.from_dict(execution_case(value)['recipe']), w) is None


def policy_lookup(policy, w):
    shape = [w.qtype, w.m, w.n, w.k]
    for point in policy['points']:
        if point['shape'] == shape:
            return 'EXACT', point['winner']
    for bucket in policy['buckets']:
        if (list(bucket_key(shape)) == bucket['key'] and
                bucket['bounds'][0][0] <= w.n <= bucket['bounds'][0][1] and
                bucket['bounds'][1][0] <= w.k <= bucket['bounds'][1][1] and
                legal(policy['recipes'][bucket['recipe']], w)):
            return 'BUCKET_PREDICTION', bucket['recipe']
    return 'OUTSIDE_NEW_TABLE', None


def make_plan(cohort, policy, helper_sha, *, max_cases=32):
    validate_cohort(cohort)
    validate_policy(policy)
    if not 8 <= max_cases <= 48:
        raise ValueError('supplement budget must be 8..48; never truncates required controls')
    points = []
    for point in cohort['points']:
        w = Workload(**point['workload'])
        if w.m not in MS:
            raise ValueError('supplement covers M128/256/512/1024/2048')
        pool = {}
        def add(case, role):
            r = Recipe.from_dict(case['recipe'])
            if r.kind == 'tc' and rejection(r, w):
                return
            key = digest(case)
            row = pool.setdefault(key, dict(copy.deepcopy(case), id=key, roles=[]))
            if role not in row['roles']:
                row['roles'].append(role)

        coverage, incumbent = policy_lookup(policy, w)
        if incumbent:
            add(execution_case(policy['recipes'][incumbent]), 'model-config')

        donors = [p for p in policy['points'] if p['shape'][0] == w.qtype]
        # Q2 has no measurements in the current cohort. Generic K-quant TC
        # geometry can be proposed, but its symbol/format is re-instantiated.
        transfer = not donors
        if transfer:
            donors = [p for p in policy['points'] if p['shape'][0] == 12]
        def distance(p):
            _, m, n, k = p['shape']
            return (abs(math.log2(m/w.m)), abs(math.log2(n/w.n))+abs(math.log2(k/w.k)), p['key'])
        nearest = sorted(donors, key=distance)[:4]
        selected = {}
        # Preserve every legal shipping TC winner for this format, not only a
        # nearest-shape tile. Keep helper variants explicit too.
        for p in donors:
            selected.setdefault(p['winner'], []).append('shipping-format-winner-proposal')
        # Two leaders per route at each of four nearby same-precision points.
        # No absolute microseconds are compared across different shapes.
        for p in nearest:
            for route in (0, 1, 2):
                costs = [(r, us) for r, us in p['costs'] if policy['recipes'][r]['route'] == route]
                for rid, _ in sorted(costs, key=lambda v:(v[1],v[0]))[:2]:
                    selected.setdefault(rid, []).append('nearby-per-route-proposal')
        for rid, roles in sorted(selected.items()):
            value = policy['recipes'][rid]
            if transfer:
                if value['tc'] is None:
                    continue
                c = execution_case(value)
                r = rebind_template(Recipe.from_dict(c['recipe']), w.qtype)
                parameters = dict(r.parameters)
                parameters['dequant_config'] = 0
                c['recipe'] = json.loads(json.dumps(asdict(Recipe.make('tc', r.route, **parameters))))
                roles = ['CROSS_FORMAT_GEOMETRY_PROPOSAL_NOT_TIMING']
            else:
                c = execution_case(value)
            for role in sorted(set(roles)):
                add(c, role)
        # Explicit Full-F16 controls use the optimized helpers, not BF16 costs
        # or the obsolete 16-way helper Cartesian product.
        variants = (38, 41) if w.qtype == 8 else (22, 23) if w.qtype == 11 else (20, 21) if w.qtype >= 12 else (0,)
        for dq in variants:
            for io in (2, 3):
                add(full_case(dq, io), 'current-full-helper-proposal')
        add(full_case(0, 0), 'typed-direct-control')
        if len(pool) > max_cases:
            raise ValueError(f'{w}: {len(pool)} candidates exceed {max_cases}; none silently dropped')
        if incumbent and not any('model-config' in c['roles'] for c in pool.values()):
            raise ValueError('current policy control was dropped')
        cases = sorted(pool.values(), key=lambda c:digest([w.key,c['id']]))
        routes = {c['arm'] for c in cases}
        required = {'sf', 'full-fp16'} | ({'fq'} if w.qtype != 8 else set())
        if not required <= routes:
            raise ValueError(f'{w}: missing route candidates {required-routes}; do not start a partial scan')
        points.append(dict(point, cases=cases, candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],
            champion_controls=[], confirmation_captures=2, fixed_schedules=True,
            table_coverage=coverage, incumbent_recipe=incumbent,
            donor_points=[p['key'] for p in nearest]))
    result = dict(schema=SCHEMA, points=points, reuse=[], confirmation_captures=2,
        counts=counts(points), max_cases=max_cases,
        inputs=dict(cohort_sha256=cohort['sha256'], policy_sha256=policy['sha256'],
                    helper_policy_sha256=helper_sha),
        exclusions=cohort['excluded'], models=cohort['models'],
        scope='TP_LOCAL_DENSE_PREFILL_ONLY_NOT_MOE_OR_TP_COLLECTIVES_OR_MODEL_TPOT',
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        historical_timing_used='CANDIDATE_PRUNING_ONLY; ALL_TARGET_ARMS_REMEASURED',
        search='BOUNDED_MEASURED_RECIPE_TRANSFER_NOT_GLOBAL_OPTIMUM',
        production_changed=False, performance_admitted=False)
    result['sha256'] = digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema') != SCHEMA or plan.get('sha256') != digest({k:v for k,v in plan.items() if k!='sha256'})
            or plan.get('confirmation_captures') != 2 or plan.get('production_changed') is not False
            or plan.get('performance_admitted') is not False or not 8 <= plan.get('max_cases',0) <= 48):
        raise ValueError('TP supplement plan contract/digest differs')
    seen = set()
    for point in plan['points']:
        w = Workload(**point['workload'])
        if (w.key != point['id'] or w.key in seen or w.operation != 'dense' or w.compute != 'f16'
                or w.storage != 'f32' or w.m not in MS or not point['fixed_schedules']
                or point['confirmation_captures'] != 2 or not 1 <= len(point['cases']) <= plan['max_cases']):
            raise ValueError('invalid supplement workload/budget')
        seen.add(w.key)
        ids = set()
        for c in point['cases']:
            r = Recipe.from_dict(c['recipe'])
            if c['id'] != digest(case_identity(c)) or c['id'] in ids or not c['roles']:
                raise ValueError('duplicate/changed supplement candidate')
            ids.add(c['id'])
            if r.kind == 'tc':
                if (c['arm'] != r.route or c['helper_mode'] != 'cohort-current' or rejection(r,w)
                        or c['helper_options'] != dict(vector_io=c['helper_options']['vector_io'])
                        or c['helper_options']['vector_io'] not in (-1,1,2,3)):
                    raise ValueError('invalid TC helper/config contract')
            elif r.kind == 'full':
                opts = c['helper_options']
                if (case_identity(c) != full_case(opts['dequant_variant'],opts['vector_io'])
                        or opts['vector_io'] not in range(4)
                        or opts['dequant_variant'] not in ((0,22,23,36,37,38,41) if w.qtype==8 else (0,8,20,21,22,23))):
                    raise ValueError('invalid Full-F16 supplement')
            else:
                raise ValueError('unexpected decode/MoE recipe in prefill supplement')
        if point['incumbent_recipe'] and not any('model-config' in c['roles'] for c in point['cases']):
            raise ValueError('missing current policy control')
        required = {'sf','full-fp16'} | ({'fq'} if w.qtype != 8 else set())
        if not required <= {c['arm'] for c in point['cases']}:
            raise ValueError('missing route comparison')
        if not any('typed-direct-control' in c['roles'] for c in point['cases']):
            raise ValueError('missing Full-F16 direct control')
        if point['candidates'] != [c['recipe'] for c in point['cases'] if c['recipe']['kind']=='tc'] or point['champion_controls']:
            raise ValueError('fixed candidate/resource inventory differs')
    if not seen or plan['counts'] != counts(plan['points']):
        raise ValueError('supplement counts differ')
    return plan
