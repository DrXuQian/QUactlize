"""Independent retiming of a bounded, historically screened prefill shortlist."""
from dataclasses import asdict
import json
import math
from pathlib import Path

from dev.strategy.prefill_closure import require
from dev.strategy.prefill_confirmation import bind_point, verdict, ROUNDS
from dev.strategy.prefill_plan import family
from dev.strategy.refresh_clock import ADVISORY, parse
from dev.strategy.resources import module_spec, probe_spec
from dev.strategy.space import resolved_grid
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT/'dev/strategy/prefill_refresh_seed.json'


def load_seed():
    value = json.loads(SEED.read_text())
    require(value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'prefill shortlist seal differs')
    require(value['policy_sha256'] == sha(ROOT/'policies/kpack_prefill_confirmation_v1.json'),
            'production prefill policy changed; review the shortlist again')
    return value


def request(w):
    return (w.qtype, 0 if w.operation == 'dense' else 2, w.n, w.k, w.experts,
            w.topk, w.channels, w.m, int(w.compute == 'bf16'))


def recipe_of(row):
    require(row['status'] == 0 and row['policy'] == 19, 'current C++ prefill selector missed')
    if row['kind'] == 3 and row['route'] == 2:
        return Recipe.make('full', 'full', dequant_config=row['dequant_config'], provider='bf16-default')
    require(row['kind'] == 0 and row['route'] in (0, 1), 'unexpected current prefill family')
    return normalize(dict(row['parent'], kind='tc', split=row['split'],
        grid_mode=row['grid_mode'], grid_b=row['grid_b']), row['route'], row['dequant_config'])


def same_recipe(a, b):
    # Symbols are aliases; all arithmetic/geometry/helper/scheduler fields stay.
    return (a.kind == b.kind and a.route == b.route and
            {k: v for k, v in a.parameters if k != 'symbol'} ==
            {k: v for k, v in b.parameters if k != 'symbol'})


def make_plan(selector, cards, seed=None):
    seed = load_seed() if seed is None else seed
    require(cards and len(cards) == len(set(cards)) and all(c.isdigit() for c in cards),
            'unique physical card selectors required')
    rows = {tuple(r['request']): r for r in selector['requests']}
    require(len(rows) == len(selector['requests']) == len(seed['points']) == 430,
            'C++ selector/shortlist denominator differs')
    points, modules, queries = [], {}, {}
    for item in seed['points']:
        w = Workload(**item['workload'])
        require(w.m in (128, 512, 1024, 2048, 4096) and w.qtype in range(10, 15) and
                w.storage == 'f32' and w.operation in ('dense', 'indexed') and
                w.compute == ('f16' if w.operation == 'dense' else 'bf16'), 'unexpected prefill semantics')
        row = rows.pop(request(w))
        incumbent = recipe_of(row)
        arms = {}
        for aid, entry in item['arms'].items():
            r = Recipe.from_dict(seed['recipes'][entry['recipe']])
            require(r.key == entry['recipe'] and r.kind in ('tc', 'full'), 'recipe seal/kind differs')
            original = aid == item['incumbent']
            roles = ['rule'] if entry['qualified'] else []
            if original or aid == item['historical_original'] or 'control' in entry['historical_roles']:
                roles += ['control']
            if original:
                require(same_recipe(incumbent, r), 'shortlist incumbent differs from current C++ selection')
                require(entry['qualified'], 'current incumbent lacks reviewed qualification')
                roles += ['original']
            arms[aid] = dict(recipe=asdict(r), source_arm=aid, roles=roles, original=original,
                historical_roles=entry['historical_roles'], qualified=entry['qualified'])
            if r.kind == 'tc':
                spec = module_spec(w, r)
                query = probe_spec(w, r, int(dict(r.parameters)['grid_mode'] >= 2))
                modules[digest(spec)] = spec
                queries[digest(query)] = query
        require(item['incumbent'] in arms and 2 <= len(arms) <= 4, 'missing incumbent/bounded finalists')
        points.append(dict(id=item['id'], workload=item['workload'], arms=arms,
            original=item['incumbent'], historical_original=item['historical_original'],
            current_selector=row))
    require(not rows and len({p['id'] for p in points}) == 430, 'point coverage changed')
    weights = seed['scheduling_seconds']
    require(set(weights) == {family(p) for p in points} and
            all(math.isfinite(v) and v > 0 for v in weights.values()), 'invalid family workload hints')
    require(len(cards) <= len(weights), 'too many shards')
    loads, owners, order = [0.]*len(cards), {}, []
    for key in sorted(weights, key=lambda k: (-weights[k], k)):
        owner = min(range(len(cards)), key=lambda i: (loads[i], i))
        owners[key] = owner
        loads[owner] += weights[key]
        order.append(key)
    for p in points:
        p['shard'] = owners[family(p)]
    value = dict(schema='quactlize.prefill-refresh.v1', points=points, modules=modules,
        queries=queries, cards=cards, family_order=order, family_owners=owners,
        scheduling_seconds=weights, historical_shard_seconds=loads,
        scheduling_scope=seed['scheduling_scope'], seed_sha256=seed['sha256'],
        counts=dict(points=430, arms=sum(len(p['arms']) for p in points), families=len(weights),
                    modules=len(modules), queries=len(queries), rounds=ROUNDS),
        scope='HISTORICAL_FINALIST_RETIMING_CURRENT_CPP_CONTROL_NOT_NEW_GLOBAL_SEARCH',
        protocol='SEPARATE_COMPONENTS_CUBLAS_ORDINARY_EVENTS_NO_GRAPH_OR_ASYS',
        excluded=['UNSEEN_SHAPES', 'Q8_LARGE_M', 'NEW_CONFIG_SEARCH', 'MODEL_TPOT', 'LLAMA_NATIVE_AB'],
        previous_timings_reused=False, production_changed=False, performance_admitted=False)
    require(value['counts']['arms'] == 1138 and len(modules) == 115, 'compile/timing budget changed')
    value['plan_sha256'] = digest(value)
    return json.loads(json.dumps(value))


def bind(point, queries):
    arms = []
    w = Workload(**point['workload'])
    for aid, item in point['arms'].items():
        r = Recipe.from_dict(item['recipe'])
        arm = dict(id=aid, recipe=item['recipe'])
        if r.kind == 'tc':
            spec = module_spec(w, r)
            expected = probe_spec(w, r, int(dict(r.parameters)['grid_mode'] >= 2))
            qid = digest(expected)
            q = queries[qid]
            require(q['request'] == expected and q['status'] == 'READY' and q['rc'] == 0 and
                    q['occupancy'] > 0, 'required finalist resource unavailable; cannot drop an arm')
            arm.update(module=digest(spec), spec=spec, binding=dict(query=qid,
                launch=resolved_grid(w, r, q['device']['compute_units'], q['occupancy'])))
        arms.append(arm)
    return bind_point(point, dict(id=point['id'], workload=point['workload'], arms=arms))


def checked_round(row, point, index, authority):
    require(row.get('record_sha256') == digest({k: v for k, v in row.items() if k != 'record_sha256'}),
            'round receipt changed')
    require(row['authority'] == dict(sha256=digest(authority), point=point['id'], round=index) and
            row['point'] == point['id'] and row['round'] == index and row['status'] == 'PASS',
            'round belongs to another run/point')
    clock = authority['clock']
    require(clock['policy'] == ADVISORY, 'unexpected clock policy')
    for side in ('before', 'after'):
        observation = row['clock_observations'][side]
        parsed = parse(observation['stdout'], authority['device']['pci'], clock['requested_mhz'], advisory=True)
        require(all(observation.get(k) == v for k, v in parsed.items()), 'clock receipt interpretation changed')
    # Reuse the full numerical/component validator for partial-round resumes.
    # It requires six indices; validation of the single round is factored out.
    from dev.strategy.prefill_confirmation import validate_round
    validate_round(point, row, index)
    return row


def collect(point, directory, authority):
    rounds = []
    for index in range(ROUNDS):
        path = directory/'points'/point['id']/f'round-{index}.json'
        if path.exists():
            rounds.append(checked_round(json.loads(path.read_text()), point, index, authority))
    if len(rounds) < ROUNDS:
        return dict(point=point['id'], status='INCOMPLETE', completed_rounds=len(rounds))
    return verdict(point, rounds)
