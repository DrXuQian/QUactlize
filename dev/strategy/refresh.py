"""Bounded new-clock decode selection; old data proposes recipes, never times."""
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path

from dev.gemv_simt.general_plan import configs as common_configs
from dev.llama_ab.contract import inventory
from dev.strategy.prescreen import rejection, semantic_config
from dev.strategy.resources import module_spec
from dev.strategy.vector import spec as vector_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'quactlize.decode-clock-refresh.v1'
Q8_BODIES = (0, 1, 3, 4, 5)


def historical():
    path = ROOT/'docs/measurements/smallm_matched_20260915.json.gz'
    data = json.loads(gzip.decompress(path.read_bytes()))
    rows = []
    for old in data['points']:
        if not old['confirmed']:
            continue
        p = old['point_spec']
        try:
            w = Workload(p['q'], p['n'], p['k'], p['tokens'],
                         'indexed' if p['mode'] else 'dense', p['compute'],
                         experts=p['experts'], topk=p['topk'], channels=p['channels'],
                         router=p.get('router', 'real') if p['mode'] else '')
        except ValueError:
            continue
        choices = []
        for row in old['confirmed']:
            if row['spread_pct'] > 5 or not math.isfinite(row['median_us']) or row['median_us'] <= 0:
                continue
            c = data['candidates'][row['candidate']]
            r = (normalize(dict(c['parent'], kind='tc', split=c['split'],
                                grid_mode=c['grid_mode'], grid_b=c['grid_b']))
                 if c['kind'] == 'tc' else normalize(dict(c['recipe'], kind=c['kind'])))
            if not rejection(r, w):
                choices.append((row['median_us'], r))
        if choices:
            rows.append((w, choices))
    return rows


def family(w):
    return digest({k: asdict(w)[k] for k in
                   ('qtype', 'operation', 'compute', 'n', 'k', 'experts')})


def request(w):
    return [w.qtype, 2 if w.operation == 'indexed' else 0, w.n, w.k,
            w.experts, w.topk, w.channels, w.m, int(w.compute == 'bf16')]


def original_recipe(row):
    from dev.strategy.decode_plan import original_recipe as actual
    r = actual(row)
    if r.kind == 'tc':
        r = Recipe.make(r.kind, r.route, **(dict(r.parameters) |
            {'dequant_config': row.get('dequant_config', -1)} |
            ({'legacy_adapters':True} if row.get('native_f16') else {})))
    return r


def proposed(point, original, history, prior, *, allow_new_family=False):
    """No tile Cartesian product. Preserve exact old recipes plus route donors.

    Common SIMT has only the 3/4 geometries just admitted across S1/2/4/8.
    Q8 has two known geometries and the signed/window alternatives. TC uses
    the closest matched historical fixture's two fastest legal recipes per
    route, with one-axis S1/2/4/8 variation, not a tile/stage/grid search.
    """
    w = Workload(**point['workload'])
    result = {}
    excluded = []

    def add(r, reason, required=False):
        why = rejection(r, w)
        if why:
            if required:
                raise ValueError('required incumbent is ineligible: '+why)
            excluded.append(dict(recipe=asdict(r), reason=why, proposal=reason))
            return
        key = semantic_config(r)
        if key not in result:
            result[key] = dict(recipe=asdict(r), reasons=[], required=False)
        result[key]['required'] |= required
        if reason not in result[key]['reasons']:
            result[key]['reasons'].append(reason)

    add(original, 'CURRENT_CPP_SELECTED_RECIPE', True)
    old = prior.get(tuple(request(w)))
    if old is None and not allow_new_family:
        raise ValueError('frozen refresh lost its prior component control')
    if old is not None:
        add(normalize(old['original']), 'PREVIOUS_COMPONENT_INCUMBENT', True)
        add(normalize(old['config'], dequant=old['dequant_config']), 'CONFIRMED_COMPONENT_CHOICE', True)
    matches = [(hw, choices) for hw, choices in history
               if (hw.qtype, hw.operation, hw.compute) == (w.qtype, w.operation, w.compute)]
    def distance(item):
        hw, _ = item
        return (abs(math.log2(hw.n/w.n)) + abs(math.log2(hw.k/w.k)),
                abs(hw.m-w.m), hw.channels != w.channels, hw.experts != w.experts,
                hw.router != w.router, hw.key)
    # Select the fixture first; compare latencies only WITHIN that fixture.
    ordered = sorted(matches, key=distance)
    if ordered:
        hw, choices = ordered[0]
        best = min(us for us, r in choices)
        for us, r in choices:
            if hw.key == w.key and us <= 1.05*best:
                add(r, 'EXACT_OLD_WINNER_OR_WITHIN_FIVE_PERCENT')
    tc_donor = next(((hw, choices) for hw, choices in ordered
                     if any(r.kind == 'tc' and not rejection(r, w) for _, r in choices)), None)
    if tc_donor:
        hw, choices = tc_donor
        for route in ('fq', 'sf'):
            ranked = sorted(((us, r) for us, r in choices if r.kind == 'tc' and
                             r.route == route and not rejection(r, w)), key=lambda x: (x[0], x[1].key))
            for _, r in ranked[:2]:
                add(r, 'HISTORICAL_TC_ROUTE_REPRESENTATIVE_GEOMETRY_ONLY')
                for split in (1, 2, 4, 8):
                    add(Recipe.make(r.kind, r.route, **(dict(r.parameters) | {'split': split})),
                        'TC_SINGLE_AXIS_SPLIT_NEIGHBOR')
    for cfg in common_configs(w.qtype):
        cfg = {k: v for k, v in cfg.items() if k != 'hoist'}
        for split in (1, 2, 4, 8):
            add(Recipe.make('simt', 'fq', **(cfg | {'split': split})), 'ADMITTED_COMMON_SIMT_GEOMETRY')
        if w.qtype == 8 and w.operation == 'dense':
            for body in Q8_BODIES:
                add(Recipe.make('simt', 'fq', **(cfg | {'split': 1, 'q8_body': body})),
                    'Q8_SIGNED_ADDRESS_WINDOW_CANDIDATE_NOT_DEFAULT')
    original_key = semantic_config(original)
    arms = []
    for key, item in result.items():
        r = Recipe.from_dict(item['recipe'])
        spec = module_spec(w, r) if r.kind == 'tc' else vector_spec(w, r)
        arms.append(dict(id='original' if key == original_key else key,
                         roles=['original'] if key == original_key else ['rule'], aliases=[],
                         **item, spec=spec, module=digest(spec)))
    arms.sort(key=lambda a: (a['id'] != 'original', a['id']))
    if not any(a['recipe']['kind'] == 'tc' for a in arms):
        raise ValueError('no legal TC comparator: '+point['id'])
    if len(arms) > 48:
        raise ValueError('bounded refresh exceeds 48 arms; prune explicitly before running: '+point['id'])
    return dict(point, original_recipe=asdict(original), arms=arms, excluded=excluded,
                sf_configs=sorted({max(0, dict(Recipe.from_dict(a['recipe']).parameters).get('dequant_config', -1))
                                   for a in arms if a['recipe']['route'] == 'sf' and w.qtype != 8}))


def make_plan(originals, phase='all'):
    points = [p for p in inventory() if p['cohort'] == 'decode']
    if [r['request'] for r in originals] != [request(Workload(**p['workload'])) for p in points]:
        raise ValueError('C++ incumbent queries differ from the frozen 1114 points')
    prior = json.loads((ROOT/'policies/kpack_component_decode_v1.json').read_text())
    prior = {tuple(r['key']): r for r in prior['rows']}
    history = historical()
    all_points = [proposed(p, original_recipe(o), history, prior) for p, o in zip(points, originals)]
    if phase == 'pilot':
        # Broad mechanism smoke before a larger run, not a selection admission.
        selected = {}
        for p in sorted(all_points, key=lambda p: (p['workload']['n']*p['workload']['k'], p['id'])):
            w = p['workload']
            selected.setdefault((w['qtype'], w['operation'], w['m'] in (1, 8)), p)
        all_points = list(selected.values())
    elif phase != 'all':
        raise ValueError('unknown refresh phase')
    modules, vectors = {}, {}
    for p in all_points:
        for a in p['arms']:
            (modules if a['recipe']['kind'] == 'tc' else vectors)[a['module']] = a['spec']
    counts = Counter(p['workload']['operation'] for p in all_points)
    result = dict(schema=SCHEMA, phase=phase, points=all_points, tc=modules, vectors=vectors,
                  counts=dict(points=len(all_points), full_points=len(points),
                              families=len({family(Workload(**p['workload'])) for p in all_points}),
                              arms=sum(len(p['arms']) for p in all_points),
                              max_arms=max(len(p['arms']) for p in all_points),
                              tc_modules=len(modules), vector_modules=len(vectors), operations=dict(counts)),
                  historical_times_reused=False, production_changed=False,
                  scope='SMALL_M_ISOLATED_COMPONENT_SELECTION_NOT_PREFILL_OR_MODEL_TPOT',
                  comparison='EXPLICIT_CANDIDATES_INCLUDING_HISTORY_NOT_BLIND_RULE_RECALL',
                  q8_bodies=list(Q8_BODIES), q8_new_bodies_scope='DENSE_S1_ONLY',
                  protocol=dict(rounds=6, samples=5, order='FORWARD_REVERSE_ALTERNATING',
                                producer_ring='AT_LEAST_2P25_L2', helper_timing='ONE_KERNEL_AT_A_TIME'),
                  omitted=['LARGE_M', 'MODEL_AB', 'NEW_Q8_MOE_BODY', 'NEW_Q8_SPLIT_BODY'],
                  sources={name: sha(ROOT/name) for name in
                           ('policies/kpack_component_decode_v1.json',
                            'policies/kpack_decode_winners_v1.json',
                            'docs/measurements/smallm_matched_20260915.json.gz')})
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))


def assign(plan, workers):
    groups = defaultdict(list)
    for p in plan['points']:
        groups[family(Workload(**p['workload']))].append(p)
    if not 1 <= workers <= len(groups):
        raise ValueError('worker count exceeds available weight families')
    def work(points):
        w = points[0]['workload']
        return w['n']*w['k']*(min(17, w['experts']) +
               sum(len(p['arms'])*(1+p['workload']['m']*p['workload']['topk']/8) for p in points))
    shards = [[] for _ in range(workers)]
    loads = [0]*workers
    for key, points in sorted(groups.items(), key=lambda item: (-work(item[1]), item[0])):
        i = min(range(workers), key=lambda j: (loads[j], j))
        shards[i].append(key)
        loads[i] += work(points)
    return shards
