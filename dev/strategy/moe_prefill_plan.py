"""Actual-model large-M audit before the bounded Asys execution pilot.

This is NOT a timing runner. Shipping Full capability is queried from C++;
an offline proposal is never sufficient evidence that a route can execute.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
import subprocess

from dev.strategy import moe_model_plan as models
from dev.strategy.moe_plan import MS as ALL_MS, RULES, load_catalog, moe_candidates, source_stamp
from dev.strategy.prefill_plan import bind_original, full_configs
from dev.strategy.prescreen import Prescreen, rejection
from dev.strategy.resources import inventory as resource_inventory, module_spec
from dev.strategy.space import factored_recall
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = models.ROOT
MS = tuple(m for m in ALL_MS if m > 8)
COHORT = 'moe-prefill-models'


def full_capabilities(points, output):
    """Exact public layout constraints, not a guessed M range or GPU proof."""
    requests = sorted({(w.qtype, w.n, w.k, w.m, w.experts, w.topk, w.channels, c)
                       for point in points for w in [Workload(**point['workload'])]
                       for c in full_configs(w)})
    exe = Path(output)/'prefill-capability'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT), '-I'+str(ROOT/'quactlize/include'),
                    str(ROOT/'dev/strategy/prefill_capability.cpp'), '-o', str(exe)], check=True)
    text = subprocess.check_output([str(exe)], text=True,
        input=''.join(' '.join(map(str, r))+'\n' for r in requests))
    rows = [json.loads(line) for line in text.splitlines()]
    if [r['request'] for r in rows] != list(map(list, requests)):
        raise ValueError('Full layout validator reordered/omitted requests')
    return dict(rows=rows, executable_sha256=sha(exe), source_sha256=sha(ROOT/'quactlize/prefill/layout.hpp'),
                scope='SHIPPING_HOST_LAYOUT_ONLY_PROVIDER_COMPILE_AND_GPU_EXECUTION_PENDING')


def key(w):
    return replace(w, router='').key


def pilot_plan(audit):
    """Mechanism coverage, not a speed shortlist or the full search pool."""
    if audit['sha256'] != digest({k:v for k,v in audit.items() if k!='sha256'}):
        raise ValueError('prefill audit digest differs')
    families = sorted({(p['workload']['qtype'],p['workload']['n'],p['workload']['k'],p['workload']['channels'])
                       for p in audit['points']})
    points, tc = [], {}
    for point in audit['points']:
        w = Workload(**point['workload'])
        f = (w.qtype,w.n,w.k,w.channels)
        # Full's legal lower bound across every family. Also test the two
        # extreme M boundaries for one family of each caller channel layout.
        first = next(family for family in families if family[-1]==w.channels)
        wanted = w.m==128 or f==first and w.m in (16,5120)
        router = 'cluster' if w.channels>1 and w.m==128 else 'spread'
        if not wanted or w.router!=router:
            continue
        arms = []
        for route in ('fq','sf','full'):
            pool = [Recipe.from_dict(r) for r in point['candidates'] if r['route']==route]
            if route!='full':
                pool = [r for r in pool if dict(r.parameters)['split']==1 and
                        dict(r.parameters).get('grid_mode',0)==0]
            if not pool:
                if route!='full' or any(r['rc']==0 for r in point['full_capability']):
                    raise ValueError('execution pilot missing legal route '+route)
                continue
            def order(r):
                p = dict(r.parameters)
                return (p.get('tm',0),p.get('tn',0),p.get('stages',0),r.key)
            recipe = min(pool,key=order)
            arm = dict(id=recipe.key,recipe=asdict(recipe),roles=['pilot'],required=True)
            if recipe.kind=='tc':
                spec = module_spec(w,recipe)
                arm.update(spec=spec,module=digest(spec));tc[arm['module']] = spec
            arms.append(arm)
        points.append(dict(id=point['id'],workload=point['workload'],consumers=point['consumers'],arms=arms,
            sf_configs=sorted({max(0,dict(a['recipe']['parameters']).get('dequant_config',-1))
                               for a in arms if a['recipe']['route']=='sf'})))
    if not points or sum(len(p['arms']) for p in points)>32:
        raise ValueError('empty/oversized large-M execution pilot; no implicit full sweep')
    result = dict(schema='quactlize.moe-prefill-pilot.v1',audit=audit['sha256'],points=points,
        tc=tc,vectors={},counts=dict(points=len(points),arms=sum(len(p['arms']) for p in points),modules=len(tc)),
        scope='GPU_ROUTING_AND_ISOLATED_ASYS_EXECUTION_NOT_WINNER_SELECTION',
        historical_times_reused=False,production_changed=False,performance_admitted=False,
        pending=['FULL_CONFIG_SCREEN_CONFIRM','NATIVE_LARGE_M_AB','PER_CALL_ERROR_RESET_API_COST','MODEL_AB'])
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))


def make_plan(output, inventory):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points = models.expanded_points(inventory, ms=MS)
    table, references, sources = load_catalog()
    screen = Prescreen(table)
    by_shape = defaultdict(list)
    for ref in references:
        w = Workload(**ref['workload'])
        if w.m > 8 and not rejection(Recipe.from_dict(ref['recipe']), w):
            by_shape[key(w)].append(ref)

    @lru_cache(maxsize=None)
    def generate(w):
        return moe_candidates(screen, w, RULES)

    capabilities = full_capabilities(points, output)
    caps = {tuple(r['request']):r for r in capabilities['rows']}
    bind_original(points, output)
    for point in points:
        w = Workload(**point['workload'])
        candidates, why = generate(replace(w, router=''))
        refs = by_shape[key(w)]
        recall = factored_recall(w, candidates, [Recipe.from_dict(r['recipe']) for r in refs])
        if recall['missing']:
            raise ValueError('MoE prefill blind rule gap; adjust rules, never append champions: '+
                             json.dumps(dict(workload=asdict(w), recall=recall)))
        if not candidates or len(candidates) > 512 or any(r.kind not in ('tc', 'full') for r in candidates):
            raise ValueError('unbounded or decode-only prefill candidate pool')
        full = [caps[(*[getattr(w, f) for f in
                ('qtype', 'n', 'k', 'm', 'experts', 'topk', 'channels')], c)] for c in full_configs(w)]
        rejected = []
        runnable = []
        for recipe in candidates:
            if recipe.kind == 'full' and not any(r['rc'] == 0 for r in full):
                rejected.append(dict(recipe=asdict(recipe), reason='SHIPPING_FULL_LAYOUT_REJECTS_ROUTED_ROWS'))
            else:
                runnable.append(asdict(recipe))
        point.update(candidates=runnable, proposed_candidates=[asdict(r) for r in candidates],
            rejected_candidates=rejected, historical_references=refs, rule_recall=recall,
            rule_reasons=why['selected_reasons'], full_capability=full, champion_controls=[],
            champion_injection=False, historical_times_reused=False,
            routed_rows=w.m*w.topk, maximum_rows_per_expert=w.m,
            token_m_is_not_average_expert_m=True)
        if not runnable:
            raise ValueError('no legal prefill route for '+point['id'])
        if point.get('original_recipe', {}).get('kind') == 'full':
            c = dict(point['original_recipe']['parameters'])['dequant_config']
            if not any(r['request'][-1] == c and r['rc'] == 0 for r in full):
                point['original_missing'] = 'CURRENT_FULL_SELECTION_FAILS_SHIPPING_LAYOUT'
    probe = resource_inventory(dict(points=points))
    # Plan a fixed execution smoke, NOT another all-config timing campaign.
    # M16 exercises the small-token large-M boundary, M128 Full's lower bound,
    # M5120 the large routed-row boundary. Test both concentrated/dispersed IDs.
    pilot = [p['id'] for p in points if p['workload']['m'] in (16,128,5120) and
             p['workload']['router'] in ('cluster','spread')]
    coverage = models.coverage_report(inventory, ms=MS)
    for row in coverage['models']:
        row['header_points'] = row['points']
        row['points'] = sum(any(c['model']==row['model'] for c in point['consumers']) for point in points)
    result = dict(schema='quactlize.moe-prefill-audit.v1', cohort=COHORT,
        status='HOST_PLAN_ONLY', points=points, resources=probe,
        capability=capabilities, rules=RULES, catalog_sources=sources,
        coverage=dict(models=coverage['models'],
            tokens=list(MS), requested_models=[m['name'] for m in inventory['models']],
            inventory_sha256=digest(inventory), actual_model_weights_used=False,
            profiles='SYNTHETIC_IDS_NOT_CAPTURED_MODEL_ROUTES', excluded=inventory['excluded']),
        counts=dict(points=len(points), families=len(points)//(len(MS)*5),
            proposed_arms=sum(len(p['proposed_candidates']) for p in points),
            candidate_arms=sum(len(p['candidates']) for p in points),
            full_unavailable_points=sum(not any(r['rc']==0 for r in p['full_capability']) for p in points),
            original_unavailable_points=sum(bool(p.get('original_missing')) for p in points),
            **probe['counts']),
        execution_pilot=dict(points=pilot, config_policy='AT_MOST_ONE_LEGAL_RECIPE_PER_ROUTE_NOT_A_SWEEP',
            required=['GPU_IDS_TO_MAPS_CONSUMED_BY_TC_AND_PROVIDER', 'CHANGED_IDS_NEGATIVE_AND_GGUF_ORACLE',
                      'GENERAL_GROUPED_V4_NOT_DECODE_BIND', 'ASYS_ISOLATED_PREPARE_MAIN_REDUCER_FINISH',
                      'ACTIVE_EXPERT_DEQUANT_NOT_ALL_E', 'PROVIDER_JIT_AND_WARMUP_OUTSIDE_PROFILE']),
        timing_method='ASYS_GPU_ACTIVITY', source_revision=source_stamp(),
        production_changed=False, performance_admitted=False, full_campaign_ready=False,
        pending=['RESOURCE_QUERY', 'GPU_ROUTE_AND_ASYS_EXECUTION_PILOT',
                 'NATIVE_LARGE_M_COMPONENT_ADAPTER', 'SCREEN_CONFIRM_AND_NATIVE_AB'])
    result['sha256'] = digest(result)
    return json.loads(json.dumps(result))
