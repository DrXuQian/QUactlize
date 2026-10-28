"""Blind three-model M1..8 config sweep, separate from SIMT-only completion."""
from collections import defaultdict
from dataclasses import asdict, replace
from functools import lru_cache
import json

from dev.strategy import moe_model_plan as models
from dev.strategy.moe_plan import load_catalog, moe_candidates, RULES
from dev.strategy.moe_token_pair_plan import production_entry
from dev.strategy.prescreen import Prescreen, rejection
from dev.strategy.refresh import request
from dev.strategy.resources import module_spec
from dev.strategy.space import factored_recall, factor_candidates
from dev.strategy.vector import spec
from quactlize.dispatch.planning import FIELDS, plan_simt_configs
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

COHORT = 'moe-config-models'


def geometry(w):
    return replace(w, router='').key


def make_plan(output, inventory):
    # Reuse actual GGUF/TP coverage and the real current public selector.
    plan = models.make_plan(output, inventory)
    table, references, sources = load_catalog()
    screen = Prescreen(table)
    reference_by_geometry = defaultdict(list)
    for row in references:
        w = Workload(**row['workload'])
        if w.m <= 8 and not rejection(Recipe.from_dict(row['recipe']), w):
            reference_by_geometry[geometry(w)].append(row)

    @lru_cache(maxsize=None)
    def generate(w):
        rules, why = moe_candidates(screen, w, RULES)
        return factor_candidates([*rules, *models.candidate_recipes(w)]), why

    queries, pending = {}, []
    for point in plan['points']:
        w = Workload(**point['workload'])
        candidates, why = generate(replace(w, router=''))
        refs = reference_by_geometry[geometry(w)]
        recall = factored_recall(w, candidates, [Recipe.from_dict(r['recipe']) for r in refs])
        if recall['missing']:
            raise ValueError('MoE rule coverage gap (fix rules, never append champions): '+
                             json.dumps(dict(workload=asdict(w), recall=recall)))
        if not any(r.kind == 'tc' for r in candidates) or not any(r.kind == 'simt' for r in candidates):
            raise ValueError('rule pool must include both TC and SIMT: '+str(request(w)))
        for r in candidates:
            if r.kind == 'simt':
                queries[tuple(request(w)+[dict(r.parameters)[f] for f in FIELDS])] = None
        pending.append((point, w, candidates, refs, recall, why))
    queries.update(zip(queries, plan_simt_configs(output, queries)))
    tc, vectors = {}, {}
    for point, w, candidates, refs, recall, why in pending:
        # The current control is independent of the blind pool, never a seed.
        original = point['arms'][0]
        arms = {Recipe.from_dict(original['recipe']).key: dict(original)}
        for r in candidates:
            if r.key in arms:
                arms[r.key]['roles'] = sorted(set(arms[r.key]['roles']+['rule']))
                continue
            if r.kind == 'simt':
                q = queries[tuple(request(w)+[dict(r.parameters)[f] for f in FIELDS])]
                if q['status']:
                    raise ValueError('generated SIMT is not a legal public body: '+str(q))
                entry = production_entry(q)
            else:
                entry = ('cutlass::device_kernel' if r.kind == 'tc' else
                         'quactlize::execution::q4_decode::kernel_bf16')
            item = module_spec(w, r) if r.kind == 'tc' else spec(w, r)
            arms[r.key] = dict(id=r.key, recipe=asdict(r), roles=['rule'], aliases=[],
                required=r.kind != 'tc', reasons=why['selected_reasons'].get(r.key, ['COMMON_SIMT_RULE']),
                spec=item, module=digest(item), expected_producer=entry)
        point.update(arms=list(arms.values()), config_sweep=True, rule_recall=recall,
                     historical_references=refs, rule_seed_count=why['seed_prototype_count'])
        point['sf_configs'] = sorted({max(0, dict(a['recipe']['parameters']).get('dequant_config', -1))
            for a in point['arms'] if a['recipe']['route'] == 'sf' and w.qtype != 8})
        # No historical small-M K-quant SF winner is hidden behind this guard.
        # A future one needs the explicit independent expansion search first.
        if len(point['sf_configs']) > 1 or any(
                r['recipe']['route'] == 'sf' and w.qtype != 8 and
                max(0, dict(r['recipe']['parameters']).get('dequant_config', -1)) not in point['sf_configs']
                for r in refs):
            raise ValueError('multiple SF helpers need a separate expansion sweep; not silently omitted')
        for a in point['arms']:
            (tc if a['recipe']['kind'] == 'tc' else vectors)[a['module']] = a['spec']
    plan.update(cohort=COHORT, tc=tc, vectors=vectors,
        scope='THREE_MODEL_TP_LOCAL_MOE_M1_TO_8_SIMT_TC_SCREEN_AND_CONFIRM_NOT_MODEL_AB',
        rules=RULES, catalog_sources=sources,
        protocol=dict(screen=[4, 3], confirm=[6, 5], screen_live_arms=12,
                      finalist_window_pct=5, finalist_limit=32,
                      grid='EXPAND_AFTER_COMPILED_RESOURCE_QUERY_DEDUP_ACTUAL_LAUNCH'),
        remaining=['BF16_STORAGE_ENDPOINTS', 'FUSED_GATE_UP_SWIGLU', 'LARGE_M', 'NATIVE_AND_MODEL_AB'])
    plan['counts'].update(arms=sum(len(p['arms']) for p in plan['points']), tc_modules=len(tc),
                          vector_modules=len(vectors), timed_grid_arms='RESOURCE_EXPANSION_PENDING')
    plan['sources'].update({p: sha(models.ROOT/p) for p in (
        'dev/strategy/moe_sweep_plan.py', 'dev/strategy/moe_sweep.py',
        'dev/strategy/moe_plan.py', 'dev/strategy/prescreen.py', 'dev/strategy/space.py')})
    plan['sha256'] = digest({k: v for k, v in plan.items() if k != 'sha256'})
    return json.loads(json.dumps(plan))
