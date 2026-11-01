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


def make_plan(output, inventory, *, source_points=None):
    # Reuse actual GGUF/TP coverage and the real current public selector.
    plan = models.make_plan(output, inventory, source_points=source_points)
    dense=source_points is not None and all(p['workload']['operation']=='dense' for p in source_points)
    table, references, sources = load_catalog(include_dense=True) if dense else load_catalog()
    options=dict(RULES,decode_exact_tiles=True) if dense else RULES
    screen = Prescreen(table)
    reference_by_geometry = defaultdict(list)
    for row in references:
        w = Workload(**row['workload'])
        if w.m <= 8 and not rejection(Recipe.from_dict(row['recipe']), w):
            reference_by_geometry[geometry(w)].append(row)

    @lru_cache(maxsize=None)
    def generate(w):
        rules, why = moe_candidates(screen, w,options)
        if dense:
            from dev.model_tuning.candidates import decode_delivery_peers
            rules+=decode_delivery_peers(w,rules,table.observations)
        common=list(models.candidate_recipes(w))
        if w.operation=='dense' and w.m>1:
            from dev.strategy.token_cta import GEOMETRIES,Q8_GEOMETRIES
            for c,warps,p in Q8_GEOMETRIES if w.qtype==8 else GEOMETRIES:
                for split in (1,2,4,8):
                    for window in (1,2) if w.qtype==8 else (None,):
                        r=Recipe.make('simt','fq',variant=5 if w.qtype==8 else 3,columns=c,
                            warps=warps,values=p,split=split,cta_tokens=w.m,
                            **(dict(cta_window=window) if window is not None else {}))
                        if not rejection(r,w):common.append(r)
        if w.operation=='dense' and w.qtype==8:
            # Columns and resident warps are independent axes of the public
            # Q8 reader. The two optimized address/token geometries below do
            # not replace the other legal row-reader combinations. No N/K or
            # historical winner lookup is used to generate this floor.
            for columns in (4,8):
                for warps in (4,8):
                    for split in (1,2,4,8):
                        r=Recipe.make('simt','fq',variant=5,columns=columns,
                                      warps=warps,values=4,split=split)
                        if not rejection(r,w):common.append(r)
            # The existing signed/window reader family is a declared S1
            # axis, independent of N/K and of the target's historical winner.
            for columns,warps in ((4,8),(8,4)):
                for body in (0,1,3,4,5):
                    common.append(Recipe.make('simt','fq',variant=5,columns=columns,warps=warps,
                                               values=4,split=1,q8_body=body))
        legal=[]
        for r in factor_candidates([*rules, *common]):
            if r.kind!='tc':
                try:spec(w,r)
                except ValueError as exc:
                    why.setdefault('carrier_rejections',[]).append(dict(recipe=asdict(r),reason=str(exc)))
                    continue
            legal.append(r)
        return legal, why

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
                     historical_references=refs, rule_seed_count=why['seed_prototype_count'],
                     carrier_rejections=why.get('carrier_rejections',[]))
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
        scope='REQUESTED_MODEL_TP_LOCAL_MOE_M1_TO_8_SIMT_TC_SCREEN_AND_CONFIRM_NOT_MODEL_AB',
        rules=options, catalog_sources=sources,
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
