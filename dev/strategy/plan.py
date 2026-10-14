"""Finite evidence-driven cohorts. No config Cartesian product or GPU use."""
from dataclasses import asdict, replace
from collections import defaultdict
import gzip
import json
from pathlib import Path

from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import load_sources, recipe as normalize
from dev.strategy.prescreen import Prescreen, compile_key, recall
from dev.strategy.space import helpers, factored_recall, grid_search

ROOT = Path(__file__).resolve().parents[2]


def profile_winners():
    """Do not prune per-router winners just because a minimax table chose another."""
    data=json.loads(gzip.decompress((ROOT/'docs/measurements/smallm_matched_20260915.json.gz').read_bytes()))
    out=defaultdict(dict)
    for row in data['points']:
        if not row.get('winner') or row['status']!='MEASURED_POOL_CLOSED':continue
        p=row['point_spec'];c=data['candidates'][row['winner']['candidate']]
        key=(p['q'],p['mode'],p['n'],p['k'],p['experts'],p['topk'],p['channels'],p['tokens'],p['compute'],
             p.get('router','real') if p['mode'] else '')
        if c['kind']=='tc':
            r=normalize(dict(c['parent'],kind='tc',split=c['split'],grid_mode=c['grid_mode'],grid_b=c['grid_b']))
        else:r=normalize(dict(c['recipe'],kind=c['kind']))
        out[key][r.key]=r
    return out


def workloads(table, cohort):
    if cohort == 'pilot':
        return [Workload(12, 512, 2048, 1), Workload(8, 2048, 4096, 1),
                Workload(14, 1024, 5120, 8), Workload(12, 1024, 5120, 512),
                Workload(12, 1024, 2048, 1, 'indexed', 'bf16', experts=256, topk=8),
                Workload(13, 2048, 512, 4, 'indexed', 'bf16', experts=256, topk=8, channels=8)]
    found = {}
    for row in table.observations:
        w = row.workload
        if row.source not in ('effective-decode', 'prefill-components'):
            continue
        if cohort == 'decode' and w.m > 8 or cohort == 'prefill' and w.m <= 8:
            continue
        # Runtime contracts: F16 dense, BF16 MoE. Keep other observations in
        # the catalog, but never relabel their timing as this compute type.
        if w.operation == 'dense' and w.compute != 'f16':
            continue
        if w.operation != 'dense' and w.compute != 'bf16':
            if row.source != 'prefill-components':
                continue
            w = Workload(**(asdict(w) | {'compute': 'bf16'}))
        # Expand only already-known router-sensitive cells, not every shape
        # times every profile. Preserve those historical profile winners.
        if row.details.get('router_sensitive') and w.operation != 'dense':
            for profile in row.details['profiles']:
                routed = replace(w, router=profile)
                found[routed.key] = routed
        else:
            found[w.key] = w
    return sorted(found.values(), key=lambda w: (w.operation, w.qtype, w.n, w.k, w.m, w.channels))


def component_champions(points):
    """Retain each route's winner, not every stable losing old configuration."""
    result=defaultdict(dict)
    for p in points:
        routes=defaultdict(list)
        for c in p['candidates']:
            if c['stable']:routes[c['route']].append(c)
        key=(p['q'],p['n'],p['k'],p['experts'],p['tokens'],p['profile'] if p['experts']>1 else '')
        for candidates in routes.values():
            # Comparable only within this one historical fixture/profile.
            c=min(candidates,key=lambda c:c['cost_us'])
            r=(Recipe.make('full','full',dequant_config=c['dequant_config'],provider='bf16-default')
               if c['route']==2 else normalize(c['tactic'],c['route'],c['dequant_config']))
            result[key][r.key]=r
    return result


def proposals(screen, w, limit, winners, components, optional=1, per_stratum=2, factorized=True, k_weight=1.0,cross_format=False,reader_strata=False,stencil='none'):
    mode=2 if w.operation=='indexed' else 0
    profile=w.router or ('real' if mode else '')
    key=(w.qtype,mode,w.n,w.k,w.experts,w.topk,w.channels,w.m,w.compute,profile)
    required=[(r,'HISTORICAL_ROUTER_WINNER') for r in winners.get(key,{}).values()]
    ckey=(w.qtype,w.n,w.k,w.experts,w.m,profile)
    # The old component board has F16 compute. Geometry can be proposed for
    # BF16, but cannot become a BF16 champion used to certify recall.
    required += [(r,'HISTORICAL_ROUTE_WINNER_GEOMETRY' if w.compute=='f16' else 'TRANSFERRED_F16_ROUTE_WINNER')
                 for r in components.get(ckey,{}).values()]
    # Generate before revealing references. The caller/old champions are an
    # independent control pool, never a way to make the rule's recall green.
    candidates, report=screen.select(w,limit=limit,optional=optional,blind=True,per_stratum=per_stratum,
                                    factorized=factorized,k_weight=k_weight,cross_format=cross_format,reader_strata=reader_strata,stencil=stencil)
    references, ref_report=screen.select(w,mandatory=required,limit=limit,optional=0)
    reference_keys={key for key,why in ref_report['selected_reasons'].items()
                    if any(x.startswith('HISTORICAL_') for x in why)}
    references=[r for r in references if r.key in reference_keys]
    report['reference_rejections']=[r for r in ref_report['rejected'] if r['mandatory']]
    report['literal_recall']=recall(candidates,references)
    report['recall']=(factored_recall(w,candidates,references) if factorized else report['literal_recall'])
    return candidates, report, references


def make_plan(cohort='pilot', limit=8, optional=1, per_stratum=2, factorized=True, k_weight=1.0,cross_format=False,reader_strata=False,stencil='warps'):
    if cohort not in ('pilot', 'decode', 'prefill', 'all') or not 2 <= limit <= 32:
        raise ValueError('unknown cohort / invalid bounded candidate limit')
    table, sources = load_sources()
    winners=profile_winners()
    components=component_champions(json.loads((ROOT/'docs/measurements/kpack_component_policy_20260914.json').read_text())['points'])
    screen=Prescreen(table)
    points = []
    for w in workloads(table, cohort):
        candidates, reasons, references = proposals(screen, w, limit, winners, components, optional, per_stratum, factorized,k_weight,cross_format,reader_strata,stencil)
        points.append(dict(id=w.key, workload=asdict(w), baseline='original',
                           candidates=[asdict(c) for c in candidates], pruning=reasons,
                           expansion_search=helpers(w,candidates) if factorized else [],
                           grid_search={r.key:g for r in candidates
                                        if factorized and (g:=grid_search(w,r))},
                           champion_controls=[asdict(c) for c in references],
                           controls_are_not_rule_candidates=True,
                           final_admission='ORIGINAL_WIRING_MODEL_AB_REQUIRED'))
    plan = dict(schema='quactlize.component-sweep.v1', cohort=cohort, points=points,
                catalog=table.identity, evidence=sources, candidate_limit=limit, optional_per_shape=optional,
                prototypes_per_rule_stratum=per_stratum,
                factorized_search=factorized,
                k_distance_weight=k_weight,
                reader_strata=reader_strata,
                neighbor_stencil=stencil,
                measurement='ISOLATED_KERNEL_COMPONENTS_ONLY',
                helper_timing='DEDUP_EXACT_IMPLEMENTATION_AND_SEMANTIC_KEYS',
                production_changed=False, inference_tuning=False,
                small_m_full_dequant=False,
                original_baseline_mandatory=True,
                rule_generation='LEAVE_NK_FAMILY_OUT_NO_CHAMPION_INJECTION',
                candidate_limit_is_hard=False,
                retained_winners_and_family_split_coverage_may_expand_pool=True,
                derived_cost_scope='SUM_OF_ISOLATED_MEASUREMENTS_NOT_E2E')
    modules=set();control_modules=set();q4_modules=set();expansions=set();extra_controls=0
    for point in points:
        w=Workload(**point['workload'])
        candidate_keys={Recipe.from_dict(c).key for c in point['candidates']}
        extra_controls+=sum(Recipe.from_dict(c).key not in candidate_keys for c in point['champion_controls'])
        for c in point['candidates']:
            key=compile_key(w,Recipe.from_dict(c))
            if key:modules.add(key)
            if c['kind']=='q4':q4_modules.add(key)
        for c in point['champion_controls']:
            key=compile_key(w,Recipe.from_dict(c))
            if key:control_modules.add(key)
            if c['kind']=='q4':q4_modules.add(key)
        for helper in point['expansion_search']:
            for config in helper['configs']:
                expansions.add(digest(helper|dict(configs=[config])))
    rule_arms=sum(len(p['candidates']) for p in points)
    plan['work_counts']=dict(points=len(points),rule_producer_arms=rule_arms,
        separate_champion_control_arms=extra_controls,
        total_arms_including_original=len(points)+rule_arms+extra_controls,
        resource_expanded_grid_arms='RESOLVE_AFTER_OCCUPANCY_AND_GRID_DEDUP_NOT_COUNTED_YET',
        unique_expansion_jobs=len(expansions),
        explicit_unique_compile_recipes=len(modules|control_modules),
        q4_shape_compute_specializations=len(q4_modules),
        q4_explicit_compile_driver='PENDING_NOT_FREE_INSTALLED_IMAGE_REUSE',
        rule_unique_compile_recipes=len(modules),control_only_compile_recipes=len(control_modules-modules),
        original_compile_recipes='RESOLVED_BEFORE_DEVICE_RUN',
        expanded_points=sum(p['pruning']['exceeds_limit_to_preserve_winners'] for p in points),
        rejected_mandatory=sum(sum(r['mandatory'] for r in p['pruning']['rejected']) for p in points),
        rule_miss_points=sum(p['pruning']['recall']['status']=='RULE_MISS' for p in points),
        predicted_gpu_seconds=None,eta_source='RESOURCE_PREFLIGHT_AND_PILOT_WALL_TIME_REQUIRED_NOT_KERNEL_US')
    plan['plan_sha256'] = digest(plan)
    return json.loads(json.dumps(plan))
