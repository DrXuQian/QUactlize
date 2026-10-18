"""Per-shape, pre-JIT candidate screening. No GPU work or online tuning.

Historical winners are separate controls, never injected into blind rules.
Transferred geometry is a proposal; neither its old latency nor its winner
label becomes evidence on another shape, precision, router or device.
"""
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import math

from quactlize.runtime.compiler import validate_parent
from quactlize.runtime.tuning import digest
from quactlize.dispatch.table import Recipe
from dev.strategy.space import prototype, split_variants, factor_candidates, rebind_template, producer_key, grid_search
from dev.strategy.neighbors import MODES, one_axis
from dev.strategy.prefill_candidates import template_family as prefill_template_family


def parent(recipe, workload):
    p = dict(recipe.parameters)
    return {k: p[k] for k in ('symbol', 'tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn')} | dict(
        qtype=p.get('qtype', workload.qtype),
        route=recipe.route + ('-dense' if workload.operation == 'dense' else '-grouped'),
        persistent=p.get('persistent', 0 if workload.operation != 'dense' and recipe.route == 'fq' else -1))


@lru_cache(maxsize=65536)
def rejection(recipe, workload, max_threads=1024):
    """Hard semantic/launch filters only; no invented occupancy or SMEM cost.

The 1024-thread ceiling is the current builder's 32-warps constraint, not a
guessed device attribute. A probed device may impose a tighter ceiling. Exact
shared storage/register occupancy remain the compiled resource query's job.
"""
    if not recipe.eligible(workload):
        return 'SHAPE_PRECISION_OR_PIPELINE_CONTRACT'
    p = dict(recipe.parameters)
    if recipe.kind == 'tc':
        try:
            c = parent(recipe, workload)
            validate_parent(c)
        except (ValueError, KeyError) as exc:
            return 'PARENT_CONTRACT: ' + str(exc)
        # ppu_tactic_space.hpp::cta_warps, asserted against TiledMma there.
        threads = 32 * (p['tm'] // p['wm']) * (p['tn'] // p['wn'])
        if threads > min(1024, max_threads):
            return 'CTA_THREAD_LIMIT'
        persistent = p.get('grid_mode', 0) >= 2
        if workload.operation == 'dense' and persistent and (recipe.route == 'fq' or p['split'] != 1):
            return 'DENSE_PERSISTENT_REQUIRES_SF_S1'
        if workload.operation != 'dense' and recipe.route == 'fq' and p.get('persistent', 0) != int(persistent):
            return 'GROUPED_SCHEDULE_DIFFERS_FROM_COMPILED_PARENT'
    if recipe.kind == 'simt':
        if (p.get('variant') not in range(6) or p.get('columns') not in (4, 8) or
                p.get('warps') not in (2, 4, 8) or p.get('values') not in (2, 4, 8) or
                p['columns'] * p['values'] > 32 or p.get('split') not in (1, 2, 4, 8)):
            return 'SIMT_CONFIG_CONTRACT'
        if workload.qtype == 8 and p['variant'] & 2:
            return 'Q8_SIMT_VARIANT'
        if p['variant'] >= 4 and (workload.qtype != 8 or workload.storage != 'f32'):
            return 'Q8_F32_VECTOR_ONLY'
        if p['warps'] * 32 > max_threads:
            return 'CTA_THREAD_LIMIT'
    if recipe.kind == 'q4':
        reader, variant, warps, values, columns = (p.get(k) for k in ('reader','variant','warps','values','columns'))
        if (any(type(x) is not int for x in (reader,variant,warps,values,columns)) or
                columns not in (4,8) or not 1<=warps<=32 or columns*values>32 or
                not (reader==0 and variant in (0,1) and values==1 or
                     reader==1 and variant in range(4) and values in (2,4) or
                     reader==2 and variant in range(8) and values in (4,8))):
            return 'Q4_READER_CONTRACT'
        if reader and (warps*32//columns)%8:
            return 'Q4_WORKER_GROUP_ALIGNMENT'
        if warps*32>max_threads:
            return 'CTA_THREAD_LIMIT'
        if workload.storage!='f32' and not (workload.storage=='bf16' and workload.compute=='bf16'):
            return 'Q4_DECODE_STORAGE_CONTRACT'
    return None


def compile_key(workload, recipe):
    """One TC parent serves many M/N/K, Split-K and grid recipes.

This is a planning identity, NOT a substitute for the compiler's source/SDK
cache key. SIMT is in the installed execution image; cuBLAS has no new parent.
"""
    if recipe.kind == 'simt' or recipe.kind == 'full' and workload.operation == 'dense':
        return None
    if recipe.kind == 'q4':
        # The shipping image contains only selected (N,K,reader) closures,
        # not every legal proposal. M/E/router are runtime fields. Count the
        # required specialization even while its explicit carrier is pending.
        return digest(dict(provider='q4-explicit-specialization',n=workload.n,k=workload.k,
                           compute=workload.compute,recipe=producer_key(recipe)))
    if recipe.kind == 'full':
        return digest(dict(provider='installed-deepgemm', workload=asdict(workload)))
    endpoint = ('grouped-typed' if workload.operation != 'dense' else
                'dense-native' if workload.compute == 'f16' and workload.m > 8 else 'dense-typed')
    return digest(dict(parent=parent(recipe, workload), compute=workload.compute, endpoint=endpoint))


def semantic_family(w):
    # Precision/storage transfers are proposal-only; fusion/layout/strides and
    # indexed channel semantics are never relaxed to fill a candidate quota.
    return (w.qtype, w.operation, w.experts, w.topk, w.channels, w.fusion,
            w.input_stride == w.k, w.output_stride == w.n)


def proposal_family(w, cross_format=False):
    # Configs do not bake E/top-k/channels into a template. Rebinding those
    # request fields is legal for a geometry proposal, but never transfers a
    # latency or a router-specific performance claim. A cross-format proposal
    # re-instantiates the target K-quant template; Q8 and specialized Q4
    # readers retain their own contracts.
    dtype = 'kquant' if cross_format and w.qtype != 8 else w.qtype
    return (dtype, w.operation, w.fusion, w.input_stride == w.k, w.output_stride == w.n)


def portable_sf_family(w):
    # SF tile geometry is shared by dense and indexed builders. Only its
    # template may cross this boundary: endpoint, metadata precision, launch
    # resources, routing helpers and timings must be resolved on the target.
    return (w.qtype, w.fusion, w.input_stride == w.k, w.output_stride == w.n)


def seed_identity(w, recipe, stencil):
    """Do not spend multiple seed slots on an axis already expanded later.

    This is a SELECTION identity only, never a timing or compilation key.
    Keep the ordinary/persistent donor strata and distinct compiled bodies;
    only ignore axes whose whole domain is explicitly generated below.
    """
    p = dict(recipe.parameters)
    p.pop('symbol', None)
    if recipe.kind == 'simt' and stencil == 'warps':
        p.pop('warps', None)  # entire closures agree only for the warp-only stencil
    if recipe.kind == 'tc' and grid_search(w, recipe):
        p.pop('grid_b', None)
        p['grid_mode'] = int(p.get('grid_mode', 0) != 0)
    return digest(dict(kind=recipe.kind, route=recipe.route, parameters=p))


class Prescreen:
    def __init__(self, table):
        self.rows = defaultdict(list)
        self.proposals = defaultdict(list)
        self.templates = defaultdict(list)
        self.portable_sf = defaultdict(list)
        self.prefill_sf = defaultdict(list)
        for row in table.observations:
            self.rows[semantic_family(row.workload)].append(row)
            self.proposals[proposal_family(row.workload)].append(row)
            self.templates[proposal_family(row.workload,True)].append(row)
            if (row.recipe.kind == 'tc' and row.recipe.route == 'sf' and
                    row.workload.m > 8 and not dict(row.recipe.parameters).get('ap')):
                self.prefill_sf[prefill_template_family(row.workload)].append(row)
            if (row.recipe.kind == 'tc' and row.recipe.route == 'sf' and
                    row.workload.operation in ('dense', 'indexed') and row.workload.m <= 8 and
                    not dict(row.recipe.parameters).get('ap')):
                self.portable_sf[portable_sf_family(row.workload)].append(row)

    def select(self, w, *, mandatory=(), limit=8, optional=1, max_threads=1024, blind=False,
               per_stratum=1, factorized=True, k_weight=1.0, cross_format=False,
               reader_strata=False, stencil='none', prefill_breadth=True):
        if not 2 <= limit <= 32 or not 0 <= optional <= 8 or not 0 < max_threads <= 1024:
            raise ValueError('invalid pre-JIT candidate limits')
        if not 1 <= per_stratum <= 16:
            raise ValueError('invalid rule stratum width')
        if not math.isfinite(k_weight) or not 0.25 <= k_weight <= 8:
            raise ValueError('invalid K-distance weight')
        if blind and mandatory:
            raise ValueError('blind rule generation cannot accept champion injection')
        if stencil not in MODES or stencil != 'none' and not (blind and factorized):
            raise ValueError('neighbor stencil requires blind factorized generation')
        protected, ranked, distant, excluded, reasons = {}, {}, {}, {}, defaultdict(set)

        def keep(recipe, reason):
            bad = rejection(recipe, w, max_threads)
            if bad:
                excluded[recipe.key] = dict(config=asdict(recipe), reason=bad, mandatory=True)
                return
            protected[recipe.key] = recipe
            reasons[recipe.key].add(reason)

        for recipe, reason in mandatory:
            keep(recipe, reason)
        rows = (self.templates.get(proposal_family(w,True), ()) if blind and factorized and cross_format else
                self.proposals.get(proposal_family(w), ()) if blind and factorized else
                self.rows.get(semantic_family(w), ()))
        nearby_count = 0
        for row in rows:
            ow, r = row.workload, row.recipe
            # Exclude the entire weight family, not just this M. Looking up
            # the same N/K at M+1 would still leak the held-out answer.
            if blind and (ow.n, ow.k) == (w.n, w.k):
                continue
            if blind and factorized:
                r = prototype(r)
                if cross_format:
                    r = rebind_template(r,w.qtype)
            exact = ((ow.n, ow.k, ow.m, ow.compute, ow.storage) ==
                     (w.n, w.k, w.m, w.compute, w.storage))
            # These imports contain selected recipes, not the old raw pool.
            # A range covering another router remains a protected challenger,
            # not an exact latency for this routed input.
            if exact and row.scope != 'PROPOSAL':
                keep(r, 'HISTORICAL_SELECTED:' + row.source)
                continue
            far_m = ((ow.m<=8)!=(w.m<=8) or
                     w.m>8 and r.kind!='full' and max(ow.m,w.m)>2*min(ow.m,w.m))
            if ((far_m and not (blind and factorized and r.kind=='tc' and w.m<=8)) or
                    (not blind and (max(ow.n,w.n)>2*min(ow.n,w.n) or max(ow.k,w.k)>2*min(ow.k,w.k)))):
                continue
            nearby_count += 1
            bad = rejection(r, w, max_threads)
            if bad:
                excluded.setdefault(r.key, dict(config=asdict(r), reason=bad, mandatory=False))
                continue
            distance = (abs(math.log2(ow.n / w.n)) + k_weight * abs(math.log2(ow.k / w.k)) +
                        2 * abs(math.log2(ow.m / w.m)))
            if blind and factorized and cross_format and ow.qtype != w.qtype:
                bits={8:8,10:2,11:3,12:4,13:5,14:6}
                distance += abs(math.log2(bits[ow.qtype]/bits[w.qtype]))
            rank = (ow.compute != w.compute, ow.storage != w.storage, distance,
                    ow.channels != w.channels, -row.priority, r.key)
            identity=producer_key(r) if blind and factorized else r.key
            target=distant if far_m else ranked
            if identity not in target or rank < target[identity][0]:
                target[identity] = (rank, r, row.source)

        # Full expansion is a supported algorithm, not something to expose
        # only when the target's old table already says it wins. These are
        # globally applicable implementations from dequant/api.h, not per-N/K
        # exceptions. Their actual expansion times must be profiled once.
        if blind and w.m >= 128 and w.qtype != 8:
            for config in ((4,) if factorized else (4,12) if w.qtype in (12,13) else (4,)):
                r=Recipe.make('full','full',dequant_config=config,provider='bf16-default')
                identity=producer_key(r) if factorized else r.key
                ranked.setdefault(identity,((False,False,0,False,0,r.key),r,'FULL_DEQUANT_API_FAMILY'))

        # Coverage precedes the optional-neighbor budget. Otherwise "four
        # configs" can silently omit SIMT, TC or a Split-K alternative whose
        # reducer cost was never comparable. Keep a representative for each
        # historically available family/route/split, not a Cartesian product
        # of every tile, stage and provider. These are still proposals, not a
        # proof that all other candidates lose.
        def tag(r):
            p=dict(r.parameters)
            extra=()
            if blind:
                if r.kind=='tc':
                    # Prefill spans narrow row tiles and throughput tiles.
                    # Two nearest seeds in a single M>8 bucket can otherwise
                    # erase either family. Decode retains its M8/other split.
                    row_band = (0 if p['tm']==8 else 1 if p['tm']<=32 else 2) if w.m>8 else int(p['tm']!=8)
                    # Decode trades wide output tiles against more CTAs.
                    # Preserve both width regimes instead of allowing two
                    # similar narrow-tile donors to hide the wide family.
                    column_band=int(p['tn']>64) if w.m<=8 else 0
                    extra=(row_band,column_band,bool(p.get('ap')),p.get('grid_mode',0)!=0)
                elif r.kind=='simt':extra=(p['variant'] if reader_strata else p['variant']>=4,)
                elif r.kind=='q4':extra=(p['reader'],)
                elif r.kind=='full':extra=(p['dequant_config']>=10,)
            return (r.kind, r.route, 1 if blind and factorized else p.get('split', 1), *extra)

        # Broader M donors fill ONLY a missing geometry stratum in a route
        # already available nearby. They never displace a nearby seed or
        # introduce a new SF/full route just to enlarge the pool. Unrestricted
        # cross-M ranking regressed in R17; preserve that counterexample.
        near_tags={tag(r) for _,r,_ in ranked.values()}
        near_routes={(r.kind,r.route) for _,r,_ in ranked.values()}
        for key,(rank,r,source) in distant.items():
            if tag(r) not in near_tags and (r.kind,r.route) in near_routes:
                ranked.setdefault(key,(rank,r,'VACANT_STRATUM_CROSS_M:'+source))

        # A leave-NK-out catalog can contain zero local TC donors (Q8 indexed
        # was one such case). Absence of observations is not evidence that
        # SIMT wins. Borrow SF templates only when the entire TC family is
        # absent, not to fill every tile stratum in an already covered family.
        # Never displace local seeds, borrow the withheld N/K, or use its old
        # latency/winner to rank proposals. FQ/AP/grouped-only recipes are not
        # portable through this rule. The normal resource and numeric gates
        # still apply after instantiating the target typed endpoint.
        if (blind and factorized and w.m <= 8 and w.operation in ('dense', 'indexed') and
                not any(r.kind == 'tc' for _,r,_ in ranked.values())):
            for row in self.portable_sf.get(portable_sf_family(w), ()):
                ow = row.workload
                if ow.operation == w.operation or (ow.n, ow.k) == (w.n, w.k):
                    continue
                r = prototype(row.recipe)
                if rejection(r, w, max_threads):
                    continue
                distance = (abs(math.log2(ow.n / w.n)) + k_weight * abs(math.log2(ow.k / w.k)) +
                            2 * abs(math.log2(ow.m / w.m)))
                rank = (ow.compute != w.compute, ow.storage != w.storage, distance,
                        ow.channels != w.channels, -row.priority, r.key)
                key = producer_key(r)
                if key not in ranked or rank < ranked[key][0]:
                    ranked[key] = (rank, r, 'MISSING_TC_FAMILY_CROSS_OPERATION:' + row.source)

        covered = {tag(r) for r in protected.values()}
        strata=defaultdict(int)
        for r in protected.values():strata[tag(r)]+=1
        coverage = {}
        seed_ids = set()
        boundary = {}
        pool = sorted((v for key, v in ranked.items() if key not in protected), key=lambda v:v[0])
        for rank, r, source in pool:
            seed_id = seed_identity(w, r, stencil) if blind and factorized else r.key
            tied = blind and factorized and rank[:-1] == boundary.get(tag(r))
            if (strata[tag(r)] < per_stratum or tied) and seed_id not in seed_ids:
                coverage[r.key] = r
                seed_ids.add(seed_id)
                covered.add(tag(r))
                strata[tag(r)]+=1
                if strata[tag(r)] == per_stratum:
                    boundary[tag(r)] = rank[:-1]
                reasons[r.key].add('FAMILY_ROUTE_SPLIT_COVERAGE:' + source)
                if strata[tag(r)] > per_stratum:
                    reasons[r.key].add('BOUNDARY_DISTANCE_TIE_NOT_HASH_PRUNED')
        aliases = [v for v in pool if v[1].key not in coverage and blind and factorized and
                   seed_identity(w, v[1], stencil) in seed_ids]
        pool = [v for v in pool if v[1].key not in coverage and v not in aliases]
        room = max(0, min(optional, limit - 1 - len(protected) - len(coverage)))
        picked = [r for _, r, _ in pool[:room]]
        for _, r, source in pool[:room]:
            reasons[r.key].add('BOUNDED_NEIGHBOR_PROPOSAL:' + source)
        selected = sorted(protected.values(), key=lambda r: r.key) + list(coverage.values()) + picked
        prefill_added = 0
        delivery_added = 0
        if blind and factorized and prefill_breadth and w.m > 8:
            from dev.strategy.prefill_candidates import supplement
            additions, why = supplement(w, selected,
                [*rows, *self.prefill_sf.get(prefill_template_family(w), ())],
                max_threads=max_threads, k_weight=k_weight)
            selected += additions
            prefill_added = len(additions)
            for key, values in why.items(): reasons[key].update(values)
            from dev.strategy.prefill_candidates import delivery_peers
            additions, why = delivery_peers(w, selected, rows, max_threads=max_threads)
            selected += additions
            delivery_added = len(additions)
            for key, values in why.items(): reasons[key].update(values)
        if blind and factorized:
            from dev.strategy.q4_space import proposals as q4_proposals
            known = {producer_key(r) for r in selected}
            for r in q4_proposals(w):
                if rejection(r,w,max_threads) or producer_key(r) in known:
                    continue
                selected.append(r);known.add(producer_key(r))
                reasons[r.key].add('Q4_API_READER_FLOOR_FROM_TARGET_K_NOT_CHAMPION')
        seed_count=len(selected)
        if blind and factorized and stencil != 'none':
            neighbors=[]
            # Snapshot the seeds: never recursively expand neighbors, or
            # take target champion controls as a seed after recall fails.
            seen={producer_key(r) for r in selected}
            for seed in tuple(selected):
                for r,axis in one_axis(seed,stencil):
                    bad=rejection(r,w,max_threads)
                    if bad:
                        excluded.setdefault(r.key,dict(config=asdict(r),reason=bad,mandatory=False))
                        continue
                    identity=producer_key(r)
                    if identity not in seen:
                        neighbors.append(r);seen.add(identity)
                    reasons[r.key].add('ONE_AXIS_FROM_BLIND_SEED:'+axis+':'+seed.key)
            selected+=neighbors
        prototype_count = len(selected)
        if blind and factorized:
            variants=[]
            for r in selected:
                for variant in split_variants(w,r):
                    bad=rejection(variant,w,max_threads)
                    if bad:
                        excluded[variant.key]=dict(config=asdict(variant),reason=bad,mandatory=False)
                        continue
                    variants.append(variant)
                    reasons[variant.key].update(reasons[r.key])
                    reasons[variant.key].add('RUNTIME_SPLIT_LEGALITY_FROM_TARGET_K')
            selected=factor_candidates(variants)
        expanded = 1 + len(protected) + len(coverage) > limit
        report = dict(stage='PRE_JIT_PER_SHAPE', original_mandatory=True,
            rule_generation='LEAVE_NK_FAMILY_OUT' if blind else 'REFERENCE_CONTROL_INVENTORY',
            rule_revision=(27 if prefill_breadth and w.m>8 else 25) if blind and factorized else 2 if blind else None,per_stratum=per_stratum,
            prefill_geometry_supplement=prefill_added,
            prefill_delivery_peers=delivery_added,
            cross_format_templates=bool(blind and factorized and cross_format),
            reader_strata=reader_strata,
            neighbor_stencil=stencil,seed_prototype_count=seed_count,
            distance_weights=dict(n=1.0,k=k_weight,m=2.0),
            stratum_boundary_ties='KEEP_EQUAL_RANK_NOT_HASH_ORDER' if blind and factorized else None,
            prototype_count=prototype_count, split_variants_generated=bool(blind and factorized),
            seed_aliases_covered_by_expansion=len(aliases),
            champion_injection=False if blind else bool(protected),
            withheld_family=[w.n,w.k] if blind else None,
            historical_selected=len(protected), optional_selected=len(picked),
            coverage_selected=len(coverage), coverage_tags=sorted(covered),
            total_with_original=1 + len(selected), requested_limit=limit,
            exceeds_limit_to_preserve_winners=expanded,
            candidate_limit_is_hard=False, search_complete=False,
            completion='BOUNDED_COVERAGE_NOT_GLOBAL_OPTIMUM; ORIGINAL_WIRING_AB_REQUIRED',
            nearby_observations=nearby_count, unique_nearby=len(pool),
            selected_reasons={k: sorted(v) for k, v in sorted(reasons.items())},
            rejected=sorted(excluded.values(), key=lambda r: (not r['mandatory'], r['reason'], str(r['config']))),
            unselected_optional=[dict(config=asdict(r), reason='DEFERRED_LOCAL_NEIGHBOR_NOT_PROVEN_LOSER')
                                 for _, r, _ in pool[room:]],
            resource_admission='COMPILED_SHARED_STORAGE_AND_OCCUPANCY_STILL_REQUIRED',
            measured_latency_reused=False)
        return selected, report


def semantic_config(recipe):
    """Parent names can be aliases; geometry/reader/route/precision may not.

The public request carries precision. Symbol spelling is the only ignored
field here: both the exact compiler tuple and all runtime recipe axes remain.
This audits config recall, not identity/equality of compiled machine code.
"""
    p=dict(recipe.parameters)
    p.pop('symbol',None)
    if recipe.kind=='tc':
        p.setdefault('dequant_config',-1)
    return digest(dict(kind=recipe.kind,route=recipe.route,parameters=p))


def recall(candidates, controls):
    pool={semantic_config(r) for r in candidates}
    missing=[r.key for r in controls if semantic_config(r) not in pool]
    return dict(status='PASS' if controls and not missing else 'NO_COMPATIBLE_REFERENCE' if not controls else 'RULE_MISS',
                controls=len(controls),recalled=len(controls)-len(missing),missing=missing,
                scope='CONFIG_RECALL_NOT_PERFORMANCE_OR_GLOBAL_OPTIMUM',
                champion_injection=False)
