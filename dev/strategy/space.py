"""Independent search axes, not a product of producer and helper configs.

These rules inspect an API/workload contract, never a target's winner. A
producer is still measured separately for each Split-K/grid recipe: those
axes change its work. Dequant implementations share an output layout and are
profiled once per exact weight/domain/precision instead of once per GEMM.
"""
from functools import lru_cache

from quactlize.dispatch.table import Recipe
from quactlize.runtime.tuning import digest


def expansion_configs(w, route):
    if w.qtype == 8 or route == 'fq':
        return ()
    if route == 'full':
        if w.m < 128:
            return ()
        # Declared scalar/vector/transpose implementations. Shared-memory
        # experiments 6..9 remain optional, not an unbounded new sweep axis.
        return tuple(range(6)) + ((10, 11, 12) if w.qtype in (12, 13) else ())
    # Dense TC uses F16 metadata even with a BF16 activation endpoint;
    # grouped BF16 explicitly uses direct BF16 metadata (module.cuh).
    bf16_metadata = w.operation != 'dense' and w.compute == 'bf16'
    base = (0,) if bf16_metadata else tuple(range(4))
    return (-1, *base, *((4, 5) if w.qtype in (12, 13) else ()))


@lru_cache(maxsize=65536)
def producer_key(recipe):
    p = dict(recipe.parameters)
    p.pop('symbol', None)
    p.pop('dequant_config', None)
    return digest(dict(kind=recipe.kind, route=recipe.route, parameters=p))


@lru_cache(maxsize=8192)
def prototype(recipe):
    """Remove execution/helper axes before selecting compiled geometries."""
    p = dict(recipe.parameters)
    if recipe.kind in ('tc', 'simt'):
        p['split'] = 1
    if recipe.kind == 'tc':
        p['dequant_config'] = -1  # production expansion as untimed setup
        # Keep the donor's scheduling stratum for geometric diversity.
        # These are not compiler axes: grid_variants() later expands and
        # deduplicates them using the target device's measured resources.
    if recipe.kind == 'full':
        p['dequant_config'] = 4
    return Recipe.make(recipe.kind, recipe.route, **p)


@lru_cache(maxsize=8192)
def rebind_template(recipe, qtype):
    """Re-instantiate a generic K-quant template, not copy quantized bytes.

    The compiler still receives the TARGET qtype and validates its transport
    TK, packed-A and metadata contract. No latency transfers between formats.
    Specialized Q4 readers and Q8's scale-only family are not translated.
    """
    if recipe.kind != 'tc':
        return recipe
    p = dict(recipe.parameters)
    if p.get('qtype') == qtype:
        return recipe
    p['qtype'] = qtype
    p['symbol'] = ('rule_' + recipe.route + '_q' + str(qtype) + '_' +
                   '_'.join(k + str(p.get(k, 0)) for k in
                            ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap', 'dn', 'persistent')))
    # Negative signs are not valid symbol characters.
    p['symbol'] = p['symbol'].replace('-', 'm')
    return Recipe.make(recipe.kind, recipe.route, **p)


@lru_cache(maxsize=65536)
def split_variants(w, recipe):
    """Supported runtime slices; legality is checked against the target K.

    Dense persistent supports S1 only, but an SF parent also contains its
    ordinary entry. Do not discard ordinary S2/4/8 merely because a donor
    used that parent's persistent entry. Exact grids are expanded afterwards.
    Never infer that S>1 has no reducer cost.
    """
    p = dict(recipe.parameters)
    if recipe.kind not in ('tc', 'simt'):
        return (recipe,)
    if recipe.kind == 'tc' and recipe.route == 'sf':
        p.update(grid_mode=0, grid_b=0)
    splits = ((1,) if recipe.kind == 'tc' and w.operation == 'dense' and
              p.get('grid_mode', 0) >= 2 else (1, 2, 4, 8))
    return tuple(Recipe.make(recipe.kind, recipe.route, **(p | {'split': s})) for s in splits)


def grid_search(w, recipe):
    if recipe.kind != 'tc':
        return None
    p=dict(recipe.parameters)
    supported=(recipe.route=='sf' or w.operation!='dense' and p.get('persistent')==1)
    if not supported or w.operation=='dense' and p['split']!=1:
        return None
    return dict(modes=[0,2,3] if recipe.route=='sf' else [2,3],
        ordinary_mode_aliases=[0,1] if recipe.route=='sf' else [],
        blocks_per_cu=[1,32],bounds='INCLUSIVE_API_BOUND_NOT_OCCUPANCY',
        expansion='AFTER_COMPILED_OCCUPANCY_QUERY',
        dedup='ACTUAL_ALGORITHM_SPLIT_GRID_ON_PROBED_DEVICE',
        cap='ACTUAL_COMPUTE_UNITS_TIMES_MIN_B_OCCUPANCY')


def tile_work(w, recipe):
    """Same conservative directory bound as dispatch/policy.hpp::recipe.

    Indexed M is tokens, not routed rows. No CPU readback of expert IDs or
    assumption that each expert receives max_rows is needed for this bound.
    Ragged grouped requests require an explicit row contract and are not
    silently interpreted as indexed input by this experiment harness.
    """
    if recipe.kind != 'tc' or w.operation == 'grouped':
        raise ValueError('grid work needs a dense or indexed TC request')
    p = dict(recipe.parameters)
    tm, tn, split = (p.get(k) for k in ('tm', 'tn', 'split'))
    if any(type(v) is not int or v <= 0 for v in (tm, tn, split)):
        raise ValueError('positive tile dimensions and split required')
    mt = (w.m + tm - 1) // tm
    if w.operation == 'indexed':
        rows = w.m * w.topk
        active = min(rows, w.experts)
        mt = min(w.experts * mt, (rows + active * (tm - 1)) // tm)
    work = mt * ((w.n + tn - 1) // tn) * (split if w.operation == 'indexed' else 1)
    if not 0 < work <= 2147483647:
        raise ValueError('grid work exceeds the module int32 contract')
    return work


def resolved_grid(w, recipe, compute_units, occupancy):
    """Resolve a config, not a performance policy; no default board size."""
    if any(type(v) is not int or v <= 0 for v in (compute_units, occupancy)):
        raise ValueError('positive probed compute_units and compiled occupancy required')
    p = dict(recipe.parameters)
    work = tile_work(w, recipe)
    mode = p.get('grid_mode', 0)
    if mode not in (0, 1, 2, 3) or not 0 <= p.get('grid_b', 0) <= 32:
        raise ValueError('invalid scheduler config')
    if mode < 2:
        if w.operation == 'indexed' and recipe.route == 'fq' and p.get('persistent') != 0:
            raise ValueError('ordinary schedule differs from compiled FQ grouped parent')
        return dict(algorithm=0, split=p['split'], grid=0, work=work)
    if w.operation == 'indexed' and recipe.route == 'fq' and p.get('persistent') != 1:
        raise ValueError('persistent schedule differs from compiled FQ grouped parent')
    if w.operation == 'dense' and (recipe.route != 'sf' or p['split'] != 1):
        raise ValueError('dense persistent requires SF S1')
    capacity = compute_units * max(1, min(p['grid_b'], occupancy))
    waves = (work + capacity - 1) // capacity
    grid = (work + waves - 1) // waves if mode == 3 else min(work, capacity)
    return dict(algorithm=1, split=p['split'], grid=grid, work=work)


def grid_variants(w, recipe, *, compute_units, occupancy):
    """Materialize only distinct launch grids after querying THIS parent.

    Occupancy must come from this exact compute/endpoint/persistent kernel,
    not a neighboring parent or a timing table. Aliases retain coverage
    evidence but cost no extra timed arm. Ordinary and persistent are never
    merged, even when their block counts happen to agree.
    """
    if recipe.kind != 'tc':
        raise ValueError('grid expansion requires a TC recipe')
    domain = grid_search(w, recipe)
    recipes = [recipe]
    if domain:
        p = dict(recipe.parameters)
        for mode in domain['modes']:
            blocks = ((0,) if mode == 0 else
                      range(domain['blocks_per_cu'][0], domain['blocks_per_cu'][1] + 1))
            for b in blocks:
                recipes.append(Recipe.make('tc', recipe.route, **(p | {'grid_mode': mode, 'grid_b': b})))
    unique = {}
    for r in recipes:
        launch = resolved_grid(w, r, compute_units, occupancy)
        key = (launch['algorithm'], launch['split'], launch['grid'])
        alias = {k: dict(r.parameters).get(k, 0) for k in ('grid_mode', 'grid_b')}
        if key not in unique:
            unique[key] = dict(recipe=r, launch=launch, aliases=[], compute_units=compute_units,
                               occupancy=occupancy, source='PROBED_CUS_COMPILED_OCCUPANCY')
        if alias not in unique[key]['aliases']:
            unique[key]['aliases'].append(alias)
    return list(unique.values())


@lru_cache(maxsize=65536)
def scheduled_key(recipe):
    p=dict(recipe.parameters)
    p.pop('symbol',None);p.pop('dequant_config',None)
    # Membership also verifies the explicit grid domain below. These are
    # not dropped from the execution identity or from component timing.
    if recipe.kind=='tc':
        p.pop('grid_mode',None);p.pop('grid_b',None)
    return digest(dict(kind=recipe.kind,route=recipe.route,parameters=p))


def scheduler_member(w, candidate, control):
    a,b=dict(candidate.parameters),dict(control.parameters)
    if candidate.kind!='tc':return True
    if (a.get('grid_mode',0),a.get('grid_b',0))==(b.get('grid_mode',0),b.get('grid_b',0)):
        return True
    domain=grid_search(w,candidate)
    modes=(domain['modes']+domain['ordinary_mode_aliases']) if domain else []
    return bool(domain and b.get('grid_mode') in modes and 0<=b.get('grid_b',-1)<=32)


def helpers(w, candidates):
    result = []
    for route in sorted({r.route for r in candidates}):
        configs = expansion_configs(w, route)
        if not configs:
            continue
        result.append(dict(operation='full' if route == 'full' else 'scale-zero', route=route,
            configs=list(configs), qtype=w.qtype, n=w.n, k=w.k, experts=w.experts,
            metadata_type='bf16' if route == 'full' or w.operation != 'dense' and w.compute == 'bf16' else 'f16',
            domain='ALL_DENSE_WEIGHTS' if w.operation == 'dense' else
                   dict(tokens=w.m, topk=w.topk, channels=w.channels, router=w.router),
            timing='ISOLATED_EXPANSION_ONCE_PER_EXACT_DOMAIN',
            source='DEQUANT_API_CONFIG_NAMESPACE_NOT_TARGET_WINNER'))
    return result


def factored_recall(w, candidates, controls):
    """Membership in the generated producer + separate helper search space.

    No Split-K, scheduler, reader or precision fields are ignored. Helper
    membership is explicit; it is NOT permission to claim the helper ran.
    """
    available = {}
    for r in candidates:available.setdefault(scheduled_key(r),[]).append(r)
    missing, reasons = [], {}
    for r in controls:
        p = dict(r.parameters)
        matching=available.get(scheduled_key(r),())
        if not matching:
            reason = 'PRODUCER_RECIPE_NOT_GENERATED'
        elif not any(scheduler_member(w,c,r) for c in matching):
            reason = 'SCHEDULER_NOT_GENERATED'
        elif r.kind in ('tc', 'full') and r.route in ('sf', 'full') and w.qtype != 8:
            reason = ('EXPANSION_NOT_GENERATED' if p.get('dequant_config', -1) not in
                      expansion_configs(w, r.route) else None)
        else:
            reason = None
        if reason:
            missing.append(r.key)
            reasons[r.key] = reason
    return dict(status='PASS' if controls and not missing else 'RULE_MISS' if controls else 'NO_COMPATIBLE_REFERENCE',
        controls=len(controls), recalled=len(controls)-len(missing), missing=missing, reasons=reasons,
        scope='GENERATED_COMPONENT_SPACE_NOT_PERFORMANCE_OR_EXECUTION_PROOF',
        champion_injection=False, implicit_axes=[],
        separately_generated_axes=['dequant_config','grid_mode','grid_b'],
        resource_validation='PENDING_ACTUAL_MODULE_QUERY')


def factor_candidates(candidates):
    """Stable dedup by the exact producer recipe, not just a compile parent."""
    unique = {}
    for r in candidates:
        old = unique.get(producer_key(r))
        if old is None or r.key < old.key:
            unique[producer_key(r)] = r
    return list(unique.values())
