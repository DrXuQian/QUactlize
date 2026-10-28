"""Sealed production catalog validation/generation; no device experiments."""
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
