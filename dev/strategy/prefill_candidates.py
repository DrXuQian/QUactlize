"""Add observed prefill geometry regimes, not a config Cartesian product.

The old closest-M seeds remain intact. Coverage is selected from OTHER weight
families, without reading their latencies or the target's control recipes.
Only SF/AP0 templates may cross dense/indexed; target resources and numerics
must still be measured. This does not transfer a donor's performance claim.
"""
import math

from dev.strategy.space import producer_key, prototype, rebind_template


def template_family(w):
    return ('kquant' if w.qtype in range(10, 15) else w.qtype, w.fusion,
            w.input_stride == w.k, w.output_stride == w.n)


def regime(recipe):
    p = dict(recipe.parameters)
    return (recipe.route,
            0 if p['tm'] == 8 else 1 if p['tm'] <= 32 else 2,
            0 if p['tn'] == 16 else 1 if p['tn'] <= 64 else 2,
            p['tk'], p['stages'], p['dn'], bool(p['ap']),
            p.get('persistent', -1) if recipe.route == 'fq' else -1)


def delivery_family(recipe):
    """Exact compiled geometry except DN, not a coarse tile-size band."""
    p = dict(recipe.parameters)
    return (recipe.route, *(p[k] for k in ('tm', 'tn', 'tk', 'wm', 'wn', 'stages', 'ap')),
            p.get('persistent', -1))


def delivery_peers(w, seeds, rows, *, max_threads=1024):
    """Retain observed DN variants of a selected parent, without cross products.

    The entire target N/K family is withheld. Only same-format/operation
    prefill observations supply variants; no timing or champion label is used.
    A different warp geometry with the same regime is not a delivery peer.
    """
    from dev.strategy.prescreen import rejection, proposal_family
    if w.m <= 8:
        return [], {}
    families = {delivery_family(r) for r in seeds if r.kind == 'tc'}
    known = {producer_key(r) for r in seeds}
    selected, reasons = {}, {}
    for row in rows:
        ow, r = row.workload, row.recipe
        if (r.kind != 'tc' or ow.m <= 8 or (ow.n, ow.k) == (w.n, w.k) or
                proposal_family(ow) != proposal_family(w) or delivery_family(r) not in families):
            continue
        r = prototype(r)
        if producer_key(r) in known or rejection(r, w, max_threads):
            continue
        # Scheduling is expanded after resource query. Preserve the seed's
        # scheduling strata rather than importing an unrelated donor grid.
        for seed in seeds:
            if seed.kind != 'tc' or delivery_family(seed) != delivery_family(r):
                continue
            p, sp = dict(r.parameters), dict(seed.parameters)
            p.update(grid_mode=sp['grid_mode'], grid_b=sp['grid_b'])
            peer = type(r).make(r.kind, r.route, **p)
            key = producer_key(peer)
            if key in known:
                continue
            if key not in selected or peer.key < selected[key].key:
                selected[key] = peer
    out = sorted(selected.values(), key=lambda r: r.key)
    for r in out:
        reasons[r.key] = ['PREFILL_OBSERVED_DELIVERY_PEER_LEAVE_NK_OUT']
    return out, reasons


def supplement(w, seeds, rows, *, max_threads=1024, k_weight=1.0):
    from dev.strategy.prescreen import rejection, proposal_family
    if w.m <= 8:
        return [], {}
    covered = {regime(r) for r in seeds if r.kind == 'tc'}
    available = []
    for row in rows:
        ow, r = row.workload, row.recipe
        if (r.kind != 'tc' or ow.m <= 8 or
                (ow.n, ow.k) == (w.n, w.k)):
            continue
        local = proposal_family(ow) == proposal_family(w)
        portable = (r.route == 'sf' and not dict(r.parameters)['ap'] and
                    ow.operation in ('dense', 'indexed') and w.operation in ('dense', 'indexed') and
                    template_family(ow) == template_family(w))
        if not (local or portable):
            continue
        r = prototype(r)
        if ow.qtype != w.qtype:
            r = rebind_template(r, w.qtype)
        if regime(r) in covered or rejection(r, w, max_threads):
            continue
        available.append((row, r, local))
    # Cross-format templates fill a genuinely vacant row/column/K regime,
    # not every pipeline/delivery variant of an already available regime.
    # This is a sparse coverage backstop, not a format x template product.
    domestic = {regime(r)[:4] for r in seeds if r.kind == 'tc'}
    domestic.update(regime(r)[:4] for row,r,_ in available if row.workload.qtype == w.qtype)
    pool = {}
    for row, r, local in available:
        ow = row.workload
        if ow.qtype != w.qtype and regime(r)[:4] in domestic:
            continue
        distance = (abs(math.log2(ow.n/w.n)) + k_weight*abs(math.log2(ow.k/w.k)) +
                    2*abs(math.log2(ow.m/w.m)))
        # Local endpoint/precision first; M distance ranks donors but does not
        # erase a vacant tile/pipeline regime. Never rank by measured time.
        rank = (ow.qtype != w.qtype, not local, ow.compute != w.compute, ow.storage != w.storage,
                distance, ow.channels != w.channels, -row.priority)
        key = producer_key(r)
        if key not in pool or rank < pool[key][0]:
            pool[key] = (rank, r, row.source, not local, ow.qtype != w.qtype)
    boundary = {}
    selected, reasons = [], {}
    for rank, r, source, transferred, cross_format in sorted(pool.values(), key=lambda item:(item[0], item[1].key)):
        tag = regime(r)
        if tag not in boundary:
            boundary[tag] = rank
        if rank != boundary[tag]:
            continue
        selected.append(r)
        reasons[r.key] = ['PREFILL_VACANT_GEOMETRY_REGIME:' + source,
                         'TRANSFERRED_SF_TEMPLATE_ONLY' if transferred else
                         'SAME_OPERATION_PREFILL_TEMPLATE']
        if cross_format:
            reasons[r.key].append('REINSTANTIATE_TARGET_KQUANT_NO_TIMING_TRANSFER')
    return selected, reasons
