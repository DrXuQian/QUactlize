"""Geometry-only additions; target N/K and reference winner labels are withheld."""
from dev.strategy.prefill_candidates import delivery_family
from dev.strategy.prescreen import proposal_family, rejection
from dev.strategy.space import prototype, split_variants, factor_candidates
from quactlize.dispatch.table import Recipe


def decode_delivery_peers(w, seeds, observations):
    """Do not drop a delivery-width variant of an otherwise selected TC tile."""
    if w.m>8:return []
    families={delivery_family(r) for r in seeds if r.kind=='tc'}
    peers=[]
    for row in observations:
        ow,r=row.workload,row.recipe
        if (r.kind!='tc' or ow.m>8 or (ow.n,ow.k)==(w.n,w.k) or
                proposal_family(ow)!=proposal_family(w) or delivery_family(r) not in families):continue
        r=prototype(r)
        for seed in seeds:
            if seed.kind!='tc' or delivery_family(seed)!=delivery_family(r):continue
            p,sp=dict(r.parameters),dict(seed.parameters)
            p.update(grid_mode=sp['grid_mode'],grid_b=sp['grid_b'])
            for candidate in split_variants(w,Recipe.make(r.kind,r.route,**p)):
                if not rejection(candidate,w):peers.append(candidate)
    return factor_candidates(peers)
