"""Deterministic numerical-test router fixture."""
import numpy as np
from ci.support.router_oracle import _routing_fixture, _splitmix64, ROUTING_SEED, _identity_sha256
def real_ids(tokens):
    """Retain the old weighted-without-replacement router's exact ID bytes."""
    state = ROUTING_SEED
    ids = []
    for _ in range(tokens):
        selected = set(); remain = 16*4+240; picks = []
        for _ in range(8):
            state, value = _splitmix64(state); lottery = value % remain
            for e in range(256):
                if e in selected:
                    continue
                weight = 4 if e < 16 else 1
                if lottery < weight:
                    picks.append(e); selected.add(e); remain -= weight; break
                lottery -= weight
        ids.append(picks)
    old = _routing_fixture(256, 8, tokens)
    if _identity_sha256(ids) != old['token_routes_sha256']:
        raise ValueError('historical router bytes differ')
    return np.asarray(ids, dtype='<i4')
