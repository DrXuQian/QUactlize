"""Finite missing-cost plan. Small M never enters full dequantization."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from tools.kpack_bf16_fixture import row_domain
from tools.run_kpack_dequant_gate import DENSE, GROUPED

ROOT = Path(__file__).resolve().parents[1]
MS = (128, 512, 1024, 2048, 4096)
PHASES = ('dense-mid', 'dense-rest', 'formats', 'grouped-mid', 'sparse')


@lru_cache(maxsize=1)
def historical_data():
    return json.loads((ROOT/'policies/kpack_zw810_runtime_v1.json').read_text())


def weight_id(w):
    return f'q{w["q"]}-n{w["n"]}-k{w["k"]}-e{w["experts"]}'


@lru_cache(maxsize=128)
def domain(tokens, experts, profile='real'):
    if profile == 'real':
        return row_domain(tokens, experts)
    if experts != 256 or profile not in ('active32', 'active64') or tokens < 128:
        raise ValueError('unsupported sparse prefill profile')
    count = int(profile[6:])
    # Fixed noncontiguous expert set; token changes do not change dequant bytes.
    active = np.random.default_rng(98712 + count).permutation(256)[:count]
    rng = np.random.default_rng(8721 + tokens)
    ids = np.stack([rng.choice(active, 8, replace=False) for _ in range(tokens)]).astype('<i4')
    rows = np.bincount(ids.reshape(-1), minlength=256).astype('<i4')
    if np.count_nonzero(rows) != count or int(rows.max()) > tokens:
        raise ValueError('router does not cover its intended expert set')
    indices = np.repeat(np.arange(256, dtype='<i4'), rows)
    return rows, indices, hashlib.sha256(ids.tobytes()).hexdigest()


def plan():
    base = historical_data()
    families = {q:sorted({(e['key'][2], e['key'][3]) for e in base['entries']
                         if e['key'][0] == q and e['key'][1] == 0}) for q in range(10,15)}
    points = []
    for q in range(10,15):
        for n,k in families[q]:
            old = q in (12,13) and (n,k) in DENSE
            phase = 'formats' if q not in (12,13) else 'dense-mid' if old else 'dense-rest'
            for t in MS[:3] if old else MS:
                points.append(dict(q=q,n=n,k=k,experts=1,tokens=t,profile='real',phase=phase))
        for n,k in GROUPED:
            for t in MS[:3] if q in (12,13) else MS:
                points.append(dict(q=q,n=n,k=k,experts=256,tokens=t,profile='real',
                                   phase='grouped-mid' if q in (12,13) else 'formats'))
            if q in (12,13):
                for profile in ('active32','active64'):
                    for t in MS[:3]:
                        points.append(dict(q=q,n=n,k=k,experts=256,tokens=t,profile=profile,phase='sparse'))
    for p in points:
        p['weight_id'] = weight_id(p)
        p['id'] = f'{p["weight_id"]}-t{p["tokens"]}-{p["profile"]}'
        rows, _, route_hash = domain(p['tokens'], p['experts'], p['profile'])
        p['rows'] = rows.tolist()
        p['routes_sha256'] = route_hash
        p['active_ids'] = np.flatnonzero(rows).tolist()
        p['full_indexed'] = len(p['active_ids']) != p['experts']
    if len({p['id'] for p in points}) != len(points):
        raise ValueError('duplicate supplement point')
    return dict(schema='quactlize.cost-supplement.v1',points=points,
        small_m_full_dequant=False, ms=list(MS), phases=list(PHASES),
        scope='ISOLATED_COMPONENTS_AND_BOUNDED_HISTORICAL_CHALLENGERS_NOT_MODEL_E2E',
        already_measured='Q4/Q5 five dense + six grouped families at tokens2048/4096',
        production_changed=False)


def request(p, route):
    return (p['q'], route, sum(p['rows']), p['n'], p['k'], p['experts'], p['tokens'])


def historical(p, route):
    """At most one applicable historical winner, using real load to rank it."""
    base = historical_data()
    choices = []
    for e in base['entries']:
        key = e['key']
        if key[:5] != [p['q'],route,p['n'],p['k'],p['experts']]:
            continue
        c = base['configurations'][e['config_id']]
        if c['split'] != 1 or c['ap'] or c['tm'] == 8 or p['k']//c['tk'] < c['stages']-1:
            continue
        m, maximum = sum(p['rows']), max(p['rows'])
        dist = abs(m-key[5])/max(m,key[5])
        if route >= 2:
            dist += abs(maximum-key[6])/max(maximum,key[6])
        choices.append((dist,e['config_id'],c))
    return min(choices,key=lambda x:x[:2])[2] if choices else None


def dequant_key(p, operation):
    active = p['active_ids'] if operation else list(range(p['experts']))
    suffix = hashlib.sha256(np.asarray(active,dtype='<i4').tobytes()).hexdigest()[:16]
    return p['weight_id'] + ('-full-' if operation else '-sf-') + suffix


def configs(q, operation, indexed=False):
    if not operation:
        return [4,5] if q in (12,13) else [0,1,2,3]
    if indexed:
        return [5,10,11] if q in (12,13) else [4,5]
    return [4,5,10,11,12] if q in (12,13) else list(range(6))


def validate_point(p):
    if p['tokens'] not in MS or p['q'] not in range(10,15) or p['phase'] not in PHASES:
        raise ValueError('point outside fixed supplement')
    rows,_,rh = domain(p['tokens'],p['experts'],p['profile'])
    if (p['rows'] != rows.tolist() or p['routes_sha256'] != rh or
            p['active_ids'] != np.flatnonzero(rows).tolist() or
            p['full_indexed'] != (len(p['active_ids']) != p['experts'])):
        raise ValueError('point router/active-domain differs')


if __name__ == '__main__':
    print(json.dumps(plan(),indent=2))
