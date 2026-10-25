"""Next dense M cohort. Existing observations are coverage, not new timings.

No model reads or GPU work. The small-M gap experiment reuses the component
runner; the large-M experiment reuses Asys and its verified native provider.
The transition interval stays explicit until its native MMQ precision is
handled by that carrier; it must not be silently labelled Full-F16.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
PHASE = 'dense-m-gaps'
SCHEMA = 'quactlize.dense-m-coverage.v1'
POLICY = 'policies/dense_prefill_v1.json'


def families():
    data = json.loads((ROOT/POLICY).read_text())
    return sorted({(p['shape'][0], *p['shape'][2:]) for p in data['points']})


def points():
    return [dict(id=w.key, workload=asdict(w), cohort='decode', consumers=[])
            for q,n,k in families() for m in range(1,9) for w in [Workload(q,n,k,m)]]


def coverage(originals):
    from dev.strategy.refresh import request
    ps = points()
    if [r['request'] for r in originals] != [request(Workload(**p['workload'])) for p in ps]:
        raise ValueError('dense M coverage query order differs')
    exact, missing = [], []
    # Explicit policy identities: a bucket/default is not a measured key.
    for p,r in zip(ps,originals):
        (exact if r['status']==0 and r.get('policy') in (12,14,15,18) else missing).append(p)
    return dict(schema=SCHEMA, families=len(families()), expected=len(ps), exact=len(exact),
        missing=len(missing), points=missing, retained_exact=[p['id'] for p in exact],
        scope='CURRENT_DENSE_PREFILL_FAMILIES_CROSSED_WITH_M1_TO_8_NOT_ALL_MOE',
        source_sha256=sha(ROOT/POLICY), native_comparison=False,
        transition_pending=list(range(9,128)), production_changed=False)


def make_decode_plan(originals):
    from dev.strategy.refresh import (original_recipe, proposed, historical, family,
                                      SCHEMA as REFRESH_SCHEMA, Q8_BODIES)
    audit = coverage(originals)
    by_key = {p['id']:r for p,r in zip(points(),originals)}
    prior = json.loads((ROOT/'policies/kpack_component_decode_v1.json').read_text())
    prior = {tuple(r['key']):r for r in prior['rows']}
    history = historical()
    planned = []
    for p in audit['points']:
        row = by_key[p['id']]
        if row['status']:
            raise ValueError('dense control missing: '+p['id'])
        planned.append(proposed(p, original_recipe(row), history, prior, allow_new_family=True))
    if not planned:
        raise ValueError('no remaining dense M gaps; do not rerun the old full cohort')
    tc, vectors = {}, {}
    for p in planned:
        for arm in p['arms']:
            (tc if arm['recipe']['kind']=='tc' else vectors)[arm['module']]=arm['spec']
    result = dict(schema=REFRESH_SCHEMA, phase=PHASE, points=planned, tc=tc, vectors=vectors,
        coverage=audit, counts=dict(points=len(planned), full_points=audit['expected'],
            families=len({family(Workload(**p['workload'])) for p in planned}),
            arms=sum(len(p['arms']) for p in planned), max_arms=max(len(p['arms']) for p in planned),
            tc_modules=len(tc), vector_modules=len(vectors), operations=dict(Counter(
                p['workload']['operation'] for p in planned))),
        historical_times_reused=False, production_changed=False, native_comparison=False,
        scope='DENSE_M1_TO_8_MISSING_EXACT_KEYS_VS_CURRENT_CPP_COMPONENTS',
        comparison='CURRENT_SELECTED_CONTROL_AND_BOUNDED_CONFIGS_NOT_NATIVE_OR_MODEL_AB',
        q8_bodies=list(Q8_BODIES), q8_new_bodies_scope='DENSE_S1_ONLY',
        protocol=dict(rounds=6,samples=5,order='FORWARD_REVERSE_ALTERNATING',
            producer_ring='AT_LEAST_2P25_L2',helper_timing='ONE_KERNEL_AT_A_TIME'),
        omitted=['NATIVE_AB','MODEL_AB','M9_TO_127','NEW_MOE_FAMILIES','PRODUCTION_PROMOTION'],
        pending=['IMPORT_EXISTING_1700MHZ_REFRESH_WINNERS','NATIVE_REGRESSION_CONFIRMATION',
                 'BUCKET_HOLDOUT_VALIDATION'],
        sources={name:sha(ROOT/name) for name in (POLICY,'policies/kpack_component_decode_v1.json',
            'policies/kpack_decode_winners_v1.json','docs/measurements/smallm_matched_20260915.json.gz')})
    result['sha256']=digest(result)
    return json.loads(json.dumps(result))
