"""Bounded Q8 SF geometry proposals, separate from retained measured controls.

The previous TP cohort tried many Full helpers but only one SF mainloop.
This is a new measurement plan, not a table update or a global optimum claim.
"""
from dataclasses import asdict
import copy
import json
import math
from pathlib import Path

from .finalists import case_identity, counts, full_case
from .supplement import MS
from dev.strategy.prescreen import rejection
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.tuning import digest

SCHEMA = 'quactlize.q8-tp-prefill-refine.v1'
MAX_CASES = 16
# A basis, not the Cartesian product. TK128 uses two stages to bound shared
# footprint. Actual legality/occupancy is still queried from compiled modules.
GEOMETRIES = (
    (32, 64, 64, 3),
    (64, 64, 64, 3), (64, 64, 128, 2),
    (64, 128, 64, 3), (64, 128, 128, 2),
    (128, 64, 64, 3), (128, 64, 128, 2),
    (128, 128, 64, 3), (128, 128, 128, 2),
)


def sf_case(geometry, *, split=1, grid_mode=0, wm=32):
    tm, tn, tk, stages = geometry
    symbol = f'sf_q8_a0_tm{tm}_tn{tn}_tk{tk}_wm{wm}_wn32_s{stages}_bc0_ap0_dn64'
    recipe = Recipe.make('tc', 'sf', qtype=8, symbol=symbol, tm=tm, tn=tn, tk=tk,
        wm=wm, wn=32, stages=stages, ap=0, dn=64, persistent=-1,
        split=split, grid_mode=grid_mode, grid_b=2 if grid_mode else 0,
        dequant_config=0)
    return dict(arm='sf', helper_mode='cohort-current', helper_options=dict(vector_io=3),
                recipe=json.loads(json.dumps(asdict(recipe))))


def proposals(w, compute_units):
    """No historical winner/timing argument: rules cannot inject a champion.

    Two-CU-waves is a screening threshold, NOT a predicted occupancy. Runtime
    persistent grids use the probed CU count and compiled occupancy as before.
    """
    if (w.qtype != 8 or w.operation != 'dense' or w.compute != 'f16' or w.storage != 'f32'
            or w.m not in MS or type(compute_units) is not int or compute_units <= 0):
        raise ValueError('Q8 refine requires dense F32/F16 prefill and a measured CU count')
    result = [(sf_case(g), 'rule:tile-warp-pipeline-basis') for g in GEOMETRIES]
    small = (64, 64, 64, 3)
    # A one-axis warp-fragment comparison. This rule happens to cover the old
    # Q8 SF geometry too; it does not obtain that geometry from its winner row.
    result.append((sf_case(small, wm=64), 'rule:large-warp-fragment-control'))
    tiles = math.ceil(w.m/64)*math.ceil(w.n/64)
    if tiles < 2*compute_units:
        for split in (2, 4):
            if w.k % (split*64) == 0 and w.k//(split*64) >= 4:
                result.append((sf_case(small, split=split), 'rule:underfilled-split-k'))
    large = (128, 128, 64, 3)
    if math.ceil(w.m/128)*math.ceil(w.n/128) >= 2*compute_units:
        result.append((sf_case(large, grid_mode=2), 'rule:occupancy-resolved-persistent'))
    for case, _ in result:
        if rejection(Recipe.from_dict(case['recipe']), w):
            raise ValueError('generated Q8 SF recipe is not legal for '+w.key)
    return result


def make_plan(review):
    from tools.review_prefill_cohort import eligible
    if (review.get('schema') != 'quactlize.prefill-component-review.v1' or
            review.get('review_sha256') != digest({k:v for k,v in review.items() if k!='review_sha256'})
            or review.get('pending_points') or review.get('incomplete')):
        raise ValueError('complete, sealed previous whole-call review required')
    points = []
    for key, old in sorted(review['points'].items()):
        w = Workload(**old['workload'])
        if key != w.key:
            raise ValueError('previous workload identity differs')
        if w.qtype != 8:
            continue
        authority = review['authorities'][old['authority']]
        # The review binds the same device identity as the actual capture.
        compute_units = authority['device']['device']['compute_units']
        pool = {}
        def add(case, role):
            cid = digest(case)
            row = pool.setdefault(cid, dict(copy.deepcopy(case), id=cid, roles=[]))
            if role not in row['roles']:
                row['roles'].append(role)
            return cid
        valid = [r for r in old['measurements'] if eligible(r)]
        native = [r for r in old['measurements'] if r['arm']=='native-fp16' and r['status']=='PASS']
        if len(native)!=1 or not valid:
            raise ValueError('previous native/candidate measurement missing')
        for r in [*valid, *native]:
            if (not math.isfinite(r['gpu_sum_us']) or r['gpu_sum_us']<=0 or
                    len(r.get('confirmation_captures', []))!=2):
                raise ValueError('previous whole-call confirmation missing')
        for r in valid:
            if r['candidate']!=digest(review['candidates'][r['candidate']]):
                raise ValueError('previous candidate identity differs')
        prior = min(valid, key=lambda r:(r['gpu_sum_us'],r['candidate']))
        add(review['candidates'][prior['candidate']], 'prior-point-winner')
        controls = {}
        for route in ('sf', 'full-fp16'):
            rows = [r for r in valid if r['arm']==route]
            if not rows:
                raise ValueError('previous per-route control missing: '+route)
            winner = min(rows, key=lambda r:(r['gpu_sum_us'],r['candidate']))
            controls[route] = add(review['candidates'][winner['candidate']], 'prior-route-winner:'+route)
        # Isolate helper effects on the OLD SF mainloop before comparing new
        # tile schedules. No reuse of another shape's microseconds.
        old_sf = copy.deepcopy(review['candidates'][controls['sf']])
        old_sf['helper_options'] = dict(vector_io=3)
        add(old_sf, 'old-sf-vector-io-control')
        full = review['candidates'][controls['full-fp16']]
        for dq in (38, 41):
            add(full_case(dq, full['helper_options']['vector_io']), 'q8-full-helper-control')
        for case, role in proposals(w, compute_units):
            add(case, role)
        if len(pool)>MAX_CASES:
            raise ValueError(f'{key}: {len(pool)} cases exceed {MAX_CASES}; no control truncated')
        cases = sorted(pool.values(), key=lambda c:digest([key,c['id']]))
        points.append(dict(id=key, workload=asdict(w), consumers=copy.deepcopy(old['consumers']), tensors=[],
            cases=cases, candidates=[c['recipe'] for c in cases if c['recipe']['kind']=='tc'],
            champion_controls=[], fixed_schedules=True, confirmation_captures=2,
            prior_winner=prior['candidate'], prior_route_winners=controls,
            proposal_compute_units=compute_units, evidence_sha256=digest(old)))
    result = dict(schema=SCHEMA, points=points, reuse=[], confirmation_captures=2, max_cases=MAX_CASES,
        counts=counts(points), inputs=dict(review_sha256=review['review_sha256'],
            helper_policy_sha256=digest(json.loads((Path(__file__).resolve().parents[2]/
                'policies/dense_full_helpers_20261023.json').read_text()))),
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        scope='Q8_TP_LOCAL_DENSE_PREFILL_REFINEMENT_NOT_MOE_COLLECTIVES_OR_MODEL_TPOT',
        search='BOUNDED_GEOMETRY_BASIS_PLUS_SEPARATE_MEASURED_CONTROLS_NOT_GLOBAL_OPTIMUM',
        historical_timing_used='CONTROLS_ONLY; ALL_TARGET_ARMS_REMEASURED',
        production_changed=False, performance_admitted=False)
    result['sha256'] = digest(result)
    return validate(result)


def validate(plan):
    if (plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'})
            or plan.get('confirmation_captures')!=2 or plan.get('max_cases')!=MAX_CASES
            or plan.get('production_changed') is not False or plan.get('performance_admitted') is not False):
        raise ValueError('Q8 refinement plan identity/contract differs')
    seen = set()
    for p in plan['points']:
        w = Workload(**p['workload'])
        expected_rules = {digest(c):role for c,role in proposals(w,p['proposal_compute_units'])}
        if (p['id']!=w.key or w.key in seen or not p['fixed_schedules'] or p['confirmation_captures']!=2
                or not 1<=len(p['cases'])<=MAX_CASES):
            raise ValueError('Q8 refinement workload/budget differs')
        seen.add(w.key)
        ids, roles = set(), set()
        for c in p['cases']:
            if c['id']!=digest(case_identity(c)) or c['id'] in ids or not c['roles']:
                raise ValueError('duplicate/changed refinement candidate')
            ids.add(c['id']); roles.update(c['roles'])
            r = Recipe.from_dict(c['recipe'])
            if r.kind=='tc':
                if (r.route!='sf' or c['arm']!='sf' or rejection(r,w) or
                        c['helper_mode']!='cohort-current' or
                        c['helper_options'] not in [dict(vector_io=i) for i in (-1,1,2,3)]):
                    raise ValueError('invalid Q8 SF geometry/helper contract')
            elif r.kind=='full':
                opts = c['helper_options']
                if (opts['dequant_variant'] not in (0,22,23,36,37,38,41) or opts['vector_io'] not in range(4)
                        or case_identity(c)!=full_case(opts['dequant_variant'],opts['vector_io'])):
                    raise ValueError('invalid Full-F16 control')
            else:
                raise ValueError('not a dense Q8 prefill recipe')
        required = {'prior-point-winner','prior-route-winner:sf','prior-route-winner:full-fp16',
                    'old-sf-vector-io-control','q8-full-helper-control'}
        if (not required<=roles or not {p['prior_winner'],*p['prior_route_winners'].values()}<=ids or
                any(not any(c['id']==cid and role in c['roles'] for c in p['cases'])
                    for cid,role in expected_rules.items())):
            raise ValueError('required rule proposal or historical control missing')
        if (p['candidates']!=[c['recipe'] for c in p['cases'] if c['recipe']['kind']=='tc'] or
                p['champion_controls']):
            raise ValueError('refinement resource inventory differs')
    if not seen or plan['reuse'] or plan['counts']!=counts(plan['points']):
        raise ValueError('Q8 refinement denominator differs')
    return plan


def dequant_diagnostics(review, plan):
    """Five worst helper gaps plus a faster/small control, one cell per N/K.

    These are priorities, not imported measurements. M is retained in the
    evidence only: the isolated dequant kernel has no M dimension.
    """
    validate(plan)
    if plan['inputs']['review_sha256']!=review['review_sha256']:
        raise ValueError('dequant priorities must use the same reviewed cohort')
    families = {}
    for p in plan['points']:
        old = review['points'][p['id']]
        chosen = next(r for r in old['measurements']
                      if r['candidate']==p['prior_route_winners']['full-fp16'])
        native = next(r for r in old['measurements'] if r['arm']=='native-fp16')
        ours, reference = chosen['phases_us']['expansion'], native['phases_us']['expansion']
        if not all(math.isfinite(x) and x>0 for x in (ours,reference)):
            raise ValueError('missing helper samples for diagnostic priorities')
        w = Workload(**p['workload'])
        families.setdefault((w.n,w.k),[]).append(dict(point=w.key,m=w.m,
            candidate=chosen['candidate'],dequant=review['candidates'][chosen['candidate']]['helper_options']['dequant_variant'],
            historical_gap_us=ours-reference))
    ranked = sorted(families, key=lambda nk:(-max(r['historical_gap_us'] for r in families[nk]),nk))
    slow = [nk for nk in ranked if max(r['historical_gap_us'] for r in families[nk])>0][:5]
    controls = [nk for nk in families if nk not in slow]
    # Prefer an actually non-regressing family; otherwise the least observed
    # gap. This does not relabel a slow point as a performance pass.
    control = min(controls, key=lambda nk:(max(r['historical_gap_us'] for r in families[nk]),nk[0]*nk[1],nk)) if controls else None
    selected = [*slow, *([control] if control else [])]
    rows = [dict(index=i,q=8,n=n,k=k,role='slow-helper-diagnostic' if (n,k) in slow else 'comparison-control',
                 evidence=families[n,k]) for i,(n,k) in enumerate(selected)]
    if not rows:
        raise ValueError('empty isolated helper cohort')
    # Inspect both the largest observed gap and the largest slow matrix;
    # profiling all six shapes/configs would mostly duplicate evidence.
    largest = max((r for r in rows if r['role']=='slow-helper-diagnostic'),
                  key=lambda r:r['n']*r['k'],default=rows[0])
    return dict(points=rows, shapes=','.join(f'{r["n"]}x{r["k"]}' for r in rows),
        profile_points=list(dict.fromkeys((0,largest['index']))),
        scope='ISOLATED_DEQUANT_NOT_CALL_CONTEXT_OR_PRODUCTION_ROUTING',
        historical_timing_used='PRIORITY_ONLY',production_changed=False)
