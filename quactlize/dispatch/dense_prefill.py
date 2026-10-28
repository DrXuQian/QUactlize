"""Production catalog validation."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest
ROOT = Path(__file__).resolve().parents[2]


EVIDENCE = ROOT / 'policies/dense_prefill_v1.json'


HEADER = ROOT / 'quactlize/dispatch/dense_prefill.inc'


SCHEMA = 'quactlize.dense-prefill-policy.v1'


REGRET_LIMIT = .05  # in-sample fitting bound, NOT a new-shape/native guarantee


def require(ok, message):
    if not ok:
        raise ValueError(message)


def band(x):
    return x.bit_length()-1


def bucket_key(shape):
    q, m, n, k = shape
    # N and K are not interchangeable: scale expansion and weight delivery
    # differ even for equal N*K. Keep both balanced orientations separate.
    # M9..15 use the first measured M16 interval, not the decode M1..8 table.
    return q, band(max(16,m)), band(n), band(k), (1 if n >= 2*k else 2 if k >= 2*n else 0 if n >= k else 3)


def choose(costs):
    return min(costs, key=lambda c: (costs[c], c))


def admitted_costs(point):
    """Helper updates require improvement in both capture orders, not a lucky median."""
    costs = dict(point['costs'])
    a = point.get('admission')
    if a is None:
        return costs
    require(a['rule'] == 'BOTH_CAPTURE_MEDIANS_BEAT_REMEASURED_CONTROL', 'unknown admission rule')
    captures, controls = a['capture_medians'], a['controls']
    require(set(captures) == set(costs) and controls and set(controls) <= costs.keys(), 'admission coverage differs')
    for rid, ts in captures.items():
        require(len(ts) == 2 and all(math.isfinite(t) and t > 0 for t in ts) and
                abs(statistics.median(ts)-costs[rid]) < 1e-6, 'admission capture median differs')
    baseline = a['baseline']
    require(baseline == choose({r:costs[r] for r in controls}), 'control is not the remeasured old minimum')
    incumbent = a['incumbent']
    require(incumbent is None or incumbent in costs, 'incumbent was not remeasured')
    comparisons = {baseline, incumbent} - {None}
    keep = incumbent or baseline
    return {rid:us for rid,us in costs.items() if rid == keep or
            all(all(x < y for x,y in zip(captures[rid], captures[old])) for old in comparisons-{rid})}


def bucket_costs(point):
    # Exact-only helper admission must not silently widen a measured launch.
    # For a refreshed existing row, preserve its original bucket observations;
    # [] marks a new exact-only row, with no bucket-training claim.
    return dict(point.get('bucket_costs', point['costs']))


def bucket_interval(shape):
    mb = bucket_key(shape)[1]
    return [9 if mb == 4 else 1 << mb, min(8192, (1 << (mb+1))-1)]


def bucket_training_costs(point, recipes):
    """Only recipes legal throughout the predicted M interval.

    The exact row retains every measured minimum. TM8 and lightweight FQ
    stop at M64; do not extrapolate them to M65..127 or large prefill.
    """
    lo, hi = bucket_interval(point['shape'])
    def legal(recipe):
        if recipe['route'] == 2:
            return lo >= 128
        tc = recipe['tc']
        return (not tc['ap'] and not tc.get('legacy_adapters') and
                (tc['tm'] != 8 or (recipe['route'] == 0 and hi <= 64)) and
                (recipe['route'] != 0 or hi <= 64))
    return {rid:us for rid,us in point['costs'] if legal(recipes[rid])}


def fit(points, *, split_m=False):
    """Log bins, subdivided only where a common recipe exceeds 5% regret.

    Subdivision uses the largest normalized N/K (optionally M) gap, not a model/tensor name.
    Single-member leaves are explicitly predictions, never holdout evidence.
    All leaves stay inside one octave in M/N/K; no remote nearest-neighbor.
    """
    groups = defaultdict(list)
    for p in points:
        if bucket_costs(p):
            groups[bucket_key(p['shape'])].append(p)
    leaves = []

    def visit(key, members, bounds):
        costs = [bucket_costs(p) for p in members]
        common = set.intersection(*(set(c) for c in costs))
        normalized = [{k: v/min(c.values()) for k, v in c.items()} for c in costs]
        best = min(common, key=lambda c: (max(r[c] for r in normalized),
                   statistics.mean(r[c] for r in normalized), c)) if common else None
        regret = max(r[best] for r in normalized)-1 if best else math.inf
        if best is not None and regret <= REGRET_LIMIT:
            leaves.append(dict(key=list(key), bounds=bounds[:2], recipe=best,
                members=[p['key'] for p in members], worst_regret_pct=100*regret,
                predicted=True, holdout_validated=False,
                **(dict(m_bounds=bounds[2]) if split_m else {})))
            return
        # Different N/K in a coarse bin require a finer rule, not a slower
        # shared FQ fallback. Distinct exact rows always terminate this split.
        gaps = []
        for dim in ((1, 2, 3) if split_m else (2, 3)):
            values = sorted({p['shape'][dim] for p in members})
            gaps += [(math.log2(b/a), dim, (a+b)//2) for a, b in zip(values, values[1:])]
        require(gaps, 'cannot subdivide incompatible observations at one shape')
        _, dim, cut = max(gaps)
        axis = 2 if dim == 1 else dim-2
        for lower in (True, False):
            subset = [p for p in members if (p['shape'][dim] <= cut) == lower]
            limits = [list(b) for b in bounds]
            limits[axis][1 if lower else 0] = cut if lower else cut+1
            visit(key, subset, limits)

    for key, members in sorted(groups.items()):
        bounds = [[1 << key[2], (1 << (key[2]+1))-1], [1 << key[3], (1 << (key[3]+1))-1]]
        if split_m:
            bounds.append(bucket_interval(members[0]['shape']))
        visit(key, sorted(members, key=lambda p:p['shape']), bounds)
    return leaves


def validate(data):
    require(data['schema'] == SCHEMA and data['sha256'] == digest({k:v for k,v in data.items() if k != 'sha256'}),
            'dense prefill catalog seal differs')
    require(data['arithmetic'] == dict(input='F32', output='F32', compute='F16', accumulator='FP32') and
            not data['model_performance_admitted'], 'arithmetic/admission changed')
    for key, recipe in data['recipes'].items():
        require(digest(recipe) == key, 'execution identity differs')
        require('dequant_first' not in recipe or (recipe['dequant_first'] is True and recipe['route'] == 2),
                'stage order is a Full-only execution axis')
    # Superseded observations remain evidence, not runtime choices or a new
    # same-run native baseline. Their old helper admission is still checked.
    seen = set()
    for p in data['points'] + data.get('superseded_points', []):
        q,m,n,k = p['shape']
        require(p['key'] == Workload(qtype=q,m=m,n=n,k=k,operation='dense',compute='f16',storage='f32').key and
                (p['key'],p['authority']) not in seen, 'point identity differs')
        seen.add((p['key'],p['authority']))
        require(p['authority'] in data['authorities'], 'missing authority')
        if p.get('admission',{}).get('review_sha256'):
            require(p['admission']['review_sha256'] in data['reviews'], 'missing admission review')
        costs = dict(p['costs'])
        require(len(costs) == len(p['costs']) and all(r in data['recipes'] and math.isfinite(t) and t > 0
                                                   for r,t in costs.items()), 'invalid costs')
        require(choose(admitted_costs(p)) == p['winner'] and costs[p['winner']] == p['us'], 'not admitted measured minimum')
        bc = bucket_costs(p)
        require(all(r in data['recipes'] and math.isfinite(t) and t > 0 for r,t in bc.items()),
                'invalid bucket training costs')
        if 'bucket_costs' in p:
            require(not bc or (p.get('bucket_evidence') and p['bucket_evidence']['authority'] in data['authorities']),
                    'missing historical bucket evidence')
        if 'mean_evidence' in p:
            from quactlize.dispatch.dense_prefill_mean import validate_mean_point
            validate_mean_point(p, data)
    require(len({p['key'] for p in data['points']}) == len(data['points']), 'duplicate active point')
    refreshed = data.get('bucket_policy') == 'ACTIVE_MEANS_MNK_INTERVALS_V2'
    if refreshed:
        observations = {(p['key'],p['authority'],p['evidence_sha256']):p
                        for p in data['points']+data.get('superseded_points', [])}
        for p in data['points']:
            if not bucket_costs(p):
                continue  # a later exact-only addition cannot invent bucket evidence
            e = p['bucket_evidence']
            source = observations.get((p['key'],e['authority'],e['evidence_sha256']))
            require(source is not None and 'mean_evidence' in source and
                    bucket_costs(p) == bucket_training_costs(source, data['recipes']),
                    'bucket training observation differs')
    require(fit(data['points'], split_m=refreshed) == data['buckets'], 'bucket fitting differs')


def render(data):
    validate(data)
    used = sorted({p['winner'] for p in data['points']} | {b['recipe'] for b in data['buckets']})
    index = {k:i for i,k in enumerate(used)}
    out = ['// Generated by tools/fit_dense_prefill.py; do not edit.',
           '// Whole GPU call, F16 operands / FP32 accumulation; not model TPOT.',
           f'inline constexpr char kCatalogSha[] = "{data["sha256"]}";',
           'inline constexpr Choice kChoices[] = {']
    for key in used:
        r = data['recipes'][key]
        tc = '{}'
        if r['tc']:
            p = r['tc']; q = p['qtype']
            mapping = '0x51384b5032540001' if q == 8 else '0x51344b5034540001' if q == 12 else '0x514b504b54000001'
            nums = [q,r['route'],*[p[n] for n in ('tm','tn','tk','wm','wn','stages','ap','dn','persistent','split','grid_mode','grid_b')],0]
            tc = '{"dense-prefill","'+p['symbol']+'","WHOLE_CALL",'+','.join(map(str,nums))+f',UINT64_C({mapping})'+'}'
        order = ',true' if r.get('dequant_first') else ''
        out.append(f'    {{{r["route"]},{r["io"]},{r["dequant"]},{tc}{order}}}, // {key}')
    out += ['};', 'inline constexpr Exact kExact[] = {']
    for p in data['points']:
        out.append('    {'+','.join(map(str, p['shape']+[index[p['winner']]]))+'},')
    out += ['};', 'inline constexpr Bucket kBuckets[] = {']
    for b in data['buckets']:
        nums = [*b['key'],*b['bounds'][0],*b['bounds'][1],index[b['recipe']],*b.get('m_bounds', [])]
        out.append('    {'+','.join(map(str,nums))+'},')
    out += ['};', '']
    return '\n'.join(out)


def summary(data):
    native = [p for p in data['points'] if p.get('native_us') is not None]
    return dict(exact=len(data['points']), buckets=len(data['buckets']),
        routes=dict(Counter(data['recipes'][p['winner']]['route'] for p in data['points'])),
        worst_bucket_training_regret_pct=max(b['worst_regret_pct'] for b in data['buckets']),
        native_comparable=len(native),
        slower_than_native=sum(p['us'] > p['native_us'] for p in native) if native else None,
        accepted_control_regressions=sum('accepted_regression' in p.get('mean_evidence', {}) for p in data['points']),
        model_performance_admitted=False)


def audit_buckets(data):
    """Leave one entire N/K family out, across all M; do not invent missing times."""
    families = sorted({(p['shape'][0],*p['shape'][2:]) for p in data['points']})
    counts, rows = Counter(), []
    for family in families:
        held = [p for p in data['points'] if (p['shape'][0],*p['shape'][2:]) == family]
        buckets = fit([p for p in data['points'] if p not in held],
                      split_m=data.get('bucket_policy') == 'ACTIVE_MEANS_MNK_INTERVALS_V2')
        for p in held:
            q,m,n,k = p['shape']
            matches = [b for b in buckets if list(bucket_key(p['shape'])) == b['key'] and
                       b['bounds'][0][0] <= n <= b['bounds'][0][1] and b['bounds'][1][0] <= k <= b['bounds'][1][1]
                       and b.get('m_bounds', [0,8192])[0] <= m <= b.get('m_bounds', [0,8192])[1]]
            require(len(matches) <= 1, 'overlapping bucket predictions')
            if not matches:
                counts['uncovered'] += 1
                continue
            rid = matches[0]['recipe']; costs = dict(p['costs'])
            if rid not in costs:
                counts['candidate_not_measured_on_held_family'] += 1
                continue
            regret = 100*(costs[rid]/p['us']-1)
            rows.append(dict(shape=p['shape'], recipe=rid, regret_pct=regret))
            counts['comparable'] += 1
    return dict(scope='LEAVE_ONE_NK_FAMILY_OUT; NOT_NEW_MODEL_HOLDOUT', counts=dict(counts),
                worst_regret_pct=max((r['regret_pct'] for r in rows), default=None),
                over_5pct=[r for r in rows if r['regret_pct'] > 5], new_model_validated=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def execution(candidate, row, definitions):
    """Freeze adapters too: cohort-current is not a stable execution identity."""
    recipe = candidate['recipe']
    p = dict(recipe['parameters'])
    if recipe['kind'] == 'full':
        dq_first = p.pop('dequant_first', False)
        require(type(dq_first) is bool, 'invalid Full stage order')
        require(set(p) == {'provider','dequant_variant','vector_io'}, 'unadmitted Full execution option')
        require(candidate['arm'] == 'full-fp16' and p['provider'] == 'typed-fp16', 'not Full-F16')
        require(candidate['helper_options'] == dict(dequant_variant=p['dequant_variant'], vector_io=p['vector_io']),
                'full helper identity differs')
        phases = [definitions[c]['phase'] for c in row['components']]
        require(len(phases)>=5 and phases[:3] == ['error_reset',*(['expansion','input'] if dq_first else ['input','expansion'])] and
                phases[3]=='main' and all(p in ('main','reducer') for p in phases[3:-1]) and phases[-1] == 'output',
                'Full trace stage order differs from recipe')
        return dict(route=2, io=p['vector_io'], dequant=p['dequant_variant'], tc=None) | (
            dict(dequant_first=True) if dq_first else {})
    names = [definitions[c]['name'] for c in row['components'] if definitions[c]['phase'] in ('input', 'output')]
    require(len(names) == 2, 'TC adapter coverage differs')
    kinds = {'::convert_rows<': -1, '::dense_convert_vec8<false,': 1,
             '::dense_convert_stripe4<false,': 2, '::dense_convert_full4<false,': 3,
             '::dense_convert_cont<false,': 4}
    ios = [next((io for name, io in kinds.items() if name in n), None) for n in names]
    require(ios[0] is not None and ios[0] == ios[1], 'unrecognized/mixed TC adapters')
    require(recipe['kind'] == 'tc' and recipe['route'] in ('fq', 'sf'), 'unsupported recipe')
    dq = p.pop('dequant_config')
    if p.get('dense_reducer'):
        width = p['dense_reducer']
        require(type(width) is int and width in (2,4,8) and p['symbol'].endswith(f'_rc{width}') and p['split'] in (2,4,8),
                'unadmitted compact reducer')
        reducers = [definitions[c] for c in row['components'] if definitions[c]['phase'] == 'reducer']
        require(len(reducers) == 1, 'compact reducer coverage differs')
        c = reducers[0]; w = c['workload']
        require(f'PpuMixedInputSplitKParallelM1FastReductionKernel<{width},{p["split"]}>' in c['name'].replace(' ','') and
                w['m']*w['n']%(32*width) == 0 and c['grid'] == [w['m']*w['n']//(32*width),1,1] and
                c['block'] == [32,1,1] and not c['shared_bytes'],
                'compact reducer launch differs')
    return dict(route=int(recipe['route'] == 'sf'), io=ios[0], dequant=dq, tc=p)
