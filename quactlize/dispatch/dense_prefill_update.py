"""Offline, exact-only admission of a reviewed fixed-mainloop helper experiment.

This updates the existing catalog, not a second runtime routing table. Missing
incumbents stay unchanged. Bucket training never consumes these helper trials.
"""
import copy
import json
import lzma
from pathlib import Path

from quactlize.dispatch.dense_prefill import (admitted_costs, bucket_costs, choose,
    execution, fit, import_reviews, require, validate)
from quactlize.runtime.tuning import digest


def update_helpers(base, review_path, plan_path):
    from dev.prefill_asys.io_refine import validate as validate_plan, HELPER_SCHEMA

    validate(base)
    plan = validate_plan(json.loads(Path(plan_path).read_text()))
    require(plan['schema'] == HELPER_SCHEMA, 'expected fixed-mainloop helper factorial')
    path = Path(review_path)
    review = json.loads(lzma.decompress(path.read_bytes()) if path.suffix == '.xz' else path.read_bytes())
    # Reuse the full sample/identity/arithmetic audit; never substitute a report
    # summary or independently summed medians for these complete-call samples.
    measured = import_reviews([path])
    require(measured['domain'] == base['domain'], 'helper update timing domain differs')
    specs = {p['id']:p for p in plan['points']}
    require(set(specs) == set(review['points']) and not review.get('diagnostics'), 'helper coverage differs')
    require(review['review_sha256'] not in base['reviews'], 'helper review already imported')
    data = copy.deepcopy(base)
    points = {p['key']:p for p in data['points']}
    pending = []
    for p in measured['points']:
        spec, raw = specs[p['key']], review['points'][p['key']]
        rows = {r['candidate']:r for r in raw['measurements'] if r['arm'] != 'native-fp16'}
        require(set(rows) == {c['id'] for c in spec['cases']}, 'helper candidate inventory differs')
        controls, captures, observations = [], {}, {}
        for c in spec['cases']:
            row = rows[c['id']]
            require(row['roles'] == c['roles'], 'helper experiment role differs')
            rid = digest(execution(review['candidates'][c['id']], row, review['components']))
            cs = row.get('confirmation_captures', [])
            require(len(cs) == 2, 'helper admission requires two reversed captures')
            captures[rid] = [s['gpu_sum_us'] for s in cs]
            observations[rid] = dict(candidate=c['id'], receipt_sha256=digest(row),
                                    captures=2, components=row['components'])
            if c['id'] in spec['prior_route_winners'].values():
                controls.append(rid)
        old = points.get(p['key'])
        if old and old['winner'] not in captures:
            pending.append(dict(key=p['key'], shape=p['shape'], incumbent=old['winner'],
                review_point_sha256=digest(raw), reason='INCUMBENT_NOT_REMEASURED_KEEP_EXISTING_EXACT'))
            continue
        costs = dict(p['costs'])
        p['admission'] = dict(rule='BOTH_CAPTURE_MEDIANS_BEAT_REMEASURED_CONTROL',
            controls=sorted(controls), baseline=choose({r:costs[r] for r in controls}),
            incumbent=old['winner'] if old else None, capture_medians=captures)
        winner = choose(admitted_costs(p))
        p.update(winner=winner, us=costs[winner], winner_evidence=observations[winner])
        p['bucket_costs'] = sorted(bucket_costs(old).items()) if old else []
        if old:
            p['bucket_evidence'] = old.get('bucket_evidence') or dict(
                authority=old['authority'], evidence_sha256=old['evidence_sha256'])
        points[p['key']] = p
    data['points'] = sorted(points.values(), key=lambda p:p['shape'])
    data['reviews'] += measured['reviews']
    data['authorities'].update(measured['authorities'])
    data['recipes'].update(measured['recipes'])
    used = {r for p in data['points'] for r,_ in p['costs']} | {r for p in data['points'] for r in bucket_costs(p)}
    data['recipes'] = {r:data['recipes'][r] for r in sorted(used)}
    data.setdefault('helper_updates', []).append(dict(base_sha256=base['sha256'], plan_sha256=plan['sha256'],
        review_sha256=review['review_sha256'], exact_only=True, reviewed=len(specs),
        imported=len(specs)-len(pending), retained_pending=pending))
    data['buckets'] = fit(data['points'])
    require(data['buckets'] == base['buckets'], 'exact helper update changed bucket predictions')
    data['criterion'] = 'MINIMUM_ADMITTED_COMPLETE_GPU_CALL; NO_CROSS_POINT_ABSOLUTE_TIME_COMPARISON'
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    validate(data)
    return data
