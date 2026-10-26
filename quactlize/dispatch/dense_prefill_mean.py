"""Exact-only import of complete M-span Asys observations, ranked by mean.

The old median-based catalog and its native comparisons stay historical.
Every new mean includes both capture directions and every measured component;
the importer never joins helper timings from different runs or drops outliers.
"""
import copy
import hashlib
import json
import lzma
import math
from pathlib import Path
import statistics

from quactlize.dispatch.dense_prefill import (audit_row, bucket_costs, choose,
    execution, fit, require, validate)
from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest

STATISTIC = 'ARITHMETIC_MEAN_COMPLETE_ASYS_GPU_CALL_BOTH_CAPTURES'


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)


def capture_mean(row):
    captures = row.get('confirmation_captures', [])
    require(len(captures) == 2 and [c['confirmation'] for c in captures] == [0, 1],
            'mean admission needs both capture directions')
    samples = [t for c in captures for t in c['complete_samples_us']]
    require(row['complete_samples_us'] == samples, 'aggregate samples differ from captures')
    require(all(math.isfinite(t) and t > 0 for t in samples), 'invalid complete sample')
    require(all(c.get('omitted_producer') == 'DETECTED' for c in captures), 'producer check missing')
    return statistics.fmean(samples), [
        [statistics.fmean(c['complete_samples_us']), len(c['complete_samples_us'])] for c in captures]


def proof(row):
    us, means = capture_mean(row)
    return dict(candidate=row['candidate'], receipt_sha256=digest(row), captures=2,
        components=row['components'], capture_means=means, mean_us=us,
        samples_sha256=digest(row['complete_samples_us']),
        raw_receipts=[c['raw_receipt_sha256'] for c in row['confirmation_captures']],
        max_relative_l1_error=max(c['relative_l1_error'] for cap in row['confirmation_captures']
                                  for c in cap['numerical']))


def accepted_exception(shape, winner, control, delta, admission):
    matches = [e for e in admission['accepted_regressions'] if e['shape'] == shape and
               e['candidate'] == winner and e['control'] == control]
    require(len(matches) == 1 and 0 < delta <= matches[0]['maximum_delta_pct'],
            f'unapproved mean regression: {shape}, {delta:.6f}%')
    return matches[0]


def mean_from_pairs(pairs):
    require(len(pairs) == 2 and all(math.isfinite(t) and t > 0 and type(n) is int and n >= 3
                                   for t, n in pairs), 'invalid mean capture')
    return sum(t*n for t,n in pairs)/sum(n for _,n in pairs)


def validate_proof(value):
    require(value['captures'] == 2 and close(mean_from_pairs(value['capture_means']), value['mean_us']),
            'proof mean differs')
    require(0 <= value['max_relative_l1_error'] < .005, 'proof numerical gate failed')
    hashes = [value['candidate'],value['receipt_sha256'],value['samples_sha256'],*value['raw_receipts']]
    require(len(value['raw_receipts']) == 2 and all(len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                                                  for h in hashes), 'invalid proof binding')


def validate_mean_point(p, data):
    e = p['mean_evidence']
    require(e['statistic'] == STATISTIC and p.get('native_us') is None, 'mean/native scope differs')
    require(e['review_sha256'] in data['reviews'] and e['source_commit'] is None and
            not e['native_source_commit_proven'], 'missing review or invented commit binding')
    costs = dict(p['costs'])
    captures = e['capture_means']
    require(set(captures) == set(costs), 'mean cost coverage differs')
    for rid, pairs in captures.items():
        require(close(mean_from_pairs(pairs), costs[rid]), 'mean cost differs')
    require(data['recipes'][p['winner']]['route'] in (1, 2), 'FQ cannot be a prefill replacement')
    require(p['winner_evidence']['captures'] == 2 and close(p['winner_evidence']['mean_us'], p['us']),
            'winner evidence differs')
    control = e['control']
    validate_proof(control)
    validate_proof(p['winner_evidence'])
    require(p['winner_evidence']['capture_means'] == captures[p['winner']], 'winning captures differ')
    delta = 100*(p['us']/control['mean_us']-1)
    require(close(delta, e['delta_pct']), 'control mean delta differs')
    if delta > 0:
        update = next(u for u in data['mean_updates'] if u['plan_sha256'] == e['plan_sha256'])
        approved = accepted_exception(p['shape'], p['winner_evidence']['candidate'],
                                      control['candidate'], delta, update['admission'])
        require(e.get('accepted_regression') == approved, 'regression approval differs')
    else:
        require('accepted_regression' not in e, 'unneeded regression exception')


def update_mean(base, paths, plan, admission):
    from dev.prefill_asys.m_span import validate as validate_plan
    from dev.prefill_asys.finalists import case_identity
    from tools.review_prefill_cohort import domain, eligible

    validate(base)
    validate_plan(plan)
    require(plan['sha256'] == admission['plan_sha256'] and plan['inputs']['policy_sha256'] == base['sha256'],
            'mean admission plan/base differs')
    require(admission['source_commit'] is None and not admission['native_source_commit_proven'],
            'full native source binding is not in these uploads')
    specs = {p['id']: p for p in plan['points']}
    data = copy.deepcopy(base)
    points = {p['key']: p for p in data['points']}
    seen, pending, sources, imported_reviews = set(), set(), [], []
    used_exceptions = []
    for path in paths:
        path = Path(path); raw = path.read_bytes()
        review = json.loads(lzma.decompress(raw) if path.suffix == '.xz' else raw)
        rid = review['review_sha256']
        require(review['schema'] == 'quactlize.prefill-component-review.v1' and
                rid == digest({k:v for k,v in review.items() if k != 'review_sha256'}), 'review seal differs')
        imported_reviews.append(rid)
        require(rid not in data['reviews'], 'review already imported')
        sources.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                            review_sha256=rid, complete_points=len(review['points'])))
        pending.update(review['pending_points'])
        for aid, authority in review['authorities'].items():
            require(aid == digest(authority) and domain(authority) == base['domain'], 'authority/domain differs')
            data['authorities'][aid] = dict(domain=domain(authority), receipt_sha256=aid,
                source_commit=None, native_source_commit_proven=False, receipt=authority)
        for key, point in sorted(review['points'].items()):
            require(key in specs and key not in seen and point['workload'] == specs[key]['workload'],
                    'complete point not in plan or measured twice')
            require(point.get('native_comparison') is False, 'expected explicit no-native M-span campaign')
            seen.add(key)
            w = Workload(**point['workload']); shape = [w.qtype,w.m,w.n,w.k]
            cases = {c['id']:c for c in specs[key]['cases']}
            measured = {r['case_id'] for r in point['measurements']}
            require(len(measured) == len(point['measurements']) and measured == set(cases),
                    'incomplete candidate denominator')
            means, costs, captures, observations = {}, {}, {}, {}
            for row in point['measurements']:
                case = cases[row['case_id']]; identity = case_identity(case)
                require(row['candidate'] == digest(identity) and
                        review['candidates'].get(row['candidate']) == identity and row['roles'] == case['roles'],
                        'candidate identity/roles differ')
                audit_row(row, review, point)
                us, pair = capture_mean(row)
                means[row['candidate']] = us
                # FQ/legacy-FP16 remain comparison controls. This campaign's
                # permitted replacement routes are SF and Full-F16 only.
                if not eligible(row) or row['arm'] not in ('sf', 'full-fp16'):
                    continue
                recipe = execution(identity, row, review['components']); recipe_id = digest(recipe)
                require(recipe_id not in costs, 'duplicate complete execution identity')
                data['recipes'][recipe_id] = recipe
                costs[recipe_id] = us; captures[recipe_id] = pair; observations[recipe_id] = row
            require(costs, 'missing SF/Full candidates')
            winner = choose(costs); row = observations[winner]
            all_best = min((r for r in point['measurements'] if eligible(r)),
                           key=lambda r: (means[r['candidate']], r['candidate']))
            require(close(costs[winner], means[all_best['candidate']]),
                    'SF/Full replacement is not the overall measured minimum')
            controls = [r for r in point['measurements'] if 'model-config' in r['roles']]
            require(controls, 'missing same-run model control')
            control = min(controls, key=lambda r: (means[r['candidate']], r['candidate']))
            delta = 100*(costs[winner]/means[control['candidate']]-1)
            evidence = dict(statistic=STATISTIC, review_sha256=rid, plan_sha256=plan['sha256'],
                capture_means=captures, control=dict(proof(control), arm=control['arm']), delta_pct=delta,
                all_candidate_means_sha256=digest(means), measured_candidates=len(measured),
                source_commit=None, native_source_commit_proven=False)
            if delta > 0:
                approval = accepted_exception(shape,row['candidate'],control['candidate'],delta,admission)
                evidence['accepted_regression'] = approval; used_exceptions.append(approval)
            old = points.get(key)
            p = dict(key=key, shape=shape, winner=winner, costs=sorted(costs.items()), us=costs[winner],
                native_us=None, authority=point['authority'], evidence_sha256=digest(point),
                winner_evidence=proof(row), mean_evidence=evidence,
                bucket_costs=sorted(bucket_costs(old).items()) if old else [])
            if old:
                data.setdefault('superseded_points', []).append(old)
                p['bucket_evidence'] = old.get('bucket_evidence') or dict(
                    authority=old['authority'], evidence_sha256=old['evidence_sha256'])
            points[key] = p
        data['reviews'].append(rid)
    require(imported_reviews == admission['reviews'], 'unreviewed measurement source')
    require(seen == set(specs) and not pending-seen, 'unresolved M-span points')
    require(used_exceptions == admission['accepted_regressions'], 'unused or duplicate regression approval')
    data['points'] = sorted(points.values(), key=lambda p:p['shape'])
    data['superseded_points'].sort(key=lambda p:p['shape'])
    data.setdefault('mean_updates', []).append(dict(plan_sha256=plan['sha256'], base_sha256=base['sha256'],
        admission=admission, sources=sources, imported=len(seen), exact_only=True,
        statistic=STATISTIC, native_comparison=False, bucket_predictions_changed=False))
    data['buckets'] = fit(data['points'])
    require(data['buckets'] == base['buckets'], 'mean exact admission changed bucket predictions')
    data['criterion'] = STATISTIC+'; HISTORICAL_BUCKET_TRAINING_UNCHANGED'
    data['sha256'] = digest({k:v for k,v in data.items() if k != 'sha256'})
    validate(data)
    return data
