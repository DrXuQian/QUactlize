"""Production catalog validation."""
import math
from quactlize.dispatch.dense_prefill import require
STATISTIC = 'ARITHMETIC_MEAN_COMPLETE_ASYS_GPU_CALL_BOTH_CAPTURES'


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8)


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
    require(e['review_sha256'] in data['reviews'] and not e['native_source_commit_proven'],
            'missing review or invented native commit binding')
    update = next(u for u in data['mean_updates'] if u['plan_sha256'] == e['plan_sha256'])
    require(e['source_commit'] == update['admission']['source_commit'], 'source commit admission differs')
    if e['source_commit'] is not None:
        receipt = data['authorities'][p['authority']]['receipt']
        require(receipt.get('source_revision', {}).get('quactlize_commit') == e['source_commit'],
                'source commit differs from measured authority')
    costs = dict(p['costs'])
    captures = e['capture_means']
    require(set(captures) == set(costs), 'mean cost coverage differs')
    for rid, pairs in captures.items():
        require(close(mean_from_pairs(pairs), costs[rid]), 'mean cost differs')
    transition = update['admission'].get('transition_fq', False) and p['shape'][1] in (16,32,64)
    require(data['recipes'][p['winner']]['route'] in ((0,1) if transition else (1,2)),
            'unadmitted FQ/Full prefill replacement')
    require(p['winner_evidence']['captures'] == 2 and close(p['winner_evidence']['mean_us'], p['us']),
            'winner evidence differs')
    control = e['control']
    validate_proof(control)
    validate_proof(p['winner_evidence'])
    require(p['winner_evidence']['capture_means'] == captures[p['winner']], 'winning captures differ')
    delta = 100*(p['us']/control['mean_us']-1)
    require(close(delta, e['delta_pct']), 'control mean delta differs')
    if delta > 0:
        approved = accepted_exception(p['shape'], p['winner_evidence']['candidate'],
                                      control['candidate'], delta, update['admission'])
        require(e.get('accepted_regression') == approved, 'regression approval differs')
    else:
        require('accepted_regression' not in e, 'unneeded regression exception')
