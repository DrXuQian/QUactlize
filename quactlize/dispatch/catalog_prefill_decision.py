"""Sealed production catalog validation/generation; no device experiments."""
from quactlize.dispatch.catalog_components import require


def choose(point, measured, qualified):
    """Fastest feasible finalist, not fastest first and then fail the point.

    Candidate stability, zero median regression and four of six paired wins
    are unchanged. A noisy control is usable only when its *best* round is
    slower than the candidate's *worst* round. Overlapping noisy ranges still
    require another device experiment. This is a component gate, not model AB.
    """
    stats = measured['statistics']
    controls = [a for a, v in point['arms'].items() if 'control' in v['roles'] or v['original']]
    require(controls and point['original'] in controls, 'missing original/control')
    best_control = min(controls, key=lambda a: (stats[a]['median_us'], a))
    against = sorted({best_control, point['original']})
    eligible = {}
    for aid in sorted(qualified):
        a = stats[aid]
        if a['spread_pct'] > 5:
            continue
        comparisons = {}
        for cid in against:
            c = stats[cid]
            paired = sum(x <= y * (1 + 1e-12) for x, y in zip(a['rounds_us'], c['rounds_us']))
            separated = max(a['rounds_us']) < min(c['rounds_us'])
            if (a['median_us'] > c['median_us'] * (1 + 1e-12) or paired < 4 or
                    (c['spread_pct'] > 5 and not separated)):
                break
            comparisons[cid] = dict(paired_wins=paired, rounds=6,
                control_spread_pct=c['spread_pct'],
                stability_basis='SEPARATED_OBSERVED_RANGES' if c['spread_pct'] > 5 else 'STABLE_CONTROL')
        else:
            eligible[aid] = comparisons
    require(eligible, 'no generated stable non-regressing finalist; device confirmation required')
    winner = min(eligible, key=lambda a: (stats[a]['median_us'], a))
    return dict(winner=winner, best_control=best_control, comparisons=eligible[winner],
                eligible=sorted(eligible), qualified=sorted(qualified))
