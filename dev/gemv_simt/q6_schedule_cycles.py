"""Rank only the measured six recipes; separate warp and pipeline hypotheses."""
from dev.gemv_simt.q6_cycles import compare, CYCLES
from dev.gemv_simt.q6_schedule import WARPS, ARMS, KERNEL_NAMES

PAIRS = ((0,1), (0,2), (0,3), (1,4), (2,5))


def comparisons(reports, point):
    expected = {(r,a) for r in range(3) for a in range(6)}
    if (len(reports) != len(expected) or {(r.get('round'),r.get('arm')) for r in reports} != expected or
            any(r.get('status') != 'PASS' for r in reports)):
        raise ValueError('need 18 unique successful single-kernel profiles')
    w = point['workload']
    rows = w['m']*(w['topk'] if w['operation'] == 'indexed' else 1)
    grid = (rows*(w['n']//32),1,1)
    for r in reports:
        node = point['nodes'][r['arm']]
        if KERNEL_NAMES[r['arm']] not in node['name'] or r['symbol'] != node['symbol']:
            raise ValueError('schedule report kernel identity differs from declared arm')
    pairs = []
    for left, right in PAIRS:
        subset = [r | dict(arm=int(r['arm']==right)) for r in reports if r['arm'] in (left,right)]
        geometry = {i:(grid,(WARPS[a]*32,1,1)) for i,a in enumerate((left,right))}
        result = compare(subset, expected_geometries=geometry)
        result.update(arms=[left,right], names=[ARMS[left],ARMS[right]],
                      axis='WARPS' if right<3 else 'SHALLOW_PIPELINE')
        pairs.append(result)
    measured = {}
    for result in pairs:
        for local, arm in enumerate(result['arms']):
            measured[arm] = dict(arm=arm, name=ARMS[arm], cycles=result['samples'][local],
                                median_cycles=result['median'][local], warps=WARPS[arm])
    ranked = sorted(measured.values(), key=lambda x:x['median_cycles'])
    matched = all(p['clock_conditions_match'] for p in pairs)
    # No estimated time/opcode-count score, no automatic selector change.
    return dict(metric=CYCLES[0], comparisons=pairs, ranking=ranked,
                fastest_measured_arm=ranked[0]['arm'] if matched else None,
                clock_conditions_match=matched, performance_admitted=False, production_changed=False,
                scope='SIX_RECIPES_ONE_POINT_NOT_GLOBAL_OPTIMUM_OR_MODEL_TPOT')
