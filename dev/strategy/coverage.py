"""Do not present a partially executed factored plan as a completed sweep."""


def execution_gaps(plan):
    """Current development runner capabilities, before ANY device/JIT work.

    The host search-space audit may be useful before its executor is ready.
    Keep that distinction machine-readable. Remove a gap only together with
    its executor and independent coverage tests, not by dropping plan axes.
    """
    gaps = {}

    def missing(reason, point):
        gaps.setdefault(reason, []).append(point['id'])

    for point in plan['points']:
        if point.get('expansion_search'):
            missing('INDEPENDENT_EXPANSION_DRIVER_PENDING', point)
        if point.get('grid_search'):
            missing('RESOURCE_GRID_EXPANSION_DRIVER_PENDING', point)
        recipes=point['candidates']+point.get('champion_controls',[])
        w = point['workload']
        if w['operation'] != 'dense' and w['m'] > 8:
            missing('GROUPED_PREFILL_DRIVER_PENDING', point)
        if any(c['kind'] == 'q4' for c in recipes):
            missing('EXPLICIT_Q4_RECIPE_CARRIER_PENDING', point)
        if w['operation'] != 'dense' and w['qtype'] != 8 and any(
                c['route'] == 'sf' for c in recipes):
            missing('INDEXED_TYPED_SF_DRIVER_PENDING', point)
    return dict(status='BLOCKED' if gaps else 'READY', gaps=gaps,
                rule_space_is_not_execution_coverage=True, device_started=False)


def require_executable(plan):
    result = execution_gaps(plan)
    if result['gaps']:
        raise ValueError('component runner is not ready for this plan: ' + ', '.join(result['gaps']) +
                         '; use --plan-only; no device workload or JIT has started')
    return result
