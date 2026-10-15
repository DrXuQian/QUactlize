"""Explicit experiment scope; never a shape exception in kernel selection."""
import json

from quactlize.dispatch.table import QTYPES
from quactlize.runtime.tuning import digest


def point_filter(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) - {'qtype', 'operation', 'n', 'k'}:
        raise ValueError('point filter accepts only qtype, operation, n and k')
    for key, item in value.items():
        if key == 'operation':
            if item not in ('dense', 'indexed'):
                raise ValueError('point filter operation must be dense or indexed')
        elif type(item) is not int or not 0 < item <= 2147483647:
            raise ValueError('point filter dimensions must be positive int32')
        elif key == 'qtype' and item not in QTYPES:
            raise ValueError('unsupported point filter qtype')
    return value


def filtered_plan(plan, selected):
    if not selected:
        return plan
    selected = point_filter(json.dumps(selected))
    points = [p for p in plan['points'] if all(p['workload'][k] == v for k, v in selected.items())]
    if not points:
        raise ValueError('point filter matches no registered small-M workload')
    # Do not copy the full registry's compile/arm counts onto this subset.
    # The campaign recomputes actual inventory immediately before building.
    result = {k:v for k,v in plan.items() if k not in ('points', 'plan_sha256', 'work_counts')}
    result.update(points=points, point_filter=selected, registry_points=len(plan['points']),
                  selection_scope='EXPLICIT_SMALLM_SUBSET_NOT_FULL_COHORT')
    result['plan_sha256'] = digest(result)
    return result


def completion_status(passed, expected, unresolved, selected):
    if passed != expected or unresolved:
        return 'INCOMPLETE'
    return 'COMPLETE_SMALLM_SUBSET' if selected else 'COMPLETE_SMALLM'
