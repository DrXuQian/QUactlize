"""Full-call cost from independently measured invocations, never guessed zeros."""
from collections import Counter
import math
import statistics

from quactlize.runtime.tuning import digest

SCOPE = 'SUM_OF_INDEPENDENT_COMPONENTS_NOT_MEASURED_E2E_OR_TPOT'
WORK = frozenset(('producer', 'provider', 'adapter', 'prepare', 'expansion',
                  'reducer', 'reducer_finish', 'postprocess', 'routing', 'reset'))


def checked_component(row):
    if row.get('correctness') != 'PASS' or row.get('role') not in WORK:
        raise ValueError('component lacks correctness or a reviewed role')
    if not row.get('first_use_excluded') or row.get('host_launch_cost_added') is not False:
        raise ValueError('first use or estimated launch cost included')
    samples = row['samples_us']
    if len(samples) < 3 or any(len(r) < 3 for r in samples) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for r in samples for v in r):
        raise ValueError('component requires at least 3x3 finite positive samples')
    provider = row['role'] == 'provider'
    if provider:
        if (row.get('timing') != 'ORDINARY_PROVIDER_EVENTS' or row.get('cuda_graph') is not False
                or row.get('internal_reduction_included') is not True):
            raise ValueError('provider must be ordinary events with internal work included')
    elif row['role']=='reset':
        if row.get('timing')!='ORDINARY_MEMORY_EVENTS' or row.get('cuda_graph') is not False:
            raise ValueError('reset must have its own memory-API event interval')
    elif row.get('timed_kernel_types') != 1:
        raise ValueError('each kernel timer must contain exactly one kernel type')
    if not row.get('identity') or not row.get('cache'):
        raise ValueError('missing implementation/device/cache identity')
    rounds = [statistics.median(r) for r in samples]
    return dict(median_us=statistics.median(rounds), rounds_us=rounds,
                spread_pct=100*(max(rounds)/min(rounds)-1))


def compose(invocations, components, *, required, inventory_complete):
    """A shared measurement may legitimately occur twice in a call.

    Invocation IDs must differ. Semantic work tokens must not overlap: a
    reducer_finish owns both reduction and output; a provider owns its own
    internal reducer. This catches double charging without set-deduplicating
    real repeated launches.
    """
    seen, work, missing, terms = set(), set(), [], []
    for call in invocations:
        if call['id'] in seen:
            raise ValueError('duplicate invocation')
        seen.add(call['id'])
        covers = set(call['covers'])
        if not covers or work & covers:
            raise ValueError('empty or overlapping component responsibilities')
        work.update(covers)
        component = components.get(call['component'])
        if component is None:
            missing.append(call['component']); continue
        stats = checked_component(component)
        terms.append(dict(invocation=call['id'], component=call['component'],
                          role=component['role'], **stats))
    missing_work = sorted(set(required)-work)
    complete = inventory_complete and not missing and not missing_work and bool(terms)
    return dict(status='COMPLETE' if complete else 'INCOMPLETE', terms=terms,
                sum_us=math.fsum(r['median_us'] for r in terms) if complete else None,
                partial_sum_us=math.fsum(r['median_us'] for r in terms),
                missing_components=missing, missing_work=missing_work,
                scope=SCOPE, model_tpot_admitted=False, production_changed=False)


def reuse_key(*, implementation, device, sdk, shape, io, router, launch, cache, timing):
    # No reuse by symbol or N/K alone. Runtime arguments, split and precision
    # are in shape/io/launch. Physical PCI remains explicit in device.
    fields = dict(implementation=implementation, device=device, sdk=sdk,
                  shape=shape, io=io, router=router, launch=launch, cache=cache, timing=timing)
    if any(v is None for v in fields.values()):
        raise ValueError('incomplete component reuse identity')
    return digest(fields)


def comparison(native, candidate, *, paired=False):
    if native['status'] != 'COMPLETE' or candidate['status'] != 'COMPLETE':
        return dict(status='INCOMPLETE', no_regression_admitted=False, scope=SCOPE)
    ratio = candidate['sum_us']/native['sum_us']
    # A comparison of historical medians is not paired performance admission.
    return dict(status='PAIRED_REVIEW_REQUIRED' if paired else 'HISTORICAL_SCREEN_ONLY',
                native_us=native['sum_us'], quactlize_us=candidate['sum_us'],
                ratio=ratio, delta_pct=100*(ratio-1), no_regression_admitted=False,
                scope=SCOPE)
