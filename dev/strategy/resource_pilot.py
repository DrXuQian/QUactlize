"""Reuse a sealed resource probe for a bounded execution-mechanism gate.

This is not a reduced performance search. It neither recommends a recipe nor
turns missing Q4/SF/full drivers into campaign coverage. Old rule generation
is frozen in the imported plan; current module identities must still match.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path

from dev.strategy.resources import inventory, module_spec
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


SCOPE = 'TC_EXECUTION_AND_SINGLE_KERNEL_TIMER_SMOKE_NOT_SELECTION'


def sealed(path, identity):
    value = json.loads(Path(path).read_text())
    if (value.get('identity') != identity or value.get('sha256') != digest(
            {k: v for k, v in value.items() if k != 'sha256'})):
        raise ValueError('resource receipt identity/digest differs: ' + str(path))
    return value['data']


def verified_plan(value):
    if value.get('plan_sha256') != digest({k: v for k, v in value.items() if k != 'plan_sha256'}):
        raise ValueError('resource plan digest differs')
    return value


def load_probe(directory):
    """Host-only checks. Device/SDK/current compiled-source checks follow.

    No requirement that an added Python test driver have the old planner's
    hash: the OLD plan is imported, not regenerated under a new rule. Every
    kernel's compiler identity is checked separately before device execution.
    """
    root = Path(directory).resolve(strict=True)
    plan = verified_plan(json.loads((root / 'plan/plan.json').read_text()))
    if plan.get('cohort') != 'pilot' or len(plan['points']) > 6:
        raise ValueError('execution smoke accepts only the six-point resource pilot')
    resources = root / 'resources'
    resolved = verified_plan(json.loads((resources / 'resolved-plan.json').read_text()))
    authority = json.loads((resources / 'authority.json').read_text())
    if (authority.get('plan') != plan['plan_sha256'] or
            authority.get('source') != plan['source_identity']['sha256'] or
            resolved.get('resource_authority') != authority):
        raise ValueError('resource plan/source/authority differs')
    inv = inventory(plan)
    if inv != json.loads((resources / 'inventory.json').read_text()):
        raise ValueError('resource inventory differs from frozen plan')
    receipts = {}
    for key, query in inv['queries'].items():
        row = sealed(resources / 'queries' / (key + '.json'), dict(authority=authority, query=key))
        if row.get('request') != query['request'] or row.get('status') not in ('READY', 'UNAVAILABLE'):
            raise ValueError('resource query is missing or incomplete: ' + key)
        if row['status'] == 'READY' and (row.get('occupancy', 0) <= 0 or
                any(row['device'].get(k) != authority['device'][k] for k in ('ordinal', 'compute_units'))):
            raise ValueError('resource query device/occupancy differs')
        receipts[key] = row
    if resolved.get('resource_receipts_sha256') != digest(receipts):
        raise ValueError('resolved plan resource digest differs')
    # Re-derive grids from the sealed receipts, not an editable resolved list.
    from dev.strategy.resources import resolve_point
    if resolved['points'] != [resolve_point(p, receipts) for p in plan['points']]:
        raise ValueError('resolved recipes/grids differ from resource queries')
    summary = json.loads((resources / 'summary.json').read_text())
    want = dict(counts=inv['counts'], status='COMPLETE', missing=[], kernel_launches=0,
                ready=sum(r['status'] == 'READY' for r in receipts.values()),
                unavailable=sum(r['status'] == 'UNAVAILABLE' for r in receipts.values()))
    if any(summary.get(k) != v for k, v in want.items()):
        raise ValueError('resource summary differs from complete receipts')
    return dict(root=root, plan=plan, resolved=resolved, authority=authority,
                inventory=inv, receipts=receipts, summary=summary)


def smoke_cases(probe):
    """At most S1 + split per workload; geometry order, never old latency.

    Prefer a persistent S1 when available and ordinary split otherwise. This
    covers mechanisms, NOT the best configuration. Keep every omission named.
    """
    cases, omitted = [], []
    for point in probe['resolved']['points']:
        w = Workload(**point['workload'])
        choices = []
        for value in point['candidates']:
            r = Recipe.from_dict(value)
            if r.kind != 'tc' or (r.route == 'sf' and w.qtype != 8):
                continue
            binding = point['resource_resolution']['bindings'].get(r.key)
            if not binding:
                raise ValueError('resolved TC has no query binding')
            choices.append((r, binding))
        for split_kind in ('s1', 'split'):
            pool = [(r, b) for r, b in choices if (dict(r.parameters)['split'] == 1) == (split_kind == 's1')]
            if not pool:
                omitted.append(dict(point=point['id'], split_kind=split_kind, reason='NO_READY_TC_IN_SMOKE_SCOPE'))
                continue
            def priority(item):
                r, b = item
                p = dict(r.parameters)
                prefer = 1 if split_kind == 's1' else 0
                return (b['launch']['algorithm'] != prefer, p['split'],
                        (p['tm'] // p['wm']) * (p['tn'] // p['wn']), p['stages'], r.key)
            r, binding = min(pool, key=priority)
            spec = module_spec(w, r)
            case = dict(point=point['id'], workload=point['workload'], recipe=asdict(r))
            case.update(binding=binding, module=digest(spec), spec=spec, split_kind=split_kind,
                        numerical='INDEPENDENT_GGUF_AND_CHANGED_INPUT_GRAPH_PROOF',
                        timing='WARM_SINGLE_KERNEL_SMOKE_NOT_COLD_PERFORMANCE')
            case['id'] = digest(case)
            cases.append(case)
    if not cases or len(cases) > 12:
        raise ValueError('empty or oversized TC execution pilot')
    result = dict(scope=SCOPE, cases=cases, omitted=omitted, recommendation=None,
                production_changed=False, performance_admitted=False,
                excluded=['SIMT_AND_Q4_SEARCH', 'KQUANT_SF_EXPANSION', 'FULL_DEQUANT',
                          'COLD_WEIGHT_COMPARISON', 'ORIGINAL_WIRING_AB', 'FULL_CAMPAIGN'])
    return json.loads(json.dumps(result))


def module_record(probe, case):
    mid = case['module']
    spec = probe['inventory']['modules'].get(mid)
    if spec != case['spec']:
        raise ValueError('smoke module not present in resource pilot')
    row = sealed(probe['root'] / 'resources/modules' / (mid + '.json'),
                 dict(authority=probe['authority'], module=mid))
    if row.get('status') != 'PASS':
        raise ValueError('resource module was not compiled')
    record = row['record']
    path = Path(record['path'])
    cache = (probe['root'] / 'cache').resolve(strict=True)
    if (path.is_symlink() or not path.resolve(strict=True).is_relative_to(cache) or
            path.name != 'kernel.so' or path.parent.name != record['key'] or
            record['parent'] != spec['parent'] or sha(path) != record['sha256']):
        raise ValueError('resource module path/parent/hash differs')
    cached = json.loads((path.parent / 'manifest.json').read_text())
    if any(record.get(k) != cached.get(k) for k in ('key', 'parent', 'identity', 'sha256')):
        raise ValueError('resource module cache manifest differs')
    return record


def current_module(record, spec, compiler):
    """Identity check ONLY: never call build() or silently JIT a new module."""
    key = digest(dict(identity=compiler.identity, parent=spec['parent'],
                      source=compiler.source(spec['parent'], '')))
    if record['identity'] != compiler.identity or record['key'] != key:
        raise ValueError('compiled source/SDK/host changed; run a new resource pilot')
    compiler.check_inputs()
    return key


def validate_pass(case, result):
    """A PASS label alone cannot make a partially measured arm resumable."""
    from dev.strategy.components import statistics_checked
    if (result.get('status') != 'PASS' or result.get('production_changed') is not False or
            result.get('performance_admitted') is not False or
            result.get('scope') != 'WARM_TC_MECHANISM_GATE_NOT_COLD_SELECTION'):
        raise ValueError('pilot PASS scope differs')
    w = Workload(**case['workload'])
    proofs = result.get('numerical', [])
    if len(proofs) != (7 if w.compute == 'bf16' else 5) or not isinstance(result.get('extracted'), dict):
        raise ValueError('pilot omitted numerical/replay checks')
    if w.compute == 'bf16':
        magnitude = result.get('large_activation_max')
        if not isinstance(magnitude, (int, float)) or not math.isfinite(magnitude) or magnitude <= 65504:
            raise ValueError('pilot BF16 overflow-range check missing')
    for proof in [*proofs, result['extracted']]:
        error = proof.get('relative_l1_error', float('inf'))
        if (proof.get('cells') != w.m * w.topk * w.n or proof.get('bad') != 0 or
                proof.get('nonfinite') != 0 or proof.get('tolerance') != .005 or
                not isinstance(error, (int, float)) or not math.isfinite(error) or not 0 <= error < .005):
            raise ValueError('pilot numerical proof differs')
    negative = result.get('negatives', {})
    planes = ['low', 'high', 'units'] if w.qtype in (11, 13, 14) else ['low', 'units']
    if (negative.get('zero_input') != 'DETECTED' or negative.get('omitted_producer') != 'DETECTED' or
            negative.get('ring_planes') != planes):
        raise ValueError('pilot negative controls incomplete')
    stages = result.get('components', [])
    kinds = [s.get('role') for s in stages]
    split = dict(Recipe.from_dict(case['recipe']).parameters)['split']
    if (kinds.count('producer') != 1 or any(k not in ('producer', 'prepare', 'reducer', 'reducer_finish', 'postprocess') for k in kinds) or
            split > 1 and not any(k in ('reducer', 'reducer_finish') for k in kinds)):
        raise ValueError('pilot component inventory incomplete')
    for stage in stages:
        statistics = statistics_checked(stage['samples_us'])
        if (any(stage.get(k) != v for k, v in statistics.items()) or
                stage.get('timed_kernel_types') != 1 or stage.get('first_use_excluded') is not True or
                stage.get('cache') != 'WARM_SMOKE_NO_COLD_OR_PERFORMANCE_ADMISSION'):
            raise ValueError('pilot component timing/scope differs')
    return result
