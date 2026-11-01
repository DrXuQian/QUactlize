"""Adapt one workload inventory to the existing measured Asys backends."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from dev.model_tuning.inventory import sealed, validate
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'quactlize.model-tuning-campaign.v1'


def partition(inventory):
    validate(inventory)
    groups, pending, bindings = defaultdict(dict), [], {}
    for point in inventory['points']:
        w = Workload(**point['workload'])
        if point['contract']['form'] == 'swiglu':
            pending.append(dict(point=point['id'], reason='PAIRED_EPILOGUE_ASYS_CARRIER_NOT_YET_CONNECTED'))
            continue
        if w.m > 8 and w.operation == 'dense' and w.m not in (16, 32, 64) and not 128 <= w.m <= 8192:
            pending.append(dict(point=point['id'], reason='DENSE_M_OUTSIDE_CURRENT_PREFILL_CARRIER'))
            continue
        backend = 'smallm-'+w.operation if w.m <= 8 else 'dense-prefill' if w.operation == 'dense' else 'moe-prefill'
        # A canonical concatenation is a real 2N projection, not a fused
        # epilogue. Share its measurement with an identical ordinary matrix.
        execution = groups[backend].setdefault(w.key, dict(id=w.key, workload=asdict(w), consumers=[], tensors=[]))
        for c in point['consumers']:
            if c not in execution['consumers']:
                execution['consumers'].append(c)
                execution['tensors'].append(c['model']+':'+c['tensor'])
        bindings[point['id']] = dict(backend=backend, execution=w.key)
    return dict(groups={k:list(v.values()) for k,v in sorted(groups.items())},
                pending=pending, bindings=bindings)


def _scope(inventory, points):
    names = {c['model'] for p in points for c in p['consumers']}
    return dict(models=[m for m in inventory['models'] if m['name'] in names], excluded=inventory['excluded'])


def make_engine(inventory, backend, points, output, *, compute_units=72):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    scope = _scope(inventory, points)
    if backend.startswith('smallm-'):
        from dev.strategy.moe_sweep_plan import make_plan
        plan = make_plan(output, scope, source_points=points)
        plan.update(cohort='model-tuning', scope='MODEL_DERIVED_DENSE_OR_INDEXED_SMALLM_ASYS')
        plan['sources']['dev/model_tuning/plans.py'] = sha(__file__)
    elif backend == 'moe-prefill':
        from dev.strategy.moe_prefill_campaign import make_plan
        audit, plan = make_plan(output, scope, source_points=points)
        atomic_json(output/'audit.json', audit)
        plan.update(cohort='model-tuning', scope='MODEL_DERIVED_INDEXED_PREFILL_ASYS_PUBLIC_HELPERS')
        keys = {tuple(p['workload'][k] for k in ('qtype','n','k','channels','m')) for p in points}
        plan['coverage'].update(host_keys=sorted(keys), keys_with_gaps=0)
    elif backend == 'dense-prefill':
        from dev.prefill_asys.m_span import make_plan
        from quactlize.dispatch.dense_prefill import EVIDENCE
        policy = json.loads(EVIDENCE.read_bytes())
        helper_sha = digest(json.loads((ROOT/'policies/dense_full_helpers_20261023.json').read_bytes()))
        plan = make_plan(policy, helper_sha, output, ms=sorted({p['workload']['m'] for p in points}),
                         compute_units=compute_units, source_points=points)
        for p in plan['points']:
            p['native_comparison'] = False
        from dev.prefill_asys.finalists import counts
        plan['counts'] = counts(plan['points'])
    else:
        raise ValueError('unknown measurement backend: '+backend)
    plan['model_tuning'] = dict(inventory=inventory['sha256'], backend=backend,
                                forms='CANONICAL_PROJECTIONS_ONLY_NOT_FUSED_EPILOGUE')
    return sealed(plan)


def validate_engine(plan, backend):
    if plan.get('sha256') != digest({k:v for k,v in plan.items() if k != 'sha256'}):
        raise ValueError('external engine plan digest differs')
    marker = plan.get('model_tuning', {})
    if not marker.get('inventory') or not plan.get('points'):
        raise ValueError('external plan lacks inventory binding or points')
    expected = ('quactlize.smallm-asys.v1' if backend == 'smallm' else
                'quactlize.moe-prefill-sweep.v1')
    if plan['schema'] != expected or not marker.get('backend','').startswith(backend):
        raise ValueError('external plan/backend differs')
    seen = set()
    for p in plan['points']:
        w = Workload(**p['workload'])
        if p['id'] != w.key or w.key in seen or w.fusion != 'none':
            raise ValueError('external plan must contain distinct canonical projections')
        seen.add(w.key)
        if (backend == 'smallm' and w.m > 8 or backend == 'moe-prefill' and
                (w.m <= 8 or w.operation != 'indexed' or w.compute != 'bf16')):
            raise ValueError('external workload outside backend contract')
        if not any(a['id'] == 'original' for a in p['arms']):
            raise ValueError('external plan lacks incumbent control')
    return plan


def prepare(inventory, output, *, compute_units=72):
    validate(inventory)
    if inventory['blocked']:
        raise ValueError('resolve missing TP shapes before preparing a sweep')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    parts = partition(inventory); jobs = []
    for backend, points in parts['groups'].items():
        directory = output/backend
        path = directory/'plan.json'
        if path.exists():
            plan = json.loads(path.read_bytes())
            if (plan.get('sha256') != digest({k:v for k,v in plan.items() if k != 'sha256'}) or
                    plan.get('model_tuning',{}).get('inventory') != inventory['sha256']):
                raise ValueError('existing engine plan belongs to another inventory')
        else:
            plan = make_engine(inventory, backend, deepcopy(points), directory, compute_units=compute_units)
            atomic_json(path, plan)
        jobs.append(dict(backend=backend, plan=str(path.resolve()), sha256=plan['sha256'],
                         points=len(points), counts=plan['counts']))
        print('MODEL_TUNING_PLAN '+json.dumps(jobs[-1]), flush=True)
    result = sealed(dict(schema=SCHEMA, inventory=inventory['sha256'], jobs=jobs,
        bindings=parts['bindings'], pending=parts['pending'],
        status='READY' if jobs and not parts['pending'] else 'INCOMPLETE',
        production_changed=False, gpu_work=False))
    atomic_json(output/'campaign.json', result)
    return result
