"""Large-M component closure, with blind rules and a separate live control.

Families, not individual candidates, are sharded. Every comparison for a
weight stays on one physical card. No timing is imported from another card.
"""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from dev.strategy.journal import planned_arms
from dev.strategy.resources import module_spec, probe_spec
from dev.strategy.space import expansion_configs, resolved_grid
from dev.strategy.prefill_selection import control_helper_config
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.export_kpack_strategy import recipe as normalize

ROOT = Path(__file__).resolve().parents[2]


def family(point):
    w = point['workload']
    return digest([w[k] for k in ('qtype', 'n', 'k', 'experts', 'operation', 'compute')])


def partition(points, shard, shards):
    if type(shards) is not int or type(shard) is not int or not 0 <= shard < shards <= 256:
        raise ValueError('require 0 <= shard < shards <= 256')
    keys = sorted({family(p) for p in points})
    owned = {key for i, key in enumerate(keys) if i % shards == shard}
    return [p for p in points if family(p) in owned]


def bind_original(points, output):
    output = Path(output)
    executable = output / 'prefill-original'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT),
                    str(ROOT/'dev/strategy/prefill_original.cpp'), '-o', str(executable)], check=True)
    requests = [[w.qtype, w.n, w.k, w.m, w.experts, w.topk, int(w.compute == 'bf16')]
                for p in points for w in [Workload(**p['workload'])]]
    rows = [json.loads(line) for line in subprocess.check_output([str(executable)], text=True,
            input=''.join(' '.join(map(str, r))+'\n' for r in requests)).splitlines()]
    if [r['request'] for r in rows] != requests:
        raise ValueError('original prefill selector omitted/reordered requests')
    for point, row in zip(points, rows):
        point['original_selector'] = row
        if row['status']:
            point['original_missing'] = 'PRODUCTION_PREFILL_SELECTOR_MISS'
            continue
        if row['route'] == 2:
            r = Recipe.make('full', 'full', dequant_config=row['dequant_config'], provider='bf16-default')
        else:
            r = normalize(dict(row['parent'], kind='tc', split=row['split'],
                grid_mode=row['grid_mode'], grid_b=row['grid_b']), row['route'], row['dequant_config'])
        point['original_recipe'] = asdict(r)
    return points


def full_configs(w):
    # Indexed full expansion preserves the original E-axis strides and only
    # writes GPU-selected experts. Do not substitute all-E scalar kernels.
    allowed = expansion_configs(w, 'full')
    if w.operation == 'indexed':
        allowed = tuple(c for c in allowed if c in (4, 5, 10, 11))
    return list(allowed)


def execution_plan(probe):
    points, blocked = [], []
    for p in probe['resolved']['points']:
        w = Workload(**p['workload'])
        if w.m <= 8 or w.operation not in ('dense', 'indexed') or w.storage != 'f32':
            raise ValueError('prefill experiment accepts large-M F32 dense/indexed workloads only')
        if w.operation == 'indexed' and (w.experts != 256 or w.topk != 8 or w.compute != 'bf16'):
            raise ValueError('installed grouped provider needs E256/top8/BF16')
        if not p.get('original_recipe'):
            blocked.append(dict(id=p['id'], reason=p.get('original_missing', 'MISSING_ORIGINAL')))
            continue
        arms, issues = [], []
        for arm in planned_arms(p):
            r = Recipe.from_dict(arm['recipe'])
            item = dict(arm)
            if r.kind == 'full':
                if not full_configs(w):
                    issues.append(dict(arm=arm['id'], reason='FULL_DEQUANT_UNSUPPORTED'))
                arms.append(item)
                continue
            if r.kind != 'tc':
                raise ValueError('large-M plan contains a decode-only recipe')
            spec = module_spec(w, r)
            qid = digest(probe_spec(w, r, int(dict(r.parameters)['grid_mode'] >= 2)))
            query = probe['receipts'].get(qid)
            if not query or query['status'] != 'READY':
                issues.append(dict(arm=arm['id'], roles=arm['roles'], reason='REQUIRED_RESOURCE_UNAVAILABLE'))
                continue
            item.update(module=digest(spec), spec=spec, binding=dict(query=qid,
                launch=resolved_grid(w, r, query['device']['compute_units'], query['occupancy'])))
            arms.append(item)
        if issues:
            blocked.append(dict(id=p['id'], reason='REQUIRED_COMPARISON_UNAVAILABLE', issues=issues))
            continue
        sf = sorted({max(0, c) for c in expansion_configs(w, 'sf')}) if any(
            a['recipe']['route'] == 'sf' for a in arms) else []
        full = full_configs(w) if any(a['recipe']['kind'] == 'full' for a in arms) else []
        # A historical full-dequant config is a separate control. Sparse
        # unsupported configs must remain explicit; never time all E instead.
        for a in arms:
            r = Recipe.from_dict(a['recipe'])
            c = dict(r.parameters).get('dequant_config', -1)
            if r.kind == 'full' and ('original' in a['roles'] or 'control' in a['roles']) and c not in full:
                issues.append(dict(arm=a['id'], reason='CONTROL_ACTIVE_DEQUANT_CONFIG_UNAVAILABLE', config=c))
            if (r.route == 'sf' and w.qtype != 8 and 'rule' not in a['roles'] and
                    control_helper_config(w, r) not in sf):
                issues.append(dict(arm=a['id'], reason='CONTROL_TYPED_SF_CONFIG_UNAVAILABLE',
                                   config=control_helper_config(w, r)))
        if issues:
            blocked.append(dict(id=p['id'], reason='CONTROL_HELPER_UNAVAILABLE', issues=issues))
            continue
        points.append(dict(p, arms=arms, sf_configs=sf, full_configs=full))
    result = dict(schema='quactlize.rule-prefill-components.v1', points=points, blocked=blocked,
        counts=dict(points=len(points), requested_points=len(points)+len(blocked), blocked_points=len(blocked),
                    arms=sum(len(p['arms']) for p in points)),
        precision='DENSE_F16_TC_MOE_BF16_TC_FULL_BF16_PROVIDER_F32_STORAGE_FP32_ACCUM',
        scope='RESIDENT_ROUTE_MAPS_SEPARATE_ADAPTERS_DEQUANT_PRODUCER_REDUCER_AND_DIRECTORY',
        excluded=['TOPK_AND_CREATION_OF_RESIDENT_ROUTING_MAPS', 'ORIGINAL_WIRING_MODEL_AB'],
        production_changed=False, performance_admitted=False)
    result['plan_sha256'] = digest(result)
    return json.loads(json.dumps(result))
