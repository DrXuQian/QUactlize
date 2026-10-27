"""Fusion is an execution/layout axis, not a dense N/K timing alias."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from quactlize.dispatch.table import Workload
from quactlize.runtime.tuning import digest

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = 'QKG_GATE_UP_N4_V1:0x47554e3400000001'


def variants(points):
    layers = {}
    for point in points:
        w = Workload(**point['workload'])
        for c in point['consumers']:
            layer = c['tensor'].split('.ffn_', 1)[0]
            key = (c['model'], c['rank'], layer, w.m)
            layers.setdefault(key, {})[c['role']] = (w, c, point['id'])
    fused, unavailable = {}, []
    for key, entries in sorted(layers.items()):
        merged = entries.get('moe_expert_gate_up')
        if merged:
            if 'moe_expert_gate' in entries or 'moe_expert_up' in entries:
                raise ValueError('ambiguous merged and separate gate/up weights in one layer')
            physical, consumer, _ = merged
            if physical.n % 2:
                raise ValueError('merged gate/up has odd physical N')
            try:
                logical = replace(physical, n=physical.n//2, output_stride=0)
            except ValueError:
                unavailable.append(dict(layer=list(key), reason='PAIRED_LOGICAL_N_OUTSIDE_FORMAT_CONTRACT'))
                continue
            refs = [merged]
        else:
            gate, up = entries.get('moe_expert_gate'), entries.get('moe_expert_up')
            if gate is None and up is None:
                continue
            if gate is None or up is None or gate[0] != up[0]:
                unavailable.append(dict(layer=list(key), reason='GATE_UP_TYPE_SHAPE_OR_PARTITION_MISMATCH'))
                continue
            logical, consumer, _ = gate
            physical = replace(logical, n=2*logical.n, output_stride=0)
            refs = [gate, up]
        down = entries.get('moe_expert_down')
        down_shape = (dict(qtype=down[0].qtype, n=down[0].n, k=down[0].k,
                           channels=down[0].channels) if down else None)
        # Canonical concatenation and paired epilogues are distinct artifacts.
        # Both retain the original projection width for the subsequent down.
        forms = [dict(fusion='gate-up', layout='CANONICAL_GATE_THEN_UP',
                      output_n=physical.n, output_type='f32', row_order='original',
                      round_projection=0, activation='SEPARATE_BF16_ROUNDED_SWIGLU'),
                 *[dict(fusion='gate-up-swiglu', layout=LAYOUT,
                        output_n=logical.n, output_type=output, row_order=order,
                        round_projection=1, activation='FUSED_SWIGLU')
                   for output, order in (('f32', 'original'), ('bf16', 'compact'))]]
        for form in forms:
            w = replace(physical if form['fusion']=='gate-up' else logical, fusion=form['fusion'])
            contract = dict(form, logical_n=logical.n, weight_n=physical.n,
                            compute='bf16', accumulator='f32', down=down_shape)
            identity = dict(workload=asdict(w), contract=contract)
            fid = digest(identity)
            point = fused.setdefault(fid, dict(id=fid, **identity, consumers=[], source_points=[],
                candidate_generation='BLIND_RULES_ON_PHYSICAL_N_PLUS_SEPARATE_FUSED_EPILOGUE_CAPABILITY',
                timing_reuse=False, native_comparison='MATCHED_GATE_UP_ACTIVATION_BOUNDARY',
                sweep_required=True, execution_ready=False))
            point['consumers'].append(dict(model=consumer['model'], rank=consumer['rank'],
                                            tensors=[r[1]['tensor'] for r in refs]))
            for ref in refs:
                if ref[2] not in point['source_points']:
                    point['source_points'].append(ref[2])
    return dict(points=sorted(fused.values(), key=lambda p: p['id']), unavailable=unavailable,
                pending=['FUSED_RECIPE_RESOURCE_AND_ASYS_CARRIER',
                         'INDEXED_TC_TOKEN_GROUPING_FOR_WEIGHT_REUSE',
                         'SF_FULL_BF16_AND_FUSED_COMPLETION_COST_COMPARISON'],
                dense_timings_reusable=False, production_changed=False)


def paired_recipes(request):
    """Public API geometry floor + current reader template families.

    Do NOT query the historical selector. Fixed N/K are target template
    arguments, not lookup keys copied from a winning row. Rebinding those
    bodies requires a new compile/numeric gate; it is not current capability.
    """
    q, n, k = (request[x] for x in ('qtype', 'n', 'k'))
    tk = 256 if q in (11, 13) else 128 if q in (10, 14) else 64
    out = []
    for split in (1, 2, 4, 8):
        if k % (tk*split):
            continue
        base = dict(split=split, reader=0, variant=5 if q==8 else 3, columns=4,
                    values=8, changes=0, hoist=0, fixed_n=0, fixed_k=0)
        if request['tokens'] <= 8:
            out.extend(dict(base, backend=0, tile_m=0, warps=w) for w in (4, 8))
        out.extend(dict(base, backend=1, tile_m=m, warps=0) for m in (8, 16))
    if (q==12 and request['experts']==256 and request['topk']==8 and
            request['channels']==1 and request['compute_type']==1 and
            request['input_type']==1 and request['output_type']==1 and
            request['round_projection']==1 and request['tokens']<=8):
        base = dict(backend=0, split=1, tile_m=0, variant=3, columns=4, hoist=0,
                    fixed_n=2*n, fixed_k=k)
        out.append(dict(base, warps=8, reader=2, values=8, changes=1))
        if request['row_order']==0:
            out.append(dict(base, warps=4, reader=3, values=4, changes=3))
    return out


def historical_recall():
    path = ROOT/'policies/kpack_gate_up_selection_v2.json'
    data = json.loads(path.read_text())
    recipes = {r['name']: {k:v for k,v in r.items() if k not in ('name', 'id')}
               for r in data['recipes']}
    points = []
    for row in data['rows']:
        req = row['request']
        if req['mode'] != 2 or req['compute_type'] != 1:
            continue
        candidates = paired_recipes(req)  # Reference revealed only afterwards.
        reference = recipes[row['recipe']]
        points.append(dict(request=req, reference=reference, candidates=candidates,
                           status='PASS' if reference in candidates else 'RULE_MISS'))
    return dict(status='PASS' if points and all(p['status']=='PASS' for p in points) else 'RULE_GAPS',
                points=points, source=str(path.relative_to(ROOT)), source_sha256=digest(data),
                scope='FUSION_CONFIG_RECALL_NOT_COMPILED_CAPABILITY_OR_PERFORMANCE',
                champion_injection=False, old_latency_rows_imported=0,
                kernel_launches=0, performance_admitted=False)
