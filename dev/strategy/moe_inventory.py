"""Actual MoE weights and CPU-derived TP-local extents, without model loading."""
from dataclasses import asdict

from dev.prefill_asys.cohort import local_shapes
from dev.prefill_asys.tp_header import read_headers
from dev.strategy.moe_plan import MS
from quactlize.dispatch.table import QTYPES, Workload
from quactlize.gguf_roles import match_role
from quactlize.runtime.tuning import digest
from tools.gguf_internal_shape_inventory import _metadata_authorities, _normalise_split


def collect(models, *, loader=read_headers, partitioner=None, ms=MS):
    if not models or len({m['name'] for m in models}) != len(models):
        raise ValueError('nonempty unique model names required')
    if not ms or len(set(ms)) != len(ms) or any(type(m) is not int or m <= 0 for m in ms):
        raise ValueError('distinct positive token counts required')
    points, excluded, blocked, identities = {}, [], [], []
    for model in models:
        world = model.get('tp', 1)
        if type(world) is not int or world <= 0:
            raise ValueError('positive TP size required')
        rows, _ = loader(model['path'])
        rows, _ = _normalise_split(rows, model['name'])
        _, arch, metadata = _metadata_authorities(rows, model['name'])
        tensors = [t for row in rows for t in row['header']['tensors']]
        if len({t['name'] for t in tensors}) != len(tensors):
            raise ValueError('duplicate tensor across model headers')
        targets = []
        for t in tensors:
            role = match_role(t['name'], len(t['dims_gguf'])) if 'exps' in t['name'] else None
            if role and role[0].operation == 'MUL_MAT_ID':
                targets.append((t, role[0]))
            elif len(t['dims_gguf']) == 3 and 'exps' in t['name']:
                blocked.append(dict(model=model['name'], tensor=t['name'], reason='UNKNOWN_EXPERT_OPERATOR'))
        identity = dict(name=model['name'], tp=world, headers_sha256=digest(rows),
                        paths=[r['path'] for r in rows], expert_weights=len(targets))
        identities.append(identity)
        if not targets:
            identity['status'] = 'NO_MOE_OPERATORS'
            continue
        experts, topk = metadata.get(arch+'.expert_count'), metadata.get(arch+'.expert_used_count')
        if any(type(v) is not int or v <= 0 for v in (experts, topk)) or topk > experts:
            raise ValueError('missing/invalid GGUF expert_count or expert_used_count')
        records, partition_exclusions = {}, {}
        if partitioner is not None:
            records, proof, partition_exclusions = partitioner(model, rows, tensors)
            if proof.get('identity', {}).get('operation') != 'MUL_MAT_ID':
                raise ValueError('TP splitter did not bind indexed targets')
            identity['partition'] = proof
        for tensor, role in targets:
            k, n, e = tensor['dims_gguf']
            ref = dict(model=model['name'], tensor=tensor['name'], qtype=tensor['qtype'],
                       role=role.name, tp=world, global_dims=[k, n, e])
            if e != experts:
                raise ValueError('expert tensor extent disagrees with GGUF metadata')
            if tensor['name'] in partition_exclusions or tensor['qtype'] not in QTYPES:
                excluded.append(dict(ref, reason=partition_exclusions.get(tensor['name'], 'UNSUPPORTED_QUANT_OR_FLOAT')))
                continue
            matrix = dict(tensor, dims_gguf=[k, n])
            local, error = local_shapes(matrix, world, records)
            if error:
                blocked.append(dict(ref, reason=error))
                continue
            channels = topk if role.name == 'moe_expert_down' else 1
            for rank, ln, lk, axis in local:
                for tokens in ms:
                    try:
                        w = Workload(tensor['qtype'], ln, lk, tokens, 'indexed', 'bf16',
                                     experts=experts, topk=topk, channels=channels)
                    except ValueError as exc:
                        excluded.append(dict(ref, rank=rank, tokens=tokens, n=ln, k=lk,
                                             reason='NATIVE_ONLY_SHAPE: '+str(exc)))
                        continue
                    point = points.setdefault(w.key, dict(id=w.key, workload=asdict(w), consumers=[]))
                    point['consumers'].append(dict(ref, rank=rank, partition=axis))
    from dev.strategy.moe_fusion import variants
    return dict(schema='quactlize.moe-model-inventory.v1', models=identities,
                points=sorted(points.values(), key=lambda p: p['id']), excluded=excluded, blocked=blocked,
                fusion=variants(points.values()),
                tokens=list(ms), status='INCOMPLETE' if blocked else 'PASS',
                scope='MODEL_HEADERS_AND_TP_LOCAL_SHAPES_NOT_MEASURED_CALL_COUNTS',
                router_data='PENDING_ACTUAL_IDS_CAPTURE_OR_EXPLICIT_SYNTHETIC_PROFILE',
                expert_parallelism=False, weights_read=False, gpu_work=False,
                production_changed=False, performance_admitted=False)


def add_candidates(inventory, table, options):
    """Plan every execution form; do not treat fusion as a dense timing alias."""
    from dataclasses import replace
    from dev.strategy.moe_plan import moe_candidates, proposal_only
    from dev.strategy.moe_fusion import paired_recipes
    from dev.strategy.prescreen import Prescreen
    screen = Prescreen(proposal_only(table.observations))
    definitions, cache = {}, {}
    for point in inventory['points']:
        w = Workload(**point['workload'])
        selected, _ = moe_candidates(screen, w, options)
        point['candidates'] = [r.key for r in selected]
        for r in selected:
            definitions[r.key] = asdict(r)
    for point in inventory['fusion']['points']:
        w, contract = Workload(**point['workload']), point['contract']
        physical = replace(w, n=contract['weight_n'], output_stride=0, fusion='none')
        key = (physical.key, contract['logical_n'])
        if key not in cache:
            # Hold out both the original projection and its merged weight.
            withheld = {(contract['logical_n'], w.k), (contract['weight_n'], w.k)}
            blind = Prescreen(proposal_only(r for r in table.observations
                                            if (r.workload.n, r.workload.k) not in withheld))
            cache[key], _ = moe_candidates(blind, physical, options)
        selected = cache[key]
        point['main_template_candidates'] = [r.key for r in selected]
        for r in selected:
            definitions[r.key] = asdict(r)
        if contract['fusion'] == 'gate-up-swiglu':
            request = dict(qtype=w.qtype, n=contract['logical_n'], k=w.k, experts=w.experts,
                           tokens=w.m, mode=2, topk=w.topk, channels=w.channels,
                           input_type=1, compute_type=1,
                           output_type=1 if contract['output_type']=='f32' else 2,
                           round_projection=contract['round_projection'],
                           row_order=int(contract['row_order']=='compact'))
            point['paired_api_candidates'] = paired_recipes(request)
            point['template_status'] = 'PAIRED_REGISTER_EPILOGUE_AND_REDUCER_CAPABILITY_GATE_REQUIRED'
        else:
            point['paired_api_candidates'] = []
            point['template_status'] = 'CANONICAL_2N_PROJECTION_PLUS_SEPARATE_ACTIVATION_GATE_REQUIRED'
    inventory['configurations'] = definitions
    inventory['counts'] = dict(unfused_points=len(inventory['points']),
                              fused_points=len(inventory['fusion']['points']),
                              distinct_template_recipes=len(definitions),
                              ready_to_time=0, resource_expansion='PENDING')
    inventory['sweep_ready'] = False
    return inventory
