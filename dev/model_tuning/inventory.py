"""One GGUF/TP inventory for ordinary and fused dense/indexed projections."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import re

from dev.prefill_asys.cohort import headers, local_shapes
from dev.prefill_asys.tp_header import CAPTURE_KEYS, HeaderTP
from quactlize.dispatch.table import QTYPES, Workload
from quactlize.gguf_roles import match_role
from quactlize.runtime.tuning import digest
from tools.gguf_internal_shape_inventory import _metadata_authorities, _normalise_split

SCHEMA = 'quactlize.model-tuning-inventory.v1'
MS = (*range(1, 9), 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
FORMS = ('projection', 'concat', 'swiglu')
OPERATIONS = ('dense', 'indexed')
PAIRED = 'QKG_GATE_UP_N4_V1:0x47554e3400000001'
PAIRS = (
    ('ffn_gate', 'ffn_up', 'ffn_down'),
    ('shared_expert_gate', 'shared_expert_up', 'shared_expert_down'),
    ('moe_expert_gate', 'moe_expert_up', 'moe_expert_down'),
)


def sealed(value):
    return dict(value, sha256=digest({k: v for k, v in value.items() if k != 'sha256'}))


def validate(value):
    if value.get('schema') != SCHEMA or value.get('sha256') != digest(
            {k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('model inventory schema/digest differs')
    operations=value.get('operations',list(OPERATIONS))
    if not operations or len(set(operations))!=len(operations) or set(operations)-set(OPERATIONS):
        raise ValueError('invalid model inventory operations')
    seen = set()
    for p in value['points']:
        w = Workload(**p['workload'])
        c = p['contract']
        if p['id'] != digest(dict(workload=asdict(w), contract=c)) or p['id'] in seen:
            raise ValueError('duplicate or changed model tuning point')
        seen.add(p['id'])
        if (not p['consumers'] or c['accumulator'] != 'f32' or c['form'] not in FORMS
                or w.operation not in operations):
            raise ValueError('point has no consumer or unsupported arithmetic')
        if c['form'] == 'swiglu':
            if c['layout'] != PAIRED or c['weight_n'] != 2*w.n or c['output_n'] != w.n:
                raise ValueError('paired logical/physical width differs')
        elif c['weight_n'] != w.n or c['output_n'] != w.n or w.fusion != 'none':
            raise ValueError('ordinary GEMM extent differs')
        if c['form']=='concat' and c['logical_n']*2!=w.n:
            raise ValueError('concatenated gate/up logical extent differs')
    return value


def _point(w, form='projection', logical_n=None):
    logical_n = logical_n or w.n
    paired = form == 'swiglu'
    contract = dict(form=form, layout=PAIRED if paired else 'CANONICAL_KPACK',
        logical_n=logical_n, weight_n=2*logical_n if paired else w.n, output_n=w.n,
        accumulator='f32', input_type='f32', output_type='f32',
        round_projection=int(paired and w.operation == 'indexed'),
        epilogue='SWIGLU' if paired else 'IDENTITY', row_order='original',
        scope='GATE_UP_SWIGLU' if paired else 'PROJECTION_ONLY',
        activation_included=paired)
    identity = dict(workload=asdict(w), contract=contract)
    return dict(id=digest(identity), **identity, consumers=[])


def _refs(model, tensor, role, rank, axis, world):
    return dict(model=model['name'], tp=world, rank=rank, tensor=tensor['name'],
                role=role.name, partition=axis, global_dims=tensor['dims_gguf'])


def _fusion(local, add, excluded, ms, routers, forms):
    layers = defaultdict(dict)
    for w, ref in local:
        layer = ref['tensor'].split('.ffn_', 1)[0]
        layers[(ref['model'], ref['rank'], layer)][ref['role']] = (w, ref)
    for key, entries in sorted(layers.items()):
        for gate_role, up_role, down_role in PAIRS:
            gate, up = entries.get(gate_role), entries.get(up_role)
            merged = entries.get('moe_expert_gate_up') if gate_role == 'moe_expert_gate' else None
            if not (gate or up or merged):
                continue
            if merged and (gate or up):
                raise ValueError('both merged and independent gate/up in '+str(key))
            if merged:
                physical, ref = merged
                if ref['partition']=='n' and not ref.get('paired_n_segments',False):
                    excluded.append(dict(layer=list(key),reason='MERGED_TP_N_PAIRING_NOT_PROVEN'))
                    continue
                if physical.n % 2:
                    excluded.append(dict(layer=list(key), reason='MERGED_GATE_UP_ODD_N'))
                    continue
                try:
                    logical = replace(physical, n=physical.n//2, output_stride=0)
                except ValueError as exc:
                    excluded.append(dict(layer=list(key), reason='PAIRED_LOGICAL_SHAPE: '+str(exc)))
                    continue
                refs = [ref]
            else:
                if not gate or not up or gate[0] != up[0] or gate[1]['partition'] != up[1]['partition']:
                    excluded.append(dict(layer=list(key), reason='GATE_UP_TYPE_SHAPE_OR_PARTITION_MISMATCH'))
                    continue
                logical = gate[0]
                refs = [gate[1], up[1]]
                physical = replace(logical, n=2*logical.n, output_stride=0)
            down = entries.get(down_role)
            consumer = dict(refs[0], tensors=[r['tensor'] for r in refs],
                            down=(asdict(down[0]) if down else None))
            selected_forms=forms
            if refs[0]['partition']=='k' and 'swiglu' in forms:
                # Nonlinear epilogues cannot precede the TP sum of partial
                # gate/up dots. Ordinary 2N partial projections remain legal.
                excluded.append(dict(layer=list(key),reason='SWIGLU_REQUIRES_ALLREDUCE_BEFORE_EPILOGUE'))
                selected_forms=[f for f in forms if f!='swiglu']
            for m in ms:
                for router in routers if logical.operation == 'indexed' else ('',):
                    for form in selected_forms:
                        if form == 'projection':
                            continue
                        base = logical if form == 'swiglu' else physical
                        w = replace(base, m=m, router=router,
                                    fusion='gate-up-swiglu' if form == 'swiglu' else 'none')
                        add(_point(w, form, logical.n), consumer)


def collect(models, *, ms=MS, forms=FORMS, routers=('spread', 'cluster'), operations=OPERATIONS,
            loader=None, partitioner=None):
    """Loader/partitioner injection permits replay of uploaded header proofs."""
    if (not models or any(not isinstance(m.get('name'), str) or not m['name'] for m in models)
            or len({m['name'] for m in models}) != len(models)):
        raise ValueError('nonempty unique model names required')
    if not ms or len(set(ms)) != len(ms) or any(type(m) is not int or not 1 <= m <= 32768 for m in ms):
        raise ValueError('distinct M in 1..32768 required')
    if not forms or len(set(forms)) != len(forms) or set(forms)-set(FORMS):
        raise ValueError('choose distinct projection,concat,swiglu forms')
    if not operations or len(set(operations))!=len(operations) or set(operations)-set(OPERATIONS):
        raise ValueError('choose distinct dense,indexed operations')
    if not routers or len(set(routers)) != len(routers) or set(routers)-{'spread', 'cluster', 'repeat2', 'repeat4', 'real'}:
        raise ValueError('unknown or duplicate routing profile')
    loader = loader or (lambda path: headers(path, capture_keys=CAPTURE_KEYS))
    points, ignored, excluded, blocked, identities = {}, [], [], [], []

    def add(p, ref):
        row = points.setdefault(p['id'], p)
        if ref not in row['consumers']:
            row['consumers'].append(ref)

    for model in models:
        world = model.get('tp', 1)
        if type(world) is not int or not 1 <= world <= 64:
            raise ValueError('TP must be in 1..64')
        rows, _ = loader(model['path'])
        rows, _ = _normalise_split(rows, model['name'])
        _, arch, metadata = _metadata_authorities(rows, model['name'])
        tensors = [t for row in rows for t in row['header']['tensors']]
        if len({t['name'] for t in tensors}) != len(tensors):
            raise ValueError('duplicate tensor across model shards')
        identity = dict(name=model['name'], tp=world, architecture=arch,
                        headers_sha256=digest(rows), files=[r['path'] for r in rows])
        records, omissions = {}, {}
        proof={}
        if world > 1:
            if partitioner is None:
                raise ValueError('TP requires caller-derived local extents; no guessed N/K division')
            records, proof, omissions = partitioner(model, rows, tensors)
            identity['partition'] = proof
        identities.append(identity)
        local = []
        for t in tensors:
            name, dims, q = t['name'], t['dims_gguf'], t['qtype']
            ref = dict(model=model['name'], tensor=name, qtype=q, dims=dims)
            if len(dims)<2:
                # Shared-expert scalar gates can be stored as 1-D even
                # though their graph consumer is MUL_MAT. N=1 is not a
                # supported K-pack matrix; retain native placement.
                ignored.append(dict(ref,reason='NOT_KPACK_MATRIX'));continue
            role = match_role(name, len(dims))
            if name in omissions:
                excluded.append(dict(ref, reason=omissions[name])); continue
            if role is None:
                (excluded if len(dims) >= 2 else ignored).append(dict(ref, reason='UNKNOWN_ROLE' if len(dims) >= 2 else 'NOT_MATRIX'))
                continue
            role = role[0]
            if role.operation not in ('MUL_MAT', 'MUL_MAT_ID'):
                ignored.append(dict(ref, reason=role.operation)); continue
            indexed = role.operation == 'MUL_MAT_ID'
            if ('indexed' if indexed else 'dense') not in operations:
                ignored.append(dict(ref,reason='OPERATION_NOT_SELECTED'));continue
            if q not in QTYPES:
                excluded.append(dict(ref, reason='UNSUPPORTED_DTYPE')); continue
            experts, topk = (dims[2], metadata.get(arch+'.expert_used_count')) if indexed else (1, 1)
            if indexed and (type(topk) is not int or not 1 <= topk <= experts
                            or metadata.get(arch+'.expert_count') != experts):
                raise ValueError('invalid expert metadata: '+name)
            matrix = dict(t, dims_gguf=dims[:2])
            shapes, error = local_shapes(matrix, world, records)
            if error:
                blocked.append(dict(ref, reason=error)); continue
            for rank, n, k, axis in shapes:
                try:
                    w = Workload(q, n, k, 1, 'indexed' if indexed else 'dense',
                        'bf16' if indexed else 'f16', experts=experts, topk=topk,
                        channels=topk if role.name == 'moe_expert_down' else 1)
                except ValueError as exc:
                    excluded.append(dict(ref, rank=rank, reason='UNSUPPORTED_LOCAL_SHAPE: '+str(exc))); continue
                consumer = _refs(model, t, role, rank, axis, world)
                if role.name=='moe_expert_gate_up' and axis=='n':
                    splits=[r for p in proof.values() if isinstance(p,dict) for r in p.get('splits',[])
                            if r['name']==name]
                    consumer['paired_n_segments']=bool(splits and all(
                        len(r['segments'])==1 and r['segments'][0]['repeat']==2 for r in splits))
                local.append((w, consumer))
                if 'projection' in forms:
                    for m in ms:
                        for router in routers if indexed else ('',):
                            add(_point(replace(w, m=m, router=router)), consumer)
        _fusion(local, add, excluded, ms, routers, forms)
    values = sorted(points.values(), key=lambda p: (p['workload']['operation'], p['id']))
    counts = Counter((p['workload']['operation'], p['contract']['form']) for p in values)
    result = dict(schema=SCHEMA, models=identities, ms=sorted(ms), forms=list(forms), routers=list(routers),
        operations=list(operations),
        points=values, ignored=ignored, excluded=excluded, blocked=blocked,
        counts=[dict(operation=op, form=form, points=n) for (op, form), n in sorted(counts.items())],
        status='READY' if values and not blocked else 'INCOMPLETE', weights_read=False, gpu_work=False,
        scope='MODEL_HEADERS_AND_TP_LOCAL_SHAPES_NOT_GRAPH_FUSION_ADMISSION',
        production_changed=False, performance_admitted=False)
    return validate(sealed(result))


class Splitter:
    """Use the same caller TP implementation for dense and indexed operands."""
    def __init__(self, llama_source, output):
        self.source, self.output, self.adapters = Path(llama_source), Path(output), {}

    def __call__(self, model, rows, tensors):
        result, proofs, omitted = {}, {}, {}
        for indexed in (False, True):
            if indexed not in self.adapters:
                self.adapters[indexed] = HeaderTP(self.source, self.output/('moe' if indexed else 'dense'), indexed=indexed)
            records, proof, exclusions = self.adapters[indexed](model, rows, tensors)
            for name, value in records.items():
                if name in result and result[name] != value:
                    raise ValueError('dense/indexed TP records disagree: '+name)
                result[name] = value
            omitted.update(exclusions)
            proofs['indexed' if indexed else 'dense'] = proof
        return result, proofs, omitted
