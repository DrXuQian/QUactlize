"""Dense prefill inventory and bounded proposals, without GPU work.

GGUF dimensions are not TP dimensions. Multi-rank workloads require actual
caller records; unknown partitions and unsupported endpoints stay visible.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re

from quactlize.dispatch.table import Recipe, Workload, QTYPES
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.gguf_internal_shape_inventory import read_gguf_header, _normalise_split, _metadata_authorities

SCHEMA = 'quactlize.dense-prefill-cohort.v1'
MODEL_PATHS = {
    'qwen3-32b': ('Qwen3-32B-Q4_K_M-GGUF/Qwen3-32B-Q4_K_M.gguf', 1),
    'qwen35-35b-a3b': ('Qwen3.5-35B-A3B-Q4_K_M-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf', 1),
    'qwen38-27b': ('Qwen3.8-27B-Q4_K_M-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf', 1),
    'deepseek4-flash-tp4': ('DeepSeek-V4-Flash-0731-UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf', 4),
}
TP_LINE = re.compile(r'\[quactlize-dense\] tensor=(.*?) device=(\d+) q=(\d+) m=(\d+) n=(\d+) k=(\d+)(?: |$)')


def bucket(w):
    # Classification precedes timing; exact N/K remain the authoritative key.
    return dict(operation=w.operation, qtype=w.qtype, compute=w.compute,
                storage=w.storage, m_band=w.m.bit_length()-1,
                n_band=w.n.bit_length()-1, k_band=w.k.bit_length()-1,
                aspect='wide-n' if w.n >= 2*w.k else 'wide-k' if w.k >= 2*w.n else 'balanced')


def headers(path):
    path = Path(path).resolve(strict=True)
    match = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', path.name)
    if match:
        if int(match[2]) != 1:
            raise ValueError('bind the first GGUF shard: '+str(path))
        files = [path.with_name(f'{match[1]}-{i:05d}-of-{match[3]}.gguf')
                 for i in range(1, int(match[3])+1)]
    else:
        files = [path]
    rows = []
    for file in files:
        with file.open('rb') as stream:
            header = read_gguf_header(stream, str(file))
        rows.append(dict(path=str(file), size=file.stat().st_size, header=header))
    rows, _ = _normalise_split(rows, path.name)
    _metadata_authorities(rows, path.name)
    tensors = [t for r in rows for t in r['header']['tensors']]
    if len({t['name'] for t in tensors}) != len(tensors):
        raise ValueError('duplicate GGUF tensor across shards')
    return rows, tensors


def tp_records(paths, world):
    records = defaultdict(set)
    receipts = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        receipts.append(dict(path=str(path), sha256=sha(path)))
        for line in path.read_text(errors='replace').splitlines():
            match = TP_LINE.search(line)
            if not match:
                continue
            name, rank, q, m, n, k = match.groups()
            rank, q, m, n, k = map(int, (rank, q, m, n, k))
            if rank >= world or min(m, n, k) < 1:
                raise ValueError('TP log contains an invalid local rank or shape')
            name = re.sub(r'(?: \(view\))+$', '', name)
            records[name].add((rank, q, n, k))
    return records, receipts


def local_shapes(tensor, world, records):
    k, n = tensor['dims_gguf']
    if world == 1:
        return [(0, n, k, 'single')], None
    found = records.get(tensor['name'], set())
    if {r[0] for r in found} != set(range(world)):
        return [], 'TP_LOCAL_RECORDS_MISSING'
    if len(found) != world or any(q != tensor['qtype'] for _, q, _, _ in found):
        return [], 'TP_LOCAL_RECORDS_AMBIGUOUS'
    found = sorted(found)
    # Validate observed extents, not an assumed "divide every dimension" rule.
    if all((ln, lk) == (n, k) for _, _, ln, lk in found):
        axis = 'replicated'
    elif all(lk == k for _, _, _, lk in found) and sum(ln for _, _, ln, _ in found) == n:
        axis = 'n'
    elif all(ln == n for _, _, ln, _ in found) and sum(lk for _, _, _, lk in found) == k:
        axis = 'k'
    else:
        return [], 'TP_LOCAL_EXTENTS_REQUIRE_REVIEW'
    return [(r, ln, lk, axis) for r, _, ln, lk in found], None


def collect(models, ms, *, loader=headers):
    if not ms or len(set(ms)) != len(ms) or any(type(m) is not int or not 128 <= m <= 4096 for m in ms):
        raise ValueError('dense prefill M must be distinct integers in 128..4096')
    if not models or len({m['name'] for m in models}) != len(models):
        raise ValueError('model names must be nonempty and unique')
    groups, excluded, blocked, identities = {}, [], [], []
    for model in models:
        world = model.get('tp', 1)
        if type(world) is not int or world < 1:
            raise ValueError('positive TP world required')
        rows, tensors = loader(model['path'])
        records, logs = tp_records(model.get('tp_logs', []), world)
        identities.append(dict(name=model['name'], tp=world, files=rows, tp_logs=logs,
                               weights_hashed=False, identity_scope='HEADERS_AND_EXTENTS_ONLY'))
        for t in tensors:
            ref = dict(model=model['name'], tensor=t['name'], tp=world, qtype=t['qtype'], dims=t['dims_gguf'])
            dims = t['dims_gguf']
            if len(dims) != 2 or not t['name'].startswith('blk.') or not t['name'].endswith('.weight'):
                excluded.append(dict(ref, reason='MOE_SEPARATE_COHORT' if len(dims) == 3 and 'exps' in t['name']
                                     else 'NOT_A_LAYER_DENSE_MATRIX'))
                continue
            if t['qtype'] not in QTYPES:
                excluded.append(dict(ref, reason='NON_KPACK_QUANT_OR_FLOAT'))
                continue
            local, error = local_shapes(t, world, records)
            if error:
                blocked.append(dict(ref, reason=error))
                continue
            for rank, n, k, axis in local:
                for m in ms:
                    try:
                        w = Workload(t['qtype'], n, k, m)
                    except ValueError as exc:
                        # This is also the production packing shape contract.
                        # Native-only narrow matrices must not be called a
                        # missing K-pack timing, nor disappear from coverage.
                        excluded.append(dict(ref, rank=rank, m=m, n=n, k=k,
                                             reason='NATIVE_ONLY_KPACK_SHAPE_CONTRACT: '+str(exc)))
                        continue
                    point = groups.setdefault(w.key, dict(id=w.key, workload=asdict(w),
                        bucket=bucket(w), consumers=[], tensors=[], call_count=None))
                    consumer = dict(ref, rank=rank, partition=axis)
                    if consumer not in point['consumers']:
                        point['consumers'].append(consumer)
                        point['tensors'].append(f"{model['name']}:rank{rank}:{t['name']}")
    result = dict(schema=SCHEMA, points=sorted(groups.values(), key=lambda p:p['id']),
        models=identities, ms=ms, excluded=excluded, blocked=blocked,
        status='READY' if groups and not blocked else 'INCOMPLETE',
        scope='DENSE_LAYER_MATRIX_MICROBENCH_NOT_MODEL_CALL_COUNTS_OR_TP_COLLECTIVES',
        precision='F32_STORAGE_F16_OPERANDS_FP32_ACCUMULATION',
        tp_scope='OBSERVED_LOCAL_DENSE_EXTENTS; NOT_TP_CORRECTNESS_OR_ALLREDUCE_TIMING',
        production_changed=False, performance_admitted=False)
    result['sha256'] = digest(result)
    return result


def validate(document):
    if document.get('schema') != SCHEMA or document.get('sha256') != digest(
            {k:v for k,v in document.items() if k != 'sha256'}):
        raise ValueError('cohort schema/digest differs')
    if document.get('status') != 'READY' or document.get('blocked'):
        raise ValueError('cohort has unresolved shapes; inspect blocked, do not silently skip TP ranks')
    for point in document['points']:
        w = Workload(**point['workload'])
        if w.key != point['id'] or w.operation != 'dense' or w.compute != 'f16' or not 128 <= w.m <= 4096:
            raise ValueError('cohort workload differs from dense prefill contract')
    return document


def partition(points, shard, shards):
    """Deterministic work-weighted family assignment, not a latency estimate."""
    if not 0 <= shard < shards <= 64:
        raise ValueError('invalid shard')
    families = defaultdict(list)
    for point in points:
        w = point['workload']
        families[digest([w[k] for k in ('qtype','n','k','compute','operation')])].append(point)
    if len(families) < shards: raise ValueError('more GPUs than weight families')
    def cost(rows):
        return sum((len(p['candidates'])+len(p['champion_controls'])+len(p['helper_candidates'])+4)*
                   (1+p['workload']['n']*p['workload']['k']*p['workload']['m']/1e10) for p in rows)
    loads = [0.0]*shards; owned = set()
    for key, rows in sorted(families.items(), key=lambda item:(-cost(item[1]),item[0])):
        index = min(range(shards), key=lambda i:(loads[i],i))
        loads[index] += cost(rows)
        if index == shard: owned.add(key)
    return [p for key in sorted(owned) for p in families[key]]


def freeze(document, output, max_cases=96):
    from dev.strategy.prescreen import Prescreen, rejection
    from dev.strategy.prefill_plan import bind_original
    from dev.strategy.resources import inventory
    from tools.export_kpack_strategy import load_sources
    validate(document)
    table, sources = load_sources()
    screen = Prescreen(table)
    points = []
    for value in document['points']:
        w = Workload(**value['workload'])
        candidates, report = screen.select(w, limit=32, optional=2, blind=True,
            per_stratum=1, factorized=False, stencil='none')
        # Observed exact selections are controls, never smuggled into blind rules.
        controls = {r.recipe.key:r.recipe for r in table.exact.get(w.key, ())
                    if r.scope != 'PROPOSAL' and not rejection(r.recipe, w)}
        point = dict(value, candidates=[asdict(r) for r in candidates if r.kind == 'tc'],
            champion_controls=[asdict(r) for r in controls.values() if r.kind == 'tc'],
            pruning=report, fixed_schedules=True, sf_config=4 if w.qtype == 12 else 0,
            cohort_explicit_helpers=True, full_configs=[])
        points.append(point)
    bind_original(points, output)
    for p in points:
        w = Workload(**p['workload'])
        if not p.get('original_recipe'):
            raise ValueError('current selector has no control for '+p['id'])
        p['model_config_control'] = p['original_recipe']
        # Helpers are explicit candidates outside the seven admitted R4 points.
        # Their use here does NOT expand the shipping helper policy.
        p['helper_candidates'] = [dict(dequant_variant=dq, vector_io=io)
            for dq in (0, 8, 20, 21) for io in (0, 1, 2, 3)
            if dq == 0 or w.qtype >= 12]
        control = Recipe.from_dict(p['original_recipe'])
        if control.route == 'sf':
            config = dict(control.parameters).get('dequant_config', p['sf_config'])
            if config >= 0: p['sf_config'] = config
        if control.kind == 'full':
            p['full_configs'] = [dict(control.parameters)['dequant_config']]
        p['full_omission'] = None
        estimate = len(p['candidates']) + len(p['champion_controls']) + len(p['helper_candidates']) + 4
        if estimate > max_cases:
            raise ValueError(f'{w}: {estimate} cases exceed budget {max_cases}; no candidates silently dropped')
        p['max_cases'] = max_cases
    result = dict(schema='quactlize.prefill-asys-cohort-plan.v1', points=points,
        cohort_sha256=document['sha256'], evidence=sources, catalog=table.identity,
        scope=document['scope'], fixed_schedules=True, production_changed=False,
        performance_admitted=False, search='BOUNDED_NEIGHBOR_SCREEN_NOT_GLOBAL_OPTIMUM',
        counts=dict(points=len(points), **inventory(dict(points=points))['counts'],
                    recipes_before_resources=sum(len(p['candidates'])+len(p['champion_controls']) for p in points)))
    result['plan_sha256'] = digest(result)
    return json.loads(json.dumps(result))
