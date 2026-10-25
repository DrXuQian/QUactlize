"""Host final decisions and their required compiled capabilities; no GPU/JIT."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('variant', 'columns', 'warps', 'values', 'split')
SCHEMA = 'quactlize.final-selection.v1'
PREFILL_SCHEMA = 'quactlize.prefill-final-selection.v1'


def plan_prefill(output, *, historical=False):
    executable=Path(output)/'prefill-selection-query'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),
                    str(ROOT/'tools/kpack_selection.cpp'),'-o',str(executable)],check=True)
    # Only frozen-experiment replay tests request the old table. Not exposed by
    # the JIT/package CLI; a fresh production plan cannot accidentally use it.
    option='--historical-prefill-inventory' if historical else '--prefill-inventory'
    text=subprocess.check_output([str(executable),option],text=True)
    rows=[json.loads(line) for line in text.splitlines()]
    if len(rows)!=430 or any(r['status']!=0 or r['policy']!=19 for r in rows):
        raise ValueError('confirmed prefill selector denominator differs')
    return dict(schema=PREFILL_SCHEMA,scope='TYPED_PREFILL_COMPONENT_SELECTION_NOT_MODEL_ADMISSION',requests=rows)


def plan_smallm(output, requests=None, *, dense_fallbacks=False):
    if requests is not None:requests=[list(r) for r in requests]
    if dense_fallbacks and (requests is None or any(len(r)!=9 or r[1]!=0 or r[-1]!=0
            or r[4:7]!=[1,1,1] or not 1<=r[7]<=8 for r in requests)):
        raise ValueError('complete dense control query requires explicit F16 dense M1..8 requests')
    output = Path(output)
    executable = output / 'selection-query'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT),
                    str(ROOT/'tools/kpack_selection.cpp'), '-o', str(executable)], check=True)
    if requests is None:
        text = subprocess.check_output([str(executable), '--inventory'], text=True)
    else:
        requests = [list(r) for r in requests]
        if any(len(r) != 9 or any(type(x) is not int for x in r) for r in requests):
            raise ValueError('small-M request requires q/mode/N/K/experts/topk/channels/tokens/compute')
        text = subprocess.check_output([str(executable)]+(['--dense-fallbacks'] if dense_fallbacks else []), text=True,
            input=''.join(' '.join(map(str, r))+'\n' for r in requests))
    rows = [json.loads(line) for line in text.splitlines()]
    if requests is not None and [r['request'] for r in rows] != requests:
        raise ValueError('final selector omitted or reordered requests')
    return dict(schema=SCHEMA, scope='TYPED_SMALLM_FINAL_SELECTION_NOT_FIXED_ROUTE', requests=rows)


def requirements(plan):
    if plan.get('schema') != SCHEMA:
        raise ValueError('unknown final-selection schema')
    simt, tc, q4 = set(), set(), set()
    for row in plan['requests']:
        if row['status'] != 0:
            continue
        q, mode, _, _, _, _, _, _, compute = row['request']
        if row['kind'] == 1:
            simt.add((q, compute, *(row['config'][f] for f in FIELDS)))
        elif row['kind'] == 0:
            tc.add((row['parent']['symbol'], compute, mode == 0))
        elif row['kind'] == 2:
            q4.add((q, compute))
        else:
            raise ValueError('unknown selected implementation kind')
    return dict(simt=sorted(simt), tc=sorted(tc), q4=sorted(q4))


def validate_inventory(plan, execution, modules=(), *, jit=False):
    """Check exact tuples, not merely the presence of a vector variant."""
    required = requirements(plan)
    available = {int(q): {tuple(c[f] for f in FIELDS) for c in rows}
                 for q, rows in execution.get('simt_configs', {}).items()}
    for q, compute, *config in required['simt']:
        if tuple(config) not in available.get(q, set()):
            raise ValueError(f'final selector recipe absent from execution: q={q} config={config}')
        if compute and ('bf16' not in execution.get('simt_compute_v2', {}).get('compute', []) or
                        q not in execution.get('simt_compute_v2', {}).get('formats', [])):
            raise ValueError(f'final selector requires unavailable BF16 execution: q={q}')
    if required['q4'] and not execution.get('q4_decode_policy_sha256'):
        raise ValueError('final selector requires the compiled Q4 decode policy')
    for row in plan['requests']:
        if row['status'] != 0 or row['kind'] != 2:
            continue
        q, _, n, k, _, _, _, _, compute = row['request']
        config = [row['config'][f] for f in ('reader', 'variant', 'warps', 'values', 'columns')]
        if q != 12 or config not in [list(c) for c in execution.get('q4_decode_configs', {}).get(f'{n}x{k}', [])]:
            raise ValueError(f'final selector Q4 specialization absent from execution: {n}x{k} config={config}')
        if compute and 'bf16' not in execution.get('q4_decode_compute_v2', {}).get('compute', []):
            raise ValueError('final selector requires unavailable BF16 Q4 execution')
    # The old SIMT ABI contains geometry, not body identity. A matching tuple
    # alone must not make an old execution library look ready for the refresh.
    q8 = [r for r in plan['requests'] if r['status'] == 0 and
          r.get('implementation', {}).get('q8_body')]
    if q8:
        from quactlize.execution.q8_refresh import capability
        expected = capability()
        if execution.get('q8_refresh') != expected:
            raise ValueError('final selector Q8 body capability differs; rebuild execution')
        supported = {(tuple(r['key']), r['body'], tuple(r['config'][f] for f in FIELDS))
                     for r in expected['keys']}
        for r in q8:
            if (tuple(r['request']), r['implementation']['q8_body'],
                    tuple(r['config'][f] for f in FIELDS)) not in supported:
                raise ValueError('final selector Q8 body/config is outside the measured scope')
    compiled = {(m['parent']['symbol'], int(m['identity'].get('compute_type', 'f16') == 'bf16'),
                 m['identity'].get('endpoints') == 'decode-m1-8-f32-bf16-v1') for m in modules}
    missing = [r for r in required['tc'] if tuple(r) not in compiled]
    if missing and not jit:
        raise ValueError(f'final selector needs {len(missing)} unpackaged typed TC parents; enable JIT or prewarm the typed plan')
    result = dict(required=required, jit_required=missing)
    if q8:
        result['q8_refresh'] = expected
    return result


def prewarm_groups(plan):
    """Only selected TC parents need JIT; compute and endpoints are identity."""
    from quactlize.runtime.tuning import ROUTES
    from quactlize.runtime.compiler import materialize_parent
    prefill=plan.get('schema')==PREFILL_SCHEMA
    if not prefill: requirements(plan)
    if any(row['status'] != 0 for row in plan['requests']):
        raise ValueError('final-selection plan has policy misses; no complete prewarm is possible')
    groups = {}
    for row in plan['requests']:
        if row['status'] != 0 or row['kind'] != 0:
            continue
        key = ('bf16' if row['request'][-1] else 'f16', row['request'][1] == 0 and not prefill)
        parent = dict(row['parent'])
        parent['route'] = ROUTES[parent['route']]
        parent = materialize_parent(parent)
        pool = groups.setdefault(key, {})
        if parent['symbol'] in pool and pool[parent['symbol']] != parent:
            raise ValueError('selected symbol aliases different parent tuples')
        pool[parent['symbol']] = parent
    return [(compute, dense, sorted(pool.values(), key=lambda p: p['symbol']))
            for (compute, dense), pool in sorted(groups.items())]
