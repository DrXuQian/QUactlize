#!/usr/bin/env python3
"""Precompile a measured model's parent closure without loading weights or a GPU.

This fills the normal JIT cache; it never sets a tactic or changes dispatch.
Unknown models/workloads retain on-demand JIT. Reuse requires the exact runtime
manifest and observed weight geometry, including typed decode endpoints.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.kpack_jit import compiler_for, parent_tuple
from tools.run_kpack_batched_bench import inventory, save
from tools.gguf_internal_shape_inventory import read_gguf_header
from quactlize.runtime.compiler import sha, source_contract


def select_closure(model, plan, bundle, closure):
    if model['name'] != closure['model']:
        return 'MODEL_NOT_RECORDED'
    if model.get('split') != 'none':
        return 'TENSOR_PARALLEL_NOT_RECORDED'
    if not {1, 16, *plan['prompts']} <= set(closure['tokens']):
        return 'TOKEN_WORKLOAD_NOT_RECORDED'
    if sha(bundle / 'manifest.json') != closure['bundle_manifest_sha256']:
        return 'RUNTIME_CHANGED'
    info = inventory(Path(model['path']))
    eligible = set(info['eligible'])
    shapes = set()
    for path in info['files']:
        with Path(path).open('rb') as stream:
            tensors = read_gguf_header(stream, path)['tensors']
        for tensor in tensors:
            if tensor['name'] not in eligible:
                continue
            dims = tensor['dims_gguf']
            if len(dims) != 2:
                return 'GROUPED_NOT_RECORDED'
            shapes.add((tensor['qtype'], dims[1], dims[0]))
    return None if shapes == {tuple(s) for s in closure['shapes']} else 'WEIGHT_GEOMETRY_CHANGED'


def prewarm(closure, sdk, cache, jobs):
    if closure.get('schema') != 'quactlize.model-prewarm.v1' or jobs < 1:
        raise ValueError('invalid prewarm closure or job limit')
    compilers, tasks = {}, {}
    for module in closure['modules']:
        kind = module['dense_io'], module['compute_type']
        if type(kind[0]) is not bool or kind[1] not in ('f16', 'bf16'):
            raise ValueError('invalid module endpoint/compute contract')
        parent = parent_tuple(module['symbol'], module['tuple'])
        if kind not in compilers:
            compiler = compiler_for(sdk, cache, jobs, *kind)
            contract = compiler.identity.get('base_source_contract', source_contract(compiler.identity))
            if contract != closure['source_contract']:
                raise ValueError('recorded model closure differs from current kernel sources')
            compilers[kind] = compiler
        key = module['symbol'], *kind
        if key in tasks and tasks[key][1] != parent:
            raise ValueError('parent symbol aliases different geometry')
        tasks[key] = compilers[kind], parent
    if not tasks:
        raise ValueError('empty recorded parent closure')
    started = time.monotonic()
    workers = min(jobs, len(tasks))
    print(f'KPACK_MODEL_PREWARM parents={len(tasks)} workers={workers} device_work=NONE tuning=NONE', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(c.build, p): key for key, (c, p) in tasks.items()}
        for future in as_completed(pending):
            record = future.result()
            records.append(record)
            print(f'KPACK_MODEL_PREWARM completed={len(records)}/{len(tasks)} '
                  f'cache_hit={int(record["cache_hit"])} elapsed_seconds={time.monotonic()-started:.1f} '
                  f'parent={record["parent"]["symbol"]}', flush=True)
    return dict(status='PASS', scope='COMPILE_ONLY_NOT_NEW_DEVICE_ADMISSION',
                parents=len(records), workers=workers, seconds=time.monotonic()-started,
                cache_hits=sum(r['cache_hit'] for r in records), modules=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-plan', 'bundle', 'sdk', 'cache', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1 or args.output.exists():
        parser.error('positive jobs and a fresh output file are required')
    plan = json.loads(args.model_plan.read_text())
    closure = json.loads((ROOT / 'tools/kpack_model_prewarm_qwen3_32b.json').read_text())
    records = []
    for model in plan['models']:
        reason = select_closure(model, plan, args.bundle, closure)
        if reason:
            records.append(dict(model=model['name'], status='ON_DEMAND', reason=reason))
            print(f'KPACK_MODEL_PREWARM model={model["name"]} scope=ON_DEMAND reason={reason}', flush=True)
            continue
        record = prewarm(closure, args.sdk, args.cache, args.jobs)
        records.append(dict(record, model=model['name'], source_archive_sha256=closure['source_archive_sha256']))
    save(args.output, dict(status='PASS', models=records))


if __name__ == '__main__':
    main()
