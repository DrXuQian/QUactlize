#!/usr/bin/env python3
"""Isolate existing small-M reducers; never retime GEMM or change selection."""
import argparse
import collections
import ctypes as C
import gzip
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from quactlize.dequant.native import bind as bind_probe
from tools.kpack_prefill_measurement import (
    BUNDLE, bind_reducer, partial_values, reducer_expected, verify,
)
from tools.run_kpack_dequant_gate import device, packages, save
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_grouped_device_gate import graph_bind

INPUTS = (
    'policies/kpack_zw810_runtime_v1.json',
    'policies/kpack_zw810_heuristic_v1.json',
    'docs/measurements/q4_decode_policy_20260913.json.gz',
)
SCOPE = 'ISOLATED_REDUCER_REUSED_PARTIALS_NOT_PRODUCER_CONSUMER_TIMING'
COPIES = 2
REPLAYS = 32
ROUNDS = 3
SAMPLES = 5


def cases(root=ROOT):
    """Union of selected/confirmed S>1, not all shapes times all splits."""
    base, recent = [json.loads((root / name).read_text()) for name in INPUTS[:2]]
    configs = base['configurations'] | recent['configurations']
    refs = collections.defaultdict(set)
    for label, policy in (('historical', base), ('calibrated', recent)):
        for entry in policy['entries']:
            q, route, n, k, experts, m, _, _ = entry['key']
            split = configs[entry['config_id']]['split']
            if route < 2 and 1 <= m <= 8 and split > 1:
                refs[0, m, n, split].add(f'{label}:q{q}:r{route}:m{m}:n{n}:k{k}')
    evidence = json.loads(gzip.decompress((root / INPUTS[2]).read_bytes()))
    if evidence['authority']['scope'] != 'DECODE_ONLY_F32_ENDPOINTS_TC_CASTS_ADAPTERS_REAL_REDUCER_INCLUDED':
        raise ValueError('decode evidence is not complete-call timing')
    for record in evidence['cases']:
        w = record['workload']
        if w['operator'] not in ('dense', 'grouped') or not 1 <= w['tokens'] <= 8:
            raise ValueError('not the admitted small-M workload')
        grouped = int(w['operator'] == 'grouped')
        for key in record['confirmed']:
            if key.startswith('tc:'):
                split = int(key.split(':s')[1].split(':')[0])
                if split > 1:
                    refs[grouped, w['rows'] if grouped else w['tokens'], w['n'], split].add(w['id'])
    result = []
    for (compact, m, n, split), origins in sorted(refs.items()):
        if split not in (2, 4, 8) or n <= 0 or n % 256 or not 1 <= m <= (64 if compact else 8):
            raise ValueError('invalid small-M reducer key')
        result.append(dict(compact=compact, m=m, n=n, split=split,
            id=f'{"grouped" if compact else "dense"}-m{m}-n{n}-s{split}',
            references=sorted(origins)))
    return result


def validate(r, w):
    values = r['samples_us']
    if (r['workload'] != w or r['status'] != 'PASS' or r['scope'] != SCOPE or
            r['producer_timed'] is not False or r['dequant_timed'] is not False or
            r['scatter_timed'] is not False or r['production_changed'] is not False):
        raise ValueError('reducer result scope/context differs')
    if (len(values) != ROUNDS * SAMPLES or
            any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values) or
            r['median_us'] != statistics.median(values) or
            r['round_medians_us'] != [statistics.median(values[i:i+SAMPLES]) for i in range(0, len(values), SAMPLES)]):
        raise ValueError('invalid reducer samples')
    expected_bytes = w['m'] * w['n'] * w['split'] * 4
    if (r['partial_bytes'] != expected_bytes or r['copies'] != COPIES or
            r['calls_per_graph'] != COPIES * REPLAYS or
            r['resident_bytes'] != COPIES * (expected_bytes + w['m'] * w['n'] * 2) or
            r['resident_bytes'] > r['device']['l2_bytes'] // 2 or
            r['cache'] != 'REUSED_TWO_BUFFERS_NO_FLUSH_HIT_RATE_NOT_MEASURED'):
        raise ValueError('small reducer memory/timing denominator differs')
    if (r['raw_bad'] != 0 or type(r['negative_bad']) is not int or r['negative_bad'] <= 0 or
            r['guards'] != 'PASS' or r['changed_partial_graph'] != 'PASS' or
            r['partial_layout'] != 'FP32_S_M_N' or r['output_dtype'] != 'FP16' or
            r['addition_order'] != 'INCREASING_S' or
            r['fast_path'] is not bool(w['compact'] or w['m'] == 1)):
        raise ValueError('missing reducer numeric/dispatch controls')
    return r


def measure(sdk, lib, dev, w):
    r = Resources(sdk)
    handles, outputs, pointers = [], [], []
    graph = None
    count = w['m'] * w['n']
    parts = np.stack([partial_values(0, count, s) for s in range(w['split'])])
    size = count * 2
    resident = COPIES * (parts.nbytes + size)
    if resident > dev['l2_bytes'] // 2:
        r.close()
        raise ValueError('small reducer diagnostic exceeds its resident working-set bound')
    want = reducer_expected(0, count, w['split']).view('<u2')
    try:
        for _ in range(COPIES):
            pointer = r.alloc(parts.nbytes + 128) + (16 if w['compact'] else 0)
            pointers.append(pointer)
            checked(sdk.lib.hggcMemcpy(pointer, parts.ctypes.data, parts.nbytes, 1), 'partial upload')
            out = r.alloc(size + 256)
            outputs.append(out)
            r.fill(out, 0xA5, size + 256)
            h = C.c_void_p()
            checked(lib.prepare(w['compact'], w['m'], w['n'], w['split'], pointer,
                                parts.nbytes, out + 128, C.byref(h)), 'reducer prepare')
            handles.append(h)
        sdk.synchronize(None)

        def sequence():
            for h in handles:
                checked(lib.run(h, r.stream), 'production reducer')
            return 0

        def proof(index=None, negative=False):
            sdk.synchronize(r.stream)
            bad = 0
            for i in range(COPIES) if index is None else (index,):
                raw = sdk.download(outputs[i], size + 256)
                if raw[:128] != b'\xa5' * 128 or raw[-128:] != b'\xa5' * 128:
                    raise ValueError('reducer changed an output guard')
                got = np.frombuffer(raw[128:-128], dtype='<u2')
                bad += int(np.count_nonzero((got != want) & ~(((got & 0x7fff) == 0) & ((want & 0x7fff) == 0))))
            if (bad == 0) == negative:
                raise ValueError(f'reducer oracle failed bad={bad} negative={negative}')
            return bad

        sequence()
        proof()
        graph = Replay(sdk, r.stream, sequence, REPLAYS)
        checked(graph(), 'first graph upload/replay excluded')
        proof()
        # The captured graph must read changed partials, not a stale output.
        r.fill(pointers[0], 0, count * 4)
        checked(graph(), 'changed partial graph')
        bad = proof(0, True)
        sdk.synchronize(r.stream)
        checked(sdk.lib.hggcMemcpy(pointers[0], parts[0].ctypes.data, count * 4, 1), 'restore partial')
        sdk.synchronize(None)
        for out in outputs:
            r.fill(out + 128, 0x7e, size)
        checked(graph(), 'restored graph replay')
        proof()
        samples = []
        for _ in range(ROUNDS):
            samples.extend(value / (COPIES * REPLAYS) for value in r.samples(graph, SAMPLES))
            proof()
        result = dict(status='PASS', workload=w, scope=SCOPE, device=dev,
            samples_us=samples, median_us=statistics.median(samples),
            round_medians_us=[statistics.median(samples[i:i+SAMPLES]) for i in range(0, len(samples), SAMPLES)],
            copies=COPIES, calls_per_graph=COPIES * REPLAYS, partial_bytes=parts.nbytes,
            resident_bytes=resident, cache='REUSED_TWO_BUFFERS_NO_FLUSH_HIT_RATE_NOT_MEASURED',
            partial_layout='FP32_S_M_N', output_dtype='FP16', addition_order='INCREASING_S',
            raw_bad=0, negative_bad=bad, guards='PASS', changed_partial_graph='PASS',
            fast_path=bool(lib.fast(handles[0])), producer_timed=False, dequant_timed=False,
            scatter_timed=False, production_changed=False)
        return validate(result, w)
    finally:
        sdk.synchronize(r.stream)
        if graph:
            graph.close()
        for h in handles:
            lib.destroy(h)
        r.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path)
    p.add_argument('--bundle', type=Path, default=BUNDLE)
    p.add_argument('--output', type=Path)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    tasks = cases()
    if a.plan_only:
        print(json.dumps(dict(cases=len(tasks), dense=sum(not w['compact'] for w in tasks),
            grouped=sum(w['compact'] for w in tasks), compile='NONE', full_dequant=False,
            scope=SCOPE, tasks=tasks)))
        return
    if not a.sdk or not a.output:
        p.error('--sdk and --output are required to measure')
    manifest, _ = verify(a.bundle, a.sdk)
    header = 'quactlize/include/actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp'
    if manifest['reducer_header_sha256'] != sha(ROOT / header):
        raise ValueError('production reducer header differs from prebuilt carrier')
    sdk = SDK(a.sdk)
    graph_bind(sdk)
    probe_lib, _, probe = bind_probe(a.bundle)
    dev = device(sdk, probe)
    expected = json.loads((a.bundle / 'fixture-receipts.json').read_text())['authority']['device']
    if dev != expected:
        raise ValueError('use the same physical PPU as previous component measurements')
    lib = bind_reducer(a.bundle / 'libprefill_reducer.so')
    sources = [*INPUTS, header, 'tools/run_kpack_decode_reducer.py',
        'tools/kpack_prefill_measurement.py', 'tools/run_kpack_dequant_gate.py',
        'tools/run_kpack_gemv_gate.py', 'tools/run_kpack_grouped_decode_probe.py',
        'tools/run_kpack_grouped_device_gate.py', 'quactlize/runtime/native.py']
    authority = dict(schema='quactlize.decode-reducer.v1', tasks=tasks, scope=SCOPE,
        sources={name:sha(ROOT / name) for name in sources}, device=dev,
        bundle_sha256=sha(a.bundle / 'manifest.json'), runtime=manifest['runtime'],
        copies=COPIES, replays=REPLAYS, rounds=ROUNDS, samples=SAMPLES, python_packages=packages())
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / 'authority.json'
    if path.exists() and json.loads(path.read_text()) != authority:
        raise ValueError('resume authority differs; use a new output directory')
    save(path, authority)
    complete, failed, fresh_seconds = [], [], []
    started = time.monotonic()
    for i, w in enumerate(tasks):
        path = a.output / (w['id'] + '.json')
        if path.exists():
            value = validate(json.loads(path.read_text()), w)
            if value['device'] != dev:
                raise ValueError('resumed reducer device differs')
            complete.append(value)
        else:
            before = time.monotonic()
            try:
                value = measure(sdk, lib, dev, w)
                save(path, value)
                complete.append(value)
            except Exception as exc:
                traceback.print_exc()
                failed.append(dict(case=w['id'], error=str(exc)))
                print(f'DECODE_REDUCER_FAILURE case={w["id"]} remaining_continue=1', flush=True)
            fresh_seconds.append(time.monotonic() - before)
        remaining = 'UNKNOWN' if not fresh_seconds else f'{statistics.mean(fresh_seconds[-16:]) * (len(tasks)-i-1) / 60:.1f}'
        print(f'DECODE_REDUCER_PROGRESS completed={len(complete)}/{len(tasks)} failed={len(failed)} '
              f'case={w["id"]} remaining_minutes={remaining} eta=OBSERVED_RECENT_CASE_AVERAGE', flush=True)
    status = 'PASS' if len(complete) == len(tasks) and not failed else 'INCOMPLETE'
    files = {path.name:sha(path) for path in a.output.glob('*.json') if path.name != 'result.json'}
    save(a.output / 'result.json', dict(status=status, expected=len(tasks), completed=len(complete),
        rows=complete, failed=failed, files=files, elapsed_seconds=time.monotonic()-started,
        production_changed=False, scope=SCOPE))
    print(f'DECODE_REDUCER_DONE status={status} measured={len(complete)}/{len(tasks)} results={a.output}', flush=True)
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
