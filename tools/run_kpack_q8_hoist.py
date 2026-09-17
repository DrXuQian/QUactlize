#!/usr/bin/env python3
"""Same-config Q8 hoist A/B through immutable production C entrypoints."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.gemv_simt.access import pattern
from dev.gemv_simt.fixture import weights
from dev.gemv_simt.native import Graph, Runtime, checked
from dev.gemv_simt.production import Library
from dev.gemv_simt.q8_vector_run import Bench, l2_identity
from quactlize.execution.simt_codegen import Config
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import Module, sdk_identity
from tools.verify_kpack_dispatch import verify

POINTS = ((512, 2048, Config(5, 4, 8, 4)),
          (2048, 512, Config(5, 8, 4, 4)),
          (2048, 4096, Config(5, 8, 4, 4, 8)))
KERNEL = 'quactlize/execution/simt_q8_vector.cuh'
LIBRARY = 'libquactlize_ppu_execution.so'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compare_contracts(baseline, candidate):
    a, b = baseline['execution_receipt'], candidate['execution_receipt']
    changed = {p for p in a['source_hashes'].keys() | b['source_hashes'].keys()
               if a['source_hashes'].get(p) != b['source_hashes'].get(p)}
    if changed != {KERNEL}:
        raise ValueError('hoist A/B must change only the Q8 body: '+str(sorted(changed)))
    for field in ('runtime', 'compiler_sha256', 'flags', 'simt_configs'):
        if a[field] != b[field]:
            raise ValueError('execution comparison differs: '+field)
    if baseline['policy_hashes'] != candidate['policy_hashes']:
        raise ValueError('hoist A/B must retain the production selectors')
    return sorted(changed)


def numeric_pair(bench, libraries, config):
    records, previous = [], {}
    for name, lib in libraries.items():
        bench.lib = lib
        for repeat in (0, 1, 2, 4):
            bench.update(repeat)
            got, error = bench.correctness(config)
            bits = got.view('<u4')
            if name == 'baseline':
                previous[repeat] = bits.copy()
            elif not np.array_equal(previous[repeat], bits):
                raise ValueError('hoist changed same-order FP32 output bits')
            records.append(dict(arm=name, repeat=repeat, error=error))
        proof = bench.replay_and_negative(config)
        bench.invalid_id_negative(config)
        records.append(dict(arm=name, replay=proof))
    return records


def device_identity(rt, bundle, manifest, l2_bytes):
    record = manifest['modules'][0]
    module = Module(dict(record, path=str(bundle/record['path'])))
    identity = module.device_identity()
    if identity['device'] != 'PPU-ZW810' or identity['compute_units'] != 72:
        raise ValueError('Q8 hoist comparison requires a ZW810 with 72 CUs: '+str(identity))
    if identity['ordinal'] != 0:
        raise ValueError('use one visible device with logical ordinal zero')
    pci = C.create_string_buffer(64)
    fn = rt.lib.hggcDeviceGetPCIBusId
    fn.argtypes, fn.restype = [C.c_char_p, C.c_int, C.c_int], C.c_int
    checked(fn(pci, len(pci), identity['ordinal']), 'physical device PCI identity')
    l2 = C.c_int()
    l2_status = rt.DeviceGetAttribute(C.byref(l2), 38, identity['ordinal'])
    if l2_status:
        rt.lib.hggcGetLastError()  # Record the unsupported property, then use the explicit receipt.
    identity = l2_identity(dict(identity, l2_bytes=l2.value if not l2_status else 0), l2_bytes)
    if identity['l2_bytes'] <= 0:
        raise ValueError('positive L2_BYTES operator receipt required when SDK reports zero')
    return dict(identity, l2_query_status=l2_status, pci=pci.value.decode(), visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                idle_status='OPERATOR_MUST_EXCLUDE_OTHER_GPU_WORK')


def measure(args):
    n, k, config = POINTS[args.point]
    contract = json.loads((args.output/'contract.json').read_text())
    if (sha(args.baseline/LIBRARY) != contract['baseline']['execution_sha256'] or
            sha(args.candidate/LIBRARY) != contract['candidate_execution_sha256'] or
            sha(args.candidate/'manifest.json') != contract['candidate_manifest_sha256'] or
            sha(Path(__file__)) != contract['source_sha256']):
        raise ValueError('child inputs differ from the A/B contract')
    manifest = json.loads((args.candidate/'manifest.json').read_text())
    rt = Runtime(args.sdk, 'ppu')
    try:
        identity = device_identity(rt, args.candidate, manifest, args.l2_bytes)
        libraries = {arm: Library(directory/LIBRARY, 0) for arm, directory in
                     (('baseline', args.baseline), ('candidate', args.candidate))}
        w = weights(8, n, k, 1)
        numeric = []
        for tokens in range(1, 9):
            for compute in (0, 1):
                for weak in (False, True):
                    for lib in libraries.values():
                        lib.compute = compute
                    bench = Bench(rt, libraries['baseline'], w, tokens, 0, 1, compute, weak=weak)
                    try:
                        rows = numeric_pair(bench, libraries, config)
                        numeric.append(dict(tokens=tokens, compute=compute, weak_scale_alignment=weak, records=rows))
                    finally:
                        bench.close()
        print(f'Q8_HOIST_NUMERIC point={args.point} status=PASS contexts={len(numeric)}', flush=True)
        useful = sum(w.planes[name].nbytes for name in ('low', 'high', 'units'))
        copies = math.ceil(2.25*identity['l2_bytes']/useful)
        traversals = max(1, math.ceil(32/copies))
        for lib in libraries.values():
            lib.compute = 0
        bench = Bench(rt, libraries['baseline'], w, 1, 0, 1, 0, copies=copies)
        graphs, rounds = {}, {arm: [] for arm in libraries}
        try:
            # Every ring copy and both real libraries must pass before timing.
            bench.update(0)
            for name, lib in libraries.items():
                for call in bench.calls:
                    bench.poison()
                    checked(lib.prepare(call, config)(), 'ring-copy independent oracle')
                    rt.sync()
                    bench.output_check()
                graphs[name] = Graph(rt, [lib.prepare(call, config) for call in bench.calls]*traversals)
                graphs[name].sample()
            for round_index in range(6):
                order = list(libraries) if round_index % 2 == 0 else list(reversed(libraries))
                for arm in order:
                    rounds[arm].append([graphs[arm].sample() for _ in range(15)])
                print(f'Q8_HOIST_PROGRESS point={args.point} round={round_index+1}/6', flush=True)
            medians = {arm: statistics.median(x for r in values for x in r) for arm, values in rounds.items()}
            access = pattern(8, config, n, k, bases=dict(A=bench.a%128, low=bench.call.low%128,
                                                       high=0, units=bench.call.units%128))
            access['hoist'] = dict(addresses_changed=False, fma_order_changed=False,
                baseline_live_packed_words=2*config.values,
                candidate_live_packed_words=(8 if config.split == 1 else 2)*config.values,
                blocks=config.split*n//config.tile_n, threads=config.warps*32,
                dequant='UNCHANGED_LOP3_HALF2_EXACT_I8_TO_F32')
            result = dict(status='PASS', point=args.point, shape=[1,n,k], config=config.key,
                device=identity, numeric=numeric, rounds=rounds, median_us=medians,
                delta_pct=100*(medians['candidate']/medians['baseline']-1),
                modeled_mbu_pct={arm:100*useful/(us*args.peak_gbps*1000) for arm,us in medians.items()},
                bandwidth_roof_gbps=args.peak_gbps, roof_source='OPERATOR_ASSUMPTION_NOT_ACU_COUNTER',
                useful_weight_bytes=useful, ring_copies=copies, traversals=traversals,
                ring_bytes=copies*useful, access=access, split_control=config.split>1,
                scope='ROTATING_WEIGHTS_PRODUCTION_FULL_CALL_INCLUDING_REDUCER',
                warmup='GRAPH_UPLOAD_FIRST_LAUNCH_AND_FIRST_SAMPLE_EXCLUDED',
                kernel_admission='CORRECTNESS_PASS_PERFORMANCE_PENDING_REVIEW')
        finally:
            for graph in graphs.values():
                graph.close()
            bench.close()
        save(args.output/f'point-{args.point}.json', result)
        print('Q8_HOIST_RESULT '+json.dumps({k:v for k,v in result.items() if k in
              ('status','shape','config','median_us','delta_pct','modeled_mbu_pct','split_control')}), flush=True)
    finally:
        rt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','baseline','candidate','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--l2-bytes', type=int, default=0)
    parser.add_argument('--peak-gbps', type=float, default=2700)
    parser.add_argument('--point', type=int, choices=range(len(POINTS)))
    args = parser.parse_args()
    if not math.isfinite(args.peak_gbps) or args.peak_gbps <= 0:
        parser.error('positive finite bandwidth roof required')
    if args.point is not None:
        measure(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    old = json.loads((args.baseline/'manifest.json').read_text())
    pin = json.loads((ROOT/'tools/kpack_q8_hoist_baseline.json').read_text())
    if (sha(args.baseline/'manifest.json') != pin['manifest_sha256'] or
            sha(args.baseline/LIBRARY) != pin['execution_sha256']):
        raise ValueError('immutable Q8 baseline identity differs')
    candidate = verify(args.candidate, sdk=args.sdk)
    changed = compare_contracts(old, candidate)
    save(args.output/'contract.json', dict(changed=changed, baseline=pin,
        candidate_manifest_sha256=sha(args.candidate/'manifest.json'),
        candidate_execution_sha256=candidate['execution_sha256'], sdk=sdk_identity(args.sdk),
        source_sha256=sha(Path(__file__)), points=[dict(n=n,k=k,config=c.key) for n,k,c in POINTS]))
    started, records = time.monotonic(), []
    for i in range(len(POINTS)):
        command = [sys.executable, '-u', str(Path(__file__)), '--point', str(i)]
        for name in ('sdk','baseline','candidate','output','l2_bytes','peak_gbps'):
            command += ['--'+name.replace('_','-'), str(getattr(args,name))]
        log = args.output/f'point-{i}.log'
        with log.open('x') as stream:
            proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'Q8_HOIST_WAIT point={i} elapsed_s={time.monotonic()-started:.1f} log={log}',flush=True)
            rc = proc.returncode
        path = args.output/f'point-{i}.json'
        row = json.loads(path.read_text()) if rc == 0 and path.exists() else dict(status='FAIL',point=i,rc=rc,log=str(log))
        records.append(row)
        save(args.output/'summary.json', dict(status='RUNNING', records=records))
        print(f'Q8_HOIST_POINT point={i} status={row["status"]} elapsed_s={time.monotonic()-started:.1f} log={log}',flush=True)
    success = all(r['status']=='PASS' for r in records)
    save(args.output/'summary.json', dict(status='PASS' if success else 'FAIL', records=records))
    return 0 if success else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
