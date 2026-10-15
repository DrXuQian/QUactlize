#!/usr/bin/env python3
"""Bounded Q3 BF16 cuBLAS gate, ordinary warmup/calls, NO CUDA Graph.

The first child supplies unprofiled provider event spans. The second supplies
Asys individual GPU activities. Neither compiles TC modules nor starts a sweep.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.cublas_trace import parse_trace, profile_command, summarize_trace, wait_process
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.probe_kpack_prefill_capture import package

WORKLOAD = Workload(11, 1024, 5120, 128)


def worker(args):
    from dev.strategy.cold_decode import Ring
    from dev.strategy.cublas_ordinary import measure
    from dev.strategy.prefill_fixture import Context, Helpers
    from dev.strategy.prefill_full import Full
    from dev.strategy.prefill_build import verify
    from quactlize.runtime.native import SDK, sdk_identity
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_decode import dequant_checked
    authority = json.loads((args.output/'authority.json').read_text())
    if (authority['harness'] != harness_identity(ROOT) or authority['sdk'] != sdk_identity(args.sdk)
            or authority['device'] != device_probe(args.sdk)):
        raise ValueError('ordinary cuBLAS source/SDK/device changed')
    verify(args.sdk, Path(authority['helpers_path']))
    dequant_checked(Path(authority['dequant_path']), args.sdk)
    out = args.output/args.worker; out.mkdir(exist_ok=False)
    ring = b = full = None
    try:
        w = Workload(**authority['workload'])
        sdk = SDK(args.sdk)
        ring = Ring(sdk, w, authority['l2']['bytes'], minimum_active=1)
        b = Context(sdk, ring, w, Helpers(Path(authority['helpers_path'])/'libstrategy_prefill.so'))
        full = Full(b, Path(authority['dequant_path']), authority['l2']['bytes'], args.sdk)
        data = measure(b, full, args.sdk, trace=args.worker == 'asys')
        full.close(); full = None
        b.close(); b = None
        ring.close(); ring = None
        atomic_json(out/'result.json', dict(status='PASS', data=data, authority_sha256=digest(authority)))
        return 0
    except BaseException as exc:
        atomic_json(out/'result.json', dict(status='FAIL', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if full: full.close()
        if b: b.close()
        if ring: ring.close()


def run_profile(sdk, output, authority, held, timeout=600):
    """Reusable gate; native builds are supplied and verified, not rebuilt."""
    output = Path(output)
    atomic_json(output/'authority.json', authority)
    asys = Path(authority['asys']['path'])
    if sha(asys) != authority['asys']['sha256']: raise ValueError('Asys binary changed')
    rows = {}
    for arm in ('events', 'asys'):
        application = [sys.executable, '-u', str(ROOT/'tools/probe_kpack_cublas_ordinary.py'),
                       '--sdk', str(sdk), '--output', str(output), '--worker', arm]
        command = (profile_command(asys, output/'ordinary.asysrep', application)
                   if arm == 'asys' else application)
        atomic_json(output/(arm+'.command.json'), command)
        try:
            process = wait_process(command, output/(arm+'.log'), held, timeout)
        finally:
            if arm == 'asys':
                # Only the unique session started above; preserve unrelated
                # users' services/traces, including when our worker times out.
                session = command[command.index('--session-new')+1]
                with (output/'asys-shutdown.log').open('x') as log:
                    try:
                        subprocess.run([str(asys), 'shutdown', '--session', session],
                            stdout=log, stderr=subprocess.STDOUT, timeout=15, check=False)
                    except subprocess.TimeoutExpired:
                        log.write('Owned profiler session shutdown timed out\n')
        atomic_json(output/(arm+'.process.json'), process)
        path = output/arm/'result.json'
        result = json.loads(path.read_text()) if path.is_file() else {}
        rows[arm] = dict(process=process, result=result)
        if process['rc'] or process['timeout'] or result.get('status') != 'PASS' or (
                result.get('authority_sha256') != digest(authority) or
                result.get('data', {}).get('correctness') != 'PASS'):
            raise ValueError(f'ordinary cuBLAS {arm} failed; see {output/(arm+".log")}')
    command = [str(asys), 'export', '--output', str(output/'ordinary.sqlite'), str(output/'ordinary.asysrep')]
    atomic_json(output/'export.command.json', command)
    process = wait_process(command, output/'export.log', held, 120)
    if process['rc'] or process['timeout']: raise ValueError('Asys export failed; see '+str(output/'export.log'))
    for key in ('receipt', 'provider_loaded_images', 'fixture', 'warmup_calls', 'samples_per_round'):
        if rows['events']['result']['data'][key] != rows['asys']['result']['data'][key]:
            raise ValueError('unprofiled/profiled workload or provider differs: '+key)
    trace = parse_trace(output/'ordinary.sqlite', rows['asys']['result']['data']['measured'])
    atomic_json(output/'kernel-activities.json', trace)
    data = dict(status='PASS', **summarize_trace(trace), arms=rows, cuda_graph=False,
        scope='ONE_DENSE_PROVIDER_GATE_NOT_FULL_SWEEP_OR_MODEL_AB',
        report=str(output/'ordinary.asysrep'), production_changed=False, performance_admitted=False)
    event_medians = {a: statistics.median(r['event_span_us'] for r in row['result']['data']['measured']
                                        if r['phase'] == 'provider') for a, row in rows.items()}
    data['auxiliary_event_median_us'] = event_medians
    data['profiled_vs_unprofiled_event_pct'] = 100*(event_medians['asys']/event_medians['events']-1)
    data['event_scope'] = 'PROVIDER_API_SPAN_NOT_GEMM_KERNEL_TIME_NOT_USED_IN_COMPONENT_SUM'
    atomic_json(output/'summary.json', data)
    return data


def run(args):
    from dev.strategy.prefill_build import build as build_helpers, verify
    from tools.build_kpack_dequant import build as build_dequant
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_decode import dequant_checked, l2_authority
    from quactlize.runtime.native import sdk_identity
    device = device_probe(args.sdk)
    lock = Path('/tmp')/('quactlize-components-'+digest(device['pci'])[:24]+'.lock')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builds = args.reuse_builds or args.output
        if args.reuse_builds:
            hp = verify(args.sdk, builds/'helpers'); dq = dequant_checked(builds/'dequant', args.sdk)
        else:
            hp = build_helpers(args.sdk, builds/'helpers')
            build_dequant(args.sdk, builds/'dequant'); dq = dequant_checked(builds/'dequant', args.sdk)
        asys = (args.asys or args.sdk/'asight/bin/asys').resolve(strict=True)
        authority = dict(harness=harness_identity(ROOT), sdk=sdk_identity(args.sdk), device=device,
            workload=asdict(WORKLOAD), l2=l2_authority(SimpleNamespace(output=builds, l2_bytes=args.l2_bytes)),
            helpers=hp, dequant=dq,
            helpers_path=str(builds/'helpers'), dequant_path=str(builds/'dequant'),
            asys=dict(path=str(asys), sha256=sha(asys)), cuda_graph=False,
            workspace='PROVIDER_DEFAULT', production_changed=False)
        print('STRATEGY_CUBLAS_START point=Q3_M128_N1024_K5120 cuda_graph=0 tc_compiles=0', flush=True)
        start = time.monotonic()
        data = run_profile(args.sdk, args.output, authority, held, args.timeout_seconds)
        print('STRATEGY_CUBLAS_DONE '+json.dumps(dict(status=data['status'],
            kernel_medians=data['phase_kernel_sum_median_us'], seconds=time.monotonic()-start,
            cuda_graph=False, performance_admitted=False, full_campaign=False)), flush=True)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-builds', type=Path, help='verified prior helpers/ and dequant/; no timing reuse')
    p.add_argument('--asys', type=Path)
    p.add_argument('--l2-bytes', type=int, default=0)
    p.add_argument('--idle-confirmed', action='store_true')
    p.add_argument('--timeout-seconds', type=int, default=600)
    p.add_argument('--worker', choices=('events', 'asys'), help=argparse.SUPPRESS)
    args = p.parse_args()
    args.sdk = args.sdk.resolve(strict=True); args.output = args.output.resolve()
    if args.worker: return worker(args)
    if not args.idle_confirmed: p.error('--idle-confirmed required on an idle card')
    if args.timeout_seconds <= 0 or args.l2_bytes < 0: p.error('invalid timeout/L2')
    if args.output.exists(): p.error('--output must be new; old results are preserved')
    if args.reuse_builds: args.reuse_builds = args.reuse_builds.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        return run(args)
    except Exception as exc:
        atomic_json(args.output/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        package(args.output)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
