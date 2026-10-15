#!/usr/bin/env python3
"""Ordinary cuBLAS CUDA-event timing. No CUDA Graph, Asys, or TC sweep."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.process import wait_process
from dev.strategy.cublas_events import summarize, review_existing
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
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
        data = measure(b, full)
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


def run_events(sdk, output, authority, held, timeout=600):
    """One unprofiled worker; native builds are verified, not rebuilt."""
    output = Path(output)
    atomic_json(output/'authority.json', authority)
    command = [sys.executable, '-u', str(ROOT/'tools/probe_kpack_cublas_ordinary.py'),
               '--sdk', str(sdk), '--output', str(output), '--worker', 'events']
    atomic_json(output/'events.command.json', command)
    process = wait_process(command, output/'events.log', held, timeout)
    atomic_json(output/'events.process.json', process)
    path = output/'events/result.json'
    result = json.loads(path.read_text()) if path.is_file() else {}
    if process['rc'] or process['timeout'] or result.get('status') != 'PASS' or (
            result.get('authority_sha256') != digest(authority)):
        raise ValueError(f'ordinary cuBLAS events failed; see {output/"events.log"}')
    data = summarize(result['data'])
    data['worker'] = result
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
        authority = dict(harness=harness_identity(ROOT), sdk=sdk_identity(args.sdk), device=device,
            workload=asdict(WORKLOAD), l2=l2_authority(SimpleNamespace(output=builds, l2_bytes=args.l2_bytes)),
            helpers=hp, dequant=dq,
            helpers_path=str(builds/'helpers'), dequant_path=str(builds/'dequant'),
            cuda_graph=False, profiler=False,
            workspace='PROVIDER_DEFAULT', production_changed=False)
        print('STRATEGY_CUBLAS_START point=Q3_M128_N1024_K5120 cuda_graph=0 tc_compiles=0', flush=True)
        start = time.monotonic()
        data = run_events(args.sdk, args.output, authority, held, args.timeout_seconds)
        print('STRATEGY_CUBLAS_DONE '+json.dumps(dict(status=data['status'],
            provider_us=data['timers']['provider']['median_us'], seconds=time.monotonic()-start,
            cuda_graph=False, profiler=False, performance_admitted=False, full_campaign=False)), flush=True)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-builds', type=Path, help='verified prior helpers/ and dequant/; no timing reuse')
    p.add_argument('--review-existing', type=Path, help='read a prior successful events worker; no SDK/GPU access')
    p.add_argument('--l2-bytes', type=int, default=0)
    p.add_argument('--idle-confirmed', action='store_true')
    p.add_argument('--timeout-seconds', type=int, default=600)
    p.add_argument('--worker', choices=('events',), help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.review_existing:
        if args.worker or args.output.exists(): p.error('offline review requires a new output directory')
        result = review_existing(args.review_existing)
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output/'summary.json', result)
        print('STRATEGY_CUBLAS_EVENTS_REVIEW '+json.dumps(dict(status=result['status'],
            provider_us=result['timers']['provider']['median_us'],
            missing_adapter_timers=result['missing_adapter_timers'], device_launches=0)), flush=True)
        package(args.output)
        return 0
    args.sdk = args.sdk.resolve(strict=True)
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
