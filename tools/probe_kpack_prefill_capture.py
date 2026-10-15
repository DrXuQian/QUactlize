#!/usr/bin/env python3
"""Two fresh processes for the failed Q3 dense BF16 provider capture, not a sweep.

The explicit workspace is a candidate, not a silently promoted default. Keep
the ordinary untimed warmup and the existing per-kernel event timers. Matching
two dispatches is neither a tuning-complete test nor a performance proof.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest

WORKLOAD = Workload(11, 1024, 5120, 128)
ARMS = {'default': 0, 'resident': 32*1024*1024}


def child(args):
    from dev.strategy.cold_decode import Ring
    from dev.strategy.prefill_fixture import Context, Helpers
    from dev.strategy.prefill_full import Full
    from dev.strategy.prefill_measure import measure_core
    from dev.strategy.prefill_build import verify
    from quactlize.runtime.native import SDK, sdk_identity
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_decode import dequant_checked
    # The parent owns the device lock. Never open/acquire it a second time.
    if args.lock_fd is None: raise ValueError('child requires the inherited device lock')
    fcntl.fcntl(args.lock_fd, fcntl.F_GETFD)
    authority = json.loads((args.output/'authority.json').read_text())
    if (authority['harness'] != harness_identity(ROOT) or authority['device'] != device_probe(args.sdk)
            or authority['sdk'] != sdk_identity(args.sdk) or authority['workload'] != asdict(WORKLOAD)):
        raise ValueError('capture probe source/SDK/device/workload changed')
    verify(args.sdk, args.output/'helpers')
    dequant_checked(args.output/'dequant', args.sdk)
    out = args.output/args.arm
    out.mkdir(exist_ok=False)
    ring = b = full = None
    start = time.monotonic()
    try:
        sdk = SDK(args.sdk)
        ring = Ring(sdk, WORKLOAD, authority['l2']['bytes'], minimum_active=1)
        b = Context(sdk, ring, WORKLOAD, Helpers(args.output/'helpers/libstrategy_prefill.so'))
        full = Full(b, args.output/'dequant', authority['l2']['bytes'], args.sdk,
                    workspace_bytes=ARMS[args.arm])
        print('STRATEGY_CAPTURE_PROVIDER '+json.dumps(full.receipt), flush=True)
        # This single config is only a weight producer. Expansion is NOT in
        # the GEMM timer, and no dequant-config or TC/SIMT search runs here.
        recipe = Recipe.make('full', 'full', provider='bf16-default', dequant_config=-1)
        data = measure_core(b, full, recipe, digest([authority, args.arm]),
            digest([authority['helpers'], authority['dequant'], full.receipt]), out, config=4, full=True)
        full.close(); full = None
        b.close(); b = None
        ring.close(); ring = None
        # A failed teardown must not publish a successful arm.
        atomic_json(out/'result.json', dict(status='PASS', arm=args.arm, data=data,
            wall_seconds=time.monotonic()-start, performance_admitted=False))
        return 0
    except BaseException as exc:
        atomic_json(out/'result.json', dict(status='FAIL', arm=args.arm, error=repr(exc),
            wall_seconds=time.monotonic()-start, performance_admitted=False))
        raise
    finally:
        if full: full.close()
        if b: b.close()
        if ring: ring.close()


def wait_arm(command, log, held, timeout):
    """A failed native runtime cannot contaminate the next arm's process."""
    with log.open('x') as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, pass_fds=(held.fileno(),))
        start = time.monotonic()
        try:
            while True:
                remaining = timeout-(time.monotonic()-start)
                if remaining <= 0:
                    p.kill(); p.wait()  # Only the process this probe created.
                    return dict(rc=p.returncode, timeout=True, seconds=time.monotonic()-start)
                try:
                    rc = p.wait(timeout=min(30, remaining))
                    return dict(rc=rc, timeout=False, seconds=time.monotonic()-start)
                except subprocess.TimeoutExpired:
                    print(f'STRATEGY_CAPTURE_WAIT log={log} seconds={time.monotonic()-start:.1f}', flush=True)
        finally:
            if p.poll() is None:
                p.kill(); p.wait()


def arm_pass(row):
    return (row['process']['rc'] == 0 and not row['process']['timeout'] and
            row.get('result', {}).get('status') == 'PASS' and
            row['result'].get('data', {}).get('correctness') == 'PASS')


def summarize(arms):
    passed = {name: arm_pass(arms[name]) for name in ARMS}
    return dict(status='CANDIDATE_PASS' if passed['resident'] else 'FAIL', arms=arms,
        default_reproduced_failure=not passed['default'],
        conclusion=('RESIDENT_CAPTURE_AND_ISOLATED_TIMERS_PASS_CAUSE_REQUIRES_NODE_LOGS' if passed['resident']
                    else 'EXPLICIT_WORKSPACE_DID_NOT_PASS_NO_CAMPAIGN_PROMOTION'),
        scope='ONE_Q3_DENSE_POINT_NOT_FULL_SWEEP_OR_MODEL_AB',
        production_changed=False, performance_admitted=False, full_campaign=False)


def package(output):
    # Keep identities, individual samples, accuracy and errors, not DSOs or
    # native ISA dumps. Raw build objects remain in the original output.
    target = Path(str(output)+'.results.tgz')
    files = [p for p in output.rglob('*') if p.is_file() and p.suffix in ('.json', '.log')
             and p.name != 'native.json']
    with tarfile.open(target, 'x:gz') as archive:
        for path in sorted(files): archive.add(path, arcname=str(path.relative_to(output)), recursive=False)
    print(f'results={target}', flush=True)


def run(args):
    from dev.strategy.isolated import check_sdk_graph_abi
    from dev.strategy.prefill_build import build as build_helpers
    from tools.build_kpack_dequant import build as build_dequant
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_decode import dequant_checked, l2_authority
    from quactlize.runtime.native import sdk_identity
    print('STRATEGY_CAPTURE_START point=Q3_M128_N1024_K5120 arms=default,resident '
          'tc_compiles=0 production_changed=0', flush=True)
    start = time.monotonic()
    harness = harness_identity(ROOT)
    graph_abi = check_sdk_graph_abi(args.sdk)
    device = device_probe(args.sdk)
    path = Path('/tmp')/('quactlize-components-'+digest(device['pci'])[:24]+'.lock')
    with path.open('a') as held:
        try: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('device is locked by another component run; choose an IDLE card, do not remove '+str(path))
        helper = build_helpers(args.sdk, args.output/'helpers')
        build_dequant(args.sdk, args.output/'dequant')
        dequant = dequant_checked(args.output/'dequant', args.sdk)
        if harness != harness_identity(ROOT): raise ValueError('capture probe sources changed during setup')
        authority = dict(workload=asdict(WORKLOAD), harness=harness, device=device, sdk=sdk_identity(args.sdk),
            helpers=helper, dequant=dequant, l2=l2_authority(args), arms=ARMS, graph_abi=graph_abi,
            warmup='UNTIMED_COMPLETE_CALLS_AND_SYNCHRONIZATION_BEFORE_ISOLATED_GRAPH_EVENTS',
            tuning='NO_ONLINE_SEARCH_OR_DISPATCH_FINGERPRINT_GATE', production_changed=False)
        atomic_json(args.output/'authority.json', authority)
        arms = {}
        for arm in ARMS:
            log = args.output/(arm+'.log')
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--sdk', str(args.sdk),
                '--output', str(args.output), '--arm', arm, '--lock-fd', str(held.fileno())]
            process = wait_arm(command, log, held, args.timeout_seconds)
            result_path = args.output/arm/'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            arms[arm] = dict(process=process, result=result, log=str(log))
            atomic_json(args.output/(arm+'.process.json'), process)
            print(f'STRATEGY_CAPTURE_ARM arm={arm} rc={process["rc"]} passed={int(arm_pass(arms[arm]))} log={log}', flush=True)
            if not arm_pass(arms[arm]):
                with log.open('rb') as f:
                    f.seek(max(0, log.stat().st_size-8192))
                    print(f.read().decode('utf-8', errors='replace'), flush=True)
        data = summarize(arms)
        data['elapsed_seconds'] = time.monotonic()-start
        atomic_json(args.output/'summary.json', data)
        print('STRATEGY_CAPTURE_DONE '+json.dumps({k: v for k, v in data.items() if k != 'arms'}), flush=True)
        return 0 if data['status'] == 'CANDIDATE_PASS' else 1


def main():
    # Retain old result readers for historical receipts, not another cuBLAS
    # graph experiment. The replacement runs ordinary warmed provider calls.
    raise SystemExit('cuBLAS capture probe retired; use tools/probe_kpack_cublas_ordinary.py (no CUDA Graph)')


def historical_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--l2-bytes', type=int, default=0)
    parser.add_argument('--idle-confirmed', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=480)
    parser.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.sdk, args.output = args.sdk.resolve(strict=True), args.output.resolve()
    if args.arm: return child(args)
    if not args.idle_confirmed: parser.error('confirm no other GPU workload, then pass --idle-confirmed')
    if args.l2_bytes < 0 or args.timeout_seconds <= 0: parser.error('invalid L2/timeout')
    if args.output.exists(): parser.error('--output must be new; previous runs are preserved')
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
