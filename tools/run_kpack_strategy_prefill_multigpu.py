#!/usr/bin/env python3
"""Resume provider measurements or supplement the reviewed prefill rules.

Match cards by physical identity, not visible ordinal. Keep every family on
its original card; do not move old timings to another card. Each worker owns
the standard device lock and a new output. No source/binary/result overwrite.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.prefill_retry import prior_root, read
from dev.strategy.provenance import harness_identity
from dev.strategy.prefill_upgrade import source_delta
from quactlize.runtime.tuning import digest


def discover(previous, supplement=False):
    paths = sorted(Path(previous).glob('shard-*/measurement/authority.json' if supplement else
                                      'gpu-*/kpack-rule-prefill.*/measurement/authority.json'))
    runs = []
    harness = harness_identity(ROOT)
    for path in paths:
        if supplement:
            from dev.strategy.prefill_supplement import previous_root, check_harness
            root = previous_root(path.parent)
        else:
            root = prior_root(path.parent)
        authority, _ = read(root, 'authority.json')
        plan, _ = read(root, 'plan/plan.json')
        runs.append(dict(previous=str(root), device=authority['resource']['device'],
                         partition=plan['partition'], cache=authority['resource']['cache'],
                         authority_sha256=digest(authority), rules_supplement=supplement))
        if supplement: check_harness(authority['harness'], harness, ROOT)
        else: source_delta(authority['harness'], harness, 'harness')
    if not runs: raise ValueError('no matching per-shard prefill measurement runs found')
    shards = {r['partition']['shards'] for r in runs}
    if shards != {len(runs)} or {r['partition']['shard'] for r in runs} != set(range(len(runs))):
        raise ValueError('previous run has missing/duplicate shards; do not silently reshard it')
    if len({(r['device']['host'], r['device']['pci']) for r in runs}) != len(runs):
        raise ValueError('previous shards did not use distinct physical devices')
    return runs


def bind_devices(runs, visible):
    if len({(d['host'], d['pci']) for _, d in visible}) != len(visible):
        raise ValueError('visible device selectors refer to the same physical device')
    result = []
    for run in runs:
        matches = [(selector, device) for selector, device in visible if device == run['device']]
        if len(matches) != 1:
            raise ValueError('missing original physical card/SDK runtime identity for shard '+
                             str(run['partition']['shard'])+': '+json.dumps(run['device']))
        result.append(dict(run, selector=matches[0][0]))
    if len(result) != len(visible): raise ValueError('devices must exactly cover the old shards')
    return sorted(result, key=lambda r:r['partition']['shard'])


def command(row, output, sdk, jobs, hours):
    parent = (['--supplement-from', row['previous'], '--max-modules', '384'] if row.get('rules_supplement') else
              ['--previous-output', row['previous'], '--upgrade-cublas-events'])
    cmd = [sys.executable, '-u', str(ROOT/'tools/run_kpack_strategy_prefill.py'),
           '--output', str(output/'measurement'), '--sdk', str(sdk), '--cache', row['cache'],
           *parent, '--idle-confirmed',
           '--shard', str(row['partition']['shard']), '--shards', str(row['partition']['shards']),
           '--jobs', str(jobs), '--hours', str(hours)]
    if (output/'measurement').exists(): cmd.append('--resume')
    return cmd


def package(output, rows):
    """Only compact review + status + bounded failure logs; no cache/DSO/fixture."""
    archive = output/('review-'+str(time.time_ns())+'.tgz')
    paths = [output/'request.json', output/'summary.json']
    for row in rows:
        folder = output/('shard-'+str(row['partition']['shard']))
        log = folder/'console.log'
        if log.is_file(): paths.append(log)
        measurement = folder/'measurement'
        if (measurement/'summary.json').is_file():
            review = folder/('review-'+str(time.time_ns())+'.jsonl.xz')
            rc = subprocess.run([sys.executable, str(ROOT/'tools/pack_kpack_strategy_review.py'),
                                 str(folder), '--output', str(review)]).returncode
            if rc == 0: paths.append(review)
            else: print(f'STRATEGY_PREFILL_EXPORT_FAIL shard={row["partition"]["shard"]} rc={rc}', flush=True)
        else:
            # Setup/runtime diagnostics remain available even before a summary.
            paths += sorted(measurement.glob('*.log'))[-3:]
            paths += sorted((measurement/'build-attempts').glob('*/*.log'))[-5:]
    with tarfile.open(archive, 'x:gz') as out:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(output):
                raise ValueError('review path escapes the new run')
            if not path.is_file(): continue
            with path.open('rb') as src:
                if path.suffix == '.log': src.seek(max(0, path.stat().st_size-262144))
                data = src.read()
            entry = tarfile.TarInfo(str(path.relative_to(output)))
            entry.size = len(data); out.addfile(entry, io.BytesIO(data))
    print('results='+str(archive), flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--supplement-rules', action='store_true',
                        help='append new blind-rule configurations to the reviewed ordinary-event screen')
    parser.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    parser.add_argument('--devices', default='1,2,3,4,5,6,7')
    parser.add_argument('--run-root', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, help='new directory, or existing directory with --resume')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--jobs', type=int, default=192)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--idle-confirmed', action='store_true')
    args = parser.parse_args()
    if not args.idle_confirmed: parser.error('confirm all requested cards are idle with --idle-confirmed')
    if not math.isfinite(args.hours) or args.hours <= 0 or args.jobs <= 0: parser.error('positive limits required')
    if args.resume and not args.output: parser.error('--resume needs --output')
    selectors = args.devices.split(',')
    if any(not s.isdigit() for s in selectors) or len(selectors) != len(set(selectors)):
        parser.error('--devices must be distinct nonnegative ordinals')
    previous, sdk = args.previous.resolve(strict=True), args.sdk.resolve(strict=True)
    runs = discover(previous, args.supplement_rules)
    code = ('import json,sys; from pathlib import Path; from tools.probe_kpack_strategy import device_probe; '
            'print(json.dumps(device_probe(Path(sys.argv[1]))))')
    visible = []
    for selector in selectors:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=selector, OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        device = json.loads(subprocess.check_output([sys.executable, '-c', code, str(sdk)],
                                                   env=env, cwd=ROOT, text=True))
        visible.append((selector, device))
    rows = bind_devices(runs, visible)
    if args.output:
        output = args.output.resolve()
        if output == previous or output.is_relative_to(previous): parser.error('output must not modify the old run')
        output.mkdir(parents=True, exist_ok=args.resume)
    else:
        output = Path(tempfile.mkdtemp(prefix='kpack-prefill-supplement-' if args.supplement_rules else
                                      'kpack-prefill-events-', dir=args.run_root.resolve(strict=True)))
    request = dict(previous=str(previous), sdk=str(sdk), rows=rows, harness=harness_identity(ROOT),
                   scope=('ALL_POINTS_NEW_RULE_CONFIGS_REUSE_IDENTICAL_PASS_NOT_MODEL_AB' if args.supplement_rules else
                          'RETAIN_ISOLATED_PASS_REPLACE_CUBLAS_COMPLETE_PENDING_NOT_MODEL_AB'))
    if (output/'request.json').exists() and json.loads((output/'request.json').read_text()) != request:
        raise ValueError('resume device/source/parent changed')
    atomic_json(output/'request.json', request)
    print(f'STRATEGY_PREFILL_MULTI_START run={output} workers={len(rows)} hours_limit={args.hours} '
          f'duration_guaranteed=0 new_tc_compiles={"RESOURCE_PLAN_PENDING" if args.supplement_rules else "0"}', flush=True)
    active, statuses = {}, {}
    start = time.monotonic()
    try:
        for row in rows:
            shard = row['partition']['shard']; folder = output/('shard-'+str(shard)); folder.mkdir(exist_ok=True)
            cmd = command(row, folder, sdk, max(1, args.jobs//len(rows)), args.hours)
            atomic_json(folder/'command.json', cmd)
            with (folder/'console.log').open('ab') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=row['selector'],
                    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            active[shard] = proc
        while active:
            for shard, proc in list(active.items()):
                rc = proc.poll()
                if rc is not None:
                    statuses[str(shard)] = rc; del active[shard]
                    print(f'STRATEGY_PREFILL_SHARD_DONE shard={shard} rc={rc} remaining_continue=1', flush=True)
            atomic_json(output/'summary.json', dict(status='RUNNING' if active else
                'COMPLETE' if all(rc == 0 for rc in statuses.values()) else 'INCOMPLETE',
                processes=statuses, active=list(active), elapsed_seconds=time.monotonic()-start,
                production_changed=False, performance_admitted=False))
            if active:
                print(f'STRATEGY_PREFILL_MULTI_WAIT alive={len(active)}/{len(rows)} '
                      f'elapsed_minutes={(time.monotonic()-start)/60:.1f} logs={output}/shard-*/console.log', flush=True)
                for shard in active:
                    with (output/('shard-'+str(shard))/'console.log').open('rb') as log:
                        log.seek(max(0, log.seek(0, 2)-8192))
                        lines = log.read().decode('utf-8', errors='replace').splitlines()
                    progress = [line for line in lines if line.startswith('STRATEGY_PREFILL_')]
                    if progress: print(f'  shard={shard} {progress[-1]}', flush=True)
                time.sleep(30)
    except BaseException as exc:
        atomic_json(output/'summary.json', dict(status='INTERRUPTED', error=repr(exc),
            processes=statuses, interrupted=list(active), elapsed_seconds=time.monotonic()-start,
            production_changed=False, performance_admitted=False))
        raise
    finally:
        # Only processes started above, each in its own session. Do not touch
        # any other GPU jobs; SIGINT still releases our children's device locks.
        for proc in active.values():
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for proc in active.values():
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=10)
        package(output, rows)
    return int(any(rc != 0 for rc in statuses.values()))


if __name__ == '__main__':
    raise SystemExit(main())
