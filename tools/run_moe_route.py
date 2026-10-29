#!/usr/bin/env python3
"""Standalone mm_ids_helper A/B. Tiny build, exact maps, per-stage Asys, no model/TC sweep."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.moe_route.build import build, revision
from dev.moe_route.protocol import Case, workloads, parse
from dev.strategy.journal import atomic_json
from quactlize.runtime.compiler import sha


def read(path): return json.loads(Path(path).read_text())


def verify(receipt):
    for path, value in receipt['binary_sha256'].items():
        if sha(path) != value: raise ValueError('route image changed: '+path)
    for path, value in receipt['libraries'].items():
        if sha(path) != value: raise ValueError('native baseline library changed: '+path)
    for path, value in receipt['sources'].items():
        if sha(ROOT/path) != value: raise ValueError('route sources changed: '+path)


def execute(command, log, env=None):
    atomic_json(log.with_suffix('.command.json'), command)
    with log.open('w') as out:
        result = subprocess.run(command, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        with log.open('rb') as src:
            src.seek(max(0, log.stat().st_size-8192))
            tail = src.read().decode(errors='replace')
        raise RuntimeError(f'rc={result.returncode} log={log}\n{tail}')


def child(a):
    from dev.moe_route.bench import measure
    receipt = read(a.output/'build/build.json'); verify(receipt)
    job = read(a.batch/'job.json')
    data = measure(a.sdk, receipt, [Case(**v) for v in job['cases']], a.rounds, a.samples, 32)
    data['latency_provenance'] = dict(quactlize=revision(ROOT), build_sha256=sha(a.output/'build/build.json'),
                                    batch_sha256=sha(a.batch/'job.json'))
    atomic_json(a.batch/'capture.json', data)
    print('MOE_ROUTE_CAPTURED '+str(a.batch), flush=True)
    return 0


def worker(a):
    from dev.prefill_asys.trace import activities, runtime_activities
    run = read(a.output/'run.json'); receipt = read(a.output/'build/build.json'); verify(receipt)
    cases = run['cases'][a.worker::len(run['devices'])]
    folder = a.output/('gpu-'+run['devices'][a.worker]); folder.mkdir(exist_ok=True)
    failures = 0
    batches = (len(cases)+a.batch_size-1)//a.batch_size
    for i in range(batches):
        batch = folder/('batch-'+str(i)); batch.mkdir(exist_ok=True)
        saved = batch/'result.json'
        if a.resume and saved.exists() and read(saved)['status'] == 'PASS':
            print(f'MOE_ROUTE_REUSE worker={a.worker} batch={i+1}/{batches}', flush=True); continue
        attempt = batch/('attempt-'+str(time.time_ns())); attempt.mkdir()
        atomic_json(attempt/'job.json', dict(cases=cases[i*a.batch_size:(i+1)*a.batch_size]))
        application = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(a.output),
                       '--sdk', str(a.sdk), '--batch', str(attempt), '--rounds', str(a.rounds), '--samples', str(a.samples)]
        profile = [str(a.asys), 'profile', '-t', 'hggc', '--sample', 'none', '--python-sampling', 'false',
                   '--capture-range', 'hggcProfilerApi', '--capture-range-end', 'merge', '--kill', 'none',
                   '--wait', 'primary', '-f', 'false', '-o', str(attempt/'trace'), *application]
        print(f'MOE_ROUTE_BATCH worker={a.worker} batch={i+1}/{batches} points={len(read(attempt/"job.json")["cases"])} log={attempt/"profile.log"}', flush=True)
        try:
            execute(profile, attempt/'profile.log')
            execute([str(a.asys), 'export', '-o', str(attempt/'trace.sqlite'), str(attempt/'trace.asysrep')], attempt/'export.log')
            capture = read(attempt/'capture.json')
            data = parse(capture, activities(attempt/'trace.sqlite', launch_shared=True), runtime_activities(attempt/'trace.sqlite'))
            verify(receipt)
            data.update(latency_provenance=capture['latency_provenance'],
                        trace=str(attempt/'trace.asysrep'), trace_sha256=sha(attempt/'trace.asysrep'),
                        sqlite_sha256=sha(attempt/'trace.sqlite'))
            atomic_json(saved, data)
            print(f'MOE_ROUTE_BATCH_PASS worker={a.worker} batch={i+1}/{batches}', flush=True)
        except Exception as exc:
            failures += 1
            atomic_json(saved, dict(status='FAIL', error=str(exc), attempt=str(attempt)))
            print('MOE_ROUTE_BATCH_FAIL '+json.dumps(dict(worker=a.worker, batch=i, error=str(exc))), flush=True)
    return int(bool(failures))


def main(a):
    if a.output is None:
        a.output = Path(tempfile.mkdtemp(prefix='kpack-moe-route.', dir=os.environ.get('RUN_ROOT', '/workspace' if Path('/workspace').is_dir() else '/tmp')))
    a.output = a.output.resolve()
    if a.batch: return child(a)
    if a.worker is not None: return worker(a)
    cases = [asdict(c) for c in workloads(a.smoke)]
    if a.plan:
        print(json.dumps(dict(points=len(cases), cases=cases, stages='WARP / REGISTER(<=256 rows) / BALLOT+SCALAR(<=512 rows) / HISTOGRAM+PREFIX+SCATTER', production_changed=False), indent=2)); return 0
    if not a.confirm_idle: raise ValueError('use --confirm-idle only after the assigned GPUs are free')
    if not a.llama_source or not a.llama_build: raise ValueError('--llama-source and --llama-build are required for the native A/B wrapper')
    devices = [str(int(v)) for v in a.devices.split(',')]
    if len(set(devices)) != len(devices) or any(int(v)<0 for v in devices): raise ValueError('duplicate/invalid devices')
    a.asys = (a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    plan = dict(cases=cases, devices=devices, sdk=str(a.sdk.resolve()), asys=str(a.asys),
                asys_sha256=sha(a.asys), rounds=a.rounds, samples=a.samples, batch_size=a.batch_size)
    a.output.mkdir(parents=True, exist_ok=True)
    if a.resume:
        if read(a.output/'run.json') != plan: raise ValueError('resume workload/devices/protocol differ')
        verify(read(a.output/'build/build.json'))
    else:
        if (a.output/'run.json').exists(): raise ValueError('use --resume or a fresh --output')
        print('MOE_ROUTE_BUILD '+str(a.output), flush=True)
        build(a.sdk, a.llama_source, a.llama_build, a.output/'build')
        atomic_json(a.output/'run.json', plan)
    processes = []
    print('MOE_ROUTE_START '+json.dumps(dict(output=str(a.output), points=len(cases), devices=devices)), flush=True)
    for i, device in enumerate(devices):
        folder = a.output/('gpu-'+device); folder.mkdir(exist_ok=True)
        log = (folder/('console-'+str(time.time_ns())+'.log')).open('w')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(a.output),
                   '--sdk', str(a.sdk), '--asys', str(a.asys), '--worker', str(i), '--rounds', str(a.rounds),
                   '--samples', str(a.samples), '--batch-size', str(a.batch_size)]
        if a.resume: command.append('--resume')
        proc = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=device), stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log))
    start = time.monotonic()
    while any(p.poll() is None for p, _ in processes):
        time.sleep(10)
        print('MOE_ROUTE_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p, _ in processes),
            elapsed_minutes=(time.monotonic()-start)/60, logs=str(a.output/'gpu-*/console-*.log'))), flush=True)
    returncodes = [p.returncode for p, _ in processes]
    for _, log in processes: log.close()
    results = [read(p) for p in sorted(a.output.glob('gpu-*/batch-*/result.json'))]
    points = [p for r in results if r['status']=='PASS' for p in r['points']]
    complete = len(points)==len(cases) and not any(returncodes)
    report = dict(status='PASS' if complete else 'INCOMPLETE', expected=len(cases), measured=len(points),
                  returncodes=returncodes, regressions=[p for p in points if p['no_regression'] is False],
                  elapsed_seconds=time.monotonic()-start,
                  points=points, production_changed=False, no_regression_admitted=False,
                  scope='STANDALONE_ROUTING_CANDIDATE_SCREEN_NOT_GEMM_OR_MODEL_TPOT')
    atomic_json(a.output/'summary.json', report)
    package = a.output/'review.tgz'
    with tarfile.open(package, 'w:gz') as tar:
        for p in [a.output/'run.json', a.output/'summary.json', a.output/'build/build.json',
                  *a.output.glob('gpu-*/batch-*/result.json'), *a.output.glob('gpu-*/console-*.log')]:
            tar.add(p, arcname=str(p.relative_to(a.output)))
        for r in results:
            if r['status']!='PASS':
                for p in Path(r['attempt']).glob('*.log'): tar.add(p, arcname=str(p.relative_to(a.output)))
    print('MOE_ROUTE_DONE '+json.dumps({k:v for k,v in report.items() if k not in ('points','regressions')}), flush=True)
    print('results='+str(package), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    p.add_argument('--llama-source', type=Path); p.add_argument('--llama-build', type=Path)
    p.add_argument('--output', type=Path); p.add_argument('--devices', default='0'); p.add_argument('--asys', type=Path)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--plan', action='store_true')
    p.add_argument('--resume', action='store_true'); p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--rounds', type=int, default=4); p.add_argument('--samples', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=6); p.add_argument('--worker', type=int); p.add_argument('--batch', type=Path)
    args = p.parse_args()
    if args.rounds<2 or args.samples<1 or args.batch_size<1: p.error('invalid sampling/batch size')
    try: raise SystemExit(main(args))
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
