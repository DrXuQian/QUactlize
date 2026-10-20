#!/usr/bin/env python3
"""Fresh large-M finalist comparison; independent of all previous Box run paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.prefill_closure import require
from dev.strategy.prefill_confirmation import ROUNDS
from dev.strategy.prefill_plan import family
from dev.strategy.prefill_refresh import make_plan, bind, collect
from dev.strategy.provenance import harness_identity, source_identity, verify_unchanged
from dev.strategy.refresh_clock import ADVISORY
from dev.strategy.resource_pilot import current_module, verified_plan
from dev.strategy.resources import compiler, NativeProbe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.confirm_kpack_prefill import read, save_once, package


def env_for(card):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=card, OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def command(args, *extra, output=None):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--sdk', str(args.sdk),
            '--output', str(output or args.output), '--l2-bytes', str(args.l2_bytes), *extra]


def stop_group(proc):
    """Terminate only this runner's owned process group, never unrelated jobs."""
    if proc.poll() is not None:
        return
    require(os.getpgid(proc.pid) == proc.pid, 'refusing to terminate an unowned process group')
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            require(os.getpgid(proc.pid) == proc.pid, 'child process group changed')
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_logged(argv, path, *, env=None, held=None, phase='setup', own_group=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                pass_fds=(held.fileno(),) if held else (), start_new_session=own_group)
        start = time.monotonic()
        try:
            while True:
                try:
                    rc = proc.wait(timeout=30)
                    if rc:
                        with path.open('rb') as stream:
                            stream.seek(max(0, path.stat().st_size-12000))
                            print(stream.read().decode(errors='replace'), flush=True)
                    return rc
                except subprocess.TimeoutExpired:
                    print(f'PREFILL_REFRESH_WAIT phase={phase} minutes={(time.monotonic()-start)/60:.1f} '
                          f'log={path}', flush=True)
        finally:
            if proc.poll() is None:
                if own_group:
                    stop_group(proc)
                else:
                    # Family children inherit the worker's group. The top-level
                    # controller terminates that whole group on interruption.
                    proc.terminate()
                    proc.wait(timeout=10)


def probe(args):
    from tools.run_decode_refresh import device
    from tools.kpack_bf16_providers import Cublas, DeepGemm
    from tools.run_kpack_strategy_pilot import oracle_identity
    from dev.strategy.isolated import check_sdk_graph_abi
    hardware = device(args)
    lock = Path('/tmp')/('quactlize-components-'+digest(hardware['device']['pci'])[:24]+'.lock')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blas = Cublas(args.sdk, None)
        try:
            providers = dict(dense=blas.identity, grouped=DeepGemm().identity)
        finally:
            blas.close()
        value = dict(hardware=hardware, providers=providers, oracle=oracle_identity(),
                     graph_abi=check_sdk_graph_abi(args.sdk))
        save_once(args.output/'probe.json', value)
    return 0


def build_all(args, plan):
    """Compile once before timing. Valid cache receipts survive interruption."""
    from dev.strategy.prefill_build import build as build_helpers, verify as verify_helpers
    from tools.build_kpack_dequant import build as build_dequant
    from tools.run_kpack_strategy_decode import dequant_checked
    from tools.run_kpack_strategy_prefill import stable_build
    compilers = {}
    for spec in plan['modules'].values():
        key = spec['endpoint'], spec['compute']
        if key not in compilers:
            compilers[key] = compiler(spec, args.sdk, args.output/'cache/tc')
    available = next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable:'))
    slots = min(args.jobs, len(plan['modules']), max(1, (available-4*2**30)//(3*2**30)))
    records, failures = {}, {}

    def build(item):
        mid, spec = item
        c = compilers[spec['endpoint'], spec['compute']]
        record = c.build(spec['parent'])
        current_module(record, spec, c)
        require(sha(Path(record['path'])) == record['sha256'], 'compiled image changed')
        record = {k: v for k, v in record.items() if k != 'cache_hit'}
        save_once(args.output/'modules'/(mid+'.json'), record)
        return mid, record

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=slots) as pool:
        pending = {pool.submit(build, item): item[0] for item in plan['modules'].items()}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                mid = pending.pop(future)
                try:
                    _, records[mid] = future.result()
                except Exception as exc:
                    failures[mid] = repr(exc)
            print('PREFILL_REFRESH_BUILD '+json.dumps(dict(completed=len(records), failed=len(failures),
                  total=len(plan['modules']), slots=slots, elapsed_seconds=time.monotonic()-start)), flush=True)
    require(not failures, 'required TC builds incomplete (cache is resumable): '+json.dumps(failures))
    hp = stable_build(args, 'helpers', build_helpers, lambda path, sdk: verify_helpers(sdk, path))
    dq = stable_build(args, 'dequant', build_dequant, dequant_checked)
    images = dict(tc=records, helpers=hp, dequant=dq)
    save_once(args.output/'images.json', images)
    return images


def prepare_worker(args, plan, request, device_receipt, images):
    from tools.probe_kpack_strategy import device_probe
    from quactlize.runtime.native import sdk_identity
    from tools.run_kpack_strategy_pilot import oracle_identity
    from tools.kpack_bf16_providers import Cublas, DeepGemm
    hardware = device_receipt['hardware']
    require(hardware['device'] == device_probe(args.sdk) and hardware['sdk'] == sdk_identity(args.sdk),
            'worker SDK/device changed')
    blas = Cublas(args.sdk, None)
    try:
        providers = dict(dense=blas.identity, grouped=DeepGemm().identity)
    finally:
        blas.close()
    require(providers == device_receipt['providers'] and oracle_identity() == device_receipt['oracle'],
            'provider/oracle changed after preflight')
    authority = dict(plan=plan['plan_sha256'], device=hardware['device'], sdk=hardware['sdk'],
        l2=hardware['l2'], providers=providers, oracle=device_receipt['oracle'],
        graph_abi=device_receipt['graph_abi'], harness=request['harness'], source=request['source'],
        cache=str(args.output/'cache/tc'), helpers_path=str(args.output/'helpers'),
        dequant_path=str(args.output/'dequant'), helpers=images['helpers']['sha256'],
        dequant=images['dequant']['sha256'],
        body_source=digest([images['dequant']['source_hashes'], images['helpers']['identity']]),
        idle='OPERATOR_CONFIRMED', clock=request['clock'], protocol=plan['protocol'],
        previous_timings_reused=False, production_changed=False)
    output = args.output/f'shard-{args.shard}'
    output.mkdir(exist_ok=True)
    save_once(output/'authority.json', authority)
    selected = [p for p in plan['points'] if p['shard'] == args.shard]
    query_ids = set()
    for p in selected:
        from dev.strategy.resources import probe_spec
        from quactlize.dispatch.table import Workload, Recipe
        for arm in p['arms'].values():
            r = Recipe.from_dict(arm['recipe'])
            if r.kind == 'tc':
                query_ids.add(digest(probe_spec(Workload(**p['workload']), r,
                    int(dict(r.parameters)['grid_mode'] >= 2))))
    queries, probes, compilers = {}, {}, {}
    for qid in sorted(query_ids):
        spec = plan['queries'][qid]
        mid = digest(spec['module'])
        record = images['tc'][mid]
        if mid not in probes:
            module = spec['module']
            key = module['endpoint'], module['compute']
            if key not in compilers:
                compilers[key] = compiler(module, args.sdk, args.output/'cache/tc')
            current_module(record, module, compilers[key])
            probes[mid] = NativeProbe(module, record)
            require(all(probes[mid].device[k] == hardware['device'][k] for k in ('ordinal', 'compute_units')),
                    'loaded module on wrong device')
        queries[qid] = probes[mid].query(spec)
        require(queries[qid]['status'] == 'READY', 'required finalist unavailable: '+qid)
    save_once(output/'resources.json', dict(authority=digest(authority), queries=queries,
                                           modules={m: images['tc'][m]['sha256'] for m in probes}))
    points = [bind(p, queries) for p in selected]
    families = {}
    for p in points:
        families.setdefault(family(p), []).append(p)
    for key, rows in families.items():
        arms = [j['arm'] for p in rows for j in p['jobs'].values() if j['role'] == 'tc']
        mids = {a['module'] for a in arms}
        qids = {a['binding']['query'] for a in arms}
        value = dict(points=rows, authority=digest(authority),
            modules={m: images['tc'][m] for m in mids}, specs={m: plan['modules'][m] for m in mids},
            queries={q: queries[q] for q in qids})
        value['sha256'] = digest(value)
        save_once(output/'families'/(key+'.json'), value)
    return output, authority, families


def worker(args):
    plan = verified_plan(read(args.output/'finalists.json'))
    request = read(args.output/'request.json')
    require(request['harness'] == harness_identity(ROOT), 'worker harness changed')
    device_receipt = read(args.output/'devices.json')[args.shard]
    pci = device_receipt['hardware']['device']['pci']
    lock = Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, authority, families = prepare_worker(args, plan, request, device_receipt, read(args.output/'images.json'))
        for key in plan['family_order']:
            if key not in families:
                continue
            points = families[key]
            skipped = set()
            # A failed point gets one attempt per invocation. Continue peers in
            # a new process, not a possibly poisoned CUDA/provider context.
            while True:
                pending = [p for p in points if p['id'] not in skipped and
                           collect(p, output, authority)['status'] == 'INCOMPLETE']
                if not pending:
                    break
                attempt = key+'-'+str(time.time_ns())
                failure = output/'failures'/(attempt+'.json')
                argv = command(args, '--child', '--family', key, '--failure-report', str(failure), output=output)
                for pid in sorted(skipped):
                    argv += ['--skip-point', pid]
                rc = run_logged(argv, output/f'family-{attempt}.log', held=held, phase='family-'+key[:12])
                print(f'PREFILL_REFRESH_FAMILY family={key} rc={rc} remaining_continue=1', flush=True)
                if rc == 0 or not failure.is_file():
                    break
                failed = read(failure).get('point')
                if failed not in {p['id'] for p in pending}:
                    break
                skipped.add(failed)
            results = []
            for p in points:
                value = collect(p, output, authority)
                if value['status'] != 'INCOMPLETE':
                    save_once(output/'results'/(p['id']+'.json'), value)
                results.append(value)
            print('PREFILL_REFRESH_PROGRESS '+json.dumps(dict(family=key,
                  measured=sum(r['status'] != 'INCOMPLETE' for r in results), expected=len(points))), flush=True)
        results = [collect(p, output, authority) for rows in families.values() for p in rows]
        summary = summarize(results)
        atomic_json(output/'summary.json', summary)
        print('PREFILL_REFRESH_SHARD_DONE '+json.dumps(summary), flush=True)
        return int(summary['status'] != 'COMPLETE')


def summarize(results):
    from collections import Counter
    statuses = dict(Counter(r['status'] for r in results))
    measured = sum(r['status'] != 'INCOMPLETE' for r in results)
    return dict(status='COMPLETE' if measured == len(results) else 'INCOMPLETE',
        expected=len(results), measured=measured, selection_statuses=statuses,
        production_changed=False, performance_admitted=False,
        scope='FRESH_SEPARATE_COMPONENTS_NOT_MODEL_TPOT_OR_NATIVE_AB')


def run_workers(args, plan):
    active, statuses = {}, {}
    start = time.monotonic()
    totals = {s: sum(p['shard'] == s for p in plan['points'])*ROUNDS for s in range(len(plan['cards']))}
    initial = {s: len(list((args.output/f'shard-{s}'/'points').glob('*/round-*.json'))) for s in totals}
    try:
        for shard, card in enumerate(plan['cards']):
            output = args.output/f'shard-{shard}'
            output.mkdir(exist_ok=True)
            argv = command(args, '--shard', str(shard))
            save_once(output/'command.json', dict(argv=argv, card=card))
            with (output/'console.log').open('ab') as log:
                active[shard] = subprocess.Popen(argv, cwd=ROOT, env=env_for(card), stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
        while active:
            for shard, proc in list(active.items()):
                if proc.poll() is not None:
                    statuses[shard] = proc.returncode
                    del active[shard]
                    print(f'PREFILL_REFRESH_WORKER shard={shard} rc={proc.returncode} '
                          f'log={args.output}/shard-{shard}/console.log', flush=True)
            if active:
                completed = {s: len(list((args.output/f'shard-{s}'/'points').glob('*/round-*.json'))) for s in totals}
                elapsed = time.monotonic()-start
                remaining = [elapsed/(completed[s]-initial[s])*(totals[s]-completed[s])/60
                             for s in active if completed[s]-initial[s] >= ROUNDS]
                eta = f'{max(remaining):.1f}' if len(remaining) == len(active) else 'UNKNOWN'
                print(f'PREFILL_REFRESH_MULTI_WAIT alive={len(active)}/{len(totals)} '
                      f'rounds={sum(completed.values())}/{sum(totals.values())} '
                      f'elapsed_minutes={elapsed/60:.1f} remaining_minutes={eta} '
                      'eta=OBSERVED_SLOWEST_SHARD_NOT_GUARANTEE', flush=True)
                time.sleep(30)
    finally:
        for proc in active.values():
            stop_group(proc)
        if active:
            atomic_json(args.output/'summary.json', dict(status='INTERRUPTED', processes=statuses,
                                                       performance_admitted=False, production_changed=False))
    results = []
    for p in plan['points']:
        path = args.output/f'shard-{p["shard"]}'/'results'/(p['id']+'.json')
        results.append(read(path) if path.is_file() else dict(point=p['id'], status='INCOMPLETE'))
    result = summarize(results) | dict(processes=statuses, elapsed_seconds=time.monotonic()-start)
    if any(statuses.values()):
        result['status'] = 'INCOMPLETE'
    atomic_json(args.output/'summary.json', result)
    print('PREFILL_REFRESH_DONE '+json.dumps(result), flush=True)
    return int(result['status'] != 'COMPLETE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--devices', default='1,2,3,4,5,6,7')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--l2-bytes', type=int, default=67108864)
    p.add_argument('--expected-ce-ghz', type=float, default=1.7)
    p.add_argument('--smi', type=Path)
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--probe', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--shard', type=int, help=argparse.SUPPRESS)
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--family', help=argparse.SUPPRESS)
    p.add_argument('--skip-point', action='append', default=[], help=argparse.SUPPRESS)
    p.add_argument('--failure-report', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output, args.sdk, args.source = args.output.resolve(), args.sdk.resolve(), ROOT
    if args.child:
        require(args.family and args.failure_report and args.failure_report.parent == args.output/'failures',
                'invalid child failure report')
        from dev.strategy.prefill_confirm_worker import child
        return child(args)
    if args.probe:
        return probe(args)
    if args.shard is not None:
        return worker(args)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('runner interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    require(args.jobs > 0 and args.l2_bytes > 0 and args.expected_ce_ghz == 1.7,
            'positive jobs/L2 and the requested 1700 MHz cohort required')
    cards = args.devices.split(',')
    if args.output.exists():
        require(args.resume, 'output exists; use --resume for this frozen run')
    else:
        require(not args.resume, 'resume output missing')
        args.output.mkdir(parents=True)
    # One coordinator per run. Per-device locks separately prevent overlapping
    # measurements, and are deliberately NOT inherited from this lock.
    run_lock = (args.output/'coordinator.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from quactlize.dispatch.planning import plan_prefill
    source = source_identity(ROOT)
    save_once(args.output/'source.json', source)
    plan = make_plan(plan_prefill(args.output), cards)
    save_once(args.output/'finalists.json', plan)
    print('PREFILL_REFRESH_PLAN '+json.dumps(plan['counts'] | dict(
        cards=cards, rounds=ROUNDS, duration_guaranteed=False, production_changed=False,
        historical_balanced_minutes=max(plan['historical_shard_seconds'])/60)), flush=True)
    if args.plan_only:
        return 0
    require(args.confirm_idle, 'confirm every assigned GPU is otherwise idle')
    from tools.run_decode_refresh import selected_smi, lock_clocks
    from tools.run_simt_split_multi import check_devices
    args.smi = selected_smi(args)
    request = dict(plan=plan['plan_sha256'], sdk=str(args.sdk), cards=cards, source=source['sha256'],
        harness=harness_identity(ROOT), clock=dict(policy=ADVISORY, smi=str(args.smi),
        smi_sha256=sha(args.smi), requested_mhz=1700), idle='OPERATOR_CONFIRMED')
    save_once(args.output/'request.json', request)
    try:
        devices = []
        for index, card in enumerate(cards):
            out = args.output/f'preflight-{index}'
            out.mkdir(exist_ok=True)
            rc = run_logged(command(args, '--probe', output=out),
                            args.output/f'probe-{index}-{time.time_ns()}.log',
                            env=env_for(card), phase='device-and-providers', own_group=True)
            require(rc == 0, 'device/provider preflight failed; no timing started')
            devices.append(read(out/'probe.json'))
        check_devices([d['hardware'] for d in devices])
        save_once(args.output/'devices.json', devices)
        build_all(args, plan)
        verify_unchanged(source, ROOT)
        if args.build_only:
            print('PREFILL_REFRESH_BUILD_ONLY performance_admitted=0 timed_samples=0', flush=True)
            return 0
        lock_clocks(args, [d['hardware'] for d in devices])
        return run_workers(args, plan)
    except BaseException as exc:
        atomic_json(args.output/f'setup-error-{time.time_ns()}.json',
                    dict(error=repr(exc), traceback=traceback.format_exc(),
                         production_changed=False, performance_admitted=False))
        raise
    finally:
        package(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
