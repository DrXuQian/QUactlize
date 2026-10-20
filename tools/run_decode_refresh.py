#!/usr/bin/env python3
"""Seven-card bounded small-M refresh, fresh component timing, no policy writes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.refresh import make_plan, request, family, assign
from dev.strategy.refresh_gaps import (PHASE as GAP_PHASE, make_plan as make_gap_plan,
                                       query_points as gap_queries, validate_producers)
from dev.strategy.refresh_clock import ADVISORY, observe, parse as parse_clock, request_lock
from dev.strategy.refresh_retry import make_retry, previous_run, cache_root
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from dev.strategy.vector import Compiler as VectorCompiler
from dev.strategy.resources import compiler as tc_compiler
from dev.strategy.followup import validate_confirmation
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.pack_simt_split_review import package
from tools.run_simt_split_multi import check_devices


def read(path):
    return json.loads(Path(path).read_bytes())


def freeze(path, data):
    if path.exists():
        if read(path) != json.loads(json.dumps(data)):
            raise ValueError('immutable run binding changed: '+str(path))
    else:
        atomic_json(path, data)


def sealed(path, data):
    value = {k: v for k, v in data.items() if k != 'sha256'}
    value['sha256'] = digest(value)
    atomic_json(path, value)


def saved(path, authority, point):
    if not path.exists():
        return None
    r = read(path)
    if (r.get('sha256') != digest({k: v for k, v in r.items() if k != 'sha256'}) or
            r.get('authority') != authority or r.get('point') != point['id']):
        raise ValueError('point receipt identity/hash changed: '+str(path))
    if r['status'] == 'PASS':
        validate_confirmation(r['data'])
        validate_producers(point, r['data'])
        expected = {a['id'] for a in point['arms']}
        unavailable = r['data'].get('resource_unavailable', {})
        if (set(r['data']['arms']) & set(unavailable) or
                (set(r['data']['arms']) | set(unavailable)) != expected):
            raise ValueError('completed point lost a requested recipe')
        for arm in point['arms']:
            if arm['id'] not in unavailable:
                continue
            query = unavailable[arm['id']]
            from dev.strategy.resources import probe_spec
            recipe = Recipe.from_dict(arm['recipe'])
            spec = probe_spec(Workload(**point['workload']), recipe,
                              int(dict(recipe.parameters).get('grid_mode', 0) >= 2))
            if (arm['required'] or arm['recipe']['kind'] != 'tc' or
                    query.get('status') != 'UNAVAILABLE' or query.get('rc') != 1 or
                    query.get('runtime_status') != 0 or
                    query.get('request') != spec or not query.get('module_sha256')):
                raise ValueError('missing or invalid resource rejection proof')
        clocks = r['data'].get('clock_observations', [])
        if len(clocks) != 6 or [c['round'] for c in clocks] != list(range(6)):
            raise ValueError('completed point lacks round-boundary clock evidence')
        for c in clocks:
            for side in ('before', 'after'):
                v = c[side]
                checked = parse_clock(v['stdout'], authority['pci'], authority['cu_mhz'],
                                      authority['memory_mhz'],
                                      advisory=authority.get('clock_policy') == ADVISORY)
                if any(v.get(k) != value for k, value in checked.items()):
                    raise ValueError('recorded clock interpretation changed')
    return r


def device(a):
    from tools.run_kpack_llama_ab import device as probe
    value = probe(a)
    if value['device']['compute_units'] != 72 or value['device']['warp_size'] != 32:
        raise ValueError('this refresh compares the ZW810/72-CU cohort, not a device-policy transplant')
    return value


def selected_smi(a):
    if a.smi:
        return a.smi.resolve(strict=True)
    for p in (a.sdk/'ppu-smi/bin/ppu-smi', a.sdk/'bin/ppu-smi'):
        if p.is_file():
            return p.resolve()
    command = shutil.which('ppu-smi')
    if command:
        return Path(command).resolve()
    raise ValueError('PPU-SMI not found; supply --smi or PPU_SMI for read-only clock checks')


def compile_all(a, plan):
    """Single shared build; no timed GPU jobs until all compilers have exited."""
    from tools.build_kpack_dequant import build as build_dequant
    from tools.run_kpack_strategy_decode import dequant_checked
    cache = cache_root(a.output, plan)
    vc = VectorCompiler(a.sdk, cache/'cache/vectors')
    compilers = {}
    for item in plan['tc'].values():
        k = (item['endpoint'], item['compute'])
        if k not in compilers:
            compilers[k] = tc_compiler(item, a.sdk, cache/'cache/tc')
    tasks = [('tc', k, v) for k, v in plan['tc'].items()]
    tasks += [('vector', k, v) for k, v in plan['vectors'].items()]
    records = {'tc': {}, 'vectors': {}}
    prior_records = read(cache/'images.json') if cache != a.output else None
    available = next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()
                     if l.startswith('MemAvailable:'))
    slots = min(a.jobs, len(tasks), max(1, (available-4*2**30)//(2*2**30)))
    def build(task):
        kind, key, item = task
        if kind == 'tc':
            compiler = compilers[item['endpoint'], item['compute']]
            if prior_records is not None:
                old = prior_records['tc'][key]
                expected = digest(dict(identity=compiler.identity, parent=item['parent'],
                                       source=compiler.source(item['parent'], '')))
                if (old['key'] != expected or
                        not (cache/'cache/tc'/expected/'manifest.json').is_file()):
                    raise ValueError('clock retry TC cache/source missing or changed; no rebuild attempted')
            record = compiler.build(item['parent'])
        else:
            if prior_records is not None:
                record = prior_records['vectors'][key]
                vc.verify(item, record)
                record = record | dict(cache_hit=True)
            else:
                record = vc.build(item)
        return kind, key, record
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=slots) as pool:
        pending = {pool.submit(build, task) for task in tasks}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                kind, key, record = future.result()
                records['tc' if kind == 'tc' else 'vectors'][key] = record
            completed = sum(map(len, records.values()))
            print('DECODE_REFRESH_BUILD '+json.dumps(dict(
                completed=completed, total=len(tasks), slots=slots, requested_jobs=a.jobs,
                elapsed_minutes=(time.monotonic()-start)/60)), flush=True)
    dequant = None
    if any(p['sf_configs'] for p in plan['points']):
        pointer = cache/'dequant.json'
        if pointer.exists():
            dequant = Path(read(pointer)['directory'])
            dequant_checked(dequant, a.sdk)
            if cache != a.output:
                freeze(a.output/'dequant.json', read(pointer))
        else:
            if cache != a.output:
                raise ValueError('clock retry expansion cache missing; no rebuild attempted')
            dequant = a.output/'build-attempts'/('dequant-'+uuid.uuid4().hex)
            build_dequant(a.sdk, dequant)
            dequant_checked(dequant, a.sdk)
            atomic_json(pointer, dict(directory=str(dequant)))
    atomic_json(a.output/'images.json', records)
    return records, dequant


def image_identity(records, dequant):
    result = {kind: {k: r['sha256'] for k, r in rows.items()} for kind, rows in records.items()}
    result['dequant'] = sha(dequant/'manifest.json') if dequant else None
    return result


def verify_images(a, plan, records, dequant, points=None):
    vc = VectorCompiler(a.sdk, cache_root(a.output, plan)/'cache/vectors')
    selected = points or plan['points']
    modules = {arm['module']: arm for p in selected for arm in p['arms']}
    for mid, arm in modules.items():
        kind = 'tc' if arm['recipe']['kind'] == 'tc' else 'vectors'
        row = records[kind][mid]
        if sha(row['path']) != row['sha256']:
            raise ValueError('device image changed: '+row['path'])
        if kind == 'vectors':
            vc.verify(arm['spec'], row)
        elif row['parent'] != arm['spec']['parent']:
            raise ValueError('TC parent differs')
    if dequant:
        from tools.run_kpack_strategy_decode import dequant_checked
        dequant_checked(dequant, a.sdk)


def authority_for(run, shard):
    hardware = run['devices'][shard]
    result = dict(run=run['sha256'], shard=shard, pci=hardware['device']['pci'],
                hardware=hardware, cu_mhz=run['cu_mhz'],
                memory_mhz=run['memory_mhz'])
    if 'clock_policy' in run:
        result['clock_policy'] = run['clock_policy']
    return result


def lock_clocks(a, devices):
    """Request once before timing, with all participating device locks held.

    No default restoration: leave the requested CU limit in place and report
    that fact. Never reset another workload's device or change memory/power.
    """
    path = a.output/('clock-lock-'+str(time.time_ns())+'.json')
    receipt = dict(clock_policy=ADVISORY, requested_mhz=a.expected_ce_ghz*1000,
                   status='STARTED', requests=[], observations=[],
                   settings_after_run='REQUEST_LEFT_IN_PLACE',
                   memory_and_power_settings_changed=False)
    atomic_json(path, receipt)
    try:
        with ExitStack() as stack:
            for dev in devices:
                pci = dev['device']['pci']
                held = stack.enter_context((Path('/tmp')/(
                    'quactlize-components-'+digest(pci)[:24]+'.lock')).open('a'))
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for dev in devices:
                pci = dev['device']['pci']
                result = request_lock(a.smi, pci, a.expected_ce_ghz*1000)
                receipt['requests'].append(result)
                atomic_json(path, receipt)
                if result['returncode']:
                    raise ValueError('PPU clock lock request failed: '+result['stdout']+result['stderr'])
                actual = observe(a.smi, pci, a.expected_ce_ghz*1000, advisory=True)
                receipt['observations'].append(actual)
                print('DECODE_REFRESH_CLOCK '+json.dumps(dict(
                    pci=pci, requested_mhz=a.expected_ce_ghz*1000,
                    actual_mhz=actual['cu_mhz'], affects_verdict=False)), flush=True)
        receipt['status'] = 'REQUESTS_ACCEPTED'
    except BaseException as exc:
        receipt.update(status='FAIL', error=str(exc))
        raise
    finally:
        atomic_json(path, receipt)


def bind_point(point, records):
    from dev.strategy.resources import NativeProbe, probe_spec
    from dev.strategy.space import resolved_grid
    w = Workload(**point['workload'])
    result = dict(point, arms=[], resource_unavailable={})
    bindings = {}
    for arm in point['arms']:
        arm = dict(arm)
        kind = 'tc' if arm['recipe']['kind'] == 'tc' else 'vectors'
        record = records[kind][arm['module']]
        query = None
        if kind == 'tc':
            r = Recipe.from_dict(arm['recipe'])
            probe = NativeProbe(arm['spec'], record)
            query = probe.query(probe_spec(w, r, int(dict(r.parameters).get('grid_mode', 0) >= 2)))
            if query['status'] != 'READY':
                if arm['required'] or query.get('runtime_status') != 0:
                    raise ValueError('required or failed TC resource query: '+arm['id']+' '+json.dumps(query))
                result['resource_unavailable'][arm['id']] = query
                continue
            arm['binding'] = dict(launch=resolved_grid(w, r, probe.device['compute_units'], query['occupancy']))
        bindings[arm['id']] = record, query
        result['arms'].append(arm)
    return result, bindings


def family_child(a):
    from dev.smallm_closure.bench import Context
    from dev.strategy.cold_decode import Ring, Expansion, measure_expansion
    from dev.strategy.indexed import point as indexed_point
    from dev.strategy.confirm import confirm_point
    from quactlize.runtime.native import SDK
    plan, run, records = (read(a.output/n) for n in ('plan.json', 'run.json', 'images.json'))
    if digest({k: v for k, v in run.items() if k != 'sha256'}) != run['sha256']:
        raise ValueError('run seal differs')
    if (a.shard is None or not 0 <= a.shard < len(run['assignments']) or
            a.family not in run['assignments'][a.shard]):
        raise ValueError('family belongs to another worker')
    if plan['sha256'] != run['plan'] or digest({k: v for k, v in plan.items() if k != 'sha256'}) != run['plan']:
        raise ValueError('worker plan differs from the frozen run')
    points = [p for p in plan['points'] if family(Workload(**p['workload'])) == a.family]
    auth = authority_for(run, a.shard)
    if device(a) != auth['hardware'] or harness_identity(ROOT) != run['harness']:
        raise ValueError('worker SDK/source/physical device changed')
    dequant = Path(read(a.output/'dequant.json')['directory']) if (a.output/'dequant.json').exists() else None
    if image_identity(records, dequant) != run['images'] or sha(a.smi) != run['smi_sha256']:
        raise ValueError('compiled module or clock-query binary binding changed')
    verify_images(a, plan, records, dequant, points)
    out = a.output/('gpu-'+run['cards'][a.shard])
    points = sorted(points, key=lambda p: (p['workload']['m'] not in (1, 8), p['workload']['m'], p['id']))
    pending = []
    for p in points:
        row = saved(out/'points'/(p['id']+'.json'), auth, p)
        if not row or row['status'] != 'PASS' and row.get('invocation') != a.invocation:
            pending.append(p)
    if not pending:
        return 0
    first = min(pending, key=lambda p: p['workload']['topk'])
    w0 = Workload(**first['workload'])
    sdk = SDK(a.sdk)
    print('DECODE_REFRESH_FIXTURE '+json.dumps(dict(
        family=a.family, points=len(pending), q=w0.qtype, n=w0.n, k=w0.k, experts=w0.experts)), flush=True)
    with ExitStack() as stack:
        ring = Ring(sdk, w0, a.l2_bytes)
        stack.callback(ring.close)
        metadata_b = metadata_sf = None
        configurations = sorted({c for p in pending for c in p['sf_configs']})
        sf_costs = {}
        if configurations:
            metadata_ring = Ring(sdk, w0, a.l2_bytes, weights=ring.w, metadata_only=True)
            stack.callback(metadata_ring.close)
            req = indexed_point(w0)
            req['mode'] = 2 if w0.operation == 'indexed' else 0
            metadata_b = Context(sdk, metadata_ring, req)
            stack.callback(metadata_b.close)
            metadata_sf = Expansion(metadata_b, dequant)
            stack.callback(metadata_sf.close)
            for cfg in configurations:
                measured = measure_expansion(metadata_b, w0, metadata_sf, cfg,
                    auth, run['body_source'], out, separate_expansion=True)
                sf_costs[cfg] = sum(s['median_us'] for s in measured['stages'])
        for p in pending:
            start = time.monotonic()
            print('DECODE_REFRESH_POINT_START point='+p['id']+' arms='+str(len(p['arms'])), flush=True)
            try:
                w = Workload(**p['workload'])
                bound, bindings = bind_point(p, records)
                req = indexed_point(w)
                req['mode'] = 2 if w.operation == 'indexed' else 0
                with ExitStack() as local:
                    b = Context(sdk, ring, req)
                    local.callback(b.close)
                    sf = Expansion(b, dequant, gold=metadata_sf.gold) if p['sf_configs'] else None
                    if sf:
                        local.callback(sf.close)
                    screen = dict(best_sf=min(sf_costs, key=sf_costs.get) if sf_costs else None)
                    shortlist = dict(status='READY', arms=[a['id'] for a in bound['arms']], aliases={})
                    clock = lambda: observe(a.smi, auth['pci'], auth['cu_mhz'], auth['memory_mhz'],
                                            advisory=auth.get('clock_policy') == ADVISORY)
                    data = confirm_point(b, metadata_b, sf, metadata_sf, bound, screen, shortlist,
                        lambda arm: bindings[arm['id']], auth, run['body_source'], round_observer=clock)
                    data['resource_unavailable'] = bound['resource_unavailable']
                    data['sf_helper_selection'] = screen
                validate_confirmation(data)
                validate_producers(p, data)
                for buf in ring.buffers:
                    buf.guard()
                row = dict(status='PASS', point=p['id'], authority=auth, data=data,
                           invocation=a.invocation, elapsed_seconds=time.monotonic()-start)
                sealed(out/'points'/(p['id']+'.json'), row)
                print('DECODE_REFRESH_POINT '+json.dumps(dict(
                    point=p['id'], status='PASS', seconds=row['elapsed_seconds'],
                    best=data['decision']['winner'],
                    baseline_us=data['decision']['statistics']['original']['median_us'],
                    selected_us=data['decision']['statistics'][data['decision']['winner']]['median_us'])), flush=True)
            except Exception as error:
                traceback.print_exc()
                sealed(out/'points'/(p['id']+'.json'), dict(
                    status='FAIL', point=p['id'], authority=auth, invocation=a.invocation,
                    error=str(error), elapsed_seconds=time.monotonic()-start))
                return 1  # Fresh process for other points after a runtime failure.
    return 0


def aggregate(plan, run, output):
    by_family = {key: i for i, keys in enumerate(run['assignments']) for key in keys}
    results = []
    for p in plan['points']:
        shard = by_family[family(Workload(**p['workload']))]
        path = output/('gpu-'+run['cards'][shard])/'points'/(p['id']+'.json')
        row = saved(path, authority_for(run, shard), p)
        result = dict(id=p['id'], workload=p['workload'], status=row['status'] if row else 'MISSING')
        if row and row['status'] == 'PASS':
            decision = row['data']['decision']
            result.update(winner=decision['winner'], decision=decision['status'],
                          baseline_us=decision['statistics']['original']['median_us'],
                          selected_us=decision['statistics'][decision['winner']]['median_us'],
                          unavailable_recipes=len(row['data'].get('resource_unavailable', {})))
        elif row:
            result['error'] = row['error']
        if row and row['status'] == 'PASS':
            result['clock_deviations'] = sum(
                bool(c[side].get('target_differs')) for c in row['data']['clock_observations']
                for side in ('before', 'after'))
        results.append(result)
    counts = Counter(p['status'] for p in results)
    result = dict(status='COMPLETE' if counts['PASS'] == len(results) else 'INCOMPLETE',
                expected=len(results), completed=counts['PASS'], counts=dict(counts), points=results,
                decisions=dict(Counter(p['decision'] for p in results if p['status'] == 'PASS')),
                clock_policy=run.get('clock_policy', 'LEGACY_STRICT'),
                clock_deviations=sum(p.get('clock_deviations', 0) for p in results),
                clock_retry=plan.get('clock_retry'),
                scope=plan['scope'], historical_times_reused=False, production_changed=False,
                no_regression_admitted=False, model_tpot_admitted=False)
    if plan.get('phase') == GAP_PHASE:
        passed = {p['id'] for p in results if p['status'] == 'PASS'}
        cells = plan['profile_gaps']['cells']
        result['profile_gap_coverage'] = dict(
            expected=len(cells), completed=sum(c['point'] in passed for c in cells),
            missing_pairs=sum(c['previous_status'] == 'NOT_MEASURED' for c in cells),
            body_rechecks=sum(c['previous_status'] == 'ELIGIBLE' for c in cells),
            production_admission='PENDING_REVIEW')
    return result


def worker(a):
    run, plan = read(a.output/'run.json'), read(a.output/'plan.json')
    if not 0 <= a.shard < len(run['assignments']):
        raise ValueError('worker outside declared assignments')
    auth = authority_for(run, a.shard)
    directory = a.output/('gpu-'+run['cards'][a.shard])
    directory.mkdir(exist_ok=True)
    lock = Path('/tmp')/('quactlize-components-'+digest(auth['pci'])[:24]+'.lock')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in run['assignments'][a.shard]:
            rows = [p for p in plan['points'] if family(Workload(**p['workload'])) == key]
            def due():
                pending = []
                for p in rows:
                    old = saved(directory/'points'/(p['id']+'.json'), auth, p)
                    if not old or old['status'] != 'PASS' and old.get('invocation') != a.invocation:
                        pending.append(p)
                return pending
            while due():
                before = {p['id'] for p in due()}
                attempt = directory/'attempts'/uuid.uuid4().hex
                attempt.mkdir(parents=True)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(a.output),
                       '--sdk', str(a.sdk), '--smi', str(a.smi), '--l2-bytes', str(a.l2_bytes),
                       '--shard', str(a.shard), '--family', key, '--invocation', a.invocation]
                atomic_json(attempt/'command.json', cmd)
                start = time.monotonic()
                with (attempt/'run.log').open('x') as log:
                    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, pass_fds=(held.fileno(),))
                    try:
                        while p.poll() is None:
                            try:
                                p.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                print(f'DECODE_REFRESH_WAIT family={key} pending_points={len(due())}/{len(rows)} '
                                      f'minutes={(time.monotonic()-start)/60:.1f} log={attempt}/run.log', flush=True)
                    finally:
                        if p.poll() is None:
                            p.terminate()
                            p.wait(timeout=30)
                if p.returncode:
                    with (attempt/'run.log').open('rb') as log:
                        log.seek(max(0, (attempt/'run.log').stat().st_size-8192))
                        tail = '\n'.join(log.read().decode(errors='replace').splitlines()[-12:])
                    print('DECODE_REFRESH_CHILD_FAIL '+json.dumps(dict(
                        family=key, rc=p.returncode, log=str(attempt/'run.log'), diagnostic=tail)), flush=True)
                after = {p['id'] for p in due()}
                if before == after:
                    # Setup failure: do not endlessly rebuild/retry a family.
                    for point in due():
                        sealed(directory/'points'/(point['id']+'.json'), dict(
                            status='FAIL', point=point['id'], authority=auth, invocation=a.invocation,
                            error=f'family setup/process failed rc={p.returncode}; see {attempt}/run.log'))
                    break
    return 0


def progress(plan, run, output, started):
    total = len(plan['points'])
    files = [p for card in run['cards'] for p in (output/('gpu-'+card)/'points').glob('*.json')]
    elapsed = time.monotonic()-started
    completed = sum(read(p)['status'] == 'PASS' for p in files)
    # Wall-time ETA per worker; never estimate from microsecond kernel costs.
    remaining = []
    for i, card in enumerate(run['cards']):
        points = [p for p in plan['points'] if family(Workload(**p['workload'])) in run['assignments'][i]]
        done = [read(output/('gpu-'+card)/'points'/(p['id']+'.json')) for p in points
                if (output/('gpu-'+card)/'points'/(p['id']+'.json')).exists()]
        fresh = [r['elapsed_seconds'] for r in done if r['status'] == 'PASS' and 'elapsed_seconds' in r]
        remaining.append(0 if len(points) == len(done) else
                         sum(fresh[-16:])/len(fresh[-16:])*(len(points)-len(done))/60 if fresh else None)
    return dict(completed=completed, expected=total, elapsed_minutes=round(elapsed/60, 1),
                remaining_minutes=round(max(remaining), 1) if all(v is not None for v in remaining) else None,
                eta='RECENT_POINT_WALL_SLOWEST_WORKER_EXCLUDES_UNSEEN_FIXTURE_COST_NOT_GUARANTEE')


def run(a):
    from dev.llama_ab.contract import inventory
    from quactlize.dispatch.planning import plan_smallm
    from quactlize.runtime.native import sdk_identity
    a.output.mkdir(parents=True, exist_ok=a.resume)
    pp = a.output/'plan.json'
    if pp.exists():
        plan = read(pp)
        if (plan['sha256'] != digest({k: v for k, v in plan.items() if k != 'sha256'}) or
                plan['phase'] != a.phase):
            raise ValueError('frozen plan/phase changed')
        for name, value in plan['sources'].items():
            if sha(ROOT/name) != value:
                raise ValueError('plan source changed')
    else:
        points = gap_queries() if a.phase == GAP_PHASE else [
            p for p in inventory() if p['cohort'] == 'decode']
        originals = plan_smallm(a.output, [request(Workload(**p['workload'])) for p in points])
        atomic_json(a.output/'cpp-originals.json', originals)
        plan = (make_gap_plan(originals['requests']) if a.phase == GAP_PHASE else
                make_plan(originals['requests'], a.phase))
        if a.clock_retry_from:
            plan = make_retry(a.clock_retry_from, plan, harness_identity(ROOT), saved, authority_for)
        atomic_json(pp, plan)
    print('DECODE_REFRESH_PLAN '+json.dumps(dict(plan['counts'], phase=a.phase,
        scope=plan['scope'], q8_new_bodies='NONE' if a.phase == GAP_PHASE else 'DENSE_S1',
        acu=False, historical_times_reused=False,
        hard_timeout=False)), flush=True)
    if a.plan_only:
        return 0
    cards = a.devices.split(',')
    if not a.confirm_idle:
        raise ValueError('confirm all requested GPUs are otherwise idle')
    a.smi = selected_smi(a)
    devices, clocks = [], []
    for card in cards:
        path = a.output/('probe-'+card+'.json')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--output', str(a.output), '--sdk', str(a.sdk),
               '--probe', str(path), '--l2-bytes', str(a.l2_bytes)]
        subprocess.run(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=card), check=True)
        dev = read(path)
        devices.append(dev)
        clocks.append(observe(a.smi, dev['device']['pci'], a.expected_ce_ghz*1000, advisory=True))
    check_devices(devices)
    previous = previous_run(plan)
    if previous:
        for key, value in dict(cards=cards, devices=devices, sdk=sdk_identity(a.sdk),
                               cu_mhz=a.expected_ce_ghz*1000, smi_sha256=sha(a.smi)).items():
            if previous[key] != value:
                raise ValueError('clock retry parent binding changed: '+key)
    # This baseline is diagnostic metadata, not a live-frequency resume gate.
    old_inputs = read(a.output/'inputs.json') if (a.output/'inputs.json').exists() else None
    memory_baseline = (old_inputs or previous or clocks[0])['memory_mhz']
    inputs = dict(plan=plan['sha256'], cards=cards, devices=devices, harness=harness_identity(ROOT),
                  sdk=sdk_identity(a.sdk), cu_mhz=a.expected_ce_ghz*1000,
                  memory_mhz=memory_baseline,
                  smi_sha256=sha(a.smi),
                  assignments=previous['assignments'] if previous else assign(plan, len(cards)),
                  clock_policy=ADVISORY)
    freeze(a.output/'inputs.json', inputs)
    # Snapshot-only clock observations are kept outside immutable input identity.
    atomic_json(a.output/('preflight-clocks-'+str(time.time_ns())+'.json'), clocks)
    records, dequant = compile_all(a, plan)
    if previous:
        images = image_identity(records, dequant)
        for kind in ('tc', 'vectors'):
            if any(previous['images'][kind].get(k) != v for k, v in images[kind].items()):
                raise ValueError('clock retry compiled image changed: '+kind)
        if dequant and images['dequant'] != previous['images']['dequant']:
            raise ValueError('clock retry dequant image changed')
    run_data = inputs | dict(images=image_identity(records, dequant),
                             body_source=digest(inputs['harness']))
    run_data['sha256'] = digest(run_data)
    freeze(a.output/'run.json', run_data)
    if a.build_only:
        print('DECODE_REFRESH_BUILD_ONLY device_timing=NOT_RUN', flush=True)
        return 0
    lock_clocks(a, devices)
    start, children = time.monotonic(), []
    attempt = a.output/'attempts'/str(time.time_ns())
    attempt.mkdir(parents=True)
    try:
        for i, card in enumerate(cards):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(a.output),
                   '--sdk', str(a.sdk), '--smi', str(a.smi), '--l2-bytes', str(a.l2_bytes),
                   '--shard', str(i), '--invocation', a.invocation]
            atomic_json(attempt/('gpu-'+card+'.command.json'), cmd)
            with (attempt/('gpu-'+card+'.log')).open('x') as log:
                children.append(subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ,
                    CUDA_VISIBLE_DEVICES=card, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        last = 0
        while any(p.poll() is None for p in children):
            now = time.monotonic()
            if now-last >= 30:
                print('DECODE_REFRESH_PROGRESS '+json.dumps(progress(plan, run_data, a.output, start) |
                    dict(alive=sum(p.poll() is None for p in children), workers=len(cards),
                         logs=str(attempt/'gpu-*.log'))), flush=True)
                last = now
            time.sleep(1)
    except BaseException:
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in children:
            p.wait()
        raise
    result = aggregate(plan, run_data, a.output)
    result['returncodes'] = [p.returncode for p in children]
    if any(result['returncodes']):
        result['status'] = 'INCOMPLETE'
    result['elapsed_seconds'] = time.monotonic()-start
    atomic_json(a.output/'summary.json', result)
    print('DECODE_REFRESH_DONE '+json.dumps({k: v for k, v in result.items() if k != 'points'}), flush=True)
    return int(result['status'] != 'COMPLETE')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--devices', default='1,2,3,4,5,6,7')
    p.add_argument('--smi', type=Path)
    p.add_argument('--expected-ce-ghz', type=float, default=1.7)
    p.add_argument('--l2-bytes', type=int, default=67108864)
    p.add_argument('--jobs', type=int, default=192)
    p.add_argument('--phase', choices=('all', 'pilot', GAP_PHASE), default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--clock-retry-from', type=Path,
                   help='reviewed strict-clock results; retry only incomplete points in a new run')
    p.add_argument('--confirm-idle', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--probe', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--shard', type=int, help=argparse.SUPPRESS)
    p.add_argument('--family', help=argparse.SUPPRESS)
    p.add_argument('--invocation', default=uuid.uuid4().hex, help=argparse.SUPPRESS)
    a = p.parse_args()
    cards = a.devices.split(',')
    if (not cards or len(set(cards)) != len(cards) or any(not c.isdecimal() or str(int(c)) != c for c in cards) or
            a.jobs < 1 or a.l2_bytes <= 0 or not 0 < a.expected_ce_ghz < 10 or
            a.expected_ce_ghz*1000 != int(a.expected_ce_ghz*1000) or
            a.resume and a.clock_retry_from or a.phase == GAP_PHASE and a.clock_retry_from):
        p.error('invalid worker/jobs/L2/clock parameters')
    a.output = a.output.resolve()
    a.sdk = a.sdk.resolve()
    if a.probe:
        atomic_json(a.probe, device(a))
        return 0
    if a.family:
        return family_child(a)
    if a.shard is not None:
        return worker(a)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run(a)
        finally:
            if a.output.is_dir():
                package(a.output)


if __name__ == '__main__':
    raise SystemExit(main())
