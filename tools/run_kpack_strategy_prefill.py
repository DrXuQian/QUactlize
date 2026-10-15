#!/usr/bin/env python3
"""Finite large-M component screen; family-sharded, resumable, no production writes.

All timers contain one kernel type. The composed sum is NOT measured model
latency. Alternating finalists and original llama wiring A/B remain required.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.strategy.journal import atomic_json
from dev.strategy.plan import make_plan
from dev.strategy.prefill_plan import family, partition, bind_original, execution_plan
from dev.strategy.prefill_retry import PrefillJournal, prior_root, previous_probe, reuse_build, import_passes
from dev.strategy.provenance import source_identity, verify_unchanged, harness_identity
from dev.strategy.resource_pilot import load_probe, verified_plan, current_module
from dev.strategy.resources import inventory, compiler
from dev.strategy.components import recommendation
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.tuning import digest
from tools.probe_kpack_strategy import device_probe
from tools.run_kpack_strategy_decode import records_for, dequant_checked, l2_authority
from tools.run_kpack_strategy_pilot import oracle_identity


def jobs(point):
    helpers = [dict(id=digest(['prefill-sf', c]), recipe=None, roles=['sf'], config=c)
               for c in point['sf_configs']]
    helpers += [dict(id=digest(['prefill-full', c]), recipe=None, roles=['full_dequant'], config=c)
                for c in point['full_configs']]
    if point['full_configs']:
        helpers.append(dict(id=digest('prefill-bf16-provider'), recipe=None, roles=['bf16'], config=-1))
    arms = [a for a in point['arms'] if a['recipe']['kind'] == 'tc']
    arms.sort(key=lambda a: digest(['prefill-screen', point['id'], a['id']]))
    return helpers + arms


def groups(points):
    result = {}
    for p in points: result.setdefault(family(p), []).append(p)
    return dict(sorted(result.items()))


def freeze(args):
    path = args.output/'plan/plan.json'
    if path.exists():
        p = verified_plan(json.loads(path.read_text()))
        verify_unchanged(p['source_identity'], ROOT)
        if p['partition'] != dict(shard=args.shard, shards=args.shards):
            raise ValueError('resume shard changed')
        return p
    before = source_identity(ROOT)
    p = make_plan('prefill')
    all_points = p['points']
    p['total_points'] = len(all_points)
    p['total_families'] = len(groups(all_points))
    p['partition'] = dict(shard=args.shard, shards=args.shards)
    p['points'] = partition(all_points, args.shard, args.shards)
    if not p['points']: raise ValueError('empty family partition; decrease shard count')
    bind_original(p['points'], args.output)
    p['source_identity'] = verify_unchanged(before, ROOT)
    p['plan_sha256'] = digest({k: v for k, v in p.items() if k != 'plan_sha256'})
    atomic_json(path, p)
    return p


def wait(command, log, *, held=None, phase='setup'):
    with log.open('x') as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
            pass_fds=(held.fileno(),) if held else ())
        start = time.monotonic()
        while True:
            try:
                rc = p.wait(timeout=30)
                if rc:
                    print(f'STRATEGY_PREFILL_CHILD_FAILURE phase={phase} rc={rc} log={log}', flush=True)
                    with log.open('rb') as failure:
                        failure.seek(max(0, log.stat().st_size-12288))
                        print(failure.read().decode('utf-8', errors='replace'), flush=True)
                return rc
            except subprocess.TimeoutExpired:
                print(f'STRATEGY_PREFILL_WAIT phase={phase} minutes={(time.monotonic()-start)/60:.1f} log={log}', flush=True)


def stable_build(args, name, build, verify):
    target = args.output/name
    if not target.exists():
        attempts = args.output/'build-attempts'; attempts.mkdir(exist_ok=True)
        attempt = attempts/(name+'-'+uuid.uuid4().hex)
        build(args.sdk, attempt)
        verify(attempt, args.sdk)
        if target.exists(): raise ValueError('concurrent build publication')
        attempt.rename(target)
    return verify(target, args.sdk)


def child(args):
    from dev.strategy.cold_decode import Ring, Expansion, measure_expansion
    from dev.strategy.prefill_fixture import Context, Helpers, routing
    from dev.strategy.prefill_native import TensorCore
    from dev.strategy.prefill_full import Full
    from dev.strategy.prefill_measure import measure_core, measure_full_expansion
    from dev.strategy.prefill_build import verify as verify_helpers
    from quactlize.runtime.native import SDK, sdk_identity
    authority = json.loads((args.output/'authority.json').read_text())
    if (harness_identity(ROOT) != authority['harness'] or device_probe(args.sdk) != authority['resource']['device']
            or sdk_identity(args.sdk) != authority['resource']['sdk']):
        raise ValueError('prefill child source/SDK/device changed')
    manifest = json.loads((args.output/'families'/(args.family+'.json')).read_text())
    if manifest['sha256'] != digest({k: v for k, v in manifest.items() if k != 'sha256'}) or (
            manifest['authority_sha256'] != digest(authority)):
        raise ValueError('prefill family manifest changed')
    points = manifest['points']
    cs = {}
    for mid, record in manifest['modules'].items():
        spec = manifest['specs'][mid]; key = (spec['endpoint'], spec['compute'])
        if key not in cs: cs[key] = compiler(spec, args.sdk, args.cache)
        current_module(record, spec, cs[key])
    dequant_checked(args.output/'dequant', args.sdk)
    verify_helpers(args.sdk, args.output/'helpers')
    journal = PrefillJournal(args.output/'arms', dict(sha256=digest(authority)), args.invocation)
    sdk = SDK(args.sdk); helpers = Helpers(args.output/'helpers/libstrategy_prefill.so')
    ring = metadata_ring = b = mb = sf = msf = full = None
    sf_results, full_results = {}, {}
    try:
        if time.monotonic() >= args.deadline: return 75
        w0 = Workload(**points[0]['workload'])
        start = time.monotonic()
        # Freeze a bound over EVERY actual fixture/replay, not a guessed
        # top8 footprint when large-M routes touch hundreds of experts.
        active_bound = min(len(routing(Workload(**p['workload']), repeat)['active'])
                           for p in points for repeat in (0, 1, 2))
        ring = Ring(sdk, w0, authority['l2']['bytes'], minimum_active=active_bound)
        ring.receipt['active_bound_scope'] = 'ALL_FAMILY_WORKLOADS_AND_CORRECTNESS_REPEATS_0_1_2'
        if any(p['sf_configs'] for p in points):
            metadata_ring = Ring(sdk, w0, authority['l2']['bytes'], weights=ring.w, metadata_only=True)
            # SF expansion consumes metadata only; use the smallest M in the
            # family for unused caller allocations, never a synthetic timing.
            wm = Workload(**min(points, key=lambda p: p['workload']['m'])['workload'])
            mb = Context(sdk, metadata_ring, wm, helpers)
            msf = Expansion(mb, args.output/'dequant')
        print('STRATEGY_PREFILL_FIXTURE '+json.dumps(dict(family=args.family,
              seconds=time.monotonic()-start, ring=ring.receipt)), flush=True)
        for point in points:
            if not any(journal.due(point, j) for j in jobs(point)): continue
            w = Workload(**point['workload'])
            b = Context(sdk, ring, w, helpers)
            if point['sf_configs']: sf = Expansion(b, args.output/'dequant', gold=msf.gold)
            for job in jobs(point):
                if not journal.due(point, job): continue
                if time.monotonic() >= args.deadline: return 75
                journal.start(point, job); start = time.monotonic()
                core = None
                try:
                    role = job['roles'][0]
                    if role == 'sf':
                        c = job['config']
                        if c not in sf_results:
                            sf_results[c] = measure_expansion(mb, mb.w, msf, c, journal.authority,
                                authority['body_source'], args.output, separate_expansion=True)
                        data = sf_results[c]
                    elif role in ('full_dequant', 'bf16'):
                        domain_key = (tuple(map(int, routing(w, 0)['active'])), job['config'])
                        if role == 'full_dequant' and domain_key in full_results:
                            journal.finish(point, job, 'PASS', data=full_results[domain_key])
                            print(f'STRATEGY_PREFILL_HELPER_REUSE point={point["id"][:12]} '
                                  f'config={job["config"]} domain=EXACT_ACTIVE_IDS', flush=True)
                            continue
                        if full is None:
                            full = Full(b, args.output/'dequant', authority['l2']['bytes'], args.sdk)
                            provider_kind = 'dense' if w.experts == 1 else 'grouped'
                            if full.provider.identity != authority['providers'][provider_kind]:
                                raise ValueError('installed BF16 provider changed after preflight')
                        # An installed BLAS may choose an internal split count
                        # not exposed by its ABI. Do not reuse that reducer
                        # across K or another provider workload by name alone.
                        provider_source = digest([authority['body_source'], full.provider.identity, w.key])
                        if role == 'full_dequant':
                            data = measure_full_expansion(b, full, job['config'], journal.authority,
                                authority['body_source'], args.output)
                            full_results[domain_key] = data
                        else:
                            r = Recipe.make('full', 'full', provider='bf16-default', dequant_config=-1)
                            data = measure_core(b, full, r, journal.authority, provider_source,
                                args.output, config=point['full_configs'][0], full=True)
                    else:
                        r = Recipe.from_dict(job['recipe'])
                        core = TensorCore(b, job, manifest['modules'][job['module']],
                            manifest['queries'][job['binding']['query']], sf)
                        canonical = Recipe.make(r.kind, r.route, **(dict(r.parameters)|{'dequant_config': -1}))
                        data = measure_core(b, core, canonical, journal.authority,
                                            authority['body_source'], args.output)
                        core.close(); core = None
                    journal.finish(point, job, 'PASS', data=data)
                    print(f'STRATEGY_PREFILL_JOB point={point["id"][:12]} job={job["id"][:12]} '
                          f'status=PASS seconds={time.monotonic()-start:.3f}', flush=True)
                except Exception as exc:
                    traceback.print_exc(); journal.finish(point, job, 'FAIL', error=repr(exc)); return 1
                finally:
                    if core: core.close()
            if full: full.close(); full = None
            if sf: sf.close(); sf = None
            b.close(); b = None
        return 0
    finally:
        if full: full.close()
        if sf: sf.close()
        if b: b.close()
        if msf: msf.close()
        if mb: mb.close()
        if metadata_ring: metadata_ring.close()
        if ring: ring.close()


def summarize(output, plan, journal):
    from dev.strategy.cold_decode import load_component
    results = []
    for p in plan['points']:
        records, data, missing = {}, {}, []
        helper = {'sf': {}, 'full_dequant': {}}
        provider = []
        for job in jobs(p):
            row = journal.read(p, job)
            if not row or row['status'] != 'PASS':
                missing.append(dict(id=job['id'], status=row['status'] if row else 'PENDING')); continue
            value = row['data']; data[job['id']] = value['components']
            for cid in value['components']:
                path = output/'components'/(cid+'.json'); raw = json.loads(path.read_text())
                if digest(raw['identity']) != cid or raw['identity']['authority'] != row['identity']['authority']:
                    raise ValueError('prefill component identity differs')
                records[cid] = load_component(path, raw['identity'])
            role = job['roles'][0]
            if role in helper: helper[role][job['config']] = value['components']
            if role == 'bf16': provider = value['components']
        best = {role: min(pool, key=lambda c: sum(records[k]['median_us'] for k in pool[c]), default=None)
                for role, pool in helper.items()}
        arms, roles = {}, {}
        for arm in p['arms']:
            r = Recipe.from_dict(arm['recipe']); aid = arm['id']; roles[aid] = arm['roles']
            components = list(provider if r.kind == 'full' else data.get(aid, []))
            role = 'full_dequant' if r.kind == 'full' else 'sf' if r.route == 'sf' and p['sf_configs'] else None
            if role:
                c = best[role] if 'rule' in roles[aid] else max(0, dict(r.parameters).get('dequant_config', -1))
                components = components+helper[role][c] if components and c in helper[role] else []
            arms[aid] = components
        selection = recommendation(arms, records, roles=roles)
        if missing: selection['recommended'] = 'original'
        result = dict(id=p['id'], workload=p['workload'], status='PASS' if not missing else 'INCOMPLETE',
            missing=missing, selection=selection, best_helpers=best, components=arms,
            scope=plan['scope'], excluded=plan['excluded'],
            admission='SCREEN_ONLY_ALTERNATING_FINALISTS_AND_ORIGINAL_MODEL_AB_REQUIRED')
        atomic_json(output/'cases'/(p['id']+'.json'), result); results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, default=Path('/usr/local/PPU_SDK'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--shard', type=int, default=0); parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=192); parser.add_argument('--hours', type=float, default=10.)
    parser.add_argument('--l2-bytes', type=int, default=0)
    parser.add_argument('--max-modules', type=int, default=256)
    parser.add_argument('--max-arms', type=int, default=160000)
    parser.add_argument('--plan-only', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--previous-output', type=Path,
                        help='stopped prefill run; retry failed/pending jobs without remeasuring its PASS jobs')
    parser.add_argument('--idle-confirmed', action='store_true')
    parser.add_argument('--family', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, default=0., help=argparse.SUPPRESS)
    parser.add_argument('--invocation', default=uuid.uuid4().hex, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.hours) or min(args.hours, args.jobs, args.max_modules, args.max_arms) <= 0:
        parser.error('finite positive limits required')
    if args.family and not re.fullmatch('[0-9a-f]{64}', args.family): parser.error('invalid family ID')
    args.output = args.output.resolve(); args.sdk = args.sdk.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume or bool(args.family))
    request = args.output/'retry-request.json'
    if request.exists():
        original = Path(json.loads(request.read_text())['previous_output'])
        if args.previous_output and prior_root(args.previous_output) != original:
            raise ValueError('retry parent changed')
        args.previous_output = original
    previous = prior_root(args.previous_output) if args.previous_output else None
    if previous:
        if args.output == previous or args.output.is_relative_to(previous):
            raise ValueError('retry needs a separate output; do not modify the original run')
        old = json.loads((previous/'authority.json').read_text())
        args.cache = args.cache or Path(old['resource']['cache'])
        # Reuse the old verified value if the SDK's L2 attribute is zero;
        # l2_authority still queries the device and rejects any disagreement.
        args.l2_bytes = args.l2_bytes or old['l2']['bytes']
        atomic_json(request, dict(previous_output=str(previous)))
    args.cache = (args.cache or args.output/'tc-cache').resolve()
    if args.family: return child(args)
    started = time.monotonic(); frozen = freeze(args); inv = inventory(frozen)
    counts = dict(points=len(frozen['points']), families=len(groups(frozen['points'])),
                  all_points=frozen['total_points'], all_families=frozen['total_families'],
                  **inv['counts'], shard=args.shard, shards=args.shards)
    atomic_json(args.output/'counts.json', counts)
    print('STRATEGY_PREFILL_PLAN '+json.dumps(counts|dict(duration_guaranteed=False,
          production_changed=False, scope='LARGE_M_COMPONENT_SCREEN')), flush=True)
    if args.plan_only: return 0
    if not args.idle_confirmed: parser.error('--idle-confirmed required')
    if counts['modules'] > args.max_modules: raise ValueError('module bound exceeded before compilation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('installed PPU Torch and exactly one visible device required')
    providers = {}
    if any(p['workload']['operation'] == 'indexed' for p in frozen['points']):
        from tools.kpack_bf16_providers import DeepGemm
        providers['grouped'] = DeepGemm().identity  # Fail before TC compilation.
    from dev.strategy.isolated import check_sdk_graph_abi
    from dev.strategy.prefill_build import build as build_helpers, verify as verify_helpers
    from tools.build_kpack_dequant import build as build_dequant
    dev = device_probe(args.sdk)
    lock = Path('/tmp')/('quactlize-components-'+digest(dev['pci'])[:24]+'.lock')
    budget = False
    # The resource probe owns this same device lock. Inheriting an already
    # locked fd does not let it flock a second open of that file. Complete
    # the independently locked probe before locking our measurement phase.
    probe = None
    if previous:
        from quactlize.runtime.native import sdk_identity
        probe = previous_probe(previous, frozen, harness_identity(ROOT), ROOT)
        if (probe['authority']['device'] != dev or probe['authority']['sdk'] != sdk_identity(args.sdk) or
                probe['authority']['cache'] != str(args.cache)):
            raise ValueError('parser-only retry requires the same device/SDK/cache')
        print('STRATEGY_PREFILL_RETRY resource=VERIFIED_PRIOR_PROBE new_resource_queries=0', flush=True)
    else:
        cmd = [sys.executable, '-u', str(ROOT/'tools/probe_kpack_strategy.py'), '--plan', str(args.output/'plan/plan.json'),
            '--output', str(args.output/'resources'), '--sdk', str(args.sdk), '--cache', str(args.cache),
            '--jobs', str(args.jobs), '--max-modules', str(args.max_modules)]
        if (args.output/'resources').exists(): cmd.append('--resume')
        resource_log = args.output/('resources-'+str(time.time_ns())+'.log')
        resource_rc = wait(cmd, resource_log, phase='compile-resources')
        if resource_rc:
            raise ValueError(f'resource phase incomplete rc={resource_rc}; see {resource_log}; completed modules are resumable')
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(p['workload']['operation'] == 'dense' for p in frozen['points']):
            from tools.kpack_bf16_providers import Cublas
            print('STRATEGY_PREFILL_PROVIDER phase=cublas-identity status=START', flush=True)
            blas = Cublas(args.sdk, None)
            providers['dense'] = blas.identity
            blas.close()
            print('STRATEGY_PREFILL_PROVIDER phase=cublas-identity status=PASS', flush=True)
        probe = probe or load_probe(args.output, pilot=False); plan = execution_plan(probe)
        if plan['counts']['arms'] > args.max_arms: raise ValueError('expanded arm limit exceeded')
        atomic_json(args.output/'execution-plan.json', plan)
        records = records_for(probe, plan, args.sdk)
        if previous:
            reuse_build(previous, args.output, 'dequant', args.sdk, dequant_checked)
            reuse_build(previous, args.output, 'helpers', args.sdk, lambda path, sdk: verify_helpers(sdk, path))
        dq = stable_build(args, 'dequant', build_dequant, dequant_checked)
        hp = stable_build(args, 'helpers', build_helpers, lambda path, sdk: verify_helpers(sdk, path))
        l2 = l2_authority(args)
        authority = dict(plan=plan['plan_sha256'], resource=probe['authority'], harness=harness_identity(ROOT),
            oracle=oracle_identity(), graph_abi=check_sdk_graph_abi(args.sdk), l2=l2,
            tc={k: v['sha256'] for k, v in records.items()}, dequant=dq['sha256'], helpers=hp['sha256'],
            providers=providers,
            body_source=digest([dq['source_hashes'], hp['identity']]),
            idle='OPERATOR_CONFIRMED', protocol='3x3_ONE_KERNEL_PER_TIMER_NO_LAUNCH_PENALTY',
            scope=plan['scope'], production_changed=False)
        path = args.output/'authority.json'
        if path.exists() and json.loads(path.read_text()) != authority: raise ValueError('resume authority changed')
        atomic_json(path, authority)
        imported = import_passes(previous, args.output, plan, authority, jobs) if previous else None
        if imported:
            print(f'STRATEGY_PREFILL_RETRY preserved_pass_jobs={imported["passed_jobs"]} '
                  'scope=FAILED_OR_PENDING_ONLY original_receipts=UNCHANGED', flush=True)
        journal = PrefillJournal(args.output/'arms', dict(sha256=digest(authority)), args.invocation)
        families = groups(plan['points'])
        for key, points in families.items():
            arms = [a for p in points for a in p['arms'] if a['recipe']['kind'] == 'tc']
            manifest = dict(points=points, authority_sha256=digest(authority),
                modules={a['module']: records[a['module']] for a in arms},
                specs={a['module']: a['spec'] for a in arms},
                queries={a['binding']['query']: probe['receipts'][a['binding']['query']] for a in arms})
            manifest['sha256'] = digest(manifest)
            atomic_json(args.output/'families'/(key+'.json'), manifest)
        for number, (key, points) in enumerate(families.items()):
            while any(journal.due(p, j) for p in points for j in jobs(p)):
                if time.monotonic()-started >= args.hours*3600: budget = True; break
                before = sum(journal.due(p, j) for p in points for j in jobs(p))
                log = args.output/(f'family-{key}-{time.time_ns()}.log')
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(args.output),
                    '--sdk', str(args.sdk), '--cache', str(args.cache), '--family', key,
                    '--invocation', args.invocation, '--deadline', str(started+args.hours*3600)]
                rc = wait(cmd, log, held=held, phase=f'family-{number+1}/{len(families)}')
                for p in points:
                    for j in jobs(p):
                        row = journal.read(p, j)
                        if row and row['status'] == 'RUNNING' and row['invocation'] == args.invocation:
                            journal.finish(p, j, 'CRASH', error=f'child rc={rc}; log={log}')
                if rc == 75: budget = True; break
                if before == sum(journal.due(p, j) for p in points for j in jobs(p)):
                    print(f'STRATEGY_PREFILL_SETUP_FAIL family={key} rc={rc} log={log} remaining_continue=1', flush=True)
                    break
            elapsed = time.monotonic()-started
            print('STRATEGY_PREFILL_PROGRESS '+json.dumps(dict(families=number+1, total=len(families),
                elapsed_minutes=elapsed/60, remaining_minutes=elapsed/(number+1)*(len(families)-number-1)/60,
                eta='OBSERVED_FAMILY_WALL_ADVISORY', budget_remaining_minutes=max(0, args.hours*60-elapsed/60))), flush=True)
            if budget: break
        verify_unchanged(frozen['source_identity'], ROOT)
        rows = summarize(args.output, plan, journal)
    passed = sum(r['status'] == 'PASS' for r in rows)
    summary = dict(status='SCREEN_COMPLETE' if passed == plan['counts']['requested_points'] else 'INCOMPLETE',
        passed=passed, expected=plan['counts']['requested_points'], blocked=plan['blocked'],
        counts=counts, elapsed_seconds=time.monotonic()-started, budget_reached=budget,
        prior_pass_jobs=imported['passed_jobs'] if imported else 0,
        production_changed=False, performance_admitted=False, full_campaign=False,
        remaining=['ALTERNATING_LARGE_M_FINALISTS', 'ORIGINAL_WIRING_MODEL_AB'],
        scope=plan['scope'], excluded=plan['excluded'])
    atomic_json(args.output/'summary.json', summary)
    print('STRATEGY_PREFILL_DONE '+json.dumps(summary), flush=True)
    return int(summary['status'] != 'SCREEN_COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
