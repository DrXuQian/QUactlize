"""One family per process; reuse fixtures, isolate stages, alternate finalists."""
import json
from pathlib import Path
import time
import traceback
import uuid

from dev.strategy.journal import atomic_json
from dev.strategy.prefill_closure import require
from dev.strategy.prefill_confirmation import ROUNDS, ordered_jobs
from dev.strategy.provenance import harness_identity
from dev.strategy.resource_pilot import current_module
from dev.strategy.resources import compiler
from quactlize.dispatch.table import Recipe, Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


def child(args):
    from dev.strategy.cold_decode import Ring, Expansion, measure_expansion
    from dev.strategy.prefill_fixture import Context, Helpers, routing
    from dev.strategy.prefill_native import TensorCore
    from dev.strategy.prefill_full import Full
    from dev.strategy.prefill_measure import measure_core, measure_full_expansion
    from dev.strategy.prefill_build import verify as verify_helpers
    from dev.strategy.cublas_ordinary import measure as measure_ordinary
    from dev.strategy.cublas_events import summarize as summarize_events
    from dev.strategy.cublas_components import event_components
    from quactlize.runtime.native import SDK, sdk_identity
    from tools.probe_kpack_strategy import device_probe
    from tools.run_kpack_strategy_decode import dequant_checked
    from tools.run_kpack_strategy_pilot import oracle_identity

    authority = json.loads((args.output/'authority.json').read_text())
    require(authority['harness'] == harness_identity(args.source) and
            authority['device'] == device_probe(args.sdk) and authority['sdk'] == sdk_identity(args.sdk) and
            authority['oracle'] == oracle_identity(),
            'confirmation source/SDK/device changed')
    manifest = json.loads((args.output/'families'/(args.family+'.json')).read_text())
    require(manifest['sha256'] == digest({k:v for k,v in manifest.items() if k != 'sha256'}) and
            manifest['authority'] == digest(authority), 'confirmation manifest changed')
    compilers = {}
    for mid, record in manifest['modules'].items():
        spec = manifest['specs'][mid]; key = (spec['endpoint'], spec['compute'])
        if key not in compilers: compilers[key] = compiler(spec, args.sdk, Path(authority['cache']))
        require(sha(Path(record['path'])) == record['sha256'], 'TC image changed')
        current_module(record, spec, compilers[key])
    helpers_path, dequant_path = Path(authority['helpers_path']), Path(authority['dequant_path'])
    hp, dq = verify_helpers(args.sdk, helpers_path), dequant_checked(dequant_path, args.sdk)
    require(hp['sha256'] == authority['helpers'] and dq['sha256'] == authority['dequant'],
            'confirmation helper image changed')
    family_points = manifest['points']
    skip = set(args.skip_point)
    require(skip <= {p['id'] for p in family_points}, 'unknown skipped point')
    points = [p for p in family_points if p['id'] not in skip]
    require(points, 'empty confirmation family')
    sdk = SDK(args.sdk); helpers = Helpers(helpers_path/'libstrategy_prefill.so')
    ring = metadata_ring = mb = msf = b = sf = full = None
    point_id = None
    try:
        # A failed peer must not change the cold ring for the remaining points
        # or for a later resume of this same family.
        active_bound = min(len(routing(Workload(**p['workload']), r)['active']) for p in family_points for r in (0,1,2))
        ring = Ring(sdk, Workload(**family_points[0]['workload']), authority['l2']['bytes'], minimum_active=active_bound)
        if any(j['role'] == 'sf' for p in family_points for j in p['jobs'].values()):
            metadata_ring = Ring(sdk, Workload(**family_points[0]['workload']), authority['l2']['bytes'],
                                 weights=ring.w, metadata_only=True)
            w = Workload(**min(family_points, key=lambda p:p['workload']['m'])['workload'])
            mb = Context(sdk, metadata_ring, w, helpers); msf = Expansion(mb, dequant_path)
        for point in points:
            point_id = point['id']
            point_root = args.output/'points'/point['id']
            paths = [point_root/f'round-{i}.json' for i in range(ROUNDS)]
            if all(path.is_file() for path in paths):
                continue
            b = Context(sdk, ring, Workload(**point['workload']), helpers)
            if any(j['role'] == 'sf' for j in point['jobs'].values()):
                sf = Expansion(b, dequant_path, gold=msf.gold)
            for index, path in enumerate(paths):
                if path.is_file(): continue
                started = time.monotonic()
                epoch = dict(sha256=digest(authority), round=index, point=point['id'])
                output = point_root/f'measure-{index}-{uuid.uuid4().hex}'
                output.mkdir(parents=True, exist_ok=False)
                records, data = {}, {}
                for job in ordered_jobs(point, index):
                    core = None
                    try:
                        role = job['role']
                        if role == 'sf':
                            value = measure_expansion(mb, mb.w, msf, job['config'], epoch,
                                authority['body_source'], output, separate_expansion=True)
                        elif role in ('full_dequant', 'bf16'):
                            if full is None:
                                full = Full(b, dequant_path, authority['l2']['bytes'], args.sdk)
                                provider = 'dense' if b.w.operation == 'dense' else 'grouped'
                                require(full.provider.identity == authority['providers'][provider],
                                        'installed provider differs from reviewed screen')
                            if role == 'full_dequant':
                                value = measure_full_expansion(b, full, job['config'], epoch,
                                                               authority['body_source'], output)
                            elif b.w.operation == 'dense':
                                proof = measure_ordinary(b, full)
                                timed = summarize_events(proof)
                                rows = event_components(timed, b.w, epoch,
                                    digest([authority['body_source'], proof['provider_loaded_images']]))
                                for row in rows:
                                    atomic_json(output/'components'/(row['component_id']+'.json'), row)
                                value = dict(proof, components=[r['component_id'] for r in rows])
                            else:
                                r = Recipe.make('full', 'full', provider='bf16-default', dequant_config=4)
                                value = measure_core(b, full, r, epoch, authority['body_source'],
                                                     output, config=4, full=True)
                        else:
                            arm = job['arm']; r = Recipe.from_dict(job['recipe'])
                            core = TensorCore(b, arm, manifest['modules'][arm['module']],
                                              manifest['queries'][arm['binding']['query']], sf)
                            value = measure_core(b, core, r, epoch, authority['body_source'], output,
                                                 config=max(0, dict(r.parameters)['dequant_config']))
                        require(value['correctness'] == 'PASS', 'finalist correctness failed')
                        data[job['id']] = value
                        for cid in value['components']:
                            records[cid] = json.loads((output/'components'/(cid+'.json')).read_text())
                        print(f'STRATEGY_CONFIRM_JOB point={point["id"][:12]} round={index+1}/{ROUNDS} '
                              f'role={role} status=PASS', flush=True)
                    finally:
                        if core: core.close()
                row = dict(status='PASS', point=point['id'], round=index, authority=epoch,
                    jobs=data, components=records, wall_seconds=time.monotonic()-started,
                    order=[j['id'] for j in ordered_jobs(point, index)])
                row['record_sha256'] = digest(row); atomic_json(path, row)
            if full: full.close(); full = None
            if sf: sf.close(); sf = None
            b.close(); b = None
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_json(args.failure_report, dict(error=repr(exc), family=args.family, point=point_id,
                                             traceback=traceback.format_exc()))
        return 1
    finally:
        if full: full.close()
        if sf: sf.close()
        if b: b.close()
        if msf: msf.close()
        if mb: mb.close()
        if metadata_ring: metadata_ring.close()
        if ring: ring.close()
