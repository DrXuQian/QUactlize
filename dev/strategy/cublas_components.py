"""Import ordinary-call *individual GPU activities*, not event/API spans."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from dev.strategy.components import statistics_checked
from dev.strategy.journal import atomic_json
from quactlize.runtime.tuning import digest


def components(trace, workload, authority, source):
    """A changing inventory remains in the raw trace, not a guessed recipe.

    Consistency is needed only to export one fixed component composition; it
    is NOT used to decide warmup length, tuning completion or optimality.
    """
    result = []
    for phase in ('input', 'provider', 'output'):
        rows = [r for r in trace['ranges'] if r['phase'] == phase]
        def nodes(row):
            return [{k: v for k, v in n.items() if k not in ('start_ns', 'end_ns', 'duration_us')}
                    for n in row['kernels']]
        template = nodes(rows[0])
        if any(nodes(r) != template for r in rows):
            raise ValueError('ordinary cuBLAS has multiple launch inventories; raw samples retained; '
                             'a single component recipe cannot represent them')
        if phase == 'provider' and not any(n['role'] == 'producer' for n in template):
            raise ValueError('provider trace has no recognized GEMM; inspect saved kernel-activities.json')
        for index, node in enumerate(template):
            if node['role'] == 'unclassified':
                raise ValueError('unclassified provider kernel retained for review: '+node['name'])
            rounds = [[r['kernels'][index]['duration_us'] for r in rows if r['round'] == i] for i in range(3)]
            identity = dict(schema='quactlize.profiled-component.v1', authority=authority,
                implementation=source, trace_sha256=trace['sqlite_sha256'], cuda_graph=False,
                contract=asdict(workload), node=node, kind=node['role'], phase=phase, stage=index,
                cache='ROTATING_BF16_WEIGHTS_2P25_L2_RESIDENT_SCRATCH_ORDINARY_CALL')
            cid = digest(identity)
            row = dict(statistics_checked(rounds), identity=identity, component_id=cid,
                kind=node['role'], timing_method='ASYS_GPU_ACTIVITY', timed_kernel_types=1,
                first_use_excluded=True, host_launch_cost_added=False)
            row['scope'] = 'PROFILED_SINGLE_KERNEL'; row['record_sha256'] = digest(row)
            result.append(row)
    return result


def measure_cublas(args, w, authority, held):
    from tools.probe_kpack_cublas_ordinary import run_profile
    path = args.output/'cublas-traces'/(w.key+'-'+str(time.time_ns()))
    path.mkdir(parents=True, exist_ok=False)
    probe = dict(harness=authority['harness'], sdk=authority['resource']['sdk'],
        device=authority['resource']['device'], workload=asdict(w), l2=authority['l2'],
        helpers_path=str(args.output/'helpers'), dequant_path=str(args.output/'dequant'),
        asys=authority['asys'], cuda_graph=False, production_changed=False)
    data = run_profile(args.sdk, path, probe, held)
    trace = json.loads((path/'kernel-activities.json').read_text())
    proof = data['arms']['asys']['result']['data']
    if proof['receipt']['provider'] != authority['providers']['dense']:
        raise ValueError('ordinary provider differs from campaign preflight')
    rows = components(trace, w, dict(sha256=digest(authority)),
                      digest([authority['body_source'], proof['provider_loaded_images']]))
    for row in rows: atomic_json(args.output/'components'/(row['component_id']+'.json'), row)
    return dict(correctness='PASS', components=[r['component_id'] for r in rows],
        stages=[dict(id=r['component_id'], role=r['kind'], median_us=r['median_us'], spread_pct=r['spread_pct'])
                for r in rows], numerical=proof['numerical'], receipt=proof['receipt'],
        profiler_report=data['report'], trace=str(path/'kernel-activities.json'), cuda_graph=False,
        timing='INDIVIDUAL_ASYS_KERNEL_ACTIVITIES_NOT_CALL_EVENTS',
        production_changed=False, performance_admitted=False)
