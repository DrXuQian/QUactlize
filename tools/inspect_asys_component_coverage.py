#!/usr/bin/env python3
"""Inspect/reparse existing Asys captures without rerunning the GPU workload."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.prefill_asys.trace import activities, runtime_activities
from dev.strategy.asys_components import parse_component_rows


def inspect(attempt, *, allow_terminal_short=False):
    attempt = Path(attempt).resolve(strict=True)
    captured = json.loads((attempt / 'capture.json').read_text())
    if 'component_records' not in captured and 'data' in captured:
        captured = captured['data']
    schedule = captured['trace_schedule']
    records = captured['component_records']
    rows = activities(attempt / 'trace.sqlite', launch_shared=True)
    kernels = [r for r in rows if r['kind'] == 'kernel']
    apis = runtime_activities(attempt / 'trace.sqlite')
    expected = Counter()
    for item in schedule:
        expected[records[item['component']]['identity']['node']['symbol']] += item['kernels']
    actual = Counter(r['symbol'] for r in kernels)
    protocol = captured.get('protocol', dict(rounds=6, samples=5))
    result = dict(attempt=str(attempt), point=captured.get('point'), phase=captured.get('phase'),
        expected_kernels=sum(expected.values()), actual_kernels=len(kernels),
        non_kernel_activities=len(rows)-len(kernels), schedule_samples=len(schedule),
        components=len(records), runtime_api_counts=dict(Counter(a['name'] for a in apis)),
        symbols=[dict(symbol=s, expected=expected[s], actual=actual[s],
                      difference=actual[s]-expected[s]) for s in sorted(set(expected) | set(actual), key=str)],
        gpu_work=False, performance_admitted=False, production_changed=False,
        scope='EXISTING_TRACE_COMPONENT_COVERAGE_ONLY_NOT_NUMERICAL_OR_TIMING_ADMISSION')
    try:
        parsed=parse_component_rows(records, schedule, kernels, {}, protocol['rounds'], protocol['samples'],
                                    runtime_apis=apis,allow_terminal_short=allow_terminal_short)
    except (KeyError, TypeError, ValueError) as error:
        result.update(status='COVERAGE_OR_CONTRACT_FAIL', error=str(error))
    else:
        exceptions=[r['terminal_activity_exception'] for r in parsed.values() if 'terminal_activity_exception' in r]
        result.update(status='COMPONENT_COVERAGE_MATCHES_WITH_TERMINAL_EXCEPTION' if exceptions else
                      'COMPONENT_COVERAGE_MATCHES',terminal_activity_exceptions=exceptions)
    return result


def reparse_moe(attempt,output,*,repair_preflight=False):
    from dev.strategy.moe_prefill_search import parse
    from dev.strategy.journal import atomic_json
    from dev.strategy.smallm_asys import revision
    from quactlize.runtime.compiler import sha
    from quactlize.runtime.tuning import digest
    from tools.run_smallm_regressions_asys import sealed
    attempt=Path(attempt).resolve(strict=True)
    if output.exists():raise ValueError('reparse output already exists')
    captured=sealed(attempt/'capture.json')
    if captured['reused_components']:
        raise ValueError('this standalone reparse requires a capture without reused components')
    parser_files=('dev/strategy/asys_components.py','dev/strategy/asys_tail.py',
                  'dev/strategy/moe_prefill_search.py','dev/prefill_asys/trace.py',
                  'tools/inspect_asys_component_coverage.py')
    provenance=dict(parser_revision=revision(ROOT),parser_sources={p:sha(ROOT/p) for p in parser_files},
        measurement_authority=captured['authority'],capture_sha256=sha(attempt/'capture.json'),
        sqlite_sha256=sha(attempt/'trace.sqlite'),
        trace_sha256=sha(attempt/'trace.asysrep') if (attempt/'trace.asysrep').exists() else None,
        scope='OFFLINE_REPARSE_NOT_CAMPAIGN_CHECKPOINT')
    run=None
    if repair_preflight:
        run=sealed(attempt.parents[2]/'run.json')
        provenance.update(run['source_revision'],source_sha256=run['source']['sha256'],
            harness_sha256=run['harness']['sha256'],images_sha256=digest(run['images']),
            device=captured['authority']['device'],
            sdk=run['sdk'],asys_sha256=run['asys_sha256'],timing_method='ASYS_GPU_ACTIVITY',
            scope='OFFLINE_REPARSE_ORIGINAL_MEASUREMENT_AUTHORITY')
    data=parse(captured,{},activities(attempt/'trace.sqlite',launch_shared=True),
               runtime_activities(attempt/'trace.sqlite'),provenance)
    data.update(gpu_work=False,performance_admitted=False,production_changed=False,
                scope=provenance['scope'])
    atomic_json(output,data)
    result=dict(output=str(output),status=data['status'],gpu_work=False,
                terminal_activity_exceptions=data['terminal_activity_exceptions'])
    if repair_preflight:
        root=attempt.parents[2]
        with root.with_name(root.name+'.lock').open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result['preflight']=repair_moe_preflight(attempt,captured,data,run)
    return result


def repair_moe_preflight(attempt,capture,data,run):
    """Publish only this passed old batch; never edit run.json or raw evidence."""
    from quactlize.runtime.compiler import sha
    from tools import run_decode_refresh as base
    from tools.run_smallm_regressions_asys import sealed
    from tools.run_moe_prefill_sweep import batch_path,cache_folder,saved
    root=attempt.parents[2];job=sealed(attempt/'job.json');point=job['point']
    authority=capture['authority'];shard=authority['shard']
    if (data['status']!='PASS' or job['run']!=run['sha256'] or capture['job']!=job['sha256'] or
            capture['point']!=point['id'] or capture['phase']!='screen' or job['phase']!='screen' or
            set(capture['arms'])!=set(job['arms']) or len(job['arms'])!=1 or
            next(a for a in point['arms'] if a['id']==job['arms'][0])['recipe']['kind']!='full' or
            authority!=dict(run=run['sha256'],shard=shard,device=run['devices'][shard],phase='screen')):
        raise ValueError('capture is not the original Full preflight batch')
    process=json.loads((attempt/'profile.process.json').read_text())
    if process['returncode']!=0:raise ValueError('profiled child did not exit successfully')
    folder=root/('gpu-'+run['cards'][shard]);path=batch_path(folder,point['id'],'screen',job['arms'])
    gate=root/'preflight.json'
    if gate.exists():
        old=sealed(gate)
        if old['status']!='PASS' or old['run']!=run['sha256']:raise ValueError('existing preflight differs')
        if saved(path,run) is None:raise ValueError('existing preflight has no passed batch')
        return dict(status='ALREADY_REPAIRED',path=str(gate))
    if path.exists():raise ValueError('batch already exists; keep it rather than overwrite')
    evidence={str(attempt/name):sha(attempt/name) for name in
        ('job.json','capture.json','profile.process.json','trace.sqlite','trace.asysrep')
        if (attempt/name).is_file()}
    for cid,row in data['component_records'].items():
        if row['kind']=='producer':continue
        target=cache_folder(folder,'screen',point)/(cid+'.json')
        if not target.exists():base.sealed(target,dict(record=row,run=run['sha256'],evidence=evidence))
    base.sealed(path,dict(status='PASS',run=run['sha256'],point=point['id'],phase='screen',
        data=data,evidence=evidence,clock_observation='NOT_SAVED_BY_ORIGINAL_FAILED_PARSE'))
    base.sealed(gate,dict(status='PASS',run=run['sha256'],point=point['id'],reset=data['reset'],
        scope='FULL_RESET_CAPTURE_GATE_NOT_CAMPAIGN',parser_revision=data['latency_provenance']['parser_revision']))
    return dict(status='REPAIRED',path=str(gate),batch=str(path),gpu_work=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--allow-terminal-short',action='store_true',
                        help='use the observed count for a single short final sample')
    parser.add_argument('--reparse-moe-output',type=Path,
                        help='write a new offline result; never overwrites the old capture/run')
    parser.add_argument('--repair-moe-preflight',action='store_true',
                        help='reparse the saved Full preflight and let the original run resume')
    args = parser.parse_args()
    try:
        if args.repair_moe_preflight:
            args.allow_terminal_short=True
            if args.reparse_moe_output is None:
                args.reparse_moe_output=args.attempt/('reparsed-'+str(time.time_ns())+'.json')
        if args.reparse_moe_output and not args.allow_terminal_short:
            parser.error('--reparse-moe-output requires --allow-terminal-short')
        result = inspect(args.attempt,allow_terminal_short=args.allow_terminal_short)
        if args.reparse_moe_output and result['status'].startswith('COMPONENT_COVERAGE_MATCHES'):
            result['moe_reparse']=reparse_moe(args.attempt,args.reparse_moe_output,
                                             repair_preflight=args.repair_moe_preflight)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.repair_moe_preflight and 'moe_reparse' in result:
        repair=result['moe_reparse']
        print('MOE_PREFILL_REPAIR '+json.dumps(dict(repair['preflight'],
            expected_kernels=result['expected_kernels'],observed_kernels=result['actual_kernels'],
            output=repair['output'],gpu_work=False),allow_nan=False))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'].startswith('COMPONENT_COVERAGE_MATCHES') else 1


if __name__ == '__main__':
    raise SystemExit(main())
