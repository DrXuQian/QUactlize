#!/usr/bin/env python3
"""Same-config cold-weight Asys AB: b23397ed prototype vs actlize extension."""
import argparse
import collections
import ctypes as C
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.decode_gemv import parse_asys
from tools.decode_gemv.retest_extension import build_reader, identity, load_cases, verdict, write_json

BASELINE = 'b23397ed2efb26d10e66b2dfea0677ae72a083a6'
ORDER = ('baseline', 'extension', 'extension', 'baseline')


def query_l2(sdk, sdk_root):
    # Resolve the enum from the selected SDK, rather than guessing a properties
    # struct's ABI or substituting a device name for its measured cache capacity.
    header = Path(sdk_root)/'include/driver_types.h'
    match = re.search(r'\bhggcDevAttrL2CacheSize\s*=\s*(\d+)', header.read_text())
    if not match:
        raise ValueError(f'cannot resolve hggcDevAttrL2CacheSize in {header}')
    fn = sdk.lib.hggcDeviceGetAttribute
    fn.argtypes, fn.restype = [C.POINTER(C.c_int), C.c_int, C.c_int], C.c_int
    value = C.c_int()
    rc = fn(C.byref(value), int(match[1]), 0)
    if rc or value.value <= 0:
        raise ValueError(f'L2 attribute query failed: rc={rc}, bytes={value.value}')
    return value.value


def cold_plan(case, weight_bytes, l2_bytes, graph_calls, max_ring):
    if min(weight_bytes, l2_bytes, graph_calls, max_ring) <= 0:
        raise ValueError('invalid cold-weight geometry')
    target = (9*l2_bytes+3)//4
    if case['kind'] == 'dense':
        ring = max(2, math.ceil(target/weight_bytes))
        if ring > max_ring:
            raise ValueError(f'cold ring needs {ring} copies, --max-ring={max_ring}')
        footprint = ring*weight_bytes
    else:
        experts, rows = case['experts'], case['m']*case['topk']
        # Never wrap a partially filled expert call onto the next ring. M3/top8
        # uses 240 unique experts, not 256 plus repeated experts at the boundary.
        ring = experts//rows
        footprint = weight_bytes//experts*(ring*rows)
        if not ring or footprint < target:
            raise ValueError('indexed unique expert working set is below 2.25x queried L2')
    calls = math.ceil(graph_calls/ring)*ring
    return dict(ring=ring, graph_calls=calls, unique_weight_bytes=footprint,
                l2_bytes=l2_bytes, minimum_weight_bytes=target,
                regime='ROTATING_WEIGHTS_ONLY_NOT_ALL_INPUTS_COLD')


def inventory(build):
    cases = []
    for delivery in ('direct', 'staged'):
        rows, _ = load_cases(delivery, build/(delivery+'.so'))
        for row in rows:
            row.update(id=delivery+'/'+row['id'], reader=delivery)
        cases.extend(rows)
    return cases


def run_logged(command, log, env):
    with log.open('w') as stream:
        child = subprocess.Popen(list(map(str, command)), env=env, stdout=stream, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while True:
            try:
                rc = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'MMA_AB_WAIT minutes={(time.monotonic()-start)/60:.1f} log={log}', flush=True)
    if rc:
        raise RuntimeError(f'rc={rc}; log={log}\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-35:]))


def device_snapshot(sdk, destination, env):
    binary = shutil.which('ppu-smi', path=env['PATH'])
    if binary is None and (sdk/'ppu-smi/bin/ppu-smi').is_file():
        binary = str(sdk/'ppu-smi/bin/ppu-smi')
    if binary is None:
        destination.write_text('ppu-smi unavailable; idleness is user-confirmed, not independently proven.\n')
        return
    with destination.open('w') as stream:
        try:
            subprocess.run([binary], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=10)
        except subprocess.TimeoutExpired:
            stream.write('\nppu-smi timed out; no performance conclusion from this snapshot.\n')


def export_baseline(destination):
    payload = subprocess.check_output(['git', '-C', str(ROOT), 'archive', BASELINE, 'quactlize'])
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for entry in archive:
            name = PurePosixPath(entry.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0] != 'quactlize':
                raise ValueError('unexpected baseline archive path')
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError('baseline snapshot must contain regular source files only')
            path = destination/entry.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(entry).read())


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'quactlize').rglob('*'))
            if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh', '.inl', '.inc', '.cpp', '.cu')}


def checked_capture(cases, raw, parsed):
    numeric = verdict(cases, raw)
    if numeric['status'] != 'PASS':
        raise ValueError('GGUF oracle incomplete or failed: '+json.dumps(numeric))
    expected = {c['id']: c for c in cases}
    if len(parsed) != len(expected) or {r['case']['id'] for r in parsed} != set(expected):
        raise ValueError('Asys case coverage differs')
    originals = {r['case']['id']: r for r in raw}
    records = []
    for row in parsed:
        case = expected[row['case']['id']]
        original = originals[case['id']]
        if row['status'] != 'OK' or row['case'] != case or original['case'] != case:
            raise ValueError(f'{case["id"]}: Asys status/case differs: {row["status"]}')
        components = row['components']
        count = 1 + int(case['proto']['split'] > 1)
        if len(components) != count or row['kernels_per_call'] != count:
            raise ValueError(f'{case["id"]}: expected main + optional Split-K reducer')
        main_name = 'mma_reader::staged_kernel<' if case['reader'] == 'staged' else 'mma_reader::kernel<'
        if main_name not in components[0]['name'] or (count == 2 and 'reduce' not in components[1]['name']):
            raise ValueError(f'{case["id"]}: unexpected kernel identities: '+repr([c['name'] for c in components]))
        if any(c['us']['n'] != original['samples'] or not math.isfinite(c['us']['mean']) or
               c['us']['min'] <= 0 for c in components):
            raise ValueError(f'{case["id"]}: incomplete/nonpositive/nonfinite Asys component timing')
        cold = original['cold_weights']
        if cold['unique_weight_bytes'] < (9*cold['l2_bytes']+3)//4 or original['graph_calls'] % cold['ring']:
            raise ValueError(f'{case["id"]}: incomplete cold-weight ring')
        main = components[0]['us']['mean']
        reducer = components[1]['us']['mean'] if count == 2 else 0.0
        if not math.isclose(main+reducer, row['call_sum_us']['mean'], rel_tol=1e-9):
            raise ValueError('component sum differs from full-call kernel sum')
        row.update(main_us=main, reducer_us=reducer, total_us=main+reducer,
                   device=original['device'], cold_weights=cold,
                   graph_calls=original['graph_calls'], fixture=original['fixture'],
                   event_us_per_call=original['event_us_per_call'])
        records.append(row)
    return records


def compare(captures, expected, repetitions):
    by_case = {name: collections.defaultdict(list) for name in expected}
    devices = set()
    for capture in captures:
        for row in capture['records']:
            name = row['case']['id']
            if name not in by_case:
                raise ValueError('unexpected timed case '+name)
            by_case[name][capture['arm']].append(row)
            devices.add(tuple(sorted(row['device'].items())))
    if len(devices) != 1:
        raise ValueError('paired captures must use the same physical device')
    results = []
    for name, arms in by_case.items():
        if any(len(arms[a]) != repetitions for a in ('baseline', 'extension')):
            raise ValueError('paired capture count incomplete: '+name)
        recipes = {(r['config'], r['split'], r['samples'], r['graph_calls'],
                    json.dumps(r['cold_weights'], sort_keys=True)) for rows in arms.values() for r in rows}
        geometries = {tuple((c['name'], tuple(c['grid']), tuple(c['block'])) for c in r['components'])
                      for rows in arms.values() for r in rows}
        gold = {r['check']['golden_sha256'] for rows in arms.values() for r in rows}
        fixtures = {json.dumps(r['fixture'], sort_keys=True) for rows in arms.values() for r in rows}
        if len(recipes) != 1 or len(geometries) != 1 or len(gold) != 1 or len(fixtures) != 1:
            raise ValueError('AB recipe/fixture/launch geometry differs: '+name)
        measurements = {}
        for arm, rows in arms.items():
            measurements[arm] = {phase: [r[phase+'_us'] for r in rows] for phase in ('main', 'reducer', 'total')}
            measurements[arm]['median_total_us'] = statistics.median(measurements[arm]['total'])
        old, new = measurements['baseline'], measurements['extension']
        delta = 100*(new['median_total_us']/old['median_total_us']-1)
        separated = (min(new['total']) > max(old['total']) or max(new['total']) < min(old['total']))
        results.append(dict(id=name, case=arms['extension'][0]['case'], measurements=measurements,
                            delta_pct=delta, repeat_ranges_separated=separated,
                            observation='SLOWER_MEDIAN' if delta > 0 else 'NOT_SLOWER_MEDIAN'))
    return dict(status='MEASURED', points=len(results), results=results,
                slower_medians=sum(r['delta_pct'] > 0 for r in results),
                slower_separated_ranges=sum(r['delta_pct'] > 0 and r['repeat_ranges_separated'] for r in results),
                performance_admitted=False, production_changed=False,
                scope='SAME_CONFIG_REFACTOR_AB_NOT_PRODUCTION_OR_MODEL_COMPARISON',
                metric='MEDIAN_OF_CAPTURE_MEAN_FULL_CALL_KERNEL_SUMS_NOT_MINIMUM_SAMPLE',
                idle='USER_CONFIRMED_NOT_INDEPENDENTLY_PROVEN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=2, help='ABBA blocks; two means four captures per arm')
    parser.add_argument('--samples', type=int, default=192, help='minimum whole-ring calls per case/capture')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confirm-idle', action='store_true', help='confirm the selected card is reserved for this test')
    parser.add_argument('--resume', action='store_true', help='reuse completed builds/captures from --output')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    if min(args.jobs, args.rounds, args.samples) <= 0 or args.device < 0:
        parser.error('jobs/rounds/samples must be positive; device must be nonnegative')
    if not (args.plan_only or args.build_only or args.confirm_idle):
        parser.error('reserve one card and pass --confirm-idle; this script does not kill/wait for other jobs')
    if args.resume and not args.output:
        parser.error('--resume requires --output')
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='qz-mma-ab-'))
    if args.output:
        try:
            output.mkdir(parents=True, exist_ok=args.resume)
        except FileExistsError:
            parser.error(f'--output exists: {output}; use --resume or a new directory')
    sdk = args.sdk.resolve()
    env = dict(os.environ, PPU_SDK=str(sdk), CUDA_VISIBLE_DEVICES=str(args.device), QGEMV_SRC=str(ROOT))
    env['LD_LIBRARY_PATH'] = str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(sdk/'bin')+':'+str(sdk/'asight/bin')+':'+env.get('PATH', '')
    summary = dict(status='INCOMPLETE', performance_admitted=False, production_changed=False)
    started = time.monotonic()
    try:
        cases = inventory(output/'extension')
        plan = dict(baseline_commit=BASELINE, extension=identity(), sources=source_hashes(ROOT),
                    harness={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__), HERE/'parse_asys.py')},
                    sdk=str(sdk), device=args.device, cases=cases, samples=args.samples,
                    rounds=args.rounds, order=list(ORDER), cold_weights='QUERIED_L2_X2.25_WHOLE_RINGS',
                    scope='75_FIXED_CASES_NOT_A_CONFIG_SWEEP')
        previous = output/'plan.json'
        if previous.exists() and json.loads(previous.read_text()) != plan:
            raise ValueError('resume source/protocol changed; use a new output directory')
        write_json(previous, plan)
        print('MMA_AB_PLAN '+json.dumps(dict(points=len(cases), captures=4*args.rounds,
              baseline=BASELINE[:8], device=args.device, output=str(output), sweep=False)), flush=True)
        if args.plan_only:
            summary['status'] = 'PLAN_ONLY'
            return 0
        asys = sdk/'asight/bin/asys'
        if not args.build_only:
            if not asys.is_file():
                raise ValueError(f'missing Asys: {asys}')
            run_logged([sys.executable, '-c', 'import numpy, torch, gguf'], output/'dependencies.log', env)
        from quactlize.runtime.compiler import LIBRARIES, sha
        sdk_hashes = {str(p): sha(p) for p in [sdk/'bin/hgcc', *[sdk/'lib'/('lib'+s+'.so') for s in LIBRARIES]]}
        run_logged([sdk/'bin/hgcc', '--version'], output/'compiler-version.log', env)
        receipt_path = output/'build.json'
        if receipt_path.exists():
            build = json.loads(receipt_path.read_text())
            if build['sdk_hashes'] != sdk_hashes or any(sha(Path(p)) != h for p,h in build['libraries'].items()):
                raise ValueError('SDK or compiled libraries changed; use a new output directory')
            build_root = Path(build['root'])
        else:
            build_root = Path(tempfile.mkdtemp(prefix='build-', dir=output))
            baseline = build_root/'baseline-source'
            export_baseline(baseline)
            for arm, source in (('baseline', baseline), ('extension', ROOT)):
                arm_output = build_root/arm
                arm_output.mkdir()
                for delivery in ('direct', 'staged'):
                    selected = [c for c in cases if c['reader'] == delivery]
                    build_reader(selected, delivery, sdk, arm_output, args.jobs, env, source_root=source)
            run_logged([sys.executable, ROOT/'tools/build_kpack_pack.py', '--sdk', sdk,
                        '--output', build_root/'packer'], build_root/'packer-build.log', env)
            if source_hashes(ROOT) != plan['sources']:
                raise ValueError('sources changed while building; no comparison admitted')
            build = dict(root=str(build_root), sdk_hashes=sdk_hashes, baseline_sources=source_hashes(baseline),
                         libraries={str(p): sha(p) for p in build_root.rglob('*.so')})
            write_json(receipt_path, build)
        if args.build_only:
            summary['status'] = 'BUILD_ONLY'
            return 0
        captures = []
        for block in range(args.rounds):
            for slot, arm in enumerate(ORDER):
                label = f'{block}-{slot}-{arm}'
                checkpoint = output/(label+'.json')
                if checkpoint.exists():
                    captures.append(json.loads(checkpoint.read_text()))
                    print('MMA_AB_REUSE '+label, flush=True)
                    continue
                attempt = Path(tempfile.mkdtemp(prefix=label+'-', dir=output))
                selected = inventory(build_root/arm)
                case_file = attempt/'cases.json'
                write_json(case_file, selected)
                raw_path = attempt/'results.jsonl'
                print(f'MMA_AB_CAPTURE block={block+1}/{args.rounds} arm={arm} slot={slot+1}/4 log={attempt}/profile.log', flush=True)
                device_snapshot(sdk, attempt/'device-before.log', env)
                run_logged([asys, 'profile', '-t', 'hggc', '-o', attempt/'trace', sys.executable,
                            HERE/'qgemv_bench.py', '--package', build_root/'packer', '--packer-only',
                            '--cases', case_file, '--out', raw_path, '--check', '--cold-weights',
                            '--samples', args.samples, '--graph-calls', 48, '--max-ring', 192,
                            '--stop-on-error'], attempt/'profile.log', env)
                device_snapshot(sdk, attempt/'device-after.log', env)
                reports = list(attempt.glob('trace*.asysrep'))
                if len(reports) != 1:
                    raise ValueError(f'expected one Asys report: {attempt}')
                database = attempt/'trace.sqlite'
                run_logged([asys, 'export', '--force-overwrite', 'true', '-o', database, reports[0]], attempt/'export.log', env)
                raw = [json.loads(line) for line in raw_path.read_text().splitlines()]
                rows = checked_capture(selected, raw, parse_asys.attribute(raw, parse_asys.load(database)))
                record = dict(arm=arm, block=block, slot=slot, records=rows,
                              report=str(reports[0]), report_sha256=sha(reports[0]), sqlite_sha256=sha(database))
                write_json(checkpoint, record)
                captures.append(record)
                print(f'MMA_AB_CAPTURE_DONE label={label} points={len(rows)}', flush=True)
        summary = compare(captures, [c['id'] for c in cases], 2*args.rounds)
        write_json(output/'comparison.json', summary)
        for row in summary['results']:
            print('MMA_AB_POINT '+json.dumps(dict(id=row['id'], delta_pct=round(row['delta_pct'], 3),
                  baseline_us=row['measurements']['baseline']['median_total_us'],
                  extension_us=row['measurements']['extension']['median_total_us'],
                  ranges_separated=row['repeat_ranges_separated'])), flush=True)
        summary = {k:v for k,v in summary.items() if k != 'results'}
        return 0
    except Exception as exc:
        summary['error'] = str(exc)
        print('MMA_AB_FAIL '+str(exc), flush=True)
        return 1
    finally:
        summary['seconds'] = time.monotonic()-started
        write_json(output/'summary.json', summary)
        archive = output/'review.tgz'
        with tarfile.open(archive, 'w:gz') as tar:
            for p in sorted(output.rglob('*')):
                if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.jsonl', '.log', '.txt'):
                    tar.add(p, arcname=str(p.relative_to(output)), recursive=False)
        print('MMA_AB_DONE '+json.dumps(summary), flush=True)
        print(f'results={archive}', flush=True)
        if summary['status'] == 'INCOMPLETE':
            print(f'Retry with --resume --output {output} and the same arguments/card.', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
