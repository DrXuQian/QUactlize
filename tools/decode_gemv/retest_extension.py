#!/usr/bin/env python3
"""Rebuild and recheck the 46 direct + 29 valid staged historical cases, without a sweep."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT/'docs/decode_gemv/data'
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def load_cases(delivery, library):
    name = 'correct-mma-reader-v3' if delivery == 'direct' else 'correct-mma-staged-v1'
    records = [json.loads(line) for line in (DATA/(name+'.jsonl')).read_text().splitlines()]
    cases, excluded = [], []
    for row in records:
        if row['status'] != 'OK':
            excluded.append(dict(id=row['case']['id'], historical_error=row.get('error')))
            continue
        case = row['case']
        case['proto']['lib'] = str(library)
        cases.append(case)
    expected = 46 if delivery == 'direct' else 29
    if len(cases) != expected or len({c['id'] for c in cases}) != expected:
        raise ValueError(f'{delivery}: historical positive inventory changed')
    if len(excluded) != (0 if delivery == 'direct' else 2):
        raise ValueError(f'{delivery}: historical exclusion inventory changed')
    return cases, excluded


def specialization(case):
    p = case['proto']
    mode = 2 if case['kind'] == 'moe' else 0
    tokens = 1 if mode == 2 else next(v for v in (1, 2, 4, 8) if v >= case['m'])
    code = p['w']
    return (case['qtype'], mode, p['compute'], p['t'], code//1000,
            (code//10)%100, tokens, code%10)


def sources(cases, delivery):
    """Only instantiate required configurations, not the prototype's full sweep catalog."""
    configs = sorted({specialization(c) for c in cases})
    groups, declarations, routes = {}, [], []
    entry = 'launch_typed' if delivery == 'direct' else 'launch_staged'
    for index, cfg in enumerate(configs):
        q, mode, compute, t, wn, wk, tokens, depth = cfg
        symbol = f'mma_retest_{index}'
        declaration = f'extern "C" int {symbol}(qkg_call_v1 const* c,int split)'
        declarations.append(declaration+';')
        body = declaration + ' { return quactlize::execution::mma_reader::' + entry + '<' + \
            ','.join(map(str, cfg)) + '>(*c,split); }\n'
        groups.setdefault((q, mode, compute, t), []).append(body)
        actual_rows = sorted({c['m'] if mode == 0 else c['m']*c['topk']
                              for c in cases if specialization(c) == cfg})
        rows = ' || '.join(f'c->rows=={r}' for r in actual_rows)
        routes.append(f'    if(c->qtype=={q} && c->mode=={mode} && compute=={compute} && '
                      f't=={t} && code=={wn*1000+wk*10+depth} && ({rows})) return {symbol}(c,split);')
    header = 'mma_reader.cuh' if delivery == 'direct' else 'mma_staged.cuh'
    result = {f'q{q}-mode{mode}-compute{compute}-t{t}.cu': f'#include "{header}"\n'+''.join(bodies)
              for (q, mode, compute, t), bodies in groups.items()}
    result['router.cpp'] = '#include "api.h"\n' + '\n'.join(declarations) + '''
extern "C" int qgemv_mma_run(qkg_call_v1 const* c,int compute,int t,int code,int split) {
    if(!c) return QKG_INVALID;
''' + '\n'.join(routes) + '\n    return QKG_SHAPE;\n}\n'
    return result


def run(command, log, env):
    with log.open('w') as stream:
        result = subprocess.run(list(map(str, command)), env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-45:])
        raise RuntimeError(f'rc={result.returncode}; log={log}\n{tail}')


def build_reader(cases, delivery, sdk, output, jobs, env):
    from quactlize.runtime.compiler import FLAGS, LIBRARIES, sha
    work = output/delivery
    work.mkdir()
    units = sources(cases, delivery)
    commands, objects = [], []
    includes = [ROOT/'quactlize/execution', ROOT/'quactlize/include',
                ROOT/'third_party/actlize/include', ROOT/'third_party/actlize/tools/util/include']
    for name, code in units.items():
        source = work/name
        source.write_text(code)
        obj = source.with_suffix('.o')
        command = ([str(sdk/'bin/hgcc'), *FLAGS] if name.endswith('.cu') else
                   ['g++', '-std=c++17', '-O2', '-fPIC'])
        command += ['-I'+str(p) for p in includes] + ['-c', str(source), '-o', str(obj)]
        commands.append((name, command)); objects.append(obj)

    def compile_one(item):
        name, command = item
        print(f'MMA_RETEST_BUILD_START reader={delivery} unit={name}', flush=True)
        run(command, work/(name+'.log'), env)
        print(f'MMA_RETEST_BUILD_PASS reader={delivery} unit={name}', flush=True)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(compile_one, commands))
    library = output/(delivery+'.so')
    link = ['g++', '-shared', '-o', library, *objects, '-L'+str(sdk/'lib'),
            *['-l'+name for name in LIBRARIES]]
    run(link, work/'link.log', env)
    write_json(work/'build.json', dict(library=str(library), sha256=sha(library),
               commands=[cmd for _, cmd in commands], configurations=len({specialization(c) for c in cases})))
    return library


def verdict(cases, rows, rc=0):
    expected = {c['id'] for c in cases}
    seen, passed, failed = set(), [], []
    for row in rows:
        name = row.get('case', {}).get('id')
        if name not in expected or name in seen:
            raise ValueError(f'unexpected/duplicate result: {name}')
        seen.add(name)
        check = row.get('check', {})
        if row.get('status') == 'OK' and check.get('cells', 0) > 0 and \
           check.get('bad') == 0 and check.get('nonfinite') == 0 and \
           check.get('tolerance') == 0.005 and check.get('relative_l1_error', 1) < 0.005:
            passed.append(name)
        else:
            failed.append(dict(id=name, error=row.get('error', 'missing/failed GGUF oracle')))
    missing = sorted(expected-seen)
    return dict(status='PASS' if rc == 0 and not failed and not missing else 'FAIL',
                expected=len(expected), passed=len(passed), failed=failed, missing=missing, runner_rc=rc)


def identity():
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    snapshot = ROOT/'source-identity.json'
    source = (dict(commit=git.stdout.strip()) if git.returncode == 0 else
              json.loads(snapshot.read_text()) if snapshot.is_file() else dict(commit=None))
    if git.returncode == 0:
        status = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain'],
                                capture_output=True, text=True, check=True)
        source['uncommitted_changes'] = bool(status.stdout.strip())
        vendor = subprocess.run(['git', '-C', str(ROOT/'third_party/actlize'), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        source['actlize_commit'] = vendor.stdout.strip() if vendor.returncode == 0 else None
    source['sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in ('quactlize/execution/mma_*.cuh',
            'quactlize/include/actlize_extensions/cutlass/gemm/collective/*exact_code*.hpp',
            'quactlize/include/actlize_extensions/cutlass/gemm/collective/detail/*exact_code*.hpp',
            'tools/decode_gemv/retest_extension.py', 'tools/decode_gemv/qgemv_bench.py')
        for p in ROOT.glob(pattern)}
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(os.environ.get('PPU_SDK', '/usr/local/PPU_SDK')))
    parser.add_argument('--device', type=int, default=int(os.environ.get('DEVICE', '0')))
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', '4')))
    parser.add_argument('--output', type=Path, help='fresh directory; default: temporary qz-mma-retest-*')
    parser.add_argument('--build-only', action='store_true', help='compile/link only, no GPU calls')
    parser.add_argument('--plan-only', action='store_true', help='emit the exact inventory, no SDK or GPU needed')
    args = parser.parse_args()
    if args.jobs < 1 or args.device < 0:
        parser.error('jobs must be positive and device nonnegative')
    if args.output:
        output = args.output.resolve()
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            parser.error(f'--output must be new: {output}')
    else:
        output = Path(tempfile.mkdtemp(prefix='qz-mma-retest-')).resolve()
    sdk = args.sdk.resolve()
    env = dict(os.environ, PPU_SDK=str(sdk), CUDA_VISIBLE_DEVICES=str(args.device), QGEMV_SRC=str(ROOT))
    env['LD_LIBRARY_PATH'] = str(sdk/'lib')+':'+env.get('LD_LIBRARY_PATH', '')
    summary = dict(status='FAIL', scope='GGUF_ORACLE_AND_GRAPH_SMOKE_NOT_PERFORMANCE',
                   production_changed=False, performance_admitted=False, device=args.device,
                   sdk=str(sdk), source=identity(), suites={})
    started, stage = time.monotonic(), 'inventory'
    print(f'MMA_RETEST_START output={output} device={args.device} jobs={args.jobs}', flush=True)
    try:
        cases = {}
        for delivery in ('direct', 'staged'):
            cases[delivery], excluded = load_cases(delivery, output/(delivery+'.so'))
            write_json(output/(delivery+'-cases.json'), cases[delivery])
            summary.setdefault('historical_exclusions', {})[delivery] = excluded
        print('MMA_RETEST_PLAN direct=46 staged=29 historical_exclusions=2 sweep=0', flush=True)
        if args.plan_only:
            summary['status'] = 'PLAN_ONLY'
            return 0
        if not (sdk/'bin/hgcc').is_file() or not (ROOT/'third_party/actlize/include/cute/arch/mma_ppu0010.hpp').is_file():
            raise ValueError('missing SDK or pinned actlize headers; nothing will be fetched automatically')
        if not args.build_only:
            stage = 'dependencies'
            run([sys.executable, '-c', 'import numpy, torch, gguf'], output/'dependencies.log', env)
        stage = 'build-packer'
        run([sys.executable, ROOT/'tools/build_kpack_pack.py', '--sdk', sdk, '--output', output/'packer'],
            output/'packer-build.log', env)
        for delivery in ('direct', 'staged'):
            stage = 'build-'+delivery
            build_reader(cases[delivery], delivery, sdk, output, args.jobs, env)
        if args.build_only:
            summary['status'] = 'BUILD_ONLY'
            return 0
        for delivery in ('direct', 'staged'):
            stage = 'check-'+delivery
            destination = output/(delivery+'-results.jsonl')
            log = output/(delivery+'-check.log')
            command = [sys.executable, HERE/'qgemv_bench.py', '--package', output/'packer',
                       '--packer-only', '--correctness-only', '--cases', output/(delivery+'-cases.json'),
                       '--out', destination, '--ring-mib', '0', '--max-ring', '2', '--graph-calls', '2']
            print(f'MMA_RETEST_CHECK reader={delivery} cases={len(cases[delivery])} log={log}', flush=True)
            with log.open('w') as stream:
                result = subprocess.run(list(map(str, command)), env=env, stdout=stream, stderr=subprocess.STDOUT)
            rows = [json.loads(line) for line in destination.read_text().splitlines()] if destination.is_file() else []
            summary['suites'][delivery] = verdict(cases[delivery], rows, result.returncode)
            print('MMA_RETEST_SUITE '+json.dumps(dict(reader=delivery, **summary['suites'][delivery])), flush=True)
            if summary['suites'][delivery]['status'] != 'PASS':
                print('\n'.join(log.read_text(errors='replace').splitlines()[-45:]), flush=True)
        summary['status'] = 'PASS' if all(r['status'] == 'PASS' for r in summary['suites'].values()) else 'FAIL'
        return 0 if summary['status'] == 'PASS' else 1
    except Exception as exc:
        summary.update(error=str(exc), stage=stage)
        print(f'MMA_RETEST_FAIL stage={stage} error={exc}', flush=True)
        return 1
    finally:
        summary['seconds'] = time.monotonic()-started
        write_json(output/'summary.json', summary)
        archive = output/'review.tgz'
        with tarfile.open(archive, 'w:gz') as tar:
            for p in sorted(output.rglob('*')):
                if p.is_file() and not p.is_symlink() and p.suffix in ('.json', '.jsonl', '.log', '.txt'):
                    tar.add(p, arcname=str(p.relative_to(output)), recursive=False)
        print(f'MMA_RETEST_DONE status={summary["status"]} results={archive}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
