#!/usr/bin/env python3
"""Compare one frozen Q4 local shard with shipped and isolated PPU code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import FLAGS, LIBRARIES

FIXTURE = ROOT / 'dev/gemv_simt/tp2_q4_fixture.tar.gz'
FIXTURE_SHA = 'dd939d422f342f570cc663ab818a77efdd262c1bf7586624dbdd7810e370fa97'
CORE = ('simt_kernel.cuh', 'simt_format.cuh', 'simt_activation.cuh',
        'q4_s1_helpers.cuh', 'q4_s1_validation.hpp', 'reader.hpp')


def sha(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unpack(path):
    if sha(FIXTURE) != FIXTURE_SHA:
        raise ValueError('frozen fixture archive differs')
    names = {f'rank{r}-replay{i}.bin' for r in (0, 1) for i in range(3)}
    with tarfile.open(FIXTURE) as t:
        members = t.getmembers()
        if len(members) != 7 or {m.name for m in members} != names | {'fixture.json'} or any(not m.isfile() for m in members):
            raise ValueError('fixture member inventory differs')
        meta = json.load(t.extractfile('fixture.json'))
        rows = [x for rank in meta['ranks'] for x in rank]
        if (meta['shape'] != [512, 512, 4] or meta['recipe'] != [0, 4, 4, 4, 1] or
                len(rows) != 6 or {x['file'] for x in rows} != names):
            raise ValueError('frozen geometry differs')
        payloads = {}
        for row in rows:
            data = t.extractfile(row['file']).read()
            if (row['bytes'] != 1187848 or len(data) != row['bytes'] or
                    row['field_bytes'] != [589824, 524288, 65536, 4096, 8, 4096] or
                    hashlib.sha256(data).hexdigest() != row['sha256']):
                raise ValueError('fixture bytes differ: ' + row['file'])
            payloads[row['file']] = data
    path.mkdir(exist_ok=False)
    for name, data in payloads.items():
        (path / name).write_bytes(data)
    write(path / 'fixture.json', meta)
    return rows


def command(argv, log, timeout=180):
    started = time.monotonic()
    print('Q4_TP2_START log=' + str(log), flush=True)
    with log.open('w') as out:
        child = subprocess.Popen(list(map(str, argv)), stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while True:
                try:
                    rc = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f'Q4_TP2_WAIT seconds={elapsed:.1f} log={log}', flush=True)
                    if elapsed >= timeout:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                        raise TimeoutError(str(log))
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    return dict(command=list(map(str, argv)), rc=rc, seconds=time.monotonic()-started)


def build(sdk, output, jobs):
    output.mkdir(exist_ok=False)
    source = ROOT / 'dev/gemv_simt/tp2_q4_replay.cu'
    includes = [ROOT, ROOT/'quactlize/include', ROOT/'third_party/actlize/include',
                sdk/'include', sdk/'targets/x86_64-linux/include']
    inc = [f'-I{p}' for p in includes]
    tasks = [
        ('fresh', [sdk/'bin/hgcc', *FLAGS, *inc, '-c', source, '-o', output/'fresh.o']),
        ('pack', [sdk/'bin/hgcc', *FLAGS, *inc, '-c', ROOT/'quactlize/packing/ppu_pack.cu', '-o', output/'pack.o']),
        ('sizes', ['g++', '-std=c++17', '-O2', '-fPIC', *inc, '-c', ROOT/'quactlize/packing/sizes.cpp', '-o', output/'sizes.o']),
        # This caller has no device image: the shipped baseline must not share
        # a process with fresh copies of identically named kernel symbols.
        ('shipped', ['g++', '-std=c++17', '-O2', '-DQTP_PPU=1', '-DQTP_SHIPPED_ONLY=1', *inc,
                     '-x', 'c++', '-c', source, '-o', output/'shipped.o']),
    ]
    write(output/'compile-plan.json', {name: list(map(str, argv)) for name, argv in tasks})
    def compile_one(item):
        name, argv = item
        receipt = command(argv, output/(name+'.log'), 600)
        if receipt['rc']:
            raise ValueError('compile failed: ' + str(output/(name+'.log')))
        return receipt
    with ThreadPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
        receipts = list(pool.map(compile_one, tasks))
    for name, objects in [('fresh', ['fresh', 'pack', 'sizes']), ('shipped', ['shipped', 'sizes'])]:
        argv = ['g++', *[output/(x+'.o') for x in objects], '-ldl', f'-L{sdk}/lib',
                *[f'-l{x}' for x in LIBRARIES], '-o', output/name]
        result = command(argv, output/(name+'-link.log'))
        receipts.append(result)
        if result['rc']:
            raise ValueError('link failed: ' + str(output/(name+'-link.log')))
    write(output/'receipt.json', dict(commands=receipts, compiler_sha256=sha(sdk/'bin/hgcc'),
        source_sha256=sha(source), flags=FLAGS, binaries={n: sha(output/n) for n in ('fresh','shipped')}))
    return output


def bundle_paths(args):
    bundle = args.bundle
    if bundle is None:
        if args.previous is None:
            raise ValueError('set PREVIOUS_RUN to the failed Q4 local diagnostic, or KPACK_BUNDLE')
        env = json.loads((args.previous/'results/q4-local/environment.json').read_text())
        bundle = Path(env['QUACTLIZE_KPACK_EXECUTION'])
    bundle = bundle.resolve(strict=True)
    manifest = json.loads((bundle/'manifest.json').read_text())
    execution = bundle/'libquactlize_ppu_execution.so'
    pack = bundle/'pack/libquactlize_ppu_pack.so'
    if sha(execution) != manifest['execution_sha256'] or sha(pack) != manifest['pack']['files']['pack/libquactlize_ppu_pack.so']:
        raise ValueError('shipped library payload differs from its manifest')
    for name in CORE:
        rel = 'quactlize/execution/' + name
        if sha(ROOT/rel) != manifest['execution_receipt']['source_hashes'][rel]:
            raise ValueError('frozen SIMT source differs from shipped image: ' + name)
    if manifest['execution_receipt']['flags'] != FLAGS:
        raise ValueError('fresh compiler flags differ from shipped image')
    return pack, execution, manifest


def evidence(text, arm, fixture, rc):
    expected = 'Q4_TP2_INPUT ' + ' '.join(f'{k}={v}' for k, v in
        zip(('raw','low','units','a','ids','golden'),fixture['field_fnv']))
    if text.splitlines().count(expected) != 1:
        raise ValueError('prelaunch input hashes differ')
    packs = re.findall(r'^Q4_TP2_PACK producer=(HOST|GPU) low_bad=(\d+) units_bad=(\d+)$', text, re.M)
    cells = []
    for match in re.finditer(r'^Q4_TP2_LOCAL reader=(simt|scalar) producer=(HOST|GPU) compute=(F16|BF16) '
        r'variant=0 columns=4 warps=4 values=4 split=1 relative=(\S+) max_abs=(\S+) '
        r'nonfinite=(\d+) bad=(\d+)/1024 status=(PASS|FAIL)$',text,re.M):
        reader, producer, compute, relative, maximum, nonfinite, bad, status = match.groups()
        cell = dict(reader=reader,producer=producer,compute=compute,relative=float(relative),
                    max_abs=float(maximum),nonfinite=int(nonfinite),bad=int(bad),status=status)
        if status == 'PASS' and (cell['bad'] or cell['nonfinite'] or
                not math.isfinite(cell['relative']) or not 0 <= cell['max_abs'] <= 2e-6):
            raise ValueError('nonfinite or incorrect PASS record')
        if status == 'FAIL' and cell['bad'] == 0:
            raise ValueError('FAIL record has no mismatches')
        # NaN/Inf are valuable failure evidence, but are not JSON numbers.
        for key in ('relative', 'max_abs'):
            if not math.isfinite(cell[key]):
                cell[key] = None
        cells.append(cell)
    expected_cells = {(reader, producer, compute) for reader in (('simt','scalar') if arm=='fresh' else ('simt',))
                      for producer in ('HOST','GPU') for compute in ('F16','BF16')}
    actual = [(c['reader'],c['producer'],c['compute']) for c in cells]
    complete = re.findall(r'^Q4_TP2_COMPLETE arm=(fresh|shipped) cells=(\d+) failures=(\d+)$',text,re.M)
    negative = re.findall(r'^Q4_TP2_NEGATIVE kind=wrong_expert relative=(\S+) status=EXPECTED_RED$',text,re.M)
    failures = sum(c['status']=='FAIL' for c in cells) + sum(int(lo)>0 or int(u)>0 for _,lo,u in packs)
    if (len(actual)!=len(expected_cells) or set(actual)!=expected_cells or
            len(packs)!=2 or {p[0] for p in packs}!={'HOST','GPU'} or
            complete!=[(arm,str(len(expected_cells)),str(failures))] or
            len(negative)!=1 or not math.isfinite(float(negative[0])) or float(negative[0])<=.02 or
            rc!=int(failures!=0)):
        raise ValueError('missing, duplicate or inconsistent diagnostic coverage')
    return dict(status='NUMERIC_MISMATCH' if failures else 'PASS',packs=packs,cells=cells)


def verdict(results):
    expected = {(arm,f'rank{rank}-replay{replay}.bin') for arm in ('shipped','fresh')
                for rank in (0,1) for replay in range(3)}
    if (len(results)!=12 or {(r['arm'],r['fixture']) for r in results} != expected or
            any(r['status'] not in ('PASS','NUMERIC_MISMATCH') for r in results)):
        return 'INCOMPLETE'
    for producer, name in [('HOST','HOST_TRANSFER_DIFFERED'),('GPU','GPU_PACK_BYTES_DIFFER')]:
        if any(p==producer and (int(lo) or int(u)) for r in results for p,lo,u in r['packs']):
            return name
    for arm, reader, name in [('fresh','scalar','SCALAR_CANONICAL_COMPUTE_FAILED'),
                             ('fresh','simt','FRESH_SIMT_COMPUTE_FAILED'),
                             ('shipped','simt','SHIPPED_SIMT_ONLY_FAILURE')]:
        if any(c['reader']==reader and c['status']=='FAIL' for r in results if r['arm']==arm for c in r['cells']):
            return name
    return 'STANDALONE_NOT_REPRODUCED'


def main(args):
    if args.jobs<1:
        raise ValueError('jobs must be positive')
    sdk = args.sdk.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    fixtures = unpack(out/'fixtures')
    pack, execution, manifest = (None,None,None) if args.compile_only else bundle_paths(args)
    if manifest:
        write(out/'shipped-manifest.json', manifest)
        old = manifest['execution_receipt']
        compiler = sha(sdk/'bin/hgcc')
        runtime = {f'lib{name}.so': sha(sdk/'lib'/f'lib{name}.so') for name in LIBRARIES}
        write(out/'sdk-comparison.json', dict(compiler_sha256=compiler, runtime=runtime,
            compiler_matches_shipped=compiler==old['compiler_sha256'],
            runtime_matches_shipped=runtime==old['runtime'],
            scope='RECORD_DIFFERENCES_NOT_A_DEVICE_ADMISSION'))
    write(out/'environment.json', {k: os.environ.get(k) for k in ('PPU_SDK','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES')})
    binaries = build(sdk, out/'build', args.jobs)
    if args.compile_only:
        print('Q4_TP2_SIMT COMPILED device_validated=0', flush=True)
        return 0
    results = []
    for arm in ('shipped','fresh'):
        for fixture in fixtures:
            case = out/(arm+'-'+Path(fixture['file']).stem)
            case.mkdir()
            argv = [binaries/arm, out/'fixtures'/fixture['file'], arm, case]
            if arm=='shipped':
                argv += [pack, execution]
            row = dict(arm=arm,fixture=fixture['file'])
            try:
                row.update(command(argv,case/'run.log'))
                row.update(evidence((case/'run.log').read_text(errors='replace'), arm, fixture, row['rc']))
            except Exception as e:
                row.update(status='INFRASTRUCTURE_OR_COVERAGE_FAIL', error=str(e))
            results.append(row)
            write(out/'summary.json', dict(status='RUNNING',cases=results))
            print(f"Q4_TP2_SIMT_PROGRESS completed={len(results)}/12 arm={arm} fixture={fixture['file']} status={row['status']}",flush=True)
    result = verdict(results)
    write(out/'summary.json', dict(status='DIAGNOSTIC_COMPLETE', verdict=result,
        device_admission='PENDING', timing_valid=False, cells_expected=72, cases=results))
    print(f'Q4_TP2_SIMT_DONE verdict={result} cells_expected=72 results={out}', flush=True)
    return int(result=='INCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--compile-only',action='store_true')
    raise SystemExit(main(parser.parse_args()))
