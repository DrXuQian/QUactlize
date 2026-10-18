#!/usr/bin/env python3
"""Actual llama raw/K-pack dispatcher A/B, without a config sweep or CUDA Graph.

Uses an existing .aoneci/scripts/build.sh build. Builds only a host-side test
carrier against that exact source/build; never edits or builds llama itself.
All-format dense + indexed coverage requires the full PoC caller, not the
dense-only MR. Unsupported operations fail, they are not skipped as passes.
"""
import argparse
from collections import defaultdict
import csv
import fcntl
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev.llama_ab.contract import SCOPE, coverage, family, inventory, judge, pilot, route_proof
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest


@lru_cache(maxsize=8192)
def file_sha(path, mtime, ctime, size):
    return sha(Path(path))


def identities(paths):
    result={}
    for p in sorted(set(paths)):
        if p.is_file():
            st=p.stat();name=str(p.resolve())
            result[name]=file_sha(name,st.st_mtime_ns,st.st_ctime_ns,st.st_size)
    return result


def runtime_payloads(args):
    roots=[args.bundle,args.legacy_bundle,args.llama_build/'bin/quactlize-runtime']
    paths=[]
    if args.pack_library:
        paths.append(args.pack_library)
        if args.pack_library.name=='libquactlize_ppu.so':
            # The merged producer also loads its private dispatch/execution
            # dependencies. Seal them even when the PoC uses a separate host
            # dispatcher, rather than rejecting their legitimate dlopen paths.
            from tools.build_ppu_library import verify
            verify(args.pack_library.parent)
            roots.append(args.pack_library.parent/'quactlize-runtime')
            paths.append(args.pack_library.parent/'manifest.json')
    for root in roots:
        if root:paths.extend(p for p in root.rglob('*') if p.is_file() and ('.so' in p.name or p.suffix=='.json'))
    if args.jit_helper:paths.append(args.jit_helper)
    return paths


def validate_loaded(data, request):
    paths=[Path(s) for s in data['loaded_images']]
    if any(not path.is_file() for path in paths):
        raise ValueError('loaded DSO disappeared or was replaced')
    actual=identities(paths)
    expected=request['candidate_runtime'] | request['carrier']['identity']['libraries']
    for name,value in actual.items():
        if name in expected and expected[name]!=value:
            raise ValueError('loaded DSO changed during the run: '+name)
    for stem in ('libggml-cuda.so','libggml-base.so'):
        matching=[p for p in actual if Path(p).name.startswith(stem)]
        if not matching or any(p not in expected for p in matching):
            raise ValueError('different llama backend image was loaded: '+stem)
    compute=[p for p in actual if Path(p).name.startswith(('libquactlize_ppu.so','libquactlize_kpack_dispatch.so'))]
    if not compute or any(p not in expected for p in compute):
        raise ValueError('candidate compute DSO is missing or outside the recorded runtime')
    return actual


def build_carrier(args, output):
    llama, build = args.llama_source.resolve(strict=True), args.llama_build.resolve(strict=True)
    cache = build / 'CMakeCache.txt'
    text = cache.read_text()
    source = re.search(r'^CMAKE_HOME_DIRECTORY:[^=]*=(.+)$', text, re.M)
    if not source or Path(source[1]).resolve() != llama:
        raise ValueError('--llama-build is not configured from --llama-source')
    if not re.search(r'^GGML_CUDA:BOOL=ON$', text, re.M):
        raise ValueError('use a CUDA/PPU llama build, not CPU-only')
    headers = [llama/'ggml/include/ggml.h', llama/'ggml/src/ggml-backend-impl.h',
               llama/'ggml/src/ggml-cuda/ggml-cuda.cu']
    event_impl = headers[2].read_text()
    # Events are constructed by this carrier with timing enabled. Validate
    # the private payload representation, not a guessed CUDA context offset.
    if not re.search(r'cudaEventRecord\(\(cudaEvent_t\)\s*event->context,\s*cuda_ctx->stream\(\)\)', event_impl):
        raise ValueError('llama backend event payload changed; review the test adapter')
    binary = output/'llama-ab-carrier'
    command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
               '-I'+str(llama/'ggml/include'), '-I'+str(llama/'ggml/src'), '-I'+str(llama/'vendor'),
               str(ROOT/'dev/llama_ab/carrier.cpp'), '-L'+str(build/'bin'),
               '-Wl,-rpath,'+str(build/'bin'), '-Wl,-rpath-link,'+str(build/'bin'),
               '-Wl,-rpath-link,'+str(args.sdk/'lib'),
               '-Wl,-rpath-link,'+str(args.sdk/'CUDA_SDK/targets/x86_64-linux/lib'),
               '-lggml-cuda', '-lggml-base', '-ldl', '-pthread',
               '-o', str(binary)]
    libs = [build/'bin/libggml-cuda.so', build/'bin/libggml-base.so']
    if any(not p.is_file() for p in libs):
        raise ValueError('missing shared llama backend libraries in build/bin')
    identity = dict(llama_source=str(llama), llama_head=subprocess.check_output(
        ['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip(),
        source_headers=identities(headers), cmake_sha256=sha(cache),
        libraries=identities([p for p in (build/'bin').glob('*.so*') if p.is_file()]),
        carrier_source=sha(ROOT/'dev/llama_ab/carrier.cpp'), command=command)
    receipt = output/'carrier.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['identity'] != identity or old['binary_sha256'] != sha(binary):
            raise ValueError('carrier/build changed; use a new run, not resume')
    else:
        if binary.exists():
            raise ValueError('unreceipted carrier output exists; use a new run')
        with (output/'carrier-build.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        atomic_json(receipt, dict(identity=identity, binary_sha256=sha(binary)))
    return binary, json.loads(receipt.read_text())


def device(args):
    import ctypes as C
    from tools.probe_kpack_strategy import device_probe
    from quactlize.runtime.native import SDK, checked, sdk_identity
    result = device_probe(args.sdk)
    header = (args.sdk/'include/driver_types.h').read_text()
    match = re.search(r'hggcDevAttrL2CacheSize\s*=\s*(\d+)', header)
    if not match:
        raise ValueError('SDK L2 attribute declaration missing')
    lib = SDK(args.sdk).lib
    fn = lib.hggcDeviceGetAttribute
    fn.argtypes, fn.restype = [C.POINTER(C.c_int), C.c_int, C.c_int], C.c_int
    value = C.c_int()
    checked(fn(C.byref(value), int(match[1]), 0), 'L2 attribute')
    if value.value < 0 or (value.value and args.l2_bytes not in (0, value.value)):
        raise ValueError('operator L2 capacity disagrees with SDK')
    l2 = value.value or args.l2_bytes
    if l2 <= 0:
        raise ValueError('SDK reports no L2; --l2-bytes must be independently verified, not guessed')
    return dict(device=result, l2=dict(bytes=l2, reported=value.value,
                source='SDK_ATTRIBUTE' if value.value else 'OPERATOR_VERIFIED'), sdk=sdk_identity(args.sdk))


def point_request(row, l2):
    from dev.strategy.prefill_fixture import routing
    from quactlize.dispatch.table import Workload
    w = Workload(**row['workload'])
    route = routing(w, 0)
    # Canonical K-pack is byte neutral with GGUF. Active experts, NOT E*N*K,
    # determine the read set. SF metadata residency is not a separate cold
    # prepass claim here; this is the complete weight-consuming operator.
    block, size = {8:(32,34), 10:(256,84), 11:(256,110), 12:(256,144),
                   13:(256,176), 14:(256,210)}[w.qtype]
    weight_bytes = w.n * w.k // block * size
    active = len(route['active'])
    read_bytes = weight_bytes * active
    copies = max(2, math.ceil(2.25*l2/read_bytes))
    return dict(row, copies=copies, owners=list(map(int, route['owners'])),
        ring=dict(copies=copies, l2_bytes=l2, active_experts=active, expert_bytes=weight_bytes,
                  minimum_read_bytes=read_bytes, active_read_bytes=read_bytes*copies,
                  total_weight_bytes_per_arm=weight_bytes*w.experts*copies,
                  cache='ROTATING_ACTIVE_QUANTIZED_WEIGHTS_AT_LEAST_2P25_L2',
                  separately_cold_sf_metadata=False, router_sha256=route['sha256']))


def environment(args):
    env = dict(os.environ)
    # Do not inherit forced algorithms or a tuning override in either arm.
    for name in list(env):
        if name.startswith(('QUACTLIZE_KPACK_', 'GGML_QUACTLIZE_')):
            env.pop(name)
    env.update(PPU_SDK=str(args.sdk), GGML_CUDA_DISABLE_GRAPHS='1',
               QUACTLIZE_KPACK_COMPUTE='bf16', QUACTLIZE_KPACK_ROUTE='auto',
               GGML_QUACTLIZE_DENSE='1', QUACTLIZE_PPU_CACHE_DIR=str(args.jit_cache),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.bundle:
        helper=args.jit_helper or args.bundle/'jit/tools/kpack_jit.py'
        if not helper.is_file():helper=ROOT/'tools/kpack_jit.py'
        env.update(QUACTLIZE_KPACK_EXECUTION=str(args.bundle),
                   QUACTLIZE_KPACK_JIT_HELPER=str(helper),
                   QUACTLIZE_KPACK_JIT_PYTHON=sys.executable,
                   QUACTLIZE_KPACK_JIT_CACHE=str(args.jit_cache))
        deepgemm=args.bundle/'kpack_deepgemm_prewarm.py'
        if deepgemm.is_file():env['QUACTLIZE_KPACK_DEEPGEMM_HELPER']=str(deepgemm)
    if args.pack_library:
        env['QUACTLIZE_PPU_PACK_LIBRARY'] = str(args.pack_library)
    if args.legacy_bundle:
        env['QUACTLIZE_PPU_BUNDLE'] = str(args.legacy_bundle)
    dirs = [args.llama_build/'bin', args.sdk/'lib', args.sdk/'CUDA_SDK/targets/x86_64-linux/lib']
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, dirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def record_load(path, authority):
    row = json.loads(path.read_text())
    if row.get('authority') != authority or row.get('sha256') != digest({k:v for k,v in row.items() if k!='sha256'}):
        raise ValueError('result authority/digest differs: '+str(path))
    if 'id' in row and row['id']!=path.stem:
        raise ValueError('result stored under another point key')
    if row.get('data') and row.get('status') in ('PASS','REGRESSION','INCONCLUSIVE'):
        if judge(row['data'])['status']!=row['status']:
            raise ValueError('saved verdict disagrees with raw samples')
    return row


def save_record(path, record):
    record['sha256'] = digest(record)
    atomic_json(path, record)


def worker(args):
    request = json.loads((args.output/'request.json').read_text())
    def unchanged():
        if (harness_identity(ROOT)!=request['harness'] or
                identities([Path(p) for p in request['sources']])!=request['sources'] or
                sha(args.output/'llama-ab-carrier')!=request['carrier']['binary_sha256']):
            raise ValueError('test source/carrier changed during the campaign; use an immutable worktree')
    unchanged()
    shard = request['shards'][args.worker]
    current = device(args)
    if current != shard['hardware']:
        raise ValueError('worker device/SDK/L2 changed')
    authority = dict(request=digest(request), hardware=current, shard=args.worker)
    lock = Path('/tmp')/('quactlize-components-'+digest(current['device']['pci'])[:24]+'.lock')
    records = args.output/'records'; records.mkdir(exist_ok=True)
    rows = [r for r in request['points'] if family(r) in shard['families']]
    executable = args.output/'llama-ab-carrier'
    env = environment(args)
    start = time.monotonic()
    with lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The pilot is mandatory for a new build. Performance regressions do
        # not stop coverage; unsupported/numeric/route failures do stop it.
        own_pilot = {r['id'] for r in pilot(request['points'])} & {r['id'] for r in rows}
        for phase in ('pilot', 'full'):
            groups = defaultdict(list)
            for row in rows:
                if (row['id'] in own_pilot) != (phase == 'pilot'):
                    continue
                path = records/(row['id']+'.json')
                if path.exists():
                    old = record_load(path, authority)
                    if old['status'] in ('PASS','REGRESSION','INCONCLUSIVE'):
                        continue
                groups[family(row)].append(row)
            if phase == 'full':
                failed = [k for k in own_pilot if not (records/(k+'.json')).exists() or
                    record_load(records/(k+'.json'), authority)['status'] not in ('PASS','REGRESSION','INCONCLUSIVE')]
                if failed:
                    print('LLAMA_AB_PILOT_BLOCKED '+json.dumps(dict(shard=args.worker, failed=failed)), flush=True)
                    return 1
                if args.phase == 'pilot':
                    break
            for fid, todo in sorted(groups.items()):
                # An abort must not poison later cases. Completed points stay
                # valid; each failed point is attempted once per invocation.
                while todo:
                    unchanged()
                    if args.hours and time.monotonic()-start >= args.hours*3600:
                        print('LLAMA_AB_BUDGET_EXHAUSTED remaining='+str(len(todo)), flush=True)
                        return 1
                    attempt = args.output/f'shard-{args.worker}'/uuid.uuid4().hex
                    attempt.mkdir(parents=True)
                    points = [point_request(r, current['l2']['bytes']) for r in todo]
                    config = dict(points=points, rounds=args.rounds, samples=args.samples, authority=authority,
                                  runtime_library=str(args.sdk/'lib/libhggc_wrapper.so'))
                    atomic_json(attempt/'request.json', config)
                    command = [str(executable), str(attempt/'request.json'), str(attempt/'raw')]
                    print(f'LLAMA_AB_FAMILY phase={phase} family={fid} points={len(todo)} log={attempt}/run.log', flush=True)
                    with (attempt/'run.log').open('x') as log:
                        p = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                        try:
                            while p.poll() is None:
                                try: p.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    print(f'LLAMA_AB_WAIT family={fid} elapsed_minutes={(time.monotonic()-start)/60:.1f} log={attempt}/run.log', flush=True)
                                    if args.hours and time.monotonic()-start >= args.hours*3600:
                                        p.terminate(); p.wait(timeout=30)
                        finally:
                            if p.poll() is None:
                                p.kill(); p.wait()
                    log_text = (attempt/'run.log').read_text(errors='replace')
                    unchanged()
                    completed = set()
                    for row in todo:
                        raw = attempt/'raw'/(row['id']+'.json')
                        if not raw.exists():
                            continue
                        try:
                            data = json.loads(raw.read_text())
                            if data['id']!=row['id'] or data['workload']!=row['workload'] or data['authority']!=authority:
                                raise ValueError('carrier result identity differs')
                            routes = route_proof(log_text, row)
                            named={re.search(r'tensor=([^\s]+)',line)[1] for line in routes}
                            if len(named)!=data['ring']['copies']:
                                raise ValueError('selected plan proof does not cover all cold weight copies')
                            loaded=validate_loaded(data,request)
                            result = judge(data)
                            record = dict(result, id=row['id'], authority=authority, data=data, routes=routes,
                                          loaded_images=loaded,raw_sha256=sha(raw), log=str(attempt/'run.log'))
                        except Exception as exc:
                            record = dict(id=row['id'], authority=authority, status='FAIL', error=repr(exc),
                                          raw_sha256=sha(raw), log=str(attempt/'run.log'))
                        save_record(records/(row['id']+'.json'), record)
                        completed.add(row['id'])
                        print('LLAMA_AB_POINT '+json.dumps({k:record[k] for k in ('id','status','ratio') if k in record}), flush=True)
                    pending = [r for r in todo if r['id'] not in completed]
                    if pending:
                        started = re.findall(r'^LLAMA_AB_START id=([0-9a-f]{64})$', log_text, re.M)
                        # A setup failure applies to this entire family, not
                        # to an arbitrary final point from another process.
                        failed = {started[-1]} if started and started[-1] not in completed else {r['id'] for r in pending}
                        for row in pending:
                            if row['id'] in failed:
                                save_record(records/(row['id']+'.json'), dict(id=row['id'],authority=authority,
                                    status='FAIL',process_rc=p.returncode,error='carrier failed; inspect log',log=str(attempt/'run.log')))
                        todo = [r for r in pending if r['id'] not in failed]
                    else:
                        todo = []
    return 0


def summarize(output, request):
    rows = {}
    for path in sorted((output/'records').glob('*.json')):
        raw = json.loads(path.read_text())
        authority = dict(request=digest(request),hardware=request['shards'][raw['authority']['shard']]['hardware'],
                         shard=raw['authority']['shard'])
        rows[path.stem] = record_load(path, authority)
    result = coverage(request['points'], rows)
    result['full_registry_coverage']=len(request['points'])==1544
    atomic_json(output/'summary.json', result)
    table=output/('comparison-'+str(time.time_ns())+'.csv')
    with table.open('x',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['id','cohort','op','q','M','N','K','E','topk','channels','router',
                         'status','native_us','candidate_us','ratio','error'])
        for point in request['points']:
            w=point['workload'];r=rows.get(point['id'],{})
            writer.writerow([point['id'],point['cohort'],*[w[k] for k in ('operation','qtype','m','n','k','experts','topk','channels','router')],
                             r.get('status','MISSING'),r.get('native_us',''),r.get('candidate_us',''),r.get('ratio',''),r.get('error','')])
    archive = output/('review-'+str(time.time_ns())+'.tgz')
    paths = [output/'request.json',output/'summary.json',output/'carrier.json',output/'carrier-build.log',table]
    paths += sorted((output/'records').glob('*.json'))
    paths += [Path(r['log']) for r in rows.values() if r['status']=='FAIL' and r.get('log')]
    paths += sorted(output.glob('shard-*/console.log'))
    with tarfile.open(archive,'x:gz') as out:
        for path in sorted(set(paths)):
            if path.is_symlink() or not path.resolve().is_relative_to(output) or not path.is_file():
                raise ValueError('invalid compact result path: '+str(path))
            with path.open('rb') as f:
                if path.suffix=='.log': f.seek(max(0,path.stat().st_size-262144))
                data=f.read()
            info=tarfile.TarInfo(str(path.relative_to(output)));info.size=len(data)
            out.addfile(info,io.BytesIO(data))
    print('LLAMA_AB_DONE '+json.dumps({k:v for k,v in result.items() if k!='missing'}),flush=True)
    print('results='+str(archive),flush=True)
    return result


def eta(output, request):
    """Conservative rolling-family extrapolation, never a deadline promise."""
    records={p.stem:json.loads(p.read_text()) for p in (output/'records').glob('*.json')}
    points={r['id']:r for r in request['points']}
    windows=[]
    for shard in request['shards']:
        scope={k for k,r in points.items() if family(r) in shard['families']}
        measured=[records[k]['data'] for k in scope & records.keys() if records[k].get('data')]
        if not measured:
            if scope-records.keys():return dict(remaining_minutes=None,method='INSUFFICIENT_WALL_SAMPLES')
            continue
        pending=scope-records.keys()
        # Fixture/JIT costs are nonuniform; report a broad observed range.
        point_times=sorted(d['point_seconds'] for d in measured)
        setups=sorted(d['setup_family_seconds'] for d in measured)
        families_left=len({family(points[k]) for k in pending})
        estimate=len(pending)*statistics.median(point_times)+families_left*statistics.median(setups)
        upper=len(pending)*max(point_times)+families_left*max(setups)
        windows.append((estimate/60,upper/60))
    return dict(remaining_minutes=max((x[0] for x in windows),default=0),
                observed_slow_bound_minutes=max((x[1] for x in windows),default=0),
                method='ROLLING_POINT_AND_FAMILY_WALL_ADVISORY',duration_guaranteed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-source',type=Path,default=Path(os.environ.get('LLAMA_CI_DIR','/sim/eec/shared/junfu.qx/llama.cpp')))
    p.add_argument('--llama-build',type=Path)
    p.add_argument('--sdk',type=Path,default=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK')))
    p.add_argument('--bundle',type=Path,default=os.environ.get('QUACTLIZE_KPACK_EXECUTION'))
    p.add_argument('--legacy-bundle',type=Path,default=os.environ.get('QUACTLIZE_PPU_BUNDLE'))
    p.add_argument('--pack-library',type=Path,default=os.environ.get('QUACTLIZE_PPU_PACK_LIBRARY'))
    p.add_argument('--jit-helper',type=Path,default=os.environ.get('QUACTLIZE_KPACK_JIT_HELPER'))
    p.add_argument('--jit-cache',type=Path,default=Path(os.environ.get('QUACTLIZE_KPACK_JIT_CACHE') or
        os.environ.get('QUACTLIZE_PPU_CACHE_DIR') or str(Path.home()/'.cache/quactlize/ppu')))
    p.add_argument('--devices',default='1,2,3,4,5,6,7')
    p.add_argument('--l2-bytes',type=int,default=int(os.environ.get('L2_BYTES','0')))
    p.add_argument('--output',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--phase',choices=('pilot','all'),default='pilot')
    p.add_argument('--cohort',choices=('all','decode','prefill'),default='all')
    p.add_argument('--operation',choices=('all','dense','indexed'),default='all')
    p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--samples',type=int,default=3)
    p.add_argument('--hours',type=float,default=0,help='explicit budget, 0 means no forced deadline')
    p.add_argument('--idle-confirmed',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    p.add_argument('--probe-device',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.rounds<6 or args.samples<3 or not math.isfinite(args.hours) or args.hours<0:
        p.error('require rounds>=6, samples>=3 and nonnegative finite hours')
    args.sdk=args.sdk.resolve()
    if args.probe_device:
        print(json.dumps(device(args)));return 0
    if args.worker is not None:
        return worker(args)
    rows=inventory()
    rows=[r for r in rows if (args.cohort=='all' or r['cohort']==args.cohort) and
          (args.operation=='all' or r['workload']['operation']==args.operation)]
    print('LLAMA_AB_PLAN '+json.dumps(dict(points=len(rows),families=len({family(r) for r in rows}),
        pilot_points=len(pilot(rows)),configs_per_shape='PRODUCTION_SELECTOR_ONLY',scope=SCOPE)),flush=True)
    if args.plan_only:return 0
    if not args.idle_confirmed:p.error('--idle-confirmed is required; graphs do not isolate other GPU work')
    if args.resume and not args.output:p.error('--resume requires --output')
    selectors=args.devices.split(',')
    if any(not x.isdigit() for x in selectors) or len(set(selectors))!=len(selectors):
        p.error('--devices must be distinct ordinals')
    args.llama_source=args.llama_source.resolve(strict=True)
    args.llama_build=(args.llama_build or args.llama_source/'build-ci').resolve(strict=True)
    for name in ('bundle','legacy_bundle','pack_library','jit_helper'):
        if getattr(args,name):setattr(args,name,Path(getattr(args,name)).resolve(strict=True))
    args.jit_cache=args.jit_cache.resolve()
    args.output=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='kpack-llama-ab-',dir='/workspace'))
    if args.output.is_symlink() or args.output in (Path('/'),ROOT,args.llama_source,args.llama_build):
        p.error('output must be a separate experiment directory')
    if args.output.exists() and not args.resume and any(args.output.iterdir()):
        p.error('output is nonempty; use a new path or --resume')
    if args.resume and not (args.output/'request.json').is_file():
        p.error('--resume requires an existing experiment request')
    args.output.mkdir(parents=True,exist_ok=True)
    _,carrier=build_carrier(args,args.output)
    hardware=[]
    for selector in selectors:
        command=[sys.executable,str(Path(__file__).resolve()),'--probe-device','--sdk',str(args.sdk),
                 '--l2-bytes',str(args.l2_bytes)]
        output=subprocess.check_output(command,env=dict(environment(args),CUDA_VISIBLE_DEVICES=selector),text=True,cwd=ROOT)
        hardware.append(json.loads(output))
    if len({(h['device']['host'],h['device']['pci']) for h in hardware})!=len(hardware):
        raise ValueError('selectors refer to the same physical GPU')
    families=sorted({family(r) for r in rows})
    sources=identities([ROOT/'dev/llama_ab/carrier.cpp',ROOT/'dev/llama_ab/contract.py',Path(__file__),
                       ROOT/'policies/kpack_component_decode_v1.json',ROOT/'policies/kpack_prefill_confirmation_v1.json'])
    runtime=runtime_payloads(args)
    request=dict(points=rows,carrier=carrier,sources=sources,harness=harness_identity(ROOT),candidate_runtime=identities(runtime),
        rounds=args.rounds,samples=args.samples,mode='EAGER_COLD_COMPLETE_OPERATOR',
        shards=[dict(selector=s,hardware=h,families=families[i::len(selectors)]) for i,(s,h) in enumerate(zip(selectors,hardware))],
        settings={k:v for k,v in environment(args).items() if k.startswith(('QUACTLIZE_','GGML_'))})
    target=args.output/'request.json'
    if target.exists() and json.loads(target.read_text())!=request:
        raise ValueError('resume request/source/runtime/device changed')
    atomic_json(target,request)
    active={};start=time.monotonic()
    try:
        for phase in ('pilot','all') if args.phase=='all' else ('pilot',):
            exit_codes={}
            for i,selector in enumerate(selectors):
                folder=args.output/f'shard-{i}';folder.mkdir(exist_ok=True)
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i),'--output',str(args.output),
                    '--llama-source',str(args.llama_source),'--llama-build',str(args.llama_build),
                    '--sdk',str(args.sdk),'--l2-bytes',str(args.l2_bytes),'--phase',phase,
                    '--jit-cache',str(args.jit_cache),
                    '--rounds',str(args.rounds),'--samples',str(args.samples),'--hours',str(args.hours)]
                for key in ('bundle','legacy_bundle','pack_library','jit_helper'):
                    if getattr(args,key):command += ['--'+key.replace('_','-'),str(getattr(args,key))]
                with (folder/'console.log').open('ab') as log:
                    active[i]=subprocess.Popen(command,cwd=ROOT,env=dict(environment(args),CUDA_VISIBLE_DEVICES=selector),
                                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while active:
                for i,proc in list(active.items()):
                    if proc.poll() is not None:
                        exit_codes[i]=proc.returncode
                        print(f'LLAMA_AB_WORKER_DONE phase={phase} shard={i} rc={proc.returncode}',flush=True);del active[i]
                completed=len(list((args.output/'records').glob('*.json')))
                print(f'LLAMA_AB_PROGRESS phase={phase} alive={len(active)}/{len(selectors)} completed={completed}/{len(rows)} '
                      f'elapsed_minutes={(time.monotonic()-start)/60:.1f} '
                      f'logs={args.output}/shard-*/console.log',flush=True)
                if phase=='all':print('LLAMA_AB_ETA '+json.dumps(eta(args.output,request)),flush=True)
                if active:time.sleep(30)
            if phase=='pilot':
                missing=[]
                for r in pilot(rows):
                    path=args.output/'records'/(r['id']+'.json')
                    if not path.is_file() or json.loads(path.read_text())['status'] not in ('PASS','REGRESSION','INCONCLUSIVE'):
                        missing.append(r['id'])
                blocked=bool(missing) or any(exit_codes.values())
                print('LLAMA_AB_PILOT '+json.dumps(dict(status='FAIL' if blocked else 'MEASURED',missing=missing,
                    worker_exit_codes=exit_codes,
                    wall_seconds=time.monotonic()-start,duration_guaranteed=False)),flush=True)
                if blocked:break
    finally:
        for proc in active.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc in active.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        result=summarize(args.output,request)
    if args.phase=='pilot':
        records=[args.output/'records'/(r['id']+'.json') for r in pilot(rows)]
        passed=sum(p.is_file() and json.loads(p.read_text())['status']=='PASS' for p in records)
        print('LLAMA_AB_PILOT_VERDICT '+json.dumps(dict(passed=passed,expected=len(records),
            status='PASS' if passed==len(records) else 'NOT_ADMITTED',full_campaign=False)),flush=True)
        return int(passed!=len(records))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
