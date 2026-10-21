#!/usr/bin/env python3
"""Qwen3-32B PP2048 config sweep, ordinary calls + Asys GPU activity durations.

No kernel/selector/caller edits. No CUDA Graph, per-stage events or HGTX
range inference. Fixed-size config batches share one cold-weight fixture;
profiler API capture ranges merge into a report, with first-ring exclusion.
"""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.prefill_asys.plan import freeze, runnable
from dev.prefill_asys.trace import parse
from dev.strategy.journal import atomic_json
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_prefill_point import build, native, wait


def read(path): return json.loads(Path(path).read_text())


def source_seal():
    from dev.strategy.provenance import harness_identity
    files = {str(p.relative_to(ROOT)):sha(p) for folder in
        ('dev/prefill_point','dev/prefill_asys','dev/native_components')
        for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.cpp','.cu','.cuh','.hpp')}
    files['tools/run_prefill_point.py'] = sha(ROOT/'tools/run_prefill_point.py')
    files['tools/run_prefill_model_asys.py'] = sha(__file__)
    files['transitive_python_harness'] = harness_identity(ROOT)['sha256']
    files['dev/llama_ab/carrier.cpp'] = sha(ROOT/'dev/llama_ab/carrier.cpp')
    return digest(files)


def validate_payload(a):
    from dev.strategy.prefill_build import verify
    from quactlize.dequant.native import verify as dequant_verify
    payload = a.reuse_build or a.output/'build'
    hp, dq = verify(a.sdk,payload/'helpers'), dequant_verify(payload/'dequant',a.sdk)
    for name,value in dq['source_hashes'].items():
        if sha(ROOT/name) != value: raise ValueError('dequant source changed: '+name)
    return dict(path=str(payload),helpers=hp['sha256'],dequant=dq['sha256'])


def validate_native(receipt, a):
    """Recheck dynamic dependencies on resume, not just the tiny carrier."""
    base=receipt['adapter_receipt']['base']['identity']
    if base['sdk']!=str(a.sdk):raise ValueError('native SDK binding changed')
    for name,want in base['libraries'].items():
        if sha(name)!=want:raise ValueError('native/provider library changed: '+name)
    for name,want in base['headers'].items():
        if sha(a.llama_source/name)!=want:raise ValueError('native header changed: '+name)
    if (sha(base['binding']['source_file'])!=base['binding']['source_sha256'] or
            sha(a.llama_build/'CMakeCache.txt')!=base['cmake_sha256']):
        raise ValueError('native source/build changed')
    if sha(receipt['adapter'])!=receipt['adapter_receipt']['adapter_sha256']:
        raise ValueError('diagnostic native adapter changed')


def worker(a):
    from tools.run_kpack_llama_ab import device
    from dev.prefill_point.bench import qz
    from dev.strategy.cold_decode import Ring
    from quactlize.runtime.native import SDK
    from dev.strategy.resources import compiler, module_spec
    from dev.strategy.resource_pilot import current_module
    job = read(a.batch)
    authority = read(a.output/'authority.json')
    if job['authority'] != digest(authority) or authority['harness'] != source_seal():
        raise ValueError('sweep worker source/authority changed')
    if device(a) != authority['device']: raise ValueError('worker physical device/SDK differs')
    if validate_payload(a) != authority['payload']: raise ValueError('worker device payload changed')
    root = a.output
    w = Workload(**job['workload'])
    a.workload, a.trace_only, a.model_prefill_probe = w, True, True
    a.rounds, a.samples = 3, 17
    a.output = a.batch.parent
    (a.output/'results').mkdir()
    # Native carrier reads this copy, but all authority fields remain exact.
    atomic_json(a.output/'authority.json',authority)
    if job['cases'][0]['arm'] == 'native-fp16':
        value = native(a,read(root/'build/receipt.json'))
        value['case_id'] = 'native-fp16'
        atomic_json(a.output/'cases.json',[value])
        return 0
    a.timer_library = root/'build/libpoint-timer.so'
    a.device_payload = Path(authority['payload']['path'])
    a.shared_ring = Ring(SDK(a.sdk),w,a.l2_bytes)
    values = []; modules = read(root/'modules.json'); compilers = {}
    try:
        for case in job['cases']:
            a.recipe, a.case_id = Recipe.from_dict(case['recipe']),case['id']
            a.module_record = None
            if case.get('module'):
                spec = module_spec(w,a.recipe)
                ckey = (spec['endpoint'],spec['compute'])
                if ckey not in compilers: compilers[ckey] = compiler(spec,a.sdk,a.cache)
                a.module_record = modules[case['module']]
                current_module(a.module_record,spec,compilers[ckey])
                if sha(a.module_record['path']) != a.module_record['sha256']:
                    raise ValueError('TC image changed')
            start = time.monotonic()
            print(f'PREFILL_ASYS_CONFIG_START id={case["id"]} arm={case["arm"]}',flush=True)
            value = qz(a,case['arm'])
            value.update(case_id=case['id'],roles=case['roles'])
            values.append(value); atomic_json(a.output/'cases.json',values)
            print(f'PREFILL_ASYS_CONFIG_PASS id={case["id"]} seconds={time.monotonic()-start:.2f}',flush=True)
    finally:
        a.shared_ring.close()
    return 0


def profile_command(asys, output, application):
    return [str(asys),'profile','-t','hggc,hgtx,acblas','--sample','none',
        '--python-sampling','false','--capture-range','hggcProfilerApi',
        '--capture-range-end','merge','--kill','none','--wait','primary',
        '-f','false','-o',str(output),*application]


def execute(command,log,env):
    atomic_json(log.with_suffix('.command.json'),command)
    with log.open('x') as stream:
        p = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        state=dict(pid=p.pid,command=command,started_ns=time.time_ns(),returncode=None)
        atomic_json(log.with_suffix('.process.json'),state)
        try:wait(p,'asys-sweep',log)
        finally:
            state.update(returncode=p.poll(),observed_ns=time.time_ns())
            atomic_json(log.with_suffix('.process.json'),state)


def profile_batch(a, point, cases, authority, env):
    # Completed attempts are immutable. Resume verifies both the numerical
    # receipt and the exact raw SQLite used by the parser, not a bare PASS.
    key = digest([point['id'],cases])
    base = a.output/'batches'/key
    base.mkdir(parents=True,exist_ok=True)
    for path in base.glob('attempt-*/profile.process.json'):
        previous=read(path)
        proc=Path('/proc')/str(previous['pid'])/'cmdline'
        if previous['returncode'] is None and proc.exists():
            # A completed PID can be recycled; match this attempt's unique
            # trace path, not merely a process number or executable basename.
            if str(path.parent).encode() in proc.read_bytes():
                raise ValueError('previous profiler still running; do not overlap: '+str(previous['pid']))
    seal = dict(authority=digest(authority),workload=point['workload'],cases=cases)
    existing = base/'pass.json'
    if existing.exists():
        old = read(existing)
        if (old['seal'] != seal or sha(old['sqlite']) != old['parsed']['sqlite_sha256'] or
                sha(old['cases_path']) != old['cases_sha256'] or old['record_sha256'] !=
                digest({k:v for k,v in old.items() if k != 'record_sha256'})):
            raise ValueError('completed batch evidence changed')
        return old['parsed']['cases']
    attempt = base/('attempt-'+str(time.time_ns())); attempt.mkdir()
    request = attempt/'request.json'; atomic_json(request,seal)
    argv = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--batch',str(request),
        '--output',str(a.output),'--sdk',str(a.sdk),'--llama-source',str(a.llama_source),
        '--llama-build',str(a.llama_build),'--device',a.device,'--cache',str(a.cache),
        '--l2-bytes',str(a.l2_bytes)]
    if a.reuse_build: argv += ['--reuse-build',str(a.reuse_build)]
    execute(profile_command(a.asys,attempt/'trace',argv),attempt/'profile.log',env)
    report, db = attempt/'trace.asysrep',attempt/'trace.sqlite'
    if not report.is_file(): raise ValueError('Asys report missing: '+str(report))
    execute([str(a.asys),'export','-o',str(db),str(report)],attempt/'export.log',env)
    values = read(attempt/'cases.json')
    if [v['case_id'] for v in values] != [c['id'] for c in cases]:
        raise ValueError('profiled config order/coverage differs')
    data = parse(db,values)
    for row,case in zip(data['cases'],cases):
        row['roles']=case.get('roles',['native'])
    saved = dict(seal=seal,sqlite=str(db),asysrep=str(report),cases_path=str(attempt/'cases.json'),
        cases_sha256=sha(attempt/'cases.json'),parsed=data)
    saved['record_sha256'] = digest(saved)
    atomic_json(existing,saved)
    return data['cases']


def summarize(points, measurements):
    result = []
    for point in points:
        found = measurements.get(point['id'],{})
        native_row = found.get('native-fp16')
        eligible = [r for key,r in found.items() if key != 'native-fp16']
        pool = {c['id']:c for c in point['cases']}
        complete = set(found) == set(pool) | {'native-fp16'}
        winner = min(eligible,key=lambda r:r['gpu_sum_us']) if eligible else None
        controls = [found[k] for k,c in pool.items() if 'model-config' in c['roles'] and k in found]
        result.append(dict(id=point['id'],workload=point['workload'],pp_calls=len(point['tensors']),
            status='SCREEN_COMPLETE' if complete else 'INCOMPLETE',measured=len(found),expected=len(pool)+1,
            native_us=native_row['gpu_sum_us'] if native_row else None,
            native_phases=native_row['phases_us'] if native_row else None,
            best_us=winner['gpu_sum_us'] if winner else None,
            best_id=winner['case_id'] if winner else None,
            best_phases=winner['phases_us'] if winner else None,
            best_recipe=pool[winner['case_id']]['recipe'] if winner else None,
            model_config_us=[r['gpu_sum_us'] for r in controls],
            model_config_phases=[r['phases_us'] for r in controls],
            delta_pct=100*(winner['gpu_sum_us']/native_row['gpu_sum_us']-1) if winner and native_row else None))
    return dict(status='SCREEN_COMPLETE' if all(r['status']=='SCREEN_COMPLETE' for r in result) else 'INCOMPLETE',
        points=result,production_changed=False,performance_admitted=False,no_regression_admitted=False,
        timing='ASYS_GPU_ACTIVITY_SUM',warmup_excluded=True,
        scope='FIXED_MODEL_M_CONFIG_SCREEN; PAIRED_FINALISTS_AND_MODEL_REPLAY_REQUIRED',
        caveat='Native F16 and Full BF16 have different rounding/provider settings; no dtype relabelling')


def package(output):
    # GUI reports/SQLite remain on Box. Upload only enough to recompute the
    # tables, inspect kernel geometry/samples, numerics and failures.
    archive = output/('review-'+str(time.time_ns())+'.tgz')
    paths = list(output.glob('*.json'))+list(output.glob('*.log'))
    paths += list((output/'batches').rglob('pass.json'))+list((output/'batches').rglob('cases.json'))
    paths += list((output/'batches').rglob('profile.log'))+list((output/'batches').rglob('export.log'))
    paths += list((output/'batches').rglob('request.json'))+list((output/'batches').rglob('*.process.json'))
    paths += list((output/'build').rglob('*.log'))+list((output/'build').glob('receipt.json'))
    paths += list((output/'probe').glob('probe-*.log'))+list((output/'probe').glob('summary.json'))
    paths += [f for f in (output/'probe/modules').glob('*.json') if read(f)['data']['status']!='PASS']
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(set(paths)):
            if not path.is_file() or path.is_symlink(): continue
            with path.open('rb') as stream:
                if path.suffix == '.log': stream.seek(max(0,path.stat().st_size-16384))
                data = stream.read()
            info = tarfile.TarInfo(str(path.relative_to(output))); info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    print('results='+str(archive),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--llama-source',type=Path,default=Path('/sim/eec/shared/junfu.qx/llama.cpp.ncp'))
    p.add_argument('--llama-build',type=Path)
    p.add_argument('--cache',type=Path)
    p.add_argument('--reuse-build',type=Path)
    p.add_argument('--device',default='1')
    p.add_argument('--asys',type=Path)
    p.add_argument('--smi',type=Path)
    p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--configs-per-batch',type=int,default=16)
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--batch',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.device.isdigit() or not 1 <= a.configs_per_batch <= 32 or not 1<=a.jobs<=64:
        p.error('one GPU; 1..32 configs/batch; 1..64 compile jobs')
    a.output=a.output.resolve()
    a.sdk=a.sdk.resolve();a.llama_source=a.llama_source.resolve()
    a.llama_build=(a.llama_build or a.llama_source/'build-ci-32F-cublas').resolve()
    if a.resume and (a.output/'authority.json').exists():
        old=read(a.output/'authority.json')
        if a.cache is None:a.cache=Path(old['cache'])
        if a.reuse_build is None and Path(old['payload']['path'])!=a.output/'build':
            a.reuse_build=Path(old['payload']['path'])
    a.cache=(a.cache or a.output/'cache').resolve()
    if a.reuse_build: a.reuse_build=a.reuse_build.resolve(strict=True)
    if a.worker: return worker(a)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if not a.model: p.error('--model is required')
    from tools.gguf_internal_shape_inventory import read_gguf_header
    from dev.strategy.provenance import source_identity
    a.model=a.model.resolve(strict=True)
    with a.model.open('rb') as stream: header=read_gguf_header(stream,str(a.model))
    identity=dict(path=str(a.model),bytes=a.model.stat().st_size,header=header)
    plan_path=a.output/'plan.json'
    if plan_path.exists():
        plan=read(plan_path)
        if plan['model_identity']!=identity or plan['harness']!=source_seal():
            raise ValueError('resume model/harness changed')
        if plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
            raise ValueError('plan digest differs')
    else:
        plan=freeze(header,a.output)
        plan.update(model_identity=identity,harness=source_seal(),source_identity=source_identity(ROOT))
        plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        atomic_json(plan_path,plan)
    print('PREFILL_ASYS_PLAN '+json.dumps(plan['counts']|dict(configs_after_resources='PENDING',production_changed=False)),flush=True)
    if a.plan_only: return 0
    if not a.confirm_idle: p.error('check assigned GPU idle then --confirm-idle')
    from tools.run_native_components import environment
    from tools.run_kpack_llama_ab import device
    from dev.strategy.refresh_clock import request_lock,observe
    env=environment(a);env.update(CUDA_VISIBLE_DEVICES=a.device,GGML_CUDA_CUBLAS_ACCUMULATOR='f32')
    os.environ.update(env)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    a.smi=a.smi or next((v for v in (Path(shutil.which('ppu-smi') or '/nonexistent'),a.sdk/'bin/ppu-smi') if v.is_file()),None)
    if a.smi is None: raise ValueError('ppu-smi required for 1700MHz request')
    d=device(a);a.l2_bytes=d['l2']['bytes'];pci=d['device']['pci']
    measurements={}; status=None
    try:
        old=read(a.output/'authority.json') if (a.output/'authority.json').exists() else None
        if old and (old['device']!=d or old['harness']!=source_seal() or old['plan']!=plan['plan_sha256']):
            raise ValueError('resume source/device/SDK differs')
        receipt=read(a.output/'build/receipt.json') if (a.output/'build/receipt.json').exists() else build(a)
        for key,path in [('carrier_sha256',Path(receipt['carrier'])),('timer_sha256',a.output/'build/libpoint-timer.so')]:
            if sha(path)!=receipt[key]:raise ValueError('diagnostic executable changed')
        validate_native(receipt,a)
        payload=validate_payload(a)
        authority=dict(device=d,payload=payload,build=receipt,harness=source_seal(),plan=plan['plan_sha256'],
            cache=str(a.cache),asys_sha256=sha(a.asys),batch_size=a.configs_per_batch,frequency_normalization=False)
        if old and old!=authority: raise ValueError('resume build/protocol changed')
        if not old:atomic_json(a.output/'authority.json',authority)
        clock=request_lock(a.smi,pci,1700);atomic_json(a.output/('clock-lock-'+str(time.time_ns())+'.json'),clock)
        if clock['returncode']: raise ValueError('1700MHz lock request failed')
        cmd=[sys.executable,'-u',str(ROOT/'tools/probe_kpack_strategy.py'),'--plan',str(plan_path),
            '--output',str(a.output/'probe'),'--sdk',str(a.sdk),'--cache',str(a.cache),
            '--jobs',str(a.jobs),'--max-modules','96']
        if (a.output/'probe').exists(): cmd.append('--resume')
        execute(cmd,a.output/('resources-'+str(time.time_ns())+'.log'),env)
        resource=read(a.output/'probe/summary.json')
        if resource['status']!='COMPLETE': raise ValueError('resources incomplete; resume will reuse finished modules')
        points=read(a.output/'probe/resolved-plan.json')['points']
        queries={f.stem:read(f)['data'] for f in (a.output/'probe/queries').glob('*.json')}
        modules={f.stem:read(f)['data']['record'] for f in (a.output/'probe/modules').glob('*.json') if read(f)['data']['status']=='PASS'}
        atomic_json(a.output/'modules.json',modules)
        for point in points:
            point['queries']=queries;point['cases']=runnable(point);point.pop('queries')
        atomic_json(a.output/'execution-plan.json',dict(points=points,authority=digest(authority)))
        total=sum(len(p['cases'])+1 for p in points)
        print(f'PREFILL_ASYS_READY shapes={len(points)} configs={total} kernel_changes=0 production_changed=0',flush=True)
        lock_path=Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')
        with lock_path.open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            started=time.monotonic();done=0
            for point in points:
                measurements[point['id']]={}
                jobs=[[dict(id='native-fp16',arm='native-fp16')]]
                # A TC + full-provider pair checks different capture ranges,
                # adapters and allocation teardown before the larger batches.
                cs=point['cases']
                smoke=[next(c for c in cs if c['arm']=='sf'),next(c for c in cs if c['arm']=='full-bf16')]
                jobs.append(smoke)
                rest=[c for c in cs if c['id'] not in {s['id'] for s in smoke}]
                jobs += [rest[i:i+a.configs_per_batch] for i in range(0,len(rest),a.configs_per_batch)]
                for cases in jobs:
                    before=observe(a.smi,pci,1700,advisory=True)
                    measured=profile_batch(a,point,cases,authority,env)
                    for r in measured: measurements[point['id']][r['case_id']]=r
                    done+=len(cases)
                    after=observe(a.smi,pci,1700,advisory=True)
                    atomic_json(a.output/('clock-'+point['id']+'-'+cases[0]['id']+'.json'),dict(before=before,after=after))
                    status=summarize(points,measurements);atomic_json(a.output/'summary.json',status)
                    print('PREFILL_ASYS_PROGRESS '+json.dumps(dict(completed=done,total=total,
                        elapsed_minutes=(time.monotonic()-started)/60,shape=point['workload'],
                        remaining_minutes=(time.monotonic()-started)/done*(total-done)/60,
                        eta='OBSERVED_CONFIG_AVERAGE_NOT_GUARANTEE')),flush=True)
        print('PREFILL_ASYS_DONE '+json.dumps(status),flush=True)
        return 0 if status['status']=='SCREEN_COMPLETE' else 1
    except BaseException as e:
        atomic_json(a.output/('error-'+str(time.time_ns())+'.json'),dict(error=repr(e),traceback=traceback.format_exc()))
        raise
    finally:package(a.output)


if __name__=='__main__':raise SystemExit(main())
