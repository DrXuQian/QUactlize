#!/usr/bin/env python3
"""Dense model/cohort config sweep, ordinary calls + Asys GPU activity durations.

Production helpers integrated; this runner does not alter route selectors.
No CUDA Graph, per-stage events or HGTX
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
import statistics
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
    files['dev/dense_helpers/plan.py'] = sha(ROOT/'dev/dense_helpers/plan.py')
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
    from tools.build_kpack_prefill_runtime import verify as verify_dense
    dense=verify_dense(a.sdk,a.output/'build/dense')
    return dict(path=str(payload),helpers=hp['sha256'],dequant=dq['sha256'],dense=dense['sha256'])


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
    a.random_dense_probe = read(a.output/'plan.json').get('schema') in (
        'quactlize.dense-random-native.v1','quactlize.dense-model-native.v1')
    a.cohort_prefill_probe = any(k in read(a.output/'plan.json') for k in
                               ('cohort_sha256','helper_plan_sha256','finalist_plan_sha256'))
    a.rounds, a.samples = 3, 17
    a.output = a.batch.parent
    (a.output/'results').mkdir()
    # Native carrier reads this copy, but all authority fields remain exact.
    atomic_json(a.output/'authority.json',authority)
    if job['cases'][0]['arm'] == 'native-fp16':
        if a.no_native:raise ValueError('native arm is forbidden in a Quactlize-only run')
        value = native(a,read(root/'build/receipt.json'))
        value['case_id'] = 'native-fp16'
        atomic_json(a.output/'cases.json',[value])
        return 0
    a.timer_library = root/'build/libpoint-timer.so'
    a.device_payload = Path(authority['payload']['path'])
    a.dense_library = root/'build/dense/libquactlize_ppu_prefill.so'
    a.shared_ring = Ring(SDK(a.sdk),w,a.l2_bytes)
    values = []; modules = read(root/'modules.json'); compilers = {}
    try:
        for case in job['cases']:
            a.recipe, a.case_id = Recipe.from_dict(case['recipe']),case['id']
            a.helper_mode = case['helper_mode']
            a.helper_options = case.get('helper_options')
            a.module_record = None
            if case.get('module'):
                spec = module_spec(w,a.recipe)
                ckey = (spec['endpoint'],spec['compute'],bool(spec.get('legacy_adapters')))
                if ckey not in compilers: compilers[ckey] = compiler(spec,a.sdk,a.cache)
                a.module_record = modules[case['module']]
                current_module(a.module_record,spec,compilers[ckey])
                if sha(a.module_record['path']) != a.module_record['sha256']:
                    raise ValueError('TC image changed')
            start = time.monotonic()
            print(f'PREFILL_ASYS_CONFIG_START id={case["id"]} arm={case["arm"]}',flush=True)
            value = qz(a,case['arm'])
            if a.random_dense_probe:
                value['random_gate_evidence']=dict(raw_sha256=value['fixture']['fixture']['raw_sha256'],
                    arithmetic=dict(input='F32',output='F32',compute='F16',accumulator='FP32'))
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


def profile_batch(a, point, cases, authority, env, confirmation=0):
    # Completed attempts are immutable. Resume verifies both the numerical
    # receipt and the exact raw SQLite used by the parser, not a bare PASS.
    key = digest([point['id'],cases]+([confirmation] if confirmation else []))
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
    if confirmation: seal['confirmation']=confirmation
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
    if a.no_native: argv.append('--no-native')
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


def confirmation_jobs(point):
    """Bracket forward/reverse batches with native; retain both directions.

    The larger M-span shortlist bounds each profiler process, so a late
    failure cannot discard every preceding config in a 48-config capture.
    Existing small finalist plans retain their original four-capture protocol.
    """
    cases=point['cases'];limit=point.get('capture_batch_limit',len(cases))
    if not cases or type(limit)is not int or limit<1:
        raise ValueError('invalid paired capture inventory/batch limit')
    native_case=[dict(id='native-fp16',arm='native-fp16')]
    native=point.get('native_comparison',True)
    jobs=[(native_case,0)] if native else []
    for confirmation,ordered in enumerate((cases,list(reversed(cases)))):
        jobs.extend((ordered[i:i+limit],confirmation) for i in range(0,len(ordered),limit))
    return jobs+([(native_case,1)] if native else [])


def confirmed_capture(previous, current):
    def sample(row):
        return dict(gpu_sum_us=row['gpu_sum_us'],phases_us=row['phases_us'])
    history=list(previous.get('confirmation_samples',[sample(previous)])) if previous else []
    history.append(sample(current))
    result=dict(current,confirmation_samples=history)
    if previous:
        def signatures(row):
            return [{k:v for k,v in node.items() if k not in ('samples_us','median_us')}
                    for node in row.get('kernel_inventory',[])]
        if (previous.get('confirmation_issue') or signatures(previous)!=signatures(current)
                or previous.get('arithmetic')!=current.get('arithmetic')):
            result['confirmation_issue']='KERNEL_OR_ARITHMETIC_DRIFT'
    result['gpu_sum_us']=statistics.median(r['gpu_sum_us'] for r in history)
    result['phases_us']={k:statistics.median(r['phases_us'][k] for r in history) for k in current['phases_us']}
    return result


def summarize(points, measurements):
    result = []
    for point in points:
        found = measurements.get(point['id'],{})
        native_row = found.get('native-fp16')
        eligible = [r for key,r in found.items() if key != 'native-fp16']
        pool = {c['id']:c for c in point['cases']}
        native_required=point.get('native_comparison',True)
        complete = set(found) == set(pool) | ({'native-fp16'} if native_required else set())
        if point.get('confirmation_captures'):
            complete &= all(len(r.get('confirmation_samples',[]))==point['confirmation_captures']
                            and not r.get('confirmation_issue') for r in found.values())
        winner = min(eligible,key=lambda r:r['gpu_sum_us']) if eligible else None
        controls = [found[k] for k,c in pool.items() if 'model-config' in c['roles'] and k in found]
        incumbent=min((r['gpu_sum_us'] for r in controls),default=None)
        same_dtype = [r for r in eligible if pool[r['case_id']].get('arm') in ('sf','fq','full-fp16')
                      and pool[r['case_id']].get('helper_mode') in ('production','cohort-current','experimental')]
        f16 = min(same_dtype,key=lambda r:r['gpu_sum_us']) if same_dtype else None
        legacy = [found[k] for k,c in pool.items() if 'legacy-model-config' in c['roles'] and k in found]
        result.append(dict(id=point['id'],workload=point['workload'],
            pp_calls=point.get('call_count',len(point['tensors'])),
            bucket=point.get('bucket'),consumers=point.get('consumers'),
            status='SCREEN_COMPLETE' if complete else 'INCOMPLETE',measured=len(found),expected=len(pool)+int(native_required),
            native_comparison=native_required,incumbent_us=incumbent,
            incumbent_delta_pct=100*(winner['gpu_sum_us']/incumbent-1) if winner and incumbent else None,
            native_us=native_row['gpu_sum_us'] if native_row else None,
            native_phases=native_row['phases_us'] if native_row else None,
            best_us=winner['gpu_sum_us'] if winner else None,
            best_id=winner['case_id'] if winner else None,
            best_phases=winner['phases_us'] if winner else None,
            best_recipe=pool[winner['case_id']]['recipe'] if winner else None,
            same_dtype_best_us=f16['gpu_sum_us'] if f16 else None,
            same_dtype_best_id=f16['case_id'] if f16 else None,
            same_dtype_best_roles=pool[f16['case_id']]['roles'] if f16 else [],
            same_dtype_phases=f16['phases_us'] if f16 else None,
            same_dtype_delta_pct=100*(f16['gpu_sum_us']/native_row['gpu_sum_us']-1) if f16 and native_row else None,
            legacy_model_config_us=[r['gpu_sum_us'] for r in legacy],
            model_config_us=[r['gpu_sum_us'] for r in controls],
            model_config_phases=[r['phases_us'] for r in controls],
            delta_pct=100*(winner['gpu_sum_us']/native_row['gpu_sum_us']-1) if winner and native_row else None))
    native_comparison=any(p.get('native_comparison',True) for p in points)
    return dict(status='SCREEN_COMPLETE' if all(r['status']=='SCREEN_COMPLETE' for r in result) else 'INCOMPLETE',
        points=result,production_changed=False,performance_admitted=False,no_regression_admitted=False,
        native_comparison=native_comparison,
        timing='ASYS_GPU_ACTIVITY_SUM',warmup_excluded=True,
        scope=('FIXED_MODEL_M_CONFIG_SCREEN; PAIRED_FINALISTS_AND_MODEL_REPLAY_REQUIRED' if native_comparison else
               'QUACTLIZE_CONFIGS_VS_INCUMBENT_NOT_NATIVE_AB'),
        caveat='Use same_dtype fields for native F16/FP32 comparison; Full BF16 is separately labelled')


def helper_review(points,measurements):
    """Data for exact-table review; never turns BF16/native mismatch into parity."""
    rows=[]
    for p in points:
        found=measurements.get(p['id'],{})
        native=found.get('native-fp16')
        for arm in ('full-fp16','full-bf16-typed'):
            planned=[c for c in p['cases'] if c['arm']==arm]
            actual=[(c,found[c['id']]) for c in planned if c['id'] in found]
            complete=bool(planned) and len(actual)==len(planned) and all(
                len(r.get('confirmation_samples',[]))==2 for _,r in actual)
            winner=min(actual,key=lambda pair:pair[1]['gpu_sum_us'],default=None)
            controls=[(c,r) for c,r in actual if any('control' in role for role in c['roles'])]
            best_control=min(controls,key=lambda pair:pair[1]['gpu_sum_us'],default=None)
            typed_workload=dict(p['workload'],compute='bf16' if arm=='full-bf16-typed' else 'f16')
            record=dict(point=p['id'],selection_key=Workload(**typed_workload).key,workload=typed_workload,
                arm=arm,status='COMPLETE' if complete else 'INCOMPLETE',
                selected=None,production_changed=False)
            if winner:
                c,r=winner;record.update(selected=c['helper_options'],median_us=r['gpu_sum_us'],
                    phases_us=r['phases_us'],confirmation_samples=r['confirmation_samples'],roles=c['roles'])
                if best_control:
                    old=best_control[1]['gpu_sum_us'];record.update(best_control_us=old,delta_pct=100*(r['gpu_sum_us']/old-1))
                if arm=='full-fp16' and native and len(native.get('confirmation_samples',[]))==2:
                    record.update(native_us=native['gpu_sum_us'],native_delta_pct=100*(r['gpu_sum_us']/native['gpu_sum_us']-1))
            rows.append(record)
    return dict(rows=rows,production_changed=False,performance_admitted=False,
                scope='EXACT_HELPER_PROPOSALS_ONLY; EXISTING_TC_ROUTES_NOT_REPLACED')


def package(output):
    # GUI reports/SQLite remain on Box. Upload only enough to recompute the
    # tables, inspect kernel geometry/samples, numerics and failures.
    archive = output/('review-'+str(time.time_ns())+'.tgz')
    paths = list(output.glob('*.json'))+list(output.glob('*.log'))
    paths += list((output/'batches').rglob('pass.json'))+list((output/'batches').rglob('cases.json'))
    paths += list((output/'batches').rglob('profile.log'))+list((output/'batches').rglob('export.log'))
    paths += list((output/'batches').rglob('request.json'))+list((output/'batches').rglob('*.process.json'))
    paths += list((output/'build').rglob('*.log'))+list((output/'build').glob('receipt.json'))
    paths += list((output/'build/dense').glob('manifest.json'))
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
    p.add_argument('--cohort',type=Path,help='validated plan_prefill_model_cohort.py inventory, instead of --model')
    p.add_argument('--helper-plan',type=Path,help='factorized helper finalists; no TC config/resource sweep')
    p.add_argument('--finalist-plan',type=Path,help='frozen affected/noisy route finalists; fixed TC schedules')
    p.add_argument('--max-cases',type=int,default=96,help='cohort per-shape budget; never silently truncates controls')
    p.add_argument('--max-modules',type=int,default=256)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--point',action='append',help='cohort point ID; new run for a single-point optimization')
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
    p.add_argument('--no-native',action='store_true',help='M-span config selection only; do not build or run llama reference')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--batch',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if not 0<=a.shard<a.shards or a.shards>64 or a.max_cases<1 or a.max_modules<1:
        p.error('invalid shard or candidate/module budget')
    if not a.device.isascii() or not a.device.isdigit():
        p.error(f'--device requires one GPU ordinal, got {a.device!r}')
    if not 1<=a.configs_per_batch<=32:
        p.error(f'--configs-per-batch must be in 1..32, got {a.configs_per_batch!r}')
    if not 1<=a.jobs<=64:
        p.error(f'--jobs must be in 1..64, got {a.jobs!r}')
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
    if sum(bool(v) for v in (a.model,a.cohort,a.helper_plan,a.finalist_plan))!=1:
        p.error('supply exactly one of --model, --cohort, --helper-plan or --finalist-plan')
    paired = bool(a.helper_plan or a.finalist_plan)
    from tools.gguf_internal_shape_inventory import read_gguf_header
    from dev.strategy.provenance import source_identity
    if a.finalist_plan:
        from dev.prefill_asys.finalists import load as validate_finalists
        finalists=validate_finalists(read(a.finalist_plan))
        if a.no_native and finalists['schema']!='quactlize.dense-m-span.v1':
            p.error('--no-native currently requires the dense M-span plan with incumbent controls')
        if finalists['inputs']['helper_policy_sha256']!=digest(read(ROOT/'policies/dense_full_helpers_20261023.json')):
            raise ValueError('finalist current-helper policy changed; regenerate the plan')
        identity=dict(finalist_plan_sha256=finalists['sha256'],shard=a.shard,shards=a.shards,
                      points=sorted(a.point or []))
    elif a.helper_plan:
        from dev.dense_helpers.plan import validate_full_plan
        helper=validate_full_plan(read(a.helper_plan))
        if not helper['points']: raise ValueError('no ready helper finalist points')
        identity=dict(helper_plan_sha256=helper['sha256'])
    elif a.cohort:
        from dev.prefill_asys.cohort import validate, freeze as freeze_cohort
        cohort=validate(read(a.cohort))
        identity=dict(cohort_sha256=cohort['sha256'],shard=a.shard,shards=a.shards,max_cases=a.max_cases,
                      points=sorted(a.point or []))
    else:
        a.model=a.model.resolve(strict=True)
        with a.model.open('rb') as stream: header=read_gguf_header(stream,str(a.model))
        identity=dict(path=str(a.model),bytes=a.model.stat().st_size,header=header)
    if a.no_native:
        if not a.finalist_plan:p.error('--no-native requires --finalist-plan')
        identity['native_reference']=False
    plan_path=a.output/'plan.json'
    if plan_path.exists():
        plan=read(plan_path)
        if plan['model_identity']!=identity or plan['harness']!=source_seal():
            raise ValueError('resume model/harness changed')
        if plan['plan_sha256']!=digest({k:v for k,v in plan.items() if k!='plan_sha256'}):
            raise ValueError('plan digest differs')
    else:
        if a.finalist_plan:
            from dev.prefill_asys.finalists import partition as finalist_partition, counts as finalist_counts
            plan=dict(finalists,finalist_plan_sha256=finalists['sha256'])
            if a.point:
                known={v['id'] for v in plan['points']}
                if len(set(a.point))!=len(a.point) or not set(a.point)<=known:
                    raise ValueError('unknown/duplicate requested finalist point ID')
                plan['points']=[v for v in plan['points'] if v['id'] in a.point]
            plan['all_counts']=dict(plan['counts'])
            plan['points']=finalist_partition(plan['points'],a.shard,a.shards)
            plan['counts']=finalist_counts(plan['points'])
        elif a.helper_plan:
            plan=dict(helper,helper_plan_sha256=helper['sha256'])
        elif a.cohort:
            if a.point:
                known={v['id'] for v in cohort['points']}
                if len(set(a.point))!=len(a.point) or not set(a.point)<=known:
                    raise ValueError('unknown/duplicate requested point ID')
                cohort=dict(cohort,points=[v for v in cohort['points'] if v['id'] in a.point])
                cohort['sha256']=digest({k:v for k,v in cohort.items() if k!='sha256'})
            plan=freeze_cohort(cohort,a.output,a.max_cases)
            # Family ownership is stable. Every config/control for a shape
            # runs on the same physical GPU; no cross-device fastest sample.
            from dev.prefill_asys.cohort import partition
            plan['all_counts']=dict(plan['counts'])
            plan['points']=partition(plan['points'],a.shard,a.shards)
            from dev.strategy.resources import inventory
            plan['counts']=dict(points=len(plan['points']),**inventory(plan)['counts'])
            if not plan['points']: raise ValueError('empty shard; reduce shard count')
        else:
            plan=freeze(header,a.output)
        if a.no_native:
            plan['points']=[dict(point,native_comparison=False) for point in plan['points']]
            plan['counts']=finalist_counts(plan['points'])
            plan['native_comparison']=False
            plan['comparison']='QUACTLIZE_CONFIGS_VS_INCUMBENT_NOT_NATIVE_AB'
        plan.update(model_identity=identity,harness=source_seal(),source_identity=source_identity(ROOT))
        if (ROOT/'.quactlize-source.json').is_file():
            from dev.strategy.smallm_asys import revision
            plan['source_revision']=revision(ROOT)
        plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        atomic_json(plan_path,plan)
    print('PREFILL_ASYS_PLAN '+json.dumps(plan['counts']|dict(configs_after_resources='PENDING',
        native_reference=not a.no_native,production_changed=False)),flush=True)
    if any(point.get('capture_batch_limit',a.configs_per_batch)!=a.configs_per_batch for point in plan['points']):
        p.error('--configs-per-batch differs from the frozen M-span capture limit')
    has_compact_reducer=any(dict(c['recipe']['parameters']).get('dense_reducer')
                           for point in plan['points'] for c in point.get('cases',[]))
    if has_compact_reducer:
        # Also run for --plan-only: reject a stale gate/workload contract
        # before SDK setup, helper compilation or any device measurements.
        from dev.dense_helpers.reducer_gate import cases as reducer_cases
        reducer_cases(plan['points'])
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
        if paired and (a.output/'build').exists() and not (a.output/'build/receipt.json').exists():
            # Retain a failed setup attempt; no measurement can precede this receipt.
            failed=a.output/'build'
            if failed.is_symlink() or not failed.is_dir():raise ValueError('unexpected helper build path')
            failed.rename(a.output/('build-failed-'+str(time.time_ns())))
        receipt=read(a.output/'build/receipt.json') if (a.output/'build/receipt.json').exists() else build(a)
        from tools.build_kpack_prefill_runtime import build as build_dense
        if not (a.output/'build/dense/manifest.json').exists():
            if paired and (a.output/'build/dense').exists():
                failed=a.output/'build/dense'
                if failed.is_symlink() or not failed.is_dir():raise ValueError('unexpected dense build path')
                failed.rename(a.output/'build'/('dense-failed-'+str(time.time_ns())))
            build_dense(a.sdk,a.output/'build/dense',a.jobs)
        if receipt.get('native_reference',True)!=(not a.no_native):
            raise ValueError('build receipt reference mode differs')
        images=[('timer_sha256',a.output/'build/libpoint-timer.so')]
        if not a.no_native:images.append(('carrier_sha256',Path(receipt['carrier'])))
        for key,path in images:
            if sha(path)!=receipt[key]:raise ValueError('diagnostic executable changed')
        if not a.no_native:validate_native(receipt,a)
        payload=validate_payload(a)
        # Full geometry must be executable before compiling/timing thousands
        # of unrelated TC proposals. Queries allocate no weight/workspace.
        from dev.prefill_point.typed import DenseLibrary,workspace_gate
        full_gate=workspace_gate(plan['points'],DenseLibrary(a.output/'build/dense/libquactlize_ppu_prefill.so'))
        atomic_json(a.output/f'full-workspace-gate-{time.time_ns()}.json',full_gate)
        print('PREFILL_FULL_WORKSPACE '+json.dumps({k:full_gate[k] for k in ('status','cases','scope')}),flush=True)
        if any(c.get('helper_options',{}).get('vector_io')==4 for p in plan['points'] for c in p.get('cases',[])):
            # A fresh process, outside profiling/timing; exact built helper ABI
            # must pass tails/strides/NaNs/error flags before IO4 can be measured.
            execute([sys.executable,str(ROOT/'tools/check_dense_io.py'),'--sdk',str(a.sdk),
                     '--library',str(a.output/'build/dense/libquactlize_ppu_prefill.so'),
                     '--output',str(a.output/f'io-gate-{time.time_ns()}.json')],
                    a.output/f'io-gate-{time.time_ns()}.log',env)
        if has_compact_reducer:
            execute([sys.executable,str(ROOT/'tools/check_dense_reducer.py'),'--sdk',str(a.sdk),
                     '--library',str(Path(payload['path'])/'helpers/libstrategy_prefill.so'),
                     '--plan',str(plan_path),'--output',str(a.output/f'reducer-gate-{time.time_ns()}.json')],
                    a.output/f'reducer-gate-{time.time_ns()}.log',env)
        authority=dict(device=d,payload=payload,build=receipt,harness=source_seal(),plan=plan['plan_sha256'],
            cache=str(a.cache),asys_sha256=sha(a.asys),batch_size=a.configs_per_batch,frequency_normalization=False)
        if 'source_revision' in plan:
            from dev.strategy.smallm_asys import revision
            if plan['source_revision']!=revision(ROOT):raise ValueError('frozen measurement commit changed')
            authority['source_revision']=plan['source_revision']
        if old and old!=authority: raise ValueError('resume build/protocol changed')
        if not old:atomic_json(a.output/'authority.json',authority)
        clock=request_lock(a.smi,pci,1700);atomic_json(a.output/('clock-lock-'+str(time.time_ns())+'.json'),clock)
        if clock['returncode']: raise ValueError('1700MHz lock request failed')
        if a.helper_plan:
            points=plan['points'];modules={}
        else:
            cmd=[sys.executable,'-u',str(ROOT/'tools/probe_kpack_strategy.py'),'--plan',str(plan_path),
                '--output',str(a.output/'probe'),'--sdk',str(a.sdk),'--cache',str(a.cache),
                '--jobs',str(a.jobs),'--max-modules',str(a.max_modules if a.cohort or a.finalist_plan else 96)]
            if (a.output/'probe').exists(): cmd.append('--resume')
            execute(cmd,a.output/('resources-'+str(time.time_ns())+'.log'),env)
            resource=read(a.output/'probe/summary.json')
            if resource['status']!='COMPLETE': raise ValueError('resources incomplete; resume will reuse finished modules')
            points=read(a.output/'probe/resolved-plan.json')['points']
            queries={f.stem:read(f)['data'] for f in (a.output/'probe/queries').glob('*.json')}
            modules={f.stem:read(f)['data']['record'] for f in (a.output/'probe/modules').glob('*.json') if read(f)['data']['status']=='PASS'}
            for point in points:
                if a.finalist_plan:
                    from dev.prefill_asys.finalists import bind_cases
                    point['cases']=bind_cases(point,queries)
                else:
                    point['queries']=queries;point['cases']=runnable(point);point.pop('queries')
        atomic_json(a.output/'modules.json',modules)
        atomic_json(a.output/'execution-plan.json',dict(points=points,authority=digest(authority)))
        if any('cases' not in point for point in plan['points']):
            # Older cohort plans form concrete cases only after resource
            # resolution. Check those too, still before any timed batch.
            full_gate=workspace_gate(points,DenseLibrary(a.output/'build/dense/libquactlize_ppu_prefill.so'))
            atomic_json(a.output/f'full-workspace-gate-resolved-{time.time_ns()}.json',full_gate)
            print('PREFILL_FULL_WORKSPACE '+json.dumps({k:full_gate[k] for k in ('status','cases','scope')}),flush=True)
        total=sum((len(p['cases'])+int(p.get('native_comparison',True)))*p.get('confirmation_captures',1) for p in points)
        print(f'PREFILL_ASYS_READY shapes={len(points)} configs={total} production_changed=0',flush=True)
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
                smoke=[next((c for c in cs if c['arm']==arm and c['helper_mode']!='legacy'),None)
                       for arm in ('sf','full-fp16','full-bf16-typed')]
                smoke=[c for c in smoke if c]
                if smoke: jobs.append(smoke)
                rest=[c for c in cs if c['id'] not in {s['id'] for s in smoke}]
                jobs += [rest[i:i+a.configs_per_batch] for i in range(0,len(rest),a.configs_per_batch)]
                jobs=[(cases,0) for cases in jobs]
                if paired:
                    # Native, forward helper pool, reverse helper pool, native.
                    # First complete rings are excluded inside each capture.
                    jobs=confirmation_jobs(point)
                for cases,confirmation in jobs:
                    before=observe(a.smi,pci,1700,advisory=True)
                    try:
                        measured=profile_batch(a,point,cases,authority,env,confirmation)
                    except Exception as exc:
                        if not (a.cohort or paired): raise
                        atomic_json(a.output/('failed-'+point['id']+'-'+cases[0]['id']+'.json'),
                            dict(point=point['id'],cases=[c['id'] for c in cases],error=repr(exc)))
                        print('PREFILL_ASYS_BATCH_FAIL '+json.dumps(dict(point=point['id'],
                            cases=[c['id'] for c in cases],error=str(exc)[-4000:],remaining_continue=1)),flush=True)
                        measured=[]
                    for r in measured:
                        previous=measurements[point['id']].get(r['case_id'])
                        measurements[point['id']][r['case_id']]=confirmed_capture(previous,r) if paired else r
                    done+=len(cases)
                    after=observe(a.smi,pci,1700,advisory=True)
                    atomic_json(a.output/('clock-'+point['id']+'-'+cases[0]['id']+'.json'),dict(before=before,after=after))
                    status=summarize(points,measurements);atomic_json(a.output/'summary.json',status)
                    if a.helper_plan:
                        atomic_json(a.output/'helper-proposals.json',helper_review(points,measurements))
                    print('PREFILL_ASYS_PROGRESS '+json.dumps(dict(completed=done,total=total,
                        elapsed_minutes=(time.monotonic()-started)/60,shape=point['workload'],
                        remaining_minutes=(time.monotonic()-started)/done*(total-done)/60,
                        eta='OBSERVED_CONFIG_AVERAGE_NOT_GUARANTEE')),flush=True)
        print('PREFILL_ASYS_DONE '+json.dumps(status),flush=True)
        return 0 if status['status']=='SCREEN_COMPLETE' else 1
    except BaseException as e:
        atomic_json(a.output/('error-'+str(time.time_ns())+'.json'),dict(error=repr(e),traceback=traceback.format_exc()))
        raise
    finally:
        if (a.cohort or a.finalist_plan) and (a.output/'execution-plan.json').exists():
            review=a.output/('review-'+str(time.time_ns())+'.json.xz')
            exported=subprocess.run([sys.executable,str(ROOT/'tools/review_prefill_cohort.py'),
                '--run',str(a.output),'--output',str(review)])
            if exported.returncode not in (0,1) or not review.is_file():
                print('PREFILL_REVIEW_FAIL raw_results_retained=1; packaging diagnostic receipts',flush=True)
                package(a.output)
            else:print('results='+str(review),flush=True)
        else:package(a.output)


if __name__=='__main__':raise SystemExit(main())
