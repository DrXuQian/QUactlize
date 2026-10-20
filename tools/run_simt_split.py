#!/usr/bin/env python3
"""Matched shipping/candidate SIMT Split-K gate; production stays unchanged."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.split_plan import inventory, plan, configs, SPLITS
from dev.gemv_simt.build_split import build
from dev.gemv_simt.build_general import inputs
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_q6_metadata import execute
from tools.run_simt_general import sealed, read_cell


def cells(spec):
    return [(m,c,s) for m in spec['numeric_m']
            for c in range(len(configs(spec['workload']['qtype']))) for s in SPLITS]


def cell_path(output,index,m,config,split):
    return output/'points'/str(index)/f'm{m}-c{config}-s{split}.json'


def build_root(a):
    return a.build_root if getattr(a,'build_root',None) is not None else a.output/'build'


def build_args(a):
    return ['--build-root',str(a.build_root)] if getattr(a,'build_root',None) is not None else []


def binding(a,spec,authority):
    w=Workload(**spec['workload'])
    manifest=json.loads((build_root(a)/str(w.qtype)/'manifest.json').read_text())
    from quactlize.runtime.native import sdk_identity
    if (manifest['sources']!=inputs() or manifest.get('sdk')!=sdk_identity(a.sdk) or
            manifest['object_only'] or authority['harness']!=harness_identity(ROOT)):
        raise ValueError('source/SDK changed; completed cells cannot be mixed')
    record,=[r for r in manifest['records'] if r['compute']==w.compute]
    library=build_root(a)/str(w.qtype)/record['image']
    if record['configs']!=configs(w.qtype) or sha(library)!=record['sha256']:
        raise ValueError('compiled recipe/image changed')
    return library,dict(authority_sha256=digest(authority),library_sha256=record['sha256'],
                        point=spec['point'],workload=asdict(w),configs=configs(w.qtype))


def validate_cell(row,spec,m,split):
    whole=row['numerical']['whole'];copies=len({0,row['ring']['copies']-1})
    checks=whole['checks']
    if (len(checks)!=3*copies*2 or not whole['raw_equal'] or whole['zero_a']!='PASS' or
            whole['omitted']!='DETECTED' or not whole['plane_negatives']):
        raise ValueError('incomplete whole-output gate')
    partial=row['numerical']['partials']
    if split==1:
        if partial!='NOT_APPLICABLE_S1':raise ValueError('unexpected S1 partial evidence')
    else:
        if (len(partial['checks'])!=3*copies*2 or not partial['raw_equal'] or
                partial['omitted_producer']!='DETECTED' or partial['omitted_reducer']!='DETECTED'):
            raise ValueError('incomplete partition gate')
        checks=checks+partial['checks']
    if any(p['bad'] or p['nonfinite'] or not 0<=p['relative_l1_error']<.005 for p in checks):
        raise ValueError('invalid independent numerical evidence')
    t=row['timing']
    if bool(t)!=(m in spec['timing_m']):raise ValueError('wrong timing coverage')
    if t:
        from dev.strategy.components import statistics_checked
        if (len(t['producer'])!=2 or bool(t['reducer'])!=(split>1) or
                t['timed_kernel_types_per_interval']!=1 or t['host_launch_cost_added'] or not t['raw_equal']):
            raise ValueError('timing scope differs')
        for component in t['producer']+([t['reducer']] if split>1 else []):
            samples=component['samples_us']
            if len(samples)!=6 or any(len(r)!=5 for r in samples):raise ValueError('sample denominator differs')
            if statistics_checked(samples)!=component:raise ValueError('component statistics differ')
        costs=[p['median_us']+(t['reducer']['median_us'] if split>1 else 0) for p in t['producer']]
        if costs!=t['component_sum_us']:raise ValueError('component sum differs')
    return row


def access_receipt(w,recipe,split,bases):
    from dev.gemv_simt.access import pattern as shipping
    from dev.gemv_simt.general_access import pattern as candidate
    from quactlize.execution.simt_codegen import Config
    cfg=Config(**(recipe|dict(split=split)))
    rows=[]
    # First two loop trips cover every packet cohort for these geometries.
    for partition in range(split):
        for warp in range(cfg.warps):
            for iteration in (0,1):
                kw=dict(bases=bases,warp=warp,partition=partition,iteration=iteration)
                old=shipping(w.qtype,cfg,w.n,w.k,**kw)
                new=candidate(w.qtype,cfg,w.n,w.k,**kw)['candidate']
                def compact(v):
                    return [dict(name=s['name'],width_bytes=s['width_bytes'],footprint=s['footprint']) for s in v['streams']]
                rows.append(dict(partition=partition,warp=warp,iteration=iteration,
                                 lane_groups=old['lane_groups'],shipping=compact(old),candidate=compact(new)))
    return dict(scope='SOURCE_REQUEST_FOOTPRINT_NOT_MEASURED_DRAM',actual_bases_mod128=bases,rows=rows)


def child(a):
    from dev.gemv_simt.general_bench import Context, single_copy
    from dev.gemv_simt.split_bench import Weights, Kernel, numerical, timing, profile
    from dev.strategy.cold_decode import Ring
    from quactlize.runtime.native import SDK
    from tools.run_kpack_llama_ab import device
    spec=inventory()[a.child];w=Workload(**spec['workload'])
    declared=json.loads((a.output/'plan.json').read_text())
    if a.child not in {p['point'] for p in declared['points']}:raise ValueError('point belongs to another worker')
    authority=json.loads((a.output/'authority.json').read_text())
    if device(a)!=authority['device']:raise ValueError('physical device changed')
    library,identity=binding(a,spec,authority)
    if a.profile_arm is not None:
        path=cell_path(a.output,a.child,8,a.config,a.split)
        row=read_cell(path,identity|dict(m=8,config=a.config,split=a.split))
        if row is None or not row['timing']:raise ValueError('profile requires matching completed pair')
        validate_cell(row,spec,8,a.split)
        with ExitStack() as cleanup:
            sdk=SDK(a.sdk);weights=Weights(w.qtype,w.n,w.k,w.experts)
            ring=single_copy(sdk,weights,cleanup)
            b=Context(sdk,ring,replace(w,m=8));cleanup.callback(b.close)
            result=profile(Kernel(b,library,a.split),a.config,a.profile_arm)
            for buf in ring.buffers:buf.guard()
        sealed(a.result,result|dict(identity=identity,config=a.config,split=a.split,arm=a.profile_arm,prior_sha256=sha(path)))
        return 0
    due=[]
    for m,c,s in cells(spec):
        row=read_cell(cell_path(a.output,a.child,m,c,s),identity|dict(m=m,config=c,split=s))
        if row is None:due.append((m,c,s))
        else:validate_cell(row,spec,m,s)
    if not due:
        print(f'SPLIT_REUSE point={a.child} all_cells=PASS',flush=True);return 0
    sdk=SDK(a.sdk);weights=Weights(w.qtype,w.n,w.k,w.experts)
    print(f'SPLIT_FIXTURE point={a.child} pending={len(due)} q={w.qtype} n={w.n} k={w.k}',flush=True)
    with ExitStack() as cleanup:
        ring=Ring(sdk,w,a.l2_bytes,weights=weights,minimum_active=w.topk if w.operation=='grouped' else None)
        cleanup.callback(ring.close)
        for m in spec['numeric_m']:
            pending=[(c,s) for mm,c,s in due if mm==m]
            if not pending:continue
            b=Context(sdk,ring,replace(w,m=m))
            try:
                for c,s in pending:
                    print(f'SPLIT_START point={a.child} m={m} config={c} split={s}',flush=True)
                    core=Kernel(b,library,s)
                    proof=numerical(core,c)
                    measured=timing(core,c) if m in spec['timing_m'] else None
                    for buf in ring.buffers:buf.guard()
                    b.partial.guard()
                    access=None
                    if measured:
                        active=len(set(map(int,b.data['owners'])))
                        weight_bytes=sum(v.nbytes for v in weights.planes.values())//w.experts*active
                        measured.update(active_experts=active,compulsory_weight_bytes=weight_bytes,
                            modeled_full_call_weight_mbu_pct=[weight_bytes/us/27000 for us in measured['component_sum_us']],
                            assumed_peak_gbps=2700,mbu_scope='WEIGHT_BYTES_PER_COMPONENT_SUM_NOT_ACU_DRAM_BANDWIDTH')
                        access=a.output/'access'/f'p{a.child}-c{c}-s{s}.json'
                        bases={name:int(ptr or 0)%128 for name,ptr in ring.planes[0].items()};bases['A']=b.a.ptr%128
                        value=access_receipt(w,configs(w.qtype)[c],s,bases)
                        if access.exists():
                            if json.loads(access.read_text())!=value:raise ValueError('access identity changed')
                        else:atomic_json(access,value)
                    row=dict(status='PASS',identity=identity|dict(m=m,config=c,split=s),numerical=proof,timing=measured,
                             ring=ring.receipt,access_path=str(access.relative_to(a.output)) if access else None,
                             access_sha256=sha(access) if access else None,production_changed=False)
                    validate_cell(row,spec,m,s)
                    sealed(cell_path(a.output,a.child,m,c,s),row)
                    print(f'SPLIT_CELL point={a.child} m={m} config={c} split={s} status=PASS timed={int(measured is not None)}',flush=True)
            finally:b.close()
    return 0


def profile_pair(a,spec,split,attempt,authority):
    from dev.native_components.profile import acu_command, check_report
    from dev.gemv_simt.q6_cycles import compare
    index=spec['point'];saved=a.output/'profiles'/f'p{index}-s{split}.json'
    identity=dict(authority_sha256=digest(authority),point=index,split=split)
    prior=read_cell(saved,identity)
    if prior:
        path=cell_path(a.output,index,8,prior['config'],split)
        if sha(path)!=prior['prior_sha256']:raise ValueError('profile parent changed')
        for r in prior['reports']:
            for field in ('raw_csv','report'):
                path=Path(r[field])
                if not path.resolve().is_relative_to(a.output) or path.is_symlink() or sha(path)!=r[field+'_sha256']:
                    raise ValueError('saved profile changed')
        return prior
    rows=[json.loads(cell_path(a.output,index,8,c,split).read_text()) for c in range(len(configs(spec['workload']['qtype'])))]
    c=min(range(len(rows)),key=lambda c:rows[c]['timing']['component_sum_us'][1])
    row=rows[c];home=attempt/f'profile-p{index}-s{split}';home.mkdir();reports=[]
    for rnd in range(3):
        for arm in ((0,1) if rnd%2==0 else (1,0)):
            stem=f'r{rnd}-a{arm}';target=home/stem
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                     '--l2-bytes',str(a.l2_bytes),'--child',str(index),'--config',str(c),'--split',str(split),
                     '--profile-arm',str(arm),'--result',str(home/(stem+'.json')),*build_args(a)]
            print(f'SPLIT_ACU point={index} split={split} config={c} round={rnd+1}/3 arm={arm} graphs=0',flush=True)
            try:
                if execute(acu_command(a.acu,target,command,graph_profiling='none'),home/(stem+'.log')):
                    raise ValueError('ACU child failed')
                paths=[p for p in (target,target.with_suffix('.acurep')) if p.is_file()]
                if len(paths)!=1:raise ValueError('missing/ambiguous ACU report')
                raw=home/(stem+'.raw.csv')
                if execute([str(a.acu),'--import',str(paths[0]),'--page','raw','--csv'],raw):raise ValueError('ACU import failed')
                reports.append(dict(status='PASS',round=rnd,arm=arm,raw_csv=str(raw),raw_csv_sha256=sha(raw),
                    report=str(paths[0]),report_sha256=sha(paths[0]),
                    **check_report(raw.read_text(errors='replace'),row['timing']['nodes'][arm][0])))
            except Exception as e:reports.append(dict(status='FAIL',round=rnd,arm=arm,error=str(e)))
    result=dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',identity=identity,config=c,
                reports=reports,prior_sha256=sha(cell_path(a.output,index,8,c,split)))
    try:
        expected={arm:(tuple(row['timing']['nodes'][arm][0]['grid']),tuple(row['timing']['nodes'][arm][0]['block'])) for arm in (0,1)}
        result['cycles']=compare(reports,expected_geometries=expected)
    except ValueError as e:result.update(status='FAIL',error=str(e))
    sealed(home/'summary.json',result)
    if result['status']=='PASS':sealed(saved,result)
    return result


def collect(a):
    selected=None if a.points=='all' else [int(i) for i in a.points.split(',')]
    frozen=plan(selected)
    if not a.resume:a.output.mkdir(parents=True,exist_ok=False)
    elif not a.output.is_dir():raise ValueError('resume directory missing')
    pp=a.output/'plan.json'
    if pp.exists():
        if json.loads(pp.read_text())!=frozen:raise ValueError('resume plan changed')
    else:atomic_json(pp,frozen)
    print('SPLIT_PLAN '+json.dumps({k:v for k,v in frozen.items() if k not in ('points','configs')}),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('confirm one otherwise idle GPU')
    from tools.run_kpack_llama_ab import device
    from dev.strategy.isolated import check_sdk_graph_abi
    actual=device(a);a.l2_bytes=actual['l2']['bytes']
    authority=dict(device=actual,harness=harness_identity(ROOT),graph_abi=check_sdk_graph_abi(a.sdk),
                   clock_policy='SAME_RUN_ALTERNATING_PAIRS_NO_HISTORICAL_TIMING_REUSE',clock_lock_verified=False)
    ap=a.output/'authority.json'
    if ap.exists():
        if json.loads(ap.read_text())!=authority:raise ValueError('resume source/device/SDK differs')
    else:atomic_json(ap,authority)
    lock=(Path('/tmp')/('quactlize-components-'+digest(actual['device']['pci'])[:24]+'.lock')).open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        compiled=build_root(a)
        if not compiled.exists() and a.build_root is None:build(a.sdk,compiled)
        if not (compiled/'manifest.json').is_file():raise ValueError('incomplete build; shared builds must be prepared by the coordinator')
        attempt=a.output/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        points=[];completed=0;timings=[];started=time.monotonic()
        for done,spec in enumerate(frozen['points'],1):
            index=spec['point']
            rc=execute([sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),
                        '--output',str(a.output),'--l2-bytes',str(a.l2_bytes),'--child',str(index),*build_args(a)],attempt/f'point-{index}.log')
            _,identity=binding(a,spec,authority);count=0
            for m,c,s in cells(spec):
                row=read_cell(cell_path(a.output,index,m,c,s),identity|dict(m=m,config=c,split=s))
                if row is None:continue
                validate_cell(row,spec,m,s);count+=1
                if row['timing']:
                    path=a.output/row['access_path']
                    if sha(path)!=row['access_sha256']:raise ValueError('access receipt changed')
                    t=row['timing'];components=t['producer']+([t['reducer']] if s>1 else [])
                    timings.append(dict(point=index,m=m,config=c,split=s,delta_pct=t['delta_pct'],
                                        component_sum_us=t['component_sum_us'],noisy=any(v['spread_pct']>5 for v in components)))
            completed+=count
            points.append(dict(point=index,rc=rc,completed=count,expected=len(cells(spec))))
            elapsed=time.monotonic()-started
            print(f'SPLIT_PROGRESS families={done}/{len(frozen["points"])} pairs={completed}/{frozen["numeric_pairs"]} '
                  f'failed_families={sum(bool(p["rc"]) for p in points)} elapsed_minutes={elapsed/60:.1f} '
                  f'remaining_minutes={elapsed/done*(len(frozen["points"])-done)/60:.1f} '
                  'eta=OBSERVED_FAMILY_AVERAGE_NOT_GUARANTEE remaining_continue=1',flush=True)
        profiles=[]
        if a.acu:
            for spec,p in zip(frozen['points'],points):
                w=spec['workload']
                if p['rc'] or p['completed']!=p['expected']:continue
                if spec['kind']=='all-format-control' and w['compute']=='bf16' and w['operation']=='indexed':
                    for split in SPLITS:
                        try:profiles.append(profile_pair(a,spec,split,attempt,authority))
                        except Exception as e:
                            profiles.append(dict(status='FAIL',point=spec['point'],split=split,error=str(e)))
        expected_profiles=(sum(p['kind']=='all-format-control' and p['workload']['compute']=='bf16' and
                               p['workload']['operation']=='indexed' for p in frozen['points'])*len(SPLITS)) if a.acu else 0
        status='PASS' if (completed==frozen['numeric_pairs'] and not any(p['rc'] for p in points) and
                          len(profiles)==expected_profiles and all(p['status']=='PASS' for p in profiles)) else 'INCOMPLETE'
        summary=dict(status=status,completed=completed,expected=frozen['numeric_pairs'],points=points,timings=timings,
                     profiles=profiles,profile_cohorts_expected=expected_profiles,
                     regressions_over_5pct=sum(t['delta_pct']>5 for t in timings),noisy_pairs=sum(t['noisy'] for t in timings),
                     elapsed_seconds=time.monotonic()-started,production_changed=False,performance_admitted=False,
                     verdict='REVIEW_REQUIRED_NOT_AUTOMATIC_PROMOTION',scope=frozen['scope'])
        atomic_json(a.output/'summary.json',summary)
        print('SPLIT_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ('points','timings','profiles')}),flush=True)
        return int(status!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--plan-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--child',type=int)
    p.add_argument('--points',default='all');p.add_argument('--build-root',type=Path)
    p.add_argument('--probe-only',action='store_true');p.add_argument('--no-pack',action='store_true')
    p.add_argument('--acu',type=Path);p.add_argument('--profile-arm',type=int,choices=(0,1))
    p.add_argument('--config',type=int);p.add_argument('--split',type=int,choices=SPLITS);p.add_argument('--result',type=Path)
    a=p.parse_args();a.sdk=a.sdk.resolve();a.output=a.output.resolve()
    if a.build_root is not None:a.build_root=a.build_root.resolve()
    if a.probe_only:
        from tools.run_kpack_llama_ab import device
        atomic_json(a.output/'device.json',device(a));return 0
    if a.child is not None:
        if a.child not in range(len(inventory())):p.error('invalid child point')
        if a.profile_arm is not None and (a.config is None or a.split is None or a.result is None):p.error('profile requires config/split/result')
        return child(a)
    try:return collect(a)
    finally:
        if a.output.is_dir() and not a.no_pack:
            from tools.pack_simt_split_review import package
            package(a.output)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
