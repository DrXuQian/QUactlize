#!/usr/bin/env python3
"""Bounded all-format reader validation, cold screen and direct ACU profiles."""
import argparse
from contextlib import ExitStack
from dataclasses import replace,asdict
import fcntl
import json
from pathlib import Path
import re
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.general_plan import inventory,plan,configs
from dev.gemv_simt.build_general import build,inputs
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_q6_metadata import execute


def sealed(path,value):
    if path.exists():raise ValueError('refusing to replace completed evidence: '+str(path))
    atomic_json(path,value|dict(record_sha256=digest(value)))


def read_cell(path,identity):
    if not path.exists():return None
    row=json.loads(path.read_text())
    if (row.get('identity')!=identity or row.get('status')!='PASS' or
            row.get('record_sha256')!=digest({k:v for k,v in row.items() if k!='record_sha256'})):
        raise ValueError('completed cell identity/hash differs: '+str(path))
    return row


def binding(a,w):
    from tools.run_kpack_llama_ab import device
    from quactlize.runtime.native import sdk_identity
    authority=json.loads((a.output/'authority.json').read_text())
    manifest=json.loads((a.output/'build'/str(w.qtype)/'manifest.json').read_text())
    if (authority['device']!=device(a) or authority['harness']!=harness_identity(ROOT) or
            manifest.get('sdk')!=sdk_identity(a.sdk) or manifest['sources']!=inputs() or manifest['object_only']):
        raise ValueError('source/SDK/device changed; do not mix experiments')
    record,=[r for r in manifest['records'] if r['compute']==w.compute]
    library=a.output/'build'/str(w.qtype)/record['image']
    if record['configs']!=configs(w.qtype) or sha(library)!=record['sha256']:raise ValueError('compiled recipe/image changed')
    return library,dict(authority_sha256=digest(authority),library_sha256=record['sha256'],point=a.child,
                        workload=asdict(w),configs=configs(w.qtype))


def complete_point(output,spec,authority):
    index=spec['point'];w=Workload(**spec['workload'])
    manifest=json.loads((output/'build'/str(w.qtype)/'manifest.json').read_text())
    record,=[r for r in manifest['records'] if r['compute']==w.compute]
    identity=dict(authority_sha256=digest(authority),library_sha256=record['sha256'],point=index,
                  workload=asdict(w),configs=configs(w.qtype))
    for m in spec['numeric_m']:
        for config in range(len(configs(w.qtype))):
            cell=read_cell(output/'points'/str(index)/f'm{m}-c{config}.json',identity|dict(m=m,config=config))
            if cell is None:raise ValueError(f'missing completed cell m={m} config={config}')
            if (not cell['numerical']['raw_equal'] or not cell['numerical']['checks'] or
                    bool(cell['timing'])!=(m in spec['timing_m'])):
                raise ValueError('incomplete numerical/timing evidence')
            if any(p['bad'] or p['nonfinite'] or not 0<=p['relative_l1_error']<.005
                   for p in cell['numerical']['checks']):raise ValueError('invalid numeric evidence')
    return True


def child(a):
    from dev.gemv_simt.general_bench import Context,Kernel,Weights,single_copy,numerical,timing,profile
    from dev.gemv_simt.general_access import pattern
    from dev.strategy.cold_decode import Ring
    from quactlize.execution.simt_codegen import Config
    from quactlize.runtime.native import SDK
    spec=inventory()[a.child];w=Workload(**spec['workload'])
    library,identity=binding(a,w);sdk=SDK(a.sdk)
    home=a.output/'points'/str(a.child);home.mkdir(parents=True,exist_ok=True)
    if a.profile_arm is not None:
        row=read_cell(home/f'm8-c{a.config}.json',identity|dict(m=8,config=a.config))
        if row is None or not row.get('timing'):raise ValueError('profile requires matching completed numerical/timing pair')
        with ExitStack() as cleanup:
            weights=Weights(w.qtype,w.n,w.k,w.experts)
            ring=single_copy(sdk,weights,cleanup)
            b=Context(sdk,ring,replace(w,m=8));cleanup.callback(b.close)
            result=profile(Kernel(b,library),a.config,a.profile_arm)
            for buf in ring.buffers:buf.guard()
        sealed(a.result,result|dict(identity=identity,config=a.config,arm=a.profile_arm,
                                   prior_sha256=sha(home/f'm8-c{a.config}.json')))
        return 0
    due=[(m,c) for m in spec['numeric_m'] for c in range(len(configs(w.qtype)))
         if read_cell(home/f'm{m}-c{c}.json',identity|dict(m=m,config=c)) is None]
    if not due:
        print(f'GENERAL_REUSE point={a.child} all_cells=PASS',flush=True);return 0
    weights=Weights(w.qtype,w.n,w.k,w.experts)
    with ExitStack() as cleanup:
        # One ring per family, reused over M/configs. Profiling never enters
        # here: it uses the one-copy path above and no graph API calls.
        ring=Ring(sdk,w,a.l2_bytes,weights=weights,minimum_active=w.topk if w.operation=='grouped' else None)
        cleanup.callback(ring.close)
        for m in spec['numeric_m']:
            pending=[c for mm,c in due if mm==m]
            if not pending:continue
            b=Context(sdk,ring,replace(w,m=m))
            try:
                core=Kernel(b,library)
                for config in pending:
                    recipe=configs(w.qtype)[config]
                    proof=numerical(core,config)
                    measured=timing(core,config,recipe) if m in spec['timing_m'] else None
                    for buf in ring.buffers:buf.guard()
                    bases={name:int(ptr or 0)%128 for name,ptr in ring.planes[0].items()};bases['A']=b.a.ptr%128
                    cfg=Config(**{k:recipe[k] for k in ('variant','columns','warps','values','split')})
                    patterns=[pattern(w.qtype,cfg,w.n,w.k,bases=bases,warp=warp) for warp in range(cfg.warps)] if measured else []
                    value=dict(status='PASS',identity=identity|dict(m=m,config=config),numerical=proof,timing=measured,
                               ring=ring.receipt,patterns=patterns,production_changed=False)
                    sealed(home/f'm{m}-c{config}.json',value)
                    print(f'GENERAL_CELL point={a.child} m={m} config={config} status=PASS timed={int(measured is not None)}',flush=True)
            finally:b.close()
    return 0


def profile_point(a,index,run):
    from dev.native_components.profile import acu_command,check_report
    from dev.gemv_simt.q6_cycles import compare
    spec=inventory()[index];q=spec['workload']['qtype'];home=a.output/'points'/str(index)
    rows=[json.loads((home/f'm8-c{c}.json').read_text()) for c in range(len(configs(q)))]
    config=min(range(len(rows)),key=lambda c:rows[c]['timing']['timings'][1]['median_us'])
    row=rows[config];reports=[]
    folder=run/f'profile-{index}';folder.mkdir()
    for rnd in range(3):
        for arm in ((0,1) if rnd%2==0 else (1,0)):
            stem=f'r{rnd}-a{arm}';target=folder/stem
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                     '--l2-bytes',str(a.l2_bytes),'--child',str(index),'--config',str(config),
                     '--profile-arm',str(arm),'--result',str(folder/(stem+'.json'))]
            print(f'GENERAL_ACU point={index} config={config} round={rnd+1}/3 arm={arm} direct_calls=1',flush=True)
            rc=execute(acu_command(a.acu,target,command,graph_profiling='none'),folder/(stem+'.log'))
            try:
                if rc:raise ValueError(f'profile child rc={rc}')
                paths=[p for p in (target,target.with_suffix('.acurep')) if p.is_file()]
                if len(paths)!=1:raise ValueError('missing/ambiguous ACU report')
                raw=folder/(stem+'.raw.csv')
                if execute([str(a.acu),'--import',str(paths[0]),'--page','raw','--csv'],raw):raise ValueError('ACU import failed')
                reports.append(dict(status='PASS',round=rnd,arm=arm,raw_csv=str(raw),raw_sha256=sha(raw),report=str(paths[0]),report_sha256=sha(paths[0]),
                                    **check_report(raw.read_text(errors='replace'),row['timing']['nodes'][arm])))
            except Exception as e:reports.append(dict(status='FAIL',round=rnd,arm=arm,error=str(e)))
    authority=json.loads((a.output/'authority.json').read_text())
    result=dict(point=index,config=config,reports=reports,status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAIL',
                identity=dict(authority_sha256=digest(authority),point=index),prior_sha256=sha(home/f'm8-c{config}.json'))
    try:
        expected={a:(tuple(row['timing']['nodes'][a]['grid']),tuple(row['timing']['nodes'][a]['block'])) for a in (0,1)}
        result['cycles']=compare(reports,expected_geometries=expected)
    except ValueError as e:result.update(status='FAIL',error=str(e))
    sealed(folder/'summary.json',result)
    if result['status']=='PASS':sealed(a.output/'profiles'/f'{index}.json',result)
    return result


def saved_profile(output,index,authority):
    value=read_cell(output/'profiles'/f'{index}.json',dict(authority_sha256=digest(authority),point=index))
    if value is None:return None
    prior=output/'points'/str(index)/f'm8-c{value["config"]}.json'
    if sha(prior)!=value['prior_sha256']:raise ValueError('profile timing/numerical parent changed')
    for row in value['reports']:
        for field,key in (('report','report_sha256'),('raw_csv','raw_sha256')):
            path=Path(row[field])
            if not path.resolve().is_relative_to(output) or path.is_symlink() or sha(path)!=row[key]:
                raise ValueError('completed ACU report changed')
    return value


def collect(a):
    selected=None if a.points=='all' else list(range(36)) if a.points=='controls' else [int(i) for i in a.points.split(',')]
    frozen=plan(selected)
    if not a.resume:a.output.mkdir(parents=True,exist_ok=False)
    elif not a.output.is_dir():raise ValueError('resume output missing')
    old=a.output/'plan.json'
    if old.exists():
        if json.loads(old.read_text())!=frozen:raise ValueError('resume plan changed')
    else:atomic_json(old,frozen)
    print('GENERAL_PLAN '+json.dumps({k:v for k,v in frozen.items() if k not in ('points','configs')}),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('confirm one otherwise idle GPU')
    from tools.run_kpack_llama_ab import device
    actual=device(a);a.l2_bytes=actual['l2']['bytes']
    authority=dict(device=actual,harness=harness_identity(ROOT))
    ap=a.output/'authority.json'
    if ap.exists():
        if json.loads(ap.read_text())!=authority:raise ValueError('resume source/device/SDK differs')
    else:atomic_json(ap,authority)
    lock=(Path('/tmp')/('quactlize-components-'+digest(actual['device']['pci'])[:24]+'.lock')).open('a')
    with lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (a.output/'build').exists():build(a.sdk,a.output/'build')
        if not (a.output/'build/manifest.json').is_file():raise ValueError('build incomplete; inspect build logs before retry')
        run=a.output/'attempts'/str(time.time_ns());run.mkdir(parents=True)
        rows=[];started=time.monotonic()
        for done,spec in enumerate(frozen['points'],1):
            index=spec['point']
            rc=execute([sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                        '--l2-bytes',str(a.l2_bytes),'--child',str(index)],run/f'point-{index}.log')
            if not rc:
                try:complete_point(a.output,spec,authority)
                except Exception as e:
                    rc=1;print(f'GENERAL_INCOMPLETE point={index} error={e}',flush=True)
            rows.append(dict(point=index,status='PASS' if rc==0 else 'FAIL',rc=rc))
            elapsed=time.monotonic()-started
            print(f'GENERAL_PROGRESS completed={done}/{len(frozen["points"])} failed={sum(r["rc"]!=0 for r in rows)} '
                  f'elapsed_minutes={elapsed/60:.1f} remaining_minutes={elapsed/done*(len(frozen["points"])-done)/60:.1f} '
                  'eta=OBSERVED_FAMILY_AVERAGE_NOT_GUARANTEE remaining_continue=1',flush=True)
        profiles=[]
        if a.acu:
            # Six independent format controls, M8 indexed/BF16. Profile only
            # the best *screened* recipe versus its own same-geometry control.
            for row in rows:
                if row['point']<36 and row['point']%6==4 and row['status']=='PASS':
                    try:profiles.append(saved_profile(a.output,row['point'],authority) or profile_point(a,row['point'],run))
                    except Exception as e:profiles.append(dict(point=row['point'],status='FAIL',error=str(e)))
        status='PASS' if all(r['status']=='PASS' for r in rows+profiles) else 'INCOMPLETE'
        summary=dict(status=status,points=rows,profiles=profiles,elapsed_seconds=time.monotonic()-started,
                     production_changed=False,performance_admitted=False,scope=frozen['scope'])
        atomic_json(a.output/'summary.json',summary)
        print('GENERAL_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ('points','profiles')}),flush=True)
        return int(status!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--points',default='all');p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--confirm-idle',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--acu',type=Path)
    p.add_argument('--child',type=int);p.add_argument('--config',type=int);p.add_argument('--profile-arm',type=int,choices=(0,1))
    p.add_argument('--result',type=Path)
    a=p.parse_args();a.sdk=a.sdk.resolve();a.output=a.output.resolve()
    if a.child is not None:
        if a.child not in range(len(inventory())):p.error('invalid child point')
        if a.profile_arm is not None and (a.config is None or a.result is None):p.error('profile requires config/result')
        return child(a)
    try:return collect(a)
    finally:
        if a.output.is_dir():
            archive=a.output/f'review-{time.time_ns()}.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                for path in sorted(a.output.rglob('*')):
                    report_without_extension=bool(re.fullmatch(r'r[0-2]-a[01]',path.name) and path.parent.name.startswith('profile-'))
                    if path.is_file() and not path.is_symlink() and (path.suffix in ('.json','.log','.csv','.txt','.acurep') or report_without_extension):
                        tar.add(path,arcname=str(path.relative_to(a.output)),recursive=False)
            print('results='+str(archive),flush=True)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
