#!/usr/bin/env python3
"""Bounded Q8 dense causal screen; unchanged production and no global sweep."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict,replace
import fcntl
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.build_q8_dense import build,inputs
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.q8_dense_plan import plan,inventory
from dev.gemv_simt.q8_dense_source import ARMS
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import harness_identity
from quactlize.dispatch.table import Workload
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_q6_metadata import execute
from tools.run_simt_general import sealed,read_cell


def identity(output,point,authority):
    w=Workload(**inventory()[point]['workload'])
    manifest=json.loads((output/'build/candidate/manifest.json').read_text())
    row,=[r for r in manifest['records'] if r['compute']==w.compute]
    return dict(authority_sha256=digest(authority),library_sha256=row['sha256'],point=point,
                workload=asdict(w),configs=configs(8),arms=list(ARMS))


def cell_path(output,point,m,config,variant):
    return output/'points'/str(point)/f'm{m}-c{config}-v{variant}.json'


def cell(output,point,m,config,variant,authority):
    key=identity(output,point,authority)|dict(m=m,config=config,variant=variant)
    return read_cell(cell_path(output,point,m,config,variant),key)


def complete_point(output,point,authority):
    spec=inventory()[point]
    for m in spec['numeric_m']:
        for config in range(len(configs(8))):
            for variant in range(1,len(ARMS)):
                row=cell(output,point,m,config,variant,authority)
                if row is None:raise ValueError('missing completed numeric pair')
                n=row['numerical']
                if (not n['raw_equal'] or len(n['checks'])!=12 or n['zero_a']!='PASS' or
                        n['omitted']!='DETECTED' or set(n['plane_negatives'])!={'low','units'} or
                        bool(row['timing'])!=(m in spec['timing_m'])):
                    raise ValueError('incomplete Q8 numerical/timing proof')
                if any(p['bad'] or p['nonfinite'] or not 0<=p['relative_l1_error']<.005 for p in n['checks']):
                    raise ValueError('invalid Q8 numerical evidence')


def binding(a):
    from tools.run_kpack_llama_ab import device
    from quactlize.runtime.native import sdk_identity
    authority=json.loads((a.output/'authority.json').read_text())
    manifest=json.loads((a.output/'build/candidate/manifest.json').read_text())
    if (authority['device']!=device(a) or authority['harness']!=harness_identity(ROOT) or
            manifest['sdk']!=sdk_identity(a.sdk) or manifest['sources']!=inputs() or manifest['object_only']):
        raise ValueError('Q8 source/SDK/device changed')
    w=Workload(**inventory()[a.child]['workload'])
    row,=[r for r in manifest['records'] if r['compute']==w.compute]
    library=a.output/'build/candidate'/row['image']
    if row['configs']!=configs(8) or sha(library)!=row['sha256']:raise ValueError('Q8 compiled image/config changed')
    return w,authority,library


def child(a):
    from dev.gemv_simt.general_bench import Context,Weights,numerical,profile,single_copy
    from dev.gemv_simt.q8_dense_bench import Kernel,timing
    from dev.gemv_simt.general_access import pattern
    from dev.strategy.cold_decode import Ring
    from quactlize.execution.simt_codegen import Config
    from quactlize.runtime.native import SDK
    w,authority,library=binding(a);key=identity(a.output,a.child,authority);sdk=SDK(a.sdk)
    spec=inventory()[a.child];home=a.output/'points'/str(a.child);home.mkdir(parents=True,exist_ok=True)
    if a.profile_arm is not None:
        variant=a.profile_arm or 5
        prior=cell(a.output,a.child,a.m,a.config,variant,authority)
        if prior is None or not prior['timing']:raise ValueError('profile lacks the exact numeric/timing parent')
        with ExitStack() as cleanup:
            weights=Weights(8,w.n,w.k,1);ring=single_copy(sdk,weights,cleanup)
            b=Context(sdk,ring,replace(w,m=a.m));cleanup.callback(b.close)
            result=profile(Kernel(b,library,variant),a.config,int(a.profile_arm!=0))
            for buf in ring.buffers:buf.guard()
        sealed(a.result,result|dict(identity=key|dict(m=a.m,config=a.config,profile_arm=a.profile_arm),
                                   prior_sha256=sha(cell_path(a.output,a.child,a.m,a.config,variant))))
        return 0
    due=[(m,c,v) for m in spec['numeric_m'] for c in range(len(configs(8))) for v in range(1,len(ARMS))
         if cell(a.output,a.child,m,c,v,authority) is None]
    if not due:
        print(f'Q8_DENSE_REUSE point={a.child} all_cells=PASS',flush=True);return 0
    with ExitStack() as cleanup:
        weights=Weights(8,w.n,w.k,1);ring=Ring(sdk,w,a.l2_bytes,weights=weights);cleanup.callback(ring.close)
        for m in spec['numeric_m']:
            pending=[(c,v) for mm,c,v in due if mm==m]
            if not pending:continue
            b=Context(sdk,ring,replace(w,m=m))
            try:
                for config,variant in pending:
                    core=Kernel(b,library,variant);recipe=configs(8)[config]
                    proof=numerical(core,config)
                    measured=timing(core,config,recipe) if m in spec['timing_m'] else None
                    for buf in ring.buffers:buf.guard()
                    bases={n:int(p or 0)%128 for n,p in ring.planes[0].items()};bases['A']=b.a.ptr%128
                    cfg=Config(**{k:recipe[k] for k in ('variant','columns','warps','values','split')})
                    patterns=[pattern(8,cfg,w.n,w.k,bases=bases,warp=i) for i in range(cfg.warps)] if measured else []
                    sealed(cell_path(a.output,a.child,m,config,variant),
                           dict(status='PASS',identity=key|dict(m=m,config=config,variant=variant),
                                numerical=proof,timing=measured,patterns=patterns,ring=ring.receipt,production_changed=False))
                    print(f'Q8_DENSE_CELL point={a.child} m={m} config={config} variant={variant} status=PASS timed={int(measured is not None)}',flush=True)
            finally:b.close()
    return 0


def profile_focus(a,focus,attempt,authority):
    from dev.native_components.profile import acu_command,check_report
    from dev.gemv_simt.q6_cycles import compare,metrics,RATES
    p,m,c=focus['point'],focus['m'],focus['config']
    name=f'p{p}-m{m}-c{c}';folder=attempt/name;folder.mkdir()
    canonical=a.output/'profiles'/name;canonical.mkdir(parents=True,exist_ok=True)
    key=identity(a.output,p,authority)|dict(m=m,config=c)
    reports=[]
    for rnd in range(3):
        for arm in (range(len(ARMS)) if rnd%2==0 else reversed(range(len(ARMS)))):
            stem=f'r{rnd}-a{arm}';path=canonical/(stem+'.json')
            k=key|dict(round=rnd,arm=arm)
            saved=read_cell(path,k)
            variant=arm or 5;prior=cell_path(a.output,p,m,c,variant)
            if saved:
                if saved['prior_sha256']!=sha(prior):raise ValueError('profile parent changed')
                for field in ('report','raw_csv'):
                    f=Path(saved[field])
                    if not f.resolve().is_relative_to(a.output) or f.is_symlink() or sha(f)!=saved[field+'_sha256']:
                        raise ValueError('profile report hash/path changed')
                reports.append(saved);continue
            target=folder/stem
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                     '--l2-bytes',str(a.l2_bytes),'--child',str(p),'--m',str(m),'--config',str(c),
                     '--profile-arm',str(arm),'--result',str(folder/(stem+'.json'))]
            print(f'Q8_DENSE_ACU focus={name} round={rnd+1}/3 arm={arm} direct_calls=1',flush=True)
            try:
                rc=execute(acu_command(a.acu,target,command,graph_profiling='none'),folder/(stem+'.log'))
                if rc:raise ValueError(f'profile child rc={rc}')
                receipt=read_cell(folder/(stem+'.json'),key|dict(profile_arm=arm))
                if (receipt is None or receipt['prior_sha256']!=sha(prior) or receipt['graphs']!=0 or
                        receipt['timed_calls']!=1 or receipt['weight_copies']!=1 or len(receipt['checks'])!=4 or
                        any(r['bad'] or r['nonfinite'] or not 0<=r['relative_l1_error']<.005 for r in receipt['checks'])):
                    raise ValueError('direct ACU child proof is missing or invalid')
                paths=[f for f in (target,target.with_suffix('.acurep')) if f.is_file()]
                if len(paths)!=1:raise ValueError('missing/ambiguous ACU report')
                raw=folder/(stem+'.raw.csv')
                if execute([str(a.acu),'--import',str(paths[0]),'--page','raw','--csv'],raw):raise ValueError('ACU import failed')
                text=raw.read_text(errors='replace');values=metrics(text)
                if abs(values['ce__cycles_elapsed.avg.per_second']/authority['expected_ce_hz']-1)>.001:
                    raise ValueError('CE clock differs from requested regime; timing not admitted')
                anchor=next((r for r in reports if r['status']=='PASS'),None)
                if anchor:
                    clocks=metrics(Path(anchor['raw_csv']).read_text(errors='replace'))
                    if any(abs(values[k]/clocks[k]-1)>.001 for k in RATES):
                        raise ValueError('core/memory clock ratio differs from this cohort; retain prior reports')
                node=json.loads(prior.read_text())['timing']['nodes'][int(arm!=0)]
                value=dict(status='PASS',identity=k,round=rnd,arm=arm,prior_sha256=sha(prior),
                           report=str(paths[0]),report_sha256=sha(paths[0]),raw_csv=str(raw),raw_csv_sha256=sha(raw),
                           **check_report(text,node))
                sealed(path,value);reports.append(value)
            except Exception as e:reports.append(dict(status='FAIL',round=rnd,arm=arm,error=str(e)))
    comparisons=[]
    for arm in range(1,len(ARMS)):
        selected=[r|dict(arm=int(r['arm']!=0)) for r in reports if r['arm'] in (0,arm)]
        try:
            row=cell(a.output,p,m,c,arm,authority)
            geometry={i:(tuple(n['grid']),tuple(n['block'])) for i,n in enumerate(row['timing']['nodes'])}
            cycles=compare(selected,expected_geometries=geometry)
            comparisons.append(dict(arm=arm,status='PASS' if cycles['clock_conditions_match'] else 'FAIL',cycles=cycles))
        except ValueError as e:comparisons.append(dict(arm=arm,status='FAIL',error=str(e)))
    result=dict(focus=focus,reports=reports,comparisons=comparisons,
                status='PASS' if all(r['status']=='PASS' for r in reports+comparisons) else 'FAIL')
    sealed(folder/'summary.json',result);return result


def collect(a):
    frozen=plan()
    if a.resume:
        if not a.output.is_dir():raise ValueError('missing resume output')
    else:a.output.mkdir(parents=True,exist_ok=False)
    if (a.output/'plan.json').exists():
        if json.loads((a.output/'plan.json').read_text())!=frozen:raise ValueError('Q8 frozen plan changed')
    else:atomic_json(a.output/'plan.json',frozen)
    print('Q8_DENSE_PLAN '+json.dumps({k:v for k,v in frozen.items() if k not in ('points','arms','configs')}),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle:raise ValueError('confirm one otherwise idle GPU')
    from tools.run_kpack_llama_ab import device
    actual=device(a);a.l2_bytes=actual['l2']['bytes']
    authority=dict(device=actual,harness=harness_identity(ROOT),expected_ce_hz=a.expected_ce_ghz*1e9)
    path=a.output/'authority.json'
    if path.exists():
        if json.loads(path.read_text())!=authority:raise ValueError('resume source/device/clock regime differs')
    else:atomic_json(path,authority)
    with (Path('/tmp')/('quactlize-components-'+digest(actual['device']['pci'])[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (a.output/'build').exists():build(a.sdk,a.output/'build')
        if not (a.output/'build/manifest.json').is_file():raise ValueError('incomplete build; inspect existing diagnostics')
        attempt=a.output/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
        started=time.monotonic();rows=[]
        for spec in frozen['points']:
            point=spec['point']
            rc=execute([sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--output',str(a.output),
                        '--l2-bytes',str(a.l2_bytes),'--child',str(point)],attempt/f'point-{point}.log')
            if not rc:
                try:complete_point(a.output,point,authority)
                except Exception as e:rc=1;print(f'Q8_DENSE_INCOMPLETE point={point} error={e}',flush=True)
            rows.append(dict(point=point,status='PASS' if rc==0 else 'FAIL',rc=rc))
            elapsed=time.monotonic()-started;done=len(rows)
            print(f'Q8_DENSE_PROGRESS completed={done}/{len(frozen["points"])} failed={sum(r["rc"]!=0 for r in rows)} '
                  f'elapsed_minutes={elapsed/60:.1f} remaining_minutes={elapsed/done*(len(frozen["points"])-done)/60:.1f} '
                  'eta=SCREEN_ONLY_OBSERVED_NOT_GUARANTEE remaining_continue=1',flush=True)
        profiles=[]
        if a.acu:
            for focus in frozen['profiles']:
                if rows[focus['point']]['status']!='PASS':continue
                try:profiles.append(profile_focus(a,focus,attempt,authority))
                except Exception as e:profiles.append(dict(focus=focus,status='FAIL',error=str(e)))
        status='PASS' if len(profiles)==len(frozen['profiles']) and all(r['status']=='PASS' for r in rows+profiles) else 'INCOMPLETE'
        summary=dict(status=status,points=rows,profiles=profiles,elapsed_seconds=time.monotonic()-started,
                     scope=frozen['scope'],production_changed=False,performance_admitted=False)
        atomic_json(a.output/'summary.json',summary)
        print('Q8_DENSE_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ('points','profiles')}),flush=True)
        return int(status!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--expected-ce-ghz',type=float,default=1.7)
    p.add_argument('--confirm-idle',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--acu',type=Path)
    p.add_argument('--child',type=int);p.add_argument('--profile-arm',type=int);p.add_argument('--config',type=int)
    p.add_argument('--m',type=int);p.add_argument('--result',type=Path)
    a=p.parse_args();a.sdk=a.sdk.resolve();a.output=a.output.resolve()
    if not 0<a.expected_ce_ghz<10:p.error('expected CE GHz must be finite positive')
    if a.child is not None:
        if a.child not in range(len(inventory())):p.error('invalid point')
        if a.profile_arm is not None and (a.profile_arm not in range(len(ARMS)) or a.config not in range(len(configs(8))) or
                                         a.m not in inventory()[a.child]['timing_m'] or a.result is None):p.error('invalid profile coordinates')
        return child(a)
    try:return collect(a)
    finally:
        if a.output.is_dir():
            archive=a.output/f'review-{time.time_ns()}.tgz'
            with tarfile.open(archive,'w:gz') as tar:
                for path in sorted(a.output.rglob('*')):
                    bare=path.name in {f'r{r}-a{i}' for r in range(3) for i in range(len(ARMS))}
                    if path.is_file() and not path.is_symlink() and (path.suffix in ('.json','.log','.csv','.txt','.acurep') or bare):
                        tar.add(path,arcname=str(path.relative_to(a.output)),recursive=False)
            print('results='+str(archive),flush=True)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
