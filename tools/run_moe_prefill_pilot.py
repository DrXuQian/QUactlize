#!/usr/bin/env python3
"""Bounded MoE large-M GPU-routing/Asys pilot, NOT a config sweep or native A/B.

One point's FQ/SF/Full arms stay on one physical card. Compile once, then use
fresh profiled children. All kernels are individually timed; no event fallback.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dev.native_components.build import build_adapter
from dev.strategy import moe_model_plan as models, moe_prefill_plan as plans
from dev.strategy.asys_components import parse_component_rows
from dev.strategy.journal import atomic_json
from dev.strategy.provenance import source_identity, harness_identity
from dev.strategy.smallm_asys import revision, assignments
from dev.prefill_asys.trace import activities
from quactlize.dispatch.table import Workload, Recipe
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools import run_decode_refresh as base
from tools.run_native_components import environment
from tools.run_smallm_regressions_asys import sealed, execute, stop_process
from tools.run_prefill_model_asys import profile_command
from tools.pack_simt_split_review import package


def verify_run(a,run,shard):
    if (run['plan']!=sealed(a.output/'plan.json')['sha256'] or
            run['source_revision']!=revision(ROOT) or run['source']!=source_identity(ROOT) or
            run['harness']!=harness_identity(ROOT) or run['devices'][shard]!=base.device(a)):
        raise ValueError('pilot source/SDK/physical device changed')
    if base.image_identity(base.read(a.output/'images.json'),Path(run['dequant']))!=run['images']:
        raise ValueError('pilot module/dequant binding changed')
    for name,want in run['immutable_files'].items():
        if sha(name)!=want:
            raise ValueError('pilot artifact changed: '+name)


def child(a):
    from dev.strategy.cold_decode import Ring, Expansion, measure_expansion
    from dev.strategy.prefill_fixture import Helpers, routing
    from dev.strategy.prefill_measure import measure_core, measure_full_expansion
    from dev.strategy.moe_prefill_native import RoutedContext, RoutedTC, RoutedFull, AsysStages
    from dev.strategy.resources import NativeProbe, probe_spec
    from dev.strategy.space import resolved_grid
    from dev.strategy.isolated import check_sdk_graph_abi
    from quactlize.runtime.native import SDK
    from tools.run_kpack_strategy_decode import dequant_checked
    from dev.strategy.prefill_build import verify as verify_helpers
    plan,run=(sealed(a.output/name) for name in ('plan.json','run.json'))
    verify_run(a,run,a.worker)
    point=next(p for p in plan['points'] if p['id']==a.point)
    arm=next(x for x in point['arms'] if x['id']==a.arm)
    if point['id'] not in run['assignments'][a.worker]:
        raise ValueError('child point belongs to another card')
    check_sdk_graph_abi(a.sdk)
    w=Workload(**point['workload']);recipe=Recipe.from_dict(arm['recipe'])
    config=max(0,dict(recipe.parameters).get('dequant_config',-1))
    helpers=Path(run['helpers']);dq=Path(run['dequant'])
    verify_helpers(a.sdk,helpers);dequant_checked(dq,a.sdk)
    sdk=SDK(a.sdk);timer=AsysStages()
    authority=dict(run=run['sha256'],shard=a.worker,device=run['devices'][a.worker])
    source=digest([run['source']['sha256'],run['images'],run['adapter_sha256']])
    data={}
    with ExitStack() as owners:
        active=min(len(routing(w,r)['active']) for r in (0,1,2))
        ring=Ring(sdk,w,a.l2_bytes,minimum_active=active);owners.callback(ring.close)
        b=RoutedContext(sdk,ring,w,Helpers(helpers/'libstrategy_prefill.so'),run['adapter'])
        owners.callback(b.close)
        sf=None
        if recipe.route=='sf' and w.qtype!=8:
            mr=Ring(sdk,w,a.l2_bytes,weights=ring.w,metadata_only=True);owners.callback(mr.close)
            mb=RoutedContext(sdk,mr,w,b.helpers,run['adapter']);owners.callback(mb.close)
            msf=Expansion(mb,dq);owners.callback(msf.close)
            data['expansion']=measure_expansion(mb,w,msf,config,authority,source,a.attempt,
                separate_expansion=True,stage_measure=timer.sf)
            sf=Expansion(b,dq,gold=msf.gold);owners.callback(sf.close)
        if recipe.kind=='full':
            core=RoutedFull(b,dq,a.l2_bytes,a.sdk);owners.callback(core.close)
            data['expansion']=measure_full_expansion(b,core,config,authority,source,a.attempt,
                                                    stage_measure=timer.stages)
            query=None
        else:
            record=base.read(a.output/'images.json')['tc'][arm['module']]
            probe=NativeProbe(arm['spec'],record)
            req=probe_spec(w,recipe,int(dict(recipe.parameters).get('grid_mode',0)>=2))
            query=probe.query(req)
            if query['status']!='READY':
                raise ValueError('pilot recipe resource unavailable; preserve query and choose no fallback: '+str(query))
            launch=resolved_grid(w,recipe,query['device']['compute_units'],query['occupancy'])
            bound=dict(arm,binding=dict(launch=launch))
            core=RoutedTC(b,bound,record,query,sf);owners.callback(core.close)
        data['compute']=measure_core(b,core,recipe,authority,source,a.attempt,
            config=config,full=recipe.kind=='full',stage_measure=timer.stages)
        data['compute']['receipt']['row_maps']='GPU_MM_IDS_HELPER_EVERY_CALL_ACTUAL_CONSUMER_POINTERS'
        b.validate_routes()
    if any(r['correctness']!='PASS' for r in data.values()):
        raise ValueError('pilot correctness incomplete')
    base.sealed(a.attempt/'capture.json',dict(status='CAPTURED_NOT_TIMED',point=a.point,arm=a.arm,
        authority=authority,resource_query=query,data=data,component_records=timer.records,
        trace_schedule=timer.collector.schedule,timing_method='ASYS_GPU_ACTIVITY'))
    return 0


def completed(path,run):
    if not path.exists():return None
    value=sealed(path)
    if (value['run']!=run['sha256'] or value['point']!=path.parent.name or value['arm']!=path.stem):
        raise ValueError('pilot resume authority/point/arm differs')
    if value['status']!='PASS':return None
    for name,want in value['evidence'].items():
        if sha(name)!=want:raise ValueError('pilot retained evidence changed: '+name)
    return value


def worker(a):
    plan,run=(sealed(a.output/name) for name in ('plan.json','run.json'))
    verify_run(a,run,a.worker)
    folder=a.output/('gpu-'+run['cards'][a.worker]);folder.mkdir(exist_ok=True)
    pci=run['devices'][a.worker]['device']['pci'];failed=0
    with (Path('/tmp')/('quactlize-components-'+digest(pci)[:24]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for point in plan['points']:
            if point['id'] not in run['assignments'][a.worker]:continue
            for arm in point['arms']:
                path=folder/'points'/point['id']/(arm['id']+'.json')
                if completed(path,run):continue
                attempt=folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
                print('MOE_PREFILL_PILOT_START '+json.dumps(dict(workload=point['workload'],
                      route=arm['recipe']['route'],log=str(attempt/'profile.log'))),flush=True)
                try:
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(a.output),
                        '--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(a.worker),
                        '--point',point['id'],'--arm',arm['id'],'--attempt',str(attempt)]
                    before=base.observe(Path(run['smi']),pci,1700,advisory=True)
                    execute(profile_command(run['asys'],attempt/'trace',cmd),attempt/'profile.log')
                    after=base.observe(Path(run['smi']),pci,1700,advisory=True)
                    report,db=attempt/'trace.asysrep',attempt/'trace.sqlite'
                    execute([run['asys'],'export','-o',str(db),str(report)],attempt/'export.log')
                    captured=sealed(attempt/'capture.json')
                    if (captured['point']!=point['id'] or captured['arm']!=arm['id'] or
                            captured['authority']!=dict(run=run['sha256'],shard=a.worker,device=run['devices'][a.worker])):
                        raise ValueError('pilot child capture differs')
                    provenance=dict(run=run['sha256'],quactlize_commit=run['source_revision']['quactlize_commit'],
                        source_sha256=run['source']['sha256'],harness_sha256=run['harness']['sha256'],
                        actlize_commit=run['source_revision']['actlize_commit'],device=run['devices'][a.worker],
                        images=run['images'],native_backend=run['backend'],
                        trace_sha256=sha(report),sqlite_sha256=sha(db),timing_method='ASYS_GPU_ACTIVITY')
                    rows=parse_component_rows(captured['component_records'],captured['trace_schedule'],
                                             activities(db,launch_shared=True),provenance)
                    for part in captured['data'].values():
                        for stage in part['stages']:
                            record=rows[stage['id']]
                            stage.update(mean_us=record['mean_us'],median_us=record['median_us'],
                                         spread_pct=record['spread_pct'])
                    kernel_sum=math.fsum(rows[cid]['mean_us'] for part in captured['data'].values()
                                         for cid in part['components'])
                    base.sealed(path,dict(status='PASS',run=run['sha256'],point=point['id'],arm=arm['id'],
                        data=captured['data'],component_records=rows,resource_query=captured['resource_query'],
                        measured_kernel_sum_us=kernel_sum,error_reset_api_included=False,
                        clocks=dict(before=before,after=after),
                        evidence={str(p):sha(p) for p in (attempt/'capture.json',report,db)},
                        performance_admitted=False,production_changed=False))
                    print('MOE_PREFILL_PILOT_PASS '+json.dumps(dict(point=point['id'],route=arm['recipe']['route'],
                        components=len(rows),performance_admitted=False)),flush=True)
                except Exception as exc:
                    failed+=1;traceback.print_exc()
                    base.sealed(path,dict(status='FAIL',run=run['sha256'],point=point['id'],arm=arm['id'],
                                         error=str(exc),attempt=str(attempt)))
    return int(bool(failed))


def run(a):
    from dev.strategy.prefill_build import build as build_helpers, verify as verify_helpers
    a.output.mkdir(parents=True,exist_ok=a.resume)
    if (a.output/'plan.json').exists():
        plan=sealed(a.output/'plan.json')
    else:
        audit=plans.make_plan(a.output,models.model_inventory(a,ms=plans.MS))
        base.freeze(a.output/'audit.json',audit)
        plan=plans.pilot_plan(audit);base.freeze(a.output/'plan.json',plan)
    models.verify_inventory(a.output,dict(cohort=plans.COHORT,coverage=sealed(a.output/'audit.json')['coverage']),a.moe_models)
    print('MOE_PREFILL_PILOT_PLAN '+json.dumps(dict(plan['counts'],full_campaign=False,config_sweep=False)),flush=True)
    if a.plan_only:return 0
    if not a.confirm_idle or not a.llama_build:raise ValueError('--confirm-idle and native --llama-build required')
    env=environment(a)
    for name in list(os.environ):
        if name not in env:os.environ.pop(name)
    os.environ.update(env)
    rev=revision(ROOT);a.smi=base.selected_smi(a)
    a.asys=(a.asys or a.sdk/'asight/bin/asys').resolve(strict=True)
    if (a.output/'run.json').exists():
        old=sealed(a.output/'run.json')
        if (old['source_revision']!=rev or old['source']!=source_identity(ROOT) or
                old['harness']!=harness_identity(ROOT) or old['plan']!=plan['sha256'] or
                old['cards']!=a.devices.split(',') or old['l2_bytes']!=a.l2_bytes):
            raise ValueError('pilot resume source/plan/devices changed before any build')
        for name,want in old['immutable_files'].items():
            if sha(name)!=want:raise ValueError('pilot retained artifact changed: '+name)
    adapter,receipt=build_adapter(a.llama_source.resolve(),a.llama_build.resolve(),a.sdk,a.output/'native')
    helper=a.output/'helpers'
    build_helpers(a.sdk,helper);verify_helpers(a.sdk,helper)
    images,dq=base.compile_all(a,plan)
    if dq is None:raise ValueError('pilot requires a verified dequant carrier')
    cards=a.devices.split(',');devices=[]
    for card in cards:
        path=a.output/('device-'+card+'.json')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),
            '--sdk',str(a.sdk),'--probe',str(path)],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card))
        devices.append(base.read(path))
    base.check_devices(devices)
    immutable=[adapter,a.output/'native/adapter.json',helper/'manifest.json',helper/'libstrategy_prefill.so',
               dq/'manifest.json',dq/'libquactlize_ppu_dequant.so',a.asys,a.smi]
    immutable += [Path(r['path']) for r in images['tc'].values()]
    run=dict(plan=plan['sha256'],source_revision=rev,source=source_identity(ROOT),harness=harness_identity(ROOT),
        cards=cards,devices=devices,l2_bytes=a.l2_bytes,assignments=assignments(plan,len(cards),balanced=True),
        adapter=str(adapter),adapter_sha256=sha(adapter),backend=receipt['identity']['libraries'],
        helpers=str(helper),dequant=str(dq),images=base.image_identity(images,dq),
        asys=str(a.asys),smi=str(a.smi),
        immutable_files={str(p):sha(p) for p in immutable} | receipt['identity']['libraries'])
    run['sha256']=digest(run);base.freeze(a.output/'run.json',run)
    a.expected_ce_ghz=1.7;base.lock_clocks(a,devices)
    logs=a.output/'workers'/uuid.uuid4().hex;logs.mkdir(parents=True);workers=[]
    started=time.monotonic()
    try:
        for i,card in enumerate(cards):
            with (logs/('gpu-'+card+'.log')).open('x') as log:
                workers.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--output',str(a.output),'--sdk',str(a.sdk),'--l2-bytes',str(a.l2_bytes),'--worker',str(i)],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in workers):
            print('MOE_PREFILL_PILOT_WAIT '+json.dumps(dict(alive=sum(p.poll() is None for p in workers),
                  minutes=(time.monotonic()-started)/60,logs=str(logs/'gpu-*.log'))),flush=True)
            for p in workers:
                if p.poll() is None:
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:pass
                    break
    finally:
        for p in workers:stop_process(p)
    rows=[]
    for i,ids in enumerate(run['assignments']):
        for point in plan['points']:
            if point['id'] not in ids:continue
            for arm in point['arms']:
                path=a.output/('gpu-'+cards[i])/'points'/point['id']/(arm['id']+'.json')
                value=completed(path,run)
                rows.append(dict(point=point['id'],arm=arm['id'],status='PASS' if value else 'FAIL'))
    result=dict(status='PASS' if all(r['status']=='PASS' for r in rows) and
                all(p.returncode==0 for p in workers) else 'INCOMPLETE',expected=plan['counts']['arms'],
                passed=sum(r['status']=='PASS' for r in rows),cases=rows,returncodes=[p.returncode for p in workers],
                scope=plan['scope'],pending=plan['pending'],performance_admitted=False,production_changed=False)
    atomic_json(a.output/'summary.json',result)
    print('MOE_PREFILL_PILOT_DONE '+json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    return int(result['status']!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sdk',type=Path,default=Path('/usr/local/PPU_SDK'))
    p.add_argument('--llama-source',type=Path);p.add_argument('--llama-build',type=Path)
    p.add_argument('--model-root',type=Path,default=Path('/sim/eec/shared/AI_workspace/llm-models/bench_model_zoo'))
    p.add_argument('--moe-models');p.add_argument('--qwen35-model',type=Path);p.add_argument('--qwen122-model',type=Path)
    p.add_argument('--deepseek-model',type=Path);p.add_argument('--devices',default='3,4,5,6,7');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--l2-bytes',type=int,default=67108864);p.add_argument('--asys',type=Path);p.add_argument('--smi',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true');p.add_argument('--confirm-idle',action='store_true')
    p.add_argument('--probe',type=Path);p.add_argument('--worker',type=int);p.add_argument('--point');p.add_argument('--arm');p.add_argument('--attempt',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.sdk=a.sdk.resolve()
    cards=a.devices.split(',')
    if (not 1<=len(cards)<=8 or len(set(cards))!=len(cards) or any(not x.isdecimal() or str(int(x))!=x for x in cards)
            or not 1<=a.jobs<=64 or a.l2_bytes<=0):p.error('invalid GPU list/jobs/L2')
    if bool(a.point)!=bool(a.arm) or bool(a.point)!=bool(a.attempt):p.error('incomplete pilot child mode')
    if a.point and (a.worker is None or not a.attempt.resolve().is_relative_to(a.output)):
        p.error('child needs assigned worker and in-run attempt')
    def stop(*_):raise KeyboardInterrupt('termination requested')
    signal.signal(signal.SIGTERM,stop)
    if a.probe:atomic_json(a.probe,base.device(a));return 0
    if a.point:return child(a)
    if a.worker is not None:return worker(a)
    if not a.resume and a.moe_models is None:a.moe_models='qwen35-35b-a3b,qwen35-122b-a10b'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:return run(a)
        finally:
            if a.output.is_dir():package(a.output)


if __name__=='__main__':raise SystemExit(main())
