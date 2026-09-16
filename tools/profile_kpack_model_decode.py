#!/usr/bin/env python3
"""ACU after model Asys: reproduce observed decode recipes in the same DSO.

Synthetic inputs match shape, compute type and C ABI, not model activations.
ACU forced-cache replay counters are not model or rotating-weight latency.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from tools.verify_kpack_dispatch import verify
from tools.profile_kpack_gpu_compact import acu_launch_command, AcuRange


def save(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def profile_plan(selection, kernels, log, helpers, matched=()):
    simt_recipe, q4_recipe, q4_matches = helpers
    names = [k['name'] for k in kernels['kernels']]
    unique = {}
    for p in selection['plans']:
        if p['op']=='dense' and p['route'] in ('fq','sf') and int(p['rows'])==1:
            if p.get('activation') not in ('FP16','BF16'):
                raise ValueError('TC profile needs explicit compute type')
            build=p.get('build','')
            if not re.fullmatch(r'[0-9a-f]{64}',build):
                raise ValueError('TC profile needs exact observed build identity')
            observed=[r['name'] for r in matched
                      if any(label.endswith('/'+build) for label in r.get('libraries',()))
                      and r['name'] in names and 'cutlass::device_kernel<' in r['name']
                      and ('SplitKParallel' in r['name'])==(int(p['split'])>1)]
            if len(set(observed))!=1:
                raise ValueError('TC parent is not uniquely bound to an Asys-observed producer: '+p['tensor'])
            point=dict(kind='tc',q=int(p['q']),n=int(p['n']),k=int(p['k']),tokens=1,
                       mode=0,experts=1,channels=1,topk=1,route=int(p['route']=='sf'),
                       compute=int(p['activation']=='BF16'),endpoint=1,
                       parent=p['parent'],build=build,split=int(p['split']),
                       algorithm=int(p['algorithm']),grid=int(p['grid']),policy=int(p['policy']))
            key=json.dumps(point,sort_keys=True)
            if key not in unique:
                unique[key]=dict(point=point,tensors=[],observed_symbols=sorted(set(observed)))
            unique[key]['tensors'].append(p['tensor'])
            continue
        if p['route'] not in ('gemv','gemv-q4-s1'):
            continue
        if int(p['rows']) != (1 if p['op']=='dense' else 8):
            continue
        if p.get('activation') not in ('FP16','BF16'):
            raise ValueError('missing explicit compute type')
        if not all(k in p for k in ('experts','channels','topk')):
            raise ValueError('profile needs updated caller expert/channel receipts')
        point = {k:int(p[k]) for k in ('q','n','k','experts','channels','topk','variant','columns','warps','values','split')}
        point.update(mode=0 if p['op']=='dense' else 2,tokens=1,
                     compute=int(p['activation']=='BF16'),kind='simt',reader=p.get('reader'))
        if p['route']=='gemv-q4-s1':
            point['kind'],point['reader']='q4',int(p['reader'])
            observed = [n for n in names if q4_matches(q4_recipe(n),p)]
        else:
            wanted = (point['q'],1,point['variant'],point['columns'],point['warps'],point['values'],point['compute'])
            observed = [n for n in names if simt_recipe(n)==wanted]
        if not observed:
            raise ValueError('selected SIMT recipe absent from completed Asys: '+p['tensor'])
        key = json.dumps(point,sort_keys=True)
        if key not in unique:
            unique[key] = dict(point=point,tensors=[],observed_symbols=sorted(set(observed)))
        unique[key]['tensors'].append(p['tensor'])
    rows = list(unique.values())
    if not rows:
        raise ValueError('completed trace contains no selected M1 compute call')
    # Match actual merged chain records, not an assumed Q4 model topology.
    prepare_names = [n for n in names if 'quactlize::runtime::prepare_detail::once<' in n and
                     re.search(r',\s*8,\s*true,',n)]
    for line in log.splitlines():
        if '[quactlize-moe]' not in line:
            continue
        c = dict(re.findall(r'([a-z_]+)=([^\s]+)',line))
        if (c.get('merged'),c.get('rows'),c.get('simt_mask')) != ('1','8','5'):
            continue
        gate = next((r for r in rows if c['gate'] in r['tensors']),None)
        down = next((r for r in rows if c['down'] in r['tensors']),None)
        if gate is None or down is None or not prepare_names:
            raise ValueError('all-SIMT chain lacks observed producer or new prepare kernel')
        g,d = gate['point'],down['point']
        if (g['n'],d['n'],d['k'])!=(1024,2048,512) or g['k'] not in (512,2048):
            continue
        point = dict(kind='prepare',tokens=1,k=g['k'],simt_mask=5,compute=g['compute'])
        if not any(r['point']==point for r in rows):
            rows.append(dict(point=point,tensors=[c['gate'],c['down']],observed_symbols=prepare_names))
    # Cover every distinct observed M1 recipe, including TC and its reducer.
    # A fixed top-three filter silently omitted the costly Q8/Q6 TC paths.
    dense = sorted((r for r in rows if r['point'].get('mode')==0),
                   key=lambda r:r['point']['n']*r['point']['k'],reverse=True)
    grouped = [r for r in rows if r['point'].get('mode')==2]
    prepares = [r for r in rows if r['point']['kind']=='prepare']
    return dense+grouped+prepares


def child(a):
    from dev.gemv_simt.native import Runtime, checked
    from dev.gemv_simt.production import Library, Q4Library
    from dev.gemv_simt.q8_vector_run import Bench
    from dev.gemv_simt.fixture import weights
    from dev.moe_prepare.production import prepare
    from tools.run_kpack_pack_gate import device_identity
    m = verify(a.bundle,sdk=a.sdk)
    job = json.loads(a.job.read_text());p=job['point']
    rt = Runtime(a.sdk,'ppu')
    bench=None
    try:
        identity = device_identity(rt)
        library = a.bundle/'libquactlize_ppu_execution.so'
        if p['kind']=='tc':
            from tools.kpack_model_profile_bench import DenseBench
            bench=DenseBench(rt,a.bundle,a.sdk,a.jit_cache,p)
            launch,check=bench.launch,bench.check
        elif p['kind']=='prepare':
            launch, check = prepare(rt,C.CDLL(str(library.resolve()),mode=C.RTLD_LOCAL),p)
        else:
            lib = Q4Library(library,p['compute']) if p['kind']=='q4' else Library(library,p['compute'])
            w = weights(p['q'],p['n'],p['k'],p['experts'])
            b = Bench(rt,lib,w,1,p['mode'],p['channels'],p['compute'])
            config = SimpleNamespace(**p)
            launch = lib.prepare(b.call,config)
            b.poison()
            def check():
                _,error=b.output_check()
                return dict(status='PASS',oracle='INDEPENDENT_GGUF_FACTOR_DOT',error=error)
        checked(launch(),'unprofiled correctness call');rt.sync();proof=check()
        for _ in range(5):
            checked(launch(),'excluded warmup')
        rt.sync()
        with AcuRange(rt):
            checked(launch(),'profiled complete call')
            rt.sync()
        check()
        save(a.output,dict(status='PASS',job=job,device=identity,proof=proof,
                          execution_sha256=m['execution_sha256'],
                          manifest_sha256=sha(a.bundle/'manifest.json'),
                          scope='SYNTHETIC_INPUT_SAME_SHAPE_RECIPE_COMPUTE_PRODUCTION_C_ABI',
                          components=['producer','reducer'] if p.get('split',1)>1 else ['producer'],
                          timing_scope='ACU_FORCED_CACHE_NOT_MODEL_LATENCY'))
    finally:
        if bench is not None:
            bench.close()
        rt.close()


def collect(a):
    sys.path.insert(0,str(a.llama/'tests'))
    from quactlize_native import simt_symbol_recipe, q4_symbol_recipe, q4_symbol_matches_plan
    m = verify(a.bundle,sdk=a.sdk)
    a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for directory in sorted(a.trace.iterdir()):
        if not directory.is_dir():
            continue
        native=directory/'native'
        try:
            proof=json.loads((native/'proof.json').read_text())
            if (proof.get('missing_ops') or proof.get('kernel_execution')!='PASS_SHORT_REQUEST' or
                    proof.get('capture_scope')!='SECOND_REQUEST_SAME_PROCESS_FIRST_USE_EXCLUDED'):
                raise ValueError('Asys request incomplete/unselected/not warmed')
            selected=native/'proof-request/0-native.selection.json'
            times=native/'kernel-times.json'
            plan=profile_plan(json.loads(selected.read_text()),json.loads(times.read_text()),
                              (native/'proof-request/0-native.log').read_text(),
                              (simt_symbol_recipe,q4_symbol_recipe,q4_symbol_matches_plan),
                              proof.get('matched',()))
            if a.kind!='all':
                plan=[r for r in plan if r['point']['kind']==a.kind]
            if not plan:
                raise ValueError('no observed jobs in requested profile scope')
        except (OSError,ValueError) as e:
            records.append(dict(model=directory.name,status='FAIL',phase='PLAN',error=str(e)))
            continue
        root=a.output/directory.name;root.mkdir()
        save(root/'plan.json',dict(jobs=plan,execution_sha256=m['execution_sha256'],
                                 selection_sha256=sha(selected),kernel_times_sha256=sha(times)))
        print(f'KPACK_MODEL_ACU_PLAN model={directory.name} reports={len(plan)} '
              'production_images=UNCHANGED tc_uses_existing_jit_cache=1',flush=True)
        for i,job in enumerate(plan):
            stem=f'{i:02d}-{job["point"]["kind"]}'
            request=root/(stem+'.request.json');save(request,job)
            receipt=root/(stem+'.json');report=root/stem;log=root/(stem+'.log')
            cmd=acu_launch_command(a.acu,report,[sys.executable,'-u',str(Path(__file__).resolve()),
                '--sdk',str(a.sdk),'--bundle',str(a.bundle),'--output',str(receipt),'--job',str(request),
                '--jit-cache',str(a.jit_cache)])
            save(root/(stem+'.command.json'),cmd)
            print(f'KPACK_MODEL_ACU_START model={directory.name} report={stem}',flush=True)
            started=time.monotonic()
            with log.open('x') as out:
                proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f'KPACK_MODEL_ACU_WAIT report={stem} seconds={time.monotonic()-started:.0f}',flush=True)
            row=dict(model=directory.name,job=job,rc=proc.returncode,status='FAIL',log=str(log))
            try:
                paths=[p for p in (report,Path(str(report)+'.acurep')) if p.is_file() and p.stat().st_size]
                r=json.loads(receipt.read_text())
                if proc.returncode or len(paths)!=1 or r.get('status')!='PASS' or r.get('job')!=job or r.get('execution_sha256')!=m['execution_sha256']:
                    raise ValueError('ACU child/report/identity failed')
                raw=root/(stem+'.csv')
                with raw.open('x') as out:
                    subprocess.run([a.acu,'--import',paths[0],'--page','raw','--csv'],stdout=out,stderr=subprocess.STDOUT,check=True)
                text=raw.read_text(errors='replace')
                normalized=re.sub(r'\s+','',text)
                if not all(re.sub(r'\s+','',name) in normalized for name in job['observed_symbols']):
                    raise ValueError('ACU export lacks the exact Asys-observed producer symbol')
                if job['point'].get('split',1)>1:
                    reducer='reduce_decode<' if job['point']['kind']=='tc' else 'register_reuse_reduce<'
                    if reducer not in normalized:
                        raise ValueError('ACU export lacks the complete-call Split-K reducer')
                row.update(status='CAPTURED',report=str(paths[0]),report_sha256=sha(paths[0]),raw_csv=str(raw))
            except (OSError,ValueError,subprocess.SubprocessError) as e:
                row['error']=str(e)
            records.append(row)
            save(a.output/'summary.json',dict(records=records,timing_scope='PROFILER_DIAGNOSTICS_NOT_TPOT'))
            print(f'KPACK_MODEL_ACU_DONE report={stem} status={row["status"]} remaining_continue=1',flush=True)
    if not records:
        raise ValueError('no model trace directory')
    status='PASS' if all(r['status']=='CAPTURED' for r in records) else 'FAIL'
    save(a.output/'summary.json',dict(status=status,records=records,execution_sha256=m['execution_sha256'],
                                   scope='SYNTHETIC_INPUTS_OBSERVED_RECIPES_NOT_MODEL_ROUTING_REPLAY',
                                   timing_scope='PROFILER_DIAGNOSTICS_NOT_TPOT'))
    return int(status!='PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','bundle','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('acu','llama','trace','job'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--jit-cache',type=Path,default=Path('/workspace/kpack-model-jit-cache'))
    p.add_argument('--kind',choices=('all','tc','simt','q4','prepare'),default='all')
    a=p.parse_args()
    if a.job:
        child(a);return 0
    if not all((a.acu,a.llama,a.trace)):
        p.error('collection requires --acu --llama --trace')
    return collect(a)


if __name__=='__main__':
    raise SystemExit(main())
