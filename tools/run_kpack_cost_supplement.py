#!/usr/bin/env python3
"""Finite, resumable component measurements; successful peers survive failures."""
import argparse
import ctypes as C
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.runtime.compiler import sha
from quactlize.runtime.tuning import digest
from tools.run_kpack_dequant_gate import save, packages
from tools.kpack_cost_supplement import PHASES, dequant_key, validate_point
from tools.kpack_cost_evidence import EVIDENCE, reuse
from tools.kpack_cost_measurements import Context, dequant_measure, gemm_measure, bf16_measure
from tools.kpack_prefill_measurement import Weights, BUNDLE, bind_reducer


def verify(bundle,sdk):
    m=json.loads((bundle/'manifest.json').read_text())
    if m['schema']!='quactlize.cost-supplement-bundle.v1':raise ValueError('wrong cost bundle')
    for name,h in m['files'].items():
        p=(bundle/name).resolve(strict=True)
        if not p.is_relative_to(bundle) or sha(p)!=h:raise ValueError('cost payload differs: '+name)
    for name,h in m['runtime'].items():
        if sha(sdk/'lib'/name)!=h:raise ValueError('cost runtime differs: '+name)
    for name,h in m['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('cost source differs: '+name)
    plan=json.loads((bundle/'plan.json').read_text())
    if digest({k:v for k,v in plan.items() if k!='plan_sha256'})!=m['plan_sha256'] or plan['plan_sha256']!=m['plan_sha256']:
        raise ValueError('cost plan digest differs')
    for p in plan['points']:validate_point(p)
    return m,plan


def timing_valid(r):
    values=r['samples_us']
    if len(values)!=15 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('need 3x5 finite positive component timings')
    if (r['median_us']!=statistics.median(values) or
            r['round_medians_us']!=[statistics.median(values[i:i+5]) for i in (0,5,10)]):
        raise ValueError('component timing aggregation differs')


def stable(r):
    v=r['round_medians_us']
    return max(v)/min(v)<=1.05


def validate(r,key,authority):
    if (r['status']!='PASS' or r['component_id']!=key or
            r['authority_sha256']!=authority or r.get('production_changed',False)):
        raise ValueError('component result context differs')
    expected=r.get('record_sha256')
    if digest({k:v for k,v in r.items() if k!='record_sha256'})!=expected:
        raise ValueError('component result checksum differs')
    if r['kind']=='dequant':
        if r['scope']!='ISOLATED_DEQUANT_ACTUAL_EXPERT_DOMAIN' or r['gemm_timed']:
            raise ValueError('dequant measurement scope differs')
        for row in r['rows']:
            timing_valid(row)
            if row['proof']['bad'] or row['proof']['signed_zero_differences']:
                raise ValueError('nonexact dequant result')
            if r.get('measurement')=='REUSED_AUDITED_MEASUREMENT':
                if row['proof']['negative_bad']<=0 or row['proof']['guard']!='PASS':
                    raise ValueError('prior dequant controls absent')
            else:
                if row['negative_bad']<=0 or row['guard']!='PASS':raise ValueError('dequant controls absent')
                if r.get('indexed') and (row['device_count_control']!='PASS' or row['changed_id_graph']!='PASS'):
                    raise ValueError('indexed GPU input controls absent')
        if not r['rows']:raise ValueError('no dequant timing')
    elif r['kind'] in ('gemm','bf16'):
        timing_valid(r)
        if not math.isfinite(r['error']) or not 0<=r['error']<.005:raise ValueError('invalid dot oracle')
        if any(r[k]!='PASS' for k in ('guards','zero_a','changed_input_graph')):
            raise ValueError('missing component correctness controls')
        if r['kind']=='gemm' and (not r['reducer_included'] or r['sf_prepass_timed']):
            raise ValueError('GEMM component scope differs')
        if r['kind']=='bf16' and r['dequant_timed']:raise ValueError('BF16 includes dequant')
        expected_scope=('SELECTED_OR_HISTORICAL_COMPLETE_GEMM_NO_DEQUANT_NO_EXTERNAL_ADAPTERS'
            if r['kind']=='gemm' else 'ISOLATED_BF16_PROVIDER_NO_DEQUANT_NO_EXTERNAL_ADAPTERS')
        if r['scope']!=expected_scope:raise ValueError('component scope differs')
    elif r['kind']=='reducer':
        from tools.run_kpack_decode_reducer import validate as validate_reducer
        validate_reducer(r,r['workload'])
    else:raise ValueError('unknown component kind')
    return r


def perform(a,key,fn):
    path=a.output/'components'/f'{key}.json'
    if path.exists():
        r=validate(json.loads(path.read_text()),key,a.authority)
        print(f'COST_COMPONENT id={key} status=REUSED',flush=True)
        return r
    start=time.monotonic()
    try:
        r=fn()
        if r['status']!='PASS':raise ValueError('component unavailable: '+str(r))
        r.update(component_id=key,authority_sha256=a.authority,elapsed_seconds=time.monotonic()-start)
        r['record_sha256']=digest(r)
        validate(r,key,a.authority)
        save(path,r)
        print(f'COST_COMPONENT id={key} status=PASS seconds={r["elapsed_seconds"]:.1f}',flush=True)
        return r
    except Exception as e:
        traceback.print_exc()
        # Failure receipts never overwrite a successful result.
        save(a.output/'failures'/f'{key}.{time.time_ns()}.json',dict(component_id=key,error=str(e),
            authority_sha256=a.authority,elapsed_seconds=time.monotonic()-start))
        print(f'COST_COMPONENT id={key} status=FAIL remaining_continue=1',flush=True)
        return None


def keys(p):
    out=[dequant_key(p,o) for o in (0,1)]
    for route, info in p['routes'].items():
        out.append(f'{p["id"]}-r{route}-current')
        if info['historical'] is not None:out.append(f'{p["id"]}-r{route}-historical')
    return out+[p['id']+'-bf16']


def weight_run(a,points):
    if all((a.output/'components'/f'{key}.json').exists() for p in points for key in keys(p)):
        for p in points:
            for key in keys(p):validate(json.loads((a.output/'components'/f'{key}.json').read_text()),key,a.authority)
        return
    p=points[0]
    w=Weights(p,lambda done,total:print(f'COST_FIXTURE weight={p["weight_id"]} experts={done}/{total}',flush=True),keep_bf16=True)
    evidence=json.loads(EVIDENCE.read_text())
    for p in points:
        dequant={}
        for operation in (0,1):
            key=dequant_key(p,operation)
            def measure(operation=operation):
                old=None if getattr(a,'no_reuse',False) else reuse(
                    evidence,p,operation,w.identity,a.device,a.manifest['runtime'],a.packages)
                return old if old is not None else dequant_measure(a,p,w,operation)
            dequant[operation]=perform(a,key,measure)
        for route,info in p['routes'].items():
            perform(a,f'{p["id"]}-r{route}-current',lambda:gemm_measure(a,p,w,int(route)))
            if info['historical'] is not None:
                perform(a,f'{p["id"]}-r{route}-historical',lambda:gemm_measure(a,p,w,int(route),True))
        if dequant[1]:
            best=min(dequant[1]['rows'],key=lambda r:r['median_us'])
            perform(a,p['id']+'-bf16',lambda:bf16_measure(a,p,w,best))
        else:print(f'COST_COMPONENT id={p["id"]}-bf16 status=BLOCKED_FULL_DEQUANT remaining_continue=1',flush=True)


def summarize(output,points,authority,include_reducer=True):
    expected={key for p in points for key in keys(p)}
    if include_reducer:expected.add('dense-m1-n4096-s4-recheck')
    records={};missing=[];noisy=[]
    for key in sorted(expected):
        path=output/'components'/f'{key}.json'
        if not path.exists():missing.append(key);continue
        r=validate(json.loads(path.read_text()),key,authority);records[key]=r
        rows=r['rows'] if r['kind']=='dequant' else [r]
        if any(not stable(row) for row in rows):noisy.append(key)
    sums=[]
    for p in points:
        needed=keys(p)
        if any(k not in records for k in needed):continue
        dq={o:min(records[dequant_key(p,o)]['rows'],key=lambda r:r['median_us'])['median_us'] for o in (0,1)}
        costs={}
        for route,info in p['routes'].items():
            for mode in ('current','historical') if info['historical'] is not None else ('current',):
                key=f'{p["id"]}-r{route}-{mode}'
                costs[f'r{route}-{mode}']=records[key]['median_us']+(dq[0] if int(route)%2 else 0)
        costs['full-bf16']=dq[1]+records[p['id']+'-bf16']['median_us']
        sums.append(dict(point=p['id'],cost_us=costs,lowest_observed=min(costs,key=costs.get),
            timing_admission='NEEDS_CONFIRMATION' if any(k in noisy for k in needed) else 'WITHIN_5PCT_ROUND_RANGE',
            scope='SUM_OF_ISOLATED_COMPONENTS_NOT_MEASURED_E2E'))
    files={str(p.relative_to(output)):sha(p) for p in output.rglob('*.json') if p.name!='result.json'}
    r=dict(status='COMPLETE' if not missing else 'INCOMPLETE',expected_components=len(expected),
        complete=len(records),missing=missing,timing_unstable=noisy,costs=sums,files=files,
        authority_sha256=authority,production_changed=False)
    save(output/'result.json',r)
    print(f'COST_DONE status={r["status"]} components={len(records)}/{len(expected)} noisy={len(noisy)} results={output}',flush=True)
    return r


def reducer_recheck(a):
    from tools.run_kpack_decode_reducer import cases, measure
    from tools.kpack_prefill_measurement import verify as verify_reducer
    manifest,_=verify_reducer(BUNDLE,a.sdk)
    header=ROOT/'quactlize/include/actlize_extensions/cutlass/gemm/device/ppu_mixed_input_splitk_parallel.hpp'
    if manifest['reducer_header_sha256']!=sha(header):raise ValueError('reducer implementation changed')
    if manifest['runtime']!=a.manifest['runtime']:raise ValueError('reducer runtime differs')
    expected=json.loads((BUNDLE/'fixture-receipts.json').read_text())['authority']['device']
    if expected!=a.device and not (getattr(a,'allow_equivalent_device',False) and same_device_class(expected,a.device)):
        raise ValueError('reducer device class differs')
    p=next(w for w in cases() if w['id']=='dense-m1-n4096-s4')
    ctx=Context(a.sdk,a.bundle,a.device)
    try:
        lib=bind_reducer(BUNDLE/'libprefill_reducer.so')
        return measure(ctx.sdk,lib,a.device,p)|dict(kind='reducer')
    finally:ctx.close()


def same_device_class(expected,observed):
    """The caller admits same-model cards, not fabricated PCI equivalence.

    Every cost GEMM module independently requires PPU-ZW810 / 72 CUs. Here
    additionally retain the measured cache size and warp-width contract.
    """
    fields=('l2_bytes','compute_units','warp')
    return all(k in expected and expected[k]==observed.get(k) for k in fields)


def measurement_sources():
    sources=[p for p in (ROOT/'tools').glob('*kpack_cost*.py')]+[
        ROOT/'tools/kpack_prefill_measurement.py',ROOT/'tools/kpack_bf16_fixture.py',
        ROOT/'tools/kpack_bf16_providers.py',ROOT/'tools/kpack_dequant_fixture.py',
        ROOT/'tools/run_kpack_decode_reducer.py',ROOT/'tools/run_kpack_dequant_gate.py',
        ROOT/'tools/run_kpack_gemv_gate.py',ROOT/'tools/run_kpack_grouped_decode_probe.py',
        ROOT/'tools/run_kpack_grouped_device_gate.py',ROOT/'quactlize/runtime/native.py',
        ROOT/'quactlize/dispatch/native.py',ROOT/'quactlize/dequant/native.py',
        ROOT/'tools/kpack_warmup_fixture.py',ROOT/'reference/gguf_kpack.py',EVIDENCE]
    return {str(p.relative_to(ROOT)):sha(p) for p in sources}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['all',*PHASES],default='dense-mid')
    parser.add_argument('--child-weight')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--probe-device',action='store_true')
    parser.add_argument('--allow-equivalent-device',action='store_true')
    parser.add_argument('--no-reuse',action='store_true',help='fresh measurements, no historical timing reuse')
    parser.add_argument('--reducer-only',action='store_true')
    a=parser.parse_args()
    a.bundle=a.bundle.resolve(strict=True);a.sdk=a.sdk.resolve(strict=True);a.output=a.output.resolve()
    a.manifest,plan=verify(a.bundle,a.sdk)
    a.packages=packages()
    ctx=Context(a.sdk,a.bundle)
    a.device=ctx.device;ctx.close()
    expected=json.loads(EVIDENCE.read_text())['entries'][0]['device']
    if a.device!=expected and not (a.allow_equivalent_device and same_device_class(expected,a.device)):
        raise ValueError('use the prior PPU or explicitly admit an equivalent device class')
    if a.probe_device:
        print('COST_DEVICE '+json.dumps(dict(device=a.device,packages=a.packages,runtime=a.manifest['runtime'])),flush=True)
        return
    authority=dict(schema='quactlize.cost-supplement-results.v1',plan_sha256=plan['plan_sha256'],
        bundle_sha256=sha(a.bundle/'manifest.json'),device=a.device,runtime=a.manifest['runtime'],
        sources=measurement_sources(),python_packages=a.packages,
        production_changed=False,small_m_full_dequant=False)
    if a.allow_equivalent_device:
        authority.update(device_admission='SAME_MODEL_CROSS_CARD_COMPARISON_ALLOWED',
                         historical_timing_reuse=not a.no_reuse)
    a.authority=digest(authority)
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('components','failures','logs'):(a.output/name).mkdir(exist_ok=True)
    path=a.output/'authority.json'
    if path.exists() and json.loads(path.read_text())!=authority:raise ValueError('resume authority differs')
    save(path,authority)
    save(a.output/'plan.json',plan)
    save(a.output/'bundle-receipt.json',a.manifest)
    points=[p for p in plan['points'] if a.phase=='all' or p['phase']==a.phase]
    if a.reducer_only:
        perform(a,'dense-m1-n4096-s4-recheck',lambda:reducer_recheck(a))
        r=summarize(a.output,[],a.authority)
        if r['status']!='COMPLETE':raise SystemExit(1)
        return
    if a.child_weight:
        points=[p for p in points if p['weight_id']==a.child_weight]
        if not points:raise ValueError('child weight outside selected phase')
        weight_run(a,points)
        if a.allow_equivalent_device:summarize(a.output,points,a.authority,include_reducer=False)
        if any(not (a.output/'components'/f'{key}.json').exists() for p in points for key in keys(p)):
            raise SystemExit(1)
        return
    if not a.summarize_only:
        perform(a,'dense-m1-n4096-s4-recheck',lambda:reducer_recheck(a))
        weights=sorted({p['weight_id'] for p in points})
        elapsed=[]
        for index,weight in enumerate(weights):
            before=time.monotonic()
            path=a.output/'logs'/f'{weight}.{time.time_ns()}.log'
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--bundle',str(a.bundle),
                 '--output',str(a.output),'--phase',a.phase,'--child-weight',weight]
            if a.allow_equivalent_device:cmd.append('--allow-equivalent-device')
            if a.no_reuse:cmd.append('--no-reuse')
            with path.open('w') as log:
                child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in child.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('COST_','ValueError:','RuntimeError:')):print(line,end='',flush=True)
                rc=child.wait()
            elapsed.append(time.monotonic()-before)
            remaining=statistics.mean(elapsed[-4:])*(len(weights)-index-1)/60
            print(f'COST_PROGRESS phase={a.phase} weights={index+1}/{len(weights)} last_rc={rc} '
                  f'remaining_minutes={remaining:.1f} eta=RECENT_WEIGHT_AVERAGE_NOT_GUARANTEE',flush=True)
    r=summarize(a.output,points,a.authority)
    if r['status']!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise SystemExit(1)
