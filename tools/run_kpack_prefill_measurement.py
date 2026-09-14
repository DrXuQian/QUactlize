#!/usr/bin/env python3
"""Two isolated measurements: heuristic-selected FQ/SF calls, or production reducers."""
import argparse
import ctypes as C
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.kpack_prefill_measurement import (BUNDLE, BOARD, SCOPE, Weights, families,
    reducer_cases, verify, validate_fixture, validate_gemm, validate_reducer,
    partial_values, reducer_expected, bind_reducer, row_domain)
from tools.run_kpack_dequant_gate import save, device, packages
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from quactlize.dequant.native import bind as bind_probe
from quactlize.dispatch.native import Dispatch, receipt
from quactlize.execution.native import arrangement
from quactlize.runtime.native import SDK, Call, checked
from quactlize.runtime.compiler import sha


def timing(r,replay,copies,proof,case):
    checked(replay(),'graph upload/first replay');r.sdk.synchronize(r.stream)
    proof()
    samples=[]
    for round_ in range(3):
        samples.extend(x/(copies*2) for x in r.samples(replay,5))
        proof()
        print(f'PREFILL_MEASURE_PROGRESS case={case} round={round_+1}/3',flush=True)
    return dict(samples_us=samples,median_us=statistics.median(samples),
        round_medians_us=[statistics.median(samples[i:i+5]) for i in (0,5,10)],
        timing='CAPTURED_COMPLETE_RING_EVENTS_FIRST_USE_EXCLUDED',calls_per_graph=copies*2)


def copy_host(sdk,pointer,value):
    value=np.ascontiguousarray(value)
    checked(sdk.lib.hggcMemcpy(pointer,value.ctypes.data,value.nbytes,1),'fixture H2D')
    sdk.synchronize(None)


def gemm_case(a,w,weights,tokens,route,evidence,plan):
    sdk=SDK(a.sdk);graph_bind(sdk);r=Resources(sdk);d=Dispatch(a.bundle);replay=None
    try:
        probe_lib,_,probe=bind_probe(a.bundle);dev=device(sdk,probe)
        if dev!=evidence['authority']['device']:raise ValueError('use the same PPU as existing component measurements')
        board=validate_fixture(w,tokens,weights.identity,evidence)
        rows,indices,route_hash=row_domain(tokens,w['experts']);m=len(indices)
        arr=arrangement(w['q']);sf=route in (1,3)
        selected=d.query(w['q'],route,m,w['n'],w['k'],w['experts'],tokens,arr.mapping_id)
        if selected is None:raise ValueError('current production heuristic missed')
        expected=next(x for x in plan['requests'] if x['request']==
            [w['q'],route,m,w['n'],w['k'],w['experts'],tokens])
        if selected.parent.decode()!=expected['parent'] or selected.split!=expected['split'] or selected.policy!=expected['policy']:
            raise ValueError('live heuristic differs from compiled selected plan')
        choice=receipt(selected)
        print('PREFILL_MEASURE_SELECTED '+json.dumps(dict(case=w['id'],tokens=tokens,route=route,selection=choice)),flush=True)
        names=('low','high','scale','zero') if sf else ('low','high','units')
        input_bytes=sum(weights.planes[name].nbytes for name in names)
        copies=max(2,math.ceil(2.25*dev['l2_bytes']/input_bytes))
        activation,coeff=weights.activation(m);ap=r.upload(activation)
        bounds=r.upload(np.r_[0,np.cumsum(rows)].astype('<i4')) if w['experts']>1 else None
        size=m*w['n']*2;outputs=[];calls=[]
        started=time.monotonic()
        for i in range(copies):
            planes={name:r.upload(weights.planes[name]) if weights.planes[name].size else None for name in names}
            out=r.alloc(size+256);outputs.append(out);r.fill(out,0xA5,size+256)
            c=Call(version=1,size=C.sizeof(Call),m=m,n=w['n'],k=w['k'],experts=w['experts'],
                group_size=arr.group_size,device=selected.device,compute_units=selected.compute_units,
                mapping_id=arr.mapping_id,a=ap,low=planes['low'],high=planes['high'],
                metadata=planes['scale'] if sf else planes['units'],zero=planes.get('zero'),output=out+128,
                offsets_device=bounds,workspace=r.alloc(max(1,selected.workspace_bytes)),
                workspace_bytes=selected.workspace_bytes,stream=r.stream.value)
            sdk.synchronize(None)
            calls.append(d.prepare(selected,c))
        def sequence():
            for fn in calls:checked(fn(),'selected full output')
            return 0
        errors=[]
        def proof(i=0,zero=False):
            sdk.synchronize(r.stream)
            raw=sdk.download(outputs[i],size+256)
            if raw[:128]!=b'\xa5'*128 or raw[-128:]!=b'\xa5'*128:raise ValueError('GEMM output guard changed')
            out=np.frombuffer(raw[128:-128],dtype='<f2').reshape(m,w['n'])
            if zero:
                if not np.isfinite(out).all() or np.any(out):raise ValueError('zero A did not overwrite output with zero')
            else:
                error=weights.error(out,coeff,indices)
                if not error<.005:raise ValueError(f'official GGUF dot mismatch {error:.6g}')
                errors.append(error)
        sequence();sdk.synchronize(r.stream)
        for i in range(copies):proof(i)
        if weights.error(np.zeros((m,w['n']),dtype='<f2'),coeff,indices)<=.005:
            raise ValueError('zero-output negative escaped oracle')
        copy_host(sdk,ap,np.zeros_like(activation))
        for out in outputs:r.fill(out+128,0x7e,size)
        sequence()
        for i in range(copies):proof(i,True)
        copy_host(sdk,ap,-activation);coeff*=-1;sequence();proof(0);proof(copies-1)
        copy_host(sdk,ap,activation);coeff*=-1
        replay=Replay(sdk,r.stream,sequence,2)
        checked(replay(),'initial graph replay');sdk.synchronize(r.stream)
        for i in range(copies):proof(i)
        copy_host(sdk,ap,-activation);coeff*=-1
        for out in outputs:r.fill(out+128,0x7e,size)
        checked(replay(),'changed-input graph replay');proof(0);proof(copies-1)
        copy_host(sdk,ap,activation);coeff*=-1
        setup_seconds=time.monotonic()-started
        result=timing(r,replay,copies,lambda:(proof(0),proof(copies-1)),f'{w["id"]}-t{tokens}-r{route}')
        result.update(status='PASS',workload=w,tokens=tokens,route=route,total_rows=m,rows=rows.tolist(),
            max_rows=int(rows.max()),max_rows_bound=tokens,active_experts=int(np.count_nonzero(rows)),
            routes_sha256=route_hash,device=dev,selection=choice,fixture=weights.identity,
            copies=copies,input_bytes=input_bytes,input_ring_bytes=copies*input_bytes,
            output_ring_bytes=copies*size,error=max(errors),guards='PASS',zero_a='PASS',
            changed_a_graph='PASS',changed_a_sequence='PASS',setup_seconds=setup_seconds,
            sf_prepass_timed=False,reducer_included=True,scope=SCOPE,production_changed=False,
            sf_dequant_us=board['sf_dequant']['us'],full_bf16_sum_us=board['cost']['sum_estimate_us'],
            precision='FP16_A_FP32_ACCUM_FP16_OUTPUT_OFFICIAL_GGUF_ORACLE',
            selection_source='PRODUCTION_CPP_HEURISTIC_NO_ONLINE_TUNING')
        validate_gemm(result,w,tokens,route)
        return result
    finally:
        sdk.synchronize(r.stream)
        if replay:replay.close()
        d.close();r.close()


def reducer_case(a,w,evidence):
    sdk=SDK(a.sdk);graph_bind(sdk);r=Resources(sdk);replay=None;handles=[]
    lib=bind_reducer(a.bundle/'libprefill_reducer.so')
    try:
        probe_lib,_,probe=bind_probe(a.bundle);dev=device(sdk,probe)
        if dev!=evidence['authority']['device']:raise ValueError('reducer PPU differs from component study')
        count=w['m']*w['n'];partial_bytes=count*w['split']*4
        copies=max(2,math.ceil(2.25*dev['l2_bytes']/partial_bytes))
        # Only a small chunk is resident on the CPU, even for multi-GB partials.
        chunk=1<<20;outputs=[];partials=[]
        for i in range(copies):
            p=r.alloc(partial_bytes+128)+16 if w['compact'] else r.alloc(partial_bytes)
            partials.append(p)
            out=r.alloc(count*2+256);outputs.append(out);r.fill(out,0xA5,count*2+256)
            for s in range(w['split']):
                for start in range(0,count,chunk):
                    data=partial_values(start,min(chunk,count-start),s)
                    checked(sdk.lib.hggcMemcpy(p+(s*count+start)*4,data.ctypes.data,data.nbytes,1),'partial H2D')
            sdk.synchronize(None)
            h=C.c_void_p()
            checked(lib.prepare(w['compact'],w['m'],w['n'],w['split'],p,partial_bytes,out+128,C.byref(h)),
                    'production reducer prepare')
            handles.append(h)
        def sequence():
            for h in handles:checked(lib.run(h,r.stream),'production reducer')
            return 0
        def proof(i=0,negative=False):
            sdk.synchronize(r.stream)
            out=outputs[i]
            if sdk.download(out,128)!=b'\xa5'*128 or sdk.download(out+128+count*2,128)!=b'\xa5'*128:
                raise ValueError('reducer guard changed')
            bad=0
            for start in range(0,count,chunk):
                length=min(chunk,count-start)
                got=np.frombuffer(sdk.download(out+128+start*2,length*2),dtype='<u2')
                want=reducer_expected(start,length,w['split']).view('<u2')
                bad+=int(np.count_nonzero((got!=want)&~(((got&0x7fff)==0)&((want&0x7fff)==0))))
            if (bad==0) == negative:raise ValueError(f'reducer numeric/negative control failed bad={bad}')
            return bad
        sequence()
        for i in range(copies):proof(i)
        # A zeroed split plane must be noticed by the fixed-order oracle.
        r.fill(partials[0],0,count*4);checked(lib.run(handles[0],r.stream),'reducer planted partial');bad=proof(0,True)
        for start in range(0,count,chunk):
            data=partial_values(start,min(chunk,count-start),0)
            checked(sdk.lib.hggcMemcpy(partials[0]+start*4,data.ctypes.data,data.nbytes,1),'restore partial')
        sdk.synchronize(None)
        for out in outputs:r.fill(out+128,0x7e,count*2)
        replay=Replay(sdk,r.stream,sequence,2)
        result=timing(r,replay,copies,lambda:(proof(0),proof(copies-1)),w['id'])
        result.update(status='PASS',workload=w,device=dev,copies=copies,partial_bytes=partial_bytes,
            input_ring_bytes=copies*partial_bytes,output_ring_bytes=copies*count*2,
            fast_path=bool(lib.fast(handles[0])),raw_bad=0,negative_bad=bad,guards='PASS',
            partial_layout='FP32_S_M_N',output_dtype='FP16',addition_order='INCREASING_S',
            scope='REDUCER_ONLY_SYNTHETIC_PARTIALS_ROTATING_NOT_PRODUCER_CONSUMER_CACHE',
            producer_timed=False,production_changed=False,
            useful_gbps=(partial_bytes+count*2)/result['median_us']/1000)
        return validate_reducer(result,w)
    finally:
        sdk.synchronize(r.stream)
        if replay:replay.close()
        for h in handles:lib.destroy(h)
        r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('gemm','reducer'));p.add_argument('--sdk',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=BUNDLE);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();tasks=families() if a.mode=='gemm' else reducer_cases()
    if a.plan_only:
        print(json.dumps(dict(mode=a.mode,cases=len(tasks),measurements=len(tasks)*4 if a.mode=='gemm' else len(tasks),tasks=tasks)));return
    manifest,plan=verify(a.bundle,a.sdk);evidence=json.loads((a.bundle/'fixture-receipts.json').read_text())
    if evidence['board_sha256']!=sha(BOARD):raise ValueError('component board differs from fixture receipt')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.case:
        w=next(w for w in tasks if w['id']==a.case)
        if a.mode=='reducer':
            save(a.output/(w['id']+'.json'),reducer_case(a,w,evidence));return
        weights=Weights(w,progress=lambda done,total:print(f'PREFILL_MEASURE_FIXTURE case={w["id"]} experts={done}/{total}',flush=True))
        failures=[]
        for t in (2048,4096):
            for route in ((0,1) if w['experts']==1 else (2,3)):
                path=a.output/(w['id']+f'-t{t}-r{route}.json')
                if path.exists():continue
                try:
                    result=gemm_case(a,w,weights,t,route,evidence,plan)
                except Exception:
                    traceback.print_exc();failures.append((t,route))
                    print(f'PREFILL_MEASURE_FAILURE case={w["id"]} tokens={t} route={route} remaining_continue=1',flush=True)
                    continue
                save(path,result)
                print('PREFILL_MEASURE_RESULT '+json.dumps({k:result[k] for k in ('workload','tokens','route','median_us','error','selection')}),flush=True)
        if failures:raise SystemExit(1)
        return
    sources=['tools/run_kpack_prefill_measurement.py','tools/kpack_prefill_measurement.py',
        'tools/kpack_warmup_fixture.py','tools/kpack_bf16_fixture.py','tools/kpack_dequant_fixture.py',
        'tools/run_kpack_gemv_gate.py','tools/run_kpack_grouped_decode_probe.py',
        'tools/run_kpack_grouped_device_gate.py','quactlize/dispatch/native.py','dev/gemv_ppu/decode_sweep.py']
    authority=dict(mode=a.mode,bundle_sha256=sha(a.bundle/'manifest.json'),board_sha256=sha(BOARD),
        sources={name:sha(ROOT/name) for name in sources},runtime=manifest['runtime'],
        python_packages=packages(),expected_device=evidence['authority']['device'])
    path=a.output/'authority.json'
    if path.exists() and json.loads(path.read_text())!=authority:raise ValueError('resume identity differs')
    save(path,authority)
    base=[sys.executable,'-u',__file__,a.mode,'--sdk',str(a.sdk),'--bundle',str(a.bundle),'--output',str(a.output)]
    failed=[];complete=[];started=time.monotonic()
    for i,w in enumerate(tasks):
        specs=[(a.output/(w['id']+f'-t{t}-r{route}.json'),t,route) for t in (2048,4096)
            for route in ((0,1) if w.get('experts')==1 else (2,3))] if a.mode=='gemm' else [(a.output/(w['id']+'.json'),None,None)]
        if not all(p.exists() for p,_,_ in specs):
            print(f'PREFILL_MEASURE_START mode={a.mode} case={w["id"]} index={i+1}/{len(tasks)}',flush=True)
            with (a.output/(w['id']+'.log')).open('a') as log:
                child=subprocess.Popen(base+['--case',w['id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in child.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                rc=child.wait()
            if rc:failed.append(dict(case=w['id'],rc=rc))
        for path,t,route in specs:
            if path.exists():
                r=json.loads(path.read_text())
                if a.mode=='gemm':validate_gemm(r,w,t,route)
                else:validate_reducer(r,w)
                if r['device']!=evidence['authority']['device']:raise ValueError('resumed device differs')
                complete.append(r)
        elapsed=time.monotonic()-started
        print(f'PREFILL_MEASURE_ETA mode={a.mode} cases={i+1}/{len(tasks)} measurements={len(complete)} failed={len(failed)} elapsed_minutes={elapsed/60:.1f} remaining_minutes={elapsed/(i+1)*(len(tasks)-i-1)/60:.1f} method=OBSERVED_CASE_AVERAGE',flush=True)
    expected=88 if a.mode=='gemm' else len(tasks)
    files={p.name:sha(p) for p in a.output.glob('*.json') if p.name!='result.json'}
    status='PASS' if len(complete)==expected and not failed else 'INCOMPLETE'
    save(a.output/'result.json',dict(status=status,mode=a.mode,expected=expected,completed=len(complete),
         elapsed_seconds=time.monotonic()-started,failed=failed,files=files,rows=complete,production_changed=False))
    with (a.output/'summary.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['case','tokens','route','split','median_us','scope'])
        for r in complete:writer.writerow([r['workload']['id'],r.get('tokens',''),r.get('route',''),
            r.get('selection',r['workload']).get('split',''),r['median_us'],r['scope']])
    print(f'PREFILL_MEASURE_DONE mode={a.mode} status={status} measured={len(complete)}/{expected} results={a.output}',flush=True)
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise SystemExit(1)
