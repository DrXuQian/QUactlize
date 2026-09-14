#!/usr/bin/env python3
"""GEMM-only timing and a labelled sum with separately measured full dequant."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quactlize.dequant.native import Call, bind, verify
from quactlize.execution.native import arrangement
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from tools.kpack_dequant_fixture import fixture, compare
from tools.kpack_bf16_fixture import Oracle, row_domain
from tools.kpack_bf16_diagnostics import zero_observation
from tools.kpack_bf16_providers import Cublas, DeepGemm, loaded_images
from tools.run_kpack_dequant_gate import BUNDLE, work, save, device, validate_result, packages
from tools.run_kpack_gemv_gate import Resources
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command


def families(qtypes, provider='all'):
    if provider not in ('all', 'cublas', 'deepgemm'):
        raise ValueError('unknown BF16 provider selection')
    experts = (1,256) if provider == 'all' else (1,) if provider == 'cublas' else (256,)
    return [w for w in work(qtypes) if not w['smoke'] and w['operation'] and w['experts'] in experts]


def source_identity():
    paths = ['tools/run_kpack_bf16_gate.py','tools/kpack_bf16_providers.py','tools/kpack_bf16_fixture.py',
             'tools/kpack_bf16_diagnostics.py',
             'tools/run_kpack_dequant_gate.py','tools/kpack_dequant_fixture.py','quactlize/dequant/native.py',
             'tools/run_kpack_gemv_gate.py','tools/profile_kpack_gpu_compact.py',
             'dev/gemv_ppu/decode_sweep.py']
    return {p:sha(ROOT/p) for p in paths}


def validate_provider(identity):
    if identity['provider']=='CUBLAS_PPU_SDK':
        if sha(identity['library'])!=identity['sha256']:raise ValueError('cuBLAS image changed')
    elif identity['provider']=='DEEPGEMM_INSTALLED':
        if (identity.get('implementation')!='PYTHON_JIT' or identity.get('entry_module')!=DeepGemm.MODULE or
            identity.get('entry_kind')!='function' or
            identity.get('benchmark_stream')!='TORCH_CURRENT_NONBLOCKING_STREAM'):
            raise ValueError('DeepGEMM receipt is not the requested Python JIT implementation')
        root=Path(identity['root']).resolve(strict=True)
        for name,h in identity['files'].items():
            path=(root/name).resolve(strict=True)
            if not path.is_relative_to(root) or sha(path)!=h:raise ValueError('DeepGEMM source/image changed: '+name)
        if identity['files'].get(identity['entry_source'])!=identity['entry_sha256']:
            raise ValueError('DeepGEMM Python entry receipt differs')
    else:raise ValueError('unknown BF16 provider')


def validate_dequant(folder, w, dev):
    authority = json.loads((folder/'authority.json').read_text())
    summary = json.loads((folder/'result.json').read_text())
    path = folder/(w['id']+'.json')
    if summary['files'].get(path.name)!=sha(path) or summary['files'].get('authority.json')!=sha(folder/'authority.json'):
        raise ValueError('dequant evidence checksum differs')
    if authority['device']!=dev or w not in authority['workloads']:
        raise ValueError('dequant device or expert/weight domain differs')
    record = validate_result(json.loads(path.read_text()),w,dev,authority['peak_gbps'])
    return record, min(record['rows'],key=lambda x:x['median_us']), authority


def cost_record(gemm_us, dequant_us):
    if any(not math.isfinite(v) or v<=0 for v in (gemm_us,dequant_us)):
        raise ValueError('cost inputs must be measured positive times')
    return dict(full_dequant_us=dequant_us,bf16_provider_us=gemm_us,
        sum_estimate_us=dequant_us+gemm_us,scope='SUM_OF_ISOLATED_MEASUREMENTS_NOT_MEASURED_E2E',
        excludes=['external routing/gather','A conversion to BF16','external output adapters'],
        consumer_cache='ROTATING_BF16_NOT_PROOF_OF_CACHE_STATE_AFTER_DEQUANT')


def prefill_candidates(*, fq_gemm=None, sf_dequant=None, sf_gemm=None, full_dequant=None, bf16_gemm=None):
    """Never choose a path whose measured components are missing."""
    candidates = {}
    for name, components in (('fq',(fq_gemm,)), ('sf',(sf_dequant,sf_gemm)),
                             ('full-bf16',(full_dequant,bf16_gemm))):
        if any(v is None for v in components):
            candidates[name] = dict(status='UNMEASURED',cost_us=None)
        elif any(not math.isfinite(v) or v<=0 for v in components):
            raise ValueError('invalid measured prefill component')
        else:
            candidates[name] = dict(status='MEASURED_COMPONENTS',cost_us=sum(components))
    return candidates


def profile_evidence(log, w, tokens, dev, provider):
    prefix='KPACK_BF16_PROFILE '
    records=[json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]
    if len(records)!=1:raise ValueError('missing/duplicate BF16 profile receipt')
    r=records[0]
    if (r['workload']!=w or r['tokens']!=tokens or r['device']!=dev or r['provider']!=provider or
        not math.isfinite(r['error']) or not 0<=r['error']<.005):
        raise ValueError('BF16 profile workload/provider/numerics differ')
    return r


class Weights:
    def __init__(self,a,w):
        import torch
        self.torch=torch;self.sdk=SDK(a.sdk);self.r=Resources(self.sdk)
        self.lib,self.fn,probe=bind(a.bundle);self.device=device(self.sdk,probe)
        self.grouped=w['experts']>1
        self.stream=torch.cuda.ExternalStream(self.r.stream.value)
        self.execution_stream=C.c_void_p(self.stream.cuda_stream)
        self.output_folder=a.output
        self.w=w;self.arr=arrangement(w['q'])
        self.provider=Cublas(a.sdk,self.execution_stream) if not self.grouped else DeepGemm()
        # Persist the actual entry/binary identity before fixture creation or
        # any numerical check can fail. Previously every failed family lost it.
        save(a.output/(w['id']+'.provider.json'),dict(provider=self.provider.identity,
            device=self.device, stream=int(self.stream.cuda_stream), loaded_images=loaded_images()))
        self.record,self.best,authority=validate_dequant(a.dequant_results,w,self.device)
        if authority['manifest_sha256']!=sha(a.bundle/'manifest.json'):
            raise ValueError('full dequant evidence is from another kernel package')
        print('KPACK_BF16_SETUP case='+w['id']+' phase=fixture',flush=True)
        planes,gold=fixture(w['q'],w['n'],w['k'],w['experts'],1,
            progress=lambda done,total:print(f'KPACK_BF16_FIXTURE case={w["id"]} experts={done}/{total}',flush=True))
        import hashlib
        if hashlib.sha256(gold.tobytes()).hexdigest()!=self.record['golden_sha256']:
            raise ValueError('BF16 weights differ from the independent dequant measurement')
        self.oracle=Oracle(gold)
        self.copies=max(2,math.ceil(2.25*self.device['l2_bytes']/gold.nbytes))
        self.weights=[]
        pointers={name:self.r.upload(planes[name]) if planes[name].size else None for name in ('low','high','units')}
        self.sdk.synchronize(None)  # default-stream pageable H2D -> nonblocking consumer
        with torch.cuda.stream(self.stream):
            weight=torch.empty(tuple(gold.shape),dtype=torch.bfloat16,device='cuda')
            call=Call(version=1,size=C.sizeof(Call),qtype=w['q'],n=w['n'],k=w['k'],experts=w['experts'],
                operation=1,config=self.best['config'],output=weight.data_ptr(),output_bytes=gold.nbytes,
                low_bytes=planes['low'].nbytes,high_bytes=planes['high'].nbytes,unit_bytes=planes['units'].nbytes,
                stream=self.execution_stream.value,**pointers)
            self.sdk.synchronize(self.execution_stream)
            checked(self.fn(C.byref(call),C.byref(self.arr)),'untimed full dequant for BF16 provider')
            self.sdk.synchronize(self.execution_stream)
            got=np.frombuffer(self.sdk.download(weight.data_ptr(),gold.nbytes),dtype='<u2').reshape(gold.shape)
            compare(got,gold)
            self.weights=[weight]+[weight.clone() for _ in range(self.copies-1)]
        self.sdk.synchronize(self.execution_stream)
        sf_work=w|dict(operation=0,id=w['id'].removesuffix('-full')+'-sf')
        self.sf_best=None
        if (a.dequant_results/(sf_work['id']+'.json')).exists():
            _,self.sf_best,_=validate_dequant(a.dequant_results,sf_work,self.device)
        self.setup_identity=dict(provider=self.provider.identity,device=self.device,
            dequant_result_sha256=sha(a.dequant_results/(w['id']+'.json')),
            dequant_config=self.best['config'],golden_sha256=self.record['golden_sha256'],
            weight_ring_bytes=gold.nbytes*self.copies,copies=self.copies,base_alignment_bytes=128)

    def read_bits(self, tensor):
        return np.frombuffer(self.sdk.download(tensor.data_ptr(),tensor.numel()*2),
                             dtype='<u2').reshape(tuple(tensor.shape))

    def check_zero_a(self, a, out, launch, indices, tokens):
        # Same negative as before, now on the same stream as the public entry.
        a.zero_();out.fill_(float('nan'));launch();self.sdk.synchronize(self.execution_stream)
        observation=zero_observation(self.read_bits(out),indices)
        torch_zero=bool((out==0).all())
        input_observation=zero_observation(self.read_bits(a),indices)
        receipt=dict(workload=self.w,tokens=tokens,provider=self.provider.identity,
            stream=int(self.stream.cuda_stream),output=observation,input=input_observation,
            torch_all_zero=torch_zero,loaded_images=loaded_images())
        save(self.output_folder/(self.w['id']+f'-m{tokens}.zero-a.json'),receipt)
        print('KPACK_BF16_ZERO_A '+json.dumps(dict(case=self.w['id'],tokens=tokens,
            stream=int(self.stream.cuda_stream),bad=observation['bad'],nan=observation['nan'],
            finite_nonzero=observation['finite_nonzero'],input_bad=input_observation['bad'],
            torch_all_zero=torch_zero,first=observation['first'])),flush=True)
        if input_observation['bad'] or observation['bad'] or not torch_zero:
            raise ValueError('zero A did not produce finite zero output; see .zero-a.json')
        return 'PASS'

    def measure(self,tokens,profile=False):
        torch=self.torch;w=self.w
        rows,indices,route_hash=row_domain(tokens,w['experts'])
        m=len(indices);bits,coeff=self.oracle.activations(m)
        active_bytes=self.setup_identity['weight_ring_bytes']//w['experts']*int(np.count_nonzero(rows))
        if active_bytes<2.25*self.device['l2_bytes']:raise ValueError('active BF16 weight ring fits in L2')
        guard=64 # BF16 elements: 128-byte aligned output base.
        with torch.cuda.stream(self.stream):
            a=torch.from_numpy(bits.view('i2')).view(torch.bfloat16).to('cuda')
            ids=torch.from_numpy(indices).to('cuda');counts=torch.from_numpy(rows).to('cuda')
            backing=[torch.full((m*w['n']+guard*2,),-123.,dtype=torch.bfloat16,device='cuda') for _ in self.weights]
            output=[b[guard:-guard].reshape(m,w['n']) for b in backing]
            if any(v.data_ptr()%128 for v in [a,*self.weights,*output]):raise ValueError('unaligned BF16 buffer')
            def launch(i=0):
                self.provider(a,self.weights[i] if w['experts']>1 else self.weights[i][0],output[i],ids,counts)
            def proof(i=0):
                self.sdk.synchronize(self.execution_stream)
                if not bool((backing[i][:guard]==-123).all()) or not bool((backing[i][-guard:]==-123).all()):
                    raise ValueError('BF16 provider wrote outside output')
                result=np.frombuffer(self.sdk.download(output[i].data_ptr(),m*w['n']*2),dtype='<u2').reshape(m,w['n'])
                err=self.oracle.error(result,coeff,indices)
                if err>=.005:raise ValueError(f'independent BF16 dot error {err:.6g}')
                return err
            first=time.monotonic();launch();self.sdk.synchronize(self.execution_stream)
            first_seconds=time.monotonic()-first
            error=proof()
            # Input-dependent negative; zero A must overwrite poisoned output.
            self.check_zero_a(a,output[0],launch,indices,tokens)
            zero_bits=np.zeros((m,w['n']),dtype='<u2')
            if self.oracle.error(zero_bits,coeff,indices)<=.005:raise ValueError('zero-A plant escaped dot oracle')
            a.copy_(torch.from_numpy(bits.view('i2')).view(torch.bfloat16));launch();error=max(error,proof())
            if profile:
                with AcuRange(self.sdk):launch();self.sdk.synchronize(self.execution_stream)
                print('KPACK_BF16_PROFILE '+json.dumps(dict(workload=w,tokens=tokens,error=error,
                    provider=self.provider.identity['provider'],device=self.device)),flush=True)
                return None
            for i in range(self.copies):launch(i)
            self.sdk.synchronize(self.execution_stream)
            def sequence():
                for _ in range(2):
                    for i in range(self.copies):launch(i)
            # Python JIT passes the current stream. PyTorch's graph owns any
            # provider-local workspace allocated while capturing the sequence.
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=self.stream):sequence()
            replay=graph.replay
            timing=dict(method='CAPTURED_COMPLETE_RING_EVENTS',stream=int(self.stream.cuda_stream),
                graph_capture=True,host_enqueue_gaps='GRAPH_REPLAY')
            replay();replay();self.sdk.synchronize(self.execution_stream)
            # Verify changed inputs on the exact sequence used for timing.
            a.neg_();replay();self.sdk.synchronize(self.execution_stream)
            coeff*=-1;error=max(error,proof(0),proof(self.copies-1));coeff*=-1
            a.neg_();replay();self.sdk.synchronize(self.execution_stream)
            samples=[]
            for round_id in range(3):
                samples += [v/(2*self.copies) for v in self.r.samples(lambda:replay() or 0,5)]
                error=max(error,proof(0),proof(self.copies-1))
                print(f'KPACK_BF16_PROGRESS case={w["id"]} tokens={tokens} round={round_id+1}/3',flush=True)
            median=statistics.median(samples)
            result=dict(status='PASS',workload=w,tokens=tokens,total_rows=m,topk=1 if w['experts']==1 else 8,
                active_experts=int(np.count_nonzero(rows)),expanded_experts=w['experts'],max_rows=int(rows.max()),
                active_weight_ring_bytes=active_bytes,
                rows=rows.tolist(),routes_sha256=route_hash,identity=self.setup_identity,
                error=error,guards='PASS',zero_a='PASS',changed_a_sequence='PASS',
                changed_a_graph='PASS',timing=timing,samples_us=samples,
                round_medians_us=[statistics.median(samples[i:i+5]) for i in range(0,15,5)],median_us=median,
                first_use_excluded_seconds=first_seconds,calls_per_graph=2*self.copies,
                scope='GEMM_PROVIDER_ONLY_DEQUANT_NEVER_INSIDE_TIMED_GRAPH',
                cache='ROTATING_BF16_WEIGHT_COMPLETE_RING_TRAVERSALS',production_changed=False,
                cost=cost_record(median,self.best['median_us']),loaded_images=loaded_images(),
                candidates=prefill_candidates(sf_dequant=self.sf_best['median_us'] if self.sf_best else None,
                    full_dequant=self.best['median_us'],bf16_gemm=median),
                sf_dequant_us=self.sf_best['median_us'] if self.sf_best else None,
                selection_admission='PENDING_MATCHED_FQ_SF_GEMM')
            del graph
        return result

    def close(self):
        self.sdk.synchronize(self.execution_stream)
        if hasattr(self,'provider'):self.provider.close()
        self.weights.clear();self.r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path,default=BUNDLE)
    p.add_argument('--dequant-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qtypes',default='12,13');p.add_argument('--ms',default='2048,4096')
    p.add_argument('--provider',choices=('all','cublas','deepgemm'),default='all')
    p.add_argument('--gate-first',action='store_true',help='stop if the first family cannot pass; keep any valid cells')
    p.add_argument('--family');p.add_argument('--profile',type=int);p.add_argument('--acu',type=Path)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();qs=list(map(int,a.qtypes.split(',')));ms=list(map(int,a.ms.split(',')))
    if not qs or len(qs)!=len(set(qs)) or any(q not in range(10,15) for q in qs):raise ValueError('invalid formats')
    if not ms or len(ms)!=len(set(ms)) or any(m<128 or m>8192 for m in ms):raise ValueError('large-M range is128..8192')
    tasks=families(qs,a.provider)
    if a.plan_only:
        print(json.dumps(dict(families=len(tasks),gemm_cells=len(tasks)*len(ms),ms=ms,
            stages='SEPARATE_DEQUANT_AND_PROVIDER',workloads=tasks)));return
    verify(a.bundle,a.sdk);a.output.mkdir(parents=True,exist_ok=True)
    if a.family:
        import torch
        torch.cuda.set_device(0)
        w=next(w for w in tasks if w['id']==a.family);b=None
        try:
            b=Weights(a,w)
            for m in ([a.profile] if a.profile else ms):
                result=b.measure(m,profile=bool(a.profile))
                if result:
                    save(a.output/(w['id']+f'-m{m}.json'),result)
                    print('KPACK_BF16_RESULT '+json.dumps({k:result[k] for k in ('workload','tokens','status','median_us','error','cost')}),flush=True)
        finally:
            if b:b.close()
        return
    identity=dict(sources=source_identity(),manifest_sha256=sha(a.bundle/'manifest.json'),provider=a.provider,
        gate_first=a.gate_first,
        dequant_authority_sha256=sha(a.dequant_results/'authority.json'),workloads=tasks,ms=ms,
        python_packages=packages(),runtime=verify(a.bundle,a.sdk)['runtime'])
    authority=a.output/'authority.json'
    if authority.exists() and json.loads(authority.read_text())!=identity:raise ValueError('BF16 resume identity differs')
    save(authority,identity)
    lib,_,probe=bind(a.bundle)
    dev=device(SDK(a.sdk),probe)
    if dev!=json.loads((a.dequant_results/'authority.json').read_text())['device']:
        raise ValueError('BF16 and dequant measurements must use the same physical device')
    command=[sys.executable,'-u',__file__,'--sdk',str(a.sdk),'--bundle',str(a.bundle),
        '--dequant-results',str(a.dequant_results),'--output',str(a.output),'--qtypes',a.qtypes,'--ms',a.ms,
        '--provider',a.provider]
    started=time.monotonic();failed=[];complete=[];profiles=[]
    for i,w in enumerate(tasks):
        wanted=[a.output/(w['id']+f'-m{m}.json') for m in ms]
        # Failed families run in fresh processes. Already complete families
        # retain their results and the exact provider/library receipts.
        if not all(path.exists() for path in wanted):
            missing=[m for path,m in zip(wanted,ms) if not path.exists()]
            retry=command.copy();retry[retry.index('--ms')+1]=','.join(map(str,missing))
            with (a.output/(w['id']+'.log')).open('a') as log:
                child=subprocess.Popen(retry+['--family',w['id']],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in child.stdout:
                    log.write(line);log.flush()
                    if line.startswith(('KPACK_BF16_','ValueError:','RuntimeError:','ImportError:')):print(line,end='',flush=True)
                rc=child.wait()
                if rc:failed.append(dict(case=w['id'],rc=rc))
        for path,m in zip(wanted,ms):
            if path.exists():
                r=json.loads(path.read_text());validate_gemm_result(r,w,m)
                if r['identity']['device']!=dev:raise ValueError('resumed provider device differs')
                if r['identity']['dequant_result_sha256']!=sha(a.dequant_results/(w['id']+'.json')):
                    raise ValueError('resumed dequant result differs')
                validate_provider(r['identity']['provider'])
                for name,h in r['loaded_images'].items():
                    if sha(name)!=h:raise ValueError('resumed provider image differs: '+name)
                complete.append(r)
        elapsed=time.monotonic()-started
        print(f'KPACK_BF16_FAMILY completed={i+1}/{len(tasks)} failed={len(failed)} elapsed_s={elapsed:.1f} remaining_minutes={elapsed/(i+1)*(len(tasks)-i-1)/60:.1f} eta=OBSERVED_FAMILY_AVERAGE',flush=True)
        if a.gate_first and i==0:
            if not all(path.exists() for path in wanted):
                print('KPACK_BF16_PREFLIGHT FAIL remaining_families=NOT_RUN see first-family diagnostics',flush=True)
                break
            print('KPACK_BF16_PREFLIGHT PASS remaining_families_continue=1',flush=True)
    if a.acu:
        anchors={(12,5120,8192,1),(13,2048,512,256)}
        for w in tasks:
            if (w['q'],w['n'],w['k'],w['experts']) not in anchors:continue
            m=2048 if 2048 in ms else ms[0]
            if not any(r['workload']==w and r['tokens']==m for r in complete):continue
            report=a.output/(w['id']+f'-m{m}.{time.time_ns()}.acurep');log=report.with_suffix('.acu.log')
            with log.open('w') as f:
                rc=subprocess.run(acu_launch_command(a.acu,report,command+['--family',w['id'],'--profile',str(m)]),stdout=f,stderr=subprocess.STDOUT).returncode
            ok=rc==0 and report.is_file() and report.stat().st_size>0
            receipt=None;error=None
            if ok:
                provider=next(r['identity']['provider']['provider'] for r in complete if r['workload']==w and r['tokens']==m)
                try:receipt=profile_evidence(log.read_text(),w,m,dev,provider)
                except (KeyError,ValueError) as exc:ok=False;error=str(exc)
            profiles.append(dict(case=w['id'],tokens=m,status='PASS' if ok else 'FAIL',report=report.name,
                report_sha256=sha(report) if report.is_file() else None,log=log.name,log_sha256=sha(log),
                receipt=receipt,error=error))
            if not ok:failed.append(dict(case=w['id']+'/acu',rc=rc))
    with (a.output/'summary.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['case','provider','tokens','total_rows','experts','active_experts','sf_dequant_us',
                         'full_dequant_us','bf16_gemm_us','sum_estimate_us','status'])
        for r in complete:
            writer.writerow([r['workload']['id'],r['identity']['provider']['provider'],r['tokens'],r['total_rows'],
                r['expanded_experts'],r['active_experts'],r['sf_dequant_us'],r['cost']['full_dequant_us'],
                r['median_us'],r['cost']['sum_estimate_us'],r['status']])
    result=dict(status='PASS' if not failed and len(complete)==len(tasks)*len(ms) else 'INCOMPLETE',
        expected=len(tasks)*len(ms),complete=len(complete),failed=failed,seconds=time.monotonic()-started,
        profiles=profiles,production_changed=False,
        files={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name not in ('result.json','console.log')})
    save(a.output/'result.json',result)
    print('KPACK_BF16_DONE '+json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


def validate_gemm_result(r,w,m):
    values=r['samples_us']
    if (r['status']!='PASS' or r['workload']!=w or r['tokens']!=m or len(values)!=15 or
        any(not math.isfinite(v) or v<=0 for v in values) or statistics.median(values)!=r['median_us'] or
        not 0<=r['error']<.005 or not math.isfinite(r['error']) or
        r['scope']!='GEMM_PROVIDER_ONLY_DEQUANT_NEVER_INSIDE_TIMED_GRAPH'):
        raise ValueError('invalid BF16 provider evidence')
    if r['cost']!=cost_record(r['median_us'],r['cost']['full_dequant_us']):raise ValueError('cost sum differs')
    if any(r.get(k)!='PASS' for k in ('guards','zero_a','changed_a_graph')):raise ValueError('missing BF16 numerical control')
    if w['experts']>1:
        timing=r.get('timing',{})
        identity=r['identity']['provider']
        if (r.get('changed_a_sequence')!='PASS' or
            timing.get('method')!='CAPTURED_COMPLETE_RING_EVENTS' or not timing.get('stream') or
            timing.get('graph_capture') is not True or timing.get('host_enqueue_gaps')!='GRAPH_REPLAY' or
            identity.get('implementation')!='PYTHON_JIT' or identity.get('entry_module')!=DeepGemm.MODULE):
            raise ValueError('DeepGEMM actual-stream timing/control differs')
    rows,_,route_hash=row_domain(m,w['experts'])
    if r['rows']!=rows.tolist() or r['routes_sha256']!=route_hash or r['total_rows']!=int(rows.sum()):
        raise ValueError('grouped row/weight domain differs')
    active=int(np.count_nonzero(rows))
    if (r['expanded_experts']!=w['experts'] or r['active_experts']!=active or r['max_rows']!=int(rows.max()) or
        r['topk']!=(1 if w['experts']==1 else 8)):
        raise ValueError('BF16 expert domain differs')
    if r['round_medians_us']!=[statistics.median(values[i:i+5]) for i in range(0,15,5)]:raise ValueError('BF16 round medians differ')
    copies=r['identity']['copies']
    if (copies<2 or r['identity']['weight_ring_bytes']!=2*w['n']*w['k']*w['experts']*copies or
        r['active_weight_ring_bytes']!=2*w['n']*w['k']*active*copies or
        r['identity']['weight_ring_bytes']<2.25*r['identity']['device']['l2_bytes'] or
        r['active_weight_ring_bytes']<2.25*r['identity']['device']['l2_bytes'] or
        r['calls_per_graph']!=copies*2 or r['cache']!='ROTATING_BF16_WEIGHT_COMPLETE_RING_TRAVERSALS'):
        raise ValueError('BF16 weight ring coverage differs')
    if not math.isfinite(r['first_use_excluded_seconds']) or r['first_use_excluded_seconds']<0:
        raise ValueError('invalid excluded first-use time')
    if (r['production_changed'] is not False or r['selection_admission']!='PENDING_MATCHED_FQ_SF_GEMM' or
        r['candidates']!=prefill_candidates(sf_dequant=r['sf_dequant_us'],
            full_dequant=r['cost']['full_dequant_us'],bf16_gemm=r['median_us'])):
        raise ValueError('unmeasured FQ/SF selection must remain pending')


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise SystemExit(1)
