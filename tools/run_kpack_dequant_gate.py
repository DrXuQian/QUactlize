#!/usr/bin/env python3
"""Measure SF expansion and full BF16 dequant independently; never launch GEMM."""
import argparse
import ctypes as C
import hashlib
import importlib
import importlib.metadata
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
from quactlize.dequant.native import Call, traffic, verify, bind, config_ids, selected_configs
from quactlize.execution.native import arrangement
from quactlize.runtime.native import SDK, checked
from quactlize.runtime.compiler import sha
from tools.kpack_dequant_fixture import fixture, compare
from tools.run_kpack_gemv_gate import Resources
from tools.run_kpack_grouped_device_gate import graph_bind
from tools.run_kpack_grouped_decode_probe import Replay
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange, acu_launch_command
from reference import gguf_kpack as ref

BUNDLE=ROOT/'prebuilt/ppu0010/kpack-dequant-v2'
DENSE=((1024,5120),(5120,8192),(5120,25600),(8192,5120),(25600,5120))
GROUPED=((512,2048),(512,3072),(2048,512),(3072,512),(1024,2048),(1024,3072))


def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def work(qtypes):
    rows=[]
    for q in range(10,15):
        rows.extend(dict(q=q,n=256,k=512,experts=3,operation=o,smoke=True) for o in (0,1))
    for q in qtypes:
        shapes=[(n,k,1) for n,k in DENSE]+[(n,k,e) for n,k in GROUPED for e in (8,256)]
        rows.extend(dict(q=q,n=n,k=k,experts=e,operation=o,smoke=False)
                    for n,k,e in shapes for o in (0,1))
    for r in rows:
        r['id']=f'q{r["q"]}-n{r["n"]}-k{r["k"]}-e{r["experts"]}-'+('full' if r['operation'] else 'sf')
        if r['smoke']:r['id']+='-smoke'
    return rows


def selected_work(qtypes,inventory):
    tasks=work(qtypes)
    if inventory=='all':return tasks
    if inventory=='full-reader' and qtypes and all(q in (12,13) for q in qtypes):
        return [w for w in tasks if w['operation']==1 and w['q'] in qtypes]
    raise ValueError('full-reader inventory is Q4/Q5 full dequant only')


def control_config(record):
    w=record['workload']
    generation=record.get('config_generation',1)
    prior=config_ids(w['q'],w['operation'],max(1,generation-1))
    controls=[r for r in record['rows'] if r['config'] in prior]
    if not controls:raise ValueError('previous best controls missing')
    return min(controls,key=lambda r:r['median_us'])['config']


def pattern(w, config):
    """First warp/pass byte footprint, not measured DRAM traffic."""
    q,n,k=w['q'],w['n'],w['k'];s=ref.SPECS[q]
    if w['operation']:
        coords=[(0,lane) if config==0 else ((lane%4)*8,lane//4) if config>=5 else (lane,0) for lane in range(32)]
    else:
        cols=8 if config==0 else 16 if config==1 else 32
        coords=[(lane//4,(lane%4)*s.group_size) if config==0
                else (lane%cols,(lane//cols)*s.group_size) for lane in range(32)]
    fields={}
    def field(name, addresses, width):
        unique={b for a in addresses for b in range(a,a+width)}
        fields[name]=dict(lane_requests=len(addresses),width=width,unique_bytes=len(unique),
            requested_bytes=width*len(addresses),sectors={str(size):len({b//size for b in unique}) for size in (32,64,128)},
            byte_addresses=addresses)
    if w['operation']:
        field('low_word',[2*(ref._placed_word_slot(kk,s.low_bits)[0]*n+col) for col,kk in coords],16 if config>=5 else 2)
        if s.high_bits:
            addresses=[]
            for col,kk in coords:
                if q==13:
                    pn=(col&~15)|(col&7)|(((kk>>7)&1)<<3)
                    kg=(kk//256)*16|(((kk>>6)&1)<<3)|(kk&7)
                else:pn,kg=col,ref._placed_word_slot(kk,s.high_bits)[0]
                addresses.append(2*(kg*n+pn))
            field('high_word',addresses,16 if config>=5 else 2)
    headers=[];codes=[];mins=[]
    for col,kk in coords:
        sb=kk//256;g=(kk//s.group_size)%s.groups
        unit=((sb//s.superblocks_per_unit)*n+col)*s.unit_bytes+(sb%s.superblocks_per_unit)*s.sb_bytes
        headers.append(unit);codes.append(unit+ref._unit_bit(s,g,0)//8)
        if s.has_min:mins.append(unit+ref._unit_bit(s,g,1)//8)
    if w['operation'] and config>=7:
        field('metadata_cta_unit16',[lane*16 for lane in range(32)],16)
    elif (not w['operation'] and config>=4) or (w['operation'] and config>=5 and q in (12,13)):
        field('metadata_unit16',headers[:4] if w['operation'] else headers,16)
    else:
        select=slice(0,4) if w['operation'] and config==5 else slice(None)
        field('header_d_u16',headers[select],2);field('first_scale_byte',codes[select],1)
        if s.has_min:
            field('header_dmin_u16',[x+2 for x in headers[select]],2);field('first_min_byte',mins[select],1)
    if w['operation'] and config>=4:
        per_row=32 if config==8 else 16
        field('bf16_output_uint4',[(i//per_row*k+i%per_row*8)*2 for i in range(32)],16)
    elif w['operation'] and config:
        per_row=64 if config==3 else 16
        field('bf16_output_pairs',[(i//per_row*k+i%per_row*2)*2 for i in range(32)],4)
    else:
        field('output_scalar',[2*(col*k+kk if w['operation'] else (kk//s.group_size)*n+col) for col,kk in coords],2)
    details={}
    if w['operation'] and config>=6:
        stage_k=256 if config==8 else 128;tile_k=256 if config>=8 else 128
        details=dict(tile_n=32,tile_k=tile_k,stage_k=stage_k,threads=128,a_bytes=0,
            cta_count=(w.get('experts',1)*n//32)*(k//tile_k),
            shared_weight_bytes=32*stage_k*4,shared_metadata_bytes=512 if config>=7 else 0,
            metadata_requested_bytes_per_n_superblock=128 if config==6 else 32 if config==7 else 16,
            cta_barriers={6:1,7:2,8:2,9:4}[config],
            shared_layout='CuTe Swizzle<3,2,3>(N*StageK+K) xor (N&24)',
            shared_load='two aligned uint4 reads per output vector; PPU bank counters decide benefit')
    return dict(scope='FIRST_WARP_PASS_SOURCE_ADDRESS_MODEL_BASE_ALIGNED_128',fields=fields,**details,
                full_arithmetic='FP32_RAW_GGUF_NOT_FP16_FAST_DEQUANT',
                metadata_straddles='second byte only for crossing fields; first-byte footprint shown')


def device(sdk, probe):
    l2,sm,warp=C.c_int(),C.c_int(),C.c_int()
    checked(probe(C.byref(l2),C.byref(sm),C.byref(warp)),'dequant device attributes')
    if min(l2.value,sm.value)<=0 or warp.value!=32:raise ValueError('invalid device attributes')
    return device_identity(sdk)|dict(l2_bytes=l2.value,compute_units=sm.value,warp=warp.value)


def packages():
    names=('numpy','torch','gguf')
    return {name:dict(version=importlib.metadata.version(name),
        module_sha256=sha(importlib.import_module(name).__file__)) for name in names} | {
        'gguf.quants':dict(module_sha256=sha(importlib.import_module('gguf.quants').__file__))}


def run_logged(command, path):
    with path.open('w') as log:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            log.write(line);log.flush()
            if line.startswith(('KPACK_DEQUANT_','ValueError:','RuntimeError:')):print(line,end='',flush=True)
        return child.wait()


class Bench:
    def __init__(self,a,w):
        self.w=w;self.sdk=SDK(a.sdk);graph_bind(self.sdk);self.r=Resources(self.sdk)
        self.lib,self.fn,probe=bind(a.bundle);self.device=device(self.sdk,probe)
        self.generation=json.loads((a.bundle/'manifest.json').read_text()).get('config_generation',1)
        self.arr=arrangement(w['q']);self.bytes=traffic(w['q'],w['n'],w['k'],w['experts'],w['operation'])
        self.graphs={};self.calls=[]
        started=time.monotonic()
        print('KPACK_DEQUANT_SETUP case='+w['id']+' phase=fixture',flush=True)
        self.planes,self.gold=fixture(w['q'],w['n'],w['k'],w['experts'],w['operation'],
            progress=lambda e,total:print(f'KPACK_DEQUANT_FIXTURE case={w["id"]} experts={e}/{total}',flush=True))
        # Rotate both inputs and outputs. Small metadata expansions cannot
        # masquerade as DRAM measurements by repeatedly overwriting hot L2.
        self.copies=1 if w['smoke'] or a.profile else max(2,math.ceil(2.25*self.device['l2_bytes']/self.bytes['reads']))
        self.replays=self.copies*2
        self.buffers={};self.output=[];self.zero=[]
        names=('low','high','units') if w['operation'] else ('units',)
        async_copy=self.sdk.lib.hggcMemcpyAsync
        async_copy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p];async_copy.restype=C.c_int
        for name in names:
            size=self.planes[name].nbytes
            if not size:continue
            p=self.r.alloc(size*self.copies)
            checked(self.sdk.lib.hggcMemcpy(p,self.planes[name].ctypes.data,size,1),'fixture upload')
            self.sdk.synchronize(None)
            for i in range(1,self.copies):
                checked(async_copy(p+i*size,p,size,3,self.r.stream),'untimed device ring initialization')
            self.buffers[name]=p
        self.guard=128
        self.stride=self.bytes['output']+2*self.guard
        for dest in (self.output,)+( (self.zero,) if not w['operation'] else ()):
            base=self.r.alloc(self.stride*self.copies);self.r.fill(base,0xa5,self.stride*self.copies)
            dest.extend(base+i*self.stride+self.guard for i in range(self.copies))
        if any(p%128 for p in [*self.buffers.values(),*self.output,*self.zero]):
            raise ValueError('fixture base alignment differs from the address model')
        for i in range(self.copies):
            pointers={name:p+i*self.planes[name].nbytes for name,p in self.buffers.items()}
            self.calls.append(Call(version=1,size=C.sizeof(Call),qtype=w['q'],n=w['n'],k=w['k'],experts=w['experts'],
                operation=w['operation'],output=self.output[i],zero=self.zero[i] if self.zero else None,
                output_bytes=self.bytes['output'],stream=self.r.stream.value,
                low_bytes=self.bytes['low'],high_bytes=self.bytes['high'],unit_bytes=self.bytes['units'],**pointers))
        self.sdk.synchronize(self.r.stream);self.setup_seconds=time.monotonic()-started

    def run(self,config,i=0):
        c=self.calls[i];c.config=config
        return self.fn(C.byref(c),C.byref(self.arr))

    def read(self,i=0):
        self.sdk.synchronize(self.r.stream)
        items=[]
        for p in (self.output[i],)+((self.zero[i],) if self.zero else ()):
            raw=np.frombuffer(self.sdk.download(p-self.guard,self.stride),dtype='u1')
            if not np.all(raw[:self.guard]==0xa5) or not np.all(raw[-self.guard:]==0xa5):raise ValueError('dequant output guard differs')
            items.append(raw[self.guard:-self.guard].view('<u2').copy())
        return np.array(items).reshape(self.gold.shape)

    def check(self,config):
        for p in (self.output[0],)+((self.zero[0],) if self.zero else ()):
            self.r.fill(p,0x7f,self.bytes['output'])
        checked(self.run(config),'dequant correctness')
        proof=compare(self.read(),self.gold)
        if not self.w['operation'] and proof['signed_zero_differences']:
            raise ValueError('SF metadata must be bit exact including signed zeros')
        name='low' if self.w['operation'] else 'units'
        p=self.buffers[name];self.r.fill(p,0,self.planes[name].nbytes)
        checked(self.run(config),'dequant negative control')
        negative=int(np.count_nonzero(self.read()!=self.gold))
        if negative<self.gold.size//100:raise ValueError('blank input escaped dequant oracle')
        checked(self.sdk.lib.hggcMemcpy(p,self.planes[name].ctypes.data,self.planes[name].nbytes,1),'restore input')
        self.sdk.synchronize(None)
        checked(self.run(config),'restore dequant');compare(self.read(),self.gold)
        return proof|dict(negative_bad=negative,guard='PASS')

    def graph(self,config):
        if config not in self.graphs:
            i=0
            def run():
                nonlocal i
                rc=self.run(config,i%self.copies);i+=1;return rc
            self.graphs[config]=Replay(self.sdk,self.r.stream,run,self.replays)
            for _ in range(2):checked(self.graphs[config](),'first graph launch/warmup excluded')
            self.sdk.synchronize(self.r.stream)
        return self.graphs[config]

    def close(self):
        self.sdk.synchronize(self.r.stream)
        for g in self.graphs.values():g.close()
        self.r.close()


def child(a,w):
    b=None
    try:
        b=Bench(a,w)
        inventory=getattr(a,'inventory','all')
        allowed=selected_configs(w['q'],w['operation'],b.generation,inventory)
        configs=[a.config] if a.profile else allowed
        if any(c not in allowed for c in configs):raise ValueError('profile config outside selected inventory')
        proofs={c:b.check(c) for c in configs}
        if a.profile:
            checked(b.run(a.config),'profile warmup');b.sdk.synchronize(b.r.stream)
            with AcuRange(b.sdk):
                checked(b.run(a.config),'profile isolated dequant');b.sdk.synchronize(b.r.stream)
            print('KPACK_DEQUANT_PROFILE '+json.dumps(dict(workload=w,config=a.config,proof=proofs[a.config],device=b.device)),flush=True)
            return
        samples={c:[] for c in configs};rounds={c:[] for c in configs}
        if not w['smoke']:
            for c in configs:b.graph(c)
            for r in range(3):
                for c in configs[::(-1 if r%2 else 1)]:
                    values=[v/b.replays for v in b.r.samples(b.graph(c),5)]
                    samples[c].extend(values);rounds[c].append(statistics.median(values))
                    compare(b.read(),b.gold);compare(b.read(b.copies-1),b.gold)
                print(f'KPACK_DEQUANT_PROGRESS case={w["id"]} round={r+1}/3 gemm_calls=0',flush=True)
        rows=[]
        for c in configs:
            us=statistics.median(samples[c]) if samples[c] else None
            bandwidth=b.bytes['useful_bytes']/us/1000 if us else None
            row=dict(config=c,proof=proofs[c],samples_us=samples[c],round_medians_us=rounds[c],median_us=us,
                effective_gbps=bandwidth,effective_pct=bandwidth/a.peak_gbps*100 if us else None,
                bandwidth_scope='USEFUL_BYTES_DIVIDED_BY_EVENT_TIME_NOT_ACU_DRAM',pattern=pattern(w,c))
            rows.append(row)
            print('KPACK_DEQUANT_RESULT '+json.dumps(dict(case=w['id'],**{k:v for k,v in row.items() if k not in ('pattern','samples_us')})),flush=True)
        result=dict(status='PASS',workload=w,device=b.device,bytes=b.bytes,rows=rows,config_generation=b.generation,inventory=inventory,
            fixture_hashes={name:hashlib.sha256(p.tobytes()).hexdigest() for name,p in b.planes.items()},
            golden_sha256=hashlib.sha256(b.gold.tobytes()).hexdigest(),copies=b.copies,
            input_ring_bytes=b.bytes['reads']*b.copies,output_ring_bytes=b.bytes['writes']*b.copies,
            calls_per_graph=b.replays,setup_seconds=b.setup_seconds,
            base_alignment_bytes=128,output_guard_bytes=b.guard,
            cache='ROTATING_INPUT_AND_OUTPUT_COMPLETE_RING_TRAVERSALS' if not w['smoke'] else 'UNTIMED_NUMERICAL_SMOKE',
            warmup='FIRST_GRAPH_LAUNCH_AND_TWO_TRAVERSALS_EXCLUDED',gemm_calls=0,
            scope='DEQUANT_ONLY_NOT_COMBINED_OR_REUSABLE_CACHE',peak_gbps=a.peak_gbps,
            peak_authority='OPERATOR_SUPPLIED',production_changed=False)
    finally:
        if b:b.close()
    # Only publish success after the stream and resources closed successfully.
    save(a.output/(w['id']+'.json'),result)


def validate_result(r,w,dev,peak):
    if (r.get('status')!='PASS' or r.get('workload')!=w or r.get('device')!=dev or
        r.get('gemm_calls')!=0 or r.get('scope')!='DEQUANT_ONLY_NOT_COMBINED_OR_REUSABLE_CACHE' or
        r.get('peak_gbps')!=peak or r.get('bytes')!=traffic(w['q'],w['n'],w['k'],w['experts'],w['operation'])):
        raise ValueError('dequant result identity/scope differs: '+w['id'])
    configs=selected_configs(w['q'],w['operation'],r.get('config_generation',1),r.get('inventory','all'))
    if [v['config'] for v in r['rows']]!=configs:raise ValueError('dequant result config set differs')
    for row in r['rows']:
        p=row['proof'];values=row['samples_us']
        if (p['bad'] or p['guard']!='PASS' or p['negative_bad']<=0 or
            (not w['operation'] and p['signed_zero_differences'])):raise ValueError('invalid dequant proof')
        if w['smoke']:
            if values or row['median_us'] is not None:raise ValueError('smoke silently timed')
        else:
            if len(values)!=15 or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('invalid dequant samples')
            if row['median_us']!=statistics.median(values):raise ValueError('dequant median differs')
            if row['round_medians_us']!=[statistics.median(values[i:i+5]) for i in range(0,15,5)]:raise ValueError('round medians differ')
            if row['effective_gbps']!=r['bytes']['useful_bytes']/row['median_us']/1000:raise ValueError('effective byte model differs')
            if row['effective_pct']!=row['effective_gbps']/peak*100:raise ValueError('effective percent differs')
            if r['input_ring_bytes']!=r['bytes']['reads']*r['copies'] or r['output_ring_bytes']!=r['bytes']['writes']*r['copies']:
                raise ValueError('ring byte accounting differs')
            if r['calls_per_graph']!=r['copies']*2:raise ValueError('incomplete ring traversal')
            if r['input_ring_bytes']<2.25*dev['l2_bytes'] or r['output_ring_bytes']<2.25*dev['l2_bytes']:
                raise ValueError('dequant ring is not larger than L2')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True);p.add_argument('--bundle',type=Path,default=BUNDLE)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qtypes',default='12,13')
    p.add_argument('--peak-gbps',type=float,default=2700)
    p.add_argument('--inventory',choices=('all','full-reader'),default='all')
    p.add_argument('--case');p.add_argument('--profile',action='store_true');p.add_argument('--config',type=int,default=0)
    p.add_argument('--acu',type=Path);p.add_argument('--probe-only',action='store_true');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();qtypes=list(map(int,a.qtypes.split(',')))
    if not qtypes or len(set(qtypes))!=len(qtypes) or any(q not in range(10,15) for q in qtypes):raise ValueError('invalid qtype set')
    if not math.isfinite(a.peak_gbps) or a.peak_gbps<=0:raise ValueError('invalid peak bandwidth')
    tasks=selected_work(qtypes,a.inventory)
    if a.plan_only:
        generation=json.loads((a.bundle/'manifest.json').read_text()).get('config_generation',1)
        print(json.dumps(dict(cases=len(tasks),numerical_smokes=sum(w['smoke'] for w in tasks),inventory=a.inventory,
            timing_cells=sum(len(selected_configs(w['q'],w['operation'],generation,a.inventory)) for w in tasks if not w['smoke']),gemm_calls=0,workloads=tasks),indent=2));return
    manifest=verify(a.bundle,a.sdk)
    for w in tasks:selected_configs(w['q'],w['operation'],manifest.get('config_generation',1),a.inventory)
    a.output.mkdir(parents=True,exist_ok=True)
    if a.probe_only:
        lib,_,probe=bind(a.bundle)
        print('KPACK_DEQUANT_DEVICE '+json.dumps(device(SDK(a.sdk),probe)));return
    if a.case:
        child(a,next(w for w in tasks if w['id']==a.case));return
    command=[sys.executable,'-u',__file__,'--sdk',str(a.sdk),'--bundle',str(a.bundle),'--output',str(a.output),
             '--qtypes',a.qtypes,'--peak-gbps',str(a.peak_gbps),'--inventory',a.inventory]
    probe=subprocess.run(command+['--probe-only'],check=True,capture_output=True,text=True)
    lines=[s.split(' ',1)[1] for s in probe.stdout.splitlines() if s.startswith('KPACK_DEQUANT_DEVICE ')]
    if len(lines)!=1:raise ValueError('missing/duplicate dequant device identity')
    dev=json.loads(lines[0])
    harness=[Path(__file__),ROOT/'tools/kpack_dequant_fixture.py',ROOT/'tools/kpack_warmup_fixture.py',
             ROOT/'quactlize/dequant/native.py',ROOT/'reference/gguf_kpack.py',ROOT/'tools/run_kpack_gemv_gate.py',
             ROOT/'tools/run_kpack_grouped_decode_probe.py',ROOT/'tools/run_kpack_grouped_device_gate.py',
             ROOT/'tools/run_kpack_pack_gate.py',ROOT/'tools/profile_kpack_gpu_compact.py',ROOT/'quactlize/runtime/native.py']
    identity=dict(manifest_sha256=sha(a.bundle/'manifest.json'),device=dev,workloads=tasks,peak_gbps=a.peak_gbps,
                  harness={str(x.relative_to(ROOT)):sha(x) for x in harness},runtime=manifest['runtime'],python_packages=packages())
    if a.inventory!='all':identity['inventory']=a.inventory
    authority=a.output/'authority.json'
    if authority.exists() and json.loads(authority.read_text())!=identity:raise ValueError('dequant resume authority differs')
    save(authority,identity)
    print('KPACK_DEQUANT_PLAN '+json.dumps(dict(cases=len(tasks),inventory=a.inventory,
        timing_cells=sum(len(selected_configs(w['q'],w['operation'],manifest.get('config_generation',1),a.inventory)) for w in tasks if not w['smoke']),gemm_calls=0)),flush=True)
    started=time.monotonic();failed=[];complete=[];executed=0
    for i,w in enumerate(tasks):
        out=a.output/(w['id']+'.json')
        if not out.exists():
            rc=run_logged(command+['--case',w['id']],a.output/(w['id']+'.log'))
            executed+=1
            if rc:failed.append(w['id'])
        if out.exists():
            record=validate_result(json.loads(out.read_text()),w,dev,a.peak_gbps)
            if record.get('inventory','all')!=a.inventory or record.get('config_generation',1)!=manifest.get('config_generation',1):
                raise ValueError('resumed candidate inventory differs')
            complete.append(record)
        elapsed=time.monotonic()-started
        eta=elapsed/max(1,executed)*(len(tasks)-i-1)/60
        print(f'KPACK_DEQUANT_CASE completed={i+1}/{len(tasks)} failed={len(failed)} elapsed_s={elapsed:.1f} remaining_minutes={eta:.1f} eta=OBSERVED_CASE_AVERAGE scope=TIMING_PHASE_ONLY_ACU_NOT_INCLUDED',flush=True)
    profiles=[]
    if a.acu:
        anchors={(12,5120,8192,1),(13,2048,512,256)}
        for record in complete:
            w=record['workload']
            if (w['q'],w['n'],w['k'],w['experts']) not in anchors or w['smoke']:continue
            winner=min(record['rows'],key=lambda r:r['median_us'])['config']
            # Compare to the previous admitted implementation, not only the
            # deliberately scalar full-dequant reference.
            old=control_config(record)
            for c in sorted({old,winner}):
                receipt=a.output/(w['id']+f'-c{c}.acu.json')
                profile_identity=dict(result_sha256=sha(a.output/(w['id']+'.json')),
                    bundle_sha256=identity['manifest_sha256'],acu_sha256=sha(a.acu))
                old=json.loads(receipt.read_text()) if receipt.exists() else None
                if (old and old.get('status')=='PASS' and old.get('identity')==profile_identity and
                    all(Path(old[f]).name==old[f] and (a.output/old[f]).is_file() and sha(a.output/old[f])==old[f+'_sha256']
                        for f in ('report','log'))):
                    profiles.append(old);continue
                report=a.output/(w['id']+f'-c{c}.{time.time_ns()}.acurep')
                launch=acu_launch_command(a.acu,report,command+['--case',w['id'],'--profile','--config',str(c)])
                with report.with_suffix('.acu.log').open('w') as log:
                    rc=subprocess.run(launch,stdout=log,stderr=subprocess.STDOUT).returncode
                files=[p for p in a.output.glob(report.name+'*') if p.is_file() and p.stat().st_size>0]
                entries=[json.loads(line.split(' ',1)[1]) for line in report.with_suffix('.acu.log').read_text(errors='replace').splitlines()
                         if line.startswith('KPACK_DEQUANT_PROFILE ')]
                ok=(rc==0 and len(files)==1 and len(entries)==1 and entries[0]['workload']==w and
                    entries[0]['config']==c and entries[0]['device']==dev and entries[0]['proof']['bad']==0)
                entry=dict(case=w['id'],config=c,status='PASS' if ok else 'FAIL',identity=profile_identity,
                    report=files[0].name if ok else report.name,report_sha256=sha(files[0]) if ok else None,
                    log=report.with_suffix('.acu.log').name,log_sha256=sha(report.with_suffix('.acu.log')),
                    cache='ACU_FORCED_COLD_NOT_EVENT_RING')
                profiles.append(entry);save(receipt,entry)
                if not ok:failed.append(w['id']+f'/acu-c{c}')
                print(f'KPACK_DEQUANT_ACU case={w["id"]} config={c} status={"PASS" if ok else "FAIL"}',flush=True)
    result=dict(status='PASS' if not failed and len(complete)==len(tasks) else 'INCOMPLETE',failed=failed,
                complete=len(complete),expected=len(tasks),profiles=profiles,seconds=time.monotonic()-started,
                gemm_calls=0,production_changed=False,
                files={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name not in ('result.json','console.log')})
    save(a.output/'result.json',result)
    print('KPACK_DEQUANT_DONE '+json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise SystemExit(1)
