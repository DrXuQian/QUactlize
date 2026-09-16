#!/usr/bin/env python3
"""Q8 M1 full-call topology sweep, immutable shipping control and exact ACU."""
import argparse
import ctypes as C
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.q8_topology import SCHEMA,POINTS,CONFIGS,inventory,config,access
from dev.gemv_simt.native import Runtime,Graph,checked
from dev.gemv_simt.production import Library as Shipping
from dev.gemv_simt.q8_vector_run import Bench,l2_identity
from dev.gemv_simt.fixture import weights
from dev.gemv_simt.access import pattern
from quactlize.execution.native import Call
from quactlize.runtime.compiler import sha


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def wait_logged(command,log,label):
    with log.open('x') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        started=time.monotonic()
        while process.poll() is None:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f'Q8_TOPOLOGY_WAIT {label} elapsed_s={time.monotonic()-started:.0f} log={log}',flush=True)
    return process.returncode


def summarize(samples,size,target):
    if not samples or 'shipping' not in samples:raise ValueError('shipping confirmation is missing')
    result={}
    for key,rounds in samples.items():
        if len(rounds)!=6 or any(len(row)!=15 or any(not math.isfinite(x) or x<=0 for x in row) for row in rounds):
            raise ValueError('confirmation requires six rounds of fifteen positive finite samples')
        medians=[statistics.median(row) for row in rounds];value=statistics.median(medians)
        mbu=size/value/2.7e6*100
        result[key]=dict(round_medians_us=medians,median_us=value,effective_weight_MBU_pct=mbu,target_met=mbu>=target)
    for row in result.values():row['delta_pct']=100*(row['median_us']/result['shipping']['median_us']-1)
    return result


def verify(bundle,shipping=None):
    data=json.loads((bundle/'manifest.json').read_text())
    if data.get('schema')!=SCHEMA or data.get('platform') not in ('ppu','cuda'):
        raise ValueError('Q8 topology manifest identity differs')
    if data.get('configs')!=[asdict(c) for c in CONFIGS]:raise ValueError('compiled configs differ')
    points=[dict(n=n,k=k,incumbent=asdict(c),**inventory(n,k)) for n,k,c in POINTS]
    if data.get('points')!=points:raise ValueError('shape/cell/pruning inventory differs')
    path=bundle/data['library']
    if path.parent!=bundle or sha(path)!=data['library_sha256']:raise ValueError('candidate binary differs')
    for name,digest in data['source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('compiled source differs: '+name)
    if data['platform']=='ppu':
        if shipping is None or sha(shipping/'libquactlize_ppu_execution.so')!=data['shipping_execution_sha256']:
            raise ValueError('immutable shipping control differs')
    return data


class Candidate:
    def __init__(self,bundle,point,cell=None):
        self.point,self.cell=point,cell
        self.lib=C.CDLL(str((bundle/'q8.so').resolve(strict=True)),mode=C.RTLD_LOCAL)
        self.fn=self.lib.q8_topology_control if cell is None else self.lib.q8_topology_run
        self.fn.argtypes=[C.POINTER(Call)]+[C.c_int]*(1 if cell is None else 3)
        self.fn.restype=C.c_int

    def prepare(self,call,unused):
        params=[self.point] if self.cell is None else [self.cell[k] for k in ('recipe','split','reducer')]
        return lambda:self.fn(C.byref(call),*params)

    def probe(self):
        fn=self.lib.simt_candidate_probe
        fn.argtypes=[C.POINTER(C.c_int),C.c_char_p];fn.restype=C.c_int
        values=(C.c_int*6)();name=C.create_string_buffer(256)
        checked(fn(values,name),'candidate image marker/properties')
        return dict(zip(('ordinal','sm','l2_bytes','warp','major','minor'),values))|dict(name=name.value.decode())


def choose_finalists(screen):
    good=[r for r in screen if r.get('status')=='PASS']
    if not good:raise ValueError('no numerically admitted candidate')
    keys=['shipping','clone']
    # Preserve both S1 and complete Split-K calls. A topology-class winner
    # is retained even when the overall screen winner uses another class.
    for subset in (good,[r for r in good if r['cell']['split']==1],
                   [r for r in good if r['cell']['split']>1],
                   [r for r in good if config(r['cell']).columns==16]):
        if subset:
            for r in sorted(subset,key=lambda r:statistics.median(r['samples_us']))[:2 if subset is good else 1]:
                if r['cell']['key'] not in keys:keys.append(r['cell']['key'])
    return keys


def validate_result(result,point,phase):
    n,k,_=POINTS[point];cells=inventory(n,k)['cells']
    if result.get('point')!=point or result.get('shape')!=[1,n,k]:raise ValueError('child shape differs')
    if [r.get('cell') for r in result['screen']]!=cells:raise ValueError('screen cell denominator differs')
    if result['status']=='PASS':
        if result.get('failed') or any(r['status']!='PASS' for r in result['screen']):raise ValueError('PASS contains failed cells')
        expected={('clone',r) for r in (0,1,2)}|{(c['key'],r) for c in cells for r in (0,1,2)}
        actual=[(r['key'],r['repeat']) for r in result['numeric']]
        if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('numeric denominator differs')
        for row in result['numeric']:
            if not math.isfinite(row['error']) or row['error']>=1e-5:raise ValueError('invalid numeric error')
            if row['key']=='clone' and row.get('matched_bits') is not True:raise ValueError('shipping clone not matched')
            if row['key'].endswith('-r1') and row.get('reducer_matched_bits') is not True:raise ValueError('reducer comparison missing')
    if len(result['weak_alignment'])!=2:raise ValueError('weak alignment denominator differs')
    for proof in result['negatives'].values():
        if proof['replays']!=3 or len(proof['errors'])!=3 or proof['negative']!='ZERO_A_REJECTED':raise ValueError('graph/negative proof missing')
    if phase=='test':
        if result['cold_weight_bytes']!=result['weight_bytes']*result['weight_copies'] or result['cold_weight_bytes']<2.25*result['l2']['l2_bytes']:
            raise ValueError('cold ring is incomplete')
        if list(result['samples'])!=choose_finalists(result['screen']):raise ValueError('finalist inventory differs')
        summary=summarize(result['samples'],result['weight_bytes'],result['target_MBU_pct'])
        if result['summary']!=summary:raise ValueError('summary differs from full-call samples')
    return result


def validate_profile(raw,n,k,key,cells):
    rows=list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])))
    cell=next((c for c in cells if c['key']==key),None)
    cfg=next(c for pn,pk,c in POINTS if (pn,pk)==(n,k)) if cell is None else config(cell)
    if cell is None and key not in ('shipping','clone'):raise ValueError('unknown ACU cell')
    if cell is None:
        name='quactlize::execution::simt::q8_vector::kernel' if cfg.variant>=4 else 'quactlize::execution::simt::register_reuse'
        args=[1,0,cfg.variant-4,cfg.columns,cfg.warps,cfg.values] if cfg.variant>=4 else [8,1,cfg.variant,cfg.columns,cfg.warps,cfg.values,0]
    else:
        name='quactlize::execution::q8_topology::kernel';args=[1,0,cfg.variant-4,cfg.columns,cfg.warps,cfg.values]
    expected=[('void'+name+'<'+','.join(map(str,args))+'>(qkg_call_v1,int)',
               f'({cfg.split*n//cfg.tile_n},1,1)',f'({cfg.warps*32},1,1)')]
    if cfg.split>1:
        if cell['reducer']==0:
            expected.append(('voidquactlize::execution::simt::register_reuse_reduce<8>(qkg_call_v1,int)',
                             f'({(n+127)//128},1,1)','(128,1,1)'))
        else:
            expected.append((f'voidquactlize::decode::reduce_decode<{cfg.split},float>(floatconst*,float*,int)',
                             f'({(n+63)//64},1,1)','(32,1,1)'))
    if len(rows)!=len(expected):raise ValueError('ACU producer/reducer denominator differs')
    for row,(signature,grid,block) in zip(rows,expected):
        if re.sub(r'\s+','',row['Kernel Name'])!=signature or row['Grid Size'].replace(' ','')!=grid or row['Block Size'].replace(' ','')!=block:
            raise ValueError('ACU exact specialization/geometry differs')
    return [dict(kernel=r['Kernel Name'],mangled=r['Kernel Mangled Name'],grid=r['Grid Size'],block=r['Block Size']) for r in rows]


def child(args):
    package=verify(args.bundle,args.shipping)
    n,k,incumbent=POINTS[args.point];cells=inventory(n,k)['cells']
    rt=Runtime(args.sdk,package['platform']);graphs={};bench=None;started=time.monotonic()
    try:
        clone=Candidate(args.bundle,args.point)
        identity=clone.probe()
        if package['platform']=='ppu':
            from tools.run_kpack_pack_gate import device_identity
            identity.update(device_identity(rt))
        l2=l2_identity(identity,args.l2_bytes)
        if l2['l2_bytes']<=0:raise ValueError('verified positive L2 capacity required')
        shipping=Shipping(args.shipping/'libquactlize_ppu_execution.so',0) if package['platform']=='ppu' else clone
        providers={'shipping':shipping,'clone':clone}
        providers.update({c['key']:Candidate(args.bundle,args.point,c) for c in cells})
        w=weights(8,n,k,1);size=sum(w.planes[x].nbytes for x in ('low','high','units'))
        copies=math.ceil(2.25*l2['l2_bytes']/size) if args.phase=='test' else 1
        bench=Bench(rt,shipping,w,1,0,1,0,copies)
        bases={key:getattr(bench.call,attr)%128 if getattr(bench.call,attr) else 0
               for key,attr in [('A','a'),('low','low'),('high','high'),('units','units')]}
        fixture_hash={key:hashlib.sha256(w.planes[key].tobytes()).hexdigest() for key in ('low','high','units')}
        fixture_hash['a']=hashlib.sha256(bench.data['a'].tobytes()).hexdigest()
        harness={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'dev/gemv_simt/q8_topology.py',
            ROOT/'dev/gemv_simt/q8_vector_run.py',ROOT/'dev/gemv_simt/run.py',ROOT/'dev/gemv_simt/fixture.py',
            ROOT/'dev/gemv_simt/native.py',ROOT/'dev/gemv_simt/production.py',ROOT/'dev/gemv_simt/access.py',
            ROOT/'dev/gemv_simt/q8_vector_access.py',ROOT/'dev/gemv_ppu/access_pattern.py')}
        authority=dict(point=args.point,shape=[1,n,k],candidate_sha256=package['library_sha256'],
            shipping_sha256=package['shipping_execution_sha256'],device=identity,fixture_sha256=fixture_hash,
            runtime_sha256=sha(args.sdk/('lib/libhggc_wrapper.so' if package['platform']=='ppu' else 'lib64/libcudart.so')),
            harness_sha256=harness,arithmetic='F32_STORAGE_F16_A_ROUNDING_F32_ACCUMULATE_F32_OUTPUT',
            shipping_control='IMMUTABLE_DSO' if package['platform']=='ppu' else 'UNCHANGED_SOURCE_CUDA_CLONE')
        save(args.output.with_suffix('.authority.json'),dict(**authority,inventory=inventory(n,k),
            weight_bytes=size,l2=l2,weight_copies=copies,cold_weight_bytes=copies*size))
        if args.phase=='profile':
            from tools.profile_kpack_gpu_compact import AcuRange
            if args.key not in providers:raise ValueError('unknown profile key')
            bench.lib=providers[args.key];_,error=bench.correctness(incumbent)
            launch=bench.lib.prepare(bench.call,incumbent)
            for _ in range(5):checked(launch(),'excluded warmup')
            rt.sync()
            with AcuRange(rt):checked(launch(),'profiled full call');rt.sync()
            bench.output_check()
            save(args.output,dict(status='PASS',key=args.key,error=error,**authority,
                scope='FORCED_COLD_ACU_FULL_CALL_NOT_EVENT_TIMING'))
            return 0
        records=[];numeric=[];failed=[];reducer_bits={}
        def journal(row):
            with args.output.with_suffix('.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        for repeat in (0,1,2):
            bench.update(repeat);bench.lib=shipping;expected,_=bench.correctness(incumbent)
            bench.lib=clone;got,error=bench.correctness(incumbent)
            if not np.array_equal(got.view('<u4'),expected.view('<u4')):raise ValueError('unchanged clone bits differ from immutable shipping')
            numeric.append(dict(key='clone',repeat=repeat,error=error,matched_bits=True))
        bench.update(0)
        for i,cell in enumerate(cells):
            key=cell['key'];bench.lib=providers[key]
            row=dict(cell=cell,status='FAIL',samples_us=[])
            try:
                for repeat in (0,1,2):
                    bench.update(repeat);got,error=bench.correctness(incumbent)
                    if error>=1e-5:raise ValueError(f'FP32 group-dot error exceeds bounded gate: {error}')
                    matched=None
                    if cell['split']>1:
                        index=(cell['recipe'],cell['split'],repeat)
                        if cell['reducer']==0:reducer_bits[index]=got.view('<u4').copy()
                        else:
                            if index not in reducer_bits:raise ValueError('scalar reducer control did not pass')
                            matched=np.array_equal(got.view('<u4'),reducer_bits[index])
                            if not matched:raise ValueError('ordered float2 reducer bits differ from scalar reducer')
                    numeric.append(dict(key=key,repeat=repeat,error=error,reducer_matched_bits=None if matched is None else bool(matched)))
                bench.update(0)
                row.update(status='PASS',access=access(cell,n,k,bases))
                if args.phase=='test':
                    graph=Graph(rt,[bench.lib.prepare(c,incumbent) for c in bench.calls]*max(1,math.ceil(32/copies)))
                    try:
                        graph.sample() # upload/first launch excluded
                        row['samples_us']=[graph.sample() for _ in range(5)]
                    finally:graph.close()
                    if any(not math.isfinite(v) or v<=0 for v in row['samples_us']):raise ValueError('invalid screen samples')
            except ValueError as e:
                # Numeric failure is not usable timing; later independent cells
                # may proceed only if the context still synchronizes cleanly.
                rt.sync();row.update(status='FAIL',error=str(e),samples_us=[]);failed.append(key)
            records.append(row);journal(row)
            if (i+1)%16==0 or i+1==len(cells):
                print(f'Q8_TOPOLOGY_PROGRESS point={args.point} phase={args.phase} completed={i+1}/{len(cells)} failed={len(failed)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
        result=dict(**authority,screen=records,numeric=numeric,inventory=inventory(n,k),
            weight_bytes=size,l2=l2,weight_copies=copies,cold_weight_bytes=copies*size,
            production_selection_changed=False,timing_idle_admission='EXTERNAL_LOAD_AUDIT_REQUIRED')
        # Exercise the public two-byte metadata alignment, plus the optional
        # reducer's stricter output alignment, without changing the timed ring.
        weak_proofs=[];original_call=bench.call
        unit_bytes=np.ascontiguousarray(w.planes['units'])
        weak_ptr=rt.allocate(unit_bytes.nbytes+32);rt.copy(weak_ptr+2,unit_bytes)
        try:
            for split in (1,4):
                cell=next(c for c in cells if config(c).columns==16 and c['split']==split and c['reducer']==(1 if split>1 else 0))
                weak=Call.from_buffer_copy(original_call);weak.units=weak_ptr+2
                bench.call=weak;bench.lib=providers[cell['key']]
                proof=bench.replay_and_negative(incumbent)
                if split>1:
                    bad=Call.from_buffer_copy(weak);bad.output+=4
                    if bench.lib.prepare(bad,incumbent)()==0:raise ValueError('float2 reducer accepted misaligned output')
                    bad=Call.from_buffer_copy(weak);bad.workspace+=4
                    if bench.lib.prepare(bad,incumbent)()==0:raise ValueError('float2 reducer accepted misaligned partials')
                weak_proofs.append(dict(key=cell['key'],units_mod_alignment=2,proof=proof,
                    rejected_odd_float_output=split>1,rejected_odd_float_partials=split>1))
        finally:bench.call=original_call
        result['weak_alignment']=weak_proofs
        if args.phase=='numeric':
            finalists=['shipping','clone']+[c['key'] for c in cells if c['key'] not in failed]
        else:finalists=choose_finalists(records)
        negatives={}
        for key in finalists:
            bench.lib=providers[key]
            negatives[key]=bench.replay_and_negative(incumbent)
            if args.phase=='numeric':continue
            for call in bench.calls:
                bench.poison();checked(bench.lib.prepare(call,incumbent)(),'ring-copy check');rt.sync();bench.output_check()
            graphs[key]=Graph(rt,[bench.lib.prepare(c,incumbent) for c in bench.calls]*max(1,math.ceil(32/copies)))
            graphs[key].sample()
        result['negatives']=negatives
        if args.phase=='test':
            samples={key:[] for key in finalists}
            for round_id in range(6):
                for key in (finalists if round_id%2==0 else list(reversed(finalists))):
                    samples[key].append([graphs[key].sample() for _ in range(15)])
                print(f'Q8_TOPOLOGY_CONFIRM point={args.point} round={round_id+1}/6 finalists={len(finalists)}',flush=True)
            target=40 if size<2*1024**2 else 60
            summary=summarize(samples,size,target)
            best=min((key for key in finalists if key not in ('shipping','clone')),key=lambda key:summary[key]['median_us'])
            admitted_best=min(('shipping',best),key=lambda key:summary[key]['median_us'])
            result.update(summary=summary,samples=samples,best_candidate=best,retained_best=admitted_best,
                target_MBU_pct=target,target_us=size/(2.7e6*target/100),
                baseline_access=pattern(8,incumbent,n,k,bases=bases),
                scope='ROTATING_FULL_CALL_INCLUDING_SPLIT_REDUCER_NOT_MODEL_TPOT')
            print('Q8_TOPOLOGY_RESULT',json.dumps(dict(shape=[1,n,k],best_candidate=best,retained_best=admitted_best,summary=summary)),flush=True)
        result.update(status='FAIL' if failed else 'PASS',failed=failed,elapsed_s=time.monotonic()-started)
        validate_result(result,args.point,args.phase)
        save(args.output,result)
        return int(bool(failed))
    finally:
        for graph in graphs.values():graph.close()
        if bench:bench.close()
        rt.close()


def collect(args):
    package=verify(args.bundle,args.shipping)
    args.output.mkdir(parents=True,exist_ok=False);records=[]
    for point,(n,k,_) in enumerate(POINTS):
        stem=args.output/f'n{n}-k{k}'
        base=[sys.executable,'-u',str(Path(__file__).resolve()),'--bundle',str(args.bundle),
              '--sdk',str(args.sdk),'--point',str(point),'--l2-bytes',str(args.l2_bytes)]
        if args.shipping:base+=['--shipping',str(args.shipping)]
        command=base+['--phase',args.phase,'--output',str(stem)+'.json']
        save(Path(str(stem)+'.command.json'),command)
        rc=wait_logged(command,Path(str(stem)+'.log'),f'Q8 point={point}')
        row=dict(point=point,shape=[1,n,k],status='FAIL',rc=rc,profiles=[])
        if Path(str(stem)+'.json').is_file():
            result=json.loads(Path(str(stem)+'.json').read_text())
            validate_result(result,point,args.phase)
            row.update(status=result['status'],summary=result.get('summary'),failed=result.get('failed'))
            if args.acu and result.get('summary'):
                from tools.profile_kpack_gpu_compact import acu_launch_command
                keys=list(dict.fromkeys(('shipping',result['best_candidate'])))
                for key in keys:
                    ps=Path(str(stem)+'-'+key)
                    command=acu_launch_command(args.acu,ps,base+['--phase','profile','--key',key,'--output',str(ps)+'.json'])
                    save(Path(str(ps)+'.command.json'),command)
                    print(f'Q8_TOPOLOGY_ACU point={point} key={key}',flush=True)
                    prc=wait_logged(command,Path(str(ps)+'.log'),f'Q8 ACU point={point} key={key}')
                    profile=dict(key=key,rc=prc,status='FAIL');report=Path(str(ps)+'.acurep')
                    try:
                        receipt=json.loads(Path(str(ps)+'.json').read_text())
                        if prc or receipt.get('status')!='PASS' or receipt.get('key')!=key:raise ValueError('ACU child failed')
                        for identity in ('point','shape','candidate_sha256','shipping_sha256','device','fixture_sha256','runtime_sha256','harness_sha256'):
                            if receipt.get(identity)!=result[identity]:raise ValueError('profile authority differs: '+identity)
                        csv_path=Path(str(ps)+'.csv')
                        if wait_logged([str(args.acu),'--import',str(report),'--page','raw','--csv'],csv_path,'Q8 ACU import'):
                            raise ValueError('ACU raw export failed')
                        profile.update(status='PASS',report_sha256=sha(report),csv_sha256=sha(csv_path),
                            kernels=validate_profile(csv_path.read_text(),n,k,key,inventory(n,k)['cells']))
                    except (OSError,ValueError,KeyError) as error:profile['error']=str(error);row['status']='FAIL'
                    row['profiles'].append(profile)
        if rc:row['status']='FAIL'
        records.append(row)
        save(args.output/'summary.json',dict(complete=False,records=records))
        print(f'Q8_TOPOLOGY_POINT point={point} status={row["status"]} remaining_continue=1',flush=True)
    status='PASS' if all(r['status']=='PASS' for r in records) else 'FAIL'
    save(args.output/'summary.json',dict(status=status,complete=True,records=records,platform=package['platform'],
        production_selection_changed=False,device_admission='EXPERIMENT_ONLY'))
    return int(status!='PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','sdk','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--shipping',type=Path)
    parser.add_argument('--point',type=int,choices=range(len(POINTS)))
    parser.add_argument('--phase',choices=('numeric','test','profile'),default='test')
    parser.add_argument('--key',default='shipping')
    parser.add_argument('--l2-bytes',type=int,default=0)
    parser.add_argument('--acu',type=Path)
    args=parser.parse_args()
    if args.phase=='profile' and args.point is None:parser.error('profile requires a point')
    raise SystemExit(child(args) if args.point is not None else collect(args))
