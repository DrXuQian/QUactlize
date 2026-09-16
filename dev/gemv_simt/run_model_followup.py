#!/usr/bin/env python3
"""Matched M1 reader A/B and ACU; frozen format, compute and shipping control."""
import argparse
import ctypes as C
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import re
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_simt.model_followup import POINTS,SCHEMA
from dev.gemv_simt.native import Runtime,Graph,NativeConfig,checked
from dev.gemv_simt.production import Library as Shipping
from dev.gemv_simt.q8_vector_run import Bench,l2_identity
from dev.gemv_simt.fixture import weights
from dev.gemv_simt.access import pattern
from quactlize.execution.native import SimtCallV2,Call
from quactlize.runtime.compiler import sha
from tools.profile_kpack_gpu_compact import AcuRange,acu_launch_command
from tools.run_kpack_pack_gate import device_identity
from tools.verify_kpack_dispatch import verify


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def verify_candidate(bundle,shipping):
    data=json.loads((bundle/'manifest.json').read_text())
    if data.get('schema')!=SCHEMA or data.get('platform')!='ppu':
        raise ValueError('candidate package identity differs')
    if sha(shipping/'libquactlize_ppu_execution.so')!=data['shipping_execution_sha256']:
        raise ValueError('immutable shipping control differs')
    if [r['point'] for r in data['records']]!=[asdict(p) for p in POINTS]:
        raise ValueError('candidate point/config inventory differs')
    for point,record in zip(POINTS,data['records']):
        if record.get('arms')!=list(point.arms):
            raise ValueError('candidate arm inventory differs')
        path=bundle/record['library']
        if path.parent!=bundle or sha(path)!=record['sha256']:
            raise ValueError('candidate image changed')
    for name,digest in data['source_hashes'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('candidate source changed: '+name)
    return data


class Candidate:
    def __init__(self,path,compute,arm):
        self.compute,self.arm=compute,arm
        self.lib=C.CDLL(str(path.resolve(strict=True)),mode=C.RTLD_LOCAL)
        self.run=self.lib.qk_model_followup_run
        self.run.argtypes=[C.POINTER(SimtCallV2),C.POINTER(NativeConfig),C.c_int]
        self.run.restype=C.c_int

    def prepare(self,call,config):
        d,f=SimtCallV2(call,self.compute),NativeConfig(config)
        return lambda:self.run(C.byref(d),C.byref(f),self.arm)


def summarize(samples,size,target):
    if not samples or any(len(rounds)!=6 or any(len(s)!=15 or
            any(not math.isfinite(x) or x<=0 for x in s) for s in rounds)
            for rounds in samples.values()):
        raise ValueError('confirmation requires six rounds of fifteen finite positive samples')
    result={arm:dict(round_medians_us=[statistics.median(s) for s in rounds])
            for arm,rounds in samples.items()}
    for value in result.values():
        value['median_us']=statistics.median(value['round_medians_us'])
        value['effective_weight_MBU_pct']=size/value['median_us']/2.7e6*100
    control=result['shipping']['median_us']
    for value in result.values():
        value['delta_pct']=100*(value['median_us']/control-1)
        value['target_met']=value['effective_weight_MBU_pct']>=target
    return result


def validate_profile(raw,point,arm):
    rows=list(csv.DictReader(io.StringIO(raw[raw.index('"ID"'):])))
    if len(rows)!=1:raise ValueError('ACU must contain exactly one S1 producer')
    row=rows[0];c=point.config
    if c.variant>=4:
        args=[1,point.compute,c.variant-4,c.columns,c.warps,c.values]
        name='quactlize::execution::simt::q8_vector::kernel'
    else:
        args=[point.q,1,c.variant,c.columns,c.warps,c.values,point.compute]
        name='quactlize::execution::simt::register_reuse'
    if arm!='shipping':
        args.insert(0,int(arm));name='quactlize::execution::model_followup::candidate'
    signature=name+'<'+','.join(map(str,args))+'>(qkg_call_v1,int)'
    if re.sub(r'\s+','',row['Kernel Name'])!='void'+signature:
        raise ValueError('ACU kernel/compute/config differs from requested arm')
    blocks=(8 if point.mode==2 else 1)*point.n//c.tile_n
    if row['Block Size'].replace(' ','')!=f'({c.warps*32},1,1)' or row['Grid Size'].replace(' ','')!=f'({blocks},1,1)':
        raise ValueError('ACU geometry differs from the observed production recipe')
    return dict(kernel=row['Kernel Name'],mangled=row['Kernel Mangled Name'],
                block=row['Block Size'],grid=row['Grid Size'])


def wait_logged(command,log,label):
    with log.open('x') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        started=time.monotonic()
        while process.poll() is None:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f'MODEL_SIMT_WAIT {label} elapsed_s={time.monotonic()-started:.0f} log={log}',flush=True)
    return process.returncode


def read_roof(rt,bench,lib):
    fn=lib.qk_model_read_roof
    fn.argtypes=[C.POINTER(Call),C.c_int];fn.restype=C.c_int
    expected=np.uint32(0)
    for key in ('low','units'):
        expected^=np.bitwise_xor.reduce(np.ascontiguousarray(bench.w.planes[key]).view('<u4').reshape(-1))
    records=[]
    for blocks in (72,144,288):
        c=bench.call
        checked(fn(C.byref(c),blocks),'memory-only check');rt.sync()
        got=np.bitwise_xor.reduce(rt.download(c.workspace,blocks*16).view('<u4'))
        if got!=expected:raise ValueError('memory-only reference skipped weight bytes')
        calls=[lambda c=c:fn(C.byref(c),blocks) for c in bench.calls]
        graph=Graph(rt,calls*max(1,math.ceil(32/len(calls))))
        try:
            graph.sample()
            samples=[graph.sample() for _ in range(15)]
            records.append(dict(blocks=blocks,threads=128,samples_us=samples,
                                median_us=statistics.median(samples),xor='PASS'))
        finally:graph.close()
    return dict(scope='MEMORY_ONLY_NO_DEQUANT_NO_DOT_NOT_GEMV_ADMISSION',records=records,
                fastest_us=min(r['median_us'] for r in records))


def child(args):
    verify(args.shipping,sdk=args.sdk)
    package=verify_candidate(args.bundle,args.shipping)
    point=next(p for p in POINTS if p.name==args.point)
    record=next(r for r in package['records'] if r['point']['name']==point.name)
    rt=Runtime(args.sdk,'ppu')
    graphs={}
    try:
        identity=device_identity(rt)
        l2=l2_identity(dict(l2_bytes=rt.attribute(38)),args.l2_bytes)
        if l2['l2_bytes']<=0:raise ValueError('verified L2 capacity required')
        experts=256 if point.mode==2 else 1
        print(f'MODEL_SIMT_FIXTURE point={point.name} experts={experts}',flush=True)
        if point.q==8:
            w=weights(point.q,point.n,point.k,experts)
        else:
            from tools.kpack_execution_fixture import IndexedWeights
            w=IndexedWeights(point.q,point.n,point.k,experts,
                progress=lambda i,total:print(f'MODEL_SIMT_FIXTURE point={point.name} experts={i}/{total}',flush=True))
        distinct=sum(w.planes[k].nbytes for k in ('low','high','units'))//experts*(8 if point.mode==2 else 1)
        copies=math.ceil(2.25*l2['l2_bytes']/distinct) if args.phase=='test' else 1
        shipping=Shipping(args.shipping/'libquactlize_ppu_execution.so',point.compute)
        providers={'shipping':shipping}
        providers.update({str(a):Candidate(args.bundle/record['library'],point.compute,a) for a in point.arms})
        bench=Bench(rt,shipping,w,1,point.mode,point.channels,point.compute,copies)
        fixture_hashes={key:hashlib.sha256(memoryview(np.ascontiguousarray(w.planes[key])).cast('B')).hexdigest()
                        for key in ('low','high','units') if w.planes[key].size}
        fixture_hashes['a']=hashlib.sha256(bench.data['a'].tobytes()).hexdigest()
        fixture_hashes['ids']=hashlib.sha256(bench.data['ids'].tobytes()).hexdigest()
        actual_bases={key:getattr(bench.call,attr)%128 if getattr(bench.call,attr) else 0
                      for key,attr in [('A','a'),('low','low'),('high','high'),('units','units')]}
        model=pattern(point.q,point.config,point.n,point.k,bases=actual_bases)
        if args.phase=='profile':
            provider=providers[args.arm]
            bench.lib=provider
            _,error=bench.correctness(point.config)
            launch=provider.prepare(bench.call,point.config)
            for _ in range(5):checked(launch(),'excluded warmup')
            rt.sync()
            with AcuRange(rt):
                checked(launch(),'profiled reader');rt.sync()
            bench.output_check()
            save(args.output,dict(status='PASS',point=asdict(point),arm=args.arm,error=error,
                execution_sha256=package['shipping_execution_sha256'],
                candidate_sha256=record['sha256'],device=identity,access=model,fixture_sha256=fixture_hashes,
                scope='FORCED_COLD_ACU_NOT_EVENT_TIMING'))
            return 0
        numeric=[]
        for repeat in (0,1,2,4):
            bench.update(repeat)
            control=None
            for arm,provider in providers.items():
                bench.lib=provider
                got,error=bench.correctness(point.config)
                bits=got.view('<u4')
                if arm=='shipping':control=bits
                elif not np.array_equal(bits,control):
                    raise ValueError(f'{point.name} arm={arm} repeat={repeat}: same-geometry output bits differ')
                numeric.append(dict(arm=arm,repeat=repeat,error=error,matched_bits=True))
        negatives={}
        for arm,provider in providers.items():
            bench.lib=provider
            negatives[arm]=bench.replay_and_negative(point.config)
        bench.lib=shipping;bench.update(0)
        if point.mode==2:bench.invalid_id_negative(point.config)
        for arm,provider in providers.items():
            # Check every physical copy, including guards, before its timed graph.
            # A validated first copy does not validate another base/placement.
            for call in bench.calls:
                bench.poison();checked(provider.prepare(call,point.config)(),'ring-copy check');rt.sync()
                bench.output_check()
            calls=[provider.prepare(c,point.config) for c in bench.calls]
            graphs[arm]=Graph(rt,calls*max(1,math.ceil(32/copies)))
            graphs[arm].sample()
        samples={arm:[] for arm in providers}
        for round_id in range(6):
            order=list(providers)
            if round_id%2:order.reverse()
            for arm in order:
                samples[arm].append([graphs[arm].sample() for _ in range(15)])
            print(f'MODEL_SIMT_PROGRESS point={point.name} round={round_id+1}/6',flush=True)
        target=40 if distinct<2*1024**2 else 60
        summary=summarize(samples,distinct,target)
        roof=read_roof(rt,bench,providers['0'].lib) if point.q==8 else None
        candidates=[arm for arm in summary if arm not in ('shipping','0')]
        best=min(candidates,key=lambda arm:summary[arm]['median_us'])
        result=dict(status='PASS',point=asdict(point),summary=summary,samples=samples,
            best_candidate=best,numeric=numeric,negatives=negatives,memory_only=roof,
            weight_bytes=distinct,target_MBU_pct=target,target_us=distinct/(2.7e6*target/100),
            l2=l2,weight_copies=copies,cold_weight_bytes=copies*distinct,access=model,
            device=identity,execution_sha256=package['shipping_execution_sha256'],
            candidate_sha256=record['sha256'],fixture_sha256=fixture_hashes,
            runtime_sha256=sha(args.sdk/'lib/libhggc_wrapper.so'),
            calls_per_graph=copies*max(1,math.ceil(32/copies)),
            timing_idle_admission='EXTERNAL_LOAD_AUDIT_REQUIRED',
            scope='ROTATING_FULL_S1_CALL_NOT_MODEL_TPOT',
            production_selection_changed=False)
        save(args.output,result)
        print('MODEL_SIMT_RESULT',json.dumps(dict(point=point.name,summary=summary,target_MBU_pct=target)),flush=True)
        return 0
    finally:
        for g in graphs.values():g.close()
        rt.close()


def collect(args):
    verify(args.shipping,sdk=args.sdk)
    verify_candidate(args.bundle,args.shipping)
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for point in POINTS:
        output=args.output/(point.name+'.json')
        log=args.output/(point.name+'.log')
        base=[sys.executable,'-u',str(Path(__file__).resolve()),'--bundle',str(args.bundle),
              '--shipping',str(args.shipping),'--sdk',str(args.sdk),'--point',point.name,
              '--l2-bytes',str(args.l2_bytes)]
        command=base+['--phase','test','--output',str(output)]
        save(args.output/(point.name+'.command.json'),command)
        rc=wait_logged(command,log,'point='+point.name)
        row=dict(point=point.name,rc=rc,status='FAIL',log=str(log))
        if rc==0:
            result=json.loads(output.read_text())
            if result.get('status')!='PASS':raise ValueError('child succeeded without full validation')
            row.update(status='PASS',summary=result['summary'])
            if args.acu:
                row['profiles']=[]
                for arm in ('shipping',result['best_candidate']):
                    stem=args.output/(point.name+'-'+arm)
                    command=acu_launch_command(args.acu,stem,base+['--phase','profile','--arm',arm,
                        '--output',str(stem)+'.json'])
                    save(Path(str(stem)+'.command.json'),command)
                    print(f'MODEL_SIMT_ACU point={point.name} arm={arm}',flush=True)
                    rc=wait_logged(command,Path(str(stem)+'.log'),'point='+point.name+' acu_arm='+arm)
                    report=Path(str(stem)+'.acurep')
                    receipt=Path(str(stem)+'.json')
                    valid=rc==0 and report.is_file() and receipt.is_file()
                    if valid:
                        received=json.loads(receipt.read_text())
                        valid=(received.get('status')=='PASS' and received.get('arm')==arm and
                               received.get('point')==asdict(point) and
                               all(received.get(k)==result[k] for k in (
                                   'execution_sha256','candidate_sha256','fixture_sha256','device')))
                    profile=dict(arm=arm,rc=rc,report=str(report),sha256=sha(report) if report.is_file() else None)
                    if valid:
                        try:
                            csv_path=Path(str(stem)+'.csv')
                            rc=wait_logged([str(args.acu),'--import',str(report),'--page','raw','--csv'],
                                           csv_path,'import='+stem.name)
                            if rc:raise ValueError('ACU export failed')
                            profile.update(validate_profile(csv_path.read_text(),point,arm))
                        except (OSError,ValueError,KeyError) as e:
                            valid=False;profile['error']=str(e)
                    profile['status']='PASS' if valid else 'FAIL'
                    row['profiles'].append(profile)
                    if not valid:row['status']='FAIL'
        records.append(row)
        save(args.output/'summary.json',dict(records=records,complete=False))
        print(f'MODEL_SIMT_POINT point={point.name} status={row["status"]} remaining_continue=1',flush=True)
    status='PASS' if all(r['status']=='PASS' for r in records) else 'FAIL'
    save(args.output/'summary.json',dict(status=status,records=records,complete=True,selector_changed=False))
    return int(status!='PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','shipping','sdk','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--point',choices=[p.name for p in POINTS])
    parser.add_argument('--phase',choices=('test','profile'),default='test')
    parser.add_argument('--arm',choices=('shipping','0','1','2','3'),default='shipping')
    parser.add_argument('--l2-bytes',type=int,default=0)
    parser.add_argument('--acu',type=Path)
    args=parser.parse_args()
    raise SystemExit(child(args) if args.point else collect(args))
