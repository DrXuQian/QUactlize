#!/usr/bin/env python3
"""Prebuilt production-fallback gate; retain per-point winners and failures."""
import argparse
import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path
import os
import re
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.tp2_decode.fallback_plan import SCHEMA,REFERENCE_MANIFEST,POINTS,FIELDS,kernel_names,frozen_kernel_names
from dev.gemv_model.plan import Candidate,Point
from dev.gemv_model.fixture import Bench
from dev.gemv_model.engine import Provider,correctness,timing_graph
from dev.gemv_model.run import save,summarize,logged,profile_records
from dev.gemv_model.access import access
from dev.gemv_simt.native import Runtime,checked
from dev.gemv_simt.q8_vector_run import l2_identity
from quactlize.dispatch.native import Dispatch
from quactlize.execution.native import SimtCallV2,SimtConfig,Arrangement,arrangement
from quactlize.runtime.compiler import sha,LIBRARIES
from tools.run_kpack_pack_gate import device_identity
from tools.profile_kpack_gpu_compact import AcuRange,acu_launch_command


def candidate_entry(record):
    entry=record.get('entry',dict(library='fallback.so',symbol='fallback_run',scope='PRODUCTION_LAUNCH_V2'))
    allowed=(('fallback.so','fallback_run','PRODUCTION_LAUNCH_V2'),
             ('libquactlize_ppu_candidate.so','quactlize_kpack_simt_run_v2','PRODUCTION_C_ABI'))
    if tuple(entry.get(k) for k in ('library','symbol','scope')) not in allowed:
        raise ValueError('candidate entry differs from compiled gate')
    return entry


def verify(bundle,sdk=None):
    m=json.loads((bundle/'manifest.json').read_text())
    if m.get('schema')!=SCHEMA or [r['point'] for r in m['records']]!=json.loads(json.dumps([asdict(p) for p in POINTS])):
        raise ValueError('fallback inventory differs')
    if sha(bundle/'reference-manifest.json')!=REFERENCE_MANIFEST:
        raise ValueError('frozen minimum authority differs')
    for name,h in m['payloads'].items():
        p=bundle/name
        if p.parent!=bundle or p.is_symlink() or sha(p)!=h:raise ValueError('payload differs: '+name)
    for name,h in m['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('compile input differs: '+name)
    for record in m['records']:
        if 'entry' in record:
            entry=candidate_entry(record)
            if entry['library'] not in m['payloads']:raise ValueError('candidate entry payload missing')
            if entry['scope']=='PRODUCTION_C_ABI':
                receipt=json.loads((bundle/'production-execution.json').read_text())
                if (m['payloads'].get('production-execution.json')!=sha(bundle/'production-execution.json') or
                        receipt['sha256']!=m['payloads'][entry['library']]):
                    raise ValueError('production execution receipt differs')
    if sdk and {name:sha(sdk/'lib'/name) for name in m['runtime']}!=m['runtime']:
        raise ValueError('SDK runtime differs')
    return m


class Fallback:
    def __init__(self,b,bundle,record):
        self.b,self.arm,self.handles=b,'fallback',[]
        self.config=Candidate(**record['candidate'])
        self.entry=candidate_entry(record)
        self.lib=C.CDLL(str(bundle/self.entry['library']),mode=C.RTLD_LOCAL)
        self.fn=getattr(self.lib,self.entry['symbol'])
        self.fn.argtypes=[C.POINTER(SimtCallV2),C.POINTER(SimtConfig),C.POINTER(Arrangement)]
        self.fn.restype=C.c_int
        if record['selection']['policy'] is not None:
            d=Dispatch(bundle)
            try:
                got=d.query_smallm_matched(b.call(),arrangement(b.point.q),b.point.compute)
                expected=record['selection']
                if got is None or any(getattr(got.base,k)!=expected[k] for k in ('kind','policy','source_n','source_k','source_tokens')) or any(
                        getattr(got.base.simt,k)!=getattr(self.config,k) for k in FIELDS):
                    raise ValueError('actual production dispatcher disagrees with compiled recipe')
            finally:d.close()

    @property
    def receipt(self):
        return dict(kind='simt',implementation=self.entry['scope'],config=asdict(self.config),
                    identity_scope='ALIGNED_M1_COMPLETE_CALL',
                    kernels=kernel_names(self.b.point,asdict(self.config)))

    def prepare(self,copy=0):
        d=SimtCallV2(self.b.call(copy),self.b.point.compute)
        f=SimtConfig(*[getattr(self.config,k) for k in FIELDS]);a=arrangement(self.b.point.q)
        return lambda:self.fn(C.byref(d),C.byref(f),C.byref(a))

    def close(self):pass


def child(a):
    m=verify(a.bundle,a.sdk);r=next(r for r in m['records'] if r['point']['name']==a.point)
    p=Point(**r['point']);rt=Runtime(a.sdk,'ppu');providers={};graphs={}
    result=dict(status='FAIL',point=p.name,manifest_sha256=sha(a.bundle/'manifest.json'),
                selection=r['selection'],reference_authority=r['reference']['authority'],numerics={})
    try:
        ident=device_identity(rt);l2=l2_identity(dict(l2_bytes=rt.attribute(38)),a.l2_bytes)
        if l2['l2_bytes']<=0:raise ValueError('verified positive L2 size required')
        b=Bench(rt,p,l2['l2_bytes'],profile=bool(a.profile_arm))
        result.update(device=ident,fixture=b.record,runtime=m['runtime'],
                      timing_idle_admission='OPERATOR_IDLE_DEVICE_REQUIRED_NO_INDEPENDENT_AUDIT')
        providers['reference']=Provider(b,a.bundle,r['reference']['record'],r['reference']['arm'])
        providers['fallback']=Fallback(b,a.bundle,r)
        for arm in ([a.profile_arm] if a.profile_arm else providers):
            proof,_=correctness(providers[arm],token_controls=(1,) if a.profile_arm else range(1,9))
            result['numerics'][arm]=proof
            print(f'FALLBACK_GATE point={p.name} arm={arm} PASS',flush=True)
        if a.profile_arm:
            b.update(1,0);b.poison();call=providers[a.profile_arm].prepare()
            for _ in range(5):checked(call(),'excluded profile warmup')
            rt.sync()
            with AcuRange(rt):checked(call(),'profile full call');rt.sync()
            b.check();result.update(status='PASS',arm=a.profile_arm,scope='ACU_FORCED_COLD_NOT_EVENT_TIMING')
            return 0
        for arm,provider in providers.items():
            b.poison();graphs[arm]=timing_graph(provider);b.check()
        samples={k:[] for k in graphs}
        for iteration in range(6):
            for arm in (list(graphs) if iteration%2==0 else list(reversed(graphs))):
                samples[arm].append([graphs[arm].sample() for _ in range(15)])
            print(f'FALLBACK_CONFIRM point={p.name} round={iteration+1}/6',flush=True)
        target=60 if b.weight_bytes>=16*1024*1024 else 40
        timing={arm:summarize(value,b.weight_bytes,target)|providers[arm].receipt for arm,value in samples.items()}
        delta=100*(timing['fallback']['median_us']/timing['reference']['median_us']-1)
        result.update(status='PASS',timing=timing,delta_pct=delta,
            winner=min(timing,key=lambda k:timing[k]['median_us']),
            performance='FALLBACK_FASTER' if delta<=0 else 'RETAIN_REFERENCE',
            scope='ROTATING_COMPLETE_CALL_NOT_MODEL_TPOT',
            access=access(p,providers['fallback'].config,dict(A=b.a.ptr%128,low=b.call().low%128,
                high=(b.call().high or 0)%128,units=b.call().units%128)))
        print('FALLBACK_RESULT '+json.dumps({k:result[k] for k in ('point','status','delta_pct','winner')}),flush=True)
        return 0
    except Exception as e:
        result['error']=str(e);traceback.print_exc();return 1
    finally:
        for graph in graphs.values():graph.close()
        for provider in providers.values():provider.close()
        rt.close();save(a.output,result)


def profile_check(text,record,arm):
    p=Point(**record['point'])
    if arm=='reference' and record['reference']['arm']!='incumbent':
        old=record['reference'];cs=[Candidate(**x) for x in old['record']['candidates']]
        return profile_records(text,p,old['arm'],cs)
    import csv,io
    offset=text.find('"ID"')
    if offset<0:raise ValueError('missing ACU raw CSV')
    rows=[r for r in csv.DictReader(io.StringIO(text[offset:])) if r.get('Kernel Name')]
    if arm=='reference':
        cfg=record['reference']['record']['candidates'][0]
        producer,reducer=frozen_kernel_names(p,cfg)
    else:
        cfg=record['candidate'];producer,reducer=kernel_names(p,cfg)
    if len(rows)!=1+int(reducer is not None):raise ValueError('ACU full-call denominator differs')
    expected=[producer]+([reducer] if reducer else [])
    grids=[(8 if p.mode else 1)*cfg['split']*p.n//(cfg['columns']*cfg['values'])]
    blocks=[cfg['warps']*32]
    if reducer:
        width=128 if reducer.startswith('register_reuse_reduce<') else 64
        grids.append(((8 if p.mode else 1)*p.n+width-1)//width)
        blocks.append(128 if width==128 else 32)
    for row,name,grid,block in zip(rows,expected,grids,blocks):
        if name not in re.sub(r'\s+','',row['Kernel Name']):raise ValueError('ACU producer/reducer identity differs')
        if row['Grid Size'].replace(' ','')!=f'({grid},1,1)' or row['Block Size'].replace(' ','')!=f'({block},1,1)':
            raise ValueError('ACU launch geometry differs')
    return [dict(kernel=r['Kernel Name'],grid=r['Grid Size'],block=r['Block Size']) for r in rows]


def profile_status(row,required):
    if not required:return 'NOT_REQUESTED'
    profiles=row.get('profiles',[])
    return 'PASS' if (len(profiles)==2 and {p.get('arm') for p in profiles}=={'reference','fallback'}
                      and all(p.get('status')=='PASS' for p in profiles)) else 'INCOMPLETE'


def checkpoint(output,records,states,row):
    """Keep later completed points and each completed profiler arm on resume."""
    states[row['point']]=row
    save(output/'summary.json',dict(complete=False,
        records=[states[r['point']['name']] for r in records if r['point']['name'] in states]))


def collect(a):
    m=verify(a.bundle,a.sdk)
    records=[r for r in m['records'] if not a.points or r['point']['name'] in a.points.split(',')]
    if not records or (a.points and set(a.points.split(','))-{r['point']['name'] for r in records}):raise ValueError('unknown/empty points')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    probe=Runtime(a.sdk,'ppu')
    try:physical_device=device_identity(probe)
    finally:probe.close()
    identity=dict(manifest=sha(a.bundle/'manifest.json'),points=[r['point']['name'] for r in records],
        device=os.environ.get('CUDA_VISIBLE_DEVICES'),l2=a.l2_bytes,runtime=m['runtime'],
        physical_device=physical_device,profile_required=bool(a.acu),
        runner={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'dev/gemv_model/engine.py',ROOT/'dev/gemv_model/fixture.py')})
    if a.resume:
        if json.loads((a.output/'identity.json').read_text())!=identity:raise ValueError('resume identity differs')
    else:save(a.output/'identity.json',identity)
    previous={}
    if a.resume and (a.output/'summary.json').is_file():
        previous={r['point']:r for r in json.loads((a.output/'summary.json').read_text()).get('records',[])}
    states=dict(previous)
    summary=[]
    for record in records:
        name=record['point']['name'];dest=a.output/(name+'.json')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--sdk',str(a.sdk),'--bundle',str(a.bundle),
             '--point',name,'--l2-bytes',str(a.l2_bytes)]
        old=json.loads(dest.read_text()) if a.resume and dest.is_file() else {}
        if old.get('status')=='PASS':rc=0
        else:rc=logged(cmd+['--output',str(dest)],a.output/(name+'.log'),'fallback '+name,True)
        value=json.loads(dest.read_text()) if dest.is_file() else dict(status='FAIL')
        row=dict(point=name,rc=rc,status=value['status'],
                 profiles=[dict(p) for p in previous.get(name,{}).get('profiles',[])])
        for key in ('timing','delta_pct','winner','selection','reference_authority','error'):
            if key in value:row[key]=value[key]
        if a.acu and rc==0 and value['status']=='PASS':
            for arm in ('reference','fallback'):
                reused=False
                for old_profile in previous.get(name,{}).get('profiles',[]):
                    if old_profile.get('arm')!=arm or old_profile.get('status')!='PASS':continue
                    report=a.output/old_profile['report'];raw=a.output/old_profile['csv']
                    if report.is_file() and raw.is_file() and sha(report)==old_profile['sha256'] and sha(raw)==old_profile.get('csv_sha256'):
                        profile_check(raw.read_text(errors='replace'),record,arm)
                        reused=True
                        print(f'FALLBACK_RESUME point={name} arm={arm} profile=REUSED',flush=True)
                        break
                if reused:continue
                stem=a.output/f'{name}-{arm}';attempt=0
                while any(a.output.glob(stem.name+'.acu.*')):
                    attempt+=1;stem=a.output/f'{name}-{arm}-{attempt}'
                stem=Path(str(stem)+'.acu');report=Path(str(stem)+'.acurep')
                proof=Path(str(stem)+'.json');raw=Path(str(stem)+'.csv')
                prof=dict(arm=arm,status='FAIL')
                try:
                    launch=acu_launch_command(a.acu,stem,cmd+['--profile-arm',arm,'--output',str(proof)])
                    save(Path(str(stem)+'.command.json'),launch)
                    if logged(launch,Path(str(stem)+'.log'),'ACU '+name+'/'+arm,True):raise ValueError('ACU launch failed')
                    receipt=json.loads(proof.read_text())
                    if receipt.get('status')!='PASS' or any(receipt[k]!=value[k] for k in ('manifest_sha256','device','runtime')) or receipt['fixture']['raw_sha256']!=value['fixture']['raw_sha256']:
                        raise ValueError('ACU numeric/device/fixture receipt differs')
                    if logged([str(a.acu),'--import',str(report),'--page','raw','--csv'],raw,'ACU export'):raise ValueError('ACU export failed')
                    prof.update(status='PASS',report=report.name,sha256=sha(report),csv=raw.name,
                                csv_sha256=sha(raw),
                                kernels=profile_check(raw.read_text(errors='replace'),record,arm))
                except Exception as e:prof['error']=str(e)
                row['profiles']=[p for p in row['profiles'] if p.get('arm')!=arm]+[prof]
                row['profile_status']=profile_status(row,True)
                row['status']='PASS' if row['profile_status']=='PASS' else 'PARTIAL'
                checkpoint(a.output,records,states,row)
        row['profile_status']=profile_status(row,bool(a.acu))
        if row['status']=='PASS' and row['profile_status']=='INCOMPLETE':row['status']='PARTIAL'
        summary.append(row);checkpoint(a.output,records,states,row)
        print(f'FALLBACK_PROGRESS completed={len(summary)}/{len(records)} point={name} status={row["status"]}',flush=True)
    passed=all(r['status']=='PASS' and r['rc']==0 and
               profile_status(r,bool(a.acu))!='INCOMPLETE' for r in summary)
    save(a.output/'summary.json',dict(status='PASS' if passed else 'INCOMPLETE',complete=True,records=summary,
        scope='COMPONENTS_NOT_MODEL_TPOT',production_replacement=False,profile_required=bool(a.acu)))
    with (a.output/'summary.tsv').open('w') as f:
        f.write('point\treference_us\tfallback_us\tdelta_pct\twinner\tstatus\n')
        for r in summary:
            t=r.get('timing',{});times=[t.get(x,{}).get('median_us','NA') for x in ('reference','fallback')]
            f.write('\t'.join(map(str,[r['point'],*times,r.get('delta_pct','NA'),r.get('winner','NONE'),r['status']]))+'\n')
    print(f'FALLBACK_DONE complete=1 gate_pass={int(passed)} profiles={"REQUIRED" if a.acu else "NOT_REQUESTED"} results={a.output}',flush=True)
    return int(not passed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('sdk','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--point');p.add_argument('--points');p.add_argument('--profile-arm',choices=('reference','fallback'))
    p.add_argument('--acu',type=Path);p.add_argument('--l2-bytes',type=int,default=67108864)
    p.add_argument('--resume',action='store_true');p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    a.sdk,a.bundle,a.output=(x.resolve() for x in (a.sdk,a.bundle,a.output))
    if a.verify_only:verify(a.bundle,a.sdk);print('FALLBACK_PACKAGE PASS prebuilt=1 production_replacement=0')
    else:raise SystemExit(child(a) if a.point else collect(a))
