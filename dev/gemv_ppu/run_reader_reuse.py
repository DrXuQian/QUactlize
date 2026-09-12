#!/usr/bin/env python3
"""Confirm the two-shape reader factorial, preserving every independent result."""
import argparse
import csv
import ctypes as C
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest
from dev.gemv_ppu.reader_reuse import SHAPES,REVIEW,Reader,inventory,lookup,selected,plan,payload,access,verify
from dev.gemv_ppu.run_config_sweep import SweepBench,execute,PREFIX,FAIL_PREFIX
from dev.gemv_ppu.run_cold_shapes import recipe as control_recipe
from dev.gemv_ppu.run_cold_geometry import probe_device
from dev.gemv_ppu.run_h800_port import PortBench
from dev.gemv_ppu.run_bload import exact_bits
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import checked
from tools.profile_kpack_gpu_compact import acu_launch_command

ANCHORS=('xplane','raw-reference','baseline')
ROUNDS=6
SAMPLES=15


def recipe(variant,n,k,key):
    if variant=='reader':return lookup(n,k,key).args
    if variant not in ANCHORS or key!='control':raise ValueError('unknown reader comparison arm')
    return selected(n,k).args if variant=='baseline' else control_recipe(variant,n,k)


class ReuseBench(SweepBench):
    def __init__(self,a):
        super().__init__(SimpleNamespace(**(vars(a)|dict(candidate=a.config_bundle))))
        self.reuse_library=C.CDLL(str(a.candidate/payload(self.n,self.k)),mode=C.RTLD_LOCAL)
        probe=self.reuse_library.q4_ppu_probe
        probe.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p];probe.restype=C.c_int
        l2,sm,warp,name=C.c_int(),C.c_int(),C.c_int(),C.create_string_buffer(256)
        checked(probe(C.byref(l2),C.byref(sm),C.byref(warp),name),'reader image marker')
        if (sm.value,warp.value,name.value.decode())!=(self.device['sm'],self.device['warp'],self.device['name']):
            raise ValueError('reader image device differs')
        self.reuse_launch=getattr(self.reuse_library,f'q4_reader_run_{self.n}_{self.k}')
        self.reuse_launch.argtypes=[C.c_int]*4+[C.c_void_p]*5;self.reuse_launch.restype=C.c_int
        self.reader_allowed={r.args for r in inventory(self.n,self.k)}
        self.matched=None

    def invoke(self,cfg,index=0,force_aiu=None):
        if force_aiu is not None or tuple(cfg) not in self.reader_allowed:raise ValueError('reader config not compiled')
        low,units=self.weight_pointers[index%self.copies]
        return self.reuse_launch(*cfg,self.a,low,units,self.output,self.r.stream)

    def direct(self,cfg,force_aiu=None):
        self.r.fill(self.output_base,0xff,self.output_bytes+32)
        low,units=self.weight_pointers[0]
        checked(self.config_launch(*cfg[1:],self.a,low,units,self.output,self.r.stream),'immutable same-config control')
        err=self.error()
        if not math.isfinite(err) or err>=.005:raise ValueError('same-config control failed independent GGUF oracle')
        old=self.sdk.download(self.output,self.output_bytes)
        new,error=super().direct(cfg,force_aiu=force_aiu)
        self.matched=exact_bits(old,new,'reader vs immutable same-config FP32 output')
        return new,max(err,error)


def child(a):
    verify(a.candidate,a.config_bundle,a.previous,a.controls,a.bundle,sources=False)
    keys=json.loads(a.keys)
    if not keys or len(set(keys))!=len(keys):raise ValueError('empty/duplicate child keys')
    bench=None
    try:
        if a.variant=='reader':bench=ReuseBench(a)
        elif a.variant=='baseline':bench=SweepBench(SimpleNamespace(**(vars(a)|dict(candidate=a.config_bundle))))
        else:bench=PortBench(SimpleNamespace(**(vars(a)|dict(candidate=a.controls))))
        n,k=bench.n,bench.k
        if (n,k) not in SHAPES:raise ValueError('shape outside reader experiment')
        for key in keys:
            try:
                row=bench.measure(list(recipe(a.variant,n,k,key)))
                row.update(arm=a.variant,variant=a.variant,config_key=key,phase=a.phase,
                    output_type='F32',inter_cta_split=1,launches_per_call=1,
                    timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',
                    cache_scope='ACU_FORCED_COLD' if a.profile else 'ROTATING_AT_LEAST_2_25_L2',
                    weight_arithmetic='FP32_GROUP_AFFINE' if a.variant in ('reader','baseline') else 'PER_WEIGHT_FP16')
                if a.variant=='reader':
                    r=lookup(n,k,key)
                    row.update(geometry=r.config.geometry(n,k),reader_switches=r.switches,
                        matched_fp32_sha256=bench.matched,same_config_control=r.config.key,
                        code_dequant='LOP3_HALF2',access_models=[access(r,n,k,dict(A=aa,B=bb,metadata=mm))
                            for aa,bb,mm in sorted({(int(bench.a)%128,int(b)%128,int(m)%128) for b,m in bench.weight_pointers})])
                print(PREFIX+json.dumps(row),flush=True)
            except Exception as exc:
                print(FAIL_PREFIX+json.dumps(dict(variant=a.variant,key=key,shape=[1,n,k],phase=a.phase,error=str(exc))),flush=True)
                traceback.print_exc();return 1
        return 0
    finally:
        if bench:bench.close()


def parse_row(row,variant,n,k,key,phase,samples,profile=False):
    row=parse_cells('Q4_PPU_CELL '+json.dumps(row),variant,[recipe(variant,n,k,key)],[1,n,k],'rotating',samples)[0]
    if (row.get('variant')!=variant or row.get('config_key')!=key or row.get('phase')!=phase or
        row.get('zero_a_check')!='PASS' or row.get('output_type')!='F32' or row.get('inter_cta_split')!=1 or
        row.get('launches_per_call')!=1 or row.get('timing_scope')!='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT' or
        row.get('cache_scope')!=('ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2') or
        row.get('weight_arithmetic')!=('FP32_GROUP_AFFINE' if variant in ('reader','baseline') else 'PER_WEIGHT_FP16') or
        row.get('storage')!=('XPLANE' if variant=='xplane' else 'RAW_GGUF' if variant=='raw-reference' else 'CANONICAL_KPACK4')):
        raise ValueError('reader result identity/precision/cache differs')
    if (profile and row['median_us'] is not None) or (not profile and not math.isfinite(row['median_us'])):
        raise ValueError('invalid event/profiler sample')
    if variant=='reader':
        r=lookup(n,k,key);h=row.get('matched_fp32_sha256','')
        if (row.get('geometry')!=r.config.geometry(n,k) or row.get('reader_switches')!=r.switches or
            row.get('same_config_control')!=r.config.key or row.get('code_dequant')!='LOP3_HALF2' or
            len(h)!=64 or any(c not in '0123456789abcdef' for c in h)):
            raise ValueError('reader mapping or same-config raw-bit proof differs')
        models=row.get('access_models',[]);seen=set()
        if not models:raise ValueError('missing observed load patterns')
        for model in models:
            bases=model['first_warp']['base_mod128'];identity=tuple(bases[x] for x in ('A','B','metadata'))
            if model!=access(r,n,k,bases) or any(type(x) is not int or not 0<=x<128 or x%16 for x in identity) or identity in seen:
                raise ValueError('observed address/footprint model differs')
            seen.add(identity)
    return row


def summarize(n,k,records):
    med={key:statistics.median(r['median_us'] for r in rows) for key,rows in records.items() if len(rows)==ROUNDS}
    required={*ANCHORS,*(r.key for r in inventory(n,k))}
    complete=set(med)==required
    candidates=[key for key in med if key not in ('xplane','raw-reference')]
    best=min(candidates,key=med.get) if candidates else None
    r=dict(shape=[1,n,k],status='PASS' if complete else 'INCOMPLETE',median_us=med,selected=best,
        reference_verdict='INCOMPLETE',same_geometry_delta_pct={})
    for reader in inventory(n,k):
        base=Reader(0,reader.config).key
        if reader.key in med and base in med:r['same_geometry_delta_pct'][reader.key]=100*(med[reader.key]/med[base]-1)
    if complete:r.update(reference_delta_pct=100*(med[best]/med['raw-reference']-1),
        previous_delta_pct=100*(med[best]/med['baseline']-1),
        reference_verdict='NOT_SLOWER_THAN_REFERENCE' if med[best]<=med['raw-reference'] else 'REF_REGRESSION')
    return r


def run(a):
    m=verify(a.candidate,a.config_bundle,a.previous,a.controls,a.bundle)
    device=probe_device(a)
    sources=set(m['source_hashes'])|{'dev/gemv_ppu/run_reader_reuse.py','dev/gemv_ppu/run_config_sweep.py',
        'dev/gemv_ppu/run_cold_shapes.py','dev/gemv_ppu/run_cold_geometry.py','dev/gemv_ppu/run_bload.py',
        'dev/gemv_ppu/run_h800_port.py','dev/gemv_ppu/campaign.py','tools/profile_kpack_gpu_compact.py',
        'tools/run_kpack_grouped_decode_probe.py','tools/run_kpack_gemv_gate.py'}
    authority=dict(schema='quactlize.q4-reader-reuse-run.v1',candidate=digest(a.candidate/'manifest.json'),
        device=device,rounds=ROUNDS,samples=SAMPLES,review=digest(REVIEW),
        sources={name:digest(ROOT/name) for name in sorted(sources)},
        fixtures={f'{n}x{k}':digest(a.fixtures/f'q12-n{n}-k{k}-e1-c1.npz') for n,k in SHAPES},
        runtime={name:digest(a.sdk/'lib'/name) for name in m['runtime']})
    a.output.mkdir(parents=True,exist_ok=True);receipt=a.output/'authority.json'
    if receipt.exists() and json.loads(receipt.read_text())!=authority:raise ValueError('resume source/device/runtime/fixture identity differs')
    receipt.write_text(json.dumps(authority,indent=2)+'\n')
    for name in ('manifest.json','isa-stats.json'):(a.output/('build-manifest.json' if name=='manifest.json' else name)).write_bytes((a.candidate/name).read_bytes())
    (a.output/'plan.json').write_text(json.dumps(plan(),indent=2)+'\n')
    (a.output/'access-patterns.json').write_text(json.dumps([dict(shape=[1,n,k],models=[access(r,n,k) for r in inventory(n,k)]) for n,k in SHAPES])+'\n')
    diff=[name for name,sha in authority['runtime'].items() if sha!=m['runtime'][name]]
    if diff:print('Q4_READER_SDK_DIFFERENCE recorded='+','.join(diff)+' real_marker_and_numeric_required=1',flush=True)
    result=dict(status='RUNNING',cases=[],profiles=[],failures=[],production_changed=False)
    started=time.monotonic()
    def save():(a.output/'summary.json').write_text(json.dumps(result)+'\n')
    def batch(variant,n,k,keys,phase,profile=False):
        receipt=a.output/f'n{n}-k{k}-{variant}-{phase}.json';cache=json.loads(receipt.read_text()) if receipt.exists() else {}
        samples=0 if profile else SAMPLES;parsed={}
        for key,entry in cache.items():
            log=a.output/entry['log']
            if log.parent!=a.output:raise ValueError('cached log path differs')
            if log not in parsed:parsed[log]=(digest(log),[json.loads(s[len(PREFIX):]) for s in log.read_text().splitlines() if s.startswith(PREFIX)])
            sha,rows=parsed[log]
            if sha!=entry['log_sha256']:raise ValueError('cached log hash differs')
            found=[r for r in rows if r.get('config_key')==key]
            if len(found)!=1:raise ValueError('cached cell missing/duplicate')
            row=parse_row(found[0],variant,n,k,key,phase,samples,profile)
            if row!=entry['row'] or row['device']!=device:raise ValueError('cached data/device differs')
            if profile:
                report=a.output/entry['report']
                if report.parent!=a.output or digest(report)!=entry['report_sha256']:raise ValueError('cached ACU differs')
        missing=[key for key in keys if key not in cache]
        while missing:
            prefix=a.output/f'n{n}-k{k}-{variant}-{phase}.{time.time_ns()}';log=prefix.with_name(prefix.name+('.acu.log' if profile else '.log'))
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--child','--sdk',str(a.sdk),
                '--candidate',str(a.candidate),'--config-bundle',str(a.config_bundle),'--previous',str(a.previous),
                '--controls',str(a.controls),'--bundle',str(a.bundle),'--fixture',str(a.fixtures/f'q12-n{n}-k{k}-e1-c1.npz'),
                '--variant',variant,'--keys',json.dumps(missing),'--phase',phase,'--l2-bytes',str(a.l2_bytes)]
            if profile:cmd=acu_launch_command(a.acu,prefix,cmd+['--profile'])
            print(f'Q4_READER_PROGRESS shape=1x{n}x{k} arm={variant} phase={phase} cells={len(missing)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            with log.open('x') as stream:rc=execute(cmd,stream)
            consumed=set();failed=set()
            for line in log.read_text().splitlines():
                if line.startswith(PREFIX):
                    raw=json.loads(line[len(PREFIX):]);key=raw.get('config_key')
                    if key not in missing or key in consumed:raise ValueError('unexpected/duplicate child key')
                    consumed.add(key)
                    try:row=parse_row(raw,variant,n,k,key,phase,samples,profile)
                    except Exception as exc:
                        failed.add(key);result['failures'].append(dict(shape=[1,n,k],variant=variant,key=key,phase=phase,error=str(exc),log=log.name));continue
                    if row['device']!=device:raise ValueError('device changed')
                    entry=dict(row=row,log=log.name,log_sha256=digest(log))
                    if profile:
                        reports=list(a.output.glob(prefix.name+'*.acurep'))
                        if len(reports)!=1:raise ValueError('ACU report missing')
                        entry.update(report=reports[0].name,report_sha256=digest(reports[0]))
                    cache[key]=entry
                elif line.startswith(FAIL_PREFIX):
                    error=json.loads(line[len(FAIL_PREFIX):]);key=error['key']
                    if key not in missing or key in consumed or error['variant']!=variant or error['phase']!=phase or error['shape']!=[1,n,k]:raise ValueError('failure identity differs')
                    consumed.add(key);failed.add(key);result['failures'].append(error|dict(log=log.name))
            if not consumed or (rc and not failed):raise ValueError(f'child infrastructure failure rc={rc}; log={log}')
            receipt.write_text(json.dumps(cache)+'\n');missing=[key for key in missing if key not in consumed]
            if missing:print(f'Q4_READER_RESTART remaining={len(missing)} failed={len(failed)} valid={len(cache)}',flush=True)
        return {key:cache[key]['row'] for key in keys if key in cache}
    for n,k in SHAPES:
        case=dict(shape=[1,n,k],records={key:[] for key in (*ANCHORS,*(r.key for r in inventory(n,k)))})
        result['cases'].append(case)
        for turn in range(ROUNDS):
            arms=(*ANCHORS,'reader')
            for variant in arms if turn%2==0 else arms[::-1]:
                keys=[r.key for r in inventory(n,k)] if variant=='reader' else ['control']
                if turn%2:keys=keys[::-1]
                try:
                    for key,row in batch(variant,n,k,keys,f'r{turn}').items():case['records'][key if variant=='reader' else variant].append(row)
                except Exception as exc:result['failures'].append(dict(shape=[1,n,k],variant=variant,phase=f'r{turn}',error=str(exc)))
                save()
        case['comparison']=summarize(n,k,case['records'])
        print('Q4_READER_RESULT '+json.dumps(case['comparison']),flush=True)
        if not a.skip_acu:
            # Every switch combination at the prior winner's geometry, plus
            # the best alternate geometry: counter evidence for each factor.
            profile_keys=[r.key for r in inventory(n,k) if r.config==selected(n,k)]
            other={key:us for key,us in case['comparison']['median_us'].items() if key not in ANCHORS and key not in profile_keys}
            if other:profile_keys.append(min(other,key=other.get))
            for variant,key in [*(('reader',key) for key in profile_keys),*((v,'control') for v in ANCHORS)]:
                try:
                    rows=batch(variant,n,k,[key],'profile',True)
                    if key not in rows:raise ValueError('profiled reader failed')
                    result['profiles'].append(dict(shape=[1,n,k],variant=variant,key=key,status='PASS',row=rows[key]))
                except Exception as exc:result['profiles'].append(dict(shape=[1,n,k],variant=variant,key=key,status='FAIL',error=str(exc)))
                save()
    verify(a.candidate,a.config_bundle,a.previous,a.controls,a.bundle)
    if (any(digest(ROOT/name)!=sha for name,sha in authority['sources'].items()) or digest(REVIEW)!=authority['review'] or
        any(digest(a.sdk/'lib'/name)!=sha for name,sha in authority['runtime'].items())):raise ValueError('authority changed during run')
    ok=not result['failures'] and all(c['comparison']['status']=='PASS' for c in result['cases']) and all(p['status']=='PASS' for p in result['profiles'])
    result.update(status='PASS' if ok else 'INCOMPLETE',seconds=time.monotonic()-started,
        not_slower_than_ref=sum(c['comparison']['reference_verdict']=='NOT_SLOWER_THAN_REFERENCE' for c in result['cases']))
    save()
    with (a.output/'summary.tsv').open('w') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['N','K','selected','best_us','ref_us','vs_ref_pct','verdict'])
        for case in result['cases']:
            r=case['comparison'];m=r['median_us']
            writer.writerow([*case['shape'][1:],r['selected'],m.get(r['selected'],'NA'),m.get('raw-reference','NA'),r.get('reference_delta_pct','NA'),r['reference_verdict']])
    print(f'Q4_READER_COMPLETE status={result["status"]} not_slower_than_ref={result["not_slower_than_ref"]}/2 results={a.output}',flush=True)
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    for name,folder in [('candidate','q4-reader-reuse-v1'),('config-bundle','q4-config-sweep-v1'),('previous','q4-cold-shapes-v1'),('controls','q4-h800-port-v1'),('bundle','q4-simt-ab-v1')]:
        p.add_argument('--'+name,type=Path,default=ROOT/'prebuilt/ppu0010'/folder)
    p.add_argument('--fixtures',type=Path);p.add_argument('--fixture',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--variant',choices=(*ANCHORS,'reader'));p.add_argument('--keys');p.add_argument('--phase',default='r0')
    p.add_argument('--l2-bytes',type=int,default=0);p.add_argument('--child',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--skip-acu',action='store_true');p.add_argument('--acu',type=Path)
    a=p.parse_args();a.mode='rotating';a.samples=SAMPLES
    if a.l2_bytes<0:p.error('invalid L2 size')
    for name in ('sdk','candidate','config_bundle','previous','controls','bundle'):setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.child:
        if a.fixture is None or a.variant is None or a.keys is None:p.error('child requires fixture/variant/keys')
        a.fixture=a.fixture.resolve(strict=True);return child(a)
    if a.fixtures is None or a.output is None:p.error('fixtures and output are required')
    a.fixtures=a.fixtures.resolve(strict=True);a.output=a.output.resolve();a.acu=a.acu or a.sdk/'asight/bin/acu'
    return run(a)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
