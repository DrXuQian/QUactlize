#!/usr/bin/env python3
"""Two-shape cold latency experiment with immutable per-shape winners."""
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
from dev.gemv_ppu import small_latency as spec
from dev.gemv_ppu.run_reader_followup import FollowupBench,hash_ok
from dev.gemv_ppu.run_config_sweep import execute,PREFIX,FAIL_PREFIX,shortlist
from dev.gemv_ppu.run_cold_shapes import recipe as control_recipe
from dev.gemv_ppu.run_cold_geometry import probe_device
from dev.gemv_ppu.run_h800_port import PortBench
from dev.gemv_ppu.run_bload import ExperimentBench,exact_bits
from dev.gemv_ppu.campaign import parse_cells
from dev.gemv_ppu.run import checked
from tools.profile_kpack_gpu_compact import acu_launch_command

ANCHORS=('xplane','raw-reference','baseline')
ROUNDS,SCREEN_SAMPLES,CONFIRM_SAMPLES,TOP_K=6,5,15,3


def recipe(variant,n,k,key):
    if variant=='reader':return spec.lookup(n,k,key).recipe
    if variant not in ANCHORS or key!='control':raise ValueError('unknown small-latency arm')
    return spec.baseline(n,k).recipe if variant=='baseline' else control_recipe(variant,n,k)


def arithmetic(variant,n,k,key):
    if variant=='reader':return spec.lookup(n,k,key).arithmetic
    return spec.baseline(n,k).arithmetic if variant=='baseline' else 'PER_WEIGHT_FP16'


def immutable_match_required(c,n,k):
    return c.family=='affine' or (c.family=='meta' and (n,k)==(512,2048))


class LatencyBench(FollowupBench):
    def __init__(self,a):
        super().__init__(SimpleNamespace(**(vars(a)|dict(candidate=a.followup))))
        self.latency_allowed={c.recipe:c for c in spec.inventory(self.n,self.k)}
        self.latency_libraries,self.latency_launch={},{}
        for family in sorted({c.family for c in self.latency_allowed.values()}):
            lib=C.CDLL(str(a.candidate/spec.payload(self.n,self.k,family)),mode=C.RTLD_LOCAL)
            probe=lib.q4_ppu_probe;probe.argtypes=[C.POINTER(C.c_int)]*3+[C.c_char_p];probe.restype=C.c_int
            l2,sm,warp,name=C.c_int(),C.c_int(),C.c_int(),C.create_string_buffer(256)
            checked(probe(C.byref(l2),C.byref(sm),C.byref(warp),name),'latency image marker')
            if (sm.value,warp.value,name.value.decode())!=(self.device['sm'],self.device['warp'],self.device['name']):
                raise ValueError('latency image device differs')
            fn=getattr(lib,f'q4_latency_run_{self.n}_{self.k}')
            fn.argtypes=[C.c_int]*7+[C.c_void_p]*5;fn.restype=C.c_int
            self.latency_libraries[family],self.latency_launch[family]=lib,fn
        self.matched,self.immutable_match=None,None

    def invoke(self,cfg,index=0,force_aiu=None):
        if force_aiu is not None or tuple(cfg) not in self.latency_allowed:
            raise ValueError('latency config not compiled')
        c=self.latency_allowed[tuple(cfg)];low,units=self.weight_pointers[index%self.copies]
        return self.latency_launch[c.family](0,*cfg[1:],self.a,low,units,self.output,self.r.stream)

    def direct(self,cfg,force_aiu=None):
        c=self.latency_allowed[tuple(cfg)];low,units=self.weight_pointers[0]
        self.immutable_match=None
        self.r.fill(self.output_base,0xff,self.output_bytes+32)
        checked(self.latency_launch[c.family](1,*cfg[1:],self.a,low,units,self.output,self.r.stream),'untouched same-geometry body')
        err=self.error()
        if not math.isfinite(err) or err>=.005:raise ValueError('same-geometry control failed GGUF oracle')
        old=self.sdk.download(self.output,self.output_bytes)
        if immutable_match_required(c,self.n,self.k):
            self.r.fill(self.output_base,0xff,self.output_bytes+32)
            if c.family=='affine':
                rc=self.followup_launch['affine'](4,c.columns,c.warps,c.values,self.a,low,units,self.output,self.r.stream)
            else:
                rc=self.followup_launch['small'](0,8,c.warps,self.a,low,units,self.output,self.r.stream)
            checked(rc,'immutable previous binary')
            older=self.error()
            if not math.isfinite(older) or older>=.005:raise ValueError('immutable control failed GGUF oracle')
            self.immutable_match=exact_bits(old,self.sdk.download(self.output,self.output_bytes),'rebuilt control vs immutable binary')
            err=max(err,older)
        new,error=ExperimentBench.direct(self,cfg,force_aiu=force_aiu)
        self.matched=exact_bits(old,new,'latency vs unchanged same-geometry FP32')
        return new,max(err,error)


def child(a):
    spec.verify(a.candidate,a.followup,a.reuse,a.config_bundle,a.previous,a.controls,a.bundle,sources=False)
    keys=json.loads(a.keys)
    if not keys or len(keys)!=len(set(keys)):raise ValueError('empty/duplicate child keys')
    bench=None
    try:
        if a.variant=='reader':bench=LatencyBench(a)
        elif a.variant=='baseline':bench=FollowupBench(SimpleNamespace(**(vars(a)|dict(candidate=a.followup))))
        else:bench=PortBench(SimpleNamespace(**(vars(a)|dict(candidate=a.controls))))
        n,k=bench.n,bench.k
        if (n,k) not in spec.SHAPES:raise ValueError('shape outside remaining two')
        for key in keys:
            try:
                row=bench.measure(list(recipe(a.variant,n,k,key)))
                row.update(arm=a.variant,variant=a.variant,config_key=key,phase=a.phase,
                    output_type='F32',inter_cta_split=1,launches_per_call=1,
                    timing_scope='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT',
                    cache_scope='ACU_FORCED_COLD' if a.profile else 'ROTATING_AT_LEAST_2_25_L2',
                    weight_arithmetic=arithmetic(a.variant,n,k,key))
                if a.variant=='reader':
                    c=spec.lookup(n,k,key)
                    row.update(family=c.family,geometry=c.geometry(n,k),code_dequant='LOP3_HALF2',
                        matched_fp32_sha256=bench.matched,immutable_control_sha256=bench.immutable_match,
                        same_geometry_control='UNCHANGED_BODY_IDENTICAL_DOT_REDUCTION',
                        access_models=[spec.access(c,n,k,dict(A=aa,B=bb,metadata=mm))
                            for aa,bb,mm in sorted({(int(bench.a)%128,int(b)%128,int(m)%128) for b,m in bench.weight_pointers})])
                print(PREFIX+json.dumps(row),flush=True)
            except Exception as exc:
                print(FAIL_PREFIX+json.dumps(dict(variant=a.variant,key=key,shape=[1,n,k],phase=a.phase,error=str(exc))),flush=True)
                traceback.print_exc();return 1
        return 0
    finally:
        if bench:bench.close()


def parse_row(raw,variant,n,k,key,phase,samples,profile=False):
    row=parse_cells('Q4_PPU_CELL '+json.dumps(raw),variant,[recipe(variant,n,k,key)],[1,n,k],'rotating',samples)[0]
    if (row.get('variant')!=variant or row.get('config_key')!=key or row.get('phase')!=phase or
        row.get('zero_a_check')!='PASS' or row.get('output_type')!='F32' or row.get('inter_cta_split')!=1 or
        row.get('launches_per_call')!=1 or row.get('timing_scope')!='RESIDENT_FULL_CALL_NO_HOST_PACK_NO_JIT' or
        row.get('cache_scope')!=('ACU_FORCED_COLD' if profile else 'ROTATING_AT_LEAST_2_25_L2') or
        row.get('weight_arithmetic')!=arithmetic(variant,n,k,key) or
        row.get('storage')!=('XPLANE' if variant=='xplane' else 'RAW_GGUF' if variant=='raw-reference' else 'CANONICAL_KPACK4')):
        raise ValueError('small-latency identity/precision/cache differs')
    if (profile and row['median_us'] is not None) or (not profile and not math.isfinite(row['median_us'])):
        raise ValueError('invalid event/profiler sample')
    if variant=='reader':
        c=spec.lookup(n,k,key)
        if (row.get('family')!=c.family or row.get('geometry')!=c.geometry(n,k) or
            row.get('same_geometry_control')!='UNCHANGED_BODY_IDENTICAL_DOT_REDUCTION' or
            row.get('code_dequant')!='LOP3_HALF2' or not hash_ok(row.get('matched_fp32_sha256'))):
            raise ValueError('reader mapping or unchanged-control proof differs')
        immutable=immutable_match_required(c,n,k)
        if (immutable and not hash_ok(row.get('immutable_control_sha256'))) or (not immutable and row.get('immutable_control_sha256') is not None):
            raise ValueError('immutable control proof differs')
        models,seen=row.get('access_models',[]),set()
        if not models:raise ValueError('missing observed load patterns')
        for model in models:
            bases=model['base_mod128'];identity=tuple(bases[x] for x in ('A','B','metadata'))
            if (model!=spec.access(c,n,k,bases) or identity in seen or
                any(type(x) is not int or not 0<=x<128 or x%16 for x in identity)):
                raise ValueError('observed address/footprint differs')
            seen.add(identity)
    return row

def summarize(n, k, screen, records, selected):
    med = {key:statistics.median(r['median_us'] for r in rows) for key,rows in records.items() if len(rows) == ROUNDS}
    complete = set(screen) == {c.key for c in spec.inventory(n,k)} and set(med) == {*ANCHORS,*selected} and len(selected) == TOP_K
    available = [key for key in ('baseline',*selected) if key in med]
    best = min(available,key=med.get) if available else None
    result = dict(shape=[1,n,k], status='PASS' if complete else 'INCOMPLETE', median_us=med, selected=best,
                  reference_limit_pct=spec.REFERENCE_LIMIT_PCT, reference_verdict='INCOMPLETE')
    if complete:
        delta = 100*(med[best]/med['raw-reference']-1)
        result.update(reference_delta_pct=delta, previous_delta_pct=100*(med[best]/med['baseline']-1),
            xplane_delta_pct=100*(med[best]/med['xplane']-1),
            reference_verdict='WITHIN_5_PERCENT' if med[best] <= med['raw-reference']*1.05 else 'PARITY_OPEN',
            paired_reference_delta_pct=[100*(r['median_us']/ref['median_us']-1)
                for r,ref in zip(records[best],records['raw-reference'])])
    return result


def run(a):
    m = spec.verify(a.candidate, a.followup, a.reuse, a.config_bundle, a.previous, a.controls, a.bundle)
    device = probe_device(a)  # Exiting child owns the probe context, never the orchestrator.
    names = set(m['source_hashes']) | {'dev/gemv_ppu/run_reader_followup.py', 'dev/gemv_ppu/run_small_latency.py', 'dev/gemv_ppu/run_config_sweep.py',
        'dev/gemv_ppu/run_cold_shapes.py', 'dev/gemv_ppu/run_cold_geometry.py', 'dev/gemv_ppu/run_bload.py',
        'dev/gemv_ppu/run_h800_port.py', 'dev/gemv_ppu/campaign.py', 'tools/profile_kpack_gpu_compact.py',
        'tools/run_kpack_grouped_decode_probe.py', 'tools/run_kpack_gemv_gate.py'}
    authority = dict(schema='quactlize.q4-small-latency-run.v1', candidate=digest(a.candidate/'manifest.json'),
        device=device, rounds=ROUNDS, screen_samples=SCREEN_SAMPLES, confirm_samples=CONFIRM_SAMPLES, top_k=TOP_K,
        sources={name:digest(ROOT/name) for name in sorted(names)},
        fixtures={f'{n}x{k}':digest(a.fixtures/f'q12-n{n}-k{k}-e1-c1.npz') for n,k in spec.SHAPES},
        runtime={name:digest(a.sdk/'lib'/name) for name in m['runtime']})
    a.output.mkdir(parents=True, exist_ok=True)
    receipt = a.output/'authority.json'
    if receipt.exists() and json.loads(receipt.read_text()) != authority:
        raise ValueError('resume source/device/runtime/fixture identity differs')
    receipt.write_text(json.dumps(authority,indent=2)+'\n')
    for name in ('manifest.json','isa-stats.json'):
        (a.output/('build-manifest.json' if name == 'manifest.json' else name)).write_bytes((a.candidate/name).read_bytes())
    (a.output/'plan.json').write_text(json.dumps(spec.plan(),indent=2)+'\n')
    diff = [name for name,sha in authority['runtime'].items() if sha != m['runtime'][name]]
    if diff:
        print('Q4_SMALL_SDK_DIFFERENCE recorded='+','.join(diff)+' real_marker_and_numeric_required=1',flush=True)
    result = dict(status='RUNNING', cases=[], profiles=[], failures=[], production_changed=False,
                  prior_closed_shapes=spec.plan()['prior_closed_shapes'], prior_result_scope='HISTORICAL_NOT_REMEASURED')
    started = time.monotonic()

    def save():
        (a.output/'summary.json').write_text(json.dumps(result)+'\n')

    def batch(variant,n,k,keys,phase,profile=False):
        receipt = a.output/f'n{n}-k{k}-{variant}-{phase}.json'
        cache = json.loads(receipt.read_text()) if receipt.exists() else {}
        samples = 0 if profile else SCREEN_SAMPLES if phase == 'screen' else CONFIRM_SAMPLES
        parsed = {}
        for key, entry in cache.items():
            log = a.output/entry['log']
            if log.parent != a.output:
                raise ValueError('cached log path differs')
            if log not in parsed:
                parsed[log] = (digest(log), [json.loads(s[len(PREFIX):]) for s in log.read_text().splitlines() if s.startswith(PREFIX)])
            sha, rows = parsed[log]
            found = [r for r in rows if r.get('config_key') == key]
            if sha != entry['log_sha256'] or len(found) != 1:
                raise ValueError('cached log hash/cell differs')
            row = parse_row(found[0],variant,n,k,key,phase,samples,profile)
            if row != entry['row'] or row['device'] != device:
                raise ValueError('cached row/device differs')
            if profile:
                report = a.output/entry['report']
                if report.parent != a.output or digest(report) != entry['report_sha256']:
                    raise ValueError('cached ACU differs')
        missing = [key for key in keys if key not in cache]
        while missing:
            prefix = a.output/f'n{n}-k{k}-{variant}-{phase}.{time.time_ns()}'
            log = prefix.with_name(prefix.name+('.acu.log' if profile else '.log'))
            cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--child','--sdk',str(a.sdk),
                '--candidate',str(a.candidate),'--followup',str(a.followup),'--reuse',str(a.reuse),'--config-bundle',str(a.config_bundle),
                '--previous',str(a.previous),'--controls',str(a.controls),'--bundle',str(a.bundle),
                '--fixture',str(a.fixtures/f'q12-n{n}-k{k}-e1-c1.npz'), '--variant',variant,
                '--keys',json.dumps(missing),'--phase',phase,'--l2-bytes',str(a.l2_bytes)]
            if profile:
                cmd = acu_launch_command(a.acu,prefix,cmd+['--profile'])
            print(f'Q4_SMALL_PROGRESS shape=1x{n}x{k} arm={variant} phase={phase} cells={len(missing)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
            with log.open('x') as stream:
                rc = execute(cmd,stream)
            consumed, failed = set(), set()
            for line in log.read_text().splitlines():
                if line.startswith(PREFIX):
                    raw = json.loads(line[len(PREFIX):]); key = raw.get('config_key')
                    if key not in missing or key in consumed:
                        raise ValueError('unexpected/duplicate child key')
                    consumed.add(key)
                    try:
                        row = parse_row(raw,variant,n,k,key,phase,samples,profile)
                    except Exception as exc:
                        failed.add(key)
                        result['failures'].append(dict(shape=[1,n,k],variant=variant,key=key,phase=phase,error=str(exc),log=log.name))
                        continue
                    if row['device'] != device:
                        raise ValueError('device changed')
                    entry = dict(row=row,log=log.name,log_sha256=digest(log))
                    if profile:
                        reports = list(a.output.glob(prefix.name+'*.acurep'))
                        if len(reports) != 1:
                            raise ValueError('ACU report missing')
                        entry.update(report=reports[0].name,report_sha256=digest(reports[0]))
                    cache[key] = entry
                elif line.startswith(FAIL_PREFIX):
                    error = json.loads(line[len(FAIL_PREFIX):]); key = error['key']
                    if key not in missing or key in consumed or error['variant'] != variant or error['phase'] != phase or error['shape'] != [1,n,k]:
                        raise ValueError('failure identity differs')
                    consumed.add(key); failed.add(key)
                    result['failures'].append(error | dict(log=log.name))
            if not consumed or (rc and not failed):
                raise ValueError(f'child infrastructure failure rc={rc}; log={log}')
            receipt.write_text(json.dumps(cache)+'\n')
            missing = [key for key in missing if key not in consumed]
            if missing:
                print(f'Q4_SMALL_RESTART remaining={len(missing)} failed={len(failed)} valid={len(cache)}',flush=True)
        return {key:cache[key]['row'] for key in keys if key in cache}

    for n,k in spec.SHAPES:
        case = dict(shape=[1,n,k],screen={},screen_anchors={},records={})
        result['cases'].append(case)
        for variant in (*ANCHORS,'reader'):
            keys = [c.key for c in spec.inventory(n,k)] if variant == 'reader' else ['control']
            try:
                rows = batch(variant,n,k,keys,'screen')
                if variant == 'reader':
                    case['screen'] = rows
                else:
                    case['screen_anchors'][variant] = rows['control']
            except Exception as exc:
                result['failures'].append(dict(shape=[1,n,k],variant=variant,phase='screen',error=str(exc)))
            save()
        selected = shortlist(case['screen'],TOP_K)
        case.update(shortlist=selected, records={key:[] for key in (*ANCHORS,*selected)})
        for turn in range(ROUNDS):
            arms = (*ANCHORS,'reader')
            for variant in arms if turn%2 == 0 else arms[::-1]:
                keys = selected if variant == 'reader' else ['control']
                if not keys:
                    continue
                try:
                    rows = batch(variant,n,k,keys if turn%2 == 0 else keys[::-1],f'r{turn}')
                    for key,row in rows.items():
                        case['records'][key if variant == 'reader' else variant].append(row)
                except Exception as exc:
                    result['failures'].append(dict(shape=[1,n,k],variant=variant,phase=f'r{turn}',error=str(exc)))
                save()
        case['comparison'] = summarize(n,k,case['screen'],case['records'],selected)
        print('Q4_SMALL_RESULT '+json.dumps(case['comparison']),flush=True)
        if not a.skip_acu:
            med = case['comparison']['median_us']
            profile_keys = sorted((key for key in selected if key in med),key=med.get)[:2]
            for variant,key in [*(('reader',key) for key in profile_keys),*((v,'control') for v in ANCHORS)]:
                try:
                    rows = batch(variant,n,k,[key],'profile',True)
                    if key not in rows:
                        raise ValueError('profiled reader failed')
                    result['profiles'].append(dict(shape=[1,n,k],variant=variant,key=key,status='PASS',row=rows[key]))
                except Exception as exc:
                    result['profiles'].append(dict(shape=[1,n,k],variant=variant,key=key,status='FAIL',error=str(exc)))
                save()
    spec.verify(a.candidate,a.followup,a.reuse,a.config_bundle,a.previous,a.controls,a.bundle)
    if (any(digest(ROOT/name) != sha for name,sha in authority['sources'].items()) or
        any(digest(a.sdk/'lib'/name) != sha for name,sha in authority['runtime'].items()) or
        any(digest(a.fixtures/f'q12-n{n}-k{k}-e1-c1.npz') != authority['fixtures'][f'{n}x{k}'] for n,k in spec.SHAPES)):
        raise ValueError('authority changed during run')
    ok = not result['failures'] and all(c['comparison']['status'] == 'PASS' for c in result['cases']) and all(p['status'] == 'PASS' for p in result['profiles'])
    result.update(status='PASS' if ok else 'INCOMPLETE',seconds=time.monotonic()-started,
                  within_5pct=sum(c['comparison']['reference_verdict'] == 'WITHIN_5_PERCENT' for c in result['cases']))
    save()
    with (a.output/'summary.tsv').open('w') as stream:
        writer = csv.writer(stream,delimiter='\t')
        writer.writerow(['N','K','selected','best_us','ref_us','vs_ref_pct','verdict'])
        for case in result['cases']:
            r = case['comparison']; med = r['median_us']
            writer.writerow([*case['shape'][1:],r['selected'],med.get(r['selected'],'NA'),med.get('raw-reference','NA'),r.get('reference_delta_pct','NA'),r['reference_verdict']])
    print(f'Q4_SMALL_COMPLETE status={result["status"]} within_5pct={result["within_5pct"]}/2 prior_closed=4/4 results={a.output}',flush=True)
    return 0 if ok else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',type=Path,required=True)
    for name,folder in [('candidate','q4-small-latency-v1'),('followup','q4-reader-followup-v1'),
                        ('reuse','q4-reader-reuse-v1'),('config-bundle','q4-config-sweep-v1'),
                        ('previous','q4-cold-shapes-v1'),('controls','q4-h800-port-v1'),('bundle','q4-simt-ab-v1')]:
        p.add_argument('--'+name,type=Path,default=ROOT/'prebuilt/ppu0010'/folder)
    p.add_argument('--fixtures',type=Path);p.add_argument('--fixture',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--variant',choices=(*ANCHORS,'reader'));p.add_argument('--keys');p.add_argument('--phase',default='screen')
    p.add_argument('--l2-bytes',type=int,default=0);p.add_argument('--child',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--skip-acu',action='store_true');p.add_argument('--acu',type=Path)
    a=p.parse_args();a.mode='rotating';a.samples=SCREEN_SAMPLES if a.phase=='screen' else CONFIRM_SAMPLES
    if a.l2_bytes<0:p.error('invalid L2 size')
    for name in ('sdk','candidate','followup','reuse','config_bundle','previous','controls','bundle'):
        setattr(a,name,getattr(a,name).resolve(strict=True))
    if a.child:
        if a.fixture is None or a.variant is None or a.keys is None:p.error('child requires fixture/variant/keys')
        a.fixture=a.fixture.resolve(strict=True);return child(a)
    if a.fixtures is None or a.output is None:p.error('fixtures and output are required')
    a.fixtures=a.fixtures.resolve(strict=True);a.output=a.output.resolve();a.acu=a.acu or a.sdk/'asight/bin/acu'
    return run(a)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:traceback.print_exc();raise SystemExit(1)
