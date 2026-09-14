"""Matched prefill requests, independently generated fixtures and result contracts."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from quactlize.runtime.compiler import sha, Compiler
from quactlize.runtime.tuning import digest
from reference import gguf_kpack as ref
from tools.kpack_bf16_fixture import as_float, row_domain
from tools.kpack_dequant_fixture import bf16
from tools.kpack_warmup_fixture import prepare_expert
from tools.run_kpack_gemv_gate import metadata_oracle
from tools.run_kpack_dequant_gate import DENSE, GROUPED

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'prebuilt/ppu0010/kpack-prefill-measure-v1'
BOARD = ROOT/'docs/measurements/kpack_prefill_components_20260914/result.json'
SCOPE = 'SELECTED_FULL_OUTPUT_INCLUDING_REDUCER_AND_INTERNAL_DIRECTORY_NO_SF_PREPASS'


def families():
    return [dict(q=q,n=n,k=k,experts=e,id=f'q{q}-n{n}-k{k}-e{e}')
            for q in (12,13) for n,k,e in
            [*( (n,k,1) for n,k in DENSE), *( (n,k,256) for n,k in GROUPED)]]


def requests():
    return [(w['q'],route,t if w['experts']==1 else 8*t,w['n'],w['k'],w['experts'],t)
            for w in families() for t in (2048,4096)
            for route in ((0,1) if w['experts']==1 else (2,3))]


def reducer_cases():
    # No qtype or K axis: these do not change the reduction operation.
    keys={(int(e>1),m,n,s) for q,r,m,n,k,e,t in requests() for s in (2,4,8)}
    return [dict(compact=c,m=m,n=n,split=s,id=f'{"grouped" if c else "dense"}-m{m}-n{n}-s{s}')
            for c,m,n,s in sorted(keys)]


def hashes(arrays):
    return {name:hashlib.sha256(np.ascontiguousarray(value).view('u1').reshape(-1)).hexdigest()
            for name,value in arrays.items()}


def verify(bundle, sdk):
    bundle=Path(bundle).resolve(strict=True);sdk=Path(sdk).resolve(strict=True)
    m=json.loads((bundle/'manifest.json').read_text())
    if m['schema']!='quactlize.prefill-measurement.v1':raise ValueError('wrong measurement bundle')
    for name,h in m['files'].items():
        p=(bundle/name).resolve(strict=True)
        if not p.is_relative_to(bundle) or sha(p)!=h:raise ValueError('payload changed: '+name)
    for name,h in m['runtime'].items():
        if sha(sdk/'lib'/name)!=h:raise ValueError('runtime changed: '+name)
    for name,h in m['policy_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('production policy changed; rebuild package: '+name)
    plan=json.loads((bundle/'plan.json').read_text())
    if [tuple(r['request']) for r in plan['requests']]!=requests():raise ValueError('request denominator differs')
    if any(r['status']!='SELECTED' for r in plan['requests']):raise ValueError('selected request missing')
    if {r['parent'] for r in plan['requests']}!={r['parent']['symbol'] for r in m['modules']}:
        raise ValueError('selected module closure differs')
    for r in m['modules']:
        source=Compiler.source(None,r['parent'],'')
        if digest(dict(identity=r['identity'],parent=r['parent'],source=source))!=r['key']:
            raise ValueError('module source receipt differs')
        if m['files'].get(r['path'])!=r['sha256']:raise ValueError('module file receipt differs')
    return m,plan


class Weights:
    """Same raw bytes as dequant/BF16; official GGUF FP32 dot, not a self-oracle."""
    def __init__(self,w,progress=None):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        q,n,k,e=w['q'],w['n'],w['k'],w['experts'];spec=ref.SPECS[q]
        self.categories=np.random.default_rng(60413+k).integers(0,4,k)
        self.sums=np.empty((e,4,n),dtype='f8');self.absolute=np.empty_like(self.sums)
        self.planes={};full_hash=hashlib.sha256();sf_hash=hashlib.sha256()
        # Keep scales separate; SF timing must never include their derivation.
        for expert in range(e):
            rng=np.random.default_rng(np.random.SeedSequence([935712,q,n,k,expert]))
            raw=rng.integers(0,256,(n*(k//256),spec.raw_bytes),dtype='u1')
            for off in (spec.d_offset,spec.dmin_offset):
                if off>=0:
                    d=(rng.random(raw.shape[0])*.02+.005).astype('<f2')
                    raw[:,off:off+2]=d.view('u1').reshape(-1,2)
            placed=prepare_expert(raw,q,n,k)
            sc,ze=metadata_oracle(placed['units'],q,n,k,1)
            placed['scale'],placed['zero']=sc[0],ze[0]
            if not self.planes:
                self.planes={name:np.empty((e,*value.shape),dtype=value.dtype) for name,value in placed.items()}
            for name in self.planes:self.planes[name][expert]=placed[name]
            official=dequantize(raw.reshape(-1),GGMLQuantizationType(q)).reshape(n,k)
            if not np.isfinite(official).all():raise ValueError('nonfinite GGUF fixture')
            full_hash.update(bf16(official).tobytes())
            for cat in range(4):
                part=official[:,self.categories==cat]
                self.sums[expert,cat]=part.sum(1,dtype='f8')
                self.absolute[expert,cat]=np.abs(part).sum(1,dtype='f8')
            if progress and (expert+1==e or (expert+1)%32==0):progress(expert+1,e)
        for name in ('scale','zero'):sf_hash.update(self.planes[name].tobytes())
        self.identity=dict(fixture_hashes=hashes({name:self.planes[name] for name in ('low','high','units')}),
                           bf16_golden_sha256=full_hash.hexdigest(),sf_golden_sha256=sf_hash.hexdigest())

    def activation(self,m):
        coeff=as_float(bf16(np.random.default_rng(76121+m).uniform(-.25,.25,(m,4)).astype('f4')))
        return np.ascontiguousarray(coeff[:,self.categories],dtype='<f2'),coeff.astype('f8')

    def error(self,out,coeff,indices):
        if out.shape!=(len(indices),self.sums.shape[-1]) or not np.isfinite(out).all():
            raise ValueError('nonfinite/incomplete GEMM output')
        worst=0.
        for e in range(len(self.sums)):
            idx=np.flatnonzero(indices==e)
            for start in range(0,len(idx),256):
                sel=idx[start:start+256];want=coeff[sel]@self.sums[e]
                denom=np.abs(coeff[sel])@self.absolute[e]
                err=np.abs(out[sel].astype('f8')-want)/np.maximum(denom,np.finfo('f8').tiny)
                worst=max(worst,float(err.max(initial=0)))
        return worst


def board_row(w,tokens):
    rows=json.loads(BOARD.read_text())['rows']
    matches=[r for r in rows if r['tokens']==tokens and
             all(r['workload'][key]==w[key] for key in ('q','n','k','experts'))]
    if len(matches)!=1:raise ValueError('component board point missing/duplicated')
    return matches[0]


def validate_fixture(w,tokens,identity,evidence):
    r=board_row(w,tokens)
    if identity!=evidence['weights'][w['id']]:raise ValueError('original packed/SF/BF16 fixture hashes differ')
    if r['golden_sha256']!=identity['bf16_golden_sha256']:raise ValueError('component board weight differs')
    return r


def validate_timing(r):
    values=r['samples_us']
    if len(values)!=15 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('need 3x5 finite positive measurements')
    if r['median_us']!=statistics.median(values):raise ValueError('wrong median')
    if r['round_medians_us']!=[statistics.median(values[i:i+5]) for i in (0,5,10)]:
        raise ValueError('wrong round medians')
    if r['timing']!='CAPTURED_COMPLETE_RING_EVENTS_FIRST_USE_EXCLUDED':raise ValueError('timing scope differs')


def validate_gemm(r,w,tokens,route):
    validate_timing(r)
    rows,_,rh=row_domain(tokens,w['experts'])
    if r['status']!='PASS' or r['workload']!=w or r['tokens']!=tokens or r['route']!=route:
        raise ValueError('result context differs')
    if r['scope']!=SCOPE or r['sf_prepass_timed'] or not r['reducer_included']:
        raise ValueError('producer-only/prepass-contaminated timing')
    if not math.isfinite(r['error']) or not 0<=r['error']<.005:raise ValueError('invalid numeric proof')
    if r['rows']!=rows.tolist() or r['routes_sha256']!=rh:raise ValueError('router differs')
    if r['max_rows_bound']!=tokens:raise ValueError('not the production row bound')
    if r['input_ring_bytes']<2.25*r['device']['l2_bytes'] or r['calls_per_graph']!=2*r['copies']:
        raise ValueError('incomplete cold ring')
    if any(r.get(k)!='PASS' for k in ('guards','zero_a','changed_a_graph','changed_a_sequence')):
        raise ValueError('missing correctness control')
    return r


def validate_reducer(r,w):
    validate_timing(r)
    count=w['m']*w['n'];size=count*w['split']*4
    if r['status']!='PASS' or r['workload']!=w or r['raw_bad']!=0 or r['negative_bad']<=0 or r['guards']!='PASS':
        raise ValueError('reducer context/numeric controls differ')
    if r['scope']!='REDUCER_ONLY_SYNTHETIC_PARTIALS_ROTATING_NOT_PRODUCER_CONSUMER_CACHE' or r['producer_timed']:
        raise ValueError('not an isolated reducer measurement')
    if r['partial_bytes']!=size or r['input_ring_bytes']!=size*r['copies'] or r['copies']<2 or r['calls_per_graph']!=2*r['copies']:
        raise ValueError('reducer ring geometry differs')
    if r['input_ring_bytes']<2.25*r['device']['l2_bytes']:raise ValueError('reducer ring fits L2')
    if (r['partial_layout'],r['output_dtype'],r['addition_order'])!=('FP32_S_M_N','FP16','INCREASING_S'):
        raise ValueError('reducer arithmetic contract differs')
    return r


def partial_values(start,count,s):
    # FP32-exact, mixed signs and S-dependent values; no cancellation-only oracle.
    i=np.arange(start,start+count,dtype='i8')
    return (((i*17+s*13)%257)-128).astype('<f4')/256+np.float32((s+1)/32)


def reducer_expected(start,count,split):
    total=np.zeros(count,dtype='<f4')
    for s in range(split):np.add(total,partial_values(start,count,s),out=total)
    return total.astype('<f2')


def bind_reducer(path):
    lib=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    specs={'prepare':([C.c_int]*4+[C.c_void_p,C.c_uint64,C.c_void_p,C.POINTER(C.c_void_p)],C.c_int),
           'run':([C.c_void_p,C.c_void_p],C.c_int),'fast':([C.c_void_p],C.c_int),
           'destroy':([C.c_void_p],None)}
    for name,(args,result) in specs.items():
        fn=getattr(lib,'prefill_reducer_'+name);fn.argtypes=args;fn.restype=result;setattr(lib,name,fn)
    return lib
