"""One remaining Q4 shape: exact CTA fold and a bounded warp neighborhood."""
from collections import Counter
from dataclasses import dataclass
import json
import re

from dev.gemv_cuda.build import digest, replace_once
from dev.gemv_ppu import small_latency as prior

ROOT=prior.ROOT
SCHEMA='quactlize.q4-medium-refine.v1'
SHAPES=((1024,5120),)
REFERENCE_LIMIT_PCT=5.0
PAYLOAD='libq4_medium_refine_n1024_k5120.so'
REVIEW=ROOT/'docs/measurements/q4_small_latency_20260912/review.json'
WARPS={2:(8,10,12,16,20),4:(10,16,20,24)}
ANCHOR_KEYS={'kpack-p2':'affine-h1-l1-a1-w10-c4-p2',
             'kpack-p4':'affine-h1-l0-a0-w20-c4-p4'}


@dataclass(frozen=True)
class Candidate:
    values:int
    warps:int
    reduction:int
    unsigned:int=0

    @property
    def key(self):return f'p{self.values}-w{self.warps}-r{self.reduction}-u{self.unsigned}'

    @property
    def recipe(self):return (self.values,self.warps,self.reduction,self.unsigned)

    @property
    def parent(self):
        mode=int(self.values==2)
        return prior.Candidate('affine',1,mode,mode,self.warps,4,self.values)

    @property
    def immutable_parent(self):
        return (self.values,self.warps) in ((2,10),(4,20))

    def geometry(self):
        out=self.parent.geometry(1024,5120)
        workers=self.warps*8
        out.update(last_pass_workers=160-(out['k_passes']-1)*workers,
                   k_workers=workers,whole_warp_k_groups=8,
                   cta_fold_rounds=(self.warps+32//out['tile_n']-1)//(32//out['tile_n']))
        return out


def inventory():
    rows=[Candidate(p,w,r) for p,warps in WARPS.items() for w in warps for r in (0,1)]
    rows += [Candidate(p,w,r,1) for p,w in ((2,10),(4,20)) for r in (0,1)]
    return rows


def lookup(key):
    rows=[c for c in inventory() if c.key==key]
    if len(rows)!=1:raise ValueError('unknown medium candidate: '+key)
    return rows[0]


def anchor(name):
    case=next(c for c in json.loads(REVIEW.read_text())['cases'] if c['shape']==[1,1024,5120])
    if case['selected']!=ANCHOR_KEYS['kpack-p2'] or not set(ANCHOR_KEYS.values()).issubset(case['median_us']):
        raise ValueError('medium immutable anchors changed')
    return prior.lookup(1024,5120,ANCHOR_KEYS[name])


def plan():
    return dict(schema=SCHEMA,qtype=12,m=1,n=1024,k=5120,mode='rotating',reference_limit_pct=REFERENCE_LIMIT_PCT,
        anchors={name:dict(key=anchor(name).key,recipe=anchor(name).recipe) for name in ANCHOR_KEYS},
        candidates=[dict(key=c.key,recipe=c.recipe,geometry=c.geometry(),
                         immutable_control_available=c.immutable_parent) for c in inventory()],
        prior_closed_shapes=[[512,2048],[4096,2048],[4096,4096],[5120,8192],[8192,5120]],
        pruning=['Only N1024/K5120, M1/Q4 cold weights; canonical bytes and S1 unchanged',
                 'Retain both nearly tied immutable P2/W10 and P4/W20 anchors',
                 'P2 uses H1/L1/A1; P4 H1/L0/A0; no duplicate A-load-source sweep',
                 'R0/R1 at nine nearby geometries; unsigned indices only at two anchor geometries',
                 'Exact per-thread addition/read order; no new tree, Split-K, A staging or AIU'])


def kernel_source():
    s=prior.affine_source()
    s=replace_once(s,'template<int Header,int Loading,int AMode,int Columns,int Warps,int P,int N,int K>',
        'template<int Reduction,int Unsigned,int Header,int Loading,int AMode,int Columns,int Warps,int P,int N,int K>')
    s=replace_once(s,'__global__ void q4_affine_pipeline(', '__global__ void q4_medium_refined(')
    s=replace_once(s,'    int const tid=threadIdx.x,worker=tid/Columns,col=blockIdx.x*(Columns*P)+(tid%Columns)*P;',
        '''    using Index=typename std::conditional<Unsigned!=0,unsigned,int>::type;
    Index const tid=threadIdx.x,worker=tid/Columns,col=blockIdx.x*(Columns*P)+(tid%Columns)*P;''')
    s=replace_once(s,'for(int pass=0;pass<(K/32+Workers-1)/Workers;++pass)',
        'for(Index pass=0;pass<(K/32+Workers-1)/Workers;++pass)')
    s=replace_once(s,'        int const g=pass*Workers+worker;','        Index const g=pass*Workers+worker;')
    old='''        #pragma unroll
        for(int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];'''
    return replace_once(s,old,'''        if constexpr(Reduction==0) {
'''+old+'''
        } else {
            q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));
        }''')


def source():
    s=prior.source(1024,5120,'affine')
    s='#include <type_traits>\n'+s[:s.index('extern "C" int q4_latency_run_1024_5120(')]
    s+='\nnamespace QKG_CONCAT(kpack_q,QKG_QTYPE) {\n'
    s+=(ROOT/'dev/gemv_ppu/medium_reduce.hpp').read_text()+'\n'+kernel_source()+'\n}\n'
    s+='''extern "C" int q4_medium_run(int control,int values,int warps,int reduction,int unsigned_index,
        void const* a,void const* low,void const* units,void* output,void* stream) {
    if(!a || !low || !units || !output || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(output)&3) || (control!=0 && control!=1)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for c in inventory():
        p,w,r,u=c.recipe;mode=int(p==2);grid=1024//(4*p)
        args='a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(output)'
        launch=f'<<<{grid},{w*32},0,static_cast<hggcStream_t>(stream)>>>'
        s+=f'''    if(values=={p} && warps=={w} && reduction=={r} && unsigned_index=={u}) {{
        if(control) q4_affine_pipeline<1,{mode},{mode},4,{w},{p},1024,5120>{launch}({args});
        else q4_medium_refined<{r},{u},1,{mode},{mode},4,{w},{p},1024,5120>{launch}({args});
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return s+'    return QKG_INVALID;\n}\n'


def access(c,bases=None):
    model=prior.access(c.parent,1024,5120,bases)
    model.update(key=c.key,geometry=c.geometry(),
        cta_fold='SAME_ORDER_CONSTANT_ROUNDS' if c.reduction else 'ORIGINAL_RUNTIME_START',
        unsigned_index=bool(c.unsigned))
    return model


def isa(text):
    sections=list(re.finditer(r'Disassembly of section \.text\.kernel\.[^\n]+:',text));rows={}
    for i,m in enumerate(sections):
        if 'q4_medium_refinedI' not in m[0]:continue
        r,u,h,l,a,col,w,p,n,k=map(int,re.findall(r'(?:I|E)Li(\d+)',m[0]))
        assert (h,l,a,col,n,k)==(1,int(p==2),int(p==2),4,1024,5120)
        c=lookup(Candidate(p,w,r,u).key)
        body=text[m.end():sections[i+1].start() if i+1<len(sections) else None]
        ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
        before,sep,after=body.partition('s.blksyn.defer')
        if not sep:raise ValueError('CTA barrier missing')
        tail=Counter(re.findall(r'\t([a-z][\w.]+)\s',after))
        rows[c.key]=dict(scope='STATIC_NATIVE_ISA_NOT_DYNAMIC_COUNTS',
            code_fastpath_present=bool(ops['v.lop3.b32'] and any('f16x2' in op for op in ops)),
            fp32_fma_present=any(op.startswith('v.fma.f32') for op in ops),
            operations={op:count for op,count in ops.items() if op.startswith(('vmem.','tsm.','s.cbr','s.blksyn')) or
                any(x in op for x in ('shuffle','f16x2','fma.f32','lop3'))},
            cta_fold_shared_load_instructions=sum(count for op,count in tail.items() if op.startswith('tsm.ld.')),
            cta_fold_conditional_branches=sum(count for op,count in tail.items() if op.startswith('s.cbr.az')),
            cta_fold_native=after.splitlines())
    return rows


def verify(candidate,latency,followup,reuse,config,previous,controls,bundle,*,sources=True):
    old=prior.verify(latency,followup,reuse,config,previous,controls,bundle,sources=sources)
    m=json.loads((candidate/'manifest.json').read_text())
    if (m.get('schema')!=SCHEMA or m.get('plan')!=json.loads(json.dumps(plan())) or
        m.get('latency_manifest_sha256')!=digest(latency/'manifest.json') or
        m.get('review_sha256')!=digest(REVIEW) or m.get('compiler_sha256')!=old['compiler_sha256'] or
        set(m.get('payloads',{}))!={PAYLOAD}):raise ValueError('medium refinement/control identity differs')
    image=candidate/PAYLOAD
    if digest(image)!=m['payloads'][PAYLOAD]['sha256'] or image.read_bytes()[:4]!=b'\x7fELF':raise ValueError('medium image differs')
    if digest(candidate/'isa-stats.json')!=m['isa_sha256']:raise ValueError('medium ISA receipt differs')
    for name,sha in m['source_hashes'].items() if sources else []:
        path=(ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=sha:raise ValueError('medium source differs: '+name)
    return m
