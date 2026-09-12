"""Two-shape reader experiment: 2 geometries x A/unit/header32 switches."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

from dev.gemv_cuda.build import digest,replace_once
from dev.gemv_ppu.config_space import Config,source as original_source,verify as verify_configs
from dev.gemv_ppu.access_pattern import warp_pattern,footprint

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='quactlize.q4-reader-reuse.v1'
SHAPES=((5120,8192),(8192,5120))
GEOMETRIES={(5120,8192):((4,8,4),(8,16,4)),(8192,5120):((8,10,4),(4,10,8))}
REVIEW=ROOT/'docs/measurements/q4_config_sweep_20260912/review.json'
NAMES=('unchanged','a-coop','unit-coop','a-unit-coop','header32','a-coop-header32','unit-coop-header32','all')


@dataclass(frozen=True)
class Reader:
    variant:int
    config:Config

    @property
    def key(self):return f'v{self.variant}-{self.config.key}'
    @property
    def args(self):return (self.variant,*self.config.args)
    @property
    def switches(self):return dict(a_cooperative=bool(self.variant&1),unit_cooperative=bool(self.variant&2),header32=bool(self.variant&4))


def inventory(n,k):return [Reader(v,Config(*c)) for c in GEOMETRIES[n,k] for v in range(8)]
def lookup(n,k,key):
    found=[r for r in inventory(n,k) if r.key==key]
    if len(found)!=1:raise ValueError('unknown reader/config: '+key)
    return found[0]


def selected(n,k):
    cases=json.loads(REVIEW.read_text())['cases']
    found=[r for r in cases if r['shape']==[1,n,k]]
    c=Config(*GEOMETRIES[n,k][0])
    if len(found)!=1 or found[0]['selected']!=c.key:raise ValueError('previous winning config differs')
    return c


def plan():
    return dict(schema=SCHEMA,m=1,qtype=12,cache='rotating',reference_limit_pct=0,
        scope='SAME_ARITHMETIC_READER_EXPERIMENT_NOT_PRODUCTION',cases=[dict(n=n,k=k,previous=selected(n,k).key,
            readers=[dict(key=r.key,name=NAMES[r.variant],switches=r.switches,geometry=r.config.geometry(n,k))
                     for r in inventory(n,k)]) for n,k in SHAPES])


def payload(n,k):
    if (n,k) not in SHAPES:raise ValueError('shape outside reader experiment')
    return f'libq4_reader_reuse_n{n}_k{k}.so'


def source(n,k):
    previous=original_source(n,k)
    body=previous[:previous.index(f'extern "C" int q4_config_run_{n}_{k}(')]
    start=body.index('template<int Columns,int Warps,int P,int N,int K,bool Early>')
    end=body.index('\n}\n',start)+3
    kernel=body[start:end]
    kernel=replace_once(kernel,'template<int Columns,int Warps,int P,int N,int K,bool Early>',
        'template<int Variant,int Columns,int Warps,int P,int N,int K>')
    kernel=replace_once(kernel,'__global__ void q4_group_affine(', '__global__ void q4_reader_reuse(')
    kernel=replace_once(kernel,'    constexpr int Workers=Warps*32/Columns,Pairs=P/2;',
        '''    constexpr bool Early=true;
    constexpr int Workers=Warps*32/Columns,Pairs=P/2;
    static_assert((Columns==4 || Columns==8) && Workers%8==0 && K%256==0);
    int const read_lane=threadIdx.x%32;''')
    old='''        uint4 metadata[Early ? P : 1];
        if constexpr(Early) {
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=aligned_unit(units_ptr+(size_t(g/8)*N+col+p)*16);
        }'''
    new='''        uint4 metadata[Early ? P : 1];
        if constexpr(Variant&2) {
            uint4 unit{};
            if(read_lane<Columns*P) unit=aligned_unit(units_ptr+(size_t(g/8)*N+blockIdx.x*(Columns*P)+read_lane)*16);
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=q4_cooperative_unit_read(unit,(read_lane%Columns)*P+p);
        } else {
            #pragma unroll
            for(int p=0;p<P;++p) metadata[p]=aligned_unit(units_ptr+(size_t(g/8)*N+col+p)*16);
        }
        uint4 a_chunk{};
        if constexpr(Variant&1) a_chunk=q4_cooperative_a_chunk<Columns>(a_ptr,g,read_lane);'''
    kernel=replace_once(kernel,old,new)
    kernel=replace_once(kernel,'                float4 av=aligned_activation<0>(a_ptr,g*32+slot*8+half*4);',
        '''                float4 av;
                if constexpr(Variant&1) av=q4_cooperative_a_read<Columns>(a_chunk,slot*8+half*4,read_lane);
                else av=aligned_activation<0>(a_ptr,g*32+slot*8+half*4);''')
    kernel=replace_once(kernel,'            float2 s0=q4_affine_header(u0,g&7),s1=q4_affine_header(u1,g&7);',
        '''            float2 s0,s1;
            if constexpr(Variant&4) {s0=q4_affine_header32(u0,g&7);s1=q4_affine_header32(u1,g&7);}
            else {s0=q4_affine_header(u0,g&7);s1=q4_affine_header(u1,g&7);}''')
    body=body[:end]+'\n'+(ROOT/'dev/gemv_ppu/reader_reuse.hpp').read_text()+'\n'+kernel+body[end:]
    body+=f'''extern "C" int q4_reader_run_{n}_{k}(int variant,int columns,int warps,int values,
        void const* a,void const* low,void const* units,void* out,void* stream) {{
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for r in inventory(n,k):
        c,w,p=r.config.args
        body+=f'''    if(variant=={r.variant} && columns=={c} && warps=={w} && values=={p}) {{
        q4_reader_reuse<{r.variant},{c},{w},{p},{n},{k}><<<{n//(c*p)},{w*32},0,static_cast<hggcStream_t>(stream)>>>(
            a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return body+'    return QKG_INVALID;\n}\n'


def access(r,n,k,bases=None):
    model=warp_pattern(r.config,n,k,base_mod128=bases)
    bases=model['base_mod128'];c,w,p=r.config.args;groups=model['lane_group']
    streams=model['streams']
    if r.variant&1:
        addresses=[bases['A']+g*64+(lane%c)*(64//c) for lane,g in enumerate(groups)]
        width=64//c
        streams['A_cooperative_chunk']=dict(lane_byte_addresses=addresses,width_bytes=width,
            granules={str(s):footprint(addresses,width,s) for s in (32,64,128)})
    if r.variant&2:
        addresses=[bases['metadata']+16*((groups[lane]//8)*n+lane) for lane in range(c*p)]
        streams['metadata_cooperative_unit']=dict(lane_byte_addresses=addresses,width_bytes=16,
            granules={str(s):footprint(addresses,16,s) for s in (32,64,128)})
    return dict(key=r.key,switches=r.switches,model='SOURCE_FOOTPRINT_NOT_MEASURED_TRAFFIC',first_warp=model,
        implemented_streams=['B_vector','metadata_cooperative_unit' if r.variant&2 else 'metadata_p0',
                             'A_cooperative_chunk' if r.variant&1 else 'A_half2'],
        logical_lane_bytes_per_call=dict(B=n*k//2,A=2*n*k//(c*p) if r.variant&1 else 2*n*k//p,
            metadata=n*k*c//64 if r.variant&2 else n*k//2),
        a_shuffle_source_steps=16 if r.variant&1 else 0,unit_shuffle_source_steps=4*p if r.variant&2 else 0)


def isa_statistics(text):
    pattern=re.compile(r'Disassembly of section \.text\.kernel\.[^\n]*q4_reader_reuseILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)E[^\n]*:')
    rows={}
    for match in pattern.finditer(text):
        end=text.find('Disassembly of section ',match.end())
        body=text[match.end():end if end>=0 else None]
        counts=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
        v,c,w,p,n,k=map(int,match.groups());key=f'n{n}-k{k}-v{v}-c{c}-w{w}-p{p}'
        rows[key]=dict(scope='STATIC_NATIVE_ISA_NOT_DYNAMIC_COUNTS',
            code_fastpath_present=bool(counts['v.lop3.b32'] and (counts['v.add.f16x2'] or counts['v.fma.f16x2'])),
            fp32_fma_present=any(op.startswith('v.fma.f32') for op in counts),
            operations={op:ct for op,ct in sorted(counts.items()) if op.startswith(('vmem.ld.','tsm.','s.cbr','s.blksyn')) or
                any(s in op for s in ('shuffle','shfl','shf.','shrl.b64','f16x2','fma.f32','lop3','perm','swzl'))},
            loop_caution='Compiler unrolling may differ: use ACU for dynamic load/shuffle counts')
    return rows


def verify(candidate,config_bundle,previous,controls,baseline,*,sources=True):
    old=verify_configs(config_bundle,previous,controls,baseline,sources=sources)
    m=json.loads((candidate/'manifest.json').read_text())
    if (m.get('schema')!=SCHEMA or m.get('plan')!=json.loads(json.dumps(plan())) or
        m.get('config_manifest_sha256')!=digest(config_bundle/'manifest.json') or
        m.get('review_sha256')!=digest(REVIEW) or m.get('compiler_sha256')!=old['compiler_sha256'] or
        set(m.get('payloads',{}))!={f'{n}x{k}' for n,k in SHAPES}):raise ValueError('reader package/control identity differs')
    for n,k in SHAPES:
        row=m['payloads'][f'{n}x{k}'];path=candidate/payload(n,k)
        if row.get('file')!=path.name or digest(path)!=row['sha256']:raise ValueError('reader payload hash differs or LFS pointer')
        with path.open('rb') as f:
            if f.read(4)!=b'\x7fELF':raise ValueError('reader payload is not ELF')
    if digest(candidate/'isa-stats.json')!=m['isa_sha256']:raise ValueError('reader ISA receipt differs')
    for name,sha in m['source_hashes'].items() if sources else []:
        path=(ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=sha:raise ValueError('reader source differs: '+name)
    return m
