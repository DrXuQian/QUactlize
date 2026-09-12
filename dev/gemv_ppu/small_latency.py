"""Two remaining Q4 shapes: issue scheduling, metadata select and small-N ownership."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

from dev.gemv_cuda.build import digest, replace_once
from dev.gemv_ppu import reader_followup as prior
from dev.gemv_ppu.access_pattern import footprint

ROOT = prior.ROOT
SCHEMA = 'quactlize.q4-small-latency.v1'
REFERENCE_LIMIT_PCT = 5.0
SHAPES = ((512,2048),(1024,5120))
META_WARPS = {(512,2048):(8,16),(1024,5120):(10,20)}
RESIDUE_WARPS = {(512,2048):(4,8),(1024,5120):(10,20)}
AFFINE = ((4,20,4),(4,10,2))
FAMILY_ID = {'meta':0,'residue2':1,'affine':2}
REVIEW = ROOT/'docs/measurements/q4_reader_followup_20260912/review.json'


@dataclass(frozen=True)
class Candidate:
    family: str
    header: int
    loading: int
    a_mode: int
    warps: int
    columns: int = 0
    values: int = 0

    @property
    def key(self):
        base = f'{self.family}-h{self.header}-l{self.loading}-a{self.a_mode}-w{self.warps}'
        return base+(f'-c{self.columns}-p{self.values}' if self.family == 'affine' else '')

    @property
    def recipe(self):
        return (FAMILY_ID[self.family],self.header,self.loading,self.a_mode,self.warps,self.columns,self.values)

    @property
    def arithmetic(self):
        return 'FP32_GROUP_AFFINE' if self.family == 'affine' else 'PER_WEIGHT_FP16'

    def geometry(self,n,k):
        tile = self.columns*self.values if self.family == 'affine' else 8 if self.family == 'meta' else 4
        chunk = 32*self.warps*32//self.columns if self.family == 'affine' else (128 if self.family == 'meta' else 256)*self.warps
        return dict(grid=n//tile,threads=self.warps*32,tile_n=tile,k_per_pass=chunk,
            k_passes=(k+chunk-1)//chunk,shared_a_bytes=k*2 if self.a_mode == 2 else 0,
            reduction_bytes=self.warps*tile*4,inter_cta_split=1)


def inventory(n,k):
    rows = [Candidate('meta',h,l,a,w) for w in META_WARPS[n,k] for h in (0,1) for l in (0,1) for a in (0,1,2)]
    rows += [Candidate('residue2',h,l,a,w) for w in RESIDUE_WARPS[n,k]
             for h,l,a in ((0,0,0),(1,0,0),(1,1,0),(1,1,2))]
    if (n,k) == (1024,5120):
        rows += [Candidate('affine',h,l,a,w,c,p) for c,w,p in AFFINE for h in (0,1)
                 for l,a in ((0,0),(1,0),(1,1),(1,2))]
    return rows


def lookup(n,k,key):
    rows = [r for r in inventory(n,k) if r.key == key]
    if len(rows) != 1:
        raise ValueError('unknown small-latency candidate: '+key)
    return rows[0]


def baseline(n,k):
    key = 'meta-h0-n8-w16' if (n,k) == (512,2048) else 'v4-c4-w20-p4'
    rows = json.loads(REVIEW.read_text())['cases']
    r = next(r for r in rows if r['shape'] == [1,n,k])
    if r['selected'] != key:
        raise ValueError('latest small-shape winner changed')
    return prior.lookup(n,k,key)


def plan():
    return dict(schema=SCHEMA,shapes=[dict(n=n,k=k,baseline=baseline(n,k).key,
        candidates=[dict(key=r.key,recipe=r.recipe,arithmetic=r.arithmetic,geometry=r.geometry(n,k)) for r in inventory(n,k)]) for n,k in SHAPES],
        qtype=12,m=1,mode='rotating',reference_limit_pct=5.0,
        prior_closed_shapes=[[4096,2048],[4096,4096],[5120,8192],[8192,5120]],
        scope='BOUNDED_SMALL_READER_EXPERIMENT_NOT_PRODUCTION',
        pruning=['Only two unresolved shapes; no inter-CTA split or new format',
                 'Retain old Width8 and affine winners; compare smaller Width4 residue reader',
                 'Restrict A/header/issue factors to selected one/two-pass geometries'])


def payload(n,k,family):
    if family not in {c.family for c in inventory(n,k)}:
        raise ValueError('unknown latency payload')
    return f'libq4_small_latency_{family}_n{n}_k{k}.so'


def original_kernel(family):
    if family == 'meta':
        s=prior.small_source();start=s.index('template<int Width, int Warps, bool StageA')
    elif family == 'affine':
        s=prior.source(1024,5120);start=s.index('template<int Variant,int Columns,int Warps,int P,int N,int K>')
    else:
        s=(ROOT/'dev/gemv_cuda/q4_cooperative_residue2.cuh').read_text()
        start=s.index('template<int Warps,int N,int K,bool Affine>')
        s=s.replace('qkg_call_v1 c','void const* a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* out_ptr')
        for old,new in (('c.a','a_ptr'),('c.low','low_ptr'),('c.units','units_ptr'),('c.output','out_ptr')):
            s=s.replace(old,new)
    return s[start:s.index('\n}\n',start)+3]


def residue_source():
    s=original_kernel('residue2')
    s=replace_once(s,'template<int Warps,int N,int K,bool Affine>',
        'template<int Header,int Loading,int AMode,int Warps,int N,int K>')
    s=replace_once(s,'__global__ void q4_cooperative_residue2(', '__global__ void q4_residue_pipeline(')
    s=replace_once(s,'    float total=0;', '''    constexpr bool Affine=false;
    auto act=static_cast<__half const*>(a_ptr);
    extern __shared__ __align__(16) unsigned char staged[];
    if constexpr(AMode==2) {
        for(int i=tid;i<K/8;i+=Warps*32)
            reinterpret_cast<uint4*>(staged)[i]=reinterpret_cast<uint4 const*>(act)[i];
        __syncthreads();act=reinterpret_cast<__half const*>(staged);
    }
    float total=0;''')
    s=s.replace('auto ptr=static_cast<__half const*>(a_ptr)+g*32+residue+slot*8;', 'auto ptr=act+g*32+residue+slot*8;')
    s=replace_once(s,'        uint32_t sz_bits=0;', '''        if constexpr(Loading==1) {
            asm volatile("" : "+r"(unit.x),"+r"(unit.y),"+r"(unit.z),"+r"(unit.w),
                "+r"(word[0].x),"+r"(word[0].y),"+r"(word[1].x),"+r"(word[1].y),
                "+f"(av[0].x),"+f"(av[0].y),"+f"(av[1].x),"+f"(av[1].y),
                "+f"(av[2].x),"+f"(av[2].y),"+f"(av[3].x),"+f"(av[3].y) : : "memory");
        }
        uint32_t sz_bits=0;''')
    return replace_once(s,'auto sz=aligned_scale_zero(unit,g&7);','auto sz=latency_half_header<Header>(unit,g&7);')


def affine_source():
    s=original_kernel('affine')
    s=replace_once(s,'template<int Variant,int Columns,int Warps,int P,int N,int K>',
        'template<int Header,int Loading,int AMode,int Columns,int Warps,int P,int N,int K>')
    s=replace_once(s,'__global__ void q4_reader_reuse(', '__global__ void q4_affine_pipeline(')
    s=replace_once(s,'    constexpr bool Early=true;', '    constexpr int Variant=4;\n    constexpr bool Early=true;')
    s=replace_once(s,'    float2 total[Pairs]{};', '''    auto act=static_cast<__half const*>(a_ptr);
    extern __shared__ __align__(16) unsigned char staged[];
    if constexpr(AMode==2) {
        for(int i=tid;i<K/8;i+=Warps*32)
            reinterpret_cast<uint4*>(staged)[i]=reinterpret_cast<uint4 const*>(act)[i];
        __syncthreads();act=reinterpret_cast<__half const*>(staged);
    }
    float2 total[Pairs]{};''')
    insertion='''        uint32_t all_words[8][Pairs];
        float4 all_a[2][4];
        if constexpr(Loading==1) {
            #pragma unroll
            for(int r=0;r<8;++r) {
                auto ptr=low+size_t(g*8+r)*N+col;
                if constexpr(P==2) all_words[r][0]=*reinterpret_cast<uint32_t const*>(ptr);
                else {uint2 b=*reinterpret_cast<uint2 const*>(ptr);all_words[r][0]=b.x;all_words[r][1]=b.y;}
            }
            #pragma unroll
            for(int slot=0;slot<4;++slot) {
                if constexpr(AMode==1) {
                    uint4 raw=*reinterpret_cast<uint4 const*>(act+g*32+slot*8);
                    uint32_t regs[4]={raw.x,raw.y,raw.z,raw.w};
                    #pragma unroll
                    for(int h=0;h<2;++h) {
                        float2 lo=__half22float2(__halves2half2(__ushort_as_half(uint16_t(regs[2*h])),__ushort_as_half(uint16_t(regs[2*h]>>16))));
                        float2 hi=__half22float2(__halves2half2(__ushort_as_half(uint16_t(regs[2*h+1])),__ushort_as_half(uint16_t(regs[2*h+1]>>16))));
                        all_a[h][slot]=make_float4(lo.x,lo.y,hi.x,hi.y);
                    }
                } else {
                    #pragma unroll
                    for(int h=0;h<2;++h) all_a[h][slot]=aligned_activation<0>(act,g*32+slot*8+h*4);
                }
            }
            // A compiler-only issue boundary. It does not synchronize threads.
            #pragma unroll
            for(int p=0;p<P;++p) asm volatile("" : "+r"(metadata[p].x),"+r"(metadata[p].y),"+r"(metadata[p].z),"+r"(metadata[p].w) : : "memory");
            #pragma unroll
            for(int r=0;r<8;++r) {
                #pragma unroll
                for(int p=0;p<Pairs;++p) asm volatile("" : "+r"(all_words[r][p]) : : "memory");
            }
            #pragma unroll
            for(int h=0;h<2;++h) {
                #pragma unroll
                for(int slot=0;slot<4;++slot) asm volatile("" : "+f"(all_a[h][slot].x),"+f"(all_a[h][slot].y),"+f"(all_a[h][slot].z),"+f"(all_a[h][slot].w) : : "memory");
            }
        }
'''
    s=replace_once(s,'        float2 dot[Pairs]{};',insertion+'        float2 dot[Pairs]{};')
    start=s.index('                auto ptr=low+size_t(g*8+half*4+r)*N+col;')
    end=s.index('\n            }\n            #pragma unroll\n            for(int slot',start)
    old=s[start:end]
    s=s[:start]+'''                if constexpr(Loading==1) {
                    #pragma unroll
                    for(int p=0;p<Pairs;++p) words[r][p]=all_words[half*4+r][p];
                } else {
'''+old+'\n                }'+s[end:]
    s=replace_once(s,'                else av=aligned_activation<0>(a_ptr,g*32+slot*8+half*4);',
        '''                else if constexpr(Loading==1) av=all_a[half][slot];
                else av=aligned_activation<0>(act,g*32+slot*8+half*4);''')
    s=replace_once(s,'s0=q4_affine_header32(u0,g&7);s1=q4_affine_header32(u1,g&7);',
        's0=latency_affine_header<Header>(u0,g&7);s1=latency_affine_header<Header>(u1,g&7);')
    return s


def source(n,k,family):
    s=prior.source(1024,5120) if family == 'affine' else prior.small_source()
    marker='extern "C" int q4_followup_run_1024_5120(' if family == 'affine' else 'extern "C" int q4_followup_small_run('
    s=s[:s.index(marker)]
    s+='\nnamespace QKG_CONCAT(kpack_q,QKG_QTYPE) {\n'+(ROOT/'dev/gemv_ppu/small_latency.hpp').read_text()
    if family == 'affine':
        s+='\n'+affine_source()
    elif family == 'residue2':
        helper=(ROOT/'dev/gemv_cuda/q4_cooperative_affine.cuh').read_text()
        helper=helper[helper.index('template<int Count'):helper.index('template<int Width')]
        affine_header=(ROOT/'dev/gemv_cuda/q4_group_affine.cuh').read_text()
        affine_header=affine_header[affine_header.index('__device__'):affine_header.index('template<int Columns')]
        s+='\n'+affine_header+'\n'+helper+'\n'+original_kernel(family)+'\n'+residue_source()
    s+='\n}\n'
    s+=f'''extern "C" int q4_latency_run_{n}_{k}(int control,int header,int loading,int a_mode,int warps,int columns,int values,
        void const* a,void const* low,void const* units,void* out,void* stream) {{
    if(!a || !low || !units || !out || (uintptr_t(a)&15) || (uintptr_t(low)&15) ||
       (uintptr_t(units)&15) || (uintptr_t(out)&3) || (control!=0 && control!=1)) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    using namespace QKG_CONCAT(kpack_q,QKG_QTYPE);
'''
    for c in inventory(n,k):
        if c.family != family: continue
        h,l,am,w,col,p=c.recipe[1:]
        if family == 'affine':
            baseline_kernel=f'q4_reader_reuse<4,{col},{w},{p},{n},{k}>'
            kernel=f'q4_affine_pipeline<{h},{l},{am},{col},{w},{p},{n},{k}>'
        elif family == 'meta':
            baseline_kernel=f'q4_cooperative_metadata<8,{w},false,true,{n},{k}>'
            kernel=f'q4_small_pipeline<{h},{l},{am},{w},{n},{k}>'
        else:
            baseline_kernel=f'q4_cooperative_residue2<{w},{n},{k},false>'
            kernel=f'q4_residue_pipeline<{h},{l},{am},{w},{n},{k}>'
        grid=c.geometry(n,k)['grid']
        s+=f'''    if(header=={h} && loading=={l} && a_mode=={am} && warps=={w} && columns=={col} && values=={p}) {{
        if(control) {baseline_kernel}<<<{grid},{w*32},0,static_cast<hggcStream_t>(stream)>>>(a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        else {kernel}<<<{grid},{w*32},{2*k if am==2 else 0},static_cast<hggcStream_t>(stream)>>>(a,static_cast<uint8_t const*>(low),static_cast<uint8_t const*>(units),static_cast<float*>(out));
        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;
    }}
'''
    return s+'    return QKG_INVALID;\n}\n'


def access(c,n,k,bases=None):
    bases=dict(A=0,B=0,metadata=0) if bases is None else dict(bases)
    streams={}; cta=c.geometry(n,k)
    def add(name,addresses,width,space='GLOBAL'):
        streams[name]=dict(lane_byte_addresses=addresses,width_bytes=width,space=space,
            granules={str(s):footprint(addresses,width,s) for s in (32,64,128)})
    if c.family == 'affine':
        g=[lane//c.columns for lane in range(32)];cols=[(lane%c.columns)*c.values for lane in range(32)]
        add('B_r0',[bases['B']+2*(group*8*n+col) for group,col in zip(g,cols)],2*c.values)
        add('units_p0',[bases['metadata']+16*(group//8*n+col) for group,col in zip(g,cols)],16)
        width=16 if c.a_mode==1 else 8
        add('A_slot0',[(0 if c.a_mode==2 else bases['A'])+group*64 for group in g],width,'SHARED' if c.a_mode==2 else 'GLOBAL')
        a_bytes=2*n*k//(c.columns*c.values) if c.a_mode==2 else 2*n*k//c.values
        unit_bytes=n*k//2
    else:
        lanes=8 if c.family=='meta' else 4
        tile=lanes;g=[lane//lanes for lane in range(32)];rs=[(lane%lanes)*(8//lanes) for lane in range(32)]
        for i in range(8//lanes):
            add(f'B_r{i}',[bases['B']+2*((group*8+residue+i)*n) for group,residue in zip(g,rs)],2*tile)
        add('units',[bases['metadata']+16*(group//8*n+lane%lanes) for lane,group in enumerate(g)],16)
        if c.family=='meta' and c.a_mode==0:
            add('A_vector',[bases['A']+group*64+residue*8 for group,residue in zip(g,rs)],8)
        else:
            for slot in range(4):
                add(f'A_slot{slot}',[(0 if c.a_mode==2 else bases['A'])+2*(group*32+residue+8*slot) for group,residue in zip(g,rs)],
                    2 if c.family=='meta' else 4,'SHARED' if c.a_mode==2 else 'GLOBAL')
        a_bytes=2*n*k//tile
        unit_bytes=n*k//2
    if c.a_mode==2:
        add('A_stage_warp0',[bases['A']+16*lane for lane in range(32)],16)
    return dict(key=c.key,model='SOURCE_FOOTPRINT_NOT_MEASURED_TRAFFIC',base_mod128=bases,geometry=cta,
        streams=streams,logical_global_lane_bytes=dict(B=n*k//2,A=a_bytes,metadata=unit_bytes),
        shared_addresses='RELATIVE_TO_DYNAMIC_A_BASE_NOT_GLOBAL_A_BASE' if c.a_mode==2 else 'NOT_APPLICABLE',
        load_order='ISSUE_JOIN_THEN_DECODE' if c.loading else 'COMPILER_SCHEDULED',
        header='LOP3_SELECT_32BIT' if c.header else 'CONDITIONAL_32BIT')


def isa(text,n,k,family):
    name={'meta':'q4_small_pipeline','residue2':'q4_residue_pipeline','affine':'q4_affine_pipeline'}[family]
    sections=list(re.finditer(r'Disassembly of section \.text\.kernel\.[^\n]+:',text));result={}
    for i,m in enumerate(sections):
        if name+'I' not in m[0]:continue
        args=list(map(int,re.findall(r'(?:I|E)Li(\d+)',m[0])))
        h,l,a=args[:3]
        if family=='affine':col,w,p,nn,kk=args[3:]
        else:w,nn,kk=args[3:];col=p=0
        assert (nn,kk)==(n,k)
        candidate=Candidate(family,h,l,a,w,col,p)
        body=text[m.end():sections[i+1].start() if i+1<len(sections) else None]
        ops=Counter(re.findall(r'\t([a-z][\w.]+)\s',body))
        lines=[line for line in body.splitlines() if any(op in line for op in ('vmem.ld.','s.wait','s.cbr','tsm.','s.blksyn'))]
        waits=[j for j,line in enumerate(lines) if 's.wait' in line and 'vldcnt' in line]
        result[candidate.key]=dict(scope='STATIC_NATIVE_ISA_NOT_DYNAMIC_COUNTS',
            code_fastpath_present=bool(ops['v.lop3.b32'] and any('f16x2' in op for op in ops)),
            fp32_fma_present=any(op.startswith('v.fma.f32') for op in ops),
            global_loads_before_first_vld_wait=sum('vmem.ld.' in line for line in lines[:waits[0]]) if waits else 0,
            load_wait_control_order=lines,
            operations={op:ct for op,ct in ops.items() if op.startswith(('vmem.','tsm.','s.cbr','s.blksyn')) or
                any(s in op for s in ('shuffle','shrl.b64','f16x2','fma.f32','lop3','perm'))})
    return result


def verify(candidate,followup,reuse,config,previous,controls,bundle,*,sources=True):
    old=prior.verify(followup,reuse,config,previous,controls,bundle,sources=sources)
    m=json.loads((candidate/'manifest.json').read_text())
    expected={payload(n,k,f) for n,k in SHAPES for f in {c.family for c in inventory(n,k)}}
    if (m.get('schema')!=SCHEMA or m.get('plan')!=json.loads(json.dumps(plan())) or
        m.get('followup_manifest_sha256')!=digest(followup/'manifest.json') or
        m.get('review_sha256')!=digest(REVIEW) or m.get('compiler_sha256')!=old['compiler_sha256'] or
        set(m.get('payloads',{}))!=expected):raise ValueError('small latency/control contract differs')
    for name,row in m['payloads'].items():
        path=candidate/name
        if digest(path)!=row['sha256'] or path.read_bytes()[:4]!=b'\x7fELF':raise ValueError('small latency image differs: '+name)
    if digest(candidate/'isa-stats.json')!=m['isa_sha256']:raise ValueError('small latency ISA differs')
    for name,sha in m['source_hashes'].items() if sources else []:
        path=(ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or digest(path)!=sha:raise ValueError('small latency source differs: '+name)
    return m
