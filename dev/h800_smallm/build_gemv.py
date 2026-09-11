#!/usr/bin/env python3
"""Lift the three FP32 H800 readers to multi-row/device-indexed launches.

The default arms lift the frozen M1 dot implementations by changing row
bases and grid geometry. Bounded small-shape alternatives also explore
multi-row reuse and different group-affine readers. None changes packing.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from dev.gemv_cuda.build import digest, replace_once
from dev.gemv_cuda.build_h800_candidates import source as candidate
from dev.gemv_cuda.build_xplane_accum import fp32_source
from dev.gemv_cuda.summarize_h800_confirmation import POLICY
from dev.gemv_ppu.bload_source import REFERENCE_SHA256, reference_fp32

SHAPES=tuple((n,k) for _,n,k in POLICY)
ARMS={"small":"meta-static-global-rs-fast-bare-av",
      "medium":"affine8-early-fast-bare-a4","large":"affine4-early-fast-bare",
      "affine2-medium":"affine2-early-fast-bare","affine4-medium":"affine4-early-fast-bare",
      "affine2-small":"affine2-early-fast-bare","affine4-small":"affine4-early-fast-bare","affine8-small":"affine8-early-fast-bare-a4",
      "rows2-small":"meta-static-global-rs-fast-bare-av","rows4-small":"meta-static-global-rs-fast-bare-av",
      "rows2w2-small":"meta2-static-global-rs-fast-bare-av","rows4w2-small":"meta2-static-global-rs-fast-bare-av",
      "scalar-small":"affine2-early-fast-bare"}
XP=[(c,w,1) for c in (1,2,4,8) for w in (2,4,8)]
REF=[(c,w,kw) for c in (1,2,4,8) for w in (1,2,4,8)
     for kw in (1,2,4,8) if w*kw<=32]

BASES='''
    int const rr=ROW_AXIS;
    int const token=rr/c.topk,slot=rr%c.topk;
    int const expert=!Indexed?0:c.ids[int64_t(token)*c.ids_stride+slot];
    int64_t const abase=!Indexed?int64_t(rr)*c.a_row_stride:
        int64_t(token)*c.a_token_stride+int64_t(slot%c.channels)*c.a_row_stride;
'''
CHECK='''
    if(!p || p->version!=1 || p->size!=sizeof(*p) || p->qtype!=12 ||
       !p->a || !p->low || !p->units || !p->output || p->rows<1 || p->rows>32 ||
       p->experts<1 || p->input_type!=QKG_F16 || p->topk<1 || p->rows%p->topk ||
       p->channels<1 || p->channels>p->topk || p->a_row_stride<p->k ||
       p->out_row_stride<p->n || p->a_row_stride%8 ||
       p->a_token_stride%8 || p->ids_stride<p->topk ||
       (uintptr_t(p->a)&15) || (uintptr_t(p->low)&15) || (uintptr_t(p->units)&15) ||
       (uintptr_t(p->output)&3) ||
       (p->mode!=QKG_DENSE && p->mode!=QKG_INDEXED) ||
       (p->mode==QKG_DENSE && (p->experts!=1 || p->topk!=1)) ||
       (p->mode==QKG_INDEXED && !p->ids)) return -1;
    auto const& c=*p;
    if(c.n<256 || c.n%256 || c.k<1024 || c.k>8192 || c.k%1024) return -1;
'''


def specialize_indexing(body,kernel):
    """Remove indexed divisions/ID loads entirely from dense specializations."""
    start=body.index(kernel+'(qkg_call_v1 c)')
    template=body.rfind('\ntemplate',0,start)
    end=body.index('>',template)
    if template<0 or end>start: raise ValueError('kernel template seam')
    return body[:end]+', bool Indexed=false'+body[end:]


def kpack_source(family):
    name=ARMS[family]; body,_=candidate(name)
    if family=='scalar-small':
        first=body.index('extern "C" int qkg_launch_12(')
        prefix=body[:first].rstrip()
        if not prefix.endswith('\n}\n}'): raise ValueError('candidate namespace seam')
        header=(ROOT/'dev/h800_smallm/scalar_affine.cuh').read_text()
        body=prefix[:-1]+header+'\n}\n'
        body+='extern "C" int q4_smallm_run(qkg_call_v1 const* p,int columns,int warps,int split) {\n'+CHECK
        body+='    using namespace kpack_q12;\n    if(c.n!=512 || c.k!=2048 || split!=1) return -1;\n'
        recipes=[(cc,ww,1) for cc in (4,8,16,32) for ww in (4,8,16)]
        for cc,ww,_ in recipes:
            body+=f'''    if(columns=={cc} && warps=={ww}) {{
        if(c.mode==QKG_DENSE) q4_scalar_affine<{cc},{ww},512,2048,false><<<dim3(512/{cc},c.rows),{32*ww},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        else q4_scalar_affine<{cc},{ww},512,2048,true><<<dim3(512/{cc},c.rows),{32*ww},0,static_cast<hggcStream_t>(c.stream)>>>(c);
        return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
    }}
'''
        return body+'    return -1;\n}\n',{'512x2048':[list(r) for r in recipes]}
    kernel='q4_cooperative_metadata' if name.startswith('meta') else 'q4_group_affine'
    signature=f'__global__ void {kernel}(void const* a_ptr,uint8_t const* low_ptr,uint8_t const* units_ptr,float* out_ptr) {{'
    bases=BASES.replace('ROW_AXIS','blockIdx.y')+'''
    void const* a_ptr=static_cast<__half const*>(c.a)+abase;
    uint8_t const* low_ptr=c.low+int64_t(expert)*N*K/2;
    uint8_t const* units_ptr=c.units+int64_t(expert)*N*K/16;
    float* out_ptr=c.output+int64_t(rr)*c.out_row_stride;
'''
    body=replace_once(body,signature,f'__global__ void {kernel}(qkg_call_v1 c) {{'+bases)
    body=specialize_indexing(body,kernel)
    row_tile=int(family[4]) if family.startswith('rows') else 1
    if row_tile>1:
        body=multirow_source(body,row_tile)
    start=body.index('extern "C" int qkg_launch_12(')
    launch_source=body[start:]
    body=body[:start]+'extern "C" int q4_smallm_run(qkg_call_v1 const* p,int columns,int warps,int split) {\n'+CHECK
    body+='    using namespace kpack_q12;\n    if(split!=1) return -1;\n'
    if row_tile>1: body+='    if(c.mode!=QKG_DENSE) return -1;\n'
    recipes={}
    for (_,n,k),(arm,(cols,warps,_)) in POLICY.items():
        if family.endswith('-small'):
            if (n,k)!=(512,2048): continue
        elif family.endswith('-medium'):
            if (n,k)!=(1024,5120): continue
        elif arm!=name: continue
        # More rows supply CTA parallelism that M1 lacks. Retune the bounded
        # per-CTA N width and K-worker count, retaining every M1 candidate.
        allowed={(cc,ww) for cc in (1,2,4) for ww in (2,4,5,8,10,16)}
        if family in ('affine2-small','affine4-small'):
            allowed|={(8,ww) for ww in (2,4,8,16)}
        if family=='affine2-small': allowed|={(16,ww) for ww in (4,8,16)}
        if row_tile>1: allowed={(cc,ww) for cc in (1,2) for ww in (4,8,16)}
        choices=[]
        for cc,ww in sorted(allowed):
            pattern=rf'    if\(f.columns=={cc} && f.warps=={ww} && c.n=={n} && c.k=={k}\) \{{\n(.*?)\n    \}}'
            found=re.findall(pattern,launch_source,re.S)
            if not found and cc>4:
                seed=pattern.replace(f'f.columns=={cc}', 'f.columns==4')
                found=re.findall(seed,launch_source,re.S)
                if len(found)!=1: raise ValueError('affine column extension seed missing')
                width=int(name.split('-')[0].removeprefix('affine'))
                if cc*width>32: raise ValueError('affine warp reduction extent')
                stmt=replace_once(found[0],f'q4_group_affine<4,{ww},',f'q4_group_affine<{cc},{ww},')
                stmt=replace_once(stmt,f'<<<c.n/{4*width},',f'<<<c.n/{cc*width},')
            elif len(found)==1: stmt=found[0]
            else: continue
            stmt,count=re.subn(r'<<<c.n/(\d+),',r'<<<dim3(c.n/\1,c.rows),',stmt)
            if count!=1: raise ValueError('candidate grid seam')
            if row_tile>1: stmt=replace_once(stmt,'c.rows)',f'(c.rows+{row_tile-1})/{row_tile})')
            stmt=replace_once(stmt,'>>>(c.a,c.low,c.units,c.output);','>>>(c);')
            line=next(line for line in stmt.splitlines() if '<<<' in line)
            dense=replace_once(line,'><<<',',false><<<')
            indexed=replace_once(line,'><<<',',true><<<')
            stmt=replace_once(stmt,line,'        if(c.mode==QKG_DENSE) '+dense.strip()+'\n        else '+indexed.strip())
            body+=f'    if(c.n=={n} && c.k=={k} && columns=={cc} && warps=={ww}) {{\n{stmt}\n    }}\n'
            choices.append([cc,ww,1])
        recipes[f'{n}x{k}']=choices
    return body+'    return -1;\n}\n',recipes


def multirow_source(body,rows):
    """Reuse each dequantized B group across 2/4 independent activation rows."""
    start=body.index('__global__ void q4_cooperative_metadata(qkg_call_v1 c)')
    end=body.index('\n}\n',start)+3
    kernel=body[start:end]
    kernel=replace_once(kernel,'int const rr=blockIdx.y;',f'constexpr int RowTile={rows};\n    int const rr=blockIdx.y*RowTile;')
    kernel=replace_once(kernel,'float2 sums[Pairs]{};','float2 sums[RowTile][Pairs]{};')
    first=kernel.index('        float act[4];')
    finish=kernel.index('        #endif',first)+len('        #endif')
    kernel=kernel[:first]+'''        float act[RowTile][4]{};
        #pragma unroll
        for(int r=0;r<RowTile;++r) if(rr+r<c.rows) {
            auto ar=static_cast<__half const*>(a_ptr)+int64_t(r)*c.a_row_stride;
            float4 av=q4_warp_activation(ar,g,residue);
            act[r][0]=av.x;act[r][1]=av.y;act[r][2]=av.z;act[r][3]=av.w;
        }
'''+kernel[finish:]
    kernel=replace_once(kernel,'''                sums[p].x = fmaf(act[slot], w.x, sums[p].x);
                sums[p].y = fmaf(act[slot], w.y, sums[p].y);''','''                #pragma unroll
                for(int r=0;r<RowTile;++r) {
                    sums[r][p].x=fmaf(act[r][slot],w.x,sums[r][p].x);
                    sums[r][p].y=fmaf(act[r][slot],w.y,sums[r][p].y);
                }''')
    first=kernel.index('    if constexpr (Scatter) {')
    kernel=kernel[:first]+'''    static_assert(Scatter && !StageA);
    __shared__ float warp_value[RowTile*Warps*Width];
    #pragma unroll
    for(int r=0;r<RowTile;++r) {
        float values[Width];
        #pragma unroll
        for(int p=0;p<Pairs;++p) {values[2*p]=sums[r][p].x;values[2*p+1]=sums[r][p].y;}
        float v=q4_reduce_scatter_steps<Width,1>(values,lane);
        if(lane<Width) warp_value[(r*Warps+warp)*Width+lane]=v;
    }
    __syncthreads();
    if(tid<32) {
        #pragma unroll
        for(int r=0;r<RowTile;++r) {
            float sum=0;
            #pragma unroll
            for(int w=tid/Width;w<Warps;w+=32/Width) sum+=warp_value[(r*Warps+w)*Width+tid%Width];
            #pragma unroll
            for(int d=Width;d<32;d*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,d);
            if(tid<Width && rr+r<c.rows) out_ptr[int64_t(r)*c.out_row_stride+first+tid]=sum;
        }
    }
}
'''
    return body[:start]+kernel+body[end:]


def control_header(kind):
    if kind=='xplane':
        body=fp32_source((ROOT/'quactlize/include/gguf_bc_q4_gemv.hpp').read_text())
        signature='''kernel(half const* act, std::uint8_t const* low, std::uint8_t const* units,
       float* out, unsigned m, unsigned n, unsigned k) {'''
        binding='''
    auto act=static_cast<half const*>(c.a)+abase;
    auto low=c.low+int64_t(expert)*c.n*c.k/2;
    auto units=c.units+int64_t(expert)*c.n*c.k/16;
    auto out=c.output+int64_t(rr)*c.out_row_stride;
    unsigned const n=c.n,k=c.k;
'''
        end=body.index('\ntemplate <int CTA_N, int WARPS_N',body.index('kernel(')+len('kernel('))
        body=body[:end]+'\n} // namespace gguf_scale::bc_q4_gemv\n'
        body=replace_once(body,signature,'kernel(qkg_call_v1 c) {'+BASES.replace('ROW_AXIS','blockIdx.x')+binding)
        body=replace_once(body,'unsigned const row = blockIdx.x;','unsigned const row = 0;')
    else:
        original=ROOT/'dev/gemv_ppu/reference/gemv_ref.cuh'
        if digest(original)!=REFERENCE_SHA256: raise ValueError('raw reference changed')
        body=reference_fp32(original.read_text(),ppu=False)
        start=body.index('q4k_gemv_kernel(')
        end=body.index('\n{',start)+2
        binding='''
    auto act=static_cast<half const*>(c.a)+abase;
    auto w=reinterpret_cast<block_q4_K const*>(c.low)+int64_t(expert)*c.n*c.k/256;
    auto out=c.output+int64_t(rr)*c.out_row_stride;
    unsigned const n=c.n,k=c.k;
'''
        body=body[:start]+'q4k_gemv_kernel(qkg_call_v1 c)\n{'+BASES.replace('ROW_AXIS','blockIdx.x')+binding+body[end:]
        body=replace_once(body,'unsigned const row        = blockIdx.x;','unsigned const row        = 0;')
        end=body.index('template <int CTA_N, int WARPS_N = 8')
        body=body[:end]+'\n} // namespace q4k_gemv_fp32\n'
    body=specialize_indexing(body,'kernel' if kind=='xplane' else 'q4k_gemv_kernel')
    return '#include "quactlize/execution/api.h"\n'+body


def control_source(kind):
    body='''#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#undef CUTLASS_HOST_DEVICE
#undef CUTLASS_DEVICE
#define CUTLASS_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#include "control.cuh"
'''
    if kind=='xplane':
        pack=(ROOT/'dev/gemv_cuda/xplane_compare.cu').read_text()
        body+=pack[pack.index('#include "gguf_unit_pack.hpp"'):pack.index('template<int C, int W>')]
    body+='extern "C" int q4_smallm_run(qkg_call_v1 const* p,int columns,int warps,int split) {\n'+CHECK
    recipes=XP if kind=='xplane' else REF
    for cc,ww,kw in recipes:
        kernel='gguf_scale::bc_q4_gemv::kernel' if kind=='xplane' else 'q4k_gemv_fp32::q4k_gemv_kernel'
        body+=f'''    if(columns=={cc} && warps=={ww} && split=={kw} && c.n%{cc*ww}==0) {{
        if(c.mode==QKG_DENSE)
          {kernel}<{cc},{ww},{kw},false><<<dim3(c.rows,c.n/{cc*ww}),{ww*kw*32},size_t(c.k)*2,static_cast<cudaStream_t>(c.stream)>>>(c);
        else
          {kernel}<{cc},{ww},{kw},true><<<dim3(c.rows,c.n/{cc*ww}),{ww*kw*32},size_t(c.k)*2,static_cast<cudaStream_t>(c.stream)>>>(c);
        return int(cudaGetLastError());
    }}
'''
    return body+'    return -1;\n}\n',recipes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--jobs',type=int,default=5)
    p.add_argument('--arms',default=','.join([*ARMS,'xplane','reference']))
    a=p.parse_args(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    includes=[ROOT/'dev/gemv_cuda/compat',ROOT,ROOT/'quactlize/execution',ROOT/'quactlize/include',
        ROOT/'benchmarks',ROOT/'third_party/actlize/include',ROOT/'third_party/actlize/tools/util/include']
    start=time.monotonic()
    def build(arm):
        folder=out/arm; folder.mkdir()
        body,recipes=kpack_source(arm) if arm in ARMS else control_source(arm)
        if arm not in ARMS: (folder/'control.cuh').write_text(control_header(arm))
        src=folder/'kernel.cu';src.write_text(body)
        command=[str(a.cuda/'bin/nvcc'),'-std=c++17','-arch=sm_90','-O3','-lineinfo',
            '--expt-relaxed-constexpr','--shared','--cudart=shared','-Xcompiler=-fPIC','-Xlinker=-Bsymbolic','-Xptxas=-v',
            '-DCUTLASS_USE_PACKED_TUPLE=1','-DCUTE_USE_PACKED_TUPLE=1',
            '-I'+str(folder),*['-I'+str(x) for x in includes]]
        if arm in ARMS: command+=['-DQKG_QTYPE=12','-include',str(ROOT/'dev/gemv_cuda/compat/compiler_bridge.h')]
        command+=[str(src),'-o',str(folder/'kernel.so')]
        with (folder/'build.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('H800_SMALLM_BUILD COMPILED '+arm,flush=True)
        return arm,dict(recipes=recipes,source_sha256=digest(src),sha256=digest(folder/'kernel.so'),command=command)
    arms=a.arms.split(',')
    if not set(arms)<={*ARMS,'xplane','reference'} or len(arms)!=len(set(arms)): raise ValueError('unknown/duplicate arms')
    with ThreadPoolExecutor(max_workers=max(1,min(a.jobs,5))) as pool:
        records=dict(pool.map(build,arms))
    (out/'manifest.json').write_text(json.dumps(dict(arms=records,seconds=time.monotonic()-start,
        scope='CUDA_H800_DENSE_M1_TO_7_INDEXED_M1',production_changed=False),indent=2)+'\n')


if __name__=='__main__': main()
