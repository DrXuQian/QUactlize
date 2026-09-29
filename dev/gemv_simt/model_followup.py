"""Bounded reuse of Q4 H32/index lessons on the five observed M1 SIMT bodies.

The shipping kernel/selector is untouched. Each isolated image contains an
unchanged clone and same-geometry single-axis/combined candidates. The runner
also loads the immutable shipping image; clone equality is mandatory.
"""
from dataclasses import dataclass
from pathlib import Path

from quactlize.execution.simt_codegen import Config

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='quactlize.model-simt-followup.v1'


@dataclass(frozen=True)
class Point:
    name: str
    q: int
    n: int
    k: int
    mode: int
    channels: int
    compute: int
    config: Config

    @property
    def arms(self):
        return (0,2) if self.q==8 else (0,1,2,3)


POINTS=(
    Point('q8-2048-4096',8,2048,4096,0,1,0,Config(1,4,8,4)),
    Point('q8-512-2048',8,512,2048,0,1,0,Config(5,4,8,4)),
    Point('q8-2048-512',8,2048,512,0,1,0,Config(5,8,4,4)),
    Point('q4-moe-gate-up',12,1024,2048,2,1,1,Config(3,4,4,4)),
    Point('q5-moe-down',13,2048,512,2,8,1,Config(3,4,2,8)),
)


def once(text, old, new):
    if text.count(old)!=1:
        raise ValueError('shipping source seam changed: '+old[:90])
    return text.replace(old,new,1)


def kernel_body(vector):
    file=ROOT/'quactlize/execution'/('simt_q8_vector.cuh' if vector else 'simt_kernel.cuh')
    text=file.read_text()
    if vector:
        start=text.index('template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false,')
        end=text.index('\ntemplate<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false>\n__global__ void kernel',start)
        return once(text[start:end],'__device__ __forceinline__ void kernel_body','__global__ void kernel')
    start=text.index(
                     'template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>\n__global__ void register_reuse')
    end=text.index('\ntemplate<',start+len('template<'))
    return text[start:end]


def candidate_body(vector):
    body=kernel_body(vector)
    if not vector:
        # Production retains the exact tested axes; experiments still select
        # them explicitly instead of invoking the shape-limited shipping gate.
        body=once(body,',int Changes=0','')
        body=once(body,'template<int ','template<int Changes,int ')
        return once(body,'__global__ void register_reuse','__global__ void candidate')
    body=once(body,'template<int ','template<int Changes,int ')
    body=once(body,'__global__ void '+('kernel' if vector else 'register_reuse'),
              '__global__ void candidate')
    if not vector:
        body=body.replace('affine<Q>(', 'selected_affine<Changes,Q>(')
    body=once(body,'    int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;',
              '    using Index=std::conditional_t<(Changes&2)!=0,unsigned,int>;\n'
              '    Index tid=threadIdx.x,lane=tid%32,worker=tid/Columns;')
    prefix='c' if vector else 'call'
    body=once(body,f'    int tile=blockIdx.x%({prefix}.n/TileN),outer=blockIdx.x/({prefix}.n/TileN);',
              f'    Index tile=blockIdx.x%({prefix}.n/TileN),outer=blockIdx.x/({prefix}.n/TileN);')
    body=once(body,'    int partition=outer%split,row=outer/split;',
              '    Index partition=outer%split,row=outer/split;')
    body=once(body,'    int col=tile*TileN+(tid%Columns)*P;',
              '    Index col=tile*TileN+(tid%Columns)*P;')
    body=once(body,'for(int g=' if vector else 'for (int g=',
              'for(Index g=' if vector else 'for (Index g=')
    if not vector:
        fold='''        #pragma unroll
        for (int w=tid/TileN;w<Warps;w+=32/TileN) sum+=partial[w*TileN+tid%TileN];'''
        body=once(body,fold,'''        if constexpr(Changes&2) {
            q4_s1::q4_medium_fold<0,Warps,TileN>(sum,partial,unsigned(tid));
        } else {
'''+fold+'\n        }')
    return body


def source(point):
    c=point.config
    vector=c.variant>=4
    text='''#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
namespace quactlize::execution::model_followup {
using namespace quactlize::execution::simt;
using namespace quactlize::execution::simt::q8_vector;
template<int Changes,int Q>
__device__ __forceinline__ float2 selected_affine(Meta<Format<Q>::words> const& m,int group) {
    if constexpr((Changes&1) && (Q==12 || Q==13)) {
        static_assert(Format<Q>::words==4 && Format<Q>::Unit::kGroups==8);
        return q4_s1::q4_affine_header32(make_uint4(m.word[0],m.word[1],m.word[2],m.word[3]),unsigned(group)&7u);
    } else return simt::affine<Q>(m,group);
}
'''
    text+=candidate_body(vector)
    text+='\n} // namespace quactlize::execution::model_followup\n'
    text+='''extern "C" int qk_model_followup_run(qkg_simt_call_v2 const* d,
    qkg_simt_config_v1 const* f,int arm) {
    using namespace quactlize::execution;
    if(!d || !f) return QKG_INVALID;
'''
    arrangement='q8_kpack2::arrangement()' if point.q==8 else 'ppu_arrangements::q4_kpack4_transpose_v1()' if point.q==12 else f'ppu_arrangements::kquant_kpack_transpose_v1({point.q})'
    text+=f'''    if(d->call.qtype!={point.q} || d->call.input_type!=QKG_F32 || d->compute_type!={point.compute} ||
       f->variant!={c.variant} || f->columns!={c.columns} || f->warps!={c.warps} || f->values!={c.values} || f->split!=1)
        return QKG_INVALID;
    auto a={arrangement};
'''
    # Use the same arrangement registry and buffer validation as production.
    text+='''    qkg_sizes_v1 sizes{};
    int rc=simt::query_v2(*d,*f,&a,sizes);if(rc) return rc;
    rc=simt::buffers_v2(*d,sizes);if(rc) return rc;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    auto stream=static_cast<hggcStream_t>(d->call.stream);
'''
    text+=f'    int blocks=d->call.rows*(d->call.n/{c.tile_n});\n'
    for arm in point.arms:
        template=f'{arm},1,{point.compute},{c.variant-4},{c.columns},{c.warps},{c.values}' if vector else \
                 f'{arm},{point.q},1,{c.variant},{c.columns},{c.warps},{c.values},{point.compute}'
        text+=f'    {"if" if arm==0 else "else if"}(arm=={arm}) model_followup::candidate<{template}><<<blocks,{c.warps*32},0,stream>>>(d->call,1);\n'
    text+='''    else return QKG_INVALID;
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
'''
    if point.q==8:
        text+='''
namespace quactlize::execution::model_followup {
// Optimistic memory-only reference, never a GEMV performance admission.
// All loaded bits affect an externally checked XOR; no weight load is dead.
__global__ void read_roof(qkg_call_v1 c) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned step=gridDim.x*blockDim.x;
    uint32_t sum=0;
    uint64_t bytes=uint64_t(c.n)*c.k;
    for(uint64_t j=i;j<bytes/16;j+=step) {
        uint4 v=reinterpret_cast<uint4 const*>(c.low)[j];sum^=v.x^v.y^v.z^v.w;
    }
    for(uint64_t j=i;j<bytes/256;j+=step) {
        uint4 v=reinterpret_cast<uint4 const*>(c.units)[j];sum^=v.x^v.y^v.z^v.w;
    }
    for(int bit=16;bit;bit/=2) sum^=__shfl_xor_sync(0xffffffffu,sum,bit);
    if((threadIdx.x&31)==0)
        static_cast<uint32_t*>(c.workspace)[blockIdx.x*4+threadIdx.x/32]=sum;
}
}
extern "C" int qk_model_read_roof(qkg_call_v1 const* c,int blocks) {
    if(!c || c->qtype!=8 || c->mode!=0 || c->rows!=1 || c->n<=0 || c->k<=0 ||
       !c->low || !c->units || !c->workspace ||
       (blocks!=72 && blocks!=144 && blocks!=288) ||
       c->workspace_bytes<uint64_t(blocks)*16 ||
       (uintptr_t(c->low)|uintptr_t(c->units)|uintptr_t(c->workspace))%16 ||
       uint64_t(c->n)*c->k%256) return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    quactlize::execution::model_followup::read_roof<<<blocks,128,0,static_cast<hggcStream_t>(c->stream)>>>(*c);
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
'''
    return text
