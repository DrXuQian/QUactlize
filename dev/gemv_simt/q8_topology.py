"""M1 Q8 topology experiment; canonical bytes and production policy are frozen."""
import math
from pathlib import Path

from dev.gemv_simt.model_followup import kernel_body
from quactlize.execution.simt_codegen import Config
from dev.gemv_simt.access import pattern

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='quactlize.q8-topology.v1'
POINTS=((512,2048,Config(5,4,8,4)),(2048,512,Config(5,8,4,4)),(2048,4096,Config(1,4,8,4)))
# Thirteen geometries, two existing A readers. The new C16/P2 path fills a
# 64-byte B request; C4/P2 tests more N tiles. Neither changes the file format.
GEOMETRIES=((4,2,2),(4,4,2),(4,8,2),(4,4,4),(4,8,4),
            (8,2,2),(8,4,2),(8,8,2),(8,4,4),(8,8,4),
            (16,2,2),(16,4,2),(16,8,2))
CONFIGS=tuple(Config(v,c,w,p) for c,w,p in GEOMETRIES for v in (4,5))


def inventory(n,k):
    cells=[];pruned=[]
    for index,c in enumerate(CONFIGS):
        workers=c.warps*32//c.columns
        for split in (1,2,4,8):
            reason=None
            if workers*split>k//32:reason='EMPTY_WORKERS_OR_SPLIT_PARTITION'
            if reason:
                pruned.append(dict(recipe=index,split=split,reason=reason));continue
            for reducer in ((0,) if split==1 else (0,1)):
                cells.append(dict(recipe=index,split=split,reducer=reducer,
                    key=f'v{c.variant}-c{c.columns}-w{c.warps}-p{c.values}-s{split}-r{reducer}'))
    return dict(cells=cells,pruned=pruned)


def config(cell):
    c=CONFIGS[cell['recipe']]
    return Config(c.variant,c.columns,c.warps,c.values,cell['split'])


def access(cell,n,k,bases=None):
    c=config(cell);p=pattern(8,c,n,k,bases=bases)
    blocks=c.split*n//c.tile_n
    p.update(ctas=blocks,threads=c.warps*32,total_warps=blocks*c.warps,
        workers=c.warps*32//c.columns,passes=math.ceil((k//32)/(c.split*c.warps*32//c.columns)),
        independent_outputs_per_cta=c.tile_n,
        reduction='S1_CTA_ONLY' if c.split==1 else 'PRODUCER_PLUS_'+('LEGACY_SCALAR' if cell['reducer']==0 else 'F32_FLOAT2_ORDERED'),
        launch_count=1 if c.split==1 else 2,
        dequant='EXISTING_LOP3_HALF2_EXACT_INTEGER_TO_F32',
        compute='F16_REGISTER_ROUNDING_F32_ACCUMULATE_AND_OUTPUT')
    return p


def source():
    body=kernel_body(True)
    text='''#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "quactlize/decode/reducer.cuh"
namespace quactlize::execution::q8_topology {
using namespace quactlize::execution::simt;
using namespace quactlize::execution::simt::q8_vector;
'''+body+'''
}
extern "C" int q8_topology_run(qkg_call_v1 const* c,int index,int split,int reducer) {
    using namespace quactlize::execution;
    if(!c || c->qtype!=8 || c->mode!=QKG_DENSE || c->rows!=1 || c->input_type!=QKG_F32 ||
       (split!=1 && split!=2 && split!=4 && split!=8) || reducer<0 || reducer>1 ||
       (split==1 && reducer!=0)) return QKG_INVALID;
    // The optional float2 reducer has a stronger alignment than the public
    // scalar endpoint. Reject it explicitly, never reinterpret an odd base.
    if(reducer && ((uintptr_t(c->output)|uintptr_t(c->workspace))&7)) return QKG_INVALID;
    qkg_simt_config_v1 control{1,sizeof(control),1,4,4,4,split};
    auto arr=q8_kpack2::arrangement();qkg_sizes_v1 sizes{};
    int rc=simt::query(*c,control,&arr,sizes);if(rc) return rc;
    rc=simt::buffers(*c,sizes);if(rc) return rc;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    auto stream=static_cast<hggcStream_t>(c->stream);
'''
    for index,c in enumerate(CONFIGS):
        text+=f'''    {'if' if index==0 else 'else if'}(index=={index}) {{
        if(split*{c.warps*32//c.columns}>c->k/32) return QKG_SHAPE;
        q8_topology::kernel<1,0,{c.variant-4},{c.columns},{c.warps},{c.values}>
            <<<split*(c->n/{c.tile_n}),{c.warps*32},0,stream>>>(*c,split);
    }}
'''
    text+='''    else return QKG_INVALID;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    if(split>1) {
        if(reducer==0) simt::register_reuse_reduce<8><<<(c->n+127)/128,128,0,stream>>>(*c,split);
        else {
            auto p=static_cast<float const*>(c->workspace);
            switch(split) {
'''
    for s in (2,4,8):
        text+=f'            case {s}: quactlize::decode::reduce_decode<{s}><<<(c->n+63)/64,32,0,stream>>>(p,c->output,c->n);break;\n'
    text+='''            }
        }
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    }
    return QKG_OK;
}
'''
    text+='''extern "C" int q8_topology_control(qkg_call_v1 const* c,int point) {
    using namespace quactlize::execution;
    if(!c || c->qtype!=8 || c->mode!=QKG_DENSE || c->rows!=1 || c->input_type!=QKG_F32)
        return QKG_INVALID;
    qkg_simt_config_v1 control{1,sizeof(control),1,4,4,4,1};
    auto arr=q8_kpack2::arrangement();qkg_sizes_v1 sizes{};
    int rc=simt::query(*c,control,&arr,sizes);if(rc) return rc;
    rc=simt::buffers(*c,sizes);if(rc) return rc;
    if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
    auto stream=static_cast<hggcStream_t>(c->stream);
'''
    for index,(n,k,c) in enumerate(POINTS):
        kernel=(f'simt::q8_vector::kernel<1,0,{c.variant-4},{c.columns},{c.warps},{c.values}>'
                if c.variant>=4 else f'simt::register_reuse<8,1,{c.variant},{c.columns},{c.warps},{c.values},0>')
        text+=f'''    {'if' if index==0 else 'else if'}(point=={index} && c->n=={n} && c->k=={k})
        {kernel}<<<{n//c.tile_n},{c.warps*32},0,stream>>>(*c,1);
'''
    return text+'''    else return QKG_INVALID;
    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;
}
'''
