"""Signed pointer induction/load-window experiment. No format or arithmetic changes."""
from dev.gemv_simt.q8_dense_source import body as previous_body
from dev.gemv_simt.model_followup import once

ARMS = (
    dict(name='original_control'),
    dict(name='signed_dense_grid_control'),
    dict(name='previous_unsigned_combined_control'),
    dict(name='signed_pointer_induction'),
    dict(name='signed_half_load_window'),
    dict(name='signed_induction_half_window'),
)


def body(arm):
    if not 1<=arm<len(ARMS):
        raise ValueError('arm0 must call the untouched original')
    prior=5 if arm==2 else 4
    text=once(previous_body(prior),f'void dense_ablation_{prior}(',f'void address_body_{arm}(')
    if arm in (3,5):
        text=once(text,'    float2 total[Pairs]{};', '''    float2 total[Pairs]{};
    uint64_t weight_cursor=(uint64_t(partition*Workers+worker)*16)*c.n+col;
    uint64_t scale_cursor=2*(uint64_t(partition*Workers+worker)*c.n+col);
    int64_t const weight_step=int64_t(split)*Workers*16*c.n;
    int64_t const scale_step=int64_t(split)*Workers*2*c.n;''')
        text=once(text,'g+=split*Workers) {',
                  'g+=split*Workers,weight_cursor+=weight_step,scale_cursor+=scale_step) {')
        text=once(text,'scales<P>(units+2*(uint64_t(g)*c.n+col),d);','scales<P>(units+scale_cursor,d);')
        old='low+(uint64_t(g)*16+segment*8+half*4+residue)*c.n+col'
        if text.count(old)!=2:raise ValueError('Q8 load-address seam differs')
        text=text.replace(old,'low+weight_cursor+uint64_t(segment*8+half*4+residue)*c.n')
    if arm in (4,5):
        start=text.index('        if constexpr(Hoist) {')
        end=text.index('        } else {\n            // Keep the original narrow load window',start)
        old=text[start:end]
        half='            for(int half=0;half<2;++half)\n'
        first=old.index(half)
        second=old.index(half,first+len(half))
        # Move the half loop outside both load and compute; the FP32 dot order
        # remains half -> segment -> slot -> residue. Only live B words change.
        loads=old[first+len(half):second]
        marker='            #pragma unroll\n'
        if not loads.endswith(marker):raise ValueError('Q8 second half-loop pragma differs')
        loads=loads[:-len(marker)]
        math=old[second+len(half):]
        window=('        if constexpr(Hoist) {\n'
                '            #pragma unroll\n'
                '            for(int half=0;half<2;++half) {\n'
                '                uint32_t words[2][4][Pairs];\n'+loads+math+
                '            }\n')
        window=window.replace('words[half][segment][residue]','words[segment][residue]')
        text=text[:start]+window+text[end:]
    return text


def symbol_name(config,arm):
    return f'q8_address_{config}_{arm}'


def geometry(w,config,arm):
    tile=config['columns']*config['values']
    return ([w.m*(w.n//tile),1,1] if arm==0 else [w.n//tile,w.m,1]),[config['warps']*32,1,1]


def source(q,compute,configs):
    if q!=8 or compute not in ('f16','bf16'):raise ValueError('Q8 dense only')
    c=int(compute=='bf16')
    text='''#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "dev/gemv_simt/general_locate.hpp"
namespace quactlize::execution::simt {
namespace q8_vector {
'''+ '\n'.join(body(i) for i in range(1,len(ARMS)))+'\n}\n'
    for i,f in enumerate(configs):
        if f['variant']!=5 or f['split']!=1 or not f.get('hoist'):raise ValueError('retain hoisted S1 controls')
        for arm in range(len(ARMS)):
            method='kernel_body' if arm==0 else f'address_body_{arm}'
            text+=f'__global__ void {symbol_name(i,arm)}(qkg_call_v1 c) {{ q8_vector::{method}<1,{c},1,{f["columns"]},{f["warps"]},{f["values"]},true>(c,1); }}\n'
    text+='''}
extern "C" int general_reader_run(qkg_simt_call_v2 const* d,quactlize_ppu_placed_arrangement_v2 const* a,int config,int arm) {
    using namespace quactlize::execution;
'''
    text+=f'    if(!d || !a || d->call.qtype!=8 || d->call.mode!=QKG_DENSE || d->call.input_type!=QKG_F32 || d->compute_type!={c} || arm<0 || arm>={len(ARMS)})return QKG_INVALID;\n'
    text+='    qkg_simt_config_v1 cfg{};\n    switch(config) {\n'
    for i,f in enumerate(configs):
        text+=f'    case {i}:cfg={{1,sizeof(cfg),5,{f["columns"]},{f["warps"]},{f["values"]},1}};break;\n'
    text+='''    default:return QKG_INVALID;
    }
    qkg_sizes_v1 sizes{};int rc=simt::query_v2(*d,cfg,a,sizes);if(rc)return rc;
    rc=simt::buffers_v2(*d,sizes);if(rc)return rc;
    auto const& call=d->call;auto stream=static_cast<hggcStream_t>(call.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    switch(config) {
'''
    for i,f in enumerate(configs):
        text+=f'    case {i}:switch(arm) {{\n'
        for arm in range(len(ARMS)):
            tile=f['columns']*f['values']
            grid=f'call.rows*(call.n/{tile})' if arm==0 else f'dim3(call.n/{tile},call.rows)'
            text+=f'    case {arm}:simt::{symbol_name(i,arm)}<<<{grid},{f["warps"]*32},0,stream>>>(call);break;\n'
        text+='    }break;\n'
    return text+'    }\n    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;\n}\n'
