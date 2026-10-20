"""Causal Q8 dense variants, canonical bytes/dot/reduction unchanged."""
from dev.gemv_simt.general_reader import ROOT
from dev.gemv_simt.model_followup import once

ARMS = (
    dict(name='control', mode=False, grid=False, unsigned=False, signed_loop=False),
    dict(name='dense_only', mode=True, grid=False, unsigned=False, signed_loop=False),
    dict(name='grid_only', mode=False, grid=True, unsigned=False, signed_loop=False),
    dict(name='unsigned_only', mode=False, grid=False, unsigned=True, signed_loop=False),
    dict(name='dense_grid_signed', mode=True, grid=True, unsigned=False, signed_loop=False),
    dict(name='previous_combined', mode=True, grid=True, unsigned=True, signed_loop=False),
    dict(name='combined_signed_loop', mode=True, grid=True, unsigned=True, signed_loop=True),
)


def original_body():
    text=(ROOT/'quactlize/execution/simt_q8_vector.cuh').read_text()
    begin=text.index('template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false,')
    end=text.index('\ntemplate<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false>\n__global__',begin)
    return text[begin:end]


def body(arm):
    if arm not in range(1,len(ARMS)):raise ValueError('control must use the untouched body')
    spec=ARMS[arm];text=original_body()
    text=once(text,'void kernel_body(',f'void dense_ablation_{arm}(')
    if spec['unsigned']:
        text=once(text,'int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;',
                       'unsigned tid=threadIdx.x,lane=tid%32,worker=tid/Columns;')
        text=once(text,'int col=tile*TileN+(tid%Columns)*P;',
                       'unsigned col=tile*TileN+(tid%Columns)*P;')
        if not spec['signed_loop']:
            text=once(text,'for(int g=partition*Workers+worker;', 'for(unsigned g=partition*Workers+worker;')
    index='unsigned' if spec['unsigned'] else 'int'
    old='int tile=blockIdx.x%(c.n/TileN),outer=blockIdx.x/(c.n/TileN);\n    int partition=outer%split,row=outer/split;'
    if spec['grid']:
        text=once(text,old,f'{index} tile=blockIdx.x,partition=0,row=blockIdx.y;')
    elif spec['unsigned']:
        text=once(text,old,'unsigned tile=blockIdx.x%(c.n/TileN),outer=blockIdx.x/(c.n/TileN);\n    unsigned partition=outer%split,row=outer/split;')
    if spec['mode']:
        text=once(text,'auto r=q4_s1::locate(c,row);','auto r=general_locate<QKG_DENSE>(c,row);')
    return text


def symbol_name(config,arm):
    if arm==0:return f'general_control_{config}'
    if arm==5:return f'general_candidate_{config}_0'
    return f'q8_dense_{config}_{arm}'


def geometry(w,config,arm):
    tile=config['columns']*config['values']
    grid=[w.n//tile,w.m,1] if ARMS[arm]['grid'] else [w.m*(w.n//tile),1,1]
    return grid,[config['warps']*32,1,1]


def source(q,compute,configs):
    if q!=8 or compute not in ('f16','bf16'):raise ValueError('Q8 dense only')
    c=int(compute=='bf16')
    text='''#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "dev/gemv_simt/general_locate.hpp"
namespace quactlize::execution::simt {
namespace q8_vector {
'''
    text+='\n'.join(body(arm) for arm in range(1,len(ARMS)))+'\n}\n'
    for i,cfg in enumerate(configs):
        if cfg['variant']!=5 or cfg['split']!=1 or not cfg.get('hoist'):raise ValueError('retain vector hoisted S1 controls')
        for arm in range(len(ARMS)):
            method='kernel_body' if arm==0 else f'dense_ablation_{arm}'
            text+=f'__global__ void {symbol_name(i,arm)}(qkg_call_v1 call) {{ q8_vector::{method}<1,{c},1,{cfg["columns"]},{cfg["warps"]},{cfg["values"]},true>(call,1); }}\n'
    text+='''}
extern "C" int general_reader_run(qkg_simt_call_v2 const* d,quactlize_ppu_placed_arrangement_v2 const* a,int config,int arm) {
    using namespace quactlize::execution;
'''
    text+=f'    if(!d || !a || d->call.qtype!=8 || d->call.mode!=QKG_DENSE || d->call.input_type!=QKG_F32 || d->compute_type!={c} || arm<0 || arm>={len(ARMS)})return QKG_INVALID;\n'
    text+='    qkg_simt_config_v1 cfg{};\n    switch(config) {\n'
    for i,cfg in enumerate(configs):
        text+=f'    case {i}:cfg={{1,sizeof(cfg),5,{cfg["columns"]},{cfg["warps"]},{cfg["values"]},1}};break;\n'
    text+='''    default:return QKG_INVALID;
    }
    qkg_sizes_v1 sizes{};int rc=simt::query_v2(*d,cfg,a,sizes);if(rc)return rc;
    rc=simt::buffers_v2(*d,sizes);if(rc)return rc;
    auto const& call=d->call;auto stream=static_cast<hggcStream_t>(call.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
    switch(config) {
'''
    for i,cfg in enumerate(configs):
        tile=cfg['columns']*cfg['values'];block=cfg['warps']*32
        text+=f'    case {i}:switch(arm) {{\n'
        for arm in range(len(ARMS)):
            grid=f'dim3(call.n/{tile},call.rows)' if ARMS[arm]['grid'] else f'call.rows*(call.n/{tile})'
            text+=f'    case {arm}:simt::{symbol_name(i,arm)}<<<{grid},{block},0,stream>>>(call);break;\n'
        text+='    }break;\n'
    return text+'    }\n    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;\n}\n'
