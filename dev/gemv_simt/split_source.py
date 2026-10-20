"""Exactly the prior common body, extended to real Split-K workspace ownership."""
from dev.gemv_simt.general_reader import source as general_source, reuse_body
from dev.gemv_simt.general_plan import configs
from dev.gemv_simt.model_followup import once
from dev.gemv_simt.split_plan import QTYPES


def body():
    text=once(reuse_body(),'void general_reuse_body(', 'void split_reuse_body(')
    return once(text,'Index tile=blockIdx.x,partition=0,row=blockIdx.y;',
                      'Index tile=blockIdx.x,partition=blockIdx.z,row=blockIdx.y;')


def source(q,compute):
    if q not in QTYPES or compute not in ('f16','bf16'):
        raise ValueError('unsupported Split-K experiment format/precision')
    # Reuse the existing format/map/metadata implementations. Do not retain
    # the old experimental wrappers or mistake the A3 Q6 control for shipping.
    old=general_source(q,compute,configs(q))
    prefix=old[:old.index('\n__global__ void general_control_0(')]
    prefix=once(prefix,reuse_body(),body())
    prefix='#include "quactlize/execution/measured_decode.cuh"\n'+prefix
    c=int(compute=='bf16')
    for i,f in enumerate(configs(q)):
        ch=3 if q==13 and compute=='bf16' and (f['columns'],f['warps'],f['values'])==(4,2,8) else 0
        for mode in range(3):
            prefix+=f'''\n__global__ void split_candidate_{i}_{mode}(qkg_call_v1 call,int split) {{
    split_reuse_body<{mode},{q},1,3,{f['columns']},{f['warps']},{f['values']},{c},{ch}>(call,split);
}}
'''
    prefix+='}\n'
    prefix+='''extern "C" int split_reader_run(qkg_simt_call_v2 const* d,
    quactlize_ppu_placed_arrangement_v2 const* a,int config,int split,int arm,int stage) {
    using namespace quactlize::execution;
'''
    prefix+=f'''    if(!d || !a || d->call.qtype!={q} || d->call.input_type!=QKG_F32 ||
       d->compute_type!={c} || arm<0 || arm>1 || stage<0 || stage>2) return QKG_INVALID;
    qkg_simt_config_v1 cfg{{}};
    switch(config) {{
'''
    for i,f in enumerate(configs(q)):
        prefix+=f'    case {i}:cfg={{1,sizeof(cfg),3,{f["columns"]},{f["warps"]},{f["values"]},split}};break;\n'
    prefix+='''    default:return QKG_INVALID;
    }
    qkg_sizes_v1 sizes{};
    int rc=simt::query_v2(*d,cfg,a,sizes);if(rc)return rc;
    rc=simt::buffers_v2(*d,sizes);if(rc)return rc;
    auto const& call=d->call;auto stream=static_cast<hggcStream_t>(call.stream);
    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;
'''
    prefix+=f'    if(stage==2) {{ simt::launch_reduction<{q}>(call,split,stream);return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME; }}\n'
    # stage1 is candidate producer only, used only for partial/ACU proof.
    prefix+='    if(!arm && stage!=0)return QKG_INVALID;\n'
    prefix+=f'    if(!arm && simt::measured_decode(*d,cfg)!=simt::MeasuredDecode::None) return simt::measured_decode_launch<{q}>(*d,cfg);\n'
    prefix+='    switch(config) {\n'
    for i,f in enumerate(configs(q)):
        prefix+=f'    case {i}:\n        if(!arm)return simt::launch_v2<{q},3,{f["columns"]},{f["warps"]},{f["values"]}>(*d,split);\n'
        prefix+='        switch(call.mode) {\n'
        for mode in range(3):
            prefix+=f'        case {mode}:simt::split_candidate_{i}_{mode}<<<dim3(call.n/{f["columns"]*f["values"]},call.rows,split),{f["warps"]*32},0,stream>>>(call,split);break;\n'
        prefix+='        default:return QKG_INVALID;\n        }break;\n'
    prefix+='    }\n    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;\n'
    prefix+=f'    if(stage==0)simt::launch_reduction<{q}>(call,split,stream);\n'
    return prefix+'    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;\n}\n'
