"""Bounded same-math SIMT experiments; no shipping selector edits.

Generate from the incumbent body with fail-closed seams so dot/reduction code
is not independently reimplemented. Both bodies are compiled into one image.
Q8 keeps its current vector reader, including its optional hoisted loads.
"""
from pathlib import Path
from dev.gemv_simt.model_followup import once

ROOT = Path(__file__).resolve().parents[2]
QTYPES = (8, 10, 11, 12, 13, 14)
ARMS = ('matched-control', 'unsigned-packet-2d')


def reuse_body():
    text=(ROOT/'quactlize/execution/simt_kernel.cuh').read_text()
    start=text.index('template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,')
    end=text.index('\ntemplate<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>\n__global__',start)
    body=text[start:end]
    body=once(body,'template<int Q,int Input,','template<int Mode,int Q,int Input,')
    body=once(body,'void register_reuse_body(', 'void general_reuse_body(')
    body=once(body,'using F=Format<Q>;', 'using F=GeneralFormat<Q>;\n    static_assert(Columns==4 || Columns==8);')
    body=once(body,'using Index=std::conditional_t<(Changes&2)!=0,unsigned,int>;', 'using Index=unsigned;')
    body=once(body,'Index tile=blockIdx.x%(call.n/TileN),outer=blockIdx.x/(call.n/TileN);\n    Index partition=outer%split,row=outer/split;',
              'Index tile=blockIdx.x,partition=0,row=blockIdx.y;')
    body=once(body,'auto r=q4_s1::locate(call,row);', 'auto r=general_locate<Mode>(call,row);')
    if body.count('int k=g*F::group+')!=3: raise ValueError('general reader local K seam changed')
    body=body.replace('int k=g*F::group+', 'unsigned k=g*F::group+')
    body=once(body,'Meta<F::words> cooperative{};', 'general_metadata::Packet cooperative{};')
    body=once(body,'load_meta<Q>(units,call.n,tile*TileN+lane,g)',
              'general_metadata::load<Q>(units+F::unit_offset(call.n,tile*TileN+lane,g),g)')
    body=once(body,'affine_selected<Q,Changes>(share_meta(cooperative,(lane%Columns)*P+p),g)',
              'general_metadata::affine<Q>(general_metadata::share(cooperative,(lane%Columns)*P+p),g)')
    body=once(body,'affine_selected<Q,Changes>(load_meta<Q>(units,call.n,col+p,g),g)',
              'general_metadata::affine<Q>(general_metadata::load<Q>(units+F::unit_offset(call.n,col+p,g),g),g)')
    return once(body,'float2 code=codes<Q>(', 'float2 code=general_codes<Q>(')


def codes_body():
    text=(ROOT/'quactlize/execution/simt_format.cuh').read_text()
    start=text.index('template<int Q>\n__device__ __forceinline__ float2 codes(')
    body=text[start:text.index('\n} // namespace',start)]
    body=once(body,'float2 codes(', 'float2 general_codes(')
    body=once(body,'int col,int k)', 'unsigned col,unsigned k)')
    body=once(body,'using F=Format<Q>;', 'using F=GeneralFormat<Q>;')
    return once(body,'int slot;', 'unsigned slot;')


def vector_body():
    text=(ROOT/'quactlize/execution/simt_q8_vector.cuh').read_text()
    start=text.index('template<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false,')
    end=text.index('\ntemplate<int Input,int Compute,int Variant,int Columns,int Warps,int P,bool Hoist=false>\n__global__',start)
    body=text[start:end]
    body=once(body,'template<int Input,', 'template<int Mode,int Input,')
    body=once(body,'void kernel_body(', 'void general_vector_body(')
    body=once(body,'int tid=threadIdx.x,lane=tid%32,worker=tid/Columns;', 'unsigned tid=threadIdx.x,lane=tid%32,worker=tid/Columns;')
    body=once(body,'int tile=blockIdx.x%(c.n/TileN),outer=blockIdx.x/(c.n/TileN);\n    int partition=outer%split,row=outer/split;',
              'unsigned tile=blockIdx.x,partition=0,row=blockIdx.y;')
    body=once(body,'auto r=q4_s1::locate(c,row);', 'auto r=general_locate<Mode>(c,row);')
    body=once(body,'int col=tile*TileN+(tid%Columns)*P;', 'unsigned col=tile*TileN+(tid%Columns)*P;')
    return once(body,'for(int g=partition*Workers+worker;', 'for(unsigned g=partition*Workers+worker;')


def source(q,compute,configs):
    if q not in QTYPES or compute not in ('f16','bf16'): raise ValueError('unsupported format/compute')
    c=int(compute=='bf16')
    src='''#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "dev/gemv_simt/general_map.hpp"
#include "dev/gemv_simt/general_metadata.cuh"
#include "dev/gemv_simt/q6_metadata.cuh"
#include "dev/gemv_simt/q6_unsigned_map.hpp"
#include "dev/gemv_simt/general_locate.hpp"
namespace quactlize::execution::simt {
template<int Q> struct GeneralFormat : Format<Q> {
    using F=Format<Q>;
    using Low=general_map::Plane<F::low_bits>;
    using High=std::conditional_t<Q==13,general_map::Q5High,general_map::Plane<(F::high_bits?F::high_bits:1)>>;
    __host__ __device__ static uint64_t unit_offset(unsigned n,unsigned col,unsigned g) {
        using U=typename F::Unit;
        return (uint64_t(g/(U::kGroups*U::kSbPerUnit))*n+col)*U::kUnitTotal;
    }
};
'''+codes_body()+reuse_body()+'\nnamespace q8_vector {\n'+vector_body()+'\n}\n'
    if q==14:
        # Keep the actual latest experimental controls, not the pre-A1 generic
        # Q6 path. W8 is A3; W2 retains the marginal pipeline winner too.
        from dev.gemv_simt.q6_index import codes_body as q6_codes
        from dev.gemv_simt.q6_schedule import body as q6_body
        src+='''\ntemplate<int Q> struct Q6UnsignedFormat : Format<Q> {
    static_assert(Q==14);
    using Low=quactlize::execution::q6_unsigned::Plane<4>;
    using High=quactlize::execution::q6_unsigned::Plane<2>;
    __host__ __device__ static constexpr uint64_t unit_offset(unsigned n,unsigned col,unsigned g) {
        return quactlize::execution::q6_unsigned::unit_offset(n,col,g);
    }
};
'''
        src+=q6_codes((ROOT/'quactlize/execution/simt_format.cuh').read_text())+q6_body()+q6_body(pipeline=True)
    # The baseline uses the existing arithmetic body with compile-time S1,
    # but fully dynamic N/K. Geometry/hoist/Changes match each candidate.
    for i,f in enumerate(configs):
        v,col,w,p=f['variant'],f['columns'],f['warps'],f['values']
        if f.get('split',1)!=1: raise ValueError('this experiment is S1 only, no omitted reducer')
        h=str(bool(f.get('hoist',False))).lower()
        ch=3 if q==13 and compute=='bf16' and (v,col,w,p)==(3,4,2,8) else f.get('changes',0)
        call=f'q8_vector::kernel_body<1,{c},{v-4},{col},{w},{p},{h}>(call,1);' if q==8 else f'register_reuse_body<{q},1,{v},{col},{w},{p},{c},{ch}>(call,1);'
        if q==14 and col==4 and p==8:
            name='q6_pipeline_body' if w==2 else 'q6_warp_body'
            call=f'{name}<14,1,{v},4,{w},8,{c},0>(call,split);'
        signature='qkg_call_v1 call,int split' if q==14 and col==4 and p==8 else 'qkg_call_v1 call'
        src+=f'\n__global__ void general_control_{i}({signature}) {{ {call} }}\n'
        for mode in range(3):
            call=f'q8_vector::general_vector_body<{mode},1,{c},{v-4},{col},{w},{p},{h}>(call,1);' if q==8 else f'general_reuse_body<{mode},{q},1,{v},{col},{w},{p},{c},{ch}>(call,1);'
            src+=f'__global__ void general_candidate_{i}_{mode}(qkg_call_v1 call) {{ {call} }}\n'
    src+='}\nextern "C" int general_reader_run(qkg_simt_call_v2 const* d, quactlize_ppu_placed_arrangement_v2 const* a,int config,int arm) {\n    using namespace quactlize::execution;\n'
    src+=f'    if(!d || !a || d->call.qtype!={q} || d->call.input_type!=QKG_F32 || d->compute_type!={c} || arm<0 || arm>1) return QKG_INVALID;\n'
    src+='    qkg_simt_config_v1 cfg{};\n    switch(config) {\n'
    for i,f in enumerate(configs):
        src+=f'    case {i}: cfg={{1,sizeof(cfg),{f["variant"]},{f["columns"]},{f["warps"]},{f["values"]},1}};break;\n'
    src+='    default:return QKG_INVALID;\n    }\n    qkg_sizes_v1 sizes{};\n    int rc=simt::query_v2(*d,cfg,a,sizes);if(rc)return rc;\n    rc=simt::buffers_v2(*d,sizes);if(rc)return rc;\n    auto const& call=d->call;\n    auto stream=static_cast<hggcStream_t>(call.stream);\n    if(hggcGetLastError()!=hggcSuccess)return QKG_RUNTIME;\n    switch(config) {\n'
    for i,f in enumerate(configs):
        tile=f['columns']*f['values'];block=f['warps']*32
        args='call,1' if q==14 and f['columns']==4 and f['values']==8 else 'call'
        src+=f'    case {i}:\n        if(!arm) simt::general_control_{i}<<<call.rows*(call.n/{tile}),{block},0,stream>>>({args});\n        else switch(call.mode) {{\n'
        for mode in range(3):
            src+=f'        case {mode}:simt::general_candidate_{i}_{mode}<<<dim3(call.n/{tile},call.rows),{block},0,stream>>>(call);break;\n'
        src+='        default:return QKG_INVALID;\n        }\n        break;\n'
    return src+'    }\n    return hggcGetLastError()==hggcSuccess?QKG_OK:QKG_RUNTIME;\n}\n'
