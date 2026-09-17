"""Exact body reuse; only Q6's metadata load block is cloned and replaced."""

from pathlib import Path
from dev.gemv_model.plan import candidates

ROOT = Path(__file__).resolve().parents[2]


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source seam changed: " + old[:100])
    return text.replace(old, new, 1)


def direct_body():
    text = (ROOT / "quactlize/execution/simt_kernel.cuh").read_text()
    begin = text.index("template<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0,\n")
    end = text.index("\ntemplate<int Q,int Input,int Variant,int Columns,int Warps,int P,int Compute=0,int Changes=0>\n__global__", begin)
    body = text[begin:end]
    body = once(body, "void register_reuse_body", "void direct_metadata_body")
    begin = body.index("        Meta<F::words> cooperative{};")
    end = body.index("        uint4 packet{};", begin)
    body = body[:begin] + """        static_assert(Q==14 && F::Unit::kSbBytes==18 && F::Unit::kSbPerUnit==2);
        float2 metadata[P];
        #pragma unroll
        for (int p=0;p<P;++p) {
            auto unit=units+F::unit_offset(call.n,col+p,g)+((g/16)&1)*18;
            float d=__half2float(__ushort_as_half(*reinterpret_cast<uint16_t const*>(unit)));
            int scale=int(*reinterpret_cast<int8_t const*>(unit+2+(g&15)));
            metadata[p]=make_float2(d*float(scale),0.f);
        }
""" + body[end:]
    return body


def source(p):
    text = '''#include "quactlize/fusion/store.cuh"
#include "quactlize/execution/q4_s1_validation.hpp"
namespace quactlize::execution::q4_s1 {
CUTLASS_DEVICE Row locate(quactlize::fusion::DeviceCall const& c,int row) {
    int source=quactlize::fusion::input_row(c,row);
    if(source<0) return {-1,0,0};
    return locate(static_cast<qkg_call_v1 const&>(c),source);
}}
#include "quactlize/execution/simt_q8_vector.cuh"
#include "quactlize/execution/simt_validation.hpp"
#include "quactlize/fusion/validation.hpp"
namespace quactlize::execution::model_gemv {
using namespace quactlize::execution::simt;
// A half-width physical tile contains two complete G4/U4 pairs. Keep
// the same sequential inter-warp sum, not the ordinary XOR fold order.
struct Paired16Finish : quactlize::fusion::SimtFinish {
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(quactlize::fusion::DeviceCall const& c,
        int row,int tile,int partition,int split,float const* partial) {
        static_assert(TileN==16, "half-width paired tile");
        int tid=int(threadIdx.x);
        if(tid<8) {
            int ng=quactlize::fusion::PairedN4::gate(tid);
            float gate=0.f,up=0.f;
            #pragma unroll
            for(int w=0;w<Warps;++w) {
                gate+=partial[w*16+ng];up+=partial[w*16+ng+4];
            }
            quactlize::fusion::output(c,row,tile*8+tid,quactlize::fusion::activate(c,gate,up));
        }
    }
};
'''
    if p.q == 14:
        text += direct_body()
    ctype = "quactlize::fusion::DeviceCall" if p.paired else "qkg_call_v1"
    for index, c in enumerate(candidates(p)):
        text += f"__global__ void point_{p.q}_{p.physical_n}_{p.k}_arm_{index}({ctype} c) {{\n"
        if c.fixed:
            text += f"    c.n={p.physical_n};c.k={p.k};c.experts={p.experts};c.mode={p.mode};c.channels={p.channels};c.topk={8 if p.mode else 1};\n"
        finish = (",Paired16Finish" if c.tile_n == 16 else ",quactlize::fusion::SimtFinish") if p.paired else ""
        if p.q == 8:
            text += f"    q8_vector::kernel_body<1,{p.compute},{c.variant-4},{c.columns},{c.warps},{c.values},{str(c.hoist).lower()}{finish}>(c,{c.split});\n"
        else:
            body = "direct_metadata_body" if c.direct_meta else "register_reuse_body"
            text += f"    {body}<{p.q},1,{c.variant},{c.columns},{c.warps},{c.values},{p.compute},{c.changes}{finish}>(c,{c.split});\n"
        text += "}\n"
    text += "} // namespace\n"
    text += '''extern "C" int model_gemv_run(qkg_simt_call_v2 const* input,
    qkg_gate_up_call_v2 const* fusion,qkg_gate_up_layout_v1 const* layout,int arm) {
    using namespace quactlize::execution;
'''
    if p.paired:
        text += '''    if(!fusion || !layout || input) return QKG_INVALID;
    auto const& d=fusion->call.input;
'''
    else:
        text += "    if(!input || fusion || layout) return QKG_INVALID;\n    auto const& d=*input;\n"
    text += f'''    auto const& c=d.call;
    if(c.qtype!={p.q} || c.n!={p.n} || c.k!={p.k} || c.experts!={p.experts} ||
       c.mode!={p.mode} || c.channels!={p.channels} || c.topk!={8 if p.mode else 1} ||
       c.input_type!=QKG_F32 || d.compute_type!={p.compute}) return QKG_SHAPE;
    qkg_sizes_v1 sizes{{}};
    auto stream=static_cast<hggcStream_t>(c.stream);
    int rc=QKG_INVALID;
    switch(arm) {{
'''
    for index, c in enumerate(candidates(p)):
        text += f"    case {index}: {{\n"
        if p.paired:
            text += f'''        qkg_gate_up_config_v1 cfg{{1,sizeof(cfg),QKG_GATE_UP_SIMT,{c.split},0,{c.warps}}};
        if(fusion->call.output_type!=QKG_F32 || fusion->call.round_projection!={p.compute}) return QKG_INVALID;
        rc=quactlize::fusion::query(fusion->call,cfg,*layout,sizes);if(rc) return rc;
        rc=quactlize::fusion::buffers(fusion->call,sizes);if(rc) return rc;
        rc=quactlize::fusion::row_buffers(*fusion,sizes);if(rc) return rc;
        quactlize::fusion::DeviceCall call{{}};
        static_cast<qkg_call_v1&>(call)=c;
        call.n*=2;call.compute_type=d.compute_type;call.output_type=QKG_F32;
        call.round_projection=fusion->call.round_projection;
        call.input_rows=fusion->input_rows;call.status=fusion->status;
'''
        else:
            a = "q8_kpack2::arrangement()" if p.q == 8 else f"ppu_arrangements::kquant_kpack_transpose_v1({p.q})"
            text += f'''        qkg_simt_config_v1 cfg{{1,sizeof(cfg),{c.variant},{c.columns},{c.warps},{c.values},{c.split}}};
        auto arrangement={a};
        rc=simt::query_v2(d,cfg,&arrangement,sizes);if(rc) return rc;
        rc=simt::buffers_v2(d,sizes);if(rc) return rc;
        auto call=c;
'''
        text += f'''        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
        model_gemv::point_{p.q}_{p.physical_n}_{p.k}_arm_{index}<<<c.rows*{c.split}*({p.physical_n}/{c.tile_n}),{c.warps*32},0,stream>>>(call);
        if(hggcGetLastError()!=hggcSuccess) return QKG_RUNTIME;
'''
        if c.split > 1:
            # Match the shipping S8 fast reduction exactly. Other shapes keep
            # the shipping generic ordered reducer, not a producer-only score.
            if p.name == "q8-ssm-out" and c.split == 8 and c.columns == 8 and c.warps == 4 and c.values == 4:
                text += '''        if(c.rows==1 && !((uintptr_t(c.output)|uintptr_t(c.workspace))&7))
            quactlize::decode::reduce_decode<8><<<(c.n+63)/64,32,0,stream>>>(static_cast<float const*>(c.workspace),c.output,c.n);
        else
'''
            text += f"        simt::register_reuse_reduce<{p.q}><<<(int64_t(c.rows)*c.n+127)/128,128,0,stream>>>(c,{c.split});\n"
        text += "        return hggcGetLastError()==hggcSuccess ? QKG_OK : QKG_RUNTIME;\n    }\n"
    return text + "    default:return QKG_INVALID;\n    }\n}\n"
