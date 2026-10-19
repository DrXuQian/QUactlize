"""A1 fixed registers versus A2 sparse metadata: one unchanged S1 geometry."""
from dev.gemv_simt.q6_metadata import ROOT, candidate_body, source as a1_source
from dev.gemv_simt.model_followup import once

ARMS=('register-fields','sparse-metadata')
KERNEL_NAMES=('::q6_register_fields(', '::q6_sparse_fields(')


def sparse_body(text):
    body=candidate_body(text)
    body=once(body,'void q6_register_fields_body(', 'void q6_sparse_fields_body(')
    body=once(body,'    using F=Format<Q>;',
              '    static_assert(Q==14 && Variant==3 && Columns==4 && Warps==8 && P==8);\n    using F=Format<Q>;')
    body=once(body,'Meta<F::words> cooperative{};', 'q6_sparse::Packet cooperative{};')
    body=once(body,'cooperative=load_meta<Q>(units,call.n,tile*TileN+lane,g);',
              'cooperative=q6_sparse::load(units+F::unit_offset(call.n,tile*TileN+lane,g),unsigned(g));')
    body=once(body,'q6_metadata::affine(share_meta(cooperative,(lane%Columns)*P+p),g)',
              'q6_sparse::affine(q6_sparse::share(cooperative,(lane%Columns)*P+p,unsigned(g)),unsigned(g))')
    return body


def source(compute):
    original=a1_source(compute)
    split=original.index('extern "C" int q6_metadata_run(')
    prefix=original[:split]
    c=int(compute=='bf16')
    prefix+='#include "dev/gemv_simt/q6_sparse.cuh"\nnamespace quactlize::execution::simt {\n'
    prefix+=sparse_body((ROOT/'quactlize/execution/simt_kernel.cuh').read_text())
    prefix+=f'''\n__global__ void q6_sparse_fields(qkg_call_v1 call,int split) {{
    q6_sparse_fields_body<14,1,3,4,8,8,{c},0>(call,split);
}}
}}\n'''
    export=original[split:]
    export=once(export,f'if(arm==0) simt::register_reuse<14,1,3,4,8,8,{c},0>',
                'if(arm==0) simt::q6_register_fields')
    export=once(export,'else simt::q6_register_fields', 'else simt::q6_sparse_fields')
    return prefix+export
