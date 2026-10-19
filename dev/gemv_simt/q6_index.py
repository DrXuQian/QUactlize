"""A2 versus A3: canonical Q6 bytes, unsigned validated reader coordinates."""
from dev.gemv_simt.q6_sparse import ROOT, sparse_body, source as a2_source
from dev.gemv_simt.model_followup import once

ARMS=('sparse-metadata','unsigned-index')
KERNEL_NAMES=('::q6_sparse_fields(', '::q6_unsigned_fields(')


def index_body(text):
    body=sparse_body(text)
    body=once(body,'void q6_sparse_fields_body(', 'void q6_unsigned_fields_body(')
    body=once(body,'    using F=Format<Q>;', '    using F=Q6UnsignedFormat<Q>;')
    body=once(body,'using Index=std::conditional_t<(Changes&2)!=0,unsigned,int>;', 'using Index=unsigned;')
    if body.count('int k=g*F::group+')!=3:raise ValueError('local K coordinate seam changed')
    body=body.replace('int k=g*F::group+', 'unsigned k=g*F::group+')
    return once(body,'float2 code=codes<Q>(', 'float2 code=q6_unsigned_codes<Q>(')


def codes_body(text):
    start=text.index('template<int Q>\n__device__ __forceinline__ float2 codes(')
    end=text.index('\n} // namespace',start)
    body=text[start:end]
    body=once(body,'float2 codes(', 'float2 q6_unsigned_codes(')
    body=once(body,'int col,int k)', 'unsigned col,unsigned k)')
    body=once(body,'using F=Format<Q>;', 'static_assert(Q==14);\n    using F=Q6UnsignedFormat<Q>;')
    return once(body,'int slot;', 'unsigned slot;')


def source(compute):
    original=a2_source(compute)
    split=original.index('extern "C" int q6_metadata_run(')
    prefix=original[:split]
    c=int(compute=='bf16')
    prefix+='''#include "dev/gemv_simt/q6_unsigned_map.hpp"
namespace quactlize::execution::simt {
template<int Q> struct Q6UnsignedFormat : Format<Q> {
    static_assert(Q==14);
    using Low=q6_unsigned::Plane<4>;
    using High=q6_unsigned::Plane<2>;
    __host__ __device__ static constexpr uint64_t unit_offset(unsigned n,unsigned col,unsigned g) {
        return q6_unsigned::unit_offset(n,col,g);
    }
};
'''
    prefix+=codes_body((ROOT/'quactlize/execution/simt_format.cuh').read_text())
    prefix+=index_body((ROOT/'quactlize/execution/simt_kernel.cuh').read_text())
    prefix+=f'''\n__global__ void q6_unsigned_fields(qkg_call_v1 call,int split) {{
    q6_unsigned_fields_body<14,1,3,4,8,8,{c},0>(call,split);
}}
}}
'''
    export=original[split:]
    export=once(export,'if(arm==0) simt::q6_register_fields', 'if(arm==0) simt::q6_sparse_fields')
    export=once(export,'else simt::q6_sparse_fields', 'else simt::q6_unsigned_fields')
    return prefix+export
